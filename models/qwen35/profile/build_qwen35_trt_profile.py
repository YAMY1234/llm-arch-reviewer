#!/usr/bin/env python3
"""Build immutable TRT-LLM AgentX profiles from worker-local Nsys SQLite files."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sqlite3
import statistics
import sys
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.common.timeline_artifact import (
    build_timeline_artifact,
    write_timeline_artifact,
)
from models.qwen35.profile.build_qwen35_sglang_decode_profile import (
    _metrics_for_rank,
    sha256_file,
)
from models.qwen35.profile.qwen35_graph_mapping import attribution_active_union_ratio
from models.qwen35.profile.qwen35_a2a_contract import (
    validate_comparison_workload,
)
from models.qwen35.profile.qwen35_nsys_mapping import (
    load_nsys_steps,
    map_decode_step,
    map_prefill_step,
    read_nsys_export_metadata,
)
from models.qwen35.profile.qwen35_timeline import QWEN35_TIMELINE_TARGETS
from models.qwen35.profile.qwen35_torch_mapping import load_trt_torch_steps


REPORT_RE = re.compile(
    r"(?P<worker>.+)-(?P<phase>prefill|decode)-rank(?P<rank>[0-3])(?:\.\d+)?\.sqlite$"
)
TORCH_TRACE_RE = re.compile(
    r".+(?:\.trace)?-host-(?P<worker>[^/]+)-rank-(?P<rank>[0-3])"
    r"(?:\.trace)?\.json(?:\.gz)?$"
)
TRT_EXACT_START_RE = re.compile(
    r"\[RANK (?P<rank>[0-3])\].*Rank-local BS32-triggered raw profiling "
    r"started at iteration (?P<iteration>\d+): local_batch=(?P<batch>\d+), "
    r"capture_raw_decode_batches=(?P<count>\d+)"
)
TRT_EXACT_STOP_RE = re.compile(
    r"\[RANK (?P<rank>[0-3])\].*Rank-local BS32-triggered raw profiling "
    r"stopped at iteration (?P<iteration>\d+): local_batch=(?P<batch>\d+), "
    r"captured_raw_decode_batches=(?P<count>\d+)"
)
TRT_COMMIT = "1cef02e901be43081b1ba6d4981e94ed3bd9c1e8"
MODEL_REVISION = "8f590eae8f10bf55d9a46f79ea0280bde435c9f8"
MODEL_CONFIG_SHA256 = "9408a9e559cc2f05f0b357738213666353e6651160ce8ff477b1c26982bc4f63"
CONTAINER_SHA256 = "1cb820b92bd7ab56ab69457500adf3b7f2928bfefe7f2920951fe7286552dcf7"
PY_EXECUTOR_BASE_SHA256 = (
    "69b566f2d30e1d1465d4ef85af1913ef3cb8d0f4e36d78bf92989837e6f4aa9a"
)
PY_EXECUTOR_PROFILE_OVERLAY_SHA256 = (
    "a0eb9784bc85c2d6e736224c5bde405649947f32b968f5d8d6c705f6cfc0f348"
)
PY_EXECUTOR_TORCH_OVERLAY_SHA256 = (
    "e8b4d2e03ecbe9a2f033e0f9afbb7c16bd4222b98d01a5ba90fabac7ae57471d"
)
DYNAMO_HANDLER_BASE_SHA256 = (
    "e44f1028ae686dd60e6ded8807735e678504898cccac0cf2b70749967714dcbc"
)
DYNAMO_EXACT_OUTPUT_OVERLAY_SHA256 = (
    "e0a6eb5eae16820c439533f69bed4ea63abffd3ad6bdcc228d47a683588e938e"
)
DYNAMO_WHEEL_BASE_SHA256 = (
    "43d2ff07ea8c60efea41c2f9085ebc846479639e63dfdb276ec1dbc93b144abf"
)
DYNAMO_EXACT_OUTPUT_WHEEL_SHA256 = (
    "2e1d883bba8dbd6aea6ed9ed264c593905168d4b569786a1ad0285690c77f536"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prefill", "decode"), required=True)
    parser.add_argument("--sqlites", type=Path, nargs="*", default=[])
    parser.add_argument("--torch-traces", type=Path, nargs="*", default=[])
    parser.add_argument(
        "--trace-format", choices=("nsys", "torch"), default="nsys"
    )
    parser.add_argument("--nsys-reports", type=Path, nargs="*", default=[])
    parser.add_argument("--worker-logs", type=Path, nargs="*", default=[])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--workload-result", type=Path)
    parser.add_argument("--job-id", type=int, default=532540)
    parser.add_argument("--decode-batch", type=int, default=32)
    parser.add_argument("--output-profile", type=Path, required=True)
    parser.add_argument("--output-timeline", type=Path, required=True)
    parser.add_argument("--output-analysis", type=Path, required=True)
    parser.add_argument("--output-mapping", type=Path, required=True)
    return parser.parse_args()


def _report_identity(path: Path, phase: str) -> tuple[str, int]:
    match = REPORT_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"unrecognized report filename {path.name}")
    if match.group("phase") != phase:
        raise ValueError(f"{path.name} is not a {phase} report")
    return match.group("worker"), int(match.group("rank"))


def _torch_trace_identity(path: Path) -> tuple[str, int]:
    match = TORCH_TRACE_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"unrecognized host-safe Torch trace filename {path.name}")
    return match.group("worker"), int(match.group("rank"))


def _validate_process(path: Path) -> dict[str, Any]:
    connection = sqlite3.connect(path)
    try:
        processes = [
            {"pid": int(pid), "name": str(name)}
            for pid, name in connection.execute(
                "select distinct p.pid, p.name from CUPTI_ACTIVITY_KIND_KERNEL k "
                "join PROCESSES p on p.globalPid=k.globalPid order by p.pid"
            )
        ]
        nvtx_steps = connection.execute(
            "select count(*) from NVTX_EVENTS n left join StringIds s on s.id=n.textId "
            "where coalesce(n.text,s.value) like '[Executor] _forward_step %' and n.end is not null"
        ).fetchone()[0]
        kernels = connection.execute(
            "select count(*) from CUPTI_ACTIVITY_KIND_KERNEL"
        ).fetchone()[0]
    finally:
        connection.close()
    if not processes or not any("python" in item["name"].lower() for item in processes):
        raise ValueError(
            f"{path.name}: kernels are not attached to a Python worker process"
        )
    if not nvtx_steps or not kernels:
        raise ValueError(f"{path.name}: missing NVTX steps or CUDA kernels")
    return {
        "processes": processes,
        "nvtx_step_count": nvtx_steps,
        "kernel_count": kernels,
    }


def _validate_exact_worker_log(path: Path, *, expected_steps: int) -> dict[str, Any]:
    text = path.read_text(errors="replace")
    starts = list(TRT_EXACT_START_RE.finditer(text))
    stops = list(TRT_EXACT_STOP_RE.finditer(text))
    if len(starts) != 4 or len(stops) != 4:
        raise ValueError(
            f"{path}: expected four rank-local starts/stops, got {len(starts)}/{len(stops)}"
        )
    starts_by_rank = {int(match.group("rank")): match for match in starts}
    stops_by_rank = {int(match.group("rank")): match for match in stops}
    if set(starts_by_rank) != {0, 1, 2, 3} or set(stops_by_rank) != {0, 1, 2, 3}:
        raise ValueError(f"{path}: incomplete or duplicate rank-local capture markers")
    for match in starts:
        if (
            int(match.group("batch")) != 32
            or int(match.group("count")) != expected_steps
        ):
            raise ValueError(f"{path}: invalid exact start gate: {match.group(0)}")
    for match in stops:
        if int(match.group("count")) != expected_steps:
            raise ValueError(f"{path}: invalid exact stop gate: {match.group(0)}")
    for rank in range(4):
        start_iter = int(starts_by_rank[rank].group("iteration"))
        stop_iter = int(stops_by_rank[rank].group("iteration"))
        if stop_iter - start_iter != expected_steps:
            raise ValueError(
                f"{path}: rank {rank} expected {expected_steps} raw iterations, "
                f"got {start_iter}:{stop_iter}"
            )
    return {
        "file": path.name,
        "sha256": sha256_file(path),
        "rank_start_count": len(starts),
        "rank_stop_count": len(stops),
        "start_iterations_by_rank": {
            str(rank): int(starts_by_rank[rank].group("iteration")) for rank in range(4)
        },
        "stop_iterations_by_rank": {
            str(rank): int(stops_by_rank[rank].group("iteration")) for rank in range(4)
        },
        "captured_raw_decode_iterations": expected_steps,
    }


def _validate_nsys_reports(
    paths: list[Path], *, workers: list[str], phase: str
) -> list[dict[str, Any]]:
    expected = {(worker, rank) for worker in workers for rank in range(4)}
    matched: dict[tuple[str, int], Path] = {}
    for raw_path in paths:
        path = raw_path.resolve()
        identities = [
            (worker, rank)
            for worker, rank in expected
            if worker in path.name and phase in path.name and f"rank{rank}" in path.name
        ]
        if len(identities) != 1:
            raise ValueError(
                f"cannot identify worker/rank for raw NSYS report {path.name}"
            )
        identity = identities[0]
        if identity in matched:
            raise ValueError(f"duplicate raw NSYS report for {identity}")
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"raw NSYS report is missing or empty: {path}")
        matched[identity] = path
    if set(matched) != expected:
        raise ValueError(f"incomplete raw NSYS reports: {sorted(matched)}")
    return [
        {
            "worker": worker,
            "rank": rank,
            "file": path.name,
            "sha256": sha256_file(path),
        }
        for (worker, rank), path in sorted(matched.items())
    ]


def _aggregate_metrics(source_metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    nodes = sorted({node for metrics in source_metrics.values() for node in metrics})
    output = {}
    for node in nodes:
        candidates = [
            (source, metrics[node])
            for source, metrics in sorted(source_metrics.items())
            if node in metrics
        ]
        selected_source, selected = max(
            candidates, key=lambda item: float(item[1]["ms_per_iter"])
        )
        values = [float(cell["ms_per_iter"]) for _source, cell in candidates]
        output[node] = {
            **selected,
            "ms_per_iter": round(float(selected["ms_per_iter"]), 6),
            "aggregation": "maximum per worker/rank kernel residency",
            "source_worker_rank": selected_source,
            "worker_rank_range_ms": [round(min(values), 6), round(max(values), 6)],
        }
    return output


def select_balanced_rank_local_steps(
    rows: list[dict[str, Any]],
    *,
    sample_count: int,
    allowed_sources: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Select a deterministic, time-spread sample balanced across sources."""

    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        source = str(row["source"])
        if allowed_sources is not None and source not in allowed_sources:
            continue
        by_source[source].append(row)
    for source_rows in by_source.values():
        source_rows.sort(key=lambda row: int(row["capture_iteration"]))
    available = sum(len(source_rows) for source_rows in by_source.values())
    if available < sample_count:
        raise ValueError(
            f"TRT exact capture has only {available} rank-local samples; "
            f"need {sample_count}"
        )

    quotas: Counter[str] = Counter()
    sources = sorted(by_source)
    while sum(quotas.values()) < sample_count:
        progressed = False
        for source in sources:
            if quotas[source] >= len(by_source[source]):
                continue
            quotas[source] += 1
            progressed = True
            if sum(quotas.values()) == sample_count:
                break
        if not progressed:
            raise AssertionError("TRT sample quota allocation stalled")

    selected = []
    for source in sources:
        source_rows = by_source[source]
        quota = quotas[source]
        indices = [
            ((2 * index + 1) * len(source_rows)) // (2 * quota)
            for index in range(quota)
        ]
        selected.extend(source_rows[index] for index in indices)
    selected.sort(
        key=lambda row: (
            int(row["capture_iteration"]),
            str(row["source"]),
        )
    )
    for sample_index, row in enumerate(selected):
        row["selected_sample_index"] = sample_index
    return selected


def elect_worker_comparison_sources(
    raw_exact_counts: Counter[str],
    *,
    workers: list[str],
    minimum_per_source: int,
) -> set[str]:
    """Elect one exact-BS source per worker, preferring count then low rank."""

    selected: set[str] = set()
    for worker in workers:
        rank = max(
            range(4),
            key=lambda candidate: (
                raw_exact_counts[f"{worker}/rank{candidate}"],
                -candidate,
            ),
        )
        source = f"{worker}/rank{rank}"
        available = raw_exact_counts[source]
        if available < minimum_per_source:
            raise ValueError(
                f"TRT {worker} has only {available} exact BS32 steps on its "
                f"best rank-local source; need {minimum_per_source}"
            )
        selected.add(source)
    return selected


def build(args: argparse.Namespace):
    trace_format = str(getattr(args, "trace_format", "nsys"))
    if trace_format == "torch" and args.phase != "decode":
        raise ValueError("the strict same-profiler Torch path currently supports decode")
    trace_inputs = (
        list(getattr(args, "torch_traces", []))
        if trace_format == "torch"
        else list(args.sqlites)
    )
    if not trace_inputs:
        raise ValueError(f"TRT {trace_format} profile has no trace inputs")
    decode_batch = int(getattr(args, "decode_batch", 32))
    if decode_batch < 1 or decode_batch > 32:
        raise ValueError(f"TRT decode batch must be in 1..32, got {decode_batch}")
    expected_workers = 3 if args.phase == "prefill" else 2
    comparison_contract = None
    workload_evidence = None
    raw_capture_step_count = 2
    selected_sample_count = 2
    if args.phase == "decode":
        required_paths = {
            "config": args.config,
            "dataset": args.dataset,
            "dataset manifest": args.dataset_manifest,
            "workload result": args.workload_result,
        }
        missing = [label for label, path in required_paths.items() if path is None]
        if missing:
            raise ValueError(f"strict TRT decode profile lacks {', '.join(missing)}")
        comparison_contract, workload_evidence = validate_comparison_workload(
            engine="trtllm",
            config=args.config,
            dataset=args.dataset,
            dataset_manifest=args.dataset_manifest,
            workload_result=args.workload_result,
        )
        selected_sample_count = int(comparison_contract["selected_rank_local_samples"])
        config_data = yaml.safe_load(args.config.read_text())
        raw_capture_step_count = int(
            (
                (
                    (config_data.get("backend") or {}).get("decode_environment") or {}
                ).get("TLLM_PROFILE_EXACT_DECODE_BATCHES", -1)
            )
        )
        if raw_capture_step_count < selected_sample_count:
            raise ValueError(
                "TRT raw NSYS capture must be at least as wide as the selected "
                f"sample: raw={raw_capture_step_count}, selected={selected_sample_count}"
            )
    paths: dict[tuple[str, int], Path] = {}
    for raw_path in trace_inputs:
        path = raw_path.resolve()
        identity = (
            _torch_trace_identity(path)
            if trace_format == "torch"
            else _report_identity(path, args.phase)
        )
        if identity in paths:
            raise ValueError(f"duplicate report for {identity}: {path}")
        paths[identity] = path
    workers = sorted({worker for worker, _rank in paths})
    if len(workers) != expected_workers:
        raise ValueError(
            f"expected {expected_workers} {args.phase} workers, got {workers}"
        )
    for worker in workers:
        ranks = {rank for candidate, rank in paths if candidate == worker}
        if ranks != {0, 1, 2, 3}:
            raise ValueError(f"worker {worker} lacks four-rank coverage: {ranks}")
    raw_nsys_reports = []
    exact_worker_logs = []
    if args.phase == "decode":
        if len(args.worker_logs) != 2:
            raise ValueError(
                "strict TRT decode profile requires two worker logs"
            )
        if trace_format == "nsys":
            if len(args.nsys_reports) != 8:
                raise ValueError(
                    "strict TRT NSYS decode profile requires eight raw reports"
                )
            raw_nsys_reports = _validate_nsys_reports(
                args.nsys_reports, workers=workers, phase=args.phase
            )
        elif args.nsys_reports:
            raise ValueError("Torch profile must not carry NSYS reports")
        exact_worker_logs = [
            _validate_exact_worker_log(path, expected_steps=raw_capture_step_count)
            for path in args.worker_logs
        ]
    nsys_export_metadata = (
        {
            identity: read_nsys_export_metadata(path)
            for identity, path in sorted(paths.items())
        }
        if trace_format == "nsys"
        else {}
    )
    if nsys_export_metadata and len(
        {tuple(sorted(row.items())) for row in nsys_export_metadata.values()}
    ) != 1:
        raise ValueError(
            f"TRT-LLM reports use different Nsight exporters: {nsys_export_metadata}"
        )

    source_metrics: dict[str, dict[str, Any]] = {}
    validations: dict[str, list[dict[str, Any]]] = {}
    process_checks: dict[str, dict[str, Any]] = {}
    timing_by_step: dict[int, list[dict[str, Any]]] = {}
    all_mappings: list[dict[str, Any]] = []
    reference_source = f"{workers[0]}/rank3"
    reference_steps: list[dict[str, Any]] = []
    observed_steps: list[dict[str, Any]] = []
    owner_rank_positions: set[int] = set()
    shape_observations: list[dict[str, Any]] = []

    for (worker, rank), path in sorted(paths.items()):
        source = f"{worker}/rank{rank}"
        if trace_format == "torch":
            steps = load_trt_torch_steps(path, rank=rank)
            process_checks[source] = {
                "gpu_step_count": len(steps),
                "kernel_count": sum(len(step.kernels) for step in steps),
                "concrete_kernel_name_count": len(
                    {kernel.name for step in steps for kernel in step.kernels}
                ),
            }
        else:
            process_checks[source] = _validate_process(path)
            steps = load_nsys_steps(path, rank=rank)
        actual_steps = [step.step_id for step in steps]
        if len(actual_steps) != raw_capture_step_count or actual_steps != list(
            range(actual_steps[0], actual_steps[0] + raw_capture_step_count)
        ):
            raise ValueError(
                f"{source}: expected {raw_capture_step_count} contiguous steps, "
                f"got {actual_steps}"
            )
        source_mappings: list[dict[str, Any]] = []
        source_validations = []
        for iteration, step in enumerate(steps):
            if args.phase == "decode":
                mappings, validation = map_decode_step(step)
            else:
                mappings, validation = map_prefill_step(step)
                if validation["owner_compute"]:
                    owner_rank_positions.add(rank)
            for mapping in mappings:
                mapping["event_id"] = f"{worker}-{mapping['event_id']}"
                mapping["worker"] = worker
            source_mappings.extend(mappings)
            source_validations.append(validation)
            shape_observations.append(
                {
                    "worker": worker,
                    "rank": rank,
                    "step_id": step.step_id,
                    "context_reqs": validation["context_reqs"],
                    "context_tokens": validation["context_tokens"],
                    "generation_reqs": validation["generation_reqs"],
                    "owner_compute": validation.get("owner_compute"),
                }
            )
            timing_by_step.setdefault(iteration, []).append(
                {
                    "source": source,
                    "capture_iteration": iteration,
                    "context_reqs": validation["context_reqs"],
                    "context_tokens": validation["context_tokens"],
                    "generation_reqs": validation["generation_reqs"],
                    "cpu_wall_us": validation["cpu_wall_us"],
                    "gpu_span_us": validation["gpu_span_us"],
                    "gpu_busy_union_us": validation["gpu_busy_union_us"],
                    "gpu_residency_us": validation["gpu_residency_us"],
                }
            )
            observed_steps.append(
                {
                    "source": source,
                    "worker": worker,
                    "rank": rank,
                    "path": path,
                    "step_id": step.step_id,
                    "capture_iteration": iteration,
                    "context_reqs": validation["context_reqs"],
                    "context_tokens": validation["context_tokens"],
                    "generation_reqs": validation["generation_reqs"],
                    "timeline_step": {
                        "step_index": step.step_id,
                        "label": step.label,
                        "trace_start_us": min(
                            float(item["ts_us"]) for item in mappings
                        ),
                        "duration_us": validation["gpu_span_us"],
                        "events": mappings,
                    },
                }
            )
        source_metrics[source] = _metrics_for_rank(source_mappings, len(steps))
        validations[source] = source_validations
        all_mappings.extend(source_mappings)

    if args.phase == "prefill" and owner_rank_positions != {0, 1, 2, 3}:
        raise ValueError(
            f"prefill reports do not validate owner compute on all rank positions: {owner_rank_positions}"
        )

    if args.phase == "prefill":
        exact_8k = [
            row
            for row in observed_steps
            if row["context_reqs"] == 1 and row["context_tokens"] == 8192
        ]
        if not exact_8k:
            raise ValueError(
                "TRT prefill capture has no exact one-request/8192-token step"
            )
        exact_counts = Counter(str(row["source"]) for row in exact_8k)
        reference_source = min(
            exact_counts,
            key=lambda source: (-exact_counts[source], source),
        )
        reference_observations = [
            row for row in exact_8k if row["source"] == reference_source
        ]
        reference_steps = [row["timeline_step"] for row in reference_observations]
        exact_events_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
        exact_step_counts: Counter[str] = Counter()
        for row in exact_8k:
            source = str(row["source"])
            exact_events_by_source[source].extend(row["timeline_step"]["events"])
            exact_step_counts[source] += 1
        source_metrics = {
            source: _metrics_for_rank(events, exact_step_counts[source])
            for source, events in sorted(exact_events_by_source.items())
        }
        all_mappings = [
            event for row in exact_8k for event in row["timeline_step"]["events"]
        ]
        exact_sources_by_step: dict[int, set[str]] = defaultdict(set)
        for row in exact_8k:
            exact_sources_by_step[int(row["capture_iteration"])].add(str(row["source"]))
        timing_by_step = {
            step_id: [
                row
                for row in rows
                if row["source"] in exact_sources_by_step.get(step_id, set())
                and row["context_reqs"] == 1
                and row["context_tokens"] == 8192
            ]
            for step_id, rows in timing_by_step.items()
            if exact_sources_by_step.get(step_id)
        }
        reference_path = Path(reference_observations[0]["path"])
        reference_rank = int(reference_observations[0]["rank"])
        reference_worker = str(reference_observations[0]["worker"])
    else:
        raw_exact_decode = [
            row for row in observed_steps if row["generation_reqs"] == decode_batch
        ]
        if not raw_exact_decode:
            raise ValueError(
                f"TRT decode capture has no exact BS{decode_batch} generation step"
            )
        raw_exact_counts = Counter(str(row["source"]) for row in raw_exact_decode)
        expected_sources = {
            f"{worker}/rank{rank}" for worker in workers for rank in range(4)
        }
        minimum_per_comparison_source = selected_sample_count // len(workers)
        if set(raw_exact_counts) != expected_sources:
            raise ValueError(
                "TRT raw capture needs complete all-rank reports: "
                f"{dict(sorted(raw_exact_counts.items()))}"
            )
        comparison_sources = elect_worker_comparison_sources(
            raw_exact_counts,
            workers=workers,
            minimum_per_source=minimum_per_comparison_source,
        )
        exact_decode = select_balanced_rank_local_steps(
            raw_exact_decode,
            sample_count=selected_sample_count,
            allowed_sources=comparison_sources,
        )
        exact_counts = Counter(str(row["source"]) for row in exact_decode)
        reference_source = min(
            comparison_sources,
            key=lambda source: (-raw_exact_counts[source], source),
        )
        reference_observations = [
            row for row in exact_decode if row["source"] == reference_source
        ]
        reference_steps = [row["timeline_step"] for row in reference_observations]
        exact_events_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
        exact_step_counts: Counter[str] = Counter()
        for row in exact_decode:
            source = str(row["source"])
            for event in row["timeline_step"]["events"]:
                event["selected_sample_index"] = int(row["selected_sample_index"])
            exact_events_by_source[source].extend(row["timeline_step"]["events"])
            exact_step_counts[source] += 1
        source_metrics = {
            source: _metrics_for_rank(events, exact_step_counts[source])
            for source, events in sorted(exact_events_by_source.items())
        }
        all_mappings = [
            event for row in exact_decode for event in row["timeline_step"]["events"]
        ]
        selected_timing_by_sample: dict[int, list[dict[str, Any]]] = {}
        for row in exact_decode:
            capture_iteration = int(row["capture_iteration"])
            source = str(row["source"])
            candidates = [
                timing
                for timing in timing_by_step[capture_iteration]
                if timing["source"] == source
                and timing["generation_reqs"] == decode_batch
            ]
            if len(candidates) != 1:
                raise ValueError(
                    f"TRT selected sample lacks unique timing row: {source} "
                    f"iteration={capture_iteration}"
                )
            selected_timing_by_sample[int(row["selected_sample_index"])] = candidates
        timing_by_step = selected_timing_by_sample
        reference_path = Path(reference_observations[0]["path"])
        reference_rank = int(reference_observations[0]["rank"])
        reference_worker = str(reference_observations[0]["worker"])

    critical_steps = {}
    for step_id, rows in sorted(timing_by_step.items()):
        selected = max(rows, key=lambda item: item["gpu_span_us"])
        active_us = selected["gpu_busy_union_us"]
        residency_us = selected["gpu_residency_us"]
        elapsed_us = selected["gpu_span_us"]
        if active_us > elapsed_us + 1e-6 or active_us > residency_us + 1e-6:
            raise ValueError(
                f"TRT {args.phase} step {step_id}: impossible timing values"
            )
        critical_steps[str(step_id)] = {
            "source_worker_rank": selected["source"],
            "elapsed_wall_us": elapsed_us,
            "active_gpu_us": active_us,
            "gpu_residency_us": residency_us,
            "gpu_overlap_us": residency_us - active_us,
            "device_gap_idle_us": elapsed_us - active_us,
            "cpu_launch_wall_us": selected["cpu_wall_us"],
        }
    critical_elapsed_ms = [
        row["elapsed_wall_us"] / 1000.0 for row in critical_steps.values()
    ]
    critical_cpu_ms = [
        row["cpu_launch_wall_us"] / 1000.0 for row in critical_steps.values()
    ]
    timing_summary = {
        "semantics": (
            "each decode sample is one real rank-local BS32 CUDA Graph period; "
            f"the {selected_sample_count}-sample pool is balanced across "
            f"runtime-elected sources {', '.join(sorted(comparison_sources))}; "
            "CPU NVTX is asynchronous launch wall and rank residency is never summed"
            if args.phase == "decode"
            else "step elapsed is first-to-last GPU kernel span from one critical worker/rank; CPU NVTX is asynchronous launch wall; rank residency is never summed"
        ),
        "critical_steps": critical_steps,
        "critical_step_wall_ms": {
            "samples": critical_elapsed_ms,
            "mean": statistics.fmean(critical_elapsed_ms),
            "median": statistics.median(critical_elapsed_ms),
            "min": min(critical_elapsed_ms),
            "max": max(critical_elapsed_ms),
        },
        "critical_cpu_launch_wall_ms": {
            "samples": critical_cpu_ms,
            "mean": statistics.fmean(critical_cpu_ms),
            "median": statistics.median(critical_cpu_ms),
            "min": min(critical_cpu_ms),
            "max": max(critical_cpu_ms),
        },
    }

    total_us = sum(float(row["dur_us"]) for row in all_mappings)
    status_us: Counter[str] = Counter()
    for row in all_mappings:
        status_us[str(row["mapping_status"])] += float(row["dur_us"])
    attributed_residency_ratio = (status_us["mapped"] + status_us["fusion"]) / total_us
    attributed_active_ratio = attribution_active_union_ratio(all_mappings)
    strict_signature_us = sum(
        float(event["dur_us"])
        for event in all_mappings
        if event.get("attribution_method") == "unique_kernel_signature"
    )
    if args.phase == "decode":
        node_metrics = _metrics_for_rank(all_mappings, len(exact_decode))
        for cell in node_metrics.values():
            cell["ms_per_iter"] = round(float(cell["ms_per_iter"]), 6)
            cell["aggregation"] = (
                "mean kernel residency over "
                f"{selected_sample_count} balanced rank-local BS32 samples"
            )
            cell["source_worker_rank"] = "balanced_pool"
            cell["selected_samples_by_source"] = dict(sorted(exact_counts.items()))
    else:
        node_metrics = _aggregate_metrics(source_metrics)

    profile_id = f"qwen35_trtllm_attention_dp4_moe_ep4_agentx_{args.phase}"
    if args.phase == "decode" and decode_batch != 32:
        profile_id += f"_bs{decode_batch}"
    timeline = build_timeline_artifact(
        profile_id=profile_id,
        phase=args.phase,
        reference_rank=reference_rank,
        steps=reference_steps,
        timing_summary=timing_summary,
        raw_trace={
            "file": reference_path.name,
            "sha256": sha256_file(reference_path),
            "format": (
                "PyTorch profiler trace JSON"
                if trace_format == "torch"
                else "Nsight Systems SQLite export"
            ),
            "rank": reference_rank,
            "worker": reference_worker,
        },
        stack_source={
            "mode": (
                "kineto_gpu_annotation_and_conservative_kernel_signature"
                if trace_format == "torch"
                else "nsight_nvtx_and_cuda_graph_node_identity"
            ),
            "file": reference_path.name,
            "sha256": sha256_file(reference_path),
            "mapped_residency_ratio": round(attributed_residency_ratio, 6),
            "mapped_active_union_ratio": round(attributed_active_ratio, 6),
            "unmapped_residency_ratio": round(status_us["unmapped"] / total_us, 6),
            "policy": "unique kernel signatures remain mapped; unresolved graph occurrences remain explicit unmapped events with candidates",
        },
        target_resolver=QWEN35_TIMELINE_TARGETS,
    )

    if args.phase == "decode":
        selected_generation_requests = [
            row["generation_reqs"] for row in exact_decode
        ]
        measured_shape = {
            "generation_requests": {
                "samples": selected_generation_requests,
                "min": min(selected_generation_requests),
                "median": statistics.median(
                    selected_generation_requests
                ),
                "max": max(selected_generation_requests),
            },
            "selected_exact_generation_requests": decode_batch,
            "selected_samples": len(exact_decode),
            "selected_samples_by_source": dict(sorted(exact_counts.items())),
            "raw_exact_samples": len(raw_exact_decode),
            "raw_exact_samples_by_source": dict(sorted(raw_exact_counts.items())),
        }
    else:
        owner_shapes = [row for row in shape_observations if row["owner_compute"]]
        measured_shape = {
            "owner_context": owner_shapes,
            "one_chunk_8k_owner_samples": sum(
                row["context_reqs"] == 1 and row["context_tokens"] == 8192
                for row in owner_shapes
            ),
            "selected_exact_context_shape": {
                "requests": 1,
                "tokens": 8192,
            },
            "selected_samples": len(exact_8k),
            "selected_samples_by_source": dict(sorted(exact_counts.items())),
        }

    node_states = {}
    if args.phase == "decode":
        node_states = {
            "generation_loop.candidate_tokens": {
                "status": "fused",
                "included_in": "generation_loop.draft_propose",
            },
            "generation_loop.target_verify": {
                "status": "fused",
                "included_in": "top.decoder_stack",
            },
            "generation_loop.tentative_state": {
                "status": "fused",
                "included_in": "generation_loop.target_verify",
            },
            "generation_loop.accept_prefix": {
                "status": "unobserved",
                "reason": "accept/sample is outside the worker-local _forward_step NVTX interval and has no uniquely attributable kernel in this capture",
            },
            "generation_loop.commit_kv": {
                "status": "unobserved",
                "reason": "the captured Torch/Kineto path has no uniquely attributable standalone KV-prefix commit kernel",
            },
            "generation_loop.replay_gdn": {
                "status": "unobserved",
                "reason": "the captured TRT path exposes state promotion/commit kernels but no separate accepted-prefix replay interval",
            },
            "generation_loop.commit_tokens": {
                "status": "unobserved",
                "reason": "token publication is outside the worker-local _forward_step NVTX interval",
            },
            "generation_loop.next_iteration": {
                "status": "unobserved",
                "reason": "host-side loop control is outside the worker-local _forward_step NVTX interval",
            },
        }

    profile = {
        "schema_version": "profile.v2",
        "profile_id": profile_id,
        "label": (
            "Qwen3.5 397B · TRT-LLM · exact 8K/1K C704 · DEP4 + MTP6 · "
            f"{trace_format.upper()} {selected_sample_count}×BS{decode_batch} decode"
            if args.phase == "decode"
            else "Qwen3.5 397B · TRT-LLM · AgentX DEP4 + MTP6 · prefill"
        ),
        "model_id": "qwen35_397b_a17b",
        "execution_path_id": "attention_dp4_moe_ep4",
        "implementation_id": "trtllm_1cef02e9_attention_dp4_moe_ep4_mtp",
        "variant_id": (
            f"trtllm_agentx_dep4_mtp6_{args.phase}_8k1k_c704"
            + (
                f"_bs{decode_batch}"
                if args.phase == "decode" and decode_batch != 32
                else ""
            )
        ),
        "phase": args.phase,
        "generation_mode": "mtp",
        "entry_view": "top",
        "execution_parameters": {
            "tp_size": 1,
            "dp_size": 4,
            "cp_size": 1,
            "ep_size": 4,
        },
        "hardware": {
            "gpu": "GB300",
            "gpus_per_node": 4,
            "nodes": expected_workers,
            "topology_scope": f"{expected_workers} disaggregated {args.phase} workers",
        },
        "workload": (
            {
                "suite": "exact-8k1k",
                "concurrency": 704,
                "comparison_contract": comparison_contract,
                "mtp_draft_tokens": 6,
                "decode_cuda_graph_batch_cap": 32,
                "accepted_length": {
                    "control": "TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS=4.80",
                    "requested_mean": 4.8,
                    "evidence": "validated immutable launch config",
                    "observed_histogram": None,
                    "interpretation": (
                        "TRT-LLM exposes the forced acceptance simulator setting but "
                        "does not log a per-iteration accepted-length histogram; this "
                        "field is configuration-bound, not an inferred measurement"
                    ),
                },
                "measured_shape": measured_shape,
            }
            if args.phase == "decode"
            else {
                "suite": "AgentX A-Z97",
                "concurrency": 704,
                "mtp_draft_tokens": 6,
                "measured_shape": measured_shape,
            }
        ),
        "profiler": {
            "type": (
                "torch_kineto_worker_local"
                if trace_format == "torch"
                else "nsight_systems_worker_local"
            ),
            "rank": (
                "one runtime-elected exact-BS32 rank on each decode worker "
                f"({', '.join(sorted(comparison_sources))}); {selected_sample_count} "
                "complete rank-local samples balanced "
                f"{selected_sample_count // 2} per worker to match the SGLang source pool"
                if args.phase == "decode"
                else "all four DEP ranks on every worker"
            ),
            "activities": (
                ["CPU", "CUDA"] if trace_format == "torch" else ["cuda", "nvtx"]
            ),
            "with_stack": trace_format == "torch",
            "record_shapes": False,
            "cuda_graph_enabled": args.phase == "decode",
            "gpu_metric_semantics": (
                "per-node kernel residency is averaged over one balanced pool of "
                f"{selected_sample_count} rank-local BS32 samples; parallel ranks "
                "and workers are never summed"
                if args.phase == "decode"
                else "maximum worker/rank residency; parallel ranks and workers are not summed"
            ),
            "runtime_launch_parallelism": {
                "framework_world_size": 4,
                "attention_dp_size": 4,
                "moe_ep_size": 4,
                "normalization": "the runtime process group carries replicated attention DP and sharded MoE EP; it is not semantic TP4",
            },
        },
        "evidence": {
            "job_id": args.job_id,
            "baseline_job_id": 501238,
            "tensorrt_llm_commit": TRT_COMMIT,
            "py_executor_base_sha256": PY_EXECUTOR_BASE_SHA256,
            "py_executor_profile_overlay_sha256": (
                PY_EXECUTOR_TORCH_OVERLAY_SHA256
                if trace_format == "torch"
                else PY_EXECUTOR_PROFILE_OVERLAY_SHA256
            ),
            "dynamo_handler_base_sha256": DYNAMO_HANDLER_BASE_SHA256,
            "dynamo_exact_output_overlay_sha256": DYNAMO_EXACT_OUTPUT_OVERLAY_SHA256,
            "dynamo_wheel_base_sha256": DYNAMO_WHEEL_BASE_SHA256,
            "dynamo_exact_output_wheel_sha256": DYNAMO_EXACT_OUTPUT_WHEEL_SHA256,
            "model_revision": MODEL_REVISION,
            "model_config_sha256": MODEL_CONFIG_SHA256,
            "container_sha256": CONTAINER_SHA256,
            "report_files": [
                {
                    "worker": worker,
                    "rank": rank,
                    "file": path.name,
                    "sha256": sha256_file(path),
                    **(
                        {"nsys_export": nsys_export_metadata[(worker, rank)]}
                        if trace_format == "nsys"
                        else {"format": "PyTorch profiler trace JSON"}
                    ),
                }
                for (worker, rank), path in sorted(paths.items())
            ],
            "nsys_report_files": raw_nsys_reports,
            "exact_worker_logs": exact_worker_logs,
            "comparison_workload": workload_evidence,
            **(
                {"nsys_export": nsys_export_metadata[min(nsys_export_metadata)]}
                if trace_format == "nsys"
                else {}
            ),
            "mapping_policy": (
                "Kineto GPU executor annotation + concrete kernel signature + exact GGGA/MTP6 order"
                if trace_format == "torch"
                else "NVTX step + runtime correlation + CUDA Graph node occurrence + exact GGGA/MTP6 order"
            ),
            "selection_policy": (
                f"exactly {selected_sample_count} real generation_reqs={decode_batch} "
                f"rank-local events, balanced {selected_sample_count // 2} per "
                "runtime-elected worker source and "
                "time-spread over each capture"
                if args.phase == "decode"
                else "exact one-request/8192-token owner events only"
            ),
            "mapped_or_fusion_duration_ratio": round(attributed_residency_ratio, 6),
            "mapped_or_fusion_active_union_ratio": round(attributed_active_ratio, 6),
            "strict_signature_duration_ratio": round(strict_signature_us / total_us, 6),
            "mapped_duration_ratio": round(status_us["mapped"] / total_us, 6),
            "fusion_duration_ratio": round(status_us["fusion"] / total_us, 6),
            "unmapped_duration_ratio": round(status_us["unmapped"] / total_us, 6),
            "timeline_interval_coverage_ratio": round(
                sum(status_us.values()) / total_us, 6
            ),
            "semantic_attribution_gate": {
                "metric": "mapped_or_fusion_active_union_ratio",
                "threshold": 0.90,
                "passed": attributed_active_ratio >= 0.90,
            },
            "critical_step_wall_ms": timing_summary["critical_step_wall_ms"],
            "critical_cpu_launch_wall_ms": timing_summary[
                "critical_cpu_launch_wall_ms"
            ],
            "four_rank_validation": True,
            "worker_count": expected_workers,
        },
        "timeline": {},
        "node_states": node_states,
        "node_metrics": node_metrics,
    }
    analysis = {
        "profile_id": profile_id,
        "phase": args.phase,
        "workers": workers,
        "process_checks": process_checks,
        "validations": validations,
        "shape_observations": shape_observations,
        "owner_rank_positions": sorted(owner_rank_positions),
        "reference_source": reference_source,
        "timing_summary": timing_summary,
        "status_duration_us": dict(status_us),
        "mapped_or_fusion_duration_ratio": attributed_residency_ratio,
        "mapped_or_fusion_active_union_ratio": attributed_active_ratio,
        "strict_signature_duration_ratio": strict_signature_us / total_us,
        "node_metrics": profile["node_metrics"],
    }
    return profile, timeline, analysis, all_mappings


def main() -> int:
    args = parse_args()
    profile, timeline, analysis, mappings = build(args)
    timeline_sha = write_timeline_artifact(args.output_timeline, timeline)
    profile["timeline"] = {
        "schema_version": "timeline.v1",
        "artifact": args.output_timeline.name,
        "sha256": timeline_sha,
        "reference_rank": timeline["reference_rank"],
        "step_count": len(timeline["steps"]),
        "event_count": sum(len(step["events"]) for step in timeline["steps"]),
        "raw_trace_file": timeline["raw_trace"]["file"],
    }
    args.output_profile.parent.mkdir(parents=True, exist_ok=True)
    args.output_profile.write_text(
        yaml.safe_dump(profile, sort_keys=False), encoding="utf-8"
    )
    args.output_analysis.parent.mkdir(parents=True, exist_ok=True)
    args.output_analysis.write_text(json.dumps(analysis, indent=2) + "\n")
    args.output_mapping.parent.mkdir(parents=True, exist_ok=True)
    with args.output_mapping.open("w") as output:
        for row in mappings:
            output.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(f"wrote {args.output_profile.resolve()}")
    print(
        f"phase={args.phase} "
        "attributed-active="
        f"{profile['evidence']['mapped_or_fusion_active_union_ratio']:.3f} "
        "attributed-residency="
        f"{profile['evidence']['mapped_or_fusion_duration_ratio']:.3f} "
        f"strict={profile['evidence']['strict_signature_duration_ratio']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
