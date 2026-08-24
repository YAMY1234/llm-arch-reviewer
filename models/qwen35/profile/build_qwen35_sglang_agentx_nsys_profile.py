#!/usr/bin/env python3
"""Build an exact-batch SGLang AgentX profile from worker-local NSYS reports."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import statistics
import sys
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.common.timeline_artifact import build_timeline_artifact, write_timeline_artifact
from models.qwen35.profile.build_qwen35_sglang_agentx_profile import (
    LOG_ROW_RE,
    _aggregate_source_metrics,
    _validate_fingerprints,
    parse_benchmark_snapshot,
    worker_identity,
)
from models.qwen35.profile.build_qwen35_sglang_decode_profile import (
    CONTAINER_SHA256,
    MODEL_CONFIG_SHA256,
    MODEL_REVISION,
    RUNTIME_SOURCE_COMMIT,
    SGLANG_DECODE_NODE_STATES,
    SOURCE_COMMIT,
    _metrics_for_rank,
    _validate_step_signatures,
    sha256_file,
)
from models.qwen35.profile.qwen35_graph_mapping import (
    attribution_active_union_ratio,
    attach_graph_stack_evidence,
    load_contextual_eager_signatures,
    load_unique_eager_kernel_signatures,
    map_graph_window,
)
from models.qwen35.profile.qwen35_nsys_mapping import (
    load_sglang_nsys_steps,
    read_nsys_export_metadata,
    sglang_nsys_trace_events,
    validate_sglang_all_rank_capture_integrity,
    validate_sglang_graph_node_stability,
)
from models.qwen35.profile.qwen35_a2a_contract import (
    validate_comparison_workload,
)
from models.qwen35.profile.qwen35_timeline import QWEN35_TIMELINE_TARGETS


DEFAULT_SELECTED_BATCH = 32
PROFILING_SOURCE_COMMIT = "128e0e31b51d58b69cf9c8006a09213d34cfb3a6"
PROFILING_OVERLAY_COMMIT = "128e0e31b51d58b69cf9c8006a09213d34cfb3a6"
SRT_SLURM_CAPTURE_COMMIT = "581ba9aa54736ef520592592bca75f5d32ca8eb9"
PROFILER_MANAGER_SHA256 = (
    "d32fefc52062b19d81c9110e74e10bfc6e1327fc80ea96a7de449f6dce1d4a45"
)
SCHEDULER_SHA256 = (
    "8676ceac0e7cbb6d8ca1c3902d143d9708c44b297b36027121fa719942b6f598"
)
SCHEDULER_NVTX_SHA256 = (
    "56610ee61c53c39e40fdd6b44c7443140eeb6e25bc499889e70f93a33bf3fcdd"
)
RUNTIME_MANIFEST_SHA256 = (
    "5e57a354a6113117bf8107fcfeb77cc543955163f31506b3502df9eb04beba5b"
)
SYMM_MEM_GATHER_SHA256 = (
    "8a1f8e9a1f13c26b89691eb0dc7bec07595b107778f180d1afa0a93d5e8af9c4"
)
SOURCE_RE = re.compile(r"^w(?P<worker>[01])/r(?P<rank>[0-3])$")
EXACT_GATE_RE = re.compile(
    r"DP(?P<rank>\d+) TP\d+ EP\d+\] "
    r"(?:Worker-wide |All-DP )?[Ee]xact running-batch Nsight gate matched: "
    r"(?:reduction=(?P<reduction>all|any) )?"
    r"batch=(?P<batch>\d+) forward_ct=(?P<forward_ct>\d+)"
    r"(?: (?:local_)?warmup_batches=(?P<warmup_batches>\d+))?"
)
SYNC_READY_RE = re.compile(
    r"DP(?P<rank>\d+) TP\d+ EP\d+\] Exact-batch Nsight sync group ready: "
    r"world_size=(?P<world_size>\d+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlites", type=Path, nargs=2, required=True)
    parser.add_argument("--nsys-reports", type=Path, nargs=2, required=True)
    parser.add_argument("--worker-logs", type=Path, nargs=2, required=True)
    parser.add_argument("--fingerprints", type=Path, nargs=2, required=True)
    parser.add_argument("--benchmark-log", type=Path, required=True)
    parser.add_argument("--job-metadata", type=Path, required=True)
    parser.add_argument("--job-id", type=int, required=True)
    parser.add_argument("--selected-batch", type=int, default=DEFAULT_SELECTED_BATCH)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--workload-result", type=Path, required=True)
    parser.add_argument("--eager-mapping", type=Path, required=True)
    parser.add_argument("--eager-trace", type=Path, required=True)
    parser.add_argument("--output-profile", type=Path, required=True)
    parser.add_argument("--output-timeline", type=Path, required=True)
    parser.add_argument("--output-analysis", type=Path, required=True)
    parser.add_argument("--output-mapping", type=Path, required=True)
    return parser.parse_args()


def _report_worker(path: Path) -> int:
    return worker_identity(path)


def _source_coordinates(source: str) -> tuple[int, int]:
    match = SOURCE_RE.fullmatch(source)
    if not match:
        raise ValueError(f"invalid worker/rank source: {source}")
    return int(match.group("worker")), int(match.group("rank"))


def _validate_nsys_report_files(paths: list[Path]) -> dict[int, Path]:
    reports = {_report_worker(path): path.resolve() for path in paths}
    if set(reports) != {0, 1}:
        raise ValueError(f"incomplete SGLang raw NSYS reports: {sorted(reports)}")
    if any(
        not path.is_file() or path.stat().st_size <= 0 for path in reports.values()
    ):
        raise ValueError("formal SGLang raw NSYS report is missing or empty")
    return reports


def _validate_nsys_capture_contract(
    profiling: dict[str, Any], decode_environment: dict[str, Any]
) -> tuple[str, str]:
    if str(decode_environment.get("NSYS_NVTX_PROFILER_REGISTER_ONLY")) != "0":
        raise ValueError(
            "formal SGLang NSYS profile requires unregistered NVTX capture messages"
        )
    capture_range = str(
        decode_environment.get("SGLANG_NSYS_NVTX_CAPTURE_RANGE") or ""
    ).strip()
    if not capture_range:
        raise ValueError(
            "formal SGLang NSYS profile requires an NVTX capture range"
        )

    extra_args = [str(arg) for arg in (profiling.get("extra_nsys_args") or [])]
    capture_mode = None
    capture_selector = None
    capture_end = None
    for index, arg in enumerate(extra_args):
        if arg in {"-c", "--capture-range"} and index + 1 < len(extra_args):
            capture_mode = extra_args[index + 1]
        elif arg.startswith("-c=") or arg.startswith("--capture-range="):
            capture_mode = arg.split("=", 1)[1]
        if arg in {"-p", "--nvtx-capture"} and index + 1 < len(extra_args):
            capture_selector = extra_args[index + 1]
        elif arg.startswith("-p=") or arg.startswith("--nvtx-capture="):
            capture_selector = arg.split("=", 1)[1]
        if arg == "--capture-range-end" and index + 1 < len(extra_args):
            capture_end = extra_args[index + 1]
        elif arg.startswith("--capture-range-end="):
            capture_end = arg.split("=", 1)[1]

    if capture_mode != "nvtx" or capture_selector != f"{capture_range}@*":
        raise ValueError(
            "formal SGLang NSYS profile requires matching '-c nvtx' and an "
            "explicit all-domain NVTX capture-range selector"
        )
    if capture_end != "repeat:1:async":
        raise ValueError(
            "formal SGLang NSYS profile requires immediate asynchronous report finalization"
        )
    if "cudaProfilerApi" in extra_args:
        raise ValueError("formal SGLang NSYS profile cannot use cudaProfilerApi")
    return capture_range, capture_end


def parse_exact_batch_capture_observations(
    path: Path,
    *,
    selected_batch: int,
    expected_steps: int,
    expected_sync_world_size: int = 4,
    expected_warmup_batches: int = 1,
    expected_gate_reduction: str | None = None,
) -> dict[int, dict[str, Any]]:
    """Prove a synchronized raw capture and recover rank-local exact events."""

    lines = path.read_text(errors="replace").splitlines()
    sync_ready: dict[int, int] = {}
    for line in lines:
        match = SYNC_READY_RE.search(line)
        if match is None:
            continue
        rank = int(match.group("rank"))
        if rank in sync_ready:
            raise ValueError(f"{path}: duplicate sync-group proof for DP{rank}")
        sync_ready[rank] = int(match.group("world_size"))
    if set(sync_ready) != {0, 1, 2, 3}:
        raise ValueError(
            f"{path}: incomplete all-DP sync-group proof: {sorted(sync_ready)}"
        )
    invalid_sync = {
        rank: world_size
        for rank, world_size in sync_ready.items()
        if world_size != expected_sync_world_size
    }
    if invalid_sync:
        raise ValueError(
            f"{path}: exact-batch sync group is not world size "
            f"{expected_sync_world_size}: {invalid_sync}"
        )

    gates: dict[int, tuple[int, re.Match[str]]] = {}
    for index, line in enumerate(lines):
        match = EXACT_GATE_RE.search(line)
        if match is None:
            continue
        rank = int(match.group("rank"))
        if rank in gates:
            raise ValueError(f"{path}: duplicate exact-batch gate for DP{rank}")
        gates[rank] = (index, match)
    if set(gates) != {0, 1, 2, 3}:
        raise ValueError(f"{path}: incomplete all-DP exact-batch gates: {sorted(gates)}")

    result: dict[int, dict[str, Any]] = {}
    local_warmups: dict[int, int] = {}
    for rank, (gate_index, gate) in sorted(gates.items()):
        gate_batch = int(gate.group("batch"))
        if gate_batch != selected_batch:
            raise ValueError(
                f"{path}: DP{rank} gate selected BS{gate_batch}, expected BS{selected_batch}"
            )
        reduction = gate.group("reduction")
        if expected_gate_reduction is not None and reduction != expected_gate_reduction:
            raise ValueError(
                f"{path}: DP{rank} gate reduction is {reduction!r}, expected "
                f"{expected_gate_reduction!r}"
            )
        warmup_batches = int(gate.group("warmup_batches") or 0)
        local_warmups[rank] = warmup_batches
        prefix = f"DP{rank} TP{rank} EP{rank}]"
        start = next(
            (
                index
                for index, line in enumerate(lines[gate_index + 1 :], gate_index + 1)
                if prefix in line and "Profiling starts." in line
            ),
            None,
        )
        if start is None:
            raise ValueError(f"{path}: DP{rank} gate lacks profiler start")
        stop = next(
            (
                index
                for index, line in enumerate(lines[start + 1 :], start + 1)
                if prefix in line and "Stop profiling..." in line
            ),
            None,
        )
        if stop is None:
            raise ValueError(f"{path}: DP{rank} capture lacks profiler stop")
        done = next(
            (
                index
                for index, line in enumerate(lines[stop + 1 :], stop + 1)
                if prefix in line and "Profiling done." in line
            ),
            None,
        )
        if done is None:
            raise ValueError(f"{path}: DP{rank} capture lacks profiler completion")

        observations = []
        for line in lines[start + 1 : stop]:
            match = LOG_ROW_RE.search(line)
            if match is None or int(match.group("rank")) != rank:
                continue
            observations.append(
                {
                    "dp_rank": rank,
                    "scheduler_step": int(match.group("step")),
                    "running_requests": int(match.group("running")),
                    "full_tokens": int(match.group("full_tokens")),
                    "accepted_length": float(match.group("accept")),
                    "retracted_requests": int(match.group("retracted")),
                    "cuda_graph": match.group("graph") == "True",
                    "queued_requests": int(match.group("queue")),
                }
            )
        if len(observations) != expected_steps:
            raise ValueError(
                f"{path}: DP{rank} expected {expected_steps} captured decode rows, "
                f"found {len(observations)}"
            )
        gate_forward_ct = int(gate.group("forward_ct"))
        # The profiler predicate runs immediately before the current forward;
        # the scheduler's post-forward log labels that same captured batch as
        # forward_ct - 1 (confirmed by the live 8K/1K gate evidence).
        first_captured_step = gate_forward_ct - 1
        expected_forward_cts = list(
            range(first_captured_step, first_captured_step + expected_steps)
        )
        actual_forward_cts = [row["scheduler_step"] for row in observations]
        if actual_forward_cts != expected_forward_cts:
            raise ValueError(
                f"{path}: DP{rank} captured scheduler steps are not contiguous from "
                f"the exact pre-forward gate: {actual_forward_cts}"
            )
        exact_observations = [
            row
            for row in observations
            if row["running_requests"] == selected_batch
        ]
        invalid_exact = [
            row
            for row in exact_observations
            if not row["cuda_graph"] or row["retracted_requests"]
        ]
        if invalid_exact:
            raise ValueError(
                f"{path}: DP{rank} exact-BS{selected_batch} candidates contain "
                f"retraction or graph-off rows: {invalid_exact[:3]}"
            )
        result[rank] = {
            "gate_forward_ct": gate_forward_ct,
            "sync_world_size": sync_ready[rank],
            "gate_reduction": reduction,
            "local_warmup_batches": warmup_batches,
            "capture_observation_count": len(observations),
            "captured_batch_distribution": dict(
                sorted(Counter(row["running_requests"] for row in observations).items())
            ),
            "exact_observation_count": len(exact_observations),
            "observations": observations,
            "profiler_completed": True,
        }
    if max(local_warmups.values(), default=0) < expected_warmup_batches:
        raise ValueError(
            f"{path}: synchronized gate has no rank with "
            f"{expected_warmup_batches} exact-batch warmup(s): {local_warmups}"
        )
    return result


def select_balanced_exact_observations(
    observations_by_worker: dict[int, dict[int, dict[str, Any]]],
    *,
    selected_batch: int,
    sample_count: int,
) -> list[dict[str, Any]]:
    """Select a deterministic, source-balanced rank-local exact-BS sample."""

    candidates: dict[str, list[dict[str, Any]]] = {}
    for worker, ranks in sorted(observations_by_worker.items()):
        for rank, evidence in sorted(ranks.items()):
            source = f"w{worker}/r{rank}"
            rows = [
                {
                    **row,
                    "worker": worker,
                    "rank": rank,
                    "source": source,
                    "capture_iteration": iteration,
                }
                for iteration, row in enumerate(evidence["observations"])
                if row["running_requests"] == selected_batch
                and row["cuda_graph"]
                and not row["retracted_requests"]
            ]
            if rows:
                candidates[source] = rows

    available = sum(len(rows) for rows in candidates.values())
    if available < sample_count:
        counts = {source: len(rows) for source, rows in candidates.items()}
        raise ValueError(
            f"SGLang raw capture has only {available} valid rank-local "
            f"BS{selected_batch} observations; need {sample_count}: {counts}"
        )

    quotas: Counter[str] = Counter()
    sources = sorted(candidates)
    while sum(quotas.values()) < sample_count:
        progressed = False
        for source in sources:
            if quotas[source] >= len(candidates[source]):
                continue
            quotas[source] += 1
            progressed = True
            if sum(quotas.values()) == sample_count:
                break
        if not progressed:
            raise AssertionError("exact-observation quota allocation stalled")

    selected: list[dict[str, Any]] = []
    for source in sources:
        rows = candidates[source]
        quota = quotas[source]
        if not quota:
            continue
        # Spread each source's quota over its complete raw capture instead of
        # taking only the earliest contiguous occurrences.
        indices = [((2 * index + 1) * len(rows)) // (2 * quota) for index in range(quota)]
        selected.extend(rows[index] for index in indices)
    selected.sort(
        key=lambda row: (
            int(row["capture_iteration"]),
            int(row["worker"]),
            int(row["rank"]),
        )
    )
    for sample_index, row in enumerate(selected):
        row["sample_index"] = sample_index
    if len(selected) != sample_count:
        raise AssertionError(f"selected {len(selected)} samples, expected {sample_count}")
    return selected


def parse_exact_batch_capture_observation(
    path: Path, *, selected_batch: int
) -> dict[str, Any]:
    """Compatibility wrapper for the historical two-step DP0 contract."""

    lines = path.read_text(errors="replace").splitlines()
    gates = [
        (index, match)
        for index, line in enumerate(lines)
        if (match := EXACT_GATE_RE.search(line)) is not None
        and int(match.group("rank")) == 0
    ]
    if len(gates) != 1:
        raise ValueError(f"{path}: expected one DP0 exact-batch gate")
    gate_index, gate = gates[0]
    start = next(
        index
        for index, line in enumerate(lines[gate_index + 1 :], gate_index + 1)
        if "DP0 TP0 EP0] Profiling starts." in line
    )
    stop = next(
        index
        for index, line in enumerate(lines[start + 1 :], start + 1)
        if "DP0 TP0 EP0] Stop profiling..." in line
    )
    done = next(
        (
            index
            for index, line in enumerate(lines[stop + 1 :], stop + 1)
            if "DP0 TP0 EP0] Profiling done." in line
        ),
        None,
    )
    if done is None:
        raise ValueError(f"{path}: missing DP0 profiler completion marker")
    rows = []
    for line in lines[start + 1 : stop]:
        match = LOG_ROW_RE.search(line)
        if match is None:
            continue
        rows.append(
            {
                "dp_rank": int(match.group("rank")),
                "scheduler_step": int(match.group("step")),
                "running_requests": int(match.group("running")),
                "full_tokens": int(match.group("full_tokens")),
                "accepted_length": float(match.group("accept")),
                "retracted_requests": int(match.group("retracted")),
                "cuda_graph": match.group("graph") == "True",
                "queued_requests": int(match.group("queue")),
            }
        )
    gate_forward_ct = int(gate.group("forward_ct"))
    dp0_rows = [row for row in rows if row["dp_rank"] == 0]
    exact_rows = [row for row in dp0_rows if row["scheduler_step"] == gate_forward_ct]
    if len(exact_rows) != 1:
        raise ValueError(f"{path}: DP0 gate step is missing or duplicated")
    exact = exact_rows[0]
    if exact["running_requests"] != selected_batch:
        raise ValueError(f"{path}: DP0 gate step is not exact BS{selected_batch}")
    peer_rows = [row for row in rows if row["scheduler_step"] == gate_forward_ct]
    if {row["dp_rank"] for row in peer_rows} != {0, 1, 2, 3}:
        raise ValueError(f"{path}: historical gate step lacks peer-rank observations")
    return {
        **exact,
        "gate_forward_ct": gate_forward_ct,
        "logged_rows_before_exact": sum(
            row["scheduler_step"] < gate_forward_ct for row in dp0_rows
        ),
        "capture_dp0_observation_count": len(dp0_rows),
        "rank_local_batches_at_exact_step": {
            f"r{row['dp_rank']}": {
                "running_requests": row["running_requests"],
                "full_tokens": row["full_tokens"],
                "accepted_length": row["accepted_length"],
                "queued_requests": row["queued_requests"],
                "retracted_requests": row["retracted_requests"],
                "cuda_graph": row["cuda_graph"],
            }
            for row in sorted(peer_rows, key=lambda item: item["dp_rank"])
        },
        "profiler_completed": True,
    }


def _timing(mapped: list[dict[str, Any]], *, logical_period_us: float) -> dict[str, float]:
    intervals = [
        (
            int(round(float(event["ts_us"]) * 1000.0)),
            int(round((float(event["ts_us"]) + float(event["dur_us"])) * 1000.0)),
        )
        for event in mapped
    ]
    start_ns = min(start for start, _end in intervals)
    end_ns = max(end for _start, end in intervals)
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    active_ns = sum(end - start for start, end in merged)
    residency_us = sum(float(event["dur_us"]) for event in mapped)
    gpu_span_us = (end_ns - start_ns) / 1000.0
    active_us = active_ns / 1000.0
    if active_us > gpu_span_us + 1e-6 or active_us > residency_us + 1e-6:
        raise ValueError("SGLang NSYS timing invariants do not close")
    return {
        "logical_step_period_us": logical_period_us,
        "gpu_span_us": gpu_span_us,
        "gpu_busy_union_us": active_us,
        "gpu_residency_us": residency_us,
    }


def build(args: argparse.Namespace):
    selected_batch = int(getattr(args, "selected_batch", DEFAULT_SELECTED_BATCH))
    if selected_batch < 1 or selected_batch > 80:
        raise ValueError(f"selected SGLang batch must be in 1..80, got {selected_batch}")
    profile_id = (
        "qwen35_sglang_attention_dp4_moe_ep4_mtp6_agentx_nsys_"
        f"bs{selected_batch}"
    )
    comparison_contract, workload_evidence = validate_comparison_workload(
        engine="sglang",
        config=args.config,
        dataset=args.dataset,
        dataset_manifest=args.dataset_manifest,
        workload_result=args.workload_result,
    )
    config = yaml.safe_load(args.config.read_text())
    job = json.loads(args.job_metadata.read_text())
    if int(job.get("job_id", -1)) != args.job_id:
        raise ValueError(
            f"job metadata ID {job.get('job_id')} does not match --job-id {args.job_id}"
        )
    actual_source = (
        (((config.get("identity") or {}).get("frameworks") or {}).get("sglang_source"))
    )
    if actual_source != PROFILING_SOURCE_COMMIT:
        raise ValueError(
            f"SGLang profiling source mismatch: expected {PROFILING_SOURCE_COMMIT}, "
            f"got {actual_source}"
        )
    profiling = config.get("profiling") or {}
    decode_environment = ((config.get("backend") or {}).get("decode_environment") or {})
    if profiling.get("type") != "nsys":
        raise ValueError("formal SGLang matched profile requires profiling.type=nsys")
    if str(decode_environment.get("SGLANG_ENABLE_NVTX_SCHEDULER")) not in {"1", "true", "True"}:
        raise ValueError("formal SGLang NSYS profile requires scheduler NVTX")
    capture_range, capture_range_end = _validate_nsys_capture_contract(
        profiling, decode_environment
    )
    decode_profiling = profiling.get("decode") or {}
    capture_start_step = int(decode_profiling.get("start_step", -1))
    capture_stop_step = int(decode_profiling.get("stop_step", -1))
    if capture_start_step < 0 or capture_stop_step <= capture_start_step:
        raise ValueError(
            "formal SGLang NSYS profile requires a valid decode scheduler-step window"
        )

    eager_signatures = load_unique_eager_kernel_signatures(args.eager_mapping)
    contextual_signatures = load_contextual_eager_signatures(
        trace_path=args.eager_trace.resolve(),
        mapping_path=args.eager_mapping.resolve(),
    )
    fingerprint_rows = _validate_fingerprints(args.fingerprints)
    benchmark = parse_benchmark_snapshot(args.benchmark_log)
    expected_selected_samples = int(
        comparison_contract["selected_rank_local_samples"]
    )
    raw_capture_steps = int(
        decode_environment.get("SGLANG_NSYS_EXACT_DECODE_BATCHES", -1)
    )
    if raw_capture_steps < expected_selected_samples:
        raise ValueError(
            "SGLang raw NSYS capture must be at least as wide as the selected "
            f"sample: raw={raw_capture_steps}, selected={expected_selected_samples}"
        )
    exact_observations = {
        _report_worker(path): parse_exact_batch_capture_observations(
            path,
            selected_batch=selected_batch,
            expected_steps=raw_capture_steps,
            expected_warmup_batches=int(
                decode_environment.get("SGLANG_NSYS_EXACT_WARMUP_BATCHES", 1)
            ),
            expected_gate_reduction=str(
                decode_environment.get("SGLANG_NSYS_EXACT_GATE_REDUCTION", "")
            ),
        )
        for path in args.worker_logs
    }
    if set(exact_observations) != {0, 1}:
        raise ValueError(f"incomplete SGLang worker logs: {sorted(exact_observations)}")
    selected_sample_rows = select_balanced_exact_observations(
        exact_observations,
        selected_batch=selected_batch,
        sample_count=expected_selected_samples,
    )
    selected_sample_by_key = {
        (int(row["worker"]), int(row["rank"]), int(row["capture_iteration"])): row
        for row in selected_sample_rows
    }

    reports = {_report_worker(path): path.resolve() for path in args.sqlites}
    if set(reports) != {0, 1}:
        raise ValueError(f"incomplete SGLang NSYS reports: {sorted(reports)}")
    nsys_reports = _validate_nsys_report_files(args.nsys_reports)
    nsys_export_metadata = {
        worker: read_nsys_export_metadata(path)
        for worker, path in sorted(reports.items())
    }
    if len({tuple(sorted(row.items())) for row in nsys_export_metadata.values()}) != 1:
        raise ValueError(
            f"SGLang workers use different Nsight exporters: {nsys_export_metadata}"
        )
    all_rank_integrity = {
        worker: validate_sglang_all_rank_capture_integrity(
            path, capture_range_label=capture_range
        )
        for worker, path in sorted(reports.items())
    }
    structurally_validated_sources = [
        f"w{worker}/r{rank}"
        for worker in sorted(all_rank_integrity)
        for rank in range(4)
    ]

    source_metrics: dict[str, dict[str, Any]] = {}
    source_validation: dict[str, dict[str, Any]] = {}
    source_selected_counts: dict[str, int] = {}
    all_mappings: list[dict[str, Any]] = []
    selected_observations: list[dict[str, Any]] = []
    selected_steps_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    timing_by_iteration: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for worker, path in sorted(reports.items()):
        for rank in range(4):
            source = f"w{worker}/r{rank}"
            steps, parser_evidence = load_sglang_nsys_steps(
                path,
                rank=rank,
                capture_range_label=capture_range,
            )
            if len(steps) != raw_capture_steps:
                raise ValueError(
                    f"{source}: expected {raw_capture_steps} complete raw NSYS steps, "
                    f"found {len(steps)}"
                )
            gate = exact_observations[worker][rank]
            observations = gate["observations"]
            selected_iterations = {
                iteration
                for candidate_worker, candidate_rank, iteration in selected_sample_by_key
                if candidate_worker == worker and candidate_rank == rank
            }
            graph_stability = validate_sglang_graph_node_stability(
                step
                for iteration, step in enumerate(steps)
                if iteration in selected_iterations
            )
            source_mappings: list[dict[str, Any]] = []
            validation_rows = []

            for iteration, (step, row) in enumerate(zip(steps, observations)):
                sample = selected_sample_by_key.get((worker, rank, iteration))
                if sample is None:
                    continue
                sample_index = int(sample["sample_index"])
                trace_events, window, graph_roles = sglang_nsys_trace_events(
                    step, batch_size=int(row["running_requests"])
                )
                mapped, validation = map_graph_window(
                    trace_events,
                    window=window,
                    rank=rank,
                    step_index=step.step_id,
                    eager_signatures=eager_signatures,
                    contextual_signatures=contextual_signatures,
                )
                _validate_step_signatures(validation, rank=rank, step=step.step_id)
                for event in mapped:
                    event["event_id"] = f"w{worker}-r{rank}-{event['event_id']}"
                    event["worker"] = worker
                    event["scheduler_step"] = row["scheduler_step"]
                    event["gate_forward_ct"] = gate["gate_forward_ct"]
                    event["capture_iteration"] = iteration
                    event["selected_sample_index"] = sample_index
                timing = _timing(mapped, logical_period_us=step.cpu_wall_us)
                validation_rows.append(
                    {
                        **validation,
                        **timing,
                        "graph_roles": graph_roles,
                        "scheduler_step": row["scheduler_step"],
                        "gate_forward_ct": gate["gate_forward_ct"],
                        "capture_iteration": iteration,
                        "selected_sample_index": sample_index,
                    }
                )
                selected_observations.append(
                    {
                        "worker": worker,
                        "rank": rank,
                        "source": source,
                        "selected_sample_index": sample_index,
                        "capture_iteration": iteration,
                        "scheduler_step": row["scheduler_step"],
                        "gate_forward_ct": gate["gate_forward_ct"],
                        "running_requests": row["running_requests"],
                        "full_tokens": row["full_tokens"],
                        "mean_full_tokens_per_request": (
                            row["full_tokens"] / row["running_requests"]
                        ),
                        "accepted_length": row["accepted_length"],
                        "retracted_requests": row["retracted_requests"],
                        **timing,
                    }
                )
                timing_by_iteration[sample_index].append({"source": source, **timing})
                selected_steps_by_source[source].append(
                    {
                        "step_index": sample_index,
                        "capture_iteration": iteration,
                        "trace_start_us": min(float(event["ts_us"]) for event in mapped),
                        "timing": timing,
                        "mapped": mapped,
                    }
                )
                source_mappings.extend(mapped)
                all_mappings.extend(mapped)

            source_selected_counts[source] = len(selected_iterations)
            source_metrics[source] = (
                _metrics_for_rank(source_mappings, len(selected_iterations))
                if selected_iterations
                else {}
            )
            source_validation[source] = {
                "parser": parser_evidence,
                "graph_node_stability": graph_stability,
                "exact_gate": gate,
                "captured_batch_distribution": gate[
                    "captured_batch_distribution"
                ],
                "selected_exact_sample_count": len(selected_iterations),
                "selected_steps": validation_rows,
            }

    selected_sources = {
        source: count for source, count in source_selected_counts.items() if count
    }
    if not selected_sources:
        distributions = {
            source: evidence["captured_batch_distribution"]
            for source, evidence in source_validation.items()
        }
        raise ValueError(
            f"SGLang NSYS capture has no exact BS{selected_batch} step; "
            f"captured distributions={distributions}"
        )
    if sum(selected_sources.values()) != expected_selected_samples:
        raise ValueError(
            "SGLang exact-event selector did not close: "
            f"expected={expected_selected_samples}, selected={selected_sources}"
        )
    reference_source = min(
        selected_sources,
        key=lambda source: (-selected_sources[source], source),
    )
    reference_worker, reference_rank = _source_coordinates(reference_source)
    reference_steps = [
        {
            "step_index": row["step_index"],
            "label": f"Exact 8K/1K steady decode · NSYS BS{selected_batch}",
            "trace_start_us": row["trace_start_us"],
            "duration_us": row["timing"]["gpu_span_us"],
            "events": attach_graph_stack_evidence(
                row["mapped"], mapping_path=args.eager_mapping
            ),
        }
        for row in selected_steps_by_source[reference_source]
    ]

    critical_steps: dict[str, dict[str, Any]] = {}
    for iteration, rows in sorted(timing_by_iteration.items()):
        selected = max(rows, key=lambda row: float(row["gpu_span_us"]))
        critical_steps[f"iteration-{iteration:02d}"] = selected
    critical_gpu_span_ms = [
        float(row["gpu_span_us"]) / 1000.0 for row in critical_steps.values()
    ]
    logical_period_ms = [
        float(row["logical_step_period_us"]) / 1000.0
        for row in critical_steps.values()
    ]
    timing_summary = {
        "semantics": (
            "each sample is one real rank-local BS32 CUDA Graph decode period, "
            "bounded by adjacent scheduler.run_batch markers; 32 samples are "
            "balanced across available worker/rank sources and never summed"
        ),
        "critical_steps": critical_steps,
        "critical_gpu_span_ms": {
            "samples": critical_gpu_span_ms,
            "mean": statistics.fmean(critical_gpu_span_ms),
            "median": statistics.median(critical_gpu_span_ms),
            "min": min(critical_gpu_span_ms),
            "max": max(critical_gpu_span_ms),
        },
        "critical_logical_period_ms": {
            "samples": logical_period_ms,
            "mean": statistics.fmean(logical_period_ms),
            "median": statistics.median(logical_period_ms),
            "min": min(logical_period_ms),
            "max": max(logical_period_ms),
        },
    }

    total_us = sum(float(row["dur_us"]) for row in all_mappings)
    status_us: Counter[str] = Counter()
    for row in all_mappings:
        status_us[str(row["mapping_status"])] += float(row["dur_us"])
    attributed_residency_ratio = (
        status_us["mapped"] + status_us["fusion"]
    ) / total_us
    attributed_active_ratio = attribution_active_union_ratio(all_mappings)
    strict_signature_us = sum(
        float(row["dur_us"])
        for row in all_mappings
        if row.get("attribution_method") == "unique_kernel_signature"
    )

    reference_path = reports[reference_worker]
    timeline = build_timeline_artifact(
        profile_id=profile_id,
        phase="decode",
        reference_rank=reference_rank,
        steps=reference_steps,
        timing_summary=timing_summary,
        raw_trace={
            "file": reference_path.name,
            "sha256": sha256_file(reference_path),
            "format": "Nsight Systems SQLite export",
            "worker": reference_worker,
            "rank": reference_rank,
        },
        stack_source={
            "mode": "eager_stack_calibration_plus_nsys_graph_node_replay",
            "file": args.eager_mapping.name,
            "sha256": sha256_file(args.eager_mapping),
            "mapped_residency_ratio": round(attributed_residency_ratio, 6),
            "mapped_active_union_ratio": round(attributed_active_ratio, 6),
            "unmapped_residency_ratio": round(status_us["unmapped"] / total_us, 6),
            "policy": (
                "unique eager signature or validated graph role/layer/round slot; "
                "ambiguous occurrences remain unmapped with candidates"
            ),
        },
        target_resolver=QWEN35_TIMELINE_TARGETS,
    )

    selected_acceptance = [float(row["accepted_length"]) for row in selected_observations]
    measured_acceptance_mean = statistics.fmean(selected_acceptance)
    if abs(measured_acceptance_mean - 4.8) > 0.05:
        raise ValueError(
            "SGLang captured accept length does not match forced mean 4.8: "
            f"{measured_acceptance_mean}"
        )
    selected_full_tokens = [int(row["full_tokens"]) for row in selected_observations]
    selected_mean_tokens = [
        float(row["mean_full_tokens_per_request"])
        for row in selected_observations
    ]
    selected_queue_requests = [
        int(row["queued_requests"]) for row in selected_observations
    ]
    node_metrics = _metrics_for_rank(all_mappings, expected_selected_samples)
    for cell in node_metrics.values():
        cell["ms_per_iter"] = round(float(cell["ms_per_iter"]), 6)
        cell["aggregation"] = (
            "mean kernel residency over 32 balanced rank-local BS32 samples"
        )
        cell["source_worker_rank"] = "balanced_pool"
        cell["selected_samples_by_source"] = dict(
            sorted(selected_sources.items())
        )
    profile = {
        "schema_version": "profile.v2",
        "profile_id": profile_id,
        "label": (
            "Qwen3.5 397B · SGLang · exact 8K/1K C704 · DEP4 + MTP6 · "
            f"NSYS 32×BS{selected_batch}"
        ),
        "model_id": "qwen35_397b_a17b",
        "execution_path_id": "attention_dp4_moe_ep4",
        "implementation_id": "sglang_85c23c62_attention_dp4_moe_ep4_mtp",
        "variant_id": (
            "sglang_agentx_8k1k_c704_3p2d_dep4_mtp6_cg_nsys_"
            f"bs{selected_batch}"
        ),
        "phase": "decode",
        "generation_mode": "mtp",
        "entry_view": "top",
        "execution_parameters": {"tp_size": 1, "dp_size": 4, "cp_size": 1, "ep_size": 4},
        "hardware": {
            "gpu": "GB300",
            "gpus_per_worker": 4,
            "prefill_workers": 3,
            "decode_workers": 2,
        },
        "workload": {
            "scenario": "exact-8k1k",
            "concurrency": 704,
            "comparison_contract": comparison_contract,
            "selected_exact_target_verify_batch": selected_batch,
            "selected_samples": sum(source_selected_counts.values()),
            "selected_samples_by_source": dict(sorted(source_selected_counts.items())),
            "selected_worker_rank_sources": sorted(selected_sources),
            "structurally_validated_worker_rank_sources": structurally_validated_sources,
            "selected_rank_local_full_tokens": {
                "semantics": (
                    "scheduler #full token for the exact selected-batch "
                    "rank-local step"
                ),
                "samples": selected_full_tokens,
                "min": min(selected_full_tokens),
                "median": statistics.median(selected_full_tokens),
                "max": max(selected_full_tokens),
            },
            "selected_mean_full_tokens_per_request": {
                "semantics": "rank-local #full token divided by actual running requests",
                "samples": selected_mean_tokens,
                "min": min(selected_mean_tokens),
                "median": statistics.median(selected_mean_tokens),
                "max": max(selected_mean_tokens),
            },
            "accepted_length": {
                "control": "SGLANG_SIMULATE_ACC_LEN=4.80",
                "evidence": "scheduler decode-batch log for every captured source/iteration",
                "histogram": {
                    str(value): count
                    for value, count in sorted(Counter(selected_acceptance).items())
                },
                "mean": measured_acceptance_mean,
                "min": min(selected_acceptance),
                "median": statistics.median(selected_acceptance),
                "max": max(selected_acceptance),
            },
            "queue_requests": {
                "semantics": (
                    "rank-local scheduler waiting queue under the production "
                    "C704 saturation load; recorded but outside the BS32 CUDA "
                    "Graph input shape"
                ),
                "samples": selected_queue_requests,
                "min": min(selected_queue_requests),
                "median": statistics.median(selected_queue_requests),
                "max": max(selected_queue_requests),
            },
            "retracted_requests": 0,
        },
        "profiler": {
            "type": "nsight_systems_worker_local",
            "rank": (
                "all four DEP ranks on both decode workers; synchronized 64-step "
                "raw windows with 32 balanced rank-local BS32 samples selected"
            ),
            "trace": ["cuda", "nvtx"],
            "capture_trigger": "nvtx",
            "capture_range": capture_range,
            "capture_range_end": capture_range_end,
            "capture_range_api": "torch.cuda.nvtx.range_start/range_end",
            "capture_finalize_gpu_synchronize": True,
            "capture_completion": "natural_scheduler_forward_count_boundary",
            "nvtx_registered_strings_only": False,
            "scheduler_capture_steps": {
                "start_inclusive": capture_start_step,
                "stop_exclusive": capture_stop_step,
            },
            "exact_capture_stop_policy": {
                "rebased_forward_count_width": capture_stop_step
                - capture_start_step,
                "completed_raw_decode_batches_per_rank": raw_capture_steps,
                "selected_rank_local_exact_samples": expected_selected_samples,
                "condition": (
                    "any DP rank reached BS32, all ranks entered and exited one "
                    "synchronized raw window, and only real BS32 CUDA Graph events "
                    "were retained for comparison"
                ),
                "external_stop_required": False,
            },
            "cuda_graph_enabled": True,
            "gpu_metric_semantics": (
                "per-node kernel residency is averaged over one balanced pool of 32 "
                "rank-local BS32 samples; parallel ranks and workers are never summed"
            ),
            "attribution_calibration": "graph-off eager Torch stack",
        },
        "evidence": {
            "job_id": args.job_id,
            "source_commit": SOURCE_COMMIT,
            "runtime_source_commit": RUNTIME_SOURCE_COMMIT,
            "profiling_source_commit": PROFILING_SOURCE_COMMIT,
            "profiling_overlay_commit": PROFILING_OVERLAY_COMMIT,
            "profiling_harness_commit": SRT_SLURM_CAPTURE_COMMIT,
            "profiler_manager_sha256": PROFILER_MANAGER_SHA256,
            "scheduler_sha256": SCHEDULER_SHA256,
            "scheduler_nvtx_sha256": SCHEDULER_NVTX_SHA256,
            "runtime_manifest_sha256": RUNTIME_MANIFEST_SHA256,
            "symm_mem_gather_sha256": SYMM_MEM_GATHER_SHA256,
            "model_revision": MODEL_REVISION,
            "model_config_sha256": MODEL_CONFIG_SHA256,
            "container_sha256": CONTAINER_SHA256,
            "config_file": args.config.name,
            "config_sha256": sha256_file(args.config),
            "job_metadata_file": args.job_metadata.name,
            "job_metadata_sha256": sha256_file(args.job_metadata),
            "benchmark_log_file": args.benchmark_log.name,
            "benchmark_log_sha256": sha256_file(args.benchmark_log),
            "benchmark_snapshot": benchmark,
            "comparison_workload": workload_evidence,
            "fingerprints": fingerprint_rows,
            "report_files": [
                {
                    "worker": worker,
                    "file": path.name,
                    "sha256": sha256_file(path),
                    "nsys_export": nsys_export_metadata[worker],
                }
                for worker, path in sorted(reports.items())
            ],
            "nsys_export": nsys_export_metadata[min(nsys_export_metadata)],
            "nsys_report_files": [
                {
                    "worker": worker,
                    "file": path.name,
                    "sha256": sha256_file(path),
                }
                for worker, path in sorted(nsys_reports.items())
            ],
            "worker_logs": [
                {
                    "worker": worker_identity(path),
                    "file": path.name,
                    "sha256": sha256_file(path),
                }
                for path in sorted(args.worker_logs, key=worker_identity)
            ],
            "eager_mapping_file": args.eager_mapping.name,
            "eager_mapping_sha256": sha256_file(args.eager_mapping),
            "contextual_eager_trace": {
                "file": args.eager_trace.name,
                "sha256": sha256_file(args.eager_trace),
                "validated_repeated_slots": len(contextual_signatures),
            },
            "mapping_policy": (
                "synchronized any-rank BS32 raw gate + exact-event filtering + "
                "32 selected scheduler boundaries + "
                "graphId/nodeId occurrence + exact GGGA/MTP5 order + eager stack "
                "leaf calibration"
            ),
            "selection_policy": (
                f"exactly 32 real running_requests={selected_batch} CUDA Graph "
                "rank-local steps, source-balanced and time-spread over the raw window"
            ),
            "mapped_or_fusion_duration_ratio": round(attributed_residency_ratio, 6),
            "mapped_or_fusion_active_union_ratio": round(attributed_active_ratio, 6),
            "strict_signature_duration_ratio": round(strict_signature_us / total_us, 6),
            "mapped_duration_ratio": round(status_us["mapped"] / total_us, 6),
            "fusion_duration_ratio": round(status_us["fusion"] / total_us, 6),
            "unmapped_duration_ratio": round(status_us["unmapped"] / total_us, 6),
            "timeline_interval_coverage_ratio": round(sum(status_us.values()) / total_us, 6),
            "semantic_attribution_gate": {
                "metric": "mapped_or_fusion_active_union_ratio",
                "threshold": 0.95,
                "passed": attributed_active_ratio >= 0.95,
            },
            "critical_gpu_span_ms": timing_summary["critical_gpu_span_ms"],
            "critical_logical_period_ms": timing_summary["critical_logical_period_ms"],
            "instrumented_worker_rank_sources": structurally_validated_sources,
            "all_rank_capture_integrity": all_rank_integrity,
            "four_rank_validation": True,
            "worker_count": 2,
        },
        "timeline": {},
        "node_states": SGLANG_DECODE_NODE_STATES,
        "node_metrics": node_metrics,
    }
    analysis = {
        "profile_id": profile_id,
        "run_identity": {"job_id": args.job_id, "name": config.get("name")},
        "source_validation": source_validation,
        "all_rank_capture_integrity": all_rank_integrity,
        "selected_observations": selected_observations,
        "timing_summary": timing_summary,
        "status_duration_us": dict(status_us),
        "mapped_or_fusion_duration_ratio": attributed_residency_ratio,
        "mapped_or_fusion_active_union_ratio": attributed_active_ratio,
        "strict_signature_duration_ratio": strict_signature_us / total_us,
        "node_metrics": node_metrics,
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
        "reference_worker": timeline["raw_trace"]["worker"],
        "reference_rank": timeline["reference_rank"],
        "step_count": len(timeline["steps"]),
        "event_count": sum(len(step["events"]) for step in timeline["steps"]),
        "raw_trace_file": timeline["raw_trace"]["file"],
    }
    args.output_profile.parent.mkdir(parents=True, exist_ok=True)
    args.output_profile.write_text(
        yaml.safe_dump(profile, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    args.output_analysis.parent.mkdir(parents=True, exist_ok=True)
    args.output_analysis.write_text(json.dumps(analysis, indent=2) + "\n")
    args.output_mapping.parent.mkdir(parents=True, exist_ok=True)
    with args.output_mapping.open("w") as output:
        for row in mappings:
            output.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(f"wrote {args.output_profile.resolve()}")
    print(
        f"exact BS{profile['workload']['selected_exact_target_verify_batch']} "
        f"samples={profile['workload']['selected_samples']} "
        "attributed-active="
        f"{profile['evidence']['mapped_or_fusion_active_union_ratio']:.3f} "
        "attributed-residency="
        f"{profile['evidence']['mapped_or_fusion_duration_ratio']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
