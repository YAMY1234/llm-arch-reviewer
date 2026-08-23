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
    validate_run_inputs,
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
from models.qwen35.profile.qwen35_timeline import QWEN35_TIMELINE_TARGETS


DEFAULT_SELECTED_BATCH = 32
PROFILING_SOURCE_COMMIT = "9d7f6d73b632076002329cd7c19dac5af9c6f76b"
PROFILING_OVERLAY_COMMIT = "29e068d852a789a297da9cb53376fdeeca6a336c"
SRT_SLURM_CAPTURE_COMMIT = "ebf9b696269c484713bd25b58feead000ca120d1"
PROFILER_MANAGER_SHA256 = (
    "a2047ca7d49e1b1adf47f1b92e820ebd4b9fdb6825c96b615ea936ceac460657"
)
SCHEDULER_NVTX_SHA256 = (
    "56610ee61c53c39e40fdd6b44c7443140eeb6e25bc499889e70f93a33bf3fcdd"
)
RUNTIME_MANIFEST_SHA256 = (
    "8c4d28ff9a142151276ed04c61536eb35782941fa8e4ea8028313803aee2f974"
)
SYMM_MEM_GATHER_SHA256 = (
    "8a1f8e9a1f13c26b89691eb0dc7bec07595b107778f180d1afa0a93d5e8af9c4"
)
SOURCE_RE = re.compile(r"^w(?P<worker>[01])/r(?P<rank>[0-3])$")
EXACT_GATE_RE = re.compile(
    r"DP(?P<rank>\d+) TP\d+ EP\d+\] Exact running-batch Nsight gate matched: "
    r"batch=(?P<batch>\d+) forward_ct=(?P<forward_ct>\d+)"
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


def parse_exact_batch_capture_observation(
    path: Path, *, selected_batch: int
) -> dict[str, Any]:
    """Prove that the first captured DP0 forward is exact-batch steady decode."""

    lines = path.read_text(errors="replace").splitlines()
    gates = [
        (index, match)
        for index, line in enumerate(lines)
        if (match := EXACT_GATE_RE.search(line)) is not None
        and int(match.group("rank")) == 0
    ]
    if len(gates) != 1:
        raise ValueError(
            f"{path}: expected one DP0 exact-batch Nsight gate, found {len(gates)}"
        )
    gate_index, gate = gates[0]
    gate_batch = int(gate.group("batch"))
    if gate_batch != selected_batch:
        raise ValueError(
            f"{path}: exact-batch gate selected BS{gate_batch}, expected BS{selected_batch}"
        )

    start = next(
        (
            index
            for index, line in enumerate(lines[gate_index + 1 :], gate_index + 1)
            if "DP0 TP0 EP0] Profiling starts." in line
        ),
        None,
    )
    if start is None:
        raise ValueError(f"{path}: exact-batch gate is not followed by DP0 profiler start")
    stop = next(
        (
            index
            for index, line in enumerate(lines[start + 1 :], start + 1)
            if "DP0 TP0 EP0] Stop profiling..." in line
        ),
        None,
    )
    if stop is None:
        raise ValueError(f"{path}: missing DP0 profiler stop marker")
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

    observations = []
    all_rank_observations = []
    for line in lines[start + 1 : stop]:
        match = LOG_ROW_RE.search(line)
        if match is None:
            continue
        observation = {
            "dp_rank": int(match.group("rank")),
            "scheduler_step": int(match.group("step")),
            "running_requests": int(match.group("running")),
            "full_tokens": int(match.group("full_tokens")),
            "accepted_length": float(match.group("accept")),
            "retracted_requests": int(match.group("retracted")),
            "cuda_graph": match.group("graph") == "True",
            "queued_requests": int(match.group("queue")),
        }
        all_rank_observations.append(observation)
        if observation["dp_rank"] == 0:
            observations.append(observation)
    if not observations:
        raise ValueError(f"{path}: no DP0 decode row inside the Nsight capture")
    gate_forward_ct = int(gate.group("forward_ct"))
    exact_rows = [
        row for row in observations if row["scheduler_step"] == gate_forward_ct
    ]
    if len(exact_rows) != 1:
        raise ValueError(
            f"{path}: expected one DP0 decode row for exact gate forward_ct "
            f"{gate_forward_ct}, found {len(exact_rows)}"
        )
    exact = exact_rows[0]
    if int(exact["running_requests"]) != selected_batch:
        raise ValueError(
            f"{path}: DP0 exact-gate decode row is BS{exact['running_requests']}, "
            f"expected exact BS{selected_batch}"
        )
    if (
        not exact["cuda_graph"]
        or exact["queued_requests"]
        or exact["retracted_requests"]
    ):
        raise ValueError(
            f"{path}: exact-gate decode row is not queue/retraction-free CUDA Graph steady state"
        )
    peer_rows = [
        row
        for row in all_rank_observations
        if row["scheduler_step"] == gate_forward_ct
    ]
    peer_ranks = {int(row["dp_rank"]) for row in peer_rows}
    if peer_ranks != {0, 1, 2, 3} or len(peer_rows) != 4:
        raise ValueError(
            f"{path}: exact-gate step {gate_forward_ct} does not have one row "
            f"for every DEP rank: ranks={sorted(peer_ranks)}, rows={len(peer_rows)}"
        )
    return {
        **exact,
        "gate_forward_ct": gate_forward_ct,
        "logged_rows_before_exact": sum(
            row["scheduler_step"] < gate_forward_ct for row in observations
        ),
        "capture_dp0_observation_count": len(observations),
        "rank_local_batches_at_exact_step": {
            f"r{row['dp_rank']}": {
                "running_requests": row["running_requests"],
                "full_tokens": row["full_tokens"],
                "accepted_length": row["accepted_length"],
                "queued_requests": row["queued_requests"],
                "retracted_requests": row["retracted_requests"],
                "cuda_graph": row["cuda_graph"],
            }
            for row in sorted(peer_rows, key=lambda item: int(item["dp_rank"]))
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
    run = validate_run_inputs(
        args,
        expected_job_id=args.job_id,
        expected_runtime_source=PROFILING_SOURCE_COMMIT,
    )
    config = run["config"]
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
    exact_observations = {
        _report_worker(path): parse_exact_batch_capture_observation(
            path, selected_batch=selected_batch
        )
        for path in args.worker_logs
    }
    if set(exact_observations) != {0, 1}:
        raise ValueError(f"incomplete SGLang worker logs: {sorted(exact_observations)}")

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
    timing_by_forward: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)

    for worker, path in sorted(reports.items()):
        rank = 0
        source = f"w{worker}/r{rank}"
        steps, parser_evidence = load_sglang_nsys_steps(
            path,
            rank=rank,
            capture_range_label=capture_range,
        )
        if not steps[0].label.endswith(":first_exact_step"):
            raise ValueError(f"{source}: parser did not recover the first capture step")
        graph_stability = validate_sglang_graph_node_stability(steps)
        row = exact_observations[worker]
        step = steps[0]

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
            event["event_id"] = f"w{worker}-{event['event_id']}"
            event["worker"] = worker
            event["scheduler_step"] = row["scheduler_step"]
            event["gate_forward_ct"] = row["gate_forward_ct"]
        timing = _timing(mapped, logical_period_us=step.cpu_wall_us)
        validation_row = {
            **validation,
            **timing,
            "graph_roles": graph_roles,
            "scheduler_step": row["scheduler_step"],
            "gate_forward_ct": row["gate_forward_ct"],
        }
        selected_observations.append(
            {
                "worker": worker,
                "rank": rank,
                "scheduler_step": row["scheduler_step"],
                "gate_forward_ct": row["gate_forward_ct"],
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
        timing_by_forward[(worker, int(row["gate_forward_ct"]))].append(
            {"source": source, **timing}
        )
        selected_steps_by_source[source].append(
            {
                "step_index": 0,
                "trace_start_us": min(float(event["ts_us"]) for event in mapped),
                "timing": timing,
                "mapped": mapped,
            }
        )
        source_metrics[source] = _metrics_for_rank(mapped, 1)
        source_selected_counts[source] = 1
        source_validation[source] = {
            "parser": parser_evidence,
            "graph_node_stability": graph_stability,
            "exact_gate_observation": row,
            "captured_batch_distribution": {selected_batch: 1},
            "selected_steps": [validation_row],
        }
        all_mappings.extend(mapped)

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
    reference_source = min(
        selected_sources,
        key=lambda source: (-selected_sources[source], source),
    )
    reference_worker, reference_rank = _source_coordinates(reference_source)
    reference_steps = [
        {
            "step_index": row["step_index"],
            "label": f"AgentX A-Z97 steady decode · NSYS exact BS{selected_batch}",
            "trace_start_us": row["trace_start_us"],
            "duration_us": row["timing"]["gpu_span_us"],
            "events": attach_graph_stack_evidence(
                row["mapped"], mapping_path=args.eager_mapping
            ),
        }
        for row in selected_steps_by_source[reference_source]
    ]

    critical_steps: dict[str, dict[str, Any]] = {}
    for (worker, gate_forward_ct), rows in sorted(timing_by_forward.items()):
        selected = max(rows, key=lambda row: float(row["gpu_span_us"]))
        critical_steps[f"w{worker}/f{gate_forward_ct}"] = selected
    critical_gpu_span_ms = [
        float(row["gpu_span_us"]) / 1000.0 for row in critical_steps.values()
    ]
    logical_period_ms = [
        float(row["logical_step_period_us"]) / 1000.0
        for row in critical_steps.values()
    ]
    timing_summary = {
        "semantics": (
            "the exact first logical period is capture-range start to the next "
            "scheduler.run_batch start; GPU span, active union, and residency "
            "come from graph replays paired to worker-local launches"
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
    selected_full_tokens = [int(row["full_tokens"]) for row in selected_observations]
    selected_mean_tokens = [
        float(row["mean_full_tokens_per_request"])
        for row in selected_observations
    ]
    node_metrics = _aggregate_source_metrics(source_metrics)
    profile = {
        "schema_version": "profile.v2",
        "profile_id": profile_id,
        "label": (
            "Qwen3.5 397B · SGLang · AgentX C704 · DEP4 + MTP6 · "
            f"NSYS exact BS{selected_batch}"
        ),
        "model_id": "qwen35_397b_a17b",
        "execution_path_id": "attention_dp4_moe_ep4",
        "implementation_id": "sglang_85c23c62_attention_dp4_moe_ep4_mtp",
        "variant_id": (
            "sglang_agentx_a_z97_c704_3p2d_dep4_mtp6_cg_nsys_"
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
            "scenario": "inferencex-agentx-mvp",
            "rank_distribution": "A-Z97",
            "concurrency": 704,
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
                "samples": selected_acceptance,
                "min": min(selected_acceptance),
                "median": statistics.median(selected_acceptance),
                "max": max(selected_acceptance),
            },
            "queue_requests": 0,
            "retracted_requests": 0,
        },
        "profiler": {
            "type": "nsight_systems_worker_local",
            "rank": (
                "exact-BS32 DP0 timing on both decode workers; all four DEP ranks "
                "validated for CUDA table and scheduler-marker integrity"
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
                "minimum_completed_decode_batches": 2,
                "condition": (
                    "both rebased forward-count width reached and at least two "
                    "real decode batches completed"
                ),
                "external_stop_required": False,
            },
            "cuda_graph_enabled": True,
            "gpu_metric_semantics": (
                "maximum of the two worker-local DP0 samples; parallel workers "
                "and DEP ranks are not summed"
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
                "exact-batch capture gate + first-step capture/scheduler boundary + "
                "graphId/nodeId occurrence + exact GGGA/MTP5 order + eager stack "
                "leaf calibration"
            ),
            "selection_policy": (
                f"first DP0 step after exact running_requests={selected_batch} "
                "capture gate on each decode worker"
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
