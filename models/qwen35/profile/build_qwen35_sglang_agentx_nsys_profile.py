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
    _aggregate_source_metrics,
    _validate_fingerprints,
    parse_benchmark_snapshot,
    parse_worker_profile_observations,
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
    sglang_nsys_trace_events,
    validate_sglang_graph_node_stability,
)
from models.qwen35.profile.qwen35_timeline import QWEN35_TIMELINE_TARGETS


DEFAULT_SELECTED_BATCH = 32
PROFILING_OVERLAY_COMMIT = "45764d936f022c73b86cda215b2acdc346516a3f"
SRT_SLURM_CAPTURE_COMMIT = "227fdf5f2850df2ef4dcb068ac6f09f7c623e61a"
PROFILER_MANAGER_SHA256 = (
    "02e19720a334a1184c5bf3ce9ec32ca097e8cac865d89225a056a51f69761d4e"
)
SOURCE_RE = re.compile(r"^w(?P<worker>[01])/r(?P<rank>[0-3])$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlites", type=Path, nargs=2, required=True)
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


def _validate_nsys_capture_contract(
    profiling: dict[str, Any], decode_environment: dict[str, Any]
) -> str:
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
    for index, arg in enumerate(extra_args):
        if arg in {"-c", "--capture-range"} and index + 1 < len(extra_args):
            capture_mode = extra_args[index + 1]
        elif arg.startswith("-c=") or arg.startswith("--capture-range="):
            capture_mode = arg.split("=", 1)[1]
        if arg in {"-p", "--nvtx-capture"} and index + 1 < len(extra_args):
            capture_selector = extra_args[index + 1]
        elif arg.startswith("-p=") or arg.startswith("--nvtx-capture="):
            capture_selector = arg.split("=", 1)[1]

    if capture_mode != "nvtx" or capture_selector != capture_range:
        raise ValueError(
            "formal SGLang NSYS profile requires matching '-c nvtx' and "
            "NVTX capture-range selector arguments"
        )
    if "cudaProfilerApi" in extra_args:
        raise ValueError("formal SGLang NSYS profile cannot use cudaProfilerApi")
    return capture_range


def _align_steps_and_logs(
    *,
    source: str,
    steps: list[Any],
    observations: list[dict[str, Any]],
) -> list[tuple[Any, dict[str, Any]]]:
    """Align two independently captured, exact-order step sequences.

    The formal recipe emits one rank-local log row per forward and one
    ``scheduler.run_batch`` NVTX range per forward. We deliberately reject an
    edge mismatch instead of guessing an offset, because the batch-size label
    is needed for exact-BS selection.
    """

    if len(steps) != len(observations):
        raise ValueError(
            f"{source}: NSYS/log step mismatch: {len(steps)} graph steps vs "
            f"{len(observations)} scheduler observations"
        )
    aligned = list(zip(steps, observations))
    if any(
        not row["cuda_graph"]
        or row["queued_requests"]
        or row["retracted_requests"]
        for _step, row in aligned
    ):
        raise ValueError(
            f"{source}: selected capture is not queue/retraction-free CUDA Graph steady state"
        )
    return aligned


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
    run = validate_run_inputs(args, expected_job_id=args.job_id)
    config = run["config"]
    profiling = config.get("profiling") or {}
    decode_environment = ((config.get("backend") or {}).get("decode_environment") or {})
    if profiling.get("type") != "nsys":
        raise ValueError("formal SGLang matched profile requires profiling.type=nsys")
    if str(decode_environment.get("SGLANG_ENABLE_NVTX_SCHEDULER")) not in {"1", "true", "True"}:
        raise ValueError("formal SGLang NSYS profile requires scheduler NVTX")
    capture_range = _validate_nsys_capture_contract(profiling, decode_environment)

    eager_signatures = load_unique_eager_kernel_signatures(args.eager_mapping)
    contextual_signatures = load_contextual_eager_signatures(
        trace_path=args.eager_trace.resolve(),
        mapping_path=args.eager_mapping.resolve(),
    )
    fingerprint_rows = _validate_fingerprints(args.fingerprints)
    benchmark = parse_benchmark_snapshot(args.benchmark_log)
    log_observations = {
        _report_worker(path): parse_worker_profile_observations(path)
        for path in args.worker_logs
    }
    if set(log_observations) != {0, 1}:
        raise ValueError(f"incomplete SGLang worker logs: {sorted(log_observations)}")

    reports = {_report_worker(path): path.resolve() for path in args.sqlites}
    if set(reports) != {0, 1}:
        raise ValueError(f"incomplete SGLang NSYS reports: {sorted(reports)}")

    source_metrics: dict[str, dict[str, Any]] = {}
    source_validation: dict[str, dict[str, Any]] = {}
    source_selected_counts: dict[str, int] = {}
    all_mappings: list[dict[str, Any]] = []
    selected_observations: list[dict[str, Any]] = []
    selected_steps_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    timing_by_forward: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)

    for worker, path in sorted(reports.items()):
        rows_by_rank = {
            rank: [row for row in log_observations[worker] if row["dp_rank"] == rank]
            for rank in range(4)
        }
        for rank in range(4):
            source = f"w{worker}/r{rank}"
            steps, parser_evidence = load_sglang_nsys_steps(path, rank=rank)
            graph_stability = validate_sglang_graph_node_stability(steps)
            aligned = _align_steps_and_logs(
                source=source, steps=steps, observations=rows_by_rank[rank]
            )
            selected = [
                (step, row)
                for step, row in aligned
                if int(row["running_requests"]) == selected_batch
            ]

            source_mappings: list[dict[str, Any]] = []
            validations = []
            for selected_index, (step, row) in enumerate(selected):
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
                timing = _timing(mapped, logical_period_us=step.cpu_wall_us)
                source_mappings.extend(mapped)
                validations.append(
                    {
                        **validation,
                        **timing,
                        "graph_roles": graph_roles,
                        "scheduler_step": row["scheduler_step"],
                    }
                )
                selected_observations.append(
                    {
                        "worker": worker,
                        "rank": rank,
                        "scheduler_step": row["scheduler_step"],
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
                timing_by_forward[(worker, int(row["scheduler_step"]))].append(
                    {"source": source, **timing}
                )
                selected_steps_by_source[source].append(
                    {
                        "step_index": selected_index,
                        "trace_start_us": min(float(event["ts_us"]) for event in mapped),
                        "timing": timing,
                        "mapped": mapped,
                    }
                )

            if source_mappings:
                source_metrics[source] = _metrics_for_rank(
                    source_mappings, len(selected)
                )
            source_selected_counts[source] = len(selected)
            source_validation[source] = {
                "parser": parser_evidence,
                "graph_node_stability": graph_stability,
                "captured_batch_distribution": dict(
                    sorted(
                        Counter(
                            int(row["running_requests"])
                            for _step, row in aligned
                        ).items()
                    )
                ),
                "selected_steps": validations,
            }
            all_mappings.extend(source_mappings)

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
    for (worker, scheduler_step), rows in sorted(timing_by_forward.items()):
        selected = max(rows, key=lambda row: float(row["gpu_span_us"]))
        critical_steps[f"w{worker}/s{scheduler_step}"] = selected
    critical_gpu_span_ms = [
        float(row["gpu_span_us"]) / 1000.0 for row in critical_steps.values()
    ]
    logical_period_ms = [
        float(row["logical_step_period_us"]) / 1000.0
        for row in critical_steps.values()
    ]
    timing_summary = {
        "semantics": (
            "logical period is scheduler-start to next scheduler-start; GPU span, "
            "active union, and residency come from graph replays paired to launches"
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
            "structurally_validated_worker_rank_sources": sorted(source_validation),
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
            "rank": "both decode workers, all four DEP ranks",
            "trace": ["cuda", "nvtx"],
            "capture_trigger": "nvtx",
            "capture_range": capture_range,
            "cuda_graph_enabled": True,
            "gpu_metric_semantics": (
                "maximum worker/rank residency; parallel workers/ranks are not summed"
            ),
            "attribution_calibration": "graph-off eager Torch stack",
        },
        "evidence": {
            "job_id": args.job_id,
            "source_commit": SOURCE_COMMIT,
            "runtime_source_commit": RUNTIME_SOURCE_COMMIT,
            "profiling_overlay_commit": PROFILING_OVERLAY_COMMIT,
            "profiling_harness_commit": SRT_SLURM_CAPTURE_COMMIT,
            "profiler_manager_sha256": PROFILER_MANAGER_SHA256,
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
                }
                for worker, path in sorted(reports.items())
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
                "scheduler NVTX launch owner + graphId/nodeId occurrence + "
                "exact GGGA/MTP5 order + eager stack leaf calibration"
            ),
            "selection_policy": f"exact running_requests={selected_batch} events only",
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
