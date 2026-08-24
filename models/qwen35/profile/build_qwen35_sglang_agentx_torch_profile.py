#!/usr/bin/env python3
"""Build the strict 8K/1K SGLang AgentX Torch/Kineto decode profile."""

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

from models.common.timeline_artifact import (
    build_timeline_artifact,
    write_timeline_artifact,
)
from models.common.trace_mapping import find_eagle_mtp_decode_windows, load_trace
from models.qwen35.profile.build_qwen35_sglang_agentx_nsys_profile import (
    _timing,
    parse_exact_batch_capture_observations,
)
from models.qwen35.profile.build_qwen35_sglang_agentx_profile import (
    _validate_fingerprints,
    worker_identity,
)
from models.qwen35.profile.build_qwen35_sglang_decode_profile import (
    CONTAINER_SHA256,
    MODEL_CONFIG_SHA256,
    MODEL_REVISION,
    SGLANG_DECODE_NODE_STATES,
    _metrics_for_rank,
    _validate_step_signatures,
    sha256_file,
)
from models.qwen35.profile.qwen35_a2a_contract import validate_comparison_workload
from models.qwen35.profile.qwen35_graph_mapping import (
    attach_graph_stack_evidence,
    attribution_active_union_ratio,
    load_contextual_eager_signatures,
    load_unique_eager_kernel_signatures,
    map_graph_window,
)
from models.qwen35.profile.qwen35_timeline import QWEN35_TIMELINE_TARGETS


PROFILE_ID = "qwen35_sglang_attention_dp4_moe_ep4_mtp6_agentx_torch_bs32"
PROFILING_SOURCE_COMMIT = "049323294ec36631b9aab74ffa5dac5ff020fdae"
SRT_SLURM_CAPTURE_COMMIT = "fb4476ed42399ca6a160565e1e6a7bb864b9a015"
PROFILER_MANAGER_SHA256 = (
    "e08fd6430c83bffbc47e83cdd8a770891c8208b3fff6126627bc231d8e338eb9"
)
RUNTIME_MANIFEST_SHA256 = (
    "f23af95628fe593cc6d1c140bc3c4a040ddbd224b94c38a7fdaa9b2c8fa46d41"
)
TRACE_RE = re.compile(
    r"^(?P<profile>.+)-TP-(?P<tp>[0-3])-DP-(?P<dp>[0-3])-EP-(?P<ep>[0-3])"
    r"\.trace\.json\.gz$"
)
PROFILE_ID_RE = re.compile(r"with profile id: (?P<profile>[^)]+)\)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, nargs=8, required=True)
    parser.add_argument("--worker-logs", type=Path, nargs=2, required=True)
    parser.add_argument("--fingerprints", type=Path, nargs=2, required=True)
    parser.add_argument("--benchmark-log", type=Path, required=True)
    parser.add_argument("--job-metadata", type=Path, required=True)
    parser.add_argument("--job-id", type=int, required=True)
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


def _worker_profile_id(path: Path) -> str:
    profile_ids = {
        match.group("profile")
        for line in path.read_text(errors="replace").splitlines()
        if (match := PROFILE_ID_RE.search(line)) is not None
    }
    if len(profile_ids) != 1:
        raise ValueError(f"{path}: expected one Torch profile id, got {profile_ids}")
    return next(iter(profile_ids))


def _trace_rank(path: Path) -> tuple[str, int]:
    match = TRACE_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"cannot parse SGLang Torch trace name {path.name}")
    ranks = {
        int(match.group("tp")),
        int(match.group("dp")),
        int(match.group("ep")),
    }
    if len(ranks) != 1:
        raise ValueError(f"{path}: TP/DP/EP ranks are not aligned")
    return match.group("profile"), next(iter(ranks))


def _spread(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if len(rows) < count:
        raise ValueError(f"cannot select {count} time-spread rows from {len(rows)}")
    indices = [((2 * index + 1) * len(rows)) // (2 * count) for index in range(count)]
    return [rows[index] for index in indices]


def select_balanced_al48_observations(
    observations_by_worker: dict[int, dict[int, dict[str, Any]]],
    *,
    selected_batch: int = 32,
) -> list[dict[str, Any]]:
    """Select five rows/worker with an exact 8xAL5 + 2xAL4 histogram."""

    selected: list[dict[str, Any]] = []
    for worker, ranks in sorted(observations_by_worker.items()):
        if len(ranks) != 1:
            raise ValueError(f"worker {worker}: expected one elected rank, got {ranks}")
        rank, evidence = next(iter(ranks.items()))
        candidates = [
            {
                **row,
                "worker": worker,
                "rank": rank,
                "source": f"w{worker}/r{rank}",
                "capture_iteration": iteration,
            }
            for iteration, row in enumerate(evidence["observations"])
            if row["running_requests"] == selected_batch
            and row["cuda_graph"]
            and not row["retracted_requests"]
        ]
        by_accept = defaultdict(list)
        for row in candidates:
            by_accept[float(row["accepted_length"])].append(row)
        selected.extend(_spread(by_accept[4.0], 1))
        selected.extend(_spread(by_accept[5.0], 4))

    selected.sort(
        key=lambda row: (
            int(row["capture_iteration"]),
            int(row["worker"]),
            int(row["rank"]),
        )
    )
    for sample_index, row in enumerate(selected):
        row["sample_index"] = sample_index
    histogram = Counter(float(row["accepted_length"]) for row in selected)
    if histogram != Counter({5.0: 8, 4.0: 2}):
        raise ValueError(f"selected acceptance histogram is not mean 4.8: {histogram}")
    if Counter(row["source"] for row in selected) != Counter(
        {f"w{worker}/r{next(iter(ranks))}": 5 for worker, ranks in observations_by_worker.items()}
    ):
        raise ValueError("selected rows are not balanced five per decode worker")
    return selected


def _summary(values: list[float]) -> dict[str, Any]:
    return {
        "samples": [round(value, 6) for value in values],
        "mean": round(statistics.fmean(values), 6),
        "median": round(statistics.median(values), 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def _benchmark_snapshot(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text())
    errors = [error for error in (result.get("errors") or []) if error]
    if result.get("completed") != 704 or errors:
        raise ValueError(f"{path}: exact workload did not complete cleanly")
    return {
        "completed": int(result["completed"]),
        "errors": 0,
        "duration_s": float(result["duration"]),
        "request_throughput": float(result["request_throughput"]),
        "output_throughput": float(result["output_throughput"]),
    }


def build(args: argparse.Namespace):
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
            f"job metadata ID {job.get('job_id')} does not match {args.job_id}"
        )
    frameworks = (config.get("identity") or {}).get("frameworks") or {}
    if frameworks.get("sglang_source") != PROFILING_SOURCE_COMMIT:
        raise ValueError("strict SGLang Torch source commit mismatch")
    if frameworks.get("srt_slurm_source") != SRT_SLURM_CAPTURE_COMMIT:
        raise ValueError("strict SGLang Torch harness commit mismatch")
    profiling = config.get("profiling") or {}
    if profiling.get("type") != "torch":
        raise ValueError("strict same-profiler SGLang profile requires Torch/Kineto")
    if profiling.get("sglang_scheduler_nsys") is not None:
        raise ValueError("Torch/Kineto run must not use an outer Nsight wrapper")

    eager_signatures = load_unique_eager_kernel_signatures(args.eager_mapping)
    contextual_signatures = load_contextual_eager_signatures(
        trace_path=args.eager_trace.resolve(),
        mapping_path=args.eager_mapping.resolve(),
    )
    fingerprint_rows = _validate_fingerprints(args.fingerprints)
    benchmark = _benchmark_snapshot(args.workload_result)

    worker_logs = {worker_identity(path): path.resolve() for path in args.worker_logs}
    if set(worker_logs) != {0, 1}:
        raise ValueError(f"incomplete SGLang worker logs: {worker_logs}")
    profile_to_worker = {_worker_profile_id(path): worker for worker, path in worker_logs.items()}
    if len(profile_to_worker) != 2:
        raise ValueError(f"worker Torch profile IDs collide: {profile_to_worker}")

    exact_observations = {
        worker: parse_exact_batch_capture_observations(
            path,
            selected_batch=32,
            expected_steps=64,
            expected_warmup_batches=1,
            expected_gate_reduction=None,
            expected_gate_ranks=None,
        )
        for worker, path in worker_logs.items()
    }
    if any(len(ranks) != 1 for ranks in exact_observations.values()):
        raise ValueError("each SGLang worker must elect exactly one representative rank")
    selected_rank_by_worker = {
        worker: next(iter(ranks)) for worker, ranks in exact_observations.items()
    }
    if any(
        evidence["gate_reduction"] != "auto"
        for ranks in exact_observations.values()
        for evidence in ranks.values()
    ):
        raise ValueError("SGLang representative ranks were not runtime auto-elected")
    selected_rows = select_balanced_al48_observations(exact_observations)
    selected_by_key = {
        (int(row["worker"]), int(row["rank"]), int(row["capture_iteration"])): row
        for row in selected_rows
    }

    paths: dict[tuple[int, int], Path] = {}
    for raw_path in args.traces:
        path = raw_path.resolve()
        profile_id, rank = _trace_rank(path)
        if profile_id not in profile_to_worker:
            raise ValueError(f"{path}: profile id is absent from decode-worker logs")
        identity = profile_to_worker[profile_id], rank
        if identity in paths:
            raise ValueError(f"duplicate SGLang Torch trace for {identity}")
        paths[identity] = path
    expected_paths = {(worker, rank) for worker in range(2) for rank in range(4)}
    if set(paths) != expected_paths:
        raise ValueError(f"incomplete two-worker/four-rank traces: {sorted(paths)}")

    trace_cache: dict[tuple[int, int], tuple[list[dict[str, Any]], list[Any]]] = {}
    trace_checks = {}
    for identity, path in sorted(paths.items()):
        trace_events = load_trace(path).get("traceEvents") or []
        windows = find_eagle_mtp_decode_windows(
            trace_events, signature="fused_qkvzba_split"
        )
        kernel_names = [
            str(event.get("name"))
            for event in trace_events
            if event.get("cat") == "kernel"
        ]
        if len(windows) != 64 or not kernel_names:
            raise ValueError(
                f"{path}: expected 64 complete MTP windows and CUDA kernels; "
                f"got windows={len(windows)} kernels={len(kernel_names)}"
            )
        trace_cache[identity] = trace_events, windows
        trace_checks[f"w{identity[0]}/r{identity[1]}"] = {
            "file": path.name,
            "sha256": sha256_file(path),
            "event_count": len(trace_events),
            "kernel_count": len(kernel_names),
            "concrete_kernel_name_count": len(set(kernel_names)),
            "complete_mtp_window_count": len(windows),
        }

    all_mappings: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    timings: list[dict[str, Any]] = []
    selected_steps_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected_rows:
        worker = int(row["worker"])
        rank = int(row["rank"])
        iteration = int(row["capture_iteration"])
        sample_index = int(row["sample_index"])
        trace_events, windows = trace_cache[(worker, rank)]
        window = windows[iteration]
        mapped, validation = map_graph_window(
            trace_events,
            window=window,
            rank=rank,
            step_index=iteration,
            eager_signatures=eager_signatures,
            contextual_signatures=contextual_signatures,
        )
        _validate_step_signatures(validation, rank=rank, step=iteration)
        if int(validation["target_verify_batch_size"]) != 32:
            raise ValueError(
                f"w{worker}/r{rank}/c{iteration}: trace target batch is not BS32"
            )
        for event in mapped:
            event["event_id"] = f"w{worker}-r{rank}-{event['event_id']}"
            event["worker"] = worker
            event["capture_iteration"] = iteration
            event["selected_sample_index"] = sample_index
        timing = _timing(
            mapped, logical_period_us=window.end_us - window.start_us
        )
        validation_row = {
            **validation,
            **timing,
            "worker": worker,
            "rank": rank,
            "source": row["source"],
            "capture_iteration": iteration,
            "selected_sample_index": sample_index,
            "running_requests": row["running_requests"],
            "accepted_length": row["accepted_length"],
            "full_tokens": row["full_tokens"],
        }
        validations.append(validation_row)
        timings.append(validation_row)
        selected_steps_by_source[row["source"]].append(
            {
                "step_index": sample_index,
                "label": "Exact 8K/1K steady decode · Torch/Kineto BS32",
                "trace_start_us": window.start_us,
                "duration_us": window.end_us - window.start_us,
                "events": mapped,
            }
        )
        all_mappings.extend(mapped)

    sample_count = int(comparison_contract["selected_rank_local_samples"])
    if len(timings) != sample_count:
        raise ValueError(f"selected {len(timings)} samples, expected {sample_count}")
    counts_by_source = Counter(row["source"] for row in selected_rows)
    node_metrics = _metrics_for_rank(all_mappings, sample_count)
    for cell in node_metrics.values():
        cell["ms_per_iter"] = round(float(cell["ms_per_iter"]), 6)
        cell["aggregation"] = (
            "mean kernel residency over 10 acceptance-balanced rank-local BS32 samples"
        )
        cell["source_worker_rank"] = "balanced_pool"
        cell["selected_samples_by_source"] = dict(sorted(counts_by_source.items()))

    total_us = sum(float(event["dur_us"]) for event in all_mappings)
    status_us: Counter[str] = Counter()
    for event in all_mappings:
        status_us[str(event["mapping_status"])] += float(event["dur_us"])
    attributed_residency_ratio = (
        status_us["mapped"] + status_us["fusion"]
    ) / total_us
    attributed_active_ratio = attribution_active_union_ratio(all_mappings)
    strict_signature_us = sum(
        float(event["dur_us"])
        for event in all_mappings
        if event.get("attribution_method") == "unique_kernel_signature"
    )

    wall_ms = [float(row["logical_step_period_us"]) / 1000.0 for row in timings]
    span_ms = [float(row["gpu_span_us"]) / 1000.0 for row in timings]
    active_ms = [float(row["gpu_busy_union_us"]) / 1000.0 for row in timings]
    residency_ms = [float(row["gpu_residency_us"]) / 1000.0 for row in timings]
    timing_summary = {
        "semantics": (
            "each sample is one real rank-local BS32 CUDA Graph period; the pool "
            "contains five time-spread samples from one runtime-elected rank on "
            "each decode worker; ranks and workers are never summed"
        ),
        "critical_steps": {
            str(row["selected_sample_index"]): {
                "source_worker_rank": row["source"],
                "elapsed_wall_us": row["logical_step_period_us"],
                "gpu_span_us": row["gpu_span_us"],
                "active_gpu_us": row["gpu_busy_union_us"],
                "gpu_residency_us": row["gpu_residency_us"],
                "gpu_overlap_us": row["gpu_residency_us"] - row["gpu_busy_union_us"],
                "device_gap_idle_us": row["logical_step_period_us"] - row["gpu_busy_union_us"],
            }
            for row in timings
        },
        "critical_step_wall_ms": _summary(wall_ms),
        "critical_gpu_span_ms": _summary(span_ms),
        "critical_active_gpu_ms": _summary(active_ms),
        "critical_gpu_residency_ms": _summary(residency_ms),
    }

    reference_source = sorted(selected_steps_by_source)[0]
    reference_worker = int(reference_source[1])
    reference_rank = int(reference_source[4])
    reference_path = paths[(reference_worker, reference_rank)]
    reference_steps = []
    for step in selected_steps_by_source[reference_source]:
        reference_steps.append(
            {
                **step,
                "events": attach_graph_stack_evidence(
                    step["events"], mapping_path=args.eager_mapping
                ),
            }
        )
    timeline = build_timeline_artifact(
        profile_id=PROFILE_ID,
        phase="decode",
        reference_rank=reference_rank,
        steps=reference_steps,
        timing_summary=timing_summary,
        raw_trace={
            "file": reference_path.name,
            "sha256": sha256_file(reference_path),
            "format": "PyTorch profiler trace JSON gzip",
            "worker": reference_worker,
            "rank": reference_rank,
        },
        stack_source={
            "mode": "kineto_gpu_annotation_and_eager_stack_transfer",
            "file": args.eager_trace.name,
            "sha256": sha256_file(args.eager_trace),
            "mapped_residency_ratio": round(attributed_residency_ratio, 6),
            "mapped_active_union_ratio": round(attributed_active_ratio, 6),
            "policy": (
                "unique eager kernel signatures map leaves; unresolved graph "
                "occurrences remain explicit unmapped events with candidates"
            ),
        },
        target_resolver=QWEN35_TIMELINE_TARGETS,
    )

    raw_acceptance = Counter(
        float(row["accepted_length"])
        for ranks in exact_observations.values()
        for evidence in ranks.values()
        for row in evidence["observations"]
        if row["running_requests"] == 32
        and row["cuda_graph"]
        and not row["retracted_requests"]
    )
    selected_acceptance = Counter(float(row["accepted_length"]) for row in selected_rows)
    profile = {
        "schema_version": "profile.v2",
        "profile_id": PROFILE_ID,
        "label": (
            "Qwen3.5 397B · SGLang · exact 8K/1K C704 · DEP4 + MTP6 · "
            "Torch/Kineto 10×BS32 decode"
        ),
        "model_id": "qwen35_397b_a17b",
        "execution_path_id": "attention_dp4_moe_ep4",
        "implementation_id": "sglang_85c23c62_attention_dp4_moe_ep4_mtp",
        "variant_id": "sglang_agentx_8k1k_c704_3p2d_dep4_mtp6_cg_torch_bs32",
        "phase": "decode",
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
            "gpus_per_worker": 4,
            "prefill_workers": 3,
            "decode_workers": 2,
        },
        "workload": {
            "suite": "exact-8k1k",
            "concurrency": 704,
            "comparison_contract": comparison_contract,
            "mtp_draft_tokens": 6,
            "selected_exact_target_verify_batch": 32,
            "selected_samples": sample_count,
            "selected_samples_by_source": dict(sorted(counts_by_source.items())),
            "selected_worker_rank_sources": sorted(counts_by_source),
            "accepted_length": {
                "control": "SGLANG_SIMULATE_ACC_LEN=4.80",
                "requested_mean": 4.8,
                "selected_samples": [float(row["accepted_length"]) for row in selected_rows],
                "selected_histogram": dict(sorted(selected_acceptance.items())),
                "selected_mean": statistics.fmean(
                    float(row["accepted_length"]) for row in selected_rows
                ),
                "raw_exact_bs32_histogram": dict(sorted(raw_acceptance.items())),
                "selection_policy": (
                    "one AL4 plus four AL5 time-spread samples per worker; exact "
                    "10-sample mean 4.8"
                ),
            },
            "selected_rank_local_full_tokens": _summary(
                [float(row["full_tokens"]) for row in selected_rows]
            ),
            "selected_queue_requests": _summary(
                [float(row["queued_requests"]) for row in selected_rows]
            ),
            "retracted_requests": 0,
        },
        "profiler": {
            "type": "torch_kineto_worker_local",
            "rank": (
                "one runtime-elected exact-BS32 DP3 rank on each decode worker; "
                "10 complete samples balanced five per worker; all eight rank "
                "traces validated for window and concrete-kernel integrity"
            ),
            "activities": ["CPU", "CUDA"],
            "with_stack": True,
            "record_shapes": False,
            "cuda_graph_enabled": True,
            "gpu_metric_semantics": (
                "per-node kernel residency is averaged over 10 balanced rank-local "
                "BS32 periods; parallel ranks and workers are never summed"
            ),
        },
        "evidence": {
            "job_id": args.job_id,
            "profiling_source_commit": PROFILING_SOURCE_COMMIT,
            "profiling_harness_commit": SRT_SLURM_CAPTURE_COMMIT,
            "profiler_manager_sha256": PROFILER_MANAGER_SHA256,
            "runtime_manifest_sha256": RUNTIME_MANIFEST_SHA256,
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
            "trace_files": [
                {
                    "worker": worker,
                    "rank": rank,
                    **trace_checks[f"w{worker}/r{rank}"],
                }
                for worker, rank in sorted(paths)
            ],
            "worker_logs": [
                {
                    "worker": worker,
                    "file": path.name,
                    "sha256": sha256_file(path),
                    "profile_id": _worker_profile_id(path),
                    "selected_rank": selected_rank_by_worker[worker],
                }
                for worker, path in sorted(worker_logs.items())
            ],
            "comparison_workload": workload_evidence,
            "eager_mapping_file": args.eager_mapping.name,
            "eager_mapping_sha256": sha256_file(args.eager_mapping),
            "contextual_eager_trace": {
                "file": args.eager_trace.name,
                "sha256": sha256_file(args.eager_trace),
            },
            "mapping_policy": (
                "Kineto GPU step annotations + exact GGGA/MTP6 order + conservative "
                "eager stack/kernel signature transfer"
            ),
            "selection_policy": (
                "10 real rank-local BS32 graph periods: five time-spread samples "
                "per runtime-elected worker source and exact AL4.8 histogram"
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
                "threshold": 0.95,
                "passed": attributed_active_ratio >= 0.95,
            },
            "critical_step_wall_ms": timing_summary["critical_step_wall_ms"],
            "critical_gpu_span_ms": timing_summary["critical_gpu_span_ms"],
            "all_eight_rank_traces_validated": True,
        },
        "timeline": {},
        "node_states": SGLANG_DECODE_NODE_STATES,
        "node_metrics": node_metrics,
    }
    analysis = {
        "profile_id": PROFILE_ID,
        "trace_checks": trace_checks,
        "exact_observations": exact_observations,
        "selected_observations": selected_rows,
        "validations": validations,
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
        "wall-mean="
        f"{profile['evidence']['critical_step_wall_ms']['mean']:.3f} ms "
        "attributed-active="
        f"{profile['evidence']['mapped_or_fusion_active_union_ratio']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
