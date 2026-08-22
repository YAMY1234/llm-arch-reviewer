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

from models.common.timeline_artifact import build_timeline_artifact, write_timeline_artifact
from models.qwen35.profile.build_qwen35_sglang_decode_profile import (
    _metrics_for_rank,
    sha256_file,
)
from models.qwen35.profile.qwen35_nsys_mapping import (
    load_nsys_steps,
    map_decode_step,
    map_prefill_step,
)
from models.qwen35.profile.qwen35_timeline import QWEN35_TIMELINE_TARGETS


REPORT_RE = re.compile(
    r"(?P<worker>.+)-(?P<phase>prefill|decode)-rank(?P<rank>[0-3])(?:\.\d+)?\.sqlite$"
)
TRT_COMMIT = "1cef02e901be43081b1ba6d4981e94ed3bd9c1e8"
MODEL_REVISION = "8f590eae8f10bf55d9a46f79ea0280bde435c9f8"
MODEL_CONFIG_SHA256 = "9408a9e559cc2f05f0b357738213666353e6651160ce8ff477b1c26982bc4f63"
CONTAINER_SHA256 = "1cb820b92bd7ab56ab69457500adf3b7f2928bfefe7f2920951fe7286552dcf7"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prefill", "decode"), required=True)
    parser.add_argument("--sqlites", type=Path, nargs="+", required=True)
    parser.add_argument("--job-id", type=int, default=532540)
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
        raise ValueError(f"{path.name}: kernels are not attached to a Python worker process")
    if not nvtx_steps or not kernels:
        raise ValueError(f"{path.name}: missing NVTX steps or CUDA kernels")
    return {"processes": processes, "nvtx_step_count": nvtx_steps, "kernel_count": kernels}


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


def build(args: argparse.Namespace):
    expected_workers = 3 if args.phase == "prefill" else 2
    # The worker-local launcher treats stop_step as exclusive, matching the
    # validated smoke range 10:12 -> steps 10 and 11.
    expected_steps = (
        set(range(10000, 10002))
        if args.phase == "prefill"
        else set(range(60000, 60020))
    )
    paths: dict[tuple[str, int], Path] = {}
    for raw_path in args.sqlites:
        path = raw_path.resolve()
        identity = _report_identity(path, args.phase)
        if identity in paths:
            raise ValueError(f"duplicate report for {identity}: {path}")
        paths[identity] = path
    workers = sorted({worker for worker, _rank in paths})
    if len(workers) != expected_workers:
        raise ValueError(f"expected {expected_workers} {args.phase} workers, got {workers}")
    for worker in workers:
        ranks = {rank for candidate, rank in paths if candidate == worker}
        if ranks != {0, 1, 2, 3}:
            raise ValueError(f"worker {worker} lacks four-rank coverage: {ranks}")

    source_metrics: dict[str, dict[str, Any]] = {}
    validations: dict[str, list[dict[str, Any]]] = {}
    process_checks: dict[str, dict[str, Any]] = {}
    timing_by_step: dict[int, list[dict[str, Any]]] = {}
    all_mappings: list[dict[str, Any]] = []
    reference_source = f"{workers[0]}/rank0"
    reference_steps: list[dict[str, Any]] = []
    observed_steps: list[dict[str, Any]] = []
    owner_rank_positions: set[int] = set()
    shape_observations: list[dict[str, Any]] = []

    for (worker, rank), path in sorted(paths.items()):
        source = f"{worker}/rank{rank}"
        process_checks[source] = _validate_process(path)
        steps = load_nsys_steps(path, rank=rank)
        actual_steps = {step.step_id for step in steps}
        if actual_steps != expected_steps:
            raise ValueError(
                f"{source}: expected steps {min(expected_steps)}..{max(expected_steps)}, "
                f"got {sorted(actual_steps)}"
            )
        source_mappings: list[dict[str, Any]] = []
        source_validations = []
        for step in steps:
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
            timing_by_step.setdefault(step.step_id, []).append(
                {
                    "source": source,
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
                    "context_reqs": validation["context_reqs"],
                    "context_tokens": validation["context_tokens"],
                    "timeline_step": {
                        "step_index": step.step_id,
                        "label": step.label,
                        "trace_start_us": min(float(item["ts_us"]) for item in mappings),
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
            raise ValueError("TRT prefill capture has no exact one-request/8192-token step")
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
        exact_sources_by_step: dict[int, set[str]] = defaultdict(set)
        for row in exact_8k:
            exact_sources_by_step[int(row["step_id"])].add(str(row["source"]))
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
        reference_steps = [
            row["timeline_step"]
            for row in observed_steps
            if row["source"] == reference_source
        ]
        reference_path = paths[(workers[0], 0)]
        reference_rank = 0
        reference_worker = workers[0]

    critical_steps = {}
    for step_id, rows in sorted(timing_by_step.items()):
        selected = max(rows, key=lambda item: item["gpu_span_us"])
        active_us = selected["gpu_busy_union_us"]
        residency_us = selected["gpu_residency_us"]
        elapsed_us = selected["gpu_span_us"]
        if active_us > elapsed_us + 1e-6 or active_us > residency_us + 1e-6:
            raise ValueError(f"TRT {args.phase} step {step_id}: impossible timing values")
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
        "semantics": "step elapsed is first-to-last GPU kernel span from one critical worker/rank; CPU NVTX is asynchronous launch wall; rank residency is never summed",
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
    attributed_ratio = (status_us["mapped"] + status_us["fusion"]) / total_us
    strict_signature_us = sum(
        float(event["dur_us"])
        for event in all_mappings
        if event.get("attribution_method") == "unique_kernel_signature"
    )

    profile_id = f"qwen35_trtllm_attention_dp4_moe_ep4_agentx_{args.phase}"
    timeline = build_timeline_artifact(
        profile_id=profile_id,
        phase=args.phase,
        reference_rank=reference_rank,
        steps=reference_steps,
        timing_summary=timing_summary,
        raw_trace={
            "file": reference_path.name,
            "sha256": sha256_file(reference_path),
            "format": "Nsight Systems SQLite export",
            "rank": reference_rank,
            "worker": reference_worker,
        },
        stack_source={
            "mode": "nsight_nvtx_and_cuda_graph_node_identity",
            "file": reference_path.name,
            "sha256": sha256_file(reference_path),
            "mapped_residency_ratio": round(attributed_ratio, 6),
            "unmapped_residency_ratio": round(status_us["unmapped"] / total_us, 6),
            "policy": "unique kernel signatures remain mapped; unresolved graph occurrences remain explicit unmapped events with candidates",
        },
        target_resolver=QWEN35_TIMELINE_TARGETS,
    )

    if args.phase == "decode":
        measured_shape = {
            "generation_requests": {
                "samples": [row["generation_reqs"] for row in shape_observations],
                "min": min(row["generation_reqs"] for row in shape_observations),
                "median": statistics.median(
                    row["generation_reqs"] for row in shape_observations
                ),
                "max": max(row["generation_reqs"] for row in shape_observations),
            }
        }
    else:
        owner_shapes = [row for row in shape_observations if row["owner_compute"]]
        measured_shape = {
            "owner_context": owner_shapes,
            "one_chunk_8k_owner_samples": sum(
                row["context_reqs"] == 1 and row["context_tokens"] == 8192
                for row in owner_shapes
            ),
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
        "label": f"Qwen3.5 397B · TRT-LLM · AgentX DEP4 + MTP6 · {args.phase}",
        "model_id": "qwen35_397b_a17b",
        "execution_path_id": "attention_dp4_moe_ep4",
        "implementation_id": "trtllm_1cef02e9_attention_dp4_moe_ep4_mtp",
        "variant_id": f"trtllm_agentx_dep4_mtp6_{args.phase}_c704_a_z97",
        "phase": args.phase,
        "generation_mode": "mtp",
        "entry_view": "top",
        "execution_parameters": {"tp_size": 1, "dp_size": 4, "cp_size": 1, "ep_size": 4},
        "hardware": {
            "gpu": "GB300",
            "gpus_per_node": 4,
            "nodes": expected_workers,
            "topology_scope": f"{expected_workers} disaggregated {args.phase} workers",
        },
        "workload": {
            "suite": "AgentX A-Z97",
            "concurrency": 704,
            "duration_seconds": 3600,
            "warmup_requests_per_lane": 10,
            "warmup_grace_seconds": 1800,
            "mtp_draft_tokens": 6,
            "decode_cuda_graph_batch_cap": 32,
            "measured_shape": measured_shape,
        },
        "profiler": {
            "type": "nsight_systems_worker_local",
            "rank": "all four DEP ranks on every worker",
            "trace": ["cuda", "nvtx"],
            "cuda_graph_enabled": args.phase == "decode",
            "gpu_metric_semantics": "maximum worker/rank residency; parallel ranks and workers are not summed",
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
            "model_revision": MODEL_REVISION,
            "model_config_sha256": MODEL_CONFIG_SHA256,
            "container_sha256": CONTAINER_SHA256,
            "report_files": [
                {
                    "worker": worker,
                    "rank": rank,
                    "file": path.name,
                    "sha256": sha256_file(path),
                }
                for (worker, rank), path in sorted(paths.items())
            ],
            "mapping_policy": "NVTX step + runtime correlation + CUDA Graph node occurrence + exact GGGA/MTP6 order",
            "mapped_or_fusion_duration_ratio": round(attributed_ratio, 6),
            "strict_signature_duration_ratio": round(strict_signature_us / total_us, 6),
            "mapped_duration_ratio": round(status_us["mapped"] / total_us, 6),
            "fusion_duration_ratio": round(status_us["fusion"] / total_us, 6),
            "unmapped_duration_ratio": round(status_us["unmapped"] / total_us, 6),
            "timeline_interval_coverage_ratio": round(sum(status_us.values()) / total_us, 6),
            "semantic_attribution_gate": {"threshold": 0.90, "passed": attributed_ratio >= 0.90},
            "critical_step_wall_ms": timing_summary["critical_step_wall_ms"],
            "critical_cpu_launch_wall_ms": timing_summary[
                "critical_cpu_launch_wall_ms"
            ],
            "four_rank_validation": True,
            "worker_count": expected_workers,
        },
        "timeline": {},
        "node_states": node_states,
        "node_metrics": _aggregate_metrics(source_metrics),
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
        "mapped_or_fusion_duration_ratio": attributed_ratio,
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
    args.output_profile.write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")
    args.output_analysis.parent.mkdir(parents=True, exist_ok=True)
    args.output_analysis.write_text(json.dumps(analysis, indent=2) + "\n")
    args.output_mapping.parent.mkdir(parents=True, exist_ok=True)
    with args.output_mapping.open("w") as output:
        for row in mappings:
            output.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(f"wrote {args.output_profile.resolve()}")
    print(
        f"phase={args.phase} attributed={profile['evidence']['mapped_or_fusion_duration_ratio']:.3f} "
        f"strict={profile['evidence']['strict_signature_duration_ratio']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
