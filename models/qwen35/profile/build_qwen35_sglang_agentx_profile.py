#!/usr/bin/env python3
"""Build the real A-Z97/C704 SGLang AgentX steady-state decode profile."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
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
from models.common.trace_mapping import find_eagle_mtp_decode_windows, load_trace
from models.qwen35.profile.build_qwen35_sglang_decode_profile import (
    MODEL_REVISION,
    RUNTIME_SOURCE_COMMIT,
    SGLANG_NODE_STATES,
    SOURCE_COMMIT,
    _metrics_for_rank,
    _validate_step_signatures,
    sha256_file,
    trace_rank,
)
from models.qwen35.profile.qwen35_graph_mapping import (
    attach_graph_stack_evidence,
    map_graph_window,
)


PROFILE_ID = "qwen35_sglang_attention_dp4_moe_ep4_mtp6_agentx_steady_c704"
JOB_ID = 3205969
SELECTED_WINDOW_INDICES = (4, 5, 6, 7)
WORKER_RE = re.compile(r"decode_w([01])")
LOG_ROW_RE = re.compile(
    r"DP(?P<rank>\d+) TP\d+ EP\d+\] Decode batch \[(?P<step>\d+)\], "
    r"#running-req: (?P<running>\d+),.*?accept len: (?P<accept>[0-9.]+),.*?"
    r"cuda graph: (?P<graph>True|False),.*?#queue-req: (?P<queue>\d+)"
)
BENCHMARK_RE = re.compile(
    r"rps=(?P<rps>[0-9.]+) \(avg (?P<avg_rps>[0-9.]+)\).*?"
    r"done=(?P<done>[0-9,]+) ok=(?P<ok>[0-9,]+) err=(?P<errors>[0-9,]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, nargs=8, required=True)
    parser.add_argument("--worker-logs", type=Path, nargs=2, required=True)
    parser.add_argument("--fingerprints", type=Path, nargs=2, required=True)
    parser.add_argument("--benchmark-log", type=Path, required=True)
    parser.add_argument("--job-metadata", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--eager-mapping", type=Path, required=True)
    parser.add_argument("--output-profile", type=Path, required=True)
    parser.add_argument("--output-timeline", type=Path, required=True)
    parser.add_argument("--output-analysis", type=Path, required=True)
    parser.add_argument("--output-mapping", type=Path, required=True)
    return parser.parse_args()


def source_identity(path: Path) -> tuple[int, int]:
    worker_match = WORKER_RE.search(str(path))
    if worker_match is None:
        raise ValueError(f"cannot parse decode worker from {path}")
    return int(worker_match.group(1)), trace_rank(path)


def worker_identity(path: Path) -> int:
    match = WORKER_RE.search(path.name)
    if match is None:
        raise ValueError(f"cannot parse decode worker from {path.name}")
    return int(match.group(1))


def parse_worker_profile_observations(path: Path) -> list[dict[str, Any]]:
    """Return only scheduler observations bounded by profiler start/stop."""

    lines = path.read_text(errors="replace").splitlines()
    start = next(
        (index for index, line in enumerate(lines) if "Profiling starts." in line),
        None,
    )
    if start is None:
        raise ValueError(f"{path}: missing profiler start marker")
    stop = next(
        (
            index
            for index, line in enumerate(lines[start + 1 :], start + 1)
            if "Stop profiling..." in line
        ),
        None,
    )
    if stop is None:
        raise ValueError(f"{path}: missing profiler stop marker")
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
                "accepted_length": float(match.group("accept")),
                "cuda_graph": match.group("graph") == "True",
                "queued_requests": int(match.group("queue")),
            }
        )
    if set(row["dp_rank"] for row in rows) != {0, 1, 2, 3}:
        raise ValueError(f"{path}: profile window does not observe all four DP ranks")
    if any(not row["cuda_graph"] or row["queued_requests"] for row in rows):
        raise ValueError(f"{path}: capture is not queue-free CUDA-Graph steady state")
    return rows


def parse_benchmark_snapshot(path: Path) -> dict[str, Any]:
    matches = list(BENCHMARK_RE.finditer(path.read_text(errors="replace")))
    if not matches:
        raise ValueError(f"{path}: no AIPerf realtime snapshot")
    match = matches[-1]
    result = {
        "instant_rps": float(match.group("rps")),
        "average_rps": float(match.group("avg_rps")),
        "done": int(match.group("done").replace(",", "")),
        "ok": int(match.group("ok").replace(",", "")),
        "errors": int(match.group("errors").replace(",", "")),
    }
    if result["errors"] or result["done"] != result["ok"]:
        raise ValueError(f"{path}: benchmark snapshot contains errors: {result}")
    return result


def validate_run_inputs(args: argparse.Namespace) -> dict[str, Any]:
    config = yaml.safe_load(args.config.read_text())
    job = json.loads(args.job_metadata.read_text())
    resources = config.get("resources") or {}
    benchmark_env = ((config.get("benchmark") or {}).get("env") or {})
    decode = (((config.get("backend") or {}).get("sglang_config") or {}).get("decode") or {})
    expected = {
        "model revision": ((config.get("identity") or {}).get("model") or {}).get("revision"),
        "runtime source": ((config.get("identity") or {}).get("frameworks") or {}).get("sglang_source"),
        "prefill workers": resources.get("prefill_workers"),
        "decode workers": resources.get("decode_workers"),
        "concurrency": benchmark_env.get("CONCURRENCY"),
        "DP": decode.get("data-parallel-size"),
        "EP": decode.get("expert-parallel-size"),
        "MTP steps": decode.get("speculative-num-steps"),
        "draft tokens": decode.get("speculative-num-draft-tokens"),
        "CUDA graph max BS": decode.get("cuda-graph-max-bs"),
    }
    required = {
        "model revision": MODEL_REVISION,
        "runtime source": RUNTIME_SOURCE_COMMIT,
        "prefill workers": 3,
        "decode workers": 2,
        "concurrency": "704",
        "DP": 4,
        "EP": 4,
        "MTP steps": 5,
        "draft tokens": 6,
        "CUDA graph max BS": 80,
    }
    mismatch = {
        key: {"expected": value, "actual": expected[key]}
        for key, value in required.items()
        if expected[key] != value
    }
    if "a-z97" not in str(config.get("name", "")).lower():
        mismatch["workload distribution"] = {"expected": "A-Z97", "actual": config.get("name")}
    if int(job.get("job_id", -1)) != JOB_ID:
        mismatch["job id"] = {"expected": JOB_ID, "actual": job.get("job_id")}
    if mismatch:
        raise ValueError(f"AgentX run identity mismatch: {mismatch}")
    return {"config": config, "job": job}


def _validate_fingerprints(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    workers = set()
    for path in paths:
        worker = worker_identity(path)
        workers.add(worker)
        data = json.loads(path.read_text())
        if ((data.get("model") or {}).get("hf_revision")) != MODEL_REVISION:
            raise ValueError(f"{path}: model revision mismatch")
        gpu_names = [gpu.get("name") for gpu in ((data.get("gpu") or {}).get("gpus") or [])]
        if gpu_names != ["NVIDIA GB300"] * 4:
            raise ValueError(f"{path}: expected four GB300 GPUs, got {gpu_names}")
        rows.append(
            {
                "worker": worker,
                "file": path.name,
                "sha256": sha256_file(path),
                "hostname": data.get("hostname"),
                "frameworks": data.get("frameworks"),
            }
        )
    if workers != {0, 1}:
        raise ValueError(f"incomplete decode-worker fingerprints: {workers}")
    return sorted(rows, key=lambda row: row["worker"])


def _aggregate_source_metrics(source_metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    nodes = sorted({node for metrics in source_metrics.values() for node in metrics})
    result = {}
    for node in nodes:
        candidates = [
            (source, metrics[node])
            for source, metrics in sorted(source_metrics.items())
            if node in metrics
        ]
        source, selected = max(candidates, key=lambda item: item[1]["ms_per_iter"])
        values = [cell["ms_per_iter"] for _source, cell in candidates]
        result[node] = {
            **selected,
            "ms_per_iter": round(selected["ms_per_iter"], 6),
            "aggregation": "maximum worker/rank kernel residency",
            "source_worker_rank": source,
            "worker_rank_range_ms": [round(min(values), 6), round(max(values), 6)],
        }
    return result


def build(args: argparse.Namespace):
    run = validate_run_inputs(args)
    fingerprint_rows = _validate_fingerprints(args.fingerprints)
    benchmark = parse_benchmark_snapshot(args.benchmark_log)
    log_observations = {
        f"w{worker_identity(path)}": parse_worker_profile_observations(path)
        for path in args.worker_logs
    }
    if set(log_observations) != {"w0", "w1"}:
        raise ValueError(f"incomplete worker logs: {log_observations.keys()}")

    paths_by_source = {source_identity(path): path.resolve() for path in args.traces}
    expected_sources = {(worker, rank) for worker in (0, 1) for rank in range(4)}
    if set(paths_by_source) != expected_sources:
        raise ValueError(f"incomplete two-worker/four-rank trace coverage: {paths_by_source}")

    source_metrics: dict[str, dict[str, Any]] = {}
    source_wall_ms: dict[str, list[float]] = {}
    source_batches: dict[str, list[int]] = {}
    source_validation: dict[str, list[dict[str, Any]]] = {}
    reference_steps = []
    all_mapping_rows = []

    for (worker, rank), path in sorted(paths_by_source.items()):
        trace_events = load_trace(path).get("traceEvents") or []
        windows = find_eagle_mtp_decode_windows(trace_events, signature="fused_qkvzba_split")
        if len(windows) < max(SELECTED_WINDOW_INDICES) + 1:
            raise ValueError(f"worker {worker} rank {rank} has only {len(windows)} windows")
        mapped_source = []
        validations = []
        walls = []
        batches = []
        for output_index, window_index in enumerate(SELECTED_WINDOW_INDICES):
            window = windows[window_index]
            mapped, validation = map_graph_window(
                trace_events,
                window=window,
                rank=rank,
                step_index=window_index,
            )
            _validate_step_signatures(validation, rank=rank, step=window_index)
            batch = int(validation["target_verify_batch_size"])
            if not 1 <= batch <= 80:
                raise ValueError(f"worker {worker} rank {rank}: graph batch {batch} outside 1..80")
            for event in mapped:
                event["worker"] = worker
                event["event_id"] = f"w{worker}-{event['event_id']}"
            mapped_source.extend(mapped)
            validations.append(validation)
            walls.append((window.end_us - window.start_us) / 1000.0)
            batches.append(batch)
            if worker == 0 and rank == 0:
                reference_steps.append(
                    {
                        "step_index": output_index,
                        "label": f"AgentX A-Z97 steady decode · graph BS{batch}",
                        "trace_start_us": window.start_us,
                        "duration_us": window.end_us - window.start_us,
                        "events": attach_graph_stack_evidence(
                            mapped, mapping_path=args.eager_mapping
                        ),
                    }
                )
        source = f"w{worker}/r{rank}"
        source_metrics[source] = _metrics_for_rank(mapped_source, len(SELECTED_WINDOW_INDICES))
        source_wall_ms[source] = walls
        source_batches[source] = batches
        source_validation[source] = validations
        all_mapping_rows.extend(mapped_source)

    reference_events = [event for step in reference_steps for event in step["events"]]
    reference_total_us = sum(float(event["dur_us"]) for event in reference_events)
    reference_stack_us = sum(
        float(event["dur_us"]) for event in reference_events if event.get("python_stack")
    )
    stack_ratio = reference_stack_us / reference_total_us if reference_total_us else 0.0
    if stack_ratio < 0.95:
        raise ValueError(f"eager stack transfer covers {stack_ratio:.4f}; required >= 0.95")

    critical_wall_ms = [
        max(values[index] for values in source_wall_ms.values())
        for index in range(len(SELECTED_WINDOW_INDICES))
    ]
    timing_summary = {
        "semantics": "critical wall time is max across concurrent workers/ranks; GPU residency is never summed",
        "critical_wall_ms": {
            "samples": [round(value, 6) for value in critical_wall_ms],
            "mean": round(statistics.fmean(critical_wall_ms), 6),
            "median": round(statistics.median(critical_wall_ms), 6),
            "min": round(min(critical_wall_ms), 6),
            "max": round(max(critical_wall_ms), 6),
        },
        "source_wall_ms": {
            source: [round(value, 6) for value in values]
            for source, values in source_wall_ms.items()
        },
    }
    reference_path = paths_by_source[(0, 0)]
    timeline = build_timeline_artifact(
        profile_id=PROFILE_ID,
        phase="decode",
        reference_rank=0,
        steps=reference_steps,
        timing_summary=timing_summary,
        raw_trace={
            "file": reference_path.name,
            "sha256": sha256_file(reference_path),
            "format": "PyTorch profiler trace JSON gzip",
            "worker": 0,
            "rank": 0,
        },
        stack_source={
            "mode": "eager_trace_transfer",
            "file": args.eager_mapping.name,
            "sha256": sha256_file(args.eager_mapping),
            "mapped_residency_ratio": round(stack_ratio, 6),
            "policy": "exact kernel+IR, representative IR node, then declared containing IR scope",
        },
    )

    total_us = sum(float(event["dur_us"]) for event in all_mapping_rows)
    status_us: Counter[str] = Counter()
    for event in all_mapping_rows:
        status_us[str(event["mapping_status"])] += float(event["dur_us"])
    attributed_ratio = (status_us["mapped"] + status_us["fusion"]) / total_us
    trace_batches = [batch for values in source_batches.values() for batch in values]
    runtime_rows = [row for rows in log_observations.values() for row in rows]
    node_metrics = _aggregate_source_metrics(source_metrics)
    profile = {
        "schema_version": "profile.v2",
        "profile_id": PROFILE_ID,
        "label": "Qwen3.5 397B · SGLang · real AgentX A-Z97 C704 · DEP4 + MTP6 steady decode",
        "model_id": "qwen35_397b_a17b",
        "execution_path_id": "attention_dp4_moe_ep4",
        "implementation_id": "sglang_85c23c62_attention_dp4_moe_ep4_mtp",
        "variant_id": "sglang_agentx_a_z97_c704_3p2d_dep4_mtp6_cg_steady",
        "phase": "decode",
        "generation_mode": "mtp",
        "entry_view": "generation_loop",
        "execution_parameters": {"tp_size": 4, "dp_size": 4, "cp_size": 1, "ep_size": 4},
        "hardware": {"gpu": "GB300", "gpus_per_worker": 4, "prefill_workers": 3, "decode_workers": 2},
        "workload": {
            "scenario": "inferencex-agentx-mvp",
            "rank_distribution": "A-Z97",
            "concurrency": 704,
            "selected_stable_iterations": list(SELECTED_WINDOW_INDICES),
            "measured_target_verify_batch": {
                "scope": "decode worker/rank graph batch",
                "samples": trace_batches,
                "min": min(trace_batches),
                "median": statistics.median(trace_batches),
                "max": max(trace_batches),
                "by_source": source_batches,
            },
            "scheduler_observations": {
                "samples": len(runtime_rows),
                "running_requests_range": [
                    min(row["running_requests"] for row in runtime_rows),
                    max(row["running_requests"] for row in runtime_rows),
                ],
                "accepted_length_range": [
                    min(row["accepted_length"] for row in runtime_rows),
                    max(row["accepted_length"] for row in runtime_rows),
                ],
                "queue_requests": 0,
                "cuda_graph": True,
            },
        },
        "profiler": {
            "type": "torch",
            "rank": "both decode workers, all four DEP ranks",
            "activities": ["CPU", "GPU"],
            "cuda_graph_enabled": True,
            "with_stack": False,
            "record_shapes": False,
            "gpu_metric_semantics": "maximum worker/rank kernel residency; concurrent workers/ranks are not summed",
        },
        "evidence": {
            "job_id": JOB_ID,
            "source_commit": SOURCE_COMMIT,
            "runtime_source_commit": RUNTIME_SOURCE_COMMIT,
            "model_revision": MODEL_REVISION,
            "config_file": args.config.name,
            "config_sha256": sha256_file(args.config),
            "job_metadata_file": args.job_metadata.name,
            "job_metadata_sha256": sha256_file(args.job_metadata),
            "benchmark_log_file": args.benchmark_log.name,
            "benchmark_log_sha256": sha256_file(args.benchmark_log),
            "benchmark_snapshot": benchmark,
            "fingerprints": fingerprint_rows,
            "worker_logs": [
                {
                    "worker": worker_identity(path),
                    "file": path.name,
                    "sha256": sha256_file(path),
                }
                for path in sorted(args.worker_logs, key=worker_identity)
            ],
            "trace_files": [
                {"worker": worker, "rank": rank, "file": path.name, "sha256": sha256_file(path)}
                for (worker, rank), path in sorted(paths_by_source.items())
            ],
            "eager_mapping_file": args.eager_mapping.name,
            "eager_mapping_sha256": sha256_file(args.eager_mapping),
            "mapped_or_fusion_duration_ratio": round(attributed_ratio, 6),
            "strict_signature_duration_ratio": round(status_us["mapped"] / total_us, 6),
            "eager_stack_transfer_duration_ratio": round(stack_ratio, 6),
            "critical_decode_step_ms": timing_summary["critical_wall_ms"],
        },
        "timeline": {},
        "node_states": SGLANG_NODE_STATES,
        "node_metrics": node_metrics,
    }
    analysis = {
        "profile_id": PROFILE_ID,
        "run_identity": {
            "job_id": run["job"].get("job_id"),
            "name": run["config"].get("name"),
        },
        "source_wall_ms": source_wall_ms,
        "critical_wall_ms": critical_wall_ms,
        "source_target_verify_batches": source_batches,
        "worker_log_observations": log_observations,
        "benchmark_snapshot": benchmark,
        "source_validation": source_validation,
        "status_duration_us": dict(status_us),
        "mapped_or_fusion_duration_ratio": attributed_ratio,
        "strict_signature_duration_ratio": status_us["mapped"] / total_us,
        "eager_stack_transfer_duration_ratio": stack_ratio,
        "node_metrics": node_metrics,
    }
    return profile, timeline, analysis, all_mapping_rows


def main() -> int:
    args = parse_args()
    profile, timeline, analysis, mappings = build(args)
    timeline_sha = write_timeline_artifact(args.output_timeline, timeline)
    profile["timeline"] = {
        "schema_version": "timeline.v1",
        "artifact": args.output_timeline.name,
        "sha256": timeline_sha,
        "reference_worker": 0,
        "reference_rank": 0,
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
    print(f"wrote {args.output_timeline.resolve()}")
    print(
        f"critical mean={profile['evidence']['critical_decode_step_ms']['mean']:.3f} ms "
        f"attributed={profile['evidence']['mapped_or_fusion_duration_ratio']:.3f} "
        f"stack={profile['evidence']['eager_stack_transfer_duration_ratio']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
