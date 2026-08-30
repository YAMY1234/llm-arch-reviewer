#!/usr/bin/env python3
"""Build one fail-closed Kimi K3 SGLang pure-TP8 production profile.

Raw Nsight reports and graph-off traces stay in task evidence.  The repository
receives only deterministic Profile v2, Timeline v1, and attribution audit
artifacts with public, path-free provenance hashes.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from typing import Any, Iterable

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from models.common.timeline_artifact import (
    attach_eager_stack_evidence,
    build_timeline_artifact,
    sha256_file,
    write_timeline_artifact,
)
from models.kimi_k3.build.kimi_k3_production_attribution import (
    ATTN_RES_ANCHOR_COUNT,
    attribute_sglang_production_events,
)
from models.kimi_k3.build.kimi_k3_profile_contract import (
    build_node_states,
    sglang_fusion_groups,
)


MODEL_REVISION = "a590ce090cb049c93a33dfe8c208ec652aa20503"
SOURCE_COMMIT = "25035bff8d34f3fcce2c1a2a5b1fe610225e84ed"
CONTAINER = (
    "immutable_sglang_runtime@"
    "sha256:a552834207a8a12b03c7b2fabcdea0406822ec1daccdc4e57c21a4c6c68f70c8"
)
IMPLEMENTATION_ID = "sglang_25035bff_kimi_k3_tp8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-root", type=Path, required=True)
    parser.add_argument("--eager-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("prefill", "decode"), required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--baseline-relative-step", type=int, required=True)
    parser.add_argument("--client-source", type=Path, required=True)
    parser.add_argument("--model-ir", type=Path, required=True)
    parser.add_argument("--output-profile", type=Path, required=True)
    parser.add_argument("--output-analysis", type=Path, required=True)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _union_duration_us(intervals: Iterable[tuple[float, float]]) -> float:
    ordered = sorted((float(a), float(b)) for a, b in intervals if b > a)
    if not ordered:
        return 0.0
    merged = [list(ordered[0])]
    for start, stop in ordered[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], stop)
        else:
            merged.append([start, stop])
    return sum(stop - start for start, stop in merged)


def read_device_kernels(sqlite_path: Path, device: int) -> list[dict[str, Any]]:
    require(sqlite_path.is_file(), f"missing Nsight SQLite export: {sqlite_path}")
    connection = sqlite3.connect(sqlite_path)
    try:
        rows = connection.execute(
            """
            SELECT k.start, k.end, k.deviceId, k.streamId, s.value,
                   k.graphNodeId, k.correlationId, k.gridId
              FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
              JOIN StringIds AS s ON s.id = k.demangledName
             WHERE k.deviceId = ?
             ORDER BY k.start, k.end, k.gridId
            """,
            (device,),
        ).fetchall()
    finally:
        connection.close()
    return [
        {
            "ts_us": start / 1000.0,
            "dur_us": (stop - start) / 1000.0,
            "device": raw_device,
            "stream": stream,
            "kernel_name": name,
            "graph_node_id": graph_node,
            "correlation": correlation,
            "grid_id": grid,
        }
        for start, stop, raw_device, stream, name, graph_node, correlation, grid in rows
    ]


def read_exact_device_kernels(
    sqlite_path: Path, device: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Slice one TP worker by launch correlation inside the node API window.

    Nsight's interactive process-tree session is armed before the formal
    request and stopped only after all four local workers finish.  One local
    TP leader broadcasts the profiler start/stop operation on each node.  Its
    CPU API interval is therefore the shared exact window; each worker remains
    source-exact by joining only its own CUDA launches to its own device
    kernels.  Launch time, rather than asynchronous GPU execution time, keeps
    kernels that execute after the stop API but were launched inside the
    formal window.
    """

    require(sqlite_path.is_file(), f"missing Nsight SQLite export: {sqlite_path}")
    connection = sqlite3.connect(sqlite_path)
    try:
        global_pids = [
            int(row[0])
            for row in connection.execute(
                """
                SELECT DISTINCT globalPid
                  FROM CUPTI_ACTIVITY_KIND_KERNEL
                 WHERE deviceId = ?
                """,
                (device,),
            ).fetchall()
        ]
        require(
            len(global_pids) == 1,
            f"device {device} does not have exactly one profiled worker: {global_pids}",
        )
        global_pid = global_pids[0]
        calls = connection.execute(
            """
            SELECT r.start, r.end, s.value, (r.globalTid >> 24)
             FROM CUPTI_ACTIVITY_KIND_RUNTIME AS r
              JOIN StringIds AS s ON s.id = r.nameId
             WHERE s.value LIKE '%ProfilerStart%' OR s.value LIKE '%ProfilerStop%'
             ORDER BY r.start, r.end
            """
        ).fetchall()
        starts = [row for row in calls if "ProfilerStart" in str(row[2])]
        stops = [row for row in calls if "ProfilerStop" in str(row[2])]
        require(
            len(starts) == 1,
            f"node profiler-leader start count {len(starts)} for device {device}",
        )
        require(
            len(stops) == 1,
            f"node profiler-leader stop count {len(stops)} for device {device}",
        )
        leader_process_key = int(starts[0][3])
        require(
            int(stops[0][3]) == leader_process_key,
            f"node profiler start/stop leaders differ for device {device}",
        )
        start_ns = int(starts[0][1])
        stop_ns = int(stops[0][0])
        require(start_ns < stop_ns, f"device {device} profiler API order mismatch")
        total_kernel_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE deviceId = ?",
                (device,),
            ).fetchone()[0]
        )
        rows = connection.execute(
            """
            SELECT DISTINCT k.start, k.end, k.deviceId, k.streamId, s.value,
                   k.graphNodeId, k.correlationId, k.gridId
              FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
              JOIN StringIds AS s ON s.id = k.demangledName
              JOIN CUPTI_ACTIVITY_KIND_RUNTIME AS r
                ON r.correlationId = k.correlationId
               AND (r.globalTid >> 24) = (k.globalPid >> 24)
             WHERE k.deviceId = ? AND k.globalPid = ?
               AND r.start >= ? AND r.end <= ?
             ORDER BY k.start, k.end, k.gridId
            """,
            (device, global_pid, start_ns, stop_ns),
        ).fetchall()
    finally:
        connection.close()
    kernels = [
        {
            "ts_us": start / 1000.0,
            "dur_us": (stop - start) / 1000.0,
            "device": raw_device,
            "stream": stream,
            "kernel_name": name,
            "graph_node_id": graph_node,
            "correlation": correlation,
            "grid_id": grid,
        }
        for start, stop, raw_device, stream, name, graph_node, correlation, grid in rows
    ]
    require(kernels, f"device {device} exact profiler interval contains no kernels")
    gpu_execution_after_stop = sum(
        1 for start, stop, *_ in rows if int(start) > stop_ns or int(stop) > stop_ns
    )
    return kernels, {
        "worker_global_pid_hash": hashlib.sha256(str(global_pid).encode()).hexdigest()[:16],
        "node_leader_global_pid_hash": hashlib.sha256(
            str(leader_process_key).encode()
        ).hexdigest()[:16],
        "node_leader_profiler_start_api_count": len(starts),
        "node_leader_profiler_stop_api_count": len(stops),
        "profiler_interval_us": round((stop_ns - start_ns) / 1000.0, 6),
        "node_collection_kernel_count": total_kernel_count,
        "launch_correlated_exact_window_kernel_count": len(kernels),
        "gpu_execution_after_profiler_stop_kernel_count": gpu_execution_after_stop,
    }


def validate_client(
    *, root: Path, batch_size: int, baseline_relative_step: int, client_source: Path
) -> dict[str, Any]:
    path = root / f"client-c{batch_size}.json"
    require(path.is_file(), f"missing exact client evidence: {path}")
    client = load_json(path)
    contract = client.get("contract") or {}
    requests = (client.get("warmup") or {}).get("requests", []) + (
        client.get("formal") or {}
    ).get("requests", [])
    require(client.get("state") == "passed", "exact client did not pass")
    require(contract.get("concurrency") == batch_size, "client concurrency mismatch")
    require(contract.get("isl") == 8192 and contract.get("osl") == 1024, "length mismatch")
    require(contract.get("warmup_request_count") == 3 * batch_size, "warmup mismatch")
    require(contract.get("formal_request_count") == batch_size, "formal mismatch")
    require(contract.get("no_intentionally_shared_prefix") is True, "shared prefix")
    require(len(requests) == 4 * batch_size, "realized request count mismatch")
    require(
        all(
            row.get("http_status") == 200
            and row.get("realized_prompt_tokens") == 8192
            and row.get("realized_completion_tokens") == 1024
            for row in requests
        ),
        "realized request contract mismatch",
    )
    prompt_hashes = [row.get("prompt_token_sha256") for row in requests]
    require(
        all(prompt_hashes) and len(set(prompt_hashes)) == len(prompt_hashes),
        "prompt streams are not unique",
    )
    require(client_source.is_file(), f"missing client source: {client_source}")
    require(
        (client.get("client_source") or {}).get("sha256") == sha256_file(client_source),
        "client source hash mismatch",
    )
    coordinate = client.get("profile_coordinate") or {}
    require(
        coordinate.get("baseline_relative_start_step") == baseline_relative_step,
        "baseline-relative step mismatch",
    )
    controls = client.get("profile_controls") or []
    require(len(controls) == 1 and controls[0].get("http_status") == 200, "profile control")
    request = controls[0].get("request") or {}
    require(request.get("num_steps") == 1, "capture must contain exactly one step")
    require(request.get("activities") == ["CUDA_PROFILER"], "Nsight capture activity")
    require(request.get("with_stack") is False, "production capture unexpectedly has stacks")
    return {
        "sha256": sha256_file(path),
        "client_source_sha256": sha256_file(client_source),
        "resolved_absolute_start_step": coordinate.get("resolved_absolute_start_step"),
        "baseline_relative_start_step": baseline_relative_step,
        "warmup_cached_token_count": coordinate.get("warmup_cached_token_count"),
        "profile_request_start_step": request.get("start_step"),
    }


def _rank_source(root: Path, rank: int) -> tuple[Path, Path, int]:
    node_rank = rank // 4
    device = rank % 4
    sqlite_path = root / "nsys" / f"node-rank{node_rank}.sqlite"
    rep_path = root / "nsys" / f"node-rank{node_rank}.nsys-rep"
    require(rep_path.is_file(), f"missing raw Nsight report: {rep_path}")
    return sqlite_path, rep_path, device


def _semantic_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row["node"]) for row in rows if row.get("node")).items()))


def validate_and_attribute_ranks(
    *, production_root: Path, eager_root: Path, phase: str
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    rank_results: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] | None = None
    reference_diagnostics: dict[str, Any] | None = None
    count_fingerprints: set[str] = set()
    for rank in range(8):
        sqlite_path, rep_path, device = _rank_source(production_root, rank)
        eager_mapping = eager_root / "mapping" / f"tp{rank}" / f"kernel_mapping.tp{rank}.jsonl"
        require(eager_mapping.is_file(), f"missing eager TP{rank} mapping")
        production, exact_window = read_exact_device_kernels(sqlite_path, device)
        attributed, diagnostics = attribute_sglang_production_events(
            production, eager_mapping
        )
        require(diagnostics["anchor_count"] == ATTN_RES_ANCHOR_COUNT, "anchor closure")
        require(diagnostics["mapped_kernel_duration_ratio"] >= 0.99, "duration coverage")
        mapped = [row for row in attributed if row.get("node")]
        if phase == "decode":
            require(mapped and all(row.get("graph_node_id") is not None for row in mapped), "decode graph state")
        else:
            require(not any(row.get("graph_node_id") is not None for row in attributed), "prefill graph spill")
        counts = _semantic_counts(attributed)
        count_fingerprints.add(json.dumps(counts, sort_keys=True))
        rank_results.append(
            {
                "rank": rank,
                "node_rank": rank // 4,
                "local_device": device,
                "kernel_count": len(attributed),
                "mapped_kernel_count": diagnostics["mapped_kernel_count"],
                "support_kernel_count": diagnostics["support_kernel_count"],
                "mapped_kernel_duration_ratio": diagnostics["mapped_kernel_duration_ratio"],
                "semantic_node_counts": counts,
                "raw_report_sha256": sha256_file(rep_path),
                "sqlite_export_sha256": sha256_file(sqlite_path),
                "eager_mapping_sha256": sha256_file(eager_mapping),
                "exact_window": exact_window,
            }
        )
        if rank == 0:
            reference_rows = attributed
            reference_diagnostics = diagnostics
    require(len(count_fingerprints) == 1, "semantic production counts differ across TP ranks")
    assert reference_rows is not None and reference_diagnostics is not None
    rank_audit = {
        "schema_version": "kimi-k3-production-rank-audit.v1",
        "state": "passed",
        "framework": "sglang",
        "source_commit": SOURCE_COMMIT,
        "phase": phase,
        "all_tp_ranks_validated": True,
        "phase_shape_rank_source_exact": True,
        "ranks": rank_results,
    }
    return reference_rows, reference_diagnostics, rank_audit


def build_node_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_node: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("node"):
            by_node[str(row["node"])].append(row)
    metrics: dict[str, Any] = {}
    for node, events in sorted(by_node.items()):
        active_us = _union_duration_us(
            (float(row["ts_us"]), float(row["ts_us"]) + float(row["dur_us"]))
            for row in events
        )
        residency_us = sum(float(row["dur_us"]) for row in events)
        label_duration: Counter[str] = Counter()
        label_count: Counter[str] = Counter()
        for row in events:
            label = str(row.get("kernel_label") or row.get("cpu_op_name") or node)
            label_duration[label] += float(row["dur_us"])
            label_count[label] += 1
        kernels = []
        for label, duration in label_duration.most_common():
            count = label_count[label]
            kernels.append(
                {
                    "name": label,
                    "count": count,
                    "count_per_iter": float(count),
                    "avg_us": round(duration / count, 6),
                    "total_us_per_iter": round(duration, 6),
                    "share_in_node_residency_pct": round(100.0 * duration / residency_us, 4),
                }
            )
        metrics[node] = {
            "ms_per_iter": round(active_us / 1000.0, 6),
            "active_gpu_ms": round(active_us / 1000.0, 6),
            "gpu_residency_ms": round(residency_us / 1000.0, 6),
            "gpu_residency_ms_per_iter": round(residency_us / 1000.0, 6),
            "attribution_status": "measured_direct",
            "metric_kind": "exclusive_leaf",
            "timing_semantics": "same-device union of directly attributed production intervals",
            "kernels": kernels,
        }
    return metrics


def profile_identity(phase: str, batch_size: int) -> tuple[str, str]:
    if phase == "prefill":
        return "kimi_k3_tp8_sglang_prefill_bs1_8k", "prefill_bs1_8k"
    return (
        f"kimi_k3_tp8_sglang_cg_decode_bs{batch_size}_8k1k",
        f"cg_decode_bs{batch_size}_8k1k",
    )


def main() -> int:
    args = parse_args()
    if args.phase == "prefill":
        require(args.batch_size == 1, "prefill is accepted only at batch 1")
    else:
        require(args.batch_size in {1, 16, 64}, "SGLang decode accepts 1/16/64; 256 has explicit unsupported evidence")
    production_root = args.production_root.resolve()
    eager_root = args.eager_root.resolve()
    client = validate_client(
        root=production_root,
        batch_size=args.batch_size,
        baseline_relative_step=args.baseline_relative_step,
        client_source=args.client_source.resolve(),
    )
    attributed, diagnostics, rank_audit = validate_and_attribute_ranks(
        production_root=production_root,
        eager_root=eager_root,
        phase=args.phase,
    )
    eager_mapping = eager_root / "mapping/tp0/kernel_mapping.tp0.jsonl"
    attributed = attach_eager_stack_evidence(attributed, mapping_path=eager_mapping)
    node_metrics = build_node_metrics(attributed)
    measured_nodes = set(node_metrics)
    fusion_groups = sglang_fusion_groups(
        phase=args.phase, batch_size=args.batch_size, measured_nodes=measured_nodes
    )
    model_ir = yaml.safe_load(args.model_ir.read_text())
    required_nodes = [
        f"{view_id}.{node['id']}"
        for view_id, view in model_ir["views"].items()
        for node in view["nodes"]
    ]
    node_states = build_node_states(
        required_nodes=required_nodes,
        measured_nodes=measured_nodes,
        fusion_groups=fusion_groups,
    )

    profile_id, variant_id = profile_identity(args.phase, args.batch_size)
    trace_start_us = min(float(row["ts_us"]) for row in attributed)
    trace_stop_us = max(float(row["ts_us"]) + float(row["dur_us"]) for row in attributed)
    duration_us = trace_stop_us - trace_start_us
    active_us = _union_duration_us(
        (float(row["ts_us"]), float(row["ts_us"]) + float(row["dur_us"]))
        for row in attributed
    )
    residency_us = sum(float(row["dur_us"]) for row in attributed)
    timing_summary = {
        "elapsed_ms": round(duration_us / 1000.0, 6),
        "active_gpu_ms": round(active_us / 1000.0, 6),
        "gpu_residency_ms": round(residency_us / 1000.0, 6),
        "device_gap_ms": round((duration_us - active_us) / 1000.0, 6),
        "gpu_overlap_ms": round((residency_us - active_us) / 1000.0, 6),
        "semantics": "same-device interval union and residency for one exact captured formal forward",
    }
    reference_rep = production_root / "nsys/node-rank0.nsys-rep"
    raw_hash = sha256_file(reference_rep)
    timeline_path = args.output_profile.with_suffix(".timeline.json.gz")
    timeline = build_timeline_artifact(
        profile_id=profile_id,
        phase=args.phase,
        reference_rank=0,
        steps=[
            {
                "step_index": 1,
                "label": f"formal {args.phase} BS{args.batch_size}",
                "trace_start_us": trace_start_us,
                "duration_us": duration_us,
                "events": attributed,
            }
        ],
        timing_summary=timing_summary,
        raw_trace={
            "file": f"capture-{raw_hash[:16]}.nsys-rep",
            "sha256": raw_hash,
            "format": "nsight_systems_nsys_rep",
            "rank": 0,
            "storage": "task_evidence_only",
        },
        stack_source={
            "source": "graph_off_eager_trace",
            "mapping_file": f"eager-mapping-{sha256_file(eager_mapping)[:16]}.jsonl",
            "mapping_sha256": sha256_file(eager_mapping),
            "policy": "AttnRes-occurrence-bounded normalized function identity and ordinal; exact eager provenance retained per mapped event",
        },
    )
    timeline_sha = write_timeline_artifact(timeline_path, timeline)

    public_rank_audit = {
        "schema_version": rank_audit["schema_version"],
        "state": rank_audit["state"],
        "framework": "sglang",
        "source_commit": SOURCE_COMMIT,
        "phase": args.phase,
        "all_tp_ranks_validated": True,
        "phase_shape_rank_source_exact": True,
        "rank_count": 8,
        "ranks": [
            {
                key: row[key]
                for key in (
                    "rank",
                    "kernel_count",
                    "mapped_kernel_count",
                    "support_kernel_count",
                    "mapped_kernel_duration_ratio",
                    "raw_report_sha256",
                    "sqlite_export_sha256",
                    "eager_mapping_sha256",
                    "exact_window",
                )
            }
            for row in rank_audit["ranks"]
        ],
    }
    analysis = {
        "schema_version": "kimi-k3-sglang-production-attribution.v1",
        "state": "passed",
        "profile_id": profile_id,
        "reference_rank": 0,
        "client_contract": client,
        "rank_audit": public_rank_audit,
        "attribution_diagnostics": diagnostics,
        "node_kernel_counts": _semantic_counts(attributed),
        "support_intervals": [
            {
                "support_class": row.get("support_class"),
                "support_reason": row.get("support_reason"),
                "duration_us": round(float(row["dur_us"]), 6),
                "attribution_method": row.get("attribution_method"),
            }
            for row in attributed
            if not row.get("node")
        ],
    }
    args.output_analysis.parent.mkdir(parents=True, exist_ok=True)
    args.output_analysis.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")

    profile = {
        "schema_version": "profile.v2",
        "profile_id": profile_id,
        "label": (
            f"NVIDIA GB300 · SGLang · pure TP8 · "
            f"{'CUDA Graph decode' if args.phase == 'decode' else 'eager prefill'} · "
            f"BS{args.batch_size} · 8k→1k"
        ),
        "model_id": "kimi_k3",
        "execution_path_id": "tp8",
        "implementation_id": IMPLEMENTATION_ID,
        "variant_id": variant_id,
        "phase": args.phase,
        "generation_mode": "autoregressive",
        "entry_view": "top",
        "execution_parameters": {"tp_size": 8, "dp_size": 1, "cp_size": 1, "ep_size": 1},
        "hardware": {"gpu": "GB300", "gpus_per_node": 4, "nodes": 2},
        "workload": {
            "isl": 8192,
            "osl": 1024,
            "batch_size": args.batch_size,
            "concurrency": args.batch_size,
            "warmup_requests": 3 * args.batch_size,
            "formal_requests": args.batch_size,
            "prompt_source": "deterministic_random_token_ids",
            "prompt_seed": 0,
            "ignore_eos": True,
            "no_intentionally_shared_prefix": True,
            "prefix_cache_enabled": False,
            "hicache_enabled": False,
            "kv_offload_enabled": False,
            "mtp_nextn_enabled": False,
            "modality": "text_only",
        },
        "profiler": {
            "type": "nsight_systems",
            "version": "2025.4.1",
            "representative_rank": 0,
            "cuda_graph_enabled": args.phase == "decode",
            "cuda_graph_trace": "node" if args.phase == "decode" else "not_applicable",
            "with_stack": False,
            "capture_control": {
                "trigger": "externally_armed_node_process_tree_session",
                "exact_window": "node_local_profiler_leader_window_plus_per_worker_launch_correlation",
                "node_collection_stop": "after_all_4_local_tp_workers_completed",
                "baseline_relative_start_step": args.baseline_relative_step,
                "num_steps": 1,
            },
            "selected_runtime_coordinate": client,
            "all_tp_ranks_validated": True,
            "gpu_metric_semantics": timing_summary["semantics"],
        },
        "evidence": {
            "capture_id": f"capture-{raw_hash[:16]}",
            "source_commit": SOURCE_COMMIT,
            "model_revision": MODEL_REVISION,
            "container": CONTAINER,
            "client_contract_sha256": client["sha256"],
            "exact_client_source_sha256": client["client_source_sha256"],
            "raw_trace_sha256": raw_hash,
            "eager_mapping_sha256": sha256_file(eager_mapping),
            "attribution_sha256": sha256_file(args.output_analysis),
            "validated_rank_count": 8,
            "mapped_kernel_count_ratio": diagnostics["mapped_kernel_count_ratio"],
            "mapped_kernel_duration_ratio": diagnostics["mapped_kernel_duration_ratio"],
            "mapping_policy": "186 ordered AttnRes anchors, occurrence-bounded eager semantic transfer, shape-preserving 1:N ownership, and explicit runtime-support classification",
            "attribution_diagnostics": diagnostics,
            "timing": timing_summary,
        },
        "timeline": {
            "schema_version": "timeline.v1",
            "artifact": timeline_path.name,
            "sha256": timeline_sha,
            "reference_rank": 0,
            "step_count": 1,
            "event_count": len(attributed),
            "raw_trace_file": f"capture-{raw_hash[:16]}.nsys-rep",
        },
        "node_states": node_states,
        "fusion_groups": fusion_groups,
        "node_metrics": node_metrics,
    }
    args.output_profile.parent.mkdir(parents=True, exist_ok=True)
    args.output_profile.write_text(
        yaml.safe_dump(profile, sort_keys=False, allow_unicode=True, width=1000)
    )
    print(
        json.dumps(
            {
                "state": "passed",
                "profile_id": profile_id,
                "event_count": len(attributed),
                "mapped_duration_ratio": diagnostics["mapped_kernel_duration_ratio"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
