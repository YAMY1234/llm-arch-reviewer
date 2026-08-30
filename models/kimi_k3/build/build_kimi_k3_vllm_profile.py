#!/usr/bin/env python3
"""Build one fail-closed Kimi K3 vLLM pure-TP8 production profile."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any

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
from models.kimi_k3.build.build_kimi_k3_sglang_profile import (
    _union_duration_us,
    build_node_metrics,
)
from models.kimi_k3.build.kimi_k3_profile_contract import (
    build_node_states,
    vllm_fusion_groups,
)
from models.kimi_k3.build.kimi_k3_vllm_production_attribution import (
    ATTN_RES_ANCHOR_COUNT,
    attribute_vllm_production_events,
)
from models.kimi_k3.build.kimi_k3_vllm_profile_evidence import (
    read_exact_worker_kernels,
    validate_production_client,
)


MODEL_REVISION = "a590ce090cb049c93a33dfe8c208ec652aa20503"
SOURCE_COMMIT = "680e2177e473ed8dfaa9773f7ead185b369cab46"
CONTAINER = (
    "vllm/vllm-openai@"
    "sha256:d61eb329832ea78aeae233c840a78b5022dca5f1200df12cb54f3f2304f63131"
)
EXTENSION_SHA256 = "80b923d451d58a731fb950af6baba749757d7e7c177ddb86a04e5ae4ee770c8d"
IMPLEMENTATION_ID = "vllm_680e2177_kimi_k3_tp8"
CANONICAL_FUSION_OWNERS = {
    "kda.tp_kda_output_collective": "kda.output_projection",
    "gated_mla.tp_mla_output_collective": "gated_mla.output_projection",
    "dense_mlp.tp_dense_output_collective": "dense_mlp.down",
    "top.logits": "top.tp_logits_materialization",
}


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


def apply_canonical_fusion_owners(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Assign every fused physical interval to its single measured owner."""

    out: list[dict[str, Any]] = []
    for row in rows:
        original = row.get("node")
        owner = CANONICAL_FUSION_OWNERS.get(str(original))
        if not owner:
            out.append(row)
            continue
        updated = dict(row)
        updated["semantic_child"] = original
        updated["node"] = owner
        updated["attribution_method"] = (
            f"{row.get('attribution_method', 'exact_eager_mapping')}+"
            "canonical_single_fusion_owner"
        )
        out.append(updated)
    return out


def _rank_source(root: Path, rank: int) -> tuple[Path, Path, int]:
    node_rank = rank // 4
    device = rank % 4
    sqlite_path = root / "nsys" / f"node-rank{node_rank}.sqlite"
    rep_path = root / "nsys" / f"node-rank{node_rank}.nsys-rep"
    require(rep_path.is_file(), f"missing raw Nsight report: {rep_path}")
    return sqlite_path, rep_path, device


def _semantic_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(
        sorted(Counter(str(row["node"]) for row in rows if row.get("node")).items())
    )


def _validate_eager_rank(root: Path, rank: int, phase: str) -> Path:
    mapping_root = root / "mapping" / f"tp{rank}"
    mapping_path = mapping_root / f"kernel_mapping.tp{rank}.jsonl"
    manifest_path = mapping_root / "input_manifest.json"
    validation_path = mapping_root / "validation_report.json"
    for path in (mapping_path, manifest_path, validation_path):
        require(path.is_file(), f"missing eager TP{rank} artifact: {path}")
    manifest = load_json(manifest_path)
    validation = load_json(validation_path)
    require(manifest.get("source_commit") == SOURCE_COMMIT, "eager source mismatch")
    require(manifest.get("rank") == rank, "eager rank mismatch")
    require(manifest.get("phase") == f"vllm_{phase}", "eager phase mismatch")
    require(validation.get("ok") is True, "eager mapping did not pass")
    require(not validation.get("errors"), "eager mapping has errors")
    require(not validation.get("warnings"), "eager mapping has warnings")
    require(not validation.get("top_unmatched"), "eager mapping has unmatched kernels")
    require(validation.get("mapped_duration_ratio") == 1.0, "eager mapping ratio")
    require(validation.get("stack_duration_ratio") == 1.0, "eager stack ratio")
    return mapping_path


def validate_and_attribute_ranks(
    *, production_root: Path, eager_root: Path, phase: str
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    rank_results: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] | None = None
    reference_diagnostics: dict[str, Any] | None = None
    count_fingerprints: set[str] = set()
    for rank in range(8):
        sqlite_path, rep_path, device = _rank_source(production_root, rank)
        eager_mapping = _validate_eager_rank(eager_root, rank, phase)
        production, exact_window = read_exact_worker_kernels(sqlite_path, device)
        attributed, diagnostics = attribute_vllm_production_events(
            production, eager_mapping
        )
        require(diagnostics["anchor_count"] == ATTN_RES_ANCHOR_COUNT, "anchor closure")
        require(
            all(row.get("node") or row.get("support_class") for row in attributed),
            "production interval lacks semantic or runtime-support ownership",
        )
        graph_kernel_count = sum(
            1 for row in attributed if row.get("graph_node_id") is not None
        )
        mapped_graph_kernel_count = sum(
            1
            for row in attributed
            if row.get("node") and row.get("graph_node_id") is not None
        )
        if phase == "decode":
            require(graph_kernel_count > 0, "decode production window has no graph nodes")
            require(
                mapped_graph_kernel_count > 0,
                "decode production graph has no mapped semantic kernels",
            )
        else:
            require(graph_kernel_count == 0, "prefill production window contains graph nodes")
        counts = _semantic_counts(attributed)
        count_fingerprints.add(json.dumps(counts, sort_keys=True))
        rank_results.append(
            {
                "rank": rank,
                "node_rank": rank // 4,
                "local_device": device,
                "kernel_count": len(attributed),
                "graph_kernel_count": graph_kernel_count,
                "eager_break_kernel_count": len(attributed) - graph_kernel_count,
                "mapped_kernel_count": diagnostics["mapped_kernel_count"],
                "support_kernel_count": diagnostics["support_kernel_count"],
                "mapped_kernel_duration_ratio": diagnostics[
                    "mapped_kernel_duration_ratio"
                ],
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
    require(len(count_fingerprints) == 1, "semantic production counts differ by TP rank")
    assert reference_rows is not None and reference_diagnostics is not None
    return reference_rows, reference_diagnostics, {
        "schema_version": "kimi-k3-production-rank-audit.v1",
        "state": "passed",
        "framework": "vllm",
        "source_commit": SOURCE_COMMIT,
        "phase": phase,
        "all_tp_ranks_validated": True,
        "phase_shape_rank_source_exact": True,
        "ranks": rank_results,
    }


def profile_identity(phase: str, batch_size: int) -> tuple[str, str]:
    if phase == "prefill":
        return "kimi_k3_tp8_vllm_prefill_bs1_8k", "prefill_bs1_8k"
    return (
        f"kimi_k3_tp8_vllm_cg_decode_bs{batch_size}_8k1k",
        f"cg_decode_bs{batch_size}_8k1k",
    )


def main() -> int:
    args = parse_args()
    if args.phase == "prefill":
        require(args.batch_size == 1, "prefill is accepted only at batch 1")
    else:
        require(
            args.batch_size in {1, 16, 64},
            "vLLM decode accepts 1/16/64; 256 has explicit unsupported evidence",
        )
    production_root = args.production_root.resolve()
    eager_root = args.eager_root.resolve()
    client = validate_production_client(
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
    attributed = apply_canonical_fusion_owners(attributed)
    node_metrics = build_node_metrics(attributed)
    measured_nodes = set(node_metrics)
    fusion_groups = vllm_fusion_groups(
        phase=args.phase,
        batch_size=args.batch_size,
        measured_nodes=measured_nodes,
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
    trace_stop_us = max(
        float(row["ts_us"]) + float(row["dur_us"]) for row in attributed
    )
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
        "semantics": (
            "same-device interval union and residency for one exact vLLM "
            "worker-local captured formal forward"
        ),
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
            "policy": (
                "187-AttnRes-occurrence-bounded normalized identity and ordinal; "
                "exact eager provenance retained per mapped event"
            ),
        },
    )
    timeline_sha = write_timeline_artifact(timeline_path, timeline)

    public_rank_audit = {
        "schema_version": rank_audit["schema_version"],
        "state": rank_audit["state"],
        "framework": "vllm",
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
                    "graph_kernel_count",
                    "eager_break_kernel_count",
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
        "schema_version": "kimi-k3-vllm-production-attribution.v1",
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
    args.output_analysis.write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n"
    )

    profile = {
        "schema_version": "profile.v2",
        "profile_id": profile_id,
        "label": (
            f"NVIDIA GB300 · vLLM · pure TP8 · "
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
                "trigger": "vllm_all_tp_worker_profiler_start_stop",
                "outer_session": "externally_armed_node_process_tree",
                "exact_window": "worker_local_profiler_api_window_plus_launch_correlation",
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
            "compiled_extension_sha256": EXTENSION_SHA256,
            "client_contract_sha256": client["sha256"],
            "exact_client_source_sha256": client["client_source_sha256"],
            "raw_trace_sha256": raw_hash,
            "eager_mapping_sha256": sha256_file(eager_mapping),
            "attribution_sha256": sha256_file(args.output_analysis),
            "validated_rank_count": 8,
            "mapped_kernel_count_ratio": diagnostics["mapped_kernel_count_ratio"],
            "mapped_kernel_duration_ratio": diagnostics["mapped_kernel_duration_ratio"],
            "mapping_policy": (
                "187 ordered AttnRes anchors, occurrence-bounded eager semantic "
                "transfer, explicit eager-break ownership, and runtime-support classification"
            ),
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
