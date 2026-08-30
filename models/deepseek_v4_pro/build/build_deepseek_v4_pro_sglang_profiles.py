#!/usr/bin/env python3
"""Build DeepSeek-V4-Pro-0813 SGLang production profiles through the shared compiler."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.deepseek_v4_pro.build.build_deepseek_v4_pro_vllm_profiles import (
    build_one,
    compiled_nodes,
    file_sha256,
    load_json,
)


SOURCE_COMMIT = "71de97b264b04dcd514cf904003028aefe9775c8"
CONTAINER_SHA256 = "ddec5cc59fa15be11e0b0b06de381cc2e0588e2d6098bd86f12a83cb4b1d58e2"
IMPLEMENTATION_ID = "sglang_71de97b_dsv4pro0813_tp8"
MATRIX_REPORT_SHA256 = "8e32e6acc965680634673621c98c020c70f1a2cc75c646b4cd91a76a24881a53"
MATRIX_MANIFEST_SHA256 = "e655f54234f118fc28819233e50d3703b8d14f4e4a8b74c5558138f1f84aa064"

PROFILE_SPECS = {
    "prefill-c1": {
        "phase": "prefill",
        "batch_size": 1,
        "job_id": "3422982",
        "eager_job_id": "3422245",
        "eager_kind": "sglang-eager-prefill",
        "production_kind": "sglang-prefill_timing",
        "variant_id": "eager_prefill_gbs001_8k",
        "file_stem": "eager_prefill_gbs001_8k",
    },
    "decode-c1": {
        "phase": "decode",
        "batch_size": 1,
        "job_id": "3422983",
        "eager_job_id": "3421642",
        "eager_kind": "sglang-eager-decode",
        "production_kind": "sglang-production",
        "variant_id": "cg_decode_gbs001_8k1k",
        "file_stem": "cg_decode_gbs001_8k1k",
    },
    "decode-c16": {
        "phase": "decode",
        "batch_size": 16,
        "job_id": "3422984",
        "eager_job_id": "3421643",
        "eager_kind": "sglang-eager-decode",
        "production_kind": "sglang-production",
        "variant_id": "cg_decode_gbs016_8k1k",
        "file_stem": "cg_decode_gbs016_8k1k",
    },
    "decode-c64": {
        "phase": "decode",
        "batch_size": 64,
        "job_id": "3422985",
        "eager_job_id": "3421644",
        "eager_kind": "sglang-eager-decode",
        "production_kind": "sglang-production",
        "variant_id": "cg_decode_gbs064_8k1k",
        "file_stem": "cg_decode_gbs064_8k1k",
    },
    "decode-c256": {
        "phase": "decode",
        "batch_size": 256,
        "job_id": "3422986",
        "eager_job_id": "3421645",
        "eager_kind": "sglang-eager-decode",
        "production_kind": "sglang-production",
        "variant_id": "cg_decode_gbs256_8k1k",
        "file_stem": "cg_decode_gbs256_8k1k",
    },
}


def fusion_specs() -> dict[str, dict]:
    """SGLang-specific N:1 semantic groups with exactly one timing owner."""

    groups: dict[str, dict] = {
        "sglang_mhc_pre_and_norm": {
            "owner": "mhc_transform.affine",
            "ir_nodes": [
                "mhc_transform.flatten_rms",
                "mhc_transform.affine",
                "mhc_transform.pre_gate",
                "mhc_transform.post_gate",
                "mhc_transform.combine_sinkhorn",
                "mhc_transform.read",
                "decoder_stack.attention_norm",
                "decoder_stack.ffn_norm",
            ],
            "source_nodes": {"mhc_transform.affine"},
            "proof": "SGLang mHC prenorm GEMM plus mhc_pre_big_fuse_with_norm exact eager sequence",
        },
        "sglang_mhc_post": {
            "owner": "mhc_transform.mix",
            "ir_nodes": ["mhc_transform.place", "mhc_transform.mix"],
            "source_nodes": {"mhc_transform.mix"},
            "proof": "SGLang mhc_post or fused_post_pre first-kernel equation contract",
        },
        "sglang_final_hc_read": {
            "owner": "final_hc_read.read",
            "ir_nodes": [
                "final_hc_read.flatten_rms",
                "final_hc_read.pre_gate",
                "final_hc_read.read",
            ],
            "source_nodes": {"final_hc_read.read"},
            "proof": "SGLang _hc_head_kernel fused normalization, gate, and residual-stream read",
        },
        "sglang_csa_indexer_q_rope": {
            "owner": "csa_indexer.q_projection",
            "ir_nodes": ["csa_indexer.q_projection", "csa_indexer.q_rope_rotate"],
            "source_nodes": {"csa_indexer.q_projection"},
            "proof": "exact eager q projection followed by fused_q_indexer_rope_hadamard_quant",
        },
        "sglang_csa_indexer_history": {
            "owner": "csa_indexer.k_compress",
            "ir_nodes": ["csa_indexer.k_compress", "csa_indexer.selected_ids"],
            "source_nodes": {"csa_indexer.k_compress"},
            "proof": "exact eager C4 compression and fused norm/RoPE indexer cache update",
        },
        "sglang_csa_compressor_state": {
            "owner": "csa_compressor.partial_state",
            "ir_nodes": [
                "csa_compressor.norm_rope",
                "csa_compressor.partial_state",
                "csa_compressor.compressed_cache",
            ],
            "source_nodes": {"csa_compressor.partial_state"},
            "proof": "fused_norm_rope_flashmla exact C4 partial-state and compressed-cache update",
        },
        "sglang_hca_compressor_state": {
            "owner": "hca_compressor.partial_state",
            "ir_nodes": [
                "hca_compressor.norm_rope",
                "hca_compressor.partial_state",
                "hca_compressor.compressed_cache",
            ],
            "source_nodes": {"hca_compressor.partial_state"},
            "proof": "fused_norm_rope_flashmla exact C128 partial-state and compressed-cache update",
        },
        "sglang_router_paths": {
            "owner": "moe.score_projection",
            "ir_nodes": [
                "moe.score_projection",
                "moe.sqrt_softplus",
                "moe.hash_select",
                "moe.learned_select",
                "moe.weights",
            ],
            "source_nodes": {
                "moe.score_projection",
                "moe.hash_select",
                "moe.learned_select",
            },
            "proof": "profile aggregate retains the separate score projection and exact hash/learned fused selection events with layer IDs",
        },
        "sglang_routed_gate_up_swiglu": {
            "owner": "moe.routed_gate_up",
            "ir_nodes": ["moe.routed_gate_up", "moe.routed_activation"],
            "source_nodes": {"moe.routed_gate_up"},
            "proof": "DeepGEMM routed gate/up BMM clmp_swiGlu epilogue signature",
        },
    }
    for prefix in ("csa", "hca"):
        groups[f"sglang_{prefix}_q_head_rope"] = {
            "owner": f"{prefix}_attention.q_head_norm",
            "ir_nodes": [
                f"{prefix}_attention.q_head_norm",
                f"{prefix}_attention.q_rope",
            ],
            "source_nodes": {f"{prefix}_attention.q_head_norm"},
            "proof": "fused_q_norm_rope exact eager kernel signature",
        }
        groups[f"sglang_{prefix}_window_kv_cache"] = {
            "owner": f"{prefix}_attention.window_kv",
            "ir_nodes": [
                f"{prefix}_attention.window_kv",
                f"{prefix}_attention.window_cache",
            ],
            "source_nodes": {f"{prefix}_attention.window_kv"},
            "proof": "fused_k_norm_rope_flashmla exact eager KV normalization, RoPE, quantization, and cache update",
        }
        groups[f"sglang_{prefix}_index_union"] = {
            "owner": f"{prefix}_attention.index_union",
            "ir_nodes": [
                f"{prefix}_attention.window_indices",
                f"{prefix}_attention.index_union",
            ],
            "source_nodes": {f"{prefix}_attention.index_union"},
            "proof": "exact window-plus-compressed-history index union kernels where the shape selects this path",
        }
    return groups


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    task_root = args.task_root.resolve()
    output_dir = args.output_dir or (
        repo_root / "catalog/deepseek_v4_pro/profiles/tp8/sglang_71de97b_dsv4pro0813_tp8"
    )
    model_ir = yaml.safe_load((repo_root / "catalog/deepseek_v4_pro/model_ir.yaml").read_text())
    execution = yaml.safe_load(
        (repo_root / "catalog/deepseek_v4_pro/execution_paths/tp8_moe_intermediate_shard.yaml").read_text()
    )
    nodes = compiled_nodes(model_ir, execution)
    if len(nodes) != 153:
        raise ValueError(f"expected 153 execution nodes, got {len(nodes)}")
    matrix_path = task_root / "production-reconciliation/sglang/matrix_report.json"
    if file_sha256(matrix_path) != MATRIX_REPORT_SHA256:
        raise ValueError("production matrix report hash does not match the accepted gate")
    matrix = load_json(matrix_path)
    if not matrix.get("ok") or matrix.get("profile_count") != 5:
        raise ValueError("production matrix did not pass all five SGLang profiles")

    outputs = []
    for name, spec in PROFILE_SPECS.items():
        path, digest = build_one(
            repo_root=repo_root,
            task_root=task_root,
            output_dir=output_dir,
            name=name,
            spec=spec,
            matrix=matrix,
            nodes=nodes,
            reconciliation_framework="sglang",
            profile_framework="sglang",
            framework_label="SGLang",
            implementation_id=IMPLEMENTATION_ID,
            source_commit=SOURCE_COMMIT,
            container_sha256=CONTAINER_SHA256,
            matrix_report_sha256=MATRIX_REPORT_SHA256,
            matrix_manifest_sha256=MATRIX_MANIFEST_SHA256,
            fusion_spec_map=fusion_specs(),
            trace_pattern="*TP-{rank}.trace.json.gz",
        )
        try:
            reported_path = path.relative_to(repo_root)
        except ValueError:
            reported_path = path
        outputs.append({"path": str(reported_path), "sha256": digest})
    print(json.dumps({"ok": True, "profiles": outputs}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
