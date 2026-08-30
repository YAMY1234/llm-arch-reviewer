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
MATRIX_REPORT_SHA256 = "27fc33f8929b51309d9265c7a13c5195bc9c7a42c8a915d377269a9287e4523b"
MATRIX_MANIFEST_SHA256 = "40f6168af097f274d26b0fc86ad355e8267b4dd3c95e0d03f129558e318a3aaf"
PROFILER_OVERLAY_V1 = {
    "base_source_commit": SOURCE_COMMIT,
    "source_lock_file": "overlays/sglang-tp-sync-profiler/source-lock.json",
    "source_lock_sha256": "0bb67bf626256b34cd3c5d69254a595323fc014d57a345afcd9be915caaa58e1",
    "scheduler_sha256": "ca39957402cc383aaed01ec4749b03c00eb7e3967001da0e517faf25ad328391",
    "profiler_manager_sha256": "31e2b1c19a9901233a3f28b15289f76a0932786b05232c185d6035f80781792d",
    "scope": "profiler activation and post-input-preparation TP CPU barriers only",
}
PROFILER_OVERLAY_V2 = {
    "base_source_commit": SOURCE_COMMIT,
    "source_lock_file": "overlays/sglang-tp-sync-profiler/source-lock.json",
    "source_lock_sha256": "d235d5e41c5a3926cc7500bb2e8d79f6e311e37dffb812242eef4144aea53702",
    "scheduler_sha256": "ca39957402cc383aaed01ec4749b03c00eb7e3967001da0e517faf25ad328391",
    "profiler_manager_sha256": "f51b5a5928656362731a81624126a6f6bdd8f821a05710c11cca5bcf662ceb7e",
    "scope": "profiler activation and post-input-preparation TP CPU barriers with per-rank CUDA backlog drains before each barrier",
}
PROFILER_OVERLAY_V3 = {
    "base_source_commit": SOURCE_COMMIT,
    "source_lock_file": "overlays/sglang-tp-sync-profiler/source-lock.json",
    "source_lock_sha256": "060e008ce5724ebb5d073d5269ad3db3c13c3203db0ee828ab870bed072e8ddb",
    "scheduler_sha256": "ca39957402cc383aaed01ec4749b03c00eb7e3967001da0e517faf25ad328391",
    "profiler_manager_sha256": "82d74e7caace8379aecde00ea5ca91afb392bc369676f31f830653c0c5bba582",
    "scope": "activation-prime and formal prefill input fences with per-rank CUDA backlog drains before each TP CPU barrier",
}

PROFILE_SPECS = {
    "prefill-c1": {
        "phase": "prefill",
        "batch_size": 1,
        "job_id": "3426447",
        "eager_job_id": "3422245",
        "eager_kind": "sglang-eager-prefill",
        "production_kind": "sglang-prefill_timing",
        "variant_id": "eager_prefill_gbs001_8k",
        "file_stem": "eager_prefill_gbs001_8k",
    },
    "decode-c1": {
        "phase": "decode",
        "batch_size": 1,
        "job_id": "3424801",
        "eager_job_id": "3421642",
        "eager_kind": "sglang-eager-decode",
        "production_kind": "sglang-production",
        "variant_id": "cg_decode_gbs001_8k1k",
        "file_stem": "cg_decode_gbs001_8k1k",
    },
    "decode-c16": {
        "phase": "decode",
        "batch_size": 16,
        "job_id": "3424802",
        "eager_job_id": "3421643",
        "eager_kind": "sglang-eager-decode",
        "production_kind": "sglang-production",
        "variant_id": "cg_decode_gbs016_8k1k",
        "file_stem": "cg_decode_gbs016_8k1k",
    },
    "decode-c64": {
        "phase": "decode",
        "batch_size": 64,
        "job_id": "3424803",
        "eager_job_id": "3421644",
        "eager_kind": "sglang-eager-decode",
        "production_kind": "sglang-production",
        "variant_id": "cg_decode_gbs064_8k1k",
        "file_stem": "cg_decode_gbs064_8k1k",
    },
    "decode-c256": {
        "phase": "decode",
        "batch_size": 256,
        "job_id": "3424804",
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
    if (
        not matrix.get("ok")
        or matrix.get("profile_count") != 5
        or matrix.get("measured_profile_count") != 4
        or matrix.get("unsupported_profile_count") != 1
    ):
        raise ValueError(
            "production matrix must retain four measured SGLang profiles and "
            "one evidence-backed unsupported prefill contract"
        )

    outputs = []
    for name, spec in PROFILE_SPECS.items():
        if matrix["profiles"][name].get("status") == "unsupported":
            outputs.append(
                {
                    "profile_contract": name,
                    "status": "unsupported",
                    "reason": matrix["profiles"][name]["unsupported_reason"],
                }
            )
            continue
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
            source_overlay=(
                PROFILER_OVERLAY_V3
                if spec["phase"] == "prefill"
                else PROFILER_OVERLAY_V1
            ),
            mapping_root_name="mappings",
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
