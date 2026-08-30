#!/usr/bin/env python3
"""Build complete commit-locked SGLang and vLLM Qwen3.5 bindings."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from llm_arch_v2.compiler import apply_execution_plan


MODEL_PATH = REPO_ROOT / "catalog/qwen35/model_ir.yaml"
EXECUTION_PATH = REPO_ROOT / "catalog/qwen35/execution_paths/tp8.yaml"
CHECKPOINT_REVISION = "8f590eae8f10bf55d9a46f79ea0280bde435c9f8"


@dataclass(frozen=True)
class SourceSpec:
    file: str
    symbol: str
    line: int
    display: str


SGLANG = {
    "top": SourceSpec("python/sglang/srt/models/qwen3_5.py", "Qwen3_5MoeForCausalLM", 1655, "Qwen3.5 target wrapper, embedding, norm and logits"),
    "layer": SourceSpec("python/sglang/srt/models/qwen3_5.py", "Qwen3_5LinearDecoderLayer", 792, "Qwen3.5 hybrid decoder schedule and residual/norm flow"),
    "full": SourceSpec("python/sglang/srt/models/qwen3_5.py", "Qwen3_5AttentionDecoderLayer", 935, "Qwen3.5 full-attention layer"),
    "gdn": SourceSpec("python/sglang/srt/models/qwen3_5.py", "Qwen3_5GatedDeltaNet", 266, "Qwen3.5 GDN projections, state and output"),
    "gdn_backend": SourceSpec("python/sglang/srt/layers/attention/linear/gdn_backend.py", "GDNAttnBackend", 378, "GDN prefill/decode recurrence and state backend"),
    "moe": SourceSpec("python/sglang/srt/models/qwen2_moe.py", "Qwen2MoeSparseMoeBlock", 262, "router, routed experts, shared expert and combine"),
    "vision": SourceSpec("python/sglang/srt/models/qwen3_vl.py", "Qwen3VLMoeVisionModel", 345, "optional Qwen3 vision encoder inherited by Qwen3.5"),
    "mtp": SourceSpec("python/sglang/srt/models/qwen3_5_mtp.py", "Qwen3_5ForCausalLMMTP", 84, "optional Qwen3.5 MTP path, disabled by TP8 Stage 1"),
    "all_reduce": SourceSpec("python/sglang/srt/distributed/communication_op.py", "tensor_model_parallel_all_reduce", 18, "TP all-reduce contract"),
    "all_gather": SourceSpec("python/sglang/srt/distributed/communication_op.py", "tensor_model_parallel_all_gather", 77, "TP all-gather contract"),
}

VLLM = {
    "top": SourceSpec("vllm/model_executor/models/qwen3_5.py", "Qwen3_5ForCausalLMBase", 287, "Qwen3.5 target wrapper, embedding, norm and logits"),
    "layer": SourceSpec("vllm/model_executor/models/qwen3_5.py", "Qwen3_5DecoderLayer", 119, "Qwen3.5 hybrid decoder construction with Qwen3Next layer flow"),
    "full": SourceSpec("vllm/model_executor/models/qwen3_next.py", "Qwen3NextAttention", 215, "full attention projection, QK norm, RoPE, cache and gate"),
    "gdn": SourceSpec("vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py", "QwenGatedDeltaNetAttention", 345, "Qwen3.5 GDN projections, recurrence state and output"),
    "gdn_backend": SourceSpec("vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py", "ChunkGatedDeltaRule", 216, "GDN prefill/decode recurrence backend"),
    "moe": SourceSpec("vllm/model_executor/models/qwen3_next.py", "Qwen3NextSparseMoeBlock", 101, "router, routed experts, shared expert and combine"),
    "vision": SourceSpec("vllm/model_executor/models/qwen3_vl.py", "Qwen3VLForConditionalGeneration", 1733, "optional Qwen3 vision encoder inherited by Qwen3.5"),
    "mtp": SourceSpec("vllm/model_executor/models/qwen3_5_mtp.py", "Qwen3_5MultiTokenPredictor", 64, "optional Qwen3.5 MTP path, disabled by TP8 Stage 1"),
    "all_reduce": SourceSpec("vllm/model_executor/layers/linear.py", "RowParallelLinear", 1504, "row-parallel TP all-reduce contract"),
    "all_gather": SourceSpec("vllm/model_executor/layers/logits_processor.py", "LogitsProcessor", 23, "TP vocabulary all-gather contract"),
}


def category(node_id: str) -> str:
    if node_id.endswith("tp_logits_all_gather"):
        return "all_gather"
    if ".tp_" in node_id:
        return "all_reduce"
    if node_id.startswith(("vision_frontend.", "vision_block.")) or node_id in {
        "top.vision_inputs",
        "top.vision_frontend",
        "top.multimodal_injection",
    }:
        return "vision"
    if node_id.startswith(("mtp_", "generation_loop.")) or node_id in {
        "top.generation_controller",
        "top.accepted_tokens",
    } or node_id.startswith("state_tensors.mtp_") or node_id == "state_tensors.verify_journal":
        return "mtp"
    if node_id.startswith("gdn_attention."):
        return "gdn_backend" if any(token in node_id for token in ("state", "causal_conv", "recurrence")) else "gdn"
    if node_id.startswith("full_attention."):
        return "full"
    if node_id.startswith("moe_block."):
        return "moe"
    if node_id.startswith(("gdn_moe_block.", "full_attention_moe_block.", "stack.", "layer_schedule.")):
        return "layer"
    if node_id.startswith("state_tensors.gdn_"):
        return "gdn_backend"
    if node_id.startswith("state_tensors.attention_"):
        return "full"
    return "top"


def node_ids() -> list[str]:
    model = yaml.safe_load(MODEL_PATH.read_text())
    plan = yaml.safe_load(EXECUTION_PATH.read_text())
    views = apply_execution_plan(model, plan, source=EXECUTION_PATH)
    return [f"{view_id}.{node['id']}" for view_id, view in views.items() for node in view["nodes"]]


def binding_entry(spec: SourceSpec) -> dict[str, Any]:
    return {
        "symbols": [spec.symbol],
        "links": [
            {
                "file": spec.file,
                "symbol": spec.symbol,
                "line": spec.line,
                "display": spec.display,
            }
        ],
    }


def build(framework: str, status: str, evidence: list[str]) -> dict[str, Any]:
    is_sglang = framework == "sglang"
    specs = SGLANG if is_sglang else VLLM
    implementation_id = (
        "sglang_f609d677b_qwen35_033446bb_tp8"
        if is_sglang
        else "vllm_487ecf187_qwen35_native_tp8"
    )
    source_commit = (
        "f609d677b909ca46c64bb6803b69a85fedbf86bc"
        if is_sglang
        else "487ecf187d3dfe74d2cf6119a92881dba403c219"
    )
    container = (
        "sglang-glm53-flash-arm64-73f9294b.sqsh@sha256:28e9545e312e344bbbf80c575b928be53c9aba6296ae55f292ce0f10750c6971"
        if is_sglang
        else "vllm-glm53-flash-arm64-905c0293.sqsh@sha256:efdfe25952dc672d4415032e2755df7d7f2bab549992a2e3f2c429334f366756"
    )
    result: dict[str, Any] = {
        "schema_version": "implementation-binding.v2",
        "implementation_id": implementation_id,
        "label": (
            "SGLang f609d677b with Qwen3.5 033446bb modules, pure TP8"
            if is_sglang
            else "vLLM 487ecf187 native Qwen3.5 pure TP8"
        ),
        "model_id": "qwen35",
        "execution_path_id": "tp8",
        "source_repo": (
            "https://github.com/sgl-project/sglang"
            if is_sglang
            else "https://github.com/vllm-project/vllm"
        ),
        "source_commit": source_commit,
        "container": container,
        "backend": "native_multi_node_tp8",
        "binding_status": status,
        "source_lock_status": "runtime_verified",
        "execution_validation": {
            "status": "pass" if status == "validated" else "pending",
            "execution_fingerprint": "exec_56198943adacd2b6",
            "required_phases": ["prefill", "decode"],
            "semantic_capture_cuda_graph_enabled": False,
            "production_decode_cuda_graph_enabled": True,
            "evidence": evidence,
            "notes": (
                "The package base is f609d677b while the Qwen3.5 model, config, GDN and MoE modules match inspected 033446bb source byte-for-byte; the mixed snapshot is explicit."
                if is_sglang
                else "The package and all inspected Qwen3.5 modules match vLLM commit 487ecf187 byte-for-byte."
            ),
        },
        "node_bindings": {
            node_id: binding_entry(specs[category(node_id)]) for node_id in node_ids()
        },
    }
    if is_sglang:
        result["runtime_module_source_commit"] = "033446bb05f35c0943aed2750c443077ffc0b92c"
        result["runtime_module_sha256"] = {
            "python/sglang/srt/configs/qwen3_5.py": "53f92b2be9d880716a1850cb4484ea2fa46408ab0b05be847ce0fa88edf71890",
            "python/sglang/srt/models/qwen3_5.py": "0981d3b9bf4fef815511f02e39843af28dfb0cbf2bf0011a08151984abd7b197",
            "python/sglang/srt/layers/attention/linear/gdn_backend.py": "c607506f003a22635156d5726c37e9fb56e1d42e784d768eee8035a2820f6251",
            "python/sglang/srt/layers/moe/fused_moe_triton/layer.py": "2c00940d386430c9bfc47d0dd4c8d62a8be68ad00ba36c082c1bf58b9697f616",
        }
    result["checkpoint_revision"] = CHECKPOINT_REVISION
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", choices=("provisional", "validated"), default="provisional")
    parser.add_argument("--evidence", action="append", default=[])
    args = parser.parse_args()
    out = REPO_ROOT / "catalog/qwen35/bindings"
    out.mkdir(parents=True, exist_ok=True)
    for framework, filename in (
        ("sglang", "sglang-f609d677b-qwen35-033446bb-tp8.yaml"),
        ("vllm", "vllm-487ecf187-qwen35-native-tp8.yaml"),
    ):
        payload = build(framework, args.status, list(args.evidence))
        (out / filename).write_text(yaml.safe_dump(payload, sort_keys=False, width=120))


if __name__ == "__main__":
    main()
