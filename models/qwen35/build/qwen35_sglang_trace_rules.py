#!/usr/bin/env python3
"""SGLang eager-trace rules for the canonical Qwen3.5 TP8 IR.

Only implementation-unique kernels or source-stack-scoped operations receive a
semantic leaf.  The rules intentionally exclude the older AgentX DEP/MTP
contract: this module describes the pure-TP target forward captured at SGLang
commit 033446bb05f35c0943aed2750c443077ffc0b92c.
"""

from __future__ import annotations

from models.common.trace_mapping import FrameRef, StackFrameRules, TraceMappingRules


QWEN35_GDN_SIGNATURE = "fused_qkvzba_split"
QWEN35_GDN_LAYERS = 45


def _has_any(value: str, *needles: str) -> bool:
    return any(needle in value for needle in needles)


def classify_qwen35_sglang_node(
    kernel_name: str,
    cpu_op_name: str | None,
    stack: list[FrameRef],
) -> tuple[str | None, str]:
    names = "\n".join(frame.raw for frame in stack).lower()
    kernel = kernel_name.lower()
    cpu = (cpu_op_name or "").lower()

    is_all_gather = _has_any(kernel, "allgather", "all_gather")
    is_all_reduce = _has_any(
        kernel,
        "allreduce",
        "all_reduce",
        "oneshotallreducefusionkernel",
    ) or "flashinfer_allreduce" in cpu
    if is_all_gather and _has_any(names, "logitsprocessor", "lm_head"):
        return "top.tp_logits_all_gather", "high"
    if is_all_reduce:
        if "vocabparallelembedding" in names:
            return "top.tp_embedding_output_collective", "high"
        if "qwen3_5gateddeltanet" in names or "gdn_backend.py" in names:
            return "gdn_attention.tp_gdn_output_collective", "high"
        if "qwen3_5attentiondecoderlayer" in names:
            return "full_attention.tp_attention_output_collective", "high"
        if _has_any(names, "qwen2moesparsemoeblock", "forward_normal"):
            return "moe_block.tp_moe_output_collective", "high"

    # Model-unique GDN kernels override an occasionally stale neighboring
    # Python range because their implementation identity is unambiguous.
    if QWEN35_GDN_SIGNATURE in kernel:
        return "gdn_attention.qkvz_projection", "high"
    if "causal_conv1d" in kernel:
        return "gdn_attention.causal_conv", "high"
    if _has_any(
        kernel,
        "fused_recurrent_gated_delta_rule",
        "gdn_decode",
        "gdn_wide_vec_kernel",
        "chunk_gated_delta_rule",
        "recompute_w_u",
        "chunk_fwd_kernel_o",
        "l2norm_fwd",
    ):
        return "gdn_attention.gated_delta_recurrence", "high"
    if _has_any(kernel, "fused_gdn_gating", "layer_norm_gated"):
        return "gdn_attention.output_gate_norm", "high"
    if "qwen3_5gateddeltanet" in names:
        if "out_proj" in names or "rowparallellinear" in names:
            return "gdn_attention.output_projection", "high"
        if "in_proj_ba" in names:
            return "gdn_attention.ba_projection", "high"
        if "in_proj_qkvz" in names or "_forward_input_proj" in names:
            return "gdn_attention.qkvz_projection", "high"
        if _has_any(kernel, "rmsnorm", "layer_norm"):
            return "gdn_attention.output_gate_norm", "medium"

    if "qwen3_5attentiondecoderlayer" in names:
        if _has_any(kernel, "fmha", "mha", "attention") or "radixattention" in names:
            return "full_attention.causal_gqa", "high"
        if "o_proj" in names or "rowparallellinear" in names:
            return "full_attention.output_projection", "high"
        if _has_any(names, "qkv", "forward_prepare") and cpu in {
            "aten::mm",
            "aten::matmul",
        }:
            return "full_attention.qkv_projection", "high"
        if _has_any(kernel, "rope", "rotary"):
            return "full_attention.partial_rope", "high"
        if _has_any(kernel, "rmsnorm", "layer_norm"):
            return "full_attention.qk_norm", "medium"

    in_moe = _has_any(names, "qwen2moesparsemoeblock", "fusedmoe", "forward_normal")
    if in_moe:
        if _has_any(kernel, "routing", "topk") or "topk" in cpu:
            return "moe_block.router", "high"
        if "qwen2moemlp" in names or "_forward_shared_experts" in names:
            return "moe_block.shared_expert", "high"
        if _has_any(kernel, "deep_gemm", "cutedsl", "moe", "bmm_", "grouped"):
            return "moe_block.routed_experts", "high"
        if "finalize" in kernel or cpu == "aten::add":
            return "moe_block.weighted_combine", "high"
        if "gate" in names and _has_any(kernel + cpu, "gemm", "matmul", "aten::mm"):
            return "moe_block.router", "medium"

    if "fused_add_rmsnorm" in kernel:
        # Sequence reconciliation assigns the first and second residual/norm
        # occurrence within each layer to their exact leaves.
        return None, "unmapped"
    if "logitsprocessor" in names or "lm_head" in names:
        return "top.lm_head", "medium"
    if "vocabparallelembedding" in names or "get_input_embeddings" in names:
        return "top.embedding", "medium"
    if "qwen3_5forcausallm" in names and _has_any(kernel, "rmsnorm", "layer_norm"):
        return "top.final_norm", "medium"
    return None, "unmapped"


QWEN35_SGLANG_TRACE_RULES = TraceMappingRules(
    model_id="qwen35_sglang_tp8",
    signature_kernel=QWEN35_GDN_SIGNATURE,
    signature_count_per_forward=QWEN35_GDN_LAYERS,
    stack=StackFrameRules(
        operator_patterns=(
            "layers/linear.py",
            "layers/quantization",
            "layers/attention",
            "layers/moe",
            "layers/attention/linear/gdn_backend.py",
            "radix_attention.py",
            "layers/logits_processor.py",
            "layers/vocab_parallel_embedding.py",
        ),
        semantic_patterns=(
            "models/qwen3_5.py",
            "models/qwen2_moe.py",
            "layers/attention/linear/gdn_backend.py",
            "layers/logits_processor.py",
            "layers/vocab_parallel_embedding.py",
            "Qwen3_5GatedDeltaNet",
            "Qwen3_5AttentionDecoderLayer",
            "Qwen2MoeSparseMoeBlock",
        ),
        model_context_patterns=(
            "Qwen3_5MoeForConditionalGeneration",
            "Qwen3_5MoeForCausalLM",
            "Qwen3_5ForCausalLM",
            "Qwen3_5LinearDecoderLayer",
            "Qwen3_5AttentionDecoderLayer",
            "Qwen3_5GatedDeltaNet",
            "Qwen2MoeSparseMoeBlock",
        ),
        phase_patterns=("_execute_extend", "_execute_decode"),
    ),
    classify_node=classify_qwen35_sglang_node,
)
