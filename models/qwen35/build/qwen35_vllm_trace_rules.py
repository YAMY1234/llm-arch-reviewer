#!/usr/bin/env python3
"""vLLM eager-trace rules for the canonical Qwen3.5 pure-TP8 IR."""

from __future__ import annotations

from models.common.trace_mapping import FrameRef, StackFrameRules, TraceMappingRules


QWEN35_VLLM_GDN_SIGNATURE = "causal_conv1d"
QWEN35_GDN_LAYERS = 45


def _has_any(value: str, *needles: str) -> bool:
    return any(needle in value for needle in needles)


def classify_qwen35_vllm_node(
    kernel_name: str,
    cpu_op_name: str | None,
    stack: list[FrameRef],
) -> tuple[str | None, str]:
    names = "\n".join(frame.raw for frame in stack).lower()
    kernel = kernel_name.lower()
    cpu = (cpu_op_name or "").lower()
    in_gdn = _has_any(
        names,
        "qwen_gdn_linear_attn.py",
        "qw3_5decoderlayer",
        "qwen3_5decoderlayer",
        "qw gated",
    ) and "linear_attn" in names
    in_attention = _has_any(names, "qwen3nextattention", "qwen3_next.py") and not in_gdn
    in_moe = _has_any(
        names,
        "qwen3nextsparsemoeblock",
        "fused_moe/",
        "fusedmoe",
    )

    if cpu == "record_param_comms":
        if "vocab_parallel_embedding" in names:
            return "top.tp_embedding_output_collective", "high"
        if in_gdn:
            return "gdn_attention.tp_gdn_output_collective", "high"
        if in_attention:
            return "full_attention.tp_attention_output_collective", "high"
        if in_moe:
            return "moe_block.tp_moe_output_collective", "high"
    if cpu == "vllm::all_gather" and "logits_processor" in names:
        return "top.tp_logits_all_gather", "high"

    if "qwen_gdn_attention_core" in cpu:
        return "gdn_attention.gated_delta_recurrence", "high"
    if "_causal_conv1d_fwd_kernel" in kernel or "causal_conv1d" in kernel:
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
    if _has_any(kernel, "layer_norm_gated", "rmsnormgated"):
        return "gdn_attention.output_gate_norm", "high"
    if in_gdn and cpu in {"aten::mm", "aten::matmul", "vllm::fp8_gemm_nt_op"}:
        if "in_proj_qkvz" in names:
            return "gdn_attention.qkvz_projection", "high"
        if "in_proj_ba" in names:
            return "gdn_attention.ba_projection", "high"
        if "rowparallellinear" in names or "out_proj" in names:
            return "gdn_attention.output_projection", "high"

    if "fused_qk_rmsnorm_rope_gate" in cpu or _has_any(
        kernel, "fused_qk_rmsnorm_rope_gate", "fused_qk_norm_rope_gate"
    ):
        # One implementation interval owns QK norm, partial RoPE and gate-copy;
        # the binding records the 1:N fusion and assigns timing to qk_norm.
        return "full_attention.qk_norm", "high"
    if in_attention:
        if cpu in {"aten::mm", "aten::matmul", "vllm::fp8_gemm_nt_op"}:
            if "qkvparallellinear" in names or "qkv_proj" in names:
                return "full_attention.qkv_projection", "high"
            if "rowparallellinear" in names or "o_proj" in names:
                return "full_attention.output_projection", "high"
        if _has_any(kernel, "fmha", "mha", "attention") or "attention/layer.py" in names:
            return "full_attention.causal_gqa", "high"
        if _has_any(kernel, "sigmoid", "mul") and "attn_output" in names:
            return "full_attention.attention_output_gate", "medium"

    if in_moe:
        if "gatelinear" in names and cpu in {"aten::mm", "aten::matmul"}:
            return "moe_block.router", "high"
        if _has_any(kernel, "routingindices", "topk") or "topk" in cpu:
            return "moe_block.router", "high"
        if "shared_expert" in names and _has_any(
            kernel + cpu, "gemm", "bmm_", "silu", "aten::mm", "fp8_gemm"
        ):
            return "moe_block.shared_expert", "high"
        if _has_any(
            kernel,
            "activationdeepseek",
            "finalizekernel",
            "bmm_",
            "grouped",
            "moe::dev::",
        ):
            return "moe_block.routed_experts", "high"
        if cpu == "aten::add" or "finalize" in kernel:
            return "moe_block.weighted_combine", "medium"

    if "vocab_parallel_embedding" in names:
        return "top.embedding", "high"
    if "logits_processor" in names and cpu in {
        "aten::mm",
        "aten::matmul",
        "vllm::fp8_gemm_nt_op",
    }:
        return "top.lm_head", "high"
    if _has_any(kernel, "rmsnorm", "layer_norm") and _has_any(
        names, "qwen3_5model", "qwen3nextmodel"
    ):
        return "top.final_norm", "medium"
    return None, "unmapped"


QWEN35_VLLM_TRACE_RULES = TraceMappingRules(
    model_id="qwen35_vllm_tp8",
    signature_kernel=QWEN35_VLLM_GDN_SIGNATURE,
    signature_count_per_forward=QWEN35_GDN_LAYERS,
    stack=StackFrameRules(
        operator_patterns=(
            "model_executor/layers/linear.py",
            "model_executor/layers/attention",
            "model_executor/layers/mamba/gdn",
            "model_executor/layers/fused_moe",
            "model_executor/layers/logits_processor.py",
            "model_executor/layers/vocab_parallel_embedding.py",
            "distributed/parallel_state.py",
        ),
        semantic_patterns=(
            "model_executor/models/qwen3_5.py",
            "model_executor/models/qwen3_next.py",
            "model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py",
            "model_executor/layers/fused_moe",
            "model_executor/layers/logits_processor.py",
            "model_executor/layers/vocab_parallel_embedding.py",
        ),
        model_context_patterns=(
            "Qwen3_5MoeForConditionalGeneration",
            "Qwen3_5MoeForCausalLM",
            "Qwen3_5ForCausalLMBase",
            "Qwen3_5Model",
            "Qwen3_5DecoderLayer",
            "Qwen3NextAttention",
            "QwenGatedDeltaNetAttention",
            "Qwen3NextSparseMoeBlock",
        ),
        phase_patterns=("execute_model",),
    ),
    classify_node=classify_qwen35_vllm_node,
)
