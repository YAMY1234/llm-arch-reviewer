#!/usr/bin/env python3
"""TensorRT-LLM eager-trace rules for the shared GLM-5.2 Model IR."""

from __future__ import annotations

from models.common.trace_mapping import FrameRef, StackFrameRules, TraceMappingRules


TRTLLM_PREFILL_SIGNATURE = "applyMLARopeAndAssignQKVKernelOptContext"
TRTLLM_DECODE_SIGNATURE = "applyMLARopeAndAssignQKVKernelGeneration"
GLM52_ATTENTION_LAYERS_PER_FORWARD = 78


def _has_any(text: str, *needles: str) -> bool:
    return any(needle in text for needle in needles)


def classify_trtllm_node(
    kernel_name: str,
    cpu_op_name: str | None,
    stack: list[FrameRef],
) -> tuple[str | None, str]:
    names = "\n".join(frame.raw for frame in stack)
    lowered = names.lower()
    kernel = kernel_name.lower()
    cpu = (cpu_op_name or "").lower()

    is_all_gather = "allgather" in kernel or "all_gather" in kernel
    is_all_reduce = _has_any(
        kernel,
        "allreduce",
        "all_reduce",
        "rmsnormlamport",
    )
    if is_all_gather:
        if "sparse_attn_indexer" in lowered:
            return "dsa_attention.tp_prefill_index_topk_all_gather", "high"
        if _has_any(lowered, "lmhead", "lm_head"):
            return "top.tp_logits_all_gather", "high"
    if is_all_reduce:
        # TRT-LLM's unfused two-shot kernel is the row-parallel attention
        # boundary in this locked implementation.  Launch overlap can move
        # the CPU stack into the following FFN, so the implementation-unique
        # kernel contract takes precedence over the enclosing Python span.
        if "twoshotallreducekernel" in kernel:
            return "dsa_attention.tp_attention_output_collective", "high"
        if _has_any(lowered, "forward_moe", "deepseekv3moe", "deepseekv3gate"):
            return "moe.tp_moe_output_collective", "high"
        if _has_any(lowered, "gatedmlp", "forward_dense"):
            return "dense_mlp.tp_dense_mlp_output_collective", "high"
        if _has_any(
            lowered,
            "_helix_cp_output_projection",
            "deepseekv32attention",
            "modules/mla.py",
        ):
            return "dsa_attention.tp_attention_output_collective", "high"

    if "embedding" in lowered and "lmhead" not in lowered and "lm_head" not in lowered:
        if _has_any(kernel, "embedding", "index_elementwise_kernel"):
            return "top.embedding", "high"
    if _has_any(lowered, "lmhead", "lm_head") and cpu in {
        "aten::mm",
        "trtllm::fp4_gemm",
    }:
        return "top.lm_head", "high"
    if (
        "rmsnorm" in kernel
        and "deepseekv3model" in lowered
        and "deepseekv3decoderlayer" not in lowered
    ):
        return "top.final_norm", "high"

    if _has_any(kernel, "applymlaropeandassignqkvkerneloptcontext", "applymlaropeandassignqkvkernelgeneration"):
        return "dsa_attention.q_split_rope", "high"
    if _has_any(kernel, "fused_q_indexer_rope_hadamard_quant", "qindexer"):
        return "dsa_attention.index_q_projection", "high"
    if _has_any(kernel, "fused_k_indexer_norm_rope_store", "kindexer"):
        return "dsa_attention.index_k_norm_rope", "high"
    if "_get_k_and_s_triton_kernel" in kernel:
        return "dsa_attention.index_k_cache", "high"
    if _has_any(kernel, "sm100_fp8_mqa_logits", "sm100_fp8_paged_mqa_logits"):
        return "dsa_attention.index_logits", "high"
    if _has_any(kernel, "topk_transform_prefill_kernel", "topk_main_kernel", "fast_topk"):
        return "dsa_attention.index_topk", "high"
    if "fmhasm103akernel_qkve4m3" in kernel:
        return "dsa_attention.sparse_mla_core", "high"

    if _has_any(lowered, "forward_dsa_proj", "modules/mla.py"):
        if cpu in {"aten::mm", "trtllm::fp4_gemm", "trtllm::nvfp4_gemm"}:
            if _has_any(lowered, "output_projection", "_helix_cp_output_projection"):
                return "dsa_attention.output_projection", "high"
            if _has_any(lowered, "q_b_proj", "qb_proj", "q_b_projection"):
                return "dsa_attention.q_b_projection", "high"
            if _has_any(lowered, "kv_a_proj", "kva_proj", "kv_a_projection"):
                return "dsa_attention.kv_a_projection", "high"
            return "dsa_attention.q_a_projection", "medium"
        if "rmsnorm" in kernel:
            if _has_any(lowered, "kv_a", "kva"):
                return "dsa_attention.kv_a_norm", "high"
            return "dsa_attention.q_a_norm", "medium"

    if _has_any(
        kernel,
        "routingindiceshistogramscoreskernel",
        "routingindicesblockkernel",
    ):
        return "moe.topk", "high"
    if _has_any(
        kernel,
        "moe::dev::routing::routinginitexpertcounts",
        "moe::dev::routing::routingindicescoopkernel",
        "nvfp4quantizetmakernel",
    ) and _has_any(lowered, "forward_moe", "deepseekv3moe"):
        return "moe.dispatch", "high"
    if _has_any(kernel, "bmm_e2m1", "bmm_bfloat16"):
        return "moe.routed_experts", "high"
    if "moe::dev::finalize::finalizekernel" in kernel:
        return "moe.routed_weighted_combine", "high"

    in_shared_expert = _has_any(lowered, "shared_expert", "sharedexpert")
    if in_shared_expert:
        if cpu in {"aten::mm", "trtllm::fp4_gemm", "trtllm::nvfp4_gemm"}:
            if _has_any(lowered, "rowlinear", "down_proj", "downprojection"):
                return "moe.shared_expert_down", "high"
            return "moe.shared_expert_up", "medium"
        if "act_and_mul" in kernel:
            return "moe.shared_expert_activation", "high"

    if _has_any(lowered, "gatedmlp", "forward_dense"):
        if cpu in {"aten::mm", "trtllm::fp4_gemm", "trtllm::nvfp4_gemm"}:
            if _has_any(lowered, "rowlinear", "down_proj", "downprojection"):
                return "dense_mlp.down_projection", "high"
            return "dense_mlp.gate_up_projection", "medium"
        if "act_and_mul" in kernel:
            return "dense_mlp.swiglu", "high"

    if _has_any(lowered, "deepseekv3gate", "forward_moe") and cpu in {
        "aten::mm",
        "trtllm::fp4_gemm",
        "trtllm::nvfp4_gemm",
    }:
        return "moe.router", "medium"
    if "rmsnorm" in kernel and "deepseekv3decoderlayer" in lowered:
        if _has_any(lowered, "forward_moe", "forward_dense"):
            return "stack.post_attention_norm", "medium"
        return "stack.input_norm", "medium"

    return None, "unmapped"


GLM52_TRTLLM_TRACE_RULES = TraceMappingRules(
    model_id="glm52",
    signature_kernel=TRTLLM_PREFILL_SIGNATURE,
    signature_count_per_forward=GLM52_ATTENTION_LAYERS_PER_FORWARD,
    stack=StackFrameRules(
        operator_patterns=(
            "tensorrt_llm/_torch/modules/",
            "tensorrt_llm/_torch/attention_backend/sparse/dsa.py",
            "tensorrt_llm/_torch/distributed/ops.py",
            "tensorrt_llm/_torch/models/modeling_deepseekv3.py",
        ),
        semantic_patterns=(
            "tensorrt_llm/_torch/models/modeling_deepseekv3.py",
            "tensorrt_llm/_torch/modules/mla.py",
            "tensorrt_llm/_torch/modules/embedding.py",
            "tensorrt_llm/_torch/modules/gated_mlp.py",
            "tensorrt_llm/_torch/attention_backend/sparse/dsa.py",
            "tensorrt_llm/_torch/distributed/ops.py",
        ),
        model_context_patterns=(
            "DeepseekV3Model",
            "DeepseekV3DecoderLayer",
            "DeepseekV32Attention",
            "DeepseekV3MoE",
            "DeepseekV3Gate",
            "GatedMLP",
            "LMHead",
        ),
        phase_patterns=("_forward_step", "model_forward", "forward"),
    ),
    classify_node=classify_trtllm_node,
)
