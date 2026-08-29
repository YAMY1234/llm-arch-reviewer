#!/usr/bin/env python3
"""vLLM eager-trace rules for the stable GLM-5.3-Flash IR.

The rules intentionally use source-stack scope plus model-unique kernel or
custom-op signatures.  vLLM's asynchronous launches can leave a neighboring
Python frame active when a kernel reaches the GPU; exact custom-op signatures
therefore take precedence where they identify a stable semantic operation.
"""

from __future__ import annotations

from models.common.trace_mapping import FrameRef, StackFrameRules, TraceMappingRules


GLM53_VLLM_DSA_SIGNATURE = "fmhaSm100fKernel_QkvE4m3OBfloat16H512"
GLM53_DSA_LAYERS = 11


def _has_any(text: str, *needles: str) -> bool:
    return any(needle in text for needle in needles)


def classify_glm53_vllm_node(
    kernel_name: str,
    cpu_op_name: str | None,
    stack: list[FrameRef],
) -> tuple[str | None, str]:
    names = "\n".join(frame.raw for frame in stack)
    lowered = names.lower()
    kernel = kernel_name.lower()
    cpu = (cpu_op_name or "").lower()

    in_kda = "models/glm5next/nvidia/kda.py" in lowered
    in_dsa = "models/glm5next/nvidia/attention.py" in lowered
    in_moe = _has_any(
        lowered,
        "fused_moe/",
        "nn.module: glm5nextmoe_",
        "models/glm5next/nvidia/model.py(250): forward",
    )
    in_dense = (
        "models/glm5next/nvidia/model.py(144): forward" in lowered
        and not in_moe
    )

    # Exact TP collectives observed as record_param_comms CPU ranges.
    if cpu == "record_param_comms":
        if "vocab_parallel_embedding" in lowered:
            return "top.tp_embedding_output_collective", "high"
        if in_kda:
            return "linear_attention.tp_kda_output_collective", "high"
        if in_dsa:
            return "dsa_attention.tp_dsa_output_collective", "high"
        if in_moe:
            return "moe.tp_moe_output_collective", "high"
        if in_dense:
            return "dense_mlp.tp_dense_mlp_output_collective", "high"
    if cpu == "vllm::all_gather" and "logits_processor" in lowered:
        return "top.tp_logits_all_gather", "high"

    # Model-unique KDA kernels override an occasionally stale neighboring frame.
    if "_causal_conv1d_fwd_kernel" in kernel:
        return "linear_attention.qkv_short_conv", "high"
    if "l2norm_fwd_kernel" in kernel:
        return "linear_attention.qk_l2_norm", "high"
    if "kda_gate_cumsum_fwd_kernel" in kernel:
        return "linear_attention.forget_decay", "high"
    if _has_any(
        kernel,
        "chunk_gated_delta_rule_fwd_kernel",
        "chunk_kda_scaled_dot_kkt_fwd_kernel",
        "recompute_w_u_fwd_kernel",
    ):
        return "linear_attention.recurrent_update", "high"
    if "chunk_gla_fwd_kernel_o" in kernel:
        return "linear_attention.query_readout", "high"
    if "layer_norm_gated_fwd_kernel" in kernel:
        return "linear_attention.gated_norm", "high"
    if "_gather_initial_states_kernel" in kernel:
        return "linear_attention.recurrent_state", "high"
    if "_scatter_states_kernel" in kernel:
        return "linear_attention.conv_state", "high"

    # DSA/indexer signatures.
    if GLM53_VLLM_DSA_SIGNATURE.lower() in kernel:
        return "dsa_attention.sparse_mla_core", "high"
    if cpu == "_c::top_k_per_row_prefill" or "topkperrowprefill" in kernel:
        return "dsa_attention.top_pool_selection", "high"
    if "_convert_req_index_to_global_index_kernel" in kernel:
        return "dsa_attention.selected_indices", "high"
    if "_expand_pools_and_append_tail_kernel" in kernel:
        return "dsa_attention.token_expansion", "high"
    if "sm100_mqa_logits" in kernel:
        return "dsa_attention.index_logits", "high"
    if "_kpool_softmax_rotate_write_cache_kernel" in kernel:
        return "dsa_attention.key_pool_compression", "high"
    if "_kpool_tail_seed_kernel" in kernel:
        return "dsa_attention.index_k_cache", "high"
    if cpu == "_c_cache_ops::concat_and_cache_mla":
        return "dsa_attention.latent_kv_cache", "high"
    if "_fused_q_kv_rmsnorm_kernel" in kernel:
        # vLLM fuses q_a/kv_a projection normalization; the binding records the
        # shared interval and timing is attached to the first stable leaf.
        return "dsa_attention.q_a_norm", "high"
    if cpu == "aten::bmm" and in_dsa:
        return "dsa_attention.latent_kv_reconstruction", "high"
    if "attention.py(315): forward" in lowered and cpu == "aten::mm":
        return "dsa_attention.index_weight_projection", "high"
    if in_dsa and cpu == "aten::mm":
        if "deepseekv2fusedqkvaprojlinear" in lowered:
            return "dsa_attention.q_a_projection", "high"
        if "rowparallellinear" in lowered:
            return "dsa_attention.output_projection", "high"
        if "columnparallellinear" in lowered:
            return "dsa_attention.q_b_projection", "medium"

    # mHC custom ops expose the fused pre/post boundary directly.
    if cpu == "vllm::mhc_pre_tilelang":
        return "mhc_transform.pre_weights", "high"
    if cpu == "vllm::mhc_fused_post_pre_tilelang":
        if "mhc_post_tilelang_kernel" in kernel:
            return "mhc_transform.residual_mix", "high"
        return "mhc_transform.pre_weights", "high"
    if cpu == "vllm::mhc_post_tilelang":
        return "mhc_transform.residual_mix", "high"
    if cpu == "aten::mean" and "hc_contract" in lowered:
        return "top.hc_contract", "high"

    # KDA projections.  The two ColumnParallelLinear calls implement the
    # separate forget and beta projections but share the same generic module
    # frame; the binding records this as a shared interval.
    if in_kda and cpu == "aten::mm":
        if "_glm5nextmergedcolumnparallellinear" in lowered:
            return "linear_attention.qkv_projection", "high"
        if "rowparallellinear" in lowered:
            return "linear_attention.output_projection", "high"
        if "columnparallellinear" in lowered:
            return "linear_attention.forget_projection", "medium"

    # Dense FFN leaves.
    if in_dense:
        if cpu == "_c::silu_and_mul_with_clamp" or "act_and_mul_kernel" in kernel:
            return "dense_mlp.clamped_swiglu", "high"
        if "mergedcolumnparallellinear" in lowered:
            return "dense_mlp.gate_up_projection", "high"
        if "rowparallellinear" in lowered:
            return "dense_mlp.down_projection", "high"

    # MoE router, routed experts, shared expert, and weighted combine.  The
    # fused_moe custom op overlaps routed and shared experts; stable signatures
    # select a representative leaf and the binding carries the full fusion set.
    if in_moe:
        if "gatelinear" in lowered and cpu == "aten::mm":
            return "moe.router", "high"
        if _has_any(kernel, "routingindiceshistogramscoreskernel", "routingindicesblockscoreskernel"):
            return "moe.topk", "high"
        if _has_any(kernel, "routingindicescoopkernel", "routingindicesclusterkernel"):
            return "moe.dispatch", "high"
        if "moe::dev::activation::activationdeepseekkernel" in kernel:
            return "moe.routed_activation", "high"
        if "moe::dev::finalize::finalizekernel" in kernel:
            return "moe.routed_weighted_combine", "high"
        if kernel.startswith("bmm_bfloat16"):
            return "moe.routed_down", "high"
        if kernel.startswith("bmm_"):
            return "moe.routed_gate_up", "medium"
        if cpu == "_c::silu_and_mul_with_clamp" or "act_and_mul_kernel" in kernel:
            return "moe.shared_activation", "high"
        if cpu == "vllm::fp8_gemm_nt_op":
            return "moe.shared_gate_up", "medium"
        if cpu == "aten::add":
            return "moe.combine", "high"

    if "vocab_parallel_embedding" in lowered:
        return "top.embedding", "high"
    if "logits_processor" in lowered and cpu == "aten::mm":
        return "top.lm_head", "high"
    if cpu == "_c::rms_norm" and "glm5nextmodel" in lowered:
        return "top.final_norm", "high"

    return None, "unmapped"


GLM53_VLLM_TRACE_RULES = TraceMappingRules(
    model_id="glm53_flash_vllm",
    signature_kernel=GLM53_VLLM_DSA_SIGNATURE,
    signature_count_per_forward=GLM53_DSA_LAYERS,
    stack=StackFrameRules(
        operator_patterns=(
            "model_executor/layers/linear.py",
            "model_executor/layers/mla.py",
            "model_executor/layers/mhc.py",
            "model_executor/layers/fused_moe",
            "model_executor/layers/sparse_attn_indexer_kpool.py",
            "model_executor/layers/logits_processor.py",
            "model_executor/layers/vocab_parallel_embedding.py",
            "distributed/parallel_state.py",
        ),
        semantic_patterns=(
            "models/glm5next/nvidia/model.py",
            "models/glm5next/nvidia/kda.py",
            "models/glm5next/nvidia/attention.py",
            "model_executor/layers/mla.py",
            "model_executor/layers/mhc.py",
            "model_executor/layers/fused_moe",
            "model_executor/layers/logits_processor.py",
            "model_executor/layers/vocab_parallel_embedding.py",
        ),
        model_context_patterns=(
            "Glm5NextForConditionalGeneration",
            "Glm5NextModel",
            "Glm5NextDecoderLayer",
            "Glm5NextLinearAttention",
            "Glm5NextMLAAttention",
            "Glm5NextMoE",
            "Glm5NextMLP",
            "Indexer",
        ),
        phase_patterns=("model_runner.py(1504): execute_model",),
    ),
    classify_node=classify_glm53_vllm_node,
)
