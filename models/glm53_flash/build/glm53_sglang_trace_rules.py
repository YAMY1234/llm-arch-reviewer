#!/usr/bin/env python3
"""SGLang eager-trace rules for the stable GLM-5.3-Flash IR.

The runtime uses the GLM-5.3 wrapper for KDA and mHC while reusing the
DeepSeek-v2 attention and MoE implementation.  Unique GLM-5.3 kernels are
classified first; stable reused leaves then fall back to the GLM-5.2 rules and
are translated to the GLM-5.3 node vocabulary.
"""

from __future__ import annotations

import re

from models.common.trace_mapping import FrameRef, StackFrameRules, TraceMappingRules
from models.glm52.build.glm52_trace_rules import classify_glm52_node


GLM53_SGLANG_DSA_SIGNATURE = "fmhaSm100fKernel_QkvE4m3OBfloat16H512"
GLM53_DSA_LAYERS = 11


def _has_any(text: str, *needles: str) -> bool:
    return any(needle in text for needle in needles)


def _decoder_layer_index(stack_text: str) -> int | None:
    match = re.search(r"glm5nextdecoderlayer_(\d+)", stack_text)
    return int(match.group(1)) if match else None


def _attention_collective_node(layer_index: int | None) -> str | None:
    if layer_index is None:
        return None
    # The official 45-layer schedule is KDA,KDA,KDA,DSA repeated, ending KDA.
    if layer_index % 4 == 3:
        return "dsa_attention.tp_dsa_output_collective"
    return "linear_attention.tp_kda_output_collective"


def classify_glm53_sglang_node(
    kernel_name: str,
    cpu_op_name: str | None,
    stack: list[FrameRef],
) -> tuple[str | None, str]:
    names = "\n".join(frame.raw for frame in stack)
    lowered = names.lower()
    kernel = kernel_name.lower()
    cpu = (cpu_op_name or "").lower()
    layer_index = _decoder_layer_index(lowered)

    is_all_gather = "allgather" in kernel or "all_gather" in kernel
    is_all_reduce = _has_any(
        kernel,
        "allreduce",
        "all_reduce",
        "oneshotallreducefusionkernel",
    ) or "flashinfer_allreduce" in cpu

    # TP boundaries.  The mHC communicator closes attention before prepare_mlp;
    # the layer schedule identifies whether that attention was KDA or DSA.
    if is_all_gather and _has_any(lowered, "logitsprocessor", "lm_head"):
        return "top.tp_logits_all_gather", "high"
    if is_all_reduce:
        if "vocabparallelembedding" in lowered:
            return "top.tp_embedding_output_collective", "high"
        if _has_any(
            lowered,
            "attention_tensor_model_parallel_all_reduce",
            "_gather_hidden_states_and_residual",
        ):
            node = _attention_collective_node(layer_index)
            if node is not None:
                return node, "high"
        # Decode may fuse the previous FFN all-reduce into the next layer's
        # prepare_attn.  Layers 0-2 are dense; layers 3-44 are MoE.
        if "prepare_attn" in lowered and layer_index is not None:
            if 1 <= layer_index <= 3:
                return "dense_mlp.tp_dense_mlp_output_collective", "high"
            if layer_index >= 4:
                return "moe.tp_moe_output_collective", "high"
        if _has_any(lowered, "glm5nextmoe", "deepseekv2moe", "forward_normal"):
            return "moe.tp_moe_output_collective", "high"
        if _has_any(lowered, "glm5nextmlp", "deepseekv2mlp"):
            return "dense_mlp.tp_dense_mlp_output_collective", "high"

    # mHC has implementation-unique fused kernels.  The binding records every
    # semantic leaf included in each fused interval.
    if _has_any(
        kernel,
        "mhc_pre_big_fuse_with_norm",
        "sm100_tf32_hc_prenorm_gemm",
        "mhc_pre_gemm_sqrsum",
        "hc_split_sinkhorn",
    ):
        return "mhc_transform.pre_weights", "high"
    if "mhc_post" in kernel:
        return "mhc_transform.residual_mix", "high"
    if "hc_expand" in lowered:
        return "top.hc_expand", "high"
    if cpu == "aten::mean" and "hc_contract" in lowered:
        return "top.hc_contract", "high"

    # KDA model-unique kernels take precedence over a possibly stale enclosing
    # Python range from an adjacent fused launch.
    if "_causal_conv1d_fwd_kernel" in kernel:
        return "linear_attention.qkv_short_conv", "high"
    if "l2norm_fwd_kernel" in kernel:
        return "linear_attention.qk_l2_norm", "high"
    if _has_any(kernel, "kda_gate_chunk_cumsum", "kda_gate_cumsum"):
        return "linear_attention.forget_decay", "high"
    if _has_any(
        kernel,
        "chunk_kda_fwd_kernel",
        "chunk_gated_delta_rule_fwd_kernel",
        "chunk_gla_fwd_kernel_o",
        "recompute_w_u_fwd_kernel",
    ):
        return "linear_attention.recurrent_update", "high"
    if "layer_norm_gated_fwd_kernel" in kernel:
        return "linear_attention.gated_norm", "high"
    if "_gather_initial_states_kernel" in kernel:
        return "linear_attention.recurrent_state", "high"
    if "_scatter_states_kernel" in kernel:
        return "linear_attention.conv_state", "high"

    in_kda = _has_any(
        lowered,
        "glm5nextlinearattention",
        "radix_linear_attention.py",
        "glm5_next.py(518): forward_qkvbfg",
        "glm5_next.py(537): forward_qkvbfg_fused",
        "glm5_next.py(559): forward",
    )
    if in_kda:
        # Chunk-state initialization and movement are explicit state costs even
        # when the lower-level implementation exposes only generic ATen ops.
        if cpu in {"aten::fill_", "aten::zeros", "aten::zero_"}:
            return "linear_attention.recurrent_state", "medium"
        if cpu in {"aten::index", "aten::_index_put_impl_"}:
            return "linear_attention.conv_state", "medium"
        if cpu == "aten::mm":
            if "qkvparallellinear" in lowered or "mergedcolumnparallelrepeatedlinear" in lowered:
                return "linear_attention.qkv_projection", "high"
            if "rowparallellinear" in lowered:
                return "linear_attention.output_projection", "high"
            # beta, forget, and output-gate projections share generic module
            # frames.  Attribute their shared interval to one representative
            # leaf at medium confidence; the binding keeps the fusion set.
            if _has_any(lowered, "columnparallellinear", "replicatedlinear"):
                return "linear_attention.forget_projection", "medium"

    # GLM-5.3 uses an H512 DSA signature rather than GLM-5.2's HQk576 name.
    if GLM53_SGLANG_DSA_SIGNATURE.lower() in kernel:
        return "dsa_attention.sparse_mla_core", "high"
    if "kpool_topk_transform_kernel" in kernel:
        return "dsa_attention.top_pool_selection", "high"

    # The GLM-5.3 checkpoint uses native FP8 routed-expert BMM names rather
    # than GLM-5.2's NVFP4 E2m1 signatures.
    if kernel.startswith("bmm_") and _has_any(
        lowered, "deepseekv2moe", "forward_normal"
    ):
        return "moe.routed_gate_up", "high"

    node, confidence = classify_glm52_node(kernel_name, cpu_op_name, stack)
    translations = {
        "dsa_attention.tp_attention_output_collective": "dsa_attention.tp_dsa_output_collective",
        "dsa_attention.index_topk": "dsa_attention.top_pool_selection",
        "moe.routed_experts": "moe.routed_gate_up",
        "moe.shared_expert_up": "moe.shared_gate_up",
        "moe.shared_expert_activation": "moe.shared_activation",
        "moe.shared_expert_down": "moe.shared_down",
        "dense_mlp.swiglu": "dense_mlp.clamped_swiglu",
    }
    return translations.get(node, node), confidence


GLM53_SGLANG_TRACE_RULES = TraceMappingRules(
    model_id="glm53_flash_sglang",
    signature_kernel=GLM53_SGLANG_DSA_SIGNATURE,
    signature_count_per_forward=GLM53_DSA_LAYERS,
    stack=StackFrameRules(
        operator_patterns=(
            "layers/linear.py",
            "layers/quantization",
            "layers/layernorm.py",
            "layers/communicator_mhc.py",
            "kernels/ops/layernorm/mhc.py",
            "layers/radix_linear_attention.py",
            "layers/attention/dsa",
            "attention_forward_methods/forward_mla.py",
            "layers/moe",
            "layers/logits_processor.py",
            "layers/vocab_parallel_embedding.py",
        ),
        semantic_patterns=(
            "models/glm5_next.py",
            "models/deepseek_v2.py",
            "layers/communicator_mhc.py",
            "kernels/ops/layernorm/mhc.py",
            "layers/radix_linear_attention.py",
            "layers/logits_processor.py",
            "layers/vocab_parallel_embedding.py",
        ),
        model_context_patterns=(
            "Glm5NextForConditionalGeneration",
            "Glm5NextModel",
            "Glm5NextDecoderLayer",
            "Glm5NextLinearAttention",
            "DeepseekV2AttentionMLA",
            "Glm5NextMoE",
            "Glm5NextMLP",
            "Indexer",
        ),
        phase_patterns=(
            "_execute_extend",
            "_execute_decode",
        ),
    ),
    classify_node=classify_glm53_sglang_node,
    # This exact H512 FlashInfer signature is unique to the DSA core. Kineto
    # can attach one asynchronous launch to a later cache-management Python
    # span; retain the kernel identity instead of silently losing one of the
    # eleven official DSA layers.
    kernel_only_nodes=frozenset({"dsa_attention.sparse_mla_core"}),
)
