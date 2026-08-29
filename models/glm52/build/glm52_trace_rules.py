#!/usr/bin/env python3
"""GLM-5.2 rules for the common eager Torch-trace mapper.

The rules describe only stable GLM/DeepSeek semantic signatures. Windowing,
stack recovery, validation, and artifact writing remain in
``models.common.trace_mapping``.
"""

from __future__ import annotations

import re

from models.common.trace_mapping import FrameRef, StackFrameRules, TraceMappingRules


GLM52_DSA_SIGNATURE_KERNEL = "fmhaSm100fKernel_QkvE4m3OBfloat16HQk576"
GLM52_ATTENTION_LAYERS_PER_FORWARD = 78


def _has_any(text: str, *needles: str) -> bool:
    return any(needle in text for needle in needles)


def classify_glm52_node(
    kernel_name: str,
    cpu_op_name: str | None,
    stack: list[FrameRef],
) -> tuple[str | None, str]:
    """Map SGLang GLM-5.2 eager kernels to stable Model/Execution IR nodes.

    Some GPU launches overlap the next Python span. Model-unique kernel
    signatures therefore deliberately take precedence over a possibly stale
    deepest frame. Generic GEMMs and elementwise kernels still require their
    enclosing implementation stack.
    """

    names = "\n".join(frame.raw for frame in stack)
    lowered = names.lower()
    kernel = kernel_name.lower()
    cpu = (cpu_op_name or "").lower()
    layer_match = re.search(r"deepseekv2decoderlayer_(\d+)", lowered)
    layer_index = int(layer_match.group(1)) if layer_match else None

    # Communication boundaries must win over their enclosing compute module.
    is_all_gather = "allgather" in kernel or "all_gather" in kernel
    is_all_reduce = _has_any(
        kernel,
        "allreduce",
        "all_reduce",
        "oneshotallreducefusionkernel",
    ) or "flashinfer_allreduce" in cpu

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
            return "dsa_attention.tp_attention_output_collective", "high"
        # Decode can defer the FFN all-reduce and fuse it into the next
        # layer's prepare_attn residual+norm. Layer N therefore closes the
        # FFN contract of layer N-1. GLM layers 0-2 are dense; 3-77 are MoE.
        if "prepare_attn" in lowered and layer_index is not None:
            if 1 <= layer_index <= 3:
                return "dense_mlp.tp_dense_mlp_output_collective", "high"
            if layer_index >= 4:
                return "moe.tp_moe_output_collective", "high"
        if "deepseekv2moe" in lowered or "forward_normal" in lowered:
            return "moe.tp_moe_output_collective", "high"
        if "deepseekv2mlp" in lowered:
            return "dense_mlp.tp_dense_mlp_output_collective", "high"

    # Model-unique DSA/indexer kernels override occasionally stale Python
    # spans caused by asynchronous launches.
    if GLM52_DSA_SIGNATURE_KERNEL.lower() in kernel:
        return "dsa_attention.sparse_mla_core", "high"
    if "ropequantizekernel" in kernel:
        return "dsa_attention.q_split_rope", "high"
    if "set_mla_kv_buffer_kernel" in kernel:
        return "dsa_attention.latent_kv_cache", "high"
    if "fused_q_indexer_rope_hadamard_quant" in kernel:
        return "dsa_attention.index_q_projection", "high"
    if "fused_k_indexer_norm_rope_store" in kernel:
        return "dsa_attention.index_k_norm_rope", "high"
    if "_get_k_and_s_triton_kernel" in kernel:
        return "dsa_attention.index_k_cache", "high"
    if "sm100_mqa_logits" in kernel or "sm100_paged_mqa_logits" in kernel:
        return "dsa_attention.index_logits", "high"
    if _has_any(kernel, "topk_transform_prefill_kernel", "topk_main_kernel") or (
        "fast_topk_transform" in cpu
    ):
        return "dsa_attention.index_topk", "high"

    # FlashInfer/TRT-LLM NVFP4 MoE kernels are unique even when their launch is
    # attributed to a neighboring indexer frame in an overlapped trace.
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
    ):
        return "moe.dispatch", "high"
    if "nvfp4quantizetmakernel" in kernel and _has_any(
        lowered, "fusedmoe", "forward_normal", "dsa_indexer.py"
    ):
        return "moe.dispatch", "high"
    if _has_any(
        kernel,
        "bmm_e2m1_e2m1e2m1_fp32",
        "bmm_bfloat16_e2m1e2m1_fp32",
    ):
        return "moe.routed_experts", "high"
    if "moe::dev::finalize::finalizekernel" in kernel:
        return "moe.routed_weighted_combine", "high"

    # Embedding, final norm, and vocabulary head.
    if "_vocab_parallel_embedding_kernel" in kernel:
        return "top.embedding", "high"
    if _has_any(lowered, "logitsprocessor", "lm_head") and cpu == "aten::mm":
        return "top.lm_head", "high"
    if (
        "rmsnormrmsnormkernel" in kernel
        and "oi646144" in kernel
        and "deepseekv2decoderlayer" not in lowered
    ):
        return "top.final_norm", "high"
    if (
        "fusedaddrmsnorm" in kernel
        and "deepseekv2model" in lowered
        and "deepseekv2decoderlayer" not in lowered
    ):
        return "top.final_norm", "high"

    # Fused residual/norm boundaries in the decoder schedule.
    if "fusedaddrmsnorm" in kernel:
        if "prepare_mlp" in lowered:
            return "stack.post_attention_norm", "high"
        if "prepare_attn" in lowered:
            return "stack.input_norm", "high"
    if (
        "rmsnormrmsnormkernel" in kernel
        and "oi646144" in kernel
        and "prepare_attn" in lowered
    ):
        return "stack.input_norm", "high"

    # MLA projection and absorbed latent reconstruction path.
    if cpu in {"aten::mm", "sglang::jit_dsv3_fused_a_gemm"} and (
        "prepare_qkv_latent" in lowered
    ):
        # SGLang fuses q_a and kv_a into one [H, 2048+576] GEMM. Attribute
        # timing to the first candidate node; the binding records the fusion.
        return "dsa_attention.q_a_projection", "high"
    if cpu in {"aten::mm", "sglang::jit_dsv3_fused_a_gemm"} and (
        "q_b_proj_forward" in lowered
    ):
        return "dsa_attention.q_b_projection", "high"
    if cpu == "aten::bmm" and _has_any(
        lowered, "forward_absorb_prepare", "forward_absorb_core"
    ):
        return "dsa_attention.latent_kv_reconstruction", "high"
    if cpu == "sglang::cutedsl_tgv_bf16_gemm" and (
        "_fused_q_prepare_and_store" in lowered
    ):
        return "dsa_attention.index_q_projection", "high"
    if cpu == "aten::mm" and "_fused_q_prepare_and_store" in lowered:
        if "_execute_decode" in lowered:
            return "dsa_attention.index_k_gate_projection", "high"
        if "80x128_64x10" in kernel:
            return "dsa_attention.index_k_gate_projection", "high"
        return "dsa_attention.index_q_projection", "high"
    if cpu == "aten::mm" and "rowparallellinear" in lowered:
        if "_forward_shared_experts" not in lowered and "deepseekv2mlp" not in lowered:
            return "dsa_attention.output_projection", "medium"

    if "rmsnormrmsnormkernel" in kernel and "forward_absorb_prepare" in lowered:
        if "oi642048" in kernel:
            return "dsa_attention.q_a_norm", "high"
        if "oi64512" in kernel:
            return "dsa_attention.kv_a_norm", "high"

    # Dense layers 0-2 and the shared expert reuse the same MLP class. The
    # explicit shared-expert call frame separates their semantic scopes.
    in_shared_expert = "_forward_shared_experts" in lowered or (
        "deepseekv2mlp" in lowered and "deepseekv2moe" in lowered
    )
    if in_shared_expert:
        if cpu == "aten::mm" and "mergedcolumnparallellinear" in lowered:
            return "moe.shared_expert_up", "high"
        if "act_and_mul_kernel" in kernel or "_run_activation_inplace" in cpu:
            return "moe.shared_expert_activation", "high"
        if cpu == "aten::mm" and "rowparallellinear" in lowered:
            return "moe.shared_expert_down", "high"

    if "deepseekv2mlp" in lowered:
        if cpu in {"aten::mm", "sglang::cutedsl_tgv_bf16_gemm"} and (
            "mergedcolumnparallellinear" in lowered
        ):
            return "dense_mlp.gate_up_projection", "high"
        if "act_and_mul_kernel" in kernel or "_run_activation_inplace" in cpu:
            return "dense_mlp.swiglu", "high"
        if cpu == "aten::mm" and "rowparallellinear" in lowered:
            return "dense_mlp.down_projection", "high"

    if "moegate" in lowered and cpu in {"aten::mm", "sglang::dsv3_router_gemm"}:
        return "moe.router", "high"
    if "maybe_fuse_routed_scale_and_shared_add" in lowered and cpu == "aten::add_":
        return "moe.combine", "high"

    return None, "unmapped"


GLM52_TRACE_RULES = TraceMappingRules(
    model_id="glm52",
    signature_kernel=GLM52_DSA_SIGNATURE_KERNEL,
    signature_count_per_forward=GLM52_ATTENTION_LAYERS_PER_FORWARD,
    stack=StackFrameRules(
        operator_patterns=(
            "layers/linear.py",
            "layers/quantization",
            "layers/layernorm.py",
            "layers/communicator.py",
            "layers/attention/dsa",
            "attention_forward_methods/forward_mla.py",
            "layers/moe",
            "radix_attention.py",
            "layers/logits_processor.py",
            "layers/vocab_parallel_embedding.py",
        ),
        semantic_patterns=(
            "models/deepseek_v2.py",
            "models/glm4_moe.py",
            "layers/logits_processor.py",
            "layers/vocab_parallel_embedding.py",
        ),
        model_context_patterns=(
            "DeepseekV2Model",
            "DeepseekV2DecoderLayer",
            "DeepseekV2AttentionMLA",
            "DeepseekV2MoE",
            "DeepseekV2MLP",
            "Indexer",
            "LogitsProcessor",
            "VocabParallelEmbedding",
        ),
        phase_patterns=(
            "forward_extend",
            "forward_decode",
            "_execute_extend",
            "_execute_decode",
            "cuda_graph_runner",
            "replay",
        ),
    ),
    classify_node=classify_glm52_node,
)
