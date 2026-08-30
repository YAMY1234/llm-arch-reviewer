#!/usr/bin/env python3
"""vLLM eager-trace rules for DeepSeek-V4-Pro-0813 pure TP8.

Shared CSA/HCA operations are intentionally emitted as ``attention.*``
contract nodes.  The occurrence compiler expands those operations through the
official 61-layer ratio schedule; implementation-unique compressor/indexer
kernels already identify their branch directly.
"""

from __future__ import annotations

from models.common.trace_mapping import FrameRef, StackFrameRules, TraceMappingRules


DSV4_FINAL_HEAD_SIGNATURE = "hc_head_fuse_tilelang_kernel"


def _has_any(text: str, *needles: str) -> bool:
    return any(needle in text for needle in needles)


def classify_deepseek_v4_pro_vllm_node(
    kernel_name: str,
    cpu_op_name: str | None,
    stack: list[FrameRef],
) -> tuple[str | None, str]:
    names = "\n".join(frame.raw for frame in stack)
    lowered = names.lower()
    kernel = kernel_name.lower()
    cpu = (cpu_op_name or "").lower()

    in_embedding = "vocab_parallel_embedding" in lowered
    in_logits = "logits_processor" in lowered or "computelogits" in lowered
    in_attention = _has_any(
        lowered,
        "models/deepseek_v4/attention.py",
        "models/deepseek_v4/nvidia/flashmla.py",
        "deepseekv4flashmlaattention",
    )
    in_indexer = _has_any(
        lowered,
        "deepseekv4indexer",
        "sparse_attn_indexer",
        "fused_indexer_q",
    )
    in_compressor = _has_any(
        lowered,
        "models/deepseek_v4/compressor.py",
        "sparse_attn_compress",
        "save_partial_states",
    )
    in_moe = _has_any(
        lowered,
        "deepseekv4moe",
        "fused_moe",
        "shared_experts",
    )

    is_all_reduce = _has_any(
        kernel,
        "allreduce",
        "all_reduce",
        "multimem_all_reduce",
    ) or cpu in {"vllm::all_reduce", "symm_mem::multimem_all_reduce_"}
    is_all_gather = _has_any(kernel, "allgather", "all_gather")
    if is_all_gather and in_logits:
        return "top.tp_logits_collective", "high"
    if is_all_reduce:
        if in_embedding:
            return "top.tp_embedding_output_collective", "high"
        if in_moe:
            return "moe.tp_moe_output_collective", "high"
        if in_attention or "deepseekv4decoderlayer" in lowered:
            return "attention.tp_output_collective", "high"

    # mHC implementation-unique signatures override stale neighboring stacks
    # from multi-stream launches.
    if DSV4_FINAL_HEAD_SIGNATURE in kernel:
        return "final_hc_read.read", "high"
    if "mhc_post" in kernel:
        return "mhc_transform.mix", "high"
    if "mhc_fused_tilelang_kernel" in kernel:
        return "mhc_transform.mix", "high"
    if _has_any(
        kernel,
        "mhc_pre_big_fuse_with_norm",
        "sm100_tf32_hc_prenorm_gemm",
        "mhc_pre_gemm_sqrsum",
        "hc_split_sinkhorn",
    ):
        return "mhc_transform.affine", "high"

    # Branch-specific compressor and CSA indexer signatures.
    if "sparseattncompressnormropestorec4kernel" in kernel:
        return "csa_compressor.softmax_pool", "high"
    if "sparseattncompressc128block8kernel" in kernel:
        return "hca_compressor.softmax_pool", "high"
    if "sparseattnnormropestorekernel" in kernel:
        return "hca_compressor.norm_rope", "high"
    if "_fused_kv_compress_norm_rope_insert_indexer_attn" in kernel:
        return "csa_indexer.k_compress", "high"
    if "indexerqfp8kernel" in kernel:
        return "csa_indexer.q_rope_rotate", "high"
    if cpu == "vllm::sparse_attn_indexer" or "sparse_attn_indexer" in kernel:
        return "csa_indexer.score", "high"
    if cpu == "aten::fill_" and "model_executor/layers/sparse_attn_indexer.py" in lowered:
        return "csa_indexer.selected_ids", "high"
    if _has_any(
        kernel,
        "topkperrowprefill",
        "filteredtopkunifiedkernel",
        "cooperative_topk_cs16",
    ) or cpu in {
        "_c::top_k_per_row_prefill",
        "_c::persistent_topk",
        "_c::cooperative_topk",
    }:
        return "csa_indexer.causal_topk", "high"
    if "_compute_global_topk_indices_and_lens_kernel" in kernel:
        return "csa_indexer.expand", "high"
    if "_combine_topk_swa_indices_kernel" in kernel:
        return "attention.index_union", "high"
    if "dequantgatherkcachekernel" in kernel or "dequant_gather_k" in kernel:
        return "attention.index_union", "high"
    if cpu == "aten::fill_" and "combine_topk_swa_indices" in lowered:
        return "attention.index_union", "high"
    if cpu == "aten::floor_divide" and "flashmla.py" in lowered:
        return "attention.index_union", "high"
    if cpu == "_c_cache_ops::cp_gather_indexer_k_quant_cache":
        return "csa_indexer.score", "high"
    if "_save_partial_states_kernel" in kernel:
        # The same state-write kernel serves every main compressor plus the 30
        # CSA indexer compressors.  Async stacks are not stable enough to split
        # those 91 occurrences here; the official layer occurrence compiler is.
        return "compressor.partial_state", "high"

    # Attention projections, cache updates, sparse MQA and output projection.
    if "_fused_q_kv_rmsnorm_kernel" in kernel:
        return "attention.q_norm", "high"
    if "fuseddeepseekv4qnormropek" in kernel or cpu.startswith(
        "_c::fused_deepseek_v4_qnorm_rope"
    ):
        return "attention.q_head_norm", "high"
    if "fused_inv_rope" in kernel or cpu == "vllm::fused_inv_rope_fp8_quant_kernel":
        return "attention.inverse_rope", "high"
    if _has_any(
        kernel,
        "sparse_attn_fwd_kernel",
        "flash_fwd_splitkv_mla_fp8_sparse_kernel",
        "flash_fwd_mla_combine_kernel",
    ):
        return "attention.sparse_mqa", "high"
    if "sm100_fp8_fp4_gemm_1d1d_impl" in kernel:
        if "2048u, 7168u" in kernel:
            return "attention.q_a", "high"
        if "8192u, 1536u" in kernel:
            return (
                ("csa_indexer.q_projection", "high")
                if in_indexer
                else ("attention.q_b", "high")
            )
        if "1024u, 4096u" in kernel and "(deep_gemm::gemmtype)4" in kernel:
            return "attention.grouped_o_a", "high"
        if "7168u, 2048u" in kernel:
            return "attention.o_b", "high"
    if cpu == "_c::per_token_group_fp8_quant_packed" and not in_moe:
        if "128, 8, 2" in kernel:
            return "attention.q_a", "high"
        if "128, 4, 4" in kernel:
            return (
                ("csa_indexer.q_projection", "high")
                if in_indexer
                else ("attention.q_b", "high")
            )
        if "128, 16, 1" in kernel:
            return "attention.o_b", "high"
    if "deep_gemm_fp8_o_proj" in lowered:
        if "o_proj" in kernel or "grouped" in kernel:
            return "attention.grouped_o_a", "high"
        return "attention.o_b", "medium"
    if in_attention and cpu == "aten::mm":
        if "fused_wqa_wkv" in lowered:
            return "attention.q_a", "high"
        if "wq_b_kv_insert" in lowered or "columnparallellinear" in lowered:
            return "attention.q_b", "medium"
        if "compressor_kv_score" in lowered and "indexer_compressor" not in lowered:
            return "compressor.kv_gate_projection", "high"
        if "indexer_weights_proj" in lowered:
            return "csa_indexer.weight_projection", "high"
        if "indexer_compressor_kv_score" in lowered:
            return "csa_indexer.k_compress", "high"

    if in_indexer and cpu == "aten::mm":
        return "csa_indexer.q_projection", "medium"

    # Router, routed experts, shared expert and the TP boundary.
    if (
        in_moe
        and (
            (cpu == "aten::mm" and "gatelinear" in lowered)
            or cpu == "_moe_c::dsv3_router_gemm"
            or "router_gemm_kernel_float_output" in kernel
        )
    ):
        return "moe.score_projection", "high"
    if "topkgatingsoftplussqrt" in kernel or cpu == "_moe_c::topk_softplus_sqrt":
        return (
            ("moe.hash_select", "high")
            if ", true," in kernel
            else ("moe.learned_select", "high")
        )
    if cpu == "vllm::mxfp8_quantize":
        return "moe.dispatch", "high"
    if cpu == "vllm::moe_forward_shared":
        if "routingindices" in kernel or "_pack_topk_ids_weights_kernel" in kernel:
            return "moe.dispatch", "high"
        if "bmm_mxe4m3" in kernel:
            return "moe.routed_gate_up", "high"
        if kernel.startswith("bmm_bfloat16"):
            return "moe.routed_down", "high"
        if "finalizekernel" in kernel:
            return "moe.routed_combine", "high"
    if in_moe and cpu == "_c::silu_and_mul_with_clamp":
        return "moe.shared_activation", "high"
    if in_moe and cpu == "vllm::fp8_gemm_nt_op":
        if "768u, 7168u" in kernel:
            return "moe.shared_gate_up", "high"
        if "7168u, 384u" in kernel:
            return "moe.shared_down", "high"
        return "moe.shared_gate_up", "medium"
    if in_moe and cpu == "_c::per_token_group_fp8_quant_packed":
        if "rowparallellinear" in lowered:
            return "moe.shared_down", "medium"
        return "moe.shared_gate_up", "medium"
    if in_moe and cpu == "aten::add":
        return "moe.combine", "high"

    if in_logits and cpu == "aten::mm":
        return "top.lm_head", "high"
    if in_embedding:
        return "top.embedding", "high"
    if cpu == "_c::rms_norm" and "deepseekv4model" in lowered:
        return "top.final_norm", "high"

    # Scheduler metadata, cache-slot preparation, sampling support, and tiny
    # implementation bookkeeping remain explicit runtime support rather than
    # unexplained/unmapped events.  The first validation pass audits every
    # model-context event that falls through here before acceptance.
    return "top.runtime_support", "low"


DEEPSEEK_V4_PRO_VLLM_TRACE_RULES = TraceMappingRules(
    model_id="deepseek_v4_pro_vllm",
    signature_kernel=DSV4_FINAL_HEAD_SIGNATURE,
    signature_count_per_forward=1,
    stack=StackFrameRules(
        operator_patterns=(
            "vllm/model_executor/",
            "vllm/v1/attention/",
            "vllm/v1/worker/",
            "vllm/distributed/",
            "vllm/models/deepseek_v4/",
        ),
        semantic_patterns=(
            "vllm/models/deepseek_v4/",
            "vllm/model_executor/",
            "vllm/v1/attention/",
            "vllm/v1/worker/",
            "vllm/distributed/",
        ),
        model_context_patterns=(
            "DeepseekV4ForCausalLM",
            "DeepseekV4Model",
            "DeepseekV4DecoderLayer",
            "DeepseekV4FlashMLAAttention",
            "DeepseekV4Indexer",
            "DeepseekCompressor",
            "DeepseekV4MoE",
            "FusedMoE",
        ),
        phase_patterns=("gpu_model_runner.py", "execute_model"),
    ),
    classify_node=classify_deepseek_v4_pro_vllm_node,
    kernel_only_nodes=frozenset(
        {
            "final_hc_read.read",
            "mhc_transform.affine",
            "mhc_transform.mix",
            "csa_compressor.softmax_pool",
            "hca_compressor.softmax_pool",
            "hca_compressor.norm_rope",
            "csa_indexer.k_compress",
            "csa_indexer.q_projection",
            "csa_indexer.q_rope_rotate",
            "csa_indexer.causal_topk",
            "csa_indexer.expand",
            "attention.index_union",
            "attention.q_norm",
            "attention.q_head_norm",
            "attention.inverse_rope",
            "attention.sparse_mqa",
            "moe.hash_select",
            "moe.learned_select",
        }
    ),
)
