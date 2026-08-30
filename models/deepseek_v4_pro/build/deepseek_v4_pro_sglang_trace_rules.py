#!/usr/bin/env python3
"""SGLang eager-trace rules for DeepSeek-V4-Pro-0813 pure TP8.

The rules describe the pinned SGLang implementation boundaries only.  Shared
``attention.*`` and ``compressor.*`` labels are expanded through the official
61-layer CSA/HCA schedule by the DeepSeek-V4-Pro occurrence compiler.
Implementation-unique fused kernels are deliberately classified before their
occasionally stale asynchronous Python launch stacks.
"""

from __future__ import annotations

from models.common.trace_mapping import FrameRef, StackFrameRules, TraceMappingRules


DSV4_SGLANG_ATTENTION_SIGNATURE = "fused_q_norm_rope"
DSV4_LAYER_COUNT = 61


def _has_any(text: str, *needles: str) -> bool:
    return any(needle in text for needle in needles)


def classify_deepseek_v4_pro_sglang_node(
    kernel_name: str,
    cpu_op_name: str | None,
    stack: list[FrameRef],
) -> tuple[str | None, str]:
    frames = "\n".join(frame.raw for frame in stack).lower()
    kernel = kernel_name.lower()
    cpu = (cpu_op_name or "").lower()

    in_embedding = "vocab_parallel_embedding" in frames
    in_logits = "logits_processor" in frames
    in_indexer = "layers/attention/dsv4/indexer.py" in frames
    in_main_compressor = (
        "compressor_v2.py" in frames and not in_indexer
    )
    in_shared_expert = "_forward_shared_experts" in frames
    in_moe = _has_any(
        frames,
        "deepseek_v2.py",
        "fused_moe_triton",
        "mxfp4_flashinfer_trtllm_moe.py",
    )

    # Exact TP semantic boundaries.
    if _has_any(kernel, "allgather", "all_gather") or cpu == "record_param_comms":
        if in_logits or "lm_head" in frames:
            return "top.tp_logits_collective", "high"
    if _has_any(kernel, "allreduce", "all_reduce", "oneshotallreducefusionkernel") \
            or cpu == "sglang::flashinfer_allreduce":
        if in_embedding:
            return "top.tp_embedding_output_collective", "high"
        if "deepseek_v2.py" in frames or "forward_normal" in frames:
            return "moe.tp_moe_output_collective", "high"
        if "linear.py" in frames and "deepseek_v4.py" in frames:
            return "attention.tp_output_collective", "high"

    # mHC launch groups.  ``fused_post_pre_fma`` owns the preceding post-mix;
    # ``pre_big`` owns the following pre-transform and attention/FFN RMSNorm.
    if "mhc_fused_post_pre_fma" in kernel or "mhc_post_tilelang" in kernel:
        return "mhc_transform.mix", "high"
    if _has_any(
        kernel,
        "mhc_pre_big_fuse_with_norm",
        "sm100_tf32_hc_prenorm_gemm",
        "mhc_pre_gemm_sqrsum",
        "hc_split_sinkhorn",
    ):
        return "mhc_transform.affine", "high"
    if "_hc_head_kernel" in kernel:
        return "final_hc_read.read", "high"

    # Unique model kernels must win over a stale asynchronous launch stack
    # that can point at the scheduler preparing the following decode step.
    if "sm100_paged_mqa_logits" in kernel:
        return "csa_indexer.score", "high"
    if "_get_k_and_s_triton_kernel" in kernel or "sm100_mqa_logits" in kernel:
        return "csa_indexer.score", "high"
    if "topk_transform_kernel" in kernel:
        return "csa_indexer.causal_topk", "high"
    if "_combine_topk_swa_indices_kernel" in kernel:
        return "attention.index_union", "high"
    if "_dequantize_k_cache_paged_kernel" in kernel:
        return "attention.sparse_mqa", "high"
    if cpu == "sgl_kernel::sparse_prefill_fwd" or (
        cpu == "aten::mul" and "_forward_prefill_sparse" in frames
    ):
        return "attention.sparse_mqa", "high"
    if (
        "sm100_fp8_fp4_gemm_1d1d_impl" in kernel
        and "1024u, 4096u" in kernel
        and "(deep_gemm::gemmtype)4" in kernel
    ):
        return "attention.grouped_o_a", "high"
    if "_router_triton_kernel" in kernel:
        return "moe.learned_select", "high"
    if "nvjet_sm103_tst_64x64_64x13" in kernel:
        return "csa_indexer.weight_projection", "high"
    if "nvjet_sm103_tss_192x128_64x6" in kernel:
        return "moe.score_projection", "high"
    if "_pack_topk_ids" in kernel:
        return "moe.dispatch", "high"
    if "routingindices" in kernel:
        return "moe.dispatch", "high"
    if kernel.startswith("bmm_mxe4m3_"):
        return "moe.routed_gate_up", "high"
    if kernel.startswith("bmm_bfloat16_"):
        return "moe.routed_down", "high"

    # Framework/scheduler support that is inside the exact forward window but
    # outside the stable mathematical model graph.
    if _has_any(
        frames,
        "cuda_graph_buffer_registry.py",
        "deepseek_v4_backend.py",
        "dsv4_attn_metadata_kernels.py",
        "metadata_kernel.py",
        "allocation.py",
        "mem_cache",
    ) and not _has_any(
        frames,
        "deepseek_v4.py",
        "compressor_v2.py:forward_unified",
    ):
        return "top.runtime_support", "support"
    if _has_any(
        kernel,
        "_page_table_positions_kernel",
        "_causal_swa_page_indices_kernel",
        "_init_compressed_attn_metadata_kernel",
        "paged_mqa_metadata",
        "topk_plan",
        "plan_compress_decode_kernel",
    ):
        return "top.runtime_support", "support"

    # Top-level value path.
    if "_vocab_parallel_embedding_kernel" in kernel or in_embedding:
        return "top.embedding", "high"
    if cpu == "aten::copy_" and "deepseek_v4model" in frames:
        return "top.hc_expand", "medium"
    if in_logits and (cpu == "aten::mm" or "_compute_lm_head" in frames):
        return "top.lm_head", "high"
    if "logits_processor" in frames:
        return "top.logits", "high"

    # Branch-specific compressor/indexer kernels precede shared attention.
    if "fused_q_indexer_rope_hadamard_quant" in kernel:
        return "csa_indexer.q_projection", "high"
    if in_indexer and cpu == "aten::mm" and "compute_weights" in frames:
        return "csa_indexer.weight_projection", "high"
    if in_indexer and cpu == "aten::mm" and "compute_kv_score" in frames:
        return "csa_indexer.k_compress", "high"
    if "flash_c4_decode<128" in kernel or "fused_norm_rope_indexer" in kernel:
        return "csa_indexer.k_compress", "high"
    if "topk_main_kernel" in kernel:
        return "csa_indexer.causal_topk", "high"
    if in_main_compressor and cpu == "aten::mm" and "compute_kv_score" in frames:
        return "compressor.kv_gate_projection", "high"
    if "flash_c128_decode" in kernel or "flash_c128_prefill" in kernel:
        return "hca_compressor.softmax_pool", "high"
    if "flash_c4_decode<512" in kernel or "flash_c4_prefill<512" in kernel \
            or "write_c4_prefill<512" in kernel:
        return "csa_compressor.softmax_pool", "high"
    if "flash_c4_prefill<128" in kernel or "write_c4_prefill<128" in kernel:
        return "csa_indexer.k_compress", "high"
    if "fused_norm_rope_flashmla" in kernel:
        return "compressor.partial_state", "high"

    # Shared attention projections, cache updates, sparse core and output.
    if "rmsnormkernel" in kernel and "_forward_prepare" in frames:
        return "attention.q_norm", "high"
    if "fused_q_norm_rope" in kernel:
        return "attention.q_head_norm", "high"
    if "fused_k_norm_rope_flashmla" in kernel:
        return "attention.window_kv", "high"
    if cpu == "sgl_kernel::sparse_decode_fwd" or _has_any(
        kernel,
        "flash_fwd_splitkv_mla_fp8_sparse_kernel",
        "flash_fwd_mla_combine_kernel",
    ):
        return "attention.sparse_mqa", "high"
    if "deepseek_rope_kernel" in kernel:
        return "attention.inverse_rope", "high"
    if "fp8_wo_a_group_major_quant" in kernel:
        return "attention.grouped_o_a", "high"

    # MoE signatures take precedence over generic quant/GEMM handling.
    if "router_gemm_kernel" in kernel or cpu == "sglang::dsv3_router_gemm":
        return "moe.score_projection", "high"
    if cpu == "aten::mm" and "nn.module: moegate_" in frames:
        return "moe.score_projection", "high"
    if "moe_hash_topk_fused" in kernel:
        return "moe.hash_select", "high"
    if "_router_triton_kernel" in kernel:
        return "moe.learned_select", "high"
    if "_pack_topk_ids" in kernel:
        return "moe.dispatch", "high"
    if "mxfp8_quantize" in kernel or cpu == "sglang::flashinfer_mxfp8_quantize":
        return "moe.dispatch", "high"
    if "routingindices" in kernel:
        return "moe.dispatch", "high"
    if kernel.startswith("bmm_mxe4m3_"):
        return "moe.routed_gate_up", "high"
    if kernel.startswith("bmm_bfloat16_"):
        return "moe.routed_down", "high"
    if "finalizekernel" in kernel:
        return "moe.routed_combine", "high"
    if cpu == "aten::add_" and "maybe_fuse_routed_scale_and_shared_add" in frames:
        return "moe.combine", "high"
    if "silu_mul_clamp" in kernel or "silu_and_mul_clamp" in frames:
        return "moe.shared_activation", "high"

    # FP8 projection groups.  Python module scope separates otherwise shared
    # physical quant/GEMM kernels; dimensions disambiguate grouped output A.
    is_quant = cpu == "sglang::per_token_group_quant"
    is_fp8_gemm = cpu == "sglang::deep_gemm_fp8_fp8_bf16_nt" or \
        "sm100_fp8_fp4_gemm_1d1d_impl" in kernel
    if is_quant or is_fp8_gemm:
        if in_shared_expert:
            # The two ordered shared-expert projections are split by the
            # explicit SiLU-clamp kernel in every layer occurrence.
            if "_forward_shared_experts" in frames:
                return "moe.shared_gate_up", "medium"
        if in_indexer and "compute_q" in frames:
            return "csa_indexer.q_projection", "high"
        if "_compute_q_b" in frames:
            return "attention.q_b", "high"
        if "_forward_prepare" in frames:
            return "attention.q_a", "medium"
        if "deepseek_v4.py" in frames:
            if "1024u, 4096u" in kernel and "(deep_gemm::gemmtype)4" in kernel:
                return "attention.grouped_o_a", "high"
            return "attention.o_b", "medium"
        if in_moe:
            return "moe.shared_down", "medium"

    # Final normalization and vocabulary projection follow the final mHC head.
    if "rmsnormkernel" in kernel and "deepseek_v4.py" in frames:
        return "top.final_norm", "high"
    # Every remaining exact-window kernel is typed support, never an
    # unexplained model leaf.  The occurrence compiler assigns a concrete
    # support class and rejects an unrecognized class.
    return "top.runtime_support", "support"


DEEPSEEK_V4_PRO_SGLANG_TRACE_RULES = TraceMappingRules(
    model_id="deepseek_v4_pro_sglang",
    signature_kernel=DSV4_SGLANG_ATTENTION_SIGNATURE,
    signature_count_per_forward=DSV4_LAYER_COUNT,
    stack=StackFrameRules(
        operator_patterns=(
            "python/sglang/kernels",
            "python/sglang/srt/layers",
            "python/sglang/srt/distributed",
            "python/sglang/srt/mem_cache",
            "python/sglang/srt/model_executor",
        ),
        semantic_patterns=(
            "python/sglang/srt/models/deepseek_v4.py",
            "python/sglang/srt/models/deepseek_v2.py",
            "python/sglang/srt/layers/attention/dsv4",
            "python/sglang/srt/layers/attention/deepseek_v4_backend.py",
            "python/sglang/srt/layers/moe",
            "python/sglang/srt/layers/logits_processor.py",
            "python/sglang/srt/layers/vocab_parallel_embedding.py",
            "python/sglang/srt/model_executor/runner/eager_runner.py",
        ),
        model_context_patterns=(
            "DeepseekV4DecoderLayer_",
            "DeepseekV4ForCausalLM",
            "DeepseekV4Model",
            "deepseek_v4.py",
            "deepseek_v2.py",
        ),
        phase_patterns=("_execute_extend", "_execute_decode"),
    ),
    classify_node=classify_deepseek_v4_pro_sglang_node,
    kernel_only_nodes=frozenset(
        {
            "mhc_transform.affine",
            "mhc_transform.mix",
            "final_hc_read.read",
            "csa_indexer.q_projection",
            "csa_indexer.k_compress",
            "csa_indexer.score",
            "csa_indexer.causal_topk",
            "csa_indexer.weight_projection",
            "csa_compressor.softmax_pool",
            "hca_compressor.softmax_pool",
            "compressor.partial_state",
            "attention.q_head_norm",
            "attention.window_kv",
            "attention.index_union",
            "attention.sparse_mqa",
            "attention.inverse_rope",
            "attention.grouped_o_a",
            "moe.hash_select",
            "moe.learned_select",
            "moe.score_projection",
            "moe.dispatch",
            "moe.routed_gate_up",
            "moe.routed_down",
            "moe.routed_combine",
        }
    ),
)
