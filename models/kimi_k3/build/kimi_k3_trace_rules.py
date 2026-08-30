#!/usr/bin/env python3
"""Kimi K3 rules for the common eager Torch-trace mapper.

The classifier is intentionally source- and kernel-specific.  It never uses a
nearest-neighbour fallback: each returned node is justified by an exact Kimi
source frame, a shape-specialized physical signature, or both.  A handful of
asynchronously launched kernels have stale Python spans; only signatures that
are unique to one Kimi semantic group are allowed to override those spans.
"""

from __future__ import annotations

import re

from models.common.trace_mapping import FrameRef, StackFrameRules, TraceMappingRules


KDA_LAYERS_PER_FORWARD = 69
KDA_DECODE_SIGNATURE = "kda_decode_fusion_many_heads_kernel"
GATED_MLA_LAYER_INDICES = frozenset(
    [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 92]
)
_DECODER_LAYER_RE = re.compile(r"nn\.Module: KimiK3DecoderLayer_(\d+)")


def _has(names: str, *needles: str) -> bool:
    return any(needle in names for needle in needles)


def _decoder_layer_index(stack: list[FrameRef]) -> int | None:
    for frame in stack:
        match = _DECODER_LAYER_RE.search(frame.raw)
        if match:
            return int(match.group(1))
    return None


def classify_kimi_k3_node(
    kernel_name: str,
    cpu_op_name: str | None,
    stack: list[FrameRef],
) -> tuple[str | None, str]:
    names = "\n".join(frame.raw for frame in stack).lower()
    kernel = kernel_name.lower()
    cpu = (cpu_op_name or "").lower()
    decoder_layer = _decoder_layer_index(stack)

    # Implementation-unique Kimi signatures win over stale launch spans from
    # the overlapped KDA, MLA, routed-expert, and shared-expert streams.
    if "attn_res_fused_tma_kernel" in kernel:
        if "kimik3decoderlayer" not in names and "kimi_k3.py(2412)" in names:
            return "top.output_attn_res", "high"
        return "attn_res.weighted_merge", "high"
    if KDA_DECODE_SIGNATURE in kernel:
        return "kda.recurrent_update", "high"
    if "causal_conv1d_fwd_kernel" in kernel or "causal_conv1d_update" in kernel:
        # Prefill launches Q/K/V depthwise convolutions in a fixed three-call
        # group.  The profile builder preserves the 1:N group and marks K/V as
        # children of this single measured owner rather than copying timing.
        return "kda.q_short_conv", "high"
    if "l2norm_fwd_kernel" in kernel:
        return "kda.qk_l2_norm", "high"
    if _has(
        kernel,
        "kda_gate_chunk_cumsum",
        "chunk_kda_fwd_kernel_intra",
        "chunk_kda_fwd_kernel_inter",
        "recompute_w_u",
        "chunk_gated_delta_rule",
    ):
        return "kda.recurrent_update", "high"
    if "chunk_gla_fwd_kernel_o" in kernel:
        return "kda.query_readout", "high"
    if "layer_norm_gated_fwd_kernel" in kernel:
        return "kda.gated_rmsnorm", "high"
    if "kimik3deltaattention" in names and "64x8_64x16_4x2" in kernel:
        # Rank-local launch timing can make this tiny f_b projection lose the
        # inner forward_qkvbfg_fused span.  The DeltaAttention module context
        # plus its shape-specialized GEMM signature remains unique.
        return "kda.decay_projection", "high"

    if "set_mla_kv_buffer_kernel" in kernel or "set_mla_kv_concat_q_kernel" in kernel:
        return "gated_mla.cache_update", "high"
    if "fmha" in kernel and _has(kernel, "hqk192", "hqk576"):
        return "gated_mla.attention", "high"
    if "mla_output_gate_kernel" in kernel:
        return "gated_mla.gated_context", "high"
    if "fused_a_gemm_kernel" in kernel:
        # The decode A projection jointly materializes q_down and kv_down.
        return "gated_mla.q_down", "high"
    if cpu == "aten::bmm" and "forward_mla.py" in names:
        # Decode algebraically absorbs the MLA kv_up weights into two BMMs:
        # query-to-latent before the cache attention and latent-to-value after
        # it.  The attention contraction is the single measured owner; kv_up
        # is a non-owner semantic child linked to this owner in the profile.
        return "gated_mla.attention", "high"

    if "route_radix_kernel" in kernel or "route_quant_fused_kernel" in kernel:
        return "stable_latent_moe.corrected_selection", "high"
    if "routingindicesclusterkernel" in kernel or "routingindicesblockkernel" in kernel:
        return "stable_latent_moe.dispatch", "high"
    if "per_token_group_quant_flat_kernel" in kernel or "pack_topk_ids" in kernel:
        return "stable_latent_moe.dispatch", "high"
    if "bmm_mxe4m3_mxe2m1mxe4m3" in kernel:
        return "stable_latent_moe.expert_gate_up", "high"
    if "bmm_bfloat16_mxe2m1mxe4m3" in kernel:
        return "stable_latent_moe.expert_down", "high"
    if "moe::dev::finalize::finalizekernel" in kernel:
        return "stable_latent_moe.weighted_reduce", "high"

    # Communication is attributed to the exact semantic boundary before any
    # enclosing module fallback can absorb it as compute.
    if _has(kernel, "all_reduce_pull_norm", "all_reduce_push_norm_cluster"):
        return "stable_latent_moe.tp_routed_latent_collective", "high"
    if "all_reduce_pull_res_kernel" in kernel and "kimi_k3.py(1033)" in names:
        return "stable_latent_moe.tp_shared_expert_collective", "high"
    if _has(kernel, "all_reduce_push_res_kernel", "all_reduce_pull_res_kernel"):
        # K3 defers every attention o_proj TP reduction to the decoder layer,
        # where the collective also folds in the live AttnRes prefix.  The
        # launch stack retains the exact configured decoder-layer index even
        # when the CUDA stream executes after later Python work has launched.
        if decoder_layer is not None:
            if decoder_layer in GATED_MLA_LAYER_INDICES:
                return "gated_mla.tp_mla_output_collective", "high"
            return "kda.tp_kda_output_collective", "high"
        if "kimi_k3.py(1644)" in names or "kimik3deltaattention" in names:
            return "kda.tp_kda_output_collective", "high"
        if "kimi_k3.py(1860)" in names or "kimik3mlaattention" in names:
            return "gated_mla.tp_mla_output_collective", "high"
    if "all_reduce" in kernel or "allreduce" in kernel:
        if "vocab_parallel_embedding" in names:
            return "top.tp_embedding_output_collective", "high"
        if "logits_processor" in names or "lm_head" in names:
            return "top.tp_logits_materialization", "high"
        if "kimi_k3.py(289)" in names or "kimik3mlp" in names:
            return "dense_mlp.tp_dense_output_collective", "high"
        if "kimi_k3.py(1644)" in names or "kimik3deltaattention" in names:
            return "kda.tp_kda_output_collective", "high"
        if "kimi_k3.py(1860)" in names or "kimik3mlaattention" in names:
            return "gated_mla.tp_mla_output_collective", "high"
    if "all_gather" in kernel or "allgather" in kernel:
        if "logits_processor" in names or "lm_head" in names:
            return "top.tp_logits_materialization", "high"

    # Stable LatentMoE's fused front and ordered tail.
    if "kimi_k3.py(1033)" in names or "kimik3moe" in names:
        if "kimi_k3.py(1015)" in names:
            if "situ_and_mul" in kernel:
                return "stable_latent_moe.shared_situ", "high"
            if "kimi_k3.py(136)" in names or cpu == "aten::mm":
                return "stable_latent_moe.shared_down", "high"
        if "topk.py" in names:
            return "stable_latent_moe.corrected_selection", "medium"
        if "fused_moe" in names or "forward_deferred_finalize" in names:
            return "stable_latent_moe.weighted_reduce", "medium"
        if "layers/linear.py(279)" in names:
            return "stable_latent_moe.routed_up", "high"
        if "kimi_k3.py(136)" in names and cpu == "aten::mm":
            # One [shared gate/up | router | routed-down] GEMM.  Router logits
            # are its measured owner; the other semantic children link to it.
            return "stable_latent_moe.router_logits", "high"
        if "add3_kernel" in kernel or cpu == "aten::add":
            return "stable_latent_moe.combine", "high"

    # Dense layer zero is the only KimiK3MLP outside Stable LatentMoE.
    if "kimi_k3.py(289)" in names or "kimik3mlp" in names:
        if "situ_and_mul" in kernel:
            return "dense_mlp.situ", "high"
        if "layers/linear.py(469)" in names:
            return "dense_mlp.gate_up", "high"
        if "layers/linear.py(1563)" in names or "layers/linear.py(279)" in names:
            return "dense_mlp.down", "high"
        if cpu == "aten::add":
            return "decoder_stack.prefix_after_ffn", "medium"

    # KDA projections and phase-local state operations.
    if "kimi_k3.py(1596)" in names or "forward_qkvbfg_fused" in names:
        if "tiny_n_gemm_kernel" in kernel or "80x128" in kernel:
            return "kda.beta_projection", "high"
        if "176x128" in kernel or "64x8_64x16_4x2" in kernel:
            return "kda.decay_projection", "high"
        return "kda.qkv_projection", "medium"
    if "kda_backend.py" in names:
        if "init_forward_metadata" in names:
            return "runtime.step_setup", "high"
        return "kda.recurrent_update", "medium"
    if "kimi_k3.py(1644)" in names or "kimik3deltaattention" in names:
        if "layers/linear.py(1563)" in names or "layers/linear.py(279)" in names:
            return "kda.output_projection", "high"
        if cpu == "aten::sigmoid":
            return "kda.beta_projection", "high"
        if "layernormgatedfunction" in cpu:
            return "kda.gated_rmsnorm", "high"
        if cpu == "aten::copy_":
            return "kda.qkv_projection", "medium"
        return "kda.kda_out", "low"

    # Gated MLA projections.  Shape-specialized GEMM signatures distinguish
    # q_up from kv_up even though both use the same column-parallel wrapper.
    if "kimi_k3.py(1860)" in names or "kimik3mlaattention" in names:
        if "trtllm_mla_backend.py" in names or "radix_attention.py" in names:
            if "init_forward_metadata" in names or "_create_block_kv_indices" in names:
                return "runtime.step_setup", "high"
            return "gated_mla.attention", "medium"
        if "kimi_k3.py(1810)" in names:
            if "layers/linear.py(1563)" in names:
                return "gated_mla.output_projection", "high"
            if "layers/linear.py(469)" in names or cpu == "aten::mm":
                return "gated_mla.output_gate", "high"
        if "layers/linear.py(279)" in names:
            return "gated_mla.q_down", "high"
        if "128x192" in kernel:
            return "gated_mla.kv_up", "high"
        if "64x8_64x16_4x2" in kernel or "layers/linear.py(469)" in names:
            return "gated_mla.q_up", "medium"
        if "rmsnorm" in kernel:
            if "o512" in kernel:
                return "gated_mla.kv_norm", "high"
            return "gated_mla.q_norm", "high"
        if cpu == "aten::copy_":
            return "gated_mla.key_compose", "medium"
        return "gated_mla.mla_out", "low"

    if "attn_residual.py" in names:
        if "write" in names:
            return "decoder_stack.block_write", "high"
        return "attn_res.weighted_merge", "medium"

    # Model input/output boundaries.
    if "vocab_parallel_embedding" in names or "vocab_parallel_embedding_kernel" in kernel:
        return "top.embedding", "high"
    if "logits_processor.py(754)" in names or "_compute_lm_head" in names:
        return "top.lm_head", "high"
    if "logits_processor" in names:
        return "top.logits", "high"

    # These are exact per-step allocation/index updates outside the portable
    # Model IR.  They remain visible in the timeline as explicitly named local
    # runtime support rather than being mislabeled as model semantics.
    if _has(
        names,
        "eager_runner.py",
        "model_runner.py",
        "memory_pool.py",
        "cuda_graph_buffer_registry.py",
        "hybrid_linear_attn_backend.py",
        "trtllm_mla_backend.py",
    ):
        return "runtime.step_setup", "medium"

    return None, "unmapped"


KIMI_K3_TRACE_RULES = TraceMappingRules(
    model_id="kimi_k3",
    signature_kernel=KDA_DECODE_SIGNATURE,
    signature_count_per_forward=KDA_LAYERS_PER_FORWARD,
    stack=StackFrameRules(
        operator_patterns=(
            "layers/linear.py",
            "layers/moe/",
            "layers/attention/linear/kda_backend.py",
            "layers/attention/trtllm_mla_backend.py",
            "layers/radix_attention.py",
            "layers/attn_residual.py",
            "kernels/ops/kimi_k3/",
            "layers/logits_processor.py",
            "layers/vocab_parallel_embedding.py",
        ),
        semantic_patterns=(
            "models/kimi_k3.py",
            "layers/attention/linear/kda_backend.py",
            "layers/attention/trtllm_mla_backend.py",
            "layers/radix_attention.py",
            "layers/attn_residual.py",
            "kernels/ops/kimi_k3/",
            "layers/moe/",
            "layers/logits_processor.py",
            "layers/vocab_parallel_embedding.py",
            "model_executor/runner/eager_runner.py",
            "model_executor/model_runner.py",
            "mem_cache/memory_pool.py",
            "cuda_graph_buffer_registry.py",
            "hybrid_linear_attn_backend.py",
        ),
        model_context_patterns=(
            "KimiK3ForConditionalGeneration",
            "KimiK3LinearForCausalLM",
            "KimiK3LinearModel",
            "KimiK3DecoderLayer",
            "KimiK3DeltaAttention",
            "KimiK3MLAAttention",
            "KimiK3MoE",
            "KimiK3MLP",
        ),
        phase_patterns=(
            "_execute_extend",
            "_execute_decode",
            "forward_extend",
            "forward_decode",
            "cuda_graph_runner",
            "replay",
        ),
    ),
    classify_node=classify_kimi_k3_node,
    kernel_only_nodes=frozenset(
        {
            "attn_res.weighted_merge",
            "kda.q_short_conv",
            "kda.k_short_conv",
            "kda.v_short_conv",
            "kda.qk_l2_norm",
            "kda.recurrent_update",
            "kda.query_readout",
            "kda.gated_rmsnorm",
            "gated_mla.q_down",
            "gated_mla.cache_update",
            "gated_mla.attention",
            "gated_mla.gated_context",
            "stable_latent_moe.corrected_selection",
            "stable_latent_moe.dispatch",
            "stable_latent_moe.expert_gate_up",
            "stable_latent_moe.expert_down",
            "stable_latent_moe.weighted_reduce",
        }
    ),
)
