#!/usr/bin/env python3
"""Qwen3.8-Flash-Next rules for the common eager Torch-trace mapper."""

from __future__ import annotations

from models.common.trace_mapping import FrameRef, StackFrameRules, TraceMappingRules


QWEN38_FLASH_NEXT_GDN_SIGNATURE_KERNEL = "fused_qkvzba_split"
QWEN38_FLASH_NEXT_GDN_LAYERS_PER_FORWARD = 36
QWEN38_FLASH_NEXT_CONFIGS = {
    "tp_only",
    "tp4_flashinfer_gdn",
    "dp_attention",
    "ep4_a2a_none",
    "dp_attention_ep4_deepep_deepgemm",
}


def _classify_qwen38_flash_next_mtp_node(
    kernel_name: str,
    cpu_op_name: str | None,
    stack: list[FrameRef],
) -> tuple[str | None, str]:
    """Classify the auxiliary MTP module in its own semantic scope.

    The MTP head reuses QSA, MoE and hyper-connection implementations from the
    target model.  The enclosing ``Qwen4ExpForCausalLMMTP`` frame is therefore
    what prevents their kernels from being silently aggregated into the
    48-layer target graph.
    """

    names = "\n".join(frame.raw for frame in stack)
    lowered = names.lower()
    kernel = kernel_name.lower()
    cpu = (cpu_op_name or "").lower()

    if "allgather" in kernel or "all_gather" in kernel:
        return "mtp_head.tp_logits_collective", "high"
    if "allreduce" in kernel or "all_reduce" in kernel:
        if "vocabparallelembedding" in lowered:
            return "mtp_head.tp_embedding_collective", "high"
        if "qwen2moe" in lowered or "fusedmoe" in lowered:
            return "mtp_layer.tp_moe_output_collective", "high"
        if "qwen4expattentiondecoderlayer" in lowered:
            return "mtp_layer.tp_attention_collective", "high"

    if "_fuse_residual_linear_shared" in lowered:
        if "pre_fc_norm_embedding" in lowered or "fc_embedding" in lowered:
            return "mtp_head.embedding_projection", "high"
        if "pre_fc_norm_hidden" in lowered or "fc_hidden" in lowered:
            return "mtp_head.hidden_projection", "high"
        return "mtp_head.residual_fusion", "medium"

    if "gatedresidualsimple" in lowered or "layers/hyperconnection.py" in lowered:
        stage = None
        if "_prepare_qwen4_exp_attn" in lowered:
            stage = "attn_hc_mix"
        elif "_prepare_qwen4_exp_mlp" in lowered:
            if "combine" in lowered:
                stage = "attn_hc_combine"
            else:
                stage = "mlp_hc_mix"
        elif "_postprocess_qwen4_exp_layer" in lowered:
            stage = "mlp_hc_combine"
        if stage is None:
            return "mtp_head.final_hc_mix", "high"
        # HC leaf semantics are implementation-independent and reusable. The
        # enclosing MTP module remains recorded as layer_kind/substage by the
        # profile builder, so these leaves can drive a correctly scoped drill
        # view without being aggregated into target-model HC metrics.
        if "grouped_gemma_rmsnorm" in kernel:
            return "hyperconnection.branch_norm", "high"
        if stage.endswith("combine"):
            return "hyperconnection.combine", "high"
        return "hyperconnection.mix", "high"

    if any(
        signature in lowered
        for signature in ("qsaindexer", "_compute_qsa_topk_indices", "qsa_indexer.py")
    ):
        return "mtp_qsa_attention.indexer", "high"
    if "qwen4expattentiondecoderlayer" in lowered:
        if "o_proj" in lowered:
            return "mtp_qsa_attention.output_projection", "medium"
        if "_prepare_qkv_gate" in lowered or "qkv" in lowered:
            return "mtp_qsa_attention.qkv_gate_projection", "medium"
        if "rotary" in lowered or "rmsnorm" in kernel:
            return "mtp_qsa_attention.qk_norm_rope", "medium"
        if (
            "radixattention" in lowered
            or "radix_attention.py" in lowered
            or "attention" in cpu
            or "attention" in kernel
            or "fmha" in kernel
        ):
            return "mtp_qsa_attention.attention_core", "medium"
        if "sigmoid" in cpu:
            return "mtp_qsa_attention.output_gate", "medium"

    if "qwen2moemlp" in lowered or "_forward_shared_experts" in lowered:
        return "mtp_moe.shared_expert", "medium"
    if "qwen2moesparsemoeblock" in lowered or "fusedmoe" in lowered:
        if "moe::dev::routing" in kernel or "topk" in cpu:
            return "mtp_moe.topk", "high"
        if "moe::dev::finalize" in kernel:
            return "mtp_moe.combine", "high"
        if "moe::dev::activation" in kernel or "bmm_" in kernel or "grouped" in kernel:
            return "mtp_moe.routed_experts", "high"
        if "gate" in lowered and ("gemm" in kernel or "gemm" in cpu):
            return "mtp_moe.router", "medium"
        return "mtp_moe.routed_experts", "low"

    if "logitsprocessor" in lowered or "lm_head" in lowered:
        return "mtp_head.lm_head", "medium"
    if "vocabparallelembedding" in lowered:
        return "mtp_head.embedding", "medium"
    if "qwen4expforcausallmmtp" in lowered:
        return "mtp_head.decoder_layer", "low"
    return None, "unmapped"


def classify_qwen38_flash_next_node(
    kernel_name: str,
    cpu_op_name: str | None,
    stack: list[FrameRef],
) -> tuple[str | None, str]:
    names = "\n".join(frame.raw for frame in stack)
    kernel = kernel_name.lower()
    cpu = (cpu_op_name or "").lower()
    lowered_names = names.lower()

    if "qwen4_exp_mtp.py" in lowered_names or "qwen4expforcausallmmtp" in lowered_names:
        return _classify_qwen38_flash_next_mtp_node(kernel_name, cpu_op_name, stack)

    # EAGLE orchestration launches a small amount of GPU work outside both the
    # target-model and auxiliary-model module scopes. Do not let those kernels
    # fall into generic runtime support: they are stable generation stages.
    in_model_scope = any(
        marker in lowered_names
        for marker in (
            "qwen4expmodel",
            "qwen4explineardecoderlayer",
            "qwen4expattentiondecoderlayer",
            "qwen4expforconditionalgeneration",
            "qwen3vlforconditionalgeneration",
        )
    )
    if not in_model_scope:
        if "_draft_extend_for_prefill" in lowered_names:
            return "mtp_generation.mtp_prefill", "high"
        if "_draft_extend_for_decode" in lowered_names or "prepare_for_draft_extend" in lowered_names:
            return "mtp_generation.mtp_draft_extend", "high"
        if any(
            marker in lowered_names
            for marker in ("draft_forward", "build_eagle_verify_input", "select_top_k_tokens")
        ):
            return "mtp_generation.draft_select", "high"
        if any(
            marker in lowered_names
            for marker in ("run_eagle_verify", "eagleverifyinput", "tree_speculative_sampling")
        ):
            return "mtp_generation.accept_commit", "high"

    # PLE context preparation and commit run outside the decoder-layer module
    # frames, so classify them from their exact source functions before the
    # module fallbacks below.  The prepared context can live across an entire
    # decoder layer before the configured PLE layer consumes it.
    if "_prepare_ple_batch" in names:
        return "ple.token_history", "high"
    if "_commit_ple_batch" in names:
        return "ple.context_commit", "high"

    # These model-specific signatures are semantically unique and must win
    # over an occasionally stale enclosing Python span in overlapped traces.
    if QWEN38_FLASH_NEXT_GDN_SIGNATURE_KERNEL in kernel:
        return "linear_attention.split_pack", "high"
    if "causal_conv1d" in kernel:
        return "linear_attention.causal_conv", "high"
    if any(
        signature in kernel
        for signature in (
            "fused_recurrent_gated_delta_rule_packed_decode",
            "gdn_decode_bf16state",
            "gdn_wide_vec_kernel",
            "chunk_gated_delta_rule",
            "recompute_w_u",
            "chunk_fwd_kernel_o",
            "l2norm_fwd",
        )
    ):
        return "linear_attention.delta_rule", "high"

    # Collectives must win over the enclosing module fallback.  In particular,
    # an MoE all-reduce still carries Qwen2Moe/FusedMoE frames and would
    # otherwise be mislabeled as expert compute.
    if "allgather" in kernel or "all_gather" in kernel:
        if "LogitsProcessor" in names or "lm_head" in names:
            return "top.tp_logits_collective", "high"

    if "allreduce" in kernel or "all_reduce" in kernel:
        if "VocabParallelEmbedding" in names:
            if "Qwen4ExpNGramEmbedding" in names:
                return "ple.tp_embedding_collective", "high"
            return "top.tp_embedding_collective", "high"
        if "Qwen2Moe" in names or "FusedMoE" in names:
            if "Qwen4ExpAttentionDecoderLayer" in names:
                return "full_layer.tp_moe_output_collective", "high"
            if "Qwen4ExpLinearDecoderLayer" in names:
                return "linear_layer.tp_moe_output_collective", "high"
            # Production traces carry a decoder-layer frame. Keep the generic
            # role only as a conservative fallback for reduced test stacks.
            return "moe.tp_output_collective", "high"
        if "Qwen4ExpAttentionDecoderLayer" in names:
            return "full_layer.tp_attention_collective", "high"
        if "Qwen4ExpLinearDecoderLayer" in names:
            return "linear_layer.tp_attention_collective", "high"

    if "Qwen4ExpPLE" in names or "Qwen4ExpNGramEmbedding" in names:
        if "_hash_contexts" in names:
            return "ple.ngram_hash", "high"
        if "ngram_embedding" in names or "_embed_ngram_ids" in names:
            return "ple.ngram_embedding", "high"
        if "key_proj" in names or "value_proj" in names:
            return "ple.key_value_projection", "high"
        if "GroupedNorm" in names or "rmsnorm" in kernel or "layer_norm" in kernel:
            return "ple.grouped_norm_gate", "high"
        if "_short_conv" in names or "conv" in kernel:
            return "ple.short_conv", "high"
        return "ple.injection", "medium"

    if "GatedResidualSimple" in names or "layers/hyperconnection.py" in names:
        if "combine" in names:
            return "hyperconnection.combine", "high"
        if "rmsnorm" in kernel or "layer_norm" in kernel:
            return "hyperconnection.branch_norm", "high"
        if "mix" in names:
            return "hyperconnection.mix", "high"
        return "hyperconnection.low_rank_gate", "medium"

    if "fused_gdn_gating" in kernel:
        return "linear_attention.gating", "high"
    if "Qwen3_5GatedDeltaNet" in names:
        if "out_proj" in names:
            return "linear_attention.output_projection", "medium"
        if "in_proj_ba" in names:
            return "linear_attention.ba_projection", "medium"
        if "in_proj_qkvz" in names or "_forward_input_proj" in names:
            return "linear_attention.qkvz_projection", "medium"
        if "norm" in names or "rmsnorm" in kernel:
            return "linear_attention.gated_norm", "medium"

    if any(
        signature in names
        for signature in (
            "QSAIndexer",
            "_compute_qsa_topk_indices",
            "qsa_indexer.py",
        )
    ):
        return "qsa_attention.indexer", "high"
    if "Qwen4ExpAttentionDecoderLayer" in names:
        if "o_proj" in names:
            return "qsa_attention.output_projection", "medium"
        if "_prepare_qkv_gate" in names or "qkv" in names.lower():
            return "qsa_attention.qkv_gate_projection", "medium"
        if "rotary" in names.lower() or "rmsnorm" in kernel:
            return "qsa_attention.qk_norm_rope", "medium"
        if (
            "RadixAttention" in names
            or "radix_attention.py" in names
            or "attention" in cpu
            or "attention" in kernel
            or "fmha" in kernel
        ):
            return "qsa_attention.attention_core", "medium"
        if "sigmoid" in cpu:
            return "qsa_attention.output_gate", "medium"

    if "Qwen2MoeMLP" in names or "_forward_shared_experts" in names:
        return "moe.shared_expert", "medium"
    if "Qwen2MoeSparseMoeBlock" in names or "FusedMoE" in names:
        if "moe::dev::routing" in kernel or "topk" in cpu:
            return "moe.topk", "high"
        if "moe::dev::finalize" in kernel:
            return "moe.combine", "high"
        if "moe::dev::activation" in kernel or "bmm_" in kernel or "grouped" in kernel:
            return "moe.routed_experts", "high"
        if "gate" in names and ("gemm" in kernel or "gemm" in cpu):
            return "moe.router", "medium"
        return "moe.routed_experts", "low"

    if "LogitsProcessor" in names or "lm_head" in names:
        return "top.lm_head", "medium"
    if "VocabParallelEmbedding" in names and "Qwen4ExpModel" in names:
        return "top.embedding", "medium"

    return None, "unmapped"


def classify_qwen38_flash_next_node_for_config(
    config_name: str,
    kernel_name: str,
    cpu_op_name: str | None,
    stack: list[FrameRef],
) -> tuple[str | None, str]:
    """Classify topology-specific bridge and dispatcher work before common ops."""

    names = "\n".join(frame.raw for frame in stack)
    lowered_names = names.lower()
    kernel = kernel_name.lower()
    dp_enabled = config_name in {
        "dp_attention",
        "dp_attention_ep4_deepep_deepgemm",
    }

    if config_name == "dp_attention_ep4_deepep_deepgemm":
        if "deepep.py" in lowered_names or "deepepdispatcher" in lowered_names:
            if "combine" in lowered_names or "combine" in kernel:
                return "moe.deepep_combine", "high"
            if "dispatch" in lowered_names or "dispatch" in kernel:
                return "moe.deepep_dispatch", "high"
        if "deepgemmrunnercore" in lowered_names or "moe_runner/deep_gemm.py" in lowered_names:
            return "moe.routed_experts", "high"

    if dp_enabled and (
        "dp_attention.py" in lowered_names
        or "dp_gather" in lowered_names
        or "dp_scatter" in lowered_names
    ):
        if "logitsprocessor" in lowered_names or "logits_processor.py" in lowered_names:
            if "scatter" in lowered_names:
                return "top.dp_logits_output_scatter", "high"
            return "top.dp_logits_input_gather", "high"
        if "qwen4expngramembedding" in lowered_names:
            if "scatter" in lowered_names:
                return "ple.dp_ngram_output_scatter", "high"
            return "ple.dp_ngram_input_gather", "high"
        layer_view = (
            "full_layer"
            if "qwen4expattentiondecoderlayer" in lowered_names
            else "linear_layer"
        )
        if "scatter" in lowered_names or "reduce_scatter" in lowered_names:
            return f"{layer_view}.dp_moe_output_scatter", "high"
        return f"{layer_view}.dp_moe_input_gather", "high"

    node, confidence = classify_qwen38_flash_next_node(kernel_name, cpu_op_name, stack)
    if config_name == "ep4_a2a_none":
        ep_collective_targets = {
            "linear_layer.tp_moe_output_collective": "linear_layer.ep_moe_output_collective",
            "full_layer.tp_moe_output_collective": "full_layer.ep_moe_output_collective",
            "moe.tp_output_collective": "moe.ep_output_collective",
        }
        if node in ep_collective_targets:
            return ep_collective_targets[node], confidence
    return node, confidence


QWEN38_FLASH_NEXT_TRACE_RULES = TraceMappingRules(
    model_id="qwen38_flash_next",
    signature_kernel=QWEN38_FLASH_NEXT_GDN_SIGNATURE_KERNEL,
    signature_count_per_forward=QWEN38_FLASH_NEXT_GDN_LAYERS_PER_FORWARD,
    stack=StackFrameRules(
        operator_patterns=(
            "layers/linear.py",
            "layers/quantization",
            "layers/hyperconnection.py",
            "layers/dp_attention.py",
            "layers/attention/qsa",
            "layers/moe/token_dispatcher/deepep.py",
            "layers/moe/moe_runner/deep_gemm.py",
            "radix_attention.py",
            "topk.py",
        ),
        semantic_patterns=(
            "models/qwen4_exp_mtp.py",
            "models/qwen4_exp.py",
            "models/qwen3_5.py",
            "models/qwen2_moe.py",
            "models/qwen3_vl.py",
            "layers/hyperconnection.py",
            "layers/dp_attention.py",
            "layers/attention/qsa",
            "layers/moe/token_dispatcher/deepep.py",
            "layers/moe/moe_runner/deep_gemm.py",
            "layers/logits_processor.py",
            "layers/vocab_parallel_embedding.py",
            "Qwen4ExpPLE",
            "Qwen4ExpNGramEmbedding",
            "Qwen4ExpAttentionDecoderLayer",
            "Qwen4ExpLinearDecoderLayer",
            "Qwen3_5GatedDeltaNet",
            "Qwen2MoeSparseMoeBlock",
            "Qwen2MoeMLP",
            "LogitsProcessor",
        ),
        model_context_patterns=(
            "Qwen4ExpForCausalLMMTP",
            "Qwen4ExpModel",
            "Qwen4ExpLinearDecoderLayer",
            "Qwen4ExpAttentionDecoderLayer",
            "Qwen4ExpPLELayer",
            "Qwen3_5GatedDeltaNet",
            "Qwen2MoeSparseMoeBlock",
            "Qwen4ExpForConditionalGeneration",
            "Qwen3VLForConditionalGeneration",
        ),
        phase_patterns=(
            "forward_extend",
            "forward_decode",
            "draft_forward",
            "_draft_extend_for_prefill",
            "_draft_extend_for_decode",
            "run_eagle_verify",
            "cuda_graph_runner",
            "replay",
        ),
    ),
    classify_node=classify_qwen38_flash_next_node,
)


def make_qwen38_flash_next_trace_rules(config_name: str) -> TraceMappingRules:
    if config_name not in QWEN38_FLASH_NEXT_CONFIGS:
        raise ValueError(f"unknown Qwen3.8-Flash-Next trace config: {config_name}")
    return TraceMappingRules(
        model_id=QWEN38_FLASH_NEXT_TRACE_RULES.model_id,
        signature_kernel=QWEN38_FLASH_NEXT_TRACE_RULES.signature_kernel,
        signature_count_per_forward=QWEN38_FLASH_NEXT_TRACE_RULES.signature_count_per_forward,
        stack=QWEN38_FLASH_NEXT_TRACE_RULES.stack,
        classify_node=lambda kernel, cpu, stack: classify_qwen38_flash_next_node_for_config(
            config_name, kernel, cpu, stack
        ),
    )
