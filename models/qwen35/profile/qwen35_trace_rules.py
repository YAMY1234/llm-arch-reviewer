#!/usr/bin/env python3
"""Qwen3.5 AgentX rules for the common eager Torch-trace mapper.

The rules deliberately target stable Model-IR view/node identifiers.  Target
and MTP-draft scopes are separated before any MoE fallback so a structurally
similar expert kernel cannot silently cross the target/draft boundary.
"""

from __future__ import annotations

from models.common.trace_mapping import FrameRef, StackFrameRules, TraceMappingRules


QWEN35_GDN_SIGNATURE_KERNEL = "fused_qkvzba_split"
QWEN35_GDN_LAYERS_PER_TARGET_FORWARD = 45


def _contains_any(value: str, needles: tuple[str, ...]) -> bool:
    return any(needle in value for needle in needles)


def classify_qwen35_node(
    kernel_name: str,
    cpu_op_name: str | None,
    stack: list[FrameRef],
) -> tuple[str | None, str]:
    raw_stack = "\n".join(frame.raw for frame in stack)
    names = raw_stack.lower()
    kernel = kernel_name.lower()
    cpu = (cpu_op_name or "").lower()

    draft = _contains_any(
        names,
        (
            "qwen3_5forcausallmmtp",
            "qwen3_5_mtp.py",
            "frozenkvmtpdraftworker",
            "draft_forward",
        ),
    )
    replay = _contains_any(
        names,
        ("draft_extend", "replayssm", "gdn_replayssm_spec_fold"),
    ) or _contains_any(kernel, ("cutedsl_gdn_mtp_ring", "replayssm"))

    # Speculative lifecycle and accepted-prefix state work takes precedence
    # over layer fallbacks because it may call the same GDN primitives.
    if replay:
        return "generation_loop.replay_gdn", "high"
    if "frozenkvmtpworker" in names and "forward_batch_generation" in names:
        if _contains_any(names, ("verify", "target_verify")):
            return "generation_loop.target_verify", "medium"
        if _contains_any(names, ("accept", "select")):
            return "generation_loop.accept_prefix", "medium"

    # Dispatcher collectives have implementation-unique kernel signatures.
    if "deep_ep::" in kernel or "deepep" in kernel:
        if "combine" in kernel:
            return "mtp_moe_block.draft_ep4_combine", "high"
        if "dispatch" in kernel:
            return "mtp_moe_block.draft_ep4_dispatch", "high"
    if _contains_any(kernel, ("moe_a2a_combine", "flashinfer_moe_combine")):
        return "moe_block.target_ep4_combine", "high"
    if _contains_any(kernel, ("moe_a2a_dispatch", "flashinfer_moe_dispatch")):
        if _contains_any(kernel, ("prepare", "preprocess", "pack")):
            return "moe_block.target_ep4_pack", "high"
        return "moe_block.target_ep4_dispatch", "high"

    if _contains_any(kernel, ("alltoall", "all_to_all")):
        if draft:
            node = (
                "mtp_moe_block.draft_ep4_combine"
                if "combine" in names
                else "mtp_moe_block.draft_ep4_dispatch"
            )
        else:
            node = (
                "moe_block.target_ep4_combine"
                if "combine" in names
                else "moe_block.target_ep4_dispatch"
            )
        return node, "medium"

    # GDN primitives are target-only in the canonical 60-layer decoder.  Any
    # GDN work reached from draft_extend was classified as replay above.
    if QWEN35_GDN_SIGNATURE_KERNEL in kernel:
        return "gdn_moe_block.qkvz_projection", "high"
    if "causal_conv1d" in kernel:
        return "gdn_moe_block.causal_conv", "high"
    if _contains_any(
        kernel,
        (
            "fused_recurrent_gated_delta_rule",
            "gdn_decode",
            "gdn_wide_vec_kernel",
            "chunk_gated_delta_rule",
            "recompute_w_u",
            "chunk_fwd_kernel_o",
            "l2norm_fwd",
        ),
    ):
        return "gdn_moe_block.gated_delta_recurrence", "high"
    if "fused_gdn_gating" in kernel:
        return "gdn_moe_block.output_gate_norm", "high"

    if "qwen3_5gateddeltanet" in names:
        if "out_proj" in names:
            return "gdn_moe_block.output_projection", "medium"
        if "in_proj_ba" in names:
            return "gdn_moe_block.ba_projection", "medium"
        if "in_proj_qkvz" in names or "_forward_input_proj" in names:
            return "gdn_moe_block.qkvz_projection", "medium"
        if "norm" in names or "rmsnorm" in kernel or "layer_norm" in kernel:
            return "gdn_moe_block.output_gate_norm", "medium"
        return "gdn_moe_block.gated_delta_recurrence", "low"

    attention_node = (
        "mtp_full_attention_moe_block.causal_gqa"
        if draft
        else "full_attention_moe_block.causal_gqa"
    )
    if "qwen3_5attentiondecoderlayer" in names:
        if _contains_any(kernel, ("fmha", "mha", "attention")) or "radixattention" in names:
            return attention_node, "high"
        if _contains_any(names, ("o_proj", "self_attention")):
            return (
                "mtp_full_attention_moe_block.output_projection"
                if draft
                else "full_attention_moe_block.output_projection"
            ), "medium"
        if _contains_any(names, ("qkv", "forward_prepare")):
            return (
                "mtp_full_attention_moe_block.qkv_projection"
                if draft
                else "full_attention_moe_block.qkv_projection"
            ), "medium"

    moe_prefix = "mtp_moe_block" if draft else "moe_block"
    if "deepepdispatcher" in names or "token_dispatcher/deepep.py" in names:
        if "combine" in names:
            return f"{moe_prefix}.draft_ep4_combine", "high"
        return f"{moe_prefix}.draft_ep4_dispatch", "high"
    if "flashinferdispatcher" in names or "token_dispatcher/flashinfer.py" in names:
        if "combine" in names:
            return f"{moe_prefix}.target_ep4_combine", "high"
        return f"{moe_prefix}.target_ep4_dispatch", "high"
    if "deepgemmrunnercore" in names or "deep_gemm::" in kernel:
        return "mtp_moe_block.routed_experts", "high"
    if "qwen2moemlp" in names or "_forward_shared_experts" in names:
        return f"{moe_prefix}.shared_expert", "medium"
    if "qwen2moesparsemoeblock" in names or "fusedmoe" in names:
        if _contains_any(kernel, ("routing", "topk")) or "topk" in cpu:
            return f"{moe_prefix}.router", "high"
        if "finalize" in kernel:
            return (
                "mtp_moe_block.draft_ep4_restore"
                if draft
                else "moe_block.target_ep4_restore"
            ), "high"
        if _contains_any(kernel, ("deep_gemm", "cutedsl", "moe", "bmm_", "grouped")):
            return f"{moe_prefix}.routed_experts", "high"
        if "gate" in names and _contains_any(kernel + cpu, ("gemm", "matmul")):
            return f"{moe_prefix}.router", "medium"
        return f"{moe_prefix}.routed_experts", "low"

    if draft:
        if "qwen3_5forcausallmmtp" in names:
            if _contains_any(names, ("fc", "projection")):
                return "mtp_draft_head.fc_projection", "medium"
            return "mtp_draft_head.draft_decoder_layer", "low"
        if "frozenkvmtpdraftworker" in names:
            return "generation_loop.draft_propose", "low"

    if "logitsprocessor" in names or "lm_head" in names:
        return "top.lm_head", "medium"
    if "vocabparallelembedding" in names or "get_input_embeddings" in names:
        return "top.embedding", "medium"
    if "qwen3_5forcausallm" in names:
        if "norm" in names or "rmsnorm" in kernel:
            return "top.final_norm", "medium"
        return "top.decoder_stack", "low"

    return None, "unmapped"


QWEN35_TRACE_RULES = TraceMappingRules(
    model_id="qwen35_397b_a17b",
    signature_kernel=QWEN35_GDN_SIGNATURE_KERNEL,
    signature_count_per_forward=QWEN35_GDN_LAYERS_PER_TARGET_FORWARD,
    stack=StackFrameRules(
        operator_patterns=(
            "layers/linear.py",
            "layers/quantization",
            "layers/attention",
            "layers/moe",
            "radix_attention.py",
            "topk.py",
            "speculative/frozen_kv_mtp_worker_v2.py",
        ),
        semantic_patterns=(
            "models/qwen3_5.py",
            "models/qwen3_5_text.py",
            "models/qwen3_5_mtp.py",
            "models/qwen2_moe.py",
            "layers/moe/token_dispatcher/flashinfer.py",
            "layers/moe/token_dispatcher/deepep.py",
            "layers/moe/moe_runner/deep_gemm.py",
            "layers/logits_processor.py",
            "layers/vocab_parallel_embedding.py",
            "speculative/frozen_kv_mtp_worker_v2.py",
            "Qwen3_5GatedDeltaNet",
            "Qwen3_5AttentionDecoderLayer",
            "Qwen3_5ForCausalLMMTP",
            "Qwen2MoeSparseMoeBlock",
            "Qwen2MoeMLP",
            "FrozenKVMTPDraftWorker",
            "FrozenKVMTPWorkerV2",
            "LogitsProcessor",
        ),
        model_context_patterns=(
            "Qwen3_5ForCausalLM",
            "Qwen3_5ForCausalLMMTP",
            "Qwen3_5LinearDecoderLayer",
            "Qwen3_5AttentionDecoderLayer",
            "Qwen3_5GatedDeltaNet",
            "Qwen2MoeSparseMoeBlock",
            "FrozenKVMTPDraftWorker",
            "FrozenKVMTPWorkerV2",
        ),
        phase_patterns=(
            "forward_extend",
            "forward_decode",
            "forward_batch_generation",
            "draft_forward",
            "draft_extend",
            "cuda_graph_runner",
            "replay",
        ),
    ),
    classify_node=classify_qwen35_node,
)
