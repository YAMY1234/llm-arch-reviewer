"""Production-profile ownership contract for Kimi K3.

The Model IR remains implementation independent.  This module records only
profile-local facts established by the locked implementation source and eager
stack evidence: structural/state nodes and semantic children whose work is
physically owned by a different measured production event set.
"""

from __future__ import annotations

from typing import Any, Iterable


VISION_PREFIXES = ("vision_frontend.", "vision_block.")
VISION_TOP_NODES = {
    "top.vision_inputs",
    "top.vision_frontend",
    "top.multimodal_injection",
}

STRUCTURAL_NODES = {
    "top.token_ids",
    "top.decoder_stack",
    "decoder_stack.stack_in",
    "decoder_stack.schedule",
    "decoder_stack.attention_schedule",
    "decoder_stack.kda",
    "decoder_stack.gated_mla",
    "decoder_stack.feed_forward_schedule",
    "decoder_stack.prefix_after_ffn",
    "decoder_stack.dense_mlp",
    "decoder_stack.stable_latent_moe",
    "decoder_stack.stack_out",
    "attn_res.bank_in",
    "attn_res.prefix_in",
    "attn_res.attn_res_out",
    "kda.kda_in",
    "kda.kda_out",
    "gated_mla.mla_in",
    "gated_mla.mla_out",
    "dense_mlp.dense_in",
    "dense_mlp.dense_out",
    "stable_latent_moe.moe_in",
    "stable_latent_moe.moe_out",
}

STATE_NODES = {
    "decoder_stack.block_write": "AttnRes source-bank state update boundary",
    "gated_mla.cache_update": "persistent latent-KV cache update boundary",
}


def _group(
    owner: str,
    nodes: Iterable[str],
    *,
    timing_semantics: str,
    mapping_method: str,
    provenance: str,
) -> dict[str, Any]:
    ordered = list(dict.fromkeys([owner, *nodes]))
    return {
        "owner": owner,
        "ir_nodes": ordered,
        "timing_semantics": timing_semantics,
        "mapping_method": mapping_method,
        "confidence": "exact",
        "evidence_scope": {
            "resolution": (
                "exact_occurrence"
                if timing_semantics == "shared_interval"
                else "profile_aggregate"
            )
        },
        "provenance": provenance,
    }


def sglang_fusion_groups(
    *, phase: str, batch_size: int, measured_nodes: set[str]
) -> dict[str, dict[str, Any]]:
    """Return source- and eager-proven SGLang fusion groups for one point."""

    if phase not in {"prefill", "decode"}:
        raise ValueError(f"unsupported phase: {phase}")
    groups = {
        "sglang_attn_res_profile_aggregate": _group(
            "attn_res.weighted_merge",
            [
                "decoder_stack.attn_res_read",
                "decoder_stack.ffn_attn_res_read",
                "decoder_stack.input_norm",
                "decoder_stack.post_attention_norm",
                "decoder_stack.prefix_after_attention",
                "attn_res.source_concat",
                "attn_res.rms_normalize",
                "attn_res.score_projection",
                "attn_res.source_softmax",
            ],
            timing_semantics="shared_event_set",
            mapping_method="attn_res_occurrence_bounded_eager_to_production",
            provenance=(
                "locked AttnResidual.forward passes the source bank, score RMSNorm, "
                "score projection, softmax mixture, and next-sublayer RMSNorm into "
                "the AttnRes owner event set"
            ),
        ),
        "sglang_final_attn_res_norm": _group(
            "top.output_attn_res",
            ["top.final_norm"],
            timing_semantics="shared_interval",
            mapping_method="exact_final_attn_res_signature_and_locked_callsite",
            provenance=(
                "KimiK3LinearModel.forward passes self.norm to the final "
                "AttnResidual.forward call; the final RMSNorm has no second interval"
            ),
        ),
        "sglang_kda_qkvg_projection_bundle": _group(
            "kda.qkv_projection",
            ["kda.output_gate"],
            timing_semantics="shared_event_set",
            mapping_method="locked_forward_qkvbfg_projection_bundle",
            provenance="the locked q/k/v/beta/decay/output-gate projection bundle materializes g with q/k/v",
        ),
        "sglang_mla_a_projection_bundle": _group(
            "gated_mla.q_down",
            ["gated_mla.kv_down"],
            timing_semantics="shared_event_set",
            mapping_method="exact_fused_a_gemm_signature",
            provenance="the fused A GEMM jointly materializes q_down and kv_down",
        ),
        "sglang_moe_front_bundle": _group(
            "stable_latent_moe.router_logits",
            [
                "stable_latent_moe.router_sigmoid",
                "stable_latent_moe.routed_down",
                "stable_latent_moe.shared_gate_up",
            ],
            timing_semantics="shared_event_set",
            mapping_method="locked_fused_front_gemm_and_eager_stack",
            provenance=(
                "one fused front GEMM emits the shared gate/up branch, router logits, "
                "and routed latent down projection"
            ),
        ),
        "sglang_moe_selection_bundle": _group(
            "stable_latent_moe.corrected_selection",
            ["stable_latent_moe.weight_normalize"],
            timing_semantics="shared_event_set",
            mapping_method="exact_route_radix_sequence",
            provenance="corrected top-k selection and selected-weight normalization share the routing event set",
        ),
        "sglang_moe_expert_situ_bundle": _group(
            "stable_latent_moe.expert_gate_up",
            ["stable_latent_moe.expert_situ"],
            timing_semantics="shared_event_set",
            mapping_method="flashinfer_mxfp4_situ_expert_signature",
            provenance="SiTU is executed inside the routed expert gate/up runner",
        ),
    }

    if phase == "decode":
        groups["sglang_kda_decode_fused_update"] = _group(
            "kda.recurrent_update",
            [
                "kda.q_short_conv",
                "kda.k_short_conv",
                "kda.v_short_conv",
                "kda.qk_l2_norm",
                "kda.lower_bounded_decay",
                "kda.query_readout",
                "kda.gated_rmsnorm",
            ],
            timing_semantics="shared_event_set",
            mapping_method="exact_kda_decode_fusion_many_heads_signature",
            provenance="the locked fused decode kernel performs convolution, normalization, recurrent update, readout, and gated RMSNorm",
        )
        groups["sglang_mla_decode_weight_absorption"] = _group(
            "gated_mla.attention",
            ["gated_mla.kv_up", "gated_mla.key_compose"],
            timing_semantics="shared_event_set",
            mapping_method="locked_decode_weight_absorption_and_eager_bmm_sequence",
            provenance="decode absorbs kv_up into the pre-attention and post-attention contractions",
        )
    else:
        groups["sglang_kda_prefill_safe_gate_update"] = _group(
            "kda.recurrent_update",
            ["kda.lower_bounded_decay"],
            timing_semantics="shared_event_set",
            mapping_method="locked_chunk_kda_safe_gate_argument_and_eager_sequence",
            provenance=(
                "the prefill backend passes the checkpoint's lower bound into "
                "chunk_kda, which applies the bounded decay inside the recurrence "
                "event set without a standalone interval"
            ),
        )

    # Shape-specialized projection kernels can collapse f and beta back into
    # the q/k/v/g bundle.  Add only children that have no independently
    # attributed event in this exact point.
    projection_children: list[str] = []
    for node in ("kda.decay_projection", "kda.beta_projection"):
        if node not in measured_nodes:
            projection_children.append(node)
    if projection_children:
        groups["sglang_kda_qkvg_projection_bundle"]["ir_nodes"].extend(
            projection_children
        )

    routed_children = ["stable_latent_moe.latent_norm"]
    if "stable_latent_moe.weighted_reduce" not in measured_nodes:
        routed_children.append("stable_latent_moe.weighted_reduce")
    groups["sglang_moe_routed_collective_tail"] = _group(
        "stable_latent_moe.tp_routed_latent_collective",
        routed_children,
        timing_semantics="shared_event_set",
        mapping_method="exact_finalize_all_reduce_norm_sequence",
        provenance="the routed tail folds selected reduction when shape-selected and always folds latent RMSNorm into the TP collective owner",
    )
    return groups


def vllm_fusion_groups(
    *, phase: str, batch_size: int, measured_nodes: set[str]
) -> dict[str, dict[str, Any]]:
    """Return source- and eager-proven vLLM fusion groups for one point."""

    if phase not in {"prefill", "decode"}:
        raise ValueError(f"unsupported phase: {phase}")
    if phase == "prefill" and batch_size != 1:
        raise ValueError("vLLM prefill is accepted only at batch 1")
    if phase == "decode" and batch_size not in {1, 16, 64}:
        raise ValueError("vLLM decode accepts 1/16/64; 256 is unsupported")

    groups: dict[str, dict[str, Any]] = {}

    def add(
        group_id: str,
        owner: str,
        candidates: Iterable[str],
        *,
        timing_semantics: str,
        mapping_method: str,
        provenance: str,
    ) -> None:
        children = [node for node in candidates if node not in measured_nodes]
        if not children:
            return
        groups[group_id] = _group(
            owner,
            children,
            timing_semantics=timing_semantics,
            mapping_method=mapping_method,
            provenance=provenance,
        )

    add(
        "vllm_attn_res_profile_aggregate",
        "attn_res.weighted_merge",
        [
            "decoder_stack.attn_res_read",
            "decoder_stack.ffn_attn_res_read",
            "decoder_stack.input_norm",
            "decoder_stack.post_attention_norm",
            "decoder_stack.prefix_after_attention",
            "attn_res.source_concat",
            "attn_res.rms_normalize",
            "attn_res.score_projection",
            "attn_res.source_softmax",
        ],
        timing_semantics="shared_event_set",
        mapping_method="attn_res_occurrence_bounded_eager_to_production",
        provenance=(
            "the locked native attn_res call jointly performs source-bank assembly, "
            "score normalization/projection, softmax mixture, and the next-sublayer norm"
        ),
    )
    add(
        "vllm_kda_qkvgfab_projection_bundle",
        "kda.qkv_projection",
        ["kda.beta_projection", "kda.output_gate"],
        timing_semantics="shared_event_set",
        mapping_method="locked_in_proj_qkvgfab_output_partition",
        provenance=(
            "in_proj_qkvgfab jointly materializes mixed Q/K/V, the output gate, "
            "the decay precursor, and beta; f_b_proj remains independently measured"
        ),
    )
    add(
        "vllm_kda_output_projection_collective",
        "kda.output_projection",
        ["kda.tp_kda_output_collective"],
        timing_semantics="shared_interval",
        mapping_method="exact_cutedsl_gemm_rs_ar_owner",
        provenance=(
            "the locked CuTeDSL GEMM-RS/AR kernel jointly performs the KDA "
            "row-parallel output projection and TP reduction"
        ),
    )
    if phase == "decode":
        add(
            "vllm_kda_decode_fused_update",
            "kda.recurrent_update",
            [
                "kda.q_short_conv",
                "kda.k_short_conv",
                "kda.v_short_conv",
                "kda.qk_l2_norm",
                "kda.lower_bounded_decay",
                "kda.query_readout",
                "kda.gated_rmsnorm",
            ],
            timing_semantics="shared_event_set",
            mapping_method="exact_native_fused_kda_decode_signature",
            provenance=(
                "the locked fused_kda_decode call performs convolution, Q/K "
                "normalization, bounded recurrence, readout, output gating, and RMSNorm"
            ),
        )
    else:
        add(
            "vllm_kda_prefill_bounded_recurrence",
            "kda.recurrent_update",
            ["kda.lower_bounded_decay"],
            timing_semantics="shared_event_set",
            mapping_method="locked_chunk_kda_lower_bound_argument",
            provenance=(
                "the Triton prefill recurrence applies the checkpoint lower bound "
                "inside its recurrence kernels without a standalone interval"
            ),
        )

    add(
        "vllm_mla_a_gate_projection_bundle",
        "gated_mla.q_down",
        ["gated_mla.kv_down", "gated_mla.output_gate"],
        timing_semantics="shared_event_set",
        mapping_method="locked_fused_qkv_a_g_projection",
        provenance=(
            "fused_qkv_a_g_proj jointly materializes q_down, kv_down, and the "
            "MLA output gate unless the exact small-token path exposes a separate gate GEMM"
        ),
    )
    add(
        "vllm_mla_q_kv_norm_bundle",
        "gated_mla.q_norm",
        ["gated_mla.kv_norm"],
        timing_semantics="shared_interval",
        mapping_method="exact_fused_q_kv_rmsnorm_signature",
        provenance="fused_q_kv_rmsnorm normalizes both low-rank branches in one owner",
    )
    add(
        "vllm_mla_key_cache_bundle",
        "gated_mla.cache_update",
        ["gated_mla.key_compose"],
        timing_semantics="shared_interval",
        mapping_method="exact_fused_mla_concat_cache_signature",
        provenance=(
            "the locked phase-specific MLA epilogue composes the key and writes "
            "the persistent latent cache in one kernel"
        ),
    )
    add(
        "vllm_mla_output_projection_collective",
        "gated_mla.output_projection",
        ["gated_mla.tp_mla_output_collective"],
        timing_semantics="shared_interval",
        mapping_method="exact_cutedsl_gemm_rs_ar_owner",
        provenance=(
            "the locked CuTeDSL GEMM-RS/AR kernel jointly performs the MLA "
            "row-parallel output projection and TP reduction"
        ),
    )
    if phase == "decode":
        add(
            "vllm_mla_decode_weight_absorption",
            "gated_mla.attention",
            ["gated_mla.kv_up"],
            timing_semantics="shared_event_set",
            mapping_method="locked_w_uk_w_uv_decode_absorption",
            provenance=(
                "decode absorbs kv_b_proj into the query-to-latent and "
                "latent-to-value contractions attributed to the attention owner"
            ),
        )

    add(
        "vllm_moe_router_selection_bundle",
        "stable_latent_moe.corrected_selection",
        [
            "stable_latent_moe.router_sigmoid",
            "stable_latent_moe.weight_normalize",
        ],
        timing_semantics="shared_event_set",
        mapping_method="locked_fused_grouped_topk_router",
        provenance=(
            "the grouped top-k router applies sigmoid scoring, correction, "
            "selection, and selected-weight normalization as one routed owner"
        ),
    )
    add(
        "vllm_moe_expert_front_bundle",
        "stable_latent_moe.expert_gate_up",
        ["stable_latent_moe.dispatch", "stable_latent_moe.expert_situ"],
        timing_semantics="shared_event_set",
        mapping_method="exact_flashinfer_mxfp4_dispatch_and_situ_owner",
        provenance=(
            "the exact FlashInfer MXFP4 runner owns packed expert dispatch and "
            "the SiTU gate/up epilogue when no independent event is emitted"
        ),
    )
    add(
        "vllm_moe_shared_front_bundle",
        "stable_latent_moe.shared_gate_up",
        ["stable_latent_moe.shared_situ"],
        timing_semantics="shared_event_set",
        mapping_method="locked_shared_expert_gate_up_situ_sequence",
        provenance=(
            "the shared expert merged gate/up path owns SiTU when the exact "
            "shape-specialized implementation emits no standalone activation event"
        ),
    )

    tail_children = ["stable_latent_moe.latent_norm"]
    if phase == "decode":
        tail_children.extend(
            [
                "stable_latent_moe.weighted_reduce",
                "stable_latent_moe.routed_up",
                "stable_latent_moe.tp_shared_expert_collective",
                "stable_latent_moe.combine",
            ]
        )
    add(
        "vllm_moe_latent_tail_bundle",
        "stable_latent_moe.tp_routed_latent_collective",
        tail_children,
        timing_semantics="shared_event_set",
        mapping_method=(
            "exact_cutedsl_latent_moe_tail"
            if phase == "decode"
            else "exact_allreduce_norm_latent_tail"
        ),
        provenance=(
            "decode batches up to 64 use the locked CuTeDSL whole-tail operator; "
            "prefill keeps independent up-projection/final-reduction intervals "
            "while fusing routed all-reduce with latent RMSNorm"
        ),
    )
    if phase == "prefill":
        add(
            "vllm_moe_prefill_up_projection_beta_add",
            "stable_latent_moe.routed_up",
            ["stable_latent_moe.combine"],
            timing_semantics="shared_interval",
            mapping_method="locked_column_parallel_up_projection_beta_add",
            provenance=(
                "the prefill-sized tier accumulates the routed up-projection into "
                "the shared partial through the GEMM beta-add epilogue"
            ),
        )
    add(
        "vllm_dense_output_projection_collective",
        "dense_mlp.down",
        ["dense_mlp.tp_dense_output_collective"],
        timing_semantics="shared_interval",
        mapping_method="exact_cutedsl_gemm_rs_ar_owner",
        provenance=(
            "the locked CuTeDSL GEMM-RS/AR kernel jointly performs the dense "
            "down projection and TP reduction"
        ),
    )
    add(
        "vllm_logits_materialization_output",
        "top.tp_logits_materialization",
        ["top.logits"],
        timing_semantics="shared_interval",
        mapping_method="exact_tp_logits_allgather_output",
        provenance=(
            "the locked TP all-gather materializes the complete logits tensor "
            "that crosses the Model IR output boundary"
        ),
    )
    return groups


def build_node_states(
    *,
    required_nodes: Iterable[str],
    measured_nodes: set[str],
    fusion_groups: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Close every non-measured node without a generic fallback state."""

    fused: dict[str, tuple[str, str]] = {}
    for group_id, group in fusion_groups.items():
        owner = str(group["owner"])
        if owner not in measured_nodes:
            raise ValueError(
                f"fusion group {group_id} has no measured owner interval: {owner}"
            )
        for node in group["ir_nodes"]:
            if node == owner:
                continue
            if node in measured_nodes:
                raise ValueError(
                    f"fusion child {node} in {group_id} has an independent measured interval"
                )
            if node in fused and fused[node][0] != owner:
                raise ValueError(f"conflicting fusion owners for {node}")
            fused[node] = (owner, group_id)

    states: dict[str, dict[str, Any]] = {}
    unresolved: list[str] = []
    for node in required_nodes:
        if node in measured_nodes:
            continue
        if node in VISION_TOP_NODES or node.startswith(VISION_PREFIXES):
            states[node] = {
                "status": "not_selected",
                "label": "text-only stage-1 profile; the vision branch is outside this invocation",
            }
        elif node in STRUCTURAL_NODES:
            states[node] = {
                "status": "structural",
                "label": "semantic tensor/control boundary without a standalone GPU interval",
            }
        elif node in STATE_NODES:
            states[node] = {"status": "state", "label": STATE_NODES[node]}
        elif node in fused:
            owner, group_id = fused[node]
            states[node] = {
                "status": "fused",
                "label": f"fused into {owner}",
                "included_in": owner,
                "fusion_group_id": group_id,
            }
        else:
            unresolved.append(node)
    if unresolved:
        raise ValueError(
            "production profile has unexplained non-measured nodes: "
            + ", ".join(sorted(unresolved))
        )
    return states
