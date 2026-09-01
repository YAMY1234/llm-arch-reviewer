"""Fail-closed vLLM eager-to-production attribution for GLM-5.3-Flash.

The vLLM eager evidence contains two forwards while one production-device
capture contains one 45-layer forward.  Each forward has exactly 90 mHC-pre
anchors (attention and feed-forward for every layer).  We use the first eager
forward as semantic evidence and transfer attribution only inside matching
anchor-bounded kernel schedules.  This prevents generic NVJet/DeepGEMM names
from leaking across KDA, DSA, dense, and MoE scopes.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from models.glm52.build.build_glm52_production_profile import (
    kernel_base,
    kernel_exact_identity,
    schedule_family,
    unique_source_index,
)
from models.glm53_flash.build.glm53_sglang_production_attribution import (
    ANCHOR_TOKEN,
    SUBLAYER_SEGMENT_COUNT,
    segment_kind,
)


FUSED_TP_NODES = {
    "linear_attention.tp_kda_output_collective",
    "dsa_attention.tp_dsa_output_collective",
    "dense_mlp.tp_dense_mlp_output_collective",
    "moe.tp_moe_output_collective",
}


def _is_anchor(row: dict[str, Any]) -> bool:
    return ANCHOR_TOKEN in str(row.get("kernel_name") or "").lower()


def _anchor_indices(rows: list[dict[str, Any]]) -> list[int]:
    return [index for index, row in enumerate(rows) if _is_anchor(row)]


def _first_eager_forward(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    anchors = _anchor_indices(rows)
    if len(anchors) < SUBLAYER_SEGMENT_COUNT:
        raise ValueError(
            f"vLLM eager evidence requires at least {SUBLAYER_SEGMENT_COUNT} "
            f"mHC anchors, got {len(anchors)}"
        )
    end = anchors[SUBLAYER_SEGMENT_COUNT] if len(anchors) > SUBLAYER_SEGMENT_COUNT else len(rows)
    forward = [dict(row) for row in rows[:end]]
    if len(_anchor_indices(forward)) != SUBLAYER_SEGMENT_COUNT:
        raise ValueError("could not isolate one exact 90-anchor vLLM eager forward")
    return forward


def _segments(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]]]:
    anchors = _anchor_indices(rows)
    if len(anchors) != SUBLAYER_SEGMENT_COUNT:
        raise ValueError(
            f"expected {SUBLAYER_SEGMENT_COUNT} production mHC anchors, got {len(anchors)}"
        )
    prefix = rows[: anchors[0]]
    segments = [
        rows[start:end]
        for start, end in zip(anchors, [*anchors[1:], len(rows)])
    ]
    return prefix, segments


def _annotate_segment_scope(
    segment: list[dict[str, Any]], segment_id: int
) -> None:
    """Persist the execution occurrence that justified a bounded transfer.

    The 90 mHC anchors are not merely a matching aid.  They are the stable
    execution scope for one attention or feed-forward invocation in one model
    layer.  Keeping that scope on every source/production event prevents a
    later aggregate from turning a per-occurrence fusion into an unexplained
    model-wide ``fused`` label.
    """

    layer_id = segment_id // 2
    substage = "attention" if segment_id % 2 == 0 else "feed_forward"
    family = segment_kind(segment_id)
    occurrence_id = f"layer_{layer_id:02d}.{substage}"
    for row in segment:
        row.update(
            {
                "layer_id": layer_id,
                "layer_kind": family,
                "substage": substage,
                "segment_id": segment_id,
                "occurrence_id": occurrence_id,
            }
        )


def _enrich_source_segment(segment: list[dict[str, Any]], kind: str) -> None:
    """Resolve stack-ambiguous leaves using reviewed source-order contracts."""

    for row in segment:
        lowered = str(row.get("kernel_name") or "").lower()
        cpu = str(row.get("cpu_op_name") or "").lower()
        stack = "\n".join(
            str((row.get(key) or {}).get("raw") or "")
            for key in ("operator_frame", "semantic_frame", "model_context_frame")
        ).lower()
        if kind == "kda":
            projection_rows = [
                item
                for item in segment
                if item.get("selected_node") == "linear_attention.forget_projection"
            ]
            if len(projection_rows) == 2:
                projection_rows[0]["selected_node"] = "linear_attention.forget_projection"
                projection_rows[1]["selected_node"] = "linear_attention.beta_projection"
            if "chunk_gla_fwd_kernel_o" in lowered:
                row["selected_node"] = "linear_attention.query_readout"
        elif kind == "dsa":
            # vLLM source contract: wq_b, fused wk+weights, explicit fp32
            # weight projection, and gate-score projection.  The following
            # fused helper names make the order independently checkable.
            index_mm = [
                item
                for item in segment
                if (item.get("model_context_frame") or {}).get("raw") == "nn.Module: Indexer_0"
                and str(item.get("cpu_op_name") or "").lower() == "aten::mm"
            ]
            if len(index_mm) >= 4:
                index_mm[0]["selected_node"] = "dsa_attention.index_q_projection"
                index_mm[1]["selected_node"] = "dsa_attention.index_k_projection"
                for item in index_mm[2:]:
                    item["selected_node"] = "dsa_attention.index_weight_projection"

            # These helpers are not generic guesses: their eager Python stacks
            # prove that they execute inside one Indexer/K-pool/MLA semantic
            # scope, and their position is bounded by the neighboring authored
            # Model-IR leaves.  Recording the source-side leaf lets an exactly
            # matching CUDA-Graph occurrence inherit the same attribution.
            if "nn.module: indexer_" in stack:
                if (
                    "_fused_indexer_k_norm" in stack
                    or "_fwht_quant_kernel" in lowered
                    or cpu == "aten::copy_"
                ):
                    row["selected_node"] = "dsa_attention.index_k_projection"
                elif "_fused_indexer_weight_scale" in stack:
                    row["selected_node"] = "dsa_attention.index_weight_projection"
            elif "_fwht_quant_kernel" in lowered or "native_layer_norm" in lowered:
                row["selected_node"] = "dsa_attention.index_k_projection"
            elif "_kpool_compress_insert" in stack:
                row["selected_node"] = "dsa_attention.key_pool_compression"
            elif "sparse_attn_indexer_kpool" in stack:
                if "cp_gather_indexer_k_quant_cache" in lowered:
                    row["selected_node"] = "dsa_attention.index_k_cache"
                elif any(
                    token in lowered
                    for token in (
                        "fillfunctor",
                        "direct_copy_kernel",
                        "cudafunctoronself_add",
                    )
                ):
                    row["selected_node"] = "dsa_attention.token_expansion"
            elif "nn.module: glm5nextmlaattention_" in stack:
                if any(
                    token in lowered
                    for token in (
                        "catarraybatchedcopy",
                        "scaled_fp8_quant_kernel",
                    )
                ):
                    row["selected_node"] = "dsa_attention.latent_kv_reconstruction"
                elif any(
                    token in lowered
                    for token in (
                        "fillfunctor",
                        "aunaryfunctor",
                        "direct_copy_kernel",
                        "maskedfill",
                        "masked_fill_kernel",
                        "clamp",
                    )
                ):
                    row["selected_node"] = "dsa_attention.selected_indices"
        elif kind == "moe":
            if lowered.startswith("bmm_e4m3"):
                row["selected_node"] = "moe.routed_gate_up"
            elif lowered.startswith("bmm_bfloat16"):
                row["selected_node"] = "moe.routed_down"

            shared_gemms = [
                item
                for item in segment
                if "deep_gemm::sm100_fp8_fp4_gemm" in str(item.get("kernel_name") or "").lower()
            ]
            if len(shared_gemms) == 2:
                shared_gemms[0]["selected_node"] = "moe.shared_gate_up"
                shared_gemms[1]["selected_node"] = "moe.shared_down"

            if "nn.module: glm5nextmlp_" in stack and "per_token_group_quant" in lowered:
                # The two quantizers bracket the shared-expert activation.
                activation_index = next(
                    (
                        index
                        for index, item in enumerate(segment)
                        if "act_and_mul_kernel" in str(item.get("kernel_name") or "").lower()
                    ),
                    None,
                )
                row_index = segment.index(row)
                row["selected_node"] = (
                    "moe.shared_gate_up"
                    if activation_index is None or row_index < activation_index
                    else "moe.shared_down"
                )
            elif "nn.module: glm5nextmoe_" in stack and (
                "per_token_group_quant" in lowered or cpu == "aten::copy_"
            ):
                row["selected_node"] = "moe.dispatch"

    if kind == "kda":
        # KDA has several PyTorch shape/copy/scan helpers whose individual
        # names are generic but whose position between unique KDA kernels is
        # stable.  Attribute only inside those semantic landmark intervals.
        def landmark(token: str) -> int | None:
            return next(
                (
                    index
                    for index, item in enumerate(segment)
                    if token in str(item.get("kernel_name") or "").lower()
                ),
                None,
            )

        state_read = landmark("_gather_initial_states_kernel")
        first_norm = landmark("l2norm_fwd_kernel")
        forget_decay = landmark("kda_gate_cumsum_fwd_kernel")
        first_update = landmark("chunk_kda_scaled_dot_kkt_fwd_kernel")
        readout = landmark("chunk_gla_fwd_kernel_o")
        if None not in (state_read, first_norm, forget_decay, first_update, readout):
            assert state_read is not None
            assert first_norm is not None
            assert forget_decay is not None
            assert first_update is not None
            assert readout is not None
            interval_nodes = (
                (state_read + 1, first_norm, "linear_attention.beta_projection"),
                (first_norm, forget_decay, "linear_attention.qk_l2_norm"),
                (first_norm + 1, first_update, "linear_attention.forget_decay"),
                (first_update, readout, "linear_attention.recurrent_update"),
            )
            for start, end, node in interval_nodes:
                for item in segment[start:end]:
                    if not item.get("selected_node"):
                        item["selected_node"] = node

    # The last decoder segment is followed by HC contraction, final norm,
    # vocabulary projection, and logits all-gather on the main stream.  A
    # neighboring embedding frame can remain active for the vocabulary GEMM;
    # its position between final norm and logits gather is the stable contract.
    final_norm_index = next(
        (
            index
            for index, row in enumerate(segment)
            if row.get("selected_node") == "top.final_norm"
        ),
        None,
    )
    logits_gather_index = next(
        (
            index
            for index, row in enumerate(segment)
            if row.get("selected_node") == "top.tp_logits_all_gather"
        ),
        None,
    )
    if final_norm_index is not None and logits_gather_index is not None:
        vocabulary_rows = [
            row
            for row in segment[final_norm_index + 1 : logits_gather_index]
            if str(row.get("kernel_name") or "").lower().startswith("nvjet_")
        ]
        if len(vocabulary_rows) == 1:
            vocabulary_rows[0]["selected_node"] = "top.lm_head"


def _assign(
    production: dict[str, Any],
    source: dict[str, Any],
    *,
    method: str,
) -> bool:
    if production.get("node") or not source.get("selected_node"):
        return False
    # vLLM CUDA Graph folds each sublayer TP output collective into the mHC
    # residual boundary.  Keep the production-side shared-interval owner.
    if source["selected_node"] in FUSED_TP_NODES:
        return False
    production.update(
        {
            "node": source["selected_node"],
            "kernel_label": source.get("cpu_op_name") or source["selected_node"],
            "attribution_method": method,
            "confidence": source.get("confidence") or "high",
            "eager_event_id": source.get("event_id"),
        }
    )
    return True


def _transfer_matching_scope(
    source: list[dict[str, Any]],
    production: list[dict[str, Any]],
    *,
    method_prefix: str,
) -> tuple[int, str]:
    source_bases = [kernel_base(str(row.get("kernel_name") or "")) for row in source]
    production_bases = [kernel_base(str(row.get("kernel_name") or "")) for row in production]
    assigned = 0
    if source_bases == production_bases:
        for source_row, production_row in zip(source, production):
            assigned += _assign(
                production_row,
                source_row,
                method=f"{method_prefix}_exact_base_sequence",
            )
        return assigned, "exact"

    # Decode CUDA graphs are separately compiled for each batch shape.  NVJet
    # and DeepGEMM encode those shapes (and schedule digits) in the kernel
    # symbol even when the complete kernel-family schedule is unchanged.  A
    # full occurrence-local family-sequence match is therefore admissible:
    # the surrounding mHC anchors preserve layer/substage scope and only
    # shape/version digits are normalized.  This remains fail-closed for any
    # inserted, removed, or reordered kernel family.
    source_families = [
        schedule_family(str(row.get("kernel_name") or "")) for row in source
    ]
    production_families = [
        schedule_family(str(row.get("kernel_name") or "")) for row in production
    ]
    if source_families == production_families:
        for source_row, production_row in zip(source, production):
            assigned += _assign(
                production_row,
                source_row,
                method=f"{method_prefix}_shape_family_sequence",
            )
        return assigned, "shape_family"

    # CUDA Graph auxiliary streams may append work after an otherwise stable
    # main-stream tail.  Transfer only the verified common prefix; never align
    # the remainder positionally.
    common_prefix = 0
    for source_base, production_base in zip(source_bases, production_bases):
        if source_base != production_base:
            break
        common_prefix += 1
    for source_row, production_row in zip(
        source[:common_prefix], production[:common_prefix]
    ):
        assigned += _assign(
            production_row,
            source_row,
            method=f"{method_prefix}_exact_base_prefix",
        )

    family_prefix = 0
    for source_family, production_family in zip(
        source_families, production_families
    ):
        if source_family != production_family:
            break
        family_prefix += 1
    for source_row, production_row in zip(
        source[:family_prefix], production[:family_prefix]
    ):
        assigned += _assign(
            production_row,
            source_row,
            method=f"{method_prefix}_shape_family_prefix",
        )

    # A mismatched scope is not positionally transferred.  Only identities
    # unique inside that same semantic scope remain admissible.
    exact = unique_source_index(source, kernel_exact_identity)
    for production_row in production:
        source_row = exact.get(kernel_exact_identity(str(production_row.get("kernel_name") or "")))
        if source_row:
            assigned += _assign(
                production_row,
                source_row,
                method=f"{method_prefix}_unique_exact_identity",
            )

    # Repeated helpers are safe to transfer only inside the same reviewed mHC
    # occurrence and only when eager/production multiplicities agree.  This
    # covers graph-captured copies and elementwise helpers without introducing
    # a framework-global kernel-name rule.
    for key_fn, suffix in (
        (kernel_exact_identity, "repeated_exact_occurrence"),
        (kernel_base, "repeated_function_occurrence"),
        (schedule_family, "repeated_shape_family_occurrence"),
    ):
        source_groups: dict[str, list[dict[str, Any]]] = {}
        production_groups: dict[str, list[dict[str, Any]]] = {}
        for row in source:
            source_groups.setdefault(key_fn(str(row.get("kernel_name") or "")), []).append(row)
        for row in production:
            production_groups.setdefault(key_fn(str(row.get("kernel_name") or "")), []).append(row)
        for identity, source_group in source_groups.items():
            production_group = production_groups.get(identity) or []
            if (
                len(source_group) < 2
                or len(source_group) != len(production_group)
                or not all(row.get("selected_node") for row in source_group)
            ):
                continue
            for source_row, production_row in zip(source_group, production_group):
                assigned += _assign(
                    production_row,
                    source_row,
                    method=f"{method_prefix}_{suffix}",
                )
    return assigned, "mismatched"


def _assign_unanchored_production_schedules(events: list[dict[str, Any]]) -> int:
    """Attribute graph-replayed attention helpers that omit the mHC anchor.

    The validated vLLM prefill request contains a large first chunk with all 90
    mHC anchors and a short graph-replayed tail.  The replay exposes the
    attention kernels but does not repeat those Python/mHC anchor kernels.
    We therefore use model-unique attention landmarks plus the reviewed
    within-attention order.  This is deliberately narrower than a global
    kernel-name rule: only complete KDA/DSA schedules in the expected 34/11
    layer order are admitted.
    """

    assigned = 0
    first_logits_gather = next(
        (
            index
            for index, row in enumerate(events)
            if row.get("node") == "top.tp_logits_all_gather"
        ),
        -1,
    )
    replay_start = first_logits_gather + 1

    def assign_range(
        start: int,
        end: int,
        node: str,
        *,
        layer_id: int | None,
        method: str,
    ) -> None:
        nonlocal assigned
        for row in events[start:end]:
            if not row.get("node"):
                _assign(
                    row,
                    {"selected_node": node, "confidence": "high"},
                    method=method,
                )
                if row.get("node"):
                    assigned += 1
            if row.get("node") and layer_id is not None:
                row.update(
                    {
                        "layer_id": layer_id,
                        "layer_kind": "dsa" if layer_id % 4 == 3 else "kda",
                        "substage": "attention",
                        "occurrence_id": f"layer_{layer_id:02d}.attention.graph_replay",
                    }
                )

    embedding_collective = next(
        (
            index
            for index, row in enumerate(events)
            if row.get("node") == "top.tp_embedding_output_collective"
        ),
        None,
    )
    if embedding_collective is not None:
        embedding_start = max(
            (
                index
                for index in range(embedding_collective)
                if events[index].get("node") == "top.embedding"
            ),
            default=embedding_collective,
        )
        assign_range(
            embedding_start,
            embedding_collective,
            "top.embedding",
            layer_id=None,
            method="pre_decoder_embedding_collective_boundary",
        )
        first_mhc = next(
            (
                index
                for index in range(embedding_collective + 1, len(events))
                if "mhc_pre_big_fuse_with_norm"
                in str(events[index].get("kernel_name") or "").lower()
            ),
            embedding_collective + 1,
        )
        assign_range(
            embedding_collective + 1,
            first_mhc,
            "top.hc_expand",
            layer_id=None,
            method="pre_decoder_hc_expand_boundary",
        )

    # KDA replay: one unique causal-conv start and query-readout end per KDA
    # layer.  The helper intervals are the same ones proved by eager stacks.
    kda_starts = [
        index
        for index, row in enumerate(events[replay_start:], start=replay_start)
        if "_causal_conv1d_fwd_kernel" in str(row.get("kernel_name") or "").lower()
    ]
    kda_layers = [layer for layer in range(45) if layer % 4 != 3]
    if len(kda_starts) == len(kda_layers):
        for start, layer_id in zip(kda_starts, kda_layers):
            end = next(
                (
                    index + 1
                    for index in range(start, min(len(events), start + 48))
                    if "_scatter_states_kernel"
                    in str(events[index].get("kernel_name") or "").lower()
                ),
                None,
            )
            if end is None:
                continue
            local = events[start:end]

            def local_index(token: str) -> int | None:
                return next(
                    (
                        index
                        for index, row in enumerate(local)
                        if token in str(row.get("kernel_name") or "").lower()
                    ),
                    None,
                )

            state_read = local_index("_gather_initial_states_kernel")
            first_norm = local_index("l2norm_fwd_kernel")
            forget_decay = local_index("kda_gate_cumsum_fwd_kernel")
            first_update = local_index("chunk_kda_scaled_dot_kkt_fwd_kernel")
            readout = local_index("chunk_gla_fwd_kernel_o")
            if None in (state_read, first_norm, forget_decay, first_update, readout):
                continue
            assert state_read is not None
            assert first_norm is not None
            assert forget_decay is not None
            assert first_update is not None
            assert readout is not None
            for left, right, node in (
                (state_read + 1, first_norm, "linear_attention.beta_projection"),
                (first_norm, forget_decay, "linear_attention.qk_l2_norm"),
                (forget_decay, first_update, "linear_attention.forget_decay"),
                (first_update, readout, "linear_attention.recurrent_update"),
            ):
                assign_range(
                    start + left,
                    start + right,
                    node,
                    layer_id=layer_id,
                    method="graph_replay_kda_landmark_interval",
                )

    # DSA replay begins at key-pool metadata preparation; Q/K projections have
    # already executed on the primary path.  The explicit pool/cache/top-k/
    # sparse-MLA landmarks bound every generic helper in this replay tail.
    pool_rows = [
        index
        for index, row in enumerate(events[replay_start:], start=replay_start)
        if "_kpool_softmax_rotate_write_cache_kernel"
        in str(row.get("kernel_name") or "").lower()
    ]
    dsa_layers = [layer for layer in range(45) if layer % 4 == 3]
    if len(pool_rows) == len(dsa_layers):
        for pool_index, layer_id in zip(pool_rows, dsa_layers):
            previous_terminal = max(
                (
                    index
                    for index in range(pool_index)
                    if "_scatter_states_kernel"
                    in str(events[index].get("kernel_name") or "").lower()
                ),
                default=pool_index - 12,
            )
            cache_index = next(
                index
                for index in range(pool_index + 1, min(len(events), pool_index + 8))
                if "_kpool_tail_seed_kernel"
                in str(events[index].get("kernel_name") or "").lower()
            )
            topk_index = next(
                index
                for index in range(cache_index + 1, min(len(events), cache_index + 12))
                if "topkperrowprefill"
                in str(events[index].get("kernel_name") or "").lower()
            )
            expand_index = next(
                index
                for index in range(topk_index + 1, min(len(events), topk_index + 12))
                if "_expand_pools_and_append_tail_kernel"
                in str(events[index].get("kernel_name") or "").lower()
            )
            selected_index = next(
                index
                for index in range(expand_index + 1, min(len(events), expand_index + 16))
                if "_convert_req_index_to_global_index_kernel"
                in str(events[index].get("kernel_name") or "").lower()
            )
            sparse_index = next(
                index
                for index in range(selected_index + 1, min(len(events), selected_index + 16))
                if "fmhasm100fkernel_qkve4m3obfloat16h512"
                in str(events[index].get("kernel_name") or "").lower()
            )
            next_attention = next(
                (
                    index
                    for index in range(sparse_index + 1, len(events))
                    if any(
                        token in str(events[index].get("kernel_name") or "").lower()
                        for token in (
                            "_causal_conv1d_fwd_kernel",
                            "_kpool_softmax_rotate_write_cache_kernel",
                        )
                    )
                ),
                sparse_index + 3,
            )
            assign_range(
                previous_terminal + 1,
                pool_index + 1,
                "dsa_attention.key_pool_compression",
                layer_id=layer_id,
                method="graph_replay_dsa_landmark_interval",
            )
            assign_range(
                cache_index,
                topk_index,
                "dsa_attention.index_k_cache",
                layer_id=layer_id,
                method="graph_replay_dsa_landmark_interval",
            )
            assign_range(
                topk_index,
                expand_index + 1,
                "dsa_attention.token_expansion",
                layer_id=layer_id,
                method="graph_replay_dsa_landmark_interval",
            )
            assign_range(
                expand_index + 1,
                selected_index,
                "dsa_attention.latent_kv_reconstruction",
                layer_id=layer_id,
                method="graph_replay_dsa_landmark_interval",
            )
            assign_range(
                selected_index,
                sparse_index + 1,
                "dsa_attention.selected_indices",
                layer_id=layer_id,
                method="graph_replay_dsa_landmark_interval",
            )
            assign_range(
                sparse_index + 1,
                next_attention,
                "dsa_attention.output_projection",
                layer_id=layer_id,
                method="graph_replay_dsa_landmark_interval",
            )

    return assigned


def _assign_decode_graph_segment_schedules(events: list[dict[str, Any]]) -> int:
    """Close batch-specialized decode schedules inside exact mHC scopes.

    The decode graph uses update kernels and batch-specific NVJet symbols that
    intentionally differ from the graph-off eager trace.  We admit only the
    complete 34-KDA update schedule, then use the already-validated 90 mHC
    segments and model-unique landmarks.  No rule crosses an attention/FFN
    occurrence boundary.
    """

    update_count = sum(
        "_causal_conv1d_update_kernel"
        in str(row.get("kernel_name") or "").lower()
        for row in events
    )
    if update_count != 34:
        return 0

    _, segments = _segments(events)
    assigned = 0

    def assign(
        row: dict[str, Any], node: str, method: str, *, force: bool = False
    ) -> None:
        nonlocal assigned
        if force:
            changed = row.get("node") != node
            row.update(
                {
                    "node": node,
                    "kernel_label": node,
                    "attribution_method": method,
                    "confidence": "high",
                    "eager_event_id": None,
                }
            )
            if changed:
                assigned += 1
            return
        if _assign(
            row,
            {"selected_node": node, "confidence": "high"},
            method=method,
        ):
            assigned += 1

    for segment_id, segment in enumerate(segments):
        kind = segment_kind(segment_id)
        names = [str(row.get("kernel_name") or "").lower() for row in segment]
        method = f"mhc_anchor_bounded_decode_{kind}_schedule"
        if kind == "kda":
            nvjet = [
                index for index, name in enumerate(names) if name.startswith("nvjet_")
            ]
            if len(nvjet) >= 4:
                for index, node in zip(
                    (nvjet[0], nvjet[1], nvjet[2], nvjet[-1]),
                    (
                        "linear_attention.qkv_projection",
                        "linear_attention.forget_projection",
                        "linear_attention.beta_projection",
                        "linear_attention.output_projection",
                    ),
                ):
                    assign(segment[index], node, method, force=True)
            conv = next(
                (
                    index
                    for index, name in enumerate(names)
                    if "_causal_conv1d_update_kernel" in name
                ),
                None,
            )
            recurrent = next(
                (
                    index
                    for index, name in enumerate(names)
                    if "fused_recurrent_gated_delta_rule_fwd_kernel" in name
                ),
                None,
            )
            if conv is not None:
                assign(segment[conv], "linear_attention.qkv_short_conv", method)
            if conv is not None and recurrent is not None and conv < recurrent:
                for row in segment[conv + 1 : recurrent + 1]:
                    assign(row, "linear_attention.recurrent_update", method)

        elif kind == "dsa":
            for index, name in enumerate(names):
                if "_kpool_decode_update_batched_kernel" in name:
                    assign(segment[index], "dsa_attention.key_pool_compression", method)
                elif "sm100_paged_mqa_logits" in name:
                    assign(segment[index], "dsa_attention.index_logits", method)
                elif "filteredtopkunifiedkernel" in name or "persistent_topk_kernel" in name:
                    assign(segment[index], "dsa_attention.top_pool_selection", method)

            qnorm = next(
                (index for index, name in enumerate(names) if "_fused_q_kv_rmsnorm" in name),
                None,
            )
            kpool = next(
                (index for index, name in enumerate(names) if "_kpool_decode_update" in name),
                None,
            )
            if qnorm is not None:
                before = [
                    index
                    for index, name in enumerate(names[:qnorm])
                    if name.startswith("nvjet_")
                ]
                if len(before) == 1:
                    assign(segment[before[0]], "dsa_attention.q_a_projection", method)
            if qnorm is not None and kpool is not None:
                after = [
                    index
                    for index in range(qnorm + 1, kpool)
                    if names[index].startswith("nvjet_")
                ]
                if len(after) >= 3:
                    assign(segment[after[0]], "dsa_attention.q_b_projection", method)
                    assign(segment[after[1]], "dsa_attention.index_q_projection", method)
                    assign(segment[after[2]], "dsa_attention.index_k_projection", method)
                    for index in after[3:]:
                        assign(
                            segment[index],
                            "dsa_attention.index_weight_projection",
                            method,
                        )

            # BS16+ graph specialization replaces the eager FP32 index-weight
            # GEMV with one SIMT SGEMM plus its split-K reduction.  The symbol
            # is not unique model-wide, so bind it only inside this exact DSA
            # occurrence and only when it is bracketed by the eager-proven
            # index-K normalization and index-weight scale landmarks.
            index_k_norm = next(
                (
                    index
                    for index, name in enumerate(names)
                    if "native_layer_norm" in name
                ),
                None,
            )
            weight_scale = next(
                (
                    index
                    for index, name in enumerate(names)
                    if "triton_poi_fused_mul_unsqueeze" in name
                ),
                None,
            )
            if index_k_norm is not None and weight_scale is not None:
                fp32_weight_gemm = [
                    index
                    for index in range(qnorm + 1 if qnorm is not None else 0, index_k_norm)
                    if "cutlass_80_simt_sgemm" in names[index]
                ]
                if len(fp32_weight_gemm) == 1:
                    start = fp32_weight_gemm[0]
                    stop = index_k_norm
                    if (
                        stop == start + 2
                        and "splitkreduce_kernel" in names[start + 1]
                    ):
                        for row in segment[start:stop]:
                            assign(
                                row,
                                "dsa_attention.index_weight_projection",
                                method,
                            )

            # The decode graph also shape-specializes latent-KV
            # reconstruction.  Eager proves the semantic sequence
            # concat-cache -> BMM -> concatenate -> FP8 quantize.  Transfer
            # that complete local interval instead of leaving the BMM and cat
            # as generic graph-runtime work.
            latent_cache = next(
                (
                    index
                    for index, name in enumerate(names)
                    if "concat_and_cache_mla_kernel" in name
                ),
                None,
            )
            latent_quant = next(
                (
                    index
                    for index, name in enumerate(names)
                    if "scaled_fp8_quant_kernel" in name
                ),
                None,
            )
            if (
                latent_cache is not None
                and latent_quant is not None
                and latent_quant == latent_cache + 3
                and names[latent_cache + 1].startswith("nvjet_")
                and "catarraybatchedcopy" in names[latent_cache + 2]
            ):
                for row in segment[latent_cache + 1 : latent_quant + 1]:
                    assign(
                        row,
                        "dsa_attention.latent_kv_reconstruction",
                        method,
                    )
            allreduce = next(
                (
                    index
                    for index, name in enumerate(names)
                    if "multimem_all_reduce_kernel" in name
                ),
                None,
            )
            if allreduce is not None:
                output_nvjet = [
                    index
                    for index in range(max(0, allreduce - 8), allreduce)
                    if names[index].startswith("nvjet_")
                ]
                if len(output_nvjet) == 2:
                    assign(
                        segment[output_nvjet[0]],
                        "dsa_attention.latent_kv_reconstruction",
                        method,
                    )
                    assign(
                        segment[output_nvjet[1]],
                        "dsa_attention.output_projection",
                        method,
                    )

        elif kind == "moe":
            dispatch_quant = next(
                (
                    index
                    for index, name in enumerate(names)
                    if "per_token_group_quant_8bit_kernel" in name
                ),
                None,
            )
            if dispatch_quant is not None:
                for row in segment[1:dispatch_quant]:
                    assign(row, "moe.router", method)
            for index, name in enumerate(names):
                if "routingindicesclusterkernel" in name:
                    assign(segment[index], "moe.dispatch", method)
                elif (
                    "routingindicesblockkernel" in name
                    or "routingindicesdynblockkernel" in name
                ):
                    assign(segment[index], "moe.topk", method)
                elif "act_and_mul_kernel" in name:
                    assign(segment[index], "moe.shared_activation", method)
            topk = next(
                (
                    index
                    for index, name in enumerate(names)
                    if "routingindicesblockscoreskernel" in name
                ),
                None,
            )
            if dispatch_quant is not None and topk is not None:
                for row in segment[dispatch_quant + 1 : topk]:
                    assign(row, "moe.correction_bias", method)

    return assigned


def _classify_runtime_support(events: list[dict[str, Any]]) -> None:
    """Classify every intentionally non-architectural production interval."""

    for row in events:
        if row.get("node"):
            continue
        name = str(row.get("kernel_name") or "").lower()
        if any(
            token in name
            for token in (
                "sampling",
                "gumbel",
                "argmax",
                "combine_draft_token",
                "apply_write_kernel",
                "logits_processor",
            )
        ):
            support_class = "sampling_and_output"
            reason = "sampling, token selection, or output materialization"
        elif any(
            token in name
            for token in (
                "mamba_state",
                "align_mamba",
                "state_indices",
                "state_slot",
            )
        ):
            support_class = "state_bookkeeping"
            reason = "persistent recurrent-state index or slot bookkeeping"
        elif any(
            token in name
            for token in (
                "block_table",
                "slot_mapping",
                "prepare_prefill",
                "kpool_build_ragged_layout",
                "paged_mqa_logits_metadata",
            )
        ):
            support_class = "attention_plan_metadata"
            reason = "attention plan/block-table metadata; no model tensor value is produced"
        elif any(
            token in name
            for token in (
                "zero_kv_blocks",
                "cache_block",
                "free_block",
                "alloc",
                "req_to_token",
            )
        ):
            support_class = "allocator_or_cache_management"
            reason = "KV/request allocation or cache-address management"
        elif any(
            token in name
            for token in (
                "fillfunctor",
                "direct_copy_kernel",
                "indexelementwise",
                "scan",
                "arange",
                "divfloor",
                "scatter",
                "cudafunctor_add",
                "compare",
                "where",
            )
        ):
            support_class = "request_batch_metadata"
            reason = "shape/index/request-batch metadata preparation outside a semantic module"
        else:
            support_class = "graph_runtime_metadata"
            reason = "captured framework/runtime helper outside the stable Model-IR contract"
        row.update(
            {
                "support_class": support_class,
                "support_reason": reason,
                "attribution_method": "explicit_runtime_support_classification",
                "confidence": "support",
            }
        )


def attribute_vllm_production_events(
    events: list[dict[str, Any]],
    eager_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    source = _first_eager_forward(eager_rows)
    source_prefix, source_segments = _segments(source)
    production_prefix, production_segments = _segments(events)

    for segment_id, segment in enumerate(source_segments):
        _annotate_segment_scope(segment, segment_id)
        _enrich_source_segment(segment, segment_kind(segment_id))
    for segment_id, segment in enumerate(production_segments):
        _annotate_segment_scope(segment, segment_id)

    assigned, prefix_state = _transfer_matching_scope(
        source_prefix,
        production_prefix,
        method_prefix="pre_decoder_eager",
    )
    segment_states: Counter[str] = Counter()
    for segment_id, (source_segment, production_segment) in enumerate(
        zip(source_segments, production_segments)
    ):
        count, state = _transfer_matching_scope(
            source_segment,
            production_segment,
            method_prefix=f"mhc_anchor_bounded_{segment_kind(segment_id)}",
        )
        assigned += count
        segment_states[state] += 1

    assigned += _assign_unanchored_production_schedules(events)
    assigned += _assign_decode_graph_segment_schedules(events)
    _classify_runtime_support(events)

    mapped = [row for row in events if row.get("node")]
    total_us = sum(float(row.get("dur_us") or 0.0) for row in events)
    mapped_us = sum(float(row.get("dur_us") or 0.0) for row in mapped)
    return {
        "eager_forward_kernel_count": len(source),
        "production_kernel_count": len(events),
        "anchor_count": SUBLAYER_SEGMENT_COUNT,
        "prefix_state": prefix_state,
        "segment_results": dict(segment_states),
        "bounded_assignments": assigned,
        "mapped_kernel_count": len(mapped),
        "mapped_kernel_count_ratio": len(mapped) / len(events),
        "mapped_kernel_duration_ratio": mapped_us / total_us,
        "support_class_counts": dict(
            Counter(
                str(row.get("support_class"))
                for row in events
                if row.get("support_class")
            )
        ),
        "method_counts": dict(Counter(str(row.get("attribution_method")) for row in events)),
    }
