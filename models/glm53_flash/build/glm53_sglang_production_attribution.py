"""Fail-closed SGLang eager-to-production attribution for GLM-5.3-Flash.

Kernel names alone are not a semantic identity: the same code-generated GEMM
name is reused by KDA, DSA, dense MLP, and MoE.  GLM-5.3 supplies a stronger
execution invariant.  Every target layer launches two stable mHC-pre anchors
(attention and feed-forward), so one forward contains exactly 90 anchor-led
segments.  Attribution is transferred inside those bounded segments and the
collective sequence is checked end-to-end.  A global name match is never used
inside a decoder segment.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

from models.glm52.build.build_glm52_production_profile import (
    inferred_collective_kind,
    kernel_base,
    kernel_exact_identity,
    schedule_family,
    sequence_family,
    unique_source_index,
)
from models.glm53_flash.build.glm53_sglang_trace_rules import (
    classify_glm53_sglang_node,
)


LAYER_COUNT = 45
SUBLAYER_SEGMENT_COUNT = LAYER_COUNT * 2
ANCHOR_TOKEN = "mhc_pre_big_fuse_with_norm"

COLLECTIVE_NODES = {
    "top.tp_embedding_output_collective",
    "linear_attention.tp_kda_output_collective",
    "dsa_attention.tp_dsa_output_collective",
    "dense_mlp.tp_dense_mlp_output_collective",
    "moe.tp_moe_output_collective",
    "top.tp_logits_all_gather",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _is_anchor(row: dict[str, Any]) -> bool:
    return ANCHOR_TOKEN in str(row.get("kernel_name") or "").lower()


def anchor_segments(rows: list[dict[str, Any]]) -> list[tuple[int, int]]:
    anchors = [index for index, row in enumerate(rows) if _is_anchor(row)]
    if len(anchors) != SUBLAYER_SEGMENT_COUNT:
        raise ValueError(
            f"expected {SUBLAYER_SEGMENT_COUNT} GLM-5.3 mHC sublayer anchors, "
            f"got {len(anchors)}"
        )
    return list(zip(anchors, [*anchors[1:], len(rows)]))


def segment_kind(segment_id: int) -> str:
    layer = segment_id // 2
    if segment_id % 2 == 0:
        return "dsa" if layer % 4 == 3 else "kda"
    return "dense" if layer < 3 else "moe"


def annotate_segment_scope(
    segment: list[dict[str, Any]], segment_id: int
) -> None:
    """Persist the semantic occurrence used for bounded attribution.

    The anchor boundary is part of the evidence, not a temporary matching
    implementation detail.  Parent timing roll-ups require this coordinate to
    separate attention from feed-forward occurrences without duplicating a
    profile-wide fusion-owner scalar.
    """

    layer_id = segment_id // 2
    substage = "attention" if segment_id % 2 == 0 else "feed_forward"
    occurrence_id = f"layer_{layer_id:02d}.{substage}"
    for row in segment:
        row.update(
            {
                "layer_id": layer_id,
                "layer_kind": segment_kind(segment_id),
                "substage": substage,
                "segment_id": segment_id,
                "occurrence_id": occurrence_id,
            }
        )


def _assign(
    row: dict[str, Any],
    node: str | None,
    *,
    method: str,
    confidence: str = "high",
    source: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> None:
    if not node or (row.get("node") and not overwrite):
        return
    row.update(
        {
            "node": node,
            "kernel_label": (source or {}).get("cpu_op_name") or node,
            "attribution_method": method,
            "confidence": confidence,
        }
    )
    if source and source.get("event_id"):
        row["eager_event_id"] = source["event_id"]


def _source_stack_text(row: dict[str, Any]) -> str:
    values: list[str] = []
    for key in ("primitive_frame", "operator_frame", "semantic_frame", "model_context_frame"):
        frame = row.get(key) or {}
        if frame.get("raw"):
            values.append(str(frame["raw"]))
    return "\n".join(values).lower()


def _enrich_source_segment(segment: list[dict[str, Any]], kind: str) -> None:
    """Close eager semantic leaves using stack/segment-local execution facts."""

    for row in segment:
        name = str(row.get("kernel_name") or "")
        lowered = name.lower()
        base = kernel_base(name)
        stack = _source_stack_text(row)
        cpu = str(row.get("cpu_op_name") or "").lower()

        if ANCHOR_TOKEN in lowered or "sm100_tf32_hc_prenorm_gemm" in lowered:
            row["selected_node"] = "mhc_transform.pre_weights"
        elif "mhc_post" in lowered:
            row["selected_node"] = "mhc_transform.residual_mix"
        elif kind == "kda":
            if "tgvgemmcuteextkernel" in lowered:
                row["selected_node"] = "linear_attention.qkv_projection"
            elif "_causal_conv1d_" in lowered:
                row["selected_node"] = "linear_attention.qkv_short_conv"
            elif "sigmoid_gating_delta_rule" in lowered:
                row["selected_node"] = "linear_attention.recurrent_update"
            elif "layer_norm_gated" in lowered:
                row["selected_node"] = "linear_attention.gated_norm"
        elif kind == "dense":
            # The dense segment is stable: quant+GEMM, activation,
            # quant+GEMM, collective, mHC post, next mHC pre-GEMM.
            quant_or_gemm = [
                item
                for item in segment
                if "per_token_group_quant" in str(item.get("kernel_name") or "").lower()
                or "sm100_fp8_fp4_gemm" in str(item.get("kernel_name") or "").lower()
            ]
            if row in quant_or_gemm:
                midpoint = len(quant_or_gemm) // 2
                row["selected_node"] = (
                    "dense_mlp.gate_up_projection"
                    if quant_or_gemm.index(row) < midpoint
                    else "dense_mlp.down_projection"
                )
            elif "silu_mul_clamp" in lowered:
                row["selected_node"] = "dense_mlp.clamped_swiglu"
        elif kind == "moe":
            if "routingindices" in lowered:
                row["selected_node"] = "moe.topk"
            elif base.startswith("bmm_e4m3"):
                row["selected_node"] = "moe.routed_gate_up"
            elif "activationdeepseek" in lowered:
                row["selected_node"] = "moe.routed_activation"
            elif base.startswith("bmm_bfloat16"):
                row["selected_node"] = "moe.routed_down"
            elif "finalizekernel" in lowered:
                row["selected_node"] = "moe.routed_weighted_combine"

            shared_rows = [
                item
                for item in segment
                if any(
                    token in str(item.get("kernel_name") or "").lower()
                    for token in (
                        "per_token_group_quant",
                        "sm100_fp8_fp4_gemm",
                        "silu_mul_clamp",
                    )
                )
            ]
            if row in shared_rows:
                activation_index = next(
                    (
                        index
                        for index, item in enumerate(shared_rows)
                        if "silu_mul_clamp"
                        in str(item.get("kernel_name") or "").lower()
                    ),
                    -1,
                )
                position = shared_rows.index(row)
                if position < activation_index:
                    row["selected_node"] = "moe.shared_gate_up"
                elif position == activation_index:
                    row["selected_node"] = "moe.shared_activation"
                elif activation_index >= 0:
                    row["selected_node"] = "moe.shared_down"
        elif kind == "dsa":
            if "prepare_qkv_latent" in stack:
                row["selected_node"] = "dsa_attention.q_a_projection"
            elif "q_b_proj_forward" in stack:
                row["selected_node"] = "dsa_attention.q_b_projection"
            elif "_get_q_k_bf16" in stack:
                if "layernorm" in base or "hadamard" in base:
                    row["selected_node"] = "dsa_attention.index_k_projection"
                elif not row.get("selected_node"):
                    # The first MM is Q-index; the split-K MM+reduce is K-index.
                    mm_rows = [
                        item
                        for item in segment
                        if "_get_q_k_bf16" in _source_stack_text(item)
                        and str(item.get("cpu_op_name") or "").lower() == "aten::mm"
                    ]
                    row["selected_node"] = (
                        "dsa_attention.index_q_projection"
                        if mm_rows and row is mm_rows[0]
                        else "dsa_attention.index_k_projection"
                    )
            elif "_compress_write" in stack:
                row["selected_node"] = (
                    "dsa_attention.index_k_cache"
                    if "kpool_decode_update" in lowered
                    else "dsa_attention.key_pool_compression"
                )
            elif "_get_logits_head_gate" in stack:
                row["selected_node"] = "dsa_attention.index_weight_projection"
            elif "forward_core" in stack:
                if "prepare_trtllm_nope_sparse_metadata" in lowered:
                    row["selected_node"] = "dsa_attention.selected_indices"
                elif cpu in {"aten::copy_", "aten::fill_", "aten::cat"}:
                    row["selected_node"] = "dsa_attention.token_expansion"
                elif "per_token_group_quant" in lowered or "sm100_fp8_fp4_gemm" in lowered:
                    row["selected_node"] = "dsa_attention.output_projection"

    if kind == "dsa":
        quant_rows = [
            row
            for row in segment
            if "per_token_group_quant" in str(row.get("kernel_name") or "").lower()
        ]
        deep_rows = [
            row
            for row in segment
            if "sm100_fp8_fp4_gemm" in str(row.get("kernel_name") or "").lower()
        ]
        for rows, nodes in (
            (quant_rows, ("dsa_attention.q_a_projection", "dsa_attention.q_b_projection", "dsa_attention.output_projection")),
            (deep_rows, ("dsa_attention.q_a_projection", "dsa_attention.q_b_projection", "dsa_attention.output_projection")),
        ):
            if len(rows) >= 3:
                rows[0]["selected_node"], rows[1]["selected_node"], rows[-1]["selected_node"] = nodes

        rms_rows = [
            row
            for row in segment
            if "rmsnormkernel" in str(row.get("kernel_name") or "").lower()
        ]
        if len(rms_rows) >= 2:
            rms_rows[0]["selected_node"] = "dsa_attention.q_a_norm"
            rms_rows[1]["selected_node"] = "dsa_attention.kv_a_norm"

        index_norm = next(
            (
                index
                for index, row in enumerate(segment)
                if "layernormkernel" in str(row.get("kernel_name") or "").lower()
            ),
            None,
        )
        if index_norm is not None:
            projection_rows = [
                row
                for row in segment[:index_norm]
                if "nvjet_sm" in str(row.get("kernel_name") or "").lower()
            ]
            if projection_rows:
                projection_rows[0]["selected_node"] = "dsa_attention.index_q_projection"
            if len(projection_rows) >= 2:
                projection_rows[1]["selected_node"] = "dsa_attention.index_k_projection"
            segment[index_norm]["selected_node"] = "dsa_attention.index_k_projection"


def _assign_dsa_production_schedule(segment: list[dict[str, Any]]) -> None:
    """Classify DSA projection bundles by their bounded launch schedule."""

    quant_rows = [
        row
        for row in segment
        if "per_token_group_quant" in str(row.get("kernel_name") or "").lower()
    ]
    deep_rows = [
        row
        for row in segment
        if "sm100_fp8_fp4_gemm" in str(row.get("kernel_name") or "").lower()
    ]
    for rows, nodes in (
        (quant_rows, ("dsa_attention.q_a_projection", "dsa_attention.q_b_projection", "dsa_attention.output_projection")),
        (deep_rows, ("dsa_attention.q_a_projection", "dsa_attention.q_b_projection", "dsa_attention.output_projection")),
    ):
        if len(rows) >= 3:
            for row, node in zip((rows[0], rows[1], rows[-1]), nodes):
                _assign(
                    row,
                    node,
                    method="mhc_anchor_bounded_dsa_projection_schedule",
                    overwrite=True,
                )

    rms_rows = [
        row
        for row in segment
        if "rmsnormkernel" in str(row.get("kernel_name") or "").lower()
    ]
    if len(rms_rows) >= 2:
        _assign(
            rms_rows[0],
            "dsa_attention.q_a_norm",
            method="mhc_anchor_bounded_dsa_norm_order",
            overwrite=True,
        )
        _assign(
            rms_rows[1],
            "dsa_attention.kv_a_norm",
            method="mhc_anchor_bounded_dsa_norm_order",
            overwrite=True,
        )

    index_norm = next(
        (
            index
            for index, row in enumerate(segment)
            if "layernormkernel" in str(row.get("kernel_name") or "").lower()
        ),
        None,
    )
    if index_norm is not None:
        projection_rows = [
            row
            for row in segment[:index_norm]
            if "nvjet_sm" in str(row.get("kernel_name") or "").lower()
        ]
        if projection_rows:
            _assign(
                projection_rows[0],
                "dsa_attention.index_q_projection",
                method="mhc_anchor_bounded_dsa_index_projection_order",
                overwrite=True,
            )
        if len(projection_rows) >= 2:
            _assign(
                projection_rows[1],
                "dsa_attention.index_k_projection",
                method="mhc_anchor_bounded_dsa_index_projection_order",
                overwrite=True,
            )
        _assign(
            segment[index_norm],
            "dsa_attention.index_k_projection",
            method="mhc_anchor_bounded_dsa_index_norm",
            overwrite=True,
        )

    for landmark in ("kpool_topk_transform", "fmhasm"):
        landmark_index = next(
            (
                index
                for index, row in enumerate(segment)
                if landmark in str(row.get("kernel_name") or "").lower()
            ),
            None,
        )
        if landmark_index is None:
            continue
        reconstruction = next(
            (
                row
                for row in segment[landmark_index + 1 :]
                if "nvjet_sm" in str(row.get("kernel_name") or "").lower()
            ),
            None,
        )
        if reconstruction:
            _assign(
                reconstruction,
                "dsa_attention.latent_kv_reconstruction",
                method="mhc_anchor_bounded_dsa_reconstruction_after_landmark",
                overwrite=True,
            )


def _assign_dsa_production_stream_schedules(rows: list[dict[str, Any]]) -> None:
    """Resolve DSA auxiliary streams by reviewed landmark-bounded schedules.

    SGLang intentionally launches the DSA indexer on per-layer auxiliary
    streams.  Those events can cross the main-stream mHC timestamp boundary,
    so a global timestamp segment alone is insufficient.  The stream-local
    schedules below are stable and independently evidenced by the eager stack:

    * index-Q -> Hadamard/quant -> index-weight projection;
    * index-K projection -> LayerNorm -> K-pool cache update;
    * logits/top-k -> KV reconstruction/packing -> sparse MLA -> output.

    A row is assigned only when its own stream contains the corresponding
    unique landmark, so generic NVJet/copy helpers cannot leak into KDA/MoE.
    """

    by_stream: dict[Any, list[dict[str, Any]]] = {}
    for row in sorted(rows, key=lambda item: float(item.get("ts_us") or 0.0)):
        by_stream.setdefault(row.get("stream"), []).append(row)

    for stream_rows in by_stream.values():
        names = [str(row.get("kernel_name") or "").lower() for row in stream_rows]

        # The main compute stream contains every decoder sublayer and can also
        # contain a Hadamard/K-pool landmark during eager prefill.  Treating
        # that one landmark as a stream-wide DSA scope would incorrectly
        # relabel unrelated KDA/MoE GEMMs.  Timestamp-bounded mHC segments are
        # authoritative for streams that carry decoder anchors; this helper is
        # only for anchor-less auxiliary streams whose work crosses those
        # timestamp boundaries.
        if any(ANCHOR_TOKEN in name for name in names):
            continue

        hadamard_index = next(
            (index for index, name in enumerate(names) if "fast_hadamard_transform_kernel" in name),
            None,
        )
        if hadamard_index is not None:
            for index, row in enumerate(stream_rows):
                name = names[index]
                node: str | None = None
                if index < hadamard_index and "nvjet_sm" in name:
                    node = "dsa_attention.index_q_projection"
                elif index == hadamard_index:
                    node = "dsa_attention.index_k_projection"
                elif "_act_quant_kernel" in name:
                    node = "dsa_attention.key_pool_compression"
                elif index > hadamard_index and any(
                    token in name
                    for token in (
                        "triton_poi_fused__to_copy",
                        "cutlass_80_simt_sgemm",
                        "splitkreduce_kernel<32, 16, int, float, float",
                        "triton_poi_fused_mul_unsqueeze",
                    )
                ):
                    node = "dsa_attention.index_weight_projection"
                if node:
                    _assign(
                        row,
                        node,
                        method="dsa_aux_stream_hadamard_weight_schedule",
                        overwrite=True,
                    )

        cache_update_index = next(
            (
                index
                for index, name in enumerate(names)
                if "kpool_decode_update_and_maybe_write_cache" in name
                or "kpool_softmax_rotate_write_cache" in name
                or "kpool_tail_seed" in name
            ),
            None,
        )
        if cache_update_index is not None:
            for index, row in enumerate(stream_rows):
                name = names[index]
                node: str | None = None
                if index < cache_update_index and (
                    "nvjet_sm" in name or "splitkreduce_kernel" in name
                ):
                    node = "dsa_attention.key_pool_compression"
                elif index < cache_update_index and "layernormkernel" in name:
                    node = "dsa_attention.index_k_projection"
                elif index == cache_update_index:
                    node = "dsa_attention.index_k_cache"
                elif "sm100_paged_mqa_logits" in name or "paged_mqa_logits" in name:
                    node = "dsa_attention.index_logits"
                elif "kpool_topk_transform" in name:
                    node = "dsa_attention.top_pool_selection"
                elif "set_mla_kv_buffer" in name or "concat_and_cache_mla" in name:
                    node = "dsa_attention.latent_kv_cache"
                elif "prepare_trtllm_nope_sparse_metadata" in name:
                    node = "dsa_attention.selected_indices"
                elif "fmhasm" in name:
                    node = "dsa_attention.sparse_mla_core"
                if node:
                    _assign(
                        row,
                        node,
                        method="dsa_aux_stream_cache_attention_schedule",
                        overwrite=True,
                    )

            topk_index = next(
                (index for index, name in enumerate(names) if "kpool_topk_transform" in name),
                None,
            )
            sparse_index = next(
                (index for index, name in enumerate(names) if "fmhasm" in name),
                None,
            )
            cache_index = next(
                (
                    index
                    for index, name in enumerate(names)
                    if "set_mla_kv_buffer" in name or "concat_and_cache_mla" in name
                ),
                None,
            )
            if topk_index is not None and sparse_index is not None:
                for index, row in enumerate(stream_rows):
                    name = names[index]
                    if topk_index < index < sparse_index and "nvjet_sm" in name:
                        _assign(
                            row,
                            "dsa_attention.latent_kv_reconstruction",
                            method="dsa_aux_stream_reconstruction_after_topk",
                            overwrite=True,
                        )
                    elif sparse_index < index and "nvjet_sm" in name:
                        _assign(
                            row,
                            "dsa_attention.latent_kv_reconstruction",
                            method="dsa_aux_stream_reconstruction_after_sparse_mla",
                            overwrite=True,
                        )
                    elif topk_index < index < sparse_index and any(
                        token in name
                        for token in (
                            "converttofloat8",
                            "float8_copy_kernel",
                            "fillfunctor<int>",
                            "catarraybatchedcopy",
                            "memcpy32_post",
                        )
                    ):
                        _assign(
                            row,
                            "dsa_attention.token_expansion",
                            method="dsa_aux_stream_token_pack_schedule",
                            overwrite=True,
                        )
                if cache_index is not None:
                    _assign(
                        stream_rows[cache_index],
                        "dsa_attention.latent_kv_cache",
                        method="dsa_aux_stream_cache_landmark",
                        overwrite=True,
                    )

        # The other DSA projection stream contains the KV-A RMSNorm followed
        # by the index-K split-K projection and its reduction.
        if (
            hadamard_index is None
            and cache_update_index is None
            and any("rmsnormkernel" in name for name in names)
            and any("splitk_tnt" in name.lower() or "splitk_tnn" in name.lower() for name in names)
        ):
            for row, name in zip(stream_rows, names):
                if "rmsnormkernel" in name:
                    node = "dsa_attention.kv_a_norm"
                elif "nvjet_sm" in name or "splitkreduce_kernel" in name:
                    node = "dsa_attention.index_k_projection"
                else:
                    continue
                _assign(
                    row,
                    node,
                    method="dsa_aux_stream_index_k_projection_schedule",
                    overwrite=True,
                )


def _assign_kda_production_stream_schedules(rows: list[dict[str, Any]]) -> None:
    """Resolve one graph-replayed KDA sublayer launched on an auxiliary stream.

    Torch traces can place the last captured KDA sublayer on its graph child
    stream.  It has one mHC-pre anchor followed by the same source-defined
    QKV/forget bundle, causal conv, gated recurrence and output projection as a
    main-stream KDA segment.  Requiring all of those landmarks makes the scope
    unique and prevents the generic NVJet names from leaking into DSA or MoE.
    """

    by_stream: dict[Any, list[dict[str, Any]]] = {}
    for row in sorted(rows, key=lambda item: float(item.get("ts_us") or 0.0)):
        by_stream.setdefault(row.get("stream"), []).append(row)

    for stream_rows in by_stream.values():
        names = [str(row.get("kernel_name") or "").lower() for row in stream_rows]
        anchor_indices = [
            index for index, name in enumerate(names) if ANCHOR_TOKEN in name
        ]
        causal_index = next(
            (index for index, name in enumerate(names) if "_causal_conv1d_" in name),
            None,
        )
        norm_index = next(
            (index for index, name in enumerate(names) if "layer_norm_gated" in name),
            None,
        )
        collective_index = next(
            (
                index
                for index, row in enumerate(stream_rows)
                if norm_index is not None
                and index > norm_index
                if inferred_collective_kind(str(row.get("kernel_name") or ""))
                == "all_reduce"
            ),
            None,
        )
        if len(anchor_indices) != 1 or causal_index is None or norm_index is None:
            continue

        anchor_index = anchor_indices[0]
        projection_rows = [
            (index, row)
            for index, row in enumerate(stream_rows)
            if anchor_index < index < causal_index
            and any(
                token in names[index]
                for token in ("nvjet_sm", "tgvgemmcute", "cutlass::kernel2", "splitkreduce")
            )
        ]
        if projection_rows:
            first_index, first_row = projection_rows[0]
            if not first_row.get("node"):
                _assign(
                    first_row,
                    "linear_attention.qkv_projection",
                    method="kda_aux_stream_projection_schedule",
                    overwrite=True,
                )
            for _index, row in projection_rows:
                if _index == first_index or row.get("node"):
                    continue
                _assign(
                    row,
                    "linear_attention.forget_projection",
                    method="kda_aux_stream_projection_schedule",
                    overwrite=True,
                )

        output_stop = collective_index if collective_index is not None else len(stream_rows)
        for index, row in enumerate(stream_rows):
            if norm_index < index < output_stop and "nvjet_sm" in names[index]:
                _assign(
                    row,
                    "linear_attention.output_projection",
                    method="kda_aux_stream_output_projection_schedule",
                    overwrite=True,
                )


def _assign_outer_model_boundaries(rows: list[dict[str, Any]]) -> None:
    """Attribute explicit model-boundary work outside decoder mHC segments."""

    first_anchor = next(
        (index for index, row in enumerate(rows) if _is_anchor(row)), None
    )
    if first_anchor is None:
        return
    embedding_collective = max(
        (
            index
            for index, row in enumerate(rows[:first_anchor])
            if inferred_collective_kind(str(row.get("kernel_name") or ""))
            == "all_reduce"
        ),
        default=-1,
    )
    for row in rows[embedding_collective + 1 : first_anchor]:
        if row.get("node"):
            continue
        name = str(row.get("kernel_name") or "").lower()
        if "fillfunctor<float>" in name or "direct_copy_kernel_cuda" in name or "cudafunctor_add<float>" in name:
            _assign(
                row,
                "top.hc_expand",
                method="embedding_collective_to_first_mhc_boundary",
                overwrite=True,
            )


def _classify_runtime_support(rows: list[dict[str, Any]]) -> None:
    """Give every non-IR production event an explicit support contract.

    This runs only after all architecture attribution passes. It does not turn
    runtime work into Model-IR leaves; it records why an interval is
    intentionally outside the architecture. The independent timeline audit
    still rejects semantic-looking kernels, so this cannot hide missing model
    GEMM, attention, MoE, norm, convolution, or collective work.
    """

    logits_gather = max(
        (
            index
            for index, row in enumerate(rows)
            if row.get("node") == "top.tp_logits_all_gather"
        ),
        default=-1,
    )
    for index, row in enumerate(rows):
        if row.get("node"):
            continue
        name = str(row.get("kernel_name") or "").lower()
        if index > logits_gather >= 0:
            support_class = "sampling_and_output"
            reason = "post-logits sampling, token selection, or response materialization"
        elif any(
            token in name
            for token in (
                "track_mamba_states_all_layers",
                "_gather_initial_states_kernel",
                "_scatter_states_kernel",
                "state_indices",
            )
        ):
            support_class = "state_bookkeeping"
            reason = "persistent recurrent-state index or cache bookkeeping"
        elif any(
            token in name
            for token in (
                "kpool_build_ragged_layout",
                "paged_mqa_logits_metadata",
                "topk_plan",
                "block_table",
                "slot_mapping",
            )
        ):
            support_class = "attention_plan_metadata"
            reason = "attention planning metadata; no model tensor value is produced"
        elif any(
            token in name
            for token in (
                "alloc",
                "request_pool",
                "req_to_token",
                "write_req",
                "zero_kv",
                "cache_indices",
            )
        ):
            support_class = "allocator_or_cache_management"
            reason = "request/KV allocation or cache-address management"
        elif any(
            token in name
            for token in (
                "scan",
                "arange",
                "divfloor",
                "indexelementwise",
                "fillfunctor",
                "direct_copy_kernel",
                "cudafunctor_add",
                "cudafunctoronself",
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


def _assign_final_model_tail(segment: list[dict[str, Any]]) -> None:
    post_index = max(
        (
            index
            for index, row in enumerate(segment)
            if "mhc_post" in str(row.get("kernel_name") or "").lower()
        ),
        default=-1,
    )
    if post_index < 0:
        return
    tail = segment[post_index + 1 :]
    gather_index = next(
        (
            index
            for index, row in enumerate(tail)
            if inferred_collective_kind(str(row["kernel_name"])) == "all_gather"
        ),
        None,
    )
    if gather_index is None:
        return
    before_gather = tail[:gather_index]
    norm_rows = [
        row
        for row in before_gather
        if "rmsnormkernel" in str(row.get("kernel_name") or "").lower()
    ]
    if norm_rows:
        _assign(
            norm_rows[-1],
            "top.final_norm",
            method="final_tail_norm_before_logits_gather",
            overwrite=True,
        )
    gemm_rows = [
        row
        for row in before_gather
        if "nvjet_sm" in str(row.get("kernel_name") or "").lower()
        or "gemm" in kernel_base(str(row.get("kernel_name") or ""))
    ]
    if gemm_rows:
        _assign(
            gemm_rows[-1],
            "top.lm_head",
            method="final_tail_lm_head_before_logits_gather",
            overwrite=True,
        )


def load_enriched_eager_mapping(path: Path) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    for segment_id, (start, stop) in enumerate(anchor_segments(rows)):
        _enrich_source_segment(rows[start:stop], segment_kind(segment_id))
    return rows


def _bounded_transfer(
    production: list[dict[str, Any]],
    source: list[dict[str, Any]],
    *,
    segment_id: int,
) -> str:
    source_names = [str(row["kernel_name"]) for row in source]
    production_names = [str(row["kernel_name"]) for row in production]
    exact = source_names == production_names
    normalized = len(source) == len(production) and [
        sequence_family(name) for name in source_names
    ] == [sequence_family(name) for name in production_names]
    if exact or normalized:
        method = (
            "mhc_anchor_bounded_exact_sequence"
            if exact
            else "mhc_anchor_bounded_normalized_sequence"
        )
        for target, evidence in zip(production, source):
            _assign(
                target,
                evidence.get("selected_node"),
                source=evidence,
                method=method,
                confidence="high" if exact else "medium",
                overwrite=True,
            )
        return "exact" if exact else "normalized"

    for key_fn, method, confidence in (
        (kernel_exact_identity, "mhc_anchor_bounded_unique_exact_identity", "high"),
        (kernel_base, "mhc_anchor_bounded_unique_function_identity", "medium"),
        (sequence_family, "mhc_anchor_bounded_unique_sequence_family", "medium"),
        (schedule_family, "mhc_anchor_bounded_unique_schedule_family", "medium"),
    ):
        index = unique_source_index(source, key_fn)
        for target in production:
            evidence = index.get(key_fn(str(target["kernel_name"])))
            if evidence:
                _assign(
                    target,
                    evidence.get("selected_node"),
                    source=evidence,
                    method=method,
                    confidence=confidence,
                )

    # CUDA Graph may preserve a repeated helper family without preserving the
    # exact surrounding launch schedule (for example copies around a fused
    # projection).  Repeated names are still admissible when the eager and
    # production multiplicities match *inside this exact mHC-bounded
    # occurrence*.  Pairing by local occurrence order is deliberately scoped
    # here; the same helper name is never transferred across layers/modules.
    for key_fn, method in (
        (kernel_exact_identity, "mhc_anchor_bounded_repeated_exact_occurrence"),
        (kernel_base, "mhc_anchor_bounded_repeated_function_occurrence"),
    ):
        source_groups: dict[str, list[dict[str, Any]]] = {}
        production_groups: dict[str, list[dict[str, Any]]] = {}
        for row in source:
            source_groups.setdefault(key_fn(str(row["kernel_name"])), []).append(row)
        for row in production:
            production_groups.setdefault(key_fn(str(row["kernel_name"])), []).append(row)
        for identity, source_group in source_groups.items():
            production_group = production_groups.get(identity) or []
            if (
                len(source_group) < 2
                or len(source_group) != len(production_group)
                or not all(row.get("selected_node") for row in source_group)
            ):
                continue
            for target, evidence in zip(production_group, source_group):
                _assign(
                    target,
                    evidence.get("selected_node"),
                    source=evidence,
                    method=method,
                    confidence="medium",
                )

    # KDA and dense launches preserve operation order across batch-specific
    # GEMM schedule selection.  Requiring equal segment length and stable
    # boundary kernels makes positional transfer stronger than a global name.
    kind = segment_kind(segment_id)
    if kind in {"kda", "dense"} and len(source) == len(production):
        for target, evidence in zip(production, source):
            _assign(
                target,
                evidence.get("selected_node"),
                source=evidence,
                method="mhc_anchor_bounded_stable_sublayer_position",
                confidence="medium",
            )
    return "mismatched"


def _production_direct_node(name: str, kind: str) -> str | None:
    lowered = name.lower()
    base = kernel_base(name)
    node, _confidence = classify_glm53_sglang_node(name, None, [])
    if node:
        return node
    if kind == "kda":
        if "tgvgemmcuteextkernel" in lowered:
            return "linear_attention.qkv_projection"
        if "_causal_conv1d_" in lowered:
            return "linear_attention.qkv_short_conv"
        if "sigmoid_gating_delta_rule" in lowered:
            return "linear_attention.recurrent_update"
    if kind == "dsa":
        if "fast_hadamard_transform_kernel" in lowered:
            return "dsa_attention.index_k_projection"
        if "_act_quant_kernel" in lowered:
            return "dsa_attention.key_pool_compression"
        if "kpool_assemble_softmax_rotate_write_cache" in lowered:
            return "dsa_attention.index_k_cache"
        if "gather_index_k_scale_prefix_into" in lowered:
            return "dsa_attention.index_k_cache"
        if "kpool_build_ragged_layout" in lowered:
            return "dsa_attention.top_pool_selection"
        if "smxx_clean_logits" in lowered:
            return "dsa_attention.index_logits"
        if "paged_mqa_logits" in lowered:
            return "dsa_attention.index_logits"
        if "kpool_topk_transform" in lowered:
            return "dsa_attention.top_pool_selection"
        if "fmhasm" in lowered:
            return "dsa_attention.sparse_mla_core"
        if "set_mla_kv_buffer" in lowered:
            return "dsa_attention.latent_kv_cache"
    if kind == "moe":
        if "routingindices" in lowered:
            return "moe.topk"
        if base.startswith("bmm_e4m3"):
            return "moe.routed_gate_up"
        if "activationdeepseek" in lowered:
            return "moe.routed_activation"
        if base.startswith("bmm_bfloat16"):
            return "moe.routed_down"
        if "finalizekernel" in lowered:
            return "moe.routed_weighted_combine"
    return None


def _assign_kda_production_schedule(segment: list[dict[str, Any]]) -> None:
    """Resolve the unfused KDA prefill helpers by their source-defined order.

    The pinned SGLang implementation executes ``q.contiguous -> l2norm`` and
    ``k.contiguous -> l2norm``, then converts/sigmoids the raw beta and invokes
    ``chunk_kda_fwd``.  Cache-index normalization and chunk-index construction
    are implementation helpers for the persistent state/recurrent update.  The
    kernels involved are generic PyTorch elementwise names, so their local
    position between the causal-conv, L2, gate-cumsum and gated-norm landmarks
    is the semantic evidence; a global name rule would be ambiguous.
    """

    names = [str(row.get("kernel_name") or "").lower() for row in segment]
    causal_index = next(
        (index for index, name in enumerate(names) if "_causal_conv1d_" in name),
        None,
    )
    l2_indices = [
        index for index, name in enumerate(names) if "l2norm_fwd_kernel" in name
    ]
    gate_index = next(
        (index for index, name in enumerate(names) if "kda_gate_chunk_cumsum" in name),
        None,
    )
    gated_norm_index = next(
        (index for index, name in enumerate(names) if "layer_norm_gated" in name),
        None,
    )
    if causal_index is None or len(l2_indices) < 2 or gate_index is None:
        return

    first_l2, second_l2 = l2_indices[:2]
    for index, row in enumerate(segment):
        if row.get("node"):
            continue
        name = names[index]
        node: str | None = None
        method = "mhc_anchor_bounded_kda_source_schedule"

        if "compare_scalar_kernel" in name:
            node = (
                "linear_attention.conv_state"
                if index < causal_index
                else "linear_attention.recurrent_state"
            )
        elif causal_index < index < first_l2 and (
            "where_kernel_impl" in name or "fillfunctor<int>" in name
        ):
            node = "linear_attention.recurrent_state"
        elif (
            index < first_l2 or first_l2 < index < second_l2
        ) and "direct_copy_kernel_cuda" in name:
            node = "linear_attention.qk_l2_norm"
        elif second_l2 < index < gate_index and (
            "sigmoid_kernel_cuda" in name
            or "direct_copy_kernel_cuda" in name
            or "loadwithcast" in name
        ):
            node = "linear_attention.beta_projection"
        elif second_l2 < index < gate_index:
            # prepare_chunk_indices and contiguous layout helpers are not new
            # Model-IR operations; their time belongs to the recurrent-update
            # implementation interval that consumes the generated metadata.
            node = "linear_attention.recurrent_update"
            method = "mhc_anchor_bounded_kda_recurrence_support_schedule"
        elif gate_index < index < (gated_norm_index or len(segment)):
            node = "linear_attention.recurrent_update"
            method = "mhc_anchor_bounded_kda_recurrence_support_schedule"

        if node:
            _assign(row, node, method=method, overwrite=True)


def _assign_dense_production_schedule(segment: list[dict[str, Any]]) -> None:
    """Resolve the two dense DeepGEMM bundles around the SwiGLU landmark."""

    activation_index = next(
        (
            index
            for index, row in enumerate(segment)
            if "silu_mul_clamp" in str(row.get("kernel_name") or "").lower()
        ),
        None,
    )
    if activation_index is None:
        return
    for index, row in enumerate(segment):
        lowered = str(row.get("kernel_name") or "").lower()
        if "silu_mul_clamp" in lowered:
            node = "dense_mlp.clamped_swiglu"
        elif "per_token_group_quant" in lowered or "sm100_fp8_fp4_gemm" in lowered:
            node = (
                "dense_mlp.gate_up_projection"
                if index < activation_index
                else "dense_mlp.down_projection"
            )
        else:
            continue
        _assign(
            row,
            node,
            method="mhc_anchor_bounded_dense_activation_landmark",
            overwrite=True,
        )


def _assign_moe_shared_production_schedule(segment: list[dict[str, Any]]) -> None:
    """Resolve shared-expert DeepGEMM work around its SwiGLU landmark.

    Routed expert GEMMs use the distinct TRT-LLM BMM families and are handled
    separately.  Therefore the quant/DeepGEMM/SwiGLU rows in one MoE mHC
    segment are the always-on shared expert path.
    """

    routing_index = next(
        (
            index
            for index, row in enumerate(segment)
            if "routingindices" in str(row.get("kernel_name") or "").lower()
        ),
        None,
    )
    if routing_index is not None:
        for index, row in enumerate(segment[:routing_index]):
            lowered = str(row.get("kernel_name") or "").lower()
            if "nvjet_sm" in lowered or "splitkreduce_kernel" in lowered:
                _assign(
                    row,
                    "moe.router",
                    method="mhc_anchor_bounded_moe_router_before_routing",
                    overwrite=True,
                )

    activation_index = next(
        (
            index
            for index, row in enumerate(segment)
            if "silu_mul_clamp" in str(row.get("kernel_name") or "").lower()
        ),
        None,
    )
    if activation_index is None:
        return
    for index, row in enumerate(segment):
        lowered = str(row.get("kernel_name") or "").lower()
        if "silu_mul_clamp" in lowered:
            node = "moe.shared_activation"
        elif "per_token_group_quant" in lowered or "sm100_fp8_fp4_gemm" in lowered:
            node = "moe.shared_gate_up" if index < activation_index else "moe.shared_down"
        else:
            continue
        _assign(
            row,
            node,
            method="mhc_anchor_bounded_moe_shared_activation_landmark",
            overwrite=True,
        )

    finalize_index = max(
        (
            index
            for index, row in enumerate(segment)
            if "finalizekernel" in str(row.get("kernel_name") or "").lower()
        ),
        default=-1,
    )
    for row in segment[finalize_index + 1 :] if finalize_index >= 0 else []:
        lowered = str(row.get("kernel_name") or "").lower()
        if "cudafunctor_add" in lowered:
            _assign(
                row,
                "moe.combine",
                method="mhc_anchor_bounded_moe_combine_after_finalize",
                overwrite=True,
            )


def _transfer_collectives(
    production_rows: list[dict[str, Any]], source_rows: list[dict[str, Any]]
) -> int:
    source = [
        row
        for row in source_rows
        if row.get("selected_node") in COLLECTIVE_NODES
    ]
    production = [
        row
        for row in production_rows
        if inferred_collective_kind(str(row["kernel_name"]))
    ]
    source_kinds = [inferred_collective_kind(str(row["kernel_name"])) for row in source]
    production_kinds = [
        inferred_collective_kind(str(row["kernel_name"])) for row in production
    ]
    if source_kinds != production_kinds:
        raise ValueError(
            "production collective order differs from eager semantic contract: "
            f"{Counter(source_kinds)} != {Counter(production_kinds)}"
        )
    for target, evidence in zip(production, source):
        _assign(
            target,
            str(evidence["selected_node"]),
            source=evidence,
            method="complete_eager_collective_kind_and_order",
            overwrite=True,
        )
    return len(production)


def attribute_sglang_production_events(
    production_rows: list[dict[str, Any]], eager_mapping_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_rows = load_enriched_eager_mapping(eager_mapping_path)
    for row in production_rows:
        row.update(
            {
                "node": None,
                "kernel_label": row.get("kernel_name"),
                "attribution_method": "explicit_unmapped_kernel",
                "confidence": "unmapped",
            }
        )

    source_segments = anchor_segments(source_rows)
    production_segments = anchor_segments(production_rows)
    segment_results: Counter[str] = Counter()
    for segment_id, ((s0, s1), (p0, p1)) in enumerate(
        zip(source_segments, production_segments)
    ):
        production_segment = production_rows[p0:p1]
        source_segment = source_rows[s0:s1]
        annotate_segment_scope(source_segment, segment_id)
        annotate_segment_scope(production_segment, segment_id)
        segment_results[
            _bounded_transfer(
                production_segment, source_segment, segment_id=segment_id
            )
        ] += 1
        kind = segment_kind(segment_id)
        if kind == "kda":
            _assign_kda_production_schedule(production_segment)
        elif kind == "dsa":
            _assign_dsa_production_schedule(production_segment)
        elif kind == "dense":
            _assign_dense_production_schedule(production_segment)
        elif kind == "moe":
            _assign_moe_shared_production_schedule(production_segment)
        for row in production_segment:
            node = _production_direct_node(str(row["kernel_name"]), kind)
            if node:
                _assign(
                    row,
                    node,
                    method=f"mhc_anchor_bounded_{kind}_semantic_kernel_rule",
                    overwrite=True,
                )
        if segment_id == SUBLAYER_SEGMENT_COUNT - 1:
            _assign_final_model_tail(production_segment)

    collective_count = _transfer_collectives(production_rows, source_rows)
    _assign_dsa_production_stream_schedules(production_rows)
    _assign_kda_production_stream_schedules(production_rows)
    _assign_outer_model_boundaries(production_rows)

    # Prefix/suffix operations such as embedding, final norm, LM head, and
    # logits are outside the 90 decoder segments.  Exact source uniqueness is
    # safe there because no decoder module boundary is being inferred.
    first_anchor = production_segments[0][0]
    last_segment_stop = production_segments[-1][1]
    outside = production_rows[:first_anchor] + production_rows[last_segment_stop:]
    for key_fn, method, confidence in (
        (kernel_exact_identity, "eager_unique_exact_identity_outside_decoder", "high"),
        (kernel_base, "eager_unique_function_identity_outside_decoder", "medium"),
    ):
        index = unique_source_index(source_rows, key_fn)
        for target in outside:
            evidence = index.get(key_fn(str(target["kernel_name"])))
            if evidence:
                _assign(
                    target,
                    evidence.get("selected_node"),
                    source=evidence,
                    method=method,
                    confidence=confidence,
                )

    _classify_runtime_support(production_rows)

    total_us = sum(float(row.get("dur_us") or 0.0) for row in production_rows)
    mapped = [row for row in production_rows if row.get("node")]
    mapped_us = sum(float(row.get("dur_us") or 0.0) for row in mapped)
    diagnostics = {
        "eager_kernel_count": len(source_rows),
        "production_kernel_count": len(production_rows),
        "anchor_count": SUBLAYER_SEGMENT_COUNT,
        "segment_results": dict(segment_results),
        "collective_count": collective_count,
        "mapped_kernel_count": len(mapped),
        "mapped_kernel_count_ratio": len(mapped) / len(production_rows),
        "mapped_kernel_duration_ratio": mapped_us / total_us,
        "support_class_counts": dict(
            Counter(
                str(row.get("support_class"))
                for row in production_rows
                if row.get("support_class")
            )
        ),
        "method_counts": dict(
            Counter(str(row["attribution_method"]) for row in production_rows)
        ),
    }
    return production_rows, diagnostics
