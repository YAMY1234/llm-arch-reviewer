#!/usr/bin/env python3
"""Deterministic Qwen 4.0 CUDA-Graph decode attribution.

CUDA Graph replay kernels do not carry Python stacks.  The eager trace is still
the source of truth for source/IR binding, while this module transfers that
binding to a replay through invariants that are present in every captured
graph:

* 96 hyper-connection mix and 96 combine kernels delimit the 48 layers;
* the 3-linear/1-QSA layer schedule is fixed by the Model IR;
* topology collectives keep the eager-proven launch order;
* unique GDN, QSA, MoE, DeepEP, and DeepGEMM kernels anchor sub-operations.

The result records the attribution method for every kernel.  Sequence-aligned
events are never presented as direct stack measurements.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import copy
import statistics
from typing import Any, Iterable


LAYER_COUNT = 48
LINEAR_LAYER_COUNT = 36
FULL_LAYER_COUNT = 12


def kernel_name(kernel: dict[str, Any]) -> str:
    return str(kernel.get("name", ""))


def collective_kind(name: str) -> str | None:
    lowered = name.lower()
    if "reducescatter" in lowered or "reduce_scatter" in lowered:
        return "reduce_scatter"
    if "allreduce" in lowered or "all_reduce" in lowered:
        return "all_reduce"
    if "allgather" in lowered or "all_gather" in lowered:
        return "all_gather"
    return None


def direct_kernel_mapping(name: str) -> tuple[str | None, str | None]:
    """Map only kernel signatures whose semantic role is unique."""

    lowered = name.lower()
    rules = (
        (
            "_qwen4_ngram_hash_kernel",
            "ple.ngram_hash",
            "fused Qwen4 PLE N-gram hash",
        ),
        (
            "_qwen4_gate_value_kernel",
            "ple.grouped_norm_gate",
            "fused Qwen4 PLE post-reduction gate + value broadcast",
        ),
        (
            "_qwen4_short_conv_state_kernel",
            "ple.short_conv",
            "fused Qwen4 PLE short-conv state movement",
        ),
        ("fused_qkvzba_split", "linear_attention.split_pack", "GDN split + pack"),
        ("causal_conv1d_update", "linear_attention.causal_conv", "GDN causal Conv1D"),
        ("_causal_conv1d_update", "linear_attention.causal_conv", "GDN causal Conv1D"),
        (
            "fused_recurrent_gated_delta_rule_packed_decode",
            "linear_attention.delta_rule",
            "GDN recurrent delta rule",
        ),
        ("gdn_decode_bf16state", "linear_attention.delta_rule", "FlashInfer GDN recurrence"),
        ("gdn_wide_vec_kernel", "linear_attention.delta_rule", "FlashInfer GDN recurrence"),
        ("fused_gdn_gating", "linear_attention.gating", "GDN beta/decay gating"),
        ("fused_qkv_split_gdn_prefill", "linear_attention.gating", "GDN beta/decay gating"),
        ("_layer_norm_fwd_1pass_kernel", "linear_attention.gated_norm", "GDN gated RMSNorm"),
        ("_hc_mix_persistent_kernel", "hyperconnection.mix", "hyper-connection fused gate + mix"),
        ("hc_combine_kernel", "hyperconnection.combine", "hyper-connection combine"),
        (
            "_fused_qk_rmsnorm_rope_gate_kernel",
            "qsa_attention.qk_norm_rope",
            "QSA Q/K norm + RoPE",
        ),
        ("qsa_index_q_prep_kernel", "qsa_attention.indexer", "QSA index query preparation"),
        ("qsa_index_k_compress_kernel", "qsa_attention.indexer", "QSA index key compression"),
        ("fast_topk_detail::fast_topk_kernel", "qsa_attention.indexer", "QSA index top-k"),
        ("_expand_qsa_block_indices_kernel", "qsa_attention.indexer", "QSA block-index expansion"),
        ("store_kvcache", "qsa_attention.kv_cache", "QSA KV-cache store"),
        ("_qsa_graph_layout_alloc_kernel", "qsa_attention.metadata", "QSA graph metadata"),
        ("_fa2_valid_counts", "qsa_attention.attention_core", "QSA valid-count preparation"),
        ("_compact_kv", "qsa_attention.attention_core", "QSA compact KV"),
        ("fmhasm100fkernel_qkv", "qsa_attention.attention_core", "QSA sparse attention"),
        ("moe::dev::routing", "moe.topk", "MoE top-k routing"),
        ("mask_topk", "moe.topk", "MoE padded-route mask"),
        ("_router_triton_kernel", "moe.router", "MoE router logits"),
        ("deep_gemm::", "moe.routed_experts", "DeepGEMM routed-expert GEMM"),
        ("_silu_and_mul_kernel", "moe.routed_experts", "routed-expert activation"),
        ("act_and_mul_kernel", "moe.shared_expert", "shared-expert activation"),
        ("moe::dev::finalize", "moe.combine", "MoE routed-expert finalize"),
        ("_fused_gate_sigmoid_mul_add_kernel", "moe.combine", "shared/routed combine"),
        ("deep_ep::internode_ll::dispatch", "moe.deepep_dispatch", "DeepEP token dispatch"),
        ("deep_ep::internode_ll::combine", "moe.deepep_combine", "DeepEP token combine"),
    )
    for signature, node, label in rules:
        if signature in lowered:
            return node, label
    return None, None


def _is_gemm(name: str) -> bool:
    lowered = name.lower()
    return any(
        signature in lowered
        for signature in (
            "gemm",
            "nvjet_",
            "bmm_",
            "dot_kernel",
            "reduce_1block_kernel",
            "cublaslt",
        )
    )


def _contains(name: str, *patterns: str) -> bool:
    lowered = name.lower()
    return any(pattern in lowered for pattern in patterns)


def _hc_mix_end(rows: list[dict[str, Any]], start: int, stop: int) -> int:
    """Return the inclusive end of one HC mix implementation.

    Small local batches use ``_hc_mix_persistent_kernel``.  Larger batches in
    the same captured model expand the exact operation into the down/up GEMMs
    and pointwise gate kernels.  The mean/mul/sigmoid kernel materializes the
    mixed hidden state and is the stable semantic endpoint.  The next GEMM is
    already the consumer (attention QKV, MoE router, or LM head), so absorbing
    it into HC mix would silently misattribute large-batch time.
    """

    fused = next(
        (
            index
            for index in range(start, stop)
            if "_hc_mix_persistent_kernel" in rows[index]["kernel_name"].lower()
        ),
        None,
    )
    if fused is not None:
        return fused
    gate = next(
        (
            index
            for index in range(start, stop)
            if _contains(rows[index]["kernel_name"], "mean_mul_sigmoid")
        ),
        None,
    )
    if gate is None:
        raise ValueError("HC mix is missing both fused and expanded anchors")
    return gate


def _assign(
    rows: list[dict[str, Any]],
    index: int,
    node: str,
    label: str,
    method: str,
    confidence: str,
    *,
    overwrite: bool = False,
) -> None:
    row = rows[index]
    if (
        row.get("attribution_method") == "eager_stack_collective_order"
        and method != "eager_stack_collective_order"
    ):
        return
    if row.get("node") is not None and not overwrite:
        return
    row.update(
        {
            "node": node,
            "kernel_label": label,
            "attribution_method": method,
            "confidence": confidence,
        }
    )


def _first_index(
    rows: list[dict[str, Any]], start: int, stop: int, targets: Iterable[str]
) -> int | None:
    wanted = set(targets)
    return next(
        (index for index in range(start, stop) if rows[index].get("node") in wanted),
        None,
    )


def _last_index(
    rows: list[dict[str, Any]], start: int, stop: int, targets: Iterable[str]
) -> int | None:
    wanted = set(targets)
    return next(
        (
            index
            for index in range(stop - 1, start - 1, -1)
            if rows[index].get("node") in wanted
        ),
        None,
    )


def _map_linear_attention(
    rows: list[dict[str, Any]], start: int, stop: int
) -> None:
    split = _first_index(rows, start, stop, ("linear_attention.split_pack",))
    conv = _first_index(rows, start, stop, ("linear_attention.causal_conv",))
    delta = _first_index(rows, start, stop, ("linear_attention.delta_rule",))
    norm = _last_index(rows, start, stop, ("linear_attention.gated_norm",))
    if None in (split, conv, delta, norm):
        raise ValueError("linear-attention segment is missing a decode anchor")
    assert split is not None and conv is not None and delta is not None and norm is not None
    if not start <= split < conv < delta <= norm < stop:
        raise ValueError("linear-attention decode anchors are out of order")

    for index in range(start, stop):
        if rows[index].get("node") is not None:
            continue
        name = rows[index]["kernel_name"]
        if index < split:
            node, label = (
                "linear_attention.qkvz_projection",
                "fused q/k/v/z + beta/alpha projection",
            )
        elif index < conv:
            node, label = "linear_attention.split_pack", "GDN split/reshape support"
        elif index < delta:
            if _is_gemm(name):
                node, label = (
                    "linear_attention.qkvz_projection",
                    "beta/alpha projection (shared projection rollup)",
                )
            else:
                node, label = "linear_attention.gating", "GDN recurrence preparation"
        elif index <= norm:
            node, label = "linear_attention.delta_rule", "GDN recurrence support"
        else:
            node, label = "linear_attention.output_projection", "GDN output projection"
        _assign(rows, index, node, label, "validated_execution_sequence", "high")


def _map_qsa_attention(rows: list[dict[str, Any]], start: int, stop: int) -> None:
    qk = _first_index(rows, start, stop, ("qsa_attention.qk_norm_rope",))
    first_indexer = _first_index(rows, start, stop, ("qsa_attention.indexer",))
    last_indexer = _last_index(rows, start, stop, ("qsa_attention.indexer",))
    last_core = _last_index(rows, start, stop, ("qsa_attention.attention_core",))
    if None in (qk, first_indexer, last_indexer, last_core):
        raise ValueError("QSA segment is missing a decode anchor")
    assert qk is not None and first_indexer is not None and last_indexer is not None
    assert last_core is not None
    indexer_stream = rows[first_indexer].get("stream")
    qk_stream = rows[qk].get("stream")

    output_gemm = next(
        (
            index
            for index in range(stop - 1, last_core, -1)
            if rows[index].get("node") is None and _is_gemm(rows[index]["kernel_name"])
        ),
        None,
    )
    for index in range(start, stop):
        if rows[index].get("node") is not None:
            continue
        name = rows[index]["kernel_name"]
        if index < qk:
            if (
                indexer_stream is not None
                and indexer_stream != qk_stream
                and rows[index].get("stream") == indexer_stream
            ):
                node, label = "qsa_attention.indexer", "QSA index projection/support"
            else:
                node, label = "qsa_attention.qkv_gate_projection", "Q/K/V + output-gate projection"
        elif index <= last_indexer:
            node, label = "qsa_attention.indexer", "QSA index projection/support"
        elif index <= last_core:
            node, label = "qsa_attention.attention_core", "QSA attention preparation"
        elif output_gemm is not None and index >= output_gemm:
            node, label = "qsa_attention.output_projection", "QSA output projection"
        else:
            node, label = "qsa_attention.output_gate", "QSA sigmoid output gate"
        _assign(rows, index, node, label, "validated_execution_sequence", "high")


_QSA_INDEXER_PARENT_NODES = {
    "qsa_attention.indexer",
    "mtp_qsa_attention.indexer",
}


def _direct_qsa_indexer_drill_target(event: dict[str, Any]) -> str | None:
    """Return only semantically unique QSA-indexer drill mappings."""

    name = str(event.get("kernel_name") or "").lower()
    label = str(event.get("kernel_label") or "").lower()
    if "qsa_index_q_prep_kernel" in name or "query preparation" in label:
        return "qsa_indexer.q_norm_rope"
    if "qsa_index_k_compress_kernel" in name or "key compression" in label:
        return "qsa_indexer.compress"
    if "fast_topk" in name or "index top-k" in label:
        return "qsa_indexer.block_topk"
    if "expand_qsa_block_indices" in name or "block-index expansion" in label:
        return "qsa_indexer.expand_tail"
    if "qsa_mqa" in name or "compressed mqa" in label:
        return "qsa_indexer.compressed_score"
    return None


def attach_qsa_indexer_drill_targets(events: list[dict[str, Any]]) -> None:
    """Attach leaf drill targets without changing the validated parent mapping.

    The eager/CUDA-Graph attribution contract already proves the whole QSA
    indexer interval.  This pass refines that interval using direct fused
    anchors plus their order inside one layer invocation.  State/cache nodes
    intentionally remain fused/structural and never receive invented kernel
    residency.
    """

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if str(event.get("node") or "") not in _QSA_INDEXER_PARENT_NODES:
            continue
        direct = _direct_qsa_indexer_drill_target(event)
        if direct is not None:
            event["qsa_indexer_drill_target"] = direct
        groups[
            (
                event.get("node"),
                event.get("step_index"),
                event.get("prefill_chunk_index"),
                event.get("invocation_id"),
                event.get("layer_id"),
                event.get("substage"),
            )
        ].append(event)

    for rows in groups.values():
        ordered = sorted(rows, key=lambda row: float(row.get("ts_us", 0.0)))
        q_preps = [
            index
            for index, row in enumerate(ordered)
            if row.get("qsa_indexer_drill_target") == "qsa_indexer.q_norm_rope"
        ]

        # Without the fused Q-prep anchor there is no safe boundary between
        # the projection and fallback normalization path. Keep ambiguous
        # support on the validated parent instead of forcing a leaf mapping.
        previous_expand = -1
        for invocation_index, q_prep in enumerate(q_preps):
            next_q_prep = (
                q_preps[invocation_index + 1]
                if invocation_index + 1 < len(q_preps)
                else len(ordered)
            )

            def first_after(target: str) -> int | None:
                return next(
                    (
                        index
                        for index in range(q_prep, next_q_prep)
                        if ordered[index].get("qsa_indexer_drill_target") == target
                    ),
                    None,
                )

            compress = first_after("qsa_indexer.compress")
            topk = first_after("qsa_indexer.block_topk")
            expand = first_after("qsa_indexer.expand_tail")
            segment_stop = expand + 1 if expand is not None else next_q_prep
            for index in range(previous_expand + 1, segment_stop):
                row = ordered[index]
                if row.get("qsa_indexer_drill_target") is not None:
                    continue
                if index < q_prep:
                    target = "qsa_indexer.qk_projection"
                elif compress is not None and index < compress:
                    target = "qsa_indexer.q_norm_rope"
                elif topk is not None and index < topk:
                    target = "qsa_indexer.compressed_score"
                elif expand is not None and index < expand:
                    target = "qsa_indexer.block_topk"
                else:
                    target = "qsa_indexer.expand_tail"
                row["qsa_indexer_drill_target"] = target
            previous_expand = expand if expand is not None else segment_stop - 1


def _map_moe(
    rows: list[dict[str, Any]], start: int, stop: int, *, config_name: str
) -> None:
    first_topk = _first_index(rows, start, stop, ("moe.topk",))
    first_dispatch = _first_index(rows, start, stop, ("moe.deepep_dispatch",))
    last_dispatch = _last_index(rows, start, stop, ("moe.deepep_dispatch",))
    first_routed = _first_index(rows, start, stop, ("moe.routed_experts",))
    last_routed = _last_index(rows, start, stop, ("moe.routed_experts",))
    first_combine = _first_index(rows, start, stop, ("moe.deepep_combine",))
    first_model_combine = _first_index(rows, start, stop, ("moe.combine",))
    if first_topk is None:
        raise ValueError("MoE segment is missing its top-k anchor")

    generic_gemms = [
        index
        for index in range(start, first_topk)
        if rows[index].get("node") is None and _is_gemm(rows[index]["kernel_name"])
    ]
    router_gemm = generic_gemms[0] if generic_gemms else None

    for index in range(start, stop):
        if rows[index].get("node") is not None:
            continue
        name = rows[index]["kernel_name"]
        if index == router_gemm or _contains(name, "router"):
            node, label = "moe.router", "MoE router logits"
        elif index < first_topk:
            node, label = "moe.shared_expert", "shared-expert overlapped projection"
        elif config_name != "dp_attention_ep4_deepep_deepgemm" and _contains(
            name, "bmm_"
        ):
            # BMM is not globally unique (attention and LM-head paths can use
            # it), but inside the validated MoE interval it is the routed
            # expert grouped GEMM.  Keep this positional qualification here,
            # never in direct_kernel_mapping().
            node, label = "moe.routed_experts", "routed-expert grouped GEMM"
        elif (
            config_name != "dp_attention_ep4_deepep_deepgemm"
            and first_model_combine is not None
            and index < first_model_combine
        ):
            node, label = "moe.shared_expert", "shared-expert projection/support"
        elif (
            config_name != "dp_attention_ep4_deepep_deepgemm"
            and first_model_combine is not None
        ):
            node, label = "moe.combine", "MoE output combine support"
        elif first_dispatch is not None and index < first_dispatch:
            node, label = "moe.deepep_dispatch", "DeepEP routing-layout preparation"
        elif (
            first_dispatch is not None
            and last_dispatch is not None
            and index <= last_dispatch
        ):
            node, label = "moe.deepep_dispatch", "DeepEP dispatch support"
        elif first_routed is not None and index <= first_routed:
            node, label = "moe.shared_expert", "shared-expert projection"
        elif last_routed is not None and index <= last_routed:
            if config_name == "dp_attention_ep4_deepep_deepgemm":
                node, label = "moe.routed_experts", "DeepGEMM routed-expert support"
            else:
                node, label = "moe.shared_expert", "shared-expert down projection"
        elif first_combine is not None and index < first_combine:
            node, label = "moe.deepep_combine", "DeepEP combine preparation"
        else:
            node, label = "moe.combine", "MoE output combine support"
        _assign(rows, index, node, label, "validated_execution_sequence", "high")


def _map_ple(rows: list[dict[str, Any]], start: int, stop: int) -> None:
    if start >= stop:
        return
    vocab = next(
        (
            index
            for index in range(start, stop)
            if "vocab_parallel_embedding" in rows[index]["kernel_name"].lower()
        ),
        None,
    )
    embed_collective = _first_index(
        rows, start, stop, ("ple.tp_embedding_collective",)
    )
    norm_indices = [
        index
        for index in range(start, stop)
        if "grouped_gemma_rmsnorm_kernel" in rows[index]["kernel_name"].lower()
    ]
    conv = next(
        (
            index
            for index in range(start, stop)
            if _contains(rows[index]["kernel_name"], "conv_depthwise", "short_conv")
        ),
        None,
    )
    if vocab is None or embed_collective is None or not norm_indices or conv is None:
        raise ValueError("PLE decode segment is missing a semantic anchor")
    first_norm = norm_indices[0]
    last_norm = norm_indices[-1]

    for index in range(start, stop):
        if rows[index].get("node") is not None:
            continue
        name = rows[index]["kernel_name"]
        if index < vocab:
            node, label = "ple.ngram_hash", "N-gram context/hash preparation"
        elif index == vocab:
            node, label = "ple.ngram_embedding", "N-gram embedding lookup"
        elif index <= embed_collective:
            node, label = "ple.ngram_embedding", "N-gram embedding lookup support"
        elif index < first_norm:
            if _contains(name, "memcpy"):
                node, label = "ple.dp_ngram_output_scatter", "DP-local N-gram embedding slice"
            else:
                node, label = "ple.key_value_projection", "PLE key/value projection"
        elif index <= last_norm or index < conv:
            node, label = "ple.grouped_norm_gate", "PLE grouped norm + gate"
        elif index <= conv:
            node, label = "ple.short_conv", "PLE depthwise short convolution"
        elif _contains(name, "index", "scatter", "arange", "fill"):
            node, label = "ple.short_conv", "PLE short-conv state update"
        else:
            node, label = "ple.injection", "PLE gated residual injection"
        _assign(rows, index, node, label, "validated_execution_sequence", "high")


def _set_context(
    rows: list[dict[str, Any]],
    start: int,
    stop: int,
    *,
    layer_id: int | None,
    layer_kind: str | None,
    substage: str,
) -> None:
    for index in range(start, stop):
        rows[index]["layer_id"] = layer_id
        rows[index]["layer_kind"] = layer_kind
        rows[index]["substage"] = substage


def map_decode_step(
    *,
    kernels: list[dict[str, Any]],
    config_name: str,
    collective_template: list[tuple[str, str]] | None,
    rank: int,
    step_index: int,
) -> list[dict[str, Any]]:
    """Map a validated CUDA-Graph decode replay to stable IR nodes."""

    rows: list[dict[str, Any]] = []
    for kernel in kernels:
        name = kernel_name(kernel)
        node, label = direct_kernel_mapping(name)
        args = kernel.get("args") or {}
        rows.append(
            {
                "rank": rank,
                "step_index": step_index,
                "kernel_name": name,
                "kernel_label": label,
                "node": node,
                "ts_us": float(kernel.get("ts", 0.0)),
                "dur_us": float(kernel.get("dur", 0.0)),
                "stream": args.get("stream"),
                "device": args.get("device"),
                "correlation": args.get("correlation"),
                "pid": kernel.get("pid"),
                "tid": kernel.get("tid"),
                "layer_id": None,
                "layer_kind": None,
                "substage": None,
                "attribution_method": "direct_signature" if node else None,
                "confidence": "high" if node else None,
            }
        )

    if collective_template is not None:
        collective_indices = [
            index
            for index, row in enumerate(rows)
            if collective_kind(row["kernel_name"]) is not None
        ]
        if len(collective_indices) != len(collective_template):
            raise ValueError(
                "collective template length does not match CUDA Graph replay: "
                f"{len(collective_template)} != {len(collective_indices)}"
            )
        for index, (expected_kind, node) in zip(
            collective_indices, collective_template
        ):
            actual_kind = collective_kind(rows[index]["kernel_name"])
            label = f"{actual_kind.replace('_', ' ')} ({node})"
            _assign(
                rows,
                index,
                node,
                label,
                "eager_stack_collective_order",
                "high",
                overwrite=True,
            )

    combines = [
        index
        for index, row in enumerate(rows)
        if "hc_combine_kernel" in row["kernel_name"].lower()
    ]
    if len(combines) != 2 * LAYER_COUNT:
        raise ValueError(
            "hyper-connection layer delimiters changed: "
            f"combine={len(combines)}"
        )

    previous_mlp_combine = -1
    for layer_id in range(LAYER_COUNT):
        layer_kind = "full" if layer_id % 4 == 3 else "linear"
        attn_combine = combines[2 * layer_id]
        mlp_combine = combines[2 * layer_id + 1]
        if not previous_mlp_combine < attn_combine < mlp_combine:
            raise ValueError(
                f"layer {layer_id} hyper-connection delimiters are out of order"
            )

        pre_attn_start = previous_mlp_combine + 1
        attention_anchor_nodes = (
            ("linear_attention.split_pack",)
            if layer_kind == "linear"
            else ("qsa_attention.qk_norm_rope",)
        )
        attention_anchor = _first_index(
            rows, pre_attn_start, attn_combine, attention_anchor_nodes
        )
        if attention_anchor is None:
            raise ValueError(f"layer {layer_id} is missing its attention anchor")
        branch_norm = next(
            (
                index
                for index in range(attention_anchor - 1, pre_attn_start - 1, -1)
                if "grouped_gemma_rmsnorm_kernel"
                in rows[index]["kernel_name"].lower()
            ),
            None,
        )
        if branch_norm is None:
            raise ValueError(f"layer {layer_id} is missing its attention branch norm")
        attn_mix_end = _hc_mix_end(rows, branch_norm + 1, attention_anchor)
        attention_start = attn_mix_end + 1
        if layer_id == 1 and pre_attn_start < branch_norm:
            _set_context(
                rows,
                pre_attn_start,
                branch_norm,
                layer_id=layer_id,
                layer_kind=layer_kind,
                substage="ple",
            )
            _map_ple(rows, pre_attn_start, branch_norm)
        elif layer_id > 0 and pre_attn_start < branch_norm:
            _set_context(
                rows,
                pre_attn_start,
                branch_norm,
                layer_id=layer_id,
                layer_kind=layer_kind,
                substage="layer_setup",
            )
            for index in range(pre_attn_start, branch_norm):
                setup_node = (
                    "linear_attention.recurrent_state"
                    if layer_kind == "linear"
                    else "qsa_attention.metadata"
                )
                _assign(
                    rows,
                    index,
                    setup_node,
                    "decode attention-state/metadata preparation",
                    "validated_execution_sequence",
                    "medium",
                )

        _set_context(
            rows,
            branch_norm,
            attention_start,
            layer_id=layer_id,
            layer_kind=layer_kind,
            substage="attn_hc_mix",
        )
        _assign(
            rows,
            branch_norm,
            "hyperconnection.branch_norm",
            "attention-branch RMSNorm",
            "validated_execution_sequence",
            "high",
        )
        for index in range(branch_norm + 1, attention_start):
            _assign(
                rows,
                index,
                "hyperconnection.mix",
                "attention-branch HC mix",
                "validated_execution_sequence",
                "high",
            )

        _set_context(
            rows,
            attention_start,
            attn_combine,
            layer_id=layer_id,
            layer_kind=layer_kind,
            substage="attention",
        )
        if layer_kind == "linear":
            _map_linear_attention(rows, attention_start, attn_combine)
        else:
            _map_qsa_attention(rows, attention_start, attn_combine)

        _set_context(
            rows,
            attn_combine,
            attn_combine + 1,
            layer_id=layer_id,
            layer_kind=layer_kind,
            substage="attn_hc_combine",
        )
        mlp_branch_norm = next(
            (
                index
                for index in range(attn_combine + 1, mlp_combine)
                if "grouped_gemma_rmsnorm_kernel"
                in rows[index]["kernel_name"].lower()
            ),
            None,
        )
        if mlp_branch_norm is None:
            raise ValueError(f"layer {layer_id} is missing its MoE branch norm")
        mlp_mix_end = _hc_mix_end(rows, mlp_branch_norm + 1, mlp_combine)
        moe_start = mlp_mix_end + 1
        if attn_combine + 1 < mlp_branch_norm:
            _set_context(
                rows,
                attn_combine + 1,
                mlp_branch_norm,
                layer_id=layer_id,
                layer_kind=layer_kind,
                substage="attention_state",
            )
            state_node = (
                "linear_attention.recurrent_state"
                if layer_kind == "linear"
                else "qsa_attention.metadata"
            )
            for index in range(attn_combine + 1, mlp_branch_norm):
                _assign(
                    rows,
                    index,
                    state_node,
                    "post-attention state/metadata update",
                    "validated_execution_sequence",
                    "medium",
                )
        _set_context(
            rows,
            attn_combine + 1,
            moe_start,
            layer_id=layer_id,
            layer_kind=layer_kind,
            substage="mlp_hc_mix",
        )
        _assign(
            rows,
            mlp_branch_norm,
            "hyperconnection.branch_norm",
            "MoE-branch RMSNorm",
            "validated_execution_sequence",
            "high",
        )
        for index in range(mlp_branch_norm + 1, moe_start):
            _assign(
                rows,
                index,
                "hyperconnection.mix",
                "MoE-branch HC mix",
                "validated_execution_sequence",
                "high",
            )

        _set_context(
            rows,
            moe_start,
            mlp_combine,
            layer_id=layer_id,
            layer_kind=layer_kind,
            substage="moe",
        )
        _map_moe(
            rows,
            moe_start,
            mlp_combine,
            config_name=config_name,
        )
        layer_view = "linear_layer" if layer_kind == "linear" else "full_layer"
        collective_role = (
            "ep_moe_output_collective"
            if config_name == "ep4_a2a_none"
            else "tp_moe_output_collective"
        )
        generic_collective = (
            "moe.ep_output_collective"
            if config_name == "ep4_a2a_none"
            else "moe.tp_output_collective"
        )
        scoped_collective = f"{layer_view}.{collective_role}"
        for index in range(moe_start, mlp_combine):
            if rows[index].get("node") == generic_collective:
                _assign(
                    rows,
                    index,
                    scoped_collective,
                    "MoE output collective boundary",
                    "validated_layer_context",
                    "high",
                    overwrite=True,
                )
        if config_name == "dp_attention":
            tp_output = _last_index(
                rows, moe_start, mlp_combine, (scoped_collective,)
            )
            if tp_output is None:
                raise ValueError(
                    f"layer {layer_id} is missing its TP MoE output collective"
                )
            scatter_node = (
                "linear_layer.dp_moe_output_scatter"
                if layer_kind == "linear"
                else "full_layer.dp_moe_output_scatter"
            )
            for index in range(tp_output + 1, mlp_combine):
                _assign(
                    rows,
                    index,
                    scatter_node,
                    "DP-local MoE output slice",
                    "validated_execution_sequence",
                    "high",
                    overwrite=True,
                )
        _set_context(
            rows,
            mlp_combine,
            mlp_combine + 1,
            layer_id=layer_id,
            layer_kind=layer_kind,
            substage="mlp_hc_combine",
        )
        previous_mlp_combine = mlp_combine

    # Prefix: model-runner metadata, token embedding, PLE history preparation,
    # and the one-time HC expansion before layer 0.
    first_layer_start = next(
        index for index, row in enumerate(rows) if row.get("layer_id") == 0
    )
    vocab = next(
        (
            index
            for index in range(0, first_layer_start)
            if (
                "vocab_parallel_embedding" in rows[index]["kernel_name"].lower()
                or (
                    "indexselect" in rows[index]["kernel_name"].lower()
                    and "bfloat16" in rows[index]["kernel_name"].lower()
                )
                or "vectorized_gather_kernel"
                in rows[index]["kernel_name"].lower()
            )
        ),
        None,
    )
    top_embedding_collective = _first_index(
        rows, 0, first_layer_start, ("top.tp_embedding_collective",)
    )
    trailing_cat = [
        index
        for index in range(0, first_layer_start)
        if _contains(rows[index]["kernel_name"], "catarray", "cat_")
    ]
    hc_expand_start = None
    if trailing_cat and trailing_cat[-1] == first_layer_start - 1:
        hc_expand_start = trailing_cat[-1]
        while (
            hc_expand_start - 1 in trailing_cat
            and hc_expand_start - 1 > (top_embedding_collective or -1)
        ):
            hc_expand_start -= 1
    for index in range(0, first_layer_start):
        if rows[index].get("node") in {
            "qsa_attention.metadata",
            "linear_attention.recurrent_state",
            "top.tp_embedding_collective",
        }:
            continue
        name = rows[index]["kernel_name"]
        if vocab is not None and index == vocab:
            node, label = "top.embedding", "token embedding lookup"
        elif vocab is not None and index < vocab:
            node, label = "top.runtime_support", "decode metadata/cache preparation"
        elif top_embedding_collective is not None and index <= top_embedding_collective:
            node, label = "top.embedding", "token embedding support"
        elif hc_expand_start is not None and index >= hc_expand_start:
            node, label = "stack.hc_expand", "initialize four HC branches"
        else:
            node, label = "ple.token_history", "prepare cached N-gram context"
        _assign(
            rows,
            index,
            node,
            label,
            "validated_execution_sequence",
            "medium",
            overwrite=True,
        )
        rows[index]["substage"] = "model_prefix"

    # Suffix: commit PLE state, final HC mixer, LM head, and logits collectives.
    suffix_start = combines[-1] + 1
    final_norm = next(
        (
            index
            for index in range(len(rows) - 1, suffix_start - 1, -1)
            if "grouped_gemma_rmsnorm_kernel" in rows[index]["kernel_name"].lower()
        ),
        None,
    )
    final_mix_end = (
        _hc_mix_end(rows, final_norm + 1, len(rows))
        if final_norm is not None
        else None
    )
    logits_collective = _last_index(
        rows,
        suffix_start,
        len(rows),
        ("top.tp_logits_collective",),
    )
    logits_input_gather = _last_index(
        rows,
        suffix_start,
        len(rows),
        ("top.dp_logits_input_gather",),
    )
    lm_head_gemm = next(
        (
            index
            for index in range((logits_collective or len(rows)) - 1, suffix_start - 1, -1)
            if _is_gemm(rows[index]["kernel_name"])
        ),
        None,
    )
    for index in range(suffix_start, len(rows)):
        name = rows[index]["kernel_name"]
        if final_norm is not None and final_mix_end is not None and index <= final_mix_end:
            if index < final_norm:
                node, label = "ple.context_commit", "commit N-gram/short-conv state"
            else:
                node, label = "top.final_hc_mix", "final HC norm + mix"
        elif (
            logits_input_gather is not None
            and lm_head_gemm is not None
            and index < lm_head_gemm
        ):
            node, label = (
                "top.dp_logits_input_gather",
                "DP hidden-state gather for TP LM head",
            )
        elif logits_collective is not None and index < logits_collective:
            node, label = "top.lm_head", "vocabulary-sharded LM head"
        elif logits_collective is not None and index == logits_collective:
            # Preserve the eager-proven collective mapping.
            continue
        else:
            if config_name in {
                "dp_attention",
                "dp_attention_ep4_deepep_deepgemm",
            }:
                node, label = (
                    "top.dp_logits_output_scatter",
                    "DP-local logits slice/materialization",
                )
            else:
                node, label = "top.logits", "logits materialization"
        _assign(
            rows,
            index,
            node,
            label,
            "validated_execution_sequence",
            "high" if node != "top.logits" else "medium",
            overwrite=True,
        )
        rows[index]["substage"] = "model_suffix"

    unresolved = [row for row in rows if row.get("node") is None]
    if unresolved:
        names = Counter(row["kernel_name"] for row in unresolved)
        raise ValueError(
            f"decode attribution left {len(unresolved)} kernels unresolved: "
            f"{names.most_common(8)}"
        )
    attach_qsa_indexer_drill_targets(rows)
    return rows


def interval_union_us(events: Iterable[dict[str, Any]]) -> float:
    return interval_union_from_ranges_us(
        (
            float(event["ts_us"]),
            float(event["ts_us"]) + float(event["dur_us"]),
        )
        for event in events
        if float(event["dur_us"]) > 0
    )


def merged_intervals_us(
    intervals: Iterable[tuple[float, float]],
) -> list[tuple[float, float]]:
    ordered = sorted(
        (float(start), float(stop))
        for start, stop in intervals
        if float(stop) > float(start)
    )
    if not ordered:
        return []
    merged: list[tuple[float, float]] = []
    start, end = ordered[0]
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            merged.append((start, end))
            start, end = next_start, next_end
    merged.append((start, end))
    return merged


def interval_union_from_ranges_us(
    intervals: Iterable[tuple[float, float]],
) -> float:
    return sum(stop - start for start, stop in merged_intervals_us(intervals))


def interval_intersection_us(
    left: Iterable[tuple[float, float]],
    right: Iterable[tuple[float, float]],
) -> float:
    left_merged = merged_intervals_us(left)
    right_merged = merged_intervals_us(right)
    total = 0.0
    left_index = right_index = 0
    while left_index < len(left_merged) and right_index < len(right_merged):
        left_start, left_stop = left_merged[left_index]
        right_start, right_stop = right_merged[right_index]
        total += max(0.0, min(left_stop, right_stop) - max(left_start, right_start))
        if left_stop <= right_stop:
            left_index += 1
        else:
            right_index += 1
    return total


def _event_ranges(events: Iterable[dict[str, Any]]) -> list[tuple[float, float]]:
    return [
        (float(event["ts_us"]), float(event["ts_us"]) + float(event["dur_us"]))
        for event in events
        if float(event["dur_us"]) > 0
    ]


def _invocation_key(event: dict[str, Any]) -> str | int | None:
    if event.get("invocation_id") is not None:
        return str(event["invocation_id"])
    if event.get("layer_id") is not None:
        return int(event["layer_id"])
    return None


def _elapsed_metrics(
    events: list[dict[str, Any]],
    *,
    all_events: list[dict[str, Any]],
    n_iters: int,
    elapsed_scope: str,
) -> dict[str, Any]:
    """Measure invocation envelopes without confusing their gaps with GPU idle.

    ``elapsed_scope=invocation`` unions one envelope per validated layer/chunk
    invocation. ``elapsed_scope=step`` uses one envelope for the whole selected
    module in each step. Within those envelopes, global GPU activity is split
    into this module's active intervals, other GPU work, and true device gaps.
    """

    if elapsed_scope not in {"invocation", "step"}:
        raise ValueError(f"unknown elapsed scope: {elapsed_scope}")
    if elapsed_scope == "invocation" and any(
        _invocation_key(event) is None for event in events
    ):
        return {}

    elapsed_total = 0.0
    active_total = 0.0
    global_active_total = 0.0
    for step_index in range(1, n_iters + 1):
        step_events = [
            event
            for event in events
            if int(event.get("step_index", 0)) == step_index
        ]
        if not step_events:
            continue
        if elapsed_scope == "step":
            grouped = {"step": step_events}
        else:
            grouped: dict[str | int, list[dict[str, Any]]] = defaultdict(list)
            for event in step_events:
                key = _invocation_key(event)
                assert key is not None
                grouped[key].append(event)
        envelopes = merged_intervals_us(
            (
                min(float(event["ts_us"]) for event in invocation),
                max(
                    float(event["ts_us"]) + float(event["dur_us"])
                    for event in invocation
                ),
            )
            for invocation in grouped.values()
        )
        target_ranges = _event_ranges(step_events)
        global_ranges = _event_ranges(
            event
            for event in all_events
            if int(event.get("step_index", 0)) == step_index
        )
        elapsed_total += interval_union_from_ranges_us(envelopes)
        active_total += interval_union_from_ranges_us(target_ranges)
        global_active_total += interval_intersection_us(envelopes, global_ranges)

    elapsed_us = elapsed_total / n_iters
    active_us = active_total / n_iters
    global_active_us = global_active_total / n_iters
    other_gpu_us = max(0.0, global_active_us - active_us)
    device_idle_us = max(0.0, elapsed_us - global_active_us)
    module_gap_us = max(0.0, elapsed_us - active_us)
    # Keep the displayed decomposition numerically closed after floating-point
    # interval clipping.
    if abs(module_gap_us - other_gpu_us - device_idle_us) < 1e-6:
        device_idle_us = max(0.0, module_gap_us - other_gpu_us)
    return {
        "gpu_elapsed_ms": round(elapsed_us / 1000.0, 6),
        "module_gap_ms": round(module_gap_us / 1000.0, 6),
        "other_gpu_work_ms": round(other_gpu_us / 1000.0, 6),
        "device_idle_ms": round(device_idle_us / 1000.0, 6),
        "module_active_pct": round(100.0 * active_us / elapsed_us, 2)
        if elapsed_us
        else 0.0,
        "device_busy_pct": round(100.0 * global_active_us / elapsed_us, 2)
        if elapsed_us
        else 0.0,
        "elapsed_scope": (
            "union of validated invocation envelopes"
            if elapsed_scope == "invocation"
            else "selected module envelope within each profiled step"
        ),
        "gap_semantics": (
            "module gap = other GPU work + device idle inside the invocation "
            "envelope; device-idle cause is unclassified"
        ),
    }


def _metric(
    events: list[dict[str, Any]],
    *,
    n_iters: int,
    metric_kind: str,
    aggregation: str,
    all_events: list[dict[str, Any]] | None = None,
    elapsed_scope: str | None = None,
) -> dict[str, Any]:
    residency_us = sum(float(event["dur_us"]) for event in events) / n_iters
    per_step_union = []
    for step_index in sorted({int(event["step_index"]) for event in events}):
        per_step_union.append(
            interval_union_us(
                event for event in events if int(event["step_index"]) == step_index
            )
        )
    # Average over the full selected window, including a zero for an iteration
    # where this node did not launch. Averaging only the iterations present in
    # ``events`` inflates conditional/occasional kernels and can make active
    # time exceed their invocation-envelope average.
    active_us = sum(per_step_union) / n_iters if per_step_union else 0.0
    method_us: Counter[str] = Counter()
    label_us: Counter[str] = Counter()
    label_count: Counter[str] = Counter()
    for event in events:
        method_us[str(event["attribution_method"])] += float(event["dur_us"])
        label = str(event.get("kernel_label") or event["kernel_name"][:120])
        label_us[label] += float(event["dur_us"])
        label_count[label] += 1
    direct_us = method_us.get("direct_signature", 0.0) + method_us.get(
        "eager_stack_collective_order", 0.0
    )
    direct_us += sum(
        duration_us
        for method, duration_us in method_us.items()
        if "python_stack" in method
    )
    total_us = sum(method_us.values())
    if metric_kind == "inclusive_rollup":
        status = "inclusive_rollup"
    elif direct_us == total_us:
        status = "measured_direct"
    elif direct_us > 0:
        status = "measured_mixed"
    else:
        status = "measured_aligned"
    result = {
        "ms_per_iter": round(active_us / 1000.0, 6),
        "active_gpu_ms": round(active_us / 1000.0, 6),
        "gpu_residency_ms": round(residency_us / 1000.0, 6),
        "gpu_overlap_ms": round(max(0.0, residency_us - active_us) / 1000.0, 6),
        "metric_kind": metric_kind,
        "aggregation": aggregation,
        "attribution_status": status,
        "attribution": {
            "direct_duration_pct": round(100.0 * direct_us / total_us, 2)
            if total_us
            else 0.0,
            "methods": {
                method: {
                    "kernel_count": sum(
                        1
                        for event in events
                        if event["attribution_method"] == method
                    ),
                    "gpu_residency_ms_per_iter": round(
                        duration_us / n_iters / 1000.0, 6
                    ),
                }
                for method, duration_us in sorted(method_us.items())
            },
        },
        "kernels": [
            {
                "name": label,
                "count": label_count[label],
                "count_per_iter": round(label_count[label] / n_iters, 3),
                "avg_us": round(duration_us / label_count[label], 3),
                "total_us_per_iter": round(duration_us / n_iters, 3),
                "share_in_node_pct": round(100.0 * duration_us / total_us, 2),
            }
            for label, duration_us in label_us.most_common(12)
        ],
    }
    if all_events is not None and elapsed_scope is not None:
        result.update(
            _elapsed_metrics(
                events,
                all_events=all_events,
                n_iters=n_iters,
                elapsed_scope=elapsed_scope,
            )
        )
    return result


def communication_semantics(
    target: str, events: list[dict[str, Any]], *, n_iters: int = 1
) -> dict[str, Any] | None:
    """Describe the measured communication operation and its payload."""

    kinds = Counter(
        kind
        for event in events
        if (kind := collective_kind(str(event.get("kernel_name", "")))) is not None
    )
    if "deepep_dispatch" in target or "deepep_combine" in target:
        kinds["all_to_all"] += sum(
            "deep_ep::" in str(event.get("kernel_name", "")).lower()
            for event in events
        )
    if "scatter" in target and not kinds:
        kinds["local_scatter"] = len(events)
    payloads = {
        "top.tp_embedding_collective": "bf16 token embeddings [B,T,H]",
        "top.dp_logits_input_gather": "bf16 DP-local final hidden states → TP LM-head input",
        "top.tp_logits_collective": "bf16 vocabulary-logit shards [B,T,V/TP]",
        "top.dp_logits_output_scatter": "fp32 logits → owning DP request ranks",
        "ple.dp_ngram_input_gather": "int64 N-gram IDs + cached-context indices",
        "ple.tp_embedding_collective": "bf16 vocabulary-sharded N-gram embeddings",
        "ple.dp_ngram_output_scatter": "bf16 N-gram embeddings → local DP token slice",
        "linear_layer.tp_attention_collective": "bf16 linear-attention output [B,T,H]",
        "full_layer.tp_attention_collective": "bf16 QSA attention output [B,T,H]",
        "linear_layer.dp_moe_input_gather": "bf16 DP-local hidden tokens → global TP MoE batch",
        "full_layer.dp_moe_input_gather": "bf16 DP-local hidden tokens → global TP MoE batch",
        "linear_layer.dp_moe_output_scatter": "bf16 TP MoE output → local DP token slice",
        "full_layer.dp_moe_output_scatter": "bf16 TP MoE output → local DP token slice",
        "linear_layer.tp_moe_output_collective": "bf16 TP-sharded routed + shared expert output",
        "full_layer.tp_moe_output_collective": "bf16 TP-sharded routed + shared expert output",
        "mtp_head.tp_embedding_collective": "bf16 MTP token embeddings [B,D,H]",
        "mtp_head.tp_logits_collective": "bf16 MTP vocabulary-logit shards [B,D,V/TP]",
        "mtp_layer.tp_attention_collective": "bf16 MTP QSA output [B,D,H]",
        "mtp_layer.tp_moe_output_collective": "bf16 TP-sharded MTP routed + shared expert output",
        "linear_layer.ep_moe_output_collective": "bf16 EP-sharded routed + shared expert output",
        "full_layer.ep_moe_output_collective": "bf16 EP-sharded routed + shared expert output",
        "moe.deepep_dispatch": (
            "bf16 routed token activations + expert IDs/top-k weights/routing metadata"
        ),
        "moe.deepep_combine": (
            "weighted bf16 expert outputs + routing handles → source DP rank"
        ),
    }
    payload = payloads.get(target)
    # Roll-up nodes can contain many communication kernels from unrelated
    # children.  Labelling the whole decoder/layer as one observed collective
    # is misleading, so communication semantics live only on explicit data-
    # movement IR nodes with a defined payload.
    if payload is None:
        return None
    return {
        "observed_collectives": [
            {
                "kind": kind,
                "kernel_count": count,
                "kernel_count_per_iter": round(count / n_iters, 3),
            }
            for kind, count in sorted(kinds.items())
        ],
        "payload": payload,
        "scope": "captured profile phase and selected reference rank",
    }


def metrics_for_rank(
    events: list[dict[str, Any]], n_iters: int
) -> dict[str, dict[str, Any]]:
    """Build exclusive leaf metrics and overlap-aware inclusive rollups."""

    attach_qsa_indexer_drill_targets(events)

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        groups[str(event["node"])].append(event)

        layer_kind = event.get("layer_kind")
        substage = event.get("substage")
        if layer_kind in {"linear", "full"}:
            layer_view = "linear_layer" if layer_kind == "linear" else "full_layer"
            groups[f"stack.{layer_view}"].append(event)
            if substage == "attn_hc_mix":
                groups[f"{layer_view}.attn_hc_mix"].append(event)
            elif substage in {"attention", "attention_state"}:
                parent = "linear_attention" if layer_kind == "linear" else "qsa_attention"
                groups[f"{layer_view}.{parent}"].append(event)
            elif substage == "attn_hc_combine":
                groups[f"{layer_view}.attn_hc_combine"].append(event)
            elif substage == "mlp_hc_mix":
                groups[f"{layer_view}.mlp_hc_mix"].append(event)
            elif substage == "moe" and str(event.get("node", "")).startswith("moe."):
                groups[f"{layer_view}.moe"].append(event)
            elif substage == "mlp_hc_combine":
                groups[f"{layer_view}.mlp_hc_combine"].append(event)
        if substage == "ple":
            groups["stack.ple_injection"].append(event)

    stack_events = [
        event
        for event in events
        if event.get("layer_id") is not None
        or (
            event.get("substage") == "model_prefix"
            and not str(event.get("node", "")).startswith("top.")
        )
    ]
    groups["top.decoder_stack"].extend(stack_events)

    metrics: dict[str, dict[str, Any]] = {}
    leaf_nodes = {str(event["node"]) for event in events}
    for target, target_events in sorted(groups.items()):
        elapsed_scope = None
        if target == "top.decoder_stack":
            elapsed_scope = "step"
        elif target_events and all(
            _invocation_key(event) is not None for event in target_events
        ):
            elapsed_scope = "invocation"
        metrics[target] = _metric(
            target_events,
            n_iters=n_iters,
            metric_kind="exclusive_leaf" if target in leaf_nodes else "inclusive_rollup",
            aggregation=(
                "interval union on one reference rank"
                if target not in leaf_nodes
                else "kernel interval union on one reference rank"
            ),
            all_events=events,
            elapsed_scope=elapsed_scope,
        )
        communication = communication_semantics(
            target, target_events, n_iters=n_iters
        )
        if communication is not None:
            metrics[target]["communication"] = communication
    attach_hyperconnection_drill_metrics(
        metrics, groups, n_iters=n_iters, all_events=events
    )
    attach_qsa_indexer_drill_metrics(
        metrics, events, n_iters=n_iters, all_events=events
    )
    return metrics


def attach_hyperconnection_drill_metrics(
    metrics: dict[str, dict[str, Any]],
    groups: dict[str, list[dict[str, Any]]],
    *,
    n_iters: int,
    all_events: list[dict[str, Any]] | None = None,
) -> None:
    """Attach context-scoped metrics to the mix and combine IR subgraphs.

    The global ``hyperconnection.*`` leaves aggregate every layer call.  A
    layer-stage drill must instead show one of two different data flows:
    branch states -> selected Attention/MoE input for ``mix``; preserved branch
    states + processed Attention/MoE output -> updated branch states for
    ``combine``.  The scoped child maps preserve that origin and its timings.
    """

    for layer_view in ("linear_layer", "full_layer"):
        for stage in (
            "attn_hc_mix",
            "attn_hc_combine",
            "mlp_hc_mix",
            "mlp_hc_combine",
        ):
            target = f"{layer_view}.{stage}"
            target_events = groups.get(target) or []
            if not target_events or target not in metrics:
                continue
            is_mix = stage.endswith("_mix")
            module_name = "attention" if stage.startswith("attn_") else "MoE"

            def measured_child(child_id: str, node: str) -> dict[str, Any]:
                child_events = [
                    event
                    for event in target_events
                    if str(event.get("node")) == node
                ]
                if not child_events:
                    raise ValueError(
                        f"{target} has no {node} events for its scoped drill view"
                    )
                child = _metric(
                    child_events,
                    n_iters=n_iters,
                    metric_kind="exclusive_leaf",
                    aggregation=(
                        "kernel interval union inside the selected "
                        "layer-stage drill context"
                    ),
                    all_events=all_events or target_events,
                    elapsed_scope=(
                        "invocation"
                        if all(
                            _invocation_key(event) is not None
                            for event in child_events
                        )
                        else None
                    ),
                )
                child["scope_target"] = target
                return child

            if is_mix:
                branch_norm = measured_child(
                    "branch_norm", "hyperconnection.branch_norm"
                )
                mix = measured_child("mix", "hyperconnection.mix")
                mix["display_label"] = (
                    f"weighted branch mix\n4 branches → {module_name} input"
                )
                scoped: dict[str, dict[str, Any]] = {
                    "branch_states": {
                        "status": "structural",
                        "label": "branch-state input boundary",
                        "display_label": "current branch states\n[B,T,4,H]",
                        "scope_target": target,
                    },
                    "branch_norm": branch_norm,
                    "low_rank_gate": {
                        "status": "fused",
                        "label": "fused into the selected hyper-connection mix kernels",
                        "included_in": "hyperconnection.mix",
                        "scope_target": target,
                    },
                    "mix": mix,
                    "module_input": {
                        "status": "structural",
                        "label": "selected module-input boundary",
                        "display_label": f"{module_name} input\n[B,T,H]",
                        "scope_target": target,
                    },
                }
                drill_view = "hyperconnection_mix"
            else:
                combine = measured_child("combine", "hyperconnection.combine")
                combine["display_label"] = (
                    f"update four branches\nwith {module_name} output"
                )
                scoped = {
                    "branch_states": {
                        "status": "structural",
                        "label": "preserved branch-state input boundary",
                        "display_label": "preserved branch states\n[B,T,4,H]",
                        "scope_target": target,
                    },
                    "module_output": {
                        "status": "structural",
                        "label": "processed module-output boundary",
                        "display_label": f"{module_name} output\n[B,T,H]",
                        "scope_target": target,
                    },
                    "combine": combine,
                    "updated_branch_states": {
                        "status": "structural",
                        "label": "updated branch-state output boundary",
                        "display_label": "updated branch states\n[B,T,4,H]",
                        "scope_target": target,
                    },
                }
                drill_view = "hyperconnection_combine"

            metrics[target]["drill_view"] = drill_view
            metrics[target]["drill_scope"] = target
            metrics[target]["drill_metrics"] = scoped

    # Keep the split views meaningful when opened directly.  Their unscoped
    # cells intentionally show the all-call aggregate; drilling from a layer
    # stage replaces them with the scoped cells above.
    for split_target, aggregate_target in (
        ("hyperconnection_mix.branch_norm", "hyperconnection.branch_norm"),
        ("hyperconnection_mix.mix", "hyperconnection.mix"),
        ("hyperconnection_combine.combine", "hyperconnection.combine"),
    ):
        if aggregate_target in metrics:
            metrics[split_target] = copy.deepcopy(metrics[aggregate_target])


def attach_qsa_indexer_drill_metrics(
    metrics: dict[str, dict[str, Any]],
    events: list[dict[str, Any]],
    *,
    n_iters: int,
    all_events: list[dict[str, Any]] | None = None,
) -> None:
    """Attach context-scoped runtime evidence to the QSA indexer drill view."""

    attach_qsa_indexer_drill_targets(events)
    all_events = events if all_events is None else all_events

    for parent in sorted(_QSA_INDEXER_PARENT_NODES):
        parent_rows = [event for event in events if str(event.get("node")) == parent]
        if not parent_rows or parent not in metrics:
            continue
        by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for event in parent_rows:
            target = event.get("qsa_indexer_drill_target")
            if target:
                by_target[str(target)].append(event)

        def measured(target: str, label: str) -> dict[str, Any]:
            selected = by_target.get(target) or []
            if not selected:
                return {
                    "status": "not_observed",
                    "label": f"{label} · no separately validated interval in this profile",
                    "scope_target": parent,
                }
            cell = _metric(
                selected,
                n_iters=n_iters,
                metric_kind="exclusive_leaf",
                aggregation="kernel interval union inside the selected QSA-indexer scope",
                all_events=all_events,
                elapsed_scope=(
                    "invocation"
                    if all(event.get("invocation_id") is not None for event in selected)
                    else None
                ),
            )
            cell["display_label"] = label
            cell["scope_target"] = parent
            return cell

        q_norm = measured(
            "qsa_indexer.q_norm_rope", "index Q RMSNorm + MRoPE + raw-K store"
        )
        compress = measured(
            "qsa_indexer.compress", "4-token average + K norm/MRoPE + compressed-K store"
        )
        mapped_residency_us = sum(
            float(event.get("dur_us", 0.0))
            for event in parent_rows
            if event.get("qsa_indexer_drill_target")
        )
        total_residency_us = sum(float(event.get("dur_us", 0.0)) for event in parent_rows)
        coverage = (
            100.0 * mapped_residency_us / total_residency_us
            if total_residency_us
            else 100.0
        )
        metrics[parent]["drill_view"] = "qsa_indexer"
        metrics[parent]["drill_scope"] = parent
        metrics[parent]["drill_mapping_coverage_pct"] = round(coverage, 2)
        metrics[parent]["drill_metrics"] = {
            "index_in": {
                "status": "structural",
                "label": "hidden-state input boundary · no standalone kernel",
                "scope_target": parent,
            },
            "qk_projection": measured(
                "qsa_indexer.qk_projection", "index Q/K projection"
            ),
            "q_norm_rope": q_norm,
            "raw_k_cache": {
                "status": "fused",
                "label": "raw index-K and MRoPE-position stores are fused into Q-prep",
                "included_in": "qsa_indexer.q_norm_rope",
                "scope_target": parent,
            },
            "compress": compress,
            "compressed_k_cache": {
                "status": "fused",
                "label": "compressed-K cache store is fused into key compression",
                "included_in": "qsa_indexer.compress",
                "scope_target": parent,
            },
            "compressed_score": measured(
                "qsa_indexer.compressed_score", "compressed MQA score"
            ),
            "block_topk": measured(
                "qsa_indexer.block_topk", "Top-512 complete blocks"
            ),
            "expand_tail": measured(
                "qsa_indexer.expand_tail", "block expansion + causal tail"
            ),
            "selected_indices": {
                "status": "structural",
                "label": "selected-token output boundary · materialized by expansion",
                "scope_target": parent,
            },
        }


def default_node_states(*, phase: str) -> dict[str, dict[str, str]]:
    """Explicit non-timing semantics so the viewer never renders a blank cell."""

    states = {
        "top.token_ids": {"status": "structural", "label": "input tensor · no GPU kernel"},
        "top.logits": {"status": "structural", "label": "output tensor/materialization"},
        "stack.stack_in": {"status": "structural", "label": "tensor boundary"},
        "stack.schedule": {"status": "structural", "label": "control schedule · no kernel"},
        "stack.stack_out": {"status": "structural", "label": "tensor boundary"},
        "linear_layer.layer_in": {"status": "structural", "label": "tensor boundary"},
        "linear_layer.layer_out": {"status": "structural", "label": "tensor boundary"},
        "full_layer.layer_in": {"status": "structural", "label": "tensor boundary"},
        "full_layer.layer_out": {"status": "structural", "label": "tensor boundary"},
        "ple.token_history": {"status": "state", "label": "state access; timed support shown when present"},
        "ple.conv_state": {"status": "state", "label": "persistent state · no standalone kernel"},
        "linear_attention.attn_in": {"status": "structural", "label": "tensor boundary"},
        "linear_attention.recurrent_state": {"status": "state", "label": "persistent state · updated by recurrence"},
        "linear_attention.attn_out": {"status": "structural", "label": "tensor boundary"},
        "linear_attention.ba_projection": {
            "status": "fused",
            "label": "fused with q/k/v/z + beta/alpha projection",
            "included_in": "linear_attention.qkvz_projection",
        },
        "hyperconnection.hc_in": {"status": "structural", "label": "tensor boundary"},
        "hyperconnection.low_rank_gate": {
            "status": "fused",
            "label": "fused into hyper-connection mix kernel",
            "included_in": "hyperconnection.mix",
        },
        "hyperconnection.hc_out": {"status": "structural", "label": "tensor boundary"},
        "hyperconnection_mix.branch_states": {
            "status": "structural",
            "label": "branch-state input boundary",
        },
        "hyperconnection_mix.low_rank_gate": {
            "status": "fused",
            "label": "fused into hyper-connection mix kernel",
            "included_in": "hyperconnection_mix.mix",
        },
        "hyperconnection_mix.module_input": {
            "status": "structural",
            "label": "selected module-input boundary",
        },
        "hyperconnection_combine.branch_states": {
            "status": "structural",
            "label": "preserved branch-state input boundary",
        },
        "hyperconnection_combine.module_output": {
            "status": "structural",
            "label": "processed module-output boundary",
        },
        "hyperconnection_combine.updated_branch_states": {
            "status": "structural",
            "label": "updated branch-state output boundary",
        },
        "qsa_attention.attn_in": {"status": "structural", "label": "tensor boundary"},
        "qsa_attention.attn_out": {"status": "structural", "label": "tensor boundary"},
        "moe.moe_in": {"status": "structural", "label": "tensor boundary"},
        "moe.moe_out": {"status": "structural", "label": "tensor boundary"},
    }
    if phase == "decode":
        states["linear_attention.gating"] = {
            "status": "fused",
            "label": "included in fused decode recurrence when no separate kernel",
            "included_in": "linear_attention.delta_rule",
        }
    return states
