#!/usr/bin/env python3
"""Fail-closed pure-TP8 production attribution for Qwen3.5.

The stable execution invariant is the ordered TP collective contract: one
embedding all-reduce followed by attention/MoE all-reduce pairs for exactly 60
decoder layers.  Each layer is therefore a bounded semantic occurrence even
when CUDA Graph replay omits Python stacks.  Kernel-family rules are applied
only inside that occurrence and are cross-checked against eager evidence by
the profile builder.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable


LAYER_COUNT = 60
LAYER_PATTERN = tuple(
    "full" if layer_id % 4 == 3 else "gdn" for layer_id in range(LAYER_COUNT)
)


def _lower(row: dict[str, Any]) -> str:
    return str(row.get("kernel_name") or "").lower()


def _nonzero_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed else None


def semantic_execution_order(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Recover CUDA Graph launch order without changing timeline timestamps.

    Multi-stream kernels complete out of launch order.  For a single replayed
    graph, Kineto retains the exact graph-node identifiers, which are the
    correct portable occurrence order for collective-bounded reconciliation.
    Graph-id-zero setup/output kernels remain ordered by their GPU timestamps.
    Piecewise graphs are intentionally not globally reordered because node
    identifiers from distinct graphs do not define a cross-graph sequence.
    """

    graph_rows = [row for row in rows if _nonzero_int(row.get("graph_id"))]
    graph_ids = {_nonzero_int(row.get("graph_id")) for row in graph_rows}
    if len(graph_ids) != 1 or any(
        _nonzero_int(row.get("graph_node_id")) is None for row in graph_rows
    ):
        return sorted(rows, key=lambda row: float(row["ts_us"]))
    graph_start = min(float(row["ts_us"]) for row in graph_rows)
    graph_row_ids = {id(row) for row in graph_rows}
    before = [
        row
        for row in rows
        if id(row) not in graph_row_ids and float(row["ts_us"]) < graph_start
    ]
    before_row_ids = {id(row) for row in before}
    after = [
        row
        for row in rows
        if id(row) not in graph_row_ids and id(row) not in before_row_ids
    ]
    return [
        *sorted(before, key=lambda row: float(row["ts_us"])),
        *sorted(graph_rows, key=lambda row: int(row["graph_node_id"])),
        *sorted(after, key=lambda row: float(row["ts_us"])),
    ]


def is_all_reduce(row: dict[str, Any]) -> bool:
    name = _lower(row)
    return any(
        token in name
        for token in (
            "allreduce",
            "all_reduce",
            "multimem_all_reduce",
        )
    )


def is_all_reduce_companion(row: dict[str, Any]) -> bool:
    """Return whether a kernel is the second half of a logical TP reduction."""

    return "rmsnormlamport" in _lower(row)


def is_all_gather(row: dict[str, Any]) -> bool:
    name = _lower(row)
    return "allgather" in name or "all_gather" in name


def _all_reduce_groups(rows: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """Return inclusive index ranges for logical TP all-reduces.

    FlashInfer changes the physical realization with shape: a small collective
    is one fused kernel, while larger shapes launch adjacent twoshot and
    RMSNorm/Lamport kernels.  Those kernels are one logical collective and one
    timing owner.  Grouping only adjacent all-reduce kernels preserves the
    ordered 1 + 2*60 execution contract without collapsing their raw events.
    """

    groups: list[tuple[int, int]] = []
    start: int | None = None
    for index, row in enumerate(rows):
        if is_all_reduce(row):
            if start is None:
                start = index
        elif start is not None:
            groups.append((start, index - 1))
            start = None
    if start is not None:
        groups.append((start, len(rows) - 1))
    return groups


def _trailing_all_reduce_start(rows: list[dict[str, Any]]) -> int:
    start = len(rows)
    while start and is_all_reduce(rows[start - 1]):
        start -= 1
    if start == len(rows):
        raise ValueError("semantic segment is missing its trailing TP all-reduce")
    return start


def _assign_collective_group(
    rows: list[dict[str, Any]],
    node: str,
    *,
    ir_targets: Iterable[str] = (),
) -> None:
    for row in rows:
        _assign(
            row,
            node,
            method="complete_eager_validated_tp_collective_order",
            ir_targets=ir_targets,
        )


def _assign(
    row: dict[str, Any],
    node: str,
    *,
    method: str,
    ir_targets: Iterable[str] = (),
    confidence: str = "high",
) -> None:
    row.update(
        {
            "node": node,
            "kernel_label": node,
            "attribution_method": method,
            "confidence": confidence,
            "ir_targets": list(dict.fromkeys((node, *ir_targets))),
        }
    )


def _support(row: dict[str, Any], support_class: str, reason: str) -> None:
    row.update(
        {
            "node": None,
            "kernel_label": row.get("kernel_name"),
            "attribution_method": "explicit_runtime_support_classification",
            "confidence": "support",
            "support_class": support_class,
            "support_reason": reason,
            "ir_targets": [],
        }
    )


def _layer_targets(layer_id: int, layer_kind: str, substage: str) -> tuple[str, ...]:
    block = "gdn_moe_block" if layer_kind == "gdn" else "full_attention_moe_block"
    layer_node = "stack.gdn_layer" if layer_kind == "gdn" else "stack.full_attention_layer"
    scope = f"{block}.attention" if substage == "attention" else f"{block}.moe"
    return (
        scope,
        layer_node,
        f"layer_schedule.layer_{layer_id:02d}",
        "top.decoder_stack",
    )


def _block(layer_kind: str) -> str:
    return "gdn_moe_block" if layer_kind == "gdn" else "full_attention_moe_block"


def _previous_block(layer_id: int) -> str | None:
    return _block(LAYER_PATTERN[layer_id - 1]) if layer_id > 0 else None


def _next_block(layer_id: int) -> str | None:
    return _block(LAYER_PATTERN[layer_id + 1]) if layer_id + 1 < LAYER_COUNT else None


def _is_norm_kernel(row: dict[str, Any]) -> bool:
    name = _lower(row)
    return any(token in name for token in ("rmsnorm", "rms_norm", "layer_norm"))


def _map_input_norm(
    row: dict[str, Any], *, layer_id: int, kind: str, framework: str, phase: str
) -> None:
    previous = _previous_block(layer_id)
    targets = (f"{previous}.layer_residual",) if phase == "prefill" and previous else ()
    _assign(
        row,
        f"{_block(kind)}.input_norm",
        method=f"{framework}_{phase}_eager_validated_layer_input_norm",
        ir_targets=targets,
    )


def _map_post_attention_norm(
    row: dict[str, Any], *, kind: str, framework: str, phase: str
) -> None:
    block = _block(kind)
    _assign(
        row,
        f"{block}.post_attention_norm",
        method=f"{framework}_{phase}_eager_validated_post_attention_norm",
        ir_targets=(f"{block}.attention_residual",),
    )


def _annotate_occurrence(
    rows: list[dict[str, Any]], layer_id: int, layer_kind: str, substage: str
) -> None:
    parents = _layer_targets(layer_id, layer_kind, substage)
    for row in rows:
        row.update(
            {
                "layer_id": layer_id,
                "layer_kind": layer_kind,
                "substage": substage,
                "segment_id": layer_id * 2 + (substage == "moe"),
                "occurrence_id": f"layer_{layer_id:02d}.{substage}",
            }
        )
        if row.get("node"):
            row["ir_targets"] = list(
                dict.fromkeys([*(row.get("ir_targets") or []), *parents])
            )


def split_tp8_forward(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[tuple[list[dict[str, Any]], list[dict[str, Any]]]], list[dict[str, Any]]]:
    """Split one exact forward through the 1 + 2*60 all-reduce contract."""

    reductions = _all_reduce_groups(rows)
    expected = 1 + 2 * LAYER_COUNT
    if len(reductions) != expected:
        raise ValueError(
            f"Qwen3.5 TP8 forward requires {expected} ordered all-reduces, "
            f"got {len(reductions)}"
        )
    prefix = rows[: reductions[0][1] + 1]
    layers: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []
    start = reductions[0][1] + 1
    for layer_id in range(LAYER_COUNT):
        attention_end = reductions[1 + 2 * layer_id][1]
        moe_end = reductions[2 + 2 * layer_id][1]
        layers.append((rows[start : attention_end + 1], rows[attention_end + 1 : moe_end + 1]))
        start = moe_end + 1
    return prefix, layers, rows[start:]


def _map_sglang_attention(
    rows: list[dict[str, Any]], kind: str, phase: str, layer_id: int
) -> None:
    if not rows:
        raise ValueError("empty SGLang attention segment")
    if kind == "gdn":
        collective_start = _trailing_all_reduce_start(rows)
        body = rows[:collective_start]
        landmarks = {
            "split": next((i for i, row in enumerate(body) if "fused_qkvzba_split" in _lower(row)), None),
            "conv": next((i for i, row in enumerate(body) if "causal_conv1d" in _lower(row)), None),
            "recur": next(
                (
                    i
                    for i, row in enumerate(body)
                    if "gated_delta_rule" in _lower(row)
                    or "gated_delta_net_chunked" in _lower(row)
                ),
                None,
            ),
            "norm": next((i for i, row in enumerate(body) if "layer_norm_fwd" in _lower(row)), None),
        }
        if any(value is None for value in landmarks.values()):
            raise ValueError(f"incomplete SGLang GDN landmarks: {landmarks}")
        split, conv, recur, norm = (int(landmarks[key]) for key in ("split", "conv", "recur", "norm"))
        norm_prefix = next(
            (i for i, row in enumerate(body[:split]) if _is_norm_kernel(row)), None
        )
        for index, row in enumerate(body):
            if norm_prefix is not None and index == norm_prefix:
                _map_input_norm(
                    row,
                    layer_id=layer_id,
                    kind=kind,
                    framework="sglang",
                    phase=phase,
                )
                continue
            if "compare_scalar" in _lower(row):
                _support(
                    row,
                    "attention_plan_metadata",
                    "sequence-shape predicate used to select the GDN prefill schedule",
                )
                continue
            if index <= split:
                node = "gdn_attention.qkvz_projection"
                fused = ("gdn_attention.ba_projection",)
            elif index <= conv:
                node = "gdn_attention.causal_conv"
                fused = ("gdn_attention.conv_state_read",)
            elif index <= recur:
                node = "gdn_attention.gated_delta_recurrence"
                fused = ("gdn_attention.recurrent_state_read", "gdn_attention.state_write")
            elif index <= norm:
                node, fused = "gdn_attention.output_gate_norm", ()
            elif index > norm:
                node, fused = "gdn_attention.output_projection", ()
            else:
                raise ValueError(f"unresolved SGLang GDN sequence slot {index}")
            _assign(row, node, method="tp_collective_bounded_sglang_gdn_sequence", ir_targets=fused)
        residual_targets = (
            (f"{_block(kind)}.attention_residual", f"{_block(kind)}.post_attention_norm")
            if phase == "decode"
            else ()
        )
        _assign_collective_group(
            rows[collective_start:],
            "gdn_moe_block.tp_attention_output_collective",
            ir_targets=residual_targets,
        )
        return

    collective_start = _trailing_all_reduce_start(rows)
    body = rows[:collective_start]
    qk = next((i for i, row in enumerate(body) if "fused_qk_rmsnorm_rope_gate" in _lower(row)), None)
    cache = next((i for i, row in enumerate(body) if "qkv_kv_cache" in _lower(row)), None)
    attention = next((i for i, row in enumerate(body) if "fmha" in _lower(row)), None)
    gate = next((i for i, row in enumerate(body) if "fused_sigmoid_mul" in _lower(row)), None)
    if None in (qk, cache, attention, gate):
        raise ValueError("incomplete SGLang full-attention landmarks")
    norm_prefix = next(
        (i for i, row in enumerate(body[: int(qk)]) if _is_norm_kernel(row)), None
    )
    for index, row in enumerate(body):
        if norm_prefix is not None and index == norm_prefix:
            _map_input_norm(
                row,
                layer_id=layer_id,
                kind=kind,
                framework="sglang",
                phase=phase,
            )
            continue
        if index < int(qk):
            node, fused = "full_attention.qkv_projection", ()
        elif cache is not None and index < int(cache):
            node, fused = "full_attention.qk_norm", ("full_attention.partial_rope",)
        elif cache is not None and index < int(attention):
            node, fused = "full_attention.kv_state_write", ()
        elif index < int(gate):
            node, fused = "full_attention.causal_gqa", ("full_attention.kv_state_read",)
        elif index == gate:
            node, fused = "full_attention.attention_output_gate", ()
        elif index > int(gate):
            node, fused = "full_attention.output_projection", ()
        else:
            node, fused = "full_attention.output_projection", ()
        _assign(row, node, method="tp_collective_bounded_sglang_full_attention_sequence", ir_targets=fused)
    residual_targets = (
        (f"{_block(kind)}.attention_residual", f"{_block(kind)}.post_attention_norm")
        if phase == "decode"
        else ()
    )
    _assign_collective_group(
        rows[collective_start:],
        "full_attention_moe_block.tp_attention_output_collective",
        ir_targets=residual_targets,
    )


def _map_sglang_moe(
    rows: list[dict[str, Any]], kind: str, phase: str, layer_id: int
) -> None:
    collective_start = _trailing_all_reduce_start(rows)
    body = rows[:collective_start]
    routing = next((i for i, row in enumerate(body) if "routingindices" in _lower(row)), None)
    gate_bmm = next((i for i, row in enumerate(body) if _lower(row).startswith("bmm_e")), None)
    down_bmm = next((i for i, row in enumerate(body) if _lower(row).startswith("bmm_bfloat16")), None)
    finalize = next((i for i, row in enumerate(body) if "finalizekernel" in _lower(row)), None)
    combine = next((i for i, row in enumerate(body) if "fused_gate_sigmoid_mul_add" in _lower(row)), None)
    if None in (routing, gate_bmm, down_bmm, finalize, combine):
        raise ValueError("incomplete SGLang MoE landmarks")
    routed_quant = next(
        (i for i, row in enumerate(body[: int(routing)]) if "nvfp4_quant" in _lower(row)),
        None,
    )
    if routed_quant is None:
        raise ValueError("SGLang MoE segment is missing routed-expert quantization")
    leading_norm = next(
        (i for i, row in enumerate(body) if _is_norm_kernel(row)), None
    )
    for index, row in enumerate(body):
        name = _lower(row)
        if leading_norm is not None and index == leading_norm:
            _map_post_attention_norm(row, kind=kind, framework="sglang", phase=phase)
            continue
        if "cudafunctoronself_add" in name:
            _support(
                row,
                "moe_schedule_metadata",
                "graph-replay expert-counter update used by the fused MoE scheduler",
            )
            continue
        if "alloc_decode_kernel" in name:
            _support(
                row,
                "linear_attention_schedule_metadata",
                "asynchronous GDN decode workspace allocation may overlap the "
                "neighboring MoE timestamp segment; it carries no model tensor "
                "transition and is excluded from MoE ownership",
            )
            continue
        if "index_put_kernel_impl" in name or "index_kernel_impl" in name:
            _support(
                row,
                "moe_schedule_metadata",
                "graph-replay index update adjacent to the router GEMM and its "
                "memcpy epilogue; it updates fused-MoE scheduler metadata rather "
                "than a Model IR tensor",
            )
            continue
        if (
            index == routed_quant
            or index in {gate_bmm, down_bmm}
            or "finalizekernel" in name
            or "direct_copy_kernel_cuda" in name
        ):
            node = "moe_block.routed_experts"
        elif "routing" in name:
            node = "moe_block.router"
        elif index == combine:
            node = "moe_block.weighted_combine"
        elif name in {"memcpy32_post", "memcpy128"} or (
            "tst_" in name and "qqtst_" not in name
        ):
            # The graph-on router GEMM may expose an extra memcpy epilogue.
            node = "moe_block.router"
        elif "splitkreduce" in name:
            # The pinned router GEMM has beta=false; shared-expert GEMMs use
            # beta=true.  This template bit remains visible even when graph
            # replay interleaves the two streams.
            node = (
                "moe_block.router"
                if "__nv_bfloat16, false, float" in name
                else "moe_block.shared_expert"
            )
        elif any(
            token in name
            for token in (
                "_static_quant_fp8",
                "act_and_mul",
                "qqtst_",
            )
        ):
            node = "moe_block.shared_expert"
        else:
            raise ValueError(f"unresolved SGLang MoE kernel: {name}")
        _assign(row, node, method="tp_collective_bounded_sglang_moe_sequence")
    block = _block(kind)
    members: list[str] = []
    if phase == "decode":
        members.append(f"{block}.layer_residual")
        next_block = _next_block(layer_id)
        if next_block:
            members.append(f"{next_block}.input_norm")
    _assign_collective_group(
        rows[collective_start:],
        f"{block}.tp_moe_output_collective",
        ir_targets=members,
    )


def _map_vllm_attention(
    rows: list[dict[str, Any]], kind: str, phase: str, layer_id: int
) -> None:
    collective_start = _trailing_all_reduce_start(rows)
    body = rows[:collective_start]
    if kind == "gdn":
        conv = next((i for i, row in enumerate(body) if "causal_conv1d" in _lower(row)), None)
        recur = next(
            (
                i
                for i, row in enumerate(body)
                if "gated_delta_rule" in _lower(row)
                or "gated_delta_net_chunked" in _lower(row)
            ),
            None,
        )
        if None in (conv, recur):
            raise ValueError("incomplete vLLM GDN landmarks")
        output_start = len(body) - 1
        output_norm_start = next(
            (
                i
                for i in range(int(recur) + 1, output_start)
                if "triton_per_fused" in _lower(body[i])
                or "layer_norm" in _lower(body[i])
            ),
            output_start - 1,
        )
        input_norm = next(
            (i for i, row in enumerate(body[: int(conv)]) if "rms_norm" in _lower(row)),
            None,
        )
        for index, row in enumerate(body):
            if input_norm is not None and index == input_norm:
                _map_input_norm(
                    row,
                    layer_id=layer_id,
                    kind=kind,
                    framework="vllm",
                    phase=phase,
                )
                continue
            if "zeros" in _lower(row):
                _support(row, "attention_plan_metadata", "compiled GDN scratch initialization")
                continue
            if index < int(conv):
                node, fused = "gdn_attention.qkvz_projection", ("gdn_attention.ba_projection",)
            elif index <= conv:
                node, fused = "gdn_attention.causal_conv", ("gdn_attention.conv_state_read",)
            elif index < output_norm_start:
                node, fused = "gdn_attention.gated_delta_recurrence", (
                    "gdn_attention.recurrent_state_read",
                    "gdn_attention.state_write",
                )
            elif index < output_start:
                node, fused = "gdn_attention.output_gate_norm", ()
            else:
                node, fused = "gdn_attention.output_projection", ()
            _assign(row, node, method=f"tp_collective_bounded_vllm_{phase}_gdn_sequence", ir_targets=fused)
        residual_targets = (
            (f"{_block(kind)}.attention_residual", f"{_block(kind)}.post_attention_norm")
            if phase == "decode"
            else ()
        )
        _assign_collective_group(
            rows[collective_start:],
            "gdn_moe_block.tp_attention_output_collective",
            ir_targets=residual_targets,
        )
        return

    attention = next((i for i, row in enumerate(body) if "fmha" in _lower(row)), None)
    cache = next((i for i, row in enumerate(body) if "reshape_and_cache" in _lower(row)), None)
    gate = next((i for i, row in enumerate(body) if "sigmoid" in _lower(row)), None)
    if attention is not None and gate is None and int(attention) + 1 < len(body):
        # torch.compile gives the output-gate pointwise kernel an ordinal-only
        # name in the prefill graph; its slot between FMHA and o_proj is stable.
        gate = int(attention) + 1
    if None in (attention, gate):
        raise ValueError("incomplete vLLM full-attention landmarks")
    # The compiled full-attention partition has an explicit, retained source
    # contract.  Its qkv projection is the quant pointwise plus bmm; kernels
    # 5/6/7 implement Q/K RMSNorm and RoPE; kernel 8 constructs and quantizes
    # K/V before reshape_and_cache writes the state.  A following FillFunctor
    # initializes attention-plan scratch and is not Q/K normalization.  Do not
    # infer these roles from qkv_start+1: that was the source of the historical
    # false FillFunctor -> qk_norm claim.
    qk_tokens = (
        ("triton_poi_fused_6", "triton_red_fused_7", "triton_poi_fused_8")
        if phase == "prefill"
        else ("triton_poi_fused_5", "triton_red_fused_6", "triton_poi_fused_7")
    )
    kv_prepare_token = (
        "rms_norm_slice_split_split_with_sizes_view_9"
        if phase == "prefill"
        else "rms_norm_slice_split_split_with_sizes_view_8"
    )
    rope_token = "triton_poi_fused_8" if phase == "prefill" else "triton_poi_fused_7"
    qk_landmarks = {
        index
        for index, row in enumerate(body)
        if any(token in _lower(row) for token in qk_tokens)
    }
    kv_prepare = next(
        (
            index
            for index, row in enumerate(body)
            if kv_prepare_token in _lower(row)
        ),
        None,
    )
    if not qk_landmarks or kv_prepare is None or cache is None:
        raise ValueError(
            "vLLM compiled full-attention source landmarks are incomplete: "
            f"qk={sorted(qk_landmarks)} kv_prepare={kv_prepare} cache={cache}"
        )
    for index, row in enumerate(body):
        name = _lower(row)
        if "fillfunctor<unsigned char>" in name:
            _support(
                row,
                "attention_plan_metadata",
                "compiled attention-plan scratch initialization before FMHA",
            )
        elif index in qk_landmarks:
            rope_targets = (
                ("full_attention.partial_rope",)
                if rope_token in name
                else ()
            )
            node, fused = "full_attention.qk_norm", rope_targets
            _assign(
                row,
                node,
                method=f"vllm_{phase}_inductor_source_nodes_qk_norm_rope",
                ir_targets=fused,
            )
        elif index in {int(kv_prepare), int(cache)}:
            _assign(
                row,
                "full_attention.kv_state_write",
                method=f"vllm_{phase}_inductor_source_nodes_kv_prepare_write",
            )
        elif index < min(qk_landmarks):
            node, fused = "full_attention.qkv_projection", ()
        elif index == attention:
            node, fused = "full_attention.causal_gqa", ("full_attention.kv_state_read",)
        elif index == gate:
            node, fused = "full_attention.attention_output_gate", ()
        elif index > int(gate):
            node, fused = "full_attention.output_projection", ()
        else:
            raise ValueError(
                f"unresolved vLLM compiled full-attention slot {index}: {name}"
            )
        if not row.get("node") and not row.get("support_class"):
            _assign(
                row,
                node,
                method=f"tp_collective_bounded_vllm_{phase}_full_attention_sequence",
                ir_targets=fused,
            )
    residual_targets = (
        (f"{_block(kind)}.attention_residual", f"{_block(kind)}.post_attention_norm")
        if phase == "decode"
        else ()
    )
    _assign_collective_group(
        rows[collective_start:],
        "full_attention_moe_block.tp_attention_output_collective",
        ir_targets=residual_targets,
    )


def _map_vllm_moe(
    rows: list[dict[str, Any]], kind: str, phase: str, layer_id: int
) -> None:
    collective_start = _trailing_all_reduce_start(rows)
    body = rows[:collective_start]
    routing = next((i for i, row in enumerate(body) if "routingindices" in _lower(row)), None)
    gate_bmm = next((i for i, row in enumerate(body) if _lower(row).startswith("bmm_e")), None)
    down_bmm = next((i for i, row in enumerate(body) if _lower(row).startswith("bmm_bfloat16")), None)
    finalize = next((i for i, row in enumerate(body) if "finalizekernel" in _lower(row)), None)
    if None in (routing, gate_bmm, down_bmm, finalize):
        raise ValueError("incomplete vLLM MoE landmarks")
    routed_quant = next(
        (i for i, row in enumerate(body[: int(routing)]) if "cvt_fp16_to_fp4" in _lower(row)),
        None,
    )
    if routed_quant is None:
        raise ValueError("vLLM MoE segment is missing routed-expert quantization")
    router_gemm = next(
        (
            index
            for index, row in enumerate(body)
            if "tst_" in _lower(row) and "qqtst_" not in _lower(row)
        ),
        None,
    )
    if router_gemm is None:
        raise ValueError("vLLM MoE segment is missing router projection")
    leading_norm = next(
        (i for i, row in enumerate(body) if "rms_norm" in _lower(row)),
        None,
    )
    for index, row in enumerate(body):
        name = _lower(row)
        cpu = str(row.get("cpu_op_name") or "").lower()
        if leading_norm is not None and index == leading_norm:
            _map_post_attention_norm(row, kind=kind, framework="vllm", phase=phase)
            continue
        if index == routed_quant or index in {gate_bmm, down_bmm} or "finalizekernel" in name:
            node = "moe_block.routed_experts"
        elif "routing" in name:
            node = "moe_block.router"
        elif "triton_poi_fused_add" in name:
            node = "moe_block.weighted_combine"
        elif "tst_" in name and "qqtst_" not in name:
            # The first TST/split-K pair (before routed quantization) is the
            # router projection.  A later TST/split-K pair is the shared
            # expert down projection.  Shape-specialized kernel names vary
            # across BS1/16/64/256, while this source/occurrence boundary does
            # not.
            node = (
                "moe_block.router"
                if index == int(router_gemm)
                else "moe_block.shared_expert"
            )
        elif "splitkreduce" in name:
            node = (
                "moe_block.router"
                if index < int(routed_quant)
                and index <= int(router_gemm) + 2
                else "moe_block.shared_expert"
            )
        elif any(
            token in name
            for token in (
                "triton_poi_fused__to_copy_clamp_mul_reciprocal_0",
                "triton_poi_fused_mul_silu_slice_0",
                "dot_kernel",
                "reduce_1block_kernel",
                "sigmoid_kernel_cuda",
                "binary_internal::mulfunctor",
            )
        ):
            node = "moe_block.shared_expert"
        elif "cutlass13device_kernel" in name or cpu == "vllm::bmm_fp8":
            node = "moe_block.shared_expert"
        else:
            raise ValueError(f"unresolved vLLM MoE kernel: {name}")
        _assign(row, node, method=f"tp_collective_bounded_vllm_{phase}_moe_sequence")
    block = _block(kind)
    members: list[str] = []
    if phase == "decode":
        members.append(f"{block}.layer_residual")
        next_block = _next_block(layer_id)
        if next_block:
            members.append(f"{next_block}.input_norm")
    _assign_collective_group(
        rows[collective_start:],
        f"{block}.tp_moe_output_collective",
        ir_targets=members,
    )


def _map_prefix(rows: list[dict[str, Any]], framework: str) -> None:
    collective_start = _trailing_all_reduce_start(rows)
    for row in rows[:collective_start]:
        name = _lower(row)
        if "vocab_parallel_embedding" in name:
            _assign(row, "top.embedding", method=f"{framework}_unique_embedding_kernel")
        else:
            _support(row, "request_batch_metadata", "request indices, positions, or state-slot preparation before the model forward")
    _assign_collective_group(
        rows[collective_start:],
        "top.tp_embedding_output_collective",
    )


def _map_suffix(rows: list[dict[str, Any]], framework: str) -> None:
    gather = next((i for i, row in enumerate(rows) if is_all_gather(row)), None)
    if gather is None:
        raise ValueError("Qwen3.5 production forward is missing logits all-gather")
    # The pinned implementations end with final norm, the vocabulary GEMM,
    # its split-K reduction, and the logits all-gather.  The names differ after
    # torch compilation, so the bounded tail order is the portable contract.
    final_norm = next(
        (
            i
            for i, row in enumerate(rows[: int(gather)])
            if "rmsnorm" in _lower(row) or "layer_norm" in _lower(row)
        ),
        None,
    )
    lm_start = (
        int(final_norm) + 1
        if framework == "sglang" and final_norm is not None
        else max(0, int(gather) - 2)
    )
    if framework == "sglang" and final_norm is None:
        raise ValueError("Qwen3.5 SGLang production forward is missing final norm")
    if framework == "vllm" and not rows[:lm_start]:
        raise ValueError("Qwen3.5 vLLM production forward is missing final norm")
    for index, row in enumerate(rows):
        if index == final_norm or (framework == "vllm" and index < lm_start):
            _assign(
                row,
                "top.final_norm",
                method=(
                    "vllm_eager_validated_compiled_final_norm_tail"
                    if framework == "vllm"
                    else "sglang_unique_fused_final_norm_kernel"
                ),
            )
        elif index < lm_start:
            _support(
                row,
                "logits_input_selection",
                "framework selection and packing of the final hidden state before the vocabulary head",
            )
        elif index < int(gather):
            _assign(row, "top.lm_head", method=f"{framework}_bounded_lm_head_tail")
        elif index == gather:
            _assign(row, "top.tp_logits_all_gather", method="complete_eager_validated_tp_collective_order")
        else:
            _support(row, "sampling_and_output", "sampling, token selection, or output materialization after full logits")


def attribute_production_forward(
    rows: list[dict[str, Any]], *, framework: str, phase: str
) -> dict[str, Any]:
    if framework not in {"sglang", "vllm"}:
        raise ValueError(f"unsupported framework {framework!r}")
    ordered_rows = semantic_execution_order(rows)
    for sequence_index, row in enumerate(ordered_rows):
        row["semantic_sequence_index"] = sequence_index
    # Larger FlashInfer collectives launch a second RMSNorm/Lamport kernel.
    # Remove those physical companions while resolving the portable ordered
    # collective spine, then bind each back to its nearest primary owner.
    companions = [row for row in ordered_rows if is_all_reduce_companion(row)]
    spine = [row for row in ordered_rows if not is_all_reduce_companion(row)]
    prefix, layers, suffix = split_tp8_forward(spine)
    _map_prefix(prefix, framework)
    for layer_id, ((attention_rows, moe_rows), kind) in enumerate(zip(layers, LAYER_PATTERN)):
        if framework == "sglang":
            _map_sglang_attention(attention_rows, kind, phase, layer_id)
            _map_sglang_moe(moe_rows, kind, phase, layer_id)
        else:
            _map_vllm_attention(attention_rows, kind, phase, layer_id)
            _map_vllm_moe(moe_rows, kind, phase, layer_id)
        _annotate_occurrence(attention_rows, layer_id, kind, "attention")
        _annotate_occurrence(moe_rows, layer_id, kind, "moe")
    _map_suffix(suffix, framework)

    primary_collectives = [row for row in spine if is_all_reduce(row)]
    for companion in companions:
        owner = min(
            primary_collectives,
            key=lambda row: abs(float(row["ts_us"]) - float(companion["ts_us"])),
        )
        for key in (
            "node",
            "kernel_label",
            "confidence",
            "ir_targets",
            "layer_id",
            "layer_kind",
            "substage",
            "segment_id",
            "occurrence_id",
        ):
            if key in owner:
                companion[key] = owner[key]
        companion["attribution_method"] = (
            "n_to_one_flashinfer_twoshot_rmsnorm_companion"
        )

    semantic_unbound = [row for row in rows if not row.get("node") and not row.get("support_class")]
    if semantic_unbound:
        raise ValueError(f"unclassified production kernels: {len(semantic_unbound)}")
    mapped = [row for row in rows if row.get("node")]
    total_us = sum(float(row.get("dur_us") or 0.0) for row in rows)
    mapped_us = sum(float(row.get("dur_us") or 0.0) for row in mapped)
    rows.sort(key=lambda row: float(row["ts_us"]))
    return {
        "framework": framework,
        "phase": phase,
        "kernel_count": len(rows),
        "mapped_kernel_count": len(mapped),
        "mapped_kernel_count_ratio": len(mapped) / len(rows),
        "mapped_kernel_duration_ratio": mapped_us / total_us if total_us else 0.0,
        "layer_pattern": "GGGA" * 15,
        "gdn_layer_count": 45,
        "full_attention_layer_count": 15,
        "tp_all_reduce_kernel_count": sum(is_all_reduce(row) for row in rows),
        "tp_logical_all_reduce_count": len(primary_collectives),
        "tp_all_gather_count": sum(is_all_gather(row) for row in rows),
        "support_class_counts": dict(Counter(str(row.get("support_class")) for row in rows if row.get("support_class"))),
        "method_counts": dict(Counter(str(row.get("attribution_method")) for row in rows)),
    }
