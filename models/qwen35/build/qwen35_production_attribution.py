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
            "gdn_attention.tp_gdn_output_collective",
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
        "full_attention.tp_attention_output_collective",
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
    router_start = int(routed_quant) - 1
    while router_start > 0 and "splitkreduce" in _lower(body[router_start]):
        router_start -= 1
    leading_norm = next(
        (i for i, row in enumerate(body[:router_start]) if _is_norm_kernel(row)), None
    )
    for index, row in enumerate(body):
        name = _lower(row)
        if leading_norm is not None and index == leading_norm:
            _map_post_attention_norm(row, kind=kind, framework="sglang", phase=phase)
            continue
        if index < router_start:
            node = "moe_block.shared_expert"
        elif index < int(routed_quant):
            node = "moe_block.router"
        elif index == routed_quant:
            node = "moe_block.routed_experts"
        elif "routing" in name:
            node = "moe_block.router"
        elif index in {gate_bmm, down_bmm} or "finalizekernel" in name or "nvfp4_quant" in name:
            node = "moe_block.routed_experts"
        elif "act_and_mul" in name or "qqtst_40x64" in name or "splitkreduce" in name:
            node = "moe_block.shared_expert"
        elif index >= int(finalize):
            node = "moe_block.weighted_combine"
        else:
            node = "moe_block.routed_experts"
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
        "moe_block.tp_moe_output_collective",
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
            "gdn_attention.tp_gdn_output_collective",
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
    input_norm = next(
        (i for i, row in enumerate(body[: int(attention)]) if "rms_norm" in _lower(row)),
        None,
    )
    qkv_start = input_norm + 1 if input_norm is not None else 0
    qk_owner = qkv_start + 1
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
        if index < qk_owner:
            node, fused = "full_attention.qkv_projection", ()
        elif index == qk_owner:
            node, fused = "full_attention.qk_norm", ("full_attention.partial_rope",)
        elif cache is not None and index == cache:
            node, fused = "full_attention.kv_state_write", ()
        elif index == attention:
            node, fused = "full_attention.causal_gqa", ("full_attention.kv_state_read",)
        elif index == gate:
            node, fused = "full_attention.attention_output_gate", ()
        elif index > int(gate):
            node, fused = "full_attention.output_projection", ()
        else:
            node, fused = "full_attention.qk_norm", ("full_attention.partial_rope",)
        _assign(row, node, method=f"tp_collective_bounded_vllm_{phase}_full_attention_sequence", ir_targets=fused)
    residual_targets = (
        (f"{_block(kind)}.attention_residual", f"{_block(kind)}.post_attention_norm")
        if phase == "decode"
        else ()
    )
    _assign_collective_group(
        rows[collective_start:],
        "full_attention.tp_attention_output_collective",
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
    router_start = int(routed_quant) - 1
    while router_start > 0 and "splitkreduce" in _lower(body[router_start]):
        router_start -= 1
    leading_norm = next(
        (i for i, row in enumerate(body[:router_start]) if "rms_norm" in _lower(row)),
        None,
    )
    combine = len(body) - 1
    for index, row in enumerate(body):
        name = _lower(row)
        if leading_norm is not None and index == leading_norm:
            _map_post_attention_norm(row, kind=kind, framework="vllm", phase=phase)
            continue
        if index < router_start:
            node = "moe_block.shared_expert"
        elif index < int(routed_quant):
            node = "moe_block.router"
        elif index == routed_quant:
            node = "moe_block.routed_experts"
        elif "routing" in name:
            node = "moe_block.router"
        elif index in {gate_bmm, down_bmm} or "finalizekernel" in name or "cvt_fp16_to_fp4" in name:
            node = "moe_block.routed_experts"
        elif index == combine:
            node = "moe_block.weighted_combine"
        elif index > int(finalize):
            node = "moe_block.shared_expert"
        else:
            node = "moe_block.routed_experts"
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
        "moe_block.tp_moe_output_collective",
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
    fused = ("top.embedding",) if framework == "vllm" else ()
    _assign_collective_group(
        rows[collective_start:],
        "top.tp_embedding_output_collective",
        ir_targets=fused,
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
    rows.sort(key=lambda row: float(row["ts_us"]))
    # Larger FlashInfer collectives launch a second RMSNorm/Lamport kernel.
    # Remove those physical companions while resolving the portable ordered
    # collective spine, then bind each back to its nearest primary owner.
    companions = [row for row in rows if is_all_reduce_companion(row)]
    spine = [row for row in rows if not is_all_reduce_companion(row)]
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
