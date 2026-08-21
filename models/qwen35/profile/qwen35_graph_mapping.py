#!/usr/bin/env python3
"""Structural mapping for Qwen3.5 AgentX CUDA-Graph traces.

CUDA-Graph traces intentionally have no Python stacks.  This module therefore
uses only invariants that are independently visible in the trace: the target
verify annotation, the exact 60-layer GGGA anchor sequence, implementation-
unique target/draft EP4 kernels, and the explicit draft/draft-extend ranges.
Every kernel receives one of three statuses: ``mapped`` for a unique signature,
``fusion`` for work attributable only to a containing IR scope, or ``unmapped``.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass
from typing import Any

from pathlib import Path

from models.common.timeline_artifact import load_eager_stack_index
from models.common.trace_mapping import ForwardWindow, _primary_gpu_annotations


TARGET_PATTERN = tuple(
    "attention" if layer_id % 4 == 3 else "gdn" for layer_id in range(60)
)

EAGER_SCOPE_FALLBACKS = {
    "gdn_moe_block.output_hidden": "gdn_moe_block.qkvz_projection",
    "full_attention_moe_block.output_hidden": "full_attention_moe_block.causal_gqa",
    "top.decoder_stack": "gdn_moe_block.qkvz_projection",
    "mtp_draft_head.draft_decoder_layer": "mtp_full_attention_moe_block.causal_gqa",
    "generation_loop.accept_prefix": "generation_loop.draft_propose",
}


@dataclass(frozen=True)
class GraphMapping:
    node: str | None
    label: str | None
    status: str
    confidence: str
    ir_targets: tuple[str, ...] = ()


def _contains(name: str, *needles: str) -> bool:
    lowered = name.lower()
    return any(needle in lowered for needle in needles)


def _moe_mapping(name: str, *, draft: bool) -> GraphMapping | None:
    lowered = name.lower()
    prefix = "mtp_moe_block" if draft else "moe_block"
    if "deep_ep::" in lowered:
        if "combine" in lowered:
            return GraphMapping(
                "mtp_moe_block.draft_ep4_combine",
                "DeepEP draft combine",
                "mapped",
                "high",
                ("mtp_moe_block.weighted_combine",),
            )
        if "dispatch" in lowered:
            return GraphMapping(
                "mtp_moe_block.draft_ep4_dispatch",
                "DeepEP draft dispatch",
                "mapped",
                "high",
                ("mtp_moe_block.routed_experts",),
            )
    if "moea2apreparedispatch" in lowered:
        return GraphMapping(
            "moe_block.target_ep4_pack",
            "target EP4 dispatch pack",
            "mapped",
            "high",
            ("moe_block.router",),
        )
    if "moea2adispatchkernel" in lowered or "moea2asanitizeexpertids" in lowered:
        return GraphMapping(
            "moe_block.target_ep4_dispatch",
            "target EP4 dispatch",
            "mapped",
            "high",
            ("moe_block.routed_experts",),
        )
    if "moea2apreparecombine" in lowered or "moea2acombinekernel" in lowered:
        return GraphMapping(
            "moe_block.target_ep4_combine",
            "target EP4 combine",
            "mapped",
            "high",
            ("moe_block.weighted_combine",),
        )
    if _contains(lowered, "routingindicesclusterkernel", "_router_triton_kernel"):
        return GraphMapping(f"{prefix}.router", "MoE top-k router", "mapped", "high")
    if _contains(
        lowered,
        "contiguous_gather_grouped_gemm_act_fusion",
        "contiguous_grouped_gemm_finalize_fusion",
        "deep_gemm::",
    ):
        return GraphMapping(
            f"{prefix}.routed_experts", "routed expert GEMM", "mapped", "high"
        )
    if _contains(lowered, "act_and_mul_kernel", "silu_and_mul_kernel"):
        return GraphMapping(
            f"{prefix}.shared_expert", "shared expert activation", "mapped", "medium"
        )
    if _contains(lowered, "fused_gate_sigmoid_mul_add", "sigmoid_gate_mul_add"):
        return GraphMapping(
            f"{prefix}.weighted_combine", "MoE weighted combine", "mapped", "high"
        )
    return None


def direct_graph_mapping(
    name: str, *, substage: str, layer_kind: str | None
) -> GraphMapping | None:
    """Map implementation-unique kernel signatures to stable IR nodes."""

    lowered = name.lower()
    draft = substage in {"draft_extend", "draft"}

    if _contains(lowered, "gdn_replayssm", "replayssm_exact_fold"):
        return GraphMapping(
            "generation_loop.replay_gdn",
            "accepted-prefix GDN state replay",
            "mapped",
            "high",
        )
    if _contains(lowered, "draft_topk1", "build_tree_efficient"):
        return GraphMapping(
            "generation_loop.draft_propose",
            "draft candidate construction",
            "mapped",
            "high",
        )

    moe = _moe_mapping(name, draft=draft)
    if moe is not None:
        return moe

    block = "mtp_full_attention_moe_block" if draft else "full_attention_moe_block"
    if "fmhasm100" in lowered:
        return GraphMapping(f"{block}.causal_gqa", "causal GQA", "mapped", "high")
    if "_fused_qk_rmsnorm_rope_gate_kernel" in lowered:
        return GraphMapping(
            f"{block}.qk_norm",
            "Q/K norm + partial RoPE + gate fusion",
            "mapped",
            "high",
            (f"{block}.partial_rope", f"{block}.attention_output_gate"),
        )
    if "fused_fp8_qkv_kv_cache_kernel" in lowered:
        return GraphMapping(
            f"{block}.qkv_projection",
            "QKV projection + KV-cache write fusion",
            "fusion",
            "high",
            (f"{block}.kv_state_write",),
        )

    if not draft and layer_kind == "gdn":
        if "fused_qkvzba_split" in lowered:
            return GraphMapping(
                "gdn_moe_block.qkvz_projection",
                "GDN QKVZBA split/reshape",
                "mapped",
                "high",
                ("gdn_moe_block.ba_projection",),
            )
        if "causal_conv1d_update" in lowered:
            return GraphMapping(
                "gdn_moe_block.causal_conv", "GDN causal convolution", "mapped", "high"
            )
        if _contains(lowered, "gdn_wide_vec_kernel", "recurrent_gated_delta_rule"):
            return GraphMapping(
                "gdn_moe_block.gated_delta_recurrence",
                "GDN recurrent update",
                "mapped",
                "high",
                ("gdn_moe_block.state_write",),
            )
        if "_layer_norm_fwd_1pass_kernel" in lowered:
            return GraphMapping(
                "gdn_moe_block.output_gate_norm",
                "GDN output gate norm",
                "mapped",
                "medium",
            )

    return None


def _annotation_in_window(
    trace_events: list[dict[str, Any]], *, prefix: str, window: ForwardWindow
) -> list[dict[str, Any]]:
    annotations, _track = _primary_gpu_annotations(trace_events, name_prefix=prefix)
    return [
        event
        for event in annotations
        if window.start_us <= float(event.get("ts", 0.0)) < window.end_us
    ]


def _target_annotation(
    trace_events: list[dict[str, Any]], window: ForwardWindow
) -> dict[str, Any]:
    candidates = _annotation_in_window(
        trace_events, prefix="step[TARGET_VERIFY", window=window
    )
    if len(candidates) != 1:
        raise ValueError(f"expected one primary target verify annotation, got {len(candidates)}")
    return candidates[0]


def _phase_ranges(
    trace_events: list[dict[str, Any]], window: ForwardWindow
) -> list[tuple[float, float, str]]:
    ranges: list[tuple[float, float, str]] = []
    target = _target_annotation(trace_events, window)
    ranges.append(
        (
            float(target["ts"]),
            float(target["ts"]) + float(target.get("dur", 0.0)),
            "target_verify",
        )
    )
    for prefix, exact_name in (("draft_extend", "draft_extend"), ("draft", "draft")):
        for event in _annotation_in_window(trace_events, prefix=prefix, window=window):
            if str(event.get("name")) != exact_name:
                continue
            ranges.append(
                (
                    float(event["ts"]),
                    float(event["ts"]) + float(event.get("dur", 0.0)),
                    exact_name,
                )
            )
    return sorted(ranges)


def _substage(ts_us: float, ranges: list[tuple[float, float, str]]) -> str:
    matches = [kind for start, end, kind in ranges if start <= ts_us < end]
    if not matches:
        return "generation_lifecycle"
    # Exact draft ranges never overlap target. Prefer the narrower draft range
    # over scheduler annotations if future traces add a larger outer range.
    for kind in ("draft_extend", "draft", "target_verify"):
        if kind in matches:
            return kind
    return matches[0]


def _target_anchors(
    kernels: list[dict[str, Any]], *, target_start: float, target_end: float
) -> list[tuple[float, str]]:
    anchors: list[tuple[float, str]] = []
    for kernel in kernels:
        ts_us = float(kernel.get("ts", 0.0))
        if not target_start <= ts_us < target_end:
            continue
        name = str(kernel.get("name", "")).lower()
        if "fused_qkvzba_split" in name:
            anchors.append((ts_us, "gdn"))
        elif "_fused_qk_rmsnorm_rope_gate_kernel" in name:
            anchors.append((ts_us, "attention"))
    anchors.sort()
    kinds = tuple(kind for _ts, kind in anchors)
    if kinds != TARGET_PATTERN:
        raise ValueError(
            "target layer anchor mismatch: "
            f"expected 45 GDN/15 attention in GGGA order, got "
            f"{len(anchors)} anchors ({kinds.count('gdn')} GDN/{kinds.count('attention')} attention)"
        )
    return anchors


def map_graph_window(
    trace_events: list[dict[str, Any]],
    *,
    window: ForwardWindow,
    rank: int,
    step_index: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Map every GPU kernel in one complete AgentX decode iteration."""

    kernels = sorted(
        (
            event
            for event in trace_events
            if event.get("cat") == "kernel"
            and event.get("ph") == "X"
            and window.start_us <= float(event.get("ts", 0.0)) < window.end_us
        ),
        key=lambda event: float(event.get("ts", 0.0)),
    )
    ranges = _phase_ranges(trace_events, window)
    target_start, target_end, _ = next(item for item in ranges if item[2] == "target_verify")
    anchors = _target_anchors(kernels, target_start=target_start, target_end=target_end)
    anchor_times = [item[0] for item in anchors]

    mapped: list[dict[str, Any]] = []
    for event_index, kernel in enumerate(kernels):
        ts_us = float(kernel.get("ts", 0.0))
        name = str(kernel.get("name", ""))
        substage = _substage(ts_us, ranges)
        layer_id = None
        layer_kind = None
        if substage == "target_verify":
            anchor_index = bisect_right(anchor_times, ts_us) - 1
            if anchor_index >= 0:
                layer_id = anchor_index
                layer_kind = anchors[anchor_index][1]

        direct = direct_graph_mapping(name, substage=substage, layer_kind=layer_kind)
        if direct is None and substage == "target_verify":
            view = "gdn_moe_block" if layer_kind == "gdn" else "full_attention_moe_block"
            node = f"{view}.output_hidden" if layer_kind else "top.decoder_stack"
            targets = (
                (f"layer_schedule.layer_{layer_id:02d}",) if layer_id is not None else ()
            )
            direct = GraphMapping(
                node,
                "target layer fused/auxiliary kernel",
                "fusion",
                "structural",
                targets,
            )
        elif direct is None and substage in {"draft_extend", "draft"}:
            direct = GraphMapping(
                "mtp_draft_head.draft_decoder_layer",
                "MTP draft fused/auxiliary kernel",
                "fusion",
                "structural",
            )
        elif direct is None:
            direct = GraphMapping(
                "generation_loop.accept_prefix",
                "speculative lifecycle fused/auxiliary kernel",
                "fusion",
                "structural",
            )

        ir_targets = list(direct.ir_targets)
        if layer_id is not None:
            ir_targets.append(f"layer_schedule.layer_{layer_id:02d}")
        mapped.append(
            {
                "event_id": f"r{rank}-s{step_index}-k{event_index}",
                "rank": rank,
                "step_index": step_index,
                "kernel_name": name,
                "kernel_label": direct.label,
                "node": direct.node,
                "ir_targets": list(dict.fromkeys(ir_targets)),
                "mapping_status": direct.status,
                "attribution_method": (
                    "unique_kernel_signature"
                    if direct.status == "mapped"
                    else "validated_phase_and_layer_scope"
                ),
                "confidence": direct.confidence,
                "substage": substage,
                "layer_id": layer_id,
                "layer_kind": layer_kind,
                "ts_us": ts_us,
                "dur_us": float(kernel.get("dur", 0.0)),
                "stream": (kernel.get("args") or {}).get("stream"),
                "device": (kernel.get("args") or {}).get("device"),
                "pid": kernel.get("pid"),
                "tid": kernel.get("tid"),
            }
        )

    counts = {
        "target_gdn_layers": sum(kind == "gdn" for _ts, kind in anchors),
        "target_attention_layers": sum(kind == "attention" for _ts, kind in anchors),
        "target_ep4_dispatch": sum(
            "moea2adispatchkernel" in event["kernel_name"].lower() for event in mapped
        ),
        "target_ep4_combine": sum(
            "moea2acombinekernel" in event["kernel_name"].lower() for event in mapped
        ),
        "draft_deepep_dispatch": sum(
            "deep_ep::" in event["kernel_name"].lower()
            and "dispatch" in event["kernel_name"].lower()
            for event in mapped
        ),
        "draft_deepep_combine": sum(
            "deep_ep::" in event["kernel_name"].lower()
            and "combine" in event["kernel_name"].lower()
            for event in mapped
        ),
        "gdn_replay": sum(
            "replayssm" in event["kernel_name"].lower() for event in mapped
        ),
    }
    total_us = sum(event["dur_us"] for event in mapped)
    by_status = {
        status: sum(event["dur_us"] for event in mapped if event["mapping_status"] == status)
        for status in ("mapped", "fusion", "unmapped")
    }
    validation = {
        "kernel_count": len(mapped),
        "duration_us": total_us,
        "status_duration_us": by_status,
        "attributed_duration_ratio": (
            min(1.0, (by_status["mapped"] + by_status["fusion"]) / total_us)
            if total_us
            else 0.0
        ),
        "strict_signature_duration_ratio": by_status["mapped"] / total_us if total_us else 0.0,
        "signature_counts": counts,
    }
    return mapped, validation


def map_prefill_window(
    trace_events: list[dict[str, Any]],
    *,
    start_us: float,
    end_us: float,
    rank: int,
    step_index: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Map one complete target-only 8192-token eager prefill forward."""

    kernels = sorted(
        (
            event
            for event in trace_events
            if event.get("cat") == "kernel"
            and event.get("ph") == "X"
            and start_us <= float(event.get("ts", 0.0)) < end_us
        ),
        key=lambda event: float(event.get("ts", 0.0)),
    )
    anchors = _target_anchors(kernels, target_start=start_us, target_end=end_us)
    anchor_times = [item[0] for item in anchors]
    mapped: list[dict[str, Any]] = []
    status_us: Counter[str] = Counter()
    for event_index, kernel in enumerate(kernels):
        ts_us = float(kernel.get("ts", 0.0))
        name = str(kernel.get("name", ""))
        layer_id = max(0, bisect_right(anchor_times, ts_us) - 1)
        layer_kind = anchors[layer_id][1]
        direct = direct_graph_mapping(
            name, substage="target_verify", layer_kind=layer_kind
        )
        if direct is None:
            view = "gdn_moe_block" if layer_kind == "gdn" else "full_attention_moe_block"
            direct = GraphMapping(
                f"{view}.output_hidden",
                "target prefill layer fused/auxiliary kernel",
                "fusion",
                "structural",
            )
        duration_us = float(kernel.get("dur", 0.0))
        status_us[direct.status] += duration_us
        targets = [*direct.ir_targets, f"layer_schedule.layer_{layer_id:02d}"]
        mapped.append(
            {
                "event_id": f"r{rank}-p{step_index}-k{event_index}",
                "rank": rank,
                "step_index": step_index,
                "kernel_name": name,
                "kernel_label": direct.label,
                "node": direct.node,
                "ir_targets": list(dict.fromkeys(targets)),
                "mapping_status": direct.status,
                "attribution_method": (
                    "unique_kernel_signature"
                    if direct.status == "mapped"
                    else "validated_layer_scope"
                ),
                "confidence": direct.confidence,
                "layer_id": layer_id,
                "layer_kind": layer_kind,
                "substage": "target_prefill",
                "ts_us": ts_us,
                "dur_us": duration_us,
                "stream": (kernel.get("args") or {}).get("stream", kernel.get("tid")),
                "device": (kernel.get("args") or {}).get("device"),
                "pid": kernel.get("pid"),
                "tid": kernel.get("tid"),
                "kernel_kind": (
                    "communication"
                    if _contains(name, "moea2adispatch", "moea2acombine", "nccl")
                    else "compute"
                ),
            }
        )

    total_us = sum(float(event["dur_us"]) for event in mapped)
    signature_counts = {
        "target_gdn_layers": sum(
            "fused_qkvzba_split" in str(kernel.get("name", "")).lower()
            for kernel in kernels
        ),
        "target_attention_layers": sum(
            "_fused_qk_rmsnorm_rope_gate_kernel"
            in str(kernel.get("name", "")).lower()
            for kernel in kernels
        ),
        "target_ep4_dispatch": sum(
            "moea2adispatchkernel" in str(kernel.get("name", "")).lower()
            for kernel in kernels
        ),
        "target_ep4_combine": sum(
            "moea2acombinekernel" in str(kernel.get("name", "")).lower()
            for kernel in kernels
        ),
    }
    expected = {
        "target_gdn_layers": 45,
        "target_attention_layers": 15,
        "target_ep4_dispatch": 60,
        "target_ep4_combine": 60,
    }
    mismatch = {
        key: {"expected": value, "actual": signature_counts[key]}
        for key, value in expected.items()
        if signature_counts[key] != value
    }
    if mismatch:
        raise ValueError(f"rank {rank} prefill step {step_index} mismatch: {mismatch}")
    attributed = status_us["mapped"] + status_us["fusion"]
    validation = {
        "rank": rank,
        "step_index": step_index,
        "kernel_count": len(mapped),
        "signature_counts": signature_counts,
        "status_duration_us": dict(status_us),
        "attributed_duration_ratio": attributed / total_us if total_us else 0.0,
        "strict_signature_duration_ratio": status_us["mapped"] / total_us if total_us else 0.0,
        "timing_closure_us": sum(float(event["dur_us"]) for event in mapped) - total_us,
    }
    if validation["attributed_duration_ratio"] < 0.95:
        raise ValueError(f"rank {rank} prefill attributed residency is below 95%")
    return mapped, validation


def attach_graph_stack_evidence(
    events: list[dict[str, Any]], *, mapping_path: Path
) -> list[dict[str, Any]]:
    """Transfer eager stack evidence through an explicit IR-scope relation."""

    index = load_eager_stack_index(mapping_path)
    enriched: list[dict[str, Any]] = []
    for raw in events:
        event = dict(raw)
        node = str(event.get("node") or "")
        name = str(event.get("kernel_name") or "")
        candidates = index["exact"].get((node, name), [])
        match = "exact_kernel_name_and_ir_node"
        source_node = node
        if not candidates:
            candidates = index["by_node"].get(node, [])
            match = "representative_ir_node_stack"
        if not candidates and node in EAGER_SCOPE_FALLBACKS:
            source_node = EAGER_SCOPE_FALLBACKS[node]
            candidates = index["by_node"].get(source_node, [])
            match = "representative_containing_ir_scope"
        if candidates:
            evidence = candidates[0]
            event["python_stack"] = evidence["python_stack"]
            event["cpu_op_name"] = evidence.get("cpu_op_name")
            event["stack_evidence"] = {
                "source": "eager_trace",
                "match": match,
                "kind": evidence["stack_kind"],
                "event_id": evidence["event_id"],
                "source_node": source_node,
                "confidence": evidence["confidence"],
            }
        enriched.append(event)
    return enriched
