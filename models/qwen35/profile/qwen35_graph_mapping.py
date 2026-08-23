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

import json
from pathlib import Path
import re

from models.common.timeline_artifact import load_eager_stack_index
from models.common.trace_mapping import ForwardWindow, _primary_gpu_annotations, load_trace


TARGET_PATTERN = tuple(
    "attention" if layer_id % 4 == 3 else "gdn" for layer_id in range(60)
)

EAGER_SCOPE_FALLBACKS = {
    "gdn_moe_block.output_hidden": "gdn_attention.qkvz_projection",
    # The eager trace records the Python launch under the enclosing attention
    # backend / MoE combine scope, while the graph trace exposes the fused
    # implementation kernel as a more specific IR node.
    "full_attention.qk_norm": "full_attention.causal_gqa",
    "full_attention.qkv_projection": "full_attention.causal_gqa",
    "full_attention_moe_block.output_hidden": "full_attention.causal_gqa",
    "moe_block.weighted_combine": "moe_block.target_ep4_combine",
    "top.decoder_stack": "gdn_attention.qkvz_projection",
    "mtp_draft_head.draft_decoder_layer": "mtp_full_attention.causal_gqa",
    "generation_loop.accept_prefix": "generation_loop.draft_propose",
    "generation_loop.commit_gdn": "generation_loop.draft_propose",
    "generation_loop.commit_tokens": "generation_loop.draft_propose",
}

NODE_MIGRATIONS = {
    "gdn_moe_block.": "gdn_attention.",
    "full_attention_moe_block.": "full_attention.",
    "mtp_full_attention_moe_block.": "mtp_full_attention.",
}
NON_LEAF_EAGER_NODES = {
    "top.decoder_stack",
    "stack.gdn_layer",
    "stack.full_attention_layer",
    "gdn_moe_block.attention",
    "gdn_moe_block.moe",
    "full_attention_moe_block.attention",
    "full_attention_moe_block.moe",
    "mtp_full_attention_moe_block.attention",
    "mtp_full_attention_moe_block.moe",
    "mtp_draft_head.draft_decoder_layer",
}


def interval_union_duration_us(events: list[dict[str, Any]]) -> float:
    """Return the overlap-safe union of GPU intervals in microseconds."""

    intervals = sorted(
        (
            float(event["ts_us"]),
            float(event["ts_us"]) + float(event["dur_us"]),
        )
        for event in events
        if float(event.get("dur_us", 0.0)) > 0.0
    )
    if not intervals:
        return 0.0
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged)


def attribution_active_union_ratio(events: list[dict[str, Any]]) -> float:
    """Measure attributed active time without mixing independent rank clocks.

    Kernel residency is still reported separately. The semantic attribution
    gate is based on the union of mapped/fusion intervals divided by the union
    of all GPU intervals, summed independently per worker/rank/step.
    """

    groups: dict[tuple[Any, Any, Any], list[dict[str, Any]]] = {}
    for event in events:
        key = (event.get("worker"), event.get("rank"), event.get("step_index"))
        groups.setdefault(key, []).append(event)
    total_active_us = 0.0
    attributed_active_us = 0.0
    for rows in groups.values():
        total_active_us += interval_union_duration_us(rows)
        attributed_active_us += interval_union_duration_us(
            [
                row
                for row in rows
                if row.get("mapping_status") in {"mapped", "fusion"}
            ]
        )
    return attributed_active_us / total_active_us if total_active_us else 0.0


@dataclass(frozen=True)
class GraphMapping:
    node: str | None
    label: str | None
    status: str
    confidence: str
    ir_targets: tuple[str, ...] = ()
    attribution_method: str | None = None
    unmapped_reason: str | None = None
    candidate_nodes: tuple[str, ...] = ()


def _unmapped_mapping(
    *, substage: str, layer_kind: str | None
) -> GraphMapping:
    if substage in {"target_verify", "target_prefill"}:
        layer_view = (
            "gdn_moe_block" if layer_kind == "gdn" else "full_attention_moe_block"
        )
        return GraphMapping(
            None,
            f"Unmapped {'target-verify' if substage == 'target_verify' else 'target-prefill'} kernel",
            "unmapped",
            "unknown",
            ("generation_loop.target_verify",) if substage == "target_verify" else (),
            f"unresolved_within_validated_{substage}_range",
            "Kernel name and existing eager evidence do not uniquely identify a leaf operation.",
            (
                f"{layer_view}.attention",
                f"{layer_view}.moe",
                f"{layer_view}.attention_residual",
                f"{layer_view}.layer_residual",
            ),
        )
    if substage in {"draft", "draft_extend"}:
        return GraphMapping(
            None,
            "Unmapped MTP draft kernel",
            "unmapped",
            "unknown",
            ("generation_loop.draft_propose",),
            "unresolved_within_validated_mtp_draft_range",
            "Kernel name and existing eager evidence do not uniquely identify a draft leaf operation.",
            (
                "mtp_full_attention_moe_block.attention",
                "mtp_full_attention_moe_block.moe",
                "mtp_draft_head.fc_projection",
                "mtp_draft_head.shared_lm_head",
            ),
        )
    return GraphMapping(
        None,
        "Unmapped speculative-lifecycle kernel",
        "unmapped",
        "unknown",
        (),
        "unresolved_speculative_lifecycle_kernel",
        "The kernel is outside a uniquely attributable measured phase range.",
        (
            "generation_loop.accept_prefix",
            "generation_loop.commit_kv",
            "generation_loop.commit_gdn",
            "generation_loop.commit_tokens",
            "generation_loop.next_iteration",
        ),
    )


def _contains(name: str, *needles: str) -> bool:
    lowered = name.lower()
    return any(needle in lowered for needle in needles)


def _migrate_eager_node(node: str) -> str:
    for old_prefix, new_prefix in NODE_MIGRATIONS.items():
        if node.startswith(old_prefix):
            leaf = node.removeprefix(old_prefix)
            if leaf in {
                "qkvz_projection",
                "ba_projection",
                "conv_state_read",
                "causal_conv",
                "recurrent_state_read",
                "gated_delta_recurrence",
                "state_write",
                "output_gate_norm",
                "output_projection",
                "qkv_projection",
                "qk_norm",
                "partial_rope",
                "kv_state_read",
                "causal_gqa",
                "kv_state_write",
                "attention_output_gate",
            }:
                return new_prefix + leaf
    return node


def _legacy_eager_node(node: str) -> str:
    for old_prefix, new_prefix in NODE_MIGRATIONS.items():
        if node.startswith(new_prefix):
            return old_prefix + node.removeprefix(new_prefix)
    return node


def load_unique_eager_kernel_signatures(mapping_path: Path) -> dict[str, str]:
    """Load only exact eager signatures that always resolve to one leaf node.

    A kernel name is rejected if any eager occurrence was unmapped, low
    confidence, mapped to more than one semantic node, or mapped only to an
    enclosing drill scope. This deliberately favors unresolved events over a
    high but untrustworthy coverage number.
    """

    observations: dict[str, set[str | None]] = {}
    for line in mapping_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        name = str(row.get("kernel_name") or "")
        node = str(row.get("selected_node") or "")
        confidence = str(row.get("confidence") or "unmapped")
        selected: str | None = None
        if node and confidence in {"high", "medium"}:
            selected = _migrate_eager_node(node)
            # Controller/top-level nodes are phase-dependent.  A bare kernel
            # name cannot distinguish target LM-head work from the shared MTP
            # head, or replay bookkeeping from draft input preparation.  The
            # phase/sequence mapper below handles those cases explicitly.
            if selected in NON_LEAF_EAGER_NODES or selected.startswith(
                ("generation_loop.", "top.")
            ):
                selected = None
        observations.setdefault(name, set()).add(selected)
    return {
        name: next(iter(nodes))
        for name, nodes in observations.items()
        if len(nodes) == 1 and None not in nodes
    }


def load_occurrence_stack_mapping(
    *, events_path: Path, mapping_path: Path, rank: int, step_index: int = 0
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load one eager prefill window with occurrence-local stack evidence.

    Unlike name-only transfer, this keeps each kernel tied to the Python stack
    that launched that exact occurrence. High/medium stack matches are mapped;
    low-confidence matches stay unresolved except for a small set of validated
    operation/sequence invariants documented below.
    """

    events = [json.loads(line) for line in events_path.read_text().splitlines() if line]
    mappings = [
        json.loads(line) for line in mapping_path.read_text().splitlines() if line
    ]
    events_by_id = {str(event["event_id"]): event for event in events}
    if len(events_by_id) != len(events) or len(mappings) != len(events):
        raise ValueError("occurrence stack mapping does not cover each kernel exactly once")
    if {str(row["event_id"]) for row in mappings} != set(events_by_id):
        raise ValueError("occurrence stack event IDs do not match the mapped kernel window")

    ordered = [events_by_id[str(row["event_id"])] for row in mappings]
    start_us = min(float(event["ts_us"]) for event in ordered)
    end_us = max(float(event["ts_us"]) + float(event["dur_us"]) for event in ordered)
    trace_like = [
        {"name": event["kernel_name"], "ts": event["ts_us"], "dur": event["dur_us"]}
        for event in ordered
    ]
    anchors = _target_anchors(trace_like, target_start=start_us, target_end=end_us)
    anchor_times = [item[0] for item in anchors]
    if tuple(kind for _ts, kind in anchors) != TARGET_PATTERN:
        raise ValueError("occurrence stack window lacks the exact 45-GDN/15-attention order")

    layer_by_index: list[int] = []
    layer_kind_by_index: list[str] = []
    residual_indices: dict[int, list[int]] = {index: [] for index in range(60)}
    for index, event in enumerate(ordered):
        layer_id = max(0, bisect_right(anchor_times, float(event["ts_us"])) - 1)
        layer_by_index.append(layer_id)
        layer_kind_by_index.append(anchors[layer_id][1])
        if "fused_add_rmsnorm" in str(event["kernel_name"]).lower():
            residual_indices[layer_id].append(index)
    if any(len(indices) != 2 for indices in residual_indices.values()):
        counts = {layer: len(indices) for layer, indices in residual_indices.items()}
        raise ValueError(f"expected two fused residual/RMSNorm kernels per layer: {counts}")

    output: list[dict[str, Any]] = []
    status_us: Counter[str] = Counter()
    for index, (event, row) in enumerate(zip(ordered, mappings)):
        name = str(event["kernel_name"])
        lowered = name.lower()
        layer_id = layer_by_index[index]
        layer_kind = layer_kind_by_index[index]
        layer_view = (
            "gdn_moe_block" if layer_kind == "gdn" else "full_attention_moe_block"
        )
        first_residual = residual_indices[layer_id][0]

        direct: GraphMapping | None = None
        if index in residual_indices[layer_id]:
            if index == first_residual:
                direct = GraphMapping(
                    f"{layer_view}.attention_residual",
                    "attention residual add + pre-MoE RMSNorm",
                    "fusion",
                    "high",
                    (f"{layer_view}.post_attention_norm",),
                    "validated_graph_sequence_fusion",
                )
            else:
                direct = GraphMapping(
                    f"{layer_view}.layer_residual",
                    "MoE residual add + next-layer input RMSNorm",
                    "fusion",
                    "high",
                    (f"{layer_view}.input_norm",),
                    "validated_graph_sequence_fusion",
                )
        if direct is None:
            direct = direct_graph_mapping(
                name, substage="target_prefill", layer_kind=layer_kind
            )

        confidence = str(row.get("confidence") or "unmapped")
        selected_node = _migrate_eager_node(str(row.get("selected_node") or ""))
        operator = row.get("operator_frame") or {}
        operator_file = str(operator.get("file") or "")
        operator_line = operator.get("line")

        # One chunked recurrence kernel occurs in every GDN layer and returns
        # the final recurrent state; direct_graph_mapping handles it as fusion.
        # The three direct-copy occurrences inside the same FlashInfer extend
        # call are recurrence input preparation, proven by their occurrence-
        # local operator stack rather than their generic kernel name.
        if (
            direct is None
            and confidence == "low"
            and "direct_copy_kernel_cuda" in lowered
            and operator_file.endswith("attention/linear/kernels/gdn_flashinfer.py")
            and operator_line == 275
        ):
            direct = GraphMapping(
                "gdn_attention.gated_delta_recurrence",
                "GDN recurrence input preparation",
                "mapped",
                "medium",
                (),
                "occurrence_python_stack",
            )
        # This exact QQ GEMM occurs once between GDN recurrence and the first
        # residual/RMSNorm in each GDN layer. The same name later in the layer
        # belongs to the shared expert, so the sequence boundary is required.
        if (
            direct is None
            and layer_kind == "gdn"
            and index < first_residual
            and "nvjet_sm103_qqtst_128x256_128x6_2x1_2cta" in lowered
        ):
            direct = GraphMapping(
                "gdn_attention.output_projection",
                "GDN output projection",
                "mapped",
                "high",
                (),
                "validated_graph_signature_slot",
            )
        if (
            direct is None
            and "nvjet_sm103_tst_128x256_64x6_2x2_2cta" in lowered
        ):
            direct = GraphMapping(
                "moe_block.router",
                "MoE router projection",
                "mapped",
                "high",
                (),
                "validated_graph_signature_slot",
            )

        if direct is None and selected_node and confidence in {"high", "medium"}:
            direct = GraphMapping(
                selected_node,
                "Occurrence-local Python stack",
                "mapped",
                confidence,
                (),
                "occurrence_python_stack",
            )
        if direct is None:
            fallback = _unmapped_mapping(
                substage="target_prefill", layer_kind=layer_kind
            )
            candidates = tuple(
                dict.fromkeys(
                    ([selected_node] if selected_node else [])
                    + list(fallback.candidate_nodes)
                )
            )
            direct = GraphMapping(
                None,
                "Unmapped occurrence-local prefill kernel",
                "unmapped",
                "unknown",
                (),
                "unresolved_occurrence_python_stack",
                "The occurrence stack or sequence does not uniquely identify a leaf operation.",
                candidates,
            )

        duration_us = float(event["dur_us"])
        status_us[direct.status] += duration_us
        attribution_method = direct.attribution_method or (
            "unique_kernel_signature"
            if direct.status == "mapped"
            else "kernel_signature_fusion"
            if direct.status == "fusion"
            else "unresolved"
        )
        output.append(
            {
                "event_id": f"r{rank}-p{step_index}-k{index}",
                "engine": "sglang",
                "rank": rank,
                "step_index": step_index,
                "kernel_name": name,
                "kernel_label": direct.label,
                "node": direct.node,
                "ir_targets": list(direct.ir_targets),
                "mapping_status": direct.status,
                "fusion_group": (
                    f"r{rank}-p{step_index}-k{index}"
                    if direct.status == "fusion"
                    else None
                ),
                "attribution_method": attribution_method,
                "confidence": direct.confidence,
                "unmapped_reason": direct.unmapped_reason,
                "candidate_nodes": list(direct.candidate_nodes),
                "layer_id": layer_id,
                "layer_kind": layer_kind,
                "substage": "target_prefill",
                "mtp_round": None,
                "ts_us": float(event["ts_us"]),
                "dur_us": duration_us,
                "stream": event.get("stream"),
                "device": event.get("device"),
                "kernel_kind": (
                    "communication"
                    if _contains(name, "moea2adispatch", "moea2acombine", "nccl")
                    else "compute"
                ),
                "python_stack": event.get("python_stack") or [],
                "cpu_op_name": event.get("cpu_op_name"),
                "cpu_input_dims": event.get("cpu_input_dims"),
                "cpu_input_types": event.get("cpu_input_types"),
            }
        )

    total_us = sum(status_us.values())
    validation = {
        "kernel_count": len(output),
        "duration_us": total_us,
        "status_duration_us": dict(status_us),
        "attributed_duration_ratio": (
            (status_us["mapped"] + status_us["fusion"]) / total_us
            if total_us
            else 0.0
        ),
        "strict_signature_duration_ratio": sum(
            event["dur_us"]
            for event in output
            if event["attribution_method"] in {
                "unique_kernel_signature",
                "kernel_signature_fusion",
            }
        )
        / total_us,
        "timeline_interval_coverage_ratio": sum(status_us.values()) / total_us,
        "signature_counts": {
            "target_gdn_layers": sum(kind == "gdn" for _ts, kind in anchors),
            "target_attention_layers": sum(
                kind == "attention" for _ts, kind in anchors
            ),
            "target_ep4_dispatch": sum(
                "moea2adispatchkernel" in event["kernel_name"].lower()
                for event in output
            ),
            "target_ep4_combine": sum(
                "moea2acombinekernel" in event["kernel_name"].lower()
                for event in output
            ),
        },
    }
    return output, validation


def transfer_occurrence_stack_mapping(
    target_events: list[dict[str, Any]],
    source_events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Transfer occurrence evidence only across one exact contiguous sequence.

    Target timestamps/durations and event IDs are retained. The transfer fails
    closed unless the target's model-core suffix is byte-for-byte identical in
    kernel-name order to a source prefix. This permits scheduler-only prefixes
    and profiler-stop suffixes without using fuzzy name or position matching.
    """

    if not target_events or not source_events:
        raise ValueError("occurrence transfer requires non-empty source and target")
    target_names = [str(event["kernel_name"]) for event in target_events]
    source_names = [str(event["kernel_name"]) for event in source_events]
    candidates = [
        index
        for index, name in enumerate(target_names)
        if name == source_names[0]
        and target_names[index:] == source_names[: len(target_names) - index]
    ]
    if len(candidates) != 1:
        raise ValueError(
            "occurrence transfer lacks one exact contiguous alignment: "
            f"candidates={candidates}, target={len(target_names)}, source={len(source_names)}"
        )
    target_start = candidates[0]
    aligned_count = len(target_events) - target_start
    output = [dict(event) for event in target_events]
    evidence_fields = {
        "node",
        "ir_targets",
        "mapping_status",
        "fusion_group",
        "attribution_method",
        "confidence",
        "unmapped_reason",
        "candidate_nodes",
        "layer_id",
        "layer_kind",
        "substage",
        "mtp_round",
        "kernel_label",
        "python_stack",
        "cpu_op_name",
        "cpu_input_dims",
        "cpu_input_types",
    }
    for offset in range(aligned_count):
        target = output[target_start + offset]
        source = source_events[offset]
        if target["kernel_name"] != source["kernel_name"]:
            raise AssertionError("validated occurrence alignment changed during transfer")
        for field in evidence_fields:
            if field in source:
                target[field] = source[field]
        target["attribution_method"] = (
            "exact_occurrence_sequence_transfer"
            if source.get("mapping_status") == "mapped"
            else source.get("attribution_method")
        )
        if source.get("mapping_status") == "fusion":
            target["fusion_group"] = target["event_id"]

    status_us: Counter[str] = Counter()
    for event in output:
        status_us[str(event["mapping_status"])] += float(event["dur_us"])
    total_us = sum(status_us.values())
    validation = {
        "kernel_count": len(output),
        "duration_us": total_us,
        "status_duration_us": dict(status_us),
        "attributed_duration_ratio": (
            (status_us["mapped"] + status_us["fusion"]) / total_us
            if total_us
            else 0.0
        ),
        "strict_signature_duration_ratio": sum(
            float(event["dur_us"])
            for event in output
            if event.get("attribution_method")
            in {"unique_kernel_signature", "kernel_signature_fusion"}
        )
        / total_us,
        "timeline_interval_coverage_ratio": sum(status_us.values()) / total_us,
        "occurrence_alignment": {
            "method": "exact_contiguous_kernel_name_sequence",
            "target_prefix_untransferred": target_start,
            "aligned_kernel_count": aligned_count,
            "source_suffix_unused": len(source_events) - aligned_count,
        },
    }
    return output, validation


def _eager_signature_mapping(
    name: str, eager_signatures: dict[str, str] | None
) -> GraphMapping | None:
    node = (eager_signatures or {}).get(name)
    if node is None:
        return None
    return GraphMapping(
        node,
        "Exact kernel signature proven by eager Python stack",
        "mapped",
        "medium",
        (),
        "exact_eager_kernel_signature",
    )


def _context_group(event: dict[str, Any]) -> tuple[str, int] | None:
    substage = str(event.get("substage") or "")
    if substage == "target_verify" and event.get("layer_id") is not None:
        return (substage, int(event["layer_id"]))
    if substage in {"draft", "draft_extend"} and event.get("mtp_round") is not None:
        return (substage, int(event["mtp_round"]))
    return None


def _contextual_keys(events: list[dict[str, Any]]) -> list[tuple[str, str, str, int] | None]:
    counts: Counter[tuple[tuple[str, int], str]] = Counter()
    output: list[tuple[str, str, str, int] | None] = []
    for event in events:
        group = _context_group(event)
        if group is None:
            output.append(None)
            continue
        name = str(event["kernel_name"])
        counter_key = (group, name)
        ordinal = counts[counter_key]
        counts[counter_key] += 1
        output.append(
            (
                str(event["substage"]),
                str(event.get("layer_kind") or ""),
                name,
                ordinal,
            )
        )
    return output


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
    if "_fused_conv_window_scatter_with_mask_kernel" in lowered:
        return GraphMapping(
            "generation_loop.commit_gdn",
            "accepted convolution-state commit",
            "mapped",
            "high",
        )
    if "verifytreegreedy" in lowered:
        return GraphMapping(
            "generation_loop.accept_prefix",
            "verify tree and select accepted prefix",
            "mapped",
            "high",
        )
    if "fill_bonus_tokens" in lowered:
        return GraphMapping(
            "generation_loop.commit_tokens",
            "publish accepted target bonus tokens",
            "mapped",
            "high",
            ("generation_loop.accept_prefix",),
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

    block = "mtp_full_attention" if draft else "full_attention"
    if "fmhasm100" in lowered:
        return GraphMapping(f"{block}.causal_gqa", "causal GQA", "mapped", "high")
    if "_fused_qk_rmsnorm_rope_gate_kernel" in lowered:
        return GraphMapping(
            f"{block}.qk_norm",
            "Q/K norm + partial RoPE + gate fusion",
            "fusion",
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
                "gdn_attention.qkvz_projection",
                "GDN QKVZBA split/reshape",
                "fusion",
                "high",
                ("gdn_attention.ba_projection",),
            )
        if "causal_conv1d_update" in lowered:
            return GraphMapping(
                "gdn_attention.causal_conv", "GDN causal convolution", "mapped", "high"
            )
        if _contains(lowered, "gdn_wide_vec_kernel", "recurrent_gated_delta_rule"):
            return GraphMapping(
                "gdn_attention.gated_delta_recurrence",
                "GDN recurrent update",
                "fusion",
                "high",
                ("gdn_attention.state_write",),
            )
        if "gateddeltanetchunkedkernel" in lowered:
            return GraphMapping(
                "gdn_attention.gated_delta_recurrence",
                "chunked GDN recurrence + final-state write",
                "fusion",
                "high",
                ("gdn_attention.state_write",),
                "kernel_signature_fusion",
            )
        if "_layer_norm_fwd_1pass_kernel" in lowered:
            return GraphMapping(
                "gdn_attention.output_gate_norm",
                "GDN output gate norm",
                "mapped",
                "medium",
            )

    return None


def _target_structural_mapping(
    name: str,
    *,
    layer_id: int,
    layer_kind: str,
    position: int,
    first_residual: int,
    second_residual: int,
) -> GraphMapping | None:
    """Map decode kernels whose leaf is fixed by validated layer boundaries.

    The CUDA graph executes work on several streams, so global timestamp order
    is not a safe occurrence-to-occurrence join.  The following slots use only
    invariants repeated in every one of the 60 target layers: the attention
    anchor, exactly two fused residual/RMSNorm kernels, and the explicit EP4
    dispatch/expert/combine sequence.  Kernel classes outside those narrow
    slots remain unresolved.
    """

    lowered = name.lower()
    layer_view = (
        "gdn_moe_block" if layer_kind == "gdn" else "full_attention_moe_block"
    )
    if position == first_residual:
        return GraphMapping(
            f"{layer_view}.attention_residual",
            "attention residual add + pre-MoE RMSNorm",
            "fusion",
            "high",
            (f"{layer_view}.post_attention_norm",),
            "validated_graph_sequence_fusion",
        )
    if position == second_residual:
        next_norm = (
            "top.final_norm"
            if layer_id == len(TARGET_PATTERN) - 1
            else (
                "gdn_moe_block.input_norm"
                if TARGET_PATTERN[layer_id + 1] == "gdn"
                else "full_attention_moe_block.input_norm"
            )
        )
        return GraphMapping(
            f"{layer_view}.layer_residual",
            "MoE residual add + following RMSNorm",
            "fusion",
            "high",
            (next_norm,),
            "validated_graph_sequence_fusion",
        )

    # Between an attention anchor and the first residual, these implementation
    # kernels are the output projection (plus its FP8 input preparation).  Full
    # attention has one separately identifiable sigmoid output-gate kernel.
    if position < first_residual:
        if layer_kind == "attention" and "_fused_sigmoid_mul_kernel" in lowered:
            return GraphMapping(
                "full_attention.attention_output_gate",
                "full-attention output gate",
                "mapped",
                "high",
                (),
                "validated_graph_signature_slot",
            )
        if _contains(
            lowered,
            "_static_quant_fp8",
            "nvjet_sm103_qqtst_",
            "cublaslt::splitkreduce_kernel",
        ):
            node = (
                "gdn_attention.output_projection"
                if layer_kind == "gdn"
                else "full_attention.output_projection"
            )
            return GraphMapping(
                node,
                "attention output-projection implementation",
                "mapped",
                "high",
                (),
                "validated_graph_signature_slot",
            )
        return None

    # The MoE interval is bounded by the two residual/RMSNorm fusions.  The
    # shared-expert and router projection kernel families are independently
    # repeated in all 45 GDN and 15 full-attention layers in the shape-matched
    # eager trace.  Generic row-sync kernels are deliberately not included.
    if first_residual < position < second_residual:
        if _contains(
            lowered,
            "_static_quant_fp8",
            "nvjet_sm103_qqtst_",
        ):
            return GraphMapping(
                "moe_block.shared_expert",
                "shared-expert implementation",
                "mapped",
                "medium",
                (),
                "validated_graph_signature_slot",
            )
        if _contains(
            lowered,
            "nvjet_sm103_tst_64x16_",
            "cublaslt::splitkreduce_kernel",
            "memcpy32_post",
            "memcpy128",
            "fillfunctor<float>",
        ):
            return GraphMapping(
                "moe_block.router",
                "router projection/top-k implementation",
                "mapped",
                "medium",
                (),
                "validated_graph_signature_slot",
            )
        if lowered == "memset32":
            return GraphMapping(
                "moe_block.target_ep4_dispatch",
                "EP4 expert-workspace initialization bounded by dispatch and expert launches",
                "fusion",
                "medium",
                ("moe_block.routed_experts",),
                "validated_graph_sequence_fusion",
            )
        return None

    # The next layer's input projection launches after the second fused
    # residual/RMSNorm and before its split/attention anchor.  The final layer
    # instead ends in the target final norm + shared LM head.
    if position > second_residual:
        if layer_id == len(TARGET_PATTERN) - 1:
            if _contains(
                lowered,
                "nvjet_sm103_tst_",
                "direct_copy_kernel_cuda",
            ):
                return GraphMapping(
                    "top.lm_head",
                    "target shared LM head",
                    "mapped",
                    "high",
                    (),
                    "validated_graph_signature_slot",
                )
            return None
        if _contains(
            lowered,
            "_static_quant_fp8",
            "nvjet_sm103_qqtst_",
            "nvjet_sm103_tst_",
            "cublaslt::splitkreduce_kernel",
        ):
            next_kind = TARGET_PATTERN[layer_id + 1]
            node = (
                "gdn_attention.qkvz_projection"
                if next_kind == "gdn"
                else "full_attention.qkv_projection"
            )
            return GraphMapping(
                node,
                "next-layer input projection",
                "mapped",
                "high",
                (),
                "validated_graph_signature_slot",
            )
    return None


def _draft_structural_mapping(
    name: str,
    *,
    position: int,
    attention_anchor: int,
    first_residual: int,
    second_residual: int,
    head_position: int | None,
) -> GraphMapping | None:
    """Map the repeated one-layer MTP draft skeleton inside one draft round."""

    lowered = name.lower()
    if position == first_residual:
        return GraphMapping(
            "mtp_full_attention_moe_block.attention_residual",
            "MTP attention residual add + pre-MoE RMSNorm",
            "fusion",
            "high",
            ("mtp_full_attention_moe_block.post_attention_norm",),
            "validated_graph_sequence_fusion",
        )
    if position == second_residual:
        return GraphMapping(
            "mtp_full_attention_moe_block.layer_residual",
            "MTP MoE residual add + draft final RMSNorm",
            "fusion",
            "high",
            ("mtp_draft_head.draft_final_norm",),
            "validated_graph_sequence_fusion",
        )
    if attention_anchor <= position < first_residual:
        if "_fused_sigmoid_mul_kernel" in lowered:
            return GraphMapping(
                "mtp_full_attention.attention_output_gate",
                "MTP attention output gate",
                "mapped",
                "high",
                (),
                "validated_graph_signature_slot",
            )
        if _contains(
            lowered,
            "nvjet_sm103_tst_64x8_",
            "cublaslt::splitkreduce_kernel",
        ):
            return GraphMapping(
                "mtp_full_attention.output_projection",
                "MTP attention output projection",
                "mapped",
                "high",
                (),
                "validated_graph_signature_slot",
            )
    if head_position is not None and position == head_position:
        return GraphMapping(
            "mtp_draft_head.shared_lm_head",
            "MTP shared LM head",
            "mapped",
            "high",
            (),
            "validated_graph_signature_slot",
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


def target_verify_batch_size(
    trace_events: list[dict[str, Any]], window: ForwardWindow
) -> int:
    """Read the measured local target-verify batch from the primary NVTX range."""

    annotation = _target_annotation(trace_events, window)
    match = re.fullmatch(r"step\[TARGET_VERIFY bs=(\d+)\]", str(annotation["name"]))
    if match is None:
        raise ValueError(f"invalid target verify annotation: {annotation['name']!r}")
    return int(match.group(1))


def complete_eager_decode_window(
    events: list[dict[str, Any]], window: ForwardWindow, *, rank: int
) -> ForwardWindow:
    """Include a preceding draft range when one-step eager stop cuts the tail.

    CUDA-Graph captures keep running through the post-verify ``draft`` range.
    With a one-step eager profiler, automatic stop can instead retain the four
    draft rounds immediately preceding TARGET_VERIFY plus the following
    DRAFT_EXTEND round. Both layouts are one complete five-round MTP cycle.
    """

    stages, _track = _primary_gpu_annotations(events, name_prefix="draft")
    kernels = [event for event in events if event.get("cat") == "kernel"]

    def draft_anchor_count(start_us: float) -> int:
        ranges = [
            (
                float(event.get("ts", 0.0)),
                float(event.get("ts", 0.0)) + float(event.get("dur", 0.0)),
            )
            for event in stages
            if str(event.get("name")) in {"draft", "draft_extend"}
            and start_us <= float(event.get("ts", 0.0)) < window.end_us
        ]
        return sum(
            "_fused_qk_rmsnorm_rope_gate_kernel"
            in str(kernel.get("name", "")).lower()
            and any(
                start <= float(kernel.get("ts", 0.0)) < end
                for start, end in ranges
            )
            for kernel in kernels
        )

    if draft_anchor_count(window.start_us) == 5:
        return window
    preceding = sorted(
        (
            event
            for event in stages
            if event.get("name") == "draft"
            and float(event.get("ts", 0.0)) + float(event.get("dur", 0.0))
            <= window.start_us
        ),
        key=lambda event: float(event.get("ts", 0.0)),
        reverse=True,
    )
    for event in preceding:
        start_us = float(event["ts"])
        if draft_anchor_count(start_us) == 5:
            return ForwardWindow(
                start_us=start_us,
                end_us=window.end_us,
                iter_bounds_us=[(start_us, window.end_us)],
                anchor_kernel_count=window.anchor_kernel_count,
            )
    raise ValueError(f"rank {rank}: eager window does not contain five MTP draft rounds")


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
    eager_signatures: dict[str, str] | None = None,
    contextual_signatures: dict[tuple[str, str, str, int], str] | None = None,
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
    draft_anchor_times = sorted(
        float(kernel.get("ts", 0.0))
        for kernel in kernels
        if _substage(float(kernel.get("ts", 0.0)), ranges)
        in {"draft_extend", "draft"}
        and "_fused_qk_rmsnorm_rope_gate_kernel"
        in str(kernel.get("name", "")).lower()
    )
    if len(draft_anchor_times) != 5:
        raise ValueError(f"expected five MTP draft rounds, got {len(draft_anchor_times)}")

    target_groups: dict[int, list[int]] = {layer_id: [] for layer_id in range(60)}
    for event_index, kernel in enumerate(kernels):
        ts_us = float(kernel.get("ts", 0.0))
        if _substage(ts_us, ranges) != "target_verify":
            continue
        layer_id = bisect_right(anchor_times, ts_us) - 1
        if layer_id >= 0:
            target_groups[layer_id].append(event_index)
    target_slots: dict[int, tuple[int, int, int]] = {}
    for layer_id, indices in target_groups.items():
        residual_positions = [
            position
            for position, event_index in enumerate(indices)
            if "fused_add_rmsnorm" in str(kernels[event_index].get("name", "")).lower()
        ]
        # Minimal synthetic/unit traces may omit both boundaries entirely; in
        # that case structural transfer is simply disabled.  A partial or
        # malformed measured layer still fails closed.
        if not residual_positions:
            continue
        if len(residual_positions) != 2:
            raise ValueError(
                f"target layer {layer_id} has {len(residual_positions)} "
                "residual/RMSNorm boundaries; expected exactly two"
            )
        for position, event_index in enumerate(indices):
            target_slots[event_index] = (
                position,
                residual_positions[0],
                residual_positions[1],
            )

    draft_groups: dict[int, list[int]] = {round_id: [] for round_id in range(5)}
    for event_index, kernel in enumerate(kernels):
        ts_us = float(kernel.get("ts", 0.0))
        if _substage(ts_us, ranges) not in {"draft", "draft_extend"}:
            continue
        round_id = max(0, bisect_right(draft_anchor_times, ts_us) - 1)
        draft_groups[round_id].append(event_index)
    draft_slots: dict[int, tuple[int, int, int, int, int | None]] = {}
    for round_id, indices in draft_groups.items():
        anchor_positions = [
            position
            for position, event_index in enumerate(indices)
            if "_fused_qk_rmsnorm_rope_gate_kernel"
            in str(kernels[event_index].get("name", "")).lower()
        ]
        residual_positions = [
            position
            for position, event_index in enumerate(indices)
            if "fused_add_rmsnorm" in str(kernels[event_index].get("name", "")).lower()
        ]
        if not residual_positions:
            continue
        if len(anchor_positions) != 1 or len(residual_positions) != 2:
            raise ValueError(
                f"MTP round {round_id} has anchors={anchor_positions}, "
                f"residuals={residual_positions}; expected one/two"
            )
        head_position = next(
            (
                position
                for position, event_index in enumerate(indices)
                if position > residual_positions[1]
                and "nvjet_sm103_tst_"
                in str(kernels[event_index].get("name", "")).lower()
            ),
            None,
        )
        for position, event_index in enumerate(indices):
            draft_slots[event_index] = (
                position,
                anchor_positions[0],
                residual_positions[0],
                residual_positions[1],
                head_position,
            )

    mapped: list[dict[str, Any]] = []
    context_counts: Counter[tuple[tuple[str, int], str]] = Counter()
    for event_index, kernel in enumerate(kernels):
        ts_us = float(kernel.get("ts", 0.0))
        name = str(kernel.get("name", ""))
        substage = _substage(ts_us, ranges)
        mtp_round = None
        if substage in {"draft_extend", "draft"}:
            mtp_round = max(0, bisect_right(draft_anchor_times, ts_us) - 1)
        layer_id = None
        layer_kind = None
        if substage == "target_verify":
            anchor_index = bisect_right(anchor_times, ts_us) - 1
            if anchor_index >= 0:
                layer_id = anchor_index
                layer_kind = anchors[anchor_index][1]

        context_group = None
        if substage == "target_verify" and layer_id is not None:
            context_group = (substage, layer_id)
        elif substage in {"draft", "draft_extend"} and mtp_round is not None:
            context_group = (substage, mtp_round)
        contextual_node = None
        if context_group is not None:
            counter_key = (context_group, name)
            ordinal = context_counts[counter_key]
            context_counts[counter_key] += 1
            contextual_node = (contextual_signatures or {}).get(
                (substage, str(layer_kind or ""), name, ordinal)
            )

        direct = direct_graph_mapping(name, substage=substage, layer_kind=layer_kind)
        if direct is None and event_index in target_slots and layer_id is not None:
            position, first_residual, second_residual = target_slots[event_index]
            direct = _target_structural_mapping(
                name,
                layer_id=layer_id,
                layer_kind=str(layer_kind),
                position=position,
                first_residual=first_residual,
                second_residual=second_residual,
            )
        if direct is None and event_index in draft_slots:
            (
                position,
                attention_anchor,
                first_residual,
                second_residual,
                head_position,
            ) = draft_slots[event_index]
            direct = _draft_structural_mapping(
                name,
                position=position,
                attention_anchor=attention_anchor,
                first_residual=first_residual,
                second_residual=second_residual,
                head_position=head_position,
            )
        if direct is None:
            direct = _eager_signature_mapping(name, eager_signatures)
        if direct is None and contextual_node is not None:
            direct = GraphMapping(
                contextual_node,
                "Exact kernel signature + repeated layer/round slot proven by eager stack",
                "mapped",
                "medium",
                (),
                "exact_eager_context_slot",
            )
        if direct is None:
            direct = _unmapped_mapping(substage=substage, layer_kind=layer_kind)

        ir_targets = list(direct.ir_targets)
        if substage == "target_verify" and not str(direct.node).startswith(
            "generation_loop."
        ):
            ir_targets.append("generation_loop.target_verify")
        if substage in {"draft_extend", "draft"}:
            ir_targets.append("generation_loop.draft_propose")
        mapped.append(
            {
                "event_id": f"r{rank}-s{step_index}-k{event_index}",
                "engine": "sglang",
                "rank": rank,
                "step_index": step_index,
                "kernel_name": name,
                "kernel_label": direct.label,
                "node": direct.node,
                "ir_targets": list(dict.fromkeys(ir_targets)),
                "mapping_status": direct.status,
                "fusion_group": (
                    f"r{rank}-s{step_index}-k{event_index}"
                    if direct.status == "fusion"
                    else None
                ),
                "attribution_method": (
                    direct.attribution_method
                    or (
                        "unique_kernel_signature"
                        if direct.status == "mapped"
                        else "kernel_signature_fusion"
                        if direct.status == "fusion"
                        else "unresolved"
                    )
                ),
                "confidence": direct.confidence,
                "unmapped_reason": direct.unmapped_reason,
                "candidate_nodes": list(direct.candidate_nodes),
                "substage": substage,
                "mtp_round": mtp_round,
                "layer_id": layer_id,
                "layer_kind": layer_kind,
                "ts_us": ts_us,
                "dur_us": float(kernel.get("dur", 0.0)),
                "stream": (kernel.get("args") or {}).get("stream"),
                "device": (kernel.get("args") or {}).get("device"),
                "correlation_id": (kernel.get("args") or {}).get("correlation"),
                "graph_id": (kernel.get("args") or {}).get("graph_id"),
                "graph_node_id": (kernel.get("args") or {}).get("graph_node_id"),
                "graph_role": (kernel.get("args") or {}).get("graph_role"),
                "pid": kernel.get("pid"),
                "tid": kernel.get("tid"),
            }
        )

    counts = {
        "target_gdn_layers": sum(kind == "gdn" for _ts, kind in anchors),
        "target_attention_layers": sum(kind == "attention" for _ts, kind in anchors),
        "target_ep4_dispatch": sum(
            event["substage"] == "target_verify"
            and "moea2adispatchkernel" in event["kernel_name"].lower()
            for event in mapped
        ),
        "target_ep4_combine": sum(
            event["substage"] == "target_verify"
            and "moea2acombinekernel" in event["kernel_name"].lower()
            for event in mapped
        ),
        "draft_deepep_dispatch": sum(
            event["substage"] in {"draft", "draft_extend"}
            and
            "deep_ep::" in event["kernel_name"].lower()
            and "dispatch" in event["kernel_name"].lower()
            for event in mapped
        ),
        "draft_deepep_combine": sum(
            event["substage"] in {"draft", "draft_extend"}
            and
            "deep_ep::" in event["kernel_name"].lower()
            and "combine" in event["kernel_name"].lower()
            for event in mapped
        ),
        "gdn_replay": sum(
            "replayssm" in event["kernel_name"].lower() for event in mapped
        ),
        "mtp_draft_rounds": len(draft_anchor_times),
    }
    total_us = sum(event["dur_us"] for event in mapped)
    by_status = {
        status: sum(event["dur_us"] for event in mapped if event["mapping_status"] == status)
        for status in ("mapped", "fusion", "unmapped")
    }
    strict_signature_us = sum(
        float(event["dur_us"])
        for event in mapped
        if event.get("attribution_method") == "unique_kernel_signature"
    )
    validation = {
        "kernel_count": len(mapped),
        "duration_us": total_us,
        "status_duration_us": by_status,
        "attributed_duration_ratio": (
            min(1.0, (by_status["mapped"] + by_status["fusion"]) / total_us)
            if total_us
            else 0.0
        ),
        "strict_signature_duration_ratio": (
            strict_signature_us / total_us if total_us else 0.0
        ),
        "timeline_interval_coverage_ratio": (
            sum(by_status.values()) / total_us if total_us else 0.0
        ),
        "signature_counts": counts,
        "target_verify_batch_size": target_verify_batch_size(trace_events, window),
    }
    return mapped, validation


def load_contextual_eager_signatures(
    *, trace_path: Path, mapping_path: Path
) -> dict[tuple[str, str, str, int], str]:
    """Build repeated layer/round slot signatures from one exact eager run.

    A slot is admitted only when every repeated occurrence has high/medium
    occurrence-local stack evidence for the same leaf: 45/45 GDN layers,
    15/15 full-attention layers, or 4/4 regular MTP draft rounds. The single
    draft-extend round and lifecycle ranges never receive contextual transfer.
    """

    root = mapping_path.parent
    manifest = json.loads((root / "input_manifest.json").read_text())
    if Path(str(manifest["trace_path"])).name != trace_path.name:
        raise ValueError(
            f"context trace {trace_path.name} does not match {manifest['trace_path']}"
        )
    window_data = manifest["window"]
    window = ForwardWindow(
        float(window_data["start_us"]),
        float(window_data["end_us"]),
        [tuple(map(float, bounds)) for bounds in window_data["iter_bounds_us"]],
        int(window_data["anchor_kernel_count"]),
    )
    trace_events = load_trace(trace_path).get("traceEvents") or []
    contextual_events, _validation = map_graph_window(
        trace_events,
        window=window,
        rank=int(manifest["rank"]),
        step_index=0,
    )
    occurrence_events = [
        json.loads(line)
        for line in (root / f"events.tp{manifest['rank']}.jsonl").read_text().splitlines()
        if line
    ]
    stack_rows = [
        json.loads(line) for line in mapping_path.read_text().splitlines() if line
    ]
    if len(occurrence_events) != len(stack_rows):
        raise ValueError("eager stack mapping does not cover each occurrence exactly once")

    common_prefix = 0
    for context_event, occurrence_event in zip(contextual_events, occurrence_events):
        if context_event["kernel_name"] != occurrence_event["kernel_name"]:
            break
        common_prefix += 1
    if common_prefix < 0.95 * min(len(contextual_events), len(occurrence_events)):
        raise ValueError(
            "contextual eager trace and stack mapping lack a stable exact prefix: "
            f"prefix={common_prefix}, context={len(contextual_events)}, "
            f"stack={len(occurrence_events)}"
        )

    contextual_events = contextual_events[:common_prefix]
    stack_rows = stack_rows[:common_prefix]
    keys = _contextual_keys(contextual_events)
    observations: dict[tuple[str, str, str, int], list[str | None]] = {}
    for key, context_event, row in zip(keys, contextual_events, stack_rows):
        if key is None:
            continue
        confidence = str(row.get("confidence") or "unmapped")
        selected = _migrate_eager_node(str(row.get("selected_node") or ""))
        candidate: str | None = (
            selected
            if selected
            and confidence in {"high", "medium"}
            and selected not in NON_LEAF_EAGER_NODES
            and not selected.startswith(("generation_loop.", "top."))
            else None
        )
        direct = direct_graph_mapping(
            context_event["kernel_name"],
            substage=context_event["substage"],
            layer_kind=context_event.get("layer_kind"),
        )
        if (
            candidate is not None
            and direct is not None
            and candidate not in {direct.node, *direct.ir_targets}
        ):
            candidate = None
        observations.setdefault(key, []).append(candidate)

    output: dict[tuple[str, str, str, int], str] = {}
    for key, nodes in observations.items():
        substage, layer_kind, _name, _ordinal = key
        required = (
            45
            if substage == "target_verify" and layer_kind == "gdn"
            else 15
            if substage == "target_verify" and layer_kind == "attention"
            else 4
            if substage == "draft"
            else None
        )
        unique = set(nodes)
        if required is not None and len(nodes) == required and len(unique) == 1 and None not in unique:
            output[key] = str(next(iter(unique)))
    return output


def map_prefill_window(
    trace_events: list[dict[str, Any]],
    *,
    start_us: float,
    end_us: float,
    rank: int,
    step_index: int,
    mtp_seed_start_us: float | None = None,
    eager_signatures: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Map one complete target prefill, optionally including its MTP seed."""

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
    target_end_us = mtp_seed_start_us if mtp_seed_start_us is not None else end_us
    anchors = _target_anchors(
        kernels, target_start=start_us, target_end=target_end_us
    )
    anchor_times = [item[0] for item in anchors]
    mapped: list[dict[str, Any]] = []
    status_us: Counter[str] = Counter()
    for event_index, kernel in enumerate(kernels):
        ts_us = float(kernel.get("ts", 0.0))
        name = str(kernel.get("name", ""))
        mtp_seed = mtp_seed_start_us is not None and ts_us >= mtp_seed_start_us
        layer_id = None if mtp_seed else max(0, bisect_right(anchor_times, ts_us) - 1)
        layer_kind = "attention" if mtp_seed else anchors[layer_id][1]
        direct = direct_graph_mapping(
            name,
            substage="draft_extend" if mtp_seed else "target_prefill",
            layer_kind=layer_kind,
        )
        if direct is None:
            direct = _eager_signature_mapping(name, eager_signatures)
        if direct is None:
            direct = _unmapped_mapping(
                substage="draft_extend" if mtp_seed else "target_prefill",
                layer_kind=layer_kind,
            )
        duration_us = float(kernel.get("dur", 0.0))
        status_us[direct.status] += duration_us
        targets = list(direct.ir_targets)
        if mtp_seed:
            targets.append("generation_loop.draft_propose")
        mapped.append(
            {
                "event_id": f"r{rank}-p{step_index}-k{event_index}",
                "engine": "sglang",
                "rank": rank,
                "step_index": step_index,
                "kernel_name": name,
                "kernel_label": direct.label,
                "node": direct.node,
                "ir_targets": list(dict.fromkeys(targets)),
                "mapping_status": direct.status,
                "fusion_group": (
                    f"r{rank}-p{step_index}-k{event_index}"
                    if direct.status == "fusion"
                    else None
                ),
                "attribution_method": (
                    direct.attribution_method
                    or (
                        "unique_kernel_signature"
                        if direct.status == "mapped"
                        else "kernel_signature_fusion"
                        if direct.status == "fusion"
                        else "unresolved"
                    )
                ),
                "confidence": direct.confidence,
                "unmapped_reason": direct.unmapped_reason,
                "candidate_nodes": list(direct.candidate_nodes),
                "layer_id": layer_id,
                "layer_kind": layer_kind,
                "substage": "mtp_seed_prefill" if mtp_seed else "target_prefill",
                "mtp_round": 0 if mtp_seed else None,
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
            if float(kernel.get("ts", 0.0)) < target_end_us
        ),
        "target_attention_layers": sum(
            "_fused_qk_rmsnorm_rope_gate_kernel"
            in str(kernel.get("name", "")).lower()
            for kernel in kernels
            if float(kernel.get("ts", 0.0)) < target_end_us
        ),
        "target_ep4_dispatch": sum(
            "moea2adispatchkernel" in str(kernel.get("name", "")).lower()
            for kernel in kernels
            if float(kernel.get("ts", 0.0)) < target_end_us
        ),
        "target_ep4_combine": sum(
            "moea2acombinekernel" in str(kernel.get("name", "")).lower()
            for kernel in kernels
            if float(kernel.get("ts", 0.0)) < target_end_us
        ),
        "mtp_seed_attention_layers": sum(
            event["substage"] == "mtp_seed_prefill"
            and "_fused_qk_rmsnorm_rope_gate_kernel"
            in event["kernel_name"].lower()
            for event in mapped
        ),
        "mtp_seed_ep4_dispatch": sum(
            event["substage"] == "mtp_seed_prefill"
            and event["node"] == "mtp_moe_block.draft_ep4_dispatch"
            for event in mapped
        ),
        "mtp_seed_ep4_combine": sum(
            event["substage"] == "mtp_seed_prefill"
            and event["node"] == "mtp_moe_block.draft_ep4_combine"
            for event in mapped
        ),
    }
    expected = {
        "target_gdn_layers": 45,
        "target_attention_layers": 15,
        "target_ep4_dispatch": 60,
        "target_ep4_combine": 60,
    }
    if mtp_seed_start_us is not None:
        expected.update(
            {
                "mtp_seed_attention_layers": 1,
                "mtp_seed_ep4_dispatch": 2,
                "mtp_seed_ep4_combine": 2,
            }
        )
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
        "timeline_interval_coverage_ratio": (
            sum(status_us.values()) / total_us if total_us else 0.0
        ),
        "timing_closure_us": sum(float(event["dur_us"]) for event in mapped) - total_us,
    }
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
        legacy_node = _legacy_eager_node(node)
        candidates = index["exact"].get((node, name), []) or index["exact"].get(
            (legacy_node, name), []
        )
        match = "exact_kernel_name_and_ir_node"
        source_node = node
        if not candidates:
            candidates = index["by_node"].get(node, []) or index["by_node"].get(
                legacy_node, []
            )
            match = "representative_ir_node_stack"
        if not candidates and node in EAGER_SCOPE_FALLBACKS:
            source_node = EAGER_SCOPE_FALLBACKS[node]
            candidates = index["by_node"].get(source_node, []) or index["by_node"].get(
                _legacy_eager_node(source_node), []
            )
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
