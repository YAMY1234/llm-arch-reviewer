#!/usr/bin/env python3
"""Build Qwen 4.0 EAGLE-MTP eager profiles from stack and timing traces."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.common.timeline_artifact import (  # noqa: E402
    build_timeline_artifact,
    write_timeline_artifact,
)
from models.qwen40.build.build_qwen40_eager_prefill_profile import (  # noqa: E402
    _transfer_chunk_timing,
    fallback_prefill_node,
    stack_text,
)
from models.qwen40.build.qwen40_decode_attribution import (  # noqa: E402
    _metric,
    attach_qsa_indexer_drill_metrics,
    attach_qsa_indexer_drill_targets,
    communication_semantics,
    default_node_states,
    direct_kernel_mapping,
    interval_union_us,
    metrics_for_rank,
)


BASE_SOURCE_COMMIT = "32e9cb5b95104dc3a10b96bafae7afa50052d94d"
SOURCE_COMMIT = "32e9cb5b95104dc3a10b96bafae7afa50052d94d"
MODEL_REVISION = "b151fd157ff99b63198ab8558432f0bf43e14d58"
SOURCE_PATCH_SHA256 = "07c22e094da7103011301ced5824134e0387b310a5a03df0579bdd7ed08f17b3"
SOURCE_PATCH_COMPONENTS = [
    {"name": "qsa_hardening", "sha256": SOURCE_PATCH_SHA256},
]
IMPLEMENTATION_ID = "sglang_qwen4_main_32e9cb5_qsa_hardening_flashinfer_gdn"
LAYER_MODULE = re.compile(r"Qwen4Exp(?:Linear|Attention)DecoderLayer_(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prefill", "decode"), required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--semantic-events", type=Path, required=True)
    parser.add_argument("--semantic-mapping", type=Path, required=True)
    parser.add_argument("--semantic-manifest", type=Path, required=True)
    parser.add_argument("--semantic-validation", type=Path, required=True)
    parser.add_argument("--semantic-protocol", type=Path, required=True)
    parser.add_argument("--timing-events", type=Path, required=True)
    parser.add_argument("--timing-manifest", type=Path, required=True)
    parser.add_argument("--timing-protocol", type=Path, required=True)
    parser.add_argument("--semantic-job-id", required=True)
    parser.add_argument("--timing-job-id", required=True)
    parser.add_argument("--output-profile", type=Path, required=True)
    parser.add_argument("--output-analysis", type=Path, required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bounds(manifest: dict[str, Any]) -> list[tuple[float, float]]:
    return [
        (float(start), float(stop))
        for start, stop in manifest.get("window", {}).get("iter_bounds_us", [])
    ]


def _split_events(
    events: list[dict[str, Any]], manifest: dict[str, Any]
) -> list[list[dict[str, Any]]]:
    bounds = _bounds(manifest)
    if not bounds:
        raise ValueError("trace manifest has no selected iteration bounds")
    chunks: list[list[dict[str, Any]]] = [[] for _ in bounds]
    for event in events:
        timestamp = float(event["ts_us"])
        matches = [
            index
            for index, (start, stop) in enumerate(bounds)
            if start <= timestamp <= stop
        ]
        if len(matches) != 1:
            raise ValueError(
                f"kernel timestamp {timestamp} belongs to {len(matches)} selected steps"
            )
        chunks[matches[0]].append(event)
    return chunks


def _layer_context(event: dict[str, Any], node: str) -> tuple[int | None, str | None]:
    lowered = stack_text(event).lower()
    if (
        node.startswith(("mtp_head.", "mtp_layer.", "mtp_qsa_attention.", "mtp_moe."))
        or "qwen4_exp_mtp.py" in lowered
        or "qwen4expforcausallmmtp" in lowered
    ):
        return 0, "mtp"
    match = LAYER_MODULE.search(stack_text(event))
    if match:
        layer_id = int(match.group(1))
        return layer_id, "full" if layer_id % 4 == 3 else "linear"
    if node.startswith(("linear_attention.", "linear_layer.")):
        return None, "linear"
    if node.startswith(("qsa_attention.", "full_layer.")):
        return None, "full"
    return None, None


def _substage(event: dict[str, Any], node: str, phase: str) -> str | None:
    lowered = stack_text(event).lower()
    mtp_phase = "mtp_prefill" if phase == "prefill" else "mtp_draft_extend"
    if node.startswith("mtp_generation."):
        return node.split(".", 1)[1]
    if node.startswith("mtp_qsa_attention."):
        return f"{mtp_phase}_attention"
    if node.startswith("mtp_moe."):
        return f"{mtp_phase}_moe"
    if node.startswith("mtp_layer."):
        return f"{mtp_phase}_{node.split('.', 1)[1]}"
    if node.startswith("mtp_head."):
        return mtp_phase
    if node.startswith("hyperconnection.") and (
        "qwen4_exp_mtp.py" in lowered or "qwen4expforcausallmmtp" in lowered
    ):
        if "_prepare_qwen4_exp_attn" in lowered:
            stage = "attn_hc_mix"
        elif "_prepare_qwen4_exp_mlp" in lowered:
            stage = "attn_hc_combine" if node.endswith("combine") else "mlp_hc_mix"
        elif "_postprocess_qwen4_exp_layer" in lowered:
            stage = "mlp_hc_combine"
        else:
            return mtp_phase
        return f"{mtp_phase}_{stage}"
    if node.startswith("ple."):
        return "ple"
    if node.startswith(("linear_attention.", "qsa_attention.")):
        return "attention"
    if node.startswith("moe.") or node.endswith("_moe_output_collective"):
        return "moe"
    if node.startswith("hyperconnection."):
        if "_prepare_qwen4_exp_attn" in lowered:
            return "attn_hc_mix"
        if "_prepare_qwen4_exp_mlp" in lowered:
            return "attn_hc_combine" if node.endswith("combine") else "mlp_hc_mix"
        if "_postprocess_qwen4_exp_layer" in lowered:
            return "mlp_hc_combine"
    return None


def semantic_events(args: argparse.Namespace) -> list[dict[str, Any]]:
    raw_events = load_jsonl(args.semantic_events)
    mappings = load_jsonl(args.semantic_mapping)
    if len(raw_events) != len(mappings):
        raise ValueError("semantic event and mapping lengths differ")
    semantic_rank = int(json.loads(args.semantic_manifest.read_text()).get("rank", 0))
    attributed = []
    for event, mapping in zip(raw_events, mappings):
        if event.get("event_id") != mapping.get("event_id"):
            raise ValueError("semantic event/mapping IDs are not aligned")
        node = mapping.get("selected_node")
        confidence = str(mapping.get("confidence") or "unmapped")
        method = "python_stack_ir_rule"
        label = None
        semantic = mapping.get("semantic_frame") or mapping.get("operator_frame") or {}
        lowered_stack = stack_text(event).lower()
        direct_node, direct_label = direct_kernel_mapping(str(event["kernel_name"]))
        if (
            node is not None
            and str(node).startswith("hyperconnection.")
            and "layers/hyperconnection.py" in lowered_stack
        ):
            # Several prefill HC kernels reuse generic GEMM/epilogue signatures
            # that also occur in MoE. Their direct Python HC frame is stronger
            # evidence than a global signature rule.
            confidence = "high"
        elif direct_node is not None:
            node = direct_node
            label = direct_label
            confidence = "high"
            method = "direct_signature_with_python_stack"
            if "qwen4_exp_mtp.py" in lowered_stack or "qwen4expforcausallmmtp" in lowered_stack:
                if node.startswith("qsa_attention."):
                    node = "mtp_qsa_attention." + node.split(".", 1)[1]
                elif node.startswith("moe."):
                    node = "mtp_moe." + node.split(".", 1)[1]
        elif node is None:
            node, label, confidence = fallback_prefill_node(event, mapping)
            method = "python_stack_semantic_fallback"
        semantic_function = str(semantic.get("function") or "")
        if (
            semantic_function in {"get_decode_mqa_inputs", "get_seqlens_expanded"}
            and str(node).startswith(("qsa_attention.", "mtp_qsa_attention."))
        ):
            node = "qsa_attention.metadata"
            label = "QSA layout and valid-count metadata"
            confidence = "high"
            method = "python_stack_semantic_function"
        if "replicatedlinear" in lowered_stack and "_forward_router_experts" in lowered_stack:
            node = "moe.router"
            label = "replicated MoE router projection"
            confidence = "high"
            method = "python_stack_semantic_function"
        node = str(node)
        layer_id, layer_kind = _layer_context(event, node)
        if layer_kind == "mtp":
            if node.startswith("qsa_attention."):
                node = "mtp_qsa_attention." + node.split(".", 1)[1]
            elif node.startswith("moe."):
                node = "mtp_moe." + node.split(".", 1)[1]
        label = label or str(
            semantic.get("function") or event.get("cpu_op_name") or node
        )
        attributed.append(
            {
                "rank": semantic_rank,
                "step_index": 1,
                "kernel_name": event["kernel_name"],
                "kernel_label": label,
                "node": node,
                "ts_us": float(event["ts_us"]),
                "dur_us": float(event["dur_us"]),
                "stream": event.get("stream"),
                "device": event.get("device"),
                "pid": event.get("pid"),
                "tid": event.get("tid"),
                "layer_id": layer_id,
                "layer_kind": layer_kind,
                "invocation_id": (
                    f"mtp:{layer_id}" if layer_kind == "mtp" else layer_id
                ),
                "substage": _substage(event, node, args.phase),
                "attribution_method": method,
                "confidence": confidence,
                "cpu_op_name": event.get("cpu_op_name"),
                "semantic_function": semantic_function,
                "python_stack": event.get("python_stack") or [],
                "stack_evidence": {
                    "source": "eager_mtp_trace",
                    "match": "direct_event",
                    "kind": "full_eager_python_stack",
                    "event_id": event.get("event_id"),
                    "confidence": confidence,
                },
            }
        )
    _reconcile_hyperconnection_structure(attributed, phase=args.phase)
    _reconcile_mtp_input_fusion(attributed)
    _reconcile_mtp_moe_structure(attributed, phase=args.phase)
    return attributed


def _reconcile_mtp_input_fusion(events: list[dict[str, Any]]) -> None:
    """Split the source-ordered five-kernel MTP residual-linear fusion."""

    indices = [
        index
        for index, event in enumerate(events)
        if event.get("node") == "mtp_head.residual_fusion"
        and event.get("semantic_function") == "_fuse_residual_linear_shared"
    ]
    if len(indices) != 5 or indices != list(range(indices[0], indices[0] + 5)):
        raise ValueError(
            "MTP input fusion must be one contiguous "
            f"RMSNorm/GEMM/RMSNorm/GEMM/add block; got {indices}"
        )
    names = [str(events[index]["kernel_name"]).lower() for index in indices]
    if not (
        "rmsnormkernel" in names[0]
        and "rmsnormkernel" in names[2]
        and "cudafunctor_add" in names[4]
    ):
        raise ValueError("MTP input fusion kernel order changed")
    assignments = (
        (indices[:2], "mtp_head.embedding_projection"),
        (indices[2:4], "mtp_head.hidden_projection"),
        (indices[4:], "mtp_head.residual_fusion"),
    )
    for members, node in assignments:
        for index in members:
            events[index].update(
                {
                    "node": node,
                    "attribution_method": "mtp_input_fusion_source_order",
                    "confidence": "high",
                    "stack_evidence": {
                        "source": "eager_mtp_trace",
                        "match": "_fuse_residual_linear_shared kernel order",
                        "kind": "source_reviewed_fusion_substage",
                        "confidence": "high",
                    },
                }
            )


def _reconcile_mtp_moe_structure(
    events: list[dict[str, Any]], *, phase: str
) -> None:
    """Recover reused routed-expert kernels inside exact MTP MoE boundaries.

    At large batch sizes the fused expert GEMMs can expose only a generic MoE
    Python frame even though the adjacent top-k and finalize kernels retain
    their MTP scope. Translate only a contiguous top-k -> routed expert ->
    combine sequence; any unrelated kernel closes the candidate block.
    """

    allowed = {
        "mtp_moe.topk",
        "mtp_moe.routed_experts",
        "moe.routed_experts",
        "mtp_moe.combine",
    }
    mtp_phase = "mtp_prefill" if phase == "prefill" else "mtp_draft_extend"
    index = 0
    while index < len(events):
        if events[index].get("node") != "mtp_moe.topk":
            index += 1
            continue
        start = index
        stop = start
        while stop < len(events) and events[stop].get("node") in allowed:
            if events[stop].get("node") == "mtp_moe.combine":
                break
            stop += 1
        if stop >= len(events) or events[stop].get("node") != "mtp_moe.combine":
            index = max(start + 1, stop)
            continue
        generic = [
            event
            for event in events[start + 1 : stop]
            if event.get("node") == "moe.routed_experts"
        ]
        if generic:
            owner = events[start]
            for event in generic:
                event.update(
                    {
                        "node": "mtp_moe.routed_experts",
                        "layer_id": owner.get("layer_id", 0),
                        "layer_kind": "mtp",
                        "invocation_id": owner.get("invocation_id") or "mtp:0",
                        "substage": f"{mtp_phase}_moe",
                        "attribution_method": str(event["attribution_method"])
                        + "+mtp_moe_boundary_context",
                        "confidence": "high",
                    }
                )
        index = stop + 1


def _reconcile_hyperconnection_structure(
    events: list[dict[str, Any]], *, phase: str
) -> None:
    """Recover HC stage identity from the stable mix/combine dataflow order.

    The persistent mix launch has a unique kernel signature, but its Python
    launch stack is often rooted in a fused model ``forward`` and does not keep
    the small ``_prepare_*`` helper frame. Every layer combine does retain its
    exact layer/stage stack, and the corresponding mix is the last mix boundary
    since the preceding combine. Pairing those two explicit HC boundaries is
    therefore stronger than borrowing the nearest model-module stack.
    """

    combine_indices = [
        index
        for index, event in enumerate(events)
        if "hc_combine_kernel" in str(event["kernel_name"]).lower()
    ]
    def is_mix_member(event: dict[str, Any]) -> bool:
        return (
            "_hc_mix_persistent_kernel" in str(event["kernel_name"]).lower()
            or (
                str(event.get("semantic_function") or "") in {"mix", "_mix_compute"}
                and "layers/hyperconnection.py" in stack_text(event).lower()
            )
        )

    mix_norm_indices = [
        index
        for index, event in enumerate(events[:-1])
        if "grouped_gemma_rmsnorm_kernel" in str(event["kernel_name"]).lower()
        and is_mix_member(events[index + 1])
    ]

    def mix_group(norm_index: int) -> list[int]:
        members = []
        index = norm_index + 1
        while index < len(events) and is_mix_member(events[index]):
            members.append(index)
            index += 1
        if not members:
            raise ValueError("HC branch norm has no adjacent mix implementation")
        return members

    paired_norms: set[int] = set()
    previous_combine = -1
    for combine_index in combine_indices:
        candidates = [
            index
            for index in mix_norm_indices
            if previous_combine < index < combine_index and index not in paired_norms
        ]
        if not candidates:
            raise ValueError(
                "HC combine has no preceding mix boundary: "
                f"previous={previous_combine} combine={combine_index}"
            )
        norm_index = candidates[-1]
        paired_norms.add(norm_index)
        combine = events[combine_index]
        combine_stage = str(combine.get("substage") or "")
        if not combine_stage.endswith("_hc_combine"):
            raise ValueError("HC combine stack does not identify its layer stage")
        members = [norm_index, *mix_group(norm_index)]
        for member in members:
            events[member].update(
                {
                    "node": (
                        "hyperconnection.branch_norm"
                        if member == norm_index
                        else "hyperconnection.mix"
                    ),
                    "layer_id": combine.get("layer_id"),
                    "layer_kind": combine.get("layer_kind"),
                    "invocation_id": combine.get("invocation_id"),
                    "substage": combine_stage.removesuffix("_combine") + "_mix",
                    "attribution_method": "hc_mix_combine_boundary_pair",
                    "confidence": "high",
                    "stack_evidence": {
                        "source": "eager_mtp_trace",
                        "match": "paired_with_following_hc_combine_boundary",
                        "kind": "stable_hyperconnection_dataflow_order",
                        "confidence": "high",
                        "combine_event_id": combine.get("stack_evidence", {}).get(
                            "event_id"
                        ),
                    },
                }
            )
        previous_combine = combine_index

    final_norm_indices = [
        index for index in mix_norm_indices if index not in paired_norms
    ]
    if len(final_norm_indices) != 2:
        raise ValueError(
            "target + one-layer MTP trace must contain exactly two final HC mixes; "
            f"found {len(final_norm_indices)}"
        )
    mtp_phase = "mtp_prefill" if phase == "prefill" else "mtp_draft_extend"
    final_scopes = (
        (final_norm_indices[0], "top.final_hc_mix", None, None, None, "target_final_hc_mix"),
        (
            final_norm_indices[1],
            "mtp_head.final_hc_mix",
            0,
            "mtp",
            "mtp:0",
            mtp_phase,
        ),
    )
    for scope_index, (
        norm_index,
        node,
        layer_id,
        layer_kind,
        invocation_id,
        substage,
    ) in enumerate(final_scopes):
        final_mix_members = mix_group(norm_index)
        for member in (norm_index, *final_mix_members):
            events[member].update(
                {
                    "node": node,
                    "layer_id": layer_id,
                    "layer_kind": layer_kind,
                    "invocation_id": invocation_id,
                    "substage": substage,
                    "attribution_method": "final_hc_boundary_structure",
                    "confidence": "high",
                }
            )
        upper_bound = (
            final_scopes[scope_index + 1][0]
            if scope_index + 1 < len(final_scopes)
            else len(events)
        )
        collective_indices = [
            index
            for index in range(final_mix_members[-1] + 1, upper_bound)
            if "_all_gather_kernel_inner"
            in str(events[index]["kernel_name"]).lower()
        ]
        if len(collective_indices) != 1:
            raise ValueError(
                f"{node} must be followed by exactly one logits all-gather; "
                f"found {len(collective_indices)}"
            )
        collective_index = collective_indices[0]
        lm_head_node = "mtp_head.lm_head" if layer_kind == "mtp" else "top.lm_head"
        collective_node = (
            "mtp_head.tp_logits_collective"
            if layer_kind == "mtp"
            else "top.tp_logits_collective"
        )
        for member in range(final_mix_members[-1] + 1, collective_index):
            events[member].update(
                {
                    "node": lm_head_node,
                    "layer_id": layer_id,
                    "layer_kind": layer_kind,
                    "invocation_id": invocation_id,
                    "substage": substage,
                    "attribution_method": "final_hc_lm_head_boundary_structure",
                    "confidence": "high",
                }
            )
        events[collective_index].update(
            {
                "node": collective_node,
                "layer_id": layer_id,
                "layer_kind": layer_kind,
                "invocation_id": invocation_id,
                "substage": substage,
                "attribution_method": "final_hc_logits_collective_boundary_structure",
                "confidence": "high",
            }
        )
        copy_index = collective_index + 1
        if copy_index < upper_bound and events[copy_index].get("cpu_op_name") == "aten::copy_":
            events[copy_index].update(
                {
                    "node": lm_head_node,
                    "layer_id": layer_id,
                    "layer_kind": layer_kind,
                    "invocation_id": invocation_id,
                    "substage": substage,
                    "attribution_method": "post_collective_logits_copy_structure",
                    "confidence": "high",
                }
            )


def transfer_timing(
    source: list[dict[str, Any]],
    timing: list[dict[str, Any]],
    semantic_manifest: dict[str, Any],
    timing_manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, int]]]:
    source_chunks = _split_events(source, semantic_manifest)
    if len(source_chunks) != 1:
        raise ValueError("semantic mapping must select exactly one MTP iteration")
    timing_chunks = _split_events(timing, timing_manifest)
    transferred = []
    accounting = []
    for step_index, timing_chunk in enumerate(timing_chunks, start=1):
        source_chunk = _append_source_reviewed_mtp_decode_runtime_tail(
            source_chunks[0], timing_chunk, phase=str(timing_manifest["phase"])
        )
        chunk, counts = _transfer_chunk_timing(
            source_chunk,
            timing_chunk,
            timing_rank=int(timing_manifest.get("rank", 0)),
            chunk_index=step_index,
            boundary_insert_resolver=lambda event, left, right: (
                _resolve_mtp_generation_boundary_insert(
                    event, left, right, phase=str(timing_manifest["phase"])
                )
            ),
        )
        for event in chunk:
            event["step_index"] = step_index
            event.pop("prefill_chunk_index", None)
        _restore_mtp_scope_for_timing_inserts(chunk, phase=str(timing_manifest["phase"]))
        transferred.extend(chunk)
        accounting.append(counts)
    return transferred, accounting


_MTP_DECODE_RUNTIME_TAIL_PATTERNS = (
    "direct_copy_kernel_cuda",
    "index_elementwise_kernel",
    "_gather_rows_kernel",
    "assign_draft_cache_locs_contiguous",
    "fillfunctor",
    "cudafunctor_add",
    "catarraybatchedcopy",
    "build_tree_efficient",
    "assign_extend_cache_locs_uniform",
)

_MTP_DECODE_RUNTIME_TAIL_ANCHORS = (
    "_gather_rows_kernel",
    "assign_draft_cache_locs_contiguous",
    "build_tree_efficient",
    "assign_extend_cache_locs_uniform",
)


def _append_source_reviewed_mtp_decode_runtime_tail(
    source_events: list[dict[str, Any]],
    timing_events: list[dict[str, Any]],
    *,
    phase: str,
) -> list[dict[str, Any]]:
    """Add the source-reviewed scheduler tail omitted by a one-step stack trace.

    The stack-on capture stops at the end of ``DRAFT_EXTEND_V2``. A closed
    stack-off decode interval continues until the next ``TARGET_VERIFY`` and
    therefore also contains proposal selection, candidate-tree construction,
    draft-cache assignment, and next-verify cache assignment. These kernels
    are a terminal block in ``prepare_for_draft`` / ``build_eagle_verify_input``
    / ``eagle_prepare_for_verify``. Classify that block as ``draft_select``
    only when all source-reviewed anchors are present and every kernel belongs
    to the reviewed implementation family. Any changed or partial tail remains
    fail-closed in the exact sequence transfer.
    """

    if "decode" not in phase or not source_events or not timing_events:
        return source_events
    terminal_name = str(source_events[-1]["kernel_name"])
    candidates: list[list[dict[str, Any]]] = []
    for index, event in enumerate(timing_events):
        if str(event["kernel_name"]) != terminal_name:
            continue
        tail = timing_events[index + 1 :]
        if not tail:
            continue
        lowered = [str(row["kernel_name"]).lower() for row in tail]
        if not all(
            any(pattern in name for pattern in _MTP_DECODE_RUNTIME_TAIL_PATTERNS)
            for name in lowered
        ):
            continue
        joined = "\n".join(lowered)
        if not all(anchor in joined for anchor in _MTP_DECODE_RUNTIME_TAIL_ANCHORS):
            continue
        candidates.append(tail)
    if not candidates:
        return source_events
    if len(candidates) != 1:
        raise ValueError("MTP decode runtime tail has more than one valid boundary")

    augmented = list(source_events)
    for event in candidates[0]:
        augmented.append(
            {
                "kernel_name": event["kernel_name"],
                "kernel_label": "proposal selection, candidate tree, and next-verify preparation",
                "node": "mtp_generation.draft_select",
                "layer_id": None,
                "layer_kind": None,
                "invocation_id": None,
                "substage": "draft_select",
                "attribution_method": "source_reviewed_speculative_runtime_tail",
                "confidence": "high",
                "cpu_op_name": event.get("cpu_op_name"),
                "python_stack": [],
                "stack_evidence": {
                    "source": "source_review",
                    "match": "terminal_runtime_tail",
                    "kind": "source_reviewed_speculative_control_flow",
                    "confidence": "high",
                    "code_symbols": [
                        "prepare_for_draft",
                        "build_eagle_verify_input",
                        "eagle_prepare_for_verify",
                    ],
                },
            }
        )
    return augmented


def _resolve_mtp_generation_boundary_insert(
    timing_event: dict[str, Any],
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    phase: str,
) -> dict[str, Any] | None:
    """Resolve stack-sensitive target metadata only at a proven stage boundary.

    QSA and GDN build data-dependent metadata before target verification. Their
    generic ATen index/scan kernels differ with accepted-token state, so the
    stack and timing traces need not have an identical name sequence. When the
    exact neighbors prove that the whole gap stays inside those two metadata
    implementations, keep the time at the target runtime-support boundary
    instead of guessing a QSA-vs-GDN leaf.
    """

    runtime_nodes = {
        "qsa_attention.metadata",
        "linear_attention.recurrent_state",
        "top.runtime_support",
    }
    left_node = str(left.get("node"))
    right_node = str(right.get("node"))
    if (
        left_node.endswith("qsa_attention.metadata")
        and right_node.endswith("qsa_attention.indexer")
        and left.get("semantic_function") == "get_decode_mqa_inputs"
        and timing_event.get("cpu_op_name") in {"aten::lt", "aten::any"}
    ):
        return {
            "kernel_label": "QSA decode metadata validity preparation",
            "node": left_node,
            "layer_id": left.get("layer_id"),
            "layer_kind": left.get("layer_kind"),
            "invocation_id": left.get("invocation_id"),
            "substage": left.get("substage"),
            "attribution_method": "qsa_metadata_boundary_context",
            "confidence": "high",
        }
    if {left_node, right_node}.issubset(runtime_nodes):
        stage = "target_prefill_runtime" if "prefill" in phase else "target_verify_runtime"
        return {
            "kernel_label": "target QSA/GDN metadata preparation",
            "node": "top.runtime_support",
            "layer_kind": None,
            "substage": stage,
            "attribution_method": "generation_runtime_boundary_context",
            "confidence": "high",
        }
    return None


def _restore_mtp_scope_for_timing_inserts(
    events: list[dict[str, Any]], *, phase: str
) -> None:
    """Keep no-stack timing-only kernels inside their measured MTP scope.

    Exact sequence matches already copy the Python-stack MTP node. A kernel
    that exists only in the stack-disabled trace can be assigned by a unique
    signature, but that signature alone cannot distinguish the target QSA/MoE
    implementation from the reused auxiliary implementation. We therefore
    allow the direct signature only when its nearest exact neighbors agree on
    the scope, and translate only one-to-one QSA/MoE leaves. Ambiguous HC or
    cross-boundary inserts fail closed for manual inspection.
    """

    phase_name = (
        "prefill" if "prefill" in phase or "extend" in phase else "decode"
    )
    direct_indices = [
        index
        for index, event in enumerate(events)
        if event.get("attribution_method") == "direct_signature_timing_insert"
    ]
    exact_indices = [
        index
        for index, event in enumerate(events)
        if event.get("attribution_method") != "direct_signature_timing_insert"
    ]
    for index in direct_indices:
        left_index = max((value for value in exact_indices if value < index), default=None)
        right_index = min((value for value in exact_indices if value > index), default=None)
        if left_index is None or right_index is None:
            continue
        left = events[left_index]
        right = events[right_index]
        left_mtp = left.get("layer_kind") == "mtp"
        right_mtp = right.get("layer_kind") == "mtp"
        if left_mtp != right_mtp:
            raise ValueError(
                "timing-only direct kernel crosses the target/MTP scope boundary: "
                f"{events[index]['kernel_name']!r}"
            )
        if not left_mtp:
            continue

        event = events[index]
        node = str(event["node"])
        if node.startswith("qsa_attention."):
            node = "mtp_qsa_attention." + node.split(".", 1)[1]
            substage = _substage(event, node, phase_name)
        elif node.startswith("moe."):
            node = "mtp_moe." + node.split(".", 1)[1]
            substage = _substage(event, node, phase_name)
        elif node.startswith("hyperconnection."):
            left_stage = str(left.get("substage") or "")
            right_stage = str(right.get("substage") or "")
            if node.endswith("mix"):
                if left_stage.endswith("attn_hc_mix"):
                    stage = "attn_hc_mix"
                elif left_stage.endswith("mlp_hc_mix"):
                    stage = "mlp_hc_mix"
                elif right_stage.endswith("_attention"):
                    stage = "attn_hc_mix"
                elif right_stage.endswith("_moe"):
                    stage = "mlp_hc_mix"
                else:
                    stage = None
            else:
                if right_stage.endswith("attn_hc_combine"):
                    stage = "attn_hc_combine"
                elif right_stage.endswith("mlp_hc_combine"):
                    stage = "mlp_hc_combine"
                elif left_stage.endswith("_attention"):
                    stage = "attn_hc_combine"
                elif left_stage.endswith("_moe"):
                    stage = "mlp_hc_combine"
                else:
                    stage = None
            if stage is None:
                raise ValueError(
                    "MTP timing-only HC kernel has ambiguous layer stage: "
                    f"{event['kernel_name']!r} between {left_stage!r} and {right_stage!r}"
                )
            mtp_phase = "mtp_prefill" if phase_name == "prefill" else "mtp_draft_extend"
            substage = f"{mtp_phase}_{stage}"
        else:
            raise ValueError(
                "MTP timing-only kernel has no scope-safe direct translation: "
                f"{event['kernel_name']!r} -> {node!r}"
            )
        event.update(
            {
                "node": node,
                "layer_id": 0,
                "layer_kind": "mtp",
                "invocation_id": "mtp:0",
                "substage": substage,
                "attribution_method": "direct_signature_timing_insert+mtp_scope_neighbors",
            }
        )


def _mtp_rollup_groups(
    events: list[dict[str, Any]], phase: str
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    mtp_model_prefixes = ("mtp_head.", "mtp_layer.", "mtp_qsa_attention.", "mtp_moe.")
    mtp_model = [
        event
        for event in events
        if event.get("layer_kind") == "mtp"
        or str(event["node"]).startswith(mtp_model_prefixes)
    ]
    mtp_decoder = [
        event
        for event in mtp_model
        if event.get("layer_kind") == "mtp"
        and not str(event["node"]).startswith("mtp_head.")
        or str(event["node"]).startswith(("mtp_layer.", "mtp_qsa_attention.", "mtp_moe."))
    ]
    groups["mtp_head.decoder_layer"] = mtp_decoder + [
        event
        for event in mtp_model
        if str(event["node"]) == "mtp_head.decoder_layer"
    ]
    groups["mtp_layer.qsa_attention"] = [
        event
        for event in mtp_decoder
        if str(event["node"]).startswith("mtp_qsa_attention.")
        or str(event["node"]) == "mtp_layer.tp_attention_collective"
        or str(event["node"]) == "mtp_layer.qsa_attention"
    ]
    groups["mtp_layer.moe"] = [
        event
        for event in mtp_decoder
        if str(event["node"]).startswith("mtp_moe.")
        or str(event["node"]) == "mtp_layer.tp_moe_output_collective"
        or str(event["node"]) == "mtp_layer.moe"
    ]
    for stage in (
        "attn_hc_mix",
        "attn_hc_combine",
        "mlp_hc_mix",
        "mlp_hc_combine",
    ):
        groups[f"mtp_layer.{stage}"] = [
            event
            for event in mtp_decoder
            if str(event.get("substage") or "").endswith(stage)
        ]
    mtp_stage = "mtp_generation.mtp_prefill" if phase == "prefill" else "mtp_generation.mtp_draft_extend"
    groups[mtp_stage] = mtp_model + [
        event for event in events if str(event["node"]) == mtp_stage
    ]
    target_stage = "mtp_generation.target_prefill" if phase == "prefill" else "mtp_generation.target_verify"
    groups[target_stage] = [
        event
        for event in events
        if event.get("layer_kind") != "mtp"
        and not str(event["node"]).startswith("mtp_")
    ] + [event for event in events if str(event["node"]) == target_stage]
    return {target: rows for target, rows in groups.items() if rows}


def build_metrics(events: list[dict[str, Any]], *, phase: str, n_iters: int) -> dict[str, Any]:
    attach_qsa_indexer_drill_targets(events)
    # Never let the one-layer auxiliary model inflate target-model rollups such
    # as top.decoder_stack. Target and MTP reuse implementations, but they are
    # distinct semantic scopes and are joined only by generation-stage rollups.
    target_events = [
        event
        for event in events
        if event.get("layer_kind") != "mtp"
        and not str(event["node"]).startswith("mtp_")
    ]
    metrics = metrics_for_rank(target_events, n_iters)

    mtp_leaves: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        node = str(event["node"])
        if node.startswith("mtp_"):
            mtp_leaves[node].append(event)
    for target, rows in mtp_leaves.items():
        metrics[target] = _metric(
            rows,
            n_iters=n_iters,
            metric_kind="exclusive_leaf",
            aggregation="kernel interval union in the MTP semantic scope",
            all_events=events,
            elapsed_scope="invocation" if all(event.get("invocation_id") for event in rows) else None,
        )
        communication = communication_semantics(target, rows, n_iters=n_iters)
        if communication is not None:
            metrics[target]["communication"] = communication

    for target, rows in _mtp_rollup_groups(events, phase).items():
        metrics[target] = _metric(
            rows,
            n_iters=n_iters,
            metric_kind="inclusive_rollup",
            aggregation="interval union on the MTP eager timing reference rank",
            all_events=events,
            elapsed_scope="step",
        )
        communication = communication_semantics(target, rows, n_iters=n_iters)
        if communication is not None:
            metrics[target]["communication"] = communication
    _attach_mtp_hc_drill_metrics(metrics, events, phase=phase, n_iters=n_iters)
    attach_qsa_indexer_drill_metrics(
        metrics, events, n_iters=n_iters, all_events=events
    )
    for metric in metrics.values():
        metric["source_rank"] = int(events[0]["rank"])
        metric["rank_policy"] = "stack-disabled MTP timing reference rank"
    return metrics


def _attach_mtp_hc_drill_metrics(
    metrics: dict[str, Any],
    events: list[dict[str, Any]],
    *,
    phase: str,
    n_iters: int,
) -> None:
    """Attach MTP-scoped HC leaf metrics to its four layer-stage parents."""

    mtp_phase = "mtp_prefill" if phase == "prefill" else "mtp_draft_extend"

    def measured(rows: list[dict[str, Any]], node: str) -> dict[str, Any]:
        selected = [event for event in rows if str(event["node"]) == node]
        if not selected:
            raise ValueError(f"MTP HC stage has no measured {node} kernel")
        return _metric(
            selected,
            n_iters=n_iters,
            metric_kind="exclusive_leaf",
            aggregation="kernel interval union inside the MTP layer-stage scope",
            all_events=events,
            elapsed_scope="invocation",
        )

    for stage in (
        "attn_hc_mix",
        "attn_hc_combine",
        "mlp_hc_mix",
        "mlp_hc_combine",
    ):
        target = f"mtp_layer.{stage}"
        rows = [
            event
            for event in events
            if event.get("layer_kind") == "mtp"
            and str(event.get("substage") or "") == f"{mtp_phase}_{stage}"
        ]
        if not rows or target not in metrics:
            continue
        is_mix = stage.endswith("_mix")
        module_name = "attention" if stage.startswith("attn_") else "MoE"
        if is_mix:
            branch_norm = measured(rows, "hyperconnection.branch_norm")
            mix = measured(rows, "hyperconnection.mix")
            mix["display_label"] = (
                f"weighted branch mix\n4 branches → MTP {module_name} input"
            )
            scoped = {
                "branch_states": {
                    "status": "structural",
                    "label": "MTP branch-state input boundary",
                    "display_label": "MTP branch states\n[B,D,4,H]",
                },
                "branch_norm": branch_norm,
                "low_rank_gate": {
                    "status": "fused",
                    "label": "fused into the MTP hyper-connection mix kernels",
                    "included_in": "hyperconnection.mix",
                },
                "mix": mix,
                "module_input": {
                    "status": "structural",
                    "label": "MTP module-input boundary",
                    "display_label": f"MTP {module_name} input\n[B,D,H]",
                },
            }
            drill_view = "hyperconnection_mix"
        else:
            combine = measured(rows, "hyperconnection.combine")
            combine["display_label"] = (
                f"update four MTP branches\nwith {module_name} output"
            )
            scoped = {
                "branch_states": {
                    "status": "structural",
                    "label": "preserved MTP branch-state boundary",
                    "display_label": "preserved MTP branches\n[B,D,4,H]",
                },
                "module_output": {
                    "status": "structural",
                    "label": "processed MTP module-output boundary",
                    "display_label": f"MTP {module_name} output\n[B,D,H]",
                },
                "combine": combine,
                "updated_branch_states": {
                    "status": "structural",
                    "label": "updated MTP branch-state boundary",
                    "display_label": "updated MTP branches\n[B,D,4,H]",
                },
            }
            drill_view = "hyperconnection_combine"
        metrics[target]["drill_view"] = drill_view
        metrics[target]["drill_scope"] = target
        metrics[target]["drill_metrics"] = scoped


def mtp_node_states(phase: str) -> dict[str, dict[str, str]]:
    states = default_node_states(phase=phase)
    states.update(
        {
            "mtp_generation.prompt": {"status": "structural", "label": "input tensor · no GPU kernel"},
            "mtp_generation.proposal_cache": {"status": "state", "label": "persistent proposal state"},
            "mtp_generation.emitted_tokens": {"status": "structural", "label": "accepted-token output boundary"},
            "mtp_head.candidate_ids": {"status": "structural", "label": "token-id input boundary"},
            "mtp_head.target_hc_states": {"status": "structural", "label": "target HC-state input boundary"},
            "mtp_head.draft_logits": {"status": "structural", "label": "draft logits / HC-state output boundary"},
            "mtp_layer.layer_in": {"status": "structural", "label": "MTP tensor boundary"},
            "mtp_layer.layer_out": {"status": "structural", "label": "MTP tensor boundary"},
            "mtp_qsa_attention.attn_in": {"status": "structural", "label": "MTP tensor boundary"},
            "mtp_qsa_attention.kv_cache": {"status": "state", "label": "MTP KV state; update kernels shown when present"},
            "mtp_qsa_attention.attn_out": {"status": "structural", "label": "MTP tensor boundary"},
            "mtp_moe.moe_in": {"status": "structural", "label": "MTP tensor boundary"},
            "mtp_moe.moe_out": {"status": "structural", "label": "MTP tensor boundary"},
        }
    )
    inactive = (
        ("mtp_generation.draft_select", "mtp_generation.target_verify", "mtp_generation.accept_commit", "mtp_generation.mtp_draft_extend")
        if phase == "prefill"
        else ("mtp_generation.prompt", "mtp_generation.target_prefill", "mtp_generation.mtp_prefill")
    )
    for target in inactive:
        states[target] = {"status": "not_in_selected_stage", "label": f"not in selected {phase} stage"}
    if phase == "prefill":
        states["mtp_qsa_attention.metadata"] = {
            "status": "not_in_selected_stage",
            "label": "decode-only QSA layout / valid-count metadata",
        }
    return states


def _average_step_timing(
    events: list[dict[str, Any]], manifest: dict[str, Any]
) -> dict[str, Any]:
    bounds = _bounds(manifest)
    n_iters = len(bounds)
    active_us = sum(
        interval_union_us(
            event for event in events if int(event["step_index"]) == step_index
        )
        for step_index in range(1, n_iters + 1)
    ) / n_iters
    residency_us = sum(float(event["dur_us"]) for event in events) / n_iters
    elapsed_us = sum(stop - start for start, stop in bounds) / n_iters
    gaps_by_step: list[list[float]] = []
    for step_index, (start, stop) in enumerate(bounds, start=1):
        intervals = sorted(
            (
                max(start, float(event["ts_us"])),
                min(stop, float(event["ts_us"]) + float(event["dur_us"])),
            )
            for event in events
            if int(event["step_index"]) == step_index
        )
        merged: list[list[float]] = []
        for interval_start, interval_stop in intervals:
            if not merged or interval_start > merged[-1][1]:
                merged.append([interval_start, interval_stop])
            else:
                merged[-1][1] = max(merged[-1][1], interval_stop)
        gaps: list[float] = []
        cursor = start
        for interval_start, interval_stop in merged:
            if interval_start > cursor:
                gaps.append(interval_start - cursor)
            cursor = max(cursor, interval_stop)
        if stop > cursor:
            gaps.append(stop - cursor)
        gaps_by_step.append(gaps)
    all_gaps = [gap for step in gaps_by_step for gap in step]
    sorted_gaps = sorted(all_gaps)
    gap_p95_us = (
        sorted_gaps[int(0.95 * (len(sorted_gaps) - 1))] if sorted_gaps else 0.0
    )
    max_gap_us = max(sorted_gaps, default=0.0)
    average_gap_count = sum(len(step) for step in gaps_by_step) / n_iters
    average_long_gap_count = (
        sum(sum(gap >= 1000.0 for gap in step) for step in gaps_by_step) / n_iters
    )
    if average_gap_count >= 100 and gap_p95_us < 1000.0 and max_gap_us < 10000.0:
        gap_reason = "distributed_inter_kernel_bubbles"
    elif max_gap_us >= 0.5 * max(1.0, elapsed_us - active_us):
        gap_reason = "one_or_few_long_waits"
    else:
        gap_reason = "mixed_gap_distribution"
    return {
        "scope": "one scheduler iteration on the selected reference rank",
        "elapsed_ms": round(elapsed_us / 1000.0, 6),
        "active_gpu_ms": round(active_us / 1000.0, 6),
        "device_gap_ms": round(max(0.0, elapsed_us - active_us) / 1000.0, 6),
        "gpu_busy_pct": round(100.0 * active_us / elapsed_us, 2) if elapsed_us else 0.0,
        "gpu_residency_ms": round(residency_us / 1000.0, 6),
        "gpu_overlap_ms": round(max(0.0, residency_us - active_us) / 1000.0, 6),
        "elapsed_source": "GPU step annotation selected by the trace mapper",
        "device_gap_reason": gap_reason,
        "gap_count_per_step": round(average_gap_count, 2),
        "gap_median_us": round(statistics.median(all_gaps), 3) if all_gaps else 0.0,
        "gap_p95_us": round(gap_p95_us, 3),
        "max_gap_ms": round(max_gap_us / 1000.0, 6),
        "gaps_ge_1ms_per_step": round(average_long_gap_count, 2),
        "gap_reason_scope": "GPU interval morphology; CPU root cause is not asserted",
        "reference_rank": int(manifest.get("rank", 0)),
        "sample_count": n_iters,
    }


def build(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if args.phase == "prefill" and args.batch_size != 1:
        raise ValueError("MTP prefill profile requires global BS1")
    if args.batch_size not in {1, 16, 64, 256}:
        raise ValueError("MTP decode batch must be one of 1,16,64,256")
    semantic_manifest = json.loads(args.semantic_manifest.read_text())
    timing_manifest = json.loads(args.timing_manifest.read_text())
    semantic_protocol = json.loads(args.semantic_protocol.read_text())
    timing_protocol = json.loads(args.timing_protocol.read_text())
    validation = json.loads(args.semantic_validation.read_text())
    if semantic_protocol.get("generation_mode") != "eagle_mtp":
        raise ValueError("semantic trace is not EAGLE MTP")
    if timing_protocol.get("generation_mode") != "eagle_mtp":
        raise ValueError("timing trace is not EAGLE MTP")
    if semantic_protocol.get("with_stack") is not True:
        raise ValueError("semantic trace must carry Python stacks")
    if timing_protocol.get("with_stack") is not False:
        raise ValueError("timing trace must disable Python stacks")
    if semantic_protocol.get("cuda_launch_blocking") is not False:
        raise ValueError("semantic MTP trace must use normal asynchronous execution")
    if timing_protocol.get("cuda_launch_blocking") is not False:
        raise ValueError("timing MTP trace must use normal asynchronous execution")
    for protocol in (semantic_protocol, timing_protocol):
        if protocol.get("catalog_evidence") is not True:
            raise ValueError("MTP trace is diagnostic-only, not catalog evidence")
        if protocol.get("mode") != "eager":
            raise ValueError("MTP catalog trace must use eager execution")
        if protocol.get("capture_phase") != args.phase:
            raise ValueError("MTP trace phase does not match requested profile")
        if protocol.get("global_batch_sizes") != [args.batch_size]:
            raise ValueError("MTP trace global batch size does not match profile")
        if protocol.get("config_name") != "mtp_tp4_flashinfer_gdn":
            raise ValueError("MTP trace backend configuration mismatch")
        if protocol.get("topology") != "tp4_dp1_ep1_flashinfer_gdn_eagle_mtp":
            raise ValueError("MTP trace topology mismatch")
        if protocol.get("dp_size") != 1:
            raise ValueError("MTP catalog trace must be pure TP4 / DP1")
        if (protocol.get("input_len"), protocol.get("output_len")) != (8192, 1024):
            raise ValueError("MTP workload must be exact ISL/OSL 8192/1024")
        if protocol.get("server_context_length") != 9218:
            raise ValueError("MTP server context must include two internal draft slots")
        if protocol.get("max_prefill_tokens") != 8192:
            raise ValueError("MTP max prefill tokens mismatch")
        if protocol.get("chunked_prefill_size_requested") != 8192:
            raise ValueError("MTP chunked prefill size mismatch")
        if protocol.get("admission_control") != "stock":
            raise ValueError("MTP trace must use stock SGLang admission control")
        if (
            protocol.get("speculative_algorithm"),
            protocol.get("speculative_num_steps"),
            protocol.get("speculative_eagle_topk"),
            protocol.get("speculative_num_draft_tokens"),
        ) != ("EAGLE", 1, 1, 2):
            raise ValueError("MTP speculative decoding configuration mismatch")
        if protocol.get("warmup_rounds") != 3 or protocol.get("formal_rounds") != 1:
            raise ValueError("MTP workload must be warmup×3, formal×1")
        if protocol.get("round_isolation") != "same_server":
            raise ValueError("MTP warmup and formal rounds must share one server")
        if protocol.get("cache_reset_policy") != "initial-only":
            raise ValueError(
                "MTP catalog traces require one initial reset followed by natural request release"
            )
        if protocol.get("overlap_schedule") != "enabled":
            raise ValueError("MTP catalog traces require the default overlap scheduler path")
        if protocol.get("source_commit") != SOURCE_COMMIT:
            raise ValueError("MTP protocol source commit mismatch")
        if protocol.get("base_source_commit") != BASE_SOURCE_COMMIT:
            raise ValueError("MTP protocol base source commit mismatch")
        if protocol.get("source_patch_sha256") != SOURCE_PATCH_SHA256:
            raise ValueError("MTP protocol source patch mismatch")
        if protocol.get("source_patch_components") != SOURCE_PATCH_COMPONENTS:
            raise ValueError("MTP protocol source patch components mismatch")
    if semantic_protocol.get("formal_profile_steps") != 1:
        raise ValueError("MTP semantic trace must capture one scheduler iteration")
    expected_timing_steps = 1 if args.phase == "prefill" else 8
    if timing_protocol.get("formal_profile_steps") != expected_timing_steps:
        raise ValueError("MTP timing trace step count mismatch")

    source = semantic_events(args)
    timing_raw = load_jsonl(args.timing_events)
    attributed, alignment = transfer_timing(
        source, timing_raw, semantic_manifest, timing_manifest
    )
    if not attributed or any(not event.get("node") for event in attributed):
        raise ValueError("MTP timing attribution is incomplete")
    n_iters = len(_bounds(timing_manifest))
    metrics = build_metrics(attributed, phase=args.phase, n_iters=n_iters)
    states = mtp_node_states(args.phase)
    timing = _average_step_timing(attributed, timing_manifest)
    phase_label = "prefill" if args.phase == "prefill" else "decode"
    profile_id = (
        "qwen40_tp4_mtp_eager_prefill_gbs1_8k"
        if args.phase == "prefill"
        else f"qwen40_tp4_mtp_eager_decode_gbs{args.batch_size:03d}_8k1k"
    )
    variant_id = profile_id.removeprefix("qwen40_")
    profile = {
        "schema_version": "profile.v2",
        "profile_id": profile_id,
        "label": f"GB300 · pure TP4 · FlashInfer GDN · MTP on · eager {phase_label} · default overlap · global BS{args.batch_size} · 8k/1k",
        "model_id": "qwen40",
        "execution_path_id": "tp_only",
        "implementation_id": IMPLEMENTATION_ID,
        "variant_id": variant_id,
        "phase": args.phase,
        "generation_mode": "eagle_mtp",
        "entry_view": "mtp_generation",
        "execution_parameters": {"tp_size": 4, "dp_size": 1, "cp_size": 1, "ep_size": 1},
        "hardware": {"gpu": "GB300", "gpus_per_node": 4, "nodes": 1, "cluster": "CMH"},
        "workload": {
            "isl": 8192,
            "osl": 1024,
            "batch_size": args.batch_size,
            "batch_size_scope": "global_request_count",
            "warmup_rounds": 3,
            "formal_rounds": 1,
            "prompt_source": "deterministic-random-ids",
            "prompt_seed": 20260819,
            "cache_policy": "radix-disabled",
            "request_state_reset": "once before warmup-1; natural request release thereafter",
            "scheduler_path": "eager_default_overlap",
            "server_context_length": 9218,
            "max_prefill_tokens": 8192,
            "chunked_prefill_size": 8192,
            "speculative_num_steps": 1,
            "speculative_eagle_topk": 1,
            "speculative_num_draft_tokens": 2,
        },
        "profiler": {
            "type": "torch",
            "rank": int(timing_manifest.get("rank", 0)),
            "activities": ["CPU", "GPU"],
            "cuda_graph_enabled": False,
            "overlap_schedule_enabled": True,
            "captured_phase": f"eagle_mtp_{args.phase}",
            "selected_iterations": n_iters,
            "with_stack": False,
            "record_shapes": False,
            "semantic_trace_with_stack": True,
            "semantic_trace_cuda_launch_blocking": False,
            "gpu_metric_semantics": "GPU kernel-interval union across streams; overlap counted once and PDL resident-wait intervals remain active; stack semantics transferred by exact HC-delimited kernel sequence",
        },
        "evidence": {
            "job_id": int(args.timing_job_id) if args.timing_job_id.isdigit() else args.timing_job_id,
            "semantic_job_id": int(args.semantic_job_id) if args.semantic_job_id.isdigit() else args.semantic_job_id,
            "source_commit": SOURCE_COMMIT,
            "source_patch_sha256": SOURCE_PATCH_SHA256,
            "base_source_commit": BASE_SOURCE_COMMIT,
            "source_delta": "qwen4-main 32e9cb5 rollback plus QSA lifecycle/capacity hardening; model math and profile-window scheduling are unchanged",
            "source_patch_components": SOURCE_PATCH_COMPONENTS,
            "scheduler_path": "eager_default_overlap",
            "model_revision": MODEL_REVISION,
            "gdn_backend": "flashinfer_bf16",
            "trace_file": Path(timing_manifest["trace_path"]).name,
            "events_file": args.timing_events.name,
            "events_sha256": sha256_file(args.timing_events),
            "stack_trace_file": Path(semantic_manifest["trace_path"]).name,
            "stack_events_file": args.semantic_events.name,
            "stack_events_sha256": sha256_file(args.semantic_events),
            "mapping_file": args.semantic_mapping.name,
            "mapping_sha256": sha256_file(args.semantic_mapping),
            "window": timing_manifest["window"],
            "stack_mapping_window": semantic_manifest["window"],
            "timing_transfer": "same-shape per-step exact kernel-name sequence alignment partitioned at every HC combine delimiter; timing-only kernels require a direct signature, identical left/right IR semantics, or an explicitly classified generation-runtime boundary",
            "timing_alignment_steps": alignment,
            "original_stack_rule_mapped_duration_ratio": validation.get("mapped_duration_ratio"),
            "attributed_kernel_duration_ratio": 1.0,
            "accounting": {
                "kernel_count": len(attributed),
                "attributed_kernel_count": len(attributed),
                "unattributed_kernel_count": 0,
                "active_gpu_ms": timing["active_gpu_ms"],
                "gpu_residency_ms": timing["gpu_residency_ms"],
                "gpu_elapsed_ms": timing["elapsed_ms"],
                "device_gap_ms": timing["device_gap_ms"],
                "gpu_busy_pct": timing["gpu_busy_pct"],
                "gpu_overlap_ms": timing["gpu_overlap_ms"],
            },
        },
        "profile_summary": {
            "timing_phase": args.phase,
            "timing_coverage": "100% of kernels in the selected MTP scheduler iterations",
            "reference_rank": int(timing_manifest.get("rank", 0)),
            "node_time": "GPU kernel-active interval union; overlap counted once; PDL resident-wait intervals remain active",
            "kernel_detail": "GPU residency from a stack-disabled MTP eager timing trace",
            "provenance": "Python-stack IR attribution from the MTP semantic trace, transferred by exact HC-delimited per-step sequence alignment",
            "request_shape": f"global BS{args.batch_size}, ISL/OSL 8192/1024, EAGLE steps1/topk1/draft-width2",
            "scheduler_path": "eager default overlap; CUDA kernel execution remains asynchronous",
            "timing": timing,
            "gap_note": "device gap is step elapsed minus the union of all GPU kernel intervals on the reference rank; the reported gap reason classifies interval morphology only and does not assert a CPU root cause",
        },
        "node_states": states,
        "node_metrics": metrics,
    }
    timeline_path = args.output_profile.with_suffix(".timeline.json.gz")
    bounds = _bounds(timing_manifest)
    timeline = build_timeline_artifact(
        profile_id=profile_id,
        phase=args.phase,
        reference_rank=int(timing_manifest.get("rank", 0)),
        steps=[
            {
                "step_index": index,
                "label": f"formal {args.phase} step {index}",
                "trace_start_us": start,
                "duration_us": stop - start,
                "events": [event for event in attributed if int(event["step_index"]) == index],
            }
            for index, (start, stop) in enumerate(bounds, start=1)
        ],
        timing_summary=timing,
        raw_trace={
            "file": Path(timing_manifest["trace_path"]).name,
            "sha256": sha256_file(Path(timing_manifest["trace_path"])),
            "format": "pytorch_trace_json_gzip",
            "rank": int(timing_manifest.get("rank", 0)),
        },
        stack_source={
            "source": "eager_mtp_trace",
            "stack_trace_file": Path(semantic_manifest["trace_path"]).name,
            "stack_events_file": args.semantic_events.name,
            "stack_events_sha256": sha256_file(args.semantic_events),
            "policy": "full eager Python stack transferred to stack-disabled timing by exact HC-delimited kernel sequence",
        },
    )
    timeline_sha256 = write_timeline_artifact(timeline_path, timeline)
    profile["timeline"] = {
        "schema_version": timeline["schema_version"],
        "artifact": timeline_path.name,
        "reference_rank": int(timing_manifest.get("rank", 0)),
        "step_count": len(timeline["steps"]),
        "event_count": sum(len(step["events"]) for step in timeline["steps"]),
        "raw_trace_file": Path(timing_manifest["trace_path"]).name,
        "sha256": timeline_sha256,
    }
    analysis = {
        "profile_id": profile_id,
        "phase": args.phase,
        "batch_size": args.batch_size,
        "semantic_job_id": args.semantic_job_id,
        "timing_job_id": args.timing_job_id,
        "timing": timing,
        "alignment": alignment,
        "node_active_ms": {
            target: metric["active_gpu_ms"]
            for target, metric in metrics.items()
            if "active_gpu_ms" in metric
        },
    }
    return profile, analysis, timeline


def main() -> int:
    args = parse_args()
    profile, analysis, _timeline = build(args)
    args.output_profile.parent.mkdir(parents=True, exist_ok=True)
    args.output_analysis.parent.mkdir(parents=True, exist_ok=True)
    args.output_profile.write_text(yaml.safe_dump(profile, sort_keys=False))
    args.output_analysis.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output_profile}")
    print(f"wrote {args.output_analysis}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
