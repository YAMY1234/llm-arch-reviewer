#!/usr/bin/env python3
"""Build a phase-explicit Qwen 4.0 eager prefill overlay from stack evidence."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Callable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.qwen40.build.qwen40_decode_attribution import (  # noqa: E402
    _metric,
    attach_hyperconnection_drill_metrics,
    attach_qsa_indexer_drill_metrics,
    attach_qsa_indexer_drill_targets,
    communication_semantics,
    default_node_states,
    direct_kernel_mapping,
    interval_union_us,
)
from models.common.timeline_artifact import (  # noqa: E402
    build_timeline_artifact,
    write_timeline_artifact,
)


SOURCE_COMMIT = "f90a941aa6ff71ac3bd7d40b8daccdf5bd914af0"
MODEL_REVISION = "b151fd157ff99b63198ab8558432f0bf43e14d58"
CONFIGS = {
    "tp_only": {
        "execution_path_id": "tp_only",
        "implementation_id": "sglang_f90a941aa",
        "label": "pure TP4 · Triton GDN",
        "tp_size": 4,
        "dp_size": 1,
        "ep_size": 1,
        "gdn_backend": "triton",
    },
    "tp4_flashinfer_gdn": {
        "execution_path_id": "tp_only",
        "implementation_id": "sglang_f90a941aa_flashinfer_gdn",
        "label": "pure TP4 · FlashInfer GDN",
        "tp_size": 4,
        "dp_size": 1,
        "ep_size": 1,
        "gdn_backend": "flashinfer_bf16",
    },
    "dp_attention": {
        "execution_path_id": "dp_attention",
        "implementation_id": "sglang_f90a941aa_dp_attention",
        "label": "Attention DP4 · TP MoE",
        "tp_size": 4,
        "dp_size": 4,
        "ep_size": 1,
        "gdn_backend": "triton",
    },
    "dp_attention_ep4_deepep_deepgemm": {
        "execution_path_id": "dp_attention_moe_ep_deepep_deepgemm",
        "implementation_id": "sglang_f90a941aa_dp_attention_ep4_deepep_deepgemm",
        "label": "Attention DP4 · EP4 · DeepEP · DeepGEMM",
        "tp_size": 4,
        "dp_size": 4,
        "ep_size": 4,
        "gdn_backend": "flashinfer_bf16",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", choices=tuple(CONFIGS), required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--mapping-job-id")
    parser.add_argument("--timing-events", type=Path)
    parser.add_argument("--timing-manifest", type=Path)
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


def stack_text(event: dict[str, Any]) -> str:
    return "\n".join(
        " ".join(
            str(frame.get(key) or "")
            for key in ("raw", "file", "function", "module")
        )
        for frame in (event.get("python_stack") or [])
    )


def fallback_prefill_node(
    event: dict[str, Any], mapping: dict[str, Any]
) -> tuple[str, str, str]:
    """Use the recorded Python stack for kernels not covered by the old rules."""

    kernel = str(event.get("kernel_name", ""))
    lowered_kernel = kernel.lower()
    names = stack_text(event)
    lowered = names.lower()
    context = str((mapping.get("model_context_frame") or {}).get("raw") or "")

    direct, label = direct_kernel_mapping(kernel)
    if direct:
        return direct, str(label), "high"
    if "allgather" in lowered_kernel or "all_gather" in lowered_kernel:
        return "top.tp_logits_collective", "TP logits all-gather", "high"
    if "bmm_" in lowered_kernel or "deep_gemm::" in lowered_kernel:
        return "moe.routed_experts", "routed-expert GEMM", "high"
    if "vocabparallelembedding" in lowered:
        if "qwen4expngramembedding" in lowered:
            return "ple.ngram_embedding", "N-gram embedding support", "high"
        return "top.embedding", "token embedding support", "high"
    if "qwen3_5gateddeltanet" in lowered or "qwen3_5gateddeltanet" in context.lower():
        if "rowparallellinear" in lowered or "out_proj" in lowered:
            return "linear_attention.output_projection", "GDN output projection", "high"
        if "mergedcolumnparallellinear" in lowered or "input_proj" in lowered:
            return "linear_attention.qkvz_projection", "GDN input projection", "high"
    if "gdn_flashinfer.py" in lowered or "kernels/ops/attention/fla" in lowered:
        return "linear_attention.delta_rule", "prefill GDN recurrence support", "high"
    if "qwen_sparse_attn_backend.py" in lowered or "qsa_kv_pool.py" in lowered:
        return "qsa_attention.metadata", "QSA layout/KV metadata", "high"
    if "hybrid_linear_attn_backend.py" in lowered or "mamba" in lowered:
        return "linear_attention.recurrent_state", "linear-attention state preparation", "high"
    if "gdn_backend.py" in lowered:
        return "linear_attention.recurrent_state", "GDN prefill metadata/state", "medium"
    if "qwen4expattentiondecoderlayer" in lowered or "self_attention" in lowered:
        if "rowparallellinear" in lowered or "o_proj" in lowered:
            return "qsa_attention.output_projection", "QSA output projection", "high"
        if "rmsnorm" in lowered_kernel or "rope" in lowered_kernel:
            return "qsa_attention.qk_norm_rope", "Q/K norm + RoPE", "medium"
        return "qsa_attention.output_gate", "QSA output/support elementwise", "medium"
    if "_prepare_ple_batch" in lowered or "ple_state_pool.py" in lowered:
        return "ple.token_history", "PLE context/state preparation", "high"
    if "_commit_ple_batch" in lowered:
        return "ple.context_commit", "PLE context/state commit", "high"
    if "qwen2moe" in lowered or "fusedmoe" in lowered:
        if "routing" in lowered_kernel or "topk" in lowered:
            return "moe.topk", "MoE route selection", "high"
        return "moe.routed_experts", "MoE expert support", "medium"
    return "top.runtime_support", "model-runner metadata/cache support", "medium"


def layer_kind(event: dict[str, Any], node: str) -> str | None:
    lowered = stack_text(event).lower()
    if "qwen4explineardecoderlayer" in lowered or "qwen3_5gateddeltanet" in lowered:
        return "linear"
    if "qwen4expattentiondecoderlayer" in lowered:
        return "full"
    if node.startswith("linear_attention.") or node.startswith("linear_layer."):
        return "linear"
    if node.startswith("qsa_attention.") or node.startswith("full_layer."):
        return "full"
    return None


def substage(event: dict[str, Any], node: str) -> str | None:
    lowered = stack_text(event).lower()
    if node.startswith("ple."):
        return "ple"
    if node.startswith("linear_attention.") or node.startswith("qsa_attention."):
        return "attention"
    if (
        node.startswith("moe.")
        or ".dp_moe_" in node
        or node.endswith("_moe_output_collective")
    ):
        return "moe"
    if node.startswith("hyperconnection."):
        if "_prepare_qwen4_exp_attn" in lowered:
            return "attn_hc_mix"
        if "_prepare_qwen4_exp_mlp" in lowered:
            return (
                "attn_hc_combine"
                if node == "hyperconnection.combine"
                else "mlp_hc_mix"
            )
        if "_postprocess_qwen4_exp_layer" in lowered:
            return "mlp_hc_combine"
    return None


def metrics_for_prefill(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    attach_qsa_indexer_drill_targets(events)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        groups[event["node"]].append(event)
        kind = event.get("layer_kind")
        stage = event.get("substage")
        if kind in {"linear", "full"}:
            layer_view = "linear_layer" if kind == "linear" else "full_layer"
            groups[f"stack.{layer_view}"].append(event)
            if stage in {
                "attn_hc_mix",
                "attn_hc_combine",
                "mlp_hc_mix",
                "mlp_hc_combine",
            }:
                groups[f"{layer_view}.{stage}"].append(event)
            elif stage == "attention":
                parent = "linear_attention" if kind == "linear" else "qsa_attention"
                groups[f"{layer_view}.{parent}"].append(event)
            elif stage == "moe" and str(event.get("node", "")).startswith("moe."):
                groups[f"{layer_view}.moe"].append(event)
        if stage == "ple":
            groups["stack.ple_injection"].append(event)
    groups["top.decoder_stack"].extend(
        event
        for event in events
        if event.get("layer_kind") is not None or event.get("substage") == "ple"
    )
    leaf_nodes = {event["node"] for event in events}
    metrics = {}
    for target, target_events in sorted(groups.items()):
        elapsed_scope = None
        if target == "top.decoder_stack":
            elapsed_scope = "step"
        elif target_events and all(
            event.get("invocation_id") is not None for event in target_events
        ):
            elapsed_scope = "invocation"
        metrics[target] = _metric(
            target_events,
            n_iters=1,
            metric_kind="exclusive_leaf" if target in leaf_nodes else "inclusive_rollup",
            aggregation=(
                "interval union on the stack-profiled active rank"
                if target not in leaf_nodes
                else "kernel interval union on the stack-profiled active rank"
            ),
            all_events=events,
            elapsed_scope=elapsed_scope,
        )
    for target, target_events in groups.items():
        communication = communication_semantics(target, target_events, n_iters=1)
        if communication is not None:
            metrics[target]["communication"] = communication
    attach_hyperconnection_drill_metrics(
        metrics, groups, n_iters=1, all_events=events
    )
    attach_qsa_indexer_drill_metrics(
        metrics, events, n_iters=1, all_events=events
    )
    return metrics


def _iteration_bounds(manifest: dict[str, Any]) -> list[tuple[float, float]]:
    bounds = manifest.get("window", {}).get("iter_bounds_us") or []
    return [(float(start), float(stop)) for start, stop in bounds]


def _split_by_iteration(
    events: list[dict[str, Any]], manifest: dict[str, Any]
) -> list[list[dict[str, Any]]]:
    """Split a selected trace window without dropping inter-chunk support work."""

    bounds = _iteration_bounds(manifest)
    if not bounds:
        raise ValueError("prefill manifest has no iteration bounds")
    chunks: list[list[dict[str, Any]]] = [[] for _ in bounds]
    for event in events:
        timestamp = float(event["ts_us"])
        matches = [index for index, (start, _) in enumerate(bounds) if timestamp >= start]
        if not matches or timestamp > bounds[-1][1]:
            raise ValueError(
                f"kernel timestamp {timestamp} is outside the selected prefill request"
            )
        # Runtime scheduling/cache kernels between GPU step annotations belong
        # to the chunk that just ran.  Splitting at the next chunk's start
        # preserves them and gives stack/timing traces the same boundaries.
        index = max(matches)
        chunks[index].append(event)
    return chunks


def _semantic_key(event: dict[str, Any]) -> tuple[str, str | None, str | None]:
    return (
        str(event["node"]),
        event.get("layer_kind"),
        event.get("substage"),
    )


def _transfer_chunk_timing(
    source_events: list[dict[str, Any]],
    timing_events: list[dict[str, Any]],
    *,
    timing_rank: int,
    chunk_index: int,
    boundary_insert_resolver: Callable[
        [dict[str, Any], dict[str, Any], dict[str, Any]],
        dict[str, Any] | None,
    ]
    | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Transfer stack semantics using exact sequence alignment.

    Equal kernel-name blocks are copied one-to-one.  Stack-only kernels may be
    deleted because stack/shape collection launches profiler support work.  A
    timing-only kernel is accepted only when it has a unique direct signature,
    or when the exact matched source events immediately surrounding the gap
    agree on node, layer kind, and substage.  This keeps the transfer complete
    without fuzzy-name or duration-based guesses.
    """

    source_for_timing: dict[int, int] = {}

    def delimiter_indices(rows: list[dict[str, Any]]) -> list[int]:
        return [
            index
            for index, event in enumerate(rows)
            if "hc_combine_kernel" in str(event.get("kernel_name", "")).lower()
        ]

    source_delimiters = delimiter_indices(source_events)
    timing_delimiters = delimiter_indices(timing_events)
    if len(source_delimiters) != len(timing_delimiters):
        raise ValueError(
            f"chunk {chunk_index}: stack/timing HC delimiter counts differ: "
            f"{len(source_delimiters)} != {len(timing_delimiters)}"
        )
    # Repeated generic GEMM/BMM names are not globally unique. First partition
    # both traces at the 96 stable per-layer HC combine delimiters, then align
    # only within the corresponding Attention/MoE segment. This prevents a
    # timing BMM in one module from borrowing the Python stack of another.
    source_stops = [index + 1 for index in source_delimiters] + [len(source_events)]
    timing_stops = [index + 1 for index in timing_delimiters] + [len(timing_events)]
    source_start = timing_start = 0
    for source_stop, timing_stop in zip(source_stops, timing_stops):
        source_names = [
            str(event["kernel_name"])
            for event in source_events[source_start:source_stop]
        ]
        timing_names = [
            str(event["kernel_name"])
            for event in timing_events[timing_start:timing_stop]
        ]
        matcher = SequenceMatcher(None, source_names, timing_names, autojunk=False)
        for local_source, local_timing, size in matcher.get_matching_blocks():
            for offset in range(size):
                source_for_timing[timing_start + local_timing + offset] = (
                    source_start + local_source + offset
                )
        source_start = source_stop
        timing_start = timing_stop

    transferred: list[dict[str, Any]] = []
    exact_count = 0
    direct_insert_count = 0
    context_insert_count = 0
    for timing_index, timing_event in enumerate(timing_events):
        source_index = source_for_timing.get(timing_index)
        if source_index is not None:
            source_event = dict(source_events[source_index])
            exact_count += 1
            method = str(source_event["attribution_method"])
            source_event["attribution_method"] = (
                "direct_signature"
                if method == "direct_signature_with_python_stack"
                else method + "+exact_sequence_timing_transfer"
            )
        else:
            direct_node, direct_label = direct_kernel_mapping(
                str(timing_event["kernel_name"])
            )
            if direct_node is not None:
                source_event = {
                    "kernel_label": direct_label,
                    "node": direct_node,
                    "layer_kind": layer_kind(timing_event, direct_node),
                    "substage": substage(timing_event, direct_node),
                    "attribution_method": "direct_signature_timing_insert",
                    "confidence": "high",
                }
                direct_insert_count += 1
            else:
                left_timing = max(
                    (index for index in source_for_timing if index < timing_index),
                    default=None,
                )
                right_timing = min(
                    (index for index in source_for_timing if index > timing_index),
                    default=None,
                )
                if left_timing is None or right_timing is None:
                    raise ValueError(
                        f"chunk {chunk_index}: timing-only boundary kernel has no "
                        f"direct signature: {timing_event['kernel_name']!r}"
                    )
                left = source_events[source_for_timing[left_timing]]
                right = source_events[source_for_timing[right_timing]]
                resolved = (
                    boundary_insert_resolver(timing_event, left, right)
                    if boundary_insert_resolver is not None
                    else None
                )
                if resolved is not None:
                    source_event = resolved
                    context_insert_count += 1
                elif _semantic_key(left) != _semantic_key(right):
                    raise ValueError(
                        f"chunk {chunk_index}: timing-only kernel crosses IR semantics "
                        f"{_semantic_key(left)!r} -> {_semantic_key(right)!r}: "
                        f"{timing_event['kernel_name']!r}"
                    )
                else:
                    source_event = {
                        "kernel_label": (
                            f"{left['kernel_label']} support (exact sequence context)"
                        ),
                        "node": left["node"],
                        "layer_kind": left.get("layer_kind"),
                        "substage": left.get("substage"),
                        "attribution_method": "exact_sequence_context_insert",
                        "confidence": "high",
                    }
                    context_insert_count += 1
        source_event.update(
            {
                "rank": timing_rank,
                # All chunks are pieces of one global 8k request.  Keeping one
                # step index makes active-time union aggregate, rather than
                # average, the four sequential DP chunks.
                "step_index": 1,
                "prefill_chunk_index": chunk_index,
                "kernel_name": timing_event["kernel_name"],
                "ts_us": float(timing_event["ts_us"]),
                "dur_us": float(timing_event["dur_us"]),
                "stream": timing_event.get("stream"),
                "device": timing_event.get("device"),
                "pid": timing_event.get("pid"),
                "tid": timing_event.get("tid"),
            }
        )
        transferred.append(source_event)

    return transferred, {
        "exact_kernel_count": exact_count,
        "direct_timing_insert_count": direct_insert_count,
        "context_timing_insert_count": context_insert_count,
        "stack_only_support_kernel_count": len(source_events) - exact_count,
    }


def build_profile(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = CONFIGS[args.config_name]
    raw_events = load_jsonl(args.events)
    mappings = load_jsonl(args.mapping)
    if len(raw_events) != len(mappings):
        raise ValueError("event and mapping lengths differ")
    manifest = json.loads(args.manifest.read_text())
    validation = json.loads(args.validation.read_text())
    protocol = json.loads(args.protocol.read_text())
    if manifest.get("phase") != "forward_extend":
        raise ValueError(f"expected forward_extend, got {manifest.get('phase')}")
    if protocol.get("mode") != "eager" or protocol.get("input_len") != 8192:
        raise ValueError("eager prefill protocol mismatch")
    stack_bounds = _iteration_bounds(manifest)
    prefill_chunk_count = len(stack_bounds)
    expected_anchors = 36 * prefill_chunk_count
    if int(manifest.get("window", {}).get("anchor_kernel_count", 0)) != expected_anchors:
        raise ValueError(
            "prefill window does not contain the expected "
            f"{expected_anchors} GDN anchors across {prefill_chunk_count} chunk(s)"
        )
    requested_chunk_size = int(
        protocol.get("chunked_prefill_size_requested") or protocol["input_len"]
    )
    per_rank_chunk_size = int(
        protocol.get("chunked_prefill_size_per_dp_rank") or requested_chunk_size
    )
    expected_chunk_count = (
        int(protocol["input_len"]) + per_rank_chunk_size - 1
    ) // per_rank_chunk_size
    if prefill_chunk_count != expected_chunk_count:
        raise ValueError(
            f"selected {prefill_chunk_count} prefill chunks, expected "
            f"{expected_chunk_count} for the full {protocol['input_len']}-token request"
        )

    attributed = []
    fallback_count = 0
    for event, mapping in zip(raw_events, mappings):
        if event.get("event_id") != mapping.get("event_id"):
            raise ValueError("event/mapping IDs are not aligned")
        node = mapping.get("selected_node")
        confidence = str(mapping.get("confidence") or "unmapped")
        method = "python_stack_ir_rule"
        label = None
        kernel_name = str(event["kernel_name"])
        lowered_kernel = kernel_name.lower()
        direct_node, direct_label = direct_kernel_mapping(kernel_name)
        names = stack_text(event).lower()
        if (
            "grouped_gemma_rmsnorm_kernel" in lowered_kernel
            and "layers/hyperconnection.py" in names
        ):
            node = "hyperconnection.branch_norm"
            label = "per-branch hyper-connection RMSNorm"
            confidence = "high"
            method = "python_stack_semantic_fallback"
            fallback_count += 1
        elif "replicatedlinear" in names and "qwen4expplelayer" in names:
            node = "ple.key_value_projection"
            label = "PLE key/value projection"
            confidence = "high"
            method = "python_stack_semantic_fallback"
            fallback_count += 1
        elif (
            "catarray" in lowered_kernel
            and "_prepare_qwen4_exp_attn" in names
            and "_prepare_ple_batch" not in names
        ):
            node = "stack.hc_expand"
            label = "initialize four hyper-connection branches"
            confidence = "high"
            method = "python_stack_semantic_fallback"
            fallback_count += 1
        elif direct_node is not None:
            node = direct_node
            label = direct_label
            confidence = "high"
            method = "direct_signature_with_python_stack"
        elif "replicatedlinear" in names and "_forward_router_experts" in names:
            node = "moe.router"
            label = "replicated MoE router projection"
            confidence = "high"
            method = "python_stack_semantic_fallback"
            fallback_count += 1
        elif (
            "compute_dp_attention_metadata" in names
            and "_gather_dp_attn_hidden_states" in names
        ):
            node = "top.dp_logits_input_gather"
            label = "DP logits-input gather metadata"
            confidence = "high"
            method = "python_stack_semantic_fallback"
            fallback_count += 1
        elif node is None:
            node, label, confidence = fallback_prefill_node(event, mapping)
            method = "python_stack_semantic_fallback"
            fallback_count += 1
        kind = layer_kind(event, str(node))
        stage = substage(event, str(node))
        if node.startswith("hyperconnection.") and kind is None:
            node = "top.final_hc_mix"
        semantic = mapping.get("semantic_frame") or mapping.get("operator_frame") or {}
        label = label or str(semantic.get("function") or event.get("cpu_op_name") or node)
        attributed.append(
            {
                "rank": manifest.get("rank"),
                "step_index": 1,
                "prefill_chunk_index": max(
                    index
                    for index, (start, _) in enumerate(stack_bounds, start=1)
                    if float(event["ts_us"]) >= start
                ),
                "kernel_name": event["kernel_name"],
                "kernel_label": label,
                "node": str(node),
                "ts_us": float(event["ts_us"]),
                "dur_us": float(event["dur_us"]),
                "stream": event.get("stream"),
                "device": event.get("device"),
                "pid": event.get("pid"),
                "tid": event.get("tid"),
                "layer_kind": kind,
                "substage": stage,
                "attribution_method": method,
                "confidence": confidence,
                "cpu_op_name": event.get("cpu_op_name"),
                "python_stack": event.get("python_stack") or [],
                "stack_evidence": {
                    "source": "eager_trace",
                    "match": "direct_event",
                    "kind": "full_eager_python_stack",
                    "event_id": event.get("event_id"),
                    "confidence": confidence,
                },
            }
        )
    if any(not event["node"] for event in attributed):
        raise ValueError("prefill attribution contains an empty node")

    timing_manifest = None
    timing_source_extra_kernel_count = 0
    timing_alignment_chunks: list[dict[str, int]] = []
    timing_transfer = args.timing_events is not None or args.timing_manifest is not None
    if timing_transfer:
        if args.timing_events is None or args.timing_manifest is None:
            raise ValueError("--timing-events and --timing-manifest must be provided together")
        timing_events = load_jsonl(args.timing_events)
        timing_manifest = json.loads(args.timing_manifest.read_text())
        if timing_manifest.get("phase") != "forward_extend":
            raise ValueError("timing trace is not a forward_extend window")
        if (
            timing_manifest.get("profiler", {}).get("with_stack") is True
            or protocol.get("with_stack") is not False
        ):
            raise ValueError("timing trace must disable Python stacks")
        timing_bounds = _iteration_bounds(timing_manifest)
        if len(timing_bounds) != prefill_chunk_count:
            raise ValueError(
                "stack-attribution and stack-disabled timing traces select "
                "different prefill chunk counts"
            )
        if int(timing_manifest.get("window", {}).get("anchor_kernel_count", 0)) != (
            36 * len(timing_bounds)
        ):
            raise ValueError("timing window has an incomplete GDN anchor count")
        source_chunks = _split_by_iteration(attributed, manifest)
        timing_chunks = _split_by_iteration(timing_events, timing_manifest)
        transferred = []
        for chunk_index, (source_chunk, timing_chunk) in enumerate(
            zip(source_chunks, timing_chunks), start=1
        ):
            chunk_events, chunk_accounting = _transfer_chunk_timing(
                source_chunk,
                timing_chunk,
                timing_rank=int(timing_manifest.get("rank", 0)),
                chunk_index=chunk_index,
            )
            transferred.extend(chunk_events)
            timing_alignment_chunks.append(chunk_accounting)
        timing_source_extra_kernel_count = sum(
            chunk["stack_only_support_kernel_count"]
            for chunk in timing_alignment_chunks
        )
        attributed = transferred

    metrics = metrics_for_prefill(attributed)
    reference_rank = int(
        (timing_manifest or manifest).get("rank", 0)
    )
    for metric in metrics.values():
        metric["source_rank"] = reference_rank
        metric["rank_policy"] = (
            "critical stack-disabled timing rank"
            if timing_transfer
            else "stack-profiled active rank"
        )
        for drill_metric in metric.get("drill_metrics", {}).values():
            drill_metric["source_rank"] = reference_rank
            drill_metric["rank_policy"] = metric["rank_policy"]
    states = default_node_states(phase="prefill")
    if args.config_name == "dp_attention_ep4_deepep_deepgemm":
        states["moe.combine"] = {
            "status": "fused",
            "label": "routed-output return is fused into DeepEP combine",
            "included_in": "moe.deepep_combine",
        }
    active_ms = interval_union_us(attributed) / 1000.0
    residency_ms = sum(event["dur_us"] for event in attributed) / 1000.0
    elapsed_ms = float((timing_manifest or manifest)["window"]["duration_ms"])
    if active_ms > elapsed_ms + 1e-6:
        raise ValueError(
            f"prefill active GPU time exceeds selected request window: "
            f"{active_ms} > {elapsed_ms}"
        )
    device_gap_ms = max(0.0, elapsed_ms - active_ms)
    overlap_ms = max(0.0, residency_ms - active_ms)
    timing_summary = {
        "scope": "full prefill request on the selected reference rank",
        "elapsed_ms": round(elapsed_ms, 6),
        "active_gpu_ms": round(active_ms, 6),
        "device_gap_ms": round(device_gap_ms, 6),
        "gpu_busy_pct": round(100.0 * active_ms / elapsed_ms, 2)
        if elapsed_ms
        else 0.0,
        "gpu_residency_ms": round(residency_ms, 6),
        "gpu_overlap_ms": round(overlap_ms, 6),
        "elapsed_source": "profiler-selected full-request iteration bounds",
        "device_gap_reason": "unclassified",
        "reference_rank": reference_rank,
        "sample_count": 1,
    }
    prefix = "tp4" if args.config_name == "tp_only" else args.config_name
    profile_id = f"qwen40_{prefix}_eager_prefill_gbs1_8k"
    variant_id = f"{prefix}_eager_prefill_gbs1_8k"
    profile = {
        "schema_version": "profile.v2",
        "profile_id": profile_id,
        "label": f"GB300 · {config['label']} · eager prefill · global BS1 · 8k",
        "model_id": "qwen40",
        "execution_path_id": config["execution_path_id"],
        "implementation_id": config["implementation_id"],
        "variant_id": variant_id,
        "phase": "prefill",
        "execution_parameters": {
            "tp_size": config["tp_size"],
            "dp_size": config["dp_size"],
            "cp_size": 1,
            "ep_size": config["ep_size"],
        },
        "hardware": {"gpu": "GB300", "gpus_per_node": 4, "nodes": 1, "cluster": "CMH"},
        "workload": {
            "isl": 8192,
            "osl": 1024,
            "batch_size": 1,
            "batch_size_scope": "global_request_count",
            "warmup_rounds": 3,
            "formal_rounds": 1,
            "prompt_source": "deterministic-random-ids",
            "prompt_seed": 20260819,
            "cache_policy": "radix-disabled",
        },
        "profiler": {
            "type": "torch",
            "rank": reference_rank,
            "activities": ["CPU", "GPU"],
            "cuda_graph_enabled": False,
            "captured_phase": "forward_extend",
            "selected_iterations": 1,
            "prefill_chunk_count": prefill_chunk_count,
            "with_stack": not timing_transfer,
            "record_shapes": not timing_transfer,
            "gpu_metric_semantics": (
                "GPU active interval union across every sequential chunk of one "
                "full eager 8k prefill request; Python-stack attribution is "
                "transferred by exact per-chunk kernel-name sequence alignment "
                "when a stack-disabled timing trace is supplied"
            ),
        },
        "evidence": {
            "job_id": int(args.job_id) if args.job_id.isdigit() else args.job_id,
            "source_commit": SOURCE_COMMIT,
            "model_revision": MODEL_REVISION,
            "gdn_backend": config["gdn_backend"],
            "trace_file": Path((timing_manifest or manifest)["trace_path"]).name,
            "events_file": (args.timing_events or args.events).name,
            "events_sha256": sha256_file(args.timing_events or args.events),
            "stack_trace_file": Path(manifest["trace_path"]).name,
            "stack_events_file": args.events.name,
            "stack_events_sha256": sha256_file(args.events),
            "mapping_file": args.mapping.name,
            "mapping_sha256": sha256_file(args.mapping),
            "mapping_job_id": (
                int(args.mapping_job_id)
                if args.mapping_job_id and args.mapping_job_id.isdigit()
                else args.mapping_job_id
            ),
            "protocol_file": args.protocol.name,
            "protocol_sha256": sha256_file(args.protocol),
            "window": (timing_manifest or manifest)["window"],
            "stack_mapping_window": manifest["window"],
            "timing_transfer": (
                "exact per-chunk kernel-name sequence alignment partitioned at stable "
                "HC combine delimiters; stack-only support kernels deleted; timing-only "
                "kernels require a direct signature or identical left/right IR semantics; "
                "no fuzzy-name or duration matching"
                if timing_transfer
                else "same stack-enabled trace supplies attribution and timing"
            ),
            "timing_alignment_chunks": timing_alignment_chunks,
            "stack_only_support_kernel_count": timing_source_extra_kernel_count,
            "original_stack_rule_mapped_duration_ratio": validation.get("mapped_duration_ratio"),
            "attributed_kernel_duration_ratio": 1.0,
            "accounting": {
                "kernel_count": len(attributed),
                "attributed_kernel_count": len(attributed),
                "unattributed_kernel_count": 0,
                "stack_rule_kernel_count": sum(
                    "semantic_fallback" not in event["attribution_method"]
                    for event in attributed
                ),
                "stack_semantic_fallback_kernel_count": sum(
                    "semantic_fallback" in event["attribution_method"]
                    for event in attributed
                ),
                "active_gpu_ms": round(active_ms, 6),
                "gpu_residency_ms": round(residency_ms, 6),
                "gpu_elapsed_ms": round(elapsed_ms, 6),
                "device_gap_ms": round(device_gap_ms, 6),
                "gpu_busy_pct": timing_summary["gpu_busy_pct"],
                "gpu_overlap_ms": round(overlap_ms, 6),
            },
        },
        "profile_summary": {
            "timing_phase": "prefill",
            "timing_coverage": (
                "100% of kernels across the full 8k eager prefill request "
                f"({prefill_chunk_count}×{per_rank_chunk_size}-token runtime chunks)"
            ),
            "reference_rank": reference_rank,
            "node_time": "active GPU time; overlap counted once",
            "kernel_detail": (
                "GPU residency from the stack-disabled timing trace"
                if timing_transfer
                else "GPU residency from the same stack-enabled trace"
            ),
            "provenance": (
                "Python-stack IR attribution transferred by exact per-chunk, "
                "HC-delimiter-partitioned kernel-name sequence alignment onto the "
                "stack-disabled timing trace"
                if timing_transfer
                else "Python stack IR rule or explicit semantic-stack fallback per node"
            ),
            "scope_note": "BS1 prefill only; decode has separate CUDA Graph BS sweeps",
            "request_shape": (
                f"one global BS1 / 8k-token request, executed as "
                f"{prefill_chunk_count}×{per_rank_chunk_size}-token chunks"
            ),
            "timing": timing_summary,
            "gap_note": (
                "device gap is request elapsed minus the union of all GPU kernel "
                "intervals on the reference rank; its CPU/synchronization cause is "
                "not inferred without direct evidence"
            ),
            "node_elapsed_coverage": (
                "decoder-stack envelope only; per-node eager-prefill elapsed/gap "
                "requires explicit invocation markers because concurrent streams "
                "interleave logical layers"
            ),
        },
        "node_states": states,
        "node_metrics": metrics,
    }
    timeline_path = args.output_profile.with_suffix(".timeline.json.gz")
    source_manifest = timing_manifest or manifest
    raw_trace_path = Path(source_manifest["trace_path"])
    window = source_manifest["window"]
    timeline = build_timeline_artifact(
        profile_id=profile_id,
        phase="prefill",
        reference_rank=reference_rank,
        steps=[
            {
                "step_index": 1,
                "label": "full 8k prefill request",
                "trace_start_us": float(window["start_us"]),
                "duration_us": float(window["duration_ms"]) * 1000.0,
                "events": attributed,
            }
        ],
        timing_summary=timing_summary,
        raw_trace={
            "file": raw_trace_path.name,
            "sha256": sha256_file(raw_trace_path),
            "format": "pytorch_trace_json_gzip",
            "rank": reference_rank,
        },
        stack_source={
            "source": "eager_trace",
            "stack_trace_file": Path(manifest["trace_path"]).name,
            "stack_events_file": args.events.name,
            "stack_events_sha256": sha256_file(args.events),
            "policy": (
                "direct same-event eager Python stack"
                if not timing_transfer
                else "eager Python stack transferred to stack-disabled timing by "
                "exact per-chunk kernel sequence; unmatched timing-only support "
                "kernels are explicitly stackless"
            ),
        },
    )
    profile["timeline"] = {
        "schema_version": timeline["schema_version"],
        "artifact": timeline_path.name,
        "reference_rank": reference_rank,
        "step_count": 1,
        "event_count": len(timeline["steps"][0]["events"]),
        "raw_trace_file": raw_trace_path.name,
    }
    analysis = {
        "profile_id": profile_id,
        "config_name": args.config_name,
        "accounting": profile["evidence"]["accounting"],
        "node_metrics": metrics,
        "fallback_nodes": dict(
            Counter(
                event["node"]
                for event in attributed
                if event["attribution_method"] == "python_stack_semantic_fallback"
            )
        ),
        "timing_transfer": profile["evidence"]["timing_transfer"],
    }
    return profile, analysis, timeline


def main() -> int:
    args = parse_args()
    profile, analysis, timeline = build_profile(args)
    timeline_path = args.output_profile.with_suffix(".timeline.json.gz")
    profile["timeline"]["sha256"] = write_timeline_artifact(
        timeline_path, timeline
    )
    args.output_profile.parent.mkdir(parents=True, exist_ok=True)
    args.output_profile.write_text(
        yaml.safe_dump(profile, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    args.output_analysis.parent.mkdir(parents=True, exist_ok=True)
    args.output_analysis.write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    accounting = profile["evidence"]["accounting"]
    print(f"wrote {args.output_profile.resolve()}")
    print(f"wrote {timeline_path.resolve()}")
    print(
        f"prefill active={accounting['active_gpu_ms']:.3f} ms, "
        f"kernels={accounting['kernel_count']}, attributed=100%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
