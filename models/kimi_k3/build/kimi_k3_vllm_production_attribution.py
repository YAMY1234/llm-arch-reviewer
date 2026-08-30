"""Fail-closed Kimi K3 vLLM eager-to-production attribution.

The exact vLLM Kimi K3 path emits two AttnRes owners per decoder layer and
one final output owner.  Those 187 ordered calls provide an implementation-
unique occurrence skeleton.  Eager semantic evidence is transferred to
production only inside the same occurrence and only by spelling-normalized
kernel identity plus same-identity ordinal.  This includes explicit eager-break
kernels before and after the captured decoder graphs.  Production-only work is
retained as explicit runtime support; a semantic eager interval may never
disappear.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import Counter, defaultdict, deque
import json
from pathlib import Path
import re
from typing import Any


ATTN_RES_ANCHOR_COUNT = 187
SEGMENT_COUNT = ATTN_RES_ANCHOR_COUNT + 1


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def kernel_exact_identity(name: str) -> str:
    normalized = re.sub(r"^void\s+", "", name.strip().lower())
    normalized = normalized.replace("<unnamed>", "(anonymous namespace)")
    normalized = re.sub(r"\(int\)(?=\d)", "", normalized)
    normalized = re.sub(r"\bconst\s+([\w:]+)", r"\1 const", normalized)
    normalized = re.sub(r"\s*\*\s*", "*", normalized)
    normalized = re.sub(r"\s*,\s*", ",", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def kernel_base(name: str) -> str:
    normalized = kernel_exact_identity(name)
    if "<" in normalized:
        normalized = normalized.split("<", 1)[0]
    else:
        normalized = re.sub(r"\([^()]*\)\s*$", "", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _is_anchor(row: dict[str, Any]) -> bool:
    base = kernel_base(str(row.get("kernel_name") or ""))
    return "attn_res_fwd_online_v2_kernel" in base or "_attn_res_kernel" in base


def anchor_segments(rows: list[dict[str, Any]]) -> list[tuple[int, int]]:
    anchors = [index for index, row in enumerate(rows) if _is_anchor(row)]
    if len(anchors) != ATTN_RES_ANCHOR_COUNT:
        raise ValueError(
            f"expected {ATTN_RES_ANCHOR_COUNT} ordered vLLM Kimi K3 AttnRes "
            f"anchors, got {len(anchors)}"
        )
    return [
        (0, anchors[0]),
        *zip(anchors[:-1], anchors[1:]),
        (anchors[-1], len(rows)),
    ]


def occurrence_for_segment(segment_id: int) -> dict[str, Any]:
    if segment_id == 0:
        return {
            "layer_id": None,
            "layer_kind": "runtime",
            "substage": "step_setup",
            "occurrence_id": "runtime.step_setup",
        }
    if segment_id == SEGMENT_COUNT - 1:
        return {
            "layer_id": None,
            "layer_kind": "output",
            "substage": "final_output",
            "occurrence_id": "final_output",
        }
    layer_id = (segment_id - 1) // 2
    substage = "attention" if segment_id % 2 else "feed_forward"
    return {
        "layer_id": layer_id,
        "layer_kind": (
            "gated_mla"
            if layer_id
            in {
                3,
                7,
                11,
                15,
                19,
                23,
                27,
                31,
                35,
                39,
                43,
                47,
                51,
                55,
                59,
                63,
                67,
                71,
                75,
                79,
                83,
                87,
                91,
                92,
            }
            else "kda"
        ),
        "substage": substage,
        "occurrence_id": f"layer_{layer_id:02d}.{substage}",
    }


def _support_contract(segment_id: int, kernel_name: str) -> tuple[str, str]:
    base = kernel_base(kernel_name)
    if segment_id == 0:
        return (
            "request_batch_metadata",
            "request-position, batch-shape, and replay-index preparation outside the stable Model IR",
        )
    if segment_id == SEGMENT_COUNT - 1:
        return (
            "sampling_and_output",
            "post-logits selection and scheduler result bookkeeping outside the stable Model IR",
        )
    if "alloc" in base or "index" in base or "cache" in base:
        return (
            "allocator_or_cache_management",
            "scheduler/cache allocation bookkeeping overlapped this exact invocation",
        )
    return (
        "state_bookkeeping",
        "scheduler request-state bookkeeping overlapped this exact invocation",
    )


def _prefill_source_pairs(
    eager_rows: list[dict[str, Any]],
    production_rows: list[dict[str, Any]],
    eager_events_path: Path,
) -> dict[int, dict[str, Any]]:
    eager_events = load_jsonl(eager_events_path)
    if len(eager_events) != len(eager_rows):
        raise ValueError("eager mapping/event length mismatch")
    if any(
        mapping.get("event_id") != event.get("event_id")
        for mapping, event in zip(eager_rows, eager_events)
    ):
        raise ValueError("eager mapping/event identity mismatch")

    source_by_stream: dict[int, list[dict[str, Any]]] = defaultdict(list)
    production_by_stream: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for mapping, event in zip(eager_rows, eager_events):
        source_by_stream[int(event["stream"])].append(mapping)
    for index, row in enumerate(production_rows):
        production_by_stream[int(row["stream"])].append((index, row))

    pairs: dict[int, dict[str, Any]] = {}
    for stream, source_rows in source_by_stream.items():
        candidates = production_by_stream.get(stream) or []
        positions_by_base: dict[str, list[int]] = defaultdict(list)
        for position, (_, candidate) in enumerate(candidates):
            positions_by_base[kernel_base(str(candidate.get("kernel_name") or ""))].append(
                position
            )
        cursor = 0
        for source in source_rows:
            source_base = kernel_base(str(source.get("kernel_name") or ""))
            positions = positions_by_base.get(source_base) or []
            start = bisect_left(positions, cursor)
            compatible = positions[start : start + 8]
            if not compatible:
                raise ValueError(
                    f"vLLM prefill stream {stream} lost eager semantic identity "
                    f"{source_base}"
                )
            source_exact = kernel_exact_identity(str(source.get("kernel_name") or ""))
            exact = [
                position
                for position in compatible
                if kernel_exact_identity(
                    str(candidates[position][1].get("kernel_name") or "")
                )
                == source_exact
            ]
            selected = exact[0] if exact else compatible[0]
            production_index, _ = candidates[selected]
            pairs[production_index] = source
            cursor = selected + 1
    return pairs


def attribute_vllm_production_events(
    production_rows: list[dict[str, Any]],
    eager_mapping_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eager_rows = load_jsonl(eager_mapping_path)
    eager_segments = anchor_segments(eager_rows)
    production_segments = anchor_segments(production_rows)
    if len(eager_segments) != len(production_segments):
        raise ValueError("vLLM eager/production occurrence segment count mismatch")

    diagnostics: dict[str, Any] = {
        "anchor_count": ATTN_RES_ANCHOR_COUNT,
        "segment_count": SEGMENT_COUNT,
        "exact_multiset_segment_count": 0,
        "mismatched_segments": [],
        "method_counts": Counter(),
        "support_class_counts": Counter(),
    }
    is_graph_decode = any(row.get("graph_node_id") is not None for row in production_rows)
    prefill_pairs: dict[int, dict[str, Any]] = {}
    if not is_graph_decode:
        eager_events_path = eager_mapping_path.with_name(
            eager_mapping_path.name.replace("kernel_mapping", "events")
        )
        if not eager_events_path.is_file():
            raise ValueError(f"missing eager event stream evidence: {eager_events_path}")
        prefill_pairs = _prefill_source_pairs(
            eager_rows, production_rows, eager_events_path
        )

    attributed: list[dict[str, Any]] = []
    for segment_id, ((eager_start, eager_stop), (prod_start, prod_stop)) in enumerate(
        zip(eager_segments, production_segments)
    ):
        eager_segment = eager_rows[eager_start:eager_stop]
        production_segment = production_rows[prod_start:prod_stop]
        eager_counts = Counter(
            kernel_base(str(row.get("kernel_name") or "")) for row in eager_segment
        )
        production_counts = Counter(
            kernel_base(str(row.get("kernel_name") or ""))
            for row in production_segment
        )
        if eager_counts == production_counts:
            diagnostics["exact_multiset_segment_count"] += 1
        else:
            diagnostics["mismatched_segments"].append(
                {
                    "segment_id": segment_id,
                    "eager_kernel_count": len(eager_segment),
                    "production_kernel_count": len(production_segment),
                    "production_only": dict(production_counts - eager_counts),
                    "eager_only": dict(eager_counts - production_counts),
                }
            )
            if is_graph_decode:
                raise ValueError(
                    f"vLLM graph decode segment {segment_id} eager/production "
                    "kernel multiset mismatch"
                )

        queues: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
        for row in eager_segment:
            queues[kernel_base(str(row.get("kernel_name") or ""))].append(row)
        scope = occurrence_for_segment(segment_id)
        for local_index, raw in enumerate(production_segment):
            row = dict(raw)
            row.update(scope)
            row["segment_id"] = segment_id
            base = kernel_base(str(row.get("kernel_name") or ""))
            production_index = prod_start + local_index
            source: dict[str, Any] | None = None
            if is_graph_decode:
                if queues[base]:
                    source = queues[base].popleft()
            else:
                source = prefill_pairs.get(production_index)
                if source is not None:
                    try:
                        queues[base].remove(source)
                    except ValueError as error:
                        raise ValueError(
                            f"vLLM prefill source escaped occurrence {segment_id}"
                        ) from error
            if source is not None and source.get("selected_node") != "runtime.step_setup":
                graph_region = (
                    "graph_node"
                    if row.get("graph_node_id") is not None
                    else "eager_break"
                )
                method = (
                    "attn_res_occurrence_bounded_normalized_identity_ordinal_"
                    f"{graph_region}"
                )
                row.update(
                    {
                        "node": source.get("selected_node"),
                        "kernel_label": source.get("cpu_op_name")
                        or source.get("selected_node"),
                        "attribution_method": method,
                        "confidence": "high",
                        "eager_event_id": source.get("event_id"),
                        "cpu_op_name": source.get("cpu_op_name"),
                        "primitive_frame": source.get("primitive_frame"),
                        "operator_frame": source.get("operator_frame"),
                        "semantic_frame": source.get("semantic_frame"),
                        "model_context_frame": source.get("model_context_frame"),
                    }
                )
                diagnostics["method_counts"][method] += 1
            else:
                support_class, support_reason = _support_contract(
                    segment_id, str(row.get("kernel_name") or "")
                )
                method = (
                    "eager_stack_proven_runtime_support"
                    if source is not None
                    else "production_only_explicit_runtime_support"
                )
                row.update(
                    {
                        "node": None,
                        "support_class": support_class,
                        "support_reason": support_reason,
                        "attribution_method": method,
                        "confidence": "high",
                        "eager_event_id": source.get("event_id") if source else None,
                    }
                )
                diagnostics["support_class_counts"][support_class] += 1
                diagnostics["method_counts"][method] += 1
            attributed.append(row)

        leftovers = [
            row
            for queue in queues.values()
            for row in queue
            if row.get("selected_node") != "runtime.step_setup"
        ]
        if leftovers:
            raise ValueError(
                f"vLLM segment {segment_id} lost semantic eager intervals: "
                + ", ".join(str(row.get("event_id")) for row in leftovers)
            )

    mapped = [row for row in attributed if row.get("node")]
    total_us = sum(float(row.get("dur_us") or 0.0) for row in attributed)
    mapped_us = sum(float(row.get("dur_us") or 0.0) for row in mapped)
    diagnostics.update(
        {
            "eager_kernel_count": len(eager_rows),
            "production_kernel_count": len(attributed),
            "mapped_kernel_count": len(mapped),
            "support_kernel_count": len(attributed) - len(mapped),
            "mapped_kernel_count_ratio": len(mapped) / len(attributed),
            "mapped_kernel_duration_ratio": mapped_us / total_us,
            "method_counts": dict(sorted(diagnostics["method_counts"].items())),
            "support_class_counts": dict(
                sorted(diagnostics["support_class_counts"].items())
            ),
        }
    )
    return attributed, diagnostics
