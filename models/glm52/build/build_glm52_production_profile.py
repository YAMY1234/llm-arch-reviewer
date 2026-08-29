#!/usr/bin/env python3
"""Build one GLM-5.2 production Profile v2 and Timeline v1 bundle.

Production Nsight captures provide timing but no Python stacks.  The graph-off
eager trace remains the binding source.  This adapter transfers eager IR nodes
only through deterministic evidence:

* the complete kernel-name sequence when it is identical;
* an identical normalized sequence inside each of 78 layer-anchor intervals;
* an eager-unique exact/base kernel signature or bounded occurrence order;
* an existing reviewed GLM model-unique kernel rule; or
* the eager-proven collective kind and launch order.

Anything else remains explicitly unmapped.  There is no nearest-neighbour or
greedy kernel-family matching.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.common.timeline_artifact import (  # noqa: E402
    attach_eager_stack_evidence,
    build_timeline_artifact,
    write_timeline_artifact,
)
from models.glm52.build.glm52_trace_rules import classify_glm52_node  # noqa: E402


MODEL_REVISION = "aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa"
FRAMEWORKS = {
    "sglang": {
        "execution_path_id": "tp8",
        "implementation_id": "sglang_fdebc938_dsa",
        "source_commit": "fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1",
        "container": "lmsysorg/sglang:v0.5.16-cu130",
        "anchor": "fmhasm100fkernel_qkve4m3obfloat16hqk576",
    },
    "trtllm": {
        "execution_path_id": "tp8_trtllm",
        "implementation_id": "trtllm_4358fb5d_dsa",
        "source_commit": "4358fb5d5222f76ba133c3ae630aa2c06e62d073",
        "container": "nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc22.post1",
        "anchor": {
            "prefill": "applymlaropeandassignqkvkerneloptcontext",
            "decode": "applymlaropeandassignqkvkernelgeneration",
        },
    },
}
LAYER_COUNT = 78
STRUCTURAL_NODE_STATES = {
    "top.token_ids": "input tensor boundary",
    "top.decoder_stack": "semantic container; timing is shown in drilled decoder nodes",
    "top.logits": "output tensor/materialization boundary",
    "stack.stack_in": "tensor boundary",
    "stack.schedule": "layer schedule control; no standalone GPU interval",
    "stack.dsa_attention": "semantic container; timing is shown in drilled attention nodes",
    "stack.feed_forward_schedule": "layer-conditioned control; no standalone GPU interval",
    "stack.dense_mlp": "semantic container; timing is shown in drilled dense-MLP nodes",
    "stack.moe": "semantic container; timing is shown in drilled MoE nodes",
    "stack.stack_out": "tensor boundary",
    "dsa_attention.attn_in": "tensor boundary",
    "dsa_attention.indexer_mode": "owner/reuse schedule control; no standalone GPU interval",
    "dsa_attention.attn_out": "tensor boundary",
    "dense_mlp.mlp_in": "tensor boundary",
    "dense_mlp.mlp_out": "tensor boundary",
    "moe.moe_in": "tensor boundary",
    "moe.moe_out": "tensor boundary",
}
STATE_NODE_STATES = {
    "dsa_attention.latent_kv_cache": "persistent FP8 latent-KV state read/write",
    "dsa_attention.index_k_cache": "persistent owner-layer index-K state read/write",
    "dsa_attention.index_topk_state": "cross-layer top-k state read/write",
}
FUSED_NODE_STATES = {
    "stack.attention_residual": (
        "stack.post_attention_norm",
        "residual dataflow boundary; no independently attributable production interval",
    ),
    "stack.feed_forward_residual": (
        "stack.input_norm",
        "residual dataflow boundary; no independently attributable production interval",
    ),
    "dsa_attention.kv_a_projection": (
        "dsa_attention.q_a_projection",
        "implementation-fused DSA latent projection interval",
    ),
    "dsa_attention.kv_latent_split": (
        "dsa_attention.q_split_rope",
        "implementation-fused Q/KV split and RoPE interval",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=tuple(FRAMEWORKS), required=True)
    parser.add_argument("--phase", choices=("prefill", "decode"), required=True)
    parser.add_argument("--batch-size", type=int, choices=(1, 16, 64, 256), required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--rank-rollup", type=Path, required=True)
    parser.add_argument("--eager-events", type=Path, required=True)
    parser.add_argument("--eager-mapping", type=Path, required=True)
    parser.add_argument("--output-profile", type=Path, required=True)
    parser.add_argument("--output-analysis", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_profile_nodes(framework: str) -> set[str]:
    model = yaml.safe_load((REPO_ROOT / "catalog/glm52/model_ir.yaml").read_text())
    nodes = {
        f"{view_id}.{node['id']}"
        for view_id, view in model["views"].items()
        for node in view["nodes"]
    }
    execution_path_id = str(FRAMEWORKS[framework]["execution_path_id"])
    plan = yaml.safe_load(
        (
            REPO_ROOT
            / "catalog/glm52/execution_paths"
            / f"{execution_path_id}.yaml"
        ).read_text()
    )
    for transform in plan["transforms"]:
        if transform["op"] != "insert_after":
            continue
        view_id = str(transform["after"]).split(".", 1)[0]
        nodes.add(f"{view_id}.{transform['node']['id']}")
    return nodes


def build_profile_node_states(
    *,
    node_metrics: dict[str, Any],
    framework: str,
    phase: str,
    unmapped_label: str | None = None,
) -> dict[str, dict[str, str]]:
    states: dict[str, dict[str, str]] = {
        node: {"status": "structural", "label": label}
        for node, label in STRUCTURAL_NODE_STATES.items()
        if node not in node_metrics
    }
    for node, label in STATE_NODE_STATES.items():
        if node not in node_metrics:
            states[node] = {"status": "state", "label": label}
    for node, (included_in, label) in FUSED_NODE_STATES.items():
        if node not in node_metrics:
            states[node] = {
                "status": "fused",
                "label": label,
                "included_in": included_in,
            }

    states["top.mtp_extension"] = {
        "status": "not_selected",
        "label": "MTP off autoregressive baseline",
    }
    for node in expected_profile_nodes(framework):
        if node.startswith("mtp_extension."):
            states[node] = {
                "status": "not_selected",
                "label": "MTP off autoregressive baseline",
            }

    if phase == "decode" and "moe.dispatch" not in node_metrics:
        states["moe.dispatch"] = {
            "status": "fused",
            "label": "decode token routing is fused into the routed-expert runner",
            "included_in": "moe.routed_experts",
        }
    prefill_collective = "dsa_attention.tp_prefill_index_topk_all_gather"
    if phase == "decode" and prefill_collective in expected_profile_nodes(framework):
        states[prefill_collective] = {
            "status": "not_selected",
            "label": "prefill-only TP index-row all-gather",
        }

    for node in expected_profile_nodes(framework):
        if node in node_metrics or node in states:
            continue
        states[node] = {
            "status": "unmapped",
            "label": unmapped_label
            or (
                "semantic node executes in this phase, but its production kernels "
                "remain ambiguous under fail-closed eager attribution transfer"
            ),
        }
    return states


def attribute_aggregate_graph_events(
    production_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Preserve graph-level timing without inventing hidden kernel attribution."""

    if not production_rows:
        raise ValueError("aggregate CUDA Graph capture has no activity event")
    if any(row.get("kind") == "kernel" for row in production_rows):
        raise ValueError("aggregate CUDA Graph path unexpectedly contains kernel events")
    for row in production_rows:
        row.update(
            {
                "node": None,
                "kernel_label": row.get("kernel_name") or "CUDA Graph aggregate",
                "attribution_method": "aggregate_cuda_graph_no_node_visibility",
                "confidence": "unmapped",
            }
        )
    total_activity_us = sum(float(row["dur_us"]) for row in production_rows)
    return production_rows, {
        "observability": "aggregate_cuda_graph",
        "eager_kernel_count": 0,
        "production_kernel_count": 0,
        "layer_anchor_count": 0,
        "exact_segment_count": 0,
        "normalized_segment_count": 0,
        "mismatched_segment_count": 0,
        "mismatched_segments": [],
        "collective_count": 0,
        "collective_auxiliary_kernel_count": 0,
        "collective_kind_counts": {},
        "method_counts": {
            "aggregate_cuda_graph_no_node_visibility": len(production_rows)
        },
        "total_kernel_us": 0.0,
        "total_activity_us": total_activity_us,
        "mapped_kernel_us": 0.0,
        "mapped_kernel_count": 0,
        "unmapped_kernel_count": 0,
        "mapped_kernel_count_ratio": 0.0,
        "mapped_kernel_duration_ratio": 0.0,
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def kernel_exact_identity(name: str) -> str:
    """Canonicalize spelling only while retaining all template/shape detail."""

    normalized = re.sub(r"^void\s+", "", name.strip().lower())
    normalized = normalized.replace("<unnamed>", "(anonymous namespace)")
    normalized = re.sub(r"\(int\)(?=\d)", "", normalized)
    normalized = re.sub(r"\bconst\s+([\w:]+)", r"\1 const", normalized)
    normalized = re.sub(r"\s*\*\s*", "*", normalized)
    normalized = re.sub(r"\s*,\s*", ",", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def kernel_base(name: str) -> str:
    """Return a shape-preserving function identity, not a fuzzy family."""

    normalized = kernel_exact_identity(name)
    if "<" in normalized:
        normalized = normalized.split("<", 1)[0]
    else:
        normalized = re.sub(r"\([^()]*\)\s*$", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def sequence_family(name: str) -> str:
    """Normalize only shape/version digits for bounded sequence equality."""

    family = kernel_base(name)
    family = re.sub(r"0x[0-9a-f]+", "#", family)
    family = re.sub(r"\d+", "#", family)
    return family


def schedule_family(name: str) -> str:
    """Normalize a codegen schedule while retaining the semantic BMM form."""

    family = sequence_family(name)
    if family.startswith("bmm_"):
        family = re.sub(r"_bn_.*?_rgtma_", "_bn_<schedule>_rgtma_", family)
    return family


def is_anchor(name: str, framework: str, phase: str) -> bool:
    anchor = FRAMEWORKS[framework]["anchor"]
    if isinstance(anchor, dict):
        anchor = anchor[phase]
    return str(anchor) in name.lower()


def source_collective_kind(row: dict[str, Any]) -> str | None:
    node = str(row.get("selected_node") or "")
    if "all_gather" in node:
        return "all_gather"
    if "collective" in node:
        return "all_reduce"
    return None


def trtllm_layer_collective_node(layer: int, slot: int) -> str:
    if not 0 <= layer < LAYER_COUNT or slot not in (0, 1):
        raise ValueError(f"invalid TRT-LLM layer collective coordinate: {layer}/{slot}")
    if slot == 0:
        return "dsa_attention.tp_attention_output_collective"
    return (
        "dense_mlp.tp_dense_mlp_output_collective"
        if layer < 3
        else "moe.tp_moe_output_collective"
    )


def inferred_collective_kind(name: str) -> str | None:
    lowered = name.lower()
    if "allgather" in lowered or "all_gather" in lowered:
        return "all_gather"
    if "reducescatter" in lowered or "reduce_scatter" in lowered:
        return "reduce_scatter"
    if "allreduce" in lowered or "all_reduce" in lowered:
        return "all_reduce"
    return None


def interval_union_us(intervals: Iterable[tuple[float, float]]) -> float:
    merged: list[list[float]] = []
    for start, stop in sorted(intervals):
        if stop <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, stop])
        else:
            merged[-1][1] = max(merged[-1][1], stop)
    return sum(stop - start for start, stop in merged)


def eager_rows(
    events_path: Path, mapping_path: Path
) -> list[dict[str, Any]]:
    events = {row["event_id"]: row for row in load_jsonl(events_path)}
    rows: list[dict[str, Any]] = []
    for mapping in load_jsonl(mapping_path):
        event = events.get(mapping["event_id"])
        if event is None:
            raise ValueError(f"missing eager event {mapping['event_id']}")
        rows.append(
            {
                **event,
                "selected_node": mapping.get("selected_node"),
                "source_confidence": mapping.get("confidence"),
                "cpu_op_name": mapping.get("cpu_op_name") or event.get("cpu_op_name"),
            }
        )
    return sorted(rows, key=lambda row: (float(row["ts_us"]), row["event_id"]))


def _assign(
    row: dict[str, Any],
    source: dict[str, Any],
    *,
    method: str,
    confidence: str,
    overwrite: bool = False,
) -> None:
    if row.get("node") is not None and not overwrite:
        return
    node = source.get("selected_node")
    if not node:
        if row.get("node") is None:
            row["attribution_method"] = "eager_source_explicit_unmapped"
            row["confidence"] = "unmapped"
        return
    row.update(
        {
            "node": node,
            "kernel_label": source.get("cpu_op_name") or node,
            "attribution_method": method,
            "confidence": confidence,
            "eager_event_id": source.get("event_id"),
        }
    )


def unique_source_index(
    source_rows: list[dict[str, Any]], key_fn
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        grouped[key_fn(str(row["kernel_name"]))].append(row)
    result: dict[str, dict[str, Any]] = {}
    for key, rows in grouped.items():
        mapped_rows = [row for row in rows if row.get("selected_node")]
        nodes = {row.get("selected_node") for row in mapped_rows}
        if len(nodes) == 1:
            # An explicit source gap supplies no positive attribution, but it
            # also does not contradict the only eager-proven node for the same
            # identity.  Always copy from a mapped row, never from the gap.
            result[key] = mapped_rows[0]
    return result


def anchor_segments(rows: list[dict[str, Any]], framework: str, phase: str) -> list[tuple[int, int]]:
    anchors = [
        index
        for index, row in enumerate(rows)
        if is_anchor(str(row["kernel_name"]), framework, phase)
    ]
    if len(anchors) != LAYER_COUNT:
        raise ValueError(
            f"{framework} {phase}: expected {LAYER_COUNT} layer anchors, got {len(anchors)}"
        )
    # Compare exactly one anchor-led interval per model layer.  Eager and
    # production captures may begin at different wrapper boundaries (for
    # example, production can include embedding/preamble kernels while the
    # eager template begins at layer 0), so a pre-anchor prefix is not a
    # layer segment and remains eligible only for independent exact evidence.
    boundaries = [*anchors, len(rows)]
    return [
        (start, stop)
        for start, stop in zip(boundaries, boundaries[1:])
        if stop > start
    ]


def attribute_events(
    *,
    production_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    framework: str,
    phase: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kernels = [row for row in production_rows if row.get("kind") == "kernel"]
    for row in production_rows:
        row.update(
            {
                "node": None,
                "kernel_label": row.get("kernel_name"),
                "attribution_method": (
                    "explicit_unmapped_kernel"
                    if row.get("kind") == "kernel"
                    else "explicit_unmapped_non_kernel_activity"
                ),
                "confidence": "unmapped",
            }
        )

    eager_node_examples = {
        str(source["selected_node"]): source
        for source in source_rows
        if source.get("selected_node")
    }
    if framework == "sglang":
        for row in kernels:
            node, confidence = classify_glm52_node(
                str(row["kernel_name"]), row.get("cpu_op_name"), []
            )
            if node:
                if node not in eager_node_examples:
                    raise ValueError(f"semantic rule node has no eager evidence: {node}")
                _assign(
                    row,
                    eager_node_examples[node],
                    method="existing_glm52_kernel_semantic_rule",
                    confidence=confidence,
                )

    exact_unique = unique_source_index(source_rows, kernel_exact_identity)
    base_unique = unique_source_index(source_rows, kernel_base)
    for row in kernels:
        name = str(row["kernel_name"])
        exact_identity = kernel_exact_identity(name)
        if exact_identity in exact_unique:
            _assign(
                row,
                exact_unique[exact_identity],
                method="eager_unique_exact_kernel_name",
                confidence="high",
            )
        elif kernel_base(name) in base_unique:
            _assign(
                row,
                base_unique[kernel_base(name)],
                method="eager_unique_function_identity",
                confidence="medium",
            )

    source_segments = anchor_segments(source_rows, framework, phase)
    production_segments = anchor_segments(kernels, framework, phase)
    if len(source_segments) != len(production_segments):
        raise ValueError("source/production layer segment count mismatch")

    exact_segment_count = 0
    normalized_segment_count = 0
    mismatched_segments: list[dict[str, Any]] = []
    for segment_id, ((source_start, source_stop), (prod_start, prod_stop)) in enumerate(
        zip(source_segments, production_segments)
    ):
        source_segment = source_rows[source_start:source_stop]
        prod_segment = kernels[prod_start:prod_stop]
        source_names = [str(row["kernel_name"]) for row in source_segment]
        prod_names = [str(row["kernel_name"]) for row in prod_segment]
        exact = source_names == prod_names
        normalized = (
            len(source_names) == len(prod_names)
            and [sequence_family(name) for name in source_names]
            == [sequence_family(name) for name in prod_names]
        )
        if not normalized:
            source_by_exact: dict[str, list[dict[str, Any]]] = defaultdict(list)
            production_by_exact: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for source in source_segment:
                source_by_exact[kernel_exact_identity(str(source["kernel_name"]))].append(source)
            for production in prod_segment:
                production_by_exact[
                    kernel_exact_identity(str(production["kernel_name"]))
                ].append(production)
            for identity, production_matches in production_by_exact.items():
                source_matches = source_by_exact.get(identity, [])
                if (
                    len(source_matches) == len(production_matches)
                    and source_matches
                    and all(source.get("selected_node") for source in source_matches)
                ):
                    for production, source in zip(production_matches, source_matches):
                        _assign(
                            production,
                            source,
                            method="hard_anchor_bounded_exact_identity_occurrence_order",
                            confidence="high",
                        )
            segment_exact_unique = unique_source_index(
                source_segment, kernel_exact_identity
            )
            segment_base_unique = unique_source_index(source_segment, kernel_base)
            segment_family_unique = unique_source_index(
                source_segment, sequence_family
            )
            segment_schedule_unique = unique_source_index(
                source_segment, schedule_family
            )
            for production in prod_segment:
                name = str(production["kernel_name"])
                exact_identity = kernel_exact_identity(name)
                if exact_identity in segment_exact_unique:
                    _assign(
                        production,
                        segment_exact_unique[exact_identity],
                        method="hard_anchor_bounded_unique_exact_kernel_name",
                        confidence="high",
                    )
                elif kernel_base(name) in segment_base_unique:
                    _assign(
                        production,
                        segment_base_unique[kernel_base(name)],
                        method="hard_anchor_bounded_unique_function_identity",
                        confidence="medium",
                    )
                elif sequence_family(name) in segment_family_unique:
                    _assign(
                        production,
                        segment_family_unique[sequence_family(name)],
                        method="hard_anchor_bounded_unique_normalized_identity",
                        confidence="medium",
                    )
                elif schedule_family(name) in segment_schedule_unique:
                    _assign(
                        production,
                        segment_schedule_unique[schedule_family(name)],
                        method="hard_anchor_bounded_unique_codegen_schedule_family",
                        confidence="medium",
                    )
            mismatched_segments.append(
                {
                    "segment_id": segment_id,
                    "eager_kernel_count": len(source_names),
                    "production_kernel_count": len(prod_names),
                }
            )
            continue
        exact_segment_count += int(exact)
        normalized_segment_count += int(not exact)
        method = (
            "hard_anchor_bounded_exact_kernel_sequence"
            if exact
            else "hard_anchor_bounded_exact_normalized_sequence"
        )
        confidence = "high" if exact else "medium"
        for production, source in zip(prod_segment, source_segment):
            _assign(
                production,
                source,
                method=method,
                confidence=confidence,
                overwrite=True,
            )

    source_collective_base_kinds: dict[str, set[str | None]] = defaultdict(set)
    for source in source_rows:
        source_collective_base_kinds[kernel_base(str(source["kernel_name"]))].add(
            source_collective_kind(source)
        )
    source_proven_collective_bases = {
        base: next(iter(kinds))
        for base, kinds in source_collective_base_kinds.items()
        if len(kinds) == 1 and None not in kinds
    }

    def production_collective_kind(row: dict[str, Any]) -> str | None:
        node = str(row.get("node") or "")
        if "all_gather" in node:
            return "all_gather"
        if "collective" in node:
            return "all_reduce"
        if framework == "sglang":
            lowered_name = str(row["kernel_name"]).lower()
            if "rmsnormlamport" in lowered_name:
                # Auxiliary kernel of the same two-shot collective; paired to
                # its primary launch after logical collective-order checking.
                return None
            # The SGLang eager binding proves that these exact function
            # identities are used only for one collective kind.  CUDA Graph
            # replay shortens the surrounding sequence, so transfer that
            # source-derived identity and still require the complete eager
            # collective order below.  TRT-LLM is intentionally excluded:
            # its fused kernel identity is reused at ambiguous boundaries.
            source_proven = source_proven_collective_bases.get(
                kernel_base(str(row["kernel_name"]))
            )
            if source_proven:
                return source_proven
            # Higher decode batches select SGLang's two-shot implementation
            # instead of the eager BS1 one-shot identity.  Accept only an
            # explicit collective symbol and still require the exact 158-item
            # eager collective-kind order before attribution.
            return inferred_collective_kind(str(row["kernel_name"]))
        # Do not promote an unmapped all-reduce-looking kernel here.  The
        # locked TRT implementation reuses the same fused kernel identity at
        # multiple module boundaries, and only the eager-bounded sequence can
        # distinguish them without guessing.
        return None

    collective_auxiliary_count = 0
    if framework == "trtllm":
        production_collectives: list[tuple[dict[str, Any], str]] = []
        for layer, ((source_start, source_stop), (prod_start, prod_stop)) in enumerate(
            zip(source_segments, production_segments)
        ):
            source_segment = source_rows[source_start:source_stop]
            production_segment = kernels[prod_start:prod_stop]
            source_primary_token = (
                "twoshotallreducekernel"
                if phase == "prefill"
                else "oneshotallreducefusionkernel"
            )
            source_primaries = [
                row
                for row in source_segment
                if source_primary_token in str(row["kernel_name"]).lower()
            ]
            production_primaries = [
                row
                for row in production_segment
                if "twoshotallreducekernel"
                in str(row["kernel_name"]).lower()
            ]
            require_count = 2
            if len(source_primaries) != require_count or len(production_primaries) != require_count:
                raise ValueError(
                    f"TRT-LLM layer {layer}: expected two eager/production all-reduce "
                    f"primaries, got {len(source_primaries)}/{len(production_primaries)}"
                )

            source_by_primary_id: dict[int, dict[str, Any]] = {}
            for slot, (production, source) in enumerate(
                zip(production_primaries, source_primaries)
            ):
                evidence = {
                    **source,
                    "selected_node": trtllm_layer_collective_node(layer, slot),
                }
                _assign(
                    production,
                    evidence,
                    method="trtllm_eager_validated_layer_collective_schedule",
                    confidence="high",
                    overwrite=True,
                )
                source_by_primary_id[id(production)] = evidence
                production_collectives.append((production, "all_reduce"))

            current_primary: dict[str, Any] | None = None
            auxiliary_count = 0
            for row in production_segment:
                if id(row) in source_by_primary_id:
                    current_primary = source_by_primary_id[id(row)]
                    continue
                if "rmsnormlamport" not in str(row["kernel_name"]).lower():
                    continue
                if current_primary is None:
                    raise ValueError(
                        f"TRT-LLM layer {layer}: collective auxiliary precedes primary"
                    )
                _assign(
                    row,
                    current_primary,
                    method="trtllm_twoshot_collective_auxiliary_to_preceding_primary",
                    confidence="high",
                    overwrite=True,
                )
                auxiliary_count += 1
            expected_auxiliaries = 1 if layer < 3 else 2
            if auxiliary_count != expected_auxiliaries:
                raise ValueError(
                    f"TRT-LLM layer {layer}: expected {expected_auxiliaries} two-shot "
                    f"auxiliaries, got {auxiliary_count}"
                )
            collective_auxiliary_count += auxiliary_count

        source_all_gathers = [
            row
            for row in source_rows
            if inferred_collective_kind(str(row["kernel_name"])) == "all_gather"
        ]
        production_all_gathers = [
            row
            for row in kernels
            if inferred_collective_kind(str(row["kernel_name"])) == "all_gather"
        ]
        if phase == "decode":
            if len(source_all_gathers) != 1 or len(production_all_gathers) != 1:
                raise ValueError(
                    "TRT-LLM decode expected one eager/production logits all-gather, got "
                    f"{len(source_all_gathers)}/{len(production_all_gathers)}"
                )
            all_gather_evidence = [source_all_gathers[0]]
        else:
            source_dsa_gathers = [
                row
                for row in source_all_gathers
                if row.get("selected_node")
                == "dsa_attention.tp_prefill_index_topk_all_gather"
            ]
            source_logits_gathers = [
                row
                for row in source_all_gathers
                if row.get("selected_node") == "top.tp_logits_all_gather"
            ]
            if (
                not source_dsa_gathers
                or len(source_logits_gathers) != 1
                or len(production_all_gathers) != 22
            ):
                raise ValueError(
                    "TRT-LLM prefill expected eager DSA/logits templates and 22 "
                    f"production all-gathers, got {len(source_dsa_gathers)}/"
                    f"{len(source_logits_gathers)}/{len(production_all_gathers)}"
                )
            # The exact eager extraction begins at layer 0's attention anchor
            # and therefore omits the first of the 21 owner-layer gathers.
            # The passing full eager attestation and Execution IR lock the
            # invocation scope at 21; transfer one reviewed DSA template to
            # those 21 ordered occurrences and retain the final logits event.
            all_gather_evidence = [source_dsa_gathers[0]] * 21 + [
                source_logits_gathers[0]
            ]
        for production, source in zip(production_all_gathers, all_gather_evidence):
            _assign(
                production,
                source,
                method=(
                    "trtllm_eager_validated_prefill_all_gather_scope_and_order"
                    if phase == "prefill"
                    else "eager_validated_collective_kind_and_order"
                ),
                confidence="high",
                overwrite=True,
            )
            production_collectives.append((production, "all_gather"))
        source_kinds = ["all_reduce"] * (2 * LAYER_COUNT) + [
            "all_gather"
        ] * len(all_gather_evidence)
        production_kinds = [kind for _, kind in production_collectives]
    else:
        source_collectives = [
            (row, kind)
            for row in source_rows
            if (kind := source_collective_kind(row)) is not None
        ]
        production_collectives = [
            (row, kind)
            for row in kernels
            if (kind := production_collective_kind(row)) is not None
        ]
        source_kinds = [kind for _, kind in source_collectives]
        production_kinds = [kind for _, kind in production_collectives]
        if source_kinds != production_kinds:
            raise ValueError(
                "production collective sequence differs from eager binding: "
                f"eager={Counter(source_kinds)}, production={Counter(production_kinds)}"
            )
        for (production, _), (source, _) in zip(
            production_collectives, source_collectives
        ):
            _assign(
                production,
                source,
                method="eager_validated_collective_kind_and_order",
                confidence="high",
                overwrite=True,
            )

        twoshot_primaries = [
            row
            for row in kernels
            if "twoshotallreducekernel" in str(row["kernel_name"]).lower()
        ]
        twoshot_auxiliaries = [
            row
            for row in kernels
            if "rmsnormlamport" in str(row["kernel_name"]).lower()
        ]
        if twoshot_primaries or twoshot_auxiliaries:
            if len(twoshot_primaries) != len(twoshot_auxiliaries):
                raise ValueError(
                    "SGLang two-shot collective primary/auxiliary count differs: "
                    f"{len(twoshot_primaries)} != {len(twoshot_auxiliaries)}"
                )
            for auxiliary, primary in zip(twoshot_auxiliaries, twoshot_primaries):
                if not primary.get("node"):
                    raise ValueError("unmapped SGLang two-shot collective primary")
                _assign(
                    auxiliary,
                    {
                        "selected_node": primary["node"],
                        "cpu_op_name": primary.get("kernel_label"),
                        "event_id": primary.get("eager_event_id"),
                    },
                    method="sglang_twoshot_collective_auxiliary_occurrence_order",
                    confidence="high",
                    overwrite=True,
                )
            collective_auxiliary_count = len(twoshot_auxiliaries)

    total_kernel_us = sum(float(row["dur_us"]) for row in kernels)
    missing_eager_provenance = [
        row for row in kernels if row.get("node") is not None and not row.get("eager_event_id")
    ]
    if missing_eager_provenance:
        raise ValueError(
            f"{len(missing_eager_provenance)} mapped kernels lack eager evidence IDs"
        )
    mapped_kernel_us = sum(
        float(row["dur_us"]) for row in kernels if row.get("node") is not None
    )
    mapped_kernel_count = sum(row.get("node") is not None for row in kernels)
    method_counts = Counter(str(row["attribution_method"]) for row in kernels)
    report = {
        "eager_kernel_count": len(source_rows),
        "production_kernel_count": len(kernels),
        "layer_anchor_count": LAYER_COUNT,
        "exact_segment_count": exact_segment_count,
        "normalized_segment_count": normalized_segment_count,
        "mismatched_segment_count": len(mismatched_segments),
        "mismatched_segments": mismatched_segments,
        "collective_count": len(production_collectives),
        "collective_auxiliary_kernel_count": collective_auxiliary_count,
        "collective_kind_counts": dict(Counter(production_kinds)),
        "method_counts": dict(method_counts),
        "total_kernel_us": total_kernel_us,
        "mapped_kernel_us": mapped_kernel_us,
        "mapped_kernel_count": mapped_kernel_count,
        "unmapped_kernel_count": len(kernels) - mapped_kernel_count,
        "mapped_kernel_count_ratio": (
            mapped_kernel_count / len(kernels) if kernels else 0.0
        ),
        "mapped_kernel_duration_ratio": (
            mapped_kernel_us / total_kernel_us if total_kernel_us else 0.0
        ),
    }
    return production_rows, report


def build_node_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    node_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("kind") == "kernel" and row.get("node"):
            node_rows[str(row["node"])].append(row)

    metrics: dict[str, Any] = {}
    for node, events in sorted(node_rows.items()):
        active_us = interval_union_us(
            (float(event["ts_us"]), float(event["ts_us"]) + float(event["dur_us"]))
            for event in events
        )
        residency_us = sum(float(event["dur_us"]) for event in events)
        label_duration: Counter[str] = Counter()
        label_count: Counter[str] = Counter()
        for event in events:
            label = str(event.get("kernel_label") or event["kernel_name"][:120])
            label_duration[label] += float(event["dur_us"])
            label_count[label] += 1
        kernels = []
        for label, duration_us in label_duration.most_common():
            count = label_count[label]
            kernels.append(
                {
                    "name": label,
                    "count": count,
                    "count_per_iter": float(count),
                    "avg_us": round(duration_us / count, 6),
                    "total_us_per_iter": round(duration_us, 6),
                    "share_in_node_residency_pct": round(
                        100.0 * duration_us / residency_us, 4
                    ),
                }
            )
        metrics[node] = {
            "ms_per_iter": round(active_us / 1000.0, 6),
            "gpu_residency_ms_per_iter": round(residency_us / 1000.0, 6),
            "timing_semantics": "same-rank active interval union",
            "kernels": kernels,
        }
    return metrics


def profile_identity(framework: str, phase: str, batch_size: int) -> tuple[str, str]:
    if phase == "prefill":
        return (
            f"glm52_tp8_{framework}_prefill_bs1_8k",
            "eager_prefill_bs1_8k",
        )
    return (
        f"glm52_tp8_{framework}_cg_decode_bs{batch_size}_8k1k",
        f"cg_decode_bs{batch_size}_8k1k",
    )


def main() -> int:
    args = parse_args()
    if args.phase == "prefill" and args.batch_size != 1:
        raise ValueError("prefill production profile is defined only for BS1")

    framework = FRAMEWORKS[args.framework]
    validation = json.loads(args.validation.read_text())
    rank_rollup = json.loads(args.rank_rollup.read_text())
    if validation.get("status") != "pass":
        raise ValueError("production validation did not pass")
    if validation.get("framework") != args.framework:
        raise ValueError("framework differs from validation")
    if validation.get("phase") != args.phase:
        raise ValueError("phase differs from validation")
    if int(validation.get("global_batch_size")) != args.batch_size:
        raise ValueError("batch size differs from validation")
    reference_rank = int(validation["reference_rank"])
    if int(rank_rollup["reference_rank"]) != reference_rank:
        raise ValueError("rank rollup/reference validation mismatch")

    production = load_jsonl(args.events)
    aggregate_graph = (
        args.framework == "trtllm"
        and args.phase == "decode"
        and validation.get("graph_state", {}).get("cuda_graph_trace") == "graph"
    )
    if aggregate_graph:
        attributed, attribution = attribute_aggregate_graph_events(production)
    else:
        source = eager_rows(args.eager_events, args.eager_mapping)
        attributed, attribution = attribute_events(
            production_rows=production,
            source_rows=source,
            framework=args.framework,
            phase=args.phase,
        )
        attributed = attach_eager_stack_evidence(
            attributed, mapping_path=args.eager_mapping
        )
    node_metrics = build_node_metrics(attributed)
    node_states = build_profile_node_states(
        node_metrics=node_metrics,
        framework=args.framework,
        phase=args.phase,
        unmapped_label=(
            "Nsight captured this CUDA Graph replay as one aggregate activity; "
            "the node executes in the selected decode step, but per-kernel and "
            "per-IR-node timing is not observable in this accepted capture"
            if aggregate_graph
            else None
        ),
    )
    profile_id, variant_id = profile_identity(
        args.framework, args.phase, args.batch_size
    )

    activity = validation["activity"]
    timing_summary = {
        "elapsed_ms": activity["elapsed_ms"],
        "active_gpu_ms": activity["active_gpu_ms"],
        "gpu_residency_ms": activity["gpu_residency_ms"],
        "device_gap_ms": activity["device_gap_ms"],
        "gpu_overlap_ms": activity["gpu_overlap_ms"],
        "semantics": "selected reference-rank executor-step GPU activity envelope",
    }
    trace_start_us = float(activity["start_ns"]) / 1000.0
    timeline_path = args.output_profile.with_suffix(".timeline.json.gz")
    timeline = build_timeline_artifact(
        profile_id=profile_id,
        phase=args.phase,
        reference_rank=reference_rank,
        steps=[
            {
                "step_index": 1,
                "label": f"formal {args.phase} BS{args.batch_size}",
                "trace_start_us": trace_start_us,
                "duration_us": float(activity["elapsed_ms"]) * 1000.0,
                "events": attributed,
            }
        ],
        timing_summary=timing_summary,
        raw_trace={
            "file": Path(validation["trace"]["raw_trace"]).name,
            "sha256": validation["trace"]["raw_trace_sha256"],
            "format": "nsight_systems_nsys_rep",
            "rank": reference_rank,
            "storage": "task_evidence_only",
        },
        stack_source={
            "source": (
                "not_available_for_aggregate_cuda_graph"
                if aggregate_graph
                else "graph_off_eager_trace"
            ),
            "mapping_file": args.eager_mapping.name,
            "mapping_sha256": sha256_file(args.eager_mapping),
            "policy": (
                "aggregate CUDA Graph timing is preserved without kernel or IR-node "
                "attribution because graph-level Nsight capture hides node events"
                if aggregate_graph
                else (
                    "production timing plus eager stack evidence; per-event match provenance "
                    "is preserved and unmatched events remain explicit"
                )
            ),
        },
    )
    timeline_sha = write_timeline_artifact(timeline_path, timeline)

    graph_enabled = args.phase == "decode"
    profile = {
        "schema_version": "profile.v2",
        "profile_id": profile_id,
        "label": (
            f"GB300 · {args.framework} · pure TP8 · "
            f"{'CUDA Graph decode' if graph_enabled else 'eager prefill'} · "
            f"BS{args.batch_size} · 8k→1k"
        ),
        "model_id": "glm52",
        "execution_path_id": framework["execution_path_id"],
        "implementation_id": framework["implementation_id"],
        "variant_id": variant_id,
        "phase": args.phase,
        "generation_mode": "autoregressive",
        "entry_view": "top",
        "execution_parameters": {
            "tp_size": 8,
            "dp_size": 1,
            "cp_size": 1,
            "ep_size": 1,
        },
        "hardware": {
            "gpu": "GB300",
            "gpus_per_node": 4,
            "nodes": 2,
            "cluster": "CMH",
        },
        "workload": {
            "isl": 8192,
            "osl": 1024,
            "batch_size": args.batch_size,
            "concurrency": args.batch_size,
            "warmup_requests": 3 * args.batch_size,
            "formal_requests": args.batch_size,
            "request_trajectory": [1, 16, 64, 256],
            "prompt_source": "deterministic_random_token_ids",
            "prompt_seed": 0,
            "ignore_eos": True,
            "prefix_cache_enabled": False,
            "hicache_enabled": False,
            "kv_offload_enabled": False,
        },
        "profiler": {
            "type": "nsight_systems",
            "representative_rank": reference_rank,
            "cuda_graph_enabled": graph_enabled,
            "cuda_graph_trace": validation.get("graph_state", {}).get(
                "cuda_graph_trace", "node"
            ),
            "event_visibility": (
                "aggregate_cuda_graph"
                if aggregate_graph
                else "kernel_and_device_activity"
            ),
            "with_stack": False,
            "capture_control": validation.get("capture_control", {}),
            "selected_runtime_coordinate": validation["actual_window"],
            "activity_boundary": validation["exact_activity_boundary"],
            "gpu_metric_semantics": "same-rank active interval union per selected step",
        },
        "evidence": {
            "job_id": validation["job_id"],
            "source_commit": framework["source_commit"],
            "model_revision": MODEL_REVISION,
            "container": framework["container"],
            "validation_file": args.validation.name,
            "validation_sha256": sha256_file(args.validation),
            "rank_rollup_file": args.rank_rollup.name,
            "rank_rollup_sha256": sha256_file(args.rank_rollup),
            "rank_aggregation_policy": rank_rollup["aggregation_policy"],
            "validated_rank_count": len(rank_rollup["observed_ranks"]),
            "full_step_rank_count": len(rank_rollup["full_step_ranks"]),
            "reference_rank": reference_rank,
            "raw_trace_sha256": validation["trace"]["raw_trace_sha256"],
            "eager_mapping_sha256": sha256_file(args.eager_mapping),
            "mapping_policy": (
                "aggregate CUDA Graph activity only; per-node timing is explicitly "
                "unmapped and no eager kernel attribution is attempted"
                if aggregate_graph
                else (
                    "78-layer hard-anchor bounded exact/normalized sequence and occurrence "
                    "order, eager-unique identity, reviewed GLM semantic signatures, and "
                    "eager-validated collective order; no greedy matching"
                )
            ),
            "mapped_kernel_count_ratio": round(
                attribution["mapped_kernel_count_ratio"], 8
            ),
            "mapped_kernel_duration_ratio": round(
                attribution["mapped_kernel_duration_ratio"], 8
            ),
            "timing": timing_summary,
        },
        "timeline": {
            "schema_version": "timeline.v1",
            "artifact": timeline_path.name,
            "sha256": timeline_sha,
            "reference_rank": reference_rank,
            "step_count": 1,
            "event_count": len(attributed),
            "raw_trace_file": Path(validation["trace"]["raw_trace"]).name,
        },
        "node_states": node_states,
        "node_metrics": node_metrics,
    }

    args.output_profile.parent.mkdir(parents=True, exist_ok=True)
    args.output_profile.write_text(
        yaml.safe_dump(profile, sort_keys=False, allow_unicode=True)
    )
    unmapped = Counter()
    for row in attributed:
        if row.get("kind") == "kernel" and row.get("node") is None:
            unmapped[str(row["kernel_name"])] += float(row["dur_us"])
    analysis = {
        "schema_version": "glm52-production-attribution.v1",
        "profile_id": profile_id,
        "status": "pass",
        "reference_rank": reference_rank,
        "validation": validation,
        "rank_rollup": rank_rollup,
        "attribution": attribution,
        "node_metrics": node_metrics,
        "top_unmapped_kernels": [
            {"name": name, "total_us": round(duration, 6)}
            for name, duration in unmapped.most_common(40)
        ],
    }
    args.output_analysis.parent.mkdir(parents=True, exist_ok=True)
    args.output_analysis.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output_profile.resolve()}")
    print(f"wrote {timeline_path.resolve()}")
    print(f"wrote {args.output_analysis.resolve()}")
    print(
        f"mapped={attribution['mapped_kernel_duration_ratio']:.6f}, "
        f"segments={attribution['exact_segment_count']} exact + "
        f"{attribution['normalized_segment_count']} normalized, "
        f"{attribution['mismatched_segment_count']} mismatched"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
