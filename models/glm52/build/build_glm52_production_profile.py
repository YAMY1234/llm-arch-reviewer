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
from models.common.trace_mapping import FrameRef  # noqa: E402
from models.glm52.build.glm52_trace_rules import classify_glm52_node  # noqa: E402
from models.glm52.build.glm52_trtllm_trace_rules import (  # noqa: E402
    classify_trtllm_node,
)


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
                "primitive_frame": mapping.get("primitive_frame"),
                "operator_frame": mapping.get("operator_frame"),
                "semantic_frame": mapping.get("semantic_frame"),
                "model_context_frame": mapping.get("model_context_frame"),
                "phase_frame": mapping.get("phase_frame"),
            }
        )
    return sorted(rows, key=lambda row: (float(row["ts_us"]), row["event_id"]))


def _source_stack(row: dict[str, Any]) -> list[FrameRef]:
    """Rehydrate reviewed eager mapping frames for rule reconciliation."""

    frames: list[FrameRef] = []
    seen: set[str] = set()
    for key in (
        "primitive_frame",
        "operator_frame",
        "semantic_frame",
        "model_context_frame",
        "phase_frame",
    ):
        value = row.get(key)
        if not isinstance(value, dict) or not value.get("raw"):
            continue
        raw = str(value["raw"])
        if raw in seen:
            continue
        seen.add(raw)
        frames.append(
            FrameRef(
                raw=raw,
                file=value.get("file"),
                line=value.get("line"),
                function=value.get("function"),
                module=value.get("module"),
                source_exists=value.get("source_exists"),
            )
        )
    return frames


def enrich_eager_semantics(
    source_rows: list[dict[str, Any]], *, framework: str
) -> int:
    """Apply current reviewed rules to gaps in the frozen eager mapping.

    The original mapping artifact remains the evidence authority. This pass
    only fills rows that were explicitly unmapped by an older rule revision,
    and every new binding still carries that row's eager event id and stack.
    """

    classifier = classify_glm52_node if framework == "sglang" else classify_trtllm_node
    enriched = 0
    for row in source_rows:
        if row.get("selected_node"):
            continue
        node, confidence = classifier(
            str(row.get("kernel_name") or ""),
            row.get("cpu_op_name"),
            _source_stack(row),
        )
        if not node:
            continue
        row["selected_node"] = node
        row["source_confidence"] = confidence
        row["eager_enrichment_method"] = "current_rule_over_frozen_eager_stack"
        enriched += 1
    return enriched


def _set_eager_schedule_node(
    row: dict[str, Any], node: str, *, method: str
) -> None:
    """Record a reviewed, hard-anchor-bounded eager semantic correction."""

    row["selected_node"] = node
    row["source_confidence"] = "high"
    row["eager_enrichment_method"] = method


def reconcile_trtllm_eager_decode_schedules(
    source_rows: list[dict[str, Any]],
) -> int:
    """Repair old TRT eager bindings with the reviewed 78-layer schedule.

    The historical mapper predated several TRT-LLM fused kernels and assigned
    generic linear launches to the first plausible node.  The eager trace
    still contains authoritative stacks and 78 hard MLA anchors.  Within each
    anchor interval, semantic landmarks distinguish attention, dense MLP, and
    routed/shared MoE roles without using timing proximity or a global kernel
    name guess.
    """

    segments = anchor_segments(source_rows, "trtllm", "decode")
    changed = 0
    method = "reviewed_trtllm_eager_layer_schedule"

    def lowered(row: dict[str, Any]) -> str:
        return str(row.get("kernel_name") or "").lower()

    def set_node(row: dict[str, Any], node: str) -> None:
        nonlocal changed
        if row.get("selected_node") != node:
            changed += 1
        _set_eager_schedule_node(row, node, method=method)

    for layer, (start, stop) in enumerate(segments):
        segment = source_rows[start:stop]
        if not segment:
            continue

        # Current layer: anchor, sparse attention, latent reconstruction and
        # row-parallel output projection.
        set_node(segment[0], "dsa_attention.q_split_rope")
        sparse_index = next(
            (
                index
                for index, row in enumerate(segment)
                if "fmhasm103akernel_qkve4m3" in lowered(row)
            ),
            None,
        )
        if sparse_index is not None:
            set_node(segment[sparse_index], "dsa_attention.sparse_mla_core")
            for row in segment[sparse_index + 1 :]:
                if row.get("cpu_op_name") == "aten::bmm":
                    set_node(row, "dsa_attention.latent_kv_reconstruction")
                    break
            for row in segment[sparse_index + 1 :]:
                semantic = str((row.get("semantic_frame") or {}).get("raw") or "")
                if row.get("cpu_op_name") == "aten::mm" and "modules/mla.py" in semantic:
                    set_node(row, "dsa_attention.output_projection")
                    break

        post_norm = next(
            (
                row
                for row in segment
                if "fused_add_rmsnorm" in lowered(row)
                and "forward_mlp"
                in str((row.get("semantic_frame") or {}).get("raw") or "")
            ),
            None,
        )
        if post_norm is not None:
            set_node(post_norm, "stack.post_attention_norm")

        if layer < 3:
            gated_rows = [
                row
                for row in segment
                if "modules/gated_mlp.py"
                in str((row.get("semantic_frame") or {}).get("raw") or "")
            ]
            silu_index = next(
                (
                    index
                    for index, row in enumerate(gated_rows)
                    if row.get("cpu_op_name") == "trtllm::silu_and_mul"
                ),
                None,
            )
            if silu_index is not None:
                for row in gated_rows[:silu_index]:
                    if row.get("cpu_op_name") == "aten::mm":
                        set_node(row, "dense_mlp.gate_up_projection")
                set_node(gated_rows[silu_index], "dense_mlp.swiglu")
                for row in gated_rows[silu_index + 1 :]:
                    if row.get("cpu_op_name") == "aten::mm":
                        set_node(row, "dense_mlp.down_projection")
        else:
            for row in segment:
                cpu = str(row.get("cpu_op_name") or "")
                semantic = str((row.get("semantic_frame") or {}).get("raw") or "")
                if cpu == "trtllm::dsv3_router_gemm_op":
                    set_node(row, "moe.router")
                elif cpu == "trtllm::fp4_quantize":
                    set_node(row, "moe.dispatch")
                elif "routingindices" in lowered(row):
                    set_node(row, "moe.topk")
                elif _has_moe_bmm(lowered(row)):
                    set_node(row, "moe.routed_experts")
                elif "finalizekernel" in lowered(row):
                    set_node(row, "moe.routed_weighted_combine")
                elif cpu == "aten::add" and "modeling_deepseekv3.py" in semantic:
                    set_node(row, "moe.combine")

            gated_rows = [
                row
                for row in segment
                if "modules/gated_mlp.py"
                in str((row.get("semantic_frame") or {}).get("raw") or "")
            ]
            silu_index = next(
                (
                    index
                    for index, row in enumerate(gated_rows)
                    if row.get("cpu_op_name") == "trtllm::silu_and_mul"
                ),
                None,
            )
            if silu_index is not None:
                for row in gated_rows[:silu_index]:
                    if row.get("cpu_op_name") == "aten::mm":
                        set_node(row, "moe.shared_expert_up")
                set_node(gated_rows[silu_index], "moe.shared_expert_activation")
                for row in gated_rows[silu_index + 1 :]:
                    if row.get("cpu_op_name") == "aten::mm":
                        set_node(row, "moe.shared_expert_down")

        # Each anchor-led eager interval ends with the next layer's projection
        # preamble.  Correct the older generic-linear bindings using the eager
        # semantic frame itself; this also supplies positive examples for the
        # graph-on tile variants below.
        fused_cat_rows: list[dict[str, Any]] = []
        for row in segment:
            cpu = str(row.get("cpu_op_name") or "")
            semantic = str((row.get("semantic_frame") or {}).get("raw") or "")
            value = lowered(row)
            if cpu == "trtllm::dsv3_fused_a_gemm_op":
                set_node(row, "dsa_attention.q_a_projection")
            elif "_q_a_layernorm_maybe_fused" in semantic:
                set_node(row, "dsa_attention.q_a_norm")
            elif "modules/mla.py(1575): <lambda>" in semantic:
                set_node(row, "dsa_attention.kv_a_norm")
            elif cpu == "aten::mm" and "forward_dsa_proj" in semantic:
                set_node(row, "dsa_attention.q_b_projection")
            elif "pre_indexer_proj" in semantic and cpu in {"aten::mm", "aten::copy_"}:
                set_node(row, "dsa_attention.index_q_projection")
            elif cpu == "trtllm::fused_cat_fp8" and "_prep_q_or_k" in semantic:
                fused_cat_rows.append(row)
            elif "_scale" in semantic:
                set_node(row, "dsa_attention.index_logits")
            elif cpu == "aten::fill_" and "sparse_attn_indexer" in semantic:
                set_node(row, "dsa_attention.index_topk")
        if fused_cat_rows:
            set_node(fused_cat_rows[0], "dsa_attention.index_q_projection")
        if len(fused_cat_rows) > 1:
            set_node(fused_cat_rows[1], "dsa_attention.index_k_norm_rope")

    return changed


def _has_moe_bmm(kernel_name: str) -> bool:
    return "bmm_e2m1" in kernel_name or "bmm_bfloat16_e2m1" in kernel_name


def _alignment_score(source_name: str, production_name: str) -> int | None:
    """Return a conservative kernel-identity score for bounded alignment."""

    if kernel_exact_identity(source_name) == kernel_exact_identity(production_name):
        return 400
    if kernel_base(source_name) == kernel_base(production_name):
        return 300
    if sequence_family(source_name) == sequence_family(production_name):
        return 200
    if schedule_family(source_name) == schedule_family(production_name):
        return 150
    return None


def align_layer_segment(
    source_segment: list[dict[str, Any]],
    production_segment: list[dict[str, Any]],
) -> list[tuple[int, int, int]]:
    """Monotonically align one eager/production layer without crossing gaps.

    CUDA Graph replay can insert runtime or collective helpers absent from the
    graph-off eager trace. A weighted LCS keeps exact and normalized identities
    in order while leaving insertions unmatched. It never uses timing proximity,
    neighboring node names, or cross-layer evidence.
    """

    source_count = len(source_segment)
    production_count = len(production_segment)
    scores = [[0] * (production_count + 1) for _ in range(source_count + 1)]
    choices = [[""] * (production_count + 1) for _ in range(source_count + 1)]
    for source_index in range(1, source_count + 1):
        choices[source_index][0] = "source_gap"
    for production_index in range(1, production_count + 1):
        choices[0][production_index] = "production_gap"

    for source_index in range(1, source_count + 1):
        source_name = str(source_segment[source_index - 1]["kernel_name"])
        for production_index in range(1, production_count + 1):
            production_name = str(
                production_segment[production_index - 1]["kernel_name"]
            )
            candidates = [
                (scores[source_index - 1][production_index], "source_gap"),
                (scores[source_index][production_index - 1], "production_gap"),
            ]
            match_score = _alignment_score(source_name, production_name)
            if match_score is not None:
                candidates.append(
                    (
                        scores[source_index - 1][production_index - 1]
                        + match_score,
                        "match",
                    )
                )
            priority = {"match": 2, "production_gap": 1, "source_gap": 0}
            value, choice = max(
                candidates, key=lambda item: (item[0], priority[item[1]])
            )
            scores[source_index][production_index] = value
            choices[source_index][production_index] = choice

    aligned: list[tuple[int, int, int]] = []
    source_index = source_count
    production_index = production_count
    while source_index or production_index:
        choice = choices[source_index][production_index]
        if choice == "match":
            match_score = _alignment_score(
                str(source_segment[source_index - 1]["kernel_name"]),
                str(production_segment[production_index - 1]["kernel_name"]),
            )
            assert match_score is not None
            aligned.append((source_index - 1, production_index - 1, match_score))
            source_index -= 1
            production_index -= 1
        elif choice == "production_gap":
            production_index -= 1
        else:
            source_index -= 1
    return list(reversed(aligned))


def _classify_runtime_support(rows: list[dict[str, Any]]) -> None:
    """Type intentionally non-architectural work after semantic attribution."""

    for row in rows:
        if row.get("node"):
            continue
        name = str(row.get("kernel_name") or "").lower()
        if any(token in name for token in ("sampling", "gumbel", "argmax")):
            support_class = "sampling_and_output"
            reason = "sampling, token selection, or output materialization"
        elif any(
            token in name
            for token in (
                "topk_plan",
                "dsa_decode_metadata",
                "block_table",
                "slot_mapping",
                "convertreqindextoglobal",
            )
        ):
            support_class = "attention_plan_metadata"
            reason = "attention planning metadata; no model tensor value is produced"
        elif any(
            token in name
            for token in ("alloc", "request_pool", "req_to_token", "cache_indices")
        ):
            support_class = "allocator_or_cache_management"
            reason = "request/KV allocation or cache-address management"
        elif any(
            token in name
            for token in (
                "foreach_copy",
                "index_elementwise",
                "indexelementwise",
                "direct_copy_kernel",
                "fillfunctor",
                "clamp_position",
                "memcpy",
                "memset",
                "scan",
                "arange",
                "divfloor",
                "compare",
                "where",
            )
        ):
            support_class = "request_batch_metadata"
            reason = "shape/index/request-batch metadata outside a semantic module"
        else:
            support_class = "graph_runtime_metadata"
            reason = "captured framework/runtime helper outside the stable Model IR"
        row.update(
            {
                "support_class": support_class,
                "support_reason": reason,
                "attribution_method": "explicit_runtime_support_classification",
                "confidence": "support",
            }
        )


def _assign_node_example(
    row: dict[str, Any],
    eager_node_examples: dict[str, dict[str, Any]],
    node: str,
    *,
    method: str,
) -> None:
    """Assign one production landmark using positive eager evidence."""

    source = eager_node_examples.get(node)
    if source is None:
        raise ValueError(f"production schedule node has no eager evidence: {node}")
    _assign(
        row,
        source,
        method=method,
        confidence="high",
        overwrite=True,
    )


def assign_sglang_decode_layer_schedules(
    kernels: list[dict[str, Any]],
    production_segments: list[tuple[int, int]],
    eager_node_examples: dict[str, dict[str, Any]],
) -> None:
    """Close graph-on GLM-5.2 schedules inside hard layer anchors.

    The eager trace proves the semantic nodes and their source stacks.  CUDA
    Graph replay changes GEMM tile shapes with batch size and can reverse the
    two independent DSA-indexer projections.  This pass therefore recognizes
    semantic order between the sparse-attention anchor, the two TP collective
    boundaries, normalization/indexer landmarks, and stream-local split-K
    pairs.  It never assigns a role from a global kernel name or wall-clock
    proximity.
    """

    method = "sglang_eager_validated_layer_schedule_landmark"

    def name(row: dict[str, Any]) -> str:
        return str(row.get("kernel_name") or "").lower()

    def find_index(
        segment: list[dict[str, Any]],
        predicate,
        *,
        start: int = 0,
        stop: int | None = None,
    ) -> int | None:
        effective_stop = len(segment) if stop is None else stop
        for index in range(start, effective_stop):
            if predicate(name(segment[index])):
                return index
        return None

    for layer, (start, stop) in enumerate(production_segments):
        segment = kernels[start:stop]
        if not segment:
            continue

        collective_indices = [
            index
            for index, row in enumerate(segment)
            if "twoshotallreducekernel" in name(row)
            or "oneshotallreducefusionkernel" in name(row)
            or "nccldevkernel_allreduce" in name(row)
        ]
        if len(collective_indices) != 2:
            raise ValueError(
                f"SGLang layer {layer}: expected two TP collective landmarks, "
                f"got {len(collective_indices)}"
            )
        attention_collective, ffn_collective = collective_indices

        # Current-layer attention: the sparse core produces a compressed
        # result, which is reconstructed and then projected before TP output
        # reduction.  Tile dimensions vary across BS1/16/64/256; order does
        # not.
        attention_launches = [
            row
            for row in segment[1:attention_collective]
            if "nvjet_sm103_tst_" in name(row) and "splitk" not in name(row)
        ]
        if len(attention_launches) >= 2:
            _assign_node_example(
                attention_launches[0],
                eager_node_examples,
                "dsa_attention.latent_kv_reconstruction",
                method=method,
            )
            _assign_node_example(
                attention_launches[1],
                eager_node_examples,
                "dsa_attention.output_projection",
                method=method,
            )

        ffn_rows = segment[attention_collective + 1 : ffn_collective]
        if layer < 3:
            activation_index = next(
                (
                    index
                    for index, row in enumerate(ffn_rows)
                    if "act_and_mul_kernel" in name(row)
                    or "silu_and_mul_kernel" in name(row)
                ),
                None,
            )
            if activation_index is not None:
                before = [
                    row
                    for row in ffn_rows[:activation_index]
                    if "tgv" in name(row) or "nvjet_sm103_" in name(row)
                ]
                after = [
                    row
                    for row in ffn_rows[activation_index + 1 :]
                    if "tgv" in name(row) or "nvjet_sm103_" in name(row)
                ]
                if before:
                    _assign_node_example(
                        before[-1],
                        eager_node_examples,
                        "dense_mlp.gate_up_projection",
                        method=method,
                    )
                if after:
                    _assign_node_example(
                        after[0],
                        eager_node_examples,
                        "dense_mlp.down_projection",
                        method=method,
                    )
        else:
            launch_by_stream: dict[int, str] = {}
            activation_streams: set[int] = set()
            for row in ffn_rows:
                value = name(row)
                stream = int(row.get("stream_id") or row.get("stream") or -1)
                if "nvjet_sm103_tss_" in value and "splitk" in value:
                    node = "moe.router"
                    launch_by_stream[stream] = node
                    _assign_node_example(row, eager_node_examples, node, method=method)
                elif "nvjet_sm103_tst_" in value and "splitk" in value:
                    node = "moe.shared_expert_up"
                    launch_by_stream[stream] = node
                    _assign_node_example(row, eager_node_examples, node, method=method)
                elif "splitkreduce_kernel" in value and stream in launch_by_stream:
                    _assign_node_example(
                        row,
                        eager_node_examples,
                        launch_by_stream[stream],
                        method=method,
                    )
                elif "routingindicesblockscoreskernel" in value:
                    _assign_node_example(row, eager_node_examples, "moe.topk", method=method)
                elif "routingindicesclusterkernel" in value:
                    _assign_node_example(row, eager_node_examples, "moe.topk", method=method)
                elif "act_and_mul_kernel" in value or "silu_and_mul_kernel" in value:
                    activation_streams.add(stream)
                elif (
                    stream in activation_streams
                    and "nvjet_sm103_tst_" in value
                    and "splitk" not in value
                ):
                    _assign_node_example(
                        row,
                        eager_node_examples,
                        "moe.shared_expert_down",
                        method=method,
                    )

        q_indexer = find_index(
            segment, lambda value: "fused_q_indexer_rope_hadamard_quant" in value
        )

        for row in segment:
            value = name(row)
            if "nvfp4quantizelinearkernel" in value:
                _assign_node_example(
                    row, eager_node_examples, "moe.dispatch", method=method
                )
            elif "routingindicesdynblockkernel" in value:
                _assign_node_example(row, eager_node_examples, "moe.topk", method=method)

        # The post-FFN interval prepares the next layer.  Its projections are
        # identified relative to the collective, norm, cat/indexer, and RoPE
        # landmarks rather than a tile shape.
        tail = segment[ffn_collective + 1 :]
        q_a_norm_tail = next(
            (
                index
                for index, row in enumerate(tail)
                if "rmsnormkernel" in name(row) and "oi642048" in name(row)
            ),
            None,
        )
        kv_a_norm_tail = next(
            (
                index
                for index, row in enumerate(tail)
                if "rmsnormkernel" in name(row) and "oi64512" in name(row)
            ),
            None,
        )
        q_a_candidates = [
            row
            for row in tail[: q_a_norm_tail if q_a_norm_tail is not None else 0]
            if "tgv" in name(row)
            or ("nvjet_sm103_" in name(row) and "splitk" not in name(row))
        ]
        if q_a_candidates:
            _assign_node_example(
                q_a_candidates[-1],
                eager_node_examples,
                "dsa_attention.q_a_projection",
                method=method,
            )
        if q_a_norm_tail is not None:
            _assign_node_example(
                tail[q_a_norm_tail],
                eager_node_examples,
                "dsa_attention.q_a_norm",
                method=method,
            )
        if kv_a_norm_tail is not None:
            _assign_node_example(
                tail[kv_a_norm_tail],
                eager_node_examples,
                "dsa_attention.kv_a_norm",
                method=method,
            )
        cat_tail = next(
            (index for index, row in enumerate(tail) if "cat" in name(row)),
            None,
        )
        q_b_start = (
            cat_tail + 1
            if cat_tail is not None
            else ((kv_a_norm_tail + 1) if kv_a_norm_tail is not None else 0)
        )
        q_indexer_tail = (
            q_indexer - (ffn_collective + 1) if q_indexer is not None else None
        )
        rope_tail = next(
            (
                index
                for index, row in enumerate(tail)
                if "ropequantizekernel" in name(row)
            ),
            None,
        )
        q_b_stop = q_indexer_tail if q_indexer_tail is not None else rope_tail
        if q_b_stop is None:
            q_b_stop = len(tail)
        projection_candidates = [
            row
            for row in tail[q_b_start:q_b_stop]
            if "tgv" in name(row)
            or ("nvjet_sm103_tst_" in name(row) and "splitk" not in name(row))
        ]
        if projection_candidates:
            _assign_node_example(
                projection_candidates[0],
                eager_node_examples,
                "dsa_attention.q_b_projection",
                method=method,
            )
        if q_indexer is not None:
            if len(projection_candidates) > 1:
                _assign_node_example(
                    projection_candidates[-1],
                    eager_node_examples,
                    "dsa_attention.index_q_projection",
                    method=method,
                )
            norm_or_rope = next(
                (
                    index
                    for index, row in enumerate(tail[q_indexer_tail + 1 :], q_indexer_tail + 1)
                    if "fused_k_indexer_norm_rope_store" in name(row)
                ),
                None,
            )
            if norm_or_rope is not None:
                index_k_candidates = [
                    row
                    for row in tail[:norm_or_rope]
                    if "nvjet_sm103_tst_" in name(row) and "splitk" in name(row)
                ]
                if index_k_candidates:
                    _assign_node_example(
                        index_k_candidates[-1],
                        eager_node_examples,
                        "dsa_attention.index_k_gate_projection",
                        method=method,
                    )
            for row in tail[:norm_or_rope]:
                if "splitkreduce_kernel" in name(row):
                    _assign_node_example(
                        row,
                        eager_node_examples,
                        "dsa_attention.index_k_gate_projection",
                        method=method,
                    )
        elif len(projection_candidates) > 1:
            # Non-owner DSA layers reuse the prior top-k state.  The second
            # projection between Q_b and RoPE is therefore the latent-KV
            # reconstruction, not an indexer projection.
            _assign_node_example(
                projection_candidates[-1],
                eager_node_examples,
                "dsa_attention.latent_kv_reconstruction",
                method=method,
            )
        topk_tail = [
            index
            for index, row in enumerate(tail)
            if "topk_main_kernel" in name(row)
        ]
        if topk_tail:
            latent = next(
                (
                    row
                    for row in tail[topk_tail[-1] + 1 :]
                    if "nvjet_sm103_tst_" in name(row) and "splitk" not in name(row)
                ),
                None,
            )
            if latent is not None:
                _assign_node_example(
                    latent,
                    eager_node_examples,
                    "dsa_attention.latent_kv_reconstruction",
                    method=method,
                )

    # Layer 0's projection preamble precedes the first sparse-core anchor and
    # is therefore not part of an anchor-led interval.  It is still bounded
    # by the embedding/input norm on the left and layer 0's sparse core on the
    # right, so apply the same exact landmarks only inside that prefix.
    first_anchor = production_segments[0][0]
    prefix = kernels[:first_anchor]
    q_a_norm = find_index(
        prefix, lambda value: "rmsnormkernel" in value and "oi642048" in value
    )
    kv_a_norm = find_index(
        prefix, lambda value: "rmsnormkernel" in value and "oi64512" in value
    )
    if q_a_norm is not None:
        q_a_candidates = [
            index
            for index, row in enumerate(prefix[:q_a_norm])
            if "tgv" in name(row)
            or "fused_a_gemm_kernel" in name(row)
            or ("nvjet_sm103_tst_" in name(row) and "splitk" not in name(row))
        ]
        q_a = q_a_candidates[-1] if q_a_candidates else None
        input_norm = find_index(
            prefix,
            lambda value: "rmsnormkernel" in value and "oi646144" in value,
            stop=q_a if q_a is not None else q_a_norm,
        )
        if input_norm is not None:
            _assign_node_example(
                prefix[input_norm], eager_node_examples, "stack.input_norm", method=method
            )
        prefix_nodes = (
            (q_a, "dsa_attention.q_a_projection"),
            (q_a_norm, "dsa_attention.q_a_norm"),
            (kv_a_norm, "dsa_attention.kv_a_norm"),
        )
        for index, node in prefix_nodes:
            if index is not None:
                _assign_node_example(prefix[index], eager_node_examples, node, method=method)
        q_indexer = find_index(
            prefix, lambda value: "fused_q_indexer_rope_hadamard_quant" in value
        )
        q_b_start = (kv_a_norm + 1) if kv_a_norm is not None else q_a_norm + 1
        q_b_stop = q_indexer if q_indexer is not None else len(prefix)
        projections = [
            index
            for index, row in enumerate(prefix[q_b_start:q_b_stop], q_b_start)
            if "tgv" in name(row)
            or "fused_a_gemm_kernel" in name(row)
            or ("nvjet_sm103_tst_" in name(row) and "splitk" not in name(row))
        ]
        if projections:
            _assign_node_example(
                prefix[projections[0]],
                eager_node_examples,
                "dsa_attention.q_b_projection",
                method=method,
            )
        if q_indexer is not None and len(projections) > 1:
            _assign_node_example(
                prefix[projections[-1]],
                eager_node_examples,
                "dsa_attention.index_q_projection",
                method=method,
            )
        index_norm = find_index(
            prefix, lambda value: "fused_k_indexer_norm_rope_store" in value
        )
        if q_indexer is not None and index_norm is not None:
            for index in range(q_b_start, index_norm):
                value = name(prefix[index])
                if "splitk" in value or "splitkreduce_kernel" in value:
                    _assign_node_example(
                        prefix[index],
                        eager_node_examples,
                        "dsa_attention.index_k_gate_projection",
                        method=method,
                    )
        rope = find_index(prefix, lambda value: "ropequantizekernel" in value)
        if rope is not None:
            topk = find_index(prefix, lambda value: "topk_main_kernel" in value)
            latent = find_index(
                prefix,
                lambda value: "nvjet_sm103_tst_" in value
                and "splitk" not in value,
                start=(topk + 1) if topk is not None else q_a_norm + 1,
                stop=rope,
            )
            if latent is not None:
                _assign_node_example(
                    prefix[latent],
                    eager_node_examples,
                    "dsa_attention.latent_kv_reconstruction",
                    method=method,
                )

    # Graph decode can launch the router GEMM with PDL and then place its
    # split-K reduction on the same stream before the primary has retired.
    # Pair only exact same-stream overlapping events already proven as router.
    router_primaries = [row for row in kernels if row.get("node") == "moe.router"]
    for row in kernels:
        if row.get("node") is not None or "splitkreduce_kernel" not in name(row):
            continue
        start_us = float(row.get("ts_us") or 0.0)
        same_stream = [
            primary
            for primary in router_primaries
            if primary.get("stream_id") == row.get("stream_id")
            and float(primary.get("ts_us") or 0.0) <= start_us
            <= float(primary.get("ts_us") or 0.0) + float(primary.get("dur_us") or 0.0)
        ]
        if len(same_stream) == 1:
            _assign_node_example(
                row, eager_node_examples, "moe.router", method=method
            )

    # The logits GEMM is outside the last layer anchor interval but is bounded
    # by final norm and the logits all-gather in the validated decode window.
    logits_gather_indices = [
        index
        for index, row in enumerate(kernels)
        if row.get("node") == "top.tp_logits_all_gather"
    ]
    ffn_collective_indices = [
        index
        for index, row in enumerate(kernels)
        if row.get("node")
        in {
            "dense_mlp.tp_dense_mlp_output_collective",
            "moe.tp_moe_output_collective",
        }
    ]
    if ffn_collective_indices and logits_gather_indices:
        # The final MoE collective may fuse final normalization, so the last
        # layer boundary is the stronger portable left landmark.
        tail_start = ffn_collective_indices[-1] + 1
        tail_stop = logits_gather_indices[-1]
        candidates = [
            row
            for row in kernels[tail_start:tail_stop]
            if "nvjet_sm103_tst_" in name(row) and "splitk" not in name(row)
        ]
        if len(candidates) == 1:
            _assign_node_example(
                candidates[0], eager_node_examples, "top.lm_head", method=method
            )


def assign_sglang_prefill_layer_schedules(
    kernels: list[dict[str, Any]],
    production_segments: list[tuple[int, int]],
    eager_node_examples: dict[str, dict[str, Any]],
) -> None:
    """Close prefill tile variants using anchor and operator order."""

    method = "sglang_eager_validated_prefill_layer_schedule_landmark"

    def lowered(row: dict[str, Any]) -> str:
        return str(row.get("kernel_name") or "").lower()

    def assign(row: dict[str, Any], node: str) -> None:
        _assign_node_example(row, eager_node_examples, node, method=method)

    for layer, (start, stop) in enumerate(production_segments):
        segment = kernels[start:stop]
        collectives = [
            index
            for index, row in enumerate(segment)
            if inferred_collective_kind(str(row.get("kernel_name") or ""))
            == "all_reduce"
        ]
        if len(collectives) < 2:
            continue
        attention_collective, ffn_collective = collectives[:2]
        attention_launches = [
            row
            for row in segment[1:attention_collective]
            if "nvjet_sm103_tst_" in lowered(row)
            and "splitk" not in lowered(row)
        ]
        if len(attention_launches) >= 2:
            assign(attention_launches[0], "dsa_attention.latent_kv_reconstruction")
            assign(attention_launches[1], "dsa_attention.output_projection")

        if layer >= 3:
            ffn_rows = segment[attention_collective + 1 : ffn_collective]
            activation_index = next(
                (
                    index
                    for index, row in enumerate(ffn_rows)
                    if "act_and_mul_kernel" in lowered(row)
                    or "silu_and_mul_kernel" in lowered(row)
                ),
                None,
            )
            if activation_index is not None:
                before = [
                    row
                    for row in ffn_rows[:activation_index]
                    if "nvjet_sm103_tst_" in lowered(row)
                    and "splitk" not in lowered(row)
                    and not row.get("node")
                ]
                after = [
                    row
                    for row in ffn_rows[activation_index + 1 :]
                    if "nvjet_sm103_tst_" in lowered(row)
                    and "splitk" not in lowered(row)
                ]
                if before:
                    assign(before[-1], "moe.shared_expert_up")
                if after:
                    assign(after[0], "moe.shared_expert_down")

    # The first layer's projection/indexer preamble lies before the first
    # sparse-attention anchor.  Its local landmark order is the same as later
    # layers and is fully represented in the eager stack evidence.
    prefix = kernels[: production_segments[0][0]]
    q_a_norm = next(
        (
            index
            for index, row in enumerate(prefix)
            if "rmsnormkernel" in lowered(row) and "oi642048" in lowered(row)
        ),
        None,
    )
    kv_a_norm = next(
        (
            index
            for index, row in enumerate(prefix)
            if "rmsnormkernel" in lowered(row) and "oi64512" in lowered(row)
        ),
        None,
    )
    if q_a_norm is not None:
        q_a_launches = [
            row
            for row in prefix[:q_a_norm]
            if "nvjet_sm103_tst_" in lowered(row)
            and "splitk" not in lowered(row)
        ]
        if q_a_launches:
            assign(q_a_launches[-1], "dsa_attention.q_a_projection")

    index_norm = next(
        (
            index
            for index, row in enumerate(prefix)
            if "fused_k_indexer_norm_rope_store" in lowered(row)
        ),
        None,
    )
    if kv_a_norm is not None and index_norm is not None:
        projected = [
            row
            for row in prefix[kv_a_norm + 1 : index_norm]
            if "nvjet_sm103_tst_" in lowered(row)
            and "splitk" not in lowered(row)
        ]
        if projected:
            assign(projected[0], "dsa_attention.q_b_projection")
        for row in projected[1:]:
            if not row.get("node"):
                assign(row, "dsa_attention.index_q_projection")

    topk = [
        index
        for index, row in enumerate(prefix)
        if "topk_transform_prefill_kernel" in lowered(row)
    ]
    if topk:
        latent = next(
            (
                row
                for row in prefix[topk[-1] + 1 :]
                if "nvjet_sm103_tst_" in lowered(row)
                and "splitk" not in lowered(row)
            ),
            None,
        )
        if latent is not None:
            assign(latent, "dsa_attention.latent_kv_reconstruction")


def assign_trtllm_decode_layer_schedules(
    kernels: list[dict[str, Any]],
    production_segments: list[tuple[int, int]],
    eager_node_examples: dict[str, dict[str, Any]],
) -> None:
    """Bind TRT graph-on tile variants inside hard layer/collective bounds."""

    method = "trtllm_eager_validated_layer_schedule_landmark"

    def lowered(row: dict[str, Any]) -> str:
        return str(row.get("kernel_name") or "").lower()

    def assign(row: dict[str, Any], node: str) -> None:
        _assign_node_example(row, eager_node_examples, node, method=method)

    for layer, (start, stop) in enumerate(production_segments):
        segment = kernels[start:stop]
        if not segment:
            continue
        assign(segment[0], "dsa_attention.q_split_rope")
        primaries = [
            index
            for index, row in enumerate(segment)
            if "twoshotallreducekernel" in lowered(row)
        ]
        if len(primaries) != 2:
            raise ValueError(
                f"TRT-LLM layer {layer}: expected two collective landmarks, "
                f"got {len(primaries)}"
            )
        attention_collective, ffn_collective = primaries

        sparse_index = next(
            (
                index
                for index, row in enumerate(segment[:attention_collective])
                if "fmhasm103akernel_qkve4m3" in lowered(row)
            ),
            None,
        )
        if sparse_index is not None:
            assign(segment[sparse_index], "dsa_attention.sparse_mla_core")
            launches = [
                segment[index]
                for index in range(sparse_index + 1, attention_collective)
                if "nvjet_sm103_tst_" in lowered(segment[index])
            ]
            if len(launches) >= 2:
                assign(launches[0], "dsa_attention.latent_kv_reconstruction")
                assign(launches[1], "dsa_attention.output_projection")

        ffn_rows = segment[attention_collective + 1 : ffn_collective]
        if layer < 3:
            activation_index = next(
                (
                    index
                    for index, row in enumerate(ffn_rows)
                    if "silu_and_mul_kernel" in lowered(row)
                ),
                None,
            )
            for row in ffn_rows:
                value = lowered(row)
                if "fused_add_rmsnorm" in value:
                    assign(row, "stack.post_attention_norm")
                elif "silu_and_mul_kernel" in value:
                    assign(row, "dense_mlp.swiglu")
            if activation_index is not None:
                before = [
                    row
                    for row in ffn_rows[:activation_index]
                    if "nvjet_sm103_" in lowered(row)
                    and "splitk" not in lowered(row)
                ]
                after = [
                    row
                    for row in ffn_rows[activation_index + 1 :]
                    if "nvjet_sm103_" in lowered(row)
                    and "splitk" not in lowered(row)
                ]
                if before:
                    assign(before[-1], "dense_mlp.gate_up_projection")
                if after:
                    assign(after[0], "dense_mlp.down_projection")
        else:
            launch_by_stream: dict[int, str] = {}
            activation_streams: set[int] = set()
            for row in ffn_rows:
                value = lowered(row)
                stream = int(row.get("stream_id") or row.get("stream") or -1)
                if "nvjet_sm103_tss_" in value and "splitk" in value:
                    assign(row, "moe.router")
                    launch_by_stream[stream] = "moe.router"
                elif "nvjet_sm103_tst_" in value and "splitk" in value:
                    assign(row, "moe.shared_expert_up")
                    launch_by_stream[stream] = "moe.shared_expert_up"
                elif "splitkreduce_kernel" in value and stream in launch_by_stream:
                    assign(row, launch_by_stream[stream])
                elif "quantize_with_block_size" in value:
                    assign(row, "moe.dispatch")
                elif "silu_and_mul_kernel" in value:
                    assign(row, "moe.shared_expert_activation")
                    activation_streams.add(stream)
                elif "routingindicesclusterkernel" in value or "routingindicesblockscoreskernel" in value:
                    assign(row, "moe.topk")
                elif (
                    stream in activation_streams
                    and "nvjet_sm103_tst_" in value
                    and "splitk" not in value
                ):
                    assign(row, "moe.shared_expert_down")
                elif _has_moe_bmm(value):
                    assign(row, "moe.routed_experts")
                elif "finalizekernel" in value:
                    assign(row, "moe.routed_weighted_combine")
                elif "vectorized_elementwise_kernel" in value:
                    assign(row, "moe.combine")

        if layer == LAYER_COUNT - 1:
            continue
        tail = segment[ffn_collective + 1 :]
        q_a_norm_index = next(
            (
                index
                for index, row in enumerate(tail)
                if "rmsnormkernel" in lowered(row) and "oi642048" in lowered(row)
            ),
            None,
        )
        q_a_candidates = [
            row
            for row in tail[: q_a_norm_index if q_a_norm_index is not None else 0]
            if "nvjet_sm103_" in lowered(row) and "splitk" not in lowered(row)
        ]
        q_a_row = q_a_candidates[-1] if q_a_candidates else None
        if q_a_row is not None:
            assign(q_a_row, "dsa_attention.q_a_projection")
        for row in tail:
            value = lowered(row)
            if "rmsnormkernel" in value and "oi642048" in value:
                assign(row, "dsa_attention.q_a_norm")
            elif "rmsnormkernel" in value and "oi64512" in value:
                assign(row, "dsa_attention.kv_a_norm")

        cat_index = next(
            (index for index, row in enumerate(tail) if "catarraybatchedcopy" in lowered(row)),
            None,
        )
        q_b_row = None
        if cat_index is not None:
            assign(tail[cat_index], "dsa_attention.q_b_projection")
            q_b_row = next(
                (
                    tail[index]
                    for index in range(cat_index + 1, len(tail))
                    if "nvjet_sm103_tst_" in lowered(tail[index])
                    and "splitk" not in lowered(tail[index])
                ),
                None,
            )
            if q_b_row is not None:
                assign(q_b_row, "dsa_attention.q_b_projection")

        indexer_nodes = (
            ("triton_poi_fused__to_copy", "dsa_attention.index_q_projection"),
            ("kernel2", "dsa_attention.index_q_projection"),
            ("splitkreduce_kernel", "dsa_attention.index_q_projection"),
            ("elementwise_kernel", "dsa_attention.index_q_projection"),
            ("triton_per_fused__to_copy_native_layer_norm", "dsa_attention.index_k_norm_rope"),
            ("batchqkapplyrotary", "dsa_attention.index_k_norm_rope"),
            ("triton_poi_fused_mul_squeeze", "dsa_attention.index_logits"),
            ("indexerkcachescatter", "dsa_attention.index_k_cache"),
            ("sm100_fp8_paged_mqa_logits", "dsa_attention.index_logits"),
            ("topkperrowdecode", "dsa_attention.index_topk"),
        )
        for row in tail:
            value = lowered(row)
            for token, node in indexer_nodes:
                if token in value:
                    assign(row, node)
                    break
        fused_cats = [row for row in tail if "fusedcatfp8kernel" in lowered(row)]
        if fused_cats:
            assign(fused_cats[0], "dsa_attention.index_q_projection")
        if len(fused_cats) > 1:
            assign(fused_cats[1], "dsa_attention.index_k_norm_rope")

        tail_nvjets = [
            row
            for row in tail
            if "nvjet_sm103_tst_" in lowered(row)
            and "splitk" not in lowered(row)
            and row is not q_a_row
            and row is not q_b_row
        ]
        topk_indices = [
            index for index, row in enumerate(tail) if "topkperrowdecode" in lowered(row)
        ]
        if topk_indices:
            topk_stop = topk_indices[-1]
            index_norm_indices = [
                index
                for index, row in enumerate(tail)
                if "triton_per_fused__to_copy_native_layer_norm" in lowered(row)
            ]
            before_topk = [row for row in tail_nvjets if tail.index(row) < topk_stop]
            after_topk = [row for row in tail_nvjets if tail.index(row) > topk_stop]
            if index_norm_indices:
                before_topk = [
                    row for row in before_topk if tail.index(row) < index_norm_indices[0]
                ]
            if before_topk:
                assign(before_topk[-1], "dsa_attention.index_k_gate_projection")
            if after_topk:
                assign(after_topk[-1], "dsa_attention.latent_kv_reconstruction")
        elif tail_nvjets:
            assign(tail_nvjets[-1], "dsa_attention.latent_kv_reconstruction")

    # Layer 0's projection preamble precedes the first eager/production anchor.
    # All semantic launches except the standalone input norm have identical
    # eager-proven roles in later layer preambles.  The norm is an explicit
    # production-window boundary exception: the frozen eager extraction starts
    # at q_split_rope and therefore cannot contain this earlier launch.
    first_anchor = production_segments[0][0]
    prefix = kernels[:first_anchor]
    prefix_q_a_norm = next(
        (
            index
            for index, row in enumerate(prefix)
            if "rmsnormkernel" in lowered(row) and "oi642048" in lowered(row)
        ),
        None,
    )
    prefix_q_a = [
        row
        for row in prefix[: prefix_q_a_norm if prefix_q_a_norm is not None else 0]
        if "nvjet_sm103_" in lowered(row) and "splitk" not in lowered(row)
    ]
    if prefix_q_a:
        assign(prefix_q_a[-1], "dsa_attention.q_a_projection")
    for row in prefix:
        value = lowered(row)
        if "rmsnormkernel" in value and "oi646144" in value:
            row.update(
                {
                    "node": "stack.input_norm",
                    "kernel_label": "Layer-0 input RMSNorm",
                    "attribution_method": (
                        "trtllm_production_window_layer0_input_norm_contract"
                    ),
                    "confidence": "high",
                }
            )
        elif "rmsnormkernel" in value and "oi642048" in value:
            assign(row, "dsa_attention.q_a_norm")
        elif "rmsnormkernel" in value and "oi64512" in value:
            assign(row, "dsa_attention.kv_a_norm")
        elif "catarraybatchedcopy" in value:
            assign(row, "dsa_attention.q_b_projection")
        elif any(token in value for token in ("kernel2", "splitkreduce_kernel", "elementwise_kernel")):
            assign(row, "dsa_attention.index_q_projection")
    prefix_cat = next(
        (index for index, row in enumerate(prefix) if "catarraybatchedcopy" in lowered(row)),
        None,
    )
    prefix_q_b = None
    if prefix_cat is not None:
        prefix_q_b = next(
            (
                row
                for row in prefix[prefix_cat + 1 :]
                if "nvjet_sm103_tst_" in lowered(row)
                and "splitk" not in lowered(row)
            ),
            None,
        )
        if prefix_q_b is not None:
            assign(prefix_q_b, "dsa_attention.q_b_projection")
    prefix_index_norm = next(
        (
            index
            for index, row in enumerate(prefix)
            if "triton_per_fused__to_copy_native_layer_norm" in lowered(row)
        ),
        None,
    )
    if prefix_index_norm is not None:
        prefix_index_k = [
            row
            for row in prefix[:prefix_index_norm]
            if "nvjet_sm103_tst_" in lowered(row)
            and "splitk" not in lowered(row)
            and row is not prefix_q_b
            and row not in prefix_q_a
        ]
        if prefix_index_k:
            assign(prefix_index_k[-1], "dsa_attention.index_k_gate_projection")
    topk_indices = [
        index for index, row in enumerate(prefix) if "topkperrowdecode" in lowered(row)
    ]
    if topk_indices:
        latent = next(
            (
                prefix[index]
                for index in range(topk_indices[-1] + 1, len(prefix))
                if "nvjet_sm103_tst_" in lowered(prefix[index])
                and "splitk" not in lowered(prefix[index])
            ),
            None,
        )
        if latent is not None:
            assign(latent, "dsa_attention.latent_kv_reconstruction")

    # Final logits GEMM lies after the last layer's second TP collective and
    # before the logits all-gather.  This pass intentionally runs before the
    # generic collective reconciliation below, so use the two hard raw
    # collective landmarks in the final production segment instead of relying
    # on node labels that have not been installed yet.
    logits_gather_indices = [
        index
        for index, row in enumerate(kernels)
        if row.get("node") == "top.tp_logits_all_gather"
        or inferred_collective_kind(str(row.get("kernel_name") or "")) == "all_gather"
    ]
    if logits_gather_indices:
        final_start, final_stop = production_segments[-1]
        final_primaries = [
            index
            for index in range(final_start, final_stop)
            if "twoshotallreducekernel" in lowered(kernels[index])
        ]
        if len(final_primaries) != 2:
            raise ValueError(
                "TRT-LLM final layer: expected two raw collective landmarks, "
                f"got {len(final_primaries)}"
            )
        tail_start = final_primaries[-1] + 1
        candidates = [
            row
            for row in kernels[tail_start : logits_gather_indices[-1]]
            if "nvjet_sm103_tst_" in lowered(row)
            and "splitk" not in lowered(row)
        ]
        if len(candidates) == 1:
            assign(candidates[0], "top.lm_head")


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
    eager_enriched_count = enrich_eager_semantics(source_rows, framework=framework)
    eager_schedule_reconciled_count = 0
    if framework == "trtllm" and phase == "decode":
        eager_schedule_reconciled_count = reconcile_trtllm_eager_decode_schedules(
            source_rows
        )
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
            for source_index, production_index, score in align_layer_segment(
                source_segment, prod_segment
            ):
                source = source_segment[source_index]
                production = prod_segment[production_index]
                if score >= 400:
                    method = "hard_anchor_bounded_monotonic_exact_identity"
                    confidence = "high"
                elif score >= 300:
                    method = "hard_anchor_bounded_monotonic_function_identity"
                    confidence = "medium"
                elif score >= 200:
                    method = "hard_anchor_bounded_monotonic_normalized_identity"
                    confidence = "medium"
                else:
                    method = "hard_anchor_bounded_monotonic_codegen_schedule_family"
                    confidence = "medium"
                _assign(
                    production,
                    source,
                    method=method,
                    confidence=confidence,
                )
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

    if framework == "sglang" and phase == "decode":
        assign_sglang_decode_layer_schedules(
            kernels,
            production_segments,
            eager_node_examples,
        )
    elif framework == "sglang" and phase == "prefill":
        assign_sglang_prefill_layer_schedules(
            kernels,
            production_segments,
            eager_node_examples,
        )
    elif framework == "trtllm" and phase == "decode":
        assign_trtllm_decode_layer_schedules(
            kernels,
            production_segments,
            eager_node_examples,
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

    _classify_runtime_support(production_rows)

    total_kernel_us = sum(float(row["dur_us"]) for row in kernels)
    production_boundary_methods = {
        "trtllm_production_window_layer0_input_norm_contract"
    }
    missing_eager_provenance = [
        row
        for row in kernels
        if row.get("node") is not None
        and not row.get("eager_event_id")
        and row.get("attribution_method") not in production_boundary_methods
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
        "eager_enriched_kernel_count": eager_enriched_count,
        "eager_schedule_reconciled_kernel_count": eager_schedule_reconciled_count,
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
