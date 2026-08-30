#!/usr/bin/env python3
"""Build DeepSeek-V4-Pro-0813 vLLM production profiles and timelines.

The input is the fail-closed 8-rank production reconciliation.  Raw traces,
full eager mappings, and non-regenerable evidence remain outside git; the
canonical profile keeps their hashes and the exact critical-rank event set.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.common.timeline_artifact import (  # noqa: E402
    attach_eager_stack_evidence,
    build_timeline_artifact,
    sha256_file,
    write_timeline_artifact,
)


MODEL_REVISION = "72e1d3230f6c080a530b0a1d46f8eb4602340597"
SOURCE_COMMIT = "dd10e03f95f94edbea1975c67ace3a35ec9a8a40"
CONTAINER_SHA256 = "111d0967a4054e22dbc8fccddfe7446cebe7b5cede51e555b563a77a5574d28c"
EXECUTION_PATH = "tp8_moe_intermediate_shard"
IMPLEMENTATION_ID = "vllm_dd10e03_dsv4pro0813_tp8"
MATRIX_REPORT_SHA256 = "a98426df442efccd113042435a98b08a3b3341593c332585c868777832cb0fc3"
MATRIX_MANIFEST_SHA256 = "bbf22613f92f601826d529a5d16642e527874392d8f0644223d1ee2da6483789"
BASELINE_SELECTIONS = {
    "vllm": {
        "path": "evidence/vllm-baseline/3415253/baseline-selection.json",
        "sha256": "8dbf1bb862ae9471a60b6d0a0b002fc8e8ce4a034916cab989226c047c9f3176",
    },
    "sglang": {
        "path": "evidence/sglang-baseline/3417439/baseline-selection.json",
        "sha256": "182e5d3c1149975795dd6c4e9e45a69e1ad2be7340f26a10c18452cff6c8812c",
    },
}


PROFILE_SPECS = {
    "prefill-c1": {
        "phase": "prefill",
        "batch_size": 1,
        "job_id": "3420168",
        "eager_job_id": "3417280",
        "eager_kind": "vllm-eager-prefill",
        "production_kind": "vllm-prefill_timing",
        "variant_id": "eager_prefill_gbs001_8k",
        "file_stem": "eager_prefill_gbs001_8k",
    },
    "decode-c1": {
        "phase": "decode",
        "batch_size": 1,
        "job_id": "3420170",
        "eager_job_id": "3417281",
        "eager_kind": "vllm-eager-decode",
        "production_kind": "vllm-production",
        "variant_id": "cg_decode_gbs001_8k1k",
        "file_stem": "cg_decode_gbs001_8k1k",
    },
    "decode-c16": {
        "phase": "decode",
        "batch_size": 16,
        "job_id": "3420173",
        "eager_job_id": "3417282",
        "eager_kind": "vllm-eager-decode",
        "production_kind": "vllm-production",
        "variant_id": "cg_decode_gbs016_8k1k",
        "file_stem": "cg_decode_gbs016_8k1k",
    },
    "decode-c64": {
        "phase": "decode",
        "batch_size": 64,
        "job_id": "3420174",
        "eager_job_id": "3417283",
        "eager_kind": "vllm-eager-decode",
        "production_kind": "vllm-production",
        "variant_id": "cg_decode_gbs064_8k1k",
        "file_stem": "cg_decode_gbs064_8k1k",
    },
    "decode-c256": {
        "phase": "decode",
        "batch_size": 256,
        "job_id": "3420178",
        "eager_job_id": "3417284",
        "eager_kind": "vllm-eager-decode",
        "production_kind": "vllm-production",
        "variant_id": "cg_decode_gbs256_8k1k",
        "file_stem": "cg_decode_gbs256_8k1k",
    },
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def interval_union_us(events: Iterable[dict[str, Any]]) -> float:
    intervals = sorted(
        (float(event["ts_us"]), float(event["ts_us"]) + float(event["dur_us"]))
        for event in events
        if float(event["dur_us"]) > 0
    )
    if not intervals:
        return 0.0
    merged = [list(intervals[0])]
    for start, stop in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], stop)
        else:
            merged.append([start, stop])
    return sum(stop - start for start, stop in merged)


def kernel_summary(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for event in events:
        grouped[str(event["kernel_name"])].append(float(event["dur_us"]))
    return [
        {
            "name": name,
            "count": len(durations),
            "count_per_iter": float(len(durations)),
            "avg_us": round(sum(durations) / len(durations), 6),
            "total_us_per_iter": round(sum(durations), 6),
        }
        for name, durations in sorted(
            grouped.items(), key=lambda item: (-sum(item[1]), item[0])
        )
    ]


def metric_from_events(
    events: list[dict[str, Any]],
    *,
    attribution_status: str,
    metric_kind: str,
    timing_semantics: str,
) -> dict[str, Any]:
    if not events:
        raise ValueError("cannot build a measured metric from an empty event set")
    active_ms = interval_union_us(events) / 1000.0
    residency_ms = sum(float(event["dur_us"]) for event in events) / 1000.0
    methods = Counter(str(event.get("attribution_method") or "") for event in events)
    metric = {
        "ms_per_iter": round(active_ms, 6),
        "active_gpu_ms": round(active_ms, 6),
        "gpu_residency_ms": round(residency_ms, 6),
        "gpu_residency_ms_per_iter": round(residency_ms, 6),
        "attribution_status": attribution_status,
        "metric_kind": metric_kind,
        "timing_semantics": timing_semantics,
        "mapped_event_count": len(events),
        "attribution": {
            "methods": {
                method: {
                    "kernel_count": count,
                    "gpu_residency_ms_per_iter": round(
                        sum(
                            float(event["dur_us"])
                            for event in events
                            if str(event.get("attribution_method") or "") == method
                        )
                        / 1000.0,
                        6,
                    ),
                }
                for method, count in sorted(methods.items())
            }
        },
        "kernels": kernel_summary(events),
    }
    layer_ids = sorted(
        {int(event["layer_id"]) for event in events if event.get("layer_id") is not None}
    )
    occurrence_ids = sorted(
        {str(event["occurrence_id"]) for event in events if event.get("occurrence_id")}
    )
    if layer_ids:
        metric["layer_ids"] = layer_ids
    if occurrence_ids:
        metric["required_occurrence_count"] = len(occurrence_ids)
    return metric


def fusion_specs() -> dict[str, dict[str, Any]]:
    """Return disjoint semantic groups with one timing owner each."""

    return {
        "vllm_mhc_pre_and_norm": {
            "owner": "mhc_transform.affine",
            "ir_nodes": [
                "mhc_transform.flatten_rms",
                "mhc_transform.affine",
                "mhc_transform.pre_gate",
                "mhc_transform.post_gate",
                "mhc_transform.combine_sinkhorn",
                "mhc_transform.read",
                "decoder_stack.attention_norm",
                "decoder_stack.ffn_norm",
            ],
            "source_nodes": {"mhc_transform.affine"},
            "proof": "source- and eager-proved mHC pre/Sinkhorn/read plus decoder RMSNorm fusion",
        },
        "vllm_mhc_post": {
            "owner": "mhc_transform.mix",
            "ir_nodes": ["mhc_transform.place", "mhc_transform.mix"],
            "source_nodes": {"mhc_transform.mix"},
            "proof": "mhc_post equation and fused_post_pre first-kernel contract",
        },
        "vllm_final_hc_read": {
            "owner": "final_hc_read.read",
            "ir_nodes": [
                "final_hc_read.flatten_rms",
                "final_hc_read.pre_gate",
                "final_hc_read.read",
            ],
            "source_nodes": {"final_hc_read.read"},
            "proof": "hc_head fused normalization, gate, and residual-stream read",
        },
        "vllm_csa_q_head_rope_window_kv": {
            "owner": "csa_attention.q_head_norm",
            "ir_nodes": [
                "csa_attention.q_head_norm",
                "csa_attention.q_rope",
                "csa_attention.window_kv",
                "csa_attention.window_cache",
            ],
            "source_nodes": {"csa_attention.q_head_norm"},
            "proof": "one exact fused Q-head norm/RoPE and KV RoPE/quantize/cache-insert physical event set; q_a and q_norm retain their independent events",
        },
        "vllm_hca_q_head_rope_window_kv": {
            "owner": "hca_attention.q_head_norm",
            "ir_nodes": [
                "hca_attention.q_head_norm",
                "hca_attention.q_rope",
                "hca_attention.window_kv",
                "hca_attention.window_cache",
            ],
            "source_nodes": {"hca_attention.q_head_norm"},
            "proof": "one exact fused Q-head norm/RoPE and KV RoPE/quantize/cache-insert physical event set; q_a and q_norm retain their independent events",
        },
        "vllm_csa_compressor": {
            "owner": "csa_compressor.softmax_pool",
            "ir_nodes": [
                "csa_compressor.overlap_layout",
                "csa_compressor.softmax_pool",
                "csa_compressor.norm_rope",
                "csa_compressor.compressed_cache",
            ],
            "source_nodes": {"csa_compressor.softmax_pool"},
            "proof": "SparseAttnCompressNormRopeStoreC4 fused overlap/pool/norm/RoPE/cache store",
        },
        "vllm_hca_compressor_pool": {
            "owner": "hca_compressor.softmax_pool",
            "ir_nodes": [
                "hca_compressor.overlap_layout",
                "hca_compressor.softmax_pool",
            ],
            "source_nodes": {"hca_compressor.softmax_pool"},
            "proof": "SparseAttnCompressC128 non-overlap block pooling",
        },
        "vllm_hca_compressor_store": {
            "owner": "hca_compressor.norm_rope",
            "ir_nodes": [
                "hca_compressor.norm_rope",
                "hca_compressor.compressed_cache",
            ],
            "source_nodes": {"hca_compressor.norm_rope"},
            "proof": "SparseAttnNormRopeStore HCA normalization/RoPE/cache store",
        },
        "vllm_csa_index_union": {
            "owner": "csa_attention.index_union",
            "ir_nodes": [
                "csa_attention.window_indices",
                "csa_attention.index_union",
            ],
            "source_nodes": {"csa_attention.index_union"},
            "proof": "exact window-plus-compressed-history index construction kernels",
        },
        "vllm_hca_index_union": {
            "owner": "hca_attention.index_union",
            "ir_nodes": [
                "hca_attention.window_indices",
                "hca_attention.index_union",
            ],
            "source_nodes": {"hca_attention.index_union"},
            "proof": "exact window-plus-compressed-history index construction kernels",
        },
        "vllm_routed_gate_up_swiglu": {
            "owner": "moe.routed_gate_up",
            "ir_nodes": ["moe.routed_gate_up", "moe.routed_activation"],
            "source_nodes": {"moe.routed_gate_up"},
            "proof": "FlashInfer routed gate/up BMM clmp_swiGlu epilogue signature",
        },
    }


def semantic_rollup_specs() -> dict[str, set[str]]:
    """Return non-exclusive semantic views over distinct physical event sets."""

    return {
        "moe.sqrt_softplus": {"moe.hash_select", "moe.learned_select"},
        "moe.weights": {"moe.hash_select", "moe.learned_select"},
    }


def compiled_nodes(model_ir: dict[str, Any], execution: dict[str, Any]) -> dict[str, dict[str, Any]]:
    nodes = {
        f"{view_id}.{node['id']}": node
        for view_id, view in (model_ir.get("views") or {}).items()
        for node in view.get("nodes") or []
    }
    for transform in execution.get("transforms") or []:
        if transform.get("op") not in {"insert_before", "insert_after"}:
            continue
        anchor = str(transform.get("before") or transform.get("after") or "")
        node = dict(transform.get("node") or {})
        if "." not in anchor or not node.get("id"):
            raise ValueError(f"invalid insertion transform: {transform}")
        nodes[f"{anchor.split('.', 1)[0]}.{node['id']}"] = node
    return nodes


def parent_targets(event: dict[str, Any]) -> list[str]:
    node = str(event["node"])
    targets: list[str] = []
    occurrence = str(event.get("occurrence_id") or "")
    substage = str(event.get("substage") or "")
    layer_kind = str(event.get("layer_kind") or "")

    if occurrence:
        targets.append("top.decoder_stack")
    if substage == "attention":
        if node == "mhc_transform.affine":
            targets.append("decoder_stack.attention_mhc_pre")
        elif node == "mhc_transform.mix":
            targets.append("decoder_stack.attention_mhc_post")
        elif node.endswith(".tp_csa_output_collective") or node.endswith(
            ".tp_hca_output_collective"
        ):
            pass
        elif layer_kind == "csa":
            targets.append("decoder_stack.csa_attention")
        elif layer_kind == "hca":
            targets.append("decoder_stack.hca_attention")
        if node.startswith("csa_compressor."):
            targets.append("csa_attention.csa_compressor")
        if node.startswith("hca_compressor."):
            targets.append("hca_attention.hca_compressor")
        if node.startswith("csa_indexer."):
            targets.append("csa_attention.indexer")
    elif substage == "feed_forward":
        if node == "mhc_transform.affine":
            targets.append("decoder_stack.ffn_mhc_pre")
        elif node == "mhc_transform.mix":
            targets.append("decoder_stack.ffn_mhc_post")
        elif node != "moe.tp_moe_output_collective" and node.startswith("moe."):
            targets.append("decoder_stack.moe")
    if node == "final_hc_read.read":
        targets.append("top.final_hc_read")
    return targets


def prepare_events(
    events: list[dict[str, Any]],
    groups: dict[str, dict[str, Any]],
    rollups: dict[str, set[str]] | None = None,
) -> list[dict[str, Any]]:
    group_targets_by_source: dict[str, list[str]] = defaultdict(list)
    owner_by_source: dict[str, str] = {}
    for group_id, group in groups.items():
        owner = str(group["owner"])
        sources = {str(source) for source in group["source_nodes"]}
        if sources != {owner}:
            raise ValueError(
                f"fusion group {group_id} must prove one physical owner event set; "
                f"source_nodes={sorted(sources)} owner={owner}"
            )
        for source in group["source_nodes"]:
            group_targets_by_source[source].extend(group["ir_nodes"])
            if source in owner_by_source and owner_by_source[source] != group["owner"]:
                raise ValueError(f"source node {source} has multiple timing owners")
            owner_by_source[source] = str(group["owner"])
    rollup_targets_by_source: dict[str, list[str]] = defaultdict(list)
    for target, sources in (rollups or semantic_rollup_specs()).items():
        for source in sources:
            rollup_targets_by_source[source].append(target)
    prepared = []
    for raw in events:
        event = dict(raw)
        event["timing_owner"] = owner_by_source.get(
            str(event["node"]), str(event["node"])
        )
        event["kernel_label"] = event["timing_owner"]
        event["ir_targets"] = list(
            dict.fromkeys(
                [
                    str(event["node"]),
                    *group_targets_by_source.get(str(event["node"]), []),
                    *rollup_targets_by_source.get(str(event["node"]), []),
                    *parent_targets(event),
                ]
            )
        )
        event["eager_event_ids"] = [str(value) for value in event.get("eager_event_ids") or []]
        if not event["eager_event_ids"]:
            raise ValueError(f"production event {event.get('event_id')} lacks eager evidence")
        prepared.append(event)
    return prepared


def fusion_groups_and_metrics(
    events: list[dict[str, Any]], specs: dict[str, dict[str, Any]]
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], set[str]]:
    groups: dict[str, dict[str, Any]] = {}
    metrics: dict[str, dict[str, Any]] = {}
    nonowners: set[str] = set()
    for group_id, spec in specs.items():
        owner = str(spec["owner"])
        sources = {str(source) for source in spec["source_nodes"]}
        if sources != {owner}:
            raise ValueError(
                f"fusion group {group_id} cannot share different physical event sets: "
                f"source_nodes={sorted(sources)} owner={owner}"
            )
        conflicting = [
            event
            for event in events
            if event["node"] in set(spec["ir_nodes"]) - {owner}
        ]
        if conflicting:
            raise ValueError(
                f"fusion group {group_id} member has an independent physical event set: "
                f"{sorted({str(event['node']) for event in conflicting})}"
            )
        selected = [event for event in events if event["node"] == owner]
        if not selected:
            # Index-union launches are decode-shape dependent.  A group with no
            # physical event is not authored; its semantic controls remain an
            # explicit source-defined structural state in that profile.
            continue
        production_event_ids = sorted({str(event["event_id"]) for event in selected})
        eager_event_ids = sorted(
            {str(eager) for event in selected for eager in event["eager_event_ids"]}
        )
        occurrence_ids = sorted(
            {str(event["occurrence_id"]) for event in selected if event.get("occurrence_id")}
        )
        layer_ids = sorted(
            {int(event["layer_id"]) for event in selected if event.get("layer_id") is not None}
        )
        substages = sorted(
            {str(event["substage"]) for event in selected if event.get("substage")}
        )
        production_event_set_sha256 = hashlib.sha256(
            json.dumps(production_event_ids, separators=(",", ":")).encode()
        ).hexdigest()
        eager_event_set_sha256 = hashlib.sha256(
            json.dumps(eager_event_ids, separators=(",", ":")).encode()
        ).hexdigest()
        member_event_sets = {
            member: {
                "production_event_count": len(production_event_ids),
                "production_event_set_sha256": production_event_set_sha256,
                "eager_event_count": len(eager_event_ids),
                "eager_event_set_sha256": eager_event_set_sha256,
            }
            for member in spec["ir_nodes"]
        }
        groups[group_id] = {
            "owner": owner,
            "ir_nodes": list(spec["ir_nodes"]),
            "timing_semantics": "shared_event_set",
            "provenance": str(spec["proof"]),
            "mapping_method": "graph-on production events reconciled to exact graph-off eager event sets",
            "confidence": "exact",
            "event_set_identity": "all semantic members map to this exact same physical event-id set",
            "member_event_sets": member_event_sets,
            "evidence_scope": {
                "resolution": "profile_aggregate",
                "layer_ids": layer_ids,
                "substages": substages,
                "occurrence_ids": occurrence_ids,
                "production_event_ids": production_event_ids,
                "eager_event_ids": eager_event_ids,
            },
        }
        metrics[owner] = metric_from_events(
            selected,
            attribution_status="measured_fusion_owner",
            metric_kind="exclusive_fusion_owner",
            timing_semantics="one exact shared physical production event-id set; counted once",
        )
        nonowners.update(set(spec["ir_nodes"]) - {owner})
    return groups, metrics, nonowners


def add_direct_metrics(
    metrics: dict[str, dict[str, Any]],
    events: list[dict[str, Any]],
    groups: dict[str, dict[str, Any]],
) -> None:
    covered = {node for group in groups.values() for node in group["ir_nodes"]}
    by_node: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        by_node[str(event["node"])].append(event)
    for node, selected in sorted(by_node.items()):
        if node in covered:
            continue
        metrics[node] = metric_from_events(
            selected,
            attribution_status="measured_direct",
            metric_kind="exclusive_leaf",
            timing_semantics="union of directly attributed production-kernel intervals; overlap counted once",
        )


def add_rollup_metrics(
    metrics: dict[str, dict[str, Any]],
    events: list[dict[str, Any]],
    groups: dict[str, dict[str, Any]],
) -> None:
    targets = sorted({target for event in events for target in event.get("ir_targets") or []})
    direct_nodes = {str(event["node"]) for event in events}
    fusion_nonowners = {
        member
        for group in groups.values()
        for member in group["ir_nodes"]
        if member != group["owner"]
    }
    for target in targets:
        if target in metrics or target in direct_nodes or target in fusion_nonowners:
            continue
        selected = [event for event in events if target in (event.get("ir_targets") or [])]
        if not selected:
            continue
        metric = metric_from_events(
            selected,
            attribution_status="inclusive_rollup",
            metric_kind="inclusive_rollup",
            timing_semantics="union of validated descendant production-kernel intervals; overlap counted once",
        )
        metric["rollup_sources"] = sorted({str(event["node"]) for event in selected})
        metrics[target] = metric


def add_communication_contracts(
    metrics: dict[str, dict[str, Any]], nodes: dict[str, dict[str, Any]]
) -> None:
    for target, metric in metrics.items():
        execution = nodes.get(target, {}).get("execution") or {}
        collective = execution.get("collective")
        if not collective:
            continue
        metric["communication"] = {
            "group": execution.get("group"),
            "payload": execution.get("payload"),
            "result": execution.get("result"),
            "observed_collectives": [
                {
                    "kind": collective,
                    "kernel_count_per_iter": metric["mapped_event_count"],
                }
            ],
        }


def node_states(
    nodes: dict[str, dict[str, Any]],
    metrics: dict[str, dict[str, Any]],
    groups: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    group_for_member = {
        member: (group_id, group)
        for group_id, group in groups.items()
        for member in group["ir_nodes"]
        if member != group["owner"]
    }
    states: dict[str, dict[str, Any]] = {}
    for target, node in sorted(nodes.items()):
        if target in metrics:
            continue
        if target in group_for_member:
            group_id, group = group_for_member[target]
            owner = str(group["owner"])
            states[target] = {
                "status": "fused",
                "label": f"fused into {owner}; one measured event set is owned only by {owner}",
                "included_in": owner,
                "fusion_group_id": group_id,
            }
            continue
        if target == "top.dspark_extension" or target.startswith(("dspark_generation.", "dspark_stage.", "dspark_attention.")):
            states[target] = {
                "status": "not_selected",
                "label": "Stage-1 is the exact autoregressive target path with DSpark disabled; this semantic contract remains available but has no event in this profile",
            }
            continue
        semantic = node.get("semantics") or {}
        kind = str(semantic.get("kind") or node.get("node_kind") or "boundary")
        states[target] = {
            "status": "structural",
            "label": f"source-defined {kind} boundary/control/state with no independent GPU interval; measured consumer and descendant events remain explicit",
        }
    unknown = sorted(set(nodes) - set(metrics) - set(states))
    if unknown:
        raise ValueError(f"profile closure missed nodes: {unknown}")
    return states


def find_single(path: Path, pattern: str) -> Path:
    matches = sorted(path.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"expected one {pattern} under {path}, got {matches}")
    return matches[0]


def profile_timing(events: list[dict[str, Any]], report: dict[str, Any]) -> dict[str, Any]:
    start = float(report["selected_window_start_us"])
    stop = float(report["selected_window_end_us"])
    elapsed_us = float(report["selected_wall_elapsed_us"])
    if abs((stop - start) - elapsed_us) > 1e-3:
        raise ValueError("selected wall interval and elapsed time disagree")
    for event in events:
        if float(event["ts_us"]) < start - 1e-3 or float(event["ts_us"]) + float(event["dur_us"]) > stop + 1e-3:
            raise ValueError(f"event {event['event_id']} falls outside selected wall interval")
    active_us = interval_union_us(events)
    residency_us = sum(float(event["dur_us"]) for event in events)
    return {
        "elapsed_ms": round(elapsed_us / 1000.0, 6),
        "active_gpu_ms": round(active_us / 1000.0, 6),
        "gpu_residency_ms": round(residency_us / 1000.0, 6),
        "device_gap_ms": round(max(0.0, elapsed_us - active_us) / 1000.0, 6),
        "gpu_overlap_ms": round(max(0.0, residency_us - active_us) / 1000.0, 6),
        "kernel_envelope_ms": round(elapsed_us / 1000.0, 6),
        "authority": "instrumented_trace_attribution_only",
        "semantics": "instrumented critical-rank trace interval; active is the cross-stream interval union and residency is the kernel-duration sum; this is not production wall-latency authority",
    }


def production_wall_timing(
    *,
    task_root: Path,
    framework: str,
    phase: str,
    batch_size: int,
    source_commit: str,
) -> dict[str, Any] | None:
    """Load the exact profiler-off matched scheduler-step wall authority."""

    if phase != "decode":
        return {
            "status": "unavailable",
            "authority": "none",
            "comparison_policy": "do_not_use_instrumented_trace_as_cross-framework_production_latency",
            "instrumented_trace_policy": "attribution evidence only",
            "reason": "no matched profiler-off pure-prefill scheduler-wall artifact is retained for this profile",
        }
    selection_spec = BASELINE_SELECTIONS[framework]
    selection_path = task_root / str(selection_spec["path"])
    if not selection_path.is_file():
        raise ValueError(f"missing profiler-off wall authority: {selection_path}")
    digest = file_sha256(selection_path)
    if digest != selection_spec["sha256"]:
        raise ValueError(
            f"profiler-off wall authority hash mismatch: {selection_path}"
        )
    baseline = load_json(selection_path)
    source_lock = baseline.get("source_lock") or {}
    if source_lock.get("commit") != source_commit:
        raise ValueError("profiler-off wall authority source commit mismatch")
    key = str(batch_size)
    if framework == "vllm":
        selection = (baseline.get("selections") or {}).get(key) or {}
        selected = selection.get("selected_decode_iteration") or {}
        elapsed_ms = selected.get("elapsed_ms")
        coordinate = {
            "iteration": selected.get("iteration"),
            "formal_step_index_1_based": selected.get("formal_step_index_1_based"),
            "plateau_median_elapsed_ms": (
                selection.get("exact_decode_plateau") or {}
            ).get("median_elapsed_ms"),
        }
    else:
        selection = (baseline.get("concurrencies") or {}).get(key) or {}
        selected = selection.get("selected_decode") or {}
        elapsed_ms = selected.get("baseline_mean_elapsed_ms")
        coordinate = {
            "baseline_global_step": selected.get("baseline_global_step"),
            "profile_relative_step": selected.get("profile_relative_step"),
            "throughput_token_s": selected.get("throughput_token_s"),
        }
    if elapsed_ms is None or float(elapsed_ms) <= 0:
        raise ValueError(
            f"profiler-off wall authority lacks decode GBS{batch_size} elapsed time"
        )
    if any(value is None for value in coordinate.values()):
        raise ValueError(
            f"profiler-off wall authority lacks decode GBS{batch_size} coordinate"
        )
    return {
        "elapsed_ms": round(float(elapsed_ms), 6),
        "authority": "profiler_off_matched_scheduler_step",
        "comparison_policy": "cross-framework performance and gap analysis must use this wall value",
        "instrumented_trace_policy": "attribution evidence only; never substitute its elapsed time as production latency",
        "source_file": str(selection_spec["path"]),
        "source_sha256": digest,
        "coordinate": coordinate,
    }


def build_one(
    *,
    repo_root: Path,
    task_root: Path,
    output_dir: Path,
    name: str,
    spec: dict[str, Any],
    matrix: dict[str, Any],
    nodes: dict[str, dict[str, Any]],
    reconciliation_framework: str = "vllm",
    profile_framework: str = "vllm",
    framework_label: str = "vLLM",
    implementation_id: str = IMPLEMENTATION_ID,
    source_commit: str = SOURCE_COMMIT,
    container_sha256: str = CONTAINER_SHA256,
    matrix_report_sha256: str = MATRIX_REPORT_SHA256,
    matrix_manifest_sha256: str = MATRIX_MANIFEST_SHA256,
    fusion_spec_map: dict[str, dict[str, Any]] | None = None,
    trace_pattern: str = "*rank{rank}.*trace.json.gz",
    source_overlay: dict[str, Any] | None = None,
    mapping_root_name: str = "mappings-phase-tail",
) -> tuple[Path, str]:
    matrix_profile = matrix["profiles"][name]
    rank = int(
        max(
            matrix_profile["rank_selected_wall_elapsed_us"],
            key=lambda item: matrix_profile["rank_selected_wall_elapsed_us"][item],
        )
    )
    reconciliation_dir = (
        task_root
        / "production-reconciliation"
        / reconciliation_framework
        / name
        / f"rank{rank}"
    )
    event_path = reconciliation_dir / "events.jsonl"
    report_path = reconciliation_dir / "report.json"
    events = load_jsonl(event_path)
    report = load_json(report_path)
    if not report.get("ok") or report["rank"] != rank:
        raise ValueError(f"critical-rank reconciliation did not pass: {report_path}")
    if report["kernel_count"] != len(events) or report["mapped_kernel_count_ratio"] != 1.0:
        raise ValueError(f"critical-rank event closure failed: {report_path}")

    mapping_dir = (
        task_root
        / mapping_root_name
        / spec["eager_kind"]
        / spec["eager_job_id"]
        / f"rank{rank}"
    )
    mapping_path = find_single(mapping_dir, "kernel_mapping.*.jsonl")
    selected_fusion_specs = fusion_spec_map or fusion_specs()
    prepared = prepare_events(events, selected_fusion_specs)
    prepared = attach_eager_stack_evidence(prepared, mapping_path=mapping_path)

    groups, metrics, _ = fusion_groups_and_metrics(prepared, selected_fusion_specs)
    add_direct_metrics(metrics, prepared, groups)
    add_rollup_metrics(metrics, prepared, groups)
    add_communication_contracts(metrics, nodes)
    states = node_states(nodes, metrics, groups)
    timing = profile_timing(prepared, report)

    phase = str(spec["phase"])
    batch = int(spec["batch_size"])
    wall_timing = production_wall_timing(
        task_root=task_root,
        framework=profile_framework,
        phase=phase,
        batch_size=batch,
        source_commit=source_commit,
    )
    profile_id = f"deepseek_v4_pro_tp8_{profile_framework}_{spec['variant_id']}"
    trace_dir = task_root / "evidence" / spec["production_kind"] / spec["job_id"] / "traces"
    trace_path = find_single(trace_dir, trace_pattern.format(rank=rank))
    if file_sha256(trace_path) != report["trace"]["sha256"]:
        raise ValueError(f"raw trace hash mismatch: {trace_path}")
    if source_overlay is not None:
        source_lock_path = trace_dir.parent / "profiler-overlay-source-lock.json"
        if not source_lock_path.is_file():
            raise ValueError(f"profile lacks profiler overlay source lock: {source_lock_path}")
        actual_source_lock_sha256 = file_sha256(source_lock_path)
        if actual_source_lock_sha256 != source_overlay["source_lock_sha256"]:
            raise ValueError(
                f"profiler overlay source lock hash mismatch: {source_lock_path}"
            )
        source_lock = load_json(source_lock_path)
        if (source_lock.get("base") or {}).get("commit") != source_commit:
            raise ValueError("profiler overlay base source commit mismatch")
        overlay_files = (source_lock.get("overlay") or {}).get("files") or {}
        if overlay_files.get("python/sglang/srt/managers/scheduler.py") != source_overlay[
            "scheduler_sha256"
        ]:
            raise ValueError("profiler overlay scheduler hash mismatch")
        if overlay_files.get(
            "python/sglang/srt/managers/scheduler_components/profiler_manager.py"
        ) != source_overlay["profiler_manager_sha256"]:
            raise ValueError("profiler overlay manager hash mismatch")
        source_overlay = {
            **source_overlay,
            "evidence_source_lock_file": source_lock_path.name,
            "evidence_source_lock_sha256": actual_source_lock_sha256,
        }
    validation_path = trace_dir.parent / "validation.json"
    validation = load_json(validation_path)
    expected_run_kind = "production" if phase == "decode" else "prefill_timing"
    if validation.get("status") != "pass":
        raise ValueError(f"run validation did not pass: {validation_path}")
    if validation.get("framework") != profile_framework:
        raise ValueError(f"run validation framework mismatch: {validation_path}")
    if validation.get("run_kind") != expected_run_kind:
        raise ValueError(f"run validation kind mismatch: {validation_path}")
    if validation.get("trace_ranks") != list(range(8)):
        raise ValueError(f"run validation does not retain all TP ranks: {validation_path}")
    validation_sha256 = file_sha256(validation_path)

    step = {
        "step_index": 0,
        "label": f"{phase} GBS{batch} critical rank {rank}",
        "trace_start_us": float(report["selected_window_start_us"]),
        "duration_us": float(report["selected_wall_elapsed_us"]),
        "events": prepared,
    }
    timeline = build_timeline_artifact(
        profile_id=profile_id,
        phase=phase,
        reference_rank=rank,
        steps=[step],
        timing_summary=timing,
        raw_trace={
            "file": trace_path.name,
            "format": "pytorch_chrome_trace_gzip",
            "rank": rank,
            "sha256": report["trace"]["sha256"],
            "storage": "task_evidence_only",
        },
        stack_source={
            "source": "graph_off_eager_trace",
            "mapping_file": mapping_path.name,
            "mapping_sha256": file_sha256(mapping_path),
            "policy": "every production event carries exact eager event IDs; full or representative eager Python stacks are attached without claiming graph-on stacks",
        },
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    timeline_path = output_dir / f"{spec['file_stem']}.timeline.json.gz"
    timeline_sha256 = write_timeline_artifact(timeline_path, timeline)

    rank_wall = {
        str(key): round(float(value) / 1000.0, 6)
        for key, value in matrix_profile["rank_selected_wall_elapsed_us"].items()
    }
    profile = {
        "schema_version": "profile.v2",
        "profile_id": profile_id,
        "label": f"GB300 · {framework_label} · pure TP8 · {'CUDA Graph decode' if phase == 'decode' else 'graph-off stable prefill'} · GBS{batch} · 8k→1k",
        "model_id": "deepseek_v4_pro",
        "execution_path_id": EXECUTION_PATH,
        "implementation_id": implementation_id,
        "variant_id": spec["variant_id"],
        "phase": phase,
        "generation_mode": "autoregressive",
        "entry_view": "top",
        "execution_parameters": {"tp_size": 8, "dp_size": 1, "cp_size": 1, "ep_size": 1},
        "hardware": {
            "gpu": "NVIDIA GB300",
            "gpus_per_node": 4,
            "nodes": 2,
            "topology": "pure TP8",
        },
        "workload": {
            "isl": 8192,
            "osl": 1024,
            "batch_size": batch,
            "concurrency": batch,
            "warmup_requests": 3 * batch,
            "formal_requests": batch,
            "prompt_source": "deterministic_random_token_ids",
            "prompt_seed": 0,
            "ignore_eos": True,
            "prefix_cache_enabled": False,
            "hicache_enabled": False,
            "kv_offload_enabled": False,
            "dspark_enabled": False,
        },
        "profiler": {
            "type": "torch_profiler",
            "representative_rank": rank,
            "activities": ["CUDA"],
            "all_tp_ranks_validated": True,
            "timing_gate_status": "passed",
            "cuda_graph_enabled": phase == "decode",
            "with_stack": False,
            "record_shapes": False,
            "semantic_stack_source": "exact same-rank graph-off eager mapping with stack and shape evidence",
            "invocation_segment": report["window_selector"],
            "gpu_metric_semantics": timing["semantics"],
        },
        "evidence": {
            "job_id": str(spec["job_id"]),
            "source_commit": source_commit,
            "model_revision": MODEL_REVISION,
            "container_sha256": container_sha256,
            "validation_file": "validation.json",
            "validation_sha256": validation_sha256,
            "raw_trace_sha256": report["trace"]["sha256"],
            "eager_mapping_sha256": file_sha256(mapping_path),
            "eager_contract_sha256": report["eager_contract"]["sha256"],
            "production_events_sha256": file_sha256(event_path),
            "production_report_sha256": file_sha256(report_path),
            "production_matrix_sha256": matrix_report_sha256,
            "production_artifact_manifest_sha256": matrix_manifest_sha256,
            "mapped_kernel_count_ratio": 1.0,
            "mapped_kernel_duration_ratio": 1.0,
            "occurrence_count": report["occurrence_count"],
            "mhc_implementation_path": report["mhc_implementation_path"],
            "critical_rank": rank,
            "rank_count": matrix_profile["rank_count"],
            "rank_instrumented_trace_elapsed_ms": rank_wall,
            "rank_instrumented_trace_policy": "attribution and rank-alignment evidence only; not profiler-off production wall authority",
            "rank_reconciliation_fingerprints": matrix_profile["rank_reconciliation_fingerprints"],
            "rank_ordered_structural_fingerprints": matrix_profile["rank_ordered_structural_fingerprints"],
            "structural_multiset_fingerprint": matrix_profile["structural_multiset_fingerprint"],
            "rank_ordering_policy": matrix_profile["rank_ordering_policy"],
            "mapping_policy": "100% graph-on events reconciled to same-phase, same-shape, same-rank eager event sets; N:1 and 1:N event IDs retained",
            "attribution_timing": timing,
            "production_wall_timing": wall_timing,
        },
        "timeline": {
            "schema_version": "timeline.v1",
            "artifact": timeline_path.name,
            "sha256": timeline_sha256,
            "reference_rank": rank,
            "step_count": 1,
            "event_count": len(prepared),
            "raw_trace_file": trace_path.name,
        },
        "node_states": states,
        "fusion_groups": groups,
        "profile_summary": {
            "timing": timing,
            "production_wall_timing": wall_timing,
            "kernel_count": len(prepared),
            "mapped_kernel_count": len(prepared),
            "mapped_kernel_count_ratio": 1.0,
            "mapped_kernel_duration_ratio": 1.0,
            "semantic_occurrence_count": report["occurrence_count"],
            "timing_owner_policy": "one owner only for an identical physical event-id set; direct independent sets remain separate and module rollups are non-exclusive interval unions",
        },
        "node_metrics": dict(sorted(metrics.items())),
    }
    if source_overlay is not None:
        profile["profiler"]["source_overlay"] = source_overlay
        profile["evidence"]["source_overlay"] = source_overlay
    for evidence_key in (
        "collective_rank_duration_audit",
        "formal_step_throughput_gate",
    ):
        if matrix_profile.get(evidence_key) is not None:
            profile["evidence"][evidence_key] = matrix_profile[evidence_key]
    profile_path = output_dir / f"{spec['file_stem']}.yaml"
    profile_path.write_text(
        yaml.safe_dump(profile, sort_keys=False, allow_unicode=True, width=120)
    )
    return profile_path, file_sha256(profile_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    task_root = args.task_root.resolve()
    output_dir = args.output_dir or (
        repo_root / "catalog/deepseek_v4_pro/profiles/tp8/vllm_dd10e03_dsv4pro0813_tp8"
    )
    model_ir = yaml.safe_load((repo_root / "catalog/deepseek_v4_pro/model_ir.yaml").read_text())
    execution = yaml.safe_load(
        (repo_root / "catalog/deepseek_v4_pro/execution_paths/tp8_moe_intermediate_shard.yaml").read_text()
    )
    nodes = compiled_nodes(model_ir, execution)
    if len(nodes) != 153:
        raise ValueError(f"expected 153 execution nodes, got {len(nodes)}")
    matrix_path = task_root / "production-reconciliation/vllm/matrix_report.json"
    if file_sha256(matrix_path) != MATRIX_REPORT_SHA256:
        raise ValueError("production matrix report hash does not match the accepted gate")
    matrix = load_json(matrix_path)
    if not matrix.get("ok") or matrix.get("profile_count") != 5:
        raise ValueError("production matrix did not pass all five vLLM profiles")
    outputs = []
    for name, spec in PROFILE_SPECS.items():
        path, digest = build_one(
            repo_root=repo_root,
            task_root=task_root,
            output_dir=output_dir,
            name=name,
            spec=spec,
            matrix=matrix,
            nodes=nodes,
        )
        try:
            reported_path = path.relative_to(repo_root)
        except ValueError:
            reported_path = path
        outputs.append({"path": str(reported_path), "sha256": digest})
    print(json.dumps({"ok": True, "profiles": outputs}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
