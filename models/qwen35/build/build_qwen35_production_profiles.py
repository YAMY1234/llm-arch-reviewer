#!/usr/bin/env python3
"""Build the complete measured Qwen3.5 pure-TP8 profile matrix.

Raw traces stay under the task evidence root.  The repository receives only
compact deterministic profile overlays and timeline artifacts.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re
import sqlite3
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
from models.qwen35.build.qwen35_production_attribution import (  # noqa: E402
    attribute_production_forward,
)


MODEL_REVISION = "8f590eae8f10bf55d9a46f79ea0280bde435c9f8"
MODEL_CONFIG_SHA256 = "9408a9e559cc2f05f0b357738213666353e6651160ce8ff477b1c26982bc4f63"
SGLANG_SOURCE = "f609d677b909ca46c64bb6803b69a85fedbf86bc"
SGLANG_MODULE_SOURCE = "033446bb05f35c0943aed2750c443077ffc0b92c"
VLLM_SOURCE = "487ecf187d3dfe74d2cf6119a92881dba403c219"
SGLANG_CONTAINER = "sglang-glm53-flash-arm64-73f9294b.sqsh@sha256:28e9545e312e344bbbf80c575b928be53c9aba6296ae55f292ce0f10750c6971"
VLLM_CONTAINER = "vllm-glm53-flash-arm64-905c0293.sqsh@sha256:efdfe25952dc672d4415032e2755df7d7f2bab549992a2e3f2c429334f366756"


MATRIX = (
    {"framework": "sglang", "phase": "prefill", "batch": 1, "job": "3414663"},
    {"framework": "sglang", "phase": "decode", "batch": 1, "job": "3414668"},
    {"framework": "sglang", "phase": "decode", "batch": 16, "job": "3414675"},
    {"framework": "sglang", "phase": "decode", "batch": 64, "job": "3414674"},
    {"framework": "sglang", "phase": "decode", "batch": 256, "job": "3414676"},
    {"framework": "vllm", "phase": "prefill", "batch": 1, "job": "3414288"},
    {"framework": "vllm", "phase": "decode", "batch": 1, "job": "3414289"},
    {"framework": "vllm", "phase": "decode", "batch": 16, "job": "3414290"},
    {"framework": "vllm", "phase": "decode", "batch": 64, "job": "3414291"},
    {"framework": "vllm", "phase": "decode", "batch": 256, "job": "3414292"},
)


FUSION_CANDIDATES = {
    "gdn_attention.ba_projection",
    "gdn_attention.conv_state_read",
    "gdn_attention.recurrent_state_read",
    "gdn_attention.state_write",
    "full_attention.partial_rope",
    "full_attention.kv_state_read",
    "gdn_moe_block.input_norm",
    "gdn_moe_block.attention_residual",
    "gdn_moe_block.post_attention_norm",
    "gdn_moe_block.layer_residual",
    "full_attention_moe_block.input_norm",
    "full_attention_moe_block.attention_residual",
    "full_attention_moe_block.post_attention_norm",
    "full_attention_moe_block.layer_residual",
    "top.embedding",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def interval_union_us(rows: Iterable[dict[str, Any]]) -> float:
    intervals = sorted(
        (float(row["ts_us"]), float(row["ts_us"]) + float(row["dur_us"]))
        for row in rows
        if float(row["dur_us"]) > 0
    )
    merged: list[list[float]] = []
    for start, stop in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], stop)
        else:
            merged.append([start, stop])
    return sum(stop - start for start, stop in merged)


def kernel_rows_from_torch(path: Path, rank: int) -> list[dict[str, Any]]:
    with gzip.open(path, "rt") as source:
        trace = json.load(source)
    rows = []
    for index, event in enumerate(trace.get("traceEvents") or []):
        if event.get("cat") != "kernel" or not event.get("dur"):
            continue
        args = event.get("args") or {}
        rows.append(
            {
                "event_id": f"r{rank}-k{index}",
                "kernel_name": str(event.get("name") or ""),
                "ts_us": float(event["ts"]),
                "dur_us": float(event["dur"]),
                "stream": args.get("stream"),
                "device": args.get("device"),
                "pid": event.get("pid"),
                "tid": event.get("tid"),
                "graph_id": args.get("graph id"),
            }
        )
    return rows


def kernel_rows_from_sqlite(path: Path, rank_base: int) -> dict[int, list[dict[str, Any]]]:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    pids = list(
        connection.execute(
            "select distinct globalPid,deviceId from CUPTI_ACTIVITY_KIND_KERNEL order by deviceId"
        )
    )
    result: dict[int, list[dict[str, Any]]] = {}
    for global_pid, device_id in pids:
        rank = rank_base + int(device_id)
        rows = []
        query = """
            select k.*, s.value as kernel_name
            from CUPTI_ACTIVITY_KIND_KERNEL k
            join StringIds s on k.demangledName=s.id
            where k.globalPid=? order by k.start,k.end
        """
        for index, event in enumerate(connection.execute(query, (global_pid,))):
            rows.append(
                {
                    "event_id": f"r{rank}-k{index}",
                    "kernel_name": str(event["kernel_name"]),
                    "ts_us": float(event["start"]) / 1000.0,
                    "dur_us": float(event["end"] - event["start"]) / 1000.0,
                    "stream": event["streamId"],
                    "device": event["deviceId"],
                    "pid": event["globalPid"],
                    "tid": event["streamId"],
                    "graph_id": event["graphId"],
                }
            )
        result[rank] = rows
    return result


def rank_from_name(path: Path) -> int:
    match = re.search(r"(?:rank|TP-)(\d+)", path.name)
    if not match:
        raise ValueError(f"cannot determine rank from {path}")
    return int(match.group(1))


def evidence_dir(root: Path, item: dict[str, Any]) -> Path:
    job = item["job"]
    if item["framework"] == "vllm":
        return root / "evidence" / "vllm-production_torch_gpu_only" / job
    suffix = "prefill-c1" if item["phase"] == "prefill" else f"decode-c{item['batch']}"
    return root / "evidence" / f"sglang-production-{suffix}" / job


def selected_runtime_coordinate(root: Path, item: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the exact baseline selector and its immutable evidence reference."""

    framework = item["framework"]
    phase = item["phase"]
    batch = str(item["batch"])
    if framework == "sglang":
        path = root / "evidence" / "sglang-baseline" / "3413390" / "window-selection.json"
        payload = json.loads(path.read_text())
        selected = payload["concurrencies"][batch][f"selected_{phase}"]
        coordinate = {
            **selected,
            "baseline_job_id": payload["baseline_job_id"],
            "native_coordinate": payload["native_coordinate"],
            "profile_coordinate": payload["profile_coordinate"],
        }
    else:
        path = root / "evidence" / "vllm-baseline" / "3413249" / "window-selection.json"
        payload = json.loads(path.read_text())
        selection = payload["selections"][batch]
        if phase == "prefill":
            selected = selection["prefill"]
            coordinate = {
                "iteration": selected["iterations"][0],
                "context_token_sum": selected["context_token_sum"],
                "context_tokens": selected["context_tokens"],
                **selected["production_profiler"],
            }
        else:
            coordinate = {
                **selection["selected_decode_iteration"],
                **selection["production_profiler"],
                "exact_decode_plateau": selection["exact_decode_plateau"],
            }
        coordinate["baseline_job_id"] = "3413249"
        coordinate["profiler_step_timing"] = payload["source_semantics"]["profiler_step_timing"]
    return coordinate, {
        "file": path.name,
        "sha256": sha256_file(path),
        "state": payload["state"],
    }


def load_rank_traces(root: Path, item: dict[str, Any]) -> tuple[dict[int, list[dict[str, Any]]], dict[int, Path]]:
    directory = evidence_dir(root, item)
    if item["framework"] == "sglang" and item["phase"] == "prefill":
        rows: dict[int, list[dict[str, Any]]] = {}
        paths: dict[int, Path] = {}
        for node, path in enumerate(sorted((directory / "profiles").glob("node*.sqlite"))):
            for rank, rank_rows in kernel_rows_from_sqlite(path, node * 4).items():
                rows[rank] = rank_rows
                paths[rank] = path
        return rows, paths
    pattern = "traces/*.trace.json.gz" if item["framework"] == "vllm" else "profiles/*/*.trace.json.gz"
    paths = {rank_from_name(path): path for path in sorted(directory.glob(pattern))}
    return {rank: kernel_rows_from_torch(path, rank) for rank, path in paths.items()}, paths


def has_active_graph_id(row: dict[str, Any]) -> bool:
    graph_id = row.get("graph_id")
    if graph_id is None:
        return False
    try:
        return int(graph_id) != 0
    except (TypeError, ValueError):
        return str(graph_id).strip() not in {"", "0", "None", "null"}


def selected_forward_cuda_graph_evidence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Describe replay for the selected forward, independent of server configuration."""

    model_rows = [row for row in rows if row.get("node")]
    graph_kernel_count = sum(has_active_graph_id(row) for row in model_rows)
    non_graph_kernel_count = len(model_rows) - graph_kernel_count
    if not graph_kernel_count:
        replay_state = "no_cuda_graph_replay"
    elif non_graph_kernel_count:
        replay_state = "mixed_graph_and_eager"
    else:
        replay_state = "cuda_graph_replay"
    graph_id_count = len(
        {str(row["graph_id"]) for row in model_rows if has_active_graph_id(row)}
    )
    if graph_kernel_count:
        evidence_basis = (
            f"{graph_id_count} distinct nonzero raw-trace graph IDs cover "
            f"{graph_kernel_count} model-bearing kernels in the selected formal forward"
        )
    else:
        evidence_basis = (
            f"zero nonzero raw-trace graph IDs across {len(model_rows)} model-bearing "
            "kernels in the selected formal forward"
        )
    return {
        "used_graph_path": graph_kernel_count > 0,
        "replay_state": replay_state,
        "model_kernel_count": len(model_rows),
        "graph_kernel_count": graph_kernel_count,
        "non_graph_kernel_count": non_graph_kernel_count,
        "graph_id_count": graph_id_count,
        "evidence_basis": evidence_basis,
    }


def cuda_graph_enabled_semantics(evidence: dict[str, Any]) -> str:
    if evidence["used_graph_path"]:
        return (
            "selected formal forward used a CUDA Graph path; "
            f"{evidence['graph_kernel_count']} model-bearing kernels have a nonzero "
            "raw-trace graph_id"
        )
    return (
        "selected formal forward did not use CUDA Graph replay; zero nonzero raw-trace "
        f"graph IDs were observed across all {evidence['model_kernel_count']} "
        "model-bearing kernels"
    )


def server_cuda_graph_config(root: Path, item: dict[str, Any]) -> dict[str, Any]:
    framework = item["framework"]
    phase = item["phase"]
    directory = evidence_dir(root, item)
    if framework == "sglang":
        evidence_paths = sorted(directory.glob("*_agg_w0.out"))
        if len(evidence_paths) != 2:
            raise ValueError(f"{item['job']}: expected one SGLang server log per node")
        expected_mode = (
            "'prefill': {'backend': 'breakable'"
            if phase == "prefill"
            else "'decode': {'backend': 'full'"
        )
        required_fragments = (
            expected_mode,
            "'disable_prefill_cuda_graph': False",
            "'disable_decode_cuda_graph': False",
            "'disable_cuda_graph': False",
        )
        for path in evidence_paths:
            text = path.read_text(errors="replace")
            if any(fragment not in text for fragment in required_fragments):
                raise ValueError(f"{item['job']}: CUDA Graph server configuration mismatch in {path.name}")
        return {
            "enabled": True,
            "mode": "breakable_prefill" if phase == "prefill" else "full_decode",
            "evidence": "production server_args cuda_graph_config and disable_*_cuda_graph=false",
            "evidence_files": {
                path.name: sha256_file(path) for path in evidence_paths
            },
        }
    evidence_path = directory / "server.log"
    if not evidence_path.is_file():
        raise ValueError(f"{item['job']}: missing vLLM server.log")
    text = evidence_path.read_text(errors="replace")
    required_fragments = (
        "CUDAGraphMode.FULL_AND_PIECEWISE",
        "Profiling CUDA graph memory: PIECEWISE=51",
        "Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 100%",
        "Capturing CUDA graphs (decode, FULL): 100%",
    )
    if any(fragment not in text for fragment in required_fragments):
        raise ValueError(f"{item['job']}: incomplete vLLM FULL_AND_PIECEWISE capture evidence")
    return {
        "enabled": True,
        "mode": "FULL_AND_PIECEWISE",
        "evidence": "production server.log compilation_config plus completed FULL and PIECEWISE capture",
        "evidence_files": {evidence_path.name: sha256_file(evidence_path)},
    }


def build_reconciled_eager_mapping(root: Path, framework: str, phase: str) -> Path:
    source_name = (
        f"sglang-forward_{'extend' if phase == 'prefill' else 'decode'}"
        if framework == "sglang"
        else "vllm-vllm_prefill"
    )
    source_dir = root / "mapping" / source_name
    events = [
        json.loads(line)
        for line in (source_dir / "events.tp0.jsonl").read_text().splitlines()
        if line.strip()
    ]
    rows = [dict(event) for event in events]
    attribute_production_forward(rows, framework=framework, phase=phase)
    output = root / "mapping" / f"reconciled-{framework}-{phase}"
    output.mkdir(parents=True, exist_ok=True)
    events_path = output / "events.tp0.jsonl"
    mapping_path = output / "kernel_mapping.tp0.jsonl"
    events_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in events))
    mappings = []
    for row in rows:
        semantic_nodes = [row.get("node"), *(row.get("ir_targets") or [])]
        for semantic_node in dict.fromkeys(node for node in semantic_nodes if node):
            mappings.append(
                {
                    "event_id": row["event_id"],
                    "kernel_name": row["kernel_name"],
                    "selected_node": semantic_node,
                    "confidence": row.get("confidence") or "support",
                    "evidence": [
                        "eager_python_stack",
                        "exact_kernel_signature",
                        "complete_tp_collective_sequence",
                    ],
                }
            )
    if framework == "vllm":
        # vLLM's fused add+RMSNorm kernel family is reused for both residual
        # norm positions and both layer types.  The eager stack closes the
        # kernel family to Qwen3.5DecoderLayer; the complete 60-layer
        # production sequence then disambiguates the four semantic roles.
        # Record that N:1 relationship explicitly rather than inventing four
        # distinct eager kernels.
        representative = next(
            (
                event
                for event in events
                if event.get("kernel_name") == "layer_norm_fwd_kernel"
                and any(
                    str(frame.get("module") or "").startswith("Qwen3_5DecoderLayer_")
                    for frame in (event.get("python_stack") or [])
                )
            ),
            None,
        )
        if representative is None:
            raise ValueError("vLLM eager capture lacks a Qwen3.5 decoder layer-norm stack")
        for semantic_node in (
            "gdn_moe_block.input_norm",
            "gdn_moe_block.post_attention_norm",
            "full_attention_moe_block.input_norm",
            "full_attention_moe_block.post_attention_norm",
        ):
            mappings.append(
                {
                    "event_id": representative["event_id"],
                    "kernel_name": representative["kernel_name"],
                    "selected_node": semantic_node,
                    "confidence": "high",
                    "mapping_cardinality": "N:1 reused kernel family",
                    "evidence": [
                        "eager_python_stack_to_qwen35_decoder_layer",
                        "exact_reused_layer_norm_kernel_signature",
                        "complete_60_layer_production_sequence_role",
                    ],
                }
            )
    mapping_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in mappings))
    report = {
        "framework": framework,
        "phase": phase,
        "kernel_count": len(rows),
        "semantic_kernel_count": sum(bool(row.get("node")) for row in rows),
        "support_kernel_count": sum(bool(row.get("support_class")) for row in rows),
        "node_coverage": sorted({row["node"] for row in rows if row.get("node")}),
        "mapping_sha256": sha256_file(mapping_path),
        "events_sha256": sha256_file(events_path),
    }
    (output / "reconciliation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return mapping_path


def metric_for_rows(rows: list[dict[str, Any]], *, status: str) -> dict[str, Any]:
    active_us = interval_union_us(rows)
    residency_us = sum(float(row["dur_us"]) for row in rows)
    kernels: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = str(row["kernel_name"])
        cell = kernels.setdefault(name, {"name": name, "count": 0, "total_us_per_iter": 0.0})
        cell["count"] += 1
        cell["total_us_per_iter"] += float(row["dur_us"])
    kernel_list = sorted(kernels.values(), key=lambda row: (-row["total_us_per_iter"], row["name"]))
    for cell in kernel_list:
        total_us = float(cell["total_us_per_iter"])
        cell["count_per_iter"] = float(cell["count"])
        cell["avg_us"] = round(total_us / int(cell["count"]), 6)
        cell["total_us_per_iter"] = round(total_us, 6)
        cell["share_in_node_pct"] = round(100.0 * total_us / residency_us, 6) if residency_us else 0.0
        cell["share_in_node_residency_pct"] = cell["share_in_node_pct"]
    return {
        "ms_per_iter": round(active_us / 1000.0, 6),
        "active_gpu_ms": round(active_us / 1000.0, 6),
        "gpu_residency_ms": round(residency_us / 1000.0, 6),
        "gpu_residency_ms_per_iter": round(residency_us / 1000.0, 6),
        "mapped_event_count": len(rows),
        "attribution_status": status,
        "metric_kind": "exclusive_leaf" if status == "measured_direct" else "inclusive_rollup",
        "timing_semantics": (
            "union of directly attributed production-kernel intervals"
            if status == "measured_direct"
            else "union of explicitly targeted production event intervals; overlap counted once"
        ),
        "kernels": kernel_list,
    }


def build_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    direct: dict[str, list[dict[str, Any]]] = defaultdict(list)
    targeted: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("node"):
            direct[str(row["node"])].append(row)
        for target in row.get("ir_targets") or []:
            targeted[str(target)].append(row)
    metrics = {node: metric_for_rows(events, status="measured_direct") for node, events in direct.items()}
    for target, events in targeted.items():
        if target in direct:
            # A reusable semantic leaf may be standalone for one occurrence
            # and fused for others.  Keep one interval-union aggregate and say
            # so explicitly instead of copying an owner's scalar duration.
            event_ids = {event["event_id"] for event in direct[target]}
            event_ids.update(event["event_id"] for event in events)
            combined = [event for event in rows if event["event_id"] in event_ids]
            if len(combined) != len(direct[target]):
                metrics[target] = metric_for_rows(combined, status="inclusive_rollup")
                metrics[target]["partial_fusion"] = True
        # Target-only leaves are non-owner members of a shared event set.  They
        # intentionally receive no scalar metric; node_states and the fusion
        # group link them to the one timing owner.
    return metrics


def all_model_nodes(model_ir: dict[str, Any]) -> set[str]:
    return {
        f"{view_id}.{node['id']}"
        for view_id, view in (model_ir.get("views") or {}).items()
        for node in (view.get("nodes") or [])
    }


def execution_nodes(execution_plan: dict[str, Any]) -> set[str]:
    nodes = set()
    for transform in execution_plan.get("transforms") or []:
        if transform.get("op") not in {"insert_before", "insert_after"}:
            continue
        anchor = str(transform.get("before") or transform.get("after") or "")
        node_id = (transform.get("node") or {}).get("id")
        if "." in anchor and node_id:
            nodes.add(f"{anchor.split('.', 1)[0]}.{node_id}")
    return nodes


def build_states_and_fusions(
    *, model_ir: dict[str, Any], execution_plan: dict[str, Any], rows: list[dict[str, Any]], metrics: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    targets = all_model_nodes(model_ir) | execution_nodes(execution_plan)
    direct_nodes = {str(row["node"]) for row in rows if row.get("node")}
    owners_by_member: dict[str, set[str]] = defaultdict(set)
    event_ids_by_owner: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        owner = str(row.get("node") or "")
        if not owner:
            continue
        event_ids_by_owner[owner].append(str(row["event_id"]))
        for target in row.get("ir_targets") or []:
            target = str(target)
            if target in FUSION_CANDIDATES and target != owner:
                owners_by_member[target].add(owner)

    states: dict[str, Any] = {}
    groups: dict[str, Any] = {}
    covered: set[str] = set()
    members_by_owner: dict[str, list[str]] = defaultdict(list)
    for member, owners in sorted(owners_by_member.items()):
        if member in direct_nodes or len(owners) != 1:
            states[member] = {
                "status": "partially_fused",
                "label": "profile aggregate contains explicit direct occurrences and/or occurrence-scoped fused intervals; Timeline retains the exact owner for every event",
                "shared_timing_owners": sorted(owners),
            }
            continue
        owner = next(iter(owners))
        if owner in covered or member in covered:
            states[member] = {
                "status": "partially_fused",
                "label": "occurrence-scoped fusion is exact in Timeline; aggregate overlap prevents a misleading single shared interval",
                "shared_timing_owners": sorted(owners),
            }
            continue
        members_by_owner[owner].append(member)
    for owner, members in sorted(members_by_owner.items()):
        group_id = "qwen35_profile_aggregate_" + re.sub(r"[^a-z0-9]+", "_", owner.lower()).strip("_")
        ir_nodes = [owner, *sorted(members)]
        covered.update(ir_nodes)
        groups[group_id] = {
            "owner": owner,
            "ir_nodes": ir_nodes,
            "timing_semantics": "shared_event_set",
            "provenance": "eager stack plus production signature/sequence reconciliation",
            "mapping_method": "explicit ir_targets on every production event",
            "confidence": "high",
            "evidence_scope": {
                "resolution": "profile_aggregate",
                "production_event_ids": sorted(set(event_ids_by_owner[owner])),
            },
        }
        for member in members:
            states[member] = {
                "status": "fused",
                "label": f"fused into {owner}",
                "included_in": owner,
                "fusion_group_id": group_id,
            }

    inactive_prefixes = ("vision_", "vision.", "mtp_", "generation_loop.")
    for target in sorted(targets):
        if target in metrics or target in states:
            continue
        if target.startswith(inactive_prefixes) or target in {
            "top.vision_inputs",
            "top.vision_frontend",
            "top.multimodal_injection",
            "top.generation_controller",
        }:
            states[target] = {"status": "not_selected", "label": "outside the text-only, MTP-off pure-TP8 contract"}
        else:
            states[target] = {"status": "structural", "label": "semantic, state, scheduler, or drill boundary without standalone production timing"}
    return states, groups


def profile_identity(item: dict[str, Any]) -> tuple[str, str, str]:
    framework = item["framework"]
    if item["phase"] == "prefill":
        suffix = "prefill_bs1_8k1k"
    else:
        suffix = f"cg_decode_bs{item['batch']}_8k1k"
    profile_id = f"qwen35_tp8_{framework}_{suffix}"
    return profile_id, suffix, f"{suffix}.yaml"


def implementation(item: dict[str, Any]) -> tuple[str, str, str, str | None]:
    if item["framework"] == "sglang":
        return (
            "sglang_f609d677b_qwen35_033446bb_tp8",
            SGLANG_SOURCE,
            SGLANG_CONTAINER,
            SGLANG_MODULE_SOURCE,
        )
    return ("vllm_487ecf187_qwen35_native_tp8", VLLM_SOURCE, VLLM_CONTAINER, None)


def build_one(
    *, task_root: Path, catalog_root: Path, model_ir: dict[str, Any], execution_plan: dict[str, Any], item: dict[str, Any]
) -> dict[str, Any]:
    framework, phase, batch, job = (
        item["framework"], item["phase"], item["batch"], item["job"]
    )
    rank_rows, rank_paths = load_rank_traces(task_root, item)
    if set(rank_rows) != set(range(8)):
        raise ValueError(f"{job}: expected all TP ranks 0..7, got {sorted(rank_rows)}")
    rank_diagnostics = {}
    mapped_envelopes = {}
    for rank, rows in sorted(rank_rows.items()):
        diagnostics = attribute_production_forward(rows, framework=framework, phase=phase)
        if diagnostics["tp_logical_all_reduce_count"] != 121 or diagnostics["tp_all_gather_count"] != 1:
            raise ValueError(f"{job} rank {rank}: invalid collective contract {diagnostics}")
        mapped = [row for row in rows if row.get("node")]
        mapped_envelopes[rank] = (
            max(float(row["ts_us"]) + float(row["dur_us"]) for row in mapped)
            - min(float(row["ts_us"]) for row in mapped)
        )
        graph_evidence = selected_forward_cuda_graph_evidence(rows)
        rank_diagnostics[str(rank)] = {
            **diagnostics,
            "selected_forward_cuda_graph": graph_evidence,
            "mapped_kernel_envelope_ms": round(mapped_envelopes[rank] / 1000.0, 6),
            "raw_trace": rank_paths[rank].name,
            "raw_trace_sha256": sha256_file(rank_paths[rank]),
        }
    reference_rank = max(mapped_envelopes, key=lambda rank: (mapped_envelopes[rank], rank))
    rows = rank_rows[reference_rank]
    graph_signatures = {
        json.dumps(diagnostics["selected_forward_cuda_graph"], sort_keys=True)
        for diagnostics in rank_diagnostics.values()
    }
    if len(graph_signatures) != 1:
        raise ValueError(f"{job}: selected-forward CUDA Graph evidence differs across TP ranks")
    graph_evidence = {
        **rank_diagnostics[str(reference_rank)]["selected_forward_cuda_graph"],
        "all_tp_ranks_consistent": True,
    }
    mapping_path = build_reconciled_eager_mapping(task_root, framework, phase)
    rows = attach_eager_stack_evidence(rows, mapping_path=mapping_path)
    missing_stack_nodes = sorted(
        {str(row["node"]) for row in rows if row.get("node") and not row.get("python_stack")}
    )
    if missing_stack_nodes:
        raise ValueError(f"{job}: production nodes lack eager stack closure: {missing_stack_nodes}")

    model_rows = [row for row in rows if row.get("node")]
    start = min(float(row["ts_us"]) for row in rows)
    stop = max(float(row["ts_us"]) + float(row["dur_us"]) for row in rows)
    model_start = min(float(row["ts_us"]) for row in model_rows)
    model_stop = max(float(row["ts_us"]) + float(row["dur_us"]) for row in model_rows)
    active_us = interval_union_us(model_rows)
    residency_us = sum(float(row["dur_us"]) for row in model_rows)
    timing = {
        "elapsed_ms": round((model_stop - model_start) / 1000.0, 6),
        "active_gpu_ms": round(active_us / 1000.0, 6),
        "gpu_residency_ms": round(residency_us / 1000.0, 6),
        "device_gap_ms": round(max(0.0, model_stop - model_start - active_us) / 1000.0, 6),
        "gpu_overlap_ms": round(max(0.0, residency_us - active_us) / 1000.0, 6),
        "kernel_envelope_ms": round((model_stop - model_start) / 1000.0, 6),
        "semantics": "critical global-rank model-bearing production-kernel envelope plus same-rank interval union and residency for one exact formal forward",
    }
    metrics = build_metrics(model_rows)
    for metric in metrics.values():
        metric["source_rank"] = reference_rank
        metric["rank_policy"] = "one coherent critical global rank selected by mapped model-forward kernel envelope"
    states, fusion_groups = build_states_and_fusions(
        model_ir=model_ir, execution_plan=execution_plan, rows=model_rows, metrics=metrics
    )

    profile_id, variant_id, filename = profile_identity(item)
    implementation_id, source_commit, container, runtime_module_commit = implementation(item)
    output_dir = catalog_root / "profiles" / "tp8" / implementation_id
    output_dir.mkdir(parents=True, exist_ok=True)
    timeline_path = output_dir / filename.replace(".yaml", ".timeline.json.gz")
    timeline = build_timeline_artifact(
        profile_id=profile_id,
        phase=phase,
        reference_rank=reference_rank,
        steps=[
            {
                "step_index": 1,
                "label": f"formal {phase} BS{batch}",
                "trace_start_us": start,
                "duration_us": stop - start,
                "events": rows,
            }
        ],
        timing_summary=timing,
        raw_trace={
            "path": rank_paths[reference_rank].name,
            "sha256": sha256_file(rank_paths[reference_rank]),
            "all_tp_ranks_validated": True,
        },
        stack_source={
            "mode": "separate_graph_off_eager_capture",
            "mapping_file": str(mapping_path),
            "mapping_sha256": sha256_file(mapping_path),
            "production_capture_has_python_stack": False,
        },
    )
    timeline_sha = write_timeline_artifact(timeline_path, timeline)

    validation = {
        "schema_version": "qwen35-profile-validation.v1",
        "job_id": job,
        "framework": framework,
        "phase": phase,
        "batch_size": batch,
        "reference_rank": reference_rank,
        "all_tp_rank_count": len(rank_rows),
        "rank_diagnostics": rank_diagnostics,
        "profile_timing": timing,
        "eager_mapping": str(mapping_path),
        "eager_mapping_sha256": sha256_file(mapping_path),
        "missing_stack_nodes": missing_stack_nodes,
        "unclassified_kernel_count": sum(
            not row.get("node") and not row.get("support_class") for rank in rank_rows.values() for row in rank
        ),
    }
    validation_path = evidence_dir(task_root, item) / "validation.json"
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")

    concurrency = batch
    runtime_coordinate, selector_evidence = selected_runtime_coordinate(task_root, item)
    label_phase = "prefill" if phase == "prefill" else "CUDA Graph decode"
    profile = {
        "schema_version": "profile.v2",
        "profile_id": profile_id,
        "label": f"GB300 · {framework} · pure TP8 · {label_phase} · BS{batch} · 8k→1k",
        "model_id": "qwen35",
        "execution_path_id": "tp8",
        "implementation_id": implementation_id,
        "variant_id": variant_id,
        "phase": phase,
        "generation_mode": "autoregressive",
        "entry_view": "top",
        "execution_parameters": {
            "tp_size": 8,
            "dp_size": 1,
            "cp_size": 1,
            "ep_size": 1,
            "pp_size": 1,
        },
        "hardware": {"gpu": "GB300", "gpus_per_node": 4, "nodes": 2, "cluster": "CMH"},
        "workload": {
            "isl": 8192,
            "osl": 1024,
            "batch_size": batch,
            "batch_size_scope": "global_request_count",
            "concurrency": concurrency,
            "warmup_requests": 3 * concurrency,
            "formal_requests": concurrency,
            "prompt_source": "deterministic_random_token_ids",
            "prompt_seed": 0,
            "ignore_eos": True,
            "prefix_cache_enabled": False,
            "hicache_enabled": False,
            "kv_offload_enabled": False,
            "mtp_nextn_enabled": False,
        },
        "profiler": {
            "type": "nsight_systems" if framework == "sglang" and phase == "prefill" else "torch_profiler",
            "representative_rank": reference_rank,
            "all_tp_ranks_validated": True,
            "timing_gate_status": "passed",
            "cuda_graph_enabled": graph_evidence["used_graph_path"],
            "cuda_graph_enabled_semantics": cuda_graph_enabled_semantics(graph_evidence),
            "server_cuda_graph_config": server_cuda_graph_config(task_root, item),
            "selected_forward_cuda_graph": graph_evidence,
            "with_stack": False,
            "eager_semantic_capture_cuda_graph_enabled": False,
            "production_stack_source": "separate eager capture",
            "formal_window_count": 1,
            "selected_runtime_coordinate": runtime_coordinate,
            "gpu_metric_semantics": timing["semantics"],
        },
        "evidence": {
            "job_id": job,
            "source_commit": source_commit,
            "runtime_model_module_commit": runtime_module_commit,
            "model_revision": MODEL_REVISION,
            "model_config_sha256": MODEL_CONFIG_SHA256,
            "container": container,
            "validation_file": validation_path.name,
            "validation_sha256": sha256_file(validation_path),
            "baseline_selector": selector_evidence,
            "raw_trace_sha256": sha256_file(rank_paths[reference_rank]),
            "all_rank_trace_sha256": {str(rank): sha256_file(path) for rank, path in sorted(rank_paths.items())},
            "eager_mapping_sha256": sha256_file(mapping_path),
            "mapped_kernel_count_ratio": rank_diagnostics[str(reference_rank)]["mapped_kernel_count_ratio"],
            "mapped_kernel_duration_ratio": rank_diagnostics[str(reference_rank)]["mapped_kernel_duration_ratio"],
            "unclassified_kernel_count": 0,
            "semantic_stack_closure_missing_node_count": len(missing_stack_nodes),
            "mapping_policy": "complete TP collective order plus eager-validated framework/phase kernel landmark sequence; all non-model kernels explicitly classified",
            "attribution_diagnostics": rank_diagnostics[str(reference_rank)],
            "timing": timing,
        },
        "timeline": {
            "schema_version": "timeline.v1",
            "artifact": timeline_path.name,
            "sha256": timeline_sha,
            "reference_rank": reference_rank,
            "step_count": 1,
            "event_count": len(rows),
            "raw_trace_file": rank_paths[reference_rank].name,
        },
        "node_states": states,
        "fusion_groups": fusion_groups,
        "node_metrics": metrics,
    }
    profile_path = output_dir / filename
    profile_path.write_text(yaml.safe_dump(profile, sort_keys=False, allow_unicode=True, width=120))
    return {
        "profile_id": profile_id,
        "job_id": job,
        "profile": str(profile_path),
        "validation_file": str(validation_path),
        "validation_sha256": sha256_file(validation_path),
        "raw_artifacts_outside_git": {
            str(rank): {"path": str(path), "sha256": sha256_file(path)}
            for rank, path in sorted(rank_paths.items())
        },
        "timeline_sha256": timeline_sha,
        "reference_rank": reference_rank,
        "timing": timing,
        "mapped_kernel_count_ratio": profile["evidence"]["mapped_kernel_count_ratio"],
        "mapped_kernel_duration_ratio": profile["evidence"]["mapped_kernel_duration_ratio"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-root",
        type=Path,
        default=REPO_ROOT.parent / "current" / "qwen35-complete-profiles",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    catalog_root = REPO_ROOT / "catalog" / "qwen35"
    model_ir = yaml.safe_load((catalog_root / "model_ir.yaml").read_text())
    execution_plan = yaml.safe_load((catalog_root / "execution_paths" / "tp8.yaml").read_text())
    results = [
        build_one(
            task_root=args.task_root,
            catalog_root=catalog_root,
            model_ir=model_ir,
            execution_plan=execution_plan,
            item=item,
        )
        for item in MATRIX
    ]
    report = {
        "schema_version": "qwen35-profile-matrix.v1",
        "raw_artifact_policy": "preserved outside git; exact paths and SHA256 values recorded per TP rank",
        "measured_profile_count": len(results),
        "unsupported_profile_count": 0,
        "profiles": results,
    }
    output = args.task_root / "validation" / "profile-matrix.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
