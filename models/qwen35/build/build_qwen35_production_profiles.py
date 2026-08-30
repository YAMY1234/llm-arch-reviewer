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
import statistics
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
    is_all_reduce,
)


MODEL_REVISION = "8f590eae8f10bf55d9a46f79ea0280bde435c9f8"
MODEL_CONFIG_SHA256 = "9408a9e559cc2f05f0b357738213666353e6651160ce8ff477b1c26982bc4f63"
SGLANG_SOURCE = "f609d677b909ca46c64bb6803b69a85fedbf86bc"
SGLANG_MODULE_SOURCE = "033446bb05f35c0943aed2750c443077ffc0b92c"
VLLM_SOURCE = "487ecf187d3dfe74d2cf6119a92881dba403c219"
SGLANG_CONTAINER = "sglang-glm53-flash-arm64-73f9294b.sqsh@sha256:28e9545e312e344bbbf80c575b928be53c9aba6296ae55f292ce0f10750c6971"
VLLM_CONTAINER = "vllm-glm53-flash-arm64-905c0293.sqsh@sha256:efdfe25952dc672d4415032e2755df7d7f2bab549992a2e3f2c429334f366756"
SGLANG_PROFILER_SYNC_SHA256 = "f38a27dbe1d876cf3c9d13b2a6f5c52fa486e4424e657c8b0beaf95c75a2f61c"


MATRIX = (
    {"framework": "sglang", "phase": "prefill", "batch": 1, "job": "3414663"},
    {"framework": "sglang", "phase": "decode", "batch": 1, "job": "3427173"},
    {"framework": "sglang", "phase": "decode", "batch": 16, "job": "3427500"},
    {"framework": "sglang", "phase": "decode", "batch": 64, "job": "3427499"},
    {"framework": "sglang", "phase": "decode", "batch": 256, "job": "3427851"},
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
                "baseline_elapsed_ms": selected["baseline_elapsed_ms"],
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


def profiler_off_wall_ms(item: dict[str, Any], coordinate: dict[str, Any]) -> float:
    """Return the selected profiler-off production forward wall authority."""

    if item["framework"] == "sglang":
        value = coordinate.get("baseline_mean_elapsed_ms")
    elif item["phase"] == "prefill":
        value = coordinate.get("baseline_elapsed_ms")
    else:
        value = coordinate.get("elapsed_ms")
    if value is None or float(value) <= 0:
        raise ValueError(
            f"{item['job']}: selected profiler-off wall authority is missing or nonpositive"
        )
    return float(value)


def wall_trace_contract_gate(
    item: dict[str, Any],
    coordinate: dict[str, Any],
    *,
    serving_wall_ms: float,
    active_gpu_ms: float,
    kernel_envelope_ms: float,
) -> dict[str, Any]:
    """Reject an unexplained profiler-off wall / trace-forward mismatch.

    A large wall interval is not automatically model time.  In particular,
    the SGLang C=1 selector used the first formal prefill scheduler interval
    (profile step zero).  That 4111.995663 ms interval is about 44x the
    instrumented model envelope and the retained production client shows a
    35.46 s request after profiler activation.  There is no evidence that the
    baseline interval and the instrumented interval bound the same isolated
    forward, so this point must remain unsupported rather than publishing the
    interval as serving/model wall authority.

    Other points are not accepted merely because they pass this local check;
    exact semantic and fusion closure are independent mandatory gates.
    """

    wall_to_envelope_ratio = serving_wall_ms / kernel_envelope_ms
    envelope_to_wall_ratio = kernel_envelope_ms / serving_wall_ms
    first_formal_scheduler_interval = (
        item["framework"] == "sglang"
        and item["phase"] == "prefill"
        and int(coordinate.get("profile_start_step", -1)) == 0
    )
    unexplained = first_formal_scheduler_interval and wall_to_envelope_ratio > 4.0
    return {
        "state": "failed" if unexplained else "passed",
        "serving_wall_ms": round(serving_wall_ms, 6),
        "instrumented_active_gpu_ms": round(active_gpu_ms, 6),
        "instrumented_kernel_envelope_ms": round(kernel_envelope_ms, 6),
        "wall_to_envelope_ratio": round(wall_to_envelope_ratio, 6),
        "envelope_to_wall_ratio": round(envelope_to_wall_ratio, 6),
        "same_isolated_forward_proven": not unexplained,
        "first_formal_scheduler_interval": first_formal_scheduler_interval,
        "reason": (
            "the profiler-off selector is the first formal prefill scheduler "
            "interval and is 44x the instrumented model envelope; retained "
            "request/profiler evidence does not isolate the same forward"
            if unexplained
            else "no unexplained first-formal scheduler-wall mismatch was detected"
        ),
    }


def profile_acceptance_gate(
    *,
    rank_diagnostics: dict[str, dict[str, Any]],
    typed_unresolved_event_count: int,
    node_states: dict[str, dict[str, Any]],
    fusion_groups: dict[str, dict[str, Any]],
    wall_trace_gate: dict[str, Any],
) -> dict[str, Any]:
    """Return the fail-closed release decision for one measured candidate."""

    reasons: list[dict[str, Any]] = []
    per_rank_unresolved = {
        rank: int(diagnostics["semantic_reconciliation"]["typed_unresolved_event_count"])
        for rank, diagnostics in sorted(rank_diagnostics.items())
    }
    if typed_unresolved_event_count or any(per_rank_unresolved.values()):
        reasons.append(
            {
                "code": "semantic_reconciliation_incomplete",
                "reference_rank_typed_unresolved_event_count": typed_unresolved_event_count,
                "per_rank_typed_unresolved_event_count": per_rank_unresolved,
                "policy": (
                    "an accepted profile requires zero model-bearing production "
                    "events without exact same-rank, same-phase, occurrence-scoped closure"
                ),
            }
        )

    partial_states = {
        node: state
        for node, state in node_states.items()
        if state.get("status") == "partially_fused"
    }
    incomplete_owner_nodes = sorted(
        node
        for node, state in partial_states.items()
        if state.get("all_owner_events_same_rank_closed") is False
    )
    if partial_states:
        reasons.append(
            {
                "code": "fusion_reconciliation_incomplete",
                "partial_fusion_node_count": len(partial_states),
                "incomplete_owner_closure_node_count": len(incomplete_owner_nodes),
                "incomplete_owner_closure_nodes": incomplete_owner_nodes,
                "policy": (
                    "partial occurrence evidence is retained, but it is not a "
                    "deliverable profile-aggregate fused attribution"
                ),
            }
        )

    invalid_full_groups = sorted(
        group_id
        for group_id, group in fusion_groups.items()
        if group.get("evidence_scope", {}).get("member_event_sets_equal_owner") is not True
        or group.get("evidence_scope", {}).get("all_owner_events_same_rank_closed") is not True
    )
    if invalid_full_groups:
        reasons.append(
            {
                "code": "invalid_complete_fusion_claim",
                "fusion_group_ids": invalid_full_groups,
            }
        )

    if wall_trace_gate.get("state") != "passed":
        reasons.append(
            {
                "code": "wall_trace_contract_mismatch",
                "gate": wall_trace_gate,
                "policy": (
                    "an unexplained profiler-off wall / instrumented forward "
                    "mismatch cannot be published as measured timing"
                ),
            }
        )

    return {
        "state": "accepted" if not reasons else "unsupported",
        "fail_closed": True,
        "reason_count": len(reasons),
        "reasons": reasons,
    }


def rank_collective_duration_gate(
    rank_rows: dict[int, list[dict[str, Any]]], *, job: str, serving_wall_ms: float
) -> dict[str, Any]:
    """Reject rank-skewed profiler captures before they become timing evidence."""

    per_rank: dict[str, dict[str, float | int]] = {}
    signatures_by_rank: dict[
        int, dict[tuple[str, str, str, int], float]
    ] = {}
    totals: list[float] = []
    max_singles: list[float] = []
    mapped_envelopes: list[float] = []
    for rank, rows in sorted(rank_rows.items()):
        physical_collectives = [row for row in rows if is_all_reduce(row)]
        # One-shot paths have one physical kernel per logical all-reduce.  The
        # two-shot path additionally emits a fused rmsNormLamport companion for
        # 119 of the 121 logical occurrences.  Count the primary all-reduce
        # kernels for the portable contract, while retaining every physical
        # kernel in the residency/outlier envelope.
        logical_collectives = [
            row
            for row in physical_collectives
            if "rmsNormLamport" not in str(row["kernel_name"])
        ]
        durations_ms = [
            float(row["dur_us"]) / 1000.0 for row in physical_collectives
        ]
        if len(logical_collectives) != 121:
            raise ValueError(
                f"{job} rank {rank}: collective-duration gate expected 121 logical "
                f"all-reduce primaries, got {len(logical_collectives)} "
                f"({len(physical_collectives)} physical kernels)"
            )
        total_ms = interval_union_us(physical_collectives) / 1000.0
        totals.append(total_ms)
        max_singles.append(max(durations_ms))
        mapped = [row for row in rows if row.get("node")]
        mapped_envelope_ms = (
            max(float(row["ts_us"]) + float(row["dur_us"]) for row in mapped)
            - min(float(row["ts_us"]) for row in mapped)
        ) / 1000.0
        mapped_envelopes.append(mapped_envelope_ms)
        per_rank[str(rank)] = {
            "logical_all_reduce_count": len(logical_collectives),
            "physical_all_reduce_kernel_count": len(physical_collectives),
            "total_all_reduce_residency_ms": round(total_ms, 6),
            "max_single_all_reduce_ms": round(max(durations_ms), 6),
            "mapped_kernel_envelope_ms": round(mapped_envelope_ms, 6),
        }
        ordinals: Counter[tuple[str, str, str]] = Counter()
        signatures: dict[tuple[str, str, str, int], float] = {}
        for row in physical_collectives:
            base = (
                str(row.get("node") or ""),
                str(row.get("occurrence_id") or "top"),
                str(row.get("kernel_name") or ""),
            )
            ordinal = ordinals[base]
            ordinals[base] += 1
            signatures[(*base, ordinal)] = float(row["dur_us"])
        signatures_by_rank[rank] = signatures
    median_total = statistics.median(totals)
    max_single = max(max_singles)
    median_max_single = statistics.median(max_singles)
    max_total = max(totals)
    min_total = min(totals)
    max_single_limit_ms = min(
        serving_wall_ms * 0.25,
        median_max_single * 4.0 + 0.1,
    )
    signature_spread_limit_us = max(250.0, serving_wall_ms * 100.0)
    mapped_envelope_upper_ms = serving_wall_ms * 1.25
    total_upper_ms = median_total * 1.5 + 0.1
    total_lower_ms = max(0.0, median_total / 1.5 - 0.1)
    failures = []
    signature_outliers: list[dict[str, Any]] = []
    worst_signature: dict[str, Any] | None = None
    all_signatures = set().union(
        *(set(signatures) for signatures in signatures_by_rank.values())
    )
    for signature in sorted(all_signatures):
        rank_durations = {
            str(rank): signatures[signature]
            for rank, signatures in sorted(signatures_by_rank.items())
            if signature in signatures
        }
        if set(rank_durations) != {str(rank) for rank in range(8)}:
            failures.append(
                f"collective signature is not present on all 8 ranks: {signature}"
            )
            continue
        minimum_us = min(rank_durations.values())
        maximum_us = max(rank_durations.values())
        ratio = maximum_us / max(minimum_us, 1e-9)
        signature_report = {
            "node": signature[0],
            "occurrence_id": signature[1],
            "kernel_name": signature[2],
            "ordinal": signature[3],
            "rank_duration_us": rank_durations,
            "min_us": minimum_us,
            "max_us": maximum_us,
            "max_to_min_ratio": ratio,
        }
        if (
            worst_signature is None
            or ratio > worst_signature["max_to_min_ratio"]
        ):
            worst_signature = signature_report
        if (
            maximum_us / 1000.0 > max_single_limit_ms
            and maximum_us - minimum_us > signature_spread_limit_us
            and ratio > 8.0
        ):
            signature_outliers.append(signature_report)
            failures.append(
                "collective rank-duration activation skew: "
                f"{signature[0]} {signature[1]} ordinal={signature[3]} "
                f"min={minimum_us:.3f}us max={maximum_us:.3f}us ratio={ratio:.3f}"
            )
    if max_single > max_single_limit_ms:
        failures.append(
            f"max single all-reduce {max_single:.6f} ms exceeds {max_single_limit_ms:.6f} ms"
        )
    if max_total > total_upper_ms:
        failures.append(
            f"max rank collective residency {max_total:.6f} ms exceeds robust upper {total_upper_ms:.6f} ms"
        )
    if min_total < total_lower_ms:
        failures.append(
            f"min rank collective residency {min_total:.6f} ms is below robust lower {total_lower_ms:.6f} ms"
        )
    if max(mapped_envelopes) > mapped_envelope_upper_ms:
        failures.append(
            f"max mapped rank envelope {max(mapped_envelopes):.6f} ms exceeds "
            f"profiler-off serving-wall bound {mapped_envelope_upper_ms:.6f} ms"
        )
    report = {
        "state": "failed" if failures else "passed",
        "policy": "all 8 TP ranks; 121 logical all-reduce primaries; every one-shot/two-shot physical kernel included; exact node/occurrence/kernel/ordinal signatures must exist on every rank; reject signature/max-single outliers relative to the selected profiler-off serving wall and the rank median, reject any mapped rank envelope above 125% of serving wall, and retain robust union-residency outlier bounds",
        "serving_wall_authority_ms": serving_wall_ms,
        "signature_count": len(all_signatures),
        "signature_outlier_count": len(signature_outliers),
        "worst_signature": worst_signature,
        "max_single_limit_ms": max_single_limit_ms,
        "signature_spread_limit_us": signature_spread_limit_us,
        "mapped_envelope_upper_ms": mapped_envelope_upper_ms,
        "median_rank_max_single_all_reduce_ms": round(median_max_single, 6),
        "median_total_all_reduce_residency_ms": round(median_total, 6),
        "total_residency_lower_ms": round(total_lower_ms, 6),
        "total_residency_upper_ms": round(total_upper_ms, 6),
        "per_rank": per_rank,
        "failures": failures,
    }
    if failures:
        raise ValueError(f"{job}: rank collective-duration/outlier gate failed: {failures}")
    return report


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


def sglang_profiler_sync_evidence(
    root: Path,
    item: dict[str, Any],
    rank_rows: dict[int, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    if item["framework"] != "sglang" or item["phase"] != "decode":
        return None
    patch_path = root / "overlays" / "sglang-profiler-sync" / "sitecustomize.py"
    source_lock_path = patch_path.with_name("source-lock.json")
    if not patch_path.is_file() or sha256_file(patch_path) != SGLANG_PROFILER_SYNC_SHA256:
        raise ValueError(
            f"{item['job']}: SGLang profiler synchronization overlay SHA mismatch"
        )
    if not source_lock_path.is_file():
        raise ValueError(f"{item['job']}: SGLang profiler source lock is missing")
    source_lock = json.loads(source_lock_path.read_text())
    if (
        source_lock.get("overlay_sha256") != SGLANG_PROFILER_SYNC_SHA256
        or source_lock.get("base_sglang_package_commit") != SGLANG_SOURCE
        or source_lock.get("base_source_files")
        != {
            "python/sglang/srt/managers/scheduler.py": "02eaf6c4db24e400d98a19cc1b7e44d7346389447cc05723e4a1678356341d0a",
            "python/sglang/srt/managers/scheduler_components/profiler_manager.py": "8e4a29923065c2c674151e94d1d4355ee295f1fca0104d36d5de70e5f7e22fe2",
        }
        or source_lock.get("measured_interval_cuda_work_added") is not False
        or source_lock.get("pre_forward_device_collective_added") is not False
    ):
        raise ValueError(f"{item['job']}: invalid SGLang profiler source lock")
    directory = evidence_dir(root, item)
    server_logs = sorted(directory.glob("*_agg_w0.out"))
    text = "\n".join(path.read_text(errors="replace") for path in server_logs)
    marker_counts: dict[str, dict[str, int]] = {}
    for marker in (
        "pre_activation_barrier",
        "post_activation_barrier",
        "activation_complete",
        "pre_input_preparation_barrier",
        "input_preparation_barrier_passed",
    ):
        counts = {
            str(rank): text.count(
                f"QWEN_TP_PROFILER_SYNC {marker} tp_rank={rank}"
            )
            for rank in range(8)
        }
        if any(count != 1 for count in counts.values()):
            raise ValueError(
                f"{item['job']}: incomplete {marker} markers across TP ranks: {counts}"
            )
        marker_counts[marker] = counts
    return {
        "state": "passed",
        "mechanism": "accepted dual-TP CPU-barrier pattern: per-rank CUDA backlog drain plus Gloo TP barrier after Kineto activation, then another per-rank CUDA drain plus Gloo TP barrier at the scheduler resolve_forward_inputs boundary immediately before model_worker.forward_batch_generation; no CUDA work is added to the selected model interval",
        "overlay_file": patch_path.name,
        "overlay_sha256": SGLANG_PROFILER_SYNC_SHA256,
        "source_lock_file": source_lock_path.name,
        "source_lock_sha256": sha256_file(source_lock_path),
        "marker_counts": marker_counts,
        "pre_forward_device_collective_added": False,
        "scheduler_fence_location": source_lock["scheduler_fence_location"],
        "all_tp_rank_count": len(rank_rows),
        "server_log_sha256": {
            path.name: sha256_file(path) for path in server_logs
        },
    }


def build_reconciled_eager_mapping(
    root: Path, framework: str, phase: str, rank: int
) -> tuple[Path, dict[str, Any]]:
    source_name = (
        f"sglang-forward_{'extend' if phase == 'prefill' else 'decode'}"
        if framework == "sglang"
        else f"vllm-vllm_{phase}"
    )
    source_phase = (
        f"forward_{'extend' if phase == 'prefill' else 'decode'}"
        if framework == "sglang"
        else f"vllm_{phase}"
    )
    source_dir = root / "mapping" / source_name
    manifest_path = source_dir / f"input_manifest.tp{rank}.json"
    events_source_path = source_dir / f"events.tp{rank}.jsonl"
    mapping_source_path = source_dir / f"kernel_mapping.tp{rank}.jsonl"
    for path in (manifest_path, events_source_path, mapping_source_path):
        if not path.is_file():
            raise ValueError(
                f"{framework}/{phase}/TP{rank}: missing rank-specific eager evidence {path}"
            )
    manifest = json.loads(manifest_path.read_text())
    expected_source_commit = (
        SGLANG_MODULE_SOURCE if framework == "sglang" else VLLM_SOURCE
    )
    exact_contract = {
        "rank": rank,
        "phase": source_phase,
        "source_commit": expected_source_commit,
    }
    for field, expected in exact_contract.items():
        if manifest.get(field) != expected:
            raise ValueError(
                f"{framework}/{phase}/TP{rank}: eager manifest {field}="
                f"{manifest.get(field)!r}, expected {expected!r}"
            )
    raw_trace_path = Path(str(manifest.get("trace_path") or ""))
    if not raw_trace_path.is_file():
        raise ValueError(
            f"{framework}/{phase}/TP{rank}: raw eager trace is missing: {raw_trace_path}"
        )
    if manifest.get("trace_sha256") != sha256_file(raw_trace_path):
        raise ValueError(
            f"{framework}/{phase}/TP{rank}: raw eager trace SHA-256 mismatch"
        )
    events = [
        json.loads(line)
        for line in events_source_path.read_text().splitlines()
        if line.strip()
    ]
    expected_kernel_count = int(manifest.get("selected_forward_kernel_count") or -1)
    if len(events) != expected_kernel_count:
        raise ValueError(
            f"{framework}/{phase}/TP{rank}: selected eager event count {len(events)} "
            f"!= manifest {expected_kernel_count}"
        )
    selected_events_sha256 = hashlib.sha256(
        json.dumps(events, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if manifest.get("selected_forward_events_sha256") != selected_events_sha256:
        raise ValueError(
            f"{framework}/{phase}/TP{rank}: selected eager event identity SHA-256 mismatch"
        )
    duration_us = sum(float(event.get("dur_us") or 0.0) for event in events)
    if abs(duration_us - float(manifest["selected_forward_kernel_duration_us"])) > 1e-3:
        raise ValueError(
            f"{framework}/{phase}/TP{rank}: selected eager duration mismatch"
        )
    source_mappings = [
        json.loads(line)
        for line in mapping_source_path.read_text().splitlines()
        if line.strip()
    ]
    source_mapping_by_event = {
        str(mapping["event_id"]): mapping for mapping in source_mappings
    }
    rows = [dict(event) for event in events]
    attribute_production_forward(rows, framework=framework, phase=phase)
    output = root / "mapping" / f"reconciled-{framework}-{phase}"
    output.mkdir(parents=True, exist_ok=True)
    events_path = output / f"events.tp{rank}.jsonl"
    mapping_path = output / f"kernel_mapping.tp{rank}.jsonl"
    events_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    mappings = []
    for sequence_index, row in enumerate(rows):
        semantic_node = row.get("node")
        if not semantic_node:
            continue
        source_mapping = source_mapping_by_event.get(str(row["event_id"]))
        if not source_mapping or source_mapping.get("selected_node") != semantic_node:
            # Ordered collective segmentation is useful portable evidence, but
            # it is not a Python-stack closure for this physical kernel.  Do
            # not manufacture a mapping entry that a later stage could mistake
            # for high-confidence semantic ownership.
            continue
        mappings.append(
            {
                "event_id": row["event_id"],
                "kernel_name": row["kernel_name"],
                "selected_node": semantic_node,
                "confidence": source_mapping.get("confidence") or "support",
                "rank": rank,
                "phase": phase,
                "source_phase": source_phase,
                "occurrence_id": row.get("occurrence_id"),
                "sequence_index": sequence_index,
                "closure_status": "closed",
                "evidence": [
                    "same_rank_eager_python_stack",
                    "phase_specific_raw_trace_identity",
                    "exact_kernel_signature",
                    "occurrence_scoped_ordered_sequence",
                ],
            }
        )
    mapping_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in mappings))
    report = {
        "framework": framework,
        "phase": phase,
        "source_phase": source_phase,
        "rank": rank,
        "kernel_count": len(rows),
        "semantic_kernel_count": sum(bool(row.get("node")) for row in rows),
        "support_kernel_count": sum(bool(row.get("support_class")) for row in rows),
        "closed_stack_event_count": len({row["event_id"] for row in mappings}),
        "node_coverage": sorted({row["node"] for row in rows if row.get("node")}),
        "closed_node_coverage": sorted({row["selected_node"] for row in mappings}),
        "raw_trace_path": str(raw_trace_path),
        "raw_trace_sha256": manifest["trace_sha256"],
        "raw_manifest_path": str(manifest_path),
        "raw_manifest_sha256": sha256_file(manifest_path),
        "source_events_sha256": sha256_file(events_source_path),
        "selected_forward_events_sha256": selected_events_sha256,
        "source_mapping_sha256": sha256_file(mapping_source_path),
        "mapping_sha256": sha256_file(mapping_path),
        "events_sha256": sha256_file(events_path),
    }
    report_path = output / f"reconciliation.tp{rank}.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if rank == 0:
        (output / "reconciliation.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
    return mapping_path, report


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
    metrics = {
        node: metric_for_rows(
            events,
            status=(
                "measured_direct"
                if all(event.get("reconciliation_status") == "closed" for event in events)
                else "typed_unresolved"
            ),
        )
        for node, events in direct.items()
    }
    for target, events in targeted.items():
        if target in direct:
            # A reusable semantic leaf may be standalone for one occurrence
            # and fused for others.  Keep one interval-union aggregate and say
            # so explicitly instead of copying an owner's scalar duration.
            event_ids = {event["event_id"] for event in direct[target]}
            event_ids.update(event["event_id"] for event in events)
            combined = [event for event in rows if event["event_id"] in event_ids]
            if len(combined) != len(direct[target]):
                metrics[target] = metric_for_rows(
                    combined,
                    status=(
                        "inclusive_rollup"
                        if all(
                            event.get("reconciliation_status") == "closed"
                            for event in combined
                        )
                        else "typed_unresolved"
                    ),
                )
                metrics[target]["metric_kind"] = "inclusive_rollup"
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


def fusion_target_is_physically_proven(
    row: dict[str, Any], owner: str, member: str
) -> bool:
    """Return whether this physical event proves the owner's fused member.

    Equality of event IDs is necessary but not sufficient: the kernel family
    must also encode the member's operation/state access.  Residual/norm and
    embedding relationships inferred only from sequence order intentionally do
    not pass this gate.
    """

    name = str(row.get("kernel_name") or "").lower()
    if (
        owner == "full_attention.causal_gqa"
        and member == "full_attention.kv_state_read"
    ):
        return any(token in name for token in ("fmha", "attention"))
    if (
        owner == "full_attention.qk_norm"
        and member == "full_attention.partial_rope"
    ):
        return "fused_qk" in name and "rope" in name
    if (
        owner == "gdn_attention.qkvz_projection"
        and member == "gdn_attention.ba_projection"
    ):
        return "fused_qkvzba_split" in name
    if (
        owner == "gdn_attention.causal_conv"
        and member == "gdn_attention.conv_state_read"
    ):
        return "causal_conv1d" in name
    if owner == "gdn_attention.gated_delta_recurrence" and member in {
        "gdn_attention.recurrent_state_read",
        "gdn_attention.state_write",
    }:
        return any(
            token in name
            for token in (
                "gated_delta_rule",
                "gdn_decode",
                "gdn_wide_vec",
                "chunk_gated_delta",
                "recompute_w_u",
                "chunk_fwd_kernel_o",
            )
        )
    return False


def build_states_and_fusions(
    *, model_ir: dict[str, Any], execution_plan: dict[str, Any], rows: list[dict[str, Any]], metrics: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    targets = all_model_nodes(model_ir) | execution_nodes(execution_plan)
    direct_nodes = {str(row["node"]) for row in rows if row.get("node")}
    owners_by_member: dict[str, set[str]] = defaultdict(set)
    event_ids_by_owner: dict[str, set[str]] = defaultdict(set)
    event_ids_by_member_owner: dict[tuple[str, str], set[str]] = defaultdict(set)
    rows_by_owner: dict[str, list[dict[str, Any]]] = defaultdict(list)
    closed_event_ids: set[str] = {
        str(row["event_id"])
        for row in rows
        if row.get("reconciliation_status") == "closed"
    }
    for row in rows:
        owner = str(row.get("node") or "")
        if not owner:
            continue
        event_id = str(row["event_id"])
        event_ids_by_owner[owner].add(event_id)
        rows_by_owner[owner].append(row)
        for target in row.get("ir_targets") or []:
            target = str(target)
            if target in FUSION_CANDIDATES and target != owner:
                owners_by_member[target].add(owner)
                event_ids_by_member_owner[(target, owner)].add(event_id)

    states: dict[str, Any] = {}
    groups: dict[str, Any] = {}
    covered: set[str] = set()
    members_by_owner: dict[str, list[str]] = defaultdict(list)
    for member, owners in sorted(owners_by_member.items()):
        equality_failures = []
        for owner in sorted(owners):
            owner_events = event_ids_by_owner[owner]
            member_events = event_ids_by_member_owner[(member, owner)]
            if member_events != owner_events:
                equality_failures.append(
                    {
                        "owner": owner,
                        "owner_event_count": len(owner_events),
                        "member_target_event_count": len(member_events),
                        "missing_owner_event_count": len(owner_events - member_events),
                        "extra_member_event_count": len(member_events - owner_events),
                    }
                )
        owner = next(iter(owners)) if len(owners) == 1 else None
        owner_events_closed = bool(owner) and event_ids_by_owner[owner] <= closed_event_ids
        physical_target_proof = bool(owner) and all(
            fusion_target_is_physically_proven(row, owner, member)
            for row in rows_by_owner[owner]
        )
        if (
            member in direct_nodes
            or len(owners) != 1
            or equality_failures
            or not owner_events_closed
            or not physical_target_proof
        ):
            states[member] = {
                "status": "partially_fused",
                "label": (
                    "occurrence-scoped partial fusion only; no profile-aggregate "
                    "shared ownership is published because direct occurrences, "
                    "multiple owners, event-set inequality, or incomplete same-rank "
                    "eager closure prevents an exact aggregate"
                ),
                "shared_timing_owners": sorted(owners),
                "event_set_equality_failures": equality_failures,
                "all_owner_events_same_rank_closed": owner_events_closed,
                "physical_target_proof": physical_target_proof,
            }
            continue
        assert owner is not None
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
                "production_event_ids": sorted(event_ids_by_owner[owner]),
                "member_event_sets_equal_owner": True,
                "all_owner_events_same_rank_closed": True,
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
    runtime_coordinate, selector_evidence = selected_runtime_coordinate(
        task_root, item
    )
    serving_wall_ms = profiler_off_wall_ms(item, runtime_coordinate)
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
    collective_duration_gate = None
    if framework == "sglang" and phase == "decode":
        collective_duration_gate = rank_collective_duration_gate(
            rank_rows, job=job, serving_wall_ms=serving_wall_ms
        )
    profiler_sync_evidence = sglang_profiler_sync_evidence(
        task_root, item, rank_rows
    )
    median_envelope = statistics.median(mapped_envelopes.values())
    reference_rank = min(
        mapped_envelopes,
        key=lambda rank: (abs(mapped_envelopes[rank] - median_envelope), rank),
    )
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
    mapping_paths: dict[int, Path] = {}
    reconciliation_reports: dict[str, dict[str, Any]] = {}
    for rank in sorted(rank_rows):
        mapping_path, reconciliation_report = build_reconciled_eager_mapping(
            task_root, framework, phase, rank
        )
        mapping_paths[rank] = mapping_path
        reconciliation_reports[str(rank)] = reconciliation_report
        rank_rows[rank] = attach_eager_stack_evidence(
            rank_rows[rank],
            mapping_path=mapping_path,
            expected_rank=rank,
            expected_phase=phase,
        )
        semantic_rows = [row for row in rank_rows[rank] if row.get("node")]
        closed_rows = [
            row
            for row in semantic_rows
            if row.get("reconciliation_status") == "closed"
        ]
        if not closed_rows:
            raise ValueError(
                f"{job} rank {rank}: no same-rank phase-specific semantic stack closure"
            )
        rank_diagnostics[str(rank)]["semantic_reconciliation"] = {
            "closed_event_count": len(closed_rows),
            "typed_unresolved_event_count": len(semantic_rows) - len(closed_rows),
            "closed_node_count": len({str(row["node"]) for row in closed_rows}),
            "typed_unresolved_nodes": sorted(
                {
                    str(row["node"])
                    for row in semantic_rows
                    if row.get("reconciliation_status") != "closed"
                }
            ),
            "mapping_sha256": sha256_file(mapping_path),
            "raw_eager_trace_sha256": reconciliation_report["raw_trace_sha256"],
            "raw_eager_manifest_sha256": reconciliation_report[
                "raw_manifest_sha256"
            ],
        }
    rows = rank_rows[reference_rank]
    mapping_path = mapping_paths[reference_rank]
    missing_stack_nodes = sorted(
        {str(row["node"]) for row in rows if row.get("node") and not row.get("python_stack")}
    )
    typed_unresolved_event_count = sum(
        bool(row.get("node")) and row.get("reconciliation_status") != "closed"
        for row in rows
    )

    model_rows = [row for row in rows if row.get("node")]
    start = min(float(row["ts_us"]) for row in rows)
    stop = max(float(row["ts_us"]) + float(row["dur_us"]) for row in rows)
    model_start = min(float(row["ts_us"]) for row in model_rows)
    model_stop = max(float(row["ts_us"]) + float(row["dur_us"]) for row in model_rows)
    active_us = interval_union_us(model_rows)
    residency_us = sum(float(row["dur_us"]) for row in model_rows)
    instrumented_envelope_ms = (model_stop - model_start) / 1000.0
    active_ms = active_us / 1000.0
    wall_trace_gate = wall_trace_contract_gate(
        item,
        runtime_coordinate,
        serving_wall_ms=serving_wall_ms,
        active_gpu_ms=active_ms,
        kernel_envelope_ms=instrumented_envelope_ms,
    )
    timing = {
        "elapsed_ms": round(serving_wall_ms, 6),
        "serving_wall_ms": round(serving_wall_ms, 6),
        "active_gpu_ms": round(active_ms, 6),
        "gpu_residency_ms": round(residency_us / 1000.0, 6),
        "device_gap_ms": round(max(0.0, serving_wall_ms - active_ms), 6),
        "gpu_overlap_ms": round(max(0.0, residency_us - active_us) / 1000.0, 6),
        "kernel_envelope_ms": round(instrumented_envelope_ms, 6),
        "instrumented_trace_overhead_ms": round(
            max(0.0, instrumented_envelope_ms - serving_wall_ms), 6
        ),
        "instrumented_active_excess_ms": round(
            max(0.0, active_ms - serving_wall_ms), 6
        ),
        "wall_authority": "selected profiler-off production baseline forward",
        "layout_active_residency_authority": "instrumented production trace; profiler overhead is explicit and does not replace serving wall",
        "wall_trace_contract_gate": wall_trace_gate,
        "semantics": (
            "elapsed/device-gap authority is the exact selected profiler-off "
            "production baseline; kernel layout, active interval union, and "
            "residency come from one median-envelope rank of the instrumented "
            "formal forward and may include explicit profiler overhead"
        ),
    }
    metrics = build_metrics(model_rows)
    for metric in metrics.values():
        metric["source_rank"] = reference_rank
        metric["rank_policy"] = (
            "one coherent global rank nearest the all-rank median mapped envelope; "
            "layout/active/residency only, never serving-wall authority"
        )
    states, fusion_groups = build_states_and_fusions(
        model_ir=model_ir, execution_plan=execution_plan, rows=model_rows, metrics=metrics
    )
    acceptance_gate = profile_acceptance_gate(
        rank_diagnostics=rank_diagnostics,
        typed_unresolved_event_count=typed_unresolved_event_count,
        node_states=states,
        fusion_groups=fusion_groups,
        wall_trace_gate=wall_trace_gate,
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
            "all_rank_mapping_sha256": {
                str(rank): sha256_file(path)
                for rank, path in sorted(mapping_paths.items())
            },
            "match_contract": "same rank + phase + occurrence + exact kernel-signature ordered sequence",
            "representative_node_fallback": False,
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
        "rank_collective_duration_gate": collective_duration_gate,
        "profiler_sync_evidence": profiler_sync_evidence,
        "profile_timing": timing,
        "wall_trace_contract_gate": wall_trace_gate,
        "acceptance_gate": acceptance_gate,
        "eager_mapping_by_rank": {
            str(rank): {"path": str(path), "sha256": sha256_file(path)}
            for rank, path in sorted(mapping_paths.items())
        },
        "eager_reconciliation_by_rank": reconciliation_reports,
        "missing_stack_nodes": missing_stack_nodes,
        "typed_unresolved_event_count": typed_unresolved_event_count,
        "unclassified_kernel_count": sum(
            not row.get("node") and not row.get("support_class") for rank in rank_rows.values() for row in rank
        ),
    }
    validation_path = evidence_dir(task_root, item) / "validation.json"
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")

    concurrency = batch
    label_phase = "prefill" if phase == "prefill" else "CUDA Graph decode"
    profile = {
        "schema_version": "profile.v2",
        "acceptance": acceptance_gate,
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
            "timing_gate_status": wall_trace_gate["state"],
            "rank_collective_duration_gate": collective_duration_gate,
            "profiler_sync_evidence": profiler_sync_evidence,
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
            "all_rank_eager_mapping_sha256": {
                str(rank): sha256_file(path)
                for rank, path in sorted(mapping_paths.items())
            },
            "all_rank_eager_raw_manifest_sha256": {
                rank: report["raw_manifest_sha256"]
                for rank, report in sorted(reconciliation_reports.items())
            },
            "all_rank_eager_phase_contract": {
                rank: {
                    "source_phase": report["source_phase"],
                    "selected_forward_kernel_count": report["kernel_count"],
                    "selected_forward_kernel_duration_us": round(
                        sum(
                            float(json.loads(line).get("dur_us") or 0.0)
                            for line in (
                                task_root
                                / "mapping"
                                / f"reconciled-{framework}-{phase}"
                                / f"events.tp{rank}.jsonl"
                            ).read_text().splitlines()
                            if line.strip()
                        ),
                        6,
                    ),
                    "raw_trace_sha256": report["raw_trace_sha256"],
                    "raw_manifest_sha256": report["raw_manifest_sha256"],
                    "selected_forward_events_sha256": report[
                        "selected_forward_events_sha256"
                    ],
                }
                for rank, report in sorted(reconciliation_reports.items())
            },
            "mapped_kernel_count_ratio": rank_diagnostics[str(reference_rank)]["mapped_kernel_count_ratio"],
            "mapped_kernel_duration_ratio": rank_diagnostics[str(reference_rank)]["mapped_kernel_duration_ratio"],
            "unclassified_kernel_count": 0,
            "semantic_stack_closure_missing_node_count": len(missing_stack_nodes),
            "typed_unresolved_semantic_event_count": typed_unresolved_event_count,
            "mapping_policy": "phase-specific same-rank eager stack plus exact occurrence-scoped physical kernel-signature sequence; no representative-node fallback; unresolved candidates are typed review_required",
            "attribution_diagnostics": rank_diagnostics[str(reference_rank)],
            "timing": timing,
            "acceptance_gate": acceptance_gate,
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
        "acceptance": acceptance_gate,
        "profile_id": profile_id,
        "framework": framework,
        "phase": phase,
        "batch_size": batch,
        "profile_path": str(profile_path),
        "timeline_path": str(timeline_path),
        "source_commit": source_commit,
        "runtime_model_module_commit": runtime_module_commit,
        "graph_evidence": graph_evidence,
        "typed_unresolved_semantic_event_count": typed_unresolved_event_count,
        "rank_typed_unresolved_semantic_event_count": {
            rank: diagnostics["semantic_reconciliation"]["typed_unresolved_event_count"]
            for rank, diagnostics in sorted(rank_diagnostics.items())
        },
        "partial_fusion_node_count": sum(
            state.get("status") == "partially_fused" for state in states.values()
        ),
        "incomplete_fusion_owner_closure_node_count": sum(
            state.get("status") == "partially_fused"
            and state.get("all_owner_events_same_rank_closed") is False
            for state in states.values()
        ),
        "full_fusion_group_count": len(fusion_groups),
        "full_fusion_groups_all_closed": all(
            group.get("evidence_scope", {}).get("member_event_sets_equal_owner") is True
            and group.get("evidence_scope", {}).get("all_owner_events_same_rank_closed") is True
            for group in fusion_groups.values()
        ),
        "wall_trace_contract_gate": wall_trace_gate,
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
    accepted = [result for result in results if result["acceptance"]["state"] == "accepted"]
    unsupported = [result for result in results if result["acceptance"]["state"] != "accepted"]
    rejected_dir = args.task_root / "validation" / "rejected-profiles"
    rejected_dir.mkdir(parents=True, exist_ok=True)
    public_unsupported = []
    for result in unsupported:
        profile_path = Path(result["profile_path"])
        timeline_path = Path(result["timeline_path"])
        profile = yaml.safe_load(profile_path.read_text())
        rejected_profile = rejected_dir / f"{result['profile_id']}.yaml"
        rejected_timeline = rejected_dir / f"{result['profile_id']}.timeline.json.gz"
        profile_path.replace(rejected_profile)
        timeline_path.replace(rejected_timeline)
        public_unsupported.append(
            {
                "profile_id": result["profile_id"],
                "state": "unsupported",
                "framework": result["framework"],
                "phase": result["phase"],
                "global_batch_size": result["batch_size"],
                "job_id": result["job_id"],
                "source_commit": result["source_commit"],
                "runtime_model_module_commit": result["runtime_model_module_commit"],
                "hardware": profile["hardware"],
                "workload": profile["workload"],
                "reason_codes": [
                    reason["code"] for reason in result["acceptance"]["reasons"]
                ],
                "typed_unresolved_semantic_event_count": result[
                    "typed_unresolved_semantic_event_count"
                ],
                "rank_typed_unresolved_semantic_event_count": result[
                    "rank_typed_unresolved_semantic_event_count"
                ],
                "partial_fusion_node_count": result["partial_fusion_node_count"],
                "incomplete_fusion_owner_closure_node_count": result[
                    "incomplete_fusion_owner_closure_node_count"
                ],
                "full_fusion_group_count": result["full_fusion_group_count"],
                "full_fusion_groups_all_closed": result[
                    "full_fusion_groups_all_closed"
                ],
                "false_fill_qk_rope_fusion_published": any(
                    group.get("owner") == "full_attention.qk_norm"
                    and "full_attention.partial_rope" in group.get("ir_nodes", [])
                    for group in profile["fusion_groups"].values()
                ),
                "wall_trace_contract_gate": result["wall_trace_contract_gate"],
                "selected_forward_cuda_graph": result["graph_evidence"],
                "all_rank_eager_phase_contract": profile["evidence"][
                    "all_rank_eager_phase_contract"
                ],
                "all_rank_eager_raw_manifest_sha256": profile["evidence"][
                    "all_rank_eager_raw_manifest_sha256"
                ],
                "all_rank_eager_mapping_sha256": profile["evidence"][
                    "all_rank_eager_mapping_sha256"
                ],
                "all_rank_production_trace_sha256": profile["evidence"][
                    "all_rank_trace_sha256"
                ],
                "validation_sha256": result["validation_sha256"],
                "rejected_profile_sha256": sha256_file(rejected_profile),
                "rejected_timeline_sha256": sha256_file(rejected_timeline),
                "diagnostic_artifacts": {
                    "profile": (
                        "current/qwen35-complete-profiles/validation/rejected-profiles/"
                        f"{rejected_profile.name}"
                    ),
                    "timeline": (
                        "current/qwen35-complete-profiles/validation/rejected-profiles/"
                        f"{rejected_timeline.name}"
                    ),
                    "validation": "retained outside git; identified by validation_sha256",
                },
            }
        )

    # Remove now-empty generated directories.  Accepted profiles, if any,
    # remain in the canonical profiles tree and are the only profiles compiled
    # into the Viewer bundle.
    for directory in sorted(
        {Path(result["profile_path"]).parent for result in unsupported},
        reverse=True,
    ):
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()

    unsupported_manifest = {
        "schema_version": "qwen35-unsupported-profile-matrix.v1",
        "model_id": "qwen35",
        "execution_path_id": "tp8",
        "expected_profile_count": len(MATRIX),
        "accepted_profile_count": len(accepted),
        "unsupported_profile_count": len(unsupported),
        "acceptance_policy": (
            "only profile.v2 files that pass zero-unresolved semantic closure, "
            "exact fusion ownership, and wall/trace contract gates are compiled "
            "into the canonical Viewer"
        ),
        "profiles": public_unsupported,
    }
    (catalog_root / "unsupported_profiles.yaml").write_text(
        yaml.safe_dump(
            unsupported_manifest, sort_keys=False, allow_unicode=True, width=120
        )
    )

    report = {
        "schema_version": "qwen35-profile-matrix.v1",
        "raw_artifact_policy": "preserved outside git; exact paths and SHA256 values recorded per TP rank",
        "expected_profile_count": len(MATRIX),
        "measured_profile_count": len(accepted),
        "unsupported_profile_count": len(unsupported),
        "accepted_profiles": accepted,
        "unsupported_profiles": unsupported,
        "public_unsupported_manifest": str(catalog_root / "unsupported_profiles.yaml"),
    }
    output = args.task_root / "validation" / "profile-matrix.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
