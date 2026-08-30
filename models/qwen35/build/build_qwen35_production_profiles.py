#!/usr/bin/env python3
"""Build the complete measured Qwen3.5 pure-TP8 profile matrix.

Raw traces stay under the task evidence root.  The repository receives only
compact deterministic profile overlays and timeline artifacts.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
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
    build_timeline_artifact,
    write_timeline_artifact,
)
from models.common.trace_mapping import (  # noqa: E402
    find_step_annotation_windows,
    load_trace,
    normalize_kernel_events,
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


GRAPH_OFF_CAPTURES = {
    ("sglang", "prefill", 1): {"job": "3435301", "profile_step": 0},
    ("sglang", "decode", 1): {"job": "3434399", "profile_step": 511},
    ("sglang", "decode", 16): {"job": "3435305", "profile_step": 529},
    ("sglang", "decode", 64): {"job": "3435306", "profile_step": 582},
    ("sglang", "decode", 256): {"job": "3436395", "profile_step": 769},
    ("vllm", "prefill", 1): {"job": "3436383", "profile_delay_iterations": 0},
    ("vllm", "decode", 1): {"job": "3434240", "profile_delay_iterations": 513},
    ("vllm", "decode", 16): {"job": "3436385", "profile_delay_iterations": 521},
    ("vllm", "decode", 64): {"job": "3436387", "profile_delay_iterations": 545},
    ("vllm", "decode", 256): {"job": "3436389", "profile_delay_iterations": 643},
}


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
                "graph_node_id": args.get("graph node id"),
                "correlation": args.get("correlation"),
                "external_id": args.get("External id"),
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
                    "graph_node_id": (
                        event["graphNodeId"]
                        if "graphNodeId" in event.keys()
                        else None
                    ),
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


def sglang_prefill_forward_timing_coordinate(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the exact profiler-off DeviceTimer authority for SGLang prefill.

    SGLang's periodic input-throughput log measures the interval since the
    previous prefill log and therefore includes intervening decode iterations.
    It is not a single-forward wall clock.  The retained ForwardPassMetrics
    stream has one CUDA-event DeviceTimer span per model forward plus the
    realized scheduler composition, so select the unique post-warmup 8K-only
    formal prefill directly from that raw stream.
    """

    selection_path = root / "validation" / "sglang-prefill-fpm-selection.json"
    selection = json.loads(selection_path.read_text())
    expected_header = {
        "schema_version": "qwen35-sglang-forward-timing-selection.v1",
        "state": "passed",
        "framework": "sglang",
        "phase": "prefill",
        "batch_size": 1,
    }
    for field, expected in expected_header.items():
        if selection.get(field) != expected:
            raise ValueError(
                f"SGLang prefill DeviceTimer selection {field}="
                f"{selection.get(field)!r}, expected {expected!r}"
            )
    contract = selection.get("contract") or {}
    expected_contract = {
        "input_length": 8192,
        "output_length": 1024,
        "warmup_request_count": 3,
        "formal_request_count": 1,
        "no_intentionally_shared_prefix": True,
        "ignore_eos": True,
        "mtp_nextn": False,
    }
    for field, expected in expected_contract.items():
        if contract.get(field) != expected:
            raise ValueError(
                f"SGLang prefill DeviceTimer contract {field}="
                f"{contract.get(field)!r}, expected {expected!r}"
            )
    expected_topology = {
        "tensor_parallel_size": 8,
        "data_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "expert_parallel_size": 1,
    }
    if contract.get("topology") != expected_topology:
        raise ValueError("SGLang prefill DeviceTimer topology mismatch")

    retained = selection.get("evidence") or {}
    client_relative = Path(str(retained.get("client_path") or ""))
    if client_relative.is_absolute() or ".." in client_relative.parts:
        raise ValueError("invalid SGLang prefill DeviceTimer client evidence path")
    client_path = root / client_relative
    if not client_path.is_file() or retained.get("client_sha256") != sha256_file(client_path):
        raise ValueError("SGLang prefill DeviceTimer client evidence SHA-256 mismatch")
    client = json.loads(client_path.read_text())
    if client.get("state") != "passed" or client.get("contract") != {
        "concurrency": 1,
        "formal_request_count": 1,
        "ignore_eos": True,
        "isl": 8192,
        "mtp_nextn": False,
        "no_intentionally_shared_prefix": True,
        "osl": 1024,
        "random_range_ratio": 1.0,
        "random_token_ids": True,
        "seed": 0,
        "warmup_request_count": 3,
    }:
        raise ValueError("SGLang prefill DeviceTimer raw client contract mismatch")
    fpm = client.get("forward_pass_metrics") or {}
    floor = int(fpm.get("counter_floor_after_warmup", -1))
    matches = []
    for message in fpm.get("messages") or []:
        metrics = message.get("metrics") or {}
        scheduled = metrics.get("scheduled_requests") or {}
        if (
            int(metrics.get("counter_id", -1)) > floor
            and float(metrics.get("wall_time") or 0.0) > 0.0
            and scheduled.get("num_prefill_requests") == 1
            and scheduled.get("sum_prefill_tokens") == 8192
            and scheduled.get("sum_prefill_kv_tokens") == 8192
            and scheduled.get("num_decode_requests") == 0
            and scheduled.get("sum_decode_kv_tokens") == 0
        ):
            matches.append(message)
    if len(matches) != 1:
        raise ValueError(
            "SGLang prefill DeviceTimer selection requires exactly one post-warmup "
            f"formal 8K-only forward, got {len(matches)}"
        )
    selected_message = matches[0]
    selected_metrics = selected_message["metrics"]
    declared = selection.get("selection") or {}
    exact_fields = {
        "counter_floor_after_warmup": floor,
        "matching_message_count": 1,
        "transport_sequence": selected_message.get("transport_sequence"),
        "counter_id": selected_metrics.get("counter_id"),
        "received_at": selected_message.get("received_at"),
        "payload_sha256": selected_message.get("payload_sha256"),
        "wall_time_seconds": selected_metrics.get("wall_time"),
    }
    for field, expected in exact_fields.items():
        if declared.get(field) != expected:
            raise ValueError(
                f"SGLang prefill DeviceTimer selection {field}="
                f"{declared.get(field)!r}, recomputed {expected!r}"
            )
    wall_ms = float(selected_metrics["wall_time"]) * 1000.0
    if abs(float(declared.get("wall_time_ms") or 0.0) - wall_ms) > 1e-9:
        raise ValueError("SGLang prefill DeviceTimer millisecond conversion mismatch")
    return {
        "baseline_mean_elapsed_ms": wall_ms,
        "timing_authority": "profiler_off_per_forward_device_timer",
        "counter_floor_after_warmup": floor,
        "selected_counter_id": selected_metrics["counter_id"],
        "selected_transport_sequence": selected_message["transport_sequence"],
        "selected_payload_sha256": selected_message["payload_sha256"],
        "selected_scheduler_composition": selected_metrics["scheduled_requests"],
        "same_isolated_forward_proven": True,
        "rejected_previous_authority": selection["rejected_previous_authority"],
    }, {
        "file": selection_path.name,
        "sha256": sha256_file(selection_path),
        "raw_client_file": client_path.name,
        "raw_client_sha256": sha256_file(client_path),
        "job_id": selection["job_id"],
        "state": selection["state"],
    }


def selected_runtime_coordinate(root: Path, item: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the exact baseline selector and its immutable evidence reference."""

    framework = item["framework"]
    phase = item["phase"]
    batch = str(item["batch"])
    if framework == "sglang":
        if phase == "prefill":
            return sglang_prefill_forward_timing_coordinate(root)
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


def graph_off_evidence_dir(root: Path, item: dict[str, Any]) -> Path:
    capture = GRAPH_OFF_CAPTURES[(item["framework"], item["phase"], item["batch"])]
    job = capture["job"]
    if item["framework"] == "vllm":
        return root / "evidence" / "vllm-graph_off" / job
    suffix = "prefill-c1" if item["phase"] == "prefill" else f"decode-c{item['batch']}"
    return root / "evidence" / f"sglang-graph_off-{suffix}" / job


def graph_off_trace_paths(
    root: Path, item: dict[str, Any]
) -> tuple[Path, dict[int, Path]]:
    directory = graph_off_evidence_dir(root, item)
    pattern = (
        "traces/*.trace.json.gz"
        if item["framework"] == "vllm"
        else "logs/profiles/*/*.trace.json.gz"
    )
    paths = {
        rank_from_name(path): path for path in sorted(directory.glob(pattern))
    }
    if set(paths) != set(range(8)):
        raise ValueError(
            f"{item['framework']}/{item['phase']}/BS{item['batch']}: expected "
            f"graph-off TP ranks 0..7 under {directory}, got {sorted(paths)}"
        )
    return directory, paths


def validate_graph_off_client_contract(
    directory: Path, item: dict[str, Any]
) -> tuple[Path, dict[str, Any]]:
    client_path = (
        directory / f"client-c{item['batch']}.json"
        if item["framework"] == "vllm"
        else directory / "logs" / f"client-c{item['batch']}.json"
    )
    if not client_path.is_file():
        raise ValueError(f"missing graph-off client contract: {client_path}")
    client = json.loads(client_path.read_text())
    expected = {
        "concurrency": item["batch"],
        "formal_request_count": item["batch"],
        "warmup_request_count": 3 * item["batch"],
        "isl": 8192,
        "osl": 1024,
        "ignore_eos": True,
        "mtp_nextn": False,
        "no_intentionally_shared_prefix": True,
        "random_token_ids": True,
        "random_range_ratio": 1.0,
        "seed": 0,
    }
    contract = client.get("contract") or {}
    mismatches = {
        key: {"actual": contract.get(key), "expected": value}
        for key, value in expected.items()
        if contract.get(key) != value
    }
    for section, count in (("warmup", 3 * item["batch"]), ("formal", item["batch"])):
        cell = client.get(section) or {}
        if (
            cell.get("request_count") != count
            or cell.get("failure_count") != 0
            or any(
                request.get("realized_prompt_tokens") != 8192
                or request.get("realized_completion_tokens") != 1024
                for request in cell.get("requests") or []
            )
        ):
            mismatches[section] = {
                "request_count": cell.get("request_count"),
                "failure_count": cell.get("failure_count"),
                "expected_request_count": count,
            }
    if client.get("state") != "passed" or mismatches:
        raise ValueError(
            f"{item['framework']}/{item['phase']}/BS{item['batch']}: graph-off "
            f"client contract mismatch: {mismatches}"
        )
    capture = GRAPH_OFF_CAPTURES[(item["framework"], item["phase"], item["batch"])]
    if item["framework"] == "sglang":
        config_path = directory / (
            f"config_eager_{item['phase']}_c{item['batch']}.yaml"
        )
        config = yaml.safe_load(config_path.read_text())
        identity = config.get("identity") or {}
        backend = config.get("backend") or {}
        environment = backend.get("aggregated_environment") or {}
        runtime = ((backend.get("sglang_config") or {}).get("aggregated") or {})
        profiling = (config.get("profiling") or {}).get("aggregated") or {}
        expected_config = {
            "model_revision": ((identity.get("model") or {}).get("revision"), MODEL_REVISION),
            "source_commit": ((identity.get("frameworks") or {}).get("source_commit"), SGLANG_SOURCE),
            "gpu_type": ((config.get("resources") or {}).get("gpu_type"), "gb300"),
            "nodes": ((config.get("resources") or {}).get("agg_nodes"), 2),
            "gpus_per_node": ((config.get("resources") or {}).get("gpus_per_node"), 4),
            "profile_kind": (environment.get("QWEN_PROFILE_KIND"), "eager"),
            "profile_phase": (environment.get("QWEN_EAGER_PHASE"), item["phase"]),
            "profile_concurrency": (environment.get("QWEN_CONCURRENCY"), str(item["batch"])),
            "profile_step": (environment.get("QWEN_PROFILE_STEP"), str(capture["profile_step"])),
            "tp_size": (runtime.get("tensor-parallel-size"), 8),
            "dp_size": (runtime.get("data-parallel-size"), 1),
            "pp_size": (runtime.get("pipeline-parallel-size"), 1),
            "ep_size": (runtime.get("expert-parallel-size"), 1),
            "cuda_graph_disabled": (runtime.get("disable-cuda-graph"), True),
            "prefix_cache_disabled": (runtime.get("disable-radix-cache"), True),
            "profile_start_step": (profiling.get("start_step"), capture["profile_step"]),
            "profile_stop_step": (profiling.get("stop_step"), capture["profile_step"] + 1),
        }
        mismatched_config = {
            name: {"actual": actual, "expected": expected}
            for name, (actual, expected) in expected_config.items()
            if actual != expected
        }
        if mismatched_config:
            raise ValueError(
                f"graph-off SGLang capture config mismatch in {config_path}: "
                f"{mismatched_config}"
            )
        coordinate = client.get("profile_coordinate") or {}
        expected_step = capture["profile_step"]
        relative_step = coordinate.get("baseline_relative_start_step")
        resolved_step = coordinate.get("resolved_absolute_start_step")
        requested_step = (
            (client.get("profile_controls") or [{}])[0]
            .get("request", {})
            .get("start_step")
        )
        if (
            coordinate.get("mode") != "formal_relative_resolved_from_scheduler_log"
            or relative_step != expected_step
            or resolved_step != requested_step
        ):
            raise ValueError(
                "graph-off SGLang formal coordinate mismatch: "
                f"relative={relative_step}, expected={expected_step}, "
                f"resolved={resolved_step}, requested={requested_step}"
            )
    else:
        server_path = directory / "server.log"
        server_text = server_path.read_text(errors="replace")
        expected_delay = capture["profile_delay_iterations"]
        required = (
            "cudagraph_mode': <CUDAGraphMode.NONE: 0>",
            "version 0.1.dev20051+g487ecf187",
            "tensor_parallel_size=8",
            "pipeline_parallel_size=1",
            "data_parallel_size=1",
            "served_model_name=nvidia/Qwen3.5-397B-A17B-NVFP4-V2",
            f"delay_iterations={expected_delay}",
            "max_iterations=1",
            r'torch_profiler_record_shapes\":true',
            "torch_profiler_with_stack=True",
        )
        if any(fragment not in server_text for fragment in required):
            raise ValueError(
                f"graph-off vLLM compile/profile contract mismatch in {server_path}"
            )
    return client_path, client


def load_graph_off_rank(
    root: Path, item: dict[str, Any], rank: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    directory, paths = graph_off_trace_paths(root, item)
    client_path, client = validate_graph_off_client_contract(directory, item)
    path = paths[rank]
    trace = load_trace(path)
    distributed = trace.get("distributedInfo") or {}
    device_properties = trace.get("deviceProperties") or []
    if (
        trace.get("with_stack") != 1
        or trace.get("record_shapes") != 1
        or distributed.get("rank") != rank
        or distributed.get("world_size") != 8
        or not device_properties
        or {device.get("name") for device in device_properties} != {"NVIDIA GB300"}
    ):
        raise ValueError(
            f"{item['framework']}/{item['phase']}/BS{item['batch']}/TP{rank}: "
            "graph-off trace rank/hardware/stack/shape contract mismatch"
        )
    if item["framework"] == "vllm" and (
        trace.get("vllm_version") != "0.1.dev20051+g487ecf187"
        or (trace.get("vllm_version_tuple") or [])[-1:] != ["g487ecf187"]
    ):
        raise ValueError(
            f"vLLM graph-off trace source version mismatch in {path}"
        )
    phase_name = (
        f"forward_{'extend' if item['phase'] == 'prefill' else 'decode'}"
        if item["framework"] == "sglang"
        else f"vllm_{item['phase']}"
    )
    windows = find_step_annotation_windows(
        trace.get("traceEvents") or [], phase=phase_name
    )
    if len(windows) != 1:
        raise ValueError(
            f"{item['framework']}/{item['phase']}/BS{item['batch']}/TP{rank}: "
            f"expected exactly one graph-off formal window, got {len(windows)}"
        )
    window = windows[0]
    raw_selected_kernels = [
        event
        for event in trace.get("traceEvents") or []
        if event.get("cat") == "kernel"
        and event.get("ph") == "X"
        and window.start_us <= float(event.get("ts", 0.0)) <= window.end_us
    ]
    nonzero_graph_ids = {
        str((event.get("args") or {}).get("graph id"))
        for event in raw_selected_kernels
        if str((event.get("args") or {}).get("graph id") or "0") not in {"0", "None", ""}
    }
    if nonzero_graph_ids:
        raise ValueError(
            f"graph-off TP{rank} selected window contains nonzero graph IDs: "
            f"{sorted(nonzero_graph_ids)}"
        )
    source_root = (
        Path("/Users/yangminl/Documents/Projects/sglang-qwen35-profile-source-20260829")
        if item["framework"] == "sglang"
        else Path("/Users/yangminl/Documents/Projects/vllm-qwen35-profile-source-20260829")
    )
    normalized = normalize_kernel_events(
        trace.get("traceEvents") or [], window=window, source_root=source_root
    )
    rows = []
    for event in normalized:
        row = asdict(event)
        row["event_id"] = f"e-r{rank}-{event.event_id}"
        row["rank"] = rank
        rows.append(row)
    if len(rows) != len(raw_selected_kernels):
        raise ValueError(
            f"graph-off TP{rank} normalized/raw kernel count mismatch: "
            f"{len(rows)} != {len(raw_selected_kernels)}"
        )
    missing_stacks = [row["event_id"] for row in rows if not row.get("python_stack")]
    if missing_stacks:
        raise ValueError(
            f"graph-off TP{rank} has {len(missing_stacks)} kernels without Python stacks"
        )
    diagnostics = attribute_production_forward(
        rows, framework=item["framework"], phase=item["phase"]
    )
    capture_metadata_path = (
        directory / f"config_eager_{item['phase']}_c{item['batch']}.yaml"
        if item["framework"] == "sglang"
        else directory / "server.log"
    )
    return rows, {
        "job_id": GRAPH_OFF_CAPTURES[
            (item["framework"], item["phase"], item["batch"])
        ]["job"],
        "trace_path": str(path),
        "trace_sha256": sha256_file(path),
        "client_path": str(client_path),
        "client_sha256": sha256_file(client_path),
        "capture_metadata_path": str(capture_metadata_path),
        "capture_metadata_sha256": sha256_file(capture_metadata_path),
        "source_commit": (
            SGLANG_SOURCE if item["framework"] == "sglang" else VLLM_SOURCE
        ),
        "runtime_source_version": (
            "0.0.0.dev1+gf609d677b"
            if item["framework"] == "sglang"
            else trace["vllm_version"]
        ),
        "hardware": "NVIDIA GB300",
        "world_size": distributed["world_size"],
        "selected_formal_coordinate": (
            {
                "mode": "sglang_formal_relative_scheduler_forward",
                "relative_step": GRAPH_OFF_CAPTURES[
                    (item["framework"], item["phase"], item["batch"])
                ]["profile_step"],
                "resolved_absolute_step": client["profile_coordinate"][
                    "resolved_absolute_start_step"
                ],
            }
            if item["framework"] == "sglang"
            else {
                "mode": "vllm_start_profile_relative_engine_iteration",
                "delay_iterations": GRAPH_OFF_CAPTURES[
                    (item["framework"], item["phase"], item["batch"])
                ]["profile_delay_iterations"],
                "active_iterations": 1,
            }
        ),
        "phase": phase_name,
        "rank": rank,
        "window_start_us": window.start_us,
        "window_end_us": window.end_us,
        "window_duration_ms": (window.end_us - window.start_us) / 1000.0,
        "selected_forward_kernel_count": len(rows),
        "selected_forward_kernel_duration_us": sum(
            float(row.get("dur_us") or 0.0) for row in rows
        ),
        "selected_forward_events_sha256": hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "nonzero_graph_id_count": 0,
        "all_kernel_python_stack_count": len(rows),
        "attribution_diagnostics": diagnostics,
        "client_contract": client["contract"],
    }


def reconciliation_kernel_family(row: dict[str, Any]) -> str:
    name = str(row.get("kernel_name") or "").lower()
    # Torch and Nsight demangle the same ATen template with different spelling
    # (for example ``4ul`` versus ``(unsigned long)4`` and anonymous lambdas
    # versus numbered lambda instances).  These families retain the semantic
    # operation and scalar-width distinction while discarding only that tool
    # presentation difference.
    if "direct_copy_kernel_cuda" in name:
        return "direct_copy_i64" if "lambda(long" in name else "direct_copy_i32"
    if "index_elementwise_kernel" in name:
        return (
            "index_copy_elementwise"
            if "index_copy_kernel_impl" in name
            else "index_elementwise"
        )
    if "exp_kernel_cuda" in name:
        return "exp_elementwise"
    if "launch_clamp_scalar" in name:
        return "clamp_elementwise"
    if "devicescaninitkernel" in name:
        return "device_scan_init"
    if "devicescankernel" in name:
        return "device_scan"
    if "cudafunctoronself_add" in name:
        return "onself_add_elementwise"
    families = (
        ("all_reduce", ("allreduce", "all_reduce", "multimem_all_reduce")),
        ("all_gather", ("allgather", "all_gather")),
        ("static_quant", ("static_quant",)),
        ("nvfp4_quant", ("nvfp4_quant", "cvt_fp16_to_fp4")),
        ("qkv_split", ("fused_qkvzba_split",)),
        ("causal_conv", ("causal_conv1d",)),
        ("gdn_recurrence", ("gated_delta", "chunkedgateddeltanet")),
        ("layer_norm", ("layer_norm", "rmsnorm", "rms_norm")),
        ("qqtst_gemm", ("qqtst_",)),
        ("tst_gemm", ("tst_",)),
        ("splitk_reduce", ("splitkreduce",)),
        ("moe_routing", ("routingindices", "routingindicescoop")),
        ("moe_up", ("bmm_e2m1",)),
        ("moe_down", ("bmm_bfloat16",)),
        ("moe_finalize", ("finalizekernel",)),
        ("fmha", ("fmha",)),
        ("kv_cache", ("reshape_and_cache", "qkv_kv_cache")),
        ("fill", ("fillfunctor",)),
        ("act_and_mul", ("act_and_mul_kernel",)),
        ("sigmoid_gate", ("sigmoid_mul", "sigmoid_kernel", "mulfunctor")),
        ("cutlass_gemm", ("cutlass13device_kernel", "kernel_cutlass_kernel")),
    )
    for family, tokens in families:
        if any(token in name for token in tokens):
            return family
    if name.startswith("triton_"):
        return name.split("_0", 1)[0]
    return name


def _binding_stack_hash(stack: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(stack, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_matched_graph_off_mapping(
    root: Path,
    item: dict[str, Any],
    rank: int,
    production_rows: list[dict[str, Any]],
) -> tuple[Path, dict[str, Any]]:
    eager_rows, capture = load_graph_off_rank(root, item, rank)
    eager_model = [row for row in eager_rows if row.get("node")]
    production_model = [row for row in production_rows if row.get("node")]
    key = lambda row: (
        str(row.get("node") or ""),
        str(row.get("occurrence_id") or "top"),
    )
    eager_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    production_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in eager_model:
        eager_groups[key(row)].append(row)
    for row in production_model:
        production_groups[key(row)].append(row)
    if set(eager_groups) != set(production_groups):
        raise ValueError(
            f"{item['framework']}/{item['phase']}/BS{item['batch']}/TP{rank}: "
            "graph-off/production semantic occurrence keys differ: "
            f"eager_only={sorted(set(eager_groups)-set(production_groups))} "
            f"production_only={sorted(set(production_groups)-set(eager_groups))}"
        )
    bindings: dict[str, dict[str, Any]] = {}
    relation_counts: Counter[str] = Counter()
    group_reports = []
    for group_key in sorted(production_groups):
        eager_group = sorted(
            eager_groups[group_key],
            key=lambda row: int(row.get("semantic_sequence_index", 0)),
        )
        production_group = sorted(
            production_groups[group_key],
            key=lambda row: int(row.get("semantic_sequence_index", 0)),
        )
        eager_by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in eager_group:
            eager_by_name[str(row["kernel_name"])].append(row)
        used_source_ids: set[str] = set()
        pending_production = []
        for row in production_group:
            candidates = eager_by_name.get(str(row["kernel_name"])) or []
            source = next(
                (candidate for candidate in candidates if candidate["event_id"] not in used_source_ids),
                None,
            )
            if source is None:
                pending_production.append(row)
                continue
            used_source_ids.add(str(source["event_id"]))
            bindings[str(row["event_id"])] = {
                "sources": [source],
                "relation": "one_to_one_exact_signature",
            }
        # CUDA Graph replay adds a router-GEMM memcpy epilogue in SGLang.  It
        # is a proved 1:N physical decomposition of the same router projection
        # within the exact layer/MoE occurrence.
        still_pending = []
        for row in pending_production:
            name = str(row.get("kernel_name") or "").lower()
            if (
                item["framework"] == "sglang"
                and group_key[0] == "moe_block.router"
                and name in {"memcpy32_post", "memcpy128"}
            ):
                sources = [
                    source
                    for source in eager_group
                    if "tst_" in str(source.get("kernel_name") or "").lower()
                    and "qqtst_" not in str(source.get("kernel_name") or "").lower()
                ]
                if len(sources) != 1:
                    raise ValueError(
                        f"{group_key}: router companion lacks one exact eager projection source"
                    )
                used_source_ids.add(str(sources[0]["event_id"]))
                bindings[str(row["event_id"])] = {
                    "sources": sources,
                    "relation": "one_eager_to_many_production_router_epilogue",
                }
            elif group_key[0] == "moe_block.router" and "splitkreduce" in name:
                sources = [
                    source
                    for source in eager_group
                    if "tst_" in str(source.get("kernel_name") or "").lower()
                    and "qqtst_" not in str(source.get("kernel_name") or "").lower()
                ]
                if len(sources) != 1:
                    raise ValueError(
                        f"{group_key}: graph-only router split-K epilogue lacks "
                        "one exact eager projection source"
                    )
                used_source_ids.add(str(sources[0]["event_id"]))
                bindings[str(row["event_id"])] = {
                    "sources": sources,
                    "relation": "one_eager_to_many_production_router_splitk_epilogue",
                }
            else:
                still_pending.append(row)
        pending_production = still_pending
        remaining_sources = [
            row for row in eager_group if str(row["event_id"]) not in used_source_ids
        ]
        if len(pending_production) == 1:
            target = pending_production[0]
            target_family = reconciliation_kernel_family(target)
            family_sources = [
                source
                for source in remaining_sources
                if reconciliation_kernel_family(source) == target_family
            ]
            splitk_sources = [
                source
                for source in remaining_sources
                if reconciliation_kernel_family(source) == "splitk_reduce"
            ]
            if len(family_sources) == 1 and len(splitk_sources) == 1:
                source_set = [family_sources[0], splitk_sources[0]]
                bindings[str(target["event_id"])] = {
                    "sources": source_set,
                    "relation": "many_eager_gemm_splitk_to_one_production",
                }
                used_source_ids.update(str(source["event_id"]) for source in source_set)
                pending_production = []
                remaining_sources = [
                    row
                    for row in remaining_sources
                    if str(row["event_id"]) not in used_source_ids
                ]
        eager_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
        production_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in remaining_sources:
            eager_by_family[reconciliation_kernel_family(row)].append(row)
        for row in pending_production:
            production_by_family[reconciliation_kernel_family(row)].append(row)
        if set(eager_by_family) != set(production_by_family):
            raise ValueError(
                f"{group_key}: unmatched graph-off/production kernel families: "
                f"eager={dict((k,len(v)) for k,v in eager_by_family.items())} "
                f"production={dict((k,len(v)) for k,v in production_by_family.items())}"
            )
        for family in sorted(eager_by_family):
            sources = eager_by_family[family]
            targets = production_by_family[family]
            if len(sources) == len(targets):
                pairs = [([source], target, "one_to_one_normalized_family") for source, target in zip(sources, targets)]
            elif len(sources) == 1:
                pairs = [([sources[0]], target, "one_eager_to_many_production") for target in targets]
            elif len(targets) == 1:
                pairs = [(sources, targets[0], "many_eager_to_one_production")]
            else:
                raise ValueError(
                    f"{group_key}/{family}: ambiguous {len(sources)}:{len(targets)} reconciliation"
                )
            for source_set, target, relation in pairs:
                bindings[str(target["event_id"])] = {
                    "sources": source_set,
                    "relation": relation,
                }
                used_source_ids.update(str(source["event_id"]) for source in source_set)
        if used_source_ids != {str(row["event_id"]) for row in eager_group}:
            raise ValueError(f"{group_key}: graph-off source events were not fully consumed")
        group_reports.append(
            {
                "node": group_key[0],
                "occurrence_id": group_key[1],
                "graph_off_event_count": len(eager_group),
                "production_event_count": len(production_group),
                "graph_off_signature_sha256": hashlib.sha256(
                    json.dumps([row["kernel_name"] for row in eager_group]).encode()
                ).hexdigest(),
                "production_signature_sha256": hashlib.sha256(
                    json.dumps([row["kernel_name"] for row in production_group]).encode()
                ).hexdigest(),
            }
        )
    if set(bindings) != {str(row["event_id"]) for row in production_model}:
        raise ValueError("not every production model event received a graph-off binding")

    output = root / "mapping" / (
        f"reconciled-{item['framework']}-{item['phase']}-bs{item['batch']}"
    )
    output.mkdir(parents=True, exist_ok=True)
    events_path = output / f"events.tp{rank}.jsonl"
    mapping_path = output / f"kernel_mapping.tp{rank}.jsonl"
    reconciled_events = []
    mappings = []
    for sequence_index, row in enumerate(
        sorted(production_model, key=lambda cell: int(cell["semantic_sequence_index"]))
    ):
        binding = bindings[str(row["event_id"])]
        sources = binding["sources"]
        stacks = [source["python_stack"] for source in sources]
        source_ids = [str(source["event_id"]) for source in sources]
        source_kernel_names = [str(source["kernel_name"]) for source in sources]
        stack_hashes = [_binding_stack_hash(stack) for stack in stacks]
        row["python_stack"] = stacks[0]
        row["cpu_op_name"] = sources[0].get("cpu_op_name")
        row["eager_event_ids"] = source_ids
        row["reconciliation_status"] = "closed"
        row["reconciliation_relation"] = binding["relation"]
        row["stack_evidence"] = {
            "source": "matched_graph_off_eager_trace",
            "match": "same_rank_phase_occurrence_signature_ordered_sequence",
            "relation": binding["relation"],
            "eager_event_ids": source_ids,
            "eager_kernel_names": source_kernel_names,
            "stack_sha256": stack_hashes,
            "rank": rank,
            "phase": item["phase"],
            "occurrence_id": row.get("occurrence_id") or "top",
            "production_sequence_index": row["semantic_sequence_index"],
        }
        relation_counts[binding["relation"]] += 1
        reconciled_events.append(
            {
                "event_id": row["event_id"],
                "kernel_name": row["kernel_name"],
                "selected_node": row["node"],
                "occurrence_id": row.get("occurrence_id") or "top",
                "rank": rank,
                "phase": item["phase"],
                "python_stack": stacks[0],
                "source_python_stacks": stacks,
                "source_eager_event_ids": source_ids,
                "source_eager_kernel_names": source_kernel_names,
                "source_stack_sha256": stack_hashes,
                "cpu_op_name": sources[0].get("cpu_op_name"),
                "relation": binding["relation"],
            }
        )
        mappings.append(
            {
                "event_id": row["event_id"],
                "kernel_name": row["kernel_name"],
                "selected_node": row["node"],
                "confidence": "high",
                "rank": rank,
                "phase": item["phase"],
                "source_phase": capture["phase"],
                "occurrence_id": row.get("occurrence_id") or "top",
                "sequence_index": sequence_index,
                "production_sequence_index": row["semantic_sequence_index"],
                "closure_status": "closed",
                "relation": binding["relation"],
                "source_eager_event_ids": source_ids,
                "source_eager_kernel_names": source_kernel_names,
                "source_stack_sha256": stack_hashes,
                "evidence": [
                    "same_rank_eager_python_stack",
                    "same_phase_graph_off_capture",
                    "phase_specific_raw_trace_identity",
                    "occurrence_scoped_ordered_sequence",
                    "explicit_1n_n1_relation",
                ],
            }
        )
    events_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in reconciled_events)
    )
    mapping_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in mappings)
    )
    report = {
        "job_id": capture["job_id"],
        "framework": item["framework"],
        "phase": item["phase"],
        "batch_size": item["batch"],
        "rank": rank,
        "production_semantic_event_count": len(production_model),
        "graph_off_semantic_event_count": len(eager_model),
        "production_support_event_count": sum(
            bool(row.get("support_class")) for row in production_rows
        ),
        "graph_off_support_event_count": sum(
            bool(row.get("support_class")) for row in eager_rows
        ),
        "production_support_class_counts": dict(
            sorted(
                Counter(
                    str(row["support_class"])
                    for row in production_rows
                    if row.get("support_class")
                ).items()
            )
        ),
        "graph_off_support_class_counts": dict(
            sorted(
                Counter(
                    str(row["support_class"])
                    for row in eager_rows
                    if row.get("support_class")
                ).items()
            )
        ),
        "closed_production_event_count": len(mappings),
        "typed_unresolved_event_count": 0,
        "relation_counts": dict(sorted(relation_counts.items())),
        "occurrence_group_count": len(group_reports),
        "occurrence_groups": group_reports,
        "raw_trace_path": capture["trace_path"],
        "raw_trace_sha256": capture["trace_sha256"],
        "raw_manifest_path": capture["client_path"],
        "raw_manifest_sha256": capture["client_sha256"],
        "capture_metadata_path": capture["capture_metadata_path"],
        "capture_metadata_sha256": capture["capture_metadata_sha256"],
        "source_commit": capture["source_commit"],
        "runtime_source_version": capture["runtime_source_version"],
        "hardware": capture["hardware"],
        "world_size": capture["world_size"],
        "selected_formal_coordinate": capture["selected_formal_coordinate"],
        "selected_forward_events_sha256": capture[
            "selected_forward_events_sha256"
        ],
        "source_phase": capture["phase"],
        "kernel_count": capture["selected_forward_kernel_count"],
        "selected_forward_kernel_duration_us": capture[
            "selected_forward_kernel_duration_us"
        ],
        "support_kernel_count": sum(bool(row.get("support_class")) for row in eager_rows),
        "all_kernel_python_stack_count": capture["all_kernel_python_stack_count"],
        "nonzero_graph_id_count": capture["nonzero_graph_id_count"],
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
    for row in rows:
        if row.get("node"):
            direct[str(row["node"])].append(row)
    return {
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
    cpu_op = str(row.get("cpu_op_name") or "").lower()
    method = str(row.get("attribution_method") or "")
    if (
        owner == "full_attention.causal_gqa"
        and member == "full_attention.kv_state_read"
    ):
        return any(token in name for token in ("fmha", "attention"))
    if (
        owner == "full_attention.qk_norm"
        and member == "full_attention.partial_rope"
    ):
        return (
            ("fused_qk" in name and "rope" in name)
            or (
                "vllm_" in method
                and "inductor_source_nodes_qk_norm_rope" in method
                and any(token in name for token in ("triton_poi_fused_7", "triton_poi_fused_8"))
            )
        )
    if (
        owner == "gdn_attention.qkvz_projection"
        and member == "gdn_attention.ba_projection"
    ):
        return "fused_qkvzba_split" in name or (
            "vllm_" in method and "gdn_sequence" in method
        )
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
                "chunkedgateddeltanet",
                "recompute_w_u",
                "chunk_fwd_kernel_o",
            )
        )
    if owner.endswith(("post_attention_norm", "input_norm")) and member.endswith(
        ("attention_residual", "layer_residual")
    ):
        # SGLang uses ``fused_add_rmsnorm`` while vLLM's generated Triton
        # symbol preserves the source-op spelling ``fused_add_rms_norm``.
        # Both explicitly identify one launch owning the add and RMS norm.
        return any(
            token in name
            for token in ("fused_add_rmsnorm", "fused_add_rms_norm")
        )
    if "collective" in owner and member.endswith(
        ("attention_residual", "post_attention_norm", "layer_residual", "input_norm")
    ):
        return any(
            token in cpu_op
            for token in (
                "allreduce_residual_rmsnorm",
                "fused_allreduce_norm",
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
        target_rows = [
            row
            for candidate_owner in owners
            for row in rows_by_owner[candidate_owner]
            if str(row["event_id"])
            in event_ids_by_member_owner[(member, candidate_owner)]
        ]
        target_events_closed = bool(target_rows) and {
            str(row["event_id"]) for row in target_rows
        } <= closed_event_ids
        physical_target_proof = bool(target_rows) and any(
            fusion_target_is_physically_proven(
                row, str(row.get("node") or ""), member
            )
            for row in target_rows
        )
        if not target_events_closed or not physical_target_proof:
            states[member] = {
                "status": "partially_fused",
                "label": (
                    "occurrence fusion proof is incomplete; profile acceptance "
                    "must fail closed"
                ),
                "shared_timing_owners": sorted(owners),
                "event_set_equality_failures": equality_failures,
                "all_owner_events_same_rank_closed": owner_events_closed,
                "all_target_events_same_rank_closed": target_events_closed,
                "physical_target_proof": physical_target_proof,
            }
            continue
        if member in direct_nodes or len(owners) != 1 or equality_failures:
            # The member is direct in some occurrences, or exact target event
            # sets differ by occurrence.  Retain that closure on Timeline but
            # publish neither a profile-aggregate fused state nor copied owner
            # timing.  A target-only leaf remains a structural semantic node.
            if member not in direct_nodes:
                states[member] = {
                    "status": "structural",
                    "label": (
                        "implemented by exact occurrence-scoped shared events; "
                        "no profile-aggregate fused timing is claimed"
                    ),
                    "occurrence_fusion_evidence": {
                        "shared_timing_owners": sorted(owners),
                        "target_event_count": len(target_rows),
                        "all_target_events_same_rank_closed": True,
                        "physical_target_proof": True,
                        "profile_aggregate_event_set_equal": not equality_failures,
                    },
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
        mapping_path, reconciliation_report = build_matched_graph_off_mapping(
            task_root, item, rank, rank_rows[rank]
        )
        mapping_paths[rank] = mapping_path
        reconciliation_reports[str(rank)] = reconciliation_report
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
            "mapping_file": (
                "current/qwen35-complete-profiles/mapping/"
                f"{mapping_path.parent.name}/{mapping_path.name}"
            ),
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
                    "job_id": report["job_id"],
                    "source_phase": report["source_phase"],
                    "selected_forward_kernel_count": report["kernel_count"],
                    "graph_off_semantic_event_count": report[
                        "graph_off_semantic_event_count"
                    ],
                    "graph_off_support_event_count": report[
                        "graph_off_support_event_count"
                    ],
                    "production_semantic_event_count": report[
                        "production_semantic_event_count"
                    ],
                    "closed_production_event_count": report[
                        "closed_production_event_count"
                    ],
                    "typed_unresolved_event_count": report[
                        "typed_unresolved_event_count"
                    ],
                    "production_support_event_count": report[
                        "production_support_event_count"
                    ],
                    "graph_off_support_class_counts": report[
                        "graph_off_support_class_counts"
                    ],
                    "production_support_class_counts": report[
                        "production_support_class_counts"
                    ],
                    "selected_forward_kernel_duration_us": round(
                        report["selected_forward_kernel_duration_us"], 6
                    ),
                    "raw_trace_sha256": report["raw_trace_sha256"],
                    "raw_manifest_sha256": report["raw_manifest_sha256"],
                    "capture_metadata_sha256": report[
                        "capture_metadata_sha256"
                    ],
                    "source_commit": report["source_commit"],
                    "runtime_source_version": report[
                        "runtime_source_version"
                    ],
                    "hardware": report["hardware"],
                    "world_size": report["world_size"],
                    "selected_formal_coordinate": report[
                        "selected_formal_coordinate"
                    ],
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
