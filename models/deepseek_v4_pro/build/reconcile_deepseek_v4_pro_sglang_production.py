#!/usr/bin/env python3
"""Reconcile one SGLang production trace to its same-rank eager contract.

The graph-off eager contract is the only semantic authority.  Production
kernels are selected by the exact SGLang scheduler GPU annotation and then
matched only within the same top-level or one of 122 layer/substage scopes.
Any CUDA-Graph-only dependency remains unmapped until an exact adjacent owner
contract is proved and encoded explicitly.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.deepseek_v4_pro.build.validate_deepseek_v4_pro_sglang_formal_window import (
    validate_formal_window,
)


SOURCE_COMMIT = "71de97b264b04dcd514cf904003028aefe9775c8"
MODEL_REVISION = "72e1d3230f6c080a530b0a1d46f8eb4602340597"
LAYER_COUNT = 61
OCCURRENCE_COUNT = 2 * LAYER_COUNT
MHC_PRE_ANCHOR = "mhc_pre_big_fuse_with_norm_tilelang_kernel"
MHC_PRE_GEMM = "sm100_tf32_hc_prenorm_gemm"
MHC_POST_NAMES = ("mhc_post_tilelang_kernel", "mhc_fused_post_pre_fma_tilelang_kernel")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_trace(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as source:
        return json.load(source)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def schedule_family(name: str) -> str:
    family = re.sub(r"0x[0-9a-f]+", "#", kernel_base(name))
    if family.startswith("bmm_mxe4m3_mxe2m1mxe4m3_fp32"):
        return "bmm_mxe4m3_routed_gate_up"
    if family.startswith("bmm_bfloat16_mxe2m1mxe4m3_fp32"):
        return "bmm_bfloat16_routed_down"
    family = re.sub(r"\d+", "#", family)
    if family.startswith("bmm_"):
        family = re.sub(r"_bn_.*?_rgtma_", "_bn_<schedule>_rgtma_", family)
    return family


def _kernel_rows(trace: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        (
            event
            for event in trace.get("traceEvents") or []
            if event.get("cat") == "kernel" and event.get("ph") == "X"
        ),
        key=lambda event: float(event.get("ts", 0.0)),
    )


def select_production_window(
    trace: dict[str, Any],
    *,
    phase: str,
    batch_size: int,
    eager_kernel_names: list[str] | None = None,
    graph_launch_index: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    events = trace.get("traceEvents") or []
    annotation_name = (
        f"step[EXTEND bs={batch_size} toks=8192]"
        if phase == "prefill"
        else f"step[DECODE bs={batch_size}]"
    )
    annotations = [
        event
        for event in events
        if event.get("cat") == "gpu_user_annotation"
        and event.get("ph") == "X"
        and event.get("name") == annotation_name
    ]
    if len(annotations) > 1:
        raise ValueError(
            f"allows at most one {annotation_name!r} GPU annotation, "
            f"got {len(annotations)}"
        )
    graph_launches = sorted(
        [
        event
        for event in events
        if event.get("cat") == "cuda_runtime"
        and event.get("ph") == "X"
        and event.get("name") == "cudaGraphLaunch"
        ],
        key=lambda event: float(event.get("ts", 0.0)),
    )
    expected_graph_launches = (
        graph_launch_index + 1 if phase == "decode" else graph_launch_index
    )
    if len(graph_launches) != expected_graph_launches:
        raise ValueError(
            f"{phase} requires {expected_graph_launches} cudaGraphLaunch, "
            f"got {len(graph_launches)}"
        )
    graph_correlation = (
        (graph_launches[graph_launch_index].get("args") or {}).get("correlation")
        if phase == "decode" and graph_launches
        else None
    )
    if phase == "decode" and graph_launches and graph_correlation is None:
        raise ValueError("cudaGraphLaunch lacks a correlation ID")

    kernels = _kernel_rows(trace)
    if annotations:
        annotation = annotations[0]
        start = float(annotation["ts"])
        stop = start + float(annotation["dur"])
        selected = [
            kernel
            for kernel in kernels
            if start <= float(kernel.get("ts", 0.0))
            and float(kernel.get("ts", 0.0))
            + float(kernel.get("dur", 0.0))
            <= stop
        ]
        method = "exact_sglang_scheduler_gpu_annotation"
        annotation_duration_us: float | None = float(annotation["dur"])
    elif phase == "decode":
        # GPU-only production traces intentionally omit CPU/user annotations.
        # The single CUDA graph launch is the exact model-step boundary: only
        # device kernels carrying its correlation ID are production graph body.
        selected = [
            kernel
            for kernel in kernels
            if (kernel.get("args") or {}).get("correlation") == graph_correlation
        ]
        method = "exact_cuda_graph_correlation"
        annotation_duration_us = None
    else:
        # An exactly validated one-step prefill timing trace has no graph launch
        # and no user annotations when activities=[GPU].  Recover the exact
        # scheduler step only when the same-rank eager ordered identity sequence
        # occurs as one unique contiguous device-kernel window.
        if not eager_kernel_names:
            raise ValueError(
                "GPU-only prefill selection requires same-rank eager kernel identities"
            )
        expected = [kernel_exact_identity(name) for name in eager_kernel_names]
        observed = [kernel_exact_identity(str(kernel.get("name") or "")) for kernel in kernels]
        candidates = [
            start
            for start in range(len(observed) - len(expected) + 1)
            if observed[start : start + len(expected)] == expected
        ]
        if len(candidates) != 1:
            raise ValueError(
                "prefill eager ordered sequence must identify exactly one contiguous window, "
                f"got {len(candidates)}"
            )
        window_start = candidates[0]
        selected = kernels[window_start : window_start + len(expected)]
        method = "exact_same_rank_eager_ordered_kernel_window"
        annotation_duration_us = None
    if not selected:
        raise ValueError("selected SGLang production window contains no kernels")

    graph_kernel_count = sum(
        (kernel.get("args") or {}).get("correlation") == graph_correlation
        for kernel in selected
    ) if graph_correlation is not None else 0
    normalized = [
        {
            "event_id": f"p_{index:06d}",
            "kernel_name": str(kernel.get("name") or ""),
            "ts_us": float(kernel.get("ts", 0.0)),
            "dur_us": float(kernel.get("dur", 0.0)),
            "stream": (kernel.get("args") or {}).get("stream"),
            "device": (kernel.get("args") or {}).get("device"),
            "correlation": (kernel.get("args") or {}).get("correlation"),
            "node": None,
        }
        for index, kernel in enumerate(selected)
    ]
    return normalized, {
        "method": method,
        "annotation_name": annotation_name if annotations else None,
        "annotation_duration_us": annotation_duration_us,
        "cuda_graph_correlation": graph_correlation,
        "graph_body_kernel_count": graph_kernel_count,
        "runtime_tail_kernel_count": len(selected) - graph_kernel_count,
        "graph_launch_index": graph_launch_index if phase == "decode" else None,
        "profile_priming_launch_count": (
            graph_launch_index if phase == "decode" else len(graph_launches)
        ),
    }


def _source_occurrences(
    source: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    occurrence_indices = [
        index for index, row in enumerate(source) if row.get("occurrence_id")
    ]
    if not occurrence_indices:
        raise ValueError("eager contract contains no layer occurrences")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source:
        if row.get("occurrence_id"):
            groups[str(row["occurrence_id"])].append(row)
    expected = {
        f"layer_{layer:02d}.{substage}"
        for layer in range(LAYER_COUNT)
        for substage in ("attention", "feed_forward")
    }
    if set(groups) != expected:
        raise ValueError("eager contract does not close all 122 layer occurrences")
    return (
        source[: occurrence_indices[0]],
        dict(groups),
        source[occurrence_indices[-1] + 1 :],
    )


def _production_scopes(
    production: list[dict[str, Any]],
) -> tuple[tuple[int, int], list[tuple[int, int]], tuple[int, int], str]:
    anchors = [
        index
        for index, row in enumerate(production)
        if MHC_PRE_ANCHOR in row["kernel_name"].lower()
    ]
    if len(anchors) != OCCURRENCE_COUNT:
        raise ValueError(
            f"production requires {OCCURRENCE_COUNT} mHC-pre anchors, got {len(anchors)}"
        )
    pre_gemms = [
        index
        for index, row in enumerate(production)
        if MHC_PRE_GEMM in row["kernel_name"].lower()
    ]
    if len(pre_gemms) == OCCURRENCE_COUNT:
        starts = [anchor - 1 for anchor in anchors]
        mhc_path = "separate_post_pre"
    elif len(pre_gemms) == 1:
        starts = [anchors[0] - 1, *anchors[1:]]
        mhc_path = "fused_post_pre"
    else:
        raise ValueError(
            "production mHC schedule is neither exact separate nor fused path: "
            f"pre_gemms={len(pre_gemms)}"
        )
    if any(start < 0 for start in starts) or starts != sorted(starts):
        raise ValueError("invalid production occurrence starts")
    post_indices = [
        index
        for index, row in enumerate(production)
        if any(name in row["kernel_name"].lower() for name in MHC_POST_NAMES)
    ]
    if not post_indices:
        raise ValueError("production trace lacks final mHC-post boundary")
    layer_end = post_indices[-1] + 1
    scopes = [
        (start, starts[index + 1] if index + 1 < len(starts) else layer_end)
        for index, start in enumerate(starts)
    ]
    return (0, starts[0]), scopes, (layer_end, len(production)), mhc_path


def _copy_source_metadata(
    production: dict[str, Any], source: dict[str, Any], *, method: str
) -> None:
    if production.get("node"):
        return
    production.update(
        {
            "node": source["selected_node"],
            "confidence": "high",
            "attribution_method": method,
            "eager_event_ids": [source["event_id"]],
        }
    )
    for key in (
        "occurrence_id",
        "layer_id",
        "layer_kind",
        "substage",
        "timing_role",
        "fused_semantic_nodes",
        "fused_nonowner_occurrence_id",
        "launch_group_id",
        "launch_group_role",
        "support_class",
        "support_reason",
    ):
        if source.get(key) is not None:
            production[key] = source[key]


def _transfer_scope(
    source: list[dict[str, Any]],
    production: list[dict[str, Any]],
    *,
    method_prefix: str,
) -> None:
    key_fns: tuple[tuple[Callable[[str], str], str], ...] = (
        (kernel_exact_identity, "exact_identity"),
        (kernel_base, "function_identity"),
        (schedule_family, "shape_family"),
    )
    for key_fn, suffix in key_fns:
        source_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        production_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in source:
            source_groups[key_fn(str(row.get("kernel_name") or ""))].append(row)
        for row in production:
            production_groups[key_fn(str(row.get("kernel_name") or ""))].append(row)
        for identity, source_group in source_groups.items():
            production_group = production_groups.get(identity) or []
            if (
                not source_group
                or len(source_group) != len(production_group)
                or not all(row.get("selected_node") for row in source_group)
            ):
                continue
            for source_row, production_row in zip(source_group, production_group):
                _copy_source_metadata(
                    production_row,
                    source_row,
                    method=f"{method_prefix}_{suffix}_ordered_sequence",
                )


def _close_decode_graph_prefix(
    source_prefix: list[dict[str, Any]],
    production: list[dict[str, Any]],
    *,
    prefix: tuple[int, int],
    expected_dependency_names: tuple[str, ...],
) -> int:
    """Close the exact SGLang graph-only metadata-copy prefix."""

    start, stop = prefix
    rows = production[start:stop]
    dependency_indices = [
        index
        for index, row in enumerate(rows)
        if row["kernel_name"].lower() in {"memcpy32_post", "memcpy128"}
        and not row.get("node")
    ]
    observed_dependency_names = tuple(
        sorted(rows[index]["kernel_name"].lower() for index in dependency_indices)
    )
    if observed_dependency_names != tuple(sorted(expected_dependency_names)):
        raise ValueError(
            "decode graph prefix dependency multiset mismatch: "
            f"expected={sorted(expected_dependency_names)} "
            f"observed={list(observed_dependency_names)}"
        )
    for index in dependency_indices:
        previous = next(
            (rows[cursor] for cursor in range(index - 1, -1, -1) if rows[cursor].get("node")),
            None,
        )
        following = next(
            (rows[cursor] for cursor in range(index + 1, len(rows)) if rows[cursor].get("node")),
            None,
        )
        if not previous or not following or {
            previous.get("node"),
            following.get("node"),
        } != {"top.runtime_support"}:
            raise ValueError("graph metadata copy is not bounded by exact runtime-support evidence")
        eager_ids = list(
            dict.fromkeys(
                [
                    *(previous.get("eager_event_ids") or []),
                    *(following.get("eager_event_ids") or []),
                ]
            )
        )
        if not eager_ids:
            raise ValueError("graph metadata-copy neighbors lack eager evidence IDs")
        rows[index].update(
            {
                "node": "top.runtime_support",
                "confidence": "high",
                "attribution_method": "decode_graph_prefix_metadata_copy_exact_bounded_neighbors",
                "eager_event_ids": eager_ids,
                "timing_role": "owner",
                "support_class": "graph_dependency",
                "support_reason": "CUDA graph metadata-state copy bounded by exact same-rank eager runtime-support operations",
            }
        )

    remaining = [row for row in rows if not row.get("node")]
    if len(remaining) != 1:
        raise ValueError(
            f"decode graph prefix expected one ordered eager copy after dependency closure, got {len(remaining)}"
        )
    target = remaining[0]
    if rows.index(target) + 1 >= len(rows) or rows[rows.index(target) + 1].get("node") != "top.embedding":
        raise ValueError("remaining decode graph prefix copy is not adjacent to top.embedding")
    candidates = [
        row
        for row in source_prefix
        if row.get("selected_node") == "top.runtime_support"
        and kernel_exact_identity(str(row.get("kernel_name") or ""))
        == kernel_exact_identity(target["kernel_name"])
    ]
    if not candidates:
        raise ValueError("eager prefix lacks the exact pre-embedding runtime copy")
    _copy_source_metadata(
        target,
        candidates[-1],
        method="top_prefix_pre_embedding_exact_ordered_copy",
    )
    return len(dependency_indices)


def reconcile(
    trace: dict[str, Any],
    source: list[dict[str, Any]],
    *,
    phase: str,
    batch_size: int,
    rank: int,
    job_id: str,
    graph_launch_index: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    production, selector = select_production_window(
        trace,
        phase=phase,
        batch_size=batch_size,
        eager_kernel_names=[str(row.get("kernel_name") or "") for row in source],
        graph_launch_index=graph_launch_index,
    )
    source_prefix, source_occurrences, source_suffix = _source_occurrences(source)
    prefix, scopes, suffix, mhc_path = _production_scopes(production)
    _transfer_scope(source_prefix, production[slice(*prefix)], method_prefix="top_prefix")
    for scope_id, (start, end) in enumerate(scopes):
        occurrence_id = (
            f"layer_{scope_id // 2:02d}."
            f"{'attention' if scope_id % 2 == 0 else 'feed_forward'}"
        )
        _transfer_scope(
            source_occurrences[occurrence_id],
            production[start:end],
            method_prefix=f"{occurrence_id}_bounded",
        )
    _transfer_scope(source_suffix, production[slice(*suffix)], method_prefix="top_suffix")

    graph_dependencies = 0
    if phase == "decode":
        graph_dependencies = _close_decode_graph_prefix(
            source_prefix,
            production,
            prefix=prefix,
            expected_dependency_names=(
                ("memcpy32_post", "memcpy32_post", "memcpy32_post")
                if batch_size == 1
                else ("memcpy32_post", "memcpy128")
                if batch_size == 256
                else ("memcpy32_post", "memcpy32_post")
            ),
        )

    unmapped = [row for row in production if not row.get("node")]
    errors: list[str] = []
    if unmapped:
        errors.append(
            f"production mapping incomplete: {len(unmapped)} kernels, "
            f"first={unmapped[0]['kernel_name']}"
        )
    occurrence_ids = {
        str(row["occurrence_id"])
        for row in production
        if row.get("occurrence_id")
    }
    expected_occurrences = {
        f"layer_{layer:02d}.{substage}"
        for layer in range(LAYER_COUNT)
        for substage in ("attention", "feed_forward")
    }
    if occurrence_ids != expected_occurrences:
        errors.append("production occurrence closure is incomplete")
    for boundary in (
        "top.tp_embedding_output_collective",
        "top.tp_logits_collective",
    ):
        if not any(row.get("node") == boundary for row in production):
            errors.append(f"production lacks {boundary}")
    if any(not row.get("eager_event_ids") for row in production if row.get("node")):
        errors.append("one or more production kernels lack eager evidence IDs")

    total_us = sum(float(row["dur_us"]) for row in production)
    mapped_us = sum(float(row["dur_us"]) for row in production if row.get("node"))
    selected_start_us = min(float(row["ts_us"]) for row in production)
    selected_end_us = max(
        float(row["ts_us"]) + float(row["dur_us"]) for row in production
    )
    fingerprint_payload = [
        {
            "kernel_family": schedule_family(row["kernel_name"]),
            "node": row.get("node"),
            "occurrence_id": row.get("occurrence_id"),
            "launch_group_id": row.get("launch_group_id"),
        }
        for row in production
    ]
    fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_payload, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    report = {
        "schema_version": "deepseek-v4-pro-sglang-production-reconciliation.v1",
        "ok": not errors,
        "errors": errors,
        "framework": "sglang",
        "source_commit": SOURCE_COMMIT,
        "model_revision": MODEL_REVISION,
        "phase": phase,
        "global_batch_size": batch_size,
        "rank": rank,
        "job_id": str(job_id),
        "window_selector": selector,
        "mhc_implementation_path": mhc_path,
        "kernel_count": len(production),
        "mapped_kernel_count": len(production) - len(unmapped),
        "mapped_kernel_count_ratio": (
            (len(production) - len(unmapped)) / len(production)
            if production
            else 0.0
        ),
        "total_kernel_us": total_us,
        "selected_window_start_us": selected_start_us,
        "selected_window_end_us": selected_end_us,
        "selected_wall_elapsed_us": selected_end_us - selected_start_us,
        "timing_semantics": {
            "top_level_runtime": "selected_wall_elapsed_us",
            "node_kernel_time": "sum_of_device_kernel_durations_may_overlap_across_streams",
        },
        "mapped_kernel_us": mapped_us,
        "mapped_kernel_duration_ratio": mapped_us / total_us if total_us else 0.0,
        "occurrence_count": len(occurrence_ids),
        "graph_dependency_kernel_count": graph_dependencies,
        "node_counts": dict(Counter(str(row.get("node")) for row in production)),
        "attribution_method_counts": dict(
            Counter(str(row.get("attribution_method")) for row in production)
        ),
        "ordered_reconciliation_fingerprint": fingerprint,
    }
    return production, report


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--eager-contract", type=Path, required=True)
    parser.add_argument("--phase", choices=("prefill", "decode"), required=True)
    parser.add_argument("--batch-size", type=int, choices=(1, 16, 64, 256), required=True)
    parser.add_argument("--rank", type=int, choices=range(8), required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument(
        "--graph-launch-index",
        type=int,
        choices=(0, 1),
        default=0,
        help="select the formal launch after any profiler-activation priming launch",
    )
    parser.add_argument("--output-events", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--run-validation", type=Path)
    parser.add_argument("--client-artifact", type=Path)
    parser.add_argument("--scheduler-log", type=Path)
    parser.add_argument("--baseline-selection", type=Path)
    args = parser.parse_args()
    if not re.search(rf"TP-{args.rank}(?:\.|-)", args.trace.name):
        raise ValueError("trace filename rank differs from --rank")
    events, report = reconcile(
        load_trace(args.trace),
        load_jsonl(args.eager_contract),
        phase=args.phase,
        batch_size=args.batch_size,
        rank=args.rank,
        job_id=args.job_id,
        graph_launch_index=args.graph_launch_index,
    )
    report["trace"] = {
        "path": str(args.trace),
        "sha256": sha256_file(args.trace),
    }
    report["eager_contract"] = {
        "path": str(args.eager_contract),
        "sha256": sha256_file(args.eager_contract),
    }
    if args.phase == "decode":
        required_formal_inputs = {
            "client_artifact": args.client_artifact,
            "scheduler_log": args.scheduler_log,
            "baseline_selection": args.baseline_selection,
        }
        missing_formal_inputs = [
            name
            for name, path in required_formal_inputs.items()
            if path is None or not path.is_file()
        ]
        independently_validated_gate = None
        if missing_formal_inputs:
            report["errors"].append(
                "decode reconciliation lacks formal-window inputs: "
                + ", ".join(missing_formal_inputs)
            )
        else:
            try:
                independently_validated_gate = validate_formal_window(
                    client=load_json(args.client_artifact),
                    scheduler_log=args.scheduler_log.read_text(errors="replace"),
                    baseline=load_json(args.baseline_selection),
                    concurrency=args.batch_size,
                )
            except ValueError as exc:
                report["errors"].append(f"formal-window validation failed: {exc}")
            report["formal_window_inputs"] = {
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in required_formal_inputs.items()
            }
        if args.run_validation is None or not args.run_validation.is_file():
            report["errors"].append("decode reconciliation lacks run validation")
        else:
            validation = load_json(args.run_validation)
            gate = (validation.get("throughput_gate") or {}).get(
                str(args.batch_size)
            )
            if validation.get("status") != "pass":
                report["errors"].append("decode run validation did not pass")
            if not gate:
                report["errors"].append("decode run lacks formal-step throughput gate")
            else:
                if gate != independently_validated_gate:
                    report["errors"].append(
                        "retained throughput gate differs from direct source-artifact validation"
                    )
                target = gate.get("formal_target") or {}
                if gate.get("profile_start_step") != gate.get("formal_target_step"):
                    report["errors"].append(
                        "formal scheduler step does not match the synchronized second-launch coordinate"
                    )
                if float(target.get("throughput_token_s") or 0.0) < float(
                    gate.get("minimum_accepted_throughput_token_s") or 0.0
                ):
                    report["errors"].append("formal step is a profile-start throughput collapse")
            report["run_validation"] = {
                "path": str(args.run_validation),
                "sha256": sha256_file(args.run_validation),
                "status": validation.get("status"),
            }
            report["formal_step_throughput_gate"] = gate
    report["ok"] = not report["errors"]
    write_jsonl(args.output_events, events)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        f"ok={report['ok']} phase={args.phase} bs={args.batch_size} "
        f"rank={args.rank} kernels={report['kernel_count']} "
        f"mapped={report['mapped_kernel_count']}"
    )
    for error in report["errors"]:
        print(f"error: {error}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
