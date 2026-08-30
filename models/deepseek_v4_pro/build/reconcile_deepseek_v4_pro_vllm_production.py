#!/usr/bin/env python3
"""Reconcile one vLLM production trace to its same-rank eager contract.

The eager contract is the only semantic authority. Production CUDA Graph
kernels carry no Python stacks, so transfer is restricted to the same one of
122 source-proved attention/feed-forward occurrences. Decode-only graph
dependency copies are admitted solely at exact collective/combine/final-HC
boundaries and remain N:1 members of that timing owner.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable


SOURCE_COMMIT = "dd10e03f95f94edbea1975c67ace3a35ec9a8a40"
MODEL_REVISION = "72e1d3230f6c080a530b0a1d46f8eb4602340597"
LAYER_COUNT = 61
OCCURRENCE_COUNT = 2 * LAYER_COUNT
MHC_PRE_ANCHOR = "mhc_pre_big_fuse_with_norm_tilelang_kernel"
MHC_PRE_GEMM = "sm100_tf32_hc_prenorm_gemm"
MHC_POST = "mhc_post_tilelang_kernel"
GRAPH_DEPENDENCY_COPY = "memcpy32_post"
GRAPH_DEPENDENCY_COPIES = frozenset({GRAPH_DEPENDENCY_COPY, "memcpy128"})


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


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
    """Canonicalize spelling only and preserve templates and shapes."""

    normalized = re.sub(r"^void\s+", "", name.strip().lower())
    normalized = normalized.replace("<unnamed>", "(anonymous namespace)")
    normalized = re.sub(r"\(int\)(?=\d)", "", normalized)
    normalized = re.sub(r"\bconst\s+([\w:]+)", r"\1 const", normalized)
    normalized = re.sub(r"\s*\*\s*", "*", normalized)
    normalized = re.sub(r"\s*,\s*", ",", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def kernel_base(name: str) -> str:
    """Return the shape-preserving function identity."""

    normalized = kernel_exact_identity(name)
    if "<" in normalized:
        normalized = normalized.split("<", 1)[0]
    else:
        normalized = re.sub(r"\([^()]*\)\s*$", "", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def schedule_family(name: str) -> str:
    """Normalize codegen shape/version digits inside an occurrence only."""

    family = re.sub(r"0x[0-9a-f]+", "#", kernel_base(name))
    # FlashInfer can select a different schedule suffix for the same
    # occurrence-local routed-expert GEMM. Preserve the two mathematical BMM
    # forms (gate/up versus down) and normalize only their schedules.
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
    trace: dict[str, Any], *, phase: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    events = trace.get("traceEvents") or []
    kernels = _kernel_rows(trace)
    if not kernels:
        raise ValueError("production trace contains no kernels")

    graph_launches = [
        event
        for event in events
        if event.get("cat") == "cuda_runtime"
        and event.get("ph") == "X"
        and event.get("name") == "cudaGraphLaunch"
    ]
    if phase == "decode":
        if len(graph_launches) != 1:
            raise ValueError(
                f"decode requires exactly one cudaGraphLaunch, got {len(graph_launches)}"
            )
        correlation = (graph_launches[0].get("args") or {}).get("correlation")
        graph_indices = [
            index
            for index, kernel in enumerate(kernels)
            if (kernel.get("args") or {}).get("correlation") == correlation
        ]
        if not graph_indices:
            raise ValueError("cudaGraphLaunch has no correlated device kernels")
        if graph_indices != list(range(graph_indices[0], graph_indices[-1] + 1)):
            raise ValueError("CUDA Graph device body is not one contiguous kernel sequence")
        selected = kernels[graph_indices[0] :]
        selector = {
            "method": "cuda_graph_correlation_plus_complete_runtime_tail",
            "cuda_graph_correlation": correlation,
            "graph_body_kernel_count": len(graph_indices),
            "runtime_tail_kernel_count": len(selected) - len(graph_indices),
        }
    else:
        if graph_launches:
            raise ValueError("prefill timing trace unexpectedly contains cudaGraphLaunch")
        annotations = [
            event
            for event in events
            if event.get("cat") == "gpu_user_annotation"
            and event.get("ph") == "X"
            and str(event.get("name") or "").startswith(
                "execute_context_1(8192)_generation_0(0)"
            )
        ]
        if not annotations:
            raise ValueError("prefill trace lacks exact 8192-token execute-context annotation")
        primary = max(annotations, key=lambda event: float(event.get("dur", 0.0)))
        start = float(primary.get("ts", 0.0))
        selected = [kernel for kernel in kernels if float(kernel.get("ts", 0.0)) >= start]
        if len(selected) != len(kernels):
            raise ValueError("prefill trace contains kernels before the selected exact annotation")
        annotation_end = start + float(primary.get("dur", 0.0))
        selector = {
            "method": "exact_prefill_gpu_annotation_plus_complete_runtime_tail",
            "annotation_name": primary.get("name"),
            "annotation_kernel_count": sum(
                start <= float(kernel.get("ts", 0.0)) <= annotation_end
                for kernel in selected
            ),
            "runtime_tail_kernel_count": sum(
                float(kernel.get("ts", 0.0)) > annotation_end for kernel in selected
            ),
        }

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
    return normalized, selector


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
        if MHC_POST in row["kernel_name"].lower()
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
    """Transfer only equal-multiplicity ordered identities in one scope."""

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


def _assign_dependency(
    row: dict[str, Any],
    *,
    owner: dict[str, Any],
    role: str,
    group_id: str,
    allowed_kernels: frozenset[str] = GRAPH_DEPENDENCY_COPIES,
) -> None:
    if row.get("node") or row.get("kernel_name") not in allowed_kernels:
        raise ValueError("graph dependency copy does not match its exact boundary")
    evidence_ids = list(owner.get("eager_event_ids") or [])
    if not evidence_ids:
        raise ValueError("graph dependency owner lacks eager evidence")
    row.update(
        {
            "node": owner["node"],
            "confidence": "high",
            "attribution_method": "eager_proved_boundary_adjacent_graph_dependency",
            "eager_event_ids": evidence_ids,
            "launch_group_id": group_id,
            "launch_group_role": role,
        }
    )
    for key in ("occurrence_id", "layer_id", "layer_kind", "substage"):
        if owner.get(key) is not None:
            row[key] = owner[key]


def _group_owner_with_dependencies(
    owner: dict[str, Any], *, group_id: str, role: str
) -> None:
    owner["launch_group_id"] = group_id
    owner["launch_group_role"] = role


def _close_decode_graph_dependencies(
    production: list[dict[str, Any]],
    *,
    prefix: tuple[int, int],
    scopes: list[tuple[int, int]],
    suffix: tuple[int, int],
) -> int:
    assigned = 0
    prefix_rows = production[slice(*prefix)]
    embedding_index = next(
        (
            index
            for index, row in enumerate(prefix_rows)
            if row.get("node") == "top.tp_embedding_output_collective"
        ),
        None,
    )
    if embedding_index is None:
        raise ValueError("production prefix lacks eager-proved embedding collective")
    if not (
        embedding_index > 0
        and embedding_index + 1 < len(prefix_rows)
        and prefix_rows[embedding_index - 1]["kernel_name"]
        in GRAPH_DEPENDENCY_COPIES
        and prefix_rows[embedding_index + 1]["kernel_name"]
        in GRAPH_DEPENDENCY_COPIES
    ):
        raise ValueError("embedding collective graph-dependency pattern changed")
    embedding_owner = prefix_rows[embedding_index]
    group_id = "top.tp_embedding_output_collective:production_graph_group"
    _group_owner_with_dependencies(
        embedding_owner, group_id=group_id, role="collective_kernel"
    )
    for index, role in (
        (embedding_index - 1, "pre_collective_dependency"),
        (embedding_index + 1, "post_collective_dependency"),
    ):
        _assign_dependency(
            prefix_rows[index], owner=embedding_owner, role=role, group_id=group_id
        )
        assigned += 1

    for scope_id, (start, end) in enumerate(scopes):
        rows = production[start:end]
        occurrence_id = (
            f"layer_{scope_id // 2:02d}."
            f"{'attention' if scope_id % 2 == 0 else 'feed_forward'}"
        )
        copies = [
            index
            for index, row in enumerate(rows)
            if row.get("kernel_name") in GRAPH_DEPENDENCY_COPIES
        ]
        if scope_id % 2 == 0:
            collective_indices = [
                index
                for index, row in enumerate(rows)
                if str(row.get("node") or "").endswith("output_collective")
            ]
            if len(collective_indices) != 1:
                raise ValueError(f"{occurrence_id}: attention collective is not unique")
            collective_index = collective_indices[0]
            if copies != [collective_index - 1, collective_index + 1]:
                raise ValueError(f"{occurrence_id}: attention graph-copy pattern changed")
            owner = rows[collective_index]
            group_id = f"{occurrence_id}:{owner['node']}:production_graph_group"
            _group_owner_with_dependencies(
                owner, group_id=group_id, role="collective_kernel"
            )
            for copy_index, role in zip(
                copies, ("pre_collective_dependency", "post_collective_dependency")
            ):
                _assign_dependency(
                    rows[copy_index], owner=owner, role=role, group_id=group_id
                )
                assigned += 1
        else:
            combine_indices = [
                index
                for index, row in enumerate(rows)
                if row.get("node") == "moe.combine"
            ]
            collective_indices = [
                index
                for index, row in enumerate(rows)
                if row.get("node") == "moe.tp_moe_output_collective"
            ]
            if len(combine_indices) != 1 or len(collective_indices) != 1:
                raise ValueError(f"{occurrence_id}: MoE combine/collective is not unique")
            combine_index = combine_indices[0]
            collective_index = collective_indices[0]
            expected = [combine_index - 1, collective_index - 1, collective_index + 1]
            if copies != expected:
                raise ValueError(f"{occurrence_id}: MoE graph-copy pattern changed")
            combine_owner = rows[combine_index]
            combine_group = f"{occurrence_id}:moe.combine:production_graph_group"
            _group_owner_with_dependencies(
                combine_owner, group_id=combine_group, role="semantic_kernel"
            )
            _assign_dependency(
                rows[copies[0]],
                owner=combine_owner,
                role="pre_combine_dependency",
                group_id=combine_group,
            )
            assigned += 1
            collective_owner = rows[collective_index]
            collective_group = (
                f"{occurrence_id}:moe.tp_moe_output_collective:production_graph_group"
            )
            _group_owner_with_dependencies(
                collective_owner,
                group_id=collective_group,
                role="collective_kernel",
            )
            for copy_index, role in (
                (copies[1], "pre_collective_dependency"),
                (copies[2], "post_collective_dependency"),
            ):
                _assign_dependency(
                    rows[copy_index],
                    owner=collective_owner,
                    role=role,
                    group_id=collective_group,
                )
                assigned += 1

    suffix_rows = production[slice(*suffix)]
    final_hc_index = next(
        (
            index
            for index, row in enumerate(suffix_rows)
            if row.get("node") == "final_hc_read.read"
        ),
        None,
    )
    if (
        final_hc_index != 1
        or suffix_rows[0]["kernel_name"] not in GRAPH_DEPENDENCY_COPIES
    ):
        raise ValueError("final-HC graph-dependency pattern changed")
    final_owner = suffix_rows[final_hc_index]
    final_group = "top.final_hc_read:production_graph_group"
    _group_owner_with_dependencies(
        final_owner, group_id=final_group, role="semantic_kernel"
    )
    _assign_dependency(
        suffix_rows[0],
        owner=final_owner,
        role="pre_final_hc_dependency",
        group_id=final_group,
        allowed_kernels=GRAPH_DEPENDENCY_COPIES,
    )
    return assigned + 1


def reconcile(
    trace: dict[str, Any],
    source: list[dict[str, Any]],
    *,
    phase: str,
    batch_size: int,
    rank: int,
    job_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    production, selector = select_production_window(trace, phase=phase)
    source_prefix, source_occurrences, source_suffix = _source_occurrences(source)
    prefix, scopes, suffix, mhc_path = _production_scopes(production)
    _transfer_scope(
        source_prefix,
        production[slice(*prefix)],
        method_prefix="top_prefix",
    )
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
    _transfer_scope(
        source_suffix,
        production[slice(*suffix)],
        method_prefix="top_suffix",
    )
    graph_dependency_count = 0
    if phase == "decode":
        graph_dependency_count = _close_decode_graph_dependencies(
            production, prefix=prefix, scopes=scopes, suffix=suffix
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
    mapped_us = sum(
        float(row["dur_us"]) for row in production if row.get("node")
    )
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
        "schema_version": "deepseek-v4-pro-vllm-production-reconciliation.v1",
        "ok": not errors,
        "errors": errors,
        "framework": "vllm",
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
        "graph_dependency_kernel_count": graph_dependency_count,
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
    parser.add_argument("--output-events", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    if f"rank{args.rank}." not in args.trace.name:
        raise ValueError("trace filename rank differs from --rank")
    trace = load_trace(args.trace)
    source = load_jsonl(args.eager_contract)
    events, report = reconcile(
        trace,
        source,
        phase=args.phase,
        batch_size=args.batch_size,
        rank=args.rank,
        job_id=args.job_id,
    )
    report["trace"] = {
        "path": str(args.trace),
        "sha256": sha256_file(args.trace),
    }
    report["eager_contract"] = {
        "path": str(args.eager_contract),
        "sha256": sha256_file(args.eager_contract),
    }
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
