#!/usr/bin/env python3
"""Build a provenance-preserving Qwen3.8-Flash-Next CUDA-Graph decode overlay.

The eager trace supplies source/IR bindings and collective ordering.  CUDA
Graph traces supply timing.  Every replay kernel is attributed, and all node
times come from one coherent critical reference rank rather than mixing maxima
from different ranks.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.qwen38_flash_next.build.build_qwen38_flash_next_cudagraph_profile import (
    direct_kernel_mapping as common_direct_kernel_mapping,
)
from models.qwen38_flash_next.build.qwen38_flash_next_decode_attribution import (
    collective_kind as canonical_collective_kind,
    default_node_states,
    direct_kernel_mapping as canonical_direct_kernel_mapping,
    interval_union_us,
    map_decode_step,
    metrics_for_rank as attributed_metrics_for_rank,
)
from models.common.timeline_artifact import (  # noqa: E402
    attach_eager_stack_evidence,
    build_timeline_artifact,
    write_timeline_artifact,
)


SOURCE_COMMIT = "f90a941aa6ff71ac3bd7d40b8daccdf5bd914af0"
MODEL_REVISION = "b151fd157ff99b63198ab8558432f0bf43e14d58"
PROFILE_STEPS = 6
SELECTED_STEPS = 5
STEP_NAME = re.compile(r"^step\[(DECODE|IDLE) bs=(\d+)\]$")
DP_RANK = re.compile(r"-DP-(\d+)(?:-|\.)")
TP_RANK = re.compile(r"-TP-(\d+)(?:-|\.)")

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
    "tp4_flashinfer_gdn": {
        "execution_path_id": "tp_only",
        "implementation_id": "sglang_f90a941aa_flashinfer_gdn",
        "label": "pure TP4 · FlashInfer GDN",
        "tp_size": 4,
        "dp_size": 1,
        "ep_size": 1,
        "gdn_backend": "flashinfer_bf16",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", choices=tuple(CONFIGS), required=True)
    parser.add_argument("--traces", type=Path, nargs=4, required=True)
    parser.add_argument("--eager-mapping", type=Path, required=True)
    parser.add_argument("--rounds", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--batch-size", type=int, choices=(1, 16, 64, 256), required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--max-prefill-tokens", type=int, default=8192)
    parser.add_argument("--chunked-prefill-size", type=int, default=8192)
    parser.add_argument(
        "--admission-control",
        choices=("stock", "prefill-first-until-local-target"),
        default="stock",
    )
    parser.add_argument("--source-patch-sha256", default=None)
    parser.add_argument("--source-commit", default=SOURCE_COMMIT)
    parser.add_argument(
        "--profile-tag",
        default="",
        help="Optional stable suffix used to install paired implementation overlays.",
    )
    parser.add_argument("--implementation-id", default=None)
    parser.add_argument("--implementation-label", default=None)
    parser.add_argument("--reference-rank", type=int, choices=(0, 1, 2, 3))
    parser.add_argument("--output-profile", type=Path, required=True)
    parser.add_argument("--output-analysis", type=Path, required=True)
    parser.add_argument("--node", default="")
    return parser.parse_args()


def load_trace(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as trace_file:
        return (json.load(trace_file).get("traceEvents") or [])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def trace_rank(path: Path, dp_size: int) -> int:
    pattern = DP_RANK if dp_size > 1 else TP_RANK
    match = pattern.search(path.name)
    if match is None:
        raise ValueError(f"cannot parse rank from {path.name}")
    return int(match.group(1))


def load_formal_round(path: Path, batch_size: int) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    matches = [
        row
        for row in rows
        if row.get("round") == "formal-1"
        and row.get("global_batch_size", row.get("batch_size")) == batch_size
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one formal row for global bs={batch_size}, got {len(matches)}"
        )
    return matches[0]


def select_reference_rank(
    rank_steps_ms: dict[int, list[float]], requested: int | None
) -> int:
    """Select one coherent rank, with an explicit override for paired A/B data."""

    if requested is not None:
        if requested not in rank_steps_ms:
            raise ValueError(f"reference rank {requested} is absent from the traces")
        return requested
    return max(
        rank_steps_ms,
        key=lambda rank: statistics.fmean(rank_steps_ms[rank]),
    )


def merged_gpu_steps(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge per-stream copies of each model-step annotation.

    Piecewise/overlapped graphs clone the same six step labels onto multiple
    GPU streams. A single track can cover only a subset of layers, so the
    complete model-step boundary is the union of the aligned track ranges.
    """

    tracks: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if (
            event.get("cat") == "gpu_user_annotation"
            and event.get("ph") == "X"
            and STEP_NAME.fullmatch(str(event.get("name", "")))
        ):
            tracks[(event.get("pid"), event.get("tid"))].append(event)
    if not tracks:
        raise ValueError("trace has no DECODE/IDLE GPU step annotations")
    invalid = {str(track): len(items) for track, items in tracks.items() if len(items) != PROFILE_STEPS}
    if invalid:
        raise ValueError(f"GPU step tracks are not six steps: {invalid}")
    ordered_tracks = [
        sorted(items, key=lambda item: float(item.get("ts", 0.0)))
        for items in tracks.values()
    ]
    merged = []
    for index in range(PROFILE_STEPS):
        copies = [items[index] for items in ordered_tracks]
        names = {str(item.get("name", "")) for item in copies}
        if len(names) != 1:
            raise ValueError(
                f"GPU track labels disagree at step {index}: {sorted(names)}"
            )
        start = min(float(item.get("ts", 0.0)) for item in copies)
        end = max(
            float(item.get("ts", 0.0)) + float(item.get("dur", 0.0))
            for item in copies
        )
        merged.append(
            {
                "name": names.pop(),
                "ts": start,
                "dur": end - start,
                "merged_gpu_track_count": len(copies),
            }
        )
    kinds = {STEP_NAME.fullmatch(str(item["name"])).group(1) for item in merged}
    if len(kinds) != 1:
        raise ValueError(f"mixed formal GPU step kinds: {[item['name'] for item in merged]}")
    return merged


def kernels_in_step(
    events: list[dict[str, Any]], step: dict[str, Any]
) -> list[dict[str, Any]]:
    start = float(step.get("ts", 0.0))
    end = start + float(step.get("dur", 0.0))
    return sorted(
        (
            event
            for event in events
            if event.get("cat") == "kernel"
            and event.get("ph") == "X"
            and start <= float(event.get("ts", 0.0)) <= end
        ),
        key=lambda event: float(event.get("ts", 0.0)),
    )


def collective_kind(name: str) -> str | None:
    canonical = canonical_collective_kind(name)
    return {"all_reduce": "reduce", "all_gather": "gather"}.get(
        canonical, canonical
    )


def eager_collective_template(path: Path) -> list[tuple[str, str]]:
    template = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        kind = collective_kind(str(row.get("kernel_name", "")))
        if kind is None:
            continue
        node = row.get("selected_node")
        if node is None and kind == "gather":
            node = "top.tp_logits_collective"
        if not node:
            raise ValueError(f"eager collective lacks an IR mapping: {row}")
        template.append((kind, str(node)))
    if not template:
        raise ValueError("eager mapping contains no collectives")
    # Some eager captures contain several identical decode iterations.  Keep
    # the shortest exact period; this is deterministic and rejects drift.
    for period in range(1, len(template) + 1):
        if len(template) % period == 0 and template == template[:period] * (
            len(template) // period
        ):
            return template[:period]
    return template


def direct_kernel_mapping(name: str) -> tuple[str | None, str | None]:
    lowered = name.lower()
    if "deep_ep::" in lowered and "dispatch<" in lowered:
        return "moe.deepep_dispatch", "DeepEP dispatch"
    if "deep_ep::" in lowered and "combine<" in lowered:
        return "moe.deepep_combine", "DeepEP combine"
    if "deep_gemm::" in lowered:
        return "moe.routed_experts", "DeepGEMM expert GEMM"
    if "gdn_decode_bf16state" in lowered or "gdn_wide_vec_kernel" in lowered:
        return "linear_attention.delta_rule", "FlashInfer GDN recurrence"
    node, label = canonical_direct_kernel_mapping(name)
    if node is not None:
        return node, label
    return common_direct_kernel_mapping(name)


def signature_counts(kernels: list[dict[str, Any]]) -> dict[str, int]:
    names = [str(kernel.get("name", "")).lower() for kernel in kernels]

    def count(*signatures: str) -> int:
        return sum(any(signature in name for signature in signatures) for name in names)

    return {
        "split_pack": count("fused_qkvzba_split"),
        "causal_conv": count("causal_conv1d_update"),
        "delta_rule": count(
            "fused_recurrent_gated_delta_rule_packed_decode",
            "gdn_decode_bf16state",
            "gdn_wide_vec_kernel",
        ),
        "gated_norm": count("_layer_norm_fwd_1pass_kernel"),
        "qsa_attention": count("fmhasm100fkernel_qkv"),
        "deepep_dispatch": count("deep_ep::internode_ll::dispatch"),
        "deepep_combine": count("deep_ep::internode_ll::combine"),
        "deepgemm": count("deep_gemm::"),
    }


def validate_step_structure(
    *,
    config_name: str,
    step: dict[str, Any],
    kernels: list[dict[str, Any]],
    template: list[tuple[str, str]],
) -> dict[str, int]:
    match = STEP_NAME.fullmatch(str(step.get("name", "")))
    assert match is not None
    counts = signature_counts(kernels)
    actual_collectives = [
        collective_kind(str(kernel.get("name", "")))
        for kernel in kernels
        if collective_kind(str(kernel.get("name", ""))) is not None
    ]
    incompatible = [
        {
            "index": index,
            "node": node,
            "eager_kind": eager_kind,
            "graph_kind": graph_kind,
        }
        for index, ((eager_kind, node), graph_kind) in enumerate(
            zip(template, actual_collectives)
        )
        if not (
            graph_kind == eager_kind
            or (
                "dp_" in node
                and "_gather" in node
                and {graph_kind, eager_kind}.issubset({"reduce", "gather"})
            )
        )
    ]
    if len(actual_collectives) != len(template) or incompatible:
        raise ValueError(
            f"collective sequence mismatch for {step['name']}: "
            f"expected {Counter(kind for kind, _node in template)}, "
            f"got {Counter(actual_collectives)}, incompatible={incompatible}"
        )
    if match.group(1) == "DECODE":
        required = {
            "split_pack": 36,
            "causal_conv": 36,
            "delta_rule": 36,
            "gated_norm": 36,
            "qsa_attention": 12,
        }
        mismatch = {
            key: {"expected": expected, "actual": counts[key]}
            for key, expected in required.items()
            if counts[key] != expected
        }
        if mismatch:
            raise ValueError(f"model signature mismatch for {step['name']}: {mismatch}")
    if config_name == "dp_attention_ep4_deepep_deepgemm":
        for key in ("deepep_dispatch", "deepep_combine", "deepgemm"):
            if counts[key] != 96:
                raise ValueError(
                    f"{step['name']} expected 96 {key} kernels, got {counts[key]}"
                )
    return counts


def map_step(
    *,
    kernels: list[dict[str, Any]],
    template: list[tuple[str, str]],
    rank: int,
    step_index: int,
) -> list[dict[str, Any]]:
    collective_kernels = [
        kernel
        for kernel in kernels
        if collective_kind(str(kernel.get("name", ""))) is not None
    ]
    if len(collective_kernels) != len(template):
        raise ValueError("collective count changed after structural validation")
    collective_nodes = {
        id(kernel): node
        for kernel, (_kind, node) in zip(collective_kernels, template)
    }
    mapped = []
    for kernel in kernels:
        name = str(kernel.get("name", ""))
        node = collective_nodes.get(id(kernel))
        label = "collective (eager-validated order)" if node else None
        if node is None:
            node, label = direct_kernel_mapping(name)
        mapped.append(
            {
                "rank": rank,
                "step_index": step_index,
                "kernel_name": name,
                "kernel_label": label,
                "node": node,
                "dur_us": float(kernel.get("dur", 0.0)),
            }
        )
    return mapped


def metrics_for_rank(events: list[dict[str, Any]], n_iters: int) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if event["node"]:
            grouped[event["node"]].append(event)
    metrics = {}
    for node, node_events in grouped.items():
        total_us = sum(event["dur_us"] for event in node_events)
        label_us: Counter[str] = Counter()
        label_count: Counter[str] = Counter()
        for event in node_events:
            label = event["kernel_label"] or event["kernel_name"][:120]
            label_us[label] += event["dur_us"]
            label_count[label] += 1
        metrics[node] = {
            "ms_per_iter": total_us / n_iters / 1000.0,
            "kernels": [
                {
                    "name": label,
                    "count": label_count[label],
                    "count_per_iter": round(label_count[label] / n_iters, 3),
                    "avg_us": round(duration_us / label_count[label], 3),
                    "total_us_per_iter": round(duration_us / n_iters, 3),
                    "share_in_node_pct": round(100.0 * duration_us / total_us, 2),
                }
                for label, duration_us in label_us.most_common()
            ],
        }
    return metrics


def aggregate_rank_metrics(
    rank_metrics: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    nodes = sorted({node for metrics in rank_metrics.values() for node in metrics})
    aggregated = {}
    for node in nodes:
        candidates = [
            (rank, metrics[node])
            for rank, metrics in rank_metrics.items()
            if node in metrics
        ]
        source_rank, selected = max(candidates, key=lambda item: item[1]["ms_per_iter"])
        values = [item[1]["ms_per_iter"] for item in candidates]
        aggregated[node] = {
            "ms_per_iter": round(selected["ms_per_iter"], 6),
            "aggregation": "maximum per-rank kernel residency",
            "source_rank": source_rank,
            "rank_range_ms": [round(min(values), 6), round(max(values), 6)],
            "kernels": selected["kernels"],
        }
    return aggregated


def build_profile(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = CONFIGS[args.config_name]
    if args.chunked_prefill_size % config["dp_size"] != 0:
        raise ValueError(
            "requested chunked-prefill size must divide evenly across DP ranks"
        )
    if args.admission_control == "prefill-first-until-local-target" and (
        config["dp_size"] != 4
        or args.batch_size != 256
        or not re.fullmatch(r"[0-9a-f]{64}", args.source_patch_sha256 or "")
    ):
        raise ValueError(
            "prefill-first admission is restricted to DP4 global BS256 and "
            "requires the recorded patch SHA256"
        )
    protocol = None
    if args.protocol is not None:
        protocol = json.loads(args.protocol.read_text())
        expected_protocol = {
            "mode": "cudagraph",
            "input_len": 8192,
            "output_len": 1024,
            "warmup_rounds": 3,
            "formal_rounds": 1,
        }
        if args.config_name == "tp_only":
            expected_protocol["topology"] = "tp4"
            protocol_batches = protocol.get("batch_sizes", [])
        else:
            expected_protocol.update(
                {
                    "config_name": args.config_name,
                    "dp_size": config["dp_size"],
                }
            )
            optional_protocol = {
                "max_prefill_tokens": args.max_prefill_tokens,
                "chunked_prefill_size_requested": args.chunked_prefill_size,
                "chunked_prefill_size_per_dp_rank": (
                    args.chunked_prefill_size // config["dp_size"]
                ),
                "admission_control": args.admission_control,
                "admission_local_target": (
                    args.batch_size // config["dp_size"]
                    if args.admission_control
                    == "prefill-first-until-local-target"
                    else None
                ),
                "source_patch_sha256": args.source_patch_sha256,
            }
            expected_protocol.update(
                {
                    key: value
                    for key, value in optional_protocol.items()
                    if key in protocol
                }
            )
            protocol_batches = protocol.get("global_batch_sizes", [])
        mismatch = {
            key: {"expected": expected, "actual": protocol.get(key)}
            for key, expected in expected_protocol.items()
            if protocol.get(key) != expected
        }
        if args.batch_size not in protocol_batches:
            mismatch["batch_sizes"] = {
                "expected_to_contain": args.batch_size,
                "actual": protocol_batches,
            }
        if mismatch:
            raise ValueError(f"profile protocol mismatch: {mismatch}")
    formal = load_formal_round(args.rounds, args.batch_size)
    trigger = (formal.get("profile_trigger") or {}).get("trigger") or {}
    running = trigger.get("global_running_reqs", trigger.get("num_running_reqs"))
    waiting = trigger.get("global_waiting_reqs", trigger.get("num_waiting_reqs"))
    waiting_tokens = trigger.get(
        "global_waiting_uncached_tokens", trigger.get("num_waiting_uncached_tokens")
    )
    if running != args.batch_size or waiting != 0 or waiting_tokens != 0:
        raise ValueError(f"formal capture gate is not exact global BS: {trigger}")
    summaries = formal.get("trace_step_summary") or {}
    if len(summaries) != 4:
        raise ValueError(f"formal row lacks four validated rank summaries: {summaries}")

    template = eager_collective_template(args.eager_mapping)
    traces_by_rank = {
        trace_rank(path, config["dp_size"]): path.resolve() for path in args.traces
    }
    if set(traces_by_rank) != {0, 1, 2, 3}:
        raise ValueError(f"rank trace coverage is incomplete: {traces_by_rank}")

    rank_metrics = {}
    rank_events = {}
    rank_steps_ms = {}
    rank_signature_counts = {}
    rank_step_bounds = {}
    all_mapped = []
    graph_launches = {}
    for rank, path in sorted(traces_by_rank.items()):
        events = load_trace(path)
        steps = merged_gpu_steps(events)
        selected = steps[1:]
        if len(selected) != SELECTED_STEPS:
            raise ValueError(f"rank {rank} does not have five selected steps")
        mapped = []
        counts = []
        for step_index, step in enumerate(selected, start=1):
            kernels = kernels_in_step(events, step)
            counts.append(
                validate_step_structure(
                    config_name=args.config_name,
                    step=step,
                    kernels=kernels,
                    template=template,
                )
            )
            mapped.extend(
                map_decode_step(
                    kernels=kernels,
                    config_name=args.config_name,
                    collective_template=template,
                    rank=rank,
                    step_index=step_index,
                )
            )
        rank_events[rank] = mapped
        rank_step_bounds[rank] = selected
        rank_metrics[rank] = attributed_metrics_for_rank(mapped, SELECTED_STEPS)
        rank_steps_ms[rank] = [float(step.get("dur", 0.0)) / 1000.0 for step in selected]
        rank_signature_counts[rank] = counts
        all_mapped.extend(mapped)
        graph_launches[rank] = sum(
            event.get("cat") == "cuda_runtime"
            and event.get("ph") == "X"
            and event.get("name") == "cudaGraphLaunch"
            for event in events
        )
        if graph_launches[rank] < PROFILE_STEPS:
            raise ValueError(f"rank {rank} has only {graph_launches[rank]} cudaGraphLaunch calls")

    critical_step_ms = [
        max(rank_steps_ms[rank][index] for rank in rank_steps_ms)
        for index in range(SELECTED_STEPS)
    ]
    reference_rank = select_reference_rank(rank_steps_ms, args.reference_rank)
    rank_events[reference_rank] = attach_eager_stack_evidence(
        rank_events[reference_rank], mapping_path=args.eager_mapping
    )
    node_metrics = rank_metrics[reference_rank]
    for metric in node_metrics.values():
        metric["source_rank"] = reference_rank
        metric["rank_policy"] = "one coherent critical reference rank"
        for drill_metric in metric.get("drill_metrics", {}).values():
            drill_metric["source_rank"] = reference_rank
            drill_metric["rank_policy"] = metric["rank_policy"]
    reference_step_wall_ms = rank_steps_ms[reference_rank]
    reference_active_ms = []
    reference_residency_ms = []
    for step_index in range(1, SELECTED_STEPS + 1):
        step_events = [
            event
            for event in rank_events[reference_rank]
            if int(event["step_index"]) == step_index
        ]
        reference_active_ms.append(interval_union_us(step_events) / 1000.0)
        reference_residency_ms.append(
            sum(float(event["dur_us"]) for event in step_events) / 1000.0
        )
    if any(
        active > elapsed + 1e-6
        for active, elapsed in zip(reference_active_ms, reference_step_wall_ms)
    ):
        raise ValueError("reference-rank active GPU time exceeds decode step wall")
    elapsed_ms = statistics.fmean(reference_step_wall_ms)
    active_ms = statistics.fmean(reference_active_ms)
    residency_ms = statistics.fmean(reference_residency_ms)
    device_gap_ms = max(0.0, elapsed_ms - active_ms)
    overlap_ms = max(0.0, residency_ms - active_ms)
    timing_summary = {
        "scope": "one CUDA Graph decode step on the selected reference rank",
        "elapsed_ms": round(elapsed_ms, 6),
        "active_gpu_ms": round(active_ms, 6),
        "device_gap_ms": round(device_gap_ms, 6),
        "gpu_busy_pct": round(100.0 * active_ms / elapsed_ms, 2)
        if elapsed_ms
        else 0.0,
        "gpu_residency_ms": round(residency_ms, 6),
        "gpu_overlap_ms": round(overlap_ms, 6),
        "elapsed_source": "ProfilerStep GPU annotation on the same reference rank",
        "device_gap_reason": "unclassified",
        "reference_rank": reference_rank,
        "sample_count": SELECTED_STEPS,
        "elapsed_samples_ms": [round(value, 6) for value in reference_step_wall_ms],
        "active_samples_ms": [round(value, 6) for value in reference_active_ms],
    }
    total_us = sum(event["dur_us"] for event in all_mapped)
    mapped_us = sum(event["dur_us"] for event in all_mapped if event["node"])
    if args.config_name == "tp_only":
        profile_id = f"qwen38_flash_next_tp4_cg_decode_bs{args.batch_size}_8k1k"
        variant_id = f"tp4_cg_decode_bs{args.batch_size}_8k1k"
    else:
        profile_id = f"qwen38_flash_next_{args.config_name}_cg_decode_gbs{args.batch_size}_8k1k"
        variant_id = f"{args.config_name}_cg_decode_gbs{args.batch_size}_8k1k"
    if args.profile_tag:
        safe_tag = re.sub(r"[^a-z0-9_]+", "_", args.profile_tag.lower()).strip("_")
        if not safe_tag:
            raise ValueError("profile tag must contain at least one letter or digit")
        profile_id = f"{profile_id}_{safe_tag}"
        variant_id = f"{variant_id}_{safe_tag}"
    implementation_id = args.implementation_id or config["implementation_id"]
    implementation_label = args.implementation_label or config["label"]
    profile = {
        "schema_version": "profile.v2",
        "profile_id": profile_id,
        "label": f"GB300 · {implementation_label} · CUDA Graph decode · global BS{args.batch_size} · 8k→1k",
        "model_id": "qwen38_flash_next",
        "execution_path_id": config["execution_path_id"],
        "implementation_id": implementation_id,
        "variant_id": variant_id,
        "phase": "decode",
        "execution_parameters": {
            "tp_size": config["tp_size"],
            "dp_size": config["dp_size"],
            "cp_size": 1,
            "ep_size": config["ep_size"],
        },
        "hardware": {
            "gpu": "GB300",
            "gpus_per_node": 4,
            "nodes": 1,
            "cluster": "CMH",
        },
        "workload": {
            "isl": 8192,
            "osl": 1024,
            "batch_size": args.batch_size,
            "batch_size_scope": "global_request_count",
            "concurrency": args.batch_size,
            "warmup_rounds": 3,
            "formal_rounds": 1,
            "prompt_source": "deterministic-random-ids",
            "prompt_seed": 20260819,
            "cache_policy": "radix-disabled; flush-before-each-case",
            "requested_chunked_prefill_size": args.chunked_prefill_size,
            "resolved_chunked_prefill_size": (
                args.chunked_prefill_size // config["dp_size"]
            ),
            "max_prefill_tokens": args.max_prefill_tokens,
            "admission_control": args.admission_control,
        },
        "profiler": {
            "type": "torch",
            "rank": "all",
            "activities": ["CPU", "GPU"],
            "cuda_graph_enabled": True,
            "captured_steps_per_rank": PROFILE_STEPS,
            "selected_iterations": SELECTED_STEPS,
            "skipped_first_profile_step": 1,
            "with_stack": False,
            "record_shapes": False,
            "capture_gate": {
                "global_running_reqs": args.batch_size,
                "global_waiting_reqs": 0,
                "global_waiting_uncached_tokens": 0,
            },
            "admission_instrumentation": {
                "mode": args.admission_control,
                "local_target": (
                    args.batch_size // config["dp_size"]
                    if args.admission_control
                    == "prefill-first-until-local-target"
                    else None
                ),
                "active_during_profile": False,
            },
            "gpu_metric_semantics": (
                "overlap-aware active GPU interval union on one coherent critical "
                "reference rank; residency is retained separately"
            ),
        },
        "evidence": {
            "job_id": int(args.job_id) if args.job_id.isdigit() else args.job_id,
            "source_commit": getattr(args, "source_commit", SOURCE_COMMIT),
            "source_patch_sha256": args.source_patch_sha256,
            "protocol_file": args.protocol.name if args.protocol is not None else None,
            "protocol_sha256": (
                sha256_file(args.protocol) if args.protocol is not None else None
            ),
            "model_revision": MODEL_REVISION,
            "gdn_backend": config["gdn_backend"],
            "trace_files": [
                {
                    "rank": rank,
                    "file": path.name,
                    "sha256": sha256_file(path),
                }
                for rank, path in sorted(traces_by_rank.items())
            ],
            "eager_mapping_file": args.eager_mapping.name,
            "eager_mapping_sha256": sha256_file(args.eager_mapping),
            "mapping_policy": (
                "eager Python-stack/source binding transferred by direct signatures, "
                "eager-validated collective order, and validated 48-layer execution sequence"
            ),
            "mapped_kernel_duration_ratio": round(mapped_us / total_us, 6),
            "reference_rank": reference_rank,
            "accounting": {
                "kernel_count": len(rank_events[reference_rank]),
                "attributed_kernel_count": sum(
                    bool(event.get("node")) for event in rank_events[reference_rank]
                ),
                "unattributed_kernel_count": sum(
                    not event.get("node") for event in rank_events[reference_rank]
                ),
                "attributed_gpu_residency_pct": round(
                    100.0
                    * sum(
                        event["dur_us"]
                        for event in rank_events[reference_rank]
                        if event.get("node")
                    )
                    / sum(event["dur_us"] for event in rank_events[reference_rank]),
                    4,
                ),
                "attribution_methods": dict(
                    Counter(
                        event["attribution_method"]
                        for event in rank_events[reference_rank]
                    )
                ),
                "active_gpu_ms": timing_summary["active_gpu_ms"],
                "gpu_residency_ms": timing_summary["gpu_residency_ms"],
                "gpu_elapsed_ms": timing_summary["elapsed_ms"],
                "device_gap_ms": timing_summary["device_gap_ms"],
                "gpu_busy_pct": timing_summary["gpu_busy_pct"],
                "gpu_overlap_ms": timing_summary["gpu_overlap_ms"],
            },
            "critical_decode_step_ms": {
                "samples": [round(value, 6) for value in critical_step_ms],
                "mean": round(statistics.fmean(critical_step_ms), 6),
                "median": round(statistics.median(critical_step_ms), 6),
                "min": round(min(critical_step_ms), 6),
                "max": round(max(critical_step_ms), 6),
            },
            "formal_request": {
                key: formal.get(key)
                for key in (
                    "latency",
                    "last_ttft",
                    "input_throughput",
                    "output_throughput",
                    "overall_throughput",
                    "started_at_unix",
                    "finished_at_unix",
                )
            },
        },
        "profile_summary": {
            "timing_phase": "decode",
            "timing_coverage": "100% of captured CUDA kernels",
            "reference_rank": reference_rank,
            "node_time": "active GPU time; overlap counted once",
            "kernel_detail": "GPU residency; concurrent streams may sum above node active time",
            "provenance": "direct / eager-stack collective / validated sequence shown per node",
            "timing": timing_summary,
            "gap_note": (
                "device gap is same-rank decode-step elapsed minus the union of all "
                "GPU kernel intervals; its CPU/synchronization cause is not inferred "
                "without direct evidence"
            ),
        },
        "node_states": default_node_states(phase="decode"),
        "node_metrics": node_metrics,
    }
    if args.node:
        profile["hardware"]["slurm_node"] = args.node

    timeline_path = args.output_profile.with_suffix(".timeline.json.gz")
    reference_trace = traces_by_rank[reference_rank]
    timeline_steps = []
    for step_index, step in enumerate(rank_step_bounds[reference_rank], start=1):
        timeline_steps.append(
            {
                "step_index": step_index,
                "label": f"formal decode step {step_index}",
                "trace_start_us": float(step["ts"]),
                "duration_us": float(step["dur"]),
                "events": [
                    event
                    for event in rank_events[reference_rank]
                    if int(event["step_index"]) == step_index
                ],
            }
        )
    timeline = build_timeline_artifact(
        profile_id=profile_id,
        phase="decode",
        reference_rank=reference_rank,
        steps=timeline_steps,
        timing_summary=timing_summary,
        raw_trace={
            "file": reference_trace.name,
            "sha256": sha256_file(reference_trace),
            "format": "pytorch_trace_json_gzip",
            "rank": reference_rank,
        },
        stack_source={
            "source": "eager_trace",
            "mapping_file": args.eager_mapping.name,
            "mapping_sha256": sha256_file(args.eager_mapping),
            "policy": (
                "formal CUDA Graph timing plus eager Python-stack evidence; "
                "each event records exact-kernel or representative-IR match provenance"
            ),
        },
    )
    profile["timeline"] = {
        "schema_version": timeline["schema_version"],
        "artifact": timeline_path.name,
        "reference_rank": reference_rank,
        "step_count": len(timeline["steps"]),
        "event_count": sum(len(step["events"]) for step in timeline["steps"]),
        "raw_trace_file": reference_trace.name,
    }

    unmapped: Counter[str] = Counter()
    for event in all_mapped:
        if event["node"] is None:
            unmapped[event["kernel_name"]] += event["dur_us"]
    analysis = {
        "profile_id": profile_id,
        "config_name": args.config_name,
        "global_batch_size": args.batch_size,
        "collective_template": template,
        "rank_step_ms": rank_steps_ms,
        "critical_step_ms": profile["evidence"]["critical_decode_step_ms"],
        "rank_signature_counts": rank_signature_counts,
        "cuda_graph_launches": graph_launches,
        "mapped_kernel_duration_ratio": profile["evidence"]["mapped_kernel_duration_ratio"],
        "reference_rank": reference_rank,
        "accounting": profile["evidence"]["accounting"],
        "node_metrics": node_metrics,
        "top_unmapped_kernels": [
            {"name": name, "total_us": round(duration_us, 6)}
            for name, duration_us in unmapped.most_common(30)
        ],
        "formal_capture_trigger": formal.get("profile_trigger"),
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
    print(f"wrote {args.output_profile.resolve()}")
    print(f"wrote {timeline_path.resolve()}")
    print(f"wrote {args.output_analysis.resolve()}")
    print(
        f"critical decode mean={profile['evidence']['critical_decode_step_ms']['mean']:.3f} ms, "
        f"mapped={profile['evidence']['mapped_kernel_duration_ratio']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
