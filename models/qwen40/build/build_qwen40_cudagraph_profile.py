#!/usr/bin/env python3
"""Build one Qwen 4.0 TP4 CUDA-Graph decode profile overlay.

The eager trace is the source of call-stack/code binding evidence.  CUDA Graph
traces intentionally omit stacks, so this builder maps only model-specific
kernel signatures and a TP-collective sequence that was first proven against
the eager trace.  Any structural mismatch is fatal rather than guessed.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any

import yaml


SOURCE_COMMIT = "f90a941aa6ff71ac3bd7d40b8daccdf5bd914af0"
MODEL_REVISION = "b151fd157ff99b63198ab8558432f0bf43e14d58"
EXPECTED_PROFILE_STEPS = 6
EXPECTED_SELECTED_STEPS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True, help="TP-rank trace")
    parser.add_argument("--rounds", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, choices=(1, 16, 64, 256), required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output-profile", type=Path, required=True)
    parser.add_argument("--output-analysis", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--node", default="")
    return parser.parse_args()


def load_trace(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as trace_file:
        return json.load(trace_file)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_formal_round(path: Path, batch_size: int) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    matches = [
        row
        for row in rows
        if row.get("round") == "formal-1" and row.get("batch_size") == batch_size
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one formal-1 row for bs={batch_size}, got {len(matches)}"
        )
    return matches[0]


def decode_steps(
    trace_events: list[dict[str, Any]], batch_size: int
) -> list[dict[str, Any]]:
    expected_name = f"step[DECODE bs={batch_size}]"
    cpu_steps = sorted(
        (
            event
            for event in trace_events
            if event.get("cat") == "user_annotation"
            and event.get("ph") == "X"
            and str(event.get("name", "")).startswith("step[")
        ),
        key=lambda event: float(event.get("ts", 0.0)),
    )
    cpu_names = [str(event.get("name")) for event in cpu_steps]
    if cpu_names != [expected_name] * EXPECTED_PROFILE_STEPS:
        raise ValueError(
            f"expected {EXPECTED_PROFILE_STEPS} exact CPU {expected_name!r} steps, "
            f"got {cpu_names}"
        )

    gpu_tracks: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for event in trace_events:
        if (
            event.get("cat") != "gpu_user_annotation"
            or event.get("ph") != "X"
            or not str(event.get("name", "")).startswith("step[")
        ):
            continue
        if str(event.get("name")) != expected_name:
            raise ValueError(
                f"GPU graph annotation mixed into bs={batch_size}: {event.get('name')!r}"
            )
        gpu_tracks[(event.get("pid"), event.get("tid"))].append(event)
    if not gpu_tracks:
        raise ValueError("trace has no GPU step annotations")
    invalid_tracks = {
        str(track_id): len(events)
        for track_id, events in gpu_tracks.items()
        if len(events) != EXPECTED_PROFILE_STEPS
    }
    if invalid_tracks:
        raise ValueError(
            f"GPU annotation tracks do not match profiler steps: {invalid_tracks}"
        )

    primary_track = max(
        gpu_tracks.values(),
        key=lambda track: statistics.fmean(float(event.get("dur", 0.0)) for event in track),
    )
    primary_track = sorted(
        primary_track, key=lambda event: float(event.get("ts", 0.0))
    )
    graph_launches = [
        event
        for event in trace_events
        if event.get("cat") == "cuda_runtime"
        and event.get("ph") == "X"
        and event.get("name") == "cudaGraphLaunch"
    ]
    if len(graph_launches) != EXPECTED_PROFILE_STEPS:
        raise ValueError(
            f"expected {EXPECTED_PROFILE_STEPS} cudaGraphLaunch calls, "
            f"got {len(graph_launches)}"
        )
    return primary_track


def kernels_in_step(
    trace_events: list[dict[str, Any]], step: dict[str, Any]
) -> list[dict[str, Any]]:
    start = float(step.get("ts", 0.0))
    end = start + float(step.get("dur", 0.0))
    return sorted(
        (
            event
            for event in trace_events
            if event.get("cat") == "kernel"
            and event.get("ph") == "X"
            and start <= float(event.get("ts", 0.0)) <= end
        ),
        key=lambda event: float(event.get("ts", 0.0)),
    )


def is_all_reduce(name: str) -> bool:
    lowered = name.lower()
    return "allreduce" in lowered or "all_reduce" in lowered


def is_all_gather(name: str) -> bool:
    lowered = name.lower()
    return "allgather" in lowered or "all_gather" in lowered


def expected_all_reduce_roles() -> list[str]:
    roles = ["top.tp_embedding_collective"]
    for layer_id in range(48):
        # PLE layer_ids=[2] is one-based in the checkpoint/source contract.
        if layer_id == 1:
            roles.append("ple.tp_embedding_collective")
        layer_view = "full_layer" if layer_id % 4 == 3 else "linear_layer"
        roles.append(f"{layer_view}.tp_attention_collective")
        roles.append("moe.tp_output_collective")
    if len(roles) != 98:
        raise AssertionError(f"invalid TP all-reduce role template: {len(roles)}")
    return roles


def direct_kernel_mapping(name: str) -> tuple[str | None, str | None]:
    """Return only mappings whose kernel signature is semantically unique."""

    lowered = name.lower()
    rules = (
        ("fused_qkvzba_split", "linear_attention.split_pack", "GDN split+pack"),
        ("causal_conv1d_update", "linear_attention.causal_conv", "GDN causal-conv update"),
        (
            "fused_recurrent_gated_delta_rule_packed_decode",
            "linear_attention.delta_rule",
            "GDN recurrent delta rule",
        ),
        ("_layer_norm_fwd_1pass_kernel", "linear_attention.gated_norm", "GDN gated RMSNorm"),
        ("_hc_mix_persistent_kernel", "hyperconnection.mix", "hyper-connection mix"),
        ("hc_combine_kernel", "hyperconnection.combine", "hyper-connection combine"),
        ("_fused_qk_rmsnorm_rope_gate_kernel", "qsa_attention.qk_norm_rope", "QSA Q/K norm + RoPE"),
        ("qsa_index_q_prep_kernel", "qsa_attention.indexer", "QSA index query prep"),
        ("qsa_index_k_compress_kernel", "qsa_attention.indexer", "QSA index key compression"),
        ("fast_topk_detail::fast_topk_kernel", "qsa_attention.indexer", "QSA index top-k"),
        ("_expand_qsa_block_indices_kernel", "qsa_attention.indexer", "QSA block-index expansion"),
        ("store_kvcache", "qsa_attention.kv_cache", "QSA KV-cache store"),
        ("_qsa_graph_layout_alloc_kernel", "qsa_attention.attention_core", "QSA graph layout"),
        ("_fa2_valid_counts", "qsa_attention.attention_core", "QSA valid-count preparation"),
        ("_compact_kv", "qsa_attention.attention_core", "QSA compact KV"),
        ("fmhasm100fkernel_qkv", "qsa_attention.attention_core", "QSA sparse attention"),
        ("moe::dev::routing", "moe.topk", "MoE top-k routing"),
        ("bmm_bfloat16", "moe.routed_experts", "MoE routed-expert GEMM"),
        ("act_and_mul_kernel", "moe.shared_expert", "MoE shared-expert activation"),
        ("moe::dev::finalize", "moe.combine", "MoE routed-expert finalize"),
        ("_fused_gate_sigmoid_mul_add_kernel", "moe.combine", "MoE shared/routed combine"),
    )
    for signature, node, label in rules:
        if signature in lowered:
            return node, label
    return None, None


def expected_signature_counts(kernels: list[dict[str, Any]]) -> dict[str, int]:
    names = [str(kernel.get("name", "")) for kernel in kernels]

    def count(signature: str) -> int:
        return sum(signature in name.lower() for name in names)

    return {
        "tp_all_reduce": sum(is_all_reduce(name) for name in names),
        "tp_logits_all_gather": sum(is_all_gather(name) for name in names),
        "gdn_split_pack": count("fused_qkvzba_split"),
        "gdn_causal_conv": count("causal_conv1d_update"),
        "gdn_recurrent_delta": count("fused_recurrent_gated_delta_rule_packed_decode"),
        "gdn_gated_norm": count("_layer_norm_fwd_1pass_kernel"),
        "qsa_attention": count("fmhasm100fkernel_qkv"),
        "moe_routing": count("moe::dev::routing"),
        "moe_routed_bmm": count("bmm_bfloat16"),
        "moe_finalize": count("moe::dev::finalize"),
    }


def validate_signature_counts(
    counts: dict[str, int], step_index: int, batch_size: int
) -> None:
    expected = {
        "tp_all_reduce": 98,
        "tp_logits_all_gather": 1,
        "gdn_split_pack": 36,
        "gdn_causal_conv": 36,
        "gdn_recurrent_delta": 36,
        "gdn_gated_norm": 36,
        "qsa_attention": 12,
        # FlashInfer uses one routing kernel per MoE call at BS1/16 and a
        # BlockScores + Cluster pair per call at BS64/256.
        "moe_routing": 48 if batch_size <= 16 else 96,
        "moe_routed_bmm": 96,
        "moe_finalize": 48,
    }
    if counts != expected:
        mismatch = {
            key: {"expected": expected[key], "actual": counts.get(key)}
            for key in expected
            if counts.get(key) != expected[key]
        }
        raise ValueError(f"decode step {step_index} signature mismatch: {mismatch}")


def map_step_kernels(
    kernels: list[dict[str, Any]], step_index: int, batch_size: int
) -> list[dict[str, Any]]:
    counts = expected_signature_counts(kernels)
    validate_signature_counts(counts, step_index, batch_size)

    all_reduces = [
        kernel for kernel in kernels if is_all_reduce(str(kernel.get("name", "")))
    ]
    reduce_node_by_identity = {
        id(kernel): role
        for kernel, role in zip(all_reduces, expected_all_reduce_roles())
    }
    mapped: list[dict[str, Any]] = []
    for kernel in kernels:
        name = str(kernel.get("name", ""))
        node = reduce_node_by_identity.get(id(kernel))
        label = "TP all-reduce" if node else None
        if node is None and is_all_gather(name):
            node, label = "top.tp_logits_collective", "TP logits all-gather"
        if node is None:
            node, label = direct_kernel_mapping(name)
        mapped.append(
            {
                "step_index": step_index,
                "kernel_name": name,
                "kernel_label": label,
                "node": node,
                "ts_us": float(kernel.get("ts", 0.0)),
                "dur_us": float(kernel.get("dur", 0.0)),
            }
        )
    return mapped


def build_node_metrics(
    mapped_events: list[dict[str, Any]], n_iters: int
) -> dict[str, Any]:
    node_events: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in mapped_events:
        if event["node"]:
            node_events[event["node"]].append(event)

    metrics: dict[str, Any] = {}
    for node, events in sorted(node_events.items()):
        total_us = sum(event["dur_us"] for event in events)
        label_us: Counter[str] = Counter()
        label_count: Counter[str] = Counter()
        for event in events:
            label = event["kernel_label"] or event["kernel_name"][:120]
            label_us[label] += event["dur_us"]
            label_count[label] += 1
        kernels = []
        for label, duration_us in label_us.most_common():
            count = label_count[label]
            kernels.append(
                {
                    "name": label,
                    "count": count,
                    "count_per_iter": round(count / n_iters, 3),
                    "avg_us": round(duration_us / count, 3),
                    "total_us_per_iter": round(duration_us / n_iters, 3),
                    "share_in_node_pct": round(100.0 * duration_us / total_us, 2),
                }
            )
        metrics[node] = {
            "ms_per_iter": round(total_us / n_iters / 1000.0, 6),
            "kernels": kernels,
        }
    return metrics


def build_profile(
    *,
    trace_path: Path,
    rounds_path: Path,
    batch_size: int,
    job_id: str,
    rank: int,
    node_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    trace = load_trace(trace_path)
    trace_events = trace.get("traceEvents") or []
    steps = decode_steps(trace_events, batch_size)
    selected_steps = steps[1:]
    if len(selected_steps) != EXPECTED_SELECTED_STEPS:
        raise AssertionError(f"invalid selected-step count: {len(selected_steps)}")

    mapped_events: list[dict[str, Any]] = []
    per_step_counts: list[dict[str, Any]] = []
    per_step_kernel_counts: list[int] = []
    for selected_index, step in enumerate(selected_steps, start=1):
        kernels = kernels_in_step(trace_events, step)
        counts = expected_signature_counts(kernels)
        validate_signature_counts(counts, selected_index, batch_size)
        per_step_counts.append(counts)
        per_step_kernel_counts.append(len(kernels))
        mapped_events.extend(map_step_kernels(kernels, selected_index, batch_size))

    node_metrics = build_node_metrics(mapped_events, len(selected_steps))
    total_kernel_us = sum(event["dur_us"] for event in mapped_events)
    mapped_kernel_us = sum(
        event["dur_us"] for event in mapped_events if event["node"] is not None
    )
    step_ms = [float(step.get("dur", 0.0)) / 1000.0 for step in selected_steps]
    formal = load_formal_round(rounds_path, batch_size)
    trigger = formal.get("profile_trigger") or {}
    trigger_sample = trigger.get("trigger") or {}
    expected_trigger = {
        "num_running_reqs": batch_size,
        "num_waiting_reqs": 0,
        "num_waiting_uncached_tokens": 0,
    }
    actual_trigger = {key: trigger_sample.get(key) for key in expected_trigger}
    if actual_trigger != expected_trigger:
        raise ValueError(
            f"formal profile was not armed at exact decode bs={batch_size}: "
            f"expected {expected_trigger}, got {actual_trigger}"
        )
    expected_step_name = f"step[DECODE bs={batch_size}]"
    rank_summaries = formal.get("trace_step_summary") or {}
    if len(rank_summaries) != 4:
        raise ValueError(
            f"expected validated trace summaries for four TP ranks, got "
            f"{sorted(rank_summaries)}"
        )
    invalid_rank_summaries = {
        rank_name: summary
        for rank_name, summary in rank_summaries.items()
        if summary.get("cpu_step_names")
        != [expected_step_name] * EXPECTED_PROFILE_STEPS
        or summary.get("primary_gpu_step_names")
        != [expected_step_name] * EXPECTED_PROFILE_STEPS
        or set(summary.get("gpu_step_name_counts") or {}) != {expected_step_name}
    }
    if invalid_rank_summaries:
        raise ValueError(
            f"one or more TP-rank trace summaries are invalid: {invalid_rank_summaries}"
        )

    profile_id = f"qwen40_tp4_cg_decode_bs{batch_size}_8k1k"
    variant_id = f"cg_decode_bs{batch_size}_8k1k"
    profile = {
        "schema_version": "profile.v2",
        "profile_id": profile_id,
        "label": f"GB300 · pure TP4 · CUDA Graph decode · BS{batch_size} · 8k→1k",
        "model_id": "qwen40",
        "execution_path_id": "tp_only",
        "implementation_id": "sglang_f90a941aa",
        "variant_id": variant_id,
        "phase": "decode",
        "execution_parameters": {
            "tp_size": 4,
            "dp_size": 1,
            "cp_size": 1,
            "ep_size": 1,
        },
        "hardware": {
            "gpu": "GB300",
            "gpus_per_node": 4,
            "nodes": 1,
            "cluster": "CMH",
            "slurm_node": node_name or None,
        },
        "workload": {
            "isl": 8192,
            "osl": 1024,
            "batch_size": batch_size,
            "concurrency": batch_size,
            "warmup_rounds": 3,
            "formal_rounds": 1,
            "prompt_source": "deterministic-random-ids",
            "prompt_seed": 20260819,
            "cache_policy": "flush-before-each-case",
            "chunked_prefill_size": 8192,
            "max_prefill_tokens": 8192,
        },
        "profiler": {
            "type": "torch",
            "rank": rank,
            "activities": ["CPU", "GPU"],
            "cuda_graph_enabled": True,
            "captured_steps": EXPECTED_PROFILE_STEPS,
            "selected_iterations": len(selected_steps),
            "skipped_first_profile_step": 1,
            "with_stack": False,
            "record_shapes": False,
            "capture_gate": expected_trigger,
            "gpu_metric_semantics": "aggregate kernel residency per decode iteration",
        },
        "evidence": {
            "job_id": int(job_id) if job_id.isdigit() else job_id,
            "source_commit": SOURCE_COMMIT,
            "model_revision": MODEL_REVISION,
            "trace_file": trace_path.name,
            "trace_sha256": sha256_file(trace_path),
            "trace_step_name": expected_step_name,
            "validated_tp_rank_count": len(rank_summaries),
            "mapping_policy": "high-confidence signatures plus eager-validated TP collective order",
            "mapped_kernel_duration_ratio": round(
                mapped_kernel_us / total_kernel_us if total_kernel_us else 0.0, 6
            ),
            "decode_step_ms": {
                "samples": [round(value, 6) for value in step_ms],
                "mean": round(statistics.fmean(step_ms), 6),
                "median": round(statistics.median(step_ms), 6),
                "min": round(min(step_ms), 6),
                "max": round(max(step_ms), 6),
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
        "node_metrics": node_metrics,
    }
    # Avoid serializing a null hostname when the caller did not provide one.
    if not node_name:
        del profile["hardware"]["slurm_node"]

    unmapped: Counter[str] = Counter()
    for event in mapped_events:
        if event["node"] is None:
            unmapped[event["kernel_name"]] += event["dur_us"]
    report = {
        "profile_id": profile_id,
        "trace": str(trace_path),
        "trace_sha256": profile["evidence"]["trace_sha256"],
        "rank": rank,
        "captured_step_ms": [
            round(float(step.get("dur", 0.0)) / 1000.0, 6) for step in steps
        ],
        "selected_step_ms": profile["evidence"]["decode_step_ms"],
        "per_step_kernel_counts": per_step_kernel_counts,
        "per_step_signature_counts": per_step_counts,
        "total_kernel_us": round(total_kernel_us, 6),
        "mapped_kernel_us": round(mapped_kernel_us, 6),
        "mapped_kernel_duration_ratio": profile["evidence"][
            "mapped_kernel_duration_ratio"
        ],
        "node_metrics": node_metrics,
        "top_unmapped_kernels": [
            {"name": name, "total_us": round(duration_us, 6)}
            for name, duration_us in unmapped.most_common(30)
        ],
        "formal_capture_trigger": trigger,
    }
    return profile, report


def main() -> int:
    args = parse_args()
    profile, report = build_profile(
        trace_path=args.trace.resolve(),
        rounds_path=args.rounds.resolve(),
        batch_size=args.batch_size,
        job_id=args.job_id,
        rank=args.rank,
        node_name=args.node,
    )
    args.output_profile.parent.mkdir(parents=True, exist_ok=True)
    args.output_profile.write_text(
        yaml.safe_dump(profile, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    args.output_analysis.parent.mkdir(parents=True, exist_ok=True)
    args.output_analysis.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output_profile.resolve()}")
    print(f"wrote {args.output_analysis.resolve()}")
    print(
        f"decode mean={profile['evidence']['decode_step_ms']['mean']:.3f} ms, "
        f"mapped={profile['evidence']['mapped_kernel_duration_ratio']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
