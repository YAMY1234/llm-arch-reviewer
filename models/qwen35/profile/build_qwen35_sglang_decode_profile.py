#!/usr/bin/env python3
"""Build the four-rank SGLang AgentX CUDA-Graph decode profile and timeline."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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

from models.common.timeline_artifact import build_timeline_artifact, write_timeline_artifact
from models.common.trace_mapping import find_eagle_mtp_decode_windows, load_trace
from models.qwen35.profile.qwen35_graph_mapping import (
    attach_graph_stack_evidence,
    map_graph_window,
)


PROFILE_ID = "qwen35_sglang_attention_dp4_moe_ep4_mtp6_cg_decode_gbs32"
SOURCE_COMMIT = "85c23c62fdc58a5a0c3b7c6d61a7bba720a6cbbf"
RUNTIME_SOURCE_COMMIT = "a31c1e52e947bcbdd0d551c5e2323e96a9bf303b"
MODEL_REVISION = "8f590eae8f10bf55d9a46f79ea0280bde435c9f8"
TRACE_RANK = re.compile(r"-TP-(\d+)-DP-(\d+)-EP-(\d+)\.trace\.json\.gz$")
SELECTED_WINDOW_INDICES = (2, 3, 4, 5)
SGLANG_NODE_STATES = {
    "gdn_moe_block.ba_projection": {
        "status": "fused",
        "included_in": "gdn_moe_block.qkvz_projection",
    },
    "gdn_moe_block.state_write": {
        "status": "fused",
        "included_in": "gdn_moe_block.gated_delta_recurrence",
    },
    "full_attention_moe_block.partial_rope": {
        "status": "fused",
        "included_in": "full_attention_moe_block.qk_norm",
    },
    "full_attention_moe_block.attention_output_gate": {
        "status": "fused",
        "included_in": "full_attention_moe_block.qk_norm",
    },
    "full_attention_moe_block.kv_state_write": {
        "status": "fused",
        "included_in": "full_attention_moe_block.qkv_projection",
    },
    "mtp_full_attention_moe_block.partial_rope": {
        "status": "fused",
        "included_in": "mtp_full_attention_moe_block.qk_norm",
    },
    "mtp_full_attention_moe_block.attention_output_gate": {
        "status": "fused",
        "included_in": "mtp_full_attention_moe_block.qk_norm",
    },
    "mtp_full_attention_moe_block.kv_state_write": {
        "status": "fused",
        "included_in": "mtp_full_attention_moe_block.qkv_projection",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, nargs=4, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--eager-mapping", type=Path, required=True)
    parser.add_argument("--output-profile", type=Path, required=True)
    parser.add_argument("--output-timeline", type=Path, required=True)
    parser.add_argument("--output-analysis", type=Path, required=True)
    parser.add_argument("--output-mapping", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def trace_rank(path: Path) -> int:
    match = TRACE_RANK.search(path.name)
    if match is None:
        raise ValueError(f"cannot parse TP/DP/EP rank from {path.name}")
    ranks = tuple(int(match.group(index)) for index in (1, 2, 3))
    if len(set(ranks)) != 1:
        raise ValueError(f"trace does not use aligned Attention-DP4/MoE-EP4 ranks: {ranks}")
    return ranks[0]


def validate_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text())
    formal = protocol.get("formal") or {}
    expected = {
        "kind": "decode-bs32",
        "expected_ranks": 4,
        "model_revision": MODEL_REVISION,
    }
    mismatch = {
        key: {"expected": value, "actual": protocol.get(key)}
        for key, value in expected.items()
        if protocol.get(key) != value
    }
    formal_expected = {
        "batch": 32,
        "isl": 128,
        "osl": 64,
        "cuda_graph": True,
        "per_rank_running": [8, 8, 8, 8],
        "profile_steps": 8,
    }
    mismatch.update(
        {
            f"formal.{key}": {"expected": value, "actual": formal.get(key)}
            for key, value in formal_expected.items()
            if formal.get(key) != value
        }
    )
    trigger = (formal.get("trigger") or {}).get("trigger") or {}
    for key, expected_value in {
        "global_running_reqs": 32,
        "global_waiting_reqs": 0,
        "global_waiting_uncached_tokens": 0,
    }.items():
        if trigger.get(key) != expected_value:
            mismatch[f"formal.trigger.selected.{key}"] = {
                "expected": expected_value,
                "actual": trigger.get(key),
            }
    server = protocol.get("server_info") or {}
    for key, expected_value in {
        "enable_dp_attention": True,
        "dp_size": 4,
        "tp_size": 4,
        "ep_size": 4,
        "disable_cuda_graph": False,
        "speculative_num_steps": 5,
        "speculative_num_draft_tokens": 6,
    }.items():
        if server.get(key) != expected_value:
            mismatch[f"server_info.{key}"] = {
                "expected": expected_value,
                "actual": server.get(key),
            }
    if mismatch:
        raise ValueError(f"decode protocol mismatch: {mismatch}")
    return protocol


def _metrics_for_rank(events: list[dict[str, Any]], n_iters: int) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if event.get("node"):
            grouped[str(event["node"])].append(event)
    metrics: dict[str, Any] = {}
    for node, node_events in sorted(grouped.items()):
        total_us = sum(float(event["dur_us"]) for event in node_events)
        label_us: Counter[str] = Counter()
        label_count: Counter[str] = Counter()
        status_us: Counter[str] = Counter()
        for event in node_events:
            label = str(event.get("kernel_label") or event["kernel_name"][:120])
            label_us[label] += float(event["dur_us"])
            label_count[label] += 1
            status_us[str(event["mapping_status"])] += float(event["dur_us"])
        metrics[node] = {
            "ms_per_iter": total_us / n_iters / 1000.0,
            "mapping_status_duration_pct": {
                status: round(100.0 * duration / total_us, 4)
                for status, duration in sorted(status_us.items())
            },
            "kernels": [
                {
                    "name": label,
                    "count": int(label_count[label]),
                    "count_per_iter": round(label_count[label] / n_iters, 3),
                    "avg_us": round(duration_us / label_count[label], 3),
                    "total_us_per_iter": round(duration_us / n_iters, 3),
                    "share_in_node_pct": round(100.0 * duration_us / total_us, 3),
                }
                for label, duration_us in label_us.most_common()
            ],
        }
    return metrics


def _aggregate_rank_metrics(rank_metrics: dict[int, dict[str, Any]]) -> dict[str, Any]:
    nodes = sorted({node for metrics in rank_metrics.values() for node in metrics})
    aggregated = {}
    for node in nodes:
        candidates = [
            (rank, metrics[node])
            for rank, metrics in sorted(rank_metrics.items())
            if node in metrics
        ]
        source_rank, selected = max(candidates, key=lambda item: item[1]["ms_per_iter"])
        values = [cell["ms_per_iter"] for _rank, cell in candidates]
        aggregated[node] = {
            **selected,
            "ms_per_iter": round(selected["ms_per_iter"], 6),
            "aggregation": "maximum per-rank kernel residency",
            "source_rank": source_rank,
            "rank_range_ms": [round(min(values), 6), round(max(values), 6)],
        }
    return aggregated


def _validate_step_signatures(validation: dict[str, Any], *, rank: int, step: int) -> None:
    expected = {
        "target_gdn_layers": 45,
        "target_attention_layers": 15,
        "target_ep4_dispatch": 60,
        "target_ep4_combine": 60,
        "draft_deepep_dispatch": 10,
        "draft_deepep_combine": 10,
        "gdn_replay": 1,
        "mtp_draft_rounds": 5,
    }
    actual = validation["signature_counts"]
    mismatch = {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatch:
        raise ValueError(f"rank {rank} step {step} signature mismatch: {mismatch}")
    if validation["attributed_duration_ratio"] < 0.95:
        raise ValueError(
            f"rank {rank} step {step} attributed duration ratio "
            f"{validation['attributed_duration_ratio']:.4f} < 0.95"
        )


def build(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    protocol = validate_protocol(args.protocol)
    paths_by_rank = {trace_rank(path): path.resolve() for path in args.traces}
    if set(paths_by_rank) != {0, 1, 2, 3}:
        raise ValueError(f"incomplete four-rank trace coverage: {paths_by_rank}")

    rank_events: dict[int, list[dict[str, Any]]] = {}
    rank_metrics: dict[int, dict[str, Any]] = {}
    rank_wall_ms: dict[int, list[float]] = {}
    rank_validation: dict[int, list[dict[str, Any]]] = {}
    reference_steps: list[dict[str, Any]] = []
    all_mapping_rows: list[dict[str, Any]] = []

    for rank, path in sorted(paths_by_rank.items()):
        trace_events = load_trace(path).get("traceEvents") or []
        windows = find_eagle_mtp_decode_windows(
            trace_events, signature="fused_qkvzba_split"
        )
        if len(windows) < max(SELECTED_WINDOW_INDICES) + 1:
            raise ValueError(f"rank {rank} has only {len(windows)} complete AgentX windows")
        selected = [windows[index] for index in SELECTED_WINDOW_INDICES]
        mapped_rank: list[dict[str, Any]] = []
        validations = []
        for selected_index, (window_index, window) in enumerate(
            zip(SELECTED_WINDOW_INDICES, selected)
        ):
            mapped, validation = map_graph_window(
                trace_events,
                window=window,
                rank=rank,
                step_index=window_index,
            )
            _validate_step_signatures(validation, rank=rank, step=window_index)
            mapped_rank.extend(mapped)
            validations.append(validation)
            if rank == 0:
                reference_steps.append(
                    {
                        "step_index": selected_index,
                        "label": f"stable AgentX decode iteration {window_index}",
                        "trace_start_us": window.start_us,
                        "duration_us": window.end_us - window.start_us,
                        "events": attach_graph_stack_evidence(
                            mapped, mapping_path=args.eager_mapping
                        ),
                    }
                )
        rank_events[rank] = mapped_rank
        rank_metrics[rank] = _metrics_for_rank(mapped_rank, len(selected))
        rank_wall_ms[rank] = [
            (window.end_us - window.start_us) / 1000.0 for window in selected
        ]
        rank_validation[rank] = validations
        all_mapping_rows.extend(mapped_rank)

    reference_events = [event for step in reference_steps for event in step["events"]]
    stack_us = sum(
        float(event["dur_us"]) for event in reference_events if event.get("python_stack")
    )
    total_reference_us = sum(float(event["dur_us"]) for event in reference_events)
    stack_ratio = stack_us / total_reference_us if total_reference_us else 0.0
    if stack_ratio < 0.95:
        raise ValueError(
            f"eager stack transfer covers {stack_ratio:.4f} of reference-rank residency; "
            "required >= 0.95"
        )

    critical_wall_ms = [
        max(rank_wall_ms[rank][index] for rank in rank_wall_ms)
        for index in range(len(SELECTED_WINDOW_INDICES))
    ]
    timing_summary = {
        "semantics": "critical wall time is maximum across ranks; GPU residency is never summed across ranks",
        "critical_wall_ms": {
            "samples": [round(value, 6) for value in critical_wall_ms],
            "mean": round(statistics.fmean(critical_wall_ms), 6),
            "median": round(statistics.median(critical_wall_ms), 6),
            "min": round(min(critical_wall_ms), 6),
            "max": round(max(critical_wall_ms), 6),
        },
        "rank_wall_ms": {
            str(rank): [round(value, 6) for value in values]
            for rank, values in rank_wall_ms.items()
        },
    }
    timeline = build_timeline_artifact(
        profile_id=PROFILE_ID,
        phase="decode",
        reference_rank=0,
        steps=reference_steps,
        timing_summary=timing_summary,
        raw_trace={
            "file": paths_by_rank[0].name,
            "sha256": sha256_file(paths_by_rank[0]),
            "format": "PyTorch profiler trace JSON gzip",
            "rank": 0,
        },
        stack_source={
            "mode": "eager_trace_transfer",
            "file": args.eager_mapping.name,
            "sha256": sha256_file(args.eager_mapping),
            "mapped_residency_ratio": round(stack_ratio, 6),
            "policy": "exact kernel+IR, representative IR node, then declared containing IR scope",
        },
    )

    total_us = sum(float(event["dur_us"]) for event in all_mapping_rows)
    status_us: Counter[str] = Counter()
    for event in all_mapping_rows:
        status_us[str(event["mapping_status"])] += float(event["dur_us"])
    attributed_ratio = (
        (status_us["mapped"] + status_us["fusion"]) / total_us if total_us else 0.0
    )
    profile = {
        "schema_version": "profile.v2",
        "profile_id": PROFILE_ID,
        "label": "Qwen3.5 397B · SGLang · AgentX DEP4 + MTP6 · CUDA Graph decode · global BS32",
        "model_id": "qwen35_397b_a17b",
        "execution_path_id": "attention_dp4_moe_ep4",
        "implementation_id": "sglang_85c23c62_attention_dp4_moe_ep4_mtp",
        "variant_id": "sglang_agentx_dep4_mtp6_cg_decode_gbs32_128x64",
        "phase": "decode",
        "generation_mode": "mtp",
        "entry_view": "generation_loop",
        "execution_parameters": {"tp_size": 4, "dp_size": 4, "cp_size": 1, "ep_size": 4},
        "hardware": {"gpu": "GB300", "gpus_per_node": 4, "nodes": 1},
        "workload": {
            "isl": 128,
            "osl": 64,
            "batch_size": 32,
            "batch_size_scope": "global_request_count",
            "per_rank_batch_size": [8, 8, 8, 8],
            "waiting_requests": 0,
            "waiting_uncached_tokens": 0,
            "selected_stable_iterations": list(SELECTED_WINDOW_INDICES),
        },
        "profiler": {
            "type": "torch",
            "rank": "all four DEP ranks",
            "activities": ["CPU", "GPU"],
            "cuda_graph_enabled": True,
            "with_stack": False,
            "record_shapes": False,
            "gpu_metric_semantics": "maximum per-rank kernel residency; parallel ranks are not summed",
        },
        "evidence": {
            "job_id": 3204736,
            "source_commit": SOURCE_COMMIT,
            "runtime_source_commit": RUNTIME_SOURCE_COMMIT,
            "model_revision": MODEL_REVISION,
            "protocol_file": args.protocol.name,
            "protocol_sha256": sha256_file(args.protocol),
            "trace_files": [
                {"rank": rank, "file": path.name, "sha256": sha256_file(path)}
                for rank, path in sorted(paths_by_rank.items())
            ],
            "eager_mapping_file": args.eager_mapping.name,
            "eager_mapping_sha256": sha256_file(args.eager_mapping),
            "mapping_policy": "unique signatures plus exact GGGA layer order and eager-stack transfer",
            "mapped_or_fusion_duration_ratio": round(attributed_ratio, 6),
            "strict_signature_duration_ratio": round(status_us["mapped"] / total_us, 6),
            "eager_stack_transfer_duration_ratio": round(stack_ratio, 6),
            "critical_decode_step_ms": timing_summary["critical_wall_ms"],
        },
        "timeline": {},
        "node_states": SGLANG_NODE_STATES,
        "node_metrics": _aggregate_rank_metrics(rank_metrics),
    }
    analysis = {
        "profile_id": PROFILE_ID,
        "rank_wall_ms": rank_wall_ms,
        "critical_wall_ms": critical_wall_ms,
        "rank_validation": rank_validation,
        "status_duration_us": dict(status_us),
        "mapped_or_fusion_duration_ratio": attributed_ratio,
        "strict_signature_duration_ratio": status_us["mapped"] / total_us,
        "eager_stack_transfer_duration_ratio": stack_ratio,
        "node_metrics": profile["node_metrics"],
        "protocol_formal": protocol["formal"],
    }
    return profile, timeline, analysis, all_mapping_rows


def main() -> int:
    args = parse_args()
    profile, timeline, analysis, mapping_rows = build(args)
    timeline_sha = write_timeline_artifact(args.output_timeline, timeline)
    profile["timeline"] = {
        "schema_version": "timeline.v1",
        "artifact": args.output_timeline.name,
        "sha256": timeline_sha,
        "reference_rank": 0,
        "step_count": len(timeline["steps"]),
        "event_count": sum(len(step["events"]) for step in timeline["steps"]),
        "raw_trace_file": timeline["raw_trace"]["file"],
    }
    args.output_profile.parent.mkdir(parents=True, exist_ok=True)
    args.output_profile.write_text(
        yaml.safe_dump(profile, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    args.output_analysis.parent.mkdir(parents=True, exist_ok=True)
    args.output_analysis.write_text(json.dumps(analysis, indent=2) + "\n")
    args.output_mapping.parent.mkdir(parents=True, exist_ok=True)
    with args.output_mapping.open("w") as output:
        for row in mapping_rows:
            output.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(f"wrote {args.output_profile.resolve()}")
    print(f"wrote {args.output_timeline.resolve()}")
    print(
        f"critical mean={profile['evidence']['critical_decode_step_ms']['mean']:.3f} ms "
        f"attributed={profile['evidence']['mapped_or_fusion_duration_ratio']:.3f} "
        f"stack={profile['evidence']['eager_stack_transfer_duration_ratio']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
