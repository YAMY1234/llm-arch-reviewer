#!/usr/bin/env python3
"""Build the bounded eager SGLang Qwen3.5 prefill-attribution profile."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.common.timeline_artifact import build_timeline_artifact, write_timeline_artifact
from models.common.trace_mapping import (
    _primary_gpu_annotations,
    find_eagle_mtp_prefill_windows,
    load_trace,
)
from models.qwen35.profile.build_qwen35_sglang_decode_profile import (
    CONTAINER_SHA256,
    MODEL_CONFIG_SHA256,
    MODEL_REVISION,
    RUNTIME_SOURCE_COMMIT,
    SGLANG_NODE_STATES,
    SOURCE_COMMIT,
    _aggregate_rank_metrics,
    _metrics_for_rank,
    sha256_file,
    trace_rank,
)
from models.qwen35.profile.qwen35_graph_mapping import (
    attach_graph_stack_evidence,
    load_unique_eager_kernel_signatures,
    map_prefill_window,
)
from models.qwen35.profile.qwen35_timeline import QWEN35_TIMELINE_TARGETS


PROFILE_ID = "qwen35_sglang_attention_dp4_moe_ep4_mtp_prefill_attribution"
PREFILL_ANNOTATION = "step[EXTEND bs=1 toks=256]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, nargs=4, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--eager-mapping", type=Path, required=True)
    parser.add_argument("--job-id", type=int, required=True)
    parser.add_argument("--output-profile", type=Path, required=True)
    parser.add_argument("--output-timeline", type=Path, required=True)
    parser.add_argument("--output-analysis", type=Path, required=True)
    parser.add_argument("--output-mapping", type=Path, required=True)
    return parser.parse_args()


def validate_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text())
    formal = protocol.get("formal") or {}
    expected = {
        "kind": "attribution-prefill",
        "expected_ranks": 4,
        "model_revision": MODEL_REVISION,
    }
    formal_expected = {
        "batch": 4,
        "per_rank_batch": 1,
        "isl": 256,
        "osl": 8,
        "profile_steps": 1,
        "cuda_graph": False,
        "capture_scope": "one complete eager target prefill iteration with MTP enabled",
    }
    mismatch = {
        key: {"expected": value, "actual": protocol.get(key)}
        for key, value in expected.items()
        if protocol.get(key) != value
    }
    mismatch.update(
        {
            f"formal.{key}": {"expected": value, "actual": formal.get(key)}
            for key, value in formal_expected.items()
            if formal.get(key) != value
        }
    )
    server = protocol.get("server_info") or {}
    for key, value in {
        "disable_cuda_graph": True,
        "enable_dp_attention": True,
        "dp_size": 4,
        "tp_size": 4,
        "ep_size": 4,
        "speculative_num_steps": 5,
        "speculative_num_draft_tokens": 6,
    }.items():
        if server.get(key) != value:
            mismatch[f"server_info.{key}"] = {"expected": value, "actual": server.get(key)}
    request = (protocol.get("start_profile_response") or {}).get("request") or {}
    for key, value in {
        "num_steps": 1,
        "with_stack": True,
        "record_shapes": True,
        "merge_profiles": False,
    }.items():
        if request.get(key) != value:
            mismatch[f"start_profile_response.request.{key}"] = {
                "expected": value,
                "actual": request.get(key),
            }
    if mismatch:
        raise ValueError(f"prefill-attribution protocol mismatch: {mismatch}")
    return protocol


def _prefill_window(events: list[dict[str, Any]], rank: int):
    windows = find_eagle_mtp_prefill_windows(
        events, signature="fused_qkvzba_split"
    )
    if len(windows) != 1:
        raise ValueError(f"rank {rank}: expected one logical MTP prefill, got {len(windows)}")
    annotations, _track = _primary_gpu_annotations(
        events, name_prefix="step[EXTEND"
    )
    candidates = [
        event
        for event in annotations
        if windows[0].start_us <= float(event.get("ts", 0.0)) < windows[0].end_us
        and event.get("name") == PREFILL_ANNOTATION
    ]
    if len(candidates) != 2:
        raise ValueError(
            f"rank {rank}: expected target plus MTP-seed {PREFILL_ANNOTATION} ranges, "
            f"got {len(candidates)}"
        )
    return windows[0], float(candidates[1]["ts"])


def build(args: argparse.Namespace):
    protocol = validate_protocol(args.protocol)
    eager_signatures = load_unique_eager_kernel_signatures(args.eager_mapping)
    paths_by_rank = {trace_rank(path): path.resolve() for path in args.traces}
    if set(paths_by_rank) != {0, 1, 2, 3}:
        raise ValueError(f"incomplete four-rank trace coverage: {paths_by_rank}")

    rank_metrics = {}
    rank_validation = {}
    rank_wall_ms = {}
    reference_events = []
    reference_start_us = 0.0
    all_mappings = []
    for rank, path in sorted(paths_by_rank.items()):
        events = load_trace(path).get("traceEvents") or []
        window, mtp_seed_start_us = _prefill_window(events, rank)
        start_us = window.start_us
        end_us = window.end_us
        mapped, validation = map_prefill_window(
            events,
            start_us=start_us,
            end_us=end_us,
            rank=rank,
            step_index=0,
            mtp_seed_start_us=mtp_seed_start_us,
            eager_signatures=eager_signatures,
        )
        rank_metrics[rank] = _metrics_for_rank(mapped, 1)
        rank_validation[rank] = validation
        rank_wall_ms[rank] = (end_us - start_us) / 1000.0
        all_mappings.extend(mapped)
        if rank == 0:
            reference_start_us = start_us
            reference_events = attach_graph_stack_evidence(
                mapped, mapping_path=args.eager_mapping
            )

    reference_total_us = sum(float(event["dur_us"]) for event in reference_events)
    stack_us = sum(
        float(event["dur_us"]) for event in reference_events if event.get("python_stack")
    )
    stack_ratio = stack_us / reference_total_us if reference_total_us else 0.0
    critical_wall_ms = max(rank_wall_ms.values())
    timing_summary = {
        "semantics": "attribution-only eager timing; critical wall is max across ranks and residency is never summed",
        "critical_wall_ms": critical_wall_ms,
        "rank_wall_ms": rank_wall_ms,
        "profiler_overhead_warning": "with_stack and record_shapes are enabled; do not use this timing for framework performance comparison",
    }
    timeline = build_timeline_artifact(
        profile_id=PROFILE_ID,
        phase="prefill",
        reference_rank=0,
        steps=[
            {
                "step_index": 0,
                "label": "eager attribution · 256-token target prefill + MTP seed",
                "trace_start_us": reference_start_us,
                "duration_us": rank_wall_ms[0] * 1000.0,
                "events": reference_events,
            }
        ],
        timing_summary=timing_summary,
        raw_trace={
            "file": paths_by_rank[0].name,
            "sha256": sha256_file(paths_by_rank[0]),
            "format": "PyTorch profiler trace JSON gzip",
            "rank": 0,
        },
        stack_source={
            "mode": "direct_eager_trace",
            "file": args.eager_mapping.name,
            "sha256": sha256_file(args.eager_mapping),
            "mapped_residency_ratio": round(stack_ratio, 6),
        },
        target_resolver=QWEN35_TIMELINE_TARGETS,
    )

    total_us = sum(float(event["dur_us"]) for event in all_mappings)
    status_us: Counter[str] = Counter()
    for event in all_mappings:
        status_us[str(event["mapping_status"])] += float(event["dur_us"])
    attributed_ratio = (status_us["mapped"] + status_us["fusion"]) / total_us
    node_metrics = _aggregate_rank_metrics(rank_metrics)
    profile = {
        "schema_version": "profile.v2",
        "profile_id": PROFILE_ID,
        "label": "Qwen3.5 397B · SGLang · eager prefill attribution · DEP4 · stacks + shapes",
        "model_id": "qwen35_397b_a17b",
        "execution_path_id": "attention_dp4_moe_ep4",
        "implementation_id": "sglang_85c23c62_attention_dp4_moe_ep4_mtp",
        "variant_id": "sglang_mtp_dep4_eager_prefill_attribution_cgoff",
        "phase": "prefill",
        "generation_mode": "mtp",
        "entry_view": "top",
        "execution_parameters": {"tp_size": 1, "dp_size": 4, "cp_size": 1, "ep_size": 4},
        "hardware": {"gpu": "GB300", "gpus_per_node": 4, "nodes": 1},
        "workload": {
            "isl": 256,
            "osl": 8,
            "global_batch_size": 4,
            "per_rank_batch_size": 1,
            "mtp_enabled": True,
            "purpose": "kernel-to-source attribution; not a performance baseline",
        },
        "profiler": {
            "type": "torch",
            "rank": "all four DEP ranks",
            "activities": ["CPU", "GPU"],
            "cuda_graph_enabled": False,
            "with_stack": True,
            "record_shapes": True,
            "gpu_metric_semantics": "maximum per-rank kernel residency; parallel ranks are not summed",
            "runtime_launch_parallelism": {
                "framework_tp_size": 4,
                "attention_dp_size": 4,
                "moe_ep_size": 4,
                "normalization": "the framework TP process group carries replicated attention DP and sharded MoE EP; it is not semantic TP4",
            },
        },
        "evidence": {
            "job_id": args.job_id,
            "source_commit": SOURCE_COMMIT,
            "runtime_source_commit": RUNTIME_SOURCE_COMMIT,
            "model_revision": MODEL_REVISION,
            "model_config_sha256": MODEL_CONFIG_SHA256,
            "container_sha256": CONTAINER_SHA256,
            "protocol_file": args.protocol.name,
            "protocol_sha256": sha256_file(args.protocol),
            "trace_files": [
                {"rank": rank, "file": path.name, "sha256": sha256_file(path)}
                for rank, path in sorted(paths_by_rank.items())
            ],
            "eager_mapping_file": args.eager_mapping.name,
            "eager_mapping_sha256": sha256_file(args.eager_mapping),
            "mapped_or_fusion_duration_ratio": round(attributed_ratio, 6),
            "strict_signature_duration_ratio": round(status_us["mapped"] / total_us, 6),
            "mapped_duration_ratio": round(status_us["mapped"] / total_us, 6),
            "fusion_duration_ratio": round(status_us["fusion"] / total_us, 6),
            "unmapped_duration_ratio": round(status_us["unmapped"] / total_us, 6),
            "timeline_interval_coverage_ratio": round(sum(status_us.values()) / total_us, 6),
            "semantic_attribution_gate": {"threshold": 0.95, "passed": attributed_ratio >= 0.95},
            "direct_eager_stack_duration_ratio": round(stack_ratio, 6),
            "critical_attribution_wall_ms": round(critical_wall_ms, 6),
        },
        "timeline": {},
        "node_states": SGLANG_NODE_STATES,
        "node_metrics": node_metrics,
    }
    analysis = {
        "profile_id": PROFILE_ID,
        "rank_wall_ms": rank_wall_ms,
        "critical_wall_ms": critical_wall_ms,
        "rank_validation": rank_validation,
        "status_duration_us": dict(status_us),
        "mapped_or_fusion_duration_ratio": attributed_ratio,
        "strict_signature_duration_ratio": status_us["mapped"] / total_us,
        "direct_eager_stack_duration_ratio": stack_ratio,
        "node_metrics": node_metrics,
        "protocol_formal": protocol["formal"],
    }
    return profile, timeline, analysis, all_mappings


def main() -> int:
    args = parse_args()
    profile, timeline, analysis, mappings = build(args)
    timeline_sha = write_timeline_artifact(args.output_timeline, timeline)
    profile["timeline"] = {
        "schema_version": "timeline.v1",
        "artifact": args.output_timeline.name,
        "sha256": timeline_sha,
        "reference_rank": 0,
        "step_count": 1,
        "event_count": len(timeline["steps"][0]["events"]),
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
        for row in mappings:
            output.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(f"wrote {args.output_profile.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
