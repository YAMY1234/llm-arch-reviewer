#!/usr/bin/env python3
"""Build the four-rank SGLang target-only one-chunk 8K prefill profile."""

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
from models.common.trace_mapping import load_trace
from models.qwen35.profile.build_qwen35_sglang_decode_profile import (
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
    map_prefill_window,
)


PROFILE_ID = "qwen35_sglang_attention_dp4_moe_ep4_target_prefill_8k"
PREFILL_ANNOTATION = "step[EXTEND bs=1 toks=8192]"


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


def _validate_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text())
    formal = protocol.get("formal") or {}
    expected = {
        "kind": "prefill8k",
        "expected_ranks": 4,
        "model_revision": MODEL_REVISION,
    }
    formal_expected = {
        "global_batch": 4,
        "per_rank_batch": 1,
        "isl": 8192,
        "osl": 1,
        "profile_steps": 4,
        "dp_size": 4,
        "generation_mode": "target_prefill_isolation",
        "speculative_generation": False,
        "chunked_prefill_size_requested_global": 32768,
        "max_prefill_tokens_requested_global": 32768,
        "chunked_prefill_size_effective_per_dp_rank": 8192,
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
        "chunked_prefill_size": 8192,
        "max_prefill_tokens": 32768,
    }.items():
        if server.get(key) != value:
            mismatch[f"server_info.{key}"] = {
                "expected": value,
                "actual": server.get(key),
            }
    summaries = ((formal.get("response_summary") or {}).get("requests") or [])
    observed_ranks = sorted((item.get("meta_info") or {}).get("dp_rank") for item in summaries)
    if observed_ranks != [0, 1, 2, 3]:
        mismatch["formal.response_summary.dp_ranks"] = {
            "expected": [0, 1, 2, 3],
            "actual": observed_ranks,
        }
    if any((item.get("meta_info") or {}).get("prompt_tokens") != 8192 for item in summaries):
        mismatch["formal.response_summary.prompt_tokens"] = {
            "expected": [8192] * 4,
            "actual": [(item.get("meta_info") or {}).get("prompt_tokens") for item in summaries],
        }
    if mismatch:
        raise ValueError(f"prefill protocol mismatch: {mismatch}")
    return protocol


def _full_prefill_annotation(events: list[dict[str, Any]], *, rank: int) -> dict[str, Any]:
    candidates = [
        event
        for event in events
        if event.get("cat") == "gpu_user_annotation"
        and event.get("ph") == "X"
        and event.get("name") == PREFILL_ANNOTATION
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"rank {rank}: expected exactly one {PREFILL_ANNOTATION!r}, got {len(candidates)}"
        )
    return candidates[0]


def build(args: argparse.Namespace):
    protocol = _validate_protocol(args.protocol)
    paths_by_rank = {trace_rank(path): path.resolve() for path in args.traces}
    if set(paths_by_rank) != {0, 1, 2, 3}:
        raise ValueError(f"incomplete four-rank trace coverage: {paths_by_rank}")

    rank_metrics: dict[int, dict[str, Any]] = {}
    rank_validation: dict[int, dict[str, Any]] = {}
    rank_wall_ms: dict[int, float] = {}
    all_mapping_rows: list[dict[str, Any]] = []
    reference_events: list[dict[str, Any]] = []
    reference_start = 0.0
    for rank, path in sorted(paths_by_rank.items()):
        events = load_trace(path).get("traceEvents") or []
        annotation = _full_prefill_annotation(events, rank=rank)
        start_us = float(annotation["ts"])
        end_us = start_us + float(annotation["dur"])
        mapped, validation = map_prefill_window(
            events,
            start_us=start_us,
            end_us=end_us,
            rank=rank,
            step_index=0,
        )
        rank_metrics[rank] = _metrics_for_rank(mapped, 1)
        rank_validation[rank] = validation
        rank_wall_ms[rank] = (end_us - start_us) / 1000.0
        all_mapping_rows.extend(mapped)
        if rank == 0:
            reference_start = start_us
            reference_events = attach_graph_stack_evidence(
                mapped, mapping_path=args.eager_mapping
            )

    reference_total_us = sum(float(event["dur_us"]) for event in reference_events)
    stack_us = sum(
        float(event["dur_us"]) for event in reference_events if event.get("python_stack")
    )
    stack_ratio = stack_us / reference_total_us if reference_total_us else 0.0
    if stack_ratio < 0.95:
        raise ValueError(
            f"eager stack transfer covers {stack_ratio:.4f} of prefill residency; required >= 0.95"
        )

    critical_wall_ms = max(rank_wall_ms.values())
    timing_summary = {
        "semantics": "critical wall time is maximum across ranks; GPU residency is never summed across ranks",
        "critical_wall_ms": critical_wall_ms,
        "rank_wall_ms": {str(rank): value for rank, value in rank_wall_ms.items()},
    }
    timeline = build_timeline_artifact(
        profile_id=PROFILE_ID,
        phase="prefill",
        reference_rank=0,
        steps=[
            {
                "step_index": 0,
                "label": "one complete 8192-token target prefill chunk",
                "trace_start_us": reference_start,
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
            "mode": "eager_trace_transfer",
            "file": args.eager_mapping.name,
            "sha256": sha256_file(args.eager_mapping),
            "mapped_residency_ratio": round(stack_ratio, 6),
        },
    )

    total_us = sum(float(event["dur_us"]) for event in all_mapping_rows)
    status_us: Counter[str] = Counter()
    for event in all_mapping_rows:
        status_us[str(event["mapping_status"])] += float(event["dur_us"])
    attributed_ratio = (status_us["mapped"] + status_us["fusion"]) / total_us
    profile = {
        "schema_version": "profile.v2",
        "profile_id": PROFILE_ID,
        "label": "Qwen3.5 397B · SGLang · DEP4 target-only one-chunk 8K prefill",
        "model_id": "qwen35_397b_a17b",
        "execution_path_id": "attention_dp4_moe_ep4",
        "implementation_id": "sglang_85c23c62_attention_dp4_moe_ep4_mtp",
        "variant_id": "sglang_dep4_target_prefill_8192_cgoff",
        "phase": "prefill",
        "generation_mode": "mtp",
        "entry_view": "top",
        "execution_parameters": {"tp_size": 4, "dp_size": 4, "cp_size": 1, "ep_size": 4},
        "hardware": {"gpu": "GB300", "gpus_per_node": 4, "nodes": 1},
        "workload": {
            "isl": 8192,
            "osl": 1,
            "global_batch_size": 4,
            "per_rank_batch_size": 1,
            "chunk_tokens_per_rank": 8192,
            "one_chunk": True,
            "speculative_generation": False,
        },
        "profiler": {
            "type": "torch",
            "rank": "all four DEP ranks",
            "activities": ["CPU", "GPU"],
            "cuda_graph_enabled": False,
            "gpu_metric_semantics": "maximum per-rank kernel residency; parallel ranks are not summed",
        },
        "evidence": {
            "job_id": args.job_id,
            "source_commit": SOURCE_COMMIT,
            "runtime_source_commit": RUNTIME_SOURCE_COMMIT,
            "model_revision": MODEL_REVISION,
            "protocol_file": args.protocol.name,
            "protocol_sha256": sha256_file(args.protocol),
            "trace_files": [
                {"rank": rank, "file": path.name, "sha256": sha256_file(path)}
                for rank, path in sorted(paths_by_rank.items())
            ],
            "mapping_policy": "unique signatures plus exact GGGA layer order and eager-stack transfer",
            "mapped_or_fusion_duration_ratio": round(attributed_ratio, 6),
            "strict_signature_duration_ratio": round(status_us["mapped"] / total_us, 6),
            "eager_stack_transfer_duration_ratio": round(stack_ratio, 6),
            "critical_prefill_wall_ms": round(critical_wall_ms, 6),
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
    profile, timeline, analysis, mappings = build(args)
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
    args.output_profile.write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")
    args.output_analysis.parent.mkdir(parents=True, exist_ok=True)
    args.output_analysis.write_text(json.dumps(analysis, indent=2) + "\n")
    args.output_mapping.parent.mkdir(parents=True, exist_ok=True)
    with args.output_mapping.open("w") as output:
        for row in mappings:
            output.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(f"wrote {args.output_profile.resolve()}")
    print(
        f"critical={profile['evidence']['critical_prefill_wall_ms']:.3f} ms "
        f"attributed={profile['evidence']['mapped_or_fusion_duration_ratio']:.3f} "
        f"stack={profile['evidence']['eager_stack_transfer_duration_ratio']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
