#!/usr/bin/env python3
"""Build the four-rank SGLang target-only one-chunk 8K prefill profile."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.common.timeline_artifact import build_timeline_artifact, write_timeline_artifact
from models.common.trace_mapping import load_trace
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
    load_occurrence_stack_mapping,
    load_unique_eager_kernel_signatures,
    map_prefill_window,
    transfer_occurrence_stack_mapping,
)
from models.qwen35.profile.qwen35_timeline import QWEN35_TIMELINE_TARGETS


PROFILE_ID = "qwen35_sglang_attention_dp4_moe_ep4_target_prefill_8k"
PREFILL_ANNOTATION = "step[EXTEND bs=1 toks=8192]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, nargs=4, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--eager-mapping", type=Path)
    parser.add_argument(
        "--occurrence-mappings",
        type=Path,
        nargs=4,
        help="four rank-local kernel_mapping.tpN.jsonl files from this exact trace set",
    )
    parser.add_argument("--attribution-job-id", type=int)
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
    occurrence_mode = bool(args.occurrence_mappings)
    if occurrence_mode == bool(args.eager_mapping):
        raise ValueError(
            "choose exactly one of --eager-mapping or --occurrence-mappings"
        )
    if occurrence_mode and args.attribution_job_id is None:
        raise ValueError("--attribution-job-id is required with occurrence transfer")
    protocol = _validate_protocol(args.protocol)
    eager_signatures = (
        {} if occurrence_mode else load_unique_eager_kernel_signatures(args.eager_mapping)
    )
    paths_by_rank = {trace_rank(path): path.resolve() for path in args.traces}
    if set(paths_by_rank) != {0, 1, 2, 3}:
        raise ValueError(f"incomplete four-rank trace coverage: {paths_by_rank}")

    occurrence_by_rank: dict[int, Path] = {}
    if occurrence_mode:
        for path in args.occurrence_mappings:
            match = re.search(r"\.tp(?P<rank>[0-3])\.jsonl$", path.name)
            if match is None:
                raise ValueError(f"cannot infer rank from occurrence mapping {path}")
            occurrence_by_rank[int(match.group("rank"))] = path.resolve()
        if set(occurrence_by_rank) != {0, 1, 2, 3}:
            raise ValueError(f"incomplete occurrence mappings: {occurrence_by_rank}")

    rank_metrics: dict[int, dict[str, Any]] = {}
    rank_validation: dict[int, dict[str, Any]] = {}
    rank_wall_ms: dict[int, float] = {}
    all_mapping_rows: list[dict[str, Any]] = []
    strict_signature_us = 0.0
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
            eager_signatures=eager_signatures,
        )
        strict_signature_us += validation["strict_signature_duration_ratio"] * sum(
            float(event["dur_us"]) for event in mapped
        )
        if occurrence_mode:
            mapping_path = occurrence_by_rank[rank]
            events_path = mapping_path.parent / f"events.tp{rank}.jsonl"
            source_mapped, _source_validation = load_occurrence_stack_mapping(
                events_path=events_path,
                mapping_path=mapping_path,
                rank=rank,
            )
            mapped, validation = transfer_occurrence_stack_mapping(
                mapped, source_mapped
            )
        rank_metrics[rank] = _metrics_for_rank(mapped, 1)
        rank_validation[rank] = validation
        rank_wall_ms[rank] = (end_us - start_us) / 1000.0
        all_mapping_rows.extend(mapped)
        if rank == 0:
            reference_start = start_us
            reference_events = (
                mapped
                if occurrence_mode
                else attach_graph_stack_evidence(
                    mapped, mapping_path=args.eager_mapping
                )
            )

    reference_total_us = sum(float(event["dur_us"]) for event in reference_events)
    stack_us = sum(
        float(event["dur_us"]) for event in reference_events if event.get("python_stack")
    )
    stack_ratio = stack_us / reference_total_us if reference_total_us else 0.0
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
        stack_source=(
            {
                "mode": "exact_occurrence_sequence_transfer",
                "files": [
                    {
                        "rank": rank,
                        "file": path.name,
                        "sha256": sha256_file(path),
                    }
                    for rank, path in sorted(occurrence_by_rank.items())
                ],
                "mapped_residency_ratio": round(stack_ratio, 6),
            }
            if occurrence_mode
            else {
                "mode": "eager_trace_transfer",
                "file": args.eager_mapping.name,
                "sha256": sha256_file(args.eager_mapping),
                "mapped_residency_ratio": round(stack_ratio, 6),
            }
        ),
        target_resolver=QWEN35_TIMELINE_TARGETS,
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
        "execution_parameters": {"tp_size": 1, "dp_size": 4, "cp_size": 1, "ep_size": 4},
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
            "mapping_policy": (
                "occurrence-local Python stacks transferred only across an exact contiguous kernel-name sequence, plus narrowly validated kernel/sequence slots; low-confidence generic occurrences remain unmapped"
                if occurrence_mode
                else "unique static signatures plus exact eager signatures that always resolve to one leaf; unresolved intervals remain unmapped"
            ),
            "mapped_or_fusion_duration_ratio": round(attributed_ratio, 6),
            "strict_signature_duration_ratio": round(strict_signature_us / total_us, 6),
            "mapped_duration_ratio": round(status_us["mapped"] / total_us, 6),
            "fusion_duration_ratio": round(status_us["fusion"] / total_us, 6),
            "unmapped_duration_ratio": round(status_us["unmapped"] / total_us, 6),
            "timeline_interval_coverage_ratio": round(sum(status_us.values()) / total_us, 6),
            "semantic_attribution_gate": {"threshold": 0.95, "passed": attributed_ratio >= 0.95},
            "eager_stack_transfer_duration_ratio": round(stack_ratio, 6),
            "occurrence_stack_files": (
                [
                    {
                        "rank": rank,
                        "file": path.name,
                        "sha256": sha256_file(path),
                    }
                    for rank, path in sorted(occurrence_by_rank.items())
                ]
                if occurrence_mode
                else []
            ),
            "attribution_job_id": args.attribution_job_id if occurrence_mode else None,
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
        "strict_signature_duration_ratio": strict_signature_us / total_us,
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
