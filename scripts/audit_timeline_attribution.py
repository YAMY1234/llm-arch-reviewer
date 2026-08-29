#!/usr/bin/env python3
"""Fail closed on unexplained production-timeline kernels.

Every CUDA interval must either bind to a stable IR node or carry an explicit
runtime/support class and reason.  A support label is not sufficient for a
kernel whose name still looks like model computation; those intervals remain
hard failures until the eager-to-production binding is fixed.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
from typing import Any


SEMANTIC_TOKENS = (
    "nvjet_",
    "gemm",
    "bmm_",
    "fmhasm",
    "_causal_conv1d_",
    "rmsnorm",
    "layer_norm",
    "l2norm",
    "kda_",
    "chunk_gla",
    "chunk_gated_delta",
    "chunk_kda",
    "moe::",
    "routingindices",
    "activationdeepseek",
    "finalizekernel",
    "nccl",
    "multimem_all_reduce",
    "all_reduce",
    "allgather",
    "reduce_scatter",
    "kpool_softmax_rotate_write_cache",
    "kpool_tail_seed",
    "topkperrow",
    "convert_req_index_to_global_index",
    "expand_pools_and_append_tail",
    "fused_q_kv_rmsnorm",
    "sm100_mqa_logits",
    "per_token_group_quant",
    "act_and_mul",
    "silu",
    "mhc_",
)

EXPLICIT_METADATA_KERNELS = (
    "paged_mqa_logits_metadata",
    "kpool_build_ragged_layout",
    "topk_plan",
)


def _string(strings: list[str], value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, int):
        return strings[value]
    return str(value)


def audit_timeline(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt") as source:
        artifact = json.load(source)
    strings = artifact.get("strings") or []
    counts: Counter[str] = Counter()
    durations: defaultdict[str, float] = defaultdict(float)
    failures: list[dict[str, Any]] = []
    total_count = 0
    total_us = 0.0
    mapped_count = 0
    mapped_us = 0.0

    for step in artifact.get("steps") or []:
        for event in step.get("events") or []:
            total_count += 1
            duration = float(event.get("duration_us") or 0.0)
            total_us += duration
            node = _string(strings, event.get("ir_node"))
            if node:
                mapped_count += 1
                mapped_us += duration
                continue

            name = _string(strings, event.get("kernel_name")) or ""
            support_class = _string(strings, event.get("support_class"))
            support_reason = _string(strings, event.get("support_reason"))
            if not support_class or not support_reason:
                failures.append(
                    {
                        "reason": "unclassified_unbound_kernel",
                        "kernel": name,
                        "duration_us": duration,
                        "step": step.get("step_index"),
                    }
                )
                continue

            counts[support_class] += 1
            durations[support_class] += duration
            lowered = name.lower()
            metadata_exception = any(
                token in lowered for token in EXPLICIT_METADATA_KERNELS
            )
            if not metadata_exception and any(
                token in lowered for token in SEMANTIC_TOKENS
            ):
                failures.append(
                    {
                        "reason": "semantic_kernel_left_outside_ir",
                        "kernel": name,
                        "duration_us": duration,
                        "support_class": support_class,
                        "step": step.get("step_index"),
                    }
                )

    return {
        "profile_id": artifact.get("profile_id"),
        "timeline": str(path),
        "total_kernel_count": total_count,
        "total_residency_us": round(total_us, 6),
        "mapped_kernel_count": mapped_count,
        "mapped_residency_us": round(mapped_us, 6),
        "mapped_kernel_count_ratio": round(mapped_count / total_count, 8)
        if total_count
        else 0.0,
        "mapped_residency_ratio": round(mapped_us / total_us, 8)
        if total_us
        else 0.0,
        "support_counts": dict(sorted(counts.items())),
        "support_residency_us": {
            key: round(value, 6) for key, value in sorted(durations.items())
        },
        "failure_count": len(failures),
        "failures": failures,
        "passed": not failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeline", action="append", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    reports = [audit_timeline(path.resolve()) for path in args.timeline]
    result = {
        "schema_version": "timeline-attribution-audit.v1",
        "passed": all(report["passed"] for report in reports),
        "profiles": reports,
    }
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload)
    print(payload, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
