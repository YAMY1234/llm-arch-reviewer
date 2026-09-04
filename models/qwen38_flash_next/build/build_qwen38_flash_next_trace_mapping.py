#!/usr/bin/env python3
"""Map one Qwen3.8-Flash-Next eager Torch trace onto stable IR nodes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.common.trace_mapping import build_trace_mapping, write_build_result
from models.qwen38_flash_next.build.qwen38_flash_next_trace_rules import make_qwen38_flash_next_trace_rules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument(
        "--config-name",
        choices=(
            "tp_only",
            "tp4_flashinfer_gdn",
            "dp_attention",
            "ep4_a2a_none",
            "dp_attention_ep4_deepep_deepgemm",
        ),
        default="tp_only",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--source-repo", default="https://github.com/Qiaolin-Yu/sglang-qwen-next"
    )
    parser.add_argument(
        "--source-commit", default="f90a941aa6ff71ac3bd7d40b8daccdf5bd914af0"
    )
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--phase", default="forward_decode")
    parser.add_argument("--expect-ms", type=float)
    parser.add_argument("--n-iters", type=int, default=1)
    parser.add_argument(
        "--skip-first", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_trace_mapping(
        trace_path=args.trace.resolve(),
        source_root=args.source_root.resolve(),
        source_repo=args.source_repo,
        source_commit=args.source_commit,
        config_path=None,
        rank=args.rank,
        phase=args.phase,
        rules=make_qwen38_flash_next_trace_rules(args.config_name),
        expect_ms=args.expect_ms,
        n_iters=args.n_iters,
        skip_first=args.skip_first,
    )
    write_build_result(args.out_dir.resolve(), result, rank=args.rank)
    print(f"wrote {args.out_dir.resolve()}")
    print(
        "validation: "
        f"ok={result.validation['ok']} "
        f"kernels={result.validation['kernel_count']} "
        f"mapped_ratio={result.validation['mapped_duration_ratio']:.3f}"
    )
    return 0 if result.validation["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
