#!/usr/bin/env python3
"""Map one GLM-5.3-Flash vLLM eager trace onto stable IR nodes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.common.trace_mapping import build_trace_mapping, write_build_result
from models.glm53_flash.build.glm53_vllm_trace_rules import GLM53_VLLM_TRACE_RULES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-repo", default="https://github.com/vllm-project/vllm")
    parser.add_argument(
        "--source-commit", default="487ecf187d3dfe74d2cf6119a92881dba403c219"
    )
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--phase", choices=("vllm_prefill", "vllm_decode"), required=True)
    parser.add_argument("--n-iters", type=int, required=True)
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
        rules=GLM53_VLLM_TRACE_RULES,
        n_iters=args.n_iters,
        skip_first=False,
        expected_phase_frame="execute_model",
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
