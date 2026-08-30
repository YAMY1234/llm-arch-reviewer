#!/usr/bin/env python3
"""Map a Qwen3.5 SGLang eager trace onto canonical TP8 IR nodes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.common.trace_mapping import build_trace_mapping, write_build_result
from models.qwen35.build.qwen35_sglang_trace_rules import QWEN35_SGLANG_TRACE_RULES


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--phase", choices=("forward_extend", "forward_decode"), required=True)
    parser.add_argument("--n-iters", type=int, required=True)
    args = parser.parse_args()
    result = build_trace_mapping(
        trace_path=args.trace.resolve(),
        source_root=args.source_root.resolve(),
        source_repo="https://github.com/sgl-project/sglang",
        source_commit="033446bb05f35c0943aed2750c443077ffc0b92c",
        config_path=None,
        rank=args.rank,
        phase=args.phase,
        rules=QWEN35_SGLANG_TRACE_RULES,
        n_iters=args.n_iters,
        skip_first=False,
        expected_phase_frame="_execute_extend" if args.phase == "forward_extend" else "_execute_decode",
    )
    write_build_result(args.out_dir.resolve(), result, rank=args.rank)
    print(
        f"validation ok={result.validation['ok']} kernels={result.validation['kernel_count']} "
        f"mapped_ratio={result.validation['mapped_duration_ratio']:.6f}"
    )
    return 0 if result.validation["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
