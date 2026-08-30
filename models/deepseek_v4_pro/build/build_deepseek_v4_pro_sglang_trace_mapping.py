#!/usr/bin/env python3
"""Map one DeepSeek-V4-Pro SGLang eager trace to stable contract nodes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.common.trace_mapping import build_trace_mapping, write_build_result
from models.deepseek_v4_pro.build.deepseek_v4_pro_sglang_trace_rules import (
    DEEPSEEK_V4_PRO_SGLANG_TRACE_RULES,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--phase", choices=("forward_extend", "forward_decode"), required=True)
    args = parser.parse_args()
    phase_frame = "_execute_extend" if args.phase == "forward_extend" else "_execute_decode"
    result = build_trace_mapping(
        trace_path=args.trace.resolve(),
        source_root=args.source_root.resolve(),
        source_repo="https://github.com/sgl-project/sglang",
        source_commit="71de97b264b04dcd514cf904003028aefe9775c8",
        config_path=None,
        rank=args.rank,
        phase=args.phase,
        rules=DEEPSEEK_V4_PRO_SGLANG_TRACE_RULES,
        n_iters=1,
        skip_first=False,
        expected_phase_frame=phase_frame,
        close_phase_tails=False,
    )
    write_build_result(args.out_dir.resolve(), result, rank=args.rank)
    print(
        f"ok={result.validation['ok']} kernels={result.validation['kernel_count']} "
        f"mapped={result.validation['mapped_kernel_count']} "
        f"ratio={result.validation['mapped_duration_ratio']:.6f}"
    )
    return 0 if result.validation["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
