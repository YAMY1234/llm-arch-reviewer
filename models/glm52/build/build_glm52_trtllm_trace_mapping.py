#!/usr/bin/env python3
"""Map one GLM-5.2 TensorRT-LLM eager Torch trace to canonical IR nodes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.common.trace_mapping import build_trace_mapping, write_build_result
from models.glm52.build.glm52_trtllm_trace_rules import (
    GLM52_ATTENTION_LAYERS_PER_FORWARD,
    GLM52_TRTLLM_TRACE_RULES,
    TRTLLM_DECODE_SIGNATURE,
    TRTLLM_PREFILL_SIGNATURE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-repo", default="https://github.com/NVIDIA/TensorRT-LLM")
    parser.add_argument(
        "--source-commit", default="4358fb5d5222f76ba133c3ae630aa2c06e62d073"
    )
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--phase", choices=("prefill", "decode"), required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    signature = (
        TRTLLM_PREFILL_SIGNATURE if args.phase == "prefill" else TRTLLM_DECODE_SIGNATURE
    )
    result = build_trace_mapping(
        trace_path=args.trace.resolve(),
        source_root=args.source_root.resolve(),
        source_repo=args.source_repo,
        source_commit=args.source_commit,
        config_path=None,
        rank=args.rank,
        phase=args.phase,
        rules=GLM52_TRTLLM_TRACE_RULES,
        n_iters=1,
        skip_first=False,
        signature_kernel=signature,
        expected_signature_count=GLM52_ATTENTION_LAYERS_PER_FORWARD,
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
