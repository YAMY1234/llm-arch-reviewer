#!/usr/bin/env python3
"""Map a Qwen3.5 vLLM eager trace onto canonical TP8 IR nodes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.common.trace_mapping import build_trace_mapping, write_build_result
from models.qwen35.build.qwen35_vllm_trace_rules import QWEN35_VLLM_TRACE_RULES


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--phase", choices=("vllm_prefill", "vllm_decode"), required=True)
    parser.add_argument("--n-iters", type=int, required=True)
    parser.add_argument(
        "--capture-contract",
        type=Path,
        required=True,
        help="Exact job/workload/formal-coordinate contract retained with every rank manifest.",
    )
    args = parser.parse_args()
    capture_contract = json.loads(args.capture_contract.read_text())
    result = build_trace_mapping(
        trace_path=args.trace.resolve(),
        source_root=args.source_root.resolve(),
        source_repo="https://github.com/vllm-project/vllm",
        source_commit="487ecf187d3dfe74d2cf6119a92881dba403c219",
        config_path=None,
        rank=args.rank,
        phase=args.phase,
        rules=QWEN35_VLLM_TRACE_RULES,
        n_iters=args.n_iters,
        skip_first=False,
        expected_phase_frame="execute_model",
        capture_contract=capture_contract,
    )
    write_build_result(args.out_dir.resolve(), result, rank=args.rank)
    print(
        f"validation ok={result.validation['ok']} kernels={result.validation['kernel_count']} "
        f"mapped_ratio={result.validation['mapped_duration_ratio']:.6f}"
    )
    return 0 if result.validation["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
