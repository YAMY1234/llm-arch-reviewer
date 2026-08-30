#!/usr/bin/env python3
"""Map one Kimi K3 graph-off Torch trace onto stable IR nodes."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.common.trace_mapping import (
    BuildResult,
    KernelMapping,
    build_trace_mapping,
    validate_mappings,
    write_build_result,
)
from models.kimi_k3.build.kimi_k3_trace_rules import KIMI_K3_TRACE_RULES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--source-repo", default="https://github.com/sgl-project/sglang"
    )
    parser.add_argument(
        "--source-commit", default="25035bff8d34f3fcce2c1a2a5b1fe610225e84ed"
    )
    parser.add_argument("--rank", type=int, choices=range(8), required=True)
    parser.add_argument("--phase", choices=("prefill", "decode"), required=True)
    parser.add_argument("--expect-ms", type=float)
    return parser.parse_args()


def _replace_node(mapping: KernelMapping, node: str, reason: str) -> KernelMapping:
    evidence = list(mapping.evidence)
    if reason not in evidence:
        evidence.append(reason)
    return replace(
        mapping,
        selected_node=node,
        confidence="high",
        evidence=evidence,
    )


def _reconcile_kimi_sequences(result: BuildResult, phase: str) -> BuildResult:
    """Resolve fixed source-order groups that share one physical signature."""

    mappings_by_id = {mapping.event_id: mapping for mapping in result.mappings}
    if phase == "prefill":
        conv_events = sorted(
            (
                event
                for event in result.events
                if "causal_conv1d_fwd_kernel" in event.kernel_name
            ),
            key=lambda event: event.ts_us,
        )
        expected = 69 * 3
        if len(conv_events) != expected:
            raise ValueError(
                "KDA prefill Q/K/V convolution sequence mismatch: "
                f"expected={expected} observed={len(conv_events)}"
            )
        conv_nodes = ("kda.q_short_conv", "kda.k_short_conv", "kda.v_short_conv")
        for index, event in enumerate(conv_events):
            mappings_by_id[event.event_id] = _replace_node(
                mappings_by_id[event.event_id],
                conv_nodes[index % 3],
                "fixed_q_k_v_source_sequence",
            )

    # Every non-dense feed-forward occurrence is bounded by two ordered
    # AttnRes owner kernels.  This boundary survives auxiliary-stream launch
    # skew that can leave a router-front or routed-up GEMM with a stale runtime
    # setup stack at C1.  Reconcile only the fixed source-order positions:
    # [AttnRes, fused router front, route/top-k, experts, shared tail,
    #  routed collective, routed up, combine].
    anchors = [
        index
        for index, event in enumerate(result.events)
        if "attn_res_fused_tma_kernel" in event.kernel_name.lower()
    ]
    if len(anchors) != 186:
        raise ValueError(
            f"Kimi K3 AttnRes occurrence count mismatch: expected=186 observed={len(anchors)}"
        )
    segments = [
        (0, anchors[0]),
        *zip(anchors[:-1], anchors[1:]),
        (anchors[-1], len(result.events)),
    ]
    for segment_id, (start, stop) in enumerate(segments):
        if segment_id == 0 or segment_id == 186 or segment_id % 2 == 0:
            continue
        layer_id = (segment_id - 1) // 2
        if layer_id == 0:  # the checkpoint's only dense feed-forward layer
            continue
        segment = result.events[start:stop]
        route_offset = next(
            (
                offset
                for offset, event in enumerate(segment)
                if "route_" in event.kernel_name.lower()
            ),
            None,
        )
        if route_offset is None:
            raise ValueError(f"MoE segment {segment_id} has no route/top-k landmark")
        for event in segment[1:route_offset]:
            name = event.kernel_name.lower()
            if any(token in name for token in ("nvjet_", "tgvgemm", "splitkreduce")):
                mappings_by_id[event.event_id] = _replace_node(
                    mappings_by_id[event.event_id],
                    "stable_latent_moe.router_logits",
                    "attn_res_bounded_moe_router_front",
                )
        routed_collective_offset = next(
            (
                offset
                for offset, event in enumerate(segment)
                if mappings_by_id[event.event_id].selected_node
                == "stable_latent_moe.tp_routed_latent_collective"
            ),
            None,
        )
        combine_offset = next(
            (
                offset
                for offset, event in enumerate(segment)
                if mappings_by_id[event.event_id].selected_node
                == "stable_latent_moe.combine"
            ),
            None,
        )
        if routed_collective_offset is None or combine_offset is None:
            raise ValueError(f"MoE segment {segment_id} tail landmarks are incomplete")
        for event in segment[routed_collective_offset + 1 : combine_offset]:
            name = event.kernel_name.lower()
            if any(token in name for token in ("nvjet_", "tgvgemm", "gemm_ag_gemv")):
                mappings_by_id[event.event_id] = _replace_node(
                    mappings_by_id[event.event_id],
                    "stable_latent_moe.routed_up",
                    "attn_res_bounded_moe_routed_up_tail",
                )
        for event in segment:
            if "routingindices" in event.kernel_name.lower():
                mappings_by_id[event.event_id] = _replace_node(
                    mappings_by_id[event.event_id],
                    "stable_latent_moe.dispatch",
                    "attn_res_bounded_moe_dispatch",
                )

    mappings = [mappings_by_id[mapping.event_id] for mapping in result.mappings]
    validation = validate_mappings(
        result.events,
        mappings,
        expected_phase=None,
        min_mapped_duration_ratio=1.0,
    )

    expected_phase_tokens = (
        ("_execute_extend", "forward_extend")
        if phase == "prefill"
        else ("_execute_decode", "forward_decode")
    )
    opposite_phase_tokens = (
        ("_execute_decode", "forward_decode")
        if phase == "prefill"
        else ("_execute_extend", "forward_extend")
    )
    expected_frames = 0
    missing_frames = 0
    for mapping in mappings:
        if not mapping.selected_node:
            continue
        raw = mapping.phase_frame.raw if mapping.phase_frame else ""
        if any(token in raw for token in opposite_phase_tokens):
            validation["errors"].append(
                f"{mapping.event_id} has opposite-phase frame {raw}"
            )
        if any(token in raw for token in expected_phase_tokens):
            expected_frames += 1
        else:
            missing_frames += 1
    validation["phase_contract"] = {
        "requested_phase": phase,
        "accepted_python_frames": list(expected_phase_tokens),
        "expected_frame_kernel_count": expected_frames,
        "missing_frame_kernel_count": missing_frames,
        "window_evidence": "exact SGLang CPU step joined by External id to all nested model GPU annotations",
        "opposite_phase_kernel_count": len(
            [error for error in validation["errors"] if "opposite-phase" in error]
        ),
    }
    validation["ok"] = not validation["errors"]
    return replace(result, mappings=mappings, validation=validation)


def main() -> int:
    args = parse_args()
    expected_phase = "_execute_extend" if args.phase == "prefill" else "_execute_decode"
    result = build_trace_mapping(
        trace_path=args.trace.resolve(),
        source_root=args.source_root.resolve(),
        source_repo=args.source_repo,
        source_commit=args.source_commit,
        config_path=None,
        rank=args.rank,
        phase=args.phase,
        rules=KIMI_K3_TRACE_RULES,
        expect_ms=args.expect_ms,
        n_iters=1,
        skip_first=False,
        expected_phase_frame=expected_phase,
    )
    result = _reconcile_kimi_sequences(result, args.phase)
    write_build_result(args.out_dir.resolve(), result, rank=args.rank)
    print(
        f"rank={args.rank} phase={args.phase} kernels={len(result.events)} "
        f"mapped_ratio={result.validation['mapped_duration_ratio']:.6f} "
        f"stack_ratio={result.validation['stack_duration_ratio']:.6f} "
        f"ok={result.validation['ok']}"
    )
    return 0 if result.validation["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
