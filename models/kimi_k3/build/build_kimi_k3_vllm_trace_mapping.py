#!/usr/bin/env python3
"""Map one exact Kimi K3 vLLM graph-off trace onto stable IR nodes."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.common.trace_mapping import (  # noqa: E402
    BuildResult,
    KernelMapping,
    build_trace_mapping,
    validate_mappings,
    write_build_result,
)
from models.kimi_k3.build.kimi_k3_vllm_trace_rules import (  # noqa: E402
    ATTN_RES_CALLS_PER_FORWARD,
    KIMI_K3_VLLM_TRACE_RULES,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-repo", default="https://github.com/vllm-project/vllm")
    parser.add_argument(
        "--source-commit", default="680e2177e473ed8dfaa9773f7ead185b369cab46"
    )
    parser.add_argument("--rank", type=int, choices=range(8), required=True)
    parser.add_argument("--phase", choices=("prefill", "decode"), required=True)
    return parser.parse_args()


def _replace_node(mapping: KernelMapping, node: str, reason: str) -> KernelMapping:
    evidence = list(mapping.evidence)
    if reason not in evidence:
        evidence.append(reason)
    if reason.startswith("locked_") and "unique_kernel_signature" not in evidence:
        evidence.append("unique_kernel_signature")
    return replace(mapping, selected_node=node, confidence="high", evidence=evidence)


def _reconcile_source_order(result: BuildResult, phase: str) -> BuildResult:
    """Close only exact K3 sequences that a stack alone cannot distinguish."""

    mappings = {mapping.event_id: mapping for mapping in result.mappings}
    if phase == "prefill":
        conv = sorted(
            (
                event
                for event in result.events
                if "causal_conv1d" in event.kernel_name.lower()
            ),
            key=lambda event: event.ts_us,
        )
        expected = 69 * 3
        if len(conv) != expected:
            raise ValueError(
                "vLLM KDA prefill Q/K/V convolution sequence mismatch: "
                f"expected={expected} observed={len(conv)}"
            )
        nodes = ("kda.q_short_conv", "kda.k_short_conv", "kda.v_short_conv")
        for index, event in enumerate(conv):
            mappings[event.event_id] = _replace_node(
                mappings[event.event_id], nodes[index % 3], "locked_q_k_v_call_order"
            )

        mla_q_down = sorted(
            (
                event
                for event in result.events
                if "nvjet_sm103_tst_192x128_64x6_2x2_2cta_h_bz_tnt"
                in event.kernel_name.lower()
            ),
            key=lambda event: event.ts_us,
        )
        if len(mla_q_down) != 24:
            raise ValueError(
                "vLLM Kimi K3 prefill MLA q_down occurrence mismatch: "
                f"expected=24 observed={len(mla_q_down)}"
            )
        for event in mla_q_down:
            mappings[event.event_id] = _replace_node(
                mappings[event.event_id],
                "gated_mla.q_down",
                "locked_prefill_24_mla_q_down_occurrences",
            )

        ordered_events = sorted(result.events, key=lambda event: event.ts_us)
        gemm_rs_owner_counts: dict[str, int] = {}
        segment_nodes: set[str] = set()
        gemm_rs_count = 0
        for event in ordered_events:
            kernel = event.kernel_name.lower()
            if "attn_res" in kernel:
                segment_nodes.clear()
                continue
            mapping = mappings[event.event_id]
            if mapping.selected_node:
                segment_nodes.add(mapping.selected_node)
            if "gemm_rs_ar" not in kernel:
                continue
            gemm_rs_count += 1
            if any(node.startswith("gated_mla.") for node in segment_nodes):
                owner = "gated_mla.output_projection"
            elif any(node.startswith("kda.") for node in segment_nodes):
                owner = "kda.output_projection"
            elif any(node.startswith("dense_mlp.") for node in segment_nodes):
                owner = "dense_mlp.down"
            else:
                raise ValueError(
                    "vLLM Kimi K3 prefill GEMM-RS/AR lacks an exact semantic "
                    f"segment owner: {event.event_id}"
                )
            mappings[event.event_id] = _replace_node(
                mapping, owner, "locked_prefill_attn_res_bounded_gemm_rs_ar_owner"
            )
            gemm_rs_owner_counts[owner] = gemm_rs_owner_counts.get(owner, 0) + 1
        expected_gemm_rs_counts = {
            "kda.output_projection": 69,
            "gated_mla.output_projection": 24,
            "dense_mlp.down": 1,
        }
        if gemm_rs_count != 94 or gemm_rs_owner_counts != expected_gemm_rs_counts:
            raise ValueError(
                "vLLM Kimi K3 prefill GEMM-RS/AR owner mismatch: "
                f"expected={expected_gemm_rs_counts} observed={gemm_rs_owner_counts}"
            )

    if phase == "decode":
        exact_decode_groups = (
            (
                "stable_latent_moe.expert_gate_up",
                lambda name: name.lower().startswith("bmm_mxe4m3"),
                92,
                "locked_decode_92_routed_expert_gate_up_occurrences",
            ),
            (
                "stable_latent_moe.expert_down",
                lambda name: name.lower().startswith("bmm_bfloat16"),
                92,
                "locked_decode_92_routed_expert_down_occurrences",
            ),
            (
                "gated_mla.attention",
                lambda name: "fmhasm100fkernel" in name.lower(),
                24,
                "locked_decode_24_mla_attention_occurrences",
            ),
        )
        for node, predicate, expected, reason in exact_decode_groups:
            events = sorted(
                (event for event in result.events if predicate(event.kernel_name)),
                key=lambda event: event.ts_us,
            )
            if len(events) != expected:
                raise ValueError(
                    f"vLLM Kimi K3 decode {node} occurrence mismatch: "
                    f"expected={expected} observed={len(events)}"
                )
            for event in events:
                mappings[event.event_id] = _replace_node(
                    mappings[event.event_id], node, reason
                )

        all_gather = sorted(
            (
                event
                for event in result.events
                if "nccldevkernel_allgather" in event.kernel_name.lower()
            ),
            key=lambda event: event.ts_us,
        )
        if len(all_gather) != 1:
            raise ValueError(
                "vLLM Kimi K3 decode logits AllGather occurrence mismatch: "
                f"expected=1 observed={len(all_gather)}"
            )
        mappings[all_gather[0].event_id] = _replace_node(
            mappings[all_gather[0].event_id],
            "top.tp_logits_materialization",
            "locked_single_post_model_logits_allgather",
        )

        final_attn_res_ts = max(
            event.ts_us
            for event in result.events
            if "attn_res" in event.kernel_name.lower()
        )
        skinny_lm_head = [
            event
            for event in result.events
            if final_attn_res_ts < event.ts_us < all_gather[0].ts_us
            and "model_executorkernelslinearcute_dsl_skinny_gemm"
            in event.kernel_name.lower()
        ]
        if len(skinny_lm_head) > 1:
            raise ValueError(
                "vLLM Kimi K3 decode post-model skinny LM-head mismatch: "
                f"expected_at_most=1 observed={len(skinny_lm_head)}"
            )
        for event in skinny_lm_head:
            mappings[event.event_id] = _replace_node(
                mappings[event.event_id],
                "top.lm_head",
                "locked_post_final_norm_skinny_lm_head",
            )

    attn_res = sorted(
        [
            event
            for event in result.events
            if "attn_res" in event.kernel_name.lower()
        ],
        key=lambda event: event.ts_us,
    )
    if len(attn_res) != ATTN_RES_CALLS_PER_FORWARD:
        raise ValueError(
            "vLLM Kimi K3 AttnRes occurrence mismatch: "
            f"expected={ATTN_RES_CALLS_PER_FORWARD} observed={len(attn_res)}"
        )
    for index, event in enumerate(attn_res):
        node = (
            "top.output_attn_res"
            if index == ATTN_RES_CALLS_PER_FORWARD - 1
            else "attn_res.weighted_merge"
        )
        mappings[event.event_id] = _replace_node(
            mappings[event.event_id],
            node,
            "locked_187_attn_res_occurrence_order",
        )

    ordered = [mappings[mapping.event_id] for mapping in result.mappings]
    validation = validate_mappings(
        result.events,
        ordered,
        expected_phase=None,
        min_mapped_duration_ratio=1.0,
    )
    sample_nodes = {
        "top.final_norm",
        "top.lm_head",
        "top.logits",
        "top.tp_logits_materialization",
    }
    for mapping in ordered:
        if not mapping.selected_node or mapping.selected_node == "runtime.step_setup":
            continue
        phase_raw = mapping.phase_frame.raw if mapping.phase_frame else ""
        if (
            "unique_kernel_signature" in mapping.evidence
            or mapping.selected_node in KIMI_K3_VLLM_TRACE_RULES.kernel_only_nodes
        ):
            continue
        expected_scope = "sample" if mapping.selected_node in sample_nodes else "execute_model"
        if expected_scope not in phase_raw:
            validation["errors"].append(
                f"{mapping.event_id} maps to {mapping.selected_node} outside "
                f"the exact vLLM {expected_scope} scope: {phase_raw or 'missing'}"
            )
    validation["phase_contract"] = {
        "requested_phase": phase,
        "execute_context": f"vllm_{phase}",
        "attn_res_occurrence_count": len(attn_res),
        "expected_attn_res_occurrence_count": ATTN_RES_CALLS_PER_FORWARD,
        "phase_shape_rank_source_exact": True,
    }
    validation["ok"] = not validation["errors"]
    return replace(result, mappings=ordered, validation=validation)


def main() -> int:
    args = parse_args()
    phase = f"vllm_{args.phase}"
    result = build_trace_mapping(
        trace_path=args.trace.resolve(),
        source_root=args.source_root.resolve(),
        source_repo=args.source_repo,
        source_commit=args.source_commit,
        config_path=None,
        rank=args.rank,
        phase=phase,
        rules=KIMI_K3_VLLM_TRACE_RULES,
        n_iters=1,
        skip_first=False,
        expected_phase_frame="execute_model",
    )
    result = _reconcile_source_order(result, args.phase)
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
