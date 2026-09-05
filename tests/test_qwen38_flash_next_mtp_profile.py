from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.qwen38_flash_next.build.build_qwen38_flash_next_mtp_eager_profile import (  # noqa: E402
    BASE_SOURCE_COMMIT,
    EXECUTION_PATH_ID,
    SOURCE_COMMIT,
    SOURCE_PATCH_COMPONENTS,
    SOURCE_PATCH_SHA256,
    _append_source_reviewed_mtp_decode_runtime_tail,
    _layer_context,
    _mtp_lm_head_range,
    _partition_mtp_draft_runtime,
    _reconcile_hyperconnection_structure,
    _reconcile_mtp_input_fusion,
    _reconcile_mtp_moe_structure,
    _resolve_mtp_generation_boundary_insert,
    _restore_mtp_scope_for_timing_inserts,
    _validated_mtp_proposal_update_indices,
    _validated_mtp_graph_output_selection_indices,
    build_metrics,
    mtp_node_states,
    validate_cudagraph_round,
)
from models.qwen38_flash_next.build.build_qwen38_flash_next_trace_mapping import (  # noqa: E402
    expected_stack_phase,
)


@pytest.mark.parametrize(
    "phase",
    [
        "eagle_mtp_decode",
        "mtp_decode",
        "eagle_mtp_cudagraph_decode",
        "mtp_cudagraph_decode",
    ],
)
def test_mtp_window_aliases_validate_the_concrete_decode_stack(phase: str) -> None:
    assert expected_stack_phase(phase) == (
        "forward_decode",
        "forward_extend",
        "draft_forward",
        "_draft_extend_for_decode",
        "run_eagle_verify",
        "forward_batch_generation",
        "event_loop_overlap",
        "cuda_graph_runner",
        "replay",
    )


def test_non_mtp_window_preserves_its_stack_phase() -> None:
    assert expected_stack_phase("forward_extend") == "forward_extend"


def test_mtp_profile_source_is_qwen4_main_with_qsa_hardening() -> None:
    assert EXECUTION_PATH_ID == "tp_only_eagle_mtp"
    assert BASE_SOURCE_COMMIT == "32e9cb5b95104dc3a10b96bafae7afa50052d94d"
    assert SOURCE_COMMIT == "32e9cb5b95104dc3a10b96bafae7afa50052d94d"
    assert SOURCE_PATCH_SHA256 == "07c22e094da7103011301ced5824134e0387b310a5a03df0579bdd7ed08f17b3"
    assert SOURCE_PATCH_COMPONENTS == [
        {"name": "qsa_hardening", "sha256": SOURCE_PATCH_SHA256}
    ]


def test_mtp_eager_stack_converts_class_local_layer_ids_to_global_schedule() -> None:
    linear = {
        "python_stack": [
            {"raw": "nn.Module: Qwen4ExpLinearDecoderLayer_3"},
        ]
    }
    attention = {
        "python_stack": [
            {"raw": "nn.Module: Qwen4ExpAttentionDecoderLayer_2"},
        ]
    }

    assert _layer_context(linear, "linear_attention.delta_rule") == (4, "linear")
    assert _layer_context(attention, "qsa_attention.attention_core") == (11, "full")


def test_mtp_cudagraph_round_requires_four_complete_rank_replays(
    tmp_path: Path,
) -> None:
    expected = ["step[TARGET_VERIFY bs=16]"] * 7
    record = {
        "round": "formal-1",
        "global_batch_size": 16,
        "profile_trigger": {
            "trigger": {
                "global_running_reqs": 16,
                "global_waiting_reqs": 0,
                "global_waiting_uncached_tokens": 0,
                "local_running_min": 16,
                "local_running_max": 16,
            }
        },
        "trace_files": [f"rank-{rank}.trace.json.gz" for rank in range(4)],
        "trace_step_summary": {
            f"rank-{rank}.trace.json.gz": {
                "cpu_step_names": expected,
                "primary_gpu_step_names": expected,
                "cuda_graph_launch_count": 14,
                "cuda_graph_launch_step_counts": {
                    "step[TARGET_VERIFY bs=16]": 7,
                },
                "cuda_graph_launch_iteration_counts": [2] * 7,
            }
            for rank in range(4)
        },
    }
    rounds = tmp_path / "rounds.jsonl"
    rounds.write_text(json.dumps(record) + "\n")

    assert validate_cudagraph_round(rounds, batch_size=16, profile_steps=7) == record


def test_mtp_cudagraph_memcpy32_setup_belongs_to_proven_successor() -> None:
    resolved = _resolve_mtp_generation_boundary_insert(
        {"kernel_name": "memcpy32_post"},
        {
            "node": "hyperconnection.mix",
            "layer_id": 4,
            "layer_kind": "linear",
            "substage": "mlp_hc_mix",
        },
        {
            "node": "moe.shared_expert",
            "kernel_label": "shared expert projection",
            "layer_id": 4,
            "layer_kind": "linear",
            "invocation_id": 4,
            "substage": "moe",
        },
        phase="eagle_mtp_cudagraph_decode",
    )

    assert resolved == {
        "kernel_label": "CUDA Graph setup for shared expert projection",
        "node": "moe.shared_expert",
        "layer_id": 4,
        "layer_kind": "linear",
        "invocation_id": 4,
        "substage": "moe",
        "attribution_method": "cudagraph_successor_boundary_context",
        "confidence": "high",
    }


def test_mtp_prefill_marks_decode_only_qsa_metadata_inactive() -> None:
    assert mtp_node_states("prefill")["mtp_qsa_attention.metadata"] == {
        "status": "not_in_selected_stage",
        "label": "decode-only QSA layout / valid-count metadata",
    }
    assert mtp_node_states("prefill")["mtp_generation.proposal_update"] == {
        "status": "not_in_selected_stage",
        "label": "not in selected prefill stage",
    }


def test_mtp_proposal_update_requires_the_reviewed_contract_sequence() -> None:
    events = [
        {
            "cpu_op_name": "aten::index",
            "kernel_name": "index_elementwise_kernel OpaqueType<4>",
        },
        {
            "cpu_op_name": "aten::index",
            "kernel_name": "index_elementwise_kernel OpaqueType<2>",
        },
        {"cpu_op_name": "aten::argmax", "kernel_name": "reduce ArgMaxOps<float>"},
        {"cpu_op_name": "aten::fill_", "kernel_name": "FillFunctor<float>"},
    ]

    assert _validated_mtp_proposal_update_indices(events, 0, len(events)) == [
        0,
        1,
        2,
        3,
    ]

    events[1]["kernel_name"] = "unrelated generic kernel"
    with pytest.raises(ValueError, match="proposal-update contract sequence changed"):
        _validated_mtp_proposal_update_indices(events, 0, len(events))



def test_mtp_graph_output_selection_and_overlapped_runtime_are_fail_closed() -> None:
    output = [{"kernel_name": "index_elementwise_kernel OpaqueType<2>"}]
    assert _validated_mtp_graph_output_selection_indices(output, 0, 1) == [0]

    setup = [{"kernel_name": f"setup-{index}"} for index in range(18)]
    setup[3]["kernel_name"] = "compute_position_kernel"
    setup[7]["kernel_name"] = "_qsa_graph_layout_kernel"
    setup[8]["kernel_name"] = "_qsa_graph_row_metadata_kernel"
    proposal = [
        {"kernel_name": "reduce ArgMaxOps<float>"},
        {"kernel_name": "FillFunctor<float>"},
        {
            "cpu_op_name": "aten::index",
            "kernel_name": "index_elementwise_kernel OpaqueType<8>",
        },
    ]
    actual_setup, actual_proposal = _partition_mtp_draft_runtime(setup + proposal)
    assert actual_setup == setup
    assert actual_proposal == proposal

    setup.pop()
    with pytest.raises(ValueError, match="input/QSA setup sequence changed"):
        _partition_mtp_draft_runtime(setup + proposal)


def test_mtp_graph_lm_head_accepts_only_exact_optional_row_select() -> None:
    events = [
        {"kernel_name": "index_elementwise_kernel OpaqueType<2>"},
        {"kernel_name": "nvjet_sm103_tst_64x8_64x16_4x1_v_bz_TNT"},
    ]

    assert _mtp_lm_head_range(events, 0, 2) == ([0], 1)
    assert _mtp_lm_head_range(events[1:], 0, 1) == ([], 0)

    events[0]["kernel_name"] = "index_elementwise_kernel OpaqueType<4>"
    with pytest.raises(ValueError, match="optional exact logits-row selection"):
        _mtp_lm_head_range(events, 0, 2)


def test_generic_routed_expert_between_mtp_topk_and_combine_gets_mtp_scope() -> None:
    events = [
        {
            "node": "mtp_moe.topk",
            "layer_id": 0,
            "layer_kind": "mtp",
            "invocation_id": "mtp:0",
            "attribution_method": "python_stack_ir_rule",
        },
        {
            "node": "moe.routed_experts",
            "layer_id": None,
            "layer_kind": None,
            "invocation_id": None,
            "attribution_method": "python_stack_semantic_fallback",
        },
        {
            "node": "mtp_moe.combine",
            "layer_id": 0,
            "layer_kind": "mtp",
            "invocation_id": "mtp:0",
            "attribution_method": "direct_signature_with_python_stack",
        },
    ]

    _reconcile_mtp_moe_structure(events, phase="decode")

    assert events[1] == {
        "node": "mtp_moe.routed_experts",
        "layer_id": 0,
        "layer_kind": "mtp",
        "invocation_id": "mtp:0",
        "substage": "mtp_draft_extend_moe",
        "attribution_method": "python_stack_semantic_fallback+mtp_moe_boundary_context",
        "confidence": "high",
    }


def test_mtp_moe_reconciliation_fails_closed_across_unrelated_kernel() -> None:
    events = [
        {"node": "mtp_moe.topk", "attribution_method": "direct_signature"},
        {"node": "moe.routed_experts", "attribution_method": "fallback"},
        {"node": "top.runtime_support", "attribution_method": "boundary"},
        {"node": "mtp_moe.combine", "attribution_method": "direct_signature"},
    ]

    _reconcile_mtp_moe_structure(events, phase="decode")

    assert events[1]["node"] == "moe.routed_experts"


def test_target_metadata_boundary_insert_stays_at_runtime_parent() -> None:
    resolved = _resolve_mtp_generation_boundary_insert(
        {"kernel_name": "generic index kernel"},
        {"node": "qsa_attention.metadata"},
        {"node": "linear_attention.recurrent_state"},
        phase="eagle_mtp_decode",
    )

    assert resolved == {
        "kernel_label": "target QSA/GDN metadata preparation",
        "node": "top.runtime_support",
        "layer_kind": None,
        "substage": "target_verify_runtime",
        "attribution_method": "generation_runtime_boundary_context",
        "confidence": "high",
    }


def test_closed_decode_interval_adds_only_reviewed_draft_select_tail() -> None:
    source = [
        {
            "kernel_name": "last draft-model kernel",
            "node": "top.runtime_support",
        }
    ]
    timing = source + [
        {"kernel_name": "direct_copy_kernel_cuda", "cpu_op_name": "aten::copy_"},
        {"kernel_name": "_gather_rows_kernel"},
        {"kernel_name": "assign_draft_cache_locs_contiguous"},
        {"kernel_name": "FillFunctor<long>"},
        {"kernel_name": "build_tree_efficient"},
        {"kernel_name": "assign_extend_cache_locs_uniform"},
    ]

    augmented = _append_source_reviewed_mtp_decode_runtime_tail(
        source, timing, phase="eagle_mtp_decode"
    )

    assert len(augmented) == len(timing)
    assert {event["node"] for event in augmented[1:]} == {
        "mtp_generation.draft_select"
    }
    assert {
        event["attribution_method"] for event in augmented[1:]
    } == {"source_reviewed_speculative_runtime_tail"}


def test_changed_decode_runtime_tail_stays_fail_closed() -> None:
    source = [{"kernel_name": "last draft-model kernel"}]
    timing = source + [
        {"kernel_name": "_gather_rows_kernel"},
        {"kernel_name": "unknown_new_kernel"},
        {"kernel_name": "assign_draft_cache_locs_contiguous"},
        {"kernel_name": "build_tree_efficient"},
        {"kernel_name": "assign_extend_cache_locs_uniform"},
    ]

    assert _append_source_reviewed_mtp_decode_runtime_tail(
        source, timing, phase="eagle_mtp_decode"
    ) == source


def test_prefill_hc_mix_groups_pair_to_combine_and_leave_two_final_mixes() -> None:
    events = []

    def add_stage(layer_kind: str, layer_id: int, stage: str, mtp: bool = False) -> None:
        prefix = "mtp_prefill_" if mtp else ""
        invocation = "mtp:0" if mtp else layer_id
        events.extend(
            [
                {"kernel_name": "grouped_gemma_rmsnorm_kernel", "node": "hyperconnection.branch_norm"},
                {
                    "kernel_name": f"mix_gemm_{len(events)}",
                    "node": "hyperconnection.mix",
                    "semantic_function": "_mix_compute",
                    "python_stack": [
                        {
                            "file": "sglang/srt/layers/hyperconnection.py",
                            "function": "mix",
                        }
                    ],
                },
                {
                    "kernel_name": "hc_combine_kernel",
                    "node": "hyperconnection.combine",
                    "layer_id": layer_id,
                    "layer_kind": layer_kind,
                    "invocation_id": invocation,
                    "substage": f"{prefix}{stage}_hc_combine",
                    "stack_evidence": {"event_id": f"combine-{len(events)}"},
                },
            ]
        )

    def add_final(node: str) -> None:
        events.extend(
            [
                {"kernel_name": "grouped_gemma_rmsnorm_kernel", "node": node},
                {
                    "kernel_name": f"final_mix_{len(events)}",
                    "node": node,
                    "semantic_function": "_mix_compute",
                    "python_stack": [
                        {
                            "file": "sglang/srt/layers/hyperconnection.py",
                            "function": "mix",
                        }
                    ],
                },
                {"kernel_name": "lm_head_kernel", "node": "wrong.runtime.node"},
                {"kernel_name": "_all_gather_kernel_inner", "node": "wrong.collective"},
                {
                    "kernel_name": "direct_copy_kernel_cuda logits materialization",
                    "cpu_op_name": "aten::copy_",
                    "node": "wrong.runtime.node",
                },
            ]
        )

    add_stage("linear", 0, "attn")
    add_stage("linear", 0, "mlp")
    add_final("top.final_hc_mix")
    add_stage("mtp", 0, "attn", mtp=True)
    add_stage("mtp", 0, "mlp", mtp=True)
    add_final("mtp_head.final_hc_mix")

    _reconcile_hyperconnection_structure(events, phase="prefill")

    target_mixes = [
        event
        for event in events
        if event.get("node") == "hyperconnection.mix"
        and event.get("layer_kind") == "linear"
    ]
    assert [event["substage"] for event in target_mixes] == [
        "attn_hc_mix",
        "mlp_hc_mix",
    ]
    assert sum(event.get("node") == "top.final_hc_mix" for event in events) == 2
    assert sum(event.get("node") == "mtp_head.final_hc_mix" for event in events) == 2
    assert sum(event.get("node") == "top.lm_head" for event in events) == 1
    assert sum(event.get("node") == "mtp_head.lm_head" for event in events) == 1
    assert sum(event.get("node") == "top.tp_logits_collective" for event in events) == 2
    assert sum(event.get("node") == "mtp_head.tp_logits_collective" for event in events) == 2


def test_mtp_input_fusion_splits_two_projection_pairs_and_residual_add() -> None:
    events = [
        {
            "kernel_name": "RMSNormKernel embedding",
            "node": "mtp_head.residual_fusion",
            "semantic_function": "_fuse_residual_linear_shared",
        },
        {
            "kernel_name": "embedding GEMM",
            "node": "mtp_head.residual_fusion",
            "semantic_function": "_fuse_residual_linear_shared",
        },
        {
            "kernel_name": "RMSNormKernel hidden",
            "node": "mtp_head.residual_fusion",
            "semantic_function": "_fuse_residual_linear_shared",
        },
        {
            "kernel_name": "hidden GEMM",
            "node": "mtp_head.residual_fusion",
            "semantic_function": "_fuse_residual_linear_shared",
        },
        {
            "kernel_name": "CUDAFunctor_add residual",
            "node": "mtp_head.residual_fusion",
            "semantic_function": "_fuse_residual_linear_shared",
        },
    ]

    _reconcile_mtp_input_fusion(events)

    assert [event["node"] for event in events] == [
        "mtp_head.embedding_projection",
        "mtp_head.embedding_projection",
        "mtp_head.hidden_projection",
        "mtp_head.hidden_projection",
        "mtp_head.residual_fusion",
    ]


def test_native_rmsnorm_mtp_input_fusion_preserves_projection_boundaries() -> None:
    native_norm = [
        "aten::copy_",
        "aten::pow",
        "aten::mean",
        "aten::add",
        "aten::rsqrt",
        "aten::mul",
        "aten::copy_",
        "aten::add",
        "aten::mul",
        "aten::copy_",
    ]
    cpu_ops = [*native_norm, "aten::mm", *native_norm, "aten::mm", "aten::add"]
    events = [
        {
            "kernel_name": op,
            "cpu_op_name": op,
            "node": "mtp_head.residual_fusion",
            "semantic_function": "_fuse_residual_linear_shared",
        }
        for op in cpu_ops
    ]

    _reconcile_mtp_input_fusion(events)

    assert [event["node"] for event in events[:11]] == [
        "mtp_head.embedding_projection"
    ] * 11
    assert [event["node"] for event in events[11:22]] == [
        "mtp_head.hidden_projection"
    ] * 11
    assert events[22]["node"] == "mtp_head.residual_fusion"


def test_split_hc_combine_requires_next_same_stream_gate_and_apply() -> None:
    from models.qwen38_flash_next.build.qwen38_flash_next_decode_attribution import (
        _hc_combine_ranges,
    )

    rows = [
        {"kernel_name": "hc_combine_gate_kernel", "stream": 7},
        {"kernel_name": "dense_bf16_gemm", "stream": 8},
        {"kernel_name": "hc_combine_apply_kernel", "stream": 7},
        {"kernel_name": "hc_combine_kernel", "stream": 7},
    ]
    assert _hc_combine_ranges(rows) == [(0, 2), (3, 3)]
    with pytest.raises(ValueError, match="lacks next-same-stream apply"):
        _hc_combine_ranges([{"kernel_name": "hc_combine_gate_kernel", "stream": 7}])


def test_pr37500_hc_mix_uses_exact_two_gemm_prefix() -> None:
    from models.qwen38_flash_next.build.qwen38_flash_next_decode_attribution import (
        _hc_mix_end,
    )

    rows = [
        {"kernel_name": "dense_bf16_gemm_mix_silu"},
        {"kernel_name": "dense_bf16_gemm_mix_gate"},
        {"kernel_name": "dense_bf16_gemm_attention_consumer"},
    ]
    assert _hc_mix_end(rows, 0, len(rows)) == 1


def test_ple_decode_accepts_prefetched_lookup_boundary() -> None:
    from models.qwen38_flash_next.build.qwen38_flash_next_decode_attribution import (
        _map_ple,
    )

    rows = [
        {"kernel_name": "allreduce", "node": "ple.tp_embedding_collective"},
        {"kernel_name": "dense_bf16_gemm", "node": None},
        {"kernel_name": "grouped_gemma_rmsnorm_kernel", "node": None},
        {"kernel_name": "conv_depthwise", "node": None},
    ]
    _map_ple(rows, 0, len(rows))
    assert [row["node"] for row in rows] == [
        "ple.tp_embedding_collective",
        "ple.key_value_projection",
        "ple.grouped_norm_gate",
        "ple.short_conv",
    ]


def test_native_mtp_rmsnorm_graph_signature_is_exact() -> None:
    from models.qwen38_flash_next.build.build_qwen38_flash_next_mtp_eager_profile import (
        _NATIVE_MTP_RMSNORM_KERNEL_PATTERNS,
        _is_native_mtp_rmsnorm_kernel_sequence,
    )

    rows = [{"kernel_name": pattern} for pattern in _NATIVE_MTP_RMSNORM_KERNEL_PATTERNS]
    assert _is_native_mtp_rmsnorm_kernel_sequence(rows, 0)
    rows[3] = {"kernel_name": "unexpected_add_kernel"}
    assert not _is_native_mtp_rmsnorm_kernel_sequence(rows, 0)


def _exact(node: str, layer_kind: str) -> dict:
    return {
        "kernel_name": node,
        "node": node,
        "layer_kind": layer_kind,
        "attribution_method": "python_stack_ir_rule+exact_sequence_timing_transfer",
    }


def test_timing_only_qsa_kernel_stays_in_mtp_scope() -> None:
    events = [
        _exact("mtp_qsa_attention.metadata", "mtp"),
        {
            "kernel_name": "_compact_kv",
            "node": "qsa_attention.attention_core",
            "layer_kind": "full",
            "attribution_method": "direct_signature_timing_insert",
        },
        _exact("mtp_qsa_attention.output_gate", "mtp"),
    ]

    _restore_mtp_scope_for_timing_inserts(events, phase="forward_decode")

    assert events[1]["node"] == "mtp_qsa_attention.attention_core"
    assert events[1]["layer_kind"] == "mtp"
    assert events[1]["substage"] == "mtp_draft_extend_attention"


def test_timing_only_hc_mix_keeps_generic_leaf_and_mtp_stage() -> None:
    left = _exact("hyperconnection.branch_norm", "mtp")
    left["substage"] = "mtp_draft_extend_attn_hc_mix"
    right = _exact("mtp_qsa_attention.qkv_gate_projection", "mtp")
    right["substage"] = "mtp_draft_extend_attention"
    events = [
        left,
        {
            "kernel_name": "_hc_mix_persistent_kernel",
            "node": "hyperconnection.mix",
            "layer_kind": None,
            "attribution_method": "direct_signature_timing_insert",
        },
        right,
    ]

    _restore_mtp_scope_for_timing_inserts(events, phase="forward_decode")

    assert events[1]["node"] == "hyperconnection.mix"
    assert events[1]["layer_kind"] == "mtp"
    assert events[1]["substage"] == "mtp_draft_extend_attn_hc_mix"


def test_timing_only_kernel_cannot_cross_target_mtp_boundary() -> None:
    events = [
        _exact("qsa_attention.output_projection", "full"),
        {
            "kernel_name": "_compact_kv",
            "node": "qsa_attention.attention_core",
            "layer_kind": "full",
            "attribution_method": "direct_signature_timing_insert",
        },
        _exact("mtp_qsa_attention.qkv_gate_projection", "mtp"),
    ]

    with pytest.raises(ValueError, match="target/MTP scope boundary"):
        _restore_mtp_scope_for_timing_inserts(events, phase="forward_decode")


def test_target_decoder_rollup_excludes_auxiliary_mtp_model() -> None:
    common = {
        "rank": 0,
        "step_index": 1,
        "stream": 23,
        "device": 0,
        "pid": 1,
        "tid": 23,
        "attribution_method": "python_stack_ir_rule",
        "confidence": "high",
    }
    target = {
        **common,
        "kernel_name": "target",
        "kernel_label": "target",
        "node": "linear_attention.delta_rule",
        "ts_us": 0.0,
        "dur_us": 1.0,
        "layer_id": 0,
        "layer_kind": "linear",
        "invocation_id": 0,
        "substage": "attention",
    }
    mtp = {
        **common,
        "kernel_name": "mtp",
        "kernel_label": "mtp",
        "node": "mtp_qsa_attention.attention_core",
        "ts_us": 1.0,
        "dur_us": 2.0,
        "layer_id": 0,
        "layer_kind": "mtp",
        "invocation_id": "mtp:0",
        "substage": "mtp_draft_extend_attention",
    }
    mtp_hc = {
        **common,
        "kernel_name": "mtp_hc",
        "kernel_label": "mtp_hc",
        "node": "hyperconnection.mix",
        "ts_us": 3.0,
        "dur_us": 3.0,
        "layer_id": 0,
        "layer_kind": "mtp",
        "invocation_id": "mtp:0",
        "substage": "mtp_draft_extend_runtime_hc",
    }

    metrics = build_metrics([target, mtp, mtp_hc], phase="decode", n_iters=1)

    assert metrics["top.decoder_stack"]["active_gpu_ms"] == pytest.approx(0.001)
    assert metrics["mtp_head.decoder_layer"]["active_gpu_ms"] == pytest.approx(0.005)
    assert metrics["mtp_generation.target_verify"]["active_gpu_ms"] == pytest.approx(0.001)
    assert metrics["mtp_generation.mtp_draft_extend"]["active_gpu_ms"] == pytest.approx(0.005)
    assert "hyperconnection.mix" not in metrics


def test_mtp_hc_stage_exposes_scoped_drill_metrics() -> None:
    common = {
        "rank": 0,
        "step_index": 1,
        "stream": 23,
        "device": 0,
        "pid": 1,
        "tid": 23,
        "layer_id": 0,
        "layer_kind": "mtp",
        "invocation_id": "mtp:0",
        "substage": "mtp_draft_extend_attn_hc_mix",
        "attribution_method": "python_stack_ir_rule",
        "confidence": "high",
    }
    events = [
        {
            **common,
            "kernel_name": "norm",
            "kernel_label": "norm",
            "node": "hyperconnection.branch_norm",
            "ts_us": 0.0,
            "dur_us": 1.0,
        },
        {
            **common,
            "kernel_name": "mix",
            "kernel_label": "mix",
            "node": "hyperconnection.mix",
            "ts_us": 1.0,
            "dur_us": 2.0,
        },
    ]

    metrics = build_metrics(events, phase="decode", n_iters=1)
    parent = metrics["mtp_layer.attn_hc_mix"]

    assert parent["active_gpu_ms"] == pytest.approx(0.003)
    assert parent["drill_view"] == "hyperconnection_mix"
    assert parent["drill_metrics"]["branch_norm"]["active_gpu_ms"] == pytest.approx(0.001)
    assert parent["drill_metrics"]["mix"]["active_gpu_ms"] == pytest.approx(0.002)
