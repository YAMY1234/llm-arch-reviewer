from models.common.trace_mapping import FrameRef
from models.glm52.build.glm52_trace_rules import classify_glm52_node
from models.glm52.build.build_glm52_production_profile import (
    anchor_segments,
    attribute_aggregate_graph_events,
    build_profile_node_states,
    expected_profile_nodes,
    trtllm_layer_collective_node,
)


def _stack(*frames: str) -> list[FrameRef]:
    return [FrameRef(raw=frame) for frame in frames]


def test_unique_moe_kernel_overrides_stale_indexer_span() -> None:
    node, confidence = classify_glm52_node(
        "bmm_E2m1_E2m1E2m1_Fp32_Ab16",
        None,
        _stack(
            "python/sglang/srt/layers/attention/dsa/dsa_indexer.py(1102): _get_topk_ragged",
            "python/sglang/srt/models/deepseek_v2.py(2198): forward",
        ),
    )
    assert (node, confidence) == ("moe.routed_experts", "high")


def test_collectives_are_split_by_semantic_boundary() -> None:
    kernel = "ncclDevKernel_AllReduce_Sum_bf16_RING_LL"
    assert classify_glm52_node(
        kernel,
        "record_param_comms",
        _stack(
            "python/sglang/srt/layers/communicator.py(1053): _gather_hidden_states_and_residual",
            "python/sglang/srt/models/deepseek_v2.py(2198): forward",
        ),
    )[0] == "dsa_attention.tp_attention_output_collective"
    assert classify_glm52_node(
        kernel,
        "record_param_comms",
        _stack("nn.Module: DeepseekV2MoE_0"),
    )[0] == "moe.tp_moe_output_collective"
    assert classify_glm52_node(
        kernel,
        "record_param_comms",
        _stack("nn.Module: DeepseekV2MLP_0"),
    )[0] == "dense_mlp.tp_dense_mlp_output_collective"


def test_deferred_decode_ffn_collective_uses_previous_layer_kind() -> None:
    kernel = "oneshotAllreduceFusionKernel"
    dense_stack = _stack(
        "python/sglang/srt/layers/communicator.py(550): prepare_attn",
        "nn.Module: DeepseekV2DecoderLayer_3",
        "python/sglang/srt/models/deepseek_v2.py(2198): forward",
    )
    moe_stack = _stack(
        "python/sglang/srt/layers/communicator.py(550): prepare_attn",
        "nn.Module: DeepseekV2DecoderLayer_4",
        "python/sglang/srt/models/deepseek_v2.py(2198): forward",
    )
    assert classify_glm52_node(kernel, None, dense_stack)[0] == (
        "dense_mlp.tp_dense_mlp_output_collective"
    )
    assert classify_glm52_node(kernel, None, moe_stack)[0] == (
        "moe.tp_moe_output_collective"
    )


def test_indexer_projection_kernels_are_separated() -> None:
    stack = _stack(
        "python/sglang/srt/layers/attention/dsa/dsa_indexer.py(727): _fused_q_prepare_and_store",
        "python/sglang/srt/models/deepseek_v2.py(2198): forward",
    )
    assert classify_glm52_node(
        "nvjet_sm103_tst_128x256_64x6_2x1_2cta_v_bz_TNT", "aten::mm", stack
    )[0] == "dsa_attention.index_q_projection"
    assert classify_glm52_node(
        "nvjet_sm103_tst_80x128_64x10_2x2_2cta_h_bz_TNN", "aten::mm", stack
    )[0] == "dsa_attention.index_k_gate_projection"


def test_shared_expert_does_not_aggregate_into_dense_mlp() -> None:
    stack = _stack(
        "nn.Module: RowParallelLinear_4",
        "python/sglang/srt/models/deepseek_v2.py(1024): _forward_shared_experts",
        "nn.Module: DeepseekV2MLP_4",
        "nn.Module: DeepseekV2MoE_4",
    )
    assert classify_glm52_node("nvjet", "aten::mm", stack)[0] == "moe.shared_expert_down"


def test_production_profile_states_cover_every_selected_execution_node() -> None:
    measured = {"top.embedding": {}}
    states = build_profile_node_states(
        node_metrics=measured,
        framework="trtllm",
        phase="decode",
    )
    assert expected_profile_nodes("trtllm") == set(measured) | set(states)
    assert states["top.token_ids"]["status"] == "structural"
    assert states["dsa_attention.kv_latent_split"] == {
        "status": "fused",
        "label": "implementation-fused Q/KV split and RoPE interval",
        "included_in": "dsa_attention.q_split_rope",
    }
    assert states["dsa_attention.tp_prefill_index_topk_all_gather"]["status"] == (
        "not_selected"
    )
    assert states["top.final_norm"]["status"] == "unmapped"
    assert all(
        states[node]["status"] == "not_selected"
        for node in states
        if node.startswith("mtp_extension.")
    )


def test_layer_segments_ignore_capture_specific_pre_anchor_prefix() -> None:
    anchor = "applyMLARopeAndAssignQKVKernelGeneration"
    eager = [
        {"kernel_name": name}
        for layer in range(78)
        for name in (anchor, f"layer_{layer}_body")
    ]
    production = [
        {"kernel_name": "embedding_preamble"},
        {"kernel_name": "wrapper_preamble"},
        *eager,
    ]
    eager_segments = anchor_segments(eager, "trtllm", "decode")
    production_segments = anchor_segments(production, "trtllm", "decode")
    assert len(eager_segments) == len(production_segments) == 78
    assert eager_segments[0] == (0, 2)
    assert production_segments[0] == (2, 4)


def test_trtllm_collective_schedule_uses_layer_and_slot_contract() -> None:
    assert trtllm_layer_collective_node(0, 0) == (
        "dsa_attention.tp_attention_output_collective"
    )
    assert trtllm_layer_collective_node(2, 1) == (
        "dense_mlp.tp_dense_mlp_output_collective"
    )
    assert trtllm_layer_collective_node(3, 1) == (
        "moe.tp_moe_output_collective"
    )


def test_aggregate_graph_event_remains_explicitly_unmapped() -> None:
    rows, report = attribute_aggregate_graph_events(
        [
            {
                "kind": "cuda_graph",
                "kernel_name": "CUDA Graph 42",
                "ts_us": 100.0,
                "dur_us": 12.5,
            }
        ]
    )
    assert rows[0]["node"] is None
    assert rows[0]["confidence"] == "unmapped"
    assert rows[0]["attribution_method"] == (
        "aggregate_cuda_graph_no_node_visibility"
    )
    assert report["observability"] == "aggregate_cuda_graph"
    assert report["mapped_kernel_duration_ratio"] == 0.0
