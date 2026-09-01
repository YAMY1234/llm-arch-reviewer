from models.common.trace_mapping import FrameRef
from models.glm52.build.glm52_trace_rules import classify_glm52_node
from models.glm52.build.build_glm52_production_profile import (
    assign_trtllm_decode_layer_schedules,
    assign_sglang_decode_layer_schedules,
    align_layer_segment,
    anchor_segments,
    attribute_aggregate_graph_events,
    build_profile_node_states,
    enrich_eager_semantics,
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


def test_graph_decode_moe_variants_use_frozen_eager_stack_semantics() -> None:
    moe_stack = _stack(
        "python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py(890): quantize_hidden_states_fp4",
        "python/sglang/srt/models/deepseek_v2.py(1024): forward_normal",
        "nn.Module: DeepseekV2MoE_3",
    )
    assert classify_glm52_node(
        "kernel_flashinfer_nvfp4_quantize_NVFP4QuantizeLinearKernel", None, moe_stack
    )[0] == "moe.dispatch"
    assert classify_glm52_node(
        "routingIndicesDynBlockKernel", None, []
    )[0] == "moe.topk"


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


def test_layer_alignment_preserves_semantics_across_production_insertions() -> None:
    source = [
        {"kernel_name": "anchor_16x16", "selected_node": "attention"},
        {"kernel_name": "gemm_64x128", "selected_node": "projection"},
        {"kernel_name": "norm_128", "selected_node": "norm"},
    ]
    production = [
        {"kernel_name": "anchor_32x16"},
        {"kernel_name": "runtime_metadata"},
        {"kernel_name": "gemm_128x128"},
        {"kernel_name": "norm_256"},
    ]
    aligned = align_layer_segment(source, production)
    assert [
        (source_index, production_index)
        for source_index, production_index, _ in aligned
    ] == [(0, 0), (1, 2), (2, 3)]


def test_eager_enrichment_uses_frozen_stack_without_overwriting_existing_node() -> None:
    rows = [
        {
            "event_id": "e1",
            "kernel_name": "bmm_E2m1_E2m1E2m1_Fp32_Ab16",
            "cpu_op_name": None,
            "selected_node": None,
            "semantic_frame": {
                "raw": "python/sglang/srt/models/deepseek_v2.py(1024): forward_normal"
            },
            "model_context_frame": {"raw": "nn.Module: DeepseekV2MoE_3"},
        },
        {
            "event_id": "e2",
            "kernel_name": "bmm_E2m1_E2m1E2m1_Fp32_Ab16",
            "cpu_op_name": None,
            "selected_node": "already.reviewed",
        },
    ]
    assert enrich_eager_semantics(rows, framework="sglang") == 1
    assert rows[0]["selected_node"] == "moe.routed_experts"
    assert rows[1]["selected_node"] == "already.reviewed"


def test_sglang_decode_schedule_closes_graph_shape_variants_inside_layer() -> None:
    names = [
        "fmhaSm100fKernel_QkvE4m3OBfloat16HQk576",
        "nvjet_sm103_tst_64x8_64x16_2x2_h_bz_TNT",
        "nvjet_sm103_tst_64x16_64x16_4x1_v_bz_TNT",
        "twoshotAllreduceKernel",
        "nvjet_sm103_tst_128x32_64x16_2x1_h_bz_TNT",
        "silu_and_mul_kernel",
        "nvjet_sm103_tst_128x16_64x16_2x1_h_bz_TNT",
        "twoshotAllreduceKernel",
        "nvjet_sm103_tst_128x8_64x16_2x1_h_bz_TNT",
        "RMSNormKernel_oi642048",
        "RMSNormKernel_oi64512",
        "catArrayBatchedCopy",
        "nvjet_sm103_tst_32x8_64x16_2x1_h_bz_TNT",
        "TgvGemmCuteExtKernel",
        "fused_q_indexer_rope_hadamard_quant",
        "nvjet_sm103_tst_32x64_64x16_4x1_v_bz_splitK_TNN",
        "splitKreduce_kernel",
        "fused_k_indexer_norm_rope_store",
        "topk_main_kernel",
        "nvjet_sm103_tst_64x8_64x16_4x2_h_bz_TNT",
    ]
    rows = [{"kernel_name": name, "node": None} for name in names]
    nodes = {
        "dsa_attention.q_a_projection",
        "dsa_attention.q_a_norm",
        "dsa_attention.kv_a_norm",
        "dsa_attention.q_b_projection",
        "dsa_attention.index_k_gate_projection",
        "dsa_attention.index_q_projection",
        "dsa_attention.latent_kv_reconstruction",
        "dsa_attention.output_projection",
        "dense_mlp.gate_up_projection",
        "dense_mlp.down_projection",
    }
    examples = {
        node: {"selected_node": node, "event_id": f"eager-{node}"}
        for node in nodes
    }
    assign_sglang_decode_layer_schedules(rows, [(0, len(rows))], examples)
    assert rows[1]["node"] == "dsa_attention.latent_kv_reconstruction"
    assert rows[2]["node"] == "dsa_attention.output_projection"
    assert rows[4]["node"] == "dense_mlp.gate_up_projection"
    assert rows[6]["node"] == "dense_mlp.down_projection"
    assert rows[8]["node"] == "dsa_attention.q_a_projection"
    assert rows[9]["node"] == "dsa_attention.q_a_norm"
    assert rows[10]["node"] == "dsa_attention.kv_a_norm"
    assert rows[12]["node"] == "dsa_attention.q_b_projection"
    assert rows[13]["node"] == "dsa_attention.index_q_projection"
    assert rows[15]["node"] == "dsa_attention.index_k_gate_projection"
    assert rows[16]["node"] == "dsa_attention.index_k_gate_projection"
    assert rows[19]["node"] == "dsa_attention.latent_kv_reconstruction"
    assert all(row.get("eager_event_id") for row in rows if row.get("node"))


def test_sglang_decode_schedule_pairs_moe_splitk_by_stream() -> None:
    rows = [
        {"kernel_name": "sparse_core", "stream_id": 0},
        {"kernel_name": "nvjet_sm103_tst_64x8_64x16_2x2_h_bz_TNT", "stream_id": 0},
        {"kernel_name": "nvjet_sm103_tst_64x16_64x16_4x1_v_bz_TNT", "stream_id": 0},
        {"kernel_name": "twoshotAllreduceKernel", "stream_id": 0},
        {"kernel_name": "nvjet_sm103_tss_32x64_64x16_4x1_v_bz_splitK_TNN", "stream_id": 7},
        {"kernel_name": "splitKreduce_kernel", "stream_id": 7},
        {"kernel_name": "nvjet_sm103_tst_64x64_64x16_4x1_v_bz_splitK_TNN", "stream_id": 9},
        {"kernel_name": "splitKreduce_kernel", "stream_id": 9},
        {"kernel_name": "silu_and_mul_kernel", "stream_id": 9},
        {"kernel_name": "nvjet_sm103_tst_64x32_64x16_4x1_v_bz_TNT", "stream_id": 9},
        {"kernel_name": "routingIndicesBlockScoresKernel", "stream_id": 0},
        {"kernel_name": "NVFP4QuantizeLinearKernel", "stream_id": 0},
        {"kernel_name": "twoshotAllreduceKernel", "stream_id": 0},
    ]
    nodes = {
        "dsa_attention.latent_kv_reconstruction",
        "dsa_attention.output_projection",
        "moe.router",
        "moe.shared_expert_up",
        "moe.shared_expert_down",
        "moe.topk",
        "moe.dispatch",
    }
    examples = {
        node: {"selected_node": node, "event_id": f"eager-{node}"}
        for node in nodes
    }
    assign_sglang_decode_layer_schedules(
        rows, [(0, 0), (0, 0), (0, 0), (0, len(rows))], examples
    )
    assert [rows[index]["node"] for index in (4, 5)] == ["moe.router"] * 2
    assert [rows[index]["node"] for index in (6, 7)] == [
        "moe.shared_expert_up"
    ] * 2
    assert rows[9]["node"] == "moe.shared_expert_down"
    assert rows[10]["node"] == "moe.topk"
    assert rows[11]["node"] == "moe.dispatch"


def test_trtllm_decode_schedule_binds_final_lm_head_from_raw_collective_boundary() -> None:
    rows = []
    segments = []
    for _ in range(78):
        start = len(rows)
        rows.extend(
            [
                {"kernel_name": "applyMLARopeAndAssignQKVKernelGeneration"},
                {"kernel_name": "twoshotAllreduceKernel"},
                {"kernel_name": "twoshotAllreduceKernel"},
            ]
        )
        segments.append((start, len(rows)))
    rows.extend(
        [
            {"kernel_name": "rmsNormLamportKernel"},
            {"kernel_name": "nvjet_sm103_tst_128x64_64x10_2x1_2cta_v_bz_TNT"},
            {"kernel_name": "ncclDevKernel_AllGather_RING_LL"},
        ]
    )
    examples = {
        node: {"selected_node": node, "event_id": f"eager-{node}"}
        for node in ("dsa_attention.q_split_rope", "top.lm_head")
    }

    assign_trtllm_decode_layer_schedules(rows, segments, examples)

    assert rows[-2]["node"] == "top.lm_head"
    assert rows[-2]["eager_event_id"] == "eager-top.lm_head"
