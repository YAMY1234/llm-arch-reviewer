from models.common.trace_mapping import (
    FrameRef,
    KernelEvent,
    build_kernel_mappings,
    validate_mappings,
)
from models.glm53_flash.build.glm53_sglang_trace_rules import (
    GLM53_SGLANG_TRACE_RULES,
    classify_glm53_sglang_node,
)
from models.glm53_flash.build.glm53_sglang_production_attribution import (
    _assign_dense_production_schedule,
    _assign_dsa_production_stream_schedules,
    _assign_kda_production_stream_schedules,
    _assign_moe_shared_production_schedule,
    _classify_runtime_support,
    _production_direct_node,
)


def frames(*rows: str) -> list[FrameRef]:
    return [FrameRef(raw=row) for row in rows]


def test_kda_unique_kernels_override_neighboring_stack() -> None:
    assert classify_glm53_sglang_node(
        "chunk_kda_fwd_kernel_intra_sub_chunk",
        None,
        frames("python/sglang/srt/models/deepseek_v2.py(2206): forward_core"),
    ) == ("linear_attention.recurrent_update", "high")
    assert classify_glm53_sglang_node(
        "_causal_conv1d_fwd_kernel",
        None,
        frames("python/sglang/srt/layers/radix_linear_attention.py(79): forward"),
    ) == ("linear_attention.qkv_short_conv", "high")


def test_attention_collective_uses_exact_hybrid_layer_schedule() -> None:
    collective = "ncclDevKernel_AllReduce_Sum_bf16_RING_LL"
    common = "python/sglang/srt/layers/communicator_mhc.py(226): _gather_hidden_states_and_residual"
    assert classify_glm53_sglang_node(
        collective,
        "record_param_comms",
        frames(common, "nn.Module: Glm5NextDecoderLayer_2"),
    ) == ("linear_attention.tp_kda_output_collective", "high")
    assert classify_glm53_sglang_node(
        collective,
        "record_param_comms",
        frames(common, "nn.Module: Glm5NextDecoderLayer_3"),
    ) == ("dsa_attention.tp_dsa_output_collective", "high")


def test_mhc_fused_intervals_map_to_representative_stable_leaves() -> None:
    assert classify_glm53_sglang_node(
        "mhc_pre_big_fuse_with_norm_tilelang_kernel",
        None,
        frames("python/sglang/srt/models/glm5_next.py(794): hc_attn_pre"),
    ) == ("mhc_transform.pre_weights", "high")
    assert classify_glm53_sglang_node(
        "mhc_post_tilelang_kernel",
        None,
        frames("python/sglang/srt/models/glm5_next.py(815): hc_post"),
    ) == ("mhc_transform.residual_mix", "high")


def test_glm53_dsa_and_reused_moe_signatures_use_glm53_node_names() -> None:
    assert classify_glm53_sglang_node(
        "fmhaSm100fKernel_QkvE4m3OBfloat16H512PagedKvDenseDynamicTokenSparse",
        None,
        frames("python/sglang/srt/models/deepseek_v2.py(2206): forward_core"),
    ) == ("dsa_attention.sparse_mla_core", "high")
    assert classify_glm53_sglang_node(
        "bmm_E4m3_E4m3E4m3_Fp32_t128x64",
        "sglang::trtllm_fp8_block_scale_moe_out_wrapper",
        frames("python/sglang/srt/models/deepseek_v2.py(1064): forward_normal"),
    ) == ("moe.routed_gate_up", "high")


def test_unique_dsa_kernel_survives_stale_asynchronous_python_stack() -> None:
    event = KernelEvent(
        event_id="k_000001",
        kernel_name=(
            "fmhaSm100fKernel_QkvE4m3OBfloat16H512"
            "PagedKvDenseDynamicTokenSparse"
        ),
        ts_us=1.0,
        dur_us=2.0,
        stream=1,
        device=0,
        correlation=1,
        external_id=None,
        cpu_op_name=None,
        cpu_input_dims=None,
        cpu_input_types=None,
        python_stack=frames(
            "python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py(863): begin_insert"
        ),
    )
    mapping = build_kernel_mappings([event], GLM53_SGLANG_TRACE_RULES)[0]
    assert mapping.selected_node == "dsa_attention.sparse_mla_core"
    assert mapping.confidence == "high"
    assert "unique_kernel_signature" in mapping.evidence
    assert validate_mappings([event], [mapping], expected_phase=None)["ok"] is True


def test_dsa_hadamard_is_bounded_index_k_projection() -> None:
    assert _production_direct_node(
        "void sglang::fast_hadamard_transform_kernel<float>", "dsa"
    ) == "dsa_attention.index_k_projection"
    assert _production_direct_node(
        "void sglang::fast_hadamard_transform_kernel<float>", "kda"
    ) is None


def _production_row(name: str) -> dict[str, object]:
    return {"kernel_name": name, "node": None}


def test_dense_production_schedule_uses_activation_landmark() -> None:
    rows = [
        _production_row("per_token_group_quant"),
        _production_row("sm100_fp8_fp4_gemm_1d1d_impl"),
        _production_row("silu_mul_clamp"),
        _production_row("per_token_group_quant"),
        _production_row("sm100_fp8_fp4_gemm_1d1d_impl"),
    ]
    _assign_dense_production_schedule(rows)
    assert [row["node"] for row in rows] == [
        "dense_mlp.gate_up_projection",
        "dense_mlp.gate_up_projection",
        "dense_mlp.clamped_swiglu",
        "dense_mlp.down_projection",
        "dense_mlp.down_projection",
    ]


def test_moe_shared_production_schedule_uses_activation_landmark() -> None:
    rows = [
        _production_row("nvjet_sm103_router"),
        _production_row("routingIndicesBlockScoresKernel"),
        _production_row("sm100_fp8_fp4_gemm_1d1d_impl"),
        _production_row("silu_mul_clamp"),
        _production_row("sm100_fp8_fp4_gemm_1d1d_impl"),
        _production_row("moe::dev::finalize::finalizeKernel"),
        _production_row("CUDAFunctor_add<bfloat16>"),
    ]
    _assign_moe_shared_production_schedule(rows)
    assert [row["node"] for row in rows] == [
        "moe.router",
        None,
        "moe.shared_gate_up",
        "moe.shared_activation",
        "moe.shared_down",
        None,
        "moe.combine",
    ]


def test_dsa_auxiliary_stream_schedule_does_not_depend_on_main_stream_time() -> None:
    rows = [
        {**_production_row("nvjet_sm103_tst_index_q"), "stream": 42, "ts_us": 1.0},
        {**_production_row("fast_hadamard_transform_kernel"), "stream": 42, "ts_us": 2.0},
        {**_production_row("_act_quant_kernel"), "stream": 42, "ts_us": 3.0},
        {**_production_row("cutlass_80_simt_sgemm"), "stream": 42, "ts_us": 4.0},
        {**_production_row("kpool_decode_update_and_maybe_write_cache"), "stream": 43, "ts_us": 2.5},
        {**_production_row("sm100_paged_mqa_logits"), "stream": 43, "ts_us": 3.5},
        {**_production_row("kpool_topk_transform"), "stream": 43, "ts_us": 4.5},
        {**_production_row("fmhaSm100fKernel_QkvE4m3OBfloat16H512"), "stream": 43, "ts_us": 5.5},
    ]
    _assign_dsa_production_stream_schedules(rows)
    assert [row["node"] for row in rows] == [
        "dsa_attention.index_q_projection",
        "dsa_attention.index_k_projection",
        "dsa_attention.key_pool_compression",
        "dsa_attention.index_weight_projection",
        "dsa_attention.index_k_cache",
        "dsa_attention.index_logits",
        "dsa_attention.top_pool_selection",
        "dsa_attention.sparse_mla_core",
    ]


def test_kda_graph_child_schedule_closes_projection_and_output_gemms() -> None:
    names = [
        "mhc_pre_big_fuse_with_norm_tilelang_kernel",
        "kernel_cutlass_kernel_TgvGemmCuteExtKernel",
        "void cutlass::Kernel2<gemm>",
        "nvjet_sm103_forget_projection",
        "_causal_conv1d_update_kernel",
        "fused_sigmoid_gating_delta_rule_update_kernel",
        "layer_norm_gated_fwd_kernel",
        "nvjet_sm103_output_projection",
    ]
    rows = [
        {**_production_row(name), "stream": 7, "ts_us": float(index)}
        for index, name in enumerate(names)
    ]
    _assign_kda_production_stream_schedules(rows)
    assert rows[1]["node"] == "linear_attention.qkv_projection"
    assert rows[2]["node"] == "linear_attention.forget_projection"
    assert rows[3]["node"] == "linear_attention.forget_projection"
    assert rows[7]["node"] == "linear_attention.output_projection"


def test_runtime_support_is_typed_without_becoming_model_ir() -> None:
    rows = [
        _production_row("track_mamba_states_all_layers_kernel"),
        _production_row("kpool_build_ragged_layout"),
    ]
    _classify_runtime_support(rows)
    assert rows[0]["node"] is None
    assert rows[0]["support_class"] == "state_bookkeeping"
    assert rows[1]["support_class"] == "attention_plan_metadata"
    assert all(row["support_reason"] for row in rows)
