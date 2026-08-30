from __future__ import annotations

from models.common.trace_mapping import FrameRef
from models.deepseek_v4_pro.build.compile_deepseek_v4_pro_vllm_eager_contract import (
    CSA_LAYERS,
    _canonicalize_sglang_attention_order,
    _recover_ordered_sglang_compute,
    compile_contract,
)
from models.deepseek_v4_pro.build.deepseek_v4_pro_sglang_trace_rules import (
    classify_deepseek_v4_pro_sglang_node,
)


def frame(raw: str) -> FrameRef:
    return FrameRef(raw=raw)


def test_shape_specific_router_and_stale_async_moe_kernels_are_exact() -> None:
    assert classify_deepseek_v4_pro_sglang_node(
        "nvjet_sm103_tss_64x32_64x16_2x2_2cta_h_bz_splitK_TNT",
        "aten::mm",
        [frame("nn.Module: MoEGate_17")],
    ) == ("moe.score_projection", "high")
    stale = [
        frame("python/sglang/srt/mem_cache/allocator/paged.py(222): alloc_decode")
    ]
    assert classify_deepseek_v4_pro_sglang_node(
        "_router_triton_kernel", None, stale
    ) == ("moe.learned_select", "high")
    assert classify_deepseek_v4_pro_sglang_node(
        "_pack_topk_ids_triton_kernel", None, stale
    ) == ("moe.dispatch", "high")
    assert classify_deepseek_v4_pro_sglang_node(
        "bmm_MxE4m3_MxE2m1MxE4m3_gate_up", None, stale
    ) == ("moe.routed_gate_up", "high")
    assert classify_deepseek_v4_pro_sglang_node(
        "bmm_Bfloat16_MxE2m1MxE4m3_down", None, stale
    ) == ("moe.routed_down", "high")


def test_logits_processor_projection_is_owned_by_lm_head() -> None:
    stack = [
        frame(
            "python/sglang/srt/layers/logits_processor.py(702): "
            "_compute_lm_head"
        )
    ]
    assert classify_deepseek_v4_pro_sglang_node(
        "nvjet_sm103_tst_64x8_64x16_1x1_h_bz_splitK_TNT",
        "aten::mm",
        stack,
    ) == ("top.lm_head", "high")
    assert classify_deepseek_v4_pro_sglang_node(
        "copy_logits_to_output",
        "aten::copy_",
        [
            frame(
                "python/sglang/srt/layers/logits_processor.py(869): "
                "_copy_logits_to_buffer"
            )
        ],
    ) == ("top.logits", "high")


def test_prefill_only_sparse_kernels_have_semantic_owners() -> None:
    assert classify_deepseek_v4_pro_sglang_node(
        "_get_k_and_s_triton_kernel", None, []
    ) == ("csa_indexer.score", "high")
    assert classify_deepseek_v4_pro_sglang_node(
        "void deep_gemm::sm100_mqa_logits<false>", None, []
    ) == ("csa_indexer.score", "high")
    assert classify_deepseek_v4_pro_sglang_node(
        "_combine_topk_swa_indices_kernel", None, []
    ) == ("attention.index_union", "high")
    assert classify_deepseek_v4_pro_sglang_node(
        "_dequantize_k_cache_paged_kernel", None, []
    ) == ("attention.sparse_mqa", "high")
    assert classify_deepseek_v4_pro_sglang_node(
        "void sglang::flash_c4_prefill<128l, float>", None, []
    ) == ("csa_indexer.k_compress", "high")
    assert classify_deepseek_v4_pro_sglang_node(
        "void sglang::flash_c4_prefill<512l, float>", None, []
    ) == ("csa_compressor.softmax_pool", "high")
    assert classify_deepseek_v4_pro_sglang_node(
        "void sglang::flash_c128_prefill<512l, float>", None, []
    ) == ("hca_compressor.softmax_pool", "high")
    assert classify_deepseek_v4_pro_sglang_node(
        "nvjet_sm103_tst_64x64_64x13_1x4_h_bz_TNT", "aten::mm", []
    ) == ("csa_indexer.weight_projection", "high")
    assert classify_deepseek_v4_pro_sglang_node(
        "nvjet_sm103_tss_192x128_64x6_2x2_2cta_h_bz_TNT", "aten::mm", []
    ) == ("moe.score_projection", "high")


def test_prefill_hca_compressor_projection_uses_exact_neighbors() -> None:
    rows = [
        {"selected_node": "attention.window_kv"},
        {
            "selected_node": "top.runtime_support",
            "cpu_op_name": "aten::mm",
        },
        {"selected_node": "hca_compressor.softmax_pool"},
    ]
    _recover_ordered_sglang_compute(
        rows, source_commit="71de97b264b04dcd514cf904003028aefe9775c8"
    )
    assert rows[1]["selected_node"] == "compressor.kv_gate_projection"
    assert rows[1]["mapping_method"] == "ordered_hca_compressor_projection"


def test_sglang_attention_order_overrides_only_exact_stale_async_boundaries() -> None:
    rows = [
        {"selected_node": "attention.q_a", "kernel_name": "qa_quant"},
        {"selected_node": "attention.q_a", "kernel_name": "qa_gemm"},
        {"selected_node": "attention.q_norm", "kernel_name": "q_norm"},
        {"selected_node": "attention.q_a", "kernel_name": "qb_quant"},
        {"selected_node": "attention.q_a", "kernel_name": "qb_gemm"},
        {"selected_node": "attention.q_head_norm", "kernel_name": "q_head_norm"},
        {
            "selected_node": "compressor.kv_gate_projection",
            "kernel_name": "nvjet_sm103_tss_128x256_64x6_2x2_2cta_h_bz_TNT",
        },
        {"selected_node": "csa_indexer.k_compress", "kernel_name": "flash_c4_prefill<128l"},
        {"selected_node": "attention.sparse_mqa", "kernel_name": "sparse_attn"},
        {
            "selected_node": "top.runtime_support",
            "kernel_name": "AUnaryFunctor and MulFunctor postprocess",
        },
        {"selected_node": "attention.inverse_rope", "kernel_name": "inverse_rope"},
    ]

    _canonicalize_sglang_attention_order(rows, kind="csa")

    assert [row["selected_node"] for row in rows[3:5]] == [
        "attention.q_b",
        "attention.q_b",
    ]
    assert rows[6]["selected_node"] == "csa_indexer.k_compress"
    assert rows[9]["selected_node"] == "attention.sparse_mqa"


def _row(event_id: int, node: str, kernel: str | None = None) -> dict:
    return {
        "event_id": f"k_{event_id:06d}",
        "kernel_name": kernel or f"kernel_{event_id}",
        "selected_node": node,
        "confidence": "high",
        "cpu_op_name": None,
        "primitive_frame": None,
        "operator_frame": {
            "raw": "python/sglang/srt/models/deepseek_v4.py(1884): forward",
            "source_exists": True,
        },
        "semantic_frame": {
            "raw": "python/sglang/srt/models/deepseek_v4.py(1884): forward",
            "source_exists": True,
        },
        "model_context_frame": {"raw": "nn.Module: DeepseekV4DecoderLayer_0"},
        "phase_frame": {
            "raw": "python/sglang/srt/model_executor/runner/eager_runner.py(229): _execute_decode",
            "source_exists": True,
        },
        "evidence": ["kernel", "python_stack", "record_shapes"],
    }


def test_occurrence_compiler_accepts_sglang_fused_indexer_state_contract() -> None:
    rows: list[dict] = []
    event_id = 0

    def add(node: str, kernel: str | None = None) -> None:
        nonlocal event_id
        rows.append(_row(event_id, node, kernel))
        event_id += 1

    add("top.tp_embedding_output_collective", "twoshotAllreduceKernel")
    for layer in range(61):
        if layer == 0:
            add("mhc_transform.affine")
        add("mhc_transform.affine")
        if layer in CSA_LAYERS:
            add("csa_indexer.k_compress")
            add("csa_indexer.k_compress")
            add("csa_indexer.k_compress")
        else:
            add("hca_compressor.softmax_pool")
        add("compressor.partial_state")
        if layer == 0:
            add("attention.o_b")
            add("top.runtime_support", "ncclDevKernel_AllReduce_Sum_bf16_RING_LL")
        else:
            add("attention.tp_output_collective")
        add("mhc_transform.mix", "mhc_fused_post_pre_fma_tilelang_kernel")
        add("mhc_transform.affine")
        add("moe.shared_gate_up")
        add("moe.shared_gate_up")
        add("moe.shared_activation")
        add("moe.shared_gate_up")
        add("moe.shared_gate_up")
        add("moe.routed_gate_up")
        if layer == 0:
            add("moe.combine")
            add("top.runtime_support", "ncclDevKernel_AllReduce_Sum_bf16_RING_LL")
        else:
            add("moe.tp_moe_output_collective")
        add("mhc_transform.mix", "mhc_fused_post_pre_fma_tilelang_kernel")
    add("top.lm_head")
    add("top.tp_logits_collective", "ncclDevKernel_AllGather_RING_LL")

    compiled, report = compile_contract(
        rows,
        {
            "source_commit": "71de97b264b04dcd514cf904003028aefe9775c8",
            "rank": 0,
            "phase": "forward_decode",
            "window": {"duration_ms": 1.0},
        },
        {
            "ok": True,
            "kernel_count": len(rows),
            "mapped_kernel_count": len(rows),
            "mapped_duration_ratio": 1.0,
        },
    )

    assert report["ok"] is True
    assert report["occurrence_count"] == 122
    assert report["node_counts"]["csa_compressor.partial_state"] == 30
    assert report["node_counts"]["hca_compressor.partial_state"] == 31
    assert report["node_counts"]["moe.shared_gate_up"] == 122
    assert report["node_counts"]["moe.shared_down"] == 122
    assert sum(
        row.get("mapping_method") == "ordered_attention_output_all_reduce_boundary"
        for row in compiled
    ) == 1
    assert sum(
        row.get("mapping_method") == "ordered_moe_output_all_reduce_boundary"
        for row in compiled
    ) == 1
    assert all(
        row.get("fused_semantic_nodes")
        == ["moe.routed_gate_up", "moe.routed_activation"]
        for row in compiled
        if row.get("selected_node") == "moe.routed_gate_up"
    )
