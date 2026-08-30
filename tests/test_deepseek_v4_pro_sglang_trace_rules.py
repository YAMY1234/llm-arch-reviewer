from __future__ import annotations

from models.common.trace_mapping import FrameRef
from models.deepseek_v4_pro.build.compile_deepseek_v4_pro_vllm_eager_contract import (
    CSA_LAYERS,
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
        add("attention.tp_output_collective")
        add("mhc_transform.mix", "mhc_fused_post_pre_fma_tilelang_kernel")
        add("mhc_transform.affine")
        add("moe.shared_gate_up")
        add("moe.shared_gate_up")
        add("moe.shared_activation")
        add("moe.shared_gate_up")
        add("moe.shared_gate_up")
        add("moe.routed_gate_up")
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
    assert all(
        row.get("fused_semantic_nodes")
        == ["moe.routed_gate_up", "moe.routed_activation"]
        for row in compiled
        if row.get("selected_node") == "moe.routed_gate_up"
    )
