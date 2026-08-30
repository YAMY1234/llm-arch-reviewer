from __future__ import annotations

from models.common.trace_mapping import FrameRef
from models.deepseek_v4_pro.build.compile_deepseek_v4_pro_vllm_eager_contract import (
    CSA_LAYERS,
    compile_contract,
)
from models.deepseek_v4_pro.build.deepseek_v4_pro_vllm_trace_rules import (
    classify_deepseek_v4_pro_vllm_node,
)


def frame(raw: str) -> FrameRef:
    return FrameRef(raw=raw)


def test_exact_indexer_router_and_prefill_index_signatures_override_stale_frames() -> None:
    q_b_kernel = (
        "void deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl<128u, 8192u, "
        "1536u>"
    )
    assert classify_deepseek_v4_pro_vllm_node(
        q_b_kernel,
        "vllm::fp8_gemm_nt_op",
        [frame("nn.Module: DeepseekV4Indexer_7")],
    ) == ("csa_indexer.q_projection", "high")
    assert classify_deepseek_v4_pro_vllm_node(
        "void router_gemm_kernel_float_output<__nv_bfloat16>",
        "_moe_C::dsv3_router_gemm",
        [frame("nn.Module: DeepseekV4MoE_12")],
    ) == ("moe.score_projection", "high")
    assert classify_deepseek_v4_pro_vllm_node(
        "DequantGatherKCacheKernel",
        None,
        [frame("vllm/models/deepseek_v4/nvidia/flashmla.py(237): _forward_prefill")],
    ) == ("attention.index_union", "high")
    assert classify_deepseek_v4_pro_vllm_node(
        "mhc_fused_tilelang_kernel",
        None,
        [frame("unrelated asynchronous frame")],
    ) == ("mhc_transform.mix", "high")


def _row(event_id: int, node: str, kernel: str | None = None) -> dict:
    return {
        "event_id": f"k_{event_id:06d}",
        "kernel_name": kernel or f"kernel_{event_id}",
        "selected_node": node,
        "confidence": "high",
        "cpu_op_name": None,
        "primitive_frame": None,
        "operator_frame": {
            "raw": "vllm/models/deepseek_v4/nvidia/model.py(866): forward",
            "source_exists": True,
        },
        "semantic_frame": {
            "raw": "vllm/models/deepseek_v4/nvidia/model.py(866): forward",
            "source_exists": True,
        },
        "model_context_frame": {"raw": "nn.Module: DeepseekV4DecoderLayer_0"},
        "phase_frame": {
            "raw": "vllm/v1/worker/gpu_model_runner.py(4069): execute_model",
            "source_exists": True,
        },
        "evidence": ["kernel", "python_stack", "record_shapes"],
    }


def _attention_rows(start: int, layer: int) -> tuple[list[dict], int]:
    kind = "csa" if layer in CSA_LAYERS else "hca"
    rows: list[dict] = []

    def add(node: str) -> None:
        nonlocal start
        rows.append(_row(start, node))
        start += 1

    if kind == "csa":
        add("csa_indexer.score")
        add("compressor.partial_state")
        add("compressor.partial_state")
    else:
        add("hca_compressor.softmax_pool")
        add("compressor.partial_state")
    add("attention.q_a")
    add("attention.tp_output_collective")
    return rows, start


def _synthetic_schedule(*, fused_launch: bool) -> list[dict]:
    rows: list[dict] = []
    event_id = 0

    def add(node: str, kernel: str | None = None) -> None:
        nonlocal event_id
        rows.append(_row(event_id, node, kernel))
        event_id += 1

    for layer in range(61):
        if not fused_launch or layer == 0:
            add("mhc_transform.affine")
        add("mhc_transform.affine")
        attention, event_id = _attention_rows(event_id, layer)
        rows.extend(attention)
        add("mhc_transform.mix", "mhc_fused_tilelang_kernel" if fused_launch else None)
        if not fused_launch:
            add("mhc_transform.affine")
        add("mhc_transform.affine")
        add("moe.tp_moe_output_collective")
        add("mhc_transform.mix", "mhc_fused_tilelang_kernel" if fused_launch and layer < 60 else None)
    return rows


def _compile(rows: list[dict]) -> tuple[list[dict], dict]:
    return compile_contract(
        rows,
        {
            "source_commit": "dd10e03f95f94edbea1975c67ace3a35ec9a8a40",
            "rank": 0,
            "phase": "vllm_decode",
            "window": {"duration_ms": 1.0},
        },
        {
            "ok": True,
            "kernel_count": len(rows),
            "mapped_kernel_count": len(rows),
            "mapped_duration_ratio": 1.0,
        },
    )


def test_occurrence_compiler_closes_separate_mhc_schedule() -> None:
    rows, report = _compile(_synthetic_schedule(fused_launch=False))
    assert report["ok"] is True
    assert report["occurrence_count"] == 122
    assert report["node_counts"]["csa_attention.tp_csa_output_collective"] == 30
    assert report["node_counts"]["hca_attention.tp_hca_output_collective"] == 31
    assert not any(row.get("launch_group_id") for row in rows)


def test_occurrence_compiler_preserves_fused_post_pre_one_to_many_launches() -> None:
    rows, report = _compile(_synthetic_schedule(fused_launch=True))
    assert report["ok"] is True
    grouped = [row for row in rows if row.get("launch_group_id")]
    assert len(grouped) == 242
    assert len({row["launch_group_id"] for row in grouped}) == 121
    assert {row["launch_group_role"] for row in grouped} == {
        "post_pre_first_kernel",
        "post_pre_second_kernel",
    }
