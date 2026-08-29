from models.common.trace_mapping import FrameRef
from models.glm53_flash.build.glm53_vllm_trace_rules import classify_glm53_vllm_node
from models.glm53_flash.build.glm53_vllm_production_attribution import (
    _annotate_segment_scope,
    _classify_runtime_support,
    _transfer_matching_scope,
)


def frames(*rows: str) -> list[FrameRef]:
    return [FrameRef(raw=row) for row in rows]


def test_mhc_anchor_scope_survives_eager_to_graph_transfer() -> None:
    attention = [{"kernel_name": "attention-kernel"}]
    feed_forward = [{"kernel_name": "ffn-kernel"}]

    _annotate_segment_scope(attention, 24)
    _annotate_segment_scope(feed_forward, 25)

    assert attention[0] | {
        "layer_id": 12,
        "layer_kind": "kda",
        "substage": "attention",
        "segment_id": 24,
        "occurrence_id": "layer_12.attention",
    } == attention[0]
    assert feed_forward[0] | {
        "layer_id": 12,
        "layer_kind": "moe",
        "substage": "feed_forward",
        "segment_id": 25,
        "occurrence_id": "layer_12.feed_forward",
    } == feed_forward[0]


def test_exact_collective_scopes_map_to_normalized_tp_contracts():
    assert classify_glm53_vllm_node(
        "multimem_all_reduce_kernel",
        "record_param_comms",
        frames(
            "vllm/distributed/parallel_state.py(686): _all_reduce_out_place",
            "vllm/models/glm5next/nvidia/kda.py(320): forward",
        ),
    ) == ("linear_attention.tp_kda_output_collective", "high")
    assert classify_glm53_vllm_node(
        "ncclDevKernel_AllGather_RING_LL",
        "vllm::all_gather",
        frames("vllm/model_executor/layers/logits_processor.py(84): _gather_logits"),
    ) == ("top.tp_logits_all_gather", "high")


def test_model_unique_kda_kernel_overrides_neighboring_stack():
    assert classify_glm53_vllm_node(
        "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
        None,
        frames("vllm/model_executor/layers/mhc.py(317): fused_post_pre"),
    ) == ("linear_attention.recurrent_update", "high")
    assert classify_glm53_vllm_node(
        "chunk_gla_fwd_kernel_o",
        None,
        frames("vllm/models/glm5next/nvidia/kda.py(373): _forward"),
    ) == ("linear_attention.query_readout", "high")


def test_moe_down_bmm_is_not_collapsed_into_gate_up():
    moe_stack = frames("nn.Module: Glm5NextMoE_3")
    assert classify_glm53_vllm_node(
        "bmm_E4m3_E4m3E4m3_Fp32_t128x128",
        "vllm::moe_forward_shared",
        moe_stack,
    ) == ("moe.routed_gate_up", "medium")
    assert classify_glm53_vllm_node(
        "bmm_Bfloat16_E4m3E4m3_Fp32_t128x128",
        "vllm::moe_forward_shared",
        moe_stack,
    ) == ("moe.routed_down", "high")


def test_indexer_and_sparse_mla_signatures_map_to_stable_leaves():
    dsa_stack = frames("vllm/models/glm5next/nvidia/attention.py(315): forward")
    assert classify_glm53_vllm_node(
        "void vllm::topKPerRowPrefill<512, false>",
        "_C::top_k_per_row_prefill",
        dsa_stack,
    ) == ("dsa_attention.top_pool_selection", "high")
    assert classify_glm53_vllm_node(
        "fmhaSm100fKernel_QkvE4m3OBfloat16H512PagedKvDenseDynamicTokenSparse",
        "vllm::unified_mla_attention_with_output",
        dsa_stack,
    ) == ("dsa_attention.sparse_mla_core", "high")


def test_repeated_helpers_transfer_only_within_equal_bounded_occurrences() -> None:
    source = [
        {
            "kernel_name": "CatArrayBatchedCopy<int>",
            "selected_node": "dsa_attention.token_expansion",
        },
        {
            "kernel_name": "CatArrayBatchedCopy<int>",
            "selected_node": "dsa_attention.selected_indices",
        },
    ]
    production = [
        {"kernel_name": "CatArrayBatchedCopy<int>", "node": None},
        {"kernel_name": "CatArrayBatchedCopy<int>", "node": None},
    ]
    assigned, state = _transfer_matching_scope(
        source, production, method_prefix="mhc_anchor_bounded_dsa"
    )
    assert state == "exact"
    assert assigned == 2
    assert [row["node"] for row in production] == [
        "dsa_attention.token_expansion",
        "dsa_attention.selected_indices",
    ]


def test_vllm_runtime_support_has_explicit_class_and_reason() -> None:
    rows = [
        {"kernel_name": "_zero_kv_blocks_kernel", "node": None},
        {"kernel_name": "compute_slot_mappings", "node": None},
        {"kernel_name": "gumbel_argmax_kernel", "node": None},
    ]
    _classify_runtime_support(rows)
    assert [row["support_class"] for row in rows] == [
        "allocator_or_cache_management",
        "attention_plan_metadata",
        "sampling_and_output",
    ]
    assert all(row["support_reason"] for row in rows)
