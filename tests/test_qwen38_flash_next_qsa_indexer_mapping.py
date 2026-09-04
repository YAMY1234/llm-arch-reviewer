from __future__ import annotations

from models.common.timeline_artifact import timeline_targets
from models.qwen38_flash_next.build.qwen38_flash_next_decode_attribution import (
    attach_qsa_indexer_drill_metrics,
    attach_qsa_indexer_drill_targets,
    reconcile_qsa_indexer_projection_ownership,
)


def _event(name: str, ts_us: float, label: str = "support") -> dict:
    return {
        "node": "qsa_attention.indexer",
        "kernel_name": name,
        "kernel_label": label,
        "ts_us": ts_us,
        "dur_us": 1.0,
        "step_index": 1,
        "layer_id": 3,
        "layer_kind": "full",
        "substage": "attention",
        "attribution_method": "validated_execution_sequence",
    }


def test_qsa_indexer_parent_is_refined_without_replacing_rollup() -> None:
    events = [
        _event("position_support", 0.0),
        _event("index_qk_projection_gemm", 1.0),
        _event("qsa_index_q_prep_kernel", 2.0, "QSA index query preparation"),
        _event("qsa_index_k_compress_kernel", 3.0, "QSA index key compression"),
        _event("score_fill", 4.0),
        _event("qsa_mqa_decode", 5.0, "compressed MQA score"),
        _event("fast_topk_detail::fast_topk_kernel", 6.0, "QSA index top-k"),
        _event("_expand_qsa_block_indices_kernel", 7.0, "QSA block-index expansion"),
    ]

    attach_qsa_indexer_drill_targets(events)

    assert [event["node"] for event in events] == ["qsa_attention.indexer"] * 8
    assert [event["qsa_indexer_drill_target"] for event in events] == [
        "qsa_indexer.qk_projection",
        "qsa_indexer.qk_projection",
        "qsa_indexer.q_norm_rope",
        "qsa_indexer.compress",
        "qsa_indexer.compressed_score",
        "qsa_indexer.compressed_score",
        "qsa_indexer.block_topk",
        "qsa_indexer.expand_tail",
    ]
    assert timeline_targets(events[2])[:2] == [
        "qsa_attention.indexer",
        "qsa_indexer.q_norm_rope",
    ]

    metrics = {"qsa_attention.indexer": {"active_gpu_ms": 0.008}}
    attach_qsa_indexer_drill_metrics(
        metrics, events, n_iters=1, all_events=events
    )
    parent = metrics["qsa_attention.indexer"]
    assert parent["drill_mapping_coverage_pct"] == 100.0
    assert parent["drill_metrics"]["qk_projection"]["active_gpu_ms"] == 0.002
    assert parent["drill_metrics"]["raw_k_cache"] == {
        "status": "fused",
        "label": "raw index-K and MRoPE-position stores are fused into Q-prep",
        "included_in": "qsa_indexer.q_norm_rope",
        "scope_target": "qsa_attention.indexer",
    }


def test_qsa_indexer_without_qprep_keeps_ambiguous_support_on_parent() -> None:
    events = [
        _event("generic_projection", 0.0),
        _event("fast_topk_detail::fast_topk_kernel", 1.0, "QSA index top-k"),
    ]

    attach_qsa_indexer_drill_targets(events)

    assert "qsa_indexer_drill_target" not in events[0]
    assert events[1]["qsa_indexer_drill_target"] == "qsa_indexer.block_topk"


def test_qsa_indexer_projection_owner_is_reconciled_from_same_stream_qprep() -> None:
    events = [
        {
            **_event("main_qkv_gemm", 0.0),
            "node": "qsa_attention.qkv_gate_projection",
            "stream": "main",
        },
        {
            **_event("index_copy", 0.5),
            "node": "qsa_attention.qkv_gate_projection",
            "stream": "indexer",
        },
        {
            **_event("index_projection_gemm", 1.0),
            "node": "qsa_attention.qkv_gate_projection",
            "stream": "indexer",
        },
        {
            **_event("qsa_index_q_prep_kernel", 2.0, "QSA index query preparation"),
            "stream": "indexer",
        },
    ]

    reconcile_qsa_indexer_projection_ownership(events)
    attach_qsa_indexer_drill_targets(events)

    assert events[0]["node"] == "qsa_attention.qkv_gate_projection"
    assert [event["node"] for event in events[1:]] == [
        "qsa_attention.indexer",
        "qsa_attention.indexer",
        "qsa_attention.indexer",
    ]
    assert [event["qsa_indexer_drill_target"] for event in events[1:]] == [
        "qsa_indexer.qk_projection",
        "qsa_indexer.qk_projection",
        "qsa_indexer.q_norm_rope",
    ]
    assert events[1]["attribution_method"] == (
        "validated_stream_anchor_reconciliation"
    )
