from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.qwen40.build.build_qwen40_eager_prefill_profile import (  # noqa: E402
    _transfer_chunk_timing,
    metrics_for_prefill,
)
from models.qwen40.build.qwen40_decode_attribution import _metric  # noqa: E402


def _source(name: str, node: str) -> dict:
    return {
        "kernel_name": name,
        "kernel_label": node,
        "node": node,
        "layer_kind": "linear",
        "substage": "moe",
        "attribution_method": "python_stack_ir_rule",
        "confidence": "high",
    }


def _timing(name: str, timestamp: float) -> dict:
    return {
        "kernel_name": name,
        "ts_us": timestamp,
        "dur_us": 1.0,
        "stream": 7,
    }


def test_timing_transfer_allows_only_exact_stack_support_deletion() -> None:
    transferred, accounting = _transfer_chunk_timing(
        [_source("a", "moe.router"), _source("stack-only", "moe.router"), _source("b", "moe.router")],
        [_timing("a", 1.0), _timing("b", 2.0)],
        timing_rank=2,
        chunk_index=1,
    )

    assert [event["kernel_name"] for event in transferred] == ["a", "b"]
    assert {event["step_index"] for event in transferred} == {1}
    assert {event["prefill_chunk_index"] for event in transferred} == {1}
    assert accounting == {
        "exact_kernel_count": 2,
        "direct_timing_insert_count": 0,
        "context_timing_insert_count": 0,
        "stack_only_support_kernel_count": 1,
    }


def test_timing_only_kernel_requires_identical_ir_context() -> None:
    transferred, accounting = _transfer_chunk_timing(
        [_source("a", "moe.router"), _source("b", "moe.router")],
        [_timing("a", 1.0), _timing("timing-only", 1.5), _timing("b", 2.0)],
        timing_rank=0,
        chunk_index=3,
    )

    inserted = transferred[1]
    assert inserted["node"] == "moe.router"
    assert inserted["attribution_method"] == "exact_sequence_context_insert"
    assert inserted["prefill_chunk_index"] == 3
    assert accounting["context_timing_insert_count"] == 1


def test_timing_only_kernel_cannot_cross_ir_nodes() -> None:
    with pytest.raises(ValueError, match="crosses IR semantics"):
        _transfer_chunk_timing(
            [_source("a", "moe.router"), _source("b", "moe.routed_experts")],
            [_timing("a", 1.0), _timing("timing-only", 1.5), _timing("b", 2.0)],
            timing_rank=0,
            chunk_index=2,
        )


def test_timing_transfer_partitions_repeated_bmms_by_hc_delimiters() -> None:
    delimiter = "void sglang::hc_combine_kernel<4, 2560>"
    source = [
        _source("generic_bmm", "moe.routed_experts"),
        _source(delimiter, "hyperconnection.combine"),
        _source("generic_bmm", "qsa_attention.qkv_gate_projection"),
        _source(delimiter, "hyperconnection.combine"),
    ]
    timing = [
        _timing(delimiter, 1.0),
        _timing("generic_bmm", 2.0),
        _timing(delimiter, 3.0),
    ]

    transferred, _ = _transfer_chunk_timing(
        source,
        timing,
        timing_rank=0,
        chunk_index=1,
    )

    assert transferred[1]["node"] == "qsa_attention.qkv_gate_projection"


def _profile_event(node: str, stage: str, duration_us: float, timestamp_us: float) -> dict:
    return {
        "kernel_name": f"kernel::{node}::{stage}",
        "kernel_label": node,
        "node": node,
        "layer_kind": "linear",
        "substage": stage,
        "attribution_method": "python_stack_ir_rule",
        "step_index": 1,
        "ts_us": timestamp_us,
        "dur_us": duration_us,
    }


def test_hyperconnection_mix_and_combine_drills_have_distinct_scoped_dataflows() -> None:
    metrics = metrics_for_prefill(
        [
            _profile_event("hyperconnection.branch_norm", "attn_hc_mix", 100.0, 0.0),
            _profile_event("hyperconnection.mix", "attn_hc_mix", 200.0, 100.0),
            _profile_event("hyperconnection.combine", "attn_hc_combine", 300.0, 300.0),
            _profile_event("hyperconnection.branch_norm", "mlp_hc_mix", 400.0, 600.0),
            _profile_event("hyperconnection.mix", "mlp_hc_mix", 500.0, 1000.0),
            _profile_event("hyperconnection.combine", "mlp_hc_combine", 600.0, 1500.0),
        ]
    )

    attention_mix = metrics["linear_layer.attn_hc_mix"]
    assert attention_mix["drill_scope"] == "linear_layer.attn_hc_mix"
    assert attention_mix["drill_view"] == "hyperconnection_mix"
    assert attention_mix["drill_metrics"]["branch_norm"]["ms_per_iter"] == 0.1
    assert attention_mix["drill_metrics"]["mix"]["ms_per_iter"] == 0.2
    assert attention_mix["drill_metrics"]["module_input"]["display_label"] == (
        "attention input\n[B,T,H]"
    )
    assert "combine" not in attention_mix["drill_metrics"]

    attention_combine = metrics["linear_layer.attn_hc_combine"]["drill_metrics"]
    assert attention_combine["combine"]["ms_per_iter"] == 0.3
    assert attention_combine["module_output"]["display_label"] == (
        "attention output\n[B,T,H]"
    )
    assert "mix" not in attention_combine

    moe_mix = metrics["linear_layer.mlp_hc_mix"]["drill_metrics"]
    assert moe_mix["branch_norm"]["ms_per_iter"] == 0.4
    assert moe_mix["mix"]["ms_per_iter"] == 0.5
    assert moe_mix["module_input"]["display_label"] == "MoE input\n[B,T,H]"

    moe_combine = metrics["linear_layer.mlp_hc_combine"]["drill_metrics"]
    assert moe_combine["combine"]["ms_per_iter"] == 0.6
    assert moe_combine["module_output"]["display_label"] == "MoE output\n[B,T,H]"
    assert "low_rank_gate" not in moe_combine

    # A direct (unscoped) view remains populated with the all-call aggregate.
    assert metrics["hyperconnection_mix.branch_norm"]["ms_per_iter"] == 0.5
    assert metrics["hyperconnection_mix.mix"]["ms_per_iter"] == 0.7
    assert metrics["hyperconnection_combine.combine"]["ms_per_iter"] == 0.9


def test_elapsed_decomposes_module_gap_into_other_gpu_work_and_device_idle() -> None:
    target = [
        {
            **_profile_event("moe.router", "moe", 2.0, 0.0),
            "layer_id": 0,
        },
        {
            **_profile_event("moe.router", "moe", 1.0, 5.0),
            "layer_id": 0,
        },
    ]
    other = {
        **_profile_event("moe.topk", "moe", 2.0, 2.0),
        "layer_id": 0,
    }
    metric = _metric(
        target,
        n_iters=1,
        metric_kind="exclusive_leaf",
        aggregation="test",
        all_events=[*target, other],
        elapsed_scope="invocation",
    )

    assert metric["gpu_elapsed_ms"] == 0.006
    assert metric["active_gpu_ms"] == 0.003
    assert metric["module_gap_ms"] == 0.003
    assert metric["other_gpu_work_ms"] == 0.002
    assert metric["device_idle_ms"] == 0.001
    assert metric["module_active_pct"] == 50.0
    assert metric["device_busy_pct"] == pytest.approx(83.33, abs=0.01)


def test_residency_keeps_stream_overlap_separate_from_active_time() -> None:
    events = [
        {**_profile_event("moe.router", "moe", 3.0, 0.0), "layer_id": 0},
        {**_profile_event("moe.router", "moe", 1.0, 1.0), "layer_id": 0},
    ]
    metric = _metric(
        events,
        n_iters=1,
        metric_kind="exclusive_leaf",
        aggregation="test",
        all_events=events,
        elapsed_scope="invocation",
    )

    assert metric["gpu_residency_ms"] == 0.004
    assert metric["active_gpu_ms"] == 0.003
    assert metric["gpu_overlap_ms"] == 0.001
    assert metric["gpu_elapsed_ms"] == 0.003


def test_active_time_averages_over_missing_selected_iterations_as_zero() -> None:
    event = _profile_event("linear_attention.gating", "attention", 2.0, 0.0)
    metric = _metric(
        [event],
        n_iters=2,
        metric_kind="exclusive_leaf",
        aggregation="test",
    )

    assert metric["active_gpu_ms"] == 0.001
    assert metric["gpu_residency_ms"] == 0.001
