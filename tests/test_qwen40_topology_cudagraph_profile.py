from __future__ import annotations

import json
from argparse import Namespace
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.qwen40.build.build_qwen40_topology_cudagraph_profile import (  # noqa: E402
    aggregate_rank_metrics,
    build_profile,
    direct_kernel_mapping,
    eager_collective_template,
    merged_gpu_steps,
    select_reference_rank,
)


def test_gpu_step_union_covers_all_stream_tracks() -> None:
    events = []
    for tid, offset, duration in ((10, 0.0, 8.0), (11, -1.0, 12.0)):
        for index in range(6):
            events.append(
                {
                    "cat": "gpu_user_annotation",
                    "ph": "X",
                    "pid": 0,
                    "tid": tid,
                    "name": "step[DECODE bs=4]",
                    "ts": 100.0 * index + offset,
                    "dur": duration,
                }
            )
    merged = merged_gpu_steps(events)
    assert len(merged) == 6
    assert merged[0]["ts"] == -1.0
    assert merged[0]["dur"] == 12.0
    assert merged[0]["merged_gpu_track_count"] == 2


def test_eager_collective_template_preserves_nodes_and_fills_logits(tmp_path: Path) -> None:
    mapping = tmp_path / "mapping.jsonl"
    mapping.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "kernel_name": "ncclDevKernel_AllReduce",
                        "selected_node": "top.dp_logits_input_gather",
                    }
                ),
                json.dumps(
                    {
                        "kernel_name": "_all_gather_kernel_inner",
                        "selected_node": None,
                    }
                ),
            )
        )
        + "\n"
    )
    assert eager_collective_template(mapping) == [
        ("reduce", "top.dp_logits_input_gather"),
        ("gather", "top.tp_logits_collective"),
    ]


def test_flashinfer_wide_gdn_kernel_maps_to_delta_rule() -> None:
    node, label = direct_kernel_mapping("kernel_cutlass_gdn_wide_vec_kernel_t1")
    assert node == "linear_attention.delta_rule"
    assert label == "FlashInfer GDN recurrence"


def test_qwen4_ple_hash_fusion_keeps_stable_ir_ownership() -> None:
    assert direct_kernel_mapping("_qwen4_ngram_hash_kernel")[0] == "ple.ngram_hash"
    assert direct_kernel_mapping("_qwen4_gate_value_kernel")[0] == "ple.grouped_norm_gate"
    assert direct_kernel_mapping("_qwen4_short_conv_state_kernel")[0] == "ple.short_conv"


def test_paired_profiles_can_force_the_same_reference_rank() -> None:
    rank_steps = {0: [1.0, 1.1], 1: [1.3, 1.4], 2: [1.1, 1.2], 3: [1.2, 1.3]}
    assert select_reference_rank(rank_steps, None) == 1
    assert select_reference_rank(rank_steps, 2) == 2


def test_rank_metrics_take_maximum_not_parallel_sum() -> None:
    rank_metrics = {
        0: {"moe.deepep_dispatch": {"ms_per_iter": 1.5, "kernels": []}},
        1: {"moe.deepep_dispatch": {"ms_per_iter": 2.0, "kernels": []}},
    }
    metric = aggregate_rank_metrics(rank_metrics)["moe.deepep_dispatch"]
    assert metric["ms_per_iter"] == 2.0
    assert metric["source_rank"] == 1
    assert metric["rank_range_ms"] == [1.5, 2.0]


def test_prefill_first_metadata_requires_dp4_bs256_patch_sha() -> None:
    args = Namespace(
        config_name="dp_attention",
        chunked_prefill_size=32768,
        admission_control="prefill-first-until-local-target",
        batch_size=64,
        source_patch_sha256="a" * 64,
    )
    try:
        build_profile(args)
    except ValueError as error:
        assert "restricted to DP4 global BS256" in str(error)
    else:
        raise AssertionError("invalid admission instrumentation was accepted")
