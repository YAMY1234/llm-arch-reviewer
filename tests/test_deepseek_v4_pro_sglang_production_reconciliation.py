from __future__ import annotations

import pytest

from models.deepseek_v4_pro.build.build_deepseek_v4_pro_sglang_profiles import (
    fusion_specs,
)
from models.deepseek_v4_pro.build.reconcile_deepseek_v4_pro_sglang_production import (
    _close_decode_graph_prefix,
    select_production_window,
)


def _range(category: str, name: str, timestamp: float, duration: float, **args) -> dict:
    return {
        "cat": category,
        "name": name,
        "ph": "X",
        "ts": timestamp,
        "dur": duration,
        "args": args,
    }


def test_prefill_selects_only_exact_scheduler_annotation() -> None:
    trace = {
        "traceEvents": [
            _range("kernel", "outside_before", 90, 5, correlation=1),
            _range("gpu_user_annotation", "step[EXTEND bs=1 toks=8192]", 100, 50),
            _range("kernel", "inside_a", 105, 5, correlation=2),
            _range("kernel", "inside_b", 120, 10, correlation=3),
            _range("kernel", "crosses_boundary", 145, 10, correlation=4),
        ]
    }

    selected, metadata = select_production_window(trace, phase="prefill", batch_size=1)

    assert [row["kernel_name"] for row in selected] == ["inside_a", "inside_b"]
    assert metadata["annotation_name"] == "step[EXTEND bs=1 toks=8192]"
    assert metadata["cuda_graph_correlation"] is None
    assert metadata["graph_body_kernel_count"] == 0
    assert metadata["runtime_tail_kernel_count"] == 2


def test_gpu_only_prefill_selects_unique_same_rank_eager_sequence() -> None:
    trace = {
        "traceEvents": [
            _range("kernel", "outside_before", 90, 5, correlation=1),
            _range("kernel", "eager_a", 100, 5, correlation=2),
            _range("kernel", "eager_b", 110, 5, correlation=3),
            _range("kernel", "outside_after", 120, 5, correlation=4),
        ]
    }

    selected, metadata = select_production_window(
        trace,
        phase="prefill",
        batch_size=1,
        eager_kernel_names=["eager_a", "eager_b"],
    )

    assert [row["kernel_name"] for row in selected] == ["eager_a", "eager_b"]
    assert metadata["method"] == "exact_same_rank_eager_ordered_kernel_window"


def test_decode_requires_one_graph_launch_in_exact_scheduler_annotation() -> None:
    trace = {
        "traceEvents": [
            _range("gpu_user_annotation", "step[DECODE bs=16]", 100, 50),
            _range("cuda_runtime", "cudaGraphLaunch", 102, 1, correlation=77),
            _range("kernel", "graph_body", 105, 5, correlation=77),
            _range("kernel", "runtime_tail", 130, 5, correlation=88),
        ]
    }

    selected, metadata = select_production_window(trace, phase="decode", batch_size=16)

    assert [row["kernel_name"] for row in selected] == ["graph_body", "runtime_tail"]
    assert metadata["cuda_graph_correlation"] == 77
    assert metadata["graph_body_kernel_count"] == 1
    assert metadata["runtime_tail_kernel_count"] == 1


def test_gpu_only_decode_selects_exact_cuda_graph_correlation() -> None:
    trace = {
        "traceEvents": [
            _range("kernel", "runtime_prefix", 90, 5, correlation=11),
            _range("cuda_runtime", "cudaGraphLaunch", 100, 2, correlation=77),
            _range("kernel", "graph_a", 105, 5, correlation=77),
            _range("kernel", "graph_b", 120, 5, correlation=77),
            _range("kernel", "runtime_suffix", 140, 5, correlation=88),
        ]
    }

    selected, metadata = select_production_window(trace, phase="decode", batch_size=1)

    assert [row["kernel_name"] for row in selected] == ["graph_a", "graph_b"]
    assert metadata["method"] == "exact_cuda_graph_correlation"
    assert metadata["graph_body_kernel_count"] == 2
    assert metadata["runtime_tail_kernel_count"] == 0


@pytest.mark.parametrize("launch_count", [0, 2])
def test_decode_rejects_wrong_graph_launch_count(launch_count: int) -> None:
    events = [_range("gpu_user_annotation", "step[DECODE bs=64]", 100, 50)]
    events.extend(
        _range("cuda_runtime", "cudaGraphLaunch", 102 + index, 1, correlation=77 + index)
        for index in range(launch_count)
    )
    events.append(_range("kernel", "inside", 110, 5, correlation=77))

    with pytest.raises(ValueError, match="requires 1 cudaGraphLaunch"):
        select_production_window({"traceEvents": events}, phase="decode", batch_size=64)


def test_sglang_fusion_groups_have_one_owner_and_disjoint_semantic_members() -> None:
    groups = fusion_specs()
    members = [member for group in groups.values() for member in group["ir_nodes"]]

    assert len(members) == len(set(members))
    assert all(group["owner"] in group["ir_nodes"] for group in groups.values())
    assert all(group["source_nodes"] for group in groups.values())


def test_decode_graph_prefix_closes_only_exact_bounded_dependencies() -> None:
    source = [
        {
            "event_id": "e0",
            "kernel_name": "copy_before_embedding<int>",
            "selected_node": "top.runtime_support",
        }
    ]
    production = [
        {"kernel_name": "fill_a", "node": "top.runtime_support", "eager_event_ids": ["e1"]},
        {"kernel_name": "memcpy32_post", "node": None},
        {"kernel_name": "memcpy32_post", "node": None},
        {"kernel_name": "fill_b", "node": "top.runtime_support", "eager_event_ids": ["e2"]},
        {"kernel_name": "memcpy32_post", "node": None},
        {"kernel_name": "plan", "node": "top.runtime_support", "eager_event_ids": ["e3"]},
        {"kernel_name": "copy_before_embedding<int>", "node": None},
        {"kernel_name": "embedding", "node": "top.embedding", "eager_event_ids": ["e4"]},
    ]

    assigned = _close_decode_graph_prefix(
        source,
        production,
        prefix=(0, len(production)),
        expected_dependency_names=("memcpy32_post",) * 3,
    )

    assert assigned == 3
    assert all(row["node"] for row in production)
    assert sum(row.get("support_class") == "graph_dependency" for row in production) == 3
    assert production[-2]["eager_event_ids"] == ["e0"]
