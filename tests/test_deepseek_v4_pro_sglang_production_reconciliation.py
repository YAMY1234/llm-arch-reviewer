from __future__ import annotations

import pytest

from models.deepseek_v4_pro.build.build_deepseek_v4_pro_sglang_profiles import (
    fusion_specs,
)
from models.deepseek_v4_pro.build.build_deepseek_v4_pro_vllm_profiles import (
    prepare_events,
)
from models.deepseek_v4_pro.build.audit_deepseek_v4_pro_sglang_production_matrix import (
    audit_collective_rank_durations,
    audit_prefill_prime_coordinate,
)
from models.deepseek_v4_pro.build.reconcile_deepseek_v4_pro_sglang_production import (
    _close_decode_graph_prefix,
    select_production_window,
)
from models.deepseek_v4_pro.build.validate_deepseek_v4_pro_sglang_formal_window import (
    validate_formal_window,
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


def test_gpu_only_prefill_selects_formal_sequence_after_decode_activation_prime() -> None:
    trace = {
        "traceEvents": [
            _range("cuda_runtime", "cudaGraphLaunch", 100, 2, correlation=70),
            _range("kernel", "decode_prime", 105, 5, correlation=70),
            _range("kernel", "prefill_a", 200, 5, correlation=80),
            _range("kernel", "prefill_b", 210, 5, correlation=81),
        ]
    }
    selected, metadata = select_production_window(
        trace,
        phase="prefill",
        batch_size=1,
        eager_kernel_names=["prefill_a", "prefill_b"],
        graph_launch_index=1,
    )
    assert [row["kernel_name"] for row in selected] == ["prefill_a", "prefill_b"]
    assert metadata["profile_priming_launch_count"] == 1
    assert metadata["graph_launch_index"] is None


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


def test_gpu_only_decode_selects_formal_launch_after_profiler_prime() -> None:
    trace = {
        "traceEvents": [
            _range("cuda_runtime", "cudaGraphLaunch", 100, 2, correlation=70),
            _range("kernel", "priming_graph", 105, 5, correlation=70),
            _range("cuda_runtime", "cudaGraphLaunch", 200, 2, correlation=77),
            _range("kernel", "formal_graph", 205, 5, correlation=77),
        ]
    }

    selected, metadata = select_production_window(
        trace, phase="decode", batch_size=16, graph_launch_index=1
    )

    assert [row["kernel_name"] for row in selected] == ["formal_graph"]
    assert metadata["cuda_graph_correlation"] == 77
    assert metadata["graph_launch_index"] == 1
    assert metadata["profile_priming_launch_count"] == 1


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
    assert all(group["source_nodes"] == {group["owner"]} for group in groups.values())


def test_fusion_group_rejects_distinct_physical_source_sets() -> None:
    invalid = {
        "invalid": {
            "owner": "moe.score_projection",
            "ir_nodes": ["moe.score_projection", "moe.hash_select"],
            "source_nodes": {"moe.score_projection", "moe.hash_select"},
            "proof": "invalid aggregate",
        }
    }
    with pytest.raises(ValueError, match="must prove one physical owner event set"):
        prepare_events([], invalid)


def test_collective_rank_duration_gate_rejects_activation_skew() -> None:
    rows = {
        rank: [
            {
                "node": "top.tp_embedding_output_collective",
                "occurrence_id": None,
                "kernel_name": "two-shot-allreduce",
                "dur_us": 33_000.0 if rank < 7 else 12.0,
            }
        ]
        for rank in range(8)
    }
    audit, errors = audit_collective_rank_durations(rows)
    assert audit["outlier_count"] == 1
    assert errors and "activation skew" in errors[0]


def test_collective_rank_duration_gate_accepts_aligned_ranks() -> None:
    rows = {
        rank: [
            {
                "node": "top.tp_embedding_output_collective",
                "occurrence_id": None,
                "kernel_name": "two-shot-allreduce",
                "dur_us": 23.0 + rank,
            }
        ]
        for rank in range(8)
    }
    audit, errors = audit_collective_rank_durations(rows)
    assert audit["outlier_count"] == 0
    assert errors == []


def test_prefill_prime_coordinate_requires_last_warmup_decode_then_formal() -> None:
    client = {
        "contract": {"warmup_request_count": 3, "formal_request_count": 1},
        "profile_coordinate": {
            "mode": "last_warmup_decode_prime_then_formal_prefill",
            "resolved_absolute_start_step": 3086,
            "resolved_absolute_target_step": 3087,
            "formal_start_forward_ct": 3087,
        },
        "profile_controls": [
            {"request": {"start_step": 3086, "num_steps": 2}}
        ],
    }
    assert audit_prefill_prime_coordinate(client) == []
    client["profile_coordinate"]["formal_start_forward_ct"] = 3086
    assert audit_prefill_prime_coordinate(client) == [
        "prime_immediately_precedes_formal"
    ]


def _formal_window_inputs(throughput: float = 1400.0) -> tuple[dict, str, dict]:
    client = {
        "state": "passed",
        "contract": {
            "isl": 8192,
            "osl": 1024,
            "random_range_ratio": 1.0,
            "concurrency": 16,
            "warmup_request_count": 48,
            "formal_request_count": 16,
            "no_intentionally_shared_prefix": True,
            "dspark_enabled": False,
        },
        "profile_coordinate": {
            "profile_prime_steps": 1,
            "resolved_absolute_start_step": 3657,
            "resolved_absolute_target_step": 3658,
        },
        "profile_controls": [
            {
                "http_status": 200,
                "request": {"start_step": 3657, "num_steps": 2},
            }
        ],
    }
    scheduler_log = (
        "Decode batch [3656], #running-req: 16, cuda graph: True, "
        "gen throughput (token/s): 83.90\n"
        f"Decode batch [3657], #running-req: 16, cuda graph: True, "
        f"gen throughput (token/s): {throughput}\n"
    )
    baseline = {
        "concurrencies": {"16": {"selected_decode": {"throughput_token_s": 1361.97}}}
    }
    return client, scheduler_log, baseline


def test_formal_window_validator_selects_stable_second_launch() -> None:
    client, scheduler_log, baseline = _formal_window_inputs()
    gate = validate_formal_window(
        client=client,
        scheduler_log=scheduler_log,
        baseline=baseline,
        concurrency=16,
    )
    assert gate["activation_affected_scheduler_step"] == 3656
    assert gate["formal_target_step"] == 3657
    assert gate["formal_target"]["throughput_token_s"] == 1400.0


def test_formal_window_validator_rejects_throughput_collapse() -> None:
    client, scheduler_log, baseline = _formal_window_inputs(83.0)
    with pytest.raises(ValueError, match="profile-start collapse"):
        validate_formal_window(
            client=client,
            scheduler_log=scheduler_log,
            baseline=baseline,
            concurrency=16,
        )


def test_formal_window_validator_rejects_single_launch_coordinate() -> None:
    client, scheduler_log, baseline = _formal_window_inputs()
    client["profile_coordinate"]["profile_prime_steps"] = 0
    with pytest.raises(ValueError, match="activation-prime plus formal"):
        validate_formal_window(
            client=client,
            scheduler_log=scheduler_log,
            baseline=baseline,
            concurrency=16,
        )


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
