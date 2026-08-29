from models.common.profile_rollup import (
    add_execution_boundary_ancestors,
    direct_metrics_from_events,
    expand_rollup_targets,
    rollup_metrics_from_events,
    unique_drill_ancestors,
)


def test_execution_module_boundary_rolls_into_outer_scope_only() -> None:
    ancestors = {
        "attention.projection": ["stack.attention", "top.stack"],
        "stack.attention": ["top.stack"],
    }
    plan = {
        "transforms": [
            {
                "op": "insert_after",
                "after": "attention.projection",
                "node": {
                    "id": "tp_output_collective",
                    "boundary_role": "module_boundary",
                },
            }
        ]
    }

    expanded = add_execution_boundary_ancestors(ancestors, plan)

    assert expanded["attention.tp_output_collective"] == ["top.stack"]


def test_rollup_uses_interval_union_and_profile_selected_drill_parent() -> None:
    model = {
        "views": {
            "top": {
                "nodes": [
                    {"id": "stack", "drill": "stack"},
                    {"id": "optional", "drill": "detail"},
                ]
            },
            "stack": {"nodes": [{"id": "core", "drill": "detail"}]},
            "detail": {"nodes": [{"id": "a"}, {"id": "b"}]},
        }
    }
    ancestors = unique_drill_ancestors(
        model,
        node_states={"top.optional": {"status": "not_selected"}},
    )
    assert ancestors["detail.a"] == ["stack.core", "top.stack"]

    events = []
    for node, start, duration in (
        ("detail.a", 0.0, 10.0),
        ("detail.b", 5.0, 10.0),
    ):
        targets = expand_rollup_targets(node, ancestors=ancestors)
        events.append(
            {
                "start_us": start,
                "duration_us": duration,
                "ir_node": node,
                "ir_targets": targets,
            }
        )
    metrics = rollup_metrics_from_events(
        [{"events": events}], rollup_targets={"stack.core", "top.stack"}
    )
    assert metrics["stack.core"]["active_gpu_ms"] == 0.015
    assert metrics["stack.core"]["gpu_residency_ms"] == 0.02
    assert metrics["stack.core"]["attribution_status"] == "inclusive_rollup"


def test_direct_metric_ignores_shared_targets_from_another_owner() -> None:
    steps = [
        {
            "events": [
                {
                    "start_us": 0.0,
                    "duration_us": 6.0,
                    "ir_node": "top.expand",
                    "ir_targets": ["top.expand"],
                },
                {
                    "start_us": 10.0,
                    "duration_us": 100.0,
                    "ir_node": "detail.fused_owner",
                    "ir_targets": ["detail.fused_owner", "top.expand"],
                },
            ]
        }
    ]

    metrics = direct_metrics_from_events(
        steps,
        direct_targets={"top.expand", "detail.fused_owner"},
    )

    assert metrics["top.expand"]["active_gpu_ms"] == 0.006
    assert metrics["top.expand"]["gpu_residency_ms"] == 0.006
    assert metrics["detail.fused_owner"]["active_gpu_ms"] == 0.1
