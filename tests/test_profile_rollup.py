import pytest

from models.common.profile_rollup import (
    add_execution_boundary_ancestors,
    direct_metrics_from_events,
    expand_rollup_targets,
    rollup_metrics_from_events,
    scoped_rollup_metric_from_events,
    unique_drill_ancestors,
)
from llm_arch_v2.profile_acceptance import validate_executable_drill_rollups
from scripts.materialize_profile_rollups import (
    drop_stale_structural_compute_states,
    drop_derived_parent_targets,
    merge_rollup_metrics,
)


def test_rollup_refresh_preserves_existing_evidence_details() -> None:
    metrics = {
        "stack.module": {
            "active_gpu_ms": 1.0,
            "kernels": [{"name": "existing-kernel"}],
            "attribution": {"methods": {"python_stack": {"kernel_count": 1}}},
        }
    }

    merge_rollup_metrics(
        metrics,
        {
            "stack.module": {
                "active_gpu_ms": 2.0,
                "metric_kind": "inclusive_rollup",
            }
        },
    )

    assert metrics["stack.module"]["active_gpu_ms"] == 2.0
    assert metrics["stack.module"]["metric_kind"] == "inclusive_rollup"
    assert metrics["stack.module"]["kernels"] == [
        {"name": "existing-kernel"}
    ]
    assert metrics["stack.module"]["attribution"]["methods"][
        "python_stack"
    ]["kernel_count"] == 1


def test_materializer_drops_only_stale_structural_compute_overrides() -> None:
    states = {
        "top.module": {"status": "structural"},
        "top.boundary": {"status": "structural"},
        "top.optional": {"status": "not_selected"},
    }
    nodes = {
        "top.module": {
            "drill": "detail",
            "semantic_details": {
                "runtime_mapping": {"expectation": "measured"}
            },
        },
        "top.boundary": {
            "drill": "boundary_detail",
            "semantic_details": {
                "runtime_mapping": {"expectation": "structural"}
            },
        },
        "top.optional": {
            "drill": "detail",
            "semantic_details": {
                "runtime_mapping": {"expectation": "measured"}
            },
        },
    }

    drop_stale_structural_compute_states(states, model_nodes=nodes)

    assert "top.module" not in states
    assert states["top.boundary"]["status"] == "structural"
    assert states["top.optional"]["status"] == "not_selected"


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


def test_rematerialization_discards_stale_drill_and_scoped_parents() -> None:
    assert drop_derived_parent_targets(
        [
            "detail.owner",
            "detail.fused_member",
            "stack.old_parent",
            "stack.scoped_parent",
        ],
        rollup_targets={"stack.old_parent"},
        scoped_targets={"stack.scoped_parent"},
    ) == ["detail.owner", "detail.fused_member"]


def test_scoped_parent_unions_only_matching_owner_occurrences() -> None:
    steps = [
        {
            "events": [
                {
                    "start_us": 0.0,
                    "duration_us": 10.0,
                    "ir_node": "detail.owner",
                    "ir_targets": ["detail.owner"],
                    "substage": "attention",
                    "occurrence_id": "layer_00.attention",
                },
                {
                    "start_us": 5.0,
                    "duration_us": 10.0,
                    "ir_node": "detail.owner",
                    "ir_targets": ["detail.owner"],
                    "substage": "attention",
                    "occurrence_id": "layer_01.attention",
                },
                {
                    "start_us": 20.0,
                    "duration_us": 100.0,
                    "ir_node": "detail.owner",
                    "ir_targets": ["detail.owner"],
                    "substage": "feed_forward",
                    "occurrence_id": "layer_00.feed_forward",
                },
                {
                    "start_us": 130.0,
                    "duration_us": 50.0,
                    "ir_node": "detail.other",
                    "ir_targets": ["detail.other"],
                    "substage": "attention",
                    "occurrence_id": "layer_02.attention",
                },
            ]
        }
    ]

    metric = scoped_rollup_metric_from_events(
        steps,
        target="stack.attention_parent",
        source_nodes={"detail.owner"},
        event_filter={"substage": "attention"},
    )

    assert metric is not None
    assert metric["active_gpu_ms"] == 0.015
    assert metric["gpu_residency_ms"] == 0.02
    assert metric["rollup_sources"] == ["detail.owner"]


def test_executable_drill_with_measured_descendants_requires_union_rollup() -> None:
    model = {
        "semantic_contract": {
            "operations": {
                "model.stack": {"kind": "module"},
                "model.control": {"kind": "control_boundary"},
                "model.optional": {"kind": "module"},
                "detail.compute": {"kind": "projection"},
                "control.compute": {"kind": "projection"},
            }
        },
        "views": {
            "top": {
                "nodes": [
                    {
                        "id": "stack",
                        "drill": "detail",
                        "semantic_op": "model.stack",
                    },
                    {
                        "id": "control",
                        "drill": "control_detail",
                        "semantic_op": "model.control",
                    },
                    {
                        "id": "optional",
                        "drill": "optional_detail",
                        "semantic_op": "model.optional",
                    },
                ]
            },
            "detail": {
                "nodes": [
                    {"id": "compute", "semantic_op": "detail.compute"}
                ]
            },
            "control_detail": {
                "nodes": [
                    {"id": "compute", "semantic_op": "control.compute"}
                ]
            },
            "optional_detail": {"nodes": [{"id": "compute"}]},
        },
    }
    profile = {
        "profile_id": "shared_acceptance_fixture",
        "node_metrics": {
            "detail.compute": {
                "metric_kind": "exclusive_leaf",
                "attribution_status": "measured_direct",
                "active_gpu_ms": 0.01,
                "gpu_residency_ms": 0.01,
            },
            "control_detail.compute": {
                "metric_kind": "exclusive_leaf",
                "attribution_status": "measured_direct",
                "active_gpu_ms": 0.002,
                "gpu_residency_ms": 0.002,
            },
        },
        "node_states": {
            "top.stack": {"status": "structural"},
            "top.control": {"status": "structural"},
            "top.optional": {"status": "not_selected"},
        },
    }

    with pytest.raises(
        ValueError,
        match="top.stack.*presented as structural boundary",
    ):
        validate_executable_drill_rollups(model, profile)

    profile["node_states"].pop("top.stack")
    profile["node_metrics"]["top.stack"] = {
        "metric_kind": "inclusive_rollup",
        "attribution_status": "inclusive_rollup",
        "active_gpu_ms": 0.01,
        "gpu_residency_ms": 0.009994,
        "mapped_event_count": 1,
    }
    validate_executable_drill_rollups(model, profile)

    profile["node_metrics"]["top.stack"]["gpu_residency_ms"] = 0.009
    with pytest.raises(ValueError, match="residency .* is below active union"):
        validate_executable_drill_rollups(model, profile)


def test_selected_nonleaf_compute_module_requires_timing_or_fusion_owner() -> None:
    model = {
        "semantic_contract": {
            "operations": {
                "model.module": {"kind": "module"},
                "model.boundary": {"kind": "boundary"},
                "detail.compute": {"kind": "projection"},
            }
        },
        "views": {
            "top": {
                "nodes": [
                    {
                        "id": "module",
                        "drill": "detail",
                        "semantic_op": "model.module",
                    },
                    {
                        "id": "boundary",
                        "drill": "boundary_detail",
                        "semantic_op": "model.boundary",
                    },
                ]
            },
            "detail": {
                "nodes": [
                    {"id": "compute", "semantic_op": "detail.compute"}
                ]
            },
            "boundary_detail": {"nodes": [{"id": "state"}]},
        },
    }
    profile = {
        "profile_id": "nonleaf_presentation_fixture",
        "node_metrics": {},
        "node_states": {
            "top.module": {"status": "structural"},
            "top.boundary": {"status": "structural"},
        },
    }

    with pytest.raises(
        ValueError,
        match="top.module.*presented as structural boundary",
    ):
        validate_executable_drill_rollups(model, profile)

    profile["node_states"]["top.module"] = {"status": "fused"}
    with pytest.raises(
        ValueError,
        match="top.module.*has no included_in timing owner",
    ):
        validate_executable_drill_rollups(model, profile)

    profile["node_states"]["top.module"] = {
        "status": "fused",
        "included_in": "detail.compute",
    }
    validate_executable_drill_rollups(model, profile)

    profile["node_states"]["top.module"] = {"status": "not_selected"}
    validate_executable_drill_rollups(model, profile)

    profile["node_states"].pop("top.module")
    profile["node_metrics"]["top.module"] = {
        "metric_kind": "inclusive_rollup",
        "attribution_status": "inclusive_rollup",
        "active_gpu_ms": 0.01,
        "gpu_residency_ms": 0.01,
    }
    validate_executable_drill_rollups(model, profile)


def test_caller_isolated_nonleaf_compute_is_fused_not_structural() -> None:
    model = {
        "semantic_contract": {
            "operations": {
                "model.parent": {"kind": "module"},
                "detail.child": {"kind": "module"},
                "detail.boundary": {"kind": "state_boundary"},
            }
        },
        "views": {
            "top": {
                "nodes": [
                    {
                        "id": "parent",
                        "drill": "detail",
                        "semantic_op": "model.parent",
                        "semantic_details": {"runtime_scope": "caller_isolated"},
                    }
                ]
            },
            "detail": {
                "nodes": [
                    {
                        "id": "child",
                        "drill": "child_detail",
                        "semantic_op": "detail.child",
                    },
                    {
                        "id": "boundary",
                        "drill": "boundary_detail",
                        "semantic_op": "detail.boundary",
                    },
                ]
            },
            "child_detail": {"nodes": [{"id": "compute"}]},
            "boundary_detail": {"nodes": [{"id": "state"}]},
        },
    }
    profile = {
        "profile_id": "caller_isolated_fixture",
        "node_metrics": {
            "top.parent": {
                "metric_kind": "inclusive_rollup",
                "attribution_status": "inclusive_rollup",
                "active_gpu_ms": 0.01,
                "gpu_residency_ms": 0.01,
            }
        },
        "node_states": {},
    }

    validate_executable_drill_rollups(model, profile)
