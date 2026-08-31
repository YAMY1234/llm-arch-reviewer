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
    materialize_typed_fusion_states,
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
        "detail.leaf": {
            "semantic_details": {
                "runtime_mapping": {"expectation": "measured"}
            },
        },
    }
    states["detail.leaf"] = {"status": "structural"}

    drop_stale_structural_compute_states(states, model_nodes=nodes)

    assert "top.module" not in states
    assert "detail.leaf" not in states
    assert states["top.boundary"]["status"] == "structural"
    assert states["top.optional"]["status"] == "not_selected"


def test_materializer_promotes_event_subset_to_typed_fusion() -> None:
    profile = {
        "profile_id": "subset-fixture",
        "node_states": {
            "top.member": {
                "status": "structural",
                "occurrence_fusion_evidence": {"physical_target_proof": True},
            }
        },
        "fusion_groups": {},
    }
    events = [
        {
            "event_id": "event0",
            "ir_node": "top.owner",
            "ir_targets": ["top.owner"],
            "reconciliation_status": "closed",
            "layer_id": 0,
            "occurrence_id": "layer0",
        },
        {
            "event_id": "event1",
            "ir_node": "top.owner",
            "ir_targets": ["top.owner", "top.member"],
            "reconciliation_status": "closed",
            "layer_id": 1,
            "occurrence_id": "layer1",
        },
    ]

    materialize_typed_fusion_states(
        profile,
        decoded_steps=[{"events": events}],
        model_nodes={
            "top.member": {
                "semantic_details": {
                    "runtime_mapping": {"expectation": "measured"}
                }
            }
        },
    )

    assert profile["node_states"]["top.member"] == {
        "status": "fused",
        "included_in": "top.owner",
        "fusion_group_id": "fusion:top.owner",
        "label": "fused into top.owner; 1/2 owner events",
        "provenance": "production_event_graph",
    }
    group = profile["fusion_groups"]["fusion:top.owner"]
    assert group["timing_semantics"] == "shared_event_coverage"
    assert group["evidence_scope"]["owner_event_ids"] == ["event0", "event1"]
    assert group["evidence_scope"]["member_event_ids"] == {
        "top.member": ["event1"]
    }


def test_materializer_partitions_multi_owner_occurrences() -> None:
    profile = {
        "profile_id": "multi-owner-fixture",
        "node_states": {
            "top.member": {
                "status": "structural",
                "occurrence_fusion_evidence": {"physical_target_proof": True},
            }
        },
    }
    events = [
        {
            "event_id": "event0",
            "ir_node": "top.owner_a",
            "ir_targets": ["top.owner_a", "top.member"],
            "reconciliation_status": "closed",
            "layer_id": 0,
            "occurrence_id": "layer0",
        },
        {
            "event_id": "event1",
            "ir_node": "top.owner_b",
            "ir_targets": ["top.owner_b", "top.member"],
            "reconciliation_status": "closed",
            "layer_id": 1,
            "occurrence_id": "layer1",
        },
    ]

    materialize_typed_fusion_states(
        profile,
        decoded_steps=[{"events": events}],
        model_nodes={"top.member": {}},
    )

    state = profile["node_states"]["top.member"]
    assert state["status"] == "fused_by_occurrence"
    assert [partition["included_in"] for partition in state["fusion_partitions"]] == [
        "top.owner_a",
        "top.owner_b",
    ]
    assert [
        partition["production_event_ids"] for partition in state["fusion_partitions"]
    ] == [["event0"], ["event1"]]


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


def test_expand_rollup_targets_respects_shared_event_coverage_membership() -> None:
    group = {
        "owner": "block.owner",
        "ir_nodes": ["block.owner", "block.always", "top.tail"],
        "timing_semantics": "shared_event_coverage",
        "evidence_scope": {
            "member_event_ids": {
                "block.always": ["event-1", "event-2"],
                "top.tail": ["event-2"],
            }
        },
    }

    first = expand_rollup_targets(
        "block.owner",
        existing_targets=["block.owner", "block.always", "top.tail"],
        ancestors={},
        fusion_groups={"fusion:block.owner": group},
        event_id="event-1",
    )
    second = expand_rollup_targets(
        "block.owner",
        ancestors={},
        fusion_groups={"fusion:block.owner": group},
        event_id="event-2",
    )

    assert first == ["block.owner", "block.always"]
    assert second == ["block.owner", "block.always", "top.tail"]


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


def test_selected_runtime_leaf_cannot_fall_back_to_structural() -> None:
    model = {
        "default_view": "top",
        "semantic_contract": {
            "operations": {
                "model.compute": {"kind": "projection"},
                "model.boundary": {"kind": "state_boundary"},
            }
        },
        "views": {
            "top": {
                "nodes": [
                    {
                        "id": "compute",
                        "semantic_op": "model.compute",
                        "semantic_details": {
                            "runtime_mapping": {"expectation": "measured"}
                        },
                    },
                    {
                        "id": "boundary",
                        "semantic_op": "model.boundary",
                        "semantic_details": {
                            "runtime_mapping": {"expectation": "structural"}
                        },
                    },
                ]
            }
        },
    }
    profile = {
        "profile_id": "runtime_leaf_fixture",
        "entry_view": "top",
        "node_metrics": {},
        "node_states": {
            "top.compute": {"status": "structural"},
            "top.boundary": {"status": "structural"},
        },
    }

    with pytest.raises(
        ValueError,
        match="top.compute.*selected runtime-bearing compute node.*structural",
    ):
        validate_executable_drill_rollups(model, profile)

    profile["node_states"]["top.compute"] = {
        "status": "fused",
        "included_in": "top.owner",
    }
    validate_executable_drill_rollups(model, profile)


def test_occurrence_partitioned_fusion_requires_physical_owners() -> None:
    model = {
        "default_view": "top",
        "semantic_contract": {
            "operations": {"model.compute": {"kind": "elementwise"}}
        },
        "views": {
            "top": {
                "nodes": [
                    {
                        "id": "compute",
                        "semantic_op": "model.compute",
                        "semantic_details": {
                            "runtime_mapping": {"expectation": "measured"}
                        },
                    }
                ]
            }
        },
    }
    profile = {
        "profile_id": "occurrence_fusion_fixture",
        "entry_view": "top",
        "node_metrics": {},
        "node_states": {
            "top.compute": {
                "status": "fused_by_occurrence",
                "fusion_partitions": [
                    {
                        "included_in": "top.owner_a",
                        "production_event_ids": ["rank0:event1"],
                    },
                    {
                        "included_in": "top.owner_b",
                        "production_event_ids": ["rank0:event2"],
                    },
                ],
            }
        },
    }

    validate_executable_drill_rollups(model, profile)
    del profile["node_states"]["top.compute"]["fusion_partitions"][0][
        "production_event_ids"
    ]
    with pytest.raises(ValueError, match="fusion_partitions"):
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
