from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import sys

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from llm_arch_v2.compiler import (  # noqa: E402
    CatalogError,
    apply_execution_plan,
    compile_profile,
    execution_fingerprint,
)
from scripts.build_v2 import copy_timeline_artifacts  # noqa: E402


def toy_model() -> dict:
    return {
        "schema_version": "model-ir.v2",
        "model_id": "toy",
        "model_label": "Toy model-neutral fixture",
        "ir_version": 1,
        "views": {
            "top": {
                "title": "Toy",
                "nodes": [
                    {
                        "id": "input",
                        "label": "Input",
                        "shape": "io",
                        "semantic_op": "model.input",
                    },
                    {
                        "id": "compute",
                        "label": "Compute",
                        "shape": "block",
                        "semantic_op": "model.compute",
                    },
                    {
                        "id": "output",
                        "label": "Output",
                        "shape": "io",
                        "semantic_op": "model.output",
                    },
                ],
                "edges": [
                    {"from": "input", "to": "compute"},
                    {"from": "compute", "to": "output"},
                ],
            }
        },
    }


def toy_plan(*, version: int = 2) -> dict:
    return {
        "schema_version": "execution-plan.v2",
        "execution_path_id": "attention_dp_moe_ep4",
        "label": "Toy DP4/EP4 execution",
        "model_id": "toy",
        "plan_version": version,
        "parallelism_axes": {
            "tp_size": 1,
            "dp_size": 4,
            "cp_size": 1,
            "ep_size": 4,
        },
        "transforms": [
            {
                "op": "insert_after",
                "after": "top.compute",
                "node": {
                    "id": "exchange",
                    "label": "Activation exchange",
                    "shape": "elem",
                    "semantic_op": "execution.collective.exchange",
                    "node_kind": "communication",
                    "boundary_role": "module_boundary",
                    "execution": {
                        "placement": "ep_group",
                        "collective": "all_to_all",
                        "parallelism": "EP",
                        "payload": "activations [tokens, hidden], bf16",
                        "result": "expert-local activations [local_tokens, hidden]",
                    },
                },
                "edge": {"dtype": "bf16"},
            }
        ],
    }


def test_model_execution_and_profile_layers_remain_separate() -> None:
    model = toy_model()
    original = copy.deepcopy(model)
    plan = toy_plan()
    views = apply_execution_plan(model, plan, source=Path("toy-plan.yaml"))

    assert model == original
    assert [node["id"] for node in model["views"]["top"]["nodes"]] == [
        "input",
        "compute",
        "output",
    ]
    exchange = next(node for node in views["top"]["nodes"] if node["id"] == "exchange")
    assert exchange["ir_origin"] == "execution_plan"
    assert exchange["node_kind"] == "communication"
    assert exchange["execution"]["payload"].endswith("bf16")

    fingerprint = execution_fingerprint(model, plan, views)
    node_targets = {
        f"{view_id}.{node['id']}"
        for view_id, view in views.items()
        for node in view["nodes"]
    }
    profile = {
        "schema_version": "profile.v2",
        "profile_id": "toy_profile",
        "label": "Toy measured profile",
        "model_id": "toy",
        "execution_path_id": plan["execution_path_id"],
        "implementation_id": "toy_engine_deadbeef",
        "variant_id": "formal_decode",
        "phase": "decode",
        "generation_mode": "mtp",
        "entry_view": "top",
        "execution_parameters": {
            "tp_size": 1,
            "dp_size": 4,
            "cp_size": 1,
            "ep_size": 4,
        },
        "timeline": {
            "schema_version": "timeline.v1",
            "artifact": "toy.timeline.json.gz",
            "sha256": "a" * 64,
            "reference_rank": 0,
            "step_count": 1,
            "event_count": 2,
        },
        "node_states": {
            "top.compute": {
                "status": "fused",
                "included_in": "top.exchange",
            }
        },
        "node_metrics": {
            "top.exchange": {"ms_per_iter": 1.25, "kernels": []}
        },
    }
    compiled = compile_profile(
        profile,
        plan=plan,
        fingerprint=fingerprint,
        node_targets=node_targets,
        source=Path("toy-profile.yaml"),
    )

    assert compiled["execution_variant"] == fingerprint
    assert compiled["meta"]["generation_mode"] == "mtp"
    assert compiled["meta"]["entry_view"] == "top"
    assert compiled["meta"]["timeline"]["url"] == "timelines/toy_profile.timeline.json.gz"
    assert compiled["fusion_groups"]["fusion:top.exchange"] == {
        "owner": "top.exchange",
        "ir_nodes": ["top.exchange", "top.compute"],
        "timing_semantics": "shared_interval",
        "provenance": "profile.node_states",
    }


def test_plan_v2_requires_payload_and_result_but_v1_remains_compatible() -> None:
    model = toy_model()
    strict_plan = toy_plan(version=2)
    del strict_plan["transforms"][0]["node"]["execution"]["payload"]

    with pytest.raises(CatalogError, match="requires execution.payload"):
        apply_execution_plan(model, strict_plan, source=Path("strict-plan.yaml"))

    legacy_plan = copy.deepcopy(strict_plan)
    legacy_plan["plan_version"] = 1
    views = apply_execution_plan(model, legacy_plan, source=Path("legacy-plan.yaml"))
    assert any(node["id"] == "exchange" for node in views["top"]["nodes"])


def test_timeline_artifacts_are_hash_checked_and_copied(tmp_path: Path) -> None:
    profile_dir = tmp_path / "catalog" / "toy" / "profiles" / "path" / "engine"
    profile_dir.mkdir(parents=True)
    artifact = profile_dir / "toy.timeline.json.gz"
    artifact.write_bytes(b"deterministic toy timeline")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    (profile_dir / "toy.yaml").write_text(
        yaml.safe_dump(
            {
                "profile_id": "toy_profile",
                "timeline": {
                    "artifact": artifact.name,
                    "sha256": digest,
                },
            },
            sort_keys=False,
        )
    )
    docs_dir = tmp_path / "docs" / "toy_v2"

    assert copy_timeline_artifacts(profile_dir.parents[2], docs_dir) == 1
    assert (docs_dir / "timelines" / "toy_profile.timeline.json.gz").read_bytes() == (
        artifact.read_bytes()
    )


def test_viewer_contains_bidirectional_architecture_timeline_navigation() -> None:
    viewer = (REPO_ROOT / "docs" / "viewer.html").read_text()
    assert "function showTimelineEventInArchitecture()" in viewer
    assert "function openNodeOnTimeline(viewName, nodeId)" in viewer
    assert "event._irTargets.includes(TIMELINE_IR_TARGET)" in viewer
    assert "Show measured kernels on timeline" in viewer
