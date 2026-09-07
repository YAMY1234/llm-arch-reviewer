from __future__ import annotations

from pathlib import Path

import yaml

from llm_arch_v2 import compile_catalog


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = REPO_ROOT / "catalog" / "glm52"


def _load(relative: str) -> dict:
    return yaml.safe_load((MODEL_ROOT / relative).read_text())


def test_glm52_revision7_pins_every_drill_and_primitive() -> None:
    model = _load("model_ir.yaml")
    release = _load("pipeline.yaml")["acceptance"]["semantic_release_contract"]
    assert model["semantic_revision"] == release["expected_revision"] == 7
    assert model["semantic_coverage"]["detail_view_closure"].startswith("every")

    drills = {
        f"{view_id}.{node['id']}": node["drill"]
        for view_id, view in model["views"].items()
        for node in view["nodes"]
        if node.get("drill")
    }
    assert release["required_drills"] == drills
    assert set(release["required_views"]) == set(drills.values())
    for view_id in release["required_views"]:
        assert release["required_nodes"][view_id] == [
            node["id"] for node in model["views"][view_id]["nodes"]
        ]


def test_glm52_routed_expert_and_nextn_decoder_are_not_opaque() -> None:
    model = _load("model_ir.yaml")
    routed = model["views"]["routed_expert"]
    assert [node["id"] for node in routed["nodes"]] == [
        "hidden_input",
        "route_assignments",
        "dispatch",
        "gate_projection",
        "up_projection",
        "silu",
        "gated_product",
        "down_projection",
        "restore",
        "routed_output",
    ]
    for node in routed["nodes"][2:-1]:
        mapping = node["semantic_details"]["runtime_mapping"]
        assert mapping["expectation"] == "fused"
        assert mapping["owner"] == "moe.routed_experts"

    nextn = model["views"]["mtp_decoder"]
    assert [node["id"] for node in nextn["nodes"]] == [
        "draft_in",
        "attention_norm",
        "dsa_attention",
        "attention_residual",
        "ffn_norm",
        "moe",
        "ffn_residual",
        "draft_out",
    ]
    assert nextn["nodes"][2]["drill"] == "dsa_attention"
    assert nextn["nodes"][5]["drill"] == "moe"


def test_glm52_every_edge_equation_and_framework_binding_closes() -> None:
    model = _load("model_ir.yaml")
    operations = model["semantic_contract"]["operations"]
    for view_id, view in model["views"].items():
        for node in view["nodes"]:
            assert operations[node["semantic_op"]]["equation"], (view_id, node["id"])
        for edge in view["edges"]:
            for field in ("identity", "shape", "layout", "dtype", "state"):
                assert edge.get(field), (view_id, edge, field)

    bundle = compile_catalog(MODEL_ROOT)
    for implementation in bundle["implementations"].values():
        execution = bundle["execution_variants"][implementation["execution_variant"]]
        expected = {
            f"{view_id}.{node['id']}"
            for view_id, view in execution["views"].items()
            for node in view["nodes"]
        }
        assert set(implementation["node_bindings"]) == expected
