from __future__ import annotations

from pathlib import Path

import yaml

from llm_arch_v2 import compile_catalog


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = REPO_ROOT / "catalog" / "deepseek_v4_pro"


def _load(relative: str) -> dict:
    return yaml.safe_load((MODEL_ROOT / relative).read_text())


def test_official_0813_identity_and_layer_schedule_are_exact() -> None:
    model = _load("model_ir.yaml")
    facts = model["facts"]
    assert facts["checkpoint"] == "deepseek-ai/DeepSeek-V4-Pro-0813"
    assert facts["checkpoint_revision"] == (
        "72e1d3230f6c080a530b0a1d46f8eb4602340597"
    )
    assert facts["checkpoint_config_sha256"] == (
        "9dd2a89255469e120b333668ef5a169b7ae46c00f6bbab786bf0be457546aec0"
    )
    assert facts["target_layers"] == 61
    assert len(facts["csa_layers"]) == 30
    assert len(facts["hca_layers"]) == 31
    assert set(facts["csa_layers"]).isdisjoint(facts["hca_layers"])
    assert sorted(facts["csa_layers"] + facts["hca_layers"]) == list(range(61))
    assert facts["hash_router_layers"] == [0, 1, 2]
    assert facts["dspark_stages"] == 3
    assert facts["dspark_target_layer_ids"] == [58, 59, 60]


def test_every_reachable_drill_has_a_boundary_contract_and_every_leaf_has_math() -> None:
    model = _load("model_ir.yaml")
    views = model["views"]
    drills = {
        f"{view_id}.{node['id']}"
        for view_id, view in views.items()
        for node in view["nodes"]
        if node.get("drill")
    }
    contracts = {row["parent_node"] for row in model["boundary_contracts"]}
    assert contracts == drills

    operations = model["semantic_contract"]["operations"]
    for view in views.values():
        for node in view["nodes"]:
            assert node["semantic_op"] in operations
            assert operations[node["semantic_op"]]["equation"]
        for edge in view["edges"]:
            for key in ("identity", "shape", "layout", "dtype", "state"):
                assert edge.get(key), (view["title"], edge, key)


def test_pure_tp4_execution_contract_preserves_attention_and_moe_sharding() -> None:
    bundle = compile_catalog(MODEL_ROOT)
    variants = {
        variant["execution_path_id"]: variant
        for variant in bundle["execution_variants"].values()
    }
    assert set(variants) == {"tp4_moe_intermediate_shard"}
    intermediate = variants["tp4_moe_intermediate_shard"]

    for variant in (intermediate,):
        parameters = variant["default_parameters"]
        assert parameters == {"tp_size": 4, "dp_size": 1, "cp_size": 1, "ep_size": 1}
        nodes = {
            f"{view_id}.{node['id']}": node
            for view_id, view in variant["views"].items()
            for node in view["nodes"]
        }
        assert nodes["csa_attention.q_b"]["execution"]["tensor_layout"] == (
            "32_of_128_query_heads_per_rank"
        )
        assert nodes["csa_attention.indexer"]["execution"]["parallelism"] == (
            "replicated"
        )
        assert nodes["top.dspark_extension"]["execution"]["selection"] == (
            "structurally_retained_not_executed_in_stage1"
        )

    intermediate_moe = next(
        node
        for node in intermediate["views"]["moe"]["nodes"]
        if node["id"] == "routed_gate_up"
    )
    assert intermediate_moe["execution"]["parallelism"] == (
        "tensor_parallel_expert_mlp"
    )


def test_semantic_source_ledger_has_no_pending_dispositions() -> None:
    ledger = _load("semantic_source_ledger.yaml")
    assert ledger["source_snapshot"]["revision"] == (
        "72e1d3230f6c080a530b0a1d46f8eb4602340597"
    )
    assert len(ledger["audit_views"]) == 14
    for entrypoint in ledger["entrypoints"]:
        assert entrypoint["review_status"] == "verified"
        assert all(
            item["disposition"] != "pending"
            for item in entrypoint.get("member_dispositions", [])
        )
        assert all(
            item["disposition"] != "pending"
            for item in entrypoint["obligations"]
        )


def test_both_commit_specific_bindings_cover_every_architecture_node() -> None:
    model = _load("model_ir.yaml")
    expected_nodes = {
        f"{view_id}.{node['id']}"
        for view_id, view in model["views"].items()
        for node in view["nodes"]
    }
    bindings = {
        path.name: yaml.safe_load(path.read_text())
        for path in sorted((MODEL_ROOT / "bindings").glob("*.yaml"))
    }
    assert set(bindings) == {
        "sglang-71de97b-dsv4pro0813-tp4.yaml",
        "vllm-dd10e03-dsv4pro0813-tp4.yaml",
    }
    assert bindings["sglang-71de97b-dsv4pro0813-tp4.yaml"]["source_commit"] == (
        "71de97b264b04dcd514cf904003028aefe9775c8"
    )
    assert bindings["vllm-dd10e03-dsv4pro0813-tp4.yaml"]["source_commit"] == (
        "dd10e03f95f94edbea1975c67ace3a35ec9a8a40"
    )
    for binding in bindings.values():
        assert set(binding["node_bindings"]) == expected_nodes
        for node_id, node_binding in binding["node_bindings"].items():
            assert node_binding["symbols"], node_id
            assert node_binding["links"], node_id
            assert all(link.get("line", 0) > 0 for link in node_binding["links"])
