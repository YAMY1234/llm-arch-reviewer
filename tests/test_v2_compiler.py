from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from llm_arch_v2.compiler import (  # noqa: E402
    CatalogError,
    apply_execution_plan,
    compile_catalog,
    compile_profile,
    execution_fingerprint,
    load_yaml,
)


MODEL_ROOT = REPO_ROOT / "catalog" / "qwen40"
QWEN40_ROOT = MODEL_ROOT


def _node_ids(bundle: dict, view_id: str) -> list[str]:
    return [node["id"] for node in bundle["views"][view_id]["nodes"]]


def test_compile_qwen40_catalog() -> None:
    bundle = compile_catalog(MODEL_ROOT)

    assert bundle["schema_version"] == "2.0"
    assert bundle["meta"]["catalog"] == "catalog/qwen40"
    assert len(bundle["execution_variants"]) == 4
    assert bundle["default_execution_variant"].startswith("exec_")
    assert bundle["default_implementation"] == "sglang_f90a941aa"
    assert bundle["default_profile"] == "qwen40_tp4_cg_decode_bs1_8k1k"

    assert "tp_attention_collective" in _node_ids(bundle, "linear_layer")
    assert "tp_attention_collective" in _node_ids(bundle, "full_layer")
    assert "tp_output_collective" in _node_ids(bundle, "moe")

    qkvz = next(
        node
        for node in bundle["views"]["linear_attention"]["nodes"]
        if node["id"] == "qkvz_projection"
    )
    assert qkvz["execution"]["parallelism"] == "column_parallel"
    assert qkvz["implementation_binding"]["implementation_id"] == "sglang_f90a941aa"
    assert qkvz["code_links"][0]["url"].startswith(
        "https://github.com/Qiaolin-Yu/sglang-qwen-next/blob/f90a941aa6ff71ac"
    )


def test_compile_qwen40_pure_tp_layout() -> None:
    bundle = compile_catalog(QWEN40_ROOT)

    assert bundle["default_implementation"] == "sglang_f90a941aa"
    assert "qwen40_tp4_cg_decode_bs1_8k1k" in bundle["profiles"]
    assert "tp_embedding_collective" in _node_ids(bundle, "top")
    assert "tp_logits_collective" in _node_ids(bundle, "top")
    assert "tp_embedding_collective" in _node_ids(bundle, "ple")
    assert "tp_attention_collective" in _node_ids(bundle, "linear_layer")
    assert "tp_attention_collective" in _node_ids(bundle, "full_layer")
    assert "tp_output_collective" in _node_ids(bundle, "moe")

    indexer = next(
        node
        for node in bundle["views"]["qsa_attention"]["nodes"]
        if node["id"] == "indexer"
    )
    assert indexer["execution"]["placement"] == "replicated_on_tp_ranks"
    assert indexer["execution"]["tensor_layout"] == "replicated"


def test_qwen40_topology_binding_inherits_common_source_mapping() -> None:
    bundle = compile_catalog(QWEN40_ROOT)
    binding = bundle["implementations"]["sglang_f90a941aa_dp_attention"]

    assert binding["extends"] == "sglang_f90a941aa"
    assert "linear_attention.qkvz_projection" in binding["node_bindings"]
    assert "linear_layer.dp_moe_input_gather" in binding["node_bindings"]
    assert "linear_layer.tp_attention_collective" not in binding["node_bindings"]


def test_insert_after_redirects_existing_output_edge() -> None:
    bundle = compile_catalog(MODEL_ROOT)
    edges = bundle["views"]["linear_layer"]["edges"]

    assert {
        "from": "linear_attention",
        "to": "tp_attention_collective",
        "shape": "[B,T,H]",
        "dtype": "bf16",
    } in edges
    assert any(
        edge["from"] == "tp_attention_collective"
        and edge["to"] == "attn_hc_combine"
        for edge in edges
    )
    assert not any(
        edge["from"] == "linear_attention" and edge["to"] == "attn_hc_combine"
        for edge in edges
    )


def test_execution_fingerprint_excludes_labels_and_profiles() -> None:
    model_path = MODEL_ROOT / "model_ir.yaml"
    plan_path = MODEL_ROOT / "execution_paths" / "tp_only.yaml"
    model = load_yaml(model_path)
    plan = load_yaml(plan_path)
    views = apply_execution_plan(model, plan, source=plan_path)
    expected = execution_fingerprint(model, plan, views)

    relabeled = copy.deepcopy(views)
    relabeled["top"]["nodes"][0]["label"] = "a presentation-only change"
    assert execution_fingerprint(model, plan, relabeled) == expected

    profile = load_yaml(
        MODEL_ROOT
        / "profiles"
        / "tp_only"
        / "sglang_f90a941aa"
        / "cg_decode_bs001_8k1k.yaml"
    )
    profile["node_metrics"]["moe.topk"]["ms_per_iter"] = 999.0
    compiled = compile_profile(
        profile,
        plan=plan,
        fingerprint=expected,
        node_targets={
            f"{view_id}.{node['id']}"
            for view_id, view in views.items()
            for node in view["nodes"]
        },
        source=Path("profile.yaml"),
    )
    assert compiled["execution_variant"] == expected


def test_profile_cannot_create_architecture_nodes() -> None:
    model_path = MODEL_ROOT / "model_ir.yaml"
    plan_path = MODEL_ROOT / "execution_paths" / "tp_only.yaml"
    model = load_yaml(model_path)
    plan = load_yaml(plan_path)
    views = apply_execution_plan(model, plan, source=plan_path)
    fingerprint = execution_fingerprint(model, plan, views)
    profile = load_yaml(
        MODEL_ROOT
        / "profiles"
        / "tp_only"
        / "sglang_f90a941aa"
        / "cg_decode_bs001_8k1k.yaml"
    )
    profile["node_metrics"]["moe.not_a_real_node"] = {"ms_per_iter": 1.0}

    with pytest.raises(CatalogError, match="unknown nodes"):
        compile_profile(
            profile,
            plan=plan,
            fingerprint=fingerprint,
            node_targets={
                f"{view_id}.{node['id']}"
                for view_id, view in views.items()
                for node in view["nodes"]
            },
            source=Path("profile.yaml"),
        )


def test_schema_documents_are_valid_json() -> None:
    schema_root = REPO_ROOT / "schema" / "v2"
    for path in schema_root.glob("*.schema.json"):
        document = json.loads(path.read_text())
        assert document["$schema"].endswith("2020-12/schema")
