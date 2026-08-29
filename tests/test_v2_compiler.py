from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest
import yaml


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
QWEN35_ROOT = REPO_ROOT / "catalog" / "qwen35"
AUDITED_MODEL_ROOTS = sorted(
    path
    for path in (REPO_ROOT / "catalog").iterdir()
    if path.is_dir() and (path / "model_ir.yaml").is_file()
)


def _node_ids(bundle: dict, view_id: str) -> list[str]:
    return [node["id"] for node in bundle["views"][view_id]["nodes"]]


@pytest.mark.parametrize("model_root", AUDITED_MODEL_ROOTS, ids=lambda path: path.name)
def test_all_audited_catalogs_compile_without_placeholder_equations(
    model_root: Path,
) -> None:
    bundle = compile_catalog(model_root)
    for view_id, view in bundle["model_ir"]["views"].items():
        for node in view["nodes"]:
            equation = node["semantics"]["equation"]
            assert equation
            assert "None" not in equation, (model_root.name, view_id, node["id"])


def test_compile_qwen40_catalog() -> None:
    bundle = compile_catalog(MODEL_ROOT)

    assert bundle["schema_version"] == "2.0"
    assert bundle["meta"]["catalog"] == "catalog/qwen40"
    assert len(bundle["execution_variants"]) == 4
    assert bundle["default_execution_variant"].startswith("exec_")
    assert bundle["default_implementation"] == "sglang_f90a941aa"
    assert bundle["default_profile"] == "qwen40_tp4_cg_decode_bs1_8k1k"
    timeline = bundle["profiles"][bundle["default_profile"]]["meta"]["timeline"]
    assert timeline["schema_version"] == "timeline.v1"
    assert timeline["url"] == (
        "timelines/qwen40_tp4_cg_decode_bs1_8k1k.timeline.json.gz"
    )
    assert timeline["event_count"] > 0
    disabled_mtp = bundle["profiles"][bundle["default_profile"]]["data"][
        "mtp_head.decoder_layer"
    ]["tp4_cg_decode_bs1_8k1k"]
    assert disabled_mtp["status"] == "disabled"
    assert disabled_mtp["label"] == "MTP is disabled in this profile"

    assert "tp_attention_collective" in _node_ids(bundle, "linear_layer")
    assert "tp_attention_collective" in _node_ids(bundle, "full_layer")
    assert "tp_moe_output_collective" in _node_ids(bundle, "linear_layer")
    assert "tp_moe_output_collective" in _node_ids(bundle, "full_layer")
    assert "tp_output_collective" not in _node_ids(bundle, "moe")
    assert "mtp_generation" in bundle["model_ir"]["views"]
    assert "mtp_head" in bundle["model_ir"]["views"]
    mtp_root_nodes = {
        node["id"]: node for node in bundle["model_ir"]["views"]["mtp_generation"]["nodes"]
    }
    assert mtp_root_nodes["target_prefill"]["drill"] == "top"
    assert mtp_root_nodes["target_verify"]["drill"] == "top"
    assert mtp_root_nodes["mtp_prefill"]["drill"] == "mtp_head"
    assert mtp_root_nodes["mtp_draft_extend"]["drill"] == "mtp_head"
    assert "initialize MTP state + first proposal" in mtp_root_nodes["mtp_prefill"]["label"]
    assert "one-layer MTP draft forward" in mtp_root_nodes["mtp_draft_extend"]["label"]
    assert "finalize next MTP proposal" in mtp_root_nodes["proposal_update"]["label"]
    mtp_edges = bundle["model_ir"]["views"]["mtp_generation"]["edges"]
    assert any(
        edge["from"] == "mtp_draft_extend" and edge["to"] == "proposal_update"
        for edge in mtp_edges
    )
    assert any(
        edge["from"] == "proposal_update" and edge["to"] == "proposal_cache"
        for edge in mtp_edges
    )
    assert "tp_embedding_collective" in _node_ids(bundle, "mtp_head")
    mtp_vocab = next(
        node
        for node in bundle["views"]["mtp_head"]["nodes"]
        if node["id"] == "tp_logits_collective"
    )
    assert "vocabulary resolution" in mtp_vocab["label"]
    assert mtp_vocab["execution"]["collective"] == "all_gather"
    assert "tp_attention_collective" in _node_ids(bundle, "mtp_layer")
    assert "tp_moe_output_collective" in _node_ids(bundle, "mtp_layer")

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


def test_model_ir_and_execution_ir_are_separate_graphs() -> None:
    bundle = compile_catalog(QWEN40_ROOT)
    model_views = bundle["model_ir"]["views"]

    assert "tp_moe_output_collective" not in [
        node["id"] for node in model_views["linear_layer"]["nodes"]
    ]
    assert all(
        node["ir_origin"] == "model_ir"
        for view in model_views.values()
        for node in view["nodes"]
    )

    collective = next(
        node
        for node in bundle["views"]["linear_layer"]["nodes"]
        if node["id"] == "tp_moe_output_collective"
    )
    assert collective["ir_origin"] == "execution_plan"
    assert collective["node_kind"] == "communication"
    assert collective["boundary_role"] == "module_boundary"


def test_generation_mode_is_profile_overlay_not_execution_cross_product() -> None:
    model_path = MODEL_ROOT / "model_ir.yaml"
    plan_path = MODEL_ROOT / "execution_paths" / "tp_only.yaml"
    model = load_yaml(model_path)
    plan = load_yaml(plan_path)
    views = apply_execution_plan(model, plan, source=plan_path)
    fingerprint = execution_fingerprint(model, plan, views)
    raw = load_yaml(
        MODEL_ROOT
        / "profiles"
        / "tp_only"
        / "sglang_f90a941aa"
        / "cg_decode_bs001_8k1k.yaml"
    )
    raw["generation_mode"] = "eagle_mtp"
    raw["entry_view"] = "mtp_generation"
    compiled = compile_profile(
        raw,
        plan=plan,
        fingerprint=fingerprint,
        node_targets={
            f"{view_id}.{node['id']}"
            for view_id, view in views.items()
            for node in view["nodes"]
        },
        source=Path("mtp_profile.yaml"),
    )

    assert compiled["execution_variant"] == fingerprint
    assert compiled["meta"]["generation_mode"] == "eagle_mtp"
    assert compiled["meta"]["entry_view"] == "mtp_generation"
    assert list(compiled["data"]) == sorted(compiled["data"])


def test_profile_data_order_is_canonical_and_complete() -> None:
    model_path = MODEL_ROOT / "model_ir.yaml"
    plan_path = MODEL_ROOT / "execution_paths" / "tp_only.yaml"
    model = load_yaml(model_path)
    plan = load_yaml(plan_path)
    views = apply_execution_plan(model, plan, source=plan_path)
    fingerprint = execution_fingerprint(model, plan, views)
    raw = load_yaml(
        MODEL_ROOT
        / "profiles"
        / "tp_only"
        / "sglang_f90a941aa"
        / "cg_decode_bs001_8k1k.yaml"
    )
    node_targets = {
        f"{view_id}.{node['id']}"
        for view_id, view in views.items()
        for node in view["nodes"]
    }
    compiled = compile_profile(
        raw,
        plan=plan,
        fingerprint=fingerprint,
        node_targets=node_targets,
        source=Path("profile.yaml"),
    )
    expected = set(raw.get("node_states") or {})
    expected.update(target for target in node_targets if target.startswith("mtp_"))
    expected.update(raw.get("node_metrics") or {})

    assert list(compiled["data"]) == sorted(expected)


def test_fused_profile_states_compile_to_shared_interval_groups() -> None:
    bundle = compile_catalog(QWEN40_ROOT)
    profile = bundle["profiles"]["qwen40_tp4_cg_decode_bs1_8k1k"]
    group = profile["fusion_groups"]["fusion:linear_attention.qkvz_projection"]

    assert group == {
        "owner": "linear_attention.qkvz_projection",
        "ir_nodes": [
            "linear_attention.qkvz_projection",
            "linear_attention.ba_projection",
        ],
        "timing_semantics": "shared_interval",
        "provenance": "profile.node_states",
    }
    fused_cell = profile["data"]["linear_attention.ba_projection"][
        "tp4_cg_decode_bs1_8k1k"
    ]
    group_id = "fusion:linear_attention.qkvz_projection"
    assert fused_cell["fusion_group_id"] == group_id
    assert group_id in profile["fusion_groups"]


def test_qwen35_collective_adapters_live_on_layer_boundaries() -> None:
    bundle = compile_catalog(QWEN35_ROOT)

    assert "tp_output_collective" not in _node_ids(bundle, "linear_attention")
    assert "tp_output_collective" not in _node_ids(bundle, "full_attention")
    assert "tp_output_collective" not in _node_ids(bundle, "moe")
    assert "tp_attention_output_collective" in _node_ids(bundle, "linear_layer")
    assert "tp_moe_output_collective" in _node_ids(bundle, "linear_layer")
    assert "tp_attention_output_collective" in _node_ids(bundle, "full_layer")
    assert "tp_moe_output_collective" in _node_ids(bundle, "full_layer")


def test_compile_qwen40_pure_tp_layout() -> None:
    bundle = compile_catalog(QWEN40_ROOT)

    assert bundle["default_implementation"] == "sglang_f90a941aa"
    assert "qwen40_tp4_cg_decode_bs1_8k1k" in bundle["profiles"]
    assert "tp_embedding_collective" in _node_ids(bundle, "top")
    assert "tp_logits_collective" in _node_ids(bundle, "top")
    assert "tp_embedding_collective" in _node_ids(bundle, "ple")
    assert "tp_attention_collective" in _node_ids(bundle, "linear_layer")
    assert "tp_attention_collective" in _node_ids(bundle, "full_layer")
    assert "tp_moe_output_collective" in _node_ids(bundle, "linear_layer")
    assert "tp_moe_output_collective" in _node_ids(bundle, "full_layer")
    assert "tp_output_collective" not in _node_ids(bundle, "moe")

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


def test_qwen40_qwen4_main_binding_explicitly_reuses_base_semantics() -> None:
    bundle = compile_catalog(QWEN40_ROOT)
    binding = bundle["implementations"][
        "sglang_qwen4_main_32e9cb5_qsa_hardening_flashinfer_gdn"
    ]

    assert binding["source_commit"] == "32e9cb5b95104dc3a10b96bafae7afa50052d94d"
    assert (
        binding["binding_compatible_base_commit"]
        == "f90a941aa6ff71ac3bd7d40b8daccdf5bd914af0"
    )
    assert binding["source_patch_sha256"] == (
        "07c22e094da7103011301ced5824134e0387b310a5a03df0579bdd7ed08f17b3"
    )
    assert "mtp_generation.target_verify" in binding["node_bindings"]
    assert "linear_attention.delta_rule" in binding["node_bindings"]
    assert "/blob/32e9cb5" in binding["node_bindings"][
        "mtp_generation.target_verify"
    ]["code_links"][0]["url"]


def test_binding_validation_attestation_is_preserved_and_fingerprint_checked(
    tmp_path: Path,
) -> None:
    model_root = tmp_path / "qwen40"
    import shutil

    shutil.copytree(QWEN40_ROOT, model_root)
    binding_path = model_root / "bindings" / "sglang_f90a941aa.yaml"
    binding = load_yaml(binding_path)
    plan_path = model_root / "execution_paths" / "tp_only.yaml"
    model = load_yaml(model_root / "model_ir.yaml")
    plan = load_yaml(plan_path)
    views = apply_execution_plan(model, plan, source=plan_path)
    fingerprint = execution_fingerprint(model, plan, views)
    binding["binding_status"] = "draft"
    binding["source_lock_status"] = "provisional"
    binding["execution_validation"] = {
        "status": "pending",
        "execution_fingerprint": fingerprint,
        "required_phases": ["prefill", "decode"],
        "cuda_graph_enabled": False,
    }
    binding_path.write_text(yaml.safe_dump(binding, sort_keys=False))

    compiled = compile_catalog(model_root)["implementations"][
        binding["implementation_id"]
    ]
    assert compiled["binding_status"] == "draft"
    assert compiled["execution_validation"]["status"] == "pending"

    binding["execution_validation"]["execution_fingerprint"] = "exec_0000000000000000"
    binding_path.write_text(yaml.safe_dump(binding, sort_keys=False))
    with pytest.raises(CatalogError, match="execution_validation fingerprint"):
        compile_catalog(model_root)


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

    assert any(
        edge["from"] == "moe" and edge["to"] == "tp_moe_output_collective"
        for edge in edges
    )
    assert any(
        edge["from"] == "tp_moe_output_collective"
        and edge["to"] == "mlp_hc_combine"
        for edge in edges
    )
    assert not any(
        edge["from"] == "moe" and edge["to"] == "mlp_hc_combine"
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


def test_execution_communication_requires_payload_and_result() -> None:
    model_path = MODEL_ROOT / "model_ir.yaml"
    plan_path = MODEL_ROOT / "execution_paths" / "tp_only.yaml"
    model = load_yaml(model_path)
    plan = load_yaml(plan_path)
    inserted = next(
        transform for transform in plan["transforms"] if transform["op"] == "insert_after"
    )
    del inserted["node"]["execution"]["payload"]

    with pytest.raises(CatalogError, match="requires execution.payload"):
        apply_execution_plan(model, plan, source=plan_path)


def test_schema_documents_are_valid_json() -> None:
    schema_root = REPO_ROOT / "schema" / "v2"
    for path in schema_root.glob("*.schema.json"):
        document = json.loads(path.read_text())
        assert document["$schema"].endswith("2020-12/schema")
