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
    _validate_leaf_equation_coverage,
    _validate_notation_contract,
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
            # ``None`` is valid inside authored indexing expressions such as
            # ``x[:, None, :]``. Reject compiler placeholders, not that syntax.
            assert equation not in {"None", "None = None(None)"}, (
                model_root.name,
                view_id,
                node["id"],
            )
            equation_exemption = (node.get("semantic_details") or {}).get(
                "equation_exempt_reason"
            )
            if (
                int(bundle["model_ir"].get("semantic_revision") or 0) >= 6
                and node["semantics"]["kind"] == "compute"
                and not equation_exemption
            ):
                assert "Composite semantic module" not in equation, (
                    model_root.name,
                    view_id,
                    node["id"],
                )

    scalar_timing_fields = {
        "ms_per_iter",
        "active_gpu_ms",
        "gpu_residency_ms",
        "gpu_residency_ms_per_iter",
        "gpu_elapsed_ms",
        "module_gap_ms",
        "device_idle_ms",
        "other_gpu_work_ms",
    }
    for profile_id, profile in bundle["profiles"].items():
        variant = profile["meta"]["variant_id"]
        for target, variants in profile["data"].items():
            cell = variants.get(variant) or {}
            if cell.get("status") != "fused":
                continue
            assert cell["timing_role"] == "fused_member", (
                model_root.name,
                profile_id,
                target,
            )
            assert cell["shared_timing_owner"] == cell["included_in"]
            assert scalar_timing_fields.isdisjoint(cell), (
                model_root.name,
                profile_id,
                target,
                scalar_timing_fields.intersection(cell),
            )


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


def test_qwen40_model_ir_has_semantic_closure_ledgers() -> None:
    bundle = compile_catalog(QWEN40_ROOT)
    model_ir = bundle["model_ir"]

    assert model_ir["semantic_revision"] == 6
    assert model_ir["semantic_coverage"]["operator_dataflow_closure"] == (
        "complete_against_pinned_source_089f8ac"
    )
    assert model_ir["semantic_coverage"]["parameter_closure"] == (
        "complete_for_target_text_and_declared_mtp"
    )
    ledger = model_ir["facts"]["parameter_ledger"]
    assert ledger["target_unique_total"] == 177392830576
    assert ledger["target_text_total"] + ledger["vision_total"] == ledger[
        "target_unique_total"
    ]
    assert ledger["target_unique_total"] + ledger["mtp_additional_unique"] == ledger[
        "target_plus_mtp_unique_total"
    ]
    assert ledger["moe_per_layer"] * 48 == ledger["moe_all_48_layers"]
    assert ledger["gdn_core_per_layer"] * 36 == ledger["gdn_core_all_36_layers"]
    assert ledger["qsa_core_per_layer"] * 12 == ledger["qsa_core_all_12_layers"]
    assert 51200245760 + 32768000 + 30720 + 40960 == ledger["ple_module_total"]
    assert (10240 + 6553600 + 40960) * 2 == ledger[
        "hyperconnection_per_layer"
    ]
    assert model_ir["facts"]["state_ledger"]["gdn_per_layer"]["growth"] == (
        "fixed_per_request"
    )
    assert "qsa_indexer" in model_ir["views"]

    assert model_ir["dimensions"]["H"] == 2560
    assert model_ir["dimensions"]["E"] == "512 routed experts"
    assert model_ir["dimensions"]["I"] == "640 expert intermediate dimension"
    router = next(
        node for node in model_ir["views"]["moe"]["nodes"] if node["id"] == "router"
    )
    assert router["operator_signature"] == {
        "symbolic": "H → E",
        "concrete": "2560 → 512",
    }
    for view in model_ir["views"].values():
        for edge in view.get("edges", []):
            if edge.get("kind", "data") == "control":
                continue
            assert edge.get("shape")
            assert edge.get("dtype")

    qsa_indexer = next(
        node
        for node in model_ir["views"]["qsa_attention"]["nodes"]
        if node["id"] == "indexer"
    )
    assert qsa_indexer["drill"] == "qsa_indexer"
    assert qsa_indexer["semantic_details"]["parameters"]["total"] == 1638656

    gdn_state = next(
        node
        for node in model_ir["views"]["linear_attention"]["nodes"]
        if node["id"] == "recurrent_state"
    )
    assert gdn_state["semantic_details"]["state"][0]["shape"] == (
        "[B,48,128,128]"
    )
    moe = next(
        node
        for node in model_ir["views"]["full_layer"]["nodes"]
        if node["id"] == "moe"
    )
    assert moe["semantic_details"]["parameters"]["total"] == 2522810880


def test_checked_in_qwen40_bundle_matches_canonical_catalog() -> None:
    """Prevent an enriched catalog from being published with a stale bundle."""

    generated = json.loads(
        (REPO_ROOT / "docs" / "qwen40_v2" / "arch_data.json").read_text()
    )
    compiled = compile_catalog(QWEN40_ROOT)

    assert generated == compiled
    assert generated["meta"]["model_semantic_revision"] == 6
    assert "/Users/" not in json.dumps(generated["model_ir"]["semantic_evidence"])
    for required_view in (
        "hyperconnection_read",
        "ple_grouped_norm_gate",
        "linear_attention",
        "qsa_indexer",
        "moe_routed_expert",
        "moe_shared_expert",
        "mtp_generation",
        "mtp_head",
    ):
        assert required_view in generated["model_ir"]["views"]


def test_notation_contract_rejects_undeclared_symbols_and_untyped_edges() -> None:
    model_ir = load_yaml(QWEN40_ROOT / "model_ir.yaml")
    broken = copy.deepcopy(model_ir)
    broken["views"]["moe"]["nodes"][1]["operator_signature"]["symbolic"] = (
        "H → UNKNOWN"
    )
    with pytest.raises(CatalogError, match="undeclared dimension symbols"):
        _validate_notation_contract(broken, source=QWEN40_ROOT / "model_ir.yaml")

    broken = copy.deepcopy(model_ir)
    del broken["views"]["moe"]["edges"][0]["dtype"]
    with pytest.raises(CatalogError, match="tensor-carrying edge"):
        _validate_notation_contract(broken, source=QWEN40_ROOT / "model_ir.yaml")


def test_qwen40_compute_leaf_requires_authored_equation_or_exemption() -> None:
    model_ir = load_yaml(QWEN40_ROOT / "model_ir.yaml")
    broken = copy.deepcopy(model_ir)
    target = next(
        node
        for node in broken["views"]["moe_routed_expert"]["nodes"]
        if node["id"] == "silu"
    )
    target["semantic_details"].pop("math")
    with pytest.raises(CatalogError, match="compute leaf moe_routed_expert.silu"):
        _validate_leaf_equation_coverage(
            broken, source=QWEN40_ROOT / "model_ir.yaml"
        )


def test_operator_signature_does_not_change_execution_fingerprint() -> None:
    model_ir = load_yaml(QWEN40_ROOT / "model_ir.yaml")
    plan = load_yaml(QWEN40_ROOT / "execution_paths" / "tp_only.yaml")
    baseline_views = apply_execution_plan(model_ir, plan, source=QWEN40_ROOT)
    baseline = execution_fingerprint(model_ir, plan, baseline_views)

    relabeled = copy.deepcopy(model_ir)
    relabeled["views"]["moe"]["nodes"][1]["operator_signature"]["concrete"] = (
        "resolved elsewhere"
    )
    relabeled_views = apply_execution_plan(relabeled, plan, source=QWEN40_ROOT)
    assert execution_fingerprint(relabeled, plan, relabeled_views) == baseline


def test_qwen40_compound_math_drills_to_primitive_model_ir_nodes() -> None:
    bundle = compile_catalog(QWEN40_ROOT)
    model_views = bundle["model_ir"]["views"]

    # Revision 6 closes previously missing tensor-edge contracts. Its new
    # fingerprint is the stable baseline; operator_signature itself remains
    # presentation/semantic metadata and is excluded from the payload.
    # The canonical MTP generation graph now includes the explicit
    # proposal-update boundary retained from main, so the structural
    # execution fingerprint differs from the pre-merge enrichment branch.
    assert bundle["default_execution_variant"] == "exec_2ae15643d6883b58"

    mix_gate = next(
        node
        for node in model_views["hyperconnection_mix"]["nodes"]
        if node["id"] == "low_rank_gate"
    )
    assert mix_gate["drill"] == "hyperconnection_read"
    assert _node_ids(bundle["model_ir"], "hyperconnection_read") == [
        "normalized_branches",
        "down_projection",
        "scaled_silu",
        "up_projection",
        "sigmoid_view",
        "weighted_apply",
        "branch_mean",
        "module_input",
    ]

    routed = next(
        node
        for node in model_views["moe"]["nodes"]
        if node["id"] == "routed_experts"
    )
    assert routed["drill"] == "moe_routed_expert"
    assert _node_ids(bundle["model_ir"], "moe_routed_expert")[1:-1] == [
        "gate_projection",
        "up_projection",
        "silu",
        "gated_product",
        "down_projection",
    ]


def test_fine_model_ir_nodes_share_measured_fusion_owner_without_double_counting() -> None:
    bundle = compile_catalog(QWEN40_ROOT)
    profile = bundle["profiles"]["qwen40_tp4_cg_decode_bs1_8k1k"]
    group = profile["fusion_groups"]["fusion:hyperconnection_mix.mix"]

    assert group["owner"] == "hyperconnection_mix.mix"
    assert "hyperconnection_read.down_projection" in group["ir_nodes"]
    assert "hyperconnection_read.branch_mean" in group["ir_nodes"]
    cell = profile["data"]["hyperconnection_read.down_projection"][
        "tp4_cg_decode_bs1_8k1k"
    ]
    assert cell["status"] == "fused"
    assert cell["included_in"] == "hyperconnection_mix.mix"
    assert cell["fusion_timing_semantics"] == "shared_interval"
    assert cell["timing_role"] == "fused_member"
    assert cell["shared_timing_owner"] == "hyperconnection_mix.mix"
    for field in (
        "ms_per_iter",
        "active_gpu_ms",
        "gpu_residency_ms",
        "gpu_elapsed_ms",
        "module_gap_ms",
        "device_idle_ms",
        "other_gpu_work_ms",
    ):
        assert field not in cell

    owner = profile["data"]["hyperconnection_mix.mix"][
        "tp4_cg_decode_bs1_8k1k"
    ]
    assert owner["timing_role"] == "fusion_owner"
    assert owner["active_gpu_ms"] > 0

    node = next(
        item
        for item in bundle["views"]["hyperconnection_read"]["nodes"]
        if item["id"] == "down_projection"
    )
    assert node["implementation_binding"]["mapping_provenance"] == (
        "shared_fused_owner"
    )


def test_hidden_timing_aggregate_declares_reachable_architecture_owner() -> None:
    bundle = compile_catalog(QWEN40_ROOT)
    for profile_id in (
        "qwen40_tp4_cg_decode_bs1_8k1k",
        "qwen40_tp4_mtp_cg_decode_gbs001_8k1k",
    ):
        profile = bundle["profiles"][profile_id]
        group = profile["fusion_groups"]["fusion:hyperconnection.mix"]
        assert group["owner"] == "hyperconnection.mix"
        assert group["architecture_owner"] == "hyperconnection_mix.mix"


def test_profile_rejects_unreachable_fusion_architecture_owner() -> None:
    model = load_yaml(QWEN40_ROOT / "model_ir.yaml")
    plan_path = QWEN40_ROOT / "execution_paths" / "tp_only.yaml"
    plan = load_yaml(plan_path)
    views = apply_execution_plan(model, plan, source=plan_path)
    node_index = {
        f"{view_id}.{node['id']}": node
        for view_id, view in views.items()
        for node in view["nodes"]
    }
    # The legacy all-call aggregate exists, but it is intentionally absent
    # from the canonical architecture drill tree. Removing its explicit
    # architecture destination must therefore fail closed.
    node_index["hyperconnection.mix"].pop("architecture_target")
    profile = load_yaml(
        QWEN40_ROOT
        / "profiles"
        / "tp_only"
        / "sglang_f90a941aa"
        / "cg_decode_bs001_8k1k.yaml"
    )

    with pytest.raises(CatalogError, match="not reachable from entry_view"):
        compile_profile(
            profile,
            plan=plan,
            fingerprint=execution_fingerprint(model, plan, views),
            node_targets=set(node_index),
            node_index=node_index,
            views=views,
            source=Path("profile.yaml"),
        )


def test_qwen40_qsa_indexer_drill_has_reconciled_binding_and_profile() -> None:
    bundle = compile_catalog(QWEN40_ROOT)
    implementation = bundle["implementations"]["sglang_f90a941aa"]
    for target in (
        "qsa_indexer.qk_projection",
        "qsa_indexer.q_norm_rope",
        "qsa_indexer.compress",
        "qsa_indexer.compressed_score",
        "qsa_indexer.block_topk",
        "qsa_indexer.expand_tail",
    ):
        assert target in implementation["node_bindings"]

    profile = bundle["profiles"]["qwen40_tp4_cg_decode_bs1_8k1k"]
    cell = profile["data"]["qsa_attention.indexer"]["tp4_cg_decode_bs1_8k1k"]
    assert cell["drill_view"] == "qsa_indexer"
    assert cell["drill_mapping_coverage_pct"] == 100.0
    assert cell["drill_metrics"]["raw_k_cache"]["status"] == "fused"
    assert cell["drill_metrics"]["compressed_k_cache"]["included_in"] == (
        "qsa_indexer.compress"
    )


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

    assert "tp_attention_output_collective" not in _node_ids(bundle, "gdn_attention")
    assert "tp_attention_output_collective" not in _node_ids(bundle, "full_attention")
    assert "tp_moe_output_collective" not in _node_ids(bundle, "moe_block")
    assert "tp_attention_output_collective" in _node_ids(bundle, "gdn_moe_block")
    assert "tp_moe_output_collective" in _node_ids(bundle, "gdn_moe_block")
    assert "tp_attention_output_collective" in _node_ids(bundle, "full_attention_moe_block")
    assert "tp_moe_output_collective" in _node_ids(bundle, "full_attention_moe_block")


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


def test_profile_rejects_fused_state_with_independent_timing() -> None:
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
    profile.setdefault("node_states", {})["moe.topk"] = {
        "status": "fused",
        "included_in": "moe.routed_experts",
    }

    with pytest.raises(
        CatalogError,
        match="cannot also carry independent node_metrics",
    ):
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
