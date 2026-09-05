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
    _validate_binding_revision_contract,
    _validate_leaf_equation_coverage,
    _validate_notation_contract,
    apply_execution_plan,
    comparison_contract,
    compile_catalog,
    compile_profile,
    execution_fingerprint,
    load_yaml,
)
from llm_arch_v2.add_trace import (  # noqa: E402
    binding_revision_id,
    mapping_rules_sha256,
    runtime_identity_sha256,
)


MODEL_ROOT = REPO_ROOT / "catalog" / "qwen38_flash_next"
QWEN38_FLASH_NEXT_ROOT = MODEL_ROOT
QWEN35_ROOT = REPO_ROOT / "catalog" / "qwen35"
AUDITED_MODEL_ROOTS = sorted(
    path
    for path in (REPO_ROOT / "catalog").iterdir()
    if path.is_dir() and (path / "model_ir.yaml").is_file()
)


def _node_ids(bundle: dict, view_id: str) -> list[str]:
    return [node["id"] for node in bundle["views"][view_id]["nodes"]]


def test_catalog_binding_revision_seals_mapping_rule_content() -> None:
    execution_id = "exec_" + "1" * 16
    identity = {
        "framework_id": "sglang",
        "source_repo": "https://github.com/sgl-project/sglang",
        "source_commit": "2" * 40,
        "container_digest": "sha256:" + "3" * 64,
        "package_lock_sha256": "4" * 64,
        "extension_artifacts": [],
        "backend_selections": {},
        "build_flags": {},
    }
    rule = {
        "rule_id": "rule.one",
        "ir_target": "top.node",
        "eager_match": {"python_stack_digest": "5" * 64},
        "production_transfer": {
            "method": "exact_sequence",
            "signature_digest": "6" * 64,
        },
        "scope": {"phase": "decode", "generation_mode": "autoregressive"},
    }
    identity_digest = runtime_identity_sha256(identity, source=Path("fixture.yaml"))
    binding = {
        "source_repo": identity["source_repo"],
        "source_commit": identity["source_commit"],
        "framework_id": "sglang",
        "binding_revision_id": binding_revision_id(identity_digest, execution_id),
        "add_trace_acceptance_sha256": "8" * 64,
        "runtime_identity": identity,
        "runtime_identity_sha256": identity_digest,
        "mapping_rules": [rule],
        "mapping_rules_sha256": mapping_rules_sha256([rule]),
    }
    _validate_binding_revision_contract(
        binding,
        execution_fingerprint_value=execution_id,
        node_targets={"top.node"},
        source=Path("fixture.yaml"),
    )
    binding["mapping_rules"][0]["production_transfer"]["signature_digest"] = "7" * 64
    with pytest.raises(CatalogError, match="mapping_rules_sha256"):
        _validate_binding_revision_contract(
            binding,
            execution_fingerprint_value=execution_id,
            node_targets={"top.node"},
            source=Path("fixture.yaml"),
        )
    binding["mapping_rules"][0]["production_transfer"].update(
        method="kernel_name_guess", signature_digest="7" * 64
    )
    binding["mapping_rules_sha256"] = mapping_rules_sha256(binding["mapping_rules"])
    with pytest.raises(CatalogError, match="invalid production transfer"):
        _validate_binding_revision_contract(
            binding,
            execution_fingerprint_value=execution_id,
            node_targets={"top.node"},
            source=Path("fixture.yaml"),
        )


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


def test_comparison_contract_matches_equivalent_framework_profiles() -> None:
    bundle = compile_catalog(QWEN35_ROOT)
    assert (
        bundle["implementations"]["sglang_f609d677b_qwen35_033446bb_tp8"][
            "framework_id"
        ]
        == "sglang"
    )
    assert (
        bundle["implementations"]["vllm_487ecf187_qwen35_native_tp8"]["framework_id"]
        == "vllm"
    )
    sglang = bundle["profiles"]["qwen35_tp8_sglang_cg_decode_bs64_8k1k"]
    vllm = bundle["profiles"]["qwen35_tp8_vllm_cg_decode_bs64_8k1k"]
    assert (
        sglang["meta"]["comparison_contract_id"]
        == vllm["meta"]["comparison_contract_id"]
    )
    assert sglang["meta"]["comparison_contract"] == vllm["meta"]["comparison_contract"]
    contract_id = sglang["meta"]["comparison_contract_id"]
    assert bundle["comparison_contracts"][contract_id][
        "profiles_by_implementation"
    ] == {
        "sglang_f609d677b_qwen35_033446bb_tp8": "qwen35_tp8_sglang_cg_decode_bs64_8k1k",
        "vllm_487ecf187_qwen35_native_tp8": "qwen35_tp8_vllm_cg_decode_bs64_8k1k",
    }
    assert bundle["comparison_contracts"][contract_id]["execution_ir_compatible"]


def test_comparison_contract_rejects_non_equivalent_production_modes() -> None:
    profile = {
        "model_id": "toy",
        "phase": "decode",
        "generation_mode": "autoregressive",
        "execution_parameters": {
            "tp_size": 8,
            "dp_size": 1,
            "cp_size": 1,
            "ep_size": 1,
        },
        "hardware": {"gpu": "GB300", "nodes": 2, "gpus_per_node": 4},
        "workload": {"isl": 8192, "osl": 1024, "batch_size": 64},
        "profiler": {"cuda_graph_enabled": True},
    }
    graph_id, _ = comparison_contract(profile, fingerprint="exec_contract")
    profile["profiler"]["cuda_graph_enabled"] = False
    eager_id, _ = comparison_contract(profile, fingerprint="exec_contract")
    assert graph_id != eager_id
    profile["profiler"]["cuda_graph_enabled"] = True
    profile["workload"]["batch_size"] = 16
    other_batch_id, _ = comparison_contract(profile, fingerprint="exec_contract")
    assert graph_id != other_batch_id
    profile["workload"]["batch_size"] = 64
    profile["comparison_config"] = {"attention_backend": "different_contract"}
    other_config_id, _ = comparison_contract(profile, fingerprint="exec_contract")
    assert graph_id != other_config_id


def test_glm52_comparison_matches_distinct_execution_ir_without_collapsing() -> None:
    bundle = compile_catalog(REPO_ROOT / "catalog" / "glm52")
    sglang = bundle["profiles"]["glm52_tp8_sglang_cg_decode_bs64_8k1k"]
    trtllm = bundle["profiles"]["glm52_tp8_trtllm_cg_decode_bs64_8k1k"]
    contract_id = sglang["meta"]["comparison_contract_id"]
    assert contract_id == trtllm["meta"]["comparison_contract_id"]
    contract = bundle["comparison_contracts"][contract_id]
    assert contract["profiles_by_implementation"] == {
        "sglang_fdebc938_dsa": "glm52_tp8_sglang_cg_decode_bs64_8k1k",
        "trtllm_4358fb5d_dsa": "glm52_tp8_trtllm_cg_decode_bs64_8k1k",
    }
    assert not contract["execution_ir_compatible"]
    assert len(set(contract["execution_variants_by_implementation"].values())) == 2


def test_compile_qwen38_flash_next_catalog() -> None:
    bundle = compile_catalog(MODEL_ROOT)

    assert bundle["schema_version"] == "2.0"
    assert bundle["meta"]["catalog"] == "catalog/qwen38_flash_next"
    assert len(bundle["execution_variants"]) == 5
    assert {
        variant["execution_path_id"]
        for variant in bundle["execution_variants"].values()
    } == {
        "tp_only",
        "tp_only_eagle_mtp",
        "dp_attention",
        "dp_attention_moe_ep_deepep_deepgemm",
        "moe_ep_a2a_none",
    }
    assert bundle["default_execution_variant"].startswith("exec_")
    assert bundle["default_implementation"] == "sglang_f90a941aa"
    assert bundle["default_profile"] == "qwen38_flash_next_tp4_cg_decode_bs1_8k1k"
    timeline = bundle["profiles"][bundle["default_profile"]]["meta"]["timeline"]
    assert timeline["schema_version"] == "timeline.v1"
    assert timeline["url"] == (
        "timelines/qwen38_flash_next_tp4_cg_decode_bs1_8k1k.timeline.json.gz"
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
        node["id"]: node
        for node in bundle["model_ir"]["views"]["mtp_generation"]["nodes"]
    }
    assert mtp_root_nodes["target_prefill"]["drill"] == "top"
    assert mtp_root_nodes["target_verify"]["drill"] == "top"
    assert mtp_root_nodes["mtp_prefill"]["drill"] == "mtp_head"
    assert mtp_root_nodes["mtp_draft_extend"]["drill"] == "mtp_head"
    assert (
        "initialize MTP state + first proposal"
        in mtp_root_nodes["mtp_prefill"]["label"]
    )
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
    bundle = compile_catalog(QWEN38_FLASH_NEXT_ROOT)
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


def test_qwen38_flash_next_model_ir_has_semantic_closure_ledgers() -> None:
    bundle = compile_catalog(QWEN38_FLASH_NEXT_ROOT)
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
    assert (
        ledger["target_text_total"] + ledger["vision_total"]
        == ledger["target_unique_total"]
    )
    assert (
        ledger["target_unique_total"] + ledger["mtp_additional_unique"]
        == ledger["target_plus_mtp_unique_total"]
    )
    assert ledger["moe_per_layer"] * 48 == ledger["moe_all_48_layers"]
    assert ledger["gdn_core_per_layer"] * 36 == ledger["gdn_core_all_36_layers"]
    assert ledger["qsa_core_per_layer"] * 12 == ledger["qsa_core_all_12_layers"]
    assert 51200245760 + 32768000 + 30720 + 40960 == ledger["ple_module_total"]
    assert (10240 + 6553600 + 40960) * 2 == ledger["hyperconnection_per_layer"]
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
    assert gdn_state["semantic_details"]["state"][0]["shape"] == ("[B,48,128,128]")
    moe = next(
        node for node in model_ir["views"]["full_layer"]["nodes"] if node["id"] == "moe"
    )
    assert moe["semantic_details"]["parameters"]["total"] == 2522810880


def test_checked_in_qwen38_flash_next_bundle_matches_canonical_catalog() -> None:
    """Prevent an enriched catalog from being published with a stale bundle."""

    generated = json.loads(
        (REPO_ROOT / "docs" / "qwen38_flash_next_v2" / "arch_data.json").read_text()
    )
    compiled = compile_catalog(QWEN38_FLASH_NEXT_ROOT)

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
    model_ir = load_yaml(QWEN38_FLASH_NEXT_ROOT / "model_ir.yaml")
    broken = copy.deepcopy(model_ir)
    broken["views"]["moe"]["nodes"][1]["operator_signature"]["symbolic"] = "H → UNKNOWN"
    with pytest.raises(CatalogError, match="undeclared dimension symbols"):
        _validate_notation_contract(
            broken, source=QWEN38_FLASH_NEXT_ROOT / "model_ir.yaml"
        )

    broken = copy.deepcopy(model_ir)
    del broken["views"]["moe"]["edges"][0]["dtype"]
    with pytest.raises(CatalogError, match="tensor-carrying edge"):
        _validate_notation_contract(
            broken, source=QWEN38_FLASH_NEXT_ROOT / "model_ir.yaml"
        )


def test_qwen38_flash_next_compute_leaf_requires_authored_equation_or_exemption() -> (
    None
):
    model_ir = load_yaml(QWEN38_FLASH_NEXT_ROOT / "model_ir.yaml")
    broken = copy.deepcopy(model_ir)
    target = next(
        node
        for node in broken["views"]["moe_routed_expert"]["nodes"]
        if node["id"] == "silu"
    )
    target["semantic_details"].pop("math")
    with pytest.raises(CatalogError, match="compute leaf moe_routed_expert.silu"):
        _validate_leaf_equation_coverage(
            broken, source=QWEN38_FLASH_NEXT_ROOT / "model_ir.yaml"
        )


def test_semantic_contract_operation_equation_satisfies_leaf_coverage() -> None:
    model_ir = {
        "semantic_revision": 6,
        "semantic_contract": {
            "operations": {
                "example.scale": {
                    "kind": "elementwise",
                    "equation": "y = alpha * x",
                }
            }
        },
        "views": {
            "top": {
                "nodes": [
                    {
                        "id": "scale",
                        "label": "scale",
                        "shape": "elem",
                        "semantic_op": "example.scale",
                    }
                ]
            }
        },
    }

    _validate_leaf_equation_coverage(
        model_ir, source=Path("catalog/example/model_ir.yaml")
    )


def test_operator_signature_does_not_change_execution_fingerprint() -> None:
    model_ir = load_yaml(QWEN38_FLASH_NEXT_ROOT / "model_ir.yaml")
    plan = load_yaml(QWEN38_FLASH_NEXT_ROOT / "execution_paths" / "tp_only.yaml")
    baseline_views = apply_execution_plan(model_ir, plan, source=QWEN38_FLASH_NEXT_ROOT)
    baseline = execution_fingerprint(model_ir, plan, baseline_views)

    relabeled = copy.deepcopy(model_ir)
    relabeled["views"]["moe"]["nodes"][1]["operator_signature"][
        "concrete"
    ] = "resolved elsewhere"
    relabeled_views = apply_execution_plan(
        relabeled, plan, source=QWEN38_FLASH_NEXT_ROOT
    )
    assert execution_fingerprint(relabeled, plan, relabeled_views) == baseline


def test_qwen38_flash_next_compound_math_drills_to_primitive_model_ir_nodes() -> None:
    bundle = compile_catalog(QWEN38_FLASH_NEXT_ROOT)
    model_views = bundle["model_ir"]["views"]

    # Revision 6 closes previously missing tensor-edge contracts. Its new
    # fingerprint is the stable baseline; operator_signature itself remains
    # presentation/semantic metadata and is excluded from the payload.
    # The canonical MTP generation graph now includes the explicit
    # proposal-update boundary retained from main, so the structural
    # execution fingerprint differs from the pre-merge enrichment branch.
    assert bundle["default_execution_variant"] == "exec_8f4c4d6b423803ed"

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
        node for node in model_views["moe"]["nodes"] if node["id"] == "routed_experts"
    )
    assert routed["drill"] == "moe_routed_expert"
    assert _node_ids(bundle["model_ir"], "moe_routed_expert")[1:-1] == [
        "gate_projection",
        "up_projection",
        "silu",
        "gated_product",
        "down_projection",
    ]


def test_fine_model_ir_nodes_share_measured_fusion_owner_without_double_counting() -> (
    None
):
    bundle = compile_catalog(QWEN38_FLASH_NEXT_ROOT)
    profile = bundle["profiles"]["qwen38_flash_next_tp4_cg_decode_bs1_8k1k"]
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

    owner = profile["data"]["hyperconnection_mix.mix"]["tp4_cg_decode_bs1_8k1k"]
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
    bundle = compile_catalog(QWEN38_FLASH_NEXT_ROOT)
    for profile_id in (
        "qwen38_flash_next_tp4_cg_decode_bs1_8k1k",
        "qwen38_flash_next_tp4_mtp_cg_decode_gbs001_8k1k",
    ):
        profile = bundle["profiles"][profile_id]
        group = profile["fusion_groups"]["fusion:hyperconnection.mix"]
        assert group["owner"] == "hyperconnection.mix"
        assert group["architecture_owner"] == "hyperconnection_mix.mix"


def test_profile_rejects_unreachable_fusion_architecture_owner() -> None:
    model = load_yaml(QWEN38_FLASH_NEXT_ROOT / "model_ir.yaml")
    plan_path = QWEN38_FLASH_NEXT_ROOT / "execution_paths" / "tp_only.yaml"
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
        QWEN38_FLASH_NEXT_ROOT
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


def test_qwen38_flash_next_qsa_indexer_drill_has_reconciled_binding_and_profile() -> (
    None
):
    bundle = compile_catalog(QWEN38_FLASH_NEXT_ROOT)
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

    profile = bundle["profiles"]["qwen38_flash_next_tp4_cg_decode_bs1_8k1k"]
    cell = profile["data"]["qsa_attention.indexer"]["tp4_cg_decode_bs1_8k1k"]
    assert cell["drill_view"] == "qsa_indexer"
    assert cell["drill_mapping_coverage_pct"] == 100.0
    assert cell["drill_metrics"]["raw_k_cache"]["status"] == "fused"
    assert cell["drill_metrics"]["compressed_k_cache"]["included_in"] == (
        "qsa_indexer.compress"
    )


def test_generation_mode_must_match_execution_contract() -> None:
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
    with pytest.raises(CatalogError, match="does not match Execution IR requirement"):
        compile_profile(
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
    bundle = compile_catalog(QWEN38_FLASH_NEXT_ROOT)
    profile = bundle["profiles"]["qwen38_flash_next_tp4_cg_decode_bs1_8k1k"]
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


def test_shared_event_coverage_requires_explicit_member_subsets() -> None:
    model = load_yaml(QWEN38_FLASH_NEXT_ROOT / "model_ir.yaml")
    plan_path = QWEN38_FLASH_NEXT_ROOT / "execution_paths" / "tp_only.yaml"
    plan = load_yaml(plan_path)
    views = apply_execution_plan(model, plan, source=plan_path)
    node_index = {
        f"{view_id}.{node['id']}": node
        for view_id, view in views.items()
        for node in view["nodes"]
    }
    profile = load_yaml(
        QWEN38_FLASH_NEXT_ROOT
        / "profiles"
        / "tp_only"
        / "sglang_f90a941aa"
        / "cg_decode_bs001_8k1k.yaml"
    )
    group_id = "coverage:linear_attention.qkvz_projection"
    profile["node_states"]["linear_attention.ba_projection"][
        "fusion_group_id"
    ] = group_id
    profile["fusion_groups"] = {
        group_id: {
            "owner": "linear_attention.qkvz_projection",
            "ir_nodes": [
                "linear_attention.qkvz_projection",
                "linear_attention.ba_projection",
            ],
            "timing_semantics": "shared_event_coverage",
            "evidence_scope": {
                "resolution": "profile_aggregate",
                "owner_event_ids": ["rank0:event0", "rank0:event1"],
                "member_event_ids": {
                    "linear_attention.ba_projection": ["rank0:event1"]
                },
            },
        }
    }

    compiled = compile_profile(
        profile,
        plan=plan,
        fingerprint=execution_fingerprint(model, plan, views),
        node_targets=set(node_index),
        node_index=node_index,
        views=views,
        source=Path("coverage-profile.yaml"),
    )
    member = compiled["data"]["linear_attention.ba_projection"][profile["variant_id"]]
    assert member["timing_role"] == "fused_member"
    assert member["fusion_timing_semantics"] == "shared_event_coverage"
    assert "active_gpu_ms" not in member

    profile["fusion_groups"][group_id]["evidence_scope"]["member_event_ids"][
        "linear_attention.ba_projection"
    ] = ["rank0:not-owned"]
    with pytest.raises(CatalogError, match="outside the owner's physical event set"):
        compile_profile(
            profile,
            plan=plan,
            fingerprint=execution_fingerprint(model, plan, views),
            node_targets=set(node_index),
            node_index=node_index,
            views=views,
            source=Path("coverage-profile.yaml"),
        )


def test_occurrence_partitioned_fusion_compiles_without_copied_timing() -> None:
    profile = {
        "profile_id": "occurrence-fusion",
        "model_id": "fixture-model",
        "variant_id": "tp1",
        "execution_path_id": "tp_only",
        "implementation_id": "fixture",
        "phase": "decode",
        "execution_parameters": {
            "tp_size": 1,
            "dp_size": 1,
            "cp_size": 1,
            "ep_size": 1,
        },
        "node_metrics": {
            "top.owner_a": {"ms_per_iter": 1.0},
            "top.owner_b": {"ms_per_iter": 2.0},
        },
        "node_states": {
            "top.member": {
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
    compiled = compile_profile(
        profile,
        plan={
            "selector": {
                "match": {
                    "generation.mode": {"equals": "autoregressive"},
                }
            },
            "parallelism_axes": {
                "tp_size": 1,
                "dp_size": 1,
                "cp_size": 1,
                "ep_size": 1,
            }
        },
        fingerprint="fixture-fingerprint",
        node_targets={"top.owner_a", "top.owner_b", "top.member"},
        source=Path("occurrence-profile.yaml"),
    )
    member = compiled["data"]["top.member"]["tp1"]
    assert member["timing_role"] == "occurrence_fused_member"
    assert member["shared_timing_owners"] == ["top.owner_a", "top.owner_b"]
    assert "ms_per_iter" not in member


def test_qwen35_collective_adapters_live_on_layer_boundaries() -> None:
    bundle = compile_catalog(QWEN35_ROOT)

    assert "tp_attention_output_collective" not in _node_ids(bundle, "gdn_attention")
    assert "tp_attention_output_collective" not in _node_ids(bundle, "full_attention")
    assert "tp_moe_output_collective" not in _node_ids(bundle, "moe_block")
    assert "tp_attention_output_collective" in _node_ids(bundle, "gdn_moe_block")
    assert "tp_moe_output_collective" in _node_ids(bundle, "gdn_moe_block")
    assert "tp_attention_output_collective" in _node_ids(
        bundle, "full_attention_moe_block"
    )
    assert "tp_moe_output_collective" in _node_ids(bundle, "full_attention_moe_block")


def test_compile_qwen38_flash_next_pure_tp_layout() -> None:
    bundle = compile_catalog(QWEN38_FLASH_NEXT_ROOT)

    assert bundle["default_implementation"] == "sglang_f90a941aa"
    assert "qwen38_flash_next_tp4_cg_decode_bs1_8k1k" in bundle["profiles"]
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


def test_qwen38_flash_next_topology_binding_inherits_common_source_mapping() -> None:
    bundle = compile_catalog(QWEN38_FLASH_NEXT_ROOT)
    binding = bundle["implementations"]["sglang_f90a941aa_dp_attention"]

    assert binding["extends"] == "sglang_f90a941aa"
    assert "linear_attention.qkvz_projection" in binding["node_bindings"]
    assert "linear_layer.dp_moe_input_gather" in binding["node_bindings"]
    assert "linear_layer.tp_attention_collective" not in binding["node_bindings"]


def test_qwen38_flash_next_qwen4_main_binding_explicitly_reuses_base_semantics() -> (
    None
):
    bundle = compile_catalog(QWEN38_FLASH_NEXT_ROOT)
    binding = bundle["implementations"][
        "sglang_qwen38_flash_next_32e9cb5_qsa_hardening_flashinfer_gdn"
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
    assert (
        "/blob/32e9cb5"
        in binding["node_bindings"]["mtp_generation.target_verify"]["code_links"][0][
            "url"
        ]
    )


def test_binding_validation_attestation_is_preserved_and_fingerprint_checked(
    tmp_path: Path,
) -> None:
    model_root = tmp_path / "qwen38_flash_next"
    import shutil

    shutil.copytree(QWEN38_FLASH_NEXT_ROOT, model_root)
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
        edge["from"] == "tp_attention_collective" and edge["to"] == "attn_hc_combine"
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
        edge["from"] == "tp_moe_output_collective" and edge["to"] == "mlp_hc_combine"
        for edge in edges
    )
    assert not any(
        edge["from"] == "moe" and edge["to"] == "mlp_hc_combine" for edge in edges
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


def test_profile_trace_time_is_preserved_and_requires_a_timezone() -> None:
    model_path = MODEL_ROOT / "model_ir.yaml"
    plan_path = MODEL_ROOT / "execution_paths" / "tp_only_eagle_mtp.yaml"
    model = load_yaml(model_path)
    plan = load_yaml(plan_path)
    views = apply_execution_plan(model, plan, source=plan_path)
    fingerprint = execution_fingerprint(model, plan, views)
    profile = load_yaml(
        MODEL_ROOT
        / "profiles"
        / "tp_only_eagle_mtp"
        / "sglang_25ee2b56_pr37500_tp4_eagle_mtp"
        / "cg_mtp_decode_gbs001_8k1k_bind_ba79b6e52262fede.yaml"
    )
    node_targets = {
        f"{view_id}.{node['id']}"
        for view_id, view in views.items()
        for node in view["nodes"]
    }
    compiled = compile_profile(
        profile,
        plan=plan,
        fingerprint=fingerprint,
        node_targets=node_targets,
        source=Path("profile.yaml"),
    )
    assert compiled["meta"]["trace_time"] == profile["trace_time"]

    profile["trace_time"]["timestamp"] = "2026-09-05T14:00:01"
    with pytest.raises(CatalogError, match="explicit timezone"):
        compile_profile(
            profile,
            plan=plan,
            fingerprint=fingerprint,
            node_targets=node_targets,
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
        transform
        for transform in plan["transforms"]
        if transform["op"] == "insert_after"
    )
    del inserted["node"]["execution"]["payload"]

    with pytest.raises(CatalogError, match="requires execution.payload"):
        apply_execution_plan(model, plan, source=plan_path)


def test_schema_documents_are_valid_json() -> None:
    schema_root = REPO_ROOT / "schema" / "v2"
    for path in schema_root.glob("*.schema.json"):
        document = json.loads(path.read_text())
        assert document["$schema"].endswith("2020-12/schema")
