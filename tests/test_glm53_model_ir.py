from __future__ import annotations

import json
import copy
from pathlib import Path

import pytest
import yaml

from llm_arch_v2 import compile_catalog
from llm_arch_v2.compiler import CatalogError, apply_execution_plan


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = REPO_ROOT / "catalog" / "glm53_flash"


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def test_glm53_stable_ir_has_exact_architecture_facts() -> None:
    model = load_yaml(MODEL_ROOT / "model_ir.yaml")
    facts = model["facts"]
    assert facts["checkpoint_revision"] == "3f1971b7b5f7a528c9c4ef6212c8785298a8c24a"
    assert facts["target_layers"] == 45
    assert len(facts["linear_attention_layers"]) == 34
    assert facts["sparse_attention_layers"] == list(range(3, 44, 4))
    assert facts["mhc_streams"] == 4
    assert facts["dsa_indexer_count"] == 11
    assert facts["routed_experts"] == 288
    assert facts["vision_depth"] == 24
    assert facts["nextn_layers"] == 1


def test_glm53_all_edges_are_tensor_and_state_explicit() -> None:
    model = load_yaml(MODEL_ROOT / "model_ir.yaml")
    required = {"identity", "shape", "layout", "dtype", "state"}
    for view_id, view in model["views"].items():
        nodes = {node["id"] for node in view["nodes"]}
        for edge in view["edges"]:
            assert edge["from"] in nodes, (view_id, edge)
            assert edge["to"] in nodes, (view_id, edge)
            assert required <= set(edge), (view_id, edge)
            assert all(str(edge[field]) for field in required), (view_id, edge)


def test_glm53_drills_have_explicit_boundary_contracts() -> None:
    model = load_yaml(MODEL_ROOT / "model_ir.yaml")
    drills = {
        f"{view_id}.{node['id']}": node["drill"]
        for view_id, view in model["views"].items()
        for node in view["nodes"]
        if "drill" in node
    }
    contracts = {entry["parent_node"]: entry for entry in model["boundary_contracts"]}
    assert drills.keys() == contracts.keys()
    for target, child in drills.items():
        assert contracts[target]["child_view"] == child
        assert contracts[target]["boundary_mode"] in {
            "exact_node",
            "exact_lifecycle",
            "external_entry",
        }
        assert contracts[target]["input_shape"]
        assert contracts[target]["output_shape"]
        if contracts[target]["boundary_mode"] == "exact_lifecycle":
            assert target in contracts[target]["scope_nodes"]
            assert len(contracts[target]["scope_nodes"]) >= 2
            assert contracts[target]["handoff_shape"]


def test_glm53_compiled_semantics_are_edge_derived_and_equation_complete() -> None:
    raw = load_yaml(MODEL_ROOT / "model_ir.yaml")
    bundle = compile_catalog(MODEL_ROOT)
    compiled_views = bundle["model_ir"]["views"]
    operations = raw["semantic_contract"]["operations"]

    for view_id, raw_view in raw["views"].items():
        incoming = {node["id"]: [] for node in raw_view["nodes"]}
        outgoing = {node["id"]: [] for node in raw_view["nodes"]}
        for edge in raw_view["edges"]:
            incoming[edge["to"]].append(
                (edge["identity"], edge["shape"], edge["layout"], edge["dtype"], edge["state"], edge["from"])
            )
            outgoing[edge["from"]].append(
                (edge["identity"], edge["shape"], edge["layout"], edge["dtype"], edge["state"], edge["to"])
            )

        compiled_nodes = {node["id"]: node for node in compiled_views[view_id]["nodes"]}
        for raw_node in raw_view["nodes"]:
            semantics = compiled_nodes[raw_node["id"]]["semantics"]
            assert semantics["semantic_op"] == raw_node["semantic_op"]
            assert semantics["equation"]
            compiled_inputs = {
                (item["name"], item["shape"], item["layout"], item["dtype"], item["state"], item["source"])
                for item in semantics["inputs"]
            }
            compiled_outputs = {
                (item["name"], item["shape"], item["layout"], item["dtype"], item["state"], item["target"])
                for item in semantics["outputs"]
            }
            assert compiled_inputs == set(incoming[raw_node["id"]])
            assert compiled_outputs == set(outgoing[raw_node["id"]])

            assert raw_node["semantic_op"] in operations
            assert semantics["equation"] == operations[raw_node["semantic_op"]]["equation"]
            assert "None" not in semantics["equation"]


def test_glm53_kda_beta_contract_is_per_head_and_broadcast_in_recurrence() -> None:
    bundle = compile_catalog(MODEL_ROOT)
    nodes = {
        node["id"]: node
        for node in bundle["model_ir"]["views"]["linear_attention"]["nodes"]
    }
    beta = nodes["beta_projection"]["semantics"]
    assert beta["outputs"] == [
        {
            "name": "beta",
            "shape": "[B,T,N]",
            "layout": "batch_sequence_head",
            "dtype": "model_activation_dtype",
            "state": "ephemeral_coefficient",
            "target": "recurrent_update",
        }
    ]
    recurrence = nodes["recurrent_update"]["semantics"]
    assert "beta has shape [B,T,N] and broadcasts over Dkda" in recurrence["invariants"]


def test_glm53_strict_semantics_fail_closed_when_any_node_equation_is_missing() -> None:
    model = copy.deepcopy(load_yaml(MODEL_ROOT / "model_ir.yaml"))
    plan_path = MODEL_ROOT / "execution_paths" / "tp8.yaml"
    plan = load_yaml(plan_path)
    del model["semantic_contract"]["operations"]["model.embedding"]
    with pytest.raises(CatalogError, match="explicit operation contract"):
        apply_execution_plan(model, plan, source=plan_path)

    model = copy.deepcopy(load_yaml(MODEL_ROOT / "model_ir.yaml"))
    del model["semantic_contract"]["operations"]["linear_attention.conv_state"]
    with pytest.raises(CatalogError, match="explicit operation contract"):
        apply_execution_plan(model, plan, source=plan_path)


def test_viewer_renders_the_compiled_semantic_contract() -> None:
    viewer = (REPO_ROOT / "docs" / "viewer.html").read_text()
    for heading in ("Semantics", "Inputs", "Transition / Equation", "Outputs"):
        assert heading in viewer
    assert "semantics.inputs" in viewer
    assert "semantics.outputs" in viewer
    assert "semantics.equation" in viewer
    assert "timing owner" in viewer
    assert "fusionScopeDescription" in viewer
    assert "fused into" in viewer


def test_glm53_production_profiles_close_all_required_node_states() -> None:
    bundle = compile_catalog(MODEL_ROOT)
    statuses = {
        cell.get("status")
        for profile in bundle["profiles"].values()
        for variants in profile["data"].values()
        for cell in variants.values()
        if cell.get("status")
    }
    assert "mapping_incomplete" not in statuses
    assert "unmapped" not in statuses
    assert statuses <= {"structural", "not_selected", "fused", "state"}

    for profile_path in (MODEL_ROOT / "profiles" / "tp8").glob("*/*.yaml"):
        profile = load_yaml(profile_path)
        assert not [
            target
            for target, state in profile["node_states"].items()
            if state["status"] == "mapping_incomplete"
        ], profile_path
        assert profile["evidence"]["mapped_kernel_duration_ratio"] >= 0.89
        diagnostics = profile["evidence"].get("attribution_diagnostics")
        if diagnostics:
            assert diagnostics["anchor_count"] == 90


def test_glm53_executable_decoder_modules_have_union_rollups() -> None:
    bundle = compile_catalog(MODEL_ROOT)
    executable = {
        "top.decoder_stack",
        "decoder_stack.linear_attention",
        "decoder_stack.dsa_attention",
        "decoder_stack.dense_mlp",
        "decoder_stack.moe",
    }
    structural = {
        "decoder_stack.schedule",
        "decoder_stack.attention_schedule",
        "decoder_stack.feed_forward_schedule",
    }
    shared_mhc = {
        "decoder_stack.attn_mhc_pre",
        "decoder_stack.attn_mhc_combine",
        "decoder_stack.ffn_mhc_pre",
        "decoder_stack.ffn_mhc_combine",
    }

    for profile_id, profile in bundle["profiles"].items():
        variant = profile["meta"]["variant_id"]
        for target in executable:
            cell = profile["data"][target][variant]
            assert cell["attribution_status"] == "inclusive_rollup", (
                profile_id,
                target,
            )
            assert cell["active_gpu_ms"] > 0, (profile_id, target)
            assert cell["gpu_residency_ms"] >= cell["active_gpu_ms"]
        for target in structural:
            cell = profile["data"][target][variant]
            assert cell["status"] == "structural", (profile_id, target)
            assert "ms_per_iter" not in cell
        for target in shared_mhc:
            cell = profile["data"][target][variant]
            assert cell["status"] == "fused", (profile_id, target)
            assert cell["timing_role"] == "fused_member"
            assert "ms_per_iter" not in cell
            assert "active_gpu_ms" not in cell
            assert cell["shared_timing_owner"] == cell["included_in"]


def test_glm53_tp_collectives_roll_up_to_decoder_not_local_model_modules() -> None:
    bundle = compile_catalog(MODEL_ROOT)
    expected = {
        "linear_attention.tp_kda_output_collective": "decoder_stack.linear_attention",
        "dsa_attention.tp_dsa_output_collective": "decoder_stack.dsa_attention",
        "dense_mlp.tp_dense_mlp_output_collective": "decoder_stack.dense_mlp",
        "moe.tp_moe_output_collective": "decoder_stack.moe",
    }

    for profile_id, profile in bundle["profiles"].items():
        if not profile_id.startswith("glm53_flash_tp8_sglang_"):
            continue
        variant = profile["meta"]["variant_id"]
        decoder_sources = set(
            profile["data"]["top.decoder_stack"][variant]["rollup_sources"]
        )
        for collective, local_module in expected.items():
            assert collective in decoder_sources, (profile_id, collective)
            local_sources = set(
                profile["data"][local_module][variant]["rollup_sources"]
            )
            assert collective not in local_sources, (profile_id, local_module)


def test_glm53_direct_leaf_metrics_are_not_contaminated_by_shared_fusion_sets() -> None:
    bundle = compile_catalog(MODEL_ROOT)
    for profile_id, profile in bundle["profiles"].items():
        variant = profile["meta"]["variant_id"]
        for target, variants in profile["data"].items():
            cell = variants.get(variant) or {}
            if cell.get("attribution_status") != "measured_direct":
                continue
            if cell.get("metric_kind") != "exclusive_leaf":
                continue
            assert cell["active_gpu_ms"] <= cell["gpu_residency_ms"] + 1e-9, (
                profile_id,
                target,
            )
            assert cell["gpu_residency_ms"] == cell["ms_per_iter"], (
                profile_id,
                target,
            )


def test_vllm_fused_states_have_one_scoped_timing_owner() -> None:
    bundle = compile_catalog(MODEL_ROOT)
    profile = bundle["profiles"]["glm53_flash_tp8_vllm_cg_prefill_bs1_8k1k"]
    variant = profile["meta"]["variant_id"]

    for target, variants in profile["data"].items():
        cell = variants.get(variant) or {}
        if cell.get("status") != "fused":
            continue
        group_id = cell.get("fusion_group_id")
        assert group_id, target
        group = profile["fusion_groups"][group_id]
        assert group["owner"] == cell["included_in"], target
        assert target in group["ir_nodes"], target

    residual = profile["fusion_groups"]["vllm_graph_mhc_tp_boundary"]
    assert "decoder_stack.attn_mhc_combine" in residual["ir_nodes"]
    assert "mhc_transform.post_weights" in residual["ir_nodes"]
    assert residual["timing_semantics"] == "shared_event_set"
    assert residual["evidence_scope"]["resolution"] == "profile_aggregate"

    pre = profile["fusion_groups"]["vllm_graph_mhc_pre_profile_aggregate"]
    assert "decoder_stack.attn_mhc_pre" in pre["ir_nodes"]
    assert "mhc_transform.flatten_norm" in pre["ir_nodes"]


def test_glm53_model_ir_is_framework_and_collective_independent() -> None:
    model = load_yaml(MODEL_ROOT / "model_ir.yaml")
    text = json.dumps(model, sort_keys=True).lower()
    for forbidden in (
        "sglang",
        "vllm",
        "triton",
        "cuda",
        "nccl",
        "all_reduce",
        "all_gather",
        "reduce_scatter",
    ):
        assert forbidden not in text


def test_glm53_tp8_plan_is_exact_and_fail_closed() -> None:
    model = load_yaml(MODEL_ROOT / "model_ir.yaml")
    plan = load_yaml(MODEL_ROOT / "execution_paths" / "tp8.yaml")
    nodes = {
        f"{view_id}.{node['id']}"
        for view_id, view in model["views"].items()
        for node in view["nodes"]
    }
    constraints = plan["constraints"]
    assert constraints["tp_size"] == 8
    assert constraints["dp_size"] == constraints["cp_size"] == constraints["ep_size"] == 1
    assert constraints["attention_dp"] is False
    assert (
        constraints["validation_state"]
        == "validated_by_sglang_and_vllm_graph_off_reconciliation"
    )
    assert constraints["mtp_enabled"] is False
    assert constraints["hicache_enabled"] is False
    assert constraints["kv_offload_enabled"] is False
    assert constraints["prefix_cache_enabled"] is True
    inserted = []
    for transform in plan["transforms"]:
        target = transform.get("target", transform.get("after"))
        assert target in nodes
        if transform["op"] == "insert_after":
            inserted.append(transform["node"])
    assert {node["execution"]["collective"] for node in inserted} == {"all_reduce", "all_gather"}
    for node in inserted:
        assert node["node_kind"] == "communication"
        assert node["boundary_role"] == "module_boundary"
        assert all(node["execution"][field] for field in ("placement", "collective", "parallelism", "payload", "result"))


def test_glm53_validated_catalog_compile_is_deterministic() -> None:
    first = compile_catalog(MODEL_ROOT)
    second = compile_catalog(MODEL_ROOT)
    assert first == second
    assert first["meta"]["view_count"] == 10
    assert first["meta"]["execution_variant_count"] == 1
    assert first["meta"]["implementation_count"] == 2
    assert first["meta"]["profile_count"] == 6


def test_glm53_bindings_cover_model_and_execution_nodes() -> None:
    bundle = compile_catalog(MODEL_ROOT)
    required_nodes = {
        f"{view_id}.{node['id']}"
        for view_id, view in bundle["views"].items()
        for node in view["nodes"]
    }
    assert set(bundle["implementations"]) == {
        "sglang_f609d677b_mixed_glm53_tp8",
        "vllm_487ecf187_native_tp8",
    }
    expected_states = {
        "sglang_f609d677b_mixed_glm53_tp8": ("validated", "pass"),
        "vllm_487ecf187_native_tp8": ("validated", "pass"),
    }
    for implementation_id, binding in bundle["implementations"].items():
        binding_state, execution_state = expected_states[implementation_id]
        assert binding["binding_status"] == binding_state
        assert binding["execution_validation"]["status"] == execution_state
        assert set(binding["node_bindings"]) == required_nodes
        for target, node_binding in binding["node_bindings"].items():
            assert node_binding["symbols"], target
            assert node_binding["code_links"], target
