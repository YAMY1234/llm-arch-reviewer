from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from llm_arch_v2 import compile_catalog
from llm_arch_v2.compiler import CatalogError, apply_execution_plan
from models.qwen35.build.build_qwen35_production_profiles import (
    profile_acceptance_gate,
    rank_collective_duration_gate,
    wall_trace_contract_gate,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = REPO_ROOT / "catalog" / "qwen35"


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def test_qwen35_stable_ir_has_exact_checkpoint_architecture() -> None:
    facts = load_yaml(MODEL_ROOT / "model_ir.yaml")["facts"]
    assert facts["checkpoint_revision"] == "8f590eae8f10bf55d9a46f79ea0280bde435c9f8"
    assert facts["checkpoint_config_sha256"] == "9408a9e559cc2f05f0b357738213666353e6651160ce8ff477b1c26982bc4f63"
    assert facts["architecture"] == "Qwen3_5MoeForConditionalGeneration"
    assert facts["text_layers"] == 60
    assert facts["gdn_layers"] == 45
    assert facts["full_attention_layers"] == 15
    assert facts["full_attention_layer_indices"] == list(range(3, 60, 4))
    assert facts["routed_experts"] == 512
    assert facts["experts_per_token"] == 10
    assert facts["vision_depth"] == 27
    assert facts["mtp_hidden_layers"] == 1


def test_qwen35_all_edges_are_tensor_and_state_explicit() -> None:
    model = load_yaml(MODEL_ROOT / "model_ir.yaml")
    required = {"identity", "shape", "layout", "dtype", "state"}
    for view_id, view in model["views"].items():
        nodes = {node["id"] for node in view["nodes"]}
        for edge in view["edges"]:
            assert edge["from"] in nodes, (view_id, edge)
            assert edge["to"] in nodes, (view_id, edge)
            assert required <= set(edge), (view_id, edge)
            assert all(str(edge[field]) for field in required), (view_id, edge)


def test_qwen35_drills_have_explicit_boundary_contracts() -> None:
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
        contract = contracts[target]
        assert contract["child_view"] == child
        assert contract["input_shape"]
        assert contract["output_shape"]
        assert contract["boundary_mode"] in {"exact_node", "exact_lifecycle", "external_entry"}
        if contract["boundary_mode"] == "exact_lifecycle":
            assert target in contract["scope_nodes"]
            assert len(contract["scope_nodes"]) >= 2
            assert contract["handoff_shape"]


def test_qwen35_compiled_semantics_are_edge_derived_and_equation_complete() -> None:
    raw = load_yaml(MODEL_ROOT / "model_ir.yaml")
    compiled_views = compile_catalog(MODEL_ROOT)["model_ir"]["views"]
    operations = raw["semantic_contract"]["operations"]
    for view_id, raw_view in raw["views"].items():
        incoming = {node["id"]: [] for node in raw_view["nodes"]}
        outgoing = {node["id"]: [] for node in raw_view["nodes"]}
        for edge in raw_view["edges"]:
            incoming[edge["to"]].append((edge["identity"], edge["shape"], edge["layout"], edge["dtype"], edge["state"], edge["from"]))
            outgoing[edge["from"]].append((edge["identity"], edge["shape"], edge["layout"], edge["dtype"], edge["state"], edge["to"]))
        compiled_nodes = {node["id"]: node for node in compiled_views[view_id]["nodes"]}
        for raw_node in raw_view["nodes"]:
            semantics = compiled_nodes[raw_node["id"]]["semantics"]
            assert semantics["semantic_op"] == raw_node["semantic_op"]
            assert semantics["equation"] == operations[raw_node["semantic_op"]]["equation"]
            assert "None" not in semantics["equation"]
            assert {
                (item["name"], item["shape"], item["layout"], item["dtype"], item["state"], item["source"])
                for item in semantics["inputs"]
            } == set(incoming[raw_node["id"]])
            assert {
                (item["name"], item["shape"], item["layout"], item["dtype"], item["state"], item["target"])
                for item in semantics["outputs"]
            } == set(outgoing[raw_node["id"]])


def test_qwen35_model_ir_is_framework_and_collective_independent() -> None:
    text = json.dumps(load_yaml(MODEL_ROOT / "model_ir.yaml"), sort_keys=True).lower()
    for forbidden in ("sglang", "vllm", "triton", "cuda", "nccl", "all_reduce", "all_gather", "reduce_scatter"):
        assert forbidden not in text


def test_qwen35_strict_semantics_fail_closed_on_missing_equation() -> None:
    model = copy.deepcopy(load_yaml(MODEL_ROOT / "model_ir.yaml"))
    plan_path = MODEL_ROOT / "execution_paths" / "tp8.yaml"
    plan = load_yaml(plan_path)
    del model["semantic_contract"]["operations"]["qwen3_5.gdn.recurrence"]
    with pytest.raises(CatalogError, match="explicit operation contract"):
        apply_execution_plan(model, plan, source=plan_path)


def test_qwen35_tp8_plan_is_exact_and_does_not_relabel_optional_paths() -> None:
    plan = load_yaml(MODEL_ROOT / "execution_paths" / "tp8.yaml")
    constraints = plan["constraints"]
    assert constraints | {
        "tp_size": 8,
        "dp_size": 1,
        "cp_size": 1,
        "ep_size": 1,
        "pp_size": 1,
        "attention_dp": False,
        "serving_mode": "aggregated",
        "modality": "text_only",
        "mtp_enabled": False,
        "prefix_cache_enabled": False,
    } == constraints
    assert any(
        transform.get("target") == "top.vision_frontend"
        and transform["set"]["execution"]["selection"] == "structurally_retained_not_executed_in_text_only_contract"
        for transform in plan["transforms"]
    )
    assert any(
        transform.get("target") == "top.generation_controller"
        and transform["set"]["execution"]["selection"] == "mtp_disabled_for_stage1_contract"
        for transform in plan["transforms"]
    )


def qwen35_profile_paths() -> list[Path]:
    return sorted((MODEL_ROOT / "profiles" / "tp8").glob("*/*.yaml"))


def test_qwen35_cross_framework_matrix_is_fail_closed_as_unsupported() -> None:
    manifest = load_yaml(MODEL_ROOT / "unsupported_profiles.yaml")
    profiles = manifest["profiles"]
    assert qwen35_profile_paths() == []
    assert manifest["expected_profile_count"] == 10
    assert manifest["accepted_profile_count"] == 0
    assert manifest["unsupported_profile_count"] == 10
    assert {
        (profile["framework"], profile["phase"], profile["global_batch_size"])
        for profile in profiles
    } == {
        (framework, phase, batch)
        for framework in ("sglang", "vllm")
        for phase, batch in (("prefill", 1), ("decode", 1), ("decode", 16), ("decode", 64), ("decode", 256))
    }
    for profile in profiles:
        batch = profile["global_batch_size"]
        assert profile["state"] == "unsupported"
        assert profile["workload"]["isl"] == 8192
        assert profile["workload"]["osl"] == 1024
        assert profile["workload"]["warmup_requests"] == 3 * batch
        assert profile["workload"]["formal_requests"] == batch
        assert "semantic_reconciliation_incomplete" in profile["reason_codes"]
        assert "fusion_reconciliation_incomplete" in profile["reason_codes"]
        assert profile["typed_unresolved_semantic_event_count"] > 0
        assert set(profile["rank_typed_unresolved_semantic_event_count"]) == {
            str(rank) for rank in range(8)
        }
        assert all(
            count > 0
            for count in profile["rank_typed_unresolved_semantic_event_count"].values()
        )
        assert profile["partial_fusion_node_count"] > 0
        assert profile["incomplete_fusion_owner_closure_node_count"] > 0
        assert profile["full_fusion_groups_all_closed"] is True
        assert profile["false_fill_qk_rope_fusion_published"] is False
        selected_graph = profile["selected_forward_cuda_graph"]
        assert selected_graph["used_graph_path"] is (selected_graph["graph_id_count"] > 0)
        assert selected_graph["model_kernel_count"] == (
            selected_graph["graph_kernel_count"] + selected_graph["non_graph_kernel_count"]
        )
        assert selected_graph["all_tp_ranks_consistent"] is True
        assert selected_graph["used_graph_path"] is (selected_graph["graph_kernel_count"] > 0)
        evidence_basis = selected_graph["evidence_basis"]
        if selected_graph["used_graph_path"]:
            assert f"{selected_graph['graph_id_count']} distinct nonzero raw-trace graph IDs" in evidence_basis
            assert f"{selected_graph['graph_kernel_count']} model-bearing kernels" in evidence_basis
        else:
            assert evidence_basis == (
                f"zero nonzero raw-trace graph IDs across {selected_graph['model_kernel_count']} "
                "model-bearing kernels in the selected formal forward"
            )
        if profile["phase"] == "decode":
            assert selected_graph["used_graph_path"] is True
            assert selected_graph["replay_state"] == "mixed_graph_and_eager"
        elif profile["framework"] == "sglang":
            assert selected_graph["used_graph_path"] is True
            assert selected_graph["replay_state"] == "mixed_graph_and_eager"
        else:
            assert selected_graph["used_graph_path"] is False
            assert selected_graph["replay_state"] == "no_cuda_graph_replay"
            assert selected_graph["graph_kernel_count"] == 0
        assert len(profile["all_rank_eager_mapping_sha256"]) == 8
        assert len(profile["all_rank_eager_raw_manifest_sha256"]) == 8
        assert len(profile["all_rank_production_trace_sha256"]) == 8
        phase_contract = profile["all_rank_eager_phase_contract"]
        assert set(phase_contract) == {str(rank) for rank in range(8)}
        expected_source_phase = (
            f"vllm_{profile['phase']}"
            if profile["framework"] == "vllm"
            else f"forward_{'extend' if profile['phase'] == 'prefill' else 'decode'}"
        )
        for rank, contract in phase_contract.items():
            assert contract["source_phase"] == expected_source_phase, rank
            assert contract["selected_forward_kernel_count"] > 0
            assert contract["selected_forward_kernel_duration_us"] > 0
            assert len(contract["raw_trace_sha256"]) == 64
            assert len(contract["raw_manifest_sha256"]) == 64
            assert len(contract["selected_forward_events_sha256"]) == 64
        if profile["framework"] == "vllm":
            expected_count, expected_duration = (
                (4133, 843078.284)
                if profile["phase"] == "prefill"
                else (3718, 741788.696)
            )
            assert phase_contract["0"]["selected_forward_kernel_count"] == expected_count
            assert phase_contract["0"]["selected_forward_kernel_duration_us"] == pytest.approx(
                expected_duration, abs=1e-3
            )
        assert len(profile["validation_sha256"]) == 64
        assert len(profile["rejected_profile_sha256"]) == 64
        assert len(profile["rejected_timeline_sha256"]) == 64

    sglang_prefill = next(
        profile
        for profile in profiles
        if profile["framework"] == "sglang" and profile["phase"] == "prefill"
    )
    gate = sglang_prefill["wall_trace_contract_gate"]
    assert gate["state"] == "failed"
    assert gate["same_isolated_forward_proven"] is False
    assert gate["serving_wall_ms"] == 4111.995663
    assert gate["instrumented_active_gpu_ms"] == 89.595218
    assert gate["instrumented_kernel_envelope_ms"] == 93.450314
    assert "wall_trace_contract_mismatch" in sglang_prefill["reason_codes"]


def test_qwen35_acceptance_gate_rejects_each_incomplete_contract() -> None:
    rank_diagnostics = {
        "0": {"semantic_reconciliation": {"typed_unresolved_event_count": 1}}
    }
    gate = profile_acceptance_gate(
        rank_diagnostics=rank_diagnostics,
        typed_unresolved_event_count=1,
        node_states={
            "full_attention.partial_rope": {
                "status": "partially_fused",
                "all_owner_events_same_rank_closed": False,
            }
        },
        fusion_groups={
            "bad": {
                "evidence_scope": {
                    "member_event_sets_equal_owner": True,
                    "all_owner_events_same_rank_closed": False,
                }
            }
        },
        wall_trace_gate={"state": "failed"},
    )
    assert gate["state"] == "unsupported"
    assert {reason["code"] for reason in gate["reasons"]} == {
        "semantic_reconciliation_incomplete",
        "fusion_reconciliation_incomplete",
        "invalid_complete_fusion_claim",
        "wall_trace_contract_mismatch",
    }

    accepted = profile_acceptance_gate(
        rank_diagnostics={
            str(rank): {
                "semantic_reconciliation": {"typed_unresolved_event_count": 0}
            }
            for rank in range(8)
        },
        typed_unresolved_event_count=0,
        node_states={},
        fusion_groups={
            "closed": {
                "evidence_scope": {
                    "member_event_sets_equal_owner": True,
                    "all_owner_events_same_rank_closed": True,
                }
            }
        },
        wall_trace_gate={"state": "passed"},
    )
    assert accepted == {
        "state": "accepted",
        "fail_closed": True,
        "reason_count": 0,
        "reasons": [],
    }


def test_qwen35_wall_trace_gate_rejects_unexplained_first_prefill_interval() -> None:
    gate = wall_trace_contract_gate(
        {"framework": "sglang", "phase": "prefill"},
        {"profile_start_step": 0},
        serving_wall_ms=4111.995663,
        active_gpu_ms=89.595218,
        kernel_envelope_ms=93.450314,
    )
    assert gate["state"] == "failed"
    assert gate["same_isolated_forward_proven"] is False
    assert gate["wall_to_envelope_ratio"] == pytest.approx(44.001946)


def test_qwen35_bindings_are_commit_specific_validated_and_complete() -> None:
    bundle = compile_catalog(MODEL_ROOT)
    assert bundle["profiles"] == {}
    variant = next(iter(bundle["execution_variants"].values()))
    target_count = sum(len(view["nodes"]) for view in variant["views"].values())
    assert target_count == 203
    assert len(bundle["implementations"]) == 2
    for implementation in bundle["implementations"].values():
        assert implementation["binding_status"] == "validated"
        assert implementation["source_lock_status"] == "runtime_verified"
        assert implementation["execution_validation"]["status"] == "pass"
        assert implementation["execution_validation"]["cuda_graph_enabled"] is False
        assert implementation["execution_validation"]["execution_fingerprint"] == "exec_50bb583c3a3d0557"
        assert len(implementation["node_bindings"]) == target_count


def test_qwen35_public_artifacts_have_no_unexplained_mapping_placeholders() -> None:
    forbidden = ("unmapped", "mapping incomplete", "mapping_incomplete", "generic fused implementation shard")
    for path in [
        MODEL_ROOT / "model_ir.yaml",
        MODEL_ROOT / "unsupported_profiles.yaml",
        *qwen35_profile_paths(),
    ]:
        text = path.read_text().lower()
        assert not [token for token in forbidden if token in text], path


def test_qwen35_rank_collective_duration_gate_rejects_one_rank_wait_outlier() -> None:
    kernel = (
        "void flashinfer::trtllm_mnnvl_allreduce::"
        "oneshotAllreduceFusionKernel<__nv_bfloat16>()"
    )
    rank_rows = {
        rank: [
            {
                "kernel_name": kernel,
                "ts_us": float(index * 20),
                "dur_us": 2222.7 if rank == 0 and index == 0 else 10.0,
                "node": "top.tp_embedding_output_collective",
                "occurrence_id": f"collective_{index:03d}",
            }
            for index in range(121)
        ]
        for rank in range(8)
    }
    with pytest.raises(ValueError, match="rank collective-duration/outlier gate failed"):
        rank_collective_duration_gate(
            rank_rows, job="synthetic-skew", serving_wall_ms=5.721765
        )
