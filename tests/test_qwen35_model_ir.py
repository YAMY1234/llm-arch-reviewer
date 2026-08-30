from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path

import pytest
import yaml

from llm_arch_v2 import compile_catalog
from llm_arch_v2.compiler import CatalogError, apply_execution_plan
from models.qwen35.build.build_qwen35_production_profiles import (
    rank_collective_duration_gate,
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


def test_qwen35_complete_cross_framework_profile_matrix() -> None:
    profiles = [load_yaml(path) for path in qwen35_profile_paths()]
    assert len(profiles) == 10
    assert {profile["implementation_id"].split("_", 1)[0] for profile in profiles} == {
        "sglang",
        "vllm",
    }
    assert {
        (profile["implementation_id"].split("_", 1)[0], profile["phase"], profile["workload"]["batch_size"])
        for profile in profiles
    } == {
        (framework, phase, batch)
        for framework in ("sglang", "vllm")
        for phase, batch in (("prefill", 1), ("decode", 1), ("decode", 16), ("decode", 64), ("decode", 256))
    }
    for profile in profiles:
        batch = profile["workload"]["batch_size"]
        assert profile["execution_parameters"] == {
            "tp_size": 8,
            "dp_size": 1,
            "cp_size": 1,
            "ep_size": 1,
            "pp_size": 1,
        }
        assert profile["workload"]["isl"] == 8192
        assert profile["workload"]["osl"] == 1024
        assert profile["workload"]["warmup_requests"] == 3 * batch
        assert profile["workload"]["formal_requests"] == batch
        assert profile["profiler"]["formal_window_count"] == 1
        profiler = profile["profiler"]
        selected_graph = profiler["selected_forward_cuda_graph"]
        assert profiler["cuda_graph_enabled"] is selected_graph["used_graph_path"]
        assert profiler["cuda_graph_enabled"] is (selected_graph["graph_id_count"] > 0)
        assert selected_graph["model_kernel_count"] == (
            selected_graph["graph_kernel_count"] + selected_graph["non_graph_kernel_count"]
        )
        assert selected_graph["all_tp_ranks_consistent"] is True
        assert selected_graph["used_graph_path"] is (selected_graph["graph_kernel_count"] > 0)
        assert profiler["server_cuda_graph_config"]["enabled"] is True
        assert profiler["server_cuda_graph_config"]["evidence_files"]
        semantics = profiler["cuda_graph_enabled_semantics"]
        evidence_basis = selected_graph["evidence_basis"]
        if selected_graph["used_graph_path"]:
            assert "selected formal forward used a CUDA Graph path" in semantics
            assert "did not use CUDA Graph replay" not in semantics
            assert f"{selected_graph['graph_id_count']} distinct nonzero raw-trace graph IDs" in evidence_basis
            assert f"{selected_graph['graph_kernel_count']} model-bearing kernels" in evidence_basis
        else:
            assert "selected formal forward did not use CUDA Graph replay" in semantics
            assert "zero nonzero raw-trace graph IDs" in semantics
            assert "selected formal forward used a CUDA Graph path" not in semantics
            assert evidence_basis == (
                f"zero nonzero raw-trace graph IDs across {selected_graph['model_kernel_count']} "
                "model-bearing kernels in the selected formal forward"
            )
        if profile["phase"] == "decode":
            assert selected_graph["used_graph_path"] is True
            assert selected_graph["replay_state"] == "mixed_graph_and_eager"
        elif profile["implementation_id"].startswith("sglang_"):
            assert profiler["server_cuda_graph_config"]["mode"] == "breakable_prefill"
            assert selected_graph["used_graph_path"] is True
            assert selected_graph["replay_state"] == "mixed_graph_and_eager"
        else:
            assert profiler["server_cuda_graph_config"]["mode"] == "FULL_AND_PIECEWISE"
            assert selected_graph["used_graph_path"] is False
            assert selected_graph["replay_state"] == "no_cuda_graph_replay"
            assert selected_graph["graph_kernel_count"] == 0
        assert profile["profiler"]["selected_runtime_coordinate"]
        timing = profile["evidence"]["timing"]
        assert timing["elapsed_ms"] == timing["serving_wall_ms"]
        assert timing["wall_authority"] == "selected profiler-off production baseline forward"
        assert timing["layout_active_residency_authority"].startswith(
            "instrumented production trace"
        )
        assert timing["kernel_envelope_ms"] > 0
        assert timing["instrumented_trace_overhead_ms"] >= 0
        assert profile["evidence"]["unclassified_kernel_count"] == 0
        assert profile["evidence"]["typed_unresolved_semantic_event_count"] >= 0
        assert len(profile["evidence"]["all_rank_eager_mapping_sha256"]) == 8
        assert len(profile["evidence"]["all_rank_eager_raw_manifest_sha256"]) == 8
        phase_contract = profile["evidence"]["all_rank_eager_phase_contract"]
        assert set(phase_contract) == {str(rank) for rank in range(8)}
        expected_source_phase = (
            f"vllm_{profile['phase']}"
            if profile["implementation_id"].startswith("vllm_")
            else f"forward_{'extend' if profile['phase'] == 'prefill' else 'decode'}"
        )
        for rank, contract in phase_contract.items():
            assert contract["source_phase"] == expected_source_phase, rank
            assert contract["selected_forward_kernel_count"] > 0
            assert contract["selected_forward_kernel_duration_us"] > 0
            assert len(contract["raw_trace_sha256"]) == 64
            assert len(contract["raw_manifest_sha256"]) == 64
            assert len(contract["selected_forward_events_sha256"]) == 64
        if profile["implementation_id"].startswith("vllm_"):
            expected_count, expected_duration = (
                (4133, 843078.284)
                if profile["phase"] == "prefill"
                else (3718, 741788.696)
            )
            assert phase_contract["0"]["selected_forward_kernel_count"] == expected_count
            assert phase_contract["0"]["selected_forward_kernel_duration_us"] == pytest.approx(
                expected_duration, abs=1e-3
            )
        if profile["implementation_id"].startswith("sglang_") and profile["phase"] == "decode":
            gate = profile["profiler"]["rank_collective_duration_gate"]
            assert gate["state"] == "passed"
            assert gate["signature_outlier_count"] == 0
            assert gate["signature_count"] in {121, 240}
            assert set(gate["per_rank"]) == {str(rank) for rank in range(8)}
            assert all(
                row["logical_all_reduce_count"] == 121
                and row["physical_all_reduce_kernel_count"] in {121, 240}
                and row["max_single_all_reduce_ms"] <= gate[
                    "max_single_limit_ms"
                ]
                and row["mapped_kernel_envelope_ms"] <= gate[
                    "mapped_envelope_upper_ms"
                ]
                for row in gate["per_rank"].values()
            )
            sync = profile["profiler"]["profiler_sync_evidence"]
            assert sync["state"] == "passed"
            assert len(sync["overlay_sha256"]) == 64
            assert len(sync["source_lock_sha256"]) == 64
            assert set(sync["marker_counts"]) == {
                "pre_activation_barrier",
                "post_activation_barrier",
                "activation_complete",
                "pre_input_preparation_barrier",
                "input_preparation_barrier_passed",
            }
            assert all(
                set(counts) == {str(rank) for rank in range(8)}
                and set(counts.values()) == {1}
                for counts in sync["marker_counts"].values()
            )
            assert sync["pre_forward_device_collective_added"] is False
            assert sync["all_tp_rank_count"] == 8
        assert profile["evidence"]["mapped_kernel_count_ratio"] >= 0.95
        assert profile["evidence"]["mapped_kernel_duration_ratio"] >= 0.95
        diagnostics = profile["evidence"]["attribution_diagnostics"]
        assert diagnostics["tp_logical_all_reduce_count"] == 121
        assert diagnostics["tp_all_gather_count"] == 1


def test_qwen35_runtime_bearing_semantics_are_measured_or_explicitly_closed() -> None:
    model = load_yaml(MODEL_ROOT / "model_ir.yaml")
    required = {
        f"{view_id}.{node['id']}"
        for view_id, view in model["views"].items()
        for node in view["nodes"]
        if (node.get("semantic_details") or {}).get("runtime_mapping", {}).get("expectation") == "measured"
    }
    assert required
    for path in qwen35_profile_paths():
        profile = load_yaml(path)
        for target in required:
            if target in profile["node_metrics"]:
                assert profile["node_metrics"][target]["ms_per_iter"] > 0
                assert profile["node_metrics"][target]["attribution_status"] in {
                    "measured_direct",
                    "typed_unresolved",
                    "inclusive_rollup",
                }
                continue
            state = profile["node_states"][target]
            assert state["status"] in {"fused", "partially_fused", "not_selected"}, (path, target, state)
            assert state["label"]


def test_qwen35_fused_members_have_exactly_one_timing_owner_and_no_copied_metric() -> None:
    for path in qwen35_profile_paths():
        profile = load_yaml(path)
        memberships: dict[str, str] = {}
        for group_id, group in profile["fusion_groups"].items():
            owner = group["owner"]
            assert owner in profile["node_metrics"], (path, group_id, owner)
            assert group["ir_nodes"][0] == owner
            for member in group["ir_nodes"][1:]:
                assert member not in memberships, (path, member)
                memberships[member] = owner
                assert member not in profile["node_metrics"]
                assert profile["node_states"][member] == {
                    "status": "fused",
                    "label": f"fused into {owner}",
                    "included_in": owner,
                    "fusion_group_id": group_id,
                }


def test_qwen35_fusion_groups_have_exact_owner_member_event_sets() -> None:
    for profile_path in qwen35_profile_paths():
        profile = load_yaml(profile_path)
        with gzip.open(profile_path.with_name(profile["timeline"]["artifact"]), "rt") as source:
            timeline = json.load(source)
        strings = timeline["strings"]
        events = [event for step in timeline["steps"] for event in step["events"]]
        for group_id, group in profile["fusion_groups"].items():
            owner = group["owner"]
            owner_ids = {
                strings[event["raw_event_id"]]
                for event in events
                if event["ir_node"] is not None and strings[event["ir_node"]] == owner
            }
            assert owner_ids == set(group["evidence_scope"]["production_event_ids"]), (
                profile_path,
                group_id,
            )
            assert group["evidence_scope"]["member_event_sets_equal_owner"] is True
            for member in group["ir_nodes"][1:]:
                member_ids = {
                    strings[event["raw_event_id"]]
                    for event in events
                    if member in {strings[index] for index in event["ir_targets"]}
                }
                assert member_ids == owner_ids, (profile_path, group_id, member)


def test_qwen35_known_unequal_fusion_candidates_remain_occurrence_scoped() -> None:
    sglang = MODEL_ROOT / "profiles" / "tp8" / "sglang_f609d677b_qwen35_033446bb_tp8"
    vllm = MODEL_ROOT / "profiles" / "tp8" / "vllm_487ecf187_qwen35_native_tp8"
    cases = (
        (sglang / "prefill_bs1_8k1k.yaml", "full_attention_moe_block.layer_residual"),
        (vllm / "prefill_bs1_8k1k.yaml", "full_attention_moe_block.layer_residual"),
        (sglang / "cg_decode_bs1_8k1k.yaml", "full_attention_moe_block.input_norm"),
        (sglang / "cg_decode_bs16_8k1k.yaml", "full_attention_moe_block.input_norm"),
        (vllm / "cg_decode_bs64_8k1k.yaml", "full_attention_moe_block.input_norm"),
        (vllm / "cg_decode_bs256_8k1k.yaml", "full_attention_moe_block.input_norm"),
    )
    for profile_path, member in cases:
        profile = load_yaml(profile_path)
        state = profile["node_states"][member]
        assert state["status"] == "partially_fused", (profile_path, member, state)
        assert "fusion_group_id" not in state
        assert state["label"].startswith("occurrence-scoped partial fusion only")


def test_qwen35_timelines_close_semantic_events_to_eager_stacks_and_exact_targets() -> None:
    for profile_path in qwen35_profile_paths():
        profile = load_yaml(profile_path)
        timeline_path = profile_path.with_name(profile["timeline"]["artifact"])
        with gzip.open(timeline_path, "rt") as source:
            timeline = json.load(source)
        strings = timeline["strings"]
        events = [event for step in timeline["steps"] for event in step["events"]]
        semantic = [event for event in events if event["ir_node"] is not None]
        support = [event for event in events if event["ir_node"] is None]
        assert semantic and support
        for event in semantic:
            node = strings[event["ir_node"]]
            assert node in {strings[index] for index in event["ir_targets"]}
            reconciliation = strings[event["reconciliation_status"]]
            if reconciliation == "closed":
                assert event["stack_id"] is not None, (profile_path, event["event_id"], node)
                stack = timeline["stacks"][event["stack_id"]]
                evidence = {
                    key: strings[value] if value is not None else None
                    for key, value in stack["evidence"].items()
                }
                assert evidence["match"] == (
                    "same_rank_phase_occurrence_signature_ordered_sequence"
                )
                assert evidence["rank"] == str(profile["timeline"]["reference_rank"])
                assert evidence["phase"] == profile["phase"]
            else:
                assert reconciliation == "typed_unresolved"
                assert event["stack_id"] is None
                assert strings[event["confidence"]] == "review_required"
                assert strings[event["reconciliation_reason"]]
        for event in support:
            assert event["support_class"] is not None
            assert event["support_reason"] is not None


def test_qwen35_vllm_large_decode_preserves_physical_n_to_one_collective_events() -> None:
    root = MODEL_ROOT / "profiles" / "tp8" / "vllm_487ecf187_qwen35_native_tp8"
    for batch in (64, 256):
        profile_path = root / f"cg_decode_bs{batch}_8k1k.yaml"
        profile = load_yaml(profile_path)
        with gzip.open(profile_path.with_name(profile["timeline"]["artifact"]), "rt") as source:
            timeline = json.load(source)
        strings = timeline["strings"]
        companions = [
            event
            for step in timeline["steps"]
            for event in step["events"]
            if strings[event["attribution_method"]] == "n_to_one_flashinfer_twoshot_rmsnorm_companion"
        ]
        assert len(companions) == 120
        assert all(event["ir_node"] is not None for event in companions)
        assert all(
            strings[event["reconciliation_status"]] in {"closed", "typed_unresolved"}
            for event in companions
        )


def test_qwen35_vllm_decode_does_not_publish_fill_as_qk_rope_fusion() -> None:
    root = MODEL_ROOT / "profiles" / "tp8" / "vllm_487ecf187_qwen35_native_tp8"
    for batch in (1, 16, 64, 256):
        profile_path = root / f"cg_decode_bs{batch}_8k1k.yaml"
        profile = load_yaml(profile_path)
        assert not [
            group_id
            for group_id, group in profile["fusion_groups"].items()
            if group["owner"] == "full_attention.qk_norm"
            and "full_attention.partial_rope" in group["ir_nodes"]
        ]
        with gzip.open(profile_path.with_name(profile["timeline"]["artifact"]), "rt") as source:
            timeline = json.load(source)
        strings = timeline["strings"]
        fill_qk_events = [
            event
            for step in timeline["steps"]
            for event in step["events"]
            if event["ir_node"] is not None
            and strings[event["ir_node"]] == "full_attention.qk_norm"
            and "FillFunctor<unsigned char>" in strings[event["kernel_name"]]
        ]
        assert fill_qk_events
        assert all(event["stack_id"] is None for event in fill_qk_events)
        assert {
            strings[event["reconciliation_status"]] for event in fill_qk_events
        } == {"typed_unresolved"}


def test_qwen35_bindings_are_commit_specific_validated_and_complete() -> None:
    bundle = compile_catalog(MODEL_ROOT)
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
    for path in [MODEL_ROOT / "model_ir.yaml", *qwen35_profile_paths()]:
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
