from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path

import pytest
import yaml

from llm_arch_v2 import compile_catalog
from llm_arch_v2.compiler import CatalogError, apply_execution_plan


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
        assert selected_graph["model_kernel_count"] == (
            selected_graph["graph_kernel_count"] + selected_graph["non_graph_kernel_count"]
        )
        assert selected_graph["all_tp_ranks_consistent"] is True
        assert selected_graph["used_graph_path"] is (selected_graph["graph_kernel_count"] > 0)
        assert profiler["server_cuda_graph_config"]["enabled"] is True
        assert profiler["server_cuda_graph_config"]["evidence_files"]
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
        assert profile["evidence"]["unclassified_kernel_count"] == 0
        assert profile["evidence"]["semantic_stack_closure_missing_node_count"] == 0
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
            assert event["stack_id"] is not None, (profile_path, event["event_id"], node)
            assert node in {strings[index] for index in event["ir_targets"]}
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
        assert all(event["stack_id"] is not None for event in companions)


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
