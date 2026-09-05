from __future__ import annotations

import copy
import gzip
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from llm_arch_v2 import compile_catalog
from llm_arch_v2.compiler import CatalogError, apply_execution_plan
from models.qwen35.build.build_qwen35_production_profiles import (
    build_states_and_fusions,
    fusion_target_is_physically_proven,
    profile_acceptance_gate,
    rank_collective_duration_gate,
    reconciliation_kernel_family,
    sglang_prefill_forward_timing_coordinate,
    wall_trace_contract_gate,
)
from models.qwen35.build.qwen35_production_attribution import (
    semantic_execution_order,
)
from models.qwen35.build.qwen35_eager_semantic_validation import (
    PUBLISHED_NODE_FAMILIES,
    validate_eager_semantic_attribution,
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


def test_qwen35_semantic_tensor_axes_keep_declared_dimension_labels() -> None:
    """Do not collapse named model axes back into unexplained constants.

    Concrete values belong to ``dimension_symbols`` and the Viewer resolver;
    Model IR edges and equations must preserve the semantic axis names that
    make the same data flow comparable across framework implementations.
    """

    model = load_yaml(MODEL_ROOT / "model_ir.yaml")
    text = json.dumps(model, sort_keys=True)
    for legacy in (
        "[N,64,128]",
        "[B,64,128,128]",
        "[N,32,256]",
        "[N,2,256]",
        "[B,2,S,256]",
        "[P,1152]",
        "[M,4096]",
    ):
        assert legacy not in text

    gdn_edges = {
        edge["identity"]: edge
        for edge in model["views"]["gdn_attention"]["edges"]
    }
    assert gdn_edges["Z gate"]["shape"] == (
        "Z [N,GDN_value_heads,GDN_value_head_dim]"
    )
    assert gdn_edges[
        "gdn_attention.gated_delta_recurrence_to_output_gate_norm"
    ]["shape"] == "[N,GDN_value_heads,GDN_value_head_dim]"

    attention_edges = {
        edge["identity"]: edge
        for edge in model["views"]["full_attention"]["edges"]
    }
    assert attention_edges["Q/K"]["shape"] == (
        "Q [N,Q_heads,attention_head_dim] + "
        "K [N,KV_heads,attention_head_dim]"
    )


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


def test_qwen35_cross_framework_matrix_is_complete_and_fail_closed() -> None:
    manifest = load_yaml(MODEL_ROOT / "unsupported_profiles.yaml")
    profile_paths = qwen35_profile_paths()
    assert manifest["expected_profile_count"] == 10
    assert manifest["accepted_profile_count"] == 10
    assert manifest["unsupported_profile_count"] == 0
    assert manifest["profiles"] == []
    assert len(profile_paths) == 10
    profiles = [load_yaml(path) for path in profile_paths]
    profile_path_by_id = {
        profile["profile_id"]: path for profile, path in zip(profiles, profile_paths)
    }
    all_eager_anchor_families: set[str] = set()
    assert {
        (
            "sglang" if "sglang" in profile["implementation_id"] else "vllm",
            profile["phase"],
            profile["workload"]["batch_size"],
        )
        for profile in profiles
    } == {
        (framework, phase, batch)
        for framework in ("sglang", "vllm")
        for phase, batch in (("prefill", 1), ("decode", 1), ("decode", 16), ("decode", 64), ("decode", 256))
    }
    for profile in profiles:
        batch = profile["workload"]["batch_size"]
        framework = (
            "sglang" if "sglang" in profile["implementation_id"] else "vllm"
        )
        assert profile["acceptance"] == {
            "state": "accepted",
            "fail_closed": True,
            "reason_count": 0,
            "reasons": [],
        }
        assert profile["workload"]["isl"] == 8192
        assert profile["workload"]["osl"] == 1024
        assert profile["workload"]["warmup_requests"] == 3 * batch
        assert profile["workload"]["formal_requests"] == batch
        assert profile["evidence"]["typed_unresolved_semantic_event_count"] == 0
        rank_diagnostics = profile["evidence"]["attribution_diagnostics"]
        assert rank_diagnostics["semantic_reconciliation"][
            "typed_unresolved_event_count"
        ] == 0
        assert all(
            state.get("status") != "partially_fused"
            for state in profile["node_states"].values()
        )
        for group in profile["fusion_groups"].values():
            evidence_scope = group["evidence_scope"]
            timing_semantics = group["timing_semantics"]
            assert timing_semantics in {
                "shared_interval",
                "shared_event_set",
                "shared_event_coverage",
            }
            if timing_semantics == "shared_event_set":
                assert evidence_scope["member_event_sets_equal_owner"] is True
                assert evidence_scope["all_owner_events_same_rank_closed"] is True
            elif timing_semantics == "shared_event_coverage":
                owner_events = set(evidence_scope["owner_event_ids"])
                assert owner_events
                for member, member_events in evidence_scope[
                    "member_event_ids"
                ].items():
                    assert member in group["ir_nodes"]
                    assert member_events
                    assert set(member_events) <= owner_events
            owner = group["owner"]
            for member in group["ir_nodes"]:
                if member == owner:
                    continue
                assert member not in profile["node_metrics"]
                assert profile["node_states"][member]["included_in"] == owner
        timeline_path = profile_path_by_id[profile["profile_id"]].with_name(
            profile["timeline"]["artifact"]
        )
        with gzip.open(timeline_path, "rt") as handle:
            timeline = json.load(handle)
        strings = timeline["strings"]

        def decode(value):
            return strings[value] if isinstance(value, int) else value

        events = [
            event for step in timeline["steps"] for event in step["events"]
        ]
        assert all(
            decode(event.get("reconciliation_status")) == "closed"
            for event in events
            if decode(event.get("ir_node"))
        )
        assert all(
            decode(event.get("ir_node")) or decode(event.get("support_class"))
            for event in events
        )
        for group in profile["fusion_groups"].values():
            timing_semantics = group["timing_semantics"]
            owner_events = {
                decode(event["raw_event_id"])
                for event in events
                if decode(event.get("ir_node")) == group["owner"]
            }
            assert owner_events
            for member in group["ir_nodes"]:
                if member == group["owner"]:
                    continue
                member_events = {
                    decode(event["raw_event_id"])
                    for event in events
                    if member
                    in [decode(target) for target in event.get("ir_targets") or []]
                }
                if timing_semantics == "shared_event_coverage":
                    assert member_events
                    assert member_events <= owner_events
                else:
                    assert member_events == owner_events
        qk_kernels = profile["node_metrics"].get("full_attention.qk_norm", {}).get(
            "kernels", []
        )
        assert all(
            "fillfunctor" not in kernel["name"].lower() for kernel in qk_kernels
        )

        selected_graph = profile["profiler"]["selected_forward_cuda_graph"]
        assert profile["profiler"]["cuda_graph_enabled"] is selected_graph[
            "used_graph_path"
        ]
        assert selected_graph["used_graph_path"] is (selected_graph["graph_id_count"] > 0)
        assert selected_graph["model_kernel_count"] == (
            selected_graph["graph_kernel_count"] + selected_graph["non_graph_kernel_count"]
        )
        assert selected_graph["all_tp_ranks_consistent"] is True
        assert selected_graph["used_graph_path"] is (selected_graph["graph_kernel_count"] > 0)
        evidence_basis = selected_graph["evidence_basis"]
        graph_semantics = profile["profiler"]["cuda_graph_enabled_semantics"]
        if selected_graph["used_graph_path"]:
            assert f"{selected_graph['graph_id_count']} distinct nonzero raw-trace graph IDs" in evidence_basis
            assert f"{selected_graph['graph_kernel_count']} model-bearing kernels" in evidence_basis
            assert graph_semantics == (
                "selected formal forward used a CUDA Graph path; "
                f"{selected_graph['graph_kernel_count']} model-bearing kernels "
                "have a nonzero raw-trace graph_id"
            )
        else:
            assert evidence_basis == (
                f"zero nonzero raw-trace graph IDs across {selected_graph['model_kernel_count']} "
                "model-bearing kernels in the selected formal forward"
            )
            assert graph_semantics == (
                "selected formal forward did not use CUDA Graph replay; zero "
                "nonzero raw-trace graph IDs were observed across all "
                f"{selected_graph['model_kernel_count']} model-bearing kernels"
            )
        if profile["phase"] == "decode":
            assert selected_graph["used_graph_path"] is True
            expected_replay_state = (
                "mixed_graph_and_eager"
                if selected_graph["non_graph_kernel_count"]
                else "cuda_graph_replay"
            )
            assert selected_graph["replay_state"] == expected_replay_state
        elif framework == "sglang":
            assert selected_graph["used_graph_path"] is True
            assert selected_graph["replay_state"] == "mixed_graph_and_eager"
        else:
            assert selected_graph["used_graph_path"] is False
            assert selected_graph["replay_state"] == "no_cuda_graph_replay"
            assert selected_graph["graph_kernel_count"] == 0
        assert len(profile["evidence"]["all_rank_eager_mapping_sha256"]) == 8
        assert len(profile["evidence"]["all_rank_eager_raw_manifest_sha256"]) == 8
        assert len(profile["evidence"]["all_rank_trace_sha256"]) == 8
        phase_contract = profile["evidence"]["all_rank_eager_phase_contract"]
        assert set(phase_contract) == {str(rank) for rank in range(8)}
        expected_source_phase = (
            f"vllm_{profile['phase']}"
            if framework == "vllm"
            else f"forward_{'extend' if profile['phase'] == 'prefill' else 'decode'}"
        )
        expected_source_commit = (
            "f609d677b909ca46c64bb6803b69a85fedbf86bc"
            if framework == "sglang"
            else "487ecf187d3dfe74d2cf6119a92881dba403c219"
        )
        expected_coordinate = (
            {("prefill", 1): 0, ("decode", 1): 511, ("decode", 16): 529,
             ("decode", 64): 582, ("decode", 256): 769}
            if framework == "sglang"
            else {("prefill", 1): 0, ("decode", 1): 513, ("decode", 16): 521,
                  ("decode", 64): 545, ("decode", 256): 643}
        )[(profile["phase"], batch)]
        for rank, contract in phase_contract.items():
            assert contract["source_phase"] == expected_source_phase, rank
            assert contract["selected_forward_kernel_count"] > 0
            assert contract["selected_forward_kernel_duration_us"] > 0
            assert contract["source_commit"] == expected_source_commit
            assert contract["hardware"] == "NVIDIA GB300"
            assert contract["world_size"] == 8
            assert len(contract["raw_trace_sha256"]) == 64
            assert len(contract["raw_manifest_sha256"]) == 64
            assert len(contract["capture_metadata_sha256"]) == 64
            assert len(contract["selected_forward_events_sha256"]) == 64
            assert contract["selected_forward_kernel_count"] == (
                contract["graph_off_semantic_event_count"]
                + contract["graph_off_support_event_count"]
            )
            production_count = (
                contract["production_semantic_event_count"]
                + contract["production_support_event_count"]
            )
            assert production_count > 0
            assert contract["typed_unresolved_event_count"] == 0
            assert contract["closed_production_event_count"] == contract[
                "production_semantic_event_count"
            ]
            owner_validation = contract[
                "independent_eager_semantic_owner_validation"
            ]
            assert owner_validation["validated_semantic_event_count"] == contract[
                "graph_off_semantic_event_count"
            ]
            assert owner_validation["unanchored_semantic_event_count"] == 0
            assert owner_validation["owner_disagreement_count"] == 0
            all_eager_anchor_families.update(
                owner_validation["published_node_families"]
            )
            coordinate = contract["selected_formal_coordinate"]
            if framework == "sglang":
                assert coordinate["mode"] == "sglang_formal_relative_scheduler_forward"
                assert coordinate["relative_step"] == expected_coordinate
            else:
                assert coordinate == {
                    "mode": "vllm_start_profile_relative_engine_iteration",
                    "delay_iterations": expected_coordinate,
                    "active_iterations": 1,
                }
        assert len(profile["evidence"]["validation_sha256"]) == 64

    assert all_eager_anchor_families == PUBLISHED_NODE_FAMILIES

    sglang_prefill = next(
        profile
        for profile in profiles
        if "sglang" in profile["implementation_id"] and profile["phase"] == "prefill"
    )
    gate = sglang_prefill["evidence"]["timing"]["wall_trace_contract_gate"]
    assert gate["state"] == "passed"
    assert gate["same_isolated_forward_proven"] is True
    assert gate["serving_wall_ms"] == pytest.approx(93.678078)
    assert gate["instrumented_active_gpu_ms"] == 89.582258
    assert gate["instrumented_kernel_envelope_ms"] == 93.450314


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


def test_qwen35_sglang_prefill_timing_selects_exact_post_warmup_forward(
    tmp_path: Path,
) -> None:
    client_relative = Path("evidence/sglang-fpm-baseline-c1/1/logs/client-c1.json")
    client_path = tmp_path / client_relative
    client_path.parent.mkdir(parents=True)
    contract = {
        "concurrency": 1,
        "formal_request_count": 1,
        "ignore_eos": True,
        "isl": 8192,
        "mtp_nextn": False,
        "no_intentionally_shared_prefix": True,
        "osl": 1024,
        "random_range_ratio": 1.0,
        "random_token_ids": True,
        "seed": 0,
        "warmup_request_count": 3,
    }
    selected = {
        "received_at": "2026-08-30T00:00:01+00:00",
        "transport_sequence": 102,
        "payload_sha256": "a" * 64,
        "metrics": {
            "counter_id": 102,
            "wall_time": 0.09367807769775391,
            "scheduled_requests": {
                "num_prefill_requests": 1,
                "sum_prefill_tokens": 8192,
                "sum_prefill_kv_tokens": 8192,
                "num_decode_requests": 0,
                "sum_decode_kv_tokens": 0,
            },
        },
    }
    client = {
        "state": "passed",
        "contract": contract,
        "forward_pass_metrics": {
            "counter_floor_after_warmup": 100,
            "messages": [
                {
                    "transport_sequence": 101,
                    "metrics": {
                        "counter_id": 101,
                        "wall_time": 0.004,
                        "scheduled_requests": {
                            "num_prefill_requests": 0,
                            "sum_prefill_tokens": 0,
                            "sum_prefill_kv_tokens": 0,
                            "num_decode_requests": 1,
                            "sum_decode_kv_tokens": 8193,
                        },
                    },
                },
                selected,
            ],
        },
    }
    client_path.write_text(json.dumps(client))
    client_sha = hashlib.sha256(client_path.read_bytes()).hexdigest()
    selection_path = tmp_path / "validation" / "sglang-prefill-fpm-selection.json"
    selection_path.parent.mkdir(parents=True)
    selection_path.write_text(
        json.dumps(
            {
                "schema_version": "qwen35-sglang-forward-timing-selection.v1",
                "state": "passed",
                "job_id": "1",
                "framework": "sglang",
                "phase": "prefill",
                "batch_size": 1,
                "contract": {
                    "input_length": 8192,
                    "output_length": 1024,
                    "warmup_request_count": 3,
                    "formal_request_count": 1,
                    "no_intentionally_shared_prefix": True,
                    "ignore_eos": True,
                    "mtp_nextn": False,
                    "topology": {
                        "tensor_parallel_size": 8,
                        "data_parallel_size": 1,
                        "pipeline_parallel_size": 1,
                        "expert_parallel_size": 1,
                    },
                },
                "selection": {
                    "counter_floor_after_warmup": 100,
                    "matching_message_count": 1,
                    "transport_sequence": 102,
                    "counter_id": 102,
                    "received_at": selected["received_at"],
                    "payload_sha256": selected["payload_sha256"],
                    "wall_time_seconds": selected["metrics"]["wall_time"],
                    "wall_time_ms": selected["metrics"]["wall_time"] * 1000.0,
                },
                "evidence": {
                    "client_path": str(client_relative),
                    "client_sha256": client_sha,
                },
                "rejected_previous_authority": {
                    "value_ms": 4111.995663,
                    "reason": "not an isolated forward",
                },
            }
        )
    )

    coordinate, evidence = sglang_prefill_forward_timing_coordinate(tmp_path)
    assert coordinate["baseline_mean_elapsed_ms"] == pytest.approx(93.6780776977539)
    assert coordinate["selected_counter_id"] == 102
    assert coordinate["same_isolated_forward_proven"] is True
    assert evidence["raw_client_sha256"] == client_sha


def test_qwen35_bindings_are_commit_specific_validated_and_complete() -> None:
    bundle = compile_catalog(MODEL_ROOT)
    assert len(bundle["profiles"]) == 10
    variant = next(iter(bundle["execution_variants"].values()))
    target_count = sum(len(view["nodes"]) for view in variant["views"].values())
    assert target_count == 203
    assert len(bundle["implementations"]) == 2
    for implementation in bundle["implementations"].values():
        assert implementation["binding_status"] == "validated"
        assert implementation["source_lock_status"] == "runtime_verified"
        assert implementation["execution_validation"]["status"] == "pass"
        assert implementation["execution_validation"]["cuda_graph_enabled"] is False
        assert implementation["execution_validation"]["execution_fingerprint"] == "exec_2ca0442bb646b1ff"
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


def test_qwen35_cuda_graph_semantic_order_uses_graph_node_ids() -> None:
    rows = [
        {"event_id": "setup", "ts_us": 1.0, "graph_id": 0, "graph_node_id": None},
        {"event_id": "late", "ts_us": 20.0, "graph_id": 7, "graph_node_id": 3},
        {"event_id": "early", "ts_us": 10.0, "graph_id": 7, "graph_node_id": 1},
        {"event_id": "middle", "ts_us": 30.0, "graph_id": 7, "graph_node_id": 2},
        {"event_id": "output", "ts_us": 40.0, "graph_id": 0, "graph_node_id": None},
    ]
    assert [row["event_id"] for row in semantic_execution_order(rows)] == [
        "setup",
        "early",
        "middle",
        "late",
        "output",
    ]

    piecewise = [dict(row) for row in rows]
    piecewise[2]["graph_id"] = 8
    assert [row["event_id"] for row in semantic_execution_order(piecewise)] == [
        "setup",
        "early",
        "late",
        "middle",
        "output",
    ]


def test_qwen35_reconciliation_normalizes_tool_demangle_spelling_only() -> None:
    torch_name = (
        "void at::native::vectorized_elementwise_kernel<4, "
        "at::native::exp_kernel_cuda(at::TensorIteratorBase&)>()"
    )
    nsys_name = (
        "void at::native::vectorized_elementwise_kernel<(int)4, "
        "at::native::exp_kernel_cuda(at::TensorIteratorBase &)>()"
    )
    assert reconciliation_kernel_family({"kernel_name": torch_name}) == "exp_elementwise"
    assert reconciliation_kernel_family({"kernel_name": nsys_name}) == "exp_elementwise"
    assert reconciliation_kernel_family({"kernel_name": "FillFunctor<unsigned char>"}) == "fill"


def test_qwen35_aggregate_fusion_requires_exact_physical_event_set() -> None:
    model_ir = {
        "views": {
            "gdn_attention": {
                "nodes": [
                    {"id": "gated_delta_recurrence"},
                    {"id": "recurrent_state_read"},
                    {"id": "state_write"},
                ]
            }
        }
    }
    rows = [
        {
            "event_id": f"event-{index}",
            "node": "gdn_attention.gated_delta_recurrence",
            "kernel_name": "chunkedGatedDeltaNetChunkedKernel",
            "reconciliation_status": "closed",
            "ir_targets": [
                "gdn_attention.gated_delta_recurrence",
                "gdn_attention.recurrent_state_read",
                "gdn_attention.state_write",
            ],
        }
        for index in range(2)
    ]
    states, groups = build_states_and_fusions(
        model_ir=model_ir,
        execution_plan={"transforms": []},
        rows=rows,
        metrics={"gdn_attention.gated_delta_recurrence": {}},
    )
    assert len(groups) == 1
    group = next(iter(groups.values()))
    assert group["evidence_scope"]["member_event_sets_equal_owner"] is True
    assert group["evidence_scope"]["production_event_ids"] == ["event-0", "event-1"]
    assert states["gdn_attention.recurrent_state_read"]["status"] == "fused"
    assert states["gdn_attention.state_write"]["status"] == "fused"

    rows[1]["ir_targets"].remove("gdn_attention.state_write")
    states, groups = build_states_and_fusions(
        model_ir=model_ir,
        execution_plan={"transforms": []},
        rows=rows,
        metrics={"gdn_attention.gated_delta_recurrence": {}},
    )
    assert all(
        "gdn_attention.state_write" not in group["ir_nodes"]
        for group in groups.values()
    )
    assert states["gdn_attention.state_write"]["status"] == "structural"
    assert "included_in" not in states["gdn_attention.state_write"]


def test_qwen35_vllm_generated_fused_add_rms_norm_is_physical_proof() -> None:
    row = {
        "kernel_name": (
            "triton_red_fused__to_copy_add_fused_add_rms_norm_"
            "moe_forward_shared_1"
        ),
        "cpu_op_name": "",
        "attribution_method": "vllm_prefill_eager_validated_post_attention_norm",
    }
    assert fusion_target_is_physically_proven(
        row,
        "full_attention_moe_block.post_attention_norm",
        "full_attention_moe_block.attention_residual",
    )


def test_qwen35_eager_owner_gate_rejects_shared_expert_as_final_norm() -> None:
    row = {
        "event_id": "e-r0-regression",
        "rank": 0,
        "node": "top.final_norm",
        "kernel_name": "triton_red_fused_add_rms_norm_3",
        "cpu_op_name": "aten::copy_",
        "ir_targets": ["top.final_norm"],
        "python_stack": [
            {"raw": "nn.Module: SiluAndMul_59"},
            {"raw": "nn.Module: Qwen2MoeMLP_59"},
            {"raw": "nn.Module: SharedExperts_59"},
            {"raw": "vllm/model_executor/models/qwen3_next.py(632): forward"},
        ],
    }
    with pytest.raises(ValueError, match="Python-stack owner.*disagrees"):
        validate_eager_semantic_attribution(
            [row], framework="vllm", phase="decode"
        )


def test_qwen35_vllm_tail_has_independent_final_norm_and_lm_head_anchors() -> None:
    profile_root = (
        MODEL_ROOT
        / "profiles"
        / "tp8"
        / "vllm_487ecf187_qwen35_native_tp8"
    )
    saw_prefill_final_norm = False
    saw_decode_fused_final_norm = False
    saw_lm_head = False
    for profile_path in sorted(profile_root.glob("*.yaml")):
        profile = load_yaml(profile_path)
        timeline_path = profile_path.with_name(profile["timeline"]["artifact"])
        with gzip.open(timeline_path, "rt") as handle:
            timeline = json.load(handle)
        strings = timeline["strings"]

        def text(value):
            return strings[value] if isinstance(value, int) else value

        def stack_text(event):
            stack_id = event.get("stack_id")
            if stack_id is None:
                return ""
            return "\n".join(
                str(text(frame["raw"]))
                for frame in timeline["stacks"][stack_id]["frames"]
            ).lower()

        events = [event for step in timeline["steps"] for event in step["events"]]
        for event in events:
            node = text(event.get("ir_node"))
            targets = [text(target) for target in event.get("ir_targets") or []]
            stack = stack_text(event)
            kernel = str(text(event.get("kernel_name")) or "").lower()
            if node == "top.final_norm":
                assert profile["phase"] == "prefill"
                assert not any(
                    token in stack
                    for token in ("sharedexperts", "qwen2moemlp", "siluandmul")
                )
                assert "qwen3_next.py" in stack
                assert "rms_norm" in kernel or "rmsnorm" in kernel
                saw_prefill_final_norm = True
            if node == "top.lm_head":
                assert (
                    "logits_processor.py" in stack
                    or (
                        "compute_logits" in stack
                        and "layers/linear.py" in stack
                    )
                )
                assert not any(
                    token in stack
                    for token in ("sharedexperts", "qwen2moemlp", "siluandmul")
                )
                saw_lm_head = True
            if "top.final_norm" in targets and profile["phase"] == "decode":
                assert node == "full_attention_moe_block.tp_moe_output_collective"
                assert "allreduce" in kernel or "all_reduce" in kernel
                evidence = timeline["stacks"][event["stack_id"]]["evidence"]
                anchored = str(text(evidence["independent_eager_semantic_owner_evidence"]))
                assert "full_attention_moe_block.tp_moe_output_collective" in anchored
                final_norm_state = profile["node_states"]["top.final_norm"]
                assert final_norm_state["status"] == "fused"
                assert final_norm_state["included_in"] == node
                fusion_group = profile["fusion_groups"][
                    final_norm_state["fusion_group_id"]
                ]
                assert fusion_group["timing_semantics"] == (
                    "shared_event_coverage"
                )
                event_id = str(text(event["event_id"]))
                assert event_id in fusion_group["evidence_scope"][
                    "member_event_ids"
                ]["top.final_norm"]
                saw_decode_fused_final_norm = True

        if profile["profile_id"] == "qwen35_tp8_vllm_cg_decode_bs1_8k1k":
            bad_old_owner = next(
                event
                for event in events
                if text(event.get("raw_event_id")) == "r0-k3878"
            )
            assert text(bad_old_owner["ir_node"]) == "moe_block.shared_expert"
            assert "sharedexperts_59" in stack_text(bad_old_owner)

    assert saw_prefill_final_norm
    assert saw_decode_fused_final_norm
    assert saw_lm_head


def test_qwen35_vllm_prefill_input_norm_has_typed_fusion_owner() -> None:
    profile = load_yaml(
        MODEL_ROOT
        / "profiles"
        / "tp8"
        / "vllm_487ecf187_qwen35_native_tp8"
        / "prefill_bs1_8k1k.yaml"
    )
    state = profile["node_states"]["full_attention_moe_block.input_norm"]
    assert state["status"] == "fused"
    assert state["included_in"] == "full_attention.qkv_projection"
    group = profile["fusion_groups"][state["fusion_group_id"]]
    assert group["timing_semantics"] == "shared_event_coverage"
    member_events = group["evidence_scope"]["member_event_ids"][
        "full_attention_moe_block.input_norm"
    ]
    owner_events = group["evidence_scope"]["owner_event_ids"]
    assert member_events
    assert set(member_events) < set(owner_events)
