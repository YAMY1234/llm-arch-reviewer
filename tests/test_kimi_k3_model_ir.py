from __future__ import annotations

import copy
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3

import pytest
import yaml

from llm_arch_v2 import compile_catalog
from llm_arch_v2.compiler import CatalogError, apply_execution_plan
from llm_arch_v2.profile_acceptance import validate_executable_drill_rollups
from models.common.trace_mapping import FrameRef
from models.kimi_k3.build.kimi_k3_trace_rules import classify_kimi_k3_node
from models.kimi_k3.build.kimi_k3_vllm_trace_rules import (
    classify_kimi_k3_vllm_node,
)
from models.kimi_k3.build.kimi_k3_profile_contract import (
    build_node_states,
    sglang_fusion_groups,
    vllm_fusion_groups,
)
from models.kimi_k3.build.kimi_k3_production_attribution import (
    ATTN_RES_ANCHOR_COUNT,
    anchor_segments,
    occurrence_for_segment,
)
from models.kimi_k3.build.kimi_k3_vllm_production_attribution import (
    ATTN_RES_ANCHOR_COUNT as VLLM_ATTN_RES_ANCHOR_COUNT,
    anchor_segments as vllm_anchor_segments,
    attribute_vllm_production_events,
    occurrence_for_segment as vllm_occurrence_for_segment,
)
from models.kimi_k3.build.build_kimi_k3_sglang_profile import (
    read_exact_device_kernels,
)
from models.kimi_k3.build.build_kimi_k3_vllm_profile import (
    apply_canonical_fusion_owners,
)
from models.kimi_k3.build.kimi_k3_vllm_profile_evidence import (
    read_exact_worker_kernels,
)
from models.kimi_k3.build.validate_kimi_k3_vllm_eager_matrix import (
    vllm_worker_trace_pattern,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = REPO_ROOT / "catalog" / "kimi_k3"


def test_kimi_k3_vllm_trace_pattern_is_rank_exact_for_tp_and_ep_fields() -> None:
    assert vllm_worker_trace_pattern(0) == (
        "*_tp0_dcp0_ep0_rank0.*.pt.trace.json.gz"
    )
    assert vllm_worker_trace_pattern(7) == (
        "*_tp7_dcp0_ep7_rank7.*.pt.trace.json.gz"
    )


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def test_kimi_k3_vllm_graph_decode_maps_semantic_eager_breaks(tmp_path: Path) -> None:
    eager_rows = [
        {
            "event_id": "runtime_setup",
            "kernel_name": "runtime_setup_kernel",
            "selected_node": "runtime.step_setup",
        },
        {
            "event_id": "embedding",
            "kernel_name": "embedding_kernel",
            "selected_node": "top.token_embedding",
        },
    ]
    production_rows = [
        {
            "kernel_name": "runtime_setup_kernel",
            "dur_us": 1.0,
            "graph_node_id": None,
        },
        {
            "kernel_name": "embedding_kernel",
            "dur_us": 2.0,
            "graph_node_id": None,
        },
    ]
    for anchor_id in range(VLLM_ATTN_RES_ANCHOR_COUNT):
        eager_rows.append(
            {
                "event_id": f"anchor_{anchor_id}",
                "kernel_name": "_attn_res_kernel",
                "selected_node": "attn_res.weighted_merge",
            }
        )
        production_rows.append(
            {
                "kernel_name": "_attn_res_kernel",
                "dur_us": 3.0,
                "graph_node_id": anchor_id,
            }
        )
    eager_rows.append(
        {
            "event_id": "final_norm",
            "kernel_name": "final_norm_kernel",
            "selected_node": "top.final_norm",
        }
    )
    production_rows.append(
        {
            "kernel_name": "final_norm_kernel",
            "dur_us": 4.0,
            "graph_node_id": None,
        }
    )

    eager_path = tmp_path / "kernel_mapping.jsonl"
    eager_path.write_text("".join(json.dumps(row) + "\n" for row in eager_rows))
    attributed, diagnostics = attribute_vllm_production_events(
        production_rows, eager_path
    )

    by_event = {row.get("eager_event_id"): row for row in attributed}
    assert by_event["embedding"]["node"] == "top.token_embedding"
    assert by_event["embedding"]["attribution_method"].endswith("eager_break")
    assert by_event["final_norm"]["node"] == "top.final_norm"
    assert by_event["final_norm"]["attribution_method"].endswith("eager_break")
    assert by_event["anchor_0"]["attribution_method"].endswith("graph_node")
    assert diagnostics["exact_multiset_segment_count"] == 188
    assert diagnostics["mismatched_segments"] == []


def test_kimi_k3_model_ir_locks_exact_official_architecture_facts() -> None:
    model = load_yaml(MODEL_ROOT / "model_ir.yaml")
    facts = model["facts"]
    assert facts["checkpoint"] == "moonshotai/Kimi-K3"
    assert facts["checkpoint_revision"] == "a590ce090cb049c93a33dfe8c208ec652aa20503"
    assert facts["checkpoint_config_sha256"] == (
        "9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213"
    )
    assert facts["decoder_layers"] == 93
    assert len(facts["kda_layers_1_based"]) == 69
    assert len(facts["gated_mla_layers_1_based"]) == 24
    assert set(facts["kda_layers_1_based"]).isdisjoint(
        facts["gated_mla_layers_1_based"]
    )
    assert sorted(
        facts["kda_layers_1_based"] + facts["gated_mla_layers_1_based"]
    ) == list(range(1, 94))
    assert facts["dense_layers_0_based"] == [0]
    assert facts["routed_experts"] == 896
    assert facts["experts_per_token"] == 16
    assert facts["shared_experts"] == 2
    assert facts["attn_res_block_size"] == 12
    assert facts["final_attn_res_source_count"] == 9
    assert facts["vision_depth"] == 27
    assert facts["nextn_layers"] == 0


def test_kimi_k3_pipeline_records_complete_bidirectional_semantic_closure() -> None:
    pipeline = load_yaml(MODEL_ROOT / "pipeline.yaml")
    closure = pipeline["source_lock"]["canonical_semantics"]["semantic_closure"]
    assert closure == {
        "status": "complete",
        "audit_fingerprint": "e455ed3e8dbe8d535ee9",
        "source_entrypoints_verified": 20,
        "source_obligations_total": 99,
        "source_obligations_pending": 0,
        "model_ir_leaf_count": 97,
        "reverse_mapped_leaf_count": 97,
        "uncovered_model_ir_leaf_count": 0,
    }


def test_kimi_k3_stage1_matrix_is_complete_and_content_addressed() -> None:
    pipeline = load_yaml(MODEL_ROOT / "pipeline.yaml")
    result = pipeline["stage1_result"]
    assert result["status"] == "complete"
    assert result["measured_profile_count"] == 8
    assert result["unsupported_profile_count"] == 2
    assert result["pending_profile_count"] == 0

    matrix = result["profile_matrix"]
    assert len(matrix) == pipeline["acceptance"]["expected_profile_points"] == 10
    assert sum(point["status"] == "measured" for point in matrix) == 8
    assert sum(point["status"] == "unsupported" for point in matrix) == 2
    assert all(point["status"] != "pending_runtime_evidence" for point in matrix)

    profile_paths = {
        load_yaml(path)["profile_id"]: path
        for path in (MODEL_ROOT / "profiles").rglob("*.yaml")
    }
    for point in matrix:
        if point["status"] != "measured":
            assert point["evidence_sha256"]
            continue
        profile = profile_paths[point["profile_id"]]
        timeline = profile.with_suffix(".timeline.json.gz")
        assert hashlib.sha256(profile.read_bytes()).hexdigest() == point[
            "profile_sha256"
        ]
        assert hashlib.sha256(timeline.read_bytes()).hexdigest() == point[
            "timeline_sha256"
        ]


def test_kimi_k3_every_edge_is_tensor_and_state_explicit() -> None:
    model = load_yaml(MODEL_ROOT / "model_ir.yaml")
    required = {"identity", "shape", "layout", "dtype", "state"}
    for view_id, view in model["views"].items():
        nodes = {node["id"] for node in view["nodes"]}
        for edge in view["edges"]:
            assert edge["from"] in nodes, (view_id, edge)
            assert edge["to"] in nodes, (view_id, edge)
            assert required <= set(edge), (view_id, edge)
            assert all(str(edge[field]) for field in required), (view_id, edge)


def test_kimi_k3_every_drill_has_an_exact_boundary_contract() -> None:
    model = load_yaml(MODEL_ROOT / "model_ir.yaml")
    drills = {
        f"{view_id}.{node['id']}": node["drill"]
        for view_id, view in model["views"].items()
        for node in view["nodes"]
        if "drill" in node
    }
    contracts = {entry["parent_node"]: entry for entry in model["boundary_contracts"]}
    assert drills.keys() == contracts.keys()
    for parent, child in drills.items():
        contract = contracts[parent]
        assert contract["child_view"] == child
        assert contract["input_shape"]
        assert contract["output_shape"]
        assert contract["boundary_mode"] in {
            "exact_node",
            "exact_lifecycle",
            "external_entry",
        }


def test_kimi_k3_compiled_semantics_are_edge_derived_and_equation_complete() -> None:
    raw = load_yaml(MODEL_ROOT / "model_ir.yaml")
    bundle = compile_catalog(MODEL_ROOT)
    operations = raw["semantic_contract"]["operations"]
    for view_id, raw_view in raw["views"].items():
        incoming = {node["id"]: [] for node in raw_view["nodes"]}
        outgoing = {node["id"]: [] for node in raw_view["nodes"]}
        for edge in raw_view["edges"]:
            incoming[edge["to"]].append(
                (
                    edge["identity"],
                    edge["shape"],
                    edge["layout"],
                    edge["dtype"],
                    edge["state"],
                    edge["from"],
                )
            )
            outgoing[edge["from"]].append(
                (
                    edge["identity"],
                    edge["shape"],
                    edge["layout"],
                    edge["dtype"],
                    edge["state"],
                    edge["to"],
                )
            )
        compiled = {
            node["id"]: node for node in bundle["model_ir"]["views"][view_id]["nodes"]
        }
        for raw_node in raw_view["nodes"]:
            semantics = compiled[raw_node["id"]]["semantics"]
            assert semantics["semantic_op"] == raw_node["semantic_op"]
            assert semantics["equation"] == operations[raw_node["semantic_op"]][
                "equation"
            ]
            assert semantics["equation"]
            assert "None" not in semantics["equation"]
            assert {
                (
                    item["name"],
                    item["shape"],
                    item["layout"],
                    item["dtype"],
                    item["state"],
                    item["source"],
                )
                for item in semantics["inputs"]
            } == set(incoming[raw_node["id"]])
            assert {
                (
                    item["name"],
                    item["shape"],
                    item["layout"],
                    item["dtype"],
                    item["state"],
                    item["target"],
                )
                for item in semantics["outputs"]
            } == set(outgoing[raw_node["id"]])


def test_kimi_k3_strict_semantics_fail_closed_on_missing_transition() -> None:
    model = copy.deepcopy(load_yaml(MODEL_ROOT / "model_ir.yaml"))
    plan_path = MODEL_ROOT / "execution_paths" / "tp8.yaml"
    plan = load_yaml(plan_path)
    del model["semantic_contract"]["operations"]["kda.recurrent_update"]
    with pytest.raises(CatalogError, match="explicit operation contract"):
        apply_execution_plan(model, plan, source=plan_path)


def test_kimi_k3_model_ir_is_framework_collective_and_kernel_independent() -> None:
    text = json.dumps(load_yaml(MODEL_ROOT / "model_ir.yaml"), sort_keys=True).lower()
    for forbidden in (
        "sglang",
        "vllm",
        "triton",
        "cuda",
        "nccl",
        "all_reduce",
        "all_gather",
        "reduce_scatter",
        "flashinfer",
    ):
        assert forbidden not in text


def test_kimi_k3_tp8_plan_is_pure_tp_and_fail_closed() -> None:
    model = load_yaml(MODEL_ROOT / "model_ir.yaml")
    plan = load_yaml(MODEL_ROOT / "execution_paths" / "tp8.yaml")
    constraints = plan["constraints"]
    assert constraints["tp_size"] == 8
    assert constraints["dp_size"] == 1
    assert constraints["cp_size"] == 1
    assert constraints["ep_size"] == 1
    assert constraints["sequence_parallel"] is False
    assert constraints["attention_dp"] is False
    assert constraints["mtp_enabled"] is False
    assert constraints["prefix_cache_enabled"] is False
    assert constraints["validation_state"] == "sglang_and_vllm_graph_off_pass"
    model_nodes = {
        f"{view_id}.{node['id']}"
        for view_id, view in model["views"].items()
        for node in view["nodes"]
    }
    inserted = []
    for transform in plan["transforms"]:
        target = transform.get("target", transform.get("after"))
        assert target in model_nodes
        if transform["op"] == "insert_after":
            inserted.append(transform["node"])
    assert inserted
    for node in inserted:
        assert node["node_kind"] == "communication"
        assert node["boundary_role"] in {"module_boundary", "module_internal"}
        assert node["execution"]["collective"]


def test_kimi_k3_candidate_plan_requires_validated_eager_reconciliation() -> None:
    pipeline = load_yaml(MODEL_ROOT / "pipeline.yaml")
    assert pipeline["execution"][
        "candidate_plan_requires_eager_reconciliation"
    ] is True
    expected_fingerprint = pipeline["execution"]["execution_fingerprint"]
    framework_targets = {
        target["framework"]: target
        for target in pipeline["source_lock"]["framework_targets"]
    }
    binding_paths = {
        "sglang": MODEL_ROOT / "bindings" / "sglang-25035bff-tp8.yaml",
        "vllm": MODEL_ROOT / "bindings" / "vllm-680e2177-tp8.yaml",
    }

    for framework, binding_path in binding_paths.items():
        target = framework_targets[framework]
        binding = load_yaml(binding_path)
        validation = binding["execution_validation"]
        assert target["eager_reconciliation"] == "passed"
        assert binding["source_commit"] == target["source_commit"]
        assert binding["binding_status"] == "validated"
        assert validation["status"] == "pass"
        assert validation["execution_fingerprint"] == expected_fingerprint
        assert validation["required_phases"] == ["prefill", "decode"]
        assert validation["cuda_graph_enabled"] is False
        assert validation["evidence"]


def test_kimi_k3_bindings_cover_all_model_and_execution_nodes() -> None:
    bundle = compile_catalog(MODEL_ROOT)
    required_nodes = {
        f"{view_id}.{node['id']}"
        for view_id, view in bundle["views"].items()
        for node in view["nodes"]
    }
    assert set(bundle["implementations"]) == {
        "sglang_25035bff_kimi_k3_tp8",
        "vllm_680e2177_kimi_k3_tp8",
    }
    expected_states = {
        "sglang_25035bff_kimi_k3_tp8": ("validated", "pass"),
        "vllm_680e2177_kimi_k3_tp8": ("validated", "pass"),
    }
    for implementation_id, implementation in bundle["implementations"].items():
        assert (
            implementation["binding_status"],
            implementation["execution_validation"]["status"],
        ) == expected_states[implementation_id]
        assert set(implementation["node_bindings"]) == required_nodes
        for target, binding in implementation["node_bindings"].items():
            assert binding["symbols"], target
            assert binding["code_links"], target


def test_kimi_k3_canonical_catalog_compile_is_deterministic() -> None:
    first = compile_catalog(MODEL_ROOT)
    second = compile_catalog(MODEL_ROOT)
    assert first == second
    assert first["meta"]["view_count"] == 9
    assert first["meta"]["execution_variant_count"] == 1
    assert first["meta"]["implementation_count"] == 2
    assert first["meta"]["profile_count"] == 8
    assert all(profile["meta"].get("timeline") for profile in first["profiles"].values())


def test_kimi_k3_runtime_drills_are_exact_production_event_unions() -> None:
    model = load_yaml(MODEL_ROOT / "model_ir.yaml")
    bundle = compile_catalog(MODEL_ROOT)
    expected_occurrences = {
        "top.decoder_stack": 186,
        "decoder_stack.kda": 69,
        "decoder_stack.gated_mla": 24,
        "decoder_stack.dense_mlp": 1,
        "decoder_stack.stable_latent_moe": 92,
    }
    scoped_prefixes = {
        "decoder_stack.kda": "kda.",
        "decoder_stack.gated_mla": "gated_mla.",
        "decoder_stack.dense_mlp": "dense_mlp.",
        "decoder_stack.stable_latent_moe": "stable_latent_moe.",
    }

    def decode(value: object, strings: list[str]) -> object:
        return strings[value] if isinstance(value, int) else value

    def union_us(intervals: list[tuple[float, float]]) -> float:
        merged: list[list[float]] = []
        for start, stop in sorted(intervals):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], stop)
            else:
                merged.append([start, stop])
        return sum(stop - start for start, stop in merged)

    for profile_path in sorted((MODEL_ROOT / "profiles").glob("*/*/*.yaml")):
        profile = load_yaml(profile_path)
        validate_executable_drill_rollups(model, profile)
        compiled = bundle["profiles"][profile["profile_id"]]
        timeline_path = profile_path.parent / profile["timeline"]["artifact"]
        with gzip.open(timeline_path, "rt") as source:
            timeline = json.load(source)
        strings = timeline["strings"]

        for target, expected_count in expected_occurrences.items():
            metric = profile["node_metrics"][target]
            assert target not in (profile.get("node_states") or {})
            assert metric["metric_kind"] == "inclusive_rollup"
            assert metric["attribution_status"] == "inclusive_rollup"
            assert compiled["data"][target][profile["variant_id"]][
                "timing_role"
            ] == "inclusive_rollup"

            event_ids: list[str] = []
            occurrences: set[str] = set()
            residency_us = 0.0
            active_us = 0.0
            direct_nodes: set[str] = set()
            for step in timeline["steps"]:
                step_intervals: list[tuple[float, float]] = []
                for event in step["events"]:
                    targets = {
                        str(decode(item, strings))
                        for item in event.get("ir_targets") or []
                    }
                    if target not in targets:
                        continue
                    start = float(event["start_us"])
                    duration = float(event["duration_us"])
                    step_intervals.append((start, start + duration))
                    residency_us += duration
                    event_ids.append(str(event["event_id"]))
                    occurrence = decode(event.get("occurrence_id"), strings)
                    if occurrence:
                        occurrences.add(str(occurrence))
                    direct = decode(event.get("ir_node"), strings)
                    if direct:
                        direct_nodes.add(str(direct))
                active_us += union_us(step_intervals)

            assert len(event_ids) == len(set(event_ids))
            assert metric["mapped_event_count"] == len(event_ids)
            assert metric["active_gpu_ms"] == pytest.approx(
                active_us / 1000.0, abs=1e-6
            )
            assert metric["gpu_residency_ms"] == pytest.approx(
                residency_us / 1000.0, abs=1e-6
            )
            assert metric["rollup_sources"] == sorted(direct_nodes)
            assert len(occurrences) == expected_count
            if target in scoped_prefixes:
                assert all(
                    node.startswith(scoped_prefixes[target])
                    for node in direct_nodes
                )


def test_kimi_k3_decode_gap_is_full_semantic_ideal_and_fail_closed() -> None:
    bundle = compile_catalog(MODEL_ROOT)
    sol = bundle["sol_profiles"][
        "kimi_k3_tp8_gb300_sglang_decode_bs1_8k1k_ideal_v1"
    ]
    gap = bundle["gap_reports"][
        "kimi_k3_tp8_gb300_sglang_decode_bs1_8k1k_gap_v1"
    ]
    measured = bundle["profiles"][sol["measured_profile_id"]]

    assert sol["status"] == "partial"
    assert gap["status"] == "partial_calibration"
    assert sol["critical_path"]["complete_step"] is True
    assert sol["coverage"] == gap["coverage"]
    assert sol["coverage"]["unsupported_targets"] == []
    assert sol["coverage"]["ideal_estimated_node_count"] == 38
    assert sol["coverage"]["observed_comparison_node_count"] == 38
    assert sol["coverage"]["structural_node_count"] == 76
    assert not gap["model_violations"]
    assert not gap["projection_violations"]
    assert sol["critical_path"]["attainable_critical_path_ms"] is None
    assert sol["critical_path"]["ideal_critical_path_ms"] < measured["meta"][
        "evidence"
    ]["timing"]["active_gpu_ms"]
    cost_ir_text = json.dumps(sol["cost_ir"], sort_keys=True).lower()
    assert "sglang" not in cost_ir_text
    assert "vllm" not in cost_ir_text
    assert "kernel" not in cost_ir_text


def test_kimi_k3_decay_projection_survives_async_inner_span_loss() -> None:
    stack = [
        FrameRef(raw="sglang/srt/models/kimi_k3.py(1644): forward"),
        FrameRef(raw="nn.Module: KimiK3DeltaAttention_7"),
        FrameRef(raw="nn.Module: KimiK3DecoderLayer_9"),
        FrameRef(raw="sglang/srt/model_executor/runner/eager_runner.py(222): _execute_decode"),
    ]
    node, confidence = classify_kimi_k3_node(
        "nvjet_sm103_tst_64x8_64x16_4x2_h_bz_TNT", "aten::mm", stack
    )
    assert node == "kda.decay_projection"
    assert confidence == "high"


def test_kimi_k3_vllm_native_decode_fusion_has_one_recurrent_owner() -> None:
    stack = [
        FrameRef(raw="vllm/models/kimi_k3/nvidia/kda.py(695): _forward"),
        FrameRef(raw="nn.Module: KimiK3DeltaAttention_9"),
        FrameRef(raw="vllm/v1/worker/gpu/model_runner.py(1504): execute_model"),
    ]
    node, confidence = classify_kimi_k3_vllm_node(
        "kda_decode_fusion_many_heads_kernel<12, 128>",
        "_C::fused_kda_decode",
        stack,
    )
    assert node == "kda.recurrent_update"
    assert confidence == "high"


@pytest.mark.parametrize(
    ("kernel", "expected"),
    [
        ("kda_gate_chunk_cumsum_vector_kernel", "kda.recurrent_update"),
        ("chunk_kda_fwd_kernel_inter_solve_fused", "kda.recurrent_update"),
        ("_recompute_w_u_fwd_kernel", "kda.recurrent_update"),
        ("chunk_gla_fwd_kernel_o", "kda.query_readout"),
    ],
)
def test_kimi_k3_vllm_triton_prefill_kda_owners(
    kernel: str, expected: str
) -> None:
    stack = [
        FrameRef(raw="vllm/models/kimi_k3/nvidia/kda.py(938): _forward"),
        FrameRef(raw="nn.Module: KimiK3DeltaAttention_9"),
        FrameRef(raw="vllm/v1/worker/gpu/model_runner.py(1504): execute_model"),
    ]
    node, confidence = classify_kimi_k3_vllm_node(kernel, None, stack)
    assert node == expected
    assert confidence == "high"


def test_kimi_k3_vllm_final_attn_res_is_distinct_from_layer_owners() -> None:
    stack = [
        FrameRef(raw="vllm/models/kimi_k3/nvidia/ops/attn_res.py(221): attn_res"),
        FrameRef(raw="vllm/models/kimi_k3/nvidia/model.py(1401): forward"),
        FrameRef(raw="nn.Module: KimiLinearModel_0"),
    ]
    node, confidence = classify_kimi_k3_vllm_node(
        "_attn_res_kernel", "aten::empty", stack
    )
    assert node == "top.output_attn_res"
    assert confidence == "high"


@pytest.mark.parametrize(
    ("kernel", "cpu_op", "frames", "expected"),
    [
        (
            "fused_q_kv_rmsnorm_kernel",
            "_C::fused_q_kv_rmsnorm",
            ["vllm/models/kimi_k3/nvidia/mla.py(529): _apply_q_lora_attention"],
            "gated_mla.q_norm",
        ),
        (
            "nvjet_sm103_gemm",
            "aten::mm",
            [
                "nn.Module: ReplicatedLinear_routed_expert_down_proj",
                "vllm/models/kimi_k3/nvidia/model.py(759): _maybe_overlap_router_and_down_proj",
            ],
            "stable_latent_moe.routed_down",
        ),
        (
            "nvjet_sm103_gemm",
            "aten::mm",
            [
                "nn.Module: MergedColumnParallelLinear_shared_experts.gate_up_proj",
                "vllm/models/kimi_k3/nvidia/latent_moe_runner.py(320): _fused_forward",
            ],
            "stable_latent_moe.shared_gate_up",
        ),
        (
            "latent_moe_tail_kernel",
            "_C::latent_moe_tail",
            [
                "vllm/models/kimi_k3/nvidia/latent_moe_runner.py(216): _small_batch_tail"
            ],
            "stable_latent_moe.tp_routed_latent_collective",
        ),
        (
            "adaptive_up_projection_kernel",
            None,
            [
                "vllm/models/kimi_k3/nvidia/ops/latent_moe_tail.py(203): __call__"
            ],
            "stable_latent_moe.tp_routed_latent_collective",
        ),
        (
            "ncclDevKernel_AllReduce_RING_LL",
            "record_param_comms",
            [
                "vllm/model_executor/layers/fused_moe/layer.py(988): _maybe_reduce_final_output",
                "vllm/models/kimi_k3/nvidia/latent_moe_runner.py(272): _shard_up_proj_tail",
            ],
            "stable_latent_moe.tp_shared_expert_collective",
        ),
        (
            "nvjet_sm103_gemm",
            "aten::addmm",
            [
                "vllm/models/kimi_k3/nvidia/latent_moe_runner.py(272): _shard_up_proj_tail"
            ],
            "stable_latent_moe.routed_up",
        ),
    ],
)
def test_kimi_k3_vllm_exact_fused_and_latent_moe_owners(
    kernel: str, cpu_op: str | None, frames: list[str], expected: str
) -> None:
    node, confidence = classify_kimi_k3_vllm_node(
        kernel, cpu_op, [FrameRef(raw=frame) for frame in frames]
    )
    assert node == expected
    assert confidence == "high"


@pytest.mark.parametrize(
    ("kernel", "frames", "expected"),
    [
        (
            "kernel_cutlass_model_executorkernelslinearcute_dsl_skinny_gemm",
            [
                "nn.Module: _KimiGDNMergedColumnParallelLinear_7",
                "vllm/models/kimi_k3/nvidia/kda.py(602): forward",
            ],
            "kda.qkv_projection",
        ),
        (
            "kernel_cutlass_model_executorkernelslinearcute_dsl_skinny_gemm",
            [
                "nn.Module: RowParallelLinear_7",
                "vllm/models/kimi_k3/nvidia/kda.py(602): forward",
            ],
            "kda.output_projection",
        ),
        (
            "kernel_cutlass_model_executorkernelslinearcute_dsl_skinny_gemm",
            [
                "nn.Module: ReplicatedLinear_7",
                "vllm/models/kimi_k3/nvidia/model.py(806): <lambda>",
                "nn.Module: KimiMoE_7",
            ],
            "stable_latent_moe.routed_down",
        ),
        (
            "void fused_a_gemm_kernel<1, 1536, 7168>",
            [
                "nn.Module: MergedColumnParallelLinear_shared_experts.gate_up_proj",
                "vllm/models/kimi_k3/nvidia/latent_moe_runner.py(320): _fused_forward",
            ],
            "stable_latent_moe.shared_gate_up",
        ),
        (
            "void fused_a_gemm_kernel<1, 2112, 7168>",
            [
                "vllm/models/kimi_k3/nvidia/mla.py(568): <lambda>",
                "vllm/models/kimi_k3/nvidia/mla.py(546): _forward_q_lora",
            ],
            "gated_mla.q_down",
        ),
        (
            "void fused_a_gemm_kernel<1, 2304, 1536>",
            [
                "vllm/models/kimi_k3/nvidia/mla.py(517): _apply_q_lora_attention",
                "vllm/models/kimi_k3/nvidia/mla.py(568): <lambda>",
            ],
            "gated_mla.q_up",
        ),
        (
            "void fused_a_gemm_kernel<1, 1536, 7168>",
            [
                "vllm/models/kimi_k3/nvidia/mla.py(573): <lambda>",
                "vllm/models/kimi_k3/nvidia/mla.py(546): _forward_q_lora",
            ],
            "gated_mla.kv_up",
        ),
        (
            "kernel_cutlass_kernel_ll_bf16_splitk",
            [
                "nn.Module: GateLinear_7",
                "vllm/models/kimi_k3/nvidia/model.py(780): _router",
            ],
            "stable_latent_moe.router_logits",
        ),
        (
            "kernel_cutlass_models_kimi_k3_nvidia_ops_cute_dsl_gemm_rs_ar",
            ["vllm/models/kimi_k3/nvidia/mla.py(620): forward"],
            "gated_mla.output_projection",
        ),
    ],
)
def test_kimi_k3_vllm_decode_low_latency_gemms_use_exact_source_owners(
    kernel: str, frames: list[str], expected: str
) -> None:
    node, confidence = classify_kimi_k3_vllm_node(
        kernel, None, [FrameRef(raw=frame) for frame in frames]
    )
    assert node == expected
    assert confidence == "high"


def test_kimi_k3_vllm_canonical_fusions_keep_one_measured_owner() -> None:
    rows = [
        {"node": "top.tp_logits_materialization", "event_id": "owner"},
        {"node": "top.logits", "event_id": "child"},
        {"node": "kda.tp_kda_output_collective", "event_id": "collective"},
    ]
    owned = apply_canonical_fusion_owners(rows)
    assert [row["node"] for row in owned] == [
        "top.tp_logits_materialization",
        "top.tp_logits_materialization",
        "kda.output_projection",
    ]
    assert owned[1]["semantic_child"] == "top.logits"
    assert owned[2]["semantic_child"] == "kda.tp_kda_output_collective"


def test_kimi_k3_sglang_profile_contract_has_one_owner_and_no_generic_gap() -> None:
    bundle = compile_catalog(MODEL_ROOT)
    required = {
        f"{view_id}.{node['id']}"
        for view_id, view in bundle["views"].items()
        for node in view["nodes"]
    }
    measured = {
        "top.embedding",
        "top.output_attn_res",
        "top.lm_head",
        "top.logits",
        "top.tp_embedding_output_collective",
        "top.tp_logits_materialization",
        "attn_res.weighted_merge",
        "decoder_stack.prefix_after_ffn",
        "dense_mlp.gate_up",
        "dense_mlp.situ",
        "dense_mlp.down",
        "dense_mlp.tp_dense_output_collective",
        "kda.qkv_projection",
        "kda.output_projection",
        "kda.recurrent_update",
        "kda.tp_kda_output_collective",
        "gated_mla.q_down",
        "gated_mla.q_norm",
        "gated_mla.q_up",
        "gated_mla.kv_norm",
        "gated_mla.attention",
        "gated_mla.output_gate",
        "gated_mla.gated_context",
        "gated_mla.output_projection",
        "gated_mla.tp_mla_output_collective",
        "stable_latent_moe.router_logits",
        "stable_latent_moe.corrected_selection",
        "stable_latent_moe.dispatch",
        "stable_latent_moe.expert_gate_up",
        "stable_latent_moe.expert_down",
        "stable_latent_moe.tp_routed_latent_collective",
        "stable_latent_moe.routed_up",
        "stable_latent_moe.shared_situ",
        "stable_latent_moe.shared_down",
        "stable_latent_moe.tp_shared_expert_collective",
        "stable_latent_moe.combine",
    }
    groups = sglang_fusion_groups(
        phase="decode", batch_size=64, measured_nodes=measured
    )
    states = build_node_states(
        required_nodes=required,
        measured_nodes=measured,
        fusion_groups=groups,
    )
    assert required == measured | set(states)
    assert not {
        state["status"]
        for state in states.values()
    } & {"unmapped", "mapping_incomplete"}
    for target, state in states.items():
        if state["status"] != "fused":
            continue
        group = groups[state["fusion_group_id"]]
        assert state["included_in"] == group["owner"]
        assert target in group["ir_nodes"]
        assert target not in measured


def test_kimi_k3_fusion_contract_rejects_missing_owner_or_measured_child() -> None:
    with pytest.raises(ValueError, match="no measured owner interval"):
        build_node_states(
            required_nodes=["owner", "child"],
            measured_nodes=set(),
            fusion_groups={
                "bad_owner": {"owner": "owner", "ir_nodes": ["owner", "child"]}
            },
        )
    with pytest.raises(ValueError, match="independent measured interval"):
        build_node_states(
            required_nodes=["owner", "child"],
            measured_nodes={"owner", "child"},
            fusion_groups={
                "duplicate_timing": {
                    "owner": "owner",
                    "ir_nodes": ["owner", "child"],
                }
            },
        )


def test_kimi_k3_sglang_prefill_bounded_decay_has_one_recurrence_owner() -> None:
    groups = sglang_fusion_groups(
        phase="prefill",
        batch_size=1,
        measured_nodes={"kda.recurrent_update"},
    )
    group = groups["sglang_kda_prefill_safe_gate_update"]
    assert group["owner"] == "kda.recurrent_update"
    assert group["ir_nodes"] == [
        "kda.recurrent_update",
        "kda.lower_bounded_decay",
    ]


def test_kimi_k3_vllm_prefill_profile_contract_closes_without_generic_gap() -> None:
    bundle = compile_catalog(MODEL_ROOT)
    required = {
        f"{view_id}.{node['id']}"
        for view_id, view in bundle["views"].items()
        for node in view["nodes"]
    }
    measured = {
        "top.embedding",
        "top.output_attn_res",
        "top.final_norm",
        "top.lm_head",
        "top.tp_embedding_output_collective",
        "top.tp_logits_materialization",
        "attn_res.weighted_merge",
        "kda.qkv_projection",
        "kda.q_short_conv",
        "kda.k_short_conv",
        "kda.v_short_conv",
        "kda.qk_l2_norm",
        "kda.decay_projection",
        "kda.recurrent_update",
        "kda.query_readout",
        "kda.gated_rmsnorm",
        "kda.output_projection",
        "gated_mla.q_down",
        "gated_mla.q_norm",
        "gated_mla.q_up",
        "gated_mla.kv_up",
        "gated_mla.cache_update",
        "gated_mla.attention",
        "gated_mla.gated_context",
        "gated_mla.output_projection",
        "dense_mlp.gate_up",
        "dense_mlp.situ",
        "dense_mlp.down",
        "stable_latent_moe.router_logits",
        "stable_latent_moe.corrected_selection",
        "stable_latent_moe.routed_down",
        "stable_latent_moe.expert_gate_up",
        "stable_latent_moe.expert_down",
        "stable_latent_moe.weighted_reduce",
        "stable_latent_moe.tp_routed_latent_collective",
        "stable_latent_moe.routed_up",
        "stable_latent_moe.shared_gate_up",
        "stable_latent_moe.shared_down",
        "stable_latent_moe.tp_shared_expert_collective",
    }
    groups = vllm_fusion_groups(
        phase="prefill", batch_size=1, measured_nodes=measured
    )
    states = build_node_states(
        required_nodes=required,
        measured_nodes=measured,
        fusion_groups=groups,
    )
    assert required == measured | set(states)
    assert not {
        state["status"] for state in states.values()
    } & {"unmapped", "mapping_incomplete"}
    assert states["decoder_stack.prefix_after_ffn"]["status"] == "structural"
    assert "top.final_norm" not in states
    assert all(
        "top.final_norm" not in group["ir_nodes"] for group in groups.values()
    )
    assert states["stable_latent_moe.combine"]["included_in"] == (
        "stable_latent_moe.routed_up"
    )


def test_kimi_k3_vllm_decode_cutedsl_tail_has_one_measured_owner() -> None:
    bundle = compile_catalog(MODEL_ROOT)
    required = {
        f"{view_id}.{node['id']}"
        for view_id, view in bundle["views"].items()
        for node in view["nodes"]
    }
    measured = {
        "top.embedding",
        "top.output_attn_res",
        "top.final_norm",
        "top.lm_head",
        "top.tp_embedding_output_collective",
        "top.tp_logits_materialization",
        "attn_res.weighted_merge",
        "kda.qkv_projection",
        "kda.decay_projection",
        "kda.recurrent_update",
        "kda.output_projection",
        "gated_mla.q_down",
        "gated_mla.q_norm",
        "gated_mla.q_up",
        "gated_mla.cache_update",
        "gated_mla.attention",
        "gated_mla.gated_context",
        "gated_mla.output_projection",
        "dense_mlp.gate_up",
        "dense_mlp.situ",
        "dense_mlp.down",
        "stable_latent_moe.router_logits",
        "stable_latent_moe.corrected_selection",
        "stable_latent_moe.routed_down",
        "stable_latent_moe.expert_gate_up",
        "stable_latent_moe.expert_down",
        "stable_latent_moe.tp_routed_latent_collective",
        "stable_latent_moe.shared_gate_up",
        "stable_latent_moe.shared_down",
    }
    groups = vllm_fusion_groups(
        phase="decode", batch_size=64, measured_nodes=measured
    )
    states = build_node_states(
        required_nodes=required,
        measured_nodes=measured,
        fusion_groups=groups,
    )
    tail = groups["vllm_moe_latent_tail_bundle"]
    assert tail["owner"] == "stable_latent_moe.tp_routed_latent_collective"
    for child in (
        "stable_latent_moe.weighted_reduce",
        "stable_latent_moe.latent_norm",
        "stable_latent_moe.routed_up",
        "stable_latent_moe.tp_shared_expert_collective",
        "stable_latent_moe.combine",
    ):
        assert states[child]["included_in"] == tail["owner"]


def test_kimi_k3_production_window_uses_leader_api_and_worker_launch_correlation(
    tmp_path: Path,
) -> None:
    sqlite_path = tmp_path / "node.sqlite"
    connection = sqlite3.connect(sqlite_path)
    global_pid = 123 << 24
    try:
        connection.executescript(
            """
            CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
                start INTEGER, end INTEGER, globalTid INTEGER, nameId INTEGER,
                correlationId INTEGER
            );
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                start INTEGER, end INTEGER, deviceId INTEGER, streamId INTEGER,
                demangledName INTEGER, graphNodeId INTEGER, correlationId INTEGER,
                gridId INTEGER, globalPid INTEGER
            );
            """
        )
        connection.executemany(
            "INSERT INTO StringIds(id, value) VALUES (?, ?)",
            [(1, "cuProfilerStart"), (2, "cuProfilerStop"), (3, "kernel")],
        )
        connection.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?, ?)",
            [
                (100, 110, global_pid + 7, 1, 1001),
                (115, 116, global_pid + 8, 3, 2),
                (145, 146, global_pid + 8, 3, 3),
                (200, 210, global_pid + 7, 2, 1002),
            ],
        )
        connection.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (90, 99, 0, 1, 3, None, 1, 1, global_pid),
                (120, 130, 0, 1, 3, 9, 2, 2, global_pid),
                (150, 180, 0, 2, 3, 10, 3, 3, global_pid),
                (201, 205, 0, 1, 3, None, 4, 4, global_pid),
            ],
        )
        connection.commit()
    finally:
        connection.close()

    kernels, audit = read_exact_device_kernels(sqlite_path, 0)
    assert [row["grid_id"] for row in kernels] == [2, 3]
    assert audit["node_leader_profiler_start_api_count"] == 1
    assert audit["node_leader_profiler_stop_api_count"] == 1
    assert audit["launch_correlated_exact_window_kernel_count"] == 2
    assert audit["gpu_execution_after_profiler_stop_kernel_count"] == 0
    assert audit["node_collection_kernel_count"] == 4


def test_kimi_k3_production_window_retains_async_gpu_work_launched_before_stop(
    tmp_path: Path,
) -> None:
    sqlite_path = tmp_path / "async.sqlite"
    connection = sqlite3.connect(sqlite_path)
    leader_pid = 123 << 24
    worker_pid = 456 << 24
    try:
        connection.executescript(
            """
            CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
                start INTEGER, end INTEGER, globalTid INTEGER, nameId INTEGER,
                correlationId INTEGER
            );
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                start INTEGER, end INTEGER, deviceId INTEGER, streamId INTEGER,
                demangledName INTEGER, graphNodeId INTEGER, correlationId INTEGER,
                gridId INTEGER, globalPid INTEGER
            );
            """
        )
        connection.executemany(
            "INSERT INTO StringIds(id, value) VALUES (?, ?)",
            [(1, "cuProfilerStart"), (2, "cuProfilerStop"), (3, "kernel")],
        )
        connection.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?, ?)",
            [
                (100, 110, leader_pid + 7, 1, 1001),
                (190, 195, worker_pid + 8, 3, 42),
                (200, 210, leader_pid + 7, 2, 1002),
            ],
        )
        connection.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (205, 230, 0, 1, 3, 9, 42, 1, worker_pid),
        )
        connection.commit()
    finally:
        connection.close()

    kernels, audit = read_exact_device_kernels(sqlite_path, 0)
    assert [row["grid_id"] for row in kernels] == [1]
    assert audit["launch_correlated_exact_window_kernel_count"] == 1
    assert audit["gpu_execution_after_profiler_stop_kernel_count"] == 1


def test_kimi_k3_vllm_production_window_is_worker_local_and_launch_correlated(
    tmp_path: Path,
) -> None:
    sqlite_path = tmp_path / "vllm-worker.sqlite"
    connection = sqlite3.connect(sqlite_path)
    worker_pid = 456 << 24
    other_pid = 789 << 24
    try:
        connection.executescript(
            """
            CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
                start INTEGER, end INTEGER, globalTid INTEGER, nameId INTEGER,
                correlationId INTEGER
            );
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                start INTEGER, end INTEGER, deviceId INTEGER, streamId INTEGER,
                demangledName INTEGER, graphNodeId INTEGER, correlationId INTEGER,
                gridId INTEGER, globalPid INTEGER
            );
            """
        )
        connection.executemany(
            "INSERT INTO StringIds(id, value) VALUES (?, ?)",
            [(1, "cudaProfilerStart_v4000"), (2, "cudaProfilerStop_v4000"), (3, "kernel")],
        )
        connection.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?, ?)",
            [
                (100, 110, worker_pid + 7, 1, 1001),
                (120, 125, worker_pid + 8, 3, 41),
                (190, 195, worker_pid + 8, 3, 42),
                (200, 210, worker_pid + 7, 2, 1002),
                (120, 125, other_pid + 8, 3, 43),
            ],
        )
        connection.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (130, 140, 0, 1, 3, 9, 41, 1, worker_pid),
                (205, 230, 0, 1, 3, None, 42, 2, worker_pid),
                (130, 140, 1, 1, 3, 9, 43, 3, other_pid),
            ],
        )
        connection.commit()
    finally:
        connection.close()

    kernels, audit = read_exact_worker_kernels(sqlite_path, 0)
    assert [row["grid_id"] for row in kernels] == [1, 2]
    assert audit["worker_profiler_start_api_count"] == 1
    assert audit["worker_profiler_stop_api_count"] == 1
    assert audit["launch_correlated_exact_window_kernel_count"] == 2
    assert audit["gpu_execution_after_profiler_stop_kernel_count"] == 1
    assert audit["node_collection_kernel_count"] == 2


def test_kimi_k3_attn_res_segments_encode_the_exact_93_layer_schedule() -> None:
    rows = [
        {"kernel_name": "prefix"},
        *(
            {"kernel_name": "sglang::attn_res_fused_tma_kernel<int>"}
            for _ in range(ATTN_RES_ANCHOR_COUNT)
        ),
        {"kernel_name": "tail"},
    ]
    segments = anchor_segments(rows)
    assert len(segments) == 187
    assert occurrence_for_segment(0)["occurrence_id"] == "layer_00.attention"
    assert occurrence_for_segment(1)["occurrence_id"] == "layer_00.feed_forward"
    assert occurrence_for_segment(2)["occurrence_id"] == "layer_01.attention"
    assert occurrence_for_segment(184)["occurrence_id"] == "layer_92.attention"
    assert occurrence_for_segment(185)["occurrence_id"] == "layer_92.feed_forward"
    assert occurrence_for_segment(186)["occurrence_id"] == "final_output"


def test_kimi_k3_vllm_attn_res_segments_encode_two_calls_per_layer() -> None:
    rows = [
        {"kernel_name": "runtime prefix"},
        *(
            {"kernel_name": "vllm::_attn_res_kernel<int>"}
            for _ in range(VLLM_ATTN_RES_ANCHOR_COUNT)
        ),
        {"kernel_name": "sampling tail"},
    ]
    segments = vllm_anchor_segments(rows)
    assert len(segments) == 188
    assert vllm_occurrence_for_segment(0)["occurrence_id"] == "runtime.step_setup"
    assert vllm_occurrence_for_segment(1)["occurrence_id"] == "layer_00.attention"
    assert vllm_occurrence_for_segment(2)["occurrence_id"] == "layer_00.feed_forward"
    assert vllm_occurrence_for_segment(185)["occurrence_id"] == "layer_92.attention"
    assert vllm_occurrence_for_segment(186)["occurrence_id"] == "layer_92.feed_forward"
    assert vllm_occurrence_for_segment(187)["occurrence_id"] == "final_output"
