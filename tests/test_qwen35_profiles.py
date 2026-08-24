from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path

import yaml

from models.qwen35.profile.qwen35_timeline import QWEN35_TIMELINE_TARGETS


REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG_ROOT = REPO_ROOT / "catalog" / "qwen35"
PROFILE_ROOT = CATALOG_ROOT / "profiles" / "attention_dp4_moe_ep4"
EXPECTED_EXECUTION_PARAMETERS = {
    "tp_size": 1,
    "dp_size": 4,
    "cp_size": 1,
    "ep_size": 4,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _profiles():
    for path in sorted(PROFILE_ROOT.glob("*/*.yaml")):
        yield path, yaml.safe_load(path.read_text())


def _decode_strings(timeline: dict, values: list[int | None]) -> set[str]:
    strings = timeline["strings"]
    return {strings[value] for value in values if value is not None}


def _timeline_targets(timeline: dict) -> set[str]:
    indexes = []
    for step in timeline["steps"]:
        for event in step["events"]:
            indexes.append(event["ir_node"])
            indexes.extend(event["ir_targets"])
    return _decode_strings(timeline, indexes)


def test_all_official_profiles_are_mtp_dep4_and_content_addressed():
    profiles = list(_profiles())
    assert len(profiles) == 9
    for path, profile in profiles:
        assert profile["generation_mode"] == "mtp"
        assert profile["execution_parameters"] == EXPECTED_EXECUTION_PARAMETERS
        evidence = profile["evidence"]
        assert len(evidence["model_revision"]) == 40
        assert len(evidence["model_config_sha256"]) == 64
        assert len(evidence["container_sha256"]) == 64
        threshold = 0.95 if profile["implementation_id"].startswith("sglang_") else 0.90
        gate = evidence["semantic_attribution_gate"]
        metric = gate.get("metric", "mapped_or_fusion_duration_ratio")
        assert gate["threshold"] == threshold
        assert gate["passed"] == (evidence[metric] >= threshold)
        assert math.isclose(
            evidence["mapped_duration_ratio"]
            + evidence["fusion_duration_ratio"]
            + evidence["unmapped_duration_ratio"],
            1.0,
            abs_tol=2e-6,
        )
        assert math.isclose(
            evidence["timeline_interval_coverage_ratio"], 1.0, abs_tol=1e-6
        )

        timeline_path = path.parent / profile["timeline"]["artifact"]
        assert _sha256(timeline_path) == profile["timeline"]["sha256"]
        timeline = json.loads(gzip.decompress(timeline_path.read_bytes()))
        assert timeline["profile_id"] == profile["profile_id"]
        assert len(timeline["steps"]) == profile["timeline"]["step_count"]
        assert (
            sum(len(step["events"]) for step in timeline["steps"])
            == profile["timeline"]["event_count"]
        )


def test_semantic_kernel_aggregates_retain_concrete_profiler_symbols():
    for path, profile in _profiles():
        timeline = json.loads(
            gzip.decompress(
                (path.parent / profile["timeline"]["artifact"]).read_bytes()
            )
        )
        strings = timeline["strings"]
        concrete_by_semantic: dict[tuple[str, str], set[str]] = {}
        for step in timeline["steps"]:
            for event in step["events"]:
                if event["ir_node"] is None or event["kernel_label"] is None:
                    continue
                key = (strings[event["ir_node"]], strings[event["kernel_label"]])
                concrete_by_semantic.setdefault(key, set()).add(
                    strings[event["kernel_name"]]
                )

        for target, metric in profile.get("node_metrics", {}).items():
            for kernel in metric.get("kernels", []):
                concrete = concrete_by_semantic.get((target, kernel["name"]), set())
                assert concrete, (profile["profile_id"], target, kernel["name"])
                assert all(name.strip() for name in concrete)


def test_trt_decode_comparison_profile_uses_exact_bs32_torch_subset():
    path = (
        PROFILE_ROOT / "trtllm_1cef02e9_attention_dp4_moe_ep4_mtp" / "decode_bs32.yaml"
    )
    profile = yaml.safe_load(path.read_text())
    assert profile["profiler"]["type"] == "torch_kineto_worker_local"
    assert profile["profiler"]["activities"] == ["CPU", "CUDA"]
    assert profile["profiler"]["with_stack"] is True
    assert profile["profiler"]["record_shapes"] is False
    shape = profile["workload"]["measured_shape"]
    assert shape["selected_exact_generation_requests"] == 32
    assert shape["selected_samples"] == sum(
        shape["selected_samples_by_source"].values()
    )
    assert set(shape["selected_samples_by_source"]) == {
        "nvl72d131-T07/rank0",
        "nvl72d131-T08/rank0",
    }
    assert shape["selected_samples"] == 10
    assert set(shape["selected_samples_by_source"].values()) == {5}
    assert profile["evidence"]["selection_policy"] == (
        "exactly 10 real generation_reqs=32 rank-local events, balanced 5 per "
        "runtime-elected worker source and time-spread over each capture"
    )
    assert profile["workload"]["comparison_contract"]["profiler"] == "torch_kineto"
    assert len(profile["evidence"]["report_files"]) == 8
    assert all(
        report["format"] == "PyTorch profiler trace JSON"
        for report in profile["evidence"]["report_files"]
    )
    assert len(profile["evidence"]["exact_worker_logs"]) == 2


def test_sglang_and_trt_decode_comparison_profiles_are_exact_batch_torch_peers():
    sglang_paths = sorted(
        (PROFILE_ROOT / "sglang_85c23c62_attention_dp4_moe_ep4_mtp").glob(
            "agentx_torch_bs*.yaml"
        )
    )
    assert len(sglang_paths) == 1
    sglang = yaml.safe_load(sglang_paths[0].read_text())
    trt = yaml.safe_load(
        (
            PROFILE_ROOT
            / "trtllm_1cef02e9_attention_dp4_moe_ep4_mtp"
            / "decode_bs32.yaml"
        ).read_text()
    )

    selected_batch = sglang["workload"]["selected_exact_target_verify_batch"]
    assert (
        selected_batch
        == trt["workload"]["measured_shape"]["selected_exact_generation_requests"]
    )
    assert (
        sglang["profiler"]["type"]
        == trt["profiler"]["type"]
        == ("torch_kineto_worker_local")
    )
    assert sglang["profiler"]["activities"] == trt["profiler"]["activities"] == [
        "CPU",
        "CUDA",
    ]
    assert sglang["profiler"]["with_stack"] is trt["profiler"]["with_stack"] is True
    assert (
        sglang["profiler"]["record_shapes"]
        is trt["profiler"]["record_shapes"]
        is False
    )
    assert sglang["workload"]["selected_samples"] == sum(
        sglang["workload"]["selected_samples_by_source"].values()
    )
    assert sglang["workload"]["selected_samples"] == 10
    assert sglang["workload"]["selected_samples_by_source"] == {
        "w0/r3": 5,
        "w1/r3": 5,
    }
    assert sglang["workload"]["accepted_length"]["selected_histogram"] == {
        4.0: 2,
        5.0: 8,
    }
    assert sglang["workload"]["accepted_length"]["selected_mean"] == 4.8
    assert (
        sglang["workload"]["comparison_contract"]
        == trt["workload"]["comparison_contract"]
    )
    evidence = sglang["evidence"]
    assert evidence["job_id"] == 3270073
    assert evidence["profiling_source_commit"] == (
        "049323294ec36631b9aab74ffa5dac5ff020fdae"
    )
    assert evidence["profiling_harness_commit"] == (
        "fb4476ed42399ca6a160565e1e6a7bb864b9a015"
    )
    assert evidence["profiler_manager_sha256"] == (
        "e08fd6430c83bffbc47e83cdd8a770891c8208b3fff6126627bc231d8e338eb9"
    )
    assert evidence["runtime_manifest_sha256"] == (
        "f23af95628fe593cc6d1c140bc3c4a040ddbd224b94c38a7fdaa9b2c8fa46d41"
    )
    assert len(evidence["trace_files"]) == 8
    assert len(evidence["worker_logs"]) == 2
    assert all(
        trace["kernel_count"] > 0
        and trace["concrete_kernel_name_count"] > 0
        and trace["complete_mtp_window_count"] == 64
        for trace in evidence["trace_files"]
    )
    assert evidence["all_eight_rank_traces_validated"] is True
    assert evidence["semantic_attribution_gate"] == {
        "metric": "mapped_or_fusion_active_union_ratio",
        "threshold": 0.95,
        "passed": True,
    }


def test_trt_prefill_profile_uses_only_exact_one_by_8k_nsys_subset():
    path = (
        PROFILE_ROOT / "trtllm_1cef02e9_attention_dp4_moe_ep4_mtp" / "prefill_8k.yaml"
    )
    profile = yaml.safe_load(path.read_text())
    shape = profile["workload"]["measured_shape"]
    assert shape["selected_exact_context_shape"] == {
        "requests": 1,
        "tokens": 8192,
    }
    assert shape["selected_samples"] == sum(
        shape["selected_samples_by_source"].values()
    )
    assert profile["evidence"]["selection_policy"] == (
        "exact one-request/8192-token owner events only"
    )


def test_timeline_timing_closes_and_every_target_is_a_real_ir_node():
    bundle = json.loads(
        (REPO_ROOT / "docs" / "qwen35_v2" / "arch_data.json").read_text()
    )
    variant = next(iter(bundle["execution_variants"].values()))
    valid_nodes = {
        f"{view_id}.{node['id']}"
        for view_id, view in variant["views"].items()
        for node in view["nodes"]
    }
    for path, profile in _profiles():
        timeline = json.loads(
            gzip.decompress(
                (path.parent / profile["timeline"]["artifact"]).read_bytes()
            )
        )
        assert _timeline_targets(timeline) <= valid_nodes
        for step in timeline["steps"]:
            elapsed = float(step["duration_us"])
            active = float(step["active_gpu_us"])
            residency = float(step["gpu_residency_us"])
            assert active <= elapsed + 1e-6
            assert active <= residency + 1e-6
            assert math.isclose(step["device_gap_us"], elapsed - active, abs_tol=2e-6)
            assert math.isclose(
                step["gpu_overlap_us"], residency - active, abs_tol=2e-6
            )
            assert math.isclose(
                sum(float(item["duration_us"]) for item in step["idle_intervals"]),
                step["device_gap_us"],
                # Each interval is serialized independently to six decimal
                # places, so accumulated sub-nanosecond rounding grows with
                # thousands of gaps in a formal step.
                abs_tol=1e-3,
            )
            for node in step["node_timings"]:
                node_elapsed = float(node["elapsed_us"])
                node_active = float(node["active_gpu_us"])
                node_residency = float(node["gpu_residency_us"])
                assert node_active <= node_elapsed + 1e-2
                assert node_active <= node_residency + 2e-2
                assert math.isclose(
                    node["gpu_overlap_us"], node_residency - node_active, abs_tol=2e-2
                )
                assert math.isclose(
                    node["module_gap_us"], node_elapsed - node_active, abs_tol=2e-2
                )
                assert node["other_gpu_work_us"] >= 0


def test_timeline_targets_are_closed_over_visible_architecture_ancestors():
    model_ir = yaml.safe_load((CATALOG_ROOT / "model_ir.yaml").read_text())
    resolver = QWEN35_TIMELINE_TARGETS
    bundle = json.loads(
        (REPO_ROOT / "docs" / "qwen35_v2" / "arch_data.json").read_text()
    )
    variant = next(iter(bundle["execution_variants"].values()))
    valid_nodes = {
        f"{view_id}.{node['id']}"
        for view_id, view in variant["views"].items()
        for node in view["nodes"]
    }

    for path, profile in _profiles():
        timeline = json.loads(
            gzip.decompress(
                (path.parent / profile["timeline"]["artifact"]).read_bytes()
            )
        )
        strings = timeline["strings"]
        profile_targets: set[str] = set()
        for step in timeline["steps"]:
            timing_targets = {strings[item["ir_node"]] for item in step["node_timings"]}
            for event in step["events"]:
                direct = (
                    strings[event["ir_node"]] if event["ir_node"] is not None else ""
                )
                targets = {strings[index] for index in event["ir_targets"]}
                profile_targets.update(targets)
                expected = set(resolver({"node": direct, "ir_targets": list(targets)}))
                assert expected <= targets
                assert targets <= timing_targets
                assert not any(
                    target.startswith("layer_schedule.layer_") for target in targets
                )

                status = strings[event["mapping_status"]]
                assert status in {"mapped", "fusion", "unmapped"}
                if status == "unmapped":
                    assert not direct
                    assert event["unmapped_reason"] is not None
                    candidates = {strings[index] for index in event["candidate_nodes"]}
                    assert candidates
                    assert candidates <= valid_nodes
                else:
                    assert direct in valid_nodes
                if status == "fusion":
                    assert event["fusion_group"] is not None
                    assert strings[event["attribution_method"]] in {
                        "kernel_signature_fusion",
                        "validated_graph_sequence_fusion",
                    }
                    assert len(targets) >= 2

        if any(target.startswith("gdn_attention.") for target in profile_targets):
            assert {
                "gdn_moe_block.attention",
                "stack.gdn_layer",
                "top.decoder_stack",
            } <= profile_targets
        if any(target.startswith("full_attention.") for target in profile_targets):
            assert {
                "full_attention_moe_block.attention",
                "stack.full_attention_layer",
                "top.decoder_stack",
            } <= profile_targets
        if any(target.startswith("moe_block.") for target in profile_targets):
            assert {
                "gdn_moe_block.moe",
                "full_attention_moe_block.moe",
            } & profile_targets


def test_architecture_exposes_timeline_rollups_on_drill_modules():
    bundle = json.loads(
        (REPO_ROOT / "docs" / "qwen35_v2" / "arch_data.json").read_text()
    )
    profile_id = bundle["default_profile"]
    assert profile_id == ("qwen35_sglang_attention_dp4_moe_ep4_mtp6_agentx_torch_bs32")
    variant_id = bundle["profiles"][profile_id]["meta"]["variant_id"]
    assert variant_id == ("sglang_agentx_8k1k_c704_3p2d_dep4_mtp6_cg_torch_bs32")
    expected_targets = {
        "top.decoder_stack",
        "stack.gdn_layer",
        "stack.full_attention_layer",
        "gdn_moe_block.attention",
        "gdn_moe_block.moe",
        "full_attention_moe_block.attention",
        "full_attention_moe_block.moe",
    }
    for target in expected_targets:
        view_id, node_id = target.split(".", 1)
        cell = bundle["enriched"][view_id]["nodes_profile"][node_id][profile_id][
            variant_id
        ]
        assert cell["attribution_status"] == "inclusive_rollup"
        assert cell["active_gpu_ms"] > 0
        assert cell["gpu_elapsed_ms"] >= cell["active_gpu_ms"]
        assert cell["gpu_residency_ms"] >= cell["active_gpu_ms"]
        assert cell["timeline_observed_step_count"] == cell["timeline_step_count"]


def test_decode_timelines_expose_lifecycle_and_preserve_capture_boundaries():
    by_engine = {}
    for path, profile in _profiles():
        if profile["phase"] != "decode" or "eager" in profile["profile_id"]:
            continue
        timeline = json.loads(
            gzip.decompress(
                (path.parent / profile["timeline"]["artifact"]).read_bytes()
            )
        )
        targets = _timeline_targets(timeline)
        if profile["implementation_id"].startswith("sglang_"):
            by_engine.setdefault("sglang", []).append((profile, targets))
        else:
            by_engine.setdefault("trtllm", []).append((profile, targets))

    for profile, targets in by_engine["sglang"]:
        assert {
            "generation_loop.target_verify",
            "generation_loop.draft_propose",
            "generation_loop.accept_prefix",
            "generation_loop.replay_gdn",
            "generation_loop.commit_gdn",
            "generation_loop.commit_tokens",
        } <= targets
        assert (
            profile["node_states"]["generation_loop.commit_kv"]["status"]
            == "unobserved"
        )

    assert len(by_engine["trtllm"]) == 1
    trt_profile, trt_targets = by_engine["trtllm"][0]
    assert {
        "generation_loop.target_verify",
        "generation_loop.draft_propose",
        "generation_loop.commit_gdn",
    } <= trt_targets
    assert (
        trt_profile["node_states"]["generation_loop.commit_kv"]["status"]
        == "unobserved"
    )
    assert (
        trt_profile["node_states"]["generation_loop.accept_prefix"]["status"]
        == "unobserved"
    )
    assert (
        trt_profile["node_states"]["generation_loop.replay_gdn"]["status"]
        == "unobserved"
    )
