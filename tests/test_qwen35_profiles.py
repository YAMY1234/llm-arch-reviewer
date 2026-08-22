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
    assert len(profiles) == 7
    for path, profile in profiles:
        assert profile["generation_mode"] == "mtp"
        assert profile["execution_parameters"] == EXPECTED_EXECUTION_PARAMETERS
        evidence = profile["evidence"]
        assert len(evidence["model_revision"]) == 40
        assert len(evidence["model_config_sha256"]) == 64
        assert len(evidence["container_sha256"]) == 64
        threshold = 0.95 if profile["implementation_id"].startswith("sglang_") else 0.90
        assert evidence["semantic_attribution_gate"] == {
            "threshold": threshold,
            "passed": evidence["mapped_or_fusion_duration_ratio"] >= threshold,
        }
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
        assert sum(len(step["events"]) for step in timeline["steps"]) == profile[
            "timeline"
        ]["event_count"]


def test_timeline_timing_closes_and_every_target_is_a_real_ir_node():
    bundle = json.loads((REPO_ROOT / "docs" / "qwen35_v2" / "arch_data.json").read_text())
    variant = next(iter(bundle["execution_variants"].values()))
    valid_nodes = {
        f"{view_id}.{node['id']}"
        for view_id, view in variant["views"].items()
        for node in view["nodes"]
    }
    for path, profile in _profiles():
        timeline = json.loads(
            gzip.decompress((path.parent / profile["timeline"]["artifact"]).read_bytes())
        )
        assert _timeline_targets(timeline) <= valid_nodes
        for step in timeline["steps"]:
            elapsed = float(step["duration_us"])
            active = float(step["active_gpu_us"])
            residency = float(step["gpu_residency_us"])
            assert active <= elapsed + 1e-6
            assert active <= residency + 1e-6
            assert math.isclose(step["device_gap_us"], elapsed - active, abs_tol=2e-6)
            assert math.isclose(step["gpu_overlap_us"], residency - active, abs_tol=2e-6)
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
                assert node_active <= node_residency + 1e-2
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
    bundle = json.loads((REPO_ROOT / "docs" / "qwen35_v2" / "arch_data.json").read_text())
    variant = next(iter(bundle["execution_variants"].values()))
    valid_nodes = {
        f"{view_id}.{node['id']}"
        for view_id, view in variant["views"].items()
        for node in view["nodes"]
    }

    for path, profile in _profiles():
        timeline = json.loads(
            gzip.decompress((path.parent / profile["timeline"]["artifact"]).read_bytes())
        )
        strings = timeline["strings"]
        profile_targets: set[str] = set()
        for step in timeline["steps"]:
            timing_targets = {
                strings[item["ir_node"]] for item in step["node_timings"]
            }
            for event in step["events"]:
                direct = strings[event["ir_node"]] if event["ir_node"] is not None else ""
                targets = {strings[index] for index in event["ir_targets"]}
                profile_targets.update(targets)
                expected = set(resolver({"node": direct, "ir_targets": list(targets)}))
                assert expected <= targets
                assert targets <= timing_targets
                assert not any(target.startswith("layer_schedule.layer_") for target in targets)

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
            assert {"gdn_moe_block.attention", "stack.gdn_layer", "top.decoder_stack"} <= profile_targets
        if any(target.startswith("full_attention.") for target in profile_targets):
            assert {"full_attention_moe_block.attention", "stack.full_attention_layer", "top.decoder_stack"} <= profile_targets
        if any(target.startswith("moe_block.") for target in profile_targets):
            assert {"gdn_moe_block.moe", "full_attention_moe_block.moe"} & profile_targets


def test_decode_timelines_expose_lifecycle_and_preserve_capture_boundaries():
    by_engine = {}
    for path, profile in _profiles():
        if profile["phase"] != "decode" or "eager" in profile["profile_id"]:
            continue
        timeline = json.loads(
            gzip.decompress((path.parent / profile["timeline"]["artifact"]).read_bytes())
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
        assert profile["node_states"]["generation_loop.commit_kv"]["status"] == "unobserved"

    assert len(by_engine["trtllm"]) == 1
    trt_profile, trt_targets = by_engine["trtllm"][0]
    assert {
        "generation_loop.target_verify",
        "generation_loop.draft_propose",
        "generation_loop.commit_kv",
        "generation_loop.commit_gdn",
    } <= trt_targets
    assert trt_profile["node_states"]["generation_loop.accept_prefix"]["status"] == "unobserved"
    assert trt_profile["node_states"]["generation_loop.replay_gdn"]["status"] == "unobserved"
