from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_CASES = {
    "vllm": (
        REPO_ROOT
        / "catalog/deepseek_v4_pro/profiles/tp8/vllm_dd10e03_dsv4pro0813_tp8",
        "dd10e03f95f94edbea1975c67ace3a35ec9a8a40",
    ),
    "sglang": (
        REPO_ROOT
        / "catalog/deepseek_v4_pro/profiles/tp8/sglang_71de97b_dsv4pro0813_tp8",
        "71de97b264b04dcd514cf904003028aefe9775c8",
    ),
}
EXPECTED = {
    "eager_prefill_gbs001_8k.yaml": ("prefill", 1, False),
    "cg_decode_gbs001_8k1k.yaml": ("decode", 1, True),
    "cg_decode_gbs016_8k1k.yaml": ("decode", 16, True),
    "cg_decode_gbs064_8k1k.yaml": ("decode", 64, True),
    "cg_decode_gbs256_8k1k.yaml": ("decode", 256, True),
}
MODEL_REVISION = "72e1d3230f6c080a530b0a1d46f8eb4602340597"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _profiles(root: Path) -> list[tuple[Path, dict]]:
    paths = sorted(root.glob("*.yaml"))
    expected = set(EXPECTED)
    if root.name.startswith("sglang_"):
        expected.remove("eager_prefill_gbs001_8k.yaml")
    assert {path.name for path in paths} == expected
    return [(path, yaml.safe_load(path.read_text())) for path in paths]


def test_profile_matrices_are_exact_and_source_locked() -> None:
    for framework, (root, source_commit) in PROFILE_CASES.items():
        for path, profile in _profiles(root):
            phase, batch_size, graph_on = EXPECTED[path.name]
            assert profile["phase"] == phase
            assert profile["profile_id"].startswith(
                f"deepseek_v4_pro_tp8_{framework}_"
            )
            assert profile["workload"]["batch_size"] == batch_size
            assert profile["workload"]["isl"] == 8192
            assert profile["workload"]["osl"] == 1024
            assert profile["workload"]["warmup_requests"] == 3 * batch_size
            assert profile["workload"]["formal_requests"] == batch_size
            assert profile["execution_parameters"] == {
                "tp_size": 8,
                "dp_size": 1,
                "cp_size": 1,
                "ep_size": 1,
            }
            assert profile["profiler"]["cuda_graph_enabled"] is graph_on
            assert profile["profiler"]["all_tp_ranks_validated"] is True
            assert profile["evidence"]["rank_count"] == 8
            assert len(profile["evidence"]["rank_reconciliation_fingerprints"]) == 8
            assert profile["evidence"]["model_revision"] == MODEL_REVISION
            assert profile["evidence"]["source_commit"] == source_commit
            assert profile["evidence"]["mapped_kernel_count_ratio"] == 1.0
            assert profile["evidence"]["mapped_kernel_duration_ratio"] == 1.0
            assert "rank_selected_wall_ms" not in profile["evidence"]
            assert len(profile["evidence"]["rank_instrumented_trace_elapsed_ms"]) == 8
            if framework == "sglang":
                overlay = profile["evidence"]["source_overlay"]
                assert overlay["base_source_commit"] == source_commit
                assert overlay["evidence_source_lock_sha256"] == overlay[
                    "source_lock_sha256"
                ]
                expected_manager = (
                    "31e2b1c19a9901233a3f28b15289f76a0932786b05232c185d6035f80781792d"
                    if phase == "decode"
                    else "82d74e7caace8379aecde00ea5ca91afb392bc369676f31f830653c0c5bba582"
                )
                assert overlay["profiler_manager_sha256"] == expected_manager
                if phase == "decode":
                    gate = profile["evidence"]["formal_step_throughput_gate"]
                    assert gate["formal_target_step"] == gate["profile_start_step"]
                    assert gate["activation_affected_scheduler_step"] == (
                        gate["formal_target_step"] - 1
                    )
                    assert gate["formal_target"]["throughput_token_s"] >= gate[
                        "minimum_accepted_throughput_token_s"
                    ]
                    collective = profile["evidence"][
                        "collective_rank_duration_audit"
                    ]
                    assert collective["outlier_count"] == 0


def test_profile_semantics_close_with_one_fusion_owner() -> None:
    for root, _ in PROFILE_CASES.values():
        for _, profile in _profiles(root):
            states = profile["node_states"]
            metrics = profile["node_metrics"]
            groups = profile["fusion_groups"]
            assert len(set(states) | set(metrics)) == 153
            assert not (set(states) & set(metrics))

            for node_id, state in states.items():
                assert state["status"] in {"structural", "fused", "not_selected"}
                label = state["label"].lower()
                assert "unmapped" not in label
                assert "mapping incomplete" not in label
                assert "generic fused implementation shard" not in label
                if state["status"] == "not_selected":
                    assert (
                        node_id.startswith("dspark_")
                        or node_id == "top.dspark_extension"
                    )
                    assert "dspark disabled" in label
                if state["status"] != "fused":
                    continue
                owner = state["included_in"]
                group = groups[state["fusion_group_id"]]
                assert group["owner"] == owner
                assert node_id in group["ir_nodes"]
                assert owner in metrics
                assert metrics[owner]["metric_kind"] == "exclusive_fusion_owner"

            for group in groups.values():
                owner = group["owner"]
                assert owner in group["ir_nodes"]
                assert len(group["ir_nodes"]) >= 2
                assert owner in metrics
                event_sets = group["member_event_sets"]
                assert set(event_sets) == set(group["ir_nodes"])
                assert len(
                    {
                        (
                            row["production_event_set_sha256"],
                            row["eager_event_set_sha256"],
                        )
                        for row in event_sets.values()
                    }
                ) == 1
                for member in group["ir_nodes"]:
                    if member != owner:
                        assert member not in metrics

            exclusive_residency_ms = sum(
                float(metric["gpu_residency_ms"])
                for metric in metrics.values()
                if metric["metric_kind"]
                in {"exclusive_leaf", "exclusive_fusion_owner"}
            )
            assert exclusive_residency_ms == pytest.approx(
                profile["evidence"]["attribution_timing"]["gpu_residency_ms"],
                abs=1e-6,
            )


def test_timelines_preserve_exact_eager_ids_and_timing_owner_links() -> None:
    for root, _ in PROFILE_CASES.values():
        for path, profile in _profiles(root):
            timeline_path = path.parent / profile["timeline"]["artifact"]
            assert _sha256(timeline_path) == profile["timeline"]["sha256"]
            with gzip.open(timeline_path, "rt") as source:
                timeline = json.load(source)
            strings = timeline["strings"]
            events = timeline["steps"][0]["events"]
            assert len(events) == profile["timeline"]["event_count"]
            assert len({event["event_id"] for event in events}) == len(events)
            assert all(event["eager_event_ids"] for event in events)
            fused_nodes = {
                node_id
                for node_id, state in profile["node_states"].items()
                if state["status"] == "fused"
            }
            assert all(
                strings[event["ir_node"]] not in fused_nodes for event in events
            )
            assert timeline["timing_summary"] == profile["evidence"]["attribution_timing"]
            wall = profile["evidence"]["production_wall_timing"]
            if profile["phase"] == "decode":
                assert wall["authority"] == "profiler_off_matched_scheduler_step"
                assert wall == profile["profile_summary"]["production_wall_timing"]
                assert timeline["timing_summary"]["authority"] == (
                    "instrumented_trace_attribution_only"
                )
            else:
                assert wall["status"] == "unavailable"
                assert wall["authority"] == "none"
                assert "no matched profiler-off" in wall["reason"]


def test_public_readme_separates_production_wall_from_instrumented_trace() -> None:
    readme = (REPO_ROOT / "catalog/deepseek_v4_pro/README.md").read_text()
    assert "Profiler-off production wall (ms)" in readme
    assert "Instrumented trace (ms)" in readme
    assert "48.671" not in readme
    assert "nine published timelines" in readme
    assert "SGLang prefill rejection is retained" in readme


def test_router_and_query_independent_event_sets_have_direct_owners() -> None:
    for framework, (root, _) in PROFILE_CASES.items():
        profile = yaml.safe_load((root / "cg_decode_gbs016_8k1k.yaml").read_text())
        timeline_path = root / profile["timeline"]["artifact"]
        with gzip.open(timeline_path, "rt") as source:
            timeline = json.load(source)
        strings = timeline["strings"]
        events = timeline["steps"][0]["events"]

        def event_sets(node_id: str) -> tuple[set[str], set[str]]:
            selected = [event for event in events if strings[event["ir_node"]] == node_id]
            return (
                {str(event["event_id"]) for event in selected},
                {
                    strings[eager_id]
                    for event in selected
                    for eager_id in event["eager_event_ids"]
                },
            )

        metrics = profile["node_metrics"]
        router_nodes = (
            "moe.score_projection",
            "moe.hash_select",
            "moe.learned_select",
        )
        for node_id in router_nodes:
            assert metrics[node_id]["metric_kind"] == "exclusive_leaf"
            assert metrics[node_id]["attribution_status"] == "measured_direct"
        router_sets = [event_sets(node_id) for node_id in router_nodes]
        assert all(physical and eager for physical, eager in router_sets)
        for index, (physical, eager) in enumerate(router_sets):
            for other_physical, other_eager in router_sets[index + 1 :]:
                assert physical.isdisjoint(other_physical)
                assert eager.isdisjoint(other_eager)
        for node_id in ("moe.sqrt_softplus", "moe.weights"):
            assert metrics[node_id]["metric_kind"] == "inclusive_rollup"
        if framework == "vllm":
            for prefix in ("csa", "hca"):
                query_nodes = tuple(
                    f"{prefix}_attention.{leaf}"
                    for leaf in ("q_a", "q_norm", "q_head_norm")
                )
                for node_id in query_nodes[:2]:
                    assert metrics[node_id]["metric_kind"] == "exclusive_leaf"
                query_sets = [event_sets(node_id) for node_id in query_nodes]
                assert all(physical and eager for physical, eager in query_sets)
                for index, (physical, eager) in enumerate(query_sets):
                    for other_physical, other_eager in query_sets[index + 1 :]:
                        assert physical.isdisjoint(other_physical)
                        assert eager.isdisjoint(other_eager)
                for leaf in ("q_a", "q_norm"):
                    assert metrics[f"{prefix}_attention.{leaf}"]["metric_kind"] == (
                        "exclusive_leaf"
                    )
