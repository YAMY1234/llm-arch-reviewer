from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_ROOT = (
    REPO_ROOT
    / "catalog/deepseek_v4_pro/profiles/tp8/vllm_dd10e03_dsv4pro0813_tp8"
)
EXPECTED = {
    "eager_prefill_gbs001_8k.yaml": ("prefill", 1, False),
    "cg_decode_gbs001_8k1k.yaml": ("decode", 1, True),
    "cg_decode_gbs016_8k1k.yaml": ("decode", 16, True),
    "cg_decode_gbs064_8k1k.yaml": ("decode", 64, True),
    "cg_decode_gbs256_8k1k.yaml": ("decode", 256, True),
}
MODEL_REVISION = "72e1d3230f6c080a530b0a1d46f8eb4602340597"
SOURCE_COMMIT = "dd10e03f95f94edbea1975c67ace3a35ec9a8a40"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _profiles() -> list[tuple[Path, dict]]:
    paths = sorted(PROFILE_ROOT.glob("*.yaml"))
    assert {path.name for path in paths} == set(EXPECTED)
    return [(path, yaml.safe_load(path.read_text())) for path in paths]


def test_vllm_profile_matrix_is_exact_and_source_locked() -> None:
    for path, profile in _profiles():
        phase, batch_size, graph_on = EXPECTED[path.name]
        assert profile["phase"] == phase
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
        assert profile["evidence"]["source_commit"] == SOURCE_COMMIT
        assert profile["evidence"]["mapped_kernel_count_ratio"] == 1.0
        assert profile["evidence"]["mapped_kernel_duration_ratio"] == 1.0


def test_vllm_profile_semantics_close_with_one_fusion_owner() -> None:
    for _, profile in _profiles():
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
                assert node_id.startswith("dspark_") or node_id == "top.dspark_extension"
                assert "dspark disabled" in label
            if state["status"] == "fused":
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
            for member in group["ir_nodes"]:
                if member != owner:
                    assert member not in metrics


def test_vllm_timelines_preserve_exact_eager_ids_and_timing_owner_links() -> None:
    for path, profile in _profiles():
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
        assert all(strings[event["ir_node"]] not in fused_nodes for event in events)
        assert timeline["timing_summary"] == profile["evidence"]["timing"]
