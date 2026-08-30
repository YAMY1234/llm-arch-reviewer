from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import statistics
import sys

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.common.timeline_artifact import (  # noqa: E402
    build_timeline_artifact,
    timeline_targets,
    write_timeline_artifact,
)


def _event(
    node: str,
    *,
    timestamp: float,
    duration: float,
    stream: int,
    substage: str = "attention",
) -> dict:
    return {
        "kernel_name": f"kernel::{node}",
        "kernel_label": node,
        "node": node,
        "ts_us": timestamp,
        "dur_us": duration,
        "stream": stream,
        "layer_id": 3,
        "layer_kind": "linear",
        "substage": substage,
        "segment_id": 6,
        "occurrence_id": "layer_03.attention",
        "eager_event_id": "eager-r0-k42",
        "attribution_method": "validated_execution_sequence",
        "confidence": "high",
        "python_stack": [
            {
                "file": "python/sglang/srt/models/qwen4_exp.py",
                "line": 100,
                "function": "forward",
                "module": "sglang.srt.models.qwen4_exp",
                "raw": "qwen4_exp.py(100): forward",
            }
        ],
        "stack_evidence": {
            "source": "eager_trace",
            "match": "representative_ir_node_stack",
            "kind": "full_eager_python_stack",
        },
    }


def test_targets_include_direct_node_and_architecture_rollups() -> None:
    targets = timeline_targets(
        _event(
            "hyperconnection.mix",
            timestamp=0.0,
            duration=1.0,
            stream=1,
            substage="attn_hc_mix",
        )
    )

    assert targets == [
        "hyperconnection.mix",
        "hyperconnection_mix.mix",
        "stack.linear_layer",
        "top.decoder_stack",
        "linear_layer.attn_hc_mix",
    ]


def test_targets_preserve_explicit_portable_ir_targets() -> None:
    event = _event(
        "gdn_attention.qkvz_projection",
        timestamp=0.0,
        duration=1.0,
        stream=1,
    )
    event["layer_kind"] = "gdn"
    event["ir_targets"] = [
        "gdn_attention.ba_projection",
        "gdn_moe_block.attention",
        "stack.gdn_layer",
        "top.decoder_stack",
    ]

    assert timeline_targets(event) == [
        "gdn_attention.qkvz_projection",
        "gdn_attention.ba_projection",
        "gdn_moe_block.attention",
        "stack.gdn_layer",
        "top.decoder_stack",
    ]


def test_timeline_persists_occurrence_and_eager_event_identity() -> None:
    artifact = build_timeline_artifact(
        profile_id="p",
        phase="decode",
        reference_rank=0,
        steps=[{
            "step_index": 0,
            "trace_start_us": 0.0,
            "duration_us": 10.0,
            "events": [_event("linear_attention.recurrent_update", timestamp=1.0, duration=2.0, stream=7)],
        }],
        timing_summary={},
        raw_trace={},
        stack_source={},
    )
    event = artifact["steps"][0]["events"][0]
    strings = artifact["strings"]
    assert event["segment_id"] == 6
    assert strings[event["occurrence_id"]] == "layer_03.attention"
    assert strings[event["eager_event_id"]] == "eager-r0-k42"


def test_timeline_persists_explicit_runtime_support_contract() -> None:
    event = _event("", timestamp=1.0, duration=2.0, stream=7)
    event["node"] = None
    event["support_class"] = "request_batch_metadata"
    event["support_reason"] = "shape/index preparation"
    artifact = build_timeline_artifact(
        profile_id="p",
        phase="decode",
        reference_rank=0,
        steps=[{
            "step_index": 0,
            "trace_start_us": 0.0,
            "duration_us": 10.0,
            "events": [event],
        }],
        timing_summary={},
        raw_trace={},
        stack_source={},
    )
    encoded = artifact["steps"][0]["events"][0]
    strings = artifact["strings"]
    assert strings[encoded["support_class"]] == "request_batch_metadata"
    assert strings[encoded["support_reason"]] == "shape/index preparation"


def test_ple_scope_takes_precedence_over_enclosing_decoder_layer() -> None:
    targets = timeline_targets(
        _event(
            "ple.ngram_hash",
            timestamp=0.0,
            duration=1.0,
            stream=1,
            substage="attention",
        )
    )

    assert targets == [
        "ple.ngram_hash",
        "stack.ple_injection",
        "top.decoder_stack",
    ]


def test_external_moe_collective_is_not_rolled_back_into_moe_compute() -> None:
    targets = timeline_targets(
        _event(
            "linear_layer.tp_moe_output_collective",
            timestamp=0.0,
            duration=1.0,
            stream=1,
            substage="moe",
        )
    )

    assert targets == [
        "linear_layer.tp_moe_output_collective",
        "stack.linear_layer",
        "top.decoder_stack",
    ]


def test_mtp_leaf_targets_preserve_auxiliary_scope_and_generation_stage() -> None:
    event = _event(
        "mtp_qsa_attention.attention_core",
        timestamp=0.0,
        duration=1.0,
        stream=1,
        substage="mtp_prefill_attention",
    )
    event["layer_kind"] = "mtp"

    assert timeline_targets(event) == [
        "mtp_qsa_attention.attention_core",
        "mtp_layer.qsa_attention",
        "mtp_head.decoder_layer",
        "mtp_generation.mtp_prefill",
    ]


def test_mtp_hc_leaf_targets_use_scoped_layer_drill() -> None:
    event = _event(
        "hyperconnection.combine",
        timestamp=0.0,
        duration=1.0,
        stream=1,
        substage="mtp_draft_extend_mlp_hc_combine",
    )
    event["layer_kind"] = "mtp"

    assert timeline_targets(event) == [
        "hyperconnection.combine",
        "hyperconnection_combine.combine",
        "mtp_layer.mlp_hc_combine",
        "mtp_head.decoder_layer",
        "mtp_generation.mtp_draft_extend",
    ]


def test_mtp_attention_and_moe_use_timeline_semantic_colors() -> None:
    attention = _event(
        "mtp_qsa_attention.attention_core",
        timestamp=0.0,
        duration=1.0,
        stream=1,
        substage="mtp_draft_extend_attention",
    )
    moe = _event(
        "mtp_moe.routed_experts",
        timestamp=1.0,
        duration=1.0,
        stream=1,
        substage="mtp_draft_extend_moe",
    )
    for event in (attention, moe):
        event["layer_kind"] = "mtp"
    artifact = build_timeline_artifact(
        profile_id="mtp-profile",
        phase="decode",
        reference_rank=0,
        steps=[
            {
                "step_index": 1,
                "trace_start_us": 0.0,
                "duration_us": 2.0,
                "events": [attention, moe],
            }
        ],
        timing_summary={},
        raw_trace={"file": "trace.json.gz", "sha256": "0" * 64},
        stack_source={"source": "eager_mtp_trace"},
    )
    strings = artifact["strings"]
    kinds = [strings[event["kernel_kind"]] for event in artifact["steps"][0]["events"]]
    assert kinds == ["attention", "moe"]


def test_timeline_separates_elapsed_active_residency_idle_and_overlap() -> None:
    artifact = build_timeline_artifact(
        profile_id="profile",
        phase="decode",
        reference_rank=0,
        steps=[
            {
                "step_index": 1,
                "trace_start_us": 100.0,
                "duration_us": 10.0,
                "events": [
                    _event(
                        "linear_attention.recurrence",
                        timestamp=101.0,
                        duration=4.0,
                        stream=23,
                    ),
                    _event(
                        "qsa_attention.indexer",
                        timestamp=103.0,
                        duration=4.0,
                        stream=3446,
                    ),
                ],
            }
        ],
        timing_summary={},
        raw_trace={"file": "trace.json.gz", "sha256": "0" * 64},
        stack_source={"source": "eager_trace"},
    )

    step = artifact["steps"][0]
    assert step["duration_us"] == 10.0
    assert step["active_gpu_us"] == 6.0
    assert step["gpu_residency_us"] == 8.0
    assert step["device_gap_us"] == 4.0
    assert step["gpu_overlap_us"] == 2.0
    assert step["idle_intervals"] == [
        {"start_us": 0.0, "duration_us": 1.0},
        {"start_us": 7.0, "duration_us": 3.0},
    ]
    assert [track["role"] for track in step["tracks"]] == [
        "main compute",
        "QSA indexer",
    ]
    assert len(artifact["stacks"]) == 1


def test_mixed_compute_collective_stream_remains_main_compute() -> None:
    artifact = build_timeline_artifact(
        profile_id="profile",
        phase="decode",
        reference_rank=0,
        steps=[
            {
                "step_index": 1,
                "trace_start_us": 0.0,
                "duration_us": 150.0,
                "events": [
                    _event(
                        "hyperconnection.combine",
                        timestamp=0.0,
                        duration=60.0,
                        stream=23,
                    ),
                    _event(
                        "mtp_layer.tp_moe_output_collective",
                        timestamp=60.0,
                        duration=70.0,
                        stream=23,
                    ),
                    _event(
                        "mtp_moe.shared_expert",
                        timestamp=5.0,
                        duration=5.0,
                        stream=55,
                    ),
                ],
            }
        ],
        timing_summary={},
        raw_trace={"file": "trace.json.gz", "sha256": "0" * 64},
        stack_source={"source": "eager_trace"},
    )

    assert [(track["stream_id"], track["role"]) for track in artifact["steps"][0]["tracks"]] == [
        ("23", "main compute"),
        ("55", "shared expert"),
    ]


def test_gzip_artifact_is_deterministic(tmp_path: Path) -> None:
    artifact = {
        "schema_version": "timeline.v1",
        "profile_id": "p",
        "steps": [],
    }
    first = tmp_path / "first.json.gz"
    second = tmp_path / "second.json.gz"

    first_sha = write_timeline_artifact(first, artifact)
    second_sha = write_timeline_artifact(second, artifact)

    assert first_sha == second_sha
    assert first.read_bytes() == second.read_bytes()
    assert json.loads(gzip.decompress(first.read_bytes())) == artifact


def test_qwen40_profile_timelines_match_profile_timing_and_ir() -> None:
    profile_paths = sorted(
        (REPO_ROOT / "catalog" / "qwen40" / "profiles").glob("*/*/*.yaml")
    )
    assert profile_paths
    for profile_path in profile_paths:
        profile = yaml.safe_load(profile_path.read_text())
        descriptor = profile.get("timeline")
        assert descriptor, profile_path
        artifact_path = profile_path.parent / descriptor["artifact"]
        assert hashlib.sha256(artifact_path.read_bytes()).hexdigest() == descriptor["sha256"]
        artifact = json.loads(gzip.decompress(artifact_path.read_bytes()))
        assert artifact["schema_version"] == "timeline.v1"
        assert artifact["profile_id"] == profile["profile_id"]
        assert artifact["phase"] == profile["phase"]
        assert len(artifact["steps"]) == descriptor["step_count"]
        assert sum(len(step["events"]) for step in artifact["steps"]) == descriptor["event_count"]

        strings = artifact["strings"]
        for step in artifact["steps"]:
            assert {track["stream_id"] for track in step["tracks"]} == {
                event["stream_id"] for event in step["events"]
            }
            assert all(strings[event["ir_node"]] for event in step["events"])
            assert all(event["ir_targets"] for event in step["events"])
            assert all(event["start_us"] >= -1e-3 for event in step["events"])
            assert all(
                event["start_us"] + event["duration_us"]
                <= step["duration_us"] + 1e-3
                for event in step["events"]
            )

        timing = profile["profile_summary"]["timing"]
        mean_ms = lambda field: statistics.fmean(  # noqa: E731
            step[field] / 1000.0 for step in artifact["steps"]
        )
        assert mean_ms("duration_us") == pytest.approx(timing["elapsed_ms"], abs=1e-6)
        assert mean_ms("active_gpu_us") == pytest.approx(timing["active_gpu_ms"], abs=1e-6)
        assert mean_ms("gpu_residency_us") == pytest.approx(
            timing["gpu_residency_ms"], abs=1e-6
        )
        assert mean_ms("device_gap_us") == pytest.approx(timing["device_gap_ms"], abs=1e-6)
        assert mean_ms("gpu_overlap_us") == pytest.approx(timing["gpu_overlap_ms"], abs=1e-6)
