from __future__ import annotations

import gzip
import json
from pathlib import Path
import sys

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
        "attribution_method": "validated_execution_sequence",
        "confidence": "high",
        "python_stack": [
            {
                "file": "models/toy.py",
                "line": 100,
                "function": "forward",
                "module": "models.toy",
                "raw": "toy.py(100): forward",
            }
        ],
        "stack_evidence": {
            "source": "eager_trace",
            "match": "representative_ir_node_stack",
            "kind": "full_eager_python_stack",
        },
    }


def test_targets_are_producer_authored_and_deduplicated() -> None:
    event = _event(
        "block.projection",
        timestamp=0.0,
        duration=1.0,
        stream=1,
    )
    event["ir_targets"] = [
        "block.projection",
        "stack.block",
        "top.stack",
        "stack.block",
    ]

    assert timeline_targets(event) == [
        "block.projection",
        "stack.block",
        "top.stack",
    ]


def test_default_targets_do_not_infer_model_hierarchy() -> None:
    event = _event(
        "block.projection",
        timestamp=0.0,
        duration=1.0,
        stream=1,
        substage="model_specific_stage",
    )
    event["layer_kind"] = "model_specific_layer"

    assert timeline_targets(event) == ["block.projection"]


def test_custom_target_and_kernel_kind_resolvers_are_supported() -> None:
    first = _event(
        "block.projection",
        timestamp=0.0,
        duration=1.0,
        stream=1,
    )
    second = _event(
        "block.exchange",
        timestamp=1.0,
        duration=1.0,
        stream=2,
    )
    second["stream_role"] = "exchange"
    artifact = build_timeline_artifact(
        profile_id="toy-profile",
        phase="decode",
        reference_rank=0,
        steps=[
            {
                "step_index": 1,
                "trace_start_us": 0.0,
                "duration_us": 2.0,
                "events": [first, second],
            }
        ],
        timing_summary={},
        raw_trace={"file": "trace.json.gz", "sha256": "0" * 64},
        stack_source={"source": "toy_trace"},
        target_resolver=lambda event: [event["node"], "top.stack"],
        kernel_kind_resolver=lambda event: (
            "communication" if event["node"].endswith("exchange") else "compute"
        ),
    )
    strings = artifact["strings"]
    events = artifact["steps"][0]["events"]
    assert [[strings[target] for target in event["ir_targets"]] for event in events] == [
        ["block.projection", "top.stack"],
        ["block.exchange", "top.stack"],
    ]
    assert [strings[event["kernel_kind"]] for event in events] == [
        "compute",
        "communication",
    ]
    assert [track["role"] for track in artifact["steps"][0]["tracks"]] == [
        "main compute",
        "exchange",
    ]


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
                        "block.primary_compute",
                        timestamp=101.0,
                        duration=4.0,
                        stream=23,
                    ),
                    {
                        **_event(
                            "block.auxiliary_compute",
                            timestamp=103.0,
                            duration=4.0,
                            stream=3446,
                        ),
                        "stream_role": "auxiliary math",
                    },
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
    assert sum(item["gpu_residency_us"] for item in step["node_timings"]) == 8.0
    assert step["idle_intervals"] == [
        {"start_us": 0.0, "duration_us": 1.0},
        {"start_us": 7.0, "duration_us": 3.0},
    ]
    assert [track["role"] for track in step["tracks"]] == [
        "main compute",
        "auxiliary math",
    ]
    assert len(artifact["stacks"]) == 1


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
