from __future__ import annotations

import gzip
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).parents[1]
VIEWER = ROOT / "docs" / "viewer.html"


def _run_compaction_case(case: dict) -> dict:
    viewer = VIEWER.read_text()
    packing_start = viewer.index("function packTimelineIntervals")
    core_start = viewer.index("// BEGIN TIMELINE COMPACTION CORE")
    core_end = viewer.index("// END TIMELINE COMPACTION CORE")
    source = viewer[packing_start:core_start] + viewer[core_start:core_end]
    script = f"""
const fs = require('fs');
{source}
const step = JSON.parse(fs.readFileSync(0, 'utf8'));
const before = JSON.stringify(step);
const compact = buildCompactTimelineTracks(step);
const packedKernelLanes = compact.tracks.map(track =>
  packTimelineIntervals(
    track.events,
    event => Number(event.start_us),
    event => Number(event.start_us) + Number(event.duration_us),
  ).laneCount
);
const compactLaneOverlapFree = compact.tracks.every(track => {{
  const segments = timelineActivitySegments(track.events, compact.toleranceUs);
  for (let left = 0; left < segments.length; left++) {{
    for (let right = left + 1; right < segments.length; right++) {{
      if (segments[left].streamId !== segments[right].streamId &&
          timelineSegmentsOverlap(segments[left], segments[right], compact.toleranceUs)) {{
        return false;
      }}
    }}
  }}
  return true;
}});
console.log(JSON.stringify({{
  physicalStreamCount: compact.physicalStreamCount,
  compactLaneCount: compact.compactLaneCount,
  peakConcurrency: compact.peakConcurrency,
  groupCount: compact.groups.length,
  packedKernelLanes,
  compactEventCount: compact.tracks.reduce((total, track) => total + track.events.length, 0),
  physicalIds: compact.physicalStreamIds,
  lanePhysicalIds: compact.tracks.map(track => track.physical_stream_ids),
  compactLaneOverlapFree,
  unchanged: before === JSON.stringify(step),
}}));
"""
    result = subprocess.run(
        ["node", "-e", script],
        input=json.dumps(case),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def _event(stream: int, start: float, duration: float) -> dict:
    return {
        "stream_id": stream,
        "start_us": start,
        "duration_us": duration,
    }


def _step(events: list[dict], roles: dict[int, str] | None = None) -> dict:
    roles = roles or {}
    stream_ids = sorted({event["stream_id"] for event in events})
    return {
        "tracks": [
            {
                "stream_id": stream_id,
                "role": roles.get(stream_id, "auxiliary compute"),
                "gpu_residency_us": sum(
                    event["duration_us"]
                    for event in events
                    if event["stream_id"] == stream_id
                ),
            }
            for stream_id in stream_ids
        ],
        "events": events,
    }


def test_many_serial_physical_streams_collapse_to_one_activity_lane() -> None:
    events = [_event(stream, stream * 2.0, 1.0) for stream in range(20)]
    result = _run_compaction_case(_step(events))

    assert result["physicalStreamCount"] == 20
    assert result["compactLaneCount"] == 1
    assert result["peakConcurrency"] == 1
    assert result["compactEventCount"] == len(events)
    assert result["unchanged"]


def test_true_concurrency_requires_the_same_number_of_lanes() -> None:
    events = [_event(stream, 0.0, 10.0) for stream in range(4)]
    result = _run_compaction_case(_step(events))

    assert result["physicalStreamCount"] == 4
    assert result["compactLaneCount"] == 4
    assert result["peakConcurrency"] == 4


def test_intermittent_stream_segments_can_reuse_a_lane() -> None:
    events = [
        _event(11, 0.0, 1.0),
        _event(11, 4.0, 1.0),
        _event(12, 1.5, 1.5),
    ]
    result = _run_compaction_case(_step(events))

    assert result["physicalStreamCount"] == 2
    assert result["compactLaneCount"] == 1
    assert result["peakConcurrency"] == 1
    assert result["lanePhysicalIds"] == [["11", "12"]]


def test_reliable_compute_and_collective_roles_remain_separate() -> None:
    events = [_event(21, 0.0, 1.0), _event(22, 2.0, 1.0)]
    result = _run_compaction_case(
        _step(events, {21: "main compute", 22: "communication"})
    )

    assert result["groupCount"] == 2
    assert result["compactLaneCount"] == 2
    assert result["peakConcurrency"] == 1


def test_main_compute_is_pinned_to_the_first_compute_lane() -> None:
    events = [
        _event(41, 4.0, 2.0),
        _event(42, 0.0, 3.0),
        _event(43, 4.5, 0.5),
    ]
    result = _run_compaction_case(
        _step(events, {41: "main compute", 42: "auxiliary compute", 43: "auxiliary compute"})
    )

    assert result["compactLaneCount"] == 2
    assert "41" in result["lanePhysicalIds"][0]


def test_same_stream_overlap_uses_kernel_sublanes_without_new_stream_rows() -> None:
    events = [_event(31, 0.0, 4.0), _event(31, 1.0, 2.0)]
    result = _run_compaction_case(_step(events, {31: "main compute"}))

    assert result["physicalStreamCount"] == 1
    assert result["compactLaneCount"] == 1
    assert result["packedKernelLanes"] == [2]
    assert result["compactEventCount"] == 2


def test_viewer_exposes_compact_default_and_physical_debug_mode() -> None:
    viewer = VIEWER.read_text()

    assert '<option value="compact">compact activity lanes</option>' in viewer
    assert '<option value="physical">physical streams</option>' in viewer
    assert 'let TIMELINE_STREAM_MODE = "compact";' in viewer
    assert 'click to expand exact physical-stream rows' in viewer
    assert 'TIMELINE_STREAM_MODE = "physical";' in viewer


def test_deepseek_v4_pro_cuda_graph_step_compacts_65_streams_to_4_lanes() -> None:
    artifact_path = (
        ROOT
        / "docs/deepseek_v4_pro_v2/timelines"
        / "deepseek_v4_pro_tp8_sglang_cg_decode_gbs001_8k1k.timeline.json.gz"
    )
    with gzip.open(artifact_path, "rt") as handle:
        step = json.load(handle)["steps"][0]

    result = _run_compaction_case(step)

    assert result["physicalStreamCount"] == 65
    assert result["compactLaneCount"] == 4
    assert result["peakConcurrency"] == 4
    assert result["compactEventCount"] == 2675
    assert result["unchanged"]


def test_every_checked_in_timeline_obeys_stream_presentation_contract() -> None:
    """New model/profile artifacts automatically enter the generic stream gate."""

    failures: list[str] = []
    artifact_paths = sorted(ROOT.glob("docs/*/timelines/*.timeline.json.gz"))
    assert artifact_paths, "no timeline artifacts found"

    for artifact_path in artifact_paths:
        with gzip.open(artifact_path, "rt") as handle:
            artifact = json.load(handle)
        for step_index, step in enumerate(artifact.get("steps") or []):
            result = _run_compaction_case(step)
            event_streams = {str(event["stream_id"]) for event in step.get("events") or []}
            lane_streams = {
                stream_id
                for lane in result["lanePhysicalIds"]
                for stream_id in lane
            }
            problems: list[str] = []
            if not result["unchanged"]:
                problems.append("compaction mutated timing evidence")
            if result["compactEventCount"] != len(step.get("events") or []):
                problems.append("compact event count differs from physical evidence")
            if result["physicalStreamCount"] != len(event_streams):
                problems.append("physical stream count differs from event stream IDs")
            if set(result["physicalIds"]) != event_streams:
                problems.append("reported physical stream IDs differ from event evidence")
            if lane_streams != event_streams:
                problems.append("compact lanes do not cover every physical stream")
            if result["compactLaneCount"] > result["physicalStreamCount"]:
                problems.append("compact lane count exceeds physical stream count")
            if not result["compactLaneOverlapFree"]:
                problems.append("simultaneously active physical streams share one compact lane")
            if problems:
                failures.append(
                    f"{artifact_path.relative_to(ROOT)} step {step_index}: "
                    + "; ".join(problems)
                )

    assert not failures, "\n" + "\n".join(failures)
