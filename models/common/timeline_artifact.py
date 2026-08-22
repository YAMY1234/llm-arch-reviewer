"""Build compact, provenance-preserving timeline artifacts for the viewer.

The architecture profile is an aggregate.  A timeline artifact keeps the
individual measured kernel intervals that produced that aggregate and attaches
the same stable IR targets used by the architecture view.  CUDA-Graph traces do
not contain Python stacks, so stack evidence is copied only from an eager event
with an explicit match kind; the viewer can then distinguish direct evidence
from transferred evidence.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable


TIMELINE_SCHEMA_VERSION = "timeline.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _interval_union(
    intervals: Iterable[tuple[float, float]],
) -> list[tuple[float, float]]:
    ordered = sorted(
        (float(start), float(stop))
        for start, stop in intervals
        if float(stop) > float(start)
    )
    if not ordered:
        return []
    merged = [ordered[0]]
    for start, stop in ordered[1:]:
        previous_start, previous_stop = merged[-1]
        if start <= previous_stop:
            merged[-1] = (previous_start, max(previous_stop, stop))
        else:
            merged.append((start, stop))
    return merged


def _idle_intervals(
    events: Iterable[dict[str, Any]], *, start_us: float, duration_us: float
) -> list[dict[str, float]]:
    stop_us = start_us + duration_us
    active = _interval_union(
        (
            max(start_us, float(event["ts_us"])),
            min(stop_us, float(event["ts_us"]) + float(event["dur_us"])),
        )
        for event in events
    )
    idle: list[dict[str, float]] = []
    cursor = start_us
    for active_start, active_stop in active:
        if active_start > cursor:
            idle.append(
                {
                    "start_us": round(cursor - start_us, 6),
                    "duration_us": round(active_start - cursor, 6),
                }
            )
        cursor = max(cursor, active_stop)
    if cursor < stop_us:
        idle.append(
            {
                "start_us": round(cursor - start_us, 6),
                "duration_us": round(stop_us - cursor, 6),
            }
        )
    return idle


def timeline_targets(event: dict[str, Any]) -> list[str]:
    """Return producer-authored direct and roll-up IR targets.

    The artifact layer deliberately knows nothing about a model's node names or
    hierarchy.  A model-specific producer may either pre-populate
    ``event.ir_targets`` or pass a resolver to :func:`build_timeline_artifact`.
    Keeping the default this small prevents one model's semantic graph from
    becoming an implicit dependency of another model package.
    """

    direct = str(event.get("node") or "")
    authored = event.get("ir_targets") or []
    if isinstance(authored, str):
        authored = [authored]
    return list(
        dict.fromkeys(
            target
            for target in (direct, *(str(value) for value in authored))
            if target
        )
    )


def build_ancestor_target_resolver(
    views: dict[str, dict[str, Any]],
) -> Callable[[dict[str, Any]], list[str]]:
    """Return a resolver that closes event targets over every drill ancestor.

    A view may have more than one parent (for example the target decoder stack
    is reachable both from the model overview and from target verification).
    Keeping all parents makes the same measured event reachable from every
    Architecture entry point without asking each profile producer to maintain
    a second, fragile hierarchy table.
    """

    parents_by_view: dict[str, list[str]] = defaultdict(list)
    for parent_view, view in views.items():
        for node in view.get("nodes") or []:
            child_view = node.get("drill")
            if child_view and node.get("timeline_rollup", True):
                parents_by_view[str(child_view)].append(
                    f"{parent_view}.{node['id']}"
                )

    def resolve(event: dict[str, Any]) -> list[str]:
        resolved = timeline_targets(event)
        seen = set(resolved)
        queue = list(resolved)
        while queue:
            target = queue.pop(0)
            view_id, separator, _node_id = target.partition(".")
            if not separator:
                continue
            for parent in parents_by_view.get(view_id, ()):
                if parent in seen:
                    continue
                seen.add(parent)
                resolved.append(parent)
                queue.append(parent)
        return resolved

    return resolve


def _events_path_for_mapping(mapping_path: Path) -> Path | None:
    name = mapping_path.name
    if not name.startswith("kernel_mapping."):
        return None
    candidate = mapping_path.with_name(name.replace("kernel_mapping.", "events.", 1))
    return candidate if candidate.exists() else None


def load_eager_stack_index(
    mapping_path: Path,
) -> dict[str, dict[Any, list[dict[str, Any]]]]:
    """Index eager Python stacks by exact kernel+IR and by representative IR.

    The full stack lives in the sibling ``events.<rank>.jsonl`` file.  If that
    file is unavailable, selected semantic frames in the mapping remain usable
    as explicitly reduced evidence.
    """

    mappings = [
        json.loads(line)
        for line in mapping_path.read_text().splitlines()
        if line.strip()
    ]
    events_path = _events_path_for_mapping(mapping_path)
    eager_events: dict[str, dict[str, Any]] = {}
    if events_path is not None:
        eager_events = {
            str(row.get("event_id")): row
            for row in (
                json.loads(line)
                for line in events_path.read_text().splitlines()
                if line.strip()
            )
        }

    exact: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_node: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for mapping in mappings:
        node = str(mapping.get("selected_node") or "")
        if not node:
            continue
        event_id = str(mapping.get("event_id") or "")
        event = eager_events.get(event_id, {})
        stack = event.get("python_stack") or []
        stack_kind = "full_eager_python_stack"
        if not stack:
            stack_kind = "selected_eager_frames"
            stack = [
                mapping.get(key)
                for key in (
                    "primitive_frame",
                    "operator_frame",
                    "semantic_frame",
                    "model_context_frame",
                    "phase_frame",
                )
                if mapping.get(key)
            ]
        if not stack:
            continue
        evidence = {
            "event_id": event_id,
            "kernel_name": str(mapping.get("kernel_name") or ""),
            "node": node,
            "confidence": str(mapping.get("confidence") or "unknown"),
            "stack_kind": stack_kind,
            "python_stack": stack,
            "cpu_op_name": event.get("cpu_op_name") or mapping.get("cpu_op_name"),
        }
        exact[(node, evidence["kernel_name"])].append(evidence)
        by_node[node].append(evidence)
    return {"exact": exact, "by_node": by_node}


def attach_eager_stack_evidence(
    events: Iterable[dict[str, Any]], *, mapping_path: Path
) -> list[dict[str, Any]]:
    """Attach eager stack evidence without presenting it as a formal stack."""

    index = load_eager_stack_index(mapping_path)
    enriched: list[dict[str, Any]] = []
    for raw in events:
        event = dict(raw)
        node = str(event.get("node") or "")
        name = str(event.get("kernel_name") or "")
        candidates = index["exact"].get((node, name), [])
        match_kind = "exact_kernel_name_and_ir_node"
        if not candidates:
            candidates = index["by_node"].get(node, [])
            match_kind = "representative_ir_node_stack"
        if candidates:
            evidence = candidates[0]
            event["python_stack"] = evidence["python_stack"]
            event["cpu_op_name"] = evidence.get("cpu_op_name")
            event["stack_evidence"] = {
                "source": "eager_trace",
                "match": match_kind,
                "kind": evidence["stack_kind"],
                "event_id": evidence["event_id"],
                "confidence": evidence["confidence"],
            }
        enriched.append(event)
    return enriched


def _kernel_kind(event: dict[str, Any]) -> str:
    authored = event.get("kernel_kind")
    if authored:
        return str(authored)
    node = str(event.get("node") or "").lower()
    name = str(event.get("kernel_name") or "").lower()
    if (
        "collective" in node
        or "dispatch_collective" in node
        or "combine_collective" in node
        or "nccl" in name
        or "allreduce" in name
        or "allgather" in name
        or "alltoall" in name
        or "all_to_all" in name
        or "reduce_scatter" in name
    ):
        return "communication"
    if any(token in name for token in ("memcpy", "copy_kernel", "fill_kernel")):
        return "memory"
    return "compute"


def _stream_key(event: dict[str, Any]) -> str:
    stream = event.get("stream")
    if stream is None:
        stream = event.get("tid")
    return str(stream if stream is not None else "unknown")


def _stream_tracks(
    events: list[dict[str, Any]],
    *,
    kernel_kind_resolver: Callable[[dict[str, Any]], str],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        grouped[_stream_key(event)].append(event)

    totals = {
        stream: sum(float(event["dur_us"]) for event in stream_events)
        for stream, stream_events in grouped.items()
    }
    roles: dict[str, str] = {}
    for stream, stream_events in grouped.items():
        by_role: Counter[str] = Counter()
        for event in stream_events:
            duration = float(event["dur_us"])
            authored_role = event.get("stream_role")
            if authored_role:
                by_role[str(authored_role)] += duration
                continue
            kind = kernel_kind_resolver(event)
            if kind == "communication":
                by_role["communication"] += duration
        role, duration = by_role.most_common(1)[0] if by_role else ("", 0.0)
        if totals[stream] and duration / totals[stream] >= 0.45:
            roles[stream] = role

    remaining = [stream for stream in grouped if stream not in roles]
    if remaining:
        main = max(remaining, key=lambda stream: totals[stream])
        roles[main] = "main compute"
    for stream in grouped:
        roles.setdefault(stream, "auxiliary compute")

    role_order = {
        "main compute": 0,
        "communication": 1,
        "auxiliary compute": 2,
    }
    return [
        {
            "stream_id": stream,
            "role": roles[stream],
            "label": f"{roles[stream]} · stream {stream}",
            "event_count": len(grouped[stream]),
            "gpu_residency_us": round(totals[stream], 6),
        }
        for stream in sorted(
            grouped,
            key=lambda stream: (
                role_order.get(roles[stream], 9),
                -totals[stream],
                stream,
            ),
        )
    ]


class _StringTable:
    def __init__(self) -> None:
        self.values: list[str] = []
        self.index: dict[str, int] = {}

    def add(self, value: Any) -> int | None:
        if value is None:
            return None
        text = str(value)
        if text not in self.index:
            self.index[text] = len(self.values)
            self.values.append(text)
        return self.index[text]


def build_timeline_artifact(
    *,
    profile_id: str,
    phase: str,
    reference_rank: int,
    steps: list[dict[str, Any]],
    timing_summary: dict[str, Any],
    raw_trace: dict[str, Any],
    stack_source: dict[str, Any],
    target_resolver: Callable[[dict[str, Any]], list[str]] = timeline_targets,
    kernel_kind_resolver: Callable[[dict[str, Any]], str] = _kernel_kind,
) -> dict[str, Any]:
    """Build a compact artifact from already-attributed measured events."""

    strings = _StringTable()
    stack_table: list[dict[str, Any]] = []
    stack_index: dict[str, int] = {}
    encoded_steps: list[dict[str, Any]] = []

    def encode_stack(event: dict[str, Any]) -> int | None:
        frames = event.get("python_stack") or []
        evidence = event.get("stack_evidence") or {}
        if not frames:
            return None
        normalized = [
            {
                "raw": frame.get("raw"),
                "file": frame.get("file"),
                "line": frame.get("line"),
                "function": frame.get("function"),
                "module": frame.get("module"),
            }
            for frame in frames
        ]
        identity = json.dumps(
            {"frames": normalized, "evidence": evidence},
            sort_keys=True,
            separators=(",", ":"),
        )
        if identity in stack_index:
            return stack_index[identity]
        stack_id = len(stack_table)
        stack_index[identity] = stack_id
        stack_table.append(
            {
                "frames": [
                    {
                        "raw": strings.add(frame.get("raw")),
                        "file": strings.add(frame.get("file")),
                        "line": frame.get("line"),
                        "function": strings.add(frame.get("function")),
                        "module": strings.add(frame.get("module")),
                    }
                    for frame in normalized
                ],
                "evidence": {
                    key: strings.add(value) if value is not None else None
                    for key, value in evidence.items()
                },
            }
        )
        return stack_id

    for raw_step in steps:
        step_events = sorted(
            raw_step.get("events") or [], key=lambda event: float(event["ts_us"])
        )
        start_us = float(raw_step["trace_start_us"])
        duration_us = float(raw_step["duration_us"])
        encoded_events = []
        resolved_targets_by_event: list[tuple[dict[str, Any], list[str]]] = []
        for index, event in enumerate(step_events):
            targets = target_resolver(event)
            resolved_targets_by_event.append((event, targets))
            encoded_events.append(
                {
                    "event_id": f"r{reference_rank}-s{raw_step['step_index']}-k{index}",
                    "engine": strings.add(event.get("engine")),
                    "start_us": round(float(event["ts_us"]) - start_us, 6),
                    "duration_us": round(float(event["dur_us"]), 6),
                    "stream_id": _stream_key(event),
                    "device": event.get("device"),
                    "pid": event.get("pid"),
                    "tid": event.get("tid"),
                    "kernel_name": strings.add(event.get("kernel_name")),
                    "kernel_label": strings.add(event.get("kernel_label")),
                    "ir_node": strings.add(event.get("node")),
                    "ir_targets": [strings.add(target) for target in targets],
                    "layer_id": event.get("layer_id"),
                    "layer_kind": strings.add(event.get("layer_kind")),
                    "substage": strings.add(event.get("substage")),
                    "mtp_round": event.get("mtp_round"),
                    "kernel_kind": strings.add(kernel_kind_resolver(event)),
                    "attribution_method": strings.add(
                        event.get("attribution_method")
                    ),
                    "mapping_status": strings.add(event.get("mapping_status")),
                    "confidence": strings.add(event.get("confidence")),
                    "fusion_group": strings.add(event.get("fusion_group")),
                    "unmapped_reason": strings.add(event.get("unmapped_reason")),
                    "candidate_nodes": [
                        strings.add(candidate)
                        for candidate in (event.get("candidate_nodes") or [])
                    ],
                    "cpu_op_name": strings.add(event.get("cpu_op_name")),
                    "stack_id": encode_stack(event),
                }
            )
        active_intervals = _interval_union(
            (
                max(start_us, float(event["ts_us"])),
                min(
                    start_us + duration_us,
                    float(event["ts_us"]) + float(event["dur_us"]),
                ),
            )
            for event in step_events
        )
        active_us = sum(stop - start for start, stop in active_intervals)
        residency_us = sum(float(event["dur_us"]) for event in step_events)
        events_by_node: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for event, targets in resolved_targets_by_event:
            for target in targets:
                events_by_node[target].append(event)
        node_timings = []
        for node, node_events in events_by_node.items():
            node_start = min(float(event["ts_us"]) for event in node_events)
            node_stop = max(
                float(event["ts_us"]) + float(event["dur_us"])
                for event in node_events
            )
            node_active = sum(
                stop - start
                for start, stop in _interval_union(
                    (
                        float(event["ts_us"]),
                        float(event["ts_us"]) + float(event["dur_us"]),
                    )
                    for event in node_events
                )
            )
            node_residency = sum(float(event["dur_us"]) for event in node_events)
            all_active_in_span = sum(
                stop - start
                for start, stop in _interval_union(
                    (
                        max(node_start, float(event["ts_us"])),
                        min(
                            node_stop,
                            float(event["ts_us"]) + float(event["dur_us"]),
                        ),
                    )
                    for event in step_events
                )
            )
            node_elapsed = node_stop - node_start
            node_timings.append(
                {
                    "ir_node": strings.add(node),
                    "occurrence_count": len(node_events),
                    "elapsed_us": round(node_elapsed, 6),
                    "active_gpu_us": round(node_active, 6),
                    "gpu_residency_us": round(node_residency, 6),
                    "gpu_overlap_us": round(max(0.0, node_residency - node_active), 6),
                    "module_gap_us": round(max(0.0, node_elapsed - node_active), 6),
                    "other_gpu_work_us": round(
                        max(0.0, all_active_in_span - node_active), 6
                    ),
                }
            )
        node_timings.sort(key=lambda item: -float(item["gpu_residency_us"]))
        encoded_steps.append(
            {
                "step_index": raw_step["step_index"],
                "label": raw_step.get("label") or f"step {raw_step['step_index']}",
                "trace_start_us": round(start_us, 6),
                "duration_us": round(duration_us, 6),
                "active_gpu_us": round(active_us, 6),
                "gpu_residency_us": round(residency_us, 6),
                "device_gap_us": round(max(0.0, duration_us - active_us), 6),
                "gpu_overlap_us": round(max(0.0, residency_us - active_us), 6),
                "node_timings": node_timings,
                "tracks": _stream_tracks(
                    step_events, kernel_kind_resolver=kernel_kind_resolver
                ),
                "idle_intervals": _idle_intervals(
                    step_events, start_us=start_us, duration_us=duration_us
                ),
                "events": encoded_events,
            }
        )

    return {
        "schema_version": TIMELINE_SCHEMA_VERSION,
        "profile_id": profile_id,
        "phase": phase,
        "time_unit": "microseconds",
        "reference_rank": reference_rank,
        "timing_summary": timing_summary,
        "raw_trace": raw_trace,
        "stack_source": stack_source,
        "strings": strings.values,
        "stacks": stack_table,
        "steps": encoded_steps,
    }


def write_timeline_artifact(path: Path, artifact: dict[str, Any]) -> str:
    """Write a deterministic JSON or JSON.GZ artifact and return its SHA256."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        artifact, ensure_ascii=False, separators=(",", ":"), sort_keys=False
    ).encode("utf-8")
    if path.suffix == ".gz":
        path.write_bytes(gzip.compress(payload, compresslevel=9, mtime=0))
    else:
        path.write_bytes(payload)
    return sha256_file(path)
