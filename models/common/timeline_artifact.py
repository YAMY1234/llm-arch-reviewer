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
from typing import Any, Iterable


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
    """Return direct and roll-up IR targets for one attributed event."""

    node = str(event.get("node") or "")
    # Attribution builders may already know the exact semantic leaf, fusion
    # members, and drill ancestors for an implementation interval.  Preserve
    # those explicit targets before applying legacy model-name roll-up rules.
    # This keeps the shared timeline schema portable without teaching the
    # viewer (or this helper) every model's node naming convention.
    targets: list[str] = [
        str(target) for target in (event.get("ir_targets") or []) if target
    ]
    if node:
        targets.insert(0, node)
    qsa_indexer_drill_target = event.get("qsa_indexer_drill_target")
    if qsa_indexer_drill_target:
        targets.append(str(qsa_indexer_drill_target))
    layer_kind = event.get("layer_kind")
    substage = event.get("substage")
    layer_view = (
        "linear_layer"
        if layer_kind == "linear"
        else "full_layer" if layer_kind == "full" else None
    )

    # The auxiliary MTP model is a separate semantic scope even though its
    # implementation reuses the target model's QSA, MoE and HC kernels.  Add
    # the stable MTP roll-ups explicitly so either the generation stage or any
    # nested MTP node can find its measured slices on the timeline.
    mtp_stage = (
        "mtp_generation.mtp_prefill"
        if "mtp_prefill" in str(substage or "")
        else "mtp_generation.mtp_draft_extend"
    )
    if node.startswith("mtp_head."):
        targets.append(mtp_stage)
    elif node.startswith("mtp_layer."):
        targets.extend(("mtp_head.decoder_layer", mtp_stage))
    elif node.startswith("mtp_qsa_attention."):
        targets.extend(
            (
                "mtp_layer.qsa_attention",
                "mtp_head.decoder_layer",
                mtp_stage,
            )
        )
    elif node.startswith("mtp_moe."):
        targets.extend(
            ("mtp_layer.moe", "mtp_head.decoder_layer", mtp_stage)
        )

    if node.startswith("hyperconnection."):
        leaf = node.split(".", 1)[1]
        normalized_stage = str(substage or "")
        for prefix in ("mtp_prefill_", "mtp_draft_extend_"):
            if normalized_stage.startswith(prefix):
                normalized_stage = normalized_stage.removeprefix(prefix)
                break
        if normalized_stage in {"attn_hc_mix", "mlp_hc_mix"}:
            targets.append(f"hyperconnection_mix.{leaf}")
        elif normalized_stage in {"attn_hc_combine", "mlp_hc_combine"}:
            targets.append(f"hyperconnection_combine.{leaf}")
        if layer_kind == "mtp" and normalized_stage:
            targets.extend(
                (
                    f"mtp_layer.{normalized_stage}",
                    "mtp_head.decoder_layer",
                    mtp_stage,
                )
            )

    # A direct semantic scope is more specific than the decoder layer that
    # happened to be executing when the kernel was launched.  In particular,
    # PLE runs at a configured decoder layer, so those events legitimately
    # carry ``layer_kind=linear``.  Resolve the direct scope first; otherwise
    # the generic layer roll-up incorrectly makes PLE a child of
    # ``stack.linear_layer`` and leaves ``stack.ple_injection`` with only the
    # layer-less token-history preparation kernels.
    if node.startswith("ple."):
        targets.extend(("stack.ple_injection", "top.decoder_stack"))
    elif layer_view:
        targets.extend((f"stack.{layer_view}", "top.decoder_stack"))
        if substage in {
            "attn_hc_mix",
            "attn_hc_combine",
            "mlp_hc_mix",
            "mlp_hc_combine",
        }:
            targets.append(f"{layer_view}.{substage}")
        elif substage == "attention":
            child = "linear_attention" if layer_kind == "linear" else "qsa_attention"
            targets.append(f"{layer_view}.{child}")
        elif substage == "moe" and node.startswith("moe."):
            targets.append(f"{layer_view}.moe")
    elif node.startswith("stack."):
        targets.append("top.decoder_stack")

    # Preserve order while removing aliases that are already present.
    return list(dict.fromkeys(target for target in targets if target))


def _events_path_for_mapping(mapping_path: Path) -> Path | None:
    name = mapping_path.name
    if not name.startswith("kernel_mapping."):
        return None
    candidate = mapping_path.with_name(name.replace("kernel_mapping.", "events.", 1))
    return candidate if candidate.exists() else None


def load_eager_stack_index(
    mapping_path: Path,
) -> dict[str, dict[Any, list[dict[str, Any]]]]:
    """Index eager Python stacks by IR node and exact ordered occurrence.

    The full stack lives in the sibling ``events.<rank>.jsonl`` file.  If that
    file is unavailable, selected semantic frames in the mapping remain usable
    as explicitly reduced evidence.  No node-only or first-stack index is
    constructed, so an occurrence cannot inherit a representative stack.
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

    by_node_occurrence: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
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
            "rank": mapping.get("rank"),
            "phase": mapping.get("phase"),
            "occurrence_id": mapping.get("occurrence_id"),
            "sequence_index": mapping.get("sequence_index"),
            "closure_status": mapping.get("closure_status"),
        }
        by_node_occurrence[
            (node, str(evidence["occurrence_id"] or ""))
        ].append(evidence)
    for candidates in by_node_occurrence.values():
        candidates.sort(
            key=lambda item: (
                int(item["sequence_index"])
                if item.get("sequence_index") is not None
                else 2**31,
                str(item["event_id"]),
            )
        )
    return {"by_node_occurrence": by_node_occurrence}


def attach_eager_stack_evidence(
    events: Iterable[dict[str, Any]],
    *,
    mapping_path: Path,
    expected_rank: int | None = None,
    expected_phase: str | None = None,
) -> list[dict[str, Any]]:
    """Attach only same-rank, same-phase, occurrence-exact eager stacks.

    A node-level representative is intentionally not a closure: one arbitrary
    stack for a semantic node cannot prove that a materially different kernel
    occurrence, signature, or rank implements that node.
    """

    index = load_eager_stack_index(mapping_path)
    raw_events = [dict(event) for event in events]
    production_sequences: dict[tuple[str, str], list[str]] = defaultdict(list)
    for event in raw_events:
        if event.get("node"):
            production_sequences[
                (
                    str(event.get("node") or ""),
                    str(event.get("occurrence_id") or ""),
                )
            ].append(str(event.get("kernel_name") or ""))
    eager_sequences: dict[tuple[str, str], list[dict[str, Any]]] = {}
    sequence_matches: dict[tuple[str, str], bool] = {}
    for key, production_signature in production_sequences.items():
        candidates = [
            candidate
            for candidate in index["by_node_occurrence"].get(key, [])
            if candidate.get("closure_status") == "closed"
            and (expected_rank is None or candidate.get("rank") == expected_rank)
            and (expected_phase is None or candidate.get("phase") == expected_phase)
        ]
        eager_sequences[key] = candidates
        sequence_matches[key] = [
            str(candidate.get("kernel_name") or "") for candidate in candidates
        ] == production_signature
    seen: Counter[tuple[str, str]] = Counter()
    enriched: list[dict[str, Any]] = []
    for event in raw_events:
        node = str(event.get("node") or "")
        occurrence = str(event.get("occurrence_id") or "")
        key = (node, occurrence)
        ordinal = seen[key]
        seen[key] += 1
        candidates = eager_sequences.get(key, [])
        if sequence_matches.get(key, False) and ordinal < len(candidates):
            evidence = candidates[ordinal]
            event["python_stack"] = evidence["python_stack"]
            event["cpu_op_name"] = evidence.get("cpu_op_name")
            event["eager_event_id"] = evidence["event_id"]
            event["reconciliation_status"] = "closed"
            event["stack_evidence"] = {
                "source": "eager_trace",
                "match": "same_rank_phase_occurrence_signature_ordered_sequence",
                "kind": evidence["stack_kind"],
                "event_id": evidence["event_id"],
                "confidence": evidence["confidence"],
                "rank": str(evidence.get("rank")),
                "phase": str(evidence.get("phase")),
                "occurrence_id": occurrence,
                "sequence_ordinal": str(ordinal),
            }
        elif node:
            event.pop("python_stack", None)
            event["reconciliation_status"] = "typed_unresolved"
            event["reconciliation_reason"] = (
                "no same-rank, same-phase eager event set with an exact "
                "occurrence-scoped kernel-signature sequence and closed semantic stacks"
            )
            event["confidence"] = "review_required"
        enriched.append(event)
    return enriched


def _kernel_kind(event: dict[str, Any]) -> str:
    node = str(event.get("node") or "").lower()
    name = str(event.get("kernel_name") or "").lower()
    if (
        "collective" in node
        or "deepep_dispatch" in node
        or "deepep_combine" in node
        or "nccl" in name
        or "allreduce" in name
        or "allgather" in name
        or "reduce_scatter" in name
    ):
        return "communication"
    if node.startswith(("moe.", "mtp_moe.")):
        return "moe"
    if node.startswith(
        ("linear_attention.", "qsa_attention.", "mtp_qsa_attention.")
    ):
        return "attention"
    if node.startswith(("hyperconnection.", "mtp_layer.")) and any(
        token in node for token in ("hc_mix", "hc_combine")
    ):
        return "hyperconnection"
    if node.startswith("ple."):
        return "ple"
    if any(token in name for token in ("memcpy", "copy_kernel", "fill_kernel")):
        return "memory"
    return "compute"


def _stream_key(event: dict[str, Any]) -> str:
    stream = event.get("stream")
    if stream is None:
        stream = event.get("tid")
    return str(stream if stream is not None else "unknown")


def _stream_tracks(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        grouped[_stream_key(event)].append(event)

    totals = {
        stream: sum(float(event["dur_us"]) for event in stream_events)
        for stream, stream_events in grouped.items()
    }
    roles: dict[str, str] = {}
    non_communication_totals: dict[str, float] = {}
    role_totals: dict[str, Counter[str]] = {}
    for stream, stream_events in grouped.items():
        by_role: Counter[str] = Counter()
        non_communication_total = 0.0
        for event in stream_events:
            duration = float(event["dur_us"])
            node = str(event.get("node") or "")
            kind = _kernel_kind(event)
            if kind == "communication":
                by_role["communication"] += duration
            else:
                non_communication_total += duration
            if node.endswith("qsa_attention.indexer"):
                by_role["QSA indexer"] += duration
            if node.endswith("moe.shared_expert"):
                by_role["shared expert"] += duration
        role_totals[stream] = by_role
        non_communication_totals[stream] = non_communication_total

    # Reserve genuinely dedicated auxiliary compute streams first.  A stream
    # that mixes long TP collectives with HC/attention/MoE work must still be
    # eligible to be the main compute stream; PDL resident-wait intervals can
    # otherwise make communication the largest duration bucket and mislabel
    # the actual execution spine.
    for stream, by_role in role_totals.items():
        dedicated = [
            (role, duration)
            for role, duration in by_role.items()
            if role in {"QSA indexer", "shared expert"}
        ]
        role, duration = max(dedicated, key=lambda item: item[1], default=("", 0.0))
        if totals[stream] and duration / totals[stream] >= 0.45:
            roles[stream] = role

    remaining = [stream for stream in grouped if stream not in roles]
    if remaining:
        main = max(
            remaining,
            key=lambda stream: (non_communication_totals[stream], totals[stream]),
        )
        roles[main] = "main compute"
    for stream in grouped:
        if stream in roles:
            continue
        communication_duration = role_totals[stream]["communication"]
        roles[stream] = (
            "communication"
            if totals[stream] and communication_duration / totals[stream] >= 0.45
            else "auxiliary compute"
        )

    role_order = {
        "main compute": 0,
        "QSA indexer": 1,
        "shared expert": 2,
        "communication": 3,
        "auxiliary compute": 4,
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
        for index, event in enumerate(step_events):
            targets = timeline_targets(event)
            encoded_events.append(
                {
                    "event_id": f"r{reference_rank}-s{raw_step['step_index']}-k{index}",
                    "raw_event_id": strings.add(event.get("event_id")),
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
                    "segment_id": event.get("segment_id"),
                    "occurrence_id": strings.add(event.get("occurrence_id")),
                    "eager_event_id": strings.add(event.get("eager_event_id")),
                    "kernel_kind": strings.add(_kernel_kind(event)),
                    "attribution_method": strings.add(
                        event.get("attribution_method")
                    ),
                    "confidence": strings.add(event.get("confidence")),
                    "reconciliation_status": strings.add(
                        event.get("reconciliation_status")
                    ),
                    "reconciliation_reason": strings.add(
                        event.get("reconciliation_reason")
                    ),
                    "support_class": strings.add(event.get("support_class")),
                    "support_reason": strings.add(event.get("support_reason")),
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
                "tracks": _stream_tracks(step_events),
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
