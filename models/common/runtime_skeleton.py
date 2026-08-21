#!/usr/bin/env python3
"""Build an observed runtime architecture skeleton from kernel mappings.

This is deliberately not a full architecture IR builder. It only records what
the trace proves: module order, stack-derived semantic nodes, kernel evidence,
and validation against optional expected runtime patterns.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised manually
    raise SystemExit("requires pyyaml") from exc


_MODULE_INDEX_RE = re.compile(r"^(?P<class>.+)_(?P<index>\d+)$")


def load_json(path: Path) -> Any:
    with path.open() as fh:
        return json.load(fh)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")


def write_yaml(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True)


def split_module_name(module: str) -> tuple[str, int | None]:
    match = _MODULE_INDEX_RE.match(module)
    if not match:
        return module, None
    return match.group("class"), int(match.group("index"))


def frame_raw(frame: dict[str, Any] | None) -> str | None:
    return frame.get("raw") if isinstance(frame, dict) else None


def event_modules(event: dict[str, Any]) -> list[str]:
    return [
        frame["module"]
        for frame in event.get("python_stack", []) or []
        if isinstance(frame, dict) and frame.get("module")
    ]


def source_frames(event: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        frame
        for frame in event.get("python_stack", []) or []
        if isinstance(frame, dict) and frame.get("file")
    ]


def duration_ms(us: float) -> float:
    return round(us / 1000.0, 6)


def _mapping_by_event_id(mappings: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {mapping["event_id"]: mapping for mapping in mappings}


def _events_by_id(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {event["event_id"]: event for event in events}


def _top_counter(counter: Counter[str], limit: int = 12) -> list[dict[str, Any]]:
    return [{"value": key, "count": value} for key, value in counter.most_common(limit)]


def _top_duration(counter: Counter[str], limit: int = 12) -> list[dict[str, Any]]:
    return [
        {"value": key, "dur_ms": duration_ms(value)}
        for key, value in counter.most_common(limit)
    ]


def _top_ms(counter: Counter[str], limit: int = 12) -> list[dict[str, Any]]:
    return [
        {"value": key, "dur_ms": round(float(value), 6)}
        for key, value in counter.most_common(limit)
    ]


def collect_module_inventory(events: list[dict[str, Any]]) -> dict[str, Any]:
    class_indices: dict[str, set[int]] = defaultdict(set)
    no_index_classes: set[str] = set()
    module_counts: Counter[str] = Counter()
    class_event_counts: Counter[str] = Counter()

    for event in events:
        for module in event_modules(event):
            module_counts[module] += 1
            cls, idx = split_module_name(module)
            class_event_counts[cls] += 1
            if idx is None:
                no_index_classes.add(cls)
            else:
                class_indices[cls].add(idx)

    inventory: dict[str, Any] = {}
    for cls in sorted(set(class_indices) | no_index_classes):
        indices = sorted(class_indices.get(cls, set()))
        inventory[cls] = {
            "observed_instance_count": len(indices) if indices else (1 if cls in no_index_classes else 0),
            "observed_indices": indices,
            "event_count": class_event_counts[cls],
        }
    return inventory


def build_layer_sequence(
    *,
    events: list[dict[str, Any]],
    mappings_by_event: dict[str, dict[str, Any]],
    layer_modules: list[dict[str, str]],
) -> list[dict[str, Any]]:
    class_to_kind = {item["class"]: item["kind"] for item in layer_modules}
    first_ts: dict[str, float] = {}
    layer_events: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for event in events:
        modules = event_modules(event)
        for module in modules:
            cls, _idx = split_module_name(module)
            if cls not in class_to_kind:
                continue
            first_ts.setdefault(module, float(event.get("ts_us", 0.0)))
            layer_events[module].append(event)

    out: list[dict[str, Any]] = []
    for global_index, module in enumerate(sorted(first_ts, key=lambda item: first_ts[item])):
        cls, local_index = split_module_name(module)
        selected_nodes: Counter[str] = Counter()
        selected_node_us: Counter[str] = Counter()
        child_modules: dict[str, set[int]] = defaultdict(set)
        child_no_index: set[str] = set()
        source_frame_counts: Counter[str] = Counter()
        kernel_count = 0
        total_us = 0.0
        event_ids: list[str] = []

        for event in layer_events[module]:
            event_ids.append(event["event_id"])
            kernel_count += 1
            total_us += float(event.get("dur_us") or 0.0)
            mapping = mappings_by_event.get(event["event_id"], {})
            node = mapping.get("selected_node") or "UNMAPPED"
            selected_nodes[node] += 1
            selected_node_us[node] += float(event.get("dur_us") or 0.0)
            for child in event_modules(event):
                if child == module:
                    break
                child_cls, child_idx = split_module_name(child)
                if child_idx is None:
                    child_no_index.add(child_cls)
                else:
                    child_modules[child_cls].add(child_idx)
            for frame in source_frames(event):
                source_frame_counts[frame["raw"]] += 1

        child_inventory: dict[str, Any] = {}
        for child_cls in sorted(set(child_modules) | child_no_index):
            indices = sorted(child_modules.get(child_cls, set()))
            child_inventory[child_cls] = {
                "observed_instance_count": len(indices) if indices else (1 if child_cls in child_no_index else 0),
                "observed_indices": indices,
            }

        out.append(
            {
                "global_index": global_index,
                "module": module,
                "module_class": cls,
                "module_index": local_index,
                "kind": class_to_kind[cls],
                "first_ts_us": first_ts[module],
                "kernel_count": kernel_count,
                "total_kernel_ms": duration_ms(total_us),
                "event_ids_sample": event_ids[:20],
                "child_module_inventory": child_inventory,
                "selected_nodes_by_count": _top_counter(selected_nodes),
                "selected_nodes_by_duration": _top_duration(selected_node_us),
                "source_frames_sample": _top_counter(source_frame_counts, limit=16),
            }
        )
    return out


def build_runtime_nodes(
    *,
    events: list[dict[str, Any]],
    mappings: list[dict[str, Any]],
) -> dict[str, Any]:
    events_by_id = _events_by_id(events)
    nodes: dict[str, Any] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for mapping in mappings:
        grouped[mapping.get("selected_node") or "UNMAPPED"].append(mapping)

    for node, node_mappings in sorted(grouped.items()):
        kernel_names: Counter[str] = Counter()
        kernel_us: Counter[str] = Counter()
        semantic_frames: Counter[str] = Counter()
        operator_frames: Counter[str] = Counter()
        primitive_frames: Counter[str] = Counter()
        model_contexts: Counter[str] = Counter()
        event_ids: list[str] = []
        total_us = 0.0

        for mapping in node_mappings:
            event = events_by_id.get(mapping["event_id"], {})
            dur_us = float(event.get("dur_us") or 0.0)
            total_us += dur_us
            event_ids.append(mapping["event_id"])
            kernel = str(event.get("kernel_name") or mapping.get("kernel_name") or "")
            kernel_names[kernel] += 1
            kernel_us[kernel] += dur_us
            for key, counter in [
                ("semantic_frame", semantic_frames),
                ("operator_frame", operator_frames),
                ("primitive_frame", primitive_frames),
                ("model_context_frame", model_contexts),
            ]:
                raw = frame_raw(mapping.get(key))
                if raw:
                    counter[raw] += 1

        nodes[node] = {
            "kernel_count": len(node_mappings),
            "total_kernel_ms": duration_ms(total_us),
            "event_ids_sample": event_ids[:24],
            "top_kernels_by_count": _top_counter(kernel_names),
            "top_kernels_by_duration": _top_duration(kernel_us),
            "semantic_frames": _top_counter(semantic_frames),
            "operator_frames": _top_counter(operator_frames),
            "primitive_frames": _top_counter(primitive_frames),
            "model_context_frames": _top_counter(model_contexts),
        }
    return nodes


def build_layer_kind_patterns(layer_sequence: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for layer in layer_sequence:
        grouped[layer["kind"]].append(layer)

    patterns: dict[str, Any] = {}
    for kind, layers in sorted(grouped.items()):
        child_presence: Counter[str] = Counter()
        child_instances: dict[str, set[int]] = defaultdict(set)
        selected_node_counts: Counter[str] = Counter()
        selected_node_ms: Counter[str] = Counter()
        source_frames: Counter[str] = Counter()

        for layer in layers:
            for child_cls, child_cell in layer.get("child_module_inventory", {}).items():
                child_presence[child_cls] += 1
                for idx in child_cell.get("observed_indices") or []:
                    child_instances[child_cls].add(idx)
            for item in layer.get("selected_nodes_by_count") or []:
                selected_node_counts[item["value"]] += int(item["count"])
            for item in layer.get("selected_nodes_by_duration") or []:
                selected_node_ms[item["value"]] += float(item["dur_ms"])
            for item in layer.get("source_frames_sample") or []:
                source_frames[item["value"]] += int(item["count"])

        patterns[kind] = {
            "layer_count": len(layers),
            "global_indices": [layer["global_index"] for layer in layers],
            "module_classes": [
                {
                    "class": child_cls,
                    "layers_present": child_presence[child_cls],
                    "observed_instance_count": len(child_instances.get(child_cls, set())),
                    "observed_indices_sample": sorted(child_instances.get(child_cls, set()))[:24],
                }
                for child_cls in sorted(child_presence)
            ],
            "runtime_nodes_by_count": _top_counter(selected_node_counts, limit=24),
            "runtime_nodes_by_duration": _top_ms(selected_node_ms, limit=24),
            "source_frames_sample": _top_counter(source_frames, limit=24),
        }
    return patterns


def compact_kind_sequence(layer_sequence: list[dict[str, Any]]) -> dict[str, Any]:
    sequence = [item["kind"] for item in layer_sequence]
    if not sequence:
        return {"sequence": [], "period": [], "repeat": 0}
    for period_len in range(1, len(sequence) + 1):
        if len(sequence) % period_len:
            continue
        period = sequence[:period_len]
        if period * (len(sequence) // period_len) == sequence:
            return {
                "sequence": sequence,
                "period": period,
                "repeat": len(sequence) // period_len,
            }
    return {"sequence": sequence, "period": sequence, "repeat": 1}


def validate_skeleton(
    *,
    skeleton: dict[str, Any],
    mappings: list[dict[str, Any]],
    expected: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    inventory = skeleton["module_inventory"]
    for cls, expected_count in (expected.get("module_counts") or {}).items():
        actual = inventory.get(cls, {}).get("observed_instance_count", 0)
        if actual != expected_count:
            errors.append(f"module class {cls}: expected {expected_count}, got {actual}")

    layer_sequence = skeleton["layer_sequence"]
    expected_layer_count = expected.get("layer_count")
    if expected_layer_count is not None and len(layer_sequence) != expected_layer_count:
        errors.append(
            f"layer_sequence length: expected {expected_layer_count}, got {len(layer_sequence)}"
        )

    expected_pattern = expected.get("layer_pattern") or []
    if expected_pattern and layer_sequence:
        for item in layer_sequence:
            expected_kind = expected_pattern[item["global_index"] % len(expected_pattern)]
            if item["kind"] != expected_kind:
                errors.append(
                    f"layer {item['global_index']} pattern mismatch: "
                    f"expected {expected_kind}, got {item['kind']}"
                )

    coarse_nodes = set(expected.get("coarse_nodes") or [])
    observed_coarse = sorted(node for node in skeleton["runtime_nodes"] if node in coarse_nodes)
    if observed_coarse:
        warnings.append(f"coarse runtime nodes need source/shape splitting: {observed_coarse}")

    for mapping in mappings:
        if not mapping.get("selected_node"):
            continue
        if not mapping.get("semantic_frame"):
            errors.append(f"{mapping['event_id']} has selected_node without semantic_frame")

    return {"ok": not errors, "errors": errors, "warnings": warnings}


def build_runtime_skeleton(
    *,
    events_path: Path,
    mapping_path: Path,
    manifest_path: Path,
    skeleton_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    events = load_jsonl(events_path)
    mappings = load_jsonl(mapping_path)
    manifest = load_json(manifest_path)
    mappings_by_event = _mapping_by_event_id(mappings)

    module_inventory = collect_module_inventory(events)
    layer_sequence = build_layer_sequence(
        events=events,
        mappings_by_event=mappings_by_event,
        layer_modules=skeleton_config.get("layer_modules") or [],
    )
    layer_kind_patterns = build_layer_kind_patterns(layer_sequence)
    runtime_nodes = build_runtime_nodes(events=events, mappings=mappings)

    skeleton = {
        "schema_version": "runtime_skeleton.v0",
        "source": {
            "manifest_path": str(manifest_path),
            "events_path": str(events_path),
            "mapping_path": str(mapping_path),
            "trace_path": manifest.get("trace_path"),
            "source_repo": manifest.get("source_repo"),
            "source_commit": manifest.get("source_commit"),
            "phase": manifest.get("phase"),
            "rank": manifest.get("rank"),
            "window": manifest.get("window"),
        },
        "generation_policy": {
            "description": "Trace-observed skeleton only; no existing hand-written architecture IR is read.",
            "layer_modules": skeleton_config.get("layer_modules") or [],
        },
        "module_inventory": module_inventory,
        "observed_layer_kind_sequence": compact_kind_sequence(layer_sequence),
        "layer_kind_patterns": layer_kind_patterns,
        "layer_sequence": layer_sequence,
        "runtime_nodes": runtime_nodes,
    }
    validation = validate_skeleton(
        skeleton=skeleton,
        mappings=mappings,
        expected=skeleton_config.get("expected") or {},
    )
    report = {
        "ok": validation["ok"],
        "errors": validation["errors"],
        "warnings": validation["warnings"],
        "summary": {
            "kernel_count": len(events),
            "mapping_count": len(mappings),
            "runtime_node_count": len(runtime_nodes),
            "observed_layer_count": len(layer_sequence),
            "observed_layer_period": skeleton["observed_layer_kind_sequence"]["period"],
            "observed_layer_period_repeat": skeleton["observed_layer_kind_sequence"]["repeat"],
            "module_class_count": len(module_inventory),
        },
        "layer_kinds": Counter(item["kind"] for item in layer_sequence),
        "top_runtime_nodes_by_duration": sorted(
            [
                {
                    "node": node,
                    "total_kernel_ms": cell["total_kernel_ms"],
                    "kernel_count": cell["kernel_count"],
                }
                for node, cell in runtime_nodes.items()
            ],
            key=lambda item: -float(item["total_kernel_ms"]),
        )[:24],
    }
    skeleton["validation"] = validation
    return skeleton, report


def write_skeleton_report_markdown(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Runtime Skeleton Report",
        "",
        f"- ok: `{report['ok']}`",
        f"- kernel_count: `{report['summary']['kernel_count']}`",
        f"- runtime_node_count: `{report['summary']['runtime_node_count']}`",
        f"- observed_layer_count: `{report['summary']['observed_layer_count']}`",
        f"- observed_layer_period: `{report['summary']['observed_layer_period']}` × {report['summary']['observed_layer_period_repeat']}",
        f"- module_class_count: `{report['summary']['module_class_count']}`",
        "",
        "## Layer Kinds",
        "",
    ]
    for kind, count in sorted(dict(report["layer_kinds"]).items()):
        lines.append(f"- `{kind}`: {count}")
    lines.extend(["", "## Top Runtime Nodes By Duration", ""])
    for item in report["top_runtime_nodes_by_duration"]:
        lines.append(
            f"- `{item['node']}`: {item['total_kernel_ms']:.3f} ms, "
            f"kernels={item['kernel_count']}"
        )
    lines.extend(["", "## Errors", ""])
    lines.extend([f"- {msg}" for msg in report["errors"]] or ["- none"])
    lines.extend(["", "## Warnings", ""])
    lines.extend([f"- {msg}" for msg in report["warnings"]] or ["- none"])
    path.write_text("\n".join(lines) + "\n")
