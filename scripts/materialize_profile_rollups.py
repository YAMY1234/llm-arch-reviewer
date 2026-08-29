#!/usr/bin/env python3
"""Materialize model-independent drill roll-ups into profile/timeline files."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

import yaml

from models.common.profile_rollup import (
    add_execution_boundary_ancestors,
    direct_metrics_from_events,
    expand_rollup_targets,
    rollup_metrics_from_events,
    unique_drill_ancestors,
)
from models.common.timeline_artifact import write_timeline_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_root", type=Path)
    return parser.parse_args()


def _decoded(value: Any, strings: list[str]) -> Any:
    return strings[value] if isinstance(value, int) else value


def main() -> int:
    args = parse_args()
    model_ir = yaml.safe_load((args.model_root / "model_ir.yaml").read_text())
    changed = []
    for profile_path in sorted((args.model_root / "profiles").glob("**/*.yaml")):
        profile = yaml.safe_load(profile_path.read_text())
        ancestors = unique_drill_ancestors(
            model_ir, node_states=profile.get("node_states") or {}
        )
        execution_path = args.model_root / "execution_paths" / (
            f"{profile['execution_path_id']}.yaml"
        )
        if execution_path.exists():
            ancestors = add_execution_boundary_ancestors(
                ancestors, yaml.safe_load(execution_path.read_text())
            )
        rollup_targets = {
            target for values in ancestors.values() for target in values
        }
        timeline_spec = profile.get("timeline") or {}
        artifact_name = timeline_spec.get("artifact")
        if not artifact_name:
            continue
        timeline_path = profile_path.parent / artifact_name
        with gzip.open(timeline_path, "rt") as source:
            timeline = json.load(source)
        strings = timeline["strings"]
        string_index = {value: index for index, value in enumerate(strings)}

        def intern(value: str) -> int:
            if value not in string_index:
                string_index[value] = len(strings)
                strings.append(value)
            return string_index[value]

        decoded_steps = []
        direct_targets: set[str] = set()
        for step in timeline.get("steps") or []:
            decoded_events = []
            for event in step.get("events") or []:
                node = _decoded(event.get("ir_node"), strings)
                if node:
                    direct_targets.add(str(node))
                existing = [
                    _decoded(target, strings)
                    for target in event.get("ir_targets") or []
                ]
                targets = expand_rollup_targets(
                    node,
                    existing_targets=existing,
                    ancestors=ancestors,
                    fusion_groups=profile.get("fusion_groups") or {},
                )
                event["ir_targets"] = [intern(target) for target in targets]
                decoded_events.append(
                    {
                        "start_us": event["start_us"],
                        "duration_us": event["duration_us"],
                        "ir_node": node,
                        "ir_targets": targets,
                    }
                )
            decoded_steps.append({"events": decoded_events})

        timing_metrics = rollup_metrics_from_events(
            decoded_steps,
            rollup_targets=rollup_targets | direct_targets,
        )
        direct_metrics = direct_metrics_from_events(
            decoded_steps,
            direct_targets=direct_targets,
        )
        rollups = {
            target: metric
            for target, metric in timing_metrics.items()
            if target in rollup_targets
        }
        for target in direct_targets:
            if target not in direct_metrics or target not in profile.get("node_metrics", {}):
                continue
            measured = profile["node_metrics"][target]
            timing = direct_metrics[target]
            measured.update(
                {
                    "active_gpu_ms": timing["active_gpu_ms"],
                    "gpu_residency_ms": timing["gpu_residency_ms"],
                    "attribution_status": "measured_direct",
                    "metric_kind": "exclusive_leaf",
                }
            )
        profile.setdefault("node_metrics", {}).update(rollups)
        for target in rollups:
            profile.get("node_states", {}).pop(target, None)

        timeline_sha = write_timeline_artifact(timeline_path, timeline)
        profile["timeline"]["sha256"] = timeline_sha
        profile_path.write_text(
            yaml.safe_dump(profile, sort_keys=False, width=1000)
        )
        changed.append(
            {
                "profile": profile["profile_id"],
                "rollups": sorted(rollups),
            }
        )

    print(json.dumps(changed, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
