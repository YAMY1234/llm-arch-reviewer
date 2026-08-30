#!/usr/bin/env python3
"""Materialize model-independent drill roll-ups into profile/timeline files."""

from __future__ import annotations

import argparse
import copy
import gzip
import json
from pathlib import Path
from typing import Any

import yaml

from models.common.profile_rollup import (
    add_execution_boundary_ancestors,
    direct_metrics_from_events,
    event_matches_scope,
    expand_rollup_targets,
    rollup_metrics_from_events,
    scoped_steps,
    unique_drill_ancestors,
)
from models.common.timeline_artifact import write_timeline_artifact
from llm_arch_v2.profile_acceptance import validate_executable_drill_rollups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_root", type=Path)
    return parser.parse_args()


def _decoded(value: Any, strings: list[str]) -> Any:
    return strings[value] if isinstance(value, int) else value


def main() -> int:
    args = parse_args()
    model_ir = yaml.safe_load((args.model_root / "model_ir.yaml").read_text())
    timing_scope_contracts = model_ir.get("timing_scope_contracts") or []
    scoped_targets = {
        str(contract["target_node"]) for contract in timing_scope_contracts
    }
    view_targets = {
        str(view_id): {
            f"{view_id}.{node['id']}" for node in view.get("nodes") or []
        }
        for view_id, view in (model_ir.get("views") or {}).items()
    }
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
                # Re-materialization must not preserve stale profile-wide
                # parent targets from an older attribution policy.  Scoped
                # parents are reattached below only when the event carries
                # the required semantic execution coordinate.
                existing = [
                    target for target in existing if target not in scoped_targets
                ]
                decoded_event = {
                    "start_us": event["start_us"],
                    "duration_us": event["duration_us"],
                    "ir_node": node,
                    "layer_id": event.get("layer_id"),
                    "layer_kind": _decoded(event.get("layer_kind"), strings),
                    "substage": _decoded(event.get("substage"), strings),
                    "segment_id": event.get("segment_id"),
                    "occurrence_id": _decoded(event.get("occurrence_id"), strings),
                }
                targets = expand_rollup_targets(
                    node,
                    existing_targets=existing,
                    ancestors=ancestors,
                    fusion_groups=profile.get("fusion_groups") or {},
                )
                for contract in timing_scope_contracts:
                    if node not in set(contract["source_nodes"]):
                        continue
                    if event_matches_scope(
                        decoded_event,
                        event_filter=contract["event_filter"],
                    ):
                        targets.append(str(contract["target_node"]))
                targets = list(dict.fromkeys(targets))
                event["ir_targets"] = [intern(target) for target in targets]
                decoded_events.append({**decoded_event, "ir_targets": targets})
            decoded_steps.append({"events": decoded_events})

        # A scoped parent is deliverable only when every expected semantic
        # occurrence exists in the production trace.  This rejects a partial
        # window instead of silently publishing a smaller parent time.
        for contract in timing_scope_contracts:
            scoped = scoped_steps(
                decoded_steps,
                event_filter=contract["event_filter"],
                source_nodes=set(contract["source_nodes"]),
            )
            occurrences = {
                str(event["occurrence_id"])
                for step in scoped
                for event in step.get("events") or []
                if event.get("occurrence_id")
            }
            expected = int(contract["required_occurrence_count"])
            if len(occurrences) != expected:
                raise RuntimeError(
                    f"{profile_path}: timing scope {contract['target_node']} "
                    f"requires {expected} occurrences, got {len(occurrences)}"
                )

        timing_metrics = rollup_metrics_from_events(
            decoded_steps,
            rollup_targets=rollup_targets | direct_targets | scoped_targets,
        )
        direct_metrics = direct_metrics_from_events(
            decoded_steps,
            direct_targets=direct_targets,
        )
        rollups = {
            target: metric
            for target, metric in timing_metrics.items()
            if target in rollup_targets or target in scoped_targets
        }
        contracts_by_target = {
            str(contract["target_node"]): contract
            for contract in timing_scope_contracts
        }
        for target, contract in contracts_by_target.items():
            metric = rollups.get(target)
            if metric is None:
                raise RuntimeError(
                    f"{profile_path}: no timing materialized for scoped parent {target}"
                )
            metric["timing_scope"] = copy.deepcopy(contract["event_filter"])
            metric["required_occurrence_count"] = int(
                contract["required_occurrence_count"]
            )
            metric["timing_semantics"] = (
                "union of context-filtered production-owner intervals; "
                "overlap counted once; no child scalar copied"
            )

            drill_view = contract.get("drill_view")
            if not drill_view:
                continue
            scoped = scoped_steps(
                decoded_steps,
                event_filter=contract["event_filter"],
            )
            child_targets = view_targets[str(drill_view)]
            child_direct = direct_metrics_from_events(
                scoped,
                direct_targets=child_targets,
            )
            drill_metrics: dict[str, dict[str, Any]] = {}
            for child_target in sorted(child_targets):
                child_id = child_target.split(".", 1)[1]
                state = (profile.get("node_states") or {}).get(child_target)
                if state is not None:
                    drill_metrics[child_id] = copy.deepcopy(state)
                    continue
                child_metric = child_direct.get(child_target)
                if child_metric is not None:
                    child_metric = copy.deepcopy(child_metric)
                    child_metric["attribution_status"] = "measured_scoped_direct"
                    child_metric["metric_kind"] = "exclusive_leaf_scoped"
                    child_metric["timing_scope"] = copy.deepcopy(
                        contract["event_filter"]
                    )
                    drill_metrics[child_id] = child_metric
            metric["drill_view"] = str(drill_view)
            metric["drill_metrics"] = drill_metrics
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

        validate_executable_drill_rollups(model_ir, profile)

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
