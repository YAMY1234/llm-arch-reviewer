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
from llm_arch_v2.profile_acceptance import (
    validate_executable_drill_rollups,
    validate_profile_timing_closure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_root", type=Path)
    parser.add_argument(
        "--profile-id",
        action="append",
        help="materialize only this profile id; repeatable",
    )
    return parser.parse_args()


def _decoded(value: Any, strings: list[str]) -> Any:
    return strings[value] if isinstance(value, int) else value


def drop_derived_parent_targets(
    targets: list[str],
    *,
    rollup_targets: set[str],
    scoped_targets: set[str],
) -> list[str]:
    """Keep direct/fusion evidence while discarding stale derived parents."""

    derived = rollup_targets | scoped_targets
    return [target for target in targets if target not in derived]


def drop_derived_fusion_members(
    fusion_groups: dict[str, dict[str, Any]],
    *,
    rollup_targets: set[str],
    scoped_targets: set[str],
) -> None:
    """Remove stale derived parents from physical fusion memberships.

    Drill roll-ups and authored timing scopes are interval unions computed from
    descendant production events.  They are navigation/aggregation nodes, not
    semantic leaves sharing an owner's physical event set.  Older generated
    profiles may nevertheless contain them in a fusion group; retaining that
    membership while re-materializing would reattach every owner event and
    destroy the scope filter.
    """

    derived = rollup_targets | scoped_targets
    for group in fusion_groups.values():
        owner = str(group.get("owner") or "")
        group["ir_nodes"] = [
            target
            for target in group.get("ir_nodes") or []
            if target == owner or target not in derived
        ]
        evidence_scope = group.get("evidence_scope") or {}
        member_event_ids = evidence_scope.get("member_event_ids")
        if isinstance(member_event_ids, dict):
            evidence_scope["member_event_ids"] = {
                target: event_ids
                for target, event_ids in member_event_ids.items()
                if target not in derived
            }


def merge_rollup_metrics(
    profile_metrics: dict[str, dict[str, Any]],
    rollups: dict[str, dict[str, Any]],
) -> None:
    """Refresh derived timing without retaining an incompatible envelope.

    Kernel/attribution evidence remains valid because it is normalized over the
    same selected steps.  An older invocation envelope is not retained unless
    the new roll-up explicitly reconstructs it from the same window: otherwise
    active and elapsed would again describe different samples.
    """

    envelope_fields = {
        "gpu_elapsed_ms",
        "module_gap_ms",
        "other_gpu_work_ms",
        "device_idle_ms",
        "module_active_pct",
        "device_busy_pct",
        "elapsed_scope",
        "gap_semantics",
    }

    for target, metric in rollups.items():
        existing_metric = profile_metrics.get(target)
        if isinstance(existing_metric, dict):
            for field in envelope_fields - metric.keys():
                existing_metric.pop(field, None)
            existing_metric.update(metric)
            try:
                refresh_timing_closure(existing_metric)
            except RuntimeError as exc:
                raise RuntimeError(f"{target}: {exc}") from exc
        else:
            profile_metrics[target] = metric


def refresh_timing_closure(metric: dict[str, Any]) -> None:
    """Recompute every scalar derived from a refreshed active interval union.

    When a roll-up explicitly reconstructs an invocation envelope from the same
    selected steps, derive all dependent values from that one unit.  Callers
    must remove an older, differently sampled envelope before reaching here.
    """

    active = metric.get("active_gpu_ms")
    residency = metric.get("gpu_residency_ms")
    if isinstance(active, (int, float)) and isinstance(residency, (int, float)):
        metric["gpu_overlap_ms"] = round(max(0.0, residency - active), 6)

    elapsed = metric.get("gpu_elapsed_ms")
    if not isinstance(active, (int, float)) or not isinstance(
        elapsed, (int, float)
    ):
        return
    if active > elapsed + 1e-5:
        raise RuntimeError(
            "refreshed per-iteration active time exceeds its invocation "
            f"envelope: active={active} ms elapsed={elapsed} ms"
        )

    gap = max(0.0, elapsed - active)
    metric["module_gap_ms"] = round(gap, 6)
    metric["module_active_pct"] = round(
        100.0 * active / elapsed if elapsed > 0.0 else 0.0,
        2,
    )

    busy_pct = metric.get("device_busy_pct")
    if isinstance(busy_pct, (int, float)) and elapsed > 0.0:
        busy = min(elapsed, max(active, elapsed * busy_pct / 100.0))
        metric["other_gpu_work_ms"] = round(max(0.0, busy - active), 6)
        metric["device_idle_ms"] = round(max(0.0, elapsed - busy), 6)
        metric["device_busy_pct"] = round(100.0 * busy / elapsed, 2)
        return

    idle = metric.get("device_idle_ms")
    if isinstance(idle, (int, float)):
        idle = min(gap, max(0.0, idle))
        metric["device_idle_ms"] = round(idle, 6)
        metric["other_gpu_work_ms"] = round(max(0.0, gap - idle), 6)


def drop_stale_structural_compute_states(
    node_states: dict[str, dict[str, Any]],
    *,
    model_nodes: dict[str, dict[str, Any]],
) -> None:
    """Remove legacy structural overrides from runtime-bearing compute nodes.

    ``structural`` describes a semantic/control/state boundary.  It must never
    be used as a fallback for a measured primitive leaf or a measured module
    whose timing/fusion evidence has not yet been materialized.
    """

    for target, state in list(node_states.items()):
        node = model_nodes.get(target) or {}
        runtime = (node.get("semantic_details") or {}).get(
            "runtime_mapping"
        ) or {}
        if (
            runtime.get("expectation") == "measured"
            and state.get("status") == "structural"
        ):
            node_states.pop(target)


def materialize_typed_fusion_states(
    profile: dict[str, Any],
    *,
    decoded_steps: list[dict[str, Any]],
    model_nodes: dict[str, dict[str, Any]],
    excluded_targets: set[str] | None = None,
) -> None:
    """Replace compute-as-structural fallbacks with physical fusion relations.

    Production events retain one primary timing owner (``ir_node``) and all
    semantic contracts covered by that event (``ir_targets``).  A secondary
    semantic target may cover all or only part of its owner's event set.  Both
    are legitimate fusion, but neither is a structural boundary.
    """

    excluded_targets = excluded_targets or set()
    events = [
        event
        for step in decoded_steps
        for event in step.get("events") or []
    ]
    owner_events: dict[str, list[dict[str, Any]]] = {}
    target_events: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        owner = str(event.get("ir_node") or "")
        if owner:
            owner_events.setdefault(owner, []).append(event)
        for target in event.get("ir_targets") or []:
            target_events.setdefault(str(target), []).append(event)

    def event_id(event: dict[str, Any]) -> str:
        return str(event.get("event_id") or event.get("raw_event_id") or "")

    node_states = profile.setdefault("node_states", {})
    fusion_groups = profile.setdefault("fusion_groups", {})
    candidates = []
    for target, state in list(node_states.items()):
        if target in excluded_targets:
            continue
        node = model_nodes.get(target) or {}
        runtime = (node.get("semantic_details") or {}).get(
            "runtime_mapping"
        ) or {}
        if state.get("status") != "structural":
            continue
        if not (
            state.get("occurrence_fusion_evidence")
            or runtime.get("expectation") == "measured"
        ):
            continue
        physical = target_events.get(target) or []
        if not physical:
            # Leave the state untouched here. The acceptance gate will reject
            # this runtime-bearing compute node until its mapping is fixed or
            # the profile explicitly proves it was not selected.
            continue
        if any(event.get("reconciliation_status") != "closed" for event in physical):
            raise RuntimeError(
                f"{profile.get('profile_id')}: {target} fusion evidence contains "
                "non-closed production events"
            )
        owners: dict[str, list[dict[str, Any]]] = {}
        for event in physical:
            owner = str(event.get("ir_node") or "")
            if owner and owner != target:
                owners.setdefault(owner, []).append(event)
        if owners:
            candidates.append((target, state, owners))

    # Multi-owner semantic reuse is represented directly on the node. The
    # partitions are disjoint physical event sets and carry no copied metric.
    for target, state, owners in candidates:
        if len(owners) <= 1:
            continue
        partitions = []
        for owner, physical in sorted(owners.items()):
            partitions.append(
                {
                    "included_in": owner,
                    "production_event_ids": sorted(
                        event_id(event) for event in physical if event_id(event)
                    ),
                    "occurrence_ids": sorted(
                        {
                            str(event.get("occurrence_id"))
                            for event in physical
                            if event.get("occurrence_id")
                        }
                    ),
                    "layer_ids": sorted(
                        {
                            int(event["layer_id"])
                            for event in physical
                            if event.get("layer_id") is not None
                        }
                    ),
                }
            )
        state.clear()
        state.update(
            {
                "status": "fused_by_occurrence",
                "label": (
                    "semantic node is fused into different timing owners in "
                    "disjoint validated occurrences"
                ),
                "fusion_partitions": partitions,
                "provenance": "production_event_graph",
            }
        )

    # Single-owner subset/equality relations are grouped by owner so every
    # member has an explicit event set under one additive timing owner.
    candidates_by_owner: dict[str, list[str]] = {}
    for target, _state, owners in candidates:
        if len(owners) == 1:
            candidates_by_owner.setdefault(next(iter(owners)), []).append(target)

    for owner, new_members in sorted(candidates_by_owner.items()):
        matching = [
            (group_id, group)
            for group_id, group in fusion_groups.items()
            if str(group.get("owner") or "") == owner
        ]
        if len(matching) > 1:
            raise RuntimeError(
                f"{profile.get('profile_id')}: owner {owner} has multiple fusion groups"
            )
        group_id, group = matching[0] if matching else (
            f"fusion:{owner}",
            {
                "owner": owner,
                "ir_nodes": [owner],
                "provenance": "production_event_graph",
                "mapping_method": "eager_semantic_to_production_event_closure",
                "confidence": "exact",
            },
        )
        members = list(
            dict.fromkeys(
                [
                    *(target for target in group.get("ir_nodes") or [] if target != owner),
                    *sorted(new_members),
                ]
            )
        )
        owner_ids = sorted(
            event_id(event)
            for event in owner_events.get(owner, [])
            if event_id(event)
        )
        if not owner_ids:
            raise RuntimeError(
                f"{profile.get('profile_id')}: fusion owner {owner} has no physical events"
            )
        member_ids = {
            member: sorted(
                event_id(event)
                for event in target_events.get(member, [])
                if event_id(event)
            )
            for member in members
        }
        owner_set = set(owner_ids)
        for member, ids in member_ids.items():
            if not ids or not set(ids).issubset(owner_set):
                raise RuntimeError(
                    f"{profile.get('profile_id')}: fusion member {member} does "
                    f"not close under owner {owner}"
                )
        scoped_events = [event for member in members for event in target_events[member]]
        group.update(
            {
                "owner": owner,
                "ir_nodes": [owner, *members],
                "timing_semantics": "shared_event_coverage",
                "evidence_scope": {
                    "resolution": "profile_aggregate",
                    "owner_event_ids": owner_ids,
                    "member_event_ids": member_ids,
                    "layer_ids": sorted(
                        {
                            int(event["layer_id"])
                            for event in scoped_events
                            if event.get("layer_id") is not None
                        }
                    ),
                    "occurrence_ids": sorted(
                        {
                            str(event.get("occurrence_id"))
                            for event in scoped_events
                            if event.get("occurrence_id")
                        }
                    ),
                    "all_owner_events_same_rank_closed": True,
                },
            }
        )
        fusion_groups[group_id] = group
        for member in members:
            state = node_states.setdefault(member, {})
            state.clear()
            state.update(
                {
                    "status": "fused",
                    "included_in": owner,
                    "fusion_group_id": group_id,
                    "label": (
                        f"fused into {owner}; {len(member_ids[member])}/"
                        f"{len(owner_ids)} owner events"
                    ),
                    "provenance": "production_event_graph",
                }
            )


def main() -> int:
    args = parse_args()
    model_ir = yaml.safe_load((args.model_root / "model_ir.yaml").read_text())
    selected_profile_ids = set(args.profile_id or [])
    model_nodes = {
        f"{view_id}.{node['id']}": node
        for view_id, view in (model_ir.get("views") or {}).items()
        for node in view.get("nodes") or []
    }
    measured_drill_targets = {
        target
        for target, node in model_nodes.items()
        if node.get("drill")
        and (
            ((node.get("semantic_details") or {}).get("runtime_mapping") or {})
            .get("expectation")
            == "measured"
        )
    }
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
        if selected_profile_ids and profile.get("profile_id") not in selected_profile_ids:
            continue
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
        drop_derived_fusion_members(
            profile.setdefault("fusion_groups", {}),
            rollup_targets=rollup_targets,
            scoped_targets=scoped_targets,
        )
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
                # Re-materialization must not preserve stale derived parent
                # targets from an older attribution policy.  Every drill or
                # scoped parent is reattached below from the current Model IR
                # ancestry/scope contract.  Keeping an old parent target here
                # can make an adjacent norm or collective look like part of a
                # drilled module even after its boundary has been corrected.
                existing = drop_derived_parent_targets(
                    existing,
                    rollup_targets=rollup_targets,
                    scoped_targets=scoped_targets,
                )
                decoded_event = {
                    "event_id": _decoded(event.get("event_id"), strings),
                    "raw_event_id": _decoded(event.get("raw_event_id"), strings),
                    "start_us": event["start_us"],
                    "duration_us": event["duration_us"],
                    "ir_node": node,
                    "layer_id": event.get("layer_id"),
                    "layer_kind": _decoded(event.get("layer_kind"), strings),
                    "substage": _decoded(event.get("substage"), strings),
                    "segment_id": event.get("segment_id"),
                    "occurrence_id": _decoded(event.get("occurrence_id"), strings),
                    "reconciliation_status": _decoded(
                        event.get("reconciliation_status"), strings
                    ),
                }
                targets = expand_rollup_targets(
                    node,
                    existing_targets=existing,
                    ancestors=ancestors,
                    fusion_groups=profile.get("fusion_groups") or {},
                    event_id=str(decoded_event.get("event_id") or "") or None,
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

        materialize_typed_fusion_states(
            profile,
            decoded_steps=decoded_steps,
            model_nodes=model_nodes,
            excluded_targets=rollup_targets | scoped_targets,
        )
        drop_stale_structural_compute_states(
            profile.setdefault("node_states", {}), model_nodes=model_nodes
        )

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
            rollup_targets=(
                rollup_targets
                | direct_targets
                | scoped_targets
                | measured_drill_targets
            ),
        )
        direct_metrics = direct_metrics_from_events(
            decoded_steps,
            direct_targets=direct_targets,
        )
        rollups = {
            target: metric
            for target, metric in timing_metrics.items()
            if target in rollup_targets
            or target in scoped_targets
            or target in measured_drill_targets
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
            if target not in direct_metrics:
                continue
            measured = (profile.get("node_metrics") or {}).get(target)
            # This command materializes missing timing and parent roll-ups.  A
            # previously accepted direct leaf already carries richer evidence
            # (kernel shares, attribution methods, rank policy) and must not be
            # rewritten from the compact timeline artifact.
            if measured is not None:
                timing = direct_metrics[target]
                measured.setdefault("active_gpu_ms", timing["active_gpu_ms"])
                measured.setdefault("gpu_residency_ms", timing["gpu_residency_ms"])
                # These two names are schema aliases for the same per-iteration
                # value.  Do not combine an accepted rich metric with a compact
                # timeline re-attribution that may use a newer direct-owner
                # boundary; preserve the accepted metric's own value.
                measured.setdefault(
                    "gpu_residency_ms_per_iter",
                    measured["gpu_residency_ms"],
                )
                method_rows = (
                    (measured.get("attribution") or {}).get("methods") or {}
                )
                method_count = sum(
                    int(row.get("kernel_count") or 0)
                    for row in method_rows.values()
                    if isinstance(row, dict)
                )
                if method_count:
                    measured["mapped_event_count"] = method_count
                else:
                    measured.setdefault(
                        "mapped_event_count", timing["mapped_event_count"]
                    )
                continue
            model_node = model_nodes.get(target)
            runtime = (
                (model_node or {}).get("semantic_details") or {}
            ).get("runtime_mapping") or {}
            if model_node is None or runtime.get("expectation") in {
                "structural",
                "fused",
                "fused_state",
                "not_selected",
            }:
                continue
            measured = {}
            profile.setdefault("node_metrics", {})[target] = measured
            timing = direct_metrics[target]
            measured.update(
                {
                    "ms_per_iter": timing["ms_per_iter"],
                    "active_gpu_ms": timing["active_gpu_ms"],
                    "gpu_residency_ms": timing["gpu_residency_ms"],
                    "gpu_residency_ms_per_iter": timing[
                        "gpu_residency_ms_per_iter"
                    ],
                    "gpu_overlap_ms": round(
                        max(
                            0.0,
                            timing["gpu_residency_ms"]
                            - timing["active_gpu_ms"],
                        ),
                        6,
                    ),
                    "attribution_status": "measured_direct",
                    "metric_kind": "exclusive_leaf",
                    "timing_semantics": timing["timing_semantics"],
                    "mapped_event_count": timing["mapped_event_count"],
                }
            )
        # A roll-up refresh owns the derived interval-union fields, but it must
        # not erase richer evidence already attached to the node (kernel
        # breakdowns, attribution methods, source symbols, etc.).  Replacing
        # the whole mapping made a targeted timing repair silently degrade the
        # published profile.  Merge the derived fields into the existing cell
        # instead; new scoped parents still receive a fresh complete metric.
        profile_metrics = profile.setdefault("node_metrics", {})
        merge_rollup_metrics(profile_metrics, rollups)
        for target in rollups:
            profile.get("node_states", {}).pop(target, None)

        validate_profile_timing_closure(profile)
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
