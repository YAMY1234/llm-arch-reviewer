"""Model-independent timing roll-ups for drillable IR nodes.

Leaf attribution remains the source of truth.  A drillable architecture node
may display the union of its measured descendant intervals, but only when the
Model IR gives that child view one unambiguous parent.  Views reused by several
parents (for example one mHC detail view used by attention and FFN) are not
guessed here; their production binding must provide an explicit fusion/event
scope instead.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable


def unique_drill_ancestors(
    model_ir: dict[str, Any],
    *,
    node_states: dict[str, dict[str, Any]] | None = None,
) -> dict[str, list[str]]:
    """Return each IR target's unambiguous drill-parent chain.

    Reused detail views remain ambiguous in the architecture.  A concrete
    profile may disambiguate them by marking all but one parent disabled or
    not-selected (for example target-model DSA versus an inactive NextN DSA).
    """

    parents_by_view: dict[str, list[str]] = defaultdict(list)
    for view_id, view in (model_ir.get("views") or {}).items():
        for node in view.get("nodes") or []:
            child_view = node.get("drill")
            if child_view:
                parents_by_view[str(child_view)].append(
                    f"{view_id}.{node['id']}"
                )

    result: dict[str, list[str]] = {}
    for view_id, view in (model_ir.get("views") or {}).items():
        for node in view.get("nodes") or []:
            target = f"{view_id}.{node['id']}"
            ancestors: list[str] = []
            current_view = str(view_id)
            seen: set[str] = set()
            candidates = parents_by_view.get(current_view, [])
            if node_states is not None:
                candidates = [
                    parent
                    for parent in candidates
                    if (node_states.get(parent) or {}).get("status")
                    not in {"disabled", "not_selected", "out_of_scope"}
                ]
            while len(candidates) == 1:
                parent = candidates[0]
                if parent in seen:
                    break
                seen.add(parent)
                ancestors.append(parent)
                current_view = parent.split(".", 1)[0]
                candidates = parents_by_view.get(current_view, [])
                if node_states is not None:
                    candidates = [
                        candidate
                        for candidate in candidates
                        if (node_states.get(candidate) or {}).get("status")
                        not in {"disabled", "not_selected", "out_of_scope"}
                    ]
            result[target] = ancestors
    return result


def add_execution_boundary_ancestors(
    ancestors: dict[str, list[str]],
    execution_plan: dict[str, Any],
) -> dict[str, list[str]]:
    """Attach execution-only boundary nodes to the enclosing semantic scope.

    Execution plans may insert a communication node into a drilled child view
    even though that node intentionally does not exist in Model IR.  A
    ``module_boundary`` node is outside the immediate semantic module, but it
    is still inside that module's enclosing decoder/scheduler scope.  Thus a
    TP output collective contributes to the decoder roll-up without being
    misreported as local attention or MoE compute.
    """

    expanded = {target: list(parents) for target, parents in ancestors.items()}
    for transform in execution_plan.get("transforms") or []:
        if transform.get("op") not in {"insert_before", "insert_after"}:
            continue
        anchor = str(transform.get("before") or transform.get("after") or "")
        node = transform.get("node") or {}
        node_id = node.get("id")
        if "." not in anchor or not node_id:
            continue
        view_id = anchor.split(".", 1)[0]
        inserted = f"{view_id}.{node_id}"

        # Any authored node in the same view has the same drill ancestry.  Use
        # all candidates and fail closed when reused-view scope is ambiguous.
        candidates = {
            tuple(parents)
            for target, parents in expanded.items()
            if target.startswith(f"{view_id}.")
        }
        if len(candidates) != 1:
            continue
        chain = list(next(iter(candidates)))
        if node.get("boundary_role") == "module_boundary" and chain:
            chain = chain[1:]
        expanded[inserted] = chain
    return expanded


def expand_rollup_targets(
    node: str | None,
    *,
    existing_targets: Iterable[str] = (),
    ancestors: dict[str, list[str]],
    fusion_groups: dict[str, dict[str, Any]] | None = None,
    event_id: str | None = None,
) -> list[str]:
    """Expand one direct event to fused semantic members and drill parents.

    ``shared_event_coverage`` is intentionally event-aware: a semantic member
    is attached only to the owner events listed for that member. Expanding it
    to every owner event would silently turn a subset relation back into copied
    timing and navigation evidence.
    """

    targets = [str(target) for target in existing_targets if target]
    if node:
        targets.insert(0, str(node))

    # A shared production event belongs to every semantic member of its
    # authored fusion group.  This is many-to-many evidence, not duplicated
    # timing: interval union below still counts the event once per roll-up.
    for group in (fusion_groups or {}).values():
        if group.get("owner") == node:
            members = [str(target) for target in group.get("ir_nodes") or []]
            if group.get("timing_semantics") == "shared_event_coverage":
                member_event_ids = (
                    (group.get("evidence_scope") or {}).get("member_event_ids")
                    or {}
                )
                allowed_members = {
                    member
                    for member in members
                    if member == str(node)
                    or (
                        event_id is not None
                        and str(event_id)
                        in {str(value) for value in member_event_ids.get(member, [])}
                    )
                }
                # Old artifacts may already carry the broader shared-event-set
                # target list.  Re-materialization must narrow those targets to
                # the authored coverage relation instead of preserving them.
                targets = [
                    target
                    for target in targets
                    if target not in members or target in allowed_members
                ]
                targets.extend(
                    member for member in members if member in allowed_members
                )
            else:
                targets.extend(members)

    expanded: list[str] = []
    queue = list(dict.fromkeys(targets))
    while queue:
        target = queue.pop(0)
        if target in expanded:
            continue
        expanded.append(target)
        queue.extend(ancestors.get(target, []))
    return expanded


def _union_duration_us(intervals: Iterable[tuple[float, float]]) -> float:
    ordered = sorted(
        (float(start), float(stop))
        for start, stop in intervals
        if float(stop) > float(start)
    )
    if not ordered:
        return 0.0
    merged = [list(ordered[0])]
    for start, stop in ordered[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], stop)
        else:
            merged.append([start, stop])
    return sum(stop - start for start, stop in merged)


def rollup_metrics_from_events(
    steps: Iterable[dict[str, Any]],
    *,
    rollup_targets: set[str],
) -> dict[str, dict[str, Any]]:
    """Compute non-additive active and additive residency for parent nodes.

    Events must expose ``start_us``, ``duration_us``, decoded ``ir_node`` and a
    decoded ``ir_targets`` list.  Active time is an interval union per formal
    step, then summed across formal steps.  It is never a sum of child metrics.
    """

    selected: dict[str, list[list[dict[str, Any]]]] = {
        target: [] for target in rollup_targets
    }
    step_list = list(steps)
    for target in rollup_targets:
        for step in step_list:
            rows = [
                event
                for event in step.get("events") or []
                if target in (event.get("ir_targets") or [])
            ]
            selected[target].append(rows)

    metrics: dict[str, dict[str, Any]] = {}
    for target, per_step in selected.items():
        rows = [event for step_rows in per_step for event in step_rows]
        if not rows:
            continue
        active_us = sum(
            _union_duration_us(
                (
                    float(event["start_us"]),
                    float(event["start_us"]) + float(event["duration_us"]),
                )
                for event in step_rows
            )
            for step_rows in per_step
        )
        residency_us = sum(float(event["duration_us"]) for event in rows)
        direct_nodes = sorted(
            {str(event["ir_node"]) for event in rows if event.get("ir_node")}
        )
        metrics[target] = {
            "ms_per_iter": round(active_us / 1000.0, 6),
            "active_gpu_ms": round(active_us / 1000.0, 6),
            "gpu_residency_ms": round(residency_us / 1000.0, 6),
            "gpu_residency_ms_per_iter": round(residency_us / 1000.0, 6),
            "attribution_status": "inclusive_rollup",
            "metric_kind": "inclusive_rollup",
            "timing_semantics": (
                "union of validated descendant production-kernel intervals; "
                "overlap counted once"
            ),
            "mapped_event_count": len(rows),
            "rollup_sources": direct_nodes,
        }
    return metrics


def direct_metrics_from_events(
    steps: Iterable[dict[str, Any]],
    *,
    direct_targets: set[str],
) -> dict[str, dict[str, Any]]:
    """Measure only events whose primary attribution is the target.

    ``ir_targets`` intentionally contains fusion members and drill ancestors
    for navigation and inclusive roll-ups.  It must never be used to overwrite
    an exclusive leaf measurement: otherwise every event owned by a shared
    fusion group is incorrectly charged to every directly measured member.
    """

    direct_steps = []
    for step in steps:
        events = []
        for event in step.get("events") or []:
            node = event.get("ir_node")
            if not node:
                continue
            events.append({**event, "ir_targets": [str(node)]})
        direct_steps.append({"events": events})
    metrics = rollup_metrics_from_events(
        direct_steps,
        rollup_targets=direct_targets,
    )
    for metric in metrics.values():
        metric["attribution_status"] = "measured_direct"
        metric["metric_kind"] = "exclusive_leaf"
        metric["timing_semantics"] = (
            "union of directly attributed production-kernel intervals; "
            "overlap counted once"
        )
    return metrics


def event_matches_scope(
    event: dict[str, Any],
    *,
    event_filter: dict[str, Any],
) -> bool:
    """Return whether one decoded event belongs to an authored timing scope.

    Scope fields are semantic execution coordinates carried by the production
    trace (for example ``substage=attention``), not kernel-name guesses.  A
    scalar value requires equality; a list means membership.  Missing event
    coordinates fail closed.
    """

    for field, expected in event_filter.items():
        actual = event.get(field)
        if actual is None:
            return False
        if isinstance(expected, list):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


def scoped_steps(
    steps: Iterable[dict[str, Any]],
    *,
    event_filter: dict[str, Any],
    source_nodes: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Select events by semantic occurrence scope and optional direct owners."""

    result: list[dict[str, Any]] = []
    for step in steps:
        events = []
        for event in step.get("events") or []:
            if not event_matches_scope(event, event_filter=event_filter):
                continue
            node = str(event.get("ir_node") or "")
            if source_nodes is not None and node not in source_nodes:
                continue
            events.append(event)
        result.append({"events": events})
    return result


def scoped_rollup_metric_from_events(
    steps: Iterable[dict[str, Any]],
    *,
    target: str,
    source_nodes: set[str],
    event_filter: dict[str, Any],
) -> dict[str, Any] | None:
    """Compute one parent metric from its context-filtered owner intervals.

    The result is an inclusive union.  It is intentionally not a copy of any
    child scalar, and residency remains the sum of the selected physical
    events so overlap is visible.
    """

    selected = scoped_steps(
        steps,
        event_filter=event_filter,
        source_nodes=source_nodes,
    )
    prepared = []
    for step in selected:
        prepared.append(
            {
                "events": [
                    {**event, "ir_targets": [target]}
                    for event in step.get("events") or []
                ]
            }
        )
    return rollup_metrics_from_events(
        prepared,
        rollup_targets={target},
    ).get(target)
