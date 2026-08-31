"""Model-independent acceptance checks for profile timing overlays."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


_INACTIVE_STATUSES = {
    "disabled",
    "not_selected",
    "not_in_selected_stage",
    "out_of_scope",
}
_EXPLICIT_BOUNDARY_KINDS = {
    "boundary",
    "control",
    "state",
    "control_boundary",
    "state_boundary",
}


def _node_semantic_kind(
    node: dict[str, Any], *, operations: dict[str, Any]
) -> str:
    return str(
        (operations.get(str(node.get("semantic_op") or "")) or {}).get("kind")
        or ""
    )


def _profile_cell(
    target: str,
    *,
    node: dict[str, Any],
    node_states: dict[str, dict[str, Any]],
    node_metrics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Return the raw profile cell plus explicit Model-IR fusion fallback."""

    cell: dict[str, Any] = {}
    cell.update(node_states.get(target) or {})
    cell.update(node_metrics.get(target) or {})
    if cell:
        return cell

    runtime = (node.get("semantic_details") or {}).get("runtime_mapping") or {}
    if runtime.get("expectation") in {"fused", "fused_state"}:
        return {
            "status": "fused",
            "included_in": runtime.get("owner"),
            "provenance": "model_ir.runtime_mapping",
        }
    return {}


def _has_positive_timing(cell: dict[str, Any]) -> bool:
    return any(
        float(cell.get(field) or 0.0) > 0.0
        for field in ("active_gpu_ms", "ms_per_iter")
    )


def _validate_reachable_drill_presentations(
    model_ir: dict[str, Any], profile: dict[str, Any]
) -> list[str]:
    """Validate every selected drillable compute module in its caller context.

    A shared child view can be reached from multiple parents.  Therefore this
    validator follows the rendered route and honors each parent's
    ``drill_metrics`` instead of trying to infer one global ancestor chain.
    This is the profile-side contract behind the Viewer rule:

    * compute modules with children are never presented as structural;
    * a selected compute module owns positive timing or is an explicit fused
      member with one timing owner;
    * only explicit boundary/control/state nodes and inactive branches are
      timing-free.
    """

    views = model_ir.get("views") or {}
    entry_view = str(
        profile.get("entry_view") or model_ir.get("default_view") or "top"
    )
    if entry_view not in views:
        return [f"entry_view {entry_view!r} is unknown"]

    node_states = profile.get("node_states") or {}
    node_metrics = profile.get("node_metrics") or {}
    operations = (
        (model_ir.get("semantic_contract") or {}).get("operations") or {}
    )
    errors: list[str] = []

    def walk(
        view_id: str,
        *,
        scoped_cells: dict[str, dict[str, Any]] | None,
        caller_isolated_owner: str,
        view_path: tuple[str, ...],
    ) -> None:
        if view_id in view_path:
            errors.append(
                "drill cycle in selected presentation: "
                + " -> ".join((*view_path, view_id))
            )
            return
        view = views.get(view_id) or {}
        for node in view.get("nodes") or []:
            child_view = node.get("drill")
            if not child_view:
                continue
            target = f"{view_id}.{node['id']}"
            if scoped_cells is not None and node["id"] in scoped_cells:
                cell = dict(scoped_cells[node["id"]] or {})
            elif caller_isolated_owner:
                semantic_kind = _node_semantic_kind(
                    node, operations=operations
                )
                if semantic_kind in _EXPLICIT_BOUNDARY_KINDS:
                    cell = {"status": "structural"}
                else:
                    cell = {
                        "status": "fused",
                        "included_in": caller_isolated_owner,
                        "provenance": "model_ir.runtime_scope",
                    }
            else:
                cell = _profile_cell(
                    target,
                    node=node,
                    node_states=node_states,
                    node_metrics=node_metrics,
                )

            status = str(cell.get("status") or "")
            if status in _INACTIVE_STATUSES:
                continue

            semantic_kind = _node_semantic_kind(node, operations=operations)
            explicit_boundary = semantic_kind in _EXPLICIT_BOUNDARY_KINDS
            if not explicit_boundary:
                if status == "structural":
                    errors.append(
                        f"{target} is a selected drillable compute module but "
                        "is presented as structural boundary"
                    )
                elif status == "fused":
                    owner = str(cell.get("included_in") or "")
                    if not owner:
                        errors.append(
                            f"{target} is fused but has no included_in timing owner"
                        )
                    # The raw Model IR is validated before an Execution IR is
                    # applied.  A valid owner may therefore be an
                    # execution-inserted communication node.  ``compile_profile``
                    # performs the authoritative existence/reachability check
                    # after topology projection; this stage only rejects an
                    # unowned fused presentation.
                elif not _has_positive_timing(cell):
                    errors.append(
                        f"{target} is a selected drillable compute module but "
                        "has neither positive timing nor fused ownership"
                    )

            next_scoped = cell.get("drill_metrics")
            if not isinstance(next_scoped, dict):
                next_scoped = None
            caller_isolated = (
                (node.get("semantic_details") or {}).get("runtime_scope")
                == "caller_isolated"
            )
            walk(
                str(child_view),
                scoped_cells=next_scoped,
                caller_isolated_owner=target if caller_isolated else "",
                view_path=(*view_path, view_id),
            )

    walk(
        entry_view,
        scoped_cells=None,
        caller_isolated_owner="",
        view_path=(),
    )
    return errors


def _selected_drill_ancestors(
    model_ir: dict[str, Any],
    *,
    node_states: dict[str, dict[str, Any]],
) -> dict[str, list[str]]:
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
            candidates = [
                parent
                for parent in parents_by_view.get(current_view, [])
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
                candidates = [
                    candidate
                    for candidate in parents_by_view.get(current_view, [])
                    if (node_states.get(candidate) or {}).get("status")
                    not in {"disabled", "not_selected", "out_of_scope"}
                ]
            result[target] = ancestors
    return result


def validate_executable_drill_rollups(
    model_ir: dict[str, Any],
    profile: dict[str, Any],
) -> None:
    """Reject executable drill nodes that hide measured descendant timing.

    This check never synthesizes timing.  A runtime-bearing drill/module is
    identified through the profile-selected ancestry of exclusively measured
    descendants and must carry a non-additive ``inclusive_rollup``.  Only an
    explicitly authored control/state boundary, or a branch without selected
    measured descendants, may remain timing-free.
    """

    node_states = profile.get("node_states") or {}
    node_metrics = profile.get("node_metrics") or {}
    ancestors = _selected_drill_ancestors(model_ir, node_states=node_states)
    direct_kinds = {"exclusive_leaf", "exclusive_leaf_scoped"}
    direct_statuses = {
        "measured",
        "measured_direct",
        "measured_scoped_direct",
    }
    direct_targets = {
        str(target)
        for target, metric in node_metrics.items()
        if metric.get("metric_kind") in direct_kinds
        or metric.get("attribution_status") in direct_statuses
        or (
            metric.get("metric_kind") != "inclusive_rollup"
            and metric.get("attribution_status") != "inclusive_rollup"
            and any(
                field in metric
                for field in ("ms_per_iter", "active_gpu_ms", "gpu_residency_ms")
            )
        )
    }

    operations = (
        (model_ir.get("semantic_contract") or {}).get("operations") or {}
    )
    errors: list[str] = []
    for view_id, view in (model_ir.get("views") or {}).items():
        for node in view.get("nodes") or []:
            if not node.get("drill"):
                continue
            target = f"{view_id}.{node['id']}"
            descendants = sorted(
                direct
                for direct in direct_targets
                if target in ancestors.get(direct, [])
            )
            if not descendants:
                continue

            semantic_op = str(node.get("semantic_op") or "")
            semantic_kind = str(
                (operations.get(semantic_op) or {}).get("kind") or ""
            )
            if semantic_kind in _EXPLICIT_BOUNDARY_KINDS:
                continue

            metric = node_metrics.get(target)
            if not metric:
                errors.append(
                    f"{target} has measured descendants {descendants} but no "
                    "inclusive_rollup metric"
                )
                continue
            if (
                metric.get("metric_kind") != "inclusive_rollup"
                or metric.get("attribution_status") != "inclusive_rollup"
            ):
                errors.append(
                    f"{target} has measured descendants {descendants} but "
                    f"metric_kind={metric.get('metric_kind')!r} and "
                    "attribution_status="
                    f"{metric.get('attribution_status')!r}"
                )
                continue
            active_ms = float(metric.get("active_gpu_ms") or 0.0)
            residency_ms = float(metric.get("gpu_residency_ms") or 0.0)
            if active_ms <= 0.0:
                errors.append(f"{target} inclusive_rollup has no active timing")
            # Both values are reconstructed from serialized microsecond
            # intervals and can differ by a few nanoseconds after repeated
            # float addition/rounding.  Ten nanoseconds is well below the
            # source trace resolution, while still rejecting any material
            # active-union/residency inconsistency.
            if residency_ms + 1e-5 < active_ms:
                errors.append(
                    f"{target} inclusive_rollup residency {residency_ms} ms "
                    f"is below active union {active_ms} ms"
                )

    errors.extend(_validate_reachable_drill_presentations(model_ir, profile))

    if errors:
        profile_id = profile.get("profile_id") or "<unknown profile>"
        raise ValueError(
            f"{profile_id}: executable drill rollup acceptance failed:\n- "
            + "\n- ".join(errors)
        )
