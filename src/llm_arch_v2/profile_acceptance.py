"""Model-independent acceptance checks for profile timing overlays."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


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
    explicit_boundary_kinds = {
        "boundary",
        "control",
        "state",
        "control_boundary",
        "state_boundary",
    }
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
            if semantic_kind in explicit_boundary_kinds:
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

    if errors:
        profile_id = profile.get("profile_id") or "<unknown profile>"
        raise ValueError(
            f"{profile_id}: executable drill rollup acceptance failed:\n- "
            + "\n- ".join(errors)
        )
