"""Hardware-aware Speed-of-Light profiles and measured gap reports.

The SoL layer is deliberately separate from Model IR, Execution IR, and
measured profiles.  It consumes those immutable contracts and emits a derived
theoretical overlay.  Unsupported operators fail closed instead of borrowing a
cost from an unrelated kernel family.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterator


class SolError(ValueError):
    """Raised when a SoL input or derived result is not trustworthy."""


_BINARY_OPERATORS = {
    ast.Add: lambda left, right: left + right,
    ast.Sub: lambda left, right: left - right,
    ast.Mult: lambda left, right: left * right,
    ast.Div: lambda left, right: left / right,
    ast.FloorDiv: lambda left, right: left // right,
}
_UNARY_OPERATORS = {
    ast.UAdd: lambda value: value,
    ast.USub: lambda value: -value,
}


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _portable_source_reference(path: Path) -> str:
    """Return checkout-independent provenance for a persisted catalog source."""

    parts = path.as_posix().split("/")
    if "catalog" in parts:
        return "/".join(parts[parts.index("catalog") :])
    return path.as_posix() if not path.is_absolute() else path.name


def _eval_expression(value: Any, variables: dict[str, float], *, field: str) -> float:
    if isinstance(value, bool):
        raise SolError(f"{field}: booleans are not numeric expressions")
    if isinstance(value, (int, float)):
        result = float(value)
    elif isinstance(value, str):
        try:
            expression = ast.parse(value, mode="eval")
        except SyntaxError as exc:
            raise SolError(f"{field}: invalid expression {value!r}") from exc

        def evaluate(node: ast.AST) -> float:
            if isinstance(node, ast.Expression):
                return evaluate(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return float(node.value)
            if isinstance(node, ast.Name):
                if node.id not in variables:
                    raise SolError(f"{field}: unknown variable {node.id!r}")
                return float(variables[node.id])
            if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
                return float(
                    _BINARY_OPERATORS[type(node.op)](
                        evaluate(node.left), evaluate(node.right)
                    )
                )
            if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
                return float(_UNARY_OPERATORS[type(node.op)](evaluate(node.operand)))
            raise SolError(
                f"{field}: only names, numbers, +, -, *, /, and // are allowed"
            )

        result = evaluate(expression)
    else:
        raise SolError(f"{field}: expected a number or expression")
    if not math.isfinite(result) or result < 0:
        raise SolError(f"{field}: expression must resolve to a finite non-negative value")
    return result


def _positive(value: Any, variables: dict[str, float], *, field: str) -> float:
    result = _eval_expression(value, variables, field=field)
    if result <= 0:
        raise SolError(f"{field}: value must be positive")
    return result


def _profile_cell(profile: dict[str, Any], target: str) -> dict[str, Any] | None:
    variants = (profile.get("data") or {}).get(target)
    if not isinstance(variants, dict) or not variants:
        return None
    cell = next((value for value in variants.values() if isinstance(value, dict)), None)
    return cell


def _all_targets(views: dict[str, Any]) -> set[str]:
    return {
        f"{view_id}.{node['id']}"
        for view_id, view in views.items()
        for node in view.get("nodes", []) or []
    }


def _base_variables(
    model_ir: dict[str, Any], profile: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, float]:
    variables: dict[str, float] = {}
    for key, value in (model_ir.get("dimensions") or {}).items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            variables[str(key)] = float(value)
    for key, value in (model_ir.get("facts") or {}).items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            variables[str(key)] = float(value)

    meta = profile.get("meta") or {}
    for key, value in (meta.get("execution_parameters") or {}).items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            variables[str(key)] = float(value)
    axis_aliases = {
        "TP": "tp_size",
        "DP": "dp_size",
        "CP": "cp_size",
        "EP": "ep_size",
    }
    for alias, source in axis_aliases.items():
        if source in variables:
            variables[alias] = variables[source]

    workload = meta.get("workload") or {}
    for key, value in workload.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            variables[str(key)] = float(value)
    if "batch_size" in variables:
        variables.setdefault("B", variables["batch_size"])

    unresolved = dict(manifest.get("variables") or {})
    while unresolved:
        progressed = False
        for key, value in list(unresolved.items()):
            try:
                variables[str(key)] = _eval_expression(
                    value, variables, field=f"variables.{key}"
                )
            except SolError as exc:
                if "unknown variable" in str(exc):
                    continue
                raise
            unresolved.pop(key)
            progressed = True
        if not progressed:
            raise SolError(
                "unresolved or cyclic variables: " + ", ".join(sorted(unresolved))
            )
    return variables


def _build_workload_ir(profile: dict[str, Any]) -> dict[str, Any]:
    """Freeze the realized workload used by one simulation.

    The simulator must not silently infer a generic batch or sequence length
    from the model config.  It binds to the same phase, graph mode, execution
    parameters, and workload metadata as the measured profile.
    """

    meta = profile.get("meta") or {}
    profiler = meta.get("profiler") or {}
    workload = {
        "schema_version": "workload-ir.v1",
        "profile_id": profile.get("profile_id"),
        "phase": meta.get("phase"),
        "generation_mode": meta.get("generation_mode"),
        "cuda_graph": profiler.get("cuda_graph_enabled", meta.get("cuda_graph")),
        "execution_parameters": copy.deepcopy(meta.get("execution_parameters") or {}),
        "workload": copy.deepcopy(meta.get("workload") or {}),
    }
    workload["fingerprint"] = _canonical_hash(workload)
    return workload


def validate_hardware_spec(hardware: dict[str, Any], *, source: Path) -> None:
    required = (
        "schema_version",
        "hardware_spec_id",
        "label",
        "architecture",
        "scope",
        "theoretical",
        "provenance",
    )
    missing = [key for key in required if key not in hardware]
    if missing:
        raise SolError(f"{source}: missing hardware fields: {', '.join(missing)}")
    if hardware["schema_version"] != "hardware-spec.v1":
        raise SolError(f"{source}: expected hardware-spec.v1")
    if hardware["scope"] != "per_gpu":
        raise SolError(f"{source}: only per_gpu hardware specs are supported")
    hbm = (((hardware.get("theoretical") or {}).get("memory") or {}).get(
        "hbm_bytes_per_s"
    ))
    if not isinstance(hbm, (int, float)) or hbm <= 0:
        raise SolError(f"{source}: theoretical.memory.hbm_bytes_per_s must be positive")
    methodologies = hardware.get("methodologies") or {}
    if not isinstance(methodologies, dict):
        raise SolError(f"{source}: methodologies must be a mapping")
    for methodology_id, methodology in methodologies.items():
        if not isinstance(methodology, dict):
            raise SolError(f"{source}: methodology {methodology_id!r} must be a mapping")
        if methodology.get("role") not in {"optimistic", "conservative", "provisional"}:
            raise SolError(f"{source}: methodology {methodology_id!r} has invalid role")
        if methodology.get("status") not in {"hypothesis", "correlated"}:
            raise SolError(f"{source}: methodology {methodology_id!r} has invalid status")
        if not isinstance(methodology.get("defaults"), dict):
            raise SolError(f"{source}: methodology {methodology_id!r} requires defaults")
        if not isinstance(methodology.get("provenance"), dict):
            raise SolError(f"{source}: methodology {methodology_id!r} requires provenance")


def validate_sol_manifest(manifest: dict[str, Any], *, source: Path) -> None:
    required = (
        "schema_version",
        "sol_profile_id",
        "gap_report_id",
        "model_id",
        "execution_path_id",
        "measured_profile_id",
        "hardware_spec_id",
        "phase",
        "assumptions",
        "nodes",
    )
    missing = [key for key in required if key not in manifest]
    if missing:
        raise SolError(f"{source}: missing SoL manifest fields: {', '.join(missing)}")
    if manifest["schema_version"] != "sol-manifest.v1":
        raise SolError(f"{source}: expected sol-manifest.v1")
    if not isinstance(manifest["nodes"], dict) or not manifest["nodes"]:
        raise SolError(f"{source}: nodes must be a non-empty mapping")
    assumptions = manifest["assumptions"]
    for key in ("semantic_math", "fusion_policy", "overlap_policy"):
        if not assumptions.get(key):
            raise SolError(f"{source}: assumptions.{key} is required")


def _match_calibration_value(actual: Any, expected: Any) -> bool:
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=1e-9)
    return actual == expected


def _calibrated_projection(
    spec: dict[str, Any],
    resolved: dict[str, Any],
    hardware: dict[str, Any],
    ideal_ms: float,
    kernel_plan: dict[str, Any] | None,
    kernel_plan_fingerprint: str | None,
) -> tuple[dict[str, float] | None, dict[str, Any] | None]:
    """Resolve a projection only from an exact, plan-identified surface.

    Shape-only efficiency coefficients are deliberately insufficient: two
    kernels with the same M/N/K can have different tile, staging, persistence,
    fusion, and cache-reuse behavior.  A projection therefore has to name the
    kernel plan it calibrated.  Missing plan identity fails closed.
    """

    surface_id = spec.get("calibration_surface")
    if not surface_id:
        return None, None
    surface = ((hardware.get("calibration") or {}).get("surfaces") or {}).get(
        surface_id
    )
    if not surface:
        return None, None
    expected_plan = surface.get("kernel_plan_fingerprint")
    if not kernel_plan or not kernel_plan_fingerprint:
        return None, {
            "surface_id": surface_id,
            "status": "ineligible_without_kernel_plan",
            "reason": "exact-shape projection requires a versioned kernel plan",
        }
    if not expected_plan:
        return None, {
            "surface_id": surface_id,
            "status": "ineligible_shape_only_surface",
            "reason": "surface does not identify the calibrated kernel plan",
        }
    if expected_plan != kernel_plan_fingerprint:
        return None, {
            "surface_id": surface_id,
            "status": "kernel_plan_mismatch",
            "expected_kernel_plan_fingerprint": expected_plan,
            "actual_kernel_plan_fingerprint": kernel_plan_fingerprint,
        }
    match_fields = list(surface.get("match_fields") or [])
    for point in surface.get("points", []) or []:
        match = point.get("match") or {}
        if not all(
            field in resolved
            and field in match
            and _match_calibration_value(resolved[field], match[field])
            for field in match_fields
        ):
            continue
        interval = point.get("attainable_interval_ms")
        if interval is not None:
            projection = {
                "p10_ms": float(interval["p10"]),
                "p50_ms": float(interval["p50"]),
                "p90_ms": float(interval["p90"]),
            }
        elif point.get("attainable_ms") is not None:
            attainable_ms = float(point["attainable_ms"])
            projection = {
                "p10_ms": attainable_ms,
                "p50_ms": attainable_ms,
                "p90_ms": attainable_ms,
            }
        elif point.get("efficiency") is not None:
            efficiency = float(point["efficiency"])
            if not 0 < efficiency <= 1:
                raise SolError(
                    f"calibration surface {surface_id!r} has invalid efficiency"
                )
            attainable_ms = ideal_ms / efficiency
            projection = {
                "p10_ms": attainable_ms,
                "p50_ms": attainable_ms,
                "p90_ms": attainable_ms,
            }
        else:
            raise SolError(
                f"calibration surface {surface_id!r} point requires attainable_ms "
                "attainable_interval_ms, or efficiency"
            )
        if not (
            0 < projection["p10_ms"]
            <= projection["p50_ms"]
            <= projection["p90_ms"]
        ):
            raise SolError(
                f"calibration surface {surface_id!r} has an invalid projection interval"
            )
        if projection["p10_ms"] + 1e-12 < ideal_ms:
            raise SolError(
                f"calibration surface {surface_id!r} is faster than the ideal bound"
            )
        return projection, {
            "surface_id": surface_id,
            "match": copy.deepcopy(match),
            "evidence": copy.deepcopy(surface.get("evidence") or {}),
            "kernel_plan_id": kernel_plan.get("plan_id"),
            "kernel_plan_fingerprint": kernel_plan_fingerprint,
            "interpolation": "exact_match_only",
            "status": "plan_exact_calibrated",
        }
    return None, None


def _transition_plan(
    *,
    target: str,
    kind: str,
    components_ms: dict[str, float],
    repetitions: float,
) -> dict[str, Any]:
    """Compile resource demands into a local transition DAG.

    Resources within one transition are concurrent and therefore use max().
    Explicit transitions are serial only through depends_on.  In particular,
    collective startup latency and wire transfer are separate transitions;
    they must never be collapsed with max().
    """

    transitions: list[dict[str, Any]] = []
    if kind == "collective":
        latency_ms = float(components_ms.get("latency", 0.0))
        if latency_ms > 0:
            transitions.append(
                {
                    "id": "startup",
                    "kind": "serial_latency",
                    "depends_on": [],
                    "resources_ms": {"collective_latency": latency_ms},
                    "duration_ms": latency_ms,
                    "limiter_vector": [
                        {"resource": "collective_latency", "time_ms": latency_ms}
                    ],
                }
            )
        transfer = {
            key: float(value)
            for key, value in components_ms.items()
            if key != "latency"
        }
        if transfer:
            limiter_vector = sorted(
                (
                    {"resource": resource, "time_ms": time_ms}
                    for resource, time_ms in transfer.items()
                ),
                key=lambda item: item["time_ms"],
                reverse=True,
            )
            transitions.append(
                {
                    "id": "wire_transfer",
                    "kind": "concurrent_resources",
                    "depends_on": ["startup"] if latency_ms > 0 else [],
                    "resources_ms": transfer,
                    "duration_ms": max(transfer.values()),
                    "limiter_vector": limiter_vector,
                }
            )
    else:
        limiter_vector = sorted(
            (
                {"resource": resource, "time_ms": float(time_ms)}
                for resource, time_ms in components_ms.items()
            ),
            key=lambda item: item["time_ms"],
            reverse=True,
        )
        transitions.append(
            {
                "id": "execute",
                "kind": "concurrent_resources",
                "depends_on": [],
                "resources_ms": copy.deepcopy(components_ms),
                "duration_ms": max(components_ms.values()),
                "limiter_vector": limiter_vector,
            }
        )

    ends: dict[str, float] = {}
    for transition in transitions:
        ends[transition["id"]] = max(
            (ends[dependency] for dependency in transition["depends_on"]),
            default=0.0,
        ) + float(transition["duration_ms"])
    ideal_ms = max(ends.values(), default=0.0)
    physical_plan = {
        "schema_version": "transition-plan.v1",
        "target": target,
        "semantics": "dependency_dag_with_concurrent_resource_bounds",
        "repetitions": repetitions,
        "transitions": transitions,
        "critical_path_ms": ideal_ms,
    }
    physical_plan["fingerprint"] = _canonical_hash(physical_plan)
    return physical_plan


def _kernel_plan_identity(spec: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    plan = spec.get("kernel_plan")
    if plan is None:
        return None, None
    if not isinstance(plan, dict):
        raise SolError("kernel_plan must be a mapping")
    for field in ("schema_version", "plan_id", "source", "algorithm"):
        if not plan.get(field):
            raise SolError(f"kernel_plan.{field} is required")
    if plan["schema_version"] != "kernel-plan.v1":
        raise SolError("kernel_plan.schema_version must be kernel-plan.v1")
    return copy.deepcopy(plan), _canonical_hash(plan)


def _methodology_estimates(
    *,
    target: str,
    spec: dict[str, Any],
    hardware: dict[str, Any],
    components_ms: dict[str, float],
    repetitions: float,
    operator_family: str,
) -> dict[str, dict[str, Any]]:
    """Apply an explicitly requested legacy sensitivity envelope.

    This is retained only for backwards-compatible what-if analysis.  It is
    not a projection model and is never enabled implicitly by the hardware
    spec.  New manifests should use a plan-identified calibration surface.
    """

    if not spec.get("legacy_sensitivity", False):
        return {}

    results: dict[str, dict[str, Any]] = {}
    methodologies = hardware.get("methodologies") or {}
    for methodology_id, methodology in methodologies.items():
        if not isinstance(methodology, dict) or not methodology.get("enabled", False):
            continue
        defaults = methodology.get("defaults") or {}
        family = (methodology.get("operator_families") or {}).get(
            operator_family, {}
        )
        if not isinstance(family, dict):
            raise SolError(
                f"{target}: methodology {methodology_id!r} family "
                f"{operator_family!r} must be a mapping"
            )
        parameters = {**defaults, **family}

        def efficiency(resource: str) -> float:
            value = parameters.get(
                f"{resource}_efficiency", parameters.get("resource_efficiency", 1.0)
            )
            value = float(value)
            if not 0 < value <= 1:
                raise SolError(
                    f"{target}: methodology {methodology_id!r} has invalid "
                    f"{resource} efficiency {value}"
                )
            return value

        adjusted: dict[str, float] = {}
        for resource, physical_ms in components_ms.items():
            resource_key = (
                "tensor_core"
                if resource == "tensor_core"
                else "interconnect"
                if resource in {"interconnect", "latency"}
                else "memory"
            )
            adjusted[resource] = float(physical_ms) / efficiency(resource_key)

        launch_us = float(parameters.get("launch_us", 0.0))
        sync_us = float(parameters.get("sync_us", 0.0))
        if launch_us < 0 or sync_us < 0:
            raise SolError(
                f"{target}: methodology {methodology_id!r} overheads must be non-negative"
            )
        fixed_overhead_ms = repetitions * (launch_us + sync_us) / 1000.0
        resource_ms = max(adjusted.values(), default=0.0)
        predicted_ms = fixed_overhead_ms + resource_ms
        limiter_vector = sorted(
            ({"resource": key, "time_ms": value} for key, value in adjusted.items()),
            key=lambda item: item["time_ms"],
            reverse=True,
        )
        if fixed_overhead_ms:
            limiter_vector.append(
                {"resource": "fixed_overhead", "time_ms": fixed_overhead_ms}
            )
            limiter_vector.sort(key=lambda item: item["time_ms"], reverse=True)
        results[str(methodology_id)] = {
            "methodology_id": str(methodology_id),
            "label": methodology.get("label") or str(methodology_id),
            "role": methodology.get("role") or "provisional",
            "status": methodology.get("status") or "hypothesis",
            "operator_family": operator_family,
            "predicted_ms": predicted_ms,
            "resource_critical_ms": resource_ms,
            "fixed_overhead_ms": fixed_overhead_ms,
            "components_ms": adjusted,
            "limiter_vector": limiter_vector,
            "parameters": copy.deepcopy(parameters),
            "provenance": copy.deepcopy(methodology.get("provenance") or {}),
        }
    return results


def _methodology_bounds(
    target: str,
    methodologies: dict[str, dict[str, Any]],
) -> tuple[float | None, float | None]:
    by_role: dict[str, list[float]] = {}
    for value in methodologies.values():
        if value.get("predicted_ms") is None:
            continue
        by_role.setdefault(str(value.get("role") or "provisional"), []).append(
            float(value["predicted_ms"])
        )
    optimistic_values = by_role.get("optimistic") or []
    conservative_values = by_role.get("conservative") or []
    optimistic = min(optimistic_values) if optimistic_values else None
    conservative = max(conservative_values) if conservative_values else None
    if optimistic is not None and conservative is not None and conservative < optimistic:
        raise SolError(
            f"{target}: conservative methodology is faster than optimistic methodology"
        )
    return optimistic, conservative


def _estimate_node(
    target: str,
    spec: dict[str, Any],
    hardware: dict[str, Any],
    variables: dict[str, float],
) -> dict[str, Any]:
    kind = spec.get("model")
    if kind == "structural":
        return {
            "target": target,
            "status": "structural",
            "reason": spec.get("reason") or "semantic boundary with no standalone cost",
            "included_in_target": spec.get("included_in_target"),
            "confidence": "not_applicable",
        }
    if kind == "unsupported":
        return {
            "target": target,
            "status": "unsupported",
            "reason": spec.get("reason") or "no validated cost adapter",
            "confidence": "none",
        }
    if kind not in {
        "gemm",
        "memory",
        "elementwise",
        "roofline",
        "collective",
        "attention",
    }:
        raise SolError(f"{target}: unsupported cost model {kind!r}")

    theoretical = hardware["theoretical"]
    hbm_bandwidth = float(theoretical["memory"]["hbm_bytes_per_s"])
    components_ms: dict[str, float] = {}
    resolved: dict[str, float] = {}
    useful_ops = 0.0
    compulsory_bytes = 0.0

    if kind == "gemm":
        for field in ("m", "n", "k"):
            resolved[field] = _positive(
                spec.get(field), variables, field=f"{target}.{field}"
            )
        dtype = str(spec.get("dtype") or "")
        dtype_bytes = _positive(
            spec.get("dtype_bytes"), variables, field=f"{target}.dtype_bytes"
        )
        output_bytes = _positive(
            spec.get("output_dtype_bytes", dtype_bytes),
            variables,
            field=f"{target}.output_dtype_bytes",
        )
        peak_key = f"{dtype}_dense_flops_per_s"
        peak = (
            ((theoretical.get("compute") or {}).get("tensor_core") or {}).get(
                peak_key
            )
        )
        if not isinstance(peak, (int, float)) or peak <= 0:
            raise SolError(f"{target}: missing theoretical Tensor Core peak {peak_key}")
        m, n, k = resolved["m"], resolved["n"], resolved["k"]
        useful_ops = 2.0 * m * n * k
        compulsory_bytes = (
            m * k * dtype_bytes + k * n * dtype_bytes + m * n * output_bytes
        )
        components_ms["tensor_core"] = useful_ops / float(peak) * 1000.0
        components_ms["hbm"] = compulsory_bytes / hbm_bandwidth * 1000.0
    elif kind in {"memory", "elementwise"}:
        compulsory_bytes = _positive(
            spec.get("bytes"), variables, field=f"{target}.bytes"
        )
        components_ms["hbm"] = compulsory_bytes / hbm_bandwidth * 1000.0
        if spec.get("ops") is not None:
            useful_ops = _eval_expression(
                spec["ops"], variables, field=f"{target}.ops"
            )
    elif kind == "roofline":
        compulsory_bytes = _eval_expression(
            spec.get("bytes", 0), variables, field=f"{target}.bytes"
        )
        useful_ops = _eval_expression(
            spec.get("ops", 0), variables, field=f"{target}.ops"
        )
        if compulsory_bytes <= 0 and useful_ops <= 0:
            raise SolError(f"{target}: roofline requires positive bytes or ops")
        if compulsory_bytes > 0:
            components_ms["hbm"] = compulsory_bytes / hbm_bandwidth * 1000.0
        if useful_ops > 0:
            dtype = str(spec.get("dtype") or "bf16")
            peak_key = f"{dtype}_dense_flops_per_s"
            peak = (
                ((theoretical.get("compute") or {}).get("tensor_core") or {}).get(
                    peak_key
                )
            )
            if not isinstance(peak, (int, float)) or peak <= 0:
                raise SolError(
                    f"{target}: missing theoretical Tensor Core peak {peak_key}"
                )
            components_ms["tensor_core"] = useful_ops / float(peak) * 1000.0
    elif kind == "attention":
        for field in ("batch", "query_tokens", "kv_tokens", "heads", "head_dim"):
            resolved[field] = _positive(
                spec.get(field), variables, field=f"{target}.{field}"
            )
        dtype = str(spec.get("dtype") or "")
        dtype_bytes = _positive(
            spec.get("dtype_bytes"), variables, field=f"{target}.dtype_bytes"
        )
        peak_key = f"{dtype}_dense_flops_per_s"
        peak = (
            ((theoretical.get("compute") or {}).get("tensor_core") or {}).get(
                peak_key
            )
        )
        if not isinstance(peak, (int, float)) or peak <= 0:
            raise SolError(f"{target}: missing theoretical Tensor Core peak {peak_key}")
        b = resolved["batch"]
        q = resolved["query_tokens"]
        kv = resolved["kv_tokens"]
        heads = resolved["heads"]
        dim = resolved["head_dim"]
        # QK^T and PV matrix products. Softmax/SFU work is deliberately not
        # hidden in this adapter; callers must add a calibrated surface before
        # treating this as attainable performance.
        useful_ops = 4.0 * b * heads * q * kv * dim
        compulsory_bytes = (
            b * heads * q * dim
            + 2.0 * b * heads * kv * dim
            + b * heads * q * dim
        ) * dtype_bytes
        components_ms["tensor_core"] = useful_ops / float(peak) * 1000.0
        components_ms["hbm"] = compulsory_bytes / hbm_bandwidth * 1000.0
    else:
        collective = str(spec.get("collective") or "")
        group_size = int(
            _positive(spec.get("group_size"), variables, field=f"{target}.group_size")
        )
        if group_size < 2:
            raise SolError(f"{target}: collective group_size must be at least 2")
        payload = _positive(
            spec.get("payload_bytes"), variables, field=f"{target}.payload_bytes"
        )
        fabric_id = str(spec.get("fabric") or "nvlink")
        fabric = ((theoretical.get("interconnect") or {}).get(fabric_id) or {})
        bandwidth = fabric.get("bandwidth_bytes_per_s")
        if not isinstance(bandwidth, (int, float)) or bandwidth <= 0:
            raise SolError(f"{target}: missing interconnect bandwidth for {fabric_id}")
        if collective == "all_reduce":
            wire_bytes = 2.0 * (group_size - 1) / group_size * payload
        elif collective == "all_gather":
            wire_bytes = (group_size - 1) * payload
        elif collective == "reduce_scatter":
            wire_bytes = (group_size - 1) / group_size * payload
        elif collective == "all_to_all":
            wire_bytes = (group_size - 1) / group_size * payload
        else:
            raise SolError(f"{target}: unsupported collective {collective!r}")
        resolved.update(
            {"group_size": float(group_size), "payload_bytes": payload, "wire_bytes": wire_bytes}
        )
        compulsory_bytes = payload
        components_ms["interconnect"] = wire_bytes / float(bandwidth) * 1000.0
        latency_us = fabric.get("latency_floor_us")
        if isinstance(latency_us, (int, float)) and latency_us >= 0:
            components_ms["latency"] = float(latency_us) / 1000.0

    repetitions = _positive(
        spec.get("repetitions", 1), variables, field=f"{target}.repetitions"
    )
    if not math.isclose(repetitions, round(repetitions), rel_tol=0, abs_tol=1e-9):
        raise SolError(f"{target}.repetitions: value must be an integer")
    repetitions = float(round(repetitions))
    if repetitions != 1:
        components_ms = {
            component: value * repetitions
            for component, value in components_ms.items()
        }
        useful_ops *= repetitions
        compulsory_bytes *= repetitions
    resolved["repetitions"] = repetitions

    operator_family = str(spec.get("operator_family") or kind)
    physical_plan = _transition_plan(
        target=target,
        kind=kind,
        components_ms=components_ms,
        repetitions=repetitions,
    )
    ideal_ms = float(physical_plan["critical_path_ms"])
    transition_limiters = [
        (transition, transition.get("limiter_vector") or [])
        for transition in physical_plan["transitions"]
    ]
    critical_transition, critical_limiters = max(
        transition_limiters,
        key=lambda item: float(item[0].get("duration_ms") or 0.0),
    )
    limiting_resource = (
        critical_limiters[0]["resource"] if critical_limiters else "unresolved"
    )
    kernel_plan, kernel_plan_fingerprint = _kernel_plan_identity(spec)
    projection, calibration = _calibrated_projection(
        spec,
        resolved,
        hardware,
        ideal_ms,
        kernel_plan,
        kernel_plan_fingerprint,
    )
    attainable_ms = projection["p50_ms"] if projection is not None else None
    methodologies = _methodology_estimates(
        target=target,
        spec=spec,
        hardware=hardware,
        components_ms=components_ms,
        repetitions=repetitions,
        operator_family=operator_family,
    )
    optimistic_ms, conservative_ms = _methodology_bounds(target, methodologies)
    cost_ir = {
        "schema_version": "cost-ir.v1",
        "target": target,
        "physical_model": kind,
        "operator_family": operator_family,
        "problem": copy.deepcopy(resolved),
        "resources": {
            "useful_ops": useful_ops,
            "compulsory_hbm_bytes": compulsory_bytes,
            "physical_components_ms": copy.deepcopy(components_ms),
            "transition_plan": copy.deepcopy(physical_plan),
        },
        "repetitions": repetitions,
        "assumptions": copy.deepcopy(spec.get("assumptions") or []),
    }
    return {
        "target": target,
        "status": "estimated",
        "model": kind,
        "operator_family": operator_family,
        "cost_ir": cost_ir,
        "ideal_ms": ideal_ms,
        "attainable_ms": attainable_ms,
        "attainable_interval_ms": projection,
        "methodology_optimistic_ms": optimistic_ms,
        "methodology_conservative_ms": conservative_ms,
        "methodologies": methodologies,
        "limiting_resource": limiting_resource,
        "critical_transition": critical_transition["id"],
        "limiter_vector": copy.deepcopy(critical_limiters),
        "components_ms": components_ms,
        "useful_ops": useful_ops,
        "compulsory_bytes": compulsory_bytes,
        "resolved": resolved,
        "physical_plan": physical_plan,
        "kernel_plan": kernel_plan,
        "kernel_plan_fingerprint": kernel_plan_fingerprint,
        "calibration": calibration,
        "confidence": (
            "plan_exact_calibrated"
            if attainable_ms is not None
            else "legacy_sensitivity_only"
            if methodologies
            else "ideal_bound_only"
        ),
        "assumptions": copy.deepcopy(spec.get("assumptions") or []),
    }


def _estimate_aggregate(
    target: str,
    spec: dict[str, Any],
    estimates: dict[str, dict[str, Any]],
    variables: dict[str, float],
) -> dict[str, Any]:
    members = list(spec.get("members") or [])
    if not members:
        raise SolError(f"{target}: aggregate requires non-empty members")
    unknown = sorted(set(members) - set(estimates))
    if unknown:
        raise SolError(f"{target}: unresolved aggregate members {unknown}")
    unsupported = [
        member for member in members if estimates[member]["status"] == "unsupported"
    ]
    if unsupported:
        return {
            "target": target,
            "status": "unsupported",
            "model": "aggregate",
            "reason": "aggregate contains unsupported members: " + ", ".join(unsupported),
            "members": members,
            "confidence": "none",
        }
    costed = [member for member in members if estimates[member].get("ideal_ms") is not None]
    scale = _positive(spec.get("scale", 1), variables, field=f"{target}.scale")
    ideal_ms = scale * sum(float(estimates[member]["ideal_ms"]) for member in costed)
    attainable_values = [estimates[member].get("attainable_ms") for member in costed]
    attainable_ms = (
        scale * sum(float(value) for value in attainable_values)
        if costed and all(value is not None for value in attainable_values)
        else None
    )
    interval_values = [
        estimates[member].get("attainable_interval_ms") for member in costed
    ]
    attainable_interval = (
        {
            "p10_ms": scale * sum(float(value["p10_ms"]) for value in interval_values),
            "p50_ms": scale * sum(float(value["p50_ms"]) for value in interval_values),
            "p90_ms": scale * sum(float(value["p90_ms"]) for value in interval_values),
        }
        if costed and all(value is not None for value in interval_values)
        else None
    )
    optimistic_values = [
        estimates[member].get("methodology_optimistic_ms") for member in costed
    ]
    conservative_values = [
        estimates[member].get("methodology_conservative_ms") for member in costed
    ]
    optimistic_ms = (
        scale * sum(float(value) for value in optimistic_values)
        if costed and all(value is not None for value in optimistic_values)
        else None
    )
    conservative_ms = (
        scale * sum(float(value) for value in conservative_values)
        if costed and all(value is not None for value in conservative_values)
        else None
    )
    cost_ir = {
        "schema_version": "cost-ir.v1",
        "target": target,
        "physical_model": "aggregate",
        "operator_family": str(spec.get("operator_family") or "aggregate"),
        "members": members,
        "scale": scale,
        "assumptions": copy.deepcopy(spec.get("assumptions") or []),
    }
    transitions: list[dict[str, Any]] = []
    previous: str | None = None
    for index, member in enumerate(costed):
        transition_id = f"member_{index}"
        duration_ms = scale * float(estimates[member]["ideal_ms"])
        transitions.append(
            {
                "id": transition_id,
                "kind": "serial_member",
                "member": member,
                "depends_on": [previous] if previous else [],
                "resources_ms": {"member_lower_bound": duration_ms},
                "duration_ms": duration_ms,
                "limiter_vector": [
                    {"resource": "member_lower_bound", "time_ms": duration_ms}
                ],
            }
        )
        previous = transition_id
    physical_plan = {
        "schema_version": "transition-plan.v1",
        "target": target,
        "semantics": "declared_serial_semantic_sum",
        "repetitions": 1,
        "transitions": transitions,
        "critical_path_ms": ideal_ms,
    }
    physical_plan["fingerprint"] = _canonical_hash(physical_plan)
    cost_ir["resources"] = {"transition_plan": copy.deepcopy(physical_plan)}
    return {
        "target": target,
        "status": "estimated",
        "model": "aggregate",
        "operator_family": cost_ir["operator_family"],
        "cost_ir": cost_ir,
        "ideal_ms": ideal_ms,
        "attainable_ms": attainable_ms,
        "attainable_interval_ms": attainable_interval,
        "methodology_optimistic_ms": optimistic_ms,
        "methodology_conservative_ms": conservative_ms,
        "methodologies": {},
        "limiting_resource": "serial_semantic_sum",
        "critical_transition": previous,
        "limiter_vector": [],
        "components_ms": {
            member: scale * float(estimates[member]["ideal_ms"]) for member in costed
        },
        "useful_ops": sum(
            float(estimates[member].get("useful_ops") or 0) for member in costed
        ) * scale,
        "compulsory_bytes": sum(
            float(estimates[member].get("compulsory_bytes") or 0) for member in costed
        ) * scale,
        "resolved": {"member_count": len(members), "scale": scale},
        "physical_plan": physical_plan,
        "kernel_plan": None,
        "kernel_plan_fingerprint": None,
        "members": members,
        "calibration": None,
        "confidence": (
            "plan_exact_calibrated"
            if attainable_ms is not None
            else "legacy_sensitivity_only"
            if optimistic_ms is not None and conservative_ms is not None
            else "ideal_bound_only"
        ),
        "assumptions": copy.deepcopy(spec.get("assumptions") or []),
    }


def _critical_path(
    estimates: dict[str, dict[str, Any]], manifest: dict[str, Any]
) -> dict[str, Any]:
    schedule = manifest.get("schedule") or {}
    dependencies = {
        target: list((spec or {}).get("depends_on") or [])
        for target, spec in (manifest.get("nodes") or {}).items()
    }
    for target, deps in dependencies.items():
        unknown = sorted(set(deps) - set(dependencies))
        if unknown:
            raise SolError(f"{target}: unknown schedule dependencies {unknown}")

    visiting: set[str] = set()
    ideal_ends: dict[str, float] = {}
    attainable_ends: dict[str, float | None] = {}
    attainable_p10_ends: dict[str, float | None] = {}
    attainable_p90_ends: dict[str, float | None] = {}
    optimistic_ends: dict[str, float | None] = {}
    conservative_ends: dict[str, float | None] = {}

    def visit(
        target: str,
    ) -> tuple[
        float,
        float | None,
        float | None,
        float | None,
        float | None,
        float | None,
    ]:
        if target in ideal_ends:
            return (
                ideal_ends[target],
                attainable_ends[target],
                attainable_p10_ends[target],
                attainable_p90_ends[target],
                optimistic_ends[target],
                conservative_ends[target],
            )
        if target in visiting:
            raise SolError(f"schedule dependency cycle at {target!r}")
        visiting.add(target)
        dep_values = [visit(dep) for dep in dependencies[target]]
        estimate = estimates[target]
        ideal = estimate.get("ideal_ms")
        if ideal is None:
            ideal_end = max((value[0] for value in dep_values), default=0.0)
            if estimate.get("status") == "structural":
                attainable_end = (
                    max((float(value[1]) for value in dep_values), default=0.0)
                    if all(value[1] is not None for value in dep_values)
                    else None
                )
                attainable_p10_end = (
                    max((float(value[2]) for value in dep_values), default=0.0)
                    if all(value[2] is not None for value in dep_values)
                    else None
                )
                attainable_p90_end = (
                    max((float(value[3]) for value in dep_values), default=0.0)
                    if all(value[3] is not None for value in dep_values)
                    else None
                )
                optimistic_end = (
                    max((float(value[4]) for value in dep_values), default=0.0)
                    if all(value[4] is not None for value in dep_values)
                    else None
                )
                conservative_end = (
                    max((float(value[5]) for value in dep_values), default=0.0)
                    if all(value[5] is not None for value in dep_values)
                    else None
                )
            else:
                attainable_end = None
                attainable_p10_end = None
                attainable_p90_end = None
                optimistic_end = None
                conservative_end = None
        else:
            ideal_end = max((value[0] for value in dep_values), default=0.0) + float(ideal)
            attainable = estimate.get("attainable_ms")
            if attainable is None or any(value[1] is None for value in dep_values):
                attainable_end = None
            else:
                attainable_end = max(
                    (float(value[1]) for value in dep_values), default=0.0
                ) + float(attainable)
            interval = estimate.get("attainable_interval_ms")
            if interval is None or any(value[2] is None for value in dep_values):
                attainable_p10_end = None
            else:
                attainable_p10_end = max(
                    (float(value[2]) for value in dep_values), default=0.0
                ) + float(interval["p10_ms"])
            if interval is None or any(value[3] is None for value in dep_values):
                attainable_p90_end = None
            else:
                attainable_p90_end = max(
                    (float(value[3]) for value in dep_values), default=0.0
                ) + float(interval["p90_ms"])
            optimistic = estimate.get("methodology_optimistic_ms")
            if optimistic is None or any(value[4] is None for value in dep_values):
                optimistic_end = None
            else:
                optimistic_end = max(
                    (float(value[4]) for value in dep_values), default=0.0
                ) + float(optimistic)
            conservative = estimate.get("methodology_conservative_ms")
            if conservative is None or any(value[5] is None for value in dep_values):
                conservative_end = None
            else:
                conservative_end = max(
                    (float(value[5]) for value in dep_values), default=0.0
                ) + float(conservative)
        visiting.remove(target)
        ideal_ends[target] = ideal_end
        attainable_ends[target] = attainable_end
        attainable_p10_ends[target] = attainable_p10_end
        attainable_p90_ends[target] = attainable_p90_end
        optimistic_ends[target] = optimistic_end
        conservative_ends[target] = conservative_end
        return (
            ideal_end,
            attainable_end,
            attainable_p10_end,
            attainable_p90_end,
            optimistic_end,
            conservative_end,
        )

    for target in dependencies:
        visit(target)
    ideal_critical = max(ideal_ends.values(), default=0.0)
    attainable_values = [value for value in attainable_ends.values() if value is not None]
    all_attainable = bool(attainable_ends) and len(attainable_values) == len(attainable_ends)
    attainable_p10_values = [
        value for value in attainable_p10_ends.values() if value is not None
    ]
    attainable_p90_values = [
        value for value in attainable_p90_ends.values() if value is not None
    ]
    optimistic_values = [value for value in optimistic_ends.values() if value is not None]
    conservative_values = [
        value for value in conservative_ends.values() if value is not None
    ]
    all_methodology = bool(optimistic_ends) and (
        len(optimistic_values) == len(optimistic_ends)
        and len(conservative_values) == len(conservative_ends)
    )
    return {
        "semantics": "dependency_only_unlimited_resources",
        "complete_step": bool(schedule.get("complete_step", False)),
        "ideal_critical_path_ms": ideal_critical,
        "attainable_critical_path_ms": max(attainable_values) if all_attainable else None,
        "attainable_critical_path_interval_ms": (
            {
                "p10_ms": max(attainable_p10_values),
                "p50_ms": max(attainable_values),
                "p90_ms": max(attainable_p90_values),
            }
            if all_attainable
            and len(attainable_p10_values) == len(attainable_p10_ends)
            and len(attainable_p90_values) == len(attainable_p90_ends)
            else None
        ),
        "methodology_optimistic_critical_path_ms": (
            max(optimistic_values) if all_methodology else None
        ),
        "methodology_conservative_critical_path_ms": (
            max(conservative_values) if all_methodology else None
        ),
        "note": schedule.get("note")
        or "Not a full-step SoL unless complete_step is true and coverage passes.",
    }


def build_sol_artifacts(
    *,
    model_ir: dict[str, Any],
    execution_variants: dict[str, Any],
    profiles: dict[str, Any],
    hardware: dict[str, Any],
    manifest: dict[str, Any],
    manifest_source: Path,
    hardware_source: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build one immutable SoL profile and its measured gap report."""

    validate_hardware_spec(hardware, source=hardware_source)
    validate_sol_manifest(manifest, source=manifest_source)
    if manifest["model_id"] != model_ir["model_id"]:
        raise SolError(f"{manifest_source}: model_id does not match Model IR")
    if manifest["hardware_spec_id"] != hardware["hardware_spec_id"]:
        raise SolError(f"{manifest_source}: hardware_spec_id does not match hardware file")

    measured_id = manifest["measured_profile_id"]
    measured = profiles.get(measured_id)
    if not measured:
        raise SolError(f"{manifest_source}: unknown measured profile {measured_id!r}")
    if measured["execution_path_id"] != manifest["execution_path_id"]:
        raise SolError(f"{manifest_source}: execution_path_id does not match profile")
    if (measured.get("meta") or {}).get("phase") != manifest["phase"]:
        raise SolError(f"{manifest_source}: phase does not match measured profile")

    hardware_name = str(((measured.get("meta") or {}).get("hardware") or {}).get("gpu") or "")
    aliases = {str(value) for value in hardware.get("aliases", []) or []}
    aliases.add(str(hardware["hardware_spec_id"]))
    if hardware_name and hardware_name not in aliases:
        raise SolError(
            f"{manifest_source}: measured hardware {hardware_name!r} is not an alias "
            f"of {hardware['hardware_spec_id']!r}"
        )

    fingerprint = measured["execution_variant"]
    variant = execution_variants[fingerprint]
    valid_targets = _all_targets(variant["views"])
    unknown_targets = sorted(set(manifest["nodes"]) - valid_targets)
    if unknown_targets:
        raise SolError(f"{manifest_source}: unknown IR targets {unknown_targets}")

    variables = _base_variables(model_ir, measured, manifest)
    estimates: dict[str, dict[str, Any]] = {}
    gap_nodes: dict[str, dict[str, Any]] = {}
    tolerance = float(manifest.get("correctness_tolerance_pct", 2.0))
    unresolved = dict(manifest["nodes"])
    while unresolved:
        progressed = False
        for target, node_spec in list(unresolved.items()):
            if not isinstance(node_spec, dict):
                raise SolError(f"{manifest_source}: node {target!r} must be a mapping")
            if node_spec.get("model") == "aggregate":
                members = list(node_spec.get("members") or [])
                invalid = sorted(set(members) - set(manifest["nodes"]))
                if invalid:
                    raise SolError(f"{target}: unknown aggregate members {invalid}")
                if any(member not in estimates for member in members):
                    continue
                estimate = _estimate_aggregate(target, node_spec, estimates, variables)
            else:
                estimate = _estimate_node(target, node_spec, hardware, variables)
            estimates[target] = estimate
            unresolved.pop(target)
            progressed = True
        if not progressed:
            raise SolError(
                "cyclic aggregate members: " + ", ".join(sorted(unresolved))
            )

    for target, node_spec in manifest["nodes"].items():
        if not isinstance(node_spec, dict):
            raise SolError(f"{manifest_source}: node {target!r} must be a mapping")
        estimate = estimates[target]
        cell = _profile_cell(measured, target)
        observed = None
        if cell:
            observed = cell.get("active_gpu_ms", cell.get("ms_per_iter"))
            if observed is not None:
                observed = float(observed)
        estimate["observed_active_ms"] = observed
        ideal = estimate.get("ideal_ms")
        attainable = estimate.get("attainable_ms")
        attainable_interval = estimate.get("attainable_interval_ms")
        methodology_optimistic = estimate.get("methodology_optimistic_ms")
        methodology_conservative = estimate.get("methodology_conservative_ms")
        violation = bool(
            observed is not None
            and ideal is not None
            and observed < float(ideal) * (1.0 - tolerance / 100.0)
        )
        ideal_efficiency = (
            float(ideal) / observed * 100.0
            if observed and ideal is not None
            else None
        )
        implementation_gap = (
            max(0.0, observed - float(attainable))
            if observed is not None and attainable is not None
            else None
        )
        attainable_coverage = (
            float(attainable) / observed * 100.0
            if observed and attainable is not None
            else None
        )
        methodology_optimistic_coverage = (
            float(methodology_optimistic) / observed * 100.0
            if observed and methodology_optimistic is not None
            else None
        )
        methodology_conservative_coverage = (
            float(methodology_conservative) / observed * 100.0
            if observed and methodology_conservative is not None
            else None
        )
        methodology_violation = bool(
            observed is not None
            and methodology_optimistic is not None
            and observed
            < float(methodology_optimistic) * (1.0 - tolerance / 100.0)
        )
        projection_violation = bool(
            observed is not None
            and attainable_interval is not None
            and observed
            < float(attainable_interval["p10_ms"]) * (1.0 - tolerance / 100.0)
        )
        gap_nodes[target] = {
            "status": (
                "model_violation"
                if violation
                else "projection_violation"
                if projection_violation
                else "methodology_violation"
                if methodology_violation
                else estimate["status"]
            ),
            "observed_active_ms": observed,
            "ideal_ms": ideal,
            "attainable_ms": attainable,
            "attainable_interval_ms": copy.deepcopy(attainable_interval),
            "methodology_optimistic_ms": methodology_optimistic,
            "methodology_conservative_ms": methodology_conservative,
            "physical_coverage_pct": ideal_efficiency,
            # Compatibility alias for bundles produced before the viewer
            # renamed this lower-bound ratio from "efficiency" to coverage.
            "ideal_efficiency_pct": ideal_efficiency,
            "attainable_coverage_pct": attainable_coverage,
            "methodology_optimistic_coverage_pct": methodology_optimistic_coverage,
            "methodology_conservative_coverage_pct": methodology_conservative_coverage,
            "implementation_gap_ms": implementation_gap,
            "methodology_gap_range_ms": (
                {
                    "lower": max(0.0, observed - float(methodology_conservative)),
                    "upper": max(0.0, observed - float(methodology_optimistic)),
                }
                if observed is not None
                and methodology_optimistic is not None
                and methodology_conservative is not None
                else None
            ),
            "unallocated_gap_above_ideal_ms": (
                max(0.0, observed - float(ideal))
                if observed is not None and ideal is not None
                else None
            ),
            "limiting_resource": estimate.get("limiting_resource"),
            "diagnosis": (
                "fix_sol_model_before_optimization"
                if violation
                else "recalibrate_plan_projection"
                if projection_violation
                else "recalibrate_methodology_envelope"
                if methodology_violation
                else "structural_boundary"
                if estimate["status"] == "structural"
                else "requires_calibration_before_framework_blame"
                if ideal is not None
                and attainable is None
                and methodology_optimistic is None
                else "implementation_headroom"
                if implementation_gap is not None and implementation_gap > 0
                else "legacy_sensitivity_available"
                if methodology_optimistic is not None
                else "unsupported"
            ),
        }

    critical_path = _critical_path(estimates, manifest)
    unsupported = sorted(
        target for target, value in estimates.items() if value["status"] == "unsupported"
    )
    estimated = [
        target for target, value in estimates.items() if value["status"] == "estimated"
    ]
    calibrated = [
        target for target, value in estimates.items() if value.get("attainable_ms") is not None
    ]
    observed = [
        target for target, value in estimates.items() if value.get("observed_active_ms") is not None
    ]
    violations = [
        target for target, value in gap_nodes.items() if value["status"] == "model_violation"
    ]
    projection_violations = [
        target
        for target, value in gap_nodes.items()
        if value["status"] == "projection_violation"
    ]
    calibration_complete = len(calibrated) == len(estimated) and not unsupported
    structural = [
        target for target, value in estimates.items() if value["status"] == "structural"
    ]
    coverage = {
        "declared_node_count": len(estimates),
        "ideal_estimated_node_count": len(estimated),
        "calibrated_node_count": len(calibrated),
        "plan_identified_node_count": sum(
            value.get("kernel_plan_fingerprint") is not None
            for value in estimates.values()
        ),
        "transition_simulated_node_count": sum(
            value.get("physical_plan") is not None for value in estimates.values()
        ),
        "legacy_sensitivity_node_count": sum(
            value.get("methodology_optimistic_ms") is not None
            and value.get("methodology_conservative_ms") is not None
            for value in estimates.values()
        ),
        "observed_comparison_node_count": len(observed),
        "structural_node_count": len(structural),
        "unsupported_targets": unsupported,
        "coverage_semantics": "declared adapter nodes; not additive timing coverage",
    }
    assumptions_hash = _canonical_hash(manifest["assumptions"])
    workload_ir = _build_workload_ir(measured)
    identity = {
        "model_id": model_ir["model_id"],
        "execution_fingerprint": fingerprint,
        "execution_path_id": manifest["execution_path_id"],
        "measured_profile_id": measured_id,
        "hardware_spec_id": hardware["hardware_spec_id"],
        "phase": manifest["phase"],
        "workload_fingerprint": workload_ir["fingerprint"],
        "assumptions_sha256": assumptions_hash,
    }
    provenance = {
        "manifest": _portable_source_reference(manifest_source),
        "manifest_sha256": _canonical_hash(manifest),
        "hardware_spec": _portable_source_reference(hardware_source),
        "hardware_spec_sha256": _canonical_hash(hardware),
        "measured_profile_id": measured_id,
    }
    sol_profile = {
        "schema_version": "sol-profile.v1",
        "sol_profile_id": manifest["sol_profile_id"],
        "label": manifest.get("label") or manifest["sol_profile_id"],
        **identity,
        "status": (
            "invalid"
            if violations or projection_violations
            else "calibrated"
            if calibration_complete
            else "partial"
        ),
        "variables": variables,
        "assumptions": copy.deepcopy(manifest["assumptions"]),
        "fused_owner_aliases": copy.deepcopy(
            manifest.get("fused_owner_aliases") or {}
        ),
        "workload_ir": workload_ir,
        "coverage": coverage,
        "critical_path": critical_path,
        "cost_ir": {
            target: copy.deepcopy(value["cost_ir"])
            for target, value in estimates.items()
            if value.get("cost_ir") is not None
        },
        "node_estimates": estimates,
        "provenance": provenance,
    }
    gap_report = {
        "schema_version": "gap-report.v1",
        "gap_report_id": manifest["gap_report_id"],
        "label": manifest.get("gap_label") or manifest["gap_report_id"],
        "sol_profile_id": manifest["sol_profile_id"],
        **identity,
        "status": (
            "invalid_sol_model"
            if violations
            else "invalid_projection_model"
            if projection_violations
            else "ready"
            if calibration_complete
            else "partial_calibration"
        ),
        "correctness_tolerance_pct": tolerance,
        "model_violations": violations,
        "projection_violations": projection_violations,
        "coverage": copy.deepcopy(coverage),
        "nodes": gap_nodes,
        "provenance": provenance,
    }
    return sol_profile, gap_report


def attach_sol_to_profile(
    profile: dict[str, Any], sol_profile: dict[str, Any], gap_report: dict[str, Any]
) -> None:
    """Attach a namespaced viewer projection without changing measured values."""

    def attachment(
        estimate: dict[str, Any],
        gap: dict[str, Any],
        *,
        included_in_target: str | None = None,
    ) -> dict[str, Any]:
        value = {
            "sol_profile_id": sol_profile["sol_profile_id"],
            "status": gap["status"],
            "ideal_ms": estimate.get("ideal_ms"),
            "attainable_ms": estimate.get("attainable_ms"),
            "attainable_interval_ms": estimate.get("attainable_interval_ms"),
            "methodology_optimistic_ms": estimate.get(
                "methodology_optimistic_ms"
            ),
            "methodology_conservative_ms": estimate.get(
                "methodology_conservative_ms"
            ),
            "physical_coverage_pct": gap.get(
                "physical_coverage_pct", gap.get("ideal_efficiency_pct")
            ),
            "ideal_efficiency_pct": gap.get("ideal_efficiency_pct"),
            "attainable_coverage_pct": gap.get("attainable_coverage_pct"),
            "methodology_optimistic_coverage_pct": gap.get(
                "methodology_optimistic_coverage_pct"
            ),
            "methodology_conservative_coverage_pct": gap.get(
                "methodology_conservative_coverage_pct"
            ),
            "implementation_gap_ms": gap.get("implementation_gap_ms"),
            "limiting_resource": estimate.get("limiting_resource"),
            "critical_transition": estimate.get("critical_transition"),
            "limiter_vector": estimate.get("limiter_vector"),
            "kernel_plan_fingerprint": estimate.get(
                "kernel_plan_fingerprint"
            ),
            "confidence": estimate.get("confidence"),
            "diagnosis": gap.get("diagnosis"),
        }
        if included_in_target is not None:
            value.update(
                {
                    "status": "included_in_parent",
                    "included_in_target": included_in_target,
                    "allocation": "shared_parent_interval_non_additive",
                    "reason": (
                        "This fused semantic leaf shares the parent SoL interval; "
                        "do not sum the displayed value across covered leaves."
                    ),
                }
            )
        return value

    estimates = sol_profile["node_estimates"]
    gaps = gap_report["nodes"]
    aliases = sol_profile.get("fused_owner_aliases") or {}

    def iter_cells() -> Iterator[tuple[str, dict[str, Any]]]:
        """Yield top-level and nested drill cells with semantic targets."""

        def nested(
            cell: dict[str, Any], target: str
        ) -> Iterator[tuple[str, dict[str, Any]]]:
            yield target, cell
            drill_view = cell.get("drill_view")
            drill_metrics = cell.get("drill_metrics")
            if not isinstance(drill_view, str) or not isinstance(drill_metrics, dict):
                return
            for node_id, drill_cell in drill_metrics.items():
                if not isinstance(drill_cell, dict):
                    continue
                yield from nested(drill_cell, f"{drill_view}.{node_id}")

        for target, variants in (profile.get("data") or {}).items():
            if not isinstance(variants, dict):
                continue
            for cell in variants.values():
                if isinstance(cell, dict):
                    yield from nested(cell, str(target))

    def resolve_owner(
        start_target: str,
    ) -> tuple[str, dict[str, Any], dict[str, Any]] | None:
        """Resolve aliases/structural wrappers without inventing a cost split."""

        target = start_target
        seen: set[str] = set()
        while target not in seen:
            seen.add(target)
            estimate = estimates.get(target)
            gap = gaps.get(target)
            next_target = (
                (estimate or {}).get("included_in_target")
                or aliases.get(target)
            )
            if next_target:
                target = str(next_target)
                continue
            if estimate is not None and gap is not None:
                return target, estimate, gap
            return None
        return None

    # Attach in one pass so fused top-level cells and nested drill cells obey
    # the same ownership rule. A fused cell is resolved from its timing owner;
    # an unmodeled drill leaf may fall back to its validated drill scope. Both
    # cases remain explicitly non-additive.
    for target, cell in iter_cells():
        fused_owner = cell.get("included_in") if cell.get("status") == "fused" else None
        candidates = [
            str(fused_owner) if fused_owner else target,
            target,
            (
                str(cell.get("scope_target"))
                if cell.get("scope_target") and cell.get("status") != "structural"
                else ""
            ),
        ]
        resolved = None
        for candidate in candidates:
            if candidate and (resolved := resolve_owner(candidate)) is not None:
                break
        if resolved is None:
            continue
        owner_target, estimate, gap = resolved
        shared = owner_target != target or bool(fused_owner)
        cell["sol"] = attachment(
            estimate,
            gap,
            included_in_target=owner_target if shared else None,
        )
