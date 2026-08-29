"""Compile stable model IR plus execution, source, and profile overlays.

The V2 catalog deliberately keeps architecture independent from source code and
measurements:

* model_ir.yaml owns stable semantic nodes and data-flow edges;
* execution plans derive topology-specific graphs from that model IR;
* implementation bindings attach commit-specific symbols and links;
* profiles attach measurements to existing execution nodes.

The compiler emits a V2 bundle and a compatibility projection consumed by the
existing static viewer.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

from .sol import SolError, attach_sol_to_profile, build_sol_artifacts

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised by the CLI
    raise SystemExit("llm-arch-reviewer V2 requires pyyaml") from exc


class CatalogError(ValueError):
    """Raised when catalog inputs violate the V2 contract."""


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        value = yaml.safe_load(fh)
    if not isinstance(value, dict):
        raise CatalogError(f"{path}: expected a YAML mapping")
    return value


def _require(document: dict[str, Any], fields: Iterable[str], *, source: Path) -> None:
    missing = [field for field in fields if field not in document]
    if missing:
        raise CatalogError(f"{source}: missing required fields: {', '.join(missing)}")


def _validate_schema_version(
    document: dict[str, Any], expected: str, *, source: Path
) -> None:
    actual = document.get("schema_version")
    if actual != expected:
        raise CatalogError(f"{source}: expected schema_version={expected!r}, got {actual!r}")


def _validate_semantic_coverage(model_ir: dict[str, Any], *, source: Path) -> None:
    """Require explicit semantic-closure evidence for enriched Model IRs."""

    if int(model_ir.get("semantic_revision") or 0) < 3:
        return
    coverage = model_ir.get("semantic_coverage")
    if not isinstance(coverage, dict):
        raise CatalogError(
            f"{source}: semantic_revision>=3 requires semantic_coverage"
        )
    required = (
        "parameter_closure",
        "state_closure",
        "layer_variant_closure",
        "config_field_disposition",
    )
    missing = [field for field in required if not coverage.get(field)]
    if missing:
        raise CatalogError(
            f"{source}: semantic_coverage is missing: {', '.join(missing)}"
        )
    disposition = coverage["config_field_disposition"]
    if not isinstance(disposition, dict):
        raise CatalogError(
            f"{source}: semantic_coverage.config_field_disposition must be a mapping"
        )
    buckets = ("model_ir", "execution_ir", "binding_profile", "excluded")
    missing_buckets = [bucket for bucket in buckets if bucket not in disposition]
    if missing_buckets:
        raise CatalogError(
            f"{source}: config_field_disposition is missing buckets: "
            f"{', '.join(missing_buckets)}"
        )
    if int(model_ir.get("semantic_revision") or 0) >= 4 and not coverage.get(
        "operator_dataflow_closure"
    ):
        raise CatalogError(
            f"{source}: semantic_revision>=4 requires "
            "semantic_coverage.operator_dataflow_closure"
        )


def _validate_operator_granularity(model_ir: dict[str, Any], *, source: Path) -> None:
    """Reject compound leaves once operator/data-flow closure is claimed."""

    if int(model_ir.get("semantic_revision") or 0) < 4:
        return
    for view_id, view in (model_ir.get("views") or {}).items():
        for node in view.get("nodes", []) or []:
            operators = (node.get("semantic_details") or {}).get("operators") or []
            if len(operators) > 1 and not node.get("drill"):
                target = f"{view_id}.{node.get('id', '<missing>')}"
                raise CatalogError(
                    f"{source}: compound Model IR leaf {target} contains "
                    f"{len(operators)} operators; split it into primitive nodes "
                    "or add a drill view"
                )


def _validate_leaf_equation_coverage(
    model_ir: dict[str, Any], *, source: Path
) -> None:
    """Require every primitive compute leaf to define its own mathematics.

    A drill node may delegate its equation to the child data-flow view. A
    primitive leaf may not fall back to the viewer's generic ``Composite
    semantic module`` text: it must carry authored math, a canonical operator
    signature, or a narrowly documented non-mathematical exemption.
    """

    if int(model_ir.get("semantic_revision") or 0) < 6:
        return
    for view_id, view in (model_ir.get("views") or {}).items():
        for node in view.get("nodes", []) or []:
            if node.get("drill") or _inferred_semantic_kind(node) != "compute":
                continue
            target = f"{view_id}.{node.get('id', '<missing>')}"
            semantic_details = node.get("semantic_details") or {}
            has_math = bool(semantic_details.get("math"))
            has_signature = bool(
                (node.get("operator_signature") or {}).get("symbolic")
            )
            exemption = semantic_details.get("equation_exempt_reason")
            if exemption is not None and (
                not isinstance(exemption, str) or not exemption.strip()
            ):
                raise CatalogError(
                    f"{source}: compute leaf {target} has an empty "
                    "semantic_details.equation_exempt_reason"
                )
            if not (has_math or has_signature or exemption):
                raise CatalogError(
                    f"{source}: compute leaf {target} requires semantic_details.math, "
                    "operator_signature, or an explicit equation_exempt_reason"
                )


_DIMENSION_TRANSFORM_IN_LABEL = re.compile(
    r"(?:\[[A-Z0-9_, ×]+\]|(?:\d+|[A-Z][A-Za-z0-9_]*)(?:\s*×\s*(?:\d+|[A-Z][A-Za-z0-9_]*))?)"
    r"\s*→\s*(?:\[[A-Z0-9_, ×]+\]|\d+|[A-Z][A-Za-z0-9_]*)"
)
_SIGNATURE_SYMBOL = re.compile(
    r"(?<![A-Za-z0-9_])[A-Z][A-Za-z0-9_]*(?![A-Za-z0-9_])"
)


def _validate_notation_contract(model_ir: dict[str, Any], *, source: Path) -> None:
    """Keep tensor layouts, operator transforms, and visual classes orthogonal."""

    if int(model_ir.get("semantic_revision") or 0) < 6:
        return
    dimensions = model_ir.get("dimensions")
    if not isinstance(dimensions, dict) or not dimensions:
        raise CatalogError(
            f"{source}: semantic_revision>=6 requires a non-empty dimensions mapping"
        )
    for view_id, view in (model_ir.get("views") or {}).items():
        for node in view.get("nodes", []) or []:
            target = f"{view_id}.{node.get('id', '<missing>')}"
            signature = node.get("operator_signature")
            if signature is not None:
                if not isinstance(signature, dict) or not signature.get("symbolic"):
                    raise CatalogError(
                        f"{source}: node {target} operator_signature requires symbolic"
                    )
                if signature.get("concrete") is not None and not isinstance(
                    signature["concrete"], str
                ):
                    raise CatalogError(
                        f"{source}: node {target} "
                        "operator_signature.concrete must be a string"
                    )
                undeclared = sorted(
                    set(_SIGNATURE_SYMBOL.findall(signature["symbolic"]))
                    - set(dimensions)
                )
                if undeclared:
                    raise CatalogError(
                        f"{source}: node {target} operator_signature uses undeclared "
                        f"dimension symbols {undeclared}"
                    )
            if _DIMENSION_TRANSFORM_IN_LABEL.search(str(node.get("label") or "")):
                raise CatalogError(
                    f"{source}: node {target} embeds a dimension transform in label; "
                    "move it to operator_signature"
                )
        for edge in view.get("edges", []) or []:
            if edge.get("kind", "data") == "control":
                continue
            if not edge.get("shape") or not edge.get("dtype"):
                raise CatalogError(
                    f"{source}: tensor-carrying edge {view_id}."
                    f"{edge.get('from')}->{edge.get('to')} requires shape and dtype; "
                    "mark non-tensor dependencies as kind=control"
                )


def _node_index(views: dict[str, Any], *, source: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    allowed_shapes = {"io", "block", "gemm", "attn", "moe", "norm", "elem", "cache"}
    for view_id, view in views.items():
        if not isinstance(view, dict):
            raise CatalogError(f"{source}: view {view_id!r} must be a mapping")
        node_ids: set[str] = set()
        for node in view.get("nodes", []) or []:
            if not isinstance(node, dict):
                raise CatalogError(f"{source}: view {view_id!r} has a non-mapping node")
            node_id = node.get("id")
            if not node_id:
                raise CatalogError(f"{source}: view {view_id!r} has a node without id")
            for field in ("label", "shape", "semantic_op"):
                if not node.get(field):
                    raise CatalogError(
                        f"{source}: node {view_id}.{node_id} is missing {field!r}"
                    )
            semantic_details = node.get("semantic_details")
            if semantic_details is not None and not isinstance(semantic_details, dict):
                raise CatalogError(
                    f"{source}: node {view_id}.{node_id} "
                    "semantic_details must be a mapping"
                )
            if isinstance(semantic_details, dict):
                for field in (
                    "operators",
                    "math",
                    "tensors",
                    "state",
                    "invariants",
                    "conditions",
                    "notes",
                    "provenance",
                ):
                    value = semantic_details.get(field)
                    if value is not None and not isinstance(value, list):
                        raise CatalogError(
                            f"{source}: node {view_id}.{node_id} "
                            f"semantic_details.{field} must be a list"
                        )
                runtime_mapping = semantic_details.get("runtime_mapping")
                if runtime_mapping is not None and not isinstance(runtime_mapping, dict):
                    raise CatalogError(
                        f"{source}: node {view_id}.{node_id} "
                        "semantic_details.runtime_mapping must be a mapping"
                    )
            if node["shape"] not in allowed_shapes:
                raise CatalogError(
                    f"{source}: node {view_id}.{node_id} has unknown shape {node['shape']!r}"
                )
            if node_id in node_ids:
                raise CatalogError(
                    f"{source}: duplicate node id {node_id!r} in view {view_id!r}"
                )
            node_ids.add(node_id)
            index[f"{view_id}.{node_id}"] = node

        for edge in view.get("edges", []) or []:
            for endpoint in ("from", "to"):
                if edge.get(endpoint) not in node_ids:
                    raise CatalogError(
                        f"{source}: edge in {view_id!r} references unknown "
                        f"{endpoint} node {edge.get(endpoint)!r}"
                    )
        for node in view.get("nodes", []) or []:
            drill = node.get("drill")
            if drill and drill not in views:
                raise CatalogError(
                    f"{source}: node {view_id}.{node['id']} drills into unknown view {drill!r}"
                )
    for target, node in index.items():
        runtime_mapping = (node.get("semantic_details") or {}).get("runtime_mapping")
        if not runtime_mapping:
            continue
        expectation = runtime_mapping.get("expectation")
        if expectation not in {"measured", "fused", "fused_state", "structural"}:
            raise CatalogError(
                f"{source}: node {target} has unknown runtime-mapping expectation "
                f"{expectation!r}"
            )
        if expectation == "measured" and runtime_mapping.get("profile_leaf") != target:
            raise CatalogError(
                f"{source}: measured node {target} must name itself as profile_leaf"
            )
        owner = runtime_mapping.get("owner")
        if expectation in {"fused", "fused_state"} and owner not in index:
            raise CatalogError(
                f"{source}: fused node {target} references unknown owner {owner!r}"
            )
    return index


def _split_target(target: str, *, source: Path) -> tuple[str, str]:
    if "." not in target:
        raise CatalogError(f"{source}: target {target!r} must be '<view>.<node>'")
    return target.split(".", 1)


def _find_node(views: dict[str, Any], target: str, *, source: Path) -> dict[str, Any]:
    view_id, node_id = _split_target(target, source=source)
    view = views.get(view_id)
    if not view:
        raise CatalogError(f"{source}: target {target!r} references unknown view")
    for node in view.get("nodes", []) or []:
        if node.get("id") == node_id:
            return node
    raise CatalogError(f"{source}: target {target!r} references unknown node")


_EDGE_CONTRACT_FIELDS = ("identity", "shape", "layout", "dtype", "state")


def _tensor_contract(edge: dict[str, Any], *, endpoint: str) -> dict[str, Any]:
    """Return the canonical tensor/state payload carried by one graph edge."""

    endpoint_key = {"source": "from", "target": "to"}[endpoint]
    return {
        "name": str(edge.get("identity") or f"{edge.get('from')}_to_{edge.get('to')}"),
        "shape": str(edge.get("shape") or "unspecified"),
        "layout": str(edge.get("layout") or "unspecified"),
        "dtype": str(edge.get("dtype") or "unspecified"),
        "state": str(edge.get("state") or "unspecified"),
        endpoint: str(edge[endpoint_key]),
        **({"kind": str(edge["kind"])} if edge.get("kind") else {}),
    }


def _inferred_semantic_kind(node: dict[str, Any]) -> str:
    semantic_op = str(node.get("semantic_op") or "")
    if node.get("shape") == "io":
        return "boundary"
    if node.get("shape") == "cache":
        return "state"
    if "schedule" in semantic_op or semantic_op.endswith(".block_stack"):
        return "control"
    if node.get("drill") or node.get("shape") == "block":
        return "module"
    return "compute"


def _compact_contract_text(value: Any) -> str:
    return "".join(str(value or "").split())


def _validate_boundary_contracts(model_ir: dict[str, Any], *, source: Path) -> None:
    """Validate that every drill-down exposes an explicit shape boundary.

    ``exact_node`` contracts must agree with both the parent graph edges and
    the child view's declared boundary nodes. ``exact_lifecycle`` is used when
    one semantic drill spans a pre/sublayer/post lifecycle (for example mHC);
    the scope and handoff are explicit instead of pretending one node owns the
    whole transformation. ``external_entry`` marks a generation/runtime entry
    that has inputs supplied outside the selected parent view.
    """

    if not model_ir.get("semantic_contract"):
        return
    views = model_ir["views"]
    node_index = _node_index(views, source=source)
    drills = {
        f"{view_id}.{node['id']}": str(node["drill"])
        for view_id, view in views.items()
        for node in view.get("nodes", []) or []
        if node.get("drill")
    }
    contracts = model_ir.get("boundary_contracts") or []
    by_parent = {
        str(contract.get("parent_node")): contract
        for contract in contracts
        if isinstance(contract, dict)
    }
    missing = sorted(set(drills) - set(by_parent))
    extra = sorted(set(by_parent) - set(drills))
    if missing or extra:
        raise CatalogError(
            f"{source}: drill boundary contracts differ from drill nodes; "
            f"missing={missing}, extra={extra}"
        )

    for parent_target, child_view_id in drills.items():
        contract = by_parent[parent_target]
        if contract.get("child_view") != child_view_id:
            raise CatalogError(
                f"{source}: boundary contract {parent_target} names child "
                f"{contract.get('child_view')!r}, expected {child_view_id!r}"
            )
        mode = str(contract.get("boundary_mode") or "")
        if mode not in {"exact_node", "exact_lifecycle", "external_entry"}:
            raise CatalogError(
                f"{source}: boundary contract {parent_target} requires boundary_mode"
            )
        input_text = _compact_contract_text(contract.get("input_shape"))
        output_text = _compact_contract_text(contract.get("output_shape"))
        handoff_text = _compact_contract_text(contract.get("handoff_shape"))
        if not input_text or not output_text:
            raise CatalogError(
                f"{source}: boundary contract {parent_target} requires input/output shapes"
            )

        child_view = views[child_view_id]
        child_nodes = {str(node["id"]): node for node in child_view.get("nodes", []) or []}
        directions = {
            node_id: str(node.get("boundary_direction"))
            for node_id, node in child_nodes.items()
            if node.get("boundary_direction")
        }
        if not directions:
            raise CatalogError(
                f"{source}: child view {child_view_id!r} requires explicit "
                "boundary_direction nodes"
            )
        for node_id, direction in directions.items():
            if direction in {"input", "handoff"}:
                shapes = {
                    _compact_contract_text(edge.get("shape"))
                    for edge in child_view.get("edges", []) or []
                    if edge.get("from") == node_id
                }
                declared = input_text if direction == "input" else handoff_text
            else:
                shapes = {
                    _compact_contract_text(edge.get("shape"))
                    for edge in child_view.get("edges", []) or []
                    if edge.get("to") == node_id
                }
                declared = output_text
            for shape in shapes:
                if shape and shape not in declared:
                    raise CatalogError(
                        f"{source}: {parent_target} {direction} contract does not "
                        f"contain child boundary shape {shape!r} from "
                        f"{child_view_id}.{node_id}"
                    )

        if mode == "exact_node":
            parent_view_id, parent_node_id = _split_target(parent_target, source=source)
            parent_view = views[parent_view_id]
            parent_inputs = {
                _compact_contract_text(edge.get("shape"))
                for edge in parent_view.get("edges", []) or []
                if edge.get("to") == parent_node_id and edge.get("kind") != "dashed"
            }
            parent_outputs = {
                _compact_contract_text(edge.get("shape"))
                for edge in parent_view.get("edges", []) or []
                if edge.get("from") == parent_node_id
                and edge.get("kind") != "dashed"
                and edge.get("state") != "captured_optional_state"
            }
            for shape in parent_inputs:
                if shape and shape not in input_text:
                    raise CatalogError(
                        f"{source}: {parent_target} input contract omits parent edge "
                        f"shape {shape!r}"
                    )
            for shape in parent_outputs:
                if shape and shape not in output_text:
                    raise CatalogError(
                        f"{source}: {parent_target} output contract omits parent edge "
                        f"shape {shape!r}"
                    )
        elif mode == "exact_lifecycle":
            scope_nodes = [str(target) for target in contract.get("scope_nodes") or []]
            if parent_target not in scope_nodes or len(scope_nodes) < 2:
                raise CatalogError(
                    f"{source}: lifecycle contract {parent_target} requires a "
                    "multi-node scope containing its parent"
                )
            unknown_scope = sorted(set(scope_nodes) - set(node_index))
            if unknown_scope:
                raise CatalogError(
                    f"{source}: lifecycle contract {parent_target} has unknown "
                    f"scope nodes {unknown_scope}"
                )
            if not handoff_text:
                raise CatalogError(
                    f"{source}: lifecycle contract {parent_target} requires handoff_shape"
                )


def _compile_semantic_transitions(
    model_ir: dict[str, Any], views: dict[str, Any], *, source: Path
) -> None:
    """Attach node-local semantic transitions derived from canonical edges.

    Edges remain the single source of truth for tensor/state contracts.  The
    model-level operation table owns mathematics.  The compiled node combines
    both, so the viewer and validators never rely on prose embedded in labels.
    """

    config = model_ir.get("semantic_contract") or {}
    strict = int(config.get("version") or 0) >= 1
    require_equations = bool(config.get("require_explicit_equations", strict))
    operations = config.get("operations") or {}
    if strict and not isinstance(operations, dict):
        raise CatalogError(f"{source}: semantic_contract.operations must be a mapping")

    used_operations: set[str] = set()
    boundary_contracts = {
        str(entry.get("parent_node")): entry
        for entry in (model_ir.get("boundary_contracts") or [])
        if isinstance(entry, dict)
    }

    for view_id, view in views.items():
        incoming: dict[str, list[dict[str, Any]]] = {
            str(node["id"]): [] for node in view.get("nodes", []) or []
        }
        outgoing: dict[str, list[dict[str, Any]]] = {
            str(node["id"]): [] for node in view.get("nodes", []) or []
        }
        for edge in view.get("edges", []) or []:
            if strict:
                missing = [field for field in _EDGE_CONTRACT_FIELDS if not edge.get(field)]
                if missing:
                    raise CatalogError(
                        f"{source}: edge {view_id}.{edge.get('from')} -> "
                        f"{edge.get('to')} is missing semantic contract fields: "
                        f"{', '.join(missing)}"
                    )
            incoming[str(edge["to"])].append(_tensor_contract(edge, endpoint="source"))
            outgoing[str(edge["from"])].append(_tensor_contract(edge, endpoint="target"))

        for node in view.get("nodes", []) or []:
            semantic_op = str(node["semantic_op"])
            semantic_details = node.get("semantic_details") or {}
            detail_math = [str(item) for item in semantic_details.get("math") or []]
            equation_exemption = str(
                semantic_details.get("equation_exempt_reason") or ""
            )
            signature = node.get("operator_signature") or {}
            signature_equation = str(signature.get("symbolic") or "")
            if signature.get("concrete"):
                signature_equation += f" ({signature['concrete']})"
            inferred_kind = _inferred_semantic_kind(node)
            is_model_node = node.get("ir_origin") != "execution_plan"
            operation = operations.get(semantic_op)
            if operation is not None and not isinstance(operation, dict):
                raise CatalogError(
                    f"{source}: operation contract {semantic_op!r} must be a mapping"
                )
            if operation:
                used_operations.add(semantic_op)
            if strict and is_model_node and require_equations and not operation:
                raise CatalogError(
                    f"{source}: model node {view_id}.{node['id']} requires an "
                    f"explicit operation contract for {semantic_op!r}"
                )
            equation = str((operation or {}).get("equation") or "")
            if (
                strict
                and is_model_node
                and require_equations
                and not equation
            ):
                raise CatalogError(
                    f"{source}: model node {view_id}.{node['id']} requires an equation"
                )

            target = f"{view_id}.{node['id']}"
            drill_contract = boundary_contracts.get(target)
            execution = node.get("execution")
            execution_equation = ""
            if isinstance(execution, dict) and all(
                execution.get(field) for field in ("result", "collective", "payload")
            ):
                execution_equation = (
                    f"{execution['result']} = "
                    f"{execution['collective']}({execution['payload']})"
                )
            node["semantics"] = {
                "semantic_op": semantic_op,
                "kind": str((operation or {}).get("kind") or inferred_kind),
                "summary": str(node.get("label") or node["id"]).splitlines()[0],
                "equation": equation or "\n".join(detail_math) or signature_equation or (
                    execution_equation
                    if execution_equation
                    else f"No model equation: {equation_exemption}"
                    if equation_exemption
                    else
                    f"See {node['drill']} semantic data-flow"
                    if node.get("drill")
                    else "No value transformation; semantic boundary"
                    if inferred_kind == "boundary"
                    else "Persistent state read/write"
                    if inferred_kind == "state"
                    else "Execution path selection"
                    if inferred_kind == "control"
                    else "Composite semantic module"
                ),
                "inputs": sorted(incoming[str(node["id"])], key=lambda item: (item["name"], item["source"])),
                "outputs": sorted(outgoing[str(node["id"])], key=lambda item: (item["name"], item["target"])),
                "invariants": list(
                    dict.fromkeys(
                        [
                            *copy.deepcopy((operation or {}).get("invariants", [])),
                            *[str(item) for item in semantic_details.get("invariants") or []],
                        ]
                    )
                ),
                "contract_source": (
                    "model_ir.semantic_contract+edges"
                    if operation
                    else "model_ir.semantic_details+edges"
                    if semantic_details or signature
                    else "model_ir.edges"
                ),
                **(
                    {"notes": str(operation["notes"])}
                    if (operation or {}).get("notes")
                    else {"notes": "\n".join(str(item) for item in semantic_details.get("notes") or [])}
                    if semantic_details.get("notes")
                    else {}
                ),
                **({"drill_boundary": copy.deepcopy(drill_contract)} if drill_contract else {}),
            }

    unused = sorted(set(operations) - used_operations)
    if strict and unused:
        raise CatalogError(
            f"{source}: semantic_contract has operations not used by any node: {unused}"
        )


def _model_views_with_provenance(
    views: dict[str, Any],
    *,
    model_ir: dict[str, Any] | None = None,
    source: Path | None = None,
) -> dict[str, Any]:
    """Return the stable semantic graph with explicit compiled provenance.

    Persisted Model IR stays implementation-independent.  The compiled bundle
    adds provenance so the viewer can distinguish canonical semantic nodes from
    nodes introduced by an execution plan without requiring authors to repeat
    that fact on every node.
    """

    compiled = copy.deepcopy(views)
    for view in compiled.values():
        for node in view.get("nodes", []) or []:
            node["ir_origin"] = "model_ir"
            node.setdefault("node_kind", "semantic")
    if model_ir is not None:
        _compile_semantic_transitions(
            model_ir,
            compiled,
            source=source or Path("model_ir.yaml"),
        )
    return compiled


def _deep_merge(target: dict[str, Any], overlay: dict[str, Any]) -> None:
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def _insert_after(
    views: dict[str, Any], transform: dict[str, Any], *, source: Path
) -> None:
    view_id, after_id = _split_target(transform.get("after", ""), source=source)
    view = views.get(view_id)
    if not view:
        raise CatalogError(f"{source}: insert_after references unknown view {view_id!r}")
    nodes = view.get("nodes", []) or []
    if not any(node.get("id") == after_id for node in nodes):
        raise CatalogError(f"{source}: insert_after references unknown node {transform['after']!r}")

    new_node = copy.deepcopy(transform.get("node"))
    if not isinstance(new_node, dict) or not new_node.get("id"):
        raise CatalogError(f"{source}: insert_after requires node.id")
    semantic_op = str(new_node.get("semantic_op") or "")
    if not semantic_op.startswith("execution."):
        raise CatalogError(
            f"{source}: inserted node {view_id}.{new_node['id']} must use an "
            "execution.* semantic_op"
        )
    execution = new_node.get("execution")
    if not isinstance(execution, dict):
        raise CatalogError(
            f"{source}: inserted node {view_id}.{new_node['id']} requires execution metadata"
        )
    for field in ("placement", "collective", "parallelism"):
        if not execution.get(field):
            raise CatalogError(
                f"{source}: inserted node {view_id}.{new_node['id']} requires "
                f"execution.{field}"
            )
    for field in ("payload", "result"):
        if not execution.get(field):
            raise CatalogError(
                f"{source}: inserted communication node {view_id}.{new_node['id']} "
                f"requires execution.{field}"
            )
    collective = str(execution["collective"])
    inferred_kind = (
        "layout_transform"
        if collective in {"local_slice", "local_index", "local_select"}
        else "communication"
    )
    new_node["ir_origin"] = "execution_plan"
    new_node.setdefault("node_kind", inferred_kind)
    new_node.setdefault("boundary_role", "module_boundary")
    if new_node["node_kind"] not in {"communication", "layout_transform"}:
        raise CatalogError(
            f"{source}: inserted node {view_id}.{new_node['id']} has invalid "
            f"node_kind {new_node['node_kind']!r}"
        )
    if new_node["boundary_role"] not in {"module_boundary", "module_internal"}:
        raise CatalogError(
            f"{source}: inserted node {view_id}.{new_node['id']} has invalid "
            f"boundary_role {new_node['boundary_role']!r}"
        )
    if any(node.get("id") == new_node["id"] for node in nodes):
        raise CatalogError(
            f"{source}: insert_after creates duplicate node {view_id}.{new_node['id']}"
        )

    insert_at = next(i for i, node in enumerate(nodes) if node.get("id") == after_id) + 1
    nodes.insert(insert_at, new_node)
    view["nodes"] = nodes

    redirected: list[dict[str, Any]] = []
    for edge in view.get("edges", []) or []:
        edge_copy = copy.deepcopy(edge)
        if edge_copy.get("from") == after_id:
            edge_copy["from"] = new_node["id"]
        redirected.append(edge_copy)
    redirected.append(
        {
            "from": after_id,
            "to": new_node["id"],
            **copy.deepcopy(transform.get("edge", {})),
        }
    )
    view["edges"] = redirected


def apply_execution_plan(
    model_ir: dict[str, Any], plan: dict[str, Any], *, source: Path
) -> dict[str, Any]:
    views = _model_views_with_provenance(model_ir["views"])
    for transform in plan.get("transforms", []) or []:
        op = transform.get("op")
        if op == "annotate_node":
            node = _find_node(views, transform.get("target", ""), source=source)
            overlay = transform.get("set")
            if not isinstance(overlay, dict):
                raise CatalogError(f"{source}: annotate_node requires a 'set' mapping")
            _deep_merge(node, overlay)
        elif op == "insert_after":
            _insert_after(views, transform, source=source)
        else:
            raise CatalogError(f"{source}: unsupported execution transform {op!r}")
    _node_index(views, source=source)
    _compile_semantic_transitions(model_ir, views, source=source)
    return views


def _fingerprint_payload(
    model_ir: dict[str, Any], plan: dict[str, Any], views: dict[str, Any]
) -> dict[str, Any]:
    """Return the structural payload used for execution-path identity.

    Labels, notes, source links, profiles, and concrete TP degree are excluded.
    A TP-only template therefore keeps one identity across TP2/TP4/TP8 while
    topology or communication-graph changes produce a new fingerprint.
    """

    semantic_only_views = {
        view_id
        for view_id, view in views.items()
        if view.get("execution_contract") is False
    }
    structural_views: dict[str, Any] = {}
    for view_id, view in sorted(views.items()):
        if view_id in semantic_only_views:
            continue
        structural_views[view_id] = {
            "nodes": [
                {
                    key: copy.deepcopy(node[key])
                    for key in (
                        "id",
                        "shape",
                        "drill",
                        "semantic_op",
                        "execution",
                    )
                    if key in node
                    and not (
                        key == "drill" and node.get("drill") in semantic_only_views
                    )
                }
                for node in view.get("nodes", []) or []
            ],
            "edges": [
                {
                    key: copy.deepcopy(edge[key])
                    for key in ("from", "to", "kind", "shape", "dtype")
                    if key in edge
                }
                for edge in view.get("edges", []) or []
            ],
        }
    return {
        "model_id": model_ir["model_id"],
        "model_ir_version": model_ir["ir_version"],
        "execution_path_id": plan["execution_path_id"],
        "execution_plan_version": plan["plan_version"],
        "parallelism_axes": plan.get("parallelism_axes", {}),
        "views": structural_views,
    }


def execution_fingerprint(
    model_ir: dict[str, Any], plan: dict[str, Any], views: dict[str, Any]
) -> str:
    payload = json.dumps(
        _fingerprint_payload(model_ir, plan, views),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return "exec_" + hashlib.sha256(payload).hexdigest()[:16]


def derive_parent_map(views: dict[str, Any]) -> dict[str, str]:
    parents: dict[str, str] = {}
    for view_id, view in views.items():
        for node in view.get("nodes", []) or []:
            drill = node.get("drill")
            if drill and drill not in parents:
                parents[drill] = view_id
    return parents


def _code_link(
    source_repo: str, source_commit: str, raw_link: dict[str, Any]
) -> dict[str, Any]:
    file_path = raw_link["file"]
    line = raw_link.get("line")
    line_end = raw_link.get("line_end")
    anchor = ""
    if line:
        anchor = f"#L{line}"
        if line_end and line_end != line:
            anchor += f"-L{line_end}"
    url = f"{source_repo.rstrip('/')}/blob/{source_commit}/{file_path}{anchor}"
    display = raw_link.get("display") or raw_link.get("symbol") or file_path
    return {
        "raw": raw_link.get("symbol") or display,
        "file": file_path,
        "line": line,
        "line_end": line_end,
        "url": url,
        "display": display,
        "symbol": raw_link.get("symbol"),
    }


def compile_binding(binding: dict[str, Any], *, source: Path) -> dict[str, Any]:
    compiled = {
        key: copy.deepcopy(binding[key])
        for key in (
            "implementation_id",
            "label",
            "model_id",
            "execution_path_id",
            "source_repo",
            "source_commit",
            "source_patch_sha256",
            "container",
            "backend",
            "binding_status",
            "source_lock_status",
            "execution_validation",
            "extends",
            "binding_compatible_base_commit",
        )
        if key in binding
    }
    compiled["node_bindings"] = {}
    for target, node_binding in (binding.get("node_bindings") or {}).items():
        if not isinstance(node_binding, dict):
            raise CatalogError(f"{source}: binding for {target!r} must be a mapping")
        for link in node_binding.get("links", []) or []:
            if not isinstance(link, dict) or not link.get("file") or not link.get("symbol"):
                raise CatalogError(
                    f"{source}: binding link for {target!r} requires file and symbol"
                )
        compiled_binding = copy.deepcopy(node_binding)
        compiled_binding["code_links"] = [
            _code_link(binding["source_repo"], str(binding["source_commit"]), link)
            for link in node_binding.get("links", []) or []
        ]
        compiled["node_bindings"][target] = compiled_binding
    return compiled


def apply_binding(views: dict[str, Any], binding: dict[str, Any], *, source: Path) -> None:
    for target, node_binding in (binding.get("node_bindings") or {}).items():
        node = _find_node(views, target, source=source)
        node["code_links"] = copy.deepcopy(node_binding.get("code_links", []))
        node["implementation_binding"] = {
            "implementation_id": binding["implementation_id"],
            "symbols": copy.deepcopy(node_binding.get("symbols", [])),
            "kernel_signatures": copy.deepcopy(
                node_binding.get("kernel_signatures", [])
            ),
        }
    # Primitive Model IR nodes can share one fused implementation interval.
    # Inherit the owner's source binding without duplicating timing ownership.
    index = _node_index(views, source=source)
    for target, node in index.items():
        if node.get("implementation_binding"):
            continue
        runtime_mapping = (node.get("semantic_details") or {}).get("runtime_mapping") or {}
        if runtime_mapping.get("expectation") not in {"fused", "fused_state"}:
            continue
        owner = index.get(str(runtime_mapping.get("owner") or ""))
        if not owner or not owner.get("implementation_binding"):
            continue
        node["code_links"] = copy.deepcopy(owner.get("code_links", []))
        node["implementation_binding"] = copy.deepcopy(owner["implementation_binding"])
        node["implementation_binding"]["mapping_provenance"] = "shared_fused_owner"
        node["implementation_binding"]["timing_owner"] = runtime_mapping["owner"]


def _validate_parallelism(
    profile: dict[str, Any], plan: dict[str, Any], *, source: Path
) -> None:
    params = profile.get("execution_parameters") or {}
    _require(params, ("tp_size", "dp_size", "cp_size", "ep_size"), source=source)
    axes = plan.get("parallelism_axes") or {}
    for axis, expected in axes.items():
        if isinstance(expected, int) and params.get(axis) != expected:
            raise CatalogError(
                f"{source}: {axis}={params.get(axis)!r} does not match "
                f"execution plan requirement {expected!r}"
            )
    min_tp = (plan.get("constraints") or {}).get("tp_size_min")
    if min_tp is not None and int(params["tp_size"]) < int(min_tp):
        raise CatalogError(f"{source}: tp_size must be >= {min_tp}")


def compile_profile(
    profile: dict[str, Any],
    *,
    plan: dict[str, Any],
    fingerprint: str,
    node_targets: set[str],
    node_index: dict[str, dict[str, Any]] | None = None,
    source: Path,
) -> dict[str, Any]:
    _validate_parallelism(profile, plan, source=source)
    effective_states = copy.deepcopy(profile.get("node_states") or {})
    if profile.get("generation_mode", "autoregressive") != "eagle_mtp":
        for target in sorted(node_targets):
            if target.startswith("mtp_"):
                effective_states.setdefault(
                    target,
                    {
                        "status": "disabled",
                        "label": "MTP is disabled in this profile",
                    },
                )
    # Semantic refinement must not force every historical profile to duplicate
    # a fused kernel interval. Runtime-mapping contracts synthesize explicit
    # fused states while the measured owner remains the sole timing source.
    for target, node in (node_index or {}).items():
        if target in effective_states or target in (profile.get("node_metrics") or {}):
            continue
        runtime_mapping = (node.get("semantic_details") or {}).get("runtime_mapping") or {}
        if runtime_mapping.get("expectation") not in {"fused", "fused_state"}:
            continue
        owner = str(runtime_mapping.get("owner") or "")
        # A profile may fuse the semantic owner again into a larger runtime
        # interval (for example beta/decay gating into the delta-rule kernel).
        # Flatten that ownership chain so primitive children join the one
        # physical event set instead of creating overlapping nested groups.
        visited = {target}
        while (
            owner not in visited
            and (effective_states.get(owner) or {}).get("status") == "fused"
        ):
            visited.add(owner)
            owner = str(effective_states[owner].get("included_in") or "")
        effective_states[target] = {
            "status": "fused",
            "included_in": owner,
            "label": runtime_mapping.get("reason")
            or "shares the fused runtime interval with its timing owner",
            "provenance": "model_ir.runtime_mapping",
        }
    unknown = sorted(set(profile.get("node_metrics") or {}) - node_targets)
    if unknown:
        raise CatalogError(f"{source}: profile references unknown nodes: {unknown}")
    for target, metric in (profile.get("node_metrics") or {}).items():
        if not isinstance(metric, dict) or "ms_per_iter" not in metric:
            raise CatalogError(f"{source}: metric {target!r} requires ms_per_iter")
        if float(metric["ms_per_iter"]) < 0:
            raise CatalogError(f"{source}: metric {target!r} has negative ms_per_iter")

    unknown_states = sorted(set(effective_states) - node_targets)
    if unknown_states:
        raise CatalogError(
            f"{source}: profile states reference unknown nodes: {unknown_states}"
        )
    for target, state in effective_states.items():
        if not isinstance(state, dict) or not state.get("status"):
            raise CatalogError(f"{source}: state {target!r} requires status")
        if state.get("status") == "fused":
            if target in (profile.get("node_metrics") or {}):
                raise CatalogError(
                    f"{source}: fused state {target!r} cannot also carry "
                    "independent node_metrics; either make it a measured timing "
                    "owner or remove the metric and point it at included_in"
                )
            owner = state.get("included_in")
            if not owner:
                raise CatalogError(
                    f"{source}: fused state {target!r} requires included_in"
                )
            if owner not in node_targets:
                raise CatalogError(
                    f"{source}: fused state {target!r} references unknown owner {owner!r}"
                )
            if owner == target:
                raise CatalogError(
                    f"{source}: fused state {target!r} cannot include itself"
                )

    # Fusion is an implementation/profile property, not architecture.  Promote
    # the existing fused/included_in states into explicit many-to-many groups
    # with shared-interval timing semantics so consumers never add the same
    # kernel residency once per covered semantic node.
    fusion_members: dict[str, list[str]] = {}
    for target, state in effective_states.items():
        if state.get("status") != "fused":
            continue
        owner = str(state["included_in"])
        fusion_members.setdefault(owner, []).append(target)
    derived_fusion_groups = {
        f"fusion:{owner}": {
            "owner": owner,
            "ir_nodes": [owner, *sorted(members)],
            "timing_semantics": "shared_interval",
            "provenance": "profile.node_states",
        }
        for owner, members in sorted(fusion_members.items())
    }
    fusion_groups: dict[str, dict[str, Any]] = {}
    covered_by_authored: set[str] = set()
    authored_group_by_owner: dict[str, str] = {}
    authored_group_for_target: dict[str, str] = {}
    for group_id, raw_group in (profile.get("fusion_groups") or {}).items():
        if not isinstance(raw_group, dict):
            raise CatalogError(f"{source}: fusion group {group_id!r} must be a mapping")
        owner = str(raw_group.get("owner") or "")
        ir_nodes = list(dict.fromkeys(raw_group.get("ir_nodes") or []))
        if len(ir_nodes) < 2 or owner not in ir_nodes:
            raise CatalogError(
                f"{source}: fusion group {group_id!r} requires an owner contained "
                "in at least two ir_nodes"
            )
        unknown_fusion_nodes = sorted(set(ir_nodes) - node_targets)
        if unknown_fusion_nodes:
            raise CatalogError(
                f"{source}: fusion group {group_id!r} references unknown nodes: "
                f"{unknown_fusion_nodes}"
            )
        if raw_group.get("timing_semantics") not in {
            "shared_interval",
            "shared_event_set",
        }:
            raise CatalogError(
                f"{source}: fusion group {group_id!r} requires "
                "timing_semantics='shared_interval' or 'shared_event_set'"
            )
        evidence_scope = raw_group.get("evidence_scope") or {}
        resolution = evidence_scope.get("resolution")
        if raw_group.get("timing_semantics") == "shared_event_set" and resolution not in {
            "exact_occurrence",
            "profile_aggregate",
        }:
            raise CatalogError(
                f"{source}: shared_event_set fusion group {group_id!r} requires "
                "an evidence_scope resolution"
            )
        if (
            raw_group.get("timing_semantics") == "shared_interval"
            and resolution == "profile_aggregate"
        ):
            raise CatalogError(
                f"{source}: fusion group {group_id!r} cannot call a profile "
                "aggregate one shared_interval"
            )
        if owner in authored_group_by_owner:
            raise CatalogError(
                f"{source}: fusion owner {owner!r} appears in multiple authored "
                "groups; one owner must have one unambiguous event-set closure"
            )
        overlap = covered_by_authored.intersection(ir_nodes)
        if overlap:
            raise CatalogError(
                f"{source}: fusion group {group_id!r} overlaps another authored "
                f"group at {sorted(overlap)}"
            )
        covered_by_authored.update(ir_nodes)
        fusion_groups[str(group_id)] = copy.deepcopy(raw_group)
        authored_group_by_owner[owner] = str(group_id)
        authored_group_for_target.update(
            {target: str(group_id) for target in ir_nodes}
        )
    for group_id, group in derived_fusion_groups.items():
        owner = group["owner"]
        members = group["ir_nodes"][1:]
        requested_group_ids = {
            str(effective_states[target].get("fusion_group_id"))
            for target in members
            if effective_states[target].get("fusion_group_id")
        }
        if len(requested_group_ids) > 1:
            raise CatalogError(
                f"{source}: fused nodes for owner {owner!r} request multiple "
                f"fusion groups: {sorted(requested_group_ids)}"
            )
        selected_group_id = next(iter(requested_group_ids), None)
        selected_group_id = selected_group_id or authored_group_by_owner.get(owner)
        if selected_group_id:
            if selected_group_id not in fusion_groups:
                raise CatalogError(
                    f"{source}: fused nodes for owner {owner!r} reference unknown "
                    f"fusion group {selected_group_id!r}"
                )
            selected_group = fusion_groups[selected_group_id]
            if selected_group["owner"] != owner:
                raise CatalogError(
                    f"{source}: fusion group {selected_group_id!r} owns "
                    f"{selected_group['owner']!r}, not included_in owner {owner!r}"
                )
            conflicts = {
                target: authored_group_for_target[target]
                for target in members
                if target in authored_group_for_target
                and authored_group_for_target[target] != selected_group_id
            }
            if conflicts:
                raise CatalogError(
                    f"{source}: fused nodes for owner {owner!r} conflict with "
                    f"authored groups: {conflicts}"
                )
            selected_group["ir_nodes"] = list(
                dict.fromkeys([*selected_group["ir_nodes"], *members])
            )
            selected_group.setdefault("derived_from_node_states", True)
            continue
        overlap = covered_by_authored.intersection(group["ir_nodes"])
        if overlap:
            raise CatalogError(
                f"{source}: derived fusion group {group_id!r} overlaps authored "
                f"coverage at {sorted(overlap)} without a matching owner/group"
            )
        fusion_groups[group_id] = group
    fusion_group_for_target = {
        target: group_id
        for group_id, group in fusion_groups.items()
        for target in group["ir_nodes"]
    }
    for target, state in effective_states.items():
        if state.get("status") != "fused":
            continue
        group_id = fusion_group_for_target.get(target)
        if not group_id:
            raise CatalogError(
                f"{source}: fused state {target!r} has no compiled fusion group"
            )
        group = fusion_groups[group_id]
        if group["owner"] != state["included_in"]:
            raise CatalogError(
                f"{source}: fused state {target!r} points to {state['included_in']!r} "
                f"but compiled fusion group owns {group['owner']!r}"
            )

    variant = profile["variant_id"]
    data = {}
    targets = list(effective_states)
    targets.extend(
        target
        for target in (profile.get("node_metrics") or {})
        if target not in effective_states
    )
    for target in sorted(targets):
        cell = copy.deepcopy(effective_states.get(target, {}))
        cell.update(copy.deepcopy((profile.get("node_metrics") or {}).get(target, {})))
        if (
            "gpu_residency_ms" not in cell
            and "gpu_residency_ms_per_iter" in cell
        ):
            cell["gpu_residency_ms"] = cell["gpu_residency_ms_per_iter"]
        group_id = fusion_group_for_target.get(target)
        if group_id:
            group = fusion_groups[group_id]
            cell["fusion_group_id"] = group_id
            cell["fusion_timing_semantics"] = group["timing_semantics"]
            if target == group["owner"]:
                cell["timing_role"] = "fusion_owner"
            else:
                # Fine-grained Model IR remains visible, but only the fusion
                # owner owns the measured production event(s).  Keeping the
                # relationship instead of copying the owner's scalar metrics
                # prevents one physical interval from looking like several
                # independently timed semantic operations.
                cell["timing_role"] = "fused_member"
                cell["shared_timing_owner"] = group["owner"]
        elif cell.get("attribution_status") == "inclusive_rollup":
            cell["timing_role"] = "inclusive_rollup"
        elif any(field in cell for field in ("ms_per_iter", "active_gpu_ms")):
            cell["timing_role"] = "standalone"
        data[target] = {variant: cell}
    meta = {
            key: copy.deepcopy(profile[key])
            for key in (
                "profile_id",
                "label",
                "phase",
                "generation_mode",
                "entry_view",
                "variant_id",
                "execution_parameters",
                "hardware",
                "workload",
                "profiler",
                "evidence",
                "profile_summary",
            )
            if key in profile
        }
    meta.setdefault("generation_mode", "autoregressive")
    meta.setdefault("entry_view", "top")
    if profile.get("timeline") is not None:
        timeline = copy.deepcopy(profile["timeline"])
        if not isinstance(timeline, dict):
            raise CatalogError(f"{source}: timeline must be a mapping")
        if timeline.get("schema_version") != "timeline.v1":
            raise CatalogError(
                f"{source}: timeline requires schema_version='timeline.v1'"
            )
        artifact = Path(str(timeline.get("artifact") or ""))
        if not artifact.name or artifact.name != str(artifact):
            raise CatalogError(
                f"{source}: timeline artifact must be a sibling file name"
            )
        sha256 = str(timeline.get("sha256") or "")
        if len(sha256) != 64 or any(char not in "0123456789abcdef" for char in sha256):
            raise CatalogError(f"{source}: timeline requires a lowercase SHA256")
        timeline["url"] = f"timelines/{profile['profile_id']}.timeline.json.gz"
        meta["timeline"] = timeline
    return {
        "meta": meta,
        "execution_variant": fingerprint,
        "execution_path_id": profile["execution_path_id"],
        "implementation_id": profile["implementation_id"],
        "fusion_groups": fusion_groups,
        "data": data,
    }


def build_enriched(
    views: dict[str, Any], profiles: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    enriched = {
        view_id: {"title": view.get("title", view_id), "nodes_profile": {}}
        for view_id, view in views.items()
    }
    for profile_id, profile in profiles.items():
        for target, variants in profile.get("data", {}).items():
            view_id, node_id = target.split(".", 1)
            enriched[view_id]["nodes_profile"].setdefault(node_id, {})[
                profile_id
            ] = copy.deepcopy(variants)
    return enriched


def _catalog_files(root: Path, pattern: str) -> list[Path]:
    return sorted(path for path in root.glob(pattern) if path.is_file())


def compile_catalog(model_root: Path) -> dict[str, Any]:
    model_root = model_root.resolve()
    model_path = model_root / "model_ir.yaml"
    model_ir = load_yaml(model_path)
    _validate_schema_version(model_ir, "model-ir.v2", source=model_path)
    _require(
        model_ir,
        ("model_id", "model_label", "ir_version", "default_view", "views"),
        source=model_path,
    )
    _node_index(model_ir["views"], source=model_path)
    _validate_semantic_coverage(model_ir, source=model_path)
    _validate_operator_granularity(model_ir, source=model_path)
    _validate_leaf_equation_coverage(model_ir, source=model_path)
    _validate_notation_contract(model_ir, source=model_path)
    _validate_boundary_contracts(model_ir, source=model_path)
    if model_ir["default_view"] not in model_ir["views"]:
        raise CatalogError(
            f"{model_path}: default_view {model_ir['default_view']!r} does not exist"
        )

    plan_paths = _catalog_files(model_root, "execution_paths/*.yaml")
    if not plan_paths:
        raise CatalogError(f"{model_root}: no execution plans found")

    binding_paths = _catalog_files(model_root, "bindings/*.yaml")
    profile_paths = _catalog_files(model_root, "profiles/*/*/*.yaml")
    sol_manifest_paths = _catalog_files(model_root, "sol_manifests/*.yaml")
    raw_bindings: list[tuple[Path, dict[str, Any]]] = []
    for path in binding_paths:
        binding = load_yaml(path)
        _validate_schema_version(binding, "implementation-binding.v2", source=path)
        _require(
            binding,
            (
                "implementation_id",
                "label",
                "model_id",
                "execution_path_id",
                "source_repo",
                "source_commit",
                "node_bindings",
            ),
            source=path,
        )
        if binding["model_id"] != model_ir["model_id"]:
            raise CatalogError(f"{path}: model_id does not match {model_ir['model_id']}")
        raw_bindings.append((path, binding))

    raw_profiles: list[tuple[Path, dict[str, Any]]] = []
    for path in profile_paths:
        profile = load_yaml(path)
        _validate_schema_version(profile, "profile.v2", source=path)
        _require(
            profile,
            (
                "profile_id",
                "label",
                "model_id",
                "execution_path_id",
                "implementation_id",
                "variant_id",
                "execution_parameters",
                "phase",
                "node_metrics",
            ),
            source=path,
        )
        if profile["model_id"] != model_ir["model_id"]:
            raise CatalogError(f"{path}: model_id does not match {model_ir['model_id']}")
        raw_profiles.append((path, profile))

    bindings_by_id: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path, binding in raw_bindings:
        implementation_id = binding["implementation_id"]
        if implementation_id in bindings_by_id:
            raise CatalogError(
                f"{path}: duplicate implementation {implementation_id!r}"
            )
        bindings_by_id[implementation_id] = (path, binding)

    def binding_lineage(
        implementation_id: str, trail: tuple[str, ...] = ()
    ) -> list[tuple[Path, dict[str, Any]]]:
        if implementation_id in trail:
            raise CatalogError(
                "implementation binding inheritance cycle: "
                + " -> ".join((*trail, implementation_id))
            )
        path, binding = bindings_by_id[implementation_id]
        base_id = binding.get("extends")
        if not base_id:
            return [(path, binding)]
        if base_id not in bindings_by_id:
            raise CatalogError(f"{path}: unknown base implementation {base_id!r}")
        base_path, base = bindings_by_id[base_id]
        for identity_key in ("model_id", "source_repo"):
            if str(binding.get(identity_key)) != str(base.get(identity_key)):
                raise CatalogError(
                    f"{path}: inherited {identity_key} must match {base_path}"
                )
        if str(binding.get("source_commit")) != str(base.get("source_commit")):
            compatible_base = binding.get("binding_compatible_base_commit")
            if str(compatible_base) != str(base.get("source_commit")):
                raise CatalogError(
                    f"{path}: inherited source_commit differs from {base_path}; "
                    "binding_compatible_base_commit must explicitly name the "
                    "base source commit"
                )
        return binding_lineage(base_id, (*trail, implementation_id)) + [
            (path, binding)
        ]

    execution_variants: dict[str, Any] = {}
    plans_by_id: dict[str, tuple[Path, dict[str, Any], str]] = {}
    for plan_path in plan_paths:
        plan = load_yaml(plan_path)
        _validate_schema_version(plan, "execution-plan.v2", source=plan_path)
        _require(
            plan,
            (
                "execution_path_id",
                "label",
                "model_id",
                "plan_version",
                "parallelism_axes",
                "transforms",
            ),
            source=plan_path,
        )
        if plan["model_id"] != model_ir["model_id"]:
            raise CatalogError(f"{plan_path}: model_id does not match {model_ir['model_id']}")
        if plan["execution_path_id"] in plans_by_id:
            raise CatalogError(
                f"{plan_path}: duplicate execution path {plan['execution_path_id']!r}"
            )
        views = apply_execution_plan(model_ir, plan, source=plan_path)
        fingerprint = execution_fingerprint(model_ir, plan, views)
        plans_by_id[plan["execution_path_id"]] = (plan_path, plan, fingerprint)
        execution_variants[fingerprint] = {
            "execution_path_id": plan["execution_path_id"],
            "label": plan["label"],
            "fingerprint": fingerprint,
            "model_ir_version": model_ir["ir_version"],
            "execution_plan_version": plan["plan_version"],
            "parallelism_axes": copy.deepcopy(plan.get("parallelism_axes", {})),
            "default_parameters": copy.deepcopy(plan.get("default_parameters", {})),
            "default_view": model_ir["default_view"],
            "views": views,
            "parent": derive_parent_map(views),
        }

    implementations: dict[str, Any] = {}
    for path, raw_binding in raw_bindings:
        path_id = raw_binding["execution_path_id"]
        if path_id not in plans_by_id:
            raise CatalogError(f"{path}: unknown execution_path_id {path_id!r}")
        _, _, fingerprint = plans_by_id[path_id]
        target_index = _node_index(
            execution_variants[fingerprint]["views"], source=path
        )
        unknown = sorted(set(raw_binding.get("node_bindings") or {}) - set(target_index))
        if unknown:
            raise CatalogError(f"{path}: binding references unknown nodes: {unknown}")
        merged_binding = copy.deepcopy(raw_binding)
        merged_nodes: dict[str, Any] = {}
        for _lineage_path, lineage_binding in binding_lineage(
            raw_binding["implementation_id"]
        ):
            merged_nodes.update(
                {
                    target: copy.deepcopy(node_binding)
                    for target, node_binding in (
                        lineage_binding.get("node_bindings") or {}
                    ).items()
                    if target in target_index
                }
            )
        merged_binding["node_bindings"] = merged_nodes
        compiled = compile_binding(merged_binding, source=path)
        compiled["execution_variant"] = fingerprint
        validation = compiled.get("execution_validation")
        if validation:
            validation_fingerprint = validation.get("execution_fingerprint")
            if validation_fingerprint != fingerprint:
                raise CatalogError(
                    f"{path}: execution_validation fingerprint "
                    f"{validation_fingerprint!r} does not match compiled "
                    f"Execution IR {fingerprint!r}"
                )
            if compiled.get("binding_status") == "validated" and validation.get(
                "status"
            ) != "pass":
                raise CatalogError(
                    f"{path}: binding_status='validated' requires "
                    "execution_validation.status='pass'"
                )
        implementations[compiled["implementation_id"]] = compiled

    profiles: dict[str, Any] = {}
    for path, raw_profile in raw_profiles:
        path_id = raw_profile["execution_path_id"]
        if path_id not in plans_by_id:
            raise CatalogError(f"{path}: unknown execution_path_id {path_id!r}")
        plan_path, plan, fingerprint = plans_by_id[path_id]
        impl_id = raw_profile["implementation_id"]
        if impl_id not in implementations:
            raise CatalogError(f"{path}: unknown implementation_id {impl_id!r}")
        if implementations[impl_id]["execution_variant"] != fingerprint:
            raise CatalogError(
                f"{path}: implementation {impl_id!r} belongs to another execution path"
            )
        evidence_commit = (raw_profile.get("evidence") or {}).get("source_commit")
        implementation_commit = implementations[impl_id].get("source_commit")
        if evidence_commit is not None and str(evidence_commit) != str(implementation_commit):
            raise CatalogError(
                f"{path}: evidence source_commit {evidence_commit!r} does not match "
                f"implementation {impl_id!r} commit {implementation_commit!r}"
            )
        implementation_patch = implementations[impl_id].get("source_patch_sha256")
        evidence_patch = (raw_profile.get("evidence") or {}).get("source_patch_sha256")
        if implementation_patch is not None and str(evidence_patch) != str(
            implementation_patch
        ):
            raise CatalogError(
                f"{path}: evidence source_patch_sha256 {evidence_patch!r} does not "
                f"match implementation {impl_id!r} patch {implementation_patch!r}"
            )
        target_index = _node_index(
            execution_variants[fingerprint]["views"], source=plan_path
        )
        targets = set(target_index)
        compiled_profile = compile_profile(
            raw_profile,
            plan=plan,
            fingerprint=fingerprint,
            node_targets=targets,
            node_index=target_index,
            source=path,
        )
        profile_id = raw_profile["profile_id"]
        if profile_id in profiles:
            raise CatalogError(f"{path}: duplicate profile_id {profile_id!r}")
        profiles[profile_id] = compiled_profile

    hardware_specs: dict[str, tuple[Path, dict[str, Any]]] = {}
    hardware_catalog_dir = model_root.parent / "hardware"
    for hardware_path in _catalog_files(model_root.parent, "hardware/*.yaml"):
        hardware = load_yaml(hardware_path)
        hardware_id = hardware.get("hardware_spec_id")
        if not hardware_id:
            raise CatalogError(f"{hardware_path}: missing hardware_spec_id")
        if hardware_id in hardware_specs:
            raise CatalogError(
                f"{hardware_path}: duplicate hardware_spec_id {hardware_id!r}"
            )
        hardware_specs[str(hardware_id)] = (hardware_path, hardware)

    sol_profiles: dict[str, Any] = {}
    gap_reports: dict[str, Any] = {}
    sol_diagnostics: list[dict[str, Any]] = []
    manifests_to_compile = sol_manifest_paths
    if sol_manifest_paths and not hardware_catalog_dir.is_dir():
        manifests_to_compile = []
        sol_diagnostics.append(
            {
                "status": "skipped",
                "reason": "missing_hardware_catalog",
                "manifest_count": len(sol_manifest_paths),
                "expected_directory": str(hardware_catalog_dir),
            }
        )
    for manifest_path in manifests_to_compile:
        manifest = load_yaml(manifest_path)
        hardware_id = str(manifest.get("hardware_spec_id") or "")
        if hardware_id not in hardware_specs:
            raise CatalogError(
                f"{manifest_path}: unknown hardware_spec_id {hardware_id!r}"
            )
        hardware_path, hardware = hardware_specs[hardware_id]
        try:
            sol_profile, gap_report = build_sol_artifacts(
                model_ir=model_ir,
                execution_variants=execution_variants,
                profiles=profiles,
                hardware=hardware,
                manifest=manifest,
                manifest_source=manifest_path,
                hardware_source=hardware_path,
            )
        except SolError as exc:
            raise CatalogError(str(exc)) from exc
        sol_profile_id = sol_profile["sol_profile_id"]
        gap_report_id = gap_report["gap_report_id"]
        if sol_profile_id in sol_profiles:
            raise CatalogError(
                f"{manifest_path}: duplicate sol_profile_id {sol_profile_id!r}"
            )
        if gap_report_id in gap_reports:
            raise CatalogError(
                f"{manifest_path}: duplicate gap_report_id {gap_report_id!r}"
            )
        sol_profiles[sol_profile_id] = sol_profile
        gap_reports[gap_report_id] = gap_report
        attach_sol_to_profile(
            profiles[sol_profile["measured_profile_id"]], sol_profile, gap_report
        )

    for fingerprint, variant in execution_variants.items():
        compatible_profiles = {
            profile_id: profile
            for profile_id, profile in profiles.items()
            if profile["execution_variant"] == fingerprint
        }
        variant["enriched"] = build_enriched(variant["views"], compatible_profiles)

    default_path_id = model_ir.get("default_execution_path") or next(iter(plans_by_id))
    if default_path_id not in plans_by_id:
        raise CatalogError(
            f"{model_path}: unknown default_execution_path {default_path_id!r}"
        )
    default_fingerprint = plans_by_id[default_path_id][2]
    default_impl = next(
        (
            impl_id
            for impl_id, impl in implementations.items()
            if impl["execution_variant"] == default_fingerprint
        ),
        "",
    )
    default_profile = next(
        (
            profile_id
            for profile_id, profile in profiles.items()
            if profile["execution_variant"] == default_fingerprint
            and (not default_impl or profile["implementation_id"] == default_impl)
        ),
        "",
    )

    default_variant = execution_variants[default_fingerprint]
    projected_views = copy.deepcopy(default_variant["views"])
    if default_impl:
        apply_binding(
            projected_views,
            implementations[default_impl],
            source=model_root / "bindings",
        )

    bundle = {
        "schema_version": "2.0",
        "default_view": model_ir["default_view"],
        "default_execution_variant": default_fingerprint,
        "default_implementation": default_impl,
        "default_profile": default_profile,
        "meta": {
            "model_id": model_ir["model_id"],
            "model_label": model_ir["model_label"],
            "subtitle": "IR-first · execution-path variants · versioned profile overlays",
            "model_ir_version": model_ir["ir_version"],
            "model_semantic_revision": model_ir.get("semantic_revision"),
            "catalog": f"catalog/{model_root.name}",
            "execution_variant_count": len(execution_variants),
            "implementation_count": len(implementations),
            "profile_count": len(profiles),
            "sol_profile_count": len(sol_profiles),
            "gap_report_count": len(gap_reports),
            "view_count": len(default_variant["views"]),
        },
        "model_ir": {
            **{
                key: copy.deepcopy(model_ir[key])
                for key in (
                    "model_id",
                    "model_label",
                    "ir_version",
                    "semantic_revision",
                    "semantic_evidence",
                    "semantic_coverage",
                    "dimensions",
                    "facts",
                    "default_view",
                )
                if key in model_ir
            },
            "semantic_contract": copy.deepcopy(model_ir.get("semantic_contract", {})),
            "boundary_contracts": copy.deepcopy(model_ir.get("boundary_contracts", [])),
            "views": _model_views_with_provenance(
                model_ir["views"], model_ir=model_ir, source=model_path
            ),
            "parent": derive_parent_map(model_ir["views"]),
        },
        "execution_variants": execution_variants,
        "implementations": implementations,
        "profiles": profiles,
        "sol_profiles": sol_profiles,
        "gap_reports": gap_reports,
        "sol_diagnostics": sol_diagnostics,
        # Compatibility projection for the existing viewer and downstream readers.
        "views": projected_views,
        "parent": copy.deepcopy(default_variant["parent"]),
        "enriched": copy.deepcopy(default_variant["enriched"]),
        "stages": {},
        "configs": {},
    }
    return bundle


def write_bundle(bundle: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(bundle, indent=2, sort_keys=False) + "\n")
