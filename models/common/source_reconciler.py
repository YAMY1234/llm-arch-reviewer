#!/usr/bin/env python3
"""Build a source-reconciled architecture draft from a runtime skeleton.

The runtime skeleton records what the profiler trace proves. This reconciler
adds source-code and config-backed semantic nodes on top of that evidence. It
intentionally writes a draft artifact instead of mutating the hand-reviewed IR.
"""

from __future__ import annotations

import ast
import copy
import json
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised manually
    raise SystemExit("requires pyyaml") from exc


def load_yaml(path: Path) -> Any:
    with path.open() as fh:
        return yaml.safe_load(fh)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")


def write_yaml(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True)


def resolve_path(raw: str | Path | None, *, base: Path) -> Path | None:
    if raw is None:
        return None
    path = Path(raw)
    return path if path.is_absolute() else (base / path).resolve()


def _display_source_path(file_path: str) -> str:
    prefix = "python/sglang/srt/"
    if file_path.startswith(prefix):
        return file_path[len(prefix) :]
    return file_path


def _source_link_to_code_link(link: dict[str, Any]) -> str:
    file_path = link["file"]
    display_file = link.get("display_file") or _display_source_path(file_path)
    line = int(link["line"])
    line_end = int(link.get("line_end") or line)
    span = str(line) if line_end == line else f"{line}-{line_end}"
    label = link.get("label")
    return f"{display_file}:{span} {label}" if label else f"{display_file}:{span}"


def _read_line_span(path: Path, line: int, line_end: int) -> str:
    lines = path.read_text().splitlines()
    if line < 1 or line > len(lines):
        return ""
    end = min(max(line_end, line), len(lines))
    return "\n".join(lines[line - 1 : end])


@lru_cache(maxsize=128)
def _parse_python_file(path_str: str) -> ast.AST:
    path = Path(path_str)
    return ast.parse(path.read_text(), filename=str(path))


def _node_end_lineno(node: ast.AST) -> int:
    return int(getattr(node, "end_lineno", getattr(node, "lineno", 0)) or 0)


def _node_contains_line(node: ast.AST, line: int) -> bool:
    return int(getattr(node, "lineno", 0) or 0) <= line <= _node_end_lineno(node)


def _find_class(tree: ast.AST, class_name: str) -> ast.ClassDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    return None


def _find_function(
    parent: ast.AST,
    function_name: str,
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for child in getattr(parent, "body", []) or []:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == function_name:
            return child
    return None


def _expr_to_dotted(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _expr_to_dotted(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Call):
        return _expr_to_dotted(node.func)
    if isinstance(node, ast.Subscript):
        return _expr_to_dotted(node.value)
    return None


def _calls_at_line(scope: ast.AST, line: int) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(scope)
        if isinstance(node, ast.Call) and _node_contains_line(node, line)
    ]


def _targets_at_line(scope: ast.AST, line: int) -> list[ast.AST]:
    targets: list[ast.AST] = []
    for node in ast.walk(scope):
        if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            continue
        if not _node_contains_line(node, line):
            continue
        if isinstance(node, ast.Assign):
            targets.extend(node.targets)
        else:
            targets.append(node.target)
    return targets


def _scope_for_identity(
    tree: ast.AST,
    identity: dict[str, Any],
) -> tuple[ast.AST | None, list[str]]:
    errors: list[str] = []
    class_name = identity.get("class")
    function_name = identity.get("function")
    scope: ast.AST = tree

    if class_name:
        class_scope = _find_class(tree, str(class_name))
        if class_scope is None:
            errors.append(f"class {class_name!r} not found")
            return None, errors
        scope = class_scope

    if function_name:
        function_scope = _find_function(scope, str(function_name))
        if function_scope is None:
            errors.append(
                f"function {function_name!r} not found"
                + (f" in class {class_name!r}" if class_name else "")
            )
            return None, errors
        scope = function_scope

    return scope, errors


def _derive_canonical_from_callee(
    *,
    class_name: str | None,
    function_name: str | None,
    callee: str,
) -> str:
    if callee.startswith("self.") and class_name:
        return f"{class_name}.{callee[len('self.'):]}"
    if class_name and function_name:
        return f"{class_name}.{function_name}::{callee}"
    if class_name:
        return f"{class_name}::{callee}"
    return callee


def _derive_canonical_from_target(
    *,
    class_name: str | None,
    target: str,
) -> str:
    if target.startswith("self.") and class_name:
        return f"{class_name}.{target[len('self.'):]}"
    if class_name:
        return f"{class_name}::{target}"
    return target


def validate_source_identity(
    identity: dict[str, Any],
    *,
    source_root: Path,
) -> tuple[dict[str, Any], list[str]]:
    """Validate that a canonical source id is derivable from an AST callsite.

    Supported kinds:
    - `self_attr_call`: line contains a call whose callee is `self.<attr>[.<sub>]`.
      Canonical id is `<Class>.<attr>[.<sub>]`.
    - `function_call`: line contains the supplied callee. Canonical id is
      `<Class>.<function>::<callee>` when class/function are supplied.
    - `self_attr_def`: line contains assignment target `self.<attr>`.
    - `method_def`: class method exists and contains the given line.
    """

    errors: list[str] = []
    kind = str(identity.get("kind") or "function_call")
    canonical = str(identity.get("canonical_source_id") or "")
    file_path = str(identity["file"])
    line = int(identity["line"])
    full_path = source_root / file_path
    evidence = {
        "canonical_source_id": canonical,
        "kind": kind,
        "file": file_path,
        "line": line,
        "code_link": _source_link_to_code_link(identity),
        "class": identity.get("class"),
        "function": identity.get("function"),
        "callee": identity.get("callee"),
        "target": identity.get("target"),
        "derived_canonical_source_id": None,
        "ok": True,
    }
    if not canonical:
        evidence["ok"] = False
        return evidence, ["missing canonical_source_id"]
    if not full_path.exists():
        evidence["ok"] = False
        return evidence, [f"missing source file: {file_path}"]

    try:
        tree = _parse_python_file(str(full_path))
    except SyntaxError as exc:
        evidence["ok"] = False
        return evidence, [f"failed to parse {file_path}: {exc}"]

    scope, scope_errors = _scope_for_identity(tree, identity)
    errors.extend(scope_errors)
    if scope is None:
        evidence["ok"] = False
        return evidence, errors

    if not _node_contains_line(scope, line):
        errors.append(
            f"line {line} is outside declared scope "
            f"{identity.get('class') or '<module>'}.{identity.get('function') or '<body>'}"
        )

    class_name = str(identity.get("class") or "") or None
    function_name = str(identity.get("function") or "") or None
    if kind in {"self_attr_call", "function_call"}:
        expected_callee = str(identity.get("callee") or "")
        if not expected_callee:
            errors.append("missing callee for call identity")
        callees = [_expr_to_dotted(call.func) for call in _calls_at_line(scope, line)]
        callees = [callee for callee in callees if callee]
        if expected_callee and expected_callee not in callees:
            errors.append(
                f"line {file_path}:{line} callsite callees {callees} "
                f"do not include {expected_callee!r}"
            )
        if kind == "self_attr_call" and not expected_callee.startswith("self."):
            errors.append(f"self_attr_call callee must start with 'self.': {expected_callee}")
        if expected_callee:
            evidence["derived_canonical_source_id"] = _derive_canonical_from_callee(
                class_name=class_name,
                function_name=function_name,
                callee=expected_callee,
            )
    elif kind == "self_attr_def":
        expected_target = str(identity.get("target") or "")
        if not expected_target:
            errors.append("missing target for definition identity")
        targets = [_expr_to_dotted(target) for target in _targets_at_line(scope, line)]
        targets = [target for target in targets if target]
        if expected_target and expected_target not in targets:
            errors.append(
                f"line {file_path}:{line} assignment targets {targets} "
                f"do not include {expected_target!r}"
            )
        if expected_target and not expected_target.startswith("self."):
            errors.append(f"self_attr_def target must start with 'self.': {expected_target}")
        if expected_target:
            evidence["derived_canonical_source_id"] = _derive_canonical_from_target(
                class_name=class_name,
                target=expected_target,
            )
    elif kind == "method_def":
        if not class_name or not function_name:
            errors.append("method_def requires class and function")
        else:
            evidence["derived_canonical_source_id"] = f"{class_name}.{function_name}"
    else:
        errors.append(f"unsupported source identity kind: {kind}")

    if evidence["derived_canonical_source_id"] != canonical:
        errors.append(
            f"canonical_source_id mismatch: expected {canonical!r}, "
            f"derived {evidence['derived_canonical_source_id']!r}"
        )

    evidence["ok"] = not errors
    return evidence, errors


def validate_source_link(
    link: dict[str, Any],
    *,
    source_root: Path,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    file_path = str(link["file"])
    line = int(link["line"])
    line_end = int(link.get("line_end") or line)
    full_path = source_root / file_path
    evidence = {
        "file": file_path,
        "line": line,
        "line_end": line_end,
        "code_link": _source_link_to_code_link(link),
        "contains": list(link.get("contains") or []),
        "ok": True,
    }
    if not full_path.exists():
        evidence["ok"] = False
        errors.append(f"missing source file: {file_path}")
        return evidence, errors

    span_text = _read_line_span(full_path, line, line_end)
    if not span_text:
        evidence["ok"] = False
        errors.append(f"empty source span: {file_path}:{line}-{line_end}")
        return evidence, errors

    for needle in evidence["contains"]:
        if needle not in span_text:
            evidence["ok"] = False
            errors.append(
                f"source span {file_path}:{line}-{line_end} does not contain {needle!r}"
            )
    return evidence, errors


def config_has_path(config: dict[str, Any], dotted_path: str) -> bool:
    cur: Any = config
    for part in dotted_path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
            continue
        if isinstance(cur, list):
            try:
                cur = cur[int(part)]
                continue
            except (ValueError, IndexError):
                return False
        return False
    return True


def collect_view_node_ids(
    ir: dict[str, Any],
    *,
    view_names: list[str] | None = None,
) -> dict[str, set[str]]:
    views = ir.get("views") or {}
    selected_views = view_names or sorted(views)
    out: dict[str, set[str]] = {}
    for view_name in selected_views:
        view = views.get(view_name) or {}
        out[view_name] = {
            node["id"]
            for node in view.get("nodes", []) or []
            if isinstance(node, dict) and node.get("id")
        }
    return out


def _flatten_ids(by_view: dict[str, set[str]]) -> set[str]:
    out: set[str] = set()
    for ids in by_view.values():
        out |= ids
    return out


def summarize_runtime_evidence(
    runtime_nodes: dict[str, Any],
    node_keys: list[str],
    *,
    timing_attribution: str,
) -> dict[str, Any]:
    total_ms = 0.0
    kernel_count = 0
    missing: list[str] = []
    top_kernels: Counter[str] = Counter()
    for key in node_keys:
        cell = runtime_nodes.get(key)
        if not cell:
            missing.append(key)
            continue
        total_ms += float(cell.get("total_kernel_ms") or 0.0)
        kernel_count += int(cell.get("kernel_count") or 0)
        for kernel in cell.get("top_kernels_by_duration") or []:
            top_kernels[kernel["value"]] += float(kernel.get("dur_ms") or 0.0)
    fine_node_ms = round(total_ms, 6) if timing_attribution == "direct" else None
    return {
        "runtime_nodes": node_keys,
        "timing_attribution": timing_attribution,
        "missing_runtime_nodes": missing,
        "bucket_total_kernel_ms": round(total_ms, 6),
        "fine_node_ms": fine_node_ms,
        "kernel_count": kernel_count,
        "top_kernels_by_duration": [
            {"value": name, "dur_ms": round(ms, 6)}
            for name, ms in top_kernels.most_common(8)
        ],
    }


def reconcile_architecture(
    *,
    skeleton_path: Path,
    config: dict[str, Any],
    config_base: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    skeleton = load_yaml(skeleton_path)
    source_root = resolve_path(config["source_root"], base=config_base)
    if source_root is None:
        raise ValueError("source_root is required")
    model_config_path = resolve_path(config.get("model_config"), base=config_base)
    model_config = load_yaml(model_config_path) if model_config_path else {}
    runtime_nodes = skeleton.get("runtime_nodes") or {}
    ignored_runtime_nodes = set(config.get("ignored_runtime_nodes") or [])

    errors: list[str] = []
    warnings: list[str] = []
    source_link_count = 0
    verified_source_link_count = 0
    source_identity_count = 0
    verified_source_identity_count = 0
    runtime_refs: Counter[str] = Counter()
    runtime_ref_nodes: dict[str, list[str]] = defaultdict(list)
    source_only_nodes: list[str] = []
    alias_nodes: list[dict[str, str]] = []

    draft_views: dict[str, Any] = {}
    for view_name, raw_view in (config.get("views") or {}).items():
        view = copy.deepcopy(raw_view)
        for node in view.get("nodes", []) or []:
            node_id = node.get("id")
            if not node_id:
                errors.append(f"view {view_name} has node without id")
                continue

            source_identity_inputs: list[dict[str, Any]] = []
            if node.get("source_identity"):
                source_identity_inputs.append(node.pop("source_identity"))
            source_identity_inputs.extend(node.pop("source_identities", []) or [])
            source_identity_evidence: list[dict[str, Any]] = []
            for identity in source_identity_inputs:
                source_identity_count += 1
                evidence, identity_errors = validate_source_identity(
                    identity, source_root=source_root
                )
                source_identity_evidence.append(evidence)
                errors.extend(f"node {node_id}: {msg}" for msg in identity_errors)
                if evidence["ok"]:
                    verified_source_identity_count += 1
            if source_identity_evidence:
                canonical_ids = [
                    item["canonical_source_id"]
                    for item in source_identity_evidence
                    if item.get("canonical_source_id")
                ]
                node["source_identity"] = {
                    "display_id": node_id,
                    "canonical_source_ids": canonical_ids,
                    "alias_role": "viewer/profile alias",
                }
                node["source_identity_evidence"] = source_identity_evidence
                for canonical_id in canonical_ids:
                    if canonical_id != node_id:
                        alias_nodes.append(
                            {"display_id": node_id, "canonical_source_id": canonical_id}
                        )

            source_links = node.pop("source_links", []) or []
            source_evidence: list[dict[str, Any]] = []
            code_links = list(node.get("code_links") or [])
            for link in source_links:
                source_link_count += 1
                evidence, link_errors = validate_source_link(link, source_root=source_root)
                source_evidence.append(evidence)
                code_links.append(evidence["code_link"])
                errors.extend(f"node {node_id}: {msg}" for msg in link_errors)
                if evidence["ok"]:
                    verified_source_link_count += 1
            if source_evidence:
                node["source_evidence"] = source_evidence
                node["code_links"] = code_links

            runtime_keys = list(node.get("runtime_nodes") or [])
            if runtime_keys:
                evidence = summarize_runtime_evidence(
                    runtime_nodes,
                    runtime_keys,
                    timing_attribution=node.get("runtime_evidence_mode") or "direct",
                )
                node["runtime_evidence"] = evidence
                for key in runtime_keys:
                    runtime_refs[key] += 1
                    runtime_ref_nodes[key].append(node_id)
                for key in evidence["missing_runtime_nodes"]:
                    errors.append(f"node {node_id}: runtime node {key!r} not found")
            elif source_evidence:
                source_only_nodes.append(node_id)

            for config_ref in node.get("config_refs") or []:
                if not config_has_path(model_config, str(config_ref)):
                    errors.append(f"node {node_id}: missing config ref {config_ref!r}")

            if "provenance" not in node:
                if runtime_keys and (source_evidence or source_identity_evidence):
                    arch_provenance = "runtime-observed + source-reconciled"
                elif runtime_keys:
                    arch_provenance = "runtime-observed"
                elif source_evidence or source_identity_evidence:
                    arch_provenance = "source-only"
                else:
                    arch_provenance = "draft"
                node["provenance"] = {
                    "architecture": arch_provenance,
                    "shape": "config/source formula" if node.get("shape_formula") else "unannotated",
                    "profile": node.get("runtime_evidence_mode") or "not-attributed",
                }
        draft_views[view_name] = view

    unresolved_runtime_nodes = sorted(
        set(runtime_nodes) - set(runtime_refs) - ignored_runtime_nodes
    )
    shared_runtime_evidence = {
        key: ids
        for key, ids in sorted(runtime_ref_nodes.items())
        if len(set(ids)) > 1
    }
    if shared_runtime_evidence:
        for view in draft_views.values():
            for node in view.get("nodes", []) or []:
                evidence = node.get("runtime_evidence")
                if not evidence:
                    continue
                if any(key in shared_runtime_evidence for key in evidence["runtime_nodes"]):
                    evidence["timing_attribution"] = "shared_parent_bucket"
                    evidence["fine_node_ms"] = None
    if shared_runtime_evidence:
        warnings.append(
            "some runtime buckets are shared parent evidence for multiple source-level nodes"
        )
    if unresolved_runtime_nodes:
        warnings.append(f"runtime nodes not referenced by draft: {unresolved_runtime_nodes}")

    draft = {
        "schema_version": "arch_draft.v0",
        "model_id": config.get("model_id"),
        "source": {
            "runtime_skeleton": str(skeleton_path),
            "source_root": str(source_root),
            "source_repo": skeleton.get("source", {}).get("source_repo"),
            "source_commit": skeleton.get("source", {}).get("source_commit"),
            "phase": skeleton.get("source", {}).get("phase"),
            "rank": skeleton.get("source", {}).get("rank"),
            "model_config": str(model_config_path) if model_config_path else None,
        },
        "generation_policy": config.get("generation_policy") or {},
        "views": draft_views,
    }

    draft_node_ids_by_view = collect_view_node_ids(draft)
    hand_ir_path = resolve_path(config.get("hand_ir"), base=config_base)
    compare_views = list(config.get("hand_ir_compare_views") or draft_node_ids_by_view)
    if hand_ir_path and hand_ir_path.exists():
        hand_ir = load_yaml(hand_ir_path)
        manual_node_ids_by_view = collect_view_node_ids(hand_ir, view_names=compare_views)
    else:
        manual_node_ids_by_view = {}
        if hand_ir_path:
            warnings.append(f"hand IR not found: {hand_ir_path}")

    generated_ids = _flatten_ids(
        {
            view: ids
            for view, ids in draft_node_ids_by_view.items()
            if view in set(compare_views)
        }
    )
    manual_ids = _flatten_ids(manual_node_ids_by_view)
    diff = {
        "hand_ir": str(hand_ir_path) if hand_ir_path else None,
        "compare_views": compare_views,
        "generated_node_count": len(generated_ids),
        "manual_node_count": len(manual_ids),
        "matched_node_count": len(generated_ids & manual_ids),
        "generated_only": sorted(generated_ids - manual_ids),
        "manual_only": sorted(manual_ids - generated_ids),
        "matched": sorted(generated_ids & manual_ids),
    }

    report = {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "views": len(draft_views),
            "nodes": sum(len(view.get("nodes") or []) for view in draft_views.values()),
            "runtime_nodes_observed": len(runtime_nodes),
            "runtime_nodes_referenced": len(runtime_refs),
            "source_links": source_link_count,
            "verified_source_links": verified_source_link_count,
            "source_identities": source_identity_count,
            "verified_source_identities": verified_source_identity_count,
            "source_only_nodes": len(source_only_nodes),
            "alias_nodes": len(alias_nodes),
            "unresolved_runtime_nodes": len(unresolved_runtime_nodes),
            "shared_runtime_buckets": len(shared_runtime_evidence),
        },
        "source_only_nodes": source_only_nodes,
        "alias_nodes": alias_nodes,
        "unresolved_runtime_nodes": unresolved_runtime_nodes,
        "shared_runtime_evidence": shared_runtime_evidence,
        "diff": diff,
    }
    draft["validation"] = {
        "ok": report["ok"],
        "errors": errors,
        "warnings": warnings,
    }
    return draft, report, diff


def write_reconcile_report_markdown(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = report["summary"]
    lines = [
        "# Source Reconciliation Report",
        "",
        f"- ok: `{report['ok']}`",
        f"- views: `{summary['views']}`",
        f"- nodes: `{summary['nodes']}`",
        f"- runtime_nodes_observed: `{summary['runtime_nodes_observed']}`",
        f"- runtime_nodes_referenced: `{summary['runtime_nodes_referenced']}`",
        f"- source_identities: `{summary['verified_source_identities']}/{summary['source_identities']}` verified",
        f"- source_links: `{summary['verified_source_links']}/{summary['source_links']}` verified",
        f"- source_only_nodes: `{summary['source_only_nodes']}`",
        f"- alias_nodes: `{summary['alias_nodes']}`",
        f"- unresolved_runtime_nodes: `{summary['unresolved_runtime_nodes']}`",
        f"- shared_runtime_buckets: `{summary['shared_runtime_buckets']}`",
        "",
        "## Errors",
        "",
    ]
    lines.extend([f"- {msg}" for msg in report["errors"]] or ["- none"])
    lines.extend(["", "## Warnings", ""])
    lines.extend([f"- {msg}" for msg in report["warnings"]] or ["- none"])
    lines.extend(["", "## Source-Only Nodes", ""])
    lines.extend([f"- `{node}`" for node in report["source_only_nodes"]] or ["- none"])
    lines.extend(["", "## Display Alias Nodes", ""])
    if report["alias_nodes"]:
        for item in report["alias_nodes"]:
            lines.append(
                f"- `{item['display_id']}` aliases `{item['canonical_source_id']}`"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Unresolved Runtime Nodes", ""])
    lines.extend([f"- `{node}`" for node in report["unresolved_runtime_nodes"]] or ["- none"])
    lines.extend(["", "## Shared Runtime Evidence", ""])
    if report["shared_runtime_evidence"]:
        for runtime_node, node_ids in report["shared_runtime_evidence"].items():
            joined = ", ".join(f"`{node_id}`" for node_id in node_ids)
            lines.append(f"- `{runtime_node}` -> {joined}")
    else:
        lines.append("- none")
    path.write_text("\n".join(lines) + "\n")


def write_diff_report_markdown(path: Path, diff: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Architecture Draft Diff Report",
        "",
        f"- hand_ir: `{diff.get('hand_ir')}`",
        f"- compare_views: `{diff.get('compare_views')}`",
        f"- generated_node_count: `{diff['generated_node_count']}`",
        f"- manual_node_count: `{diff['manual_node_count']}`",
        f"- matched_node_count: `{diff['matched_node_count']}`",
        "",
        "## Generated Only",
        "",
    ]
    lines.extend([f"- `{node}`" for node in diff["generated_only"]] or ["- none"])
    lines.extend(["", "## Manual Only", ""])
    lines.extend([f"- `{node}`" for node in diff["manual_only"]] or ["- none"])
    lines.extend(["", "## Matched", ""])
    lines.extend([f"- `{node}`" for node in diff["matched"]] or ["- none"])
    path.write_text("\n".join(lines) + "\n")
