"""Fail-closed source-to-Model-IR semantic closure audit.

The normal compiler can prove that catalog documents are internally
consistent.  It cannot prove that an author enumerated every stable operation
present in the pinned construction source.  This module adds that external
obligation ledger and validates it in both directions:

* source obligation -> reviewed Model IR target or explicit exclusion;
* Model IR leaf -> pinned source obligation or explicit reverse exclusion.

An incomplete audit is still a useful artifact, but it never reports PASS.
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


OBLIGATION_KINDS = {
    "model_ir_primitive",
    "tensor_boundary",
    "state_read",
    "state_update",
    "execution_ir",
    "implementation_only",
    "shape_only_omitted",
    "training_only_excluded",
}
MAPPED_KINDS = {
    "model_ir_primitive",
    "tensor_boundary",
    "state_read",
    "state_update",
}


class SemanticAuditError(ValueError):
    """Raised when the ledger itself is malformed or cannot be verified."""


@dataclass(frozen=True)
class SourceFile:
    path: str
    expected_oid: str
    actual_oid: str | None
    status: str


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise SemanticAuditError(f"{path}: expected a YAML mapping")
    return value


def _git(repo: Path, *args: str) -> str:
    process = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.returncode:
        detail = process.stderr.strip() or process.stdout.strip()
        raise SemanticAuditError(f"git {' '.join(args)} failed: {detail}")
    return process.stdout


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _node_index(model_ir: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        f"{view_id}.{node['id']}": node
        for view_id, view in (model_ir.get("views") or {}).items()
        for node in (view.get("nodes") or [])
    }


def _audited_leaves(
    model_ir: dict[str, Any], audit_views: list[str]
) -> tuple[set[str], list[str]]:
    leaves: set[str] = set()
    errors: list[str] = []
    views = model_ir.get("views") or {}
    for view_id in audit_views:
        view = views.get(view_id)
        if not isinstance(view, dict):
            errors.append(f"audit view {view_id!r} does not exist in Model IR")
            continue
        for node in view.get("nodes") or []:
            if not node.get("drill"):
                leaves.add(f"{view_id}.{node['id']}")
    return leaves, errors


def _symbol_members(source: str, symbol: str) -> list[str]:
    """Return the concrete source members covered by a ledger entrypoint.

    A class entrypoint must classify every locally declared method.  A
    ``Class.method`` entrypoint covers only that method, while a top-level
    function covers itself.  This turns a forgotten helper/forward variant into
    a visible gap instead of trusting a hand-written class label.
    """

    tree = ast.parse(source)
    parts = symbol.split(".")
    for item in tree.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == parts[0]:
            return [item.name] if len(parts) == 1 else []
        if isinstance(item, ast.ClassDef) and item.name == parts[0]:
            methods = [
                child.name
                for child in item.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            if len(parts) == 1:
                return methods
            return [parts[1]] if parts[1] in methods else []
    return []


def audit_semantic_closure(
    *, model_ir_path: Path, ledger_path: Path, source_repo: Path
) -> dict[str, Any]:
    model_ir = _load_yaml(model_ir_path)
    ledger = _load_yaml(ledger_path)
    errors: list[str] = []

    if ledger.get("schema_version") != "semantic-source-ledger.v1":
        raise SemanticAuditError(
            f"{ledger_path}: expected schema_version='semantic-source-ledger.v1'"
        )
    if model_ir.get("model_id") != ledger.get("model_id"):
        raise SemanticAuditError(
            f"model mismatch: {model_ir.get('model_id')!r} != {ledger.get('model_id')!r}"
        )

    snapshot = ledger.get("source_snapshot") or {}
    revision = str(snapshot.get("revision") or "")
    if not revision:
        raise SemanticAuditError(f"{ledger_path}: source_snapshot.revision is required")

    source_files: list[SourceFile] = []
    source_text: dict[str, str] = {}
    declared_files: set[str] = set()
    for item in snapshot.get("files") or []:
        path = str(item.get("path") or "")
        expected_oid = str(item.get("git_blob_oid") or "")
        if not path or not expected_oid:
            errors.append("source snapshot file requires path and git_blob_oid")
            continue
        if path in declared_files:
            errors.append(f"duplicate source snapshot file: {path}")
            continue
        declared_files.add(path)
        try:
            actual_oid = _git(source_repo, "rev-parse", f"{revision}:{path}").strip()
            text = _git(source_repo, "show", f"{revision}:{path}")
        except SemanticAuditError as exc:
            source_files.append(SourceFile(path, expected_oid, None, "unavailable"))
            errors.append(str(exc))
            continue
        status = "verified" if actual_oid == expected_oid else "digest_mismatch"
        if status != "verified":
            errors.append(
                f"source digest mismatch for {path}: {actual_oid} != {expected_oid}"
            )
        source_files.append(SourceFile(path, expected_oid, actual_oid, status))
        source_text[path] = text

    node_index = _node_index(model_ir)
    audit_views = [str(item) for item in ledger.get("audit_views") or []]
    audited_leaves, view_errors = _audited_leaves(model_ir, audit_views)
    errors.extend(view_errors)

    entrypoint_ids: set[str] = set()
    obligation_ids: set[str] = set()
    mapped_targets: set[str] = set()
    pending_entrypoints: list[dict[str, Any]] = []
    pending_obligations: list[dict[str, Any]] = []
    unclassified_source_members: list[dict[str, Any]] = []
    obligation_rows: list[dict[str, Any]] = []
    primitive_owners: dict[str, list[str]] = {}

    for entrypoint in ledger.get("entrypoints") or []:
        entrypoint_id = str(entrypoint.get("id") or "")
        if not entrypoint_id:
            errors.append("entrypoint without id")
            continue
        if entrypoint_id in entrypoint_ids:
            errors.append(f"duplicate entrypoint id: {entrypoint_id}")
        entrypoint_ids.add(entrypoint_id)

        source_file = str(entrypoint.get("source_file") or "")
        anchor = str(entrypoint.get("anchor") or "")
        if source_file not in declared_files:
            errors.append(
                f"entrypoint {entrypoint_id} uses undeclared source file {source_file!r}"
            )
        text = source_text.get(source_file)
        if text is not None:
            occurrences = text.count(anchor)
            if occurrences != 1:
                errors.append(
                    f"entrypoint {entrypoint_id} anchor must occur exactly once; "
                    f"found {occurrences}: {anchor!r}"
                )
        symbol = str(entrypoint.get("symbol") or "")
        discovered_members = _symbol_members(text, symbol) if text is not None else []
        member_rows = entrypoint.get("member_dispositions") or []
        classified_members = {str(item.get("name") or "") for item in member_rows}
        unknown_members = sorted(classified_members - set(discovered_members))
        for member in unknown_members:
            errors.append(
                f"entrypoint {entrypoint_id} classifies unknown source member {member!r}"
            )
        missing_members = sorted(set(discovered_members) - classified_members)
        if missing_members:
            unclassified_source_members.append(
                {
                    "entrypoint": entrypoint_id,
                    "symbol": symbol,
                    "members": missing_members,
                }
            )
        for item in member_rows:
            disposition = item.get("disposition")
            if disposition not in {"reviewed", "pending", "excluded"}:
                errors.append(
                    f"entrypoint {entrypoint_id} member {item.get('name')!r} "
                    f"has invalid disposition {disposition!r}"
                )
            if disposition == "excluded" and not item.get("reason"):
                errors.append(
                    f"entrypoint {entrypoint_id} excluded member "
                    f"{item.get('name')!r} has no reason"
                )
        for view_id in entrypoint.get("expected_ir_views") or []:
            if view_id not in (model_ir.get("views") or {}):
                errors.append(
                    f"entrypoint {entrypoint_id} references unknown view {view_id!r}"
                )

        review_status = entrypoint.get("review_status")
        if review_status not in {"verified", "pending_review"}:
            errors.append(f"entrypoint {entrypoint_id} has invalid review_status")
        obligations = entrypoint.get("obligations") or []
        if review_status == "verified" and not obligations:
            errors.append(f"verified entrypoint {entrypoint_id} has no obligations")
        if review_status != "verified":
            pending_entrypoints.append(
                {
                    "id": entrypoint_id,
                    "component": entrypoint.get("component"),
                    "symbol": entrypoint.get("symbol"),
                    "expected_ir_views": entrypoint.get("expected_ir_views") or [],
                    "reason": entrypoint.get("pending_reason")
                    or "source obligations have not been reviewed",
                }
            )

        for obligation in obligations:
            obligation_id = str(obligation.get("id") or "")
            qualified_id = f"{entrypoint_id}.{obligation_id}"
            if not obligation_id:
                errors.append(f"entrypoint {entrypoint_id} has obligation without id")
                continue
            if qualified_id in obligation_ids:
                errors.append(f"duplicate obligation id: {qualified_id}")
            obligation_ids.add(qualified_id)
            kind = obligation.get("kind")
            disposition = obligation.get("disposition")
            if kind not in OBLIGATION_KINDS:
                errors.append(f"obligation {qualified_id} has invalid kind {kind!r}")
            if disposition not in {"mapped", "excluded", "pending"}:
                errors.append(
                    f"obligation {qualified_id} has invalid disposition {disposition!r}"
                )
            source_anchor = str(obligation.get("source_anchor") or "")
            if text is not None and source_anchor and source_anchor not in text:
                errors.append(
                    f"obligation {qualified_id} source anchor was not found: "
                    f"{source_anchor!r}"
                )
            targets = [str(item) for item in obligation.get("ir_targets") or []]
            if disposition == "mapped" and kind in MAPPED_KINDS and not targets:
                errors.append(f"mapped obligation {qualified_id} has no ir_targets")
            for target in targets:
                if target not in node_index:
                    errors.append(
                        f"obligation {qualified_id} references unknown IR target {target}"
                    )
                else:
                    mapped_targets.add(target)
                    if kind == "model_ir_primitive" and disposition == "mapped":
                        primitive_owners.setdefault(target, []).append(qualified_id)
            if disposition == "excluded" and not obligation.get("reason"):
                errors.append(f"excluded obligation {qualified_id} has no reason")
            if disposition == "pending":
                pending_obligations.append(
                    {
                        "id": qualified_id,
                        "kind": kind,
                        "source_anchor": source_anchor,
                        "candidate_targets": targets,
                    }
                )
            obligation_rows.append(
                {
                    "id": qualified_id,
                    "kind": kind,
                    "disposition": disposition,
                    "ir_targets": targets,
                }
            )

    reverse_exclusions: dict[str, str] = {}
    for item in ledger.get("reverse_exclusions") or []:
        target = str(item.get("target") or "")
        reason = str(item.get("reason") or "")
        if target not in node_index:
            errors.append(f"reverse exclusion references unknown IR target {target}")
        if not reason:
            errors.append(f"reverse exclusion {target} has no reason")
        reverse_exclusions[target] = reason

    uncovered_leaves = sorted(audited_leaves - mapped_targets - set(reverse_exclusions))
    compound_target_collisions = {
        target: owners
        for target, owners in sorted(primitive_owners.items())
        if len(owners) > 1
    }
    pending_source_count = (
        len(pending_entrypoints)
        + len(pending_obligations)
        + len(unclassified_source_members)
    )
    source_to_ir_complete = (
        pending_source_count == 0 and not compound_target_collisions and not errors
    )
    ir_to_source_complete = (
        not uncovered_leaves and not compound_target_collisions and not errors
    )
    complete = source_to_ir_complete and ir_to_source_complete
    claimed_closure = str(
        (model_ir.get("semantic_coverage") or {}).get("operator_dataflow_closure")
        or ""
    )
    attestation_contradiction = claimed_closure.startswith("complete") and not complete

    fingerprint_input = {
        "model_ir_sha256": _sha256(model_ir_path),
        "ledger_sha256": _sha256(ledger_path),
        "source_files": [item.__dict__ for item in source_files],
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_input, sort_keys=True).encode()
    ).hexdigest()[:20]

    return {
        "schema_version": "semantic-closure-report.v1",
        "model_id": model_ir["model_id"],
        "scope_id": ledger["scope_id"],
        "status": "complete" if complete else "incomplete",
        "audit_fingerprint": fingerprint,
        "inputs": fingerprint_input,
        "gates": {
            "source_snapshot_integrity": all(
                item.status == "verified" for item in source_files
            ),
            "source_to_ir_closure": source_to_ir_complete,
            "ir_to_source_closure": ir_to_source_complete,
            "ledger_integrity": not errors,
            "catalog_attestation_honest": not attestation_contradiction,
        },
        "counts": {
            "source_files": len(source_files),
            "entrypoints": len(entrypoint_ids),
            "verified_entrypoints": len(entrypoint_ids) - len(pending_entrypoints),
            "pending_entrypoints": len(pending_entrypoints),
            "obligations": len(obligation_rows),
            "pending_obligations": len(pending_obligations),
            "unclassified_source_members": sum(
                len(item["members"]) for item in unclassified_source_members
            ),
            "audited_model_ir_leaves": len(audited_leaves),
            "mapped_model_ir_targets": len(mapped_targets & audited_leaves),
            "reverse_exclusions": len(set(reverse_exclusions) & audited_leaves),
            "uncovered_model_ir_leaves": len(uncovered_leaves),
            "compound_primitive_targets": len(compound_target_collisions),
        },
        "source_files": [item.__dict__ for item in source_files],
        "catalog_attestation": {
            "claimed_operator_dataflow_closure": claimed_closure,
            "contradiction": attestation_contradiction,
        },
        "pending_entrypoints": pending_entrypoints,
        "pending_obligations": pending_obligations,
        "unclassified_source_members": unclassified_source_members,
        "uncovered_model_ir_leaves": uncovered_leaves,
        "compound_primitive_targets": compound_target_collisions,
        "reverse_exclusions": reverse_exclusions,
        "obligations": obligation_rows,
        "errors": errors,
    }


def render_markdown(report: dict[str, Any]) -> str:
    counts = report["counts"]
    lines = [
        f"# {report['model_id']} semantic closure gap report",
        "",
        f"- Status: **{report['status'].upper()}**",
        f"- Scope: `{report['scope_id']}`",
        f"- Audit fingerprint: `{report['audit_fingerprint']}`",
        "- This is a fail-closed report: an incomplete ledger is not a semantic PASS.",
        "",
        "## Gate summary",
        "",
        "| Gate | Result |",
        "|---|---|",
    ]
    for gate, value in report["gates"].items():
        lines.append(f"| `{gate}` | {'PASS' if value else 'FAIL'} |")
    lines.extend(
        [
            "",
            "## Coverage",
            "",
            f"- Source files: {counts['source_files']}",
            f"- Entrypoints: {counts['verified_entrypoints']} verified / "
            f"{counts['pending_entrypoints']} pending",
            f"- Source obligations: {counts['obligations']} total / "
            f"{counts['pending_obligations']} pending",
            f"- Unclassified source members: {counts['unclassified_source_members']}",
            f"- Audited Model IR leaves: {counts['audited_model_ir_leaves']}",
            f"- Reverse mapped leaves: {counts['mapped_model_ir_targets']}",
            f"- Explicit reverse exclusions: {counts['reverse_exclusions']}",
            f"- Uncovered Model IR leaves: {counts['uncovered_model_ir_leaves']}",
            f"- Compound primitive targets: {counts['compound_primitive_targets']}",
            "",
            "## Pending source entrypoints",
            "",
        ]
    )
    if report["pending_entrypoints"]:
        lines.extend(
            "- `{id}` ({component}) → {views}: {reason}".format(
                id=item["id"],
                component=item.get("component") or "unknown",
                views=", ".join(f"`{view}`" for view in item["expected_ir_views"]),
                reason=item["reason"],
            )
            for item in report["pending_entrypoints"]
        )
    else:
        lines.append("- None")
    lines.extend(["", "## Unclassified source members", ""])
    if report["unclassified_source_members"]:
        for item in report["unclassified_source_members"]:
            members = ", ".join(f"`{member}`" for member in item["members"])
            lines.append(
                f"- `{item['entrypoint']}` / `{item['symbol']}`: {members}"
            )
    else:
        lines.append("- None")
    lines.extend(["", "## Uncovered Model IR leaves", ""])
    if report["uncovered_model_ir_leaves"]:
        lines.extend(f"- `{target}`" for target in report["uncovered_model_ir_leaves"])
    else:
        lines.append("- None")
    lines.extend(["", "## Compound primitive targets", ""])
    if report["compound_primitive_targets"]:
        for target, owners in report["compound_primitive_targets"].items():
            lines.append(f"- `{target}`")
            lines.extend(f"  - `{owner}`" for owner in owners)
    else:
        lines.append("- None")
    lines.extend(["", "## Ledger/source errors", ""])
    if report["errors"]:
        lines.extend(f"- {error}" for error in report["errors"])
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Review rule",
            "",
            "Do not batch-edit Model IR from this report until every pending source "
            "entrypoint has a reviewed obligation list. After that review, patch the "
            "Model IR atomically, reconcile Binding/Profile mappings for new runtime "
            "leaves, and rerun this audit. Only a `complete` report may replace the "
            "catalog's semantic-closure attestation.",
            "",
        ]
    )
    return "\n".join(lines)
