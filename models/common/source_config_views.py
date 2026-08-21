#!/usr/bin/env python3
"""Merge source/config-generated canonical views with trace-derived views.

This module is intentionally model-neutral. Model-specific knowledge lives in a
YAML contract that lists the source/config-only views, checked source spans, and
the trace-derived architecture artifact to merge with.
"""

from __future__ import annotations

import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised manually
    raise SystemExit("requires pyyaml") from exc

from models.common.source_reconciler import (
    config_has_path,
    load_yaml,
    resolve_path,
    validate_source_identity,
    validate_source_link,
)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def write_yaml(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True)


def _source_link_to_code_link(link: dict[str, Any]) -> str:
    evidence = link.get("evidence") or {}
    if evidence.get("code_link"):
        return str(evidence["code_link"])
    file_path = str(link["file"])
    display_file = link.get("display_file") or file_path.removeprefix("python/sglang/srt/")
    line = int(link["line"])
    line_end = int(link.get("line_end") or line)
    span = str(line) if line == line_end else f"{line}-{line_end}"
    label = link.get("label")
    return f"{display_file}:{span} {label}" if label else f"{display_file}:{span}"


def _node_source_identities(node: dict[str, Any]) -> list[dict[str, Any]]:
    identities: list[dict[str, Any]] = []
    if node.get("source_identity"):
        identities.append(node.pop("source_identity"))
    identities.extend(node.pop("source_identities", []) or [])
    return identities


def _default_provenance(node: dict[str, Any]) -> dict[str, Any]:
    existing = dict(node.get("provenance") or {})
    existing.setdefault("architecture", "source-config-generated")
    has_source = bool(node.get("source_evidence") or node.get("source_identity_evidence"))
    existing.setdefault(
        "source",
        "validated source span / AST identity" if has_source else "not-used",
    )
    existing.setdefault("config", "validated config refs" if node.get("config_refs") else "not-used")
    existing.setdefault("runtime", "not_observed_in_current_trace")
    existing.setdefault("profile", "phase_not_executed_or_not_attributed")
    return existing


def _validate_edges_and_drills(
    *,
    views: dict[str, Any],
    errors: list[str],
    validate_drills: bool = True,
) -> None:
    for view_name, view in views.items():
        if "same_as" in view:
            target = view["same_as"]
            if target not in views:
                errors.append(f"view {view_name} aliases missing view {target!r}")
            continue
        node_ids = {
            str(node["id"])
            for node in view.get("nodes", []) or []
            if isinstance(node, dict) and node.get("id")
        }
        for node in view.get("nodes", []) or []:
            node_id = str(node.get("id") or "<missing>")
            drill = node.get("drill")
            if validate_drills and drill and drill not in views:
                errors.append(f"{view_name}.{node_id} drills into missing view {drill!r}")
        for edge in view.get("edges", []) or []:
            src = edge.get("from")
            dst = edge.get("to")
            if src not in node_ids:
                errors.append(f"{view_name} edge has missing source node {src!r}")
            if dst not in node_ids:
                errors.append(f"{view_name} edge has missing target node {dst!r}")


def _collect_node_ids(views: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for view_name, view in views.items():
        for node in view.get("nodes", []) or []:
            node_id = node.get("id")
            if node_id:
                out[str(node_id)] = view_name
    return out


def _validate_canonical_source_conflicts(
    *,
    views: dict[str, Any],
    allowed_aliases: dict[str, list[str]],
    errors: list[str],
    warnings: list[str],
) -> None:
    canonical_to_display: dict[str, set[str]] = defaultdict(set)
    for view in views.values():
        for node in view.get("nodes", []) or []:
            source_identity = node.get("source_identity") or {}
            for canonical_id in source_identity.get("canonical_source_ids") or []:
                canonical_to_display[str(canonical_id)].add(str(node["id"]))

    for canonical_id, display_ids in sorted(canonical_to_display.items()):
        if len(display_ids) <= 1:
            continue
        allowed = set(allowed_aliases.get(canonical_id) or [])
        if display_ids <= allowed:
            warnings.append(
                f"canonical source id {canonical_id!r} is intentionally shared by {sorted(display_ids)}"
            )
        else:
            errors.append(
                f"canonical source id {canonical_id!r} maps to multiple display ids "
                f"{sorted(display_ids)} without an allowlist entry"
            )


def _process_source_config_views(
    *,
    config: dict[str, Any],
    config_base: Path,
    source_root: Path,
    model_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    errors: list[str] = []
    warnings: list[str] = []
    summary = {
        "views": 0,
        "nodes": 0,
        "source_links": 0,
        "verified_source_links": 0,
        "source_identities": 0,
        "verified_source_identities": 0,
        "config_refs": 0,
        "verified_config_refs": 0,
    }

    views = copy.deepcopy(config.get("views") or {})
    summary["views"] = len(views)
    for view_name, view in views.items():
        for node in view.get("nodes", []) or []:
            node_id = node.get("id")
            if not node_id:
                errors.append(f"view {view_name} has node without id")
                continue
            summary["nodes"] += 1

            identity_evidence: list[dict[str, Any]] = []
            for identity in _node_source_identities(node):
                summary["source_identities"] += 1
                evidence, identity_errors = validate_source_identity(
                    identity, source_root=source_root
                )
                identity_evidence.append(evidence)
                errors.extend(f"node {node_id}: {msg}" for msg in identity_errors)
                if evidence["ok"]:
                    summary["verified_source_identities"] += 1
            if identity_evidence:
                canonical_ids = [
                    item["canonical_source_id"]
                    for item in identity_evidence
                    if item.get("canonical_source_id")
                ]
                node["source_identity"] = {
                    "display_id": node_id,
                    "canonical_source_ids": canonical_ids,
                    "alias_role": "canonical source/config node",
                }
                node["source_identity_evidence"] = identity_evidence

            source_links = node.pop("source_links", []) or []
            source_evidence: list[dict[str, Any]] = []
            code_links = list(node.get("code_links") or [])
            for link in source_links:
                summary["source_links"] += 1
                evidence, link_errors = validate_source_link(link, source_root=source_root)
                source_evidence.append(evidence)
                code_links.append(evidence["code_link"])
                errors.extend(f"node {node_id}: {msg}" for msg in link_errors)
                if evidence["ok"]:
                    summary["verified_source_links"] += 1
            if source_evidence:
                node["source_evidence"] = source_evidence
                node["code_links"] = code_links

            for config_ref in node.get("config_refs") or []:
                summary["config_refs"] += 1
                if config_has_path(model_config, str(config_ref)):
                    summary["verified_config_refs"] += 1
                else:
                    errors.append(f"node {node_id}: missing config ref {config_ref!r}")

            node["provenance"] = _default_provenance(node)

    _validate_edges_and_drills(views=views, errors=errors, validate_drills=False)
    return views, {"errors": errors, "warnings": warnings, "summary": summary}


def _merge_views(
    *,
    source_views: dict[str, Any],
    trace_arch: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    trace_views = copy.deepcopy(trace_arch.get("views") or {})
    duplicate_views = sorted(set(source_views) & set(trace_views))
    if duplicate_views:
        errors.append(
            "source/config views conflict with trace-derived views: "
            + ", ".join(duplicate_views)
        )

    merged_views = {**source_views, **trace_views}
    _validate_edges_and_drills(views=merged_views, errors=errors)

    duplicate_nodes: dict[str, list[str]] = defaultdict(list)
    for view_name, view in merged_views.items():
        for node in view.get("nodes", []) or []:
            if node.get("id"):
                duplicate_nodes[str(node["id"])].append(view_name)
            if "provenance" not in node:
                errors.append(f"{view_name}.{node.get('id', '<missing>')} has no provenance")
    for node_id, view_names in sorted(duplicate_nodes.items()):
        if len(view_names) > 1:
            warnings.append(f"node id {node_id!r} appears in multiple views: {view_names}")

    allowed_aliases = {
        item["canonical_source_id"]: list(item.get("display_ids") or [])
        for item in config.get("allowed_canonical_aliases", []) or []
    }
    _validate_canonical_source_conflicts(
        views=merged_views,
        allowed_aliases=allowed_aliases,
        errors=errors,
        warnings=warnings,
    )
    return merged_views, errors, warnings


def _artifact_entry(path: Path | str, **fields: Any) -> dict[str, Any]:
    entry = {"path": str(path)}
    entry.update(fields)
    return entry


def _build_artifact_index(
    *,
    config: dict[str, Any],
    config_path: Path,
    output_dir: Path,
    trace_arch_path: Path,
    report: dict[str, Any],
) -> dict[str, Any]:
    base = config_path.parent
    entries = []
    for item in config.get("artifact_index", {}).get("entries", []) or []:
        raw_path = item.get("path")
        if raw_path:
            resolved = resolve_path(raw_path, base=base)
            item = {**item, "path": str(resolved) if resolved else str(raw_path)}
        entries.append(item)

    generated = [
        _artifact_entry(
            output_dir / "arch_generated.yaml",
            stage="source_config_merge",
            role="output",
            provenance="script_generated",
            producer="models/common/source_config_views.py",
            consumers=["models/common/build_view.py"],
        ),
        _artifact_entry(
            output_dir / "arch_generated_report.json",
            stage="source_config_merge",
            role="validation_report",
            provenance="script_generated",
            producer="models/common/source_config_views.py",
            consumers=["human_review", "tests"],
        ),
        _artifact_entry(
            output_dir / "artifact_index.json",
            stage="source_config_merge",
            role="provenance_index",
            provenance="script_generated",
            producer="models/common/source_config_views.py",
            consumers=["human_review", "future_docs"],
        ),
        _artifact_entry(
            trace_arch_path,
            stage="source_reconciliation",
            role="input_trace_derived_architecture",
            provenance="script_generated",
            producer="models/common/source_reconciler.py",
            consumers=["models/common/source_config_views.py"],
        ),
    ]
    return {
        "schema_version": "artifact_index.v0",
        "model_id": config.get("model_id"),
        "source_config": str(config_path),
        "ok": report["ok"],
        "summary": report["summary"],
        "entries": entries + generated,
    }


def build_source_config_architecture(
    *,
    source_config_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source_config_path = source_config_path.resolve()
    config = load_yaml(source_config_path)
    base = source_config_path.parent

    source_root = resolve_path(config["source_root"], base=base)
    trace_arch_path = resolve_path(config["trace_arch"], base=base)
    output_dir = resolve_path(config["output_dir"], base=base)
    model_config_path = resolve_path(config.get("model_config"), base=base)
    if source_root is None or trace_arch_path is None or output_dir is None:
        raise ValueError("source_root, trace_arch, and output_dir are required")

    model_config = load_yaml(model_config_path) if model_config_path else {}
    trace_arch = load_yaml(trace_arch_path)

    source_views, source_report = _process_source_config_views(
        config=config,
        config_base=base,
        source_root=source_root,
        model_config=model_config,
    )
    merged_views, merge_errors, merge_warnings = _merge_views(
        source_views=source_views,
        trace_arch=trace_arch,
        config=config,
    )

    errors = list(source_report["errors"]) + merge_errors
    warnings = list(source_report["warnings"]) + merge_warnings
    trace_view_count = len(trace_arch.get("views") or {})
    source_summary = source_report["summary"]
    trace_node_count = sum(
        len(view.get("nodes", []) or [])
        for view in (trace_arch.get("views") or {}).values()
        if "same_as" not in view
    )
    source_node_count = source_summary["nodes"]
    report = {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "source_config_views": source_summary["views"],
            "trace_views": trace_view_count,
            "merged_views": len(merged_views),
            "source_config_nodes": source_node_count,
            "trace_nodes": trace_node_count,
            "merged_nodes": source_node_count + trace_node_count,
            "source_links": source_summary["source_links"],
            "verified_source_links": source_summary["verified_source_links"],
            "source_identities": source_summary["source_identities"],
            "verified_source_identities": source_summary["verified_source_identities"],
            "config_refs": source_summary["config_refs"],
            "verified_config_refs": source_summary["verified_config_refs"],
        },
        "phase_overlay_contracts": config.get("phase_overlay_contracts") or [],
    }

    artifact_index = _build_artifact_index(
        config=config,
        config_path=source_config_path,
        output_dir=output_dir,
        trace_arch_path=trace_arch_path,
        report=report,
    )
    arch = {
        "schema_version": "arch_generated.v0",
        "model_id": config.get("model_id"),
        "source": {
            "source_config": str(source_config_path),
            "trace_arch": str(trace_arch_path),
            "source_root": str(source_root),
            "source_repo": trace_arch.get("source", {}).get("source_repo"),
            "source_commit": trace_arch.get("source", {}).get("source_commit"),
            "model_config": str(model_config_path) if model_config_path else None,
        },
        "generation_policy": config.get("generation_policy") or {},
        "phase_overlay_contracts": config.get("phase_overlay_contracts") or [],
        "views": merged_views,
        "validation": {
            "ok": report["ok"],
            "errors": errors,
            "warnings": warnings,
        },
        "artifact_index": str(output_dir / "artifact_index.json"),
    }
    return arch, report, artifact_index


def write_source_config_report_markdown(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = report["summary"]
    lines = [
        "# Source/Config Merge Report",
        "",
        f"- ok: `{report['ok']}`",
        f"- source_config_views: `{summary['source_config_views']}`",
        f"- trace_views: `{summary['trace_views']}`",
        f"- merged_views: `{summary['merged_views']}`",
        f"- source_config_nodes: `{summary['source_config_nodes']}`",
        f"- trace_nodes: `{summary['trace_nodes']}`",
        f"- merged_nodes: `{summary['merged_nodes']}`",
        f"- source_links: `{summary['verified_source_links']}/{summary['source_links']}` verified",
        f"- source_identities: `{summary['verified_source_identities']}/{summary['source_identities']}` verified",
        f"- config_refs: `{summary['verified_config_refs']}/{summary['config_refs']}` verified",
        "",
        "## Phase Overlay Contracts",
        "",
    ]
    contracts = report.get("phase_overlay_contracts") or []
    if contracts:
        for contract in contracts:
            lines.append(
                f"- `{contract.get('phase')}`: `{contract.get('policy')}` "
                f"on {contract.get('canonical_nodes')}"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Errors", ""])
    lines.extend([f"- {msg}" for msg in report["errors"]] or ["- none"])
    lines.extend(["", "## Warnings", ""])
    lines.extend([f"- {msg}" for msg in report["warnings"]] or ["- none"])
    path.write_text("\n".join(lines) + "\n")
