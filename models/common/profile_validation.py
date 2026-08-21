#!/usr/bin/env python3
"""Config-driven validation for architecture/profile mappings.

The validator intentionally avoids model-specific stage names. Per-model
expectations live in validation YAML files and per-stage profile policy lives in
`stages.yaml`.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised manually
    raise SystemExit("requires pyyaml") from exc


@dataclass
class ValidationReport:
    model_id: str
    profile_id: str
    used_stage_keys: int
    profile_stages: int
    variants: list[str]
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def load_yaml(path: Path) -> Any:
    with path.open() as fh:
        return yaml.safe_load(fh)


def load_json(path: Path) -> Any:
    with path.open() as fh:
        return json.load(fh)


def find_single_arch_file(ir_dir: Path) -> Path:
    candidates = sorted(ir_dir.glob("arch.*.yaml"))
    if len(candidates) != 1:
        raise RuntimeError(f"expected exactly one arch.*.yaml under {ir_dir}, got {candidates}")
    return candidates[0]


def iter_arch_nodes(arch: dict[str, Any]):
    for view_name, view in arch.get("views", {}).items():
        if "same_as" in view:
            continue
        for node in view.get("nodes", []) or []:
            yield view_name, node


def collect_stage_aliases(stages: list[dict[str, Any]]) -> dict[str, set[str]]:
    alias_to_stage_ids: dict[str, set[str]] = {}
    for stage in stages:
        aliases = set(stage.get("trace_aliases") or [])
        if stage.get("pdf_name"):
            aliases.add(stage["pdf_name"])
        for alias in aliases:
            alias_to_stage_ids.setdefault(alias, set()).add(stage["id"])
    return alias_to_stage_ids


def collect_profile_variants(profile_data: dict[str, Any]) -> list[str]:
    return sorted(
        {
            variant
            for stage_cell in profile_data.values()
            if isinstance(stage_cell, dict)
            for variant in stage_cell.keys()
        }
    )


def resolve_variants(spec: Any, actual_variants: list[str]) -> list[str]:
    if spec is None or spec == "all":
        return actual_variants
    if isinstance(spec, str):
        return [spec]
    return list(spec)


def stage_profile_keys(stage_id: str, stages_by_id: dict[str, dict[str, Any]]) -> list[str]:
    stage = stages_by_id.get(stage_id, {})
    candidates = [stage_id]
    candidates.extend(stage.get("trace_aliases") or [])
    if stage.get("pdf_name"):
        candidates.append(stage["pdf_name"])
    out: list[str] = []
    seen: set[str] = set()
    for key in candidates:
        if key and key not in seen:
            out.append(key)
            seen.add(key)
    return out


def profile_cell(
    profile_data: dict[str, Any],
    stages_by_id: dict[str, dict[str, Any]],
    stage_id: str,
    variant: str,
) -> dict[str, Any] | None:
    for key in stage_profile_keys(stage_id, stages_by_id):
        cell = profile_data.get(key, {})
        if isinstance(cell, dict) and isinstance(cell.get(variant), dict):
            return cell[variant]
    return None


def kernel_names(cell: dict[str, Any] | None) -> set[str]:
    if not cell:
        return set()
    return {str(kernel.get("name", "")) for kernel in cell.get("kernels", []) or []}


def names_matching_substrings(names: set[str], substrings: list[str]) -> list[str]:
    matches: list[str] = []
    for name in names:
        low = name.lower()
        if any(substring.lower() in low for substring in substrings):
            matches.append(name)
    return sorted(matches)


def validate_stage_references(
    *,
    arch: dict[str, Any],
    stages: list[dict[str, Any]],
    profile_data: dict[str, Any],
    require_all_arch_stages: bool = True,
    errors: list[str],
    warnings: list[str],
) -> set[str]:
    stage_ids = {stage["id"] for stage in stages}
    stages_by_id = {stage["id"]: stage for stage in stages}
    alias_to_stage_ids = collect_stage_aliases(stages)

    used_stage_keys: set[str] = set()
    for view_name, node in iter_arch_nodes(arch):
        for stage_key in node.get("stage_keys") or []:
            used_stage_keys.add(stage_key)
            if stage_key not in stage_ids:
                errors.append(f"{view_name}.{node['id']} references unknown stage {stage_key}")

    for alias, ids in sorted(alias_to_stage_ids.items()):
        if len(ids) > 1:
            errors.append(f"trace alias {alias!r} maps to multiple stages: {sorted(ids)}")

    known_profile_keys = set(stage_ids) | set(alias_to_stage_ids)
    for key in sorted(profile_data):
        if key not in known_profile_keys:
            warnings.append(f"profile key {key!r} is not a declared stage id or trace alias")

    if require_all_arch_stages:
        for stage_key in sorted(used_stage_keys):
            if any(key in profile_data for key in stage_profile_keys(stage_key, stages_by_id)):
                continue
            stage = stages_by_id.get(stage_key, {})
            policy = stage.get("profile_policy", "required")
            reason = stage.get("profile_policy_reason", "")
            if policy == "required":
                errors.append(f"stage {stage_key} is used by arch but missing from profile")
            elif policy not in {"optional", "source_only", "phase_not_applicable"}:
                errors.append(f"stage {stage_key} has unknown profile_policy {policy!r}")
            elif reason:
                warnings.append(f"{stage_key} has no profile data by policy {policy}: {reason}")
    else:
        warnings.append("require_all_arch_stages=false; partial profile overlay is allowed")

    return used_stage_keys


def validate_expected_variants(
    *,
    config: dict[str, Any],
    variants: list[str],
    errors: list[str],
) -> None:
    expected = config.get("expected_variants")
    if expected is None:
        return
    expected_variants = sorted(expected)
    if variants != expected_variants:
        errors.append(f"profile variants mismatch: expected {expected_variants}, got {variants}")


def validate_required_profile_stages(
    *,
    config: dict[str, Any],
    profile_data: dict[str, Any],
    stages_by_id: dict[str, dict[str, Any]],
    variants: list[str],
    errors: list[str],
) -> None:
    for item in config.get("required_profile_stages", []) or []:
        stage_id = item["stage"]
        selected_variants = resolve_variants(item.get("variants"), variants)
        nonzero = item.get("nonzero", True)
        if stage_id not in stages_by_id:
            errors.append(f"required_profile_stages references unknown stage {stage_id}")
            continue
        for variant in selected_variants:
            cell = profile_cell(profile_data, stages_by_id, stage_id, variant)
            if not cell:
                errors.append(f"{stage_id}.{variant} is missing")
            elif nonzero and not cell.get("ms_per_iter"):
                errors.append(f"{stage_id}.{variant} has zero ms_per_iter")


def validate_kernel_expectations(
    *,
    config: dict[str, Any],
    profile_data: dict[str, Any],
    stages_by_id: dict[str, dict[str, Any]],
    variants: list[str],
    errors: list[str],
) -> None:
    for item in config.get("forbidden_kernels", []) or []:
        stage_id = item["stage"]
        selected_variants = resolve_variants(item.get("variants", item.get("variant")), variants)
        for variant in selected_variants:
            names = kernel_names(profile_cell(profile_data, stages_by_id, stage_id, variant))
            forbidden = set(item.get("names") or [])
            bad = sorted(names & forbidden)
            bad.extend(names_matching_substrings(names, item.get("substrings") or []))
            if bad:
                errors.append(f"{stage_id}.{variant} contains forbidden kernels: {sorted(set(bad))}")

    for item in config.get("required_kernels", []) or []:
        stage_id = item["stage"]
        selected_variants = resolve_variants(item.get("variants", item.get("variant")), variants)
        for variant in selected_variants:
            names = kernel_names(profile_cell(profile_data, stages_by_id, stage_id, variant))
            for expected in item.get("names") or []:
                if expected not in names:
                    errors.append(f"{stage_id}.{variant} lacks expected kernel {expected!r}")
            for substring in item.get("substrings") or []:
                if not names_matching_substrings(names, [substring]):
                    errors.append(f"{stage_id}.{variant} lacks kernel substring {substring!r}")


def enriched_cell(
    arch_data: dict[str, Any],
    *,
    view: str,
    node: str,
    profile: str,
    variant: str,
) -> dict[str, Any] | None:
    return (
        arch_data.get("enriched", {})
        .get(view, {})
        .get("nodes_profile", {})
        .get(node, {})
        .get(profile, {})
        .get(variant)
    )


def validate_enriched_expectations(
    *,
    config: dict[str, Any],
    arch_data: dict[str, Any] | None,
    errors: list[str],
) -> None:
    needs_arch_data = bool(
        config.get("required_enriched_nodes")
        or config.get("forbidden_enriched_kernel_substrings")
    )
    if needs_arch_data and arch_data is None:
        errors.append("validation config requires arch_data.json but it is missing")
        return

    if arch_data is None:
        return

    for item in config.get("required_enriched_nodes", []) or []:
        cell = enriched_cell(
            arch_data,
            view=item["view"],
            node=item["node"],
            profile=item["profile"],
            variant=item["variant"],
        )
        if not cell or not cell.get("ms_per_iter"):
            errors.append(
                f"{item['view']}.{item['node']}.{item['profile']}.{item['variant']} "
                "missing enriched ms"
            )

    for item in config.get("forbidden_enriched_kernel_substrings", []) or []:
        cell = enriched_cell(
            arch_data,
            view=item["view"],
            node=item["node"],
            profile=item["profile"],
            variant=item["variant"],
        )
        names = kernel_names(cell)
        bad = names_matching_substrings(names, item.get("substrings") or [])
        if bad:
            errors.append(
                f"{item['view']}.{item['node']}.{item['profile']}.{item['variant']} "
                f"contains forbidden kernels: {bad}"
            )


def validate_model_profile(
    *,
    model_root: Path,
    validation_config_path: Path,
    arch_data_path: Path | None = None,
) -> ValidationReport:
    model_root = model_root.resolve()
    ir_dir = model_root / "ir"
    config = load_yaml(validation_config_path)
    profile_id = config["profile"]
    profile_path = Path(config.get("profile_path") or ir_dir / "profiles" / f"{profile_id}.yaml")
    if not profile_path.is_absolute():
        profile_path = validation_config_path.parent / profile_path

    if arch_data_path is None:
        raw_arch_data_path = config.get("arch_data_path")
        if raw_arch_data_path:
            arch_data_path = Path(raw_arch_data_path)
            if not arch_data_path.is_absolute():
                arch_data_path = validation_config_path.parent / arch_data_path
        else:
            arch_data_path = model_root.parent.parent / "docs" / model_root.name / "arch_data.json"

    raw_arch_path = config.get("arch_path")
    if raw_arch_path:
        arch_path = Path(raw_arch_path)
        if not arch_path.is_absolute():
            arch_path = validation_config_path.parent / arch_path
        arch = load_yaml(arch_path)
    else:
        arch = load_yaml(find_single_arch_file(ir_dir))
    stages = load_yaml(ir_dir / "stages.yaml")["stages"]
    stages_by_id = {stage["id"]: stage for stage in stages}
    profile_data = load_yaml(profile_path)["data"]
    arch_data = load_json(arch_data_path) if arch_data_path and arch_data_path.exists() else None

    errors: list[str] = []
    warnings: list[str] = []
    used_stage_keys = validate_stage_references(
        arch=arch,
        stages=stages,
        profile_data=profile_data,
        require_all_arch_stages=bool(config.get("require_all_arch_stages", True)),
        errors=errors,
        warnings=warnings,
    )
    variants = collect_profile_variants(profile_data)
    validate_expected_variants(config=config, variants=variants, errors=errors)
    validate_required_profile_stages(
        config=config,
        profile_data=profile_data,
        stages_by_id=stages_by_id,
        variants=variants,
        errors=errors,
    )
    validate_kernel_expectations(
        config=config,
        profile_data=profile_data,
        stages_by_id=stages_by_id,
        variants=variants,
        errors=errors,
    )
    validate_enriched_expectations(config=config, arch_data=arch_data, errors=errors)

    return ValidationReport(
        model_id=model_root.name,
        profile_id=profile_id,
        used_stage_keys=len(used_stage_keys),
        profile_stages=len(profile_data),
        variants=variants,
        errors=errors,
        warnings=warnings,
    )


def print_report(report: ValidationReport) -> None:
    print("Profile mapping validation")
    print(f"  model={report.model_id}")
    print(f"  profile={report.profile_id}")
    print(f"  used_stage_keys={report.used_stage_keys}")
    print(f"  profile_stages={report.profile_stages}")
    print(f"  variants={report.variants}")
    for msg in report.warnings:
        print(f"  WARN: {msg}")
    if report.errors:
        for msg in report.errors:
            print(f"  ERROR: {msg}", file=sys.stderr)
        return
    print("  OK")


def validate_and_print(
    *,
    model_root: Path,
    validation_config_path: Path,
    arch_data_path: Path | None = None,
) -> int:
    report = validate_model_profile(
        model_root=model_root,
        validation_config_path=validation_config_path,
        arch_data_path=arch_data_path,
    )
    print_report(report)
    return 0 if report.ok else 1
