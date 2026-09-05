"""Validate independent evidence provenance for the four IR acceptance gates.

This module deliberately validates *where expectations came from* separately
from validating the resulting Model IR, Execution IR, Binding, or Profile.  It
does not pretend that an offline attestation re-extracted an external source:
the report preserves whether an authority was machine-resolved locally or was
reviewed against an immutable external revision and digest.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


GATE_NAMES = (
    "semantic_ir",
    "execution_contract",
    "binding_reconciliation",
    "production_evidence",
)

ALLOWED_AUTHORITY_KINDS = {
    "semantic_ir": {
        "publisher_checkpoint",
        "canonical_model_source",
        "technical_specification",
    },
    "execution_contract": {
        "pinned_framework_source",
        "runtime_configuration",
        "topology_manifest",
    },
    "binding_reconciliation": {
        "pinned_framework_source",
        "eager_reconciliation",
    },
    "production_evidence": {
        "eager_reconciliation",
        "capture_manifest",
        "production_trace",
    },
}

REQUIRED_AUTHORITY_FAMILIES = {
    "semantic_ir": (
        {
            "publisher_checkpoint",
            "canonical_model_source",
            "technical_specification",
        },
    ),
    "execution_contract": (
        {"pinned_framework_source"},
        {"runtime_configuration", "topology_manifest"},
    ),
    "binding_reconciliation": (
        {"pinned_framework_source"},
        {"eager_reconciliation"},
    ),
    "production_evidence": (
        {"eager_reconciliation"},
        {"production_trace"},
    ),
}

EXPECTED_VERIFICATION_MODES = {
    "semantic_ir": "reviewed_external_semantics",
    "execution_contract": "source_and_config_contract",
    "binding_reconciliation": "graph_off_eager_reconciliation",
    "production_evidence": "graph_on_production_reconciliation",
}

CANONICAL_SUBJECT_GLOBS = {
    "semantic_ir": ("model_ir.yaml",),
    "execution_contract": ("execution_paths/*.yaml",),
    "binding_reconciliation": ("bindings/*.yaml",),
    "production_evidence": ("profiles/*/*/*.yaml",),
}


@dataclass(frozen=True)
class LoadedDocument:
    path: Path
    data: Any


def _load(path: Path) -> LoadedDocument:
    if path.suffix in {".yaml", ".yml"}:
        return LoadedDocument(path=path, data=yaml.safe_load(path.read_text()))
    return LoadedDocument(path=path, data=None)


def _select(value: Any, selector: str) -> Any:
    current = value
    for component in selector.split("."):
        if not isinstance(current, dict) or component not in current:
            raise KeyError(selector)
        current = current[component]
    return current


def _resolve_files(model_root: Path, spec: dict[str, Any]) -> list[Path]:
    if spec.get("artifact"):
        return [model_root / str(spec["artifact"])]
    return sorted(model_root.glob(str(spec.get("artifact_glob") or "")))


def _relative(path: Path, model_root: Path) -> str:
    try:
        return str(path.relative_to(model_root))
    except ValueError:
        return str(path)


def _authority_resolution_files(
    model_root: Path, authority: dict[str, Any]
) -> list[Path]:
    resolution = (authority.get("source") or {}).get("local_resolution")
    return _resolve_files(model_root, resolution) if resolution else []


def _validate_local_authority(
    *,
    model_root: Path,
    authority: dict[str, Any],
    errors: list[str],
) -> dict[str, Any]:
    authority_id = str(authority.get("id") or "<missing>")
    source = authority.get("source") or {}
    resolution = source.get("local_resolution")
    assurance = authority.get("assurance")
    if assurance == "machine_resolved" and not resolution:
        errors.append(f"authority {authority_id}: machine_resolved requires local_resolution")
        return {"resolved_artifact_count": 0}
    if assurance == "immutable_external_attestation":
        if not source.get("revision") or not source.get("digest"):
            errors.append(
                f"authority {authority_id}: external attestation requires revision and digest"
            )
        if resolution:
            errors.append(
                f"authority {authority_id}: external attestation cannot claim local resolution"
            )
        return {"resolved_artifact_count": 0}
    if not resolution:
        return {"resolved_artifact_count": 0}

    files = _resolve_files(model_root, resolution)
    minimum = int(resolution.get("minimum_matches") or 1)
    if len(files) < minimum:
        errors.append(
            f"authority {authority_id}: resolved {len(files)} artifacts, expected at least {minimum}"
        )
    missing = [path for path in files if not path.is_file()]
    for path in missing:
        errors.append(f"authority {authority_id}: missing artifact {_relative(path, model_root)}")

    selector = resolution.get("selector")
    required_fields = resolution.get("required_fields") or []
    resolved_values = 0
    for path in files:
        if not path.is_file() or path.suffix not in {".yaml", ".yml"}:
            continue
        document = _load(path)
        if selector:
            try:
                _select(document.data, str(selector))
                resolved_values += 1
            except KeyError:
                errors.append(
                    f"authority {authority_id}: {_relative(path, model_root)} "
                    f"does not resolve selector {selector}"
                )
        for field in required_fields:
            try:
                _select(document.data, str(field))
            except KeyError:
                errors.append(
                    f"authority {authority_id}: {_relative(path, model_root)} "
                    f"does not resolve required field {field}"
                )
    return {
        "resolved_artifact_count": len(files) - len(missing),
        "resolved_value_count": resolved_values,
    }


def _validate_assertion(
    *,
    model_root: Path,
    assertion: dict[str, Any],
    authority_by_id: dict[str, dict[str, Any]],
    allowed_authorities: set[str],
    errors: list[str],
) -> None:
    assertion_id = str(assertion.get("id") or "<missing>")
    refs = set(assertion.get("authority_refs") or [])
    missing_refs = refs - set(authority_by_id)
    if missing_refs:
        errors.append(f"assertion {assertion_id}: unknown authorities {sorted(missing_refs)}")
    disallowed_refs = refs - allowed_authorities
    if disallowed_refs:
        errors.append(
            f"assertion {assertion_id}: authorities are not admitted by its gate "
            f"{sorted(disallowed_refs)}"
        )

    subject = assertion.get("subject") or {}
    files = _resolve_files(model_root, subject)
    operator = assertion.get("operator")
    expected = assertion.get("expected")
    if operator == "collection_count_equals":
        if len(files) != expected:
            errors.append(
                f"assertion {assertion_id}: collection count {len(files)} != {expected}"
            )
        return

    existing = [path for path in files if path.is_file()]
    if len(existing) != len(files) or not existing:
        errors.append(f"assertion {assertion_id}: subject artifacts do not resolve")
        return

    if operator == "all_documents_have_fields":
        for path in existing:
            document = _load(path)
            for selector in assertion.get("required_fields") or []:
                try:
                    _select(document.data, str(selector))
                except KeyError:
                    errors.append(
                        f"assertion {assertion_id}: {_relative(path, model_root)} "
                        f"does not resolve {selector}"
                    )
        return

    if operator == "all_values_in_authority":
        authority_ref = str(assertion.get("authority_ref") or "")
        if authority_ref not in refs:
            errors.append(
                f"assertion {assertion_id}: authority_ref {authority_ref!r} "
                "must also appear in authority_refs"
            )
            return
        authority = authority_by_id.get(authority_ref) or {}
        authority_files = _authority_resolution_files(model_root, authority)
        if len(authority_files) != 1 or not authority_files[0].is_file():
            errors.append(
                f"assertion {assertion_id}: authority {authority_ref} must resolve one artifact"
            )
            return
        try:
            collection = _select(
                _load(authority_files[0]).data,
                str(assertion.get("authority_selector") or ""),
            )
        except KeyError:
            errors.append(
                f"assertion {assertion_id}: authority selector does not resolve"
            )
            return
        if not isinstance(collection, list):
            errors.append(
                f"assertion {assertion_id}: authority selector must resolve a list"
            )
            return
        admitted_values: set[Any] = set()
        for item in collection:
            for selector in assertion.get("authority_value_selectors") or []:
                try:
                    admitted_values.add(_select(item, str(selector)))
                except KeyError:
                    continue
        if not admitted_values:
            errors.append(
                f"assertion {assertion_id}: authority collection provides no admitted values"
            )
            return
        selector = str(assertion.get("selector") or "")
        for path in existing:
            try:
                actual = _select(_load(path).data, selector)
            except KeyError:
                errors.append(
                    f"assertion {assertion_id}: {_relative(path, model_root)} "
                    f"does not resolve {selector}"
                )
                continue
            if actual not in admitted_values:
                errors.append(
                    f"assertion {assertion_id}: {_relative(path, model_root)} value "
                    f"{actual!r} is absent from authority {authority_ref}"
                )
        return

    if len(existing) != 1:
        errors.append(f"assertion {assertion_id}: {operator} requires one subject")
        return
    selector = assertion.get("selector")
    if not selector:
        errors.append(f"assertion {assertion_id}: {operator} requires selector")
        return
    try:
        actual = _select(_load(existing[0]).data, str(selector))
    except KeyError:
        errors.append(f"assertion {assertion_id}: selector {selector} does not resolve")
        return
    if operator == "length_equals":
        try:
            actual = len(actual)
        except TypeError:
            errors.append(f"assertion {assertion_id}: selected value has no length")
            return
    if actual != expected:
        errors.append(f"assertion {assertion_id}: observed {actual!r} != expected {expected!r}")


def validate_validation_evidence(model_root: Path) -> dict[str, Any]:
    """Validate one catalog's four-gate evidence and anti-circularity contract."""

    model_root = model_root.resolve()
    contract_path = model_root / "validation_evidence.yaml"
    errors: list[str] = []
    if not contract_path.is_file():
        return {
            "schema_version": "validation-evidence-report.v1",
            "model_id": model_root.name,
            "status": "fail",
            "errors": ["missing validation_evidence.yaml"],
            "gates": {},
        }

    contract = yaml.safe_load(contract_path.read_text()) or {}
    if contract.get("schema_version") != "validation-evidence.v1":
        errors.append("unsupported validation evidence schema_version")
    if contract.get("model_id") != model_root.name:
        errors.append(
            f"model_id {contract.get('model_id')!r} does not match catalog {model_root.name!r}"
        )

    authorities = contract.get("authorities") or []
    authority_by_id = {
        str(authority.get("id")): authority
        for authority in authorities
        if authority.get("id")
    }
    if len(authority_by_id) != len(authorities):
        errors.append("authority IDs must be present and unique")
    authority_reports: dict[str, Any] = {}
    for authority_id, authority in authority_by_id.items():
        authority_errors_before = len(errors)
        resolution_report = _validate_local_authority(
            model_root=model_root, authority=authority, errors=errors
        )
        authority_reports[authority_id] = {
            "kind": authority.get("kind"),
            "assurance": authority.get("assurance"),
            "status": "pass" if len(errors) == authority_errors_before else "fail",
            "source_uri": (authority.get("source") or {}).get("uri"),
            "source_revision": (authority.get("source") or {}).get("revision"),
            "source_digest": (authority.get("source") or {}).get("digest"),
            **resolution_report,
        }

    gate_reports: dict[str, Any] = {}
    gates = contract.get("gates") or {}
    for gate_name in GATE_NAMES:
        gate = gates.get(gate_name) or {}
        gate_errors_before = len(errors)
        if gate.get("status") != "verified":
            errors.append(f"gate {gate_name}: status must be verified")
        expected_mode = EXPECTED_VERIFICATION_MODES[gate_name]
        if gate.get("verification_mode") != expected_mode:
            errors.append(
                f"gate {gate_name}: verification_mode must be {expected_mode}"
            )

        refs = set(gate.get("authority_refs") or [])
        missing_refs = refs - set(authority_by_id)
        if missing_refs:
            errors.append(f"gate {gate_name}: unknown authorities {sorted(missing_refs)}")
        for ref in refs.intersection(authority_by_id):
            if authority_reports[ref]["status"] != "pass":
                errors.append(f"gate {gate_name}: authority {ref} did not validate")
        if gate_name == "binding_reconciliation":
            for binding_path in sorted(model_root.glob("bindings/*.yaml")):
                binding = yaml.safe_load(binding_path.read_text()) or {}
                acceptance_sha = binding.get("add_trace_acceptance_sha256")
                if not acceptance_sha:
                    continue
                revision_id = binding.get("binding_revision_id")
                matching_authorities = []
                for authority_id, authority in authority_by_id.items():
                    source = authority.get("source") or {}
                    digest = source.get("digest") or {}
                    if (
                        authority.get("kind") == "eager_reconciliation"
                        and source.get("revision") == revision_id
                        and digest.get("algorithm") == "sha256"
                        and digest.get("value") == acceptance_sha
                    ):
                        matching_authorities.append(authority_id)
                if len(matching_authorities) != 1:
                    errors.append(
                        f"gate {gate_name}: versioned Binding "
                        f"{_relative(binding_path, model_root)} requires exactly one "
                        "matching add-trace acceptance authority"
                    )
                elif matching_authorities[0] not in refs:
                    errors.append(
                        f"gate {gate_name}: add-trace acceptance authority "
                        f"{matching_authorities[0]} is not referenced by the gate"
                    )
        kinds = {
            authority_by_id[ref].get("kind")
            for ref in refs
            if ref in authority_by_id
        }
        disallowed_kinds = kinds - ALLOWED_AUTHORITY_KINDS[gate_name]
        if disallowed_kinds:
            errors.append(
                f"gate {gate_name}: disallowed authority kinds {sorted(disallowed_kinds)}"
            )
        for family in REQUIRED_AUTHORITY_FAMILIES[gate_name]:
            if not kinds.intersection(family):
                errors.append(
                    f"gate {gate_name}: missing authority family {sorted(family)}"
                )

        subject_files: set[Path] = set()
        for subject in gate.get("subjects") or []:
            resolved = _resolve_files(model_root, subject)
            minimum = int(subject.get("minimum_matches") or 1)
            if len(resolved) < minimum:
                errors.append(
                    f"gate {gate_name}: subject resolves {len(resolved)} artifacts, "
                    f"expected at least {minimum}"
                )
            for path in resolved:
                if not path.is_file():
                    errors.append(
                        f"gate {gate_name}: missing subject {_relative(path, model_root)}"
                    )
                else:
                    subject_files.add(path.resolve())

        canonical_files = {
            path.resolve()
            for pattern in CANONICAL_SUBJECT_GLOBS[gate_name]
            for path in model_root.glob(pattern)
            if path.is_file()
        }
        if subject_files != canonical_files:
            missing = canonical_files - subject_files
            extra = subject_files - canonical_files
            if missing:
                errors.append(
                    f"gate {gate_name}: uncovered canonical subjects "
                    f"{sorted(_relative(path, model_root) for path in missing)}"
                )
            if extra:
                errors.append(
                    f"gate {gate_name}: subjects outside canonical layer "
                    f"{sorted(_relative(path, model_root) for path in extra)}"
                )

        for ref in refs:
            authority = authority_by_id.get(ref)
            if not authority:
                continue
            overlap = subject_files.intersection(
                path.resolve()
                for path in _authority_resolution_files(model_root, authority)
            )
            if overlap:
                errors.append(
                    f"gate {gate_name}: self-validation authority {ref} overlaps subjects "
                    f"{sorted(_relative(path, model_root) for path in overlap)}"
                )

        assertions = gate.get("assertions") or []
        assertion_ids = [assertion.get("id") for assertion in assertions]
        if not assertions or len(set(assertion_ids)) != len(assertion_ids):
            errors.append(f"gate {gate_name}: assertion IDs must be non-empty and unique")
        for assertion in assertions:
            _validate_assertion(
                model_root=model_root,
                assertion=assertion,
                authority_by_id=authority_by_id,
                allowed_authorities=refs,
                errors=errors,
            )

        gate_reports[gate_name] = {
            "status": "pass" if len(errors) == gate_errors_before else "fail",
            "verification_mode": gate.get("verification_mode"),
            "subject_count": len(subject_files),
            "assertion_count": len(assertions),
            "authority_ids": sorted(refs),
            "authority_kinds": sorted(kind for kind in kinds if kind),
            "machine_resolved_authority_count": sum(
                authority_by_id[ref].get("assurance") == "machine_resolved"
                for ref in refs
                if ref in authority_by_id
            ),
            "external_attestation_authority_count": sum(
                authority_by_id[ref].get("assurance")
                == "immutable_external_attestation"
                for ref in refs
                if ref in authority_by_id
            ),
        }

    return {
        "schema_version": "validation-evidence-report.v1",
        "model_id": model_root.name,
        "status": "pass" if not errors else "fail",
        "contract_revision": contract.get("contract_revision"),
        "contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
        "anti_self_validation": "pass" if not errors else "fail",
        "authorities": authority_reports,
        "gates": gate_reports,
        "errors": errors,
    }
