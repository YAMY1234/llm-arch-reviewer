"""Deterministic, fail-closed planning and acceptance for new trace evidence.

This module deliberately does not infer Model IR or Execution IR from a trace.
It resolves an independently normalized runtime configuration to one authored
Execution Plan, versions the runtime Binding separately, and accepts timing
only when eager reconciliation and production attribution agree on every
identity and rank boundary.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

from .compiler import CatalogError, compile_catalog, load_yaml


class AddTraceError(ValueError):
    """Raised when an add-trace stage cannot close without guessing."""


SCHEMA_ROOT = Path(__file__).resolve().parents[2] / "schema" / "v2"


def validate_schema(
    document: dict[str, Any], schema_name: str, *, source: Path
) -> None:
    """Validate one persisted stage artifact against the complete V2 registry."""

    resources: dict[str, Resource] = {}
    for path in sorted(SCHEMA_ROOT.glob("*.schema.json")):
        contents = json.loads(path.read_text())
        resource = Resource.from_contents(contents)
        resources[path.name] = resource
        if contents.get("$id"):
            resources[str(contents["$id"])] = resource
    schema_path = SCHEMA_ROOT / schema_name
    if not schema_path.is_file():
        raise AddTraceError(f"missing schema: {schema_path}")
    schema = json.loads(schema_path.read_text())
    registry = Registry().with_resources(resources.items())
    errors = sorted(
        Draft202012Validator(
            schema, registry=registry, format_checker=FormatChecker()
        ).iter_errors(document),
        key=lambda error: [str(component) for component in error.absolute_path],
    )
    if errors:
        rendered = "; ".join(
            f"{'.'.join(map(str, error.absolute_path)) or '<root>'}: {error.message}"
            for error in errors
        )
        raise AddTraceError(f"{source}: {schema_name} validation failed: {rendered}")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _value_at_path(document: dict[str, Any], dotted_path: str) -> Any:
    cursor: Any = document
    for component in dotted_path.split("."):
        if not isinstance(cursor, dict) or component not in cursor:
            raise KeyError(dotted_path)
        cursor = cursor[component]
    return cursor


def _leaf_items(value: Any, prefix: str) -> list[tuple[str, Any]]:
    if isinstance(value, dict) and value:
        return [
            child
            for key, nested in value.items()
            for child in _leaf_items(nested, f"{prefix}.{key}" if prefix else key)
        ]
    return [(prefix, value)]


def _validate_manifest(manifest: dict[str, Any], *, source: Path) -> None:
    validate_schema(manifest, "add-trace-run.schema.json", source=source)
    if manifest.get("schema_version") != "add-trace-run.v1":
        raise AddTraceError(f"{source}: expected schema_version='add-trace-run.v1'")
    for key in (
        "run_id",
        "model_id",
        "raw_config",
        "normalized_config",
        "raw_config_disposition",
    ):
        if key not in manifest:
            raise AddTraceError(f"{source}: missing required field {key!r}")

    normalized = manifest["normalized_config"]
    if not isinstance(normalized, dict):
        raise AddTraceError(f"{source}: normalized_config must be a mapping")
    for bucket in (
        "model_contract",
        "execution_contract",
        "runtime_implementation",
        "profile_contract",
        "capture_procedure",
    ):
        if not isinstance(normalized.get(bucket), dict):
            raise AddTraceError(
                f"{source}: normalized_config.{bucket} must be a mapping"
            )

    raw_config = manifest["raw_config"]
    if not isinstance(raw_config, dict) or not raw_config:
        raise AddTraceError(f"{source}: raw_config must be a non-empty mapping")
    dispositions = manifest["raw_config_disposition"]
    if not isinstance(dispositions, list) or not dispositions:
        raise AddTraceError(f"{source}: raw_config_disposition must be non-empty")
    raw_keys: set[str] = set()
    normalized_paths: dict[str, str] = {}
    allowed = {
        "model_contract",
        "execution_contract",
        "runtime_implementation",
        "profile_contract",
        "capture_procedure",
        "ignored",
    }
    raw_values = dict(_leaf_items(raw_config, ""))
    for index, item in enumerate(dispositions):
        if not isinstance(item, dict):
            raise AddTraceError(
                f"{source}: raw_config_disposition[{index}] must be a mapping"
            )
        raw_key = str(item.get("raw_key") or "")
        if not raw_key or raw_key in raw_keys:
            raise AddTraceError(
                f"{source}: raw config key {raw_key!r} is missing or repeated"
            )
        raw_keys.add(raw_key)
        if raw_key not in raw_values:
            raise AddTraceError(
                f"{source}: disposition references unknown raw config field {raw_key!r}"
            )
        if item.get("value") != raw_values[raw_key]:
            raise AddTraceError(
                f"{source}: disposition value for raw field {raw_key!r} does not match raw_config"
            )
        disposition = item.get("disposition")
        if disposition not in allowed:
            raise AddTraceError(
                f"{source}: {raw_key!r} has unknown disposition {disposition!r}"
            )
        if not item.get("evidence"):
            raise AddTraceError(f"{source}: {raw_key!r} requires extraction evidence")
        if disposition == "ignored":
            if not item.get("ignored_reason") or item.get("normalized_path"):
                raise AddTraceError(
                    f"{source}: ignored field {raw_key!r} requires only ignored_reason"
                )
            continue
        normalized_path = str(item.get("normalized_path") or "")
        expected_prefix = disposition + "."
        if not normalized_path.startswith(expected_prefix):
            raise AddTraceError(
                f"{source}: {raw_key!r} path {normalized_path!r} must start with "
                f"{expected_prefix!r}"
            )
        if normalized_path in normalized_paths:
            raise AddTraceError(
                f"{source}: raw fields {normalized_paths[normalized_path]!r} and "
                f"{raw_key!r} both own {normalized_path!r}"
            )
        normalized_paths[normalized_path] = raw_key
        relative_path = normalized_path.split(".", 1)[1]
        try:
            normalized_value = _value_at_path(normalized[disposition], relative_path)
        except KeyError as exc:
            raise AddTraceError(
                f"{source}: {raw_key!r} points to missing normalized field "
                f"{normalized_path!r}"
            ) from exc
        if normalized_value != item.get("normalized_value"):
            raise AddTraceError(
                f"{source}: {raw_key!r} normalized_value does not equal {normalized_path!r}"
            )

    missing_raw_dispositions = sorted(set(raw_values) - raw_keys)
    if missing_raw_dispositions:
        raise AddTraceError(
            f"{source}: raw config fields lack disposition: {missing_raw_dispositions}"
        )

    normalized_leaves = {
        path
        for bucket in (
            "model_contract",
            "execution_contract",
            "runtime_implementation",
            "profile_contract",
            "capture_procedure",
        )
        for path, _value in _leaf_items(normalized[bucket], bucket)
    }
    uncovered = sorted(
        leaf
        for leaf in normalized_leaves
        if not any(
            leaf == owner or leaf.startswith(owner + ".") for owner in normalized_paths
        )
    )
    if uncovered:
        raise AddTraceError(
            f"{source}: normalized config fields lack raw-config disposition: {uncovered}"
        )

    execution = normalized["execution_contract"]
    parallelism = execution.get("parallelism")
    if not isinstance(parallelism, dict):
        raise AddTraceError(
            f"{source}: execution_contract.parallelism must be a mapping"
        )
    for axis in ("tp_size", "dp_size", "cp_size", "ep_size", "pp_size"):
        value = parallelism.get(axis)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise AddTraceError(
                f"{source}: parallelism.{axis} must be a positive integer"
            )
    generation = execution.get("generation")
    if not isinstance(generation, dict) or not generation.get("mode"):
        raise AddTraceError(f"{source}: execution_contract.generation.mode is required")

    runtime_identity_payload(normalized["runtime_implementation"], source=source)
    procedure = normalized["capture_procedure"]
    if procedure.get("eager_cuda_graph_enabled") is not False:
        raise AddTraceError(
            f"{source}: eager reconciliation must run with CUDA Graph off"
        )


def resolve_model(
    manifest: dict[str, Any],
    *,
    model_root: Path,
    source: Path,
    catalog: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind the run to the independently authored model artifact and Model IR."""

    pipeline_path = model_root / "pipeline.yaml"
    pipeline = load_yaml(pipeline_path)
    source_lock = pipeline.get("source_lock")
    if not isinstance(source_lock, dict):
        raise AddTraceError(f"{pipeline_path}: source_lock must be a mapping")
    model_contract = manifest["normalized_config"]["model_contract"]
    expected = {
        "model_artifact_id": source_lock.get("model_id"),
        "model_revision": source_lock.get("model_revision"),
    }
    observed = {
        "model_artifact_id": model_contract.get("model_artifact_id"),
        "model_revision": model_contract.get("model_revision"),
    }
    if observed != expected:
        raise AddTraceError(
            canonical_json(
                {
                    "state": "new_model_ir_required",
                    "model_id": manifest["model_id"],
                    "expected_model_contract": expected,
                    "observed_model_contract": observed,
                }
            )
        )
    if catalog is None:
        catalog = compile_catalog(model_root)
    model_ir = catalog["model_ir"]
    return {
        "state": "matched_existing_model_ir",
        **observed,
        "semantic_revision": model_ir["semantic_revision"],
        "model_ir_sha256": sha256_json(model_ir),
    }


def runtime_identity_payload(
    identity: dict[str, Any], *, source: Path
) -> dict[str, Any]:
    required = (
        "framework_id",
        "source_repo",
        "source_commit",
        "container_digest",
        "package_lock_sha256",
        "extension_artifacts",
        "backend_selections",
        "build_flags",
    )
    missing = [field for field in required if field not in identity]
    if missing:
        raise AddTraceError(
            f"{source}: incomplete runtime identity; missing {', '.join(missing)}"
        )
    allowed = set(required) | {"source_patch_sha256"}
    unknown = sorted(set(identity) - allowed)
    if unknown:
        raise AddTraceError(
            f"{source}: runtime identity has non-identity fields: {unknown}"
        )
    if identity["framework_id"] not in {"sglang", "vllm", "tensorrt_llm"}:
        raise AddTraceError(f"{source}: unsupported framework_id")
    source_repo = urlsplit(str(identity["source_repo"]))
    if source_repo.scheme not in {"http", "https"} or not source_repo.netloc:
        raise AddTraceError(f"{source}: source_repo must be an absolute HTTP(S) URI")
    source_commit = str(identity["source_commit"])
    if len(source_commit) != 40 or any(
        character not in "0123456789abcdef" for character in source_commit
    ):
        raise AddTraceError(f"{source}: source_commit must be a full lowercase Git SHA")
    digest = str(identity["container_digest"])
    if (
        not digest.startswith("sha256:")
        or len(digest) != 71
        or any(character not in "0123456789abcdef" for character in digest[7:])
    ):
        raise AddTraceError(f"{source}: container_digest must be an immutable sha256")
    for key in ("package_lock_sha256", "source_patch_sha256"):
        value = identity.get(key)
        if value is not None and (
            len(str(value)) != 64
            or any(c not in "0123456789abcdef" for c in str(value))
        ):
            raise AddTraceError(f"{source}: {key} must be a lowercase SHA256")
    artifacts = identity["extension_artifacts"]
    if not isinstance(artifacts, list):
        raise AddTraceError(f"{source}: extension_artifacts must be an array")
    names: set[str] = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict) or set(artifact) != {"name", "sha256"}:
            raise AddTraceError(
                f"{source}: every extension artifact requires name and sha256"
            )
        if artifact["name"] in names:
            raise AddTraceError(
                f"{source}: duplicate extension artifact {artifact['name']!r}"
            )
        names.add(artifact["name"])
        artifact_digest = str(artifact["sha256"])
        if len(artifact_digest) != 64 or any(
            character not in "0123456789abcdef" for character in artifact_digest
        ):
            raise AddTraceError(f"{source}: invalid extension artifact SHA256")
    # Sort set-like fields before hashing. Function names, Python stacks, and
    # kernel names are intentionally absent: they are Binding content, not the
    # identity of the runtime artifact that must be rebound.
    payload = json.loads(canonical_json(identity))
    payload["extension_artifacts"] = sorted(
        payload["extension_artifacts"], key=lambda item: (item["name"], item["sha256"])
    )
    return payload


def runtime_identity_sha256(identity: dict[str, Any], *, source: Path) -> str:
    return sha256_json(runtime_identity_payload(identity, source=source))


def binding_revision_id(identity_digest: str, execution_fingerprint: str) -> str:
    return (
        "bind_"
        + hashlib.sha256(
            f"{execution_fingerprint}:{identity_digest}".encode()
        ).hexdigest()[:16]
    )


def mapping_rules_sha256(rules: list[dict[str, Any]]) -> str:
    """Hash the authored Binding rules independently of runtime identity.

    ``binding_revision_id`` selects the Execution/runtime pair before capture.
    This digest seals the actual mapping content produced by reconciliation, so
    a rule edit cannot silently preserve the same accepted Binding artifact.
    """

    normalized = json.loads(canonical_json(rules))
    for rule in normalized:
        layer_ids = rule.get("scope", {}).get("layer_ids")
        if layer_ids is not None:
            rule["scope"]["layer_ids"] = sorted(layer_ids)
    return sha256_json(sorted(normalized, key=lambda rule: rule["rule_id"]))


def _selector_matches(
    selector: dict[str, Any],
    *,
    framework_id: str,
    execution_contract: dict[str, Any],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if framework_id not in (selector.get("framework_ids") or []):
        reasons.append(f"framework_id={framework_id!r} is not allowed")
    match = selector.get("match")
    if not isinstance(match, dict) or not match:
        reasons.append("selector.match is missing")
        return False, reasons
    contract_paths = {
        path for path, _value in _leaf_items(execution_contract, "") if path
    }
    selector_paths = set(match)
    unconstrained = sorted(contract_paths - selector_paths)
    if unconstrained:
        reasons.append(
            "normalized execution fields are not constrained by this selector: "
            f"{unconstrained}"
        )
    for path, condition in sorted(match.items()):
        if not isinstance(condition, dict) or len(condition) != 1:
            reasons.append(f"{path}: selector condition must contain one operator")
            continue
        try:
            actual = _value_at_path(execution_contract, path)
        except KeyError:
            reasons.append(f"{path}: normalized config field is missing")
            continue
        operator, expected = next(iter(condition.items()))
        if operator == "equals" and actual != expected:
            reasons.append(f"{path}: expected {expected!r}, got {actual!r}")
        elif operator == "one_of" and actual not in expected:
            reasons.append(f"{path}: {actual!r} is not one of {expected!r}")
        elif operator == "minimum" and (
            not isinstance(actual, (int, float)) or actual < expected
        ):
            reasons.append(f"{path}: {actual!r} is below {expected!r}")
        elif operator == "maximum" and (
            not isinstance(actual, (int, float)) or actual > expected
        ):
            reasons.append(f"{path}: {actual!r} is above {expected!r}")
        elif operator not in {"equals", "one_of", "minimum", "maximum"}:
            reasons.append(f"{path}: unsupported selector operator {operator!r}")
    return not reasons, reasons


def resolve_execution(
    manifest: dict[str, Any],
    *,
    model_root: Path,
    source: Path,
    catalog: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _validate_manifest(manifest, source=source)
    if manifest["model_id"] != model_root.name:
        raise AddTraceError(
            f"{source}: model_id {manifest['model_id']!r} does not match {model_root.name!r}"
        )
    normalized = manifest["normalized_config"]
    execution_contract = normalized["execution_contract"]
    framework_id = normalized["runtime_implementation"]["framework_id"]
    candidates: list[str] = []
    diagnostics: dict[str, list[str]] = {}
    for path in sorted((model_root / "execution_paths").glob("*.yaml")):
        plan = load_yaml(path)
        matches, reasons = _selector_matches(
            plan.get("selector") or {},
            framework_id=framework_id,
            execution_contract=execution_contract,
        )
        plan_id = str(plan.get("execution_path_id") or path.stem)
        diagnostics[plan_id] = reasons
        if matches:
            candidates.append(plan_id)
    if len(candidates) != 1:
        state = "new_execution_required" if not candidates else "ambiguous_execution"
        raise AddTraceError(
            canonical_json(
                {
                    "state": state,
                    "model_id": manifest["model_id"],
                    "candidate_execution_paths": candidates,
                    "diagnostics": diagnostics,
                }
            )
        )
    bundle = catalog if catalog is not None else compile_catalog(model_root)
    plan_id = candidates[0]
    variant = next(
        item
        for item in bundle["execution_variants"].values()
        if item["execution_path_id"] == plan_id
    )
    return {
        "state": "matched_existing_execution",
        "execution_path_id": plan_id,
        "execution_fingerprint": variant["fingerprint"],
        "selector_diagnostics": diagnostics,
    }


def resolve_binding_revision(
    manifest: dict[str, Any],
    *,
    model_root: Path,
    execution: dict[str, Any],
    source: Path,
) -> dict[str, Any]:
    identity = manifest["normalized_config"]["runtime_implementation"]
    identity_digest = runtime_identity_sha256(identity, source=source)
    revision_id = binding_revision_id(
        identity_digest, execution["execution_fingerprint"]
    )
    matches: list[str] = []
    legacy_incomplete: list[str] = []
    for path in sorted((model_root / "bindings").glob("*.yaml")):
        binding = load_yaml(path)
        if binding.get("execution_path_id") != execution["execution_path_id"]:
            continue
        existing_identity = binding.get("runtime_identity")
        existing_digest = binding.get("runtime_identity_sha256")
        if not isinstance(existing_identity, dict) or not existing_digest:
            legacy_incomplete.append(binding["implementation_id"])
            continue
        calculated = runtime_identity_sha256(existing_identity, source=path)
        if calculated != existing_digest:
            raise AddTraceError(
                f"{path}: authored runtime_identity_sha256 does not match its payload"
            )
        if calculated == identity_digest:
            matches.append(binding["implementation_id"])
    if len(matches) > 1:
        raise AddTraceError(f"runtime identity matches multiple bindings: {matches}")
    return {
        "state": (
            "reuse_existing_binding" if matches else "new_binding_revision_required"
        ),
        "implementation_id": matches[0] if matches else None,
        "binding_revision_id": revision_id,
        "runtime_identity_sha256": identity_digest,
        "legacy_bindings_without_complete_identity": legacy_incomplete,
    }


def build_plan(
    manifest: dict[str, Any], *, model_root: Path, source: Path
) -> dict[str, Any]:
    _validate_manifest(manifest, source=source)
    catalog = compile_catalog(model_root)
    model = resolve_model(
        manifest, model_root=model_root, source=source, catalog=catalog
    )
    execution = resolve_execution(
        manifest, model_root=model_root, source=source, catalog=catalog
    )
    binding = resolve_binding_revision(
        manifest,
        model_root=model_root,
        execution=execution,
        source=source,
    )
    payload = {
        "schema_version": "add-trace-plan.v1",
        "run_id": manifest["run_id"],
        "model_id": manifest["model_id"],
        "manifest_sha256": sha256_json(manifest),
        "model_resolution": model,
        "execution_resolution": execution,
        "binding_resolution": binding,
        "required_stages": [
            "graph_off_eager_reconciliation",
            (
                "graph_on_production_attribution"
                if manifest["normalized_config"]["capture_procedure"][
                    "production_cuda_graph_enabled"
                ]
                else "production_attribution"
            ),
            "profile_materialization",
            "release_audit",
        ],
    }
    payload["plan_sha256"] = sha256_json(payload)
    validate_schema(payload, "add-trace-plan.schema.json", source=source)
    return payload


def _assert_same_identity(
    plan: dict[str, Any],
    binding_revision: dict[str, Any],
    reconciliation: dict[str, Any],
    attribution: dict[str, Any],
) -> None:
    expected = {
        "run_id": plan["run_id"],
        "model_id": plan["model_id"],
        "execution_path_id": plan["execution_resolution"]["execution_path_id"],
        "execution_fingerprint": plan["execution_resolution"]["execution_fingerprint"],
        "binding_revision_id": plan["binding_resolution"]["binding_revision_id"],
        "plan_sha256": plan["plan_sha256"],
    }
    for key, value in expected.items():
        binding_mismatch = (
            key not in {"run_id", "plan_sha256"} and binding_revision.get(key) != value
        )
        if (
            binding_mismatch
            or reconciliation.get(key) != value
            or attribution.get(key) != value
        ):
            raise AddTraceError(
                f"binding/eager/production identity mismatch for {key}: expected {value!r}"
            )


def _validate_rank_artifacts(
    artifacts: Any, *, tp_size: int, source: Path, verify_files: bool
) -> list[int]:
    if not isinstance(artifacts, list):
        raise AddTraceError(f"{source}: rank_artifacts must be an array")
    ranks = [item.get("rank") for item in artifacts if isinstance(item, dict)]
    if sorted(ranks) != list(range(tp_size)):
        raise AddTraceError(
            f"{source}: expected exact ranks {list(range(tp_size))}, got {sorted(ranks)}"
        )
    paths = [item.get("path") for item in artifacts if isinstance(item, dict)]
    if len(paths) != len(set(paths)):
        raise AddTraceError(f"{source}: rank artifact paths must be unique")
    for item in artifacts:
        artifact = Path(item["path"])
        if not artifact.is_absolute():
            artifact = source.resolve().parent / artifact
        expected_sha = item.get("sha256")
        if not isinstance(expected_sha, str) or len(expected_sha) != 64:
            raise AddTraceError(f"{source}: invalid rank artifact SHA256")
        if verify_files:
            if not artifact.is_file():
                raise AddTraceError(
                    f"{source}: rank artifact does not exist: {artifact}"
                )
            if sha256_file(artifact) != expected_sha:
                raise AddTraceError(f"{source}: rank artifact SHA mismatch: {artifact}")
    return ranks


def _validate_artifact(
    artifact_record: Any, *, source: Path, verify_files: bool
) -> str:
    if not isinstance(artifact_record, dict):
        raise AddTraceError(f"{source}: artifact record must be a mapping")
    artifact = Path(str(artifact_record.get("path") or ""))
    if not artifact.is_absolute():
        artifact = source.resolve().parent / artifact
    expected_sha = str(artifact_record.get("sha256") or "")
    if len(expected_sha) != 64:
        raise AddTraceError(f"{source}: invalid artifact SHA256")
    if verify_files:
        if not artifact.is_file():
            raise AddTraceError(f"{source}: artifact does not exist: {artifact}")
        if sha256_file(artifact) != expected_sha:
            raise AddTraceError(f"{source}: artifact SHA mismatch: {artifact}")
    return expected_sha


def _validate_artifacts(
    artifact_records: Any, *, source: Path, verify_files: bool
) -> str:
    if not isinstance(artifact_records, list) or not artifact_records:
        raise AddTraceError(
            f"{source}: runtime_evidence_artifacts must be a non-empty array"
        )
    paths = [
        str(record.get("path") or "")
        for record in artifact_records
        if isinstance(record, dict)
    ]
    if len(paths) != len(artifact_records) or any(not path for path in paths):
        raise AddTraceError(f"{source}: invalid runtime evidence artifact record")
    if len(paths) != len(set(paths)):
        raise AddTraceError(f"{source}: runtime evidence artifact paths must be unique")
    for record in artifact_records:
        _validate_artifact(record, source=source, verify_files=verify_files)
    return sha256_json(artifact_records)


def accept_evidence(
    manifest: dict[str, Any],
    plan: dict[str, Any],
    binding_revision: dict[str, Any],
    reconciliation: dict[str, Any],
    attribution: dict[str, Any],
    *,
    model_root: Path,
    source: Path,
    verify_files: bool = True,
) -> dict[str, Any]:
    _validate_manifest(manifest, source=source)
    validate_schema(plan, "add-trace-plan.schema.json", source=source)
    validate_schema(binding_revision, "binding-revision.schema.json", source=source)
    validate_schema(reconciliation, "binding-reconciliation.schema.json", source=source)
    validate_schema(attribution, "trace-attribution.schema.json", source=source)
    if sha256_json(manifest) != plan["manifest_sha256"]:
        raise AddTraceError("plan manifest digest does not match the supplied manifest")
    plan_payload = dict(plan)
    recorded_plan_sha = plan_payload.pop("plan_sha256")
    if sha256_json(plan_payload) != recorded_plan_sha:
        raise AddTraceError("plan digest does not match the supplied plan")
    _assert_same_identity(plan, binding_revision, reconciliation, attribution)
    if (
        binding_revision.get("binding_revision_id")
        != plan["binding_resolution"]["binding_revision_id"]
    ):
        raise AddTraceError(
            "binding revision does not match the planned runtime identity"
        )
    identity = binding_revision.get("runtime_identity") or {}
    digest = runtime_identity_sha256(identity, source=source)
    if (
        digest != binding_revision.get("runtime_identity_sha256")
        or digest != plan["binding_resolution"]["runtime_identity_sha256"]
    ):
        raise AddTraceError("binding runtime identity digest mismatch")
    if (
        binding_revision.get("execution_fingerprint")
        != plan["execution_resolution"]["execution_fingerprint"]
    ):
        raise AddTraceError("binding revision targets a different Execution IR")
    expected_revision = binding_revision_id(
        digest, plan["execution_resolution"]["execution_fingerprint"]
    )
    if binding_revision["binding_revision_id"] != expected_revision:
        raise AddTraceError("binding revision ID is not content-addressed correctly")
    rules = binding_revision.get("mapping_rules")
    if not isinstance(rules, list) or not rules:
        raise AddTraceError(
            "binding revision requires at least one deterministic mapping rule"
        )
    rule_ids = [rule.get("rule_id") for rule in rules if isinstance(rule, dict)]
    if len(rule_ids) != len(set(rule_ids)) or None in rule_ids:
        raise AddTraceError("binding mapping rule IDs must be unique and non-empty")
    rules_digest = mapping_rules_sha256(rules)
    if binding_revision.get("mapping_rules_sha256") != rules_digest:
        raise AddTraceError("binding mapping-rules digest mismatch")
    runtime_evidence_sha = _validate_artifacts(
        binding_revision.get("runtime_evidence_artifacts"),
        source=source,
        verify_files=verify_files,
    )
    rules_by_id = {rule["rule_id"]: rule for rule in rules}
    catalog = compile_catalog(model_root)
    if sha256_json(catalog["model_ir"]) != plan["model_resolution"]["model_ir_sha256"]:
        raise AddTraceError("planned Model IR is not the current compiled Model IR")
    fingerprint = plan["execution_resolution"]["execution_fingerprint"]
    variant = catalog["execution_variants"].get(fingerprint)
    if (
        variant is None
        or variant.get("execution_path_id")
        != plan["execution_resolution"]["execution_path_id"]
    ):
        raise AddTraceError(
            "planned Execution IR is not present in the current compiled catalog"
        )
    valid_ir_targets = {
        f"{view_id}.{node['id']}"
        for view_id, view in variant["views"].items()
        for node in view["nodes"]
    }
    unknown_ir_targets = sorted(
        {rule["ir_target"] for rule in rules} - valid_ir_targets
    )
    if unknown_ir_targets:
        raise AddTraceError(
            f"binding rules reference unknown compiled IR targets: {unknown_ir_targets}"
        )
    profile_phase = manifest["normalized_config"]["profile_contract"]["phase"]
    generation_mode = manifest["normalized_config"]["execution_contract"]["generation"][
        "mode"
    ]
    for rule in rules:
        if rule["scope"]["phase"] != profile_phase:
            raise AddTraceError(
                f"binding rule {rule['rule_id']!r} phase differs from the run profile"
            )
        if rule["scope"]["generation_mode"] != generation_mode:
            raise AddTraceError(
                f"binding rule {rule['rule_id']!r} generation mode differs from the Execution contract"
            )

    if (
        reconciliation.get("status") != "pass"
        or reconciliation.get("cuda_graph_enabled") is not False
    ):
        raise AddTraceError(
            "binding reconciliation must be passing graph-off eager evidence"
        )
    if reconciliation.get("unresolved") or reconciliation.get("discrepancies"):
        raise AddTraceError("eager reconciliation contains unresolved evidence")
    if attribution.get("status") != "pass" or attribution.get("unresolved"):
        raise AddTraceError("production attribution contains unresolved evidence")
    procedure = manifest["normalized_config"]["capture_procedure"]
    if (
        attribution.get("cuda_graph_enabled")
        is not procedure["production_cuda_graph_enabled"]
    ):
        raise AddTraceError("production CUDA Graph mode differs from the manifest")
    profile_contract = manifest["normalized_config"]["profile_contract"]
    if attribution.get("phase") != profile_contract["phase"]:
        raise AddTraceError("production phase differs from the manifest")

    tp_size = manifest["normalized_config"]["execution_contract"]["parallelism"][
        "tp_size"
    ]
    _validate_rank_artifacts(
        reconciliation.get("rank_artifacts"),
        tp_size=tp_size,
        source=source,
        verify_files=verify_files,
    )
    _validate_rank_artifacts(
        attribution.get("rank_artifacts"),
        tp_size=tp_size,
        source=source,
        verify_files=verify_files,
    )
    eager_protocol_sha = _validate_artifact(
        reconciliation.get("protocol_artifact"),
        source=source,
        verify_files=verify_files,
    )
    production_protocol_sha = _validate_artifact(
        attribution.get("protocol_artifact"),
        source=source,
        verify_files=verify_files,
    )
    if eager_protocol_sha == production_protocol_sha:
        raise AddTraceError(
            "eager and production evidence must reference distinct protocol artifacts"
        )
    window_sha = _validate_artifact(
        attribution.get("window_selection_artifact"),
        source=source,
        verify_files=verify_files,
    )

    results = reconciliation.get("rule_results") or []
    observed_rule_ids = {item.get("rule_id") for item in results}
    missing_rules = sorted(set(rule_ids) - observed_rule_ids)
    unknown_rules = sorted(observed_rule_ids - set(rule_ids))
    if missing_rules or unknown_rules:
        raise AddTraceError(
            f"eager mapping-rule closure failed; missing={missing_rules}, unknown={unknown_rules}"
        )
    observed_pairs = {(item.get("rule_id"), item.get("rank")) for item in results}
    expected_pairs = {
        (rule_id, rank) for rule_id in rule_ids for rank in range(tp_size)
    }
    if observed_pairs != expected_pairs or len(results) != len(expected_pairs):
        raise AddTraceError(
            "eager mapping rules must pass on every TP rank; "
            f"missing={sorted(expected_pairs - observed_pairs)}, "
            f"unexpected={sorted(observed_pairs - expected_pairs)}"
        )
    eager_ids_by_rule_rank: dict[tuple[str, int], set[str]] = {}
    eager_mapped_event_keys: list[tuple[int, str]] = []
    eager_mapped_duration = 0.0
    for result in results:
        rule = rules_by_id[result["rule_id"]]
        if result["ir_target"] != rule["ir_target"]:
            raise AddTraceError(
                f"eager result {result['rule_id']!r} targets {result['ir_target']!r}, "
                f"expected {rule['ir_target']!r}"
            )
        if result["matched_evidence"] != rule["eager_match"]:
            raise AddTraceError(
                f"eager result {result['rule_id']!r} does not prove its authored match predicate"
            )
        eager_ids_by_rule_rank[(result["rule_id"], result["rank"])] = set(
            result["eager_event_ids"]
        )
        eager_mapped_event_keys.extend(
            (result["rank"], event_id) for event_id in result["eager_event_ids"]
        )
        eager_mapped_duration += float(result["duration_us"])

    eager_support = reconciliation.get("support_events") or []
    eager_support_event_keys = [
        (item["rank"], item["event_id"]) for item in eager_support
    ]
    eager_event_keys = [*eager_mapped_event_keys, *eager_support_event_keys]
    if len(eager_event_keys) != len(set(eager_event_keys)):
        raise AddTraceError("eager event IDs must be unique within each TP rank")
    if any(rank not in range(tp_size) for rank, _event_id in eager_event_keys):
        raise AddTraceError("eager event uses a rank outside the Execution contract")
    if {rank for rank, _event_id in eager_event_keys} != set(range(tp_size)):
        raise AddTraceError("eager reconciliation must account for every TP rank")

    eager_coverage = reconciliation.get("coverage") or {}
    eager_event_count = len(eager_event_keys)
    eager_duration = eager_mapped_duration + sum(
        float(item["duration_us"]) for item in eager_support
    )
    if eager_coverage.get("event_count_total") != eager_coverage.get(
        "event_count_accounted"
    ):
        raise AddTraceError("eager event-count coverage is not closed")
    if eager_coverage.get("event_count_total") != eager_event_count:
        raise AddTraceError(
            "eager event-count summary does not match mapped and support records"
        )
    eager_total_duration = float(eager_coverage.get("duration_us_total") or 0.0)
    eager_accounted_duration = float(
        eager_coverage.get("duration_us_accounted") or 0.0
    )
    eager_tolerance = max(1e-6, eager_total_duration * 1e-9)
    if (
        abs(eager_total_duration - eager_accounted_duration) > eager_tolerance
        or abs(eager_total_duration - eager_duration) > eager_tolerance
    ):
        raise AddTraceError("eager duration coverage is not closed")

    event_mappings = attribution.get("event_mappings") or []
    production_rule_ids = {item.get("rule_id") for item in event_mappings}
    if not production_rule_ids.issubset(set(rule_ids)):
        raise AddTraceError(
            f"production attribution uses unknown rules: "
            f"{sorted(production_rule_ids - set(rule_ids))}"
        )
    for item in event_mappings:
        rule = rules_by_id[item["rule_id"]]
        if item["ir_target"] != rule["ir_target"]:
            raise AddTraceError(
                f"production event {item['event_id']!r} targets {item['ir_target']!r}, "
                f"expected {rule['ir_target']!r}"
            )
        if item["transfer_method"] != rule["production_transfer"]["method"]:
            raise AddTraceError(
                f"production event {item['event_id']!r} uses a transfer method "
                "different from its Binding rule"
            )
        if (
            item["transfer_signature_digest"]
            != rule["production_transfer"]["signature_digest"]
        ):
            raise AddTraceError(
                f"production event {item['event_id']!r} uses a transfer signature "
                "different from its Binding rule"
            )
        known_eager_ids = eager_ids_by_rule_rank.get(
            (item["rule_id"], item["rank"]), set()
        )
        if not set(item["eager_event_ids"]).issubset(known_eager_ids):
            raise AddTraceError("production transfer references unknown eager events")

    support_events = attribution.get("support_events") or []
    production_ranks = {item.get("rank") for item in [*event_mappings, *support_events]}
    if production_ranks != set(range(tp_size)):
        raise AddTraceError(
            f"production attribution must account for every TP rank; got {sorted(production_ranks)}"
        )

    all_events = [*event_mappings, *support_events]
    event_ids = [item["event_id"] for item in all_events]
    if len(event_ids) != len(set(event_ids)):
        raise AddTraceError("production event IDs must be globally unique")
    if any(item["rank"] not in range(tp_size) for item in all_events):
        raise AddTraceError(
            "production event uses a rank outside the Execution contract"
        )

    event_mappings_by_id = {item["event_id"]: item for item in event_mappings}
    fusion_groups = attribution.get("fusion_groups") or []
    group_ids = [group["group_id"] for group in fusion_groups]
    if len(group_ids) != len(set(group_ids)):
        raise AddTraceError("production fusion group IDs must be unique")
    fused_member_rules_by_rank: set[tuple[int, str]] = set()
    fused_event_ids: set[str] = set()
    binding_targets = {rule["ir_target"] for rule in rules}
    for group in fusion_groups:
        rank = group["rank"]
        owner_rule_id = group["owner_rule_id"]
        member_rule_ids = set(group["member_rule_ids"])
        owner = group["owner_ir_target"]
        members = set(group["member_ir_targets"])
        if rank not in range(tp_size):
            raise AddTraceError(
                "fusion group uses a rank outside the Execution contract"
            )
        if owner in members:
            raise AddTraceError("fusion group members must not repeat the timing owner")
        if owner_rule_id in member_rule_ids:
            raise AddTraceError("fusion group members must not repeat the owner rule")
        unknown_rule_ids = sorted(({owner_rule_id} | member_rule_ids) - set(rule_ids))
        if unknown_rule_ids:
            raise AddTraceError(
                f"fusion group references rules outside the Binding: {unknown_rule_ids}"
            )
        if rules_by_id[owner_rule_id]["ir_target"] != owner:
            raise AddTraceError("fusion group owner rule and IR target disagree")
        expected_member_targets = {
            rules_by_id[rule_id]["ir_target"] for rule_id in member_rule_ids
        }
        if expected_member_targets != members:
            raise AddTraceError("fusion group member rules and IR targets disagree")
        if any(
            rules_by_id[rule_id]["production_transfer"]["method"] != "reviewed_fusion"
            for rule_id in member_rule_ids
        ):
            raise AddTraceError(
                "every fused member rule must use reviewed_fusion transfer"
            )
        unknown_targets = sorted(({owner} | members) - binding_targets)
        if unknown_targets:
            raise AddTraceError(
                f"fusion group references targets outside the Binding: {unknown_targets}"
            )
        for event_id in group["event_ids"]:
            event = event_mappings_by_id.get(event_id)
            if (
                not event
                or event["rank"] != rank
                or event["ir_target"] != owner
                or event["rule_id"] != owner_rule_id
            ):
                raise AddTraceError(
                    f"fusion group event {event_id!r} is not timed by its declared owner"
                )
            if event_id in fused_event_ids:
                raise AddTraceError(
                    f"production event {event_id!r} belongs to multiple fusion groups"
                )
            fused_event_ids.add(event_id)
        for rule_id in member_rule_ids:
            key = (rank, rule_id)
            if key in fused_member_rules_by_rank:
                raise AddTraceError(
                    f"fusion member rule {rule_id!r} is owned by multiple groups on rank {rank}"
                )
            fused_member_rules_by_rank.add(key)

    direct_rules_by_rank = {(item["rank"], item["rule_id"]) for item in event_mappings}
    duplicated_fusion_ownership = direct_rules_by_rank & fused_member_rules_by_rank
    if duplicated_fusion_ownership:
        raise AddTraceError(
            "fused member rules must not also own direct production timing; "
            f"duplicates={sorted(duplicated_fusion_ownership)}"
        )
    expected_rules_by_rank = {
        (rank, rule["rule_id"]) for rank in range(tp_size) for rule in rules
    }
    accounted_rules_by_rank = direct_rules_by_rank | fused_member_rules_by_rank
    if accounted_rules_by_rank != expected_rules_by_rank:
        raise AddTraceError(
            "production Binding rule closure failed; "
            f"missing={sorted(expected_rules_by_rank - accounted_rules_by_rank)}, "
            f"unexpected={sorted(accounted_rules_by_rank - expected_rules_by_rank)}"
        )

    coverage = attribution.get("coverage") or {}
    event_count = len(event_mappings) + len(support_events)
    duration = sum(float(item.get("duration_us") or 0.0) for item in event_mappings)
    duration += sum(float(item.get("duration_us") or 0.0) for item in support_events)
    if coverage.get("event_count_total") != coverage.get("event_count_accounted"):
        raise AddTraceError("production event-count coverage is not closed")
    if coverage.get("event_count_total") != event_count:
        raise AddTraceError(
            "production event-count summary does not match event records"
        )
    total_duration = float(coverage.get("duration_us_total") or 0.0)
    accounted_duration = float(coverage.get("duration_us_accounted") or 0.0)
    tolerance = max(1e-6, total_duration * 1e-9)
    if (
        abs(total_duration - accounted_duration) > tolerance
        or abs(total_duration - duration) > tolerance
    ):
        raise AddTraceError("production duration coverage is not closed")

    accepted = {
        "schema_version": "add-trace-acceptance.v1",
        "run_id": plan["run_id"],
        "model_id": plan["model_id"],
        "model_revision": plan["model_resolution"]["model_revision"],
        "semantic_revision": plan["model_resolution"]["semantic_revision"],
        "model_ir_sha256": plan["model_resolution"]["model_ir_sha256"],
        "execution_path_id": plan["execution_resolution"]["execution_path_id"],
        "execution_fingerprint": plan["execution_resolution"]["execution_fingerprint"],
        "binding_revision_id": binding_revision["binding_revision_id"],
        "manifest_sha256": sha256_json(manifest),
        "plan_sha256": plan["plan_sha256"],
        "binding_revision_sha256": sha256_json(binding_revision),
        "eager_reconciliation_sha256": sha256_json(reconciliation),
        "trace_attribution_sha256": sha256_json(attribution),
        "runtime_identity_sha256": digest,
        "runtime_evidence_sha256": runtime_evidence_sha,
        "mapping_rules_sha256": rules_digest,
        "eager_protocol_sha256": eager_protocol_sha,
        "production_protocol_sha256": production_protocol_sha,
        "window_selection_sha256": window_sha,
        "rank_count": tp_size,
        "eager_rule_count": len(rule_ids),
        "eager_event_count": eager_event_count,
        "eager_duration_us": eager_total_duration,
        "production_event_count": event_count,
        "production_duration_us": total_duration,
        "status": "pass",
    }
    accepted["acceptance_sha256"] = sha256_json(accepted)
    validate_schema(accepted, "add-trace-acceptance.schema.json", source=source)
    return accepted
