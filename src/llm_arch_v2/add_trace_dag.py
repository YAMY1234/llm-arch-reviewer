"""Resumable, content-addressed orchestration for one add-trace run.

The DAG automates deterministic validation and artifact materialization.  It
does not invent a Binding from kernel names or traces: graph-off Binding,
graph-on attribution, and the catalog Profile remain explicit evidence inputs.
Missing inputs produce a typed ``needs_input`` state that can be resumed by
running the same command again.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

from .add_trace import (
    AddTraceError,
    accept_evidence,
    build_plan,
    canonical_json,
    sha256_file,
    sha256_json,
    validate_schema,
)
from .compiler import compile_catalog, write_bundle


STAGE_CONTRACT_VERSION = "add-trace-dag-stage.v1"


def _load_mapping(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise AddTraceError(f"{path}: expected a YAML/JSON mapping")
    return value


def _leaf_count(value: Any) -> int:
    if isinstance(value, dict) and value:
        return sum(_leaf_count(child) for child in value.values())
    if isinstance(value, list) and value:
        return sum(_leaf_count(child) for child in value)
    return 1


def _stable_locator(path: Path, *, run_root: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    for prefix, root in (("run", run_root.resolve()), ("repo", repo_root.resolve())):
        try:
            return f"{prefix}://{resolved.relative_to(root).as_posix()}"
        except ValueError:
            pass
    return f"file://{resolved.as_posix()}"


def _pretty_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _store_artifact(
    workspace: Path,
    *,
    stage: str,
    document: dict[str, Any],
    dependencies: list[dict[str, Any]],
    inputs: dict[str, str],
) -> tuple[dict[str, Any], bool]:
    payload = _pretty_json(document)
    document_sha256 = hashlib.sha256(payload).hexdigest()
    content_id = sha256_json(
        {
            "stage_contract_version": STAGE_CONTRACT_VERSION,
            "stage": stage,
            "dependencies": [
                {
                    "name": dependency["name"],
                    "artifact_sha256": dependency["artifact"]["sha256"],
                }
                for dependency in dependencies
            ],
            "inputs": inputs,
            "document_sha256": document_sha256,
        }
    )
    relative = Path("artifacts") / stage / f"{content_id}.json"
    destination = workspace / relative
    cache_hit = destination.is_file()
    if cache_hit:
        existing = destination.read_bytes()
        if existing != payload:
            raise AddTraceError(
                f"content-address collision or tampering at {destination}"
            )
    else:
        _atomic_write(destination, payload)
    return (
        {
            "name": stage,
            "kind": "computed",
            "status": "complete",
            "dependencies": [item["name"] for item in dependencies],
            "content_id": content_id,
            "artifact": {
                "path": relative.as_posix(),
                "sha256": document_sha256,
            },
        },
        cache_hit,
    )


def _ingest_document(
    workspace: Path,
    *,
    stage: str,
    path: Path,
    schema_name: str,
    dependencies: list[dict[str, Any]],
    run_root: Path,
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    document = _load_mapping(path)
    validate_schema(document, schema_name, source=path)
    source_sha256 = sha256_file(path)
    record, cache_hit = _store_artifact(
        workspace,
        stage=stage,
        document=document,
        dependencies=dependencies,
        inputs={"source_sha256": source_sha256},
    )
    record["kind"] = "evidence"
    record["source"] = {
        "locator": _stable_locator(path, run_root=run_root, repo_root=repo_root),
        "sha256": source_sha256,
    }
    return document, record, cache_hit


def _blocked_stage(
    name: str, *, kind: str, dependencies: list[str], reason: str
) -> dict[str, Any]:
    return {
        "name": name,
        "kind": kind,
        "status": "blocked",
        "dependencies": dependencies,
        "reason": reason,
    }


def _missing_stage(
    name: str,
    *,
    option: str,
    contract: str,
    dependencies: list[str],
    reason: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        {
            "name": name,
            "kind": "evidence" if name != "release_audit" else "gate",
            "status": "needs_input",
            "dependencies": dependencies,
            "reason": reason,
        },
        {
            "stage": name,
            "required_option": option,
            "expected_contract": contract,
            "reason": reason,
        },
    )


def _matching_profile_paths(
    *,
    model_root: Path,
    acceptance: dict[str, Any],
    manifest: dict[str, Any],
) -> list[Path]:
    implementation_ids: set[str] = set()
    for path in sorted((model_root / "bindings").glob("*.yaml")):
        binding = _load_mapping(path)
        if (
            binding.get("binding_revision_id") == acceptance["binding_revision_id"]
            and binding.get("add_trace_acceptance_sha256")
            == acceptance["acceptance_sha256"]
        ):
            implementation_ids.add(str(binding.get("implementation_id") or ""))
    profile_contract = manifest["normalized_config"]["profile_contract"]
    generation_mode = manifest["normalized_config"]["execution_contract"][
        "generation"
    ]["mode"]
    matches: list[Path] = []
    for path in sorted((model_root / "profiles").glob("*/*/*.yaml")):
        profile = _load_mapping(path)
        workload = profile.get("workload") or {}
        if (
            profile.get("implementation_id") in implementation_ids
            and profile.get("execution_path_id") == acceptance["execution_path_id"]
            and profile.get("phase") == profile_contract["phase"]
            and profile.get("generation_mode", "autoregressive") == generation_mode
            and workload.get("batch_size") == profile_contract.get("batch_size")
            and (
                profile_contract.get("isl") is None
                or workload.get("isl") == profile_contract.get("isl")
            )
            and (
                profile_contract.get("osl") is None
                or workload.get("osl") == profile_contract.get("osl")
            )
        ):
            matches.append(path)
    return matches


def validate_profile_materialization(
    profile: dict[str, Any],
    *,
    profile_path: Path,
    acceptance: dict[str, Any],
    manifest: dict[str, Any],
    model_root: Path,
) -> dict[str, Any]:
    """Prove that one accepted evidence set is materialized in the catalog."""

    validate_schema(profile, "profile.schema.json", source=profile_path)
    expected = {
        "model_id": manifest["model_id"],
        "execution_path_id": acceptance["execution_path_id"],
        "phase": manifest["normalized_config"]["profile_contract"]["phase"],
        "generation_mode": manifest["normalized_config"]["execution_contract"][
            "generation"
        ]["mode"],
    }
    observed = {key: profile.get(key) for key in expected}
    observed["generation_mode"] = profile.get("generation_mode", "autoregressive")
    if observed != expected:
        raise AddTraceError(
            f"{profile_path}: Profile contract differs from accepted run: "
            f"{canonical_json({'expected': expected, 'observed': observed})}"
        )
    execution_parameters = profile.get("execution_parameters") or {}
    expected_parallelism = manifest["normalized_config"]["execution_contract"][
        "parallelism"
    ]
    for axis in ("tp_size", "dp_size", "cp_size", "ep_size"):
        if execution_parameters.get(axis) != expected_parallelism[axis]:
            raise AddTraceError(
                f"{profile_path}: Profile {axis} differs from accepted run"
            )
    workload = profile.get("workload") or {}
    profile_contract = manifest["normalized_config"]["profile_contract"]
    for profile_key, contract_key in (
        ("batch_size", "batch_size"),
        ("isl", "isl"),
        ("osl", "osl"),
    ):
        if contract_key in profile_contract and workload.get(profile_key) != profile_contract[
            contract_key
        ]:
            raise AddTraceError(
                f"{profile_path}: workload.{profile_key} differs from accepted run"
            )

    catalog = compile_catalog(model_root)
    profile_id = str(profile["profile_id"])
    if profile_id not in catalog["profiles"]:
        raise AddTraceError(f"{profile_path}: Profile is not present in compiled catalog")
    implementation_id = str(profile["implementation_id"])
    implementation = catalog["implementations"].get(implementation_id)
    if not implementation:
        raise AddTraceError(
            f"{profile_path}: implementation {implementation_id!r} is not compiled"
        )
    binding_expected = {
        "execution_variant": acceptance["execution_fingerprint"],
        "binding_revision_id": acceptance["binding_revision_id"],
        "add_trace_acceptance_sha256": acceptance["acceptance_sha256"],
        "runtime_identity_sha256": acceptance["runtime_identity_sha256"],
        "mapping_rules_sha256": acceptance["mapping_rules_sha256"],
    }
    binding_observed = {key: implementation.get(key) for key in binding_expected}
    if binding_observed != binding_expected:
        raise AddTraceError(
            f"{profile_path}: compiled Binding does not carry the exact acceptance "
            f"authority: {canonical_json({'expected': binding_expected, 'observed': binding_observed})}"
        )
    timeline = profile.get("timeline") or {}
    timeline_path = profile_path.parent / str(timeline.get("artifact") or "")
    if not timeline_path.is_file():
        raise AddTraceError(f"{profile_path}: missing timeline artifact {timeline_path}")
    if sha256_file(timeline_path) != timeline.get("sha256"):
        raise AddTraceError(f"{profile_path}: timeline artifact SHA256 mismatch")
    return {
        "profile_id": profile_id,
        "implementation_id": implementation_id,
        "execution_fingerprint": acceptance["execution_fingerprint"],
        "binding_revision_id": acceptance["binding_revision_id"],
        "add_trace_acceptance_sha256": acceptance["acceptance_sha256"],
        "timeline_sha256": timeline["sha256"],
        "timeline_event_count": timeline["event_count"],
        "timeline_step_count": timeline["step_count"],
    }


def validate_release_materialization(
    release_report: dict[str, Any],
    *,
    model_id: str,
    compiled_bundle: dict[str, Any],
    release_level: str,
    source: Path,
) -> dict[str, Any]:
    """Prove that a release report audited this exact compiled bundle."""

    if release_report.get("schema_version") != "release-audit.v1":
        raise AddTraceError(f"{source}: expected release-audit.v1")
    if release_report.get("acceptance_level") != release_level:
        raise AddTraceError(
            f"{source}: release level differs from requested {release_level}"
        )
    if release_report.get("static_gate") != "pass":
        raise AddTraceError(f"{source}: static release gate failed")
    if release_level == "release" and (
        release_report.get("browser_gate") != "pass"
        or release_report.get("release_ready") is not True
    ):
        raise AddTraceError(f"{source}: browser release gate failed")
    model_reports = {
        item.get("model"): item for item in release_report.get("models") or []
    }
    model_report = model_reports.get(model_id)
    if not model_report or model_report.get("status") != "pass":
        raise AddTraceError(f"{source}: target model did not pass release audit")
    expected_bundle_sha256 = sha256_json(compiled_bundle)
    bundle_report = model_report.get("bundle") or {}
    observed_bundle_contract = {
        "matches_compiler": bundle_report.get("matches_compiler"),
        "compiled_sha256": bundle_report.get("compiled_sha256"),
        "published_sha256": bundle_report.get("published_sha256"),
    }
    expected_bundle_contract = {
        "matches_compiler": True,
        "compiled_sha256": expected_bundle_sha256,
        "published_sha256": expected_bundle_sha256,
    }
    if observed_bundle_contract != expected_bundle_contract:
        raise AddTraceError(
            f"{source}: release report does not cover the exact current bundle: "
            + canonical_json(
                {
                    "expected": expected_bundle_contract,
                    "observed": observed_bundle_contract,
                }
            )
        )
    return {
        "acceptance_level": release_level,
        "static_gate": "pass",
        "browser_gate": release_report.get("browser_gate"),
        "release_ready": bool(release_report.get("release_ready")),
        "bundle_sha256": expected_bundle_sha256,
    }


def _review_packet(
    *,
    manifest: dict[str, Any],
    plan: dict[str, Any],
    stages: list[dict[str, Any]],
    next_actions: list[dict[str, Any]],
    acceptance: dict[str, Any] | None,
    materialization: dict[str, Any] | None,
    release_report: dict[str, Any] | None,
    release_level: str,
) -> dict[str, Any]:
    complete = not next_actions and all(
        stage["status"] == "complete" for stage in stages
    )
    release_ready = bool(release_report and release_report.get("release_ready"))
    if complete and release_ready:
        decision = "accepted_for_release"
    elif complete and release_level == "static":
        decision = "static_pass_browser_required"
    else:
        decision = "hold_for_evidence"
    risks: list[str] = []
    legacy = plan["binding_resolution"].get(
        "legacy_bindings_without_complete_identity"
    ) or []
    if legacy:
        risks.append(f"legacy bindings lack complete runtime identity: {', '.join(legacy)}")
    if next_actions:
        risks.append("required evidence or gate artifacts remain unresolved")
    if complete and release_level == "static":
        risks.append("real-browser release gate has not been evaluated")
    normalized = manifest["normalized_config"]
    packet: dict[str, Any] = {
        "schema_version": "add-trace-review-packet.v1",
        "run_id": manifest["run_id"],
        "model_id": manifest["model_id"],
        "status": "pass" if complete else "needs_input",
        "decision": decision,
        "identity": {
            "model_artifact_id": plan["model_resolution"]["model_artifact_id"],
            "model_revision": plan["model_resolution"]["model_revision"],
            "semantic_revision": plan["model_resolution"]["semantic_revision"],
            "model_ir_sha256": plan["model_resolution"]["model_ir_sha256"],
            "execution_path_id": plan["execution_resolution"]["execution_path_id"],
            "execution_fingerprint": plan["execution_resolution"][
                "execution_fingerprint"
            ],
            "binding_state": plan["binding_resolution"]["state"],
            "binding_revision_id": plan["binding_resolution"][
                "binding_revision_id"
            ],
            "runtime_identity_sha256": plan["binding_resolution"][
                "runtime_identity_sha256"
            ],
        },
        "configuration_closure": {
            "raw_field_count": len(manifest["raw_config"]),
            "raw_leaf_count": _leaf_count(manifest["raw_config"]),
            "disposition_count": len(manifest["raw_config_disposition"]),
            "normalized_leaf_count": _leaf_count(normalized),
            "ignored_fields": sorted(
                item["raw_key"]
                for item in manifest["raw_config_disposition"]
                if item["disposition"] == "ignored"
            ),
        },
        "stage_ledger": [
            {
                key: stage[key]
                for key in (
                    "name",
                    "kind",
                    "status",
                    "dependencies",
                    "content_id",
                    "artifact",
                    "source",
                    "reason",
                )
                if key in stage
            }
            for stage in stages
        ],
        "evidence_closure": (
            {
                key: acceptance[key]
                for key in (
                    "acceptance_sha256",
                    "rank_count",
                    "eager_rule_count",
                    "eager_event_count",
                    "eager_duration_us",
                    "production_event_count",
                    "production_duration_us",
                    "production_captured_at",
                    "mapping_rules_sha256",
                    "eager_protocol_sha256",
                    "production_protocol_sha256",
                    "window_selection_sha256",
                    "capture_time_artifact_sha256",
                )
            }
            if acceptance
            else None
        ),
        "materialization": materialization,
        "release_gate": (
            {
                "acceptance_level": release_report.get("acceptance_level"),
                "static_gate": release_report.get("static_gate"),
                "browser_gate": release_report.get("browser_gate"),
                "release_ready": release_report.get("release_ready"),
                "bundle_sha256": next(
                    item["bundle"]["compiled_sha256"]
                    for item in release_report["models"]
                    if item.get("model") == manifest["model_id"]
                ),
            }
            if release_report
            else None
        ),
        "risks": risks,
        "next_actions": next_actions,
    }
    packet["review_packet_sha256"] = sha256_json(packet)
    return packet


def validate_review_packet(packet: dict[str, Any], *, source: Path) -> None:
    """Validate both the packet schema and its self-addressed payload."""

    validate_schema(
        packet, "add-trace-review-packet.schema.json", source=source
    )
    config = packet["configuration_closure"]
    if config["raw_field_count"] != config["disposition_count"]:
        raise AddTraceError(
            f"{source}: review packet does not account for every raw configuration field"
        )
    stage_names = [stage["name"] for stage in packet["stage_ledger"]]
    if len(stage_names) != len(set(stage_names)):
        raise AddTraceError(f"{source}: review packet repeats a stage name")
    known: set[str] = set()
    for stage in packet["stage_ledger"]:
        unknown = set(stage["dependencies"]) - known
        if unknown:
            raise AddTraceError(
                f"{source}: stage {stage['name']} has unresolved dependencies {sorted(unknown)}"
            )
        known.add(stage["name"])
    payload = dict(packet)
    recorded = payload.pop("review_packet_sha256")
    if sha256_json(payload) != recorded:
        raise AddTraceError(f"{source}: review packet SHA256 does not match payload")


def render_review_markdown(packet: dict[str, Any]) -> str:
    identity = packet["identity"]
    lines = [
        f"# Add-trace review: `{packet['run_id']}`",
        "",
        f"**Decision:** `{packet['decision']}`  ",
        f"**Status:** `{packet['status']}`",
        "",
        "## Stable identities",
        "",
        "| Contract | Identity |",
        "|---|---|",
        f"| Model artifact | `{identity['model_artifact_id']}@{identity['model_revision']}` |",
        f"| Model IR | semantic revision `{identity['semantic_revision']}` / `{identity['model_ir_sha256']}` |",
        f"| Execution IR | `{identity['execution_path_id']}` / `{identity['execution_fingerprint']}` |",
        f"| Runtime Binding | `{identity['binding_revision_id']}` / `{identity['runtime_identity_sha256']}` |",
        "",
        "## Stage ledger",
        "",
        "| Stage | Depends on | Kind | Status | Artifact SHA256 |",
        "|---|---|---|---|---|",
    ]
    for stage in packet["stage_ledger"]:
        artifact_sha = (stage.get("artifact") or {}).get("sha256", "—")
        dependencies = ", ".join(stage.get("dependencies") or []) or "—"
        lines.append(
            f"| `{stage['name']}` | {dependencies} | {stage['kind']} | `{stage['status']}` | `{artifact_sha}` |"
        )
    config = packet["configuration_closure"]
    lines.extend(
        [
            "",
            "## Configuration closure",
            "",
            f"- Raw top-level fields: **{config['raw_field_count']}**",
            f"- Raw leaf values: **{config['raw_leaf_count']}**",
            f"- Explicit field dispositions: **{config['disposition_count']}**",
            f"- Normalized leaf values: **{config['normalized_leaf_count']}**",
            "- Intentionally ignored fields: "
            + (", ".join(f"`{item}`" for item in config["ignored_fields"]) or "none"),
        ]
    )
    closure = packet.get("evidence_closure")
    if closure:
        lines.extend(
            [
                "",
                "## Eager → production closure",
                "",
                f"- TP ranks: **{closure['rank_count']}**",
                f"- Eager: **{closure['eager_event_count']}** events across **{closure['eager_rule_count']}** rules; **{closure['eager_duration_us']:.6f} μs**",
                f"- Production: **{closure['production_event_count']}** events; **{closure['production_duration_us']:.6f} μs**",
                f"- Capture time: `{closure['production_captured_at']}`",
                f"- Capture-time authority: `{closure['capture_time_artifact_sha256']}`",
                f"- Acceptance: `{closure['acceptance_sha256']}`",
            ]
        )
    materialization = packet.get("materialization")
    if materialization:
        lines.extend(
            [
                "",
                "## Catalog materialization",
                "",
                f"- Profile: `{materialization['profile_id']}`",
                f"- Implementation: `{materialization['implementation_id']}`",
                f"- Timeline: `{materialization['timeline_sha256']}` ({materialization['timeline_event_count']} events / {materialization['timeline_step_count']} steps)",
            ]
        )
    gate = packet.get("release_gate")
    if gate:
        lines.extend(
            [
                "",
                "## Release gate",
                "",
                f"- Level: `{gate['acceptance_level']}`",
                f"- Static: `{gate['static_gate']}`",
                f"- Browser: `{gate['browser_gate']}`",
                f"- Release ready: `{str(bool(gate['release_ready'])).lower()}`",
                f"- Audited bundle: `{gate['bundle_sha256']}`",
            ]
        )
    if packet["risks"]:
        lines.extend(["", "## Remaining risks", ""])
        lines.extend(f"- {risk}" for risk in packet["risks"])
    if packet["next_actions"]:
        lines.extend(["", "## Required next actions", ""])
        lines.extend(
            f"- `{action['stage']}`: supply `{action['required_option']}` ({action['expected_contract']}) — {action['reason']}"
            for action in packet["next_actions"]
        )
    lines.extend(
        [
            "",
            f"Packet SHA256: `{packet['review_packet_sha256']}`",
            "",
        ]
    )
    return "\n".join(lines)


def run_add_trace_dag(
    *,
    manifest_path: Path,
    workspace: Path,
    catalog_root: Path,
    repo_root: Path,
    binding_revision_path: Path | None = None,
    eager_reconciliation_path: Path | None = None,
    trace_attribution_path: Path | None = None,
    profile_path: Path | None = None,
    release_report_path: Path | None = None,
    release_level: str = "static",
    verify_files: bool = True,
) -> dict[str, Any]:
    """Run every currently satisfiable stage and persist resumable state."""

    if release_level not in {"static", "release"}:
        raise AddTraceError(f"unsupported release level {release_level!r}")
    manifest_path = manifest_path.resolve()
    workspace = workspace.resolve()
    catalog_root = catalog_root.resolve()
    repo_root = repo_root.resolve()
    run_root = manifest_path.parent
    manifest = _load_mapping(manifest_path)
    manifest_file_sha256 = sha256_file(manifest_path)
    previous_state_path = workspace / "run-state.json"
    if previous_state_path.is_file():
        previous = _load_mapping(previous_state_path)
        if (
            previous.get("run_id") != manifest.get("run_id")
            or previous.get("model_id") != manifest.get("model_id")
            or previous.get(
                "manifest_file_sha256", previous.get("manifest_sha256")
            )
            != manifest_file_sha256
        ):
            raise AddTraceError(
                f"{workspace}: workspace belongs to a different immutable run"
            )

    model_root = catalog_root / str(manifest.get("model_id") or "")
    stages: list[dict[str, Any]] = []
    cache_hits: list[str] = []
    next_actions: list[dict[str, Any]] = []

    plan = build_plan(manifest, model_root=model_root, source=manifest_path)
    plan_stage, cache_hit = _store_artifact(
        workspace,
        stage="plan",
        document=plan,
        dependencies=[],
        inputs={"manifest_file_sha256": manifest_file_sha256},
    )
    if cache_hit:
        cache_hits.append("plan")
    stages.append(plan_stage)

    input_specs = [
        (
            "binding_revision",
            binding_revision_path,
            "binding-revision.schema.json",
            "--binding-revision",
            "binding-revision.v1",
        ),
        (
            "graph_off_eager_reconciliation",
            eager_reconciliation_path,
            "binding-reconciliation.schema.json",
            "--eager-reconciliation",
            "binding-reconciliation.v1 (CUDA Graph off, all ranks)",
        ),
        (
            "graph_on_production_attribution",
            trace_attribution_path,
            "trace-attribution.schema.json",
            "--trace-attribution",
            "trace-attribution.v1 (production mode, all ranks)",
        ),
    ]
    evidence: dict[str, dict[str, Any]] = {}
    evidence_stages: dict[str, dict[str, Any]] = {}
    for name, path, schema_name, option, contract in input_specs:
        dependencies = [plan_stage]
        if path is None:
            stage, action = _missing_stage(
                name,
                option=option,
                contract=contract,
                dependencies=["plan"],
                reason=f"{name} is independently authored evidence and cannot be inferred",
            )
            stages.append(stage)
            next_actions.append(action)
            continue
        document, stage, cache_hit = _ingest_document(
            workspace,
            stage=name,
            path=path.resolve(),
            schema_name=schema_name,
            dependencies=dependencies,
            run_root=run_root,
            repo_root=repo_root,
        )
        evidence[name] = document
        evidence_stages[name] = stage
        if cache_hit:
            cache_hits.append(name)
        stages.append(stage)

    acceptance: dict[str, Any] | None = None
    acceptance_stage: dict[str, Any] | None = None
    if len(evidence) == len(input_specs):
        acceptance = accept_evidence(
            manifest,
            plan,
            evidence["binding_revision"],
            evidence["graph_off_eager_reconciliation"],
            evidence["graph_on_production_attribution"],
            model_root=model_root,
            source=manifest_path,
            verify_files=verify_files,
        )
        acceptance_dependencies = [
            plan_stage,
            evidence_stages["binding_revision"],
            evidence_stages["graph_off_eager_reconciliation"],
            evidence_stages["graph_on_production_attribution"],
        ]
        acceptance_stage, cache_hit = _store_artifact(
            workspace,
            stage="acceptance",
            document=acceptance,
            dependencies=acceptance_dependencies,
            inputs={},
        )
        if cache_hit:
            cache_hits.append("acceptance")
        stages.append(acceptance_stage)
    else:
        stages.append(
            _blocked_stage(
                "acceptance",
                kind="computed",
                dependencies=[name for name, *_rest in input_specs],
                reason="binding, eager reconciliation, and production attribution must all close",
            )
        )

    materialization: dict[str, Any] | None = None
    profile_stage: dict[str, Any] | None = None
    resolved_profile_path = profile_path.resolve() if profile_path else None
    if acceptance:
        if resolved_profile_path is None:
            matches = _matching_profile_paths(
                model_root=model_root,
                acceptance=acceptance,
                manifest=manifest,
            )
            if len(matches) == 1:
                resolved_profile_path = matches[0].resolve()
            elif len(matches) > 1:
                raise AddTraceError(
                    "multiple Profiles materialize this acceptance; --profile is required: "
                    + canonical_json([str(path) for path in matches])
                )
        if resolved_profile_path is None:
            stage, action = _missing_stage(
                "profile_materialization",
                option="--profile",
                contract="profile.v2 plus exact accepted Binding and timeline",
                dependencies=["acceptance"],
                reason="no unique catalog Profile carries this exact acceptance authority",
            )
            stages.append(stage)
            next_actions.append(action)
        else:
            profile = _load_mapping(resolved_profile_path)
            materialization = validate_profile_materialization(
                profile,
                profile_path=resolved_profile_path,
                acceptance=acceptance,
                manifest=manifest,
                model_root=model_root,
            )
            document = {
                "schema_version": "add-trace-profile-materialization.v1",
                **materialization,
                "profile_source_sha256": sha256_file(resolved_profile_path),
            }
            profile_stage, cache_hit = _store_artifact(
                workspace,
                stage="profile_materialization",
                document=document,
                dependencies=[acceptance_stage],
                inputs={"profile_source_sha256": sha256_file(resolved_profile_path)},
            )
            profile_stage["source"] = {
                "locator": _stable_locator(
                    resolved_profile_path, run_root=run_root, repo_root=repo_root
                ),
                "sha256": sha256_file(resolved_profile_path),
            }
            if cache_hit:
                cache_hits.append("profile_materialization")
            stages.append(profile_stage)
    else:
        stages.append(
            _blocked_stage(
                "profile_materialization",
                kind="computed",
                dependencies=["acceptance"],
                reason="acceptance must pass before catalog materialization is trusted",
            )
        )

    compiled_bundle: dict[str, Any] | None = None
    bundle_stage: dict[str, Any] | None = None
    if profile_stage:
        compiled_bundle = compile_catalog(model_root)
        bundle_stage, cache_hit = _store_artifact(
            workspace,
            stage="bundle_build",
            document=compiled_bundle,
            dependencies=[profile_stage],
            inputs={},
        )
        if cache_hit:
            cache_hits.append("bundle_build")
        # Also expose the compiler-native formatting without changing the catalog.
        write_bundle(compiled_bundle, workspace / "compiled" / "arch_data.json")
        stages.append(bundle_stage)
    else:
        stages.append(
            _blocked_stage(
                "bundle_build",
                kind="computed",
                dependencies=["profile_materialization"],
                reason="the exact Profile and Binding must be materialized first",
            )
        )

    release_report: dict[str, Any] | None = None
    release_stage: dict[str, Any] | None = None
    if bundle_stage:
        if release_report_path is None:
            stage, action = _missing_stage(
                "release_audit",
                option="--release-report",
                contract=f"release-audit.v1 ({release_level})",
                dependencies=["bundle_build"],
                reason="the unified static/browser release gate has not run",
            )
            stages.append(stage)
            next_actions.append(action)
        else:
            release_report_path = release_report_path.resolve()
            release_report = _load_mapping(release_report_path)
            validate_release_materialization(
                release_report,
                model_id=manifest["model_id"],
                compiled_bundle=compiled_bundle,
                release_level=release_level,
                source=release_report_path,
            )
            release_stage, cache_hit = _store_artifact(
                workspace,
                stage="release_audit",
                document=release_report,
                dependencies=[bundle_stage],
                inputs={"release_report_sha256": sha256_file(release_report_path)},
            )
            release_stage["kind"] = "gate"
            release_stage["source"] = {
                "locator": _stable_locator(
                    release_report_path, run_root=run_root, repo_root=repo_root
                ),
                "sha256": sha256_file(release_report_path),
            }
            if cache_hit:
                cache_hits.append("release_audit")
            stages.append(release_stage)
    else:
        stages.append(
            _blocked_stage(
                "release_audit",
                kind="gate",
                dependencies=["bundle_build"],
                reason="bundle build must complete before release audit",
            )
        )

    packet = _review_packet(
        manifest=manifest,
        plan=plan,
        stages=stages,
        next_actions=next_actions,
        acceptance=acceptance,
        materialization=materialization,
        release_report=release_report,
        release_level=release_level,
    )
    validate_review_packet(packet, source=manifest_path)
    packet_dependencies = [
        stage for stage in stages if stage.get("status") == "complete"
    ]
    packet_stage, cache_hit = _store_artifact(
        workspace,
        stage="review_packet",
        document=packet,
        dependencies=packet_dependencies,
        inputs={},
    )
    packet_stage["kind"] = "packet"
    if cache_hit:
        cache_hits.append("review_packet")
    stages.append(packet_stage)

    state: dict[str, Any] = {
        "schema_version": "add-trace-dag-state.v1",
        "run_id": manifest["run_id"],
        "model_id": manifest["model_id"],
        "manifest_file_sha256": manifest_file_sha256,
        "status": "needs_input" if next_actions else "pass",
        "release_level": release_level,
        "stages": stages,
        "next_actions": next_actions,
    }
    state["state_sha256"] = sha256_json(state)
    validate_schema(state, "add-trace-dag-state.schema.json", source=manifest_path)
    _atomic_write(workspace / "review-packet.json", _pretty_json(packet))
    _atomic_write(
        workspace / "REVIEW.md", render_review_markdown(packet).encode()
    )
    _atomic_write(previous_state_path, _pretty_json(state))
    return {"state": state, "review_packet": packet, "cache_hits": cache_hits}
