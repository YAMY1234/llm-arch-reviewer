from __future__ import annotations

import copy
import shutil
from pathlib import Path

import pytest

from llm_arch_v2.add_trace import (
    AddTraceError,
    accept_evidence,
    binding_revision_id,
    build_plan,
    runtime_identity_sha256,
    mapping_rules_sha256,
    sha256_file,
    sha256_json,
)
from llm_arch_v2.compiler import CatalogError
from scripts.materialize_binding_revision import (
    _node_bindings_from_rules,
    materialize_binding,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = REPO_ROOT / "catalog" / "qwen38_flash_next"
HEX = "a" * 64


def _leaf_items(value: object, prefix: str) -> list[tuple[str, object]]:
    if isinstance(value, dict) and value:
        return [
            item
            for key, nested in value.items()
            for item in _leaf_items(nested, f"{prefix}.{key}" if prefix else key)
        ]
    return [(prefix, value)]


def _manifest() -> dict:
    normalized = {
        "model_contract": {
            "model_artifact_id": "Qwen/Qwen3.8-Flash-Next",
            "model_revision": "b151fd157ff99b63198ab8558432f0bf43e14d58",
        },
        "execution_contract": {
            "parallelism": {
                "tp_size": 4,
                "dp_size": 1,
                "cp_size": 1,
                "ep_size": 1,
                "pp_size": 1,
            },
            "generation": {
                "mode": "eagle_mtp",
                "speculative_num_steps": 1,
                "speculative_topk": 1,
                "speculative_num_draft_tokens": 2,
            },
            "attention": {"mode": "tensor_parallel"},
            "moe": {"mode": "tensor_parallel"},
        },
        "runtime_implementation": {
            "framework_id": "sglang",
            "source_repo": "https://github.com/sgl-project/sglang",
            "source_commit": "839f546a11c36f26a2adfa6c211ee9a85c6892d8",
            "container_digest": "sha256:" + "1" * 64,
            "package_lock_sha256": "2" * 64,
            "extension_artifacts": [],
            "backend_selections": {
                "linear_attention": "flashinfer",
                "attention": "fa3",
            },
            "build_flags": {"cuda_arch": "sm_100a"},
        },
        "profile_contract": {
            "phase": "decode",
            "batch_size": 1,
            "hardware_id": "cmh_gb300",
            "isl": 8192,
            "osl": 1024,
        },
        "capture_procedure": {
            "eager_cuda_graph_enabled": False,
            "production_cuda_graph_enabled": True,
            "warmup_rounds": 3,
        },
    }
    dispositions = []
    raw_config = {}
    for bucket, payload in normalized.items():
        for index, (path, value) in enumerate(_leaf_items(payload, bucket)):
            raw_key = f"{bucket}-{index}"
            raw_config[raw_key] = value
            dispositions.append(
                {
                    "raw_key": raw_key,
                    "value": value,
                    "normalized_value": value,
                    "disposition": bucket,
                    "normalized_path": path,
                    "evidence": "fixture normalized config extract",
                }
            )
    raw_config["log-level"] = "info"
    dispositions.append(
        {
            "raw_key": "log-level",
            "value": "info",
            "disposition": "ignored",
            "ignored_reason": "diagnostic verbosity only",
            "evidence": "fixture normalized config extract",
        }
    )
    return {
        "schema_version": "add-trace-run.v1",
        "run_id": "qwen38-pr37500-tp4-mtp-decode-bs1",
        "model_id": "qwen38_flash_next",
        "raw_config": raw_config,
        "normalized_config": normalized,
        "raw_config_disposition": dispositions,
    }


def _set_normalized(manifest: dict, path: str, value: object) -> None:
    cursor = manifest["normalized_config"]
    for component in path.split(".")[:-1]:
        cursor = cursor[component]
    cursor[path.split(".")[-1]] = value
    disposition = next(
        item
        for item in manifest["raw_config_disposition"]
        if item.get("normalized_path") == path
    )
    disposition["value"] = value
    disposition["normalized_value"] = value
    manifest["raw_config"][disposition["raw_key"]] = value


def _drop_normalized(manifest: dict, path: str) -> None:
    cursor = manifest["normalized_config"]
    for component in path.split(".")[:-1]:
        cursor = cursor[component]
    cursor.pop(path.split(".")[-1])
    disposition = next(
        item
        for item in manifest["raw_config_disposition"]
        if item.get("normalized_path") == path
    )
    disposition["disposition"] = "ignored"
    disposition["ignored_reason"] = (
        "inactive generation flag outside the selected autoregressive execution"
    )
    disposition.pop("normalized_path")
    disposition.pop("normalized_value")


def test_config_resolves_exactly_one_existing_execution() -> None:
    manifest = _manifest()
    plan = build_plan(manifest, model_root=MODEL_ROOT, source=Path("fixture.yaml"))
    assert plan["execution_resolution"]["state"] == "matched_existing_execution"
    assert plan["execution_resolution"]["execution_path_id"] == "tp_only_eagle_mtp"
    assert plan["model_resolution"]["state"] == "matched_existing_model_ir"
    assert plan["model_resolution"]["semantic_revision"] == 6
    assert plan["binding_resolution"]["state"] == "new_binding_revision_required"
    assert plan["binding_resolution"]["implementation_id"] is None
    assert plan["required_stages"] == [
        "graph_off_eager_reconciliation",
        "graph_on_production_attribution",
        "profile_materialization",
        "release_audit",
    ]
    assert plan == build_plan(
        copy.deepcopy(manifest), model_root=MODEL_ROOT, source=Path("fixture.yaml")
    )


def test_generation_graph_change_resolves_to_a_distinct_execution_ir() -> None:
    mtp_plan = build_plan(
        _manifest(), model_root=MODEL_ROOT, source=Path("fixture.yaml")
    )
    autoregressive = _manifest()
    _set_normalized(
        autoregressive, "execution_contract.generation.mode", "autoregressive"
    )
    for field in (
        "speculative_num_steps",
        "speculative_topk",
        "speculative_num_draft_tokens",
    ):
        _drop_normalized(autoregressive, f"execution_contract.generation.{field}")
    autoregressive_plan = build_plan(
        autoregressive, model_root=MODEL_ROOT, source=Path("fixture.yaml")
    )
    assert mtp_plan["execution_resolution"]["execution_path_id"] == (
        "tp_only_eagle_mtp"
    )
    assert autoregressive_plan["execution_resolution"]["execution_path_id"] == (
        "tp_only"
    )
    assert mtp_plan["execution_resolution"]["execution_fingerprint"] != (
        autoregressive_plan["execution_resolution"]["execution_fingerprint"]
    )

    changed_mtp = _manifest()
    _set_normalized(
        changed_mtp,
        "execution_contract.generation.speculative_num_steps",
        2,
    )
    with pytest.raises(AddTraceError, match="new_execution_required"):
        build_plan(changed_mtp, model_root=MODEL_ROOT, source=Path("fixture.yaml"))


def test_checkpoint_revision_must_match_the_authored_model_ir_source_lock() -> None:
    manifest = _manifest()
    _set_normalized(manifest, "model_contract.model_revision", "b" * 40)
    with pytest.raises(AddTraceError, match="new_model_ir_required"):
        build_plan(manifest, model_root=MODEL_ROOT, source=Path("fixture.yaml"))


def test_parallelism_or_generation_contract_change_requires_new_execution() -> None:
    manifest = _manifest()
    _set_normalized(manifest, "execution_contract.parallelism.tp_size", 8)
    with pytest.raises(AddTraceError, match="new_execution_required"):
        build_plan(manifest, model_root=MODEL_ROOT, source=Path("fixture.yaml"))


def test_unconstrained_execution_contract_field_requires_new_execution() -> None:
    manifest = _manifest()
    manifest["normalized_config"]["execution_contract"]["scheduler"] = {
        "graph_variant": "new-structural-path"
    }
    manifest["raw_config"]["scheduler-graph-variant"] = "new-structural-path"
    manifest["raw_config_disposition"].append(
        {
            "raw_key": "scheduler-graph-variant",
            "value": "new-structural-path",
            "normalized_value": "new-structural-path",
            "disposition": "execution_contract",
            "normalized_path": "execution_contract.scheduler.graph_variant",
            "evidence": "fixture normalized config extract",
        }
    )
    with pytest.raises(AddTraceError, match="new_execution_required") as error:
        build_plan(manifest, model_root=MODEL_ROOT, source=Path("fixture.yaml"))
    assert "not constrained by this selector" in str(error.value)

    manifest = _manifest()
    _set_normalized(manifest, "execution_contract.generation.mode", "dspark")
    with pytest.raises(AddTraceError, match="new_execution_required"):
        build_plan(manifest, model_root=MODEL_ROOT, source=Path("fixture.yaml"))


def test_dp_execution_is_selected_from_normalized_contract() -> None:
    manifest = _manifest()
    changes = {
        "execution_contract.parallelism.dp_size": 4,
        "execution_contract.generation.mode": "autoregressive",
        "execution_contract.attention.mode": "data_parallel",
    }
    for path, value in changes.items():
        _set_normalized(manifest, path, value)
    for field in (
        "speculative_num_steps",
        "speculative_topk",
        "speculative_num_draft_tokens",
    ):
        _drop_normalized(manifest, f"execution_contract.generation.{field}")
    plan = build_plan(manifest, model_root=MODEL_ROOT, source=Path("fixture.yaml"))
    assert plan["execution_resolution"]["execution_path_id"] == "dp_attention"


def test_backend_or_source_change_revisions_binding_not_execution() -> None:
    original = _manifest()
    changed = _manifest()
    _set_normalized(changed, "runtime_implementation.source_commit", "b" * 40)
    original_plan = build_plan(
        original, model_root=MODEL_ROOT, source=Path("fixture.yaml")
    )
    changed_plan = build_plan(
        changed, model_root=MODEL_ROOT, source=Path("fixture.yaml")
    )
    assert original_plan["execution_resolution"] == changed_plan["execution_resolution"]
    assert (
        original_plan["binding_resolution"]["binding_revision_id"]
        != changed_plan["binding_resolution"]["binding_revision_id"]
    )


def test_profile_change_does_not_revision_execution_or_binding() -> None:
    original = _manifest()
    changed = _manifest()
    _set_normalized(changed, "profile_contract.batch_size", 64)
    original_plan = build_plan(
        original, model_root=MODEL_ROOT, source=Path("fixture.yaml")
    )
    changed_plan = build_plan(
        changed, model_root=MODEL_ROOT, source=Path("fixture.yaml")
    )
    assert original_plan["execution_resolution"] == changed_plan["execution_resolution"]
    assert original_plan["binding_resolution"] == changed_plan["binding_resolution"]


def test_runtime_identity_rejects_function_or_kernel_names() -> None:
    manifest = _manifest()
    manifest["normalized_config"]["runtime_implementation"]["kernel_name"] = "gemm"
    manifest["raw_config"]["kernel-name"] = "gemm"
    manifest["raw_config_disposition"].append(
        {
            "raw_key": "kernel-name",
            "value": "gemm",
            "disposition": "runtime_implementation",
            "normalized_path": "runtime_implementation.kernel_name",
            "evidence": "fixture",
        }
    )
    with pytest.raises(AddTraceError, match="Additional properties are not allowed"):
        build_plan(manifest, model_root=MODEL_ROOT, source=Path("fixture.yaml"))


def test_runtime_identity_requires_full_git_sha() -> None:
    manifest = _manifest()
    _set_normalized(manifest, "runtime_implementation.source_commit", "839f546")
    with pytest.raises(AddTraceError, match="source_commit.*does not match"):
        build_plan(manifest, model_root=MODEL_ROOT, source=Path("fixture.yaml"))


def test_runtime_identity_requires_valid_uri_and_hex_digests() -> None:
    manifest = _manifest()
    _set_normalized(manifest, "runtime_implementation.source_repo", "not a source URI")
    with pytest.raises(AddTraceError, match=r"absolute HTTP\(S\) URI"):
        build_plan(manifest, model_root=MODEL_ROOT, source=Path("fixture.yaml"))

    manifest = _manifest()
    _set_normalized(
        manifest,
        "runtime_implementation.container_digest",
        "sha256:" + "z" * 64,
    )
    with pytest.raises(AddTraceError, match="does not match"):
        build_plan(manifest, model_root=MODEL_ROOT, source=Path("fixture.yaml"))


def test_mapping_rules_digest_normalizes_set_like_order() -> None:
    first = _accepted_fixture()[2]["mapping_rules"][0]
    first["scope"]["layer_ids"] = [2, 0, 1]
    second = copy.deepcopy(first)
    second["scope"]["layer_ids"] = [1, 2, 0]
    assert mapping_rules_sha256([first]) == mapping_rules_sha256([second])


def test_every_raw_and_normalized_config_leaf_requires_one_disposition() -> None:
    manifest = _manifest()
    manifest["raw_config_disposition"] = [
        item
        for item in manifest["raw_config_disposition"]
        if item.get("normalized_path") != "execution_contract.parallelism.tp_size"
    ]
    with pytest.raises(AddTraceError, match="raw config fields lack disposition"):
        build_plan(manifest, model_root=MODEL_ROOT, source=Path("fixture.yaml"))

    manifest = _manifest()
    manifest["raw_config"]["unclassified-new-flag"] = True
    with pytest.raises(AddTraceError, match="raw config fields lack disposition"):
        build_plan(manifest, model_root=MODEL_ROOT, source=Path("fixture.yaml"))

    manifest = _manifest()
    manifest["raw_config"]["log-level"] = "debug"
    with pytest.raises(AddTraceError, match="does not match raw_config"):
        build_plan(manifest, model_root=MODEL_ROOT, source=Path("fixture.yaml"))

    manifest = _manifest()
    duplicate = copy.deepcopy(manifest["raw_config_disposition"][0])
    duplicate["raw_key"] = "alias-of-first-field"
    manifest["raw_config"][duplicate["raw_key"]] = duplicate["value"]
    manifest["raw_config_disposition"].append(duplicate)
    with pytest.raises(AddTraceError, match="both own"):
        build_plan(manifest, model_root=MODEL_ROOT, source=Path("fixture.yaml"))


def test_raw_config_may_be_evidence_backed_normalized_without_value_aliasing() -> None:
    manifest = _manifest()
    item = next(
        item
        for item in manifest["raw_config_disposition"]
        if item.get("normalized_path") == "model_contract.model_artifact_id"
    )
    manifest["raw_config"][item["raw_key"]] = "/model"
    item["value"] = "/model"
    plan = build_plan(manifest, model_root=MODEL_ROOT, source=Path("fixture.yaml"))
    assert plan["model_resolution"]["model_artifact_id"] == "Qwen/Qwen3.8-Flash-Next"

    item["normalized_value"] = "wrong/model"
    with pytest.raises(AddTraceError, match="normalized_value does not equal"):
        build_plan(manifest, model_root=MODEL_ROOT, source=Path("fixture.yaml"))


def test_overlapping_execution_selectors_fail_closed(tmp_path: Path) -> None:
    model_root = tmp_path / "qwen38_flash_next"
    shutil.copytree(MODEL_ROOT, model_root)
    execution_root = model_root / "execution_paths"
    duplicate = (
        (execution_root / "tp_only.yaml")
        .read_text()
        .replace("execution_path_id: tp_only", "execution_path_id: duplicate")
    )
    (execution_root / "duplicate.yaml").write_text(duplicate)
    with pytest.raises(CatalogError, match="execution selectors overlap"):
        build_plan(_manifest(), model_root=model_root, source=Path("fixture.yaml"))


def test_generation_graph_selector_requires_exact_execution_ir(tmp_path: Path) -> None:
    model_root = tmp_path / "qwen38_flash_next"
    shutil.copytree(MODEL_ROOT, model_root)
    path = model_root / "execution_paths" / "tp_only.yaml"
    path.write_text(
        path.read_text().replace(
            "generation.mode: {equals: autoregressive}",
            "generation.mode: {one_of: [autoregressive, eagle_mtp]}",
        )
    )
    with pytest.raises(CatalogError, match="generation.mode must use one exact equals"):
        build_plan(_manifest(), model_root=model_root, source=Path("fixture.yaml"))


def _accepted_fixture() -> tuple[dict, dict, dict, dict, dict]:
    manifest = _manifest()
    plan = build_plan(manifest, model_root=MODEL_ROOT, source=Path("fixture.yaml"))
    identity = manifest["normalized_config"]["runtime_implementation"]
    identity_digest = runtime_identity_sha256(identity, source=Path("fixture.yaml"))
    revision_id = binding_revision_id(
        identity_digest, plan["execution_resolution"]["execution_fingerprint"]
    )
    rule = {
        "rule_id": "target-verify",
        "ir_target": "mtp_generation.target_verify",
        "eager_match": {"python_stack_digest": HEX},
        "production_transfer": {
            "method": "annotated_scope",
            "signature_digest": HEX,
        },
        "scope": {"phase": "decode", "generation_mode": "eagle_mtp"},
    }
    binding = {
        "schema_version": "binding-revision.v1",
        "binding_revision_id": revision_id,
        "model_id": manifest["model_id"],
        "execution_path_id": plan["execution_resolution"]["execution_path_id"],
        "execution_fingerprint": plan["execution_resolution"]["execution_fingerprint"],
        "runtime_identity": identity,
        "runtime_identity_sha256": identity_digest,
        "runtime_evidence_artifacts": [
            {"path": "runtime-evidence.json", "sha256": "d" * 64}
        ],
        "mapping_rules_sha256": mapping_rules_sha256([rule]),
        "mapping_rules": [rule],
    }
    artifacts = [
        {"rank": rank, "path": f"rank{rank}.trace.json.gz", "sha256": HEX}
        for rank in range(4)
    ]
    reconciliation = {
        "schema_version": "binding-reconciliation.v1",
        "run_id": manifest["run_id"],
        "model_id": manifest["model_id"],
        "execution_path_id": plan["execution_resolution"]["execution_path_id"],
        "execution_fingerprint": plan["execution_resolution"]["execution_fingerprint"],
        "binding_revision_id": revision_id,
        "plan_sha256": plan["plan_sha256"],
        "status": "pass",
        "cuda_graph_enabled": False,
        "protocol_artifact": {"path": "eager-protocol.json", "sha256": "b" * 64},
        "rank_artifacts": artifacts,
        "rule_results": [
            {
                "rule_id": "target-verify",
                "ir_target": "mtp_generation.target_verify",
                "rank": rank,
                "eager_event_ids": [f"eager-r{rank}"],
                "duration_us": 1.0,
                "matched_evidence": {"python_stack_digest": HEX},
                "status": "pass",
            }
            for rank in range(4)
        ],
        "support_events": [],
        "coverage": {
            "event_count_total": 4,
            "event_count_accounted": 4,
            "duration_us_total": 4.0,
            "duration_us_accounted": 4.0,
        },
        "unresolved": [],
        "discrepancies": [],
    }
    attribution = {
        "schema_version": "trace-attribution.v1",
        "run_id": manifest["run_id"],
        "model_id": manifest["model_id"],
        "execution_path_id": plan["execution_resolution"]["execution_path_id"],
        "execution_fingerprint": plan["execution_resolution"]["execution_fingerprint"],
        "binding_revision_id": revision_id,
        "plan_sha256": plan["plan_sha256"],
        "status": "pass",
        "phase": "decode",
        "cuda_graph_enabled": True,
        "protocol_artifact": {
            "path": "production-protocol.json",
            "sha256": "c" * 64,
        },
        "rank_artifacts": artifacts,
        "window_selection_artifact": {
            "path": "window-selection.json",
            "sha256": HEX,
        },
        "event_mappings": [
            {
                "event_id": f"production-r{rank}",
                "rank": rank,
                "ir_target": "mtp_generation.target_verify",
                "rule_id": "target-verify",
                "eager_event_ids": [f"eager-r{rank}"],
                "transfer_method": "annotated_scope",
                "transfer_signature_digest": HEX,
                "confidence": "exact",
                "occurrence_id": f"rank{rank}:step0",
                "duration_us": 1.0,
            }
            for rank in range(4)
        ],
        "support_events": [],
        "fusion_groups": [],
        "coverage": {
            "event_count_total": 4,
            "event_count_accounted": 4,
            "duration_us_total": 4.0,
            "duration_us_accounted": 4.0,
        },
        "unresolved": [],
    }
    return manifest, plan, binding, reconciliation, attribution


def test_eager_and_production_evidence_accept_only_with_full_closure() -> None:
    fixture = _accepted_fixture()
    accepted = accept_evidence(
        *fixture,
        model_root=MODEL_ROOT,
        source=Path("fixture.yaml"),
        verify_files=False,
    )
    assert accepted["status"] == "pass"
    assert accepted["rank_count"] == 4
    assert accepted["eager_event_count"] == 4
    assert accepted["eager_duration_us"] == 4.0
    assert accepted["production_event_count"] == 4
    assert accepted["eager_protocol_sha256"] == "b" * 64
    assert accepted["production_protocol_sha256"] == "c" * 64
    assert accepted["runtime_evidence_sha256"] == sha256_json(
        fixture[2]["runtime_evidence_artifacts"]
    )
    assert accepted["window_selection_sha256"] == HEX
    assert accepted["manifest_sha256"] == fixture[1]["manifest_sha256"]
    assert accepted["plan_sha256"] == fixture[1]["plan_sha256"]
    assert "model_ir" not in accepted
    assert "execution_ir" not in accepted


def test_binding_materialization_drops_incompatible_template_inheritance() -> None:
    fixture = list(_accepted_fixture())
    revision = fixture[2]
    revision["mapping_rules"][0]["eager_match"] = {
        "source_symbol": "sglang/model.py::forward"
    }
    revision["mapping_rules_sha256"] = mapping_rules_sha256(
        revision["mapping_rules"]
    )
    for result in fixture[3]["rule_results"]:
        result["matched_evidence"] = {
            "source_symbol": "sglang/model.py::forward"
        }
    acceptance = accept_evidence(
        *fixture,
        model_root=MODEL_ROOT,
        source=Path("fixture.yaml"),
        verify_files=False,
    )
    template = {
        "schema_version": "implementation-binding.v2",
        "implementation_id": "old",
        "label": "old",
        "model_id": revision["model_id"],
        "execution_path_id": revision["execution_path_id"],
        "source_repo": "https://example.com/old.git",
        "source_commit": "f" * 40,
        "source_patch_sha256": "e" * 64,
        "extends": "old-base",
        "binding_compatible_base_commit": "f" * 40,
        "node_bindings": {"top.embedding": {"symbols": ["stale"]}},
    }
    materialized = materialize_binding(
        template,
        revision,
        acceptance,
        implementation_id="new",
        label="new",
        container="image@sha256:" + "1" * 64,
        eager_evidence="evidence://eager",
        production_evidence="evidence://production",
    )
    assert materialized["source_repo"] == revision["runtime_identity"]["source_repo"]
    assert materialized["add_trace_acceptance_sha256"] == acceptance[
        "acceptance_sha256"
    ]
    assert materialized["node_bindings"] == {
        "mtp_generation.target_verify": {
            "symbols": ["forward"],
            "kernel_signatures": [],
            "links": [
                {
                    "file": "python/sglang/model.py",
                    "symbol": "forward",
                    "display": "sglang/model.py — forward",
                }
            ],
        }
    }
    assert "extends" not in materialized
    assert "binding_compatible_base_commit" not in materialized
    assert "source_patch_sha256" not in materialized


def test_binding_materialization_rejects_unaccepted_revision() -> None:
    fixture = _accepted_fixture()
    _, _, revision, _, _ = fixture
    acceptance = accept_evidence(
        *fixture,
        model_root=MODEL_ROOT,
        source=Path("fixture.yaml"),
        verify_files=False,
    )
    acceptance["mapping_rules_sha256"] = "b" * 64
    acceptance_without_digest = {
        key: value
        for key, value in acceptance.items()
        if key != "acceptance_sha256"
    }
    acceptance["acceptance_sha256"] = sha256_json(acceptance_without_digest)
    with pytest.raises(
        AddTraceError, match="does not authorize this exact Binding revision"
    ):
        materialize_binding(
            {"schema_version": "implementation-binding.v2"},
            revision,
            acceptance,
            implementation_id="new",
            label="new",
            container="image@sha256:" + "1" * 64,
            eager_evidence="evidence://eager",
            production_evidence="evidence://production",
        )

    acceptance["mapping_rules_sha256"] = revision["mapping_rules_sha256"]
    with pytest.raises(AddTraceError, match="acceptance digest"):
        materialize_binding(
            {"schema_version": "implementation-binding.v2"},
            revision,
            acceptance,
            implementation_id="new",
            label="new",
            container="image@sha256:" + "1" * 64,
            eager_evidence="evidence://eager",
            production_evidence="evidence://production",
        )


def test_binding_materialization_rejects_non_repository_source_path() -> None:
    rule = copy.deepcopy(_accepted_fixture()[2]["mapping_rules"][0])
    rule["eager_match"] = {"source_symbol": "/tmp/installed/model.py::forward"}
    with pytest.raises(AddTraceError, match="non-repository source path"):
        _node_bindings_from_rules([rule], framework_id="sglang")


def test_acceptance_verifies_all_referenced_artifacts(tmp_path: Path) -> None:
    values = list(_accepted_fixture())
    payloads = {
        **{f"rank{rank}.trace.json.gz": f"rank-{rank}" for rank in range(4)},
        "eager-protocol.json": "eager-protocol",
        "production-protocol.json": "production-protocol",
        "runtime-evidence.json": "runtime-evidence",
        "window-selection.json": "window",
    }
    for name, payload in payloads.items():
        (tmp_path / name).write_text(payload)
    for artifact_set in (values[3]["rank_artifacts"], values[4]["rank_artifacts"]):
        for item in artifact_set:
            item["sha256"] = sha256_file(tmp_path / item["path"])
    values[4]["window_selection_artifact"]["sha256"] = sha256_file(
        tmp_path / "window-selection.json"
    )
    values[3]["protocol_artifact"]["sha256"] = sha256_file(
        tmp_path / "eager-protocol.json"
    )
    values[4]["protocol_artifact"]["sha256"] = sha256_file(
        tmp_path / "production-protocol.json"
    )
    values[2]["runtime_evidence_artifacts"][0]["sha256"] = sha256_file(
        tmp_path / "runtime-evidence.json"
    )
    accepted = accept_evidence(
        *values,
        model_root=MODEL_ROOT,
        source=tmp_path / "acceptance-input.json",
        verify_files=True,
    )
    assert accepted["status"] == "pass"

    (tmp_path / "rank2.trace.json.gz").write_text("tampered")
    with pytest.raises(AddTraceError, match="rank artifact SHA mismatch"):
        accept_evidence(
            *values,
            model_root=MODEL_ROOT,
            source=tmp_path / "acceptance-input.json",
            verify_files=True,
        )


def test_acceptance_reopens_and_distinguishes_capture_protocols(
    tmp_path: Path,
) -> None:
    values = list(_accepted_fixture())
    for rank in range(4):
        rank_artifact = tmp_path / f"rank{rank}.trace.json.gz"
        rank_artifact.write_text(f"rank-{rank}")
        for artifact_set in (
            values[3]["rank_artifacts"],
            values[4]["rank_artifacts"],
        ):
            artifact_set[rank]["sha256"] = sha256_file(rank_artifact)
    window_selection = tmp_path / "window-selection.json"
    window_selection.write_text("window")
    values[4]["window_selection_artifact"]["sha256"] = sha256_file(
        window_selection
    )
    runtime_evidence = tmp_path / "runtime-evidence.json"
    runtime_evidence.write_text("runtime")
    values[2]["runtime_evidence_artifacts"][0]["sha256"] = sha256_file(
        runtime_evidence
    )
    eager_protocol = tmp_path / "eager-protocol.json"
    production_protocol = tmp_path / "production-protocol.json"
    eager_protocol.write_text("eager")
    production_protocol.write_text("production")
    values[3]["protocol_artifact"] = {
        "path": str(eager_protocol),
        "sha256": sha256_file(eager_protocol),
    }
    values[4]["protocol_artifact"] = {
        "path": str(production_protocol),
        "sha256": sha256_file(production_protocol),
    }
    accept_evidence(
        *values,
        model_root=MODEL_ROOT,
        source=tmp_path / "acceptance-input.json",
        verify_files=True,
    )

    production_protocol.write_text("tampered")
    with pytest.raises(AddTraceError, match="artifact SHA mismatch"):
        accept_evidence(
            *values,
            model_root=MODEL_ROOT,
            source=tmp_path / "acceptance-input.json",
            verify_files=True,
        )


def test_acceptance_reopens_runtime_identity_evidence(tmp_path: Path) -> None:
    values = list(_accepted_fixture())
    for rank in range(4):
        rank_artifact = tmp_path / f"rank{rank}.trace.json.gz"
        rank_artifact.write_text(f"rank-{rank}")
        for artifact_set in (
            values[3]["rank_artifacts"],
            values[4]["rank_artifacts"],
        ):
            artifact_set[rank]["sha256"] = sha256_file(rank_artifact)
    for document, record_key, filename, payload in (
        (values[3], "protocol_artifact", "eager-protocol.json", "eager"),
        (
            values[4],
            "protocol_artifact",
            "production-protocol.json",
            "production",
        ),
        (
            values[4],
            "window_selection_artifact",
            "window-selection.json",
            "window",
        ),
    ):
        artifact = tmp_path / filename
        artifact.write_text(payload)
        document[record_key] = {
            "path": str(artifact),
            "sha256": sha256_file(artifact),
        }
    runtime_evidence = tmp_path / "runtime-evidence.json"
    runtime_evidence.write_text("runtime")
    values[2]["runtime_evidence_artifacts"] = [
        {"path": str(runtime_evidence), "sha256": sha256_file(runtime_evidence)}
    ]
    accept_evidence(
        *values,
        model_root=MODEL_ROOT,
        source=tmp_path / "acceptance-input.json",
        verify_files=True,
    )
    runtime_evidence.write_text("tampered")
    with pytest.raises(AddTraceError, match="artifact SHA mismatch"):
        accept_evidence(
            *values,
            model_root=MODEL_ROOT,
            source=tmp_path / "acceptance-input.json",
            verify_files=True,
        )

    values = list(_accepted_fixture())
    values[4]["protocol_artifact"] = copy.deepcopy(
        values[3]["protocol_artifact"]
    )
    with pytest.raises(AddTraceError, match="distinct protocol artifacts"):
        accept_evidence(
            *values,
            model_root=MODEL_ROOT,
            source=Path("fixture.yaml"),
            verify_files=False,
        )


def test_reviewed_fusion_accounts_for_untimed_members_without_copying_time() -> None:
    values = list(_accepted_fixture())
    rule = copy.deepcopy(values[2]["mapping_rules"][0])
    rule["rule_id"] = "aux-fused"
    rule["ir_target"] = "mtp_generation.accept_commit"
    rule["production_transfer"]["method"] = "reviewed_fusion"
    values[2]["mapping_rules"].append(rule)
    values[2]["mapping_rules_sha256"] = mapping_rules_sha256(values[2]["mapping_rules"])
    for rank in range(4):
        values[3]["rule_results"].append(
            {
                "rule_id": "aux-fused",
                "ir_target": "mtp_generation.accept_commit",
                "rank": rank,
                "eager_event_ids": [f"eager-aux-r{rank}"],
                "duration_us": 1.0,
                "matched_evidence": {"python_stack_digest": HEX},
                "status": "pass",
            }
        )
        values[4]["fusion_groups"].append(
            {
                "group_id": f"fusion-r{rank}",
                "rank": rank,
                "owner_rule_id": "target-verify",
                "owner_ir_target": "mtp_generation.target_verify",
                "member_rule_ids": ["aux-fused"],
                "member_ir_targets": ["mtp_generation.accept_commit"],
                "event_ids": [f"production-r{rank}"],
                "evidence": "exact eager-to-production fusion proof",
            }
        )
    values[3]["coverage"] = {
        "event_count_total": 8,
        "event_count_accounted": 8,
        "duration_us_total": 8.0,
        "duration_us_accounted": 8.0,
    }
    accepted = accept_evidence(
        *values,
        model_root=MODEL_ROOT,
        source=Path("fixture.yaml"),
        verify_files=False,
    )
    assert accepted["production_event_count"] == 4

    values[4]["fusion_groups"][0].update(
        owner_rule_id="aux-fused",
        owner_ir_target="mtp_generation.accept_commit",
        member_rule_ids=["target-verify"],
        member_ir_targets=["mtp_generation.target_verify"],
    )
    with pytest.raises(AddTraceError, match="member rule must use reviewed_fusion"):
        accept_evidence(
            *values,
            model_root=MODEL_ROOT,
            source=Path("fixture.yaml"),
            verify_files=False,
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda values: values[3].update(cuda_graph_enabled=True),
            "False was expected",
        ),
        (
            lambda values: values[3]["rank_artifacts"].pop(),
            "expected exact ranks",
        ),
        (
            lambda values: values[4]["event_mappings"][0].update(
                eager_event_ids=["unknown"]
            ),
            "unknown eager events",
        ),
        (
            lambda values: values[4]["coverage"].update(event_count_accounted=3),
            "event-count coverage",
        ),
        (
            lambda values: values[3]["coverage"].update(event_count_accounted=3),
            "eager event-count coverage",
        ),
        (
            lambda values: values[3]["coverage"].update(duration_us_accounted=3.0),
            "eager duration coverage",
        ),
        (
            lambda values: values[4]["event_mappings"][0].update(
                transfer_method="exact_sequence"
            ),
            "transfer method",
        ),
        (
            lambda values: values[4]["event_mappings"][0].update(
                transfer_signature_digest="b" * 64
            ),
            "transfer signature",
        ),
        (
            lambda values: values[4]["event_mappings"][0].update(
                event_id="production-r1"
            ),
            "globally unique",
        ),
        (
            lambda values: values[1].update(manifest_sha256="b" * 64),
            "plan manifest digest",
        ),
        (
            lambda values: values[1].update(plan_sha256="b" * 64),
            "plan digest",
        ),
        (
            lambda values: values[4].update(plan_sha256="b" * 64),
            "identity mismatch for plan_sha256",
        ),
        (
            lambda values: values[3]["rule_results"].append(
                copy.deepcopy(values[3]["rule_results"][0])
            ),
            "must pass on every TP rank",
        ),
        (
            lambda values: values[3]["rule_results"][0].update(
                ir_target="mtp_generation.wrong"
            ),
            "targets .* expected",
        ),
        (
            lambda values: values[3]["rule_results"][0].update(
                matched_evidence={"python_stack_digest": "b" * 64}
            ),
            "does not prove its authored match predicate",
        ),
        (
            lambda values: values[2]["mapping_rules"][0].update(
                ir_target="mtp_generation.silently_changed"
            ),
            "mapping-rules digest mismatch",
        ),
        (
            lambda values: (
                values[2]["mapping_rules"][0].update(
                    ir_target="mtp_generation.unknown_compiled_node"
                ),
                values[2].update(
                    mapping_rules_sha256=mapping_rules_sha256(
                        values[2]["mapping_rules"]
                    )
                ),
            ),
            "unknown compiled IR targets",
        ),
        (
            lambda values: (
                values[2]["mapping_rules"][0]["scope"].update(
                    generation_mode="autoregressive"
                ),
                values[2].update(
                    mapping_rules_sha256=mapping_rules_sha256(
                        values[2]["mapping_rules"]
                    )
                ),
            ),
            "generation mode differs",
        ),
    ],
)
def test_acceptance_fails_closed(mutate, message: str) -> None:
    values = list(_accepted_fixture())
    mutate(values)
    with pytest.raises(AddTraceError, match=message):
        accept_evidence(
            *values,
            model_root=MODEL_ROOT,
            source=Path("fixture.yaml"),
            verify_files=False,
        )
