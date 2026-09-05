from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

from llm_arch_v2.validation_evidence import validate_validation_evidence


REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG_ROOT = REPO_ROOT / "catalog"


def _write_yaml(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(document, sort_keys=False))


def _toy_contract() -> dict:
    return {
        "schema_version": "validation-evidence.v1",
        "model_id": "toy",
        "contract_revision": 1,
        "authorities": [
            {
                "id": "official",
                "kind": "publisher_checkpoint",
                "assurance": "immutable_external_attestation",
                "source": {
                    "uri": "https://example.invalid/toy",
                    "revision": "deadbeef",
                    "digest": {"algorithm": "git_commit", "value": "deadbeef"},
                },
            },
            {
                "id": "source_lock",
                "kind": "pinned_framework_source",
                "assurance": "machine_resolved",
                "source": {
                    "uri": "catalog://toy/source-lock",
                    "local_resolution": {
                        "artifact": "pipeline.yaml",
                        "selector": "source_lock.framework_targets",
                    },
                },
            },
            {
                "id": "runtime",
                "kind": "runtime_configuration",
                "assurance": "machine_resolved",
                "source": {
                    "uri": "catalog://toy/execution",
                    "local_resolution": {
                        "artifact": "pipeline.yaml",
                        "selector": "execution",
                    },
                },
            },
            {
                "id": "eager",
                "kind": "eager_reconciliation",
                "assurance": "immutable_external_attestation",
                "source": {
                    "uri": "evidence://toy/eager",
                    "revision": "eager-v1",
                    "digest": {"algorithm": "sha256", "value": "a" * 64},
                },
            },
            {
                "id": "trace",
                "kind": "production_trace",
                "assurance": "machine_resolved",
                "source": {
                    "uri": "catalog://toy/trace",
                    "local_resolution": {
                        "artifact_glob": "profiles/*/*/*.timeline.json.gz"
                    },
                },
            },
        ],
        "gates": {
            "semantic_ir": {
                "status": "verified",
                "verification_mode": "reviewed_external_semantics",
                "subjects": [{"artifact": "model_ir.yaml"}],
                "authority_refs": ["official"],
                "assertions": [
                    {
                        "id": "layers",
                        "operator": "field_equals",
                        "subject": {"artifact": "model_ir.yaml"},
                        "selector": "facts.layers",
                        "expected": 1,
                        "authority_refs": ["official"],
                    }
                ],
            },
            "execution_contract": {
                "status": "verified",
                "verification_mode": "source_and_config_contract",
                "subjects": [{"artifact_glob": "execution_paths/*.yaml"}],
                "authority_refs": ["source_lock", "runtime"],
                "assertions": [
                    {
                        "id": "paths",
                        "operator": "collection_count_equals",
                        "subject": {"artifact_glob": "execution_paths/*.yaml"},
                        "expected": 1,
                        "authority_refs": ["runtime"],
                    }
                ],
            },
            "binding_reconciliation": {
                "status": "verified",
                "verification_mode": "graph_off_eager_reconciliation",
                "subjects": [{"artifact_glob": "bindings/*.yaml"}],
                "authority_refs": ["source_lock", "eager"],
                "assertions": [
                    {
                        "id": "sources",
                        "operator": "all_values_in_authority",
                        "subject": {"artifact_glob": "bindings/*.yaml"},
                        "selector": "source_commit",
                        "authority_ref": "source_lock",
                        "authority_selector": "source_lock.framework_targets",
                        "authority_value_selectors": ["source_commit"],
                        "authority_refs": ["source_lock"],
                    }
                ],
            },
            "production_evidence": {
                "status": "verified",
                "verification_mode": "graph_on_production_reconciliation",
                "subjects": [{"artifact_glob": "profiles/*/*/*.yaml"}],
                "authority_refs": ["eager", "trace"],
                "assertions": [
                    {
                        "id": "profiles",
                        "operator": "collection_count_equals",
                        "subject": {"artifact_glob": "profiles/*/*/*.yaml"},
                        "expected": 1,
                        "authority_refs": ["eager", "trace"],
                    }
                ],
            },
        },
    }


def _write_toy_catalog(tmp_path: Path) -> tuple[Path, dict]:
    root = tmp_path / "toy"
    _write_yaml(
        root / "model_ir.yaml",
        {"model_id": "toy", "default_execution_path": "tp", "facts": {"layers": 1}},
    )
    _write_yaml(root / "execution_paths" / "tp.yaml", {"execution_path_id": "tp"})
    _write_yaml(
        root / "bindings" / "impl.yaml",
        {
            "implementation_id": "impl",
            "execution_path_id": "tp",
            "source_repo": "https://example.invalid/runtime",
            "source_commit": "cafebabe",
        },
    )
    _write_yaml(
        root / "profiles" / "tp" / "impl" / "decode.yaml",
        {
            "profile_id": "decode",
            "implementation_id": "impl",
            "execution_path_id": "tp",
            "phase": "decode",
            "timeline": {"artifact": "decode.timeline.json.gz", "sha256": "b" * 64},
        },
    )
    timeline = root / "profiles" / "tp" / "impl" / "decode.timeline.json.gz"
    timeline.write_bytes(b"trace")
    _write_yaml(
        root / "pipeline.yaml",
        {
            "source_lock": {
                "framework_targets": [{"source_commit": "cafebabe"}]
            },
            "execution": {"execution_path_id": "tp"},
        },
    )
    contract = _toy_contract()
    _write_yaml(root / "validation_evidence.yaml", contract)
    return root, contract


def test_all_catalogs_pass_the_independent_evidence_contract() -> None:
    for model_root in sorted(CATALOG_ROOT.iterdir()):
        if not (model_root / "model_ir.yaml").is_file():
            continue
        report = validate_validation_evidence(model_root)
        assert report["status"] == "pass", report["errors"]
        assert set(report["gates"]) == {
            "semantic_ir",
            "execution_contract",
            "binding_reconciliation",
            "production_evidence",
        }
        assert len(report["contract_sha256"]) == 64
        assert report["authorities"]["official_model"]["source_revision"]
        assert report["authorities"]["official_model"]["source_digest"]
        assert all(gate["status"] == "pass" for gate in report["gates"].values())


def test_validator_rejects_a_trace_as_semantic_authority(tmp_path: Path) -> None:
    root, contract = _write_toy_catalog(tmp_path)
    contract = deepcopy(contract)
    contract["gates"]["semantic_ir"]["authority_refs"] = ["eager"]
    contract["gates"]["semantic_ir"]["assertions"][0]["authority_refs"] = ["eager"]
    _write_yaml(root / "validation_evidence.yaml", contract)

    report = validate_validation_evidence(root)

    assert report["status"] == "fail"
    assert any("disallowed authority kinds" in error for error in report["errors"])


def test_validator_rejects_subject_as_its_own_authority(tmp_path: Path) -> None:
    root, contract = _write_toy_catalog(tmp_path)
    contract = deepcopy(contract)
    official = contract["authorities"][0]
    official["assurance"] = "machine_resolved"
    official["source"] = {
        "uri": "catalog://toy/model-ir",
        "local_resolution": {"artifact": "model_ir.yaml"},
    }
    _write_yaml(root / "validation_evidence.yaml", contract)

    report = validate_validation_evidence(root)

    assert report["status"] == "fail"
    assert any("self-validation" in error for error in report["errors"])


def test_validator_rejects_a_wrong_independently_attested_value(tmp_path: Path) -> None:
    root, contract = _write_toy_catalog(tmp_path)
    contract = deepcopy(contract)
    contract["gates"]["semantic_ir"]["assertions"][0]["expected"] = 2
    _write_yaml(root / "validation_evidence.yaml", contract)

    report = validate_validation_evidence(root)

    assert report["status"] == "fail"
    assert any("observed 1 != expected 2" in error for error in report["errors"])


def test_validator_rejects_an_unlocked_binding_source(tmp_path: Path) -> None:
    root, _ = _write_toy_catalog(tmp_path)
    binding = yaml.safe_load((root / "bindings" / "impl.yaml").read_text())
    binding["source_commit"] = "unlocked"
    _write_yaml(root / "bindings" / "impl.yaml", binding)

    report = validate_validation_evidence(root)

    assert report["status"] == "fail"
    assert any("is absent from authority source_lock" in error for error in report["errors"])


def test_validator_rejects_a_new_uncovered_subject(tmp_path: Path) -> None:
    root, contract = _write_toy_catalog(tmp_path)
    contract = deepcopy(contract)
    contract["gates"]["binding_reconciliation"]["subjects"] = [
        {"artifact": "bindings/impl.yaml"}
    ]
    _write_yaml(root / "validation_evidence.yaml", contract)
    _write_yaml(
        root / "bindings" / "new.yaml",
        {
            "implementation_id": "new",
            "execution_path_id": "tp",
            "source_repo": "https://example.invalid/runtime",
            "source_commit": "cafebabe",
        },
    )

    report = validate_validation_evidence(root)

    assert report["status"] == "fail"
    assert any("uncovered canonical subjects" in error for error in report["errors"])
