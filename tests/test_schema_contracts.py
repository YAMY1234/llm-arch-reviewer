from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator

from llm_arch_v2.compiler import CatalogError, compile_catalog


REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG_ROOT = REPO_ROOT / "catalog"
MODEL_IR_SCHEMA = REPO_ROOT / "schema" / "v2" / "model-ir.schema.json"
SEMANTIC_LEDGER_SCHEMA = (
    REPO_ROOT / "schema" / "v2" / "semantic-source-ledger.schema.json"
)
VALIDATION_EVIDENCE_SCHEMA = (
    REPO_ROOT / "schema" / "v2" / "validation-evidence.schema.json"
)
EXECUTION_PLAN_SCHEMA = REPO_ROOT / "schema" / "v2" / "execution-plan.schema.json"


def _validate_yaml(data_path: Path, schema_path: Path) -> None:
    schema = json.loads(schema_path.read_text())
    Draft202012Validator.check_schema(schema)
    document = yaml.safe_load(data_path.read_text())
    errors = sorted(
        Draft202012Validator(schema).iter_errors(document),
        key=lambda error: [str(item) for item in error.absolute_path],
    )
    assert not errors, "\n".join(
        f"{data_path}:{'.'.join(map(str, error.absolute_path))}: {error.message}"
        for error in errors
    )


@pytest.mark.parametrize(
    "model_ir_path",
    sorted(CATALOG_ROOT.glob("*/model_ir.yaml")),
    ids=lambda path: path.parent.name,
)
def test_model_ir_documents_match_json_schema(model_ir_path: Path) -> None:
    _validate_yaml(model_ir_path, MODEL_IR_SCHEMA)


@pytest.mark.parametrize(
    "ledger_path",
    sorted(CATALOG_ROOT.glob("*/semantic_source_ledger.yaml")),
    ids=lambda path: path.parent.name,
)
def test_semantic_source_ledgers_match_json_schema(ledger_path: Path) -> None:
    _validate_yaml(ledger_path, SEMANTIC_LEDGER_SCHEMA)


@pytest.mark.parametrize(
    "evidence_path",
    sorted(CATALOG_ROOT.glob("*/validation_evidence.yaml")),
    ids=lambda path: path.parent.name,
)
def test_validation_evidence_contracts_match_json_schema(evidence_path: Path) -> None:
    _validate_yaml(evidence_path, VALIDATION_EVIDENCE_SCHEMA)


@pytest.mark.parametrize(
    "execution_path",
    sorted(CATALOG_ROOT.glob("*/execution_paths/*.yaml")),
    ids=lambda path: f"{path.parents[1].name}:{path.stem}",
)
def test_execution_plans_match_json_schema(execution_path: Path) -> None:
    _validate_yaml(execution_path, EXECUTION_PLAN_SCHEMA)


@pytest.mark.parametrize(
    "schema_path",
    sorted((REPO_ROOT / "schema" / "v2").glob("*.schema.json")),
    ids=lambda path: path.name,
)
def test_every_v2_json_schema_is_well_formed(schema_path: Path) -> None:
    Draft202012Validator.check_schema(json.loads(schema_path.read_text()))


def test_every_catalog_has_a_validation_evidence_contract() -> None:
    catalogs = {
        path.parent for path in CATALOG_ROOT.glob("*/model_ir.yaml")
    }
    contracted = {
        path.parent for path in CATALOG_ROOT.glob("*/validation_evidence.yaml")
    }
    assert contracted == catalogs


def test_dimension_symbol_contracts_are_complete_and_fail_closed() -> None:
    for model in ("qwen38_flash_next", "qwen35"):
        bundle = compile_catalog(CATALOG_ROOT / model)
        dimensions = bundle["model_ir"]["dimensions"]
        symbols = bundle["model_ir"]["dimension_symbols"]
        assert set(symbols) == set(dimensions)
        assert symbols["B"] == {
            "meaning": "request batch count",
            "value_class": "profile_runtime",
            "source_path": "workload.batch_size",
            "provenance": "validated profile workload.batch_size",
        }

    qwen38_flash_next = compile_catalog(CATALOG_ROOT / "qwen38_flash_next")
    draft = qwen38_flash_next["model_ir"]["dimension_symbols"]["D"]
    assert draft["value_class"] == "stage_dependent"
    assert draft["stage_resolutions"] == [
        {
            "scope_targets": ["mtp_generation.mtp_draft_extend"],
            "phases": ["decode"],
            "generation_modes": ["eagle_mtp"],
            "source_path": "workload.speculative_num_draft_tokens",
            "provenance": "validated MTP decode workload.speculative_num_draft_tokens",
        }
    ]
    assert "global" in draft["provenance"]


def test_dimension_symbol_schema_rejects_unscoped_or_unknown_stage_resolution(
    tmp_path: Path,
) -> None:
    source = yaml.safe_load((CATALOG_ROOT / "qwen38_flash_next" / "model_ir.yaml").read_text())
    source["dimension_symbols"]["D"]["stage_resolutions"][0]["scope_targets"] = [
        "missing.stage"
    ]
    model_root = tmp_path / "qwen38_flash_next"
    model_root.mkdir()
    (model_root / "model_ir.yaml").write_text(yaml.safe_dump(source, sort_keys=False))
    # Copying a complete catalog is unnecessary: dimension validation happens
    # before execution/binding/profile discovery.
    with pytest.raises(CatalogError, match="unknown targets"):
        compile_catalog(model_root)
