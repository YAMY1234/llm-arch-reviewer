from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator


REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG_ROOT = REPO_ROOT / "catalog"
MODEL_IR_SCHEMA = REPO_ROOT / "schema" / "v2" / "model-ir.schema.json"
SEMANTIC_LEDGER_SCHEMA = (
    REPO_ROOT / "schema" / "v2" / "semantic-source-ledger.schema.json"
)


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
