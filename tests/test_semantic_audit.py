from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from llm_arch_v2.semantic_audit import audit_semantic_closure  # noqa: E402


QWEN38_FLASH_NEXT_ROOT = REPO_ROOT / "catalog" / "qwen38_flash_next"
SOURCE_REPO = REPO_ROOT.parent / "sglang-qwen-next"
pytestmark = pytest.mark.skipif(
    not (SOURCE_REPO / ".git").exists(),
    reason="Qwen38FlashNext pinned-source audit requires the sibling source checkout",
)


def test_qwen38_flash_next_source_ledger_is_pinned_and_complete() -> None:
    report = audit_semantic_closure(
        model_ir_path=QWEN38_FLASH_NEXT_ROOT / "model_ir.yaml",
        ledger_path=QWEN38_FLASH_NEXT_ROOT / "semantic_source_ledger.yaml",
        source_repo=SOURCE_REPO,
    )

    assert report["status"] == "complete"
    assert report["gates"]["source_snapshot_integrity"] is True
    assert report["gates"]["source_to_ir_closure"] is True
    assert report["gates"]["ir_to_source_closure"] is True
    assert report["gates"]["catalog_attestation_honest"] is True
    assert report["counts"]["source_files"] == 15
    assert report["counts"]["pending_entrypoints"] == 0
    assert report["counts"]["pending_obligations"] == 0
    assert report["counts"]["unclassified_source_members"] == 0
    assert report["counts"]["uncovered_model_ir_leaves"] == 0
    assert report["counts"]["compound_primitive_targets"] == 0
    assert report["errors"] == []
    checked_in_report = (
        REPO_ROOT / "docs" / "qwen38_flash_next-model-ir-enrichment-audit.zh-CN.md"
    ).read_text()
    assert f"Audit fingerprint: `{report['audit_fingerprint']}`" in checked_in_report
    assert "Status: **COMPLETE**" in checked_in_report


def test_qwen38_flash_next_primitive_leaves_have_single_source_owners() -> None:
    report = audit_semantic_closure(
        model_ir_path=QWEN38_FLASH_NEXT_ROOT / "model_ir.yaml",
        ledger_path=QWEN38_FLASH_NEXT_ROOT / "semantic_source_ledger.yaml",
        source_repo=SOURCE_REPO,
    )
    assert report["compound_primitive_targets"] == {}
    assert "ple_key_value_projection.key_projection" in {
        target
        for obligation in report["obligations"]
        for target in obligation["ir_targets"]
    }
