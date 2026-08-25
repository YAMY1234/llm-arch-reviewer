from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from llm_arch_v2.semantic_audit import audit_semantic_closure  # noqa: E402


QWEN40_ROOT = REPO_ROOT / "catalog" / "qwen40"
SOURCE_REPO = REPO_ROOT.parent / "sglang-qwen-next"
pytestmark = pytest.mark.skipif(
    not (SOURCE_REPO / ".git").exists(),
    reason="Qwen40 pinned-source audit requires the sibling source checkout",
)


def test_qwen40_source_ledger_is_pinned_and_fail_closed() -> None:
    report = audit_semantic_closure(
        model_ir_path=QWEN40_ROOT / "model_ir.yaml",
        ledger_path=QWEN40_ROOT / "semantic_source_ledger.yaml",
        source_repo=SOURCE_REPO,
    )

    assert report["status"] == "incomplete"
    assert report["gates"]["source_snapshot_integrity"] is True
    assert report["gates"]["source_to_ir_closure"] is False
    assert report["gates"]["ir_to_source_closure"] is False
    assert report["gates"]["catalog_attestation_honest"] is False
    assert report["counts"]["source_files"] == 9
    assert report["counts"]["unclassified_source_members"] > 0
    assert report["errors"] == []


def test_qwen40_ple_audit_exposes_compound_model_ir_leaves() -> None:
    report = audit_semantic_closure(
        model_ir_path=QWEN40_ROOT / "model_ir.yaml",
        ledger_path=QWEN40_ROOT / "semantic_source_ledger.yaml",
        source_repo=SOURCE_REPO,
    )
    collisions = report["compound_primitive_targets"]

    assert set(collisions) == {
        "ple.key_value_projection",
        "ple.short_conv",
        "ple_grouped_norm_gate.grouped_norm",
        "ple_grouped_norm_gate.query_key_gate",
    }
    assert collisions["ple.key_value_projection"] == [
        "ple_forward.key_projection",
        "ple_forward.value_projection",
    ]
