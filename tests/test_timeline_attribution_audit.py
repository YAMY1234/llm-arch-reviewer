from __future__ import annotations

import gzip
import json
from pathlib import Path

from scripts.audit_timeline_attribution import audit_timeline


def _write(path: Path, event: dict) -> None:
    artifact = {
        "profile_id": "p",
        "strings": [
            "nvjet_semantic_gemm",
            "runtime_helper",
            "request_batch_metadata",
            "documented support reason",
            "model.node",
        ],
        "steps": [{"step_index": 0, "events": [event]}],
    }
    with gzip.open(path, "wt") as target:
        json.dump(artifact, target)


def test_audit_rejects_semantic_kernel_hidden_as_support(tmp_path: Path) -> None:
    path = tmp_path / "timeline.json.gz"
    _write(
        path,
        {
            "duration_us": 1.0,
            "kernel_name": 0,
            "ir_node": None,
            "support_class": 2,
            "support_reason": 3,
        },
    )
    report = audit_timeline(path)
    assert report["passed"] is False
    assert report["failures"][0]["reason"] == "semantic_kernel_left_outside_ir"


def test_audit_accepts_bound_ir_or_typed_runtime_support(tmp_path: Path) -> None:
    bound = tmp_path / "bound.json.gz"
    _write(
        bound,
        {
            "duration_us": 1.0,
            "kernel_name": 0,
            "ir_node": 4,
            "support_class": None,
            "support_reason": None,
        },
    )
    support = tmp_path / "support.json.gz"
    _write(
        support,
        {
            "duration_us": 1.0,
            "kernel_name": 1,
            "ir_node": None,
            "support_class": 2,
            "support_reason": 3,
        },
    )
    assert audit_timeline(bound)["passed"] is True
    assert audit_timeline(support)["passed"] is True
