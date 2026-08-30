from __future__ import annotations

import hashlib
from pathlib import Path

from models.deepseek_v4_pro.build.audit_deepseek_v4_pro_vllm_eager_matrix import (
    verify_artifact_manifest,
)


def test_retention_gate_requires_existing_hash_matched_bounded_files(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evidence"
    root.mkdir()
    artifact = root / "rank0.trace.json.gz"
    artifact.write_bytes(b"rank-zero")
    manifest = {
        "artifacts": [
            {
                "root": "evidence",
                "path": artifact.name,
                "bytes": artifact.stat().st_size,
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            }
        ]
    }

    assert verify_artifact_manifest(manifest, roots={"evidence": root}) == []
    artifact.write_bytes(b"tampered")
    errors = verify_artifact_manifest(manifest, roots={"evidence": root})
    assert any("hash mismatch" in error for error in errors)
    artifact.unlink()
    errors = verify_artifact_manifest(manifest, roots={"evidence": root})
    assert any("missing" in error for error in errors)


def test_retention_gate_rejects_paths_outside_the_declared_root(tmp_path: Path) -> None:
    manifest = {
        "artifacts": [
            {"root": "evidence", "path": "../escape", "bytes": 0, "sha256": "0" * 64}
        ]
    }
    errors = verify_artifact_manifest(manifest, roots={"evidence": tmp_path})
    assert errors == ["artifact[0]: unbounded path ../escape"]
