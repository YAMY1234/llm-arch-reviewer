import hashlib
import json
from pathlib import Path
import subprocess

import pytest
import yaml

from scripts.materialize_model_authority import materialize


def source_catalog(tmp_path: Path):
    publisher = tmp_path / "publisher"
    publisher.mkdir()
    data = b'{"hidden_size": 4}\n'
    (publisher / "config.json").write_bytes(data)
    def git(*args):
        return subprocess.check_output(["git", "-C", str(publisher), *args], text=True).strip()
    git("init", "-q")
    git("add", "config.json")
    git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
        "-c", "commit.gpgsign=false", "commit", "-qm", "authority")
    revision = git("rev-parse", "HEAD")
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    lock = {"revision": revision, "files": [{"path": "config.json",
             "sha256": hashlib.sha256(data).hexdigest(), "url": (publisher / "config.json").as_uri()}]}
    (catalog / "lock.json").write_text(json.dumps(lock))
    (catalog / "pipeline.yaml").write_text("authority_lock: lock.json\n")
    ledger = {"source_snapshot": {"repository": str(publisher), "revision": revision,
              "files": [{"path": "config.json", "git_blob_oid": git("rev-parse", "HEAD:config.json")}]}}
    (catalog / "semantic_source_ledger.yaml").write_text(yaml.safe_dump(ledger))
    return catalog, publisher, data


def test_materialize_exact_git_and_resolved_bytes_without_checkout(tmp_path):
    catalog, _, data = source_catalog(tmp_path)
    source = materialize(catalog, tmp_path / "output")
    assert (source / ".git").is_dir()
    assert not (source / "config.json").exists()
    assert (source.parent / "files/config.json").read_bytes() == data
    assert materialize(catalog, tmp_path / "output") == source


def test_materialize_rejects_changed_authority_bytes(tmp_path):
    catalog, publisher, _ = source_catalog(tmp_path)
    (publisher / "config.json").write_text("modified authority")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        materialize(catalog, tmp_path / "output")


def test_materialize_rejects_wrong_git_anchor(tmp_path):
    catalog, _, _ = source_catalog(tmp_path)
    path = catalog / "semantic_source_ledger.yaml"
    ledger = yaml.safe_load(path.read_text())
    ledger["source_snapshot"]["files"][0]["git_blob_oid"] = "0" * 40
    path.write_text(yaml.safe_dump(ledger))
    with pytest.raises(ValueError, match="Git object digest mismatch"):
        materialize(catalog, tmp_path / "output")
