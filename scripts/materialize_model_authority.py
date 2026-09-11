#!/usr/bin/env python3
"""Fetch a catalog's locked source Git metadata and SHA256-authorized small files.

No checkout or Git LFS smudge is performed. The separate files/ directory holds
resolved publisher bytes (including reports), whereas Git retains original
objects for source-ledger anchors. Never downloads weight shards.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from urllib.request import urlopen

import yaml


def materialize(model_root: Path, output: Path) -> Path:
    pipeline = yaml.safe_load((model_root / "pipeline.yaml").read_text())
    ledger = yaml.safe_load((model_root / "semantic_source_ledger.yaml").read_text())
    snapshot = ledger["source_snapshot"]
    lock_path = (model_root / pipeline["authority_lock"]).resolve()
    lock = json.loads(lock_path.read_text())
    if lock["revision"] != snapshot["revision"]:
        raise ValueError("authority and semantic source revisions differ")
    output.mkdir(parents=True, exist_ok=True)
    git_root = output / "git"
    env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
    def git(*args):
        return subprocess.run(["git", "-C", str(git_root), *args], env=env,
                              check=True, text=True, capture_output=True).stdout.strip()
    if not git_root.exists():
        git_root.mkdir()
        git("init", "--quiet")
        git("remote", "add", "origin", snapshot["repository"])
    if git("remote", "get-url", "origin") != snapshot["repository"]:
        raise ValueError("existing source repository remote differs from source lock")
    revision = snapshot["revision"]
    present = subprocess.run(["git", "-C", str(git_root), "cat-file", "-e", revision],
                             capture_output=True).returncode == 0
    if not present:
        git("-c", "http.version=HTTP/1.1", "fetch", "--depth=1", "origin", revision)
    for item in snapshot["files"]:
        if git("rev-parse", f"{revision}:{item['path']}") != item["git_blob_oid"]:
            raise ValueError(f"Git object digest mismatch: {item['path']}")
    for item in lock["files"]:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts or relative.suffix in {".safetensors", ".bin", ".pt", ".pth"}:
            raise ValueError(f"not a small source authority: {relative}")
        destination = output / "files" / relative
        if destination.is_file() and hashlib.sha256(destination.read_bytes()).hexdigest() == item["sha256"]:
            continue
        with urlopen(item["url"], timeout=90) as response:
            data = response.read(32 * 1024 * 1024 + 1)
        if len(data) > 32 * 1024 * 1024:
            raise ValueError(f"source authority exceeds 32 MiB: {relative}")
        if hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError(f"SHA256 mismatch: {relative}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    return git_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(materialize(args.model_root.resolve(), args.output.resolve()))


if __name__ == "__main__":
    main()
