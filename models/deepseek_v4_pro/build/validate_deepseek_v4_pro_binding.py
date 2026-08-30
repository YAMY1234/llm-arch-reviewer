#!/usr/bin/env python3
"""Validate a commit-specific DeepSeek-V4-Pro binding against exact source."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml


def load_json(path: Path) -> Any:
    with path.open() as source:
        return json.load(source)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_execution_nodes(bundle: dict[str, Any], fingerprint: str) -> set[str]:
    execution = bundle["execution_variants"][fingerprint]
    return {
        f"{view_id}.{node['id']}"
        for view_id, view in execution["views"].items()
        for node in view["nodes"]
    }


def symbol_tokens(symbol: str) -> list[str]:
    return [token for token in re.split(r"[^A-Za-z0-9_]+", symbol) if token]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--eager-matrix-report", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    binding = yaml.safe_load(args.binding.read_text())
    bundle = load_json(args.bundle)
    source_root = args.source_root.resolve()
    fingerprint = str(binding["execution_validation"]["execution_fingerprint"])
    expected_nodes = expected_execution_nodes(bundle, fingerprint)
    actual_nodes = set(binding["node_bindings"])
    errors: list[str] = []
    if actual_nodes != expected_nodes:
        errors.append(
            f"binding node closure mismatch: missing={sorted(expected_nodes - actual_nodes)} "
            f"extra={sorted(actual_nodes - expected_nodes)}"
        )

    source_files: dict[str, dict[str, Any]] = {}
    link_count = 0
    for node_id, node_binding in binding["node_bindings"].items():
        symbols = set(str(symbol) for symbol in node_binding.get("symbols") or [])
        links = node_binding.get("links") or []
        if not links:
            errors.append(f"{node_id}: no source link")
            continue
        for link in links:
            link_count += 1
            relative = str(link["file"])
            path = source_root / relative
            if not path.is_file():
                errors.append(f"{node_id}: missing source file {relative}")
                continue
            text = path.read_text(errors="replace")
            lines = text.splitlines()
            line = int(link["line"])
            if line < 1 or line > len(lines):
                errors.append(
                    f"{node_id}: line {line} outside {relative} ({len(lines)} lines)"
                )
            symbol = str(link["symbol"])
            if symbols and symbol not in symbols:
                errors.append(f"{node_id}: link symbol {symbol} absent from symbols list")
            tokens = symbol_tokens(symbol)
            if not tokens or not all(token in text for token in tokens):
                errors.append(f"{node_id}: symbol {symbol} absent from {relative}")
            source_files.setdefault(
                relative,
                {
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                },
            )

    eager_matrix: dict[str, Any] | None = None
    if args.eager_matrix_report:
        eager_matrix = load_json(args.eager_matrix_report)
        if not eager_matrix.get("ok"):
            errors.append("eager matrix report did not pass")
        if eager_matrix.get("source_commit") != binding.get("source_commit"):
            errors.append("eager matrix source commit differs from binding")
        if eager_matrix.get("rank_mapping_count") != 40:
            errors.append("eager matrix does not contain five profiles by eight ranks")

    report = {
        "ok": not errors,
        "errors": errors,
        "implementation_id": binding["implementation_id"],
        "source_repo": binding["source_repo"],
        "source_commit": binding["source_commit"],
        "execution_fingerprint": fingerprint,
        "execution_node_count": len(expected_nodes),
        "bound_node_count": len(actual_nodes),
        "source_link_count": link_count,
        "source_file_count": len(source_files),
        "source_files": dict(sorted(source_files.items())),
        "eager_matrix": {
            "ok": eager_matrix.get("ok"),
            "profile_count": eager_matrix.get("profile_count"),
            "rank_mapping_count": eager_matrix.get("rank_mapping_count"),
            "phase_profile_counts": eager_matrix.get("phase_profile_counts"),
        }
        if eager_matrix
        else None,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as output:
        json.dump(report, output, indent=2, sort_keys=True)
        output.write("\n")
    print(
        f"ok={report['ok']} implementation={report['implementation_id']} "
        f"nodes={report['bound_node_count']} links={report['source_link_count']} "
        f"files={report['source_file_count']}"
    )
    for error in errors:
        print(f"error: {error}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
