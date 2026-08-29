#!/usr/bin/env python3
"""Audit pinned construction source against Model IR in both directions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from llm_arch_v2.semantic_audit import (  # noqa: E402
    audit_semantic_closure,
    render_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source-repo", required=True, type=Path)
    parser.add_argument("--catalog-root", type=Path, default=REPO_ROOT / "catalog")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="write a gap report and return zero even when closure is incomplete",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_root = args.catalog_root / args.model
    report = audit_semantic_closure(
        model_ir_path=model_root / "model_ir.yaml",
        ledger_path=model_root / "semantic_source_ledger.yaml",
        source_repo=args.source_repo.resolve(),
    )
    rendered = render_markdown(report)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(rendered)
    if not args.json_out and not args.markdown_out:
        print(rendered)
    if report["status"] != "complete" and not args.allow_incomplete:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
