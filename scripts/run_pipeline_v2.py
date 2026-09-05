#!/usr/bin/env python3
"""Plan or accept one deterministic add-trace run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from llm_arch_v2.add_trace import (  # noqa: E402
    AddTraceError,
    accept_evidence,
    build_plan,
)
from llm_arch_v2.compiler import CatalogError  # noqa: E402


def _load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise AddTraceError(f"{path}: expected a YAML/JSON mapping")
    return value


def _write(value: dict, output: Path | None) -> None:
    rendered = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if output is None:
        sys.stdout.write(rendered)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan", help="resolve config to Execution and Binding")
    plan.add_argument("--manifest", type=Path, required=True)
    plan.add_argument("--catalog-root", type=Path, default=REPO_ROOT / "catalog")
    plan.add_argument("--output", type=Path)

    accept = subparsers.add_parser(
        "accept", help="validate eager and production evidence against one plan"
    )
    accept.add_argument("--manifest", type=Path, required=True)
    accept.add_argument("--plan", type=Path, required=True)
    accept.add_argument("--binding-revision", type=Path, required=True)
    accept.add_argument("--eager-reconciliation", type=Path, required=True)
    accept.add_argument("--trace-attribution", type=Path, required=True)
    accept.add_argument("--catalog-root", type=Path, default=REPO_ROOT / "catalog")
    accept.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = _load(args.manifest)
        if args.command == "plan":
            model_root = args.catalog_root / manifest.get("model_id", "")
            result = build_plan(manifest, model_root=model_root, source=args.manifest)
        else:
            result = accept_evidence(
                manifest,
                _load(args.plan),
                _load(args.binding_revision),
                _load(args.eager_reconciliation),
                _load(args.trace_attribution),
                model_root=args.catalog_root / manifest.get("model_id", ""),
                source=args.manifest,
                verify_files=True,
            )
        _write(result, args.output)
        return 0
    except (AddTraceError, CatalogError, OSError) as exc:
        sys.stderr.write(f"add-trace pipeline failed closed: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
