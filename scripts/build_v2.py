#!/usr/bin/env python3
"""Compile one IR-first V2 model catalog into a static viewer bundle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from llm_arch_v2 import compile_catalog, write_bundle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="catalog model directory name")
    parser.add_argument("--catalog-root", type=Path, default=REPO_ROOT / "catalog")
    parser.add_argument("--docs-root", type=Path, default=REPO_ROOT / "docs")
    parser.add_argument(
        "--output-model-id",
        help="viewer model id; defaults to '<model>_v2' to avoid replacing legacy data",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_root = (args.catalog_root / args.model).resolve()
    output_model_id = args.output_model_id or f"{args.model}_v2"
    output_path = args.docs_root / output_model_id / "arch_data.json"
    bundle = compile_catalog(model_root)
    write_bundle(bundle, output_path)
    print(f"wrote {output_path}")
    print(
        "V2 bundle: "
        f"execution_variants={len(bundle['execution_variants'])} "
        f"implementations={len(bundle['implementations'])} "
        f"profiles={len(bundle['profiles'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
