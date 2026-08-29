#!/usr/bin/env python3
"""Compile one or every audited IR-first V2 catalog into static viewer bundles."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from llm_arch_v2 import compile_catalog, write_bundle  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_timeline_artifacts(model_root: Path, output_dir: Path) -> int:
    timeline_dir = output_dir / "timelines"
    if timeline_dir.exists():
        shutil.rmtree(timeline_dir)
    copied = 0
    for profile_path in sorted(model_root.glob("profiles/*/*/*.yaml")):
        profile = yaml.safe_load(profile_path.read_text())
        timeline = (profile or {}).get("timeline")
        if not timeline:
            continue
        source = profile_path.parent / str(timeline["artifact"])
        if not source.is_file():
            raise FileNotFoundError(f"missing timeline artifact: {source}")
        actual_sha256 = sha256_file(source)
        if actual_sha256 != timeline["sha256"]:
            raise ValueError(
                f"timeline SHA256 mismatch for {source}: "
                f"{actual_sha256} != {timeline['sha256']}"
            )
        destination = (
            output_dir / "timelines" / f"{profile['profile_id']}.timeline.json.gz"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        copied += 1
    return copied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--model",
        action="append",
        help="catalog model directory name; repeatable",
    )
    selection.add_argument(
        "--all",
        action="store_true",
        help="compile every catalog directory containing model_ir.yaml",
    )
    parser.add_argument("--catalog-root", type=Path, default=REPO_ROOT / "catalog")
    parser.add_argument("--docs-root", type=Path, default=REPO_ROOT / "docs")
    parser.add_argument(
        "--output-model-id",
        help="viewer model id; defaults to '<model>_v2'",
    )
    return parser.parse_args()


def build_model(
    model_name: str,
    *,
    catalog_root: Path,
    docs_root: Path,
    output_model_id: str | None = None,
) -> None:
    model_root = (catalog_root / model_name).resolve()
    output_model_id = output_model_id or f"{model_name}_v2"
    output_path = docs_root / output_model_id / "arch_data.json"
    bundle = compile_catalog(model_root)
    write_bundle(bundle, output_path)
    timeline_count = copy_timeline_artifacts(model_root, output_path.parent)
    print(f"wrote {output_path}")
    print(
        "V2 bundle: "
        f"execution_variants={len(bundle['execution_variants'])} "
        f"implementations={len(bundle['implementations'])} "
        f"profiles={len(bundle['profiles'])}"
        f" timelines={timeline_count}"
    )


def main() -> int:
    args = parse_args()
    if args.all and args.output_model_id:
        raise SystemExit("--output-model-id is only valid with one --model")
    if args.all:
        models = sorted(
            path.name
            for path in args.catalog_root.iterdir()
            if path.is_dir() and (path / "model_ir.yaml").is_file()
        )
    else:
        models = list(dict.fromkeys(args.model or []))
    for model_name in models:
        build_model(
            model_name,
            catalog_root=args.catalog_root,
            docs_root=args.docs_root,
            output_model_id=args.output_model_id,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
