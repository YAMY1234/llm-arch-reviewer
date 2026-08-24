#!/usr/bin/env python3
"""Backfill QSA-indexer drill mappings from immutable timeline evidence.

This migration does not reinterpret the parent QSA attribution. It refines
already-attributed ``qsa_attention.indexer`` / ``mtp_qsa_attention.indexer``
events into the new drill view, updates the timeline target list, and derives
overlap-aware child metrics from the same measured intervals.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.qwen40.build.qwen40_decode_attribution import (  # noqa: E402
    _QSA_INDEXER_PARENT_NODES,
    attach_qsa_indexer_drill_metrics,
    attach_qsa_indexer_drill_targets,
)


def _string(strings: list[str], value: Any) -> Any:
    return strings[value] if isinstance(value, int) else value


def _decode_events(artifact: dict[str, Any]) -> tuple[list[dict[str, Any]], list[tuple[dict[str, Any], dict[str, Any]]]]:
    strings = artifact["strings"]
    events: list[dict[str, Any]] = []
    encoded_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for step in artifact["steps"]:
        trace_start = float(step["trace_start_us"])
        for encoded in step.get("events") or []:
            event = {
                "step_index": int(step["step_index"]),
                "node": _string(strings, encoded.get("ir_node")),
                "kernel_name": _string(strings, encoded.get("kernel_name")),
                "kernel_label": _string(strings, encoded.get("kernel_label")),
                "attribution_method": _string(
                    strings, encoded.get("attribution_method")
                ),
                "confidence": _string(strings, encoded.get("confidence")),
                "layer_id": encoded.get("layer_id"),
                "layer_kind": _string(strings, encoded.get("layer_kind")),
                "substage": _string(strings, encoded.get("substage")),
                "cpu_op_name": _string(strings, encoded.get("cpu_op_name")),
                "ts_us": trace_start + float(encoded["start_us"]),
                "dur_us": float(encoded["duration_us"]),
                "stream": encoded.get("stream_id"),
                "device": encoded.get("device"),
                "pid": encoded.get("pid"),
                "tid": encoded.get("tid"),
            }
            events.append(event)
            encoded_pairs.append((event, encoded))
    return events, encoded_pairs


def _add_string(artifact: dict[str, Any], value: str) -> int:
    strings = artifact["strings"]
    try:
        return strings.index(value)
    except ValueError:
        strings.append(value)
        return len(strings) - 1


def _gzip_json(document: dict[str, Any]) -> bytes:
    payload = json.dumps(document, separators=(",", ":")).encode()
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as stream:
        stream.write(payload)
    return output.getvalue()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def backfill(profile_path: Path, *, check: bool) -> tuple[bool, str]:
    profile = yaml.safe_load(profile_path.read_text())
    metrics = profile.get("node_metrics") or {}
    parents = sorted(set(metrics).intersection(_QSA_INDEXER_PARENT_NODES))
    if not parents or not profile.get("timeline"):
        return False, "no QSA indexer timeline scope"

    artifact_path = profile_path.parent / profile["timeline"]["artifact"]
    with gzip.open(artifact_path, "rt") as stream:
        artifact = json.load(stream)
    events, encoded_pairs = _decode_events(artifact)
    attach_qsa_indexer_drill_targets(events)

    scoped_metrics = {parent: copy.deepcopy(metrics[parent]) for parent in parents}
    attach_qsa_indexer_drill_metrics(
        scoped_metrics,
        events,
        n_iters=len(artifact["steps"]),
        all_events=events,
    )
    coverages = []
    for parent in parents:
        cell = scoped_metrics[parent]
        if "drill_metrics" not in cell:
            raise ValueError(f"{profile_path}: no drill metrics derived for {parent}")
        source_rank = metrics[parent].get("source_rank")
        rank_policy = metrics[parent].get("rank_policy")
        for child in cell["drill_metrics"].values():
            if child.get("active_gpu_ms") is not None:
                child["source_rank"] = source_rank
                child["rank_policy"] = rank_policy
        metrics[parent] = cell
        coverages.append(float(cell["drill_mapping_coverage_pct"]))

    for event, encoded in encoded_pairs:
        target = event.get("qsa_indexer_drill_target")
        if not target:
            continue
        target_index = _add_string(artifact, str(target))
        encoded.setdefault("ir_targets", [])
        if target_index not in encoded["ir_targets"]:
            encoded["ir_targets"].append(target_index)

    artifact_payload = _gzip_json(artifact)
    profile["timeline"]["sha256"] = _sha256(artifact_payload)
    profile_text = yaml.safe_dump(profile, sort_keys=False, allow_unicode=True)
    changed = artifact_path.read_bytes() != artifact_payload or profile_path.read_text() != profile_text
    if changed and not check:
        artifact_path.write_bytes(artifact_payload)
        profile_path.write_text(profile_text, encoding="utf-8")
    return changed, f"drill residency coverage min={min(coverages):.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catalog-root",
        type=Path,
        default=REPO_ROOT / "catalog" / "qwen40" / "profiles",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    changed = 0
    processed = 0
    for profile_path in sorted(args.catalog_root.rglob("*.yaml")):
        did_change, summary = backfill(profile_path, check=args.check)
        if summary == "no QSA indexer timeline scope":
            continue
        processed += 1
        changed += int(did_change)
        print(f"{profile_path.relative_to(REPO_ROOT)}: {summary}")
    print(f"processed={processed} changed={changed} check={args.check}")
    if args.check and changed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
