#!/usr/bin/env python3
"""Compile and audit the complete DeepSeek-V4-Pro vLLM eager rank matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.deepseek_v4_pro.build.compile_deepseek_v4_pro_vllm_eager_contract import (
    compile_contract,
    load_json,
    load_jsonl,
    write_json,
    write_jsonl,
)


SOURCE_COMMIT = "dd10e03f95f94edbea1975c67ace3a35ec9a8a40"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rank_from_dir(path: Path) -> int:
    name = path.name
    if not name.startswith("rank") or not name[4:].isdigit():
        raise ValueError(f"invalid rank directory: {path}")
    return int(name[4:])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    mapping_root = args.mapping_root.resolve()
    output_root = args.out_root.resolve()

    reports: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    errors: list[str] = []
    rank_dirs = sorted(
        path.parent for path in mapping_root.glob("**/validation_report.json")
    )
    if len(rank_dirs) != 40:
        errors.append(f"expected 40 rank mappings, got {len(rank_dirs)}")

    for rank_dir in rank_dirs:
        rank = _rank_from_dir(rank_dir)
        mapping_path = rank_dir / f"kernel_mapping.tp{rank}.jsonl"
        manifest_path = rank_dir / "input_manifest.json"
        validation_path = rank_dir / "validation_report.json"
        relative = rank_dir.relative_to(mapping_root)
        out_dir = output_root / relative
        try:
            manifest = load_json(manifest_path)
            rows, report = compile_contract(
                load_jsonl(mapping_path),
                manifest,
                load_json(validation_path),
            )
            if manifest.get("rank") != rank:
                report["errors"].append(
                    f"manifest rank {manifest.get('rank')} != directory rank {rank}"
                )
            if manifest.get("source_commit") != SOURCE_COMMIT:
                report["errors"].append("source commit mismatch")
            report["ok"] = not report["errors"]
            write_jsonl(out_dir / "eager_contract.jsonl", rows)
            write_json(out_dir / "eager_contract_report.json", report)
            report = {
                **report,
                "profile_key": str(relative.parent),
                "rank": rank,
            }
            reports.append(report)
            for kind, path in (
                ("input_mapping", mapping_path),
                ("input_manifest", manifest_path),
                ("input_validation", validation_path),
                ("compiled_contract", out_dir / "eager_contract.jsonl"),
                ("compiled_report", out_dir / "eager_contract_report.json"),
            ):
                artifacts.append(
                    {
                        "kind": kind,
                        "path": str(path.relative_to(mapping_root))
                        if path.is_relative_to(mapping_root)
                        else str(path.relative_to(output_root)),
                        "sha256": sha256_file(path),
                        "bytes": path.stat().st_size,
                    }
                )
        except Exception as exc:  # fail closed and retain the exact rank failure
            errors.append(f"{relative}: {type(exc).__name__}: {exc}")

    profiles: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for report in reports:
        profiles[str(report["profile_key"])].append(report)
        if not report["ok"]:
            errors.extend(
                f"{report['profile_key']}/rank{report['rank']}: {error}"
                for error in report["errors"]
            )

    phase_counts: Counter[str] = Counter()
    profile_reports: dict[str, Any] = {}
    for profile_key, rank_reports in sorted(profiles.items()):
        rank_reports.sort(key=lambda report: int(report["rank"]))
        ranks = [int(report["rank"]) for report in rank_reports]
        phases = {str(report["phase"]) for report in rank_reports}
        if ranks != list(range(8)):
            errors.append(f"{profile_key}: expected ranks 0..7, got {ranks}")
        if len(phases) != 1:
            errors.append(f"{profile_key}: phase mismatch {sorted(phases)}")
        phase = next(iter(phases), "unknown")
        phase_counts[phase] += 1
        kernel_counts = {int(report["kernel_count"]) for report in rank_reports}
        occurrence_counts = {
            int(report["occurrence_count"]) for report in rank_reports
        }
        if len(kernel_counts) != 1:
            errors.append(f"{profile_key}: rank kernel-count mismatch")
        if occurrence_counts != {122}:
            errors.append(f"{profile_key}: rank occurrence-count mismatch")
        profile_reports[profile_key] = {
            "phase": phase,
            "ranks": ranks,
            "kernel_count_per_rank": sorted(kernel_counts),
            "occurrence_count_per_rank": sorted(occurrence_counts),
            "rank_execution_fingerprints": {
                str(report["rank"]): report["ordered_execution_fingerprint"]
                for report in rank_reports
            },
            "rank_semantic_count_fingerprints": {
                str(report["rank"]): hashlib.sha256(
                    json.dumps(
                        report["node_counts"], sort_keys=True, separators=(",", ":")
                    ).encode()
                ).hexdigest()
                for report in rank_reports
            },
            "rank_window_duration_ms": {
                str(report["rank"]): (report.get("window") or {}).get("duration_ms")
                for report in rank_reports
            },
            "support_class_counts": rank_reports[0]["support_class_counts"]
            if rank_reports
            else {},
            "rank_node_counts": {
                str(report["rank"]): report["node_counts"] for report in rank_reports
            },
        }

    if phase_counts != Counter({"vllm_prefill": 1, "vllm_decode": 4}):
        errors.append(
            "expected one prefill and four decode profiles, got "
            + json.dumps(dict(phase_counts), sort_keys=True)
        )

    matrix_report = {
        "ok": not errors,
        "errors": errors,
        "source_commit": SOURCE_COMMIT,
        "profile_count": len(profiles),
        "rank_mapping_count": len(reports),
        "phase_profile_counts": dict(phase_counts),
        "profiles": profile_reports,
        "artifact_count": len(artifacts),
    }
    write_json(output_root / "matrix_report.json", matrix_report)
    write_json(output_root / "artifact_manifest.json", {"artifacts": artifacts})
    print(
        f"ok={matrix_report['ok']} profiles={len(profiles)} "
        f"rank_mappings={len(reports)} artifacts={len(artifacts)}"
    )
    for error in errors:
        print(f"error: {error}")
    return 0 if matrix_report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
