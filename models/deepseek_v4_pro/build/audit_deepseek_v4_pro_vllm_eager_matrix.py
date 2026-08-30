#!/usr/bin/env python3
"""Compile and audit one complete DeepSeek-V4-Pro eager rank matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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


DEFAULT_SOURCE_COMMIT = "dd10e03f95f94edbea1975c67ace3a35ec9a8a40"


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


def verify_artifact_manifest(
    manifest: dict[str, Any], *, roots: dict[str, Path]
) -> list[str]:
    """Fail closed unless every published item exists under its bounded root."""

    errors: list[str] = []
    artifacts = manifest.get("artifacts") or []
    for index, artifact in enumerate(artifacts):
        root_name = str(artifact.get("root") or "")
        relative = Path(str(artifact.get("path") or ""))
        root = roots.get(root_name)
        if root is None:
            errors.append(f"artifact[{index}]: unknown root {root_name!r}")
            continue
        if relative.is_absolute() or ".." in relative.parts:
            errors.append(f"artifact[{index}]: unbounded path {relative}")
            continue
        path = root / relative
        if not path.is_file():
            errors.append(f"artifact[{index}]: missing {root_name}/{relative}")
            continue
        if path.stat().st_size != artifact.get("bytes"):
            errors.append(f"artifact[{index}]: size mismatch {root_name}/{relative}")
        if sha256_file(path) != artifact.get("sha256"):
            errors.append(f"artifact[{index}]: hash mismatch {root_name}/{relative}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument(
        "--evidence-root",
        type=Path,
        required=True,
        help="bounded evidence directory containing framework eager profile/job/traces",
    )
    parser.add_argument("--source-commit", default=DEFAULT_SOURCE_COMMIT)
    parser.add_argument("--profile-prefix", default="vllm-")
    parser.add_argument(
        "--expected-phase-counts",
        default="vllm_prefill=1,vllm_decode=4",
        help="comma-separated phase=count contract",
    )
    args = parser.parse_args()
    mapping_root = args.mapping_root.resolve()
    output_root = args.out_root.resolve()
    evidence_root = args.evidence_root.resolve()

    reports: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    errors: list[str] = []
    retained_raw_ranks: dict[str, list[int]] = defaultdict(list)
    rank_dirs = sorted(
        path.parent
        for path in mapping_root.glob(
            f"{args.profile_prefix}*/**/validation_report.json"
        )
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
            evidence_kind = relative.parts[0].replace("-prefill", "_prefill").replace(
                "-decode", "_decode"
            )
            raw_dir = evidence_root / evidence_kind / relative.parts[1] / "traces"
            raw_matches = sorted(raw_dir.glob(f"*rank{rank}.*trace.json.gz"))
            if not raw_matches and args.profile_prefix.startswith("sglang-"):
                raw_matches = sorted(raw_dir.glob(f"*TP-{rank}*.trace.json.gz"))
            if len(raw_matches) != 1:
                raise ValueError(
                    f"expected one retained raw eager trace for {relative}, got {raw_matches}"
                )
            raw_trace = raw_matches[0]
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
            if manifest.get("source_commit") != args.source_commit:
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
            retained_raw_ranks[str(relative.parent)].append(rank)
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
                        "root": "mapping" if path.is_relative_to(mapping_root) else "contract",
                        "path": str(path.relative_to(mapping_root))
                        if path.is_relative_to(mapping_root)
                        else str(path.relative_to(output_root)),
                        "sha256": sha256_file(path),
                        "bytes": path.stat().st_size,
                    }
                )
            artifacts.append(
                {
                    "kind": "raw_eager_trace",
                    "root": "evidence",
                    "path": str(raw_trace.relative_to(evidence_root)),
                    "sha256": sha256_file(raw_trace),
                    "bytes": raw_trace.stat().st_size,
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
            "retained_raw_trace_ranks": sorted(retained_raw_ranks[profile_key]),
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

    expected_phase_counts = Counter(
        {
            phase: int(count)
            for item in args.expected_phase_counts.split(",")
            for phase, count in [item.split("=", 1)]
        }
    )
    if phase_counts != expected_phase_counts:
        errors.append(
            "phase profile counts differ from the requested matrix: got "
            + json.dumps(dict(phase_counts), sort_keys=True)
        )
    for profile_key in sorted(profiles):
        retained = sorted(retained_raw_ranks[profile_key])
        if retained != list(range(8)):
            errors.append(
                f"{profile_key}: retained raw trace ranks must be 0..7, got {retained}"
            )

    task_root = Path(
        os.path.commonpath((mapping_root, output_root, evidence_root))
    )
    artifact_manifest = {
        "schema_version": "deepseek-v4-pro-eager-retention-manifest.v2",
        "root_contract": {
            "mapping": str(mapping_root.relative_to(task_root)),
            "contract": str(output_root.relative_to(task_root)),
            "evidence": str(evidence_root.relative_to(task_root)),
        },
        "artifacts": artifacts,
    }
    retention_errors = verify_artifact_manifest(
        artifact_manifest,
        roots={
            "mapping": mapping_root,
            "contract": output_root,
            "evidence": evidence_root,
        },
    )
    errors.extend(retention_errors)

    matrix_report = {
        "ok": not errors,
        "errors": errors,
        "source_commit": args.source_commit,
        "profile_count": len(profiles),
        "rank_mapping_count": len(reports),
        "phase_profile_counts": dict(phase_counts),
        "profiles": profile_reports,
        "artifact_count": len(artifacts),
        "raw_trace_count": sum(len(ranks) for ranks in retained_raw_ranks.values()),
        "retention_gate": {
            "ok": not retention_errors,
            "checked_artifact_count": len(artifacts),
            "errors": retention_errors,
        },
    }
    matrix_report["ok"] = not errors
    matrix_report["errors"] = errors
    write_json(output_root / "matrix_report.json", matrix_report)
    write_json(output_root / "artifact_manifest.json", artifact_manifest)
    print(
        f"ok={matrix_report['ok']} profiles={len(profiles)} "
        f"rank_mappings={len(reports)} artifacts={len(artifacts)}"
    )
    for error in errors:
        print(f"error: {error}")
    return 0 if matrix_report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
