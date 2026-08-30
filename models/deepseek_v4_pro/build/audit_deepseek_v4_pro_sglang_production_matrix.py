#!/usr/bin/env python3
"""Audit all five DeepSeek-V4-Pro SGLang production reconciliations."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.deepseek_v4_pro.build.audit_deepseek_v4_pro_vllm_production_matrix import (
    load_json,
    load_jsonl,
    sha256_file,
    structural_fingerprint,
    structural_multiset_fingerprint,
)


SOURCE_COMMIT = "71de97b264b04dcd514cf904003028aefe9775c8"
PROFILES = {
    "prefill-c1": {
        "phase": "prefill",
        "batch_size": 1,
        "job_id": "3422982",
        "kernel_count": 3034,
        "graph_dependencies": 0,
        "mhc_path": "separate_post_pre",
    },
    "decode-c1": {
        "phase": "decode",
        "batch_size": 1,
        "job_id": "3422983",
        "kernel_count": 2675,
        "graph_dependencies": 3,
        "mhc_path": "fused_post_pre",
    },
    "decode-c16": {
        "phase": "decode",
        "batch_size": 16,
        "job_id": "3422984",
        "kernel_count": 2736,
        "graph_dependencies": 2,
        "mhc_path": "fused_post_pre",
    },
    "decode-c64": {
        "phase": "decode",
        "batch_size": 64,
        "job_id": "3422985",
        "kernel_count": 2857,
        "graph_dependencies": 2,
        "mhc_path": "separate_post_pre",
    },
    "decode-c256": {
        "phase": "decode",
        "batch_size": 256,
        "job_id": "3422986",
        "kernel_count": 2796,
        "graph_dependencies": 2,
        "mhc_path": "separate_post_pre",
    },
}


def audit(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    errors: list[str] = []
    profiles: dict[str, Any] = {}
    artifacts: list[dict[str, Any]] = []
    for profile_id, expected in PROFILES.items():
        rank_reports: dict[str, Any] = {}
        rank_structural_fingerprints: dict[str, str] = {}
        rank_structural_multiset_fingerprints: dict[str, str] = {}
        for rank in range(8):
            rank_root = root / profile_id / f"rank{rank}"
            report_path = rank_root / "report.json"
            events_path = rank_root / "events.jsonl"
            for path in (report_path, events_path):
                if not path.is_file():
                    errors.append(f"{profile_id}/rank{rank}: missing {path.name}")
                    continue
                artifacts.append(
                    {
                        "path": path.relative_to(root).as_posix(),
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
            if not report_path.is_file() or not events_path.is_file():
                continue
            report = load_json(report_path)
            rows = load_jsonl(events_path)
            rank_reports[str(rank)] = report
            rank_structural_fingerprints[str(rank)] = structural_fingerprint(rows)
            rank_structural_multiset_fingerprints[str(rank)] = (
                structural_multiset_fingerprint(rows)
            )
            checks = {
                "ok": report.get("ok") is True,
                "source_commit": report.get("source_commit") == SOURCE_COMMIT,
                "phase": report.get("phase") == expected["phase"],
                "global_batch_size": report.get("global_batch_size")
                == expected["batch_size"],
                "job_id": report.get("job_id") == expected["job_id"],
                "rank": report.get("rank") == rank,
                "kernel_count": report.get("kernel_count") == expected["kernel_count"],
                "mapped_kernel_count": report.get("mapped_kernel_count")
                == expected["kernel_count"],
                "mapped_kernel_count_ratio": report.get("mapped_kernel_count_ratio")
                == 1.0,
                "mapped_kernel_duration_ratio": report.get(
                    "mapped_kernel_duration_ratio"
                )
                == 1.0,
                "selected_wall_elapsed_us": float(
                    report.get("selected_wall_elapsed_us") or 0.0
                )
                > 0.0,
                "top_level_timing_semantics": (
                    report.get("timing_semantics") or {}
                ).get("top_level_runtime")
                == "selected_wall_elapsed_us",
                "occurrence_count": report.get("occurrence_count") == 122,
                "graph_dependency_kernel_count": report.get(
                    "graph_dependency_kernel_count"
                )
                == expected["graph_dependencies"],
                "mhc_implementation_path": report.get("mhc_implementation_path")
                == expected["mhc_path"],
                "event_row_count": len(rows) == expected["kernel_count"],
                "all_events_mapped": all(
                    row.get("node") and row.get("eager_event_ids") for row in rows
                ),
            }
            for check, passed in checks.items():
                if not passed:
                    errors.append(f"{profile_id}/rank{rank}: {check} failed")

        node_count_fingerprints = {
            json.dumps(report.get("node_counts"), sort_keys=True)
            for report in rank_reports.values()
        }
        if len(rank_reports) != 8:
            errors.append(f"{profile_id}: expected 8 rank reports, got {len(rank_reports)}")
        if len(node_count_fingerprints) != 1:
            errors.append(f"{profile_id}: rank node-count structures differ")
        if len(set(rank_structural_multiset_fingerprints.values())) != 1:
            errors.append(f"{profile_id}: rank structural multisets differ")
        profiles[profile_id] = {
            **expected,
            "rank_count": len(rank_reports),
            "node_counts": next(iter(rank_reports.values())).get("node_counts")
            if rank_reports
            else None,
            "structural_multiset_fingerprint": next(
                iter(rank_structural_multiset_fingerprints.values()), None
            ),
            "rank_ordering_policy": "rank_specific_multistream_order_preserved",
            "rank_ordered_structural_fingerprints": rank_structural_fingerprints,
            "rank_reconciliation_fingerprints": {
                rank: report.get("ordered_reconciliation_fingerprint")
                for rank, report in rank_reports.items()
            },
            "rank_total_kernel_us": {
                rank: report.get("total_kernel_us")
                for rank, report in rank_reports.items()
            },
            "rank_selected_wall_elapsed_us": {
                rank: report.get("selected_wall_elapsed_us")
                for rank, report in rank_reports.items()
            },
        }

    report = {
        "schema_version": "deepseek-v4-pro-sglang-production-matrix-audit.v1",
        "ok": not errors,
        "errors": errors,
        "source_commit": SOURCE_COMMIT,
        "profile_count": len(PROFILES),
        "rank_reconciliation_count": sum(
            profile["rank_count"] for profile in profiles.values()
        ),
        "phase_profile_counts": dict(
            Counter(profile["phase"] for profile in PROFILES.values())
        ),
        "artifact_count": len(artifacts),
        "profiles": profiles,
    }
    manifest = {
        "schema_version": "deepseek-v4-pro-sglang-production-artifacts.v1",
        "artifacts": sorted(artifacts, key=lambda row: row["path"]),
    }
    return report, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)
    args = parser.parse_args()
    report, manifest = audit(args.root)
    for path, value in (
        (args.out_report, report),
        (args.out_manifest, manifest),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(
        f"ok={report['ok']} profiles={report['profile_count']} "
        f"rank_reconciliations={report['rank_reconciliation_count']} "
        f"artifacts={report['artifact_count']}"
    )
    for error in report["errors"]:
        print(f"error: {error}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
