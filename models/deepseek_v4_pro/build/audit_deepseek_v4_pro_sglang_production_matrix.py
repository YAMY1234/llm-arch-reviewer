#!/usr/bin/env python3
"""Audit all five DeepSeek-V4-Pro SGLang production reconciliations."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
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
from models.deepseek_v4_pro.build.audit_deepseek_v4_pro_vllm_eager_matrix import (
    verify_artifact_manifest,
)


SOURCE_COMMIT = "71de97b264b04dcd514cf904003028aefe9775c8"
PROFILES = {
    "prefill-c1": {
        "status": "unsupported",
        "phase": "prefill",
        "batch_size": 1,
        "job_id": "3426447",
        "production_kind": "sglang-prefill_timing",
        "kernel_count": 3034,
        "graph_dependencies": 0,
        "mhc_path": "separate_post_pre",
    },
    "decode-c1": {
        "status": "measured",
        "phase": "decode",
        "batch_size": 1,
        "job_id": "3424801",
        "production_kind": "sglang-production",
        "kernel_count": 2675,
        "graph_dependencies": 3,
        "mhc_path": "fused_post_pre",
    },
    "decode-c16": {
        "status": "measured",
        "phase": "decode",
        "batch_size": 16,
        "job_id": "3424802",
        "production_kind": "sglang-production",
        "kernel_count": 2736,
        "graph_dependencies": 2,
        "mhc_path": "fused_post_pre",
    },
    "decode-c64": {
        "status": "measured",
        "phase": "decode",
        "batch_size": 64,
        "job_id": "3424803",
        "production_kind": "sglang-production",
        "kernel_count": 2857,
        "graph_dependencies": 2,
        "mhc_path": "separate_post_pre",
    },
    "decode-c256": {
        "status": "measured",
        "phase": "decode",
        "batch_size": 256,
        "job_id": "3424804",
        "production_kind": "sglang-production",
        "kernel_count": 2796,
        "graph_dependencies": 2,
        "mhc_path": "separate_post_pre",
    },
}


def audit_collective_rank_durations(
    rank_rows: dict[int, list[dict[str, Any]]],
) -> tuple[dict[str, Any], list[str]]:
    """Reject rank-local profiler activation skew in identical collectives."""

    by_rank: dict[int, dict[tuple[str, str, str, int], float]] = {}
    for rank, rows in rank_rows.items():
        ordinals: Counter[tuple[str, str, str]] = Counter()
        signatures: dict[tuple[str, str, str, int], float] = {}
        for row in rows:
            node = str(row.get("node") or "")
            if not node.endswith("collective"):
                continue
            base = (
                node,
                str(row.get("occurrence_id") or "top"),
                str(row.get("kernel_name") or ""),
            )
            ordinal = ordinals[base]
            ordinals[base] += 1
            signatures[(*base, ordinal)] = float(row.get("dur_us") or 0.0)
        by_rank[rank] = signatures

    errors: list[str] = []
    all_signatures = set().union(*(set(rows) for rows in by_rank.values()))
    outliers: list[dict[str, Any]] = []
    worst: dict[str, Any] | None = None
    for signature in sorted(all_signatures):
        durations = {
            str(rank): rows[signature]
            for rank, rows in sorted(by_rank.items())
            if signature in rows
        }
        if set(durations) != {str(rank) for rank in range(8)}:
            errors.append(
                f"collective signature is not present on all ranks: {signature}"
            )
            continue
        minimum = min(durations.values())
        maximum = max(durations.values())
        ratio = maximum / max(minimum, 1e-9)
        row = {
            "node": signature[0],
            "occurrence_id": signature[1],
            "kernel_name": signature[2],
            "ordinal": signature[3],
            "rank_duration_us": durations,
            "min_us": minimum,
            "max_us": maximum,
            "max_to_min_ratio": ratio,
        }
        if worst is None or row["max_to_min_ratio"] > worst["max_to_min_ratio"]:
            worst = row
        if maximum >= 100.0 and maximum - minimum >= 500.0 and ratio > 8.0:
            outliers.append(row)
            errors.append(
                "collective rank-duration activation skew: "
                f"{signature[0]} {signature[1]} ordinal={signature[3]} "
                f"min={minimum:.3f}us max={maximum:.3f}us ratio={ratio:.3f}"
            )
    return {
        "policy": "same formal step and exact collective signature must exist on all 8 ranks; reject max>=100us, spread>=500us, and max/min>8",
        "signature_count": len(all_signatures),
        "outlier_count": len(outliers),
        "worst_signature": worst,
        "rank_max_collective_duration_us": {
            str(rank): max(rows.values(), default=0.0)
            for rank, rows in sorted(by_rank.items())
        },
    }, errors


def audit_prefill_prime_coordinate(client: dict[str, Any]) -> list[str]:
    """Require one last-warmup decode prime immediately before formal prefill."""

    coordinate = client.get("profile_coordinate") or {}
    controls = client.get("profile_controls") or []
    control_request = controls[0].get("request") if len(controls) == 1 else {}
    start_step = coordinate.get("resolved_absolute_start_step")
    target_step = coordinate.get("resolved_absolute_target_step")
    formal_step = coordinate.get("formal_start_forward_ct")
    contract = client.get("contract") or {}
    checks = {
        "profile_coordinate_mode": coordinate.get("mode")
        == "last_warmup_decode_prime_then_formal_prefill",
        "prime_immediately_precedes_formal": isinstance(start_step, int)
        and start_step + 1 == formal_step == target_step,
        "exact_warmup_formal_contract": contract.get("warmup_request_count") == 3
        and contract.get("formal_request_count") == 1,
        "profile_control_prime_plus_formal": control_request.get("start_step")
        == start_step
        and control_request.get("num_steps") == 2,
    }
    return [check for check, passed in checks.items() if not passed]


def audit(root: Path, evidence_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    errors: list[str] = []
    profiles: dict[str, Any] = {}
    artifacts: list[dict[str, Any]] = []
    baseline_path = evidence_root / "sglang-baseline/3417439/baseline-selection.json"
    if not baseline_path.is_file():
        errors.append(f"missing retained profiler-off baseline: {baseline_path}")
    else:
        artifacts.append(
            {
                "kind": "profiler_off_baseline_selection",
                "root": "evidence",
                "path": baseline_path.relative_to(evidence_root).as_posix(),
                "bytes": baseline_path.stat().st_size,
                "sha256": sha256_file(baseline_path),
            }
        )
    for profile_id, expected in PROFILES.items():
        job_root = evidence_root / expected["production_kind"] / expected["job_id"]
        raw_trace_by_rank: dict[int, Path] = {}
        for rank in range(8):
            matches = sorted((job_root / "traces").glob(f"*TP-{rank}.trace.json.gz"))
            if len(matches) != 1:
                errors.append(
                    f"{profile_id}/rank{rank}: expected one retained raw trace, got {matches}"
                )
                continue
            raw_trace_by_rank[rank] = matches[0]
            artifacts.append(
                {
                    "kind": "raw_production_trace",
                    "root": "evidence",
                    "path": matches[0].relative_to(evidence_root).as_posix(),
                    "bytes": matches[0].stat().st_size,
                    "sha256": sha256_file(matches[0]),
                }
            )
        for kind, path in (
            ("run_validation", job_root / "validation.json"),
            ("profiler_overlay_source_lock", job_root / "profiler-overlay-source-lock.json"),
            ("exact_serving_client", job_root / f"client-c{expected['batch_size']}.json"),
            ("scheduler_log", job_root / "server.log"),
        ):
            if not path.is_file():
                errors.append(f"{profile_id}: missing retained {kind}: {path}")
                continue
            artifacts.append(
                {
                    "kind": kind,
                    "root": "evidence",
                    "path": path.relative_to(evidence_root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        validation_path = job_root / "validation.json"
        if validation_path.is_file():
            validation = load_json(validation_path)
            if validation.get("status") != "pass":
                errors.append(f"{profile_id}: retained run validation did not pass")
            if validation.get("trace_ranks") != list(range(8)):
                errors.append(f"{profile_id}: retained run validation lacks ranks 0..7")
        client_path = job_root / f"client-c{expected['batch_size']}.json"
        client = load_json(client_path) if client_path.is_file() else {}
        if expected["phase"] == "prefill" and client:
            errors.extend(
                f"{profile_id}: {check} failed"
                for check in audit_prefill_prime_coordinate(client)
            )

        rank_reports: dict[str, Any] = {}
        rank_structural_fingerprints: dict[str, str] = {}
        rank_structural_multiset_fingerprints: dict[str, str] = {}
        rank_event_rows: dict[int, list[dict[str, Any]]] = {}
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
                        "kind": "production_reconciliation",
                        "root": "reconciliation",
                        "path": path.relative_to(root).as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
            if not report_path.is_file() or not events_path.is_file():
                continue
            report = load_json(report_path)
            rows = load_jsonl(events_path)
            rank_event_rows[rank] = rows
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
                "retained_raw_trace_hash": rank in raw_trace_by_rank
                and (report.get("trace") or {}).get("sha256")
                == sha256_file(raw_trace_by_rank[rank]),
            }
            if expected["phase"] == "decode":
                gate = report.get("formal_step_throughput_gate") or {}
                target = gate.get("formal_target") or {}
                checks.update(
                    {
                        "run_validation_passed": (
                            report.get("run_validation") or {}
                        ).get("status")
                        == "pass",
                        "formal_step_is_second_launch_scheduler_coordinate": gate.get(
                            "formal_target_step"
                        )
                        == gate.get("profile_start_step"),
                        "one_activation_priming_launch": (
                            report.get("window_selector") or {}
                        ).get("profile_priming_launch_count")
                        == 1,
                        "formal_step_throughput_not_collapsed": float(
                            target.get("throughput_token_s") or 0.0
                        )
                        >= float(
                            gate.get("minimum_accepted_throughput_token_s") or 1.0
                        ),
                    }
                )
            else:
                checks["one_activation_priming_launch"] = (
                    report.get("window_selector") or {}
                ).get("profile_priming_launch_count") == 1
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
        collective_audit, collective_errors = audit_collective_rank_durations(
            rank_event_rows
        )
        if expected["status"] == "unsupported":
            if not collective_errors:
                errors.append(
                    f"{profile_id}: expected evidence-backed collective skew rejection"
                )
            unexpected = [
                error
                for error in collective_errors
                if "collective rank-duration activation skew" not in error
            ]
            errors.extend(f"{profile_id}: {error}" for error in unexpected)
            profile_status = "unsupported"
            unsupported_reason = (
                "two independent synchronized 8-rank captures retained exact "
                "formal prefill windows but failed the collective rank-duration "
                "outlier gate; instrumented timing is not publishable"
            )
        else:
            errors.extend(f"{profile_id}: {error}" for error in collective_errors)
            profile_status = "measured"
            unsupported_reason = None
        profiles[profile_id] = {
            **expected,
            "status": profile_status,
            "unsupported_reason": unsupported_reason,
            "rank_count": len(rank_reports),
            "retained_raw_trace_ranks": sorted(raw_trace_by_rank),
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
            "collective_rank_duration_audit": collective_audit,
            "formal_step_throughput_gate": (
                next(iter(rank_reports.values())).get("formal_step_throughput_gate")
                if expected["phase"] == "decode" and rank_reports
                else None
            ),
        }

    task_root = Path(os.path.commonpath((root, evidence_root)))
    manifest = {
        "schema_version": "deepseek-v4-pro-sglang-production-artifacts.v2",
        "root_contract": {
            "reconciliation": str(root.relative_to(task_root)),
            "evidence": str(evidence_root.relative_to(task_root)),
        },
        "artifacts": sorted(
            artifacts, key=lambda row: (row["root"], row["path"], row["kind"])
        ),
    }
    retention_errors = verify_artifact_manifest(
        manifest, roots={"reconciliation": root, "evidence": evidence_root}
    )
    errors.extend(retention_errors)
    report = {
        "schema_version": "deepseek-v4-pro-sglang-production-matrix-audit.v1",
        "ok": not errors,
        "errors": errors,
        "source_commit": SOURCE_COMMIT,
        "profile_count": len(PROFILES),
        "measured_profile_count": sum(
            profile["status"] == "measured" for profile in profiles.values()
        ),
        "unsupported_profile_count": sum(
            profile["status"] == "unsupported" for profile in profiles.values()
        ),
        "rank_reconciliation_count": sum(
            profile["rank_count"] for profile in profiles.values()
        ),
        "phase_profile_counts": dict(
            Counter(profile["phase"] for profile in PROFILES.values())
        ),
        "artifact_count": len(artifacts),
        "raw_trace_count": sum(
            len(profile["retained_raw_trace_ranks"]) for profile in profiles.values()
        ),
        "retention_gate": {
            "ok": not retention_errors,
            "checked_artifact_count": len(artifacts),
            "errors": retention_errors,
        },
        "profiles": profiles,
    }
    return report, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)
    args = parser.parse_args()
    report, manifest = audit(args.root.resolve(), args.evidence_root.resolve())
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
