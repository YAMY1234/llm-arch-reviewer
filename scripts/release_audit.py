#!/usr/bin/env python3
"""Run the model-neutral M0 release-readiness gates.

The static gate proves that every audited catalog compiles, the checked-in
bundle is exactly the compiler output, every published Timeline is the exact
content-addressed artifact declared by its Profile, and every Timeline kernel
is either bound to IR or explicitly classified as runtime/support work. It also
proves that the Model IR, Execution Contract, eager Binding reconciliation, and
production Profile each cite an independent, layer-appropriate authority rather
than using a downstream artifact as its own expectation.

The release gate adds the real-browser audits. A static pass is useful during
development, but is deliberately reported as not release ready until the
browser gate has also passed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from llm_arch_v2 import compile_catalog, validate_validation_evidence  # noqa: E402
from scripts.audit_timeline_attribution import audit_timeline  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def tree_manifest_sha256(root: Path) -> str:
    """Hash a catalog as a path-addressed, deterministic file manifest."""

    entries = [
        {
            "path": str(path.relative_to(root)),
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    ]
    return object_sha256(entries)


def build_acceptance_summary(
    *,
    repo_root: Path,
    catalog_root: Path,
    docs_root: Path,
    models: list[str],
    model_reports: list[dict[str, Any]],
    acceptance_level: str,
    static_gate: str,
    browser_report: dict[str, Any] | None,
    release_ready: bool,
) -> dict[str, Any]:
    """Build the concise, deterministic public acceptance ledger for M0."""

    reports = {report["model"]: report for report in model_reports}
    model_summaries: list[dict[str, Any]] = []
    for model in models:
        bundle = json.loads(
            (docs_root / f"{model}_v2" / "arch_data.json").read_text()
        )
        report = reports[model]
        timeline_reports = {
            item["profile"]: item for item in report.get("timelines") or []
        }
        implementations = bundle.get("implementations") or {}
        profile_summaries: list[dict[str, Any]] = []
        for profile_id, profile in sorted((bundle.get("profiles") or {}).items()):
            meta = profile.get("meta") or {}
            profiler = meta.get("profiler") or {}
            contract = {
                "phase": meta.get("phase"),
                "generation_mode": meta.get("generation_mode"),
                "execution_parameters": meta.get("execution_parameters") or {},
                "hardware": meta.get("hardware") or {},
                "workload": meta.get("workload") or {},
                "profiler": {
                    key: profiler.get(key)
                    for key in (
                        "type",
                        "cuda_graph_enabled",
                        "with_stack",
                        "formal_window_count",
                        "all_tp_ranks_validated",
                        "timing_gate_status",
                    )
                    if key in profiler
                },
            }
            timeline = timeline_reports.get(profile_id) or {}
            profile_summaries.append(
                {
                    "profile_id": profile_id,
                    "implementation_id": profile.get("implementation_id"),
                    "execution_fingerprint": profile.get("execution_variant"),
                    "contract": contract,
                    "contract_sha256": object_sha256(contract),
                    "timeline_sha256": timeline.get("source_sha256"),
                    "mapped_kernel_count_ratio": timeline.get(
                        "mapped_kernel_count_ratio"
                    ),
                    "mapped_residency_ratio": timeline.get(
                        "mapped_residency_ratio"
                    ),
                    "attribution_passed": timeline.get("attribution_passed"),
                }
            )

        source_revisions = [
            {
                "implementation_id": implementation_id,
                "framework_id": implementation.get("framework_id"),
                "source_repo": implementation.get("source_repo"),
                "source_commit": implementation.get("source_commit"),
                "binding_status": implementation.get("binding_status"),
                "execution_fingerprint": implementation.get("execution_variant"),
            }
            for implementation_id, implementation in sorted(implementations.items())
        ]
        evidence_hashes = sorted(
            item["timeline_sha256"]
            for item in profile_summaries
            if item.get("timeline_sha256")
        )
        model_summaries.append(
            {
                "model": model,
                "status": report.get("status"),
                "validation_evidence": report.get("validation_evidence"),
                "model_ir_version": (bundle.get("meta") or {}).get(
                    "model_ir_version"
                ),
                "model_semantic_revision": (bundle.get("meta") or {}).get(
                    "model_semantic_revision"
                ),
                "catalog_manifest_sha256": tree_manifest_sha256(
                    catalog_root / model
                ),
                "published_bundle_sha256": (report.get("bundle") or {}).get(
                    "published_sha256"
                ),
                "evidence_set_sha256": object_sha256(evidence_hashes),
                "execution_variants": [
                    {
                        "execution_fingerprint": fingerprint,
                        "execution_path_id": variant.get("execution_path_id"),
                        "execution_plan_version": variant.get(
                            "execution_plan_version"
                        ),
                    }
                    for fingerprint, variant in sorted(
                        (bundle.get("execution_variants") or {}).items()
                    )
                ],
                "source_revisions": source_revisions,
                "profiles": profile_summaries,
            }
        )

    browser_checks = [
        {"name": check.get("name"), "passed": bool(check.get("passed"))}
        for check in ((browser_report or {}).get("checks") or [])
    ]
    return {
        "schema_version": "release-acceptance.v1",
        "acceptance_level": acceptance_level,
        "static_gate": static_gate,
        "browser_gate": (
            browser_report.get("status") if browser_report else "not_evaluated"
        ),
        "release_ready": release_ready,
        "release_identity": {
            "compiler_sha256": sha256_file(
                repo_root / "src" / "llm_arch_v2" / "compiler.py"
            ),
            "viewer_sha256": sha256_file(repo_root / "docs" / "viewer.html"),
            "model_set_sha256": object_sha256(
                [
                    {
                        "model": item["model"],
                        "catalog_manifest_sha256": item[
                            "catalog_manifest_sha256"
                        ],
                        "published_bundle_sha256": item[
                            "published_bundle_sha256"
                        ],
                        "evidence_set_sha256": item["evidence_set_sha256"],
                    }
                    for item in model_summaries
                ]
            ),
        },
        "browser_checks": browser_checks,
        "models": model_summaries,
    }


def summarize_attribution_failures(
    failures: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse repeated interval failures into actionable kernel signatures."""

    groups: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(
        lambda: {"event_count": 0, "residency_us": 0.0, "steps": set()}
    )
    for failure in failures:
        key = (
            str(failure.get("reason") or "unknown"),
            str(failure.get("support_class") or ""),
            str(failure.get("kernel") or ""),
        )
        group = groups[key]
        group["event_count"] += 1
        group["residency_us"] += float(failure.get("duration_us") or 0.0)
        if failure.get("step") is not None:
            group["steps"].add(failure["step"])

    return [
        {
            "reason": reason,
            "support_class": support_class or None,
            "kernel": kernel,
            "event_count": value["event_count"],
            "residency_us": round(value["residency_us"], 6),
            "steps": sorted(value["steps"]),
        }
        for (reason, support_class, kernel), value in sorted(groups.items())
    ]


def discover_models(catalog_root: Path) -> list[str]:
    return sorted(
        path.name
        for path in catalog_root.iterdir()
        if path.is_dir() and (path / "model_ir.yaml").is_file()
    )


def audit_published_bundle(
    *,
    model_name: str,
    compiled: dict[str, Any],
    published_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    if not published_path.is_file():
        failures.append(
            {"kind": "missing_published_bundle", "path": str(published_path)}
        )
        return {"exists": False}, failures

    try:
        published = json.loads(published_path.read_text())
    except Exception as error:
        failures.append(
            {
                "kind": "invalid_published_bundle",
                "path": str(published_path),
                "error": str(error),
            }
        )
        return {"exists": True, "valid_json": False}, failures

    compiled_sha256 = object_sha256(compiled)
    published_sha256 = object_sha256(published)
    if compiled != published:
        failures.append(
            {
                "kind": "stale_published_bundle",
                "path": str(published_path),
                "compiled_sha256": compiled_sha256,
                "published_sha256": published_sha256,
            }
        )
    return {
        "exists": True,
        "valid_json": True,
        "matches_compiler": compiled == published,
        "compiled_sha256": compiled_sha256,
        "published_sha256": published_sha256,
        "execution_variants": len(compiled.get("execution_variants") or {}),
        "implementations": len(compiled.get("implementations") or {}),
        "profiles": len(compiled.get("profiles") or {}),
        "views": len((compiled.get("model_ir") or {}).get("views") or {}),
        "model": model_name,
    }, failures


def audit_model(
    *,
    repo_root: Path,
    catalog_root: Path,
    docs_root: Path,
    model_name: str,
) -> dict[str, Any]:
    model_root = catalog_root / model_name
    failures: list[dict[str, Any]] = []
    validation_evidence = validate_validation_evidence(model_root)
    if validation_evidence.get("status") != "pass":
        failures.extend(
            {
                "kind": "validation_evidence_failure",
                "error": error,
            }
            for error in validation_evidence.get("errors") or []
        )
    try:
        compiled = compile_catalog(model_root)
    except Exception as error:
        return {
            "model": model_name,
            "status": "fail",
            "validation_evidence": validation_evidence,
            "failures": failures
            + [
                {
                    "kind": "catalog_compile_failure",
                    "exception_type": type(error).__name__,
                    "error": str(error),
                }
            ],
        }

    bundle_path = docs_root / f"{model_name}_v2" / "arch_data.json"
    bundle_report, bundle_failures = audit_published_bundle(
        model_name=model_name,
        compiled=compiled,
        published_path=bundle_path,
    )
    failures.extend(bundle_failures)

    expected_link = f"model={model_name}_v2"
    inventory_checks: dict[str, bool] = {}
    for source in (repo_root / "README.md", docs_root / "index.html"):
        present = source.is_file() and expected_link in source.read_text()
        inventory_checks[str(source.relative_to(repo_root))] = present
        if not present:
            failures.append(
                {
                    "kind": "missing_public_model_inventory_entry",
                    "path": str(source),
                    "expected": expected_link,
                }
            )

    timeline_reports: list[dict[str, Any]] = []
    for profile_path in sorted(model_root.glob("profiles/*/*/*.yaml")):
        profile = yaml.safe_load(profile_path.read_text()) or {}
        timeline = profile.get("timeline")
        if not timeline:
            failures.append(
                {
                    "kind": "accepted_profile_without_timeline",
                    "profile": profile.get("profile_id"),
                    "path": str(profile_path),
                }
            )
            continue
        profile_id = str(profile.get("profile_id") or "")
        source = profile_path.parent / str(timeline.get("artifact") or "")
        published = (
            docs_root
            / f"{model_name}_v2"
            / "timelines"
            / f"{profile_id}.timeline.json.gz"
        )
        declared_sha256 = str(timeline.get("sha256") or "")
        if not source.is_file():
            failures.append(
                {
                    "kind": "missing_source_timeline",
                    "profile": profile_id,
                    "path": str(source),
                }
            )
            continue
        source_sha256 = sha256_file(source)
        published_sha256 = sha256_file(published) if published.is_file() else ""
        if source_sha256 != declared_sha256:
            failures.append(
                {
                    "kind": "source_timeline_hash_mismatch",
                    "profile": profile_id,
                    "declared_sha256": declared_sha256,
                    "actual_sha256": source_sha256,
                }
            )
        if not published.is_file() or published_sha256 != source_sha256:
            failures.append(
                {
                    "kind": "published_timeline_mismatch",
                    "profile": profile_id,
                    "source_sha256": source_sha256,
                    "published_sha256": published_sha256 or None,
                    "path": str(published),
                }
            )

        attribution = audit_timeline(source)
        attribution_failure_groups = summarize_attribution_failures(
            attribution["failures"]
        )
        timeline_reports.append(
            {
                "profile": profile_id,
                "source_sha256": source_sha256,
                "published_sha256": published_sha256 or None,
                "kernel_count": attribution["total_kernel_count"],
                "mapped_kernel_count_ratio": attribution[
                    "mapped_kernel_count_ratio"
                ],
                "mapped_residency_ratio": attribution[
                    "mapped_residency_ratio"
                ],
                "support_counts": attribution["support_counts"],
                "attribution_passed": attribution["passed"],
                "attribution_failure_count": attribution["failure_count"],
                "attribution_failure_groups": attribution_failure_groups,
            }
        )
        for failure in attribution_failure_groups:
            failures.append(
                {
                    "kind": "timeline_attribution_failure",
                    "profile": profile_id,
                    **failure,
                }
            )

    return {
        "model": model_name,
        "status": "pass" if not failures else "fail",
        "validation_evidence": validation_evidence,
        "bundle": bundle_report,
        "public_inventory": inventory_checks,
        "timeline_count": len(timeline_reports),
        "timelines": timeline_reports,
        "failures": failures,
    }


def run_browser_gates(
    *,
    repo_root: Path,
    docs_root: Path,
    models: list[str],
    base_url: str,
    output_dir: Path,
    browser: str | None,
) -> dict[str, Any]:
    commands: list[tuple[str, list[str], Path]] = []
    viewer_root = output_dir / "viewer"
    for model in models:
        command = [
            sys.executable,
            str(repo_root / "scripts" / "audit_viewer_render.py"),
            str(docs_root / f"{model}_v2" / "arch_data.json"),
            "--base-url",
            base_url,
            "--output",
            str(viewer_root / model),
        ]
        if browser:
            command.extend(["--browser", browser])
        commands.append(
            (f"viewer:{model}", command, viewer_root / model / "report.json")
        )

    stream_report = output_dir / "timeline-streams.json"
    stream_command = [
        sys.executable,
        str(repo_root / "scripts" / "audit_timeline_stream_modes.py"),
        *[
            str(docs_root / f"{model}_v2" / "arch_data.json")
            for model in models
        ],
        "--base-url",
        base_url,
        "--output",
        str(stream_report),
    ]
    if browser:
        stream_command.extend(["--browser", browser])
    commands.append(("timeline-streams", stream_command, stream_report))

    comparison_root = output_dir / "framework-comparison"
    comparison_command = [
        sys.executable,
        str(repo_root / "scripts" / "audit_framework_comparison.py"),
        "--base-url",
        base_url,
        "--output",
        str(comparison_root),
    ]
    if browser:
        comparison_command.extend(["--browser", browser])
    commands.append(
        (
            "framework-comparison",
            comparison_command,
            comparison_root / "report.json",
        )
    )

    checks: list[dict[str, Any]] = []
    for name, command, report_path in commands:
        completed = subprocess.run(
            command,
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        report: dict[str, Any] = {}
        if report_path.is_file():
            try:
                report = json.loads(report_path.read_text())
            except Exception as error:
                report = {"status": "fail", "error": str(error)}
        passed = completed.returncode == 0 and report.get("status") == "pass"
        checks.append(
            {
                "name": name,
                "passed": passed,
                "returncode": completed.returncode,
                "report": str(report_path),
                "stderr_tail": completed.stderr[-2000:] if completed.stderr else "",
            }
        )
    return {
        "status": "pass" if all(check["passed"] for check in checks) else "fail",
        "base_url": base_url,
        "checks": checks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--model", action="append")
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--catalog-root", type=Path, default=REPO_ROOT / "catalog")
    parser.add_argument("--docs-root", type=Path, default=REPO_ROOT / "docs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--level",
        choices=("static", "release"),
        default="static",
        help="release adds all real-browser gates and is the only release-ready level",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8765",
        help="running viewer server used by --level release",
    )
    parser.add_argument("--browser", help="optional Chromium/Chrome executable")
    parser.add_argument(
        "--publish-summary",
        type=Path,
        help="also write the deterministic release-acceptance.v1 ledger here",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    catalog_root = args.catalog_root.resolve()
    docs_root = args.docs_root.resolve()
    models = (
        discover_models(catalog_root)
        if args.all
        else list(dict.fromkeys(args.model or []))
    )
    unknown = [
        model
        for model in models
        if not (catalog_root / model / "model_ir.yaml").is_file()
    ]
    if unknown:
        raise SystemExit(f"unknown catalog models: {unknown}")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    model_reports = [
        audit_model(
            repo_root=REPO_ROOT,
            catalog_root=catalog_root,
            docs_root=docs_root,
            model_name=model,
        )
        for model in models
    ]
    static_passed = all(report["status"] == "pass" for report in model_reports)
    browser_report: dict[str, Any] | None = None
    if args.level == "release" and static_passed:
        browser_report = run_browser_gates(
            repo_root=REPO_ROOT,
            docs_root=docs_root,
            models=models,
            base_url=args.base_url,
            output_dir=output / "browser",
            browser=args.browser,
        )
    elif args.level == "release":
        browser_report = {
            "status": "blocked",
            "reason": "static_gate_failed",
            "checks": [],
        }

    release_ready = bool(
        args.level == "release"
        and static_passed
        and browser_report
        and browser_report.get("status") == "pass"
    )
    report = {
        "schema_version": "release-audit.v1",
        "acceptance_level": args.level,
        "static_gate": "pass" if static_passed else "fail",
        "browser_gate": (
            browser_report.get("status") if browser_report else "not_evaluated"
        ),
        "release_ready": release_ready,
        "model_count": len(models),
        "models": model_reports,
        "browser": browser_report,
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    acceptance_summary = build_acceptance_summary(
        repo_root=REPO_ROOT,
        catalog_root=catalog_root,
        docs_root=docs_root,
        models=models,
        model_reports=model_reports,
        acceptance_level=args.level,
        static_gate=report["static_gate"],
        browser_report=browser_report,
        release_ready=release_ready,
    )
    acceptance_path = output / "acceptance-summary.json"
    acceptance_path.write_text(
        json.dumps(acceptance_summary, indent=2, sort_keys=True) + "\n"
    )
    if args.publish_summary:
        published_acceptance = args.publish_summary.resolve()
        published_acceptance.parent.mkdir(parents=True, exist_ok=True)
        published_acceptance.write_text(
            json.dumps(acceptance_summary, indent=2, sort_keys=True) + "\n"
        )
    print(
        json.dumps(
            {
                "report": str(report_path),
                "acceptance_summary": str(acceptance_path),
                "acceptance_level": args.level,
                "static_gate": report["static_gate"],
                "browser_gate": report["browser_gate"],
                "release_ready": release_ready,
                "models": {
                    item["model"]: {
                        "status": item["status"],
                        "profiles": (item.get("bundle") or {}).get("profiles", 0),
                        "timelines": item.get("timeline_count", 0),
                        "failure_count": len(item.get("failures") or []),
                    }
                    for item in model_reports
                },
            },
            indent=2,
        )
    )
    gate_passed = release_ready if args.level == "release" else static_passed
    return 0 if gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
