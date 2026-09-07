#!/usr/bin/env python3
"""Plan, accept, or resume one deterministic add-trace run."""

from __future__ import annotations

import argparse
import json
import subprocess
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
from llm_arch_v2.add_trace_dag import run_add_trace_dag  # noqa: E402
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

    run = subparsers.add_parser(
        "run",
        help=(
            "run or resume the content-addressed stage DAG; missing authored "
            "evidence is reported as needs_input"
        ),
    )
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--workspace", type=Path, required=True)
    run.add_argument(
        "--evidence-dir",
        type=Path,
        help=(
            "directory containing binding-revision.json, eager-reconciliation.json, "
            "and trace-attribution.json"
        ),
    )
    run.add_argument("--binding-revision", type=Path)
    run.add_argument("--eager-reconciliation", type=Path)
    run.add_argument("--trace-attribution", type=Path)
    run.add_argument(
        "--profile",
        type=Path,
        help="materialized catalog profile; auto-resolved when exactly one matches",
    )
    run.add_argument(
        "--release-report",
        type=Path,
        help="existing release-audit.v1 report; otherwise the CLI runs the gate",
    )
    run.add_argument(
        "--release-level", choices=("static", "release"), default="static"
    )
    run.add_argument(
        "--base-url",
        default="http://127.0.0.1:8765",
        help="running Viewer URL used by the real-browser release gate",
    )
    run.add_argument("--browser", help="optional Chromium/Chrome executable")
    run.add_argument(
        "--no-auto-release-audit",
        action="store_true",
        help="leave release_audit in needs_input instead of running it",
    )
    run.add_argument("--catalog-root", type=Path, default=REPO_ROOT / "catalog")
    run.add_argument("--docs-root", type=Path, default=REPO_ROOT / "docs")
    return parser.parse_args()


def _resolved_input(
    explicit: Path | None, evidence_dir: Path | None, name: str
) -> Path | None:
    if explicit is not None:
        return explicit
    if evidence_dir is None:
        return None
    candidate = evidence_dir / name
    return candidate if candidate.is_file() else None


def _run_dag(args: argparse.Namespace) -> tuple[dict, int]:
    evidence_dir = args.evidence_dir.resolve() if args.evidence_dir else None
    inputs = {
        "binding_revision_path": _resolved_input(
            args.binding_revision, evidence_dir, "binding-revision.json"
        ),
        "eager_reconciliation_path": _resolved_input(
            args.eager_reconciliation, evidence_dir, "eager-reconciliation.json"
        ),
        "trace_attribution_path": _resolved_input(
            args.trace_attribution, evidence_dir, "trace-attribution.json"
        ),
    }
    result = run_add_trace_dag(
        manifest_path=args.manifest,
        workspace=args.workspace,
        catalog_root=args.catalog_root,
        repo_root=REPO_ROOT,
        profile_path=args.profile,
        release_report_path=args.release_report,
        release_level=args.release_level,
        **inputs,
    )
    state = result["state"]
    next_stages = [item["stage"] for item in state["next_actions"]]
    if (
        next_stages == ["release_audit"]
        and args.release_report is None
        and not args.no_auto_release_audit
    ):
        release_output = args.workspace.resolve() / f"release-audit-{args.release_level}"
        build_command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "build_v2.py"),
            "--model",
            state["model_id"],
            "--catalog-root",
            str(args.catalog_root.resolve()),
            "--docs-root",
            str(args.docs_root.resolve()),
        ]
        built = subprocess.run(
            build_command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if built.returncode:
            details = (built.stdout + "\n" + built.stderr).strip()[-4000:]
            raise AddTraceError(
                "automatic public bundle materialization failed:\n" + details
            )
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "release_audit.py"),
            "--model",
            state["model_id"],
            "--catalog-root",
            str(args.catalog_root.resolve()),
            "--docs-root",
            str(args.docs_root.resolve()),
            "--level",
            args.release_level,
            "--output",
            str(release_output),
        ]
        if args.release_level == "release":
            command.extend(["--base-url", args.base_url])
            if args.browser:
                command.extend(["--browser", args.browser])
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode:
            details = (completed.stdout + "\n" + completed.stderr).strip()[-4000:]
            raise AddTraceError(f"automatic release audit failed:\n{details}")
        result = run_add_trace_dag(
            manifest_path=args.manifest,
            workspace=args.workspace,
            catalog_root=args.catalog_root,
            repo_root=REPO_ROOT,
            profile_path=args.profile,
            release_report_path=release_output / "report.json",
            release_level=args.release_level,
            **inputs,
        )
        state = result["state"]
    summary = {
        "status": state["status"],
        "run_id": state["run_id"],
        "model_id": state["model_id"],
        "release_level": state["release_level"],
        "state_sha256": state["state_sha256"],
        "workspace": str(args.workspace.resolve()),
        "state": str(args.workspace.resolve() / "run-state.json"),
        "review_packet": str(args.workspace.resolve() / "review-packet.json"),
        "review_markdown": str(args.workspace.resolve() / "REVIEW.md"),
        "cache_hits": result["cache_hits"],
        "next_actions": state["next_actions"],
    }
    return summary, 0 if state["status"] == "pass" else 3


def main() -> int:
    args = parse_args()
    try:
        if args.command == "run":
            result, returncode = _run_dag(args)
            _write(result, None)
            return returncode
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
