from __future__ import annotations

import json
from pathlib import Path

from scripts.release_audit import (
    audit_published_bundle,
    build_acceptance_summary,
    discover_models,
    summarize_attribution_failures,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_release_audit_discovers_exact_public_catalogs() -> None:
    models = discover_models(REPO_ROOT / "catalog")
    public_models = sorted(
        path.name.removesuffix("_v2")
        for path in (REPO_ROOT / "docs").iterdir()
        if path.is_dir() and (path / "arch_data.json").is_file()
    )
    assert models == public_models


def test_release_audit_rejects_stale_published_bundle(tmp_path: Path) -> None:
    published = tmp_path / "arch_data.json"
    published.write_text(json.dumps({"schema_version": "stale"}) + "\n")

    report, failures = audit_published_bundle(
        model_name="toy",
        compiled={"schema_version": "2.0", "profiles": {}},
        published_path=published,
    )

    assert report["matches_compiler"] is False
    assert failures[0]["kind"] == "stale_published_bundle"


def test_release_audit_distinguishes_missing_bundle(tmp_path: Path) -> None:
    report, failures = audit_published_bundle(
        model_name="toy",
        compiled={"schema_version": "2.0"},
        published_path=tmp_path / "missing.json",
    )

    assert report == {"exists": False}
    assert failures == [
        {
            "kind": "missing_published_bundle",
            "path": str(tmp_path / "missing.json"),
        }
    ]


def test_release_audit_groups_repeated_attribution_failures() -> None:
    failures = summarize_attribution_failures(
        [
            {
                "reason": "unclassified_unbound_kernel",
                "kernel": "same_kernel",
                "duration_us": 1.25,
                "step": 1,
            },
            {
                "reason": "unclassified_unbound_kernel",
                "kernel": "same_kernel",
                "duration_us": 2.5,
                "step": 2,
            },
        ]
    )

    assert failures == [
        {
            "reason": "unclassified_unbound_kernel",
            "support_class": None,
            "kernel": "same_kernel",
            "event_count": 2,
            "residency_us": 3.75,
            "steps": [1, 2],
        }
    ]


def test_release_acceptance_summary_is_deterministic_and_content_addressed(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    catalog_root = repo_root / "catalog"
    docs_root = repo_root / "docs"
    (repo_root / "src" / "llm_arch_v2").mkdir(parents=True)
    (repo_root / "src" / "llm_arch_v2" / "compiler.py").write_text("compiler\n")
    docs_root.mkdir()
    (docs_root / "viewer.html").write_text("viewer\n")
    (catalog_root / "toy").mkdir(parents=True)
    (catalog_root / "toy" / "model_ir.yaml").write_text("model_id: toy\n")
    (docs_root / "toy_v2").mkdir()
    bundle = {
        "meta": {"model_ir_version": 1, "model_semantic_revision": 6},
        "execution_variants": {
            "exec_123": {
                "execution_path_id": "tp8",
                "execution_plan_version": 1,
            }
        },
        "implementations": {
            "impl": {
                "framework_id": "sglang",
                "source_repo": "https://example.invalid/runtime",
                "source_commit": "abc123",
                "binding_status": "validated",
                "execution_variant": "exec_123",
            }
        },
        "profiles": {
            "profile": {
                "implementation_id": "impl",
                "execution_variant": "exec_123",
                "meta": {
                    "phase": "decode",
                    "generation_mode": "autoregressive",
                    "execution_parameters": {"tp_size": 8},
                    "hardware": {"gpu": "GB300"},
                    "workload": {"batch_size": 1, "isl": 8192, "osl": 1024},
                    "profiler": {
                        "type": "torch_profiler",
                        "cuda_graph_enabled": True,
                    },
                },
            }
        },
    }
    (docs_root / "toy_v2" / "arch_data.json").write_text(json.dumps(bundle))
    reports = [
        {
            "model": "toy",
            "status": "pass",
            "validation_evidence": {
                "schema_version": "validation-evidence-report.v1",
                "status": "pass",
                "anti_self_validation": "pass",
                "gates": {
                    gate: {"status": "pass"}
                    for gate in (
                        "semantic_ir",
                        "execution_contract",
                        "binding_reconciliation",
                        "production_evidence",
                    )
                },
            },
            "bundle": {"published_sha256": "bundle-sha"},
            "timelines": [
                {
                    "profile": "profile",
                    "source_sha256": "trace-sha",
                    "mapped_kernel_count_ratio": 1.0,
                    "mapped_residency_ratio": 1.0,
                    "attribution_passed": True,
                }
            ],
        }
    ]

    kwargs = {
        "repo_root": repo_root,
        "catalog_root": catalog_root,
        "docs_root": docs_root,
        "models": ["toy"],
        "model_reports": reports,
        "acceptance_level": "release",
        "static_gate": "pass",
        "browser_report": {
            "status": "pass",
            "checks": [{"name": "viewer:toy", "passed": True}],
        },
        "release_ready": True,
    }
    first = build_acceptance_summary(**kwargs)
    second = build_acceptance_summary(**kwargs)

    assert first == second
    assert first["schema_version"] == "release-acceptance.v1"
    assert first["release_ready"] is True
    assert first["release_identity"]["compiler_sha256"]
    assert first["models"][0]["catalog_manifest_sha256"]
    assert first["models"][0]["validation_evidence"]["anti_self_validation"] == "pass"
    profile = first["models"][0]["profiles"][0]
    assert profile["execution_fingerprint"] == "exec_123"
    assert profile["timeline_sha256"] == "trace-sha"
    assert profile["contract_sha256"]
