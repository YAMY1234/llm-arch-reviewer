from __future__ import annotations

import json
from pathlib import Path

from scripts.release_audit import (
    audit_published_bundle,
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
