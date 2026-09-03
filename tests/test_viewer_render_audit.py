from argparse import Namespace
import json
from pathlib import Path
import subprocess
import sys

from scripts.audit_viewer_render import TIMELINE_LOAD_TIMEOUT_MS, write_exception_report


def test_viewer_audit_timeline_wait_is_bounded() -> None:
    assert 0 < TIMELINE_LOAD_TIMEOUT_MS <= 60_000


def test_every_timeline_data_consumer_has_a_bounded_wait() -> None:
    script = (
        Path(__file__).parents[1] / "scripts" / "audit_viewer_render.py"
    ).read_text()
    for stage in (
        "cross_link_setup",
        "bidirectional_dom_click_setup",
        "runtime_support_detail",
        "typed_unresolved_detail",
    ):
        marker = f"stage: '{stage}'"
        assert marker in script
        block_start = script.rfind('"""async timeoutMs => {', 0, script.index(marker))
        block = script[block_start : script.index(marker)]
        assert "while (!TIMELINE_DATA && Date.now() < deadline)" in block
        assert "setViewMode('split', false, true)" in block


def test_geometry_audit_uses_rendered_screen_coordinates() -> None:
    script = (
        Path(__file__).parents[1] / "scripts" / "audit_viewer_render.py"
    ).read_text()
    geometry = script[script.index("const bg = node.querySelector('.node-bg')") :]
    geometry = geometry[: geometry.index("return {issues, groups:")]
    assert "bg.getBoundingClientRect()" in geometry
    assert "text.getBoundingClientRect()" in geometry
    assert ".getBBox()" not in geometry


def test_zero_profile_audit_is_architecture_only_not_vacuous() -> None:
    script = (
        Path(__file__).parents[1] / "scripts" / "audit_viewer_render.py"
    ).read_text()
    assert 'if not bundle["profiles"]:' in script
    assert '"architecture_only_implementations"' in script
    assert '"architecture_only_no_accepted_profiles"' in script
    assert 'page.wait_for_selector("g.view-group g.node"' in script


def test_fused_owner_audit_is_navigation_aware_and_fail_closed() -> None:
    """Every rendered ``fused into`` row must link to its real owner.

    The owner may be inserted by Execution IR rather than authored in Model IR,
    so both the detail-panel and node-card checks must use the navigation-aware
    lookup and record a concrete issue when the link is missing.
    """

    script = (
        Path(__file__).parents[1] / "scripts" / "audit_viewer_render.py"
    ).read_text()
    assert script.count("irTargetExistsForNavigation(architectureOwner)") == 2
    assert "fusion owner detail link" in script
    assert "fusion_owner_card_link" in script
    assert "fusion_owner_dom_clicks" in script


def test_viewer_reports_independent_active_share_of_timed_parent() -> None:
    """Node percentages must use independent active timing, never copied fusion time."""

    viewer = (Path(__file__).parents[1] / "docs" / "viewer.html").read_text()
    assert 'function independentActiveMs(cell)' in viewer
    assert '["fused", "fused_by_occurrence", "structural", "out_of_scope"]' in viewer
    assert 'function parentActiveShare(cell, parentCell)' in viewer
    assert 'return share > 0 && share < 0.1 ? "<0.1%"' in viewer
    assert 'active share ${activeSharePercentText(share)} of parent' in viewer
    assert 'comparisonParentActiveShare(context, target)' in viewer
    assert 'of the nearest timed parent module · non-additive across overlapping children' in viewer
    audit = (
        Path(__file__).parents[1] / "scripts" / "audit_viewer_render.py"
    ).read_text()
    assert '"parent_active_share_cards": 0' in audit
    assert "type: 'parent_active_share_mismatch'" in audit
    assert "type: 'unexpected_parent_active_share'" in audit


def test_viewer_audit_writes_structured_failure_on_exception(tmp_path) -> None:
    output = tmp_path / "audit"
    args = Namespace(bundle=tmp_path / "qwen35_v2" / "arch_data.json", output=output)

    assert write_exception_report(args, RuntimeError("synthetic browser failure")) == 1

    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "fail"
    assert report["failures"] == [
        {
            "kind": "audit_exception",
            "exception_type": "RuntimeError",
            "error": "synthetic browser failure",
        }
    ]


def test_viewer_audit_entrypoint_is_fail_closed(tmp_path) -> None:
    output = tmp_path / "audit"
    script = Path(__file__).parents[1] / "scripts" / "audit_viewer_render.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            str(tmp_path / "missing" / "arch_data.json"),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "fail"
    assert report["failures"][0]["kind"] == "audit_exception"
    assert report["failures"][0]["exception_type"] == "FileNotFoundError"
