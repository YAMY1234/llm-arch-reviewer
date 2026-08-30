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
