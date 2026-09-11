from pathlib import Path

import pytest

from scripts.audit_model_only_viewer import drill_inventory
from scripts.release_audit import discover_profiled_models


def test_inventory_visits_every_shared_drill_without_duplicate_subtrees():
    views = {"top": {"nodes": [{"id": "a", "drill": "detail"},
                                 {"id": "b", "drill": "detail"}]},
             "detail": {"nodes": [{"id": "primitive"}]}}
    drills = drill_inventory(views, "top")
    assert [(d["node"], d["child"]) for d in drills] == [("a", "detail"), ("b", "detail")]


@pytest.mark.parametrize("views,match", [
    ({"top": {"nodes": [{"id": "a", "drill": "absent"}]}}, "missing drill"),
    ({"top": {"nodes": [{"id": "a", "drill": "top"}]}}, "cyclic drill"),
    ({"top": {"nodes": []}, "hidden": {"nodes": []}}, "unreachable"),
])
def test_inventory_fails_closed(views, match):
    with pytest.raises(ValueError, match=match):
        drill_inventory(views, "top")


def test_release_scope_does_not_exclude_unknown_or_missing_lifecycle(tmp_path: Path):
    for name, lifecycle in [("semantic", "model_only"), ("runtime", "profiled"),
                            ("invalid", "skip_everything"), ("legacy", None)]:
        root = tmp_path / name
        root.mkdir()
        (root / "model_ir.yaml").write_text("model_id: fixture\n")
        if lifecycle:
            (root / "pipeline.yaml").write_text(f"lifecycle: {lifecycle}\n")
    assert discover_profiled_models(tmp_path) == ["invalid", "legacy", "runtime"]
