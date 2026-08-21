from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.common.source_config_views import (  # noqa: E402
    build_source_config_architecture,
    write_yaml,
)


class CommonSourceConfigViewsTest(unittest.TestCase):
    def _make_fixture(self, tmp: Path) -> tuple[Path, Path, Path, Path]:
        source_root = tmp / "src"
        model_file = source_root / "toy/model.py"
        model_file.parent.mkdir(parents=True)
        model_file.write_text(
            "class ToyTop:\n"
            "    def __init__(self):\n"
            "        self.language_model = ToyLM()\n"
            "\n"
            "class ToyLM:\n"
            "    pass\n"
        )
        model_config = tmp / "config.yaml"
        write_yaml(model_config, {"hidden_size": 16})
        trace_arch = tmp / "arch_draft.yaml"
        write_yaml(
            trace_arch,
            {
                "schema_version": "arch_draft.v0",
                "model_id": "toy",
                "source": {"source_repo": "example/toy", "source_commit": "abc123"},
                "views": {
                    "stack": {
                        "nodes": [
                            {
                                "id": "stack_in",
                                "shape": "io",
                                "provenance": {"architecture": "runtime-observed"},
                            },
                            {
                                "id": "stack_out",
                                "shape": "io",
                                "provenance": {"architecture": "runtime-observed"},
                            },
                        ],
                        "edges": [{"from": "stack_in", "to": "stack_out"}],
                    }
                },
            },
        )
        out_dir = tmp / "out"
        return source_root, model_config, trace_arch, out_dir

    def test_source_config_views_merge_with_trace_views(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            source_root, model_config, trace_arch, out_dir = self._make_fixture(tmp)
            source_config = tmp / "source_config.yaml"
            write_yaml(
                source_config,
                {
                    "model_id": "toy",
                    "source_root": str(source_root),
                    "trace_arch": str(trace_arch),
                    "model_config": str(model_config),
                    "output_dir": str(out_dir),
                    "views": {
                        "top": {
                            "nodes": [
                                {"id": "tokens", "shape": "io"},
                                {
                                    "id": "language_model",
                                    "shape": "block",
                                    "drill": "stack",
                                    "config_refs": ["hidden_size"],
                                    "source_identity": {
                                        "canonical_source_id": "ToyTop.language_model",
                                        "kind": "self_attr_def",
                                        "file": "toy/model.py",
                                        "class": "ToyTop",
                                        "function": "__init__",
                                        "line": 3,
                                        "target": "self.language_model",
                                    },
                                    "source_links": [
                                        {
                                            "file": "toy/model.py",
                                            "line": 3,
                                            "contains": ["self.language_model"],
                                        }
                                    ],
                                },
                            ],
                            "edges": [{"from": "tokens", "to": "language_model"}],
                        }
                    },
                },
            )

            arch, report, artifact_index = build_source_config_architecture(
                source_config_path=source_config
            )

            self.assertTrue(report["ok"], report["errors"])
            self.assertEqual(set(arch["views"]), {"top", "stack"})
            self.assertEqual(report["summary"]["verified_source_links"], 1)
            self.assertEqual(report["summary"]["verified_source_identities"], 1)
            node = arch["views"]["top"]["nodes"][1]
            self.assertEqual(node["provenance"]["architecture"], "source-config-generated")
            self.assertEqual(
                node["source_identity"]["canonical_source_ids"],
                ["ToyTop.language_model"],
            )
            self.assertIn("artifact_index.v0", artifact_index["schema_version"])

    def test_duplicate_view_names_are_contract_errors(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            source_root, model_config, trace_arch, out_dir = self._make_fixture(tmp)
            source_config = tmp / "source_config.yaml"
            write_yaml(
                source_config,
                {
                    "model_id": "toy",
                    "source_root": str(source_root),
                    "trace_arch": str(trace_arch),
                    "model_config": str(model_config),
                    "output_dir": str(out_dir),
                    "views": {
                        "stack": {
                            "nodes": [{"id": "source_stack", "shape": "block"}],
                            "edges": [],
                        }
                    },
                },
            )

            _arch, report, _artifact_index = build_source_config_architecture(
                source_config_path=source_config
            )

            self.assertFalse(report["ok"])
            self.assertIn("conflict with trace-derived views", "\n".join(report["errors"]))

    def test_missing_drill_target_is_contract_error(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            source_root, model_config, trace_arch, out_dir = self._make_fixture(tmp)
            source_config = tmp / "source_config.yaml"
            write_yaml(
                source_config,
                {
                    "model_id": "toy",
                    "source_root": str(source_root),
                    "trace_arch": str(trace_arch),
                    "model_config": str(model_config),
                    "output_dir": str(out_dir),
                    "views": {
                        "top": {
                            "nodes": [
                                {
                                    "id": "language_model",
                                    "shape": "block",
                                    "drill": "missing_stack",
                                }
                            ],
                            "edges": [],
                        }
                    },
                },
            )

            _arch, report, _artifact_index = build_source_config_architecture(
                source_config_path=source_config
            )

            self.assertFalse(report["ok"])
            self.assertIn("missing view", "\n".join(report["errors"]))


if __name__ == "__main__":
    unittest.main()
