from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.common.runtime_skeleton import write_yaml  # noqa: E402
from models.common.source_reconciler import reconcile_architecture  # noqa: E402


class CommonSourceReconcilerTest(unittest.TestCase):
    def test_reconciler_requires_runtime_and_source_evidence(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            source_root = tmp / "src"
            model_file = source_root / "toy/model.py"
            model_file.parent.mkdir(parents=True)
            model_file.write_text(
                "class ToyLayer:\n"
                "    def forward(self, x):\n"
                "        y = self.proj(x)\n"
                "        return y\n"
            )
            model_config = tmp / "config.yaml"
            write_yaml(model_config, {"hidden_size": 16})
            skeleton_path = tmp / "runtime_skeleton.yaml"
            write_yaml(
                skeleton_path,
                {
                    "source": {"phase": "toy_extend", "rank": 0},
                    "runtime_nodes": {
                        "toy_proj": {
                            "total_kernel_ms": 1.25,
                            "kernel_count": 2,
                            "top_kernels_by_duration": [
                                {"value": "toy_kernel", "dur_ms": 1.25}
                            ],
                        }
                    },
                },
            )
            hand_ir = tmp / "hand.yaml"
            write_yaml(
                hand_ir,
                {
                    "views": {
                        "toy": {
                            "nodes": [
                                {"id": "toy_in"},
                                {"id": "toy_proj"},
                                {"id": "toy_out"},
                            ]
                        }
                    }
                },
            )

            draft, report, diff = reconcile_architecture(
                skeleton_path=skeleton_path,
                config={
                    "model_id": "toy",
                    "source_root": str(source_root),
                    "model_config": str(model_config),
                    "hand_ir": str(hand_ir),
                    "hand_ir_compare_views": ["toy"],
                    "views": {
                        "toy": {
                            "nodes": [
                                {"id": "toy_in", "shape": "io"},
                                {
                                    "id": "toy_proj",
                                    "shape": "gemm",
                                    "runtime_nodes": ["toy_proj"],
                                    "config_refs": ["hidden_size"],
                                    "source_identity": {
                                        "canonical_source_id": "ToyLayer.proj",
                                        "kind": "self_attr_call",
                                        "file": "toy/model.py",
                                        "class": "ToyLayer",
                                        "function": "forward",
                                        "line": 3,
                                        "callee": "self.proj",
                                    },
                                    "source_links": [
                                        {
                                            "file": "toy/model.py",
                                            "line": 3,
                                            "contains": ["self.proj"],
                                        }
                                    ],
                                },
                                {"id": "toy_out", "shape": "io"},
                            ]
                        }
                    },
                },
                config_base=tmp,
            )

            self.assertTrue(report["ok"], report["errors"])
            self.assertEqual(report["summary"]["verified_source_links"], 1)
            self.assertEqual(report["summary"]["verified_source_identities"], 1)
            node = draft["views"]["toy"]["nodes"][1]
            self.assertEqual(
                node["source_identity"]["canonical_source_ids"],
                ["ToyLayer.proj"],
            )
            self.assertEqual(node["runtime_evidence"]["bucket_total_kernel_ms"], 1.25)
            self.assertEqual(node["runtime_evidence"]["fine_node_ms"], 1.25)
            self.assertEqual(diff["manual_only"], [])
            self.assertEqual(diff["generated_only"], [])

    def test_reconciler_fails_on_missing_source_needle(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            source_root = tmp / "src"
            model_file = source_root / "toy/model.py"
            model_file.parent.mkdir(parents=True)
            model_file.write_text("def forward(x):\n    return x\n")
            skeleton_path = tmp / "runtime_skeleton.yaml"
            write_yaml(skeleton_path, {"source": {}, "runtime_nodes": {}})

            _draft, report, _diff = reconcile_architecture(
                skeleton_path=skeleton_path,
                config={
                    "model_id": "toy",
                    "source_root": str(source_root),
                    "views": {
                        "toy": {
                            "nodes": [
                                {
                                    "id": "bad",
                                    "source_links": [
                                        {
                                            "file": "toy/model.py",
                                            "line": 2,
                                            "contains": ["missing_call"],
                                        }
                                    ],
                                }
                            ]
                        }
                    },
                },
                config_base=tmp,
            )

            self.assertFalse(report["ok"])
            self.assertIn("missing_call", "\n".join(report["errors"]))

    def test_reconciler_fails_when_display_alias_does_not_match_source_symbol(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            source_root = tmp / "src"
            model_file = source_root / "toy/model.py"
            model_file.parent.mkdir(parents=True)
            model_file.write_text(
                "class ToyLayer:\n"
                "    def forward(self, x):\n"
                "        return self.proj(x)\n"
            )
            skeleton_path = tmp / "runtime_skeleton.yaml"
            write_yaml(skeleton_path, {"source": {}, "runtime_nodes": {}})

            _draft, report, _diff = reconcile_architecture(
                skeleton_path=skeleton_path,
                config={
                    "model_id": "toy",
                    "source_root": str(source_root),
                    "views": {
                        "toy": {
                            "nodes": [
                                {
                                    "id": "made_up_alias",
                                    "source_identity": {
                                        "canonical_source_id": "ToyLayer.not_proj",
                                        "kind": "self_attr_call",
                                        "file": "toy/model.py",
                                        "class": "ToyLayer",
                                        "function": "forward",
                                        "line": 3,
                                        "callee": "self.proj",
                                    },
                                }
                            ]
                        }
                    },
                },
                config_base=tmp,
            )

            self.assertFalse(report["ok"])
            self.assertIn("canonical_source_id mismatch", "\n".join(report["errors"]))


if __name__ == "__main__":
    unittest.main()
