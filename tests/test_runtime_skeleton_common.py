from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.common.runtime_skeleton import build_runtime_skeleton  # noqa: E402


def _event(event_id: str, module_chain: list[str], dur_us: float = 1.0):
    return {
        "event_id": event_id,
        "kernel_name": f"{event_id}_kernel",
        "ts_us": float(int(event_id.split("_")[-1]) * 10),
        "dur_us": dur_us,
        "python_stack": [
            {"raw": f"nn.Module: {module}", "module": module}
            for module in module_chain
        ]
        + [
            {
                "raw": "toy/model.py(1): forward",
                "file": "toy/model.py",
                "line": 1,
                "function": "forward",
                "source_exists": True,
            }
        ],
    }


def _mapping(event_id: str, node: str):
    return {
        "event_id": event_id,
        "kernel_name": f"{event_id}_kernel",
        "selected_node": node,
        "confidence": "high",
        "semantic_frame": {"raw": "toy/model.py(1): forward"},
        "operator_frame": None,
        "primitive_frame": None,
        "model_context_frame": {"raw": "nn.Module: ToyLayer_0", "module": "ToyLayer_0"},
    }


class CommonRuntimeSkeletonTest(unittest.TestCase):
    def test_runtime_skeleton_uses_trace_module_order_and_expected_pattern(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            events = [
                _event("k_000000", ["ToyLinearLayer_0", "ToyModel_0"], 2.0),
                _event("k_000001", ["ToyMlp_0", "ToyLinearLayer_0", "ToyModel_0"], 3.0),
                _event("k_000002", ["ToyFullLayer_0", "ToyModel_0"], 5.0),
            ]
            mappings = [
                _mapping("k_000000", "toy_linear"),
                _mapping("k_000001", "toy_mlp"),
                _mapping("k_000002", "toy_full"),
            ]
            manifest = {"phase": "toy_phase", "rank": 0}
            events_path = tmp / "events.jsonl"
            mapping_path = tmp / "mapping.jsonl"
            manifest_path = tmp / "manifest.json"
            events_path.write_text("\n".join(json.dumps(row) for row in events) + "\n")
            mapping_path.write_text("\n".join(json.dumps(row) for row in mappings) + "\n")
            manifest_path.write_text(json.dumps(manifest))

            skeleton, report = build_runtime_skeleton(
                events_path=events_path,
                mapping_path=mapping_path,
                manifest_path=manifest_path,
                skeleton_config={
                    "layer_modules": [
                        {"class": "ToyLinearLayer", "kind": "linear"},
                        {"class": "ToyFullLayer", "kind": "full"},
                    ],
                    "expected": {
                        "layer_count": 2,
                        "layer_pattern": ["linear", "full"],
                        "module_counts": {
                            "ToyLinearLayer": 1,
                            "ToyFullLayer": 1,
                            "ToyMlp": 1,
                        },
                    },
                },
            )

            self.assertTrue(report["ok"], report["errors"])
            self.assertEqual(
                [item["kind"] for item in skeleton["layer_sequence"]],
                ["linear", "full"],
            )
            self.assertIn("toy_mlp", skeleton["runtime_nodes"])
            self.assertEqual(
                skeleton["module_inventory"]["ToyMlp"]["observed_instance_count"],
                1,
            )


if __name__ == "__main__":
    unittest.main()
