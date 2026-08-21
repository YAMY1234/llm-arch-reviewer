from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.common.profile_validation import validate_model_profile  # noqa: E402


def _write_toy_model(tmp: Path, *, missing_stage_policy: str | None) -> tuple[Path, Path]:
    model_root = tmp / "models" / "toy"
    ir = model_root / "ir"
    profiles = ir / "profiles"
    profiles.mkdir(parents=True)

    (ir / "arch.toy.yaml").write_text(
        """
views:
  top:
    title: toy
    nodes:
      - {id: dense, label: Dense, shape: gemm, stage_keys: [toy_dense]}
      - {id: source_only, label: Source Only, shape: elem, stage_keys: [toy_source_only]}
    edges: []
""".lstrip()
    )

    policy_lines = ""
    if missing_stage_policy:
        policy_lines = f"""
    profile_policy: {missing_stage_policy}
    profile_policy_reason: covered by source only in this test
"""
    (ir / "stages.yaml").write_text(
        f"""
stages:
  - id: toy_dense
    label: toy dense
    pdf_name: toy_dense_pdf
    trace_aliases: [toy_dense_alias]
  - id: toy_source_only
    label: toy source-only stage
    pdf_name: toy_source_only_pdf
    trace_aliases: [toy_source_only_alias]{policy_lines}
""".lstrip()
    )

    (profiles / "toy_profile.yaml").write_text(
        """
meta:
  source: unit test
data:
  toy_dense_alias:
    fast:
      ms_per_iter: 1.25
      kernels:
        - {name: toy_dense_kernel, total_us: 1250}
""".lstrip()
    )

    (ir / "toy_arch_data.json").write_text(
        json.dumps(
            {
                "enriched": {
                    "top": {
                        "nodes_profile": {
                            "dense": {
                                "toy_profile": {
                                    "fast": {
                                        "ms_per_iter": 1.25,
                                        "kernels": [{"name": "toy_dense_kernel"}],
                                    }
                                }
                            }
                        }
                    }
                }
            }
        )
    )

    validation_config = ir / "validation.yaml"
    validation_config.write_text(
        """
profile: toy_profile
arch_data_path: toy_arch_data.json
expected_variants: [fast]
required_profile_stages:
  - {stage: toy_dense, variants: all, nonzero: true}
required_kernels:
  - {stage: toy_dense, variant: fast, names: [toy_dense_kernel]}
required_enriched_nodes:
  - {view: top, node: dense, profile: toy_profile, variant: fast}
forbidden_enriched_kernel_substrings:
  - {view: top, node: dense, profile: toy_profile, variant: fast, substrings: [allreduce]}
""".lstrip()
    )
    return model_root, validation_config


class CommonProfileValidationTest(unittest.TestCase):
    def test_validation_uses_model_config_not_qwen_names(self):
        with TemporaryDirectory() as td:
            model_root, config = _write_toy_model(
                Path(td), missing_stage_policy="source_only"
            )

            report = validate_model_profile(
                model_root=model_root,
                validation_config_path=config,
            )

            self.assertTrue(report.ok, report.errors)
            self.assertEqual(report.variants, ["fast"])
            self.assertTrue(
                any("toy_source_only" in warning for warning in report.warnings),
                report.warnings,
            )

    def test_missing_profile_stage_requires_explicit_policy(self):
        with TemporaryDirectory() as td:
            model_root, config = _write_toy_model(Path(td), missing_stage_policy=None)

            report = validate_model_profile(
                model_root=model_root,
                validation_config_path=config,
            )

            self.assertFalse(report.ok)
            self.assertTrue(
                any("toy_source_only" in error for error in report.errors),
                report.errors,
            )


if __name__ == "__main__":
    unittest.main()
