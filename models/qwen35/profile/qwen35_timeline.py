"""Qwen3.5 Timeline routing derived from the canonical Architecture hierarchy."""

from __future__ import annotations

from pathlib import Path
import sys

import yaml

from models.common.timeline_artifact import build_ancestor_target_resolver


REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_IR_PATH = REPO_ROOT / "catalog" / "qwen35" / "model_ir.yaml"
EXECUTION_PATH = (
    REPO_ROOT
    / "catalog"
    / "qwen35"
    / "execution_paths"
    / "attention_dp4_moe_ep4.yaml"
)
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from llm_arch_v2.compiler import apply_execution_plan  # noqa: E402


MODEL_IR = yaml.safe_load(MODEL_IR_PATH.read_text())
EXECUTION_PLAN = yaml.safe_load(EXECUTION_PATH.read_text())
MODEL_VIEWS = apply_execution_plan(
    MODEL_IR, EXECUTION_PLAN, source=EXECUTION_PATH
)
QWEN35_TIMELINE_TARGETS = build_ancestor_target_resolver(MODEL_VIEWS)
