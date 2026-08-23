#!/usr/bin/env python3
"""Build DeepSeek-V4 viewer data from IR and profile YAML."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.common.build_view import build_and_write_view


MODEL_ROOT = Path(__file__).resolve().parent.parent
MODEL_META = {
    "model_label": "DeepSeek-V4",
    "subtitle": "ELK + SVG · drill-down · sglang trace overlay",
}


def swap_hca_csa_for_trace_profiles(profile: dict[str, Any], _path: Path) -> dict[str, Any]:
    """Compatibility shim for legacy DSV4 trace CSV labels.

    Older trace-kernel-learning CSVs mislabeled HCA and CSA. Keeping the fix as
    a DSV4 profile transform makes the common builder model-agnostic.
    """

    if profile.get("meta", {}).get("source") != "trace":
        return profile
    data = profile.get("data", {}) or {}
    for cell in data.values():
        if not isinstance(cell, dict):
            continue
        hca = cell.get("HCA")
        csa = cell.get("CSA")
        if hca is not None and csa is not None:
            cell["HCA"], cell["CSA"] = csa, hca
        elif hca is not None:
            cell["CSA"] = cell.pop("HCA")
        elif csa is not None:
            cell["HCA"] = cell.pop("CSA")
    profile.setdefault("meta", {})["hca_csa_swapped_at_load"] = True
    return profile


def main() -> int:
    build_and_write_view(
        model_root=MODEL_ROOT,
        model_meta=MODEL_META,
        profile_transform=swap_hca_csa_for_trace_profiles,
        profile_layer_types=["HCA", "CSA", "ALL"],
        config_order=["v4_pro", "v4_flash"],
        node_kernel_limit=6,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
