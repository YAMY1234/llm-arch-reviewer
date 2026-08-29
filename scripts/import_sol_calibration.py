#!/usr/bin/env python3
"""Validate and import one exact-shape SoL calibration surface.

The command writes a new hardware-spec file. It never modifies the input in
place and refuses to overwrite an existing output unless --force is supplied.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", type=Path, required=True)
    parser.add_argument("--surface", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_mapping(path: Path) -> dict[str, Any]:
    if path.suffix == ".json":
        value = json.loads(path.read_text())
    else:
        value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a mapping")
    return value


def validate_surface(surface: dict[str, Any]) -> None:
    required = (
        "schema_version",
        "hardware_spec_id",
        "surface_id",
        "kernel_plan_fingerprint",
        "match_fields",
        "evidence",
        "points",
    )
    missing = [field for field in required if field not in surface]
    if missing:
        raise ValueError("missing fields: " + ", ".join(missing))
    if surface["schema_version"] != "sol-calibration-surface.v1":
        raise ValueError("expected schema_version=sol-calibration-surface.v1")
    plan_fingerprint = str(surface["kernel_plan_fingerprint"])
    if len(plan_fingerprint) != 64 or any(
        char not in "0123456789abcdef" for char in plan_fingerprint
    ):
        raise ValueError("kernel_plan_fingerprint must be a lowercase SHA256")
    match_fields = list(surface["match_fields"])
    if not match_fields or len(set(match_fields)) != len(match_fields):
        raise ValueError("match_fields must be non-empty and unique")
    evidence = surface["evidence"]
    for field in ("benchmark", "artifact_sha256", "methodology"):
        if not evidence.get(field):
            raise ValueError(f"evidence.{field} is required")
    sha256 = str(evidence["artifact_sha256"])
    if len(sha256) != 64 or any(char not in "0123456789abcdef" for char in sha256):
        raise ValueError("evidence.artifact_sha256 must be a lowercase SHA256")

    identities: set[tuple[tuple[str, str], ...]] = set()
    for index, point in enumerate(surface["points"]):
        match = point.get("match") or {}
        if set(match) != set(match_fields):
            raise ValueError(
                f"points[{index}].match must contain exactly {match_fields}"
            )
        identity = tuple(
            sorted(
                (key, json.dumps(value, sort_keys=True, separators=(",", ":")))
                for key, value in match.items()
            )
        )
        if identity in identities:
            raise ValueError(f"points[{index}] duplicates an exact-shape match")
        identities.add(identity)
        has_time = point.get("attainable_ms") is not None
        has_interval = point.get("attainable_interval_ms") is not None
        has_efficiency = point.get("efficiency") is not None
        if sum((has_time, has_interval, has_efficiency)) != 1:
            raise ValueError(
                f"points[{index}] requires exactly one of attainable_ms, "
                "attainable_interval_ms, or efficiency"
            )
        if has_time and float(point["attainable_ms"]) <= 0:
            raise ValueError(f"points[{index}].attainable_ms must be positive")
        if has_efficiency and not 0 < float(point["efficiency"]) <= 1:
            raise ValueError(f"points[{index}].efficiency must be in (0, 1]")
        if has_interval:
            interval = point["attainable_interval_ms"]
            try:
                p10, p50, p90 = (
                    float(interval["p10"]),
                    float(interval["p50"]),
                    float(interval["p90"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"points[{index}].attainable_interval_ms requires p10/p50/p90"
                ) from exc
            if not 0 < p10 <= p50 <= p90:
                raise ValueError(
                    f"points[{index}].attainable_interval_ms must satisfy "
                    "0 < p10 <= p50 <= p90"
                )


def main() -> int:
    args = parse_args()
    hardware = load_mapping(args.hardware)
    surface = load_mapping(args.surface)
    validate_surface(surface)
    if hardware.get("schema_version") != "hardware-spec.v1":
        raise ValueError(f"{args.hardware}: expected hardware-spec.v1")
    if surface["hardware_spec_id"] != hardware.get("hardware_spec_id"):
        raise ValueError("surface hardware_spec_id does not match the hardware spec")
    if args.output.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite {args.output}; pass --force")

    calibration = hardware.setdefault("calibration", {})
    surfaces = calibration.setdefault("surfaces", {})
    surface_id = surface["surface_id"]
    if surface_id in surfaces:
        raise ValueError(f"hardware spec already contains surface {surface_id!r}")
    surfaces[surface_id] = {
        "kernel_plan_fingerprint": surface["kernel_plan_fingerprint"],
        "match_fields": list(surface["match_fields"]),
        "evidence": surface["evidence"],
        "points": surface["points"],
    }
    hardware["status"] = "partially_calibrated"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(hardware, sort_keys=False))
    print(f"wrote {args.output} with calibration surface {surface_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
