#!/usr/bin/env python3
"""Persist GLM-5.3 mHC occurrence scopes and retire composite fusion members.

This migration upgrades already accepted artifacts.  New SGLang/vLLM profile
builders persist the same layer/substage coordinates during eager-to-production
attribution, so future models should not require a post-hoc migration.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

import yaml

from models.common.timeline_artifact import write_timeline_artifact
from models.glm53_flash.build.glm53_sglang_production_attribution import (
    ANCHOR_TOKEN,
    SUBLAYER_SEGMENT_COUNT,
    segment_kind,
)


COMPOSITE_SCOPED_NODES = {
    "decoder_stack.attn_mhc_pre",
    "decoder_stack.attn_mhc_combine",
    "decoder_stack.ffn_mhc_pre",
    "decoder_stack.ffn_mhc_combine",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_root", type=Path)
    return parser.parse_args()


def _decoded(value: Any, strings: list[str]) -> Any:
    return strings[value] if isinstance(value, int) else value


def _annotate_sglang_step(step: dict[str, Any], strings: list[str]) -> None:
    events = step.get("events") or []
    anchors = [
        index
        for index, event in enumerate(events)
        if ANCHOR_TOKEN
        in str(_decoded(event.get("kernel_name"), strings) or "").lower()
    ]
    if len(anchors) != SUBLAYER_SEGMENT_COUNT:
        raise RuntimeError(
            f"expected {SUBLAYER_SEGMENT_COUNT} mHC anchors, got {len(anchors)}"
        )
    index = {value: offset for offset, value in enumerate(strings)}

    def intern(value: str) -> int:
        if value not in index:
            index[value] = len(strings)
            strings.append(value)
        return index[value]

    for segment_id, (start, stop) in enumerate(
        zip(anchors, [*anchors[1:], len(events)])
    ):
        layer_id = segment_id // 2
        substage = "attention" if segment_id % 2 == 0 else "feed_forward"
        occurrence_id = f"layer_{layer_id:02d}.{substage}"
        for event in events[start:stop]:
            event.update(
                {
                    "layer_id": layer_id,
                    "layer_kind": intern(segment_kind(segment_id)),
                    "substage": intern(substage),
                    "segment_id": segment_id,
                    "occurrence_id": intern(occurrence_id),
                }
            )


def main() -> int:
    args = parse_args()
    changed = []
    for profile_path in sorted((args.model_root / "profiles").glob("**/*.yaml")):
        profile = yaml.safe_load(profile_path.read_text())
        timeline_name = (profile.get("timeline") or {}).get("artifact")
        if not timeline_name:
            continue
        timeline_path = profile_path.parent / str(timeline_name)
        with gzip.open(timeline_path, "rt") as source:
            timeline = json.load(source)
        strings = timeline["strings"]

        if str(profile.get("implementation_id") or "").startswith("sglang_"):
            for step in timeline.get("steps") or []:
                _annotate_sglang_step(step, strings)

        for group in (profile.get("fusion_groups") or {}).values():
            group["ir_nodes"] = [
                target
                for target in group.get("ir_nodes") or []
                if target not in COMPOSITE_SCOPED_NODES
            ]
            if "mhc" in str(group.get("mapping_method") or ""):
                group.setdefault("evidence_scope", {})[
                    "semantic_occurrence_coordinates"
                ] = ["layer_id", "substage", "segment_id", "occurrence_id"]

        for target in COMPOSITE_SCOPED_NODES:
            (profile.get("node_states") or {}).pop(target, None)
            (profile.get("node_metrics") or {}).pop(target, None)

        profile["timeline"]["sha256"] = write_timeline_artifact(
            timeline_path, timeline
        )
        profile_path.write_text(
            yaml.safe_dump(profile, sort_keys=False, width=1000)
        )
        changed.append(profile["profile_id"])

    print(json.dumps({"updated_profiles": changed}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
