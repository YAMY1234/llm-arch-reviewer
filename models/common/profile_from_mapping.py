#!/usr/bin/env python3
"""Build a profile YAML from trace-derived kernel mapping artifacts."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised manually
    raise SystemExit("requires pyyaml") from exc


def load_json(path: Path) -> Any:
    with path.open() as fh:
        return json.load(fh)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_yaml(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")


def normalize_kernel_name(kernel_name: str, rules: list[dict[str, Any]]) -> str:
    lowered = kernel_name.lower()
    for rule in rules:
        if "contains" in rule and rule["contains"].lower() in lowered:
            return rule["name"]
        if "prefix" in rule and kernel_name.startswith(rule["prefix"]):
            return rule["name"]
    return kernel_name[:120]


def _event_by_id(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {event["event_id"]: event for event in events}


def build_profile_from_mapping(
    *,
    events_path: Path,
    mapping_path: Path,
    manifest_path: Path,
    profile_id: str,
    variant: str,
    node_to_stage: dict[str, str],
    kernel_name_rules: list[dict[str, Any]],
    meta: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    events = load_jsonl(events_path)
    mappings = load_jsonl(mapping_path)
    manifest = load_json(manifest_path)
    events_by_id = _event_by_id(events)
    n_iters = max(1, len(manifest.get("window", {}).get("iter_bounds_us") or []))

    stage_kernel_us: dict[str, Counter[str]] = defaultdict(Counter)
    stage_kernel_count: dict[str, Counter[str]] = defaultdict(Counter)
    stage_total_us: Counter[str] = Counter()
    included_node_us: Counter[str] = Counter()
    excluded_node_us: Counter[str] = Counter()
    unmapped_us = 0.0

    for mapping in mappings:
        event = events_by_id.get(mapping["event_id"])
        if not event:
            continue
        dur_us = float(event.get("dur_us") or 0.0)
        node = mapping.get("selected_node")
        if not node:
            unmapped_us += dur_us
            continue
        stage = node_to_stage.get(node)
        if not stage:
            excluded_node_us[node] += dur_us
            continue
        kernel_name = normalize_kernel_name(str(event.get("kernel_name") or ""), kernel_name_rules)
        stage_total_us[stage] += dur_us
        stage_kernel_us[stage][kernel_name] += dur_us
        stage_kernel_count[stage][kernel_name] += 1
        included_node_us[node] += dur_us

    data: dict[str, Any] = {}
    for stage, total_us in sorted(stage_total_us.items()):
        kernels: list[dict[str, Any]] = []
        for name, kernel_us in stage_kernel_us[stage].most_common():
            count = stage_kernel_count[stage][name]
            kernels.append(
                {
                    "name": name,
                    "count": int(count),
                    "count_per_iter": round(count / n_iters, 3),
                    "avg_us": round(kernel_us / count, 3) if count else 0.0,
                    "total_us": round(kernel_us / n_iters, 3),
                    "share_in_stage_pct": round(100.0 * kernel_us / total_us, 2)
                    if total_us
                    else 0.0,
                }
            )
        data[stage] = {
            variant: {
                "ms_per_iter": round(total_us / n_iters / 1000.0, 3),
                "kernels": kernels,
            }
        }

    profile = {
        "meta": {
            "source": "pytorch_profiler_trace_mapping",
            "profile_id": profile_id,
            "phase": manifest.get("phase"),
            "variant": variant,
            "rank": manifest.get("rank"),
            "n_iters": n_iters,
            "trace_path": manifest.get("trace_path"),
            "config_path": manifest.get("config_path"),
            "source_repo": manifest.get("source_repo"),
            "source_commit": manifest.get("source_commit"),
            "mapping_manifest": str(manifest_path),
            **(meta or {}),
        },
        "data": data,
    }
    total_us = sum(float(event.get("dur_us") or 0.0) for event in events)
    included_us = sum(included_node_us.values())
    report = {
        "profile_id": profile_id,
        "variant": variant,
        "n_iters": n_iters,
        "stage_count": len(data),
        "total_kernel_us": total_us,
        "included_kernel_us": included_us,
        "included_duration_ratio": included_us / total_us if total_us else 0.0,
        "unmapped_kernel_us": unmapped_us,
        "included_nodes": dict(included_node_us),
        "excluded_nodes": dict(excluded_node_us),
        "stages": {
            stage: {
                "ms_per_iter": data[stage][variant]["ms_per_iter"],
                "kernel_count": sum(stage_kernel_count[stage].values()),
            }
            for stage in sorted(data)
        },
    }
    return profile, report


def write_profile_report_markdown(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Profile From Mapping Report",
        "",
        f"- profile_id: `{report['profile_id']}`",
        f"- variant: `{report['variant']}`",
        f"- n_iters: `{report['n_iters']}`",
        f"- stage_count: `{report['stage_count']}`",
        f"- included_duration_ratio: `{report['included_duration_ratio']:.4f}`",
        "",
        "## Stages",
        "",
    ]
    for stage, cell in report["stages"].items():
        lines.append(
            f"- `{stage}`: {cell['ms_per_iter']:.3f} ms/iter, "
            f"kernels={cell['kernel_count']}"
        )
    lines.extend(["", "## Included Nodes", ""])
    for node, dur_us in sorted(
        report["included_nodes"].items(), key=lambda item: -float(item[1])
    ):
        lines.append(f"- `{node}`: {float(dur_us) / 1000.0:.3f} ms total")
    lines.extend(["", "## Excluded Nodes", ""])
    for node, dur_us in sorted(
        report["excluded_nodes"].items(), key=lambda item: -float(item[1])
    ):
        lines.append(f"- `{node}`: {float(dur_us) / 1000.0:.3f} ms total")
    path.write_text("\n".join(lines) + "\n")
