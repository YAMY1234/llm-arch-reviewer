#!/usr/bin/env python3
"""Common view builder for model architecture IR.

Model build wrappers should only provide paths, metadata, and optional profile
transforms. This module owns the generic conversion from view-based IR plus
profile YAML files to `arch_data.json` for the static viewer.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Callable

try:
    import yaml
except ImportError:
    print("pip install pyyaml", file=sys.stderr)
    raise


ProfileTransform = Callable[[dict[str, Any], Path], dict[str, Any]]


def load_yaml(path: Path) -> Any:
    with path.open() as fh:
        return yaml.safe_load(fh)


def load_profiles(
    profiles_dir: Path,
    *,
    profile_transform: ProfileTransform | None = None,
) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    if not profiles_dir.exists():
        return profiles
    for path in sorted(profiles_dir.glob("*.yaml")):
        profile = load_yaml(path)
        if profile_transform:
            profile = profile_transform(profile, path)
        profiles[path.stem] = profile
    return profiles


def infer_layer_types(profiles: dict[str, dict[str, Any]]) -> list[str]:
    layer_types: set[str] = set()
    for profile in profiles.values():
        for cell in (profile.get("data", {}) or {}).values():
            if not isinstance(cell, dict):
                continue
            for layer_type, value in cell.items():
                if isinstance(value, dict):
                    layer_types.add(layer_type)
    if not layer_types:
        return ["ALL"]
    if "ALL" in layer_types:
        return sorted(layer_type for layer_type in layer_types if layer_type != "ALL") + ["ALL"]
    return sorted(layer_types)


def lookup_stage_ms(
    stage_id: str,
    layer_type: str,
    stages_by_id: dict[str, dict[str, Any]],
    profile_data: dict[str, Any],
) -> float | None:
    stage = stages_by_id.get(stage_id)
    if not stage:
        return None
    candidates = list(stage.get("trace_aliases", []) or []) + [
        stage.get("pdf_name") or stage_id
    ]
    seen: set[str] = set()
    total = 0.0
    hit = False
    for key in candidates:
        if not key or key in seen:
            continue
        seen.add(key)
        cell = profile_data.get(key)
        if not cell:
            continue
        if layer_type in cell:
            total += float(cell[layer_type].get("ms_per_iter") or 0)
            hit = True
        elif "ALL" in cell:
            total += float(cell["ALL"].get("ms_per_iter") or 0)
            hit = True
    return total if hit else None


def collect_node_kernels(
    node: dict[str, Any],
    layer_type: str,
    stages_by_id: dict[str, dict[str, Any]],
    profile_data: dict[str, Any],
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for stage_key in node.get("stage_keys") or []:
        stage = stages_by_id.get(stage_key)
        if not stage:
            continue
        for alias in stage.get("trace_aliases", []) or []:
            cell = profile_data.get(alias)
            if not cell:
                continue
            cell_layer_type = cell.get(layer_type) or cell.get("ALL")
            if not cell_layer_type:
                continue
            for kernel in cell_layer_type.get("kernels", []) or []:
                out.append({**kernel, "stage_alias": alias})
    out.sort(key=lambda kernel: -kernel.get("total_us", 0))
    return out[:limit]


def _collect_leaf_stage_keys(
    view_name: str,
    views: dict[str, dict[str, Any]],
    memo: dict[str, set[str]],
) -> set[str]:
    if view_name in memo:
        return memo[view_name]
    memo[view_name] = set()
    out: set[str] = set()
    view = views.get(view_name)
    if not view or "same_as" in view:
        return out
    for node in view.get("nodes", []):
        out.update(node.get("stage_keys") or [])
        drill = node.get("drill")
        if drill and drill != view_name:
            out |= _collect_leaf_stage_keys(drill, views, memo)
    memo[view_name] = out
    return out


def _node_ms_recursive(
    node: dict[str, Any],
    layer_type: str,
    views: dict[str, dict[str, Any]],
    stages_by_id: dict[str, dict[str, Any]],
    profile_data: dict[str, Any],
    memo: dict[str, set[str]],
) -> tuple[float, bool]:
    stage_keys = set(node.get("stage_keys") or [])
    drill = node.get("drill")
    if drill and drill in views:
        stage_keys |= _collect_leaf_stage_keys(drill, views, memo)
    if not stage_keys:
        return 0.0, False
    total = 0.0
    hit = False
    for stage_key in stage_keys:
        value = lookup_stage_ms(stage_key, layer_type, stages_by_id, profile_data)
        if value is not None:
            total += value
            hit = True
    return total, hit


def aggregate_drill_blocks(
    views: dict[str, dict[str, Any]],
    enriched: dict[str, dict[str, Any]],
    stages_by_id: dict[str, dict[str, Any]],
    profiles: dict[str, dict[str, Any]],
    layer_types: list[str],
) -> None:
    leaf_memo: dict[str, set[str]] = {}

    for view_name, view in views.items():
        if "same_as" in view:
            continue
        for node in view.get("nodes", []):
            drill = node.get("drill")
            if not drill or drill not in views:
                continue
            target = views[drill]
            if "same_as" in target:
                continue

            child_stage_keys = set(_collect_leaf_stage_keys(drill, views, leaf_memo))
            child_stage_keys.update(node.get("stage_keys") or [])
            if not child_stage_keys:
                continue

            agg_per_profile: dict[str, dict[str, Any]] = {}
            for profile_id, profile in profiles.items():
                profile_data = profile.get("data", {})
                per_layer_type: dict[str, Any] = {}
                for layer_type in layer_types:
                    ms = 0.0
                    hit = False
                    kernels_acc: list[dict[str, Any]] = []
                    for stage_key in sorted(child_stage_keys):
                        value = lookup_stage_ms(stage_key, layer_type, stages_by_id, profile_data)
                        if value is not None:
                            ms += value
                            hit = True
                        stage = stages_by_id.get(stage_key) or {}
                        for alias in stage.get("trace_aliases", []) or []:
                            cell = profile_data.get(alias)
                            if not cell:
                                continue
                            cell_layer_type = cell.get(layer_type) or cell.get("ALL")
                            if not cell_layer_type:
                                continue
                            for kernel in cell_layer_type.get("kernels", []) or []:
                                kernels_acc.append({**kernel, "stage_alias": alias})
                    if not hit:
                        continue
                    kernels_acc.sort(key=lambda kernel: -kernel.get("total_us", 0))

                    stage_leaf_count: dict[str, int] = {}
                    for child in target.get("nodes", []):
                        if child.get("drill"):
                            continue
                        for stage_key in child.get("stage_keys") or []:
                            stage_leaf_count[stage_key] = stage_leaf_count.get(stage_key, 0) + 1

                    children = []
                    for child in target.get("nodes", []):
                        if child.get("drill"):
                            child_ms, child_hit = _node_ms_recursive(
                                child, layer_type, views, stages_by_id, profile_data, leaf_memo
                            )
                            kind = "module"
                        else:
                            child_ms = 0.0
                            child_hit = False
                            for stage_key in child.get("stage_keys") or []:
                                value = lookup_stage_ms(stage_key, layer_type, stages_by_id, profile_data)
                                if value is not None:
                                    child_ms += value / max(1, stage_leaf_count.get(stage_key, 1))
                                    child_hit = True
                            kind = "leaf"
                        if child_hit:
                            children.append(
                                {
                                    "id": child["id"],
                                    "label": (child.get("label") or child["id"]).replace("\n", " "),
                                    "ms": round(child_ms, 3),
                                    "share_pct": round(100 * child_ms / ms, 1) if ms > 0 else 0,
                                    "kind": kind,
                                }
                            )
                    children.sort(key=lambda child: -child["ms"])

                    per_layer_type[layer_type] = {
                        "ms_per_iter": round(ms, 3),
                        "kernels": kernels_acc[:10],
                        "children": children[:10],
                    }
                if per_layer_type:
                    agg_per_profile[profile_id] = per_layer_type

            if not agg_per_profile:
                continue

            node_slot = enriched.setdefault(
                view_name,
                {"title": view.get("title", view_name), "nodes_profile": {}},
            )["nodes_profile"].setdefault(node["id"], {})
            for profile_id, per_layer_type in agg_per_profile.items():
                current = node_slot.setdefault(profile_id, {})
                for layer_type, aggregate in per_layer_type.items():
                    if layer_type in current:
                        if aggregate["ms_per_iter"] > current[layer_type].get("ms_per_iter", 0):
                            current[layer_type]["ms_per_iter"] = aggregate["ms_per_iter"]
                        current[layer_type]["kernels"] = aggregate["kernels"]
                        current[layer_type]["children"] = aggregate["children"]
                        current[layer_type]["from_aggregate"] = True
                    else:
                        current[layer_type] = {**aggregate, "from_aggregate": True}


def enrich_views(
    views: dict[str, dict[str, Any]],
    stages_by_id: dict[str, dict[str, Any]],
    profiles: dict[str, dict[str, Any]],
    layer_types: list[str],
    *,
    node_kernel_limit: int = 10,
) -> dict[str, dict[str, Any]]:
    enriched: dict[str, dict[str, Any]] = {}
    for view_name, view in views.items():
        if "same_as" in view:
            enriched[view_name] = {"same_as": view["same_as"], "title": view.get("title", "")}
            continue
        node_data: dict[str, dict[str, Any]] = {}
        for node in view.get("nodes", []):
            stage_keys = node.get("stage_keys") or []
            if not stage_keys:
                continue
            per_profile: dict[str, dict[str, Any]] = {}
            for profile_id, profile in profiles.items():
                profile_data = profile.get("data", {})
                per_layer_type: dict[str, Any] = {}
                for layer_type in layer_types:
                    total = 0.0
                    hit = False
                    for stage_key in stage_keys:
                        value = lookup_stage_ms(stage_key, layer_type, stages_by_id, profile_data)
                        if value is not None:
                            total += value
                            hit = True
                    if hit:
                        per_layer_type[layer_type] = {
                            "ms_per_iter": round(total, 3),
                            "kernels": collect_node_kernels(
                                node,
                                layer_type,
                                stages_by_id,
                                profile_data,
                                limit=node_kernel_limit,
                            ),
                        }
                if per_layer_type:
                    per_profile[profile_id] = per_layer_type
            if per_profile:
                node_data[node["id"]] = per_profile
        enriched[view_name] = {
            "title": view.get("title", view_name),
            "nodes_profile": node_data,
        }
    return enriched


def transform_code_links(arch: dict[str, Any], source_map_doc: dict[str, Any]) -> dict[str, Any]:
    rules = source_map_doc.get("source_map", []) or []
    display_cfg = source_map_doc.get("display", {}) or {}
    shorten = display_cfg.get("shorten_paths", True)

    def to_dict(raw: str) -> dict[str, Any]:
        match = re.match(r"^(\S+\.[A-Za-z0-9]+)(?::([A-Za-z0-9_\-]+))?(\s+.*)?$", raw)
        if not match:
            return {"raw": raw, "file": None, "line": None, "url": None, "display": raw}
        file, ref, sym = match.group(1), match.group(2), (match.group(3) or "").strip()
        line_i = None
        line_end = None
        if ref:
            range_match = re.match(r"^(\d+)-(\d+)$", ref)
            single_match = re.match(r"^(\d+)$", ref)
            if range_match:
                line_i = int(range_match.group(1))
                line_end = int(range_match.group(2))
            elif single_match:
                line_i = int(single_match.group(1))
            else:
                sym = (ref + (" " + sym if sym else "")).strip()

        url = None
        for rule in rules:
            prefix = rule.get("prefix", "")
            if prefix and not file.startswith(prefix):
                continue
            tail = file[len(prefix) :] if prefix else file
            full = rule.get("path_prefix", "") + tail
            base = f"https://github.com/{rule['repo']}/blob/{rule['commit']}/{full}"
            if line_i and line_end:
                base += f"#L{line_i}-L{line_end}"
            elif line_i:
                base += f"#L{line_i}"
            url = base
            break

        if shorten:
            short_file = file.split("/")[-1]
            if line_i and line_end:
                display = f"{short_file}:{line_i}-{line_end}"
            elif line_i:
                display = f"{short_file}:{line_i}"
            else:
                display = short_file
            if sym:
                display += " — " + sym
        else:
            display = raw

        return {
            "raw": raw,
            "file": file,
            "line": line_i,
            "line_end": line_end,
            "url": url,
            "display": display,
        }

    for view in arch.get("views", {}).values():
        if "same_as" in view:
            continue
        for node in view.get("nodes", []):
            code_links = node.get("code_links")
            if not code_links:
                continue
            node["code_links"] = [
                to_dict(link) if isinstance(link, str) else link for link in code_links
            ]
    return arch


def resolve_same_as_views(views: dict[str, dict[str, Any]]) -> None:
    for view_name, view in list(views.items()):
        if "same_as" not in view:
            continue
        target = view["same_as"]
        if target in views and "same_as" not in views[target]:
            views[view_name] = {
                "title": view.get("title", views[target].get("title", target)),
                "nodes": views[target]["nodes"],
                "edges": views[target]["edges"],
                "alias_of": target,
            }


def build_parent_map(views: dict[str, dict[str, Any]]) -> dict[str, str]:
    parent_map: dict[str, str] = {}
    for view_name, view in views.items():
        for node in view.get("nodes", []):
            drill = node.get("drill")
            if drill and drill not in parent_map:
                parent_map[drill] = view_name
    return parent_map


def load_configs(ir_dir: Path, config_order: list[str] | None = None) -> dict[str, Any]:
    paths = {
        path.stem[len("config.") :]: path for path in sorted(ir_dir.glob("config.*.yaml"))
    }
    keys: list[str] = []
    if config_order:
        keys.extend(key for key in config_order if key in paths)
    keys.extend(key for key in sorted(paths) if key not in keys)
    return {key: load_yaml(paths[key]) for key in keys}


def build_view_bundle(
    *,
    model_root: Path,
    model_meta: dict[str, Any],
    arch_path: Path | None = None,
    profile_transform: ProfileTransform | None = None,
    profile_layer_types: list[str] | None = None,
    config_order: list[str] | None = None,
    node_kernel_limit: int = 10,
    default_view: str = "top",
) -> dict[str, Any]:
    model_id = model_root.name
    repo_root = model_root.parent.parent
    ir_dir = model_root / "ir"
    profiles_dir = ir_dir / "profiles"

    if arch_path is None:
        arch_files = sorted(ir_dir.glob("arch.*.yaml"))
        if len(arch_files) != 1:
            raise RuntimeError(
                f"expected exactly one arch.*.yaml under {ir_dir}, got {arch_files}"
            )
        arch_path = arch_files[0]
    arch = load_yaml(arch_path)
    stages_doc = load_yaml(ir_dir / "stages.yaml")
    profiles = load_profiles(profiles_dir, profile_transform=profile_transform)
    layer_types = profile_layer_types or infer_layer_types(profiles)

    configs = load_configs(ir_dir, config_order=config_order)

    source_map_path = ir_dir / "source_map.yaml"
    if source_map_path.exists():
        arch = transform_code_links(arch, load_yaml(source_map_path))

    stages_by_id = {stage["id"]: stage for stage in stages_doc["stages"]}
    views = arch.get("views", {})
    resolve_same_as_views(views)

    enriched = enrich_views(
        views,
        stages_by_id,
        profiles,
        layer_types,
        node_kernel_limit=node_kernel_limit,
    )
    aggregate_drill_blocks(views, enriched, stages_by_id, profiles, layer_types)

    meta = dict(model_meta)
    meta.update(
        {
            "model_id": model_id,
            "build": f"models/{model_id}/build/build_view.py",
            "arch_source": str(arch_path.relative_to(repo_root))
            if arch_path.is_relative_to(repo_root)
            else str(arch_path),
            "ir_files": sorted(path.name for path in ir_dir.glob("*.yaml")),
            "profile_count": len(profiles),
            "view_count": len(views),
        }
    )

    return {
        "schema_version": "0.3",
        "meta": meta,
        "parent": build_parent_map(views),
        "views": views,
        "stages": stages_by_id,
        "configs": configs,
        "profiles": profiles,
        "enriched": enriched,
        "default_view": default_view,
    }


def write_bundle_outputs(model_root: Path, bundle: dict[str, Any]) -> list[Path]:
    repo_root = model_root.parent.parent
    model_id = model_root.name
    out_targets: list[Path] = []
    docs_root = repo_root / "docs"
    if docs_root.exists():
        docs_dir = docs_root / model_id
        docs_dir.mkdir(parents=True, exist_ok=True)
        out_targets.append(docs_dir / "arch_data.json")
    local_out = model_root / "out"
    local_out.mkdir(exist_ok=True)
    out_targets.append(local_out / "arch_data.json")

    for out_path in out_targets:
        with out_path.open("w") as fh:
            json.dump(bundle, fh, indent=2, ensure_ascii=False)
        print(f"wrote {out_path}")
    return out_targets


def print_bundle_summary(bundle: dict[str, Any]) -> None:
    views = bundle["views"]
    enriched = bundle["enriched"]
    meta = bundle["meta"]
    print(
        f"  model: {meta['model_id']}, views: {len(views)}, "
        f"profiles: {meta['profile_count']}"
    )
    for view_name, view in views.items():
        node_count = len(view.get("nodes", []))
        edge_count = len(view.get("edges", []))
        enriched_count = len(enriched[view_name].get("nodes_profile", {}))
        print(
            f"    {view_name:22s} nodes={node_count:3d} "
            f"edges={edge_count:3d} enriched={enriched_count}"
        )


def build_and_write_view(
    *,
    model_root: Path,
    model_meta: dict[str, Any],
    arch_path: Path | None = None,
    profile_transform: ProfileTransform | None = None,
    profile_layer_types: list[str] | None = None,
    config_order: list[str] | None = None,
    node_kernel_limit: int = 10,
    default_view: str = "top",
) -> dict[str, Any]:
    bundle = build_view_bundle(
        model_root=model_root,
        model_meta=model_meta,
        arch_path=arch_path,
        profile_transform=profile_transform,
        profile_layer_types=profile_layer_types,
        config_order=config_order,
        node_kernel_limit=node_kernel_limit,
        default_view=default_view,
    )
    write_bundle_outputs(model_root, bundle)
    print_bundle_summary(bundle)
    return bundle
