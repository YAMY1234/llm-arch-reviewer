"""Compile stable model IR plus execution, source, and profile overlays.

The V2 catalog deliberately keeps architecture independent from source code and
measurements:

* model_ir.yaml owns stable semantic nodes and data-flow edges;
* execution plans derive topology-specific graphs from that model IR;
* implementation bindings attach commit-specific symbols and links;
* profiles attach measurements to existing execution nodes.

The compiler emits a V2 bundle and a compatibility projection consumed by the
existing static viewer.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised by the CLI
    raise SystemExit("llm-arch-reviewer V2 requires pyyaml") from exc


class CatalogError(ValueError):
    """Raised when catalog inputs violate the V2 contract."""


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        value = yaml.safe_load(fh)
    if not isinstance(value, dict):
        raise CatalogError(f"{path}: expected a YAML mapping")
    return value


def _require(document: dict[str, Any], fields: Iterable[str], *, source: Path) -> None:
    missing = [field for field in fields if field not in document]
    if missing:
        raise CatalogError(f"{source}: missing required fields: {', '.join(missing)}")


def _validate_schema_version(
    document: dict[str, Any], expected: str, *, source: Path
) -> None:
    actual = document.get("schema_version")
    if actual != expected:
        raise CatalogError(f"{source}: expected schema_version={expected!r}, got {actual!r}")


def _node_index(views: dict[str, Any], *, source: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    allowed_shapes = {"io", "block", "gemm", "attn", "moe", "norm", "elem", "cache"}
    for view_id, view in views.items():
        if not isinstance(view, dict):
            raise CatalogError(f"{source}: view {view_id!r} must be a mapping")
        node_ids: set[str] = set()
        for node in view.get("nodes", []) or []:
            if not isinstance(node, dict):
                raise CatalogError(f"{source}: view {view_id!r} has a non-mapping node")
            node_id = node.get("id")
            if not node_id:
                raise CatalogError(f"{source}: view {view_id!r} has a node without id")
            for field in ("label", "shape", "semantic_op"):
                if not node.get(field):
                    raise CatalogError(
                        f"{source}: node {view_id}.{node_id} is missing {field!r}"
                    )
            if node["shape"] not in allowed_shapes:
                raise CatalogError(
                    f"{source}: node {view_id}.{node_id} has unknown shape {node['shape']!r}"
                )
            if node_id in node_ids:
                raise CatalogError(
                    f"{source}: duplicate node id {node_id!r} in view {view_id!r}"
                )
            node_ids.add(node_id)
            index[f"{view_id}.{node_id}"] = node

        for edge in view.get("edges", []) or []:
            for endpoint in ("from", "to"):
                if edge.get(endpoint) not in node_ids:
                    raise CatalogError(
                        f"{source}: edge in {view_id!r} references unknown "
                        f"{endpoint} node {edge.get(endpoint)!r}"
                    )
        for node in view.get("nodes", []) or []:
            drill = node.get("drill")
            if drill and drill not in views:
                raise CatalogError(
                    f"{source}: node {view_id}.{node['id']} drills into unknown view {drill!r}"
                )
    return index


def _split_target(target: str, *, source: Path) -> tuple[str, str]:
    if "." not in target:
        raise CatalogError(f"{source}: target {target!r} must be '<view>.<node>'")
    return target.split(".", 1)


def _find_node(views: dict[str, Any], target: str, *, source: Path) -> dict[str, Any]:
    view_id, node_id = _split_target(target, source=source)
    view = views.get(view_id)
    if not view:
        raise CatalogError(f"{source}: target {target!r} references unknown view")
    for node in view.get("nodes", []) or []:
        if node.get("id") == node_id:
            return node
    raise CatalogError(f"{source}: target {target!r} references unknown node")


def _deep_merge(target: dict[str, Any], overlay: dict[str, Any]) -> None:
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def _insert_after(
    views: dict[str, Any], transform: dict[str, Any], *, source: Path
) -> None:
    view_id, after_id = _split_target(transform.get("after", ""), source=source)
    view = views.get(view_id)
    if not view:
        raise CatalogError(f"{source}: insert_after references unknown view {view_id!r}")
    nodes = view.get("nodes", []) or []
    if not any(node.get("id") == after_id for node in nodes):
        raise CatalogError(f"{source}: insert_after references unknown node {transform['after']!r}")

    new_node = copy.deepcopy(transform.get("node"))
    if not isinstance(new_node, dict) or not new_node.get("id"):
        raise CatalogError(f"{source}: insert_after requires node.id")
    if any(node.get("id") == new_node["id"] for node in nodes):
        raise CatalogError(
            f"{source}: insert_after creates duplicate node {view_id}.{new_node['id']}"
        )

    insert_at = next(i for i, node in enumerate(nodes) if node.get("id") == after_id) + 1
    nodes.insert(insert_at, new_node)
    view["nodes"] = nodes

    redirected: list[dict[str, Any]] = []
    for edge in view.get("edges", []) or []:
        edge_copy = copy.deepcopy(edge)
        if edge_copy.get("from") == after_id:
            edge_copy["from"] = new_node["id"]
        redirected.append(edge_copy)
    redirected.append(
        {
            "from": after_id,
            "to": new_node["id"],
            **copy.deepcopy(transform.get("edge", {})),
        }
    )
    view["edges"] = redirected


def apply_execution_plan(
    model_ir: dict[str, Any], plan: dict[str, Any], *, source: Path
) -> dict[str, Any]:
    views = copy.deepcopy(model_ir["views"])
    for transform in plan.get("transforms", []) or []:
        op = transform.get("op")
        if op == "annotate_node":
            node = _find_node(views, transform.get("target", ""), source=source)
            overlay = transform.get("set")
            if not isinstance(overlay, dict):
                raise CatalogError(f"{source}: annotate_node requires a 'set' mapping")
            _deep_merge(node, overlay)
        elif op == "insert_after":
            _insert_after(views, transform, source=source)
        else:
            raise CatalogError(f"{source}: unsupported execution transform {op!r}")
    _node_index(views, source=source)
    return views


def _fingerprint_payload(
    model_ir: dict[str, Any], plan: dict[str, Any], views: dict[str, Any]
) -> dict[str, Any]:
    """Return the structural payload used for execution-path identity.

    Labels, notes, source links, profiles, and concrete TP degree are excluded.
    A TP-only template therefore keeps one identity across TP2/TP4/TP8 while
    topology or communication-graph changes produce a new fingerprint.
    """

    structural_views: dict[str, Any] = {}
    for view_id, view in sorted(views.items()):
        structural_views[view_id] = {
            "nodes": [
                {
                    key: copy.deepcopy(node[key])
                    for key in (
                        "id",
                        "shape",
                        "drill",
                        "semantic_op",
                        "execution",
                    )
                    if key in node
                }
                for node in view.get("nodes", []) or []
            ],
            "edges": [
                {
                    key: copy.deepcopy(edge[key])
                    for key in ("from", "to", "kind", "shape", "dtype")
                    if key in edge
                }
                for edge in view.get("edges", []) or []
            ],
        }
    return {
        "model_id": model_ir["model_id"],
        "model_ir_version": model_ir["ir_version"],
        "execution_path_id": plan["execution_path_id"],
        "execution_plan_version": plan["plan_version"],
        "parallelism_axes": plan.get("parallelism_axes", {}),
        "views": structural_views,
    }


def execution_fingerprint(
    model_ir: dict[str, Any], plan: dict[str, Any], views: dict[str, Any]
) -> str:
    payload = json.dumps(
        _fingerprint_payload(model_ir, plan, views),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return "exec_" + hashlib.sha256(payload).hexdigest()[:16]


def derive_parent_map(views: dict[str, Any]) -> dict[str, str]:
    parents: dict[str, str] = {}
    for view_id, view in views.items():
        for node in view.get("nodes", []) or []:
            drill = node.get("drill")
            if drill and drill not in parents:
                parents[drill] = view_id
    return parents


def _code_link(
    source_repo: str, source_commit: str, raw_link: dict[str, Any]
) -> dict[str, Any]:
    file_path = raw_link["file"]
    line = raw_link.get("line")
    line_end = raw_link.get("line_end")
    anchor = ""
    if line:
        anchor = f"#L{line}"
        if line_end and line_end != line:
            anchor += f"-L{line_end}"
    url = f"{source_repo.rstrip('/')}/blob/{source_commit}/{file_path}{anchor}"
    display = raw_link.get("display") or raw_link.get("symbol") or file_path
    return {
        "raw": raw_link.get("symbol") or display,
        "file": file_path,
        "line": line,
        "line_end": line_end,
        "url": url,
        "display": display,
        "symbol": raw_link.get("symbol"),
    }


def compile_binding(binding: dict[str, Any], *, source: Path) -> dict[str, Any]:
    compiled = {
        key: copy.deepcopy(binding[key])
        for key in (
            "implementation_id",
            "label",
            "model_id",
            "execution_path_id",
            "source_repo",
            "source_commit",
            "container",
            "backend",
            "extends",
        )
        if key in binding
    }
    compiled["node_bindings"] = {}
    for target, node_binding in (binding.get("node_bindings") or {}).items():
        if not isinstance(node_binding, dict):
            raise CatalogError(f"{source}: binding for {target!r} must be a mapping")
        for link in node_binding.get("links", []) or []:
            if not isinstance(link, dict) or not link.get("file") or not link.get("symbol"):
                raise CatalogError(
                    f"{source}: binding link for {target!r} requires file and symbol"
                )
        compiled_binding = copy.deepcopy(node_binding)
        compiled_binding["code_links"] = [
            _code_link(binding["source_repo"], str(binding["source_commit"]), link)
            for link in node_binding.get("links", []) or []
        ]
        compiled["node_bindings"][target] = compiled_binding
    return compiled


def apply_binding(views: dict[str, Any], binding: dict[str, Any], *, source: Path) -> None:
    for target, node_binding in (binding.get("node_bindings") or {}).items():
        node = _find_node(views, target, source=source)
        node["code_links"] = copy.deepcopy(node_binding.get("code_links", []))
        node["implementation_binding"] = {
            "implementation_id": binding["implementation_id"],
            "symbols": copy.deepcopy(node_binding.get("symbols", [])),
            "kernel_signatures": copy.deepcopy(
                node_binding.get("kernel_signatures", [])
            ),
        }


def _validate_parallelism(
    profile: dict[str, Any], plan: dict[str, Any], *, source: Path
) -> None:
    params = profile.get("execution_parameters") or {}
    _require(params, ("tp_size", "dp_size", "cp_size", "ep_size"), source=source)
    axes = plan.get("parallelism_axes") or {}
    for axis, expected in axes.items():
        if isinstance(expected, int) and params.get(axis) != expected:
            raise CatalogError(
                f"{source}: {axis}={params.get(axis)!r} does not match "
                f"execution plan requirement {expected!r}"
            )
    min_tp = (plan.get("constraints") or {}).get("tp_size_min")
    if min_tp is not None and int(params["tp_size"]) < int(min_tp):
        raise CatalogError(f"{source}: tp_size must be >= {min_tp}")


def compile_profile(
    profile: dict[str, Any],
    *,
    plan: dict[str, Any],
    fingerprint: str,
    node_targets: set[str],
    source: Path,
) -> dict[str, Any]:
    _validate_parallelism(profile, plan, source=source)
    unknown = sorted(set(profile.get("node_metrics") or {}) - node_targets)
    if unknown:
        raise CatalogError(f"{source}: profile references unknown nodes: {unknown}")
    for target, metric in (profile.get("node_metrics") or {}).items():
        if not isinstance(metric, dict) or "ms_per_iter" not in metric:
            raise CatalogError(f"{source}: metric {target!r} requires ms_per_iter")
        if float(metric["ms_per_iter"]) < 0:
            raise CatalogError(f"{source}: metric {target!r} has negative ms_per_iter")

    variant = profile["variant_id"]
    data = {
        target: {variant: copy.deepcopy(metric)}
        for target, metric in (profile.get("node_metrics") or {}).items()
    }
    return {
        "meta": {
            key: copy.deepcopy(profile[key])
            for key in (
                "profile_id",
                "label",
                "phase",
                "variant_id",
                "execution_parameters",
                "hardware",
                "workload",
                "profiler",
                "evidence",
            )
            if key in profile
        },
        "execution_variant": fingerprint,
        "execution_path_id": profile["execution_path_id"],
        "implementation_id": profile["implementation_id"],
        "data": data,
    }


def build_enriched(
    views: dict[str, Any], profiles: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    enriched = {
        view_id: {"title": view.get("title", view_id), "nodes_profile": {}}
        for view_id, view in views.items()
    }
    for profile_id, profile in profiles.items():
        for target, variants in profile.get("data", {}).items():
            view_id, node_id = target.split(".", 1)
            enriched[view_id]["nodes_profile"].setdefault(node_id, {})[
                profile_id
            ] = copy.deepcopy(variants)
    return enriched


def _catalog_files(root: Path, pattern: str) -> list[Path]:
    return sorted(path for path in root.glob(pattern) if path.is_file())


def compile_catalog(model_root: Path) -> dict[str, Any]:
    model_root = model_root.resolve()
    model_path = model_root / "model_ir.yaml"
    model_ir = load_yaml(model_path)
    _validate_schema_version(model_ir, "model-ir.v2", source=model_path)
    _require(
        model_ir,
        ("model_id", "model_label", "ir_version", "default_view", "views"),
        source=model_path,
    )
    _node_index(model_ir["views"], source=model_path)
    if model_ir["default_view"] not in model_ir["views"]:
        raise CatalogError(
            f"{model_path}: default_view {model_ir['default_view']!r} does not exist"
        )

    plan_paths = _catalog_files(model_root, "execution_paths/*.yaml")
    if not plan_paths:
        raise CatalogError(f"{model_root}: no execution plans found")

    binding_paths = _catalog_files(model_root, "bindings/*.yaml")
    profile_paths = _catalog_files(model_root, "profiles/*/*/*.yaml")
    raw_bindings: list[tuple[Path, dict[str, Any]]] = []
    for path in binding_paths:
        binding = load_yaml(path)
        _validate_schema_version(binding, "implementation-binding.v2", source=path)
        _require(
            binding,
            (
                "implementation_id",
                "label",
                "model_id",
                "execution_path_id",
                "source_repo",
                "source_commit",
                "node_bindings",
            ),
            source=path,
        )
        if binding["model_id"] != model_ir["model_id"]:
            raise CatalogError(f"{path}: model_id does not match {model_ir['model_id']}")
        raw_bindings.append((path, binding))

    raw_profiles: list[tuple[Path, dict[str, Any]]] = []
    for path in profile_paths:
        profile = load_yaml(path)
        _validate_schema_version(profile, "profile.v2", source=path)
        _require(
            profile,
            (
                "profile_id",
                "label",
                "model_id",
                "execution_path_id",
                "implementation_id",
                "variant_id",
                "execution_parameters",
                "phase",
                "node_metrics",
            ),
            source=path,
        )
        if profile["model_id"] != model_ir["model_id"]:
            raise CatalogError(f"{path}: model_id does not match {model_ir['model_id']}")
        raw_profiles.append((path, profile))

    bindings_by_id: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path, binding in raw_bindings:
        implementation_id = binding["implementation_id"]
        if implementation_id in bindings_by_id:
            raise CatalogError(
                f"{path}: duplicate implementation {implementation_id!r}"
            )
        bindings_by_id[implementation_id] = (path, binding)

    def binding_lineage(
        implementation_id: str, trail: tuple[str, ...] = ()
    ) -> list[tuple[Path, dict[str, Any]]]:
        if implementation_id in trail:
            raise CatalogError(
                "implementation binding inheritance cycle: "
                + " -> ".join((*trail, implementation_id))
            )
        path, binding = bindings_by_id[implementation_id]
        base_id = binding.get("extends")
        if not base_id:
            return [(path, binding)]
        if base_id not in bindings_by_id:
            raise CatalogError(f"{path}: unknown base implementation {base_id!r}")
        base_path, base = bindings_by_id[base_id]
        for identity_key in ("model_id", "source_repo", "source_commit"):
            if str(binding.get(identity_key)) != str(base.get(identity_key)):
                raise CatalogError(
                    f"{path}: inherited {identity_key} must match {base_path}"
                )
        return binding_lineage(base_id, (*trail, implementation_id)) + [
            (path, binding)
        ]

    execution_variants: dict[str, Any] = {}
    plans_by_id: dict[str, tuple[Path, dict[str, Any], str]] = {}
    for plan_path in plan_paths:
        plan = load_yaml(plan_path)
        _validate_schema_version(plan, "execution-plan.v2", source=plan_path)
        _require(
            plan,
            (
                "execution_path_id",
                "label",
                "model_id",
                "plan_version",
                "parallelism_axes",
                "transforms",
            ),
            source=plan_path,
        )
        if plan["model_id"] != model_ir["model_id"]:
            raise CatalogError(f"{plan_path}: model_id does not match {model_ir['model_id']}")
        if plan["execution_path_id"] in plans_by_id:
            raise CatalogError(
                f"{plan_path}: duplicate execution path {plan['execution_path_id']!r}"
            )
        views = apply_execution_plan(model_ir, plan, source=plan_path)
        fingerprint = execution_fingerprint(model_ir, plan, views)
        plans_by_id[plan["execution_path_id"]] = (plan_path, plan, fingerprint)
        execution_variants[fingerprint] = {
            "execution_path_id": plan["execution_path_id"],
            "label": plan["label"],
            "fingerprint": fingerprint,
            "model_ir_version": model_ir["ir_version"],
            "execution_plan_version": plan["plan_version"],
            "parallelism_axes": copy.deepcopy(plan.get("parallelism_axes", {})),
            "default_parameters": copy.deepcopy(plan.get("default_parameters", {})),
            "default_view": model_ir["default_view"],
            "views": views,
            "parent": derive_parent_map(views),
        }

    implementations: dict[str, Any] = {}
    for path, raw_binding in raw_bindings:
        path_id = raw_binding["execution_path_id"]
        if path_id not in plans_by_id:
            raise CatalogError(f"{path}: unknown execution_path_id {path_id!r}")
        _, _, fingerprint = plans_by_id[path_id]
        target_index = _node_index(
            execution_variants[fingerprint]["views"], source=path
        )
        unknown = sorted(set(raw_binding.get("node_bindings") or {}) - set(target_index))
        if unknown:
            raise CatalogError(f"{path}: binding references unknown nodes: {unknown}")
        merged_binding = copy.deepcopy(raw_binding)
        merged_nodes: dict[str, Any] = {}
        for _lineage_path, lineage_binding in binding_lineage(
            raw_binding["implementation_id"]
        ):
            merged_nodes.update(
                {
                    target: copy.deepcopy(node_binding)
                    for target, node_binding in (
                        lineage_binding.get("node_bindings") or {}
                    ).items()
                    if target in target_index
                }
            )
        merged_binding["node_bindings"] = merged_nodes
        compiled = compile_binding(merged_binding, source=path)
        compiled["execution_variant"] = fingerprint
        implementations[compiled["implementation_id"]] = compiled

    profiles: dict[str, Any] = {}
    for path, raw_profile in raw_profiles:
        path_id = raw_profile["execution_path_id"]
        if path_id not in plans_by_id:
            raise CatalogError(f"{path}: unknown execution_path_id {path_id!r}")
        plan_path, plan, fingerprint = plans_by_id[path_id]
        impl_id = raw_profile["implementation_id"]
        if impl_id not in implementations:
            raise CatalogError(f"{path}: unknown implementation_id {impl_id!r}")
        if implementations[impl_id]["execution_variant"] != fingerprint:
            raise CatalogError(
                f"{path}: implementation {impl_id!r} belongs to another execution path"
            )
        evidence_commit = (raw_profile.get("evidence") or {}).get("source_commit")
        implementation_commit = implementations[impl_id].get("source_commit")
        if evidence_commit is not None and str(evidence_commit) != str(implementation_commit):
            raise CatalogError(
                f"{path}: evidence source_commit {evidence_commit!r} does not match "
                f"implementation {impl_id!r} commit {implementation_commit!r}"
            )
        targets = set(
            _node_index(execution_variants[fingerprint]["views"], source=plan_path)
        )
        compiled_profile = compile_profile(
            raw_profile,
            plan=plan,
            fingerprint=fingerprint,
            node_targets=targets,
            source=path,
        )
        profile_id = raw_profile["profile_id"]
        if profile_id in profiles:
            raise CatalogError(f"{path}: duplicate profile_id {profile_id!r}")
        profiles[profile_id] = compiled_profile

    for fingerprint, variant in execution_variants.items():
        compatible_profiles = {
            profile_id: profile
            for profile_id, profile in profiles.items()
            if profile["execution_variant"] == fingerprint
        }
        variant["enriched"] = build_enriched(variant["views"], compatible_profiles)

    default_path_id = model_ir.get("default_execution_path") or next(iter(plans_by_id))
    if default_path_id not in plans_by_id:
        raise CatalogError(
            f"{model_path}: unknown default_execution_path {default_path_id!r}"
        )
    default_fingerprint = plans_by_id[default_path_id][2]
    default_impl = next(
        (
            impl_id
            for impl_id, impl in implementations.items()
            if impl["execution_variant"] == default_fingerprint
        ),
        "",
    )
    default_profile = next(
        (
            profile_id
            for profile_id, profile in profiles.items()
            if profile["execution_variant"] == default_fingerprint
            and (not default_impl or profile["implementation_id"] == default_impl)
        ),
        "",
    )

    default_variant = execution_variants[default_fingerprint]
    projected_views = copy.deepcopy(default_variant["views"])
    if default_impl:
        apply_binding(
            projected_views,
            implementations[default_impl],
            source=model_root / "bindings",
        )

    bundle = {
        "schema_version": "2.0",
        "default_view": model_ir["default_view"],
        "default_execution_variant": default_fingerprint,
        "default_implementation": default_impl,
        "default_profile": default_profile,
        "meta": {
            "model_id": model_ir["model_id"],
            "model_label": model_ir["model_label"],
            "subtitle": "IR-first · execution-path variants · versioned profile overlays",
            "model_ir_version": model_ir["ir_version"],
            "catalog": f"catalog/{model_root.name}",
            "execution_variant_count": len(execution_variants),
            "implementation_count": len(implementations),
            "profile_count": len(profiles),
            "view_count": len(default_variant["views"]),
        },
        "model_ir": {
            key: copy.deepcopy(model_ir[key])
            for key in ("model_id", "model_label", "ir_version", "dimensions", "facts")
            if key in model_ir
        },
        "execution_variants": execution_variants,
        "implementations": implementations,
        "profiles": profiles,
        # Compatibility projection for the existing viewer and downstream readers.
        "views": projected_views,
        "parent": copy.deepcopy(default_variant["parent"]),
        "enriched": copy.deepcopy(default_variant["enriched"]),
        "stages": {},
        "configs": {},
    }
    return bundle


def write_bundle(bundle: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(bundle, indent=2, sort_keys=False) + "\n")
