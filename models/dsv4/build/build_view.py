#!/usr/bin/env python3
"""build/build_view.py — IR + profiles → out/arch_data.json

新版 IR 是 view-based: 每个 view 是独立的 dataflow 子图.
profile enrich 改成: 每个 view 内, 每个 node 按 stage_keys × variant 汇总 ms.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    print("pip install pyyaml", file=sys.stderr)
    raise

ROOT = Path(__file__).resolve().parent.parent          # models/<m>/
IR = ROOT / "ir"
PROFILES = IR / "profiles"

# 模型 id = models/<m> 的 m
MODEL_ID = ROOT.name

# 输出: 优先写到 repo 根 docs/<MODEL_ID>/arch_data.json (供 GH Pages 使用),
# fallback 到 ROOT/out/arch_data.json (本地开发时仍可用).
REPO_ROOT_CANDIDATE = ROOT.parent.parent               # ROOT/../../ = repo 根
DOCS_DIR = REPO_ROOT_CANDIDATE / "docs" / MODEL_ID
LOCAL_OUT = ROOT / "out"

# 模型元数据 (header 显示用); 想本地化进每个 model 自己的 ir/meta.yaml 也行,
# 这里先简单 hardcode dsv4 的, 后续可改成读 ir/meta.yaml.
MODEL_META = {
    "dsv4": {
        "model_label": "DeepSeek-V4",
        "subtitle": "ELK + SVG · drill-down · sglang trace overlay",
    },
}


def load_yaml(p: Path) -> Any:
    with p.open() as fh:
        return yaml.safe_load(fh)


def load_profiles() -> dict[str, dict[str, Any]]:
    """加载 profile yaml.

    NOTE: 当前 trace-kernel-learning 出的 CSV 把 HCA / CSA 命名标反了 (历史 bug):
      - 它说的 'HCA' = 有 indexer 的那种 layer = 实际上是 CSA (compress_ratio=4)
      - 它说的 'CSA' = 无 indexer 的那种 layer = 实际上是 HCA (compress_ratio=128)
    我们在加载阶段直接 swap, 这样下游 IR / 视图层完全按代码事实走.
    源 CSV 还没修, 一旦 trace-kernel-learning 那边修了, 把这里 swap 关掉.
    """
    profs: dict[str, dict[str, Any]] = {}
    for p in sorted(PROFILES.glob("*.yaml")):
        d = load_yaml(p)
        if d.get("meta", {}).get("source") == "trace":
            d = _swap_layer_types(d)
            d.setdefault("meta", {})["hca_csa_swapped_at_load"] = True
        profs[p.stem] = d
    return profs


def _swap_layer_types(profile: dict) -> dict:
    """In-place swap HCA <-> CSA in profile['data'][stage]."""
    data = profile.get("data", {}) or {}
    for stage_id, cell in data.items():
        if not isinstance(cell, dict):
            continue
        h = cell.get("HCA")
        c = cell.get("CSA")
        if h is not None and c is not None:
            cell["HCA"], cell["CSA"] = c, h
        elif h is not None:
            cell["CSA"] = cell.pop("HCA")
        elif c is not None:
            cell["HCA"] = cell.pop("CSA")
    return profile


def lookup_stage_ms(stage_id: str, layer_type: str,
                    stages_by_id: dict, profile_data: dict) -> float | None:
    """同一 stage 可能有多个 trace_aliases, 全部相加. ALL fallback."""
    stage = stages_by_id.get(stage_id)
    if not stage:
        return None
    candidates = list(stage.get("trace_aliases", []) or []) + [stage.get("pdf_name") or stage_id]
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


def collect_node_kernels(node: dict, layer_type: str,
                         stages_by_id: dict, profile_data: dict) -> list[dict]:
    """把这个 node 关联 stage 的 kernels list 取出来 (top-N)."""
    sks = node.get("stage_keys") or []
    out = []
    for sk in sks:
        stage = stages_by_id.get(sk)
        if not stage:
            continue
        for al in stage.get("trace_aliases", []) or []:
            cell = profile_data.get(al)
            if not cell:
                continue
            cell_lt = cell.get(layer_type) or cell.get("ALL")
            if not cell_lt:
                continue
            for k in cell_lt.get("kernels", []) or []:
                out.append({**k, "stage_alias": al})
    out.sort(key=lambda k: -k.get("total_us", 0))
    return out[:6]


def _collect_leaf_stage_keys(view_name: str, views: dict,
                             memo: dict[str, set[str]]) -> set[str]:
    """递归: 把这张 view 里所有 leaf 节点的 stage_keys 全部并起来.
    若 view 里有 drill 节点, 也展开它的 drill view (避免遗漏 mqa_block 这种)."""
    if view_name in memo:
        return memo[view_name]
    memo[view_name] = set()  # 防循环
    out: set[str] = set()
    view = views.get(view_name)
    if not view or "same_as" in view:
        return out
    for n in view.get("nodes", []):
        for sk in (n.get("stage_keys") or []):
            out.add(sk)
        d = n.get("drill")
        if d and d != view_name:
            out |= _collect_leaf_stage_keys(d, views, memo)
    memo[view_name] = out
    return out


def _node_ms_recursive(node: dict, view_name: str, lt: str, views: dict,
                       stages_by_id: dict, pdata: dict,
                       memo: dict[str, set[str]]) -> tuple[float, bool]:
    """单个 node 的 ms: 优先用 stage_keys; 否则递归取它 drill view 的 union."""
    sks = set(node.get("stage_keys") or [])
    drill = node.get("drill")
    if drill and drill in views:
        sks |= _collect_leaf_stage_keys(drill, views, memo)
    if not sks:
        return 0.0, False
    ms = 0.0
    hit = False
    for sk in sks:
        v = lookup_stage_ms(sk, lt, stages_by_id, pdata)
        if v is not None:
            ms += v
            hit = True
    return ms, hit


def aggregate_drill_blocks(views: dict, enriched: dict, stages_by_id: dict,
                           profiles: dict[str, dict]) -> None:
    """对每个有 drill 的 block 节点 (e.g. csa_mqa_block → drill: csa_mqa),
    递归收集 drill view 内所有 leaf stage_keys, 按 stage 唯一性求 ms,
    生成 aggregate (ms_per_iter + top kernels + children breakdown) 挂到
    enriched[parent_view][block_id].

    children 包含 leaf 和 module 子节点 (module 子节点的 ms 通过递归算出).
    1-stage → N-node 时, ms 在 leaf 间平摊以避免 sum > 100%.
    """
    leaf_memo: dict[str, set[str]] = {}

    for view_name, view in views.items():
        if "same_as" in view:
            continue
        for n in view.get("nodes", []):
            drill = n.get("drill")
            if not drill or drill not in views:
                continue
            target = views[drill]
            if "same_as" in target:
                continue

            # union: leaf stage_keys (递归) + block 自身 stage_keys
            child_sk = set(_collect_leaf_stage_keys(drill, views, leaf_memo))
            for sk in (n.get("stage_keys") or []):
                child_sk.add(sk)
            if not child_sk:
                continue

            agg_per_profile: dict[str, dict] = {}
            for pid, prof in profiles.items():
                pdata = prof.get("data", {})
                per_lt: dict[str, dict] = {}
                for lt in ("HCA", "CSA", "ALL"):
                    # block aggregate ms = union 所有 stage 唯一计数
                    ms = 0.0
                    hit = False
                    kernels_acc: list[dict] = []
                    for sk in sorted(child_sk):
                        v = lookup_stage_ms(sk, lt, stages_by_id, pdata)
                        if v is not None:
                            ms += v
                            hit = True
                        stage = stages_by_id.get(sk) or {}
                        for al in stage.get("trace_aliases", []) or []:
                            cell = pdata.get(al)
                            if not cell:
                                continue
                            cell_lt = cell.get(lt) or cell.get("ALL")
                            if not cell_lt:
                                continue
                            for k in cell_lt.get("kernels", []) or []:
                                kernels_acc.append({**k, "stage_alias": al})
                    if not hit:
                        continue
                    kernels_acc.sort(key=lambda k: -k.get("total_us", 0))

                    # children breakdown: drill view 里每个子节点 (leaf + module)
                    # leaf: 用自己 stage_keys 求 ms (1-stage→N-node 平摊)
                    # module: 递归 ms (它的 drill 子 view union)
                    sk_leaf_count: dict[str, int] = {}
                    for cn in target.get("nodes", []):
                        if cn.get("drill"):
                            continue   # module 不参与 leaf 平摊
                        for sk in (cn.get("stage_keys") or []):
                            sk_leaf_count[sk] = sk_leaf_count.get(sk, 0) + 1

                    children = []
                    for cn in target.get("nodes", []):
                        cd = cn.get("drill")
                        if cd and cd in views:
                            cms, chit = _node_ms_recursive(
                                cn, drill, lt, views, stages_by_id, pdata, leaf_memo)
                            kind = "module"
                        else:
                            csks = cn.get("stage_keys") or []
                            if not csks:
                                continue
                            cms = 0.0; chit = False
                            for sk in csks:
                                v = lookup_stage_ms(sk, lt, stages_by_id, pdata)
                                if v is not None:
                                    n_share = max(1, sk_leaf_count.get(sk, 1))
                                    cms += v / n_share
                                    chit = True
                            kind = "leaf"
                        if chit:
                            children.append({
                                "id": cn["id"],
                                "label": (cn.get("label") or cn["id"]).replace("\n", " "),
                                "ms": round(cms, 3),
                                "share_pct": round(100*cms/ms, 1) if ms > 0 else 0,
                                "kind": kind,
                            })
                    children.sort(key=lambda c: -c["ms"])
                    per_lt[lt] = {
                        "ms_per_iter": round(ms, 3),
                        "kernels": kernels_acc[:10],
                        "children": children[:10],
                    }
                if per_lt:
                    agg_per_profile[pid] = per_lt

            if not agg_per_profile:
                continue

            ne = enriched.setdefault(view_name, {"title": view.get("title", view_name),
                                                  "nodes_profile": {}})
            np_dict = ne.setdefault("nodes_profile", {})
            slot = np_dict.setdefault(n["id"], {})
            for pid, per_lt in agg_per_profile.items():
                # 不覆盖原有 nodes_profile (block 自己有 stage 时), 而是合并:
                #   - 若已有 ms_per_iter, 取 max(原来, aggregate)
                #   - aggregate 多出 children, kernels 字段
                cur_pid = slot.setdefault(pid, {})
                for lt, agg in per_lt.items():
                    cur_lt = cur_pid.get(lt)
                    if cur_lt:
                        if agg["ms_per_iter"] > cur_lt.get("ms_per_iter", 0):
                            cur_lt["ms_per_iter"] = agg["ms_per_iter"]
                        # 用 aggregate 的 kernels 替换 (更全)
                        cur_lt["kernels"] = agg["kernels"]
                        cur_lt["children"] = agg["children"]
                        cur_lt["from_aggregate"] = True
                    else:
                        cur_pid[lt] = {**agg, "from_aggregate": True}


def enrich_views(views: dict, stages_by_id: dict,
                 profiles: dict[str, dict]) -> dict:
    """对每张 view 的每个 node, 在每份 profile × 每个 layer_type 下挂 ms + kernels."""
    enriched: dict[str, dict] = {}
    for view_name, view in views.items():
        if "same_as" in view:
            enriched[view_name] = {"same_as": view["same_as"], "title": view.get("title", "")}
            continue
        node_data: dict[str, dict] = {}
        for n in view.get("nodes", []):
            nid = n["id"]
            sks = n.get("stage_keys") or []
            if not sks:
                continue
            per_profile: dict[str, dict] = {}
            for pid, prof in profiles.items():
                pdata = prof.get("data", {})
                per_lt: dict[str, dict] = {}
                for lt in ("HCA", "CSA", "ALL"):
                    ms = 0.0
                    hit = False
                    for sk in sks:
                        v = lookup_stage_ms(sk, lt, stages_by_id, pdata)
                        if v is not None:
                            ms += v
                            hit = True
                    if hit:
                        per_lt[lt] = {
                            "ms_per_iter": round(ms, 3),
                            "kernels": collect_node_kernels(n, lt, stages_by_id, pdata),
                        }
                if per_lt:
                    per_profile[pid] = per_lt
            if per_profile:
                node_data[nid] = per_profile
        enriched[view_name] = {
            "title": view.get("title", view_name),
            "nodes_profile": node_data,
        }
    return enriched


def transform_code_links(arch: dict, source_map_doc: dict) -> dict:
    """对每个 node.code_links: list[str], 在原 string 旁边附加 url 字段,
    生成新的 list[dict]: [{raw, file, line, url, display}].

    本地相对路径 e.g. "models/deepseek_v4.py:524 Compressor (...)" →
      file = "models/deepseek_v4.py"
      line = 524
      url  = "https://github.com/sgl-project/sglang/blob/<commit>/python/sglang/srt/models/deepseek_v4.py#L524"
    """
    rules = source_map_doc.get("source_map", []) or []
    display_cfg = source_map_doc.get("display", {}) or {}
    shorten = display_cfg.get("shorten_paths", True)

    def to_dict(raw: str) -> dict:
        # 解析多种格式:
        #   "models/foo.py:123 symbol"       → line=123, line_end=None
        #   "models/foo.py:123-456 symbol"   → line=123, line_end=456 (区间)
        #   "models/foo.py:my_symbol"        → line=None, sym=my_symbol (跳文件首)
        #   "models/foo.py"                  → line=None, sym="" (跳文件首)
        import re
        m = re.match(r"^(\S+\.[A-Za-z0-9]+)(?::([A-Za-z0-9_\-]+))?(\s+.*)?$", raw)
        if not m:
            return {"raw": raw, "file": None, "line": None, "url": None, "display": raw}
        file, ref, sym = m.group(1), m.group(2), (m.group(3) or "").strip()
        line_i = None
        line_end = None
        if ref:
            # 区分纯数字 vs 数字-数字 vs symbol
            mr = re.match(r"^(\d+)-(\d+)$", ref)
            if mr:
                line_i = int(mr.group(1))
                line_end = int(mr.group(2))
            else:
                mn = re.match(r"^(\d+)$", ref)
                if mn:
                    line_i = int(mn.group(1))
                else:
                    # symbol 形式: 把它并到 sym
                    sym = (ref + (" " + sym if sym else "")).strip()

        # 配 source_map rule
        url = None
        for rule in rules:
            pre = rule.get("prefix", "")
            if pre and not file.startswith(pre):
                continue
            tail = file[len(pre):] if pre else file
            full = rule.get("path_prefix", "") + tail
            base = f"https://github.com/{rule['repo']}/blob/{rule['commit']}/{full}"
            if line_i and line_end:
                base += f"#L{line_i}-L{line_end}"
            elif line_i:
                base += f"#L{line_i}"
            url = base
            break

        # 显示文本: 短文件名 + line[ -line_end] + symbol
        if shorten:
            short_file = file.split("/")[-1]
            if line_i and line_end:
                disp = f"{short_file}:{line_i}-{line_end}"
            elif line_i:
                disp = f"{short_file}:{line_i}"
            else:
                disp = short_file
            if sym:
                disp += " — " + sym
        else:
            disp = raw
        return {"raw": raw, "file": file, "line": line_i, "line_end": line_end,
                "url": url, "display": disp}

    for vname, view in arch.get("views", {}).items():
        if "same_as" in view:
            continue
        for n in view.get("nodes", []):
            cls = n.get("code_links")
            if not cls:
                continue
            n["code_links"] = [to_dict(c) if isinstance(c, str) else c for c in cls]
    return arch


def main() -> int:
    arch = load_yaml(IR / "arch.dsv4.yaml")
    stages_doc = load_yaml(IR / "stages.yaml")
    cfg_pro = load_yaml(IR / "config.v4_pro.yaml")
    cfg_flash = load_yaml(IR / "config.v4_flash.yaml")
    profiles = load_profiles()
    source_map_path = IR / "source_map.yaml"
    source_map_doc = load_yaml(source_map_path) if source_map_path.exists() else {}
    if source_map_doc:
        arch = transform_code_links(arch, source_map_doc)

    stages_by_id = {s["id"]: s for s in stages_doc["stages"]}
    views = arch.get("views", {})

    # resolve same_as: 把 hca_moe -> csa_moe 这种引用展开成真正的 nodes/edges
    for vname, view in list(views.items()):
        if "same_as" in view:
            target = view["same_as"]
            if target in views and "same_as" not in views[target]:
                views[vname] = {
                    "title": view.get("title", views[target].get("title", target)),
                    "nodes": views[target]["nodes"],
                    "edges": views[target]["edges"],
                    "alias_of": target,
                }

    enriched = enrich_views(views, stages_by_id, profiles)
    aggregate_drill_blocks(views, enriched, stages_by_id, profiles)

    # 派生 parent: child_view → parent_view 通过扫所有 view 里有 drill 的 node
    parent_map: dict[str, str] = {}
    for vname, view in views.items():
        for n in view.get("nodes", []):
            d = n.get("drill")
            if d and d not in parent_map:
                parent_map[d] = vname

    meta = dict(MODEL_META.get(MODEL_ID, {}))
    meta.update({
        "model_id": MODEL_ID,
        "build": "models/{}/build/build_view.py".format(MODEL_ID),
        "ir_files": sorted([p.name for p in IR.glob("*.yaml")]),
        "profile_count": len(profiles),
        "view_count": len(views),
    })

    bundle = {
        "schema_version": "0.3",
        "meta": meta,
        "parent": parent_map,
        "views": views,
        "stages": stages_by_id,
        "configs": {"v4_pro": cfg_pro, "v4_flash": cfg_flash},
        "profiles": profiles,
        "enriched": enriched,
        "default_view": "top",
    }

    out_targets: list[Path] = []
    # 写到 repo docs/<m>/arch_data.json (GH Pages)
    if (REPO_ROOT_CANDIDATE / "docs").exists():
        DOCS_DIR.mkdir(parents=True, exist_ok=True)
        out_targets.append(DOCS_DIR / "arch_data.json")
    # 也写一份到本地 out/ 方便开发
    LOCAL_OUT.mkdir(exist_ok=True)
    out_targets.append(LOCAL_OUT / "arch_data.json")

    for out_path in out_targets:
        with out_path.open("w") as fh:
            json.dump(bundle, fh, indent=2, ensure_ascii=False)
        print(f"wrote {out_path}")

    print(f"  model: {MODEL_ID}, views: {len(views)}")
    for vn, v in views.items():
        nn = len(v.get("nodes", []))
        en = len(v.get("edges", []))
        en_d = len(enriched[vn].get("nodes_profile", {}))
        print(f"    {vn:18s}  nodes={nn:3d}  edges={en:3d}  enriched={en_d}")
    print(f"  profiles: {len(profiles)}, stages: {len(stages_by_id)}, parent: {len(parent_map)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
