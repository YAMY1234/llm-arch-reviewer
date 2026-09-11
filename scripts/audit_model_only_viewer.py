#!/usr/bin/env python3
"""Exercise every semantic drill and node using real shared-Viewer DOM clicks.

This is presentation acceptance for an explicitly model-only bundle. It cannot
attest runtime execution, binding, generation quality, or production readiness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlencode

from playwright.sync_api import sync_playwright


def drill_inventory(views: dict, root: str) -> list[dict]:
    """One route per view, plus every declared parent/child drill edge."""
    routes = {root: ([root], [])}
    pending = [root]
    drills = []
    while pending:
        view = pending.pop(0)
        path, origins = routes[view]
        for node in views[view].get("nodes", []):
            child = node.get("drill")
            if not child:
                continue
            if child not in views:
                raise ValueError(f"missing drill {view}.{node['id']} -> {child}")
            if child in path:
                raise ValueError(f"cyclic drill {view}.{node['id']} -> {child}")
            drills.append({"path": path, "origins": origins,
                           "node": node["id"], "child": child})
            if child not in routes:
                routes[child] = (path + [child], origins + [node["id"]])
                pending.append(child)
    unreachable = set(views) - routes.keys()
    if unreachable:
        raise ValueError(f"unreachable semantic views: {sorted(unreachable)}")
    return drills


def audit(bundle_path: Path, base_url: str, output: Path, browser_path: str | None) -> dict:
    raw = bundle_path.read_bytes()
    bundle = json.loads(raw)
    if bundle.get("meta", {}).get("lifecycle") != "model_only":
        raise ValueError("model-only browser audit requires declared model_only lifecycle")
    if any(bundle.get(key) for key in ("profiles", "implementations", "execution_variants")):
        raise ValueError("model_only bundle contains runtime artifacts")
    views = bundle["model_ir"]["views"]
    root = bundle["default_view"]
    drills = drill_inventory(views, root)
    output.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": "model-only-viewer-audit.v1", "status": "fail",
              "production_release_ready": False, "bundle_sha256": hashlib.sha256(raw).hexdigest(),
              "views": [], "drills": [], "nodes": [], "failures": [], "screenshots": []}
    url = base_url.rstrip("/") + "/viewer.html?" + urlencode({
        "model": bundle_path.parent.name, "irLayer": "model", "viewMode": "architecture"})
    report["url"] = url
    routes = {root: ([root], [])}
    for drill in drills:
        routes.setdefault(drill["child"], (drill["path"] + [drill["child"]],
                                           drill["origins"] + [drill["node"]]))
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=browser_path, headless=True)
        page = browser.new_page(viewport={"width": 1920, "height": 1200})
        page.on("pageerror", lambda e: report["failures"].append({"javascript": str(e)}))
        served = page.request.get(base_url.rstrip("/") + f"/{bundle_path.parent.name}/arch_data.json")
        served_sha = hashlib.sha256(served.body()).hexdigest()
        report["served_bundle_sha256"] = served_sha
        if not served.ok or served_sha != report["bundle_sha256"]:
            raise ValueError("HTTP-served bundle differs from the audited local artifact")

        def open_route(path: list[str], origins: list[str]) -> None:
            # Hash is the public deep-link interface, not a private navigation helper.
            page.goto(url + "#views=" + ",".join(path) + "&from=" + ",".join([""] + origins),
                      wait_until="networkidle")
            page.wait_for_selector(f'g.view-group[data-view="{path[-1]}"] g.node')
            page.wait_for_function("expected => JSON.stringify(VIEW_STACK) === JSON.stringify(expected)", arg=path)
            if not report["views"] and page.evaluate("() => RAW_DATA") != bundle:
                raise ValueError("Viewer loaded a different bundle than the audited local artifact")

        for view, (path, origins) in routes.items():
            open_route(path, origins)
            group = page.locator(f'g.view-group[data-view="{view}"]')
            expected_nodes = [n["id"] for n in views[view]["nodes"]]
            actual_nodes = group.locator("g.node").evaluate_all("ns => ns.map(n => n.dataset.id)")
            if sorted(expected_nodes) != sorted(actual_nodes):
                report["failures"].append({"view": view, "inventory": actual_nodes, "expected": expected_nodes})
            geometry = group.evaluate("""group => {
              const issues = [], nodes = [...group.querySelectorAll('g.node')];
              for (const node of nodes) {
                const bg = node.querySelector('.node-bg');
                if (!bg) { issues.push({node:node.dataset.id, missing_background:true}); continue; }
                const box = bg.getBoundingClientRect();
                for (const text of node.querySelectorAll('text')) {
                  const r = text.getBoundingClientRect();
                  if (r.left < box.left-3 || r.right > box.right+3 || r.top < box.top-3 || r.bottom > box.bottom+3)
                    issues.push({node:node.dataset.id, text_overflow:text.textContent});
                }
              }
              for (let i=0; i<nodes.length; i++) for (let j=i+1; j<nodes.length; j++) {
                const a=nodes[i].getBoundingClientRect(), b=nodes[j].getBoundingClientRect();
                if (Math.min(a.right,b.right)-Math.max(a.left,b.left)>2 &&
                    Math.min(a.bottom,b.bottom)-Math.max(a.top,b.top)>2)
                  issues.push({overlapping_nodes:[nodes[i].dataset.id,nodes[j].dataset.id]});
              }
              const labels = [...group.querySelectorAll('g.edge text.edge-label')];
              for (let i=0; i<labels.length; i++) for (let j=i+1; j<labels.length; j++) {
                if (labels[i].parentElement === labels[j].parentElement) continue;
                const a=labels[i].getBoundingClientRect(), b=labels[j].getBoundingClientRect();
                if (Math.min(a.right,b.right)-Math.max(a.left,b.left)>1 &&
                    Math.min(a.bottom,b.bottom)-Math.max(a.top,b.top)>1)
                  issues.push({overlapping_edge_labels:[labels[i].textContent,labels[j].textContent]});
              }
              return issues;
            }""")
            report["failures"].extend({"view": view, "geometry": issue} for issue in geometry)
            for node in views[view]["nodes"]:
                target = group.locator(f'g.node[data-id="{node["id"]}"]')
                target.click(timeout=10000)
                page.wait_for_function("target => SELECTED?.view === target[0] && SELECTED?.nodeId === target[1]",
                                       arg=[view, node["id"]])
                detail = page.locator("#detail")
                text = detail.inner_text()
                required = ["Semantics", "Inputs", "Transition / Equation", "Outputs"]
                missing = [heading for heading in required if heading not in text]
                if missing or "Equation unavailable" in text:
                    report["failures"].append({"view": view, "node": node["id"], "missing": missing})
                equation = node.get("semantics", {}).get("equation", "")
                if not equation or equation not in text:
                    report["failures"].append({"view": view, "node": node["id"], "equation_mismatch": True})
                if node.get("code_links") and not detail.locator("a.codelink[href^='https://']").count():
                    report["failures"].append({"view": view, "node": node["id"], "source_links": "missing"})
                overflow = detail.evaluate("el => ({scroll: el.scrollWidth, client: el.clientWidth})")
                if overflow["scroll"] > overflow["client"] + 2:
                    report["failures"].append({"view": view, "node": node["id"], "detail_overflow": overflow})
                report["nodes"].append(f"{view}.{node['id']}")
            screenshot = f"{len(report['views']):02d}-{view}.png"
            page.screenshot(path=str(output / screenshot))
            report["screenshots"].append(screenshot)
            report["views"].append(view)
            page.reload(wait_until="networkidle")
            page.wait_for_function("expected => JSON.stringify(VIEW_STACK) === JSON.stringify(expected)", arg=path)
            if len(path) > 1:
                page.locator("#crumbs-list .crumb").first.click()
                page.wait_for_function("root => VIEW_STACK.length === 1 && VIEW_STACK[0] === root", arg=root)

        for drill in drills:
            open_route(drill["path"], drill["origins"])
            parent = drill["path"][-1]
            page.locator(f'g.view-group[data-view="{parent}"] g.node[data-id="{drill["node"]}"]').dblclick(timeout=10000)
            expected = drill["path"] + [drill["child"]]
            page.wait_for_function("expected => JSON.stringify(VIEW_STACK) === JSON.stringify(expected)", arg=expected)
            page.wait_for_selector(f'g.view-group[data-view="{drill["child"]}"] g.node')
            report["drills"].append(f"{parent}.{drill['node']} -> {drill['child']}")

        open_route([root], [])
        subtitle = page.locator("#model-subtitle").inner_text()
        if "no profile attached" not in subtitle or not page.locator("#model-subtitle").is_visible():
            report["failures"].append({"subtitle": subtitle})
        controls = page.locator("header select").evaluate_all(
            "els => els.filter(e => e.offsetParent !== null).map(e => ({id:e.id, disabled:e.disabled, title:e.title || e.parentElement.title}))")
        report["controls"] = controls
        for control in controls:
            if not control["disabled"]:
                report["failures"].append({"runtime_control_enabled": control})
            if not control["title"]:
                report["failures"].append({"runtime_control_reason_missing": control})
        browser.close()
    report["status"] = "pass" if not report["failures"] else "fail"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--browser")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        report = audit(args.bundle, args.base_url, args.output, args.browser)
    except Exception as error:
        report = {"status": "fail", "production_release_ready": False,
                  "failures": [{"exception": str(error)}]}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: len(value) if isinstance(value, list) else value
                      for key, value in report.items() if key != "controls"}, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
