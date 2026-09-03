#!/usr/bin/env python3
"""Render and interactively audit every route/profile in one V2 viewer bundle.

This complements schema/compiler tests with a real-browser acceptance pass. It
checks every drill occurrence, every node detail panel, SVG geometry, and both
Architecture -> Timeline and Timeline -> Architecture navigation.  The script
is model-independent; all routes and profiles come from the compiled bundle.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import quote, urlencode

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright


TIMELINE_LOAD_TIMEOUT_MS = 30_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8766")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--browser",
        help="optional browser executable; defaults to Playwright Chromium",
    )
    parser.add_argument("--viewport-width", type=int, default=1800)
    parser.add_argument("--viewport-height", type=int, default=1200)
    parser.add_argument(
        "--device-scale-factor",
        type=float,
        default=1.0,
        help="Emulate a Retina/HiDPI display (for example, 2.0).",
    )
    return parser.parse_args()


def drill_routes(views: dict, root: str) -> list[tuple[list[str], list[str]]]:
    routes: list[tuple[list[str], list[str]]] = []

    def visit(path: list[str], origins: list[str]) -> None:
        routes.append((path, origins))
        current = path[-1]
        for node in views[current].get("nodes") or []:
            child = node.get("drill")
            if not child or child in path:
                continue
            visit(path + [child], origins + [str(node["id"])])

    visit([root], [])
    return routes


def write_exception_report(args: argparse.Namespace, error: Exception) -> int:
    """Persist a fail-closed report when the browser audit raises unexpectedly."""

    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": "viewer-render-audit.v1",
        "bundle": str(args.bundle),
        "model": args.bundle.parent.name,
        "checks": {},
        "route_count_per_profile": {},
        "interaction_performance": [],
        "status": "fail",
        "failures": [
            {
                "kind": "audit_exception",
                "exception_type": type(error).__name__,
                "error": str(error),
            }
        ],
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 1


def main(args: argparse.Namespace | None = None) -> int:
    args = args or parse_args()
    bundle = json.loads(args.bundle.read_text())
    model = args.bundle.parent.name
    route_counts: dict[str, int] = {}
    args.output.mkdir(parents=True, exist_ok=True)
    failures: list[dict] = []
    interaction_performance: list[dict] = []
    checks = {
        "profiles": 0,
        "routes": 0,
        "nodes": 0,
        "cross_links": 0,
        "bidirectional_dom_clicks": 0,
        "runtime_support_details": 0,
        "typed_unresolved_details": 0,
        "double_click_drills": 0,
        "parent_active_share_cards": 0,
        "fusion_owner_card_links": 0,
        "fusion_owner_detail_links": 0,
        "fusion_owner_navigations": 0,
        "fusion_owner_dom_clicks": 0,
        "timeline_interaction_performance": 0,
        "timeline_stream_mode_invariants": 0,
        "compact_lane_expansions": 0,
        "real_timeline_gestures": 0,
        "architecture_only_implementations": 0,
    }
    captured_implementations: set[str] = set()

    def fail(kind: str, **context: object) -> None:
        failures.append({"kind": kind, **context})

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=args.browser,
            headless=True,
        )
        context = browser.new_context(
            viewport={
                "width": args.viewport_width,
                "height": args.viewport_height,
            },
            device_scale_factor=args.device_scale_factor,
        )
        page = context.new_page()
        page_errors: list[str] = []
        page.on("pageerror", lambda error: page_errors.append(str(error)))

        # A fail-closed catalog may intentionally contain validated
        # Architecture bindings but zero accepted timing profiles.  Audit that
        # real rendered state instead of reporting a vacuous zero-check pass.
        # Timeline and bidirectional DOM-click counters remain zero and the
        # report scope states why they are not applicable.
        if not bundle["profiles"]:
            for implementation_id, implementation in bundle["implementations"].items():
                variant_id = implementation["execution_variant"]
                variant = bundle["execution_variants"][variant_id]
                entry_view = variant["default_view"]
                routes = drill_routes(variant["views"], entry_view)
                route_key = f"architecture:{implementation_id}"
                route_counts[route_key] = len(routes)
                checks["architecture_only_implementations"] += 1
                query = urlencode(
                    {
                        "model": model,
                        "execution": variant_id,
                        "implementation": implementation_id,
                        "viewMode": "architecture",
                        "irLayer": "execution",
                        "metric": "active",
                        "audit": "1",
                    }
                )
                url = f"{args.base_url.rstrip('/')}/viewer.html?{query}#views={entry_view}&from="
                page.goto(url, wait_until="domcontentloaded", timeout=60_000)
                page.wait_for_selector("g.view-group g.node", timeout=60_000)
                selection = page.evaluate(
                    """() => ({
                      profile: CURRENT_PROFILE,
                      implementation: CURRENT_IMPLEMENTATION,
                      mode: CURRENT_VIEW_MODE,
                    })"""
                )
                if selection != {
                    "profile": "",
                    "implementation": implementation_id,
                    "mode": "architecture",
                }:
                    fail(
                        "architecture_only_selection_mismatch",
                        expected={
                            "profile": "",
                            "implementation": implementation_id,
                            "mode": "architecture",
                        },
                        actual=selection,
                    )

                detail_issues = page.evaluate(
                    """() => {
                      const issues = [];
                      let checked = 0;
                      for (const [viewName, view] of Object.entries(DATA.views)) {
                        for (const node of (view.nodes || [])) {
                          checked += 1;
                          showDetail(node, viewName);
                          const text = document.getElementById('detail').innerText;
                          for (const heading of ['Semantics', 'Inputs', 'Transition / Equation', 'Outputs']) {
                            if (!text.includes(heading)) issues.push({view: viewName, node: node.id, missing: heading});
                          }
                          const equation = String(node?.semantics?.equation || '');
                          if (!equation.trim() || text.includes('Equation unavailable') ||
                              /\\b(?:undefined|NaN|Infinity)\\b/.test(equation)) {
                            issues.push({view: viewName, node: node.id, invalidEquation: equation});
                          }
                        }
                      }
                      return {checked, issues};
                    }"""
                )
                checks["nodes"] += int(detail_issues["checked"])
                for issue in detail_issues["issues"]:
                    fail("architecture_only_semantic_panel", implementation=implementation_id, **issue)

                for route_index, (path, origins) in enumerate(routes):
                    checks["routes"] += 1
                    render = page.evaluate(
                        """async ({path, origins}) => {
                          VIEW_STACK = path;
                          DRILL_FROM = [null, ...origins];
                          await renderView();
                          await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
                          const issues = [];
                          for (const node of document.querySelectorAll('g.view-group g.node')) {
                            const rect = node.getBoundingClientRect();
                            if (!(rect.width > 0 && rect.height > 0)) {
                              issues.push({node: node.dataset.id, geometry: [rect.x, rect.y, rect.width, rect.height]});
                            }
                          }
                          const invalid = ['NaN', 'undefined', 'Infinity'].filter(token =>
                            new RegExp(`\\\\b${token}\\\\b`).test(document.body.innerText));
                          return {issues, invalid, renderedNodes: document.querySelectorAll('g.view-group g.node').length};
                        }""",
                        {"path": path, "origins": origins},
                    )
                    if not render["renderedNodes"]:
                        fail(
                            "architecture_only_empty_route",
                            implementation=implementation_id,
                            path=path,
                        )
                    for issue in render["issues"]:
                        fail(
                            "architecture_only_geometry",
                            implementation=implementation_id,
                            path=path,
                            **issue,
                        )
                    for token in render["invalid"]:
                        fail(
                            "architecture_only_invalid_presentation_token",
                            implementation=implementation_id,
                            path=path,
                            token=token,
                        )
                    if route_index == 0:
                        page.screenshot(
                            path=str(args.output / f"{implementation_id}-architecture.png"),
                            full_page=True,
                        )

        for profile_id, profile in bundle["profiles"].items():
            variant = bundle["execution_variants"][profile["execution_variant"]]
            entry_view = profile.get("meta", {}).get("entry_view") or variant["default_view"]
            routes = drill_routes(variant["views"], entry_view)
            route_counts[profile_id] = len(routes)
            checks["profiles"] += 1
            query = urlencode(
                {
                    "model": model,
                    "execution": profile["execution_variant"],
                    "implementation": profile["implementation_id"],
                    "profile": profile_id,
                    "phase": profile["meta"]["phase"],
                    "viewMode": "architecture",
                    "irLayer": "execution",
                    "metric": "active",
                    "audit": "1",
                }
            )
            url = f"{args.base_url.rstrip('/')}/viewer.html?{query}#views={entry_view}&from="
            page.goto(url, wait_until="domcontentloaded", timeout=60_000)
            page.wait_for_selector("g.view-group g.node", timeout=60_000)

            selection = page.evaluate(
                """() => ({
                  profile: CURRENT_PROFILE,
                  implementation: CURRENT_IMPLEMENTATION,
                  phase: DATA?.profiles?.[CURRENT_PROFILE]?.meta?.phase,
                })"""
            )
            expected_selection = {
                "profile": profile_id,
                "implementation": profile["implementation_id"],
                "phase": profile["meta"]["phase"],
            }
            if selection != expected_selection:
                fail(
                    "profile_selection_mismatch",
                    expected=expected_selection,
                    actual=selection,
                )

            invalid_tokens = page.evaluate(
                """() => {
                  const text = document.body.innerText;
                  return ['NaN', 'undefined', 'Infinity'].filter(token =>
                    new RegExp(`\\\\b${token}\\\\b`).test(text));
                }"""
            )
            for token in invalid_tokens:
                fail("invalid_presentation_token", profile=profile_id, token=token)

            # Every compiled node must render a complete semantic panel.  This
            # invokes the same showDetail path as a click, without needlessly
            # rerunning ELK layout hundreds of times.
            detail_issues = page.evaluate(
                """() => {
                  const issues = [];
                  let checked = 0;
                  let fusionLinks = 0;
                  for (const [viewName, view] of Object.entries(DATA.views)) {
                    for (const node of (view.nodes || [])) {
                      checked += 1;
                      showDetail(node, viewName);
                      const detail = document.getElementById('detail');
                      const text = detail.innerText;
                      const target = `${viewName}.${node.id}`;
                      const cell = profileCellForTarget(target);
                      if (cell?.status === 'fused') {
                        const owner = fusionOwnerForTarget(target, cell);
                        const architectureOwner = fusionArchitectureOwnerForTarget(target, owner);
                        const link = [...detail.querySelectorAll('[data-fusion-owner]')]
                          .find(candidate => candidate.dataset.fusionOwner === architectureOwner &&
                            candidate.dataset.fusionSource === target);
                        if (!owner || !irTargetExistsForNavigation(architectureOwner) || !link) {
                          issues.push({view: viewName, node: node.id, missing: 'fusion owner detail link', owner, architectureOwner});
                        } else {
                          fusionLinks += 1;
                        }
                      }
                      for (const heading of ['Semantics', 'Inputs', 'Transition / Equation', 'Outputs']) {
                        if (!text.includes(heading)) issues.push({view: viewName, node: node.id, missing: heading});
                      }
                      const equation = String(node?.semantics?.equation || '');
                      if (text.includes('Equation unavailable') || !equation.trim()) {
                        issues.push({view: viewName, node: node.id, missing: 'authored equation'});
                      }
                      // ``None`` is valid in authored Python-style indexing
                      // such as ``x[:, None, :]``.  Reject actual compiler
                      // placeholders and non-finite tokens instead.
                      const equationExemption = String(
                        node?.semantic_details?.equation_exempt_reason || ''
                      ).trim();
                      const requiresPrimitiveEquation =
                        node?.semantics?.kind === 'compute' && !equationExemption;
                      if (equation.trim() === 'None' || equation.includes('None = None(None)') ||
                          /\b(?:undefined|NaN|Infinity)\b/.test(equation) ||
                          (requiresPrimitiveEquation && equation.includes('Composite semantic module'))) {
                        issues.push({view: viewName, node: node.id, invalidEquation: equation});
                      }
                    }
                  }
                  return {checked, fusionLinks, issues};
                }"""
            )
            checks["nodes"] += int(detail_issues["checked"])
            checks["fusion_owner_detail_links"] += int(detail_issues["fusionLinks"])
            for issue in detail_issues["issues"]:
                fail("semantic_panel", profile=profile_id, **issue)

            for route_index, (path, origins) in enumerate(routes):
                checks["routes"] += 1
                render = page.evaluate(
                    """async ({path, origins}) => {
                      VIEW_STACK = path;
                      DRILL_FROM = [null, ...origins];
                      await renderView();
                      await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
                      const issues = [];
                      let fusionLinks = 0;
                      let activeShareCards = 0;
                      for (const group of document.querySelectorAll('g.view-group')) {
                        const view = group.dataset.view;
                        const nodes = [...group.querySelectorAll('g.node')];
                        const rects = nodes.map(node => ({id: node.dataset.id, rect: node.getBoundingClientRect()}));
                        for (const node of nodes) {
                          const target = `${view}.${node.dataset.id}`;
                          const cell = profileCellForTarget(target);
                          const authoredNode = (DATA.views[view]?.nodes || []).find(
                            candidate => candidate.id === node.dataset.id
                          );
                          const expectedShare = authoredNode == null
                            ? null : nodeParentActiveShare(authoredNode, view);
                          const shareRows = [...node.querySelectorAll('text.node-ms')]
                            .filter(text => text.textContent.startsWith('active share '));
                          if (expectedShare == null && shareRows.length) {
                            issues.push({type: 'unexpected_parent_active_share', view, node: node.dataset.id});
                          } else if (expectedShare != null) {
                            const expectedText = activeShareText(expectedShare);
                            if (shareRows.length !== 1 || shareRows[0].textContent !== expectedText) {
                              issues.push({
                                type: 'parent_active_share_mismatch',
                                view,
                                node: node.dataset.id,
                                expected: expectedText,
                                actual: shareRows.map(text => text.textContent),
                              });
                            } else {
                              activeShareCards += 1;
                            }
                          }
                          if (cell?.status === 'fused') {
                            const owner = fusionOwnerForTarget(target, cell);
                            const architectureOwner = fusionArchitectureOwnerForTarget(target, owner);
                            const link = node.querySelector('a.fusion-owner-link');
                            if (!owner || !irTargetExistsForNavigation(architectureOwner) || !link ||
                                link.dataset.fusionOwner !== architectureOwner || link.dataset.fusionSource !== target) {
                              issues.push({type: 'fusion_owner_card_link', view, node: node.dataset.id, owner, architectureOwner});
                            } else {
                              fusionLinks += 1;
                            }
                          }
                          const bg = node.querySelector('.node-bg');
                          if (!bg) { issues.push({type: 'missing_node_background', view, node: node.dataset.id}); continue; }
                          // Compare final rendered screen-space rectangles so
                          // nested transforms (for example the drill glyph's
                          // translated badge) are included on both sides.
                          const box = bg.getBoundingClientRect();
                          for (const text of node.querySelectorAll('text')) {
                            const t = text.getBoundingClientRect();
                            if (t.left < box.left - 3 || t.right > box.right + 3 ||
                                t.top < box.top - 3 || t.bottom > box.bottom + 3) {
                              issues.push({type: 'node_text_overflow', view, node: node.dataset.id, text: text.textContent});
                            }
                          }
                        }
                        for (let i = 0; i < rects.length; i++) for (let j = i + 1; j < rects.length; j++) {
                          const a = rects[i].rect, b = rects[j].rect;
                          const overlapW = Math.min(a.right,b.right) - Math.max(a.left,b.left);
                          const overlapH = Math.min(a.bottom,b.bottom) - Math.max(a.top,b.top);
                          if (overlapW > 2 && overlapH > 2) {
                            issues.push({type: 'node_overlap', view, nodes: [rects[i].id, rects[j].id]});
                          }
                        }
                      }
                      return {issues, groups: document.querySelectorAll('g.view-group').length,
                        fusionLinks, activeShareCards};
                    }""",
                    {"path": path, "origins": origins},
                )
                if int(render["groups"]) != len(path):
                    fail(
                        "route_depth",
                        profile=profile_id,
                        path=path,
                        expected=len(path),
                        actual=render["groups"],
                    )
                checks["fusion_owner_card_links"] += int(render["fusionLinks"])
                checks["parent_active_share_cards"] += int(render["activeShareCards"])
                for issue in render["issues"]:
                    fail("geometry", profile=profile_id, path=path, **issue)

                # Keep one visual artifact per unique route. Numeric labels
                # are additionally covered by the geometry pass on all profiles.
                if profile["implementation_id"] not in captured_implementations:
                    page.screenshot(
                        path=args.output
                        / (
                            f"{profile['implementation_id']}-route-{route_index:02d}-"
                            f"{'-'.join(path)}.png"
                        ),
                        full_page=False,
                    )

                # Exercise every drill edge with a real browser double-click.
                # This catches races where the first click replaces the SVG
                # node before the native dblclick event can reach it.
                if len(path) > 1:
                    parent_path = path[:-1]
                    parent_origins = origins[:-1]
                    parent_view = parent_path[-1]
                    origin_node = origins[-1]
                    page.evaluate(
                        """async ({path, origins}) => {
                          VIEW_STACK = path;
                          DRILL_FROM = [null, ...origins];
                          await renderView();
                          await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
                        }""",
                        {"path": parent_path, "origins": parent_origins},
                    )
                    target = page.locator(
                        f'g.view-group[data-view="{parent_view}"] '
                        f'g.node[data-id="{origin_node}"]'
                    )
                    try:
                        target.dblclick(timeout=5_000)
                        page.wait_for_function(
                            "expected => JSON.stringify(VIEW_STACK) === JSON.stringify(expected)",
                            arg=path,
                            timeout=2_000,
                        )
                        checks["double_click_drills"] += 1
                    except PlaywrightTimeoutError:
                        fail(
                            "double_click_drill",
                            profile=profile_id,
                            parent_path=parent_path,
                            origin_node=origin_node,
                            expected_path=path,
                            actual_path=page.evaluate("() => VIEW_STACK"),
                        )

            # Every compiled fusion group gets one real navigation exercise.
            # The destination is resolved from the owner target, never from
            # the rendered label, and must finish on a visible selected leaf.
            fusion_navigation = page.evaluate(
                """async () => {
                  const results = [];
                  let exercisedDomClick = false;
                  for (const [groupId, group] of Object.entries(currentFusionGroups())) {
                    const architectureOwner = String(group.architecture_owner || group.owner);
                    const source = (group.ir_nodes || []).find(target => {
                      if (target === group.owner) return false;
                      return profileCellForTarget(target)?.status === 'fused';
                    });
                    if (!source) continue;
                    let navigated = false;
                    let domClick = false;
                    if (!exercisedDomClick) {
                      await showTimelineEventInArchitecture({
                        event: {_irNode: source, _irTargets: group.ir_nodes || [], _layerKind: source},
                        preserveViewMode: true,
                      });
                      const [sourceView, sourceNodeId] = String(source).split('.', 2);
                      const link = document.querySelector(
                        `g.view-group[data-view="${CSS.escape(sourceView)}"] ` +
                        `g.node[data-id="${CSS.escape(sourceNodeId)}"] a.fusion-owner-link`
                      );
                      if (link) {
                        exercisedDomClick = true;
                        domClick = true;
                        link.dispatchEvent(new MouseEvent('click', {bubbles: true, cancelable: true}));
                        const deadline = Date.now() + 5000;
                        const [ownerView, ownerNodeId] = architectureOwner.split('.', 2);
                        const ownerIsVisible = () => !!document.querySelector(
                          `g.view-group[data-view="${CSS.escape(ownerView)}"] ` +
                          `g.node[data-id="${CSS.escape(ownerNodeId)}"].selected`
                        );
                        while (Date.now() < deadline && !ownerIsVisible()) {
                          await new Promise(resolve => setTimeout(resolve, 20));
                        }
                        navigated = SELECTED?.view === ownerView &&
                          SELECTED?.nodeId === ownerNodeId && ownerIsVisible();
                      }
                    }
                    if (!domClick) navigated = await showFusionOwnerInArchitecture(group.owner, source);
                    const [view, nodeId] = architectureOwner.split('.', 2);
                    const selected = SELECTED?.view === view && SELECTED?.nodeId === nodeId;
                    const visible = !!document.querySelector(
                      `g.view-group[data-view="${CSS.escape(view)}"] ` +
                      `g.node[data-id="${CSS.escape(nodeId)}"].selected`
                    );
                    results.push({groupId, owner: group.owner, architectureOwner, source, navigated, selected, visible, domClick});
                  }
                  return results;
                }"""
            )
            checks["fusion_owner_navigations"] += len(fusion_navigation)
            checks["fusion_owner_dom_clicks"] += sum(
                1 for result in fusion_navigation if result["domClick"]
            )
            for result in fusion_navigation:
                if not result["navigated"] or not result["selected"] or not result["visible"]:
                    fail("fusion_owner_navigation", profile=profile_id, **result)

            # A profile without a Timeline artifact is still a valid
            # Architecture/profile overlay. Audit every architecture route,
            # but do not invent Timeline acceptance checks for absent evidence.
            if not (profile.get("meta") or {}).get("timeline"):
                captured_implementations.add(profile["implementation_id"])
                continue

            # Verify both directions using actual viewer functions and the
            # loaded production timeline.
            cross = page.evaluate(
                """async timeoutMs => {
                  setViewMode('split', false, true);
                  const deadline = Date.now() + timeoutMs;
                  while (!TIMELINE_DATA && Date.now() < deadline) {
                    await new Promise(resolve => setTimeout(resolve, 50));
                  }
                  if (!TIMELINE_DATA) {
                    return {error: 'timeline_not_loaded', stage: 'cross_link_setup', timeoutMs};
                  }
                  const event = (TIMELINE_DATA.steps || []).flatMap(step => step.events || [])
                    .find(candidate => candidate._irNode);
                  if (!event) return {error: 'no_mapped_timeline_event'};
                  const [view, nodeId] = String(event._irNode).split('.', 2);
                  const route = architectureRouteForEvent(event);
                  highlightNodeOnTimeline(view, nodeId, false);
                  const forward = TIMELINE_IR_TARGET === (fusionGroupForTarget(event._irNode)?.[1]?.owner || event._irNode);
                  TIMELINE_SELECTED_EVENT = event;
                  await showTimelineEventInArchitecture({event, preserveViewMode: true});
                  const reverse = SELECTED?.view === route.view && SELECTED?.nodeId === route.nodeId &&
                    !!document.querySelector(`g.view-group[data-view="${route.view}"] g.node[data-id="${route.nodeId}"].selected`);
                  return {forward, reverse, event: event._irNode, route};
                }""",
                arg=TIMELINE_LOAD_TIMEOUT_MS,
            )
            checks["cross_links"] += 1
            if cross.get("error") or not cross.get("forward") or not cross.get("reverse"):
                fail("cross_link", profile=profile_id, **cross)
            else:
                page.screenshot(
                    path=args.output
                    / f"{profile['implementation_id']}-{profile_id}-split.png",
                    full_page=False,
                )

            # Compact lanes are presentation-only. Exercise both modes and a
            # real compact-label click while proving that the production event
            # set and all timing inputs remain byte-for-byte unchanged.
            stream_modes = page.evaluate(
                """async () => {
                  const step = currentTimelineStep();
                  const eventEvidence = () => JSON.stringify((step.events || []).map(event => [
                    event.start_us, event.duration_us, event.stream_id,
                    event._irNode, event._irTargets,
                  ]));
                  const timingEvidence = () => JSON.stringify([
                    step.duration_us, step.active_gpu_us, step.device_gap_us,
                    step.gpu_residency_us, step.gpu_overlap_us,
                  ]);
                  const beforeEvents = eventEvidence();
                  const beforeTiming = timingEvidence();
                  TIMELINE_STREAM_MODE = 'compact';
                  document.getElementById('timeline-stream-mode').value = 'compact';
                  renderTimeline();
                  const compact = {...TIMELINE_LAST_RENDER_STATS};
                  const compactHit = TIMELINE_TRACK_HIT_RECTS[0];
                  let expandedByClick = false;
                  if (compactHit) {
                    const canvas = document.getElementById('timeline-canvas');
                    const rect = canvas.getBoundingClientRect();
                    const clientX = rect.left + compactHit.x + 8;
                    const clientY = rect.top + compactHit.y + 8;
                    canvas.dispatchEvent(new MouseEvent('mousedown', {
                      button: 0, bubbles: true, cancelable: true, clientX, clientY,
                    }));
                    window.dispatchEvent(new MouseEvent('mouseup', {
                      button: 0, bubbles: true, cancelable: true, clientX, clientY,
                    }));
                    await new Promise(resolve => requestAnimationFrame(resolve));
                    expandedByClick = TIMELINE_STREAM_MODE === 'physical';
                  }
                  if (!expandedByClick) {
                    TIMELINE_STREAM_MODE = 'physical';
                    document.getElementById('timeline-stream-mode').value = 'physical';
                    renderTimeline();
                  }
                  const physical = {...TIMELINE_LAST_RENDER_STATS};
                  const unchanged = beforeEvents === eventEvidence() && beforeTiming === timingEvidence();
                  TIMELINE_STREAM_MODE = 'compact';
                  document.getElementById('timeline-stream-mode').value = 'compact';
                  renderTimeline();
                  return {compact, physical, unchanged, expandedByClick};
                }"""
            )
            checks["timeline_stream_mode_invariants"] += 1
            if stream_modes.get("expandedByClick"):
                checks["compact_lane_expansions"] += 1
            compact_stats = stream_modes.get("compact") or {}
            physical_stats = stream_modes.get("physical") or {}
            if (
                not stream_modes.get("unchanged")
                or compact_stats.get("streamMode") != "compact"
                or physical_stats.get("streamMode") != "physical"
                or physical_stats.get("trackCount")
                != physical_stats.get("physicalStreamCount")
                or compact_stats.get("trackCount")
                != compact_stats.get("compactLaneCount")
                or compact_stats.get("compactLaneCount", 0)
                > compact_stats.get("physicalStreamCount", 0)
                or not stream_modes.get("expandedByClick")
            ):
                fail("timeline_stream_modes", profile=profile_id, **stream_modes)

            # Exercise the user-facing event bindings, not only the helper
            # functions above.  This caught a real merge regression where the
            # navigation helpers survived but the clicked-kernel selection
            # state (one solid slice, all unrelated slices faded) disappeared.
            dom_setup = page.evaluate(
                """async timeoutMs => {
                  setViewMode('split', false, true);
                  const deadline = Date.now() + timeoutMs;
                  while (!TIMELINE_DATA && Date.now() < deadline) {
                    await new Promise(resolve => setTimeout(resolve, 50));
                  }
                  if (!TIMELINE_DATA) {
                    return {error: 'timeline_not_loaded', stage: 'bidirectional_dom_click_setup', timeoutMs};
                  }
                  const event = (TIMELINE_DATA.steps || []).flatMap(step => step.events || [])
                    .find(candidate => candidate._irNode && architectureRouteForEvent(candidate).nodeId);
                  if (!event) return {error: 'no_mapped_timeline_event'};
                  const route = architectureRouteForEvent(event);
                  await showTimelineEventInArchitecture({event, preserveViewMode: true, preserveDetail: true});
                  const node = document.querySelector(
                    `g.view-group[data-view="${CSS.escape(route.view)}"] ` +
                    `g.node[data-id="${CSS.escape(route.nodeId)}"]`
                  );
                  if (!node) return {error: 'architecture_node_not_visible', route};
                  node.dispatchEvent(new MouseEvent('click', {bubbles: true, cancelable: true}));
                  const target = `${route.view}.${route.nodeId}`;
                  const expectedTarget = fusionGroupForTarget(target)?.[1]?.owner || target;
                  const architectureClick = {
                    selected: node.classList.contains('selected'),
                    target: TIMELINE_IR_TARGET,
                    expectedTarget,
                    explicitSelectionCleared: TIMELINE_EXPLICIT_SELECTION == null,
                    highlightedKernels: currentTimelineStep().events.filter(timelineEventIsHighlighted).length,
                  };

                  // Use the real detail-panel button to make a mapped kernel
                  // comfortably clickable even in long decode windows.
                  const centerButton = [...document.querySelectorAll('#detail button')]
                    .find(button => button.innerText.includes('Center nearest measured kernel'));
                  if (!centerButton) return {error: 'center_button_missing', architectureClick};
                  centerButton.click();
                  await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
                  const viewport = document.getElementById('timeline-viewport');
                  const canvas = document.getElementById('timeline-canvas');
                  let candidates = TIMELINE_HIT_RECTS.filter(hit =>
                    hit.event._irNode && hit.width >= 2 &&
                    hit.y + hit.height >= viewport.scrollTop &&
                    hit.y <= viewport.scrollTop + viewport.clientHeight
                  ).sort((left, right) => right.width - left.width);
                  if (!candidates.length && TIMELINE_SELECTED_EVENT) {
                    const selected = TIMELINE_SELECTED_EVENT;
                    const center = Number(selected.start_us) + Number(selected.duration_us) / 2;
                    const stepDuration = Number(currentTimelineStep().duration_us);
                    const span = Math.min(
                      stepDuration,
                      Math.max(Number(selected.duration_us) * 12, 40),
                    );
                    const start = Math.max(0, Math.min(stepDuration - span, center - span / 2));
                    TIMELINE_RANGE = {startUs: start, endUs: start + span};
                    renderTimeline();
                    centerTimelineEventVertically(selected);
                    candidates = TIMELINE_HIT_RECTS.filter(hit =>
                      hit.event._irNode && hit.width >= 2 &&
                      hit.y + hit.height >= viewport.scrollTop &&
                      hit.y <= viewport.scrollTop + viewport.clientHeight
                    ).sort((left, right) => right.width - left.width);
                  }
                  const hit = candidates[0];
                  if (!hit) return {error: 'no_clickable_kernel', architectureClick};
                  window.__viewerAuditTimelineClickEvent = hit.event;
                  const rect = canvas.getBoundingClientRect();
                  return {
                    architectureClick,
                    click: {
                      x: rect.left + hit.x + hit.width / 2,
                      y: rect.top + hit.y + hit.height / 2,
                    },
                  };
                }""",
                arg=TIMELINE_LOAD_TIMEOUT_MS,
            )
            if dom_setup.get("error"):
                fail("bidirectional_dom_setup", profile=profile_id, **dom_setup)
            else:
                page.mouse.click(dom_setup["click"]["x"], dom_setup["click"]["y"])
                try:
                    page.wait_for_function(
                        """() => {
                          const event = window.__viewerAuditTimelineClickEvent;
                          const route = architectureRouteForEvent(event);
                          const node = document.querySelector(
                            `g.view-group[data-view="${CSS.escape(route.view)}"] ` +
                            `g.node[data-id="${CSS.escape(route.nodeId)}"].selected`
                          );
                          return TIMELINE_EXPLICIT_SELECTION?.kind === 'event' &&
                            TIMELINE_EXPLICIT_SELECTION.event === event &&
                            SELECTED?.view === route.view && SELECTED?.nodeId === route.nodeId &&
                            !!node;
                        }""",
                        timeout=3000,
                    )
                except PlaywrightTimeoutError:
                    pass
                dom_result = page.evaluate(
                    """() => {
                      const event = window.__viewerAuditTimelineClickEvent;
                      const route = architectureRouteForEvent(event);
                      const highlighted = currentTimelineStep().events.filter(timelineEventIsHighlighted);
                      const selected = document.querySelector(
                        `g.view-group[data-view="${CSS.escape(route.view)}"] ` +
                        `g.node[data-id="${CSS.escape(route.nodeId)}"].selected`
                      );
                      return {
                        explicitKind: TIMELINE_EXPLICIT_SELECTION?.kind || '',
                        exactKernel: TIMELINE_EXPLICIT_SELECTION?.event === event,
                        highlightedKernels: highlighted.length,
                        architectureSelected: !!selected,
                        selectedState: SELECTED,
                        route,
                      };
                    }"""
                )
                checks["bidirectional_dom_clicks"] += 1
                architecture_click = dom_setup["architectureClick"]
                if (
                    not architecture_click.get("selected")
                    or architecture_click.get("target") != architecture_click.get("expectedTarget")
                    or not architecture_click.get("explicitSelectionCleared")
                    or architecture_click.get("highlightedKernels", 0) < 1
                    or dom_result.get("explicitKind") != "event"
                    or not dom_result.get("exactKernel")
                    or dom_result.get("highlightedKernels") != 1
                    or not dom_result.get("architectureSelected")
                ):
                    fail(
                        "bidirectional_dom_click",
                        profile=profile_id,
                        architectureClick=architecture_click,
                        timelineClick=dom_result,
                    )

            # Runtime/support activity is deliberately outside Model IR, but it
            # must still be explainable.  Exercise the real kernel-detail path
            # and require both the typed class and concrete reason to render;
            # such events must not offer a misleading architecture jump.
            support = page.evaluate(
                """async timeoutMs => {
                  setViewMode('split', false, true);
                  const deadline = Date.now() + timeoutMs;
                  while (!TIMELINE_DATA && Date.now() < deadline) {
                    await new Promise(resolve => setTimeout(resolve, 50));
                  }
                  if (!TIMELINE_DATA) {
                    return {error: 'timeline_not_loaded', stage: 'runtime_support_detail', timeoutMs};
                  }
                  const event = (TIMELINE_DATA.steps || []).flatMap(step => step.events || [])
                    .find(candidate => candidate._supportClass && !candidate._irNode);
                  if (!event) return {absent: true};
                  showTimelineEventDetail(event);
                  const detail = document.getElementById('detail');
                  const text = detail.innerText;
                  return {
                    supportClass: event._supportClass,
                    supportReason: event._supportReason,
                    hasClass: text.includes('runtime/support class') && text.includes(event._supportClass),
                    hasReason: text.includes('outside-IR reason') && text.includes(event._supportReason),
                    hasArchitectureButton: [...detail.querySelectorAll('button')]
                      .some(button => button.innerText.includes('Show in architecture')),
                  };
                }""",
                arg=TIMELINE_LOAD_TIMEOUT_MS,
            )
            if support.get("error"):
                fail("runtime_support_detail", profile=profile_id, **support)
                captured_implementations.add(profile["implementation_id"])
                continue
            if not support.get("absent"):
                checks["runtime_support_details"] += 1
                if (
                    support.get("error")
                    or not support.get("supportClass")
                    or not support.get("supportReason")
                    or not support.get("hasClass")
                    or not support.get("hasReason")
                    or support.get("hasArchitectureButton")
                ):
                    fail("runtime_support_detail", profile=profile_id, **support)

            # A sequence candidate that lacks same-rank/phase physical closure
            # remains navigable, but the real detail panel must expose its
            # typed unresolved state and concrete reason instead of presenting
            # an arbitrary representative stack as high-confidence evidence.
            unresolved = page.evaluate(
                """async timeoutMs => {
                  setViewMode('split', false, true);
                  const deadline = Date.now() + timeoutMs;
                  while (!TIMELINE_DATA && Date.now() < deadline) {
                    await new Promise(resolve => setTimeout(resolve, 50));
                  }
                  if (!TIMELINE_DATA) {
                    return {error: 'timeline_not_loaded', stage: 'typed_unresolved_detail', timeoutMs};
                  }
                  const event = (TIMELINE_DATA.steps || []).flatMap(step => step.events || [])
                    .find(candidate => candidate._reconciliationStatus === 'typed_unresolved' && candidate._irNode);
                  if (!event) return {absent: true};
                  showTimelineEventDetail(event);
                  const detail = document.getElementById('detail');
                  const text = detail.innerText;
                  return {
                    status: event._reconciliationStatus,
                    reason: event._reconciliationReason,
                    confidence: event._confidence,
                    hasStatus: text.includes('reconciliation') && text.includes(event._reconciliationStatus),
                    hasReason: !!event._reconciliationReason && text.includes(event._reconciliationReason),
                    hasNoStack: text.includes('not available for this timing-only/support kernel'),
                    hasArchitectureButton: [...detail.querySelectorAll('button')]
                      .some(button => button.innerText.includes('Show in architecture')),
                  };
                }""",
                arg=TIMELINE_LOAD_TIMEOUT_MS,
            )
            if unresolved.get("error"):
                fail("typed_unresolved_detail", profile=profile_id, **unresolved)
                captured_implementations.add(profile["implementation_id"])
                continue
            if not unresolved.get("absent"):
                checks["typed_unresolved_details"] += 1
                if (
                    unresolved.get("error")
                    or unresolved.get("status") != "typed_unresolved"
                    or unresolved.get("confidence") != "review_required"
                    or not unresolved.get("reason")
                    or not unresolved.get("hasStatus")
                    or not unresolved.get("hasReason")
                    or not unresolved.get("hasNoStack")
                    or not unresolved.get("hasArchitectureButton")
                ):
                    fail("typed_unresolved_detail", profile=profile_id, **unresolved)

            # Burst input must be coalesced to animation frames.  A trackpad
            # can emit hundreds of wheel/move events per second; rebuilding
            # semantic lane ownership for each event makes the UI queue input
            # and feel progressively more sluggish.
            interaction = page.evaluate(
                """async () => {
                  const frames = count => new Promise(resolve => {
                    const next = () => count-- > 0 ? requestAnimationFrame(next) : resolve();
                    requestAnimationFrame(next);
                  });
                  await frames(2);
                  const original = renderTimeline;
                  const samples = [];
                  renderTimeline = function() {
                    const start = performance.now();
                    const result = original();
                    samples.push(performance.now() - start);
                    return result;
                  };
                  const canvas = document.getElementById('timeline-canvas');
                  const rect = canvas.getBoundingClientRect();
                  const clientX = rect.left + rect.width * 0.65;
                  const clientY = rect.top + Math.min(100, rect.height / 2);

                  const wheelStart = performance.now();
                  for (let index = 0; index < 180; index++) {
                    canvas.dispatchEvent(new WheelEvent('wheel', {
                      deltaY: index % 2 ? 5 : -5,
                      ctrlKey: true,
                      bubbles: true,
                      cancelable: true,
                      clientX,
                      clientY,
                    }));
                  }
                  const wheelDispatchMs = performance.now() - wheelStart;
                  await frames(3);
                  const wheelSamples = samples.splice(0);

                  canvas.dispatchEvent(new MouseEvent('mousedown', {
                    button: 0, bubbles: true, cancelable: true, clientX, clientY,
                  }));
                  const dragStart = performance.now();
                  for (let index = 0; index < 180; index++) {
                    window.dispatchEvent(new MouseEvent('mousemove', {
                      bubbles: true,
                      clientX: clientX + (index % 30),
                      clientY,
                    }));
                  }
                  window.dispatchEvent(new MouseEvent('mouseup', {
                    button: 0, bubbles: true, clientX: clientX + 29, clientY,
                  }));
                  const dragDispatchMs = performance.now() - dragStart;
                  await frames(3);
                  const dragSamples = samples.splice(0);
                  renderTimeline = original;

                  const summarize = (dispatchMs, values) => ({
                    dispatchMs,
                    renders: values.length,
                    maxRenderMs: Math.max(0, ...values),
                    totalRenderMs: values.reduce((total, value) => total + value, 0),
                  });
                  return {
                    wheel: summarize(wheelDispatchMs, wheelSamples),
                    drag: summarize(dragDispatchMs, dragSamples),
                  };
                }"""
            )
            checks["timeline_interaction_performance"] += 1
            interaction_performance.append({"profile": profile_id, **interaction})
            for gesture in ("wheel", "drag"):
                metrics = interaction[gesture]
                if (
                    metrics["renders"] > 3
                    or metrics["dispatchMs"] > 100
                    or metrics["maxRenderMs"] > 50
                ):
                    fail(
                        "timeline_interaction_performance",
                        profile=profile_id,
                        gesture=gesture,
                        **metrics,
                    )

            # Use Chrome's real input path as well as synthetic burst events.
            # Plain vertical wheel input must scroll the long stream list.
            # Ctrl/Cmd + wheel zooms time; horizontal wheel input pans time.
            page.evaluate("resetTimelineRange()")
            page.wait_for_function(
                "() => TIMELINE_LAST_GESTURE_RENDER_AT == null || "
                "TIMELINE_LAST_GESTURE_RENDER_AT >= TIMELINE_LAST_GESTURE_AT"
            )
            page.evaluate(
                "document.getElementById('timeline-viewport').scrollTop = 0"
            )
            canvas_box = page.locator("#timeline-canvas").bounding_box()
            if not canvas_box:
                fail("timeline_canvas_missing", profile=profile_id)
            else:
                render_stats = page.evaluate("TIMELINE_LAST_RENDER_STATS")
                if (
                    not render_stats
                    or render_stats["backingPixels"] > 24_100_000
                    or render_stats["effectiveDpr"] > 1.001
                ):
                    fail(
                        "timeline_backing_store_budget",
                        profile=profile_id,
                        render_stats=render_stats,
                    )
                page.mouse.move(
                    canvas_box["x"] + canvas_box["width"] * 0.65,
                    canvas_box["y"] + min(100, canvas_box["height"] / 2),
                )
                real_gestures: dict[str, dict] = {}

                before_scroll = page.evaluate(
                    "document.getElementById('timeline-viewport').scrollTop"
                )
                before_span = page.evaluate(
                    "TIMELINE_RANGE.endUs - TIMELINE_RANGE.startUs"
                )
                scroll_expected = page.evaluate(
                    "document.getElementById('timeline-viewport').scrollHeight > "
                    "document.getElementById('timeline-viewport').clientHeight"
                )
                page.mouse.wheel(0, 400)
                if scroll_expected:
                    page.wait_for_function(
                        "before => document.getElementById('timeline-viewport').scrollTop > before",
                        arg=before_scroll,
                        timeout=1_000,
                    )
                real_gestures["vertical_scroll"] = page.evaluate(
                    "([beforeScroll, beforeSpan, scrollExpected]) => ({"
                    "scrollDelta: document.getElementById('timeline-viewport').scrollTop - beforeScroll, "
                    "scrollExpected, "
                    "rangeUnchanged: Math.abs((TIMELINE_RANGE.endUs - TIMELINE_RANGE.startUs) - beforeSpan) < 1e-6"
                    "})",
                    arg=[before_scroll, before_span, scroll_expected],
                )
                page.evaluate(
                    "document.getElementById('timeline-viewport').scrollTop = 0"
                )

                before = page.evaluate(
                    "TIMELINE_RANGE.endUs - TIMELINE_RANGE.startUs"
                )
                page.keyboard.down("Control")
                page.mouse.wheel(0, -120)
                page.keyboard.up("Control")
                page.wait_for_function(
                    "before => (TIMELINE_RANGE.endUs - TIMELINE_RANGE.startUs) < before "
                    "&& TIMELINE_LAST_GESTURE_RENDER_AT != null",
                    arg=before,
                    timeout=1_000,
                )
                real_gestures["modified_wheel_zoom_in"] = page.evaluate(
                    "() => ({latencyMs: TIMELINE_LAST_GESTURE_RENDER_AT - "
                    "TIMELINE_LAST_GESTURE_AT, range: {...TIMELINE_RANGE}})"
                )

                before = page.evaluate("TIMELINE_RANGE.startUs")
                page.mouse.wheel(120, 0)
                page.wait_for_function(
                    "before => TIMELINE_RANGE.startUs > before "
                    "&& TIMELINE_LAST_GESTURE_RENDER_AT != null",
                    arg=before,
                    timeout=1_000,
                )
                real_gestures["horizontal_pan"] = page.evaluate(
                    "() => ({latencyMs: TIMELINE_LAST_GESTURE_RENDER_AT - "
                    "TIMELINE_LAST_GESTURE_AT, range: {...TIMELINE_RANGE}})"
                )

                interaction_performance[-1]["real_input"] = real_gestures
                for gesture, metrics in real_gestures.items():
                    checks["real_timeline_gestures"] += 1
                    if (
                        metrics.get("rangeUnchanged") is False
                        or (
                            metrics.get("scrollExpected") is True
                            and metrics.get("scrollDelta", 0) <= 0
                        )
                        or metrics.get("latencyMs", 0) > 100
                    ):
                        fail(
                            "real_timeline_gesture_latency",
                            profile=profile_id,
                            gesture=gesture,
                            **metrics,
                        )

            invalid_tokens = page.evaluate(
                """() => {
                  const text = document.body.innerText;
                  return ['NaN', 'undefined', 'Infinity'].filter(token =>
                    new RegExp(`\\\\b${token}\\\\b`).test(text));
                }"""
            )
            for token in invalid_tokens:
                fail(
                    "invalid_split_presentation_token",
                    profile=profile_id,
                    token=token,
                )
            captured_implementations.add(profile["implementation_id"])

        context.close()
        browser.close()
        for error in page_errors:
            fail("page_error", error=error)

    report = {
        "schema_version": "viewer-render-audit.v1",
        "bundle": str(args.bundle),
        "model": model,
        "checks": checks,
        "route_count_per_profile": route_counts,
        "interaction_performance": interaction_performance,
        "scope": (
            "accepted_profiles"
            if bundle["profiles"]
            else "architecture_only_no_accepted_profiles"
        ),
        "status": "pass" if not failures else "fail",
        "failures": failures,
    }
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    parsed_args = parse_args()
    try:
        exit_code = main(parsed_args)
    except Exception as error:
        exit_code = write_exception_report(parsed_args, error)
    raise SystemExit(exit_code)
