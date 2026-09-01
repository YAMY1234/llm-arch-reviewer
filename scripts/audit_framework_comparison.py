#!/usr/bin/env python3
"""Real-browser acceptance audit for generic multi-framework comparison.

The fixtures deliberately cover a shared Execution IR (Qwen3.5), distinct
Execution IR fingerprints for one exact workload (GLM-5.2), an unavailable
exact match, and a synthetic three-framework UI state.  No viewer behavior is
special-cased for these model ids; they are acceptance data only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlencode

from playwright.sync_api import sync_playwright


QWEN_SGLANG = "sglang_f609d677b_qwen35_033446bb_tp8"
QWEN_VLLM = "vllm_487ecf187_qwen35_native_tp8"
QWEN_PROFILE = "qwen35_tp8_sglang_cg_decode_bs64_8k1k"
GLM53_SGLANG = "sglang_f609d677b_mixed_glm53_tp8"
GLM53_VLLM = "vllm_487ecf187_native_tp8"
GLM53_PROFILE = "glm53_flash_tp8_sglang_cg_decode_bs64_8k1k"
GLM_SGLANG = "sglang_fdebc938_dsa"
GLM_TRT = "trtllm_4358fb5d_dsa"
GLM_PROFILE = "glm52_tp8_sglang_cg_decode_bs64_8k1k"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8766")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--browser", help="optional Chromium/Chrome executable")
    return parser.parse_args()


def viewer_url(base_url: str, **query: str) -> str:
    return f"{base_url.rstrip('/')}/viewer.html?{urlencode(query)}"


def comparison_frames(page, expected: int):
    page.wait_for_function(
        "count => comparisonContextList().length === count",
        arg=expected,
        timeout=60_000,
    )
    page.wait_for_function(
        "count => document.querySelectorAll('#timeline-comparison iframe').length === count",
        arg=expected,
        timeout=60_000,
    )
    frames = []
    for index in range(expected):
        frame = page.locator("#timeline-comparison iframe").nth(index).element_handle().content_frame()
        frame.wait_for_function(
            'typeof TIMELINE_DATA === "object" && TIMELINE_DATA !== null',
            timeout=60_000,
        )
        frames.append(frame)
    return frames


def comparison_detail_frames(page, expected: int):
    page.wait_for_function(
        "count => document.querySelectorAll('#comparison-details iframe.comparison-detail-frame').length === count",
        arg=expected,
        timeout=60_000,
    )
    frames = []
    for index in range(expected):
        frame = page.locator(
            "#comparison-details iframe.comparison-detail-frame"
        ).nth(index).element_handle().content_frame()
        frame.wait_for_function(
            "RAW_DATA && document.body.classList.contains('embedded-detail')",
            timeout=60_000,
        )
        frames.append(frame)
    return frames


def main(args: argparse.Namespace | None = None) -> int:
    args = args or parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, object]] = []
    checks: dict[str, object] = {}

    def require(condition: bool, kind: str, **context: object) -> None:
        if not condition:
            failures.append({"kind": kind, **context})

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=args.browser,
            headless=True,
        )
        context = browser.new_context(viewport={"width": 1800, "height": 1100})
        page = context.new_page()
        page_errors: list[str] = []
        page.on("pageerror", lambda error: page_errors.append(str(error)))

        # 1/2 selection, fixed order, shared Execution IR, URL restore, and
        # both directions of the real rendered Architecture/Timeline link.
        page.goto(
            viewer_url(
                args.base_url,
                model="qwen35_v2",
                profile=QWEN_PROFILE,
                implementation=QWEN_SGLANG,
                implementations=f"{QWEN_VLLM},{QWEN_SGLANG}",
                viewMode="split",
                irLayer="model",
                metric="active",
            ),
            wait_until="domcontentloaded",
            timeout=60_000,
        )
        frames = comparison_frames(page, 2)
        detail_frames = comparison_detail_frames(page, 2)
        state = page.evaluate(
            """() => ({
              selected: SELECTED_IMPLEMENTATIONS,
              compatible: comparisonExecutionIrCompatible(),
              mode: CURRENT_VIEW_MODE,
              contract: CURRENT_COMPARISON_CONTRACT,
              cards: document.querySelectorAll('.timeline-comparison-card').length,
            })"""
        )
        require(state["selected"] == [QWEN_SGLANG, QWEN_VLLM], "fixed_framework_order", actual=state)
        require(state["compatible"] is True, "shared_execution_ir", actual=state)
        require(state["mode"] == "split", "view_mode_restore", actual=state)
        require(state["cards"] == 2, "two_timeline_cards", actual=state)
        require(
            page.locator(".comparison-detail-card").count() == 2,
            "two_framework_detail_panes",
        )
        checks["comparison_dual_details"] = len(detail_frames)

        # Symbol resolution is profile- and stage-aware, while the standalone
        # exporter reuses this exact viewer and embedded compiled contract.
        symbol_resolution = page.evaluate(
            """() => ({
              batch: resolveDimensionSymbol('B', 'top.decoder_stack'),
              hidden: resolveDimensionSymbol('H', 'top.decoder_stack'),
              token: resolveDimensionSymbol('T', 'top.decoder_stack'),
              shape: resolveSymbolicShape('[B,H]', 'top.decoder_stack'),
            })"""
        )
        require(
            symbol_resolution["batch"].get("resolved") is True
            and symbol_resolution["batch"].get("value") == 64
            and symbol_resolution["hidden"].get("value") == 4096
            and symbol_resolution["token"].get("resolved") is False
            and symbol_resolution["shape"].get("resolved") == "[64,4096]",
            "profile_symbol_resolution",
            actual=symbol_resolution,
        )
        checks["tensor_symbol_resolution"] = 1

        # Profile selection must update the resolver in-place. This catches a
        # stale glossary/detail cache that would otherwise keep showing the
        # previous batch after the user changes the selected profile.
        profile_refresh = page.evaluate(
            """async () => {
              const select = document.getElementById('profile-select');
              const switchTo = async (profileId) => {
                select.value = profileId;
                select.dispatchEvent(new Event('change', {bubbles: true}));
                for (let i = 0; i < 200 && CURRENT_PROFILE !== profileId; i++) {
                  await new Promise(resolve => setTimeout(resolve, 10));
                }
                return resolveDimensionSymbol('B', 'top.decoder_stack');
              };
              const bs1 = await switchTo('qwen35_tp8_sglang_cg_decode_bs1_8k1k');
              const bs64 = await switchTo('qwen35_tp8_sglang_cg_decode_bs64_8k1k');
              return {bs1, bs64, current: CURRENT_PROFILE};
            }"""
        )
        require(
            profile_refresh["bs1"].get("value") == 1
            and profile_refresh["bs64"].get("value") == 64
            and profile_refresh["current"] == QWEN_PROFILE,
            "profile_symbol_refresh",
            actual=profile_refresh,
        )
        checks["profile_symbol_refresh"] = 2

        # A drillable GDN parent must expose each framework's descendant-event
        # union. Reading the raw structural authoring state here would hide
        # real measured work even though the child view is fully attributed.
        rollup = page.evaluate(
            """async () => {
              VIEW_STACK = ['top', 'stack', 'gdn_moe_block'];
              DRILL_FROM = [null, 'decoder_stack', 'gdn_layer'];
              SELECTED = {view: 'gdn_moe_block', nodeId: 'attention'};
              await renderView();
              await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
              const rows = comparisonContextList().map(context => {
                const cell = comparisonCellForTarget(context, 'gdn_moe_block.attention');
                return {
                  framework: context.framework,
                  status: cell?.status || '',
                  attribution: cell?.attribution_status || '',
                  active: Number(cell?.active_gpu_ms || 0),
                  sources: cell?.rollup_sources || [],
                };
              });
              const text = document.querySelector(
                'g.view-group[data-view="gdn_moe_block"] g.node[data-id="attention"]'
              )?.textContent || '';
              return {rows, text};
            }"""
        )
        expected_gdn_sources = {
            "gdn_attention.causal_conv",
            "gdn_attention.gated_delta_recurrence",
            "gdn_attention.output_gate_norm",
            "gdn_attention.output_projection",
            "gdn_attention.qkvz_projection",
        }
        require(
            len(rollup["rows"]) == 2
            and all(
                row["attribution"] == "inclusive_rollup"
                and row["active"] > 0
                and set(row["sources"]) == expected_gdn_sources
                for row in rollup["rows"]
            )
            and rollup["text"].count("∪ active") == 2,
            "comparison_parent_union_timing",
            actual=rollup,
        )
        checks["comparison_parent_union_timing"] = 2

        # Framework comparison must retain the canonical fusion navigation
        # affordance in the SVG, not downgrade "fused into" to plain text.
        page.evaluate(
            """async () => {
              VIEW_STACK = ['top', 'stack', 'gdn_moe_block', 'gdn_attention'];
              DRILL_FROM = [null, 'decoder_stack', 'gdn_layer', 'attention'];
              SELECTED = {view: 'gdn_attention', nodeId: 'ba_projection'};
              await renderView();
              await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
            }"""
        )
        fusion_links = page.locator(
            '#svg-host g.view-group[data-view="gdn_attention"] '
            'g.node[data-id="ba_projection"] a.fusion-owner-link'
        )
        require(fusion_links.count() == 2, "comparison_fusion_owner_link_count")
        if fusion_links.count():
            fusion_links.first.click()
            page.wait_for_function(
                "SELECTED?.view === 'gdn_attention' && SELECTED?.nodeId === 'qkvz_projection'",
                timeout=60_000,
            )
        checks["comparison_fusion_owner_links"] = fusion_links.count()

        # Audit every fused row in a representative composite module.  This
        # includes owners inserted only by Execution IR (the TP collectives),
        # which used to render as plain text because they are absent from raw
        # Model IR.  Every rendered ``fused into`` row must remain navigable.
        page.evaluate(
            """async () => {
              VIEW_STACK = ['top', 'stack', 'gdn_moe_block'];
              DRILL_FROM = [null, 'decoder_stack', 'gdn_layer'];
              SELECTED = {view: 'gdn_moe_block', nodeId: 'attention_residual'};
              await renderView();
              await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
            }"""
        )
        exhaustive_fusion = page.evaluate(
            """() => {
              const view = 'gdn_moe_block';
              const rows = [];
              for (const node of DATA.views[view].nodes || []) {
                const target = `${view}.${node.id}`;
                for (const context of comparisonContextList()) {
                  const cell = comparisonCellForTarget(context, target);
                  if (cell?.status !== 'fused') continue;
                  const selector = `#svg-host g.view-group[data-view="${view}"] g.node[data-id="${node.id}"] a.fusion-owner-link[data-comparison-owner="${context.implementation_id}"]`;
                  rows.push({
                    target,
                    implementation: context.implementation_id,
                    owner: cell.included_in || '',
                    linked: Boolean(document.querySelector(selector)),
                  });
                }
              }
              return rows;
            }"""
        )
        require(
            bool(exhaustive_fusion)
            and all(row["owner"] and row["linked"] for row in exhaustive_fusion),
            "comparison_all_fused_rows_linked",
            actual=exhaustive_fusion,
        )
        execution_owner_link = page.locator(
            '#svg-host g.view-group[data-view="gdn_moe_block"] '
            'g.node[data-id="attention_residual"] '
            f'a.fusion-owner-link[data-comparison-owner="{QWEN_SGLANG}"]'
        )
        rendered_owner_links = page.evaluate(
            """() => Array.from(document.querySelectorAll(
              '#svg-host g.view-group[data-view="gdn_moe_block"] a.fusion-owner-link'
            )).map(link => ({
              implementation: link.dataset.comparisonOwner || '',
              source: link.dataset.comparisonSource || '',
              text: link.textContent || '',
            }))"""
        )
        require(
            execution_owner_link.count() == 1,
            "comparison_execution_owner_link_missing",
            actual=rendered_owner_links,
        )
        if execution_owner_link.count():
            page.evaluate("fitToSvg()")
            page.evaluate(
                "() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))"
            )
            execution_owner_link.click()
            page.wait_for_function(
                "CURRENT_IR_LAYER === 'execution' && "
                "SELECTED?.view === 'gdn_moe_block' && "
                "SELECTED?.nodeId === 'tp_attention_output_collective'",
                timeout=60_000,
            )
        checks["comparison_all_fused_rows_linked"] = len(exhaustive_fusion)

        # An actual rendered SVG click selects the node and broadcasts a module
        # highlight into each isolated physical timeline.
        # Profile switching rebuilds the comparison iframes, so reacquire the
        # live frames instead of retaining the deliberately detached BS64
        # handles created before the refresh test above.
        frames = comparison_frames(page, 2)
        node = page.locator("#svg-host g.view-group g.node.selected").first
        node.click()
        page.wait_for_function("SELECTED && document.querySelectorAll('.comparison-evidence-table tbody tr').length === 2")
        for frame in frames:
            frame.wait_for_function(
                "TIMELINE_IR_TARGET !== '' && TIMELINE_SELECTED_EVENT !== null"
            )
        detail_frames = comparison_detail_frames(page, 2)
        for frame in detail_frames:
            frame.wait_for_function("document.querySelector('#detail h2') !== null")
        centered = [
            frame.evaluate(
                """() => {
                  const step = currentTimelineStep();
                  const event = TIMELINE_SELECTED_EVENT;
                  const center = Number(event.start_us) + Number(event.duration_us) / 2;
                  const rangeCenter = (TIMELINE_RANGE.startUs + TIMELINE_RANGE.endUs) / 2;
                  const span = TIMELINE_RANGE.endUs - TIMELINE_RANGE.startUs;
                  const atBoundary = TIMELINE_RANGE.startUs === 0
                    || Math.abs(TIMELINE_RANGE.endUs - Number(step.duration_us)) < 1e-6;
                  return {center, rangeCenter, span, atBoundary, target: TIMELINE_IR_TARGET};
                }"""
            )
            for frame in frames
        ]
        require(
            all(
                row["atBoundary"]
                or abs(row["center"] - row["rangeCenter"]) <= max(1.0, row["span"] * 0.01)
                for row in centered
            ),
            "comparison_timeline_not_independently_centered",
            actual=centered,
        )
        checks["comparison_independent_center"] = len(centered)
        checks["architecture_to_timeline"] = 2

        # An actual Canvas hit selects exactly one kernel in the source iframe,
        # navigates the shared Architecture, and highlights only the matching
        # module in the peer iframe.
        source = frames[0]
        peer = frames[1]
        source.wait_for_function("TIMELINE_HIT_RECTS.length > 0")
        source.evaluate(
            """() => {
              const seed = TIMELINE_HIT_RECTS.find(item => item.event?._irNode || item.event?._irTargets?.length)?.event;
              if (!seed) return;
              TIMELINE_IR_TARGET = seed._irNode || seed._irTargets[0];
              focusNearestTimelineEvent();
              renderTimeline();
              TIMELINE_SELECTED_EVENT = null;
              TIMELINE_EXPLICIT_SELECTION = null;
            }"""
        )
        hit = source.evaluate(
            """() => {
              const hit = TIMELINE_HIT_RECTS.find(item => item.width >= 4
                && (item.event?._irNode || item.event?._irTargets?.length));
              return hit ? {x: hit.x + hit.width / 2, y: hit.y + hit.height / 2} : null;
            }"""
        )
        require(hit is not None, "clickable_kernel_missing")
        if hit:
            source.locator("#timeline-canvas").click(position=hit)
            page.wait_for_function("SELECTED !== null", timeout=60_000)
            source.wait_for_function("TIMELINE_EXPLICIT_SELECTION?.kind === 'event'", timeout=60_000)
            peer.wait_for_function("TIMELINE_IR_TARGET !== ''", timeout=60_000)
            detail_frames[0].wait_for_function(
                "document.querySelector('#detail h3')?.textContent === 'mapped Python stack' || document.querySelector('#detail .kv .k')?.textContent === 'IR node'",
                timeout=60_000,
            )
            detail_frames[1].wait_for_function(
                "document.querySelector('#detail h2') !== null",
                timeout=60_000,
            )
            detail_state = [
                frame.evaluate(
                    """() => ({
                      title: document.querySelector('#detail h2')?.textContent || '',
                      hasKernelField: Array.from(document.querySelectorAll('#detail .kv .k'))
                        .some(node => node.textContent === 'kernel'),
                    })"""
                )
                for frame in detail_frames
            ]
            require(
                detail_state[0]["hasKernelField"] is True
                and detail_state[1]["title"],
                "comparison_kernel_and_peer_details",
                actual=detail_state,
            )
            checks["comparison_kernel_peer_details"] = 2
        checks["timeline_to_architecture"] = 1 if hit else 0

        # The two frameworks have different formal-step durations and matching
        # modules at different timestamps.  After independent centering, zoom
        # and pan must therefore synchronize as relative transforms, not as one
        # absolute normalized range that would make one anchor jump away.
        source_before = source.evaluate(
            """() => ({
              start: TIMELINE_RANGE.startUs,
              end: TIMELINE_RANGE.endUs,
              center: (TIMELINE_RANGE.startUs + TIMELINE_RANGE.endUs) / 2,
              span: TIMELINE_RANGE.endUs - TIMELINE_RANGE.startUs,
            })"""
        )
        peer_before = peer.evaluate(
            """() => ({
              start: TIMELINE_RANGE.startUs,
              end: TIMELINE_RANGE.endUs,
              center: (TIMELINE_RANGE.startUs + TIMELINE_RANGE.endUs) / 2,
              span: TIMELINE_RANGE.endUs - TIMELINE_RANGE.startUs,
            })"""
        )
        source.evaluate(
            """() => {
              const base = {...TIMELINE_RANGE};
              const center = (base.startUs + base.endUs) / 2;
              const nextSpan = (base.endUs - base.startUs) * 0.65;
              EMBEDDED_RANGE_SYNC_BASE = base;
              TIMELINE_RANGE = {
                startUs: center - nextSpan / 2,
                endUs: center + nextSpan / 2,
              };
              EMBEDDED_LAST_RANGE_KEY = '';
              publishEmbeddedTimelineRange();
            }"""
        )
        peer.wait_for_function(
            """before => {
              const span = TIMELINE_RANGE.endUs - TIMELINE_RANGE.startUs;
              return Math.abs(span / before.span - 0.65) < 0.002;
            }""",
            arg=peer_before,
            timeout=60_000,
        )
        source_after = source.evaluate(
            """() => ({
              center: (TIMELINE_RANGE.startUs + TIMELINE_RANGE.endUs) / 2,
              span: TIMELINE_RANGE.endUs - TIMELINE_RANGE.startUs,
            })"""
        )
        peer_after = peer.evaluate(
            """() => ({
              center: (TIMELINE_RANGE.startUs + TIMELINE_RANGE.endUs) / 2,
              span: TIMELINE_RANGE.endUs - TIMELINE_RANGE.startUs,
            })"""
        )
        source.evaluate(
            """() => {
              const base = {...TIMELINE_RANGE};
              const shift = (base.endUs - base.startUs) * 0.10;
              EMBEDDED_RANGE_SYNC_BASE = base;
              TIMELINE_RANGE = {
                startUs: base.startUs + shift,
                endUs: base.endUs + shift,
              };
              EMBEDDED_LAST_RANGE_KEY = '';
              publishEmbeddedTimelineRange();
            }"""
        )
        peer.wait_for_function(
            """before => {
              const center = (TIMELINE_RANGE.startUs + TIMELINE_RANGE.endUs) / 2;
              return Math.abs((center - before.center) / before.span - 0.10) < 0.002;
            }""",
            arg=peer_after,
            timeout=60_000,
        )
        source_after_pan = source.evaluate(
            """() => ({
              center: (TIMELINE_RANGE.startUs + TIMELINE_RANGE.endUs) / 2,
              span: TIMELINE_RANGE.endUs - TIMELINE_RANGE.startUs,
            })"""
        )
        peer_after_pan = peer.evaluate(
            """() => ({
              center: (TIMELINE_RANGE.startUs + TIMELINE_RANGE.endUs) / 2,
              span: TIMELINE_RANGE.endUs - TIMELINE_RANGE.startUs,
            })"""
        )
        source_scroll_before = source.evaluate(
            """() => {
              const viewport = document.querySelector('#timeline-viewport');
              return {
                top: viewport.scrollTop,
                height: viewport.clientHeight,
                max: Math.max(0, viewport.scrollHeight - viewport.clientHeight),
              };
            }"""
        )
        peer_scroll_before = peer.evaluate(
            """() => {
              const viewport = document.querySelector('#timeline-viewport');
              return {
                top: viewport.scrollTop,
                height: viewport.clientHeight,
                max: Math.max(0, viewport.scrollHeight - viewport.clientHeight),
              };
            }"""
        )
        vertical_shift = source.evaluate(
            """() => {
              const viewport = document.querySelector('#timeline-viewport');
              const max = Math.max(0, viewport.scrollHeight - viewport.clientHeight);
              const requested = max - viewport.scrollTop >= viewport.clientHeight * 0.08
                ? 0.08 : -0.08;
              EMBEDDED_SCROLL_SYNC_BASE = viewport.scrollTop;
              viewport.scrollTop += requested * viewport.clientHeight;
              return (viewport.scrollTop - EMBEDDED_SCROLL_SYNC_BASE) / viewport.clientHeight;
            }"""
        )
        peer.wait_for_function(
            """args => {
              const viewport = document.querySelector('#timeline-viewport');
              const expected = Math.max(
                0,
                Math.min(args.before.max, args.before.top + args.shift * args.before.height),
              );
              return Math.abs(viewport.scrollTop - expected) < 1;
            }""",
            arg={"before": peer_scroll_before, "shift": vertical_shift},
            timeout=60_000,
        )
        source_scroll_after = source.evaluate(
            """() => document.querySelector('#timeline-viewport').scrollTop"""
        )
        peer_scroll_after = peer.evaluate(
            """() => document.querySelector('#timeline-viewport').scrollTop"""
        )
        require(
            abs(source_after["span"] / source_before["span"] - 0.65) < 0.002
            and abs(peer_after["span"] / peer_before["span"] - 0.65) < 0.002
            and abs(source_after["center"] - source_before["center"]) < 0.01
            and abs(peer_after["center"] - peer_before["center"]) < 0.01
            and abs(
                (source_after_pan["center"] - source_after["center"])
                / source_after["span"]
                - 0.10
            ) < 0.002
            and abs(
                (peer_after_pan["center"] - peer_after["center"])
                / peer_after["span"]
                - 0.10
            ) < 0.002
            and abs(
                (source_scroll_after - source_scroll_before["top"])
                / source_scroll_before["height"]
                - vertical_shift
            ) < 0.01
            and abs(
                peer_scroll_after
                - max(
                    0,
                    min(
                        peer_scroll_before["max"],
                        peer_scroll_before["top"]
                        + vertical_shift * peer_scroll_before["height"],
                    ),
                )
            ) < 1,
            "comparison_relative_transform_sync",
            actual={
                "source_before": source_before,
                "source_after": source_after,
                "source_after_pan": source_after_pan,
                "peer_before": peer_before,
                "peer_after": peer_after,
                "peer_after_pan": peer_after_pan,
                "vertical_shift": vertical_shift,
                "source_scroll_before": source_scroll_before,
                "source_scroll_after": source_scroll_after,
                "peer_scroll_before": peer_scroll_before,
                "peer_scroll_after": peer_scroll_after,
            },
        )
        checks["relative_range_transform_sync"] = 1

        # Unselect one implementation (pushState), then exercise browser back
        # and reload to prove query and hash state are durable.
        page.locator("#implementation-multi > summary").click()
        page.locator(f'#implementation-options input[value="{QWEN_VLLM}"]').click()
        page.wait_for_function("SELECTED_IMPLEMENTATIONS.length === 1")
        require(page.locator(".timeline-comparison-card").count() == 0, "single_selection_not_restored")
        page.go_back(wait_until="domcontentloaded", timeout=60_000)
        frames = comparison_frames(page, 2)
        require(page.evaluate("CURRENT_VIEW_MODE") == "split", "history_lost_view_mode")
        page.reload(wait_until="domcontentloaded", timeout=60_000)
        comparison_frames(page, 2)
        require(page.evaluate("CURRENT_VIEW_MODE") == "split", "reload_lost_view_mode")
        checks["url_history_reload"] = 3

        # Draft width resolves only within the authored MTP draft-extend
        # occurrence. The same selected profile must not leak D into the seed
        # prefill occurrence or any other model scope.
        page.goto(
            viewer_url(
                args.base_url,
                model="qwen40_v2",
                profile="qwen40_tp4_mtp_cg_decode_gbs001_8k1k",
                implementation="sglang_qwen4_main_32e9cb5_qsa_hardening_flashinfer_gdn",
                generation="eagle_mtp",
                phase="decode",
                viewMode="architecture",
                irLayer="model",
            ),
            wait_until="domcontentloaded",
            timeout=60_000,
        )
        page.wait_for_function(
            "CURRENT_PROFILE === 'qwen40_tp4_mtp_cg_decode_gbs001_8k1k'",
            timeout=60_000,
        )
        draft_resolution = page.evaluate(
            """() => {
              VIEW_STACK = ['mtp_generation', 'mtp_head'];
              DRILL_FROM = [null, 'mtp_draft_extend'];
              const draft = resolveDimensionSymbol('D', 'mtp_head.candidate_ids');
              const shape = resolveSymbolicShape('[B,D,R,H]', 'mtp_head.candidate_ids');
              DRILL_FROM = [null, 'mtp_prefill'];
              const seed = resolveDimensionSymbol('D', 'mtp_head.candidate_ids');
              return {draft, shape, seed};
            }"""
        )
        require(
            draft_resolution["draft"].get("resolved") is True
            and draft_resolution["draft"].get("value") == 2
            and draft_resolution["shape"].get("resolved") == "[1,2,4,2560]"
            and draft_resolution["seed"].get("resolved") is False,
            "mtp_stage_scoped_dimension",
            actual=draft_resolution,
        )
        checks["mtp_stage_scoped_dimension"] = 1

        # GLM-5.3-Flash is the second real SGLang/vLLM acceptance model. It
        # must resolve to two decode traces under one exact contract rather
        # than accidentally reusing the historical vLLM prefill window.
        page.goto(
            viewer_url(
                args.base_url,
                model="glm53_flash_v2",
                profile=GLM53_PROFILE,
                implementation=GLM53_SGLANG,
                implementations=f"{GLM53_VLLM},{GLM53_SGLANG}",
                viewMode="split",
                irLayer="model",
            ),
            wait_until="domcontentloaded",
            timeout=60_000,
        )
        comparison_frames(page, 2)
        glm53 = page.evaluate(
            """() => ({
              selected: SELECTED_IMPLEMENTATIONS,
              compatible: comparisonExecutionIrCompatible(),
              contexts: comparisonContextList().map(context => ({
                framework: context.framework,
                profile: context.profile_id,
                phase: DATA.profiles[context.profile_id]?.meta?.phase,
                trace: DATA.profiles[context.profile_id]?.meta?.timeline || '',
              })),
            })"""
        )
        require(glm53["selected"] == [GLM53_SGLANG, GLM53_VLLM], "glm53_fixed_framework_order", actual=glm53)
        require(glm53["compatible"] is True, "glm53_shared_execution_ir", actual=glm53)
        require(
            all(row["phase"] == "decode" and row["trace"] for row in glm53["contexts"]),
            "glm53_non_decode_or_missing_trace",
            actual=glm53,
        )
        checks["glm53_real_sglang_vllm"] = 2

        # Distinct exact Execution IR fingerprints remain separate. The viewer
        # shares Model IR only and disables the misleading collapsed layer.
        page.goto(
            viewer_url(
                args.base_url,
                model="glm52_v2",
                profile=GLM_PROFILE,
                implementation=GLM_SGLANG,
                implementations=f"{GLM_TRT},{GLM_SGLANG}",
                viewMode="split",
                irLayer="execution",
            ),
            wait_until="domcontentloaded",
            timeout=60_000,
        )
        comparison_frames(page, 2)
        distinct = page.evaluate(
            """() => ({
              selected: SELECTED_IMPLEMENTATIONS,
              compatible: comparisonExecutionIrCompatible(),
              layer: CURRENT_IR_LAYER,
              disabled: document.querySelector('#ir-layer-select option[value="execution"]').disabled,
              fingerprints: comparisonContextList().map(context => context.execution_variant_id),
            })"""
        )
        require(distinct["selected"] == [GLM_SGLANG, GLM_TRT], "glm_fixed_framework_order", actual=distinct)
        require(distinct["compatible"] is False, "distinct_execution_ir_collapsed", actual=distinct)
        require(distinct["layer"] == "model" and distinct["disabled"], "distinct_execution_ir_not_forced_to_model", actual=distinct)
        require(len(set(distinct["fingerprints"])) == 2, "distinct_fingerprint_missing", actual=distinct)
        checks["distinct_execution_ir"] = 2

        # A contract with no exact peer is visible but disabled with a reason.
        page.goto(
            viewer_url(
                args.base_url,
                model="glm52_v2",
                profile="glm52_tp8_sglang_prefill_bs1_8k",
                implementation=GLM_SGLANG,
                phase="prefill",
                viewMode="architecture",
                irLayer="model",
            ),
            wait_until="domcontentloaded",
            timeout=60_000,
        )
        page.wait_for_function("CURRENT_PROFILE === 'glm52_tp8_sglang_prefill_bs1_8k'")
        unavailable = page.evaluate(
            f"""() => {{
              const input = document.querySelector('#implementation-options input[value="{GLM_TRT}"]');
              return {{disabled: input?.disabled, reason: input?.closest('label')?.title || ''}};
            }}"""
        )
        require(unavailable["disabled"] is True and "exact comparison contract" in unavailable["reason"], "missing_exact_match_reason", actual=unavailable)
        checks["missing_exact_match"] = 1

        # Generic 3-framework UI fixture. The third implementation intentionally
        # has no timeline artifact, proving the comparison remains explicit
        # rather than borrowing another framework's trace.
        page.goto(
            viewer_url(
                args.base_url,
                model="qwen35_v2",
                profile=QWEN_PROFILE,
                implementation=QWEN_SGLANG,
                viewMode="split",
                irLayer="model",
            ),
            wait_until="domcontentloaded",
            timeout=60_000,
        )
        page.wait_for_function(f"RAW_DATA && CURRENT_PROFILE === '{QWEN_PROFILE}'", timeout=60_000)
        synthetic = page.evaluate(
            f"""() => {{
              const syntheticImplementation = 'synthetic_trtllm';
              const syntheticProfile = 'synthetic_trtllm_exact_contract';
              RAW_DATA.implementations[syntheticImplementation] = {{
                ...cloneJson(RAW_DATA.implementations['{QWEN_VLLM}']),
                implementation_id: syntheticImplementation,
                framework_id: 'tensorrt_llm',
                label: 'synthetic TensorRT-LLM acceptance fixture',
              }};
              const profile = cloneJson(RAW_DATA.profiles['qwen35_tp8_vllm_cg_decode_bs64_8k1k']);
              profile.implementation_id = syntheticImplementation;
              profile.meta.timeline = null;
              RAW_DATA.profiles[syntheticProfile] = profile;
              DATA.profiles[syntheticProfile] = profile;
              const contract = RAW_DATA.comparison_contracts[comparisonContractForProfile('{QWEN_PROFILE}')];
              contract.profiles_by_implementation[syntheticImplementation] = syntheticProfile;
              contract.execution_variants_by_implementation[syntheticImplementation] = profile.execution_variant;
              rebuildComparisonContexts(['{QWEN_SGLANG}', '{QWEN_VLLM}', syntheticImplementation]);
              refreshComparisonImplementationOptions();
              loadTimelineForCurrentProfile();
              return {{
                selected: SELECTED_IMPLEMENTATIONS,
                cards: document.querySelectorAll('.timeline-comparison-card').length,
                emptyCards: document.querySelectorAll('.timeline-comparison-card .empty').length,
              }};
            }}"""
        )
        require(
            synthetic["selected"] == [QWEN_SGLANG, QWEN_VLLM, "synthetic_trtllm"],
            "three_framework_order",
            actual=synthetic,
        )
        require(synthetic["cards"] == 3 and synthetic["emptyCards"] == 1, "three_framework_cards", actual=synthetic)
        checks["synthetic_three_frameworks"] = 3

        browser.close()
        for error in page_errors:
            failures.append({"kind": "page_error", "error": error})

    report = {
        "schema_version": "framework-comparison-audit.v1",
        "checks": checks,
        "status": "fail" if failures else "pass",
        "failures": failures,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
