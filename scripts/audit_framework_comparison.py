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

        # An actual rendered SVG click selects the node and broadcasts a module
        # highlight into each isolated physical timeline.
        node = page.locator("#svg-host g.view-group g.node").first
        node.click()
        page.wait_for_function("SELECTED && document.querySelectorAll('.comparison-evidence-table tbody tr').length === 2")
        for frame in frames:
            frame.wait_for_function("TIMELINE_IR_TARGET !== ''")
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
        checks["timeline_to_architecture"] = 1 if hit else 0

        # Ranges synchronize by normalized formal-step coordinates; physical
        # event identities and stream ids remain local to each iframe.
        source.evaluate(
            """() => {
              const duration = Number(currentTimelineStep().duration_us);
              TIMELINE_RANGE = {startUs: duration * 0.2, endUs: duration * 0.55};
              EMBEDDED_LAST_RANGE_KEY = '';
              publishEmbeddedTimelineRange();
            }"""
        )
        peer.wait_for_function(
            """() => {
              const duration = Number(currentTimelineStep().duration_us);
              return Math.abs(TIMELINE_RANGE.startUs / duration - 0.2) < 0.002
                && Math.abs(TIMELINE_RANGE.endUs / duration - 0.55) < 0.002;
            }""",
            timeout=60_000,
        )
        checks["normalized_range_sync"] = 1

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
