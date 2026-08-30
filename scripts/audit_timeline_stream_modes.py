#!/usr/bin/env python3
"""Audit compact/physical Timeline presentation for one or more V2 bundles.

The compact view is a presentation projection only.  This real-browser gate
loads every accepted profile, checks every formal step, and proves that mode
switching neither mutates nor drops physical event/timing evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlencode

from playwright.sync_api import sync_playwright


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundles", nargs="+", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--browser",
        help="optional browser executable; defaults to Playwright Chromium",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    failures: list[dict[str, object]] = []
    checks = {"bundles": 0, "profiles": 0, "steps": 0, "compact_clicks": 0}
    page_errors: list[dict[str, str]] = []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=args.browser,
            headless=True,
        )
        context = browser.new_context(viewport={"width": 1600, "height": 1000})
        page = context.new_page()
        current_profile = ""
        page.on(
            "pageerror",
            lambda error: page_errors.append(
                {"profile": current_profile, "error": str(error)}
            ),
        )

        for bundle_path in args.bundles:
            bundle = json.loads(bundle_path.read_text())
            model = bundle_path.parent.name
            checks["bundles"] += 1
            for profile_id, profile in bundle.get("profiles", {}).items():
                current_profile = profile_id
                checks["profiles"] += 1
                query = urlencode(
                    {
                        "model": model,
                        "execution": profile["execution_variant"],
                        "implementation": profile["implementation_id"],
                        "profile": profile_id,
                        "phase": profile["meta"]["phase"],
                        "viewMode": "timeline",
                        "irLayer": "execution",
                        "metric": "active",
                        "audit": "1",
                    }
                )
                url = f"{args.base_url.rstrip('/')}/viewer.html?{query}"
                page.goto(url, wait_until="domcontentloaded", timeout=60_000)
                page.wait_for_function(
                    "profile => CURRENT_PROFILE === profile && TIMELINE_DATA != null",
                    arg=profile_id,
                    timeout=60_000,
                )
                result = page.evaluate(
                    """async () => {
                      const evidence = step => JSON.stringify({
                        timing: [
                          step.duration_us, step.active_gpu_us, step.device_gap_us,
                          step.gpu_residency_us, step.gpu_overlap_us,
                        ],
                        events: (step.events || []).map(event => [
                          event.id, event.start_us, event.duration_us, event.stream_id,
                          event._irNode, event._irTargets,
                        ]),
                      });
                      const before = (TIMELINE_DATA.steps || []).map(evidence);
                      const stepChecks = (TIMELINE_DATA.steps || []).map((step, stepIndex) => {
                        const compact = buildCompactTimelineTracks(step);
                        const eventStreams = new Set((step.events || []).map(event => String(event.stream_id)));
                        const laneStreams = new Set(compact.tracks.flatMap(track => track.physical_stream_ids || []));
                        const overlapFree = compact.tracks.every(track => {
                          const segments = timelineActivitySegments(track.events, compact.toleranceUs);
                          for (let left = 0; left < segments.length; left++) {
                            for (let right = left + 1; right < segments.length; right++) {
                              if (segments[left].streamId !== segments[right].streamId &&
                                  timelineSegmentsOverlap(segments[left], segments[right], compact.toleranceUs)) {
                                return false;
                              }
                            }
                          }
                          return true;
                        });
                        return {
                          stepIndex,
                          physicalStreamCount: compact.physicalStreamCount,
                          compactLaneCount: compact.compactLaneCount,
                          eventCount: (step.events || []).length,
                          compactEventCount: compact.tracks.reduce(
                            (total, track) => total + track.events.length, 0
                          ),
                          completeStreamCoverage:
                            eventStreams.size === laneStreams.size &&
                            [...eventStreams].every(streamId => laneStreams.has(streamId)),
                          overlapFree,
                        };
                      });

                      TIMELINE_STEP_INDEX = 0;
                      refreshTimelineControls();
                      resetTimelineRange();
                      TIMELINE_STREAM_MODE = 'compact';
                      document.getElementById('timeline-stream-mode').value = 'compact';
                      renderTimeline();
                      const compactRender = {...TIMELINE_LAST_RENDER_STATS};
                      const hit = TIMELINE_TRACK_HIT_RECTS[0];
                      let expandedByClick = false;
                      if (hit) {
                        const canvas = document.getElementById('timeline-canvas');
                        const rect = canvas.getBoundingClientRect();
                        const clientX = rect.left + hit.x + 8;
                        const clientY = rect.top + hit.y + 8;
                        canvas.dispatchEvent(new MouseEvent('mousedown', {
                          button: 0, bubbles: true, cancelable: true, clientX, clientY,
                        }));
                        window.dispatchEvent(new MouseEvent('mouseup', {
                          button: 0, bubbles: true, cancelable: true, clientX, clientY,
                        }));
                        await new Promise(resolve => requestAnimationFrame(resolve));
                        expandedByClick = TIMELINE_STREAM_MODE === 'physical';
                      }
                      const physicalRender = {...TIMELINE_LAST_RENDER_STATS};
                      const after = (TIMELINE_DATA.steps || []).map(evidence);
                      return {
                        stepChecks,
                        compactRender,
                        physicalRender,
                        expandedByClick,
                        evidenceUnchanged: JSON.stringify(before) === JSON.stringify(after),
                      };
                    }"""
                )
                checks["steps"] += len(result["stepChecks"])
                if result["expandedByClick"]:
                    checks["compact_clicks"] += 1

                profile_issues: list[dict[str, object]] = []
                for step in result["stepChecks"]:
                    if (
                        step["compactLaneCount"] > step["physicalStreamCount"]
                        or step["compactEventCount"] != step["eventCount"]
                        or not step["completeStreamCoverage"]
                        or not step["overlapFree"]
                    ):
                        profile_issues.append(step)
                compact_render = result["compactRender"]
                physical_render = result["physicalRender"]
                if (
                    not result["evidenceUnchanged"]
                    or not result["expandedByClick"]
                    or compact_render.get("streamMode") != "compact"
                    or physical_render.get("streamMode") != "physical"
                    or compact_render.get("trackCount")
                    != compact_render.get("compactLaneCount")
                    or physical_render.get("trackCount")
                    != physical_render.get("physicalStreamCount")
                    or profile_issues
                ):
                    failures.append(
                        {
                            "kind": "timeline_stream_contract",
                            "bundle": str(bundle_path),
                            "profile": profile_id,
                            "result": result,
                            "step_issues": profile_issues,
                        }
                    )

        context.close()
        browser.close()

    failures.extend({"kind": "page_error", **error} for error in page_errors)
    report = {
        "schema_version": "timeline-stream-audit.v1",
        "checks": checks,
        "status": "pass" if not failures else "fail",
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
