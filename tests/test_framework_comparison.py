from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from llm_arch_v2.compiler import compile_catalog


REPO_ROOT = Path(__file__).parents[1]


def test_comparison_compiler_indexes_only_explicit_framework_ids() -> None:
    for model in ("qwen35", "glm52"):
        bundle = compile_catalog(REPO_ROOT / "catalog" / model)
        assert bundle["comparison_contracts"]
        for implementation in bundle["implementations"].values():
            assert implementation["framework_id"] in {
                "sglang",
                "vllm",
                "tensorrt_llm",
            }
        for contract in bundle["comparison_contracts"].values():
            assert set(contract["profiles_by_implementation"]) == set(
                contract["execution_variants_by_implementation"]
            )


def test_qwen38_current_and_previous_traces_are_exact_same_execution_candidates() -> None:
    bundle = compile_catalog(REPO_ROOT / "catalog" / "qwen38_flash_next")
    profile_ids = (
        "qwen38_flash_next_tp4_mtp_cg_decode_gbs001_8k1k_bind_ba79b6e52262fede",
        "qwen38_flash_next_tp4_mtp_cg_decode_gbs001_8k1k",
    )
    profiles = [bundle["profiles"][profile_id] for profile_id in profile_ids]
    assert len({profile["execution_variant"] for profile in profiles}) == 1
    assert len({profile["meta"]["comparison_contract_id"] for profile in profiles}) == 1
    assert {profile["meta"]["trace_time"]["basis"] for profile in profiles} == {
        "cataloged"
    }
    assert all(profile["meta"]["timeline"] for profile in profiles)


def test_viewer_comparison_is_metadata_driven_and_timeline_isolated() -> None:
    viewer = (REPO_ROOT / "docs" / "viewer.html").read_text()
    assert 'const requestedProfile = d.profiles?.[qs.get("profile") || ""]' in viewer
    assert "requestedProfile?.execution_variant" in viewer
    assert "requestedProfile?.implementation_id" in viewer
    assert "RAW_DATA?.comparison_contracts?.[contractId]" in viewer
    assert "implementation.framework_id" in viewer
    assert "renderComparisonTimelines" in viewer
    assert 'document.createElement("iframe")' in viewer
    assert "comparisonExecutionIrCompatible" in viewer
    assert "MAX_COMPARISON_IMPLEMENTATIONS = 3" in viewer
    assert "comparisonContractDifference" in viewer
    assert "profilesForImplementationInExecution" in viewer
    assert 'shortReason: "Different Execution IR"' in viewer
    assert 'badge.textContent = "Exact workload"' in viewer
    assert "Different workload" in viewer
    assert "traceTimeForProfile" in viewer
    assert "positionImplementationOptions" in viewer
    assert "Comparable · same Execution IR" in viewer
    assert "execution?.enriched?.[viewName]" in viewer
    assert "appendComparisonFusionOwnerNodeLink" in viewer
    assert "renderComparisonDetails" in viewer
    assert "comparison-detail-frame" in viewer
    assert "independentCenter: true" in viewer
    assert 'type: "llm-arch-reviewer:timeline-transform"' in viewer
    assert 'type: "llm-arch-reviewer:timeline-vertical-transform"' in viewer
    assert "function applyEmbeddedTimelineTransform" in viewer
    assert "function applyEmbeddedTimelineScrollTransform" in viewer
    assert "EMBEDDED_RANGE_SYNC_BASE = {...TIMELINE_RANGE}" in viewer
    assert "dimensionResolutionScope" in viewer
    assert "Tensor symbols and profile-resolved dimensions" in viewer
    assert "symbolic shape" in viewer
    assert "resolved shape" in viewer
    assert 'drawEdge(inner, e, layoutedEdge, viewName)' in viewer
    assert 'parts.push({text: shape.symbolic, cls: "shape-symbolic"})' in viewer
    assert 'parts.push({text: `resolved ${shape.resolved}`, cls: "shape-resolved"})' in viewer
    assert "irTargetExistsForNavigation" in viewer
    assert "RAW_DATA?.execution_variants?.[executionVariantId]?.views" in viewer
    assert "CURRENT_EXECUTION = context.execution_variant_id;" in viewer
    assert "showComparisonFusionOwnerInArchitecture(context.implementation_id" in viewer
    assert "profile_id.includes" not in viewer


def test_standalone_export_keeps_symbol_resolution_and_comparison_details(
    tmp_path: Path,
) -> None:
    elk = tmp_path / "elk.js"
    elk.write_text("window.ELK = function ELK() {};\n")
    output = tmp_path / "qwen35-standalone.html"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "export_standalone.py"),
            "--model",
            "qwen35_v2",
            "--output",
            str(output),
            "--elk-js",
            str(elk),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    standalone = output.read_text()
    assert "window.__LLM_ARCH_STANDALONE__ = true" in standalone
    assert "function resolveDimensionSymbol" in standalone
    assert "Tensor symbols and profile-resolved dimensions" in standalone
    assert "function renderComparisonDetails" in standalone
    assert "comparison-detail-frame" in standalone


def test_qwen35_comparison_drill_parent_has_framework_specific_union_timing() -> None:
    bundle = compile_catalog(REPO_ROOT / "catalog" / "qwen35")
    expected_sources = {
        "gdn_attention.causal_conv",
        "gdn_attention.gated_delta_recurrence",
        "gdn_attention.output_gate_norm",
        "gdn_attention.output_projection",
        "gdn_attention.qkvz_projection",
    }
    for profile_id in (
        "qwen35_tp8_sglang_cg_decode_bs64_8k1k",
        "qwen35_tp8_vllm_cg_decode_bs64_8k1k",
    ):
        variants = bundle["profiles"][profile_id]["data"][
            "gdn_moe_block.attention"
        ]
        cell = variants.get("cg_decode_bs64_8k1k") or variants.get("ALL")
        assert cell["attribution_status"] == "inclusive_rollup"
        assert cell["metric_kind"] == "inclusive_rollup"
        assert cell["active_gpu_ms"] > 0
        assert set(cell["rollup_sources"]) == expected_sources
        assert "gdn_moe_block.input_norm" not in cell["rollup_sources"]
        assert "gdn_moe_block.tp_attention_output_collective" not in cell[
            "rollup_sources"
        ]


def test_real_browser_comparison_audit_covers_release_contract() -> None:
    audit = (REPO_ROOT / "scripts" / "audit_framework_comparison.py").read_text()
    for required in (
        "architecture_to_timeline",
        "timeline_to_architecture",
        "comparison_parent_union_timing",
        "comparison_fusion_owner_links",
        "comparison_all_fused_rows_linked",
        "comparison_dual_details",
        "comparison_independent_center",
        "comparison_kernel_peer_details",
        "tensor_symbol_resolution",
        "profile_symbol_refresh",
        "mtp_stage_scoped_dimension",
        "relative_range_transform_sync",
        "url_history_reload",
        "glm53_real_sglang_vllm",
        "different_execution_ir_disabled",
        "same_execution_different_workload_allowed",
        "compact_trace_picker",
        "synthetic_three_frameworks",
    ):
        assert required in audit
