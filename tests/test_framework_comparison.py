from __future__ import annotations

from pathlib import Path

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


def test_viewer_comparison_is_metadata_driven_and_timeline_isolated() -> None:
    viewer = (REPO_ROOT / "docs" / "viewer.html").read_text()
    assert "RAW_DATA?.comparison_contracts?.[contractId]" in viewer
    assert "implementation.framework_id" in viewer
    assert "renderComparisonTimelines" in viewer
    assert 'document.createElement("iframe")' in viewer
    assert "comparisonExecutionIrCompatible" in viewer
    assert "MAX_COMPARISON_IMPLEMENTATIONS = 3" in viewer
    assert "comparisonContractDifference" in viewer
    assert "shared Model IR with separate validated Execution IR fingerprints" in viewer
    assert "execution?.enriched?.[viewName]" in viewer
    assert "appendComparisonFusionOwnerNodeLink" in viewer
    assert "showComparisonFusionOwnerInArchitecture(context.implementation_id" in viewer
    assert "profile_id.includes" not in viewer


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
        "normalized_range_sync",
        "url_history_reload",
        "glm53_real_sglang_vllm",
        "distinct_execution_ir",
        "missing_exact_match",
        "synthetic_three_frameworks",
    ):
        assert required in audit
