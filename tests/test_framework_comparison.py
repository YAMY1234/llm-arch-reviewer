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
    assert "profile_id.includes" not in viewer


def test_real_browser_comparison_audit_covers_release_contract() -> None:
    audit = (REPO_ROOT / "scripts" / "audit_framework_comparison.py").read_text()
    for required in (
        "architecture_to_timeline",
        "timeline_to_architecture",
        "normalized_range_sync",
        "url_history_reload",
        "glm53_real_sglang_vllm",
        "distinct_execution_ir",
        "missing_exact_match",
        "synthetic_three_frameworks",
    ):
        assert required in audit
