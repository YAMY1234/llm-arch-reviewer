from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from llm_arch_v2.compiler import compile_catalog  # noqa: E402
from llm_arch_v2.sol import (  # noqa: E402
    SolError,
    _canonical_hash,
    _stable_numeric_tree,
    attach_sol_to_profile,
    build_sol_artifacts,
)


QWEN38_FLASH_NEXT_ROOT = REPO_ROOT / "catalog" / "qwen38_flash_next"
HARDWARE_PATH = REPO_ROOT / "catalog" / "hardware" / "gb300_nvl72.yaml"
MANIFEST_PATH = (
    QWEN38_FLASH_NEXT_ROOT / "sol_manifests" / "tp4_gb300_decode_gbs1_8k1k.yaml"
)


def test_sol_persisted_floats_are_cross_platform_canonical() -> None:
    macos_result = {
        "duration_ms": 0.008253246666666667,
        "critical_path_ms": 0.6613886946666666,
        "nested": [0.466880484, -0.0],
    }
    linux_result = {
        "duration_ms": 0.008253246666666669,
        "critical_path_ms": 0.6613886946666665,
        "nested": [0.4668804839999999, 0.0],
    }

    assert _stable_numeric_tree(macos_result) == _stable_numeric_tree(linux_result)
    assert _canonical_hash(macos_result) == _canonical_hash(linux_result)


def test_qwen38_flash_next_sol_profile_is_separate_fail_closed_overlay() -> None:
    bundle = compile_catalog(QWEN38_FLASH_NEXT_ROOT)

    assert bundle["meta"]["sol_profile_count"] == 1
    assert bundle["meta"]["gap_report_count"] == 1
    sol = bundle["sol_profiles"]["qwen38_flash_next_tp4_gb300_decode_gbs1_8k1k_ideal_v1"]
    gap = bundle["gap_reports"]["qwen38_flash_next_tp4_gb300_decode_gbs1_8k1k_gap_v1"]

    assert sol["status"] == "partial"
    assert gap["status"] == "partial_calibration"
    assert sol["critical_path"]["complete_step"] is True
    assert sol["workload_ir"]["phase"] == "decode"
    assert sol["workload_ir"]["cuda_graph"] is True
    assert sol["workload_fingerprint"] == sol["workload_ir"]["fingerprint"]
    assert sol["node_estimates"]["top.decoder_stack"]["status"] == "estimated"
    assert sol["coverage"]["unsupported_targets"] == []
    assert sol["coverage"]["ideal_estimated_node_count"] >= 60
    assert sol["coverage"]["transition_simulated_node_count"] >= 60
    assert sol["coverage"]["legacy_sensitivity_node_count"] == 0
    assert sol["coverage"]["structural_node_count"] >= 20
    assert sol["node_estimates"]["moe.routed_experts"]["ideal_ms"] > 0
    assert sol["node_estimates"]["top.lm_head"]["limiting_resource"] == "hbm"
    assert sol["node_estimates"]["top.lm_head"]["ideal_ms"] == 0.03974736
    lm_head = sol["node_estimates"]["top.lm_head"]
    assert lm_head["attainable_ms"] is None
    assert lm_head["methodology_optimistic_ms"] is None
    assert lm_head["methodology_conservative_ms"] is None
    assert lm_head["confidence"] == "ideal_bound_only"
    assert lm_head["physical_plan"]["schema_version"] == "transition-plan.v1"
    assert lm_head["physical_plan"]["critical_path_ms"] == lm_head["ideal_ms"]
    # Uncalibrated efficiency seeds are no longer emitted as projections.
    assert gap["nodes"]["top.lm_head"]["diagnosis"] == (
        "requires_calibration_before_framework_blame"
    )
    qsa = sol["node_estimates"]["qsa_attention.indexer"]
    assert qsa["operator_family"] == "qsa_indexer"
    assert qsa["cost_ir"]["schema_version"] == "cost-ir.v1"
    assert qsa["cost_ir"]["physical_model"] == "roofline"
    assert "sglang" not in json.dumps(qsa["cost_ir"]).lower()
    assert "kernel" not in json.dumps(qsa["cost_ir"]).lower()
    assert gap["nodes"]["qsa_attention.indexer"]["diagnosis"] == (
        "requires_calibration_before_framework_blame"
    )
    assert not gap["model_violations"]

    measured = bundle["profiles"][sol["measured_profile_id"]]
    cell = next(iter(measured["data"]["top.lm_head"].values()))
    assert cell["active_gpu_ms"] == 0.050061
    assert cell["sol"]["ideal_ms"] == 0.03974736
    assert cell["sol"]["physical_coverage_pct"] == pytest.approx(
        0.03974736 / 0.050061 * 100.0
    )
    assert cell["sol"]["attainable_ms"] is None
    assert cell["sol"]["critical_transition"] == "execute"


def test_fused_semantic_leaf_inherits_non_additive_parent_sol_assignment() -> None:
    bundle = compile_catalog(QWEN38_FLASH_NEXT_ROOT)
    sol = bundle["sol_profiles"]["qwen38_flash_next_tp4_gb300_decode_gbs1_8k1k_ideal_v1"]
    gap = bundle["gap_reports"]["qwen38_flash_next_tp4_gb300_decode_gbs1_8k1k_gap_v1"]
    sol = copy.deepcopy(sol)
    gap = copy.deepcopy(gap)
    profile = copy.deepcopy(bundle["profiles"][sol["measured_profile_id"]])
    sol["node_estimates"]["semantic_drill.fused_owner"] = {
        "target": "semantic_drill.fused_owner",
        "status": "structural",
        "included_in_target": "top.lm_head",
        "confidence": "not_applicable",
    }
    gap["nodes"]["semantic_drill.fused_owner"] = {
        "status": "structural",
        "diagnosis": "structural_boundary",
    }
    profile["data"]["semantic_drill.fused_leaf"] = {
        "default": {
            "status": "fused",
            "included_in": "semantic_drill.fused_owner",
            "label": "shared fused interval",
        }
    }

    attach_sol_to_profile(profile, sol, gap)

    attached = profile["data"]["semantic_drill.fused_leaf"]["default"]["sol"]
    assert attached["status"] == "included_in_parent"
    assert attached["included_in_target"] == "top.lm_head"
    assert attached["allocation"] == "shared_parent_interval_non_additive"
    assert attached["ideal_ms"] == sol["node_estimates"]["top.lm_head"]["ideal_ms"]


def test_nested_drill_metrics_receive_fail_closed_sol_ownership() -> None:
    bundle = compile_catalog(QWEN38_FLASH_NEXT_ROOT)
    sol = copy.deepcopy(
        bundle["sol_profiles"]["qwen38_flash_next_tp4_gb300_decode_gbs1_8k1k_ideal_v1"]
    )
    gap = copy.deepcopy(
        bundle["gap_reports"]["qwen38_flash_next_tp4_gb300_decode_gbs1_8k1k_gap_v1"]
    )
    profile = {
        "data": {
            "qsa_attention.indexer": {
                "default": {
                    "drill_view": "qsa_indexer",
                    "drill_metrics": {
                        "q_norm_rope": {
                            "active_gpu_ms": 0.01,
                            "scope_target": "qsa_attention.indexer",
                        },
                        "raw_k_cache": {
                            "status": "fused",
                            "included_in": "qsa_indexer.q_norm_rope",
                            "scope_target": "qsa_attention.indexer",
                        },
                        "index_in": {
                            "status": "structural",
                            "scope_target": "qsa_attention.indexer",
                        },
                    },
                }
            }
        }
    }

    attach_sol_to_profile(profile, sol, gap)

    drill = profile["data"]["qsa_attention.indexer"]["default"]["drill_metrics"]
    for node_id in ("q_norm_rope", "raw_k_cache"):
        attached = drill[node_id]["sol"]
        assert attached["status"] == "included_in_parent"
        assert attached["included_in_target"] == "qsa_attention.indexer"
        assert attached["allocation"] == "shared_parent_interval_non_additive"
    # A structural boundary has no independent physical cost and must not
    # inherit a performance percentage merely because it encloses a scope.
    assert "sol" not in drill["index_in"]


def test_fused_direct_structural_estimate_resolves_to_timing_owner() -> None:
    bundle = compile_catalog(QWEN38_FLASH_NEXT_ROOT)
    sol = copy.deepcopy(
        bundle["sol_profiles"]["qwen38_flash_next_tp4_gb300_decode_gbs1_8k1k_ideal_v1"]
    )
    gap = copy.deepcopy(
        bundle["gap_reports"]["qwen38_flash_next_tp4_gb300_decode_gbs1_8k1k_gap_v1"]
    )
    profile = {
        "data": {
            "linear_attention.ba_projection": {
                "default": {
                    "status": "fused",
                    "included_in": "linear_attention.qkvz_projection",
                }
            }
        }
    }

    attach_sol_to_profile(profile, sol, gap)

    attached = profile["data"]["linear_attention.ba_projection"]["default"]["sol"]
    assert attached["status"] == "included_in_parent"
    assert attached["included_in_target"] == "linear_attention.qkvz_projection"
    assert attached["ideal_ms"] == sol["node_estimates"][
        "linear_attention.qkvz_projection"
    ]["ideal_ms"]


def test_measured_faster_than_ideal_invalidates_sol_model() -> None:
    bundle = compile_catalog(QWEN38_FLASH_NEXT_ROOT)
    model_ir = yaml.safe_load((QWEN38_FLASH_NEXT_ROOT / "model_ir.yaml").read_text())
    hardware = yaml.safe_load(HARDWARE_PATH.read_text())
    manifest = yaml.safe_load(MANIFEST_PATH.read_text())

    # Deliberately make the hardware impossibly slow. The engine must flag the
    # model rather than reporting a negative optimization gap.
    hardware = copy.deepcopy(hardware)
    hardware["theoretical"]["memory"]["hbm_bytes_per_s"] = 1_000_000
    sol, gap = build_sol_artifacts(
        model_ir=model_ir,
        execution_variants=bundle["execution_variants"],
        profiles=bundle["profiles"],
        hardware=hardware,
        manifest=manifest,
        manifest_source=MANIFEST_PATH,
        hardware_source=HARDWARE_PATH,
    )

    assert sol["status"] == "invalid"
    assert gap["status"] == "invalid_sol_model"
    assert "top.lm_head" in gap["model_violations"]
    assert gap["nodes"]["top.lm_head"]["diagnosis"] == (
        "fix_sol_model_before_optimization"
    )


def test_sol_provenance_is_checkout_independent() -> None:
    bundle = compile_catalog(QWEN38_FLASH_NEXT_ROOT)
    model_ir = yaml.safe_load((QWEN38_FLASH_NEXT_ROOT / "model_ir.yaml").read_text())
    hardware = yaml.safe_load(HARDWARE_PATH.read_text())
    manifest = yaml.safe_load(MANIFEST_PATH.read_text())
    sol, _ = build_sol_artifacts(
        model_ir=model_ir,
        execution_variants=bundle["execution_variants"],
        profiles=bundle["profiles"],
        hardware=hardware,
        manifest=manifest,
        manifest_source=MANIFEST_PATH,
        hardware_source=HARDWARE_PATH,
    )

    assert sol["provenance"]["manifest"].startswith("catalog/")
    assert sol["provenance"]["hardware_spec"].startswith("catalog/")
    assert "/Users/" not in json.dumps(sol["provenance"])


def test_methodology_role_order_is_fail_closed() -> None:
    bundle = compile_catalog(QWEN38_FLASH_NEXT_ROOT)
    model_ir = yaml.safe_load((QWEN38_FLASH_NEXT_ROOT / "model_ir.yaml").read_text())
    hardware = yaml.safe_load(HARDWARE_PATH.read_text())
    manifest = yaml.safe_load(MANIFEST_PATH.read_text())
    hardware = copy.deepcopy(hardware)
    optimistic = hardware["methodologies"]["gb300_methodology_optimistic_seed_v0"]
    conservative = hardware["methodologies"]["gb300_methodology_conservative_seed_v0"]
    optimistic["enabled"] = True
    conservative["enabled"] = True
    manifest["nodes"]["top.lm_head"]["legacy_sensitivity"] = True
    optimistic["operator_families"] = {}
    optimistic["defaults"].update(
        tensor_core_efficiency=0.01,
        memory_efficiency=0.01,
        interconnect_efficiency=0.01,
        launch_us=10.0,
    )
    conservative["operator_families"] = {}
    conservative["defaults"].update(
        tensor_core_efficiency=1.0,
        memory_efficiency=1.0,
        interconnect_efficiency=1.0,
        launch_us=0.0,
        sync_us=0.0,
    )

    with pytest.raises(SolError, match="conservative methodology is faster"):
        build_sol_artifacts(
            model_ir=model_ir,
            execution_variants=bundle["execution_variants"],
            profiles=bundle["profiles"],
            hardware=hardware,
            manifest=manifest,
            manifest_source=MANIFEST_PATH,
            hardware_source=HARDWARE_PATH,
        )


def test_collective_startup_and_wire_transfer_are_serial_transitions() -> None:
    bundle = compile_catalog(QWEN38_FLASH_NEXT_ROOT)
    model_ir = yaml.safe_load((QWEN38_FLASH_NEXT_ROOT / "model_ir.yaml").read_text())
    hardware = yaml.safe_load(HARDWARE_PATH.read_text())
    manifest = yaml.safe_load(MANIFEST_PATH.read_text())
    hardware = copy.deepcopy(hardware)
    hardware["theoretical"]["interconnect"]["nvlink"]["latency_floor_us"] = 10.0

    sol, _ = build_sol_artifacts(
        model_ir=model_ir,
        execution_variants=bundle["execution_variants"],
        profiles=bundle["profiles"],
        hardware=hardware,
        manifest=manifest,
        manifest_source=MANIFEST_PATH,
        hardware_source=HARDWARE_PATH,
    )

    estimate = sol["node_estimates"]["top.tp_embedding_collective"]
    transitions = estimate["physical_plan"]["transitions"]
    assert [transition["id"] for transition in transitions] == [
        "startup",
        "wire_transfer",
    ]
    assert transitions[1]["depends_on"] == ["startup"]
    assert estimate["ideal_ms"] == pytest.approx(
        estimate["components_ms"]["latency"]
        + estimate["components_ms"]["interconnect"]
    )


def test_attainable_projection_requires_exact_kernel_plan_identity() -> None:
    bundle = compile_catalog(QWEN38_FLASH_NEXT_ROOT)
    model_ir = yaml.safe_load((QWEN38_FLASH_NEXT_ROOT / "model_ir.yaml").read_text())
    hardware = yaml.safe_load(HARDWARE_PATH.read_text())
    manifest = yaml.safe_load(MANIFEST_PATH.read_text())
    hardware = copy.deepcopy(hardware)
    manifest = copy.deepcopy(manifest)
    plan = {
        "schema_version": "kernel-plan.v1",
        "plan_id": "lm_head_cutlass_bf16_v1",
        "source": "microbenchmark",
        "algorithm": "cutlass_gemm",
        "dtype": "bf16",
        "tile": {"cta_m": 64, "cta_n": 64, "cta_k": 16},
    }
    fingerprint = hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    surface_id = "lm_head_plan_exact"
    hardware["calibration"]["surfaces"][surface_id] = {
        "kernel_plan_fingerprint": fingerprint,
        "match_fields": ["m", "n", "k"],
        "evidence": {
            "benchmark": "cutlass_profiler",
            "artifact_sha256": "c" * 64,
            "methodology": "exact shape; stable clocks; warmed; held-out repeats",
        },
        "points": [
            {
                "match": {"m": 1, "n": 62080, "k": 2560},
                "attainable_interval_ms": {
                    "p10": 0.044,
                    "p50": 0.046,
                    "p90": 0.048,
                },
            }
        ],
    }
    manifest["nodes"]["top.lm_head"]["kernel_plan"] = plan
    manifest["nodes"]["top.lm_head"]["calibration_surface"] = surface_id

    sol, gap = build_sol_artifacts(
        model_ir=model_ir,
        execution_variants=bundle["execution_variants"],
        profiles=bundle["profiles"],
        hardware=hardware,
        manifest=manifest,
        manifest_source=MANIFEST_PATH,
        hardware_source=HARDWARE_PATH,
    )

    estimate = sol["node_estimates"]["top.lm_head"]
    assert estimate["attainable_ms"] == 0.046
    assert estimate["attainable_interval_ms"] == {
        "p10_ms": 0.044,
        "p50_ms": 0.046,
        "p90_ms": 0.048,
    }
    assert estimate["confidence"] == "plan_exact_calibrated"
    assert gap["nodes"]["top.lm_head"]["implementation_gap_ms"] == pytest.approx(
        0.050061 - 0.046
    )


def test_sol_schemas_are_valid_json_documents() -> None:
    for name in (
        "hardware-spec.schema.json",
        "cost-ir.schema.json",
        "kernel-plan.schema.json",
        "transition-plan.schema.json",
        "workload-ir.schema.json",
        "sol-manifest.schema.json",
        "sol-profile.schema.json",
        "gap-report.schema.json",
        "sol-calibration-surface.schema.json",
    ):
        value = json.loads((REPO_ROOT / "schema" / "v2" / name).read_text())
        assert value["$schema"] == "https://json-schema.org/draft/2020-12/schema"


def test_calibration_import_is_content_hashed_and_non_destructive(
    tmp_path: Path,
) -> None:
    surface = {
        "schema_version": "sol-calibration-surface.v1",
        "hardware_spec_id": "nvidia_gb300_nvl72_per_gpu_v1",
        "surface_id": "gemm_bf16_exact",
        "kernel_plan_fingerprint": "b" * 64,
        "match_fields": ["m", "n", "k"],
        "evidence": {
            "benchmark": "cutlass_profiler",
            "artifact_sha256": "a" * 64,
            "methodology": "stable clocks; warmed; correctness checked",
        },
        "points": [
            {"match": {"m": 1, "n": 62080, "k": 2560}, "efficiency": 0.8}
        ],
    }
    surface_path = tmp_path / "surface.json"
    output_path = tmp_path / "gb300-calibrated.yaml"
    surface_path.write_text(json.dumps(surface))
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "import_sol_calibration.py"),
            "--hardware",
            str(HARDWARE_PATH),
            "--surface",
            str(surface_path),
            "--output",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    imported = yaml.safe_load(output_path.read_text())
    assert imported["status"] == "partially_calibrated"
    point = imported["calibration"]["surfaces"]["gemm_bf16_exact"]["points"][0]
    assert point["efficiency"] == 0.8
    assert (
        imported["calibration"]["surfaces"]["gemm_bf16_exact"][
            "kernel_plan_fingerprint"
        ]
        == "b" * 64
    )
    original = yaml.safe_load(HARDWARE_PATH.read_text())
    assert original["status"] == "theoretical_only"
