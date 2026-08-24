"""Validate the strict Qwen3.5 8K/1K cross-engine workload contract."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


CONTRACT_ID = "qwen35-agentx-dep4-mtp6-8k1k-c704-controlled-bs32-nsys-v4"
MODEL_REVISION = "8f590eae8f10bf55d9a46f79ea0280bde435c9f8"
EXPECTED_DATASET_SHA256 = (
    "3e4011a3de2b6d83d5800b27e31dfc6d13b062f521b10ed90869e0136bc73ab2"
)
EXPECTED_RUN_EXACT_SHA256 = (
    "0d0e6c0c1c696af24bb55b28fdd9bf64ff2d1980c0f1b696d2185d2787b8d52f"
)
EXPECTED_GENERATOR_SHA256 = (
    "c6060009fe3bc3ffe7bd39d1a3ab8183bb8a65b2294f4ba68d493b25d92c9298"
)
SA_BENCH_SOURCE_COMMIT = "581ba9aa54736ef520592592bca75f5d32ca8eb9"
EXPECTED_BENCHMARK_SERVING_SHA256 = (
    "d3fa99513b9f25a5a98f73260cf9ae26a44939f2deb2185776ba2ac7f16ebe7a"
)
EXPECTED_BACKEND_REQUEST_FUNC_SHA256 = (
    "2677208c7cfc159b3a4136cc4043a3bae9c62216ef332030350044df0b7f413b"
)
SGLANG_PROFILING_SOURCE_COMMIT = "743ebb718caec3e46cf669b5043692119b5a5a13"
TRT_PY_EXECUTOR_PROFILE_OVERLAY_SHA256 = (
    "a0eb9784bc85c2d6e736224c5bde405649947f32b968f5d8d6c705f6cfc0f348"
)
TRT_DYNAMO_HANDLER_BASE_SHA256 = (
    "e44f1028ae686dd60e6ded8807735e678504898cccac0cf2b70749967714dcbc"
)
TRT_DYNAMO_EXACT_OUTPUT_OVERLAY_SHA256 = (
    "3cb63d65872f82df2377ae7790d59ae9b8a8f090fa502d0a88c5faaa0cb6ef1c"
)
TRT_DYNAMO_WHEEL_BASE_SHA256 = (
    "43d2ff07ea8c60efea41c2f9085ebc846479639e63dfdb276ec1dbc93b144abf"
)
TRT_DYNAMO_EXACT_OUTPUT_WHEEL_SHA256 = (
    "cf3c330a15fbb40fd38c42b59cc192617f1d27c02c3bfcaf83f8fc3ab3af0ca5"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(f"{label}: expected {expected!r}, got {actual!r}")


def _load_manifest(path: Path) -> dict[str, Any]:
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"{path}: expected one dataset manifest JSON row")
    return json.loads(lines[0])


def _validate_dataset(dataset: Path, manifest_path: Path) -> dict[str, Any]:
    dataset_sha = sha256_file(dataset)
    _require_equal("dataset SHA256", dataset_sha, EXPECTED_DATASET_SHA256)
    manifest = _load_manifest(manifest_path)
    _require_equal("manifest dataset SHA256", manifest.get("dataset_sha256"), dataset_sha)
    _require_equal("manifest input tokens", manifest.get("input_tokens"), 8192)
    _require_equal("manifest output tokens", manifest.get("output_tokens"), 1024)
    _require_equal("manifest prompt count", manifest.get("num_prompts"), 704)
    _require_equal("manifest unique prompt count", manifest.get("unique_prompts"), 704)

    rows = [json.loads(line) for line in dataset.read_text().splitlines() if line.strip()]
    _require_equal("dataset row count", len(rows), 704)
    _require_equal("dataset input lengths", Counter(row.get("prompt_len") for row in rows), Counter({8192: 704}))
    _require_equal(
        "dataset output lengths",
        Counter(row.get("expected_output_len") for row in rows),
        Counter({1024: 704}),
    )
    _require_equal(
        "dataset unique prompt digests",
        len({row.get("prompt_sha256") for row in rows}),
        704,
    )
    return {
        "dataset_file": dataset.name,
        "dataset_sha256": dataset_sha,
        "dataset_manifest_file": manifest_path.name,
        "dataset_manifest_sha256": sha256_file(manifest_path),
    }


def _validate_result(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text())
    observed = {
        "completed": result.get("completed"),
        "input_lens": Counter(result.get("input_lens") or []),
        "output_lens": Counter(result.get("output_lens") or []),
        "total_input_tokens": result.get("total_input_tokens"),
        "total_output_tokens": result.get("total_output_tokens"),
    }
    expected = {
        "completed": 704,
        "input_lens": Counter({8192: 704}),
        "output_lens": Counter({1024: 704}),
        "total_input_tokens": 5_767_168,
        "total_output_tokens": 720_896,
    }
    _require_equal("exact workload result", observed, expected)
    return {
        "workload_result_file": path.name,
        "workload_result_sha256": sha256_file(path),
        "completion": {
            "completed": 704,
            "input_lens": {"8192": 704},
            "output_lens": {"1024": 704},
            "total_input_tokens": 5_767_168,
            "total_output_tokens": 720_896,
        },
    }


def _validate_config(path: Path, engine: str) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    _require_equal(
        "model revision",
        (((config.get("identity") or {}).get("model") or {}).get("revision")),
        MODEL_REVISION,
    )
    _require_equal("model precision", (config.get("model") or {}).get("precision"), "fp4")
    resources = config.get("resources") or {}
    _require_equal("GPU type", str(resources.get("gpu_type", "")).lower(), "gb300")
    _require_equal("prefill workers", resources.get("prefill_workers"), 3)
    _require_equal("decode workers", resources.get("decode_workers"), 2)
    _require_equal("GPUs per node", resources.get("gpus_per_node"), 4)
    benchmark = config.get("benchmark") or {}
    benchmark_env = benchmark.get("env") or {}
    _require_equal("benchmark concurrency", str(benchmark_env.get("EXACT_CONCURRENCY")), "704")
    _require_equal("benchmark prompt count", str(benchmark_env.get("EXACT_NUM_PROMPTS")), "704")
    _require_equal(
        "expected dataset SHA256",
        benchmark_env.get("EXPECTED_DATASET_SHA256"),
        EXPECTED_DATASET_SHA256,
    )
    _require_equal(
        "expected benchmark script SHA256",
        benchmark_env.get("EXPECTED_RUN_EXACT_SHA256"),
        EXPECTED_RUN_EXACT_SHA256,
    )
    _require_equal(
        "expected dataset generator SHA256",
        benchmark_env.get("EXPECTED_GENERATOR_SHA256"),
        EXPECTED_GENERATOR_SHA256,
    )
    _require_equal(
        "expected benchmark client SHA256",
        benchmark_env.get("EXPECTED_BENCHMARK_SERVING_SHA256"),
        EXPECTED_BENCHMARK_SERVING_SHA256,
    )
    _require_equal(
        "expected backend request client SHA256",
        benchmark_env.get("EXPECTED_BACKEND_REQUEST_FUNC_SHA256"),
        EXPECTED_BACKEND_REQUEST_FUNC_SHA256,
    )
    _require_equal("profile engine", benchmark_env.get("PROFILE_ENGINE"), engine)
    frameworks = ((config.get("identity") or {}).get("frameworks") or {})
    _require_equal(
        "SA-Bench fixed-shape source",
        frameworks.get("sa_bench_source"),
        SA_BENCH_SOURCE_COMMIT,
    )
    _require_equal(
        "benchmark client SHA256",
        frameworks.get("benchmark_serving_sha256"),
        EXPECTED_BENCHMARK_SERVING_SHA256,
    )
    _require_equal(
        "backend request client SHA256",
        frameworks.get("backend_request_func_sha256"),
        EXPECTED_BACKEND_REQUEST_FUNC_SHA256,
    )

    backend = config.get("backend") or {}
    decode_env = backend.get("decode_environment") or {}
    if engine == "sglang":
        _require_equal(
            "SGLang profiling source",
            frameworks.get("sglang_source"),
            SGLANG_PROFILING_SOURCE_COMMIT,
        )
        decode = ((backend.get("sglang_config") or {}).get("decode") or {})
        _require_equal("tensor parallel size", decode.get("tensor-parallel-size"), 4)
        _require_equal("attention data parallel size", decode.get("data-parallel-size"), 4)
        _require_equal("MoE expert parallel size", decode.get("expert-parallel-size"), 4)
        _require_equal("MTP draft tokens", decode.get("speculative-num-draft-tokens"), 6)
        _require_equal("stream interval", decode.get("stream-interval"), 30)
        _require_equal("forced accept length", decode_env.get("SGLANG_SIMULATE_ACC_LEN"), "4.80")
        _require_equal("exact rank-local batch", decode_env.get("SGLANG_NSYS_EXACT_RUNNING_BATCH"), "32")
        _require_equal("exact sync world size", decode_env.get("SGLANG_NSYS_EXACT_SYNC_WORLD_SIZE"), "4")
        _require_equal("exact gate warm-up batches", decode_env.get("SGLANG_NSYS_EXACT_WARMUP_BATCHES"), "1")
        _require_equal("exact gate reduction", decode_env.get("SGLANG_NSYS_EXACT_GATE_REDUCTION"), "any")
        _require_equal("variable-shape raw capture", decode_env.get("SGLANG_NSYS_REQUIRE_FIXED_CAPTURE"), "0")
        _require_equal("raw captured decode iterations", decode_env.get("SGLANG_NSYS_EXACT_DECODE_BATCHES"), "64")
        _require_equal(
            "controlled worker-wide request cap",
            decode.get("max-running-requests"),
            128,
        )
        _require_equal("profiler type", (config.get("profiling") or {}).get("type"), "nsys")
        _require_equal("CUDA Graph enabled", bool(decode.get("disable-cuda-graph", False)), False)
    elif engine == "trtllm":
        _require_equal(
            "TRT py_executor base SHA256",
            frameworks.get("py_executor_base_sha256"),
            "69b566f2d30e1d1465d4ef85af1913ef3cb8d0f4e36d78bf92989837e6f4aa9a",
        )
        _require_equal(
            "TRT py_executor overlay SHA256",
            frameworks.get("py_executor_profile_overlay_sha256"),
            TRT_PY_EXECUTOR_PROFILE_OVERLAY_SHA256,
        )
        _require_equal(
            "Dynamo TRT handler base SHA256",
            frameworks.get("dynamo_handler_base_sha256"),
            TRT_DYNAMO_HANDLER_BASE_SHA256,
        )
        _require_equal(
            "Dynamo exact-output overlay SHA256",
            frameworks.get("dynamo_exact_output_overlay_sha256"),
            TRT_DYNAMO_EXACT_OUTPUT_OVERLAY_SHA256,
        )
        _require_equal(
            "Dynamo base wheel SHA256",
            frameworks.get("dynamo_wheel_base_sha256"),
            TRT_DYNAMO_WHEEL_BASE_SHA256,
        )
        _require_equal(
            "Dynamo exact-output wheel SHA256",
            frameworks.get("dynamo_exact_output_wheel_sha256"),
            TRT_DYNAMO_EXACT_OUTPUT_WHEEL_SHA256,
        )
        decode = ((backend.get("trtllm_config") or {}).get("decode") or {})
        _require_equal("tensor parallel size", decode.get("tensor_parallel_size"), 4)
        _require_equal("attention data parallel", decode.get("enable_attention_dp"), True)
        _require_equal("MoE expert parallel size", decode.get("moe_expert_parallel_size"), 4)
        _require_equal("MTP draft tokens", (decode.get("speculative_config") or {}).get("max_draft_len"), 6)
        _require_equal("stream interval", decode.get("stream_interval"), 30)
        _require_equal("forced accept length", decode_env.get("TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS"), "4.80")
        _require_equal("exact rank-local batch", decode_env.get("TLLM_PROFILE_EXACT_RUNNING_BATCH"), "32")
        _require_equal("raw captured decode iterations", decode_env.get("TLLM_PROFILE_EXACT_DECODE_BATCHES"), "64")
        _require_equal("profile ranks", decode_env.get("TLLM_PROFILE_LOG_RANKS"), "all")
        _require_equal("rank-local request cap", decode.get("max_batch_size"), 32)
        _require_equal("CUDA Graph batch 32", 32 in ((decode.get("cuda_graph_config") or {}).get("batch_sizes") or []), True)
    else:
        raise ValueError(f"unsupported comparison engine: {engine}")
    return {
        "config_file": path.name,
        "config_sha256": sha256_file(path),
    }


def validate_comparison_workload(
    *,
    engine: str,
    config: Path,
    dataset: Path,
    dataset_manifest: Path,
    workload_result: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the common contract plus engine-local immutable evidence."""

    evidence = {
        **_validate_config(config, engine),
        **_validate_dataset(dataset, dataset_manifest),
        **_validate_result(workload_result),
    }
    contract = {
        "validated": True,
        "contract_id": CONTRACT_ID,
        "model_revision": MODEL_REVISION,
        "model_precision": "fp4",
        "dataset_sha256": evidence["dataset_sha256"],
        "input_tokens": 8192,
        "output_tokens": 1024,
        "request_count": 704,
        "concurrency": 704,
        "prefill_workers": 3,
        "decode_workers": 2,
        "gpu_type": "GB300",
        "tensor_parallel_size": 4,
        "effective_attention_tensor_parallel_size": 1,
        "attention_data_parallel_size": 4,
        "moe_expert_parallel_size": 4,
        "rank_local_batch": 32,
        "mtp_draft_tokens": 6,
        "forced_mean_accept_length": 4.8,
        "stream_interval": 30,
        "injected_scheduler_sleep": False,
        "cuda_graph": True,
        "profiler": "nsys",
        "sample_unit": "one real rank-local BS32 CUDA Graph decode period",
        "sample_aggregation": "mean over one source-balanced pool; parallel ranks are never summed",
        "selected_rank_local_samples": 32,
    }
    return contract, evidence
