#!/usr/bin/env python3
"""Drive bounded Qwen3.5 DEP4 Torch-profiler captures through SGLang HTTP APIs."""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("attribution", "prefill8k", "decode-bs32"), required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-ranks", type=int, default=4)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--model-revision", required=True)
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def get_json(base_url: str, path: str, *, timeout: float = 30) -> dict[str, Any]:
    response = requests.get(base_url + path, timeout=timeout)
    response.raise_for_status()
    return response.json()


def generate(base_url: str, *, input_len: int, output_len: int, token_id: int) -> dict[str, Any]:
    payload = {
        "input_ids": [token_id] * input_len,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": output_len,
            "ignore_eos": True,
        },
    }
    response = requests.post(base_url + "/generate", json=payload, timeout=1800)
    response.raise_for_status()
    return response.json()


def run_batch(
    base_url: str, *, batch_size: int, input_len: int, output_len: int, token_seed: int
) -> list[dict[str, Any]]:
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = [
            executor.submit(
                generate,
                base_url,
                input_len=input_len,
                output_len=output_len,
                token_id=token_seed + index % 17,
            )
            for index in range(batch_size)
        ]
        return [future.result() for future in futures]


def summarize_generation_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep token/speculation counters without persisting generated content."""

    rows = []
    for index, result in enumerate(results):
        meta = result.get("meta_info") or {}
        scalar_meta = {
            str(key): value
            for key, value in meta.items()
            if isinstance(value, (str, int, float, bool)) or value is None
        }
        output_ids = result.get("output_ids") or []
        rows.append(
            {
                "request_index": index,
                "output_token_count": len(output_ids),
                "meta_info": scalar_meta,
            }
        )
    return {"request_count": len(results), "requests": rows}


def start_profile(
    base_url: str,
    output_dir: Path,
    *,
    profile_id: str,
    num_steps: int,
    with_stack: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    payload = {
        "output_dir": str(output_dir),
        "num_steps": num_steps,
        "activities": ["CPU", "GPU"],
        "with_stack": with_stack,
        "record_shapes": with_stack,
        "profile_id": profile_id,
        "profile_prefix": profile_id,
        "merge_profiles": False,
    }
    response = requests.post(base_url + "/start_profile", json=payload, timeout=60)
    response.raise_for_status()
    record: dict[str, Any] = {
        "request": payload,
        "status_code": response.status_code,
        "content_type": response.headers.get("content-type"),
        "body_prefix": response.text[:512],
    }
    if response.content.strip():
        try:
            body = response.json()
        except requests.exceptions.JSONDecodeError:
            body = None
        record["body"] = body
        if isinstance(body, dict) and body.get("success") is False:
            raise RuntimeError(f"profiler rejected request: {body}")
    return record


def normalized_loads(base_url: str, expected_ranks: int) -> list[dict[str, Any]]:
    body = get_json(base_url, "/v1/loads?include=core")
    loads = body.get("loads") or []
    if len(loads) != expected_ranks:
        raise RuntimeError(f"expected {expected_ranks} DP loads, got {body}")
    result = []
    for index, load in enumerate(loads):
        result.append(
            {
                "dp_rank": int(load.get("dp_rank") if load.get("dp_rank") is not None else index),
                "num_running_reqs": int(load.get("num_running_reqs") or 0),
                "num_waiting_reqs": int(load.get("num_waiting_reqs") or 0),
                "num_waiting_uncached_tokens": int(load.get("num_waiting_uncached_tokens") or 0),
                "num_active_tokens": int(load.get("num_active_tokens") or 0),
            }
        )
    return sorted(result, key=lambda item: item["dp_rank"])


def wait_for_decode_batch(
    base_url: str,
    request_future: Future[list[dict[str, Any]]],
    *,
    expected_ranks: int,
    batch_size: int,
) -> dict[str, Any]:
    deadline = time.monotonic() + 1800
    recent: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        if request_future.done():
            request_future.result()
            raise RuntimeError("formal decode batch completed before the profiler trigger")
        ranks = normalized_loads(base_url, expected_ranks)
        running = [rank["num_running_reqs"] for rank in ranks]
        sample = {
            "observed_at_unix": time.time(),
            "ranks": ranks,
            "global_running_reqs": sum(running),
            "global_waiting_reqs": sum(rank["num_waiting_reqs"] for rank in ranks),
            "global_waiting_uncached_tokens": sum(
                rank["num_waiting_uncached_tokens"] for rank in ranks
            ),
        }
        recent.append(sample)
        recent = recent[-64:]
        if (
            sample["global_running_reqs"] == batch_size
            and sample["global_waiting_reqs"] == 0
            and sample["global_waiting_uncached_tokens"] == 0
            and max(running) - min(running) <= 1
        ):
            return {"trigger": sample, "recent": recent}
        time.sleep(0.2)
    raise RuntimeError(f"timed out waiting for exact decode batch; recent={recent}")


def wait_for_traces(output_dir: Path, expected_ranks: int) -> list[Path]:
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        traces = sorted(output_dir.glob("*.trace.json.gz"))
        if len(traces) == expected_ranks:
            return traces
        if len(traces) > expected_ranks:
            raise RuntimeError(f"expected {expected_ranks} traces, found {len(traces)}")
        time.sleep(2)
    raise RuntimeError(f"expected {expected_ranks} traces in {output_dir}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    trace_dir = args.output_dir / "traces"
    # The first scheduler-side endpoint can trigger lazy Triton compilation on
    # a cold cache. Keep the liveness polling strict, but allow this provenance
    # snapshot to span that one-time compile instead of aborting a valid run.
    server_info = get_json(args.base_url, "/server_info", timeout=300)
    protocol: dict[str, Any] = {
        "schema_version": "qwen35-sglang-profile-protocol.v1",
        "kind": args.kind,
        "source_commit": args.source_commit,
        "source_manifest_sha256": args.source_manifest_sha256,
        "model_revision": args.model_revision,
        "expected_ranks": args.expected_ranks,
        "server_info": server_info,
        "warmup": [],
        "formal": {},
    }

    if args.kind == "attribution":
        protocol["warmup"].append({"batch": 4, "isl": 64, "osl": 8})
        run_batch(args.base_url, batch_size=4, input_len=64, output_len=8, token_seed=101)
        response = start_profile(
            args.base_url,
            trace_dir,
            profile_id="qwen35-dep4-attribution-cgoff",
            num_steps=8,
            with_stack=True,
        )
        protocol["formal"] = {"batch": 4, "isl": 256, "osl": 8, "profile_steps": 8}
        formal_results = run_batch(
            args.base_url, batch_size=4, input_len=256, output_len=8, token_seed=211
        )
    elif args.kind == "prefill8k":
        protocol["warmup"].append({"batch": 1, "isl": 512, "osl": 1})
        run_batch(args.base_url, batch_size=1, input_len=512, output_len=1, token_seed=307)
        response = start_profile(
            args.base_url,
            trace_dir,
            profile_id="qwen35-dep4-prefill8k-cgoff",
            num_steps=1,
            with_stack=False,
        )
        protocol["formal"] = {
            "batch": 1,
            "isl": 8192,
            "osl": 1,
            "profile_steps": 1,
            "dp_size": 4,
            "chunked_prefill_size_requested_global": 32768,
            "max_prefill_tokens_requested_global": 32768,
            "chunked_prefill_size_effective_per_dp_rank": 8192,
        }
        formal_results = run_batch(
            args.base_url, batch_size=1, input_len=8192, output_len=1, token_seed=401
        )
    else:
        protocol["warmup"].append({"batch": 32, "isl": 32, "osl": 16})
        run_batch(args.base_url, batch_size=32, input_len=32, output_len=16, token_seed=503)
        with ThreadPoolExecutor(max_workers=1) as executor:
            formal_future = executor.submit(
                run_batch,
                args.base_url,
                batch_size=32,
                input_len=128,
                output_len=64,
                token_seed=607,
            )
            trigger = wait_for_decode_batch(
                args.base_url,
                formal_future,
                expected_ranks=args.expected_ranks,
                batch_size=32,
            )
            response = start_profile(
                args.base_url,
                trace_dir,
                profile_id="qwen35-dep4-decode-cgon-gbs32",
                num_steps=8,
                with_stack=False,
            )
            formal_results = formal_future.result()
        protocol["formal"] = {
            "batch": 32,
            "per_rank_running": [
                rank["num_running_reqs"] for rank in trigger["trigger"]["ranks"]
            ],
            "isl": 128,
            "osl": 64,
            "profile_steps": 8,
            "cuda_graph": True,
            "trigger": trigger,
        }

    protocol["formal"]["response_summary"] = summarize_generation_results(
        formal_results
    )

    traces = wait_for_traces(trace_dir, args.expected_ranks)
    protocol["start_profile_response"] = response
    protocol["trace_files"] = [str(path) for path in traces]
    write_json(args.output_dir / "protocol.json", protocol)
    write_json(
        args.output_dir / "sha256.json",
        {
            "schema_version": "sha256-manifest.v1",
            "files": [
                {"path": str(path.relative_to(args.output_dir)), "sha256": sha256(path)}
                for path in traces
            ],
        },
    )
    print(json.dumps(protocol["formal"], indent=2, sort_keys=True))
    print(f"QWEN35_SGLANG_PROFILE_OK kind={args.kind} traces={len(traces)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
