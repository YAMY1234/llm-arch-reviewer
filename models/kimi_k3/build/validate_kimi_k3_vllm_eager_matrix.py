#!/usr/bin/env python3
"""Fail-closed audit of graph-off vLLM Kimi K3 TP8 semantic captures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from models.kimi_k3.build.validate_kimi_k3_eager_matrix import (
    load_json,
    parse_point,
    require,
    sha256,
)


VLLM_COMMIT = "680e2177e473ed8dfaa9773f7ead185b369cab46"
ATTN_RES_CALLS_PER_FORWARD = 187


def vllm_worker_trace_pattern(rank: int) -> str:
    """Return the exact torch-profiler name for this locked TP8 worker.

    vLLM records the global worker rank in both the EP and final rank fields
    even though the serving topology has ``ep_size=1``.  Keeping both fields
    rank-exact prevents a trace from another worker from satisfying this gate.
    """

    return f"*_tp{rank}_dcp0_ep{rank}_rank{rank}.*.pt.trace.json.gz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point", type=parse_point, action="append", required=True)
    parser.add_argument("--client-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def validate_vllm_client(
    point: dict[str, Any], client_source: Path
) -> dict[str, Any]:
    concurrency = point["concurrency"]
    path = point["root"] / f"client-c{concurrency}.json"
    require(path.is_file(), f"missing client evidence: {path}")
    client = load_json(path)
    contract = client.get("contract") or {}
    requests = (client.get("warmup") or {}).get("requests", []) + (
        client.get("formal") or {}
    ).get("requests", [])
    require(client.get("state") == "passed", f"{point['name']}: client failed")
    require(contract.get("concurrency") == concurrency, f"{point['name']}: concurrency")
    require(contract.get("isl") == 8192, f"{point['name']}: ISL")
    require(contract.get("osl") == 1024, f"{point['name']}: OSL")
    require(contract.get("warmup_request_count") == 3 * concurrency, f"{point['name']}: warmup")
    require(contract.get("formal_request_count") == concurrency, f"{point['name']}: formal")
    require(contract.get("no_intentionally_shared_prefix") is True, f"{point['name']}: prefix")
    require(len(requests) == 4 * concurrency, f"{point['name']}: request count")
    require(
        all(
            request.get("http_status") == 200
            and request.get("realized_prompt_tokens") == 8192
            and request.get("realized_completion_tokens") == 1024
            for request in requests
        ),
        f"{point['name']}: realized length or HTTP mismatch",
    )
    prompt_hashes = [request.get("prompt_token_sha256") for request in requests]
    require(
        all(prompt_hashes) and len(set(prompt_hashes)) == len(prompt_hashes),
        f"{point['name']}: prompt token streams are not unique",
    )
    require(client_source.is_file(), f"missing client source: {client_source}")
    require(
        (client.get("client_source") or {}).get("sha256") == sha256(client_source),
        f"{point['name']}: client source hash mismatch",
    )
    coordinate = client.get("profile_coordinate") or {}
    require(
        coordinate.get("mode") == "vllm_server_profiler_delay_iterations",
        f"{point['name']}: profiler coordinate mode",
    )
    require(
        coordinate.get("baseline_relative_start_step")
        == point["baseline_relative_step"],
        f"{point['name']}: baseline-relative coordinate mismatch",
    )
    expected_delay = (
        0
        if point["baseline_relative_step"] == 0
        else point["baseline_relative_step"] + 1
    )
    require(
        coordinate.get("profiler_delay_iterations") == expected_delay,
        f"{point['name']}: server profiler delay mismatch",
    )
    require(coordinate.get("warmup_cached_token_count") == 0, f"{point['name']}: cache")
    controls = client.get("profile_controls") or []
    require(
        [row.get("action") for row in controls] == ["start", "stop"],
        f"{point['name']}: controls",
    )
    require(
        all(row.get("http_status") == 200 for row in controls),
        f"{point['name']}: profile HTTP",
    )
    require(
        all(not (row.get("request") or {}) for row in controls),
        f"{point['name']}: profile payload",
    )
    return {
        "path": str(path),
        "sha256": sha256(path),
        "contract": contract,
        "profile_coordinate": coordinate,
    }


def validate_rank(point: dict[str, Any], rank: int) -> dict[str, Any]:
    root = point["root"]
    traces = sorted((root / "traces").glob(vllm_worker_trace_pattern(rank)))
    require(len(traces) == 1, f"{point['name']}: TP{rank} trace count {len(traces)}")
    mapping_root = root / "mapping" / f"tp{rank}"
    manifest_path = mapping_root / "input_manifest.json"
    validation_path = mapping_root / "validation_report.json"
    events_path = mapping_root / f"events.tp{rank}.jsonl"
    mapping_path = mapping_root / f"kernel_mapping.tp{rank}.jsonl"
    for path in (manifest_path, validation_path, events_path, mapping_path):
        require(path.is_file(), f"{point['name']}: missing TP{rank} artifact {path.name}")

    manifest = load_json(manifest_path)
    validation = load_json(validation_path)
    require(manifest.get("source_commit") == VLLM_COMMIT, f"{point['name']}: source")
    require(manifest.get("rank") == rank, f"{point['name']}: rank")
    require(
        manifest.get("phase") == f"vllm_{point['phase']}",
        f"{point['name']}: phase",
    )
    require(Path(manifest["trace_path"]).name == traces[0].name, f"{point['name']}: trace")
    require(validation.get("ok") is True, f"{point['name']}: TP{rank} mapping failed")
    require(not validation.get("errors"), f"{point['name']}: TP{rank} errors")
    require(not validation.get("warnings"), f"{point['name']}: TP{rank} warnings")
    require(not validation.get("top_unmatched"), f"{point['name']}: TP{rank} unmatched")
    require(validation.get("mapped_duration_ratio") == 1.0, f"{point['name']}: mapped ratio")
    require(validation.get("stack_duration_ratio") == 1.0, f"{point['name']}: stack ratio")
    phase_contract = validation.get("phase_contract") or {}
    require(
        phase_contract.get("requested_phase") == point["phase"],
        f"{point['name']}: requested phase",
    )
    require(
        phase_contract.get("execute_context") == f"vllm_{point['phase']}",
        f"{point['name']}: execute context",
    )
    require(
        phase_contract.get("attn_res_occurrence_count")
        == ATTN_RES_CALLS_PER_FORWARD,
        f"{point['name']}: AttnRes occurrence count",
    )
    require(
        phase_contract.get("phase_shape_rank_source_exact") is True,
        f"{point['name']}: exactness contract",
    )
    nodes = validation.get("nodes") or {}
    counts = {node: int(cell["count"]) for node, cell in nodes.items()}
    require("kda.kda_out" not in counts, f"{point['name']}: generic KDA shard")
    require("gated_mla.mla_out" not in counts, f"{point['name']}: generic MLA shard")
    require(
        counts.get("attn_res.weighted_merge") == ATTN_RES_CALLS_PER_FORWARD - 1,
        f"{point['name']}: decoder AttnRes owners",
    )
    require(counts.get("top.output_attn_res") == 1, f"{point['name']}: final AttnRes")
    require(counts.get("top.final_norm", 0) > 0, f"{point['name']}: final norm")
    require(
        counts.get("kda.recurrent_update", 0) > 0,
        f"{point['name']}: KDA recurrence owner",
    )

    return {
        "rank": rank,
        "trace": {"path": str(traces[0]), "sha256": sha256(traces[0])},
        "mapping_manifest_sha256": sha256(manifest_path),
        "events_sha256": sha256(events_path),
        "mapping_sha256": sha256(mapping_path),
        "validation_sha256": sha256(validation_path),
        "kernel_count": validation["kernel_count"],
        "node_counts": counts,
        "window": manifest["window"],
        "phase_contract": phase_contract,
    }


def main() -> int:
    args = parse_args()
    require(len(args.point) == 4, "the vLLM eager matrix must contain four points")
    results = []
    for point in args.point:
        ranks = [validate_rank(point, rank) for rank in range(8)]
        count_sets = {json.dumps(rank["node_counts"], sort_keys=True) for rank in ranks}
        require(len(count_sets) == 1, f"{point['name']}: semantic counts differ by rank")
        results.append(
            {
                "name": point["name"],
                "phase": point["phase"],
                "concurrency": point["concurrency"],
                "baseline_relative_step": point["baseline_relative_step"],
                "client": validate_vllm_client(point, args.client_source.resolve()),
                "ranks": ranks,
            }
        )
    payload = {
        "schema_version": "kimi-k3-eager-matrix-validation.v1",
        "state": "passed",
        "framework": "vllm",
        "source_commit": VLLM_COMMIT,
        "checks": {
            "exact_8k_1k_warmup_3c_formal_c": True,
            "pure_tp8_all_rank_trace_set": True,
            "phase_shape_rank_source_exact": True,
            "mapped_kernel_duration_ratio": 1.0,
            "stack_kernel_duration_ratio": 1.0,
            "semantic_occurrence_counts_identical_across_ranks": True,
            "attn_res_occurrence_closure": True,
            "generic_output_shards_absent": True,
            "final_norm_separate_from_final_attn_res": True,
        },
        "points": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"state": "passed", "points": len(results)}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as error:
        print(json.dumps({"state": "failed", "error": str(error)}, indent=2))
        raise SystemExit(1)
