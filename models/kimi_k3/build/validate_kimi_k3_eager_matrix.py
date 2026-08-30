#!/usr/bin/env python3
"""Fail-closed audit of graph-off Kimi K3 TP8 semantic captures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SGLANG_COMMIT = "25035bff8d34f3fcce2c1a2a5b1fe610225e84ed"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def parse_point(raw: str) -> dict[str, Any]:
    fields = raw.split(",", 4)
    if len(fields) != 5:
        raise argparse.ArgumentTypeError(
            "point must be NAME,PHASE,CONCURRENCY,BASELINE_RELATIVE_STEP,ROOT"
        )
    name, phase, concurrency, step, root = fields
    if phase not in {"prefill", "decode"}:
        raise argparse.ArgumentTypeError(f"unsupported phase {phase!r}")
    return {
        "name": name,
        "phase": phase,
        "concurrency": int(concurrency),
        "baseline_relative_step": int(step),
        "root": Path(root).resolve(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point", type=parse_point, action="append", required=True)
    parser.add_argument("--client-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def validate_client(point: dict[str, Any], client_source: Path) -> dict[str, Any]:
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
    controls = client.get("profile_controls") or []
    require(len(controls) == 1, f"{point['name']}: profile control count")
    request = controls[0].get("request") or {}
    require(controls[0].get("http_status") == 200, f"{point['name']}: profile HTTP")
    require(request.get("num_steps") == 1, f"{point['name']}: profile step count")
    require(request.get("activities") == ["CPU", "GPU"], f"{point['name']}: activities")
    require(request.get("with_stack") is True, f"{point['name']}: stacks")
    require(request.get("record_shapes") is True, f"{point['name']}: shapes")
    coordinate = client.get("profile_coordinate") or {}
    require(
        coordinate.get("baseline_relative_start_step")
        == point["baseline_relative_step"],
        f"{point['name']}: baseline-relative coordinate mismatch",
    )
    require(coordinate.get("warmup_cached_token_count") == 0, f"{point['name']}: cache")
    return {
        "path": str(path),
        "sha256": sha256(path),
        "contract": contract,
        "profile_coordinate": coordinate,
    }


def validate_rank(point: dict[str, Any], rank: int) -> dict[str, Any]:
    root = point["root"]
    traces = sorted((root / "traces").glob(f"*-TP-{rank}.trace.json.gz"))
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
    require(manifest.get("source_commit") == SGLANG_COMMIT, f"{point['name']}: source")
    require(manifest.get("rank") == rank, f"{point['name']}: rank")
    require(manifest.get("phase") == point["phase"], f"{point['name']}: phase")
    require(Path(manifest["trace_path"]).name == traces[0].name, f"{point['name']}: trace")
    require(validation.get("ok") is True, f"{point['name']}: TP{rank} mapping failed")
    require(not validation.get("errors"), f"{point['name']}: TP{rank} errors")
    require(not validation.get("warnings"), f"{point['name']}: TP{rank} warnings")
    require(not validation.get("top_unmatched"), f"{point['name']}: TP{rank} unmatched")
    require(validation.get("mapped_duration_ratio") == 1.0, f"{point['name']}: mapped ratio")
    require(validation.get("stack_duration_ratio") == 1.0, f"{point['name']}: stack ratio")
    nodes = validation.get("nodes") or {}
    counts = {node: int(cell["count"]) for node, cell in nodes.items()}
    require("kda.kda_out" not in counts, f"{point['name']}: generic KDA output shard")
    require("gated_mla.mla_out" not in counts, f"{point['name']}: generic MLA output shard")
    require(counts.get("kda.tp_kda_output_collective") == 69, f"{point['name']}: KDA AR")
    require(counts.get("gated_mla.tp_mla_output_collective") == 24, f"{point['name']}: MLA AR")
    require(counts.get("stable_latent_moe.tp_shared_expert_collective") == 92, f"{point['name']}: shared AR")
    # The router is one semantic matmul per MoE block.  At decode batch 16/64,
    # cuBLASLt realizes each matmul as a Split-K main kernel plus its reduction,
    # so the physical 1:N binding contains two kernels per semantic occurrence.
    expected_router_kernels = 92 if point["phase"] == "prefill" or point["concurrency"] == 1 else 184
    require(
        counts.get("stable_latent_moe.router_logits") == expected_router_kernels,
        f"{point['name']}: router front",
    )
    require(counts.get("stable_latent_moe.routed_up") == 92, f"{point['name']}: routed up")
    if point["phase"] == "prefill":
        require(counts.get("stable_latent_moe.dispatch") == 276, f"{point['name']}: expert dispatch")
        require(counts.get("runtime.step_setup") == 19, f"{point['name']}: setup closure")
        require(counts.get("stable_latent_moe.weighted_reduce") == 92, f"{point['name']}: routed reduction")
        for node in ("kda.q_short_conv", "kda.k_short_conv", "kda.v_short_conv"):
            require(counts.get(node) == 69, f"{point['name']}: {node}")
    else:
        require(counts.get("stable_latent_moe.dispatch") == 92, f"{point['name']}: expert dispatch")
        require(counts.get("runtime.step_setup") == 7, f"{point['name']}: setup closure")
        require("stable_latent_moe.weighted_reduce" not in counts, f"{point['name']}: fused routed reduction")
        require(counts.get("kda.recurrent_update") == 69, f"{point['name']}: fused KDA")
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
        "phase_contract": validation["phase_contract"],
    }


def main() -> int:
    args = parse_args()
    require(len(args.point) == 4, "the SGLang eager matrix must contain four points")
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
                "client": validate_client(point, args.client_source.resolve()),
                "ranks": ranks,
            }
        )
    payload = {
        "schema_version": "kimi-k3-eager-matrix-validation.v1",
        "state": "passed",
        "framework": "sglang",
        "source_commit": SGLANG_COMMIT,
        "checks": {
            "exact_8k_1k_warmup_3c_formal_c": True,
            "pure_tp8_all_rank_trace_set": True,
            "phase_shape_rank_source_exact": True,
            "mapped_kernel_duration_ratio": 1.0,
            "stack_kernel_duration_ratio": 1.0,
            "semantic_occurrence_counts_identical_across_ranks": True,
            "generic_output_shards_absent": True,
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
