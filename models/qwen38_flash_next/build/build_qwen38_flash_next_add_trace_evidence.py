#!/usr/bin/env python3
"""Materialize strict add-trace evidence for Qwen3.8-Flash-Next EAGLE MTP.

This builder does not infer Model IR or Execution IR from a trace.  It consumes
the plan produced by ``scripts/run_pipeline_v2.py plan``, reconciles four-rank
graph-off eager stacks, transfers that binding to the selected four-rank
production CUDA Graph windows, and emits the three independently hash-sealed
artifacts required by the generic acceptance gate.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
for root in (REPO_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from llm_arch_v2.add_trace import (  # noqa: E402
    accept_evidence,
    mapping_rules_sha256,
    runtime_identity_sha256,
    sha256_file,
    sha256_json,
    validate_schema,
)
from models.qwen38_flash_next.build.build_qwen38_flash_next_mtp_eager_profile import (  # noqa: E402
    load_jsonl,
    semantic_events,
    transfer_timing,
)
from models.qwen38_flash_next.build.qwen38_flash_next_decode_attribution import (  # noqa: E402
    _is_gemm,
    collective_kind,
    direct_kernel_mapping,
)


TP_SIZE = 4
TOPOLOGY_ID = "tp4_dp1_ep1_flashinfer_gdn_eagle_mtp"
SUPPORT_PRODUCTION_NODES = {
    "qsa_attention.metadata": (
        "framework_runtime",
        "CUDA-Graph-only QSA layout setup has no graph-off kernel counterpart; "
        "it is retained as typed execution support rather than guessed into a Binding",
    ),
    "mtp_generation.draft_select": (
        "sampling_runtime",
        "post-model candidate-tree/cache selection is generation runtime outside "
        "the stable model-forward Binding",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--eager-root", type=Path, required=True)
    parser.add_argument("--production-root", type=Path, required=True)
    parser.add_argument("--eager-protocol", type=Path, required=True)
    parser.add_argument("--production-protocol", type=Path, required=True)
    parser.add_argument("--window-selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-root",
        type=Path,
        default=REPO_ROOT / "catalog" / "qwen38_flash_next",
    )
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected one mapping")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _artifact_record(path: Path, *, evidence_root: Path) -> dict[str, str]:
    resolved = path.resolve()
    root = evidence_root.resolve()
    try:
        stable_path = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"evidence artifact must be contained by the run root {root}: {resolved}"
        ) from exc
    return {"path": stable_path.as_posix(), "sha256": sha256_file(resolved)}


def _rank_paths(root: Path, rank: int) -> dict[str, Path]:
    directory = root / f"tp{rank}"
    return {
        "events": directory / f"events.tp{rank}.jsonl",
        "mapping": directory / f"kernel_mapping.tp{rank}.jsonl",
        "manifest": directory / f"input_manifest.tp{rank}.json",
        "validation": directory / f"validation_report.tp{rank}.json",
    }


def _validate_protocols(
    eager: dict[str, Any], production: dict[str, Any], run: dict[str, Any]
) -> None:
    normalized = run["normalized_config"]
    runtime = normalized["runtime_implementation"]
    execution = normalized["execution_contract"]
    profile = normalized["profile_contract"]
    procedure = normalized["capture_procedure"]
    if eager.get("generation_mode") != "eagle_mtp":
        raise ValueError("eager protocol is not EAGLE MTP")
    if production.get("generation_mode") != "eagle_mtp":
        raise ValueError("production protocol is not EAGLE MTP")
    if eager.get("with_stack") is not True:
        raise ValueError("eager protocol must record Python stacks")
    if production.get("with_stack") is not False:
        raise ValueError("production timing protocol must disable Python stacks")
    if eager.get("mode") != "eager":
        raise ValueError("eager protocol does not declare eager mode")
    if production.get("mode") != "cudagraph":
        raise ValueError("production protocol does not declare CUDA Graph mode")
    for name, protocol in (("eager", eager), ("production", production)):
        if protocol.get("topology") != TOPOLOGY_ID:
            raise ValueError(
                f"{name} protocol is not the exact {TOPOLOGY_ID} execution topology"
            )
        if int(protocol.get("dp_size", -1)) != 1:
            raise ValueError(f"{name} protocol is not exact DP1")
        if protocol.get("capture_phase") != "decode":
            raise ValueError(f"{name} protocol is not a decode capture")
    for identity_field in (
        "source_commit",
        "source_patch_sha256",
        "speculative_algorithm",
        "speculative_num_steps",
        "speculative_eagle_topk",
        "speculative_num_draft_tokens",
    ):
        if eager.get(identity_field) != production.get(identity_field):
            raise ValueError(
                f"eager/production protocol mismatch for {identity_field}"
            )
    expected_shared = {
        "source_commit": runtime["source_commit"],
        "source_patch_sha256": runtime.get("source_patch_sha256"),
        "capture_phase": profile["phase"],
        "generation_mode": execution["generation"]["mode"],
        "global_batch_sizes": [profile["batch_size"]],
        "input_len": profile["isl"],
        "output_len": profile["osl"],
        "warmup_rounds": procedure["warmup_rounds"],
        "formal_rounds": procedure["formal_rounds"],
        "cache_reset_policy": procedure["cache_reset_policy"],
        "speculative_num_steps": execution["generation"][
            "speculative_num_steps"
        ],
        "speculative_eagle_topk": execution["generation"]["speculative_topk"],
        "speculative_num_draft_tokens": execution["generation"][
            "speculative_num_draft_tokens"
        ],
    }
    for name, protocol in (("eager", eager), ("production", production)):
        mismatches = {
            key: {"expected": expected, "observed": protocol.get(key)}
            for key, expected in expected_shared.items()
            if protocol.get(key) != expected
        }
        if mismatches:
            raise ValueError(
                f"{name} protocol differs from normalized run contract: "
                f"{json.dumps(mismatches, sort_keys=True)}"
            )
    expected_modes = {
        "eager": {
            "mode": "eager",
            "with_stack": procedure["eager_with_stack"],
            "formal_profile_steps": procedure["eager_profile_steps"],
            "cuda_graph_decode_backend": "disabled",
            "cuda_graph_prefill_backend": "disabled",
            "cuda_graph_batch_sizes": [],
        },
        "production": {
            "mode": "cudagraph",
            "with_stack": False,
            "formal_profile_steps": procedure["production_profile_steps"],
            "cuda_graph_decode_backend": procedure[
                "production_cuda_graph_backend"
            ],
            "cuda_graph_prefill_backend": procedure[
                "production_cuda_graph_prefill_backend"
            ],
            "cuda_graph_batch_sizes": [profile["batch_size"]],
        },
    }
    for name, protocol in (("eager", eager), ("production", production)):
        mismatches = {
            key: {"expected": expected, "observed": protocol.get(key)}
            for key, expected in expected_modes[name].items()
            if protocol.get(key) != expected
        }
        if mismatches:
            raise ValueError(
                f"{name} protocol differs from normalized capture procedure: "
                f"{json.dumps(mismatches, sort_keys=True)}"
            )


def _runtime_evidence_artifacts(
    *,
    run: dict[str, Any],
    eager_protocol: Path,
    production_protocol: Path,
    evidence_root: Path,
) -> list[dict[str, str]]:
    """Reopen and seal the immutable evidence behind Runtime Identity."""

    normalized = run["normalized_config"]
    runtime = normalized["runtime_implementation"]
    model = normalized["model_contract"]
    execution = normalized["execution_contract"]
    artifact_names = (
        "source-verification.json",
        "model-revision.json",
        "container-identity.tsv",
        "package-lock.txt",
        "runtime-extension-artifacts.json",
        "runtime-versions.json",
        "effective-server-args.json",
    )
    roots = {
        "eager": eager_protocol.resolve().parent.parent,
        "production": production_protocol.resolve().parent.parent,
    }
    artifacts: list[dict[str, str]] = []
    loaded: dict[str, dict[str, Any]] = {}
    immutable_hashes: dict[str, dict[str, str]] = defaultdict(dict)
    for mode, root in roots.items():
        for artifact_name in artifact_names:
            path = root / artifact_name
            if not path.is_file():
                raise ValueError(f"missing {mode} runtime evidence artifact: {path}")
            digest = sha256_file(path)
            record = _artifact_record(path, evidence_root=evidence_root)
            if record["sha256"] != digest:
                raise ValueError(f"runtime artifact changed while being sealed: {path}")
            artifacts.append(record)
            immutable_hashes[artifact_name][mode] = digest
        loaded[f"{mode}_source"] = json.loads(
            (root / "source-verification.json").read_text()
        )
        loaded[f"{mode}_model"] = json.loads(
            (root / "model-revision.json").read_text()
        )
        loaded[f"{mode}_extensions"] = json.loads(
            (root / "runtime-extension-artifacts.json").read_text()
        )
        loaded[f"{mode}_versions"] = json.loads(
            (root / "runtime-versions.json").read_text()
        )
        loaded[f"{mode}_args"] = json.loads(
            (root / "effective-server-args.json").read_text()
        )
        container_identity = (root / "container-identity.tsv").read_text()
        expected_container_values = (
            runtime["container_digest"],
            runtime["build_flags"]["container_index_digest"],
        )
        if any(value not in container_identity for value in expected_container_values):
            raise ValueError(f"{mode} container evidence differs from Runtime Identity")
        if sha256_file(root / "package-lock.txt") != runtime["package_lock_sha256"]:
            raise ValueError(f"{mode} package lock differs from Runtime Identity")

    for artifact_name in artifact_names[:-1]:
        hashes = immutable_hashes[artifact_name]
        if hashes["eager"] != hashes["production"]:
            raise ValueError(
                f"eager/production immutable runtime evidence differs: {artifact_name}"
            )

    expected_extensions = sorted(
        ({"name": item["name"], "sha256": item["sha256"]}
         for item in runtime["extension_artifacts"]),
        key=lambda item: item["name"],
    )
    parallelism = execution["parallelism"]
    generation = execution["generation"]
    for mode in roots:
        source = loaded[f"{mode}_source"]
        if (
            source.get("status") != "pass"
            or source.get("expected_commit") != runtime["source_commit"]
            or source.get("failures")
            or source.get("allowed_patches")
        ):
            raise ValueError(f"{mode} source-tree evidence differs from Runtime Identity")
        if loaded[f"{mode}_model"].get("revision") != model["model_revision"]:
            raise ValueError(f"{mode} checkpoint evidence differs from Model identity")
        extensions = sorted(
            ({"name": item["name"], "sha256": item["sha256"]}
             for item in loaded[f"{mode}_extensions"]),
            key=lambda item: item["name"],
        )
        if extensions != expected_extensions:
            raise ValueError(f"{mode} extension evidence differs from Runtime Identity")
        versions = loaded[f"{mode}_versions"]
        if (
            versions.get("cuda") != runtime["build_flags"]["cuda_runtime"]
            or runtime["build_flags"]["compute_capability"]
            not in versions.get("compute_capabilities", [])
            or int(versions.get("visible_gpu_count", -1)) != parallelism["tp_size"]
        ):
            raise ValueError(f"{mode} build evidence differs from Runtime Identity")
        args = loaded[f"{mode}_args"]
        expected_args = {
            "tp_size": parallelism["tp_size"],
            "dp_size": parallelism["dp_size"],
            "attn_cp_size": parallelism["cp_size"],
            "ep_size": parallelism["ep_size"],
            "pp_size": parallelism["pp_size"],
            "speculative_algorithm": "EAGLE",
            "speculative_num_steps": generation["speculative_num_steps"],
            "speculative_eagle_topk": generation["speculative_topk"],
            "speculative_num_draft_tokens": generation[
                "speculative_num_draft_tokens"
            ],
            "attention_backend": runtime["backend_selections"]["attention"],
            "linear_attn_backend": runtime["backend_selections"][
                "linear_attention"
            ],
            "mamba_backend": runtime["backend_selections"]["mamba"],
            "speculative_moe_runner_backend": runtime["backend_selections"][
                "speculative_moe"
            ],
        }
        mismatches = {
            key: {"expected": expected, "observed": args.get(key)}
            for key, expected in expected_args.items()
            if args.get(key) != expected
        }
        if mismatches:
            raise ValueError(
                f"{mode} effective runtime differs from normalized run contract: "
                f"{json.dumps(mismatches, sort_keys=True)}"
            )
    return artifacts


def _source_frames(event: dict[str, Any]) -> dict[str, int]:
    frames: dict[str, int] = {}
    for index, frame in enumerate(event.get("python_stack") or []):
        if not isinstance(frame, dict):
            continue
        path = str(frame.get("file") or "")
        function = str(frame.get("function") or "")
        if not function or not path.startswith(("sglang/srt/", "sglang/kernels/")):
            continue
        symbol = f"{path}::{function}"
        frames[symbol] = min(index, frames.get(symbol, index))
    return frames


def _source_anchor(per_rank: list[list[dict[str, Any]]]) -> str:
    rank_candidates: list[dict[str, int]] = []
    for events in per_rank:
        candidates: dict[str, int] = {}
        for event in events:
            for symbol, position in _source_frames(event).items():
                candidates[symbol] = min(position, candidates.get(symbol, position))
        if not candidates:
            raise ValueError("Binding rule has no SGLang source frame on one TP rank")
        rank_candidates.append(candidates)
    common = set.intersection(*(set(candidates) for candidates in rank_candidates))
    if not common:
        raise ValueError("Binding rule has no source symbol common to every TP rank")
    return min(
        common,
        key=lambda symbol: (
            sum(candidates[symbol] for candidates in rank_candidates),
            len(symbol),
            symbol,
        ),
    )


def _transfer_method(node: str) -> str:
    if "collective" in node:
        return "collective_order"
    if node.startswith("mtp_generation."):
        return "annotated_scope"
    if "state" in node or node.endswith("context_commit"):
        return "state_boundary"
    return "exact_sequence"


def _event_signature(event: dict[str, Any]) -> str:
    name = str(event.get("kernel_name") or "")
    direct, _label = direct_kernel_mapping(name)
    if (kind := collective_kind(name)) is not None:
        return kind
    if direct is not None:
        return direct
    if _is_gemm(name):
        return "gemm"
    return str(event.get("attribution_method") or "unknown")


def _production_signature(events: list[dict[str, Any]]) -> str:
    # Physical launch order across independent CUDA streams is not a stable
    # Binding identity.  Seal the exact per-step multiset instead: this keeps
    # every occurrence and semantic scope in the contract while making the
    # fingerprint invariant to legal cross-stream interleavings.
    counts: dict[tuple[Any, ...], int] = defaultdict(int)
    for event in events:
        key = (
            int(event["step_index"]),
            event.get("layer_id"),
            event.get("layer_kind"),
            event.get("substage"),
            _event_signature(event),
            "graph" if int(event.get("graph_id") or 0) > 0 else "outside_graph",
        )
        counts[key] += 1
    payload = [
        {
            "step": key[0],
            "layer": key[1],
            "layer_kind": key[2],
            "substage": key[3],
            "signature": key[4],
            "launch_domain": key[5],
            "count": count,
        }
        for key, count in sorted(counts.items(), key=lambda item: repr(item[0]))
    ]
    return sha256_json(payload)


def _build_rules(
    eager_by_rank: list[dict[str, list[dict[str, Any]]]],
    production_by_rank: list[dict[str, list[dict[str, Any]]]],
) -> list[dict[str, Any]]:
    eager_nodes = set(eager_by_rank[0])
    production_nodes = set(production_by_rank[0])
    if any(set(nodes) != eager_nodes for nodes in eager_by_rank):
        raise ValueError("eager Binding node set differs across TP ranks")
    if any(set(nodes) != production_nodes for nodes in production_by_rank):
        raise ValueError("production Binding node set differs across TP ranks")
    if eager_nodes != production_nodes:
        raise ValueError(
            "eager/production Binding node sets differ after typed support removal: "
            f"eager-only={sorted(eager_nodes - production_nodes)} "
            f"production-only={sorted(production_nodes - eager_nodes)}"
        )

    rules = []
    for node in sorted(eager_nodes):
        per_eager = [rank[node] for rank in eager_by_rank]
        per_production = [rank[node] for rank in production_by_rank]
        signatures = [_production_signature(events) for events in per_production]
        if len(set(signatures)) != 1:
            raise ValueError(f"production transfer signature differs by rank for {node}")
        layers = sorted(
            {
                int(event["layer_id"])
                for events in per_production
                for event in events
                if isinstance(event.get("layer_id"), int)
                and int(event["layer_id"]) >= 0
            }
        )
        substages = {
            str(event["substage"])
            for events in per_production
            for event in events
            if event.get("substage")
        }
        scope: dict[str, Any] = {
            "phase": "decode",
            "generation_mode": "eagle_mtp",
        }
        if layers:
            scope["layer_ids"] = layers
        if len(substages) == 1:
            scope["substage"] = next(iter(substages))
        rules.append(
            {
                "rule_id": "eagle_mtp_decode__" + node.replace(".", "__"),
                "ir_target": node,
                "eager_match": {"source_symbol": _source_anchor(per_eager)},
                "production_transfer": {
                    "method": _transfer_method(node),
                    "signature_digest": signatures[0],
                },
                "scope": scope,
            }
        )
    return rules


def _load_eager(
    root: Path, *, evidence_root: Path
) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]]]:
    ranks: list[list[dict[str, Any]]] = []
    artifacts = []
    for rank in range(TP_SIZE):
        paths = _rank_paths(root, rank)
        missing = [path for path in paths.values() if not path.is_file()]
        if missing:
            raise ValueError(f"missing eager rank-{rank} artifacts: {missing}")
        validation = _load(paths["validation"])
        if (
            validation.get("ok") is not True
            or int(validation.get("mapped_kernel_count", -1))
            != int(validation.get("kernel_count", -2))
            or float(validation.get("mapped_duration_ratio", -1.0)) != 1.0
            or validation.get("errors")
        ):
            raise ValueError(f"eager rank {rank} mapper validation did not close")
        namespace = argparse.Namespace(
            phase="decode",
            semantic_events=paths["events"],
            semantic_mapping=paths["mapping"],
            semantic_manifest=paths["manifest"],
        )
        events = semantic_events(namespace)
        if len(events) != len(load_jsonl(paths["events"])):
            raise ValueError(f"eager rank {rank} semantic event coverage changed")
        if any(not event.get("python_stack") for event in events):
            raise ValueError(f"eager rank {rank} contains an event without Python stack")
        for index, event in enumerate(events):
            evidence_event_id = f"k_{index:06d}"
            recorded_event_id = (event.get("stack_evidence") or {}).get("event_id")
            if recorded_event_id is not None and recorded_event_id != evidence_event_id:
                raise ValueError(
                    f"eager rank {rank} event-order identity changed at {index}: "
                    f"{recorded_event_id!r} != {evidence_event_id!r}"
                )
            event["evidence_event_id"] = evidence_event_id
        ranks.append(events)
        artifacts.append(
            {"rank": rank, **_artifact_record(paths["events"], evidence_root=evidence_root)}
        )
    return ranks, artifacts


def _load_production(
    root: Path,
    eager_root: Path,
    eager_ranks: list[list[dict[str, Any]]],
    *,
    evidence_root: Path,
) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]]]:
    ranks: list[list[dict[str, Any]]] = []
    artifacts = []
    for rank in range(TP_SIZE):
        paths = _rank_paths(root, rank)
        for required in (paths["events"], paths["manifest"]):
            if not required.is_file():
                raise ValueError(f"missing production rank-{rank} artifact: {required}")
        timing = load_jsonl(paths["events"])
        semantic_manifest = _load(_rank_paths(eager_root, rank)["manifest"])
        timing_manifest = _load(paths["manifest"])
        if int(timing_manifest.get("rank", -1)) != rank:
            raise ValueError(f"production manifest rank mismatch for TP{rank}")
        events, accounting = transfer_timing(
            eager_ranks[rank], timing, semantic_manifest, timing_manifest
        )
        if len(events) != len(timing):
            raise ValueError(f"production rank {rank} event coverage did not close")
        if any(row.get("attributed_kernel_count") is None for row in accounting):
            raise ValueError(f"production rank {rank} lacks step accounting")
        ranks.append(events)
        artifacts.append(
            {"rank": rank, **_artifact_record(paths["events"], evidence_root=evidence_root)}
        )
    return ranks, artifacts


def _group_by_node(
    ranks: list[list[dict[str, Any]]], *, production: bool
) -> tuple[list[dict[str, list[dict[str, Any]]]], list[list[dict[str, Any]]]]:
    grouped_ranks = []
    support_ranks = []
    for events in ranks:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        support = []
        for event in events:
            node = str(event.get("node") or "")
            if (not production and node == "top.runtime_support") or (
                production and node in SUPPORT_PRODUCTION_NODES
            ):
                support.append(event)
            else:
                grouped[node].append(event)
        if "" in grouped:
            raise ValueError("an event has no IR target and was not typed as support")
        grouped_ranks.append(dict(grouped))
        support_ranks.append(support)
    return grouped_ranks, support_ranks


def build(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    run = _load(args.run)
    plan = _load(args.plan)
    evidence_root = args.run.resolve().parent
    eager_protocol = _load(args.eager_protocol)
    production_protocol = _load(args.production_protocol)
    _validate_protocols(eager_protocol, production_protocol, run)
    runtime_evidence_artifacts = _runtime_evidence_artifacts(
        run=run,
        eager_protocol=args.eager_protocol,
        production_protocol=args.production_protocol,
        evidence_root=evidence_root,
    )
    if plan["run_id"] != run["run_id"] or plan["model_id"] != run["model_id"]:
        raise ValueError("run manifest and add-trace plan identity differ")

    eager_ranks, eager_artifacts = _load_eager(
        args.eager_root, evidence_root=evidence_root
    )
    production_ranks, production_artifacts = _load_production(
        args.production_root,
        args.eager_root,
        eager_ranks,
        evidence_root=evidence_root,
    )
    eager_by_rank, eager_support = _group_by_node(eager_ranks, production=False)
    production_by_rank, production_support = _group_by_node(
        production_ranks, production=True
    )
    rules = _build_rules(eager_by_rank, production_by_rank)
    rules_by_target = {rule["ir_target"]: rule for rule in rules}

    runtime_identity = run["normalized_config"]["runtime_implementation"]
    runtime_digest = runtime_identity_sha256(runtime_identity, source=args.run)
    binding = {
        "schema_version": "binding-revision.v1",
        "binding_revision_id": plan["binding_resolution"]["binding_revision_id"],
        "model_id": plan["model_id"],
        "execution_path_id": plan["execution_resolution"]["execution_path_id"],
        "execution_fingerprint": plan["execution_resolution"]["execution_fingerprint"],
        "runtime_identity": runtime_identity,
        "runtime_identity_sha256": runtime_digest,
        "runtime_evidence_artifacts": runtime_evidence_artifacts,
        "mapping_rules_sha256": mapping_rules_sha256(rules),
        "mapping_rules": rules,
    }

    eager_results = []
    eager_support_records = []
    for rank in range(TP_SIZE):
        for node, events in sorted(eager_by_rank[rank].items()):
            rule = rules_by_target[node]
            eager_results.append(
                {
                    "rule_id": rule["rule_id"],
                    "ir_target": node,
                    "rank": rank,
                    "eager_event_ids": [
                        f"tp{rank}:eager:{event['evidence_event_id']}"
                        for event in events
                    ],
                    "duration_us": sum(float(event["dur_us"]) for event in events),
                    "matched_evidence": rule["eager_match"],
                    "status": "pass",
                }
            )
        for event in eager_support[rank]:
            eager_support_records.append(
                {
                    "event_id": f"tp{rank}:eager:{event['evidence_event_id']}",
                    "rank": rank,
                    "support_class": "framework_runtime",
                    "support_reason": "EAGLE scheduler/cache preparation outside model-forward IR",
                    "duration_us": float(event["dur_us"]),
                }
            )
    eager_total = sum(sum(float(event["dur_us"]) for event in rank) for rank in eager_ranks)
    reconciliation = {
        "schema_version": "binding-reconciliation.v1",
        "run_id": plan["run_id"],
        "model_id": plan["model_id"],
        "execution_path_id": plan["execution_resolution"]["execution_path_id"],
        "execution_fingerprint": plan["execution_resolution"]["execution_fingerprint"],
        "binding_revision_id": binding["binding_revision_id"],
        "plan_sha256": plan["plan_sha256"],
        "status": "pass",
        "cuda_graph_enabled": False,
        "protocol_artifact": {
            **_artifact_record(args.eager_protocol, evidence_root=evidence_root),
        },
        "rank_artifacts": eager_artifacts,
        "rule_results": eager_results,
        "support_events": eager_support_records,
        "coverage": {
            "event_count_total": sum(len(rank) for rank in eager_ranks),
            "event_count_accounted": len(eager_support_records)
            + sum(len(result["eager_event_ids"]) for result in eager_results),
            "duration_us_total": eager_total,
            "duration_us_accounted": eager_total,
        },
        "unresolved": [],
        "discrepancies": [],
    }

    eager_ids = {
        (result["rank"], result["ir_target"]): result["eager_event_ids"]
        for result in eager_results
    }
    mappings = []
    production_support_records = []
    for rank, events in enumerate(production_ranks):
        for index, event in enumerate(events):
            event_id = f"tp{rank}:step{int(event['step_index'])}:p{index:06d}"
            node = str(event["node"])
            if node in SUPPORT_PRODUCTION_NODES:
                support_class, support_reason = SUPPORT_PRODUCTION_NODES[node]
                production_support_records.append(
                    {
                        "event_id": event_id,
                        "rank": rank,
                        "support_class": support_class,
                        "support_reason": support_reason,
                        "duration_us": float(event["dur_us"]),
                    }
                )
                continue
            rule = rules_by_target[node]
            mappings.append(
                {
                    "event_id": event_id,
                    "rank": rank,
                    "ir_target": node,
                    "rule_id": rule["rule_id"],
                    "eager_event_ids": [eager_ids[(rank, node)][0]],
                    "transfer_method": rule["production_transfer"]["method"],
                    "transfer_signature_digest": rule["production_transfer"][
                        "signature_digest"
                    ],
                    "confidence": "exact" if event.get("confidence") == "exact" else "high",
                    "occurrence_id": event_id.replace(":p", ":occurrence"),
                    "duration_us": float(event["dur_us"]),
                }
            )
    production_total = sum(
        sum(float(event["dur_us"]) for event in rank) for rank in production_ranks
    )
    attribution = {
        "schema_version": "trace-attribution.v1",
        "run_id": plan["run_id"],
        "model_id": plan["model_id"],
        "execution_path_id": plan["execution_resolution"]["execution_path_id"],
        "execution_fingerprint": plan["execution_resolution"]["execution_fingerprint"],
        "binding_revision_id": binding["binding_revision_id"],
        "plan_sha256": plan["plan_sha256"],
        "status": "pass",
        "phase": "decode",
        "cuda_graph_enabled": True,
        "protocol_artifact": {
            **_artifact_record(args.production_protocol, evidence_root=evidence_root),
        },
        "rank_artifacts": production_artifacts,
        "window_selection_artifact": {
            **_artifact_record(args.window_selection, evidence_root=evidence_root),
        },
        "event_mappings": mappings,
        "support_events": production_support_records,
        "fusion_groups": [],
        "coverage": {
            "event_count_total": sum(len(rank) for rank in production_ranks),
            "event_count_accounted": len(mappings) + len(production_support_records),
            "duration_us_total": production_total,
            "duration_us_accounted": production_total,
        },
        "unresolved": [],
    }

    for value, schema in (
        (binding, "binding-revision.schema.json"),
        (reconciliation, "binding-reconciliation.schema.json"),
        (attribution, "trace-attribution.schema.json"),
    ):
        validate_schema(value, schema, source=args.run)
    acceptance = accept_evidence(
        run,
        plan,
        binding,
        reconciliation,
        attribution,
        model_root=args.model_root,
        source=args.run,
        verify_files=True,
    )
    return binding, reconciliation, attribution, acceptance


def main() -> int:
    args = parse_args()
    binding, reconciliation, attribution, acceptance = build(args)
    _write(args.output_dir / "binding-revision.json", binding)
    _write(args.output_dir / "eager-reconciliation.json", reconciliation)
    _write(args.output_dir / "trace-attribution.json", attribution)
    _write(args.output_dir / "acceptance.json", acceptance)
    print(json.dumps(acceptance, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
