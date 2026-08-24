#!/usr/bin/env python3
"""Build an exact-batch SGLang AgentX profile from worker-local NSYS reports."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.common.timeline_artifact import (
    build_timeline_artifact,
    write_timeline_artifact,
)
from models.qwen35.profile.build_qwen35_sglang_agentx_profile import (
    LOG_ROW_RE,
    _aggregate_source_metrics,
    _validate_fingerprints,
    parse_benchmark_snapshot,
    worker_identity,
)
from models.qwen35.profile.build_qwen35_sglang_decode_profile import (
    CONTAINER_SHA256,
    MODEL_CONFIG_SHA256,
    MODEL_REVISION,
    RUNTIME_SOURCE_COMMIT,
    SGLANG_DECODE_NODE_STATES,
    SOURCE_COMMIT,
    _metrics_for_rank,
    _validate_step_signatures,
    sha256_file,
)
from models.qwen35.profile.qwen35_graph_mapping import (
    attribution_active_union_ratio,
    attach_graph_stack_evidence,
    load_contextual_eager_signatures,
    load_unique_eager_kernel_signatures,
    map_graph_window,
)
from models.qwen35.profile.qwen35_nsys_mapping import (
    load_sglang_nsys_steps,
    read_nsys_export_metadata,
    sglang_nsys_trace_events,
    validate_sglang_all_rank_capture_integrity,
    validate_sglang_graph_node_stability,
)
from models.qwen35.profile.qwen35_a2a_contract import (
    validate_comparison_workload,
)
from models.qwen35.profile.qwen35_timeline import QWEN35_TIMELINE_TARGETS


DEFAULT_SELECTED_BATCH = 32
PROFILING_SOURCE_COMMIT = "049323294ec36631b9aab74ffa5dac5ff020fdae"
PROFILING_OVERLAY_COMMIT = "cd4701b32d27bab08a54f1c17eb8710bfe343175"
SRT_SLURM_CAPTURE_COMMIT = "fb4476ed42399ca6a160565e1e6a7bb864b9a015"
PROFILER_MANAGER_SHA256 = (
    "e08fd6430c83bffbc47e83cdd8a770891c8208b3fff6126627bc231d8e338eb9"
)
SCHEDULER_SHA256 = "3b65e34e1ee7e142c0f6d4169d1a486c8c8e78b732183b49b36830d42fa4ba79"
NUMA_UTILS_SHA256 = "04c7ca2968edd9fd4fc93d4bd54deddc8e52489afca25e7f959871d62d03c6be"
SCHEDULER_NVTX_SHA256 = (
    "56610ee61c53c39e40fdd6b44c7443140eeb6e25bc499889e70f93a33bf3fcdd"
)
RUNTIME_MANIFEST_SHA256 = (
    "f23af95628fe593cc6d1c140bc3c4a040ddbd224b94c38a7fdaa9b2c8fa46d41"
)
SGLANG_INIT_SHA256 = "2478bab560301dd170038193509d74935b218675a7793623173932bdb7bf500a"
FLASHINFER_NSYS_PATCH_SHA256 = (
    "c5e7c93e02740946c5c346686275279b647d757c878f7959614c6e7a66423473"
)
SYMM_MEM_GATHER_SHA256 = (
    "8a1f8e9a1f13c26b89691eb0dc7bec07595b107778f180d1afa0a93d5e8af9c4"
)
SOURCE_RE = re.compile(r"^w(?P<worker>[01])/r(?P<rank>[0-3])$")
RANK_LOCAL_REPORT_RE = re.compile(
    r"^(?P<hostname>.+)-decode-rank(?P<rank>[0-3])"
    r"(?:\.(?P<capture>\d+))?\.(?:nsys-rep|sqlite)$"
)
OUTER_WORKER_REPORT_RE = re.compile(
    r"^(?P<hostname>.+)_decode_w(?P<worker>[01])_profile_gpu0-1-2-3"
    r"(?:\.\d+)?\.(?:nsys-rep|sqlite)$"
)
EXACT_GATE_RE = re.compile(
    r"DP(?P<rank>\d+) TP\d+ EP\d+\] "
    r"(?:Worker-wide |All-DP )?[Ee]xact running-batch Nsight gate matched: "
    r"(?:reduction=(?P<reduction>all|any|auto|local|rank[0-3]) )?"
    r"batch=(?P<batch>\d+) forward_ct=(?P<forward_ct>\d+)"
    r"(?: (?:local_)?warmup_batches=(?P<warmup_batches>\d+))?"
    r"(?: selected_rank=(?P<selected_rank>None|[0-3]))?"
)
CAPTURE_OBSERVATION_RE = re.compile(
    r"DP(?P<rank>\d+) TP\d+ EP\d+\] Exact-batch Nsight capture observation: "
    r"selected_rank=(?P<selected_rank>[0-3]) batch=(?P<batch>\d+) "
    r"forward_ct=(?P<forward_ct>\d+) capture_index=(?P<capture_index>\d+)"
)
SYNC_READY_RE = re.compile(
    r"DP(?P<rank>\d+) TP\d+ EP\d+\] Exact-batch Nsight sync group ready: "
    r"world_size=(?P<world_size>\d+)"
)
CAPTURE_RANGE_START_RE = re.compile(
    r"DP(?P<rank>\d+) TP\d+ EP\d+\] "
    r"Started Nsight Systems NVTX capture range:"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlites", type=Path, nargs="+", required=True)
    parser.add_argument("--nsys-reports", type=Path, nargs="+", required=True)
    parser.add_argument("--worker-logs", type=Path, nargs=2, required=True)
    parser.add_argument("--fingerprints", type=Path, nargs=2, required=True)
    parser.add_argument("--benchmark-log", type=Path, required=True)
    parser.add_argument("--job-metadata", type=Path, required=True)
    parser.add_argument("--job-id", type=int, required=True)
    parser.add_argument("--selected-batch", type=int, default=DEFAULT_SELECTED_BATCH)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--workload-result", type=Path, required=True)
    parser.add_argument("--eager-mapping", type=Path, required=True)
    parser.add_argument("--eager-trace", type=Path, required=True)
    parser.add_argument("--output-profile", type=Path, required=True)
    parser.add_argument("--output-timeline", type=Path, required=True)
    parser.add_argument("--output-analysis", type=Path, required=True)
    parser.add_argument("--output-mapping", type=Path, required=True)
    return parser.parse_args()


def _report_worker(path: Path) -> int:
    return worker_identity(path)


def _source_coordinates(source: str) -> tuple[int, int]:
    match = SOURCE_RE.fullmatch(source)
    if not match:
        raise ValueError(f"invalid worker/rank source: {source}")
    return int(match.group("worker")), int(match.group("rank"))


def _validate_nsys_report_files(
    paths: list[Path],
    fingerprint_rows: list[dict[str, Any]],
    *,
    expected_ranks: tuple[int, ...] = (0, 1, 2, 3),
    expected_reports_per_source: int = 1,
    expected_reports_by_rank: dict[int, int] | None = None,
) -> dict[tuple[int, int, int], Path]:
    worker_by_hostname = {
        str(row["hostname"]): int(row["worker"]) for row in fingerprint_rows
    }
    reports: dict[tuple[int, int, int], Path] = {}
    for path in paths:
        match = RANK_LOCAL_REPORT_RE.fullmatch(path.name)
        if match is None:
            raise ValueError(f"cannot parse rank-local NSYS report from {path.name}")
        hostname = match.group("hostname")
        if hostname not in worker_by_hostname:
            raise ValueError(
                f"{path}: hostname {hostname!r} has no decode-worker fingerprint"
            )
        source = (
            worker_by_hostname[hostname],
            int(match.group("rank")),
            int(match.group("capture") or 0),
        )
        if source in reports:
            raise ValueError(f"duplicate SGLang rank-local NSYS report for {source}")
        reports[source] = path.resolve()
    expected_sources = {
        (worker, rank) for worker in range(2) for rank in expected_ranks
    }
    actual_sources = {(worker, rank) for worker, rank, _capture in reports}
    if actual_sources != expected_sources:
        raise ValueError(f"incomplete SGLang rank-local sources: {sorted(reports)}")
    source_counts = Counter((worker, rank) for worker, rank, _capture in reports)
    invalid_counts = {}
    for source, count in source_counts.items():
        expected_count = (
            expected_reports_by_rank[source[1]]
            if expected_reports_by_rank is not None
            else expected_reports_per_source
        )
        if count != expected_count:
            invalid_counts[source] = {"actual": count, "expected": expected_count}
    if invalid_counts:
        raise ValueError(
            "SGLang one-step report counts do not match the capture contract: "
            f"{invalid_counts}"
        )
    if any(not path.is_file() or path.stat().st_size <= 0 for path in reports.values()):
        raise ValueError("formal SGLang rank-local NSYS report is missing or empty")
    return reports


def _validate_outer_worker_report_files(
    paths: list[Path],
    fingerprint_rows: list[dict[str, Any]],
    selected_rank_by_worker: dict[int, int],
) -> dict[tuple[int, int, int], Path]:
    """Validate one all-rank process-tree report from each decode worker."""

    worker_by_hostname = {
        str(row["hostname"]): int(row["worker"]) for row in fingerprint_rows
    }
    reports: dict[tuple[int, int, int], Path] = {}
    for path in paths:
        match = OUTER_WORKER_REPORT_RE.fullmatch(path.name)
        if match is None:
            raise ValueError(f"cannot parse outer-worker NSYS report from {path.name}")
        hostname = match.group("hostname")
        if hostname not in worker_by_hostname:
            raise ValueError(
                f"{path}: hostname {hostname!r} has no decode-worker fingerprint"
            )
        worker = int(match.group("worker"))
        if worker_by_hostname[hostname] != worker:
            raise ValueError(
                f"{path}: filename worker {worker} disagrees with fingerprint "
                f"worker {worker_by_hostname[hostname]}"
            )
        source = (worker, selected_rank_by_worker[worker], 0)
        if source in reports:
            raise ValueError(f"duplicate SGLang outer-worker NSYS report for {source}")
        reports[source] = path.resolve()
    expected_sources = {
        (worker, selected_rank, 0)
        for worker, selected_rank in selected_rank_by_worker.items()
    }
    if set(reports) != expected_sources:
        raise ValueError(f"incomplete SGLang outer-worker reports: {sorted(reports)}")
    if any(not path.is_file() or path.stat().st_size <= 0 for path in reports.values()):
        raise ValueError("formal SGLang outer-worker NSYS report is missing or empty")
    return reports


def _profiled_scheduler_ranks(decode_environment: dict[str, Any]) -> tuple[int, ...]:
    rank_spec = str(decode_environment.get("SGLANG_NSYS_SCHEDULER_RANKS") or "").strip()
    if rank_spec:
        raise ValueError(
            "outer-worker SGLang NSYS profile cannot select rank-local scheduler wrappers"
        )
    if str(decode_environment.get("SGLANG_NSYS_PULSE_CAPTURE_PER_STEP") or ""):
        raise ValueError(
            "outer-worker SGLang NSYS profile requires one continuous capture range"
        )
    return (0, 1, 2, 3)


def _validate_nsys_capture_contract(
    profiling: dict[str, Any], decode_environment: dict[str, Any]
) -> tuple[str, str, str]:
    if profiling.get("sglang_scheduler_nsys") is not False:
        raise ValueError(
            "formal SGLang NSYS profile requires one outer worker process-tree wrapper"
        )
    if str(decode_environment.get("NSYS_NVTX_PROFILER_REGISTER_ONLY")) != "0":
        raise ValueError(
            "formal SGLang NSYS profile requires unregistered NVTX capture messages"
        )
    capture_range = str(
        decode_environment.get("SGLANG_NSYS_NVTX_CAPTURE_RANGE") or ""
    ).strip()
    if not capture_range:
        raise ValueError("formal SGLang NSYS profile requires an NVTX capture range")

    extra_args = [str(arg) for arg in (profiling.get("extra_nsys_args") or [])]
    capture_mode = None
    capture_selector = None
    capture_end = None
    for index, arg in enumerate(extra_args):
        if arg in {"-c", "--capture-range"} and index + 1 < len(extra_args):
            capture_mode = extra_args[index + 1]
        elif arg.startswith("-c=") or arg.startswith("--capture-range="):
            capture_mode = arg.split("=", 1)[1]
        if arg in {"-p", "--nvtx-capture"} and index + 1 < len(extra_args):
            capture_selector = extra_args[index + 1]
        elif arg.startswith("-p=") or arg.startswith("--nvtx-capture="):
            capture_selector = arg.split("=", 1)[1]
        if arg == "--capture-range-end" and index + 1 < len(extra_args):
            capture_end = extra_args[index + 1]
        elif arg.startswith("--capture-range-end="):
            capture_end = arg.split("=", 1)[1]

    if capture_mode != "nvtx" or capture_selector != f"{capture_range}@*":
        raise ValueError(
            "formal SGLang NSYS profile requires matching '-c nvtx' and an "
            "explicit all-domain NVTX capture-range selector"
        )
    if capture_end != "repeat:1:async":
        raise ValueError(
            "formal SGLang NSYS profile requires one asynchronously finalized "
            "continuous NVTX capture range"
        )
    if "cudaProfilerApi" in extra_args:
        raise ValueError("formal SGLang NSYS profile cannot use cudaProfilerApi")
    cuda_graph_trace = str(profiling.get("cuda_graph_trace") or "").strip()
    if cuda_graph_trace != "node":
        raise ValueError(
            "formal SGLang NSYS profile requires CUDA Graph node tracing"
        )
    return capture_range, capture_end, cuda_graph_trace


def parse_exact_batch_capture_observations(
    path: Path,
    *,
    selected_batch: int,
    expected_steps: int,
    expected_sync_world_size: int = 4,
    expected_warmup_batches: int = 1,
    expected_gate_reduction: str | None = None,
    expected_gate_ranks: tuple[int, ...] | None = (0, 1, 2, 3),
) -> dict[int, dict[str, Any]]:
    """Prove the runtime sync group and recover the required gate-rank events."""

    lines = path.read_text(errors="replace").splitlines()
    sync_ready: dict[int, int] = {}
    for line in lines:
        match = SYNC_READY_RE.search(line)
        if match is None:
            continue
        rank = int(match.group("rank"))
        if rank in sync_ready:
            raise ValueError(f"{path}: duplicate sync-group proof for DP{rank}")
        sync_ready[rank] = int(match.group("world_size"))
    if set(sync_ready) != {0, 1, 2, 3}:
        raise ValueError(
            f"{path}: incomplete all-DP sync-group proof: {sorted(sync_ready)}"
        )
    invalid_sync = {
        rank: world_size
        for rank, world_size in sync_ready.items()
        if world_size != expected_sync_world_size
    }
    if invalid_sync:
        raise ValueError(
            f"{path}: exact-batch sync group is not world size "
            f"{expected_sync_world_size}: {invalid_sync}"
        )

    gates: dict[int, tuple[int, re.Match[str]]] = {}
    for index, line in enumerate(lines):
        match = EXACT_GATE_RE.search(line)
        if match is None:
            continue
        rank = int(match.group("rank"))
        if expected_gate_ranks is not None and rank not in expected_gate_ranks:
            continue
        if rank in gates:
            raise ValueError(f"{path}: duplicate exact-batch gate for DP{rank}")
        gates[rank] = (index, match)
    if expected_gate_ranks is None:
        selected_ranks = {
            int(match.group("selected_rank"))
            for _index, match in gates.values()
            if match.group("selected_rank") not in {None, "None"}
        }
        if len(selected_ranks) != 1:
            raise ValueError(
                f"{path}: auto gate lacks one worker-local selected rank: "
                f"{sorted(selected_ranks)}"
            )
        selected_rank = next(iter(selected_ranks))
        if selected_rank not in gates:
            raise ValueError(
                f"{path}: selected DP{selected_rank} lacks an exact-batch gate log"
            )
        gates = {selected_rank: gates[selected_rank]}
    elif set(gates) != set(expected_gate_ranks):
        raise ValueError(
            f"{path}: exact-batch gates differ from required ranks "
            f"{list(expected_gate_ranks)}: {sorted(gates)}"
        )

    if expected_gate_reduction == "auto":
        selected_rank = next(iter(gates))
        capture_owner_ranks = [
            int(match.group("rank"))
            for line in lines
            if (match := CAPTURE_RANGE_START_RE.search(line)) is not None
        ]
        if capture_owner_ranks != [selected_rank]:
            raise ValueError(
                f"{path}: outer-worker NVTX range owner must be elected DP"
                f"{selected_rank}, got {capture_owner_ranks}"
            )

    result: dict[int, dict[str, Any]] = {}
    local_warmups: dict[int, int] = {}
    for rank, (gate_index, gate) in sorted(gates.items()):
        gate_batch = int(gate.group("batch"))
        if gate_batch != selected_batch:
            raise ValueError(
                f"{path}: DP{rank} gate selected BS{gate_batch}, expected BS{selected_batch}"
            )
        reduction = gate.group("reduction")
        if expected_gate_reduction is not None and reduction != expected_gate_reduction:
            raise ValueError(
                f"{path}: DP{rank} gate reduction is {reduction!r}, expected "
                f"{expected_gate_reduction!r}"
            )
        warmup_batches = int(gate.group("warmup_batches") or 0)
        local_warmups[rank] = warmup_batches
        prefix = f"DP{rank} TP{rank} EP{rank}]"
        start = next(
            (
                index
                for index, line in enumerate(lines[gate_index + 1 :], gate_index + 1)
                if prefix in line and "Profiling starts." in line
            ),
            None,
        )
        if start is None:
            raise ValueError(f"{path}: DP{rank} gate lacks profiler start")
        stop = next(
            (
                index
                for index, line in enumerate(lines[start + 1 :], start + 1)
                if prefix in line and "Stop profiling..." in line
            ),
            None,
        )
        if stop is None:
            raise ValueError(f"{path}: DP{rank} capture lacks profiler stop")
        done = next(
            (
                index
                for index, line in enumerate(lines[stop + 1 :], stop + 1)
                if prefix in line and "Profiling done." in line
            ),
            None,
        )
        if done is None:
            raise ValueError(f"{path}: DP{rank} capture lacks profiler completion")

        gate_selected_rank = gate.group("selected_rank")
        if gate_selected_rank not in {None, "None"} and int(gate_selected_rank) != rank:
            raise ValueError(
                f"{path}: DP{rank} gate records selected DP{gate_selected_rank}"
            )
        post_forward_rows = []
        for line in lines[start + 1 : stop]:
            match = LOG_ROW_RE.search(line)
            if match is None or int(match.group("rank")) != rank:
                continue
            post_forward_rows.append(
                {
                    "dp_rank": rank,
                    "scheduler_step": int(match.group("step")),
                    "running_requests": int(match.group("running")),
                    "full_tokens": int(match.group("full_tokens")),
                    "accepted_length": float(match.group("accept")),
                    "retracted_requests": int(match.group("retracted")),
                    "cuda_graph": match.group("graph") == "True",
                    "queued_requests": int(match.group("queue")),
                }
            )
        if len(post_forward_rows) != expected_steps:
            raise ValueError(
                f"{path}: DP{rank} expected {expected_steps} captured decode rows, "
                f"found {len(post_forward_rows)}"
            )
        gate_forward_ct = int(gate.group("forward_ct"))
        # The profiler predicate runs immediately before the current forward;
        # the scheduler's post-forward log labels that same captured batch as
        # forward_ct - 1 (confirmed by the live 8K/1K gate evidence).
        first_captured_step = gate_forward_ct - 1
        expected_forward_cts = list(
            range(first_captured_step, first_captured_step + expected_steps)
        )
        actual_forward_cts = [row["scheduler_step"] for row in post_forward_rows]
        if actual_forward_cts != expected_forward_cts:
            raise ValueError(
                f"{path}: DP{rank} captured scheduler steps are not contiguous from "
                f"the exact pre-forward gate: {actual_forward_cts}"
            )

        pre_forward_rows = []
        for line in lines[start + 1 : stop]:
            match = CAPTURE_OBSERVATION_RE.search(line)
            if match is None or int(match.group("rank")) != rank:
                continue
            selected_rank = int(match.group("selected_rank"))
            if selected_rank != rank:
                raise ValueError(
                    f"{path}: DP{rank} capture observation selects DP{selected_rank}"
                )
            pre_forward_rows.append(
                {
                    "capture_index": int(match.group("capture_index")),
                    "pre_forward_running_requests": int(match.group("batch")),
                    "pre_forward_ct": int(match.group("forward_ct")),
                }
            )
        if pre_forward_rows:
            if len(pre_forward_rows) != expected_steps:
                raise ValueError(
                    f"{path}: DP{rank} expected {expected_steps} pre-forward "
                    f"capture observations, found {len(pre_forward_rows)}"
                )
            if [row["capture_index"] for row in pre_forward_rows] != list(
                range(expected_steps)
            ):
                raise ValueError(
                    f"{path}: DP{rank} capture observation indices are not contiguous"
                )
            if [row["pre_forward_ct"] for row in pre_forward_rows] != list(
                range(gate_forward_ct, gate_forward_ct + expected_steps)
            ):
                raise ValueError(
                    f"{path}: DP{rank} pre-forward observations are not contiguous "
                    "from the exact gate"
                )
            observations = [
                {
                    **post,
                    **pre,
                    "post_forward_running_requests": post["running_requests"],
                    "running_requests": pre["pre_forward_running_requests"],
                }
                for pre, post in zip(pre_forward_rows, post_forward_rows)
            ]
            observation_semantics = "pre_forward_runtime_gate"
        else:
            observations = post_forward_rows
            observation_semantics = "legacy_post_forward_scheduler_log"
        exact_observations = [
            row for row in observations if row["running_requests"] == selected_batch
        ]
        invalid_exact = [
            row
            for row in exact_observations
            if not row["cuda_graph"] or row["retracted_requests"]
        ]
        if invalid_exact:
            raise ValueError(
                f"{path}: DP{rank} exact-BS{selected_batch} candidates contain "
                f"retraction or graph-off rows: {invalid_exact[:3]}"
            )
        result[rank] = {
            "gate_forward_ct": gate_forward_ct,
            "sync_world_size": sync_ready[rank],
            "gate_reduction": reduction,
            "selected_rank": rank,
            "capture_owner_rank": rank,
            "local_warmup_batches": warmup_batches,
            "capture_observation_count": len(observations),
            "observation_semantics": observation_semantics,
            "captured_batch_distribution": dict(
                sorted(Counter(row["running_requests"] for row in observations).items())
            ),
            "post_forward_batch_distribution": dict(
                sorted(
                    Counter(
                        row.get("post_forward_running_requests", row["running_requests"])
                        for row in observations
                    ).items()
                )
            ),
            "exact_observation_count": len(exact_observations),
            "observations": observations,
            "profiler_completed": True,
        }
    named_gate = re.fullmatch(r"rank(?P<rank>[0-3])", expected_gate_reduction or "")
    if named_gate is not None:
        gate_rank = int(named_gate.group("rank"))
        warmup_gate_passed = local_warmups.get(gate_rank, 0) >= expected_warmup_batches
    else:
        warmup_gate_passed = (
            max(local_warmups.values(), default=0) >= expected_warmup_batches
        )
    if not warmup_gate_passed:
        raise ValueError(
            f"{path}: exact gate lacks the required rank with "
            f"{expected_warmup_batches} exact-batch warmup(s): {local_warmups}"
        )
    return result


def select_balanced_exact_observations(
    observations_by_worker: dict[int, dict[int, dict[str, Any]]],
    *,
    selected_batch: int,
    sample_count: int,
    allowed_sources: set[str] | None = None,
    min_capture_iteration: int = 0,
) -> list[dict[str, Any]]:
    """Select a deterministic, source-balanced rank-local exact-BS sample."""

    candidates: dict[str, list[dict[str, Any]]] = {}
    for worker, ranks in sorted(observations_by_worker.items()):
        for rank, evidence in sorted(ranks.items()):
            source = f"w{worker}/r{rank}"
            if allowed_sources is not None and source not in allowed_sources:
                continue
            rows = [
                {
                    **row,
                    "worker": worker,
                    "rank": rank,
                    "source": source,
                    "capture_iteration": iteration,
                }
                for iteration, row in enumerate(evidence["observations"])
                if iteration >= min_capture_iteration
                and row["running_requests"] == selected_batch
                and row["cuda_graph"]
                and not row["retracted_requests"]
            ]
            if rows:
                candidates[source] = rows

    available = sum(len(rows) for rows in candidates.values())
    if available < sample_count:
        counts = {source: len(rows) for source, rows in candidates.items()}
        raise ValueError(
            f"SGLang raw capture has only {available} valid rank-local "
            f"BS{selected_batch} observations; need {sample_count}: {counts}"
        )

    quotas: Counter[str] = Counter()
    sources = sorted(candidates)
    while sum(quotas.values()) < sample_count:
        progressed = False
        for source in sources:
            if quotas[source] >= len(candidates[source]):
                continue
            quotas[source] += 1
            progressed = True
            if sum(quotas.values()) == sample_count:
                break
        if not progressed:
            raise AssertionError("exact-observation quota allocation stalled")

    selected: list[dict[str, Any]] = []
    for source in sources:
        rows = candidates[source]
        quota = quotas[source]
        if not quota:
            continue
        # Spread each source's quota over its complete raw capture instead of
        # taking only the earliest contiguous occurrences.
        indices = [
            ((2 * index + 1) * len(rows)) // (2 * quota) for index in range(quota)
        ]
        selected.extend(rows[index] for index in indices)
    selected.sort(
        key=lambda row: (
            int(row["capture_iteration"]),
            int(row["worker"]),
            int(row["rank"]),
        )
    )
    for sample_index, row in enumerate(selected):
        row["sample_index"] = sample_index
    if len(selected) != sample_count:
        raise AssertionError(
            f"selected {len(selected)} samples, expected {sample_count}"
        )
    return selected


def parse_exact_batch_capture_observation(
    path: Path, *, selected_batch: int
) -> dict[str, Any]:
    """Compatibility wrapper for the historical two-step DP0 contract."""

    lines = path.read_text(errors="replace").splitlines()
    gates = [
        (index, match)
        for index, line in enumerate(lines)
        if (match := EXACT_GATE_RE.search(line)) is not None
        and int(match.group("rank")) == 0
    ]
    if len(gates) != 1:
        raise ValueError(f"{path}: expected one DP0 exact-batch gate")
    gate_index, gate = gates[0]
    start = next(
        index
        for index, line in enumerate(lines[gate_index + 1 :], gate_index + 1)
        if "DP0 TP0 EP0] Profiling starts." in line
    )
    stop = next(
        index
        for index, line in enumerate(lines[start + 1 :], start + 1)
        if "DP0 TP0 EP0] Stop profiling..." in line
    )
    done = next(
        (
            index
            for index, line in enumerate(lines[stop + 1 :], stop + 1)
            if "DP0 TP0 EP0] Profiling done." in line
        ),
        None,
    )
    if done is None:
        raise ValueError(f"{path}: missing DP0 profiler completion marker")
    rows = []
    for line in lines[start + 1 : stop]:
        match = LOG_ROW_RE.search(line)
        if match is None:
            continue
        rows.append(
            {
                "dp_rank": int(match.group("rank")),
                "scheduler_step": int(match.group("step")),
                "running_requests": int(match.group("running")),
                "full_tokens": int(match.group("full_tokens")),
                "accepted_length": float(match.group("accept")),
                "retracted_requests": int(match.group("retracted")),
                "cuda_graph": match.group("graph") == "True",
                "queued_requests": int(match.group("queue")),
            }
        )
    gate_forward_ct = int(gate.group("forward_ct"))
    dp0_rows = [row for row in rows if row["dp_rank"] == 0]
    exact_rows = [row for row in dp0_rows if row["scheduler_step"] == gate_forward_ct]
    if len(exact_rows) != 1:
        raise ValueError(f"{path}: DP0 gate step is missing or duplicated")
    exact = exact_rows[0]
    if exact["running_requests"] != selected_batch:
        raise ValueError(f"{path}: DP0 gate step is not exact BS{selected_batch}")
    peer_rows = [row for row in rows if row["scheduler_step"] == gate_forward_ct]
    if {row["dp_rank"] for row in peer_rows} != {0, 1, 2, 3}:
        raise ValueError(f"{path}: historical gate step lacks peer-rank observations")
    return {
        **exact,
        "gate_forward_ct": gate_forward_ct,
        "logged_rows_before_exact": sum(
            row["scheduler_step"] < gate_forward_ct for row in dp0_rows
        ),
        "capture_dp0_observation_count": len(dp0_rows),
        "rank_local_batches_at_exact_step": {
            f"r{row['dp_rank']}": {
                "running_requests": row["running_requests"],
                "full_tokens": row["full_tokens"],
                "accepted_length": row["accepted_length"],
                "queued_requests": row["queued_requests"],
                "retracted_requests": row["retracted_requests"],
                "cuda_graph": row["cuda_graph"],
            }
            for row in sorted(peer_rows, key=lambda item: item["dp_rank"])
        },
        "profiler_completed": True,
    }


def _timing(
    mapped: list[dict[str, Any]], *, logical_period_us: float
) -> dict[str, float]:
    intervals = [
        (
            int(round(float(event["ts_us"]) * 1000.0)),
            int(round((float(event["ts_us"]) + float(event["dur_us"])) * 1000.0)),
        )
        for event in mapped
    ]
    start_ns = min(start for start, _end in intervals)
    end_ns = max(end for _start, end in intervals)
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    active_ns = sum(end - start for start, end in merged)
    residency_us = sum(float(event["dur_us"]) for event in mapped)
    gpu_span_us = (end_ns - start_ns) / 1000.0
    active_us = active_ns / 1000.0
    if active_us > gpu_span_us + 1e-6 or active_us > residency_us + 1e-6:
        raise ValueError("SGLang NSYS timing invariants do not close")
    return {
        "logical_step_period_us": logical_period_us,
        "gpu_span_us": gpu_span_us,
        "gpu_busy_union_us": active_us,
        "gpu_residency_us": residency_us,
    }


def build(args: argparse.Namespace):
    selected_batch = int(getattr(args, "selected_batch", DEFAULT_SELECTED_BATCH))
    if selected_batch < 1 or selected_batch > 80:
        raise ValueError(
            f"selected SGLang batch must be in 1..80, got {selected_batch}"
        )
    profile_id = (
        "qwen35_sglang_attention_dp4_moe_ep4_mtp6_agentx_nsys_" f"bs{selected_batch}"
    )
    comparison_contract, workload_evidence = validate_comparison_workload(
        engine="sglang",
        config=args.config,
        dataset=args.dataset,
        dataset_manifest=args.dataset_manifest,
        workload_result=args.workload_result,
    )
    config = yaml.safe_load(args.config.read_text())
    job = json.loads(args.job_metadata.read_text())
    if int(job.get("job_id", -1)) != args.job_id:
        raise ValueError(
            f"job metadata ID {job.get('job_id')} does not match --job-id {args.job_id}"
        )
    actual_source = ((config.get("identity") or {}).get("frameworks") or {}).get(
        "sglang_source"
    )
    if actual_source != PROFILING_SOURCE_COMMIT:
        raise ValueError(
            f"SGLang profiling source mismatch: expected {PROFILING_SOURCE_COMMIT}, "
            f"got {actual_source}"
        )
    profiling = config.get("profiling") or {}
    decode_environment = (config.get("backend") or {}).get("decode_environment") or {}
    if profiling.get("type") != "nsys":
        raise ValueError("formal SGLang matched profile requires profiling.type=nsys")
    if str(decode_environment.get("SGLANG_ENABLE_NVTX_SCHEDULER")) not in {
        "1",
        "true",
        "True",
    }:
        raise ValueError("formal SGLang NSYS profile requires scheduler NVTX")
    capture_range, capture_range_end, cuda_graph_trace = _validate_nsys_capture_contract(
        profiling, decode_environment
    )
    profiled_ranks = _profiled_scheduler_ranks(decode_environment)
    if str(decode_environment.get("SGLANG_USE_SYMM_MEM_DP_SYNC")).lower() not in {
        "1",
        "true",
    }:
        raise ValueError(
            "formal worker-balanced SGLang profile requires the production "
            "symmetric-memory scheduler metadata sync"
        )
    decode_profiling = profiling.get("decode") or {}
    capture_start_step = int(decode_profiling.get("start_step", -1))
    capture_stop_step = int(decode_profiling.get("stop_step", -1))
    if capture_start_step < 0 or capture_stop_step <= capture_start_step:
        raise ValueError(
            "formal SGLang NSYS profile requires a valid decode scheduler-step window"
        )

    eager_signatures = load_unique_eager_kernel_signatures(args.eager_mapping)
    contextual_signatures = load_contextual_eager_signatures(
        trace_path=args.eager_trace.resolve(),
        mapping_path=args.eager_mapping.resolve(),
    )
    fingerprint_rows = _validate_fingerprints(args.fingerprints)
    benchmark = parse_benchmark_snapshot(args.benchmark_log)
    expected_selected_samples = int(comparison_contract["selected_rank_local_samples"])
    expected_samples_per_source = int(
        comparison_contract["selected_samples_per_source"]
    )
    raw_capture_steps = int(
        decode_environment.get("SGLANG_NSYS_EXACT_DECODE_BATCHES", -1)
    )
    if raw_capture_steps - 1 < expected_samples_per_source:
        raise ValueError(
            "SGLang clean NSYS capture must be at least as wide as the selected "
            "per-source sample after excluding the setup report: "
            f"clean={raw_capture_steps - 1}, selected={expected_samples_per_source}"
        )
    if capture_stop_step - capture_start_step != raw_capture_steps:
        raise ValueError(
            "SGLang scheduler-step window must equal the continuous capture width: "
            f"window={capture_stop_step - capture_start_step}, raw={raw_capture_steps}"
        )
    exact_observations = {
        _report_worker(path): parse_exact_batch_capture_observations(
            path,
            selected_batch=selected_batch,
            expected_steps=raw_capture_steps,
            expected_warmup_batches=int(
                decode_environment.get("SGLANG_NSYS_EXACT_WARMUP_BATCHES", 1)
            ),
            expected_gate_reduction=str(
                decode_environment.get("SGLANG_NSYS_EXACT_GATE_REDUCTION", "")
            ),
            expected_gate_ranks=None,
        )
        for path in args.worker_logs
    }
    if set(exact_observations) != {0, 1}:
        raise ValueError(f"incomplete SGLang worker logs: {sorted(exact_observations)}")
    if any(len(ranks) != 1 for ranks in exact_observations.values()):
        raise ValueError(
            "SGLang auto gate must elect exactly one representative rank per worker: "
            f"{ {worker: sorted(ranks) for worker, ranks in exact_observations.items()} }"
        )

    selected_rank_by_worker = {
        worker: next(iter(ranks))
        for worker, ranks in sorted(exact_observations.items())
    }
    reports = _validate_outer_worker_report_files(
        args.sqlites, fingerprint_rows, selected_rank_by_worker
    )
    nsys_reports = _validate_outer_worker_report_files(
        args.nsys_reports, fingerprint_rows, selected_rank_by_worker
    )
    if set(nsys_reports) != set(reports):
        raise ValueError("SGLang SQLite and raw rank-local report sources differ")
    report_sources = {
        f"w{worker}/r{rank}" for worker, rank, _capture in sorted(reports)
    }
    comparison_sources = {
        f"w{worker}/r{rank}"
        for worker, rank in sorted(selected_rank_by_worker.items())
    }
    if report_sources != comparison_sources:
        raise ValueError(
            "SGLang reports must contain two outer-worker process-tree sources "
            "owned by the runtime-elected exact-BS32 ranks: "
            f"{sorted(report_sources)}"
        )
    selected_sample_rows = select_balanced_exact_observations(
        exact_observations,
        selected_batch=selected_batch,
        sample_count=expected_selected_samples,
        allowed_sources=comparison_sources,
        min_capture_iteration=1,
    )
    selected_sample_by_key = {
        (int(row["worker"]), int(row["rank"]), int(row["capture_iteration"])): row
        for row in selected_sample_rows
    }

    nsys_export_metadata = {
        f"w{worker}/r{rank}/c{capture:02d}": read_nsys_export_metadata(path)
        for (worker, rank, capture), path in sorted(reports.items())
    }
    if len({tuple(sorted(row.items())) for row in nsys_export_metadata.values()}) != 1:
        raise ValueError(
            f"SGLang workers use different Nsight exporters: {nsys_export_metadata}"
        )
    rank_local_integrity = {
        f"w{worker}/r{rank}/c{capture:02d}": validate_sglang_all_rank_capture_integrity(
            path,
            capture_range_label=capture_range,
        )
        for (worker, rank, capture), path in sorted(reports.items())
    }
    worker_capture_integrity: dict[int, dict[str, Any]] = {}
    for worker in range(2):
        selected_rank = selected_rank_by_worker[worker]
        row = rank_local_integrity[f"w{worker}/r{selected_rank}/c00"]
        ranks = row["ranks"]
        worker_capture_integrity[worker] = {
            "capture_scope": "outer_worker_process_tree_continuous_range",
            "profiled_rank_count": len(ranks),
            "participating_rank_count": 4,
            "profiled_ranks": list(profiled_ranks),
            "worker_report_count": 1,
            "captured_model_step_count": row[
                "consistent_graph_bearing_scheduler_marker_count"
            ],
            "consistent_graph_bearing_scheduler_marker_count": row[
                "consistent_graph_bearing_scheduler_marker_count"
            ],
            "ranks": ranks,
        }
    structurally_validated_sources = [
        f"w{worker}/r{rank}"
        for worker in sorted(worker_capture_integrity)
        for rank in profiled_ranks
    ]

    source_metrics: dict[str, dict[str, Any]] = {}
    source_validation: dict[str, dict[str, Any]] = {}
    source_selected_counts: dict[str, int] = {}
    all_mappings: list[dict[str, Any]] = []
    selected_observations: list[dict[str, Any]] = []
    selected_steps_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    timing_by_iteration: dict[int, list[dict[str, Any]]] = defaultdict(list)
    reports_by_source: dict[tuple[int, int], list[tuple[int, Path]]] = defaultdict(list)
    for (worker, selected_rank, capture), path in reports.items():
        reports_by_source[(worker, selected_rank)].append((capture, path))

    for (worker, rank), capture_paths in sorted(reports_by_source.items()):
        source = f"w{worker}/r{rank}"
        step_rows: list[tuple[int, Any]] = []
        parser_evidence = []
        for capture, path in sorted(capture_paths):
            report_steps, report_evidence = load_sglang_nsys_steps(
                path,
                rank=rank,
                capture_range_label=capture_range,
            )
            if len(report_steps) != raw_capture_steps:
                raise ValueError(
                    f"{path}: expected {raw_capture_steps} complete NSYS steps "
                    "inside the continuous outer-worker capture range, "
                    f"found {len(report_steps)}"
                )
            parser_evidence.append(
                {
                    "capture": capture,
                    "file": path.name,
                    "setup_contaminated_step_index": 0,
                    **report_evidence,
                }
            )
            step_rows.extend(enumerate(report_steps[1:], start=1))
        if len(step_rows) != raw_capture_steps - 1:
            raise ValueError(
                f"{source}: expected {raw_capture_steps - 1} clean NSYS steps "
                "after the setup-conservative first-step exclusion, "
                f"found {len(step_rows)}"
            )
        gate = exact_observations[worker][rank]
        observations = gate["observations"]
        selected_iterations = {
            iteration
            for candidate_worker, candidate_rank, iteration in selected_sample_by_key
            if candidate_worker == worker and candidate_rank == rank
        }
        graph_stability = (
            validate_sglang_graph_node_stability(
                step
                for iteration, step in step_rows
                if iteration in selected_iterations
            )
            if selected_iterations
            else {
                "validated": False,
                "reason": "rank captured for collective symmetry but not selected",
            }
        )
        source_mappings: list[dict[str, Any]] = []
        validation_rows = []

        for iteration, step in step_rows:
            row = observations[iteration]
            sample = selected_sample_by_key.get((worker, rank, iteration))
            if sample is None:
                continue
            sample_index = int(sample["sample_index"])
            trace_events, window, graph_roles = sglang_nsys_trace_events(
                step, batch_size=int(row["running_requests"])
            )
            mapped, validation = map_graph_window(
                trace_events,
                window=window,
                rank=rank,
                step_index=iteration,
                eager_signatures=eager_signatures,
                contextual_signatures=contextual_signatures,
            )
            _validate_step_signatures(validation, rank=rank, step=iteration)
            for event in mapped:
                event["event_id"] = f"w{worker}-r{rank}-{event['event_id']}"
                event["worker"] = worker
                event["scheduler_step"] = row["scheduler_step"]
                event["gate_forward_ct"] = gate["gate_forward_ct"]
                event["capture_iteration"] = iteration
                event["selected_sample_index"] = sample_index
            timing = _timing(mapped, logical_period_us=step.cpu_wall_us)
            validation_rows.append(
                {
                    **validation,
                    **timing,
                    "graph_roles": graph_roles,
                    "scheduler_step": row["scheduler_step"],
                    "gate_forward_ct": gate["gate_forward_ct"],
                    "capture_iteration": iteration,
                    "selected_sample_index": sample_index,
                }
            )
            selected_observations.append(
                {
                    "worker": worker,
                    "rank": rank,
                    "source": source,
                    "selected_sample_index": sample_index,
                    "capture_iteration": iteration,
                    "scheduler_step": row["scheduler_step"],
                    "gate_forward_ct": gate["gate_forward_ct"],
                    "running_requests": row["running_requests"],
                    "post_forward_running_requests": row.get(
                        "post_forward_running_requests", row["running_requests"]
                    ),
                    "full_tokens": row["full_tokens"],
                    "mean_full_tokens_per_request": (
                        row["full_tokens"] / row["running_requests"]
                    ),
                    "accepted_length": row["accepted_length"],
                    "retracted_requests": row["retracted_requests"],
                    **timing,
                }
            )
            timing_by_iteration[sample_index].append({"source": source, **timing})
            selected_steps_by_source[source].append(
                {
                    "step_index": sample_index,
                    "capture_iteration": iteration,
                    "trace_start_us": min(float(event["ts_us"]) for event in mapped),
                    "timing": timing,
                    "mapped": mapped,
                }
            )
            source_mappings.extend(mapped)
            all_mappings.extend(mapped)

        source_selected_counts[source] = len(selected_iterations)
        source_metrics[source] = (
            _metrics_for_rank(source_mappings, len(selected_iterations))
            if selected_iterations
            else {}
        )
        source_validation[source] = {
            "parser": parser_evidence,
            "graph_node_stability": graph_stability,
            "exact_gate": gate,
            "captured_batch_distribution": gate["captured_batch_distribution"],
            "selected_exact_sample_count": len(selected_iterations),
            "selected_steps": validation_rows,
        }

    selected_sources = {
        source: count for source, count in source_selected_counts.items() if count
    }
    if not selected_sources:
        distributions = {
            source: evidence["captured_batch_distribution"]
            for source, evidence in source_validation.items()
        }
        raise ValueError(
            f"SGLang NSYS capture has no exact BS{selected_batch} step; "
            f"captured distributions={distributions}"
        )
    if sum(selected_sources.values()) != expected_selected_samples:
        raise ValueError(
            "SGLang exact-event selector did not close: "
            f"expected={expected_selected_samples}, selected={selected_sources}"
        )
    reference_source = min(
        selected_sources,
        key=lambda source: (-selected_sources[source], source),
    )
    reference_worker, reference_rank = _source_coordinates(reference_source)
    reference_steps = [
        {
            "step_index": row["step_index"],
            "label": f"Exact 8K/1K steady decode · NSYS BS{selected_batch}",
            "trace_start_us": row["trace_start_us"],
            "duration_us": row["timing"]["gpu_span_us"],
            "events": attach_graph_stack_evidence(
                row["mapped"], mapping_path=args.eager_mapping
            ),
        }
        for row in selected_steps_by_source[reference_source]
    ]

    critical_steps: dict[str, dict[str, Any]] = {}
    for iteration, rows in sorted(timing_by_iteration.items()):
        selected = max(rows, key=lambda row: float(row["gpu_span_us"]))
        critical_steps[f"iteration-{iteration:02d}"] = selected
    critical_gpu_span_ms = [
        float(row["gpu_span_us"]) / 1000.0 for row in critical_steps.values()
    ]
    logical_period_ms = [
        float(row["logical_step_period_us"]) / 1000.0 for row in critical_steps.values()
    ]
    timing_summary = {
        "semantics": (
            "each sample is one real rank-local BS32 CUDA Graph decode period, "
            "bounded by consecutive scheduler.run_batch markers inside one "
            f"continuous worker-local NVTX capture; {expected_selected_samples} "
            f"samples are balanced {expected_samples_per_source}/worker across "
            f"{sorted(comparison_sources)} and "
            "parallel ranks are never summed"
        ),
        "critical_steps": critical_steps,
        "critical_gpu_span_ms": {
            "samples": critical_gpu_span_ms,
            "mean": statistics.fmean(critical_gpu_span_ms),
            "median": statistics.median(critical_gpu_span_ms),
            "min": min(critical_gpu_span_ms),
            "max": max(critical_gpu_span_ms),
        },
        "critical_logical_period_ms": {
            "samples": logical_period_ms,
            "mean": statistics.fmean(logical_period_ms),
            "median": statistics.median(logical_period_ms),
            "min": min(logical_period_ms),
            "max": max(logical_period_ms),
        },
    }

    total_us = sum(float(row["dur_us"]) for row in all_mappings)
    status_us: Counter[str] = Counter()
    for row in all_mappings:
        status_us[str(row["mapping_status"])] += float(row["dur_us"])
    attributed_residency_ratio = (status_us["mapped"] + status_us["fusion"]) / total_us
    attributed_active_ratio = attribution_active_union_ratio(all_mappings)
    strict_signature_us = sum(
        float(row["dur_us"])
        for row in all_mappings
        if row.get("attribution_method") == "unique_kernel_signature"
    )

    reference_capture_paths = [
        path
        for _capture, path in sorted(
            reports_by_source[(reference_worker, reference_rank)]
        )
    ]
    reference_iterations = {
        int(row["capture_iteration"])
        for row in selected_steps_by_source[reference_source]
    }
    reference_paths = [
        path
        for iteration, path in enumerate(reference_capture_paths)
        if iteration in reference_iterations
    ]
    reference_digest = hashlib.sha256(
        "\n".join(sha256_file(path) for path in reference_paths).encode()
    ).hexdigest()
    timeline = build_timeline_artifact(
        profile_id=profile_id,
        phase="decode",
        reference_rank=reference_rank,
        steps=reference_steps,
        timing_summary=timing_summary,
        raw_trace={
            "file": f"{len(reference_paths)} one-step Nsight Systems exports",
            "files": [path.name for path in reference_paths],
            "sha256": reference_digest,
            "sha256_semantics": "sha256 of newline-joined per-file SHA256 digests",
            "format": "Nsight Systems SQLite export",
            "worker": reference_worker,
            "rank": reference_rank,
        },
        stack_source={
            "mode": "eager_stack_calibration_plus_nsys_graph_node_replay",
            "file": args.eager_mapping.name,
            "sha256": sha256_file(args.eager_mapping),
            "mapped_residency_ratio": round(attributed_residency_ratio, 6),
            "mapped_active_union_ratio": round(attributed_active_ratio, 6),
            "unmapped_residency_ratio": round(status_us["unmapped"] / total_us, 6),
            "policy": (
                "unique eager signature or validated graph role/layer/round slot; "
                "ambiguous occurrences remain unmapped with candidates"
            ),
        },
        target_resolver=QWEN35_TIMELINE_TARGETS,
    )

    selected_acceptance = [
        float(row["accepted_length"]) for row in selected_observations
    ]
    measured_acceptance_mean = statistics.fmean(selected_acceptance)
    if abs(measured_acceptance_mean - 4.8) > 0.05:
        raise ValueError(
            "SGLang captured accept length does not match forced mean 4.8: "
            f"{measured_acceptance_mean}"
        )
    selected_full_tokens = [int(row["full_tokens"]) for row in selected_observations]
    selected_mean_tokens = [
        float(row["mean_full_tokens_per_request"]) for row in selected_observations
    ]
    selected_queue_requests = [
        int(row["queued_requests"]) for row in selected_observations
    ]
    selected_post_forward_requests = [
        int(row["post_forward_running_requests"]) for row in selected_observations
    ]
    node_metrics = _metrics_for_rank(all_mappings, expected_selected_samples)
    for cell in node_metrics.values():
        cell["ms_per_iter"] = round(float(cell["ms_per_iter"]), 6)
        cell["aggregation"] = (
            "mean kernel residency over "
            f"{expected_selected_samples} balanced rank-local BS32 samples"
        )
        cell["source_worker_rank"] = "balanced_pool"
        cell["selected_samples_by_source"] = dict(sorted(selected_sources.items()))
    profile = {
        "schema_version": "profile.v2",
        "profile_id": profile_id,
        "label": (
            "Qwen3.5 397B · SGLang · exact 8K/1K C704 · DEP4 + MTP6 · "
            f"NSYS {expected_selected_samples}×BS{selected_batch}"
        ),
        "model_id": "qwen35_397b_a17b",
        "execution_path_id": "attention_dp4_moe_ep4",
        "implementation_id": "sglang_85c23c62_attention_dp4_moe_ep4_mtp",
        "variant_id": (
            "sglang_agentx_8k1k_c704_3p2d_dep4_mtp6_cg_nsys_" f"bs{selected_batch}"
        ),
        "phase": "decode",
        "generation_mode": "mtp",
        "entry_view": "top",
        "execution_parameters": {
            "tp_size": 1,
            "dp_size": 4,
            "cp_size": 1,
            "ep_size": 4,
        },
        "hardware": {
            "gpu": "GB300",
            "gpus_per_worker": 4,
            "prefill_workers": 3,
            "decode_workers": 2,
        },
        "workload": {
            "scenario": "exact-8k1k",
            "concurrency": 704,
            "comparison_contract": comparison_contract,
            "selected_exact_target_verify_batch": selected_batch,
            "selected_samples": sum(source_selected_counts.values()),
            "selected_samples_by_source": dict(sorted(selected_sources.items())),
            "selected_worker_rank_sources": sorted(selected_sources),
            "selected_post_forward_running_requests": {
                "semantics": (
                    "post-forward scheduler count retained separately from the "
                    "authoritative pre-forward BS32 gate observation"
                ),
                "samples": selected_post_forward_requests,
                "min": min(selected_post_forward_requests),
                "median": statistics.median(selected_post_forward_requests),
                "max": max(selected_post_forward_requests),
            },
            "structurally_validated_worker_rank_sources": structurally_validated_sources,
            "selected_rank_local_full_tokens": {
                "semantics": (
                    "scheduler #full token for the exact selected-batch "
                    "rank-local step"
                ),
                "samples": selected_full_tokens,
                "min": min(selected_full_tokens),
                "median": statistics.median(selected_full_tokens),
                "max": max(selected_full_tokens),
            },
            "selected_mean_full_tokens_per_request": {
                "semantics": "rank-local #full token divided by actual running requests",
                "samples": selected_mean_tokens,
                "min": min(selected_mean_tokens),
                "median": statistics.median(selected_mean_tokens),
                "max": max(selected_mean_tokens),
            },
            "accepted_length": {
                "control": "SGLANG_SIMULATE_ACC_LEN=4.80",
                "evidence": "scheduler decode-batch log for every captured source/iteration",
                "histogram": {
                    str(value): count
                    for value, count in sorted(Counter(selected_acceptance).items())
                },
                "mean": measured_acceptance_mean,
                "min": min(selected_acceptance),
                "median": statistics.median(selected_acceptance),
                "max": max(selected_acceptance),
            },
            "queue_requests": {
                "semantics": (
                    "rank-local scheduler waiting queue under the production "
                    "C704 saturation load; recorded but outside the BS32 CUDA "
                    "Graph input shape"
                ),
                "samples": selected_queue_requests,
                "min": min(selected_queue_requests),
                "median": statistics.median(selected_queue_requests),
                "max": max(selected_queue_requests),
            },
            "retracted_requests": 0,
        },
        "profiler": {
            "type": "nsight_systems_worker_local",
            "rank": (
                "one outer Nsight process traces all four DEP scheduler children; "
                "each worker elects its first exact-BS32 DP rank before the NVTX "
                f"range; {raw_capture_steps} raw steps are recorded and the final "
                f"{expected_selected_samples}-sample pool selects "
                f"{expected_samples_per_source} clean BS32 steps per worker"
            ),
            "trace": ["cuda", "nvtx"],
            "capture_trigger": "nvtx_on_runtime_elected_exact_bs32_rank",
            "capture_range": capture_range,
            "capture_range_end": capture_range_end,
            "cuda_graph_trace": cuda_graph_trace,
            "capture_range_api": "torch.cuda.nvtx.range_start/range_end",
            "process_scope": (
                "one Nsight process wraps each Dynamo/SGLang decode worker with "
                "trace-fork-before-exec enabled, so its four scheduler children and CUDA "
                "devices share one process-tree report"
            ),
            "capture_report_layout": (
                "one all-rank report per decode worker containing one continuous "
                f"{raw_capture_steps}-step range; the first representative-rank step is "
                f"excluded conservatively, leaving {raw_capture_steps - 1} clean model "
                "steps per comparison source"
            ),
            "scheduler_metadata_sync": (
                "production symmetric-memory scheduler metadata gather with its "
                "standard 60-second timeout; the profiling gate uses a CPU all-gather "
                "only before the measured NVTX range"
            ),
            "capture_finalize_gpu_synchronize": True,
            "capture_completion": "one_continuous_nvtx_range_finalized_asynchronously",
            "flashinfer_moe_a2a_peer_wait": (
                "profiling-only JIT of the production source with DISABLE_TIMEOUT=1; "
                "removes only the 300-second device trap while NSYS expands graph nodes"
            ),
            "nvtx_registered_strings_only": False,
            "scheduler_capture_steps": {
                "start_inclusive": capture_start_step,
                "stop_exclusive": capture_stop_step,
            },
            "exact_capture_stop_policy": {
                "rebased_forward_count_width": capture_stop_step - capture_start_step,
                "completed_raw_decode_batches_per_rank": raw_capture_steps,
                "clean_model_steps_per_selected_rank": raw_capture_steps - 1,
                "setup_conservative_steps_excluded_per_selected_rank": 1,
                "outer_worker_reports": 2,
                "selected_rank_local_exact_samples": expected_selected_samples,
                "condition": (
                    "one rank per worker reached BS32 and was elected before capture; "
                    f"only its logged pre-forward BS32 occurrences in the complete "
                    f"{raw_capture_steps}-step raw window are selected. The enclosing "
                    "outer-worker Nsight session records all four ranks without "
                    "capture-stop barriers between model collectives"
                ),
                "external_stop_required": False,
            },
            "cuda_graph_enabled": True,
            "gpu_metric_semantics": (
                "per-node kernel residency is averaged over one balanced pool of "
                f"{expected_selected_samples} rank-local BS32 samples; parallel ranks "
                "and workers are never summed"
            ),
            "attribution_calibration": "graph-off eager Torch stack",
        },
        "evidence": {
            "job_id": args.job_id,
            "source_commit": SOURCE_COMMIT,
            "runtime_source_commit": RUNTIME_SOURCE_COMMIT,
            "profiling_source_commit": PROFILING_SOURCE_COMMIT,
            "profiling_overlay_commit": PROFILING_OVERLAY_COMMIT,
            "profiling_harness_commit": SRT_SLURM_CAPTURE_COMMIT,
            "profiler_manager_sha256": PROFILER_MANAGER_SHA256,
            "scheduler_sha256": SCHEDULER_SHA256,
            "numa_utils_sha256": NUMA_UTILS_SHA256,
            "scheduler_nvtx_sha256": SCHEDULER_NVTX_SHA256,
            "sglang_init_sha256": SGLANG_INIT_SHA256,
            "flashinfer_nsys_patch_sha256": FLASHINFER_NSYS_PATCH_SHA256,
            "runtime_manifest_sha256": RUNTIME_MANIFEST_SHA256,
            "symm_mem_gather_sha256": SYMM_MEM_GATHER_SHA256,
            "model_revision": MODEL_REVISION,
            "model_config_sha256": MODEL_CONFIG_SHA256,
            "container_sha256": CONTAINER_SHA256,
            "config_file": args.config.name,
            "config_sha256": sha256_file(args.config),
            "job_metadata_file": args.job_metadata.name,
            "job_metadata_sha256": sha256_file(args.job_metadata),
            "benchmark_log_file": args.benchmark_log.name,
            "benchmark_log_sha256": sha256_file(args.benchmark_log),
            "benchmark_snapshot": benchmark,
            "comparison_workload": workload_evidence,
            "fingerprints": fingerprint_rows,
            "report_files": [
                {
                    "worker": worker,
                    "rank": rank,
                    "capture": capture,
                    "file": path.name,
                    "sha256": sha256_file(path),
                    "nsys_export": nsys_export_metadata[
                        f"w{worker}/r{rank}/c{capture:02d}"
                    ],
                }
                for (worker, rank, capture), path in sorted(reports.items())
            ],
            "nsys_export": nsys_export_metadata[min(nsys_export_metadata)],
            "nsys_report_files": [
                {
                    "worker": worker,
                    "rank": rank,
                    "capture": capture,
                    "file": path.name,
                    "sha256": sha256_file(path),
                }
                for (worker, rank, capture), path in sorted(nsys_reports.items())
            ],
            "worker_logs": [
                {
                    "worker": worker_identity(path),
                    "file": path.name,
                    "sha256": sha256_file(path),
                }
                for path in sorted(args.worker_logs, key=worker_identity)
            ],
            "eager_mapping_file": args.eager_mapping.name,
            "eager_mapping_sha256": sha256_file(args.eager_mapping),
            "contextual_eager_trace": {
                "file": args.eager_trace.name,
                "sha256": sha256_file(args.eager_trace),
                "validated_repeated_slots": len(contextual_signatures),
            },
            "mapping_policy": (
                "worker-local exact-BS32 rank election + one all-rank outer-worker "
                "continuous NVTX report per worker + first-step conservative exclusion + "
                "pre-forward exact-event filtering + graphId/"
                "nodeId occurrence + exact GGGA/MTP5 order + eager stack leaf calibration"
            ),
            "selection_policy": (
                f"exactly {expected_selected_samples} real pre-forward "
                f"running_requests={selected_batch} CUDA Graph rank-local steps, "
                f"balanced {expected_samples_per_source} per worker-elected source and "
                "time-spread over each raw window"
            ),
            "mapped_or_fusion_duration_ratio": round(attributed_residency_ratio, 6),
            "mapped_or_fusion_active_union_ratio": round(attributed_active_ratio, 6),
            "strict_signature_duration_ratio": round(strict_signature_us / total_us, 6),
            "mapped_duration_ratio": round(status_us["mapped"] / total_us, 6),
            "fusion_duration_ratio": round(status_us["fusion"] / total_us, 6),
            "unmapped_duration_ratio": round(status_us["unmapped"] / total_us, 6),
            "timeline_interval_coverage_ratio": round(
                sum(status_us.values()) / total_us, 6
            ),
            "semantic_attribution_gate": {
                "metric": "mapped_or_fusion_active_union_ratio",
                "threshold": 0.95,
                "passed": attributed_active_ratio >= 0.95,
            },
            "critical_gpu_span_ms": timing_summary["critical_gpu_span_ms"],
            "critical_logical_period_ms": timing_summary["critical_logical_period_ms"],
            "instrumented_worker_rank_sources": structurally_validated_sources,
            "rank_local_capture_integrity": rank_local_integrity,
            "worker_capture_integrity": worker_capture_integrity,
            "representative_rank_trace_validation": True,
            "four_rank_trace_validation": True,
            "four_rank_collective_execution_validated_from_worker_logs": True,
            "worker_count": 2,
        },
        "timeline": {},
        "node_states": SGLANG_DECODE_NODE_STATES,
        "node_metrics": node_metrics,
    }
    analysis = {
        "profile_id": profile_id,
        "run_identity": {"job_id": args.job_id, "name": config.get("name")},
        "source_validation": source_validation,
        "rank_local_capture_integrity": rank_local_integrity,
        "worker_capture_integrity": worker_capture_integrity,
        "selected_observations": selected_observations,
        "timing_summary": timing_summary,
        "status_duration_us": dict(status_us),
        "mapped_or_fusion_duration_ratio": attributed_residency_ratio,
        "mapped_or_fusion_active_union_ratio": attributed_active_ratio,
        "strict_signature_duration_ratio": strict_signature_us / total_us,
        "node_metrics": node_metrics,
    }
    return profile, timeline, analysis, all_mappings


def main() -> int:
    args = parse_args()
    profile, timeline, analysis, mappings = build(args)
    timeline_sha = write_timeline_artifact(args.output_timeline, timeline)
    profile["timeline"] = {
        "schema_version": "timeline.v1",
        "artifact": args.output_timeline.name,
        "sha256": timeline_sha,
        "reference_worker": timeline["raw_trace"]["worker"],
        "reference_rank": timeline["reference_rank"],
        "step_count": len(timeline["steps"]),
        "event_count": sum(len(step["events"]) for step in timeline["steps"]),
        "raw_trace_file": timeline["raw_trace"]["file"],
    }
    args.output_profile.parent.mkdir(parents=True, exist_ok=True)
    args.output_profile.write_text(
        yaml.safe_dump(profile, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    args.output_analysis.parent.mkdir(parents=True, exist_ok=True)
    args.output_analysis.write_text(json.dumps(analysis, indent=2) + "\n")
    args.output_mapping.parent.mkdir(parents=True, exist_ok=True)
    with args.output_mapping.open("w") as output:
        for row in mappings:
            output.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(f"wrote {args.output_profile.resolve()}")
    print(
        f"exact BS{profile['workload']['selected_exact_target_verify_batch']} "
        f"samples={profile['workload']['selected_samples']} "
        "attributed-active="
        f"{profile['evidence']['mapped_or_fusion_active_union_ratio']:.3f} "
        "attributed-residency="
        f"{profile['evidence']['mapped_or_fusion_duration_ratio']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
