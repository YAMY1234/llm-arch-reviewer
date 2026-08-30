"""Exact production-evidence readers for Kimi K3 on vLLM.

vLLM starts and stops the CUDA profiler independently in every tensor-parallel
worker.  A node-wide Nsight process-tree report therefore contains four
distinct API windows.  This module closes each rank over its own process,
launch correlations, device, and profiler interval before any semantic
attribution is attempted.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_exact_worker_kernels(
    sqlite_path: Path, device: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return kernels launched by one worker inside its exact profiler window."""

    require(sqlite_path.is_file(), f"missing Nsight SQLite export: {sqlite_path}")
    connection = sqlite3.connect(sqlite_path)
    try:
        global_pids = [
            int(row[0])
            for row in connection.execute(
                """
                SELECT DISTINCT globalPid
                  FROM CUPTI_ACTIVITY_KIND_KERNEL
                 WHERE deviceId = ?
                """,
                (device,),
            ).fetchall()
        ]
        require(
            len(global_pids) == 1,
            f"device {device} does not have exactly one profiled worker: {global_pids}",
        )
        global_pid = global_pids[0]
        process_key = global_pid >> 24
        calls = connection.execute(
            """
            SELECT r.start, r.end, s.value
              FROM CUPTI_ACTIVITY_KIND_RUNTIME AS r
              JOIN StringIds AS s ON s.id = r.nameId
             WHERE (r.globalTid >> 24) = ?
               AND (s.value LIKE '%ProfilerStart%'
                    OR s.value LIKE '%ProfilerStop%')
             ORDER BY r.start, r.end
            """,
            (process_key,),
        ).fetchall()
        starts = [row for row in calls if "ProfilerStart" in str(row[2])]
        stops = [row for row in calls if "ProfilerStop" in str(row[2])]
        require(
            len(starts) == 1,
            f"worker {process_key} profiler start count is {len(starts)}",
        )
        require(
            len(stops) == 1,
            f"worker {process_key} profiler stop count is {len(stops)}",
        )
        start_ns = int(starts[0][1])
        stop_ns = int(stops[0][0])
        require(start_ns < stop_ns, f"worker {process_key} profiler API order mismatch")
        total_kernel_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE deviceId = ?",
                (device,),
            ).fetchone()[0]
        )
        rows = connection.execute(
            """
            SELECT DISTINCT k.start, k.end, k.deviceId, k.streamId, s.value,
                   k.graphNodeId, k.correlationId, k.gridId
              FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
              JOIN StringIds AS s ON s.id = k.demangledName
              JOIN CUPTI_ACTIVITY_KIND_RUNTIME AS r
                ON r.correlationId = k.correlationId
               AND (r.globalTid >> 24) = (k.globalPid >> 24)
             WHERE k.deviceId = ?
               AND k.globalPid = ?
               AND r.start >= ?
               AND r.end <= ?
             ORDER BY k.start, k.end, k.gridId
            """,
            (device, global_pid, start_ns, stop_ns),
        ).fetchall()
    finally:
        connection.close()

    kernels = [
        {
            "ts_us": start / 1000.0,
            "dur_us": (stop - start) / 1000.0,
            "device": raw_device,
            "stream": stream,
            "kernel_name": name,
            "graph_node_id": graph_node,
            "correlation": correlation,
            "grid_id": grid,
        }
        for start, stop, raw_device, stream, name, graph_node, correlation, grid in rows
    ]
    require(kernels, f"device {device} exact worker interval contains no kernels")
    return kernels, {
        "worker_global_pid_hash": hashlib.sha256(str(global_pid).encode()).hexdigest()[:16],
        "worker_process_key_hash": hashlib.sha256(str(process_key).encode()).hexdigest()[:16],
        "worker_profiler_start_api_count": len(starts),
        "worker_profiler_stop_api_count": len(stops),
        "profiler_interval_us": round((stop_ns - start_ns) / 1000.0, 6),
        "node_collection_kernel_count": total_kernel_count,
        "launch_correlated_exact_window_kernel_count": len(kernels),
        "gpu_execution_after_profiler_stop_kernel_count": sum(
            1 for start, stop, *_ in rows if int(start) > stop_ns or int(stop) > stop_ns
        ),
    }


def validate_production_client(
    *,
    root: Path,
    batch_size: int,
    baseline_relative_step: int,
    client_source: Path,
) -> dict[str, Any]:
    """Validate the exact 3C warmup/1C formal request and vLLM coordinate."""

    path = root / f"client-c{batch_size}.json"
    require(path.is_file(), f"missing exact client evidence: {path}")
    client = json.loads(path.read_text())
    contract = client.get("contract") or {}
    requests = (client.get("warmup") or {}).get("requests", []) + (
        client.get("formal") or {}
    ).get("requests", [])
    require(client.get("state") == "passed", "exact client did not pass")
    require(contract.get("concurrency") == batch_size, "client concurrency mismatch")
    require(contract.get("isl") == 8192 and contract.get("osl") == 1024, "length mismatch")
    require(contract.get("warmup_request_count") == 3 * batch_size, "warmup mismatch")
    require(contract.get("formal_request_count") == batch_size, "formal mismatch")
    require(contract.get("no_intentionally_shared_prefix") is True, "shared prefix")
    require(len(requests) == 4 * batch_size, "realized request count mismatch")
    require(
        all(
            row.get("http_status") == 200
            and row.get("realized_prompt_tokens") == 8192
            and row.get("realized_completion_tokens") == 1024
            for row in requests
        ),
        "realized request contract mismatch",
    )
    prompt_hashes = [row.get("prompt_token_sha256") for row in requests]
    require(
        all(prompt_hashes) and len(set(prompt_hashes)) == len(prompt_hashes),
        "prompt streams are not unique",
    )
    require(client_source.is_file(), f"missing client source: {client_source}")
    require(
        (client.get("client_source") or {}).get("sha256") == sha256_file(client_source),
        "client source hash mismatch",
    )
    coordinate = client.get("profile_coordinate") or {}
    require(
        coordinate.get("mode") == "vllm_server_profiler_delay_iterations",
        "profiler coordinate mode mismatch",
    )
    require(
        coordinate.get("baseline_relative_start_step") == baseline_relative_step,
        "baseline-relative step mismatch",
    )
    expected_delay = 0 if baseline_relative_step == 0 else baseline_relative_step + 1
    require(
        coordinate.get("profiler_delay_iterations") == expected_delay,
        "profiler delay mismatch",
    )
    controls = client.get("profile_controls") or []
    require(
        [row.get("action") for row in controls] == ["start", "stop"],
        "profile controls mismatch",
    )
    require(
        all(row.get("http_status") == 200 and not (row.get("request") or {}) for row in controls),
        "profile control response mismatch",
    )
    return {
        "sha256": sha256_file(path),
        "client_source_sha256": sha256_file(client_source),
        "baseline_relative_start_step": baseline_relative_step,
        "profiler_delay_iterations": expected_delay,
        "warmup_cached_token_count": coordinate.get("warmup_cached_token_count"),
    }
