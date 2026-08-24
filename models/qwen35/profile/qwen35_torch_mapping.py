#!/usr/bin/env python3
"""Load TensorRT-LLM Torch/Kineto steps for the Qwen3.5 mapper.

Newer Kineto builds may expose an executor NVTX range as paired CPU/GPU user
annotations.  The release build used by the strict 8K/1K run instead exposes
exactly one Python ``_forward_step`` interval per captured iteration and omits
GPU NVTX annotations.  CUDA-Graph child kernels also have no external id in
that format.  For that trace we reproduce the same occurrence-index join used
by the NSYS parser: the Nth occurrence of every stable ``(graph id, graph node
id)`` belongs to the Nth ordered graph launch.  Direct kernels remain joined by
their Kineto external id to launches inside the matching Python interval.

Both paths preserve the concrete kernel symbol and convert only the trace
container into the framework-neutral ``NsysStep`` representation consumed by
the conservative Qwen3.5 attribution rules.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from models.common.trace_mapping import load_trace
from models.qwen35.profile.qwen35_nsys_mapping import (
    NsysKernel,
    NsysStep,
    STEP_RE,
)


def _start_us(event: dict[str, Any]) -> float:
    return float(event.get("ts", 0.0))


def _end_us(event: dict[str, Any]) -> float:
    return _start_us(event) + float(event.get("dur", 0.0))


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _arg(event: dict[str, Any], *names: str) -> Any:
    args = event.get("args") or {}
    for name in names:
        if name in args:
            return args[name]
    return None


def _primary_gpu_step_track(
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    tracks: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if event.get("cat") != "gpu_user_annotation" or event.get("ph") != "X":
            continue
        if STEP_RE.fullmatch(str(event.get("name", ""))) is None:
            continue
        tracks[(event.get("pid"), event.get("tid"))].append(event)
    if not tracks:
        return []
    _track, steps = max(
        tracks.items(),
        key=lambda item: sum(float(event.get("dur", 0.0)) for event in item[1]),
    )
    return sorted(steps, key=_start_us)


def _primary_python_step_track(
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    tracks: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if event.get("cat") != "python_function" or event.get("ph") != "X":
            continue
        if not str(event.get("name", "")).endswith(": _forward_step"):
            continue
        tracks[(event.get("pid"), event.get("tid"))].append(event)
    if not tracks:
        return []
    _track, steps = max(tracks.items(), key=lambda item: len(item[1]))
    return sorted(steps, key=_start_us)


def _convert_kernel(event: dict[str, Any], *, rank: int) -> NsysKernel:
    return NsysKernel(
        start_ns=round(_start_us(event) * 1000.0),
        end_ns=round(_end_us(event) * 1000.0),
        name=str(event.get("name", "")),
        stream=_as_int(_arg(event, "stream")),
        correlation_id=_as_int(_arg(event, "correlation")),
        graph_id=(
            _as_int(_arg(event, "graph id", "graphId"))
            if _arg(event, "graph id", "graphId") is not None
            else None
        ),
        graph_node_id=(
            _as_int(_arg(event, "graph node id", "graphNodeId"))
            if _arg(event, "graph node id", "graphNodeId") is not None
            else None
        ),
        device_id=_as_int(_arg(event, "device"), rank),
    )


def _load_python_occurrence_steps(
    events: list[dict[str, Any]], *, path: Path, rank: int
) -> list[NsysStep]:
    python_steps = _primary_python_step_track(events)
    if not python_steps:
        raise ValueError(f"{path.name}: no TensorRT-LLM executor boundaries")

    kernels = sorted(
        (
            event
            for event in events
            if event.get("cat") == "kernel" and event.get("ph") == "X"
        ),
        key=_start_us,
    )
    graph_occurrences: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    direct_kernels = []
    for event in kernels:
        graph_id = _as_int(_arg(event, "graph id", "graphId"))
        graph_node_id = _as_int(_arg(event, "graph node id", "graphNodeId"))
        if graph_id and graph_node_id:
            graph_occurrences[(graph_id, graph_node_id)].append(event)
        else:
            direct_kernels.append(event)
    if not graph_occurrences:
        raise ValueError(f"{path.name}: Python fallback has no CUDA-Graph nodes")
    invalid_occurrences = {
        identity: len(rows)
        for identity, rows in graph_occurrences.items()
        if len(rows) != len(python_steps)
    }
    if invalid_occurrences:
        preview = list(sorted(invalid_occurrences.items()))[:8]
        raise ValueError(
            f"{path.name}: CUDA-Graph node occurrences do not close over "
            f"{len(python_steps)} steps: {preview}"
        )

    launch_events = [
        event
        for event in events
        if event.get("cat") in {"cpu_op", "cuda_runtime"}
        and event.get("ph") == "X"
        and _arg(event, "External id") is not None
    ]
    runtime_events = [
        event
        for event in events
        if event.get("cat") == "cuda_runtime" and event.get("ph") == "X"
    ]
    output = []
    assigned_graph_kernel_count = 0
    for step_index, annotation in enumerate(python_steps):
        cpu_start_us = _start_us(annotation)
        cpu_end_us = _end_us(annotation)
        pid_tid = annotation.get("pid"), annotation.get("tid")
        external_ids = {
            _arg(event, "External id")
            for event in launch_events
            if (event.get("pid"), event.get("tid")) == pid_tid
            and cpu_start_us <= _start_us(event) <= cpu_end_us
        }
        graph_kernels = [rows[step_index] for rows in graph_occurrences.values()]
        assigned_graph_kernel_count += len(graph_kernels)
        step_kernels = graph_kernels + [
            event
            for event in direct_kernels
            if _arg(event, "External id") in external_ids
        ]
        step_kernels.sort(key=_start_us)
        if not step_kernels:
            raise ValueError(f"{path.name}: reconstructed step {step_index} is empty")
        graph_launch_count = sum(
            "cudagraphlaunch" in str(event.get("name", "")).lower()
            and (event.get("pid"), event.get("tid")) == pid_tid
            and cpu_start_us <= _start_us(event) <= cpu_end_us
            for event in runtime_events
        )
        if graph_launch_count != 1:
            raise ValueError(
                f"{path.name}: reconstructed step {step_index} has "
                f"{graph_launch_count} CUDA Graph launches"
            )
        label = (
            f"[Executor] _forward_step {step_index}: "
            "0 ctx reqs, 0 ctx tokens, 32 gen reqs"
        )
        output.append(
            NsysStep(
                step_id=step_index,
                rank=rank,
                label=label,
                cpu_start_ns=round(cpu_start_us * 1000.0),
                cpu_end_ns=round(cpu_end_us * 1000.0),
                context_reqs=0,
                context_tokens=0,
                generation_reqs=32,
                kernels=tuple(_convert_kernel(event, rank=rank) for event in step_kernels),
                graph_launch_count=graph_launch_count,
            )
        )
    expected_graph_kernel_count = sum(len(rows) for rows in graph_occurrences.values())
    if assigned_graph_kernel_count != expected_graph_kernel_count:
        raise AssertionError(
            f"assigned {assigned_graph_kernel_count}/{expected_graph_kernel_count} "
            "CUDA-Graph child kernels"
        )
    return output


def load_trt_torch_steps(path: Path, *, rank: int) -> list[NsysStep]:
    """Return ordered TensorRT-LLM executor steps from one rank-local trace."""

    events = load_trace(path).get("traceEvents") or []
    gpu_steps = _primary_gpu_step_track(events)
    if not gpu_steps:
        return _load_python_occurrence_steps(events, path=path, rank=rank)

    cpu_by_step: dict[int, dict[str, Any]] = {}
    for event in events:
        if event.get("cat") != "user_annotation" or event.get("ph") != "X":
            continue
        match = STEP_RE.fullmatch(str(event.get("name", "")))
        if match is not None:
            cpu_by_step[int(match.group("step"))] = event

    kernels = sorted(
        (
            event
            for event in events
            if event.get("cat") == "kernel" and event.get("ph") == "X"
        ),
        key=_start_us,
    )
    runtime_events = [
        event
        for event in events
        if event.get("cat") == "cuda_runtime" and event.get("ph") == "X"
    ]
    output: list[NsysStep] = []
    for annotation in gpu_steps:
        label = str(annotation.get("name", ""))
        match = STEP_RE.fullmatch(label)
        if match is None:  # guarded by _primary_gpu_step_track
            raise AssertionError(label)
        step_id = int(match.group("step"))
        gpu_start_us = _start_us(annotation)
        gpu_end_us = _end_us(annotation)
        step_kernels = [
            event
            for event in kernels
            if gpu_start_us <= _start_us(event) and _end_us(event) <= gpu_end_us
        ]
        if not step_kernels:
            raise ValueError(f"{path.name}: executor step {step_id} has no CUDA kernels")

        cpu_event = cpu_by_step.get(step_id, annotation)
        cpu_start_us = _start_us(cpu_event)
        cpu_end_us = _end_us(cpu_event)
        graph_launch_count = sum(
            "cudagraphlaunch" in str(event.get("name", "")).lower()
            and cpu_start_us <= _start_us(event) <= cpu_end_us
            for event in runtime_events
        )
        converted = tuple(_convert_kernel(event, rank=rank) for event in step_kernels)
        output.append(
            NsysStep(
                step_id=step_id,
                rank=rank,
                label=label,
                cpu_start_ns=round(cpu_start_us * 1000.0),
                cpu_end_ns=round(cpu_end_us * 1000.0),
                context_reqs=int(match.group("context_reqs")),
                context_tokens=int(match.group("context_tokens")),
                generation_reqs=int(match.group("generation_reqs")),
                kernels=converted,
                graph_launch_count=graph_launch_count,
            )
        )
    return output
