#!/usr/bin/env python3
"""Parse and attribute worker-local Qwen3.5 Nsight Systems reports.

TensorRT-LLM launches the decode model as a CUDA Graph.  Nsight records graph
child kernels with correlation id zero, so a normal runtime-correlation join
cannot associate them with ``[Executor] _forward_step`` NVTX ranges.  The
stable identity is ``graphNodeId``: each kernel-bearing node occurs exactly
once per launch.  We therefore split overlapping graph executions by the
occurrence index of each graph node and pair those executions with the ordered
``cudaGraphLaunch`` calls.  Direct launches remain associated by correlation
id. SGLang reports instead use one ``scheduler.run_batch`` NVTX range per
rank-local decode step. Within that wall-time boundary, the 60-layer target
graph and the four-plus-one MTP graphs are identified by their complete
kernel-bearing node sequences. The parser fails closed if a required invariant
does not hold.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import product
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterable

from models.common.trace_mapping import ForwardWindow


STEP_RE = re.compile(
    r"\[Executor\] _forward_step (?P<step>\d+): "
    r"(?P<context_reqs>\d+) ctx reqs, (?P<context_tokens>\d+) ctx tokens, "
    r"(?P<generation_reqs>\d+) gen reqs"
)
RANK_RE = re.compile(r"rank(?P<rank>\d+)")
SGLANG_STEP_LABEL = "scheduler.run_batch"
SGLANG_GDN_ANCHOR = "fused_qkvzba_split"
SGLANG_ATTENTION_ANCHOR = "_fused_qk_rmsnorm_rope_gate_kernel"
TARGET_PATTERN = tuple(
    "attention" if layer_id % 4 == 3 else "gdn" for layer_id in range(60)
)


@dataclass(frozen=True)
class NsysKernel:
    start_ns: int
    end_ns: int
    name: str
    stream: int
    correlation_id: int
    graph_id: int | None
    graph_node_id: int | None
    device_id: int | None = None
    context_id: int | None = None
    global_pid: int | None = None

    @property
    def duration_us(self) -> float:
        return (self.end_ns - self.start_ns) / 1000.0


@dataclass(frozen=True)
class NsysStep:
    step_id: int
    rank: int
    label: str
    cpu_start_ns: int
    cpu_end_ns: int
    context_reqs: int
    context_tokens: int
    generation_reqs: int
    kernels: tuple[NsysKernel, ...]
    graph_launch_count: int

    @property
    def cpu_wall_us(self) -> float:
        return (self.cpu_end_ns - self.cpu_start_ns) / 1000.0

    @property
    def gpu_start_ns(self) -> int:
        return min(kernel.start_ns for kernel in self.kernels)

    @property
    def gpu_end_ns(self) -> int:
        return max(kernel.end_ns for kernel in self.kernels)


@dataclass(frozen=True)
class NsysAttribution:
    node: str
    label: str
    status: str = "mapped"
    ir_targets: tuple[str, ...] = ()
    attribution_method: str = "unique_kernel_signature"
    confidence: str = "high"


def _strings(connection: sqlite3.Connection) -> dict[int, str]:
    return {int(key): str(value) for key, value in connection.execute(
        "select id, value from StringIds"
    )}


def _table_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in connection.execute(f"pragma table_info({table})")}


def _step_rows(
    connection: sqlite3.Connection, strings: dict[int, str]
) -> list[dict[str, Any]]:
    rows = []
    for start, end, text, text_id in connection.execute(
        "select start, end, text, textId from NVTX_EVENTS "
        "where end is not null order by start"
    ):
        label = str(text) if text is not None else strings.get(int(text_id), "")
        match = STEP_RE.fullmatch(label)
        if match is None:
            continue
        rows.append(
            {
                "start": int(start),
                "end": int(end),
                "label": label,
                **{key: int(value) for key, value in match.groupdict().items()},
            }
        )
    if not rows:
        raise ValueError("Nsight report contains no complete Executor forward-step NVTX ranges")
    return rows


def _kernel_rows(
    connection: sqlite3.Connection,
    strings: dict[int, str],
    *,
    global_pid: int | None = None,
) -> list[NsysKernel]:
    columns = _table_columns(connection, "CUPTI_ACTIVITY_KIND_KERNEL")
    device = "deviceId" if "deviceId" in columns else "null"
    context = "contextId" if "contextId" in columns else "null"
    process = "globalPid" if "globalPid" in columns else "null"
    name = "demangledName" if "demangledName" in columns else "shortName"
    where = ""
    parameters: tuple[int, ...] = ()
    if global_pid is not None:
        if "globalPid" not in columns:
            raise ValueError("Nsight kernel table has no globalPid column")
        where = " where globalPid = ?"
        parameters = (global_pid,)
    return [
        NsysKernel(
            start_ns=int(start),
            end_ns=int(end),
            name=str(strings.get(int(name_id), f"StringIds[{name_id}]")),
            stream=int(stream),
            correlation_id=int(correlation),
            graph_id=None if graph_id is None else int(graph_id),
            graph_node_id=None if graph_node_id is None else int(graph_node_id),
            device_id=None if device_id is None else int(device_id),
            context_id=None if context_id is None else int(context_id),
            global_pid=None if row_global_pid is None else int(row_global_pid),
        )
        for (
            start,
            end,
            stream,
            correlation,
            graph_id,
            graph_node_id,
            name_id,
            device_id,
            context_id,
            row_global_pid,
        ) in connection.execute(
            "select start, end, streamId, correlationId, graphId, graphNodeId, "
            f"{name}, {device}, {context}, {process} "
            f"from CUPTI_ACTIVITY_KIND_KERNEL{where} order by start",
            parameters,
        )
    ]


def _runtime_rows(
    connection: sqlite3.Connection, strings: dict[int, str]
) -> list[tuple[int, int, int, str]]:
    return [
        (int(start), int(end), int(correlation), strings.get(int(name_id), ""))
        for start, end, correlation, name_id in connection.execute(
            "select start, end, correlationId, nameId "
            "from CUPTI_ACTIVITY_KIND_RUNTIME order by start"
        )
    ]


def _containing_step(
    steps: list[dict[str, Any]], timestamp_ns: int
) -> int | None:
    for index, step in enumerate(steps):
        if step["start"] <= timestamp_ns < step["end"]:
            return index
    return None


def _graph_executions(
    kernels: list[NsysKernel], launch_count: int
) -> list[list[NsysKernel]]:
    graph_kernels = [kernel for kernel in kernels if kernel.graph_id is not None]
    if not graph_kernels:
        return []
    if launch_count <= 0:
        raise ValueError("CUDA Graph child kernels exist without cudaGraphLaunch calls")

    by_graph_and_node: dict[int, dict[int, list[NsysKernel]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for kernel in graph_kernels:
        if kernel.graph_id is None or kernel.graph_node_id is None:
            raise ValueError("CUDA Graph child kernel lacks graphNodeId")
        by_graph_and_node[kernel.graph_id][kernel.graph_node_id].append(kernel)

    # Decode replays one large graph per forward. TRT prefill instead launches
    # one graph per target layer/segment (61 graph IDs in the formal capture),
    # with the CPU enqueue running well ahead of GPU execution. Split each
    # graph independently by graph-node occurrence, then rebuild the repeated
    # launch sequence. This retains the same fail-closed graphNodeId invariant
    # without assuming a decode-only, single-graph topology.
    per_graph_executions: dict[int, list[list[NsysKernel]]] = {}
    graph_multiplicities: dict[int, int] = {}
    for graph_id, by_node in by_graph_and_node.items():
        multiplicity = Counter(len(events) for events in by_node.values())
        if len(multiplicity) != 1:
            raise ValueError(
                "CUDA Graph node occurrence mismatch: "
                f"graph_id={graph_id}, node multiplicities={dict(multiplicity)}"
            )
        occurrences = next(iter(multiplicity))
        graph_multiplicities[graph_id] = occurrences
        executions: list[list[NsysKernel]] = [[] for _ in range(occurrences)]
        for events in by_node.values():
            if len({event.name for event in events}) != 1:
                raise ValueError(
                    f"CUDA Graph {graph_id} node produced inconsistent kernel signatures"
                )
            for occurrence, event in enumerate(
                sorted(events, key=lambda item: item.start_ns)
            ):
                executions[occurrence].append(event)
        for events in executions:
            events.sort(key=lambda item: item.start_ns)
        per_graph_executions[graph_id] = executions

    occurrence_counts = set(graph_multiplicities.values())
    if len(occurrence_counts) != 1:
        raise ValueError(
            "CUDA Graphs were not launched with one repeated sequence: "
            f"per-graph launch counts={graph_multiplicities}"
        )
    occurrences = next(iter(occurrence_counts))
    expected_launches = len(per_graph_executions) * occurrences
    if expected_launches != launch_count:
        raise ValueError(
            "CUDA Graph launch/execution mismatch: "
            f"runtime launches={launch_count}, graph executions={expected_launches}"
        )

    launch_order: list[int] | None = None
    output: list[list[NsysKernel]] = []
    for occurrence in range(occurrences):
        current_order = sorted(
            per_graph_executions,
            key=lambda graph_id: per_graph_executions[graph_id][occurrence][0].start_ns,
        )
        if launch_order is None:
            launch_order = current_order
        elif current_order != launch_order:
            raise ValueError(
                "CUDA Graph execution order changed across repeated forward steps"
            )
        output.extend(per_graph_executions[graph_id][occurrence] for graph_id in current_order)
    return output


def load_nsys_steps(path: Path, *, rank: int | None = None) -> list[NsysStep]:
    """Load complete forward steps and assign every selected GPU kernel once."""

    if rank is None:
        match = RANK_RE.search(path.name)
        if match is None:
            raise ValueError(f"cannot infer rank from {path.name}")
        rank = int(match.group("rank"))

    connection = sqlite3.connect(path)
    try:
        strings = _strings(connection)
        steps = _step_rows(connection, strings)
        kernels = _kernel_rows(connection, strings)
        runtime = _runtime_rows(connection, strings)
    finally:
        connection.close()

    graph_launches = [
        row for row in runtime if row[3].lower().startswith("cudagraphlaunch")
    ]
    executions = _graph_executions(kernels, len(graph_launches))
    graph_by_step: dict[int, list[NsysKernel]] = defaultdict(list)
    graph_launch_counts: Counter[int] = Counter()
    for execution, launch in zip(executions, graph_launches):
        step_index = _containing_step(steps, launch[0])
        if step_index is None:
            raise ValueError(
                f"cudaGraphLaunch at {launch[0]} is outside every forward-step range"
            )
        graph_by_step[step_index].extend(execution)
        graph_launch_counts[step_index] += 1

    runtime_correlations: dict[int, set[int]] = defaultdict(set)
    for start, _end, correlation, _name in runtime:
        step_index = _containing_step(steps, start)
        if step_index is not None:
            runtime_correlations[step_index].add(correlation)
    direct_by_step: dict[int, list[NsysKernel]] = defaultdict(list)
    for kernel in kernels:
        if kernel.graph_id is not None:
            continue
        for step_index, correlations in runtime_correlations.items():
            if kernel.correlation_id in correlations:
                direct_by_step[step_index].append(kernel)
                break

    output: list[NsysStep] = []
    selected_kernel_ids: set[int] = set()
    for index, step in enumerate(steps):
        selected = sorted(
            [*direct_by_step[index], *graph_by_step[index]],
            key=lambda item: item.start_ns,
        )
        if not selected:
            raise ValueError(f"forward step {step['step']} has no associated GPU kernels")
        for kernel in selected:
            identity = id(kernel)
            if identity in selected_kernel_ids:
                raise ValueError(f"kernel assigned to more than one step: {kernel}")
            selected_kernel_ids.add(identity)
        output.append(
            NsysStep(
                step_id=step["step"],
                rank=rank,
                label=step["label"],
                cpu_start_ns=step["start"],
                cpu_end_ns=step["end"],
                context_reqs=step["context_reqs"],
                context_tokens=step["context_tokens"],
                generation_reqs=step["generation_reqs"],
                kernels=tuple(selected),
                graph_launch_count=int(graph_launch_counts[index]),
            )
        )
    return output


def _global_pid(global_tid: int) -> int:
    """Return the process portion of an Nsight ``globalTid`` value."""

    return int(global_tid) & ~((1 << 24) - 1)


def _sglang_process_by_device(connection: sqlite3.Connection) -> dict[int, int]:
    columns = _table_columns(connection, "CUPTI_ACTIVITY_KIND_KERNEL")
    if not {"globalPid", "deviceId"}.issubset(columns):
        raise ValueError("SGLang Nsight report lacks globalPid/deviceId identity")
    candidates: dict[int, set[int]] = defaultdict(set)
    for process, device in connection.execute(
        "select globalPid, deviceId from CUPTI_ACTIVITY_KIND_KERNEL "
        "group by globalPid, deviceId order by deviceId"
    ):
        candidates[int(device)].add(int(process))
    ambiguous = {device: values for device, values in candidates.items() if len(values) != 1}
    if ambiguous:
        raise ValueError(f"SGLang device has multiple kernel processes: {ambiguous}")
    if not candidates:
        raise ValueError("SGLang Nsight report contains no CUDA kernels")
    return {device: next(iter(values)) for device, values in candidates.items()}


def _sglang_step_boundaries(
    connection: sqlite3.Connection,
    strings: dict[int, str],
    *,
    global_pid: int,
    allow_cuda_sync_fallback: bool,
) -> tuple[str, list[tuple[int, int]]]:
    ranges: list[tuple[int, int]] = []
    for start, end, text_value, text_id, global_tid in connection.execute(
        "select start, end, text, textId, globalTid from NVTX_EVENTS "
        "where end is not null order by start"
    ):
        if global_tid is None or _global_pid(int(global_tid)) != global_pid:
            continue
        label = (
            str(text_value)
            if text_value is not None
            else strings.get(int(text_id), "") if text_id is not None else ""
        )
        if label == SGLANG_STEP_LABEL:
            ranges.append((int(start), int(end)))
    if len(ranges) >= 2:
        return SGLANG_STEP_LABEL, ranges
    if not allow_cuda_sync_fallback:
        raise ValueError(
            "SGLang Nsight report has fewer than two scheduler.run_batch NVTX ranges; "
            "set SGLANG_ENABLE_NVTX_SCHEDULER=1 for formal captures"
        )

    runtime_columns = _table_columns(connection, "CUPTI_ACTIVITY_KIND_RUNTIME")
    if "globalTid" not in runtime_columns:
        raise ValueError("legacy CUDA-sync fallback requires runtime globalTid")
    fallback = [
        (int(start), int(end))
        for start, end, global_tid in connection.execute(
            "select r.start, r.end, r.globalTid "
            "from CUPTI_ACTIVITY_KIND_RUNTIME r "
            "join StringIds s on s.id = r.nameId "
            "where s.value = 'cudaEventSynchronize_v3020' "
            "and r.end - r.start > 1000000 order by r.start"
        )
        if global_tid is not None and _global_pid(int(global_tid)) == global_pid
    ]
    if len(fallback) < 2:
        raise ValueError("SGLang Nsight report has no complete legacy CUDA-sync boundaries")
    return "cudaEventSynchronize_v3020 (legacy fallback)", fallback


def _sglang_graph_launches(
    connection: sqlite3.Connection,
    strings: dict[int, str],
    *,
    global_pid: int,
) -> list[int]:
    columns = _table_columns(connection, "CUPTI_ACTIVITY_KIND_RUNTIME")
    if "globalTid" not in columns:
        raise ValueError("SGLang Nsight runtime table lacks globalTid")
    return [
        int(start)
        for start, name_id, global_tid in connection.execute(
            "select start, nameId, globalTid from CUPTI_ACTIVITY_KIND_RUNTIME "
            "order by start"
        )
        if global_tid is not None
        and _global_pid(int(global_tid)) == global_pid
        and strings.get(int(name_id), "").lower().startswith("cudagraphlaunch")
    ]


def _sglang_runtime_rows(
    connection: sqlite3.Connection,
    strings: dict[int, str],
    *,
    global_pid: int,
) -> list[tuple[int, int, int, str]]:
    columns = _table_columns(connection, "CUPTI_ACTIVITY_KIND_RUNTIME")
    if "globalTid" not in columns:
        raise ValueError("SGLang Nsight runtime table lacks globalTid")
    return [
        (int(start), int(end), int(correlation), strings.get(int(name_id), ""))
        for start, end, correlation, name_id, global_tid in connection.execute(
            "select start, end, correlationId, nameId, globalTid "
            "from CUPTI_ACTIVITY_KIND_RUNTIME order by start"
        )
        if global_tid is not None and _global_pid(int(global_tid)) == global_pid
    ]


def _sglang_graph_role(events: Iterable[NsysKernel]) -> str | None:
    anchors: list[str] = []
    for event in sorted(events, key=lambda item: item.start_ns):
        lowered = event.name.lower()
        if SGLANG_GDN_ANCHOR in lowered:
            anchors.append("gdn")
        elif SGLANG_ATTENTION_ANCHOR in lowered:
            anchors.append("attention")
    kinds = tuple(anchors)
    if kinds == TARGET_PATTERN:
        return "target_verify"
    if kinds == ("attention",) * 4:
        return "draft"
    if kinds == ("attention",):
        return "draft_extend"
    return None


def _sglang_graph_execution_options(
    kernels: Iterable[NsysKernel],
) -> dict[int, list[tuple[str, str, list[list[NsysKernel]]]]]:
    """Build leading/trailing-boundary candidates for every concrete graph."""

    by_graph_and_node: dict[int, dict[int, list[NsysKernel]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for kernel in kernels:
        if kernel.graph_id is None:
            continue
        if kernel.graph_node_id is None:
            raise ValueError("SGLang CUDA Graph child lacks graphNodeId")
        by_graph_and_node[kernel.graph_id][kernel.graph_node_id].append(kernel)
    if not by_graph_and_node:
        raise ValueError("SGLang report contains no CUDA Graph child kernels")

    output: dict[int, list[tuple[str, str, list[list[NsysKernel]]]]] = {}
    for graph_id, by_node in sorted(by_graph_and_node.items()):
        counts = {node: len(events) for node, events in by_node.items()}
        occurrences = min(counts.values())
        if occurrences <= 0 or max(counts.values()) - occurrences > 1:
            raise ValueError(
                f"SGLang graph {graph_id} has non-boundary node multiplicities: {counts}"
            )
        for node_id, events in by_node.items():
            if len({event.name for event in events}) != 1:
                raise ValueError(
                    f"SGLang graph {graph_id} node {node_id} changed kernel signature"
                )

        modes = ("exact",) if len(set(counts.values())) == 1 else ("drop_leading", "drop_trailing")
        candidates = []
        for mode in modes:
            executions: list[list[NsysKernel]] = [[] for _ in range(occurrences)]
            for events in by_node.values():
                ordered = sorted(events, key=lambda event: event.start_ns)
                if len(ordered) > occurrences:
                    ordered = (
                        ordered[1:]
                        if mode == "drop_leading"
                        else ordered[:-1]
                    )
                for occurrence, event in enumerate(ordered):
                    executions[occurrence].append(event)
            roles = set()
            for events in executions:
                events.sort(key=lambda event: event.start_ns)
                role = _sglang_graph_role(events)
                if role is None:
                    break
                roles.add(role)
            else:
                if len(roles) == 1:
                    candidates.append((mode, next(iter(roles)), executions))
        if not candidates:
            raise ValueError(
                f"SGLang graph {graph_id} has no complete target/draft graph execution"
            )
        output[graph_id] = candidates
    return output


def _marker_index(
    markers: list[tuple[int, int]], timestamp_ns: int
) -> int | None:
    for index, (start, end) in enumerate(markers):
        if start <= timestamp_ns < end:
            return index
    return None


def _pair_sglang_graph_executions(
    kernels: Iterable[NsysKernel],
    launches: list[int],
    markers: list[tuple[int, int]],
) -> tuple[list[tuple[int, str, list[NsysKernel]]], dict[str, Any]]:
    """Pair graph-node occurrence groups with ordered CPU graph launches."""

    options = _sglang_graph_execution_options(kernels)
    graph_ids = sorted(options)
    candidates: list[
        tuple[
            tuple[int, int, int, int],
            list[tuple[int, str, list[NsysKernel]]],
            dict[str, Any],
        ]
    ] = []
    for choice in product(*(options[graph_id] for graph_id in graph_ids)):
        flattened: list[tuple[int, str, list[NsysKernel]]] = []
        modes = {}
        for graph_id, (mode, role, executions) in zip(graph_ids, choice):
            modes[graph_id] = mode
            flattened.extend(
                (events[0].start_ns, role, events) for events in executions
            )
        flattened.sort(key=lambda row: row[0])
        if len(flattened) > len(launches):
            continue
        extra_launches = len(launches) - len(flattened)
        for launch_offset in range(extra_launches + 1):
            selected_launches = launches[
                launch_offset : launch_offset + len(flattened)
            ]
            if any(
                launch > execution_start
                for launch, (execution_start, _role, _events) in zip(
                    selected_launches, flattened
                )
            ):
                continue

            paired = [
                (launch, role, events)
                for launch, (_start, role, events) in zip(
                    selected_launches, flattened
                )
            ]
            roles_by_marker: dict[int, list[str]] = defaultdict(list)
            outside = 0
            for launch, role, _events in paired:
                index = _marker_index(markers, launch)
                if index is None:
                    outside += 1
                else:
                    roles_by_marker[index].append(role)
            required = ["draft", "draft_extend", "target_verify"]
            complete = sum(
                sorted(roles) == required for roles in roles_by_marker.values()
            )
            malformed = sum(
                sorted(roles) != required for roles in roles_by_marker.values()
            )
            total_span = sum(
                max(event.end_ns for event in events)
                - min(event.start_ns for event in events)
                for _launch, _role, events in paired
            )
            score = (complete, -malformed, -outside, -total_span)
            choice_evidence: dict[str, Any] = {
                "boundary_trim_by_graph": modes,
                "unpaired_launch_count": extra_launches,
                "leading_unpaired_launch_count": launch_offset,
                "trailing_unpaired_launch_count": extra_launches - launch_offset,
            }
            candidates.append((score, paired, choice_evidence))
    if not candidates:
        raise ValueError(
            "SGLang graph executions cannot be paired one-to-one with cudaGraphLaunch"
        )
    candidates.sort(key=lambda row: row[0], reverse=True)
    score, paired, choice_evidence = candidates[0]
    if score[0] <= 0:
        raise ValueError("SGLang report contains no complete three-graph decode step")
    return paired, {
        "complete_graph_step_count": score[0],
        "malformed_graph_step_count": -score[1],
        "launches_outside_step_ranges": -score[2],
        **choice_evidence,
    }


def load_sglang_nsys_steps(
    path: Path,
    *,
    rank: int,
    allow_cuda_sync_fallback: bool = False,
) -> tuple[list[NsysStep], dict[str, Any]]:
    """Load complete rank-local SGLang steps from a worker-local NSYS report.

    Formal traces require ``scheduler.run_batch`` NVTX. The optional CUDA-sync
    fallback exists only to validate the parser against older local reports and
    is recorded in the returned evidence; it must not be used for a delivered
    matched profile.
    """

    connection = sqlite3.connect(path)
    try:
        strings = _strings(connection)
        process_by_device = _sglang_process_by_device(connection)
        if rank not in process_by_device:
            raise ValueError(
                f"SGLang rank/device {rank} is absent; present devices={sorted(process_by_device)}"
            )
        process = process_by_device[rank]
        marker_source, boundaries = _sglang_step_boundaries(
            connection,
            strings,
            global_pid=process,
            allow_cuda_sync_fallback=allow_cuda_sync_fallback,
        )
        kernels = _kernel_rows(connection, strings, global_pid=process)
        graph_launches = _sglang_graph_launches(
            connection, strings, global_pid=process
        )
        runtime = _sglang_runtime_rows(connection, strings, global_pid=process)
    finally:
        connection.close()

    if marker_source == SGLANG_STEP_LABEL:
        paired, pairing_evidence = _pair_sglang_graph_executions(
            kernels, graph_launches, boundaries
        )
        graph_by_step: dict[int, list[NsysKernel]] = defaultdict(list)
        graph_launch_counts: Counter[int] = Counter()
        graph_roles_by_step: dict[int, list[str]] = defaultdict(list)
        for launch, role, execution in paired:
            step_index = _marker_index(boundaries, launch)
            if step_index is None:
                continue
            graph_by_step[step_index].extend(execution)
            graph_launch_counts[step_index] += 1
            graph_roles_by_step[step_index].append(role)

        correlations_by_step: dict[int, set[int]] = defaultdict(set)
        for start, _end, correlation, _name in runtime:
            step_index = _marker_index(boundaries, start)
            if step_index is not None:
                correlations_by_step[step_index].add(correlation)
        direct_by_step: dict[int, list[NsysKernel]] = defaultdict(list)
        for kernel in kernels:
            if kernel.graph_id is not None:
                continue
            for step_index, correlations in correlations_by_step.items():
                if kernel.correlation_id in correlations:
                    direct_by_step[step_index].append(kernel)
                    break

        steps = []
        assigned_kernel_ids: set[int] = set()
        required_roles = ["draft", "draft_extend", "target_verify"]
        crossing_kernel_count = 0
        for step_index, (start, marker_end) in enumerate(boundaries):
            if sorted(graph_roles_by_step.get(step_index, [])) != required_roles:
                continue
            wall_end = (
                boundaries[step_index + 1][0]
                if step_index + 1 < len(boundaries)
                else marker_end
            )
            selected = tuple(
                sorted(
                    [*direct_by_step[step_index], *graph_by_step[step_index]],
                    key=lambda kernel: kernel.start_ns,
                )
            )
            if not selected:
                raise ValueError(f"SGLang graph step {step_index} has no kernels")
            crossing_kernel_count += sum(
                kernel.start_ns < start or kernel.end_ns > wall_end
                for kernel in selected
            )
            for kernel in selected:
                identity = id(kernel)
                if identity in assigned_kernel_ids:
                    raise ValueError(f"SGLang kernel assigned twice: {kernel}")
                assigned_kernel_ids.add(identity)
            steps.append(
                NsysStep(
                    step_id=step_index,
                    rank=rank,
                    label=SGLANG_STEP_LABEL,
                    cpu_start_ns=start,
                    cpu_end_ns=wall_end,
                    context_reqs=0,
                    context_tokens=0,
                    generation_reqs=0,
                    kernels=selected,
                    graph_launch_count=int(graph_launch_counts[step_index]),
                )
            )
        if not steps:
            raise ValueError(f"{path}: no complete SGLang rank {rank} graph steps")
        return steps, {
            "rank": rank,
            "device_id": rank,
            "global_pid": process,
            "marker_source": marker_source,
            "marker_count": len(boundaries),
            "complete_step_count": len(steps),
            "crossing_kernel_count": crossing_kernel_count,
            **pairing_evidence,
        }

    # Legacy reports without scheduler NVTX can only be sliced by the next
    # blocking CUDA synchronization. Keep this path explicit and excluded from
    # formal profile delivery; it exists to regression-test graph signatures.
    steps: list[NsysStep] = []
    assigned_kernel_ids: set[int] = set()
    crossing_kernel_count = 0
    for step_index, ((start, marker_end), (next_start, _next_end)) in enumerate(
        zip(boundaries, boundaries[1:])
    ):
        if next_start <= start:
            raise ValueError("SGLang step markers are not strictly increasing")
        selected = tuple(
            kernel for kernel in kernels if start <= kernel.start_ns < next_start
        )
        if not selected:
            continue
        crossing = [kernel for kernel in selected if kernel.end_ns > next_start]
        crossing_kernel_count += len(crossing)
        for kernel in selected:
            identity = id(kernel)
            if identity in assigned_kernel_ids:
                raise ValueError(f"SGLang kernel assigned twice: {kernel}")
            assigned_kernel_ids.add(identity)
        launch_count = sum(start <= timestamp < next_start for timestamp in graph_launches)
        steps.append(
            NsysStep(
                step_id=step_index,
                rank=rank,
                label=SGLANG_STEP_LABEL,
                cpu_start_ns=start,
                cpu_end_ns=next_start,
                context_reqs=0,
                context_tokens=0,
                generation_reqs=0,
                kernels=selected,
                graph_launch_count=launch_count,
            )
        )
    if not steps:
        raise ValueError(f"{path}: no complete SGLang rank {rank} steps with kernels")
    return steps, {
        "rank": rank,
        "device_id": rank,
        "global_pid": process,
        "marker_source": marker_source,
        "marker_count": len(boundaries),
        "complete_step_count": len(steps),
        "crossing_kernel_count": crossing_kernel_count,
    }


def sglang_graph_roles(kernels: Iterable[NsysKernel]) -> dict[str, int]:
    """Prove target/draft/draft-extend roles from one SGLang step's graphs."""

    by_graph: dict[int, list[NsysKernel]] = defaultdict(list)
    for kernel in kernels:
        if kernel.graph_id is None:
            continue
        if kernel.graph_node_id is None:
            raise ValueError("SGLang CUDA Graph child lacks graphNodeId")
        by_graph[kernel.graph_id].append(kernel)
    if not by_graph:
        raise ValueError("SGLang step contains no CUDA Graph child kernels")

    roles: dict[str, int] = {}
    unresolved: dict[int, tuple[str, ...]] = {}
    for graph_id, events in sorted(by_graph.items()):
        node_ids = [event.graph_node_id for event in events]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError(
                f"SGLang graph {graph_id} has repeated graphNodeId in one step"
            )
        kinds = tuple(
            "gdn"
            if SGLANG_GDN_ANCHOR in event.name.lower()
            else "attention"
            for event in sorted(events, key=lambda item: item.start_ns)
            if SGLANG_GDN_ANCHOR in event.name.lower()
            or SGLANG_ATTENTION_ANCHOR in event.name.lower()
        )
        role = _sglang_graph_role(events)
        if role is None:
            unresolved[graph_id] = kinds
            continue
        if role in roles:
            raise ValueError(
                f"SGLang step has multiple graphs classified as {role}: "
                f"{roles[role]}, {graph_id}"
            )
        roles[role] = graph_id

    required = {"target_verify", "draft", "draft_extend"}
    if set(roles) != required or unresolved:
        raise ValueError(
            "SGLang graph-role proof failed: "
            f"roles={roles}, unresolved_anchor_sequences={unresolved}"
        )
    return roles


def validate_sglang_graph_node_stability(steps: Iterable[NsysStep]) -> dict[str, Any]:
    """Verify every concrete graph node keeps one kernel signature."""

    names: dict[tuple[int, int, int, int], set[str]] = defaultdict(set)
    role_counts: Counter[str] = Counter()
    step_count = 0
    for step in steps:
        step_count += 1
        roles = sglang_graph_roles(step.kernels)
        role_by_graph = {graph_id: role for role, graph_id in roles.items()}
        role_counts.update(role_by_graph.values())
        for kernel in step.kernels:
            if kernel.graph_id is None:
                continue
            key = (
                int(kernel.global_pid or 0),
                int(kernel.device_id if kernel.device_id is not None else step.rank),
                kernel.graph_id,
                int(kernel.graph_node_id),
            )
            names[key].add(kernel.name)
    inconsistent = {key: values for key, values in names.items() if len(values) != 1}
    if inconsistent:
        raise ValueError(
            f"SGLang graph nodes changed kernel signature: {len(inconsistent)} nodes"
        )
    return {
        "step_count": step_count,
        "stable_graph_node_count": len(names),
        "role_counts": dict(role_counts),
    }


def sglang_nsys_trace_events(
    step: NsysStep, *, batch_size: int
) -> tuple[list[dict[str, Any]], ForwardWindow, dict[str, int]]:
    """Convert one proven SGLang NSYS step into the common mapping trace form."""

    if batch_size <= 0:
        raise ValueError(f"invalid SGLang target-verify batch size: {batch_size}")
    roles = sglang_graph_roles(step.kernels)
    role_by_graph = {graph_id: role for role, graph_id in roles.items()}
    trace_events: list[dict[str, Any]] = []
    process = next(
        (kernel.global_pid for kernel in step.kernels if kernel.global_pid is not None),
        0,
    )
    for kernel in step.kernels:
        trace_events.append(
            {
                "name": kernel.name,
                "cat": "kernel",
                "ph": "X",
                "ts": kernel.start_ns / 1000.0,
                "dur": (kernel.end_ns - kernel.start_ns) / 1000.0,
                "pid": process,
                "tid": kernel.stream,
                "args": {
                    "stream": kernel.stream,
                    "device": kernel.device_id,
                    "correlation": kernel.correlation_id,
                    "graph_id": kernel.graph_id,
                    "graph_node_id": kernel.graph_node_id,
                    "graph_role": role_by_graph.get(kernel.graph_id),
                },
            }
        )

    annotation_name = {
        "target_verify": f"step[TARGET_VERIFY bs={batch_size}]",
        "draft": "draft",
        "draft_extend": "draft_extend",
    }
    for role, graph_id in roles.items():
        events = [kernel for kernel in step.kernels if kernel.graph_id == graph_id]
        start = min(kernel.start_ns for kernel in events)
        end = max(kernel.end_ns for kernel in events)
        trace_events.append(
            {
                "name": annotation_name[role],
                "cat": "gpu_user_annotation",
                "ph": "X",
                "ts": start / 1000.0,
                "dur": (end - start) / 1000.0,
                "pid": process,
                "tid": "nsys-graph-role",
                "args": {"graph_id": graph_id, "evidence": "validated_graph_node_sequence"},
            }
        )
    trace_events.sort(key=lambda event: (float(event["ts"]), event["cat"] != "kernel"))
    window_start_ns = min(step.cpu_start_ns, step.gpu_start_ns)
    window_end_ns = max(step.cpu_end_ns, step.gpu_end_ns)
    return (
        trace_events,
        ForwardWindow(
            start_us=window_start_ns / 1000.0,
            end_us=window_end_ns / 1000.0,
            iter_bounds_us=[
                (window_start_ns / 1000.0, window_end_ns / 1000.0)
            ],
            anchor_kernel_count=60,
        ),
        roles,
    )


def _contains(name: str, *needles: str) -> bool:
    lowered = name.lower()
    return any(needle in lowered for needle in needles)


def _anchor_kind(name: str) -> str | None:
    lowered = name.lower()
    if (
        "_causal_conv1d_update_kernel" in lowered
        or "causal_conv1d_fwd_kernel" in lowered
    ):
        return "gdn"
    if "fmhasm10" in lowered:
        return "attention"
    return None


def _moe_node(name: str, *, draft: bool) -> NsysAttribution | None:
    lowered = name.lower()
    prefix = "mtp_moe_block" if draft else "moe_block"
    if "moea2apreparedispatch" in lowered:
        return NsysAttribution(
            f"{prefix}.{'draft' if draft else 'target'}_ep4_pack",
            "MoE EP4 dispatch pack",
        )
    if "moea2adispatchkernel" in lowered or "moea2asanitizeexpertids" in lowered:
        return NsysAttribution(
            f"{prefix}.{'draft' if draft else 'target'}_ep4_dispatch",
            "MoE EP4 dispatch",
        )
    if "moea2apreparecombine" in lowered or "moea2acombinekernel" in lowered:
        return NsysAttribution(
            f"{prefix}.{'draft' if draft else 'target'}_ep4_combine",
            "MoE EP4 combine",
        )
    if _contains(lowered, "custommoeroutingkernel", "routingindicesclusterkernel"):
        return NsysAttribution(f"{prefix}.router", "MoE top-k router")
    if _contains(
        lowered,
        "contiguous_gather_grouped_gemm_act_fusion",
        "contiguous_grouped_gemm_finalize_fusion",
        "groupproblemshape",
        "expandinputrowskernel",
        "doactivationkernel",
    ):
        return NsysAttribution(f"{prefix}.routed_experts", "routed expert GEMM")
    if "silu_and_mul_kernel" in lowered:
        return NsysAttribution(f"{prefix}.shared_expert", "shared expert activation")
    if "sigmoid_gate_mul_add_kernel" in lowered:
        return NsysAttribution(f"{prefix}.weighted_combine", "MoE weighted combine")
    return None


def _direct_node(
    name: str,
    *,
    section: str,
    layer_kind: str | None,
    before_moe: bool | None = None,
) -> NsysAttribution | None:
    lowered = name.lower()
    draft = section == "draft"
    moe = _moe_node(name, draft=draft)
    if moe is not None:
        return moe
    attention_prefix = "mtp_full_attention" if draft else "full_attention"
    if "fmhasm10" in lowered:
        return NsysAttribution(f"{attention_prefix}.causal_gqa", "causal GQA")
    if "_fused_qkv_gemma_rmsnorm_rope_gate_kernel" in lowered:
        return NsysAttribution(
            f"{attention_prefix}.qkv_projection",
            "QKV projection + Q/K norm + RoPE + output-gate projection",
            "fusion",
            (
                f"{attention_prefix}.qk_norm",
                f"{attention_prefix}.partial_rope",
                f"{attention_prefix}.attention_output_gate",
            ),
            "kernel_signature_fusion",
        )
    if "applybiasropeupdatekvcachev2" in lowered:
        return NsysAttribution(
            f"{attention_prefix}.qkv_projection",
            "QKV projection epilogue + RoPE + KV-cache write",
            "fusion",
            (
                f"{attention_prefix}.partial_rope",
                f"{attention_prefix}.kv_state_write",
            ),
            "kernel_signature_fusion",
        )
    if "_fused_sigmoid_mul_kernel" in lowered:
        return NsysAttribution(
            f"{attention_prefix}.attention_output_gate",
            "attention output gate",
        )

    layer_view = (
        "mtp_full_attention_moe_block"
        if draft
        else "gdn_moe_block"
        if layer_kind == "gdn"
        else "full_attention_moe_block"
    )
    if "fused_add_rmsnorm" in lowered and before_moe is not None:
        if before_moe:
            return NsysAttribution(
                f"{layer_view}.attention_residual",
                "attention residual add + pre-MoE RMSNorm",
                "fusion",
                (f"{layer_view}.post_attention_norm",),
                "validated_graph_sequence_fusion",
            )
        return NsysAttribution(
            f"{layer_view}.layer_residual",
            "MoE residual add + next-layer input RMSNorm",
            "fusion",
            (f"{layer_view}.input_norm",),
            "validated_graph_sequence_fusion",
        )

    # TRT's stable CUDA-Graph schedule has four block-scaled QQ GEMMs per
    # target layer. Their exact tile signatures and one-per-layer cardinality
    # distinguish attention input/output, router, and shared-expert work. Do
    # not use a generic QQTST fallback: any unlisted shape remains unmapped.
    if not draft and "nvjet_sm103_qqtst_144x128_128x8" in lowered:
        return NsysAttribution(
            "gdn_attention.qkvz_projection",
            "GDN Q/K/V/Z projection",
            attribution_method="validated_graph_signature_slot",
        )
    if not draft and "nvjet_sm103_tst_32x64_64x16_4x2_2cta" in lowered:
        return NsysAttribution(
            "gdn_attention.ba_projection",
            "GDN B/A gate projection",
            attribution_method="validated_graph_signature_slot",
        )
    if not draft and "nvjet_sm103_qqtst_256x112_128x5" in lowered:
        return NsysAttribution(
            "full_attention.qkv_projection",
            "full-attention Q/K/V projection",
            attribution_method="validated_graph_signature_slot",
        )
    if not draft and "nvjet_sm103_qqtst_112x64_128x14" in lowered:
        output_prefix = "gdn_attention" if layer_kind == "gdn" else "full_attention"
        return NsysAttribution(
            f"{output_prefix}.output_projection",
            "attention output projection",
            attribution_method="validated_graph_signature_slot",
        )
    if not draft and "nvjet_sm103_qqtst_64x64_128x16" in lowered:
        return NsysAttribution(
            "moe_block.router",
            "MoE router projection",
            attribution_method="validated_graph_signature_slot",
        )
    if not draft and "nvjet_sm103_qqtst_64x112_128x14" in lowered:
        return NsysAttribution(
            "moe_block.shared_expert",
            "shared-expert projection",
            attribution_method="validated_graph_signature_slot",
        )

    if draft and "nvjet_sm103_tst_448x64_64x3" in lowered:
        return NsysAttribution(
            "mtp_draft_head.shared_lm_head",
            "MTP shared LM-head projection",
            attribution_method="validated_graph_signature_slot",
        )
    if not draft and layer_kind == "gdn":
        if (
            "_causal_conv1d_update_kernel" in lowered
            or "causal_conv1d_fwd_kernel" in lowered
        ):
            return NsysAttribution("gdn_attention.causal_conv", "GDN causal convolution")
        if "_cached_replay_kernel" in lowered:
            return NsysAttribution(
                "gdn_attention.gated_delta_recurrence", "GDN recurrent update"
            )
        if "gateddeltanetchunkedkernel" in lowered:
            return NsysAttribution(
                "gdn_attention.gated_delta_recurrence",
                "chunked GDN recurrence + final-state write",
                "fusion",
                ("gdn_attention.state_write",),
                "kernel_signature_fusion",
            )
        if "_fused_gdn_post_conv_kernel" in lowered:
            return NsysAttribution(
                "gdn_attention.gated_delta_recurrence",
                "GDN post-convolution gate preparation",
            )
        if "_rms_norm_gated_fwd_multirow_kernel" in lowered:
            return NsysAttribution(
                "gdn_attention.output_gate_norm", "GDN gated output RMSNorm"
            )
        if "_gather_cast_vk_to_fp32_vk_kernel" in lowered:
            return NsysAttribution(
                "gdn_attention.recurrent_state_read", "GDN recurrent-state gather"
            )
        if "_cast_scatter_fp32_vk_to_vk_kernel" in lowered:
            return NsysAttribution(
                "gdn_attention.state_write", "GDN recurrent-state scatter"
            )
    if "_cached_replay_layered_commit_kernel" in lowered:
        return NsysAttribution(
            "generation_loop.commit_gdn", "accepted layered GDN-state commit"
        )
    if "_promote_mamba_state_kernel" in lowered:
        return NsysAttribution(
            "generation_loop.commit_gdn", "accepted GDN state commit"
        )
    if "copybatchblockoffsetstodevicekernel" in lowered:
        return NsysAttribution("generation_loop.commit_kv", "KV block-table commit")
    if _contains(lowered, "sampling", "sampler", "topk", "topp"):
        return NsysAttribution("generation_loop.accept_prefix", "accept/sample")
    return None


def _union_duration_ns(kernels: Iterable[NsysKernel]) -> int:
    intervals = sorted((kernel.start_ns, kernel.end_ns) for kernel in kernels)
    if not intervals:
        return 0
    total = 0
    start, end = intervals[0]
    for next_start, next_end in intervals[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def map_decode_step(step: NsysStep) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attribute one MTP6 CUDA-Graph decode execution to Qwen3.5 IR nodes."""

    kernels = list(step.kernels)
    prepare = [
        index
        for index, kernel in enumerate(kernels)
        if "moea2apreparedispatch" in kernel.name.lower()
    ]
    if len(prepare) != 66:
        raise ValueError(f"step {step.step_id}: expected 66 MoE calls, got {len(prepare)}")

    target_prepare = prepare[:60]
    target_anchors: list[int] = []
    kinds: list[str] = []
    previous_prepare = -1
    for prepare_index in target_prepare:
        candidates = [
            (index, _anchor_kind(kernels[index].name))
            for index in range(previous_prepare + 1, prepare_index)
            if _anchor_kind(kernels[index].name) is not None
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"step {step.step_id}: target MoE call {len(target_anchors)} has "
                f"{len(candidates)} GDN/attention anchors"
            )
        anchor_index, kind = candidates[0]
        target_anchors.append(anchor_index)
        kinds.append(str(kind))
        previous_prepare = prepare_index
    if tuple(kinds) != TARGET_PATTERN:
        raise ValueError(
            f"step {step.step_id}: target layer order is not exact 45-GDN/15-attention GGGA"
        )

    last_target_prepare = target_prepare[-1]
    draft_anchors = [
        index
        for index in range(last_target_prepare + 1, len(kernels))
        if _anchor_kind(kernels[index].name) == "attention"
    ]
    if len(draft_anchors) != 6:
        raise ValueError(
            f"step {step.step_id}: expected six MTP attention passes, got {len(draft_anchors)}"
        )
    if any(not draft_anchors[index] < prepare[60 + index] for index in range(6)):
        raise ValueError(f"step {step.step_id}: MTP attention/MoE order mismatch")

    mapped: list[dict[str, Any]] = []
    total_ns = sum(kernel.end_ns - kernel.start_ns for kernel in kernels)
    status_ns: Counter[str] = Counter()
    for event_index, kernel in enumerate(kernels):
        if event_index < draft_anchors[0]:
            section = "target"
            layer_id = max(0, bisect_right(target_anchors, event_index) - 1)
            layer_kind = kinds[layer_id]
        else:
            section = "draft"
            layer_id = None
            layer_kind = "attention"
        mtp_round = (
            max(0, bisect_right(draft_anchors, event_index) - 1)
            if section == "draft"
            else None
        )

        if section == "target":
            before_moe = event_index < target_prepare[int(layer_id)]
        else:
            before_moe = event_index < prepare[60 + int(mtp_round)]
        direct = _direct_node(
            kernel.name,
            section=section,
            layer_kind=layer_kind,
            before_moe=before_moe,
        )
        candidates: list[str] = []
        unmapped_reason = None
        if direct is None:
            status = "unmapped"
            node = None
            label = f"Unmapped TRT {section} kernel"
            unmapped_reason = (
                "Graph occurrence identifies the containing target layer but not a unique leaf operation."
                if section == "target"
                else "Graph occurrence identifies the MTP draft range but not a unique draft leaf operation."
            )
            layer_view = (
                "gdn_moe_block" if layer_kind == "gdn" else "full_attention_moe_block"
            )
            candidates = (
                [
                    f"{layer_view}.attention",
                    f"{layer_view}.moe",
                    f"{layer_view}.attention_residual",
                    f"{layer_view}.layer_residual",
                ]
                if section == "target"
                else [
                    "mtp_full_attention_moe_block.attention",
                    "mtp_full_attention_moe_block.moe",
                    "mtp_draft_head.fc_projection",
                    "mtp_draft_head.shared_lm_head",
                ]
            )
        else:
            node = direct.node
            label = direct.label
            status = direct.status
        duration_ns = kernel.end_ns - kernel.start_ns
        status_ns[status] += duration_ns
        ir_targets = list(direct.ir_targets) if direct is not None else []
        if section == "target" and (node is None or not node.startswith("generation_loop.")):
            ir_targets.append("generation_loop.target_verify")
        if section == "draft":
            ir_targets.append("generation_loop.draft_propose")
        mapped.append(
            {
                "event_id": f"r{step.rank}-s{step.step_id}-k{event_index}",
                "engine": "trtllm",
                "rank": step.rank,
                "device": step.rank,
                "step_index": step.step_id,
                "kernel_name": kernel.name,
                "kernel_label": label,
                "node": node,
                "ir_targets": ir_targets,
                "mapping_status": status,
                "fusion_group": (
                    f"r{step.rank}-s{step.step_id}-k{event_index}"
                    if status == "fusion"
                    else None
                ),
                "attribution_method": (
                    direct.attribution_method if direct is not None else "unresolved"
                ),
                "confidence": direct.confidence if direct is not None else "unknown",
                "unmapped_reason": unmapped_reason,
                "candidate_nodes": candidates,
                "layer_id": layer_id,
                "layer_kind": layer_kind,
                "substage": section,
                "mtp_round": mtp_round,
                "ts_us": kernel.start_ns / 1000.0,
                "dur_us": kernel.duration_us,
                "stream": kernel.stream,
                "kernel_kind": (
                    "communication"
                    if _contains(kernel.name, "moea2adispatch", "moea2acombine")
                    else "compute"
                ),
                "graph_id": kernel.graph_id,
                "graph_node_id": kernel.graph_node_id,
                "correlation_id": kernel.correlation_id,
            }
        )

    attributed_ns = status_ns["mapped"] + status_ns["fusion"]
    strict_signature_ns = sum(
        float(event["dur_us"]) * 1000.0
        for event in mapped
        if event.get("attribution_method") == "unique_kernel_signature"
    )
    validation = {
        "step_id": step.step_id,
        "rank": step.rank,
        "context_reqs": step.context_reqs,
        "context_tokens": step.context_tokens,
        "generation_reqs": step.generation_reqs,
        "kernel_count": len(kernels),
        "graph_launch_count": step.graph_launch_count,
        "target_gdn_layers": kinds.count("gdn"),
        "target_attention_layers": kinds.count("attention"),
        "target_ep4_dispatch": sum(
            "moea2adispatchkernel" in kernel.name.lower()
            for kernel in kernels[: draft_anchors[0]]
        ),
        "target_ep4_combine": sum(
            "moea2acombinekernel" in kernel.name.lower()
            for kernel in kernels[: draft_anchors[0]]
        ),
        "draft_ep4_dispatch": sum(
            "moea2adispatchkernel" in kernel.name.lower()
            for kernel in kernels[draft_anchors[0] :]
        ),
        "draft_ep4_combine": sum(
            "moea2acombinekernel" in kernel.name.lower()
            for kernel in kernels[draft_anchors[0] :]
        ),
        "mtp_passes": len(draft_anchors),
        "cpu_wall_us": step.cpu_wall_us,
        "gpu_span_us": (step.gpu_end_ns - step.gpu_start_ns) / 1000.0,
        "gpu_busy_union_us": _union_duration_ns(kernels) / 1000.0,
        "gpu_residency_us": total_ns / 1000.0,
        "timing_closure_us": (
            sum(float(event["dur_us"]) for event in mapped) - total_ns / 1000.0
        ),
        "status_duration_us": {
            status: duration / 1000.0 for status, duration in sorted(status_ns.items())
        },
        "attributed_duration_ratio": attributed_ns / total_ns if total_ns else 0.0,
        "strict_signature_duration_ratio": (
            strict_signature_ns / total_ns if total_ns else 0.0
        ),
        "timeline_interval_coverage_ratio": (
            sum(status_ns.values()) / total_ns if total_ns else 0.0
        ),
    }
    expected = {
        "target_gdn_layers": 45,
        "target_attention_layers": 15,
        "target_ep4_dispatch": 60,
        "target_ep4_combine": 60,
        "draft_ep4_dispatch": 6,
        "draft_ep4_combine": 6,
        "mtp_passes": 6,
    }
    mismatch = {
        key: {"expected": value, "actual": validation[key]}
        for key, value in expected.items()
        if validation[key] != value
    }
    if mismatch:
        raise ValueError(f"step {step.step_id}: structural validation failed: {mismatch}")
    if abs(validation["timing_closure_us"]) > 1e-6:
        raise ValueError(f"step {step.step_id}: kernel timing does not close")
    return mapped, validation


def map_prefill_step(step: NsysStep) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attribute one TRT-LLM AgentX target-prefill scheduler step.

    Attention-DP owner ranks execute the full target model; the other EP ranks
    can contain only the same 60 dispatch/combine calls.  Both are first-class
    observations and are labelled explicitly instead of pretending collective-
    only ranks ran attention/GDN compute.
    """

    kernels = list(step.kernels)
    prepare = [
        index
        for index, kernel in enumerate(kernels)
        if "moea2apreparedispatch" in kernel.name.lower()
    ]
    if len(prepare) != 60:
        raise ValueError(
            f"prefill step {step.step_id}: expected 60 target MoE calls, got {len(prepare)}"
        )
    anchor_pairs = [
        (index, _anchor_kind(kernel.name))
        for index, kernel in enumerate(kernels)
        if _anchor_kind(kernel.name) is not None
    ]
    owner_compute = bool(anchor_pairs)
    if owner_compute and tuple(kind for _index, kind in anchor_pairs) != TARGET_PATTERN:
        raise ValueError(
            f"prefill step {step.step_id}: owner-rank layer order is not exact GGGA"
        )
    if owner_compute:
        layer_anchors = [index for index, _kind in anchor_pairs]
        layer_kinds = [str(kind) for _index, kind in anchor_pairs]
    else:
        layer_anchors = prepare
        layer_kinds = list(TARGET_PATTERN)

    mapped: list[dict[str, Any]] = []
    status_ns: Counter[str] = Counter()
    for event_index, kernel in enumerate(kernels):
        layer_id = max(0, bisect_right(layer_anchors, event_index) - 1)
        layer_kind = layer_kinds[layer_id]
        direct = _direct_node(
            kernel.name,
            section="target",
            layer_kind=layer_kind,
            before_moe=event_index < prepare[layer_id],
        )
        candidates: list[str] = []
        unmapped_reason = None
        if direct is None:
            node = None
            label = "Unmapped TRT target-prefill kernel"
            status = "unmapped"
            unmapped_reason = (
                "Owner-rank layer occurrence does not uniquely identify a leaf operation."
                if owner_compute
                else "Collective-only EP-rank occurrence has no unique target-model leaf attribution."
            )
            layer_view = (
                "gdn_moe_block" if layer_kind == "gdn" else "full_attention_moe_block"
            )
            candidates = [f"{layer_view}.attention", f"{layer_view}.moe"]
        else:
            node = direct.node
            label = direct.label
            status = direct.status
        duration_ns = kernel.end_ns - kernel.start_ns
        status_ns[status] += duration_ns
        mapped.append(
            {
                "event_id": f"r{step.rank}-p{step.step_id}-k{event_index}",
                "engine": "trtllm",
                "rank": step.rank,
                "device": step.rank,
                "step_index": step.step_id,
                "kernel_name": kernel.name,
                "kernel_label": label,
                "node": node,
                "ir_targets": list(direct.ir_targets) if direct is not None else [],
                "mapping_status": status,
                "fusion_group": (
                    f"r{step.rank}-p{step.step_id}-k{event_index}"
                    if status == "fusion"
                    else None
                ),
                "attribution_method": (
                    direct.attribution_method if direct is not None else "unresolved"
                ),
                "confidence": direct.confidence if direct is not None else "unknown",
                "unmapped_reason": unmapped_reason,
                "candidate_nodes": candidates,
                "layer_id": layer_id,
                "layer_kind": layer_kind,
                "substage": "target_prefill",
                "owner_compute": owner_compute,
                "ts_us": kernel.start_ns / 1000.0,
                "dur_us": kernel.duration_us,
                "stream": kernel.stream,
                "kernel_kind": (
                    "communication"
                    if _contains(kernel.name, "moea2adispatch", "moea2acombine")
                    else "compute"
                ),
                "graph_id": kernel.graph_id,
                "graph_node_id": kernel.graph_node_id,
                "correlation_id": kernel.correlation_id,
            }
        )

    total_ns = sum(kernel.end_ns - kernel.start_ns for kernel in kernels)
    strict_signature_ns = sum(
        float(event["dur_us"]) * 1000.0
        for event in mapped
        if event.get("attribution_method") == "unique_kernel_signature"
    )
    validation = {
        "step_id": step.step_id,
        "rank": step.rank,
        "context_reqs": step.context_reqs,
        "context_tokens": step.context_tokens,
        "generation_reqs": step.generation_reqs,
        "owner_compute": owner_compute,
        "kernel_count": len(kernels),
        "target_gdn_layers": sum(kind == "gdn" for _index, kind in anchor_pairs),
        "target_attention_layers": sum(
            kind == "attention" for _index, kind in anchor_pairs
        ),
        "target_ep4_dispatch": sum(
            "moea2adispatchkernel" in kernel.name.lower() for kernel in kernels
        ),
        "target_ep4_combine": sum(
            "moea2acombinekernel" in kernel.name.lower() for kernel in kernels
        ),
        "cpu_wall_us": step.cpu_wall_us,
        "gpu_span_us": (step.gpu_end_ns - step.gpu_start_ns) / 1000.0,
        "gpu_busy_union_us": _union_duration_ns(kernels) / 1000.0,
        "gpu_residency_us": total_ns / 1000.0,
        "timing_closure_us": (
            sum(float(event["dur_us"]) for event in mapped) - total_ns / 1000.0
        ),
        "status_duration_us": {
            status: duration / 1000.0 for status, duration in sorted(status_ns.items())
        },
        "attributed_duration_ratio": (
            (status_ns["mapped"] + status_ns["fusion"]) / total_ns if total_ns else 0.0
        ),
        "strict_signature_duration_ratio": (
            strict_signature_ns / total_ns if total_ns else 0.0
        ),
        "timeline_interval_coverage_ratio": (
            sum(status_ns.values()) / total_ns if total_ns else 0.0
        ),
    }
    if validation["target_ep4_dispatch"] != 60 or validation["target_ep4_combine"] != 60:
        raise ValueError(f"prefill step {step.step_id}: incomplete EP4 collectives")
    if owner_compute and (
        validation["target_gdn_layers"] != 45
        or validation["target_attention_layers"] != 15
    ):
        raise ValueError(f"prefill step {step.step_id}: incomplete target-model layers")
    if abs(validation["timing_closure_us"]) > 1e-6:
        raise ValueError(f"prefill step {step.step_id}: kernel timing does not close")
    return mapped, validation
