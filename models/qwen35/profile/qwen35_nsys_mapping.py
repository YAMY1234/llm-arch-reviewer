#!/usr/bin/env python3
"""Parse and attribute worker-local TensorRT-LLM Nsight Systems reports.

TensorRT-LLM launches the decode model as a CUDA Graph.  Nsight records graph
child kernels with correlation id zero, so a normal runtime-correlation join
cannot associate them with ``[Executor] _forward_step`` NVTX ranges.  The
stable identity is ``graphNodeId``: each kernel-bearing node occurs exactly
once per launch.  We therefore split overlapping graph executions by the
occurrence index of each graph node and pair those executions with the ordered
``cudaGraphLaunch`` calls.  Direct launches remain associated by correlation
id.  The parser fails closed if either invariant does not hold.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterable


STEP_RE = re.compile(
    r"\[Executor\] _forward_step (?P<step>\d+): "
    r"(?P<context_reqs>\d+) ctx reqs, (?P<context_tokens>\d+) ctx tokens, "
    r"(?P<generation_reqs>\d+) gen reqs"
)
RANK_RE = re.compile(r"rank(?P<rank>\d+)")
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


def _strings(connection: sqlite3.Connection) -> dict[int, str]:
    return {int(key): str(value) for key, value in connection.execute(
        "select id, value from StringIds"
    )}


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
    connection: sqlite3.Connection, strings: dict[int, str]
) -> list[NsysKernel]:
    return [
        NsysKernel(
            start_ns=int(start),
            end_ns=int(end),
            name=str(strings.get(int(name_id), f"StringIds[{name_id}]")),
            stream=int(stream),
            correlation_id=int(correlation),
            graph_id=None if graph_id is None else int(graph_id),
            graph_node_id=None if graph_node_id is None else int(graph_node_id),
        )
        for start, end, stream, correlation, graph_id, graph_node_id, name_id
        in connection.execute(
            "select start, end, streamId, correlationId, graphId, graphNodeId, "
            "demangledName from CUPTI_ACTIVITY_KIND_KERNEL order by start"
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


def _moe_node(name: str, *, draft: bool) -> tuple[str, str, str] | None:
    lowered = name.lower()
    prefix = "mtp_moe_block" if draft else "moe_block"
    if "moea2apreparedispatch" in lowered:
        return (f"{prefix}.{'draft' if draft else 'target'}_ep4_pack", "MoE EP4 dispatch pack", "mapped")
    if "moea2adispatchkernel" in lowered or "moea2asanitizeexpertids" in lowered:
        return (f"{prefix}.{'draft' if draft else 'target'}_ep4_dispatch", "MoE EP4 dispatch", "mapped")
    if "moea2apreparecombine" in lowered or "moea2acombinekernel" in lowered:
        return (f"{prefix}.{'draft' if draft else 'target'}_ep4_combine", "MoE EP4 combine", "mapped")
    if _contains(lowered, "custommoeroutingkernel", "routingindicesclusterkernel"):
        return (f"{prefix}.router", "MoE top-k router", "mapped")
    if _contains(
        lowered,
        "contiguous_gather_grouped_gemm_act_fusion",
        "contiguous_grouped_gemm_finalize_fusion",
        "nvjet_sm103_qqtst",
    ):
        return (f"{prefix}.routed_experts", "routed expert GEMM", "mapped")
    if "silu_and_mul_kernel" in lowered:
        return (f"{prefix}.shared_expert", "shared expert activation", "mapped")
    if "sigmoid_gate_mul_add_kernel" in lowered:
        return (f"{prefix}.weighted_combine", "MoE weighted combine", "mapped")
    return None


def _direct_node(
    name: str, *, section: str, layer_kind: str | None
) -> tuple[str, str, str] | None:
    lowered = name.lower()
    draft = section == "draft"
    moe = _moe_node(name, draft=draft)
    if moe is not None:
        return moe
    attention_prefix = "mtp_full_attention_moe_block" if draft else "full_attention_moe_block"
    if "fmhasm10" in lowered:
        return (f"{attention_prefix}.causal_gqa", "causal GQA", "mapped")
    if not draft and layer_kind == "gdn":
        if (
            "_causal_conv1d_update_kernel" in lowered
            or "causal_conv1d_fwd_kernel" in lowered
        ):
            return ("gdn_moe_block.causal_conv", "GDN causal convolution", "mapped")
        if "_cached_replay_kernel" in lowered:
            return ("gdn_moe_block.gated_delta_recurrence", "GDN recurrent update", "mapped")
    if "_promote_mamba_state_kernel" in lowered:
        return ("generation_loop.commit_gdn", "accepted GDN state commit", "mapped")
    if "copybatchblockoffsetstodevicekernel" in lowered:
        return ("generation_loop.commit_kv", "KV block-table commit", "mapped")
    if _contains(lowered, "sampling", "sampler", "topk", "topp"):
        return ("generation_loop.accept_prefix", "accept/sample", "mapped")
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

        direct = _direct_node(kernel.name, section=section, layer_kind=layer_kind)
        if direct is None and section == "target":
            view = "gdn_moe_block" if layer_kind == "gdn" else "full_attention_moe_block"
            direct = (f"{view}.output_hidden", "target layer fused/auxiliary kernel", "fusion")
        elif direct is None:
            direct = (
                "mtp_draft_head.draft_decoder_layer",
                "MTP draft fused/auxiliary kernel",
                "fusion",
            )
        node, label, status = direct
        duration_ns = kernel.end_ns - kernel.start_ns
        status_ns[status] += duration_ns
        ir_targets = []
        if layer_id is not None:
            ir_targets.append(f"layer_schedule.layer_{layer_id:02d}")
        if section == "target" and not node.startswith("generation_loop."):
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
                    "unique_kernel_signature" if status == "mapped" else "validated_graph_scope"
                ),
                "confidence": "high" if status == "mapped" else "structural",
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
        "strict_signature_duration_ratio": status_ns["mapped"] / total_ns if total_ns else 0.0,
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
    if validation["attributed_duration_ratio"] < 0.90:
        raise ValueError(f"step {step.step_id}: attributed residency is below 90%")
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
        direct = _direct_node(kernel.name, section="target", layer_kind=layer_kind)
        if direct is None:
            if owner_compute:
                view = "gdn_moe_block" if layer_kind == "gdn" else "full_attention_moe_block"
                direct = (
                    f"{view}.output_hidden",
                    "target prefill layer fused/auxiliary kernel",
                    "fusion",
                )
            else:
                direct = (
                    "top.decoder_stack",
                    "collective-only EP-rank auxiliary kernel",
                    "fusion",
                )
        node, label, status = direct
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
                "ir_targets": [f"layer_schedule.layer_{layer_id:02d}"],
                "mapping_status": status,
                "fusion_group": (
                    f"r{step.rank}-p{step.step_id}-k{event_index}"
                    if status == "fusion"
                    else None
                ),
                "attribution_method": (
                    "unique_kernel_signature"
                    if status == "mapped"
                    else "validated_owner_layer_scope"
                    if owner_compute
                    else "validated_collective_only_rank_scope"
                ),
                "confidence": "high" if status == "mapped" else "structural",
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
            status_ns["mapped"] / total_ns if total_ns else 0.0
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
    if validation["attributed_duration_ratio"] < 0.90:
        raise ValueError(f"prefill step {step.step_id}: attributed residency is below 90%")
    return mapped, validation
