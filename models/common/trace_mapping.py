#!/usr/bin/env python3
"""Generic PyTorch-profiler trace mapping engine.

Model-specific code should provide a small :class:`TraceMappingRules` object:
how to slice forwards, how to choose semantic stack frames, and how to map a
kernel + stack to an IR node. The rest of the pipeline is deliberately common.
"""

from __future__ import annotations

import gzip
import json
import re
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class FrameRef:
    raw: str
    file: str | None = None
    line: int | None = None
    function: str | None = None
    module: str | None = None
    source_exists: bool | None = None


@dataclass(frozen=True)
class KernelEvent:
    event_id: str
    kernel_name: str
    ts_us: float
    dur_us: float
    stream: int | str | None
    device: int | str | None
    correlation: int | None
    external_id: int | None
    cpu_op_name: str | None
    cpu_input_dims: Any
    cpu_input_types: Any
    python_stack: list[FrameRef]


@dataclass(frozen=True)
class KernelMapping:
    event_id: str
    kernel_name: str
    selected_node: str | None
    confidence: str
    primitive_frame: FrameRef | None
    operator_frame: FrameRef | None
    semantic_frame: FrameRef | None
    model_context_frame: FrameRef | None
    phase_frame: FrameRef | None
    cpu_op_name: str | None
    evidence: list[str]


@dataclass(frozen=True)
class ForwardWindow:
    start_us: float
    end_us: float
    iter_bounds_us: list[tuple[float, float]]
    anchor_kernel_count: int


@dataclass(frozen=True)
class BuildResult:
    manifest: dict[str, Any]
    events: list[KernelEvent]
    mappings: list[KernelMapping]
    validation: dict[str, Any]
    stack_samples: list[dict[str, Any]]


@dataclass(frozen=True)
class StackFrameRules:
    operator_patterns: tuple[str, ...]
    semantic_patterns: tuple[str, ...]
    model_context_patterns: tuple[str, ...]
    phase_patterns: tuple[str, ...] = (
        "forward_extend",
        "forward_decode",
        "cuda_graph_runner",
        "replay",
    )


ClassifyNodeFn = Callable[[str, Optional[str], list[FrameRef]], tuple[Optional[str], str]]


@dataclass(frozen=True)
class TraceMappingRules:
    model_id: str
    signature_kernel: str
    signature_count_per_forward: int
    stack: StackFrameRules
    classify_node: ClassifyNodeFn


def load_trace(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as fh:
            return json.load(fh)
    with path.open() as fh:
        return json.load(fh)


def _args(event: dict[str, Any]) -> dict[str, Any]:
    return event.get("args") or {}


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _event_end_us(event: dict[str, Any]) -> float:
    return float(event.get("ts", 0.0)) + float(event.get("dur", 0.0))


def _kernel_stream(event: dict[str, Any]) -> int | str | None:
    return _args(event).get("stream")


def _kernel_device(event: dict[str, Any]) -> int | str | None:
    return _args(event).get("device")


def find_forward_windows(
    trace_events: list[dict[str, Any]],
    *,
    signature: str,
    expected_per_forward: int,
) -> list[ForwardWindow]:
    """Find repeated forward windows using a model-provided signature kernel."""

    anchors = sorted(
        float(event.get("ts", 0.0))
        for event in trace_events
        if event.get("cat") == "kernel" and signature in event.get("name", "")
    )
    windows: list[ForwardWindow] = []
    i = 0
    while i + expected_per_forward <= len(anchors):
        start = anchors[i]
        if i + expected_per_forward < len(anchors):
            end = anchors[i + expected_per_forward] - 0.001
        else:
            end = anchors[i + expected_per_forward - 1] + 5000.0
        windows.append(
            ForwardWindow(
                start_us=start,
                end_us=end,
                iter_bounds_us=[(start, end)],
                anchor_kernel_count=expected_per_forward,
            )
        )
        i += expected_per_forward
    return windows


def find_step_annotation_windows(
    trace_events: list[dict[str, Any]],
    *,
    phase: str,
    signature: str | None = None,
) -> list[ForwardWindow]:
    """Find full model steps from SGLang GPU annotations.

    A GPU-side ``step[EXTEND ...]`` / ``step[DECODE ...]`` range is a more
    complete forward boundary than a model-specific kernel anchor: it includes
    work before the first repeated layer kernel and after the last one.
    """

    phase_lower = phase.lower()
    if phase_lower in {"eagle_mtp_prefill", "mtp_prefill"}:
        return find_eagle_mtp_prefill_windows(trace_events, signature=signature)
    if phase_lower in {"eagle_mtp_decode", "mtp_decode"}:
        return find_eagle_mtp_decode_windows(trace_events, signature=signature)
    if "decode" in phase_lower:
        step_kind = "DECODE"
    elif "extend" in phase_lower or "prefill" in phase_lower:
        step_kind = "EXTEND"
    else:
        return []

    kernels = [event for event in trace_events if event.get("cat") == "kernel"]
    windows: list[ForwardWindow] = []
    for event in trace_events:
        if event.get("cat") != "gpu_user_annotation" or event.get("ph") != "X":
            continue
        if not str(event.get("name", "")).startswith(f"step[{step_kind}"):
            continue
        start = float(event.get("ts", 0.0))
        end = _event_end_us(event)
        anchor_count = 0
        if signature:
            anchor_count = sum(
                1
                for kernel in kernels
                if signature in str(kernel.get("name", ""))
                and start <= float(kernel.get("ts", 0.0)) <= end
            )
        windows.append(
            ForwardWindow(
                start_us=start,
                end_us=end,
                iter_bounds_us=[(start, end)],
                anchor_kernel_count=anchor_count,
            )
        )
    return sorted(windows, key=lambda window: window.start_us)


def _primary_gpu_annotations(
    trace_events: list[dict[str, Any]], *, name_prefix: str
) -> tuple[list[dict[str, Any]], tuple[Any, Any] | None]:
    """Return annotations from the GPU track carrying the full stage span."""

    tracks: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    for event in trace_events:
        if event.get("cat") != "gpu_user_annotation" or event.get("ph") != "X":
            continue
        if not str(event.get("name", "")).startswith(name_prefix):
            continue
        tracks.setdefault((event.get("pid"), event.get("tid")), []).append(event)
    if not tracks:
        return [], None
    track_key, annotations = max(
        tracks.items(),
        key=lambda item: sum(float(event.get("dur", 0.0)) for event in item[1]),
    )
    return sorted(annotations, key=lambda event: float(event.get("ts", 0.0))), track_key


def _anchor_count(
    kernels: list[dict[str, Any]], *, signature: str | None, start: float, end: float
) -> int:
    if not signature:
        return 0
    return sum(
        1
        for kernel in kernels
        if signature in str(kernel.get("name", ""))
        and start <= float(kernel.get("ts", 0.0)) <= end
    )


def find_eagle_mtp_prefill_windows(
    trace_events: list[dict[str, Any]], *, signature: str | None = None
) -> list[ForwardWindow]:
    """Pair target prefill and auxiliary MTP seed prefill as one stage."""

    annotations, _track = _primary_gpu_annotations(
        trace_events, name_prefix="step[EXTEND"
    )
    kernels = [event for event in trace_events if event.get("cat") == "kernel"]
    windows = []
    for index in range(0, len(annotations) - 1, 2):
        target = annotations[index]
        auxiliary = annotations[index + 1]
        start = float(target.get("ts", 0.0))
        end = _event_end_us(auxiliary)
        if float(auxiliary.get("ts", 0.0)) < _event_end_us(target):
            continue
        windows.append(
            ForwardWindow(
                start_us=start,
                end_us=end,
                iter_bounds_us=[(start, end)],
                anchor_kernel_count=_anchor_count(
                    kernels, signature=signature, start=start, end=end
                ),
            )
        )
    return windows


def find_eagle_mtp_decode_windows(
    trace_events: list[dict[str, Any]], *, signature: str | None = None
) -> list[ForwardWindow]:
    """Build complete EAGLE decode iterations from target verify to draft select."""

    targets, track = _primary_gpu_annotations(
        trace_events, name_prefix="step[TARGET_VERIFY"
    )
    if not targets or track is None:
        return []
    stage_events = sorted(
        (
            event
            for event in trace_events
            if event.get("cat") == "gpu_user_annotation"
            and event.get("ph") == "X"
            and (event.get("pid"), event.get("tid")) == track
            and (
                str(event.get("name", "")).startswith("step[DRAFT_EXTEND_V2")
                or str(event.get("name", "")) in {"draft_extend", "draft"}
            )
        ),
        key=lambda event: float(event.get("ts", 0.0)),
    )
    kernels = [event for event in trace_events if event.get("cat") == "kernel"]
    windows = []
    for index, target in enumerate(targets):
        start = float(target.get("ts", 0.0))
        next_start = (
            float(targets[index + 1].get("ts", 0.0))
            if index + 1 < len(targets)
            else None
        )
        following = [
            event
            for event in stage_events
            if float(event.get("ts", 0.0)) >= _event_end_us(target)
            and (next_start is None or float(event.get("ts", 0.0)) < next_start)
        ]
        if not any(
            str(event.get("name", "")).startswith("step[DRAFT_EXTEND_V2")
            for event in following
        ):
            continue
        end = next_start if next_start is not None else max(
            _event_end_us(target), *(_event_end_us(event) for event in following)
        )
        windows.append(
            ForwardWindow(
                start_us=start,
                end_us=end,
                iter_bounds_us=[(start, end)],
                anchor_kernel_count=_anchor_count(
                    kernels, signature=signature, start=start, end=end
                ),
            )
        )
    return windows


def choose_forward_window(
    windows: list[ForwardWindow],
    *,
    expect_ms: float | None,
    n_iters: int,
    skip_first: bool = True,
) -> ForwardWindow:
    if not windows:
        raise ValueError("no forward windows found")
    start_idx = 1 if skip_first and len(windows) > 1 else 0
    candidates = windows[start_idx:]
    n_iters = max(1, min(n_iters, len(candidates)))

    if expect_ms is None:
        chosen_start = 0
    else:
        durations = [
            (window.end_us - window.start_us) / 1000.0 for window in candidates
        ]
        chosen_start = min(
            range(0, len(candidates) - n_iters + 1),
            key=lambda idx: max(
                abs(durations[idx + offset] - expect_ms)
                for offset in range(n_iters)
            ),
        )
    selected = candidates[chosen_start : chosen_start + n_iters]
    return ForwardWindow(
        start_us=selected[0].start_us,
        end_us=selected[-1].end_us,
        iter_bounds_us=[(w.start_us, w.end_us) for w in selected],
        anchor_kernel_count=sum(w.anchor_kernel_count for w in selected),
    )


_FILE_FRAME_RE = re.compile(r"(?P<file>[^()]+\.py)\((?P<line>\d+)\): (?P<func>.+)$")
_MODULE_FRAME_RE = re.compile(r"nn\.Module: (?P<module>.+)$")


def parse_frame(raw: str, source_root: Path | None = None) -> FrameRef:
    file: str | None = None
    line: int | None = None
    function: str | None = None
    module: str | None = None

    file_match = _FILE_FRAME_RE.search(raw)
    if file_match:
        file = file_match.group("file")
        line = int(file_match.group("line"))
        function = file_match.group("func")
    else:
        module_match = _MODULE_FRAME_RE.search(raw)
        if module_match:
            module = module_match.group("module")
            function = module
        else:
            function = raw

    source_exists = None
    if source_root and file:
        source_exists = (source_root / file).exists()

    return FrameRef(
        raw=raw,
        file=file,
        line=line,
        function=function,
        module=module,
        source_exists=source_exists,
    )


def _python_id(event: dict[str, Any]) -> int | None:
    return _as_int(_args(event).get("Python id"))


def _python_parent_id(event: dict[str, Any]) -> int | None:
    return _as_int(_args(event).get("Python parent id"))


def _filter_python_spans(
    trace_events: list[dict[str, Any]], start_us: float, end_us: float
) -> list[dict[str, Any]]:
    out = []
    for event in trace_events:
        if event.get("cat") != "python_function" or event.get("ph") != "X":
            continue
        if _python_id(event) is None:
            continue
        event_start = float(event.get("ts", 0.0))
        event_end = _event_end_us(event)
        if event_end >= start_us and event_start <= end_us:
            out.append(event)
    return out


def assign_deepest_python_spans(
    python_spans: list[dict[str, Any]],
    targets_us: dict[str, float],
) -> dict[str, dict[str, Any] | None]:
    """Assign each target timestamp to the smallest active Python span."""

    actions: list[tuple[float, int, str, int | None]] = []
    span_by_id: dict[int, dict[str, Any]] = {}
    for span in python_spans:
        py_id = _python_id(span)
        if py_id is None:
            continue
        span_by_id[py_id] = span
        actions.append((float(span.get("ts", 0.0)), 0, "start", py_id))
        actions.append((_event_end_us(span), 2, "end", py_id))
    for target_id, ts in targets_us.items():
        actions.append((ts, 1, target_id, None))
    actions.sort()

    active: set[int] = set()
    assigned: dict[str, dict[str, Any] | None] = {}
    depth_cache: dict[int, int] = {}

    def span_depth(py_id: int) -> int:
        if py_id in depth_cache:
            return depth_cache[py_id]
        depth = 0
        current = span_by_id.get(py_id)
        seen: set[int] = set()
        while current is not None:
            parent = _python_parent_id(current)
            if parent is None or parent in seen:
                break
            seen.add(parent)
            depth += 1
            current = span_by_id.get(parent)
        depth_cache[py_id] = depth
        return depth

    for _ts, _order, kind, py_id in actions:
        if kind == "start" and py_id is not None:
            active.add(py_id)
        elif kind == "end" and py_id is not None:
            active.discard(py_id)
        else:
            best_id = min(
                active,
                key=lambda item: (
                    float(span_by_id[item].get("dur", 0.0)),
                    -span_depth(item),
                    -float(span_by_id[item].get("ts", 0.0)),
                ),
                default=None,
            )
            assigned[kind] = span_by_id.get(best_id) if best_id is not None else None
    return assigned


def build_python_chain(
    deepest: dict[str, Any] | None,
    python_by_id: dict[int, dict[str, Any]],
    *,
    source_root: Path | None = None,
    max_depth: int = 64,
) -> list[FrameRef]:
    if not deepest:
        return []
    chain: list[FrameRef] = []
    current = deepest
    seen: set[int] = set()
    for _ in range(max_depth):
        raw = str(current.get("name", ""))
        chain.append(parse_frame(raw, source_root=source_root))
        parent_id = _python_parent_id(current)
        if parent_id is None or parent_id in seen:
            break
        seen.add(parent_id)
        parent = python_by_id.get(parent_id)
        if parent is None:
            break
        current = parent
    return chain


def _frame_contains(frame: FrameRef | None, needle: str) -> bool:
    return bool(frame and needle in frame.raw)


def _first_frame(stack: Iterable[FrameRef], patterns: Iterable[str]) -> FrameRef | None:
    needles = tuple(patterns)
    for frame in stack:
        if any(needle in frame.raw for needle in needles):
            return frame
    return None


def choose_stack_frames(
    stack: list[FrameRef],
    rules: StackFrameRules,
) -> tuple[FrameRef | None, FrameRef | None, FrameRef | None, FrameRef | None, FrameRef | None]:
    primitive = stack[0] if stack else None
    operator = _first_frame(stack, rules.operator_patterns)
    semantic = _first_frame(stack, rules.semantic_patterns)
    model_context = _first_frame(stack, rules.model_context_patterns)
    phase = _first_frame(stack, rules.phase_patterns)
    return primitive, operator, semantic, model_context, phase


def _build_cpu_maps(
    trace_events: list[dict[str, Any]],
) -> tuple[dict[int, dict[str, Any]], dict[int, int], dict[int, dict[str, Any]]]:
    cpu_by_external: dict[int, dict[str, Any]] = {}
    runtime_corr_to_external: dict[int, int] = {}
    runtime_by_corr: dict[int, dict[str, Any]] = {}
    for event in trace_events:
        args = _args(event)
        if event.get("cat") == "cpu_op":
            external = _as_int(args.get("External id"))
            if external is not None:
                cpu_by_external[external] = event
        elif event.get("cat") == "cuda_runtime":
            corr = _as_int(args.get("correlation"))
            external = _as_int(args.get("External id"))
            if corr is not None:
                runtime_by_corr[corr] = event
            if corr is not None and external is not None:
                runtime_corr_to_external[corr] = external
    return cpu_by_external, runtime_corr_to_external, runtime_by_corr


def normalize_kernel_events(
    trace_events: list[dict[str, Any]],
    *,
    window: ForwardWindow,
    source_root: Path | None = None,
) -> list[KernelEvent]:
    cpu_by_external, runtime_corr_to_external, runtime_by_corr = _build_cpu_maps(
        trace_events
    )
    raw_kernels = [
        event
        for event in trace_events
        if event.get("cat") == "kernel"
        and event.get("ph") == "X"
        and window.start_us <= float(event.get("ts", 0.0)) <= window.end_us
    ]

    target_ts: dict[str, float] = {}
    kernel_meta: dict[str, tuple[dict[str, Any], int | None, dict[str, Any] | None]] = {}
    for index, kernel in enumerate(raw_kernels):
        event_id = f"k_{index:06d}"
        args = _args(kernel)
        corr = _as_int(args.get("correlation"))
        external = _as_int(args.get("External id"))
        if external is None and corr is not None:
            external = runtime_corr_to_external.get(corr)
        cpu_op = cpu_by_external.get(external) if external is not None else None
        # Attribute kernels at CPU launch time rather than GPU execution time.
        # Async kernels can execute after Python has moved to a later module.
        runtime_event = runtime_by_corr.get(corr) if corr is not None else None
        target_ts[event_id] = float((cpu_op or runtime_event or kernel).get("ts", 0.0))
        kernel_meta[event_id] = (kernel, external, cpu_op)

    span_start = min([window.start_us, *target_ts.values()]) if target_ts else window.start_us
    span_end = max([window.end_us, *target_ts.values()]) if target_ts else window.end_us
    python_spans = _filter_python_spans(trace_events, span_start, span_end)
    python_by_id = {
        py_id: event
        for event in python_spans
        if (py_id := _python_id(event)) is not None
    }

    assigned = assign_deepest_python_spans(python_spans, target_ts)
    out: list[KernelEvent] = []
    for event_id, (kernel, external, cpu_op) in kernel_meta.items():
        args = _args(kernel)
        cpu_args = _args(cpu_op) if cpu_op else {}
        stack = build_python_chain(
            assigned.get(event_id),
            python_by_id,
            source_root=source_root,
        )
        out.append(
            KernelEvent(
                event_id=event_id,
                kernel_name=str(kernel.get("name", "")),
                ts_us=float(kernel.get("ts", 0.0)),
                dur_us=float(kernel.get("dur", 0.0)),
                stream=_kernel_stream(kernel),
                device=_kernel_device(kernel),
                correlation=_as_int(args.get("correlation")),
                external_id=external,
                cpu_op_name=str(cpu_op.get("name")) if cpu_op else None,
                cpu_input_dims=cpu_args.get("Input Dims"),
                cpu_input_types=cpu_args.get("Input type"),
                python_stack=stack,
            )
        )
    return out


def build_kernel_mappings(
    events: list[KernelEvent],
    rules: TraceMappingRules,
) -> list[KernelMapping]:
    mappings: list[KernelMapping] = []
    for event in events:
        primitive, operator, semantic, context, phase = choose_stack_frames(
            event.python_stack,
            rules.stack,
        )
        node, confidence = rules.classify_node(
            event.kernel_name,
            event.cpu_op_name,
            event.python_stack,
        )
        if node and not semantic:
            node = None
            confidence = "unmapped_no_semantic_frame"
        evidence = ["kernel"]
        if event.cpu_op_name:
            evidence.append("cpu_op")
        if event.python_stack:
            evidence.append("python_stack")
        if semantic:
            evidence.append("semantic_frame")
        if event.cpu_input_dims:
            evidence.append("record_shapes")
        mappings.append(
            KernelMapping(
                event_id=event.event_id,
                kernel_name=event.kernel_name,
                selected_node=node,
                confidence=confidence,
                primitive_frame=primitive,
                operator_frame=operator,
                semantic_frame=semantic,
                model_context_frame=context,
                phase_frame=phase,
                cpu_op_name=event.cpu_op_name,
                evidence=evidence,
            )
        )
    return mappings


def _duration_by_event(events: list[KernelEvent]) -> dict[str, float]:
    return {event.event_id: event.dur_us for event in events}


def validate_mappings(
    events: list[KernelEvent],
    mappings: list[KernelMapping],
    *,
    expected_phase: str | None = "forward_extend",
    min_mapped_duration_ratio: float = 0.70,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    durations = _duration_by_event(events)
    total_us = sum(durations.values())
    mapped_us = sum(
        durations.get(mapping.event_id, 0.0)
        for mapping in mappings
        if mapping.selected_node
    )
    stack_us = sum(
        durations.get(event.event_id, 0.0) for event in events if event.python_stack
    )

    for mapping in mappings:
        if mapping.selected_node and not mapping.semantic_frame:
            errors.append(
                f"{mapping.event_id} maps to {mapping.selected_node} without semantic_frame"
            )
        if mapping.selected_node and "python_stack" not in mapping.evidence:
            errors.append(
                f"{mapping.event_id} maps to {mapping.selected_node} without python_stack"
            )
        if (
            expected_phase
            and mapping.selected_node
            and mapping.selected_node != "decode_graph_replay"
            and not _frame_contains(mapping.phase_frame, expected_phase)
        ):
            warnings.append(
                f"{mapping.event_id} maps to {mapping.selected_node} but phase is "
                f"{mapping.phase_frame.raw if mapping.phase_frame else 'missing'}"
            )

    mapped_ratio = mapped_us / total_us if total_us else 0.0
    if mapped_ratio < min_mapped_duration_ratio:
        warnings.append(
            f"mapped duration ratio {mapped_ratio:.3f} < {min_mapped_duration_ratio:.3f}"
        )

    unmatched = [
        {
            "event_id": event.event_id,
            "kernel_name": event.kernel_name,
            "dur_us": event.dur_us,
        }
        for event, mapping in zip(events, mappings)
        if not mapping.selected_node
    ]
    unmatched.sort(key=lambda item: -float(item["dur_us"]))

    nodes: dict[str, dict[str, Any]] = {}
    for mapping in mappings:
        if not mapping.selected_node:
            continue
        cell = nodes.setdefault(mapping.selected_node, {"count": 0, "dur_us": 0.0})
        cell["count"] += 1
        cell["dur_us"] += durations.get(mapping.event_id, 0.0)

    return {
        "ok": not errors,
        "kernel_count": len(events),
        "mapped_kernel_count": sum(1 for mapping in mappings if mapping.selected_node),
        "total_kernel_us": total_us,
        "mapped_kernel_us": mapped_us,
        "stack_kernel_us": stack_us,
        "mapped_duration_ratio": mapped_ratio,
        "stack_duration_ratio": stack_us / total_us if total_us else 0.0,
        "nodes": dict(sorted(nodes.items())),
        "top_unmatched": unmatched[:20],
        "errors": errors,
        "warnings": warnings[:100],
    }


def stack_samples(mappings: list[KernelMapping], *, limit: int = 24) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    seen_nodes: set[str] = set()
    for mapping in mappings:
        node = mapping.selected_node or "UNMAPPED"
        if node in seen_nodes:
            continue
        seen_nodes.add(node)
        samples.append(
            {
                "event_id": mapping.event_id,
                "selected_node": mapping.selected_node,
                "confidence": mapping.confidence,
                "kernel_name": mapping.kernel_name,
                "cpu_op_name": mapping.cpu_op_name,
                "primitive_frame": asdict(mapping.primitive_frame)
                if mapping.primitive_frame
                else None,
                "operator_frame": asdict(mapping.operator_frame)
                if mapping.operator_frame
                else None,
                "semantic_frame": asdict(mapping.semantic_frame)
                if mapping.semantic_frame
                else None,
                "model_context_frame": asdict(mapping.model_context_frame)
                if mapping.model_context_frame
                else None,
                "phase_frame": asdict(mapping.phase_frame)
                if mapping.phase_frame
                else None,
            }
        )
        if len(samples) >= limit:
            break
    return samples


def build_trace_mapping(
    *,
    trace_path: Path,
    source_root: Path | None,
    source_repo: str,
    source_commit: str,
    config_path: Path | None,
    rank: int,
    phase: str,
    rules: TraceMappingRules,
    expect_ms: float | None = None,
    n_iters: int = 2,
    skip_first: bool = True,
    signature_kernel: str | None = None,
    expected_signature_count: int | None = None,
) -> BuildResult:
    trace = load_trace(trace_path)
    trace_events = trace.get("traceEvents") or []
    signature = signature_kernel or rules.signature_kernel
    signature_count = expected_signature_count or rules.signature_count_per_forward
    windows = find_step_annotation_windows(
        trace_events,
        phase=phase,
        signature=signature,
    )
    window_method = "gpu_step_annotation"
    if not windows:
        windows = find_forward_windows(
            trace_events,
            signature=signature,
            expected_per_forward=signature_count,
        )
        window_method = "signature_kernel_count"
    window = choose_forward_window(
        windows,
        expect_ms=expect_ms,
        n_iters=n_iters,
        skip_first=skip_first,
    )
    events = normalize_kernel_events(
        trace_events,
        window=window,
        source_root=source_root,
    )
    mappings = build_kernel_mappings(events, rules)
    validation = validate_mappings(events, mappings, expected_phase=phase)
    manifest = {
        "trace_path": str(trace_path),
        "config_path": str(config_path) if config_path else None,
        "source_root": str(source_root) if source_root else None,
        "source_repo": source_repo,
        "source_commit": source_commit,
        "rank": rank,
        "phase": phase,
        "rules_model_id": rules.model_id,
        "signature_kernel": signature,
        "signature_count_per_forward": signature_count,
        "window_selector": {
            "method": window_method,
            "expect_ms_per_iter": expect_ms,
            "n_iters": n_iters,
            "skip_first": skip_first,
        },
        "profiler": {
            "record_shapes": trace.get("record_shapes"),
            "with_stack": trace.get("with_stack"),
            "cuda_runtime_version": trace.get("cuda_runtime_version"),
            "cuda_driver_version": trace.get("cuda_driver_version"),
            "cupti_version": trace.get("cupti_version"),
        },
        "window": {
            "start_us": window.start_us,
            "end_us": window.end_us,
            "duration_ms": (window.end_us - window.start_us) / 1000.0,
            "iter_bounds_us": window.iter_bounds_us,
            "anchor_kernel_count": window.anchor_kernel_count,
        },
        "trace_event_count": len(trace_events),
        "kernel_event_count": sum(
            1 for event in trace_events if event.get("cat") == "kernel"
        ),
    }
    return BuildResult(
        manifest=manifest,
        events=events,
        mappings=mappings,
        validation=validation,
        stack_samples=stack_samples(mappings),
    )


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    raise TypeError(f"not JSON serializable: {type(obj)!r}")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(data, fh, indent=2, default=_json_default)
        fh.write("\n")


def write_jsonl(path: Path, rows: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            json.dump(row, fh, default=_json_default)
            fh.write("\n")


def write_stack_samples_markdown(path: Path, samples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Stack Samples", ""]
    for sample in samples:
        lines.append(f"## {sample['event_id']} -> {sample['selected_node']}")
        lines.append("")
        lines.append(f"- kernel: `{sample['kernel_name']}`")
        lines.append(f"- cpu_op: `{sample.get('cpu_op_name')}`")
        for key in [
            "primitive_frame",
            "operator_frame",
            "semantic_frame",
            "model_context_frame",
            "phase_frame",
        ]:
            frame = sample.get(key)
            if frame:
                lines.append(f"- {key}: `{frame['raw']}`")
        lines.append("")
    path.write_text("\n".join(lines))


def write_validation_markdown(path: Path, validation: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Validation Report",
        "",
        f"- ok: `{validation['ok']}`",
        f"- kernel_count: `{validation['kernel_count']}`",
        f"- mapped_kernel_count: `{validation['mapped_kernel_count']}`",
        f"- mapped_duration_ratio: `{validation['mapped_duration_ratio']:.4f}`",
        f"- stack_duration_ratio: `{validation['stack_duration_ratio']:.4f}`",
        "",
        "## Nodes",
        "",
    ]
    for node, cell in validation["nodes"].items():
        lines.append(f"- `{node}`: count={cell['count']}, dur_us={cell['dur_us']:.3f}")
    lines.extend(["", "## Errors", ""])
    if validation["errors"]:
        lines.extend(f"- {msg}" for msg in validation["errors"])
    else:
        lines.append("- none")
    lines.extend(["", "## Warnings", ""])
    if validation["warnings"]:
        lines.extend(f"- {msg}" for msg in validation["warnings"])
    else:
        lines.append("- none")
    lines.extend(["", "## Top Unmatched", ""])
    for item in validation["top_unmatched"]:
        lines.append(
            f"- `{item['event_id']}` {item['dur_us']:.3f} us `{item['kernel_name']}`"
        )
    path.write_text("\n".join(lines) + "\n")


def write_build_result(out_dir: Path, result: BuildResult, *, rank: int) -> None:
    write_json(out_dir / "input_manifest.json", result.manifest)
    write_jsonl(out_dir / f"events.tp{rank}.jsonl", result.events)
    write_jsonl(out_dir / f"kernel_mapping.tp{rank}.jsonl", result.mappings)
    write_json(out_dir / "validation_report.json", result.validation)
    write_validation_markdown(out_dir / "validation_report.md", result.validation)
    write_json(out_dir / "stack_samples.json", result.stack_samples)
    write_stack_samples_markdown(out_dir / "stack_samples.md", result.stack_samples)
