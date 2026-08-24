import sys
from collections import Counter
from pathlib import Path
import sqlite3

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.qwen35.profile.qwen35_nsys_mapping import (
    NsysKernel,
    NsysStep,
    TARGET_PATTERN,
    _direct_node,
    load_nsys_steps,
    load_sglang_nsys_steps,
    map_decode_step,
    map_prefill_step,
    read_nsys_export_metadata,
    sglang_graph_roles,
    sglang_nsys_trace_events,
    validate_sglang_rank_local_capture_integrity,
    validate_sglang_graph_node_stability,
)
from models.qwen35.profile.qwen35_graph_mapping import map_graph_window
from models.qwen35.profile.build_qwen35_sglang_agentx_nsys_profile import (
    parse_exact_batch_capture_observation,
    parse_exact_batch_capture_observations,
    select_balanced_exact_observations,
)
from models.qwen35.profile.build_qwen35_trt_profile import (
    _validate_exact_worker_log,
    elect_worker_comparison_sources,
    select_balanced_rank_local_steps,
)


def _kernel(name: str, index: int) -> NsysKernel:
    return NsysKernel(
        start_ns=index * 1000,
        end_ns=index * 1000 + 500,
        name=name,
        stream=7,
        correlation_id=0,
        graph_id=1,
        graph_node_id=index,
    )


def test_sglang_nsys_export_metadata_is_auditable_and_narrow(tmp_path):
    path = tmp_path / "report.sqlite"
    connection = sqlite3.connect(path)
    connection.execute("create table META_DATA_EXPORT(name text, value text)")
    connection.executemany(
        "insert into META_DATA_EXPORT values (?, ?)",
        [
            ("EXPORT_PRODUCT_NAME", "NVIDIA Nsight Systems"),
            ("EXPORT_PRODUCT_VERSION", "2026.4.1.191"),
            ("EXPORT_SCHEMA_VERSION", "3.28.0"),
            ("SENSITIVE_OR_UNRELATED", "must not be returned"),
        ],
    )
    connection.commit()
    connection.close()

    assert read_nsys_export_metadata(path) == {
        "product": "NVIDIA Nsight Systems",
        "version": "2026.4.1.191",
        "schema_version": "3.28.0",
    }


def test_sglang_nsys_rejects_graph_metadata_without_cuda_activity(tmp_path):
    path = tmp_path / "metadata-only.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        create table StringIds(id integer primary key, value text);
        create table NVTX_EVENTS(
          start integer, end integer, text text, textId integer, globalTid integer
        );
        create table CUDA_GRAPH_NODE_EVENTS(start integer, end integer);
        """
    )
    connection.commit()
    connection.close()

    with pytest.raises(ValueError, match="lacks CUPTI_ACTIVITY_KIND_KERNEL"):
        load_sglang_nsys_steps(path, rank=0)


def _moe(kernels: list[NsysKernel], call: int, *, draft: bool) -> None:
    suffix = "draft" if draft else "target"
    for name in (
        f"customMoeRoutingKernel_{suffix}",
        "moeA2APrepareDispatchKernel",
        "moeA2ADispatchKernel",
        f"routed_expert_{suffix}",
        "moeA2APrepareCombineKernel",
        "moeA2ACombineKernel",
    ):
        kernels.append(_kernel(name, len(kernels)))


def _synthetic_step() -> NsysStep:
    kernels: list[NsysKernel] = []
    for layer_id, kind in enumerate(TARGET_PATTERN):
        anchor = "_causal_conv1d_update_kernel" if kind == "gdn" else "fmhaSm100_target"
        kernels.append(_kernel(anchor, len(kernels)))
        _moe(kernels, layer_id, draft=False)
    kernels.extend(
        _kernel("_promote_mamba_state_kernel", len(kernels) + i) for i in range(45)
    )
    for draft_pass in range(6):
        kernels.append(_kernel(f"fmhaSm100_draft_{draft_pass}", len(kernels)))
        _moe(kernels, 60 + draft_pass, draft=True)
    return NsysStep(
        step_id=60000,
        rank=0,
        label="[Executor] _forward_step 60000: 0 ctx reqs, 0 ctx tokens, 32 gen reqs",
        cpu_start_ns=0,
        cpu_end_ns=1_000_000,
        context_reqs=0,
        context_tokens=0,
        generation_reqs=32,
        kernels=tuple(kernels),
        graph_launch_count=1,
    )


def test_trt_decode_mapping_keeps_target_and_six_mtp_collectives_separate():
    mapped, validation = map_decode_step(_synthetic_step())
    assert validation["target_gdn_layers"] == 45
    assert validation["target_attention_layers"] == 15
    assert validation["target_ep4_dispatch"] == 60
    assert validation["draft_ep4_dispatch"] == 6
    assert validation["mtp_passes"] == 6
    assert validation["timing_closure_us"] == 0
    assert validation["attributed_duration_ratio"] < 1.0
    assert validation["timeline_interval_coverage_ratio"] == 1.0
    unresolved = [event for event in mapped if event["mapping_status"] == "unmapped"]
    assert unresolved
    assert all(event["node"] is None for event in unresolved)
    assert all(
        event["candidate_nodes"] and event["unmapped_reason"] for event in unresolved
    )
    assert (
        sum(event["node"] == "moe_block.target_ep4_dispatch" for event in mapped) == 60
    )
    assert (
        sum(event["node"] == "mtp_moe_block.draft_ep4_dispatch" for event in mapped)
        == 6
    )


def test_trt_decode_mapping_rejects_missing_mtp_pass():
    step = _synthetic_step()
    truncated = NsysStep(**{**step.__dict__, "kernels": step.kernels[:-7]})
    try:
        map_decode_step(truncated)
    except ValueError as error:
        assert "66 MoE calls" in str(error) or "six MTP" in str(error)
    else:
        raise AssertionError("missing MTP pass was accepted")


def test_trt_signature_slots_are_narrow_and_residual_fusion_has_two_leaves():
    assert (
        _direct_node(
            "nvjet_sm103_qqtst_unknown_shape",
            section="target",
            layer_kind="gdn",
        )
        is None
    )
    qkvz = _direct_node(
        "nvjet_sm103_qqtst_144x128_128x8_2x2f_2cta_h_bz_TNN",
        section="target",
        layer_kind="gdn",
    )
    assert qkvz is not None
    assert qkvz.node == "gdn_attention.qkvz_projection"
    assert qkvz.attribution_method == "validated_graph_signature_slot"

    residual = _direct_node(
        "kernel_flashinfernorm_fused_add_rmsnorm_kernel",
        section="target",
        layer_kind="gdn",
        before_moe=True,
    )
    assert residual is not None
    assert residual.status == "fusion"
    assert residual.node == "gdn_moe_block.attention_residual"
    assert residual.ir_targets == ("gdn_moe_block.post_attention_norm",)


def test_trt_prefill_mapping_distinguishes_owner_compute_and_collective_only_rank():
    owner_kernels: list[NsysKernel] = []
    collective_kernels: list[NsysKernel] = []
    for layer_id, kind in enumerate(TARGET_PATTERN):
        owner_kernels.append(
            _kernel(
                "_causal_conv1d_update_kernel" if kind == "gdn" else "fmhaSm100_target",
                len(owner_kernels),
            )
        )
        _moe(owner_kernels, layer_id, draft=False)
        _moe(collective_kernels, layer_id, draft=False)
    common = {
        "step_id": 10000,
        "rank": 0,
        "label": "[Executor] _forward_step 10000: 1 ctx reqs, 8192 ctx tokens, 0 gen reqs",
        "cpu_start_ns": 0,
        "cpu_end_ns": 1_000_000,
        "context_reqs": 1,
        "context_tokens": 8192,
        "generation_reqs": 0,
        "graph_launch_count": 0,
    }
    _mapped, owner = map_prefill_step(NsysStep(**common, kernels=tuple(owner_kernels)))
    _mapped, collective = map_prefill_step(
        NsysStep(**common, kernels=tuple(collective_kernels))
    )
    assert owner["owner_compute"] is True
    assert owner["target_gdn_layers"] == 45
    assert owner["target_attention_layers"] == 15
    assert collective["owner_compute"] is False
    assert collective["target_gdn_layers"] == 0
    assert collective["target_ep4_dispatch"] == 60


def test_trt_prefill_mapping_recognizes_sm103_and_context_causal_conv_signatures():
    kernels: list[NsysKernel] = []
    for layer_id, kind in enumerate(TARGET_PATTERN):
        kernels.append(
            _kernel(
                (
                    "causal_conv1d_fwd_kernel<128>"
                    if kind == "gdn"
                    else "fmhaSm103aKernel_Context"
                ),
                len(kernels),
            )
        )
        _moe(kernels, layer_id, draft=False)
    step = NsysStep(
        step_id=10000,
        rank=0,
        label="[Executor] _forward_step 10000: 1 ctx reqs, 8192 ctx tokens, 0 gen reqs",
        cpu_start_ns=0,
        cpu_end_ns=1_000_000,
        context_reqs=1,
        context_tokens=8192,
        generation_reqs=0,
        kernels=tuple(kernels),
        graph_launch_count=61,
    )
    mapped, validation = map_prefill_step(step)
    assert validation["owner_compute"] is True
    assert validation["target_gdn_layers"] == 45
    assert validation["target_attention_layers"] == 15
    assert sum(event["node"] == "gdn_attention.causal_conv" for event in mapped) == 45
    assert sum(event["node"] == "full_attention.causal_gqa" for event in mapped) == 15


def test_trt_prefill_maps_repeated_gemm_only_with_complete_layer_slot_proof():
    projection = "nvjet_sm103_qqtst_128x256_128x6_2x1_2cta_v_bz_TNT"
    kernels: list[NsysKernel] = []
    for layer_id, kind in enumerate(TARGET_PATTERN):
        kernels.append(_kernel(projection, len(kernels)))
        kernels.append(
            _kernel(
                (
                    "causal_conv1d_fwd_kernel<128>"
                    if kind == "gdn"
                    else "fmhaSm103aKernel_Context"
                ),
                len(kernels),
            )
        )
        kernels.append(_kernel(projection, len(kernels)))
        kernels.append(_kernel("fused_add_rmsnorm_attention", len(kernels)))
        _moe(kernels, layer_id, draft=False)
        kernels.append(_kernel("silu_and_mul_kernel", len(kernels)))
        kernels.append(_kernel(projection, len(kernels)))
        kernels.append(_kernel("sigmoid_gate_mul_add_kernel", len(kernels)))
        kernels.append(_kernel("fused_add_rmsnorm_layer", len(kernels)))
    step = NsysStep(
        step_id=10000,
        rank=0,
        label="[Executor] _forward_step 10000: 1 ctx reqs, 8192 ctx tokens, 0 gen reqs",
        cpu_start_ns=0,
        cpu_end_ns=1_000_000,
        context_reqs=1,
        context_tokens=8192,
        generation_reqs=0,
        kernels=tuple(kernels),
        graph_launch_count=61,
    )
    mapped, validation = map_prefill_step(step)
    slot_events = [
        event
        for event in mapped
        if event["attribution_method"] == "validated_prefill_layer_sequence"
    ]
    assert validation["validated_prefill_projection_slots"] == 180
    assert (
        sum(event["node"] == "gdn_attention.qkvz_projection" for event in slot_events)
        == 45
    )
    assert (
        sum(event["node"] == "full_attention.qkv_projection" for event in slot_events)
        == 15
    )
    assert (
        sum(event["node"] == "gdn_attention.output_projection" for event in slot_events)
        == 45
    )
    assert (
        sum(
            event["node"] == "full_attention.output_projection" for event in slot_events
        )
        == 15
    )
    assert (
        sum(event["node"] == "moe_block.shared_expert" for event in slot_events) == 60
    )
    input_slots = [
        event
        for event in slot_events
        if event["node"]
        in {"gdn_attention.qkvz_projection", "full_attention.qkv_projection"}
    ]
    assert [event["layer_id"] for event in input_slots] == list(range(60))
    assert all(
        event["layer_kind"] == TARGET_PATTERN[event["layer_id"]]
        for event in slot_events
    )


def test_nsys_parser_splits_overlapping_graph_executions_by_node_occurrence(tmp_path):
    path = tmp_path / "worker-decode-rank0.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        create table StringIds(id integer primary key, value text);
        create table NVTX_EVENTS(start integer, end integer, text text, textId integer);
        create table CUPTI_ACTIVITY_KIND_RUNTIME(
          start integer, end integer, correlationId integer, nameId integer
        );
        create table CUPTI_ACTIVITY_KIND_KERNEL(
          start integer, end integer, streamId integer, correlationId integer,
          graphId integer, graphNodeId integer, demangledName integer
        );
        """
    )
    connection.executemany(
        "insert into StringIds values (?,?)",
        [(1, "cudaGraphLaunch_v10000"), (2, "node_a"), (3, "node_b"), (4, "direct")],
    )
    connection.executemany(
        "insert into NVTX_EVENTS values (?,?,?,null)",
        [
            (
                0,
                200,
                "[Executor] _forward_step 10: 0 ctx reqs, 0 ctx tokens, 1 gen reqs",
            ),
            (
                200,
                400,
                "[Executor] _forward_step 11: 0 ctx reqs, 0 ctx tokens, 1 gen reqs",
            ),
        ],
    )
    connection.executemany(
        "insert into CUPTI_ACTIVITY_KIND_RUNTIME values (?,?,?,?)",
        [(100, 110, 10, 1), (250, 260, 20, 1), (120, 121, 30, 4)],
    )
    connection.executemany(
        "insert into CUPTI_ACTIVITY_KIND_KERNEL values (?,?,?,?,?,?,?)",
        [
            (150, 160, 1, 0, 7, 100, 2),
            (250, 260, 1, 0, 7, 100, 2),
            (350, 360, 1, 0, 7, 101, 3),
            (450, 460, 1, 0, 7, 101, 3),
            (122, 124, 2, 30, None, None, 4),
        ],
    )
    connection.commit()
    connection.close()

    steps = load_nsys_steps(path)
    assert [step.step_id for step in steps] == [10, 11]
    assert [kernel.name for kernel in steps[0].kernels] == [
        "direct",
        "node_a",
        "node_b",
    ]
    assert [kernel.name for kernel in steps[1].kernels] == ["node_a", "node_b"]


def test_nsys_parser_supports_repeated_multi_graph_prefill_sequence(tmp_path):
    path = tmp_path / "worker-prefill-rank0.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        create table StringIds(id integer primary key, value text);
        create table NVTX_EVENTS(start integer, end integer, text text, textId integer);
        create table CUPTI_ACTIVITY_KIND_RUNTIME(
          start integer, end integer, correlationId integer, nameId integer
        );
        create table CUPTI_ACTIVITY_KIND_KERNEL(
          start integer, end integer, streamId integer, correlationId integer,
          graphId integer, graphNodeId integer, demangledName integer
        );
        """
    )
    connection.executemany(
        "insert into StringIds values (?,?)",
        [(1, "cudaGraphLaunch_v10000"), (2, "graph_a"), (3, "graph_b")],
    )
    connection.executemany(
        "insert into NVTX_EVENTS values (?,?,?,null)",
        [
            (
                0,
                200,
                "[Executor] _forward_step 10: 1 ctx reqs, 8 ctx tokens, 0 gen reqs",
            ),
            (
                200,
                400,
                "[Executor] _forward_step 11: 1 ctx reqs, 8 ctx tokens, 0 gen reqs",
            ),
        ],
    )
    connection.executemany(
        "insert into CUPTI_ACTIVITY_KIND_RUNTIME values (?,?,?,?)",
        [(10, 20, 10, 1), (30, 40, 20, 1), (210, 220, 30, 1), (230, 240, 40, 1)],
    )
    connection.executemany(
        "insert into CUPTI_ACTIVITY_KIND_KERNEL values (?,?,?,?,?,?,?)",
        [
            (100, 110, 1, 0, 7, 700, 2),
            (300, 310, 1, 0, 7, 700, 2),
            (120, 130, 1, 0, 8, 800, 3),
            (320, 330, 1, 0, 8, 800, 3),
        ],
    )
    connection.commit()
    connection.close()

    steps = load_nsys_steps(path)
    assert [step.graph_launch_count for step in steps] == [2, 2]
    assert [kernel.name for kernel in steps[0].kernels] == ["graph_a", "graph_b"]
    assert [kernel.name for kernel in steps[1].kernels] == ["graph_a", "graph_b"]


def test_sglang_nsys_uses_scheduler_wall_and_proves_three_graph_roles(tmp_path):
    path = tmp_path / "sglang-worker.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        create table StringIds(id integer primary key, value text);
        create table NVTX_EVENTS(
          start integer, end integer, text text, textId integer, globalTid integer
        );
        create table CUPTI_ACTIVITY_KIND_RUNTIME(
          start integer, end integer, correlationId integer, nameId integer,
          globalTid integer
        );
        create table CUPTI_ACTIVITY_KIND_KERNEL(
          start integer, end integer, deviceId integer, contextId integer,
          streamId integer, correlationId integer, globalPid integer,
          graphId integer, graphNodeId integer, demangledName integer
        );
        """
    )
    names = {
        1: "cudaGraphLaunch_v10000",
        2: "fused_qkvzba_split",
        3: "_fused_qk_rmsnorm_rope_gate_kernel",
    }
    connection.executemany("insert into StringIds values (?,?)", names.items())
    process = 100 << 24
    global_tid = process + 17
    connection.executemany(
        "insert into NVTX_EVENTS values (?,?,?,null,?)",
        [
            (0, 900_000, "scheduler.run_batch", global_tid),
            (1_000_000, 1_900_000, "scheduler.run_batch", global_tid),
            (2_000_000, 2_900_000, "scheduler.run_batch", global_tid),
        ],
    )
    runtime_rows = []
    kernel_rows = []
    for step in range(2):
        base = step * 1_000_000 + 100_000
        runtime_rows.extend(
            (base + offset, base + offset + 100, 10 + offset, 1, global_tid)
            for offset in (0, 300_000, 600_000)
        )
        node = 0
        for layer_id, kind in enumerate(TARGET_PATTERN):
            name_id = 2 if kind == "gdn" else 3
            start = base + layer_id * 2_000
            kernel_rows.append(
                (start, start + 500, 0, 1, 7, 0, process, 2, node, name_id)
            )
            node += 1
        for round_id in range(4):
            start = base + 300_000 + round_id * 2_000
            kernel_rows.append(
                (start, start + 500, 0, 1, 7, 0, process, 23, round_id, 3)
            )
        start = base + 600_000
        kernel_rows.append((start, start + 500, 0, 1, 7, 0, process, 44, 0, 3))
    # Nsight can stop after the next logical step's three CPU launches but
    # while only the first node of one GPU graph has landed. The parser must
    # trim this capture edge, not shift that partial replay into step 1.
    runtime_rows.extend(
        (2_100_000 + offset, 2_100_100 + offset, 100 + offset, 1, global_tid)
        for offset in (0, 300_000, 600_000)
    )
    kernel_rows.append((2_400_000, 2_400_500, 0, 1, 7, 0, process, 23, 0, 3))
    connection.executemany(
        "insert into CUPTI_ACTIVITY_KIND_RUNTIME values (?,?,?,?,?)", runtime_rows
    )
    connection.executemany(
        "insert into CUPTI_ACTIVITY_KIND_KERNEL values (?,?,?,?,?,?,?,?,?,?)",
        kernel_rows,
    )
    connection.commit()
    connection.close()

    steps, evidence = load_sglang_nsys_steps(path, rank=0)
    assert len(steps) == 2
    assert evidence["marker_source"] == "scheduler.run_batch"
    assert evidence["unpaired_launch_count"] == 3
    assert evidence["leading_unpaired_launch_count"] == 0
    assert evidence["trailing_unpaired_launch_count"] == 3
    assert evidence["boundary_trim_by_graph"][23] == "drop_trailing"
    assert all(step.cpu_wall_us == 1000.0 for step in steps)
    assert all(step.graph_launch_count == 3 for step in steps)
    assert sglang_graph_roles(steps[0].kernels) == {
        "target_verify": 2,
        "draft": 23,
        "draft_extend": 44,
    }
    stability = validate_sglang_graph_node_stability(steps)
    assert stability["step_count"] == 2
    assert stability["role_counts"] == {
        "target_verify": 2,
        "draft": 2,
        "draft_extend": 2,
    }

    trace_events, window, _roles = sglang_nsys_trace_events(steps[0], batch_size=32)
    mapped, validation = map_graph_window(
        trace_events, window=window, rank=0, step_index=0
    )
    assert validation["target_verify_batch_size"] == 32
    assert validation["signature_counts"]["target_gdn_layers"] == 45
    assert validation["signature_counts"]["target_attention_layers"] == 15
    assert validation["signature_counts"]["mtp_draft_rounds"] == 5
    graph_events = [event for event in mapped if event["graph_id"] is not None]
    assert len(graph_events) == 65
    assert all(event["graph_node_id"] is not None for event in graph_events)
    assert {event["graph_role"] for event in graph_events} == {
        "target_verify",
        "draft",
        "draft_extend",
    }


def test_sglang_nsys_recovers_first_exact_step_from_capture_range(tmp_path):
    path = tmp_path / "sglang-exact-capture.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        create table StringIds(id integer primary key, value text);
        create table NVTX_EVENTS(
          start integer, end integer, text text, textId integer, globalTid integer
        );
        create table CUPTI_ACTIVITY_KIND_RUNTIME(
          start integer, end integer, correlationId integer, nameId integer,
          globalTid integer
        );
        create table CUPTI_ACTIVITY_KIND_KERNEL(
          start integer, end integer, deviceId integer, contextId integer,
          streamId integer, correlationId integer, globalPid integer,
          graphId integer, graphNodeId integer, demangledName integer
        );
        """
    )
    names = {
        1: "cudaGraphLaunch_v10000",
        2: "fused_qkvzba_split",
        3: "_fused_qk_rmsnorm_rope_gate_kernel",
    }
    connection.executemany("insert into StringIds values (?,?)", names.items())
    process = 101 << 24
    global_tid = process + 19
    connection.executemany(
        "insert into NVTX_EVENTS values (?,?,?,null,?)",
        [
            (50_000, 2_950_000, "agentx_decode_capture", global_tid),
            # Capture starts inside the exact step's outer run_batch range, so
            # only the following scheduler ranges are present in the report.
            (1_000_000, 1_900_000, "scheduler.run_batch", global_tid),
            (2_000_000, 2_900_000, "scheduler.run_batch", global_tid),
        ],
    )
    runtime_rows = []
    kernel_rows = []
    for step in range(2):
        base = step * 1_000_000 + 100_000
        runtime_rows.extend(
            (base + offset, base + offset + 100, 10 + offset, 1, global_tid)
            for offset in (0, 300_000, 600_000)
        )
        for layer_id, kind in enumerate(TARGET_PATTERN):
            name_id = 2 if kind == "gdn" else 3
            start = base + layer_id * 2_000
            kernel_rows.append(
                (start, start + 500, 0, 1, 7, 0, process, 2, layer_id, name_id)
            )
        for round_id in range(4):
            start = base + 300_000 + round_id * 2_000
            kernel_rows.append(
                (start, start + 500, 0, 1, 7, 0, process, 23, round_id, 3)
            )
        start = base + 600_000
        kernel_rows.append((start, start + 500, 0, 1, 7, 0, process, 44, 0, 3))
    connection.executemany(
        "insert into CUPTI_ACTIVITY_KIND_RUNTIME values (?,?,?,?,?)", runtime_rows
    )
    connection.executemany(
        "insert into CUPTI_ACTIVITY_KIND_KERNEL values (?,?,?,?,?,?,?,?,?,?)",
        kernel_rows,
    )
    connection.commit()
    connection.close()

    steps, evidence = load_sglang_nsys_steps(
        path, rank=0, capture_range_label="agentx_decode_capture"
    )
    assert len(steps) == 2
    assert steps[0].label == "agentx_decode_capture:first_exact_step"
    assert steps[0].cpu_start_ns == 50_000
    assert steps[0].cpu_end_ns == 1_000_000
    assert steps[0].graph_launch_count == 3
    assert evidence["marker_source"] == "agentx_decode_capture"
    assert evidence["scheduler_marker_count"] == 2
    assert evidence["first_step_boundary"] == (
        "capture-range start to next scheduler.run_batch start"
    )
    assert sglang_graph_roles(steps[0].kernels) == {
        "target_verify": 2,
        "draft": 23,
        "draft_extend": 44,
    }


def test_sglang_one_step_pulse_uses_its_capture_range_as_the_exact_boundary(tmp_path):
    path = tmp_path / "one-step-pulse.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        create table StringIds(id integer primary key, value text);
        create table NVTX_EVENTS(
          start integer, end integer, text text, textId integer, globalTid integer
        );
        create table CUPTI_ACTIVITY_KIND_RUNTIME(
          start integer, end integer, correlationId integer, nameId integer,
          globalTid integer
        );
        create table CUPTI_ACTIVITY_KIND_KERNEL(
          start integer, end integer, deviceId integer, contextId integer,
          streamId integer, correlationId integer, globalPid integer,
          graphId integer, graphNodeId integer, demangledName integer
        );
        """
    )
    names = {
        1: "cudaGraphLaunch_v10000",
        2: "fused_qkvzba_split",
        3: "_fused_qk_rmsnorm_rope_gate_kernel",
    }
    connection.executemany("insert into StringIds values (?,?)", names.items())
    process = 101 << 24
    global_tid = process + 19
    connection.execute(
        "insert into NVTX_EVENTS values (50000,950000,'agentx_decode_capture',null,?)",
        (global_tid,),
    )
    runtime_rows = [
        (100_000 + offset, 100_100 + offset, 10 + offset, 1, global_tid)
        for offset in (0, 300_000, 600_000)
    ]
    kernel_rows = []
    for layer_id, kind in enumerate(TARGET_PATTERN):
        name_id = 2 if kind == "gdn" else 3
        start = 100_000 + layer_id * 2_000
        kernel_rows.append(
            (start, start + 500, 0, 1, 7, 0, process, 2, layer_id, name_id)
        )
    for round_id in range(4):
        start = 400_000 + round_id * 2_000
        kernel_rows.append(
            (start, start + 500, 0, 1, 7, 0, process, 23, round_id, 3)
        )
    kernel_rows.append((700_000, 700_500, 0, 1, 7, 0, process, 44, 0, 3))
    connection.executemany(
        "insert into CUPTI_ACTIVITY_KIND_RUNTIME values (?,?,?,?,?)", runtime_rows
    )
    connection.executemany(
        "insert into CUPTI_ACTIVITY_KIND_KERNEL values (?,?,?,?,?,?,?,?,?,?)",
        kernel_rows,
    )
    connection.commit()
    connection.close()

    integrity = validate_sglang_rank_local_capture_integrity(
        path,
        capture_range_label="agentx_decode_capture",
        rank=0,
        expected_capture_count=1,
    )
    assert integrity["capture_range_count"] == 1
    assert integrity["ranks"]["r0"]["cuda_graph_launch_count"] == 3

    steps, evidence = load_sglang_nsys_steps(
        path, rank=0, capture_range_label="agentx_decode_capture"
    )
    assert len(steps) == 1
    assert steps[0].cpu_start_ns == 50_000
    assert steps[0].cpu_end_ns == 950_000
    assert steps[0].graph_launch_count == 3
    assert evidence["scheduler_marker_count"] == 0
    assert evidence["capture_range_count"] == 1


def test_sglang_one_step_pulse_rejects_incomplete_graph_capture(tmp_path):
    path = tmp_path / "idle-truncated.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        create table StringIds(id integer primary key, value text);
        create table NVTX_EVENTS(
          start integer, end integer, text text, textId integer, globalTid integer
        );
        create table CUPTI_ACTIVITY_KIND_KERNEL(
          start integer, end integer, deviceId integer, contextId integer,
          streamId integer, correlationId integer, globalPid integer,
          graphId integer, graphNodeId integer, demangledName integer
        );
        create table CUPTI_ACTIVITY_KIND_RUNTIME(
          start integer, end integer, correlationId integer, nameId integer,
          globalTid integer
        );
        """
    )
    process = 101 << 24
    connection.execute("insert into StringIds values (1,'direct_kernel')")
    connection.execute(
        "insert into NVTX_EVENTS values (100,200,'agentx_decode_capture',null,?)",
        (process + 7,),
    )
    connection.execute(
        "insert into CUPTI_ACTIVITY_KIND_KERNEL values " "(110,120,0,1,7,0,?,1,1,1)",
        (process,),
    )
    connection.commit()
    connection.close()

    with pytest.raises(
        ValueError,
        match="no complete target/draft graph execution",
    ):
        load_sglang_nsys_steps(
            path, rank=0, capture_range_label="agentx_decode_capture"
        )


def test_sglang_exact_batch_log_selects_gate_step_after_delayed_previous_row(tmp_path):
    worker_log = tmp_path / "node_decode_w0.out"
    worker_log.write_text(
        "[x DP0 TP0 EP0] Exact running-batch Nsight gate matched: "
        "batch=32 forward_ct=30123\n"
        "[x DP0 TP0 EP0] Profiling starts. Traces will be saved to: /tmp\n"
        "[x DP1 TP1 EP1] Decode batch [77], #running-req: 41, #full token: 99, "
        "accept len: 4.8, #retracted-req: 0, cuda graph: True, #queue-req: 0\n"
        "[x DP0 TP0 EP0] Decode batch [30122], #running-req: 31, #full token: 6300, "
        "accept len: 4.75, #retracted-req: 0, cuda graph: True, #queue-req: 0\n"
        "[x DP1 TP1 EP1] Decode batch [30123], #running-req: 29, #full token: 6200, "
        "accept len: 4.75, #retracted-req: 0, cuda graph: True, #queue-req: 0\n"
        "[x DP2 TP2 EP2] Decode batch [30123], #running-req: 34, #full token: 6500, "
        "accept len: 4.75, #retracted-req: 0, cuda graph: True, #queue-req: 0\n"
        "[x DP3 TP3 EP3] Decode batch [30123], #running-req: 33, #full token: 6450, "
        "accept len: 4.75, #retracted-req: 0, cuda graph: True, #queue-req: 0\n"
        "[x DP0 TP0 EP0] Decode batch [30123], #running-req: 32, #full token: 6400, "
        "accept len: 4.75, #retracted-req: 0, cuda graph: True, #queue-req: 0\n"
        "[x DP0 TP0 EP0] Decode batch [30124], #running-req: 35, #full token: 7000, "
        "accept len: 4.80, #retracted-req: 0, cuda graph: True, #queue-req: 0\n"
        "[x DP0 TP0 EP0] Stop profiling...\n"
        "[x DP0 TP0 EP0] Profiling done. Traces are saved to: /tmp\n"
    )
    observation = parse_exact_batch_capture_observation(worker_log, selected_batch=32)
    assert observation["gate_forward_ct"] == 30123
    assert observation["scheduler_step"] == 30123
    assert observation["running_requests"] == 32
    assert observation["logged_rows_before_exact"] == 1
    assert observation["capture_dp0_observation_count"] == 3
    assert {
        rank: row["running_requests"]
        for rank, row in observation["rank_local_batches_at_exact_step"].items()
    } == {"r0": 32, "r1": 29, "r2": 34, "r3": 33}
    assert observation["profiler_completed"] is True


def _write_all_dp_exact_capture(
    path: Path, *, steps: int = 32, invalid_rank=None, queued_requests: int = 0
) -> None:
    lines = []
    for rank in range(4):
        lines.append(
            f"[x DP{rank} TP{rank} EP{rank}] Exact-batch Nsight sync group ready: "
            "world_size=4"
        )
        lines.append(
            f"[x DP{rank} TP{rank} EP{rank}] All-DP exact running-batch Nsight "
            f"gate matched: batch=32 forward_ct=4000 warmup_batches=16"
        )
        lines.append(
            f"[x DP{rank} TP{rank} EP{rank}] Profiling starts. "
            "Traces will be saved to: /tmp"
        )
        rank_steps = steps - 1 if rank == invalid_rank else steps
        for offset in range(rank_steps):
            lines.append(
                f"[x DP{rank} TP{rank} EP{rank}] Decode batch [{3999 + offset}], "
                "#running-req: 32, #full token: 6400, accept len: 4.80, "
                "#retracted-req: 0, cuda graph: True, "
                f"#queue-req: {queued_requests}"
            )
        lines.append(f"[x DP{rank} TP{rank} EP{rank}] Stop profiling...")
        lines.append(
            f"[x DP{rank} TP{rank} EP{rank}] Profiling done. "
            "Traces are saved to: /tmp"
        )
    path.write_text("\n".join(lines) + "\n")


def test_sglang_exact_capture_requires_32_contiguous_bs32_steps_on_all_dp_ranks(
    tmp_path,
):
    worker_log = tmp_path / "node_decode_w0.out"
    _write_all_dp_exact_capture(worker_log)

    observations = parse_exact_batch_capture_observations(
        worker_log, selected_batch=32, expected_steps=32
    )

    assert set(observations) == {0, 1, 2, 3}
    assert all(row["gate_forward_ct"] == 4000 for row in observations.values())
    assert all(row["sync_world_size"] == 4 for row in observations.values())
    assert all(row["local_warmup_batches"] == 16 for row in observations.values())
    assert all(row["capture_observation_count"] == 32 for row in observations.values())
    assert all(
        [item["scheduler_step"] for item in row["observations"]]
        == list(range(3999, 4031))
        for row in observations.values()
    )


def test_sglang_exact_capture_rejects_incomplete_rank(tmp_path):
    worker_log = tmp_path / "node_decode_w0.out"
    _write_all_dp_exact_capture(worker_log, invalid_rank=2)

    with pytest.raises(ValueError, match="DP2 expected 32 captured decode rows"):
        parse_exact_batch_capture_observations(
            worker_log, selected_batch=32, expected_steps=32
        )


def test_sglang_exact_capture_allows_recorded_saturation_queue(tmp_path):
    worker_log = tmp_path / "node_decode_w0.out"
    _write_all_dp_exact_capture(worker_log, queued_requests=37)

    observations = parse_exact_batch_capture_observations(
        worker_log, selected_batch=32, expected_steps=32
    )

    assert all(
        {row["queued_requests"] for row in rank["observations"]} == {37}
        for rank in observations.values()
    )


def test_sglang_exact_capture_rejects_missing_worker_wide_sync_proof(tmp_path):
    worker_log = tmp_path / "node_decode_w0.out"
    _write_all_dp_exact_capture(worker_log)
    worker_log.write_text(
        worker_log.read_text().replace(
            "[x DP3 TP3 EP3] Exact-batch Nsight sync group ready: world_size=4\n",
            "",
        )
    )

    with pytest.raises(ValueError, match="incomplete all-DP sync-group proof"):
        parse_exact_batch_capture_observations(
            worker_log, selected_batch=32, expected_steps=32
        )


def test_sglang_exact_capture_rejects_short_exact_batch_warmup(tmp_path):
    worker_log = tmp_path / "node_decode_w0.out"
    _write_all_dp_exact_capture(worker_log)
    worker_log.write_text(
        worker_log.read_text().replace("warmup_batches=16", "warmup_batches=0")
    )

    with pytest.raises(ValueError, match="exact gate lacks the required rank"):
        parse_exact_batch_capture_observations(
            worker_log,
            selected_batch=32,
            expected_steps=32,
            expected_warmup_batches=1,
        )


def _write_any_rank_variable_capture(
    path: Path,
    *,
    steps: int = 64,
    exact_per_rank: tuple[int, ...] = (8, 12, 16, 20),
    gate_reduction: str = "any",
    gate_rank: int = 2,
) -> None:
    lines = []
    for rank in range(4):
        lines.append(
            f"[x DP{rank} TP{rank} EP{rank}] Exact-batch Nsight sync group ready: "
            "world_size=4"
        )
        lines.append(
            f"[x DP{rank} TP{rank} EP{rank}] Worker-wide exact running-batch "
            f"Nsight gate matched: reduction={gate_reduction} batch=32 "
            f"forward_ct=5000 local_warmup_batches={int(rank == gate_rank)}"
        )
        lines.append(
            f"[x DP{rank} TP{rank} EP{rank}] Profiling starts. "
            "Traces will be saved to: /tmp"
        )
        exact_offsets = {
            ((2 * index + 1) * steps) // (2 * exact_per_rank[rank])
            for index in range(exact_per_rank[rank])
        }
        for offset in range(steps):
            batch = 32 if offset in exact_offsets else 28 + ((offset + rank) % 4)
            lines.append(
                f"[x DP{rank} TP{rank} EP{rank}] Decode batch [{4999 + offset}], "
                f"#running-req: {batch}, #full token: 6400, accept len: 4.80, "
                "#retracted-req: 0, cuda graph: True, #queue-req: 37"
            )
        lines.append(f"[x DP{rank} TP{rank} EP{rank}] Stop profiling...")
        lines.append(
            f"[x DP{rank} TP{rank} EP{rank}] Profiling done. "
            "Traces are saved to: /tmp"
        )
    path.write_text("\n".join(lines) + "\n")


def _write_auto_rank_capture(
    path: Path,
    *,
    selected_rank: int,
    steps: int = 64,
    exact_offsets: tuple[int, ...] = (0, 1, 2, 3, 4, 5),
) -> None:
    lines = []
    for rank in range(4):
        lines.append(
            f"[x DP{rank} TP{rank} EP{rank}] Exact-batch Nsight sync group ready: "
            "world_size=4"
        )
        lines.append(
            f"[x DP{rank} TP{rank} EP{rank}] Worker-wide exact running-batch "
            "Nsight gate matched: reduction=auto batch=32 forward_ct=5000 "
            f"local_warmup_batches={int(rank == selected_rank)} "
            f"selected_rank={selected_rank}"
        )
        lines.append(
            f"[x DP{rank} TP{rank} EP{rank}] Profiling starts. "
            "Traces will be saved to: /tmp"
        )
        if rank == selected_rank:
            lines.append(
                f"[x DP{rank} TP{rank} EP{rank}] Started Nsight Systems NVTX "
                "capture range: agentx_decode_capture"
            )
        for offset in range(steps):
            pre_batch = 32 if offset in exact_offsets else 28 + (offset % 4)
            if rank == selected_rank:
                lines.append(
                    f"[x DP{rank} TP{rank} EP{rank}] Exact-batch Nsight capture "
                    f"observation: selected_rank={rank} batch={pre_batch} "
                    f"forward_ct={5000 + offset} capture_index={offset}"
                )
            post_batch = pre_batch - 1 if pre_batch == 32 else pre_batch
            lines.append(
                f"[x DP{rank} TP{rank} EP{rank}] Decode batch [{4999 + offset}], "
                f"#running-req: {post_batch}, #full token: 6400, accept len: 4.80, "
                "#retracted-req: 0, cuda graph: True, #queue-req: 37"
            )
        lines.append(f"[x DP{rank} TP{rank} EP{rank}] Stop profiling...")
        lines.append(
            f"[x DP{rank} TP{rank} EP{rank}] Profiling done. "
            "Traces are saved to: /tmp"
        )
    path.write_text("\n".join(lines) + "\n")


def test_sglang_auto_gate_uses_pre_forward_bs32_and_worker_local_rank(tmp_path):
    workers = {}
    elected = {0: 3, 1: 1}
    for worker, selected_rank in elected.items():
        worker_log = tmp_path / f"node_decode_w{worker}.out"
        _write_auto_rank_capture(worker_log, selected_rank=selected_rank)
        workers[worker] = parse_exact_batch_capture_observations(
            worker_log,
            selected_batch=32,
            expected_steps=64,
            expected_gate_reduction="auto",
            expected_gate_ranks=None,
        )

    assert {worker: next(iter(ranks)) for worker, ranks in workers.items()} == elected
    assert all(
        evidence["observation_semantics"] == "pre_forward_runtime_gate"
        for ranks in workers.values()
        for evidence in ranks.values()
    )
    assert all(
        evidence["capture_owner_rank"] == rank
        for ranks in workers.values()
        for rank, evidence in ranks.items()
    )
    assert all(
        evidence["exact_observation_count"] == 6
        for ranks in workers.values()
        for evidence in ranks.values()
    )
    assert all(
        32 not in evidence["post_forward_batch_distribution"]
        for ranks in workers.values()
        for evidence in ranks.values()
    )

    selected = select_balanced_exact_observations(
        workers,
        selected_batch=32,
        sample_count=10,
        allowed_sources={"w0/r3", "w1/r1"},
        min_capture_iteration=1,
    )
    assert Counter(row["source"] for row in selected) == {
        "w0/r3": 5,
        "w1/r1": 5,
    }
    assert {row["running_requests"] for row in selected} == {32}
    assert {row["post_forward_running_requests"] for row in selected} == {31}


def test_sglang_auto_gate_rejects_capture_on_non_elected_rank(tmp_path):
    worker_log = tmp_path / "node_decode_w0.out"
    _write_auto_rank_capture(worker_log, selected_rank=3)
    text = worker_log.read_text()
    text = text.replace(
        "[x DP3 TP3 EP3] Started Nsight Systems NVTX capture range:",
        "[x DP0 TP0 EP0] Started Nsight Systems NVTX capture range:",
    )
    worker_log.write_text(text)

    with pytest.raises(ValueError, match="owner must be elected DP3"):
        parse_exact_batch_capture_observations(
            worker_log,
            selected_batch=32,
            expected_steps=64,
            expected_gate_reduction="auto",
            expected_gate_ranks=None,
        )


def test_sglang_any_rank_capture_filters_and_balances_32_real_bs32_samples(
    tmp_path,
):
    workers = {}
    for worker in range(2):
        worker_log = tmp_path / f"node_decode_w{worker}.out"
        _write_any_rank_variable_capture(worker_log)
        workers[worker] = parse_exact_batch_capture_observations(
            worker_log,
            selected_batch=32,
            expected_steps=64,
            expected_warmup_batches=1,
            expected_gate_reduction="any",
        )

    selected = select_balanced_exact_observations(
        workers, selected_batch=32, sample_count=32
    )

    assert len(selected) == 32
    assert {row["sample_index"] for row in selected} == set(range(32))
    assert {row["running_requests"] for row in selected} == {32}
    assert {row["source"] for row in selected} == {
        f"w{worker}/r{rank}" for worker in range(2) for rank in range(4)
    }

    rank_zero_selected = select_balanced_exact_observations(
        workers,
        selected_batch=32,
        sample_count=16,
        allowed_sources={"w0/r0", "w1/r0"},
    )
    assert len(rank_zero_selected) == 16
    assert {row["source"] for row in rank_zero_selected} == {"w0/r0", "w1/r0"}
    assert {
        source: sum(row["source"] == source for row in rank_zero_selected)
        for source in ("w0/r0", "w1/r0")
    } == {"w0/r0": 8, "w1/r0": 8}
    assert all(
        row["gate_reduction"] == "any"
        for ranks in workers.values()
        for row in ranks.values()
    )


def test_sglang_any_rank_capture_rejects_fewer_than_32_exact_samples(tmp_path):
    workers = {}
    for worker in range(2):
        worker_log = tmp_path / f"node_decode_w{worker}.out"
        _write_any_rank_variable_capture(worker_log, exact_per_rank=(1, 1, 1, 1))
        workers[worker] = parse_exact_batch_capture_observations(
            worker_log,
            selected_batch=32,
            expected_steps=64,
            expected_gate_reduction="any",
        )

    with pytest.raises(ValueError, match="only 8 valid rank-local BS32"):
        select_balanced_exact_observations(workers, selected_batch=32, sample_count=32)


def test_sglang_rank0_gate_selects_only_two_representative_sources(tmp_path):
    workers = {}
    for worker in range(2):
        worker_log = tmp_path / f"node_decode_w{worker}.out"
        _write_any_rank_variable_capture(
            worker_log,
            steps=32,
            exact_per_rank=(32, 1, 1, 1),
            gate_reduction="rank0",
            gate_rank=0,
        )
        workers[worker] = parse_exact_batch_capture_observations(
            worker_log,
            selected_batch=32,
            expected_steps=32,
            expected_gate_reduction="rank0",
        )

    selected = select_balanced_exact_observations(
        workers,
        selected_batch=32,
        sample_count=32,
        allowed_sources={"w0/r0", "w1/r0"},
    )

    assert Counter(row["source"] for row in selected) == {
        "w0/r0": 16,
        "w1/r0": 16,
    }


def test_sglang_local_gate_requires_only_dp0_capture_markers(tmp_path):
    worker_log = tmp_path / "node_decode_w0.out"
    _write_any_rank_variable_capture(
        worker_log,
        steps=17,
        exact_per_rank=(17, 1, 1, 1),
        gate_reduction="local",
        gate_rank=0,
    )

    observations = parse_exact_batch_capture_observations(
        worker_log,
        selected_batch=32,
        expected_steps=17,
        expected_gate_reduction="local",
        expected_gate_ranks=(0,),
    )

    assert set(observations) == {0}
    assert observations[0]["gate_reduction"] == "local"
    assert observations[0]["exact_observation_count"] == 17


def test_trt_exact_selector_balances_four_time_spread_samples_per_source():
    rows = [
        {
            "source": f"worker{worker}/rank{rank}",
            "capture_iteration": iteration,
        }
        for worker in range(2)
        for rank in range(4)
        for iteration in range(32)
    ]

    selected = select_balanced_rank_local_steps(rows, sample_count=32)

    counts = Counter(row["source"] for row in selected)
    assert set(counts.values()) == {4}
    assert {row["selected_sample_index"] for row in selected} == set(range(32))
    assert {
        row["capture_iteration"] for row in selected if row["source"] == "worker0/rank0"
    } == {4, 12, 20, 28}

    rank_three_selected = select_balanced_rank_local_steps(
        rows,
        sample_count=32,
        allowed_sources={"worker0/rank3", "worker1/rank3"},
    )
    assert Counter(row["source"] for row in rank_three_selected) == {
        "worker0/rank3": 16,
        "worker1/rank3": 16,
    }


def test_trt_comparison_source_election_is_worker_local_and_deterministic():
    counts = Counter(
        {
            "worker0/rank0": 9,
            "worker0/rank1": 12,
            "worker0/rank2": 12,
            "worker0/rank3": 10,
            "worker1/rank0": 8,
            "worker1/rank1": 11,
            "worker1/rank2": 13,
            "worker1/rank3": 14,
        }
    )

    assert elect_worker_comparison_sources(
        counts,
        workers=["worker0", "worker1"],
        minimum_per_source=5,
    ) == {"worker0/rank1", "worker1/rank3"}

    with pytest.raises(ValueError, match="best rank-local source"):
        elect_worker_comparison_sources(
            Counter({f"worker0/rank{rank}": 4 for rank in range(4)}),
            workers=["worker0"],
            minimum_per_source=5,
        )


def test_trt_rank_local_raw_capture_markers_allow_independent_boundaries(tmp_path):
    worker_log = tmp_path / "node_decode_w0.out"
    lines = []
    for rank in range(4):
        start = 100 + rank * 7
        lines.extend(
            [
                f"[TRT-LLM] [_torch][RANK {rank}] Rank-local BS32-triggered "
                f"raw profiling started at iteration {start}: local_batch=32, "
                "capture_raw_decode_batches=64",
                f"[TRT-LLM] [_torch][RANK {rank}] Rank-local BS32-triggered "
                f"raw profiling stopped at iteration {start + 64}: "
                f"local_batch={31 + rank % 2}, captured_raw_decode_batches=64",
            ]
        )
    worker_log.write_text("\n".join(lines) + "\n")

    evidence = _validate_exact_worker_log(worker_log, expected_steps=64)

    assert evidence["rank_start_count"] == 4
    assert evidence["rank_stop_count"] == 4
    assert evidence["start_iterations_by_rank"] == {
        "0": 100,
        "1": 107,
        "2": 114,
        "3": 121,
    }
    assert evidence["captured_raw_decode_iterations"] == 64
