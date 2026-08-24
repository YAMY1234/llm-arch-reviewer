from collections import Counter
import math
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.common.trace_mapping import ForwardWindow
from models.qwen35.profile.qwen35_graph_mapping import (
    TARGET_PATTERN,
    attribution_active_union_ratio,
    complete_eager_decode_window,
    direct_graph_mapping,
    interval_union_duration_us,
    map_graph_window,
    map_prefill_window,
    transfer_occurrence_stack_mapping,
)
from models.qwen35.profile.build_qwen35_sglang_agentx_profile import (
    parse_benchmark_snapshot,
    parse_worker_profile_observations,
)
from models.qwen35.profile.build_qwen35_sglang_agentx_nsys_profile import (
    _profiled_scheduler_ranks,
    _source_coordinates,
    _validate_nsys_capture_contract,
    _validate_outer_worker_report_files,
    _validate_nsys_report_files,
)


def _annotation(name: str, ts: float, dur: float, tid: int = 1):
    return {
        "cat": "gpu_user_annotation",
        "ph": "X",
        "name": name,
        "ts": ts,
        "dur": dur,
        "pid": 0,
        "tid": tid,
    }


def _kernel(name: str, ts: float, dur: float = 1.0):
    return {
        "cat": "kernel",
        "ph": "X",
        "name": name,
        "ts": ts,
        "dur": dur,
        "pid": 0,
        "tid": 10,
        "args": {"stream": 10, "device": 0},
    }


def test_sglang_nsys_source_coordinates_are_strict():
    assert _source_coordinates("w0/r0") == (0, 0)
    assert _source_coordinates("w1/r3") == (1, 3)
    with pytest.raises(ValueError, match="invalid worker/rank source"):
        _source_coordinates("worker1/rank3")


def test_sglang_nsys_raw_reports_require_worker_balanced_rank3_pulses(tmp_path):
    fingerprints = [
        {"worker": 0, "hostname": "node-a"},
        {"worker": 1, "hostname": "node-b"},
    ]
    reports = [
        tmp_path / f"node-{hostname}-decode-rank3.{capture}.nsys-rep"
        for hostname in ("a", "b")
        for capture in range(1, 3)
    ]
    for path in reports:
        path.write_bytes(b"nsys")
    assert _validate_nsys_report_files(
        reports,
        fingerprints,
        expected_ranks=(3,),
        expected_reports_per_source=2,
    ) == {
        (worker, 3, capture): reports[worker * 2 + capture - 1].resolve()
        for worker in range(2)
        for capture in range(1, 3)
    }

    reports[-1].write_bytes(b"")
    with pytest.raises(ValueError, match="missing or empty"):
        _validate_nsys_report_files(
            reports,
            fingerprints,
            expected_ranks=(3,),
            expected_reports_per_source=2,
        )
    with pytest.raises(ValueError, match="report counts"):
        _validate_nsys_report_files(
            reports[:3],
            fingerprints,
            expected_ranks=(3,),
            expected_reports_per_source=2,
        )


def test_sglang_nsys_raw_reports_accept_one_peer_prime_and_rank3_pulses(tmp_path):
    fingerprints = [
        {"worker": 0, "hostname": "node-a"},
        {"worker": 1, "hostname": "node-b"},
    ]
    reports = []
    for hostname in ("a", "b"):
        for rank in range(4):
            captures = (1, 2) if rank == 3 else (1,)
            for capture in captures:
                path = tmp_path / f"node-{hostname}-decode-rank{rank}.{capture}.nsys-rep"
                path.write_bytes(b"nsys")
                reports.append(path)

    validated = _validate_nsys_report_files(
        reports,
        fingerprints,
        expected_ranks=(0, 1, 2, 3),
        expected_reports_by_rank={0: 1, 1: 1, 2: 1, 3: 2},
    )
    assert len(validated) == 10
    assert Counter(rank for _worker, rank, _capture in validated) == Counter(
        {0: 2, 1: 2, 2: 2, 3: 4}
    )


def test_sglang_nsys_outer_worker_scope_rejects_rank_local_pulses():
    assert _profiled_scheduler_ranks({}) == (0, 1, 2, 3)
    with pytest.raises(ValueError, match="cannot select rank-local"):
        _profiled_scheduler_ranks({"SGLANG_NSYS_SCHEDULER_RANKS": "0,1"})
    with pytest.raises(ValueError, match="continuous capture"):
        _profiled_scheduler_ranks(
            {"SGLANG_NSYS_PULSE_CAPTURE_PER_STEP": "1"}
        )


def test_sglang_outer_worker_reports_are_worker_balanced(tmp_path):
    fingerprints = [
        {"worker": 0, "hostname": "node-a"},
        {"worker": 1, "hostname": "node-b"},
    ]
    selected_rank_by_worker = {0: 3, 1: 1}
    reports = [
        tmp_path / f"node-{suffix}_decode_w{worker}_profile_gpu0-1-2-3.1.nsys-rep"
        for worker, suffix in enumerate(("a", "b"))
    ]
    for path in reports:
        path.write_bytes(b"nsys")
    assert _validate_outer_worker_report_files(
        reports, fingerprints, selected_rank_by_worker
    ) == {
        (0, 3, 0): reports[0].resolve(),
        (1, 1, 0): reports[1].resolve(),
    }


def test_sglang_nsys_capture_contract_requires_matching_nvtx_trigger():
    profiling = {
        "type": "nsys",
        "sglang_scheduler_nsys": False,
        "cuda_graph_trace": "node",
        "extra_nsys_args": [
            "-c",
            "nvtx",
            "-p",
            "agentx_decode_capture@*",
            "--capture-range-end",
            "repeat:1:async",
            "--kill",
            "none",
        ],
    }
    environment = {
        "SGLANG_NSYS_NVTX_CAPTURE_RANGE": "agentx_decode_capture",
        "NSYS_NVTX_PROFILER_REGISTER_ONLY": "0",
    }
    assert _validate_nsys_capture_contract(profiling, environment) == (
        "agentx_decode_capture",
        "repeat:1:async",
        "node",
    )

    profiling["sglang_scheduler_nsys"] = True
    with pytest.raises(ValueError, match="outer worker"):
        _validate_nsys_capture_contract(profiling, environment)
    profiling["sglang_scheduler_nsys"] = False

    environment.pop("NSYS_NVTX_PROFILER_REGISTER_ONLY")
    with pytest.raises(ValueError, match="unregistered NVTX"):
        _validate_nsys_capture_contract(profiling, environment)
    environment["NSYS_NVTX_PROFILER_REGISTER_ONLY"] = "0"

    profiling["extra_nsys_args"] = ["-c", "cudaProfilerApi"]
    with pytest.raises(ValueError, match="matching '-c nvtx'"):
        _validate_nsys_capture_contract(profiling, environment)

    profiling["extra_nsys_args"] = [
        "-c",
        "nvtx",
        "-p",
        "agentx_decode_capture",
        "--capture-range-end",
        "repeat:1:async",
    ]
    with pytest.raises(ValueError, match="all-domain"):
        _validate_nsys_capture_contract(profiling, environment)

    profiling["extra_nsys_args"] = [
        "-c",
        "nvtx",
        "-p",
        "agentx_decode_capture@*",
        "--capture-range-end",
        "stop",
    ]
    with pytest.raises(ValueError, match="asynchronously finalized"):
        _validate_nsys_capture_contract(profiling, environment)

    profiling["extra_nsys_args"] = [
        "-c",
        "nvtx",
        "-p",
        "agentx_decode_capture@*",
        "--capture-range-end",
        "repeat:1:async",
    ]
    profiling["cuda_graph_trace"] = "graph"
    with pytest.raises(ValueError, match="CUDA Graph node tracing"):
        _validate_nsys_capture_contract(profiling, environment)


def test_target_and_draft_collectives_remain_separate():
    target = direct_graph_mapping(
        "moeA2ADispatchKernel", substage="target_verify", layer_kind="gdn"
    )
    draft = direct_graph_mapping(
        "deep_ep::internode_ll::dispatch", substage="draft", layer_kind=None
    )
    assert target.node == "moe_block.target_ep4_dispatch"
    assert draft.node == "mtp_moe_block.draft_ep4_dispatch"
    assert target.status == draft.status == "mapped"


def test_attribution_gate_uses_per_source_active_union_not_residency_sum():
    def event(worker, rank, step, start, duration, status):
        return {
            "worker": worker,
            "rank": rank,
            "step_index": step,
            "ts_us": start,
            "dur_us": duration,
            "mapping_status": status,
        }

    rows = [
        event(0, 0, 1, 0.0, 10.0, "mapped"),
        event(0, 0, 1, 5.0, 10.0, "fusion"),
        event(0, 0, 1, 14.0, 6.0, "unmapped"),
        event(0, 0, 2, 100.0, 10.0, "mapped"),
        event(0, 0, 2, 110.0, 10.0, "unmapped"),
        # Same timestamps on another rank are an independent clock/source and
        # must contribute another 10 us, not be unioned with rank 0.
        event(0, 1, 1, 0.0, 10.0, "mapped"),
    ]
    assert math.isclose(interval_union_duration_us(rows[:3]), 20.0)
    assert math.isclose(attribution_active_union_ratio(rows), 0.7)


def test_generation_lifecycle_signatures_are_not_left_in_a_generic_scope():
    assert (
        direct_graph_mapping(
            "void VerifyTreeGreedy<int, long>()",
            substage="generation_lifecycle",
            layer_kind=None,
        ).node
        == "generation_loop.accept_prefix"
    )
    assert (
        direct_graph_mapping(
            "_fused_conv_window_scatter_with_mask_kernel",
            substage="generation_lifecycle",
            layer_kind=None,
        ).node
        == "generation_loop.commit_gdn"
    )
    bonus = direct_graph_mapping(
        "fill_bonus_tokens",
        substage="generation_lifecycle",
        layer_kind=None,
    )
    assert bonus.node == "generation_loop.commit_tokens"
    assert bonus.ir_targets == ("generation_loop.accept_prefix",)


def test_graph_window_preserves_unknown_kernels_as_explicit_unmapped_events():
    events = [
        _annotation("step[TARGET_VERIFY bs=8]", 100.0, 700.0),
        _annotation("draft_extend", 820.0, 40.0),
        _annotation("draft", 880.0, 80.0),
    ]
    for layer_id, kind in enumerate(TARGET_PATTERN):
        name = (
            "fused_qkvzba_split_reshape_cat_contiguous_kernel"
            if kind == "gdn"
            else "_fused_qk_rmsnorm_rope_gate_kernel"
        )
        events.append(_kernel(name, 110.0 + layer_id * 10.0))
        events.append(_kernel("nvjet_generic_projection", 115.0 + layer_id * 10.0))
    events.extend(
        [
            _kernel("_fused_qk_rmsnorm_rope_gate_kernel", 825.0),
            _kernel("deep_ep::internode_ll::dispatch", 830.0),
            _kernel("draft_auxiliary", 840.0),
            _kernel("_fused_qk_rmsnorm_rope_gate_kernel", 885.0),
            _kernel("deep_ep::internode_ll::combine", 900.0),
            _kernel("_fused_qk_rmsnorm_rope_gate_kernel", 905.0),
            _kernel("_fused_qk_rmsnorm_rope_gate_kernel", 925.0),
            _kernel("_fused_qk_rmsnorm_rope_gate_kernel", 945.0),
            _kernel("gdn_replayssm_exact_fold_kernel", 975.0),
        ]
    )
    mapped, validation = map_graph_window(
        events,
        window=ForwardWindow(100.0, 1000.0, [(100.0, 1000.0)], 45),
        rank=0,
        step_index=2,
    )
    assert validation["signature_counts"]["target_gdn_layers"] == 45
    assert validation["signature_counts"]["target_attention_layers"] == 15
    assert validation["attributed_duration_ratio"] < 1.0
    assert validation["timeline_interval_coverage_ratio"] == 1.0
    assert validation["target_verify_batch_size"] == 8
    assert validation["signature_counts"]["mtp_draft_rounds"] == 5
    assert {event["mapping_status"] for event in mapped} == {
        "mapped",
        "fusion",
        "unmapped",
    }
    unresolved = [event for event in mapped if event["mapping_status"] == "unmapped"]
    assert unresolved
    assert all(event["node"] is None for event in unresolved)
    assert all(event["candidate_nodes"] for event in unresolved)
    assert all(event["unmapped_reason"] for event in unresolved)
    assert any(event["layer_id"] == 59 for event in mapped)
    assert any(event["node"] == "generation_loop.replay_gdn" for event in mapped)
    assert all(
        "generation_loop.target_verify" in event["ir_targets"]
        for event in mapped
        if event["substage"] == "target_verify"
    )
    assert all(
        "generation_loop.draft_propose" in event["ir_targets"]
        for event in mapped
        if event["substage"] in {"draft", "draft_extend"}
    )


def test_prefill_window_separates_target_layers_from_mtp_seed():
    events = []
    for layer_id, kind in enumerate(TARGET_PATTERN):
        ts = 100.0 + layer_id * 10.0
        events.append(
            _kernel(
                (
                    "fused_qkvzba_split_reshape_cat_contiguous_kernel"
                    if kind == "gdn"
                    else "_fused_qk_rmsnorm_rope_gate_kernel"
                ),
                ts,
            )
        )
        events.append(_kernel("moeA2ADispatchKernel", ts + 2.0))
        events.append(_kernel("moeA2ACombineKernel", ts + 4.0))
    seed_start = 800.0
    events.extend(
        [
            _kernel("_fused_qk_rmsnorm_rope_gate_kernel", seed_start),
            _kernel("deep_ep::internode_ll::dispatch", seed_start + 2.0),
            _kernel("deep_ep::internode_ll::dispatch", seed_start + 3.0),
            _kernel("deep_ep::internode_ll::combine", seed_start + 4.0),
            _kernel("deep_ep::internode_ll::combine", seed_start + 5.0),
        ]
    )
    mapped, validation = map_prefill_window(
        events,
        start_us=100.0,
        end_us=900.0,
        rank=0,
        step_index=0,
        mtp_seed_start_us=seed_start,
    )
    assert validation["signature_counts"]["target_gdn_layers"] == 45
    assert validation["signature_counts"]["target_attention_layers"] == 15
    assert validation["signature_counts"]["mtp_seed_attention_layers"] == 1
    assert validation["signature_counts"]["mtp_seed_ep4_dispatch"] == 2
    assert all(
        event["layer_id"] is None
        for event in mapped
        if event["substage"] == "mtp_seed_prefill"
    )


def test_occurrence_transfer_requires_one_exact_contiguous_sequence():
    target = [
        {
            "event_id": f"target-{index}",
            "kernel_name": name,
            "dur_us": 2.0,
            "mapping_status": "unmapped",
            "attribution_method": "unresolved",
        }
        for index, name in enumerate(("scheduler", "a", "b", "c"))
    ]
    source = [
        {
            "event_id": f"source-{index}",
            "kernel_name": name,
            "dur_us": 99.0,
            "node": "gdn_attention.causal_conv",
            "ir_targets": [],
            "mapping_status": "mapped",
            "attribution_method": "occurrence_python_stack",
            "confidence": "high",
            "python_stack": [name],
        }
        for index, name in enumerate(("a", "b", "c", "profiler_tail"))
    ]
    transferred, validation = transfer_occurrence_stack_mapping(target, source)
    assert transferred[0]["mapping_status"] == "unmapped"
    assert [event["node"] for event in transferred[1:]] == [
        "gdn_attention.causal_conv"
    ] * 3
    assert [event["dur_us"] for event in transferred] == [2.0] * 4
    assert validation["occurrence_alignment"] == {
        "method": "exact_contiguous_kernel_name_sequence",
        "target_prefix_untransferred": 1,
        "aligned_kernel_count": 3,
        "source_suffix_unused": 1,
    }

    source[1]["kernel_name"] = "different"
    try:
        transfer_occurrence_stack_mapping(target, source)
    except ValueError as error:
        assert "exact contiguous alignment" in str(error)
    else:
        raise AssertionError("fuzzy occurrence sequence was accepted")


def test_eager_window_uses_preceding_draft_when_one_step_stop_cuts_tail():
    events = [
        _annotation("draft", 20.0, 70.0),
        _annotation("step[TARGET_VERIFY bs=1]", 100.0, 100.0),
        _annotation("draft_extend", 220.0, 50.0),
    ]
    events.extend(
        _kernel("_fused_qk_rmsnorm_rope_gate_kernel", ts)
        for ts in (30.0, 45.0, 60.0, 75.0, 230.0)
    )
    window = complete_eager_decode_window(
        events,
        ForwardWindow(100.0, 270.0, [(100.0, 270.0)], 45),
        rank=0,
    )
    assert window.start_us == 20.0
    assert window.end_us == 270.0


def test_agentx_log_parsers_keep_only_the_measured_steady_window(tmp_path):
    worker_log = tmp_path / "node_decode_w0.out"
    worker_log.write_text(
        "[x DP0 TP0 EP0] Decode batch [9], #running-req: 99, accept len: 1.0, "
        "cuda graph: False, #queue-req: 3\n"
        "[x DP0 TP0 EP0] Profiling starts.\n"
        + "".join(
            f"[x DP{rank} TP{rank} EP{rank}] Decode batch [{10 + rank}], "
            f"#running-req: {31 + rank}, #full token: {6200000 + rank}, other, "
            f"accept len: 4.{7 + rank}, #retracted-req: 0, other, "
            "cuda graph: True, other, #queue-req: 0\n"
            for rank in range(4)
        )
        + "[x DP0 TP0 EP0] Stop profiling...\n"
        "[x DP0 TP0 EP0] Decode batch [20], #running-req: 88, accept len: 1.0, "
        "cuda graph: False, #queue-req: 2\n"
    )
    rows = parse_worker_profile_observations(worker_log)
    assert [row["running_requests"] for row in rows] == [31, 32, 33, 34]
    assert [row["full_tokens"] for row in rows] == [6200000, 6200001, 6200002, 6200003]
    assert all(
        row["cuda_graph"]
        and row["queued_requests"] == 0
        and row["retracted_requests"] == 0
        for row in rows
    )

    benchmark = tmp_path / "benchmark.out"
    benchmark.write_text(
        "rps=27.8 (avg 27.2) tput=1 done=28,533 ok=28,533 err=0\n"
        "rps=30.4 (avg 27.3) tput=1 done=29,567 ok=29,567 err=0\n"
    )
    snapshot = parse_benchmark_snapshot(benchmark)
    assert snapshot == {
        "instant_rps": 30.4,
        "average_rps": 27.3,
        "done": 29567,
        "ok": 29567,
        "errors": 0,
    }


def test_agentx_log_parser_rejects_retracted_profile_rows(tmp_path):
    worker_log = tmp_path / "node_decode_w0.out"
    worker_log.write_text(
        "[x DP0 TP0 EP0] Profiling starts.\n"
        + "".join(
            f"[x DP{rank} TP{rank} EP{rank}] Decode batch [{10 + rank}], "
            f"#running-req: 32, #full token: 6200000, accept len: 4.8, "
            f"#retracted-req: {1 if rank == 2 else 0}, cuda graph: True, "
            "#queue-req: 0\n"
            for rank in range(4)
        )
        + "[x DP0 TP0 EP0] Stop profiling...\n"
    )
    with pytest.raises(ValueError, match="queue/retraction-free"):
        parse_worker_profile_observations(worker_log)
