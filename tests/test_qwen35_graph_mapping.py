import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.common.trace_mapping import ForwardWindow
from models.qwen35.profile.qwen35_graph_mapping import (
    TARGET_PATTERN,
    complete_eager_decode_window,
    direct_graph_mapping,
    map_graph_window,
    map_prefill_window,
)
from models.qwen35.profile.build_qwen35_sglang_agentx_profile import (
    parse_benchmark_snapshot,
    parse_worker_profile_observations,
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


def test_generation_lifecycle_signatures_are_not_left_in_a_generic_scope():
    assert direct_graph_mapping(
        "void VerifyTreeGreedy<int, long>()",
        substage="generation_lifecycle",
        layer_kind=None,
    ).node == "generation_loop.accept_prefix"
    assert direct_graph_mapping(
        "_fused_conv_window_scatter_with_mask_kernel",
        substage="generation_lifecycle",
        layer_kind=None,
    ).node == "generation_loop.commit_gdn"
    bonus = direct_graph_mapping(
        "fill_bonus_tokens",
        substage="generation_lifecycle",
        layer_kind=None,
    )
    assert bonus.node == "generation_loop.commit_tokens"
    assert bonus.ir_targets == ("generation_loop.accept_prefix",)


def test_graph_window_requires_exact_60_layer_ggga_sequence_and_labels_every_kernel():
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
    assert validation["attributed_duration_ratio"] == 1.0
    assert validation["target_verify_batch_size"] == 8
    assert validation["signature_counts"]["mtp_draft_rounds"] == 5
    assert {event["mapping_status"] for event in mapped} == {"mapped", "fusion"}
    assert all(event["node"] for event in mapped)
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
                "fused_qkvzba_split_reshape_cat_contiguous_kernel"
                if kind == "gdn"
                else "_fused_qk_rmsnorm_rope_gate_kernel",
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
    assert all(event["layer_id"] is None for event in mapped if event["substage"] == "mtp_seed_prefill")


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
            f"#running-req: {31 + rank}, other, accept len: 4.{7 + rank}, other, "
            "cuda graph: True, other, #queue-req: 0\n"
            for rank in range(4)
        )
        + "[x DP0 TP0 EP0] Stop profiling...\n"
        "[x DP0 TP0 EP0] Decode batch [20], #running-req: 88, accept len: 1.0, "
        "cuda graph: False, #queue-req: 2\n"
    )
    rows = parse_worker_profile_observations(worker_log)
    assert [row["running_requests"] for row in rows] == [31, 32, 33, 34]
    assert all(row["cuda_graph"] and row["queued_requests"] == 0 for row in rows)

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
