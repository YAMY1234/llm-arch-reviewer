import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.common.trace_mapping import ForwardWindow
from models.qwen35.profile.qwen35_graph_mapping import (
    TARGET_PATTERN,
    direct_graph_mapping,
    map_graph_window,
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
            _kernel("deep_ep::internode_ll::dispatch", 830.0),
            _kernel("draft_auxiliary", 840.0),
            _kernel("deep_ep::internode_ll::combine", 900.0),
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
    assert {event["mapping_status"] for event in mapped} == {"mapped", "fusion"}
    assert all(event["node"] for event in mapped)
    assert any(event["layer_id"] == 59 for event in mapped)
    assert any(event["node"] == "generation_loop.replay_gdn" for event in mapped)


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
