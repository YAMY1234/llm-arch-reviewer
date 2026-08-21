import sys
from pathlib import Path
import sqlite3


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.qwen35.profile.qwen35_nsys_mapping import (
    NsysKernel,
    NsysStep,
    TARGET_PATTERN,
    load_nsys_steps,
    map_decode_step,
    map_prefill_step,
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
        anchor = (
            "_causal_conv1d_update_kernel"
            if kind == "gdn"
            else "fmhaSm100_target"
        )
        kernels.append(_kernel(anchor, len(kernels)))
        _moe(kernels, layer_id, draft=False)
    kernels.extend(_kernel("_promote_mamba_state_kernel", len(kernels) + i) for i in range(45))
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
    assert validation["attributed_duration_ratio"] == 1.0
    assert all(event["mapping_status"] in {"mapped", "fusion"} for event in mapped)
    assert sum(event["node"] == "moe_block.target_ep4_dispatch" for event in mapped) == 60
    assert sum(event["node"] == "mtp_moe_block.draft_ep4_dispatch" for event in mapped) == 6


def test_trt_decode_mapping_rejects_missing_mtp_pass():
    step = _synthetic_step()
    truncated = NsysStep(**{**step.__dict__, "kernels": step.kernels[:-7]})
    try:
        map_decode_step(truncated)
    except ValueError as error:
        assert "66 MoE calls" in str(error) or "six MTP" in str(error)
    else:
        raise AssertionError("missing MTP pass was accepted")


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
                "causal_conv1d_fwd_kernel<128>"
                if kind == "gdn"
                else "fmhaSm103aKernel_Context",
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
    assert sum(event["node"] == "gdn_moe_block.causal_conv" for event in mapped) == 45
    assert sum(event["node"] == "full_attention_moe_block.causal_gqa" for event in mapped) == 15


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
            (0, 200, "[Executor] _forward_step 10: 0 ctx reqs, 0 ctx tokens, 1 gen reqs"),
            (200, 400, "[Executor] _forward_step 11: 0 ctx reqs, 0 ctx tokens, 1 gen reqs"),
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
    assert [kernel.name for kernel in steps[0].kernels] == ["direct", "node_a", "node_b"]
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
            (0, 200, "[Executor] _forward_step 10: 1 ctx reqs, 8 ctx tokens, 0 gen reqs"),
            (200, 400, "[Executor] _forward_step 11: 1 ctx reqs, 8 ctx tokens, 0 gen reqs"),
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
