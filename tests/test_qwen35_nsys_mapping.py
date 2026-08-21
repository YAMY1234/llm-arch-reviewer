import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.qwen35.profile.qwen35_nsys_mapping import (
    NsysKernel,
    NsysStep,
    TARGET_PATTERN,
    map_decode_step,
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
