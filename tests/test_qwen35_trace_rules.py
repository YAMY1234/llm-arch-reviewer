import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.common.trace_mapping import FrameRef
from models.qwen35.profile.qwen35_trace_rules import classify_qwen35_node


def frames(*values: str) -> list[FrameRef]:
    return [FrameRef(raw=value) for value in values]


def test_target_and_draft_moe_compute_stay_separate() -> None:
    target = classify_qwen35_node(
        "flashinfer_cutedsl_moe_kernel",
        None,
        frames("nn.Module: Qwen2MoeSparseMoeBlock", "nn.Module: Qwen3_5ForCausalLM"),
    )
    draft = classify_qwen35_node(
        "deep_gemm::fp8_moe",
        None,
        frames("nn.Module: Qwen2MoeSparseMoeBlock", "nn.Module: Qwen3_5ForCausalLMMTP"),
    )
    assert target == ("moe_block.routed_experts", "high")
    assert draft == ("mtp_moe_block.routed_experts", "high")


def test_dispatch_backends_bind_independent_wire_scopes() -> None:
    assert classify_qwen35_node(
        "moe_a2a_dispatch", None, frames("FlashinferDispatcher.dispatch")
    ) == ("moe_block.target_ep4_dispatch", "high")
    assert classify_qwen35_node(
        "deep_ep::internode_ll::combine",
        None,
        frames("nn.Module: Qwen3_5ForCausalLMMTP"),
    ) == ("mtp_moe_block.draft_ep4_combine", "high")


def test_accepted_prefix_replay_wins_over_gdn_signature() -> None:
    assert classify_qwen35_node(
        "gdn_wide_vec_kernel",
        None,
        frames("FrozenKVMTPDraftWorker.draft_extend", "Qwen3_5GatedDeltaNet.forward"),
    ) == ("generation_loop.replay_gdn", "high")


def test_full_attention_target_and_draft_stay_separate() -> None:
    target = classify_qwen35_node(
        "trtllm_mha_fmha",
        None,
        frames("nn.Module: Qwen3_5AttentionDecoderLayer", "nn.Module: Qwen3_5ForCausalLM"),
    )
    draft = classify_qwen35_node(
        "trtllm_mha_fmha",
        None,
        frames(
            "nn.Module: Qwen3_5AttentionDecoderLayer",
            "nn.Module: Qwen3_5ForCausalLMMTP",
        ),
    )
    assert target == ("full_attention_moe_block.causal_gqa", "high")
    assert draft == ("mtp_full_attention_moe_block.causal_gqa", "high")
