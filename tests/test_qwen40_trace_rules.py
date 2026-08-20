from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.common.trace_mapping import FrameRef  # noqa: E402
from models.qwen40.build.qwen40_trace_rules import (  # noqa: E402
    classify_qwen40_node,
    classify_qwen40_node_for_config,
)


def frames(*raw: str) -> list[FrameRef]:
    return [FrameRef(item) for item in raw]


class Qwen40TraceRulesTest(unittest.TestCase):
    def test_token_embedding_collective_is_separate_from_lookup(self):
        node, confidence = classify_qwen40_node(
            "flashinfer::trtllm_allreduce_fusion_kernel",
            None,
            frames(
                "nn.Module: VocabParallelEmbedding_0",
                "nn.Module: Qwen4ExpModel_0",
            ),
        )
        self.assertEqual(node, "top.tp_embedding_collective")
        self.assertEqual(confidence, "high")

    def test_ngram_embedding_collective_uses_ple_execution_node(self):
        node, confidence = classify_qwen40_node(
            "ncclDevKernel_AllReduce",
            None,
            frames(
                "nn.Module: VocabParallelEmbedding_0",
                "nn.Module: Qwen4ExpNGramEmbedding_0",
            ),
        )
        self.assertEqual(node, "ple.tp_embedding_collective")
        self.assertEqual(confidence, "high")

    def test_logits_allgather_is_separate_from_lm_head(self):
        node, confidence = classify_qwen40_node(
            "ncclDevKernel_AllGather",
            None,
            frames(
                "python/sglang/srt/layers/logits_processor.py(1): _get_logits",
                "nn.Module: LogitsProcessor_0",
            ),
        )
        self.assertEqual(node, "top.tp_logits_collective")
        self.assertEqual(confidence, "high")

    def test_moe_collective_precedes_moe_fallback(self):
        node, confidence = classify_qwen40_node(
            "ncclDevKernel_AllReduce",
            None,
            frames(
                "python/sglang/srt/layers/moe/fused_moe_triton/layer.py(1): forward",
                "nn.Module: FusedMoE_0",
                "nn.Module: Qwen2MoeSparseMoeBlock_0",
            ),
        )
        self.assertEqual(node, "moe.tp_output_collective")
        self.assertEqual(confidence, "high")

    def test_linear_attention_collective_uses_layer_context(self):
        node, confidence = classify_qwen40_node(
            "ncclSymkDevKernel_AllReduce",
            None,
            frames("nn.Module: Qwen4ExpLinearDecoderLayer_7"),
        )
        self.assertEqual(node, "linear_layer.tp_attention_collective")
        self.assertEqual(confidence, "high")

    def test_generic_qsa_path_is_not_automatically_the_indexer(self):
        node, _ = classify_qwen40_node(
            "flash_attention_kernel",
            None,
            frames(
                "python/sglang/srt/layers/attention/qsa/backend.py(1): forward",
                "nn.Module: Qwen4ExpAttentionDecoderLayer_3",
                "python/sglang/srt/layers/radix_attention.py(1): forward",
            ),
        )
        self.assertEqual(node, "qsa_attention.attention_core")

    def test_flashinfer_gdn_signature_wins_over_stale_stack_context(self):
        node, confidence = classify_qwen40_node(
            "kernel_cutlass_gdn_decode_bf16state_mtp_ilp4_kernel",
            None,
            frames("nn.Module: Qwen2MoeSparseMoeBlock_0"),
        )
        self.assertEqual(node, "linear_attention.delta_rule")
        self.assertEqual(confidence, "high")

    def test_explicit_qsa_indexer_frame_maps_to_indexer(self):
        node, confidence = classify_qwen40_node(
            "qsa_topk_kernel",
            None,
            frames(
                "python/sglang/srt/layers/attention/qsa/qsa_indexer.py(1): forward",
                "nn.Module: QSAIndexer_0",
            ),
        )
        self.assertEqual(node, "qsa_attention.indexer")
        self.assertEqual(confidence, "high")

    def test_dp_moe_gather_uses_layer_specific_bridge_node(self):
        node, confidence = classify_qwen40_node_for_config(
            "dp_attention",
            "ncclDevKernel_AllReduce",
            None,
            frames(
                "python/sglang/srt/layers/dp_attention.py(1): _dp_gather",
                "python/sglang/srt/models/qwen4_exp.py(1): _run_qwen4_exp_mlp",
                "nn.Module: Qwen4ExpAttentionDecoderLayer_3",
            ),
        )
        self.assertEqual(node, "full_layer.dp_moe_input_gather")
        self.assertEqual(confidence, "high")

    def test_dp_logits_scatter_uses_topology_bridge_node(self):
        node, confidence = classify_qwen40_node_for_config(
            "dp_attention",
            "memcpy_triton_kernel",
            None,
            frames(
                "python/sglang/srt/layers/dp_attention.py(1): dp_scatter",
                "python/sglang/srt/layers/logits_processor.py(1): _scatter_dp_attn_logits",
                "nn.Module: LogitsProcessor_0",
            ),
        )
        self.assertEqual(node, "top.dp_logits_output_scatter")
        self.assertEqual(confidence, "high")

    def test_ep_collective_is_not_labeled_tp(self):
        node, confidence = classify_qwen40_node_for_config(
            "ep4_a2a_none",
            "ncclDevKernel_AllReduce",
            None,
            frames(
                "python/sglang/srt/layers/moe/fused_moe_triton/layer.py(1): forward",
                "nn.Module: FusedMoE_0",
                "nn.Module: Qwen2MoeSparseMoeBlock_0",
            ),
        )
        self.assertEqual(node, "moe.ep_output_collective")
        self.assertEqual(confidence, "high")

    def test_deepep_dispatch_precedes_generic_moe_fallback(self):
        node, confidence = classify_qwen40_node_for_config(
            "dp_attention_ep4_deepep_deepgemm",
            "deep_ep_dispatch_kernel",
            None,
            frames(
                "python/sglang/srt/layers/moe/token_dispatcher/deepep.py(1): _dispatch_core",
                "nn.Module: Qwen2MoeSparseMoeBlock_0",
            ),
        )
        self.assertEqual(node, "moe.deepep_dispatch")
        self.assertEqual(confidence, "high")


if __name__ == "__main__":
    unittest.main()
