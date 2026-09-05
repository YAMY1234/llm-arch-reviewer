from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.common.trace_mapping import FrameRef  # noqa: E402
from models.qwen38_flash_next.build.qwen38_flash_next_trace_rules import (  # noqa: E402
    classify_qwen38_flash_next_node,
    classify_qwen38_flash_next_node_for_config,
)


def frames(*raw: str) -> list[FrameRef]:
    return [FrameRef(item) for item in raw]


class Qwen38FlashNextTraceRulesTest(unittest.TestCase):
    def test_ple_context_preparation_maps_without_ple_module_frame(self):
        node, confidence = classify_qwen38_flash_next_node(
            "void at::native::vectorized_elementwise_kernel",
            "aten::where",
            frames(
                "sglang/srt/models/qwen4_exp.py(97): _prepare_ple_batch",
                "nn.Module: Qwen4ExpModel_0",
            ),
        )
        self.assertEqual(node, "ple.token_history")
        self.assertEqual(confidence, "high")

    def test_ple_context_commit_maps_without_ple_module_frame(self):
        node, confidence = classify_qwen38_flash_next_node(
            "void at::native::index_elementwise_kernel",
            "aten::index_put_",
            frames(
                "sglang/srt/models/qwen4_exp.py(234): _commit_ple_batch",
                "nn.Module: Qwen4ExpModel_0",
            ),
        )
        self.assertEqual(node, "ple.context_commit")
        self.assertEqual(confidence, "high")

    def test_token_embedding_collective_is_separate_from_lookup(self):
        node, confidence = classify_qwen38_flash_next_node(
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
        node, confidence = classify_qwen38_flash_next_node(
            "ncclDevKernel_AllReduce",
            None,
            frames(
                "nn.Module: VocabParallelEmbedding_0",
                "nn.Module: Qwen4ExpNGramEmbedding_0",
            ),
        )
        self.assertEqual(node, "ple.tp_embedding_collective")
        self.assertEqual(confidence, "high")

    def test_token_embedding_collective_wins_over_eagle_orchestration(self):
        node, confidence = classify_qwen38_flash_next_node(
            "flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel",
            "sglang::flashinfer_allreduce",
            frames(
                "sglang/srt/layers/vocab_parallel_embedding.py(578): forward",
                "nn.Module: VocabParallelEmbedding_0",
                "sglang/srt/speculative/eagle_worker_common.py(461): run_eagle_verify",
            ),
        )
        self.assertEqual(node, "top.tp_embedding_collective")
        self.assertEqual(confidence, "high")

    def test_deferred_ple_collective_wins_over_linear_layer_context(self):
        node, confidence = classify_qwen38_flash_next_node(
            "flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel",
            "sglang::flashinfer_allreduce",
            frames(
                "sglang/srt/models/qwen4_exp.py(852): reduce",
                "sglang/srt/models/qwen4_exp.py(1133): _consume_prefetched_embeddings",
                "nn.Module: Qwen4ExpPLELayer_0",
                "nn.Module: Qwen4ExpLinearDecoderLayer_1",
            ),
        )
        self.assertEqual(node, "ple.tp_embedding_collective")
        self.assertEqual(confidence, "high")

    def test_logits_allgather_is_separate_from_lm_head(self):
        node, confidence = classify_qwen38_flash_next_node(
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
        node, confidence = classify_qwen38_flash_next_node(
            "ncclDevKernel_AllReduce",
            None,
            frames(
                "python/sglang/srt/layers/moe/fused_moe_triton/layer.py(1): forward",
                "nn.Module: FusedMoE_0",
                "nn.Module: Qwen2MoeSparseMoeBlock_0",
                "nn.Module: Qwen4ExpLinearDecoderLayer_0",
            ),
        )
        self.assertEqual(node, "linear_layer.tp_moe_output_collective")
        self.assertEqual(confidence, "high")

    def test_linear_attention_collective_uses_layer_context(self):
        node, confidence = classify_qwen38_flash_next_node(
            "ncclSymkDevKernel_AllReduce",
            None,
            frames("nn.Module: Qwen4ExpLinearDecoderLayer_7"),
        )
        self.assertEqual(node, "linear_layer.tp_attention_collective")
        self.assertEqual(confidence, "high")

    def test_generic_qsa_path_is_not_automatically_the_indexer(self):
        node, _ = classify_qwen38_flash_next_node(
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
        node, confidence = classify_qwen38_flash_next_node(
            "kernel_cutlass_gdn_decode_bf16state_mtp_ilp4_kernel",
            None,
            frames("nn.Module: Qwen2MoeSparseMoeBlock_0"),
        )
        self.assertEqual(node, "linear_attention.delta_rule")
        self.assertEqual(confidence, "high")

    def test_eager_prefill_gdn_split_signature_maps_to_split_pack(self):
        node, confidence = classify_qwen38_flash_next_node(
            "fused_qkv_split_gdn_prefill_kernel",
            None,
            frames(
                "sglang/srt/layers/attention/linear/gdn_backend.py(686): forward_extend",
                "nn.Module: Qwen3_5GatedDeltaNet_0",
            ),
        )
        self.assertEqual(node, "linear_attention.split_pack")
        self.assertEqual(confidence, "high")

    def test_flashinfer_dense_row_parallel_gdn_gemm_maps_to_output_projection(self):
        node, confidence = classify_qwen38_flash_next_node(
            "kernel_cutlass_dense_bf16_gemm_sm100",
            None,
            frames(
                "sglang/srt/layers/linear.py(1612): forward",
                "nn.Module: RowParallelLinear_0",
                "nn.Module: Qwen3_5GatedDeltaNet_0",
                "nn.Module: Qwen4ExpLinearDecoderLayer_0",
            ),
        )
        self.assertEqual(node, "linear_attention.output_projection")
        self.assertEqual(confidence, "medium")

    def test_flashinfer_dense_row_parallel_qsa_gemm_maps_to_output_projection(self):
        node, confidence = classify_qwen38_flash_next_node(
            "kernel_cutlass_dense_bf16_gemm_sm100",
            None,
            frames(
                "sglang/srt/layers/linear.py(1612): forward",
                "nn.Module: RowParallelLinear_6",
                "sglang/srt/models/qwen4_exp.py(1483): self_attention",
                "nn.Module: Qwen4ExpAttentionDecoderLayer_0",
            ),
        )
        self.assertEqual(node, "qsa_attention.output_projection")
        self.assertEqual(confidence, "medium")

    def test_fused_sigmoid_mul_maps_to_qsa_output_gate_without_cpu_op(self):
        node, confidence = classify_qwen38_flash_next_node(
            "_fused_sigmoid_mul_kernel",
            None,
            frames(
                "sglang/srt/models/qwen4_exp.py(1483): self_attention",
                "nn.Module: Qwen4ExpAttentionDecoderLayer_0",
            ),
        )
        self.assertEqual(node, "qsa_attention.output_gate")
        self.assertEqual(confidence, "medium")

    def test_explicit_qsa_indexer_frame_maps_to_indexer(self):
        node, confidence = classify_qwen38_flash_next_node(
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
        node, confidence = classify_qwen38_flash_next_node_for_config(
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
        node, confidence = classify_qwen38_flash_next_node_for_config(
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
        node, confidence = classify_qwen38_flash_next_node_for_config(
            "ep4_a2a_none",
            "ncclDevKernel_AllReduce",
            None,
            frames(
                "python/sglang/srt/layers/moe/fused_moe_triton/layer.py(1): forward",
                "nn.Module: FusedMoE_0",
                "nn.Module: Qwen2MoeSparseMoeBlock_0",
                "nn.Module: Qwen4ExpAttentionDecoderLayer_3",
            ),
        )
        self.assertEqual(node, "full_layer.ep_moe_output_collective")
        self.assertEqual(confidence, "high")

    def test_deepep_dispatch_precedes_generic_moe_fallback(self):
        node, confidence = classify_qwen38_flash_next_node_for_config(
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

    def test_mtp_embedding_collective_stays_in_auxiliary_scope(self):
        node, confidence = classify_qwen38_flash_next_node(
            "ncclDevKernel_AllReduce",
            None,
            frames(
                "nn.Module: VocabParallelEmbedding_0",
                "python/sglang/srt/models/qwen4_exp_mtp.py(133): _prepare_input_embeds",
                "nn.Module: Qwen4ExpForCausalLMMTP_0",
            ),
        )
        self.assertEqual(node, "mtp_head.tp_embedding_collective")
        self.assertEqual(confidence, "high")

    def test_mtp_qsa_is_not_aggregated_into_target_qsa(self):
        node, _ = classify_qwen38_flash_next_node(
            "flash_attention_kernel",
            None,
            frames(
                "python/sglang/srt/layers/radix_attention.py(1): forward",
                "nn.Module: Qwen4ExpAttentionDecoderLayer_0",
                "python/sglang/srt/models/qwen4_exp_mtp.py(183): forward",
                "nn.Module: Qwen4ExpForCausalLMMTP_0",
            ),
        )
        self.assertEqual(node, "mtp_qsa_attention.attention_core")

    def test_mtp_qsa_output_gate_is_not_aggregated_into_target_qsa(self):
        node, confidence = classify_qwen38_flash_next_node(
            "_fused_sigmoid_mul_kernel",
            None,
            frames(
                "sglang/srt/models/qwen4_exp.py(1483): self_attention",
                "nn.Module: Qwen4ExpAttentionDecoderLayer_0",
                "sglang/srt/models/qwen4_exp_mtp.py(197): forward",
                "nn.Module: Qwen4ExpForCausalLMMTP_0",
            ),
        )
        self.assertEqual(node, "mtp_qsa_attention.output_gate")
        self.assertEqual(confidence, "medium")

    def test_eagle_acceptance_stack_is_semantic_evidence(self):
        node, confidence = classify_qwen38_flash_next_node(
            "sgl_kernel::verify_tree_greedy",
            "sgl_kernel::verify_tree_greedy",
            frames(
                "sglang/srt/speculative/eagle_utils.py(377): verify_tree_greedy_func",
                "sglang/srt/speculative/eagle_utils.py(713): eagle_sample",
                "sglang/srt/speculative/eagle_worker_common.py(461): run_eagle_verify",
            ),
        )
        self.assertEqual(node, "mtp_generation.accept_commit")
        self.assertEqual(confidence, "high")

    def test_bare_eagle_verify_executor_work_is_runtime_support(self):
        node, confidence = classify_qwen38_flash_next_node(
            "void at::native::index_elementwise_kernel",
            "aten::index",
            frames(
                "sglang/srt/model_executor/cuda_graph_buffer_registry.py(88): copy",
                "sglang/srt/speculative/eagle_worker_common.py(461): run_eagle_verify",
            ),
        )
        self.assertEqual(node, "top.runtime_support")
        self.assertEqual(confidence, "high")

    def test_scheduler_request_broadcast_collective_is_runtime_support(self):
        node, confidence = classify_qwen38_flash_next_node(
            "_all_gather_kernel_inner",
            None,
            frames(
                "sglang/srt/managers/scheduler.py(1929): event_loop_overlap",
                "sglang/srt/managers/scheduler.py(5582): dispatch_event_loop",
            ),
        )
        self.assertEqual(node, "top.runtime_support")
        self.assertEqual(confidence, "high")

    def test_hc_expansion_cat_uses_source_exact_layer_boundary(self):
        node, confidence = classify_qwen38_flash_next_node(
            "void at::native::CatArrayBatchedCopy",
            "aten::cat",
            frames(
                "sglang/srt/models/qwen4_exp.py(1320): _prepare_qwen4_exp_attn",
                "nn.Module: Qwen4ExpLinearDecoderLayer_0",
            ),
        )
        self.assertEqual(node, "stack.hc_expand")
        self.assertEqual(confidence, "high")

    def test_configured_ple_residual_add_uses_source_exact_layer_boundary(self):
        node, confidence = classify_qwen38_flash_next_node(
            "void at::native::gpu_kernel<CUDAGenericFunctor_add>",
            "aten::add",
            frames(
                "sglang/srt/models/qwen4_exp.py(1320): _prepare_qwen4_exp_attn",
                "nn.Module: Qwen4ExpLinearDecoderLayer_1",
            ),
        )
        self.assertEqual(node, "ple.injection")
        self.assertEqual(confidence, "high")

    def test_async_moe_bmm_uses_unique_backend_signature(self):
        node, confidence = classify_qwen38_flash_next_node(
            "Bmm_Bfloat16_foo",
            None,
            frames("sglang/srt/models/qwen4_exp.py(1400): _postprocess_qwen4_exp_layer"),
        )
        self.assertEqual(node, "moe.routed_experts")
        self.assertEqual(confidence, "high")

    def test_async_moe_combine_uses_unique_backend_signature(self):
        node, confidence = classify_qwen38_flash_next_node(
            "_fused_gate_sigmoid_mul_add_kernel",
            None,
            frames("sglang/srt/models/qwen4_exp.py(1400): _postprocess_qwen4_exp_layer"),
        )
        self.assertEqual(node, "moe.combine")
        self.assertEqual(confidence, "high")

    def test_eagle_draft_extend_stack_is_semantic_evidence(self):
        node, confidence = classify_qwen38_flash_next_node(
            "void at::native::index_elementwise_kernel",
            "aten::index",
            frames(
                "sglang/srt/speculative/eagle_worker_v2.py(994): _draft_extend_for_decode",
                "sglang/srt/speculative/eagle_worker_v2.py(1252): forward_batch_generation",
            ),
        )
        self.assertEqual(node, "mtp_generation.mtp_draft_extend")
        self.assertEqual(confidence, "high")

    def test_eagle_mtp_execution_path_uses_the_same_tp_classifier(self):
        node, confidence = classify_qwen38_flash_next_node_for_config(
            "tp_only_eagle_mtp",
            "_fused_sigmoid_mul_kernel",
            None,
            frames(
                "sglang/srt/models/qwen4_exp.py(1483): self_attention",
                "nn.Module: Qwen4ExpAttentionDecoderLayer_0",
            ),
        )
        self.assertEqual(node, "qsa_attention.output_gate")
        self.assertEqual(confidence, "medium")

    def test_mtp_moe_collective_stays_in_auxiliary_layer(self):
        node, confidence = classify_qwen38_flash_next_node(
            "ncclSymkDevKernel_AllReduce",
            None,
            frames(
                "nn.Module: FusedMoE_0",
                "nn.Module: Qwen2MoeSparseMoeBlock_0",
                "nn.Module: Qwen4ExpAttentionDecoderLayer_0",
                "nn.Module: Qwen4ExpForCausalLMMTP_0",
            ),
        )
        self.assertEqual(node, "mtp_layer.tp_moe_output_collective")
        self.assertEqual(confidence, "high")

    def test_mtp_async_moe_bmm_overrides_stale_hc_span(self):
        node, confidence = classify_qwen38_flash_next_node(
            "bmm_Bfloat16_Bfloat16Bfloat16_Fp32",
            None,
            frames(
                "python/sglang/srt/layers/hyperconnection.py(1): mix",
                "nn.Module: Qwen4ExpForCausalLMMTP_0",
            ),
        )
        self.assertEqual(node, "mtp_moe.routed_experts")
        self.assertEqual(confidence, "high")

    def test_mtp_hyperconnection_keeps_reusable_leaf_semantics(self):
        node, confidence = classify_qwen38_flash_next_node(
            "void sglang::hc_combine_kernel",
            None,
            frames(
                "python/sglang/srt/layers/hyperconnection.py(1): combine",
                "python/sglang/srt/models/qwen4_exp.py(1): _postprocess_qwen4_exp_layer",
                "nn.Module: Qwen4ExpForCausalLMMTP_0",
            ),
        )
        self.assertEqual(node, "hyperconnection.combine")
        self.assertEqual(confidence, "high")


if __name__ == "__main__":
    unittest.main()
