#!/usr/bin/env python3
"""Fail-closed vLLM eager-trace rules for the Kimi K3 stable IR.

Only exact source scopes and implementation-unique operation signatures are
classified here.  Ambiguous generic kernels deliberately remain unresolved
until an occurrence-bounded reconciliation proves their source order.
"""

from __future__ import annotations

from models.common.trace_mapping import FrameRef, StackFrameRules, TraceMappingRules


ATTN_RES_CALLS_PER_FORWARD = 187
ATTN_RES_SIGNATURE = "attn_res"


def _has(text: str, *needles: str) -> bool:
    return any(needle in text for needle in needles)


def classify_kimi_k3_vllm_node(
    kernel_name: str,
    cpu_op_name: str | None,
    stack: list[FrameRef],
) -> tuple[str | None, str]:
    names = "\n".join(frame.raw for frame in stack).lower()
    kernel = kernel_name.lower()
    cpu = (cpu_op_name or "").lower()
    in_kda = _has(names, "models/kimi_k3/nvidia/kda.py", "kimik3deltaattention")
    in_mla = _has(names, "models/kimi_k3/nvidia/mla.py", "multiheadlatentattention")
    in_moe = _has(
        names,
        "models/kimi_k3/nvidia/latent_moe_runner.py",
        "nn.module: kimimoe_",
        "nn.module: kimik3moe_",
        "routed_expert_down_proj",
        "routed_expert_up_proj",
        "shared_experts",
        "models/kimi_k3/nvidia/ops/latent_moe_tail.py",
        "block_sparse_moe",
        "fused_moe/",
    )
    in_dense = "nn.module: kimimlp_" in names and not in_moe
    is_regular_skinny_gemm = (
        "model_executorkernelslinearcute_dsl_skinny_gemm" in kernel
    )
    is_low_latency_bf16_gemm = "fused_a_gemm_kernel" in kernel

    # Decode's low-latency GEMM paths launch through TVM/CuTeDSL and do not
    # expose an aten::mm CPU op.  Their exact K3 module and authored wrapper
    # stacks still distinguish every projection without relying on shapes.
    if is_regular_skinny_gemm:
        if in_kda and "_kimigdnmergedcolumnparallellinear" in names:
            return "kda.qkv_projection", "high"
        if in_kda and "rowparallellinear" in names:
            return "kda.output_projection", "high"
        if in_mla and "rowparallellinear" in names:
            return "gated_mla.output_projection", "high"
        if in_moe and "model.py(806): <lambda>" in names:
            return "stable_latent_moe.routed_down", "high"

    if is_low_latency_bf16_gemm:
        if in_moe and "shared_experts" in names:
            if "rowparallellinear" in names:
                return "stable_latent_moe.shared_down", "high"
            if "mergedcolumnparallellinear" in names:
                return "stable_latent_moe.shared_gate_up", "high"
        if in_mla:
            if "mla.py(573): <lambda>" in names:
                return "gated_mla.kv_up", "high"
            if "_apply_q_lora_attention" in names:
                return "gated_mla.q_up", "high"
            if "mla.py(568): <lambda>" in names:
                return "gated_mla.q_down", "high"

    if _has(kernel, "ll_bf16_dotprod", "ll_bf16_splitk") and "gatelinear" in names:
        return "stable_latent_moe.router_logits", "high"

    # The exact CuTeDSL owner fuses each row-parallel output projection with
    # its TP reduce-scatter/all-reduce.  Keep one measured projection owner;
    # the profile contract links the collective child to that owner.
    if "gemm_rs_ar" in kernel:
        if in_kda:
            return "kda.output_projection", "high"
        if in_mla:
            return "gated_mla.output_projection", "high"
        if in_dense:
            return "dense_mlp.down", "high"

    # AttnRes is the stable 187-call execution skeleton: twice per decoder
    # layer plus one output aggregation.  Both the native SM100 kernel and the
    # Triton fallback preserve an implementation-unique name.
    if _has(kernel, "attn_res_fwd_online_v2_kernel", "_attn_res_kernel"):
        if "model.py(1401): forward" in names:
            return "top.output_attn_res", "high"
        return "attn_res.weighted_merge", "high"

    # Native K3 decode performs convolution, bounded decay, state update,
    # readout and gated RMSNorm in one physical owner.
    if "kda_decode_fusion_many_heads_kernel" in kernel:
        return "kda.recurrent_update", "high"
    if "causal_conv1d" in kernel:
        return "kda.q_short_conv", "high"
    if _has(
        kernel,
        "flashkda",
        "flash_kda",
        "kda_gate_chunk_cumsum",
        "chunk_kda_fwd_kernel_intra",
        "chunk_kda_fwd_kernel_inter",
        "recompute_w_u",
        "chunk_gated_delta_rule",
    ):
        return "kda.recurrent_update", "high"
    if "store_cache_checkpoints" in kernel:
        return "kda.recurrent_update", "high"
    if "gather_initial_states_kernel" in kernel:
        return "kda.recurrent_update", "high"
    if "chunk_gla_fwd_kernel_o" in kernel:
        return "kda.query_readout", "high"
    if "l2norm_fwd_kernel" in kernel:
        return "kda.qk_l2_norm", "high"
    if "layer_norm_gated" in kernel:
        return "kda.gated_rmsnorm", "high"

    # K3 fused MLA epilogues have one unambiguous semantic owner.  Source-order
    # reconciliation later records absorbed or jointly materialized children.
    if _has(
        kernel,
        "fusedkimik3mladecodeqconcat",
        "fused_kimi_k3_mla_decode_q_concat",
        "fusedkimik3mlakeyconcat",
        "fused_kimi_k3_mla_key_concat",
        "fusedkimik3mlaqkvquant",
    ):
        return "gated_mla.cache_update", "high"
    if in_mla and cpu == "aten::bmm":
        return "gated_mla.attention", "high"
    if in_mla and _has(kernel, "fmha", "flash_mla", "flashmla"):
        return "gated_mla.attention", "high"
    if _has(kernel, "fused_q_kv_rmsnorm", "fusedqkvrmsnorm"):
        return "gated_mla.q_norm", "high"
    if in_mla and _has(kernel, "sigmoid", "gate_sigmoid_mul"):
        return "gated_mla.gated_context", "high"

    # Stable LatentMoE routing and expert signatures.
    if in_moe:
        if "models/kimi_k3/nvidia/ops/latent_moe_tail.py" in names:
            return "stable_latent_moe.tp_routed_latent_collective", "high"
        if cpu == "record_param_comms" or _has(
            kernel, "allreduce", "all_reduce", "reduce_scatter"
        ):
            if "_maybe_reduce_final_output" in names or (
                "_overlap_allreduce_tail" in names
                and "allreduce_norm_latent_out" not in names
            ):
                return "stable_latent_moe.tp_shared_expert_collective", "high"
            return "stable_latent_moe.tp_routed_latent_collective", "high"
        if _has(
            kernel,
            "latent_moe_tail",
            "allreduce_rmsnorm_reduce_scatter_early_exit",
            "all_reduce_rms_norm",
        ):
            return "stable_latent_moe.tp_routed_latent_collective", "high"
        if "gatelinear" in names and (
            cpu == "aten::mm"
            or _has(kernel, "ll_bf16_dotprod", "ll_bf16_splitk")
        ):
            return "stable_latent_moe.router_logits", "high"
        if "model.py(806): <lambda>" in names and cpu == "aten::mm":
            return "stable_latent_moe.routed_down", "high"
        if "routed_expert_down_proj" in names and cpu == "aten::mm":
            return "stable_latent_moe.routed_down", "high"
        if "routed_expert_up_proj" in names and cpu == "aten::mm":
            return "stable_latent_moe.routed_up", "high"
        if cpu in {"aten::addmm", "aten::addmm_", "aten::mm"} and _has(
            names, "_shard_up_proj_tail", "_overlap_allreduce_tail"
        ):
            return "stable_latent_moe.routed_up", "high"
        if cpu == "vllm::mxfp8_quantize":
            return "stable_latent_moe.dispatch", "high"
        if cpu == "_c::rms_norm" and "allreduce_norm_latent_out" in names:
            return "stable_latent_moe.latent_norm", "high"
        if "shared_experts" in names and cpu == "aten::mm":
            if "rowparallellinear" in names or "down_proj" in names:
                return "stable_latent_moe.shared_down", "high"
            if "mergedcolumnparallellinear" in names or "gate_up_proj" in names:
                return "stable_latent_moe.shared_gate_up", "high"
        if _has(kernel, "route_", "routingindices"):
            return "stable_latent_moe.corrected_selection", "medium"
        if _has(kernel, "pack_topk", "per_token_group_quant"):
            return "stable_latent_moe.dispatch", "high"
        if kernel.startswith("bmm_") and "bfloat16" not in kernel:
            return "stable_latent_moe.expert_gate_up", "medium"
        if kernel.startswith("bmm_bfloat16"):
            return "stable_latent_moe.expert_down", "high"
        if "finalizekernel" in kernel:
            return "stable_latent_moe.weighted_reduce", "high"
        if "shared_experts" in names and ("situ" in kernel or "act_and_mul" in kernel):
            return "stable_latent_moe.shared_situ", "medium"
        if cpu == "aten::add":
            return "stable_latent_moe.combine", "high"

    # TP collectives remain attached to the exact enclosing semantic scope.
    if cpu == "record_param_comms" or _has(
        kernel, "allreduce", "all_reduce", "allgather", "all_gather"
    ):
        if "vocab_parallel_embedding" in names:
            return "top.tp_embedding_output_collective", "high"
        if "logits_processor" in names:
            return "top.tp_logits_materialization", "high"
        if in_kda:
            return "kda.tp_kda_output_collective", "high"
        if in_mla:
            return "gated_mla.tp_mla_output_collective", "high"
        if in_moe:
            return "stable_latent_moe.tp_routed_latent_collective", "medium"
        if in_dense:
            return "dense_mlp.tp_dense_output_collective", "high"

    # Projection classifications require both the exact K3 module scope and
    # the authored linear wrapper.  Generic GEMMs are intentionally omitted.
    if in_kda and cpu == "aten::mm":
        if "_kimigdnmergedcolumnparallellinear" in names:
            return "kda.qkv_projection", "high"
        if "rowparallellinear" in names:
            return "kda.output_projection", "high"
        if "f_b_proj" in names or "columnparallellinear" in names:
            return "kda.decay_projection", "medium"
    if in_mla and cpu == "aten::mm":
        if "rowparallellinear" in names:
            return "gated_mla.output_projection", "high"
        if "q_a" in names or "fusedqkv" in names:
            return "gated_mla.q_down", "medium"
        if "q_b_proj" in names:
            return "gated_mla.q_up", "high"
        if "kv_b_proj" in names:
            return "gated_mla.kv_up", "high"
        if "g_proj" in names:
            return "gated_mla.output_gate", "high"
        if "_forward_prefill_fused" in names:
            return "gated_mla.kv_up", "high"
        if "_apply_q_lora_attention" in names:
            return "gated_mla.q_up", "high"
        if "_forward_q_lora" in names:
            return "gated_mla.q_down", "high"
    if in_dense:
        if "situ" in kernel or "act_and_mul" in kernel:
            return "dense_mlp.situ", "high"
        if "mergedcolumnparallellinear" in names:
            return "dense_mlp.gate_up", "high"
        if "rowparallellinear" in names:
            return "dense_mlp.down", "high"

    if "logits_processor" in names and (
        cpu == "aten::mm" or "cute_dsl_skinny_gemm" in kernel
    ):
        return "top.lm_head", "high"
    if "logits_processor" in names:
        return "top.logits", "medium"
    if "vocab_parallel_embedding" in names:
        return "top.embedding", "high"
    if cpu == "_c::rms_norm" and "compute_logits" in names:
        return "top.final_norm", "high"

    # Exact KDA graph-off helpers that materialize/read/write recurrence state
    # are either owned by the recurrence or retained as explicit runtime
    # bookkeeping.  They are never left as an unexplained generic kernel.
    if in_kda and cpu == "aten::_index_put_impl_":
        return "kda.recurrent_update", "high"
    if in_kda and cpu in {
        "aten::fill_",
        "aten::copy_",
        "aten::sub",
        "aten::add",
        "aten::floor_divide",
        "aten::cumsum",
        "aten::cat",
    }:
        return "runtime.step_setup", "high"

    runtime_support_signatures = (
        "_apply_write_kernel",
        "_zero_kv_blocks_kernel",
        "_prepare_prefill_inputs_kernel",
        "_prepare_pos_seq_lens_kernel",
        "_combine_sampled_and_draft_tokens_kernel",
        "_gather_block_tables_kernel",
        "_compute_slot_mappings_kernel",
        "flashinfer::plan_kernel",
        "_bias_kernel",
        "_gumbel_sample_kernel",
        "_get_num_sampled_and_rejected_kernel",
        "_post_update_kernel",
        "_scatter_num_accepted_kernel",
    )
    if _has(kernel, *runtime_support_signatures):
        return "runtime.step_setup", "high"
    if not any((in_kda, in_mla, in_moe, in_dense)) and _has(
        names,
        "gpu/model_runner.py",
        "gpu_model_runner.py",
        "v1/attention/backend.py",
        "kda_metadata.py",
        "flashinfer/prefill.py",
    ):
        return "runtime.step_setup", "high"

    return None, "unmapped"


KIMI_K3_VLLM_TRACE_RULES = TraceMappingRules(
    model_id="kimi_k3_vllm",
    signature_kernel=ATTN_RES_SIGNATURE,
    signature_count_per_forward=ATTN_RES_CALLS_PER_FORWARD,
    stack=StackFrameRules(
        operator_patterns=(
            "model_executor/layers/linear.py",
            "model_executor/layers/fused_moe",
            "model_executor/layers/layernorm.py",
            "model_executor/layers/logits_processor.py",
            "model_executor/layers/vocab_parallel_embedding.py",
            "distributed/parallel_state.py",
        ),
        semantic_patterns=(
            "models/kimi_k3/nvidia/model.py",
            "models/kimi_k3/nvidia/kda.py",
            "models/kimi_k3/nvidia/mla.py",
            "models/kimi_k3/nvidia/latent_moe_runner.py",
            "models/kimi_k3/nvidia/ops/latent_moe_tail.py",
            "models/kimi_k3/nvidia/ops/attn_res.py",
            "model_executor/layers/fused_moe",
            "model_executor/layers/logits_processor.py",
            "model_executor/layers/vocab_parallel_embedding.py",
        ),
        model_context_patterns=(
            "KimiK3ForConditionalGeneration",
            "KimiLinearForCausalLM",
            "KimiLinearModel",
            "KimiDecoderLayer",
            "KimiK3DeltaAttention",
            "MultiHeadLatentAttention",
            "KimiMoE",
            "KimiMLP",
            "LatentMoERunner",
        ),
        phase_patterns=("model_runner.py",),
    ),
    classify_node=classify_kimi_k3_vllm_node,
    kernel_only_nodes=frozenset(
        {
            "runtime.step_setup",
            "attn_res.weighted_merge",
            "top.output_attn_res",
            "gated_mla.q_norm",
            "kda.q_short_conv",
            "kda.k_short_conv",
            "kda.v_short_conv",
            "kda.qk_l2_norm",
            "kda.recurrent_update",
            "kda.query_readout",
            "kda.gated_rmsnorm",
        }
    ),
)
