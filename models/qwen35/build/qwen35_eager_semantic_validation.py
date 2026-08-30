#!/usr/bin/env python3
"""Independent semantic-owner contracts for Qwen3.5 graph-off evidence.

Production CUDA Graph traces need collective-bounded sequence recovery because
they do not carry Python stacks.  A graph-off trace must not use that sequence
recovery as its semantic oracle, however.  This module validates every proposed
eager owner against the launch's Python module/source stack and an explicit
kernel/source contract.  The profile builder only copies a stack to a
production event after this independent owner agrees with the production
owner.

Some vLLM regions are TorchInductor-compiled and therefore expose the compiled
cache file plus ``Qwen3NextModel.forward`` rather than an individual Python
module frame.  Those rows use commit-specific compiled-source contracts below;
direct module frames always take precedence and a disagreement fails closed.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any


SGLANG_SOURCE = "f609d677b909ca46c64bb6803b69a85fedbf86bc"
VLLM_SOURCE = "487ecf187d3dfe74d2cf6119a92881dba403c219"


PUBLISHED_NODE_FAMILIES = frozenset(
    {
        "top.embedding",
        "top.tp_embedding_output_collective",
        "gdn_moe_block.input_norm",
        "full_attention_moe_block.input_norm",
        "gdn_attention.qkvz_projection",
        "gdn_attention.causal_conv",
        "gdn_attention.gated_delta_recurrence",
        "gdn_attention.output_gate_norm",
        "gdn_attention.output_projection",
        "full_attention.qkv_projection",
        "full_attention.qk_norm",
        "full_attention.kv_state_write",
        "full_attention.causal_gqa",
        "full_attention.attention_output_gate",
        "full_attention.output_projection",
        "gdn_moe_block.tp_attention_output_collective",
        "full_attention_moe_block.tp_attention_output_collective",
        "gdn_moe_block.post_attention_norm",
        "full_attention_moe_block.post_attention_norm",
        "moe_block.router",
        "moe_block.routed_experts",
        "moe_block.shared_expert",
        "moe_block.weighted_combine",
        "gdn_moe_block.tp_moe_output_collective",
        "full_attention_moe_block.tp_moe_output_collective",
        "top.final_norm",
        "top.lm_head",
        "top.tp_logits_all_gather",
    }
)


def _lower(row: dict[str, Any]) -> tuple[str, str, str]:
    kernel = str(row.get("kernel_name") or "").lower()
    cpu = str(row.get("cpu_op_name") or "").lower()
    stack = "\n".join(
        str(frame.get("raw") or "") for frame in row.get("python_stack") or []
    ).lower()
    return kernel, cpu, stack


def _has_any(value: str, *tokens: str) -> bool:
    return any(token in value for token in tokens)


def _is_all_reduce(kernel: str) -> bool:
    return _has_any(kernel, "allreduce", "all_reduce", "multimem_all_reduce")


def _is_all_gather(kernel: str) -> bool:
    return _has_any(kernel, "allgather", "all_gather")


def _is_norm(kernel: str, cpu: str) -> bool:
    return _has_any(kernel + "\n" + cpu, "rmsnorm", "rms_norm", "layer_norm")


def _is_gemm(kernel: str, cpu: str) -> bool:
    return _has_any(
        kernel + "\n" + cpu,
        "tst_",
        "qqtst_",
        "splitkreduce",
        "cutlass13device_kernel",
        "kernel_cutlass_kernel",
        "bmm_fp8",
        "aten::mm",
        "aten::matmul",
    )


def _strong_stack_owner(
    row: dict[str, Any], *, framework: str
) -> tuple[str | None, str | None]:
    """Return an owner proved directly by a module stack, if one exists."""

    kernel, cpu, stack = _lower(row)
    if framework == "vllm" and _has_any(
        stack, "sharedexperts_", "shared_experts.py", "qwen2moemlp_"
    ):
        return "moe_block.shared_expert", "shared_expert_module_stack"
    if _has_any(stack, "logitsprocessor_", "logits_processor.py"):
        if _is_all_gather(kernel):
            return "top.tp_logits_all_gather", "logits_processor_all_gather_stack"
        if _is_gemm(kernel, cpu):
            return "top.lm_head", "logits_processor_head_gemm_stack"
        return None, "logits_processor_support_stack"
    if _has_any(stack, "vocabparallelembedding", "vocab_parallel_embedding.py"):
        if _is_all_reduce(kernel):
            return (
                "top.tp_embedding_output_collective",
                "embedding_collective_stack",
            )
        if "embedding" in kernel:
            return "top.embedding", "embedding_lookup_stack"
    return None, None


def _source_scope_ok(
    row: dict[str, Any], *, framework: str, node: str
) -> tuple[bool, list[str]]:
    _kernel, _cpu, stack = _lower(row)
    frames = [
        str(frame.get("raw") or "")
        for frame in row.get("python_stack") or []
        if str(frame.get("raw") or "")
    ]
    if framework == "vllm":
        allowed = (
            "vllm/model_executor/models/qwen3_next.py",
            "vllm/model_executor/models/qwen3_5.py",
            "vllm/model_executor/layers/fused_moe",
            "vllm/model_executor/layers/logits_processor.py",
            "vllm/model_executor/layers/vocab_parallel_embedding.py",
            "/root/.cache/vllm/torch_compile_cache/",
        )
    else:
        allowed = (
            "sglang/srt/models/qwen3_5.py",
            "sglang/srt/models/qwen2_moe.py",
            "sglang/srt/layers/",
            "sglang/srt/model_executor/",
            "sglang/srt/managers/",
            "qwen3_5gateddeltanet",
            "qwen3_5attentiondecoderlayer",
            "qwen2moe",
        )
    matched = [frame for frame in frames if any(token in frame.lower() for token in allowed)]
    # Top-level direct module stacks are stronger than the generic compiled
    # model frame and are accepted even if path spelling differs by package.
    if node.startswith("top.") and _has_any(
        stack, "logitsprocessor", "vocabparallelembedding", "qwen3_5", "qwen3next"
    ):
        matched = matched or frames[:1]
    return bool(matched), matched[:8]


def _node_contract_matches(row: dict[str, Any], node: str) -> bool:
    """Commit-specific kernel/source anchor for every published node family."""

    kernel, cpu, _stack = _lower(row)
    substage = row.get("substage")
    kind = row.get("layer_kind")
    occurrence = str(row.get("occurrence_id") or "top")

    if node == "top.embedding":
        return "embedding" in kernel
    if node == "top.tp_embedding_output_collective":
        return _is_all_reduce(kernel) and occurrence == "top"
    if node in {"gdn_moe_block.input_norm", "full_attention_moe_block.input_norm"}:
        expected_kind = "gdn" if node.startswith("gdn_") else "full"
        return _is_norm(kernel, cpu) and substage == "attention" and kind == expected_kind
    if node == "gdn_attention.qkvz_projection":
        return substage == "attention" and kind == "gdn" and _has_any(
            kernel + "\n" + cpu,
            "static_quant",
            "qqtst_",
            "tst_",
            "splitkreduce",
            "fused_qkvzba_split",
            "cutlass",
            "triton_poi_fused_2",
            "triton_poi_fused_3",
            "triton_poi_fused_4",
            "triton_poi_fused_6",
            "to_copy_clamp_mul_reciprocal_2",
            "to_copy_clamp_mul_reciprocal_3",
        )
    if node == "gdn_attention.causal_conv":
        return substage == "attention" and kind == "gdn" and "causal_conv1d" in kernel
    if node == "gdn_attention.gated_delta_recurrence":
        return substage == "attention" and kind == "gdn" and _has_any(
            kernel + "\n" + cpu,
            "gated_delta",
            "gdn_",
            "l2norm_fwd",
            "fused_post_conv",
            "fused_qkv_split_gdn",
            "fused_gdn_gating",
            "direct_copy_kernel",
            "index_elementwise",
            "bitwise_not",
            "masked_fill",
            "exp_kernel",
            "launch_clamp_scalar",
            "triton_poi_fused_0",
        )
    if node == "gdn_attention.output_gate_norm":
        return substage == "attention" and kind == "gdn" and _has_any(
            kernel + "\n" + cpu,
            "layer_norm",
            "rmsnorm",
            "rms_norm",
            "triton_per_fused",
            "rsqrt_silu",
            "index_copy_elementwise",
            "index_copy_kernel_impl",
        )
    if node == "gdn_attention.output_projection":
        return substage == "attention" and kind == "gdn" and _has_any(
            kernel + "\n" + cpu,
            "static_quant",
            "qqtst_",
            "cutlass13device_kernel",
            "bmm_fp8",
        )
    if node == "full_attention.qkv_projection":
        return substage == "attention" and kind == "full" and _has_any(
            kernel + "\n" + cpu,
            "static_quant",
            "qqtst_",
            "splitkreduce",
            "cutlass13device_kernel",
            "bmm_fp8",
            "rms_norm",
            "rmsnorm",
            "to_copy_clamp_mul_reciprocal_4",
        )
    if node == "full_attention.qk_norm":
        return substage == "attention" and kind == "full" and _has_any(
            kernel + "\n" + cpu,
            "rmsnorm",
            "rms_norm",
            "layer_norm",
            "triton_poi_fused_5",
            "triton_poi_fused_6",
            "triton_poi_fused_7",
            "triton_poi_fused_8",
            "triton_red_fused_6",
            "triton_red_fused_7",
        )
    if node == "full_attention.kv_state_write":
        return substage == "attention" and kind == "full" and _has_any(
            kernel + "\n" + cpu,
            "reshape_and_cache",
            "qkv_kv_cache",
            "split_with_sizes",
            "fillfunctor",
        )
    if node == "full_attention.causal_gqa":
        return substage == "attention" and kind == "full" and _has_any(
            kernel + "\n" + cpu, "fmha", "attention"
        )
    if node == "full_attention.attention_output_gate":
        return substage == "attention" and kind == "full" and _has_any(
            kernel + "\n" + cpu, "sigmoid", "mul", "triton_poi_fused_0"
        )
    if node == "full_attention.output_projection":
        return substage == "attention" and kind == "full" and _has_any(
            kernel + "\n" + cpu,
            "static_quant",
            "qqtst_",
            "cutlass13device_kernel",
            "bmm_fp8",
        )
    if node.endswith("tp_attention_output_collective"):
        expected_kind = "gdn" if node.startswith("gdn_") else "full"
        return _is_all_reduce(kernel) and substage == "attention" and kind == expected_kind
    if node in {"gdn_moe_block.post_attention_norm", "full_attention_moe_block.post_attention_norm"}:
        expected_kind = "gdn" if node.startswith("gdn_") else "full"
        return _is_norm(kernel, cpu) and substage == "moe" and kind == expected_kind
    if node == "moe_block.router":
        return substage == "moe" and _has_any(
            kernel + "\n" + cpu, "routing", "topk", "tst_", "splitkreduce"
        ) and "qqtst_" not in kernel
    if node == "moe_block.routed_experts":
        return substage == "moe" and _has_any(
            kernel + "\n" + cpu,
            "cvt_fp16_to_fp4",
            "scaled_fp4_quant",
            "nvfp4_quantize",
            "bmm_e2m1",
            "bmm_bfloat16",
            "finalizekernel",
            "deep_gemm",
            "grouped",
        )
    if node == "moe_block.shared_expert":
        return substage == "moe" and _has_any(
            kernel + "\n" + cpu,
            "static_quant",
            "qqtst_",
            "splitkreduce",
            "act_and_mul",
            "cutlass13device_kernel",
            "bmm_fp8",
            "tst_",
            "dot_kernel",
            "reduce_1block_kernel",
            "sigmoid",
            "mulfunctor",
            "direct_copy_kernel",
            "index_elementwise",
            "vectorized_gather_kernel",
            "rms_norm",
            "to_copy_clamp_mul_reciprocal",
            "mul_silu_slice",
        )
    if node == "moe_block.weighted_combine":
        return substage == "moe" and _has_any(
            kernel + "\n" + cpu, "triton_poi_fused_add", "sigmoid_mul"
        )
    if node.endswith("tp_moe_output_collective"):
        expected_kind = "gdn" if node.startswith("gdn_") else "full"
        return _is_all_reduce(kernel) and substage == "moe" and kind == expected_kind
    if node == "top.final_norm":
        return occurrence == "top" and _is_norm(kernel, cpu)
    if node == "top.lm_head":
        return occurrence == "top" and _is_gemm(kernel, cpu)
    if node == "top.tp_logits_all_gather":
        return occurrence == "top" and _is_all_gather(kernel)
    raise ValueError(f"Qwen3.5 eager owner has no explicit anchor contract: {node}")


def _shared_expert_stack_layer(stack: str) -> int | None:
    matches = {
        int(value)
        for value in re.findall(
            r"(?:sharedexperts|qwen2moemlp|siluandmul)_(\d+)", stack
        )
    }
    if len(matches) > 1:
        raise ValueError(f"inconsistent shared-expert layer module stack: {sorted(matches)}")
    return next(iter(matches)) if matches else None


def validate_eager_semantic_attribution(
    rows: list[dict[str, Any]], *, framework: str, phase: str
) -> dict[str, Any]:
    """Validate and stamp independently anchored owners on eager rows."""

    if framework not in {"sglang", "vllm"}:
        raise ValueError(f"unsupported Qwen3.5 eager framework {framework!r}")
    source_commit = SGLANG_SOURCE if framework == "sglang" else VLLM_SOURCE
    owner_counts: Counter[str] = Counter()
    basis_counts: Counter[str] = Counter()
    observed_nodes: set[str] = set()
    final_norm_target_count = 0
    for row in rows:
        stack_frames = row.get("python_stack") or []
        if not stack_frames:
            raise ValueError(f"{row.get('event_id')}: eager kernel has no Python stack")
        node = row.get("node")
        if not node:
            if not row.get("support_class"):
                raise ValueError(f"{row.get('event_id')}: eager kernel lacks owner/support class")
            row["runtime_support_evidence"] = {
                "basis": "python_stack_plus_explicit_support_contract",
                "support_class": row["support_class"],
                "source_commit": source_commit,
                "phase": phase,
            }
            continue
        node = str(node)
        observed_nodes.add(node)
        if node not in PUBLISHED_NODE_FAMILIES:
            raise ValueError(f"{row.get('event_id')}: unpublished eager node family {node}")

        direct_owner, direct_anchor = _strong_stack_owner(
            row, framework=framework
        )
        if direct_owner is not None and direct_owner != node:
            raise ValueError(
                f"{row.get('event_id')}: eager Python-stack owner {direct_owner} "
                f"disagrees with proposed owner {node} ({direct_anchor})"
            )
        if not _node_contract_matches(row, node):
            raise ValueError(
                f"{row.get('event_id')}: {node} fails its explicit eager "
                f"kernel/source anchor contract ({row.get('kernel_name')})"
            )
        source_ok, matched_frames = _source_scope_ok(
            row, framework=framework, node=node
        )
        if not source_ok:
            raise ValueError(
                f"{row.get('event_id')}: {node} lacks a pinned model/source stack anchor"
            )

        _kernel, _cpu, stack = _lower(row)
        if node == "top.final_norm" and _has_any(
            stack, "sharedexperts", "qwen2moemlp", "siluandmul"
        ):
            raise ValueError(
                f"{row.get('event_id')}: final norm stack contains shared-expert frames"
            )
        if node == "top.lm_head" and not _has_any(
            stack, "logitsprocessor", "logits_processor.py", "lm_head"
        ) and not (
            "compute_logits" in stack
            and _has_any(stack, "layers/linear.py", "default_unquantized_gemm")
        ):
            raise ValueError(
                f"{row.get('event_id')}: LM head lacks a logits-processor/head stack anchor"
            )
        if node == "moe_block.shared_expert" and framework == "vllm":
            stack_layer = _shared_expert_stack_layer(stack)
            if stack_layer is not None and stack_layer != row.get("layer_id"):
                raise ValueError(
                    f"{row.get('event_id')}: shared-expert stack layer {stack_layer} "
                    f"!= occurrence layer {row.get('layer_id')}"
                )

        basis = (
            "direct_python_module_stack"
            if direct_owner == node
            else "compiled_source_kernel_occurrence_contract"
        )
        evidence = {
            "owner": node,
            "anchor_id": f"qwen35-{framework}-{node}-owner-v1",
            "basis": basis,
            "source_commit": source_commit,
            "phase": phase,
            "rank": row.get("rank"),
            "occurrence_id": row.get("occurrence_id") or "top",
            "kernel_name": row.get("kernel_name"),
            "cpu_op_name": row.get("cpu_op_name"),
            "matched_stack_frames": matched_frames,
            "direct_stack_anchor": direct_anchor,
            "forbidden_shared_expert_final_norm_stack_checked": node
            == "top.final_norm",
        }
        row["semantic_owner_evidence"] = evidence
        owner_counts[node] += 1
        basis_counts[basis] += 1

        if "top.final_norm" in (row.get("ir_targets") or []) and node.endswith(
            "tp_moe_output_collective"
        ):
            if not (
                framework == "vllm"
                and phase == "decode"
                and row.get("layer_id") == 59
                and _is_all_reduce(_lower(row)[0])
            ):
                raise ValueError(
                    f"{row.get('event_id')}: invalid final-norm fusion target contract"
                )
            row.setdefault("semantic_fused_target_evidence", {})[
                "top.final_norm"
            ] = {
                "anchor_id": "qwen35-vllm-decode-final-norm-fused-last-tp-moe-ar-v1",
                "basis": "pinned_model_source_plus_compiler_allreduce_rms_fusion_contract",
                "model_source": "vllm/model_executor/models/qwen3_next.py:695",
                "compiler_source": "vllm/compilation/passes/fusion/allreduce_rms_fusion.py",
                "source_commit": source_commit,
                "physical_owner": node,
                "copied_timing": False,
            }
            final_norm_target_count += 1

    return {
        "framework": framework,
        "phase": phase,
        "validated_semantic_event_count": sum(owner_counts.values()),
        "semantic_owner_counts": dict(sorted(owner_counts.items())),
        "semantic_owner_basis_counts": dict(sorted(basis_counts.items())),
        "published_node_family_count": len(observed_nodes),
        "published_node_families": sorted(observed_nodes),
        "unanchored_semantic_event_count": 0,
        "owner_disagreement_count": 0,
        "final_norm_fused_target_event_count": final_norm_target_count,
    }
