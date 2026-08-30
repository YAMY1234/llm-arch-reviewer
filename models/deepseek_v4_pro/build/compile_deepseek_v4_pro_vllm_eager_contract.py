#!/usr/bin/env python3
"""Compile one eager mapping into exact DeepSeek-V4-Pro occurrences.

The low-level mapper deliberately emits shared ``attention.*`` and
``compressor.*`` labels where CUDA symbols do not encode the official layer
kind.  This compiler admits those events only inside a complete 61-layer
schedule bounded by the attention and MoE TP collectives.  It also types every
runtime-support event, preserves the rank-specific ordered fingerprint, and
fails closed on schedule, branch, source, or coverage discrepancies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


CSA_LAYERS = tuple(range(2, 61, 2))
HCA_LAYERS = tuple(layer for layer in range(61) if layer not in CSA_LAYERS)
LAYER_COUNT = 61


def load_json(path: Path) -> Any:
    with path.open() as source:
        return json.load(source)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as source:
        return [json.loads(line) for line in source if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True) + "\n")


def _frame_text(row: dict[str, Any]) -> str:
    return "\n".join(
        str((row.get(key) or {}).get("raw") or "")
        for key in (
            "primitive_frame",
            "operator_frame",
            "semantic_frame",
            "model_context_frame",
            "phase_frame",
        )
    ).lower()


def _normalize_direct_evidence(row: dict[str, Any]) -> None:
    """Close exact stack-resolvable leaves before schedule expansion."""

    node = str(row.get("selected_node") or "")
    frames = _frame_text(row)
    cpu = str(row.get("cpu_op_name") or "").lower()
    kernel = str(row.get("kernel_name") or "").lower()
    if node == "top.tp_logits_all_gather":
        row.update(
            {
                "selected_node": "top.tp_logits_collective",
                "mapping_method": "normalized_tp_logits_collective_contract",
                "confidence": "high",
            }
        )
    elif "mhc_fused_tilelang_kernel" in kernel:
        row.update(
            {
                "selected_node": "mhc_transform.mix",
                "mapping_method": "exact_mhc_fused_post_pre_kernel",
                "confidence": "high",
            }
        )
    elif (
        "cooperative_topk_cs16" in kernel or cpu == "_c::cooperative_topk"
    ):
        row.update(
            {
                "selected_node": "csa_indexer.causal_topk",
                "mapping_method": "exact_csa_cooperative_topk_kernel",
                "confidence": "high",
            }
        )
    elif (
        "router_gemm_kernel_float_output" in kernel
        or cpu == "_moe_c::dsv3_router_gemm"
    ):
        row.update(
            {
                "selected_node": "moe.score_projection",
                "mapping_method": "exact_router_projection_kernel",
                "confidence": "high",
            }
        )
    elif (
        "dequantgatherkcachekernel" in kernel
        or "dequant_gather_k" in kernel
        or (cpu == "aten::fill_" and "combine_topk_swa_indices" in frames)
        or (cpu == "aten::floor_divide" and "flashmla.py" in frames)
    ):
        row.update(
            {
                "selected_node": "attention.index_union",
                "mapping_method": "source_scoped_prefill_sparse_index_union",
                "confidence": "high",
            }
        )
    elif cpu == "_c_cache_ops::cp_gather_indexer_k_quant_cache":
        row.update(
            {
                "selected_node": "csa_indexer.score",
                "mapping_method": "exact_indexer_quant_cache_gather",
                "confidence": "high",
            }
        )
    elif node == "attention.q_b" and "deepseekv4indexer_" in frames:
        row.update(
            {
                "selected_node": "csa_indexer.q_projection",
                "mapping_method": "exact_indexer_module_scope",
                "confidence": "high",
            }
        )
    elif node == "csa_indexer.selected_ids":
        row.update(
            {
                "selected_node": "csa_indexer.causal_topk",
                "mapping_method": "topk_output_buffer_initialization",
                "confidence": "high",
            }
        )
    elif (
        node == "top.runtime_support"
        and "deepseekv4model_" in frames
        and cpu == "aten::copy_"
    ):
        row.update(
            {
                "selected_node": "top.hc_expand",
                "mapping_method": "exact_model_forward_hc_initialization",
                "confidence": "high",
            }
        )
    elif (
        node == "top.runtime_support"
        and "base_device_communicator.py" in frames
        and "all_gather" in frames
    ):
        row.update(
            {
                "selected_node": "top.tp_logits_collective",
                "mapping_method": "collective_result_materialization",
                "confidence": "high",
            }
        )


def _runtime_support(row: dict[str, Any]) -> None:
    if row.get("selected_node") != "top.runtime_support":
        return
    text = f"{row.get('kernel_name') or ''}\n{row.get('cpu_op_name') or ''}\n{_frame_text(row)}".lower()
    if "cuda_graph_buffer_registry.py" in text or "load_batch" in text:
        support_class = "eager_input_buffer_copy"
        reason = "formal-batch tensors copied into the eager runner input registry"
    elif any(
        token in text
        for token in (
            "block_table",
            "slot_mapping",
            "mem_cache/allocator",
            "translate_loc_from_full_to_swa",
        )
    ):
        support_class = "cache_slot_metadata"
        reason = "request-to-KV-cache slot and block-table address preparation"
    elif any(
        token in text
        for token in (
            "_build_attn_group_metadata",
            "sparse_mla.py",
            "compressor_utils.py",
            "attention/backends/mla/indexer.py",
            "sparse_swa.py",
            "get_mla_metadata_kernel",
            "deepseek_v4_backend.py",
            "dsv4_attn_metadata_kernels.py",
            "metadata_kernel.py",
            "paged_mqa_metadata",
            "topk_plan",
            "get_paged_mqa_logits_metadata",
        )
    ):
        support_class = "attention_plan_metadata"
        reason = "shape, compressed-slot, sparse-index, or FlashMLA launch metadata; no model value is produced"
    elif "sparse_prefill_utils.py" in text:
        support_class = "sparse_prefill_cache_metadata"
        reason = "one-time sparse-prefill workspace, cache-address, and union-index metadata construction"
    elif "_get_nonpaged_indexer_plan" in text:
        support_class = "indexer_plan_metadata"
        reason = "one-time non-paged indexer schedule and score-buffer metadata construction"
    elif any(
        token in text
        for token in (
            "compressor.py(101): build",
            "create_paged_compressor_data",
            "plan_compress_decode_kernel",
        )
    ):
        support_class = "compressor_plan_metadata"
        reason = "compressor scheduler bounds and slot metadata; no compressed model value is produced"
    elif "_prepare_inputs" in text:
        support_class = "request_batch_metadata"
        reason = "positions, request indices, and realized batch metadata preparation"
    elif any(token in text for token in ("sampl", "execute_model", "output")):
        support_class = "sampling_and_output"
        reason = "token-selection or output materialization outside the stable model graph"
    elif "request_receiver.py" in text or "broadcast_pyobj" in text:
        support_class = "request_broadcast_overlap"
        reason = "asynchronous request broadcast launch correlated inside the forward window"
    elif "scheduler.py" in text or "schedule_batch.py" in text:
        support_class = "scheduler_overlap"
        reason = "asynchronous next-step scheduler work correlated inside the forward window"
    else:
        support_class = "framework_runtime_metadata"
        reason = "typed framework bookkeeping outside the stable Model-IR value flow"
    row.update(
        {
            "support_class": support_class,
            "support_reason": reason,
            "mapping_method": "explicit_runtime_support_classification",
            "confidence": "support",
        }
    )


def _indices(rows: list[dict[str, Any]], node: str) -> list[int]:
    return [index for index, row in enumerate(rows) if row.get("selected_node") == node]


def _recover_ordered_sglang_collectives(
    rows: list[dict[str, Any]], *, source_commit: str
) -> None:
    """Recover async all-reduce launches from their exact semantic neighbors."""

    if source_commit != "71de97b264b04dcd514cf904003028aefe9775c8":
        return
    for index in range(1, len(rows) - 1):
        row = rows[index]
        kernel = str(row.get("kernel_name") or "").lower()
        if (
            row.get("selected_node") != "top.runtime_support"
            or "allreduce" not in kernel
            or rows[index + 1].get("selected_node") != "mhc_transform.mix"
        ):
            continue
        previous = str(rows[index - 1].get("selected_node") or "")
        if previous == "attention.o_b":
            owner = "attention.tp_output_collective"
            method = "ordered_attention_output_all_reduce_boundary"
        elif previous == "moe.combine":
            owner = "moe.tp_moe_output_collective"
            method = "ordered_moe_output_all_reduce_boundary"
        else:
            continue
        row.update(
            {
                "selected_node": owner,
                "mapping_method": method,
                "confidence": "high",
            }
        )


def _recover_ordered_sglang_compute(
    rows: list[dict[str, Any]], *, source_commit: str
) -> None:
    """Recover async GEMM launches only inside exact proved neighbors."""

    if source_commit != "71de97b264b04dcd514cf904003028aefe9775c8":
        return
    for index in range(1, len(rows) - 1):
        row = rows[index]
        if (
            row.get("selected_node") != "top.runtime_support"
            or row.get("cpu_op_name") != "aten::mm"
        ):
            continue
        previous = str(rows[index - 1].get("selected_node") or "")
        following = str(rows[index + 1].get("selected_node") or "")
        if (
            previous == "attention.window_kv"
            and following == "hca_compressor.softmax_pool"
        ):
            row.update(
                {
                    "selected_node": "compressor.kv_gate_projection",
                    "mapping_method": "ordered_hca_compressor_projection",
                    "confidence": "high",
                }
            )


def _expand_branch_node(node: str, kind: str) -> str:
    if node.startswith("attention."):
        suffix = node.split(".", 1)[1]
        if suffix == "tp_output_collective":
            return f"{kind}_attention.tp_{kind}_output_collective"
        return f"{kind}_attention.{suffix}"
    if node == "compressor.kv_gate_projection":
        return f"{kind}_compressor.kv_gate_projection"
    if node == "compressor.partial_state":
        return f"{kind}_compressor.partial_state"
    if node.startswith("mhc_transform."):
        return node
    return node


MHC_PRE_FUSED_MEMBERS = (
    "mhc_transform.flatten_rms",
    "mhc_transform.affine",
    "mhc_transform.pre_gate",
    "mhc_transform.post_gate",
    "mhc_transform.combine_sinkhorn",
    "mhc_transform.read",
)


def _annotate_schedule(
    rows: list[dict[str, Any]], *, source_commit: str | None = None
) -> list[dict[str, Any]]:
    attention_collectives = _indices(rows, "attention.tp_output_collective")
    moe_collectives = _indices(rows, "moe.tp_moe_output_collective")
    mixes = _indices(rows, "mhc_transform.mix")
    affines = _indices(rows, "mhc_transform.affine")
    expected = {
        "attention TP collectives": (len(attention_collectives), LAYER_COUNT),
        "MoE TP collectives": (len(moe_collectives), LAYER_COUNT),
    }
    errors = [
        f"{label}: expected {want}, got {got}"
        for label, (got, want) in expected.items()
        if got != want
    ]
    if errors:
        raise ValueError("incomplete 61-layer schedule: " + "; ".join(errors))

    separated_mhc = len(mixes) == 2 * LAYER_COUNT and len(affines) == 4 * LAYER_COUNT
    fused_launch_mhc = (
        len(mixes) == 2 * LAYER_COUNT and len(affines) == 2 * LAYER_COUNT + 1
    )
    legacy_unclassified_fused_mhc = (
        len(mixes) == 1 and len(affines) == 2 * LAYER_COUNT + 1
    )
    if not (separated_mhc or fused_launch_mhc or legacy_unclassified_fused_mhc):
        raise ValueError(
            "unrecognized mHC implementation schedule: "
            f"mixes={len(mixes)} affines={len(affines)}"
        )

    start = min(affines)
    for layer_id in range(LAYER_COUNT):
        attention_collective = attention_collectives[layer_id]
        moe_collective = moe_collectives[layer_id]
        if separated_mhc or fused_launch_mhc:
            attention_end = next(index for index in mixes if index > attention_collective)
            ffn_end = next(index for index in mixes if index > moe_collective)
        else:
            # Small decode shapes use mhc_fused_post_pre_tilelang.  The
            # physical symbol is the same pre kernel, but the source contract
            # proves it owns the preceding post-mix and fuses the following
            # sublayer pre-transform plus RMSNorm.  Keep the duration only on
            # the preceding mix and record the cross-occurrence non-owners.
            attention_end = next(
                index for index in affines if index > attention_collective
            )
            ffn_end = (
                next(index for index in affines if index > moe_collective)
                if layer_id + 1 < LAYER_COUNT
                else next(index for index in mixes if index > moe_collective)
            )
            rows[attention_end].update(
                {
                    "selected_node": "mhc_transform.mix",
                    "mapping_method": "source_proved_fused_attention_post_ffn_pre",
                    "timing_role": "fusion_owner",
                    "fused_semantic_nodes": [
                        "mhc_transform.mix",
                        *MHC_PRE_FUSED_MEMBERS,
                        "decoder_stack.ffn_norm",
                    ],
                    "fused_nonowner_occurrence_id": f"layer_{layer_id:02d}.feed_forward",
                    "confidence": "high",
                }
            )
            if layer_id + 1 < LAYER_COUNT:
                rows[ffn_end].update(
                    {
                        "selected_node": "mhc_transform.mix",
                        "mapping_method": "source_proved_fused_ffn_post_next_attention_pre",
                        "timing_role": "fusion_owner",
                        "fused_semantic_nodes": [
                            "mhc_transform.mix",
                            *MHC_PRE_FUSED_MEMBERS,
                            "decoder_stack.attention_norm",
                        ],
                        "fused_nonowner_occurrence_id": f"layer_{layer_id + 1:02d}.attention",
                        "confidence": "high",
                    }
                )
        if not (start < attention_collective < attention_end < moe_collective < ffn_end):
            raise ValueError(f"layer {layer_id} has crossed semantic/collective boundaries")
        attention = rows[start : attention_end + 1]
        ffn = rows[attention_end + 1 : ffn_end + 1]
        if separated_mhc:
            if sum(row.get("selected_node") == "mhc_transform.affine" for row in attention) != 2:
                raise ValueError(f"layer {layer_id} attention lacks exact two-kernel mHC-pre group")
            if sum(row.get("selected_node") == "mhc_transform.affine" for row in ffn) != 2:
                raise ValueError(f"layer {layer_id} FFN lacks exact two-kernel mHC-pre group")
        elif fused_launch_mhc:
            expected_attention_affines = 2 if layer_id == 0 else 1
            if sum(
                row.get("selected_node") == "mhc_transform.affine"
                for row in attention
            ) != expected_attention_affines:
                raise ValueError(
                    f"layer {layer_id} attention has an invalid fused-launch mHC-pre group"
                )
            if sum(
                row.get("selected_node") == "mhc_transform.affine" for row in ffn
            ) != 1:
                raise ValueError(
                    f"layer {layer_id} FFN has an invalid fused-launch mHC-pre group"
                )
            for substage, segment, previous_occurrence in (
                ("attention", attention, None if layer_id == 0 else f"layer_{layer_id - 1:02d}.feed_forward"),
                ("feed_forward", ffn, f"layer_{layer_id:02d}.attention"),
            ):
                affine_rows = [
                    row
                    for row in segment
                    if row.get("selected_node") == "mhc_transform.affine"
                ]
                launch_affine = affine_rows[-1]
                if previous_occurrence is not None:
                    launch_affine.update(
                        {
                            "launch_group_id": f"{previous_occurrence}__to__layer_{layer_id:02d}.{substage}",
                            "launch_group_role": "post_pre_second_kernel",
                            "source_call": "mhc_fused_post_pre_tilelang",
                        }
                    )
                    prior_mix = next(
                        row
                        for row in reversed(rows[: rows.index(launch_affine)])
                        if row.get("selected_node") == "mhc_transform.mix"
                    )
                    prior_mix.update(
                        {
                            "launch_group_id": launch_affine["launch_group_id"],
                            "launch_group_role": "post_pre_first_kernel",
                            "source_call": "mhc_fused_post_pre_tilelang",
                        }
                    )
        elif layer_id == 0 and sum(
            row.get("selected_node") == "mhc_transform.affine" for row in attention
        ) != 2:
            raise ValueError("first layer lacks the exact standalone mHC-pre group")

        kind = "csa" if layer_id in CSA_LAYERS else "hca"
        wrong_prefix = "hca_" if kind == "csa" else "csa_"
        direct_branch_nodes = [
            str(row.get("selected_node") or "")
            for row in attention
            if str(row.get("selected_node") or "").startswith(("csa_", "hca_"))
        ]
        if any(node.startswith(wrong_prefix) for node in direct_branch_nodes):
            raise ValueError(f"layer {layer_id} contains {wrong_prefix.rstrip('_')} evidence")
        if kind == "csa" and not any(
            node.startswith("csa_indexer.") for node in direct_branch_nodes
        ):
            raise ValueError(f"CSA layer {layer_id} lacks direct indexer evidence")
        if kind == "hca" and not any(
            node.startswith("hca_compressor.") for node in direct_branch_nodes
        ):
            raise ValueError(f"HCA layer {layer_id} lacks direct compressor evidence")

        partial_states = [
            row
            for row in attention
            if row.get("selected_node") == "compressor.partial_state"
        ]
        if kind == "csa":
            if len(partial_states) == 2:
                partial_states[0].update(
                    {
                        "selected_node": "csa_indexer.k_compress",
                        "mapping_method": "source_order_indexer_compressor_state_write",
                        "confidence": "high",
                    }
                )
                partial_states[1].update(
                    {
                        "selected_node": "csa_compressor.partial_state",
                        "mapping_method": "source_order_main_compressor_state_write",
                        "confidence": "high",
                    }
                )
            elif len(partial_states) == 1 and sum(
                row.get("selected_node") == "csa_indexer.k_compress"
                for row in attention
            ) >= 3:
                partial_states[0].update(
                    {
                        "selected_node": "csa_compressor.partial_state",
                        "mapping_method": "source_proved_sglang_main_compressor_state_write",
                        "confidence": "high",
                    }
                )
            else:
                raise ValueError(
                    f"CSA layer {layer_id} requires indexer and main-compressor state writes"
                )
        elif len(partial_states) != 1:
            raise ValueError(
                f"HCA layer {layer_id} requires one main-compressor state write"
            )

        for substage, segment in (("attention", attention), ("feed_forward", ffn)):
            occurrence_id = f"layer_{layer_id:02d}.{substage}"
            for row in segment:
                row.update(
                    {
                        "layer_id": layer_id,
                        "layer_kind": kind,
                        "substage": substage,
                        "occurrence_id": occurrence_id,
                    }
                )
                if substage == "attention":
                    row["selected_node"] = _expand_branch_node(
                        str(row.get("selected_node") or ""), kind
                    )
                    row.setdefault(
                        "mapping_method",
                        "official_layer_schedule_inside_collective_bounded_occurrence",
                    )
        shared = [
            row for row in ffn if row.get("selected_node") == "moe.shared_gate_up"
        ]
        activations = [
            row for row in ffn if row.get("selected_node") == "moe.shared_activation"
        ]
        if source_commit == "71de97b264b04dcd514cf904003028aefe9775c8" and (
            shared or activations
        ):
            if len(shared) != 4 or len(activations) != 1:
                raise ValueError(
                    f"layer {layer_id} has invalid SGLang shared-expert launch group: "
                    f"linear={len(shared)} activation={len(activations)}"
                )
            activation_index = rows.index(activations[0])
            for row in shared:
                if rows.index(row) > activation_index:
                    row.update(
                        {
                            "selected_node": "moe.shared_down",
                            "mapping_method": "ordered_shared_expert_projection_after_swiglu",
                            "confidence": "high",
                        }
                    )
        start = ffn_end + 1
    for row in rows:
        if row.get("selected_node") == "mhc_transform.affine":
            row.setdefault("timing_role", "fusion_owner")
            row.setdefault("fused_semantic_nodes", list(MHC_PRE_FUSED_MEMBERS))
        node = row.get("selected_node")
        if node == "csa_indexer.q_projection":
            row.setdefault(
                "fused_semantic_nodes",
                ["csa_indexer.q_projection", "csa_indexer.q_rope_rotate"],
            )
        elif node == "attention.q_head_norm":
            row.setdefault(
                "fused_semantic_nodes",
                ["attention.q_head_norm", "attention.q_rope"],
            )
        elif node == "attention.window_kv":
            row.setdefault(
                "fused_semantic_nodes",
                ["attention.window_kv", "attention.window_cache"],
            )
        elif node in {"csa_compressor.partial_state", "hca_compressor.partial_state"}:
            prefix = str(node).split(".", 1)[0]
            row.setdefault(
                "fused_semantic_nodes",
                [
                    f"{prefix}.norm_rope",
                    f"{prefix}.partial_state",
                    f"{prefix}.compressed_cache",
                ],
            )
        elif node == "csa_indexer.k_compress":
            row.setdefault(
                "fused_semantic_nodes",
                ["csa_indexer.k_compress", "csa_indexer.selected_ids"],
            )
        elif node == "moe.routed_gate_up":
            row.setdefault(
                "fused_semantic_nodes",
                ["moe.routed_gate_up", "moe.routed_activation"],
            )
        elif node in {"moe.hash_select", "moe.learned_select"}:
            row.setdefault(
                "fused_semantic_nodes",
                ["moe.sqrt_softplus", str(node), "moe.weights"],
            )
    return rows


def _ordered_fingerprint(rows: list[dict[str, Any]]) -> str:
    payload = [
        {
            "kernel_name": row.get("kernel_name"),
            "selected_node": row.get("selected_node"),
            "occurrence_id": row.get("occurrence_id"),
            "cpu_op_name": row.get("cpu_op_name"),
            "semantic_frame": (row.get("semantic_frame") or {}).get("raw"),
            "model_context_frame": (row.get("model_context_frame") or {}).get("raw"),
        }
        for row in rows
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _annotate_top_collective_group(
    rows: list[dict[str, Any]], node: str
) -> list[int]:
    """Preserve one semantic TP boundary across its N physical kernels."""

    indices = [
        index for index, row in enumerate(rows) if row.get("selected_node") == node
    ]
    for index in indices:
        row = rows[index]
        kernel = str(row.get("kernel_name") or "").lower()
        row["launch_group_id"] = node
        row["launch_group_role"] = (
            "collective_kernel"
            if any(
                token in kernel
                for token in (
                    "nccl",
                    "multimem_all_reduce",
                    "allreducefusionkernel",
                    "allreducekernel",
                    "all_gather",
                    "allgather",
                )
            )
            else "result_materialization"
        )
    return indices


def compile_contract(
    mappings: list[dict[str, Any]],
    manifest: dict[str, Any],
    validation: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [dict(row) for row in mappings]
    for row in rows:
        _normalize_direct_evidence(row)
    source_commit = str(manifest.get("source_commit") or "")
    _recover_ordered_sglang_collectives(rows, source_commit=source_commit)
    _recover_ordered_sglang_compute(rows, source_commit=source_commit)
    embedding_collective = next(
        (
            index
            for index, row in enumerate(rows)
            if row.get("selected_node") == "top.tp_embedding_output_collective"
        ),
        None,
    )
    first_affine = next(
        (
            index
            for index, row in enumerate(rows)
            if row.get("selected_node") == "mhc_transform.affine"
        ),
        None,
    )
    if embedding_collective is not None and first_affine is not None:
        for row in rows[embedding_collective + 1 : first_affine]:
            if (
                row.get("selected_node") == "top.runtime_support"
                and row.get("cpu_op_name") == "aten::copy_"
            ):
                row.update(
                    {
                        "selected_node": "top.hc_expand",
                        "mapping_method": "ordered_model_forward_hc_initialization",
                        "confidence": "high",
                    }
                )
    _annotate_schedule(rows, source_commit=source_commit)
    for row in rows:
        _runtime_support(row)

    errors: list[str] = []
    if not validation.get("ok"):
        errors.append("low-level mapping validation did not pass")
    if validation.get("mapped_kernel_count") != validation.get("kernel_count"):
        errors.append("low-level mapping is not complete")
    if any(not row.get("selected_node") for row in rows):
        errors.append("compiled mapping contains an unselected event")
    untyped_support = [
        row["event_id"]
        for row in rows
        if row.get("selected_node") == "top.runtime_support"
        and not row.get("support_reason")
    ]
    if untyped_support:
        errors.append(f"untyped runtime support: {untyped_support[:8]}")
    generic_support = [
        row["event_id"]
        for row in rows
        if row.get("support_class") == "framework_runtime_metadata"
    ]
    if generic_support:
        errors.append(
            "generic framework runtime support remains unexplained: "
            f"{generic_support[:8]}"
        )
    missing_source = [
        row["event_id"]
        for row in rows
        if row.get("selected_node") != "top.runtime_support"
        and "unique_kernel_signature" not in (row.get("evidence") or [])
        and not any(
            (row.get(key) or {}).get("source_exists") is True
            for key in ("operator_frame", "semantic_frame", "phase_frame")
        )
        and row.get("selected_node")
        not in {
            "final_hc_read.read",
            "mhc_transform.affine",
            "mhc_transform.mix",
            "csa_compressor.softmax_pool",
            "hca_compressor.softmax_pool",
            "hca_compressor.norm_rope",
            "csa_indexer.k_compress",
            "csa_indexer.q_rope_rotate",
            "csa_indexer.causal_topk",
            "csa_indexer.expand",
            "moe.hash_select",
            "moe.learned_select",
        }
    ]
    if missing_source:
        errors.append(f"events without source-backed scope: {missing_source[:8]}")

    occurrence_ids = {
        str(row["occurrence_id"]) for row in rows if row.get("occurrence_id")
    }
    expected_occurrences = {
        f"layer_{layer:02d}.{substage}"
        for layer in range(LAYER_COUNT)
        for substage in ("attention", "feed_forward")
    }
    if occurrence_ids != expected_occurrences:
        errors.append("compiled layer occurrence set is incomplete")

    embedding_collective_indices = _annotate_top_collective_group(
        rows, "top.tp_embedding_output_collective"
    )
    logits_collective_indices = _annotate_top_collective_group(
        rows, "top.tp_logits_collective"
    )
    node_counts = Counter(str(row.get("selected_node")) for row in rows)
    for boundary, indices in (
        ("top.tp_embedding_output_collective", embedding_collective_indices),
        ("top.tp_logits_collective", logits_collective_indices),
    ):
        if not indices:
            errors.append(f"complete top-level runtime lacks {boundary}")
        collective_kernels = sum(
            rows[index].get("launch_group_role") == "collective_kernel"
            for index in indices
        )
        if indices and collective_kernels != 1:
            errors.append(
                f"{boundary} requires exactly one physical collective owner, "
                f"got {collective_kernels}"
            )
    lm_head_indices = [
        index
        for index, row in enumerate(rows)
        if row.get("selected_node") == "top.lm_head"
    ]
    if (
        lm_head_indices
        and logits_collective_indices
        and logits_collective_indices[0] <= lm_head_indices[-1]
    ):
        errors.append("TP logits collective does not follow the complete LM-head launch group")

    report = {
        "ok": not errors,
        "errors": errors,
        "source_commit": manifest.get("source_commit"),
        "rank": manifest.get("rank"),
        "phase": manifest.get("phase"),
        "window": manifest.get("window"),
        "kernel_count": len(rows),
        "mapped_kernel_count": sum(bool(row.get("selected_node")) for row in rows),
        "mapped_duration_ratio": validation.get("mapped_duration_ratio"),
        "layer_count": LAYER_COUNT,
        "csa_layers": list(CSA_LAYERS),
        "hca_layers": list(HCA_LAYERS),
        "occurrence_count": len(occurrence_ids),
        "node_counts": dict(node_counts),
        "top_collective_groups": {
            boundary: {
                "launch_group_id": boundary,
                "physical_kernel_count": len(indices),
                "collective_owner_count": sum(
                    rows[index].get("launch_group_role") == "collective_kernel"
                    for index in indices
                ),
            }
            for boundary, indices in (
                ("top.tp_embedding_output_collective", embedding_collective_indices),
                ("top.tp_logits_collective", logits_collective_indices),
            )
        },
        "support_class_counts": dict(
            Counter(
                str(row.get("support_class"))
                for row in rows
                if row.get("support_class")
            )
        ),
        "ordered_execution_fingerprint": _ordered_fingerprint(rows),
    }
    return rows, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    rows, report = compile_contract(
        load_jsonl(args.mapping),
        load_json(args.manifest),
        load_json(args.validation),
    )
    output_dir = args.out_dir.resolve()
    write_jsonl(output_dir / "eager_contract.jsonl", rows)
    write_json(output_dir / "eager_contract_report.json", report)
    print(
        f"ok={report['ok']} rank={report['rank']} phase={report['phase']} "
        f"kernels={report['kernel_count']} occurrences={report['occurrence_count']}"
    )
    for error in report["errors"]:
        print(f"error: {error}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
