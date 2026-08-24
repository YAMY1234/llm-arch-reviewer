#!/usr/bin/env python3
"""Generate the Qwen3.5 Model IR and DEP4 plan from frozen Qwen3.5 evidence.

The generator intentionally accepts only the Qwen3.5 checkpoint config.  It does
not read another model catalog, implementation profile, or viewer artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "catalog" / "qwen35" / "source_configs" / "config.json"
DEFAULT_CATALOG = REPO_ROOT / "catalog" / "qwen35"
EXPECTED_CONFIG_SHA256 = "9408a9e559cc2f05f0b357738213666353e6651160ce8ff477b1c26982bc4f63"
SGLANG_SEMANTIC_COMMIT = "5be8757c0d99f83bde9f5254b1fee30b97dcf66f"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _node(
    node_id: str,
    label: str,
    shape: str,
    semantic_op: str,
    **metadata: Any,
) -> dict[str, Any]:
    return {
        "id": node_id,
        "label": label,
        "shape": shape,
        "semantic_op": semantic_op,
        **metadata,
    }


def _edge(
    source: str,
    target: str,
    *,
    shape: str | None = None,
    dtype: str | None = None,
    kind: str | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {"from": source, "to": target}
    for key, item in (
        ("shape", shape),
        ("dtype", dtype),
        ("kind", kind),
        ("label", label),
    ):
        if item is not None:
            value[key] = item
    return value


def _top_view(hidden_size: int, vocab_size: int) -> dict[str, Any]:
    h = str(hidden_size)
    return {
        "title": "Qwen3.5 target model and MTP generation",
        "nodes": [
            _node("token_input", "Token IDs", "io", "model.token_ids", tensor="[B,T] int32"),
            _node(
                "embedding",
                "Token embedding",
                "gemm",
                "qwen3_5.token_embedding",
                weight=f"[{vocab_size},{h}]",
                output=f"[B,T,{h}]",
            ),
            _node(
                "decoder_stack",
                "60-layer hybrid decoder\nrepeat 15×: GDN ×3 + full attention ×1",
                "block",
                "qwen3_5.hybrid_decoder",
                drill="stack",
            ),
            _node(
                "state_store",
                "Persistent KV and GDN state",
                "cache",
                "qwen3_5.decode_state",
                drill="state_tensors",
            ),
            _node("final_norm", "Final RMSNorm", "norm", "qwen3_5.final_rms_norm"),
            _node(
                "lm_head",
                "Untied language-model head",
                "gemm",
                "qwen3_5.lm_head",
                weight=f"[{h},{vocab_size}]",
            ),
            _node("target_logits", "Target logits", "io", "qwen3_5.target_logits"),
            _node(
                "generation_controller",
                "MTP speculative generation loop",
                "block",
                "generation.mtp_loop",
                drill="generation_loop",
            ),
            _node("accepted_tokens", "Committed output tokens", "io", "generation.output"),
        ],
        "edges": [
            _edge("token_input", "embedding", shape="[B,T]", dtype="int32"),
            _edge("embedding", "decoder_stack", shape=f"[B,T,{h}]", dtype="bf16"),
            _edge(
                "state_store",
                "decoder_stack",
                kind="state_read",
                label="KV/GDN state",
            ),
            _edge(
                "decoder_stack",
                "state_store",
                kind="state_write",
                label="next KV/GDN state",
            ),
            _edge("decoder_stack", "final_norm", shape=f"[B,T,{h}]", dtype="bf16"),
            _edge("final_norm", "lm_head", shape=f"[B,T,{h}]", dtype="bf16"),
            _edge("lm_head", "target_logits", shape=f"[B,T,{vocab_size}]", dtype="bf16"),
            _edge("target_logits", "generation_controller", label="verification logits"),
            _edge("generation_controller", "accepted_tokens", dtype="int32"),
        ],
    }


def _stack_view(layer_types: list[str]) -> dict[str, Any]:
    linear_indices = [
        index for index, layer_type in enumerate(layer_types)
        if layer_type == "linear_attention"
    ]
    attention_indices = [
        index for index, layer_type in enumerate(layer_types)
        if layer_type == "full_attention"
    ]
    return {
        "title": "Compact 60-layer hybrid decoder schedule",
        "nodes": [
            _node("stack_input", "Embedded tokens", "io", "qwen3_5.stack.input"),
            _node(
                "schedule",
                "Repeat 15 times\nGDN + MoE ×3 → full attention + MoE ×1",
                "elem",
                "qwen3_5.stack.schedule",
                exact_layer_types=list(layer_types),
                full_attention_layer_indices=list(attention_indices),
            ),
            _node(
                "gdn_layer",
                "GDN + MoE decoder layer\n45 layers",
                "block",
                "qwen3_5.decoder.gdn_moe_layer",
                drill="gdn_moe_block",
                layer_count=len(linear_indices),
                layer_indices=linear_indices,
            ),
            _node(
                "full_attention_layer",
                "Full-attention + MoE decoder layer\n15 layers",
                "block",
                "qwen3_5.decoder.full_attention_moe_layer",
                drill="full_attention_moe_block",
                layer_count=len(attention_indices),
                layer_indices=list(attention_indices),
            ),
            _node("stack_output", "Decoder hidden states", "io", "qwen3_5.stack.output"),
        ],
        "edges": [
            _edge("stack_input", "schedule", shape="[B,T,H]", dtype="bf16"),
            _edge(
                "schedule",
                "gdn_layer",
                kind="dashed",
                label="layers 0–2, 4–6, …, 56–58",
            ),
            _edge(
                "schedule",
                "full_attention_layer",
                kind="dashed",
                label="layers 3, 7, …, 59",
            ),
            _edge("gdn_layer", "stack_output", shape="[B,T,H]", dtype="bf16"),
            _edge(
                "full_attention_layer",
                "stack_output",
                shape="[B,T,H]",
                dtype="bf16",
            ),
        ],
    }


def _layer_schedule(layer_types: list[str]) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    for index, layer_type in enumerate(layer_types):
        is_linear = layer_type == "linear_attention"
        nodes.append(
            _node(
                f"layer_{index:02d}",
                f"Layer {index}: {'GDN' if is_linear else 'full attention'} + MoE",
                "block",
                (
                    "qwen3_5.decoder.gdn_moe_layer"
                    if is_linear
                    else "qwen3_5.decoder.full_attention_moe_layer"
                ),
                drill="gdn_moe_block" if is_linear else "full_attention_moe_block",
                timeline_rollup=False,
                layer_index=index,
                layer_type=layer_type,
            )
        )
        if index:
            edges.append(
                _edge(
                    f"layer_{index - 1:02d}",
                    f"layer_{index:02d}",
                    shape="[B,T,H]",
                    dtype="bf16",
                )
            )
    return {
        "title": "Exact 60-layer order from text_config.layer_types",
        "nodes": nodes,
        "edges": edges,
    }


def _decoder_layer_view(*, mtp: bool = False, gdn: bool = False) -> dict[str, Any]:
    semantic_prefix = "generation.mtp" if mtp else "qwen3_5"
    attention_label = "Gated Delta Network" if gdn else "Gated grouped-query attention"
    attention_view = "gdn_attention" if gdn else ("mtp_full_attention" if mtp else "full_attention")
    moe_view = "mtp_moe_block" if mtp else "moe_block"

    def semantic(suffix: str) -> str:
        return f"{semantic_prefix}.{suffix}"

    return {
        "title": (
            "Qwen3.5 MTP full-attention + MoE decoder layer"
            if mtp
            else f"Qwen3.5 {attention_label} + MoE decoder layer"
        ),
        "nodes": [
            _node("input_hidden", "Layer input", "io", semantic("layer_input")),
            _node(
                "input_norm",
                "Pre-attention RMSNorm" if not gdn else "Pre-GDN RMSNorm",
                "norm",
                semantic("input_rms_norm"),
            ),
            _node(
                "attention",
                attention_label,
                "attn",
                semantic("gdn" if gdn else "attention"),
                drill=attention_view,
            ),
            _node(
                "attention_residual",
                "Attention residual add" if not gdn else "GDN residual add",
                "elem",
                semantic("residual_add"),
            ),
            _node(
                "post_attention_norm",
                "Pre-MoE RMSNorm",
                "norm",
                semantic("post_attention_rms_norm"),
            ),
            _node(
                "moe",
                "Sparse MoE + shared expert",
                "moe",
                semantic("moe"),
                drill=moe_view,
            ),
            _node("layer_residual", "MoE residual add", "elem", semantic("residual_add")),
            _node("output_hidden", "Layer output", "io", semantic("layer_output")),
        ],
        "edges": [
            _edge("input_hidden", "input_norm", shape="[N,H]", dtype="bf16"),
            _edge("input_norm", "attention", dtype="bf16"),
            _edge("input_hidden", "attention_residual", label="residual"),
            _edge("attention", "attention_residual"),
            _edge("attention_residual", "post_attention_norm", dtype="bf16"),
            _edge("post_attention_norm", "moe", dtype="bf16"),
            _edge("attention_residual", "layer_residual", label="residual"),
            _edge("moe", "layer_residual"),
            _edge("layer_residual", "output_hidden", shape="[N,H]", dtype="bf16"),
        ],
    }


def _gdn_attention_view(config: dict[str, Any]) -> dict[str, Any]:
    hidden = config["hidden_size"]
    key_dim = config["linear_num_key_heads"] * config["linear_key_head_dim"]
    value_dim = config["linear_num_value_heads"] * config["linear_value_head_dim"]
    conv_dim = 2 * key_dim + value_dim
    window = config["linear_conv_kernel_dim"] - 1
    return {
        "title": "Qwen3.5 Gated Delta Network module",
        "nodes": [
            _node("module_input", "Normalized hidden states", "io", "qwen3_5.gdn.input"),
            _node(
                "qkvz_projection",
                "Q/K/V/Z projection",
                "gemm",
                "qwen3_5.gdn.project_qkvz",
                output=f"[N,{2 * key_dim + 2 * value_dim}]",
            ),
            _node(
                "ba_projection",
                "B/A gate projection",
                "gemm",
                "qwen3_5.gdn.project_ba",
                output=f"[N,{2 * config['linear_num_value_heads']}]",
            ),
            _node(
                "conv_state_read",
                "Read causal-convolution window",
                "cache",
                "qwen3_5.gdn.conv_state.read",
                tensor=f"[B,{conv_dim},{window}]",
                dtype="bf16",
            ),
            _node(
                "causal_conv",
                "Short causal convolution over Q/K/V",
                "attn",
                "qwen3_5.gdn.causal_conv1d",
            ),
            _node(
                "recurrent_state_read",
                "Read GDN recurrent state",
                "cache",
                "qwen3_5.gdn.recurrent_state.read",
                tensor=(
                    f"[B,{config['linear_num_value_heads']},"
                    f"{config['linear_value_head_dim']},{config['linear_key_head_dim']}]"
                ),
                dtype=config["mamba_ssm_dtype"],
            ),
            _node(
                "gated_delta_recurrence",
                "Gated delta-rule recurrence",
                "attn",
                "qwen3_5.gdn.recurrence",
            ),
            _node(
                "state_write",
                "Write next convolution + recurrent state",
                "cache",
                "qwen3_5.gdn.state.write",
            ),
            _node(
                "output_gate_norm",
                "Gated RMSNorm",
                "norm",
                "qwen3_5.gdn.output_gate_norm",
            ),
            _node("output_projection", "GDN output projection", "gemm", "qwen3_5.gdn.out_proj"),
            _node("module_output", "GDN output", "io", "qwen3_5.gdn.output"),
        ],
        "edges": [
            _edge("module_input", "qkvz_projection", dtype="bf16"),
            _edge("module_input", "ba_projection", dtype="bf16"),
            _edge("qkvz_projection", "causal_conv", label="Q/K/V"),
            _edge("conv_state_read", "causal_conv", kind="state_read"),
            _edge("causal_conv", "gated_delta_recurrence", label="convolved Q/K/V"),
            _edge("ba_projection", "gated_delta_recurrence", label="decay/update gates"),
            _edge("recurrent_state_read", "gated_delta_recurrence", kind="state_read"),
            _edge("gated_delta_recurrence", "state_write", kind="state_write"),
            _edge("gated_delta_recurrence", "output_gate_norm"),
            _edge("qkvz_projection", "output_gate_norm", label="Z gate"),
            _edge("output_gate_norm", "output_projection", dtype="bf16"),
            _edge("output_projection", "module_output", shape="[N,H]", dtype="bf16"),
        ],
    }


def _full_attention_module_view(config: dict[str, Any], *, mtp: bool = False) -> dict[str, Any]:
    q_width = config["num_attention_heads"] * config["head_dim"]
    kv_width = config["num_key_value_heads"] * config["head_dim"]
    semantic_prefix = "generation.mtp" if mtp else "qwen3_5"
    title_prefix = "Qwen3.5 MTP draft" if mtp else "Qwen3.5"

    def semantic(suffix: str) -> str:
        return f"{semantic_prefix}.{suffix}"

    return {
        "title": f"{title_prefix} gated grouped-query attention module",
        "nodes": [
            _node("module_input", "Normalized hidden states", "io", semantic("attention.input")),
            _node(
                "qkv_projection",
                "Q/K/V projection",
                "gemm",
                semantic("attention.project_qkv"),
                output=f"[N,{q_width + 2 * kv_width}]",
            ),
            _node("qk_norm", "Per-head Q/K RMSNorm", "norm", semantic("attention.qk_norm")),
            _node(
                "partial_rope",
                "Partial RoPE (25% of head channels)",
                "elem",
                semantic("attention.partial_rope"),
            ),
            _node(
                "kv_state_read",
                "Read draft KV state" if mtp else "Read full-attention KV state",
                "cache",
                semantic("attention.kv_cache.read"),
                tensor="K,V [B,2,T,256]",
            ),
            _node("causal_gqa", "Causal grouped-query attention", "attn", semantic("attention.gqa")),
            _node(
                "kv_state_write",
                "Append draft K/V state" if mtp else "Append K/V state",
                "cache",
                semantic("attention.kv_cache.write"),
            ),
            _node(
                "attention_output_gate",
                "Attention output gate",
                "elem",
                semantic("attention.output_gate"),
            ),
            _node("output_projection", "Attention output projection", "gemm", semantic("attention.o_proj")),
            _node("module_output", "Attention output", "io", semantic("attention.output")),
        ],
        "edges": [
            _edge("module_input", "qkv_projection", dtype="bf16"),
            _edge("qkv_projection", "qk_norm", label="Q/K"),
            _edge("qk_norm", "partial_rope"),
            _edge("partial_rope", "causal_gqa", label="Q/K"),
            _edge("qkv_projection", "causal_gqa", label="V + gate"),
            _edge("kv_state_read", "causal_gqa", kind="state_read"),
            _edge("partial_rope", "kv_state_write", kind="state_write", label="K"),
            _edge("qkv_projection", "kv_state_write", kind="state_write", label="V"),
            _edge("causal_gqa", "attention_output_gate"),
            _edge("qkv_projection", "attention_output_gate", label="gate"),
            _edge("attention_output_gate", "output_projection", dtype="bf16"),
            _edge("output_projection", "module_output", dtype="bf16"),
        ],
    }


def _moe_view(config: dict[str, Any], *, mtp: bool = False) -> dict[str, Any]:
    semantic_prefix = "generation.mtp.moe" if mtp else "qwen3_5.moe"
    title_scope = "MTP draft" if mtp else "target"

    def semantic(suffix: str) -> str:
        return f"{semantic_prefix}.{suffix}"

    return {
        "title": f"Qwen3.5 {title_scope} logical routed + shared expert computation",
        "nodes": [
            _node("input_hidden", "MoE input", "io", semantic("input")),
            _node(
                "router",
                "Router and top-10 selection",
                "gemm",
                semantic("router_topk"),
                routed_experts=config["num_experts"],
                experts_per_token=config["num_experts_per_tok"],
            ),
            _node(
                "routed_experts",
                "512 routed SwiGLU experts",
                "moe",
                semantic("routed_experts"),
                expert_intermediate_size=config["moe_intermediate_size"],
            ),
            _node(
                "shared_expert",
                "Always-on shared SwiGLU expert",
                "moe",
                semantic("shared_expert"),
                intermediate_size=config["shared_expert_intermediate_size"],
            ),
            _node(
                "weighted_combine",
                "Weighted routed sum + shared expert",
                "elem",
                semantic("combine"),
            ),
            _node("output_hidden", "MoE output", "io", semantic("output")),
        ],
        "edges": [
            _edge("input_hidden", "router", shape="[N,H]", dtype="bf16"),
            _edge("router", "routed_experts", label="top-10 expert assignments"),
            _edge("input_hidden", "shared_expert", dtype="bf16"),
            _edge("routed_experts", "weighted_combine", dtype="bf16"),
            _edge("shared_expert", "weighted_combine", dtype="bf16"),
            _edge("weighted_combine", "output_hidden", shape="[N,H]", dtype="bf16"),
        ],
    }


def _state_view(config: dict[str, Any]) -> dict[str, Any]:
    key_dim = config["linear_num_key_heads"] * config["linear_key_head_dim"]
    value_dim = config["linear_num_value_heads"] * config["linear_value_head_dim"]
    conv_dim = 2 * key_dim + value_dim
    window = config["linear_conv_kernel_dim"] - 1
    return {
        "title": "Canonical per-request decode state (unsharded model semantics)",
        "nodes": [
            _node(
                "attention_keys",
                "15-layer key cache",
                "cache",
                "qwen3_5.state.attention_keys",
                tensor=f"[B,15,{config['num_key_value_heads']},T,{config['head_dim']}]",
                dtype="bf16",
                note="Runtime profiles may overlay FP8 storage.",
            ),
            _node(
                "attention_values",
                "15-layer value cache",
                "cache",
                "qwen3_5.state.attention_values",
                tensor=f"[B,15,{config['num_key_value_heads']},T,{config['head_dim']}]",
                dtype="bf16",
                note="Runtime profiles may overlay FP8 storage.",
            ),
            _node(
                "mtp_draft_keys",
                "One-layer MTP draft key cache",
                "cache",
                "generation.mtp.state.attention_keys",
                tensor=f"[B,1,{config['num_key_value_heads']},T,{config['head_dim']}]",
                dtype="bf16",
                lifecycle="advance along the candidate chain; trim/reset at the accepted boundary",
            ),
            _node(
                "mtp_draft_values",
                "One-layer MTP draft value cache",
                "cache",
                "generation.mtp.state.attention_values",
                tensor=f"[B,1,{config['num_key_value_heads']},T,{config['head_dim']}]",
                dtype="bf16",
                lifecycle="advance along the candidate chain; trim/reset at the accepted boundary",
            ),
            _node(
                "gdn_conv_windows",
                "45-layer GDN convolution windows",
                "cache",
                "qwen3_5.state.gdn_conv",
                tensor=f"[B,45,{conv_dim},{window}]",
                dtype="bf16",
            ),
            _node(
                "gdn_recurrent_states",
                "45-layer GDN recurrent matrices",
                "cache",
                "qwen3_5.state.gdn_recurrent",
                tensor=(
                    f"[B,45,{config['linear_num_value_heads']},"
                    f"{config['linear_value_head_dim']},{config['linear_key_head_dim']}]"
                ),
                dtype=config["mamba_ssm_dtype"],
                note="Runtime profiles may explicitly override the configured dtype.",
            ),
            _node(
                "verify_journal",
                "Per-draft-step tentative GDN/KV journal",
                "cache",
                "generation.mtp.tentative_state",
                tensor="[B,draft_steps,state]",
                lifecycle="allocate during target verify; accept prefix; discard suffix",
            ),
            _node(
                "committed_lengths",
                "Committed sequence lengths",
                "cache",
                "generation.committed_lengths",
                tensor="[B] int32",
            ),
        ],
        "edges": [
            _edge("attention_keys", "verify_journal", kind="snapshot"),
            _edge("attention_values", "verify_journal", kind="snapshot"),
            _edge("mtp_draft_keys", "verify_journal", kind="draft_boundary"),
            _edge("mtp_draft_values", "verify_journal", kind="draft_boundary"),
            _edge("gdn_conv_windows", "verify_journal", kind="snapshot"),
            _edge("gdn_recurrent_states", "verify_journal", kind="snapshot"),
            _edge("committed_lengths", "verify_journal", kind="boundary"),
        ],
    }


def _mtp_head_view(config: dict[str, Any]) -> dict[str, Any]:
    hidden = config["hidden_size"]
    return {
        "title": "One-layer Qwen3.5 MTP draft head",
        "nodes": [
            _node("draft_input_token", "Previous draft token", "io", "generation.mtp.input_token"),
            _node("target_hidden", "Target hidden state", "io", "generation.mtp.target_hidden"),
            _node(
                "shared_embedding",
                "Target token embedding (not dedicated)",
                "gemm",
                "generation.mtp.shared_embedding",
            ),
            _node("embedding_norm", "Embedding RMSNorm", "norm", "generation.mtp.embedding_norm"),
            _node("hidden_norm", "Target-hidden RMSNorm", "norm", "generation.mtp.hidden_norm"),
            _node("concat", "Concatenate embedding + hidden", "elem", "generation.mtp.concat"),
            _node(
                "fc_projection",
                "2H → H projection",
                "gemm",
                "generation.mtp.fc_projection",
                weight=f"[{2 * hidden},{hidden}]",
            ),
            _node(
                "draft_decoder_layer",
                "One full-attention + MoE draft layer",
                "block",
                "generation.mtp.decoder_layer",
                drill="mtp_full_attention_moe_block",
            ),
            _node("draft_final_norm", "Draft final RMSNorm", "norm", "generation.mtp.final_norm"),
            _node("shared_lm_head", "Target LM head", "gemm", "generation.mtp.shared_lm_head"),
            _node("draft_logits", "Draft logits", "io", "generation.mtp.logits"),
        ],
        "edges": [
            _edge("draft_input_token", "shared_embedding", dtype="int32"),
            _edge("shared_embedding", "embedding_norm", dtype="bf16"),
            _edge("target_hidden", "hidden_norm", dtype="bf16"),
            _edge("embedding_norm", "concat", dtype="bf16"),
            _edge("hidden_norm", "concat", dtype="bf16"),
            _edge("concat", "fc_projection", shape="[N,2H]", dtype="bf16"),
            _edge("fc_projection", "draft_decoder_layer", shape="[N,H]", dtype="bf16"),
            _edge("draft_decoder_layer", "draft_final_norm", dtype="bf16"),
            _edge("draft_final_norm", "shared_lm_head", dtype="bf16"),
            _edge("shared_lm_head", "draft_logits", dtype="bf16"),
        ],
    }


def _generation_view() -> dict[str, Any]:
    return {
        "title": "MTP draft, target verify, accept, replay, and atomic commit",
        "nodes": [
            _node("committed_context", "Read committed token/KV/GDN state", "cache", "generation.state.read"),
            _node(
                "draft_propose",
                "Draft candidate chain",
                "block",
                "generation.mtp.draft",
                drill="mtp_draft_head",
                timeline_rollup=False,
            ),
            _node("candidate_tokens", "Candidate tokens", "io", "generation.mtp.candidates"),
            _node(
                "target_verify",
                "Target-model batched verification",
                "block",
                "generation.mtp.target_verify",
                drill="stack",
                timeline_rollup=False,
            ),
            _node(
                "tentative_state",
                "Tentative per-step KV/GDN states",
                "cache",
                "generation.mtp.verify_journal",
            ),
            _node(
                "accept_prefix",
                "Select accepted prefix + target bonus token",
                "elem",
                "generation.mtp.accept",
            ),
            _node(
                "replay_gdn",
                "Replay/fold accepted GDN transitions",
                "block",
                "generation.mtp.replay_gdn",
                note="Reconstruct the recurrent state for the accepted path; discard rejected suffix states.",
            ),
            _node(
                "commit_kv",
                "Commit accepted KV prefix",
                "cache",
                "generation.mtp.commit_kv",
            ),
            _node(
                "commit_gdn",
                "Commit accepted convolution + recurrent state",
                "cache",
                "generation.mtp.commit_gdn",
            ),
            _node(
                "commit_tokens",
                "Atomically publish accepted tokens and length",
                "io",
                "generation.mtp.commit_tokens",
            ),
            _node("next_iteration", "Advance generation loop", "block", "generation.mtp.next_iteration"),
        ],
        "edges": [
            _edge("committed_context", "draft_propose", kind="state_read"),
            _edge("draft_propose", "candidate_tokens", dtype="int32"),
            _edge("candidate_tokens", "target_verify", dtype="int32"),
            _edge("committed_context", "target_verify", kind="state_read"),
            _edge("target_verify", "tentative_state", kind="state_write"),
            _edge("target_verify", "accept_prefix", label="target logits"),
            _edge("candidate_tokens", "accept_prefix", label="draft path"),
            _edge("accept_prefix", "replay_gdn", label="accepted path indices"),
            _edge("tentative_state", "replay_gdn", kind="state_read", label="raw transitions"),
            _edge("accept_prefix", "commit_kv", label="accepted length"),
            _edge("tentative_state", "commit_kv", kind="state_read"),
            _edge("replay_gdn", "commit_gdn", kind="state_write"),
            _edge("commit_kv", "commit_tokens", kind="commit_barrier"),
            _edge("commit_gdn", "commit_tokens", kind="commit_barrier"),
            _edge("accept_prefix", "commit_tokens", label="accepted tokens + bonus"),
            _edge("commit_tokens", "next_iteration"),
            _edge("next_iteration", "committed_context", kind="loop_back"),
        ],
    }


def build_model_ir(raw_config: dict[str, Any], config_sha256: str) -> dict[str, Any]:
    text = raw_config["text_config"]
    layer_types = list(text["layer_types"])
    if raw_config.get("model_type") != "qwen3_5_moe":
        raise ValueError("expected model_type=qwen3_5_moe")
    if text.get("model_type") != "qwen3_5_moe_text":
        raise ValueError("expected text_config.model_type=qwen3_5_moe_text")
    if len(layer_types) != text["num_hidden_layers"]:
        raise ValueError("layer_types length does not match num_hidden_layers")
    unknown = sorted(set(layer_types) - {"linear_attention", "full_attention"})
    if unknown:
        raise ValueError(f"unknown Qwen3.5 layer types: {unknown}")
    linear_count = layer_types.count("linear_attention")
    attention_count = layer_types.count("full_attention")
    if (len(layer_types), linear_count, attention_count) != (60, 45, 15):
        raise ValueError("frozen Qwen3.5 config must contain 60 layers: 45 GDN + 15 full attention")
    if (text["num_experts"], text["num_experts_per_tok"]) != (512, 10):
        raise ValueError("frozen Qwen3.5 config must contain 512 routed experts with top-10")
    if text["mtp_num_hidden_layers"] != 1:
        raise ValueError("frozen Qwen3.5 config must contain one MTP hidden layer")

    key_dim = text["linear_num_key_heads"] * text["linear_key_head_dim"]
    value_dim = text["linear_num_value_heads"] * text["linear_value_head_dim"]
    conv_dim = 2 * key_dim + value_dim
    return {
        "schema_version": "model-ir.v2",
        "model_id": "qwen35_397b_a17b",
        "model_label": "Qwen3.5 397B-A17B NVFP4 (text backbone)",
        "ir_version": 2,
        "default_view": "top",
        "default_execution_path": "attention_dp4_moe_ep4",
        "default_profile": (
            "qwen35_sglang_attention_dp4_moe_ep4_mtp6_agentx_torch_bs32"
        ),
        "dimensions": {
            "B": "request batch",
            "B_local": "requests assigned to one attention-DP rank",
            "T": "token positions",
            "N": "flattened active tokens",
            "H": text["hidden_size"],
            "V": text["vocab_size"],
            "Q_heads": text["num_attention_heads"],
            "KV_heads": text["num_key_value_heads"],
            "attention_head_dim": text["head_dim"],
            "GDN_key_heads": text["linear_num_key_heads"],
            "GDN_value_heads": text["linear_num_value_heads"],
            "GDN_key_head_dim": text["linear_key_head_dim"],
            "GDN_value_head_dim": text["linear_value_head_dim"],
        },
        "facts": {
            "evidence": {
                "checkpoint_config": "source_configs/config.json",
                "checkpoint_config_sha256": config_sha256,
                "checkpoint_repo": "nvidia/Qwen3.5-397B-A17B-NVFP4-V2",
                "checkpoint_revision": "8f590eae8f10bf55d9a46f79ea0280bde435c9f8",
                "sglang_semantic_source": (
                    "https://github.com/sgl-project/sglang/commit/" + SGLANG_SEMANTIC_COMMIT
                ),
                "derivation": "models/qwen35/build/build_qwen35_ir.py",
                "generation_policy": "Qwen3.5 config/source only; no other model IR or profile input",
            },
            "architecture": {
                "layers": 60,
                "gdn_layers": 45,
                "full_attention_layers": 15,
                "full_attention_layer_indices": [
                    index for index, value in enumerate(layer_types) if value == "full_attention"
                ],
                "routed_experts_per_layer": text["num_experts"],
                "experts_selected_per_token": text["num_experts_per_tok"],
                "shared_experts_per_layer": 1,
                "mtp_hidden_layers": text["mtp_num_hidden_layers"],
                "mtp_dedicated_embeddings": text["mtp_use_dedicated_embeddings"],
                "tie_word_embeddings": text["tie_word_embeddings"],
            },
            "state": {
                "full_attention_kv_per_layer": (
                    f"K,V [B,{text['num_key_value_heads']},T,{text['head_dim']}]"
                ),
                "gdn_conv_per_layer": (
                    f"[B,{conv_dim},{text['linear_conv_kernel_dim'] - 1}] bf16"
                ),
                "gdn_recurrent_per_layer": (
                    f"[B,{text['linear_num_value_heads']},{text['linear_value_head_dim']},"
                    f"{text['linear_key_head_dim']}] {text['mamba_ssm_dtype']}"
                ),
                "speculative_lifecycle": "verify journal -> accepted-prefix replay -> KV/GDN commit",
            },
        },
        "views": {
            "top": _top_view(text["hidden_size"], text["vocab_size"]),
            "stack": _stack_view(layer_types),
            "layer_schedule": _layer_schedule(layer_types),
            "gdn_moe_block": _decoder_layer_view(gdn=True),
            "gdn_attention": _gdn_attention_view(text),
            "full_attention_moe_block": _decoder_layer_view(),
            "full_attention": _full_attention_module_view(text),
            "moe_block": _moe_view(text),
            "mtp_full_attention_moe_block": _decoder_layer_view(mtp=True),
            "mtp_full_attention": _full_attention_module_view(text, mtp=True),
            "mtp_moe_block": _moe_view(text, mtp=True),
            "state_tensors": _state_view(text),
            "mtp_draft_head": _mtp_head_view(text),
            "generation_loop": _generation_view(),
        },
    }


def _annotate(target: str, **execution: Any) -> dict[str, Any]:
    return {"op": "annotate_node", "target": target, "set": {"execution": execution}}


def _ep_execution(role: str) -> dict[str, Any]:
    return {
        "placement": "one DEP group spanning ranks 0..3",
        "parallelism": "moe_ep4",
        "expert_ownership": "expert_id mod 4; 128 of 512 routed experts per rank",
        "role": role,
    }


def _moe_scope_transforms(view_id: str, scope: str) -> list[dict[str, Any]]:
    """Build an explicit, independently bindable EP4 boundary for one MoE scope."""
    prefix = f"{scope}_ep4"
    scope_label = "target model" if scope == "target" else "MTP draft model"
    group = "DEP/EP ranks [0,1,2,3]"
    wire_contract = (
        f"{scope} binding must record its own hidden encoding and metadata ABI; "
        "it must not inherit these choices from the other scope"
    )
    return [
        _annotate(f"{view_id}.router", **_ep_execution(f"{scope_label} local routing")),
        _annotate(f"{view_id}.routed_experts", **_ep_execution(f"{scope_label} expert-local compute")),
        _annotate(
            f"{view_id}.shared_expert",
            placement="replicated on every DEP rank",
            parallelism="attention_dp4",
            sharding="B_local tokens; shared-expert weights replicated",
            scope=scope,
        ),
        _annotate(f"{view_id}.weighted_combine", **_ep_execution(f"{scope_label} origin-rank sum")),
        {
            "op": "insert_after",
            "after": f"{view_id}.router",
            "node": {
                "id": f"{prefix}_pack",
                "label": f"{scope_label} EP4 dispatch pack",
                "shape": "elem",
                "semantic_op": "execution.layout.moe_dispatch_pack",
                "node_kind": "layout_transform",
                "boundary_role": "module_internal",
                "execution": {
                    "scope": scope,
                    "placement": "each origin rank",
                    "collective": "none (local stable pack)",
                    "group": group,
                    "parallelism": "moe_ep4",
                    "payload": (
                        "hidden [N_local,H] bf16, expert_ids [N_local,10] int32, "
                        "route_weights [N_local,10] fp32, origin_row [N_local] int32"
                    ),
                    "result": (
                        "four destination buckets of logical route records; each record is "
                        "(origin_rank, origin_row, route_slot, expert_id, weight, hidden[H])"
                    ),
                    "dtype": "logical bf16 + int32 + fp32",
                    "tensor_layout": "destination-rank-major, then expert-id-major, stable origin-row order",
                },
            },
            "edge": {"shape": "[N_local,10] route records", "dtype": "structured"},
        },
        {
            "op": "insert_after",
            "after": f"{view_id}.{prefix}_pack",
            "node": {
                "id": f"{prefix}_dispatch",
                "label": f"{scope_label} EP4 variable-size dispatch",
                "shape": "moe",
                "semantic_op": "execution.collective.moe_dispatch",
                "node_kind": "communication",
                "boundary_role": "module_internal",
                "execution": {
                    "scope": scope,
                    "placement": "all four ranks in the DEP/EP group",
                    "collective": "all_to_all_v",
                    "group": group,
                    "parallelism": "moe_ep4",
                    "payload": "per-destination packed logical route records from the local pack adapter",
                    "result": (
                        "expert-owner records [N_recv,H] logical bf16 plus int32 origin/expert/slot "
                        "and fp32 route-weight metadata; 128 logical experts per rank"
                    ),
                    "dtype": "logical structured bf16/int32/fp32; physical wire dtype is binding-owned",
                    "tensor_layout": "receive-source-major, then local-expert-major",
                    "wire_encoding_contract": wire_contract,
                },
            },
            "edge": {"shape": "[N_recv,H] + route metadata", "dtype": "structured"},
        },
        {
            "op": "insert_after",
            "after": f"{view_id}.routed_experts",
            "node": {
                "id": f"{prefix}_combine",
                "label": f"{scope_label} EP4 routed-output return",
                "shape": "moe",
                "semantic_op": "execution.collective.moe_combine",
                "node_kind": "communication",
                "boundary_role": "module_internal",
                "execution": {
                    "scope": scope,
                    "placement": "all four ranks in the DEP/EP group",
                    "collective": "all_to_all_v",
                    "group": group,
                    "parallelism": "moe_ep4",
                    "payload": (
                        "expert outputs [N_recv,H] logical bf16 with int32 origin-row/route-slot "
                        "and fp32 route-weight metadata"
                    ),
                    "result": "origin-rank routed records [N_local,10,H] logical bf16 plus fp32 weights",
                    "dtype": "logical structured bf16/int32/fp32; physical wire dtype is binding-owned",
                    "tensor_layout": "origin-rank-major on wire; receive-source-major before restore",
                    "wire_encoding_contract": wire_contract,
                },
            },
            "edge": {"shape": "[N_recv,H] + route metadata", "dtype": "structured"},
        },
        {
            "op": "insert_after",
            "after": f"{view_id}.{prefix}_combine",
            "node": {
                "id": f"{prefix}_restore",
                "label": f"{scope_label} routed-output restore",
                "shape": "elem",
                "semantic_op": "execution.layout.moe_combine_restore",
                "node_kind": "layout_transform",
                "boundary_role": "module_internal",
                "execution": {
                    "scope": scope,
                    "placement": "each origin rank",
                    "collective": "none (local inverse permutation)",
                    "group": group,
                    "parallelism": "moe_ep4",
                    "payload": "returned route records in receive-source-major order",
                    "result": "[N_local,10,H] bf16 in origin-token/route-slot order with [N_local,10] fp32 weights",
                    "dtype": "logical bf16 + fp32",
                    "tensor_layout": "token-major, route-slot-major, H contiguous",
                },
            },
            "edge": {"shape": "[N_local,10,H]", "dtype": "bf16"},
        },
    ]


def build_execution_plan(layer_types: list[str]) -> dict[str, Any]:
    transforms: list[dict[str, Any]] = []
    dp_execution = {
        "placement": "replicated weights on each of ranks 0..3",
        "parallelism": "attention_dp4",
        "request_sharding": "B -> B_local by request; hidden/head dimensions unsharded",
        "state_ownership": "KV and GDN state stay with the request-owning DP rank",
    }
    for target in (
        "top.embedding",
        "top.decoder_stack",
        "top.final_norm",
        "top.lm_head",
        "stack.gdn_layer",
        "stack.full_attention_layer",
        "gdn_moe_block.input_norm",
        "gdn_moe_block.attention",
        "gdn_attention.qkvz_projection",
        "gdn_attention.ba_projection",
        "gdn_attention.causal_conv",
        "gdn_attention.gated_delta_recurrence",
        "gdn_attention.output_gate_norm",
        "gdn_attention.output_projection",
        "full_attention_moe_block.input_norm",
        "full_attention_moe_block.attention",
        "full_attention.qkv_projection",
        "full_attention.qk_norm",
        "full_attention.partial_rope",
        "full_attention.causal_gqa",
        "full_attention.attention_output_gate",
        "full_attention.output_projection",
        "mtp_full_attention_moe_block.input_norm",
        "mtp_full_attention_moe_block.attention",
        "mtp_full_attention.qkv_projection",
        "mtp_full_attention.qk_norm",
        "mtp_full_attention.partial_rope",
        "mtp_full_attention.causal_gqa",
        "mtp_full_attention.attention_output_gate",
        "mtp_full_attention.output_projection",
        "mtp_draft_head.fc_projection",
        "mtp_draft_head.draft_decoder_layer",
        "mtp_draft_head.shared_lm_head",
        "generation_loop.draft_propose",
        "generation_loop.target_verify",
        "generation_loop.accept_prefix",
        "generation_loop.replay_gdn",
    ):
        transforms.append(_annotate(target, **dp_execution))
    for index, _layer_type in enumerate(layer_types):
        transforms.append(_annotate(f"layer_schedule.layer_{index:02d}", **dp_execution))
    for target in (
        "gdn_attention.conv_state_read",
        "gdn_attention.recurrent_state_read",
        "gdn_attention.state_write",
        "full_attention.kv_state_read",
        "full_attention.kv_state_write",
        "mtp_full_attention.kv_state_read",
        "mtp_full_attention.kv_state_write",
        "state_tensors.attention_keys",
        "state_tensors.attention_values",
        "state_tensors.mtp_draft_keys",
        "state_tensors.mtp_draft_values",
        "state_tensors.gdn_conv_windows",
        "state_tensors.gdn_recurrent_states",
        "state_tensors.verify_journal",
        "generation_loop.tentative_state",
        "generation_loop.commit_kv",
        "generation_loop.commit_gdn",
    ):
        transforms.append(
            _annotate(
                target,
                placement="request-owning attention-DP rank",
                parallelism="attention_dp4",
                sharding="request dimension B/4; state tensor otherwise unsharded",
            )
        )
    transforms.extend(
        [
            {
                "op": "insert_after",
                "after": "top.embedding",
                "node": {
                    "id": "dp4_request_partition",
                    "label": "Attention-DP4 request partition",
                    "shape": "elem",
                    "semantic_op": "execution.layout.dp4_request_partition",
                    "node_kind": "layout_transform",
                    "boundary_role": "module_boundary",
                    "execution": {
                        "placement": "request ingress for ranks 0..3",
                        "collective": "local_select",
                        "group": "attention-DP ranks [0,1,2,3]",
                        "parallelism": "attention_dp4",
                        "payload": "global logical hidden rows [B,T,H], bf16, row-major",
                        "result": "rank-local hidden rows [B_local,T,H], bf16, row-major",
                        "dtype": "bf16",
                        "tensor_layout": "token-major contiguous; H unsharded",
                    },
                },
                "edge": {"shape": "[B,T,H]", "dtype": "bf16"},
            },
            {
                "op": "insert_after",
                "after": "generation_loop.commit_tokens",
                "node": {
                    "id": "dp4_output_commit",
                    "label": "Attention-DP4 owner-local output commit",
                    "shape": "elem",
                    "semantic_op": "execution.layout.dp4_output_commit",
                    "node_kind": "layout_transform",
                    "boundary_role": "module_boundary",
                    "execution": {
                        "placement": "request-owning rank",
                        "collective": "none (owner-local commit)",
                        "group": "attention-DP ranks [0,1,2,3]",
                        "parallelism": "attention_dp4",
                        "payload": "accepted token IDs and committed length for B_local requests",
                        "result": "owner-local [B_local,accepted] int32 rows for frontend response routing",
                        "dtype": "int32",
                        "tensor_layout": "request-major; request identity retained for frontend merge",
                    },
                },
                "edge": {"shape": "[B_local,accepted]", "dtype": "int32"},
            },
        ]
    )
    transforms.extend(_moe_scope_transforms("moe_block", "target"))
    transforms.extend(_moe_scope_transforms("mtp_moe_block", "draft"))
    return {
        "schema_version": "execution-plan.v2",
        "execution_path_id": "attention_dp4_moe_ep4",
        "label": "Framework-independent Attention DP4 + MoE EP4 (DEP4)",
        "model_id": "qwen35_397b_a17b",
        "plan_version": 3,
        "parallelism_axes": {
            "tp_size": 1,
            "dp_size": 4,
            "cp_size": 1,
            "ep_size": 4,
            "attention_axis": "data_parallel",
            "moe_axis": "expert_parallel",
        },
        "default_parameters": {
            "physical_ranks": 4,
            "attention_weight_replicas": 4,
            "routed_experts_per_rank": 128,
            "shared_expert_replicas": 4,
        },
        "constraints": {
            "world_size_equation": "physical_ranks = dp_size = ep_size = 4",
            "expert_partition": "512 routed experts / EP4 = 128 experts per rank",
            "request_state_locality": "KV/GDN state follows its attention-DP request owner",
            "semantic_dtype_note": (
                "The plan preserves bf16 logical activations. Engine-specific FP4 dispatch "
                "or FP8 KV encodings belong to implementation/profile overlays."
            ),
            "scope_binding": (
                "Target and MTP draft dispatch/combine are distinct nodes. Every implementation "
                "must bind their backend, wire dtype, fusion, and padding independently."
            ),
        },
        "transforms": transforms,
    }


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False, width=110))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--catalog-root", type=Path, default=DEFAULT_CATALOG)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config_sha256 = _sha256(config_path)
    if config_sha256 != EXPECTED_CONFIG_SHA256:
        raise ValueError(
            f"unexpected Qwen3.5 config SHA256: {config_sha256}; "
            f"expected {EXPECTED_CONFIG_SHA256}"
        )
    raw_config = json.loads(config_path.read_text())
    model_ir = build_model_ir(raw_config, config_sha256)
    plan = build_execution_plan(list(raw_config["text_config"]["layer_types"]))
    _write_yaml(args.catalog_root / "model_ir.yaml", model_ir)
    _write_yaml(
        args.catalog_root / "execution_paths" / "attention_dp4_moe_ep4.yaml",
        plan,
    )
    print(f"wrote {args.catalog_root / 'model_ir.yaml'}")
    print(f"wrote {args.catalog_root / 'execution_paths' / 'attention_dp4_moe_ep4.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
