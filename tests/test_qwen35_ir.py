from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from llm_arch_v2.compiler import compile_catalog  # noqa: E402
from models.qwen35.build.build_qwen35_ir import (  # noqa: E402
    EXPECTED_CONFIG_SHA256,
    build_execution_plan,
    build_model_ir,
)


CATALOG_ROOT = REPO_ROOT / "catalog" / "qwen35"
CONFIG_PATH = CATALOG_ROOT / "source_configs" / "config.json"


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _nodes(document: dict, view_id: str) -> dict[str, dict]:
    return {node["id"]: node for node in document["views"][view_id]["nodes"]}


def test_qwen35_ir_is_reproducible_from_frozen_qwen35_config_only() -> None:
    raw_bytes = CONFIG_PATH.read_bytes()
    assert hashlib.sha256(raw_bytes).hexdigest() == EXPECTED_CONFIG_SHA256
    raw_config = json.loads(raw_bytes)
    committed_ir = _load(CATALOG_ROOT / "model_ir.yaml")
    committed_plan = _load(
        CATALOG_ROOT / "execution_paths" / "attention_dp4_moe_ep4.yaml"
    )

    assert build_model_ir(raw_config, EXPECTED_CONFIG_SHA256) == committed_ir
    assert build_execution_plan(raw_config["text_config"]["layer_types"]) == committed_plan

    generator = (
        REPO_ROOT / "models" / "qwen35" / "build" / "build_qwen35_ir.py"
    ).read_text().lower()
    assert "catalog/qwen40" not in generator
    assert "catalog" not in generator.split("def build_model_ir", 1)[1].split(
        "def _annotate", 1
    )[0]


def test_exact_layer_order_and_moe_cardinality_are_preserved() -> None:
    raw_config = json.loads(CONFIG_PATH.read_text())
    text = raw_config["text_config"]
    model_ir = _load(CATALOG_ROOT / "model_ir.yaml")
    schedule = model_ir["views"]["layer_schedule"]["nodes"]

    assert len(schedule) == text["num_hidden_layers"] == 60
    assert [node["layer_type"] for node in schedule] == text["layer_types"]
    assert sum(node["layer_type"] == "linear_attention" for node in schedule) == 45
    assert sum(node["layer_type"] == "full_attention" for node in schedule) == 15
    assert [
        node["layer_index"] for node in schedule if node["layer_type"] == "full_attention"
    ] == list(range(3, 60, 4))

    moe = _nodes(model_ir, "moe_block")
    assert moe["router"]["routed_experts"] == 512
    assert moe["router"]["experts_per_token"] == 10
    assert moe["shared_expert"]["intermediate_size"] == 1024


def test_kv_gdn_state_and_mtp_transaction_are_explicit() -> None:
    model_ir = _load(CATALOG_ROOT / "model_ir.yaml")
    states = _nodes(model_ir, "state_tensors")
    assert states["attention_keys"]["tensor"] == "[B,15,2,T,256]"
    assert states["attention_values"]["tensor"] == "[B,15,2,T,256]"
    assert states["gdn_conv_windows"]["tensor"] == "[B,45,12288,3]"
    assert states["gdn_recurrent_states"]["tensor"] == "[B,45,64,128,128]"
    assert states["gdn_recurrent_states"]["dtype"] == "float32"

    mtp_nodes = _nodes(model_ir, "mtp_draft_head")
    assert mtp_nodes["draft_decoder_layer"]["drill"] == "full_attention_moe_block"
    assert "not dedicated" in mtp_nodes["shared_embedding"]["label"].lower()

    generation = _nodes(model_ir, "generation_loop")
    for node_id in (
        "draft_propose",
        "target_verify",
        "accept_prefix",
        "replay_gdn",
        "commit_kv",
        "commit_gdn",
        "commit_tokens",
    ):
        assert node_id in generation
    edges = {
        (edge["from"], edge["to"])
        for edge in model_ir["views"]["generation_loop"]["edges"]
    }
    assert ("draft_propose", "candidate_tokens") in edges
    assert ("candidate_tokens", "target_verify") in edges
    assert ("target_verify", "accept_prefix") in edges
    assert ("accept_prefix", "replay_gdn") in edges
    assert ("replay_gdn", "commit_gdn") in edges
    assert ("commit_kv", "commit_tokens") in edges
    assert ("commit_gdn", "commit_tokens") in edges


def test_framework_independent_dep4_plan_compiles_with_explicit_payloads() -> None:
    plan = _load(CATALOG_ROOT / "execution_paths" / "attention_dp4_moe_ep4.yaml")
    assert plan["parallelism_axes"] == {
        "tp_size": 1,
        "dp_size": 4,
        "cp_size": 1,
        "ep_size": 4,
        "attention_axis": "data_parallel",
        "moe_axis": "expert_parallel",
    }
    assert plan["default_parameters"]["routed_experts_per_rank"] == 128

    inserted = [
        transform["node"]
        for transform in plan["transforms"]
        if transform["op"] == "insert_after"
    ]
    assert {node["id"] for node in inserted} == {
        "dp4_request_partition",
        "ep4_dispatch",
        "ep4_combine",
    }
    for node in inserted:
        execution = node["execution"]
        assert execution["payload"]
        assert execution["result"]
        assert execution["dtype"]
        assert execution["tensor_layout"]

    bundle = compile_catalog(CATALOG_ROOT)
    assert bundle["meta"]["model_id"] == "qwen35_397b_a17b"
    assert bundle["meta"]["execution_variant_count"] == 1
    assert bundle["meta"]["implementation_count"] == 0
    assert bundle["meta"]["profile_count"] == 0
    compiled_nodes = _nodes(bundle, "moe_block")
    assert compiled_nodes["ep4_dispatch"]["ir_origin"] == "execution_plan"
    assert compiled_nodes["ep4_dispatch"]["boundary_role"] == "module_internal"
    assert compiled_nodes["ep4_combine"]["execution"]["collective"] == "all_to_all_v"


def test_model_ir_does_not_contain_runtime_profile_choices() -> None:
    model_ir_text = (CATALOG_ROOT / "model_ir.yaml").read_text().lower()
    for implementation_choice in (
        "cuda graph",
        "deep_gemm",
        "deepep",
        "flashinfer",
        "trtllm_mha",
        "job 3109160",
        "job 501238",
    ):
        assert implementation_choice not in model_ir_text
