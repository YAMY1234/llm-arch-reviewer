"""Publisher bytes -> catalog checks, and publisher CPU arithmetic -> scalar oracles.

Nothing in the numerical expectations is obtained from Model IR or its builder.
GPU-only packed primitives are outside this CPU execution validation.
"""
from __future__ import annotations

import ast
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import math
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "catalog/deepseek_v41_flash"
FIXTURE = ROOT / "tests/fixtures/deepseek_v41_flash"
REVISION = "dba1be0a40aa45a94ad051997016db3960a90277"


def model():
    return yaml.safe_load((CATALOG / "model_ir.yaml").read_text())


def verify_sources(root, required=("config.json", "inference/config.json", "inference/model.py", "inference/kernel.py", "inference/vision.py", "inference/image_processor.py", "LICENSE")):
    for name in required:
        assert (root / name).is_file(), name
    lock = json.loads((FIXTURE / "source-lock.json").read_text())
    for row in lock["files"]:
        path = root / row["path"]
        if path.exists():
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"], row["path"]


def test_publisher_fixture_digests_and_mutated_source_rejected(tmp_path):
    verify_sources(FIXTURE)
    (tmp_path / "config.json").write_bytes((FIXTURE / "config.json").read_bytes() + b" ")
    with pytest.raises(AssertionError, match="config.json"):
        verify_sources(tmp_path, required=("config.json",))


def test_checkpoint_and_reference_config_correspondence():
    root = json.loads((FIXTURE / "config.json").read_text())
    ref = json.loads((FIXTURE / "inference/config.json").read_text())
    mapping = {
        "hidden_size": "dim", "num_hidden_layers": "n_layers",
        "moe_intermediate_size": "moe_inter_dim", "num_attention_heads": "n_heads",
        "num_experts_per_tok": "n_activated_experts", "scoring_func": "score_func",
        "routed_scaling_factor": "route_scale", "sliding_window": "window_size",
        "kv_source_layer_ids": "kv_source_layers", "index_source_layer_ids": "index_source_layers",
        "candidate_source_layer_id": "candidate_source_layer", "rms_norm_eps": "norm_eps",
        "num_nextn_predict_layers": "n_mtp_layers", "qk_rope_head_dim": "rope_head_dim",
        "engram_pad_token_id": "engram_pad_id",
        "dspark_num_experts_per_tok": "dspark_n_activated_experts",
    }
    for key, value in root["text_config"].items():
        target = mapping.get(key, key)
        if target in ref:
            assert value == ref[target], (key, target)
    assert len(ref["compress_ratios"]) == ref["n_layers"] + ref["n_mtp_layers"]
    assert ref["compress_ratios"][40:] == [0, 0, 0]
    assert root["quantization_config"]["expert_dtype"] == ref["expert_dtype"] == "fp4"


@pytest.fixture
def torch():
    try:
        import torch
    except ImportError:
        pytest.fail("Install CPU PyTorch for required DeepSeek V4.1 mathematical acceptance")
    return torch


def upstream(symbol, torch, replacement=None, source_file="inference/model.py", **extra):
    """Execute a selected exact source AST; GPU imports and checkpoint init never run."""
    text = (FIXTURE / source_file).read_text()
    if replacement:
        old, new = replacement
        assert old in text
        text = text.replace(old, new)
    parts = symbol.split(".")
    node = next(n for n in ast.parse(text).body if getattr(n, "name", None) == parts[0])
    if len(parts) == 2:
        node = next(n for n in node.body if getattr(n, "name", None) == parts[1])
    # Postponed annotations avoid constructing unrelated publisher classes.
    unit = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    env = {"lru_cache": lru_cache, "torch": torch, "F": torch.nn.functional, "linear": torch.nn.functional.linear, **extra}
    exec(compile(ast.fix_missing_locations(unit), str(FIXTURE / source_file), "exec"), env)
    return env[node.name]


def scalar_score(x):
    return math.sqrt(math.log1p(math.exp(x)))


def routing_case(torch, replacement=None):
    fn = upstream("Gate.forward", torch, replacement)
    args = SimpleNamespace(weight=torch.eye(3), gate_temp=1., score_func="sqrtsoftplus",
                           bias=torch.tensor([5., 0., 0.]), bias_vl=torch.tensor([0., 5., 0.]),
                           topk=2, norm_topk_prob=True, route_scale=1.5)
    x = torch.tensor([[-2., 0., 2.], [-2., 0., 2.]])
    weights, indices = fn(args, x, torch.tensor([False, True]))
    assert indices.tolist() == [[0, 2], [1, 2]]
    expected = [[1.5 * scalar_score(v) / sum(scalar_score(z) for z in pair) for v in pair]
                for pair in [(-2., 2.), (0., 2.)]]
    torch.testing.assert_close(weights, torch.tensor(expected), rtol=2e-6, atol=2e-7)


def test_routing_selection_bias_is_not_combination_weight(torch):
    routing_case(torch)
    with pytest.raises(AssertionError):
        routing_case(torch, ("F.softplus(scores).sqrt()", "scores.sigmoid()"))


def test_top1_router_does_not_normalize(torch):
    args = SimpleNamespace(weight=torch.eye(2), gate_temp=1., score_func="sqrtsoftplus",
                           bias=torch.zeros(2), bias_vl=None, topk=1, norm_topk_prob=True, route_scale=1.5)
    weights, indices = upstream("Gate.forward", torch)(args, torch.tensor([[0., 2.]]))
    assert indices.item() == 1
    assert weights.item() == pytest.approx(1.5 * scalar_score(2.), rel=1e-6)


def expert_case(torch, replacement=None):
    x = torch.tensor([[-20., 20.], [2., -3.]])
    args = SimpleNamespace(w1=lambda x: x, w3=lambda x: x.flip(-1),
                           w2=lambda x: x @ torch.tensor([[1., 2.], [3., -1.]]), swiglu_limit=10.)
    weights = torch.tensor([[.25], [.75]])
    actual = upstream("Expert.forward", torch, replacement)(args, x, weights)
    expected = []
    for row, weight in zip(x.tolist(), [.25, .75]):
        z = [min(g, 10.) / (1 + math.exp(-min(g, 10.))) * max(-10., min(u, 10.)) * weight
             for g, u in zip(row, row[::-1])]
        expected.append([z[0] + 3*z[1], 2*z[0] - z[1]])
    torch.testing.assert_close(actual, torch.tensor(expected), rtol=2e-6, atol=1e-7)


def test_expert_clipping_activation_weight_and_down_order(torch):
    expert_case(torch)
    with pytest.raises(AssertionError):
        expert_case(torch, ("gate = torch.clamp(gate, max=self.swiglu_limit)",
                            "gate = torch.clamp(F.silu(gate), max=self.swiglu_limit)"))


@pytest.mark.parametrize("streams,dim", [(2, 3), (4, 2)])
def test_mhc_pre_and_post_use_correct_stream_orientation(torch, streams, dim):
    residual = torch.arange(1., streams*dim+1).reshape(1, 1, streams, dim)
    pre = torch.arange(1., streams+1).reshape(1, 1, streams)/10
    comb = torch.arange(1., streams*streams+1).reshape(1, 1, streams, streams)/20
    post = pre * 2
    x = torch.arange(1., dim+1).reshape(1, 1, dim)
    collapse = upstream("Block.hc_pre", torch)(None, residual, pre)
    recombine = upstream("Block.hc_post", torch)(None, x, residual, post, comb)
    expected_pre = [sum(pre[0,0,i].item()*residual[0,0,i,d].item() for i in range(streams)) for d in range(dim)]
    expected_post = [[post[0,0,j].item()*x[0,0,d].item() + sum(comb[0,0,i,j].item()*residual[0,0,i,d].item() for i in range(streams)) for d in range(dim)] for j in range(streams)]
    torch.testing.assert_close(collapse, torch.tensor([[expected_pre]]))
    torch.testing.assert_close(recombine, torch.tensor([[expected_post]]))


@pytest.mark.parametrize("window,length", [(4, 1), (4, 3), (4, 7), (2, 5)])
def test_prefill_window_causal_boundaries(torch, window, length):
    rows = upstream("get_window_topk_idxs", torch)(window, 2, length, 0)
    assert rows.shape == (2,length,min(length,window))
    for q, row in enumerate(rows[0].tolist()):
        assert [p for p in row if p >= 0] == list(range(max(0,q-window+1),q+1))
    assert torch.equal(rows[0], rows[1])


@pytest.mark.parametrize("position", [1, 2, 3, 4, 7, 8])
def test_decode_ring_contains_current_token_and_no_stale_slot(torch, position):
    window = 4
    row = upstream("get_window_topk_idxs", torch)(window, 1, 1, position)[0,0].tolist()
    assert len(row) == window
    expected = [p % window for p in range(max(0,position-window+1),position+1)]
    assert [p for p in row if p >= 0] == expected
    assert position % window in row


def test_candidate_pin_padding_and_no_reachable_blocks(torch):
    fn = upstream("select_candidate_blocks", torch)
    scores = torch.tensor([[9., 8., 7., 6., .1], [-math.inf]*5])
    result = fn(scores, torch.tensor([[5], [0]]), 1, 2)
    assert result.tolist() == [[False, False, False, False, True], [False]*5]
    # Last block has only one token: it is pinned despite lower scores.
    stale = fn(scores[:1], 2, 1, 2)
    assert not torch.equal(stale, result[:1])


def test_compressor_partial_group_commit_and_resume(torch):
    fn = upstream("Compressor.forward", torch)
    args = SimpleNamespace(compress_ratio=2, wkv=lambda x:x, wgate=lambda x:torch.zeros_like(x),
                           norm=lambda x:x, kv_state=torch.zeros(1,2,2),
                           score_state=torch.full((1,2,2), -math.inf))
    initial = torch.tensor([[[2., 4.], [4., 8.], [10., 20.]]])
    torch.testing.assert_close(fn(args, initial, 0), torch.tensor([[[3., 6.]]]))
    torch.testing.assert_close(fn(args, torch.tensor([[[14., 28.]]]), 3), torch.tensor([[[12., 24.]]]))
    assert fn(args, torch.tensor([[[100., 200.]]]), 4) is None
    torch.testing.assert_close(fn(args, torch.tensor([[[200., 400.]]]), 5), torch.tensor([[[150., 300.]]]))


def test_reuse_reads_published_index_identity_and_source_replaces_stale(torch):
    state = SimpleNamespace(topk_idxs=torch.tensor([[[999]]]))
    fn = upstream("Attention._compress_topk_idxs", torch, shared_attn=state)
    x = torch.zeros(1,2,3)
    reuse = SimpleNamespace(is_index_source=False)
    assert fn(reuse,x,None,None,0,2,1) is state.topk_idxs
    source = SimpleNamespace(is_index_source=True)
    fresh = fn(source,x,None,None,0,2,0)
    assert fresh.shape == (1,2,0)
    assert fn(reuse,x,None,None,0,2,0) is fresh
    assert state.topk_idxs is fresh


def test_single_pass_mhc_delays_pre_coefficients_to_next_sublayer(torch):
    events = []
    incoming = torch.tensor([1.])
    attention_pre = torch.tensor([2.])
    ffn_pre = torch.tensor([3.])
    def mixes(x, fn, scale, base):
        return (attention_pre if fn == "attention" else ffn_pre), None, None
    def collapse(x, coefficients):
        events.append(coefficients.item())
        return x
    args = SimpleNamespace(hc_attn_fn="attention", hc_attn_scale=None, hc_attn_base=None,
                           hc_ffn_fn="ffn", hc_ffn_scale=None, hc_ffn_base=None,
                           hc_mixes=mixes, hc_pre=collapse, attn_norm=lambda x:x,
                           attn=lambda x,*a:x, hc_post=lambda x,*a:x,
                           ffn_norm=lambda x:x, ffn=lambda x,*a:x)
    _, outgoing = upstream("Block.forward", torch)(args, torch.zeros(1), 0, incoming, None)
    assert events == [1., 2.]
    assert outgoing is ffn_pre


def test_compressed_cache_index_reads_before_in_place_rope_and_cache_read_after_write(torch):
    state = SimpleNamespace(compress_kv=torch.full((1,2,2), -999.))
    latent = torch.tensor([[[2., 4.]]])
    cache = torch.zeros(1,2,2)
    events = []
    def indices(x, qr, value, start, offset, length):
        assert value is latent
        assert value.tolist() == [[[2.,4.]]]
        events.append("index_unrotated")
        return torch.tensor([[[offset]]])
    def rotate(value, freqs):
        events.append("rope")
        value.add_(10.)
    def quantize(value, group, inplace, **kw):
        assert group == 16
        assert kw["scale_dtype"] == torch.float8_e4m3fn
        events.append("fp4")
    args = SimpleNamespace(compress_ratio=2, is_kv_source=True, compressor=lambda x,p:latent,
                           compress_kv_cache=cache, _compress_topk_idxs=indices,
                           freqs_cis=torch.ones(4,1), rope_head_dim=2)
    fn = upstream("Attention._compress_kv", torch, shared_attn=state,
                  apply_rotary_emb=rotate, fp4_act_quant=quantize)
    values, indices_out = fn(args, torch.zeros(1,2,2), None, 0, 2)
    assert events == ["index_unrotated", "rope", "fp4"]
    assert values.tolist() == [[[12.,14.]]]
    assert state.compress_kv is cache
    assert indices_out.tolist() == [[[2]]]


def test_engram_signed_sqrt_gate_and_image_mask(torch):
    # Two hc streams with opposite signed dot products; only the first token is active.
    hidden = torch.tensor([[[[1., 0.], [-1., 0.]], [[1., 0.], [-1., 0.]]]])
    kv = torch.tensor([[[1., 0., 1., 0., 2., 3.], [1., 0., 1., 0., 2., 3.]]])
    args = SimpleNamespace(hc_mult=2, dim=2, eps=1e-20, clamp_value=1e-6,
                           embed=lambda ids:torch.zeros(1,2,1,1), wkv=lambda x:kv,
                           q_weight=torch.ones(2,2), k_weight=torch.ones(2,2))
    output = upstream("Engram.forward", torch)(args, hidden, torch.zeros(1,2,1,dtype=torch.long), torch.tensor([[True,False]]))
    gate_positive = 1 / (1 + math.exp(-math.sqrt(math.sqrt(2))))
    gate_negative = 1 - gate_positive
    expected = hidden.clone()
    expected[0,0,0] += gate_positive * torch.tensor([2.,3.])
    expected[0,0,1] += gate_negative * torch.tensor([2.,3.])
    torch.testing.assert_close(output, expected, rtol=2e-6, atol=2e-7)
    assert torch.equal(output[:,1], hidden[:,1])


def test_indexer_completed_groups_causality_and_candidate_restriction(torch):
    state = SimpleNamespace(index_k=torch.tensor([[[1.], [9.]]]), candidates=torch.ones(1,4,2,dtype=torch.bool))
    args = SimpleNamespace(freqs_cis=torch.zeros(4,1), compress_ratio=2, rope_head_dim=0,
                           owns_k=False, wq_b=lambda x:x, n_local_heads=1, index_head_dim=1,
                           weights_proj=lambda x:torch.ones_like(x), softmax_scale=1., n_heads=1,
                           is_candidate_source=False, uses_candidates=True, index_topk=1)
    fn = upstream("Indexer.forward", torch, shared_attn=state, world_size=1, fp4_block_size=32,
                  apply_rotary_emb=lambda *a:None, fp4_act_quant=lambda *a:None)
    x = torch.ones(1,4,1)
    assert fn(args,x,x,None,0,4).tolist() == [[[-1],[4],[4],[5]]]
    args.index_topk = 512
    short = fn(args,x,x,None,0,4)
    assert short.shape == (1,4,2)  # min(configured512,completed groups2)
    assert short.tolist() == [[[-1,-1],[4,-1],[4,-1],[4,5]]]
    args.index_topk = 1
    # A later indexer scores only source candidate positions even if excluded scores are larger.
    state.candidates[:,:,-1] = False
    assert fn(args,x,x,None,0,4).tolist() == [[[-1],[4],[4],[4]]]


def assert_authority_facts(document):
    config = json.loads((FIXTURE / "config.json").read_text())
    t = config["text_config"]
    facts = document["facts"]
    assert facts["checkpoint_config"] == config
    assert facts["checkpoint_revision"] == REVISION
    assert facts["target_layers"] == t["num_hidden_layers"]
    assert facts["compress_ratios"] == t["compress_ratios"]
    assert facts["kv_source_layers"] == t["kv_source_layer_ids"]
    assert facts["index_source_layers"] == t["index_source_layer_ids"]
    assert facts["csa2_full_layers"] == t["kv_source_layer_ids"]
    assert facts["csa2_reindex_layers"] == [i for i in t["index_source_layer_ids"] if i not in t["kv_source_layer_ids"]]
    assert facts["csa2_reuse_layers"] == [i for i in range(t["num_hidden_layers"]) if t["compress_ratios"][i] and i not in t["index_source_layer_ids"]]
    assert facts["engram_layer_ids"] == t["engram_layer_ids"]
    assert facts["dspark_target_layer_ids"] == t["dspark_target_layer_ids"]


def test_model_facts_match_independent_config_and_wrong_schedule_is_rejected():
    actual = model()
    assert_authority_facts(actual)
    for field in ("kv_source_layers", "index_source_layers", "csa2_full_layers", "compress_ratios"):
        mutated = deepcopy(actual)
        mutated["facts"][field][0] += 1
        with pytest.raises(AssertionError):
            assert_authority_facts(mutated)


def assert_source_leaf_coverage(document, ledger):
    mapped = {target for entry in ledger["entrypoints"] for obligation in entry["obligations"]
              for target in obligation.get("ir_targets", [])}
    operations = document["semantic_contract"]["operations"]
    for view_id, view in document["views"].items():
        for node in view["nodes"]:
            op = operations[node["semantic_op"]]
            assert op["equation"].strip(), node["semantic_op"]
            assert node["semantic_details"]["provenance"], node["semantic_op"]
            if not node.get("drill"):
                assert f"{view_id}.{node['id']}" in mapped, node["semantic_op"]


def test_every_leaf_equation_has_a_source_obligation_and_removal_is_detected():
    actual = model()
    ledger = yaml.safe_load((CATALOG / "semantic_source_ledger.yaml").read_text())
    assert_source_leaf_coverage(actual, ledger)
    # Remove an actual primitive's support everywhere, retaining every display label.
    target = next(f"{v}.{n['id']}" for v,view in actual["views"].items() for n in view["nodes"] if not n.get("drill"))
    mutant = deepcopy(ledger)
    for entry in mutant["entrypoints"]:
        for obligation in entry["obligations"]:
            obligation["ir_targets"] = [x for x in obligation.get("ir_targets",[]) if x != target]
    with pytest.raises(AssertionError):
        assert_source_leaf_coverage(actual, mutant)
    actual["semantic_contract"]["operations"][target]["equation"] = ""
    with pytest.raises(AssertionError):
        assert_source_leaf_coverage(actual, ledger)


def compile_mutant(tmp_path, document):
    from llm_arch_v2 import compile_catalog
    tmp_path.mkdir(exist_ok=True)
    for path in CATALOG.glob("*.yaml"):
        (tmp_path / path.name).write_bytes(path.read_bytes())
    (tmp_path / "model_ir.yaml").write_text(yaml.safe_dump(document, sort_keys=False))
    return compile_catalog(tmp_path)


def test_compound_drill_and_primitive_removal_fail_closure(tmp_path, compiled_catalog):
    from llm_arch_v2.compiler import CatalogError
    actual = model()
    mutant = deepcopy(actual)
    node = next(n for view in mutant["views"].values() for n in view["nodes"] if n.get("drill"))
    del node["drill"]
    with pytest.raises(CatalogError):
        compile_mutant(tmp_path / "drill", mutant)
    mutant = deepcopy(actual)
    # Remove the source-required score primitive AND its edges: failure must come
    # from semantic inventory closure, not merely a dangling graph endpoint.
    view = mutant["views"]["router"]
    node = next(n for n in view["nodes"] if n["id"] == "score")
    view["nodes"].remove(node)
    view["edges"] = [e for e in view["edges"] if "score" not in (e["from"],e["to"])]
    del mutant["semantic_contract"]["operations"][node["semantic_op"]]
    with pytest.raises(CatalogError, match="missing|required|pin"):
        compile_mutant(tmp_path / "primitive", mutant)


@pytest.mark.parametrize("field,value", [("shape", "[B,T,H,H]"), ("dtype", "int8"), ("identity", "wrong_state_owner")])
def test_child_output_axis_dtype_and_identity_mutations_fail(tmp_path, field, value, compiled_catalog):
    from llm_arch_v2.compiler import CatalogError
    mutant = model()
    contract = next(c for c in mutant["boundary_contracts"] if c["boundary_mode"] == "exact_node")
    child = mutant["views"][contract["child_view"]]
    outputs = {n["id"] for n in child["nodes"] if n.get("boundary_direction") == "output"}
    edge = next(e for e in child["edges"] if e["to"] in outputs)
    edge[field] = value
    with pytest.raises(CatalogError):
        compile_mutant(tmp_path, mutant)


@pytest.mark.parametrize("start,window,block", [(1,4,3), (5,4,2), (8,4,5)])
def test_dspark_rows_include_entire_draft_block_without_causal_mask(torch,start,window,block):
    rows = upstream("get_dspark_topk_idxs", torch)(window,2,block,start)
    expected = list(range(min(window,start+1))) + list(range(window,window+block))
    assert rows.tolist() == [[expected]*block]*2
    with pytest.raises(AssertionError):
        upstream("get_dspark_topk_idxs",torch)(window,1,block,0)


def test_dspark_confidence_is_raw_projection_not_probability(torch):
    args = SimpleNamespace(proj=lambda x:x.sum(-1,keepdim=True))
    result = upstream("DSparkConfidenceHead.forward",torch)(args,torch.tensor([[[-3.,-2.]]]),torch.tensor([[[1.]]]))
    assert result.item() == -4.


def test_viewer_dimension_values_match_checkpoint_not_modelargs_defaults():
    config = json.loads((FIXTURE / "config.json").read_text())
    t, v = config["text_config"], config["vision_config"]
    expected = {"H":t["hidden_size"], "R":t["hc_mult"], "V":t["vocab_size"],
                "N":t["num_attention_heads"], "D":t["head_dim"], "Dr":t["qk_rope_head_dim"],
                "Dn":t["head_dim"]-t["qk_rope_head_dim"], "Cq":t["q_lora_rank"],
                "Go":t["o_groups"], "Co":t["o_lora_rank"], "I":t["moe_intermediate_size"],
                "W":t["sliding_window"], "Ni":t["index_n_heads"], "Di":t["index_head_dim"],
                "Ki":t["index_topk"], "Cb":t["candidate_block_size"], "Kcb":t["candidate_topk_blocks"],
                "Eh":t["engram_n_heads"], "Ed":t["engram_head_dim"],
                "Ec":(t["engram_max_ngram_size"]-1)*t["engram_n_heads"],
                "El":len(t["engram_layer_ids"]), "Ng":t["engram_max_ngram_size"],
                "Hv":v["hidden_size"], "Nv":v["num_attention_heads"],
                "Dv":v["hidden_size"]//v["num_attention_heads"], "Iv":v["intermediate_size"],
                "Pv":v["patch_size"], "Rv":v["downsample_ratio"],
                "Dm":t["dspark_markov_rank"], "G":t["dspark_block_size"],
                "M":t["num_nextn_predict_layers"]}
    symbols = model()["dimension_symbols"]
    for name,value in expected.items():
        assert symbols[name]["value_class"] == "model_constant"
        assert symbols[name]["value"] == value, name
    for name,normal,aux in [("E",t["n_routed_experts"],t["dspark_n_routed_experts"]),
                            ("K",t["num_experts_per_tok"],t["dspark_num_experts_per_tok"])]:
        assert symbols[name]["value_class"] == "stage_dependent"
        assert {r["value"] for r in symbols[name]["stage_resolutions"]} == {normal,aux}


@pytest.mark.parametrize("amax", [0.0001, 0.5, 6., 6.01, 12., 100., 448., 449.])
def test_quantization_power_of_two_scale_uses_ceil_not_nearest(torch,amax):
    # Execute the publisher's scalar bit helper with IEEE754 reinterpretation,
    # independently compare with the mathematical ceiling. No packed kernel runs.
    language = SimpleNamespace(
        reinterpret=lambda dtype,x:struct.unpack("I" if dtype=="uint32" else "f",struct.pack("f" if dtype=="uint32" else "I",x))[0],
        Cast=lambda dtype,x:int(x), if_then_else=lambda condition,a,b:a if condition else b)
    logceil = upstream("fast_log2_ceil",torch,source_file="inference/kernel.py",T=language)
    pow2 = upstream("fast_pow2",torch,source_file="inference/kernel.py",T=language)
    scale = upstream("fast_round_scale",torch,source_file="inference/kernel.py",fast_log2_ceil=logceil,fast_pow2=pow2)
    for maximum in (6.,448.):
        assert scale(amax,1/maximum) == 2**math.ceil(math.log2(amax/maximum))


def test_router_and_expert_ir_preserve_source_operator_order_and_float32_intermediates():
    actual = model()
    router = actual["views"]["router"]
    edges = {(e["from"],e["to"]):e for e in router["edges"]}
    # Gate.forward casts projection operands to FP32, and never casts weights back.
    for pair in [("project","score"),("score","select"),("score","gather"),
                 ("gather","normalize"),("normalize","scale"),("scale","weights")]:
        assert edges[pair]["dtype"] == "float32", pair
    assert edges[("select","gather")]["dtype"] == "int64"
    assert edges[("select","ids")]["dtype"] == "int64"
    for name in ("shared_expert","routed_experts"):
        view = actual["views"][name]
        edges = {(e["from"],e["to"]):e for e in view["edges"]}
        for pair in [("gate","gate_clip"),("gate_clip","silu"),("silu","product"),("up","up_clip"),("up_clip","product")]:
            assert edges[pair]["dtype"] == "float32", (name,pair)
        assert edges[("cast","down")]["dtype"] == "bfloat16"
        if name == "routed_experts":
            for pair in [("product","weight"),("weight","cast"),("weights","weight")]:
                assert edges[pair]["dtype"] == "float32", pair
            assert edges[("reduce","output")]["dtype"] == "float32"
        else:
            assert edges[("product","cast")]["dtype"] == "float32"


@pytest.fixture(scope="module")
def compiled_catalog():
    # A malformed baseline cannot make unrelated mutation failures count as proof.
    from llm_arch_v2 import compile_catalog
    return compile_catalog(CATALOG)


def assert_exact_boundary_ports(document):
    fields = ("identity","shape","layout","dtype","state")
    for contract in document["boundary_contracts"]:
        if contract["boundary_mode"] != "exact_node":
            continue
        view_id,node_id = contract["parent_node"].split(".")
        parent = document["views"][view_id]
        child = document["views"][contract["child_view"]]
        for direction in ("input","output"):
            parent_edges = [e for e in parent["edges"] if e["to" if direction=="input" else "from"]==node_id]
            boundary_ids = {n["id"] for n in child["nodes"] if n.get("boundary_direction")==direction}
            child_edges = [e for e in child["edges"] if e["from" if direction=="input" else "to"] in boundary_ids]
            # Fanout repeats the same value, not a new port. Compare every field,
            # never textual shape inclusion or a generated contract union.
            parent_ports = {tuple(e[f] for f in fields) for e in parent_edges}
            child_ports = {tuple(e[f] for f in fields) for e in child_edges}
            aliases = contract.get("port_bindings", {}).get(direction+"s", {})
            assert set(aliases) <= {p[0] for p in child_ports}
            assert set(aliases.values()) <= {p[0] for p in parent_ports}
            original_names = {p[0] for p in child_ports}
            mapped_names = {aliases.get(name,name) for name in original_names}
            assert len(mapped_names) == len(original_names), "boundary aliases must be injective"
            child_ports = {(aliases.get(p[0],p[0]), *p[1:]) for p in child_ports}
            assert parent_ports == child_ports, (contract["parent_node"],direction,parent_ports-child_ports,child_ports-parent_ports)


def test_actual_parent_child_ports_match_without_union_shape_escape():
    assert_exact_boundary_ports(model())


def assert_state_ownership(document):
    config = json.loads((FIXTURE / "config.json").read_text())["text_config"]
    layers = config["num_hidden_layers"]
    kv = config["kv_source_layer_ids"]
    index = config["index_source_layer_ids"]
    candidate = config["candidate_source_layer_id"]
    expected = {
        "global_kv":[{"owner_layer":owner,"consumer_layers":list(range(owner+1,(kv+[layers])[i+1]))} for i,owner in enumerate(kv)],
        "index_k":[{"owner_layer":owner,"consumer_layers":[l for l in index if owner<l<(kv+[layers])[i+1]]} for i,owner in enumerate(kv)],
        "selected_indices":[{"owner_layer":owner,"consumer_layers":list(range(owner+1,(index+[layers])[i+1]))} for i,owner in enumerate(index)],
        "candidates":[{"owner_layer":candidate,"consumer_layers":[l for l in index if l>candidate]}],
    }
    state = document["state_contracts"]
    nodes = {f"{v}.{n['id']}" for v,view in document["views"].items() for n in view["nodes"]}
    for key,ownership in expected.items():
        assert state[key]["ownership"] == ownership, key
        for field in ("producer_nodes","consumer_nodes"):
            assert state[key][field]
            assert set(state[key][field]) <= nodes
        for field in ("initialization","update","validity","reset","alias"):
            assert state[key][field]


def test_actual_state_owners_and_consumers_match_source_not_only_facts():
    actual = model()
    assert_state_ownership(actual)
    for name in ("global_kv","index_k","selected_indices","candidates"):
        mutant = deepcopy(actual)
        mutant["state_contracts"][name]["ownership"][-1]["owner_layer"] -= 1
        with pytest.raises(AssertionError):
            assert_state_ownership(mutant)
        mutant = deepcopy(actual)
        mutant["state_contracts"][name]["ownership"][-1]["consumer_layers"].append(40)
        with pytest.raises(AssertionError):
            assert_state_ownership(mutant)


def eval_scalar_ir_assignment(equation, variables):
    """Evaluate the small authored scalar-expression subset, not labels or facts.

    This is intentionally not a whole-IR evaluator: projection matrices, kernels,
    distributed execution and arbitrary prose are outside its validation scope.
    """
    assignment = equation.split(";",1)[0]
    lhs,rhs = assignment.split("=",1)
    tree = ast.parse(rhs.strip(),mode="eval")
    functions = {"sqrt":math.sqrt,"log":math.log,"exp":math.exp,
                 "floor":math.floor,"ceil":math.ceil,"cos":math.cos,"sin":math.sin,
                 "sigmoid":lambda x:1/(1+math.exp(-x)),"min":min,"max":max,
                 "clamp":lambda x,lo,hi:max(lo,min(x,hi))}
    def visit(node):
        if isinstance(node,ast.Expression):return visit(node.body)
        if isinstance(node,ast.Constant) and isinstance(node.value,(int,float)):return node.value
        if isinstance(node,ast.Name):return variables[node.id]
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.USub):return -visit(node.operand)
        if isinstance(node,ast.BinOp):
            a,b = visit(node.left),visit(node.right)
            if isinstance(node.op,ast.Add):return a+b
            if isinstance(node.op,ast.Sub):return a-b
            if isinstance(node.op,ast.Mult):return a*b
            if isinstance(node.op,ast.Div):return a/b
            if isinstance(node.op,ast.Pow):return a**b
        if isinstance(node,ast.IfExp):
            return visit(node.body if visit(node.test) else node.orelse)
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and not node.keywords:
            return functions[node.func.id](*(visit(a) for a in node.args))
        raise AssertionError(f"Unvalidated scalar expression: {ast.dump(node)}")
    variables[lhs.strip()] = visit(tree)
    return variables[lhs.strip()]


def assert_authored_rope_chain(document, torch, yarn_enabled, inverse=False):
    """Execute the delivered equations, then compare with independent publisher AST.

    This is a targeted RoPE composition regression, not a whole-IR math gate.
    Configuration and expected rotations come from the locked source fixture.
    """
    config = json.loads((FIXTURE / "inference/config.json").read_text())
    dim = config["rope_head_dim"]
    theta = config["compress_rope_theta"] if yarn_enabled else config["rope_theta"]
    positions = (0, 257, 4096, config["original_seq_len"])
    frequencies = upstream("precompute_freqs_cis", torch, math=math)(
        dim, max(positions)+1, config["original_seq_len"] if yarn_enabled else 0,
        theta, config["rope_factor"], config["beta_fast"], config["beta_slow"])
    rotate = upstream("apply_rotary_emb", torch)
    # Includes unchanged, interpolation and fully scaled frequency bands.
    pairs = (0, 16, 20, 31)
    ops = document["semantic_contract"]["operations"]
    for position in positions:
        x = torch.linspace(-0.75, 1.25, dim).reshape(1, 1, dim)
        expected = rotate(x.clone(), frequencies[position:position+1], inverse=inverse)
        for pair in pairs:
            values = dict(theta=theta, Dr=dim, j=pair, pi=math.pi,
                          original_context=config["original_seq_len"],
                          beta_fast=config["beta_fast"], beta_slow=config["beta_slow"],
                          factor=config["rope_factor"], yarn_enabled=yarn_enabled,
                          direction=-1 if inverse else 1, p=position,
                          x_even=x[0,0,2*pair].item(), x_odd=x[0,0,2*pair+1].item())
            for primitive in ("rope.frequencies", "rope.rotate"):
                for statement in ops[primitive]["equation"].split(";"):
                    eval_scalar_ir_assignment(statement, values)
            # Reference uses float32 frequencies; authored scalar math uses double.
            assert [values["y_even"], values["y_odd"]] == pytest.approx(
                expected[0,0,2*pair:2*pair+2].tolist(), rel=2e-5, abs=2e-4)


@pytest.mark.parametrize("yarn_enabled", [False, True])
@pytest.mark.parametrize("inverse", [False, True])
def test_authored_rope_frequency_rotation_chain_matches_publisher(torch, yarn_enabled, inverse):
    assert_authored_rope_chain(model(), torch, yarn_enabled, inverse)


def test_rope_chain_rejects_base_frequency_bypass_and_wrong_policy(torch):
    actual = model()
    mutant = deepcopy(actual)
    op = mutant["semantic_contract"]["operations"]["rope.rotate"]
    assert "direction*p*f_eff" in op["equation"]
    op["equation"] = op["equation"].replace("direction*p*f_eff", "direction*p*f_base")
    with pytest.raises(AssertionError):
        assert_authored_rope_chain(mutant, torch, True)
    mutant = deepcopy(actual)
    op = mutant["semantic_contract"]["operations"]["rope.frequencies"]
    assert "if yarn_enabled else f_base" in op["equation"]
    op["equation"] = op["equation"].replace("if yarn_enabled else f_base", "if True else f_base")
    with pytest.raises(AssertionError):
        assert_authored_rope_chain(mutant, torch, False)


def assert_authored_router_expert_math(document):
    ops = document["semantic_contract"]["operations"]
    # Gate.forward's configured sqrt(softplus(z)) compared at signed test points.
    router_views = [name for name in document["views"] if name.endswith("router")]
    for name in router_views:
        for z in (-20.,-2.,0.,2.,20.):
            score = eval_scalar_ir_assignment(ops[f"{name}.score"]["equation"],{"z_e":z})
            assert score == pytest.approx(math.sqrt(math.log1p(math.exp(z))),rel=1e-7,abs=1e-8)
    # Expert.forward: clip gate above only, clip up symmetrically, THEN SiLU.
    # Expected scalar arithmetic is independently written from the pinned source.
    expert_views = [name for name in document["views"] if name.endswith(("shared_expert","routed_experts"))]
    for name in expert_views:
        for gate,up in [(-20.,20.),(20.,-20.),(-2.,3.),(2.,-3.)]:
            values = {"g":gate,"u":up}
            for primitive in ("gate_clip","up_clip","silu","product"):
                actual = eval_scalar_ir_assignment(ops[f"{name}.{primitive}"]["equation"],values)
            clipped_gate = min(gate,10.)
            expected = clipped_gate/(1+math.exp(-clipped_gate))*max(-10.,min(up,10.))
            assert actual == pytest.approx(expected,rel=1e-7,abs=1e-8)


def test_authored_ir_score_and_expert_equations_numerically_match_source_specification():
    actual = model()
    assert_authored_router_expert_math(actual)
    mutant = deepcopy(actual)
    mutant["semantic_contract"]["operations"]["router.score"]["equation"] = "s_e=sigmoid(z_e)"
    with pytest.raises(AssertionError):
        assert_authored_router_expert_math(mutant)
    mutant = deepcopy(actual)
    mutant["semantic_contract"]["operations"]["shared_expert.gate_clip"]["equation"] = "g_clip=min(g*sigmoid(g),10)"
    with pytest.raises(AssertionError):
        assert_authored_router_expert_math(mutant)


@pytest.mark.parametrize("weight_format",["float4_e2m1fn_x2","float8_e4m3fn"])
def test_fp4_and_fp8_weight_paths_both_quantize_activations_as_fp8(torch,weight_format):
    # Only the Python dispatch contract executes; mock GEMM is not kernel validation.
    weight = SimpleNamespace(dtype=getattr(torch,weight_format),scale=object())
    x = torch.zeros(1,32)
    quantized = torch.zeros(1,32,dtype=torch.float8_e4m3fn)
    scale = torch.ones(1,1)
    calls = []
    def activation(value,block,fmt,dtype):
        assert value is x
        assert (block,fmt,dtype)==(32,"ue8m0",torch.float8_e8m0fnu)
        calls.append("fp8_activation")
        return quantized,scale
    def gemm(a,s,w,ws,dtype,**kw):
        assert a is quantized and a.dtype==torch.float8_e4m3fn
        assert s is scale and w is weight and ws is weight.scale
        assert next(iter(kw.values()))==32
        calls.append("gemm")
        return x
    fn = upstream("linear",torch,act_quant=activation,fp4_gemm=gemm,fp8_gemm=gemm,
                  fp8_block_size=32,scale_fmt="ue8m0",scale_dtype=torch.float8_e8m0fnu)
    assert fn(x,weight) is x
    assert calls == ["fp8_activation","gemm"]


@pytest.mark.parametrize("height,width,channels,ratio",[(2,3,2,2),(4,5,1,3)])
def test_vision_aligner_padding_and_channel_major_patch_order(torch,height,width,channels,ratio):
    image = torch.arange(1.,height*width*channels+1).reshape(height*width,channels)
    expected_blocks = []
    for by in range(math.ceil(height/ratio)):
        for bx in range(math.ceil(width/ratio)):
            block = []
            for c in range(channels):
                for dy in range(ratio):
                    for dx in range(ratio):
                        y,x = by*ratio+dy,bx*ratio+dx
                        block.append(image[y*width+x,c].item() if y<height and x<width else 0.)
            expected_blocks.append(block)
    def first_projection(actual):
        torch.testing.assert_close(actual,torch.tensor(expected_blocks))
        return actual
    args = SimpleNamespace(downsample_ratio=ratio,w1=first_projection,w2=lambda x:x)
    result = upstream("Aligner.forward",torch,source_file="inference/vision.py")(args,image,height,width)
    expected = [[.5*z*(1+math.erf(z/math.sqrt(2))) for z in row] for row in expected_blocks]
    torch.testing.assert_close(result,torch.tensor(expected),atol=1e-6,rtol=1e-6)


def test_vision_rope_has_separate_cos_sin_half_head_tensors(torch):
    height,width,half_head = 2,3,4
    cos,sin = upstream("get_vision_cos_sin",torch,source_file="inference/vision.py")(height,width,half_head,10000.)
    angles = [[position * 10000**(-i/half_head) for position in (y,x) for i in range(0,half_head,2)]
              for y in range(height) for x in range(width)]
    assert cos.shape == sin.shape == (height*width,1,half_head)
    torch.testing.assert_close(cos,torch.tensor([[[math.cos(a) for a in row]] for row in angles]),rtol=1e-6,atol=1e-7)
    torch.testing.assert_close(sin,torch.tensor([[[math.sin(a) for a in row]] for row in angles]),rtol=1e-6,atol=1e-7)


def test_image_token_types_include_row_newlines_and_boundary_tokens(torch):
    fn = upstream("image_token_types",torch,source_file="inference/image_processor.py",IMAGE_START=0,IMAGE=1,IMAGE_NEW_LINE=2,IMAGE_END=3)
    actual = fn(2,3)
    assert actual.dtype == torch.int64
    assert actual.tolist() == [0,1,1,1,2,1,1,1,2,3]
    assert actual.numel() == 2*(3+1)+2


def test_decode_compressor_emits_one_group_but_reads_entire_committed_history(torch):
    cache = torch.zeros(1,8,2)
    cache[:,:2] = torch.tensor([[[1.,2.],[3.,4.]]])
    latent = torch.tensor([[[5.,6.]]])
    state = SimpleNamespace(compress_kv=None)
    def indices(x,qr,value,start,offset,length):
        assert value.shape == (1,1,2)  # newly emitted Ce, not history C
        assert length == 3
        return torch.tensor([[[offset,offset+1,offset+2]]])
    args = SimpleNamespace(compress_ratio=2,is_kv_source=True,compressor=lambda x,p:latent,
                           compress_kv_cache=cache,_compress_topk_idxs=indices,
                           freqs_cis=torch.zeros(8,1),rope_head_dim=2)
    fn = upstream("Attention._compress_kv",torch,shared_attn=state,
                  apply_rotary_emb=lambda *a:None,fp4_act_quant=lambda *a,**k:None)
    values,_ = fn(args,torch.zeros(1,1,2),None,5,4)
    assert values.shape == (1,3,2)
    assert values.tolist() == [[[1.,2.],[3.,4.],[5.,6.]]]


def test_vision_ir_shapes_follow_pixel_patch_and_spatial_boundaries():
    actual = model()
    def edge(view,start,end):
        return next(e for e in actual["views"][view]["edges"] if e["from"]==start and e["to"]==end)
    expected = [
        ("image_preprocessing","input","resize","[IH,IW,RGB]","uint8"),
        ("image_preprocessing","resize","normalize","[RH,RW,RGB]","uint8"),
        ("image_preprocessing","normalize","patchify","[RGB,RH,RW]","bfloat16"),
        ("image_preprocessing","patchify","output","[P,RGB,Pv,Pv]","bfloat16"),
        ("image_preprocessing","types","types_out","[Ts]","int64"),
        ("vision_projector","grid","pad","[Hv,Nh,Nw]","bfloat16"),
        ("vision_projector","pad","unshuffle","[Hv,Nhp,Nwp]","bfloat16"),
        ("vision_projector","unshuffle","project1","[U,Rv*Rv*Hv]","bfloat16"),
    ]
    for view,start,end,shape,dtype in expected:
        actual_edge = edge(view,start,end)
        assert (actual_edge["shape"],actual_edge["dtype"]) == (shape,dtype), (view,start,end)
    rotations = [e for e in actual["views"]["vision"]["edges"] if e["from"]=="position"]
    assert len(rotations)==2
    assert {e["shape"] for e in rotations} == {"[P,1,Dv/2]"}
    assert {e["dtype"] for e in rotations} == {"float32"}
    assert len({e["identity"] for e in rotations}) == 2


def test_ir_distinguishes_emitted_groups_history_and_effective_index_width():
    actual = model()
    def edges(view,start,end):
        return [e for e in actual["views"][view]["edges"] if e["from"]==start and e["to"]==end]
    for view in ("swa","csa2_full","csa2_reindex","csa2_reuse"):
        window = edges(view,"window","gather")
        assert {(e["shape"],e["dtype"]) for e in window} == {("[B,Lw,D]","bfloat16"),("[B,T,Lidx]","int32")}
    for start,end in [("input","rotate"),("rotate","quant"),("quant","write")]:
        assert edges("compressed_cache",start,end)[0]["shape"] == "[B,Ce,D]"
    assert edges("compressed_cache","read","output")[0]["shape"] == "[B,C,D]"
    assert edges("hierarchical_indexer","latent","k_project")[0]["shape"] == "[B,Ce,D]"
    for start,end in [("k_project","k_norm"),("k_norm","k_rotate"),("k_rotate","k_quant"),("k_quant","k_write")]:
        assert edges("hierarchical_indexer",start,end)[0]["shape"] == "[B,Ce,Di]"
    assert edges("hierarchical_indexer","k_write","dot")[0]["shape"] == "[B,C,Di]"
    assert edges("hierarchical_indexer","publish","output")[0]["shape"] == "[B,T,Keff]"
    for symbol in ("Lw","Lidx","Keff","Ce","C"):
        assert actual["dimension_symbols"][symbol]["value_class"] != "model_constant"


def test_weight_activation_ir_keeps_fp8_activation_for_fp4_weight_path():
    actual = model()
    edges = actual["views"]["weight_activation_contract"]["edges"]
    quantized = next(e for e in edges if e["from"]=="quant" and e["to"]=="block_dot")
    assert quantized["dtype"] == "fp8_e4m3"
    assert "FP8 activations" in actual["semantic_contract"]["operations"]["weight_activation_contract.quant"]["equation"]
    assert "activation FP4" not in actual["facts"]["weight_storage"]["routed_expert"]


def test_indexer_publisher_bf16_visible_scores_are_not_float32_accumulators(torch):
    observed = []
    class TorchSpy:
        def __getattr__(self,name):
            return getattr(torch,name)
        def einsum(self,expression,*args):
            result = torch.einsum(expression,*args)
            observed.append(("dot",result.dtype))
            return result
    state = SimpleNamespace(index_k=torch.tensor([[[1.],[9.]]],dtype=torch.bfloat16),candidates=None)
    def candidates(logits,lens,topk,block):
        observed.append(("weighted_masked_scores",logits.dtype))
        return upstream("select_candidate_blocks",torch)(logits,lens,topk,block)
    args = SimpleNamespace(freqs_cis=torch.zeros(4,1),compress_ratio=2,rope_head_dim=0,
                           owns_k=False,wq_b=lambda x:x,n_local_heads=1,index_head_dim=1,
                           weights_proj=lambda x:torch.ones_like(x),softmax_scale=1.,n_heads=1,
                           is_candidate_source=True,uses_candidates=False,index_topk=1,
                           candidate_topk_blocks=1,candidate_block_size=8)
    fn = upstream("Indexer.forward",TorchSpy(),shared_attn=state,world_size=1,fp4_block_size=32,
                  apply_rotary_emb=lambda *a:None,fp4_act_quant=lambda *a:None,
                  select_candidate_blocks=candidates)
    x = torch.ones(1,4,1,dtype=torch.bfloat16)
    result = fn(args,x,x,None,0,4)
    assert result.dtype == torch.int32
    assert observed == [("dot",torch.bfloat16),("weighted_masked_scores",torch.bfloat16)]
    assert state.candidates.dtype == torch.bool


def test_ir_indexer_visible_bf16_scores_and_padded_candidate_block_axes():
    actual = model()
    for name in ("hierarchical_indexer","reindex_indexer"):
        view = actual["views"][name]
        by_pair = {(e["from"],e["to"]):e for e in view["edges"]}
        for pair in [("dot","relu"),("relu","sum"),("sum","causal"),("causal","restrict"),("restrict","topk")]:
            assert by_pair[pair]["dtype"] == "bfloat16", (name,pair)
    candidates = {(e["from"],e["to"]):e for e in actual["views"]["candidate_blocks"]["edges"]}
    for pair,shape in [(('input','pad'),'[B,T,C]'),(('pad','max'),'[B,T,Cp]'),(('max','pin'),'[B,T,Nc]'),(('pin','select'),'[B,T,Nc]')]:
        assert candidates[pair]["dtype"] == "bfloat16"
        assert candidates[pair]["shape"] == shape
    assert candidates[("select","expand")]["shape"] == "[B,T,Nc]"
    assert candidates[("expand","output")]["shape"] == "[B,T,C]"


def test_engram_and_vision_intermediates_match_source_casts_and_splits():
    actual = model()
    def by_pair(view,start,end):
        return [e for e in actual["views"][view]["edges"] if e["from"]==start and e["to"]==end]
    for start,end in [("dot","signed_root"),("signed_root","gate")]:
        assert by_pair("engram",start,end)[0]["dtype"] == "float32"
    assert by_pair("vision_mlp","gate_up","silu")[0]["shape"] == "[P,Iv]"
    for start,end in [("qkv","rope"),("rope","scores")]:
        tensors = by_pair("vision_attention",start,end)
        assert len(tensors)==2
        assert {e["shape"] for e in tensors} == {"[P,Nv,Dv]"}
        assert len({e["identity"] for e in tensors})==2
    assert by_pair("vision_attention","qkv","weighted")[0]["shape"] == "[P,Nv,Dv]"


def test_dspark_effective_index_width_is_distinct_from_full_ring_storage():
    actual = model()
    edges = {(e["from"],e["to"]):e for e in actual["views"]["dspark_attention"]["edges"]}
    assert edges[("indices","sparse")]["shape"] == "[B,G,Jd]"
    assert edges[("join","sparse")]["shape"] == "[B,W+G,D]"
    assert actual["dimension_symbols"]["Jd"]["value_class"] != "model_constant"
