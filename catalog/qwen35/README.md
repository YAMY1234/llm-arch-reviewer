# Qwen3.5 397B-A17B IR-first catalog

This catalog is generated from the frozen Qwen3.5 checkpoint configuration and the
matching public SGLang semantic source. It does not use Qwen4 IR, profiles, mappings, or
viewer output as an input.

## Evidence boundary

| Evidence | Identity | Purpose |
|---|---|---|
| `source_configs/config.json` | SHA256 `9408a9e559cc2f05f0b357738213666353e6651160ce8ff477b1c26982bc4f63` | Canonical Qwen3.5 dimensions, exact 60-entry layer order, MoE cardinality, and MTP depth |
| `sgl-project/sglang` | commit [`5be8757c`](https://github.com/sgl-project/sglang/commit/5be8757c0d99f83bde9f5254b1fee30b97dcf66f) | Public source for Qwen3.5 GDN, full-attention, MoE, state-shape, and MTP module semantics |
| Generator | `models/qwen35/build/build_qwen35_ir.py` | Deterministic derivation of Model IR and the framework-independent DEP4 plan |

The three runtime model files captured from SGLang job `3109160` are byte-identical to
that public SGLang commit. Their SHA256 values are:

- `qwen3_5.py`: `0bfeee88a3bc757c76826dcb09bdba8d6d2730b980609a0a9356ecb576fcd9c4`
- `qwen3_5_text.py`: `298d82edf54f1bb3c801d54ba9b5a2a1c1e69d49d393e8730d6d2dc8fc3304ef`
- `qwen3_5_mtp.py`: `263695710ffe8fd80a973e5fcda47fac4bd516a355ec87eaa968430b8ac75953`

## Canonical semantics

The Model IR contains:

- the exact 60-layer sequence: 45 Gated Delta Network layers and 15 full-attention
  layers at zero-based indices `3, 7, …, 59`;
- 512 routed experts with top-10 selection plus one always-on shared expert in every
  decoder layer;
- unsharded semantic state layouts for 15-layer K/V cache, 45-layer GDN convolution
  windows, and 45-layer GDN recurrent matrices;
- a one-layer full-attention + MoE MTP draft head with shared target embedding/head;
- explicit draft → target verify → accept → accepted-prefix GDN replay → KV/GDN commit
  control flow, including a tentative state journal and rejected-suffix discard.

Canonical state dtype follows model/source semantics: BF16 K/V, BF16 convolution state,
and the checkpoint-configured FP32 recurrent state. Runtime choices such as KV FP8 or the
SGLang baseline's BF16 recurrent-state override are implementation/profile facts and do
not mutate this Model IR.

## DEP4 execution plan

`execution_paths/attention_dp4_moe_ep4.yaml` defines one four-rank DEP group:

- attention/GDN weights are replicated and requests plus their KV/GDN state are sharded
  over Attention DP4;
- the 512 routed experts are partitioned over EP4 (128 logical experts per rank);
- routed activations use explicit variable-size dispatch and return collectives with
  logical payload, dtype, and layout contracts;
- the shared expert remains replicated and request-local;
- engine-specific wire compression, kernel fusion, CUDA Graphs, and collective backends
  are deferred to implementation bindings and measured profiles.

The required profile parameters are therefore `tp_size=1`, `dp_size=4`, `cp_size=1`, and
`ep_size=4`. Framework flags that reuse a four-rank tensor-parallel process group while
enabling DP attention are recorded later as binding evidence, not mislabeled as TP4 model
execution.

## Rebuild

```bash
python3 models/qwen35/build/build_qwen35_ir.py
python3 scripts/build_v2.py --model qwen35
python3 -m pytest -q tests/test_qwen35_ir.py
```

The generator rejects any config whose hash or mandatory Qwen3.5 invariants differ from
the frozen evidence.
