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
- a one-layer full-attention + MoE MTP draft head with shared target embedding/head,
  its own semantic attention/MoE scope, and an explicit one-layer draft K/V cache;
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
- target and MTP-draft routed activations use separate pack → variable-size dispatch →
  return → restore boundaries, each with explicit group, payload, result, dtype, and
  layout contracts;
- the shared expert remains replicated and request-local;
- target and draft wire encodings are deliberately not inherited from one another;
  engine-specific wire compression, kernel fusion, CUDA Graphs, padding, and collective
  backends must be bound independently in implementation profiles.

For the pinned SGLang baseline, source inspection proves that Attention DP4 leaves each
rank's tokens local and the selected A2A dispatcher performs the cross-rank expert
routing; there is no separate DP all-gather/reduce-scatter boundary in this normalized
path. The target uses FlashInfer A2A while the MTP draft uses DeepEP, which is why the two
logical scopes remain distinct even when both normalize to EP4 variable-size dispatch
and return.

The required profile parameters are therefore `tp_size=1`, `dp_size=4`, `cp_size=1`, and
`ep_size=4`. Framework flags that reuse a four-rank tensor-parallel process group while
enabling DP attention are recorded later as binding evidence, not mislabeled as TP4 model
execution.

## Implementation bindings and immutable profiles

Both bindings compile against the same normalized execution fingerprint,
`exec_25a414805d12fed3`:

- SGLang source commit `85c23c62fdc58a5a0c3b7c6d61a7bba720a6cbbf`, with the
  measured runtime overlay identified separately as `a31c1e52e947bcbdd0d551c5e2323e96a9bf303b`;
- TensorRT-LLM source commit `1cef02e901be43081b1ba6d4981e94ed3bd9c1e8`.

The binding files keep target and draft MoE contracts separate. In the measured SGLang
path the target uses FlashInfer A2A plus CuTeDSL NVFP4 expert compute, while the draft uses
DeepEP low-latency plus DeepGEMM. The TRT-LLM target uses CUTEDSL MoE EP4, while the
quantization-excluded MTP experts use a CUTLASS BF16 fallback. These are implementation
facts, not architecture mutations.

Nine official profiles are published under `profiles/attention_dp4_moe_ep4/`, and every
one has `generation_mode: mtp`:

| Engine | Profile | Capture identity | Critical GPU wall |
|---|---|---:|---:|
| SGLang | eager prefill attribution | job `3207730` | 1030.853 ms (attribution only) |
| SGLang | eager decode attribution | job `3208209` | 654.552 ms (attribution only) |
| SGLang | one-chunk 8K target prefill | job `3207938` | 367.889 ms |
| SGLang | CUDA-Graph global-BS32 decode | job `3204736` | mean 18.510 ms |
| SGLang | real A-Z97/C704 steady decode | job `3205969` | mean 55.478 ms |
| SGLang | worker-local NSYS exact rank-local BS32 | job `3256437` | mean 59.002 ms |
| SGLang | strict 8K/1K C704 worker-local Torch/Kineto rank-local BS32 | job `3270073` | mean 30.644 ms |
| TRT-LLM | exact one-request/8K prefill | job `532540` | mean 374.482 ms |
| TRT-LLM | strict 8K/1K C704 worker-local Torch/Kineto rank-local BS32 | job `553916` | mean 23.832 ms |

All selected kernel intervals are retained and classified as mapped, evidence-backed
fusion, or explicit unmapped with candidates and a reason.  Those three classes close
to 100%; that closure is not a claim of 100% precise attribution.  The semantic gate is
evaluated against the overlap-safe mapped/fusion active union (95% for SGLang and 90%
for TRT-LLM), while strict-signature and residency coverage remain separate diagnostics.
Each profile references a deterministic compressed timeline with wall, active GPU union,
residency, overlap, idle/gap, per-node elapsed/active/module-gap/other-work, streams,
fusion groups, layer identifiers, and MTP rounds. One reference rank is displayed; all
four ranks are validated and parallel rank residency is never summed.

The exact one-chunk 8K prefill profiles remain a small-sample descriptive pair with a
TRT/SGL wall ratio of 1.0179x. The new decode pair is the strict cross-engine comparison:
both jobs completed the same 704 unique requests at exactly 8192 input and 1024 output
tokens, use 3P+2D GB300 workers, Attention-DP4/MoE-EP4, MTP6, rank-local BS32, forced
mean accept length 4.8, stream interval 30, CUDA Graphs, and worker-local Torch/Kineto
CPU+CUDA capture with stack enabled and shape recording disabled. Each side contributes
five time-spread samples from one representative rank on each decode worker (10 total);
parallel ranks/workers are never summed. SGLang's selected histogram is exactly two AL4
plus eight AL5 samples; TRT-LLM exposes the immutable AL4.8 simulator setting but no
per-step accepted-length histogram, so its acceptance evidence is configuration-bound.

Under that frozen contract, SGLang step wall is 30.644 ms mean / 29.517 ms median and
TRT-LLM is 23.832 ms mean / 22.622 ms median. The mean ratio is 1.2858x (SGLang 28.6%
slower, +6.812 ms). This is a matched decode-step comparison, not an end-to-end throughput
claim. SGLang reaches 96.19% mapped-or-fusion active-union attribution; TRT-LLM reaches
90.07%. TRT's Kineto trace does not carry executor GPU annotations, so its CUDA Graph
events are reconstructed fail-closed by matching the Nth stable graph-node occurrence to
the Nth Python `_forward_step`; concrete profiler kernel symbols and unmapped intervals
remain visible. The older SGLang global-BS32 and NSYS profiles remain historical scopes
and are not selected by default.

## Evidence boundaries

- TRT outer-wrapper Nsys job `502606` is excluded because MPI/UCX failed before serving.
  Worker-local smoke `531997` proved Python worker, NVTX, CUDA-kernel, step, and rank
  visibility before formal job `532540`.
- The TRT worker-local `_forward_step` range exposes target/draft execution and KV/GDN
  commit kernels, but not accept/sample or token publication. Those lifecycle nodes are
  explicitly `unobserved`; no generic kernel is guessed.
- The SGLang traces expose accept, accepted-prefix GDN replay, GDN commit, and token commit.
  KV commit has no uniquely attributable standalone CUDA interval and is explicitly
  `unobserved`.
- The eager profiles establish Python-stack attribution. CUDA-Graph profiles transfer
  evidence only through exact kernel+IR or declared containing-scope relationships.
- Container SHA256 values in the bindings and profiles were computed from the actual
  runtime `.sqsh` contents, not from image tags.

## Rebuild

```bash
python3 models/qwen35/build/build_qwen35_ir.py
python3 scripts/build_v2.py --model qwen35
python3 scripts/export_standalone.py \
  --model qwen35_v2 \
  --output docs/qwen35_v2/standalone.html
python3 -m pytest -q \
  tests/test_qwen35_ir.py \
  tests/test_qwen35_profiles.py \
  tests/test_qwen35_trace_rules.py \
  tests/test_qwen35_graph_mapping.py \
  tests/test_qwen35_nsys_mapping.py \
  tests/test_qwen35_torch_mapping.py \
  tests/test_trace_mapping_common.py \
  tests/test_timeline_artifact.py \
  tests/test_v2_compiler.py \
  tests/test_v2_handoff_common.py
```

The generator rejects any config whose hash or mandatory Qwen3.5 invariants differ from
the frozen evidence.
