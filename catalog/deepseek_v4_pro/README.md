# DeepSeek V4 Pro Model/Execution IR

This catalog entry is pinned to the public
`deepseek-ai/DeepSeek-V4-Pro-0813` checkpoint at revision
`72e1d3230f6c080a530b0a1d46f8eb4602340597`. It does not describe the earlier
Pro preview, DeepSeek V3/V3.2, a generic “DSV4” placeholder, or an internal
checkpoint. `model_ir.yaml` is derived from the public 0813 configuration,
reference implementation, and paper; framework-specific source and runtime
details live only in bindings and profiles.

The stable Model IR closes the 61-layer target model and its attached three-stage
DSpark structure. Stage-1 production profiles use ordinary autoregressive target
generation with DSpark structurally retained but disabled. The architecture has
30 CSA layers (4:1 compression plus learned causal top-k indexing), 31 HCA layers
(128:1 compression plus all-completed compressed history), a 128-token sliding
window in every layer, four-stream mHC transforms around both attention and MoE,
384 routed experts plus one shared expert, three token-hash router layers, and 58
learned-router layers.

## Portable pure-TP contracts

The default stage-1 topology is pure TP8 across two four-GPU GB300 nodes: DP=CP=EP=1,
with attention query heads and projection dimensions sharded across TP ranks and
the KV/compressor/indexer state replicated where required by the implementation.
Both source-locked runtimes use `tp8_moe_intermediate_shard`: all 384 experts
remain logically present on every rank, their gate/up and down weights are
tensor-sharded, and the partial outputs are materialized through TP reductions.
vLLM's separate physical-expert placement path requires expert parallelism and
is therefore not part of this EP=1 stage. Equivalent tensor-parallel all-reduce
semantics are normalized to the same execution fingerprint.

## Evidence pipeline

`pipeline.yaml` is the single acceptance manifest. The required order is:

1. exact checkpoint/config and framework/container source lock;
2. ordinary profiler-disabled baseline to discover stable scheduler windows;
3. graph-off eager semantic traces with Python stacks and shapes;
4. commit-specific binding validation against the eager ordered sequence;
5. stack-disabled stable prefill timing where supported;
6. graph-on production decode timing at global batch sizes 1, 16, 64, and 256,
   with every TP rank present;
7. exact eager-to-production reconciliation, profile/timeline generation,
   deterministic bundle construction, and rendered Viewer interaction audit.

The fixed workload is ISL/OSL 8192/1024, ignore-EOS, request-rate infinity,
random-range-ratio 1.0, 3×concurrency warmup, and 1×concurrency formal requests.
Any mode, phase, shape, rank, source, or selected-window mismatch is rejected.
Raw traces and checkpoint files stay under
`current/deepseek-v4-pro-ir-profile/`; only manifests, bindings, profiles,
timelines, and reproducible hashes belong in this repository.

## Accepted vLLM production profiles

The source-locked vLLM implementation has five accepted pure-TP8 profiles. All
eight TP ranks passed exact source, mode, phase, shape, selected-window, and
ordered eager-to-production reconciliation. The published timeline uses the
slowest selected-window rank for each point; elapsed time is the selected wall
interval, active time is the cross-stream interval union, and residency is the
sum of kernel durations.

| Phase | GBS | CUDA Graph | Critical-rank elapsed (ms) | Kernel mapping |
|---|---:|---|---:|---:|
| Prefill | 1 | off | 507.298 | 100% |
| Decode | 1 | on | 11.179 | 100% |
| Decode | 16 | on | 14.502 | 100% |
| Decode | 64 | on | 20.016 | 100% |
| Decode | 256 | on | 31.430 | 100% |

Every graph-on decode event retains its exact same-rank eager event IDs and
stack/shape evidence. Many-to-one and one-to-many mappings remain explicit,
while each fused physical event set has one timing owner. Semantic non-owners
are marked `fused into <owner>` and navigate to that owner; hierarchy totals are
interval-union rollups rather than additional timing owners. The five timelines
have 100% mapped kernel-count and residency coverage. No attainable projection
is claimed without an exact kernel-plan calibration surface.

## Accepted SGLang production profiles

The source-locked SGLang implementation has the same five accepted pure-TP8
points. Every point passed exact all-rank source, mode, phase, shape, selected-
window, and ordered eager-to-production reconciliation. CUDA Graph launch-body
events are mapped through exact same-rank eager IDs; the bounded launch-prefix
copies that are absent from the captured graph body remain explicit runtime
support nodes rather than being hidden or assigned proxy semantics.

| Phase | GBS | CUDA Graph | Critical-rank elapsed (ms) | Kernel mapping |
|---|---:|---|---:|---:|
| Prefill | 1 | off | 335.843 | 100% |
| Decode | 1 | on | 19.433 | 100% |
| Decode | 16 | on | 48.671 | 100% |
| Decode | 64 | on | 21.943 | 100% |
| Decode | 256 | on | 36.790 | 100% |

Across both frameworks, 80/80 eager and 80/80 production rank windows pass the
fixed contract. Both bindings cover all 153 Execution-IR nodes, all 122 layer
occurrences close in every profile, and all ten production timelines have 100%
mapped kernel-count and residency coverage. A two-pass rebuild of all 31
generated DeepSeek files was byte-identical with combined tree SHA-256
`da662073c644df37c4d1ddfa2c2448fb0fce29781acf0a2177a6f965e4d2aa9a`;
the canonical bundle SHA-256 is
`03e7d36b7596f2f08dd09fbe6712272a450987734242a5a60f15ce0e13a1749d`.
The real browser audit exercised all ten profiles, 190 routes, 1,530 expanded
architecture nodes, 857 fused-owner links, and 30 zoom/pan/scroll gestures with
no failure or clipping/overlap report.

Direct Viewer links:

- [DeepSeek V4 Pro canonical Viewer](https://yamy1234.github.io/llm-arch-reviewer/viewer.html?model=deepseek_v4_pro_v2)
- [SGLang decode GBS 1](https://yamy1234.github.io/llm-arch-reviewer/viewer.html?model=deepseek_v4_pro_v2&execution=exec_6178deeaa361c4f1&implementation=sglang_71de97b_dsv4pro0813_tp8&profile=deepseek_v4_pro_tp8_sglang_cg_decode_gbs001_8k1k&phase=decode)
- [vLLM decode GBS 1](https://yamy1234.github.io/llm-arch-reviewer/viewer.html?model=deepseek_v4_pro_v2&execution=exec_6178deeaa361c4f1&implementation=vllm_dd10e03_dsv4pro0813_tp8&profile=deepseek_v4_pro_tp8_vllm_cg_decode_gbs001_8k1k&phase=decode)
