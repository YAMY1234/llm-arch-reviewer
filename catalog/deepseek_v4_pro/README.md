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
ordered eager-to-production reconciliation. Decode production latency is the
matched profiler-off scheduler wall. The published timeline separately uses the
critical instrumented rank for attribution: its elapsed time is the trace
interval, active time is the cross-stream interval union, and residency is the
sum of kernel durations. Instrumented elapsed is not cross-framework latency
authority.

| Phase | GBS | CUDA Graph | Profiler-off production wall (ms) | Instrumented trace (ms) | Kernel mapping |
|---|---:|---|---:|---:|---:|
| Prefill | 1 | off | unavailable | 507.298 | 100% |
| Decode | 1 | on | 10.930 | 11.179 | 100% |
| Decode | 16 | on | 13.820 | 14.502 | 100% |
| Decode | 64 | on | 19.020 | 20.016 | 100% |
| Decode | 256 | on | 26.790 | 31.430 | 100% |

Every graph-on decode event retains its exact same-rank eager event IDs and
stack/shape evidence. Many-to-one and one-to-many mappings remain explicit,
while each fused physical event set has one timing owner. Semantic non-owners
are marked `fused into <owner>` and navigate to that owner; hierarchy totals are
interval-union rollups rather than additional timing owners. The five timelines
have 100% mapped kernel-count and residency coverage. No attainable projection
is claimed without an exact kernel-plan calibration surface.

## SGLang production contracts

The source-locked SGLang implementation has four accepted pure-TP8 decode
points. Each passed exact all-rank source, mode, phase, shape, selected-window,
formal-step throughput, collective-duration, and ordered eager-to-production
reconciliation. CUDA Graph launch-body events are mapped through exact
same-rank eager IDs; the bounded launch-prefix copies that are absent from the
captured graph body remain explicit runtime support nodes rather than being
hidden or assigned proxy semantics.

| Phase | GBS | CUDA Graph | Status | Profiler-off production wall (ms) | Instrumented trace (ms) | Kernel mapping |
|---|---:|---|---|---:|---:|---:|
| Prefill | 1 | off | unsupported | unavailable | 335.814–336.015 rejected evidence | not published |
| Decode | 1 | on | measured | 9.290 | 9.393 | 100% |
| Decode | 16 | on | measured | 11.748 | 11.504 | 100% |
| Decode | 64 | on | measured | 17.649 | 17.521 | 100% |
| Decode | 256 | on | measured | 29.861 | 29.751 | 100% |

SGLang prefill job `3426447` retained all eight exact synchronized-rank traces,
but the first HCA collective ranged from 0.338 to 7.113 ms across ranks and
failed the fail-closed outlier gate. It is retained as evidence-backed
`unsupported`, not zero-filled or substituted with another model, mode, or
shape.

Across both frameworks, 80/80 eager rank windows pass the fixed contract. The
measured production set closes 72/72 rank windows; the additional eight-rank
SGLang prefill rejection is retained and hash-checked. Both bindings cover all
153 Execution-IR nodes, all 122 layer occurrences close in every measured
profile, and all nine published timelines have 100% mapped kernel-count and
residency coverage. A two-pass rebuild of all 28
generated DeepSeek files was byte-identical with combined tree SHA-256
`b22a30942722b1c53d2b769696191b4e96a8c27c25cb54f740da7dd6ae108971`;
the canonical bundle SHA-256 is
`77fdea325376f7ce2f63a3ce8dd2ed875b469e24a16a19f5ddaf239cf785f2f0`.
The real browser audit exercised all nine published profiles, 171 routes, 1,377
expanded architecture nodes, 839 fused-owner card/detail links, and 27 real
zoom/pan/scroll gestures with
no failure or clipping/overlap report.

Direct Viewer links:

- [DeepSeek V4 Pro canonical Viewer](https://yamy1234.github.io/llm-arch-reviewer/viewer.html?model=deepseek_v4_pro_v2)
- [SGLang decode GBS 1](https://yamy1234.github.io/llm-arch-reviewer/viewer.html?model=deepseek_v4_pro_v2&execution=exec_9208b2a45f2a90e7&implementation=sglang_71de97b_dsv4pro0813_tp8&profile=deepseek_v4_pro_tp8_sglang_cg_decode_gbs001_8k1k&phase=decode)
- [vLLM decode GBS 1](https://yamy1234.github.io/llm-arch-reviewer/viewer.html?model=deepseek_v4_pro_v2&execution=exec_9208b2a45f2a90e7&implementation=vllm_dd10e03_dsv4pro0813_tp8&profile=deepseek_v4_pro_tp8_vllm_cg_decode_gbs001_8k1k&phase=decode)
