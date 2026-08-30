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

The default stage-1 topology is pure TP4 on one four-GPU GB300 node: DP=CP=EP=1,
with attention query heads and projection dimensions sharded across TP ranks and
the KV/compressor/indexer state replicated where required by the implementation.
Both source-locked runtimes use `tp4_moe_intermediate_shard`: all 384 experts
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
