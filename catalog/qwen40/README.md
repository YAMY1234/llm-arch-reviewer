# Qwen 4.0 IR-first catalog

`model_ir.yaml` is the stable text-inference semantic graph for the
`Qwen4ExpForConditionalGeneration` Day-0 BF16 checkpoint.  It is sourced from
the checkpoint configuration, not reconstructed from one code revision or one
profile.

`execution_paths/tp_only.yaml` remains the default. Separate plans represent
Attention DP, all-reduce expert parallelism, and Attention DP + DeepEP +
DeepGEMM because their placement, collectives, or token-dispatch flow differ.
Commit-specific symbols and measured profiles live in the sibling `bindings/`
and `profiles/` directories and must reference existing execution nodes.

For the pinned SGLang implementation, the pure-TP overlay makes the following
code-path facts explicit:

- token and PLE N-gram embeddings are vocabulary-sharded and all-reduced;
- QSA's main attention heads are sharded, while its `QSAIndexer.index_qk_proj`
  is a replicated `ReplicatedLinear` on every TP rank;
- attention output is all-reduced before the hyper-connection combine;
- routed and shared MoE hidden dimensions are TP-sharded and use one
  post-expert all-reduce after their local results are combined;
- the vocabulary-sharded LM head is followed by the logits all-gather.

## Profile contract

The pinned implementation uses a two-stage evidence pipeline:

1. one CUDA-Graph-disabled Torch trace with stack and shapes binds runtime
   kernels to source symbols and stable IR nodes;
2. CUDA-Graph-enabled formal traces at BS 1/16/64/256 attach performance to
   those existing nodes without changing the architecture or execution graph.

`models/qwen40/build/build_qwen40_cudagraph_profile.py` accepts one TP-rank
formal trace and its benchmark record.  It skips the first profiler-perturbed
graph replay and averages the following five.  The builder refuses a trace if
the exact-decode admission snapshot is missing, a prefill or wrong-batch step
appears, or any replay does not contain the source-proven 36 GDN, 12 QSA,
48 MoE, 98 TP all-reduce, and one logits all-gather structure.  Generic GEMMs
whose node cannot be identified from a unique signature remain unmapped.

`models/qwen40/build/build_qwen40_topology_cudagraph_profile.py` is the
multi-rank counterpart. It requires all four traces, validates DP-local DECODE
versus zero-request IDLE steps, reuses the collective sequence proven by the
eager stack trace, and reports the maximum per-rank kernel residency rather
than incorrectly summing parallel GPU work. Topology bindings use `extends`
to inherit common source mappings from the pinned pure-TP implementation and
override only the nodes introduced or changed by the execution plan.

For DP4 global BS256, 256 independent 8k prefills take longer to admit than
the earliest request needs to finish its 1k output under the stock scheduler.
The BS256 DP overlays therefore use the auditable
`models/qwen40/profile_patches/prefill_first_admission.patch`: it clears only a
stale per-step `batch_is_full` admission marker while queued/chunked prefills
remain and the local population is below 64. The condition is false at the
exact formal gate (local BS64, no waiting work), so no patched branch executes
during the captured steady-state decode graph. Each affected overlay records
the patch SHA256 and this inactive-during-profile invariant.

Generated overlays belong under
`profiles/tp_only/sglang_f90a941aa/`; one file represents one immutable batch
size/workload measurement.  A code update adds a sibling implementation
binding and sibling profile set.  A topology or code-path change such as DP
Attention, CP, or MoE EP adds a separate execution plan rather than mutating
this pure-TP path.
