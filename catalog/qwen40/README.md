# Qwen 4.0 IR-first catalog

`model_ir.yaml` is the stable text-inference semantic graph for the
`Qwen4ExpForConditionalGeneration` Day-0 BF16 checkpoint.  It is sourced from
the checkpoint configuration, not reconstructed from one code revision or one
profile.

`execution_paths/tp_only.yaml` remains the default. Separate plans represent
Attention DP + TP MoE and Attention DP + DeepEP + DeepGEMM because their
placement, collectives, payloads, and token-dispatch flow differ.
Commit-specific symbols and measured profiles live in the sibling `bindings/`
and `profiles/` directories and must reference existing execution nodes.

Generation strategy is an orthogonal profile dimension. Autoregressive and
EAGLE-MTP profiles share the same canonical target-model and TP/DP/EP execution
plans; MTP adds a separate generation-control view plus a one-layer auxiliary
head scope. This avoids duplicating every parallel execution path merely to
represent MTP on/off, and prevents reused auxiliary QSA/MoE kernels from being
aggregated into the 48-layer target model.

For the pinned SGLang implementation, the pure-TP overlay makes the following
code-path facts explicit:

- token and PLE N-gram embeddings are vocabulary-sharded and all-reduced;
- QSA's main attention heads are sharded, while its `QSAIndexer.index_qk_proj`
  is a replicated `ReplicatedLinear` on every TP rank;
- attention output is all-reduced before the hyper-connection combine;
- routed and shared MoE hidden dimensions are TP-sharded and use one
  post-expert all-reduce after their local results are combined;
- the vocabulary-sharded LM head is followed by a vocabulary-resolution
  boundary: logits all-gather plus materialization of the full-logit result.

## Execution-path contract

The graph shows semantic data movement rather than treating every collective
as interchangeable:

- Model IR stops at logical module boundaries. TP/DP/EP input/output adapters
  are introduced only by an execution plan and stay outside the Attention or
  MoE semantic module even when the implementation launches them from inside
  that module's Python call;
- module-internal communication is reserved for an operation that implements
  the module's semantics, such as DeepEP expert dispatch/combine. It is marked
  `boundary_role: module_internal`; output materialization such as TP MoE
  all-reduce is a `module_boundary` node;

- pure TP uses all-reduce for partial token/N-gram embeddings, row-parallel
  attention outputs, and TP-sharded MoE outputs; TP vocabulary-logit shards use
  all-gather;
- Attention DP keeps attention and its KV/state local to each DP rank.  The
  DP-to-TP MoE bridge gathers local hidden tokens, runs the TP MoE on the global
  token batch, then returns the owner slice.  The pinned runtime may implement
  the gather-like bridge as all-gather/all-gatherv or as an all-reduce fallback,
  and may implement the return as reduce-scatterv or a local slice.  Each
  profile reports the operation actually observed, while the execution IR
  records the source-supported alternatives;
- DeepEP replaces the TP-MoE bridge with dispatch all-to-all carrying BF16
  activations plus expert IDs/top-k/routing metadata, owner-local DeepGEMM
  expert compute, and combine all-to-all carrying weighted expert outputs plus
  reverse-routing handles.

## Profile contract

The pinned implementation uses a two-stage evidence pipeline:

1. one CUDA-Graph-disabled Torch trace with Python stacks and shapes binds
   runtime kernels to source symbols and stable IR nodes;
2. a second CUDA-Graph-disabled trace without stacks supplies unperturbed eager
   prefill timing.  Semantics transfer by exact, order-preserving kernel-name
   alignment inside each runtime chunk, partitioned at the 96 stable per-layer
   hyper-connection combine delimiters so repeated generic BMM names cannot
   match across distant physical execution regions; no fuzzy-name or duration
   matching is allowed;
3. CUDA-Graph-enabled formal traces at BS 1/16/64/256 attach decode performance
   to the same IR nodes without changing the architecture or execution graph.

For EAGLE MTP, one eager stack/shape trace supplies direct
target-versus-auxiliary scope evidence. Stack-disabled eager timing supplies
prefill BS1, while stack-disabled CUDA Graph traces supply production decode
timing at BS1/16/64/256. Transfer remains exact and order-preserving inside
HC-delimited segments; a timing-only signature is not allowed to cross the
target/MTP scope or collective boundary. MTP timing windows use
generation-level runtime annotations rather than a target-model kernel
signature: prefill spans target EXTEND plus the MTP seed EXTEND, while decode
spans TARGET_VERIFY, accept/commit, one-layer MTP graph replay, proposal-state
commit, and draft selection. This keeps embedding/HC work, auxiliary work, and
inter-stage device gaps inside the reported iteration wall interval.
The request remains ISL/OSL 8192/1024. Server context capacity is 9218 because
draft width 2 requires two internal candidate slots; those slots are not
counted as prompt or requested output tokens.

The MTP profiles use qwen4-main `32e9cb5`, whose upstream change is
`revert sync free mtp (#10)`, plus the content-addressed QSA lifecycle/capacity
patch `07c22e09…`. The patch orders linked-QSA page/request-slot publication
and prices the proven compressed-page upper bound; it does not change model
math, collectives, or formal-window scheduling. Both the older `f90a941aa`
sync-free path and clean `32e9cb5` reproduced illegal QSA memory access under
the exact long-sequence workload, so their partial traces are excluded.

The server disables radix caching and performs one reset before warmup-1.
Warmup×3 and formal×1 then use normal request completion and resource release
on the same server. Repeating `/flush_cache` between requests was rejected as a
catalog protocol because that maintenance path itself invalidates the MTP/QSA
state combination used by the next request; it is neither required for cache
isolation with radix disabled nor representative of serving. Every MTP profile
records this boundary and uses the distinct implementation ID
`sglang_qwen4_main_32e9cb5_qsa_hardening_flashinfer_gdn`.

Prefill and decode are separate profile phases.  Prefill covers one global BS1
8k request.  Pure TP executes it as one 8k forward; DP paths execute the same
request as four sequential 2k runtime chunks, all of which are included in the
reported active GPU time. Non-MTP CUDA-Graph decode skips the first
profiler-perturbed replay and uses the following five steady-state iterations.
MTP eager decode records eight target-verify/draft-extend pairs after the three
request-level warmups and reports the first seven complete start-to-next-start
generation intervals; the trailing pair has no following start marker and is
kept in the raw trace but excluded from wall-time averaging.

MTP semantic traces enable Python stacks, while both semantic and timing runs
retain normal asynchronous CUDA execution. The semantic trace is used only for
stack and IR attribution; every reported duration and overlap
comes from the matching stack-disabled trace. The builder rejects a pair that
does not preserve these protocol boundaries.

MTP decode attribution also enforces the contract-level graph boundary after
the auxiliary head. The sharded vocabulary GEMM belongs only to
`mtp_head.lm_head`; all-gather plus its immediate full-logit copy belong to
`mtp_head.tp_logits_collective`; the eager-proven logits/HC selection and state
writes belong to `mtp_generation.proposal_update`. Index, argmax, fill, and
index-put kernels remain visible in the timeline/detail panel but do not become
framework-specific IR nodes. If this exact post-collective sequence changes,
the build fails rather than sweeping the new kernels into the LM head.

Every profile keeps four timing concepts separate:

- **residency** is the sum of kernel durations, so concurrent streams may be
  counted more than once;
- **active GPU** is the interval union on one reference rank, so overlap is
  counted once and remains the default node heat metric;
- **elapsed** is the same-rank step/request wall interval, or a validated
  module-invocation envelope where such a boundary exists;
- **device gap** is elapsed minus the union of all GPU kernels. A module's
  wider **module gap** is split into other GPU work and true device idle inside
  that module envelope. CPU/synchronization/remote-rank causes remain
unclassified unless the trace contains direct evidence.

Each current Qwen 4.0 profile also owns a content-addressed `timeline.v1`
artifact. Decode contains the five selected formal CUDA Graph replays;
prefill contains the complete measured 8k request. Every timeline event keeps
its real timestamp, duration, raw stream ID, layer/substage context, direct and
roll-up IR targets, attribution method, and confidence. Prefill stacks are
direct same-event evidence (or exact sequence transfers onto the stack-disabled
timing trace). CUDA Graph events explicitly label their Python stack as exact
kernel+IR or representative-IR evidence transferred from the eager trace; they
never claim that the graph replay itself recorded a Python stack.

The Timeline view draws all observed streams and a separate union-derived
device-idle lane. Its selected-step summary shows elapsed, active GPU,
residency, overlap, and device gap together, so a kernel sum is never presented
as wall time. Architecture and timeline selections are bidirectional. The
Perfetto action uses the raw trace SHA256 plus the selected event's absolute
timestamp/duration to open an exact forensic range when the local trace helper
is running.

The Viewer reports exact request-level elapsed/active/device-gap/busy values
for prefill and exact step-level values for decode. CUDA-Graph decode has
validated step/layer invocation boundaries, so its timed module nodes also
show elapsed, other-GPU work, and device idle. The current eager-prefill trace
does not carry an explicit numeric layer invocation marker and logical layers
interleave across streams; therefore only its decoder-stack envelope receives
module elapsed/gap. Prefill child nodes show active/residency and explicitly
state that elapsed is unavailable instead of deriving a false boundary from
timestamp order.

`models/qwen40/build/build_qwen40_topology_cudagraph_profile.py` is the
decode builder. It requires all four traces, validates exact admission and
DP-local DECODE versus zero-request IDLE steps, reuses the collective sequence
proven by the eager stack trace, and attributes every replay kernel.  Node time
is overlap-aware active GPU time from one coherent critical reference rank;
kernel residency is retained separately.  No node is left visually blank: a
node either has a measured timing or an explicit structural/fused/state-only
status.  Topology bindings use `extends` to inherit common source mappings from
the pinned pure-TP implementation and override only the nodes introduced or
changed by the execution plan.

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
