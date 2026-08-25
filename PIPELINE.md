# Canonical IR-first Pipeline

[中文版](PIPELINE.zh-CN.md)

Status: **the only supported pipeline for new models and profiles**.

This document is the source of truth for turning a model definition and runtime
traces into the Architecture and Timeline views. A trace may provide evidence
for an implementation and its timing, but it must never invent or rewrite the
model architecture.

## 1. Design goals

The pipeline must satisfy all of the following:

- **Stable semantics:** the architecture remains understandable when source
  files, helper functions, fusion boundaries, kernels, or serving frameworks
  change.
- **Cross-framework comparison:** SGLang, vLLM, TensorRT-LLM, and future
  implementations can bind to the same Model IR and, when their distributed
  contract is identical, the same Execution IR.
- **Topology-aware execution:** TP, attention DP, CP, MoE EP, DeepEP, and other
  paths are separate execution plans whenever placement, tensor layout,
  collectives, state ownership, or data flow changes.
- **Trace-validated execution and timing:** eager Python stacks, shapes, scopes,
  and collective order validate the proposed Execution IR and establish
  semantic attribution; production traces establish CUDA Graph timing,
  streams, overlap, and idle.
- **Fail closed:** uncertain events remain explicitly unmapped. They are never
  assigned to a nearby node merely to fill a blank.
- **Reproducibility:** every output records source commit, config, execution
  fingerprint, workload, trace hashes, attribution method, and producer version.
- **Data-driven presentation:** the viewer renders compiled metadata. It does
  not contain model-, framework-, or generation-mode-specific routing rules.

## 2. The five persisted contracts

### 2.1 Model IR

Model IR owns code-independent semantics:

- logical operators and module boundaries;
- tensor/state data flow, symbolic shapes, and dtypes;
- repeat/layer structure and stable drill-down hierarchy;
- optional semantic paths such as an MTP auxiliary head.

It is drafted from the model config, model specification or paper, and source
review, then human-reviewed. It is **not generated from a single trace**.

#### Shape notation and visual grammar

Model IR stores three otherwise-confusable concepts separately:

- **tensor layouts** live on data edges and use declared symbolic dimensions,
  for example `[B,T,H] · bf16`;
- **operator transforms** live in structured `operator_signature` metadata and
  render symbolic first and model-resolved concrete second, for example
  `H → E  (2560 → 512)`;
- a node's `shape` field is only its viewer visual class, never a tensor shape.
  The detail panel labels it `visual class`.

Every dimension symbol must be declared in the Model IR `dimensions` mapping.
Labels in overview and drill views may not embed raw transforms such as
`2560 → 640`. When config values resolve a signature, both symbolic and
concrete forms are stored. Every ordinary data edge has shape and dtype;
residual and cache/state read-write edges also carry their tensor contract;
pure scheduling edges are explicitly `control`, not shape-less data edges.

The viewer uses orthogonal visual channels: fill color is operator family;
geometry is compute, tensor boundary, persistent state/cache, or
composite/drillable role; line style is data, residual, cache read/write, or
control; a blue outline is selection only. Profile heat uses a separate heat
bar or halo and never replaces the operator-family color. An accessible legend
is required. The shared compiler/viewer policy owns colors, glyphs, and line
styles; model-specific viewer branches are forbidden.

`operator_signature` and equivalent symbolic/concrete spelling changes are
semantic presentation metadata and do not alter the Execution fingerprint.
Edge shape, dtype, layout, edge role, placement, and collective contracts are
structural: adding a previously missing contract or changing its meaning must
produce a new fingerprint, even when the runtime implementation has not yet
changed.

#### Repeated layers and drill-down

- A repeated layer stack is collapsed by default. It shows the layer count and
  stable schedule instead of duplicating one diagram for every layer instance.
- Each semantically distinct layer type has one representative view. Layers
  with the same structure share that view; for example, linear-attention and
  full-attention layers are separate types, while 36 identical
  linear-attention layers are not expanded individually.
- A representative layer view expands the major stable semantic modules and
  their data flow, such as Attention, MoE/MLP, and residual/HyperConnection.
  A module may drill down further when its internal data flow is
  architecturally meaningful.
- A one-off or layer-specific side path, such as PLE injection, appears
  separately in the stack view. It does not require a duplicated special-layer
  diagram.

Timeline and Profile evidence may retain the actual `layer_id`, expert/rank,
and other invocation context. Navigation from a concrete event opens the
corresponding representative layer/module leaf and shows the actual instance in
the detail panel; runtime instance context does not create duplicate Model IR
nodes.

Implementation helpers, CUDA streams, kernels, fusion, and collectives do not
belong here unless the collective is itself part of the model's mathematical
semantics.

### 2.2 Execution Plan and compiled Execution IR

An Execution Plan describes a framework-independent distributed contract:

- parallelism dimensions and rank groups;
- tensor placement and layout at each module boundary;
- communication operations, payloads, and results;
- state ownership and layout transitions;
- topology-specific additions or replacements to Model IR.

The compiler first applies the plan to Model IR and produces a **candidate**
Execution IR plus a deterministic structural fingerprint. That fingerprint is
computed only from the normalized execution contract; source symbols, Python
function names, kernel names, and trace timestamps are deliberately excluded.
The candidate becomes a validated Execution IR only after Stage 4 reconciles it
against a CUDA-Graph-disabled eager run.

Two frameworks may share an Execution IR when they are observationally
equivalent at canonical IR boundaries: placement, layout, state ownership,
data dependencies, and logical communication results match. Physical
collective algorithms, fusion, kernel sequences, and stream scheduling are
binding/timeline details and do not change the fingerprint. For example, NCCL
all-reduce, a custom two-shot all-reduce, and a reduce-scatter + all-gather
lowering may all implement one `TP output collective` contract when they turn
per-rank partial hidden states into the same replicated hidden state.

A new Execution Plan is required only when an intermediate layout or state
becomes visible across a canonical boundary, is consumed by another module, or
changes observable data flow. If a reduce-scatter result remains sharded and
the next module consumes it before a later all-gather, that is a different plan;
if reduce-scatter + all-gather is merely an internal all-reduce algorithm, it is
the same plan.

Execution IR stays at **contract granularity**. Operations such as local argmax,
global top-k helpers, allocator calls, and framework scheduling helpers belong
in bindings or timeline evidence unless they introduce an architecturally
meaningful layout, communication, or state boundary.

### 2.3 Implementation Binding

A binding is developed in two passes. A draft binding maps source/config
evidence to candidate Execution IR nodes and supplies enough anchors to capture
and interpret the eager run. The binding is finalized only after eager
reconciliation validates the candidate graph. It records:

- framework and version or source commit;
- model/backend configuration;
- canonical Python/C++ symbols and source permalinks;
- eager-stack match rules and stable runtime anchors;
- eager trace hash, validated scopes, observed shapes, collective order, and
  the execution-validation result;
- kernel signatures, fusion groups, and known framework helper scopes;
- proof that the binding implements the selected execution fingerprint.

A binding may map several source scopes or fused kernels to one IR node, and one
kernel may be shared evidence for several IR nodes through an explicit
`fusion_group`. It may not create semantic or execution nodes.

### 2.4 Profile

A profile is an immutable measurement overlay for exactly one:

`model + execution fingerprint + implementation + generation mode + phase + hardware + workload + rank policy`.

It stores per-node timing, provenance, coverage, mapping state, workload, and
raw-artifact hashes. Prefill and decode are always separate profiles. Batch-size
variants are separate measurements, not averaged into a single number.

### 2.5 Timeline artifact

The Timeline artifact stores exact runtime evidence:

- kernel start, duration, device, rank, stream, and correlation identifiers;
- eager stack/source evidence when available;
- mapped Model/Execution IR target and mapping confidence;
- layer/module display lanes and lower-level kernel lanes;
- idle intervals, overlap lanes, synchronization, and collective events;
- links to the raw trace by content hash.

Timeline data cannot redefine either IR.

Generation mode is a profile/binding dimension, **not a sixth IR layer**. For
example, EAGLE MTP uses stable MTP Model IR views, reuses the target-model graph
for verification, and selects an MTP-specific entry view. There is no separate
"Generation IR".

### 2.6 Optional SoL and gap-analysis derivatives

SoL does not add another canonical IR layer. It is a theoretical overlay keyed
by existing Model/Execution IR node IDs:

- `workload-ir.v1` freezes the realized phase, graph mode, batch/sequence
  shape, cache/expert/MTP state, and scheduler facts from the measured profile;
- `cost-ir.v1` is a derived, framework-independent operator contract containing
  resolved problem shape, useful work, compulsory traffic, repetitions, and an
  operator family. It contains no framework symbol or kernel name;
- `transition-plan.v1` is the explicit resource DAG: resources within a
  transition overlap via a max lower bound; only declared dependencies add;
- `kernel-plan.v1` identifies the implementation algorithm, tile, persistence,
  fusion, cache reuse, launches, and synchronization being calibrated;
- `hardware-spec.v1` is a cross-model, per-GPU, sourced and versioned set of
  theoretical ceilings and kernel-plan-identified calibration surfaces;
- `sol-profile.v1` binds one model, execution fingerprint, hardware, workload,
  phase, and assumption set, and stores per-node transition-derived physical
  `ideal_ms`, optional plan-exact attainable P10/P50/P90, limiter vectors,
  coverage, and a dependency critical path;
- `gap-report.v1` compares one immutable measured Profile with its matching SoL
  Profile and separates hardware/shape gap, implementation gap, and currently
  unallocated gap.

The evidence order is transition-derived physical ideal, plan-exact calibrated
projection, then measured silicon. Physical times must satisfy
`ideal <= observed` within tolerance or the physical model is invalid. A
projection requires both exact workload shape and an identical kernel-plan
fingerprint, reports P10/P50/P90, and is invalid when silicon beats P10 outside
tolerance. Without a matching projection, `observed - ideal` is not a framework
implementation gap. Legacy operator-family efficiency/fixed-overhead envelopes
are explicit opt-in sensitivity experiments, disabled by default, and never
populate `attainable_ms`.

## 3. End-to-end flow

```text
model config + specification/paper + source review
                         |
                         v
              draft and review Model IR
                         |
 Model IR + Execution Plan(s) + source/config
                         |
                         v
            candidate Execution IR(s)
                         |
                         +<----------------------+
                         |                       |
                         |             eager semantic run
                         |             CUDA Graph off
                         |             stacks + shapes + order
                         +------ reconciliation
                         |
                         v
 validated Execution IR fingerprint + finalized Binding
                         |
                         v
              production timing run
              real serving mode / CUDA Graph
                         |
                         v
                   Profile + Timeline
                         |
                         v
       validate -> compile -> static viewer bundle
```

The pipeline is deliberately asymmetric: source/config and an Execution Plan
propose the contract; eager evidence verifies **what actually executed and what
each event means**; the production trace answers **when and where it ran**.
Python stacks validate a structural fingerprint but are not hashed into it, so
different frameworks can still prove that they implement the same contract.

## 4. Pipeline stages

### Stage 0 — Freeze a run manifest

Before profiling, record:

- model config and weights identifier;
- source repository, exact commit, and dirty-patch hash if applicable;
- framework/backend versions and launch command;
- hardware, rank topology, parallelism, dtype, quantization, and generation mode;
- phase, requested and realized ISL/OSL, global/local batch size, request rate,
  request ordering, warmup/formal request multipliers, and seed;
- scheduler policy, chunked/mixed-prefill settings, token budget, preemption or
  retraction policy, prefix-cache state, and the framework-native step counter;
- requested artifacts and acceptance level.

The manifest is the single orchestration input. No builder may silently
substitute a different batch size, backend, CUDA Graph mode, or topology.

### Stage 1 — Author and review Model IR

1. Read model config and the stable architecture definition.
2. Build one compact default graph, plus drill views for meaningful repeated
   layer types and stateful modules. Repetition is represented once with its
   placement/count; heads, experts, and identical layers are not cloned.
3. For every architecture-bearing config field, record one disposition:
   Model IR, Execution IR, Binding/Profile evidence, or explicitly excluded.
   Use construction source to resolve and verify semantics, never to copy the
   current framework call graph.
4. Assign stable IDs and symbolic shape/dtype contracts. Store every operator
   transform as structured `operator_signature` metadata, for example
   `H → I  (2560 → 640)`, and every data edge as a complete tensor layout plus
   dtype, for example `[B,T,H] · bf16`. Add structured
   `semantic_details` where they matter: logical operators or formulas,
   parameter shapes/counts and sharing, persistent state/cache lifecycle, and
   placement/optional-path conditions. Keep kernel/backend/fusion details out.
   A Model IR **leaf is a stable mathematical or data-flow primitive**, not an
   implementation fusion boundary. Parameterized transforms, nonlinearities,
   reductions/selections, semantic forks/joins, and persistent-state updates
   must remain individually visible in a drill view. For example, a fused
   `down projection + SiLU + up projection + sigmoid + branch reduction`
   kernel still maps to five Model IR nodes. Pure reshape/view, launch, tiling,
   allocator, and framework-scheduling helpers may remain in Binding/Timeline
   evidence unless they change a tensor, communication, or state contract.
5. Close five ledgers before runtime work starts:
   - **data-flow closure:** each semantic input, output, residual, cache read,
     and state update has an owner and edge;
   - **variant closure:** every distinct layer type and optional model path is
     represented once and its placement rule is explicit;
   - **parameter closure:** derived subtotals reconcile with the declared model
     scope, with tied/shared weights counted once;
   - **state closure:** every persistent tensor has shape, dtype when known,
     lifecycle, update rule, and context-growth behavior.
   - **operator/data-flow closure:** every architecture-bearing transform is a
     visible node or drills into visible primitive nodes; every fused runtime
     interval names one timing owner and all covered Model IR nodes, so timing
     is shared rather than repeated.
6. Persist an external `semantic-source-ledger` against an immutable source
   revision. It must enumerate the declared source files and architecture
   entrypoints, classify every locally declared source member, and attach each
   architecture-bearing obligation to a source anchor. Every obligation is
   one of: Model IR primitive, tensor boundary, state read/update, Execution
   IR, implementation-only, shape-only omission, or training-only exclusion.
7. Run the fail-closed semantic closure audit in both directions. Source→IR
   requires every obligation to map to a primitive target or a reasoned
   exclusion. IR→Source requires every audited leaf to have pinned evidence or
   a reasoned reverse exclusion. Multiple independent primitives mapped to one
   Model IR leaf are a compound-target failure even when one runtime kernel
   fuses them. Source blob OIDs, the ledger, and Model IR are fingerprinted;
   changing any input invalidates the attestation.
8. Review the compact graph, source ledger, and generated gap report before
   editing Model IR or attaching runtime data. A self-declared `complete`
   string in Model IR is not evidence and cannot satisfy this gate.

The default view stays readable; semantic precision belongs in drill views,
while repeated heads, experts, and identical layers remain symbolic rather than
cloned. A formula chain must not be hidden only in prose on a compound leaf.
A topology can therefore be correct yet still fail Stage 1 if its ledgers are
incomplete.

Changing Model IR requires a semantic architecture change or correction. A
framework refactor or kernel fusion is not sufficient.

### Stage 2 — Author Execution Plans

Create the default pure-TP plan first, then add plans only for meaningful,
working code paths. Examples include attention DP, CP, MoE EP, and
attention-DP + MoE-EP with a selected communication backend.

Every inserted communication node must state:

- collective and rank group;
- input payload, shape/layout, and dtype;
- output/result layout;
- module-boundary or module-internal role.

The compiler validates references and emits a candidate structural fingerprint.
Pure TP MoE output reduction, for example, is a module-boundary TP output
operation and must not be hidden inside the MoE semantic module merely because
one framework implements it there. The fingerprint is not accepted as
deliverable until eager reconciliation passes.

### Stage 3 — Create a framework binding

The binding adapter reads the exact source revision and first produces a draft
binding: canonical symbol identities, source links, stack rules, and runtime
anchors. Source AST or callsite validation prevents display aliases from
becoming false canonical identities. Stage 4 then turns this draft into the
final binding by attaching observed eager evidence and validation results.

Framework-specific helpers stay in the binding. Stable execution contracts stay
in Execution IR. This is what allows SGLang and vLLM traces to be compared on
the same graph.

### Stage 4 — Capture eager evidence and validate Execution IR

Capture each distinct code path and phase with CUDA Graph disabled so Python
stacks and operator scopes are available. At minimum:

- prefill and decode are separate;
- each execution fingerprint is captured;
- generation-off and generation-on paths are separate when MTP or another
  auxiliary path changes the invoked modules;
- target-model verification and auxiliary/draft scopes are separately bounded.

The semantic trace records stacks, tensor shapes where available, operator
order, collective order and payload, rank, stream, and exact invocation
boundaries. It is reconciled against the complete candidate Execution IR:

1. every runtime scope must resolve to an existing contract node, an allowed
   framework-helper category, or an explicit unexplained discrepancy;
2. observed placement/layout transitions and collective results must agree with
   the plan;
3. layer multiplicity, optional paths, state updates, and phase/generation
   scopes must agree with the candidate graph;
4. planned nodes that should execute but have no eager evidence are failures,
   unless they are explicitly structural or not selected for that run;
5. unexpected eager scopes may propose a missing or misplaced Execution IR
   contract, but cannot mutate the graph automatically.

Adding or splitting a Model IR drill view requires a separate **mapping
reconciliation**:

- every child declares `measured`, `fused`, `fused_state`, or `structural`; a timed
  parent never implies that its new children are already bound;
- a `measured` child requires both a commit-specific Binding and an
  eager/sequence-validated Profile leaf;
- a fused mathematical primitive or cache/state update points to its real
  timing owner and never copies that owner's time;
- each Timeline event retains the parent roll-up target and adds the finest
  validated leaf target for bidirectional navigation;
- every delivered profile receives the reconciled leaf disposition. A compiler
  may derive an explicit `fused` state from a reviewed Model IR timing-owner
  contract without changing the trace artifact; it may never derive a new
  `measured` interval. If existing evidence cannot uniquely refine the
  interval, keep the parent mapping and do not guess from a generic kernel
  name. Recompute artifact hashes only when the artifact itself changes.
- When a target model, MTP auxiliary head, or another framework reuses one
  stable semantic drill view, runtime attribution remains caller-scoped. If
  that caller has only a parent interval and no child-leaf evidence, show the
  leaves as included in the caller's parent or as separately unattributed;
  never fall back to a same-named leaf measurement from another caller/profile.
  Shared Model IR does not imply shared measurements.

A semantic revision can therefore preserve the Execution fingerprint, but any
new runtime-bearing drill leaf invalidates the relevant Binding/Profile
validation until this reconciliation succeeds.

A mismatch sends the plan back to review. A successful reconciliation seals the
structural fingerprint with an implementation-specific validation attestation
stored in the finalized Binding. The trace does not generate Model IR, and its
framework-specific stack names do not become part of the shared fingerprint.

Stage 4 materializes three reviewable artifacts:

- `observed_execution`: a framework-specific graph derived from eager stacks,
  invocation order, shapes, collectives, and state updates;
- `execution_reconciliation`: an explicit matched/missing/unexpected diff
  between that observed graph and the candidate Execution IR;
- the finalized Binding with a content-hashed validation attestation.

`observed_execution` is the concrete, trace-derived execution graph that a
traditional profiler user might call an execution IR. It is valuable evidence,
but it is not another canonical IR layer: helper scopes and framework call
structure remain implementation-specific, and the Binding is the reviewed map
from this observed graph to the shared contract-level Execution IR.

### Stage 5 — Capture production timing evidence

Use the actual intended serving mode:

- for target concurrency `C`, submit `3 × C` requests in the warmup round and
  `1 × C` requests in the formal round. These are request-count multipliers,
  not model-forward iteration counts;
- in the canonical srt-slurm/sa-bench workload, set
  `random_range_ratio: 1.0`, use a fixed OSL with EOS ignored/disabled, and
  record every realized request length. This keeps request completion aligned
  and makes a stable decode batch possible;
- prefill: eager timing trace with stacks disabled when stack collection would
  distort the measurement;
- decode: CUDA Graph enabled and capture only from a validated formal-round
  steady-state window;
- default decode sweep: global BS 1, 16, 64, and 256;
- record ISL/OSL explicitly, such as 8K/1K;
- MTP profiles must capture both target verification and auxiliary/draft work
  using their real CUDA Graph path.

`random_range_ratio` is not a portable native-CLI contract. The canonical
manifest value above is exact for our common sa-bench workflow; adapters must
not blindly forward it to a framework-native benchmark client. For example,
current vLLM native benchmark code uses `0.0` to mean exact target lengths.
Prefer the common workload generator. If a native client is unavoidable, the
adapter must translate the normalized `fixed_lengths: true` intent, preserve
the requested manifest value, and prove equality from the realized per-request
ISL/OSL.

#### Stage 5A — Baseline run and step selection

Window selection is a mandatory two-run protocol:

1. Run the exact workload once **without profiling or capture**. Keep the
   normal serving configuration, seed, request order, `3 × C` warmup round, and
   `1 × C` formal round. Log the framework-native scheduler step at sufficient
   granularity together with forward mode, running requests, scheduled tokens,
   realized shapes, and batch composition.
2. From that baseline, find contiguous stable intervals for formal prefill and
   formal decode separately. Step numbers are valid only for that exact model,
   framework commit, topology, scheduler configuration, generation mode, and
   workload; MTP and non-MTP must never reuse a window.
3. Choose the middle of each stable interval, with enough guard distance from
   admission, phase transitions, batch drain, and request completion. The raw
   profiler window may span several steps so stability can be verified; the
   canonical profile sample is one representative validated step or one
   explicitly declared rollup from that window.
4. Run the workload a second time with the same trajectory-affecting fields and
   profiling enabled. Only profiler controls and the selected start/stop trigger
   may change.
5. After capture, verify the actual profiled steps again. Baseline step numbers
   are a selection aid, not proof that the second run landed correctly.

The framework adapter converts its native counter (`forward_ct`, scheduler
iteration, executor iteration, or an equivalent monotonic coordinate) into a
`window_selection` artifact containing the baseline log hash, candidate stable
ranges, selected start/stop steps, selection reason, and post-capture evidence.

#### Stage 5B — Stable-step acceptance

A pure prefill sample requires a stable forward mode and stable scheduled
request/token shape, with no decode tokens mixed into the selected invocation.
A pure decode sample requires all selected steps to have:

- actual global/local decode batch size equal to the target concurrency;
- no prefill/extend work mixed into the batch;
- no request admission, completion, preemption, retraction, or KV-cache
  recomputation inside the selected interval;
- stable sequence-length/shape bucket and the requested CUDA Graph state;
- for MTP, the same declared verification/draft configuration and scheduler
  iteration scope.

Do not infer these conditions from the framework name. Scheduler behavior is a
versioned configuration dimension:

- SGLang has prefill admission, chunked-prefill, and mixed-batch paths; the
  exact source revision and flags determine the observed ordering.
- vLLM V1 enables chunked prefill whenever possible and, in that mode,
  prioritizes decode before filling the remaining token budget with prefill.
- TensorRT-LLM in-flight batching can place context- and generation-phase
  sequences in the same iteration, while its capacity scheduler policy affects
  request admission and pausing.

Primary references: [SGLang scheduler source](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py),
[SGLang scheduler arguments](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/server_arguments.md),
[vLLM chunked-prefill policy](https://docs.vllm.ai/en/latest/configuration/optimization/#chunked-prefill),
[TensorRT-LLM in-flight batching](https://nvidia.github.io/TensorRT-LLM/advanced/gpt-attention.html#in-flight-batching),
and [TensorRT-LLM scheduler policies](https://nvidia.github.io/TensorRT-LLM/latest/legacy/performance/performance-tuning-guide/useful-runtime-flags.html#capacity-scheduler-policy).

If the intended native serving mode necessarily mixes phases, preserve it as a
separate `mixed_serving` profile. It is valid for end-to-end scheduler analysis
but must not be labeled or compared as a pure prefill/decode kernel profile. For
architecture/kernel comparison across frameworks, use phase-isolated windows
or a common phase-specific harness with equal realized batch composition.

Apple-to-apple acceptance compares observed values, not just CLI names: same
realized ISL/OSL, target concurrency, scheduled sequence/token counts, phase
composition, generation mode, execution fingerprint, graph mode, and cache
state. Scheduler policy remains visible as a profile dimension rather than
being silently normalized away.

The pipeline must reject a trace whose actual mode, phase, realized shape/batch
composition, graph state, selected step, or formal window differs from the
manifest and `window_selection` evidence.

### Stage 6 — Transfer attribution from eager to production

CUDA Graph traces generally lack Python stacks, so mapping is transferred only
inside validated invocation segments.

Required guards:

1. Match stable segment anchors and execution fingerprint.
2. Match phase, generation scope, layer/module scope, rank, shape bucket, and
   formal-step cardinality.
3. Align exact kernel/event subsequences; use kernel families only after exact
   structural alignment succeeds.
4. Treat collectives, graph boundaries, synchronization, and state commits as
   hard barriers. Attribution cannot spill across them.
5. Preserve order and multiplicity. A greedy nearest-neighbor match is not
   acceptable.
6. Record the eager evidence ID and transfer rule on every mapped production
   event.
7. Leave unmatched or ambiguous events explicitly `unmapped` with a reason.

These rules prevent a node such as `lm_head` from absorbing post-logit helper
kernels across the TP vocabulary collective.

### Stage 7 — Compute timing metrics

For each validated node invocation on one rank:

- `elapsed`: invocation-envelope wall time;
- `active_gpu`: union of attributed GPU intervals, so overlap counts once;
- `residency`: sum of attributed kernel durations;
- `overlap_repeated = residency - active_gpu`;
- `other_gpu_only`: GPU-active time inside the envelope that is not attributed
  to the node and does not overlap its active union;
- `device_idle`: envelope time with no GPU work on any stream.

The disjoint envelope identity is:

```text
elapsed = active_gpu + other_gpu_only + device_idle
residency >= active_gpu
```

Multi-rank profiles preserve per-rank measurements. They never sum rank wall
times. A reported request wall time uses the declared policy, normally the
critical/tail rank, and stores the chosen rank plus the cross-rank distribution.

Parent and child rollups carry `exclusive` or `inclusive` semantics explicitly
and are never mechanically summed when their intervals overlap.

### Stage 8 — Build the Timeline hierarchy

Each physical CUDA stream remains visible. Inside a stream:

- upper IR lanes show stable layer/module-level ownership;
- lower kernel lanes show individual kernels with kernel-family colors;
- overlapping intervals are stacked into additional lanes instead of painted
  on top of each other;
- PDL or other intentional overlap remains visible as overlap, not serialized;
- selecting either an IR interval or kernel navigates to the canonical
  architecture drill path.

Layer/module labels, timeline tiers, colors, and drill paths are compiled from
catalog metadata. The viewer must not infer them from names such as `qsa`,
`eagle_mtp`, or a framework class.

### Stage 9 — Compile the static bundle

The compiler combines only compatible documents:

```text
Model IR
  + matching Execution Plan / fingerprint
  + matching Implementation Binding
  + matching Profile and Timeline
  -> deterministic static bundle
```

The bundle includes a canonical navigation index so Architecture, Timeline,
Split, source links, profile selectors, generation entry views, and deep links
all use the same IDs.

### Stage 10 — Acceptance

A profile is visible as deliverable only after all applicable gates pass.

### Stage 11 — Optional SoL and gap analysis

1. Select the exact execution fingerprint, phase, graph mode, realized
   workload, and hardware spec of the measured Profile, then freeze a Workload
   IR fingerprint.
2. Compile framework-independent Cost IR for each stable node: resolved problem
   shape, useful work, compulsory bytes, communication payload/state traffic,
   repetitions, and operator family. Reject framework symbols or kernel names.
3. Compile Cost IR into a resource-transition DAG. Resources within one
   transition overlap via a max lower bound; explicit dependencies add. Keep
   collective startup and wire transfer in separate transitions.
4. Hardware adapters provide dtype-specific Tensor Core, memory-hierarchy, SFU,
   and interconnect ceilings. Emit `ideal_ms` without empirical efficiencies.
5. Build a versioned Kernel Plan from code review, eager/production traces, or
   microbenchmarks. Algorithm, tile, persistence, fusion, cache reuse, launches,
   and sync events are part of its identity.
6. Emit attainable P10/P50/P90 only when exact workload shape and kernel-plan
   fingerprint match the calibration surface. Launch/sync overheads come from
   actual plan events and evidenced structural models, never arbitrary per-node
   constants. Runtime CPU/scheduler idle remains outside kernel projection.
7. Compute the critical path on the Execution IR dependency DAG. Incomplete
   coverage is a modeled-subgraph critical path, never a full-step SoL.
8. Generate a Gap Report and enforce work/byte conservation, plan identity,
   hardware provenance, exact-shape microbenchmark, holdout,
   measured-faster-than-SoL, and measured-faster-than-P10 gates.

## 5. Adapter boundaries

The reusable engine owns:

- schema and cross-document validation;
- deterministic Execution IR compilation and fingerprinting;
- trace parsing and interval arithmetic;
- exact-sequence attribution transfer;
- profile/timeline generation and bundle compilation;
- acceptance reports.

A **model adapter** owns only model-specific semantic aliases, repeated layer
schedule, state scopes, and truly unique signatures.

A **framework adapter** owns stack formats, annotation conventions, graph
capture conventions, runtime helper scopes, and source-link extraction for one
framework.

An **execution/backend adapter** owns trace recognition for collectives and
backend-specific kernels, but maps them to an already-declared execution
contract.

Adapters may add evidence and aliases. They may not mutate canonical IR or add
special cases to the viewer.

## 6. Catalog and artifact layout

```text
catalog/<model>/
  model_ir.yaml
  execution_paths/<plan>.yaml
  bindings/<framework>-<commit>-<backend>.yaml
  profiles/<execution>/<binding>/<profile>.yaml
  sol_manifests/<workload>.yaml          # SoL adapter inputs for measured profiles
  pipeline.yaml                         # run manifest(s) and requested targets

catalog/hardware/<gpu>.yaml             # shared theoretical ceilings + calibration

schema/v2/                              # executable persisted contracts
src/llm_arch_v2/
  compiler.py
  attribution.py
  metrics.py
  timeline.py
  adapters/
    models/<model>.py
    frameworks/<framework>.py
    backends/<backend>.py

current/<profile-task>/                 # raw traces, logs, intermediate evidence
docs/<model>_v2/                        # generated static bundle only
```

Raw traces and intermediate task material do not belong in the repository.
Catalog documents contain content hashes and resolvable local/artifact
references.

The intended single command is:

```bash
python3 scripts/run_pipeline_v2.py \
  --manifest catalog/<model>/pipeline.yaml \
  --target deliverable
```

`scripts/build_v2.py` remains the compiler for already-prepared catalog data.
The orchestrator above is the next implementation step after this contract is
reviewed; it will call capture, attribution, validation, and compilation stages
without introducing a second pipeline.

## 7. Profile matrix and growth policy

Start with pure TP as the default reference. Add a new profile to an existing
execution fingerprint when only hardware, workload, backend implementation, or
measurement changes. Add a new binding when framework/source/backend code
changes. Add a new Execution Plan when the distributed contract changes.

Recommended progression:

1. pure TP, prefill and CUDA Graph decode BS 1/16/64/256;
2. the most-used or best-performing attention-DP/CP path;
3. MoE EP and the selected communication/GEMM backend;
4. useful combinations such as attention DP + MoE EP;
5. additional frameworks bound to the same Model/Execution IR;
6. optional generation modes such as MTP, using the same contracts and separate
   profile dimensions.

Every distinct code path is profiled independently. Results from one path are
never copied into another merely because their Model IR nodes share names.

## 8. Acceptance gates

### Structural

- All documents pass their JSON schema and cross-document reference checks.
- Model IR IDs are stable and every drill target resolves.
- Every dimension symbol is declared; operator transforms are absent from
  labels; every non-control data/residual/state edge has shape and dtype; and
  the shared legend, operator-family color, glyph, selection, and heat-channel
  visual grammar validates.
- Every architecture-bearing config field has an explicit disposition; no
  field is silently dropped.
- Model IR passes data-flow, layer/optional-variant, parameter, and persistent
  state/cache closure for its declared scope. Any intentionally excluded path
  is named with a reason.
- The generated semantic closure report is `complete`: pinned source digests
  match, all source members and obligations are classified, Source→IR and
  IR→Source closure both pass, and no compound primitive target remains.
- Stable formulas, tensor contracts, weight sharing, state lifecycle, and
  placement conditions are visible in Model IR detail without depending on a
  framework trace.
- The compiled Execution IR fingerprint matches binding and profile.
- The binding contains a passing eager-validation attestation for that exact
  structural fingerprint, source revision, phase, and execution path.
- Every communication node names collective, group, payload, and result.
- No framework/model-specific identifier is required by viewer code.

### Attribution

- Eager semantic evidence exists for every production code path and phase.
- The complete candidate Execution IR has been reconciled against eager stacks,
  shapes, invocation multiplicity, state transitions, and collective order.
- Every measured event is mapped, explicitly fused/shared, framework overhead,
  or explicitly unmapped with a reason.
- Collective ordering and scope boundaries match the Execution Plan.
- Exact-sequence transfer passes; no attribution crosses a collective,
  generation scope, layer invocation, or formal-step boundary.
- Coverage is reported by duration and count. Unknown evidence is never hidden.

### Timing

- Formal iteration boundaries are validated.
- A passing `window_selection` artifact proves that an unprofiled baseline was
  run first and that the second run captured the selected formal steady-state
  interval.
- Warmup/formal request counts equal `3 × C` and `1 × C`; requested and realized
  per-request lengths, scheduled batch composition, and target concurrency are
  recorded and match the profile identity.
- A pure phase profile contains no mixed phase, admission/completion churn,
  preemption, retraction, or recomputation in the selected interval.
- `residency >= active_gpu` and the elapsed-envelope identity hold within
  numeric tolerance.
- Rank policy is explicit; tail/critical-rank wall time is not confused with
  aggregate residency.
- Prefill/decode, eager/CUDA Graph, batch size, and generation mode are visible
  in the profile identity.

### Presentation

- Every IR node shows one of: measured, fused/shared, structural, state-only,
  not selected for this phase, or unmapped-with-reason. Unexplained blank timing
  is a failure.
- Architecture selection highlights all matching timeline events; timeline
  selection expands, centers, and selects the exact architecture leaf.
- Stream, overlap, idle, module wall envelope, active GPU, and residency remain
  separately inspectable.
- Raw trace/Perfetto handoff is content-hash checked.

### Reproducibility

- Source/config/run/baseline-log/window-selection/trace hashes and producer
  version are recorded.
- Rebuilding the same catalog is deterministic.
- At least one second framework or model fixture exercises the generic path
  before an adapter or viewer change is considered reusable.

## 9. Human review gates

Human review is required when:

- Model IR semantics or stable data flow changes;
- a new Execution Plan or boundary contract is introduced;
- an ambiguous eager-to-production transfer cannot be resolved automatically;
- eager reconciliation disagrees with a candidate Execution Plan;
- a new adapter requires a generic contract change.

A new profile with an existing fingerprint and a fully passing exact transfer
does not require architecture review.

## 10. Migration state

The Qwen 4.0 catalog is the current feature reference, and Qwen3.5 V2 remains a
small cross-model structural fixture. The repository already has the five V2
documents and a catalog compiler, but the following work is still required to
fully satisfy this pipeline:

1. implement the manifest-driven `run_pipeline_v2.py` orchestrator;
2. move Qwen-specific extraction logic behind model/framework/backend adapters;
3. replace remaining Qwen/MTP navigation heuristics in the viewer with compiled
   navigation metadata;
4. add the eager-validation attestation contract, then make JSON Schema,
   cross-document validation, and acceptance reports mandatory in the build;
5. add a vLLM binding/profile fixture against a shared execution fingerprint.

The removed Qwen3.5 trace-first/manual pipeline is not a second supported path.
Its useful ideas survive here as frozen inputs, reusable trace parsing,
source/callsite validation, artifact provenance, config-driven validation, and
a single orchestrated command. Its runtime-skeleton-as-architecture behavior
does not survive.
