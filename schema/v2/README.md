# V2 catalog contract

V2 separates stable model semantics from everything that changes with runtime
or implementation:

1. `model-ir.v2` owns semantic nodes, symbolic shapes, and data-flow edges.
2. `execution-plan.v2` derives a topology-specific graph with explicit
   sharding, placement, and collectives.
3. `implementation-binding.v2` attaches commit-specific source symbols and
   kernel signatures to existing execution nodes.
4. `profile.v2` attaches measured values to existing execution nodes.
5. `timeline.v1` stores exact events, streams, idle intervals, and attribution
   provenance for one profile.
6. `validation-evidence.v1` records the independent authorities and executable
   assertions behind the Semantic IR, Execution Contract, eager Binding, and
   production-evidence gates. It prevents a downstream IR, Profile, Timeline,
   or generated bundle from being reused as its own upstream expectation.

The M0.5 add-trace workflow persists six additional stage contracts:

- `add-trace-run.v1` normalizes the raw launch/config inventory into independent
  Model, Execution, Runtime Implementation, Profile, and capture-procedure buckets;
  each disposition preserves the observed raw value separately from the
  evidence-backed normalized value;
- `add-trace-plan.v1` records the unique selector result, Execution fingerprint,
  exact checkpoint-to-Model-IR resolution, pre-capture Execution/runtime Binding
  revision ID, and immutable manifest hash;
- `binding-revision.v1` versions deterministic eager match and production
  transfer rules separately from the Execution IR, seals that rule content
  with `mapping_rules_sha256`, and carries the hash-addressed source,
  checkpoint, container, package, extension, build, and effective-config
  artifacts that prove Runtime Identity;
- `binding-reconciliation.v1` proves every rule on every TP rank with CUDA Graph
  disabled, accounts for every eager kernel as either rule-owned or explicitly
  typed runtime support with count/duration closure, and binds that evidence to
  the exact add-trace plan digest;
- `trace-attribution.v1` accounts for every production event and duration as an
  IR event or typed support event, with explicit fusion ownership and a hashed
  window-selection artifact, and records the authoritative timezone-qualified
  `captured_at`; acceptance additionally resolves every authored
  target against the current compiled Execution IR, while its plan digest and
  rule-level fusion IDs prevent cross-run or same-target rule substitution;
- `add-trace-acceptance.v1` is emitted only after cross-document identity,
  compiled Model IR identity, all-rank, mapping-predicate, transfer-signature,
  fusion, capture-protocol, runtime-evidence, artifact-hash, and
  coverage closure passes; it content-addresses all five accepted input
  documents so an evidence mutation cannot preserve the same attestation. The
  accepted production timestamp is copied to `production_captured_at` and is
  the canonical source for a materialized profile's displayed `trace_time`.

`scripts/materialize_binding_revision.py` consumes both the Binding revision and
its exact acceptance artifact. It verifies their content-addressed identities,
removes stale template inheritance, reconstructs node source links from the
accepted eager rules, persists `add_trace_acceptance_sha256`, and validates the
emitted `implementation-binding.v2`;
materialization is therefore downstream of, not a bypass around, acceptance.

Optional optimization-analysis derivatives do not change the first five
runtime contracts or the sixth validation contract:

- `hardware-spec.v1` stores sourced per-GPU ceilings and optional exact-shape,
  kernel-plan-identified calibration surfaces shared by models;
- `workload-ir.v1` freezes the realized phase, graph mode, batch/sequence
  shape, and scheduler metadata copied from the measured profile;
- `cost-ir.v1` is a framework-independent, per-node resource-demand contract
  (problem shape, useful work, compulsory traffic, repetitions, and operator
  family) derived from Model and Execution IR;
- `transition-plan.v1` expresses serial dependencies and concurrent resource
  lower bounds. Collective startup and transfer are separate transitions;
- `kernel-plan.v1` identifies the concrete algorithm, tile, persistence,
  fusion, cache-reuse, launch, and synchronization plan calibrated by a
  projection surface. It is implementation evidence, not Model IR;
- `sol-manifest.v1` declares workload variables, legal fusion/overlap
  assumptions, node cost adapters, and schedule dependencies;
- `sol-calibration-surface.v1` is the reviewed, content-hashed exact-shape
  microbenchmark input accepted by `scripts/import_sol_calibration.py`;
- `sol-profile.v1` stores the transition-derived physical ideal and optional
  plan-exact attainable P10/P50/P90 projection keyed to an execution and
  workload fingerprint;
- `gap-report.v1` stores measured-versus-SoL comparisons and correctness
  violations.

The compiler never fabricates calibrated values. Legacy operator-family
efficiency envelopes are disabled by default and can only be requested as a
clearly labeled sensitivity experiment; they never populate `attainable_ms`.
A calibration surface without the exact kernel-plan fingerprint fails closed.
Missing operator adapters, kernel plans, or calibration surfaces remain
explicit and prevent a partial subgraph estimate from being labeled a
deliverable projection.

The normative end-to-end workflow, adapter boundaries, and acceptance gates are
defined in [`PIPELINE.md`](../../PIPELINE.md). These schemas are persisted
contracts within that one pipeline, not an independent build path.

Execution IR is intentionally **contract-level**, not a transcription of one
framework's helper functions. Add a visible node when an operation changes
tensor placement/layout, crosses a communication boundary, updates persistent
state, or selects a materially different execution path. Framework-local
primitives such as copies, indexing, local argmax/top-k, temporary allocation,
and fused epilogues remain implementation/profile details under the nearest
contract node. This keeps one Execution IR comparable across SGLang, vLLM, or
another runtime when they implement the same parallel dataflow differently.

Equivalence is evaluated at canonical IR boundaries. NCCL all-reduce, a custom
all-reduce, and an internal reduce-scatter + all-gather lowering can therefore
share one `TP output collective` node when their observable input/output
placement and logical result are the same. A new Execution Plan is required
when an intermediate shard/state crosses a canonical boundary or is consumed by
another module, not merely because the physical collective algorithm differs.

Applying a plan first produces a candidate structural fingerprint. The hash
contains only the normalized execution contract, including exact TP/DP/CP/EP/PP
degrees and generation/control selectors. Each exact implementation must
then reconcile a CUDA-Graph-disabled eager trace—Python stacks, shapes, scope
multiplicity, state transitions, and collective order—against the entire
candidate Execution IR. A passing evidence attestation belongs to the finalized
binding; framework-specific stack names never enter the shared fingerprint.
`generation.mode` is a required exact `equals` selector: materially different
autoregressive and MTP generation graphs cannot share one Execution Plan via
`one_of`. Runtime resolution also requires the selector to constrain every leaf
of the normalized Execution contract, so a new execution-affecting config field
cannot be ignored while reusing an older plan.

Generation/control flow is not a separate IR layer. A
profile declares `generation_mode` (for example `autoregressive` or
`eagle_mtp`) and an `entry_view`. The stable Model IR may therefore expose an
optional auxiliary MTP head and its generation loop once, while TP/DP/EP
execution plan may cover it only when the exact operator/state/placement/
communication contract is already represented. A different draft architecture
or generation control flow requires another Execution Plan; the Model IR views
may still be reused without duplicating architecture semantics.

The compiled bundle preserves both the raw Model IR views and every derived
Execution IR. Compiled nodes expose `ir_origin`; execution-plan insertions also
expose `node_kind` and `boundary_role`. Boundary communication is the safe
default. A plan must opt into `module_internal` for communication such as
expert dispatch that implements a module's own semantics, and every inserted
communication/layout node declares its payload and result.

Bindings may declare `extends: <implementation_id>` when the source commit is
the same but an execution plan changes topology-specific nodes. The compiler
reuses only inherited bindings whose node IDs exist in the derived execution
graph, then applies the derived file's node overrides. This keeps source
identity versioned without duplicating the common model-to-code mapping for
every parallelism path.

A child from a different source commit may inherit only when it also declares
`binding_compatible_base_commit` equal to the immediate base commit. This is an
explicit assertion that the source delta preserves those semantic/operator
bindings; code links are regenerated against the child commit. Without that
field, cross-revision inheritance is rejected.

Profiles and bindings may never create or mutate semantic model nodes. A new
execution graph is created only when an execution plan changes the structural
fingerprint (operator flow, sharding, placement, or collectives).

The compiler additionally derives `comparison-contract.v1` from each profile's
normalized model/workload/hardware/production-mode fields. This identity omits
framework and profiling procedure details, while indexing each implementation's
validated Execution IR fingerprint separately. Consequently, an exact workload
may be compared across different Execution IRs on shared Model IR without ever
collapsing those execution plans. Bindings expose a canonical `framework_id`;
custom repositories must author it explicitly rather than relying on Viewer
name matching.

## Trace-to-IR attribution contract

Profile generation uses two evidence layers without promoting trace accidents
into graph structure:

1. a CUDA-Graph-disabled eager trace with Python stacks, shapes, invocation
   scopes, state transitions, and collective order validates the candidate
   Execution IR and binds source functions to its contracts;
2. the production-mode timing trace (including CUDA Graph replay) supplies
   timestamps, streams, overlap, residency, and wall intervals;
3. timing events inherit eager semantics only through a validated execution
   template: stable module boundaries, collective order/payload, tensor shape,
   and exact reviewed kernel subsequences;
4. a collective is a hard attribution boundary. Adjacent kernels cannot inherit
   the pre-collective module merely because they are nearby in time. Result
   materialization belongs to the communication/layout contract, while later
   selection or state-write primitives belong to the corresponding state
   transition contract;
5. an unrecognized sequence fails closed and requires a fresh eager review.
   Generic or fused kernels are never force-fit solely to reach 100% mapping.

An eager mismatch cannot silently rewrite the plan. It produces a discrepancy,
returns the candidate Execution IR to review, and requires a new structural
fingerprint only if the normalized contract actually changes.

The architecture view therefore exposes stable contracts, while the timeline
and detail panel retain the exact kernel names, Python-stack provenance,
framework-specific helper sequence, and timing. Fusion changes the overlay,
not the stable graph, unless it also changes one of the contracts above.

Fused kernels remain an implementation/profile overlay. `status: fused` plus
`included_in` is compiled into exactly one `fusion_group` covering two or more
stable IR nodes. `shared_interval` means one exact production interval;
`shared_event_set` means an explicitly scoped aggregate of validated production
intervals. Both have one timing owner and are counted only once. A group may
carry `evidence_scope`, `mapping_method`, and `confidence`; production timeline
events preserve the eager event ID plus layer/sublayer `occurrence_id` used for
transfer. The compiler rejects conflicting owners/groups instead of silently
dropping members, and the viewer must distinguish exact occurrence evidence
from a profile aggregate.

The JSON Schema files document the persisted contract. The compiler also runs
cross-document checks that JSON Schema cannot express, including node-reference
integrity, execution-path compatibility, topology constraints, and deterministic
fingerprinting.
