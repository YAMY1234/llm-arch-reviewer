# V2 catalog contract

V2 separates stable model semantics from everything that changes with runtime
or implementation:

1. `model-ir.v2` owns semantic nodes, symbolic shapes, and data-flow edges.
2. `execution-plan.v2` derives a topology-specific graph with explicit
   sharding, placement, and collectives.
3. `implementation-binding.v2` attaches commit-specific source symbols and
   kernel signatures to existing execution nodes.
4. `profile.v2` attaches measured values to existing execution nodes.

Execution IR is intentionally **contract-level**, not a transcription of one
framework's helper functions. Add a visible node when an operation changes
tensor placement/layout, crosses a communication boundary, updates persistent
state, or selects a materially different execution path. Framework-local
primitives such as copies, indexing, local argmax/top-k, temporary allocation,
and fused epilogues remain implementation/profile details under the nearest
contract node. This keeps one Execution IR comparable across SGLang, vLLM, or
another runtime when they implement the same parallel dataflow differently.

Generation/control flow is orthogonal to the parallel execution topology. A
profile declares `generation_mode` (for example `autoregressive` or
`eagle_mtp`) and an `entry_view`. The stable Model IR may therefore expose an
optional auxiliary MTP head and its generation loop once, while TP/DP/EP
execution plans continue to describe only placement, sharding, and
communication. This avoids a TP × DP × EP × MTP cross product of duplicated
architecture graphs.

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

## Trace-to-IR attribution contract

Profile generation uses two evidence layers without promoting trace accidents
into graph structure:

1. a CUDA-Graph-disabled eager trace with Python stacks and shapes binds source
   functions to Model/Execution IR contracts;
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

The architecture view therefore exposes stable contracts, while the timeline
and detail panel retain the exact kernel names, Python-stack provenance,
framework-specific helper sequence, and timing. Fusion changes the overlay,
not the stable graph, unless it also changes one of the contracts above.

Fused kernels remain an implementation/profile overlay. `status: fused` plus
`included_in` is compiled into a `fusion_group` covering two or more stable IR
nodes with `timing_semantics: shared_interval`. The shared interval is valid
evidence for every covered node but is counted only once in timing rollups.

The JSON Schema files document the persisted contract. The compiler also runs
cross-document checks that JSON Schema cannot express, including node-reference
integrity, execution-path compatibility, topology constraints, and deterministic
fingerprinting.
