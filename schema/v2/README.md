# V2 catalog contract

V2 separates stable model semantics from everything that changes with runtime
or implementation:

1. `model-ir.v2` owns semantic nodes, symbolic shapes, and data-flow edges.
2. `execution-plan.v2` derives a topology-specific graph with explicit
   sharding, placement, and collectives.
3. `implementation-binding.v2` attaches commit-specific source symbols and
   kernel signatures to existing execution nodes.
4. `profile.v2` attaches measured values to existing execution nodes.

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
expert dispatch that implements a module's own semantics. Plan version 2 and
newer require every inserted communication/layout node to declare its payload
and result; version 1 remains readable only for existing catalog compatibility.

Bindings may declare `extends: <implementation_id>` when the source commit is
the same but an execution plan changes topology-specific nodes. The compiler
reuses only inherited bindings whose node IDs exist in the derived execution
graph, then applies the derived file's node overrides. This keeps source
identity versioned without duplicating the common model-to-code mapping for
every parallelism path.

Profiles and bindings may never create or mutate semantic model nodes. A new
execution graph is created only when an execution plan changes the structural
fingerprint (operator flow, sharding, placement, or collectives).

Fused kernels remain an implementation/profile overlay. `status: fused` plus
`included_in` is compiled into a `fusion_group` covering two or more stable IR
nodes with `timing_semantics: shared_interval`. The shared interval is valid
evidence for every covered node but is counted only once in timing rollups.

The JSON Schema files document the persisted contract. The compiler also runs
cross-document checks that JSON Schema cannot express, including node-reference
integrity, execution-path compatibility, topology constraints, and deterministic
fingerprinting.
