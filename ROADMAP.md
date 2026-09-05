# llm-arch-reviewer roadmap

## North star

Turn a model implementation and its serving evidence into a reproducible
optimization gap analysis:

```text
model/config/source
  -> framework-independent Model IR
  -> topology-specific Execution Plan / Execution IR
  -> graph-off eager semantic reconciliation
  -> graph-on production timing
  -> cross-framework and hardware-SoL gap analysis
  -> ranked optimization hypotheses
  -> measured validation
```

The product is not merely a diagram generator. It should answer, with linked
evidence: what the model computes, how a framework executes it, where the time
goes, why implementations differ, and which gap is realistically recoverable.

## Product principles

1. **One pipeline and one viewer.** Models and frameworks extend catalog data
   or adapters; they do not add alternate viewers or manual publishing paths.
2. **Stable semantics, versioned evidence.** Model IR owns mathematical data
   flow. Execution IR owns parallelism and communication contracts. Bindings,
   eager traces, production traces, and SoL projections remain distinct.
3. **Fail closed.** Unknown kernels, stale bundles, copied fused timing,
   unresolved shapes, or an invalid capture window block release instead of
   becoming plausible-looking UI.
4. **Optimize the critical path.** Residency, active union, elapsed time, idle,
   overlap, communication, and CPU/runtime gaps remain separate quantities.
5. **No unevidenced SoL.** Physical lower bounds and calibrated attainable
   projections are labeled separately, with coverage and uncertainty visible.

## M0 — Trustworthy release baseline

**Goal:** every published model can be regenerated and audited by the same
model-neutral command; the published Viewer is an exact, tested consequence of
the catalog and immutable evidence.

Already present:

- one V2 compiler and one shared Architecture/Timeline Viewer;
- Model IR shape/dtype/state and semantic-closure validators;
- Execution fingerprints, binding/profile acceptance, fusion ownership rules,
  timeline attribution checks, and real-browser interaction audits;
- six public catalogs and 77 checked-in timeline artifacts.

Work in this milestone:

- [x] Add a unified static release audit that recompiles every catalog, compares
  the exact compiled object with the published bundle, verifies public model
  inventory, validates content-addressed timelines, and fails on unexplained
  production kernels.
- [x] Make the audit emit one machine-readable `release-audit.v1` report and
  explicitly distinguish `static_gate`, `browser_gate`, and `release_ready`.
- [x] Close the attribution debt exposed by the new gate. All six public
  catalogs now pass fail-closed Timeline attribution; GLM-5.2 and
  GLM-5.3-Flash use exact semantic classification rather than generic support
  labels or copied rollup timing.
- [x] Run all interaction, compact/physical-stream, framework-comparison, and
  HiDPI rendering checks behind the same `--level release` command in CI.
- [x] Add a concise published acceptance summary containing source revision,
  execution fingerprint, profile contract, evidence hashes, coverage, and
  audit result for every model/framework/profile.
- [x] Keep the compiler and release gate model-discovered rather than
  model-name-dispatched, and record a stable release identity for the compiler,
  viewer, each catalog/bundle/evidence set, and the complete published model
  set. Model-specific browser fixtures remain explicit acceptance scenarios,
  not runtime dispatch logic.
- [x] Add a model-discovered, four-gate independent-evidence contract. Semantic
  IR, Execution Contract, eager Binding reconciliation, and graph-on production
  evidence now declare disjoint authority classes; CI rejects circular
  authority/subject reuse, uncovered artifacts, unlocked binding revisions, and
  model-specific assertions without their immutable provenance.

M0 exit criteria:

- every public catalog compiles deterministically and matches its bundle;
- every public catalog passes its independent-evidence and anti-self-validation
  contract, with machine-resolved and externally attested claims distinguished;
- every selected semantic GPU event is mapped or explicitly typed runtime work;
- all Model IR views close mathematically over shape, dtype, state, and ports;
- every non-leaf rollup has valid ownership and no fused timing is copied;
- every published profile matches its exact phase, graph mode, rank evidence,
  workload, source, hardware, and timeline hash;
- real-browser Architecture/Timeline, fusion-link, comparison, synchronized
  navigation, physical/compact stream, and HiDPI interaction tests all pass;
- `release_ready` is true only when both static and browser gates pass.

## M0.5 — Deterministic add-trace contract

- [x] Add executable run, plan, Binding-revision, eager-reconciliation,
  production-attribution, and acceptance schemas.
- [x] Resolve the checkpoint artifact and immutable revision against the
  independently authored Model IR source lock before Execution matching.
- [x] Make config-to-Execution resolution authored and deterministic: zero
  matches means a new Execution is required; multiple matches are an error.
- [x] Content-address Runtime Implementation Identity from immutable source,
  container, package, extension, backend, and build artifacts. Function names,
  stacks, and kernel names remain versioned Binding content.
- [x] Reopen, machine-compare, and hash-seal the eager and production capture
  protocols plus the concrete source/checkpoint/container/package/extension/
  build/effective-config evidence behind Runtime Identity.
- [x] Require exact manifest/plan identity, graph-off eager rule closure on
  every TP rank, observed-predicate and transfer-signature equality, graph-on
  production transfer, compiled Model IR identity and target existence, explicit fusion ownership,
  controlled typed support work, selected-window hashing, complete event/duration
  accounting, and content-addressed final input-document closure.
- [x] Require an authoritative production capture timestamp, carry it through
  acceptance, and render trace time/provenance in the bounded comparison picker.
- [x] Provide `scripts/run_pipeline_v2.py plan|accept` as the working interface.
  Require `materialize_binding_revision.py` to verify the exact accepted
  revision before emitting a schema-valid catalog Binding, without stale
  template inheritance and with source links rebuilt from accepted rules.
  M1 extends it with resumable capture/parse/map/materialize producers; it does
  not replace or weaken these gates.

## M1 — One-command model and framework onboarding

**Goal:** a new model/framework is produced by adapters and manifests, not by
editing generated JSON or viewer code.

- Implement `run_pipeline_v2.py` as a resumable stage DAG with immutable inputs,
  content hashes, cached outputs, and per-stage acceptance reports.
- Define model, framework, backend, hardware, and workload adapter interfaces.
- Automate source/config extraction, semantic-ledger creation, candidate
  Execution Plan generation, eager reconciliation, production trace capture,
  bundle compilation, and browser QA.
- Replace every eligible `immutable_external_attestation` with an
  adapter-produced, content-addressed evidence extract whose source locator and
  value are machine compared with the assertion. Retain manual attestation only
  for an upstream source that cannot be ingested in CI, and keep that weaker
  assurance visible rather than presenting it as machine verification.
- Generate a review packet that isolates semantic changes, execution-contract
  changes, unresolved evidence, and rendering screenshots.
- Prove the generic path with at least two model families and two frameworks.

Exit: onboarding a supported model requires a manifest plus narrowly scoped
adapter data, and the pipeline refuses to publish any unresolved stage.

## M2 — Cross-framework gap analysis

**Goal:** compare SGLang, vLLM, TensorRT-LLM, and future runtimes on the same
semantic and workload contract.

- Align profiles by model revision, quantization, topology, phase, stable batch,
  ISL/OSL, graph mode, hardware, and scheduler policy.
- Compute per-node critical-path deltas, idle/launch gaps, communication costs,
  overlap, fusion differences, and kernel-family changes.
- Add synchronized comparison navigation, linked evidence, confidence labels,
  and regression detection across revisions.
- Separate framework gaps from backend, kernel, scheduler, and topology gaps.

Exit: every reported delta is traceable to matched profiles and shared IR
contracts; unmatched comparisons are visibly rejected.

## M3 — Physics-based SoL and attainable projection

**Goal:** quantify the distance from measured execution to a defensible physical
lower bound and to calibrated, implementable performance.

- Build transition-, operator-, communication-, and state-aware cost models from
  hardware specifications instead of fixed efficiency percentages.
- Model Tensor Core instruction families, memory hierarchy, launch cost,
  collectives, quantization, sparsity, and overlap constraints.
- Keep physical ideal, kernel-plan attainable P10/P50/P90, coverage, uncertainty,
  and unsupported regions independently visible.
- Calibrate against microbenchmarks and known silicon results, then validate on
  held-out shapes, models, and hardware generations.

Exit: projections carry provenance and error bars, and held-out silicon error is
reported by operator family, shape regime, and end-to-end critical path.

## M4 — Optimization advisor and validation loop

**Goal:** turn a validated gap into ranked engineering work.

- Generate opportunities for fusion, layout removal, collective replacement,
  overlap, batching, scheduling, kernel selection, and state/cache handling.
- Estimate recoverable critical-path time, confidence, engineering scope, and
  correctness risks; never sum overlapping opportunities as independent wins.
- Produce framework-specific code anchors and a benchmark/validation plan.
- Re-ingest the optimized trace and close, reject, or refine the hypothesis.

Exit: recommendations are evidence-backed, measurable, and automatically
reconciled with the resulting implementation.

## M5 — Continuous optimization platform

**Goal:** make the system useful for ongoing model and framework development.

- Versioned model/framework/hardware registry and artifact storage.
- Scheduled profile refresh, source-drift detection, regression alerts, and
  comparison dashboards.
- Review/approval workflow for semantic and execution-contract changes.
- Exportable reports and APIs for CI, performance teams, and upstream PRs.

## Scorecard

The roadmap is measured by evidence quality, not the number of diagrams:

- semantic closure and shape/dtype/state closure;
- mapped semantic kernel count and residency coverage;
- full-rank and eager-to-production reconciliation coverage;
- deterministic bundle and artifact-hash closure;
- real-browser interaction and rendering pass rate;
- matched cross-framework profile coverage;
- SoL adapter/calibration coverage and held-out prediction error;
- measured realized speedup versus predicted recoverable gap.

## Immediate execution order

1. Keep the M0 release gate mandatory and regenerate the published acceptance
   ledger whenever catalog, compiler, Viewer, or evidence changes.
2. Begin M1 with the resumable, manifest-driven stage DAG while preserving the
   same fail-closed compiler, attribution, and real-browser release contracts.
3. Prove M1 using one existing catalog refresh and one net-new model without
   adding a second Viewer or a model-specific publishing path.
