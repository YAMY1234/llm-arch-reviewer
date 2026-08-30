# Qwen3.5 V2 catalog

This catalog covers `nvidia/Qwen3.5-397B-A17B-NVFP4-V2` revision
`8f590eae8f10bf55d9a46f79ea0280bde435c9f8`. `model_ir.yaml` is the
framework-independent semantic source of truth: it preserves mathematical and
data-flow operations even when a runtime fuses them, and explicitly records
tensor shape/dtype transitions plus KV, convolution, recurrent, vision, and MTP
state boundaries.

The accepted first-stage execution contract is text-only aggregated pure TP8:
TP8/DP1/CP1/EP1/PP1, MTP off, prefix cache off, two nodes with four GB300 GPUs
per node. It is represented once by `execution_paths/tp8.yaml`. Both framework
bindings compile to execution fingerprint `exec_50bb583c3a3d0557`:

- SGLang package commit `f609d677b909ca46c64bb6803b69a85fedbf86bc`,
  with the runtime Qwen3.5/config/GDN/MoE modules byte-matched to commit
  `033446bb05f35c0943aed2750c443077ffc0b92c`.
- vLLM commit `487ecf187d3dfe74d2cf6119a92881dba403c219`.

The attempted 8K/1K production matrix has **zero accepted profiles and ten
unsupported candidates**. The canonical Viewer therefore publishes the two
validated bindings and shared Architecture IR, but no Qwen3.5 timing profile or
Timeline. `unsupported_profiles.yaml` is the public fail-closed audit: it keeps
the exact SGLang/vLLM, phase, global batch, job, source, hardware, workload,
all-rank trace/eager hashes, CUDA Graph observation, unresolved counts, fusion
closure counts, and validation hashes for every rejected point.

All ten candidates used 3×concurrency warmup requests and 1×concurrency formal
requests. They are not deliverable because each reference rank still has
1,439–2,043 model-bearing production events without exact same-rank,
same-phase, occurrence-scoped eager closure. The graph-off evidence frequently
has a materially different 1:N/N:1 event sequence, and some eager stack rules
are absent or conflict with the production candidate owner. Those events are
not reclassified as runtime support and are not promoted to high-confidence
fusion ownership.

Full fusion groups in the rejected diagnostics are emitted only when member
and owner production event sets are equal and every owner event has same-rank
closure. Other relationships remain occurrence-scoped partial evidence. Since
the partial nodes do not form complete profile-aggregate ownership, no rejected
diagnostic is compiled into the Viewer.

The SGLang prefill candidate has an additional timing-contract failure. Its
profiler-off selector reports 4111.995663 ms for the first formal scheduler
interval, while the instrumented selected forward has 89.595218 ms active GPU
time and a 93.450314 ms model envelope. The retained profiler-controlled request
also stretches to 35.46 s. There is no proof that the 4.112 s interval and the
instrumented interval isolate the same stable forward, so neither number is
published as an accepted profile wall time.

In the diagnostics, `cuda_graph_enabled`/`used_graph_path` refers only to the
selected formal forward and is proven by nonzero raw-trace graph IDs. Server
CUDA Graph configuration is a separate fact. Thus vLLM prefill remains
`used_graph_path: false` with zero graph IDs even though the server configured
`FULL_AND_PIECEWISE`; decode candidates retain their observed graph-path state.

SGLang graph-on decode capture drains each rank's preceding CUDA backlog and
uses Gloo TP barriers after Kineto activation and after formal-forward input
preparation, at the scheduler boundary immediately before
`model_worker.forward_batch_generation`. No synchronization GPU kernel is
added to the selected model interval. The retained diagnostic checks all eight ranks
for exactly 121 logical all-reduce primaries, exact cross-rank signatures,
baseline-relative max-single and mapped-envelope outliers, and robust physical
residency outliers. The profiler-off baseline is the wall authority; the
instrumented trace remains layout/active/residency evidence with its overhead
recorded explicitly. One-shot and two-shot/RMSNorm companion kernels remain
separately visible. A fusion group
is emitted only when every member's physical production event set exactly
equals its single timing owner's set. Unequal occurrence subsets remain
partial evidence and never inherit the owner's aggregate timing.

The shared Architecture remains available for both bindings:

- [SGLang Architecture](https://yamy1234.github.io/llm-arch-reviewer/viewer.html?model=qwen35_v2&implementation=sglang_f609d677b_qwen35_033446bb_tp8&viewMode=architecture)
- [vLLM Architecture](https://yamy1234.github.io/llm-arch-reviewer/viewer.html?model=qwen35_v2&implementation=vllm_487ecf187_qwen35_native_tp8&viewMode=architecture)

Raw traces, eager mappings, selectors, manifests, and validation hashes are
retained outside git under `current/qwen35-complete-profiles/`; rejected compact
profiles and timelines are retained there and identified by public hashes.

Build the catalog with:

```bash
python3 scripts/build_v2.py --model qwen35
```

The catalog uses the shared compiler/schema and the single canonical
`docs/viewer.html`; there is no Qwen3.5-specific viewer path. Future profiles
will appear only after all acceptance gates pass.
