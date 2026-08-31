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
bindings compile to execution fingerprint `exec_9772c24a3aa3f623`:

- SGLang package commit `f609d677b909ca46c64bb6803b69a85fedbf86bc`,
  with the runtime Qwen3.5/config/GDN/MoE modules byte-matched to commit
  `033446bb05f35c0943aed2750c443077ffc0b92c`.
- vLLM commit `487ecf187d3dfe74d2cf6119a92881dba403c219`.

The 8K/1K production matrix has **ten accepted measured profiles**: SGLang and
vLLM prefill at GBS1 plus CUDA-Graph decode at GBS1/16/64/256. Every point uses
3×concurrency warmup requests and 1×concurrency formal requests. The graph-off
semantic capture is taken at the matched stable formal coordinate for the same
framework, phase, batch, rank, source, and hardware contract.

All model-bearing production events have exact same-rank, same-phase,
occurrence-scoped eager closure. The graph-off semantic owner is derived
independently from validated Python module stacks and pinned framework-source,
kernel-family, layer-occurrence, and collective-order anchors; production
sequence attribution is not its semantic oracle. Exact signatures are used
where graph-off and graph-on launches are identical; explicit ordered 1:N and
N:1 relations retain the source event IDs, kernel names, Python stacks, and
stack hashes where CUDA Graph specialization changes the physical
decomposition. A production/eager owner disagreement fails closed. Framework
scheduler, planning, sampling, and output helpers are separately typed as
runtime support and are not used to hide unresolved Model IR work.

For vLLM, post-collective kernels whose eager stacks remain under
`SharedExperts_<layer>` stay with that layer's shared-expert/MoE owner. Prefill
maps only the independently anchored RMSNorm kernel to `top.final_norm`, then
the logits-processor LM-head GEMM and logits collective. Decode retains the
documented compiler fusion of the final norm into the last layer's TP MoE
all-reduce/RMSNorm owner; no shared-expert tail kernel is relabeled as final
norm and no duration is copied to the semantic target.

Profile-aggregate fusion groups are emitted only when each member's physical
production event set exactly equals its single timing owner's set and every
owner event is closed. A relationship that is fused in only some layer
occurrences remains occurrence-scoped structural evidence: it receives no
aggregate fusion claim and no copied timing. This preserves one timing owner
for 1:N and N:1 semantic bindings.

SGLang prefill wall time is the exact post-warmup, prefill-only
ForwardPassMetrics DeviceTimer span: 93.678078 ms for one request with 8192
prefill tokens, zero decode requests, and zero decode tokens. It matches the
same instrumented forward's 93.450314 ms model envelope. The earlier
4111.995663 ms scheduler interval is retained as rejected evidence because it
included intervening decode work and was not an isolated forward.

In the diagnostics, `cuda_graph_enabled`/`used_graph_path` refers only to the
selected formal forward and is proven by nonzero raw-trace graph IDs. Server
CUDA Graph configuration is a separate fact. Thus vLLM prefill remains
`used_graph_path: false` with zero graph IDs even though the server configured
`FULL_AND_PIECEWISE`; decode profiles retain their observed graph-path state.

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
separately visible.

The shared Architecture remains available for both bindings:

- [SGLang Architecture](https://yamy1234.github.io/llm-arch-reviewer/viewer.html?model=qwen35_v2&implementation=sglang_f609d677b_qwen35_033446bb_tp8&viewMode=architecture)
- [vLLM Architecture](https://yamy1234.github.io/llm-arch-reviewer/viewer.html?model=qwen35_v2&implementation=vllm_487ecf187_qwen35_native_tp8&viewMode=architecture)

Raw traces, eager mappings, selectors, manifests, compile caches, and validation
hashes are retained outside git under `current/qwen35-complete-profiles/` and
the recorded cluster scratch roots. Public profiles retain all-rank hashes for
the non-regenerable evidence.

Build the catalog with:

```bash
python3 scripts/build_v2.py --model qwen35
```

The catalog uses the shared compiler/schema and the single canonical
`docs/viewer.html`; there is no Qwen3.5-specific viewer path. Future profiles
must pass the same fail-closed acceptance gates.
