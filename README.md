# llm-arch-reviewer

> Interactive architecture diagrams for LLM inference, with profile data overlaid.

**Live:** https://yamy1234.github.io/llm-arch-reviewer/

Each model gets a modular, zoomable, click-to-drill diagram. Architectural blocks
(attention, MoE, compressor, indexer, …) are annotated with measured per-block ms
and kernel breakdown from real `sglang` / `vllm` traces.

## Why

PDF whitepapers and source code give two halves of the picture; you still spend
hours mapping "this paper paragraph" ↔ "this Python class" ↔ "this trace stage".
This tool makes the mapping clickable:

- click a block → see its source code links (GitHub permalinks pinned to a commit)
- click a block → see its rolled-up ms + which kernels dominate
- drill in → expand into a sub-diagram (fully orthogonal layout, ELK)
- switch to Timeline → inspect measured kernels on every CUDA stream, with
  device-idle intervals and the same IR/source attribution
- jump Architecture ↔ Timeline, or open the selected timestamp range in Perfetto
- breadcrumb + URL hash for shareable deep links

## Models

| model | status | notes |
|-------|--------|-------|
| [DeepSeek-V4](https://yamy1234.github.io/llm-arch-reviewer/viewer.html?model=dsv4) | ✅ live | 62-layer (30 CSA + 31 HCA + 1 SWA + NextN), sparse-MLA, MoE, mHC |
| [Qwen 4.0 Air Example IR-first V2](https://yamy1234.github.io/llm-arch-reviewer/viewer.html?model=qwen40_v2) | ✅ local/profiled | stable 48-layer Model IR + pure TP4, Attention DP4, and DP4/EP4 DeepEP execution paths + pinned SGLang bindings + GB300 CUDA Graph BS1/16/64/256 overlays |
| [Qwen3.5 IR-first V2](https://yamy1234.github.io/llm-arch-reviewer/viewer.html?model=qwen35_v2) | ✅ local | stable Model IR + pure-TP Execution IR + versioned binding/profile overlays |
| [Qwen3.5-397B-A17B](https://yamy1234.github.io/llm-arch-reviewer/viewer.html?model=qwen35) | ✅ local/live data | pipeline-generated source/config canonical views + trace-derived text detail, backed by P1f prefill profiler trace |
| [Qwen3.5-397B-A17B Manual](https://yamy1234.github.io/llm-arch-reviewer/viewer.html?model=qwen35_manual) | ✅ baseline | previous hand-authored full multimodal view: vision + text + MTP, decode + prefill overlays |
| Llama-4 | planned | |

## Repo layout

```
llm-arch-reviewer/
├── catalog/                    # V2 source of truth
│   └── qwen35/
│       ├── model_ir.yaml       # stable, code-independent semantics
│       ├── execution_paths/    # TP/DP/CP/EP execution plans
│       ├── bindings/           # commit-specific source/kernel bindings
│       └── profiles/           # immutable measurement overlays
├── schema/v2/                  # persisted V2 contracts
├── src/llm_arch_v2/            # V2 compiler, validation, fingerprinting
├── scripts/build_v2.py         # V2 static-bundle builder
├── docs/                       # GitHub Pages root
│   ├── index.html              # landing page (model list)
│   ├── viewer.html             # generic viewer (model-agnostic)
│   ├── dsv4/
│   │   └── arch_data.json      # built artifact for DeepSeek-V4
│   ├── qwen35/
│   │   └── arch_data.json      # pipeline-generated Qwen3.5 artifact
│   ├── qwen35_manual/
│   │   └── arch_data.json      # previous hand-authored Qwen3.5 baseline
│   └── <model>/                # other models
│       └── arch_data.json
├── models/                     # source of truth (per model)
│   ├── dsv4/
│   │   ├── ir/                 # YAML: arch.yaml, stages.yaml, profiles/*.yaml,
│   │   │                       #       config.*.yaml, source_map.yaml
│   │   ├── build/              # build_view.py, parse_trace_csv.py
│   │   └── MODEL_README.md
│   ├── qwen35/
│   │   ├── ir/
│   │   ├── build/
│   │   └── MODEL_README.md
│   ├── qwen35_manual/          # previous hand-authored Qwen3.5 baseline
│   │   ├── ir/
│   │   ├── build/
│   │   └── MODEL_README.md
│   └── common/                 # shared builders, trace mapping, validators
└── README.md                   # this file
```

The viewer (`docs/viewer.html`) is **model-agnostic**: it loads
`./<model_id>/arch_data.json` based on the `?model=…` URL parameter. To add a
model you only need to populate `models/<model_id>/ir/` and run its build
script — no JS changes.

## Local dev

```bash
git clone git@github.com:YAMY1234/llm-arch-reviewer.git
cd llm-arch-reviewer
pip install pyyaml

# rebuild data for one model
python3 models/dsv4/build/build_view.py
python3 models/qwen35/build/run_pipeline.py --skip-trace-mapping
python3 scripts/build_v2.py --model qwen35
python3 scripts/build_v2.py --model qwen40

# serve docs/ locally; the allowlisted trace endpoint enables exact Perfetto jumps
python3 scripts/serve_viewer.py --port 8765
open http://localhost:8765/                              # landing
open 'http://localhost:8765/viewer.html?model=dsv4'      # one model
open 'http://localhost:8765/viewer.html?model=qwen35'    # pipeline-generated Qwen3.5
open 'http://localhost:8765/viewer.html?model=qwen35_v2' # IR-first Qwen3.5 V2
open 'http://localhost:8765/viewer.html?model=qwen40_v2' # IR-first Qwen 4.0 V2
```

For a viewer-only session, `python3 -m http.server -d docs 8765` still works.
`scripts/serve_viewer.py` additionally exposes only exact
`*.trace.json.gz` filename+SHA256 matches under its allowlisted `--trace-root`
directories. The viewer transfers that buffer directly to `ui.perfetto.dev`
using Perfetto's browser `postMessage` interface; the raw trace remains in the
browser. If the endpoint is unavailable, the viewer asks for the matching local
trace file instead.

## IR-first V2

V2 makes four independently versioned documents explicit:

1. **Model IR** owns stable semantic nodes, symbolic shapes, and data flow.
2. **Execution Plan** derives topology-specific sharding, placement, and
   collectives. `tp_only` is the default Qwen3.5 path.
3. **Implementation Binding** maps execution nodes to symbols and kernel
   signatures for one exact source commit.
4. **Profile** attaches measurements to existing nodes for one exact execution,
   implementation, hardware, and workload. It cannot create architecture.
5. An optional **Timeline artifact** stores the individual measured kernel
   intervals, stream IDs, idle intervals, IR targets, and stack provenance for
   that exact profile. It cannot redefine Model or Execution IR.

```text
Model IR + Execution Plan -> fingerprinted Execution IR
Execution IR + Binding + Profile -> static viewer bundle
```

The Architecture pane has an explicit IR-layer selector. **Model semantics**
shows the framework-independent graph from `model_ir.yaml`; **Executed
topology** shows the selected Execution Plan after placement, layout adapters,
and collectives are applied. Implementation and profile overlays can be viewed
on either graph, so SGLang and vLLM measurements can share one canonical Model
IR whenever they implement the same model semantics.

Execution nodes carry compiled provenance (`ir_origin`, `node_kind`, and
`boundary_role`). Communication or layout operations that materialize a
module's logical input/output placement belong on the module boundary. An
internal collective such as expert dispatch may remain inside the module only
when it implements the module's own semantics. Every inserted communication
node must name its payload and result.

Fusion never rewrites Model or Execution IR. A profile may instead expose a
`fusion_group` whose `ir_nodes` share one measured kernel interval. Consumers
must treat that interval as shared evidence and must not sum it once for every
covered semantic node.

The viewer exposes execution, implementation, profile, profile-variant, and
Architecture/Timeline/Split selectors for V2 bundles while remaining compatible
with legacy bundles. Architecture-node selection filters every corresponding
kernel occurrence on the timeline; a timeline kernel restores its precise
architecture drill path and source/stack evidence.

To add a profile, place an immutable `profile.v2` YAML under the matching
`catalog/<model>/profiles/<execution_path>/<implementation>/` directory and
rebuild. If the trace has the same execution fingerprint, no diagram changes.
Create a new execution plan only when operator flow, sharding, placement, or
collectives change.

The Qwen3.5 reference build is:

```bash
python3 scripts/build_v2.py --model qwen35
```

Schema documentation lives in `schema/v2/`.

## Legacy model paths

There are two supported paths.

**Manual IR path** is still available for models like DSV4. Create
`models/<model>/ir/arch.<model>.yaml`, `stages.yaml`, `source_map.yaml`, optional
`profiles/*.yaml`, then add a thin `build/build_view.py` wrapper around
`models.common.build_view`.

**Pipeline-generated path** is retained for historical reproduction when you have a PyTorch profiler trace
with Python stack attribution and a fixed source commit. Qwen3.5 is the current
reference implementation:

1. Freeze inputs in `ir/trace.<phase>.yaml`: raw trace, source root/commit,
   run config, rank, phase, window/signature rules.
2. Generate `events` and `kernel_mapping` with `models.common.trace_mapping`.
   Model-specific stack patterns should stay in a small rule module, not in the
   common engine.
3. Generate `runtime_skeleton.yaml` from the trace-derived mapping. This file
   only records what the selected runtime iteration actually executed.
4. Reconcile trace-observed runtime nodes with source/config evidence into
   `arch_draft.yaml`. Display aliases must validate against AST/callsite-derived
   canonical source IDs.
5. Generate source/config-only canonical views, such as top-level wrappers,
   vision encoders, lm heads, or MTP paths that may not appear in the current
   trace.
6. Merge source/config canonical views and trace-derived detail views into
   `arch_generated.yaml`, and write `artifact_index.json` for producer/consumer
   provenance.
7. Build profile overlays from kernel mapping and `stages.yaml`; profile overlay
   attaches timing to existing canonical nodes and must not redefine architecture.
8. Build the dashboard bundle with `models.common.build_view`, then run
   config-driven validation and unit tests.

For Qwen3.5 the full command is:

```bash
python3 models/qwen35/build/run_pipeline.py
```

This writes `docs/qwen35/arch_data.json`, with `arch_source` pointing to
`models/qwen35/out/generated_arch/prefill_p1f_tp0/arch_generated.yaml`.

## IR schema (short)

```yaml
views:
  top:
    title: "model top"
    nodes:
      - {id: stack, label: "Decoder Stack", shape: block,
         drill: stack,                            # click → expand into "stack" view
         code_links: ["models/foo.py:1322"]}      # source links
      - {id: lm_head, label: "LM head", shape: gemm,
         stage_keys: [lm_head]}                   # link to a stage in stages.yaml
    edges:
      - {from: embed, to: stack, shape: "[B,S,D]", dtype: bf16}
```

`shape` ∈ {`io`, `block`, `gemm`, `attn`, `moe`, `norm`, `elem`, `cache`} —
controls which SVG glyph is drawn.

`stage_keys` map to entries in `stages.yaml` which in turn map to trace aliases.
The build pipeline rolls up profile data per (view, node, profile, variant) and
also computes **aggregate ms** for any node that has `drill:`.

Profile validation is intentionally config-driven: generic checks live in
`models/common/profile_validation.py`; per-model assertions such as expected
variants, required kernels, and source-only stages live in the model's IR YAMLs.

## Tech

- **layout** — [ELK.js](https://github.com/kieler/elkjs) `org.eclipse.elk.layered`
  (orthogonal routing)
- **rendering** — pure SVG, no framework
- **V2 compiler** — Python + PyYAML, deterministic execution fingerprinting
- **legacy pipeline** — retained for historical reproduction

No backend, no build step at deploy time — `docs/` is the entirety of the
public site.

## License

MIT
