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
- breadcrumb + URL hash for shareable deep links

## Models

| model | status | notes |
|-------|--------|-------|
| [DeepSeek-V4](https://yamy1234.github.io/llm-arch-reviewer/viewer.html?model=dsv4) | ✅ live | 62-layer (30 CSA + 31 HCA + 1 SWA + NextN), sparse-MLA, MoE, mHC |
| [Qwen 4.0 Air Example IR-first V2](https://yamy1234.github.io/llm-arch-reviewer/viewer.html?model=qwen40_v2) | ✅ profiled | stable 48-layer Model IR + pure TP4, Attention DP4, and DP4/EP4 DeepEP paths + pinned SGLang bindings + GB300 CUDA Graph BS1/16/64/256 overlays |
| Qwen3.5 | planned | |
| Llama-4 | planned | |

## Repo layout

```
llm-arch-reviewer/
├── docs/                       # GitHub Pages root
│   ├── index.html              # landing page (model list)
│   ├── viewer.html             # generic viewer (model-agnostic)
│   ├── dsv4/
│   │   └── arch_data.json      # built artifact for DeepSeek-V4
│   └── <model>/                # other models
│       └── arch_data.json
├── models/                     # source of truth (per model)
│   ├── dsv4/
│   │   ├── ir/                 # YAML: arch.yaml, stages.yaml, profiles/*.yaml,
│   │   │                       #       config.*.yaml, source_map.yaml
│   │   ├── build/              # build_view.py, parse_trace_csv.py
│   │   └── MODEL_README.md
│   └── _common/                # shared helpers (future)
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
python3 scripts/build_v2.py --model qwen40

# serve docs/ locally (CORS-safe)
python3 -m http.server -d docs 8765
open http://localhost:8765/                              # landing
open 'http://localhost:8765/viewer.html?model=dsv4'      # one model
open 'http://localhost:8765/viewer.html?model=qwen40_v2' # IR-first Qwen 4.0 V2
```

## IR-first V2

V2 makes four independently versioned documents explicit:

1. **Model IR** owns stable semantic nodes, symbolic shapes, and data flow.
2. **Execution Plan** derives topology-specific sharding, placement, and
   collectives. Pure TP remains the default path.
3. **Implementation Binding** maps execution nodes to symbols and kernel
   signatures for one exact source commit.
4. **Profile** attaches measurements to existing nodes for one exact execution,
   implementation, hardware, and workload. It cannot create architecture.

```text
Model IR + Execution Plan -> fingerprinted Execution IR
Execution IR + Binding + Profile -> static viewer bundle
```

The viewer exposes execution, implementation, profile, and profile-variant
selectors for V2 bundles while remaining compatible with legacy bundles.

To add a profile, place an immutable `profile.v2` YAML under the matching
`catalog/<model>/profiles/<execution_path>/<implementation>/` directory and
rebuild. If the trace has the same execution fingerprint, no diagram changes.
Create a new execution plan only when operator flow, sharding, placement, or
collectives change.

The Qwen 4.0 reference build is:

```bash
python3 scripts/build_v2.py --model qwen40
```

Schema documentation lives in `schema/v2/`.

## Adding a new model

1. **Copy the dsv4 template:**
   ```bash
   cp -r models/dsv4 models/<your_model>
   rm -rf models/<your_model>/out  # regen below
   ```
2. **Edit IR YAMLs** in `models/<your_model>/ir/`:
   - `arch.<your_model>.yaml` — the views (top → stack → layer → module → leaf),
     each node carries `id`, `label`, `shape`, optional `drill`, optional
     `stage_keys`, optional `code_links`.
   - `stages.yaml` — map official stage names to trace aliases.
   - `source_map.yaml` — point `code_links` to your repo + commit.
   - `profiles/*.yaml` — your trace data (or use `parse_trace_csv.py` if you
     have a `trace-kernel-learning`-style CSV).
3. **Update `MODEL_META`** in `models/<your_model>/build/build_view.py` so the
   header shows a nice `model_label` and `subtitle`.
4. **Add a card** for it in `docs/index.html`.
5. **Build:**
   ```bash
   python3 models/<your_model>/build/build_view.py
   ```
   This writes both `docs/<your_model>/arch_data.json` (for GH Pages) and a
   local copy in `models/<your_model>/out/`.
6. **Commit & push.** GH Pages auto-deploys.

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

## Tech

- **layout** — [ELK.js](https://github.com/kieler/elkjs) `org.eclipse.elk.layered`
  (orthogonal routing)
- **rendering** — pure SVG, no framework
- **V2 compiler** — Python + PyYAML, deterministic execution fingerprinting
- **data pipeline** — Python + PyYAML, ~600 LoC

No backend, no build step at deploy time — `docs/` is the entirety of the
public site.

## License

MIT
