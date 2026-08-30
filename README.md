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
- switch to Timeline → inspect measured kernels in concurrency-aware compact
  activity lanes by default, or expand the lossless physical CUDA streams;
  both retain device-idle intervals and the same IR/source attribution
- jump Architecture ↔ Timeline, or open the selected timestamp range in Perfetto
- breadcrumb + URL hash for shareable deep links

## Models

| model | status | notes |
|-------|--------|-------|
| [Qwen 4.0 Air Example IR-first V2](https://yamy1234.github.io/llm-arch-reviewer/viewer.html?model=qwen40_v2) | ✅ local/profiled | stable 48-layer Model IR + pure TP4, Attention DP4, and DP4/EP4 DeepEP execution paths + pinned SGLang bindings + GB300 CUDA Graph BS1/16/64/256 overlays |
| [Qwen3.5 IR-first V2](https://yamy1234.github.io/llm-arch-reviewer/viewer.html?model=qwen35_v2) | ✅ local/profiled | framework-independent 60-layer hybrid Model IR + validated pure TP8 SGLang/vLLM bindings + 10 CMH GB300 production profiles (prefill BS1 and CUDA Graph decode BS1/16/64/256) |
| [GLM-5.2 NVFP4 IR-first V2](https://yamy1234.github.io/llm-arch-reviewer/viewer.html?model=glm52_v2) | ✅ local/profiled | pure TP8 SGLang/TRT-LLM bindings on CMH GB300; 7 accepted profiles, with TRT-LLM prefill/BS1/BS16 explicitly unsupported under the fixed production capture contract |
| [GLM-5.3-Flash IR-first V2](https://yamy1234.github.io/llm-arch-reviewer/viewer.html?model=glm53_flash_v2) | ✅ local/profiled | one stable multimodal Model IR; pure TP8 SGLang/vLLM bindings; 6 accepted CMH GB300 profiles and 4 explicit unsupported matrix points |
| [Kimi K3 IR-first V2](https://yamy1234.github.io/llm-arch-reviewer/viewer.html?model=kimi_k3_v2) | ✅ local/profiled | official checkpoint-locked multimodal Model IR; pure TP8 portable execution contract; commit-specific SGLang/vLLM bindings; 8 measured 8K/1K GB300 production profiles and 2 explicit unsupported BS256 points |
| [DeepSeek V4 Pro 0813 IR-first V2](https://yamy1234.github.io/llm-arch-reviewer/viewer.html?model=deepseek_v4_pro_v2) | ✅ local/profiled | exact public 0813 Model IR; pure TP8 SGLang/vLLM bindings; 10 accepted two-node GB300 profiles covering stable 8K prefill and CUDA Graph 8K/1K decode GBS 1/16/64/256 |

## Repo layout

```
llm-arch-reviewer/
├── PIPELINE.md                 # one canonical IR-first workflow
├── catalog/                    # source of truth
│   └── <model>/
│       ├── model_ir.yaml       # stable, code-independent semantics
│       ├── execution_paths/    # TP/DP/CP/EP execution plans
│       ├── bindings/           # commit-specific source/kernel bindings
│       ├── profiles/           # immutable measurement overlays
│       └── sol_manifests/      # optional hardware SoL/gap-analysis inputs
│   └── hardware/               # shared sourced ceilings + calibration surfaces
├── schema/v2/                  # persisted contracts
├── src/llm_arch_v2/            # compiler, validation, fingerprinting
├── scripts/build_v2.py         # static-bundle compiler
├── docs/                       # GitHub Pages root
│   ├── index.html              # landing page (model list)
│   ├── viewer.html             # shared Architecture/Timeline viewer
│   └── <model>_v2/
│       └── arch_data.json
├── models/common/              # shared trace, attribution, and profile utilities
└── README.md                   # this file
```

The canonical workflow is specified in [PIPELINE.md](PIPELINE.md) and its
[Chinese version](PIPELINE.zh-CN.md). New models
must use `catalog/<model>/`; trace data may attach implementation and timing
evidence but may not generate or mutate Model IR. Model-specific behavior must
live in catalog metadata or adapters, not in viewer JavaScript.

## Local dev

```bash
git clone git@github.com:YAMY1234/llm-arch-reviewer.git
cd llm-arch-reviewer
python3 -m pip install -e '.[dev]'

# rebuild catalog data for one model
python3 scripts/build_v2.py --model qwen35
python3 scripts/build_v2.py --model qwen40
python3 scripts/build_v2.py --model kimi_k3
python3 scripts/build_v2.py --model deepseek_v4_pro

# rebuild every audited catalog through the same compiler
python3 scripts/build_v2.py --all

# canonical upstream checks
python3 -m pytest -q
git diff --exit-code -- docs

# real-browser stream-mode acceptance for every compiled profile
python3 scripts/audit_timeline_stream_modes.py docs/*/arch_data.json \
  --base-url http://127.0.0.1:8765 \
  --output /tmp/llm-arch-reviewer-stream-audit.json

# serve docs/ locally; the allowlisted trace endpoint enables exact Perfetto jumps
python3 scripts/serve_viewer.py --port 8765

# persistent background server (defaults to 127.0.0.1:8766)
scripts/viewer_server.sh
scripts/viewer_server.sh status
scripts/viewer_server.sh restart
scripts/viewer_server.sh stop
open http://localhost:8765/                              # landing
open 'http://localhost:8765/viewer.html?model=qwen35_v2' # IR-first Qwen3.5 V2
open 'http://localhost:8765/viewer.html?model=qwen40_v2' # IR-first Qwen 4.0 V2
open 'http://localhost:8765/viewer.html?model=glm52_v2'  # IR-first GLM-5.2 V2
open 'http://localhost:8765/viewer.html?model=glm53_flash_v2' # IR-first GLM-5.3-Flash V2
open 'http://localhost:8765/viewer.html?model=kimi_k3_v2' # IR-first Kimi K3 V2
open 'http://localhost:8765/viewer.html?model=deepseek_v4_pro_v2' # IR-first DeepSeek V4 Pro V2
```

For a viewer-only session, `python3 -m http.server -d docs 8765` still works.
`scripts/serve_viewer.py` additionally exposes only exact
`*.trace.json.gz` filename+SHA256 matches under its allowlisted `--trace-root`
directories passed with repeatable `--trace-root` arguments. With no trace root,
the viewer remains fully usable and exact Perfetto handoff is simply unavailable. The
viewer transfers that buffer directly to `ui.perfetto.dev`
using Perfetto's browser `postMessage` interface; the raw trace remains in the
browser. If the endpoint is unavailable, the viewer asks for the matching local
trace file instead.

`scripts/viewer_server.sh` launches `serve_viewer.py` through `nohup`, so the
server survives the terminal that started it. When invoked by Codex on macOS,
the script automatically asks Terminal to own that same `nohup` launch; this
places the server outside Codex's disposable child-process tree while retaining
access to the project and trace directories. Its PID and log live under
`${TMPDIR:-/tmp}/llm-arch-reviewer-viewer-<uid>/`; use the script's `status` and
`logs` actions instead of relying on a foreground terminal. Override the default
bind with `VIEWER_HOST` or `VIEWER_PORT` when needed.

## Canonical IR-first pipeline

The complete workflow and acceptance gates live in [PIPELINE.md](PIPELINE.md)
([中文版](PIPELINE.zh-CN.md)).
It makes five independently versioned documents explicit:

1. **Model IR** owns stable semantic nodes, symbolic shapes, and data flow.
2. **Execution Plan** derives topology-specific sharding, placement, and
   collectives. `tp8` is the accepted Qwen3.5 first-stage path.
3. **Implementation Binding** maps execution nodes to symbols and kernel
   signatures for one exact source commit.
4. **Profile** attaches measurements to existing nodes for one exact execution,
   implementation, hardware, and workload. It cannot create architecture.
5. An optional **Timeline artifact** stores the individual measured kernel
   intervals, stream IDs, idle intervals, IR targets, and stack provenance for
   that exact profile. It cannot redefine Model or Execution IR.

The same bundle may also contain optional `workload-ir.v1`, `cost-ir.v1`,
`transition-plan.v1`, `kernel-plan.v1`, `hardware-spec.v1`, `sol-profile.v1`,
and `gap-report.v1` derivatives. They reuse canonical IR IDs to compare
measured active time against two deliberately separate evidence levels: a
transition-derived physical lower bound and a plan-exact calibrated attainable
P10/P50/P90 projection. Legacy operator-family efficiency seeds are disabled
by default and are not projections. Missing adapters, kernel plans,
calibration, and silicon-bound violations remain explicit. The viewer can show
measured active GPU beside physical ideal, attainable projection/coverage, and
implementation gap where calibration is actually valid.

```text
Model IR + Execution Plan + source/config -> candidate Execution IR
candidate Execution IR + eager semantic trace -> validated fingerprint + Binding
validated Execution IR + Binding + Profile -> static viewer bundle
```

The execution fingerprint hashes the normalized, framework-independent
contract—not Python symbols or kernel sequences. A CUDA-Graph-disabled eager
trace must validate that contract for each exact Binding before production
timing can be attached.

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
covered semantic node. Only the group owner displays measured timing; covered
semantic nodes display `fused into <owner>` without copied timing scalars.

The viewer exposes execution, implementation, profile, profile-variant, and
Architecture/Timeline/Split selectors. Architecture-node selection filters every corresponding
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

## Tech

- **layout** — [ELK.js](https://github.com/kieler/elkjs) `org.eclipse.elk.layered`
  (orthogonal routing)
- **rendering** — pure SVG, no framework
- **V2 compiler** — Python + PyYAML, deterministic execution fingerprinting

No backend, no build step at deploy time — `docs/` is the entirety of the
public site.

## License

MIT
