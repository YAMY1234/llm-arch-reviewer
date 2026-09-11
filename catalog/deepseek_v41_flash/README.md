# DeepSeek-V4.1-Flash Model IR

Pinned publisher revision: `dba1be0a40aa45a94ad051997016db3960a90277`.

This catalog covers model mathematics, state contracts and primitive Viewer drills, including vision and DSpark. It has an explicit `model_only` lifecycle: no Execution Plans, Bindings, Profiles, timelines or performance acceptance.

The target is 40 layers, grouped as causal encoder0–19 and decoder20–39; compression entries40–42 belong to three DSpark auxiliary layers. Full layers2,8,14,20 publish global KV and index K; Reindex layers24,28,32,36 rescore decoder layer20 keys; other compressed layers reuse the current selection. Every layer owns a distinct SWA cache. Candidate masks are produced by layer20 and overwritten each forward before later indexers consume them.

The reference runs all40 target layers during prefill. Report-described encoder-only prefill with bounded final128-token replay is an approximate deployment schedule; it is shown as specification-only control and is not claimed as a runnable reference path. DSpark forward is provided but its confidence-scheduled verification loop is not present in the reference generation code. Engram deliberately omits convolution. Single-pass mHC uses the previous sublayer's pre weights; its current pre weights feed the next sublayer. DSpark uses the means of the attention inputs to layers37–39, and raw linear confidence scores.

Source acquisition and authority SHA256 verification use the committed test fixture lock through the shared pipeline. Source ledger Git object hashes refer to publisher Python blobs. The report SHA256 covers downloaded PDF bytes, independently of its Git LFS pointer. Tests compare extracted official config/source functions against independent arithmetic cases; no full-model or GPU equivalence is claimed.

## Reproduce the review

Install the repository's `dev` dependencies and PyTorch 2.8 for CPU numerical tests. The Linux CI installs the CPU-only wheel from the PyTorch CPU index. No GPU, model weights, or private paths are required.

```bash
python3 scripts/materialize_model_authority.py \
  --model-root catalog/deepseek_v41_flash \
  --output /tmp/deepseek-v41-flash-authority
python3 -m pytest -q tests/test_deepseek_v41_flash_model_ir.py \
  tests/test_model_only.py tests/test_semantic_audit.py tests/test_validation_evidence.py
python3 scripts/build_v2.py --model deepseek_v41_flash
python3 scripts/audit_model_only.py --model deepseek_v41_flash \
  --source-repo /tmp/deepseek-v41-flash-authority/git \
  --json-out /tmp/deepseek-v41-flash-semantic.json
python3 scripts/serve_viewer.py --port 8765
```

Open `http://127.0.0.1:8765/viewer.html?model=deepseek_v41_flash_v2&irLayer=model`. In a second terminal:

```bash
python3 scripts/audit_model_only_viewer.py docs/deepseek_v41_flash_v2/arch_data.json \
  --base-url http://127.0.0.1:8765 --output /tmp/deepseek-v41-flash-browser
```

Use `--browser /path/to/chrome` if Playwright Chromium is not installed. The browser audit compares HTTP-served and local bundle bytes, clicks every node and declared drill, checks exact primitive inventories, geometry, semantic details, breadcrumbs and URL reload, and saves one screenshot per view. CI uploads this review evidence separately from production release acceptance.

## Generic lifecycle prerequisite

`pipeline.lifecycle: model_only` opts into zero runtime artifacts and strict semantic boundaries. Child-to-parent `port_bindings` may rename identities only; shapes, dtypes, layouts and lifetimes must match exactly. Unreachable views, fake timing/fusion and missing semantic equations are rejected. Unavailable runtime evidence is `out_of_scope` with a reason, never `verified`. A separate small scale-model fixture and negative profiled tests exercise the shared path; see `tests/test_model_only.py` and `PIPELINE.md`.

The ordinary production release audit rejects direct model-only requests. Its `--all` production scope excludes only explicitly model-only catalogs, which have their own mandatory source/semantic/browser CI job. A model-only pass cannot establish production readiness.
