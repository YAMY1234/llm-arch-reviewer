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

The immutable 8K/1K production matrix contains ten measured profiles and zero
unsupported substitutions. Each framework has stable prefill at global BS1 and
CUDA Graph decode at global BS1/16/64/256. Every point uses 3×concurrency warmup
requests and 1×concurrency formal requests, with one exact selected formal
forward. Separate graph-off eager captures provide Python-stack semantic
evidence; production captures provide timing. All eight TP ranks are validated
for every point.

| Framework | Phase | Global BS | Production job | Viewer |
|---|---:|---:|---:|---|
| SGLang | prefill | 1 | 3414663 | [Architecture](https://yamy1234.github.io/llm-arch-reviewer/viewer.html?model=qwen35_v2&implementation=sglang_f609d677b_qwen35_033446bb_tp8&phase=prefill&profile=qwen35_tp8_sglang_prefill_bs1_8k1k&viewMode=architecture) · [Timeline](https://yamy1234.github.io/llm-arch-reviewer/viewer.html?model=qwen35_v2&implementation=sglang_f609d677b_qwen35_033446bb_tp8&phase=prefill&profile=qwen35_tp8_sglang_prefill_bs1_8k1k&viewMode=timeline) |
| SGLang | decode | 1 | 3414668 | [Architecture](https://yamy1234.github.io/llm-arch-reviewer/viewer.html?model=qwen35_v2&implementation=sglang_f609d677b_qwen35_033446bb_tp8&phase=decode&profile=qwen35_tp8_sglang_cg_decode_bs1_8k1k&viewMode=architecture) · [Timeline](https://yamy1234.github.io/llm-arch-reviewer/viewer.html?model=qwen35_v2&implementation=sglang_f609d677b_qwen35_033446bb_tp8&phase=decode&profile=qwen35_tp8_sglang_cg_decode_bs1_8k1k&viewMode=timeline) |
| SGLang | decode | 16 | 3414675 | [Viewer](https://yamy1234.github.io/llm-arch-reviewer/viewer.html?model=qwen35_v2&implementation=sglang_f609d677b_qwen35_033446bb_tp8&phase=decode&profile=qwen35_tp8_sglang_cg_decode_bs16_8k1k) |
| SGLang | decode | 64 | 3414674 | [Viewer](https://yamy1234.github.io/llm-arch-reviewer/viewer.html?model=qwen35_v2&implementation=sglang_f609d677b_qwen35_033446bb_tp8&phase=decode&profile=qwen35_tp8_sglang_cg_decode_bs64_8k1k) |
| SGLang | decode | 256 | 3414676 | [Viewer](https://yamy1234.github.io/llm-arch-reviewer/viewer.html?model=qwen35_v2&implementation=sglang_f609d677b_qwen35_033446bb_tp8&phase=decode&profile=qwen35_tp8_sglang_cg_decode_bs256_8k1k) |
| vLLM | prefill | 1 | 3414288 | [Architecture](https://yamy1234.github.io/llm-arch-reviewer/viewer.html?model=qwen35_v2&implementation=vllm_487ecf187_qwen35_native_tp8&phase=prefill&profile=qwen35_tp8_vllm_prefill_bs1_8k1k&viewMode=architecture) · [Timeline](https://yamy1234.github.io/llm-arch-reviewer/viewer.html?model=qwen35_v2&implementation=vllm_487ecf187_qwen35_native_tp8&phase=prefill&profile=qwen35_tp8_vllm_prefill_bs1_8k1k&viewMode=timeline) |
| vLLM | decode | 1 | 3414289 | [Architecture](https://yamy1234.github.io/llm-arch-reviewer/viewer.html?model=qwen35_v2&implementation=vllm_487ecf187_qwen35_native_tp8&phase=decode&profile=qwen35_tp8_vllm_cg_decode_bs1_8k1k&viewMode=architecture) · [Timeline](https://yamy1234.github.io/llm-arch-reviewer/viewer.html?model=qwen35_v2&implementation=vllm_487ecf187_qwen35_native_tp8&phase=decode&profile=qwen35_tp8_vllm_cg_decode_bs1_8k1k&viewMode=timeline) |
| vLLM | decode | 16 | 3414290 | [Viewer](https://yamy1234.github.io/llm-arch-reviewer/viewer.html?model=qwen35_v2&implementation=vllm_487ecf187_qwen35_native_tp8&phase=decode&profile=qwen35_tp8_vllm_cg_decode_bs16_8k1k) |
| vLLM | decode | 64 | 3414291 | [Viewer](https://yamy1234.github.io/llm-arch-reviewer/viewer.html?model=qwen35_v2&implementation=vllm_487ecf187_qwen35_native_tp8&phase=decode&profile=qwen35_tp8_vllm_cg_decode_bs64_8k1k) |
| vLLM | decode | 256 | 3414292 | [Viewer](https://yamy1234.github.io/llm-arch-reviewer/viewer.html?model=qwen35_v2&implementation=vllm_487ecf187_qwen35_native_tp8&phase=decode&profile=qwen35_tp8_vllm_cg_decode_bs256_8k1k) |

Raw traces, eager mappings, selectors, manifests, and validation hashes are
retained outside git under `current/qwen35-complete-profiles/`; the compact
timelines and immutable hashes are committed with each profile.

Build the catalog with:

```bash
python3 scripts/build_v2.py --model qwen35
```

All profiles use the shared compiler/schema and the single canonical
`docs/viewer.html`; there is no Qwen3.5-specific viewer path.
