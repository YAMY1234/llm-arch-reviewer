# Kimi K3 canonical v2 catalog

This catalog models the official `moonshotai/Kimi-K3` checkpoint at revision
`a590ce090cb049c93a33dfe8c208ec652aa20503`. `model_ir.yaml` is the
framework-independent architecture truth. `execution_paths/tp8.yaml` adds the
portable pure-TP8 collective, layout, and state contract. Framework bindings
and profiles must be pinned to their own source revisions and must not alter
the Model IR.

Stage 1 profiles are text-only, non-speculative, aggregated serving with fixed
ISL 8192 and OSL 1024. Decode points use global batch sizes 1, 16, 64, and 256;
prefill is retained only where a stable production window is demonstrated.
The optional vision path remains fully modeled but is explicitly not executed
by text-only profiles.

The accepted matrix contains eight measured profiles: prefill BS1 and CUDA
Graph decode BS1/16/64 for both SGLang and vLLM. Decode BS256 is explicitly
unsupported for each framework under the locked admission, memory, and graph
contract; no timing is substituted for either point.

The pinned checkpoint source is authoritative when published prose and the
executable graph differ. It performs joint attention over each packed
`t*h*w` media segment and applies the multimodal projector as
`Linear -> GELU -> Linear -> RMSNorm`; the catalog records that exact path and
does not substitute the report's separate spatial/temporal-pass description.

Large raw traces and checkpoint evidence are retained outside git under the
task evidence root. Git contains only portable IR, bindings, compact profile
artifacts, manifests, hashes, and validation reports.
