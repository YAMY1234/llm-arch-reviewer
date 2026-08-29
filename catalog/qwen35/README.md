# Qwen3.5 V2 catalog

`model_ir.yaml` is the stable source of truth. It describes the text-model
semantics and symbolic data flow without importing SGLang source locations,
kernel names, hardware, or measurements.

The default execution path is `tp_only`. Its plan derives explicit TP sharding
annotations and output collectives from the stable model graph. Concrete TP
degree belongs to a profile; TP2/TP4/TP8 therefore reuse the same execution
template and fingerprint.

The current profile is a faithful migration of the existing P1f artifact:

- SGLang commit `88f88fc06`
- GB300, TP4, DP1, CP1, EP1
- ISL 1000, OSL 1, concurrency 256
- chunked-prefill Torch Profiler window from job `1672695`

It is not the proposed 8K/1K BS sweep. Future BS1/16/64/256 runs should be
added as new immutable profile documents with their exact workload and evidence
manifests.

Build the catalog with:

```bash
python3 scripts/build_v2.py --model qwen35
```

A profile may only reference existing execution nodes. If a new trace changes
only timings, kernel dispatch, or source symbols, add a profile and/or binding.
Add a new execution plan only when normalized operator flow, sharding,
placement, or collectives change.
