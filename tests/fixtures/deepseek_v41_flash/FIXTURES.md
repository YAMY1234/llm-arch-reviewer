# Independent publisher authority fixtures

These files are unmodified publisher bytes from DeepSeek-V4.1-Flash revision
`dba1be0a40aa45a94ad051997016db3960a90277`. `source-lock.json` records public
immutable URLs and SHA256s for the complete reviewed source inventory. Only
small config, source and license files are vendored; the report and weight index
can be downloaded from their locked URLs. No weights are needed.

CPU mathematical tests execute selected original Python function ASTs from
`inference/model.py`, without importing its GPU kernels or instantiating the
checkpoint. Expectations are independently expressed scalar mathematics and
explicit causal/state cases. PyTorch is required for those CPU tests; other
catalog/source tests do not require it. No kernel numerical equivalence,
full-model numerical equivalence, generation quality, or performance is claimed.
