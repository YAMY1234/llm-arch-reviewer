# qwen38_flash_next semantic closure gap report

- Status: **COMPLETE**
- Scope: `target_text_model_plus_eagle_mtp`
- Audit fingerprint: `2b1352de7fe29393ab36`
- This is a fail-closed report: an incomplete ledger is not a semantic PASS.

## Gate summary

| Gate | Result |
|---|---|
| `source_snapshot_integrity` | PASS |
| `source_to_ir_closure` | PASS |
| `ir_to_source_closure` | PASS |
| `ledger_integrity` | PASS |
| `catalog_attestation_honest` | PASS |

## Coverage

- Source files: 15
- Entrypoints: 26 verified / 0 pending
- Source obligations: 156 total / 0 pending
- Unclassified source members: 0
- Audited Model IR leaves: 214
- Reverse mapped leaves: 204
- Explicit reverse exclusions: 10
- Uncovered Model IR leaves: 0
- Compound primitive targets: 0

## Pending source entrypoints

- None

## Unclassified source members

- None

## Uncovered Model IR leaves

- None

## Compound primitive targets

- None

## Ledger/source errors

- None

## Review rule

Do not batch-edit Model IR from this report until every pending source entrypoint has a reviewed obligation list. After that review, patch the Model IR atomically, reconcile Binding/Profile mappings for new runtime leaves, and rerun this audit. Only a `complete` report may replace the catalog's semantic-closure attestation.
