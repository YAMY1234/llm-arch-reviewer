#!/usr/bin/env python3
"""Validate semantic-only catalog acceptance; never production release acceptance."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'src'))
from llm_arch_v2 import compile_catalog, validate_validation_evidence
from llm_arch_v2.semantic_audit import audit_semantic_closure


def audit_model_only(model_root: Path, source_repo: Path, bundle_path: Path) -> dict:
    errors = []
    bundle = compile_catalog(model_root)
    if bundle['meta'].get('lifecycle') != 'model_only':
        errors.append('model-only audit requires an explicitly model_only catalog')
    evidence = validate_validation_evidence(model_root)
    if evidence['status'] != 'pass':
        errors.extend(evidence['errors'])
    semantic = audit_semantic_closure(model_ir_path=model_root / 'model_ir.yaml',
                                     ledger_path=model_root / 'semantic_source_ledger.yaml',
                                     source_repo=source_repo)
    if semantic['status'] != 'complete':
        errors.extend(semantic['errors'] or ['semantic closure incomplete'])
    if not bundle_path.is_file() or json.loads(bundle_path.read_text()) != bundle:
        errors.append('published bundle differs from canonical compilation')
    return {'schema_version': 'model-only-acceptance.v1', 'model_id': model_root.name,
            'status': 'pass' if not errors else 'fail', 'acceptance_scope': 'model_only',
            'production_release_ready': False, 'browser_verified': False,
            'bundle_sha256': hashlib.sha256(bundle_path.read_bytes()).hexdigest() if bundle_path.is_file() else None,
            'validation_evidence': evidence, 'semantic_closure': semantic, 'errors': errors}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--source-repo', required=True, type=Path)
    parser.add_argument('--catalog-root', type=Path, default=REPO_ROOT / 'catalog')
    parser.add_argument('--docs-root', type=Path, default=REPO_ROOT / 'docs')
    parser.add_argument('--json-out', type=Path)
    args = parser.parse_args()
    report = audit_model_only(args.catalog_root / args.model, args.source_repo.resolve(),
                             args.docs_root / f'{args.model}_v2' / 'arch_data.json')
    output = json.dumps(report, indent=2, sort_keys=True) + '\n'
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(output)
    else:
        print(output)
    return 0 if report['status'] == 'pass' else 1

if __name__ == '__main__':
    raise SystemExit(main())
