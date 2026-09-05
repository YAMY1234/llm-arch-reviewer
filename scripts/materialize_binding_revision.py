#!/usr/bin/env python3
"""Materialize a catalog implementation Binding from an accepted revision."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from llm_arch_v2.add_trace import AddTraceError, sha256_json, validate_schema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--binding-revision", type=Path, required=True)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--implementation-id", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--container", required=True)
    parser.add_argument("--eager-evidence", required=True)
    parser.add_argument("--production-evidence", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _node_bindings_from_rules(
    rules: list[dict], *, framework_id: str
) -> dict[str, dict]:
    """Derive reviewable source bindings from accepted eager match rules."""

    grouped: dict[str, dict[str, dict]] = {}
    for rule in rules:
        source_symbol = str(rule.get("eager_match", {}).get("source_symbol") or "")
        if not source_symbol:
            continue
        file_path, separator, symbol = source_symbol.partition("::")
        display_symbol = symbol if separator else source_symbol
        entry = grouped.setdefault(rule["ir_target"], {})
        if separator:
            source_path = Path(file_path)
            if source_path.is_absolute() or ".." in source_path.parts:
                raise AddTraceError(
                    f"mapping rule {rule['rule_id']!r} has a non-repository source path"
                )
            repository_path = file_path
            if framework_id == "sglang" and repository_path.startswith("sglang/"):
                repository_path = "python/" + repository_path
            entry[source_symbol] = {
                "symbol": display_symbol,
                "link": {
                    "file": repository_path,
                    "symbol": display_symbol,
                    "display": f"{file_path} — {display_symbol}",
                },
            }
        else:
            entry[source_symbol] = {"symbol": display_symbol, "link": None}

    return {
        target: {
            "symbols": sorted({item["symbol"] for item in entries.values()}),
            "kernel_signatures": [],
            "links": [
                item["link"]
                for _source, item in sorted(entries.items())
                if item["link"] is not None
            ],
        }
        for target, entries in sorted(grouped.items())
    }


def materialize_binding(
    template: dict,
    revision: dict,
    acceptance: dict,
    *,
    implementation_id: str,
    label: str,
    container: str,
    eager_evidence: str,
    production_evidence: str,
) -> dict:
    validate_schema(
        revision,
        "binding-revision.schema.json",
        source=Path("binding-revision"),
    )
    validate_schema(
        acceptance,
        "add-trace-acceptance.schema.json",
        source=Path("acceptance"),
    )
    expected_acceptance_sha = sha256_json(
        {key: value for key, value in acceptance.items() if key != "acceptance_sha256"}
    )
    if acceptance["acceptance_sha256"] != expected_acceptance_sha:
        raise AddTraceError("acceptance digest is not content-addressed correctly")
    accepted_identity = {
        "model_id": acceptance["model_id"],
        "execution_path_id": acceptance["execution_path_id"],
        "execution_fingerprint": acceptance["execution_fingerprint"],
        "binding_revision_id": acceptance["binding_revision_id"],
        "runtime_identity_sha256": acceptance["runtime_identity_sha256"],
        "mapping_rules_sha256": acceptance["mapping_rules_sha256"],
        "binding_revision_sha256": acceptance["binding_revision_sha256"],
    }
    revision_identity = {
        "model_id": revision["model_id"],
        "execution_path_id": revision["execution_path_id"],
        "execution_fingerprint": revision["execution_fingerprint"],
        "binding_revision_id": revision["binding_revision_id"],
        "runtime_identity_sha256": revision["runtime_identity_sha256"],
        "mapping_rules_sha256": revision["mapping_rules_sha256"],
        "binding_revision_sha256": sha256_json(revision),
    }
    if accepted_identity != revision_identity:
        raise AddTraceError(
            "acceptance does not authorize this exact Binding revision"
        )
    runtime = revision["runtime_identity"]
    output = dict(template)
    output.update(
        {
            "implementation_id": implementation_id,
            "label": label,
            "model_id": revision["model_id"],
            "execution_path_id": revision["execution_path_id"],
            "framework_id": runtime["framework_id"],
            "source_repo": runtime["source_repo"],
            "source_commit": runtime["source_commit"],
            "container": container,
            "backend": runtime["framework_id"],
            "binding_status": "validated",
            "source_lock_status": "runtime_verified",
            "execution_validation": {
                "status": "pass",
                "execution_fingerprint": revision["execution_fingerprint"],
                "required_phases": ["decode"],
                "cuda_graph_enabled": False,
                "evidence": [eager_evidence, production_evidence],
                "notes": (
                    "All TP ranks passed graph-off stack reconciliation and "
                    "stack-disabled production attribution with exact event-count "
                    "and duration closure."
                ),
            },
            "binding_revision_id": revision["binding_revision_id"],
            "add_trace_acceptance_sha256": acceptance["acceptance_sha256"],
            "runtime_identity": runtime,
            "runtime_identity_sha256": revision["runtime_identity_sha256"],
            "mapping_rules_sha256": revision["mapping_rules_sha256"],
            "mapping_rules": revision["mapping_rules"],
            "node_bindings": _node_bindings_from_rules(
                revision["mapping_rules"], framework_id=runtime["framework_id"]
            ),
        }
    )
    output.pop("extends", None)
    output.pop("binding_compatible_base_commit", None)
    if runtime.get("source_patch_sha256"):
        output["source_patch_sha256"] = runtime["source_patch_sha256"]
    else:
        output.pop("source_patch_sha256", None)
    return output


def main() -> int:
    args = parse_args()
    template = yaml.safe_load(args.template.read_text())
    revision = yaml.safe_load(args.binding_revision.read_text())
    acceptance = yaml.safe_load(args.acceptance.read_text())
    output = materialize_binding(
        template,
        revision,
        acceptance,
        implementation_id=args.implementation_id,
        label=args.label,
        container=args.container,
        eager_evidence=args.eager_evidence,
        production_evidence=args.production_evidence,
    )
    validate_schema(
        output,
        "implementation-binding.schema.json",
        source=args.output,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(output, sort_keys=False))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
