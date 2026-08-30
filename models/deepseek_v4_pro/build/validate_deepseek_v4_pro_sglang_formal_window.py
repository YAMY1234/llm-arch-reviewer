#!/usr/bin/env python3
"""Fail-closed validation of one SGLang formal decode profile coordinate."""

from __future__ import annotations

import re
from typing import Any


SGLANG_DECODE_RE = re.compile(
    r"Decode batch \[(?P<step>\d+)\], #running-req: (?P<running>\d+).*?"
    r"cuda graph: (?P<cuda_graph>True|False), gen throughput \(token/s\): "
    r"(?P<throughput>[0-9.]+)"
)


def validate_formal_window(
    *,
    client: dict[str, Any],
    scheduler_log: str,
    baseline: dict[str, Any],
    concurrency: int,
) -> dict[str, Any]:
    """Return the exact second-launch throughput gate or raise ValueError."""

    contract = client.get("contract") or {}
    expected_contract = {
        "isl": 8192,
        "osl": 1024,
        "random_range_ratio": 1.0,
        "concurrency": concurrency,
        "warmup_request_count": 3 * concurrency,
        "formal_request_count": concurrency,
        "no_intentionally_shared_prefix": True,
        "dspark_enabled": False,
    }
    mismatches = {
        key: (contract.get(key), expected)
        for key, expected in expected_contract.items()
        if contract.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"exact serving contract mismatch: {mismatches}")
    if client.get("state") != "passed":
        raise ValueError("exact serving client did not pass")

    coordinate = client.get("profile_coordinate") or {}
    start_step = coordinate.get("resolved_absolute_start_step")
    client_target = coordinate.get("resolved_absolute_target_step")
    if (
        coordinate.get("profile_prime_steps") != 1
        or not isinstance(start_step, int)
        or client_target != start_step + 1
    ):
        raise ValueError("profile coordinate is not activation-prime plus formal")
    controls = client.get("profile_controls") or []
    request = controls[0].get("request") if len(controls) == 1 else {}
    if (
        len(controls) != 1
        or controls[0].get("http_status") != 200
        or request.get("start_step") != start_step
        or request.get("num_steps") != 2
    ):
        raise ValueError("profile control does not capture exactly prime plus formal")

    rows = {
        int(match.group("step")): {
            "running": int(match.group("running")),
            "cuda_graph": match.group("cuda_graph") == "True",
            "throughput_token_s": float(match.group("throughput")),
        }
        for match in SGLANG_DECODE_RE.finditer(scheduler_log)
    }
    formal = rows.get(start_step)
    if formal is None:
        raise ValueError("formal second-launch scheduler row is absent")
    if formal["running"] != concurrency or formal["cuda_graph"] is not True:
        raise ValueError("formal second-launch mode or shape is not exact")

    selected = (
        (baseline.get("concurrencies") or {})
        .get(str(concurrency), {})
        .get("selected_decode")
        or {}
    )
    baseline_throughput = selected.get("throughput_token_s")
    if baseline_throughput is None:
        raise ValueError("profiler-off baseline lacks matched decode throughput")
    minimum = 0.8 * float(baseline_throughput)
    if formal["throughput_token_s"] < minimum:
        raise ValueError(
            "formal second-launch throughput is a profile-start collapse: "
            f"{formal['throughput_token_s']} < {minimum:.3f}"
        )
    return {
        "profile_start_step": start_step,
        "profiler_activation_forward_ct": start_step,
        "activation_affected_scheduler_step": start_step - 1,
        "activation_affected_scheduler_row": rows.get(start_step - 1),
        "client_resolved_target_forward_ct": client_target,
        "formal_target_step": start_step,
        "profile_start": formal,
        "formal_target": formal,
        "profiler_off_baseline_throughput_token_s": baseline_throughput,
        "minimum_accepted_throughput_token_s": minimum,
    }
