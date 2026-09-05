from __future__ import annotations

import copy

import pytest

from models.common.trace_mapping import ForwardWindow, normalize_kernel_events
from models.qwen38_flash_next.build.build_qwen38_flash_next_add_trace_evidence import (
    TOPOLOGY_ID,
    _artifact_record,
    _production_signature,
    _source_anchor,
    _validate_protocols,
)


def test_artifact_records_are_run_root_relative_and_location_independent(
    tmp_path,
) -> None:
    run_root = tmp_path / "run"
    artifact = run_root / "pilot" / "evidence.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("evidence")
    record = _artifact_record(artifact, evidence_root=run_root)
    assert record["path"] == "pilot/evidence.json"
    assert not record["path"].startswith("/")

    outside = tmp_path / "outside.json"
    outside.write_text("outside")
    with pytest.raises(ValueError, match="contained by the run root"):
        _artifact_record(outside, evidence_root=run_root)


def _protocol(*, mode: str, with_stack: bool) -> dict[str, object]:
    protocol: dict[str, object] = {
        "mode": mode,
        "with_stack": with_stack,
        "topology": TOPOLOGY_ID,
        "dp_size": 1,
        "capture_phase": "decode",
        "generation_mode": "eagle_mtp",
        "source_commit": "a" * 40,
        "source_patch_sha256": "b" * 64,
        "speculative_algorithm": "EAGLE",
        "speculative_num_steps": 1,
        "speculative_eagle_topk": 1,
        "speculative_num_draft_tokens": 2,
        "global_batch_sizes": [1],
        "input_len": 8192,
        "output_len": 1024,
        "warmup_rounds": 3,
        "formal_rounds": 1,
        "cache_reset_policy": "initial-only",
    }
    if mode == "eager":
        protocol.update(
            {
                "formal_profile_steps": 1,
                "cuda_graph_decode_backend": "disabled",
                "cuda_graph_prefill_backend": "disabled",
                "cuda_graph_batch_sizes": [],
            }
        )
    else:
        protocol.update(
            {
                "formal_profile_steps": 7,
                "cuda_graph_decode_backend": "full",
                "cuda_graph_prefill_backend": "disabled",
                "cuda_graph_batch_sizes": [1],
            }
        )
    return protocol


def _run() -> dict[str, object]:
    return {
        "normalized_config": {
            "runtime_implementation": {
                "source_commit": "a" * 40,
                "source_patch_sha256": "b" * 64,
            },
            "execution_contract": {
                "generation": {
                    "mode": "eagle_mtp",
                    "speculative_num_steps": 1,
                    "speculative_topk": 1,
                    "speculative_num_draft_tokens": 2,
                }
            },
            "profile_contract": {
                "phase": "decode",
                "batch_size": 1,
                "isl": 8192,
                "osl": 1024,
            },
            "capture_procedure": {
                "warmup_rounds": 3,
                "formal_rounds": 1,
                "cache_reset_policy": "initial-only",
                "eager_with_stack": True,
                "eager_profile_steps": 1,
                "production_profile_steps": 7,
                "production_cuda_graph_backend": "full",
                "production_cuda_graph_prefill_backend": "disabled",
            },
        }
    }


def test_pilot_protocol_requires_stack_on_eager_and_stack_off_graph() -> None:
    eager = _protocol(mode="eager", with_stack=True)
    production = _protocol(mode="cudagraph", with_stack=False)
    _validate_protocols(eager, production, _run())

    invalid = copy.deepcopy(production)
    invalid["with_stack"] = True
    with pytest.raises(ValueError, match="disable Python stacks"):
        _validate_protocols(eager, invalid, _run())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("topology", "tp8", "exact .* execution topology"),
        ("dp_size", 2, "exact DP1"),
        ("capture_phase", "prefill", "decode capture"),
        ("source_commit", "c" * 40, "source_commit"),
        ("speculative_num_draft_tokens", 3, "speculative_num_draft_tokens"),
    ],
)
def test_pilot_protocol_identity_is_fail_closed(
    field: str, value: object, message: str
) -> None:
    eager = _protocol(mode="eager", with_stack=True)
    production = _protocol(mode="cudagraph", with_stack=False)
    production[field] = value
    with pytest.raises(ValueError, match=message):
        _validate_protocols(eager, production, _run())


def test_pilot_protocol_must_equal_the_normalized_run_contract() -> None:
    eager = _protocol(mode="eager", with_stack=True)
    production = _protocol(mode="cudagraph", with_stack=False)
    run = _run()
    run["normalized_config"]["profile_contract"]["isl"] = 4096
    with pytest.raises(ValueError, match="normalized run contract"):
        _validate_protocols(eager, production, run)


def _event(*, step: int, signature: str, graph_id: int) -> dict[str, object]:
    return {
        "step_index": step,
        "layer_id": None,
        "layer_kind": None,
        "substage": "proposal_update",
        "kernel_name": "generic",
        "attribution_method": signature,
        "graph_id": graph_id,
    }


def test_transfer_signature_is_cross_stream_order_invariant_but_count_exact() -> None:
    events = [
        _event(step=1, signature="proposal", graph_id=0),
        _event(step=1, signature="state_write", graph_id=0),
        _event(step=2, signature="proposal", graph_id=0),
    ]
    assert _production_signature(events) == _production_signature(
        [events[1], events[0], events[2]]
    )
    assert _production_signature(events) != _production_signature(events[:-1])


def test_source_anchor_must_be_present_on_every_tp_rank() -> None:
    common = {
        "python_stack": [
            {
                "file": "sglang/srt/models/qwen4_exp_mtp.py",
                "function": "forward",
            }
        ]
    }
    assert _source_anchor([[common], [common], [common], [common]]) == (
        "sglang/srt/models/qwen4_exp_mtp.py::forward"
    )
    with pytest.raises(ValueError, match="no SGLang source frame"):
        _source_anchor([[common], [common], [common], [{"python_stack": []}]])


def test_normalized_kernel_keeps_graph_and_cpu_launch_identity() -> None:
    trace = [
        {
            "cat": "cuda_runtime",
            "ph": "X",
            "name": "cudaLaunchKernel",
            "ts": 10.0,
            "dur": 0.1,
            "args": {"correlation": 7},
        },
        {
            "cat": "kernel",
            "ph": "X",
            "name": "graph child",
            "ts": 20.0,
            "dur": 2.0,
            "args": {"correlation": 7, "graph id": 11, "stream": 3, "device": 0},
        },
    ]
    events = normalize_kernel_events(
        trace,
        window=ForwardWindow(
            start_us=0.0,
            end_us=30.0,
            iter_bounds_us=[(0.0, 30.0)],
            anchor_kernel_count=1,
        ),
    )
    assert len(events) == 1
    assert events[0].graph_id == 11
    assert events[0].launch_ts_us == 10.0
