from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.common.trace_mapping import (  # noqa: E402
    FrameRef,
    ForwardWindow,
    StackFrameRules,
    TraceMappingRules,
    build_trace_mapping,
    close_window_phase_tails,
    find_eagle_mtp_cudagraph_decode_windows,
    find_eagle_mtp_cudagraph_substages,
    find_eagle_mtp_decode_windows,
    find_eagle_mtp_prefill_windows,
    find_vllm_execute_context_windows,
)


def _python_event(name: str, py_id: int, parent: int | None, ts: float, dur: float):
    args = {"Python id": py_id, "Ev Idx": py_id}
    if parent is not None:
        args["Python parent id"] = parent
    return {
        "ph": "X",
        "cat": "python_function",
        "name": name,
        "pid": 1,
        "tid": 1,
        "ts": ts,
        "dur": dur,
        "args": args,
    }


def _classify_toy_node(
    kernel_name: str,
    _cpu_op_name: str | None,
    _stack: list[FrameRef],
):
    if "toy_kernel" in kernel_name:
        return "toy_node", "high"
    return None, "unmapped"


TOY_RULES = TraceMappingRules(
    model_id="toy",
    signature_kernel="toy_anchor",
    signature_count_per_forward=1,
    stack=StackFrameRules(
        operator_patterns=("toy/operator.py",),
        semantic_patterns=("toy/model.py",),
        model_context_patterns=("ToyLayer",),
        phase_patterns=("toy_phase",),
    ),
    classify_node=_classify_toy_node,
)


class CommonTraceMappingTest(unittest.TestCase):
    def test_phase_tail_closure_admits_only_smallest_enclosing_python_tail(self):
        window = ForwardWindow(
            start_us=100.0,
            end_us=180.0,
            iter_bounds_us=[(100.0, 180.0)],
            anchor_kernel_count=0,
        )
        events = [
            _python_event("runner.py(1): execute_model", 1, None, 90.0, 140.0),
            _python_event("worker.py(2): execute_model", 2, 1, 95.0, 105.0),
            _python_event("unrelated.py(3): execute_model", 3, None, 250.0, 50.0),
        ]

        closed = close_window_phase_tails(
            events, window, phase_frame="execute_model"
        )

        self.assertEqual(closed.start_us, 100.0)
        self.assertEqual(closed.end_us, 200.0)
        self.assertEqual(closed.iter_bounds_us, [(100.0, 200.0)])

    def test_phase_tail_closure_keeps_longer_gpu_annotation(self):
        window = ForwardWindow(
            start_us=100.0,
            end_us=240.0,
            iter_bounds_us=[(100.0, 240.0)],
            anchor_kernel_count=0,
        )
        events = [
            _python_event("runner.py(1): execute_model", 1, None, 90.0, 110.0)
        ]

        closed = close_window_phase_tails(
            events, window, phase_frame="execute_model"
        )

        self.assertEqual(closed.iter_bounds_us, [(100.0, 240.0)])

    def test_phase_tail_closure_includes_late_async_kernel_from_owned_launch(self):
        window = ForwardWindow(
            start_us=100.0,
            end_us=180.0,
            iter_bounds_us=[(100.0, 180.0)],
            anchor_kernel_count=0,
        )
        events = [
            _python_event("runner.py(1): execute_model", 1, None, 90.0, 110.0),
            {
                "ph": "X",
                "cat": "cuda_runtime",
                "name": "cudaLaunchKernel",
                "ts": 195.0,
                "dur": 1.0,
                "args": {"correlation": 7},
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "late_collective",
                "ts": 250.0,
                "dur": 25.0,
                "args": {"correlation": 7},
            },
            {
                "ph": "X",
                "cat": "cuda_runtime",
                "name": "cudaLaunchKernel",
                "ts": 220.0,
                "dur": 1.0,
                "args": {"correlation": 8},
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "next_step_kernel",
                "ts": 280.0,
                "dur": 20.0,
                "args": {"correlation": 8},
            },
        ]

        closed = close_window_phase_tails(
            events, window, phase_frame="execute_model"
        )

        self.assertEqual(closed.iter_bounds_us, [(100.0, 275.0)])

    def test_vllm_execute_context_preserves_chunked_prefill_and_decode(self):
        def annotation(name, ts, dur, tid=19):
            return {
                "ph": "X",
                "cat": "gpu_user_annotation",
                "name": name,
                "pid": 0,
                "tid": tid,
                "ts": ts,
                "dur": dur,
            }

        events = [
            annotation("execute_context_1(8064)_generation_0(0)", 10, 40),
            annotation("execute_context_1(128)_generation_0(0)", 55, 20),
            annotation("execute_context_0(0)_generation_1(1)", 80, 5),
            annotation("execute_context_1(128)_generation_0(0)", 57, 3, tid=31),
        ]

        prefill = find_vllm_execute_context_windows(events, phase="vllm_prefill")
        decode = find_vllm_execute_context_windows(events, phase="vllm_decode")

        self.assertEqual(
            [(window.start_us, window.end_us) for window in prefill],
            [(10, 50), (55, 75)],
        )
        self.assertEqual(
            [(window.start_us, window.end_us) for window in decode],
            [(80, 85)],
        )

    def test_eagle_mtp_prefill_pairs_target_and_auxiliary_extend(self):
        events = [
            {
                "ph": "X",
                "cat": "gpu_user_annotation",
                "name": "step[EXTEND bs=1 toks=8192]",
                "pid": 0,
                "tid": 23,
                "ts": 10,
                "dur": 50,
            },
            {
                "ph": "X",
                "cat": "gpu_user_annotation",
                "name": "step[EXTEND bs=1 toks=8192]",
                "pid": 0,
                "tid": 7,
                "ts": 40,
                "dur": 5,
            },
            {
                "ph": "X",
                "cat": "gpu_user_annotation",
                "name": "step[EXTEND bs=1 toks=8192]",
                "pid": 0,
                "tid": 23,
                "ts": 70,
                "dur": 20,
            },
        ]

        windows = find_eagle_mtp_prefill_windows(events)

        self.assertEqual([(window.start_us, window.end_us) for window in windows], [(10, 90)])

    def test_eagle_mtp_decode_spans_verify_draft_extend_and_select(self):
        def annotation(name, ts, dur, tid=23):
            return {
                "ph": "X",
                "cat": "gpu_user_annotation",
                "name": name,
                "pid": 0,
                "tid": tid,
                "ts": ts,
                "dur": dur,
            }

        events = [
            annotation("step[TARGET_VERIFY bs=16]", 10, 30),
            annotation("step[TARGET_VERIFY bs=16]", 12, 3, tid=7),
            annotation("draft_extend", 42, 10),
            annotation("step[DRAFT_EXTEND_V2 bs=16]", 44, 7),
            annotation("draft", 53, 2),
            annotation("step[TARGET_VERIFY bs=16]", 60, 28),
            annotation("draft_extend", 90, 9),
            annotation("step[DRAFT_EXTEND_V2 bs=16]", 91, 7),
            annotation("draft", 100, 2),
        ]

        windows = find_eagle_mtp_decode_windows(events)

        self.assertEqual(
            [(window.start_us, window.end_us) for window in windows],
            [(10, 60), (60, 102)],
        )

    def test_eagle_mtp_cudagraph_decode_uses_adjacent_target_boundaries(self):
        def annotation(name, ts, dur, tid=23):
            return {
                "ph": "X",
                "cat": "gpu_user_annotation",
                "name": name,
                "pid": 0,
                "tid": tid,
                "ts": ts,
                "dur": dur,
            }

        events = [
            annotation("step[TARGET_VERIFY bs=16]", 10, 20),
            annotation("step[TARGET_VERIFY bs=16]", 12, 3, tid=7),
            {"ph": "X", "cat": "kernel", "name": "draft graph kernel", "ts": 35, "dur": 4},
            annotation("step[TARGET_VERIFY bs=16]", 50, 18),
            {"ph": "X", "cat": "kernel", "name": "draft graph kernel", "ts": 72, "dur": 3},
            annotation("step[TARGET_VERIFY bs=16]", 90, 19),
        ]

        windows = find_eagle_mtp_cudagraph_decode_windows(events)

        self.assertEqual(
            [(window.start_us, window.end_us) for window in windows],
            [(10, 50), (50, 90)],
        )

    def test_eagle_mtp_cudagraph_substages_use_primary_gpu_tracks(self):
        def annotation(name, ts, dur, tid=23):
            return {
                "ph": "X",
                "cat": "gpu_user_annotation",
                "name": name,
                "pid": 0,
                "tid": tid,
                "ts": ts,
                "dur": dur,
            }

        events = [
            annotation("step[TARGET_VERIFY bs=16]", 10, 20),
            annotation("step[TARGET_VERIFY bs=16]", 12, 3, tid=7),
            annotation("draft_extend", 32, 8),
            annotation("draft_extend", 34, 2, tid=8),
            annotation("draft", 43, 4),
        ]
        window = find_eagle_mtp_cudagraph_decode_windows(
            events + [annotation("step[TARGET_VERIFY bs=16]", 50, 18)]
        )[0]

        substages = find_eagle_mtp_cudagraph_substages(events, window)

        self.assertEqual(
            substages,
            [
                {
                    "step_index": 1,
                    "target_verify_us": [10.0, 30.0],
                    "mtp_draft_extend_us": [32.0, 40.0],
                    "draft_select_us": [43.0, 47.0],
                }
            ],
        )

    def test_gpu_step_annotation_defines_complete_forward_without_anchor(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            source_root = tmp / "src"
            source_file = source_root / "toy/model.py"
            source_file.parent.mkdir(parents=True)
            source_file.write_text("def semantic_call():\n    pass\n")

            trace_path = tmp / "trace.json"
            trace_path.write_text(
                json.dumps(
                    {
                        "record_shapes": 1,
                        "with_stack": 1,
                        "traceEvents": [
                            _python_event("toy/runner.py(1): toy_phase", 1, None, 0, 100),
                            _python_event("nn.Module: ToyLayer_0", 2, 1, 1, 90),
                            _python_event("toy/model.py(1): semantic_call", 3, 2, 2, 80),
                            {
                                "ph": "X",
                                "cat": "gpu_user_annotation",
                                "name": "step[DECODE bs=1]",
                                "pid": 0,
                                "tid": 0,
                                "ts": 10,
                                "dur": 20,
                                "args": {},
                            },
                            {
                                "ph": "X",
                                "cat": "kernel",
                                "name": "toy_kernel_main",
                                "pid": 0,
                                "tid": 0,
                                "ts": 12,
                                "dur": 2,
                                "args": {"stream": 1, "device": 0},
                            },
                        ],
                    }
                )
            )

            result = build_trace_mapping(
                trace_path=trace_path,
                source_root=source_root,
                source_repo="https://example.test/toy",
                source_commit="abc123",
                config_path=None,
                rank=0,
                phase="forward_decode",
                rules=TOY_RULES,
                skip_first=False,
                capture_contract={
                    "job_id": "job-1",
                    "phase": "decode",
                    "concurrency": 1,
                    "selected_formal_step": 7,
                },
            )

            self.assertEqual(
                result.manifest["window_selector"]["method"],
                "gpu_step_annotation",
            )
            self.assertEqual(result.manifest["window"]["start_us"], 10)
            self.assertEqual(result.validation["kernel_count"], 1)
            self.assertEqual(result.mappings[0].selected_node, "toy_node")
            self.assertEqual(
                len(result.manifest["selected_forward_events_sha256"]), 64
            )
            self.assertEqual(
                result.manifest["capture_contract"],
                {
                    "job_id": "job-1",
                    "phase": "decode",
                    "concurrency": 1,
                    "selected_formal_step": 7,
                },
            )

    def test_common_engine_uses_supplied_rules_not_qwen_names(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            source_root = tmp / "src"
            source_file = source_root / "toy/model.py"
            source_file.parent.mkdir(parents=True)
            source_file.write_text("def semantic_call():\n    pass\n")

            trace_path = tmp / "trace.json"
            trace_path.write_text(
                json.dumps(
                    {
                        "record_shapes": 1,
                        "with_stack": 1,
                        "traceEvents": [
                            _python_event("toy/runner.py(1): toy_phase", 1, None, 0, 100),
                            _python_event("nn.Module: ToyLayer_0", 2, 1, 1, 90),
                            _python_event("toy/model.py(1): semantic_call", 3, 2, 2, 80),
                            _python_event("toy/operator.py(1): op_call", 4, 3, 3, 70),
                            {
                                "ph": "X",
                                "cat": "kernel",
                                "name": "toy_anchor_kernel",
                                "pid": 0,
                                "tid": 0,
                                "ts": 10,
                                "dur": 1,
                                "args": {"stream": 1, "device": 0},
                            },
                            {
                                "ph": "X",
                                "cat": "kernel",
                                "name": "toy_kernel_main",
                                "pid": 0,
                                "tid": 0,
                                "ts": 11,
                                "dur": 2,
                                "args": {"stream": 1, "device": 0},
                            },
                        ],
                    }
                )
            )

            result = build_trace_mapping(
                trace_path=trace_path,
                source_root=source_root,
                source_repo="https://example.test/toy",
                source_commit="abc123",
                config_path=None,
                rank=0,
                phase="toy_phase",
                rules=TOY_RULES,
                expect_ms=None,
                n_iters=1,
            )

            self.assertTrue(result.validation["ok"], result.validation["errors"])
            nodes = {mapping.selected_node for mapping in result.mappings}
            self.assertIn("toy_node", nodes)
            mapped = next(m for m in result.mappings if m.selected_node == "toy_node")
            self.assertIn("toy/model.py", mapped.semantic_frame.raw)
            self.assertIn("ToyLayer", mapped.model_context_frame.raw)

    def test_no_cpu_op_kernel_uses_cuda_runtime_launch_time_for_stack(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            source_root = tmp / "src"
            source_file = source_root / "toy/model.py"
            source_file.parent.mkdir(parents=True)
            source_file.write_text("def semantic_call():\n    pass\n")

            trace_path = tmp / "trace.json"
            trace_path.write_text(
                json.dumps(
                    {
                        "record_shapes": 1,
                        "with_stack": 1,
                        "traceEvents": [
                            _python_event("toy/runner.py(1): toy_phase", 1, None, 0, 300),
                            _python_event("nn.Module: ToyLayer_0", 2, 1, 0, 100),
                            _python_event("toy/model.py(1): semantic_call", 3, 2, 0, 100),
                            _python_event("toy/operator.py(1): op_call", 4, 3, 0, 100),
                            _python_event("nn.Module: OtherLayer_0", 5, 1, 150, 100),
                            _python_event("toy/other.py(1): unrelated", 6, 5, 150, 100),
                            {
                                "ph": "X",
                                "cat": "cuda_runtime",
                                "name": "cudaLaunchKernel",
                                "pid": 1,
                                "tid": 1,
                                "ts": 20,
                                "dur": 5,
                                "args": {"correlation": 123},
                            },
                            {
                                "ph": "X",
                                "cat": "kernel",
                                "name": "toy_anchor_kernel",
                                "pid": 0,
                                "tid": 0,
                                "ts": 170,
                                "dur": 1,
                                "args": {"correlation": 123, "stream": 1, "device": 0},
                            },
                            {
                                "ph": "X",
                                "cat": "kernel",
                                "name": "toy_kernel_main",
                                "pid": 0,
                                "tid": 0,
                                "ts": 171,
                                "dur": 2,
                                "args": {"correlation": 123, "stream": 1, "device": 0},
                            },
                        ],
                    }
                )
            )

            result = build_trace_mapping(
                trace_path=trace_path,
                source_root=source_root,
                source_repo="https://example.test/toy",
                source_commit="abc123",
                config_path=None,
                rank=0,
                phase="toy_phase",
                rules=TOY_RULES,
                expect_ms=None,
                n_iters=1,
            )

            mapped = next(m for m in result.mappings if m.selected_node == "toy_node")
            self.assertIn("toy/model.py", mapped.semantic_frame.raw)
            self.assertNotIn("toy/other.py", mapped.semantic_frame.raw)


if __name__ == "__main__":
    unittest.main()
