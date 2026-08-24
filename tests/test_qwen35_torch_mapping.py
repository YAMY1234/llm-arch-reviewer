from __future__ import annotations

import gzip
import json

from models.qwen35.profile.qwen35_torch_mapping import load_trt_torch_steps


def test_load_trt_torch_steps_uses_gpu_annotation_and_preserves_kernel(tmp_path):
    label = "[Executor] _forward_step 71: 0 ctx reqs, 0 ctx tokens, 32 gen reqs"
    trace = {
        "traceEvents": [
            {
                "cat": "user_annotation",
                "ph": "X",
                "name": label,
                "ts": 90.0,
                "dur": 25.0,
                "pid": 1,
                "tid": 1,
            },
            {
                "cat": "gpu_user_annotation",
                "ph": "X",
                "name": label,
                "ts": 100.0,
                "dur": 10.0,
                "pid": 2,
                "tid": 7,
            },
            {
                "cat": "kernel",
                "ph": "X",
                "name": "concrete::kernel<int>",
                "ts": 101.0,
                "dur": 3.5,
                "args": {"stream": 17, "device": 2, "correlation": 9},
            },
            {
                "cat": "kernel",
                "ph": "X",
                "name": "outside",
                "ts": 112.0,
                "dur": 1.0,
                "args": {},
            },
            {
                "cat": "cuda_runtime",
                "ph": "X",
                "name": "cudaGraphLaunch",
                "ts": 95.0,
                "dur": 1.0,
            },
        ]
    }
    path = tmp_path / "rank2.trace.json.gz"
    with gzip.open(path, "wt") as output:
        json.dump(trace, output)

    steps = load_trt_torch_steps(path, rank=2)

    assert len(steps) == 1
    step = steps[0]
    assert (step.step_id, step.generation_reqs, step.graph_launch_count) == (71, 32, 1)
    assert step.cpu_wall_us == 25.0
    assert [kernel.name for kernel in step.kernels] == ["concrete::kernel<int>"]
    assert step.kernels[0].duration_us == 3.5


def test_load_trt_torch_steps_reconstructs_python_boundary_graph_occurrences(tmp_path):
    trace = {"traceEvents": []}
    for step_index, start in enumerate((100.0, 200.0)):
        trace["traceEvents"].extend(
            [
                {
                    "cat": "python_function",
                    "ph": "X",
                    "name": "py_executor.py(6781): _forward_step",
                    "ts": start,
                    "dur": 20.0,
                    "pid": 1,
                    "tid": 7,
                },
                {
                    "cat": "cpu_op",
                    "ph": "X",
                    "name": "aten::copy_",
                    "ts": start + 1.0,
                    "dur": 1.0,
                    "pid": 1,
                    "tid": 7,
                    "args": {"External id": 100 + step_index},
                },
                {
                    "cat": "cuda_runtime",
                    "ph": "X",
                    "name": "cudaGraphLaunch_ptsz",
                    "ts": start + 2.0,
                    "dur": 1.0,
                    "pid": 1,
                    "tid": 7,
                    "args": {"External id": 100 + step_index},
                },
                {
                    "cat": "kernel",
                    "ph": "X",
                    "name": f"direct_{step_index}",
                    "ts": start + 5.0,
                    "dur": 1.0,
                    "args": {
                        "External id": 100 + step_index,
                        "graph id": 0,
                        "graph node id": 0,
                        "device": 0,
                    },
                },
                {
                    "cat": "kernel",
                    "ph": "X",
                    "name": f"graph_{step_index}",
                    # Deliberately detached from the CPU annotation interval.
                    "ts": 1000.0 + step_index * 50.0,
                    "dur": 3.0,
                    "args": {
                        "graph id": 9,
                        "graph node id": 11,
                        "device": 0,
                    },
                },
            ]
        )
    path = tmp_path / "rank0.trace.json"
    path.write_text(json.dumps(trace))

    steps = load_trt_torch_steps(path, rank=0)

    assert [step.step_id for step in steps] == [0, 1]
    assert [step.generation_reqs for step in steps] == [32, 32]
    assert [step.graph_launch_count for step in steps] == [1, 1]
    assert [[kernel.name for kernel in step.kernels] for step in steps] == [
        ["direct_0", "graph_0"],
        ["direct_1", "graph_1"],
    ]
