from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.qwen40.build.build_qwen40_cudagraph_profile import (  # noqa: E402
    decode_steps,
    direct_kernel_mapping,
    expected_all_reduce_roles,
)


def _range(cat: str, name: str, tid: int, ts: float, dur: float):
    return {
        "cat": cat,
        "ph": "X",
        "name": name,
        "pid": 0,
        "tid": tid,
        "ts": ts,
        "dur": dur,
    }


class Qwen40CudaGraphProfileTest(unittest.TestCase):
    def test_collective_template_matches_pure_tp_schedule(self):
        roles = expected_all_reduce_roles()
        self.assertEqual(len(roles), 98)
        self.assertEqual(roles.count("top.tp_embedding_collective"), 1)
        self.assertEqual(roles.count("ple.tp_embedding_collective"), 1)
        self.assertEqual(roles.count("linear_layer.tp_attention_collective"), 36)
        self.assertEqual(roles.count("full_layer.tp_attention_collective"), 12)
        self.assertEqual(roles.count("moe.tp_output_collective"), 48)
        self.assertEqual(
            roles[:6],
            [
                "top.tp_embedding_collective",
                "linear_layer.tp_attention_collective",
                "moe.tp_output_collective",
                "ple.tp_embedding_collective",
                "linear_layer.tp_attention_collective",
                "moe.tp_output_collective",
            ],
        )

    def test_decode_steps_treats_gpu_stream_ranges_as_replay_copies(self):
        name = "step[DECODE bs=16]"
        events = []
        for index in range(6):
            events.append(_range("user_annotation", name, 1, index * 1000, 10))
            events.append(
                _range("gpu_user_annotation", name, 20, index * 1000 + 20, 800)
            )
            events.append(
                _range("gpu_user_annotation", name, 21, index * 1000 + 40, 200)
            )
            events.append(
                _range("cuda_runtime", "cudaGraphLaunch", 1, index * 1000 + 10, 5)
            )

        steps = decode_steps(events, 16)
        self.assertEqual(len(steps), 6)
        self.assertTrue(all(step["tid"] == 20 for step in steps))

    def test_decode_steps_rejects_mixed_graph_batch(self):
        name = "step[DECODE bs=16]"
        events = []
        for index in range(6):
            events.append(_range("user_annotation", name, 1, index * 1000, 10))
            gpu_name = "step[DECODE bs=1]" if index == 4 else name
            events.append(
                _range("gpu_user_annotation", gpu_name, 20, index * 1000 + 20, 800)
            )
            events.append(
                _range("cuda_runtime", "cudaGraphLaunch", 1, index * 1000 + 10, 5)
            )

        with self.assertRaisesRegex(ValueError, "mixed into bs=16"):
            decode_steps(events, 16)

    def test_direct_kernel_mapping_keeps_unique_semantics(self):
        self.assertEqual(
            direct_kernel_mapping("_causal_conv1d_update_kernel")[0],
            "linear_attention.causal_conv",
        )
        self.assertEqual(
            direct_kernel_mapping("void moe::dev::finalize::finalizeKernel")[0],
            "moe.combine",
        )
        self.assertEqual(direct_kernel_mapping("generic_dense_gemm"), (None, None))


if __name__ == "__main__":
    unittest.main()
