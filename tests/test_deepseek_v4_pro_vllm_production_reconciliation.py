from __future__ import annotations

from models.deepseek_v4_pro.build.reconcile_deepseek_v4_pro_vllm_production import (
    GRAPH_DEPENDENCY_COPY,
    GRAPH_DEPENDENCY_COPIES,
    OCCURRENCE_COUNT,
    _close_decode_graph_dependencies,
    _production_scopes,
    _transfer_scope,
    schedule_family,
)


def test_occurrence_transfer_preserves_order_for_repeated_exact_identity() -> None:
    source = [
        {
            "event_id": "e0",
            "kernel_name": "shared_kernel<int>",
            "selected_node": "csa_indexer.q_projection",
        },
        {
            "event_id": "e1",
            "kernel_name": "shared_kernel<int>",
            "selected_node": "csa_attention.q_b",
        },
    ]
    production = [
        {"kernel_name": "shared_kernel<int>", "node": None},
        {"kernel_name": "shared_kernel<int>", "node": None},
    ]

    _transfer_scope(source, production, method_prefix="layer_02.attention_bounded")

    assert [row["node"] for row in production] == [
        "csa_indexer.q_projection",
        "csa_attention.q_b",
    ]
    assert [row["eager_event_ids"] for row in production] == [["e0"], ["e1"]]


def test_routed_bmm_family_keeps_math_form_but_normalizes_schedule() -> None:
    assert schedule_family(
        "bmm_Bfloat16_MxE2m1MxE4m3_Fp32_shape_s3_clmp_dynB"
    ) == schedule_family(
        "bmm_Bfloat16_MxE2m1MxE4m3_Fp32_other_s8_bias"
    )
    assert schedule_family(
        "bmm_MxE4m3_MxE2m1MxE4m3_Fp32_gate_s3"
    ) != schedule_family(
        "bmm_Bfloat16_MxE2m1MxE4m3_Fp32_down_s3"
    )


def test_separate_mhc_schedule_closes_all_122_occurrences() -> None:
    production = []
    for _ in range(OCCURRENCE_COUNT):
        production.extend(
            [
                {"kernel_name": "sm100_tf32_hc_prenorm_gemm", "node": None},
                {
                    "kernel_name": "mhc_pre_big_fuse_with_norm_tilelang_kernel",
                    "node": None,
                },
                {"kernel_name": "mhc_post_tilelang_kernel", "node": None},
            ]
        )

    prefix, scopes, suffix, path = _production_scopes(production)

    assert prefix == (0, 0)
    assert len(scopes) == OCCURRENCE_COUNT
    assert scopes[0] == (0, 3)
    assert scopes[-1] == (3 * (OCCURRENCE_COUNT - 1), len(production))
    assert suffix == (len(production), len(production))
    assert path == "separate_post_pre"


def _owner(node: str, occurrence_id: str | None = None) -> dict:
    row = {
        "kernel_name": f"kernel:{node}",
        "node": node,
        "eager_event_ids": [f"eager:{node}"],
    }
    if occurrence_id:
        layer, substage = occurrence_id.split(".")
        row.update(
            {
                "occurrence_id": occurrence_id,
                "layer_id": int(layer.removeprefix("layer_")),
                "substage": substage,
            }
        )
    return row


def _copy() -> dict:
    return {"kernel_name": GRAPH_DEPENDENCY_COPY, "node": None}


def test_decode_graph_dependencies_close_exact_boundary_patterns() -> None:
    production = [_copy(), _owner("top.tp_embedding_output_collective"), _copy()]
    prefix = (0, len(production))
    scopes = []
    for layer in range(61):
        attention_id = f"layer_{layer:02d}.attention"
        attention_start = len(production)
        production.extend(
            [
                _copy(),
                _owner("csa_attention.tp_csa_output_collective", attention_id),
                _copy(),
            ]
        )
        scopes.append((attention_start, len(production)))

        ffn_id = f"layer_{layer:02d}.feed_forward"
        ffn_start = len(production)
        production.extend(
            [
                _copy(),
                _owner("moe.combine", ffn_id),
                _copy(),
                _owner("moe.tp_moe_output_collective", ffn_id),
                _copy(),
            ]
        )
        scopes.append((ffn_start, len(production)))
    suffix = (len(production), len(production) + 2)
    production.extend(
        [{"kernel_name": "memcpy128", "node": None}, _owner("final_hc_read.read")]
    )

    assigned = _close_decode_graph_dependencies(
        production, prefix=prefix, scopes=scopes, suffix=suffix
    )

    assert assigned == 308
    assert all(row["node"] for row in production)
    assert all(row["eager_event_ids"] for row in production)
    assert all(
        row.get("launch_group_id")
        for row in production
        if row["kernel_name"] in GRAPH_DEPENDENCY_COPIES
    )
