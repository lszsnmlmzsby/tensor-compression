from __future__ import annotations

import random
from collections import Counter

import pytest
import torch

from scripts.build_tensor_patch_matched_qa import (
    build_state_records,
    finalize_numeric_rank_audit,
    grounding_target_from_source,
    numeric_group_rank_audit,
    separated_option_cells,
    verify_extreme_replay,
)
from tensor_compression.downstream.patch_qa_contract import PATCH_LATENT_AUDIT_FORMAT


def _source_record(state_ref: str = "density_s000000_t0000_r000_c000") -> dict:
    return {
        "qa_id": f"{state_ref}_source",
        "patch_id": state_ref,
        "state_ref": state_ref,
        "sample_index": 0,
        "time_index": 0,
        "field": "density",
        "top_left": [0, 0],
        "metadata": {
            "field": "density",
            "grid_shape": [16, 16],
            "coordinate_origin": 1,
        },
        "latent_audit": {
            "format": PATCH_LATENT_AUDIT_FORMAT,
            "mean": 10.0,
            "std": 2.0,
            "scale": 2.0,
        },
    }


def _values() -> torch.Tensor:
    return torch.linspace(-1.75, 1.75, 16 * 16, dtype=torch.float32).reshape(16, 16)


def _extreme_record(source: dict) -> dict:
    question = (
        "The tensor soft tokens encode the standardized matrix. "
        "Which quadrant contains the maximum value?"
    )
    return {
        **source,
        "qa_id": f"{source['state_ref']}_extreme_v0",
        "task_type": "extreme_quadrant",
        "question_variant": 0,
        "question": question,
        "query": question,
        "choices": ["A", "B", "C", "D"],
        "answer": "D",
    }


def _build(
    spatial_family: str,
    extreme_audit_counts: Counter[str] | None = None,
) -> list[dict]:
    source = _source_record()
    return build_state_records(
        source,
        _extreme_record(source),
        _values(),
        seed=42,
        numeric_gap=0.25,
        region_gap=0.05,
        region_size=4,
        decimal_places=6,
        spatial_family=spatial_family,
        extreme_audit_counts=extreme_audit_counts,
    )


def test_matched_builder_emits_nine_atomic_records_without_numeric_shortcuts() -> None:
    extreme_audit_counts: Counter[str] = Counter()
    records = _build("point", extreme_audit_counts)

    assert len(records) == 9
    assert [record["task_type"] for record in records[:6]] == [
        "normalized_point_value",
    ] * 3 + ["raw_point_value_with_stats"] * 3
    assert len({record["matched_group"]["batch_group_id"] for record in records}) == 3
    assert all(record["matched_group"]["batch_group_size"] == 3 for record in records)
    assert dict(extreme_audit_counts) == {
        "records": 1,
        "maximum_records": 1,
        "unique_cell_records": 1,
        "fp16_extreme_position_count": 1,
    }

    normalized_specs = [record["matched_group"]["query_spec"] for record in records[:3]]
    raw_specs = [record["matched_group"]["query_spec"] for record in records[3:6]]
    assert normalized_specs == raw_specs
    assert [record["answer"] for record in records[:3]] != [
        record["answer"] for record in records[3:6]
    ]
    assert all(
        set(record["prompt_data"]) >= {
            "field",
            "mean",
            "std",
            "scale",
            "row",
            "col",
            "option_text",
        }
        for record in records[3:6]
    )

    for group in (records[:3], records[3:6]):
        audit = numeric_group_rank_audit(group)
        assert audit["distractor_numeric_rank_1_based"] in {1, 2, 3, 4}
        assert sorted(audit["correct_numeric_ranks_1_based"] + [
            audit["distractor_numeric_rank_1_based"]
        ]) == [1, 2, 3, 4]


def test_extreme_replay_accepts_supported_cross_quadrant_fp16_tie() -> None:
    source = _source_record()
    record = {**_extreme_record(source), "answer": "C"}
    values = torch.full((16, 16), -1.0, dtype=torch.float32)
    values[12, 2] = 5.0
    values[12, 12] = 5.0

    audit = verify_extreme_replay(record, values)

    assert audit == {
        "extreme": "maximum",
        "fp16_position_count": 2,
        "fp16_quadrants": ["C", "D"],
        "tie_scope": "cross_quadrant_tie",
    }
    with pytest.raises(ValueError, match="do not support the source float32 replay label"):
        verify_extreme_replay({**record, "answer": "A"}, values)


def test_point_and_region_role_swaps_reverse_targets_and_labels() -> None:
    point_records = _build("point")
    point_a, point_b = point_records[6:8]
    point_a_spec = point_a["matched_group"]["query_spec"]
    point_b_spec = point_b["matched_group"]["query_spec"]
    assert point_a_spec["a"] == point_b_spec["b"]
    assert point_a_spec["b"] == point_b_spec["a"]
    assert {point_a["answer"], point_b["answer"]} == {"A", "B"}
    assert point_records[8]["matched_group"]["query_spec"]["type"] == "none"

    region_records = _build("region")
    region_a, region_b = region_records[6:8]
    region_a_spec = region_a["matched_group"]["query_spec"]
    region_b_spec = region_b["matched_group"]["query_spec"]
    assert region_a_spec["a"] == region_b_spec["b"]
    assert region_a_spec["b"] == region_b_spec["a"]
    assert {region_a["answer"], region_b["answer"]} == {"A", "B"}


def test_distractor_numeric_rank_is_not_fixed_by_sampler() -> None:
    observed_ranks: set[int] = set()
    values = _values()
    for seed in range(256):
        option_values, _coordinates, target_indices = separated_option_cells(
            values,
            minimum_gap=0.25,
            rng=random.Random(seed),
        )
        distractor_index = next(iter(set(range(4)) - set(target_indices)))
        ordered = sorted(range(4), key=option_values.__getitem__)
        observed_ranks.add(ordered.index(distractor_index) + 1)

    assert observed_ranks == {1, 2, 3, 4}


def test_numeric_rank_summary_rejects_a_fixed_distractor_rank() -> None:
    balanced = []
    fixed = []
    for task in ("normalized_point_value", "raw_point_value_with_stats"):
        for index in range(16):
            rank = index % 4 + 1
            balanced.append(
                {
                    "task_type": task,
                    "distractor_label": "ABCD"[index % 4],
                    "distractor_numeric_rank_1_based": rank,
                    "correct_numeric_ranks_1_based": [
                        value for value in (1, 2, 3, 4) if value != rank
                    ],
                }
            )
            fixed.append(
                {
                    "task_type": task,
                    "distractor_label": "D",
                    "distractor_numeric_rank_1_based": 4,
                    "correct_numeric_ranks_1_based": [1, 2, 3],
                }
            )

    summary = finalize_numeric_rank_audit(balanced, expected_groups_per_task=16)
    assert all(item["all_four_distractor_ranks_observed"] for item in summary.values())
    with pytest.raises(ValueError, match="fixed shortcut"):
        finalize_numeric_rank_audit(fixed, expected_groups_per_task=16)


@pytest.mark.parametrize(
    ("task", "query", "expected_type"),
    [
        (
            "normalized_point_value",
            "Which option is closest to z at row 2, column 4? Options: A: 0; B: 1; C: 2; D: 3.",
            "point",
        ),
        (
            "raw_point_value_with_stats",
            "Which option is closest to the original value x at row 3, column 5? "
            "Options: A: 0; B: 1; C: 2; D: 3.",
            "point",
        ),
        (
            "point_compare",
            "Which location is larger: A at row 2, column 3, or B at row 8, column 9?",
            "point_pair",
        ),
        (
            "region_mean_compare",
            "Compare the mean values of two 4 by 4 regions. Region A starts at row 2, column 3; "
            "region B starts at row 8, column 9.",
            "region_pair",
        ),
        (
            "extreme_quadrant",
            "Which quadrant contains the minimum value?",
            "none",
        ),
    ],
)
def test_evaluation_grounding_targets_cover_every_source_task(
    task: str,
    query: str,
    expected_type: str,
) -> None:
    record = {
        **_source_record(),
        "task_type": task,
        "query": query,
        "question": query,
    }

    target = grounding_target_from_source(record)

    assert target["type"] == expected_type
    assert target["coordinate_origin"] == 0
