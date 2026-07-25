from __future__ import annotations

import hashlib
import random
from collections import Counter
from pathlib import Path

import pytest
import torch

from scripts.build_tensor_patch_matched_qa import (
    assign_spatial_families,
    build_state_records,
    evaluation_record_replay,
    finalize_numeric_rank_audit,
    grounding_target_from_source,
    load_preserved_z,
    numeric_group_rank_audit,
    separated_option_cells,
    state_selection_summary,
    train_state_capability,
    verify_extreme_replay,
)
from tensor_compression.downstream.patch_qa_contract import (
    PATCH_LATENT_AUDIT_FORMAT,
    PATCH_LATENT_FORMAT,
)


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


def _float32_scale(raw_std: float) -> float:
    return float((torch.tensor(float(raw_std), dtype=torch.float32) + 1.0e-6).item())


def _write_latent_fixture(
    tmp_path: Path,
    values: torch.Tensor,
    *,
    raw_std: float,
    scale: float | None = None,
) -> tuple[dict, Path, dict]:
    checkpoint = tmp_path / "alignment_best.pt"
    checkpoint.write_bytes(b"stage1-for-matched-builder-test")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    record = _source_record()
    record["latent_audit"] = {
        "format": PATCH_LATENT_AUDIT_FORMAT,
        "mean": 10.0,
        "std": float(raw_std),
        "scale": _float32_scale(raw_std) if scale is None else float(scale),
    }
    normalization = {
        "mode": "zscore",
        "scope": "channel",
        "stats_path": None,
        "clip_min": None,
        "clip_max": None,
    }
    latent_path = tmp_path / f"{record['state_ref']}.pt"
    latent_map = torch.zeros(2, 16, 16, dtype=torch.float16)
    latent_map[0] = values.to(dtype=torch.float16)
    latent_map[1] = 0.25
    torch.save(
        {
            "format": PATCH_LATENT_FORMAT,
            "latent_map": latent_map,
            "patch_id": record["patch_id"],
            "field": record["field"],
            "sample_index": record["sample_index"],
            "time_index": record["time_index"],
            "top_left": record["top_left"],
            "alignment_checkpoint": str(checkpoint.resolve()),
            "alignment_checkpoint_sha256": checkpoint_sha,
            "encoder_input_normalization": normalization,
            "qa_value_space": {
                "mode": "per_patch_zscore",
                "mean": record["latent_audit"]["mean"],
                "std": record["latent_audit"]["std"],
                "scale": record["latent_audit"]["scale"],
            },
        },
        latent_path,
    )
    contract = {
        "alignment_checkpoint": str(checkpoint.resolve()),
        "alignment_checkpoint_sha256": checkpoint_sha,
        "normalization": normalization,
        "latent_shape": [2, 16, 16],
        "storage_dtype": "float16",
    }
    return record, latent_path, contract


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


def test_preserved_z_accepts_constant_patch_when_metadata_declares_it(tmp_path: Path) -> None:
    record, latent_path, contract = _write_latent_fixture(
        tmp_path,
        torch.zeros(16, 16),
        raw_std=0.0,
    )

    values = load_preserved_z(record, latent_path, contract)

    assert torch.equal(values, torch.zeros_like(values))


@pytest.mark.parametrize("raw_std", [1.0e-6, 1.0])
def test_preserved_z_accepts_metadata_conditioned_variance(
    tmp_path: Path,
    raw_std: float,
) -> None:
    expected_z_std = raw_std / _float32_scale(raw_std)
    values = _values()
    values = (values - values.mean()) / values.std(unbiased=False) * expected_z_std
    record, latent_path, contract = _write_latent_fixture(
        tmp_path,
        values,
        raw_std=raw_std,
    )

    loaded = load_preserved_z(record, latent_path, contract)

    assert float(loaded.std(unbiased=False).item()) == pytest.approx(expected_z_std, abs=5.0e-3)


def test_preserved_z_rejects_zero_channel_with_nonconstant_metadata(tmp_path: Path) -> None:
    record, latent_path, contract = _write_latent_fixture(
        tmp_path,
        torch.zeros(16, 16),
        raw_std=1.0,
    )

    with pytest.raises(ValueError, match="constant despite non-degenerate"):
        load_preserved_z(record, latent_path, contract)


def test_preserved_z_rejects_stale_epsilon_scale(tmp_path: Path) -> None:
    record, latent_path, contract = _write_latent_fixture(
        tmp_path,
        torch.zeros(16, 16),
        raw_std=0.0,
        scale=2.0e-6,
    )

    with pytest.raises(ValueError, match="stale scale"):
        load_preserved_z(record, latent_path, contract)


def test_train_capability_excludes_constants_but_keeps_sparse_point_states() -> None:
    constant = train_state_capability(
        torch.zeros(16, 16),
        numeric_gap=0.5,
        region_gap=0.2,
        region_size=4,
    )
    assert constant["eligible"] is False
    assert constant["exclusion_reason"] == "constant_preserved_channel"

    sparse = torch.zeros(16, 16)
    sparse[0, 0] = -0.5
    sparse[-1, -1] = 0.5
    capability = train_state_capability(
        sparse,
        numeric_gap=0.5,
        region_gap=0.2,
        region_size=4,
    )
    assert capability["eligible"] is True
    assert capability["point_pair_supported"] is True
    assert capability["region_pair_supported"] is False
    option_values, _coordinates, target_indices = separated_option_cells(
        sparse,
        minimum_gap=0.5,
        rng=random.Random(7),
    )
    assert len(target_indices) == 3
    assert len({option_values[index] for index in target_indices}) == 3


def test_family_assignment_and_selection_audit_respect_capabilities() -> None:
    point_only = "density_s000000_t0000_r000_c000"
    region_ready = "density_s000001_t0000_r000_c000"
    records = {
        point_only: _source_record(point_only),
        region_ready: _source_record(region_ready),
    }
    capabilities = {
        point_only: {
            "eligible": True,
            "point_pair_supported": True,
            "region_pair_supported": False,
        },
        region_ready: {
            "eligible": True,
            "point_pair_supported": True,
            "region_pair_supported": True,
        },
    }

    families = assign_spatial_families(records, capabilities, seed=42)
    summary = state_selection_summary(records, {point_only}, {region_ready: "test_exclusion"})

    assert families[point_only] == "point"
    assert families[region_ready] == "region"
    assert summary["source_states"] == 2
    assert summary["included_states"] == 1
    assert summary["excluded_by_reason"] == {"test_exclusion": 1}


def test_evaluation_replay_rejects_weak_or_duplicate_point_comparisons() -> None:
    source = _source_record()
    values = torch.zeros(16, 16)
    values[1, 2] = -1.0
    values[7, 8] = 1.0
    strong_query = (
        "Which location has the larger value: A at row 2, column 3, "
        "or B at row 8, column 9?"
    )
    strong = {
        **source,
        "task_type": "point_compare",
        "query": strong_query,
        "question": strong_query,
        "choices": ["A", "B"],
        "answer": "B",
    }
    replay = evaluation_record_replay(strong, values, numeric_gap=0.5, region_gap=0.2)
    assert replay["eligible"] is True

    weak_values = values.clone()
    weak_values[7, 8] = -0.8
    weak = evaluation_record_replay(strong, weak_values, numeric_gap=0.5, region_gap=0.2)
    assert weak["eligible"] is False
    assert weak["reason"] == "insufficient_point_compare_stored_fp16_gap"

    duplicate_query = (
        "Which location has the larger value: A at row 2, column 3, "
        "or B at row 2, column 3?"
    )
    duplicate = evaluation_record_replay(
        {**strong, "query": duplicate_query, "question": duplicate_query},
        values,
        numeric_gap=0.5,
        region_gap=0.2,
    )
    assert duplicate["eligible"] is False
    assert duplicate["reason"] == "duplicate_point_compare_coordinates"


def test_evaluation_replay_validates_region_gap_and_answer() -> None:
    values = torch.zeros(16, 16)
    values[8:12, 8:12] = 1.0
    query = (
        "Compare the mean values of two 4 by 4 regions. Region A starts at row 1, column 1; "
        "region B starts at row 9, column 9. Which region has the larger mean?"
    )
    record = {
        **_source_record(),
        "task_type": "region_mean_compare",
        "query": query,
        "question": query,
        "choices": ["A", "B"],
        "answer": "B",
    }

    replay = evaluation_record_replay(record, values, numeric_gap=0.5, region_gap=0.2)
    stale = evaluation_record_replay(
        {**record, "answer": "A"},
        values,
        numeric_gap=0.5,
        region_gap=0.2,
    )

    assert replay["eligible"] is True
    assert stale["eligible"] is False
    assert stale["reason"] == "stale_region_compare_stored_fp16_answer"


@pytest.mark.parametrize(
    ("task", "query", "answer"),
    [
        (
            "normalized_point_value",
            "Which option is closest to z at row 2, column 3? "
            "Options: A: -1; B: 0; C: 1; D: 2.",
            "C",
        ),
        (
            "raw_point_value_with_stats",
            "Recover x where mean is 10 and scale is 2. Which option is closest to the original "
            "value x at row 2, column 3? Options: A: 8; B: 10; C: 12; D: 14.",
            "C",
        ),
    ],
)
def test_evaluation_replay_uses_stored_fp16_for_numeric_answers(
    task: str,
    query: str,
    answer: str,
) -> None:
    values = torch.zeros(16, 16)
    values[1, 2] = 1.0
    record = {
        **_source_record(),
        "task_type": task,
        "query": query,
        "question": query,
        "choices": ["A", "B", "C", "D"],
        "answer": answer,
    }

    replay = evaluation_record_replay(record, values, numeric_gap=0.5, region_gap=0.2)
    stale = evaluation_record_replay(
        {**record, "answer": "A"},
        values,
        numeric_gap=0.5,
        region_gap=0.2,
    )

    assert replay["eligible"] is True
    assert stale["eligible"] is False
    assert stale["reason"] == f"stale_{task}_stored_fp16_answer"


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
