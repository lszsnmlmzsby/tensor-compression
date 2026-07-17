from __future__ import annotations

import pytest
import torch

from scripts.scan_tensor_teacher_layers import (
    paired_metrics,
    parse_layer_indices,
    perturb_patches,
    readout_hidden,
    readout_indices,
    representation_metrics,
)


def test_readout_position_is_preserved_across_hidden_layers() -> None:
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    indices = readout_indices(attention_mask)
    assert indices.tolist() == [2, 1]

    layer_one = torch.arange(2 * 4 * 3).reshape(2, 4, 3)
    layer_two = layer_one + 100
    assert torch.equal(readout_hidden(layer_one, indices), torch.stack([layer_one[0, 2], layer_one[1, 1]]))
    assert torch.equal(readout_hidden(layer_two, indices), torch.stack([layer_two[0, 2], layer_two[1, 1]]))


def test_parse_layer_indices_defaults_and_validates_bounds() -> None:
    assert parse_layer_indices("all", 4) == [1, 2, 3, 4]
    assert parse_layer_indices("4,2,2,0", 4) == [0, 2, 4]
    with pytest.raises(ValueError, match="outside"):
        parse_layer_indices("5", 4)


def test_representation_metrics_detect_collapsed_vectors() -> None:
    collapsed = torch.ones(4, 3)
    spread = torch.eye(4)
    collapsed_metrics = representation_metrics(collapsed)
    spread_metrics = representation_metrics(spread)
    assert collapsed_metrics["off_diagonal_cosine_mean"] == pytest.approx(1.0)
    assert spread_metrics["off_diagonal_cosine_mean"] == pytest.approx(0.0)
    assert spread_metrics["effective_rank_participation"] > collapsed_metrics["effective_rank_participation"]


def test_perturbation_is_deterministic_and_paired_metrics_are_bounded() -> None:
    patches = torch.arange(2 * 1 * 4 * 4, dtype=torch.float32).reshape(2, 1, 4, 4)
    first = perturb_patches(patches, 0.01)
    second = perturb_patches(patches, 0.01)
    assert torch.equal(first, second)
    assert not torch.equal(first, patches)

    original = patches.flatten(1)
    changed = first.flatten(1)
    metrics = paired_metrics(original, changed)
    assert -1.0 <= metrics["perturbed_pair_cosine_mean"] <= 1.0
    assert metrics["perturbed_relative_l2_mean"] > 0.0
