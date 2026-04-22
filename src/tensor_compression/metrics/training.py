from __future__ import annotations

from typing import Any

import torch

from tensor_compression.data.normalization import denormalize_tensor, normalize_tensor
from tensor_compression.metrics.reconstruction import compute_reconstruction_metrics


def compute_training_reconstruction_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    physical_target: torch.Tensor | None = None,
    normalization_cfg: dict[str, Any] | None = None,
) -> dict[str, float]:
    normalized_metrics = compute_reconstruction_metrics(prediction, target)
    metrics = dict(normalized_metrics)
    for key, value in normalized_metrics.items():
        metrics[f"normalized_{key}"] = float(value)

    physical_metrics = _compute_physical_metrics(prediction, physical_target, normalization_cfg)
    if physical_metrics is not None:
        for key, value in physical_metrics.items():
            metrics[f"physical_{key}"] = float(value)
    return metrics


def _compute_physical_metrics(
    prediction: torch.Tensor,
    physical_target: torch.Tensor | None,
    normalization_cfg: dict[str, Any] | None,
) -> dict[str, float] | None:
    if physical_target is None or normalization_cfg is None:
        return None
    states = _compute_normalization_states(physical_target, normalization_cfg)
    if states is None:
        return None

    restored_prediction = torch.stack(
        [denormalize_tensor(sample.detach().cpu(), state) for sample, state in zip(prediction, states)],
        dim=0,
    )
    return compute_reconstruction_metrics(restored_prediction, physical_target.detach().cpu())


def _compute_normalization_states(
    physical_target: torch.Tensor,
    normalization_cfg: dict[str, Any] | None,
) -> list[dict[str, torch.Tensor | float | str | None]] | None:
    states: list[dict[str, torch.Tensor | float | str | None]] = []
    for sample in physical_target.detach().cpu():
        _, state = normalize_tensor(sample, normalization_cfg)
        states.append(state)
    return states
