from __future__ import annotations

from typing import Any

import torch


def normalize_tensor(
    tensor: torch.Tensor,
    normalization_cfg: dict[str, Any] | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float | str | None]]:
    cfg = normalization_cfg or {}
    mode = str(cfg.get("mode", "none")).lower()
    clip_min = cfg.get("clip_min")
    clip_max = cfg.get("clip_max")
    scope = str(cfg.get("scope", "global")).lower()

    if clip_min is not None or clip_max is not None:
        tensor = torch.clamp(tensor, min=clip_min, max=clip_max)

    state: dict[str, torch.Tensor | float | str | None] = {
        "mode": mode,
        "scope": scope,
        "clip_min": clip_min,
        "clip_max": clip_max,
        "offset": None,
        "scale": None,
    }
    if mode == "none":
        return tensor, state
    if mode == "minmax":
        min_value, max_value = _normalization_range(tensor, scope)
        state["offset"] = min_value
        state["scale"] = max_value - min_value
        return (tensor - min_value) / (max_value - min_value + 1.0e-6), state
    if mode == "zscore":
        mean, std = _normalization_stats(tensor, scope)
        state["offset"] = mean
        state["scale"] = std
        return (tensor - mean) / (std + 1.0e-6), state
    raise ValueError(f"Unsupported normalization mode: {mode}")


def denormalize_tensor(
    tensor: torch.Tensor,
    state: dict[str, torch.Tensor | float | str | None],
) -> torch.Tensor:
    mode = str(state.get("mode", "none")).lower()
    if mode == "none":
        return tensor

    offset = state.get("offset")
    scale = state.get("scale")
    if not isinstance(offset, torch.Tensor) or not isinstance(scale, torch.Tensor):
        raise ValueError("Normalization state is missing tensor offset/scale for inverse transform.")
    offset = offset.to(device=tensor.device, dtype=tensor.dtype)
    scale = scale.to(device=tensor.device, dtype=tensor.dtype)
    restored = tensor * (scale + 1.0e-6) + offset
    clip_min = state.get("clip_min")
    clip_max = state.get("clip_max")
    if clip_min is not None or clip_max is not None:
        restored = torch.clamp(restored, min=clip_min, max=clip_max)
    return restored


def _normalization_range(tensor: torch.Tensor, scope: str) -> tuple[torch.Tensor, torch.Tensor]:
    if scope == "global":
        return tensor.amin(), tensor.amax()
    if scope == "channel":
        dims = tuple(range(1, tensor.ndim))
        return tensor.amin(dim=dims, keepdim=True), tensor.amax(dim=dims, keepdim=True)
    raise ValueError(f"Unsupported normalization scope: {scope}")


def _normalization_stats(tensor: torch.Tensor, scope: str) -> tuple[torch.Tensor, torch.Tensor]:
    if scope == "global":
        return tensor.mean(), tensor.std(unbiased=False)
    if scope == "channel":
        dims = tuple(range(1, tensor.ndim))
        return (
            tensor.mean(dim=dims, keepdim=True),
            tensor.std(dim=dims, keepdim=True, unbiased=False),
        )
    raise ValueError(f"Unsupported normalization scope: {scope}")
