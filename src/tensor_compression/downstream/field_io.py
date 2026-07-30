from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F


def resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    requested = str(device).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def resize_chw_batch(frames: torch.Tensor, input_size: Sequence[int]) -> torch.Tensor:
    target_size = tuple(int(dim) for dim in input_size)
    if tuple(frames.shape[-2:]) == target_size:
        return frames
    return F.interpolate(frames, size=target_size, mode="bilinear", align_corners=False)


def _parse_fields(raw: Any) -> list[str] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        values = [part.strip() for part in raw.split(",") if part.strip()]
        return values or None
    if isinstance(raw, Sequence):
        return [str(item) for item in raw]
    return [str(raw)]


def resolve_checkpoint_field_keys(config: Mapping[str, Any] | None) -> list[str] | None:
    if not isinstance(config, Mapping):
        return None
    data_cfg = config.get("data")
    if not isinstance(data_cfg, Mapping):
        return None
    dataset_cfg = data_cfg.get("dataset")
    if not isinstance(dataset_cfg, Mapping):
        return None

    multi_keys = _parse_fields(dataset_cfg.get("hdf5_dataset_keys"))
    if multi_keys:
        return multi_keys
    single_key = dataset_cfg.get("hdf5_dataset_key") or dataset_cfg.get("field_key")
    return [str(single_key)] if single_key else None


def validate_checkpoint_field_keys_against_model(
    config: Mapping[str, Any] | None,
    field_keys: Sequence[str] | None,
) -> None:
    if not isinstance(config, Mapping) or not field_keys:
        return
    model_cfg = config.get("model")
    if not isinstance(model_cfg, Mapping):
        return
    in_channels = model_cfg.get("in_channels")
    if in_channels is not None and int(in_channels) != len(field_keys):
        raise ValueError(
            "Checkpoint field order is inconsistent with model.in_channels. "
            f"field_keys={list(field_keys)}, in_channels={in_channels}."
        )
