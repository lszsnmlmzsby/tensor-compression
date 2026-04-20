from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import yaml


def _infer_dataset_channels(dataset_cfg: dict[str, Any]) -> int | None:
    if not isinstance(dataset_cfg, dict):
        return None
    explicit_channels = dataset_cfg.get("channels")
    dataset_keys = dataset_cfg.get("hdf5_dataset_keys")
    if isinstance(dataset_keys, (list, tuple)) and dataset_keys:
        inferred = len(dataset_keys)
        if explicit_channels is not None and int(explicit_channels) != inferred:
            raise ValueError(
                "Configured data.dataset.channels does not match the number of "
                f"hdf5_dataset_keys ({inferred})."
            )
        return inferred
    if dataset_cfg.get("hdf5_dataset_key") or dataset_cfg.get("field_key"):
        if explicit_channels is not None and int(explicit_channels) != 1:
            raise ValueError(
                "Configured data.dataset.channels must be 1 when using a single "
                "hdf5_dataset_key/field_key."
            )
        return 1
    if explicit_channels is not None:
        return int(explicit_channels)
    return None


def _synchronize_channel_config(config: dict[str, Any]) -> dict[str, Any]:
    data_cfg = config.get("data")
    model_cfg = config.get("model")
    if not isinstance(data_cfg, dict) or not isinstance(model_cfg, dict):
        return config

    dataset_cfg = data_cfg.get("dataset")
    if not isinstance(dataset_cfg, dict):
        return config

    inferred_channels = _infer_dataset_channels(dataset_cfg)
    if inferred_channels is None:
        return config

    dataset_cfg["channels"] = inferred_channels

    in_channels = model_cfg.get("in_channels")
    out_channels = model_cfg.get("out_channels")
    if in_channels is None:
        model_cfg["in_channels"] = inferred_channels
    if out_channels is None:
        model_cfg["out_channels"] = inferred_channels

    if int(model_cfg["in_channels"]) != inferred_channels:
        raise ValueError(
            "Configured channel count is inconsistent between data.dataset.channels "
            f"({inferred_channels}) and model.in_channels ({model_cfg['in_channels']})."
        )
    if int(model_cfg["out_channels"]) != inferred_channels:
        raise ValueError(
            "Configured channel count is inconsistent between data.dataset.channels "
            f"({inferred_channels}) and model.out_channels ({model_cfg['out_channels']})."
        )

    normalization_cfg = dataset_cfg.get("normalization")
    if normalization_cfg is None:
        dataset_cfg["normalization"] = {}
    elif not isinstance(normalization_cfg, dict):
        raise ValueError("data.dataset.normalization must be a mapping when provided.")

    return config


def _synchronize_latent_config(config: dict[str, Any]) -> dict[str, Any]:
    model_cfg = config.get("model")
    if not isinstance(model_cfg, dict):
        return config

    latent_dim = model_cfg.get("latent_dim")
    if latent_dim is None:
        return config

    if not bool(model_cfg.get("latent_dim_scale_with_channels", False)):
        return config

    channels = model_cfg.get("in_channels")
    if channels is None:
        return config

    reference_channels = int(model_cfg.get("latent_dim_reference_channels", 1))
    if reference_channels <= 0:
        raise ValueError("model.latent_dim_reference_channels must be a positive integer.")

    round_to = int(model_cfg.get("latent_dim_round_to", 1))
    if round_to <= 0:
        raise ValueError("model.latent_dim_round_to must be a positive integer.")

    base_latent_dim = int(model_cfg.get("latent_dim_base", latent_dim))
    if base_latent_dim <= 0:
        raise ValueError("model.latent_dim or model.latent_dim_base must be a positive integer.")

    scaled_latent_dim = math.ceil(base_latent_dim * int(channels) / reference_channels)
    if round_to > 1:
        scaled_latent_dim = int(math.ceil(scaled_latent_dim / round_to) * round_to)

    model_cfg["latent_dim_base"] = base_latent_dim
    model_cfg["latent_dim"] = scaled_latent_dim
    return config


def _resolve_value(value: Any, root: Path) -> Any:
    if isinstance(value, dict):
        return {key: _resolve_value(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_value(item, root) for item in value]
    if isinstance(value, str):
        if value.startswith("./"):
            return str((root / value[2:]).resolve())
        return value
    return value


def load_config(path: str | Path, base_root: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML mapping.")
    config = _synchronize_channel_config(config)
    config = _synchronize_latent_config(config)
    resolution_root = Path(base_root).resolve() if base_root is not None else Path.cwd().resolve()
    resolved = _resolve_value(copy.deepcopy(config), resolution_root)
    resolved["config_path"] = str(config_path)
    resolved["project_root"] = str(resolution_root)
    return resolved
