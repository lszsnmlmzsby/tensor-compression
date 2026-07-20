from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed.nn.functional import all_gather as differentiable_all_gather
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tensor_compression.config import load_config  # noqa: E402
from tensor_compression.downstream.pdebench import (  # noqa: E402
    resolve_checkpoint_field_keys,
    resolve_device,
    resize_chw_batch,
    validate_checkpoint_field_keys_against_model,
)
from tensor_compression.models import build_model  # noqa: E402
from tensor_compression.integrations import WandbLogger  # noqa: E402
from tensor_compression.utils.pipeline_config import (  # noqa: E402
    first_nested,
    load_yaml_mapping,
    require_args,
    resolve_path_string,
    set_default,
    value_to_csv,
)

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "scripts/train_tensor_patch_text_alignment.py requires transformers. "
        "Install it with: pip install transformers accelerate safetensors"
    ) from exc


@dataclass(frozen=True)
class PatchRecord:
    sample_index: int
    time_index: int
    row: int
    col: int
    field_key: str | None = None


@dataclass(frozen=True)
class HiddenBatch:
    hidden: torch.Tensor
    metrics: dict[str, float]


@dataclass(frozen=True)
class SharedSuffixTokenBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    metrics: dict[str, float]
    suffix_token_ids: tuple[int, ...]


@dataclass(frozen=True)
class AlignmentAnchor:
    name: str
    mode: str
    token_ids: tuple[int, ...]
    text: str | None = None
    probe_family: str | None = None
    probe_template_index: int | None = None
    probe_parameters: tuple[int, ...] = ()


PROBE_FAMILIES = (
    "point_value",
    "point_difference",
    "point_mean",
    "region_mean",
    "region_range",
)
PROBE_TEMPLATE_COUNTS = {
    "point_value": 8,
    "point_difference": 4,
    "point_mean": 4,
    "region_mean": 4,
    "region_range": 4,
}
PROBE_READOUT_ENDINGS = (" is", " equals", " gives", " contains")
PROBE_FORBIDDEN_INPUT_MARKERS = ("answer:", "a or b", "a/b", "choose from", "options:", "?")
REMOVED_PATCH_ALIGNMENT_OPTIONS = (
    "center_embeddings",
    "cosine_loss_weight",
    "probe_answer_ce_weight",
    "probe_teacher_kl_weight",
    "probe_kl_temperature",
    "probe_teacher_preflight_records",
    # The old parameter turned a noisy semantic diagnostic into a hard training gate.
    "teacher_probe_min_correlation",
)


def parse_csv(raw: str | Sequence[str] | None) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, Sequence) and not isinstance(raw, str):
        return [str(part).strip() for part in raw if str(part).strip()]
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def reject_removed_alignment_options(config: Mapping[str, Any]) -> None:
    configured = [
        name
        for name in REMOVED_PATCH_ALIGNMENT_OPTIONS
        if first_nested(config, [f"patch_alignment.{name}"]) is not None
    ]
    if configured:
        raise ValueError(
            "Removed patch_alignment options are still present in the config: "
            f"{configured}. Delete them before running the alignment objective."
        )


def validate_teacher_tensor_source(
    normalization_cfg: Mapping[str, Any],
    teacher_text_source: str,
) -> None:
    mode = str((normalization_cfg or {}).get("mode", "none")).lower()
    clip_min = (normalization_cfg or {}).get("clip_min")
    clip_max = (normalization_cfg or {}).get("clip_max")
    source = str(teacher_text_source).lower()
    if source not in {"raw", "normalized"}:
        raise ValueError("patch_alignment.teacher_text_source must be raw or normalized.")
    if (mode != "none" or clip_min is not None or clip_max is not None) and source != "normalized":
        raise ValueError(
            "A normalized or clipped patch encoder requires patch_alignment.teacher_text_source=normalized "
            "so the tensor and text branches receive the same values."
        )


def parse_index_spec(raw: str | Sequence[int] | None, max_count: int) -> list[int]:
    if raw is None or str(raw).strip().lower() == "all":
        return list(range(int(max_count)))
    if isinstance(raw, str):
        return [int(part.strip()) for part in raw.split(",") if part.strip()]
    return [int(item) for item in raw]


def dump_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def distributed_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def distributed_world_size() -> int:
    return int(dist.get_world_size()) if distributed_is_initialized() else 1


def distributed_rank() -> int:
    return int(dist.get_rank()) if distributed_is_initialized() else 0


def is_main_process() -> bool:
    return distributed_rank() == 0


def setup_distributed_from_env(args: argparse.Namespace) -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        args.distributed = False
        args.rank = 0
        args.local_rank = 0
        args.world_size = 1
        return
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available, but WORLD_SIZE > 1.")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        args.device = f"cuda:{local_rank}"
    dist.init_process_group(
        backend=backend,
        timeout=timedelta(seconds=float(args.distributed_timeout_seconds)),
    )
    args.distributed = True
    args.rank = distributed_rank()
    args.local_rank = local_rank
    args.world_size = distributed_world_size()


def cleanup_distributed() -> None:
    if distributed_is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def distributed_barrier(stage: str | None = None) -> None:
    if distributed_is_initialized():
        if stage is not None:
            print(
                f"ddp_wait rank={distributed_rank()} stage={stage}",
                flush=True,
            )
        dist.barrier()
        if stage is not None and is_main_process():
            print(f"ddp_synced stage={stage}", flush=True)


def broadcast_object_from_main(value: Any) -> Any:
    if not distributed_is_initialized():
        return value
    payload = [value if is_main_process() else None]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


def broadcast_module_state(module: nn.Module | None) -> None:
    if module is None or not distributed_is_initialized():
        return
    with torch.no_grad():
        for parameter in module.parameters():
            dist.broadcast(parameter.data, src=0)
        for buffer in module.buffers():
            dist.broadcast(buffer.data, src=0)


def stable_name_fingerprint(names: Sequence[str]) -> int:
    digest = hashlib.sha256("\n".join(names).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**62)


def distributed_collective_device() -> torch.device:
    if torch.cuda.is_available() and (not distributed_is_initialized() or dist.get_backend() == "nccl"):
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def gradient_parameter_entries(
    modules: Sequence[nn.Module | None],
) -> list[tuple[str, nn.Parameter]]:
    entries: list[tuple[str, nn.Parameter]] = []
    seen: set[int] = set()
    for module_index, module in enumerate(modules):
        if module is None:
            continue
        for name, parameter in module.named_parameters():
            if not parameter.requires_grad or id(parameter) in seen:
                continue
            seen.add(id(parameter))
            entries.append((f"module_{module_index}.{name}", parameter))
    return entries


def verify_distributed_signature(values: torch.Tensor, description: str) -> None:
    if not distributed_is_initialized():
        return
    gathered = [torch.empty_like(values) for _ in range(distributed_world_size())]
    dist.all_gather(gathered, values)
    signatures = [tuple(int(item) for item in tensor.cpu().tolist()) for tensor in gathered]
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise RuntimeError(f"Distributed {description} differs across ranks: {signatures}.")


def synchronize_gradients(modules: Sequence[nn.Module | None]) -> None:
    if not distributed_is_initialized():
        return
    entries = gradient_parameter_entries(modules)
    if not entries:
        return
    missing = [name for name, parameter in entries if parameter.grad is None]
    signature_names = [
        f"{name}:{tuple(parameter.shape)}:{parameter.dtype}:{parameter.device.type}"
        for name, parameter in entries
    ]
    signature = torch.tensor(
        [
            len(entries),
            sum(parameter.numel() for _name, parameter in entries),
            len(missing),
            stable_name_fingerprint(signature_names),
        ],
        dtype=torch.int64,
        device=distributed_collective_device(),
    )
    verify_distributed_signature(signature, "gradient schema")
    if missing:
        preview = ", ".join(missing[:8])
        raise RuntimeError(
            "Trainable parameters are missing gradients before distributed synchronization: "
            f"{preview}{' ...' if len(missing) > 8 else ''}."
        )

    buckets: dict[tuple[torch.device, torch.dtype], list[nn.Parameter]] = {}
    for _name, parameter in entries:
        gradient = parameter.grad
        if gradient is None:  # pragma: no cover - guarded above
            continue
        if gradient.is_sparse:
            raise TypeError("Sparse gradients are not supported by flat distributed synchronization.")
        buckets.setdefault((gradient.device, gradient.dtype), []).append(parameter)

    world_size = float(distributed_world_size())
    for parameters in buckets.values():
        flat = torch.cat([parameter.grad.reshape(-1) for parameter in parameters])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(world_size)
        offset = 0
        with torch.no_grad():
            for parameter in parameters:
                count = parameter.numel()
                parameter.grad.copy_(flat[offset : offset + count].view_as(parameter))
                offset += count


def average_metrics_across_processes(metrics: Mapping[str, float]) -> dict[str, float]:
    if not distributed_is_initialized():
        return dict(metrics)
    keys = sorted(
        key
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    )
    if not keys:
        return {}
    device = distributed_collective_device()
    signature = torch.tensor(
        [len(keys), stable_name_fingerprint(keys)],
        dtype=torch.int64,
        device=device,
    )
    verify_distributed_signature(signature, "metric schema")
    values = torch.tensor([float(metrics[key]) for key in keys], dtype=torch.float64, device=device)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values.div_(distributed_world_size())
    return {key: float(value) for key, value in zip(keys, values.cpu().tolist(), strict=True)}


def weighted_average_metrics_across_processes(
    metrics: Mapping[str, float],
    local_weight: int,
) -> dict[str, float]:
    if not distributed_is_initialized():
        return dict(metrics)
    keys = sorted(
        key
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    )
    if not keys:
        return {}
    device = distributed_collective_device()
    signature = torch.tensor(
        [len(keys), stable_name_fingerprint(keys)],
        dtype=torch.int64,
        device=device,
    )
    verify_distributed_signature(signature, "weighted metric schema")
    weight = max(0, int(local_weight))
    values = torch.tensor(
        [float(metrics[key]) * weight for key in keys] + [float(weight)],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    total_weight = float(values[-1].item())
    if total_weight <= 0.0:
        raise RuntimeError("Distributed metric aggregation received zero total records.")
    return {
        key: float(value / total_weight)
        for key, value in zip(keys, values[:-1].cpu().tolist(), strict=True)
    }


def gather_variable_rows_without_grad(tensor: torch.Tensor) -> torch.Tensor:
    """Gather uneven row shards on every rank while preserving paired row order."""
    if not distributed_is_initialized():
        return tensor.detach()
    device = distributed_collective_device()
    local = tensor.detach().contiguous().to(device)
    row_count = torch.tensor([local.shape[0]], dtype=torch.int64, device=device)
    gathered_counts = [torch.empty_like(row_count) for _ in range(distributed_world_size())]
    dist.all_gather(gathered_counts, row_count)
    counts = [int(item.item()) for item in gathered_counts]
    max_rows = max(counts)
    padded_shape = (max_rows, *local.shape[1:])
    padded = torch.zeros(padded_shape, dtype=local.dtype, device=device)
    padded[: local.shape[0]].copy_(local)
    gathered = [torch.empty_like(padded) for _ in range(distributed_world_size())]
    dist.all_gather(gathered, padded)
    return torch.cat(
        [shard[:count].cpu() for shard, count in zip(gathered, counts, strict=True)],
        dim=0,
    )


def gather_with_grad(tensor: torch.Tensor) -> torch.Tensor:
    if not distributed_is_initialized():
        return tensor
    return torch.cat(tuple(differentiable_all_gather(tensor.contiguous())), dim=0)


def gather_without_grad(tensor: torch.Tensor) -> torch.Tensor:
    if not distributed_is_initialized():
        return tensor.detach()
    gathered = [torch.empty_like(tensor) for _ in range(distributed_world_size())]
    dist.all_gather(gathered, tensor.detach().contiguous())
    return torch.cat(gathered, dim=0)


def redacted_args(args: argparse.Namespace) -> dict[str, Any]:
    payload = dict(vars(args))
    if payload.get("wandb_api_key"):
        payload["wandb_api_key"] = "***REDACTED***"
    return payload


def load_checkpoint_and_config(
    checkpoint_path: str | Path,
    config_path: str | Path | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(Path(checkpoint_path).expanduser(), map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
    if config_path is not None:
        config = load_config(config_path, base_root=PROJECT_ROOT)
    else:
        raw_config = checkpoint.get("compressor_config") or checkpoint.get("config")
        if not isinstance(raw_config, Mapping):
            raise ValueError(
                "Compressor checkpoint does not contain compressor_config/config. "
                "Pass --compressor-config explicitly."
            )
        config = dict(raw_config)
    return dict(checkpoint), config


def resolve_checkpoint_state_dict(checkpoint: Mapping[str, Any], checkpoint_path: str | Path) -> Mapping[str, Any]:
    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        state_dict = checkpoint.get("compressor_state_dict")
    if state_dict is None:
        raise ValueError(
            "Checkpoint does not contain model_state_dict or compressor_state_dict: "
            f"{checkpoint_path}"
        )
    if not isinstance(state_dict, Mapping):
        raise ValueError(f"Unsupported state_dict format in checkpoint: {checkpoint_path}")
    return state_dict


def resolve_field_keys(
    cli_fields: str | Sequence[str] | None,
    config: Mapping[str, Any] | None,
) -> list[str]:
    parsed_fields = parse_csv(cli_fields)
    if config is None:
        if not parsed_fields:
            raise ValueError("Could not resolve field keys. Pass --fields explicitly.")
        return [str(field) for field in parsed_fields]
    checkpoint_fields = resolve_checkpoint_field_keys(config)
    if parsed_fields and checkpoint_fields and parsed_fields != checkpoint_fields:
        raise ValueError(
            "Provided fields differ from the compressor config field order. "
            f"CLI={parsed_fields}, checkpoint={checkpoint_fields}."
        )
    fields = parsed_fields or checkpoint_fields
    if not fields:
        raise ValueError("Could not resolve field keys. Pass --fields explicitly.")
    validate_checkpoint_field_keys_against_model(config, fields)
    return [str(field) for field in fields]


def build_patch_encoder_config(
    *,
    patch_encoder_cfg: Mapping[str, Any] | None,
    field_keys: Sequence[str],
    patch_size: int,
) -> dict[str, Any]:
    cfg = dict(patch_encoder_cfg or {})
    model_cfg = dict(cfg.get("model") or {})
    patch_size = int(patch_size)
    multipliers = [int(value) for value in model_cfg.get("channel_multipliers", [1, 2])]
    down_factor = 2 ** len(multipliers)
    if patch_size % down_factor != 0:
        raise ValueError(
            "patch_size must be divisible by the patch encoder downsampling factor. "
            f"Got patch_size={patch_size}, down_factor={down_factor} from channel_multipliers={multipliers}."
        )
    model_defaults = {
        "name": "conv_token_autoencoder_2d",
        "input_size": [patch_size, patch_size],
        "base_channels": 32,
        "channel_multipliers": multipliers,
        "num_res_blocks": 1,
        "latent_dim": 128,
        "latent_grid": [patch_size // down_factor, patch_size // down_factor],
        "dropout": 0.0,
        "norm": "group",
        "activation": "gelu",
        "output_activation": "identity",
    }
    for key, value in model_defaults.items():
        model_cfg.setdefault(key, value)
    model_cfg["in_channels"] = int(model_cfg.get("in_channels", len(field_keys)))
    model_cfg["out_channels"] = int(model_cfg.get("out_channels", len(field_keys)))
    if model_cfg["in_channels"] != len(field_keys) or model_cfg["out_channels"] != len(field_keys):
        raise ValueError(
            "patch_encoder.model in_channels/out_channels must match the selected fields. "
            f"Got in_channels={model_cfg['in_channels']}, out_channels={model_cfg['out_channels']}, "
            f"fields={list(field_keys)}."
        )
    normalization_cfg = dict(
        cfg.get(
            "normalization",
            {
                "mode": "zscore",
                "scope": "channel",
                "stats_path": None,
                "clip_min": None,
                "clip_max": None,
            },
        )
        or {}
    )
    return {
        "model": model_cfg,
        "data": {
            "dataset": {
                "hdf5_dataset_key": str(field_keys[0]) if len(field_keys) == 1 else None,
                "hdf5_dataset_keys": list(field_keys) if len(field_keys) > 1 else [],
                "normalization": normalization_cfg,
            }
        },
    }


def checkpoint_channel_count(config: Mapping[str, Any]) -> int | None:
    model_cfg = config.get("model", {})
    if not isinstance(model_cfg, Mapping):
        return None
    for key in ("in_channels", "out_channels"):
        value = model_cfg.get(key)
        if value is not None:
            return int(value)
    dataset_cfg = config.get("data", {}).get("dataset", {}) if isinstance(config.get("data"), Mapping) else {}
    if isinstance(dataset_cfg, Mapping):
        keys = dataset_cfg.get("hdf5_dataset_keys")
        if isinstance(keys, Sequence) and not isinstance(keys, str) and keys:
            return len(keys)
        if dataset_cfg.get("hdf5_dataset_key"):
            return 1
    return None


def hdf5_axis_sizes(hdf5_path: str | Path, field: str) -> tuple[int, int, int, int]:
    with h5py.File(Path(hdf5_path).expanduser(), "r") as handle:
        if field not in handle or not isinstance(handle[field], h5py.Dataset):
            raise KeyError(f"HDF5 dataset key {field!r} not found in {hdf5_path}.")
        shape = tuple(int(dim) for dim in handle[field].shape)
    if len(shape) != 4:
        raise ValueError(f"Expected PDEBench 2D field [sample,time,height,width], got {shape}.")
    return shape


def validate_field_shapes(hdf5_path: str | Path, field_keys: Sequence[str]) -> tuple[int, int, int, int]:
    if not field_keys:
        raise ValueError("At least one field key is required.")
    reference_field = str(field_keys[0])
    reference_shape = hdf5_axis_sizes(hdf5_path, reference_field)
    for field in field_keys[1:]:
        shape = hdf5_axis_sizes(hdf5_path, str(field))
        if shape != reference_shape:
            raise ValueError(
                "All fields must have the same [sample,time,height,width] shape for patch alignment. "
                f"{reference_field} has {reference_shape}, but {field} has {shape}."
            )
    return reference_shape


def split_axis_indices(
    indices: Sequence[int],
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    axis_name: str,
) -> dict[str, list[int]]:
    values = sorted({int(index) for index in indices})
    if len(values) < 3:
        raise ValueError(
            f"split_mode requires at least 3 distinct {axis_name} indices, got {len(values)}. "
            "Use split_mode=random_record only for non-isolated smoke tests."
        )
    ratio_sum = float(train_ratio) + float(val_ratio) + float(test_ratio)
    if ratio_sum <= 0.0:
        raise ValueError("split_train_ratio + split_val_ratio + split_test_ratio must be positive.")
    train_fraction = float(train_ratio) / ratio_sum
    val_fraction = float(val_ratio) / ratio_sum
    rng = random.Random(int(seed))
    shuffled = list(values)
    rng.shuffle(shuffled)
    train_count = max(1, int(len(shuffled) * train_fraction))
    val_count = max(1, int(len(shuffled) * val_fraction))
    if train_count + val_count >= len(shuffled):
        train_count = max(1, len(shuffled) - 2)
        val_count = 1
    test_count = len(shuffled) - train_count - val_count
    if test_count <= 0:
        raise ValueError(f"Could not create non-empty train/val/test split for {axis_name}.")
    return {
        "train": sorted(shuffled[:train_count]),
        "val": sorted(shuffled[train_count : train_count + val_count]),
        "test": sorted(shuffled[train_count + val_count :]),
    }


def build_axis_split_plan(
    *,
    hdf5_path: str | Path,
    field: str,
    sample_indices: str | Sequence[int],
    time_indices: str | Sequence[int],
    split_mode: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, Any]:
    sample_count, time_count, _height, _width = hdf5_axis_sizes(hdf5_path, field)
    samples = parse_index_spec(sample_indices, sample_count)
    times = parse_index_spec(time_indices, time_count)
    if not samples or not times:
        raise ValueError("sample_indices and time_indices must not be empty.")
    mode = str(split_mode).lower()
    sample_splits = {"train": list(samples), "val": list(samples), "test": list(samples)}
    time_splits = {"train": list(times), "val": list(times), "test": list(times)}
    if mode == "sample":
        sample_splits = split_axis_indices(
            samples,
            train_ratio=float(train_ratio),
            val_ratio=float(val_ratio),
            test_ratio=float(test_ratio),
            seed=int(seed),
            axis_name="sample",
        )
    elif mode == "time":
        time_splits = split_axis_indices(
            times,
            train_ratio=float(train_ratio),
            val_ratio=float(val_ratio),
            test_ratio=float(test_ratio),
            seed=int(seed),
            axis_name="time",
        )
    elif mode == "sample_time":
        sample_splits = split_axis_indices(
            samples,
            train_ratio=float(train_ratio),
            val_ratio=float(val_ratio),
            test_ratio=float(test_ratio),
            seed=int(seed),
            axis_name="sample",
        )
        time_splits = split_axis_indices(
            times,
            train_ratio=float(train_ratio),
            val_ratio=float(val_ratio),
            test_ratio=float(test_ratio),
            seed=int(seed) + 9973,
            axis_name="time",
        )
    elif mode != "random_record":
        raise ValueError(f"Unsupported patch_alignment.split_mode: {split_mode}")
    return {
        "mode": mode,
        "samples": sample_splits,
        "times": time_splits,
        "available_sample_count": len(set(samples)),
        "available_time_count": len(set(times)),
    }


def record_key(record: PatchRecord) -> tuple[str, int, int, int, int]:
    return (
        "" if record.field_key is None else str(record.field_key),
        int(record.sample_index),
        int(record.time_index),
        int(record.row),
        int(record.col),
    )


def summarize_records(records: Sequence[PatchRecord]) -> dict[str, Any]:
    sample_values = sorted({int(record.sample_index) for record in records})
    time_values = sorted({int(record.time_index) for record in records})
    field_values = sorted({str(record.field_key) for record in records if record.field_key is not None})
    return {
        "record_count": len(records),
        "unique_record_count": len({record_key(record) for record in records}),
        "field_count": len(field_values),
        "field_preview": field_values[:16],
        "sample_count": len(sample_values),
        "time_count": len(time_values),
        "sample_preview": sample_values[:16],
        "time_preview": time_values[:16],
    }


def split_overlap_summary(
    train_records: Sequence[PatchRecord],
    val_records: Sequence[PatchRecord],
    test_records: Sequence[PatchRecord],
) -> dict[str, int]:
    train_keys = {record_key(record) for record in train_records}
    val_keys = {record_key(record) for record in val_records}
    test_keys = {record_key(record) for record in test_records}
    return {
        "train_val_exact_record_overlap": len(train_keys & val_keys),
        "train_test_exact_record_overlap": len(train_keys & test_keys),
        "val_test_exact_record_overlap": len(val_keys & test_keys),
    }


def build_patch_records(
    *,
    hdf5_path: str | Path,
    field: str,
    record_fields: Sequence[str] | None = None,
    sample_indices: str | Sequence[int],
    time_indices: str | Sequence[int],
    patch_size: int,
    count: int,
    seed: int,
    unique_records: bool,
) -> list[PatchRecord]:
    sample_count, time_count, height, width = hdf5_axis_sizes(hdf5_path, field)
    samples = parse_index_spec(sample_indices, sample_count)
    times = parse_index_spec(time_indices, time_count)
    if not samples or not times:
        raise ValueError("sample_indices and time_indices must not be empty.")
    if int(patch_size) <= 0 or int(patch_size) > min(height, width):
        raise ValueError(f"Invalid patch_size={patch_size} for spatial shape {(height, width)}.")
    rng = random.Random(int(seed))
    sampled_fields = [str(item) for item in (record_fields or []) if str(item)]
    records: list[PatchRecord] = []
    seen: set[tuple[str, int, int, int, int]] = set()
    max_row = height - int(patch_size)
    max_col = width - int(patch_size)
    max_attempts = max(int(count) * 50, 1000)
    attempts = 0
    while len(records) < int(count):
        attempts += 1
        if attempts > max_attempts:
            raise ValueError(
                f"Could not draw {count} unique patch records after {max_attempts} attempts. "
                "Reduce record count, relax split constraints, or set unique_records=false for a smoke test."
            )
        record = PatchRecord(
            sample_index=int(rng.choice(samples)),
            time_index=int(rng.choice(times)),
            row=int(rng.randint(0, max_row)),
            col=int(rng.randint(0, max_col)),
            field_key=str(rng.choice(sampled_fields)) if sampled_fields else None,
        )
        key = record_key(record)
        if bool(unique_records) and key in seen:
            continue
        seen.add(key)
        records.append(record)
    return records


def serialize_patch_text(
    *,
    record: Mapping[str, Any] | PatchRecord,
    patch: torch.Tensor,
    field_keys: Sequence[str],
    decimal_places: int,
    prompt_template: str,
) -> str:
    decimals = max(0, int(decimal_places))
    patch_cpu = patch.detach().cpu()
    field_chunks: list[str] = []
    for channel, field in enumerate(field_keys):
        rows: list[str] = []
        values = patch_cpu[channel]
        for row in range(values.shape[0]):
            row_values = ", ".join(f"{float(value):.{decimals}f}" for value in values[row])
            rows.append(f"[{row_values}]")
        field_chunks.append(f"{field}=[{'; '.join(rows)}]")
    body = "\n".join(field_chunks)
    patch_size = int(patch_cpu.shape[-1])
    if prompt_template == "compact":
        return (
            "Represent this PDE tensor patch for numeric reasoning.\n"
            f"fields={','.join(str(field) for field in field_keys)} patch_size={patch_size}\n"
            f"{body}\nRepresentation:"
        )
    if prompt_template == "compact_with_metadata":
        if isinstance(record, PatchRecord):
            sample_index = int(record.sample_index)
            time_index = int(record.time_index)
            row = int(record.row)
            col = int(record.col)
        else:
            sample_index = int(record.get("sample_index", -1))
            time_index = int(record.get("time_index", -1))
            row = int(record.get("row", -1))
            col = int(record.get("col", -1))
        return (
            "Represent this PDE tensor patch for numeric reasoning.\n"
            f"sample={sample_index} time={time_index} "
            f"top_left=({row},{col}) patch_size={patch_size}\n"
            f"{body}\nRepresentation:"
        )
    if prompt_template == "plain":
        return body
    raise ValueError(f"Unsupported text_prompt_template: {prompt_template}")


def serialize_tensor_values(patch: torch.Tensor, decimal_places: int) -> str:
    """Serialize only tensor values and shape delimiters, without field or provenance text."""
    if patch.ndim != 3:
        raise ValueError(f"Expected a [channels,height,width] tensor, got {tuple(patch.shape)}.")
    if not bool(torch.isfinite(patch).all()):
        non_finite = int((~torch.isfinite(patch)).sum().item())
        raise ValueError(f"Tensor text source contains {non_finite} non-finite values.")
    decimals = max(0, int(decimal_places))
    channel_chunks: list[str] = []
    for channel in patch.detach().cpu():
        rows = [
            "[" + ", ".join(f"{float(value):.{decimals}f}" for value in row) + "]"
            for row in channel
        ]
        channel_chunks.append("[" + "; ".join(rows) + "]")
    return "[" + "; ".join(channel_chunks) + "]"


def serialize_tensor_value_batch(patches: torch.Tensor, decimal_places: int) -> list[str]:
    return [serialize_tensor_values(patch, decimal_places) for patch in patches]


def serialize_patch_batch(
    *,
    records: Sequence[Mapping[str, Any]],
    patches: torch.Tensor,
    decimal_places: int,
    prompt_template: str,
) -> list[str]:
    texts: list[str] = []
    for record, patch in zip(records, patches, strict=True):
        fields = record.get("fields", [])
        if isinstance(fields, Sequence) and not isinstance(fields, str):
            field_keys = [str(field) for field in fields]
        else:
            field_keys = [str(fields)]
        texts.append(
            serialize_patch_text(
                record=record,
                patch=patch,
                field_keys=field_keys,
                decimal_places=int(decimal_places),
                prompt_template=str(prompt_template),
            )
        )
    return texts


class PDEBenchPatchTextDataset(Dataset):
    def __init__(
        self,
        *,
        hdf5_path: str | Path,
        field_keys: Sequence[str],
        records: Sequence[PatchRecord],
        patch_size: int,
        decimal_places: int,
        prompt_template: str,
        include_raw_text: bool,
    ) -> None:
        self.hdf5_path = Path(hdf5_path).expanduser()
        self.field_keys = [str(field) for field in field_keys]
        self.records = list(records)
        self.patch_size = int(patch_size)
        self.decimal_places = int(decimal_places)
        self.prompt_template = str(prompt_template)
        self.include_raw_text = bool(include_raw_text)
        self._hdf5_handle: h5py.File | None = None
        self._hdf5_pid: int | None = None

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        # h5py handles cannot be pickled or safely shared by spawned DataLoader workers.
        state["_hdf5_handle"] = None
        state["_hdf5_pid"] = None
        return state

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        handle = getattr(self, "_hdf5_handle", None)
        if handle is not None:
            try:
                handle.close()
            except (OSError, RuntimeError, ValueError):
                pass
        self._hdf5_handle = None
        self._hdf5_pid = None

    def _open_hdf5(self) -> h5py.File:
        process_id = os.getpid()
        handle = self._hdf5_handle
        if handle is not None and self._hdf5_pid == process_id and bool(handle.id.valid):
            return handle
        # A forked worker may inherit the parent's Python object. Reopen lazily in the worker process.
        self.close()
        self._hdf5_handle = h5py.File(self.hdf5_path, "r")
        self._hdf5_pid = process_id
        return self._hdf5_handle

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[int(index)]
        selected_fields = self._record_field_keys(record)
        patch = self._read_patch(record)
        text = self._serialize_patch(record, patch) if self.include_raw_text else ""
        return {
            "record": {
                "sample_index": int(record.sample_index),
                "time_index": int(record.time_index),
                "row": int(record.row),
                "col": int(record.col),
                "patch_size": int(self.patch_size),
                "fields": list(selected_fields),
            },
            "patch": patch,
            "text": text,
        }

    def _record_field_keys(self, record: PatchRecord) -> list[str]:
        if record.field_key is not None:
            return [str(record.field_key)]
        return list(self.field_keys)

    def _read_patch(self, record: PatchRecord) -> torch.Tensor:
        arrays: list[np.ndarray] = []
        row_slice = slice(int(record.row), int(record.row) + self.patch_size)
        col_slice = slice(int(record.col), int(record.col) + self.patch_size)
        handle = self._open_hdf5()
        for field in self._record_field_keys(record):
            dataset = handle[field]
            arrays.append(
                np.asarray(
                    dataset[int(record.sample_index), int(record.time_index), row_slice, col_slice],
                    dtype=np.float32,
                )
            )
        stacked = np.stack(arrays, axis=0)
        return torch.as_tensor(stacked, dtype=torch.float32)

    def _serialize_patch(self, record: PatchRecord, patch: torch.Tensor) -> str:
        return serialize_patch_text(
            record=record,
            patch=patch,
            field_keys=self._record_field_keys(record),
            decimal_places=int(self.decimal_places),
            prompt_template=str(self.prompt_template),
        )


class DistributedEvalSampler(Sampler[int]):
    """Partition evaluation records exactly once across ranks, without padding duplicates."""

    def __init__(self, dataset: Dataset, num_replicas: int, rank: int) -> None:
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        if self.num_replicas <= 0 or not 0 <= self.rank < self.num_replicas:
            raise ValueError(
                f"Invalid distributed sampler rank={self.rank}, replicas={self.num_replicas}."
            )

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self) -> int:
        if self.rank >= len(self.dataset):
            return 0
        return (len(self.dataset) - 1 - self.rank) // self.num_replicas + 1


def collate_patch_text(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "records": [item["record"] for item in items],
        "patch": torch.stack([item["patch"] for item in items], dim=0),
        "texts": [str(item["text"]) for item in items],
    }


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=int(heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.cross_query_norm = nn.LayerNorm(dim)
        self.cross_context_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=int(heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(dim * 4, dim),
            nn.Dropout(float(dropout)),
        )
        self.capture_attention = False
        self.last_self_attention_weights: torch.Tensor | None = None
        self.last_attention_weights: torch.Tensor | None = None

    def forward(self, queries: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        self_attended, self_weights = self.self_attn(
            self.self_norm(queries),
            self.self_norm(queries),
            self.self_norm(queries),
            need_weights=bool(self.capture_attention),
            average_attn_weights=False,
        )
        self.last_self_attention_weights = (
            self_weights.detach().float().cpu()
            if self.capture_attention and self_weights is not None
            else None
        )
        queries = queries + self_attended
        cross_attended, cross_weights = self.cross_attn(
            self.cross_query_norm(queries),
            self.cross_context_norm(context),
            self.cross_context_norm(context),
            need_weights=bool(self.capture_attention),
            average_attn_weights=False,
        )
        self.last_attention_weights = (
            cross_weights.detach().float().cpu()
            if self.capture_attention and cross_weights is not None
            else None
        )
        queries = queries + cross_attended
        return queries + self.ffn(self.ffn_norm(queries))


def sinusoidal_position_encoding(length: int, dim: int) -> torch.Tensor:
    """Return a deterministic [length, dim] sinusoidal position table."""
    if int(length) <= 0 or int(dim) <= 0:
        raise ValueError(f"Position encoding length and dim must be positive, got {length} and {dim}.")
    positions = torch.arange(int(length), dtype=torch.float32).unsqueeze(1)
    frequency_count = (int(dim) + 1) // 2
    frequencies = torch.exp(
        torch.arange(frequency_count, dtype=torch.float32)
        * (-math.log(10000.0) / max(frequency_count - 1, 1))
    )
    angles = positions * frequencies.unsqueeze(0)
    encoding = torch.empty(int(length), frequency_count * 2, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(angles)
    encoding[:, 1::2] = torch.cos(angles)
    return encoding[:, : int(dim)]


def sinusoidal_2d_position_encoding(height: int, width: int, dim: int) -> torch.Tensor:
    """Encode row and column independently and flatten positions in row-major order."""
    if int(height) <= 0 or int(width) <= 0:
        raise ValueError(f"Spatial dimensions must be positive, got {(height, width)}.")
    row_dim = int(dim) // 2
    col_dim = int(dim) - row_dim
    if row_dim <= 0 or col_dim <= 0:
        raise ValueError(f"2D position encoding requires dim >= 2, got {dim}.")
    rows = sinusoidal_position_encoding(int(height), row_dim)
    cols = sinusoidal_position_encoding(int(width), col_dim)
    row_grid = rows[:, None, :].expand(int(height), int(width), row_dim)
    col_grid = cols[None, :, :].expand(int(height), int(width), col_dim)
    return torch.cat([row_grid, col_grid], dim=-1).reshape(1, int(height) * int(width), int(dim))


class SpatialTransformerBlock(nn.Module):
    """Contextualize spatial tokens without replacing their position-specific states."""

    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=int(heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(dim * 4, dim),
            nn.Dropout(float(dropout)),
        )
        self.capture_attention = False
        self.last_self_attention_weights: torch.Tensor | None = None
        self.last_attention_weights: torch.Tensor | None = None

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        normalized = self.self_norm(tokens)
        attended, weights = self.self_attn(
            normalized,
            normalized,
            normalized,
            need_weights=bool(self.capture_attention),
            average_attn_weights=False,
        )
        self.last_self_attention_weights = (
            weights.detach().float().cpu()
            if self.capture_attention and weights is not None
            else None
        )
        self.last_attention_weights = self.last_self_attention_weights
        tokens = tokens + attended
        return tokens + self.ffn(self.ffn_norm(tokens))


class TensorPatchAlignmentAdapter(nn.Module):
    def __init__(
        self,
        latent_channels: int,
        latent_grid: Sequence[int],
        adapter_dim: int,
        projection_dim: int,
        dropout: float,
        adapter_type: str,
        query_tokens: int,
        adapter_layers: int,
        adapter_heads: int,
        soft_prompt_scale: float,
    ) -> None:
        super().__init__()
        self.adapter_type = str(adapter_type).lower()
        self.latent_grid = tuple(int(dim) for dim in latent_grid)
        if len(self.latent_grid) != 2 or any(dim <= 0 for dim in self.latent_grid):
            raise ValueError(f"latent_grid must contain two positive dimensions, got {self.latent_grid}.")
        self.latent_token_count = int(self.latent_grid[0] * self.latent_grid[1])
        self.soft_prompt_tokens = (
            self.latent_token_count
            if self.adapter_type == "spatial_transformer"
            else int(query_tokens)
            if self.adapter_type == "qformer"
            else 1
        )
        self.structured_query_conditioning = False
        self.soft_prompt_scale = float(soft_prompt_scale)
        self.adapter_dim = int(adapter_dim)
        adapter_dim = self.adapter_dim
        projection_dim = int(projection_dim)
        if adapter_dim % int(adapter_heads) != 0:
            raise ValueError(
                f"adapter_dim must be divisible by adapter_heads. Got {adapter_dim} and {adapter_heads}."
            )
        if self.adapter_type == "pooled_mlp":
            input_dim = int(latent_channels) * 2
            self.projection = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, adapter_dim),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(adapter_dim, projection_dim),
            )
            return
        if self.adapter_type == "spatial_transformer":
            if int(query_tokens) != self.latent_token_count:
                raise ValueError(
                    "spatial_transformer requires one output token per latent-grid position: "
                    f"query_tokens={int(query_tokens)}, latent_grid={self.latent_grid}, "
                    f"expected={self.latent_token_count}."
                )
            self.latent_projection = nn.Linear(int(latent_channels), adapter_dim)
            self.local_residual_projection = nn.Linear(int(latent_channels), adapter_dim)
            self.register_buffer(
                "spatial_pos_encoding",
                sinusoidal_2d_position_encoding(*self.latent_grid, adapter_dim),
                persistent=True,
            )
            # These paths are architectural guarantees, not gates for the retrieval objective to disable.
            # Persistent buffers preserve strict compatibility with older checkpoints that stored trainable scales.
            self.register_buffer("spatial_pos_scale", torch.tensor(1.0), persistent=True)
            self.register_buffer("local_residual_scale", torch.tensor(1.0), persistent=True)
            self.capture_spatial_path_metrics = False
            self.last_spatial_path_metrics: dict[str, float] = {}
            self.blocks = nn.ModuleList(
                [
                    SpatialTransformerBlock(adapter_dim, int(adapter_heads), float(dropout))
                    for _ in range(int(adapter_layers))
                ]
            )
            self.output = nn.Sequential(
                nn.LayerNorm(adapter_dim),
                nn.Linear(adapter_dim, projection_dim),
            )
            return
        if self.adapter_type != "qformer":
            raise ValueError(f"Unsupported patch alignment adapter_type: {adapter_type}")
        if int(query_tokens) <= 0:
            raise ValueError("query_tokens must be positive for qformer adapter.")
        self.latent_projection = nn.Linear(int(latent_channels), adapter_dim)
        self.latent_pos_embed = nn.Parameter(torch.zeros(1, self.latent_token_count, adapter_dim))
        self.query_tokens = nn.Parameter(torch.empty(1, int(query_tokens), adapter_dim))
        self.blocks = nn.ModuleList(
            [
                CrossAttentionBlock(adapter_dim, int(adapter_heads), float(dropout))
                for _ in range(int(adapter_layers))
            ]
        )
        self.output = nn.Sequential(
            nn.LayerNorm(adapter_dim),
            nn.Linear(adapter_dim, projection_dim),
        )
        nn.init.normal_(self.query_tokens, mean=0.0, std=0.02)
        nn.init.normal_(self.latent_pos_embed, mean=0.0, std=0.02)

    def _load_from_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        prefix: str,
        local_metadata: Mapping[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        if self.adapter_type == "spatial_transformer":
            # Discard legacy learned gate values so Stage 1 and Stage 2 always retain both paths.
            self.spatial_pos_scale.fill_(1.0)
            self.local_residual_scale.fill_(1.0)

    def flatten_latent_tokens(self, latent_map: torch.Tensor) -> torch.Tensor:
        if latent_map.ndim != 4:
            raise ValueError(f"Expected latent_map [B,C,H,W], got {tuple(latent_map.shape)}.")
        if tuple(int(dim) for dim in latent_map.shape[-2:]) != self.latent_grid:
            raise ValueError(
                f"Expected latent grid {self.latent_grid}, got {tuple(int(dim) for dim in latent_map.shape[-2:])}."
            )
        return latent_map.flatten(2).transpose(1, 2).contiguous()

    def spatial_input_states(self, latent_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.adapter_type != "spatial_transformer":
            raise ValueError("spatial_input_states is available only for spatial_transformer adapters.")
        latent_tokens = self.flatten_latent_tokens(latent_map).to(dtype=self.latent_projection.weight.dtype)
        local_residual = self.local_residual_projection(latent_tokens)
        content_states = self.latent_projection(latent_tokens)
        position_states = self.spatial_pos_scale.to(dtype=content_states.dtype) * self.spatial_pos_encoding.to(
            device=content_states.device,
            dtype=content_states.dtype,
        )
        states = content_states + position_states
        if self.capture_spatial_path_metrics:
            self.last_spatial_path_metrics = {
                "spatial_position_to_content_rms_ratio": float(
                    position_states.detach().float().square().mean().sqrt().div(
                        content_states.detach().float().square().mean().sqrt().clamp_min(1.0e-8)
                    ).cpu().item()
                )
            }
        else:
            self.last_spatial_path_metrics = {}
        return states, local_residual

    def spatial_output_states(
        self,
        contextual_states: torch.Tensor,
        local_residual: torch.Tensor,
    ) -> torch.Tensor:
        if self.adapter_type != "spatial_transformer":
            raise ValueError("spatial_output_states is available only for spatial_transformer adapters.")
        local_path = self.local_residual_scale.to(dtype=contextual_states.dtype) * local_residual
        states = contextual_states + local_path
        if self.capture_spatial_path_metrics:
            self.last_spatial_path_metrics["local_residual_to_context_rms_ratio"] = float(
                local_path.detach().float().square().mean().sqrt().div(
                    contextual_states.detach().float().square().mean().sqrt().clamp_min(1.0e-8)
                ).cpu().item()
            )
        return self.scale_soft_prompts(self.output(states))

    def forward_soft_prompts(self, latent_map: torch.Tensor) -> torch.Tensor:
        if self.adapter_type == "spatial_transformer":
            states, local_residual = self.spatial_input_states(latent_map)
            for block in self.blocks:
                states = block(states)
            return self.spatial_output_states(states, local_residual)
        if self.adapter_type == "qformer":
            latent_tokens = self.flatten_latent_tokens(latent_map)
            context = self.latent_projection(latent_tokens) + self.latent_pos_embed
            queries = self.query_tokens.expand(latent_map.shape[0], -1, -1)
            for block in self.blocks:
                queries = block(queries, context)
            return self.scale_soft_prompts(self.output(queries))
        pooled = torch.cat(
            [
                latent_map.mean(dim=tuple(range(2, latent_map.ndim))),
                latent_map.std(dim=tuple(range(2, latent_map.ndim)), unbiased=False),
            ],
            dim=-1,
        )
        return self.scale_soft_prompts(self.projection(pooled).unsqueeze(1))

    def scale_soft_prompts(self, soft_prompts: torch.Tensor) -> torch.Tensor:
        if self.soft_prompt_scale <= 0.0:
            return soft_prompts
        return torch.tanh(soft_prompts) * self.soft_prompt_scale

    def forward_tensor(self, latent_map: torch.Tensor) -> torch.Tensor:
        soft_prompts = self.forward_soft_prompts(latent_map)
        return F.normalize(soft_prompts.mean(dim=1), dim=-1)

    def forward(
        self,
        latent_map: torch.Tensor,
        question_embeds: torch.Tensor | None = None,
        question_mask: torch.Tensor | None = None,
        structured_query: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del question_embeds, question_mask, structured_query
        return self.forward_soft_prompts(latent_map)


def alignment_adapter_parameter_metrics(adapter: TensorPatchAlignmentAdapter) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name in ("spatial_pos_scale", "local_residual_scale"):
        parameter = getattr(adapter, name, None)
        if isinstance(parameter, torch.Tensor) and parameter.numel() == 1:
            metrics[name] = float(parameter.detach().float().cpu().item())
    return metrics


def alignment_adapter_path_metrics(adapter: TensorPatchAlignmentAdapter) -> dict[str, float]:
    metrics = getattr(adapter, "last_spatial_path_metrics", None)
    return dict(metrics) if isinstance(metrics, Mapping) else {}


def build_projection_head(
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    layers: int,
    dropout: float,
) -> nn.Module:
    input_dim = int(input_dim)
    output_dim = int(output_dim)
    hidden_dim = int(hidden_dim)
    layers = int(layers)
    if layers <= 0:
        raise ValueError("alignment_projection.layers must be positive.")
    if layers == 1:
        return nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, output_dim))

    modules: list[nn.Module] = [
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
    ]
    if float(dropout) > 0.0:
        modules.append(nn.Dropout(float(dropout)))
    for _ in range(layers - 2):
        modules.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        if float(dropout) > 0.0:
            modules.append(nn.Dropout(float(dropout)))
    modules.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*modules)


class AlignmentProjectionPair(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        layers: int,
        dropout: float,
        shared: bool,
    ) -> None:
        super().__init__()
        self.shared = bool(shared)
        self.student = build_projection_head(input_dim, output_dim, hidden_dim, layers, dropout)
        self.teacher = (
            self.student
            if self.shared
            else build_projection_head(input_dim, output_dim, hidden_dim, layers, dropout)
        )

    def forward(
        self,
        student_hidden: torch.Tensor,
        teacher_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.student(student_hidden.float()), self.teacher(teacher_hidden.float())


class FixedTeacherWhitening(nn.Module):
    """A frozen PCA-whitening transform fitted on train-split teacher hidden states."""

    def __init__(
        self,
        hidden_dim: int,
        shrinkage: float,
        epsilon: float,
        output_dim: int | None = None,
        max_condition_number: float = 1000.0,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        output_dim = hidden_dim if output_dim is None else int(output_dim)
        if hidden_dim <= 0:
            raise ValueError("Whitening hidden_dim must be positive.")
        if output_dim <= 0 or output_dim > hidden_dim:
            raise ValueError(
                f"Whitening output_dim must be in [1, {hidden_dim}], got {output_dim}."
            )
        if not 0.0 <= float(shrinkage) <= 1.0:
            raise ValueError("alignment_transform.whitening.shrinkage must be in [0, 1].")
        if float(epsilon) <= 0.0:
            raise ValueError("alignment_transform.whitening.epsilon must be positive.")
        if float(max_condition_number) < 1.0:
            raise ValueError("alignment_transform.whitening.max_condition_number must be at least 1.")
        self.shrinkage = float(shrinkage)
        self.epsilon = float(epsilon)
        self.max_condition_number = float(max_condition_number)
        self.register_buffer("mean", torch.zeros(hidden_dim, dtype=torch.float32))
        self.register_buffer("matrix", torch.zeros(hidden_dim, output_dim, dtype=torch.float32))
        self.register_buffer("fitted_records", torch.zeros((), dtype=torch.long))
        self.fit_metrics: dict[str, float] = {}

    @property
    def is_fitted(self) -> bool:
        return int(self.fitted_records.item()) >= 2

    @torch.no_grad()
    def fit(
        self,
        teacher_hidden: torch.Tensor,
        covariance_residuals: torch.Tensor | None = None,
    ) -> dict[str, float]:
        samples = teacher_hidden.detach().float()
        if samples.ndim != 2 or int(samples.shape[1]) != int(self.mean.numel()):
            raise ValueError(
                "Teacher hidden states for whitening must have shape [records, hidden_dim]; "
                f"got {tuple(samples.shape)} for hidden_dim={int(self.mean.numel())}."
            )
        record_count = int(samples.shape[0])
        if record_count < 2:
            raise ValueError("Whitening requires at least two teacher records.")
        mean = samples.mean(dim=0)
        centered = (
            samples - mean
            if covariance_residuals is None
            else covariance_residuals.detach().float()
        )
        if centered.shape != samples.shape:
            raise ValueError(
                "Whitening covariance residuals must match teacher hidden shape; "
                f"got {tuple(centered.shape)} and {tuple(samples.shape)}."
            )
        covariance = centered.T @ centered / float(record_count - 1)
        average_variance = float(covariance.diag().mean().item())
        if not np.isfinite(average_variance) or average_variance <= 0.0:
            raise ValueError(f"Teacher covariance is degenerate: average_variance={average_variance!r}.")
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        order = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        output_dim = int(self.matrix.shape[1])
        selected_values = eigenvalues[:output_dim]
        selected_vectors = eigenvectors[:, :output_dim]
        regularized = (1.0 - self.shrinkage) * selected_values + self.shrinkage * average_variance
        numerical_floor = max(self.epsilon * average_variance, torch.finfo(eigenvalues.dtype).eps)
        condition_floor = float(regularized.max().item()) / self.max_condition_number
        eigenvalue_floor = max(numerical_floor, condition_floor)
        clamped = regularized.clamp_min(eigenvalue_floor)
        matrix = selected_vectors * clamped.rsqrt().unsqueeze(0)
        if not bool(torch.isfinite(matrix).all()):
            raise ValueError("Teacher whitening matrix contains non-finite values.")
        self.mean.copy_(mean.to(device=self.mean.device, dtype=self.mean.dtype))
        self.matrix.copy_(matrix.to(device=self.matrix.device, dtype=self.matrix.dtype))
        self.fitted_records.fill_(record_count)
        self.fit_metrics = {
            "records": float(record_count),
            "mean_norm": float(mean.norm().item()),
            "average_variance": average_variance,
            "input_dim": float(samples.shape[1]),
            "output_dim": float(output_dim),
            "within_anchor_covariance": float(covariance_residuals is not None),
            "eigenvalue_min": float(eigenvalues.min().item()),
            "eigenvalue_max": float(eigenvalues.max().item()),
            "selected_eigenvalue_min": float(selected_values.min().item()),
            "selected_eigenvalue_max": float(selected_values.max().item()),
            "explained_variance_ratio": float(
                selected_values.clamp_min(0.0).sum().div(eigenvalues.clamp_min(0.0).sum().clamp_min(1.0e-12)).item()
            ),
            "configured_max_condition_number": float(self.max_condition_number),
            "eigenvalue_floor": float(eigenvalue_floor),
            "regularized_condition_number": float((clamped.max() / clamped.min()).item()),
        }
        return dict(self.fit_metrics)

    def transform(self, hidden: torch.Tensor) -> torch.Tensor:
        if not self.is_fitted:
            raise RuntimeError("Teacher whitening must be fitted before alignment training or evaluation.")
        hidden_float = hidden.float()
        return (hidden_float - self.mean) @ self.matrix

    def forward(
        self,
        student_hidden: torch.Tensor,
        teacher_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # The same affine map preserves a single shared coordinate system for both branches.
        return self.transform(student_hidden), self.transform(teacher_hidden)


AlignmentFeatureTransform = AlignmentProjectionPair | FixedTeacherWhitening


def apply_alignment_feature_transform(
    feature_transform: AlignmentFeatureTransform | None,
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if feature_transform is None:
        return student_hidden, teacher_hidden
    return feature_transform(student_hidden, teacher_hidden)


def exclude_semantic_false_negatives(
    logits: torch.Tensor,
    labels: torch.Tensor,
    query_target_ids: torch.Tensor | None,
    candidate_target_ids: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Exclude same-answer candidates from the denominator without changing the paired positive."""
    if query_target_ids is None or candidate_target_ids is None:
        return logits, {}
    query_ids = query_target_ids.to(device=logits.device, dtype=torch.long).flatten()
    candidate_ids = candidate_target_ids.to(device=logits.device, dtype=torch.long).flatten()
    if int(query_ids.numel()) != int(logits.shape[0]) or int(candidate_ids.numel()) != int(logits.shape[1]):
        raise ValueError(
            "Semantic target IDs must match the contrastive logits: "
            f"queries={query_ids.numel()}/{logits.shape[0]}, "
            f"candidates={candidate_ids.numel()}/{logits.shape[1]}."
        )
    same_target = query_ids[:, None].eq(candidate_ids[None, :])
    paired_positive = torch.zeros_like(same_target)
    paired_positive.scatter_(1, labels[:, None], True)
    false_negative_mask = same_target & ~paired_positive
    masked_logits = logits.masked_fill(false_negative_mask, torch.finfo(logits.dtype).min)
    possible_negative_count = max(1, int(logits.shape[1]) - 1)
    valid_negative_count = (~false_negative_mask & ~paired_positive).sum(dim=1).float().mean()
    return masked_logits, {
        "semantic_collision_fraction": float(
            (false_negative_mask.float().sum(dim=1) / possible_negative_count).mean().detach().cpu().item()
        ),
        "valid_negative_count": float(valid_negative_count.detach().cpu().item()),
        "semantic_target_unique_fraction": float(
            candidate_ids.unique().numel() / max(1, candidate_ids.numel())
        ),
    }


@torch.no_grad()
def semantic_top1_accuracy(
    logits: torch.Tensor,
    query_target_ids: torch.Tensor | None,
    candidate_target_ids: torch.Tensor | None,
) -> float:
    if query_target_ids is None or candidate_target_ids is None:
        return 0.0
    query_ids = query_target_ids.to(device=logits.device, dtype=torch.long).flatten()
    candidate_ids = candidate_target_ids.to(device=logits.device, dtype=torch.long).flatten()
    predicted_ids = candidate_ids[logits.argmax(dim=1)]
    return float(predicted_ids.eq(query_ids).float().mean().cpu().item())


@torch.no_grad()
def top1_candidate_usage_metrics(
    predictions: torch.Tensor,
    candidate_count: int,
) -> dict[str, float]:
    predictions = predictions.detach().long().flatten()
    candidate_count = int(candidate_count)
    if predictions.numel() == 0 or candidate_count <= 0:
        return {
            "candidate_coverage": 0.0,
            "max_candidate_hit_fraction": 0.0,
            "candidate_hit_entropy": 0.0,
        }
    counts = torch.bincount(predictions.cpu(), minlength=candidate_count).float()
    probabilities = counts[counts > 0] / float(predictions.numel())
    entropy = -(probabilities * probabilities.log()).sum()
    normalized_entropy = (
        entropy / float(np.log(candidate_count))
        if candidate_count > 1
        else torch.ones((), dtype=entropy.dtype)
    )
    return {
        "candidate_coverage": float(counts.gt(0).sum().item() / candidate_count),
        "max_candidate_hit_fraction": float(counts.max().item() / predictions.numel()),
        "candidate_hit_entropy": float(normalized_entropy.item()),
    }


def prefixed_metrics(metrics: Mapping[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}_{key}": float(value) for key, value in metrics.items()}


def normalized_contrastive_direction_weights(
    i2t_weight: float = 0.5,
    t2i_weight: float = 0.5,
) -> tuple[float, float]:
    i2t = float(i2t_weight)
    t2i = float(t2i_weight)
    if i2t < 0.0 or t2i < 0.0 or i2t + t2i <= 0.0:
        raise ValueError(
            "Contrastive direction weights must be non-negative and have a positive sum. "
            f"Got i2t={i2t}, t2i={t2i}."
        )
    total = i2t + t2i
    return i2t / total, t2i / total


def symmetric_contrastive_loss(
    tensor_embedding: torch.Tensor,
    text_embedding: torch.Tensor,
    temperature: float,
    semantic_target_ids: torch.Tensor | None = None,
    *,
    i2t_weight: float = 0.5,
    t2i_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    i2t_weight, t2i_weight = normalized_contrastive_direction_weights(i2t_weight, t2i_weight)
    logits = tensor_embedding @ text_embedding.T / max(float(temperature), 1.0e-6)
    labels = torch.arange(logits.shape[0], device=logits.device)
    masked_logits, collision_metrics = exclude_semantic_false_negatives(
        logits,
        labels,
        semantic_target_ids,
        semantic_target_ids,
    )
    loss_i2t = F.cross_entropy(masked_logits, labels)
    loss_t2i = F.cross_entropy(masked_logits.T, labels)
    loss = i2t_weight * loss_i2t + t2i_weight * loss_t2i
    with torch.no_grad():
        i2t_predictions = logits.argmax(dim=1)
        t2i_predictions = logits.T.argmax(dim=1)
        i2t_accuracy = (i2t_predictions == labels).float().mean()
        t2i_accuracy = (t2i_predictions == labels).float().mean()
        strict_i2t_loss = F.cross_entropy(logits, labels)
        strict_t2i_loss = F.cross_entropy(logits.T, labels)
        strict_loss = i2t_weight * strict_i2t_loss + t2i_weight * strict_t2i_loss
        semantic_i2t_accuracy = semantic_top1_accuracy(logits, semantic_target_ids, semantic_target_ids)
        semantic_t2i_accuracy = semantic_top1_accuracy(logits.T, semantic_target_ids, semantic_target_ids)
    metrics = {
        "i2t_loss": float(loss_i2t.detach().cpu().item()),
        "t2i_loss": float(loss_t2i.detach().cpu().item()),
        "i2t_accuracy": float(i2t_accuracy.detach().cpu().item()),
        "t2i_accuracy": float(t2i_accuracy.detach().cpu().item()),
        "candidate_count": float(logits.shape[1]),
        "strict_i2t_loss": float(strict_i2t_loss.detach().cpu().item()),
        "strict_t2i_loss": float(strict_t2i_loss.detach().cpu().item()),
        "strict_contrastive_loss": float(strict_loss.detach().cpu().item()),
        "i2t_weight": float(i2t_weight),
        "t2i_weight": float(t2i_weight),
    }
    metrics.update(prefixed_metrics(top1_candidate_usage_metrics(i2t_predictions, logits.shape[1]), "i2t"))
    metrics.update(prefixed_metrics(top1_candidate_usage_metrics(t2i_predictions, logits.shape[0]), "t2i"))
    metrics.update(collision_metrics)
    if semantic_target_ids is not None:
        metrics["semantic_i2t_accuracy"] = semantic_i2t_accuracy
        metrics["semantic_t2i_accuracy"] = semantic_t2i_accuracy
    return loss, metrics


def distributed_symmetric_contrastive_loss(
    tensor_embedding: torch.Tensor,
    text_embedding: torch.Tensor,
    temperature: float,
    semantic_target_ids: torch.Tensor | None = None,
    *,
    i2t_weight: float = 0.5,
    t2i_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    i2t_weight, t2i_weight = normalized_contrastive_direction_weights(i2t_weight, t2i_weight)
    if not distributed_is_initialized():
        return symmetric_contrastive_loss(
            tensor_embedding,
            text_embedding,
            temperature,
            semantic_target_ids,
            i2t_weight=i2t_weight,
            t2i_weight=t2i_weight,
        )
    local_batch = int(tensor_embedding.shape[0])
    tensor_all = gather_with_grad(tensor_embedding)
    text_all = gather_with_grad(text_embedding)
    label_offset = distributed_rank() * local_batch
    labels = torch.arange(local_batch, device=tensor_embedding.device) + int(label_offset)
    logits_i2t = tensor_embedding @ text_all.T / max(float(temperature), 1.0e-6)
    logits_t2i = text_embedding @ tensor_all.T / max(float(temperature), 1.0e-6)
    target_ids_all = gather_without_grad(semantic_target_ids) if semantic_target_ids is not None else None
    masked_logits_i2t, collision_metrics = exclude_semantic_false_negatives(
        logits_i2t,
        labels,
        semantic_target_ids,
        target_ids_all,
    )
    masked_logits_t2i, _ = exclude_semantic_false_negatives(
        logits_t2i,
        labels,
        semantic_target_ids,
        target_ids_all,
    )
    loss_i2t = F.cross_entropy(masked_logits_i2t, labels)
    loss_t2i = F.cross_entropy(masked_logits_t2i, labels)
    loss = i2t_weight * loss_i2t + t2i_weight * loss_t2i
    with torch.no_grad():
        i2t_predictions = logits_i2t.argmax(dim=1)
        t2i_predictions = logits_t2i.argmax(dim=1)
        i2t_accuracy = (i2t_predictions == labels).float().mean()
        t2i_accuracy = (t2i_predictions == labels).float().mean()
        strict_i2t_loss = F.cross_entropy(logits_i2t, labels)
        strict_t2i_loss = F.cross_entropy(logits_t2i, labels)
        strict_loss = i2t_weight * strict_i2t_loss + t2i_weight * strict_t2i_loss
        semantic_i2t_accuracy = semantic_top1_accuracy(logits_i2t, semantic_target_ids, target_ids_all)
        semantic_t2i_accuracy = semantic_top1_accuracy(logits_t2i, semantic_target_ids, target_ids_all)
        all_i2t_predictions = gather_without_grad(i2t_predictions)
        all_t2i_predictions = gather_without_grad(t2i_predictions)
    metrics = {
        "i2t_loss": float(loss_i2t.detach().cpu().item()),
        "t2i_loss": float(loss_t2i.detach().cpu().item()),
        "i2t_accuracy": float(i2t_accuracy.detach().cpu().item()),
        "t2i_accuracy": float(t2i_accuracy.detach().cpu().item()),
        "candidate_count": float(text_all.shape[0]),
        "strict_i2t_loss": float(strict_i2t_loss.detach().cpu().item()),
        "strict_t2i_loss": float(strict_t2i_loss.detach().cpu().item()),
        "strict_contrastive_loss": float(strict_loss.detach().cpu().item()),
        "i2t_weight": float(i2t_weight),
        "t2i_weight": float(t2i_weight),
    }
    metrics.update(prefixed_metrics(top1_candidate_usage_metrics(all_i2t_predictions, text_all.shape[0]), "i2t"))
    metrics.update(prefixed_metrics(top1_candidate_usage_metrics(all_t2i_predictions, tensor_all.shape[0]), "t2i"))
    metrics.update(collision_metrics)
    if semantic_target_ids is not None:
        metrics["semantic_i2t_accuracy"] = semantic_i2t_accuracy
        metrics["semantic_t2i_accuracy"] = semantic_t2i_accuracy
    return loss, metrics


@torch.no_grad()
def retrieval_accuracy(
    tensor_embedding: torch.Tensor,
    text_embedding: torch.Tensor,
    temperature: float,
    *,
    i2t_weight: float = 0.5,
    t2i_weight: float = 0.5,
) -> dict[str, float]:
    i2t_weight, t2i_weight = normalized_contrastive_direction_weights(i2t_weight, t2i_weight)
    logits = tensor_embedding.detach().float() @ text_embedding.detach().float().T / max(float(temperature), 1.0e-6)
    labels = torch.arange(logits.shape[0], device=logits.device)
    i2t_predictions = logits.argmax(dim=1)
    t2i_predictions = logits.T.argmax(dim=1)
    i2t_loss = F.cross_entropy(logits, labels)
    t2i_loss = F.cross_entropy(logits.T, labels)
    result = {
        "contrastive_loss": float((i2t_weight * i2t_loss + t2i_weight * t2i_loss).cpu().item()),
        "i2t_loss": float(i2t_loss.cpu().item()),
        "t2i_loss": float(t2i_loss.cpu().item()),
        "i2t_accuracy": float((i2t_predictions == labels).float().mean().cpu().item()),
        "t2i_accuracy": float((t2i_predictions == labels).float().mean().cpu().item()),
        "candidate_count": float(logits.shape[1]),
    }
    result.update(prefixed_metrics(top1_candidate_usage_metrics(i2t_predictions, logits.shape[1]), "i2t"))
    result.update(prefixed_metrics(top1_candidate_usage_metrics(t2i_predictions, logits.shape[0]), "t2i"))
    return result


@torch.no_grad()
def full_retrieval_accuracy(
    tensor_embedding: torch.Tensor,
    text_embedding: torch.Tensor,
    temperature: float,
    chunk_size: int,
    semantic_target_ids: torch.Tensor | None = None,
    *,
    i2t_weight: float = 0.5,
    t2i_weight: float = 0.5,
) -> dict[str, float]:
    i2t_weight, t2i_weight = normalized_contrastive_direction_weights(i2t_weight, t2i_weight)
    tensor_cpu = tensor_embedding.detach().float().cpu()
    text_cpu = text_embedding.detach().float().cpu()
    total = int(tensor_cpu.shape[0])
    if total == 0:
        return {
            "contrastive_loss": 0.0,
            "i2t_loss": 0.0,
            "t2i_loss": 0.0,
            "strict_contrastive_loss": 0.0,
            "strict_i2t_loss": 0.0,
            "strict_t2i_loss": 0.0,
            "i2t_accuracy": 0.0,
            "t2i_accuracy": 0.0,
            "i2t_weight": float(i2t_weight),
            "t2i_weight": float(t2i_weight),
        }
    labels = torch.arange(total)
    chunk = max(1, int(chunk_size))
    i2t_correct = 0
    t2i_correct = 0
    i2t_loss_sum = 0.0
    t2i_loss_sum = 0.0
    strict_i2t_loss_sum = 0.0
    strict_t2i_loss_sum = 0.0
    semantic_i2t_correct = 0
    semantic_t2i_correct = 0
    i2t_predictions_all: list[torch.Tensor] = []
    t2i_predictions_all: list[torch.Tensor] = []
    target_ids = semantic_target_ids.detach().long().cpu().flatten() if semantic_target_ids is not None else None
    if target_ids is not None and int(target_ids.numel()) != total:
        raise ValueError(
            f"Global semantic target count {target_ids.numel()} does not match retrieval rows {total}."
        )
    for start in range(0, total, chunk):
        end = min(total, start + chunk)
        logits = tensor_cpu[start:end] @ text_cpu.T / max(float(temperature), 1.0e-6)
        local_labels = labels[start:end]
        predictions = logits.argmax(dim=1)
        i2t_predictions_all.append(predictions)
        i2t_correct += int((predictions == local_labels).sum().item())
        strict_i2t_loss_sum += float(F.cross_entropy(logits, local_labels, reduction="sum").item())
        masked_logits, _ = exclude_semantic_false_negatives(
            logits,
            local_labels,
            target_ids[start:end] if target_ids is not None else None,
            target_ids,
        )
        i2t_loss_sum += float(F.cross_entropy(masked_logits, local_labels, reduction="sum").item())
        if target_ids is not None:
            semantic_i2t_correct += int(target_ids[predictions].eq(target_ids[start:end]).sum().item())
    for start in range(0, total, chunk):
        end = min(total, start + chunk)
        logits = text_cpu[start:end] @ tensor_cpu.T / max(float(temperature), 1.0e-6)
        local_labels = labels[start:end]
        predictions = logits.argmax(dim=1)
        t2i_predictions_all.append(predictions)
        t2i_correct += int((predictions == local_labels).sum().item())
        strict_t2i_loss_sum += float(F.cross_entropy(logits, local_labels, reduction="sum").item())
        masked_logits, _ = exclude_semantic_false_negatives(
            logits,
            local_labels,
            target_ids[start:end] if target_ids is not None else None,
            target_ids,
        )
        t2i_loss_sum += float(F.cross_entropy(masked_logits, local_labels, reduction="sum").item())
        if target_ids is not None:
            semantic_t2i_correct += int(target_ids[predictions].eq(target_ids[start:end]).sum().item())
    i2t_loss = i2t_loss_sum / max(1, total)
    t2i_loss = t2i_loss_sum / max(1, total)
    strict_i2t_loss = strict_i2t_loss_sum / max(1, total)
    strict_t2i_loss = strict_t2i_loss_sum / max(1, total)
    result = {
        "contrastive_loss": i2t_weight * i2t_loss + t2i_weight * t2i_loss,
        "i2t_loss": i2t_loss,
        "t2i_loss": t2i_loss,
        "strict_contrastive_loss": i2t_weight * strict_i2t_loss + t2i_weight * strict_t2i_loss,
        "strict_i2t_loss": strict_i2t_loss,
        "strict_t2i_loss": strict_t2i_loss,
        "i2t_accuracy": i2t_correct / max(1, total),
        "t2i_accuracy": t2i_correct / max(1, total),
        "i2t_weight": float(i2t_weight),
        "t2i_weight": float(t2i_weight),
    }
    result.update(
        prefixed_metrics(
            top1_candidate_usage_metrics(torch.cat(i2t_predictions_all), total),
            "i2t",
        )
    )
    result.update(
        prefixed_metrics(
            top1_candidate_usage_metrics(torch.cat(t2i_predictions_all), total),
            "t2i",
        )
    )
    if target_ids is not None:
        _, counts = target_ids.unique(return_counts=True)
        collision_count = (counts.float() * (counts.float() - 1.0)).sum()
        result.update(
            {
                "semantic_i2t_accuracy": semantic_i2t_correct / max(1, total),
                "semantic_t2i_accuracy": semantic_t2i_correct / max(1, total),
                "semantic_collision_fraction": float(
                    (collision_count / max(1, total * (total - 1))).item()
                ),
                "valid_negative_count": float(
                    (total - counts.float().square().sum() / max(1, total)).item()
                ),
                "semantic_target_unique_fraction": float(counts.numel() / max(1, total)),
            }
        )
    return result


def reconstruction_loss_with_diagnostics(
    compressor: nn.Module,
    latent_map: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    reconstruction = compressor.decode({"latent_map": latent_map})
    target_float = target.float()
    reconstruction_float = reconstruction.float()
    error_flat = (reconstruction_float - target_float).flatten(1)
    target_flat = target_float.flatten(1)
    mse_per_record = error_flat.square().mean(dim=1)
    target_mean = target_flat.mean(dim=1, keepdim=True)
    mean_baseline_mse_per_record = (target_flat - target_mean).square().mean(dim=1)
    zero_baseline_mse_per_record = target_flat.square().mean(dim=1)
    eps = torch.finfo(torch.float32).eps
    mean_record_relative_mse = mse_per_record / mean_baseline_mse_per_record.clamp_min(eps)
    mean_record_relative_rmse = mse_per_record.sqrt() / mean_baseline_mse_per_record.sqrt().clamp_min(eps)
    loss = mse_per_record.mean()
    mean_baseline_mse = mean_baseline_mse_per_record.mean()
    relative_mse = loss / mean_baseline_mse.clamp_min(eps)
    relative_rmse = loss.sqrt() / mean_baseline_mse.sqrt().clamp_min(eps)
    diagnostics = {
        "rmse": float(loss.detach().sqrt().cpu().item()),
        "target_abs_mean": float(target_flat.abs().mean().detach().cpu().item()),
        "target_std": float(mean_baseline_mse.detach().sqrt().cpu().item()),
        "mean_baseline_mse": float(mean_baseline_mse.detach().cpu().item()),
        "zero_baseline_mse": float(zero_baseline_mse_per_record.mean().detach().cpu().item()),
        "relative_mse_to_mean_baseline": float(relative_mse.detach().cpu().item()),
        "relative_rmse_to_target_std": float(relative_rmse.detach().cpu().item()),
        "mean_record_relative_mse_to_mean_baseline": float(
            mean_record_relative_mse.mean().detach().cpu().item()
        ),
        "mean_record_relative_rmse_to_target_std": float(
            mean_record_relative_rmse.mean().detach().cpu().item()
        ),
    }
    return loss, diagnostics


def hidden_at_last_non_padding(
    hidden_states: Sequence[torch.Tensor],
    attention_mask: torch.Tensor,
    teacher_layer: int,
    prefix_tokens: int = 0,
) -> torch.Tensor:
    layer_index = int(teacher_layer)
    if not -len(hidden_states) <= layer_index < len(hidden_states):
        raise ValueError(
            f"teacher_layer={teacher_layer} is out of range for {len(hidden_states)} hidden-state tensors."
        )
    hidden = hidden_states[layer_index]
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device).unsqueeze(0)
    last_indices = (attention_mask.long() * positions).amax(dim=1) + int(prefix_tokens)
    batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[batch_indices, last_indices]


def validate_teacher_hidden_state_index(teacher_layer: int, num_hidden_layers: int) -> None:
    layer_index = int(teacher_layer)
    layer_count = int(num_hidden_layers)
    if layer_count <= 0:
        raise ValueError(f"The LLM reports an invalid num_hidden_layers={layer_count}.")
    if layer_index <= 0:
        raise ValueError(
            "patch_alignment.teacher_layer must be at least 1. hidden_states[0] is the input embedding "
            "before any transformer block, so the shared readout token cannot depend on the preceding tensor."
        )
    if layer_index > layer_count:
        raise ValueError(
            "patch_alignment.teacher_layer exceeds the LLM depth: "
            f"teacher_layer={layer_index}, num_hidden_layers={layer_count}. "
            "For Hugging Face causal LMs, hidden_states[k] is the output after k transformer blocks."
        )


def llm_backbone(llm: nn.Module) -> nn.Module:
    backbone = getattr(llm, "model", None)
    return backbone if isinstance(backbone, nn.Module) else llm


def truncate_llm_backbone_to_layer(llm: nn.Module, teacher_layer: int) -> int | None:
    """Keep only the transformer blocks needed for a frozen shallow hidden readout when safe."""
    backbone = llm_backbone(llm)
    layers = getattr(backbone, "layers", None)
    if not isinstance(layers, nn.ModuleList):
        return None
    requested = int(teacher_layer)
    if requested <= 0 or requested > len(layers):
        raise ValueError(
            f"Cannot truncate LLM backbone to teacher_layer={requested}; available blocks={len(layers)}."
        )
    if len(layers) > requested:
        backbone.layers = nn.ModuleList(list(layers[:requested]))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return len(backbone.layers)


def transformer_block_hidden_states(
    llm: nn.Module,
    *,
    attention_mask: torch.Tensor,
    layer_indices: Sequence[int],
    input_ids: torch.Tensor | None = None,
    inputs_embeds: torch.Tensor | None = None,
) -> dict[int, torch.Tensor]:
    """Capture only requested transformer-block outputs, before any final backbone norm."""
    backbone = llm_backbone(llm)
    layers = getattr(backbone, "layers", None)
    requested = sorted({int(index) for index in layer_indices})
    if not requested:
        raise ValueError("At least one transformer layer must be requested.")
    if isinstance(layers, nn.ModuleList):
        invalid = [index for index in requested if index <= 0 or index > len(layers)]
        if invalid:
            raise ValueError(
                f"Transformer block indices {invalid} are outside hidden_states[1..{len(layers)}]."
            )
        captured: dict[int, torch.Tensor] = {}

        def capture(index: int):
            def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
                hidden = output[0] if isinstance(output, (tuple, list)) else output
                if not torch.is_tensor(hidden):
                    raise TypeError(
                        f"Transformer block {index} returned unsupported output type {type(output).__name__}."
                    )
                captured[index] = hidden

            return hook

        handles = [layers[index - 1].register_forward_hook(capture(index)) for index in requested]
        try:
            backbone(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=False,
                use_cache=False,
            )
        finally:
            for handle in handles:
                handle.remove()
        missing = [index for index in requested if index not in captured]
        if missing:
            raise RuntimeError(f"Transformer hooks did not capture requested blocks {missing}.")
        return captured

    outputs = backbone(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    hidden_states = outputs.hidden_states
    return {index: hidden_states[index] for index in requested}


def forward_teacher_readout_hidden(
    llm: nn.Module,
    *,
    attention_mask: torch.Tensor,
    teacher_layer: int,
    prefix_tokens: int = 0,
    input_ids: torch.Tensor | None = None,
    inputs_embeds: torch.Tensor | None = None,
) -> torch.Tensor:
    hidden = transformer_block_hidden_states(
        llm,
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        layer_indices=[int(teacher_layer)],
    )[int(teacher_layer)]
    return hidden_at_last_non_padding(
        [hidden],
        attention_mask[:, int(prefix_tokens) :],
        0,
        prefix_tokens=int(prefix_tokens),
    )


def masked_token_norm(embeddings: torch.Tensor, attention_mask: torch.Tensor) -> float:
    mask = attention_mask.to(device=embeddings.device, dtype=torch.bool)
    if not bool(mask.any()):
        return 0.0
    norms = embeddings.detach().float().norm(dim=-1)
    return float(norms[mask].mean().cpu().item())


def mean_token_norm(tokens: torch.Tensor) -> float:
    if tokens.numel() == 0:
        return 0.0
    return float(tokens.detach().float().norm(dim=-1).mean().cpu().item())


def off_diagonal_cosine_mean(embeddings: torch.Tensor) -> float:
    if int(embeddings.shape[0]) < 2:
        return 0.0
    normalized = F.normalize(embeddings.detach().float(), dim=-1)
    similarity = normalized @ normalized.T
    batch_size = int(similarity.shape[0])
    off_diagonal_sum = similarity.sum() - similarity.diag().sum()
    return float((off_diagonal_sum / max(1, batch_size * (batch_size - 1))).cpu().item())


def tokenizer_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(
        str(text),
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]
    if isinstance(encoded, torch.Tensor):
        encoded = encoded.detach().cpu().tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return [int(token_id) for token_id in encoded]


def tokenizer_batch_ids(tokenizer: Any, texts: Sequence[str]) -> list[list[int]]:
    if not texts:
        return []
    encoded = tokenizer(
        [str(text) for text in texts],
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]
    if isinstance(encoded, torch.Tensor):
        encoded = encoded.detach().cpu().tolist()
    return [[int(token_id) for token_id in row] for row in encoded]


def shared_suffix_token_ids(tokenizer: Any, suffix: str, max_suffix_tokens: int) -> tuple[int, ...]:
    if not str(suffix):
        raise ValueError("Alignment anchor text must not be empty.")
    token_ids = tuple(tokenizer_ids(tokenizer, str(suffix)))
    if not token_ids:
        raise ValueError("Alignment anchor text tokenized to an empty sequence.")
    if len(token_ids) > int(max_suffix_tokens):
        raise ValueError(
            "Alignment anchor text is too long after tokenization: "
            f"{len(token_ids)} tokens exceeds max_shared_suffix_tokens={int(max_suffix_tokens)}."
        )
    return token_ids


def build_static_alignment_anchor(
    *,
    tokenizer: Any,
    mode: str,
    representation_suffix: str,
    max_anchor_tokens: int,
) -> AlignmentAnchor:
    normalized_mode = str(mode).strip().lower()
    if normalized_mode == "eos":
        eos_token_id = tokenizer.eos_token_id
        if eos_token_id is None:
            raise ValueError("Tokenizer must define eos_token_id when the eos alignment anchor is enabled.")
        return AlignmentAnchor(
            name="eos",
            mode="eos",
            token_ids=(int(eos_token_id),),
        )
    if normalized_mode == "representation":
        return AlignmentAnchor(
            name="representation",
            mode="representation",
            token_ids=shared_suffix_token_ids(
                tokenizer,
                str(representation_suffix),
                int(max_anchor_tokens),
            ),
            text=str(representation_suffix),
        )
    raise ValueError(f"Static alignment anchor mode must be eos or representation, got {mode!r}.")


def validate_probe_anchor_contract(anchor: AlignmentAnchor) -> None:
    if anchor.mode != "probe" or anchor.probe_family not in PROBE_FAMILIES:
        raise ValueError("Probe contract validation requires a supported probe anchor.")
    template_count = int(PROBE_TEMPLATE_COUNTS[str(anchor.probe_family)])
    if anchor.probe_template_index is None or not 0 <= int(anchor.probe_template_index) < template_count:
        raise ValueError(f"Probe anchor has an invalid template index: {anchor.probe_template_index!r}.")
    text = str(anchor.text or "")
    if not text.startswith("\n") or not text.endswith(PROBE_READOUT_ENDINGS):
        raise ValueError(
            "Probe stem must start with a newline and end immediately before a numeric readout "
            f"using one of {PROBE_READOUT_ENDINGS}: {text!r}."
        )
    lowered = text.lower()
    leaked_markers = [marker for marker in PROBE_FORBIDDEN_INPUT_MARKERS if marker in lowered]
    if leaked_markers:
        raise ValueError(f"Probe stem contains forbidden choice/QA-format markers {leaked_markers}: {text!r}.")


def build_numeric_probe_anchor(
    *,
    tokenizer: Any,
    patch_size: int,
    channel_count: int,
    families: str | Sequence[str],
    region_size: int,
    probe_index: int,
    seed: int,
    max_anchor_tokens: int,
    template_index: int | None = None,
) -> AlignmentAnchor:
    probe_families = [family.lower() for family in parse_csv(families)]
    if not probe_families:
        raise ValueError("patch_alignment.probe_families must not be empty in probe mode.")
    unsupported = sorted(set(probe_families) - set(PROBE_FAMILIES))
    if unsupported:
        raise ValueError(f"Unsupported probe families: {unsupported}.")
    if int(patch_size) <= 1:
        raise ValueError("Probe anchors require patch_size greater than 1.")
    if int(channel_count) <= 0:
        raise ValueError("Probe anchors require at least one tensor channel.")

    family = probe_families[int(probe_index) % len(probe_families)]
    rng = random.Random(int(seed) + 1_000_003 * int(probe_index))
    channel = rng.randrange(int(channel_count))

    def select_template(templates: Sequence[str]) -> tuple[int, str]:
        expected_count = int(PROBE_TEMPLATE_COUNTS[family])
        if len(templates) != expected_count:
            raise ValueError(
                f"Probe family {family} must define exactly {expected_count} templates, "
                f"but {family} defines {len(templates)}."
            )
        selected = (
            int(template_index)
            if template_index is not None
            else (int(probe_index) // len(probe_families)) % expected_count
        )
        if not 0 <= selected < expected_count:
            raise ValueError(
                f"Probe template_index must be between 0 and {expected_count - 1}, got {selected}."
            )
        return selected, str(templates[selected])

    def point_text(row: int, col: int) -> str:
        location = f"row {row + 1}, column {col + 1}"
        if int(channel_count) > 1:
            location += f" in channel {channel + 1}"
        return location

    def region_text(row: int, col: int, size: int) -> str:
        location = f"rows {row + 1}-{row + size} and columns {col + 1}-{col + size}"
        if int(channel_count) > 1:
            location += f" in channel {channel + 1}"
        return location

    if family == "point_value":
        position = rng.randrange(int(patch_size) * int(patch_size))
        row, col = divmod(position, int(patch_size))
        location = point_text(row, col)
        templates = (
            f"\nThe value at {location} is",
            f"\nThe entry at {location} equals",
            f"\nReading {location} gives",
            f"\nAt {location}, this matrix contains",
            f"\nAt {location}, the value is",
            f"\nThe matrix entry at {location} is",
            f"\nFor {location}, the recorded value is",
            f"\nThe number stored at {location} is",
        )
        selected_template_index, text = select_template(templates)
        parameters = (channel, row, col)
    elif family in {"point_difference", "point_mean"}:
        first, second = rng.sample(range(int(patch_size) * int(patch_size)), 2)
        row_a, col_a = divmod(first, int(patch_size))
        row_b, col_b = divmod(second, int(patch_size))
        location_a = point_text(row_a, col_a)
        location_b = point_text(row_b, col_b)
        if family == "point_difference":
            templates = (
                f"\nThe value at {location_a} minus the value at {location_b} is",
                f"\nThe result of subtracting the value at {location_b} from the value at {location_a} is",
                f"\nThe signed difference from {location_b} to {location_a} is",
                f"\nThe signed difference, value at {location_a} minus value at {location_b}, is",
            )
        else:
            templates = (
                f"\nThe mean of the values at {location_a} and {location_b} is",
                f"\nThe result of averaging the values at {location_a} and {location_b} is",
                f"\nThe two-point average for {location_a} and {location_b} is",
                f"\nThe arithmetic mean of the values at {location_a} and {location_b} is",
            )
        selected_template_index, text = select_template(templates)
        parameters = (channel, row_a, col_a, row_b, col_b)
    else:
        size = int(region_size)
        if size <= 0 or size >= int(patch_size):
            raise ValueError("patch_alignment.probe_region_size must be between 1 and patch_size - 1.")
        positions_per_axis = int(patch_size) - size + 1
        position = rng.randrange(positions_per_axis * positions_per_axis)
        row, col = divmod(position, positions_per_axis)
        region = region_text(row, col, size)
        if family == "region_mean":
            templates = (
                f"\nThe mean over {region} is",
                f"\nThe average value over {region} is",
                f"\nThe regional mean for {region} is",
                f"\nThe result of averaging all values over {region} is",
            )
        else:
            templates = (
                f"\nThe maximum minus the minimum over {region} is",
                f"\nThe max-minus-min range over {region} is",
                f"\nThe difference between the maximum and minimum over {region} is",
                f"\nThe numerical span from minimum to maximum over {region} is",
            )
        selected_template_index, text = select_template(templates)
        parameters = (channel, row, col, size)

    anchor = AlignmentAnchor(
        name=f"probe_{int(probe_index):02d}_{family}_t{selected_template_index}",
        mode="probe",
        token_ids=shared_suffix_token_ids(tokenizer, text, int(max_anchor_tokens)),
        text=text,
        probe_family=family,
        probe_template_index=int(selected_template_index),
        probe_parameters=parameters,
    )
    validate_probe_anchor_contract(anchor)
    return anchor


def probe_targets_from_patches(
    anchor: AlignmentAnchor,
    patches: torch.Tensor,
    decimal_places: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return hidden-only probe targets; these values are never appended to either LLM input."""
    if anchor.mode != "probe" or anchor.probe_family not in PROBE_FAMILIES:
        raise ValueError("Probe targets require a supported probe anchor.")
    if patches.ndim != 4:
        raise ValueError(f"Probe targets require [B,C,H,W] patches, got {tuple(patches.shape)}.")
    decimals = int(decimal_places)
    if decimals < 0 or decimals > 8:
        raise ValueError(f"Probe target decimal_places must be between 0 and 8, got {decimals}.")
    scale = float(10**decimals)
    visible = torch.round(patches.detach().float() * scale) / scale
    parameters = tuple(int(value) for value in anchor.probe_parameters)
    family = str(anchor.probe_family)
    if family == "point_value":
        channel, row, col = parameters
        targets = visible[:, channel, row, col]
    elif family in {"point_difference", "point_mean"}:
        channel, row_a, col_a, row_b, col_b = parameters
        value_a = visible[:, channel, row_a, col_a]
        value_b = visible[:, channel, row_b, col_b]
        targets = value_a - value_b if family == "point_difference" else 0.5 * (value_a + value_b)
    elif family in {"region_mean", "region_range"}:
        channel, row, col, size = parameters
        region = visible[:, channel, row : row + size, col : col + size].flatten(1)
        targets = region.mean(dim=1) if family == "region_mean" else region.amax(dim=1) - region.amin(dim=1)
    else:  # pragma: no cover - guarded by the contract above
        raise ValueError(f"Unsupported probe family: {family!r}.")
    if not bool(torch.isfinite(targets).all()):
        raise ValueError(f"Probe family {family!r} produced non-finite targets.")
    target_ids = torch.round(targets * scale).to(dtype=torch.long)
    quantized_targets = target_ids.to(dtype=torch.float32) / scale
    return quantized_targets, target_ids


def tokenize_contents_with_anchor(
    *,
    tokenizer: Any,
    contents: Sequence[str],
    anchor: AlignmentAnchor,
    max_tokens: int,
    require_under_max_length: bool,
    context: str,
) -> SharedSuffixTokenBatch:
    suffix_ids = tuple(int(token_id) for token_id in anchor.token_ids)
    if not suffix_ids:
        raise ValueError(f"Alignment anchor {anchor.name!r} has no token IDs.")
    content_budget = int(max_tokens) - len(suffix_ids)
    if content_budget <= 0:
        raise ValueError(
            "patch_alignment.max_text_tokens must leave room for tensor values before the alignment anchor. "
            f"Got max_text_tokens={int(max_tokens)} and anchor_tokens={len(suffix_ids)}."
        )

    packed_rows: list[list[int]] = []
    content_lengths: list[int] = []
    truncated_count = 0
    for content_ids in tokenizer_batch_ids(tokenizer, contents):
        content_lengths.append(len(content_ids))
        if len(content_ids) > content_budget:
            truncated_count += 1
            content_ids = content_ids[:content_budget]
        packed_rows.append(content_ids + list(suffix_ids))

    if not packed_rows:
        raise ValueError(f"{context} received an empty batch.")
    if require_under_max_length and truncated_count > 0:
        raise ValueError(
            f"{context} truncated numeric tensor content for {truncated_count}/{len(packed_rows)} sequences. "
            "Increase patch_alignment.max_text_tokens or reduce patch_alignment.patch_size/text_decimal_places. "
            "The alignment anchor was preserved, but training on incomplete tensor text is disabled."
        )

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("Tokenizer must define pad_token_id before alignment-anchor tokenization.")
    padded_length = max(len(row) for row in packed_rows)
    input_ids = torch.full((len(packed_rows), padded_length), int(pad_id), dtype=torch.long)
    attention_mask = torch.zeros((len(packed_rows), padded_length), dtype=torch.long)
    for row_index, row in enumerate(packed_rows):
        input_ids[row_index, : len(row)] = torch.tensor(row, dtype=torch.long)
        attention_mask[row_index, : len(row)] = 1

    lengths = attention_mask.sum(dim=1)
    missing_anchor = 0
    suffix_tensor = torch.tensor(suffix_ids, dtype=torch.long)
    for row_index, length in enumerate(lengths.tolist()):
        length = int(length)
        if length < len(suffix_ids):
            missing_anchor += 1
            continue
        observed_suffix = input_ids[row_index, length - len(suffix_ids) : length]
        if not torch.equal(observed_suffix.cpu(), suffix_tensor):
            missing_anchor += 1
    if missing_anchor:
        raise ValueError(
            f"{context} did not preserve the alignment anchor at the final non-padding token for "
            f"{missing_anchor}/{len(packed_rows)} sequences. This violates the shared-anchor readout contract."
        )
    max_length_hits = int((lengths >= int(max_tokens)).sum().item())
    batch_size = len(packed_rows)
    metrics = {
        "token_count_mean": float(lengths.float().mean().item()),
        "token_count_max": float(lengths.max().item()),
        "content_token_count_mean": float(sum(content_lengths) / max(1, batch_size)),
        "content_token_count_max": float(max(content_lengths, default=0)),
        "suffix_token_count": float(len(suffix_ids)),
        "content_truncated_fraction": float(truncated_count / max(1, batch_size)),
        "max_length_hit_fraction": float(max_length_hits / max(1, batch_size)),
        "anchor_missing_fraction": float(missing_anchor / max(1, batch_size)),
    }
    return SharedSuffixTokenBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        metrics=metrics,
        suffix_token_ids=suffix_ids,
    )


def tokenize_contents_with_shared_suffix(
    *,
    tokenizer: Any,
    contents: Sequence[str],
    suffix: str,
    max_tokens: int,
    max_suffix_tokens: int,
    require_under_max_length: bool,
    context: str,
) -> SharedSuffixTokenBatch:
    return tokenize_contents_with_anchor(
        tokenizer=tokenizer,
        contents=contents,
        anchor=AlignmentAnchor(
            name="shared_suffix",
            mode="representation",
            token_ids=shared_suffix_token_ids(tokenizer, suffix, max_suffix_tokens),
            text=str(suffix),
        ),
        max_tokens=int(max_tokens),
        require_under_max_length=bool(require_under_max_length),
        context=str(context),
    )


def tokenizer_anchor_metrics(
    *,
    tokenizer: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_tokens: int,
    anchor_text: str,
    require_anchor: bool,
    require_under_max_length: bool,
    context: str,
) -> dict[str, float]:
    lengths = attention_mask.long().sum(dim=1)
    max_observed = int(lengths.max().item()) if lengths.numel() else 0
    max_length_hits = int((lengths >= int(max_tokens)).sum().item()) if int(max_tokens) > 0 else 0
    anchor_missing = 0
    first_missing_tail = ""
    if anchor_text:
        anchor_probe = str(anchor_text).rstrip(":")
        positions = torch.arange(attention_mask.shape[1]).unsqueeze(0)
        last_indices = (attention_mask.long().cpu() * positions).amax(dim=1)
        ids_cpu = input_ids.detach().cpu()
        for row, last_index_tensor in enumerate(last_indices):
            last_index = int(last_index_tensor.item())
            tail_start = max(0, last_index - 24)
            tail_ids = ids_cpu[row, tail_start : last_index + 1]
            tail = tokenizer.decode(tail_ids, skip_special_tokens=True)
            if anchor_probe not in tail:
                anchor_missing += 1
                if not first_missing_tail:
                    first_missing_tail = tail
    if require_anchor and anchor_missing > 0:
        raise ValueError(
            f"{context} lost the anchor text during tokenization. "
            f"missing={anchor_missing}/{int(input_ids.shape[0])}, max_text_tokens={max_tokens}. "
            "Increase patch_alignment.max_text_tokens, reduce patch_alignment.patch_size/text_decimal_places, "
            f"or inspect truncation. First decoded tail: {first_missing_tail!r}"
        )
    if require_under_max_length and max_length_hits > 0:
        raise ValueError(
            f"{context} reached max_text_tokens for {max_length_hits}/{int(input_ids.shape[0])} sequences. "
            "This may mean the numeric patch text was truncated. Increase patch_alignment.max_text_tokens, "
            "reduce patch_alignment.patch_size/text_decimal_places, or pass --no-fail-on-text-max-length-hit "
            "only for diagnostics."
        )
    batch_size = max(1, int(input_ids.shape[0]))
    return {
        "token_count_mean": float(lengths.float().mean().item()) if lengths.numel() else 0.0,
        "token_count_max": float(max_observed),
        "max_length_hit_fraction": float(max_length_hits / batch_size),
        "anchor_missing_fraction": float(anchor_missing / batch_size),
    }


def normalize_alignment_embeddings(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return F.normalize(student_hidden.float(), dim=-1), F.normalize(teacher_hidden.float(), dim=-1)


def alignment_branch_means(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    *,
    distributed_batch: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    student_float = student_hidden.float()
    teacher_float = teacher_hidden.float()
    if distributed_batch and distributed_is_initialized():
        student_float = gather_with_grad(student_float)
        teacher_float = gather_without_grad(teacher_float)
    return student_float.mean(dim=0, keepdim=True), teacher_float.mean(dim=0, keepdim=True)


def centered_alignment_embeddings(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    *,
    distributed_batch: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    tensor_embedding = student_hidden.float()
    text_embedding = teacher_hidden.float()
    can_center = int(tensor_embedding.shape[0]) > 1 or (
        bool(distributed_batch) and distributed_is_initialized() and distributed_world_size() > 1
    )
    if can_center:
        student_mean, teacher_mean = alignment_branch_means(
            tensor_embedding,
            text_embedding,
            distributed_batch=bool(distributed_batch),
        )
        tensor_embedding = tensor_embedding - student_mean
        text_embedding = text_embedding - teacher_mean
    return F.normalize(tensor_embedding, dim=-1), F.normalize(text_embedding, dim=-1)


def branch_mean_alignment_loss(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    *,
    distributed_batch: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    student_mean, teacher_mean = alignment_branch_means(
        student_hidden,
        teacher_hidden,
        distributed_batch=bool(distributed_batch),
    )
    student_norm = student_mean.norm(dim=-1).clamp_min(1.0e-8)
    teacher_norm = teacher_mean.norm(dim=-1).clamp_min(1.0e-8)
    cosine = F.cosine_similarity(student_mean, teacher_mean, dim=-1).mean()
    direction_loss = 1.0 - cosine
    log_norm_ratio = torch.log(student_norm / teacher_norm)
    norm_loss = F.smooth_l1_loss(log_norm_ratio, torch.zeros_like(log_norm_ratio))
    loss = direction_loss + norm_loss
    with torch.no_grad():
        l2_distance = (student_mean - teacher_mean).norm(dim=-1).mean()
    return loss, {
        "loss": float(loss.detach().cpu().item()),
        "direction_loss": float(direction_loss.detach().cpu().item()),
        "norm_loss": float(norm_loss.detach().cpu().item()),
        "cosine": float(cosine.detach().cpu().item()),
        "l2_distance": float(l2_distance.detach().cpu().item()),
        "student_norm": float(student_norm.mean().detach().cpu().item()),
        "teacher_norm": float(teacher_norm.mean().detach().cpu().item()),
        "norm_ratio": float((student_norm / teacher_norm).mean().detach().cpu().item()),
    }


def add_weighted_metrics(
    totals: dict[str, float],
    metrics: Mapping[str, float],
    batch_size: int,
    prefix: str,
) -> None:
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            totals[f"{prefix}{key}"] = totals.get(f"{prefix}{key}", 0.0) + float(value) * int(batch_size)


def averaged_metrics(totals: Mapping[str, float], total_records: int) -> dict[str, float]:
    return {key: float(value) / max(1, int(total_records)) for key, value in totals.items()}


def build_student_anchor_texts(
    records: Sequence[Mapping[str, Any]],
    patch_size_override: int | None = None,
) -> list[str]:
    texts: list[str] = []
    for record in records:
        fields = record.get("fields", [])
        if isinstance(fields, Sequence) and not isinstance(fields, str):
            field_text = ",".join(str(field) for field in fields)
        else:
            field_text = str(fields)
        patch_size = int(patch_size_override) if patch_size_override is not None else int(record.get("patch_size", 0))
        texts.append(
            "Represent this PDE tensor patch for numeric reasoning.\n"
            f"fields={field_text} patch_size={patch_size}\n"
            "Representation:"
        )
    return texts


@torch.no_grad()
def text_teacher_hidden(
    llm: nn.Module,
    tokenizer: Any,
    texts: Sequence[str],
    device: torch.device,
    max_tokens: int,
    teacher_layer: int,
    require_anchor: bool,
    require_under_max_length: bool,
    text_layout: str = "legacy_prompt",
    shared_suffix: str = "\nRepresentation:",
    max_shared_suffix_tokens: int = 8,
    alignment_anchor: AlignmentAnchor | None = None,
) -> HiddenBatch:
    if str(text_layout) == "values_shared_suffix":
        anchor = alignment_anchor or AlignmentAnchor(
            name="representation",
            mode="representation",
            token_ids=shared_suffix_token_ids(tokenizer, str(shared_suffix), int(max_shared_suffix_tokens)),
            text=str(shared_suffix),
        )
        packed = tokenize_contents_with_anchor(
            tokenizer=tokenizer,
            contents=texts,
            anchor=anchor,
            max_tokens=int(max_tokens),
            require_under_max_length=bool(require_under_max_length),
            context="teacher tensor text",
        )
        input_ids = packed.input_ids.to(device)
        attention_mask = packed.attention_mask.to(device)
        metrics = dict(packed.metrics)
    else:
        encoded = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=int(max_tokens),
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        metrics = tokenizer_anchor_metrics(
            tokenizer=tokenizer,
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            max_tokens=int(max_tokens),
            anchor_text="Representation:" if require_anchor else "",
            require_anchor=bool(require_anchor),
            require_under_max_length=bool(require_under_max_length),
            context="teacher text",
        )
    text_embeds = llm.get_input_embeddings()(input_ids)
    metrics["token_embedding_norm"] = masked_token_norm(text_embeds, attention_mask)
    hidden = forward_teacher_readout_hidden(
        llm,
        input_ids=input_ids,
        attention_mask=attention_mask,
        teacher_layer=int(teacher_layer),
    )
    hidden = hidden.detach()
    metrics["hidden_norm"] = mean_token_norm(hidden.unsqueeze(1))
    metrics["hidden_pairwise_cosine"] = off_diagonal_cosine_mean(hidden)
    return HiddenBatch(hidden=hidden, metrics=metrics)


def tensor_student_hidden(
    llm: nn.Module,
    tokenizer: Any,
    soft_prompts: torch.Tensor,
    records: Sequence[Mapping[str, Any]],
    device: torch.device,
    max_tokens: int,
    teacher_layer: int,
    patch_size: int | None,
    require_under_max_length: bool,
    text_layout: str = "legacy_prompt",
    shared_suffix: str = "\nRepresentation:",
    max_shared_suffix_tokens: int = 8,
    alignment_anchor: AlignmentAnchor | None = None,
) -> HiddenBatch:
    if str(text_layout) == "values_shared_suffix":
        anchor = alignment_anchor or AlignmentAnchor(
            name="representation",
            mode="representation",
            token_ids=shared_suffix_token_ids(tokenizer, str(shared_suffix), int(max_shared_suffix_tokens)),
            text=str(shared_suffix),
        )
        suffix_ids = anchor.token_ids
        if int(soft_prompts.shape[1]) + len(suffix_ids) > int(max_tokens):
            raise ValueError(
                "Student soft prompts plus shared suffix exceed patch_alignment.max_text_tokens: "
                f"{int(soft_prompts.shape[1])}+{len(suffix_ids)}>{int(max_tokens)}."
            )
        input_ids = torch.tensor(suffix_ids, dtype=torch.long, device=device).unsqueeze(0).expand(
            int(soft_prompts.shape[0]), -1
        )
        text_attention_mask = torch.ones_like(input_ids)
        metrics = {
            "token_count_mean": float(len(suffix_ids)),
            "token_count_max": float(len(suffix_ids)),
            "content_token_count_mean": 0.0,
            "content_token_count_max": 0.0,
            "suffix_token_count": float(len(suffix_ids)),
            "content_truncated_fraction": 0.0,
            "max_length_hit_fraction": 0.0,
            "anchor_missing_fraction": 0.0,
        }
    else:
        encoded = tokenizer(
            build_student_anchor_texts(records, patch_size_override=patch_size),
            padding=True,
            truncation=True,
            max_length=int(max_tokens),
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        text_attention_mask = encoded["attention_mask"].to(device)
        metrics = tokenizer_anchor_metrics(
            tokenizer=tokenizer,
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            max_tokens=int(max_tokens),
            anchor_text="Representation:",
            require_anchor=True,
            require_under_max_length=bool(require_under_max_length),
            context="student anchor text",
        )
    text_embeds = llm.get_input_embeddings()(input_ids)
    soft_prompts = soft_prompts.to(device=device, dtype=text_embeds.dtype)
    metrics["text_token_embedding_norm"] = masked_token_norm(text_embeds, text_attention_mask)
    metrics["soft_prompt_token_norm"] = mean_token_norm(soft_prompts)
    inputs_embeds = torch.cat([soft_prompts, text_embeds], dim=1)
    soft_attention = torch.ones(
        (input_ids.shape[0], soft_prompts.shape[1]),
        dtype=text_attention_mask.dtype,
        device=device,
    )
    attention_mask = torch.cat([soft_attention, text_attention_mask], dim=1)
    hidden = forward_teacher_readout_hidden(
        llm,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        teacher_layer=int(teacher_layer),
        prefix_tokens=int(soft_prompts.shape[1]),
    )
    metrics["hidden_norm"] = mean_token_norm(hidden.unsqueeze(1))
    metrics["hidden_pairwise_cosine"] = off_diagonal_cosine_mean(hidden)
    return HiddenBatch(hidden=hidden, metrics=metrics)


def normalize_patch_batch(
    patches: torch.Tensor,
    input_size: Sequence[int],
    normalization_cfg: Mapping[str, Any],
    resize_to_input: bool,
) -> torch.Tensor:
    batch = patches.to(dtype=torch.float32)
    if bool(resize_to_input):
        batch = resize_chw_batch(batch, input_size)
    config = dict(normalization_cfg or {})
    clip_min = config.get("clip_min")
    clip_max = config.get("clip_max")
    if clip_min is not None or clip_max is not None:
        batch = torch.clamp(batch, min=clip_min, max=clip_max)

    mode = str(config.get("mode", "none")).lower()
    if mode == "none":
        return batch
    if mode not in {"minmax", "zscore"}:
        raise ValueError(f"Unsupported normalization mode: {mode}")
    scope = str(config.get("scope", "global")).lower()
    if scope == "global":
        reduction_dims = tuple(range(1, batch.ndim))
    elif scope == "channel":
        reduction_dims = tuple(range(2, batch.ndim))
    else:
        raise ValueError(f"Unsupported normalization scope: {scope}")

    if mode == "minmax":
        minimum = batch.amin(dim=reduction_dims, keepdim=True)
        maximum = batch.amax(dim=reduction_dims, keepdim=True)
        return (batch - minimum) / (maximum - minimum + 1.0e-6)
    if mode == "zscore":
        mean = batch.mean(dim=reduction_dims, keepdim=True)
        std = batch.std(dim=reduction_dims, keepdim=True, unbiased=False)
        return (batch - mean) / (std + 1.0e-6)
    raise AssertionError(f"Unhandled normalization mode: {mode}")


def build_teacher_texts_for_batch(
    batch: Mapping[str, Any],
    normalized_patches: torch.Tensor,
    args: argparse.Namespace,
) -> list[str]:
    source = str(args.teacher_text_source).lower()
    if str(args.alignment_text_layout) == "values_shared_suffix":
        patches = teacher_source_patches(batch, normalized_patches, source)
        return serialize_tensor_value_batch(patches, int(args.text_decimal_places))
    if source == "raw":
        return [str(text) for text in batch["texts"]]
    if source == "normalized":
        return serialize_patch_batch(
            records=batch["records"],
            patches=normalized_patches,
            decimal_places=int(args.text_decimal_places),
            prompt_template=str(args.text_prompt_template),
        )
    raise ValueError(f"Unsupported teacher_text_source: {args.teacher_text_source}")


def teacher_source_patches(
    batch: Mapping[str, Any],
    normalized_patches: torch.Tensor,
    source: str,
) -> torch.Tensor:
    normalized_source = str(source).lower()
    if normalized_source == "raw":
        return batch["patch"]
    if normalized_source == "normalized":
        return normalized_patches
    raise ValueError(f"Unsupported teacher_text_source: {source}")


def probe_targets_for_batch(
    batch: Mapping[str, Any],
    normalized_patches: torch.Tensor,
    args: argparse.Namespace,
    alignment_anchor: AlignmentAnchor | None,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
    if alignment_anchor is None or alignment_anchor.mode != "probe":
        return None, None
    source_patches = teacher_source_patches(batch, normalized_patches, str(args.teacher_text_source))
    return probe_targets_from_patches(
        alignment_anchor,
        source_patches,
        int(args.text_decimal_places),
    )


@torch.no_grad()
def target_geometry_metrics(embeddings: torch.Tensor, target_values: torch.Tensor | None) -> dict[str, float]:
    if target_values is None or int(embeddings.shape[0]) < 2:
        return {}
    values = target_values.detach().float().to(embeddings.device).flatten()
    normalized = F.normalize(embeddings.detach().float(), dim=-1)
    similarity = normalized @ normalized.T
    distance = (values[:, None] - values[None, :]).abs()
    upper = torch.triu(torch.ones_like(similarity, dtype=torch.bool), diagonal=1)
    pair_similarity = similarity[upper]
    negative_distance = -distance[upper]
    similarity_centered = pair_similarity - pair_similarity.mean()
    distance_centered = negative_distance - negative_distance.mean()
    denominator = similarity_centered.square().sum().sqrt() * distance_centered.square().sum().sqrt()
    correlation = (
        similarity_centered.mul(distance_centered).sum() / denominator.clamp_min(1.0e-12)
        if pair_similarity.numel() > 1
        else torch.zeros((), device=similarity.device)
    )
    nearest_logits = similarity.masked_fill(torch.eye(similarity.shape[0], device=similarity.device, dtype=torch.bool), -1.0e9)
    nearest_indices = nearest_logits.argmax(dim=1)
    return {
        "target_abs_mean": float(values.abs().mean().cpu().item()),
        "target_std": float(values.std(unbiased=False).cpu().item()),
        "hidden_similarity_vs_negative_target_distance_pearson": float(correlation.cpu().item()),
        "nearest_hidden_target_abs_error": float(
            (values - values[nearest_indices]).abs().mean().cpu().item()
        ),
    }


def duplicate_text_fraction(texts: Sequence[str]) -> float:
    if not texts:
        return 0.0
    unique_count = len(set(str(text) for text in texts))
    return float((len(texts) - unique_count) / len(texts))


def alignment_anchors_from_args(
    tokenizer: Any,
    args: argparse.Namespace,
    *,
    evaluation: bool,
) -> list[AlignmentAnchor]:
    mode = str(args.alignment_anchor_mode)
    if mode == "probe":
        channel_count = 1 if str(args.field_sampling_mode).lower() == "single" else len(parse_csv(args.fields))
        probe_count = int(args.evaluation_probe_count) if evaluation else 1
        return [
            build_numeric_probe_anchor(
                tokenizer=tokenizer,
                patch_size=int(args.patch_size),
                channel_count=int(channel_count),
                families=args.probe_families,
                region_size=int(args.probe_region_size),
                probe_index=index,
                seed=int(args.seed) + (100_000 if evaluation else 0),
                max_anchor_tokens=int(args.max_shared_suffix_tokens),
            )
            for index in range(probe_count)
        ]
    return [
        build_static_alignment_anchor(
            tokenizer=tokenizer,
            mode=mode,
            representation_suffix=str(args.representation_suffix),
            max_anchor_tokens=int(args.max_shared_suffix_tokens),
        )
    ]


def probe_contract_anchors(tokenizer: Any, args: argparse.Namespace) -> list[AlignmentAnchor]:
    if str(args.alignment_anchor_mode) != "probe":
        return []
    families = [family.lower() for family in parse_csv(args.probe_families)]
    channel_count = 1 if str(args.field_sampling_mode).lower() == "single" else len(parse_csv(args.fields))
    anchors: list[AlignmentAnchor] = []
    for family_index, family in enumerate(families):
        template_count = int(PROBE_TEMPLATE_COUNTS[family])
        for template_index in range(template_count):
            anchor = build_numeric_probe_anchor(
                tokenizer=tokenizer,
                patch_size=int(args.patch_size),
                channel_count=int(channel_count),
                families=[family],
                region_size=int(args.probe_region_size),
                # Keep coordinates fixed across templates so preflight isolates wording differences.
                probe_index=family_index,
                seed=int(args.seed) + 200_000,
                max_anchor_tokens=int(args.max_shared_suffix_tokens),
                template_index=template_index,
            )
            anchors.append(anchor)
    expected_count = sum(int(PROBE_TEMPLATE_COUNTS[family]) for family in families)
    observed_pairs = {(anchor.probe_family, anchor.probe_template_index) for anchor in anchors}
    if len(anchors) != expected_count or len(observed_pairs) != expected_count:
        raise ValueError(
            "Probe contract preflight did not cover every family/template pair: "
            f"expected={expected_count}, anchors={len(anchors)}, unique_pairs={len(observed_pairs)}."
        )
    for family in families:
        family_stems = {anchor.token_ids for anchor in anchors if anchor.probe_family == family}
        expected_family_count = int(PROBE_TEMPLATE_COUNTS[family])
        if len(family_stems) != expected_family_count:
            raise ValueError(
                f"Probe templates for {family} collapsed to duplicate token sequences after tokenization: "
                f"expected={expected_family_count}, unique_token_sequences={len(family_stems)}."
            )
    return anchors


def train_compressor_during_alignment(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "alignment_train_patch_ae", args.train_patch_ae))


def set_frozen_llm_student_mode(llm: nn.Module, gradient_checkpointing: bool) -> None:
    if not bool(gradient_checkpointing):
        llm.eval()
        return
    # HF decoder checkpointing is active only in train mode. Keep stochastic layers
    # disabled so the frozen teacher/student mapping remains deterministic.
    llm.train()
    for module in llm.modules():
        if isinstance(module, nn.Dropout):
            module.eval()


def train_one_epoch(
    *,
    compressor: nn.Module,
    adapter: TensorPatchAlignmentAdapter,
    projector: AlignmentFeatureTransform | None,
    llm: nn.Module,
    tokenizer: Any,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    compressor_input_size: Sequence[int],
    normalization_cfg: Mapping[str, Any],
    epoch: int,
) -> dict[str, float]:
    sampler = getattr(loader, "sampler", None)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(int(epoch))
    adapter.train()
    if projector is not None:
        projector.train()
    set_frozen_llm_student_mode(llm, bool(args.llm_gradient_checkpointing))
    train_compressor = train_compressor_during_alignment(args)
    if train_compressor:
        compressor.train()
    else:
        compressor.eval()
    total_loss = 0.0
    total_contrastive = 0.0
    total_reconstruction = 0.0
    total_i2t = 0.0
    total_t2i = 0.0
    total_records = 0
    metric_totals: dict[str, float] = {}
    training_anchors = alignment_anchors_from_args(tokenizer, args, evaluation=False)
    progress = tqdm(loader, desc=f"train align epoch {epoch}", leave=False, disable=not is_main_process())
    for step, batch in enumerate(progress, start=1):
        if str(args.alignment_anchor_mode) == "probe":
            global_batch_index = (int(epoch) - 1) * len(loader) + int(step) - 1
            channel_count = int(batch["patch"].shape[1])
            alignment_anchor = build_numeric_probe_anchor(
                tokenizer=tokenizer,
                patch_size=int(args.patch_size),
                channel_count=channel_count,
                families=args.probe_families,
                region_size=int(args.probe_region_size),
                probe_index=global_batch_index,
                seed=int(args.seed),
                max_anchor_tokens=int(args.max_shared_suffix_tokens),
            )
        else:
            alignment_anchor = training_anchors[0]
        normalized_patches = normalize_patch_batch(
            batch["patch"],
            compressor_input_size,
            normalization_cfg,
            bool(args.resize_patch_to_compressor_input),
        )
        patches = normalized_patches.to(device)
        texts = build_teacher_texts_for_batch(batch, normalized_patches, args)
        probe_target_values, probe_target_ids = probe_targets_for_batch(
            batch,
            normalized_patches,
            args,
            alignment_anchor,
        )
        if probe_target_ids is not None:
            probe_target_ids = probe_target_ids.to(device)
        teacher_duplicate_fraction = duplicate_text_fraction(texts)
        llm_was_training = llm.training
        llm.eval()
        with torch.no_grad():
            teacher_output = text_teacher_hidden(
                llm,
                tokenizer,
                texts,
                device,
                int(args.max_text_tokens),
                int(args.teacher_layer),
                bool(args.fail_on_text_anchor_missing) and str(args.text_prompt_template) != "plain",
                bool(args.fail_on_text_max_length_hit)
                and (
                    str(args.alignment_text_layout) == "values_shared_suffix"
                    or str(args.text_prompt_template) != "plain"
                ),
                text_layout=str(args.alignment_text_layout),
                shared_suffix=str(args.shared_suffix),
                max_shared_suffix_tokens=int(args.max_shared_suffix_tokens),
                alignment_anchor=alignment_anchor,
            )
        if llm_was_training:
            set_frozen_llm_student_mode(llm, bool(args.llm_gradient_checkpointing))
        if train_compressor:
            latent = compressor.encode(patches)["latent_map"]
        else:
            with torch.no_grad():
                latent = compressor.encode(patches)["latent_map"]
        soft_prompts = adapter.forward_soft_prompts(latent)
        if soft_prompts.requires_grad:
            soft_prompts.retain_grad()
        student_output = tensor_student_hidden(
            llm,
            tokenizer,
            soft_prompts,
            batch["records"],
            device,
            int(args.max_text_tokens),
            int(args.teacher_layer),
            int(normalized_patches.shape[-1]),
            bool(args.fail_on_text_max_length_hit)
            and (
                str(args.alignment_text_layout) == "values_shared_suffix"
                or str(args.text_prompt_template) != "plain"
            ),
            text_layout=str(args.alignment_text_layout),
            shared_suffix=str(args.shared_suffix),
            max_shared_suffix_tokens=int(args.max_shared_suffix_tokens),
            alignment_anchor=alignment_anchor,
        )
        student_hidden = student_output.hidden
        teacher_hidden = teacher_output.hidden.to(dtype=student_hidden.dtype)
        student_features, teacher_features = apply_alignment_feature_transform(
            projector,
            student_hidden,
            teacher_hidden,
        )
        tensor_embedding, text_embedding = normalize_alignment_embeddings(
            student_features,
            teacher_features,
        )
        contrastive, contrastive_metrics = distributed_symmetric_contrastive_loss(
            tensor_embedding,
            text_embedding,
            float(args.temperature),
            probe_target_ids,
            i2t_weight=float(args.contrastive_i2t_weight),
            t2i_weight=float(args.contrastive_t2i_weight),
        )
        if float(args.centered_contrastive_loss_weight) > 0.0:
            centered_tensor_embedding, centered_text_embedding = centered_alignment_embeddings(
                student_features,
                teacher_features,
                distributed_batch=True,
            )
            centered_contrastive, centered_contrastive_metrics = distributed_symmetric_contrastive_loss(
                centered_tensor_embedding,
                centered_text_embedding,
                float(args.temperature),
                probe_target_ids,
                i2t_weight=float(args.contrastive_i2t_weight),
                t2i_weight=float(args.contrastive_t2i_weight),
            )
        else:
            with torch.no_grad():
                centered_tensor_embedding, centered_text_embedding = centered_alignment_embeddings(
                    student_features,
                    teacher_features,
                    distributed_batch=True,
                )
                centered_contrastive, centered_contrastive_metrics = distributed_symmetric_contrastive_loss(
                    centered_tensor_embedding,
                    centered_text_embedding,
                    float(args.temperature),
                    probe_target_ids,
                    i2t_weight=float(args.contrastive_i2t_weight),
                    t2i_weight=float(args.contrastive_t2i_weight),
                )
        if float(args.native_centered_contrastive_loss_weight) > 0.0:
            native_centered_student, native_centered_teacher = centered_alignment_embeddings(
                student_hidden,
                teacher_hidden,
                distributed_batch=True,
            )
            native_centered_contrastive, native_centered_metrics = distributed_symmetric_contrastive_loss(
                native_centered_student,
                native_centered_teacher,
                float(args.temperature),
                probe_target_ids,
                i2t_weight=float(args.contrastive_i2t_weight),
                t2i_weight=float(args.contrastive_t2i_weight),
            )
        else:
            native_centered_contrastive = student_hidden.new_zeros(())
            native_centered_metrics = {}
        if float(args.mean_alignment_loss_weight) > 0.0:
            transformed_mean_loss, transformed_mean_metrics = branch_mean_alignment_loss(
                student_features,
                teacher_features,
                distributed_batch=True,
            )
            native_mean_loss, native_mean_metrics = branch_mean_alignment_loss(
                student_hidden,
                teacher_hidden,
                distributed_batch=True,
            )
            mean_alignment_loss = 0.5 * (transformed_mean_loss + native_mean_loss)
        else:
            with torch.no_grad():
                transformed_mean_loss, transformed_mean_metrics = branch_mean_alignment_loss(
                    student_features,
                    teacher_features,
                    distributed_batch=True,
                )
                native_mean_loss, native_mean_metrics = branch_mean_alignment_loss(
                    student_hidden,
                    teacher_hidden,
                    distributed_batch=True,
                )
                mean_alignment_loss = 0.5 * (transformed_mean_loss + native_mean_loss)
        reconstruction, reconstruction_metrics = reconstruction_loss_with_diagnostics(compressor, latent, patches)
        reconstruction_weight = float(args.reconstruction_loss_weight) if train_compressor else 0.0
        loss = (
            float(args.contrastive_loss_weight) * contrastive
            + float(args.centered_contrastive_loss_weight) * centered_contrastive
            + float(args.native_centered_contrastive_loss_weight) * native_centered_contrastive
            + float(args.mean_alignment_loss_weight) * mean_alignment_loss
            + reconstruction_weight * reconstruction
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if soft_prompts.grad is not None:
            token_gradient_norms = soft_prompts.grad.detach().float().norm(dim=-1)
            soft_prompt_gradient_norm = float(token_gradient_norms.mean().cpu().item())
            relative_threshold = token_gradient_norms.amax(dim=1, keepdim=True) * 1.0e-3
            soft_prompt_active_token_fraction = float(
                token_gradient_norms.gt(relative_threshold).float().mean().cpu().item()
            )
            gradient_probabilities = token_gradient_norms / token_gradient_norms.sum(
                dim=1,
                keepdim=True,
            ).clamp_min(1.0e-20)
            gradient_entropy = -(
                gradient_probabilities * gradient_probabilities.clamp_min(1.0e-20).log()
            ).sum(dim=1)
            entropy_denominator = math.log(max(int(token_gradient_norms.shape[1]), 2))
            soft_prompt_gradient_entropy = float(
                (gradient_entropy / entropy_denominator).mean().cpu().item()
            )
            soft_prompt_gradient_min = float(token_gradient_norms.amin(dim=1).mean().cpu().item())
            soft_prompt_gradient_max = float(token_gradient_norms.amax(dim=1).mean().cpu().item())
        else:
            soft_prompt_gradient_norm = 0.0
            soft_prompt_active_token_fraction = 0.0
            soft_prompt_gradient_entropy = 0.0
            soft_prompt_gradient_min = 0.0
            soft_prompt_gradient_max = 0.0
        synchronize_gradients([compressor if train_compressor else None, adapter, projector])
        if float(args.grad_clip_norm) > 0:
            torch.nn.utils.clip_grad_norm_(
                [parameter for group in optimizer.param_groups for parameter in group["params"]],
                float(args.grad_clip_norm),
            )
        optimizer.step()

        batch_size = int(patches.shape[0])
        total_loss += float(loss.detach().cpu().item()) * batch_size
        total_contrastive += float(contrastive.detach().cpu().item()) * batch_size
        total_reconstruction += float(reconstruction.detach().cpu().item()) * batch_size
        total_i2t += float(contrastive_metrics["i2t_accuracy"]) * batch_size
        total_t2i += float(contrastive_metrics["t2i_accuracy"]) * batch_size
        add_weighted_metrics(metric_totals, reconstruction_metrics, batch_size, "reconstruction_")
        add_weighted_metrics(metric_totals, contrastive_metrics, batch_size, "contrastive_")
        add_weighted_metrics(
            metric_totals,
            {
                "contrastive_loss": float(centered_contrastive.detach().cpu().item()),
                **centered_contrastive_metrics,
            },
            batch_size,
            "centered_",
        )
        add_weighted_metrics(
            metric_totals,
            {
                "contrastive_loss": float(native_centered_contrastive.detach().cpu().item()),
                **native_centered_metrics,
            },
            batch_size,
            "native_centered_",
        )
        add_weighted_metrics(
            metric_totals,
            {
                "loss": float(mean_alignment_loss.detach().cpu().item()),
                **prefixed_metrics(transformed_mean_metrics, "transformed"),
                **prefixed_metrics(native_mean_metrics, "native"),
            },
            batch_size,
            "mean_alignment_",
        )
        add_weighted_metrics(metric_totals, teacher_output.metrics, batch_size, "teacher_")
        add_weighted_metrics(
            metric_totals,
            {"duplicate_text_fraction": teacher_duplicate_fraction},
            batch_size,
            "teacher_",
        )
        add_weighted_metrics(metric_totals, student_output.metrics, batch_size, "student_")
        add_weighted_metrics(
            metric_totals,
            target_geometry_metrics(teacher_hidden, probe_target_values),
            batch_size,
            "teacher_probe_",
        )
        add_weighted_metrics(
            metric_totals,
            target_geometry_metrics(student_hidden, probe_target_values),
            batch_size,
            "student_probe_",
        )
        add_weighted_metrics(
            metric_totals,
            {
                "positive_cosine": float(
                    F.cosine_similarity(
                        tensor_embedding.detach(),
                        text_embedding.detach(),
                        dim=-1,
                    ).mean().cpu().item()
                ),
                "student_embedding_pairwise_cosine": off_diagonal_cosine_mean(tensor_embedding),
                "teacher_embedding_pairwise_cosine": off_diagonal_cosine_mean(text_embedding),
                "centered_student_embedding_pairwise_cosine": off_diagonal_cosine_mean(centered_tensor_embedding),
                "centered_teacher_embedding_pairwise_cosine": off_diagonal_cosine_mean(centered_text_embedding),
                "hidden_positive_cosine": float(
                    F.cosine_similarity(student_hidden.detach().float(), teacher_hidden.detach().float(), dim=-1)
                    .mean()
                    .cpu()
                    .item()
                ),
                "student_hidden_pairwise_cosine": off_diagonal_cosine_mean(student_hidden),
                "teacher_hidden_pairwise_cosine": off_diagonal_cosine_mean(teacher_hidden),
                "soft_prompt_gradient_norm": soft_prompt_gradient_norm,
                "soft_prompt_active_token_fraction": soft_prompt_active_token_fraction,
                "soft_prompt_gradient_entropy": soft_prompt_gradient_entropy,
                "soft_prompt_gradient_min": soft_prompt_gradient_min,
                "soft_prompt_gradient_max": soft_prompt_gradient_max,
            },
            batch_size,
            "alignment_",
        )
        total_records += batch_size
        if int(args.log_interval) > 0 and step % int(args.log_interval) == 0:
            progress.set_postfix(
                loss=f"{total_loss / max(1, total_records):.4f}",
                i2t=f"{total_i2t / max(1, total_records):.3f}",
                t2i=f"{total_t2i / max(1, total_records):.3f}",
                anchor=alignment_anchor.name,
            )
    metrics = {
        "loss": total_loss / max(1, total_records),
        "contrastive_loss": total_contrastive / max(1, total_records),
        "reconstruction_loss": total_reconstruction / max(1, total_records),
        "i2t_accuracy": total_i2t / max(1, total_records),
        "t2i_accuracy": total_t2i / max(1, total_records),
    }
    metrics.update(averaged_metrics(metric_totals, total_records))
    metrics.update(alignment_adapter_parameter_metrics(adapter))
    for key in (
        "candidate_count",
        "i2t_loss",
        "t2i_loss",
        "strict_i2t_loss",
        "strict_t2i_loss",
        "strict_contrastive_loss",
        "i2t_candidate_coverage",
        "t2i_candidate_coverage",
        "i2t_max_candidate_hit_fraction",
        "t2i_max_candidate_hit_fraction",
        "i2t_candidate_hit_entropy",
        "t2i_candidate_hit_entropy",
        "semantic_i2t_accuracy",
        "semantic_t2i_accuracy",
        "semantic_collision_fraction",
        "semantic_target_unique_fraction",
        "valid_negative_count",
    ):
        prefixed_key = f"contrastive_{key}"
        if prefixed_key in metrics:
            metrics[key] = metrics[prefixed_key]
    return average_metrics_across_processes(metrics)


def pretrain_patch_encoder_one_epoch(
    *,
    compressor: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    compressor_input_size: Sequence[int],
    normalization_cfg: Mapping[str, Any],
    epoch: int,
    wandb_logger: WandbLogger | None = None,
    global_step: int = 0,
) -> tuple[dict[str, float], int]:
    sampler = getattr(loader, "sampler", None)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(int(epoch))
    compressor.train()
    total_loss = 0.0
    total_records = 0
    metric_totals: dict[str, float] = {}
    progress = tqdm(loader, desc=f"pretrain patch AE epoch {epoch}", leave=False, disable=not is_main_process())
    for step, batch in enumerate(progress, start=1):
        patches = normalize_patch_batch(
            batch["patch"],
            compressor_input_size,
            normalization_cfg,
            bool(args.resize_patch_to_compressor_input),
        ).to(device)
        latent = compressor.encode(patches)["latent_map"]
        loss, reconstruction_metrics = reconstruction_loss_with_diagnostics(compressor, latent, patches)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        synchronize_gradients([compressor])
        if float(args.grad_clip_norm) > 0:
            torch.nn.utils.clip_grad_norm_(
                [parameter for group in optimizer.param_groups for parameter in group["params"]],
                float(args.grad_clip_norm),
            )
        optimizer.step()
        batch_size = int(patches.shape[0])
        total_loss += float(loss.detach().cpu().item()) * batch_size
        total_records += batch_size
        add_weighted_metrics(metric_totals, reconstruction_metrics, batch_size, "")
        average_loss = total_loss / max(1, total_records)
        progress.set_postfix(
            recon=f"{average_loss:.4f}",
            rel=f"{metric_totals.get('relative_rmse_to_target_std', 0.0) / max(1, total_records):.3f}",
        )
        global_step += 1
        if wandb_logger is not None and int(args.log_interval) > 0 and step % int(args.log_interval) == 0:
            wandb_logger.log(
                {
                    "patch_ae_pretrain_step/reconstruction_loss": average_loss,
                    "patch_ae_pretrain_step/current_reconstruction_loss": float(loss.detach().cpu().item()),
                    "patch_ae_pretrain_step/relative_rmse_to_target_std": float(
                        reconstruction_metrics["relative_rmse_to_target_std"]
                    ),
                    "patch_ae_pretrain_step/target_std": float(reconstruction_metrics["target_std"]),
                    "patch_ae_pretrain_step/epoch": float(epoch),
                    "patch_ae_pretrain_step/epoch_step": float(step),
                    "patch_ae_pretrain_step/lr": float(optimizer.param_groups[0]["lr"]),
                },
                step=global_step,
            )
    metrics = {"reconstruction_loss": total_loss / max(1, total_records)}
    metrics.update(averaged_metrics(metric_totals, total_records))
    return average_metrics_across_processes(metrics), global_step


@torch.no_grad()
def evaluate_patch_encoder_reconstruction(
    *,
    compressor: nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    compressor_input_size: Sequence[int],
    normalization_cfg: Mapping[str, Any],
) -> dict[str, float]:
    compressor.eval()
    total_loss = 0.0
    total_records = 0
    metric_totals: dict[str, float] = {}
    for batch in tqdm(loader, desc="eval patch AE reconstruction", leave=False, disable=not is_main_process()):
        patches = normalize_patch_batch(
            batch["patch"],
            compressor_input_size,
            normalization_cfg,
            bool(args.resize_patch_to_compressor_input),
        ).to(device)
        latent = compressor.encode(patches)["latent_map"]
        loss, reconstruction_metrics = reconstruction_loss_with_diagnostics(compressor, latent, patches)
        batch_size = int(patches.shape[0])
        total_loss += float(loss.detach().cpu().item()) * batch_size
        total_records += batch_size
        add_weighted_metrics(metric_totals, reconstruction_metrics, batch_size, "")
    metrics = {"reconstruction_loss": total_loss / max(1, total_records)}
    metrics.update(averaged_metrics(metric_totals, total_records))
    return weighted_average_metrics_across_processes(metrics, total_records)


@torch.no_grad()
def global_retrieval_metrics_from_features(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    probe_target_ids: torch.Tensor | None,
    args: argparse.Namespace,
) -> dict[str, float]:
    result: dict[str, float] = {}
    global_tensor, global_text = normalize_alignment_embeddings(student_features, teacher_features)
    global_centered_tensor, global_centered_text = centered_alignment_embeddings(
        student_features,
        teacher_features,
    )
    retrieval_kwargs = {
        "temperature": float(args.temperature),
        "chunk_size": int(args.global_retrieval_chunk_size),
        "semantic_target_ids": probe_target_ids,
        "i2t_weight": float(args.contrastive_i2t_weight),
        "t2i_weight": float(args.contrastive_t2i_weight),
    }
    result.update(
        {
            f"global_{key}": value
            for key, value in full_retrieval_accuracy(
                global_tensor,
                global_text,
                **retrieval_kwargs,
            ).items()
        }
    )
    result["global_candidate_count"] = float(int(student_features.shape[0]))
    result.update(
        {
            f"global_centered_{key}": value
            for key, value in full_retrieval_accuracy(
                global_centered_tensor,
                global_centered_text,
                **retrieval_kwargs,
            ).items()
        }
    )
    global_hidden, global_teacher_hidden = normalize_alignment_embeddings(
        student_hidden,
        teacher_hidden,
    )
    result.update(
        {
            f"global_hidden_uncentered_{key}": value
            for key, value in full_retrieval_accuracy(
                global_hidden,
                global_teacher_hidden,
                **retrieval_kwargs,
            ).items()
        }
    )
    global_hidden_centered, global_teacher_hidden_centered = centered_alignment_embeddings(
        student_hidden,
        teacher_hidden,
    )
    result.update(
        {
            f"global_hidden_centered_{key}": value
            for key, value in full_retrieval_accuracy(
                global_hidden_centered,
                global_teacher_hidden_centered,
                **retrieval_kwargs,
            ).items()
        }
    )
    transformed_mean_loss, transformed_mean_metrics = branch_mean_alignment_loss(
        student_features,
        teacher_features,
    )
    native_mean_loss, native_mean_metrics = branch_mean_alignment_loss(
        student_hidden,
        teacher_hidden,
    )
    result["global_mean_alignment_loss"] = float(
        (0.5 * (transformed_mean_loss + native_mean_loss)).cpu().item()
    )
    result.update(prefixed_metrics(transformed_mean_metrics, "global_mean_alignment_transformed"))
    result.update(prefixed_metrics(native_mean_metrics, "global_mean_alignment_native"))
    return result


@torch.no_grad()
def evaluate(
    *,
    compressor: nn.Module,
    adapter: TensorPatchAlignmentAdapter,
    projector: AlignmentFeatureTransform | None,
    llm: nn.Module,
    tokenizer: Any,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    compressor_input_size: Sequence[int],
    normalization_cfg: Mapping[str, Any],
    alignment_anchor: AlignmentAnchor | None = None,
) -> dict[str, float]:
    compressor.eval()
    adapter.eval()
    llm.eval()
    if projector is not None:
        projector.eval()
    adapter.capture_spatial_path_metrics = adapter.adapter_type == "spatial_transformer"
    total_loss = 0.0
    total_contrastive = 0.0
    total_reconstruction = 0.0
    total_i2t = 0.0
    total_t2i = 0.0
    total_records = 0
    metric_totals: dict[str, float] = {}
    collected_student_features: list[torch.Tensor] = []
    collected_teacher_features: list[torch.Tensor] = []
    collected_student_hidden: list[torch.Tensor] = []
    collected_teacher_hidden: list[torch.Tensor] = []
    collected_probe_target_ids: list[torch.Tensor] = []
    distributed_eval = distributed_is_initialized()
    train_compressor = train_compressor_during_alignment(args)
    if alignment_anchor is None and str(args.alignment_text_layout) == "values_shared_suffix":
        alignment_anchor = alignment_anchors_from_args(tokenizer, args, evaluation=True)[0]
    for batch in tqdm(loader, desc="eval align", leave=False, disable=not is_main_process()):
        normalized_patches = normalize_patch_batch(
            batch["patch"],
            compressor_input_size,
            normalization_cfg,
            bool(args.resize_patch_to_compressor_input),
        )
        patches = normalized_patches.to(device)
        teacher_texts = build_teacher_texts_for_batch(batch, normalized_patches, args)
        probe_target_values, probe_target_ids = probe_targets_for_batch(
            batch,
            normalized_patches,
            args,
            alignment_anchor,
        )
        if probe_target_ids is not None:
            probe_target_ids = probe_target_ids.to(device)
        teacher_output = text_teacher_hidden(
            llm,
            tokenizer,
            teacher_texts,
            device,
            int(args.max_text_tokens),
            int(args.teacher_layer),
            bool(args.fail_on_text_anchor_missing) and str(args.text_prompt_template) != "plain",
            bool(args.fail_on_text_max_length_hit)
            and (
                str(args.alignment_text_layout) == "values_shared_suffix"
                or str(args.text_prompt_template) != "plain"
            ),
            text_layout=str(args.alignment_text_layout),
            shared_suffix=str(args.shared_suffix),
            max_shared_suffix_tokens=int(args.max_shared_suffix_tokens),
            alignment_anchor=alignment_anchor,
        )
        latent = compressor.encode(patches)["latent_map"]
        soft_prompts = adapter.forward_soft_prompts(latent)
        add_weighted_metrics(
            metric_totals,
            alignment_adapter_path_metrics(adapter),
            int(patches.shape[0]),
            "adapter_",
        )
        student_output = tensor_student_hidden(
            llm,
            tokenizer,
            soft_prompts,
            batch["records"],
            device,
            int(args.max_text_tokens),
            int(args.teacher_layer),
            int(normalized_patches.shape[-1]),
            bool(args.fail_on_text_max_length_hit),
            text_layout=str(args.alignment_text_layout),
            shared_suffix=str(args.shared_suffix),
            max_shared_suffix_tokens=int(args.max_shared_suffix_tokens),
            alignment_anchor=alignment_anchor,
        )
        student_hidden = student_output.hidden
        teacher_hidden = teacher_output.hidden.to(dtype=student_hidden.dtype)
        student_features, teacher_features = apply_alignment_feature_transform(
            projector,
            student_hidden,
            teacher_hidden,
        )
        tensor_embedding, text_embedding = normalize_alignment_embeddings(
            student_features,
            teacher_features,
        )
        centered_tensor_embedding, centered_text_embedding = centered_alignment_embeddings(
            student_features,
            teacher_features,
            distributed_batch=distributed_eval,
        )
        contrastive_loss_fn = (
            distributed_symmetric_contrastive_loss
            if distributed_eval
            else symmetric_contrastive_loss
        )
        contrastive, contrastive_metrics = contrastive_loss_fn(
            tensor_embedding,
            text_embedding,
            float(args.temperature),
            probe_target_ids,
            i2t_weight=float(args.contrastive_i2t_weight),
            t2i_weight=float(args.contrastive_t2i_weight),
        )
        centered_contrastive, centered_contrastive_metrics = contrastive_loss_fn(
            centered_tensor_embedding,
            centered_text_embedding,
            float(args.temperature),
            probe_target_ids,
            i2t_weight=float(args.contrastive_i2t_weight),
            t2i_weight=float(args.contrastive_t2i_weight),
        )
        native_centered_student, native_centered_teacher = centered_alignment_embeddings(
            student_hidden,
            teacher_hidden,
            distributed_batch=distributed_eval,
        )
        native_centered_contrastive, native_centered_metrics = contrastive_loss_fn(
            native_centered_student,
            native_centered_teacher,
            float(args.temperature),
            probe_target_ids,
            i2t_weight=float(args.contrastive_i2t_weight),
            t2i_weight=float(args.contrastive_t2i_weight),
        )
        transformed_mean_loss, transformed_mean_metrics = branch_mean_alignment_loss(
            student_features,
            teacher_features,
            distributed_batch=distributed_eval,
        )
        native_mean_loss, native_mean_metrics = branch_mean_alignment_loss(
            student_hidden,
            teacher_hidden,
            distributed_batch=distributed_eval,
        )
        mean_alignment_loss = 0.5 * (transformed_mean_loss + native_mean_loss)
        reconstruction, reconstruction_metrics = reconstruction_loss_with_diagnostics(compressor, latent, patches)
        reconstruction_weight = float(args.reconstruction_loss_weight) if train_compressor else 0.0
        loss = (
            float(args.contrastive_loss_weight) * contrastive
            + float(args.centered_contrastive_loss_weight) * centered_contrastive
            + float(args.native_centered_contrastive_loss_weight) * native_centered_contrastive
            + float(args.mean_alignment_loss_weight) * mean_alignment_loss
            + reconstruction_weight * reconstruction
        )
        batch_size = int(patches.shape[0])
        total_loss += float(loss.detach().cpu().item()) * batch_size
        total_contrastive += float(contrastive.detach().cpu().item()) * batch_size
        total_reconstruction += float(reconstruction.detach().cpu().item()) * batch_size
        total_i2t += float(contrastive_metrics["i2t_accuracy"]) * batch_size
        total_t2i += float(contrastive_metrics["t2i_accuracy"]) * batch_size
        add_weighted_metrics(metric_totals, reconstruction_metrics, batch_size, "reconstruction_")
        add_weighted_metrics(metric_totals, contrastive_metrics, batch_size, "contrastive_")
        add_weighted_metrics(
            metric_totals,
            {
                "contrastive_loss": float(centered_contrastive.detach().cpu().item()),
                **centered_contrastive_metrics,
            },
            batch_size,
            "centered_",
        )
        add_weighted_metrics(
            metric_totals,
            {
                "contrastive_loss": float(native_centered_contrastive.detach().cpu().item()),
                **native_centered_metrics,
            },
            batch_size,
            "native_centered_",
        )
        add_weighted_metrics(
            metric_totals,
            {
                "loss": float(mean_alignment_loss.detach().cpu().item()),
                **prefixed_metrics(transformed_mean_metrics, "transformed"),
                **prefixed_metrics(native_mean_metrics, "native"),
            },
            batch_size,
            "mean_alignment_",
        )
        add_weighted_metrics(metric_totals, teacher_output.metrics, batch_size, "teacher_")
        add_weighted_metrics(
            metric_totals,
            {"duplicate_text_fraction": duplicate_text_fraction(teacher_texts)},
            batch_size,
            "teacher_",
        )
        add_weighted_metrics(metric_totals, student_output.metrics, batch_size, "student_")
        add_weighted_metrics(
            metric_totals,
            target_geometry_metrics(teacher_hidden, probe_target_values),
            batch_size,
            "teacher_probe_",
        )
        add_weighted_metrics(
            metric_totals,
            target_geometry_metrics(student_hidden, probe_target_values),
            batch_size,
            "student_probe_",
        )
        add_weighted_metrics(
            metric_totals,
            {
                "positive_cosine": float(
                    F.cosine_similarity(
                        tensor_embedding,
                        text_embedding,
                        dim=-1,
                    ).mean().cpu().item()
                ),
                "student_embedding_pairwise_cosine": off_diagonal_cosine_mean(tensor_embedding),
                "teacher_embedding_pairwise_cosine": off_diagonal_cosine_mean(text_embedding),
                "centered_student_embedding_pairwise_cosine": off_diagonal_cosine_mean(centered_tensor_embedding),
                "centered_teacher_embedding_pairwise_cosine": off_diagonal_cosine_mean(centered_text_embedding),
                "hidden_positive_cosine": float(
                    F.cosine_similarity(student_hidden.float(), teacher_hidden.float(), dim=-1)
                    .mean()
                    .cpu()
                    .item()
                ),
                "student_hidden_pairwise_cosine": off_diagonal_cosine_mean(student_hidden),
                "teacher_hidden_pairwise_cosine": off_diagonal_cosine_mean(teacher_hidden),
            },
            batch_size,
            "alignment_",
        )
        total_records += batch_size
        if bool(args.global_retrieval_eval):
            collected_student_features.append(student_features.detach().float().cpu())
            collected_teacher_features.append(teacher_features.detach().float().cpu())
            collected_student_hidden.append(student_hidden.detach().float().cpu())
            collected_teacher_hidden.append(teacher_hidden.detach().float().cpu())
            if probe_target_ids is not None:
                collected_probe_target_ids.append(probe_target_ids.detach().long().cpu())
    adapter.capture_spatial_path_metrics = False
    metrics = {
        "loss": total_loss / max(1, total_records),
        "contrastive_loss": total_contrastive / max(1, total_records),
        "reconstruction_loss": total_reconstruction / max(1, total_records),
        "i2t_accuracy": total_i2t / max(1, total_records),
        "t2i_accuracy": total_t2i / max(1, total_records),
    }
    metrics.update(averaged_metrics(metric_totals, total_records))
    metrics.update(alignment_adapter_parameter_metrics(adapter))
    for key in (
        "candidate_count",
        "i2t_loss",
        "t2i_loss",
        "strict_i2t_loss",
        "strict_t2i_loss",
        "strict_contrastive_loss",
        "i2t_candidate_coverage",
        "t2i_candidate_coverage",
        "i2t_max_candidate_hit_fraction",
        "t2i_max_candidate_hit_fraction",
        "i2t_candidate_hit_entropy",
        "t2i_candidate_hit_entropy",
        "semantic_i2t_accuracy",
        "semantic_t2i_accuracy",
        "semantic_collision_fraction",
        "semantic_target_unique_fraction",
        "valid_negative_count",
    ):
        prefixed_key = f"contrastive_{key}"
        if prefixed_key in metrics:
            metrics[key] = metrics[prefixed_key]
    metrics = weighted_average_metrics_across_processes(metrics, total_records)
    if bool(args.global_retrieval_eval) and collected_student_features:
        student_all = torch.cat(collected_student_features, dim=0)
        teacher_all = torch.cat(collected_teacher_features, dim=0)
        student_hidden_all = torch.cat(collected_student_hidden, dim=0)
        teacher_hidden_all = torch.cat(collected_teacher_hidden, dim=0)
        probe_target_ids_all = (
            torch.cat(collected_probe_target_ids, dim=0) if collected_probe_target_ids else None
        )
        if distributed_is_initialized():
            student_all = gather_variable_rows_without_grad(student_all)
            teacher_all = gather_variable_rows_without_grad(teacher_all)
            student_hidden_all = gather_variable_rows_without_grad(student_hidden_all)
            teacher_hidden_all = gather_variable_rows_without_grad(teacher_hidden_all)
            if probe_target_ids_all is not None:
                probe_target_ids_all = gather_variable_rows_without_grad(probe_target_ids_all)
        max_global_records = int(args.global_retrieval_max_records)
        student_all = student_all[:max_global_records]
        teacher_all = teacher_all[:max_global_records]
        student_hidden_all = student_hidden_all[:max_global_records]
        teacher_hidden_all = teacher_hidden_all[:max_global_records]
        if probe_target_ids_all is not None:
            probe_target_ids_all = probe_target_ids_all[:max_global_records]
        global_metrics = (
            global_retrieval_metrics_from_features(
                student_all,
                teacher_all,
                student_hidden_all,
                teacher_hidden_all,
                probe_target_ids_all,
                args,
            )
            if is_main_process()
            else {}
        )
        metrics.update(broadcast_object_from_main(global_metrics))
    elif bool(args.global_retrieval_eval):
        metrics["global_retrieval_skipped"] = 1.0
    return metrics


def average_metric_dicts(metric_dicts: Sequence[Mapping[str, float]]) -> dict[str, float]:
    keys = sorted({key for metrics in metric_dicts for key in metrics})
    averaged: dict[str, float] = {}
    for key in keys:
        values = [float(metrics[key]) for metrics in metric_dicts if isinstance(metrics.get(key), (int, float))]
        if values:
            averaged[key] = sum(values) / len(values)
    return averaged


def checkpoint_selection_value(metrics: Mapping[str, Any], args: argparse.Namespace) -> tuple[str, float]:
    strict_i2t_key = (
        "global_strict_i2t_loss"
        if "global_strict_i2t_loss" in metrics
        else "strict_i2t_loss"
    )
    strict_t2i_key = (
        "global_strict_t2i_loss"
        if "global_strict_t2i_loss" in metrics
        else "strict_t2i_loss"
    )
    for key in (strict_i2t_key, strict_t2i_key):
        if key not in metrics:
            raise ValueError(f"Validation metrics are missing directional checkpoint metric {key!r}.")
    primary_weight = float(args.contrastive_loss_weight)
    i2t_weight, t2i_weight = normalized_contrastive_direction_weights(
        float(args.contrastive_i2t_weight),
        float(args.contrastive_t2i_weight),
    )
    directional_loss = (
        i2t_weight * float(metrics[strict_i2t_key])
        + t2i_weight * float(metrics[strict_t2i_key])
    )
    value = primary_weight * directional_loss
    # Global centered retrieval subtracts the validation candidate-library mean and is not
    # available for a single tensor at deployment. Keep it diagnostic-only; checkpoint
    # selection uses the per-batch centered objective instead.
    centered_key = "centered_strict_contrastive_loss"
    centered_weight = float(args.centered_contrastive_loss_weight)
    if centered_weight > 0.0:
        if centered_key not in metrics:
            raise ValueError(f"Validation metrics are missing centered checkpoint metric {centered_key!r}.")
        value += centered_weight * float(metrics[centered_key])
    native_key = "native_centered_strict_contrastive_loss"
    native_weight = float(args.native_centered_contrastive_loss_weight)
    if native_weight > 0.0:
        if native_key not in metrics:
            raise ValueError(f"Validation metrics are missing native checkpoint metric {native_key!r}.")
        value += native_weight * float(metrics[native_key])
    # The full-validation mean is also a candidate-library statistic. Use the same
    # per-batch branch-mean regularizer that was used during training.
    mean_key = "mean_alignment_loss"
    mean_weight = float(args.mean_alignment_loss_weight)
    if mean_weight > 0.0:
        if mean_key not in metrics:
            raise ValueError(f"Validation metrics are missing mean-alignment checkpoint metric {mean_key!r}.")
        value += mean_weight * float(metrics[mean_key])
    return (
        f"{primary_weight:g}*({i2t_weight:g}*{strict_i2t_key}+{t2i_weight:g}*{strict_t2i_key})"
        f"+{centered_weight:g}*{centered_key}"
        f"+{native_weight:g}*{native_key}"
        f"+{mean_weight:g}*{mean_key}",
        value,
    )


def teacher_probe_preflight_warnings(
    preflight: Mapping[str, Any],
    warn_below_correlation: float | None,
) -> list[str]:
    if not preflight or warn_below_correlation is None:
        return []
    threshold = float(warn_below_correlation)
    groups = dict(preflight.get("families") or preflight.get("anchors") or {})
    warnings: list[str] = []
    for name, metrics in groups.items():
        correlation = float(
            metrics.get(
                "hidden_similarity_vs_negative_target_distance_pearson_median",
                metrics.get("hidden_similarity_vs_negative_target_distance_pearson", float("nan")),
            )
        )
        if not np.isfinite(correlation):
            warnings.append(f"{name}=non-finite")
        elif correlation < threshold:
            warnings.append(f"{name}={correlation:.4f}")
    return warnings


@torch.no_grad()
def evaluate_anchor_bank(
    *,
    compressor: nn.Module,
    adapter: TensorPatchAlignmentAdapter,
    projector: AlignmentFeatureTransform | None,
    llm: nn.Module,
    tokenizer: Any,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    compressor_input_size: Sequence[int],
    normalization_cfg: Mapping[str, Any],
) -> dict[str, float]:
    if str(args.alignment_text_layout) != "values_shared_suffix":
        return evaluate(
            compressor=compressor,
            adapter=adapter,
            projector=projector,
            llm=llm,
            tokenizer=tokenizer,
            loader=loader,
            device=device,
            args=args,
            compressor_input_size=compressor_input_size,
            normalization_cfg=normalization_cfg,
        )

    anchors = alignment_anchors_from_args(tokenizer, args, evaluation=True)
    metrics_by_anchor: list[tuple[AlignmentAnchor, dict[str, float]]] = []
    for anchor in anchors:
        metrics = evaluate(
            compressor=compressor,
            adapter=adapter,
            projector=projector,
            llm=llm,
            tokenizer=tokenizer,
            loader=loader,
            device=device,
            args=args,
            compressor_input_size=compressor_input_size,
            normalization_cfg=normalization_cfg,
            alignment_anchor=anchor,
        )
        metrics_by_anchor.append((anchor, metrics))

    combined = average_metric_dicts([metrics for _anchor, metrics in metrics_by_anchor])
    for anchor, metrics in metrics_by_anchor:
        combined.update({f"anchor_{anchor.name}_{key}": float(value) for key, value in metrics.items()})
    probe_metrics = [metrics for anchor, metrics in metrics_by_anchor if anchor.mode == "probe"]
    if probe_metrics:
        combined.update(
            {f"probe_macro_{key}": value for key, value in average_metric_dicts(probe_metrics).items()}
        )
    combined["evaluation_anchor_count"] = float(len(metrics_by_anchor))
    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train tensor-as-text patch alignment: a PDEBench patch goes through an AE/adapter path, "
            "and its serialized text goes through a frozen LLM teacher hidden state."
        )
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--hdf5-path", type=str, default=None)
    parser.add_argument("--fields", type=str, default=None)
    parser.add_argument(
        "--field-sampling-mode",
        type=str,
        choices=("channels", "single"),
        default=None,
        help=(
            "channels stacks all --fields as patch channels; single samples one field per record "
            "so multiple fields form a larger pool of single-channel patches."
        ),
    )
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--train-records", type=int, default=None)
    parser.add_argument("--val-records", type=int, default=None)
    parser.add_argument("--test-records", type=int, default=None)
    parser.add_argument("--sample-indices", type=str, default=None)
    parser.add_argument("--time-indices", type=str, default=None)
    parser.add_argument(
        "--split-mode",
        type=str,
        choices=("random_record", "sample", "time", "sample_time"),
        default=None,
    )
    parser.add_argument("--split-train-ratio", type=float, default=None)
    parser.add_argument("--split-val-ratio", type=float, default=None)
    parser.add_argument("--split-test-ratio", type=float, default=None)
    parser.add_argument("--unique-records", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--ensure-disjoint-records", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--distributed-timeout-seconds", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--encoder-source", type=str, choices=("checkpoint", "patch_ae_config"), default=None)
    parser.add_argument("--train-patch-ae", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--freeze-patch-ae-after-pretrain", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--patch-ae-pretrain-epochs", type=int, default=None)
    parser.add_argument("--patch-ae-pretrain-batch-size", type=int, default=None)
    parser.add_argument("--compressor-checkpoint", type=str, default=None)
    parser.add_argument("--compressor-config", type=str, default=None)
    parser.add_argument("--resize-patch-to-compressor-input", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--model-name-or-path", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--hf-home", type=str, default=None)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--torch-dtype", type=str, choices=("auto", "float32", "float16", "bfloat16"), default=None)
    parser.add_argument("--llm-gradient-checkpointing", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--adapter-dim", type=int, default=None)
    parser.add_argument(
        "--adapter-type",
        type=str,
        choices=("qformer", "spatial_transformer", "pooled_mlp"),
        default=None,
    )
    parser.add_argument("--query-tokens", type=int, default=None)
    parser.add_argument("--adapter-layers", type=int, default=None)
    parser.add_argument("--adapter-heads", type=int, default=None)
    parser.add_argument("--projection-dim", type=int, default=None)
    parser.add_argument(
        "--alignment-transform-mode",
        type=str,
        choices=("none", "projection", "whitening"),
        default=None,
        help="Feature space used by stage-1 InfoNCE after the frozen LLM hidden readout.",
    )
    parser.add_argument("--alignment-whitening-records", type=int, default=None)
    parser.add_argument("--alignment-whitening-dim", type=int, default=None)
    parser.add_argument("--alignment-whitening-shrinkage", type=float, default=None)
    parser.add_argument("--alignment-whitening-epsilon", type=float, default=None)
    parser.add_argument("--alignment-whitening-max-condition-number", type=float, default=None)
    # Legacy compatibility. Prefer --alignment-transform-mode for new runs.
    parser.add_argument("--alignment-projection-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--alignment-projection-dim", type=int, default=None)
    parser.add_argument("--alignment-projection-hidden-dim", type=int, default=None)
    parser.add_argument("--alignment-projection-layers", type=int, default=None)
    parser.add_argument("--alignment-projection-dropout", type=float, default=None)
    parser.add_argument("--alignment-projection-shared", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--soft-prompt-scale", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--contrastive-loss-weight", type=float, default=None)
    parser.add_argument("--contrastive-i2t-weight", type=float, default=None)
    parser.add_argument("--contrastive-t2i-weight", type=float, default=None)
    parser.add_argument("--centered-contrastive-loss-weight", type=float, default=None)
    parser.add_argument("--native-centered-contrastive-loss-weight", type=float, default=None)
    parser.add_argument("--mean-alignment-loss-weight", type=float, default=None)
    parser.add_argument("--reconstruction-loss-weight", type=float, default=None)
    parser.add_argument("--alignment-patch-ae-lr-scale", type=float, default=None)
    parser.add_argument(
        "--teacher-text-source",
        type=str,
        choices=("normalized", "raw"),
        default=None,
        help="Use normalized AE-input patches or raw patches when serializing the teacher text branch.",
    )
    parser.add_argument(
        "--alignment-text-layout",
        type=str,
        choices=("values_shared_suffix", "legacy_prompt"),
        default=None,
        help=(
            "values_shared_suffix serializes only tensor values and appends the same setting-specific suffix used after "
            "student soft embeddings; legacy_prompt preserves the original asymmetric prompts."
        ),
    )
    parser.add_argument("--shared-suffix", type=str, default=None)
    parser.add_argument(
        "--alignment-anchor-mode",
        type=str,
        choices=("eos", "representation", "probe"),
        default=None,
        help="Select exactly one stage-1 alignment setting.",
    )
    parser.add_argument("--representation-suffix", type=str, default=None)
    parser.add_argument("--probe-families", type=str, default=None)
    parser.add_argument("--probe-region-size", type=int, default=None)
    parser.add_argument("--evaluation-probe-count", type=int, default=None)
    parser.add_argument("--teacher-probe-warn-below-correlation", type=float, default=None)
    parser.add_argument("--teacher-probe-diagnostic-records", type=int, default=None)
    parser.add_argument("--max-shared-suffix-tokens", type=int, default=None)
    parser.add_argument("--fail-on-text-anchor-missing", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--fail-on-text-max-length-hit", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--global-retrieval-eval", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--global-retrieval-max-records", type=int, default=None)
    parser.add_argument("--global-retrieval-chunk-size", type=int, default=None)
    parser.add_argument(
        "--text-prompt-template",
        type=str,
        choices=("compact", "compact_with_metadata", "plain"),
        default=None,
    )
    parser.add_argument("--text-decimal-places", type=int, default=None)
    parser.add_argument("--max-text-tokens", type=int, default=None)
    parser.add_argument("--text-preflight-records", type=int, default=None)
    parser.add_argument("--teacher-layer", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--wandb-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--wandb-api-key", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, default=None)
    parser.add_argument(
        "--wandb-mode",
        type=str,
        choices=("online", "offline", "disabled"),
        default=None,
    )
    parser.add_argument("--wandb-log-model", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype | str:
    if name == "auto":
        return "auto"
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def apply_config_defaults(args: argparse.Namespace, config: Mapping[str, Any]) -> argparse.Namespace:
    reject_removed_alignment_options(config)
    set_default(args, "encoder_source", first_nested(config, ["patch_alignment.encoder_source"]), "patch_ae_config")
    path_defaults = {
        "hdf5_path": first_nested(config, ["patch_alignment.hdf5_path", "data.hdf5_path"]),
        "output_root": first_nested(config, ["patch_alignment.output_root", "llm_training.output_root"]),
        "compressor_config": first_nested(config, ["patch_alignment.compressor_config", "compressor.config"]),
        "cache_dir": first_nested(config, ["model.cache_dir", "storage.hf_home"]),
        "hf_home": first_nested(config, ["storage.hf_home"]),
    }
    compressor_checkpoint = first_nested(config, ["patch_alignment.compressor_checkpoint"])
    if compressor_checkpoint is None and str(args.encoder_source) == "checkpoint":
        compressor_checkpoint = first_nested(config, ["compressor.checkpoint"])
    if args.compressor_checkpoint is None and compressor_checkpoint is not None:
        args.compressor_checkpoint = resolve_path_string(compressor_checkpoint, PROJECT_ROOT)
    for attr, value in path_defaults.items():
        if getattr(args, attr, None) is None and value is not None:
            setattr(args, attr, resolve_path_string(value, PROJECT_ROOT))
    if args.model_name_or_path is None:
        model_path = first_nested(config, ["model.local_dir", "model.name_or_path"])
        if model_path is not None:
            args.model_name_or_path = str(model_path)

    set_default(args, "fields", value_to_csv(first_nested(config, ["patch_alignment.fields", "data.fields"])), None)
    set_default(args, "field_sampling_mode", first_nested(config, ["patch_alignment.field_sampling_mode"]), "channels")
    set_default(args, "run_name", first_nested(config, ["patch_alignment.run_name"]), "tensor_patch_text_alignment")
    set_default(args, "patch_size", first_nested(config, ["patch_alignment.patch_size"]), 16)
    set_default(args, "train_records", first_nested(config, ["patch_alignment.train_records"]), 4096)
    set_default(args, "val_records", first_nested(config, ["patch_alignment.val_records"]), 512)
    set_default(args, "test_records", first_nested(config, ["patch_alignment.test_records"]), 512)
    set_default(args, "sample_indices", value_to_csv(first_nested(config, ["patch_alignment.sample_indices"])), "all")
    set_default(args, "time_indices", value_to_csv(first_nested(config, ["patch_alignment.time_indices"])), "all")
    set_default(args, "split_mode", first_nested(config, ["patch_alignment.split_mode"]), "sample")
    set_default(args, "split_train_ratio", first_nested(config, ["patch_alignment.split_train_ratio"]), 0.8)
    set_default(args, "split_val_ratio", first_nested(config, ["patch_alignment.split_val_ratio"]), 0.1)
    set_default(args, "split_test_ratio", first_nested(config, ["patch_alignment.split_test_ratio"]), 0.1)
    set_default(args, "unique_records", first_nested(config, ["patch_alignment.unique_records"]), True)
    set_default(args, "ensure_disjoint_records", first_nested(config, ["patch_alignment.ensure_disjoint_records"]), True)
    set_default(args, "seed", first_nested(config, ["patch_alignment.seed", "runtime.seed"]), 42)
    set_default(args, "device", first_nested(config, ["patch_alignment.device", "runtime.device"]), "auto")
    set_default(args, "batch_size", first_nested(config, ["patch_alignment.batch_size"]), 8)
    set_default(args, "eval_batch_size", first_nested(config, ["patch_alignment.eval_batch_size"]), args.batch_size)
    set_default(args, "epochs", first_nested(config, ["patch_alignment.epochs"]), 10)
    set_default(args, "lr", first_nested(config, ["patch_alignment.lr"]), 1.0e-4)
    set_default(args, "weight_decay", first_nested(config, ["patch_alignment.weight_decay"]), 1.0e-4)
    set_default(args, "grad_clip_norm", first_nested(config, ["patch_alignment.grad_clip_norm"]), 1.0)
    set_default(args, "num_workers", first_nested(config, ["patch_alignment.num_workers"]), 0)
    set_default(
        args,
        "distributed_timeout_seconds",
        first_nested(config, ["patch_alignment.distributed_timeout_seconds"]),
        1800.0,
    )
    set_default(args, "train_patch_ae", first_nested(config, ["patch_alignment.train_patch_ae"]), args.encoder_source == "patch_ae_config")
    set_default(args, "freeze_patch_ae_after_pretrain", first_nested(config, ["patch_alignment.freeze_patch_ae_after_pretrain"]), True)
    set_default(args, "patch_ae_pretrain_epochs", first_nested(config, ["patch_alignment.patch_ae_pretrain_epochs"]), 0)
    set_default(
        args,
        "patch_ae_pretrain_batch_size",
        first_nested(config, ["patch_alignment.patch_ae_pretrain_batch_size"]),
        args.batch_size,
    )
    set_default(
        args,
        "resize_patch_to_compressor_input",
        first_nested(config, ["patch_alignment.resize_patch_to_compressor_input"]),
        args.encoder_source == "checkpoint",
    )
    set_default(args, "trust_remote_code", first_nested(config, ["model.trust_remote_code"]), False)
    set_default(args, "torch_dtype", first_nested(config, ["model.torch_dtype"]), "bfloat16")
    set_default(
        args,
        "llm_gradient_checkpointing",
        first_nested(config, ["patch_alignment.llm_gradient_checkpointing"]),
        False,
    )
    set_default(args, "adapter_dim", first_nested(config, ["patch_alignment.adapter_dim"]), 512)
    set_default(args, "adapter_type", first_nested(config, ["patch_alignment.adapter_type"]), "qformer")
    set_default(args, "query_tokens", first_nested(config, ["patch_alignment.query_tokens"]), 8)
    set_default(args, "adapter_layers", first_nested(config, ["patch_alignment.adapter_layers"]), 2)
    set_default(args, "adapter_heads", first_nested(config, ["patch_alignment.adapter_heads"]), 8)
    set_default(args, "projection_dim", first_nested(config, ["patch_alignment.projection_dim"]), None)
    configured_transform_mode = first_nested(config, ["patch_alignment.alignment_transform.mode"])
    legacy_projection_enabled = first_nested(config, ["patch_alignment.alignment_projection.enabled"])
    if args.alignment_transform_mode is None:
        args.alignment_transform_mode = configured_transform_mode
    if args.alignment_transform_mode is None:
        legacy_value = (
            args.alignment_projection_enabled
            if args.alignment_projection_enabled is not None
            else legacy_projection_enabled
        )
        args.alignment_transform_mode = "projection" if bool(legacy_value) else "none"
    else:
        explicit_legacy_value = (
            args.alignment_projection_enabled
            if args.alignment_projection_enabled is not None
            else legacy_projection_enabled
        )
        if explicit_legacy_value is not None and bool(explicit_legacy_value) != (
            str(args.alignment_transform_mode).lower() == "projection"
        ):
            raise ValueError(
                "Conflicting alignment transform settings: patch_alignment.alignment_transform.mode="
                f"{args.alignment_transform_mode!r} but alignment_projection.enabled="
                f"{bool(explicit_legacy_value)!r}. Remove the legacy enabled field."
            )
    args.alignment_transform_mode = str(args.alignment_transform_mode).lower()
    args.alignment_projection_enabled = args.alignment_transform_mode == "projection"
    set_default(
        args,
        "alignment_whitening_records",
        first_nested(config, ["patch_alignment.alignment_transform.whitening.records"]),
        8192,
    )
    set_default(
        args,
        "alignment_whitening_dim",
        first_nested(config, ["patch_alignment.alignment_transform.whitening.dim"]),
        512,
    )
    set_default(
        args,
        "alignment_whitening_shrinkage",
        first_nested(config, ["patch_alignment.alignment_transform.whitening.shrinkage"]),
        0.01,
    )
    set_default(
        args,
        "alignment_whitening_epsilon",
        first_nested(config, ["patch_alignment.alignment_transform.whitening.epsilon"]),
        1.0e-5,
    )
    set_default(
        args,
        "alignment_whitening_max_condition_number",
        first_nested(config, ["patch_alignment.alignment_transform.whitening.max_condition_number"]),
        1000.0,
    )
    set_default(
        args,
        "alignment_projection_dim",
        first_nested(config, ["patch_alignment.alignment_projection.dim"]),
        512,
    )
    set_default(
        args,
        "alignment_projection_hidden_dim",
        first_nested(config, ["patch_alignment.alignment_projection.hidden_dim"]),
        1024,
    )
    set_default(
        args,
        "alignment_projection_layers",
        first_nested(config, ["patch_alignment.alignment_projection.layers"]),
        1,
    )
    set_default(
        args,
        "alignment_projection_dropout",
        first_nested(config, ["patch_alignment.alignment_projection.dropout"]),
        0.0,
    )
    set_default(
        args,
        "alignment_projection_shared",
        first_nested(config, ["patch_alignment.alignment_projection.shared"]),
        False,
    )
    set_default(args, "dropout", first_nested(config, ["patch_alignment.dropout"]), 0.0)
    set_default(args, "soft_prompt_scale", first_nested(config, ["patch_alignment.soft_prompt_scale"]), 0.05)
    set_default(args, "temperature", first_nested(config, ["patch_alignment.temperature"]), 0.07)
    set_default(args, "contrastive_loss_weight", first_nested(config, ["patch_alignment.contrastive_loss_weight"]), 1.0)
    set_default(
        args,
        "contrastive_i2t_weight",
        first_nested(config, ["patch_alignment.contrastive_direction_weights.i2t"]),
        0.75,
    )
    set_default(
        args,
        "contrastive_t2i_weight",
        first_nested(config, ["patch_alignment.contrastive_direction_weights.t2i"]),
        0.25,
    )
    set_default(
        args,
        "centered_contrastive_loss_weight",
        first_nested(config, ["patch_alignment.centered_contrastive_loss_weight"]),
        0.0,
    )
    set_default(
        args,
        "native_centered_contrastive_loss_weight",
        first_nested(config, ["patch_alignment.native_centered_contrastive_loss_weight"]),
        0.0,
    )
    set_default(
        args,
        "mean_alignment_loss_weight",
        first_nested(config, ["patch_alignment.mean_alignment_loss_weight"]),
        0.0,
    )
    set_default(args, "reconstruction_loss_weight", first_nested(config, ["patch_alignment.reconstruction_loss_weight"]), 1.0)
    set_default(
        args,
        "alignment_patch_ae_lr_scale",
        first_nested(config, ["patch_alignment.alignment_patch_ae_lr_scale"]),
        1.0,
    )
    set_default(args, "teacher_text_source", first_nested(config, ["patch_alignment.teacher_text_source"]), "normalized")
    set_default(
        args,
        "alignment_text_layout",
        first_nested(config, ["patch_alignment.alignment_text_layout"]),
        "legacy_prompt",
    )
    set_default(args, "shared_suffix", first_nested(config, ["patch_alignment.shared_suffix"]), "\nRepresentation:")
    set_default(
        args,
        "alignment_anchor_mode",
        first_nested(config, ["patch_alignment.alignment_anchor_mode"]),
        "representation",
    )
    set_default(
        args,
        "representation_suffix",
        first_nested(config, ["patch_alignment.representation_suffix"]),
        args.shared_suffix,
    )
    set_default(
        args,
        "probe_families",
        value_to_csv(first_nested(config, ["patch_alignment.probe_families"])),
        "point_value,point_difference,point_mean,region_mean,region_range",
    )
    set_default(args, "probe_region_size", first_nested(config, ["patch_alignment.probe_region_size"]), 4)
    set_default(args, "evaluation_probe_count", first_nested(config, ["patch_alignment.evaluation_probe_count"]), 3)
    set_default(
        args,
        "teacher_probe_warn_below_correlation",
        first_nested(config, ["patch_alignment.teacher_probe_warn_below_correlation"]),
        0.1,
    )
    set_default(
        args,
        "teacher_probe_diagnostic_records",
        first_nested(config, ["patch_alignment.teacher_probe_diagnostic_records"]),
        128,
    )
    set_default(
        args,
        "max_shared_suffix_tokens",
        first_nested(config, ["patch_alignment.max_shared_suffix_tokens"]),
        96,
    )
    set_default(
        args,
        "fail_on_text_anchor_missing",
        first_nested(config, ["patch_alignment.fail_on_text_anchor_missing"]),
        True,
    )
    set_default(
        args,
        "fail_on_text_max_length_hit",
        first_nested(config, ["patch_alignment.fail_on_text_max_length_hit"]),
        True,
    )
    set_default(args, "global_retrieval_eval", first_nested(config, ["patch_alignment.global_retrieval_eval"]), True)
    set_default(
        args,
        "global_retrieval_max_records",
        first_nested(config, ["patch_alignment.global_retrieval_max_records"]),
        8192,
    )
    set_default(
        args,
        "global_retrieval_chunk_size",
        first_nested(config, ["patch_alignment.global_retrieval_chunk_size"]),
        1024,
    )
    set_default(args, "text_prompt_template", first_nested(config, ["patch_alignment.text_prompt_template"]), "compact")
    set_default(args, "text_decimal_places", first_nested(config, ["patch_alignment.text_decimal_places"]), 3)
    set_default(args, "max_text_tokens", first_nested(config, ["patch_alignment.max_text_tokens"]), 1024)
    set_default(args, "text_preflight_records", first_nested(config, ["patch_alignment.text_preflight_records"]), 32)
    set_default(args, "teacher_layer", first_nested(config, ["patch_alignment.teacher_layer"]), 14)
    set_default(args, "log_interval", first_nested(config, ["patch_alignment.log_interval"]), 20)
    set_default(args, "wandb_enabled", first_nested(config, ["wandb.enabled"]), False)
    set_default(args, "wandb_api_key", first_nested(config, ["wandb.api_key"]), None)
    set_default(args, "wandb_project", first_nested(config, ["wandb.project"]), "tensor-compression")
    set_default(args, "wandb_entity", first_nested(config, ["wandb.entity"]), None)
    set_default(args, "wandb_group", first_nested(config, ["wandb.group"]), "patch-alignment")
    set_default(args, "wandb_tags", value_to_csv(first_nested(config, ["wandb.tags"])), "adapter,tensor-llm,patch-alignment")
    set_default(args, "wandb_mode", first_nested(config, ["wandb.mode"]), "offline")
    set_default(args, "wandb_log_model", first_nested(config, ["wandb.log_model"]), False)
    args.patch_encoder_config = first_nested(config, ["patch_alignment.patch_encoder"])

    require_args(args, ["hdf5_path", "model_name_or_path", "output_root"])
    if str(args.encoder_source) == "checkpoint" and not args.compressor_checkpoint:
        raise ValueError("encoder_source=checkpoint requires --compressor-checkpoint or patch_alignment.compressor_checkpoint.")
    for name in (
        "patch_size",
        "train_records",
        "val_records",
        "test_records",
        "batch_size",
        "eval_batch_size",
        "patch_ae_pretrain_batch_size",
        "epochs",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"patch_alignment.{name} must be positive.")
    if int(args.patch_ae_pretrain_epochs) < 0:
        raise ValueError("patch_alignment.patch_ae_pretrain_epochs must be non-negative.")
    if float(args.distributed_timeout_seconds) <= 0.0:
        raise ValueError("patch_alignment.distributed_timeout_seconds must be positive.")
    for name in ("adapter_dim", "query_tokens", "adapter_layers", "adapter_heads"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"patch_alignment.{name} must be positive.")
    if str(args.adapter_type).lower() not in {"qformer", "spatial_transformer", "pooled_mlp"}:
        raise ValueError(f"Unsupported patch_alignment.adapter_type: {args.adapter_type}")
    if int(args.max_text_tokens) <= 0:
        raise ValueError("patch_alignment.max_text_tokens must be positive.")
    if int(args.max_shared_suffix_tokens) <= 0:
        raise ValueError("patch_alignment.max_shared_suffix_tokens must be positive.")
    args.alignment_anchor_mode = str(args.alignment_anchor_mode).lower()
    args.probe_families = [family.lower() for family in parse_csv(args.probe_families)]
    if str(args.alignment_text_layout) != "values_shared_suffix" and args.alignment_anchor_mode != "representation":
        raise ValueError(
            "eos/probe alignment_anchor_mode requires alignment_text_layout=values_shared_suffix."
        )
    if str(args.alignment_text_layout) == "values_shared_suffix":
        valid_anchor_modes = {"eos", "representation", "probe"}
        if args.alignment_anchor_mode not in valid_anchor_modes:
            raise ValueError(
                "patch_alignment.alignment_anchor_mode must be eos, representation, or probe."
            )
        if args.alignment_anchor_mode == "representation" and not str(args.representation_suffix):
            raise ValueError("patch_alignment.representation_suffix must not be empty.")
        if args.alignment_anchor_mode == "probe":
            if not args.probe_families:
                raise ValueError("patch_alignment.probe_families must not be empty in probe mode.")
            unsupported_probe_families = sorted(
                set(args.probe_families) - set(PROBE_FAMILIES)
            )
            if unsupported_probe_families:
                raise ValueError(f"Unsupported patch_alignment.probe_families: {unsupported_probe_families}.")
            if len(args.probe_families) != len(set(args.probe_families)):
                raise ValueError("patch_alignment.probe_families must not contain duplicates.")
            if int(args.probe_region_size) <= 0 or int(args.probe_region_size) >= int(args.patch_size):
                raise ValueError("patch_alignment.probe_region_size must be between 1 and patch_size - 1.")
            if int(args.evaluation_probe_count) <= 0:
                raise ValueError("patch_alignment.evaluation_probe_count must be positive.")
            if args.teacher_probe_diagnostic_records < 2:
                raise ValueError(
                    "patch_alignment.teacher_probe_diagnostic_records must be at least 2 in probe mode."
                )
            if args.teacher_probe_warn_below_correlation is not None and not -1.0 <= float(
                args.teacher_probe_warn_below_correlation
            ) <= 1.0:
                raise ValueError(
                    "patch_alignment.teacher_probe_warn_below_correlation must be in [-1, 1]."
                )
    if int(args.text_preflight_records) < 0:
        raise ValueError("patch_alignment.text_preflight_records must be non-negative.")
    if int(args.global_retrieval_max_records) <= 0:
        raise ValueError("patch_alignment.global_retrieval_max_records must be positive.")
    if int(args.global_retrieval_chunk_size) <= 0:
        raise ValueError("patch_alignment.global_retrieval_chunk_size must be positive.")
    if str(args.field_sampling_mode).lower() not in {"channels", "single"}:
        raise ValueError("patch_alignment.field_sampling_mode must be 'channels' or 'single'.")
    split_ratio_sum = float(args.split_train_ratio) + float(args.split_val_ratio) + float(args.split_test_ratio)
    if split_ratio_sum <= 0.0:
        raise ValueError("patch_alignment split ratios must sum to a positive value.")
    if float(args.soft_prompt_scale) < 0.0:
        raise ValueError("patch_alignment.soft_prompt_scale must be non-negative.")
    if float(args.temperature) <= 0.0:
        raise ValueError("patch_alignment.temperature must be positive.")
    if args.alignment_transform_mode not in {"none", "projection", "whitening"}:
        raise ValueError("patch_alignment.alignment_transform.mode must be none, projection, or whitening.")
    if args.alignment_transform_mode == "projection":
        if int(args.alignment_projection_dim) <= 0:
            raise ValueError("patch_alignment.alignment_projection.dim must be positive.")
        if int(args.alignment_projection_hidden_dim) <= 0:
            raise ValueError("patch_alignment.alignment_projection.hidden_dim must be positive.")
        if int(args.alignment_projection_layers) <= 0:
            raise ValueError("patch_alignment.alignment_projection.layers must be positive.")
        if float(args.alignment_projection_dropout) < 0.0:
            raise ValueError("patch_alignment.alignment_projection.dropout must be non-negative.")
    if args.alignment_transform_mode == "whitening":
        if int(args.alignment_whitening_records) < 2:
            raise ValueError("patch_alignment.alignment_transform.whitening.records must be at least 2.")
        if int(args.alignment_whitening_dim) <= 0:
            raise ValueError("patch_alignment.alignment_transform.whitening.dim must be positive.")
        if not 0.0 <= float(args.alignment_whitening_shrinkage) <= 1.0:
            raise ValueError("patch_alignment.alignment_transform.whitening.shrinkage must be in [0, 1].")
        if float(args.alignment_whitening_epsilon) <= 0.0:
            raise ValueError("patch_alignment.alignment_transform.whitening.epsilon must be positive.")
        if float(args.alignment_whitening_max_condition_number) < 1.0:
            raise ValueError(
                "patch_alignment.alignment_transform.whitening.max_condition_number must be at least 1."
            )
    if int(args.text_decimal_places) < 0:
        raise ValueError("patch_alignment.text_decimal_places must be non-negative.")
    if str(args.alignment_anchor_mode) == "probe" and int(args.text_decimal_places) > 8:
        raise ValueError(
            "probe mode supports at most 8 text decimal places so quantized semantic target IDs remain stable."
        )
    if float(args.contrastive_loss_weight) <= 0.0:
        raise ValueError("patch_alignment.contrastive_loss_weight must be positive for alignment training.")
    normalized_contrastive_direction_weights(
        float(args.contrastive_i2t_weight),
        float(args.contrastive_t2i_weight),
    )
    if float(args.centered_contrastive_loss_weight) < 0.0:
        raise ValueError("patch_alignment.centered_contrastive_loss_weight must be non-negative.")
    if float(args.native_centered_contrastive_loss_weight) < 0.0:
        raise ValueError("patch_alignment.native_centered_contrastive_loss_weight must be non-negative.")
    if float(args.mean_alignment_loss_weight) < 0.0:
        raise ValueError("patch_alignment.mean_alignment_loss_weight must be non-negative.")
    if float(args.reconstruction_loss_weight) < 0.0:
        raise ValueError("patch_alignment.reconstruction_loss_weight must be non-negative.")
    if not 0.0 < float(args.alignment_patch_ae_lr_scale) <= 1.0:
        raise ValueError("patch_alignment.alignment_patch_ae_lr_scale must be in (0, 1].")
    if bool(args.train_patch_ae) and float(args.reconstruction_loss_weight) <= 0.0:
        raise ValueError(
            "train_patch_ae=true requires a positive patch_alignment.reconstruction_loss_weight."
        )
    if str(args.encoder_source) == "patch_ae_config" and not bool(args.train_patch_ae):
        raise ValueError(
            "encoder_source=patch_ae_config constructs a new random AE and therefore requires train_patch_ae=true. "
            "Use encoder_source=checkpoint to freeze and reuse an existing AE."
        )
    if str(args.alignment_text_layout) == "values_shared_suffix":
        if int(args.text_preflight_records) <= 0:
            raise ValueError(
                "values_shared_suffix requires a positive patch_alignment.text_preflight_records so tensor text "
                "truncation is checked before AE warmup."
            )
        teacher_source = str(args.teacher_text_source).lower()
        if teacher_source not in {"raw", "normalized"}:
            raise ValueError(
                "values_shared_suffix requires patch_alignment.teacher_text_source to be raw or normalized."
            )
        if str(args.adapter_type).lower() not in {"qformer", "spatial_transformer"}:
            raise ValueError(
                "values_shared_suffix requires a qformer or spatial_transformer soft-prefix adapter."
            )
    return args


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    sampler: torch.utils.data.Sampler | None = None,
    drop_last: bool = False,
) -> DataLoader:
    worker_count = int(num_workers)
    loader_kwargs: dict[str, Any] = {}
    if worker_count > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle) if sampler is None else False,
        sampler=sampler,
        num_workers=worker_count,
        pin_memory=torch.cuda.is_available(),
        drop_last=bool(drop_last),
        collate_fn=collate_patch_text,
        **loader_kwargs,
    )


def preflight_teacher_text_tokenization(
    *,
    dataset: Dataset,
    tokenizer: Any,
    args: argparse.Namespace,
    compressor_input_size: Sequence[int],
    normalization_cfg: Mapping[str, Any],
) -> dict[str, float]:
    contract_anchors = (
        probe_contract_anchors(tokenizer, args)
        if str(args.alignment_text_layout) == "values_shared_suffix"
        and str(args.alignment_anchor_mode) == "probe"
        else []
    )
    record_count = min(int(args.text_preflight_records), len(dataset))
    if record_count <= 0:
        return {
            "probe_contract_anchor_count": float(len(contract_anchors)),
            "probe_contract_family_count": float(len(parse_csv(args.probe_families))),
            "probe_contract_template_count": float(len(contract_anchors)),
        } if contract_anchors else {}
    items = [dataset[index] for index in range(record_count)]
    batch = collate_patch_text(items)
    normalized_patches = normalize_patch_batch(
        batch["patch"],
        compressor_input_size,
        normalization_cfg,
        bool(args.resize_patch_to_compressor_input),
    )
    texts = build_teacher_texts_for_batch(batch, normalized_patches, args)
    if str(args.alignment_text_layout) == "values_shared_suffix":
        anchors: list[AlignmentAnchor] = []
        seen: set[tuple[str, tuple[int, ...]]] = set()
        anchors_to_check = contract_anchors or (
            alignment_anchors_from_args(tokenizer, args, evaluation=False)
            + alignment_anchors_from_args(tokenizer, args, evaluation=True)
        )
        for anchor in anchors_to_check:
            key = (anchor.mode, anchor.token_ids)
            if key not in seen:
                seen.add(key)
                anchors.append(anchor)
        anchor_metrics: list[tuple[AlignmentAnchor, dict[str, float]]] = []
        for anchor in anchors:
            packed = tokenize_contents_with_anchor(
                tokenizer=tokenizer,
                contents=texts,
                anchor=anchor,
                max_tokens=int(args.max_text_tokens),
                require_under_max_length=bool(args.fail_on_text_max_length_hit),
                context=f"teacher text preflight ({anchor.name})",
            )
            anchor_metrics.append((anchor, dict(packed.metrics)))
        summary = average_metric_dicts([metrics for _anchor, metrics in anchor_metrics])
        for risk_key in (
            "token_count_max",
            "content_token_count_max",
            "suffix_token_count",
            "content_truncated_fraction",
            "max_length_hit_fraction",
            "anchor_missing_fraction",
        ):
            summary[risk_key] = max(metrics[risk_key] for _anchor, metrics in anchor_metrics)
        summary["anchor_count"] = float(len(anchor_metrics))
        for anchor, metrics in anchor_metrics:
            summary.update({f"anchor_{anchor.name}_{key}": value for key, value in metrics.items()})
        if contract_anchors:
            summary["probe_contract_anchor_count"] = float(len(contract_anchors))
            summary["probe_contract_family_count"] = float(len(parse_csv(args.probe_families)))
            summary["probe_contract_template_count"] = float(len(contract_anchors))
        return summary
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=int(args.max_text_tokens),
        return_tensors="pt",
    )
    return tokenizer_anchor_metrics(
        tokenizer=tokenizer,
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        max_tokens=int(args.max_text_tokens),
        anchor_text="Representation:" if str(args.text_prompt_template) != "plain" else "",
        require_anchor=bool(args.fail_on_text_anchor_missing) and str(args.text_prompt_template) != "plain",
        require_under_max_length=bool(args.fail_on_text_max_length_hit)
        and str(args.text_prompt_template) != "plain",
        context="teacher text preflight",
    )


@torch.no_grad()
def preflight_teacher_probe_semantics(
    *,
    dataset: Dataset,
    llm: nn.Module,
    tokenizer: Any,
    device: torch.device,
    args: argparse.Namespace,
    compressor_input_size: Sequence[int],
    normalization_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    if str(args.alignment_anchor_mode) != "probe" or int(args.teacher_probe_diagnostic_records) < 2:
        return {}
    record_limit = min(len(dataset), int(args.teacher_probe_diagnostic_records))
    if record_limit < 2:
        return {}
    loader = DataLoader(
        dataset,
        batch_size=min(4, int(args.eval_batch_size), record_limit),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_patch_text,
    )
    by_anchor: dict[str, dict[str, float]] = {}
    contract_anchors = probe_contract_anchors(tokenizer, args)
    for alignment_anchor in contract_anchors:
        hidden_rows: list[torch.Tensor] = []
        target_rows: list[torch.Tensor] = []
        target_id_rows: list[torch.Tensor] = []
        collected = 0
        for batch in loader:
            normalized_patches = normalize_patch_batch(
                batch["patch"],
                compressor_input_size,
                normalization_cfg,
                bool(args.resize_patch_to_compressor_input),
            )
            teacher_texts = build_teacher_texts_for_batch(batch, normalized_patches, args)
            target_values, target_ids = probe_targets_for_batch(
                batch,
                normalized_patches,
                args,
                alignment_anchor,
            )
            if target_values is None or target_ids is None:
                raise RuntimeError("Probe semantic preflight did not produce probe targets.")
            teacher_output = text_teacher_hidden(
                llm,
                tokenizer,
                teacher_texts,
                device,
                int(args.max_text_tokens),
                int(args.teacher_layer),
                bool(args.fail_on_text_anchor_missing) and str(args.text_prompt_template) != "plain",
                bool(args.fail_on_text_max_length_hit),
                text_layout=str(args.alignment_text_layout),
                shared_suffix=str(args.shared_suffix),
                max_shared_suffix_tokens=int(args.max_shared_suffix_tokens),
                alignment_anchor=alignment_anchor,
            )
            take = min(int(teacher_output.hidden.shape[0]), record_limit - collected)
            hidden_rows.append(teacher_output.hidden[:take].detach().float().cpu())
            target_rows.append(target_values[:take].detach().float().cpu())
            target_id_rows.append(target_ids[:take].detach().long().cpu())
            collected += take
            if collected >= record_limit:
                break
        hidden = torch.cat(hidden_rows, dim=0)
        targets = torch.cat(target_rows, dim=0)
        target_ids = torch.cat(target_id_rows, dim=0)
        _, counts = target_ids.unique(return_counts=True)
        metrics = target_geometry_metrics(hidden, targets)
        metrics.update(
            {
                "record_count": float(collected),
                "hidden_pairwise_cosine": off_diagonal_cosine_mean(hidden),
                "semantic_target_unique_fraction": float(counts.numel() / max(1, collected)),
                "semantic_collision_fraction": float(
                    ((counts.float() * (counts.float() - 1.0)).sum() / max(1, collected * (collected - 1))).item()
                ),
            }
        )
        by_anchor[alignment_anchor.name] = metrics
    by_family: dict[str, dict[str, float]] = {}
    anchor_family_by_name = {
        str(anchor.name): str(anchor.probe_family)
        for anchor in contract_anchors
    }
    for family in sorted(set(anchor_family_by_name.values())):
        template_metrics = [
            metrics
            for anchor_name, metrics in by_anchor.items()
            if anchor_family_by_name.get(str(anchor_name)) == family
        ]
        if not template_metrics:
            continue
        # The anchor name is intentionally not used as the grouping key. Probe templates
        # with different wording are observations of the same numeric operation.
        family_metrics = average_metric_dicts(template_metrics)
        correlations = [
            float(metrics["hidden_similarity_vs_negative_target_distance_pearson"])
            for metrics in template_metrics
            if np.isfinite(float(metrics.get("hidden_similarity_vs_negative_target_distance_pearson", float("nan"))))
        ]
        if correlations:
            family_metrics["hidden_similarity_vs_negative_target_distance_pearson_median"] = float(
                np.median(correlations)
            )
            family_metrics["hidden_similarity_vs_negative_target_distance_pearson_min"] = float(
                min(correlations)
            )
        family_metrics["template_count"] = float(len(template_metrics))
        by_family[family] = family_metrics
    macro = average_metric_dicts(list(by_family.values()) or list(by_anchor.values()))
    return {
        "record_limit": int(record_limit),
        "teacher_layer": int(args.teacher_layer),
        "macro": macro,
        "anchors": by_anchor,
        "families": by_family,
    }


def gather_feature_rows(local_rows: torch.Tensor) -> torch.Tensor | None:
    """Gather variable-length feature rows; only rank 0 retains the concatenated result."""
    if not distributed_is_initialized():
        return local_rows.detach().float()
    length = torch.tensor([int(local_rows.shape[0])], device=local_rows.device, dtype=torch.long)
    gathered_lengths = [torch.zeros_like(length) for _ in range(distributed_world_size())]
    dist.all_gather(gathered_lengths, length)
    row_counts = [int(item.item()) for item in gathered_lengths]
    max_rows = max(row_counts)
    if max_rows <= 0:
        return None
    if int(local_rows.shape[0]) < max_rows:
        padding = torch.zeros(
            max_rows - int(local_rows.shape[0]),
            int(local_rows.shape[1]),
            device=local_rows.device,
            dtype=local_rows.dtype,
        )
        local_rows = torch.cat([local_rows, padding], dim=0)
    gathered = [torch.empty_like(local_rows) for _ in range(distributed_world_size())]
    dist.all_gather(gathered, local_rows.contiguous())
    if not is_main_process():
        return None
    return torch.cat(
        [rows[:count].detach().float() for rows, count in zip(gathered, row_counts, strict=True)],
        dim=0,
    )


@torch.no_grad()
def fit_teacher_whitening(
    *,
    whitener: FixedTeacherWhitening,
    loader: DataLoader,
    llm: nn.Module,
    tokenizer: Any,
    device: torch.device,
    args: argparse.Namespace,
    compressor_input_size: Sequence[int],
    normalization_cfg: Mapping[str, Any],
) -> dict[str, float]:
    requested_records = min(int(args.alignment_whitening_records), len(loader.dataset))
    if requested_records < 2:
        raise ValueError("alignment_transform.whitening.records must select at least two train records.")
    local_target = (requested_records + distributed_world_size() - 1) // distributed_world_size()
    local_hidden: list[torch.Tensor] = []
    local_anchor_groups: list[torch.Tensor] = []
    local_count = 0
    static_anchors = alignment_anchors_from_args(tokenizer, args, evaluation=False)
    progress = tqdm(loader, desc="fit teacher whitening", leave=False, disable=not is_main_process())
    for step, batch in enumerate(progress, start=1):
        if local_count >= local_target:
            break
        if str(args.alignment_anchor_mode) == "probe":
            alignment_anchor = build_numeric_probe_anchor(
                tokenizer=tokenizer,
                patch_size=int(args.patch_size),
                channel_count=int(batch["patch"].shape[1]),
                families=args.probe_families,
                region_size=int(args.probe_region_size),
                probe_index=int(step) - 1,
                seed=int(args.seed),
                max_anchor_tokens=int(args.max_shared_suffix_tokens),
            )
        else:
            alignment_anchor = static_anchors[0]
        normalized_patches = normalize_patch_batch(
            batch["patch"],
            compressor_input_size,
            normalization_cfg,
            bool(args.resize_patch_to_compressor_input),
        )
        teacher_texts = build_teacher_texts_for_batch(batch, normalized_patches, args)
        teacher_output = text_teacher_hidden(
            llm,
            tokenizer,
            teacher_texts,
            device,
            int(args.max_text_tokens),
            int(args.teacher_layer),
            bool(args.fail_on_text_anchor_missing) and str(args.text_prompt_template) != "plain",
            bool(args.fail_on_text_max_length_hit)
            and (
                str(args.alignment_text_layout) == "values_shared_suffix"
                or str(args.text_prompt_template) != "plain"
            ),
            text_layout=str(args.alignment_text_layout),
            shared_suffix=str(args.shared_suffix),
            max_shared_suffix_tokens=int(args.max_shared_suffix_tokens),
            alignment_anchor=alignment_anchor,
        )
        take = min(int(teacher_output.hidden.shape[0]), local_target - local_count)
        selected_hidden = teacher_output.hidden[:take].detach().float()
        local_hidden.append(selected_hidden)
        anchor_group = int(step) - 1 if str(args.alignment_anchor_mode) == "probe" else 0
        local_anchor_groups.append(
            torch.full(
                (int(take), 1),
                float(anchor_group),
                device=selected_hidden.device,
                dtype=torch.float32,
            )
        )
        local_count += int(take)
    if not local_hidden:
        raise ValueError("Teacher whitening loader produced no hidden states.")
    gathered = gather_feature_rows(torch.cat(local_hidden, dim=0))
    gathered_anchor_groups = gather_feature_rows(torch.cat(local_anchor_groups, dim=0))
    if is_main_process():
        if gathered is None or gathered_anchor_groups is None:
            raise RuntimeError("Rank 0 did not receive teacher hidden states for whitening.")
        fitted_hidden = gathered[:requested_records]
        fitted_groups = gathered_anchor_groups[:requested_records, 0].long()
        covariance_residuals = torch.empty_like(fitted_hidden)
        for group in fitted_groups.unique():
            group_mask = fitted_groups.eq(group)
            group_hidden = fitted_hidden[group_mask]
            covariance_residuals[group_mask] = group_hidden - group_hidden.mean(dim=0, keepdim=True)
        fit_metrics = whitener.fit(
            fitted_hidden,
            covariance_residuals=covariance_residuals,
        )
        fit_metrics["anchor_groups"] = float(fitted_groups.unique().numel())
    else:
        fit_metrics = {}
    broadcast_module_state(whitener)
    fit_metrics = dict(broadcast_object_from_main(fit_metrics))
    whitener.fit_metrics = fit_metrics
    return fit_metrics


def save_checkpoint(
    path: Path,
    *,
    compressor: nn.Module,
    adapter: TensorPatchAlignmentAdapter,
    projector: AlignmentFeatureTransform | None,
    args: argparse.Namespace,
    metrics: Mapping[str, Any],
    compressor_config: Mapping[str, Any],
    save_compressor: bool,
) -> None:
    payload = {
        "adapter_state_dict": adapter.state_dict(),
        "compressor_config": dict(compressor_config),
        "args": redacted_args(args),
        "metrics": metrics,
    }
    if projector is not None:
        payload["alignment_feature_transform_mode"] = str(args.alignment_transform_mode)
        payload["alignment_feature_transform_state_dict"] = projector.state_dict()
        if isinstance(projector, AlignmentProjectionPair):
            payload["alignment_projector_state_dict"] = projector.state_dict()
    if save_compressor:
        payload["compressor_state_dict"] = compressor.state_dict()
    temporary_path = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def numeric_payload(prefix: str, metrics: Mapping[str, Any]) -> dict[str, float]:
    payload: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            payload[f"{prefix}/{key}"] = float(value)
    return payload


def alignment_wandb_payload(prefix: str, metrics: Mapping[str, Any]) -> dict[str, float]:
    summary_metrics = {
        key: value
        for key, value in metrics.items()
        if not key.startswith(("anchor_", "probe_macro_"))
        and (not key.startswith("contrastive_") or key == "contrastive_loss")
    }
    return numeric_payload(prefix, summary_metrics)


def fmt_metric(metrics: Mapping[str, Any], key: str) -> str:
    value = metrics.get(key)
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return "n/a"


def alignment_metric_summary(metrics: Mapping[str, Any]) -> str:
    return (
        f"i2t={fmt_metric(metrics, 'i2t_accuracy')} "
        f"t2i={fmt_metric(metrics, 't2i_accuracy')} "
        f"semantic_i2t={fmt_metric(metrics, 'semantic_i2t_accuracy')} "
        f"semantic_t2i={fmt_metric(metrics, 'semantic_t2i_accuracy')} "
        f"global_i2t={fmt_metric(metrics, 'global_i2t_accuracy')} "
        f"global_t2i={fmt_metric(metrics, 'global_t2i_accuracy')} "
        f"centered_global_i2t={fmt_metric(metrics, 'global_centered_i2t_accuracy')} "
        f"centered_global_t2i={fmt_metric(metrics, 'global_centered_t2i_accuracy')} "
        f"t2i_coverage={fmt_metric(metrics, 'global_t2i_candidate_coverage')} "
        f"mean_cos={fmt_metric(metrics, 'global_mean_alignment_native_cosine')} "
        f"raw_global_i2t={fmt_metric(metrics, 'global_hidden_uncentered_i2t_accuracy')} "
        f"raw_global_t2i={fmt_metric(metrics, 'global_hidden_uncentered_t2i_accuracy')} "
        f"raw_centered_global_i2t={fmt_metric(metrics, 'global_hidden_centered_i2t_accuracy')} "
        f"raw_centered_global_t2i={fmt_metric(metrics, 'global_hidden_centered_t2i_accuracy')} "
        f"collision={fmt_metric(metrics, 'semantic_collision_fraction')} "
        f"prompt_grad={fmt_metric(metrics, 'alignment_soft_prompt_gradient_norm')} "
        f"prompt_active={fmt_metric(metrics, 'alignment_soft_prompt_active_token_fraction')} "
        f"prompt_grad_entropy={fmt_metric(metrics, 'alignment_soft_prompt_gradient_entropy')} "
        f"recon={fmt_metric(metrics, 'reconstruction_loss')} "
        f"recon_rel={fmt_metric(metrics, 'reconstruction_relative_rmse_to_target_std')}"
    )


def build_wandb_config(args: argparse.Namespace, summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "experiment": {"name": str(args.run_name)},
        "data": {
            "hdf5_path": str(args.hdf5_path),
            "fields": parse_csv(args.fields),
            "field_sampling_mode": str(args.field_sampling_mode),
            "patch_size": int(args.patch_size),
            "train_records": int(args.train_records),
            "val_records": int(args.val_records),
            "test_records": int(args.test_records),
            "sample_indices": str(args.sample_indices),
            "time_indices": str(args.time_indices),
            "split_mode": str(args.split_mode),
            "split_train_ratio": float(args.split_train_ratio),
            "split_val_ratio": float(args.split_val_ratio),
            "split_test_ratio": float(args.split_test_ratio),
            "unique_records": bool(args.unique_records),
            "ensure_disjoint_records": bool(args.ensure_disjoint_records),
        },
        "model": {
            "name_or_path": str(args.model_name_or_path),
            "torch_dtype": str(args.torch_dtype),
            "trust_remote_code": bool(args.trust_remote_code),
        },
        "patch_alignment": {
            "encoder_source": str(args.encoder_source),
            "train_patch_ae": bool(args.train_patch_ae),
            "freeze_patch_ae_after_pretrain": bool(args.freeze_patch_ae_after_pretrain),
            "alignment_train_patch_ae": bool(getattr(args, "alignment_train_patch_ae", args.train_patch_ae)),
            "patch_ae_pretrain_epochs": int(args.patch_ae_pretrain_epochs),
            "patch_ae_pretrain_batch_size": int(args.patch_ae_pretrain_batch_size),
            "compressor_checkpoint": args.compressor_checkpoint,
            "resize_patch_to_compressor_input": bool(args.resize_patch_to_compressor_input),
            "adapter_type": str(args.adapter_type),
            "adapter_dim": int(args.adapter_dim),
            "query_tokens": int(args.query_tokens),
            "adapter_layers": int(args.adapter_layers),
            "adapter_heads": int(args.adapter_heads),
            "projection_dim": args.projection_dim,
            "alignment_transform": {
                "mode": str(args.alignment_transform_mode),
                "whitening": {
                    "records": int(args.alignment_whitening_records),
                    "dim": int(args.alignment_whitening_dim),
                    "shrinkage": float(args.alignment_whitening_shrinkage),
                    "epsilon": float(args.alignment_whitening_epsilon),
                    "max_condition_number": float(args.alignment_whitening_max_condition_number),
                },
            },
            "alignment_projection": {
                "enabled": str(args.alignment_transform_mode) == "projection",
                "dim": int(args.alignment_projection_dim),
                "hidden_dim": int(args.alignment_projection_hidden_dim),
                "layers": int(args.alignment_projection_layers),
                "dropout": float(args.alignment_projection_dropout),
                "shared": bool(args.alignment_projection_shared),
            },
            "dropout": float(args.dropout),
            "soft_prompt_scale": float(args.soft_prompt_scale),
            "temperature": float(args.temperature),
            "contrastive_loss_weight": float(args.contrastive_loss_weight),
            "contrastive_direction_weights": {
                "i2t": float(args.contrastive_i2t_weight),
                "t2i": float(args.contrastive_t2i_weight),
            },
            "centered_contrastive_loss_weight": float(args.centered_contrastive_loss_weight),
            "native_centered_contrastive_loss_weight": float(args.native_centered_contrastive_loss_weight),
            "mean_alignment_loss_weight": float(args.mean_alignment_loss_weight),
            "reconstruction_loss_weight": float(args.reconstruction_loss_weight),
            "alignment_patch_ae_lr_scale": float(args.alignment_patch_ae_lr_scale),
            "teacher_text_source": str(args.teacher_text_source),
            "alignment_text_layout": str(args.alignment_text_layout),
            "shared_suffix": str(args.shared_suffix),
            "alignment_anchor_mode": str(args.alignment_anchor_mode),
            "representation_suffix": str(args.representation_suffix),
            "probe_families": list(args.probe_families),
            "probe_region_size": int(args.probe_region_size),
            "evaluation_probe_count": int(args.evaluation_probe_count),
            "teacher_probe_warn_below_correlation": args.teacher_probe_warn_below_correlation,
            "teacher_probe_diagnostic_records": int(args.teacher_probe_diagnostic_records),
            "max_shared_suffix_tokens": int(args.max_shared_suffix_tokens),
            "fail_on_text_anchor_missing": bool(args.fail_on_text_anchor_missing),
            "fail_on_text_max_length_hit": bool(args.fail_on_text_max_length_hit),
            "global_retrieval_eval": bool(args.global_retrieval_eval),
            "global_retrieval_max_records": int(args.global_retrieval_max_records),
            "global_retrieval_chunk_size": int(args.global_retrieval_chunk_size),
            "text_prompt_template": str(args.text_prompt_template),
            "text_decimal_places": int(args.text_decimal_places),
            "max_text_tokens": int(args.max_text_tokens),
            "text_preflight_records": int(args.text_preflight_records),
            "teacher_layer": int(args.teacher_layer),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "eval_batch_size": int(args.eval_batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "grad_clip_norm": float(args.grad_clip_norm),
            "log_interval": int(args.log_interval),
            "patch_encoder": args.patch_encoder_config,
        },
        "run_summary": dict(summary),
        "wandb": {
            "enabled": bool(args.wandb_enabled),
            "api_key": args.wandb_api_key,
            "project": str(args.wandb_project),
            "entity": args.wandb_entity,
            "group": args.wandb_group,
            "tags": parse_csv(args.wandb_tags),
            "mode": str(args.wandb_mode),
            "log_model": bool(args.wandb_log_model),
        },
    }


def log_checkpoint_artifact(wandb_logger: WandbLogger, path: Path, name: str, artifact_type: str) -> None:
    if wandb_logger.run is None or wandb_logger._wandb is None or not path.exists():
        return
    artifact = wandb_logger._wandb.Artifact(name=name, type=artifact_type)
    artifact.add_file(str(path))
    wandb_logger.run.log_artifact(artifact)


def main() -> None:
    run_started_unix = time.time()
    run_started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    args = parse_args()
    config = load_yaml_mapping(args.config)
    args = apply_config_defaults(args, config)
    setup_distributed_from_env(args)
    if int(args.batch_size) * int(distributed_world_size()) <= 1:
        raise ValueError(
            "Symmetric InfoNCE requires at least two global candidates. Increase patch_alignment.batch_size "
            "or the distributed world size."
        )
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if args.hf_home:
        __import__("os").environ["HF_HOME"] = str(args.hf_home)

    if is_main_process():
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = Path(args.output_root) / f"{timestamp}_{args.run_name}"
        run_dir.mkdir(parents=True, exist_ok=False)
        dump_json(run_dir / "args.json", redacted_args(args))
    else:
        run_dir = None
    run_dir = Path(broadcast_object_from_main(str(run_dir)))
    if is_main_process():
        print(f"run_started_at={run_started_at} run_dir={run_dir}")

    checkpoint: dict[str, Any] | None = None
    field_sampling_mode = str(args.field_sampling_mode).lower()
    if str(args.encoder_source) == "checkpoint":
        checkpoint, compressor_config = load_checkpoint_and_config(args.compressor_checkpoint, args.compressor_config)
        state_dict = resolve_checkpoint_state_dict(checkpoint, args.compressor_checkpoint)
        if field_sampling_mode == "single":
            field_keys = parse_csv(args.fields) or resolve_checkpoint_field_keys(compressor_config)
            if not field_keys:
                raise ValueError("field_sampling_mode=single requires at least one field.")
            channel_count = checkpoint_channel_count(compressor_config)
            if channel_count is not None and int(channel_count) != 1:
                raise ValueError(
                    "field_sampling_mode=single requires a single-channel patch encoder checkpoint. "
                    f"Got checkpoint channel count {channel_count}."
                )
        else:
            field_keys = resolve_field_keys(args.fields, compressor_config)
        compressor = build_model(compressor_config)
        compressor.load_state_dict(state_dict)
    else:
        field_keys = resolve_field_keys(args.fields, None)
        encoder_field_keys = [field_keys[0]] if field_sampling_mode == "single" else field_keys
        compressor_config = build_patch_encoder_config(
            patch_encoder_cfg=args.patch_encoder_config,
            field_keys=encoder_field_keys,
            patch_size=int(args.patch_size),
        )
        compressor = build_model(compressor_config)
    # Probe construction must use the fields actually resolved from the checkpoint/config,
    # not only the optional CLI string. This matters when a checkpoint supplies its own field keys.
    args.fields = value_to_csv(field_keys)
    validate_field_shapes(args.hdf5_path, field_keys)
    compressor_input_size = tuple(int(dim) for dim in compressor_config["model"]["input_size"])
    configured_latent_grid = tuple(int(dim) for dim in compressor_config["model"]["latent_grid"])
    if str(args.adapter_type).lower() == "spatial_transformer":
        expected_spatial_tokens = int(configured_latent_grid[0] * configured_latent_grid[1])
        if int(args.query_tokens) != expected_spatial_tokens:
            raise ValueError(
                "spatial_transformer token/grid mismatch before LLM loading: "
                f"query_tokens={int(args.query_tokens)}, latent_grid={configured_latent_grid}, "
                f"expected={expected_spatial_tokens}."
            )
    normalization_cfg = dict(compressor_config.get("data", {}).get("dataset", {}).get("normalization", {}))
    validate_teacher_tensor_source(normalization_cfg, str(args.teacher_text_source))
    if not bool(args.resize_patch_to_compressor_input) and tuple(compressor_input_size) != (
        int(args.patch_size),
        int(args.patch_size),
    ):
        raise ValueError(
            "--no-resize-patch-to-compressor-input requires compressor input_size to match patch_size. "
            f"Got input_size={compressor_input_size}, patch_size={args.patch_size}."
        )
    if (
        str(args.alignment_anchor_mode) == "probe"
        and str(args.teacher_text_source).lower() == "normalized"
        and tuple(compressor_input_size) != (int(args.patch_size), int(args.patch_size))
    ):
        raise ValueError(
            "probe coordinates describe patch_alignment.patch_size, but normalized teacher text would use the "
            f"resized compressor input_size={compressor_input_size}. Use a patch-sized encoder or raw teacher text "
            "without value-changing normalization."
        )

    device = resolve_device(args.device)
    compressor.to(device)
    if bool(args.train_patch_ae):
        for parameter in compressor.parameters():
            parameter.requires_grad_(True)
    else:
        for parameter in compressor.parameters():
            parameter.requires_grad_(False)
        compressor.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        trust_remote_code=bool(args.trust_remote_code),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = None
    llm_num_hidden_layers = -1
    active_teacher_layers = None
    # Serialize large-model construction across ranks. Each process moves its
    # truncated backbone to its GPU before the next rank allocates host weights.
    for load_rank in range(distributed_world_size()):
        if distributed_rank() == load_rank:
            llm = AutoModel.from_pretrained(
                args.model_name_or_path,
                cache_dir=args.cache_dir,
                trust_remote_code=bool(args.trust_remote_code),
                dtype=dtype_from_name(str(args.torch_dtype)),
            )
            llm_num_hidden_layers = int(getattr(llm.config, "num_hidden_layers", -1))
            validate_teacher_hidden_state_index(int(args.teacher_layer), llm_num_hidden_layers)
            active_teacher_layers = truncate_llm_backbone_to_layer(llm, int(args.teacher_layer))
            if bool(args.llm_gradient_checkpointing):
                try:
                    llm.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False}
                    )
                except TypeError:
                    llm.gradient_checkpointing_enable()
                llm.config.use_cache = False
            llm.to(device)
        distributed_barrier()
    if llm is None:
        raise RuntimeError("The local rank failed to construct the frozen teacher backbone.")
    llm.eval()
    for parameter in llm.parameters():
        parameter.requires_grad_(False)
    llm_hidden_size = int(llm.config.hidden_size)
    projection_dim = int(args.projection_dim or llm_hidden_size)
    if projection_dim != llm_hidden_size:
        raise ValueError(
            "patch_alignment.projection_dim must be null or equal to the LLM hidden size "
            f"({llm_hidden_size}) because the text teacher side is fixed and unprojected."
        )
    if active_teacher_layers is not None:
        distributed_barrier()
    elif is_main_process() and int(args.teacher_layer) < int(llm_num_hidden_layers):
        print(
            "teacher_backbone_truncation=unavailable "
            "(backbone has no safe ModuleList named 'layers'); continuing with the full frozen LLM"
        )

    first_field = field_keys[0]
    record_fields = field_keys if field_sampling_mode == "single" else None
    split_plan = build_axis_split_plan(
        hdf5_path=args.hdf5_path,
        field=first_field,
        sample_indices=str(args.sample_indices),
        time_indices=str(args.time_indices),
        split_mode=str(args.split_mode),
        train_ratio=float(args.split_train_ratio),
        val_ratio=float(args.split_val_ratio),
        test_ratio=float(args.split_test_ratio),
        seed=int(args.seed),
    )
    train_records = build_patch_records(
        hdf5_path=args.hdf5_path,
        field=first_field,
        record_fields=record_fields,
        sample_indices=split_plan["samples"]["train"],
        time_indices=split_plan["times"]["train"],
        patch_size=int(args.patch_size),
        count=int(args.train_records),
        seed=int(args.seed),
        unique_records=bool(args.unique_records),
    )
    val_records = build_patch_records(
        hdf5_path=args.hdf5_path,
        field=first_field,
        record_fields=record_fields,
        sample_indices=split_plan["samples"]["val"],
        time_indices=split_plan["times"]["val"],
        patch_size=int(args.patch_size),
        count=int(args.val_records),
        seed=int(args.seed) + 1,
        unique_records=bool(args.unique_records),
    )
    test_records = build_patch_records(
        hdf5_path=args.hdf5_path,
        field=first_field,
        record_fields=record_fields,
        sample_indices=split_plan["samples"]["test"],
        time_indices=split_plan["times"]["test"],
        patch_size=int(args.patch_size),
        count=int(args.test_records),
        seed=int(args.seed) + 2,
        unique_records=bool(args.unique_records),
    )
    overlap_summary = split_overlap_summary(train_records, val_records, test_records)
    if bool(args.ensure_disjoint_records) and any(value > 0 for value in overlap_summary.values()):
        raise ValueError(f"Exact patch record overlap across splits is not allowed: {overlap_summary}")
    dataset_kwargs = {
        "hdf5_path": args.hdf5_path,
        "field_keys": field_keys,
        "patch_size": int(args.patch_size),
        "decimal_places": int(args.text_decimal_places),
        "prompt_template": str(args.text_prompt_template),
        "include_raw_text": (
            str(args.teacher_text_source).lower() == "raw"
            and str(args.alignment_text_layout) == "legacy_prompt"
        ),
    }
    train_dataset = PDEBenchPatchTextDataset(records=train_records, **dataset_kwargs)
    val_dataset = PDEBenchPatchTextDataset(records=val_records, **dataset_kwargs)
    test_dataset = PDEBenchPatchTextDataset(records=test_records, **dataset_kwargs)
    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=distributed_world_size(),
            rank=distributed_rank(),
            shuffle=True,
            drop_last=True,
        )
        if distributed_is_initialized()
        else None
    )
    train_loader = make_loader(
        train_dataset,
        int(args.batch_size),
        True,
        int(args.num_workers),
        sampler=train_sampler,
        drop_last=distributed_is_initialized(),
    )
    pretrain_loader = (
        make_loader(
            train_dataset,
            int(args.patch_ae_pretrain_batch_size),
            True,
            int(args.num_workers),
            sampler=train_sampler,
            drop_last=distributed_is_initialized(),
        )
        if bool(args.train_patch_ae) and int(args.patch_ae_pretrain_epochs) > 0
        else None
    )
    if len(train_loader) <= 0:
        raise ValueError(
            "The training DataLoader has zero batches. In distributed mode, train_records must provide at least "
            "one full global batch because drop_last=true."
        )
    if pretrain_loader is not None and len(pretrain_loader) <= 0:
        raise ValueError(
            "The patch-AE pretrain DataLoader has zero batches. Reduce "
            "patch_alignment.patch_ae_pretrain_batch_size or increase train_records."
        )
    val_sampler = (
        DistributedEvalSampler(val_dataset, distributed_world_size(), distributed_rank())
        if distributed_is_initialized()
        else None
    )
    test_sampler = (
        DistributedEvalSampler(test_dataset, distributed_world_size(), distributed_rank())
        if distributed_is_initialized()
        else None
    )
    if distributed_is_initialized() and (
        len(val_dataset) < distributed_world_size() or len(test_dataset) < distributed_world_size()
    ):
        raise ValueError(
            "Distributed evaluation requires val_records and test_records to be at least WORLD_SIZE "
            "so every rank participates in the same collectives."
        )
    distributed_eval_batch = int(args.eval_batch_size) * int(distributed_world_size())
    if distributed_is_initialized() and (
        len(val_dataset) % distributed_eval_batch != 0
        or len(test_dataset) % distributed_eval_batch != 0
    ):
        raise ValueError(
            "Distributed val_records and test_records must be divisible by "
            "WORLD_SIZE * eval_batch_size. This keeps every evaluation step aligned across ranks "
            "without dropping or padding samples. "
            f"Got val={len(val_dataset)}, test={len(test_dataset)}, "
            f"global_eval_batch={distributed_eval_batch}."
        )
    val_loader = make_loader(
        val_dataset,
        int(args.eval_batch_size),
        False,
        int(args.num_workers),
        sampler=val_sampler,
    )
    test_loader = make_loader(
        test_dataset,
        int(args.eval_batch_size),
        False,
        int(args.num_workers),
        sampler=test_sampler,
    )
    whitening_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=distributed_world_size(),
            rank=distributed_rank(),
            shuffle=False,
            drop_last=False,
        )
        if distributed_is_initialized() and str(args.alignment_transform_mode) == "whitening"
        else None
    )
    whitening_loader = (
        make_loader(
            train_dataset,
            int(args.eval_batch_size),
            False,
            int(args.num_workers),
            sampler=whitening_sampler,
        )
        if str(args.alignment_transform_mode) == "whitening"
        else None
    )
    probe_contract_anchor_bank = probe_contract_anchors(tokenizer, args)
    if is_main_process() and probe_contract_anchor_bank:
        dump_json(
            run_dir / "probe_contract.json",
            {
                "family_count": len(parse_csv(args.probe_families)),
                "template_counts": {
                    family: int(PROBE_TEMPLATE_COUNTS[family])
                    for family in args.probe_families
                },
                "anchors": [
                    {
                        "name": anchor.name,
                        "text": anchor.text,
                        "token_ids": list(anchor.token_ids),
                        "token_count": len(anchor.token_ids),
                        "probe_family": anchor.probe_family,
                        "probe_template_index": anchor.probe_template_index,
                        "probe_parameters": list(anchor.probe_parameters),
                    }
                    for anchor in probe_contract_anchor_bank
                ],
            },
        )
    text_preflight_metrics = preflight_teacher_text_tokenization(
        dataset=train_dataset,
        tokenizer=tokenizer,
        args=args,
        compressor_input_size=compressor_input_size,
        normalization_cfg=normalization_cfg,
    )
    if is_main_process() and text_preflight_metrics:
        dump_json(run_dir / "probe_tokenization_preflight.json", text_preflight_metrics)
    if text_preflight_metrics:
        distributed_barrier()
    training_anchors = (
        alignment_anchors_from_args(tokenizer, args, evaluation=False)
        if str(args.alignment_text_layout) == "values_shared_suffix"
        else []
    )
    evaluation_anchors = (
        alignment_anchors_from_args(tokenizer, args, evaluation=True)
        if str(args.alignment_text_layout) == "values_shared_suffix"
        else []
    )
    teacher_probe_preflight = preflight_teacher_probe_semantics(
        dataset=train_dataset,
        llm=llm,
        tokenizer=tokenizer,
        device=device,
        args=args,
        compressor_input_size=compressor_input_size,
        normalization_cfg=normalization_cfg,
    )
    teacher_probe_warnings = teacher_probe_preflight_warnings(
        teacher_probe_preflight,
        args.teacher_probe_warn_below_correlation,
    )
    if is_main_process() and teacher_probe_preflight:
        dump_json(run_dir / "teacher_probe_preflight.json", teacher_probe_preflight)
        macro = teacher_probe_preflight["macro"]
        print(
            "teacher_probe_preflight "
            f"records={teacher_probe_preflight['record_limit']} "
            f"pair_cos={macro.get('hidden_pairwise_cosine', float('nan')):.6f} "
            f"target_corr_median={macro.get('hidden_similarity_vs_negative_target_distance_pearson_median', float('nan')):.4f} "
            f"nearest_target_error={macro.get('nearest_hidden_target_abs_error', float('nan')):.6g} "
            f"target_unique={macro.get('semantic_target_unique_fraction', float('nan')):.4f}"
        )
        if teacher_probe_warnings:
            print(
                "WARNING teacher_probe_preflight "
                f"family median correlation below {float(args.teacher_probe_warn_below_correlation):.4f}: "
                + ", ".join(teacher_probe_warnings)
            )
    if teacher_probe_preflight:
        distributed_barrier()

    with torch.no_grad():
        probe_patch = normalize_patch_batch(
            torch.stack([train_dataset[0]["patch"]], dim=0),
            compressor_input_size,
            normalization_cfg,
            bool(args.resize_patch_to_compressor_input),
        ).to(device)
        probe_latent_map = compressor.encode(probe_patch)["latent_map"]
        latent_channels = int(probe_latent_map.shape[1])
        latent_grid = tuple(int(dim) for dim in probe_latent_map.shape[-2:])
    # Preflight reads happen in the rank process. Workers must open their own clean handles lazily.
    train_dataset.close()
    val_dataset.close()
    test_dataset.close()
    adapter = TensorPatchAlignmentAdapter(
        latent_channels=latent_channels,
        latent_grid=latent_grid,
        adapter_dim=int(args.adapter_dim),
        projection_dim=projection_dim,
        dropout=float(args.dropout),
        adapter_type=str(args.adapter_type),
        query_tokens=int(args.query_tokens),
        adapter_layers=int(args.adapter_layers),
        adapter_heads=int(args.adapter_heads),
        soft_prompt_scale=float(args.soft_prompt_scale),
    ).to(device)
    alignment_projector: AlignmentFeatureTransform | None = None
    if str(args.alignment_transform_mode) == "projection":
        alignment_projector = AlignmentProjectionPair(
            input_dim=llm_hidden_size,
            output_dim=int(args.alignment_projection_dim),
            hidden_dim=int(args.alignment_projection_hidden_dim),
            layers=int(args.alignment_projection_layers),
            dropout=float(args.alignment_projection_dropout),
            shared=bool(args.alignment_projection_shared),
        ).to(device)
    elif str(args.alignment_transform_mode) == "whitening":
        alignment_projector = FixedTeacherWhitening(
            hidden_dim=llm_hidden_size,
            shrinkage=float(args.alignment_whitening_shrinkage),
            epsilon=float(args.alignment_whitening_epsilon),
            output_dim=int(args.alignment_whitening_dim),
            max_condition_number=float(args.alignment_whitening_max_condition_number),
        ).to(device)
    broadcast_module_state(compressor)
    broadcast_module_state(adapter)
    broadcast_module_state(alignment_projector)

    whitening_metrics: dict[str, float] = {}
    if isinstance(alignment_projector, FixedTeacherWhitening):
        if whitening_loader is None:
            raise RuntimeError("Whitening mode did not create a statistics DataLoader.")
        whitening_metrics = fit_teacher_whitening(
            whitener=alignment_projector,
            loader=whitening_loader,
            llm=llm,
            tokenizer=tokenizer,
            device=device,
            args=args,
            compressor_input_size=compressor_input_size,
            normalization_cfg=normalization_cfg,
        )
        if is_main_process():
            dump_json(run_dir / "alignment_whitening.json", whitening_metrics)
    # The whitening dataset is used once; do not retain its persistent workers during training.
    whitening_loader = None

    args.alignment_train_patch_ae = bool(args.train_patch_ae) and not (
        bool(args.freeze_patch_ae_after_pretrain) and int(args.patch_ae_pretrain_epochs) > 0
    )

    run_summary = {
        "started_at": run_started_at,
        "hdf5_path": str(args.hdf5_path),
        "field_keys": field_keys,
        "field_sampling_mode": field_sampling_mode,
        "encoder_field_keys": [field_keys[0]] if field_sampling_mode == "single" else field_keys,
        "patch_size": int(args.patch_size),
        "train_records": len(train_dataset),
        "val_records": len(val_dataset),
        "test_records": len(test_dataset),
        "split_mode": str(args.split_mode),
        "split_train_ratio": float(args.split_train_ratio),
        "split_val_ratio": float(args.split_val_ratio),
        "split_test_ratio": float(args.split_test_ratio),
        "unique_records": bool(args.unique_records),
        "ensure_disjoint_records": bool(args.ensure_disjoint_records),
        "split_plan": {
            "mode": str(split_plan["mode"]),
            "available_sample_count": int(split_plan["available_sample_count"]),
            "available_time_count": int(split_plan["available_time_count"]),
            "train_sample_count": len(split_plan["samples"]["train"]),
            "val_sample_count": len(split_plan["samples"]["val"]),
            "test_sample_count": len(split_plan["samples"]["test"]),
            "train_time_count": len(split_plan["times"]["train"]),
            "val_time_count": len(split_plan["times"]["val"]),
            "test_time_count": len(split_plan["times"]["test"]),
            "train_sample_preview": list(split_plan["samples"]["train"][:16]),
            "val_sample_preview": list(split_plan["samples"]["val"][:16]),
            "test_sample_preview": list(split_plan["samples"]["test"][:16]),
            "train_time_preview": list(split_plan["times"]["train"][:16]),
            "val_time_preview": list(split_plan["times"]["val"][:16]),
            "test_time_preview": list(split_plan["times"]["test"][:16]),
        },
        "record_summary": {
            "train": summarize_records(train_records),
            "val": summarize_records(val_records),
            "test": summarize_records(test_records),
            "overlap": dict(overlap_summary),
        },
        "distributed_evaluation": {
            "exact_nonpadding_shards": bool(distributed_is_initialized()),
            "val_local_records": len(val_sampler) if val_sampler is not None else len(val_dataset),
            "test_local_records": len(test_sampler) if test_sampler is not None else len(test_dataset),
        },
        "encoder_source": str(args.encoder_source),
        "compressor_checkpoint": str(args.compressor_checkpoint) if args.compressor_checkpoint else None,
        "train_patch_ae": bool(args.train_patch_ae),
        "freeze_patch_ae_after_pretrain": bool(args.freeze_patch_ae_after_pretrain),
        "alignment_train_patch_ae": bool(args.alignment_train_patch_ae),
        "patch_ae_pretrain_epochs": int(args.patch_ae_pretrain_epochs),
        "patch_ae_pretrain_batch_size": int(args.patch_ae_pretrain_batch_size),
        "alignment_patch_ae_lr_scale": float(args.alignment_patch_ae_lr_scale),
        "reconstruction_loss_weight": float(args.reconstruction_loss_weight),
        "resize_patch_to_compressor_input": bool(args.resize_patch_to_compressor_input),
        "compressor_input_size": list(compressor_input_size),
        "latent_channels": int(latent_channels),
        "latent_grid": list(latent_grid),
        "latent_token_count": int(latent_grid[0] * latent_grid[1]),
        "llm_hidden_size": int(llm_hidden_size),
        "llm_num_hidden_layers": int(llm_num_hidden_layers),
        "llm_gradient_checkpointing": bool(args.llm_gradient_checkpointing),
        "active_teacher_layers": (
            int(active_teacher_layers) if active_teacher_layers is not None else int(llm_num_hidden_layers)
        ),
        "projection_dim": int(projection_dim),
        "normalization": dict(normalization_cfg),
        "alignment_transform": {
            "mode": str(args.alignment_transform_mode),
            "parameters": (
                sum(parameter.numel() for parameter in alignment_projector.parameters())
                if alignment_projector is not None
                else 0
            ),
            "whitening": {
                "records_requested": int(args.alignment_whitening_records),
                "dim": int(args.alignment_whitening_dim),
                "shrinkage": float(args.alignment_whitening_shrinkage),
                "epsilon": float(args.alignment_whitening_epsilon),
                "max_condition_number": float(args.alignment_whitening_max_condition_number),
                "fit": dict(whitening_metrics),
            },
        },
        "native_centered_contrastive_loss_weight": float(args.native_centered_contrastive_loss_weight),
        "mean_alignment_loss_weight": float(args.mean_alignment_loss_weight),
        "alignment_projection": {
            "enabled": str(args.alignment_transform_mode) == "projection",
            "dim": int(args.alignment_projection_dim),
            "hidden_dim": int(args.alignment_projection_hidden_dim),
            "layers": int(args.alignment_projection_layers),
            "dropout": float(args.alignment_projection_dropout),
            "shared": bool(args.alignment_projection_shared),
            "parameters": (
                sum(parameter.numel() for parameter in alignment_projector.parameters())
                if isinstance(alignment_projector, AlignmentProjectionPair)
                else 0
            ),
        },
        "teacher_layer": int(args.teacher_layer),
        "teacher_transformer_blocks_applied": int(args.teacher_layer),
        "teacher_text_source": str(args.teacher_text_source),
        "alignment_text_layout": str(args.alignment_text_layout),
        "alignment_anchor_mode": str(args.alignment_anchor_mode),
        "representation_suffix": str(args.representation_suffix),
        "probe_families": list(args.probe_families),
        "probe_region_size": int(args.probe_region_size),
        "evaluation_probe_count": int(args.evaluation_probe_count),
        "teacher_probe_warn_below_correlation": args.teacher_probe_warn_below_correlation,
        "teacher_probe_diagnostic_records": int(args.teacher_probe_diagnostic_records),
        "teacher_probe_warnings": list(teacher_probe_warnings),
        "probe_contract_anchors": [
            {
                "name": anchor.name,
                "text": anchor.text,
                "token_ids": list(anchor.token_ids),
                "token_count": len(anchor.token_ids),
                "probe_family": anchor.probe_family,
                "probe_template_index": anchor.probe_template_index,
                "probe_parameters": list(anchor.probe_parameters),
            }
            for anchor in probe_contract_anchor_bank
        ],
        "training_anchors": [
            {
                "name": anchor.name,
                "mode": anchor.mode,
                "text": anchor.text,
                "token_ids": list(anchor.token_ids),
                "token_count": len(anchor.token_ids),
                "probe_family": anchor.probe_family,
                "probe_template_index": anchor.probe_template_index,
                "probe_parameters": list(anchor.probe_parameters),
            }
            for anchor in training_anchors
        ],
        "evaluation_anchors": [
            {
                "name": anchor.name,
                "mode": anchor.mode,
                "text": anchor.text,
                "token_ids": list(anchor.token_ids),
                "token_count": len(anchor.token_ids),
                "probe_family": anchor.probe_family,
                "probe_template_index": anchor.probe_template_index,
                "probe_parameters": list(anchor.probe_parameters),
            }
            for anchor in evaluation_anchors
        ],
        "anchor_aggregation": {
            "training_schedule": "one shared probe per batch; families and configured templates cycle uniformly",
            "primary_validation": "mean over fixed probes only when probe mode is selected",
            "global_retrieval": "separate_candidate_library_per_anchor",
            "probe_target_visibility": "internal_diagnostic_only_never_appended_to_llm_input",
            "negative_policy": "exclude_quantized_equal_probe_targets_keep_paired_positive",
            "strict_retrieval_policy": "argmax_over_complete_unmasked_candidate_library",
            "alignment_hidden": f"{args.alignment_transform_mode}_shared_anchor_hidden",
            "primary_embedding_centering": "separate_branch_ddp_global_same_probe_mean",
            "centered_retrieval": (
                "ddp_global_batch_primary_residual_loss_and_diagnostic"
                if float(args.centered_contrastive_loss_weight) > 0.0
                else "diagnostic_only"
            ),
            "mean_alignment": "ddp_global_same_probe_transformed_and_native_branch_means",
        },
        "max_shared_suffix_tokens": int(args.max_shared_suffix_tokens),
        "contrastive_loss_weight": float(args.contrastive_loss_weight),
        "contrastive_direction_weights": {
            "i2t": float(args.contrastive_i2t_weight),
            "t2i": float(args.contrastive_t2i_weight),
        },
        "centered_contrastive_loss_weight": float(args.centered_contrastive_loss_weight),
        "fail_on_text_anchor_missing": bool(args.fail_on_text_anchor_missing),
        "fail_on_text_max_length_hit": bool(args.fail_on_text_max_length_hit),
        "global_retrieval_eval": bool(args.global_retrieval_eval),
        "global_retrieval_max_records": int(args.global_retrieval_max_records),
        "global_retrieval_chunk_size": int(args.global_retrieval_chunk_size),
        "checkpoint_selection": "i2t_primary_global_uncentered_plus_batch_centered_native_and_mean_losses",
        "text_preflight_records": int(args.text_preflight_records),
        "text_preflight": dict(text_preflight_metrics),
        "teacher_probe_preflight": dict(teacher_probe_preflight),
        "adapter_type": str(args.adapter_type),
        "query_tokens": int(args.query_tokens),
        "adapter_layers": int(args.adapter_layers),
        "adapter_heads": int(args.adapter_heads),
        "soft_prompt_scale": float(args.soft_prompt_scale),
        "alignment_mode": (
            "input_soft_prompt_shared_anchor_hidden_probe"
            if str(args.alignment_anchor_mode) == "probe"
            else "input_soft_prompt_shared_suffix_hidden"
            if str(args.alignment_text_layout) == "values_shared_suffix"
            else "input_soft_prompt_hidden"
        ),
        "distributed": {
            "enabled": bool(distributed_is_initialized()),
            "rank": int(distributed_rank()),
            "world_size": int(distributed_world_size()),
            "local_rank": int(getattr(args, "local_rank", 0)),
            "per_rank_batch_size": int(args.batch_size),
            "train_contrastive_candidates": int(args.batch_size) * int(distributed_world_size()),
            "gradient_sync": "manual_all_reduce",
            "negative_gather": "differentiable_embedding_all_gather",
            "train_drop_last": bool(distributed_is_initialized()),
        },
        "adapter_parameters": sum(parameter.numel() for parameter in adapter.parameters() if parameter.requires_grad),
        "alignment_transform_trainable_parameters": (
            sum(parameter.numel() for parameter in alignment_projector.parameters() if parameter.requires_grad)
            if alignment_projector is not None
            else 0
        ),
        "alignment_projector_parameters": (
            sum(parameter.numel() for parameter in alignment_projector.parameters() if parameter.requires_grad)
            if isinstance(alignment_projector, AlignmentProjectionPair)
            else 0
        ),
        "pretrain_trainable_compressor_parameters": sum(
            parameter.numel() for parameter in compressor.parameters() if parameter.requires_grad
        ),
        "alignment_trainable_compressor_parameters": (
            sum(parameter.numel() for parameter in compressor.parameters()) if bool(args.alignment_train_patch_ae) else 0
        ),
    }
    if is_main_process():
        dump_json(run_dir / "run_summary.json", run_summary)
        print(
            "patch_split "
            f"mode={run_summary['split_mode']} "
            f"train_samples={run_summary['split_plan']['train_sample_count']} "
            f"val_samples={run_summary['split_plan']['val_sample_count']} "
            f"test_samples={run_summary['split_plan']['test_sample_count']} "
            f"overlap={run_summary['record_summary']['overlap']}"
        )
        print(
            "alignment_objective "
            f"text_layout={str(args.alignment_text_layout)} "
            f"train_anchors={','.join(anchor.name for anchor in training_anchors) or 'legacy'} "
            f"eval_anchors={','.join(anchor.name for anchor in evaluation_anchors) or 'legacy'} "
            f"normalization={str(normalization_cfg.get('mode', 'none'))} "
            f"teacher_source={str(args.teacher_text_source)} "
            "embedding_centering=ddp_global_same_probe "
            f"contrastive_weight={float(args.contrastive_loss_weight):.4g} "
            f"direction_i2t={float(args.contrastive_i2t_weight):.4g} "
            f"direction_t2i={float(args.contrastive_t2i_weight):.4g} "
            f"centered_contrastive_weight={float(args.centered_contrastive_loss_weight):.4g} "
            f"native_centered_weight={float(args.native_centered_contrastive_loss_weight):.4g} "
            f"mean_alignment_weight={float(args.mean_alignment_loss_weight):.4g} "
            f"alignment_transform={str(args.alignment_transform_mode)} "
            f"projection_dim={int(args.alignment_projection_dim) if isinstance(alignment_projector, AlignmentProjectionPair) else 'n/a'} "
            f"whitening_records={int(whitening_metrics.get('records', 0.0)) if isinstance(alignment_projector, FixedTeacherWhitening) else 'n/a'} "
            f"whitening_dim={int(whitening_metrics.get('output_dim', 0.0)) if isinstance(alignment_projector, FixedTeacherWhitening) else 'n/a'} "
            f"whitening_variance={whitening_metrics.get('explained_variance_ratio', 'n/a')} "
            f"whitening_condition={whitening_metrics.get('regularized_condition_number', 'n/a')} "
            f"active_teacher_layers={int(active_teacher_layers) if active_teacher_layers is not None else int(llm_num_hidden_layers)} "
            f"distributed={bool(distributed_is_initialized())} "
            f"world_size={int(distributed_world_size())} "
            f"train_candidates={int(args.batch_size) * int(distributed_world_size())} "
            f"global_eval={bool(args.global_retrieval_eval)}"
        )
        if text_preflight_metrics:
            print(
                "teacher_text_preflight "
                f"token_mean={text_preflight_metrics.get('token_count_mean', 0.0):.1f} "
                f"token_max={text_preflight_metrics.get('token_count_max', 0.0):.0f} "
                f"content_token_max={text_preflight_metrics.get('content_token_count_max', 0.0):.0f} "
                f"suffix_tokens={text_preflight_metrics.get('suffix_token_count', 0.0):.0f} "
                f"content_truncated={text_preflight_metrics.get('content_truncated_fraction', 0.0):.3f} "
                f"anchor_missing={text_preflight_metrics.get('anchor_missing_fraction', 0.0):.3f} "
                f"max_len_hit={text_preflight_metrics.get('max_length_hit_fraction', 0.0):.3f} "
                f"probe_anchors={text_preflight_metrics.get('probe_contract_anchor_count', 0.0):.0f}"
            )
    wandb_logger = WandbLogger(config=build_wandb_config(args, run_summary), run_dir=run_dir) if is_main_process() else None
    global_step = 0

    metrics_history: dict[str, Any] = {}
    best_patch_ae_val = float("inf")
    best_patch_ae_epoch = 0
    if bool(args.train_patch_ae) and int(args.patch_ae_pretrain_epochs) > 0:
        if pretrain_loader is None:
            raise RuntimeError("Patch-AE pretraining was enabled without a pretrain DataLoader.")
        compressor_optimizer = torch.optim.AdamW(
            [parameter for parameter in compressor.parameters() if parameter.requires_grad],
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
        )
        for pretrain_epoch in range(1, int(args.patch_ae_pretrain_epochs) + 1):
            pretrain_metrics, global_step = pretrain_patch_encoder_one_epoch(
                compressor=compressor,
                loader=pretrain_loader,
                optimizer=compressor_optimizer,
                device=device,
                args=args,
                compressor_input_size=compressor_input_size,
                normalization_cfg=normalization_cfg,
                epoch=pretrain_epoch,
                wandb_logger=wandb_logger,
                global_step=global_step,
            )
            # In distributed runs every rank must keep doing comparable work before a NCCL barrier.
            # If only rank 0 runs a long validation pass, the other ranks can time out while waiting.
            pretrain_val_metrics = evaluate_patch_encoder_reconstruction(
                compressor=compressor,
                loader=val_loader,
                device=device,
                args=args,
                compressor_input_size=compressor_input_size,
                normalization_cfg=normalization_cfg,
            )
            if is_main_process():
                pretrain_epoch_metrics = {
                    "train": pretrain_metrics,
                    "val": pretrain_val_metrics,
                }
                metrics_history[f"patch_ae_pretrain_{pretrain_epoch:04d}"] = pretrain_epoch_metrics
                dump_json(run_dir / "metrics_latest.json", metrics_history)
                if wandb_logger is not None:
                    wandb_logger.log(
                        {
                            "patch_ae_pretrain/reconstruction_loss": float(pretrain_metrics["reconstruction_loss"]),
                            "patch_ae_pretrain/val_reconstruction_loss": float(
                                pretrain_val_metrics["reconstruction_loss"]
                            ),
                            "patch_ae_pretrain/relative_rmse_to_target_std": float(
                                pretrain_metrics.get("relative_rmse_to_target_std", float("nan"))
                            ),
                            "patch_ae_pretrain/val_relative_rmse_to_target_std": float(
                                pretrain_val_metrics.get("relative_rmse_to_target_std", float("nan"))
                            ),
                            "patch_ae_pretrain/target_std": float(pretrain_metrics.get("target_std", float("nan"))),
                            "patch_ae_pretrain/val_target_std": float(
                                pretrain_val_metrics.get("target_std", float("nan"))
                            ),
                            "patch_ae_pretrain/epoch": float(pretrain_epoch),
                            "patch_ae_pretrain/lr": float(compressor_optimizer.param_groups[0]["lr"]),
                        },
                        step=global_step,
                    )
                save_checkpoint(
                    run_dir / "patch_ae_pretrain_last.pt",
                    compressor=compressor,
                    adapter=adapter,
                    projector=alignment_projector,
                    args=args,
                    metrics=pretrain_epoch_metrics,
                    compressor_config=compressor_config,
                    save_compressor=True,
                )
                current_patch_ae_val = float(pretrain_val_metrics["reconstruction_loss"])
                if current_patch_ae_val < best_patch_ae_val:
                    best_patch_ae_val = current_patch_ae_val
                    best_patch_ae_epoch = int(pretrain_epoch)
                    save_checkpoint(
                        run_dir / "patch_ae_pretrain_best.pt",
                        compressor=compressor,
                        adapter=adapter,
                        projector=alignment_projector,
                        args=args,
                        metrics=pretrain_epoch_metrics,
                        compressor_config=compressor_config,
                        save_compressor=True,
                    )
                print(
                    f"patch_ae_pretrain_epoch={pretrain_epoch:04d} "
                    f"train_recon={pretrain_metrics['reconstruction_loss']:.4f} "
                    f"train_rel={pretrain_metrics.get('relative_rmse_to_target_std', float('nan')):.4f} "
                    f"val_recon={pretrain_val_metrics['reconstruction_loss']:.4f} "
                    f"val_rel={pretrain_val_metrics.get('relative_rmse_to_target_std', float('nan')):.4f}"
                )
            distributed_barrier(f"patch_ae_pretrain_epoch_{pretrain_epoch:04d}_saved")

        best_patch_ae = torch.load(run_dir / "patch_ae_pretrain_best.pt", map_location=device)
        compressor.load_state_dict(best_patch_ae["compressor_state_dict"])
        if is_main_process():
            metrics_history["patch_ae_pretrain_best"] = {
                "epoch": int(best_patch_ae_epoch),
                "val_reconstruction_loss": float(best_patch_ae_val),
            }
            dump_json(run_dir / "metrics_latest.json", metrics_history)
            print(
                f"patch_ae_pretrain_restored_best epoch={best_patch_ae_epoch:04d} "
                f"val_recon={best_patch_ae_val:.6g}"
            )
        distributed_barrier("patch_ae_pretrain_best_restored")
        # Release persistent pretrain workers and their HDF5 handles before alignment starts.
        pretrain_loader = None

    if bool(args.alignment_train_patch_ae):
        for parameter in compressor.parameters():
            parameter.requires_grad_(True)
    else:
        for parameter in compressor.parameters():
            parameter.requires_grad_(False)
        compressor.eval()
    bridge_parameters = list(adapter.parameters())
    if alignment_projector is not None:
        bridge_parameters += [parameter for parameter in alignment_projector.parameters() if parameter.requires_grad]
    optimizer_groups: list[dict[str, Any]] = [
        {
            "name": "alignment_bridge",
            "params": bridge_parameters,
            "lr": float(args.lr),
        }
    ]
    if bool(args.alignment_train_patch_ae):
        optimizer_groups.append(
            {
                "name": "patch_ae",
                "params": [parameter for parameter in compressor.parameters() if parameter.requires_grad],
                "lr": float(args.lr) * float(args.alignment_patch_ae_lr_scale),
            }
        )
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=float(args.weight_decay))

    best_val_selection = float("inf")
    best_val_metric = ""
    best_epoch = 0
    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = train_one_epoch(
            compressor=compressor,
            adapter=adapter,
            projector=alignment_projector,
            llm=llm,
            tokenizer=tokenizer,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            args=args,
            compressor_input_size=compressor_input_size,
            normalization_cfg=normalization_cfg,
            epoch=epoch,
        )
        global_step += len(train_loader)
        val_metrics = evaluate_anchor_bank(
            compressor=compressor,
            adapter=adapter,
            projector=alignment_projector,
            llm=llm,
            tokenizer=tokenizer,
            loader=val_loader,
            device=device,
            args=args,
            compressor_input_size=compressor_input_size,
            normalization_cfg=normalization_cfg,
        )
        if is_main_process():
            epoch_metrics = {"epoch": int(epoch), "train": train_metrics, "val": val_metrics}
            metrics_history[f"epoch_{epoch:04d}"] = epoch_metrics
            dump_json(run_dir / "metrics_latest.json", metrics_history)
            save_checkpoint(
                run_dir / "alignment_last.pt",
                compressor=compressor,
                adapter=adapter,
                projector=alignment_projector,
                args=args,
                metrics=epoch_metrics,
                compressor_config=compressor_config,
                save_compressor=True,
            )
            selection_metric, selection_value = checkpoint_selection_value(val_metrics, args)
            if best_val_metric and selection_metric != best_val_metric:
                raise RuntimeError(
                    "Validation checkpoint metric changed during one run: "
                    f"{best_val_metric!r} -> {selection_metric!r}."
                )
            best_val_metric = selection_metric
            if selection_value < best_val_selection:
                best_val_selection = selection_value
                best_epoch = int(epoch)
                save_checkpoint(
                    run_dir / "alignment_best.pt",
                    compressor=compressor,
                    adapter=adapter,
                    projector=alignment_projector,
                    args=args,
                    metrics=epoch_metrics,
                    compressor_config=compressor_config,
                    save_compressor=True,
                )
            wandb_payload = {
                "epoch": float(epoch),
                "lr/alignment_bridge": float(optimizer.param_groups[0]["lr"]),
                "lr/patch_ae": float(
                    next(
                        (group["lr"] for group in optimizer.param_groups if group.get("name") == "patch_ae"),
                        0.0,
                    )
                ),
                "best_val/selection_value": float(best_val_selection),
                "best_val/epoch": float(best_epoch),
            }
            wandb_payload.update(alignment_wandb_payload("train", train_metrics))
            wandb_payload.update(alignment_wandb_payload("val", val_metrics))
            if wandb_logger is not None:
                wandb_logger.log(wandb_payload, step=global_step)
            print(
                f"epoch={epoch:04d} train_loss={train_metrics['loss']:.4f} "
                f"train[{alignment_metric_summary(train_metrics)}] "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val[{alignment_metric_summary(val_metrics)}] "
                f"select={selection_metric}:{selection_value:.4f}"
            )
        distributed_barrier(f"alignment_epoch_{epoch:04d}_checkpointed")

    best_checkpoint = torch.load(run_dir / "alignment_best.pt", map_location=device)
    adapter.load_state_dict(best_checkpoint["adapter_state_dict"])
    if alignment_projector is not None:
        transform_state = best_checkpoint.get("alignment_feature_transform_state_dict")
        if transform_state is None and isinstance(alignment_projector, AlignmentProjectionPair):
            transform_state = best_checkpoint.get("alignment_projector_state_dict")
        if not isinstance(transform_state, Mapping):
            raise ValueError(
                f"{args.alignment_transform_mode}-mode alignment checkpoint is missing "
                "alignment_feature_transform_state_dict."
            )
        alignment_projector.load_state_dict(transform_state)
    if "compressor_state_dict" in best_checkpoint:
        compressor.load_state_dict(best_checkpoint["compressor_state_dict"])
    test_metrics = evaluate_anchor_bank(
        compressor=compressor,
        adapter=adapter,
        projector=alignment_projector,
        llm=llm,
        tokenizer=tokenizer,
        loader=test_loader,
        device=device,
        args=args,
        compressor_input_size=compressor_input_size,
        normalization_cfg=normalization_cfg,
    )
    if is_main_process():
        metrics_history["best"] = {
            "epoch": int(best_epoch),
            "metric": str(best_val_metric),
            "val_selection_value": float(best_val_selection),
        }
        metrics_history["test"] = test_metrics
        dump_json(run_dir / "metrics_latest.json", metrics_history)
        dump_json(run_dir / "test_metrics.json", test_metrics)
        if wandb_logger is not None:
            wandb_logger.log(alignment_wandb_payload("test", test_metrics), step=global_step)
            if bool(args.wandb_log_model):
                log_checkpoint_artifact(
                    wandb_logger,
                    run_dir / "patch_ae_pretrain_last.pt",
                    f"{args.run_name}-patch-ae-pretrain-last",
                    "patch-ae-checkpoint",
                )
                log_checkpoint_artifact(
                    wandb_logger,
                    run_dir / "alignment_best.pt",
                    f"{args.run_name}-alignment-best",
                    "patch-alignment-checkpoint",
                )
                log_checkpoint_artifact(
                    wandb_logger,
                    run_dir / "alignment_last.pt",
                    f"{args.run_name}-alignment-last",
                    "patch-alignment-checkpoint",
                )
            wandb_logger.finish()
        run_finished_unix = time.time()
        run_summary["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        run_summary["duration_seconds"] = float(run_finished_unix - run_started_unix)
        dump_json(run_dir / "run_summary.json", run_summary)
        print(f"Run directory: {run_dir}")
        print(
            f"best_epoch={best_epoch} test_loss={float(test_metrics['loss']):.4f} "
            f"test[{alignment_metric_summary(test_metrics)}] "
            f"duration_hours={run_summary['duration_seconds'] / 3600.0:.2f}"
        )
    distributed_barrier("final_test_written")
    cleanup_distributed()


if __name__ == "__main__":
    try:
        main()
    finally:
        if distributed_is_initialized():
            dist.destroy_process_group()
