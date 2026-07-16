from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tensor_compression.config import load_config  # noqa: E402
from tensor_compression.data.normalization import normalize_tensor  # noqa: E402
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
    from transformers import AutoModelForCausalLM, AutoTokenizer
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
    final_hidden: torch.Tensor
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
    probe_answers: tuple[str, ...] = ()


PROBE_FAMILIES = (
    "point_sign",
    "point_relation",
    "region_mean_relation",
    "region_range_relation",
    "directional_change",
)
PROBE_TEMPLATES_PER_FAMILY = 4
PROBE_FORBIDDEN_INPUT_MARKERS = ("answer:", "a or b", "a/b", "choose from", "options:", "?")


def parse_csv(raw: str | Sequence[str] | None) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, Sequence) and not isinstance(raw, str):
        return [str(part).strip() for part in raw if str(part).strip()]
    return [part.strip() for part in str(raw).split(",") if part.strip()]


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
    dist.init_process_group(backend=backend)
    args.distributed = True
    args.rank = distributed_rank()
    args.local_rank = local_rank
    args.world_size = distributed_world_size()


def cleanup_distributed() -> None:
    if distributed_is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def distributed_barrier() -> None:
    if distributed_is_initialized():
        dist.barrier()


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


def synchronize_gradients(modules: Sequence[nn.Module | None]) -> None:
    if not distributed_is_initialized():
        return
    world_size = float(distributed_world_size())
    for module in modules:
        if module is None:
            continue
        for parameter in module.parameters():
            if parameter.grad is None:
                continue
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            parameter.grad.div_(world_size)


def average_metrics_across_processes(metrics: Mapping[str, float]) -> dict[str, float]:
    if not distributed_is_initialized():
        return dict(metrics)
    averaged: dict[str, float] = {}
    for key, value in metrics.items():
        if not isinstance(value, (int, float)):
            continue
        tensor = torch.tensor(float(value), device=torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu"))
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        averaged[key] = float((tensor / distributed_world_size()).detach().cpu().item())
    return averaged


def gather_with_local_grad(tensor: torch.Tensor) -> torch.Tensor:
    if not distributed_is_initialized():
        return tensor
    gathered = [torch.zeros_like(tensor) for _ in range(distributed_world_size())]
    dist.all_gather(gathered, tensor.contiguous())
    gathered[distributed_rank()] = tensor
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
        with h5py.File(self.hdf5_path, "r") as handle:
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
        self.latent_token_count = int(self.latent_grid[0] * self.latent_grid[1])
        self.soft_prompt_tokens = int(query_tokens) if self.adapter_type == "qformer" else 1
        self.structured_query_conditioning = False
        self.soft_prompt_scale = float(soft_prompt_scale)
        adapter_dim = int(adapter_dim)
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

    def forward_soft_prompts(self, latent_map: torch.Tensor) -> torch.Tensor:
        if self.adapter_type == "qformer":
            latent_tokens = latent_map.flatten(2).transpose(1, 2)
            if int(latent_tokens.shape[1]) != self.latent_token_count:
                raise ValueError(
                    "Latent token count changed after adapter construction. "
                    f"Expected {self.latent_token_count}, got {int(latent_tokens.shape[1])}."
                )
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
        raise ValueError("alignment_projection_layers must be positive.")
    if layers == 1:
        return nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
        )
    modules: list[nn.Module] = [nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU()]
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
        self.student = build_projection_head(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            layers=layers,
            dropout=dropout,
        )
        if self.shared:
            self.teacher = self.student
        else:
            self.teacher = build_projection_head(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=hidden_dim,
                layers=layers,
                dropout=dropout,
            )

    def forward(self, student_hidden: torch.Tensor, teacher_hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.student(student_hidden.float()), self.teacher(teacher_hidden.float())


def symmetric_contrastive_loss(
    tensor_embedding: torch.Tensor,
    text_embedding: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    logits = tensor_embedding @ text_embedding.T / max(float(temperature), 1.0e-6)
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    loss = 0.5 * (loss_i2t + loss_t2i)
    with torch.no_grad():
        i2t_accuracy = (logits.argmax(dim=1) == labels).float().mean()
        t2i_accuracy = (logits.argmax(dim=0) == labels).float().mean()
    return loss, {
        "i2t_accuracy": float(i2t_accuracy.detach().cpu().item()),
        "t2i_accuracy": float(t2i_accuracy.detach().cpu().item()),
        "candidate_count": float(logits.shape[1]),
    }


def distributed_symmetric_contrastive_loss(
    tensor_embedding: torch.Tensor,
    text_embedding: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if not distributed_is_initialized():
        return symmetric_contrastive_loss(tensor_embedding, text_embedding, temperature)
    local_batch = int(tensor_embedding.shape[0])
    tensor_all = gather_with_local_grad(tensor_embedding)
    text_all = gather_with_local_grad(text_embedding)
    label_offset = distributed_rank() * local_batch
    labels = torch.arange(local_batch, device=tensor_embedding.device) + int(label_offset)
    logits_i2t = tensor_embedding @ text_all.T / max(float(temperature), 1.0e-6)
    logits_t2i = text_embedding @ tensor_all.T / max(float(temperature), 1.0e-6)
    loss_i2t = F.cross_entropy(logits_i2t, labels)
    loss_t2i = F.cross_entropy(logits_t2i, labels)
    loss = 0.5 * (loss_i2t + loss_t2i)
    with torch.no_grad():
        i2t_accuracy = (logits_i2t.argmax(dim=1) == labels).float().mean()
        t2i_accuracy = (logits_t2i.argmax(dim=1) == labels).float().mean()
    return loss, {
        "i2t_accuracy": float(i2t_accuracy.detach().cpu().item()),
        "t2i_accuracy": float(t2i_accuracy.detach().cpu().item()),
        "candidate_count": float(text_all.shape[0]),
    }


@torch.no_grad()
def retrieval_accuracy(
    tensor_embedding: torch.Tensor,
    text_embedding: torch.Tensor,
    temperature: float,
) -> dict[str, float]:
    logits = tensor_embedding.detach().float() @ text_embedding.detach().float().T / max(float(temperature), 1.0e-6)
    labels = torch.arange(logits.shape[0], device=logits.device)
    return {
        "contrastive_loss": float(
            (0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))).cpu().item()
        ),
        "i2t_accuracy": float((logits.argmax(dim=1) == labels).float().mean().cpu().item()),
        "t2i_accuracy": float((logits.argmax(dim=0) == labels).float().mean().cpu().item()),
        "candidate_count": float(logits.shape[1]),
    }


@torch.no_grad()
def distributed_retrieval_accuracy(
    tensor_embedding: torch.Tensor,
    text_embedding: torch.Tensor,
    temperature: float,
) -> dict[str, float]:
    if not distributed_is_initialized():
        return retrieval_accuracy(tensor_embedding, text_embedding, temperature)
    local_batch = int(tensor_embedding.shape[0])
    tensor_all = gather_with_local_grad(tensor_embedding.detach())
    text_all = gather_with_local_grad(text_embedding.detach())
    labels = torch.arange(local_batch, device=tensor_embedding.device) + distributed_rank() * local_batch
    logits_i2t = tensor_embedding.detach().float() @ text_all.float().T / max(float(temperature), 1.0e-6)
    logits_t2i = text_embedding.detach().float() @ tensor_all.float().T / max(float(temperature), 1.0e-6)
    return {
        "contrastive_loss": float(
            (0.5 * (F.cross_entropy(logits_i2t, labels) + F.cross_entropy(logits_t2i, labels))).cpu().item()
        ),
        "i2t_accuracy": float((logits_i2t.argmax(dim=1) == labels).float().mean().cpu().item()),
        "t2i_accuracy": float((logits_t2i.argmax(dim=1) == labels).float().mean().cpu().item()),
        "candidate_count": float(text_all.shape[0]),
    }


@torch.no_grad()
def full_retrieval_accuracy(
    tensor_embedding: torch.Tensor,
    text_embedding: torch.Tensor,
    temperature: float,
    chunk_size: int,
) -> dict[str, float]:
    tensor_cpu = tensor_embedding.detach().float().cpu()
    text_cpu = text_embedding.detach().float().cpu()
    total = int(tensor_cpu.shape[0])
    if total == 0:
        return {"contrastive_loss": 0.0, "i2t_accuracy": 0.0, "t2i_accuracy": 0.0}
    labels = torch.arange(total)
    chunk = max(1, int(chunk_size))
    i2t_correct = 0
    t2i_correct = 0
    i2t_loss_sum = 0.0
    t2i_loss_sum = 0.0
    for start in range(0, total, chunk):
        end = min(total, start + chunk)
        logits = tensor_cpu[start:end] @ text_cpu.T / max(float(temperature), 1.0e-6)
        local_labels = labels[start:end]
        i2t_correct += int((logits.argmax(dim=1) == local_labels).sum().item())
        i2t_loss_sum += float(F.cross_entropy(logits, local_labels, reduction="sum").item())
    for start in range(0, total, chunk):
        end = min(total, start + chunk)
        logits = text_cpu[start:end] @ tensor_cpu.T / max(float(temperature), 1.0e-6)
        local_labels = labels[start:end]
        t2i_correct += int((logits.argmax(dim=1) == local_labels).sum().item())
        t2i_loss_sum += float(F.cross_entropy(logits, local_labels, reduction="sum").item())
    return {
        "contrastive_loss": 0.5 * (i2t_loss_sum + t2i_loss_sum) / max(1, total),
        "i2t_accuracy": i2t_correct / max(1, total),
        "t2i_accuracy": t2i_correct / max(1, total),
    }


def cosine_alignment_loss(tensor_embedding: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(tensor_embedding, text_embedding, dim=-1).mean()


def reconstruction_mse(compressor: nn.Module, latent_map: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    reconstruction = compressor.decode({"latent_map": latent_map})
    return F.mse_loss(reconstruction, target)


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


def llm_backbone(llm: nn.Module) -> nn.Module:
    backbone = getattr(llm, "model", None)
    return backbone if isinstance(backbone, nn.Module) else llm


@torch.no_grad()
def probe_only_hidden(
    llm: nn.Module,
    anchor: AlignmentAnchor,
    device: torch.device,
    teacher_layer: int,
) -> torch.Tensor:
    input_ids = torch.tensor(anchor.token_ids, dtype=torch.long, device=device).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    outputs = llm_backbone(llm)(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    return hidden_at_last_non_padding(outputs.hidden_states, attention_mask, int(teacher_layer)).detach()


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
    if anchor.probe_template_index is None or not 0 <= int(anchor.probe_template_index) < PROBE_TEMPLATES_PER_FAMILY:
        raise ValueError(f"Probe anchor has an invalid template index: {anchor.probe_template_index!r}.")
    text = str(anchor.text or "")
    if not text.startswith("\n") or not text.endswith(" is"):
        raise ValueError(f"Probe stem must start with a newline and end at a natural ' is' continuation: {text!r}.")
    lowered = text.lower()
    leaked_markers = [marker for marker in PROBE_FORBIDDEN_INPUT_MARKERS if marker in lowered]
    if leaked_markers:
        raise ValueError(f"Probe stem contains forbidden answer-format markers {leaked_markers}: {text!r}.")
    if not anchor.probe_answers or len(anchor.probe_answers) != len(set(anchor.probe_answers)):
        raise ValueError(f"Probe continuations must be non-empty and unique: {anchor.probe_answers!r}.")
    leaked_answers = [answer for answer in anchor.probe_answers if str(answer).lower() in lowered]
    if leaked_answers:
        raise ValueError(f"Probe stem leaks canonical continuations {leaked_answers}: {text!r}.")


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
        raise ValueError("Behavior probes require patch_size greater than 1.")
    if int(channel_count) <= 0:
        raise ValueError("Behavior probes require at least one tensor channel.")

    family = probe_families[int(probe_index) % len(probe_families)]
    rng = random.Random(int(seed) + 1_000_003 * int(probe_index))
    channel = rng.randrange(int(channel_count))

    def select_template(templates: Sequence[str]) -> tuple[int, str]:
        if len(templates) != PROBE_TEMPLATES_PER_FAMILY:
            raise ValueError(
                f"Every probe family must define exactly {PROBE_TEMPLATES_PER_FAMILY} templates, "
                f"but {family} defines {len(templates)}."
            )
        selected = (
            int(template_index)
            if template_index is not None
            else (int(probe_index) // len(probe_families)) % PROBE_TEMPLATES_PER_FAMILY
        )
        if not 0 <= selected < PROBE_TEMPLATES_PER_FAMILY:
            raise ValueError(
                f"Probe template_index must be between 0 and {PROBE_TEMPLATES_PER_FAMILY - 1}, got {selected}."
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

    # `answers` maps tensor-derived class indices to hidden next-token targets; it is never serialized into the stem.
    if family == "point_sign":
        position = rng.randrange(int(patch_size) * int(patch_size))
        row, col = divmod(position, int(patch_size))
        location = point_text(row, col)
        templates = (
            f"\nThe value at {location} is",
            f"\nAt {location}, the value is",
            f"\nThe sign of the value at {location} is",
            f"\nFor {location}, the recorded value is",
        )
        selected_template_index, text = select_template(templates)
        parameters = (channel, row, col)
        answers = ("positive", "negative", "zero")
    elif family in {"point_relation", "directional_change"}:
        first, second = rng.sample(range(int(patch_size) * int(patch_size)), 2)
        row_a, col_a = divmod(first, int(patch_size))
        row_b, col_b = divmod(second, int(patch_size))
        location_a = point_text(row_a, col_a)
        location_b = point_text(row_b, col_b)
        if family == "point_relation":
            templates = (
                f"\nCompared with the value at {location_b}, the value at {location_a} is",
                f"\nRelative to the value at {location_b}, the value at {location_a} is",
                f"\nThe value at {location_a}, compared with the value at {location_b}, is",
                f"\nWith the value at {location_b} as reference, the value at {location_a} is",
            )
            selected_template_index, text = select_template(templates)
            answers = ("larger", "smaller", "equal")
        else:
            templates = (
                f"\nAt {location_b}, relative to {location_a}, the value is",
                f"\nCompared with its value at {location_a}, the value at {location_b} is",
                f"\nThe value at {location_b}, compared with the value at {location_a}, is",
                f"\nWith {location_a} as the starting point, the value at {location_b} is",
            )
            selected_template_index, text = select_template(templates)
            answers = ("higher", "lower", "unchanged")
        parameters = (channel, row_a, col_a, row_b, col_b)
    else:
        size = int(region_size)
        if size <= 0 or size >= int(patch_size):
            raise ValueError("patch_alignment.probe_region_size must be between 1 and patch_size - 1.")
        positions_per_axis = int(patch_size) - size + 1
        first, second = rng.sample(range(positions_per_axis * positions_per_axis), 2)
        row_a, col_a = divmod(first, positions_per_axis)
        row_b, col_b = divmod(second, positions_per_axis)
        statistic = "mean" if family == "region_mean_relation" else "value range"
        region_a = region_text(row_a, col_a, size)
        region_b = region_text(row_b, col_b, size)
        templates = (
            f"\nCompared with the {statistic} over {region_b}, the {statistic} over {region_a} is",
            f"\nRelative to the {statistic} over {region_b}, the {statistic} over {region_a} is",
            f"\nThe {statistic} over {region_a}, compared with the {statistic} over {region_b}, is",
            f"\nWith the {statistic} over {region_b} as reference, the {statistic} over {region_a} is",
        )
        selected_template_index, text = select_template(templates)
        parameters = (channel, row_a, col_a, row_b, col_b, size)
        answers = (
            ("higher", "lower", "equal")
            if family == "region_mean_relation"
            else ("larger", "smaller", "equal")
        )

    anchor = AlignmentAnchor(
        name=f"probe_{int(probe_index):02d}_{family}_t{selected_template_index}",
        mode="probe",
        token_ids=shared_suffix_token_ids(tokenizer, text, int(max_anchor_tokens)),
        text=text,
        probe_family=family,
        probe_template_index=int(selected_template_index),
        probe_parameters=parameters,
        probe_answers=answers,
    )
    validate_probe_anchor_contract(anchor)
    return anchor


def probe_target_indices(patches: torch.Tensor, anchor: AlignmentAnchor) -> torch.Tensor:
    family = anchor.probe_family
    if family is None:
        raise ValueError("probe_target_indices requires a behavior-probe anchor.")
    if patches.ndim != 4:
        raise ValueError(f"Expected probe patches [batch,channels,height,width], got {tuple(patches.shape)}.")
    channel, *parameters = anchor.probe_parameters
    values = patches[:, int(channel)].double()
    if family == "point_sign":
        row, col = parameters
        scores = values[:, int(row), int(col)]
        targets = torch.full_like(scores, 2, dtype=torch.long)
        targets = torch.where(scores > 0, torch.zeros_like(targets), targets)
        return torch.where(scores < 0, torch.ones_like(targets), targets)

    row_a, col_a, row_b, col_b, *rest = parameters
    if family in {"point_relation", "directional_change"}:
        score_a = values[:, int(row_a), int(col_a)]
        score_b = values[:, int(row_b), int(col_b)]
    else:
        size = int(rest[0])
        region_a = values[:, int(row_a) : int(row_a) + size, int(col_a) : int(col_a) + size]
        region_b = values[:, int(row_b) : int(row_b) + size, int(col_b) : int(col_b) + size]
        if family == "region_mean_relation":
            # Regions have the same area, so comparing sums avoids an unnecessary floating-point division.
            score_a = region_a.sum(dim=(-2, -1))
            score_b = region_b.sum(dim=(-2, -1))
        elif family == "region_range_relation":
            score_a = region_a.amax(dim=(-2, -1)) - region_a.amin(dim=(-2, -1))
            score_b = region_b.amax(dim=(-2, -1)) - region_b.amin(dim=(-2, -1))
        else:
            raise ValueError(f"Unsupported probe family: {family}")
    if family == "directional_change":
        first, second = score_b, score_a
    else:
        first, second = score_a, score_b
    targets = torch.full_like(first, 2, dtype=torch.long)
    targets = torch.where(first > second, torch.zeros_like(targets), targets)
    return torch.where(first < second, torch.ones_like(targets), targets)


def quantize_probe_values(patches: torch.Tensor, decimal_places: int) -> torch.Tensor:
    if not bool(torch.isfinite(patches).all()):
        non_finite = int((~torch.isfinite(patches)).sum().item())
        raise ValueError(f"Probe source tensor contains {non_finite} non-finite values.")
    scale = float(10 ** max(0, int(decimal_places)))
    return torch.round(patches.float() * scale) / scale


def probe_answer_token_ids(tokenizer: Any, anchor: AlignmentAnchor) -> tuple[int, ...]:
    if not anchor.probe_answers:
        raise ValueError("probe_answer_token_ids requires a probe anchor with canonical continuations.")
    token_ids: list[int] = []
    for answer in anchor.probe_answers:
        continuation = f" {answer}"
        encoded = tokenizer_ids(tokenizer, continuation)
        if len(encoded) != 1:
            raise ValueError(
                f"Probe continuation {continuation!r} must tokenize to one token, got {encoded}. "
                "Choose a canonical one-token continuation or extend sequence scoring."
            )
        combined = tokenizer_ids(tokenizer, f"{anchor.text}{continuation}")
        expected = list(anchor.token_ids) + encoded
        if combined != expected:
            raise ValueError(
                f"Probe continuation {continuation!r} changes tokenization at the stem boundary. "
                f"Expected stem+continuation IDs {expected}, got {combined}."
            )
        token_ids.append(int(encoded[0]))
    if len(token_ids) != len(set(token_ids)):
        raise ValueError(f"Probe continuations resolved to duplicate token IDs: {token_ids}.")
    return tuple(token_ids)


def probe_behavior_loss(
    *,
    llm: nn.Module,
    student_final_hidden: torch.Tensor,
    teacher_final_hidden: torch.Tensor,
    targets: torch.Tensor,
    continuation_token_ids: Sequence[int],
    kl_temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    output_embeddings = llm.get_output_embeddings()
    if output_embeddings is None:
        raise ValueError("The frozen causal LLM must expose an output embedding layer for probe scoring.")
    student_vocab_logits = output_embeddings(student_final_hidden).float()
    continuation_index = torch.tensor(
        continuation_token_ids,
        dtype=torch.long,
        device=student_vocab_logits.device,
    )
    student_logits = student_vocab_logits.index_select(1, continuation_index)
    with torch.no_grad():
        teacher_vocab_logits = output_embeddings(teacher_final_hidden).float()
        teacher_logits = teacher_vocab_logits.index_select(1, continuation_index)
    answer_token_ids = continuation_index.index_select(0, targets)
    answer_ce = F.cross_entropy(student_vocab_logits, answer_token_ids)
    temperature = float(kl_temperature)
    teacher_distribution = F.softmax(teacher_logits / temperature, dim=-1)
    per_record_kl = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        teacher_distribution,
        reduction="none",
    ).sum(dim=-1) * (temperature**2)
    teacher_completion_correct = teacher_logits.argmax(dim=-1) == targets
    teacher_top_tokens = teacher_vocab_logits.argmax(dim=-1)
    teacher_correct = teacher_top_tokens == answer_token_ids
    if bool(teacher_correct.any()):
        teacher_kl = per_record_kl[teacher_correct].mean()
    else:
        teacher_kl = per_record_kl.sum() * 0.0
    student_top_tokens = student_vocab_logits.argmax(dim=-1)
    student_completion_predictions = student_logits.argmax(dim=-1)
    teacher_completion_predictions = teacher_logits.argmax(dim=-1)
    if int(student_vocab_logits.shape[0]) > 1:
        shuffled_targets = targets.roll(shifts=-1, dims=0)
        shuffled_answer_token_ids = answer_token_ids.roll(shifts=-1, dims=0)
        shuffled_completion_accuracy = (student_completion_predictions == shuffled_targets).float().mean()
        shuffled_answer_ce = F.cross_entropy(student_vocab_logits, shuffled_answer_token_ids)
    else:
        shuffled_completion_accuracy = (student_completion_predictions == targets).float().mean()
        shuffled_answer_ce = answer_ce.detach()
    completion_accuracy = (student_completion_predictions == targets).float().mean()
    student_correct_logits = student_logits.gather(1, targets.unsqueeze(1)).squeeze(1)
    teacher_correct_logits = teacher_logits.gather(1, targets.unsqueeze(1)).squeeze(1)
    student_incorrect_logits = student_logits.scatter(1, targets.unsqueeze(1), float("-inf")).amax(dim=1)
    teacher_incorrect_logits = teacher_logits.scatter(1, targets.unsqueeze(1), float("-inf")).amax(dim=1)
    metrics = {
        "completion_accuracy": float(completion_accuracy.detach().cpu().item()),
        "shuffled_completion_accuracy": float(shuffled_completion_accuracy.detach().cpu().item()),
        "completion_latent_gain": float(
            (completion_accuracy - shuffled_completion_accuracy).detach().cpu().item()
        ),
        "shuffled_answer_ce": float(shuffled_answer_ce.detach().cpu().item()),
        "answer_token_accuracy": float(
            (student_top_tokens == answer_token_ids).float().mean().detach().cpu().item()
        ),
        "answer_format_rate": float(
            (student_top_tokens.unsqueeze(1) == continuation_index.unsqueeze(0))
            .any(dim=1)
            .float()
            .mean()
            .detach()
            .cpu()
            .item()
        ),
        "teacher_completion_accuracy": float(
            teacher_completion_correct.float().mean().detach().cpu().item()
        ),
        "teacher_answer_token_accuracy": float(teacher_correct.float().mean().detach().cpu().item()),
        "teacher_answer_nll": float(
            F.cross_entropy(teacher_vocab_logits, answer_token_ids).detach().cpu().item()
        ),
        "teacher_answer_format_rate": float(
            (teacher_top_tokens.unsqueeze(1) == continuation_index.unsqueeze(0))
            .any(dim=1)
            .float()
            .mean()
            .detach()
            .cpu()
            .item()
        ),
        "teacher_kl_active_fraction": float(teacher_correct.float().mean().detach().cpu().item()),
        "target_class_0_fraction": float((targets == 0).float().mean().detach().cpu().item()),
        "student_predicted_class_0_fraction": float(
            (student_completion_predictions == 0).float().mean().detach().cpu().item()
        ),
        "teacher_predicted_class_0_fraction": float(
            (teacher_completion_predictions == 0).float().mean().detach().cpu().item()
        ),
        "student_completion_margin": float(
            (student_correct_logits - student_incorrect_logits).mean().detach().cpu().item()
        ),
        "teacher_completion_margin": float(
            (teacher_correct_logits - teacher_incorrect_logits).mean().detach().cpu().item()
        ),
    }
    return answer_ce, teacher_kl, metrics


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
        "anchor_missing_fraction": 0.0,
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


def prepare_alignment_embeddings(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    center_embeddings: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    tensor_embedding = student_hidden.float()
    text_embedding = teacher_hidden.float()
    if bool(center_embeddings) and int(tensor_embedding.shape[0]) > 1:
        tensor_embedding = tensor_embedding - tensor_embedding.mean(dim=0, keepdim=True)
        text_embedding = text_embedding - text_embedding.mean(dim=0, keepdim=True)
    return F.normalize(tensor_embedding, dim=-1), F.normalize(text_embedding, dim=-1)


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
    outputs = llm_backbone(llm)(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    hidden = hidden_at_last_non_padding(outputs.hidden_states, attention_mask, teacher_layer).detach()
    final_hidden = hidden_at_last_non_padding(outputs.hidden_states, attention_mask, -1).detach()
    metrics["hidden_norm"] = mean_token_norm(hidden.unsqueeze(1))
    metrics["hidden_pairwise_cosine"] = off_diagonal_cosine_mean(hidden)
    metrics["final_hidden_norm"] = mean_token_norm(final_hidden.unsqueeze(1))
    return HiddenBatch(hidden=hidden, final_hidden=final_hidden, metrics=metrics)


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
    outputs = llm_backbone(llm)(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    hidden = hidden_at_last_non_padding(
        outputs.hidden_states,
        text_attention_mask,
        teacher_layer,
        prefix_tokens=int(soft_prompts.shape[1]),
    )
    final_hidden = hidden_at_last_non_padding(
        outputs.hidden_states,
        text_attention_mask,
        -1,
        prefix_tokens=int(soft_prompts.shape[1]),
    )
    metrics["hidden_norm"] = mean_token_norm(hidden.unsqueeze(1))
    metrics["hidden_pairwise_cosine"] = off_diagonal_cosine_mean(hidden)
    metrics["final_hidden_norm"] = mean_token_norm(final_hidden.unsqueeze(1))
    return HiddenBatch(hidden=hidden, final_hidden=final_hidden, metrics=metrics)


def normalize_patch_batch(
    patches: torch.Tensor,
    input_size: Sequence[int],
    normalization_cfg: Mapping[str, Any],
    resize_to_input: bool,
) -> torch.Tensor:
    batch = patches.to(dtype=torch.float32)
    if bool(resize_to_input):
        batch = resize_chw_batch(batch, input_size)
    normalized = []
    for patch in batch:
        normalized_patch, _state = normalize_tensor(patch.cpu(), dict(normalization_cfg or {}))
        normalized.append(normalized_patch)
    return torch.stack(normalized, dim=0)


def build_teacher_texts_for_batch(
    batch: Mapping[str, Any],
    normalized_patches: torch.Tensor,
    args: argparse.Namespace,
) -> list[str]:
    source = str(args.teacher_text_source).lower()
    if str(args.alignment_text_layout) == "values_shared_suffix":
        patches = batch["patch"] if source == "raw" else normalized_patches
        if source not in {"raw", "normalized"}:
            raise ValueError(f"Unsupported teacher_text_source: {args.teacher_text_source}")
        visible_patches = quantize_probe_values(patches, int(args.text_decimal_places))
        return serialize_tensor_value_batch(visible_patches, int(args.text_decimal_places))
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
                template_index=(
                    (index // len(args.probe_families) + index % len(args.probe_families))
                    % PROBE_TEMPLATES_PER_FAMILY
                    if evaluation
                    else None
                ),
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
        for template_index in range(PROBE_TEMPLATES_PER_FAMILY):
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
            probe_answer_token_ids(tokenizer, anchor)
            anchors.append(anchor)
    expected_count = len(families) * PROBE_TEMPLATES_PER_FAMILY
    observed_pairs = {(anchor.probe_family, anchor.probe_template_index) for anchor in anchors}
    if len(anchors) != expected_count or len(observed_pairs) != expected_count:
        raise ValueError(
            "Probe contract preflight did not cover every family/template pair: "
            f"expected={expected_count}, anchors={len(anchors)}, unique_pairs={len(observed_pairs)}."
        )
    for family in families:
        family_stems = {anchor.token_ids for anchor in anchors if anchor.probe_family == family}
        if len(family_stems) != PROBE_TEMPLATES_PER_FAMILY:
            raise ValueError(
                f"Probe templates for {family} collapsed to duplicate token sequences after tokenization: "
                f"expected={PROBE_TEMPLATES_PER_FAMILY}, unique_token_sequences={len(family_stems)}."
            )
    return anchors


def train_compressor_during_alignment(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "alignment_train_patch_ae", args.train_patch_ae))


def train_one_epoch(
    *,
    compressor: nn.Module,
    adapter: TensorPatchAlignmentAdapter,
    projector: AlignmentProjectionPair | None,
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
    train_compressor = train_compressor_during_alignment(args)
    if train_compressor:
        compressor.train()
    else:
        compressor.eval()
    total_loss = 0.0
    total_contrastive = 0.0
    total_cosine = 0.0
    total_reconstruction = 0.0
    total_probe_answer_ce = 0.0
    total_probe_teacher_kl = 0.0
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
        if train_compressor:
            latent = compressor.encode(patches)["latent_map"]
        else:
            with torch.no_grad():
                latent = compressor.encode(patches)["latent_map"]
        soft_prompts = adapter.forward_soft_prompts(latent)
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
        if alignment_anchor.mode == "probe":
            baseline_hidden = probe_only_hidden(
                llm,
                alignment_anchor,
                device,
                int(args.teacher_layer),
            ).to(dtype=student_hidden.dtype)
            student_alignment_hidden = student_hidden - baseline_hidden
            teacher_alignment_hidden = teacher_hidden - baseline_hidden
        else:
            baseline_hidden = student_hidden.new_zeros((1, student_hidden.shape[-1]))
            student_alignment_hidden = student_hidden
            teacher_alignment_hidden = teacher_hidden
        if projector is None:
            student_features = student_alignment_hidden
            teacher_features = teacher_alignment_hidden
        else:
            student_features, teacher_features = projector(student_alignment_hidden, teacher_alignment_hidden)
        tensor_embedding, text_embedding = prepare_alignment_embeddings(
            student_features,
            teacher_features,
            bool(args.center_embeddings),
        )
        centered_tensor_embedding, centered_text_embedding = prepare_alignment_embeddings(
            student_features,
            teacher_features,
            True,
        )
        uncentered_tensor_embedding, uncentered_text_embedding = prepare_alignment_embeddings(
            student_features,
            teacher_features,
            False,
        )
        hidden_uncentered_tensor_embedding, hidden_uncentered_text_embedding = prepare_alignment_embeddings(
            student_hidden,
            teacher_hidden,
            False,
        )
        hidden_centered_tensor_embedding, hidden_centered_text_embedding = prepare_alignment_embeddings(
            student_hidden,
            teacher_hidden,
            True,
        )
        contrastive, contrastive_metrics = distributed_symmetric_contrastive_loss(
            tensor_embedding,
            text_embedding,
            float(args.temperature),
        )
        centered_contrastive, centered_contrastive_metrics = distributed_symmetric_contrastive_loss(
            centered_tensor_embedding,
            centered_text_embedding,
            float(args.temperature),
        )
        cosine = cosine_alignment_loss(tensor_embedding, text_embedding)
        reconstruction = reconstruction_mse(compressor, latent, patches)
        probe_answer_ce = student_hidden.new_zeros(())
        probe_teacher_kl = student_hidden.new_zeros(())
        probe_metrics: dict[str, float] = {}
        if alignment_anchor.mode == "probe":
            probe_source_patches = (
                batch["patch"]
                if str(args.teacher_text_source).lower() == "raw"
                else normalized_patches
            )
            targets = probe_target_indices(
                quantize_probe_values(probe_source_patches, int(args.text_decimal_places)).to(device),
                alignment_anchor,
            )
            probe_answer_ce, probe_teacher_kl, probe_metrics = probe_behavior_loss(
                llm=llm,
                student_final_hidden=student_output.final_hidden,
                teacher_final_hidden=teacher_output.final_hidden,
                targets=targets,
                continuation_token_ids=probe_answer_token_ids(tokenizer, alignment_anchor),
                kl_temperature=float(args.probe_kl_temperature),
            )
        reconstruction_weight = float(args.reconstruction_loss_weight) if train_compressor else 0.0
        loss = (
            float(args.contrastive_loss_weight) * contrastive
            + float(args.centered_contrastive_loss_weight) * centered_contrastive
            + float(args.cosine_loss_weight) * cosine
            + reconstruction_weight * reconstruction
            + float(args.probe_answer_ce_weight) * probe_answer_ce
            + float(args.probe_teacher_kl_weight) * probe_teacher_kl
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
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
        total_cosine += float(cosine.detach().cpu().item()) * batch_size
        total_reconstruction += float(reconstruction.detach().cpu().item()) * batch_size
        total_probe_answer_ce += float(probe_answer_ce.detach().cpu().item()) * batch_size
        total_probe_teacher_kl += float(probe_teacher_kl.detach().cpu().item()) * batch_size
        total_i2t += float(contrastive_metrics["i2t_accuracy"]) * batch_size
        total_t2i += float(contrastive_metrics["t2i_accuracy"]) * batch_size
        add_weighted_metrics(
            metric_totals,
            {
                "contrastive_loss": float(centered_contrastive.detach().cpu().item()),
                "i2t_accuracy": float(centered_contrastive_metrics["i2t_accuracy"]),
                "t2i_accuracy": float(centered_contrastive_metrics["t2i_accuracy"]),
            },
            batch_size,
            "centered_",
        )
        add_weighted_metrics(
            metric_totals,
            distributed_retrieval_accuracy(
                uncentered_tensor_embedding,
                uncentered_text_embedding,
                float(args.temperature),
            ),
            batch_size,
            "uncentered_",
        )
        add_weighted_metrics(
            metric_totals,
            distributed_retrieval_accuracy(
                hidden_uncentered_tensor_embedding,
                hidden_uncentered_text_embedding,
                float(args.temperature),
            ),
            batch_size,
            "hidden_uncentered_",
        )
        add_weighted_metrics(
            metric_totals,
            distributed_retrieval_accuracy(
                hidden_centered_tensor_embedding,
                hidden_centered_text_embedding,
                float(args.temperature),
            ),
            batch_size,
            "hidden_centered_",
        )
        add_weighted_metrics(metric_totals, teacher_output.metrics, batch_size, "teacher_")
        add_weighted_metrics(metric_totals, student_output.metrics, batch_size, "student_")
        add_weighted_metrics(metric_totals, probe_metrics, batch_size, "probe_")
        add_weighted_metrics(
            metric_totals,
            {
                "positive_cosine": float((1.0 - cosine.detach()).cpu().item()),
                "student_embedding_pairwise_cosine": off_diagonal_cosine_mean(tensor_embedding),
                "teacher_embedding_pairwise_cosine": off_diagonal_cosine_mean(text_embedding),
                "centered_student_embedding_pairwise_cosine": off_diagonal_cosine_mean(centered_tensor_embedding),
                "centered_teacher_embedding_pairwise_cosine": off_diagonal_cosine_mean(centered_text_embedding),
                "uncentered_student_embedding_pairwise_cosine": off_diagonal_cosine_mean(uncentered_tensor_embedding),
                "uncentered_teacher_embedding_pairwise_cosine": off_diagonal_cosine_mean(uncentered_text_embedding),
                "hidden_positive_cosine": float(
                    F.cosine_similarity(
                        hidden_uncentered_tensor_embedding,
                        hidden_uncentered_text_embedding,
                        dim=-1,
                    )
                    .mean()
                    .detach()
                    .cpu()
                    .item()
                ),
                "student_hidden_pairwise_cosine": off_diagonal_cosine_mean(hidden_uncentered_tensor_embedding),
                "teacher_hidden_pairwise_cosine": off_diagonal_cosine_mean(hidden_uncentered_text_embedding),
                "probe_baseline_hidden_norm": mean_token_norm(baseline_hidden.unsqueeze(1)),
                "student_conditioned_delta_norm": mean_token_norm(student_alignment_hidden.unsqueeze(1)),
                "teacher_conditioned_delta_norm": mean_token_norm(teacher_alignment_hidden.unsqueeze(1)),
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
        "cosine_loss": total_cosine / max(1, total_records),
        "reconstruction_loss": total_reconstruction / max(1, total_records),
        "probe_answer_ce": total_probe_answer_ce / max(1, total_records),
        "probe_teacher_kl": total_probe_teacher_kl / max(1, total_records),
        "i2t_accuracy": total_i2t / max(1, total_records),
        "t2i_accuracy": total_t2i / max(1, total_records),
    }
    metrics.update(averaged_metrics(metric_totals, total_records))
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
    progress = tqdm(loader, desc=f"pretrain patch AE epoch {epoch}", leave=False, disable=not is_main_process())
    for step, batch in enumerate(progress, start=1):
        patches = normalize_patch_batch(
            batch["patch"],
            compressor_input_size,
            normalization_cfg,
            bool(args.resize_patch_to_compressor_input),
        ).to(device)
        latent = compressor.encode(patches)["latent_map"]
        loss = reconstruction_mse(compressor, latent, patches)
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
        average_loss = total_loss / max(1, total_records)
        progress.set_postfix(recon=f"{average_loss:.4f}")
        global_step += 1
        if wandb_logger is not None and int(args.log_interval) > 0 and step % int(args.log_interval) == 0:
            wandb_logger.log(
                {
                    "patch_ae_pretrain_step/reconstruction_loss": average_loss,
                    "patch_ae_pretrain_step/current_reconstruction_loss": float(loss.detach().cpu().item()),
                    "patch_ae_pretrain_step/epoch": float(epoch),
                    "patch_ae_pretrain_step/epoch_step": float(step),
                    "patch_ae_pretrain_step/lr": float(optimizer.param_groups[0]["lr"]),
                },
                step=global_step,
            )
    metrics = {"reconstruction_loss": total_loss / max(1, total_records)}
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
    for batch in tqdm(loader, desc="eval patch AE reconstruction", leave=False, disable=not is_main_process()):
        patches = normalize_patch_batch(
            batch["patch"],
            compressor_input_size,
            normalization_cfg,
            bool(args.resize_patch_to_compressor_input),
        ).to(device)
        latent = compressor.encode(patches)["latent_map"]
        loss = reconstruction_mse(compressor, latent, patches)
        batch_size = int(patches.shape[0])
        total_loss += float(loss.detach().cpu().item()) * batch_size
        total_records += batch_size
    return {"reconstruction_loss": total_loss / max(1, total_records)}


@torch.no_grad()
def evaluate(
    *,
    compressor: nn.Module,
    adapter: TensorPatchAlignmentAdapter,
    projector: AlignmentProjectionPair | None,
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
    if projector is not None:
        projector.eval()
    total_loss = 0.0
    total_contrastive = 0.0
    total_cosine = 0.0
    total_reconstruction = 0.0
    total_probe_answer_ce = 0.0
    total_probe_teacher_kl = 0.0
    total_i2t = 0.0
    total_t2i = 0.0
    total_records = 0
    metric_totals: dict[str, float] = {}
    collected_student_features: list[torch.Tensor] = []
    collected_teacher_features: list[torch.Tensor] = []
    collected_student_hidden: list[torch.Tensor] = []
    collected_teacher_hidden: list[torch.Tensor] = []
    train_compressor = train_compressor_during_alignment(args)
    if alignment_anchor is None and str(args.alignment_text_layout) == "values_shared_suffix":
        alignment_anchor = alignment_anchors_from_args(tokenizer, args, evaluation=True)[0]
    probe_continuation_token_ids = (
        probe_answer_token_ids(tokenizer, alignment_anchor)
        if alignment_anchor and alignment_anchor.mode == "probe"
        else ()
    )
    evaluation_baseline_hidden = (
        probe_only_hidden(llm, alignment_anchor, device, int(args.teacher_layer))
        if alignment_anchor and alignment_anchor.mode == "probe"
        else None
    )
    for batch in tqdm(loader, desc="eval align", leave=False, disable=not is_main_process()):
        normalized_patches = normalize_patch_batch(
            batch["patch"],
            compressor_input_size,
            normalization_cfg,
            bool(args.resize_patch_to_compressor_input),
        )
        patches = normalized_patches.to(device)
        teacher_output = text_teacher_hidden(
            llm,
            tokenizer,
            build_teacher_texts_for_batch(batch, normalized_patches, args),
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
        if evaluation_baseline_hidden is not None:
            baseline_hidden = evaluation_baseline_hidden.to(dtype=student_hidden.dtype)
            student_alignment_hidden = student_hidden - baseline_hidden
            teacher_alignment_hidden = teacher_hidden - baseline_hidden
        else:
            baseline_hidden = student_hidden.new_zeros((1, student_hidden.shape[-1]))
            student_alignment_hidden = student_hidden
            teacher_alignment_hidden = teacher_hidden
        if projector is None:
            student_features = student_alignment_hidden
            teacher_features = teacher_alignment_hidden
        else:
            student_features, teacher_features = projector(student_alignment_hidden, teacher_alignment_hidden)
        tensor_embedding, text_embedding = prepare_alignment_embeddings(
            student_features,
            teacher_features,
            bool(args.center_embeddings),
        )
        centered_tensor_embedding, centered_text_embedding = prepare_alignment_embeddings(
            student_features,
            teacher_features,
            True,
        )
        uncentered_tensor_embedding, uncentered_text_embedding = prepare_alignment_embeddings(
            student_features,
            teacher_features,
            False,
        )
        hidden_uncentered_tensor_embedding, hidden_uncentered_text_embedding = prepare_alignment_embeddings(
            student_hidden,
            teacher_hidden,
            False,
        )
        hidden_centered_tensor_embedding, hidden_centered_text_embedding = prepare_alignment_embeddings(
            student_hidden,
            teacher_hidden,
            True,
        )
        contrastive, contrastive_metrics = symmetric_contrastive_loss(
            tensor_embedding,
            text_embedding,
            float(args.temperature),
        )
        centered_contrastive, centered_contrastive_metrics = symmetric_contrastive_loss(
            centered_tensor_embedding,
            centered_text_embedding,
            float(args.temperature),
        )
        cosine = cosine_alignment_loss(tensor_embedding, text_embedding)
        reconstruction = reconstruction_mse(compressor, latent, patches)
        probe_answer_ce = student_hidden.new_zeros(())
        probe_teacher_kl = student_hidden.new_zeros(())
        probe_metrics: dict[str, float] = {}
        if alignment_anchor is not None and alignment_anchor.mode == "probe":
            probe_source_patches = (
                batch["patch"]
                if str(args.teacher_text_source).lower() == "raw"
                else normalized_patches
            )
            targets = probe_target_indices(
                quantize_probe_values(probe_source_patches, int(args.text_decimal_places)).to(device),
                alignment_anchor,
            )
            probe_answer_ce, probe_teacher_kl, probe_metrics = probe_behavior_loss(
                llm=llm,
                student_final_hidden=student_output.final_hidden,
                teacher_final_hidden=teacher_output.final_hidden,
                targets=targets,
                continuation_token_ids=probe_continuation_token_ids,
                kl_temperature=float(args.probe_kl_temperature),
            )
        reconstruction_weight = float(args.reconstruction_loss_weight) if train_compressor else 0.0
        loss = (
            float(args.contrastive_loss_weight) * contrastive
            + float(args.centered_contrastive_loss_weight) * centered_contrastive
            + float(args.cosine_loss_weight) * cosine
            + reconstruction_weight * reconstruction
            + float(args.probe_answer_ce_weight) * probe_answer_ce
            + float(args.probe_teacher_kl_weight) * probe_teacher_kl
        )
        batch_size = int(patches.shape[0])
        total_loss += float(loss.detach().cpu().item()) * batch_size
        total_contrastive += float(contrastive.detach().cpu().item()) * batch_size
        total_cosine += float(cosine.detach().cpu().item()) * batch_size
        total_reconstruction += float(reconstruction.detach().cpu().item()) * batch_size
        total_probe_answer_ce += float(probe_answer_ce.detach().cpu().item()) * batch_size
        total_probe_teacher_kl += float(probe_teacher_kl.detach().cpu().item()) * batch_size
        total_i2t += float(contrastive_metrics["i2t_accuracy"]) * batch_size
        total_t2i += float(contrastive_metrics["t2i_accuracy"]) * batch_size
        add_weighted_metrics(
            metric_totals,
            {
                "contrastive_loss": float(centered_contrastive.detach().cpu().item()),
                "i2t_accuracy": float(centered_contrastive_metrics["i2t_accuracy"]),
                "t2i_accuracy": float(centered_contrastive_metrics["t2i_accuracy"]),
            },
            batch_size,
            "centered_",
        )
        add_weighted_metrics(
            metric_totals,
            retrieval_accuracy(
                uncentered_tensor_embedding,
                uncentered_text_embedding,
                float(args.temperature),
            ),
            batch_size,
            "uncentered_",
        )
        add_weighted_metrics(
            metric_totals,
            retrieval_accuracy(
                hidden_uncentered_tensor_embedding,
                hidden_uncentered_text_embedding,
                float(args.temperature),
            ),
            batch_size,
            "hidden_uncentered_",
        )
        add_weighted_metrics(
            metric_totals,
            retrieval_accuracy(
                hidden_centered_tensor_embedding,
                hidden_centered_text_embedding,
                float(args.temperature),
            ),
            batch_size,
            "hidden_centered_",
        )
        add_weighted_metrics(metric_totals, teacher_output.metrics, batch_size, "teacher_")
        add_weighted_metrics(metric_totals, student_output.metrics, batch_size, "student_")
        add_weighted_metrics(metric_totals, probe_metrics, batch_size, "probe_")
        add_weighted_metrics(
            metric_totals,
            {
                "positive_cosine": float((1.0 - cosine.detach()).cpu().item()),
                "student_embedding_pairwise_cosine": off_diagonal_cosine_mean(tensor_embedding),
                "teacher_embedding_pairwise_cosine": off_diagonal_cosine_mean(text_embedding),
                "centered_student_embedding_pairwise_cosine": off_diagonal_cosine_mean(centered_tensor_embedding),
                "centered_teacher_embedding_pairwise_cosine": off_diagonal_cosine_mean(centered_text_embedding),
                "uncentered_student_embedding_pairwise_cosine": off_diagonal_cosine_mean(uncentered_tensor_embedding),
                "uncentered_teacher_embedding_pairwise_cosine": off_diagonal_cosine_mean(uncentered_text_embedding),
                "hidden_positive_cosine": float(
                    F.cosine_similarity(
                        hidden_uncentered_tensor_embedding,
                        hidden_uncentered_text_embedding,
                        dim=-1,
                    )
                    .mean()
                    .detach()
                    .cpu()
                    .item()
                ),
                "student_hidden_pairwise_cosine": off_diagonal_cosine_mean(hidden_uncentered_tensor_embedding),
                "teacher_hidden_pairwise_cosine": off_diagonal_cosine_mean(hidden_uncentered_text_embedding),
                "probe_baseline_hidden_norm": mean_token_norm(baseline_hidden.unsqueeze(1)),
                "student_conditioned_delta_norm": mean_token_norm(student_alignment_hidden.unsqueeze(1)),
                "teacher_conditioned_delta_norm": mean_token_norm(teacher_alignment_hidden.unsqueeze(1)),
            },
            batch_size,
            "alignment_",
        )
        total_records += batch_size
        if bool(args.global_retrieval_eval) and total_records <= int(args.global_retrieval_max_records):
            collected_student_features.append(student_features.detach().float().cpu())
            collected_teacher_features.append(teacher_features.detach().float().cpu())
            collected_student_hidden.append(student_hidden.detach().float().cpu())
            collected_teacher_hidden.append(teacher_hidden.detach().float().cpu())
    metrics = {
        "loss": total_loss / max(1, total_records),
        "contrastive_loss": total_contrastive / max(1, total_records),
        "cosine_loss": total_cosine / max(1, total_records),
        "reconstruction_loss": total_reconstruction / max(1, total_records),
        "probe_answer_ce": total_probe_answer_ce / max(1, total_records),
        "probe_teacher_kl": total_probe_teacher_kl / max(1, total_records),
        "i2t_accuracy": total_i2t / max(1, total_records),
        "t2i_accuracy": total_t2i / max(1, total_records),
    }
    metrics.update(averaged_metrics(metric_totals, total_records))
    if (
        bool(args.global_retrieval_eval)
        and collected_student_features
        and total_records <= int(args.global_retrieval_max_records)
    ):
        student_all = torch.cat(collected_student_features, dim=0)
        teacher_all = torch.cat(collected_teacher_features, dim=0)
        student_hidden_all = torch.cat(collected_student_hidden, dim=0)
        teacher_hidden_all = torch.cat(collected_teacher_hidden, dim=0)
        global_tensor, global_text = prepare_alignment_embeddings(student_all, teacher_all, bool(args.center_embeddings))
        global_centered_tensor, global_centered_text = prepare_alignment_embeddings(student_all, teacher_all, True)
        global_uncentered_tensor, global_uncentered_text = prepare_alignment_embeddings(student_all, teacher_all, False)
        global_hidden_centered_tensor, global_hidden_centered_text = prepare_alignment_embeddings(
            student_hidden_all,
            teacher_hidden_all,
            True,
        )
        global_hidden_uncentered_tensor, global_hidden_uncentered_text = prepare_alignment_embeddings(
            student_hidden_all,
            teacher_hidden_all,
            False,
        )
        metrics.update(
            {
                f"global_{key}": value
                for key, value in full_retrieval_accuracy(
                    global_tensor,
                    global_text,
                    float(args.temperature),
                    int(args.global_retrieval_chunk_size),
                ).items()
            }
        )
        metrics.update(
            {
                f"global_centered_{key}": value
                for key, value in full_retrieval_accuracy(
                    global_centered_tensor,
                    global_centered_text,
                    float(args.temperature),
                    int(args.global_retrieval_chunk_size),
                ).items()
            }
        )
        metrics.update(
            {
                f"global_uncentered_{key}": value
                for key, value in full_retrieval_accuracy(
                    global_uncentered_tensor,
                    global_uncentered_text,
                    float(args.temperature),
                    int(args.global_retrieval_chunk_size),
                ).items()
            }
        )
        metrics.update(
            {
                f"global_hidden_centered_{key}": value
                for key, value in full_retrieval_accuracy(
                    global_hidden_centered_tensor,
                    global_hidden_centered_text,
                    float(args.temperature),
                    int(args.global_retrieval_chunk_size),
                ).items()
            }
        )
        metrics.update(
            {
                f"global_hidden_uncentered_{key}": value
                for key, value in full_retrieval_accuracy(
                    global_hidden_uncentered_tensor,
                    global_hidden_uncentered_text,
                    float(args.temperature),
                    int(args.global_retrieval_chunk_size),
                ).items()
            }
        )
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


@torch.no_grad()
def evaluate_anchor_bank(
    *,
    compressor: nn.Module,
    adapter: TensorPatchAlignmentAdapter,
    projector: AlignmentProjectionPair | None,
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
    parser.add_argument("--compressor-checkpoint", type=str, default=None)
    parser.add_argument("--compressor-config", type=str, default=None)
    parser.add_argument("--resize-patch-to-compressor-input", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--model-name-or-path", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--hf-home", type=str, default=None)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--torch-dtype", type=str, choices=("auto", "float32", "float16", "bfloat16"), default=None)
    parser.add_argument("--adapter-dim", type=int, default=None)
    parser.add_argument("--adapter-type", type=str, choices=("qformer", "pooled_mlp"), default=None)
    parser.add_argument("--query-tokens", type=int, default=None)
    parser.add_argument("--adapter-layers", type=int, default=None)
    parser.add_argument("--adapter-heads", type=int, default=None)
    parser.add_argument("--projection-dim", type=int, default=None)
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
    parser.add_argument("--centered-contrastive-loss-weight", type=float, default=None)
    parser.add_argument("--cosine-loss-weight", type=float, default=None)
    parser.add_argument("--reconstruction-loss-weight", type=float, default=None)
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
    parser.add_argument("--probe-answer-ce-weight", type=float, default=None)
    parser.add_argument("--probe-teacher-kl-weight", type=float, default=None)
    parser.add_argument("--probe-kl-temperature", type=float, default=None)
    parser.add_argument("--probe-teacher-preflight-records", type=int, default=None)
    parser.add_argument("--max-shared-suffix-tokens", type=int, default=None)
    parser.add_argument("--center-embeddings", action=argparse.BooleanOptionalAction, default=None)
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
    set_default(args, "train_patch_ae", first_nested(config, ["patch_alignment.train_patch_ae"]), args.encoder_source == "patch_ae_config")
    set_default(args, "freeze_patch_ae_after_pretrain", first_nested(config, ["patch_alignment.freeze_patch_ae_after_pretrain"]), True)
    set_default(args, "patch_ae_pretrain_epochs", first_nested(config, ["patch_alignment.patch_ae_pretrain_epochs"]), 0)
    set_default(
        args,
        "resize_patch_to_compressor_input",
        first_nested(config, ["patch_alignment.resize_patch_to_compressor_input"]),
        args.encoder_source == "checkpoint",
    )
    set_default(args, "trust_remote_code", first_nested(config, ["model.trust_remote_code"]), False)
    set_default(args, "torch_dtype", first_nested(config, ["model.torch_dtype"]), "bfloat16")
    set_default(args, "adapter_dim", first_nested(config, ["patch_alignment.adapter_dim"]), 512)
    set_default(args, "adapter_type", first_nested(config, ["patch_alignment.adapter_type"]), "qformer")
    set_default(args, "query_tokens", first_nested(config, ["patch_alignment.query_tokens"]), 8)
    set_default(args, "adapter_layers", first_nested(config, ["patch_alignment.adapter_layers"]), 2)
    set_default(args, "adapter_heads", first_nested(config, ["patch_alignment.adapter_heads"]), 8)
    set_default(args, "projection_dim", first_nested(config, ["patch_alignment.projection_dim"]), None)
    set_default(
        args,
        "alignment_projection_enabled",
        first_nested(config, ["patch_alignment.alignment_projection.enabled"]),
        False,
    )
    set_default(args, "alignment_projection_dim", first_nested(config, ["patch_alignment.alignment_projection.dim"]), 512)
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
        "centered_contrastive_loss_weight",
        first_nested(config, ["patch_alignment.centered_contrastive_loss_weight"]),
        0.0,
    )
    set_default(args, "cosine_loss_weight", first_nested(config, ["patch_alignment.cosine_loss_weight"]), 0.0)
    set_default(args, "reconstruction_loss_weight", first_nested(config, ["patch_alignment.reconstruction_loss_weight"]), 1.0)
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
        "point_sign,point_relation,region_mean_relation,region_range_relation,directional_change",
    )
    set_default(args, "probe_region_size", first_nested(config, ["patch_alignment.probe_region_size"]), 4)
    set_default(args, "evaluation_probe_count", first_nested(config, ["patch_alignment.evaluation_probe_count"]), 3)
    set_default(args, "probe_answer_ce_weight", first_nested(config, ["patch_alignment.probe_answer_ce_weight"]), 0.2)
    set_default(args, "probe_teacher_kl_weight", first_nested(config, ["patch_alignment.probe_teacher_kl_weight"]), 0.2)
    set_default(args, "probe_kl_temperature", first_nested(config, ["patch_alignment.probe_kl_temperature"]), 1.0)
    set_default(
        args,
        "probe_teacher_preflight_records",
        first_nested(config, ["patch_alignment.probe_teacher_preflight_records"]),
        8,
    )
    set_default(
        args,
        "max_shared_suffix_tokens",
        first_nested(config, ["patch_alignment.max_shared_suffix_tokens"]),
        96,
    )
    set_default(args, "center_embeddings", first_nested(config, ["patch_alignment.center_embeddings"]), False)
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
            for name in ("probe_answer_ce_weight", "probe_teacher_kl_weight"):
                if float(getattr(args, name)) < 0.0:
                    raise ValueError(f"patch_alignment.{name} must be non-negative.")
            if float(args.probe_kl_temperature) <= 0.0:
                raise ValueError("patch_alignment.probe_kl_temperature must be positive.")
            if int(args.probe_teacher_preflight_records) < 0:
                raise ValueError("patch_alignment.probe_teacher_preflight_records must be non-negative.")
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
    if bool(args.alignment_projection_enabled):
        if int(args.alignment_projection_dim) <= 0:
            raise ValueError("patch_alignment.alignment_projection.dim must be positive.")
        if int(args.alignment_projection_hidden_dim) <= 0:
            raise ValueError("patch_alignment.alignment_projection.hidden_dim must be positive.")
        if int(args.alignment_projection_layers) <= 0:
            raise ValueError("patch_alignment.alignment_projection.layers must be positive.")
        if float(args.alignment_projection_dropout) < 0.0:
            raise ValueError("patch_alignment.alignment_projection.dropout must be non-negative.")
    if float(args.soft_prompt_scale) < 0.0:
        raise ValueError("patch_alignment.soft_prompt_scale must be non-negative.")
    if float(args.temperature) <= 0.0:
        raise ValueError("patch_alignment.temperature must be positive.")
    for name in (
        "contrastive_loss_weight",
        "centered_contrastive_loss_weight",
        "cosine_loss_weight",
        "reconstruction_loss_weight",
    ):
        if float(getattr(args, name)) < 0.0:
            raise ValueError(f"patch_alignment.{name} must be non-negative.")
    return args


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    sampler: torch.utils.data.Sampler | None = None,
    drop_last: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle) if sampler is None else False,
        sampler=sampler,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=bool(drop_last),
        collate_fn=collate_patch_text,
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
            "probe_contract_templates_per_family": float(PROBE_TEMPLATES_PER_FAMILY),
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
            source_patches = (
                batch["patch"]
                if str(args.teacher_text_source).lower() == "raw"
                else normalized_patches
            )
            visible_patches = quantize_probe_values(source_patches, int(args.text_decimal_places))
            active_class_counts: list[int] = []
            majority_fractions: list[float] = []
            for anchor in contract_anchors:
                targets = probe_target_indices(visible_patches, anchor)
                class_counts = torch.bincount(targets.cpu(), minlength=len(anchor.probe_answers))
                active_class_count = int((class_counts > 0).sum().item())
                majority_fraction = float(class_counts.max().item() / max(1, int(targets.numel())))
                active_class_counts.append(active_class_count)
                majority_fractions.append(majority_fraction)
                summary[f"anchor_{anchor.name}_target_active_classes"] = float(active_class_count)
                summary[f"anchor_{anchor.name}_target_majority_fraction"] = majority_fraction
                for class_index, count in enumerate(class_counts.tolist()):
                    summary[f"anchor_{anchor.name}_target_class_{class_index}_fraction"] = float(
                        count / max(1, int(targets.numel()))
                    )
            summary["probe_contract_anchor_count"] = float(len(contract_anchors))
            summary["probe_contract_family_count"] = float(len(parse_csv(args.probe_families)))
            summary["probe_contract_templates_per_family"] = float(PROBE_TEMPLATES_PER_FAMILY)
            summary["probe_target_min_active_classes"] = float(min(active_class_counts))
            summary["probe_target_max_majority_fraction"] = float(max(majority_fractions))
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
def preflight_probe_teacher_behavior(
    *,
    dataset: Dataset,
    tokenizer: Any,
    llm: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    compressor_input_size: Sequence[int],
    normalization_cfg: Mapping[str, Any],
) -> dict[str, float]:
    if str(args.alignment_anchor_mode) != "probe":
        return {}
    record_count = min(int(args.probe_teacher_preflight_records), len(dataset))
    if record_count <= 0:
        return {}
    items = [dataset[index] for index in range(record_count)]
    batch = collate_patch_text(items)
    normalized_patches = normalize_patch_batch(
        batch["patch"],
        compressor_input_size,
        normalization_cfg,
        bool(args.resize_patch_to_compressor_input),
    )
    texts = build_teacher_texts_for_batch(batch, normalized_patches, args)
    source_patches = (
        batch["patch"]
        if str(args.teacher_text_source).lower() == "raw"
        else normalized_patches
    )
    visible_patches = quantize_probe_values(source_patches, int(args.text_decimal_places))
    anchors = probe_contract_anchors(tokenizer, args)
    output_embeddings = llm.get_output_embeddings()
    if output_embeddings is None:
        raise ValueError("The frozen causal LLM must expose an output embedding layer for probe preflight.")

    total_records = 0
    total_completion_correct = 0
    total_strict_correct = 0
    total_format_correct = 0
    total_answer_nll = 0.0
    family_totals: dict[str, dict[str, float]] = {}
    metrics: dict[str, float] = {
        "anchor_count": float(len(anchors)),
        "record_count_per_anchor": float(record_count),
    }
    progress = tqdm(
        anchors,
        desc="probe teacher preflight",
        leave=False,
        disable=not is_main_process(),
    )
    for anchor in progress:
        teacher_output = text_teacher_hidden(
            llm,
            tokenizer,
            texts,
            device,
            int(args.max_text_tokens),
            int(args.teacher_layer),
            bool(args.fail_on_text_anchor_missing) and str(args.text_prompt_template) != "plain",
            bool(args.fail_on_text_max_length_hit),
            text_layout=str(args.alignment_text_layout),
            shared_suffix=str(args.shared_suffix),
            max_shared_suffix_tokens=int(args.max_shared_suffix_tokens),
            alignment_anchor=anchor,
        )
        targets = probe_target_indices(visible_patches, anchor).to(device=device, dtype=torch.long)
        continuation_ids = torch.tensor(
            probe_answer_token_ids(tokenizer, anchor),
            dtype=torch.long,
            device=device,
        )
        answer_token_ids = continuation_ids.index_select(0, targets)
        vocab_logits = output_embeddings(teacher_output.final_hidden).float()
        continuation_logits = vocab_logits.index_select(1, continuation_ids)
        completion_correct = int((continuation_logits.argmax(dim=-1) == targets).sum().item())
        top_tokens = vocab_logits.argmax(dim=-1)
        strict_correct = int((top_tokens == answer_token_ids).sum().item())
        format_correct = int(
            (top_tokens.unsqueeze(1) == continuation_ids.unsqueeze(0)).any(dim=1).sum().item()
        )
        answer_nll_sum = float(F.cross_entropy(vocab_logits, answer_token_ids, reduction="sum").item())
        anchor_records = int(targets.numel())
        anchor_prefix = f"anchor_{anchor.name}"
        metrics[f"{anchor_prefix}_completion_accuracy"] = completion_correct / max(1, anchor_records)
        metrics[f"{anchor_prefix}_strict_accuracy"] = strict_correct / max(1, anchor_records)
        metrics[f"{anchor_prefix}_format_rate"] = format_correct / max(1, anchor_records)
        metrics[f"{anchor_prefix}_answer_nll"] = answer_nll_sum / max(1, anchor_records)

        family = str(anchor.probe_family)
        family_total = family_totals.setdefault(
            family,
            {"records": 0.0, "completion": 0.0, "strict": 0.0, "format": 0.0, "nll": 0.0},
        )
        family_total["records"] += anchor_records
        family_total["completion"] += completion_correct
        family_total["strict"] += strict_correct
        family_total["format"] += format_correct
        family_total["nll"] += answer_nll_sum
        total_records += anchor_records
        total_completion_correct += completion_correct
        total_strict_correct += strict_correct
        total_format_correct += format_correct
        total_answer_nll += answer_nll_sum

    metrics["teacher_completion_accuracy"] = total_completion_correct / max(1, total_records)
    metrics["teacher_strict_accuracy"] = total_strict_correct / max(1, total_records)
    metrics["teacher_answer_format_rate"] = total_format_correct / max(1, total_records)
    metrics["teacher_answer_nll"] = total_answer_nll / max(1, total_records)
    metrics["teacher_kl_active_fraction"] = metrics["teacher_strict_accuracy"]
    for family, totals in family_totals.items():
        family_records = max(1.0, totals["records"])
        metrics[f"family_{family}_completion_accuracy"] = totals["completion"] / family_records
        metrics[f"family_{family}_strict_accuracy"] = totals["strict"] / family_records
        metrics[f"family_{family}_format_rate"] = totals["format"] / family_records
        metrics[f"family_{family}_answer_nll"] = totals["nll"] / family_records

    all_families_activate_kl = all(totals["strict"] > 0.0 for totals in family_totals.values())
    metrics["teacher_kl_viable"] = float(
        float(args.probe_teacher_kl_weight) <= 0.0 or all_families_activate_kl
    )
    return metrics


def save_checkpoint(
    path: Path,
    *,
    compressor: nn.Module,
    adapter: TensorPatchAlignmentAdapter,
    projector: AlignmentProjectionPair | None,
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
        payload["alignment_projector_state_dict"] = projector.state_dict()
    if save_compressor:
        payload["compressor_state_dict"] = compressor.state_dict()
    torch.save(payload, path)


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
        f"global_i2t={fmt_metric(metrics, 'global_i2t_accuracy')} "
        f"global_t2i={fmt_metric(metrics, 'global_t2i_accuracy')} "
        f"completion={fmt_metric(metrics, 'probe_completion_accuracy')} "
        f"shuffled={fmt_metric(metrics, 'probe_shuffled_completion_accuracy')} "
        f"gain={fmt_metric(metrics, 'probe_completion_latent_gain')} "
        f"strict={fmt_metric(metrics, 'probe_answer_token_accuracy')} "
        f"format={fmt_metric(metrics, 'probe_answer_format_rate')} "
        f"teacher_completion={fmt_metric(metrics, 'probe_teacher_completion_accuracy')} "
        f"teacher_strict={fmt_metric(metrics, 'probe_teacher_answer_token_accuracy')} "
        f"answer_ce={fmt_metric(metrics, 'probe_answer_ce')} "
        f"teacher_kl={fmt_metric(metrics, 'probe_teacher_kl')}"
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
            "compressor_checkpoint": args.compressor_checkpoint,
            "resize_patch_to_compressor_input": bool(args.resize_patch_to_compressor_input),
            "adapter_type": str(args.adapter_type),
            "adapter_dim": int(args.adapter_dim),
            "query_tokens": int(args.query_tokens),
            "adapter_layers": int(args.adapter_layers),
            "adapter_heads": int(args.adapter_heads),
            "projection_dim": args.projection_dim,
            "alignment_projection": {
                "enabled": bool(args.alignment_projection_enabled),
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
            "centered_contrastive_loss_weight": float(args.centered_contrastive_loss_weight),
            "cosine_loss_weight": float(args.cosine_loss_weight),
            "reconstruction_loss_weight": float(args.reconstruction_loss_weight),
            "teacher_text_source": str(args.teacher_text_source),
            "alignment_text_layout": str(args.alignment_text_layout),
            "shared_suffix": str(args.shared_suffix),
            "alignment_anchor_mode": str(args.alignment_anchor_mode),
            "representation_suffix": str(args.representation_suffix),
            "probe_families": list(args.probe_families),
            "probe_region_size": int(args.probe_region_size),
            "evaluation_probe_count": int(args.evaluation_probe_count),
            "probe_answer_ce_weight": float(args.probe_answer_ce_weight),
            "probe_teacher_kl_weight": float(args.probe_teacher_kl_weight),
            "probe_kl_temperature": float(args.probe_kl_temperature),
            "probe_teacher_preflight_records": int(args.probe_teacher_preflight_records),
            "max_shared_suffix_tokens": int(args.max_shared_suffix_tokens),
            "center_embeddings": bool(args.center_embeddings),
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
    args = parse_args()
    config = load_yaml_mapping(args.config)
    args = apply_config_defaults(args, config)
    setup_distributed_from_env(args)
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
    validate_field_shapes(args.hdf5_path, field_keys)
    compressor_input_size = tuple(int(dim) for dim in compressor_config["model"]["input_size"])
    normalization_cfg = dict(compressor_config.get("data", {}).get("dataset", {}).get("normalization", {}))
    if not bool(args.resize_patch_to_compressor_input) and tuple(compressor_input_size) != (
        int(args.patch_size),
        int(args.patch_size),
    ):
        raise ValueError(
            "--no-resize-patch-to-compressor-input requires compressor input_size to match patch_size. "
            f"Got input_size={compressor_input_size}, patch_size={args.patch_size}."
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
    llm = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        trust_remote_code=bool(args.trust_remote_code),
        torch_dtype=dtype_from_name(str(args.torch_dtype)),
    )
    llm.to(device)
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
    val_loader = make_loader(val_dataset, int(args.eval_batch_size), False, int(args.num_workers))
    test_loader = make_loader(test_dataset, int(args.eval_batch_size), False, int(args.num_workers))
    probe_contract_anchor_bank = probe_contract_anchors(tokenizer, args)
    if is_main_process() and probe_contract_anchor_bank:
        dump_json(
            run_dir / "probe_contract.json",
            {
                "family_count": len(parse_csv(args.probe_families)),
                "templates_per_family": PROBE_TEMPLATES_PER_FAMILY,
                "answers_are_input": False,
                "anchors": [
                    {
                        "name": anchor.name,
                        "text": anchor.text,
                        "token_ids": list(anchor.token_ids),
                        "token_count": len(anchor.token_ids),
                        "probe_family": anchor.probe_family,
                        "probe_template_index": anchor.probe_template_index,
                        "probe_parameters": list(anchor.probe_parameters),
                        "probe_answers": list(anchor.probe_answers),
                        "probe_answer_token_ids": list(probe_answer_token_ids(tokenizer, anchor)),
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
    if (
        str(args.alignment_anchor_mode) == "probe"
        and int(args.text_preflight_records) > 0
        and text_preflight_metrics.get("probe_target_min_active_classes", 0.0) < 2.0
    ):
        raise ValueError(
            "Probe tokenization preflight found a family/template with fewer than two target classes. "
            f"See {run_dir / 'probe_tokenization_preflight.json'}. Increase text_preflight_records or "
            "revise field/record sampling before a long run."
        )
    if (
        is_main_process()
        and str(args.alignment_anchor_mode) == "probe"
        and int(args.probe_teacher_preflight_records) > 0
    ):
        print(
            "probe_teacher_preflight_start "
            f"families={len(parse_csv(args.probe_families))} "
            f"templates_per_family={PROBE_TEMPLATES_PER_FAMILY} "
            f"records_per_template={min(int(args.probe_teacher_preflight_records), len(train_dataset))}"
        )
    probe_teacher_preflight_metrics = preflight_probe_teacher_behavior(
        dataset=train_dataset,
        tokenizer=tokenizer,
        llm=llm,
        device=device,
        args=args,
        compressor_input_size=compressor_input_size,
        normalization_cfg=normalization_cfg,
    )
    if is_main_process() and probe_teacher_preflight_metrics:
        dump_json(run_dir / "probe_teacher_preflight.json", probe_teacher_preflight_metrics)
        print(
            "probe_teacher_preflight "
            f"completion={probe_teacher_preflight_metrics['teacher_completion_accuracy']:.4f} "
            f"strict={probe_teacher_preflight_metrics['teacher_strict_accuracy']:.4f} "
            f"format={probe_teacher_preflight_metrics['teacher_answer_format_rate']:.4f} "
            f"kl_active={probe_teacher_preflight_metrics['teacher_kl_active_fraction']:.4f} "
            f"answer_nll={probe_teacher_preflight_metrics['teacher_answer_nll']:.4f}"
        )
    if probe_teacher_preflight_metrics:
        distributed_barrier()
    if (
        probe_teacher_preflight_metrics
        and float(args.probe_teacher_kl_weight) > 0.0
        and probe_teacher_preflight_metrics["teacher_kl_viable"] == 0.0
    ):
        missing_kl_families = [
            family
            for family in parse_csv(args.probe_families)
            if probe_teacher_preflight_metrics.get(f"family_{family}_strict_accuracy", 0.0) == 0.0
        ]
        raise ValueError(
            "Probe Teacher preflight found no strict canonical continuation for these families: "
            f"{missing_kl_families}. The configured Teacher KL would be inactive for them. "
            f"See {run_dir / 'probe_contract.json'} and {run_dir / 'probe_teacher_preflight.json'}, "
            "revise the continuation interface, "
            "or explicitly set patch_alignment.probe_teacher_kl_weight: 0 before a long run."
        )
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
    alignment_projector: AlignmentProjectionPair | None = None
    if bool(args.alignment_projection_enabled):
        alignment_projector = AlignmentProjectionPair(
            input_dim=llm_hidden_size,
            output_dim=int(args.alignment_projection_dim),
            hidden_dim=int(args.alignment_projection_hidden_dim),
            layers=int(args.alignment_projection_layers),
            dropout=float(args.alignment_projection_dropout),
            shared=bool(args.alignment_projection_shared),
        ).to(device)
    broadcast_module_state(compressor)
    broadcast_module_state(adapter)
    broadcast_module_state(alignment_projector)

    args.alignment_train_patch_ae = bool(args.train_patch_ae) and not (
        bool(args.freeze_patch_ae_after_pretrain) and int(args.patch_ae_pretrain_epochs) > 0
    )

    run_summary = {
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
        "encoder_source": str(args.encoder_source),
        "compressor_checkpoint": str(args.compressor_checkpoint) if args.compressor_checkpoint else None,
        "train_patch_ae": bool(args.train_patch_ae),
        "freeze_patch_ae_after_pretrain": bool(args.freeze_patch_ae_after_pretrain),
        "alignment_train_patch_ae": bool(args.alignment_train_patch_ae),
        "patch_ae_pretrain_epochs": int(args.patch_ae_pretrain_epochs),
        "reconstruction_loss_weight": float(args.reconstruction_loss_weight),
        "resize_patch_to_compressor_input": bool(args.resize_patch_to_compressor_input),
        "compressor_input_size": list(compressor_input_size),
        "latent_channels": int(latent_channels),
        "latent_grid": list(latent_grid),
        "latent_token_count": int(latent_grid[0] * latent_grid[1]),
        "llm_hidden_size": int(llm_hidden_size),
        "llm_num_hidden_layers": int(getattr(llm.config, "num_hidden_layers", -1)),
        "projection_dim": int(projection_dim),
        "alignment_projection": {
            "enabled": bool(args.alignment_projection_enabled),
            "dim": int(args.alignment_projection_dim),
            "hidden_dim": int(args.alignment_projection_hidden_dim),
            "layers": int(args.alignment_projection_layers),
            "dropout": float(args.alignment_projection_dropout),
            "shared": bool(args.alignment_projection_shared),
            "parameters": (
                sum(parameter.numel() for parameter in alignment_projector.parameters())
                if alignment_projector is not None
                else 0
            ),
        },
        "teacher_layer": int(args.teacher_layer),
        "teacher_text_source": str(args.teacher_text_source),
        "alignment_text_layout": str(args.alignment_text_layout),
        "alignment_anchor_mode": str(args.alignment_anchor_mode),
        "representation_suffix": str(args.representation_suffix),
        "probe_families": list(args.probe_families),
        "probe_region_size": int(args.probe_region_size),
        "evaluation_probe_count": int(args.evaluation_probe_count),
        "probe_answer_ce_weight": float(args.probe_answer_ce_weight),
        "probe_teacher_kl_weight": float(args.probe_teacher_kl_weight),
        "probe_kl_temperature": float(args.probe_kl_temperature),
        "probe_teacher_preflight_records": int(args.probe_teacher_preflight_records),
        "probe_contract_anchors": [
            {
                "name": anchor.name,
                "text": anchor.text,
                "token_ids": list(anchor.token_ids),
                "token_count": len(anchor.token_ids),
                "probe_family": anchor.probe_family,
                "probe_template_index": anchor.probe_template_index,
                "probe_parameters": list(anchor.probe_parameters),
                "probe_answers": list(anchor.probe_answers),
                "probe_answer_token_ids": list(probe_answer_token_ids(tokenizer, anchor)),
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
                "probe_answers": list(anchor.probe_answers),
                "probe_answer_token_ids": (
                    list(probe_answer_token_ids(tokenizer, anchor)) if anchor.mode == "probe" else []
                ),
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
                "probe_answers": list(anchor.probe_answers),
                "probe_answer_token_ids": (
                    list(probe_answer_token_ids(tokenizer, anchor)) if anchor.mode == "probe" else []
                ),
            }
            for anchor in evaluation_anchors
        ],
        "anchor_aggregation": {
            "training_schedule": "one shared probe per batch; families and four templates cycle uniformly",
            "primary_validation": "mean over fixed probes only when probe mode is selected",
            "global_retrieval": "separate_candidate_library_per_anchor",
            "probe_hidden": "conditioned_hidden_minus_probe_only_hidden",
        },
        "max_shared_suffix_tokens": int(args.max_shared_suffix_tokens),
        "center_embeddings": bool(args.center_embeddings),
        "contrastive_loss_weight": float(args.contrastive_loss_weight),
        "centered_contrastive_loss_weight": float(args.centered_contrastive_loss_weight),
        "cosine_loss_weight": float(args.cosine_loss_weight),
        "fail_on_text_anchor_missing": bool(args.fail_on_text_anchor_missing),
        "fail_on_text_max_length_hit": bool(args.fail_on_text_max_length_hit),
        "global_retrieval_eval": bool(args.global_retrieval_eval),
        "global_retrieval_max_records": int(args.global_retrieval_max_records),
        "global_retrieval_chunk_size": int(args.global_retrieval_chunk_size),
        "text_preflight_records": int(args.text_preflight_records),
        "text_preflight": dict(text_preflight_metrics),
        "probe_teacher_preflight": dict(probe_teacher_preflight_metrics),
        "adapter_type": str(args.adapter_type),
        "query_tokens": int(args.query_tokens),
        "adapter_layers": int(args.adapter_layers),
        "adapter_heads": int(args.adapter_heads),
        "soft_prompt_scale": float(args.soft_prompt_scale),
        "alignment_mode": (
            "input_soft_prompt_natural_completion_probe"
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
            "negative_gather": "embedding_all_gather",
            "train_drop_last": bool(distributed_is_initialized()),
        },
        "adapter_parameters": sum(parameter.numel() for parameter in adapter.parameters() if parameter.requires_grad),
        "alignment_projector_parameters": (
            sum(parameter.numel() for parameter in alignment_projector.parameters() if parameter.requires_grad)
            if alignment_projector is not None
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
            f"center_embeddings={bool(args.center_embeddings)} "
            f"contrastive_weight={float(args.contrastive_loss_weight):.4g} "
            f"centered_contrastive_weight={float(args.centered_contrastive_loss_weight):.4g} "
            f"cosine_weight={float(args.cosine_loss_weight):.4g} "
            f"projection_enabled={bool(args.alignment_projection_enabled)} "
            f"projection_dim={int(args.alignment_projection_dim)} "
            f"projection_layers={int(args.alignment_projection_layers)} "
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
                f"probe_anchors={text_preflight_metrics.get('probe_contract_anchor_count', 0.0):.0f} "
                f"min_active_classes={text_preflight_metrics.get('probe_target_min_active_classes', 0.0):.0f} "
                f"max_target_majority={text_preflight_metrics.get('probe_target_max_majority_fraction', 0.0):.3f}"
            )
    wandb_logger = WandbLogger(config=build_wandb_config(args, run_summary), run_dir=run_dir) if is_main_process() else None
    global_step = 0

    metrics_history: dict[str, Any] = {}
    if bool(args.train_patch_ae) and int(args.patch_ae_pretrain_epochs) > 0:
        compressor_optimizer = torch.optim.AdamW(
            [parameter for parameter in compressor.parameters() if parameter.requires_grad],
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
        )
        for pretrain_epoch in range(1, int(args.patch_ae_pretrain_epochs) + 1):
            pretrain_metrics, global_step = pretrain_patch_encoder_one_epoch(
                compressor=compressor,
                loader=train_loader,
                optimizer=compressor_optimizer,
                device=device,
                args=args,
                compressor_input_size=compressor_input_size,
                normalization_cfg=normalization_cfg,
                epoch=pretrain_epoch,
                wandb_logger=wandb_logger,
                global_step=global_step,
            )
            pretrain_val_metrics = evaluate_patch_encoder_reconstruction(
                compressor=compressor,
                loader=val_loader,
                device=device,
                args=args,
                compressor_input_size=compressor_input_size,
                normalization_cfg=normalization_cfg,
            ) if is_main_process() else {}
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
                print(
                    f"patch_ae_pretrain_epoch={pretrain_epoch:04d} "
                    f"train_recon={pretrain_metrics['reconstruction_loss']:.4f} "
                    f"val_recon={pretrain_val_metrics['reconstruction_loss']:.4f}"
                )
            distributed_barrier()

    if bool(args.alignment_train_patch_ae):
        for parameter in compressor.parameters():
            parameter.requires_grad_(True)
    else:
        for parameter in compressor.parameters():
            parameter.requires_grad_(False)
        compressor.eval()
    optimizer_params = list(adapter.parameters())
    if alignment_projector is not None:
        optimizer_params += list(alignment_projector.parameters())
    if bool(args.alignment_train_patch_ae):
        optimizer_params += [parameter for parameter in compressor.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(optimizer_params, lr=float(args.lr), weight_decay=float(args.weight_decay))

    best_val = float("inf")
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
        if is_main_process():
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
                save_compressor=bool(args.train_patch_ae),
            )
            if float(val_metrics["loss"]) < best_val:
                best_val = float(val_metrics["loss"])
                best_epoch = int(epoch)
                save_checkpoint(
                    run_dir / "alignment_best.pt",
                    compressor=compressor,
                    adapter=adapter,
                    projector=alignment_projector,
                    args=args,
                    metrics=epoch_metrics,
                    compressor_config=compressor_config,
                    save_compressor=bool(args.train_patch_ae),
                )
            wandb_payload = {
                "epoch": float(epoch),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "best_val/loss": float(best_val),
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
                f"val[{alignment_metric_summary(val_metrics)}]"
            )
        distributed_barrier()

    if is_main_process():
        best_checkpoint = torch.load(run_dir / "alignment_best.pt", map_location=device)
        adapter.load_state_dict(best_checkpoint["adapter_state_dict"])
        if alignment_projector is not None and "alignment_projector_state_dict" in best_checkpoint:
            alignment_projector.load_state_dict(best_checkpoint["alignment_projector_state_dict"])
        if bool(args.train_patch_ae) and "compressor_state_dict" in best_checkpoint:
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
        metrics_history["best"] = {"epoch": int(best_epoch), "val_loss": float(best_val)}
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
        print(f"Run directory: {run_dir}")
        print(
            f"best_epoch={best_epoch} test_loss={float(test_metrics['loss']):.4f} "
            f"test[{alignment_metric_summary(test_metrics)}]"
        )
    distributed_barrier()
    cleanup_distributed()


if __name__ == "__main__":
    main()
