from __future__ import annotations

import argparse
import json
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
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
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


@dataclass(frozen=True)
class HiddenBatch:
    hidden: torch.Tensor
    metrics: dict[str, float]


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


def hdf5_axis_sizes(hdf5_path: str | Path, field: str) -> tuple[int, int, int, int]:
    with h5py.File(Path(hdf5_path).expanduser(), "r") as handle:
        if field not in handle or not isinstance(handle[field], h5py.Dataset):
            raise KeyError(f"HDF5 dataset key {field!r} not found in {hdf5_path}.")
        shape = tuple(int(dim) for dim in handle[field].shape)
    if len(shape) != 4:
        raise ValueError(f"Expected PDEBench 2D field [sample,time,height,width], got {shape}.")
    return shape


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


def record_key(record: PatchRecord) -> tuple[int, int, int, int]:
    return (int(record.sample_index), int(record.time_index), int(record.row), int(record.col))


def summarize_records(records: Sequence[PatchRecord]) -> dict[str, Any]:
    sample_values = sorted({int(record.sample_index) for record in records})
    time_values = sorted({int(record.time_index) for record in records})
    return {
        "record_count": len(records),
        "unique_record_count": len({record_key(record) for record in records}),
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
    records: list[PatchRecord] = []
    seen: set[tuple[int, int, int, int]] = set()
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
        patch = self._read_patch(record)
        text = self._serialize_patch(record, patch) if self.include_raw_text else ""
        return {
            "record": {
                "sample_index": int(record.sample_index),
                "time_index": int(record.time_index),
                "row": int(record.row),
                "col": int(record.col),
                "patch_size": int(self.patch_size),
                "fields": list(self.field_keys),
            },
            "patch": patch,
            "text": text,
        }

    def _read_patch(self, record: PatchRecord) -> torch.Tensor:
        arrays: list[np.ndarray] = []
        row_slice = slice(int(record.row), int(record.row) + self.patch_size)
        col_slice = slice(int(record.col), int(record.col) + self.patch_size)
        with h5py.File(self.hdf5_path, "r") as handle:
            for field in self.field_keys:
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
            field_keys=self.field_keys,
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

    def forward(self, queries: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        self_attended, _ = self.self_attn(
            self.self_norm(queries),
            self.self_norm(queries),
            self.self_norm(queries),
            need_weights=False,
        )
        queries = queries + self_attended
        cross_attended, _ = self.cross_attn(
            self.cross_query_norm(queries),
            self.cross_context_norm(context),
            self.cross_context_norm(context),
            need_weights=False,
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
) -> HiddenBatch:
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
    outputs = llm(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    hidden = hidden_at_last_non_padding(outputs.hidden_states, attention_mask, teacher_layer).detach()
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
) -> HiddenBatch:
    encoded = tokenizer(
        build_student_anchor_texts(records, patch_size_override=patch_size),
        padding=True,
        truncation=True,
        max_length=int(max_tokens),
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    text_attention_mask = encoded["attention_mask"].to(device)
    text_embeds = llm.get_input_embeddings()(input_ids)
    soft_prompts = soft_prompts.to(device=device, dtype=text_embeds.dtype)
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
    metrics["text_token_embedding_norm"] = masked_token_norm(text_embeds, text_attention_mask)
    metrics["soft_prompt_token_norm"] = mean_token_norm(soft_prompts)
    inputs_embeds = torch.cat([soft_prompts, text_embeds], dim=1)
    soft_attention = torch.ones(
        (input_ids.shape[0], soft_prompts.shape[1]),
        dtype=text_attention_mask.dtype,
        device=device,
    )
    attention_mask = torch.cat([soft_attention, text_attention_mask], dim=1)
    outputs = llm(
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


def train_compressor_during_alignment(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "alignment_train_patch_ae", args.train_patch_ae))


def train_one_epoch(
    *,
    compressor: nn.Module,
    adapter: TensorPatchAlignmentAdapter,
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
    adapter.train()
    train_compressor = train_compressor_during_alignment(args)
    if train_compressor:
        compressor.train()
    else:
        compressor.eval()
    total_loss = 0.0
    total_contrastive = 0.0
    total_cosine = 0.0
    total_reconstruction = 0.0
    total_i2t = 0.0
    total_t2i = 0.0
    total_records = 0
    metric_totals: dict[str, float] = {}
    progress = tqdm(loader, desc=f"train align epoch {epoch}", leave=False)
    for step, batch in enumerate(progress, start=1):
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
                bool(args.fail_on_text_max_length_hit) and str(args.text_prompt_template) != "plain",
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
            bool(args.fail_on_text_max_length_hit),
        )
        student_hidden = student_output.hidden
        teacher_hidden = teacher_output.hidden
        tensor_embedding, text_embedding = prepare_alignment_embeddings(
            student_hidden,
            teacher_hidden.to(dtype=student_hidden.dtype),
            bool(args.center_embeddings),
        )
        centered_tensor_embedding, centered_text_embedding = prepare_alignment_embeddings(
            student_hidden,
            teacher_hidden.to(dtype=student_hidden.dtype),
            True,
        )
        uncentered_tensor_embedding, uncentered_text_embedding = prepare_alignment_embeddings(
            student_hidden,
            teacher_hidden.to(dtype=student_hidden.dtype),
            False,
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
        reconstruction_weight = float(args.reconstruction_loss_weight) if train_compressor else 0.0
        loss = (
            float(args.contrastive_loss_weight) * contrastive
            + float(args.centered_contrastive_loss_weight) * centered_contrastive
            + float(args.cosine_loss_weight) * cosine
            + reconstruction_weight * reconstruction
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
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
        add_weighted_metrics(metric_totals, teacher_output.metrics, batch_size, "teacher_")
        add_weighted_metrics(metric_totals, student_output.metrics, batch_size, "student_")
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
            )
    metrics = {
        "loss": total_loss / max(1, total_records),
        "contrastive_loss": total_contrastive / max(1, total_records),
        "cosine_loss": total_cosine / max(1, total_records),
        "reconstruction_loss": total_reconstruction / max(1, total_records),
        "i2t_accuracy": total_i2t / max(1, total_records),
        "t2i_accuracy": total_t2i / max(1, total_records),
    }
    metrics.update(averaged_metrics(metric_totals, total_records))
    return metrics


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
    compressor.train()
    total_loss = 0.0
    total_records = 0
    progress = tqdm(loader, desc=f"pretrain patch AE epoch {epoch}", leave=False)
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
    return {"reconstruction_loss": total_loss / max(1, total_records)}, global_step


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
    for batch in tqdm(loader, desc="eval patch AE reconstruction", leave=False):
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
    llm: nn.Module,
    tokenizer: Any,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    compressor_input_size: Sequence[int],
    normalization_cfg: Mapping[str, Any],
) -> dict[str, float]:
    compressor.eval()
    adapter.eval()
    total_loss = 0.0
    total_contrastive = 0.0
    total_cosine = 0.0
    total_reconstruction = 0.0
    total_i2t = 0.0
    total_t2i = 0.0
    total_records = 0
    metric_totals: dict[str, float] = {}
    collected_student_hidden: list[torch.Tensor] = []
    collected_teacher_hidden: list[torch.Tensor] = []
    train_compressor = train_compressor_during_alignment(args)
    for batch in tqdm(loader, desc="eval align", leave=False):
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
            bool(args.fail_on_text_max_length_hit) and str(args.text_prompt_template) != "plain",
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
        )
        student_hidden = student_output.hidden
        teacher_hidden = teacher_output.hidden
        tensor_embedding, text_embedding = prepare_alignment_embeddings(
            student_hidden,
            teacher_hidden.to(dtype=student_hidden.dtype),
            bool(args.center_embeddings),
        )
        centered_tensor_embedding, centered_text_embedding = prepare_alignment_embeddings(
            student_hidden,
            teacher_hidden.to(dtype=student_hidden.dtype),
            True,
        )
        uncentered_tensor_embedding, uncentered_text_embedding = prepare_alignment_embeddings(
            student_hidden,
            teacher_hidden.to(dtype=student_hidden.dtype),
            False,
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
        reconstruction_weight = float(args.reconstruction_loss_weight) if train_compressor else 0.0
        loss = (
            float(args.contrastive_loss_weight) * contrastive
            + float(args.centered_contrastive_loss_weight) * centered_contrastive
            + float(args.cosine_loss_weight) * cosine
            + reconstruction_weight * reconstruction
        )
        batch_size = int(patches.shape[0])
        total_loss += float(loss.detach().cpu().item()) * batch_size
        total_contrastive += float(contrastive.detach().cpu().item()) * batch_size
        total_cosine += float(cosine.detach().cpu().item()) * batch_size
        total_reconstruction += float(reconstruction.detach().cpu().item()) * batch_size
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
        add_weighted_metrics(metric_totals, teacher_output.metrics, batch_size, "teacher_")
        add_weighted_metrics(metric_totals, student_output.metrics, batch_size, "student_")
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
            },
            batch_size,
            "alignment_",
        )
        total_records += batch_size
        if bool(args.global_retrieval_eval) and total_records <= int(args.global_retrieval_max_records):
            collected_student_hidden.append(student_hidden.detach().float().cpu())
            collected_teacher_hidden.append(teacher_hidden.detach().float().cpu())
    metrics = {
        "loss": total_loss / max(1, total_records),
        "contrastive_loss": total_contrastive / max(1, total_records),
        "cosine_loss": total_cosine / max(1, total_records),
        "reconstruction_loss": total_reconstruction / max(1, total_records),
        "i2t_accuracy": total_i2t / max(1, total_records),
        "t2i_accuracy": total_t2i / max(1, total_records),
    }
    metrics.update(averaged_metrics(metric_totals, total_records))
    if (
        bool(args.global_retrieval_eval)
        and collected_student_hidden
        and total_records <= int(args.global_retrieval_max_records)
    ):
        student_all = torch.cat(collected_student_hidden, dim=0)
        teacher_all = torch.cat(collected_teacher_hidden, dim=0)
        global_tensor, global_text = prepare_alignment_embeddings(student_all, teacher_all, bool(args.center_embeddings))
        global_centered_tensor, global_centered_text = prepare_alignment_embeddings(student_all, teacher_all, True)
        global_uncentered_tensor, global_uncentered_text = prepare_alignment_embeddings(student_all, teacher_all, False)
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
    elif bool(args.global_retrieval_eval):
        metrics["global_retrieval_skipped"] = 1.0
    return metrics


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
    if int(args.text_preflight_records) < 0:
        raise ValueError("patch_alignment.text_preflight_records must be non-negative.")
    if int(args.global_retrieval_max_records) <= 0:
        raise ValueError("patch_alignment.global_retrieval_max_records must be positive.")
    if int(args.global_retrieval_chunk_size) <= 0:
        raise ValueError("patch_alignment.global_retrieval_chunk_size must be positive.")
    split_ratio_sum = float(args.split_train_ratio) + float(args.split_val_ratio) + float(args.split_test_ratio)
    if split_ratio_sum <= 0.0:
        raise ValueError("patch_alignment split ratios must sum to a positive value.")
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


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
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
    record_count = min(int(args.text_preflight_records), len(dataset))
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


def save_checkpoint(
    path: Path,
    *,
    compressor: nn.Module,
    adapter: TensorPatchAlignmentAdapter,
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


def fmt_metric(metrics: Mapping[str, Any], key: str) -> str:
    value = metrics.get(key)
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return "n/a"


def alignment_metric_summary(metrics: Mapping[str, Any]) -> str:
    return (
        f"main_i2t={fmt_metric(metrics, 'i2t_accuracy')} "
        f"main_t2i={fmt_metric(metrics, 't2i_accuracy')} "
        f"centered_i2t={fmt_metric(metrics, 'centered_i2t_accuracy')} "
        f"centered_t2i={fmt_metric(metrics, 'centered_t2i_accuracy')} "
        f"uncentered_i2t={fmt_metric(metrics, 'uncentered_i2t_accuracy')} "
        f"uncentered_t2i={fmt_metric(metrics, 'uncentered_t2i_accuracy')} "
        f"global_i2t={fmt_metric(metrics, 'global_i2t_accuracy')} "
        f"global_t2i={fmt_metric(metrics, 'global_t2i_accuracy')} "
        f"global_centered_i2t={fmt_metric(metrics, 'global_centered_i2t_accuracy')} "
        f"global_centered_t2i={fmt_metric(metrics, 'global_centered_t2i_accuracy')} "
        f"global_uncentered_i2t={fmt_metric(metrics, 'global_uncentered_i2t_accuracy')} "
        f"global_uncentered_t2i={fmt_metric(metrics, 'global_uncentered_t2i_accuracy')}"
    )


def build_wandb_config(args: argparse.Namespace, summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "experiment": {"name": str(args.run_name)},
        "data": {
            "hdf5_path": str(args.hdf5_path),
            "fields": parse_csv(args.fields),
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
            "dropout": float(args.dropout),
            "soft_prompt_scale": float(args.soft_prompt_scale),
            "temperature": float(args.temperature),
            "contrastive_loss_weight": float(args.contrastive_loss_weight),
            "centered_contrastive_loss_weight": float(args.centered_contrastive_loss_weight),
            "cosine_loss_weight": float(args.cosine_loss_weight),
            "reconstruction_loss_weight": float(args.reconstruction_loss_weight),
            "teacher_text_source": str(args.teacher_text_source),
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
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if args.hf_home:
        __import__("os").environ["HF_HOME"] = str(args.hf_home)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / f"{timestamp}_{args.run_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    dump_json(run_dir / "args.json", redacted_args(args))

    checkpoint: dict[str, Any] | None = None
    if str(args.encoder_source) == "checkpoint":
        checkpoint, compressor_config = load_checkpoint_and_config(args.compressor_checkpoint, args.compressor_config)
        state_dict = resolve_checkpoint_state_dict(checkpoint, args.compressor_checkpoint)
        field_keys = resolve_field_keys(args.fields, compressor_config)
        compressor = build_model(compressor_config)
        compressor.load_state_dict(state_dict)
    else:
        field_keys = resolve_field_keys(args.fields, None)
        compressor_config = build_patch_encoder_config(
            patch_encoder_cfg=args.patch_encoder_config,
            field_keys=field_keys,
            patch_size=int(args.patch_size),
        )
        compressor = build_model(compressor_config)
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
        "include_raw_text": str(args.teacher_text_source).lower() == "raw",
    }
    train_dataset = PDEBenchPatchTextDataset(records=train_records, **dataset_kwargs)
    val_dataset = PDEBenchPatchTextDataset(records=val_records, **dataset_kwargs)
    test_dataset = PDEBenchPatchTextDataset(records=test_records, **dataset_kwargs)
    train_loader = make_loader(train_dataset, int(args.batch_size), True, int(args.num_workers))
    val_loader = make_loader(val_dataset, int(args.eval_batch_size), False, int(args.num_workers))
    test_loader = make_loader(test_dataset, int(args.eval_batch_size), False, int(args.num_workers))
    text_preflight_metrics = preflight_teacher_text_tokenization(
        dataset=train_dataset,
        tokenizer=tokenizer,
        args=args,
        compressor_input_size=compressor_input_size,
        normalization_cfg=normalization_cfg,
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

    args.alignment_train_patch_ae = bool(args.train_patch_ae) and not (
        bool(args.freeze_patch_ae_after_pretrain) and int(args.patch_ae_pretrain_epochs) > 0
    )

    run_summary = {
        "hdf5_path": str(args.hdf5_path),
        "field_keys": field_keys,
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
        "teacher_layer": int(args.teacher_layer),
        "teacher_text_source": str(args.teacher_text_source),
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
        "adapter_type": str(args.adapter_type),
        "query_tokens": int(args.query_tokens),
        "adapter_layers": int(args.adapter_layers),
        "adapter_heads": int(args.adapter_heads),
        "soft_prompt_scale": float(args.soft_prompt_scale),
        "alignment_mode": "input_soft_prompt_hidden",
        "adapter_parameters": sum(parameter.numel() for parameter in adapter.parameters() if parameter.requires_grad),
        "pretrain_trainable_compressor_parameters": sum(
            parameter.numel() for parameter in compressor.parameters() if parameter.requires_grad
        ),
        "alignment_trainable_compressor_parameters": (
            sum(parameter.numel() for parameter in compressor.parameters()) if bool(args.alignment_train_patch_ae) else 0
        ),
    }
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
        f"center_embeddings={bool(args.center_embeddings)} "
        f"contrastive_weight={float(args.contrastive_loss_weight):.4g} "
        f"centered_contrastive_weight={float(args.centered_contrastive_loss_weight):.4g} "
        f"cosine_weight={float(args.cosine_loss_weight):.4g} "
        f"global_eval={bool(args.global_retrieval_eval)}"
    )
    if text_preflight_metrics:
        print(
            "teacher_text_preflight "
            f"token_mean={text_preflight_metrics.get('token_count_mean', 0.0):.1f} "
            f"token_max={text_preflight_metrics.get('token_count_max', 0.0):.0f} "
            f"anchor_missing={text_preflight_metrics.get('anchor_missing_fraction', 0.0):.3f} "
            f"max_len_hit={text_preflight_metrics.get('max_length_hit_fraction', 0.0):.3f}"
        )
    wandb_logger = WandbLogger(config=build_wandb_config(args, run_summary), run_dir=run_dir)
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
            )
            pretrain_epoch_metrics = {
                "train": pretrain_metrics,
                "val": pretrain_val_metrics,
            }
            metrics_history[f"patch_ae_pretrain_{pretrain_epoch:04d}"] = pretrain_epoch_metrics
            dump_json(run_dir / "metrics_latest.json", metrics_history)
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

    if bool(args.alignment_train_patch_ae):
        for parameter in compressor.parameters():
            parameter.requires_grad_(True)
    else:
        for parameter in compressor.parameters():
            parameter.requires_grad_(False)
        compressor.eval()
    optimizer_params = list(adapter.parameters())
    if bool(args.alignment_train_patch_ae):
        optimizer_params += [parameter for parameter in compressor.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(optimizer_params, lr=float(args.lr), weight_decay=float(args.weight_decay))

    best_val = float("inf")
    best_epoch = 0
    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = train_one_epoch(
            compressor=compressor,
            adapter=adapter,
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
        val_metrics = evaluate(
            compressor=compressor,
            adapter=adapter,
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
        global_step += len(train_loader)
        save_checkpoint(
            run_dir / "alignment_last.pt",
            compressor=compressor,
            adapter=adapter,
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
        wandb_payload.update(numeric_payload("train", train_metrics))
        wandb_payload.update(numeric_payload("val", val_metrics))
        wandb_logger.log(wandb_payload, step=global_step)
        print(
            f"epoch={epoch:04d} train_loss={train_metrics['loss']:.4f} "
            f"train[{alignment_metric_summary(train_metrics)}] "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val[{alignment_metric_summary(val_metrics)}]"
        )

    best_checkpoint = torch.load(run_dir / "alignment_best.pt", map_location=device)
    adapter.load_state_dict(best_checkpoint["adapter_state_dict"])
    if bool(args.train_patch_ae) and "compressor_state_dict" in best_checkpoint:
        compressor.load_state_dict(best_checkpoint["compressor_state_dict"])
    test_metrics = evaluate(
        compressor=compressor,
        adapter=adapter,
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
    wandb_logger.log(numeric_payload("test", test_metrics), step=global_step)
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
    print(json.dumps({"best_epoch": best_epoch, "test": test_metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
