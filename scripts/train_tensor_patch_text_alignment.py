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


def build_patch_records(
    *,
    hdf5_path: str | Path,
    field: str,
    sample_indices: str | Sequence[int],
    time_indices: str | Sequence[int],
    patch_size: int,
    count: int,
    seed: int,
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
    max_row = height - int(patch_size)
    max_col = width - int(patch_size)
    for _ in range(int(count)):
        records.append(
            PatchRecord(
                sample_index=int(rng.choice(samples)),
                time_index=int(rng.choice(times)),
                row=int(rng.randint(0, max_row)),
                col=int(rng.randint(0, max_col)),
            )
        )
    return records


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
    ) -> None:
        self.hdf5_path = Path(hdf5_path).expanduser()
        self.field_keys = [str(field) for field in field_keys]
        self.records = list(records)
        self.patch_size = int(patch_size)
        self.decimal_places = int(decimal_places)
        self.prompt_template = str(prompt_template)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[int(index)]
        patch = self._read_patch(record)
        text = self._serialize_patch(record, patch)
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
        decimals = max(0, int(self.decimal_places))
        field_chunks: list[str] = []
        for channel, field in enumerate(self.field_keys):
            rows: list[str] = []
            values = patch[channel]
            for row in range(values.shape[0]):
                row_values = ", ".join(f"{float(value):.{decimals}f}" for value in values[row])
                rows.append(f"[{row_values}]")
            field_chunks.append(f"{field}=[{'; '.join(rows)}]")
        body = "\n".join(field_chunks)
        if self.prompt_template == "compact":
            return (
                "Represent this PDE tensor patch for numeric reasoning.\n"
                f"fields={','.join(self.field_keys)} patch_size={self.patch_size}\n"
                f"{body}\nRepresentation:"
            )
        if self.prompt_template == "compact_with_metadata":
            return (
                "Represent this PDE tensor patch for numeric reasoning.\n"
                f"sample={record.sample_index} time={record.time_index} "
                f"top_left=({record.row},{record.col}) patch_size={self.patch_size}\n"
                f"{body}\nRepresentation:"
            )
        if self.prompt_template == "plain":
            return body
        raise ValueError(f"Unsupported text_prompt_template: {self.prompt_template}")


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
    ) -> None:
        super().__init__()
        self.adapter_type = str(adapter_type).lower()
        self.latent_grid = tuple(int(dim) for dim in latent_grid)
        self.latent_token_count = int(self.latent_grid[0] * self.latent_grid[1])
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
            return self.output(queries)
        pooled = torch.cat(
            [
                latent_map.mean(dim=tuple(range(2, latent_map.ndim))),
                latent_map.std(dim=tuple(range(2, latent_map.ndim)), unbiased=False),
            ],
            dim=-1,
        )
        return self.projection(pooled).unsqueeze(1)

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


def build_student_anchor_texts(records: Sequence[Mapping[str, Any]]) -> list[str]:
    texts: list[str] = []
    for record in records:
        fields = record.get("fields", [])
        if isinstance(fields, Sequence) and not isinstance(fields, str):
            field_text = ",".join(str(field) for field in fields)
        else:
            field_text = str(fields)
        patch_size = int(record.get("patch_size", 0))
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
) -> torch.Tensor:
    encoded = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=int(max_tokens),
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    outputs = llm(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    return hidden_at_last_non_padding(outputs.hidden_states, attention_mask, teacher_layer).detach()


def tensor_student_hidden(
    llm: nn.Module,
    tokenizer: Any,
    soft_prompts: torch.Tensor,
    records: Sequence[Mapping[str, Any]],
    device: torch.device,
    max_tokens: int,
    teacher_layer: int,
) -> torch.Tensor:
    encoded = tokenizer(
        build_student_anchor_texts(records),
        padding=True,
        truncation=True,
        max_length=int(max_tokens),
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    text_attention_mask = encoded["attention_mask"].to(device)
    text_embeds = llm.get_input_embeddings()(input_ids)
    soft_prompts = soft_prompts.to(device=device, dtype=text_embeds.dtype)
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
    return hidden_at_last_non_padding(
        outputs.hidden_states,
        text_attention_mask,
        teacher_layer,
        prefix_tokens=int(soft_prompts.shape[1]),
    )


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
    progress = tqdm(loader, desc=f"train align epoch {epoch}", leave=False)
    for step, batch in enumerate(progress, start=1):
        patches = normalize_patch_batch(
            batch["patch"],
            compressor_input_size,
            normalization_cfg,
            bool(args.resize_patch_to_compressor_input),
        ).to(device)
        texts = batch["texts"]
        with torch.no_grad():
            teacher_hidden = text_teacher_hidden(
                llm,
                tokenizer,
                texts,
                device,
                int(args.max_text_tokens),
                int(args.teacher_layer),
            )
        if train_compressor:
            latent = compressor.encode(patches)["latent_map"]
        else:
            with torch.no_grad():
                latent = compressor.encode(patches)["latent_map"]
        soft_prompts = adapter.forward_soft_prompts(latent)
        student_hidden = tensor_student_hidden(
            llm,
            tokenizer,
            soft_prompts,
            batch["records"],
            device,
            int(args.max_text_tokens),
            int(args.teacher_layer),
        )
        tensor_embedding = F.normalize(student_hidden.float(), dim=-1)
        text_embedding = F.normalize(teacher_hidden.to(dtype=tensor_embedding.dtype), dim=-1)
        contrastive, contrastive_metrics = symmetric_contrastive_loss(
            tensor_embedding,
            text_embedding,
            float(args.temperature),
        )
        cosine = cosine_alignment_loss(tensor_embedding, text_embedding)
        reconstruction = reconstruction_mse(compressor, latent, patches)
        reconstruction_weight = float(args.reconstruction_loss_weight) if train_compressor else 0.0
        loss = (
            float(args.contrastive_loss_weight) * contrastive
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
        total_records += batch_size
        if int(args.log_interval) > 0 and step % int(args.log_interval) == 0:
            progress.set_postfix(
                loss=f"{total_loss / max(1, total_records):.4f}",
                i2t=f"{total_i2t / max(1, total_records):.3f}",
                t2i=f"{total_t2i / max(1, total_records):.3f}",
            )
    return {
        "loss": total_loss / max(1, total_records),
        "contrastive_loss": total_contrastive / max(1, total_records),
        "cosine_loss": total_cosine / max(1, total_records),
        "reconstruction_loss": total_reconstruction / max(1, total_records),
        "i2t_accuracy": total_i2t / max(1, total_records),
        "t2i_accuracy": total_t2i / max(1, total_records),
    }


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
    train_compressor = train_compressor_during_alignment(args)
    for batch in tqdm(loader, desc="eval align", leave=False):
        patches = normalize_patch_batch(
            batch["patch"],
            compressor_input_size,
            normalization_cfg,
            bool(args.resize_patch_to_compressor_input),
        ).to(device)
        teacher_hidden = text_teacher_hidden(
            llm,
            tokenizer,
            batch["texts"],
            device,
            int(args.max_text_tokens),
            int(args.teacher_layer),
        )
        latent = compressor.encode(patches)["latent_map"]
        soft_prompts = adapter.forward_soft_prompts(latent)
        student_hidden = tensor_student_hidden(
            llm,
            tokenizer,
            soft_prompts,
            batch["records"],
            device,
            int(args.max_text_tokens),
            int(args.teacher_layer),
        )
        tensor_embedding = F.normalize(student_hidden.float(), dim=-1)
        text_embedding = F.normalize(teacher_hidden.to(dtype=tensor_embedding.dtype), dim=-1)
        contrastive, contrastive_metrics = symmetric_contrastive_loss(
            tensor_embedding,
            text_embedding,
            float(args.temperature),
        )
        cosine = cosine_alignment_loss(tensor_embedding, text_embedding)
        reconstruction = reconstruction_mse(compressor, latent, patches)
        reconstruction_weight = float(args.reconstruction_loss_weight) if train_compressor else 0.0
        loss = (
            float(args.contrastive_loss_weight) * contrastive
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
        total_records += batch_size
    return {
        "loss": total_loss / max(1, total_records),
        "contrastive_loss": total_contrastive / max(1, total_records),
        "cosine_loss": total_cosine / max(1, total_records),
        "reconstruction_loss": total_reconstruction / max(1, total_records),
        "i2t_accuracy": total_i2t / max(1, total_records),
        "t2i_accuracy": total_t2i / max(1, total_records),
    }


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
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--contrastive-loss-weight", type=float, default=None)
    parser.add_argument("--cosine-loss-weight", type=float, default=None)
    parser.add_argument("--reconstruction-loss-weight", type=float, default=None)
    parser.add_argument(
        "--text-prompt-template",
        type=str,
        choices=("compact", "compact_with_metadata", "plain"),
        default=None,
    )
    parser.add_argument("--text-decimal-places", type=int, default=None)
    parser.add_argument("--max-text-tokens", type=int, default=None)
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
    set_default(args, "temperature", first_nested(config, ["patch_alignment.temperature"]), 0.07)
    set_default(args, "contrastive_loss_weight", first_nested(config, ["patch_alignment.contrastive_loss_weight"]), 1.0)
    set_default(args, "cosine_loss_weight", first_nested(config, ["patch_alignment.cosine_loss_weight"]), 0.2)
    set_default(args, "reconstruction_loss_weight", first_nested(config, ["patch_alignment.reconstruction_loss_weight"]), 1.0)
    set_default(args, "text_prompt_template", first_nested(config, ["patch_alignment.text_prompt_template"]), "compact")
    set_default(args, "text_decimal_places", first_nested(config, ["patch_alignment.text_decimal_places"]), 3)
    set_default(args, "max_text_tokens", first_nested(config, ["patch_alignment.max_text_tokens"]), 1024)
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
            "temperature": float(args.temperature),
            "contrastive_loss_weight": float(args.contrastive_loss_weight),
            "cosine_loss_weight": float(args.cosine_loss_weight),
            "reconstruction_loss_weight": float(args.reconstruction_loss_weight),
            "text_prompt_template": str(args.text_prompt_template),
            "text_decimal_places": int(args.text_decimal_places),
            "max_text_tokens": int(args.max_text_tokens),
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
    train_records = build_patch_records(
        hdf5_path=args.hdf5_path,
        field=first_field,
        sample_indices=str(args.sample_indices),
        time_indices=str(args.time_indices),
        patch_size=int(args.patch_size),
        count=int(args.train_records),
        seed=int(args.seed),
    )
    val_records = build_patch_records(
        hdf5_path=args.hdf5_path,
        field=first_field,
        sample_indices=str(args.sample_indices),
        time_indices=str(args.time_indices),
        patch_size=int(args.patch_size),
        count=int(args.val_records),
        seed=int(args.seed) + 1,
    )
    test_records = build_patch_records(
        hdf5_path=args.hdf5_path,
        field=first_field,
        sample_indices=str(args.sample_indices),
        time_indices=str(args.time_indices),
        patch_size=int(args.patch_size),
        count=int(args.test_records),
        seed=int(args.seed) + 2,
    )
    dataset_kwargs = {
        "hdf5_path": args.hdf5_path,
        "field_keys": field_keys,
        "patch_size": int(args.patch_size),
        "decimal_places": int(args.text_decimal_places),
        "prompt_template": str(args.text_prompt_template),
    }
    train_dataset = PDEBenchPatchTextDataset(records=train_records, **dataset_kwargs)
    val_dataset = PDEBenchPatchTextDataset(records=val_records, **dataset_kwargs)
    test_dataset = PDEBenchPatchTextDataset(records=test_records, **dataset_kwargs)
    train_loader = make_loader(train_dataset, int(args.batch_size), True, int(args.num_workers))
    val_loader = make_loader(val_dataset, int(args.eval_batch_size), False, int(args.num_workers))
    test_loader = make_loader(test_dataset, int(args.eval_batch_size), False, int(args.num_workers))

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
        "adapter_type": str(args.adapter_type),
        "query_tokens": int(args.query_tokens),
        "adapter_layers": int(args.adapter_layers),
        "adapter_heads": int(args.adapter_heads),
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
            f"train_i2t={train_metrics['i2t_accuracy']:.4f} "
            f"train_t2i={train_metrics['t2i_accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_i2t={val_metrics['i2t_accuracy']:.4f} "
            f"val_t2i={val_metrics['t2i_accuracy']:.4f}"
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
