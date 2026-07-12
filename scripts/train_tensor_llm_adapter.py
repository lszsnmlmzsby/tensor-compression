from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.train_tensor_patch_text_alignment import TensorPatchAlignmentAdapter  # noqa: E402

from tensor_compression.downstream.pdebench import resolve_device  # noqa: E402
from tensor_compression.integrations import WandbLogger  # noqa: E402
from tensor_compression.utils import dump_json  # noqa: E402
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
except ImportError as exc:  # pragma: no cover - exercised only in missing-dependency envs
    raise ImportError(
        "scripts/train_tensor_llm_adapter.py requires transformers. "
        "Install it with: pip install transformers accelerate safetensors"
    ) from exc


IGNORE_INDEX = -100
STRUCTURED_QUERY_FEATURE_DIM = 32


def _normalize_coordinate(value: Any, size: int) -> float:
    if size <= 1:
        return 0.0
    clipped = max(0.0, min(float(value), float(size - 1)))
    return (clipped / float(size - 1)) * 2.0 - 1.0


def _normalize_length(value: Any, size: int) -> float:
    if size <= 0:
        return 0.0
    clipped = max(0.0, min(float(value), float(size)))
    return clipped / float(size)


def structured_query_features_for_record(record: Mapping[str, Any]) -> list[float]:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), Mapping) else {}
    grid_shape = metadata.get("grid_shape") if isinstance(metadata, Mapping) else None
    has_grid_shape = isinstance(grid_shape, Sequence) and not isinstance(grid_shape, str)
    height = int(grid_shape[0]) if has_grid_shape and len(grid_shape) >= 1 else 512
    width = int(grid_shape[1]) if has_grid_shape and len(grid_shape) >= 2 else 512
    task_type = str(record.get("task_type", ""))
    query = str(record.get("query") or record.get("question") or "")
    choices = record.get("choices")
    choice_count = len(choices) if isinstance(choices, Sequence) and not isinstance(choices, str) else 0

    features = [0.0] * STRUCTURED_QUERY_FEATURE_DIM
    task_order = [
        "normalized_point_value",
        "raw_point_value_with_stats",
        "point_compare",
        "region_mean_compare",
        "extreme_quadrant",
        "point_bin",
        "patch_compare",
        "max_speed_quadrant",
        "global_stat_bin",
    ]
    if task_type in task_order:
        features[task_order.index(task_type)] = 1.0
    field = str(record.get("field") or metadata.get("field") or "").lower()
    for offset, field_name in enumerate(("density", "pressure", "vx", "vy"), start=9):
        features[offset] = float(field == field_name)
    features[13] = _normalize_length(choice_count, 16)
    features[14] = 1.0 if "maximum" in query.lower() else 0.0
    features[15] = 1.0 if "minimum" in query.lower() else 0.0

    point = re.search(r"row(?:=|\s+)(\d+)[,\s]+col(?:umn)?(?:=|\s+)(\d+)", query, re.IGNORECASE)
    if point:
        row = int(point.group(1))
        col = int(point.group(2))
        features[16] = _normalize_coordinate(row, height)
        features[17] = _normalize_coordinate(col, width)

    point_pair = re.search(
        r"A at row (\d+), column (\d+).*?B at row (\d+), column (\d+)",
        query,
        re.IGNORECASE,
    ) or re.search(r"A=\((\d+),(\d+)\)\s+B=\((\d+),(\d+)\)", query)
    if point_pair:
        row_a, col_a, row_b, col_b = [int(group) for group in point_pair.groups()]
        features[16] = _normalize_coordinate(row_a, height)
        features[17] = _normalize_coordinate(col_a, width)
        features[18] = _normalize_coordinate(row_b, height)
        features[19] = _normalize_coordinate(col_b, width)
        features[20] = (float(row_b) - float(row_a)) / max(1.0, float(height - 1))
        features[21] = (float(col_b) - float(col_a)) / max(1.0, float(width - 1))

    region_pair = re.search(
        r"Region A starts at row (\d+), column (\d+); region B starts at row (\d+), column (\d+)",
        query,
        re.IGNORECASE,
    )
    patch_pair = re.search(
        r"A=\[(\d+):(\d+),(\d+):(\d+)\]\s+B=\[(\d+):(\d+),(\d+):(\d+)\]",
        query,
    )
    if region_pair:
        row_a, col_a, row_b, col_b = [int(group) for group in region_pair.groups()]
        size_match = re.search(r"two (\d+) by (\d+) regions", query, re.IGNORECASE)
        region_h = int(size_match.group(1)) if size_match else 1
        region_w = int(size_match.group(2)) if size_match else region_h
        features[22] = _normalize_coordinate(row_a, height)
        features[23] = _normalize_coordinate(col_a, width)
        features[24] = _normalize_coordinate(row_b, height)
        features[25] = _normalize_coordinate(col_b, width)
        features[26] = _normalize_length(region_h, height)
        features[27] = _normalize_length(region_w, width)
    elif patch_pair:
        row0_a, row1_a, col0_a, col1_a, row0_b, row1_b, col0_b, col1_b = [
            int(group) for group in patch_pair.groups()
        ]
        center_row_a = (row0_a + row1_a - 1) / 2.0
        center_col_a = (col0_a + col1_a - 1) / 2.0
        center_row_b = (row0_b + row1_b - 1) / 2.0
        center_col_b = (col0_b + col1_b - 1) / 2.0
        features[22] = _normalize_coordinate(center_row_a, height)
        features[23] = _normalize_coordinate(center_col_a, width)
        features[24] = _normalize_coordinate(center_row_b, height)
        features[25] = _normalize_coordinate(center_col_b, width)
        features[26] = _normalize_length(row1_a - row0_a, height)
        features[27] = _normalize_length(col1_a - col0_a, width)
    prompt_data = record.get("prompt_data")
    if isinstance(prompt_data, Mapping):
        features[28] = math.tanh(float(prompt_data.get("mean", 0.0)))
        features[29] = math.tanh(math.log1p(abs(float(prompt_data.get("std", 0.0)))))
    return features


def structured_query_features(records: Sequence[Mapping[str, Any]], device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [structured_query_features_for_record(record) for record in records],
        dtype=torch.float32,
        device=device,
    )


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"adapter_dim={dim} must be divisible by adapter_heads={heads}.")
        self.query_norm = nn.LayerNorm(dim)
        self.latent_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        queries: torch.Tensor,
        latents: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attended, _weights = self.attention(
            query=self.query_norm(queries),
            key=self.latent_norm(latents),
            value=self.latent_norm(latents),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        queries = queries + attended
        return queries + self.ffn(self.ffn_norm(queries))


class TensorSoftPromptAdapter(nn.Module):
    """A small Perceiver/Q-Former style adapter from tensor latents to LLM soft prompts."""

    def __init__(
        self,
        latent_channels: int,
        llm_hidden_size: int,
        soft_prompt_tokens: int,
        adapter_dim: int,
        adapter_layers: int,
        adapter_heads: int,
        dropout: float,
        latent_pos_encoding: str,
        question_conditioning: bool,
        question_condition_gate_init: float,
        structured_query_conditioning: bool,
        soft_prompt_scale: float,
    ) -> None:
        super().__init__()
        self.soft_prompt_tokens = int(soft_prompt_tokens)
        self.latent_pos_encoding = str(latent_pos_encoding)
        self.question_conditioning = bool(question_conditioning)
        self.structured_query_conditioning = bool(structured_query_conditioning)
        self.soft_prompt_scale = float(soft_prompt_scale)
        self.input_projection = nn.Linear(int(latent_channels), int(adapter_dim))
        if self.latent_pos_encoding == "grid":
            self.position_projection = nn.Linear(2, int(adapter_dim))
        elif self.latent_pos_encoding == "none":
            self.position_projection = None
        else:
            raise ValueError(f"Unsupported latent_pos_encoding: {latent_pos_encoding}")
        if self.question_conditioning:
            self.question_projection = nn.Sequential(
                nn.LayerNorm(int(llm_hidden_size)),
                nn.Linear(int(llm_hidden_size), int(adapter_dim)),
                nn.GELU(),
                nn.Linear(int(adapter_dim), int(adapter_dim)),
            )
            self.question_gate = nn.Parameter(torch.tensor(float(question_condition_gate_init)))
        else:
            self.question_projection = None
            self.register_parameter("question_gate", None)
        if self.structured_query_conditioning:
            self.structured_query_projection = nn.Sequential(
                nn.LayerNorm(STRUCTURED_QUERY_FEATURE_DIM),
                nn.Linear(STRUCTURED_QUERY_FEATURE_DIM, int(adapter_dim)),
                nn.GELU(),
                nn.Linear(int(adapter_dim), int(adapter_dim)),
            )
            self.structured_query_gate = nn.Parameter(torch.tensor(1.0))
        else:
            self.structured_query_projection = None
            self.register_parameter("structured_query_gate", None)
        self.query_tokens = nn.Parameter(torch.randn(1, self.soft_prompt_tokens, int(adapter_dim)) * 0.02)
        self.blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    dim=int(adapter_dim),
                    heads=int(adapter_heads),
                    dropout=float(dropout),
                )
                for _ in range(int(adapter_layers))
            ]
        )
        self.output_norm = nn.LayerNorm(int(adapter_dim))
        self.output_projection = nn.Linear(int(adapter_dim), int(llm_hidden_size))

    def _grid_position_tokens(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        rows = torch.linspace(-1.0, 1.0, int(height), device=device, dtype=dtype)
        cols = torch.linspace(-1.0, 1.0, int(width), device=device, dtype=dtype)
        yy, xx = torch.meshgrid(rows, cols, indexing="ij")
        coords = torch.stack([yy, xx], dim=-1).reshape(1, int(height) * int(width), 2)
        coords = coords.expand(int(batch_size), -1, -1)
        return self.position_projection(coords)

    def _question_condition(
        self,
        question_embeds: torch.Tensor | None,
        question_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if not self.question_conditioning or question_embeds is None or self.question_projection is None:
            return None
        if question_mask is None:
            question_mask = torch.ones(
                question_embeds.shape[:2],
                dtype=torch.bool,
                device=question_embeds.device,
            )
        mask = question_mask.to(device=question_embeds.device, dtype=question_embeds.dtype).unsqueeze(-1)
        pooled = (question_embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        # One-way valve: text can condition query slots, but gradients and updates do not flow into text embeddings.
        pooled = pooled.detach().to(dtype=self.input_projection.weight.dtype)
        return self.question_projection(pooled)

    def forward(
        self,
        latent_map: torch.Tensor,
        question_embeds: torch.Tensor | None = None,
        question_mask: torch.Tensor | None = None,
        structured_query: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if latent_map.ndim == 4:
            batch_size, _channels, height, width = latent_map.shape
            latent_tokens = latent_map.flatten(2).transpose(1, 2).contiguous()
        elif latent_map.ndim == 3:
            batch_size = latent_map.shape[0]
            height = width = None
            latent_tokens = latent_map
        else:
            raise ValueError(f"Expected latent_map [B,C,H,W] or latent_tokens [B,N,C], got {latent_map.shape}.")
        latent_tokens = latent_tokens.to(dtype=self.input_projection.weight.dtype)
        latents = self.input_projection(latent_tokens)
        if self.position_projection is not None and height is not None and width is not None:
            latents = latents + self._grid_position_tokens(
                batch_size=int(batch_size),
                height=int(height),
                width=int(width),
                device=latents.device,
                dtype=latents.dtype,
            )
        queries = self.query_tokens.expand(int(batch_size), -1, -1)
        question_condition = self._question_condition(question_embeds, question_mask)
        if question_condition is not None:
            queries = queries + self.question_gate.to(dtype=queries.dtype) * question_condition.unsqueeze(1)
        if (
            self.structured_query_conditioning
            and structured_query is not None
            and self.structured_query_projection is not None
        ):
            structured_condition = self.structured_query_projection(
                structured_query.to(device=queries.device, dtype=self.input_projection.weight.dtype)
            )
            queries = queries + self.structured_query_gate.to(dtype=queries.dtype) * structured_condition.unsqueeze(1)
        for block in self.blocks:
            queries = block(queries, latents)
        soft_prompt = self.output_projection(self.output_norm(queries))
        if self.soft_prompt_scale > 0.0:
            soft_prompt = torch.tanh(soft_prompt) * self.soft_prompt_scale
        return soft_prompt


class QuestionConditionedLocalAdapter(nn.Module):
    """Extract local latent evidence using the question and explicit non-answer query metadata."""

    def __init__(
        self,
        latent_channels: int,
        latent_grid: Sequence[int],
        llm_hidden_size: int,
        adapter_dim: int,
        local_tokens: int,
        local_layers: int,
        adapter_heads: int,
        dropout: float,
        soft_prompt_scale: float,
        gate_init: float,
        max_text_tokens: int,
    ) -> None:
        super().__init__()
        if int(adapter_dim) % int(adapter_heads) != 0:
            raise ValueError("adapter_dim must be divisible by adapter_heads for the local adapter.")
        self.soft_prompt_tokens = int(local_tokens)
        self.latent_grid = tuple(int(dim) for dim in latent_grid)
        self.soft_prompt_scale = float(soft_prompt_scale)
        self.latent_projection = nn.Linear(int(latent_channels), int(adapter_dim))
        self.position_projection = nn.Linear(2, int(adapter_dim))
        self.text_projection = nn.Sequential(
            nn.LayerNorm(int(llm_hidden_size)),
            nn.Linear(int(llm_hidden_size), int(adapter_dim)),
        )
        self.text_pos_embed = nn.Parameter(torch.zeros(1, int(max_text_tokens), int(adapter_dim)))
        self.structured_projection = nn.Sequential(
            nn.LayerNorm(STRUCTURED_QUERY_FEATURE_DIM),
            nn.Linear(STRUCTURED_QUERY_FEATURE_DIM, int(adapter_dim)),
            nn.GELU(),
            nn.Linear(int(adapter_dim), int(adapter_dim)),
        )
        self.query_tokens = nn.Parameter(torch.empty(1, int(local_tokens), int(adapter_dim)))
        self.text_blocks = nn.ModuleList(
            [CrossAttentionBlock(int(adapter_dim), int(adapter_heads), float(dropout)) for _ in range(int(local_layers))]
        )
        self.latent_blocks = nn.ModuleList(
            [CrossAttentionBlock(int(adapter_dim), int(adapter_heads), float(dropout)) for _ in range(int(local_layers))]
        )
        self.output = nn.Sequential(
            nn.LayerNorm(int(adapter_dim)),
            nn.Linear(int(adapter_dim), int(llm_hidden_size)),
        )
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        nn.init.normal_(self.query_tokens, mean=0.0, std=0.02)
        nn.init.normal_(self.text_pos_embed, mean=0.0, std=0.02)

    def _position_tokens(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        height, width = self.latent_grid
        rows = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        cols = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(rows, cols, indexing="ij")
        coords = torch.stack([yy, xx], dim=-1).reshape(1, height * width, 2)
        return self.position_projection(coords.expand(int(batch_size), -1, -1))

    def forward(
        self,
        latent_map: torch.Tensor,
        question_embeds: torch.Tensor,
        question_mask: torch.Tensor | None,
        structured_query: torch.Tensor,
    ) -> torch.Tensor:
        latent_tokens = latent_map.flatten(2).transpose(1, 2).to(dtype=self.latent_projection.weight.dtype)
        if int(latent_tokens.shape[1]) != int(self.latent_grid[0] * self.latent_grid[1]):
            raise ValueError(f"Expected local latent grid {self.latent_grid}, got {tuple(latent_map.shape[-2:])}.")
        latents = self.latent_projection(latent_tokens)
        latents = latents + self._position_tokens(latents.shape[0], latents.device, latents.dtype)

        text_context = self.text_projection(question_embeds.detach().to(dtype=self.text_projection[1].weight.dtype))
        if int(text_context.shape[1]) > int(self.text_pos_embed.shape[1]):
            raise ValueError(
                f"Question length {int(text_context.shape[1])} exceeds local adapter limit "
                f"{int(self.text_pos_embed.shape[1])}."
            )
        text_context = text_context + self.text_pos_embed[:, : text_context.shape[1]]
        key_padding_mask = None
        if question_mask is not None:
            key_padding_mask = ~question_mask.to(device=text_context.device, dtype=torch.bool)
        queries = self.query_tokens.expand(latents.shape[0], -1, -1)
        query_condition = self.structured_projection(
            structured_query.to(device=queries.device, dtype=self.structured_projection[1].weight.dtype)
        )
        queries = queries + query_condition.unsqueeze(1)
        for text_block, latent_block in zip(self.text_blocks, self.latent_blocks):
            queries = text_block(queries, text_context, key_padding_mask=key_padding_mask)
            queries = latent_block(queries, latents)
        local_prompts = self.output(queries)
        if self.soft_prompt_scale > 0.0:
            local_prompts = torch.tanh(local_prompts) * self.soft_prompt_scale
        return self.gate.to(dtype=local_prompts.dtype) * local_prompts


class HybridGlobalLocalAdapter(nn.Module):
    def __init__(
        self,
        global_adapter: TensorPatchAlignmentAdapter,
        local_adapter: QuestionConditionedLocalAdapter,
        freeze_global: bool,
    ) -> None:
        super().__init__()
        self.global_adapter = global_adapter
        self.local_adapter = local_adapter
        self.freeze_global = bool(freeze_global)
        self.soft_prompt_tokens = int(global_adapter.soft_prompt_tokens + local_adapter.soft_prompt_tokens)
        self.structured_query_conditioning = True

        self.set_global_trainable(not self.freeze_global)

    def set_global_trainable(self, trainable: bool) -> None:
        self.freeze_global = not bool(trainable)
        for parameter in self.global_adapter.parameters():
            parameter.requires_grad_(bool(trainable))
        if self.freeze_global:
            self.global_adapter.eval()
        else:
            self.global_adapter.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_global:
            self.global_adapter.eval()
        return self

    def forward(
        self,
        latent_map: torch.Tensor,
        question_embeds: torch.Tensor | None = None,
        question_mask: torch.Tensor | None = None,
        structured_query: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if question_embeds is None or structured_query is None:
            raise ValueError("hybrid_local_qformer requires question embeddings and structured query features.")
        if self.freeze_global:
            with torch.no_grad():
                global_prompts = self.global_adapter.forward_soft_prompts(latent_map)
        else:
            global_prompts = self.global_adapter.forward_soft_prompts(latent_map)
        local_prompts = self.local_adapter(latent_map, question_embeds, question_mask, structured_query)
        # Keeping local tokens first preserves the relative positions between global tokens and text.
        return torch.cat([local_prompts, global_prompts], dim=1)


class TensorReadoutQADataset(Dataset):
    def __init__(
        self,
        jsonl_path: str | Path,
        latent_dir: str | Path,
        max_records: int | None = None,
        prefer_record_latent_ref: bool = False,
        shuffle_seed: int = 42,
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.latent_dir = Path(latent_dir)
        self.prefer_record_latent_ref = bool(prefer_record_latent_ref)
        self.records = self._load_records(self.jsonl_path)
        if max_records is not None:
            self.records = self.records[: max(0, int(max_records))]
        if not self.records:
            raise RuntimeError(f"No QA records found in {self.jsonl_path}.")
        self._random_different_indices = self._build_random_different_indices(int(shuffle_seed))

    @staticmethod
    def _load_records(path: Path) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                payload = json.loads(stripped)
                if not isinstance(payload, dict):
                    raise ValueError(f"Expected JSON object at {path}:{line_number}.")
                records.append(payload)
        return records

    def _build_random_different_indices(self, seed: int) -> list[int]:
        total = len(self.records)
        unique_states = {str(record.get("state_ref", "")) for record in self.records}
        if len(unique_states) < 2:
            raise RuntimeError(
                "Cannot build shuffled latent baseline: every record belongs to the same state_ref."
            )
        rng = random.Random(seed)
        indices_by_field_task: dict[tuple[str, str], list[int]] = defaultdict(list)
        indices_by_field: dict[str, list[int]] = defaultdict(list)
        for candidate_index, candidate_record in enumerate(self.records):
            field = str(
                candidate_record.get("field")
                or candidate_record.get("metadata", {}).get("field")
                or ""
            )
            task = str(candidate_record.get("task_type", ""))
            indices_by_field_task[(field, task)].append(candidate_index)
            indices_by_field[field].append(candidate_index)
        indices: list[int] = []
        for index, record in enumerate(self.records):
            state_ref = str(record.get("state_ref", ""))
            field = str(record.get("field") or record.get("metadata", {}).get("field") or "")
            task = str(record.get("task_type", ""))
            candidates = [
                candidate
                for candidate in indices_by_field_task.get((field, task), [])
                if str(self.records[candidate].get("state_ref", "")) != state_ref
            ]
            if not candidates:
                candidates = [
                    candidate
                    for candidate in indices_by_field.get(field, [])
                    if str(self.records[candidate].get("state_ref", "")) != state_ref
                ]
            if not candidates:
                candidates = [
                    candidate
                    for candidate in range(total)
                    if str(self.records[candidate].get("state_ref", "")) != state_ref
                ]
            if not candidates:
                raise RuntimeError("Failed to sample a different-state shuffled latent.")
            candidate = int(rng.choice(candidates))
            indices.append(int(candidate))
        return indices

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[int(index)]
        return {
            "index": int(index),
            "record": record,
            "latent_map": self.load_latent_for_record(record),
        }

    def latent_path_for_record(self, record: Mapping[str, Any]) -> Path:
        state_ref = str(record.get("state_ref") or "")
        if not state_ref:
            sample_index = int(record["sample_index"])
            time_index = int(record["time_index"])
            state_ref = f"sample{sample_index:06d}_t{time_index:04d}"
        latent_from_dir = self.latent_dir / f"{state_ref}.pt"
        record_ref = record.get("latent_ref")
        if self.prefer_record_latent_ref and record_ref:
            return Path(str(record_ref))
        if latent_from_dir.exists():
            return latent_from_dir
        if record_ref:
            return Path(str(record_ref))
        return latent_from_dir

    def load_latent_for_record(self, record: Mapping[str, Any]) -> torch.Tensor:
        path = self.latent_path_for_record(record)
        if not path.exists():
            raise FileNotFoundError(f"Latent cache file not found: {path}")
        payload = torch.load(path, map_location="cpu")
        latent = payload.get("latent_map") if isinstance(payload, Mapping) else payload
        if not isinstance(latent, torch.Tensor):
            raise ValueError(f"Latent cache file does not contain a tensor latent_map: {path}")
        if latent.ndim == 4 and latent.shape[0] == 1:
            latent = latent.squeeze(0)
        if latent.ndim != 3:
            raise ValueError(f"Expected latent_map [C,H,W], got {tuple(latent.shape)} from {path}")
        return latent.to(dtype=torch.float32)

    def load_shuffled_latent(self, index: int) -> torch.Tensor:
        other_index = self._random_different_indices[int(index)]
        return self.load_latent_for_record(self.records[other_index])

    def shuffled_record_for_index(self, index: int) -> Mapping[str, Any]:
        other_index = self._random_different_indices[int(index)]
        return self.records[other_index]


def collate_tensor_readout(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "indices": [int(item["index"]) for item in items],
        "records": [item["record"] for item in items],
        "latent_map": torch.stack([item["latent_map"] for item in items], dim=0),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a soft-prompt adapter from cached tensor latents into a frozen causal LM."
    )
    parser.add_argument("--config", type=str, default=None, help="Optional tensor-LLM pipeline YAML config.")
    parser.add_argument("--qa-dir", type=str, default=None)
    parser.add_argument("--latent-dir", type=str, default=None)
    parser.add_argument("--model-name-or-path", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--hf-home", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--train-split", type=str, default=None)
    parser.add_argument("--val-split", type=str, default=None)
    parser.add_argument("--test-split", type=str, default=None)
    parser.add_argument("--max-train-records", type=int, default=None)
    parser.add_argument("--max-val-records", type=int, default=None)
    parser.add_argument("--max-test-records", type=int, default=None)
    parser.add_argument("--initial-eval-records", type=int, default=None)
    parser.add_argument("--prefer-record-latent-ref", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default=None,
        choices=("auto", "float32", "float16", "bfloat16"),
        help="LLM parameter dtype. CPU always uses float32.",
    )
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--shuffle-seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--eval-choice-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    parser.add_argument("--ce-loss-weight", type=float, default=None)
    parser.add_argument("--choice-ce-loss-weight", type=float, default=None)
    parser.add_argument("--ranking-loss-weight", type=float, default=None)
    parser.add_argument("--ranking-loss-margin", type=float, default=None)
    parser.add_argument(
        "--ranking-loss-negative",
        type=str,
        default=None,
        choices=("shuffled", "random", "no_latent", "zero_latent"),
    )
    parser.add_argument("--soft-prompt-tokens", type=int, default=None)
    parser.add_argument(
        "--adapter-architecture",
        type=str,
        choices=("legacy", "alignment_qformer", "hybrid_local_qformer"),
        default=None,
    )
    parser.add_argument("--adapter-init-checkpoint", type=str, default=None)
    parser.add_argument("--adapter-dim", type=int, default=None)
    parser.add_argument("--adapter-layers", type=int, default=None)
    parser.add_argument("--adapter-heads", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument(
        "--latent-pos-encoding",
        type=str,
        default=None,
        choices=("none", "grid"),
        help="Positional encoding added to latent tokens before adapter cross-attention.",
    )
    parser.add_argument("--question-conditioning", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--question-condition-gate-init", type=float, default=None)
    parser.add_argument("--structured-query-conditioning", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--local-soft-prompt-tokens", type=int, default=None)
    parser.add_argument("--local-adapter-layers", type=int, default=None)
    parser.add_argument("--local-gate-init", type=float, default=None)
    parser.add_argument("--freeze-global-adapter", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--global-unfreeze-epoch", type=int, default=None)
    parser.add_argument("--global-lr", type=float, default=None)
    parser.add_argument("--soft-prompt-scale", type=float, default=None)
    parser.add_argument(
        "--prompt-template",
        type=str,
        default=None,
        choices=("generic", "task_specific"),
        help="Text prompt template used before the answer target.",
    )
    parser.add_argument("--max-prompt-tokens", type=int, default=None)
    parser.add_argument("--max-target-tokens", type=int, default=None)
    parser.add_argument("--append-eos", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--eval-baselines",
        type=str,
        default=None,
        help="Comma-separated: correct,no_latent,zero_latent,shuffled,random,shuffled_stats.",
    )
    parser.add_argument(
        "--choice-score",
        type=str,
        default=None,
        choices=("mean", "sum"),
        help="Normalize candidate NLL by target-token count or not.",
    )
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument(
        "--checkpoint-metric",
        type=str,
        default=None,
        choices=("correct_accuracy", "macro_latent_gain"),
    )
    parser.add_argument("--wandb-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--wandb-api-key", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument(
        "--wandb-tags",
        type=str,
        default=None,
        help="Comma-separated W&B tags.",
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default=None,
        choices=("online", "offline", "disabled"),
    )
    parser.add_argument("--wandb-log-model", action=argparse.BooleanOptionalAction, default=None)
    return apply_config_defaults(parser.parse_args())


def apply_config_defaults(args: argparse.Namespace) -> argparse.Namespace:
    config = load_yaml_mapping(args.config)

    model_local_dir = first_nested(config, ["model.local_dir"])
    model_name = first_nested(config, ["model.name_or_path", "model.model_name_or_path"])
    if args.model_name_or_path is None:
        args.model_name_or_path = (
            resolve_path_string(model_local_dir, PROJECT_ROOT)
            if model_local_dir
            else model_name
        )

    path_defaults = {
        "qa_dir": first_nested(config, ["patch_qa.qa_dir", "data.qa_dir"]),
        "latent_dir": first_nested(config, ["patch_qa.latent_dir", "data.latent_dir", "latent_export.output_dir"]),
        "adapter_init_checkpoint": first_nested(config, ["adapter.init_checkpoint", "patch_qa.alignment_checkpoint"]),
        "output_root": first_nested(config, ["llm_training.output_root", "storage.output_root"]),
        "cache_dir": first_nested(config, ["model.cache_dir", "storage.hf_home"]),
        "hf_home": first_nested(config, ["storage.hf_home"]),
    }
    for attr, value in path_defaults.items():
        if getattr(args, attr, None) is None and value is not None:
            setattr(args, attr, resolve_path_string(value, PROJECT_ROOT))

    set_default(args, "run_name", first_nested(config, ["llm_training.run_name"]), "tensor_llm_adapter")
    set_default(args, "train_split", first_nested(config, ["llm_training.train_split"]), "train")
    set_default(args, "val_split", first_nested(config, ["llm_training.val_split"]), "val")
    set_default(args, "test_split", first_nested(config, ["llm_training.test_split"]), "test")
    set_default(args, "max_train_records", first_nested(config, ["llm_training.max_train_records"]), None)
    set_default(args, "max_val_records", first_nested(config, ["llm_training.max_val_records"]), None)
    set_default(args, "max_test_records", first_nested(config, ["llm_training.max_test_records"]), None)
    set_default(args, "initial_eval_records", first_nested(config, ["llm_training.initial_eval_records"]), 512)
    set_default(
        args,
        "prefer_record_latent_ref",
        first_nested(config, ["llm_training.prefer_record_latent_ref"]),
        False,
    )
    set_default(args, "device", first_nested(config, ["llm_training.device", "runtime.device"]), "auto")
    set_default(args, "torch_dtype", first_nested(config, ["llm_training.torch_dtype", "model.torch_dtype"]), "auto")
    set_default(args, "trust_remote_code", first_nested(config, ["model.trust_remote_code"]), False)
    set_default(args, "seed", first_nested(config, ["llm_training.seed", "runtime.seed"]), 42)
    set_default(args, "shuffle_seed", first_nested(config, ["llm_training.shuffle_seed", "runtime.seed"]), 42)
    set_default(args, "epochs", first_nested(config, ["llm_training.epochs"]), 3)
    set_default(args, "batch_size", first_nested(config, ["llm_training.batch_size"]), 2)
    set_default(args, "eval_batch_size", first_nested(config, ["llm_training.eval_batch_size"]), 2)
    set_default(args, "eval_choice_batch_size", first_nested(config, ["llm_training.eval_choice_batch_size"]), 16)
    set_default(
        args,
        "gradient_accumulation_steps",
        first_nested(config, ["llm_training.gradient_accumulation_steps"]),
        1,
    )
    set_default(args, "lr", first_nested(config, ["llm_training.lr"]), 1.0e-4)
    set_default(args, "weight_decay", first_nested(config, ["llm_training.weight_decay"]), 1.0e-2)
    set_default(args, "grad_clip_norm", first_nested(config, ["llm_training.grad_clip_norm"]), 1.0)
    set_default(args, "ce_loss_weight", first_nested(config, ["llm_training.ce_loss_weight"]), 0.5)
    set_default(args, "choice_ce_loss_weight", first_nested(config, ["llm_training.choice_ce_loss_weight"]), 0.0)
    set_default(args, "ranking_loss_weight", first_nested(config, ["llm_training.ranking_loss_weight"]), 0.2)
    set_default(args, "ranking_loss_margin", first_nested(config, ["llm_training.ranking_loss_margin"]), 0.1)
    set_default(args, "ranking_loss_negative", first_nested(config, ["llm_training.ranking_loss_negative"]), "shuffled")
    set_default(args, "adapter_architecture", first_nested(config, ["adapter.architecture"]), "legacy")
    set_default(args, "soft_prompt_tokens", first_nested(config, ["adapter.soft_prompt_tokens"]), 32)
    set_default(args, "adapter_dim", first_nested(config, ["adapter.adapter_dim"]), 512)
    set_default(args, "adapter_layers", first_nested(config, ["adapter.adapter_layers"]), 2)
    set_default(args, "adapter_heads", first_nested(config, ["adapter.adapter_heads"]), 8)
    set_default(args, "dropout", first_nested(config, ["adapter.dropout"]), 0.1)
    set_default(args, "latent_pos_encoding", first_nested(config, ["adapter.latent_pos_encoding"]), "grid")
    set_default(args, "question_conditioning", first_nested(config, ["adapter.question_conditioning"]), True)
    set_default(
        args,
        "question_condition_gate_init",
        first_nested(config, ["adapter.question_condition_gate_init"]),
        1.0,
    )
    set_default(
        args,
        "structured_query_conditioning",
        first_nested(config, ["adapter.structured_query_conditioning"]),
        True,
    )
    set_default(args, "local_soft_prompt_tokens", first_nested(config, ["adapter.local_soft_prompt_tokens"]), 8)
    set_default(args, "local_adapter_layers", first_nested(config, ["adapter.local_adapter_layers"]), 2)
    set_default(args, "local_gate_init", first_nested(config, ["adapter.local_gate_init"]), 0.1)
    set_default(args, "freeze_global_adapter", first_nested(config, ["adapter.freeze_global_adapter"]), True)
    set_default(args, "global_unfreeze_epoch", first_nested(config, ["adapter.global_unfreeze_epoch"]), 0)
    set_default(args, "global_lr", first_nested(config, ["adapter.global_lr"]), 1.0e-5)
    set_default(args, "soft_prompt_scale", first_nested(config, ["adapter.soft_prompt_scale"]), 0.05)
    set_default(
        args,
        "prompt_template",
        first_nested(config, ["llm_training.prompt_template", "prompt.template"]),
        "task_specific",
    )
    set_default(args, "max_prompt_tokens", first_nested(config, ["llm_training.max_prompt_tokens"]), 256)
    set_default(args, "max_target_tokens", first_nested(config, ["llm_training.max_target_tokens"]), 8)
    set_default(args, "append_eos", first_nested(config, ["llm_training.append_eos"]), True)
    set_default(
        args,
        "eval_baselines",
        value_to_csv(first_nested(config, ["llm_training.eval_baselines"])),
        "correct,no_latent,zero_latent,shuffled",
    )
    set_default(args, "choice_score", first_nested(config, ["llm_training.choice_score"]), "mean")
    set_default(args, "log_interval", first_nested(config, ["llm_training.log_interval"]), 20)
    set_default(
        args,
        "checkpoint_metric",
        first_nested(config, ["llm_training.checkpoint_metric"]),
        "correct_accuracy",
    )
    set_default(args, "wandb_enabled", first_nested(config, ["wandb.enabled"]), False)
    set_default(args, "wandb_api_key", first_nested(config, ["wandb.api_key"]), None)
    set_default(args, "wandb_project", first_nested(config, ["wandb.project"]), "tensor-compression")
    set_default(args, "wandb_entity", first_nested(config, ["wandb.entity"]), None)
    set_default(args, "wandb_group", first_nested(config, ["wandb.group"]), "adapter")
    set_default(args, "wandb_tags", value_to_csv(first_nested(config, ["wandb.tags"])), "adapter,tensor-llm")
    set_default(args, "wandb_mode", first_nested(config, ["wandb.mode"]), "offline")
    set_default(args, "wandb_log_model", first_nested(config, ["wandb.log_model"]), False)
    require_args(args, ["qa_dir", "latent_dir", "model_name_or_path"])
    return args


def parse_csv(raw: str | Sequence[str] | None) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, Sequence) and not isinstance(raw, str):
        return [str(part).strip() for part in raw if str(part).strip()]
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_run_dir(output_root: str | Path, run_name: str) -> Path:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = root / f"{timestamp}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def resolve_model_dtype(raw: str, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if raw == "auto":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[raw]


def load_tokenizer_and_llm(args: argparse.Namespace, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        trust_remote_code=bool(args.trust_remote_code),
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            raise ValueError("Tokenizer has no pad/eos/unk token; cannot build padded batches.")

    model_dtype = resolve_model_dtype(str(args.torch_dtype), device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        torch_dtype=model_dtype,
        trust_remote_code=bool(args.trust_remote_code),
    )
    model.to(device)
    model.eval()
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return tokenizer, model, model_dtype


def apply_runtime_environment(args: argparse.Namespace) -> None:
    if args.hf_home:
        os.environ.setdefault("HF_HOME", str(args.hf_home))
    if args.cache_dir:
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(args.cache_dir) / "hub"))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(args.cache_dir))


def qa_path(qa_dir: str | Path, split: str) -> Path:
    path = Path(qa_dir) / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"QA split file not found: {path}")
    return path


def task_specific_instruction(record: Mapping[str, Any]) -> str:
    task_type = str(record.get("task_type", "")).strip()
    if task_type == "normalized_point_value":
        return (
            "Rule: read the standardized value at the requested patch-local row and column from the tensor soft tokens. "
            "Choose the closest numeric option and return only its label."
        )
    if task_type == "raw_point_value_with_stats":
        return (
            "Rule: read the standardized value z at the requested patch-local position from the tensor soft tokens, "
            "then use x = z * standard deviation + mean with the statistics stated in the question. "
            "Choose the closest original-value option and return only its label."
        )
    if task_type == "point_bin":
        return (
            "Rule: read the requested field value at the given row and col from the tensor soft tokens. "
            "Return its quantile-bin label. Bin labels B00,B01,... are ordered from low to high: "
            "B00 means the lowest value range, larger bin numbers mean larger value ranges, and the last bin "
            "means the highest value range. Return exactly one listed label and no extra text."
        )
    if task_type == "point_compare":
        return (
            "Rule: compare the requested field value at point A with point B using the tensor soft tokens. "
            "Choice A means point A is greater than or tied with point B. "
            "Choice B means point B is strictly greater than point A. "
            "Return exactly A or B and no extra text."
        )
    if task_type == "patch_compare":
        return (
            "Rule: compare the mean requested field value over patch A with patch B using the tensor soft tokens. "
            "Choice A means patch A has greater or tied mean. "
            "Choice B means patch B has strictly greater mean. "
            "Return exactly A or B and no extra text."
        )
    if task_type == "region_mean_compare":
        return (
            "Rule: compare the mean standardized values in the two stated patch-local regions using the tensor soft tokens. "
            "Return A if region A has the greater or tied mean; otherwise return B."
        )
    if task_type == "extreme_quadrant":
        return (
            "Rule: locate the requested maximum or minimum in the standardized patch using the tensor soft tokens. "
            "Return A for top-left, B for top-right, C for bottom-left, or D for bottom-right."
        )
    if task_type == "max_speed_quadrant":
        return (
            "Rule: find the grid cell with maximum speed magnitude from the tensor soft tokens. "
            "Return the quadrant label of that cell. Return exactly one listed label and no extra text."
        )
    if task_type == "global_stat_bin":
        return (
            "Rule: compute the requested global speed statistic from the tensor soft tokens. "
            "Return its quantile-bin label. Bin labels B00,B01,... are ordered from low to high: "
            "B00 means the lowest value range, larger bin numbers mean larger value ranges, and the last bin "
            "means the highest value range. Return exactly one listed label and no extra text."
        )
    return "Rule: answer the tensor readout query using the tensor soft tokens. Return exactly one listed label and no extra text."


def choice_semantics(record: Mapping[str, Any]) -> str:
    task_type = str(record.get("task_type", "")).strip()
    choices = record.get("choices")
    if not isinstance(choices, Sequence) or isinstance(choices, str):
        choices = []
    labels = [str(choice) for choice in choices]
    if task_type in {"point_bin", "global_stat_bin"} and labels:
        return (
            "Choice meanings: "
            + "; ".join(
                f"{label}=quantile bin {index} of {len(labels) - 1}, ordered from low to high"
                for index, label in enumerate(labels)
            )
            + "."
        )
    if task_type == "point_compare":
        return "Choice meanings: A=point A is greater than or tied with point B; B=point B is strictly greater than point A."
    if task_type == "patch_compare":
        return "Choice meanings: A=patch A mean is greater than or tied with patch B mean; B=patch B mean is strictly greater than patch A mean."
    if task_type == "region_mean_compare":
        return "Choice meanings: A=region A has greater or tied mean; B=region B has strictly greater mean."
    if task_type == "extreme_quadrant":
        return "Choice meanings: A=top-left; B=top-right; C=bottom-left; D=bottom-right."
    if task_type == "max_speed_quadrant":
        return "Choice meanings: quadrant labels refer to the location of the maximum-speed grid cell."
    if labels:
        return "Choice meanings: choose exactly one of the listed labels."
    return ""


def build_prompt(record: Mapping[str, Any], prompt_template: str) -> str:
    query = str(record.get("query") or record.get("question") or "")
    choices = record.get("choices")
    if not isinstance(choices, Sequence) or isinstance(choices, str):
        choices = []
    choice_text = ", ".join(str(choice) for choice in choices)
    if prompt_template == "generic":
        return (
            "Tensor-state soft tokens are prepended before this text.\n"
            "Answer the tensor readout query with exactly one choice label.\n\n"
            f"Query: {query}\n"
            f"Choices: {choice_text}\n"
            "Answer:"
        )
    if prompt_template != "task_specific":
        raise ValueError(f"Unsupported prompt template: {prompt_template}")
    return (
        "Tensor soft tokens before this text encode the tensor state.\n"
        f"{task_specific_instruction(record)}\n"
        "Do not answer from coordinate or label priors alone; use the tensor soft tokens for numeric values.\n\n"
        f"Query: {query}\n"
        f"Choices: {choice_text}\n"
        f"{choice_semantics(record)}\n"
        "Output format: one choice label only, such as A, B, B00, B01, B02, ... .\n"
        "Answer:"
    )


def encode_example(
    record: Mapping[str, Any],
    answer: str,
    tokenizer,
    max_prompt_tokens: int,
    max_target_tokens: int,
    append_eos: bool,
    prompt_template: str,
) -> tuple[list[int], list[int]]:
    prompt_ids = tokenizer(
        build_prompt(record, prompt_template=prompt_template),
        add_special_tokens=True,
        truncation=False,
    )["input_ids"]
    if len(prompt_ids) > max_prompt_tokens:
        prompt_ids = prompt_ids[-max_prompt_tokens:]
    answer_ids = tokenizer(
        " " + str(answer),
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]
    if append_eos and tokenizer.eos_token_id is not None:
        answer_ids = list(answer_ids) + [int(tokenizer.eos_token_id)]
    if len(answer_ids) > max_target_tokens:
        answer_ids = answer_ids[:max_target_tokens]
    if not answer_ids:
        raise ValueError(f"Answer tokenized to an empty sequence: {answer!r}")
    return list(prompt_ids), list(answer_ids)


def build_text_tensors(
    records: Sequence[Mapping[str, Any]],
    answers: Sequence[str],
    tokenizer,
    max_prompt_tokens: int,
    max_target_tokens: int,
    append_eos: bool,
    prompt_template: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    encoded = [
        encode_example(
            record=record,
            answer=answer,
            tokenizer=tokenizer,
            max_prompt_tokens=max_prompt_tokens,
            max_target_tokens=max_target_tokens,
            append_eos=append_eos,
            prompt_template=prompt_template,
        )
        for record, answer in zip(records, answers)
    ]
    total_lengths = [len(prompt_ids) + len(answer_ids) for prompt_ids, answer_ids in encoded]
    max_length = max(total_lengths)
    pad_id = int(tokenizer.pad_token_id)
    input_ids = torch.full((len(encoded), max_length), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(encoded), max_length), dtype=torch.long)
    labels = torch.full((len(encoded), max_length), IGNORE_INDEX, dtype=torch.long)
    for row, (prompt_ids, answer_ids) in enumerate(encoded):
        ids = prompt_ids + answer_ids
        input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[row, : len(ids)] = 1
        labels[row, len(prompt_ids) : len(ids)] = torch.tensor(answer_ids, dtype=torch.long)
    return input_ids, attention_mask, labels


def adapter_soft_embeds(
    adapter: TensorSoftPromptAdapter,
    latent_map: torch.Tensor,
    text_embeds: torch.Tensor,
    question_embeds: torch.Tensor | None,
    question_mask: torch.Tensor | None,
    records: Sequence[Mapping[str, Any]] | None,
    mode: str,
) -> torch.Tensor:
    structured_query = (
        structured_query_features(records, text_embeds.device)
        if records is not None and adapter.structured_query_conditioning
        else None
    )
    if mode == "correct":
        return adapter(
            latent_map,
            question_embeds=question_embeds,
            question_mask=question_mask,
            structured_query=structured_query,
        ).to(dtype=text_embeds.dtype)
    if mode == "no_latent":
        batch_size = latent_map.shape[0]
        return text_embeds.new_zeros((batch_size, adapter.soft_prompt_tokens, text_embeds.shape[-1]))
    if mode in {"shuffled", "random", "zero_latent"}:
        return adapter(
            latent_map,
            question_embeds=question_embeds,
            question_mask=question_mask,
            structured_query=structured_query,
        ).to(dtype=text_embeds.dtype)
    raise ValueError(f"Unsupported soft prompt mode: {mode}")


def forward_loss(
    llm,
    adapter: TensorSoftPromptAdapter,
    tokenizer,
    records: Sequence[Mapping[str, Any]],
    answers: Sequence[str],
    latent_map: torch.Tensor,
    device: torch.device,
    max_prompt_tokens: int,
    max_target_tokens: int,
    append_eos: bool,
    prompt_template: str,
    soft_prompt_mode: str = "correct",
) -> torch.Tensor:
    input_ids, text_attention_mask, text_labels = build_text_tensors(
        records=records,
        answers=answers,
        tokenizer=tokenizer,
        max_prompt_tokens=max_prompt_tokens,
        max_target_tokens=max_target_tokens,
        append_eos=append_eos,
        prompt_template=prompt_template,
    )
    input_ids = input_ids.to(device)
    text_attention_mask = text_attention_mask.to(device)
    text_labels = text_labels.to(device)
    latent_map = latent_map.to(device)

    text_embeds = llm.get_input_embeddings()(input_ids)
    prompt_mask = text_labels.eq(IGNORE_INDEX) & text_attention_mask.bool()
    soft_embeds = adapter_soft_embeds(
        adapter,
        latent_map,
        text_embeds,
        question_embeds=text_embeds,
        question_mask=prompt_mask,
        records=records,
        mode=soft_prompt_mode,
    )
    inputs_embeds = torch.cat([soft_embeds, text_embeds], dim=1)
    soft_attention = torch.ones(
        (input_ids.shape[0], soft_embeds.shape[1]),
        dtype=text_attention_mask.dtype,
        device=device,
    )
    attention_mask = torch.cat([soft_attention, text_attention_mask], dim=1)
    soft_labels = torch.full(
        (input_ids.shape[0], soft_embeds.shape[1]),
        IGNORE_INDEX,
        dtype=text_labels.dtype,
        device=device,
    )
    labels = torch.cat([soft_labels, text_labels], dim=1)
    sequence_nll, target_counts = selective_answer_nll(
        llm=llm,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        labels=labels,
    )
    return sequence_nll.sum() / target_counts.sum().clamp_min(1)


def selective_answer_nll(
    llm,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    decoder = None
    get_decoder = getattr(llm, "get_decoder", None)
    if callable(get_decoder):
        decoder = get_decoder()
    if decoder is None or decoder is llm:
        base_model_prefix = str(getattr(llm, "base_model_prefix", ""))
        decoder = getattr(llm, base_model_prefix, None) if base_model_prefix else None
    if decoder is None or decoder is llm:
        raise ValueError("The causal LLM does not expose a decoder for memory-efficient answer scoring.")
    output_embeddings = llm.get_output_embeddings()
    if output_embeddings is None:
        raise ValueError("The causal LLM does not expose output embeddings for answer scoring.")

    decoder_outputs = decoder(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    shift_hidden = decoder_outputs.last_hidden_state[:, :-1, :]
    shift_labels = labels[:, 1:]
    target_mask = shift_labels.ne(IGNORE_INDEX)
    if not bool(target_mask.any()):
        raise ValueError("Answer scoring received a batch without target tokens.")

    target_hidden = shift_hidden[target_mask]
    target_labels = shift_labels[target_mask]
    target_logits = output_embeddings(target_hidden).float()
    token_nll = F.cross_entropy(target_logits, target_labels, reduction="none")
    sequence_indices = (
        torch.arange(labels.shape[0], device=labels.device)
        .unsqueeze(1)
        .expand_as(target_mask)[target_mask]
    )
    sequence_nll = token_nll.new_zeros(labels.shape[0]).scatter_add(0, sequence_indices, token_nll)
    target_counts = target_mask.sum(dim=1)
    return sequence_nll, target_counts


def forward_answer_nll(
    llm,
    adapter: TensorSoftPromptAdapter,
    tokenizer,
    records: Sequence[Mapping[str, Any]],
    answers: Sequence[str],
    latent_map: torch.Tensor,
    device: torch.device,
    max_prompt_tokens: int,
    max_target_tokens: int,
    append_eos: bool,
    prompt_template: str,
    soft_prompt_mode: str = "correct",
    reduction: str = "mean",
    return_target_counts: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    input_ids, text_attention_mask, text_labels = build_text_tensors(
        records=records,
        answers=answers,
        tokenizer=tokenizer,
        max_prompt_tokens=max_prompt_tokens,
        max_target_tokens=max_target_tokens,
        append_eos=append_eos,
        prompt_template=prompt_template,
    )
    input_ids = input_ids.to(device)
    text_attention_mask = text_attention_mask.to(device)
    text_labels = text_labels.to(device)
    latent_map = latent_map.to(device)

    text_embeds = llm.get_input_embeddings()(input_ids)
    prompt_mask = text_labels.eq(IGNORE_INDEX) & text_attention_mask.bool()
    soft_embeds = adapter_soft_embeds(
        adapter,
        latent_map,
        text_embeds,
        question_embeds=text_embeds,
        question_mask=prompt_mask,
        records=records,
        mode=soft_prompt_mode,
    )
    inputs_embeds = torch.cat([soft_embeds, text_embeds], dim=1)
    soft_attention = torch.ones(
        (input_ids.shape[0], soft_embeds.shape[1]),
        dtype=text_attention_mask.dtype,
        device=device,
    )
    attention_mask = torch.cat([soft_attention, text_attention_mask], dim=1)
    soft_labels = torch.full(
        (input_ids.shape[0], soft_embeds.shape[1]),
        IGNORE_INDEX,
        dtype=text_labels.dtype,
        device=device,
    )
    labels = torch.cat([soft_labels, text_labels], dim=1)

    nll, target_counts = selective_answer_nll(
        llm=llm,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        labels=labels,
    )
    if reduction == "mean":
        nll = nll / target_counts.clamp_min(1)
    elif reduction != "sum":
        raise ValueError(f"Unsupported NLL reduction: {reduction}")
    if return_target_counts:
        return nll, target_counts
    return nll


def choice_ce_loss(
    llm,
    adapter: TensorSoftPromptAdapter,
    tokenizer,
    records: Sequence[Mapping[str, Any]],
    latent_map: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
    soft_prompt_mode: str = "correct",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    candidate_records: list[Mapping[str, Any]] = []
    candidate_answers: list[str] = []
    candidate_latents: list[torch.Tensor] = []
    candidate_counts: list[int] = []
    target_indices: list[int] = []
    for record_index, record in enumerate(records):
        choices = record.get("choices")
        if not isinstance(choices, Sequence) or isinstance(choices, str) or not choices:
            choices = [str(record["answer"])]
        string_choices = [str(choice) for choice in choices]
        answer = str(record["answer"])
        if answer not in string_choices:
            string_choices = [answer] + string_choices
        candidate_counts.append(len(string_choices))
        target_indices.append(string_choices.index(answer))
        for choice in string_choices:
            candidate_records.append(record)
            candidate_answers.append(choice)
            candidate_latents.append(latent_map[record_index])

    flat_nll_sum, flat_target_counts = forward_answer_nll(
        llm=llm,
        adapter=adapter,
        tokenizer=tokenizer,
        records=candidate_records,
        answers=candidate_answers,
        latent_map=torch.stack(candidate_latents, dim=0),
        device=device,
        max_prompt_tokens=int(args.max_prompt_tokens),
        max_target_tokens=int(args.max_target_tokens),
        append_eos=bool(args.append_eos),
        prompt_template=str(args.prompt_template),
        soft_prompt_mode=soft_prompt_mode,
        reduction="sum",
        return_target_counts=True,
    )
    if str(args.choice_score) == "mean":
        flat_nll = flat_nll_sum / flat_target_counts.clamp_min(1)
    elif str(args.choice_score) == "sum":
        flat_nll = flat_nll_sum
    else:
        raise ValueError(f"Unsupported choice_score: {args.choice_score}")
    losses: list[torch.Tensor] = []
    correct_nll_sums: list[torch.Tensor] = []
    correct_target_counts: list[torch.Tensor] = []
    correct_choice_nlls: list[torch.Tensor] = []
    hard_correct = 0
    start = 0
    for count, target_index in zip(candidate_counts, target_indices):
        scores = -flat_nll[start : start + count]
        target = torch.tensor([int(target_index)], dtype=torch.long, device=device)
        losses.append(F.cross_entropy(scores.unsqueeze(0), target))
        correct_flat_index = start + int(target_index)
        correct_nll_sums.append(flat_nll_sum[correct_flat_index])
        correct_target_counts.append(flat_target_counts[correct_flat_index])
        correct_choice_nlls.append(flat_nll[correct_flat_index])
        prediction = int(torch.argmax(scores.detach()).item())
        hard_correct += int(prediction == int(target_index))
        start += count
    if not losses:
        raise ValueError("choice_ce_loss received an empty record batch.")
    loss = torch.stack(losses).mean()
    correct_answer_ce = (
        torch.stack(correct_nll_sums).sum()
        / torch.stack(correct_target_counts).sum().clamp_min(1)
    )
    accuracy = hard_correct / max(1, len(losses))
    return loss, correct_answer_ce, torch.stack(correct_choice_nlls), {
        "choice_accuracy": float(accuracy),
        "choice_01_loss": float(1.0 - accuracy),
    }


def training_loss(
    llm,
    adapter: TensorSoftPromptAdapter,
    tokenizer,
    dataset: TensorReadoutQADataset,
    batch: Mapping[str, Any],
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    records = batch["records"]
    answers = [str(record["answer"]) for record in records]
    ce_weight = float(args.ce_loss_weight)
    choice_ce_weight = float(args.choice_ce_loss_weight)
    ranking_weight = float(args.ranking_loss_weight)
    choice_metrics = {
        "choice_accuracy": 0.0,
        "choice_01_loss": 0.0,
    }
    if choice_ce_weight > 0.0:
        choice_loss_value, ce_loss, positive_nll, choice_metrics = choice_ce_loss(
            llm=llm,
            adapter=adapter,
            tokenizer=tokenizer,
            records=records,
            latent_map=batch["latent_map"],
            device=device,
            args=args,
            soft_prompt_mode="correct",
        )
    else:
        ce_loss = forward_loss(
            llm=llm,
            adapter=adapter,
            tokenizer=tokenizer,
            records=records,
            answers=answers,
            latent_map=batch["latent_map"],
            device=device,
            max_prompt_tokens=int(args.max_prompt_tokens),
            max_target_tokens=int(args.max_target_tokens),
            append_eos=bool(args.append_eos),
            prompt_template=str(args.prompt_template),
        )
        choice_loss_value = ce_loss.new_zeros(())
        positive_nll = None
    if ranking_weight <= 0.0:
        weighted_ce_loss = ce_weight * ce_loss
        weighted_choice_ce_loss = choice_ce_weight * choice_loss_value
        total_loss = weighted_ce_loss + weighted_choice_ce_loss
        return total_loss, {
            "loss": float(total_loss.detach().cpu().item()),
            "ce_loss": float(ce_loss.detach().cpu().item()),
            "weighted_ce_loss": float(weighted_ce_loss.detach().cpu().item()),
            "choice_ce_loss": float(choice_loss_value.detach().cpu().item()),
            "weighted_choice_ce_loss": float(weighted_choice_ce_loss.detach().cpu().item()),
            "choice_accuracy": float(choice_metrics["choice_accuracy"]),
            "choice_01_loss": float(choice_metrics["choice_01_loss"]),
            "ranking_loss": 0.0,
            "weighted_ranking_loss": 0.0,
            "ranking_margin_mean": 0.0,
        }

    if positive_nll is None:
        positive_nll = forward_answer_nll(
            llm=llm,
            adapter=adapter,
            tokenizer=tokenizer,
            records=records,
            answers=answers,
            latent_map=batch["latent_map"],
            device=device,
            max_prompt_tokens=int(args.max_prompt_tokens),
            max_target_tokens=int(args.max_target_tokens),
            append_eos=bool(args.append_eos),
            prompt_template=str(args.prompt_template),
            soft_prompt_mode="correct",
            reduction=str(args.choice_score),
        )
    negative_mode = str(args.ranking_loss_negative)
    if negative_mode == "shuffled":
        negative_latents = baseline_latents("shuffled", batch, dataset)
    elif negative_mode == "random":
        negative_latents = baseline_latents("random", batch, dataset)
    elif negative_mode == "no_latent":
        negative_latents = batch["latent_map"]
    elif negative_mode == "zero_latent":
        negative_latents = torch.zeros_like(batch["latent_map"])
    else:
        raise ValueError(f"Unsupported ranking_loss_negative: {negative_mode}")
    negative_nll = forward_answer_nll(
        llm=llm,
        adapter=adapter,
        tokenizer=tokenizer,
        records=records,
        answers=answers,
        latent_map=negative_latents,
        device=device,
        max_prompt_tokens=int(args.max_prompt_tokens),
        max_target_tokens=int(args.max_target_tokens),
        append_eos=bool(args.append_eos),
        prompt_template=str(args.prompt_template),
        soft_prompt_mode=negative_mode,
        reduction=str(args.choice_score),
    )
    margin = float(args.ranking_loss_margin)
    ranking_terms = F.relu(margin + positive_nll - negative_nll)
    ranking_loss = ranking_terms.mean()
    weighted_ce_loss = ce_weight * ce_loss
    weighted_choice_ce_loss = choice_ce_weight * choice_loss_value
    weighted_ranking_loss = ranking_weight * ranking_loss
    total_loss = weighted_ce_loss + weighted_choice_ce_loss + weighted_ranking_loss
    detached_positive = positive_nll.detach()
    detached_negative = negative_nll.detach()
    return total_loss, {
        "loss": float(total_loss.detach().cpu().item()),
        "ce_loss": float(ce_loss.detach().cpu().item()),
        "weighted_ce_loss": float(weighted_ce_loss.detach().cpu().item()),
        "choice_ce_loss": float(choice_loss_value.detach().cpu().item()),
        "weighted_choice_ce_loss": float(weighted_choice_ce_loss.detach().cpu().item()),
        "choice_accuracy": float(choice_metrics["choice_accuracy"]),
        "choice_01_loss": float(choice_metrics["choice_01_loss"]),
        "ranking_loss": float(ranking_loss.detach().cpu().item()),
        "weighted_ranking_loss": float(weighted_ranking_loss.detach().cpu().item()),
        "ranking_margin_mean": float((detached_negative - detached_positive).mean().cpu().item()),
    }


@torch.no_grad()
def score_candidate_batch(
    llm,
    adapter: TensorSoftPromptAdapter,
    tokenizer,
    records: Sequence[Mapping[str, Any]],
    answers: Sequence[str],
    latent_map: torch.Tensor,
    device: torch.device,
    max_prompt_tokens: int,
    max_target_tokens: int,
    append_eos: bool,
    prompt_template: str,
    soft_prompt_mode: str,
    choice_score: str,
) -> list[float]:
    input_ids, text_attention_mask, text_labels = build_text_tensors(
        records=records,
        answers=answers,
        tokenizer=tokenizer,
        max_prompt_tokens=max_prompt_tokens,
        max_target_tokens=max_target_tokens,
        append_eos=append_eos,
        prompt_template=prompt_template,
    )
    input_ids = input_ids.to(device)
    text_attention_mask = text_attention_mask.to(device)
    text_labels = text_labels.to(device)
    latent_map = latent_map.to(device)

    text_embeds = llm.get_input_embeddings()(input_ids)
    prompt_mask = text_labels.eq(IGNORE_INDEX) & text_attention_mask.bool()
    soft_embeds = adapter_soft_embeds(
        adapter,
        latent_map,
        text_embeds,
        question_embeds=text_embeds,
        question_mask=prompt_mask,
        records=records,
        mode=soft_prompt_mode,
    )
    inputs_embeds = torch.cat([soft_embeds, text_embeds], dim=1)
    soft_attention = torch.ones(
        (input_ids.shape[0], soft_embeds.shape[1]),
        dtype=text_attention_mask.dtype,
        device=device,
    )
    attention_mask = torch.cat([soft_attention, text_attention_mask], dim=1)
    soft_labels = torch.full(
        (input_ids.shape[0], soft_embeds.shape[1]),
        IGNORE_INDEX,
        dtype=text_labels.dtype,
        device=device,
    )
    labels = torch.cat([soft_labels, text_labels], dim=1)

    nll, target_counts = selective_answer_nll(
        llm=llm,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        labels=labels,
    )
    if choice_score == "mean":
        nll = nll / target_counts.clamp_min(1)
    return [float(value) for value in nll.detach().cpu()]


def collect_candidate_scores(
    llm,
    adapter: TensorSoftPromptAdapter,
    tokenizer,
    records: Sequence[Mapping[str, Any]],
    latent_map: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
    mode: str,
) -> list[str]:
    candidate_records: list[Mapping[str, Any]] = []
    candidate_answers: list[str] = []
    candidate_latents: list[torch.Tensor] = []
    candidate_owner: list[int] = []
    choice_lists: list[list[str]] = []
    for record_index, record in enumerate(records):
        choices = record.get("choices")
        if not isinstance(choices, Sequence) or isinstance(choices, str) or not choices:
            choices = [str(record["answer"])]
        string_choices = [str(choice) for choice in choices]
        choice_lists.append(string_choices)
        for choice in string_choices:
            candidate_records.append(record)
            candidate_answers.append(choice)
            candidate_latents.append(latent_map[record_index].detach().cpu())
            candidate_owner.append(record_index)

    scores_by_record: list[list[float]] = [[] for _ in records]
    batch_size = max(1, int(args.eval_choice_batch_size))
    for start in range(0, len(candidate_records), batch_size):
        end = start + batch_size
        scores = score_candidate_batch(
            llm=llm,
            adapter=adapter,
            tokenizer=tokenizer,
            records=candidate_records[start:end],
            answers=candidate_answers[start:end],
            latent_map=torch.stack(candidate_latents[start:end], dim=0),
            device=device,
            max_prompt_tokens=int(args.max_prompt_tokens),
            max_target_tokens=int(args.max_target_tokens),
            append_eos=bool(args.append_eos),
            prompt_template=str(args.prompt_template),
            soft_prompt_mode=mode,
            choice_score=str(args.choice_score),
        )
        for owner, score in zip(candidate_owner[start:end], scores):
            scores_by_record[owner].append(score)

    predictions: list[str] = []
    for choices, scores in zip(choice_lists, scores_by_record):
        best_index = min(range(len(scores)), key=lambda index: scores[index])
        predictions.append(choices[best_index])
    return predictions


def baseline_latents(
    mode: str,
    batch: Mapping[str, Any],
    dataset: TensorReadoutQADataset,
) -> torch.Tensor:
    latents = batch["latent_map"]
    if mode in {"correct", "no_latent", "shuffled_stats"}:
        return latents
    if mode == "zero_latent":
        return torch.zeros_like(latents)
    if mode == "random":
        return torch.randn_like(latents)
    if mode == "shuffled":
        return torch.stack(
            [dataset.load_shuffled_latent(index) for index in batch["indices"]],
            dim=0,
        )
    raise ValueError(f"Unsupported baseline mode: {mode}")


def records_for_baseline(
    mode: str,
    batch: Mapping[str, Any],
    dataset: TensorReadoutQADataset,
) -> list[Mapping[str, Any]]:
    records = list(batch["records"])
    if mode != "shuffled_stats":
        return records
    updated: list[Mapping[str, Any]] = []
    for index, record in zip(batch["indices"], records):
        if str(record.get("task_type")) != "raw_point_value_with_stats":
            updated.append(record)
            continue
        prompt_data = record.get("prompt_data")
        shuffled = dataset.shuffled_record_for_index(index)
        shuffled_data = shuffled.get("prompt_data") if isinstance(shuffled, Mapping) else None
        if not isinstance(prompt_data, Mapping) or not isinstance(shuffled_data, Mapping):
            updated.append(record)
            continue
        digits = int(prompt_data.get("significant_digits", 6))
        mean = float(shuffled_data.get("mean", prompt_data["mean"]))
        std = float(shuffled_data.get("std", prompt_data["std"]))
        question = (
            f"A 16 by 16 patch of {prompt_data['field']} was standardized using "
            "z = (x - mean) / standard deviation. "
            f"Its mean is {mean:.{digits}g} and its standard deviation is {std:.{digits}g}. "
            "The standardized patch is encoded in the tensor soft tokens. Which option is closest to the "
            f"original value x at row {int(prompt_data['row'])}, column {int(prompt_data['col'])}? "
            f"Options: {prompt_data['option_text']}."
        )
        changed = dict(record)
        changed["question"] = question
        changed["query"] = question
        updated.append(changed)
    return updated


@torch.no_grad()
def evaluate_choice_accuracy(
    llm,
    adapter: nn.Module,
    tokenizer,
    dataset: TensorReadoutQADataset,
    device: torch.device,
    args: argparse.Namespace,
    baseline_modes: Sequence[str],
) -> dict[str, Any]:
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(args.eval_batch_size)),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_tensor_readout,
    )
    adapter.eval()
    metrics: dict[str, Any] = {}
    for mode in baseline_modes:
        total = 0
        correct = 0
        task_total: dict[str, int] = defaultdict(int)
        task_correct: dict[str, int] = defaultdict(int)
        field_total: dict[str, int] = defaultdict(int)
        field_correct: dict[str, int] = defaultdict(int)
        task_field_total: dict[str, int] = defaultdict(int)
        task_field_correct: dict[str, int] = defaultdict(int)
        for batch in tqdm(loader, desc=f"Eval [{mode}]", leave=False):
            records = records_for_baseline(mode, batch, dataset)
            latents = baseline_latents(mode, batch, dataset)
            predictions = collect_candidate_scores(
                llm=llm,
                adapter=adapter,
                tokenizer=tokenizer,
                records=records,
                latent_map=latents,
                device=device,
                args=args,
                mode="correct" if mode == "shuffled_stats" else mode,
            )
            for record, prediction in zip(records, predictions):
                answer = str(record["answer"])
                task_type = str(record.get("task_type", "unknown"))
                field = str(record.get("field") or record.get("metadata", {}).get("field") or "unknown")
                task_field = f"{task_type}/{field}"
                hit = int(prediction == answer)
                total += 1
                correct += hit
                task_total[task_type] += 1
                task_correct[task_type] += hit
                field_total[field] += 1
                field_correct[field] += hit
                task_field_total[task_field] += 1
                task_field_correct[task_field] += hit
        metrics[mode] = {
            "accuracy": correct / max(1, total),
            "correct": correct,
            "total": total,
            "by_task": {
                task: {
                    "accuracy": task_correct[task] / max(1, count),
                    "correct": task_correct[task],
                    "total": count,
                }
                for task, count in sorted(task_total.items())
            },
            "by_field": {
                field: {
                    "accuracy": field_correct[field] / max(1, count),
                    "correct": field_correct[field],
                    "total": count,
                }
                for field, count in sorted(field_total.items())
            },
            "by_task_field": {
                key: {
                    "accuracy": task_field_correct[key] / max(1, count),
                    "correct": task_field_correct[key],
                    "total": count,
                }
                for key, count in sorted(task_field_total.items())
            },
        }
    adapter.train()
    return metrics


def print_evaluation_summary(
    stage: str,
    metrics: Mapping[str, Any],
    detail_path: str | Path,
) -> None:
    print(f"{stage}: detailed metrics saved to {detail_path}")
    for mode, raw_mode_metrics in metrics.items():
        if not isinstance(raw_mode_metrics, Mapping):
            continue
        accuracy = float(raw_mode_metrics.get("accuracy", 0.0))
        correct = int(raw_mode_metrics.get("correct", 0))
        total = int(raw_mode_metrics.get("total", 0))
        by_task = raw_mode_metrics.get("by_task")
        task_parts: list[str] = []
        if isinstance(by_task, Mapping):
            for task, raw_task_metrics in sorted(by_task.items()):
                if isinstance(raw_task_metrics, Mapping):
                    task_parts.append(f"{task}={float(raw_task_metrics.get('accuracy', 0.0)):.3f}")
        task_suffix = f" tasks[{', '.join(task_parts)}]" if task_parts else ""
        print(f"  {mode}: accuracy={accuracy:.4f} ({correct}/{total}){task_suffix}")


def save_adapter_checkpoint(
    path: str | Path,
    adapter: nn.Module,
    args: argparse.Namespace,
    latent_shape: Sequence[int],
    llm_hidden_size: int,
    metrics: Mapping[str, Any] | None = None,
) -> None:
    payload = {
        "adapter_state_dict": adapter.state_dict(),
        "args": redacted_args(args),
        "latent_shape_chw": list(int(dim) for dim in latent_shape),
        "llm_hidden_size": int(llm_hidden_size),
        "metrics": dict(metrics or {}),
    }
    torch.save(payload, path)


def redacted_args(args: argparse.Namespace) -> dict[str, Any]:
    payload = dict(vars(args))
    if payload.get("wandb_api_key"):
        payload["wandb_api_key"] = "***REDACTED***"
    return payload


def flatten_numeric_metrics(prefix: str, metrics: Mapping[str, Any]) -> dict[str, float]:
    flattened: dict[str, float] = {}
    for key, value in metrics.items():
        metric_key = f"{prefix}/{key}"
        if isinstance(value, Mapping):
            flattened.update(flatten_numeric_metrics(metric_key, value))
        elif isinstance(value, bool):
            continue
        elif isinstance(value, (int, float)) and math.isfinite(float(value)):
            flattened[metric_key] = float(value)
    return flattened


def add_accuracy_deltas(prefix: str, metrics: Mapping[str, Any], payload: dict[str, float]) -> None:
    correct = metrics.get("correct")
    if not isinstance(correct, Mapping) or not isinstance(correct.get("accuracy"), (int, float)):
        return
    correct_accuracy = float(correct["accuracy"])
    for baseline in ("no_latent", "zero_latent", "shuffled", "shuffled_stats", "random"):
        baseline_metrics = metrics.get(baseline)
        if isinstance(baseline_metrics, Mapping) and isinstance(baseline_metrics.get("accuracy"), (int, float)):
            payload[f"{prefix}/correct_minus_{baseline}_accuracy"] = correct_accuracy - float(
                baseline_metrics["accuracy"]
            )


def macro_latent_gain(metrics: Mapping[str, Any], baseline: str = "shuffled") -> float:
    correct = metrics.get("correct")
    baseline_metrics = metrics.get(baseline)
    if not isinstance(correct, Mapping) or not isinstance(baseline_metrics, Mapping):
        return -math.inf
    correct_by_task = correct.get("by_task")
    baseline_by_task = baseline_metrics.get("by_task")
    if not isinstance(correct_by_task, Mapping) or not isinstance(baseline_by_task, Mapping):
        return -math.inf
    gains: list[float] = []
    for task, task_metrics in correct_by_task.items():
        baseline_task_metrics = baseline_by_task.get(task)
        if not isinstance(task_metrics, Mapping) or not isinstance(baseline_task_metrics, Mapping):
            continue
        correct_accuracy = task_metrics.get("accuracy")
        baseline_accuracy = baseline_task_metrics.get("accuracy")
        if isinstance(correct_accuracy, (int, float)) and isinstance(baseline_accuracy, (int, float)):
            gains.append(float(correct_accuracy) - float(baseline_accuracy))
    return sum(gains) / len(gains) if gains else -math.inf


def checkpoint_score(metrics: Mapping[str, Any], metric_name: str) -> float:
    if metric_name == "macro_latent_gain":
        return macro_latent_gain(metrics)
    if metric_name == "correct_accuracy":
        correct = metrics.get("correct")
        return float(correct.get("accuracy", -math.inf)) if isinstance(correct, Mapping) else -math.inf
    raise ValueError(f"Unsupported checkpoint metric: {metric_name}")


def build_wandb_config(args: argparse.Namespace, summary: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "experiment": {"name": str(args.run_name)},
        "data": {
            "qa_dir": str(args.qa_dir),
            "latent_dir": str(args.latent_dir),
            "train_split": str(args.train_split),
            "val_split": str(args.val_split),
            "test_split": str(args.test_split),
            "max_train_records": args.max_train_records,
            "max_val_records": args.max_val_records,
            "max_test_records": args.max_test_records,
            "initial_eval_records": int(args.initial_eval_records),
            "shuffle_seed": int(args.shuffle_seed),
        },
        "model": {
            "name_or_path": str(args.model_name_or_path),
            "torch_dtype": str(args.torch_dtype),
            "trust_remote_code": bool(args.trust_remote_code),
        },
        "adapter": {
            "architecture": str(args.adapter_architecture),
            "init_checkpoint": args.adapter_init_checkpoint,
            "soft_prompt_tokens": int(args.soft_prompt_tokens),
            "adapter_dim": int(args.adapter_dim),
            "adapter_layers": int(args.adapter_layers),
            "adapter_heads": int(args.adapter_heads),
            "dropout": float(args.dropout),
            "latent_pos_encoding": str(args.latent_pos_encoding),
            "question_conditioning": bool(args.question_conditioning),
            "question_condition_gate_init": float(args.question_condition_gate_init),
            "structured_query_conditioning": bool(args.structured_query_conditioning),
            "local_soft_prompt_tokens": int(args.local_soft_prompt_tokens),
            "local_adapter_layers": int(args.local_adapter_layers),
            "local_gate_init": float(args.local_gate_init),
            "freeze_global_adapter": bool(args.freeze_global_adapter),
            "global_unfreeze_epoch": int(args.global_unfreeze_epoch),
            "global_lr": float(args.global_lr),
            "soft_prompt_scale": float(args.soft_prompt_scale),
        },
        "llm_training": {
            "prompt_template": str(args.prompt_template),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "eval_batch_size": int(args.eval_batch_size),
            "eval_choice_batch_size": int(args.eval_choice_batch_size),
            "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "grad_clip_norm": float(args.grad_clip_norm),
            "ce_loss_weight": float(args.ce_loss_weight),
            "choice_ce_loss_weight": float(args.choice_ce_loss_weight),
            "ranking_loss_weight": float(args.ranking_loss_weight),
            "ranking_loss_margin": float(args.ranking_loss_margin),
            "ranking_loss_negative": str(args.ranking_loss_negative),
            "max_prompt_tokens": int(args.max_prompt_tokens),
            "max_target_tokens": int(args.max_target_tokens),
            "append_eos": bool(args.append_eos),
            "checkpoint_metric": str(args.checkpoint_metric),
            "eval_baselines": parse_csv(args.eval_baselines),
            "choice_score": str(args.choice_score),
            "log_interval": int(args.log_interval),
        },
        "run_summary": dict(summary or {}),
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


def log_adapter_artifact(wandb_logger: WandbLogger, path: Path, name: str) -> None:
    if wandb_logger.run is None or wandb_logger._wandb is None or not path.exists():
        return
    artifact = wandb_logger._wandb.Artifact(name=name, type="adapter-checkpoint")
    artifact.add_file(str(path))
    wandb_logger.run.log_artifact(artifact)


def main() -> None:
    args = parse_args()
    apply_runtime_environment(args)
    seed_everything(int(args.seed))
    device = resolve_device(args.device)
    run_dir = build_run_dir(args.output_root, args.run_name)
    dump_json(run_dir / "args.json", redacted_args(args))

    tokenizer, llm, model_dtype = load_tokenizer_and_llm(args, device)
    llm_hidden_size = int(llm.get_input_embeddings().embedding_dim)

    train_dataset = TensorReadoutQADataset(
        qa_path(args.qa_dir, args.train_split),
        latent_dir=args.latent_dir,
        max_records=args.max_train_records,
        prefer_record_latent_ref=bool(args.prefer_record_latent_ref),
        shuffle_seed=int(args.shuffle_seed),
    )
    val_dataset = TensorReadoutQADataset(
        qa_path(args.qa_dir, args.val_split),
        latent_dir=args.latent_dir,
        max_records=args.max_val_records,
        prefer_record_latent_ref=bool(args.prefer_record_latent_ref),
        shuffle_seed=int(args.shuffle_seed),
    )
    test_dataset = TensorReadoutQADataset(
        qa_path(args.qa_dir, args.test_split),
        latent_dir=args.latent_dir,
        max_records=args.max_test_records,
        prefer_record_latent_ref=bool(args.prefer_record_latent_ref),
        shuffle_seed=int(args.shuffle_seed),
    )
    first_latent = train_dataset[0]["latent_map"]
    latent_shape = tuple(int(dim) for dim in first_latent.shape)
    latent_channels = int(latent_shape[0])

    initialization = "random"
    if str(args.adapter_architecture) in {"alignment_qformer", "hybrid_local_qformer"}:
        checkpoint: Mapping[str, Any] | None = None
        checkpoint_args: Mapping[str, Any] = {}
        hybrid_state_dict: Mapping[str, Any] | None = None
        init_checkpoint = str(args.adapter_init_checkpoint or "").strip()
        if init_checkpoint.lower() in {"", "none", "null", "random"}:
            init_checkpoint = ""
        if init_checkpoint:
            loaded = torch.load(Path(init_checkpoint).expanduser(), map_location="cpu")
            if not isinstance(loaded, Mapping):
                raise ValueError(f"Unsupported alignment checkpoint: {args.adapter_init_checkpoint}")
            checkpoint = loaded
            checkpoint_args = loaded.get("args", {}) if isinstance(loaded.get("args"), Mapping) else {}
        query_tokens = int(checkpoint_args.get("query_tokens", args.soft_prompt_tokens))
        adapter_layers = int(checkpoint_args.get("adapter_layers", args.adapter_layers))
        adapter_heads = int(checkpoint_args.get("adapter_heads", args.adapter_heads))
        adapter_dim = int(checkpoint_args.get("adapter_dim", args.adapter_dim))
        checkpoint_projection_dim = int(checkpoint_args.get("projection_dim") or llm_hidden_size)
        if checkpoint_projection_dim != llm_hidden_size:
            raise ValueError(
                "Alignment adapter soft-prompt dimension must equal the downstream LLM hidden size. "
                f"Got checkpoint projection_dim={checkpoint_projection_dim}, llm_hidden_size={llm_hidden_size}."
            )
        latent_grid = tuple(int(dim) for dim in latent_shape[-2:])
        adapter = TensorPatchAlignmentAdapter(
            latent_channels=latent_channels,
            latent_grid=latent_grid,
            adapter_dim=adapter_dim,
            projection_dim=checkpoint_projection_dim,
            dropout=float(checkpoint_args.get("dropout", args.dropout)),
            adapter_type=str(checkpoint_args.get("adapter_type", "qformer")),
            query_tokens=query_tokens,
            adapter_layers=adapter_layers,
            adapter_heads=adapter_heads,
            soft_prompt_scale=float(checkpoint_args.get("soft_prompt_scale", args.soft_prompt_scale)),
        ).to(device)
        if checkpoint is not None:
            state_dict = checkpoint.get("adapter_state_dict")
            if not isinstance(state_dict, Mapping):
                raise ValueError("Alignment checkpoint does not contain adapter_state_dict.")
            if str(args.adapter_architecture) == "hybrid_local_qformer" and any(
                str(key).startswith("global_adapter.") for key in state_dict
            ):
                hybrid_state_dict = state_dict
                state_dict = {
                    str(key).removeprefix("global_adapter."): value
                    for key, value in state_dict.items()
                    if str(key).startswith("global_adapter.")
                }
            adapter.load_state_dict(state_dict, strict=True)
            initialization = "alignment_checkpoint"
            args.adapter_init_checkpoint = init_checkpoint
        else:
            args.adapter_init_checkpoint = None
        args.soft_prompt_tokens = query_tokens
        args.adapter_layers = adapter_layers
        args.adapter_heads = adapter_heads
        args.adapter_dim = adapter_dim
        args.question_conditioning = False
        args.structured_query_conditioning = False
        if str(args.adapter_architecture) == "hybrid_local_qformer":
            local_soft_prompt_tokens = int(
                checkpoint_args.get("local_soft_prompt_tokens", args.local_soft_prompt_tokens)
            )
            local_adapter_layers = int(checkpoint_args.get("local_adapter_layers", args.local_adapter_layers))
            local_gate_init = float(checkpoint_args.get("local_gate_init", args.local_gate_init))
            freeze_global_adapter = bool(
                checkpoint_args.get("freeze_global_adapter", args.freeze_global_adapter)
            )
            global_adapter = adapter
            local_adapter = QuestionConditionedLocalAdapter(
                latent_channels=latent_channels,
                latent_grid=latent_grid,
                llm_hidden_size=llm_hidden_size,
                adapter_dim=adapter_dim,
                local_tokens=local_soft_prompt_tokens,
                local_layers=local_adapter_layers,
                adapter_heads=adapter_heads,
                dropout=float(args.dropout),
                soft_prompt_scale=float(args.soft_prompt_scale),
                gate_init=local_gate_init,
                max_text_tokens=int(args.max_prompt_tokens) + int(args.max_target_tokens),
            )
            adapter = HybridGlobalLocalAdapter(
                global_adapter=global_adapter,
                local_adapter=local_adapter,
                freeze_global=freeze_global_adapter,
            ).to(device)
            if hybrid_state_dict is not None:
                adapter.load_state_dict(hybrid_state_dict, strict=True)
            args.soft_prompt_tokens = int(adapter.soft_prompt_tokens)
            args.local_soft_prompt_tokens = local_soft_prompt_tokens
            args.local_adapter_layers = local_adapter_layers
            args.local_gate_init = local_gate_init
            args.freeze_global_adapter = freeze_global_adapter
            args.question_conditioning = True
            args.structured_query_conditioning = True
    else:
        adapter = TensorSoftPromptAdapter(
            latent_channels=latent_channels,
            llm_hidden_size=llm_hidden_size,
            soft_prompt_tokens=int(args.soft_prompt_tokens),
            adapter_dim=int(args.adapter_dim),
            adapter_layers=int(args.adapter_layers),
            adapter_heads=int(args.adapter_heads),
            dropout=float(args.dropout),
            latent_pos_encoding=str(args.latent_pos_encoding),
            question_conditioning=bool(args.question_conditioning),
            question_condition_gate_init=float(args.question_condition_gate_init),
            structured_query_conditioning=bool(args.structured_query_conditioning),
            soft_prompt_scale=float(args.soft_prompt_scale),
        ).to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=max(1, int(args.batch_size)),
        shuffle=True,
        num_workers=0,
        collate_fn=collate_tensor_readout,
    )
    if isinstance(adapter, HybridGlobalLocalAdapter):
        optimizer = torch.optim.AdamW(
            [
                {"params": list(adapter.local_adapter.parameters()), "lr": float(args.lr), "name": "local"},
                {
                    "params": list(adapter.global_adapter.parameters()),
                    "lr": float(args.global_lr),
                    "name": "global",
                },
            ],
            weight_decay=float(args.weight_decay),
        )
    else:
        optimizer = torch.optim.AdamW(
            adapter.parameters(),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
        )
    baseline_modes = parse_csv(args.eval_baselines)
    if not baseline_modes:
        baseline_modes = ["correct"]

    summary = {
        "device": str(device),
        "model_dtype": str(model_dtype).replace("torch.", ""),
        "llm_hidden_size": llm_hidden_size,
        "latent_shape_chw": list(latent_shape),
        "train_records": len(train_dataset),
        "val_records": len(val_dataset),
        "test_records": len(test_dataset),
        "shuffle_seed": int(args.shuffle_seed),
        "ce_loss_weight": float(args.ce_loss_weight),
        "choice_ce_loss_weight": float(args.choice_ce_loss_weight),
        "ranking_loss_weight": float(args.ranking_loss_weight),
        "ranking_loss_margin": float(args.ranking_loss_margin),
        "ranking_loss_negative": str(args.ranking_loss_negative),
        "soft_prompt_tokens": int(args.soft_prompt_tokens),
        "adapter_layers": int(args.adapter_layers),
        "latent_pos_encoding": str(args.latent_pos_encoding),
        "question_conditioning": bool(args.question_conditioning),
        "question_condition_gate_init": float(args.question_condition_gate_init),
        "structured_query_conditioning": bool(args.structured_query_conditioning),
        "soft_prompt_scale": float(args.soft_prompt_scale),
        "adapter_architecture": str(args.adapter_architecture),
        "adapter_initialization": initialization,
        "adapter_init_checkpoint": str(args.adapter_init_checkpoint) if args.adapter_init_checkpoint else None,
        "local_soft_prompt_tokens": int(args.local_soft_prompt_tokens),
        "local_adapter_layers": int(args.local_adapter_layers),
        "local_gate_init": float(args.local_gate_init),
        "freeze_global_adapter": bool(args.freeze_global_adapter),
        "global_unfreeze_epoch": int(args.global_unfreeze_epoch),
        "global_lr": float(args.global_lr),
        "checkpoint_metric": str(args.checkpoint_metric),
        "trainable_adapter_parameters": sum(p.numel() for p in adapter.parameters() if p.requires_grad),
        "frozen_llm_parameters": sum(p.numel() for p in llm.parameters()),
    }
    dump_json(run_dir / "run_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    wandb_logger = WandbLogger(config=build_wandb_config(args, summary), run_dir=run_dir)

    best_val_score = -math.inf
    best_epoch = 0
    history: dict[str, Any] = {}
    accumulation_steps = max(1, int(args.gradient_accumulation_steps))
    global_step = 0
    try:
        initial_count = min(max(0, int(args.initial_eval_records)), len(val_dataset))
        if initial_count > 0:
            initial_dataset = TensorReadoutQADataset(
                qa_path(args.qa_dir, args.val_split),
                latent_dir=args.latent_dir,
                max_records=initial_count,
                prefer_record_latent_ref=bool(args.prefer_record_latent_ref),
                shuffle_seed=int(args.shuffle_seed),
            )
            initial_metrics = evaluate_choice_accuracy(
                llm=llm,
                adapter=adapter,
                tokenizer=tokenizer,
                dataset=initial_dataset,
                device=device,
                args=args,
                baseline_modes=baseline_modes,
            )
            history["initial_eval"] = initial_metrics
            metrics_path = run_dir / "metrics_latest.json"
            dump_json(metrics_path, history)
            initial_payload = flatten_numeric_metrics("initial_eval", initial_metrics)
            add_accuracy_deltas("initial_eval", initial_metrics, initial_payload)
            wandb_logger.log(initial_payload, step=0)
            print_evaluation_summary("initial_eval", initial_metrics, metrics_path)
            if device.type == "cuda":
                torch.cuda.empty_cache()
        for epoch in range(1, int(args.epochs) + 1):
            if (
                isinstance(adapter, HybridGlobalLocalAdapter)
                and adapter.freeze_global
                and int(args.global_unfreeze_epoch) > 0
                and epoch >= int(args.global_unfreeze_epoch)
            ):
                adapter.set_global_trainable(True)
                print(
                    f"unfroze global adapter at epoch {epoch}: "
                    f"global_lr={float(args.global_lr):.3g}, local_lr={float(args.lr):.3g}"
                )
            adapter.train()
            running_loss = 0.0
            running_ce_loss = 0.0
            running_weighted_ce_loss = 0.0
            running_choice_ce_loss = 0.0
            running_weighted_choice_ce_loss = 0.0
            running_choice_accuracy = 0.0
            running_choice_01_loss = 0.0
            running_ranking_loss = 0.0
            running_weighted_ranking_loss = 0.0
            running_ranking_margin = 0.0
            optimizer.zero_grad(set_to_none=True)
            progress = tqdm(train_loader, desc=f"Epoch {epoch:03d} [train]")
            for step, batch in enumerate(progress, start=1):
                loss, loss_parts = training_loss(
                    llm=llm,
                    adapter=adapter,
                    tokenizer=tokenizer,
                    dataset=train_dataset,
                    batch=batch,
                    device=device,
                    args=args,
                )
                (loss / accumulation_steps).backward()
                current_loss = float(loss_parts["loss"])
                running_loss += current_loss
                running_ce_loss += float(loss_parts["ce_loss"])
                running_weighted_ce_loss += float(loss_parts["weighted_ce_loss"])
                running_choice_ce_loss += float(loss_parts["choice_ce_loss"])
                running_weighted_choice_ce_loss += float(loss_parts["weighted_choice_ce_loss"])
                running_choice_accuracy += float(loss_parts["choice_accuracy"])
                running_choice_01_loss += float(loss_parts["choice_01_loss"])
                running_ranking_loss += float(loss_parts["ranking_loss"])
                running_weighted_ranking_loss += float(loss_parts["weighted_ranking_loss"])
                running_ranking_margin += float(loss_parts["ranking_margin_mean"])

                if step % accumulation_steps == 0 or step == len(train_loader):
                    if float(args.grad_clip_norm) > 0:
                        torch.nn.utils.clip_grad_norm_(adapter.parameters(), float(args.grad_clip_norm))
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                average_loss = running_loss / step
                average_ce_loss = running_ce_loss / step
                average_weighted_ce_loss = running_weighted_ce_loss / step
                average_choice_ce_loss = running_choice_ce_loss / step
                average_weighted_choice_ce_loss = running_weighted_choice_ce_loss / step
                average_choice_accuracy = running_choice_accuracy / step
                average_choice_01_loss = running_choice_01_loss / step
                average_ranking_loss = running_ranking_loss / step
                average_weighted_ranking_loss = running_weighted_ranking_loss / step
                average_ranking_margin = running_ranking_margin / step
                progress.set_postfix(
                    loss=f"{average_loss:.4f}",
                    ce=f"{average_ce_loss:.4f}",
                    choice=f"{average_choice_ce_loss:.4f}",
                    acc=f"{average_choice_accuracy:.3f}",
                    rank=f"{average_ranking_loss:.4f}",
                )
                if step % max(1, int(args.log_interval)) == 0:
                    step_payload = {
                        "epoch": epoch,
                        "step": step,
                        "global_step": global_step,
                        "train_loss": average_loss,
                        "train_ce_loss": average_ce_loss,
                        "train_weighted_ce_loss": average_weighted_ce_loss,
                        "train_choice_ce_loss": average_choice_ce_loss,
                        "train_weighted_choice_ce_loss": average_weighted_choice_ce_loss,
                        "train_choice_accuracy": average_choice_accuracy,
                        "train_choice_01_loss": average_choice_01_loss,
                        "train_ranking_loss": average_ranking_loss,
                        "train_weighted_ranking_loss": average_weighted_ranking_loss,
                        "train_ranking_margin": average_ranking_margin,
                    }
                    history[f"epoch_{epoch:04d}_step_{step:06d}"] = step_payload
                    dump_json(run_dir / "metrics_latest.json", history)
                    wandb_logger.log(
                        {
                            "train_step/loss": average_loss,
                            "train_step/ce_loss": average_ce_loss,
                            "train_step/weighted_ce_loss": average_weighted_ce_loss,
                            "train_step/choice_ce_loss": average_choice_ce_loss,
                            "train_step/weighted_choice_ce_loss": average_weighted_choice_ce_loss,
                            "train_step/choice_accuracy": average_choice_accuracy,
                            "train_step/choice_01_loss": average_choice_01_loss,
                            "train_step/ranking_loss": average_ranking_loss,
                            "train_step/weighted_ranking_loss": average_weighted_ranking_loss,
                            "train_step/ranking_margin": average_ranking_margin,
                            "train_step/current_loss": current_loss,
                            "train_step/current_ce_loss": float(loss_parts["ce_loss"]),
                            "train_step/current_weighted_ce_loss": float(loss_parts["weighted_ce_loss"]),
                            "train_step/current_choice_ce_loss": float(loss_parts["choice_ce_loss"]),
                            "train_step/current_weighted_choice_ce_loss": float(
                                loss_parts["weighted_choice_ce_loss"]
                            ),
                            "train_step/current_choice_accuracy": float(loss_parts["choice_accuracy"]),
                            "train_step/current_choice_01_loss": float(loss_parts["choice_01_loss"]),
                            "train_step/current_ranking_loss": float(loss_parts["ranking_loss"]),
                            "train_step/current_weighted_ranking_loss": float(loss_parts["weighted_ranking_loss"]),
                            "train_step/current_ranking_margin": float(loss_parts["ranking_margin_mean"]),
                            "train_step/epoch": float(epoch),
                            "train_step/epoch_step": float(step),
                            "train_step/lr": float(optimizer.param_groups[0]["lr"]),
                            "train_step/local_lr": float(optimizer.param_groups[0]["lr"]),
                            "train_step/global_lr": float(
                                optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else 0.0
                            ),
                            "train_step/global_trainable": float(
                                isinstance(adapter, HybridGlobalLocalAdapter) and not adapter.freeze_global
                            ),
                        },
                        step=global_step,
                    )

            train_loss = running_loss / max(1, len(train_loader))
            train_ce_loss = running_ce_loss / max(1, len(train_loader))
            train_weighted_ce_loss = running_weighted_ce_loss / max(1, len(train_loader))
            train_choice_ce_loss = running_choice_ce_loss / max(1, len(train_loader))
            train_weighted_choice_ce_loss = running_weighted_choice_ce_loss / max(1, len(train_loader))
            train_choice_accuracy = running_choice_accuracy / max(1, len(train_loader))
            train_choice_01_loss = running_choice_01_loss / max(1, len(train_loader))
            train_ranking_loss = running_ranking_loss / max(1, len(train_loader))
            train_weighted_ranking_loss = running_weighted_ranking_loss / max(1, len(train_loader))
            train_ranking_margin = running_ranking_margin / max(1, len(train_loader))
            val_metrics = evaluate_choice_accuracy(
                llm=llm,
                adapter=adapter,
                tokenizer=tokenizer,
                dataset=val_dataset,
                device=device,
                args=args,
                baseline_modes=baseline_modes,
            )
            epoch_payload = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_ce_loss": train_ce_loss,
                "train_weighted_ce_loss": train_weighted_ce_loss,
                "train_choice_ce_loss": train_choice_ce_loss,
                "train_weighted_choice_ce_loss": train_weighted_choice_ce_loss,
                "train_choice_accuracy": train_choice_accuracy,
                "train_choice_01_loss": train_choice_01_loss,
                "train_ranking_loss": train_ranking_loss,
                "train_weighted_ranking_loss": train_weighted_ranking_loss,
                "train_ranking_margin": train_ranking_margin,
                "val": val_metrics,
            }
            history[f"epoch_{epoch:04d}"] = epoch_payload
            dump_json(run_dir / "metrics_latest.json", history)
            save_adapter_checkpoint(
                run_dir / "adapter_last.pt",
                adapter=adapter,
                args=args,
                latent_shape=latent_shape,
                llm_hidden_size=llm_hidden_size,
                metrics=epoch_payload,
            )
            val_accuracy = float(val_metrics.get("correct", {}).get("accuracy", 0.0))
            val_macro_latent_gain = macro_latent_gain(val_metrics)
            val_score = checkpoint_score(val_metrics, str(args.checkpoint_metric))
            wandb_payload = {
                "epoch": float(epoch),
                "train/loss": float(train_loss),
                "train/ce_loss": float(train_ce_loss),
                "train/weighted_ce_loss": float(train_weighted_ce_loss),
                "train/choice_ce_loss": float(train_choice_ce_loss),
                "train/weighted_choice_ce_loss": float(train_weighted_choice_ce_loss),
                "train/choice_accuracy": float(train_choice_accuracy),
                "train/choice_01_loss": float(train_choice_01_loss),
                "train/ranking_loss": float(train_ranking_loss),
                "train/weighted_ranking_loss": float(train_weighted_ranking_loss),
                "train/ranking_margin": float(train_ranking_margin),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "local_lr": float(optimizer.param_groups[0]["lr"]),
                "global_lr": float(optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else 0.0),
                "global_trainable": float(
                    isinstance(adapter, HybridGlobalLocalAdapter) and not adapter.freeze_global
                ),
                "val/macro_latent_gain": float(val_macro_latent_gain),
                "best_val/checkpoint_score": float(max(best_val_score, val_score)),
            }
            wandb_payload.update(flatten_numeric_metrics("val", val_metrics))
            add_accuracy_deltas("val", val_metrics, wandb_payload)
            wandb_logger.log(wandb_payload, step=global_step)
            if val_score > best_val_score:
                best_val_score = val_score
                best_epoch = epoch
                save_adapter_checkpoint(
                    run_dir / "adapter_best.pt",
                    adapter=adapter,
                    args=args,
                    latent_shape=latent_shape,
                    llm_hidden_size=llm_hidden_size,
                    metrics=epoch_payload,
                )

        best_checkpoint_path = run_dir / "adapter_best.pt"
        if not best_checkpoint_path.exists():
            raise FileNotFoundError("Training completed without producing adapter_best.pt.")
        best_checkpoint = torch.load(best_checkpoint_path, map_location="cpu")
        best_state_dict = best_checkpoint.get("adapter_state_dict")
        if not isinstance(best_state_dict, Mapping):
            raise ValueError("adapter_best.pt does not contain adapter_state_dict.")
        adapter.load_state_dict(best_state_dict, strict=True)
        adapter.to(device)
        print(
            f"testing best checkpoint: epoch={best_epoch} "
            f"{args.checkpoint_metric}={best_val_score:.6f}"
        )
        test_metrics = evaluate_choice_accuracy(
            llm=llm,
            adapter=adapter,
            tokenizer=tokenizer,
            dataset=test_dataset,
            device=device,
            args=args,
            baseline_modes=baseline_modes,
        )
        dump_json(run_dir / "test_metrics.json", test_metrics)
        test_payload = flatten_numeric_metrics("test", test_metrics)
        add_accuracy_deltas("test", test_metrics, test_payload)
        wandb_logger.log(test_payload, step=global_step)
        if bool(args.wandb_log_model):
            log_adapter_artifact(wandb_logger, run_dir / "adapter_best.pt", f"{args.run_name}-best")
            log_adapter_artifact(wandb_logger, run_dir / "adapter_last.pt", f"{args.run_name}-last")
        print(f"run_dir: {run_dir}")
        print_evaluation_summary("test", test_metrics, run_dir / "test_metrics.json")
    finally:
        wandb_logger.finish()


if __name__ == "__main__":
    main()
