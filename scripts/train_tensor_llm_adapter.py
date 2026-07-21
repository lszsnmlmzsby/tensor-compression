from __future__ import annotations

import argparse
import atexit
import copy
import json
import math
import os
import random
import re
import sys
import time
from collections import OrderedDict, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.train_tensor_patch_text_alignment import (  # noqa: E402
    TensorPatchAlignmentAdapter,
    synchronize_gradients,
)

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
    from transformers import logging as transformers_logging
except ImportError as exc:  # pragma: no cover - exercised only in missing-dependency envs
    raise ImportError(
        "scripts/train_tensor_llm_adapter.py requires transformers. "
        "Install it with: pip install transformers accelerate safetensors"
    ) from exc


IGNORE_INDEX = -100
STRUCTURED_QUERY_FEATURE_DIM = 32
PATCH_QA_FORMAT = "tensor_patch_qa_v2"
PATCH_QA_PROMPT_CONTRACT = "encoder_zscore_one_based_v2"
SUPPORTED_BASELINE_MODES = {
    "correct",
    "global_only",
    "local_only",
    "no_latent",
    "zero_latent",
    "shuffled",
    "random",
    "shuffled_stats",
}
LOCAL_QUESTION_ANCHOR_TEXT = "Tensor evidence requested:"
_ACTIVE_RUN_LIFECYCLE: "RunLifecycle | None" = None


def distributed_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def distributed_rank() -> int:
    return int(dist.get_rank()) if distributed_is_initialized() else 0


def distributed_world_size() -> int:
    return int(dist.get_world_size()) if distributed_is_initialized() else 1


def is_main_process() -> bool:
    return distributed_rank() == 0


def initialize_distributed_device(requested_device: str) -> torch.device:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return resolve_device(requested_device)
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed Stage-2 training requires CUDA and the NCCL backend.")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank < 0 or local_rank >= torch.cuda.device_count():
        raise ValueError(
            f"LOCAL_RANK={local_rank} is invalid for {torch.cuda.device_count()} visible CUDA devices."
        )
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return torch.device("cuda", local_rank)


def distributed_barrier() -> None:
    if distributed_is_initialized():
        dist.barrier()


def broadcast_object_from_rank_zero(value: Any) -> Any:
    if not distributed_is_initialized():
        return value
    payload = [value if is_main_process() else None]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


def build_distributed_run_dir(output_root: str | Path, run_name: str) -> Path:
    if not distributed_is_initialized():
        return build_run_dir(output_root, run_name)
    payload: list[str | None] = [
        str(build_run_dir(output_root, run_name)) if is_main_process() else None
    ]
    dist.broadcast_object_list(payload, src=0)
    if payload[0] is None:
        raise RuntimeError("Rank 0 did not broadcast the Stage-2 run directory.")
    run_dir = Path(payload[0])
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


@torch.no_grad()
def synchronize_module_from_rank_zero(module: nn.Module) -> None:
    if not distributed_is_initialized():
        return
    for parameter in module.parameters():
        dist.broadcast(parameter.data, src=0)
    for buffer in module.buffers():
        dist.broadcast(buffer.data, src=0)


def average_trainable_gradients(module: nn.Module) -> None:
    synchronize_gradients([module])


def distributed_sum_scalars(values: Mapping[str, float], device: torch.device) -> dict[str, float]:
    names = list(values)
    tensor = torch.tensor(
        [float(values[name]) for name in names],
        dtype=torch.float64,
        device=device,
    )
    if distributed_is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return {name: float(value) for name, value in zip(names, tensor.cpu().tolist())}


def gather_cuda_memory(device: torch.device) -> list[dict[str, float]]:
    if device.type != "cuda":
        return []
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    local = torch.tensor(
        [free_bytes, total_bytes, torch.cuda.memory_allocated(device)],
        dtype=torch.int64,
        device=device,
    )
    if distributed_is_initialized():
        gathered = [torch.empty_like(local) for _ in range(distributed_world_size())]
        dist.all_gather(gathered, local)
    else:
        gathered = [local]
    return [
        {
            "rank": float(rank),
            "free_gib": float(values[0].item()) / 1024**3,
            "total_gib": float(values[1].item()) / 1024**3,
            "allocated_gib": float(values[2].item()) / 1024**3,
        }
        for rank, values in enumerate(gathered)
    ]


def local_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_dump_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    temporary = target.with_name(f".{target.name}.tmp")
    dump_json(temporary, payload)
    os.replace(temporary, target)


def atomic_torch_save(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, target)


class RunLifecycle:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.started_at = local_timestamp()
        self.started_monotonic = time.monotonic()
        self.finished = False
        self.last_payload: dict[str, Any] | None = None
        self._write("running")
        atexit.register(self._finish_at_exit)

    def _payload(self, status: str, error: BaseException | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": status,
            "started_at": self.started_at,
            "ended_at": None if status == "running" else local_timestamp(),
            "duration_seconds": None if status == "running" else round(time.monotonic() - self.started_monotonic, 3),
            "timezone": datetime.now().astimezone().tzname(),
        }
        if error is not None:
            payload["error_type"] = type(error).__name__
            payload["error_message"] = str(error)[:2000]
        return payload

    def _write(self, status: str, error: BaseException | None = None) -> dict[str, Any]:
        timing = self._payload(status, error)
        self.last_payload = timing
        dump_json(self.run_dir / "run_timing.json", timing)
        summary_path = self.run_dir / "run_summary.json"
        if summary_path.exists():
            with summary_path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            if isinstance(summary, dict):
                summary["timing"] = timing
                dump_json(summary_path, summary)
        return timing

    def finish(self, status: str, error: BaseException | None = None) -> dict[str, Any]:
        if self.finished:
            return dict(self.last_payload or self._payload(status, error))
        self.finished = True
        return self._write(status, error)

    def _finish_at_exit(self) -> None:
        if not self.finished:
            self.finish("aborted_or_failed")


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
    coordinate_origin = int(metadata.get("coordinate_origin", 0))
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
        row = int(point.group(1)) - coordinate_origin
        col = int(point.group(2)) - coordinate_origin
        features[16] = _normalize_coordinate(row, height)
        features[17] = _normalize_coordinate(col, width)

    point_pair = re.search(
        r"A at row (\d+), column (\d+).*?B at row (\d+), column (\d+)",
        query,
        re.IGNORECASE,
    ) or re.search(r"A=\((\d+),(\d+)\)\s+B=\((\d+),(\d+)\)", query)
    if point_pair:
        row_a, col_a, row_b, col_b = [int(group) for group in point_pair.groups()]
        row_a, col_a, row_b, col_b = (
            row_a - coordinate_origin,
            col_a - coordinate_origin,
            row_b - coordinate_origin,
            col_b - coordinate_origin,
        )
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
        row_a, col_a, row_b, col_b = (
            row_a - coordinate_origin,
            col_a - coordinate_origin,
            row_b - coordinate_origin,
            col_b - coordinate_origin,
        )
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
        features[29] = math.tanh(math.log1p(abs(float(prompt_data.get("scale", prompt_data.get("std", 0.0))))))
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
        self.capture_attention = False
        self.last_attention_weights: torch.Tensor | None = None

    def forward(
        self,
        queries: torch.Tensor,
        latents: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attended, weights = self.attention(
            query=self.query_norm(queries),
            key=self.latent_norm(latents),
            value=self.latent_norm(latents),
            key_padding_mask=key_padding_mask,
            need_weights=bool(self.capture_attention),
            average_attn_weights=False,
        )
        self.last_attention_weights = (
            weights.detach().float().cpu() if self.capture_attention and weights is not None else None
        )
        queries = queries + attended
        return queries + self.ffn(self.ffn_norm(queries))


class GatedTextCrossAttentionBlock(nn.Module):
    """Condition inherited Q-Former queries without discarding their aligned initialization."""

    def __init__(self, dim: int, heads: int, dropout: float, gate_init: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.text_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.capture_attention = False
        self.last_attention_weights: torch.Tensor | None = None

    def forward(
        self,
        queries: torch.Tensor,
        text: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attended, weights = self.attention(
            query=self.query_norm(queries),
            key=self.text_norm(text),
            value=self.text_norm(text),
            key_padding_mask=key_padding_mask,
            need_weights=bool(self.capture_attention),
            average_attn_weights=False,
        )
        self.last_attention_weights = (
            weights.detach().float().cpu() if self.capture_attention and weights is not None else None
        )
        return queries + self.gate.to(dtype=queries.dtype) * attended


def last_nonpadding_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    if attention_mask.ndim != 2:
        raise ValueError(f"Expected a [batch,tokens] attention mask, got {tuple(attention_mask.shape)}.")
    valid = attention_mask.to(dtype=torch.bool)
    if int(valid.shape[1]) == 0:
        raise ValueError("Natural-language questions must contain at least one token.")
    positions = torch.arange(valid.shape[1], device=valid.device).unsqueeze(0).expand_as(valid)
    return positions.masked_fill(~valid, -1).max(dim=1).values


class BidirectionalCrossAttentionBlock(nn.Module):
    """Update text tokens from latent tokens without collapsing the text sequence first."""

    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.text_norm = nn.LayerNorm(dim)
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

    def forward(self, text: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        attended, _weights = self.attention(
            query=self.text_norm(text),
            key=self.latent_norm(latents),
            value=self.latent_norm(latents),
            need_weights=False,
        )
        text = text + attended
        return text + self.ffn(self.ffn_norm(text))


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
    """Extract local latent evidence using a natural-language question."""

    def __init__(
        self,
        latent_channels: int,
        latent_grid: Sequence[int],
        llm_hidden_size: int,
        adapter_dim: int,
        local_tokens: int,
        local_layers: int,
        text_encoder_layers: int,
        adapter_heads: int,
        dropout: float,
        soft_prompt_scale: float,
        gate_init: float,
        max_text_tokens: int,
        structured_query_conditioning: bool,
        question_input_mode: str = "input_embeddings",
        fusion_mode: str = "text_latent_pool",
    ) -> None:
        super().__init__()
        if int(adapter_dim) % int(adapter_heads) != 0:
            raise ValueError("adapter_dim must be divisible by adapter_heads for the local adapter.")
        self.soft_prompt_tokens = int(local_tokens)
        self.latent_grid = tuple(int(dim) for dim in latent_grid)
        self.soft_prompt_scale = float(soft_prompt_scale)
        self.structured_query_conditioning = bool(structured_query_conditioning)
        self.question_input_mode = str(question_input_mode)
        if self.question_input_mode not in {"input_embeddings", "contextual_tokens"}:
            raise ValueError(f"Unsupported local question_input_mode: {question_input_mode}")
        self.fusion_mode = str(fusion_mode)
        if self.fusion_mode not in {"text_latent_pool", "anchor_queries"}:
            raise ValueError(f"Unsupported local fusion_mode: {fusion_mode}")
        self.latent_projection = nn.Linear(int(latent_channels), int(adapter_dim))
        self.position_projection = nn.Linear(2, int(adapter_dim))
        self.text_projection = nn.Sequential(
            nn.LayerNorm(int(llm_hidden_size)),
            nn.Linear(int(llm_hidden_size), int(adapter_dim)),
        )
        self.text_pos_embed = nn.Parameter(torch.zeros(1, int(max_text_tokens), int(adapter_dim)))
        self.text_encoder = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=int(adapter_dim),
                    nhead=int(adapter_heads),
                    dim_feedforward=int(adapter_dim) * 4,
                    dropout=float(dropout),
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(int(text_encoder_layers))
            ]
        )
        self.text_encoder_norm = nn.LayerNorm(int(adapter_dim)) if int(text_encoder_layers) > 0 else nn.Identity()
        if self.structured_query_conditioning:
            self.structured_projection = nn.Sequential(
                nn.LayerNorm(STRUCTURED_QUERY_FEATURE_DIM),
                nn.Linear(STRUCTURED_QUERY_FEATURE_DIM, int(adapter_dim)),
                nn.GELU(),
                nn.Linear(int(adapter_dim), int(adapter_dim)),
            )
        else:
            self.structured_projection = None
        self.query_tokens = nn.Parameter(torch.empty(1, int(local_tokens), int(adapter_dim)))
        use_query_blocks = self.question_input_mode == "input_embeddings" or self.fusion_mode == "anchor_queries"
        self.text_blocks = nn.ModuleList(
            [CrossAttentionBlock(int(adapter_dim), int(adapter_heads), float(dropout)) for _ in range(int(local_layers))]
            if use_query_blocks
            else []
        )
        self.latent_blocks = nn.ModuleList(
            [CrossAttentionBlock(int(adapter_dim), int(adapter_heads), float(dropout)) for _ in range(int(local_layers))]
            if use_query_blocks
            else []
        )
        self.text_latent_blocks = nn.ModuleList(
            [
                BidirectionalCrossAttentionBlock(int(adapter_dim), int(adapter_heads), float(dropout))
                for _ in range(int(local_layers))
            ]
            if self.question_input_mode == "contextual_tokens" and self.fusion_mode == "text_latent_pool"
            else []
        )
        self.pool_blocks = nn.ModuleList(
            [CrossAttentionBlock(int(adapter_dim), int(adapter_heads), float(dropout))]
            if self.question_input_mode == "contextual_tokens" and self.fusion_mode == "text_latent_pool"
            else []
        )
        if self.question_input_mode == "contextual_tokens" and self.fusion_mode == "anchor_queries":
            self.anchor_projection = nn.Sequential(
                nn.LayerNorm(int(adapter_dim)),
                nn.Linear(int(adapter_dim), int(adapter_dim)),
                nn.GELU(),
                nn.Linear(int(adapter_dim), int(adapter_dim)),
            )
            self.anchor_gate = nn.Parameter(torch.tensor(1.0))
        else:
            self.anchor_projection = None
            self.register_parameter("anchor_gate", None)
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
        structured_query: torch.Tensor | None,
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
        if self.question_input_mode == "input_embeddings":
            text_context = text_context + self.text_pos_embed[:, : text_context.shape[1]]
        key_padding_mask = None
        if question_mask is not None:
            key_padding_mask = ~question_mask.to(device=text_context.device, dtype=torch.bool)
        for layer in self.text_encoder:
            text_context = layer(text_context, src_key_padding_mask=key_padding_mask)
        text_context = self.text_encoder_norm(text_context)
        queries = self.query_tokens.expand(latents.shape[0], -1, -1)
        if self.anchor_projection is not None:
            if question_mask is None:
                last_indices = torch.full(
                    (text_context.shape[0],),
                    int(text_context.shape[1]) - 1,
                    dtype=torch.long,
                    device=text_context.device,
                )
            else:
                last_indices = last_nonpadding_indices(question_mask.to(device=text_context.device))
            anchor = text_context[
                torch.arange(text_context.shape[0], device=text_context.device),
                last_indices,
            ]
            anchor_condition = self.anchor_projection(anchor)
            queries = queries + self.anchor_gate.to(dtype=queries.dtype) * anchor_condition.unsqueeze(1)
        if self.structured_projection is not None and structured_query is not None:
            query_condition = self.structured_projection(
                structured_query.to(device=queries.device, dtype=self.structured_projection[1].weight.dtype)
            )
            queries = queries + query_condition.unsqueeze(1)
        if self.question_input_mode == "contextual_tokens" and self.fusion_mode == "text_latent_pool":
            fused_text = text_context
            for text_latent_block in self.text_latent_blocks:
                fused_text = text_latent_block(fused_text, latents)
            for pool_block in self.pool_blocks:
                queries = pool_block(queries, fused_text, key_padding_mask=key_padding_mask)
        else:
            for text_block, latent_block in zip(self.text_blocks, self.latent_blocks):
                queries = text_block(queries, text_context, key_padding_mask=key_padding_mask)
                queries = latent_block(queries, latents)
        local_prompts = self.output(queries)
        if self.soft_prompt_scale > 0.0:
            local_prompts = torch.tanh(local_prompts) * self.soft_prompt_scale
        return self.gate.to(dtype=local_prompts.dtype) * local_prompts


class ResidualQuestionConditionedAdapter(nn.Module):
    """Question-condition a stage-1 adapter while preserving its token layout and initialization."""

    def __init__(
        self,
        aligned_adapter: TensorPatchAlignmentAdapter,
        llm_hidden_size: int,
        context_layers: Sequence[int],
        adapter_heads: int,
        dropout: float,
        text_gate_init: float,
        residual_gate_init: float,
    ) -> None:
        super().__init__()
        if str(aligned_adapter.adapter_type) not in {"qformer", "spatial_transformer"}:
            raise ValueError(
                "Residual question conditioning requires a stage-1 qformer or spatial_transformer checkpoint."
            )
        self.backbone = copy.deepcopy(aligned_adapter)
        self.soft_prompt_tokens = int(aligned_adapter.soft_prompt_tokens)
        self.latent_grid = tuple(int(value) for value in aligned_adapter.latent_grid)
        self.context_layers = tuple(int(value) for value in context_layers)
        if not self.context_layers:
            raise ValueError("adapter.local_context_layers must contain at least one Qwen hidden-state index.")
        adapter_dim = int(aligned_adapter.adapter_dim)
        self.text_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(int(llm_hidden_size)),
                    nn.Linear(int(llm_hidden_size), adapter_dim),
                )
                for _ in self.context_layers
            ]
        )
        self.text_layer_logits = nn.Parameter(torch.zeros(len(self.context_layers)))
        self.text_blocks = nn.ModuleList(
            [
                GatedTextCrossAttentionBlock(
                    dim=adapter_dim,
                    heads=int(adapter_heads),
                    dropout=float(dropout),
                    gate_init=float(text_gate_init),
                )
                for _ in self.backbone.blocks
            ]
        )
        self.gate = nn.Parameter(torch.tensor(float(residual_gate_init)))
        self.structured_query_conditioning = False
        self.question_input_mode = "contextual_tokens"
        self.fusion_mode = (
            "residual_spatial_transformer"
            if str(aligned_adapter.adapter_type) == "spatial_transformer"
            else "residual_qformer"
        )

    @property
    def query_tokens(self) -> nn.Parameter:
        if not hasattr(self.backbone, "query_tokens"):
            raise AttributeError("A spatial_transformer adapter has no free query tokens.")
        return self.backbone.query_tokens

    def forward(
        self,
        latent_map: torch.Tensor,
        question_embeds: torch.Tensor,
        question_mask: torch.Tensor | None,
        structured_query: torch.Tensor | None,
    ) -> torch.Tensor:
        if structured_query is not None:
            raise ValueError("Residual question conditioning does not accept parsed structured query features.")
        if question_embeds.ndim == 3:
            question_embeds = question_embeds.unsqueeze(1)
        if question_embeds.ndim != 4 or int(question_embeds.shape[1]) != len(self.context_layers):
            raise ValueError(
                "Expected contextual question states [batch,layers,tokens,hidden] for layers "
                f"{self.context_layers}, got {tuple(question_embeds.shape)}."
            )
        projected_layers = [
            projection(question_embeds[:, index].detach().to(dtype=projection[1].weight.dtype))
            for index, projection in enumerate(self.text_projections)
        ]
        fusion_weights = torch.softmax(self.text_layer_logits.float(), dim=0).to(projected_layers[0].dtype)
        text_context = sum(
            fusion_weights[index] * projected
            for index, projected in enumerate(projected_layers)
        )
        key_padding_mask = (
            ~question_mask.to(device=text_context.device, dtype=torch.bool)
            if question_mask is not None
            else None
        )

        if str(self.backbone.adapter_type) == "spatial_transformer":
            states, local_residual = self.backbone.spatial_input_states(latent_map)
            for text_block, spatial_block in zip(self.text_blocks, self.backbone.blocks):
                states = text_block(states, text_context, key_padding_mask=key_padding_mask)
                states = spatial_block(states)
            return self.backbone.spatial_output_states(states, local_residual)

        latent_tokens = self.backbone.flatten_latent_tokens(latent_map)
        latent_context = (
            self.backbone.latent_projection(latent_tokens.to(dtype=self.backbone.latent_projection.weight.dtype))
            + self.backbone.latent_pos_embed
        )
        queries = self.backbone.query_tokens.expand(latent_map.shape[0], -1, -1)
        for text_block, latent_block in zip(self.text_blocks, self.backbone.blocks):
            queries = text_block(queries, text_context, key_padding_mask=key_padding_mask)
            queries = latent_block(queries, latent_context)
        return self.backbone.scale_soft_prompts(self.backbone.output(queries))


class HybridGlobalLocalAdapter(nn.Module):
    def __init__(
        self,
        global_adapter: TensorPatchAlignmentAdapter,
        local_adapter: nn.Module,
        freeze_global: bool,
        global_prompt_dropout: float = 0.0,
        combine_mode: str = "concat",
    ) -> None:
        super().__init__()
        self.global_adapter = global_adapter
        self.local_adapter = local_adapter
        self.freeze_global = bool(freeze_global)
        self.global_prompt_dropout = float(global_prompt_dropout)
        self.combine_mode = str(combine_mode)
        if self.combine_mode not in {"concat", "residual"}:
            raise ValueError(f"Unsupported global/local combine mode: {combine_mode}")
        if not 0.0 <= self.global_prompt_dropout < 1.0:
            raise ValueError("global_prompt_dropout must be in [0, 1).")
        self.drop_global_prompts_for_batch = False
        self.soft_prompt_tokens = int(
            global_adapter.soft_prompt_tokens
            if self.combine_mode == "residual"
            else global_adapter.soft_prompt_tokens + local_adapter.soft_prompt_tokens
        )
        self.structured_query_conditioning = bool(local_adapter.structured_query_conditioning)

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

    def set_global_prompt_dropout_for_batch(self, enabled: bool) -> None:
        self.drop_global_prompts_for_batch = bool(enabled)

    @property
    def residual_mode(self) -> bool:
        return self.combine_mode == "residual"

    def forward_components(
        self,
        latent_map: torch.Tensor,
        question_embeds: torch.Tensor | None = None,
        question_mask: torch.Tensor | None = None,
        structured_query: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if question_embeds is None:
            raise ValueError("hybrid_local_qformer requires natural-language question embeddings.")
        if self.freeze_global:
            with torch.no_grad():
                global_prompts = self.global_adapter.forward_soft_prompts(latent_map)
        else:
            global_prompts = self.global_adapter.forward_soft_prompts(latent_map)
        conditioned_prompts = self.local_adapter(latent_map, question_embeds, question_mask, structured_query)
        if self.residual_mode:
            if conditioned_prompts.shape != global_prompts.shape:
                raise ValueError(
                    "Residual conditioned/global prompts must have identical shapes, got "
                    f"{tuple(conditioned_prompts.shape)} and {tuple(global_prompts.shape)}."
                )
            local_prompts = self.local_adapter.gate.to(dtype=conditioned_prompts.dtype) * (
                conditioned_prompts - global_prompts
            )
            visible_global = (
                torch.zeros_like(global_prompts)
                if self.training and self.drop_global_prompts_for_batch
                else global_prompts
            )
            return visible_global, local_prompts, visible_global + local_prompts
        local_prompts = conditioned_prompts
        if self.training and self.drop_global_prompts_for_batch:
            global_prompts = torch.zeros_like(global_prompts)
        # Keeping local tokens first preserves the relative positions between global tokens and text.
        return global_prompts, local_prompts, torch.cat([local_prompts, global_prompts], dim=1)

    def forward(
        self,
        latent_map: torch.Tensor,
        question_embeds: torch.Tensor | None = None,
        question_mask: torch.Tensor | None = None,
        structured_query: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_components(latent_map, question_embeds, question_mask, structured_query)[2]


class TensorReadoutQADataset(Dataset):
    def __init__(
        self,
        jsonl_path: str | Path,
        latent_dir: str | Path,
        max_records: int | None = None,
        prefer_record_latent_ref: bool = False,
        shuffle_seed: int = 42,
        latent_cache_size: int = 0,
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.latent_dir = Path(latent_dir)
        self.prefer_record_latent_ref = bool(prefer_record_latent_ref)
        self.records = self._load_records(self.jsonl_path)
        if max_records is not None:
            self.records = self.records[: max(0, int(max_records))]
        if not self.records:
            raise RuntimeError(f"No QA records found in {self.jsonl_path}.")
        self.latent_cache_size = max(0, int(latent_cache_size))
        self._latent_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._latent_path_cache: dict[str, Path] = {}
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
                # Oracle values are generation/debug metadata and never enter the training process.
                payload.pop("oracle", None)
                records.append(payload)
        return records

    def _build_random_different_indices(self, seed: int) -> list[int]:
        unique_states = {str(record.get("state_ref", "")) for record in self.records}
        if len(unique_states) < 2:
            raise RuntimeError(
                "Cannot build shuffled latent baseline: every record belongs to the same state_ref."
            )
        rng = random.Random(seed)
        by_field_task_sample: dict[tuple[str, str], dict[int, list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        by_field_task_state: dict[tuple[str, str], dict[str, list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        by_field_sample: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
        by_field_state: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        by_sample: dict[int, list[int]] = defaultdict(list)
        by_state: dict[str, list[int]] = defaultdict(list)
        for candidate_index, candidate_record in enumerate(self.records):
            field = str(
                candidate_record.get("field")
                or candidate_record.get("metadata", {}).get("field")
                or ""
            )
            task = str(candidate_record.get("task_type", ""))
            sample_index = int(candidate_record.get("sample_index", -1))
            state_ref = str(candidate_record.get("state_ref", ""))
            by_field_task_sample[(field, task)][sample_index].append(candidate_index)
            by_field_task_state[(field, task)][state_ref].append(candidate_index)
            by_field_sample[field][sample_index].append(candidate_index)
            by_field_state[field][state_ref].append(candidate_index)
            by_sample[sample_index].append(candidate_index)
            by_state[state_ref].append(candidate_index)

        def index_buckets(
            buckets: Mapping[Any, Sequence[int]],
        ) -> tuple[tuple[Any, ...], dict[Any, int], Mapping[Any, Sequence[int]]]:
            keys = tuple(buckets)
            return keys, {key: index for index, key in enumerate(keys)}, buckets

        field_task_sources = {
            key: index_buckets(buckets) for key, buckets in by_field_task_sample.items()
        }
        field_task_state_sources = {
            key: index_buckets(buckets) for key, buckets in by_field_task_state.items()
        }
        field_sources = {key: index_buckets(buckets) for key, buckets in by_field_sample.items()}
        field_state_sources = {key: index_buckets(buckets) for key, buckets in by_field_state.items()}
        sample_source = index_buckets(by_sample)
        state_source = index_buckets(by_state)

        def choose_from_other_bucket(
            source: tuple[tuple[Any, ...], Mapping[Any, int], Mapping[Any, Sequence[int]]],
            excluded_key: Any,
        ) -> int | None:
            keys, positions, buckets = source
            if not keys:
                return None
            excluded_position = positions.get(excluded_key)
            if excluded_position is None:
                selected_key = keys[rng.randrange(len(keys))]
            elif len(keys) <= 1:
                return None
            else:
                selected_position = rng.randrange(len(keys) - 1)
                if selected_position >= excluded_position:
                    selected_position += 1
                selected_key = keys[selected_position]
            return int(rng.choice(buckets[selected_key]))

        indices: list[int] = []
        for record in self.records:
            state_ref = str(record.get("state_ref", ""))
            sample_index = int(record.get("sample_index", -1))
            field = str(record.get("field") or record.get("metadata", {}).get("field") or "")
            task = str(record.get("task_type", ""))
            candidate = choose_from_other_bucket(
                field_task_sources.get((field, task), ((), {}, {})), sample_index
            )
            if candidate is None:
                candidate = choose_from_other_bucket(field_sources.get(field, ((), {}, {})), sample_index)
            if candidate is None:
                candidate = choose_from_other_bucket(sample_source, sample_index)
            if candidate is None:
                candidate = choose_from_other_bucket(
                    field_task_state_sources.get((field, task), ((), {}, {})), state_ref
                )
            if candidate is None:
                candidate = choose_from_other_bucket(field_state_sources.get(field, ((), {}, {})), state_ref)
            if candidate is None:
                candidate = choose_from_other_bucket(state_source, state_ref)
            if candidate is None:
                raise RuntimeError("Failed to sample a different-state shuffled latent.")
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
        cached = self._latent_path_cache.get(state_ref)
        if cached is not None:
            return cached
        latent_from_dir = self.latent_dir / f"{state_ref}.pt"
        record_ref = record.get("latent_ref")
        if self.prefer_record_latent_ref and record_ref:
            selected = Path(str(record_ref))
        elif latent_from_dir.exists():
            selected = latent_from_dir
        elif record_ref:
            selected = Path(str(record_ref))
        else:
            selected = latent_from_dir
        self._latent_path_cache[state_ref] = selected
        return selected

    def load_latent_for_record(self, record: Mapping[str, Any]) -> torch.Tensor:
        path = self.latent_path_for_record(record)
        if not path.exists():
            raise FileNotFoundError(f"Latent cache file not found: {path}")
        cache_key = str(path)
        cached = self._latent_cache.get(cache_key)
        if cached is not None:
            self._latent_cache.move_to_end(cache_key)
            return cached
        payload = torch.load(path, map_location="cpu")
        latent = payload.get("latent_map") if isinstance(payload, Mapping) else payload
        if not isinstance(latent, torch.Tensor):
            raise ValueError(f"Latent cache file does not contain a tensor latent_map: {path}")
        if latent.ndim == 4 and latent.shape[0] == 1:
            latent = latent.squeeze(0)
        if latent.ndim != 3:
            raise ValueError(f"Expected latent_map [C,H,W], got {tuple(latent.shape)} from {path}")
        latent = latent.to(dtype=torch.float32)
        if self.latent_cache_size > 0:
            self._latent_cache[cache_key] = latent
            self._latent_cache.move_to_end(cache_key)
            while len(self._latent_cache) > self.latent_cache_size:
                self._latent_cache.popitem(last=False)
        return latent

    def load_shuffled_latent(self, index: int) -> torch.Tensor:
        other_index = self._random_different_indices[int(index)]
        return self.load_latent_for_record(self.records[other_index])

    def shuffled_record_for_index(self, index: int) -> Mapping[str, Any]:
        other_index = self._random_different_indices[int(index)]
        return self.records[other_index]


class StateTaskGroupedBatchSampler(Sampler[list[int]]):
    """Keep a few same-tensor, same-operation questions together without parsing their text."""

    def __init__(
        self,
        dataset: TensorReadoutQADataset,
        batch_size: int,
        questions_per_group: int,
        seed: int,
        rank: int = 0,
        num_replicas: int = 1,
    ) -> None:
        self.dataset = dataset
        self.batch_size = max(1, int(batch_size))
        self.questions_per_group = max(1, int(questions_per_group))
        if self.questions_per_group > self.batch_size:
            raise ValueError(
                "llm_training.questions_per_state_group cannot exceed llm_training.batch_size."
            )
        self.seed = int(seed)
        self.rank = int(rank)
        self.num_replicas = max(1, int(num_replicas))
        if self.rank < 0 or self.rank >= self.num_replicas:
            raise ValueError(f"rank must be in [0, {self.num_replicas}), got {self.rank}.")
        self.epoch = 0
        initial_batches = self._global_batches(epoch=0)
        self._length = math.ceil(len(initial_batches) / self.num_replicas)
        initial_sizes = [len(batch) for batch in initial_batches]
        self.initial_batch_size_min = min(initial_sizes, default=0)
        self.initial_batch_size_max = max(initial_sizes, default=0)
        self.initial_batch_size_mean = (
            sum(initial_sizes) / len(initial_sizes) if initial_sizes else 0.0
        )

    def __len__(self) -> int:
        return self._length

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _global_batches(self, epoch: int) -> list[list[int]]:
        rng = random.Random(self.seed + int(epoch))
        grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
        for index, record in enumerate(self.dataset.records):
            key = (str(record.get("state_ref", "")), str(record.get("task_type", "")))
            grouped[key].append(index)

        units: list[list[int]] = []
        for indices in grouped.values():
            rng.shuffle(indices)
            units.extend(
                indices[start : start + self.questions_per_group]
                for start in range(0, len(indices), self.questions_per_group)
            )
        rng.shuffle(units)
        units.sort(key=len, reverse=True)
        batches: list[list[int]] = []
        available_by_capacity: dict[int, list[int]] = defaultdict(list)
        for unit in units:
            selected_batch: int | None = None
            for capacity in range(len(unit), self.batch_size + 1):
                if available_by_capacity[capacity]:
                    selected_batch = available_by_capacity[capacity].pop()
                    break
            if selected_batch is None:
                batches.append(list(unit))
                selected_batch = len(batches) - 1
            else:
                batches[selected_batch].extend(unit)
            remaining = self.batch_size - len(batches[selected_batch])
            if remaining > 0:
                available_by_capacity[remaining].append(selected_batch)
        rng.shuffle(batches)
        return batches

    def __iter__(self):
        batches = self._global_batches(self.epoch)
        if self.num_replicas > 1:
            target_count = len(self) * self.num_replicas
            if len(batches) < target_count:
                if not batches:
                    return
                missing = target_count - len(batches)
                padding_source = list(batches)
                batches.extend(
                    copy.deepcopy(padding_source[index % len(padding_source)])
                    for index in range(missing)
                )
            batches = batches[self.rank:target_count:self.num_replicas]
        yield from batches


class ExactDistributedEvalSampler(Sampler[int]):
    """Shard evaluation records without padding or repeating any example."""

    def __init__(self, dataset: Dataset, rank: int, num_replicas: int) -> None:
        self.dataset = dataset
        self.rank = int(rank)
        self.num_replicas = max(1, int(num_replicas))
        if self.rank < 0 or self.rank >= self.num_replicas:
            raise ValueError(f"rank must be in [0, {self.num_replicas}), got {self.rank}.")

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self) -> int:
        if self.rank >= len(self.dataset):
            return 0
        return ((len(self.dataset) - 1 - self.rank) // self.num_replicas) + 1


def collate_tensor_readout(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "indices": [int(item["index"]) for item in items],
        "records": [item["record"] for item in items],
        "latent_map": torch.stack([item["latent_map"] for item in items], dim=0),
    }


_DISPLAYED_OPTION_PATTERN = re.compile(
    r"(?:Options:\s*|;\s*)([A-D]):\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)


def audit_qa_datasets(
    datasets: Mapping[str, TensorReadoutQADataset],
    require_disjoint_splits: bool,
) -> dict[str, Any]:
    split_states: dict[str, set[str]] = {}
    split_samples: dict[str, set[int]] = {}
    split_latent_paths: dict[str, set[str]] = {}
    split_tasks: dict[str, set[str]] = {}
    split_fields: dict[str, set[str]] = {}
    summary: dict[str, Any] = {}
    for split, dataset in datasets.items():
        qa_ids: set[str] = set()
        states: set[str] = set()
        samples: set[int] = set()
        latent_paths: set[str] = set()
        task_counts: dict[str, int] = defaultdict(int)
        field_counts: dict[str, int] = defaultdict(int)
        answer_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        choice_labels: dict[str, set[str]] = defaultdict(set)
        numeric_option_records = 0
        ascending_numeric_option_records = 0
        for record in dataset.records:
            qa_id = str(record.get("qa_id", ""))
            if not qa_id or qa_id in qa_ids:
                raise ValueError(f"QA audit found a missing or duplicate qa_id in {split}: {qa_id!r}")
            qa_ids.add(qa_id)
            state_ref = str(record.get("state_ref", ""))
            if not state_ref:
                raise ValueError(f"QA audit found a missing state_ref: {qa_id}")
            states.add(state_ref)
            samples.add(int(record["sample_index"]))
            task = str(record.get("task_type", "unknown"))
            field = str(record.get("field") or record.get("metadata", {}).get("field") or "unknown")
            task_counts[task] += 1
            field_counts[field] += 1
            answer_counts[task][str(record.get("answer", ""))] += 1
            choices = record.get("choices")
            if not isinstance(choices, Sequence) or isinstance(choices, str) or str(record.get("answer")) not in choices:
                raise ValueError(f"QA audit found invalid choices/answer: {qa_id}")
            choice_labels[task].update(str(choice) for choice in choices)
            latent_path = dataset.latent_path_for_record(record)
            resolved_latent_path = str(latent_path.resolve())
            if resolved_latent_path not in latent_paths:
                if not latent_path.exists():
                    raise FileNotFoundError(f"QA audit found a missing latent cache file: {latent_path}")
                latent_paths.add(resolved_latent_path)
            if task in {"normalized_point_value", "raw_point_value_with_stats"}:
                displayed = _DISPLAYED_OPTION_PATTERN.findall(str(record.get("query") or record.get("question") or ""))
                displayed_values = [float(value) for _label, value in displayed]
                if len(displayed_values) != len(choices) or len(set(displayed_values)) != len(displayed_values):
                    raise ValueError(
                        f"QA audit found missing or duplicate displayed numeric options: {qa_id}. "
                        "Regenerate patch QA with sufficient decimal_places."
                    )
                numeric_option_records += 1
                ascending_numeric_option_records += int(
                    all(left < right for left, right in zip(displayed_values, displayed_values[1:]))
                )
        split_states[split] = states
        split_samples[split] = samples
        split_latent_paths[split] = latent_paths
        split_tasks[split] = set(task_counts)
        split_fields[split] = set(field_counts)
        missing_answer_labels = {
            task: sorted(labels - set(answer_counts[task]))
            for task, labels in choice_labels.items()
            if labels - set(answer_counts[task])
        }
        if missing_answer_labels and require_disjoint_splits:
            raise ValueError(
                f"QA audit found answer labels absent from formal split {split}: {missing_answer_labels}"
            )
        ascending_fraction = ascending_numeric_option_records / max(1, numeric_option_records)
        if require_disjoint_splits and numeric_option_records >= 16 and ascending_fraction > 0.5:
            raise ValueError(
                f"QA audit found {ascending_fraction:.1%} ascending numeric option lists in {split}. "
                "This looks like stale QA generated before option-order randomization; rerun "
                "scripts/build_tensor_patch_qa.py."
            )
        summary[split] = {
            "records": len(dataset),
            "states": len(states),
            "samples": len(samples),
            "latent_files": len(latent_paths),
            "by_task": dict(sorted(task_counts.items())),
            "by_field": dict(sorted(field_counts.items())),
            "answers_by_task": {
                task: dict(sorted(counts.items())) for task, counts in sorted(answer_counts.items())
            },
            "missing_answer_labels": missing_answer_labels,
            "numeric_option_records": numeric_option_records,
            "ascending_numeric_option_fraction": ascending_fraction,
        }
    reference_split = "train" if "train" in split_tasks else next(iter(split_tasks))
    for split in split_tasks:
        if not require_disjoint_splits or split == reference_split:
            continue
        if split_tasks[split] != split_tasks[reference_split]:
            raise ValueError(
                f"QA audit found task coverage mismatch between {reference_split} and {split}: "
                f"{sorted(split_tasks[reference_split])} vs {sorted(split_tasks[split])}"
            )
        if split_fields[split] != split_fields[reference_split]:
            raise ValueError(
                f"QA audit found field coverage mismatch between {reference_split} and {split}: "
                f"{sorted(split_fields[reference_split])} vs {sorted(split_fields[split])}"
            )
    split_names = list(split_states)
    overlaps: dict[str, int] = {}
    sample_overlaps: dict[str, int] = {}
    latent_file_overlaps: dict[str, int] = {}
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            count = len(split_states[left] & split_states[right])
            overlaps[f"{left}_{right}"] = count
            if count and require_disjoint_splits:
                raise ValueError(
                    f"QA audit found {count} state_ref values shared by {left} and {right}; "
                    "formal evaluation requires disjoint states."
                )
            sample_count = len(split_samples[left] & split_samples[right])
            sample_overlaps[f"{left}_{right}"] = sample_count
            if sample_count and require_disjoint_splits:
                raise ValueError(
                    f"QA audit found {sample_count} sample_index values shared by {left} and {right}; "
                    "use patch_qa.split_mode: sample for generalization evaluation."
                )
            latent_count = len(split_latent_paths[left] & split_latent_paths[right])
            latent_file_overlaps[f"{left}_{right}"] = latent_count
            if latent_count and require_disjoint_splits:
                raise ValueError(
                    f"QA audit found {latent_count} latent files shared by {left} and {right}."
                )
    summary["state_overlap"] = overlaps
    summary["sample_overlap"] = sample_overlaps
    summary["latent_file_overlap"] = latent_file_overlaps
    summary["require_disjoint_splits"] = bool(require_disjoint_splits)
    summary["evaluation_scope"] = "formal_generalization" if require_disjoint_splits else "sanity_only"
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a soft-prompt adapter from cached tensor latents into a frozen causal LM."
    )
    parser.add_argument("--config", type=str, default=None, help="Optional tensor-LLM pipeline YAML config.")
    parser.add_argument("--qa-dir", type=str, default=None)
    parser.add_argument("--latent-dir", type=str, default=None)
    parser.add_argument(
        "--qa-alignment-checkpoint",
        type=str,
        default=None,
        help="Checkpoint provenance expected in patch QA metadata; useful for isolated transfer tests.",
    )
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
    parser.add_argument("--require-disjoint-splits", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--require-untruncated-prompts", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--prefer-record-latent-ref", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--latent-cache-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
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
    parser.add_argument(
        "--train-choice-batch-size",
        type=int,
        default=None,
        help="Maximum candidate-answer sequences sent through the frozen LLM in one training forward.",
    )
    parser.add_argument(
        "--train-grounding-batch-size",
        type=int,
        default=None,
        help="Maximum ranking/swap sequences sent through the frozen LLM in one training forward.",
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument(
        "--llm-gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Checkpoint frozen-LLM decoder layers while retaining gradients to tensor soft prompts.",
    )
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lr-scheduler", choices=("constant", "cosine"), default=None)
    parser.add_argument("--warmup-ratio", type=float, default=None)
    parser.add_argument("--min-lr-ratio", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    parser.add_argument("--ce-loss-weight", type=float, default=None)
    parser.add_argument("--choice-ce-loss-weight", type=float, default=None)
    parser.add_argument("--ranking-loss-weight", type=float, default=None)
    parser.add_argument("--ranking-loss-margin", type=float, default=None)
    parser.add_argument("--swapped-question-loss-weight", type=float, default=None)
    parser.add_argument("--swapped-question-loss-margin", type=float, default=None)
    parser.add_argument("--swapped-question-max-records", type=int, default=None)
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
        choices=(
            "legacy",
            "alignment_qformer",
            "alignment_adapter",
            "hybrid_local_qformer",
            "residual_question_qformer",
            "residual_question_adapter",
        ),
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
    parser.add_argument("--local-text-encoder-layers", type=int, default=None)
    parser.add_argument(
        "--local-question-input-mode",
        type=str,
        choices=("input_embeddings", "contextual_tokens"),
        default=None,
    )
    parser.add_argument("--local-context-layer", type=int, default=None)
    parser.add_argument(
        "--local-context-layers",
        type=str,
        default=None,
        help="Comma-separated Qwen hidden-state indices fused for question conditioning.",
    )
    parser.add_argument(
        "--local-fusion-mode",
        type=str,
        choices=(
            "text_latent_pool",
            "anchor_queries",
            "residual_qformer",
            "residual_spatial_transformer",
        ),
        default=None,
    )
    parser.add_argument("--local-gate-init", type=float, default=None)
    parser.add_argument("--local-text-gate-init", type=float, default=None)
    parser.add_argument("--freeze-global-adapter", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--global-unfreeze-epoch", type=int, default=None)
    parser.add_argument("--global-lr", type=float, default=None)
    parser.add_argument("--global-prompt-dropout", type=float, default=None)
    parser.add_argument("--group-questions-by-state", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--questions-per-state-group", type=int, default=None)
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
        help=(
            "Comma-separated: correct,global_only,local_only,no_latent,zero_latent,"
            "shuffled,random,shuffled_stats."
        ),
    )
    parser.add_argument(
        "--final-eval-baselines",
        type=str,
        default=None,
        help="Comma-separated baselines used once on the final best checkpoint.",
    )
    parser.add_argument(
        "--choice-score",
        type=str,
        default=None,
        choices=("mean", "sum"),
        help="Normalize candidate NLL by target-token count or not.",
    )
    parser.add_argument(
        "--choice-scoring-mode",
        type=str,
        default=None,
        choices=("auto", "label", "sequence"),
        help="Use one-forward restricted-label scoring when possible, or retain the exact sequence scorer.",
    )
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--console-progress", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save-step-metrics", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--diagnostics-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--diagnostics-every-epochs", type=int, default=None)
    parser.add_argument("--diagnostics-records-per-task", type=int, default=None)
    parser.add_argument("--diagnostics-generation-max-new-tokens", type=int, default=None)
    parser.add_argument(
        "--diagnostics-layers",
        type=str,
        default=None,
        help="Comma-separated decoder hidden-state indices; -1 selects the final state.",
    )
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
    parser.add_argument("--wandb-detailed-metrics", action=argparse.BooleanOptionalAction, default=None)
    return apply_config_defaults(parser.parse_args())


def apply_config_defaults(args: argparse.Namespace) -> argparse.Namespace:
    config = load_yaml_mapping(args.config)
    configured_loss_fields = {
        "ce_loss_weight": first_nested(config, ["llm_training.ce_loss_weight"]),
        "choice_ce_loss_weight": first_nested(config, ["llm_training.choice_ce_loss_weight"]),
        "ranking_loss_weight": first_nested(config, ["llm_training.ranking_loss_weight"]),
        "ranking_loss_margin": first_nested(config, ["llm_training.ranking_loss_margin"]),
        "swapped_question_loss_weight": first_nested(config, ["llm_training.swapped_question_loss_weight"]),
        "swapped_question_loss_margin": first_nested(config, ["llm_training.swapped_question_loss_margin"]),
    }
    defaulted_loss_fields = [
        field
        for field, configured_value in configured_loss_fields.items()
        if args.config and getattr(args, field, None) is None and configured_value is None
    ]

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
        "qa_alignment_checkpoint": first_nested(config, ["patch_qa.alignment_checkpoint"]),
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
    set_default(args, "require_disjoint_splits", first_nested(config, ["llm_training.require_disjoint_splits"]), True)
    set_default(
        args,
        "require_untruncated_prompts",
        first_nested(config, ["llm_training.require_untruncated_prompts"]),
        True,
    )
    set_default(
        args,
        "prefer_record_latent_ref",
        first_nested(config, ["llm_training.prefer_record_latent_ref"]),
        False,
    )
    set_default(args, "latent_cache_size", first_nested(config, ["llm_training.latent_cache_size"]), 32768)
    set_default(args, "num_workers", first_nested(config, ["llm_training.num_workers"]), 0)
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
        "train_choice_batch_size",
        first_nested(config, ["llm_training.train_choice_batch_size"]),
        4,
    )
    set_default(
        args,
        "train_grounding_batch_size",
        first_nested(config, ["llm_training.train_grounding_batch_size"]),
        args.train_choice_batch_size,
    )
    set_default(
        args,
        "gradient_accumulation_steps",
        first_nested(config, ["llm_training.gradient_accumulation_steps"]),
        1,
    )
    set_default(
        args,
        "llm_gradient_checkpointing",
        first_nested(config, ["llm_training.llm_gradient_checkpointing"]),
        True,
    )
    set_default(args, "lr", first_nested(config, ["llm_training.lr"]), 1.0e-4)
    set_default(args, "lr_scheduler", first_nested(config, ["llm_training.lr_scheduler"]), "constant")
    set_default(args, "warmup_ratio", first_nested(config, ["llm_training.warmup_ratio"]), 0.0)
    set_default(args, "min_lr_ratio", first_nested(config, ["llm_training.min_lr_ratio"]), 0.1)
    set_default(args, "weight_decay", first_nested(config, ["llm_training.weight_decay"]), 1.0e-2)
    set_default(args, "grad_clip_norm", first_nested(config, ["llm_training.grad_clip_norm"]), 1.0)
    set_default(args, "ce_loss_weight", configured_loss_fields["ce_loss_weight"], 0.05)
    set_default(args, "choice_ce_loss_weight", configured_loss_fields["choice_ce_loss_weight"], 1.0)
    set_default(args, "ranking_loss_weight", configured_loss_fields["ranking_loss_weight"], 0.1)
    set_default(args, "ranking_loss_margin", configured_loss_fields["ranking_loss_margin"], 0.1)
    set_default(args, "swapped_question_loss_weight", configured_loss_fields["swapped_question_loss_weight"], 0.0)
    set_default(args, "swapped_question_loss_margin", configured_loss_fields["swapped_question_loss_margin"], 0.1)
    set_default(
        args,
        "swapped_question_max_records",
        first_nested(config, ["llm_training.swapped_question_max_records"]),
        8,
    )
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
        False,
    )
    set_default(args, "local_soft_prompt_tokens", first_nested(config, ["adapter.local_soft_prompt_tokens"]), 8)
    set_default(args, "local_adapter_layers", first_nested(config, ["adapter.local_adapter_layers"]), 2)
    set_default(args, "local_text_encoder_layers", first_nested(config, ["adapter.local_text_encoder_layers"]), 2)
    set_default(
        args,
        "local_question_input_mode",
        first_nested(config, ["adapter.local_question_input_mode"]),
        "input_embeddings",
    )
    set_default(args, "local_context_layer", first_nested(config, ["adapter.local_context_layer"]), 2)
    set_default(
        args,
        "local_context_layers",
        value_to_csv(first_nested(config, ["adapter.local_context_layers"])),
        str(args.local_context_layer),
    )
    set_default(args, "local_fusion_mode", first_nested(config, ["adapter.local_fusion_mode"]), "text_latent_pool")
    set_default(args, "local_gate_init", first_nested(config, ["adapter.local_gate_init"]), 0.1)
    set_default(args, "local_text_gate_init", first_nested(config, ["adapter.local_text_gate_init"]), 0.05)
    set_default(args, "freeze_global_adapter", first_nested(config, ["adapter.freeze_global_adapter"]), True)
    set_default(args, "global_unfreeze_epoch", first_nested(config, ["adapter.global_unfreeze_epoch"]), 0)
    set_default(args, "global_lr", first_nested(config, ["adapter.global_lr"]), 1.0e-5)
    set_default(args, "global_prompt_dropout", first_nested(config, ["adapter.global_prompt_dropout"]), 0.0)
    set_default(
        args,
        "group_questions_by_state",
        first_nested(config, ["llm_training.group_questions_by_state"]),
        False,
    )
    set_default(
        args,
        "questions_per_state_group",
        first_nested(config, ["llm_training.questions_per_state_group"]),
        2,
    )
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
    set_default(
        args,
        "final_eval_baselines",
        value_to_csv(first_nested(config, ["llm_training.final_eval_baselines"])),
        args.eval_baselines,
    )
    set_default(args, "choice_score", first_nested(config, ["llm_training.choice_score"]), "mean")
    set_default(args, "choice_scoring_mode", first_nested(config, ["llm_training.choice_scoring_mode"]), "auto")
    set_default(args, "log_interval", first_nested(config, ["llm_training.log_interval"]), 20)
    set_default(args, "console_progress", first_nested(config, ["llm_training.console_progress"]), False)
    set_default(args, "save_step_metrics", first_nested(config, ["llm_training.save_step_metrics"]), False)
    set_default(args, "diagnostics_enabled", first_nested(config, ["llm_training.diagnostics.enabled"]), True)
    set_default(
        args,
        "diagnostics_every_epochs",
        first_nested(config, ["llm_training.diagnostics.every_epochs"]),
        1,
    )
    set_default(
        args,
        "diagnostics_records_per_task",
        first_nested(config, ["llm_training.diagnostics.records_per_task"]),
        1,
    )
    set_default(
        args,
        "diagnostics_generation_max_new_tokens",
        first_nested(config, ["llm_training.diagnostics.generation_max_new_tokens"]),
        8,
    )
    set_default(
        args,
        "diagnostics_layers",
        value_to_csv(first_nested(config, ["llm_training.diagnostics.layers"])),
        "0,2,8,14,-1",
    )
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
    set_default(args, "wandb_detailed_metrics", first_nested(config, ["wandb.detailed_metrics"]), False)
    require_args(args, ["qa_dir", "latent_dir", "model_name_or_path", "output_root"])
    positive_integer_settings = (
        "epochs",
        "batch_size",
        "eval_batch_size",
        "eval_choice_batch_size",
        "train_choice_batch_size",
        "train_grounding_batch_size",
        "gradient_accumulation_steps",
        "max_prompt_tokens",
        "max_target_tokens",
        "log_interval",
    )
    for setting in positive_integer_settings:
        if int(getattr(args, setting)) <= 0:
            raise ValueError(f"llm_training.{setting} must be positive.")
    supported_adapter_architectures = {
        "legacy",
        "alignment_qformer",
        "alignment_adapter",
        "hybrid_local_qformer",
        "residual_question_qformer",
        "residual_question_adapter",
    }
    if str(args.adapter_architecture) not in supported_adapter_architectures:
        raise ValueError(f"Unsupported adapter.architecture: {args.adapter_architecture}")
    if str(args.choice_scoring_mode) not in {"auto", "label", "sequence"}:
        raise ValueError(f"Unsupported llm_training.choice_scoring_mode: {args.choice_scoring_mode}")
    if str(args.adapter_architecture) in {
        "residual_question_qformer",
        "residual_question_adapter",
    } and str(args.adapter_init_checkpoint or "").strip().lower() in {"", "none", "null", "random"}:
        raise ValueError(
            f"adapter.architecture={args.adapter_architecture} requires adapter.init_checkpoint from stage 1."
        )
    supported_local_fusion_modes = {
        "text_latent_pool",
        "anchor_queries",
        "residual_qformer",
        "residual_spatial_transformer",
    }
    if str(args.local_fusion_mode) not in supported_local_fusion_modes:
        raise ValueError(f"Unsupported adapter.local_fusion_mode: {args.local_fusion_mode}")
    if int(args.initial_eval_records) < 0:
        raise ValueError("llm_training.initial_eval_records must be non-negative.")
    if int(args.latent_cache_size) < 0 or int(args.num_workers) < 0:
        raise ValueError("llm_training.latent_cache_size and num_workers must be non-negative.")
    for setting in (
        "ce_loss_weight",
        "choice_ce_loss_weight",
        "ranking_loss_weight",
        "ranking_loss_margin",
        "swapped_question_loss_weight",
        "swapped_question_loss_margin",
    ):
        if float(getattr(args, setting)) < 0.0:
            raise ValueError(f"llm_training.{setting} must be non-negative.")
    if not any(
        float(value) > 0.0
        for value in (
            args.ce_loss_weight,
            args.choice_ce_loss_weight,
            args.ranking_loss_weight,
            args.swapped_question_loss_weight,
        )
    ):
        raise ValueError("At least one training loss weight must be positive.")
    if not 0.0 <= float(args.dropout) < 1.0:
        raise ValueError("adapter.dropout must be in [0, 1).")
    if int(args.local_text_encoder_layers) < 0:
        raise ValueError("adapter.local_text_encoder_layers must be non-negative.")
    if int(args.local_context_layer) < 0:
        raise ValueError("adapter.local_context_layer must be non-negative.")
    local_context_layers = [int(value) for value in parse_csv(args.local_context_layers)]
    if not local_context_layers or any(value < 0 for value in local_context_layers):
        raise ValueError("adapter.local_context_layers must contain non-negative hidden-state indices.")
    args.local_context_layers = ",".join(str(value) for value in local_context_layers)
    if not 0.0 <= float(args.global_prompt_dropout) < 1.0:
        raise ValueError("adapter.global_prompt_dropout must be in [0, 1).")
    if int(args.questions_per_state_group) <= 0:
        raise ValueError("llm_training.questions_per_state_group must be positive.")
    if bool(args.group_questions_by_state) and int(args.questions_per_state_group) > int(args.batch_size):
        raise ValueError(
            "llm_training.questions_per_state_group cannot exceed llm_training.batch_size."
        )
    if float(args.swapped_question_loss_weight) > 0.0 and not bool(args.group_questions_by_state):
        raise ValueError("swapped_question_loss_weight requires llm_training.group_questions_by_state: true.")
    if float(args.swapped_question_loss_weight) > 0.0 and float(args.choice_ce_loss_weight) <= 0.0:
        raise ValueError("swapped_question_loss_weight requires a positive choice_ce_loss_weight.")
    if int(args.swapped_question_max_records) <= 0:
        raise ValueError("llm_training.swapped_question_max_records must be positive.")
    if not 0.0 <= float(args.warmup_ratio) < 1.0:
        raise ValueError("llm_training.warmup_ratio must be in [0, 1).")
    if not 0.0 <= float(args.min_lr_ratio) <= 1.0:
        raise ValueError("llm_training.min_lr_ratio must be in [0, 1].")
    if int(args.diagnostics_every_epochs) < 0 or int(args.diagnostics_records_per_task) <= 0:
        raise ValueError("Diagnostic cadence must be non-negative and records_per_task must be positive.")
    if int(args.diagnostics_generation_max_new_tokens) <= 0:
        raise ValueError("llm_training.diagnostics.generation_max_new_tokens must be positive.")
    if bool(args.diagnostics_enabled) and not parse_csv(args.diagnostics_layers):
        raise ValueError("llm_training.diagnostics.layers cannot be empty when diagnostics are enabled.")
    for setting in ("eval_baselines", "final_eval_baselines"):
        configured_baselines = parse_csv(getattr(args, setting))
        unknown_baselines = sorted(set(configured_baselines) - SUPPORTED_BASELINE_MODES)
        if unknown_baselines:
            raise ValueError(f"llm_training.{setting} contains unsupported modes: {unknown_baselines}")
        if "correct" not in configured_baselines:
            raise ValueError(f"llm_training.{setting} must include correct.")
    if str(args.checkpoint_metric) == "macro_latent_gain" and "shuffled" not in parse_csv(args.eval_baselines):
        raise ValueError("checkpoint_metric=macro_latent_gain requires shuffled in llm_training.eval_baselines.")
    if defaulted_loss_fields:
        print(
            "warning: config omits "
            f"{','.join(f'llm_training.{field}' for field in defaulted_loss_fields)}; "
            "using built-in training defaults."
        )
    return args


def parse_csv(raw: str | Sequence[str] | None) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, Sequence) and not isinstance(raw, str):
        return [str(part).strip() for part in raw if str(part).strip()]
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def validate_diagnostic_layers(llm, raw_layers: str | Sequence[str] | None) -> dict[str, Any]:
    requested = [int(value) for value in parse_csv(raw_layers)]
    decoder_layers = getattr(llm.config, "num_hidden_layers", None)
    if decoder_layers is None:
        return {
            "validated_before_training": False,
            "requested": requested,
            "reason": "model config does not expose num_hidden_layers",
        }
    hidden_state_count = int(decoder_layers) + 1
    invalid = [value for value in requested if not -hidden_state_count <= value < hidden_state_count]
    if invalid:
        raise ValueError(
            f"Diagnostic layers {invalid} are invalid for a model with {decoder_layers} decoder layers "
            f"and {hidden_state_count} returned hidden states."
        )
    return {
        "validated_before_training": True,
        "requested": requested,
        "resolved": sorted({value if value >= 0 else hidden_state_count + value for value in requested}),
        "hidden_state_count": hidden_state_count,
    }


def validate_local_context_layer(llm, layer_index: int) -> dict[str, Any]:
    decoder = _decoder_for_diagnostics(llm)
    layers = getattr(decoder, "layers", None)
    if not isinstance(layers, nn.ModuleList):
        raise ValueError("contextual_tokens requires a causal decoder with an exposed ModuleList named layers.")
    if not 0 <= int(layer_index) <= len(layers):
        raise ValueError(
            f"adapter.local_context_layer={layer_index} is invalid for a decoder with {len(layers)} layers."
        )
    return {
        "validated_before_training": True,
        "layer": int(layer_index),
        "decoder_layers": len(layers),
        "execution": "input_embeddings" if int(layer_index) == 0 else "early_exit_forward_hook",
    }


def gradient_l2_norm(parameters) -> float:
    norms = [parameter.grad.detach().norm(2).float() for parameter in parameters if parameter.grad is not None]
    if not norms:
        return 0.0
    return float(torch.linalg.vector_norm(torch.stack(norms)).cpu().item())


def adamw_parameter_groups(
    module: nn.Module,
    learning_rate: float,
    weight_decay: float,
    name: str,
    include_frozen: bool = False,
) -> list[dict[str, Any]]:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for parameter_name, parameter in module.named_parameters():
        if not parameter.requires_grad and not bool(include_frozen):
            continue
        if parameter.ndim < 2 or parameter_name.endswith("bias"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    groups: list[dict[str, Any]] = []
    if decay:
        groups.append(
            {
                "params": decay,
                "lr": float(learning_rate),
                "weight_decay": float(weight_decay),
                "name": f"{name}_decay",
            }
        )
    if no_decay:
        groups.append(
            {
                "params": no_decay,
                "lr": float(learning_rate),
                "weight_decay": 0.0,
                "name": f"{name}_no_decay",
            }
        )
    return groups


def optimizer_group_lr(optimizer: torch.optim.Optimizer, prefix: str, default: float = 0.0) -> float:
    for group in optimizer.param_groups:
        if str(group.get("name", "")).startswith(str(prefix)):
            return float(group["lr"])
    return float(default)


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str,
    total_updates: int,
    warmup_ratio: float,
    min_lr_ratio: float,
) -> tuple[torch.optim.lr_scheduler.LambdaLR, int]:
    total_updates = max(1, int(total_updates))
    warmup_updates = min(total_updates - 1, int(round(total_updates * float(warmup_ratio))))

    def lr_factor(step: int) -> float:
        if warmup_updates > 0 and int(step) < warmup_updates:
            return max(1.0 / warmup_updates, float(step + 1) / warmup_updates)
        if str(scheduler_name) == "constant":
            return 1.0
        decay_updates = max(1, total_updates - warmup_updates)
        progress = min(1.0, max(0.0, (float(step) - warmup_updates) / decay_updates))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(min_lr_ratio) + (1.0 - float(min_lr_ratio)) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor), warmup_updates


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


def set_frozen_llm_execution_mode(model: nn.Module, checkpoint_training: bool) -> None:
    checkpointing_active = bool(getattr(model, "is_gradient_checkpointing", False))
    if checkpoint_training and checkpointing_active:
        model.train()
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.eval()
    else:
        model.eval()


def load_tokenizer(args: argparse.Namespace):
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
    return tokenizer


def load_llm(args: argparse.Namespace, device: torch.device):
    model_dtype = resolve_model_dtype(str(args.torch_dtype), device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        dtype=model_dtype,
        trust_remote_code=bool(args.trust_remote_code),
    )
    model.to(device)
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if bool(args.llm_gradient_checkpointing):
        enable_checkpointing = getattr(model, "gradient_checkpointing_enable", None)
        if not callable(enable_checkpointing):
            raise ValueError("The selected causal LLM does not support gradient checkpointing.")
        try:
            enable_checkpointing(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            enable_checkpointing()
        if not bool(getattr(model, "is_gradient_checkpointing", False)):
            raise RuntimeError("The causal LLM did not report active gradient checkpointing.")
        # Transformers activates decoder checkpointing only in training mode. Qwen2.5 uses zero
        # dropout, but keep every Dropout module deterministic in case another compatible LLM does not.
        set_frozen_llm_execution_mode(model, checkpoint_training=True)
    else:
        set_frozen_llm_execution_mode(model, checkpoint_training=False)
    return model, model_dtype


def load_tokenizer_and_llm(args: argparse.Namespace, device: torch.device):
    tokenizer = load_tokenizer(args)
    model, model_dtype = load_llm(args, device)
    return tokenizer, model, model_dtype


def apply_runtime_environment(args: argparse.Namespace) -> None:
    if args.hf_home:
        os.environ.setdefault("HF_HOME", str(args.hf_home))
    if args.cache_dir:
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(args.cache_dir) / "hub"))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(args.cache_dir))
    if not bool(args.console_progress):
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        transformers_logging.disable_progress_bar()


def qa_path(qa_dir: str | Path, split: str) -> Path:
    path = Path(qa_dir) / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"QA split file not found: {path}")
    return path


def audit_qa_metadata(args: argparse.Namespace) -> dict[str, Any]:
    metadata_path = Path(args.qa_dir) / "metadata.json"
    if not metadata_path.exists():
        if bool(args.require_disjoint_splits):
            raise FileNotFoundError(
                f"Formal patch QA training requires metadata for provenance checks: {metadata_path}"
            )
        return {"available": False, "path": str(metadata_path), "evaluation_scope": "sanity_only"}
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, Mapping):
        raise ValueError(f"Expected a JSON object in {metadata_path}.")
    metadata_format = str(metadata.get("format", ""))
    prompt_contract = str(metadata.get("prompt_contract", ""))
    coordinate_origin = int(metadata.get("natural_language_coordinate_origin", -1))
    if bool(args.require_disjoint_splits) and (
        metadata_format != PATCH_QA_FORMAT
        or prompt_contract != PATCH_QA_PROMPT_CONTRACT
        or coordinate_origin != 1
    ):
        raise ValueError(
            "Formal patch QA training requires regenerated encoder-zscore, one-based natural-language prompts. "
            f"Observed format={metadata_format!r}, prompt_contract={prompt_contract!r}, "
            f"coordinate_origin={coordinate_origin}. Run scripts/build_tensor_patch_qa.py with the current code; "
            "matching latent files will be reused."
        )
    qa_fields = [str(field) for field in metadata.get("fields", [])]
    alignment_fields = [str(field) for field in metadata.get("alignment_fields", [])]
    allow_unseen_alignment_fields = bool(metadata.get("allow_unseen_alignment_fields", False))
    unseen_alignment_fields = sorted(set(qa_fields) - set(alignment_fields))
    if bool(args.require_disjoint_splits) and (
        not alignment_fields or (unseen_alignment_fields and not allow_unseen_alignment_fields)
    ):
        raise ValueError(
            "Formal patch QA metadata does not prove that stage 1 covered every QA field: "
            f"qa_fields={qa_fields}, alignment_fields={alignment_fields}, "
            f"unseen={unseen_alignment_fields}, allow_unseen={allow_unseen_alignment_fields}. "
            "Regenerate with a matching multi-field stage-1 checkpoint."
        )
    observed_alignment = metadata.get("alignment_checkpoint")
    configured_alignment = getattr(args, "qa_alignment_checkpoint", None)
    if observed_alignment and configured_alignment:
        observed_path = Path(str(observed_alignment)).expanduser().resolve()
        configured_path = Path(str(configured_alignment)).expanduser().resolve()
        if observed_path != configured_path:
            raise ValueError(
                "Patch QA metadata was generated with a different alignment checkpoint. "
                f"metadata={observed_path}, config={configured_path}."
            )
    adapter_init = str(args.adapter_init_checkpoint or "").strip()
    if configured_alignment and adapter_init and Path(adapter_init).name == "alignment_best.pt":
        if Path(adapter_init).expanduser().resolve() != Path(str(configured_alignment)).expanduser().resolve():
            raise ValueError(
                "adapter.init_checkpoint and patch_qa.alignment_checkpoint must match when initializing "
                "directly from alignment_best.pt."
            )
    split_mode = str(metadata.get("split_mode", "unknown"))
    if bool(args.require_disjoint_splits) and split_mode != "sample":
        raise ValueError(
            f"Formal patch QA training requires metadata split_mode=sample, got {split_mode!r}."
        )
    question_seed_mode = str(metadata.get("question_seed_mode", "legacy_record_order"))
    supported_seed_modes = {"sha256(seed|patch_id)", "sha256(seed|patch_id|variant)"}
    if bool(args.require_disjoint_splits) and question_seed_mode not in supported_seed_modes:
        raise ValueError(
            "Formal patch QA training requires independently seeded questions. Regenerate the QA JSONL with "
            "scripts/build_tensor_patch_qa.py; existing latent files will be reused."
        )
    question_variants = dict(metadata.get("question_variants", {}))
    if bool(args.group_questions_by_state):
        train_variants = int(question_variants.get(str(args.train_split), 1))
        if train_variants < int(args.questions_per_state_group):
            raise ValueError(
                "Grouped natural-language training requires at least "
                f"{int(args.questions_per_state_group)} train question variants per tensor/task, but QA metadata "
                f"reports {train_variants}. Run scripts/build_tensor_patch_qa.py with the current config first."
            )
    return {
        "available": True,
        "path": str(metadata_path),
        "format": metadata_format,
        "prompt_contract": prompt_contract,
        "natural_language_coordinate_origin": coordinate_origin,
        "alignment_checkpoint": str(observed_alignment or ""),
        "hdf5_path": str(metadata.get("hdf5_path", "")),
        "fields": qa_fields,
        "alignment_fields": alignment_fields,
        "allow_unseen_alignment_fields": allow_unseen_alignment_fields,
        "patch_size": int(metadata.get("patch_size", -1)),
        "split_mode": split_mode,
        "question_seed_mode": question_seed_mode,
        "question_variants": question_variants,
        "include_oracle_in_source": bool(metadata.get("include_oracle", False)),
    }


def task_specific_instruction(record: Mapping[str, Any]) -> str:
    task_type = str(record.get("task_type", "")).strip()
    if task_type == "normalized_point_value":
        return (
            "Rule: read the standardized value z directly at the requested patch-local row and column from "
            "the tensor soft tokens. "
            "Choose the closest numeric option and return only its label."
        )
    if task_type == "raw_point_value_with_stats":
        return (
            "Rule: read standardized z at the requested patch-local position from the tensor soft tokens, then "
            "recover the original value with x = mean + scale * z using the stated patch statistics. "
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
            "Rule: compare the standardized values at point A and point B using the tensor soft tokens; "
            "per-patch standardization preserves their original ordering. "
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
            "Rule: compare the standardized means in the two stated patch-local regions using the tensor soft "
            "tokens; per-patch standardization preserves their original ordering. "
            "Return A if region A has the greater or tied mean; otherwise return B."
        )
    if task_type == "extreme_quadrant":
        return (
            "Rule: locate the requested maximum or minimum in the standardized patch using the tensor soft "
            "tokens; per-patch standardization preserves extrema and their locations. "
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


def valid_choice_instruction(record: Mapping[str, Any]) -> str:
    choices = record.get("choices")
    if not isinstance(choices, Sequence) or isinstance(choices, str) or not choices:
        raise ValueError(f"Record has no valid choices: {record.get('qa_id', '<unknown>')}")
    labels = [str(choice) for choice in choices]
    return (
        f"Required output: exactly one of {', '.join(labels)}. "
        "Output only that label, with no explanation, punctuation, or other text."
    )


def build_prompt(record: Mapping[str, Any], prompt_template: str) -> str:
    query = str(record.get("query") or record.get("question") or "")
    choices = record.get("choices")
    if not isinstance(choices, Sequence) or isinstance(choices, str):
        choices = []
    choice_text = ", ".join(str(choice) for choice in choices)
    if prompt_template == "generic":
        return (
            "Tensor-state soft tokens are prepended before this text.\n"
            "Answer the tensor readout query using those tokens.\n\n"
            f"Query: {query}\n"
            f"Choices: {choice_text}\n"
            f"{valid_choice_instruction(record)}\n"
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
        f"{valid_choice_instruction(record)}\n"
        "Answer:"
    )


def build_local_conditioning_prompt(record: Mapping[str, Any], prompt_template: str) -> str:
    prompt = build_prompt(record, prompt_template=prompt_template)
    answer_anchor = "Answer:"
    if not prompt.endswith(answer_anchor):
        raise ValueError("The local conditioning prompt requires the main prompt to end with 'Answer:'.")
    return f"{prompt[: -len(answer_anchor)]}{LOCAL_QUESTION_ANCHOR_TEXT}"


def audit_prompt_tokenization(
    datasets: Mapping[str, TensorReadoutQADataset],
    tokenizer,
    max_prompt_tokens: int,
    prompt_template: str,
) -> dict[str, Any]:
    limit = int(max_prompt_tokens)
    summary: dict[str, Any] = {
        "max_prompt_tokens": limit,
        "prompt_template": str(prompt_template),
        "splits": {},
        "truncated_records": 0,
        "local_truncated_records": 0,
        "truncated_examples": [],
    }
    for split, dataset in datasets.items():
        split_total_tokens = 0
        split_max_tokens = 0
        split_truncated = 0
        split_local_max_tokens = 0
        split_local_truncated = 0
        task_stats: dict[str, dict[str, int]] = defaultdict(
            lambda: {"records": 0, "max_tokens": 0, "truncated": 0}
        )
        batch_size = 256
        for start in range(0, len(dataset.records), batch_size):
            records = dataset.records[start : start + batch_size]
            prompts: list[str] = []
            for record in records:
                query = str(record.get("query") or record.get("question") or "").strip()
                choices = record.get("choices")
                if not query:
                    raise ValueError(f"Prompt audit found an empty natural-language query: {record.get('qa_id')}")
                if not isinstance(choices, Sequence) or isinstance(choices, str) or not choices:
                    raise ValueError(f"Prompt audit found empty choices: {record.get('qa_id')}")
                prompt = build_prompt(record, prompt_template=prompt_template)
                expected_choices = "Choices: " + ", ".join(str(choice) for choice in choices)
                if query not in prompt or expected_choices not in prompt:
                    raise ValueError(
                        f"Prompt audit found a query/choice rendering mismatch: {record.get('qa_id')}"
                    )
                prompts.append(prompt)
            encoded = tokenizer(prompts, add_special_tokens=True, truncation=False)["input_ids"]
            local_encoded = tokenizer(
                [build_local_conditioning_prompt(record, prompt_template=prompt_template) for record in records],
                add_special_tokens=True,
                truncation=False,
            )["input_ids"]
            for record, token_ids, local_token_ids in zip(records, encoded, local_encoded):
                token_count = len(token_ids)
                local_token_count = len(local_token_ids)
                task = str(record.get("task_type", "unknown"))
                task_stat = task_stats[task]
                task_stat["records"] += 1
                task_stat["max_tokens"] = max(task_stat["max_tokens"], token_count)
                split_total_tokens += token_count
                split_max_tokens = max(split_max_tokens, token_count)
                split_local_max_tokens = max(split_local_max_tokens, local_token_count)
                if token_count > limit:
                    split_truncated += 1
                    task_stat["truncated"] += 1
                    if len(summary["truncated_examples"]) < 8:
                        summary["truncated_examples"].append(
                            {
                                "split": split,
                                "qa_id": str(record.get("qa_id", "")),
                                "task_type": task,
                                "tokens": token_count,
                            }
                        )
                if local_token_count > limit:
                    split_local_truncated += 1
                    if len(summary["truncated_examples"]) < 8:
                        summary["truncated_examples"].append(
                            {
                                "split": split,
                                "qa_id": str(record.get("qa_id", "")),
                                "task_type": task,
                                "tokens": local_token_count,
                                "path": "local_conditioning_prompt",
                            }
                        )
        summary["splits"][split] = {
            "records": len(dataset),
            "mean_tokens": split_total_tokens / max(1, len(dataset)),
            "max_tokens": split_max_tokens,
            "truncated_records": split_truncated,
            "local_max_tokens": split_local_max_tokens,
            "local_truncated_records": split_local_truncated,
            "by_task": dict(sorted(task_stats.items())),
        }
        summary["truncated_records"] += split_truncated
        summary["local_truncated_records"] += split_local_truncated
    summary["all_prompts_fit"] = (
        int(summary["truncated_records"]) == 0 and int(summary["local_truncated_records"]) == 0
    )
    return summary


def audit_choice_tokenization(
    datasets: Mapping[str, TensorReadoutQADataset],
    tokenizer,
) -> dict[str, Any]:
    labels = sorted(
        {
            str(choice)
            for dataset in datasets.values()
            for record in dataset.records
            for choice in (
                record.get("choices")
                if isinstance(record.get("choices"), Sequence)
                and not isinstance(record.get("choices"), str)
                else [str(record.get("answer", ""))]
            )
        }
    )
    token_ids: dict[str, list[int]] = {
        label: [
            int(value)
            for value in tokenizer(
                " " + label,
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]
        ]
        for label in labels
    }
    single_token_ids = [values[0] for values in token_ids.values() if len(values) == 1]
    all_single_token = all(len(values) == 1 for values in token_ids.values()) and len(
        set(single_token_ids)
    ) == len(labels)
    return {
        "labels": labels,
        "token_ids": token_ids,
        "all_labels_single_token": bool(all_single_token),
        "training_path": "single_forward_restricted_logits" if all_single_token else "sequence_likelihood_fallback",
        "evaluation_path": "prompt_only_restricted_logits" if all_single_token else "sequence_likelihood_fallback",
    }


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


@torch.no_grad()
def contextual_question_token_layers(
    llm,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_mask: torch.Tensor,
    layer_indices: Sequence[int],
) -> torch.Tensor:
    decoder = _decoder_for_diagnostics(llm)
    requested = tuple(int(value) for value in layer_indices)
    if not requested:
        raise ValueError("At least one local context layer is required.")
    layers = getattr(decoder, "layers", None)
    if not isinstance(layers, nn.ModuleList) or any(value < 0 or value > len(layers) for value in requested):
        count = len(layers) + 1 if isinstance(layers, nn.ModuleList) else 0
        raise ValueError(f"local_context_layers={requested} are invalid for {count} hidden-state tensors.")

    captured: dict[int, torch.Tensor] = {}
    if 0 in requested:
        captured[0] = llm.get_input_embeddings()(input_ids).detach()

    positive_layers = sorted({value for value in requested if value > 0})
    if positive_layers:
        class _ContextReady(RuntimeError):
            pass

        def capture_layer(layer_index: int, stop: bool):
            def hook(_module, _inputs, output) -> None:
                value = output[0] if isinstance(output, tuple) else output
                if not isinstance(value, torch.Tensor):
                    raise TypeError("The selected Qwen decoder layer did not return a hidden-state tensor.")
                captured[layer_index] = value.detach()
                if stop:
                    raise _ContextReady

            return hook

        handles = [
            layers[layer_index - 1].register_forward_hook(
                capture_layer(layer_index, layer_index == positive_layers[-1])
            )
            for layer_index in positive_layers
        ]
        try:
            try:
                decoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
            except _ContextReady:
                pass
        finally:
            for handle in handles:
                handle.remove()
    missing = [value for value in requested if value not in captured]
    if missing:
        raise RuntimeError(f"The shallow Qwen context hooks did not capture layers {missing}.")
    mask = prompt_mask.to(device=input_ids.device).unsqueeze(-1)
    return torch.stack(
        [captured[value] * mask.to(dtype=captured[value].dtype) for value in requested],
        dim=1,
    )


@torch.no_grad()
def contextual_question_tokens(
    llm,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_mask: torch.Tensor,
    layer_index: int,
) -> torch.Tensor:
    return contextual_question_token_layers(
        llm=llm,
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_mask=prompt_mask,
        layer_indices=[int(layer_index)],
    )[:, 0]


def contextual_question_tokens_for_adapter(
    llm,
    adapter: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_mask: torch.Tensor,
    fallback_layer: int,
) -> torch.Tensor:
    if isinstance(adapter, HybridGlobalLocalAdapter) and isinstance(
        adapter.local_adapter, ResidualQuestionConditionedAdapter
    ):
        return contextual_question_token_layers(
            llm=llm,
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_mask=prompt_mask,
            layer_indices=adapter.local_adapter.context_layers,
        )
    return contextual_question_tokens(
        llm=llm,
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_mask=prompt_mask,
        layer_index=int(fallback_layer),
    )


def build_local_question_tensors(
    records: Sequence[Mapping[str, Any]],
    tokenizer,
    device: torch.device,
    max_tokens: int,
    prompt_template: str = "task_specific",
) -> tuple[torch.Tensor, torch.Tensor]:
    prompts = []
    for record in records:
        question = str(record.get("query") or record.get("question") or "").strip()
        if not question:
            raise ValueError("The contextual local adapter received an empty natural-language question.")
        prompts.append(build_local_conditioning_prompt(record, prompt_template=prompt_template))
    encoded = tokenizer(prompts, padding=True, truncation=False, return_tensors="pt")
    if int(encoded["input_ids"].shape[1]) > int(max_tokens):
        raise ValueError(
            f"A local question uses {int(encoded['input_ids'].shape[1])} tokens, exceeding "
            f"max_prompt_tokens={max_tokens}."
        )
    return encoded["input_ids"].to(device), encoded["attention_mask"].to(device)


def contextual_adapter_question_context(
    llm,
    adapter: nn.Module,
    tokenizer,
    records: Sequence[Mapping[str, Any]],
    device: torch.device,
    max_prompt_tokens: int,
    layer_index: int,
    prompt_template: str,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not (
        isinstance(adapter, HybridGlobalLocalAdapter)
        and adapter.local_adapter.question_input_mode == "contextual_tokens"
    ):
        return None
    question_ids, question_mask = build_local_question_tensors(
        records=records,
        tokenizer=tokenizer,
        device=device,
        max_tokens=int(max_prompt_tokens),
        prompt_template=str(prompt_template),
    )
    question_embeds = contextual_question_tokens_for_adapter(
        llm=llm,
        adapter=adapter,
        input_ids=question_ids,
        attention_mask=question_mask,
        prompt_mask=question_mask.bool(),
        fallback_layer=int(layer_index),
    )
    return question_embeds, question_mask.bool()


def contextual_adapter_soft_embeds(
    llm,
    adapter: nn.Module,
    tokenizer,
    records: Sequence[Mapping[str, Any]],
    latent_map: torch.Tensor,
    device: torch.device,
    max_prompt_tokens: int,
    layer_index: int,
    mode: str,
    prompt_template: str,
    precomputed_question_context: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor | None:
    question_context = precomputed_question_context
    if question_context is None:
        question_context = contextual_adapter_question_context(
            llm=llm,
            adapter=adapter,
            tokenizer=tokenizer,
            records=records,
            device=device,
            max_prompt_tokens=int(max_prompt_tokens),
            layer_index=int(layer_index),
            prompt_template=str(prompt_template),
        )
    if question_context is None:
        return None
    question_embeds, question_mask = question_context
    return adapter_soft_embeds(
        adapter=adapter,
        latent_map=latent_map.to(device, non_blocking=True),
        text_embeds=question_embeds,
        question_embeds=question_embeds,
        question_mask=question_mask,
        records=records,
        mode=mode,
    )


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
    if mode in {"correct", "global_only", "local_only"}:
        if isinstance(adapter, HybridGlobalLocalAdapter):
            global_prompts, local_prompts, combined_prompts = adapter.forward_components(
                latent_map,
                question_embeds=question_embeds,
                question_mask=question_mask,
                structured_query=structured_query,
            )
            selected = {
                "correct": combined_prompts,
                "global_only": global_prompts,
                "local_only": local_prompts,
            }[mode]
            return selected.to(dtype=text_embeds.dtype)
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
    local_context_layer: int = 2,
    precomputed_question_context: tuple[torch.Tensor, torch.Tensor] | None = None,
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
    latent_map = latent_map.to(device, non_blocking=True)

    text_embeds = llm.get_input_embeddings()(input_ids)
    prompt_mask = text_labels.eq(IGNORE_INDEX) & text_attention_mask.bool()
    soft_embeds = contextual_adapter_soft_embeds(
        llm=llm,
        adapter=adapter,
        tokenizer=tokenizer,
        records=records,
        latent_map=latent_map,
        device=device,
        max_prompt_tokens=max_prompt_tokens,
        layer_index=int(local_context_layer),
        mode=soft_prompt_mode,
        prompt_template=str(prompt_template),
        precomputed_question_context=precomputed_question_context,
    )
    if soft_embeds is None:
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


def selective_answer_statistics(
    llm,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    return_first_logits: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
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

    first_logits: torch.Tensor | None = None
    if return_first_logits:
        first_positions = target_mask.long().argmax(dim=1)
        batch_positions = torch.arange(labels.shape[0], device=labels.device)
        first_hidden = shift_hidden[batch_positions, first_positions]
        first_logits = output_embeddings(first_hidden).float()

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
    return sequence_nll, target_counts, first_logits


def selective_answer_nll(
    llm,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    sequence_nll, target_counts, _first_logits = selective_answer_statistics(
        llm=llm,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        labels=labels,
        return_first_logits=False,
    )
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
    local_context_layer: int = 2,
    precomputed_soft_embeds: torch.Tensor | None = None,
    precomputed_question_context: tuple[torch.Tensor, torch.Tensor] | None = None,
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
    latent_map = latent_map.to(device, non_blocking=True)

    text_embeds = llm.get_input_embeddings()(input_ids)
    prompt_mask = text_labels.eq(IGNORE_INDEX) & text_attention_mask.bool()
    soft_embeds = precomputed_soft_embeds
    if soft_embeds is None:
        soft_embeds = contextual_adapter_soft_embeds(
            llm=llm,
            adapter=adapter,
            tokenizer=tokenizer,
            records=records,
            latent_map=latent_map,
            device=device,
            max_prompt_tokens=max_prompt_tokens,
            layer_index=int(local_context_layer),
            mode=soft_prompt_mode,
            prompt_template=str(prompt_template),
            precomputed_question_context=precomputed_question_context,
        )
    if soft_embeds is None:
        soft_embeds = adapter_soft_embeds(
            adapter,
            latent_map,
            text_embeds,
            question_embeds=text_embeds,
            question_mask=prompt_mask,
            records=records,
            mode=soft_prompt_mode,
        )
    soft_embeds = soft_embeds.to(device=device, dtype=text_embeds.dtype)
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


def single_token_choice_ids(
    records: Sequence[Mapping[str, Any]],
    tokenizer,
) -> tuple[list[list[int]], list[int]] | None:
    token_ids_by_record: list[list[int]] = []
    target_indices: list[int] = []
    for record in records:
        choices = record.get("choices")
        if not isinstance(choices, Sequence) or isinstance(choices, str) or not choices:
            choices = [str(record["answer"])]
        string_choices = [str(choice) for choice in choices]
        answer = str(record["answer"])
        if answer not in string_choices:
            string_choices = [answer] + string_choices
        encoded_choices: list[int] = []
        for choice in string_choices:
            encoded = tokenizer(
                " " + choice,
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]
            if len(encoded) != 1:
                return None
            encoded_choices.append(int(encoded[0]))
        if len(set(encoded_choices)) != len(encoded_choices):
            return None
        token_ids_by_record.append(encoded_choices)
        target_indices.append(string_choices.index(answer))
    return token_ids_by_record, target_indices


def single_token_choice_ce_loss(
    llm,
    adapter: TensorSoftPromptAdapter,
    tokenizer,
    records: Sequence[Mapping[str, Any]],
    latent_map: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
    soft_prompt_mode: str = "correct",
    choice_token_spec: tuple[list[list[int]], list[int]] | None = None,
    precomputed_question_context: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, dict[str, float]]:
    answers = [str(record["answer"]) for record in records]
    input_ids, text_attention_mask, text_labels = build_text_tensors(
        records=records,
        answers=answers,
        tokenizer=tokenizer,
        max_prompt_tokens=int(args.max_prompt_tokens),
        max_target_tokens=int(args.max_target_tokens),
        append_eos=bool(args.append_eos),
        prompt_template=str(args.prompt_template),
    )
    input_ids = input_ids.to(device)
    text_attention_mask = text_attention_mask.to(device)
    text_labels = text_labels.to(device)
    latent_map = latent_map.to(device, non_blocking=True)
    text_embeds = llm.get_input_embeddings()(input_ids)
    prompt_mask = text_labels.eq(IGNORE_INDEX) & text_attention_mask.bool()
    soft_embeds = contextual_adapter_soft_embeds(
        llm=llm,
        adapter=adapter,
        tokenizer=tokenizer,
        records=records,
        latent_map=latent_map,
        device=device,
        max_prompt_tokens=int(args.max_prompt_tokens),
        layer_index=int(args.local_context_layer),
        mode=soft_prompt_mode,
        prompt_template=str(args.prompt_template),
        precomputed_question_context=precomputed_question_context,
    )
    if soft_embeds is None:
        soft_embeds = adapter_soft_embeds(
            adapter=adapter,
            latent_map=latent_map,
            text_embeds=text_embeds,
            question_embeds=text_embeds,
            question_mask=prompt_mask,
            records=records,
            mode=soft_prompt_mode,
        )
    soft_embeds = soft_embeds.to(device=device, dtype=text_embeds.dtype)
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
    sequence_nll, target_counts, first_logits = selective_answer_statistics(
        llm=llm,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        labels=labels,
        return_first_logits=True,
    )
    if first_logits is None:
        raise RuntimeError("Single-token choice scoring did not produce first-answer logits.")
    if choice_token_spec is None:
        choice_token_spec = single_token_choice_ids(records, tokenizer)
    if choice_token_spec is None:
        raise ValueError("Single-token choice scoring received a non-single-token choice set.")
    token_ids_by_record, target_indices = choice_token_spec
    losses: list[torch.Tensor] = []
    correct_choice_nlls: list[torch.Tensor] = []
    hard_correct = 0
    for row, (candidate_ids, target_index) in enumerate(zip(token_ids_by_record, target_indices)):
        candidate_logits = first_logits[row, torch.tensor(candidate_ids, device=device)]
        losses.append(F.cross_entropy(candidate_logits.unsqueeze(0), torch.tensor([target_index], device=device)))
        correct_choice_nlls.append(
            sequence_nll[row] / target_counts[row].clamp_min(1)
            if str(args.choice_score) == "mean"
            else sequence_nll[row]
        )
        hard_correct += int(int(torch.argmax(candidate_logits.detach()).item()) == int(target_index))
    if not losses:
        raise ValueError("single_token_choice_ce_loss received an empty record batch.")
    return (
        torch.stack(losses).mean(),
        sequence_nll.sum() / target_counts.sum().clamp_min(1),
        torch.stack(correct_choice_nlls),
        soft_embeds,
        {
            "choice_accuracy": hard_correct / max(1, len(losses)),
            "choice_01_loss": 1.0 - hard_correct / max(1, len(losses)),
            "choice_single_token_path": 1.0,
        },
    )


def _sequence_choice_ce_loss(
    llm,
    adapter: TensorSoftPromptAdapter,
    tokenizer,
    records: Sequence[Mapping[str, Any]],
    latent_map: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
    soft_prompt_mode: str = "correct",
    precomputed_question_context: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, dict[str, float]]:
    candidate_records: list[Mapping[str, Any]] = []
    candidate_answers: list[str] = []
    candidate_latents: list[torch.Tensor] = []
    candidate_counts: list[int] = []
    target_indices: list[int] = []
    candidate_owners: list[int] = []
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
            candidate_owners.append(record_index)

    base_soft_embeds = contextual_adapter_soft_embeds(
        llm=llm,
        adapter=adapter,
        tokenizer=tokenizer,
        records=records,
        latent_map=latent_map,
        device=device,
        max_prompt_tokens=int(args.max_prompt_tokens),
        layer_index=int(args.local_context_layer),
        mode=soft_prompt_mode,
        prompt_template=str(args.prompt_template),
        precomputed_question_context=precomputed_question_context,
    )
    candidate_soft_embeds = (
        torch.stack([base_soft_embeds[index] for index in candidate_owners], dim=0)
        if base_soft_embeds is not None
        else None
    )

    nll_chunks: list[torch.Tensor] = []
    count_chunks: list[torch.Tensor] = []
    candidate_batch_size = max(1, int(args.train_choice_batch_size))
    for start in range(0, len(candidate_records), candidate_batch_size):
        end = min(len(candidate_records), start + candidate_batch_size)
        chunk_nll, chunk_counts = forward_answer_nll(
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
            soft_prompt_mode=soft_prompt_mode,
            reduction="sum",
            return_target_counts=True,
            local_context_layer=int(args.local_context_layer),
            precomputed_soft_embeds=(
                candidate_soft_embeds[start:end]
                if candidate_soft_embeds is not None
                else None
            ),
        )
        nll_chunks.append(chunk_nll)
        count_chunks.append(chunk_counts)
    flat_nll_sum = torch.cat(nll_chunks, dim=0)
    flat_target_counts = torch.cat(count_chunks, dim=0)
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
    return loss, correct_answer_ce, torch.stack(correct_choice_nlls), base_soft_embeds, {
        "choice_accuracy": float(accuracy),
        "choice_01_loss": float(1.0 - accuracy),
        "choice_single_token_path": 0.0,
    }


def choice_ce_loss(
    llm,
    adapter: TensorSoftPromptAdapter,
    tokenizer,
    records: Sequence[Mapping[str, Any]],
    latent_map: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
    soft_prompt_mode: str = "correct",
    precomputed_question_context: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, dict[str, float]]:
    choice_token_spec = single_token_choice_ids(records, tokenizer)
    scoring_mode = str(args.choice_scoring_mode)
    if scoring_mode == "sequence":
        choice_token_spec = None
    elif choice_token_spec is not None:
        return single_token_choice_ce_loss(
            llm=llm,
            adapter=adapter,
            tokenizer=tokenizer,
            records=records,
            latent_map=latent_map,
            device=device,
            args=args,
            soft_prompt_mode=soft_prompt_mode,
            choice_token_spec=choice_token_spec,
            precomputed_question_context=precomputed_question_context,
        )
    elif scoring_mode == "label":
        raise ValueError("choice_scoring_mode=label requires every choice label to tokenize as one unique token.")
    return _sequence_choice_ce_loss(
        llm=llm,
        adapter=adapter,
        tokenizer=tokenizer,
        records=records,
        latent_map=latent_map,
        device=device,
        args=args,
        soft_prompt_mode=soft_prompt_mode,
        precomputed_question_context=precomputed_question_context,
    )


def same_state_question_swap_indices(records: Sequence[Mapping[str, Any]]) -> tuple[list[int], list[int]]:
    grouped: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        key = (
            str(record.get("state_ref", "")),
            str(record.get("task_type", "")),
            str(record.get("field") or record.get("metadata", {}).get("field") or ""),
        )
        grouped[key].append(index)
    owners: list[int] = []
    swapped: list[int] = []
    for indices in grouped.values():
        distinct = [
            index
            for index in indices
            if any(
                str(records[index].get("query") or records[index].get("question") or "")
                != str(records[other].get("query") or records[other].get("question") or "")
                for other in indices
            )
        ]
        if len(distinct) < 2:
            continue
        for position, owner in enumerate(distinct):
            candidate = distinct[(position + 1) % len(distinct)]
            if str(records[owner].get("query") or records[owner].get("question") or "") == str(
                records[candidate].get("query") or records[candidate].get("question") or ""
            ):
                candidate = next(
                    other
                    for other in distinct
                    if str(records[owner].get("query") or records[owner].get("question") or "")
                    != str(records[other].get("query") or records[other].get("question") or "")
                )
            owners.append(owner)
            swapped.append(candidate)
    return owners, swapped


def swapped_question_grounding_loss(
    llm,
    adapter: nn.Module,
    tokenizer,
    records: Sequence[Mapping[str, Any]],
    latent_map: torch.Tensor,
    positive_nll: torch.Tensor,
    soft_embeds: torch.Tensor | None,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    owners, swapped = same_state_question_swap_indices(records)
    max_records = int(args.swapped_question_max_records)
    owners = owners[:max_records]
    swapped = swapped[:max_records]
    if not owners or soft_embeds is None:
        zero = positive_nll.new_zeros(())
        return zero, {"swapped_question_pairs": 0.0, "swapped_question_margin_mean": 0.0}
    swapped_soft = torch.stack([soft_embeds[index] for index in swapped], dim=0)
    selected_records = [records[index] for index in owners]
    selected_answers = [str(record["answer"]) for record in selected_records]
    swapped_nll = forward_answer_nll(
        llm=llm,
        adapter=adapter,
        tokenizer=tokenizer,
        records=selected_records,
        answers=selected_answers,
        latent_map=torch.stack([latent_map[index] for index in owners], dim=0),
        device=device,
        max_prompt_tokens=int(args.max_prompt_tokens),
        max_target_tokens=int(args.max_target_tokens),
        append_eos=bool(args.append_eos),
        prompt_template=str(args.prompt_template),
        soft_prompt_mode="correct",
        reduction=str(args.choice_score),
        local_context_layer=int(args.local_context_layer),
        precomputed_soft_embeds=swapped_soft,
    )
    selected_positive = torch.stack([positive_nll[index] for index in owners])
    margin = swapped_nll - selected_positive
    loss = F.relu(float(args.swapped_question_loss_margin) - margin).mean()
    return loss, {
        "swapped_question_pairs": float(len(owners)),
        "swapped_question_margin_mean": float(margin.detach().mean().cpu().item()),
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
    question_context = contextual_adapter_question_context(
        llm=llm,
        adapter=adapter,
        tokenizer=tokenizer,
        records=records,
        device=device,
        max_prompt_tokens=int(args.max_prompt_tokens),
        layer_index=int(args.local_context_layer),
        prompt_template=str(args.prompt_template),
    )
    ce_weight = float(args.ce_loss_weight)
    choice_ce_weight = float(args.choice_ce_loss_weight)
    ranking_weight = float(args.ranking_loss_weight)
    swapped_weight = float(args.swapped_question_loss_weight)
    choice_metrics = {
        "choice_accuracy": 0.0,
        "choice_01_loss": 0.0,
        "choice_single_token_path": 0.0,
    }
    if choice_ce_weight > 0.0:
        choice_loss_value, ce_loss, positive_nll, positive_soft_embeds, choice_metrics = choice_ce_loss(
            llm=llm,
            adapter=adapter,
            tokenizer=tokenizer,
            records=records,
            latent_map=batch["latent_map"],
            device=device,
            args=args,
            soft_prompt_mode="correct",
            precomputed_question_context=question_context,
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
            local_context_layer=int(args.local_context_layer),
            precomputed_question_context=question_context,
        )
        choice_loss_value = ce_loss.new_zeros(())
        positive_nll = None
        positive_soft_embeds = None

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
            local_context_layer=int(args.local_context_layer),
            precomputed_question_context=question_context,
        )
    ranking_loss = positive_nll.new_zeros(())
    ranking_margin_mean = 0.0
    swapped_loss = positive_nll.new_zeros(())
    swapped_metrics = {"swapped_question_pairs": 0.0, "swapped_question_margin_mean": 0.0}
    combined_records: list[Mapping[str, Any]] = []
    combined_answers: list[str] = []
    combined_latents: list[torch.Tensor] = []
    combined_soft_embeds: list[torch.Tensor] = []
    ranking_count = 0
    negative_mode = str(args.ranking_loss_negative)
    if ranking_weight > 0.0:
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
        negative_soft_embeds = contextual_adapter_soft_embeds(
            llm=llm,
            adapter=adapter,
            tokenizer=tokenizer,
            records=records,
            latent_map=negative_latents,
            device=device,
            max_prompt_tokens=int(args.max_prompt_tokens),
            layer_index=int(args.local_context_layer),
            mode=negative_mode,
            prompt_template=str(args.prompt_template),
            precomputed_question_context=question_context,
        )
        if negative_soft_embeds is None:
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
                local_context_layer=int(args.local_context_layer),
                precomputed_question_context=question_context,
            )
            ranking_terms = F.relu(float(args.ranking_loss_margin) + positive_nll - negative_nll)
            ranking_loss = ranking_terms.mean()
            ranking_margin_mean = float(
                (negative_nll.detach() - positive_nll.detach()).mean().cpu().item()
            )
        else:
            ranking_count = len(records)
            combined_records.extend(records)
            combined_answers.extend(answers)
            combined_latents.extend(negative_latents)
            combined_soft_embeds.extend(negative_soft_embeds)

    swap_owners: list[int] = []
    if swapped_weight > 0.0 and positive_soft_embeds is not None:
        swap_owners, swap_sources = same_state_question_swap_indices(records)
        max_swap_records = int(args.swapped_question_max_records)
        swap_owners = swap_owners[:max_swap_records]
        swap_sources = swap_sources[:max_swap_records]
        for owner, source in zip(swap_owners, swap_sources):
            combined_records.append(records[owner])
            combined_answers.append(str(records[owner]["answer"]))
            combined_latents.append(batch["latent_map"][owner])
            combined_soft_embeds.append(positive_soft_embeds[source])

    if combined_records:
        combined_nll_chunks: list[torch.Tensor] = []
        grounding_batch_size = max(1, int(args.train_grounding_batch_size))
        for start in range(0, len(combined_records), grounding_batch_size):
            end = min(len(combined_records), start + grounding_batch_size)
            combined_nll_chunks.append(
                forward_answer_nll(
                    llm=llm,
                    adapter=adapter,
                    tokenizer=tokenizer,
                    records=combined_records[start:end],
                    answers=combined_answers[start:end],
                    latent_map=torch.stack(combined_latents[start:end], dim=0),
                    device=device,
                    max_prompt_tokens=int(args.max_prompt_tokens),
                    max_target_tokens=int(args.max_target_tokens),
                    append_eos=bool(args.append_eos),
                    prompt_template=str(args.prompt_template),
                    soft_prompt_mode="correct",
                    reduction=str(args.choice_score),
                    local_context_layer=int(args.local_context_layer),
                    precomputed_soft_embeds=torch.stack(combined_soft_embeds[start:end], dim=0),
                )
            )
        combined_nll = torch.cat(combined_nll_chunks, dim=0)
        if ranking_count > 0:
            negative_nll = combined_nll[:ranking_count]
            ranking_terms = F.relu(float(args.ranking_loss_margin) + positive_nll - negative_nll)
            ranking_loss = ranking_terms.mean()
            ranking_margin_mean = float(
                (negative_nll.detach() - positive_nll.detach()).mean().cpu().item()
            )
        if swap_owners:
            swapped_nll = combined_nll[ranking_count : ranking_count + len(swap_owners)]
            selected_positive = torch.stack([positive_nll[index] for index in swap_owners])
            swap_margin = swapped_nll - selected_positive
            swapped_loss = F.relu(float(args.swapped_question_loss_margin) - swap_margin).mean()
            swapped_metrics = {
                "swapped_question_pairs": float(len(swap_owners)),
                "swapped_question_margin_mean": float(swap_margin.detach().mean().cpu().item()),
            }
    elif swapped_weight > 0.0:
        swapped_loss, swapped_metrics = swapped_question_grounding_loss(
            llm=llm,
            adapter=adapter,
            tokenizer=tokenizer,
            records=records,
            latent_map=batch["latent_map"],
            positive_nll=positive_nll,
            soft_embeds=positive_soft_embeds,
            device=device,
            args=args,
        )
    weighted_ce_loss = ce_weight * ce_loss
    weighted_choice_ce_loss = choice_ce_weight * choice_loss_value
    weighted_ranking_loss = ranking_weight * ranking_loss
    weighted_swapped_loss = swapped_weight * swapped_loss
    total_loss = weighted_ce_loss + weighted_choice_ce_loss + weighted_ranking_loss + weighted_swapped_loss
    return total_loss, {
        "loss": float(total_loss.detach().cpu().item()),
        "ce_loss": float(ce_loss.detach().cpu().item()),
        "weighted_ce_loss": float(weighted_ce_loss.detach().cpu().item()),
        "choice_ce_loss": float(choice_loss_value.detach().cpu().item()),
        "weighted_choice_ce_loss": float(weighted_choice_ce_loss.detach().cpu().item()),
        "choice_accuracy": float(choice_metrics["choice_accuracy"]),
        "choice_01_loss": float(choice_metrics["choice_01_loss"]),
        "choice_single_token_path": float(choice_metrics.get("choice_single_token_path", 0.0)),
        "ranking_loss": float(ranking_loss.detach().cpu().item()),
        "weighted_ranking_loss": float(weighted_ranking_loss.detach().cpu().item()),
        "ranking_margin_mean": ranking_margin_mean,
        "swapped_question_loss": float(swapped_loss.detach().cpu().item()),
        "weighted_swapped_question_loss": float(weighted_swapped_loss.detach().cpu().item()),
        **swapped_metrics,
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
    local_context_layer: int = 2,
    precomputed_soft_embeds: torch.Tensor | None = None,
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
    latent_map = latent_map.to(device, non_blocking=True)

    text_embeds = llm.get_input_embeddings()(input_ids)
    prompt_mask = text_labels.eq(IGNORE_INDEX) & text_attention_mask.bool()
    soft_embeds = precomputed_soft_embeds
    if soft_embeds is None:
        soft_embeds = contextual_adapter_soft_embeds(
            llm=llm,
            adapter=adapter,
            tokenizer=tokenizer,
            records=records,
            latent_map=latent_map,
            device=device,
            max_prompt_tokens=max_prompt_tokens,
            layer_index=int(local_context_layer),
            mode=soft_prompt_mode,
            prompt_template=str(prompt_template),
        )
    if soft_embeds is None:
        soft_embeds = adapter_soft_embeds(
            adapter,
            latent_map,
            text_embeds,
            question_embeds=text_embeds,
            question_mask=prompt_mask,
            records=records,
            mode=soft_prompt_mode,
        )
    soft_embeds = soft_embeds.to(device=device, dtype=text_embeds.dtype)
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


def parse_generated_choice(text: str, choices: Sequence[str]) -> dict[str, Any]:
    raw = str(text)
    stripped = raw.strip()
    labels = [str(choice) for choice in choices]
    if not labels:
        return {
            "raw_output": raw,
            "stripped_output": stripped,
            "parsed_choice": None,
            "format_valid": False,
            "contains_single_valid_choice": False,
            "matched_choices": [],
        }
    exact = stripped in labels
    label_pattern = "|".join(re.escape(label) for label in sorted(labels, key=len, reverse=True))
    matches = re.findall(rf"(?<![A-Za-z0-9_])(?:{label_pattern})(?![A-Za-z0-9_])", stripped)
    unique_matches = list(dict.fromkeys(matches))
    parsed = stripped if exact else unique_matches[0] if len(unique_matches) == 1 else None
    return {
        "raw_output": raw,
        "stripped_output": stripped,
        "parsed_choice": parsed,
        "format_valid": bool(exact),
        "contains_single_valid_choice": len(unique_matches) == 1,
        "matched_choices": unique_matches,
    }


@torch.no_grad()
def generate_diagnostic_answer(
    llm,
    tokenizer,
    record: Mapping[str, Any],
    soft_embeds: torch.Tensor,
    device: torch.device,
    prompt_template: str,
    max_prompt_tokens: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    prompt = build_prompt(record, prompt_template=prompt_template)
    prompt_ids = tokenizer(prompt, add_special_tokens=True, truncation=False)["input_ids"]
    if len(prompt_ids) > int(max_prompt_tokens):
        prompt_ids = prompt_ids[-int(max_prompt_tokens) :]
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    text_embeds = llm.get_input_embeddings()(input_ids)
    soft_embeds = soft_embeds.to(device=device, dtype=text_embeds.dtype)
    inputs_embeds = torch.cat([soft_embeds, text_embeds], dim=1)
    attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)
    outputs = llm(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    )
    generated_ids: list[int] = []
    past_key_values = outputs.past_key_values
    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
    eos_id = int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else None
    for _ in range(max(1, int(max_new_tokens))):
        token_id = int(next_token.item())
        generated_ids.append(token_id)
        if eos_id is not None and token_id == eos_id:
            break
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=device)], dim=1
        )
        outputs = llm(
            input_ids=next_token,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    parsed = parse_generated_choice(text, [str(value) for value in record.get("choices", [])])
    parsed["generated_token_ids"] = generated_ids
    return parsed


@torch.no_grad()
def prompt_choice_label_logits(
    llm,
    adapter: TensorSoftPromptAdapter,
    tokenizer,
    records: Sequence[Mapping[str, Any]],
    latent_map: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
    mode: str,
) -> torch.Tensor:
    prompts = [build_prompt(record, prompt_template=str(args.prompt_template)) for record in records]
    encoded = tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
        add_special_tokens=True,
    )
    if int(encoded["input_ids"].shape[1]) > int(args.max_prompt_tokens):
        raise ValueError(
            f"A choice prompt uses {int(encoded['input_ids'].shape[1])} tokens, exceeding "
            f"max_prompt_tokens={int(args.max_prompt_tokens)}."
        )
    input_ids = encoded["input_ids"].to(device)
    text_attention_mask = encoded["attention_mask"].to(device)
    text_embeds = llm.get_input_embeddings()(input_ids)
    prompt_mask = text_attention_mask.bool()
    soft_embeds = contextual_adapter_soft_embeds(
        llm=llm,
        adapter=adapter,
        tokenizer=tokenizer,
        records=records,
        latent_map=latent_map.to(device, non_blocking=True),
        device=device,
        max_prompt_tokens=int(args.max_prompt_tokens),
        layer_index=int(args.local_context_layer),
        mode=mode,
        prompt_template=str(args.prompt_template),
    )
    if soft_embeds is None:
        soft_embeds = adapter_soft_embeds(
            adapter=adapter,
            latent_map=latent_map.to(device, non_blocking=True),
            text_embeds=text_embeds,
            question_embeds=text_embeds,
            question_mask=prompt_mask,
            records=records,
            mode=mode,
        )
    soft_embeds = soft_embeds.to(device=device, dtype=text_embeds.dtype)
    inputs_embeds = torch.cat([soft_embeds, text_embeds], dim=1)
    soft_attention = torch.ones(
        (input_ids.shape[0], soft_embeds.shape[1]),
        dtype=text_attention_mask.dtype,
        device=device,
    )
    attention_mask = torch.cat([soft_attention, text_attention_mask], dim=1)
    decoder = _decoder_for_diagnostics(llm)
    output_embeddings = llm.get_output_embeddings()
    if output_embeddings is None:
        raise ValueError("The causal LLM does not expose output embeddings for choice scoring.")
    outputs = decoder(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    last_indices = last_nonpadding_indices(text_attention_mask) + int(soft_embeds.shape[1])
    batch_indices = torch.arange(input_ids.shape[0], device=device)
    next_hidden = outputs.last_hidden_state[batch_indices, last_indices]
    return output_embeddings(next_hidden).float()


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
    single_token_spec = single_token_choice_ids(records, tokenizer)
    if str(args.choice_scoring_mode) == "sequence":
        single_token_spec = None
    elif single_token_spec is None and str(args.choice_scoring_mode) == "label":
        raise ValueError("choice_scoring_mode=label requires every choice label to tokenize as one unique token.")
    if single_token_spec is not None:
        token_ids_by_record, _target_indices = single_token_spec
        logits = prompt_choice_label_logits(
            llm=llm,
            adapter=adapter,
            tokenizer=tokenizer,
            records=records,
            latent_map=latent_map,
            device=device,
            args=args,
            mode=mode,
        )
        predictions: list[str] = []
        for row, record in enumerate(records):
            choices = record.get("choices")
            if not isinstance(choices, Sequence) or isinstance(choices, str) or not choices:
                choices = [str(record["answer"])]
            string_choices = [str(choice) for choice in choices]
            answer = str(record["answer"])
            if answer not in string_choices:
                string_choices = [answer] + string_choices
            candidate_logits = logits[row, torch.tensor(token_ids_by_record[row], device=device)]
            predictions.append(string_choices[int(torch.argmax(candidate_logits).item())])
        return predictions

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

    base_soft_embeds = contextual_adapter_soft_embeds(
        llm=llm,
        adapter=adapter,
        tokenizer=tokenizer,
        records=records,
        latent_map=latent_map,
        device=device,
        max_prompt_tokens=int(args.max_prompt_tokens),
        layer_index=int(args.local_context_layer),
        mode=mode,
        prompt_template=str(args.prompt_template),
    )
    candidate_soft_embeds = (
        [base_soft_embeds[index] for index in candidate_owner] if base_soft_embeds is not None else None
    )

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
            local_context_layer=int(args.local_context_layer),
            precomputed_soft_embeds=(
                torch.stack(candidate_soft_embeds[start:end], dim=0)
                if candidate_soft_embeds is not None
                else None
            ),
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
    if mode in {"correct", "global_only", "local_only", "no_latent", "shuffled_stats"}:
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
        patch_size = int(prompt_data.get("patch_size", 16))
        mean = float(shuffled_data.get("mean", prompt_data["mean"]))
        scale = float(shuffled_data.get("scale", shuffled_data.get("std", prompt_data.get("scale", prompt_data["std"]))))
        question = (
            f"The tensor soft tokens encode the per-patch standardized {patch_size} by {patch_size} matrix z "
            f"of {prompt_data['field']}. Recover an original value with x = mean + scale * z, "
            f"where mean is {mean:.{digits}g} and scale is {scale:.{digits}g}. "
            "Which option is closest to the "
            f"original value x at row {int(prompt_data['row'])}, column {int(prompt_data['col'])}? "
            f"Options: {prompt_data['option_text']}."
        )
        changed = dict(record)
        changed["question"] = question
        changed["query"] = question
        changed_prompt_data = dict(prompt_data)
        changed_prompt_data["mean"] = mean
        changed_prompt_data["scale"] = scale
        changed["prompt_data"] = changed_prompt_data
        updated.append(changed)
    return updated


def aggregate_evaluation_counts(
    *,
    total: int,
    correct: int,
    task_total: Mapping[str, int],
    task_correct: Mapping[str, int],
    field_total: Mapping[str, int],
    field_correct: Mapping[str, int],
    task_field_total: Mapping[str, int],
    task_field_correct: Mapping[str, int],
) -> tuple[
    int,
    int,
    dict[str, int],
    dict[str, int],
    dict[str, int],
    dict[str, int],
    dict[str, int],
    dict[str, int],
]:
    local_payload = {
        "total": int(total),
        "correct": int(correct),
        "task_total": dict(task_total),
        "task_correct": dict(task_correct),
        "field_total": dict(field_total),
        "field_correct": dict(field_correct),
        "task_field_total": dict(task_field_total),
        "task_field_correct": dict(task_field_correct),
    }
    gathered: list[Mapping[str, Any] | None]
    if distributed_is_initialized():
        gathered = [None] * distributed_world_size()
        dist.all_gather_object(gathered, local_payload)
    else:
        gathered = [local_payload]

    merged_total = 0
    merged_correct = 0
    merged_maps = {
        "task_total": defaultdict(int),
        "task_correct": defaultdict(int),
        "field_total": defaultdict(int),
        "field_correct": defaultdict(int),
        "task_field_total": defaultdict(int),
        "task_field_correct": defaultdict(int),
    }
    for payload in gathered:
        if payload is None:
            continue
        merged_total += int(payload["total"])
        merged_correct += int(payload["correct"])
        for name, target in merged_maps.items():
            values = payload.get(name, {})
            if not isinstance(values, Mapping):
                continue
            for key, value in values.items():
                target[str(key)] += int(value)
    return (
        merged_total,
        merged_correct,
        dict(merged_maps["task_total"]),
        dict(merged_maps["task_correct"]),
        dict(merged_maps["field_total"]),
        dict(merged_maps["field_correct"]),
        dict(merged_maps["task_field_total"]),
        dict(merged_maps["task_field_correct"]),
    )


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
    llm_was_training = bool(llm.training)
    llm.eval()
    eval_sampler = (
        ExactDistributedEvalSampler(
            dataset,
            rank=distributed_rank(),
            num_replicas=distributed_world_size(),
        )
        if distributed_is_initialized()
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(args.eval_batch_size)),
        shuffle=False,
        sampler=eval_sampler,
        num_workers=int(args.num_workers),
        persistent_workers=int(args.num_workers) > 0,
        pin_memory=device.type == "cuda",
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
        for batch in tqdm(
            loader,
            desc=f"Eval [{mode}]",
            leave=False,
            disable=not bool(args.console_progress),
        ):
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
        (
            total,
            correct,
            task_total,
            task_correct,
            field_total,
            field_correct,
            task_field_total,
            task_field_correct,
        ) = aggregate_evaluation_counts(
            total=total,
            correct=correct,
            task_total=task_total,
            task_correct=task_correct,
            field_total=field_total,
            field_correct=field_correct,
            task_field_total=task_field_total,
            task_field_correct=task_field_correct,
        )
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
    if llm_was_training:
        set_frozen_llm_execution_mode(llm, checkpoint_training=True)
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


def _cosine_and_relative_l2(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    left = left.float().reshape(-1)
    right = right.float().reshape(-1)
    return {
        "cosine": float(F.cosine_similarity(left, right, dim=0).cpu().item()),
        "relative_l2": float(((left - right).norm() / left.norm().clamp_min(1.0e-8)).cpu().item()),
    }


def _rms(value: torch.Tensor) -> float:
    return float(value.float().square().mean().sqrt().cpu().item())


def _mean_off_diagonal_cosine(tokens: torch.Tensor) -> float:
    normalized = F.normalize(tokens.float(), dim=-1)
    similarities = normalized @ normalized.transpose(-1, -2)
    count = int(similarities.shape[-1])
    if count <= 1:
        return 1.0
    mask = ~torch.eye(count, dtype=torch.bool, device=similarities.device)
    return float(similarities[..., mask].mean().cpu().item())


def _diagnostic_records(dataset: TensorReadoutQADataset, records_per_task: int) -> list[int]:
    selected: list[int] = []
    counts: dict[str, int] = defaultdict(int)
    used_states: set[str] = set()
    for index, record in enumerate(dataset.records):
        task = str(record.get("task_type", "unknown"))
        state_ref = str(record.get("state_ref", ""))
        if counts[task] >= records_per_task or state_ref in used_states:
            continue
        selected.append(index)
        counts[task] += 1
        used_states.add(state_ref)
    if any(count < records_per_task for count in counts.values()):
        for index, record in enumerate(dataset.records):
            task = str(record.get("task_type", "unknown"))
            if counts[task] >= records_per_task or index in selected:
                continue
            selected.append(index)
            counts[task] += 1
    return selected


def _alternate_question_record(dataset: TensorReadoutQADataset, index: int) -> Mapping[str, Any] | None:
    source = dataset.records[index]
    state_ref = str(source.get("state_ref", ""))
    task = str(source.get("task_type", ""))
    for record in dataset.records:
        if str(record.get("state_ref", "")) == state_ref and str(record.get("task_type", "")) != task:
            return record
    return None


def _same_task_alternate_question_record(
    dataset: TensorReadoutQADataset,
    index: int,
) -> Mapping[str, Any] | None:
    source = dataset.records[index]
    task = str(source.get("task_type", ""))
    field = str(source.get("field", ""))
    state_ref = str(source.get("state_ref", ""))
    source_query = str(source.get("query") or source.get("question") or "")
    for record in dataset.records:
        if (
            str(record.get("state_ref", "")) == state_ref
            and str(record.get("task_type", "")) == task
            and str(record.get("field", "")) == field
            and str(record.get("query") or record.get("question") or "") != source_query
        ):
            return record
    for record in dataset.records:
        if (
            str(record.get("task_type", "")) == task
            and str(record.get("field", "")) == field
            and str(record.get("query") or record.get("question") or "") != source_query
        ):
            return record
    return None


def _decoder_for_diagnostics(llm) -> nn.Module:
    get_decoder = getattr(llm, "get_decoder", None)
    decoder = get_decoder() if callable(get_decoder) else None
    if decoder is None or decoder is llm:
        prefix = str(getattr(llm, "base_model_prefix", ""))
        decoder = getattr(llm, prefix, None) if prefix else None
    if decoder is None or decoder is llm:
        raise ValueError("The causal LLM does not expose decoder hidden states for diagnostics.")
    return decoder


def _adapter_forward_with_trace(
    adapter: nn.Module,
    latent: torch.Tensor,
    text_embeds: torch.Tensor,
    prompt_mask: torch.Tensor,
    structured_query: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    trace: dict[str, torch.Tensor] = {}
    handles: list[Any] = []
    input_mask = prompt_mask[0].to(device=text_embeds.device, dtype=text_embeds.dtype).unsqueeze(-1)
    trace_input = text_embeds[0].mean(dim=0) if text_embeds.ndim == 4 else text_embeds[0]
    trace["llm_input_embedding_mean"] = (
        (trace_input * input_mask).sum(dim=0) / input_mask.sum().clamp_min(1.0)
    ).detach().float().cpu()

    def capture(name: str, pool_text: bool = False):
        def hook(_module, _inputs, output) -> None:
            value = output[0] if isinstance(output, tuple) else output
            if not isinstance(value, torch.Tensor):
                return
            value = value[0]
            if pool_text:
                mask = prompt_mask[0].to(device=value.device, dtype=value.dtype).unsqueeze(-1)
                value = (value * mask).sum(dim=0) / mask.sum().clamp_min(1.0)
            trace[name] = value.detach().float().cpu()

        return hook

    if isinstance(adapter, HybridGlobalLocalAdapter):
        local_query_tokens = getattr(adapter.local_adapter, "query_tokens", None)
        global_query_tokens = getattr(adapter.global_adapter, "query_tokens", None)
        if isinstance(local_query_tokens, torch.Tensor):
            trace["local_query_tokens"] = local_query_tokens[0].detach().float().cpu()
        if isinstance(global_query_tokens, torch.Tensor):
            trace["global_query_tokens"] = global_query_tokens[0].detach().float().cpu()
        anchor_projection = getattr(adapter.local_adapter, "anchor_projection", None)
        if anchor_projection is not None:
            anchor_indices = last_nonpadding_indices(prompt_mask.to(device=text_embeds.device))
            trace["local_anchor_hidden"] = text_embeds[
                torch.arange(text_embeds.shape[0], device=text_embeds.device),
                anchor_indices,
            ][0].detach().float().cpu()
        text_projections = getattr(adapter.local_adapter, "text_projections", None)
        if text_projections is not None:
            for index, module in enumerate(text_projections):
                handles.append(module.register_forward_hook(capture(f"local_text_projected_layer_{index}", pool_text=True)))
        elif hasattr(adapter.local_adapter, "text_projection"):
            handles.append(
                adapter.local_adapter.text_projection.register_forward_hook(
                    capture("local_text_projected", pool_text=True)
                )
            )
        local_backbone = getattr(adapter.local_adapter, "backbone", adapter.local_adapter)
        if hasattr(local_backbone, "latent_projection"):
            handles.append(local_backbone.latent_projection.register_forward_hook(capture("local_latent_projected")))
        for index, module in enumerate(getattr(adapter.local_adapter, "text_encoder", [])):
            handles.append(module.register_forward_hook(capture(f"local_text_encoder_{index}", pool_text=True)))
        for index, module in enumerate(getattr(adapter.local_adapter, "text_blocks", [])):
            handles.append(module.register_forward_hook(capture(f"local_after_text_{index}")))
        for index, module in enumerate(getattr(adapter.local_adapter, "latent_blocks", [])):
            handles.append(module.register_forward_hook(capture(f"local_after_latent_{index}")))
        if isinstance(adapter.local_adapter, ResidualQuestionConditionedAdapter):
            for index, module in enumerate(adapter.local_adapter.backbone.blocks):
                handles.append(module.register_forward_hook(capture(f"local_after_latent_{index}")))
        for index, module in enumerate(getattr(adapter.local_adapter, "text_latent_blocks", [])):
            handles.append(module.register_forward_hook(capture(f"local_text_after_latent_{index}", pool_text=True)))
        for index, module in enumerate(getattr(adapter.local_adapter, "pool_blocks", [])):
            handles.append(module.register_forward_hook(capture(f"local_after_pool_{index}")))
        local_output = getattr(local_backbone, "output", None)
        if local_output is not None:
            handles.append(local_output.register_forward_hook(capture("local_output_pre_gate")))
        if anchor_projection is not None:
            handles.append(
                anchor_projection.register_forward_hook(capture("local_anchor_condition"))
            )
        if hasattr(adapter.global_adapter, "latent_projection"):
            handles.append(
                adapter.global_adapter.latent_projection.register_forward_hook(capture("global_latent_projected"))
            )
        for index, module in enumerate(adapter.global_adapter.blocks):
            handles.append(module.register_forward_hook(capture(f"global_after_latent_{index}")))
        handles.append(adapter.global_adapter.output.register_forward_hook(capture("global_output_pre_scale")))
    attention_blocks = []
    if isinstance(adapter, HybridGlobalLocalAdapter):
        attention_blocks = [
            *getattr(adapter.local_adapter, "text_blocks", []),
            *getattr(adapter.local_adapter, "latent_blocks", []),
            *(
                list(adapter.local_adapter.backbone.blocks)
                if isinstance(adapter.local_adapter, ResidualQuestionConditionedAdapter)
                else []
            ),
        ]
        for block in attention_blocks:
            block.capture_attention = True
    try:
        if isinstance(adapter, HybridGlobalLocalAdapter):
            global_prompts, local_prompts, soft = adapter.forward_components(
                latent,
                question_embeds=text_embeds,
                question_mask=prompt_mask,
                structured_query=structured_query,
            )
            trace["global_prompt"] = global_prompts[0].detach().float().cpu()
            trace["local_prompt_or_residual"] = local_prompts[0].detach().float().cpu()
            trace["combined_prompt"] = soft[0].detach().float().cpu()
        else:
            soft = adapter(
                latent,
                question_embeds=text_embeds,
                question_mask=prompt_mask,
                structured_query=structured_query,
            )
    finally:
        if isinstance(adapter, HybridGlobalLocalAdapter):
            local_latent_blocks = (
                adapter.local_adapter.backbone.blocks
                if isinstance(adapter.local_adapter, ResidualQuestionConditionedAdapter)
                else getattr(adapter.local_adapter, "latent_blocks", [])
            )
            for prefix, blocks in (
                ("local_text_attention", getattr(adapter.local_adapter, "text_blocks", [])),
                ("local_latent_attention", local_latent_blocks),
            ):
                for index, block in enumerate(blocks):
                    if block.last_attention_weights is not None:
                        trace[f"{prefix}_{index}"] = block.last_attention_weights
                    self_weights = getattr(block, "last_self_attention_weights", None)
                    if self_weights is not None:
                        trace[f"local_query_self_attention_{index}"] = self_weights
                    block.capture_attention = False
                    block.last_attention_weights = None
                    if hasattr(block, "last_self_attention_weights"):
                        block.last_self_attention_weights = None
        for handle in handles:
            handle.remove()
    return soft, trace


@torch.no_grad()
def run_embedded_diagnostics(
    stage: str,
    llm,
    adapter: nn.Module,
    tokenizer,
    dataset: TensorReadoutQADataset,
    device: torch.device,
    args: argparse.Namespace,
    run_dir: Path,
) -> dict[str, Any]:
    selected = _diagnostic_records(dataset, max(1, int(args.diagnostics_records_per_task)))
    decoder = _decoder_for_diagnostics(llm)
    requested_layers = [int(value) for value in parse_csv(args.diagnostics_layers)]
    tensor_payload: dict[str, Any] = {"stage": stage, "records": {}}
    summaries: list[dict[str, Any]] = []
    was_training = adapter.training
    llm_was_training = bool(llm.training)
    adapter.eval()
    llm.eval()
    for index in selected:
        record = dataset.records[index]
        records = [record]
        answer = str(record["answer"])
        correct_latent = dataset.load_latent_for_record(record).unsqueeze(0)
        shuffled_latent = dataset.load_shuffled_latent(index).unsqueeze(0)
        input_ids, text_mask, text_labels = build_text_tensors(
            records=records,
            answers=[answer],
            tokenizer=tokenizer,
            max_prompt_tokens=int(args.max_prompt_tokens),
            max_target_tokens=int(args.max_target_tokens),
            append_eos=bool(args.append_eos),
            prompt_template=str(args.prompt_template),
        )
        input_ids = input_ids.to(device)
        text_mask = text_mask.to(device)
        text_labels = text_labels.to(device)
        text_embeds = llm.get_input_embeddings()(input_ids)
        prompt_mask = text_labels.eq(IGNORE_INDEX) & text_mask.bool()
        local_text_embeds = text_embeds
        local_text_mask = prompt_mask
        local_ids: torch.Tensor | None = None
        local_mask: torch.Tensor | None = None
        if (
            isinstance(adapter, HybridGlobalLocalAdapter)
            and adapter.local_adapter.question_input_mode == "contextual_tokens"
        ):
            local_ids, local_mask = build_local_question_tensors(
                records=records,
                tokenizer=tokenizer,
                device=device,
                max_tokens=int(args.max_prompt_tokens),
                prompt_template=str(args.prompt_template),
            )
            local_text_embeds = contextual_question_tokens_for_adapter(
                llm=llm,
                adapter=adapter,
                input_ids=local_ids,
                attention_mask=local_mask,
                prompt_mask=local_mask.bool(),
                fallback_layer=int(args.local_context_layer),
            )
            local_text_mask = local_mask.bool()
        condition = (
            structured_query_features(records, device)
            if bool(getattr(adapter, "structured_query_conditioning", False))
            else None
        )

        mode_states: dict[str, Any] = {}
        for mode, latent in (("correct", correct_latent), ("shuffled", shuffled_latent)):
            latent = latent.to(device)
            soft, adapter_hidden = _adapter_forward_with_trace(
                adapter=adapter,
                latent=latent,
                text_embeds=local_text_embeds,
                prompt_mask=local_text_mask,
                structured_query=condition,
            )
            soft = soft.to(dtype=text_embeds.dtype)
            inputs = torch.cat([soft, text_embeds], dim=1)
            attention = torch.cat(
                [torch.ones((1, soft.shape[1]), dtype=text_mask.dtype, device=device), text_mask],
                dim=1,
            )
            outputs = decoder(
                inputs_embeds=inputs,
                attention_mask=attention,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = outputs.hidden_states
            if hidden_states is None:
                raise RuntimeError("LLM diagnostics requested hidden states but the decoder returned none.")
            invalid_layers = [
                value for value in requested_layers if not -len(hidden_states) <= value < len(hidden_states)
            ]
            if invalid_layers:
                raise ValueError(
                    f"Diagnostic hidden-state layers {invalid_layers} are invalid for "
                    f"{len(hidden_states)} returned states."
                )
            resolved_layers = sorted(
                {value if value >= 0 else len(hidden_states) + value for value in requested_layers}
            )
            prompt_positions = torch.nonzero(prompt_mask[0], as_tuple=False).flatten()
            question_last = int(soft.shape[1] + prompt_positions[-1].item())
            local_count = (
                0
                if isinstance(adapter, HybridGlobalLocalAdapter) and adapter.residual_mode
                else int(adapter.local_adapter.soft_prompt_tokens)
                if isinstance(adapter, HybridGlobalLocalAdapter)
                else 0
            )
            layer_states: dict[str, Any] = {}
            for layer_index in resolved_layers:
                hidden = hidden_states[layer_index][0]
                layer_payload: dict[str, torch.Tensor] = {
                    "question_last": hidden[question_last].detach().float().cpu(),
                }
                if local_count:
                    layer_payload["local_mean"] = hidden[:local_count].mean(dim=0).detach().float().cpu()
                    layer_payload["global_mean"] = hidden[local_count : soft.shape[1]].mean(dim=0).detach().float().cpu()
                else:
                    layer_payload["soft_mean"] = hidden[: soft.shape[1]].mean(dim=0).detach().float().cpu()
                layer_states[str(layer_index)] = layer_payload
            mode_states[mode] = {
                "latent": latent[0].detach().float().cpu(),
                "soft_prompt": soft[0].detach().float().cpu(),
                "global_prompt": adapter_hidden.get("global_prompt"),
                "local_prompt_or_residual": adapter_hidden.get("local_prompt_or_residual"),
                "adapter_hidden": adapter_hidden,
                "layers": layer_states,
            }

        choices = [str(value) for value in record.get("choices", [answer])]
        score_modes: dict[str, Any] = {}
        for mode, latent in (("correct", correct_latent), ("shuffled", shuffled_latent)):
            scores = score_candidate_batch(
                llm=llm,
                adapter=adapter,
                tokenizer=tokenizer,
                records=[record] * len(choices),
                answers=choices,
                latent_map=latent.repeat(len(choices), 1, 1, 1),
                device=device,
                max_prompt_tokens=int(args.max_prompt_tokens),
                max_target_tokens=int(args.max_target_tokens),
                append_eos=bool(args.append_eos),
                prompt_template=str(args.prompt_template),
                soft_prompt_mode=mode,
                choice_score=str(args.choice_score),
                local_context_layer=int(args.local_context_layer),
            )
            target_index = choices.index(answer)
            wrong_scores = [value for choice_index, value in enumerate(scores) if choice_index != target_index]
            score_modes[mode] = {
                "prediction": choices[min(range(len(scores)), key=lambda choice_index: scores[choice_index])],
                "choice_nll": {choice: float(score) for choice, score in zip(choices, scores)},
                "answer_margin": float(min(wrong_scores) - scores[target_index]) if wrong_scores else 0.0,
            }

        generated_modes = {
            mode: generate_diagnostic_answer(
                llm=llm,
                tokenizer=tokenizer,
                record=record,
                soft_embeds=mode_states[mode]["soft_prompt"].unsqueeze(0),
                device=device,
                prompt_template=str(args.prompt_template),
                max_prompt_tokens=int(args.max_prompt_tokens),
                max_new_tokens=int(args.diagnostics_generation_max_new_tokens),
            )
            for mode in ("correct", "shuffled")
        }

        correct_state = mode_states["correct"]
        shuffled_state = mode_states["shuffled"]
        local_count = (
            0
            if isinstance(adapter, HybridGlobalLocalAdapter) and adapter.residual_mode
            else int(adapter.local_adapter.soft_prompt_tokens)
            if isinstance(adapter, HybridGlobalLocalAdapter)
            else 0
        )
        local_attention_summary: dict[str, Any] | None = None
        if isinstance(adapter, HybridGlobalLocalAdapter) and local_ids is not None:
            local_attention_summary = {}
            text_attention_keys = sorted(
                key for key in correct_state["adapter_hidden"] if key.startswith("local_text_attention_")
            )
            latent_attention_keys = sorted(
                key for key in correct_state["adapter_hidden"] if key.startswith("local_latent_attention_")
            )
            if text_attention_keys:
                token_ids = local_ids[0].detach().cpu()
                local_attention_summary["text_blocks"] = {}
                for key in text_attention_keys:
                    text_weights = correct_state["adapter_hidden"][key].mean(dim=(0, 1, 2))
                    top_count = min(8, int(text_weights.numel()))
                    values, indices = torch.topk(text_weights, k=top_count)
                    local_attention_summary["text_blocks"][key] = [
                        {
                            "index": int(index),
                            "token": str(tokenizer.convert_ids_to_tokens(int(token_ids[index]))),
                            "text": tokenizer.decode([int(token_ids[index])]),
                            "weight": float(value),
                        }
                        for value, index in zip(values.tolist(), indices.tolist())
                    ]
            if latent_attention_keys:
                latent_width = int(adapter.local_adapter.latent_grid[1])
                local_attention_summary["latent_blocks"] = {}
                for key in latent_attention_keys:
                    latent_weights = correct_state["adapter_hidden"][key].mean(dim=(0, 1, 2))
                    top_count = min(8, int(latent_weights.numel()))
                    values, indices = torch.topk(latent_weights, k=top_count)
                    local_attention_summary["latent_blocks"][key] = [
                        {
                            "index": int(index),
                            "row": int(index) // latent_width,
                            "col": int(index) % latent_width,
                            "weight": float(value),
                        }
                        for value, index in zip(values.tolist(), indices.tolist())
                    ]
        soft_comparison = _cosine_and_relative_l2(correct_state["soft_prompt"], shuffled_state["soft_prompt"])
        soft_comparison["query_off_diagonal_cosine"] = _mean_off_diagonal_cosine(
            correct_state["soft_prompt"]
        )
        if isinstance(adapter, HybridGlobalLocalAdapter) and adapter.residual_mode:
            local_comparison = _cosine_and_relative_l2(
                correct_state["local_prompt_or_residual"], shuffled_state["local_prompt_or_residual"]
            )
            global_comparison = _cosine_and_relative_l2(
                correct_state["global_prompt"], shuffled_state["global_prompt"]
            )
            soft_comparison.update(
                {
                    "local_cosine": local_comparison["cosine"],
                    "local_relative_l2": local_comparison["relative_l2"],
                    "global_cosine": global_comparison["cosine"],
                    "global_relative_l2": global_comparison["relative_l2"],
                    "local_rms": _rms(correct_state["local_prompt_or_residual"]),
                    "global_rms": _rms(correct_state["global_prompt"]),
                }
            )
        elif local_count:
            local_comparison = _cosine_and_relative_l2(
                correct_state["soft_prompt"][:local_count], shuffled_state["soft_prompt"][:local_count]
            )
            global_comparison = _cosine_and_relative_l2(
                correct_state["soft_prompt"][local_count:], shuffled_state["soft_prompt"][local_count:]
            )
            soft_comparison.update(
                {
                    "local_cosine": local_comparison["cosine"],
                    "local_relative_l2": local_comparison["relative_l2"],
                    "global_cosine": global_comparison["cosine"],
                    "global_relative_l2": global_comparison["relative_l2"],
                    "local_rms": _rms(correct_state["soft_prompt"][:local_count]),
                    "global_rms": _rms(correct_state["soft_prompt"][local_count:]),
                }
            )
        hidden_comparison: dict[str, Any] = {}
        for layer_index, correct_layer in correct_state["layers"].items():
            hidden_comparison[layer_index] = {
                name: _cosine_and_relative_l2(value, shuffled_state["layers"][layer_index][name])
                for name, value in correct_layer.items()
            }
        adapter_hidden_comparison = {
            name: _cosine_and_relative_l2(value, shuffled_state["adapter_hidden"][name])
            for name, value in correct_state["adapter_hidden"].items()
            if name in shuffled_state["adapter_hidden"]
        }
        alternate_record = _alternate_question_record(dataset, index)
        question_sensitivity: dict[str, Any] | None = None
        if alternate_record is not None:
            alt_answer = str(alternate_record["answer"])
            alt_ids, alt_mask, alt_labels = build_text_tensors(
                records=[alternate_record],
                answers=[alt_answer],
                tokenizer=tokenizer,
                max_prompt_tokens=int(args.max_prompt_tokens),
                max_target_tokens=int(args.max_target_tokens),
                append_eos=bool(args.append_eos),
                prompt_template=str(args.prompt_template),
            )
            alt_ids = alt_ids.to(device)
            alt_mask = alt_mask.to(device)
            alt_labels = alt_labels.to(device)
            alt_text_embeds = llm.get_input_embeddings()(alt_ids)
            alt_prompt_mask = alt_labels.eq(IGNORE_INDEX) & alt_mask.bool()
            if (
                isinstance(adapter, HybridGlobalLocalAdapter)
                and adapter.local_adapter.question_input_mode == "contextual_tokens"
            ):
                alt_local_ids, alt_local_mask = build_local_question_tensors(
                    records=[alternate_record],
                    tokenizer=tokenizer,
                    device=device,
                    max_tokens=int(args.max_prompt_tokens),
                    prompt_template=str(args.prompt_template),
                )
                alt_text_embeds = contextual_question_tokens_for_adapter(
                    llm=llm,
                    adapter=adapter,
                    input_ids=alt_local_ids,
                    attention_mask=alt_local_mask,
                    prompt_mask=alt_local_mask.bool(),
                    fallback_layer=int(args.local_context_layer),
                )
                alt_prompt_mask = alt_local_mask.bool()
            alt_condition = (
                structured_query_features([alternate_record], device)
                if bool(getattr(adapter, "structured_query_conditioning", False))
                else None
            )
            alt_soft, alt_trace = _adapter_forward_with_trace(
                adapter=adapter,
                latent=correct_latent.to(device),
                text_embeds=alt_text_embeds,
                prompt_mask=alt_prompt_mask,
                structured_query=alt_condition,
            )
            alt_soft = alt_soft[0].detach().float().cpu()
            question_sensitivity = {
                "alternate_qa_id": str(alternate_record.get("qa_id", "")),
                "alternate_task_type": str(alternate_record.get("task_type", "unknown")),
                "same_latent_soft_prompt": _cosine_and_relative_l2(correct_state["soft_prompt"], alt_soft),
            }
            if isinstance(adapter, HybridGlobalLocalAdapter) and adapter.residual_mode:
                question_sensitivity["same_latent_local_prompt"] = _cosine_and_relative_l2(
                    correct_state["local_prompt_or_residual"], alt_trace["local_prompt_or_residual"]
                )
            elif local_count:
                question_sensitivity["same_latent_local_prompt"] = _cosine_and_relative_l2(
                    correct_state["soft_prompt"][:local_count], alt_soft[:local_count]
                )
        same_task_record = _same_task_alternate_question_record(dataset, index)
        same_task_sensitivity: dict[str, Any] | None = None
        if same_task_record is not None and isinstance(adapter, HybridGlobalLocalAdapter):
            same_ids, same_mask = build_local_question_tensors(
                records=[same_task_record],
                tokenizer=tokenizer,
                device=device,
                max_tokens=int(args.max_prompt_tokens),
                prompt_template=str(args.prompt_template),
            )
            same_text_embeds = llm.get_input_embeddings()(same_ids)
            if (
                isinstance(adapter, HybridGlobalLocalAdapter)
                and adapter.local_adapter.question_input_mode == "contextual_tokens"
            ):
                same_text_embeds = contextual_question_tokens_for_adapter(
                    llm=llm,
                    adapter=adapter,
                    input_ids=same_ids,
                    attention_mask=same_mask,
                    prompt_mask=same_mask.bool(),
                    fallback_layer=int(args.local_context_layer),
                )
            same_soft, same_trace = _adapter_forward_with_trace(
                adapter=adapter,
                latent=correct_latent.to(device),
                text_embeds=same_text_embeds,
                prompt_mask=same_mask.bool(),
                structured_query=None,
            )
            same_soft = same_soft[0].detach().float().cpu()
            swapped_scores = score_candidate_batch(
                llm=llm,
                adapter=adapter,
                tokenizer=tokenizer,
                records=[record] * len(choices),
                answers=choices,
                latent_map=correct_latent.repeat(len(choices), 1, 1, 1),
                device=device,
                max_prompt_tokens=int(args.max_prompt_tokens),
                max_target_tokens=int(args.max_target_tokens),
                append_eos=bool(args.append_eos),
                prompt_template=str(args.prompt_template),
                soft_prompt_mode="correct",
                choice_score=str(args.choice_score),
                local_context_layer=int(args.local_context_layer),
                precomputed_soft_embeds=same_soft.unsqueeze(0).repeat(len(choices), 1, 1),
            )
            target_index = choices.index(answer)
            swapped_wrong_scores = [
                value for choice_index, value in enumerate(swapped_scores) if choice_index != target_index
            ]
            same_local_prompt = (
                same_trace["local_prompt_or_residual"]
                if adapter.residual_mode
                else same_soft[:local_count]
            )
            correct_local_prompt = (
                correct_state["local_prompt_or_residual"]
                if adapter.residual_mode
                else correct_state["soft_prompt"][:local_count]
            )
            same_task_sensitivity = {
                "alternate_qa_id": str(same_task_record.get("qa_id", "")),
                "same_latent_local_prompt": _cosine_and_relative_l2(
                    correct_local_prompt, same_local_prompt
                ),
                "swapped_question_prediction": choices[
                    min(range(len(swapped_scores)), key=lambda choice_index: swapped_scores[choice_index])
                ],
                "swapped_question_answer_margin": float(
                    min(swapped_wrong_scores) - swapped_scores[target_index]
                    if swapped_wrong_scores
                    else 0.0
                ),
            }
        qa_id = str(record.get("qa_id", f"index_{index}"))
        tensor_payload["records"][qa_id] = {
            "input_ids": input_ids[0].detach().cpu(),
            "text_attention_mask": text_mask[0].detach().cpu(),
            "text_labels": text_labels[0].detach().cpu(),
            "prompt_mask": prompt_mask[0].detach().cpu(),
            "local_input_ids": local_ids[0].detach().cpu() if local_ids is not None else None,
            "local_attention_mask": local_mask[0].detach().cpu() if local_mask is not None else None,
            "modes": mode_states,
        }
        summaries.append(
            {
                "qa_id": qa_id,
                "task_type": str(record.get("task_type", "unknown")),
                "field": str(record.get("field", "unknown")),
                "question": str(record.get("query") or record.get("question") or ""),
                "rendered_prompt": build_prompt(record, prompt_template=str(args.prompt_template)),
                "local_conditioning_prompt": build_local_conditioning_prompt(
                    record, prompt_template=str(args.prompt_template)
                ),
                "choices": choices,
                "answer": answer,
                "scores": score_modes,
                "generation": generated_modes,
                "soft_prompt_correct_vs_shuffled": soft_comparison,
                "adapter_hidden_correct_vs_shuffled": adapter_hidden_comparison,
                "question_sensitivity": question_sensitivity,
                "same_task_question_sensitivity": same_task_sensitivity,
                "local_attention": local_attention_summary,
                "hidden_correct_vs_shuffled": hidden_comparison,
            }
        )

    diagnostic_dir = run_dir / "diagnostics"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    layer_names = sorted(
        {layer for record_summary in summaries for layer in record_summary["hidden_correct_vs_shuffled"]},
        key=int,
    )
    question_sensitive_records = [item for item in summaries if item["question_sensitivity"] is not None]
    same_task_sensitive_records = [
        item for item in summaries if item["same_task_question_sensitivity"] is not None
    ]
    local_rms_values = [
        float(item["soft_prompt_correct_vs_shuffled"].get("local_rms", 0.0)) for item in summaries
    ]
    global_rms_values = [
        float(item["soft_prompt_correct_vs_shuffled"].get("global_rms", 0.0)) for item in summaries
    ]
    same_task_sensitivity_by_task: dict[str, float] = {}
    same_task_margin_delta_by_task: dict[str, float] = {}
    for task in sorted({str(item["task_type"]) for item in same_task_sensitive_records}):
        task_items = [item for item in same_task_sensitive_records if str(item["task_type"]) == task]
        same_task_sensitivity_by_task[task] = sum(
            item["same_task_question_sensitivity"]["same_latent_local_prompt"]["relative_l2"]
            for item in task_items
        ) / max(1, len(task_items))
        same_task_margin_delta_by_task[task] = sum(
            item["scores"]["correct"]["answer_margin"]
            - item["same_task_question_sensitivity"]["swapped_question_answer_margin"]
            for item in task_items
        ) / max(1, len(task_items))
    generation_by_task: dict[str, dict[str, float]] = {}
    score_diagnostics_by_task: dict[str, dict[str, float]] = {}
    for task in sorted({str(item["task_type"]) for item in summaries}):
        task_items = [item for item in summaries if str(item["task_type"]) == task]
        generation_by_task[task] = {}
        for mode in ("correct", "shuffled"):
            generation_by_task[task][f"{mode}_format_valid_rate"] = sum(
                bool(item["generation"][mode]["format_valid"]) for item in task_items
            ) / max(1, len(task_items))
            generation_by_task[task][f"{mode}_parsed_accuracy"] = sum(
                item["generation"][mode]["parsed_choice"] == item["answer"] for item in task_items
            ) / max(1, len(task_items))
        correct_margin = sum(item["scores"]["correct"]["answer_margin"] for item in task_items) / max(
            1, len(task_items)
        )
        shuffled_margin = sum(item["scores"]["shuffled"]["answer_margin"] for item in task_items) / max(
            1, len(task_items)
        )
        score_diagnostics_by_task[task] = {
            "correct_accuracy": sum(
                item["scores"]["correct"]["prediction"] == item["answer"] for item in task_items
            ) / max(1, len(task_items)),
            "shuffled_accuracy": sum(
                item["scores"]["shuffled"]["prediction"] == item["answer"] for item in task_items
            ) / max(1, len(task_items)),
            "correct_answer_margin": correct_margin,
            "shuffled_answer_margin": shuffled_margin,
            "correct_minus_shuffled_answer_margin": correct_margin - shuffled_margin,
        }
    text_layer_weights: dict[str, float] = {}
    text_attention_gates: dict[str, float] = {}
    residual_gate = 0.0
    if isinstance(adapter, HybridGlobalLocalAdapter):
        residual_gate = float(adapter.local_adapter.gate.detach().float().cpu().item())
        layer_logits = getattr(adapter.local_adapter, "text_layer_logits", None)
        context_layers = getattr(adapter.local_adapter, "context_layers", ())
        if isinstance(layer_logits, torch.Tensor):
            layer_weights = torch.softmax(layer_logits.detach().float().cpu(), dim=0)
            text_layer_weights = {
                str(layer): float(weight) for layer, weight in zip(context_layers, layer_weights.tolist())
            }
        text_attention_gates = {
            str(index): float(block.gate.detach().float().cpu().item())
            for index, block in enumerate(getattr(adapter.local_adapter, "text_blocks", []))
            if isinstance(getattr(block, "gate", None), torch.Tensor)
        }
    aggregate = {
        "records": len(summaries),
        "soft_prompt_relative_l2_mean": sum(
            item["soft_prompt_correct_vs_shuffled"]["relative_l2"] for item in summaries
        )
        / max(1, len(summaries)),
        "local_prompt_relative_l2_mean": sum(
            item["soft_prompt_correct_vs_shuffled"].get("local_relative_l2", 0.0) for item in summaries
        )
        / max(1, len(summaries)),
        "global_prompt_relative_l2_mean": sum(
            item["soft_prompt_correct_vs_shuffled"].get("global_relative_l2", 0.0) for item in summaries
        )
        / max(1, len(summaries)),
        "answer_margin_correct_mean": sum(item["scores"]["correct"]["answer_margin"] for item in summaries)
        / max(1, len(summaries)),
        "answer_margin_shuffled_mean": sum(item["scores"]["shuffled"]["answer_margin"] for item in summaries)
        / max(1, len(summaries)),
        "answer_margin_correct_minus_shuffled": sum(
            item["scores"]["correct"]["answer_margin"] - item["scores"]["shuffled"]["answer_margin"]
            for item in summaries
        )
        / max(1, len(summaries)),
        "score_diagnostics_by_task": score_diagnostics_by_task,
        "correct_prediction_accuracy": sum(
            item["scores"]["correct"]["prediction"] == item["answer"] for item in summaries
        )
        / max(1, len(summaries)),
        "shuffled_prediction_accuracy": sum(
            item["scores"]["shuffled"]["prediction"] == item["answer"] for item in summaries
        )
        / max(1, len(summaries)),
        "same_latent_different_question_local_relative_l2_mean": sum(
            item["question_sensitivity"]["same_latent_local_prompt"]["relative_l2"]
            for item in question_sensitive_records
            if "same_latent_local_prompt" in item["question_sensitivity"]
        )
        / max(1, len(question_sensitive_records)),
        "same_task_different_question_local_relative_l2_mean": sum(
            item["same_task_question_sensitivity"]["same_latent_local_prompt"]["relative_l2"]
            for item in same_task_sensitive_records
        )
        / max(1, len(same_task_sensitive_records)),
        "same_task_different_question_local_relative_l2_by_task": same_task_sensitivity_by_task,
        "same_task_correct_minus_swapped_answer_margin_by_task": same_task_margin_delta_by_task,
        "same_task_swapped_question_answer_margin_mean": sum(
            item["same_task_question_sensitivity"]["swapped_question_answer_margin"]
            for item in same_task_sensitive_records
        )
        / max(1, len(same_task_sensitive_records)),
        "same_task_correct_minus_swapped_answer_margin": sum(
            item["scores"]["correct"]["answer_margin"]
            - item["same_task_question_sensitivity"]["swapped_question_answer_margin"]
            for item in same_task_sensitive_records
        )
        / max(1, len(same_task_sensitive_records)),
        "local_anchor_gate": float(
            adapter.local_adapter.anchor_gate.detach().float().cpu().item()
            if isinstance(adapter, HybridGlobalLocalAdapter)
            and getattr(adapter.local_adapter, "anchor_gate", None) is not None
            else 0.0
        ),
        "local_residual_gate": residual_gate,
        "text_context_layer_weights": text_layer_weights,
        "text_cross_attention_gates": text_attention_gates,
        "generation": {
            "correct_format_valid_rate": sum(
                bool(item["generation"]["correct"]["format_valid"]) for item in summaries
            ) / max(1, len(summaries)),
            "correct_parsed_accuracy": sum(
                item["generation"]["correct"]["parsed_choice"] == item["answer"] for item in summaries
            ) / max(1, len(summaries)),
            "correct_semantic_but_format_invalid_rate": sum(
                item["generation"]["correct"]["parsed_choice"] == item["answer"]
                and not bool(item["generation"]["correct"]["format_valid"])
                for item in summaries
            ) / max(1, len(summaries)),
            "shuffled_format_valid_rate": sum(
                bool(item["generation"]["shuffled"]["format_valid"]) for item in summaries
            ) / max(1, len(summaries)),
            "shuffled_parsed_accuracy": sum(
                item["generation"]["shuffled"]["parsed_choice"] == item["answer"] for item in summaries
            ) / max(1, len(summaries)),
            "by_task": generation_by_task,
        },
        "local_prompt_rms_mean": sum(local_rms_values) / max(1, len(local_rms_values)),
        "global_prompt_rms_mean": sum(global_rms_values) / max(1, len(global_rms_values)),
        "local_to_global_prompt_rms_ratio": (
            (sum(local_rms_values) / max(1, len(local_rms_values)))
            / max(sum(global_rms_values) / max(1, len(global_rms_values)), 1.0e-8)
        ),
        "soft_prompt_query_off_diagonal_cosine_mean": sum(
            item["soft_prompt_correct_vs_shuffled"].get("query_off_diagonal_cosine", 1.0)
            for item in summaries
        )
        / max(1, len(summaries)),
        "question_last_relative_l2_by_layer": {
            layer: sum(
                item["hidden_correct_vs_shuffled"][layer]["question_last"]["relative_l2"] for item in summaries
            )
            / max(1, len(summaries))
            for layer in layer_names
        },
    }
    summary = {
        "stage": stage,
        "aggregate": aggregate,
        "records": summaries,
        "state_file": str(diagnostic_dir / f"{stage}_states.pt"),
        "structured_query_conditioning": bool(getattr(adapter, "structured_query_conditioning", False)),
    }
    atomic_dump_json(diagnostic_dir / f"{stage}_summary.json", summary)
    atomic_torch_save(diagnostic_dir / f"{stage}_states.pt", tensor_payload)
    if was_training:
        adapter.train()
    if llm_was_training:
        set_frozen_llm_execution_mode(llm, checkpoint_training=True)
    return summary


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
    atomic_torch_save(path, payload)


def redacted_args(args: argparse.Namespace) -> dict[str, Any]:
    payload = dict(vars(args))
    if payload.get("wandb_api_key"):
        payload["wandb_api_key"] = "***REDACTED***"
    return payload


def redacted_config_snapshot(config: Mapping[str, Any]) -> dict[str, Any]:
    payload = json.loads(json.dumps(dict(config)))
    wandb_config = payload.get("wandb")
    if isinstance(wandb_config, dict) and wandb_config.get("api_key"):
        wandb_config["api_key"] = "***REDACTED***"
    return payload


def load_compatible_hybrid_state(
    adapter: HybridGlobalLocalAdapter,
    state_dict: Mapping[str, Any],
) -> dict[str, Any]:
    current = adapter.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for raw_key, value in state_dict.items():
        key = str(raw_key)
        if key.startswith("global_adapter."):
            continue
        if adapter.local_adapter.question_input_mode == "contextual_tokens":
            skipped.append(key)
            continue
        if key.startswith("local_adapter.structured_projection."):
            skipped.append(key)
            continue
        if key in current and isinstance(value, torch.Tensor) and tuple(value.shape) == tuple(current[key].shape):
            compatible[key] = value
        else:
            skipped.append(key)
    result = adapter.load_state_dict(compatible, strict=False)
    contextual_reinit = adapter.local_adapter.question_input_mode == "contextual_tokens"
    return {
        "mode": "global_only_contextual_local_reinit" if contextual_reinit else "compatible_local_warm_start",
        "local_loaded_parameter_tensors": len(compatible),
        "skipped_keys": sorted(skipped),
        "new_or_missing_keys": sorted(key for key in result.missing_keys if key.startswith("local_adapter.")),
        "unexpected_keys": sorted(result.unexpected_keys),
    }


def adapter_from_checkpoint(
    checkpoint: Mapping[str, Any],
    latent_shape: Sequence[int],
    llm_hidden_size: int,
) -> nn.Module:
    ckpt_args = checkpoint.get("args")
    state_dict = checkpoint.get("adapter_state_dict")
    if not isinstance(ckpt_args, Mapping) or not isinstance(state_dict, Mapping):
        raise ValueError("Adapter checkpoint must contain args and adapter_state_dict mappings.")
    architecture = str(ckpt_args.get("adapter_architecture", ""))
    if not architecture:
        checkpoint_adapter_type = str(ckpt_args.get("adapter_type", ""))
        if checkpoint_adapter_type == "spatial_transformer":
            architecture = "alignment_adapter"
        else:
            architecture = (
                "alignment_qformer"
                if "query_tokens" in state_dict and "latent_pos_embed" in state_dict
                else "legacy"
            )
    latent_channels = int(latent_shape[0])
    if architecture == "legacy":
        adapter = TensorSoftPromptAdapter(
            latent_channels=latent_channels,
            llm_hidden_size=llm_hidden_size,
            soft_prompt_tokens=int(ckpt_args.get("soft_prompt_tokens", 32)),
            adapter_dim=int(ckpt_args.get("adapter_dim", 512)),
            adapter_layers=int(ckpt_args.get("adapter_layers", 2)),
            adapter_heads=int(ckpt_args.get("adapter_heads", 8)),
            dropout=float(ckpt_args.get("dropout", 0.1)),
            latent_pos_encoding=str(ckpt_args.get("latent_pos_encoding", "grid")),
            question_conditioning=bool(ckpt_args.get("question_conditioning", True)),
            question_condition_gate_init=float(ckpt_args.get("question_condition_gate_init", 1.0)),
            structured_query_conditioning=bool(ckpt_args.get("structured_query_conditioning", False)),
            soft_prompt_scale=float(ckpt_args.get("soft_prompt_scale", 0.0)),
        )
        adapter.load_state_dict(state_dict, strict=True)
        return adapter

    adapter_dim = int(ckpt_args.get("adapter_dim", 512))
    adapter_heads = int(ckpt_args.get("adapter_heads", 8))
    global_prefix = (
        "global_adapter."
        if architecture in {
            "hybrid_local_qformer",
            "residual_question_qformer",
            "residual_question_adapter",
        }
        else ""
    )
    global_adapter_type = str(
        ckpt_args.get("global_adapter_type", ckpt_args.get("adapter_type", "qformer"))
    ).lower()
    if global_adapter_type == "spatial_transformer":
        global_tokens = int(latent_shape[-2]) * int(latent_shape[-1])
    else:
        query_key = f"{global_prefix}query_tokens"
        query_tensor = state_dict.get(query_key)
        if not isinstance(query_tensor, torch.Tensor):
            raise ValueError(f"Checkpoint is missing {query_key}.")
        global_tokens = int(query_tensor.shape[1])
    global_layers = max(
        [
            int(str(key).split(".")[2 if global_prefix else 1]) + 1
            for key in state_dict
            if str(key).startswith(f"{global_prefix}blocks.") and str(key).split(".")[2 if global_prefix else 1].isdigit()
        ]
        or [int(ckpt_args.get("adapter_layers", 2))]
    )
    global_adapter = TensorPatchAlignmentAdapter(
        latent_channels=latent_channels,
        latent_grid=tuple(int(value) for value in latent_shape[-2:]),
        adapter_dim=adapter_dim,
        projection_dim=llm_hidden_size,
        dropout=float(ckpt_args.get("global_dropout", ckpt_args.get("dropout", 0.0))),
        adapter_type=global_adapter_type,
        query_tokens=global_tokens,
        adapter_layers=global_layers,
        adapter_heads=adapter_heads,
        soft_prompt_scale=float(
            ckpt_args.get("global_soft_prompt_scale", ckpt_args.get("soft_prompt_scale", 0.05))
        ),
    )
    if architecture in {"alignment_qformer", "alignment_adapter"}:
        global_adapter.load_state_dict(state_dict, strict=True)
        return global_adapter

    if architecture in {"residual_question_qformer", "residual_question_adapter"}:
        local_adapter = ResidualQuestionConditionedAdapter(
            aligned_adapter=global_adapter,
            llm_hidden_size=llm_hidden_size,
            context_layers=[int(value) for value in parse_csv(ckpt_args.get("local_context_layers", "2,6"))],
            adapter_heads=adapter_heads,
            dropout=float(ckpt_args.get("dropout", 0.0)),
            text_gate_init=float(ckpt_args.get("local_text_gate_init", 0.05)),
            residual_gate_init=float(ckpt_args.get("local_gate_init", 0.1)),
        )
        adapter = HybridGlobalLocalAdapter(
            global_adapter=global_adapter,
            local_adapter=local_adapter,
            freeze_global=True,
            global_prompt_dropout=float(ckpt_args.get("global_prompt_dropout", 0.0)),
            combine_mode="residual",
        )
        adapter.load_state_dict(state_dict, strict=True)
        return adapter

    if global_adapter_type != "qformer":
        raise ValueError(
            f"{architecture} supports only qformer global adapters; use residual_question_adapter for "
            f"stage-1 adapter_type={global_adapter_type}."
        )

    local_query = state_dict.get("local_adapter.query_tokens")
    text_pos = state_dict.get("local_adapter.text_pos_embed")
    if not isinstance(local_query, torch.Tensor) or not isinstance(text_pos, torch.Tensor):
        raise ValueError("Hybrid checkpoint is missing local query or text position tensors.")
    local_adapter = QuestionConditionedLocalAdapter(
        latent_channels=latent_channels,
        latent_grid=tuple(int(value) for value in latent_shape[-2:]),
        llm_hidden_size=llm_hidden_size,
        adapter_dim=adapter_dim,
        local_tokens=int(local_query.shape[1]),
        local_layers=int(ckpt_args.get("local_adapter_layers", 2)),
        text_encoder_layers=int(ckpt_args.get("local_text_encoder_layers", 0)),
        adapter_heads=adapter_heads,
        dropout=float(ckpt_args.get("dropout", 0.0)),
        soft_prompt_scale=float(ckpt_args.get("soft_prompt_scale", 0.05)),
        gate_init=float(ckpt_args.get("local_gate_init", 0.1)),
        max_text_tokens=int(text_pos.shape[1]),
        structured_query_conditioning=bool(ckpt_args.get("structured_query_conditioning", False)),
        question_input_mode=str(ckpt_args.get("local_question_input_mode", "input_embeddings")),
        fusion_mode=str(ckpt_args.get("local_fusion_mode", "text_latent_pool")),
    )
    adapter = HybridGlobalLocalAdapter(
        global_adapter,
        local_adapter,
        freeze_global=False,
        global_prompt_dropout=float(ckpt_args.get("global_prompt_dropout", 0.0)),
    )
    adapter.load_state_dict(state_dict, strict=True)
    return adapter


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
    for baseline in (
        "global_only",
        "local_only",
        "no_latent",
        "zero_latent",
        "shuffled",
        "shuffled_stats",
        "random",
    ):
        baseline_metrics = metrics.get(baseline)
        if isinstance(baseline_metrics, Mapping) and isinstance(baseline_metrics.get("accuracy"), (int, float)):
            payload[f"{prefix}/correct_minus_{baseline}_accuracy"] = correct_accuracy - float(
                baseline_metrics["accuracy"]
            )


def compact_accuracy_metrics(prefix: str, metrics: Mapping[str, Any]) -> dict[str, float]:
    """Keep W&B readable while detailed field/task matrices remain in local JSON files."""
    payload: dict[str, float] = {}
    for mode, raw_metrics in metrics.items():
        if not isinstance(raw_metrics, Mapping):
            continue
        accuracy = raw_metrics.get("accuracy")
        if isinstance(accuracy, (int, float)):
            payload[f"{prefix}/{mode}/accuracy"] = float(accuracy)
        by_task = raw_metrics.get("by_task")
        if mode == "correct" and isinstance(by_task, Mapping):
            for task, task_metrics in by_task.items():
                if isinstance(task_metrics, Mapping) and isinstance(task_metrics.get("accuracy"), (int, float)):
                    payload[f"{prefix}/task/{task}/accuracy"] = float(task_metrics["accuracy"])
    correct = metrics.get("correct")
    shuffled = metrics.get("shuffled")
    if isinstance(correct, Mapping) and isinstance(shuffled, Mapping):
        correct_tasks = correct.get("by_task")
        shuffled_tasks = shuffled.get("by_task")
        if isinstance(correct_tasks, Mapping) and isinstance(shuffled_tasks, Mapping):
            for task, correct_task in correct_tasks.items():
                shuffled_task = shuffled_tasks.get(task)
                if (
                    isinstance(correct_task, Mapping)
                    and isinstance(shuffled_task, Mapping)
                    and isinstance(correct_task.get("accuracy"), (int, float))
                    and isinstance(shuffled_task.get("accuracy"), (int, float))
                ):
                    payload[f"{prefix}/task/{task}/latent_gain"] = float(correct_task["accuracy"]) - float(
                        shuffled_task["accuracy"]
                    )
    add_accuracy_deltas(prefix, metrics, payload)
    return payload


def compact_diagnostic_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    keys = (
        "local_prompt_relative_l2_mean",
        "answer_margin_correct_minus_shuffled",
        "same_task_different_question_local_relative_l2_mean",
        "same_task_swapped_question_answer_margin_mean",
        "local_to_global_prompt_rms_ratio",
    )
    payload = {
        f"diagnostics/{key}": float(metrics[key])
        for key in keys
        if isinstance(metrics.get(key), (int, float))
    }
    for key in ("local_residual_gate",):
        if isinstance(metrics.get(key), (int, float)):
            payload[f"diagnostics/{key}"] = float(metrics[key])
    generation = metrics.get("generation")
    if isinstance(generation, Mapping):
        for key in (
            "correct_format_valid_rate",
            "correct_parsed_accuracy",
            "correct_semantic_but_format_invalid_rate",
        ):
            if isinstance(generation.get(key), (int, float)):
                payload[f"diagnostics/generation/{key}"] = float(generation[key])
    return payload


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
    raw_summary = dict(summary or {})
    compact_summary = {
        key: raw_summary[key]
        for key in (
            "device",
            "distributed",
            "model_dtype",
            "train_records",
            "val_records",
            "test_records",
            "adapter_initialization",
            "adapter_parameters_total",
            "trainable_adapter_parameters",
            "frozen_llm_parameters",
        )
        if key in raw_summary
    }
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
            "require_disjoint_splits": bool(args.require_disjoint_splits),
            "require_untruncated_prompts": bool(args.require_untruncated_prompts),
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
            "local_text_encoder_layers": int(args.local_text_encoder_layers),
            "local_question_input_mode": str(args.local_question_input_mode),
            "local_context_layer": int(args.local_context_layer),
            "local_context_layers": parse_csv(args.local_context_layers),
            "local_fusion_mode": str(args.local_fusion_mode),
            "local_gate_init": float(args.local_gate_init),
            "local_text_gate_init": float(args.local_text_gate_init),
            "freeze_global_adapter": bool(args.freeze_global_adapter),
            "global_unfreeze_epoch": int(args.global_unfreeze_epoch),
            "global_lr": float(args.global_lr),
            "global_prompt_dropout": float(args.global_prompt_dropout),
            "global_dropout": float(getattr(args, "global_dropout", args.dropout)),
            "global_soft_prompt_scale": float(
                getattr(args, "global_soft_prompt_scale", args.soft_prompt_scale)
            ),
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
            "lr_scheduler": str(args.lr_scheduler),
            "warmup_ratio": float(args.warmup_ratio),
            "min_lr_ratio": float(args.min_lr_ratio),
            "weight_decay": float(args.weight_decay),
            "grad_clip_norm": float(args.grad_clip_norm),
            "ce_loss_weight": float(args.ce_loss_weight),
            "choice_ce_loss_weight": float(args.choice_ce_loss_weight),
            "ranking_loss_weight": float(args.ranking_loss_weight),
            "ranking_loss_margin": float(args.ranking_loss_margin),
            "ranking_loss_negative": str(args.ranking_loss_negative),
            "swapped_question_loss_weight": float(args.swapped_question_loss_weight),
            "swapped_question_loss_margin": float(args.swapped_question_loss_margin),
            "swapped_question_max_records": int(args.swapped_question_max_records),
            "max_prompt_tokens": int(args.max_prompt_tokens),
            "max_target_tokens": int(args.max_target_tokens),
            "append_eos": bool(args.append_eos),
            "checkpoint_metric": str(args.checkpoint_metric),
            "eval_baselines": parse_csv(args.eval_baselines),
            "final_eval_baselines": parse_csv(args.final_eval_baselines),
            "choice_score": str(args.choice_score),
            "log_interval": int(args.log_interval),
            "console_progress": bool(args.console_progress),
            "save_step_metrics": bool(args.save_step_metrics),
            "group_questions_by_state": bool(args.group_questions_by_state),
            "questions_per_state_group": int(args.questions_per_state_group),
            "diagnostics": {
                "enabled": bool(args.diagnostics_enabled),
                "every_epochs": int(args.diagnostics_every_epochs),
                "records_per_task": int(args.diagnostics_records_per_task),
                "generation_max_new_tokens": int(args.diagnostics_generation_max_new_tokens),
                "layers": parse_csv(args.diagnostics_layers),
            },
        },
        "run_summary": compact_summary,
        "wandb": {
            "enabled": bool(args.wandb_enabled),
            "api_key": args.wandb_api_key,
            "project": str(args.wandb_project),
            "entity": args.wandb_entity,
            "group": args.wandb_group,
            "tags": parse_csv(args.wandb_tags),
            "mode": str(args.wandb_mode),
            "log_model": bool(args.wandb_log_model),
            "detailed_metrics": bool(args.wandb_detailed_metrics),
        },
    }


def log_adapter_artifact(wandb_logger: WandbLogger, path: Path, name: str) -> None:
    if wandb_logger.run is None or wandb_logger._wandb is None or not path.exists():
        return
    artifact = wandb_logger._wandb.Artifact(name=name, type="adapter-checkpoint")
    artifact.add_file(str(path))
    wandb_logger.run.log_artifact(artifact)


def main() -> None:
    global _ACTIVE_RUN_LIFECYCLE
    args = parse_args()
    if bool(args.require_disjoint_splits) and bool(args.structured_query_conditioning):
        raise ValueError(
            "Formal runs cannot enable adapter.structured_query_conditioning because it uses "
            "regex-parsed task/coordinate features. Disable it so the adapter reads the natural-language question."
        )
    apply_runtime_environment(args)
    device = initialize_distributed_device(str(args.device))
    seed_everything(int(args.seed))
    run_dir = build_distributed_run_dir(args.output_root, args.run_name)
    lifecycle = RunLifecycle(run_dir) if is_main_process() else None
    _ACTIVE_RUN_LIFECYCLE = lifecycle
    if is_main_process():
        dump_json(run_dir / "args_requested.json", redacted_args(args))
        if args.config:
            dump_json(
                run_dir / "config_snapshot.json",
                redacted_config_snapshot(load_yaml_mapping(args.config)),
            )
        print(
            f"run={run_dir.name} started_at={lifecycle.started_at if lifecycle is not None else local_timestamp()} "
            "startup=metadata_audit"
        )
    qa_metadata_audit = broadcast_object_from_rank_zero(
        audit_qa_metadata(args) if is_main_process() else None
    )
    if is_main_process():
        dump_json(run_dir / "qa_metadata_audit.json", qa_metadata_audit)
        print("startup=dataset_index")

    train_dataset = TensorReadoutQADataset(
        qa_path(args.qa_dir, args.train_split),
        latent_dir=args.latent_dir,
        max_records=args.max_train_records,
        prefer_record_latent_ref=bool(args.prefer_record_latent_ref),
        shuffle_seed=int(args.shuffle_seed),
        latent_cache_size=int(args.latent_cache_size),
    )
    val_dataset = TensorReadoutQADataset(
        qa_path(args.qa_dir, args.val_split),
        latent_dir=args.latent_dir,
        max_records=args.max_val_records,
        prefer_record_latent_ref=bool(args.prefer_record_latent_ref),
        shuffle_seed=int(args.shuffle_seed),
        latent_cache_size=int(args.latent_cache_size),
    )
    test_dataset = TensorReadoutQADataset(
        qa_path(args.qa_dir, args.test_split),
        latent_dir=args.latent_dir,
        max_records=args.max_test_records,
        prefer_record_latent_ref=bool(args.prefer_record_latent_ref),
        shuffle_seed=int(args.shuffle_seed),
        latent_cache_size=int(args.latent_cache_size),
    )
    first_latent = train_dataset[0]["latent_map"]
    latent_shape = tuple(int(dim) for dim in first_latent.shape)
    latent_channels = int(latent_shape[0])
    datasets = {"train": train_dataset, "val": val_dataset, "test": test_dataset}
    if is_main_process():
        print(
            f"startup=data_audit train/val/test={len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}"
        )
    data_audit = broadcast_object_from_rank_zero(
        audit_qa_datasets(
            datasets,
            require_disjoint_splits=bool(args.require_disjoint_splits),
        )
        if is_main_process()
        else None
    )
    if is_main_process():
        dump_json(run_dir / "data_audit.json", data_audit)

    if is_main_process():
        print("startup=tokenizer_and_prompt_audit")
    tokenizer = load_tokenizer(args)
    prompt_audit = broadcast_object_from_rank_zero(
        audit_prompt_tokenization(
            datasets=datasets,
            tokenizer=tokenizer,
            max_prompt_tokens=int(args.max_prompt_tokens),
            prompt_template=str(args.prompt_template),
        )
        if is_main_process()
        else None
    )
    choice_tokenization_audit = broadcast_object_from_rank_zero(
        audit_choice_tokenization(datasets, tokenizer) if is_main_process() else None
    )
    choice_tokenization_audit = dict(choice_tokenization_audit)
    configured_choice_mode = str(args.choice_scoring_mode)
    choice_tokenization_audit["configured_mode"] = configured_choice_mode
    choice_tokenization_audit["effective_training_path"] = (
        "sequence_likelihood"
        if configured_choice_mode == "sequence"
        else str(choice_tokenization_audit["training_path"])
    )
    choice_tokenization_audit["effective_evaluation_path"] = (
        "sequence_likelihood"
        if configured_choice_mode == "sequence"
        else str(choice_tokenization_audit["evaluation_path"])
    )
    if is_main_process():
        dump_json(run_dir / "prompt_audit.json", prompt_audit)
        dump_json(run_dir / "choice_tokenization_audit.json", choice_tokenization_audit)
    if configured_choice_mode == "label" and not bool(
        choice_tokenization_audit["all_labels_single_token"]
    ):
        raise ValueError(
            "choice_scoring_mode=label requires all labels to tokenize as one unique token; "
            "see choice_tokenization_audit.json."
        )
    if bool(args.require_untruncated_prompts) and not bool(prompt_audit["all_prompts_fit"]):
        raise ValueError(
            "Prompt audit found main/local prompts longer than "
            f"max_prompt_tokens={int(args.max_prompt_tokens)}. Increase the limit so formal runs do not "
            "silently remove natural-language instructions. See prompt_audit.json."
        )
    pre_load_memory = gather_cuda_memory(device)
    if is_main_process():
        memory_suffix = ""
        if pre_load_memory:
            memory_suffix = " gpu_memory=" + ",".join(
                f"r{int(item['rank'])}:{item['free_gib']:.2f}/{item['total_gib']:.2f}GiB"
                for item in pre_load_memory
            )
            if any(item["free_gib"] < 0.95 * item["total_gib"] for item in pre_load_memory):
                memory_suffix += " warning=visible_gpu_not_empty"
        print(
            f"startup=llm_load visible_cuda={os.environ.get('CUDA_VISIBLE_DEVICES', '<all>')}"
            f"{memory_suffix}"
        )
    llm, model_dtype = load_llm(args, device)
    post_load_memory = gather_cuda_memory(device)
    if is_main_process() and post_load_memory:
        print(
            "startup=llm_loaded gpu_memory="
            + ",".join(
                f"r{int(item['rank'])}:allocated={item['allocated_gib']:.2f}GiB,"
                f"free={item['free_gib']:.2f}/{item['total_gib']:.2f}GiB"
                for item in post_load_memory
            )
            + f" gradient_checkpointing={int(bool(args.llm_gradient_checkpointing))}"
        )
    llm_hidden_size = int(llm.get_input_embeddings().embedding_dim)
    diagnostic_layer_audit = (
        validate_diagnostic_layers(llm, args.diagnostics_layers)
        if bool(args.diagnostics_enabled)
        else {"validated_before_training": False, "reason": "diagnostics disabled"}
    )
    context_layer_values = [int(value) for value in parse_csv(args.local_context_layers)]
    local_context_audit = (
        {
            "validated_before_training": True,
            "layers": [
                validate_local_context_layer(llm, layer_index)
                for layer_index in context_layer_values
            ],
        }
        if str(args.local_question_input_mode) == "contextual_tokens"
        else {"validated_before_training": False, "reason": "input_embeddings mode"}
    )
    if str(args.local_question_input_mode) == "contextual_tokens":
        llm_was_training = bool(llm.training)
        llm.eval()
        preflight_ids, preflight_mask = build_local_question_tensors(
            records=[train_dataset.records[0]],
            tokenizer=tokenizer,
            device=device,
            max_tokens=int(args.max_prompt_tokens),
            prompt_template=str(args.prompt_template),
        )
        preflight_context = contextual_question_token_layers(
            llm=llm,
            input_ids=preflight_ids,
            attention_mask=preflight_mask,
            prompt_mask=preflight_mask.bool(),
            layer_indices=context_layer_values,
        )
        if not bool(torch.isfinite(preflight_context).all()):
            raise ValueError("The local contextual Qwen preflight produced non-finite hidden states.")
        local_context_audit.update(
            {
                "forward_preflight_passed": True,
                "question_tokens": int(preflight_mask.sum().item()),
                "hidden_shape": [int(value) for value in preflight_context.shape],
                "hidden_rms": _rms(preflight_context),
            }
        )
        del preflight_context
        if llm_was_training:
            set_frozen_llm_execution_mode(llm, checkpoint_training=True)

    initialization = "random"
    checkpoint_load_report: dict[str, Any] = {"mode": "random"}
    global_checkpoint_load_report: dict[str, Any] | None = None
    if str(args.adapter_architecture) in {
        "alignment_qformer",
        "alignment_adapter",
        "hybrid_local_qformer",
        "residual_question_qformer",
        "residual_question_adapter",
    }:
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
        raw_checkpoint_state = checkpoint.get("adapter_state_dict") if checkpoint is not None else None
        hybrid_checkpoint = isinstance(raw_checkpoint_state, Mapping) and any(
            str(key).startswith("global_adapter.") for key in raw_checkpoint_state
        )
        if str(args.adapter_architecture) in {
            "residual_question_qformer",
            "residual_question_adapter",
        } and hybrid_checkpoint:
            raise ValueError(
                "Residual question conditioning must start from a stage-1 alignment checkpoint, not a downstream "
                "hybrid checkpoint. Set adapter.init_checkpoint to alignment_best.pt."
            )
        if hybrid_checkpoint:
            global_adapter_type = str(checkpoint_args.get("global_adapter_type", "")).lower()
            if not global_adapter_type:
                global_adapter_type = (
                    "qformer"
                    if isinstance(raw_checkpoint_state.get("global_adapter.query_tokens"), torch.Tensor)
                    else "spatial_transformer"
                    if "global_adapter.spatial_pos_encoding" in raw_checkpoint_state
                    else ""
                )
            if global_adapter_type == "qformer":
                global_query = raw_checkpoint_state.get("global_adapter.query_tokens")
                if not isinstance(global_query, torch.Tensor):
                    raise ValueError("Hybrid Q-Former checkpoint is missing global_adapter.query_tokens.")
                query_tokens = int(global_query.shape[1])
            elif global_adapter_type == "spatial_transformer":
                query_tokens = int(latent_shape[-2]) * int(latent_shape[-1])
            else:
                raise ValueError("Could not infer the global adapter type from the hybrid checkpoint.")
            inferred_global_layers = [
                int(str(key).split(".")[2]) + 1
                for key in raw_checkpoint_state
                if str(key).startswith("global_adapter.blocks.") and str(key).split(".")[2].isdigit()
            ]
            adapter_layers = max(inferred_global_layers or [int(args.adapter_layers)])
        else:
            global_adapter_type = str(checkpoint_args.get("adapter_type", "qformer")).lower()
            query_tokens = (
                int(latent_shape[-2]) * int(latent_shape[-1])
                if global_adapter_type == "spatial_transformer"
                else int(checkpoint_args.get("query_tokens", args.soft_prompt_tokens))
            )
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
        global_dropout = float(checkpoint_args.get("global_dropout", checkpoint_args.get("dropout", args.dropout)))
        global_soft_prompt_scale = float(
            checkpoint_args.get(
                "global_soft_prompt_scale",
                checkpoint_args.get("soft_prompt_scale", args.soft_prompt_scale),
            )
        )
        adapter = TensorPatchAlignmentAdapter(
            latent_channels=latent_channels,
            latent_grid=latent_grid,
            adapter_dim=adapter_dim,
            projection_dim=checkpoint_projection_dim,
            dropout=global_dropout,
            adapter_type=global_adapter_type,
            query_tokens=query_tokens,
            adapter_layers=adapter_layers,
            adapter_heads=adapter_heads,
            soft_prompt_scale=global_soft_prompt_scale,
        ).to(device)
        if checkpoint is not None:
            state_dict = checkpoint.get("adapter_state_dict")
            if not isinstance(state_dict, Mapping):
                raise ValueError("Alignment checkpoint does not contain adapter_state_dict.")
            if str(args.adapter_architecture) == "hybrid_local_qformer" and hybrid_checkpoint:
                hybrid_state_dict = state_dict
                state_dict = {
                    str(key).removeprefix("global_adapter."): value
                    for key, value in state_dict.items()
                    if str(key).startswith("global_adapter.")
                }
            adapter.load_state_dict(state_dict, strict=True)
            global_loaded_parameter_tensors = sum(
                1 for value in state_dict.values() if isinstance(value, torch.Tensor)
            )
            global_loaded_parameters = sum(
                int(value.numel()) for value in state_dict.values() if isinstance(value, torch.Tensor)
            )
            if global_loaded_parameter_tensors <= 0:
                raise ValueError("The adapter checkpoint did not provide any global adapter tensors.")
            global_checkpoint_load_report = {
                "global_loaded_parameter_tensors": global_loaded_parameter_tensors,
                "global_loaded_parameters": global_loaded_parameters,
                "global_strict_load": True,
            }
            checkpoint_load_report = {
                "mode": "strict_global_checkpoint",
                "loaded_parameter_tensors": global_loaded_parameter_tensors,
                "local_loaded_parameter_tensors": 0,
                **global_checkpoint_load_report,
            }
            initialization = "alignment_checkpoint"
            args.adapter_init_checkpoint = init_checkpoint
        else:
            args.adapter_init_checkpoint = None
        args.soft_prompt_tokens = query_tokens
        args.adapter_layers = adapter_layers
        args.adapter_heads = adapter_heads
        args.adapter_dim = adapter_dim
        args.global_adapter_type = global_adapter_type
        args.global_dropout = global_dropout
        args.global_soft_prompt_scale = global_soft_prompt_scale
        if str(args.adapter_architecture) in {"alignment_qformer", "alignment_adapter"}:
            args.question_conditioning = False
            args.structured_query_conditioning = False
        if str(args.adapter_architecture) == "hybrid_local_qformer":
            new_local_architecture = str(args.local_question_input_mode) == "contextual_tokens"
            local_soft_prompt_tokens = int(
                args.local_soft_prompt_tokens
                if new_local_architecture
                else checkpoint_args.get("local_soft_prompt_tokens", args.local_soft_prompt_tokens)
            )
            local_adapter_layers = int(
                args.local_adapter_layers
                if new_local_architecture
                else checkpoint_args.get("local_adapter_layers", args.local_adapter_layers)
            )
            local_gate_init = float(
                args.local_gate_init
                if new_local_architecture
                else checkpoint_args.get("local_gate_init", args.local_gate_init)
            )
            freeze_global_adapter = bool(args.freeze_global_adapter)
            global_adapter = adapter
            local_adapter = QuestionConditionedLocalAdapter(
                latent_channels=latent_channels,
                latent_grid=latent_grid,
                llm_hidden_size=llm_hidden_size,
                adapter_dim=adapter_dim,
                local_tokens=local_soft_prompt_tokens,
                local_layers=local_adapter_layers,
                text_encoder_layers=int(args.local_text_encoder_layers),
                adapter_heads=adapter_heads,
                dropout=float(args.dropout),
                soft_prompt_scale=float(args.soft_prompt_scale),
                gate_init=local_gate_init,
                max_text_tokens=int(args.max_prompt_tokens) + int(args.max_target_tokens),
                structured_query_conditioning=bool(args.structured_query_conditioning),
                question_input_mode=str(args.local_question_input_mode),
                fusion_mode=str(args.local_fusion_mode),
            )
            adapter = HybridGlobalLocalAdapter(
                global_adapter=global_adapter,
                local_adapter=local_adapter,
                freeze_global=freeze_global_adapter,
                global_prompt_dropout=float(args.global_prompt_dropout),
            ).to(device)
            if hybrid_state_dict is not None:
                local_load_report = load_compatible_hybrid_state(adapter, hybrid_state_dict)
                if global_checkpoint_load_report is None:
                    raise RuntimeError("Hybrid warm start did not record a strict global checkpoint load.")
                local_loaded_parameter_tensors = int(
                    local_load_report.get("local_loaded_parameter_tensors", 0)
                )
                checkpoint_load_report = {
                    **local_load_report,
                    **global_checkpoint_load_report,
                    "loaded_parameter_tensors": (
                        int(global_checkpoint_load_report["global_loaded_parameter_tensors"])
                        + local_loaded_parameter_tensors
                    ),
                }
                initialization = "hybrid_compatible_warm_start"
            elif checkpoint is not None:
                checkpoint_load_report["mode"] = "strict_global_alignment_checkpoint"
            args.soft_prompt_tokens = int(adapter.soft_prompt_tokens)
            args.local_soft_prompt_tokens = local_soft_prompt_tokens
            args.local_adapter_layers = local_adapter_layers
            args.local_question_input_mode = str(local_adapter.question_input_mode)
            args.local_fusion_mode = str(local_adapter.fusion_mode)
            args.local_gate_init = local_gate_init
            args.freeze_global_adapter = freeze_global_adapter
            args.question_conditioning = True
            args.structured_query_conditioning = bool(adapter.structured_query_conditioning)
        if str(args.adapter_architecture) in {
            "residual_question_qformer",
            "residual_question_adapter",
        }:
            if checkpoint is None:
                raise ValueError("Residual question conditioning requires adapter.init_checkpoint from stage 1.")
            context_layers = [int(value) for value in parse_csv(args.local_context_layers)]
            global_adapter = adapter
            local_adapter = ResidualQuestionConditionedAdapter(
                aligned_adapter=global_adapter,
                llm_hidden_size=llm_hidden_size,
                context_layers=context_layers,
                adapter_heads=adapter_heads,
                dropout=float(args.dropout),
                text_gate_init=float(args.local_text_gate_init),
                residual_gate_init=float(args.local_gate_init),
            )
            adapter = HybridGlobalLocalAdapter(
                global_adapter=global_adapter,
                local_adapter=local_adapter,
                freeze_global=True,
                global_prompt_dropout=float(args.global_prompt_dropout),
                combine_mode="residual",
            ).to(device)
            checkpoint_load_report["mode"] = "stage1_cloned_residual_aligned_adapter"
            checkpoint_load_report["conditioned_backbone_initialized_parameters"] = sum(
                int(parameter.numel()) for parameter in local_adapter.backbone.parameters()
            )
            initialization = "stage1_residual_question_adapter"
            args.soft_prompt_tokens = int(adapter.soft_prompt_tokens)
            args.local_soft_prompt_tokens = int(local_adapter.soft_prompt_tokens)
            args.local_adapter_layers = int(len(local_adapter.backbone.blocks))
            args.local_question_input_mode = "contextual_tokens"
            args.local_context_layer = int(max(context_layers))
            args.local_fusion_mode = str(local_adapter.fusion_mode)
            args.freeze_global_adapter = True
            args.global_unfreeze_epoch = 0
            args.question_conditioning = True
            args.structured_query_conditioning = False
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

    synchronize_module_from_rank_zero(adapter)
    seed_everything(int(args.seed) + distributed_rank())
    train_epoch_sampler: Any = None
    if bool(args.group_questions_by_state):
        train_epoch_sampler = StateTaskGroupedBatchSampler(
            dataset=train_dataset,
            batch_size=int(args.batch_size),
            questions_per_group=int(args.questions_per_state_group),
            seed=int(args.shuffle_seed),
            rank=distributed_rank(),
            num_replicas=distributed_world_size(),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_epoch_sampler,
            num_workers=int(args.num_workers),
            persistent_workers=int(args.num_workers) > 0,
            pin_memory=device.type == "cuda",
            collate_fn=collate_tensor_readout,
        )
    else:
        if distributed_is_initialized():
            train_epoch_sampler = DistributedSampler(
                train_dataset,
                num_replicas=distributed_world_size(),
                rank=distributed_rank(),
                shuffle=True,
                seed=int(args.shuffle_seed),
                drop_last=False,
            )
        train_loader = DataLoader(
            train_dataset,
            batch_size=max(1, int(args.batch_size)),
            shuffle=train_epoch_sampler is None,
            sampler=train_epoch_sampler,
            num_workers=int(args.num_workers),
            persistent_workers=int(args.num_workers) > 0,
            pin_memory=device.type == "cuda",
            collate_fn=collate_tensor_readout,
        )
    if isinstance(adapter, HybridGlobalLocalAdapter):
        local_groups = adamw_parameter_groups(
            adapter.local_adapter,
            learning_rate=float(args.lr),
            weight_decay=float(args.weight_decay),
            name="local",
        )
        global_groups = adamw_parameter_groups(
            adapter.global_adapter,
            learning_rate=float(args.global_lr),
            weight_decay=float(args.weight_decay),
            name="global",
            include_frozen=not adapter.residual_mode,
        )
        optimizer = torch.optim.AdamW(
            local_groups + global_groups,
        )
    else:
        optimizer = torch.optim.AdamW(
            adamw_parameter_groups(
                adapter,
                learning_rate=float(args.lr),
                weight_decay=float(args.weight_decay),
                name="adapter",
            ),
        )
    accumulation_steps = max(1, int(args.gradient_accumulation_steps))
    updates_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
    total_optimizer_updates = max(1, updates_per_epoch * int(args.epochs))
    lr_scheduler, warmup_updates = build_lr_scheduler(
        optimizer=optimizer,
        scheduler_name=str(args.lr_scheduler),
        total_updates=total_optimizer_updates,
        warmup_ratio=float(args.warmup_ratio),
        min_lr_ratio=float(args.min_lr_ratio),
    )
    baseline_modes = parse_csv(args.eval_baselines)
    if not baseline_modes:
        baseline_modes = ["correct"]
    final_baseline_modes = parse_csv(args.final_eval_baselines)
    if not final_baseline_modes:
        final_baseline_modes = list(baseline_modes)

    summary = {
        "device": str(device),
        "distributed": {
            "enabled": distributed_is_initialized(),
            "backend": dist.get_backend() if distributed_is_initialized() else None,
            "world_size": distributed_world_size(),
            "rank": distributed_rank(),
            "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
            "per_rank_batch_size": int(args.batch_size),
            "train_choice_batch_size": int(args.train_choice_batch_size),
            "train_grounding_batch_size": int(args.train_grounding_batch_size),
            "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
            "effective_train_batch_size": (
                int(args.batch_size)
                * distributed_world_size()
                * int(args.gradient_accumulation_steps)
            ),
            "gradient_sync": "manual_all_reduce_adapter_only",
            "evaluation_sharding": "exact_nonpadding",
        },
        "grouped_batch_size_epoch_zero": (
            {
                "configured_max": int(args.batch_size),
                "minimum": int(train_epoch_sampler.initial_batch_size_min),
                "maximum": int(train_epoch_sampler.initial_batch_size_max),
                "mean": float(train_epoch_sampler.initial_batch_size_mean),
            }
            if isinstance(train_epoch_sampler, StateTaskGroupedBatchSampler)
            else None
        ),
        "model_dtype": str(model_dtype).replace("torch.", ""),
        "llm_gradient_checkpointing": bool(args.llm_gradient_checkpointing),
        "llm_hidden_size": llm_hidden_size,
        "latent_shape_chw": list(latent_shape),
        "train_records": len(train_dataset),
        "val_records": len(val_dataset),
        "test_records": len(test_dataset),
        "latent_cache_size": int(args.latent_cache_size),
        "num_workers": int(args.num_workers),
        "shuffle_seed": int(args.shuffle_seed),
        "shuffled_negative_policy": "same_field_task_different_sample_then_fallback",
        "ce_loss_weight": float(args.ce_loss_weight),
        "choice_ce_loss_weight": float(args.choice_ce_loss_weight),
        "ranking_loss_weight": float(args.ranking_loss_weight),
        "ranking_loss_margin": float(args.ranking_loss_margin),
        "ranking_loss_negative": str(args.ranking_loss_negative),
        "swapped_question_loss_weight": float(args.swapped_question_loss_weight),
        "swapped_question_loss_margin": float(args.swapped_question_loss_margin),
        "swapped_question_max_records": int(args.swapped_question_max_records),
        "soft_prompt_tokens": int(args.soft_prompt_tokens),
        "adapter_layers": int(args.adapter_layers),
        "latent_pos_encoding": str(args.latent_pos_encoding),
        "question_conditioning": bool(args.question_conditioning),
        "question_condition_gate_init": float(args.question_condition_gate_init),
        "structured_query_conditioning": bool(args.structured_query_conditioning),
        "question_input_mode": (
            "latent_only_global"
            if str(args.adapter_architecture) in {"alignment_qformer", "alignment_adapter"}
            else (
                "legacy_parsed_features"
                if bool(args.structured_query_conditioning)
                else str(args.local_question_input_mode)
            )
        ),
        "local_context_layer": int(args.local_context_layer),
        "local_context_layers": [int(value) for value in parse_csv(args.local_context_layers)],
        "local_fusion_mode": str(args.local_fusion_mode),
        "soft_prompt_scale": float(args.soft_prompt_scale),
        "adapter_architecture": str(args.adapter_architecture),
        "global_adapter_type": str(getattr(args, "global_adapter_type", "legacy")),
        "adapter_initialization": initialization,
        "adapter_init_checkpoint": str(args.adapter_init_checkpoint) if args.adapter_init_checkpoint else None,
        "local_soft_prompt_tokens": int(args.local_soft_prompt_tokens),
        "local_adapter_layers": int(args.local_adapter_layers),
        "local_text_encoder_layers": int(args.local_text_encoder_layers),
        "local_gate_init": float(args.local_gate_init),
        "local_text_gate_init": float(args.local_text_gate_init),
        "freeze_global_adapter": bool(args.freeze_global_adapter),
        "global_unfreeze_epoch": int(args.global_unfreeze_epoch),
        "global_lr": float(args.global_lr),
        "global_prompt_dropout": float(args.global_prompt_dropout),
        "group_questions_by_state": bool(args.group_questions_by_state),
        "questions_per_state_group": int(args.questions_per_state_group),
        "lr_scheduler": str(args.lr_scheduler),
        "warmup_updates": int(warmup_updates),
        "total_optimizer_updates": int(total_optimizer_updates),
        "min_lr_ratio": float(args.min_lr_ratio),
        "global_dropout": float(getattr(args, "global_dropout", args.dropout)),
        "global_soft_prompt_scale": float(
            getattr(args, "global_soft_prompt_scale", args.soft_prompt_scale)
        ),
        "checkpoint_metric": str(args.checkpoint_metric),
        "diagnostics_generation_max_new_tokens": int(args.diagnostics_generation_max_new_tokens),
        "checkpoint_load_report": checkpoint_load_report,
        "qa_metadata_audit": qa_metadata_audit,
        "data_audit": data_audit,
        "prompt_audit": prompt_audit,
        "choice_tokenization_audit": choice_tokenization_audit,
        "choice_scoring_mode": str(args.choice_scoring_mode),
        "diagnostic_layer_audit": diagnostic_layer_audit,
        "local_context_audit": local_context_audit,
        "adapter_parameters_total": sum(p.numel() for p in adapter.parameters()),
        "trainable_adapter_parameters": sum(p.numel() for p in adapter.parameters() if p.requires_grad),
        "local_adapter_parameters": (
            sum(p.numel() for p in adapter.local_adapter.parameters())
            if isinstance(adapter, HybridGlobalLocalAdapter)
            else None
        ),
        "global_adapter_parameters": (
            sum(p.numel() for p in adapter.global_adapter.parameters())
            if isinstance(adapter, HybridGlobalLocalAdapter)
            else None
        ),
        "frozen_llm_parameters": sum(p.numel() for p in llm.parameters()),
    }
    if is_main_process():
        dump_json(run_dir / "args.json", redacted_args(args))
        dump_json(run_dir / "run_summary.json", summary)
        if lifecycle is None:
            raise RuntimeError("Rank 0 did not create a run lifecycle.")
        lifecycle._write("running")
        print(
            f"run={run_dir.name} started_at={lifecycle.started_at} device={device} "
            f"distributed={int(distributed_is_initialized())} world_size={distributed_world_size()} "
            f"train/val/test={len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)} "
            f"question_input={summary['question_input_mode']} fusion={summary['local_fusion_mode']} "
            f"scheduler={summary['lr_scheduler']} grouped={int(summary['group_questions_by_state'])} "
            f"effective_batch={summary['distributed']['effective_train_batch_size']} "
            f"eval_batch={int(args.eval_batch_size)} "
            f"grouped_batch_range={summary['grouped_batch_size_epoch_zero']} "
            f"grounding_forward_batch={int(args.train_grounding_batch_size)} "
            f"global_dropout={summary['global_prompt_dropout']:.2f} "
            f"params={summary['trainable_adapter_parameters']:,} "
            f"prompt_max={max(item['max_tokens'] for item in prompt_audit['splits'].values())} "
            f"local_prompt_max={max(item['local_max_tokens'] for item in prompt_audit['splits'].values())} "
            f"loss_weights=ce:{float(args.ce_loss_weight):g},choice:{float(args.choice_ce_loss_weight):g},"
            f"ranking:{float(args.ranking_loss_weight):g},swap:{float(args.swapped_question_loss_weight):g} "
            f"choice_path={choice_tokenization_audit['effective_training_path']} "
            f"checkpoint_load={checkpoint_load_report.get('mode')} "
            f"global/local_tensors={int(checkpoint_load_report.get('global_loaded_parameter_tensors', 0))}/"
            f"{int(checkpoint_load_report.get('local_loaded_parameter_tensors', 0))}"
        )
    wandb_config = build_wandb_config(args, summary)
    if not is_main_process():
        wandb_config["wandb"]["enabled"] = False
    wandb_logger = WandbLogger(config=wandb_config, run_dir=run_dir)

    best_val_score = -math.inf
    best_epoch = 0
    history: dict[str, Any] = {}
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
                latent_cache_size=int(args.latent_cache_size),
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
            if is_main_process():
                dump_json(metrics_path, history)
            initial_payload = (
                flatten_numeric_metrics("initial_eval", initial_metrics)
                if bool(args.wandb_detailed_metrics)
                else compact_accuracy_metrics("initial_eval", initial_metrics)
            )
            wandb_logger.log(initial_payload, step=0)
            if is_main_process():
                print_evaluation_summary("initial_eval", initial_metrics, metrics_path)
            if bool(args.diagnostics_enabled) and is_main_process():
                pretrain_diagnostic = run_embedded_diagnostics(
                    stage="pretrain",
                    llm=llm,
                    adapter=adapter,
                    tokenizer=tokenizer,
                    dataset=val_dataset,
                    device=device,
                    args=args,
                    run_dir=run_dir,
                )
                history["pretrain_diagnostics"] = dict(pretrain_diagnostic["aggregate"])
                dump_json(metrics_path, history)
                wandb_logger.log(
                    compact_diagnostic_metrics(pretrain_diagnostic["aggregate"]), step=0
                )
            distributed_barrier()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        for epoch in range(1, int(args.epochs) + 1):
            if train_epoch_sampler is not None and hasattr(train_epoch_sampler, "set_epoch"):
                train_epoch_sampler.set_epoch(epoch - 1)
            if (
                isinstance(adapter, HybridGlobalLocalAdapter)
                and adapter.freeze_global
                and not adapter.residual_mode
                and int(args.global_unfreeze_epoch) > 0
                and epoch >= int(args.global_unfreeze_epoch)
            ):
                adapter.set_global_trainable(True)
                if is_main_process():
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
            running_swapped_question_loss = 0.0
            running_weighted_swapped_question_loss = 0.0
            running_swapped_question_margin = 0.0
            running_swapped_question_pairs = 0.0
            running_total_grad_norm = 0.0
            running_local_grad_norm = 0.0
            running_global_grad_norm = 0.0
            running_global_dropout_batches = 0
            optimizer_update_count = 0
            optimizer.zero_grad(set_to_none=True)
            progress = tqdm(
                train_loader,
                desc=f"Epoch {epoch:03d} [train]",
                disable=not bool(args.console_progress) or not is_main_process(),
            )
            final_accumulation_steps = len(train_loader) % accumulation_steps
            for step, batch in enumerate(progress, start=1):
                drop_global_for_batch = bool(
                    isinstance(adapter, HybridGlobalLocalAdapter)
                    and float(args.global_prompt_dropout) > 0.0
                    and random.random() < float(args.global_prompt_dropout)
                )
                if isinstance(adapter, HybridGlobalLocalAdapter):
                    adapter.set_global_prompt_dropout_for_batch(drop_global_for_batch)
                try:
                    loss, loss_parts = training_loss(
                        llm=llm,
                        adapter=adapter,
                        tokenizer=tokenizer,
                        dataset=train_dataset,
                        batch=batch,
                        device=device,
                        args=args,
                    )
                finally:
                    if isinstance(adapter, HybridGlobalLocalAdapter):
                        adapter.set_global_prompt_dropout_for_batch(False)
                running_global_dropout_batches += int(drop_global_for_batch)
                accumulation_divisor = (
                    final_accumulation_steps
                    if final_accumulation_steps > 0 and step > len(train_loader) - final_accumulation_steps
                    else accumulation_steps
                )
                (loss / accumulation_divisor).backward()
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
                running_swapped_question_loss += float(loss_parts["swapped_question_loss"])
                running_weighted_swapped_question_loss += float(loss_parts["weighted_swapped_question_loss"])
                running_swapped_question_margin += float(loss_parts["swapped_question_margin_mean"])
                running_swapped_question_pairs += float(loss_parts["swapped_question_pairs"])

                if step % accumulation_steps == 0 or step == len(train_loader):
                    average_trainable_gradients(adapter)
                    if isinstance(adapter, HybridGlobalLocalAdapter):
                        local_grad_norm = gradient_l2_norm(adapter.local_adapter.parameters())
                        global_grad_norm = gradient_l2_norm(adapter.global_adapter.parameters())
                        total_grad_norm = math.hypot(local_grad_norm, global_grad_norm)
                    else:
                        total_grad_norm = gradient_l2_norm(adapter.parameters())
                        local_grad_norm = total_grad_norm
                        global_grad_norm = 0.0
                    if float(args.grad_clip_norm) > 0:
                        torch.nn.utils.clip_grad_norm_(adapter.parameters(), float(args.grad_clip_norm))
                    running_total_grad_norm += total_grad_norm
                    running_local_grad_norm += local_grad_norm
                    running_global_grad_norm += global_grad_norm
                    optimizer_update_count += 1
                    optimizer.step()
                    lr_scheduler.step()
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
                average_swapped_question_loss = running_swapped_question_loss / step
                average_swapped_question_margin = running_swapped_question_margin / step
                average_total_grad_norm = running_total_grad_norm / max(1, optimizer_update_count)
                average_local_grad_norm = running_local_grad_norm / max(1, optimizer_update_count)
                average_global_grad_norm = running_global_grad_norm / max(1, optimizer_update_count)
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
                        "train_swapped_question_loss": average_swapped_question_loss,
                        "train_swapped_question_margin": average_swapped_question_margin,
                        "train_total_grad_norm": average_total_grad_norm,
                        "train_local_grad_norm": average_local_grad_norm,
                        "train_global_grad_norm": average_global_grad_norm,
                    }
                    if bool(args.save_step_metrics) and is_main_process():
                        history[f"epoch_{epoch:04d}_step_{step:06d}"] = step_payload
                        dump_json(run_dir / "metrics_latest.json", history)
                    wandb_logger.log(
                        {
                            "train_step/loss": average_loss,
                            "train_step/choice_ce_loss": average_choice_ce_loss,
                            "train_step/choice_accuracy": average_choice_accuracy,
                            "train_step/ranking_loss": average_ranking_loss,
                            "train_step/ranking_margin": average_ranking_margin,
                            "train_step/swapped_question_loss": average_swapped_question_loss,
                            "train_step/swapped_question_margin": average_swapped_question_margin,
                            "train_step/lr": optimizer_group_lr(optimizer, "local", float(args.lr)),
                        },
                        step=global_step,
                    )

            train_totals = distributed_sum_scalars(
                {
                    "loss": running_loss,
                    "ce_loss": running_ce_loss,
                    "weighted_ce_loss": running_weighted_ce_loss,
                    "choice_ce_loss": running_choice_ce_loss,
                    "weighted_choice_ce_loss": running_weighted_choice_ce_loss,
                    "choice_accuracy": running_choice_accuracy,
                    "choice_01_loss": running_choice_01_loss,
                    "ranking_loss": running_ranking_loss,
                    "weighted_ranking_loss": running_weighted_ranking_loss,
                    "ranking_margin": running_ranking_margin,
                    "swapped_question_loss": running_swapped_question_loss,
                    "weighted_swapped_question_loss": running_weighted_swapped_question_loss,
                    "swapped_question_margin": running_swapped_question_margin,
                    "swapped_question_pairs": running_swapped_question_pairs,
                    "total_grad_norm": running_total_grad_norm,
                    "local_grad_norm": running_local_grad_norm,
                    "global_grad_norm": running_global_grad_norm,
                    "global_dropout_batches": float(running_global_dropout_batches),
                    "batch_count": float(len(train_loader)),
                    "optimizer_update_count": float(optimizer_update_count),
                },
                device=device,
            )
            global_batch_count = max(1.0, train_totals["batch_count"])
            global_update_count = max(1.0, train_totals["optimizer_update_count"])
            train_loss = train_totals["loss"] / global_batch_count
            train_ce_loss = train_totals["ce_loss"] / global_batch_count
            train_weighted_ce_loss = train_totals["weighted_ce_loss"] / global_batch_count
            train_choice_ce_loss = train_totals["choice_ce_loss"] / global_batch_count
            train_weighted_choice_ce_loss = (
                train_totals["weighted_choice_ce_loss"] / global_batch_count
            )
            train_choice_accuracy = train_totals["choice_accuracy"] / global_batch_count
            train_choice_01_loss = train_totals["choice_01_loss"] / global_batch_count
            train_ranking_loss = train_totals["ranking_loss"] / global_batch_count
            train_weighted_ranking_loss = (
                train_totals["weighted_ranking_loss"] / global_batch_count
            )
            train_ranking_margin = train_totals["ranking_margin"] / global_batch_count
            train_swapped_question_loss = (
                train_totals["swapped_question_loss"] / global_batch_count
            )
            train_weighted_swapped_question_loss = (
                train_totals["weighted_swapped_question_loss"] / global_batch_count
            )
            train_swapped_question_margin = (
                train_totals["swapped_question_margin"] / global_batch_count
            )
            train_swapped_question_pairs = (
                train_totals["swapped_question_pairs"] / global_batch_count
            )
            train_total_grad_norm = train_totals["total_grad_norm"] / global_update_count
            train_local_grad_norm = train_totals["local_grad_norm"] / global_update_count
            train_global_grad_norm = train_totals["global_grad_norm"] / global_update_count
            train_global_dropout_rate = (
                train_totals["global_dropout_batches"] / global_batch_count
            )
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
                "train_swapped_question_loss": train_swapped_question_loss,
                "train_weighted_swapped_question_loss": train_weighted_swapped_question_loss,
                "train_swapped_question_margin": train_swapped_question_margin,
                "train_swapped_question_pairs_per_batch": train_swapped_question_pairs,
                "train_total_grad_norm": train_total_grad_norm,
                "train_local_grad_norm": train_local_grad_norm,
                "train_global_grad_norm": train_global_grad_norm,
                "train_global_prompt_dropout_rate": train_global_dropout_rate,
                "val": val_metrics,
            }
            history[f"epoch_{epoch:04d}"] = epoch_payload
            if is_main_process():
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
                "train/swapped_question_loss": float(train_swapped_question_loss),
                "train/swapped_question_margin": float(train_swapped_question_margin),
                "train/swapped_question_pairs_per_batch": float(train_swapped_question_pairs),
                "train/total_grad_norm": float(train_total_grad_norm),
                "train/local_grad_norm": float(train_local_grad_norm),
                "train/global_grad_norm": float(train_global_grad_norm),
                "train/global_prompt_dropout_rate": float(train_global_dropout_rate),
                "adapter/local_gate": float(
                    adapter.local_adapter.gate.detach().float().cpu().item()
                    if isinstance(adapter, HybridGlobalLocalAdapter)
                    else 0.0
                ),
                "adapter/local_anchor_gate": float(
                    adapter.local_adapter.anchor_gate.detach().float().cpu().item()
                    if isinstance(adapter, HybridGlobalLocalAdapter)
                    and getattr(adapter.local_adapter, "anchor_gate", None) is not None
                    else 0.0
                ),
                "lr": optimizer_group_lr(optimizer, "local", float(args.lr)),
                "local_lr": optimizer_group_lr(optimizer, "local", float(args.lr)),
                "global_lr": optimizer_group_lr(optimizer, "global", 0.0),
                "global_trainable": float(
                    isinstance(adapter, HybridGlobalLocalAdapter) and not adapter.freeze_global
                ),
                "val/macro_latent_gain": float(val_macro_latent_gain),
                "best_val/checkpoint_score": float(max(best_val_score, val_score)),
            }
            wandb_payload.update(
                flatten_numeric_metrics("val", val_metrics)
                if bool(args.wandb_detailed_metrics)
                else compact_accuracy_metrics("val", val_metrics)
            )
            if val_score > best_val_score:
                best_val_score = val_score
                best_epoch = epoch
                if is_main_process():
                    save_adapter_checkpoint(
                        run_dir / "adapter_best.pt",
                        adapter=adapter,
                        args=args,
                        latent_shape=latent_shape,
                        llm_hidden_size=llm_hidden_size,
                        metrics=epoch_payload,
                    )
            diagnostic_suffix = ""
            if (
                bool(args.diagnostics_enabled)
                and int(args.diagnostics_every_epochs) > 0
                and epoch % int(args.diagnostics_every_epochs) == 0
                and is_main_process()
            ):
                diagnostic_summary = run_embedded_diagnostics(
                    stage=f"epoch_{epoch:04d}",
                    llm=llm,
                    adapter=adapter,
                    tokenizer=tokenizer,
                    dataset=val_dataset,
                    device=device,
                    args=args,
                    run_dir=run_dir,
                )
                diagnostic_aggregate = dict(diagnostic_summary["aggregate"])
                epoch_payload["diagnostics"] = diagnostic_aggregate
                history[f"epoch_{epoch:04d}"] = epoch_payload
                dump_json(run_dir / "metrics_latest.json", history)
                wandb_payload.update(
                    flatten_numeric_metrics("diagnostics", diagnostic_aggregate)
                    if bool(args.wandb_detailed_metrics)
                    else compact_diagnostic_metrics(diagnostic_aggregate)
                )
                diagnostic_suffix = (
                    f" diag_local_l2={float(diagnostic_aggregate['local_prompt_relative_l2_mean']):.4f}"
                    f" diag_margin_gain={float(diagnostic_aggregate['answer_margin_correct_minus_shuffled']):.4f}"
                )
            distributed_barrier()
            wandb_logger.log(wandb_payload, step=global_step)
            if is_main_process():
                print(
                    f"epoch={epoch:03d}/{int(args.epochs):03d} "
                    f"loss={train_loss:.4f} train_acc={train_choice_accuracy:.4f} "
                    f"val={val_accuracy:.4f} "
                    f"shuffled={float(val_metrics.get('shuffled', {}).get('accuracy', 0.0)):.4f} "
                    f"macro_gain={val_macro_latent_gain:.4f} best_epoch={best_epoch}"
                    f"{diagnostic_suffix}"
                )

        distributed_barrier()
        best_checkpoint_path = run_dir / "adapter_best.pt"
        if not best_checkpoint_path.exists():
            raise FileNotFoundError("Training completed without producing adapter_best.pt.")
        best_checkpoint = torch.load(best_checkpoint_path, map_location="cpu")
        best_state_dict = best_checkpoint.get("adapter_state_dict")
        if not isinstance(best_state_dict, Mapping):
            raise ValueError("adapter_best.pt does not contain adapter_state_dict.")
        adapter.load_state_dict(best_state_dict, strict=True)
        adapter.to(device)
        if is_main_process():
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
            baseline_modes=final_baseline_modes,
        )
        if is_main_process():
            dump_json(run_dir / "test_metrics.json", test_metrics)
        test_payload = (
            flatten_numeric_metrics("test", test_metrics)
            if bool(args.wandb_detailed_metrics)
            else compact_accuracy_metrics("test", test_metrics)
        )
        wandb_logger.log(test_payload, step=global_step + 1)
        if is_main_process():
            summary["result"] = {
                "best_epoch": int(best_epoch),
                "best_val_score": float(best_val_score),
                "checkpoint_metric": str(args.checkpoint_metric),
                "test_correct_accuracy": float(
                    test_metrics.get("correct", {}).get("accuracy", 0.0)
                ),
                "test_shuffled_accuracy": float(
                    test_metrics.get("shuffled", {}).get("accuracy", 0.0)
                ),
                "test_correct_by_task": dict(
                    test_metrics.get("correct", {}).get("by_task", {})
                ),
            }
            dump_json(run_dir / "run_summary.json", summary)
            if bool(args.wandb_log_model):
                log_adapter_artifact(
                    wandb_logger,
                    run_dir / "adapter_best.pt",
                    f"{args.run_name}-best",
                )
                log_adapter_artifact(
                    wandb_logger,
                    run_dir / "adapter_last.pt",
                    f"{args.run_name}-last",
                )
            print(f"run_dir: {run_dir}")
            print_evaluation_summary("test", test_metrics, run_dir / "test_metrics.json")
    finally:
        wandb_logger.finish()
    distributed_barrier()
    if is_main_process():
        if lifecycle is None:
            raise RuntimeError("Rank 0 lost its run lifecycle before completion.")
        timing = lifecycle.finish("completed")
        print(
            f"completed_at={timing['ended_at']} duration_seconds={timing['duration_seconds']} "
            f"timing_file={run_dir / 'run_timing.json'}"
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as exc:
        if _ACTIVE_RUN_LIFECYCLE is not None:
            _ACTIVE_RUN_LIFECYCLE.finish("interrupted", exc)
        raise
    except BaseException as exc:
        if _ACTIVE_RUN_LIFECYCLE is not None:
            _ACTIVE_RUN_LIFECYCLE.finish("failed", exc)
        raise
    finally:
        if distributed_is_initialized():
            dist.destroy_process_group()
