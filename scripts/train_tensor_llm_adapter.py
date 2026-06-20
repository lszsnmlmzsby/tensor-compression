from __future__ import annotations

import argparse
import json
import math
import os
import random
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
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tensor_compression.downstream.pdebench import resolve_device  # noqa: E402
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

    def forward(self, queries: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        attended, _weights = self.attention(
            query=self.query_norm(queries),
            key=self.latent_norm(latents),
            value=self.latent_norm(latents),
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
    ) -> None:
        super().__init__()
        self.soft_prompt_tokens = int(soft_prompt_tokens)
        self.input_projection = nn.Linear(int(latent_channels), int(adapter_dim))
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

    def forward(self, latent_map: torch.Tensor) -> torch.Tensor:
        if latent_map.ndim == 4:
            latent_tokens = latent_map.flatten(2).transpose(1, 2).contiguous()
        elif latent_map.ndim == 3:
            latent_tokens = latent_map
        else:
            raise ValueError(f"Expected latent_map [B,C,H,W] or latent_tokens [B,N,C], got {latent_map.shape}.")
        latent_tokens = latent_tokens.to(dtype=self.input_projection.weight.dtype)
        latents = self.input_projection(latent_tokens)
        queries = self.query_tokens.expand(latent_map.shape[0], -1, -1)
        for block in self.blocks:
            queries = block(queries, latents)
        return self.output_projection(self.output_norm(queries))


class TensorReadoutQADataset(Dataset):
    def __init__(
        self,
        jsonl_path: str | Path,
        latent_dir: str | Path,
        max_records: int | None = None,
        prefer_record_latent_ref: bool = False,
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.latent_dir = Path(latent_dir)
        self.prefer_record_latent_ref = bool(prefer_record_latent_ref)
        self.records = self._load_records(self.jsonl_path)
        if max_records is not None:
            self.records = self.records[: max(0, int(max_records))]
        if not self.records:
            raise RuntimeError(f"No QA records found in {self.jsonl_path}.")
        self._next_different_indices = self._build_next_different_indices()

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

    def _build_next_different_indices(self) -> list[int]:
        indices: list[int] = []
        total = len(self.records)
        for index, record in enumerate(self.records):
            state_ref = str(record.get("state_ref", ""))
            candidate = (index + 1) % total
            attempts = 0
            while attempts < total and str(self.records[candidate].get("state_ref", "")) == state_ref:
                candidate = (candidate + 1) % total
                attempts += 1
            indices.append(candidate)
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
        other_index = self._next_different_indices[int(index)]
        return self.load_latent_for_record(self.records[other_index])


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
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--eval-choice-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    parser.add_argument("--soft-prompt-tokens", type=int, default=None)
    parser.add_argument("--adapter-dim", type=int, default=None)
    parser.add_argument("--adapter-layers", type=int, default=None)
    parser.add_argument("--adapter-heads", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--max-prompt-tokens", type=int, default=None)
    parser.add_argument("--max-target-tokens", type=int, default=None)
    parser.add_argument("--append-eos", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--eval-baselines",
        type=str,
        default=None,
        help="Comma-separated: correct,no_latent,shuffled,random.",
    )
    parser.add_argument(
        "--choice-score",
        type=str,
        default=None,
        choices=("mean", "sum"),
        help="Normalize candidate NLL by target-token count or not.",
    )
    parser.add_argument("--log-interval", type=int, default=None)
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
        "qa_dir": first_nested(config, ["data.qa_dir"]),
        "latent_dir": first_nested(config, ["data.latent_dir", "latent_export.output_dir"]),
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
    set_default(args, "soft_prompt_tokens", first_nested(config, ["adapter.soft_prompt_tokens"]), 32)
    set_default(args, "adapter_dim", first_nested(config, ["adapter.adapter_dim"]), 512)
    set_default(args, "adapter_layers", first_nested(config, ["adapter.adapter_layers"]), 2)
    set_default(args, "adapter_heads", first_nested(config, ["adapter.adapter_heads"]), 8)
    set_default(args, "dropout", first_nested(config, ["adapter.dropout"]), 0.1)
    set_default(args, "max_prompt_tokens", first_nested(config, ["llm_training.max_prompt_tokens"]), 192)
    set_default(args, "max_target_tokens", first_nested(config, ["llm_training.max_target_tokens"]), 8)
    set_default(args, "append_eos", first_nested(config, ["llm_training.append_eos"]), True)
    set_default(
        args,
        "eval_baselines",
        value_to_csv(first_nested(config, ["llm_training.eval_baselines"])),
        "correct,no_latent,shuffled",
    )
    set_default(args, "choice_score", first_nested(config, ["llm_training.choice_score"]), "mean")
    set_default(args, "log_interval", first_nested(config, ["llm_training.log_interval"]), 20)
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


def build_prompt(record: Mapping[str, Any]) -> str:
    query = str(record.get("query") or record.get("question") or "")
    choices = record.get("choices")
    if not isinstance(choices, Sequence) or isinstance(choices, str):
        choices = []
    choice_text = ", ".join(str(choice) for choice in choices)
    return (
        "Tensor-state soft tokens are prepended before this text.\n"
        "Answer the tensor readout query with exactly one choice label.\n\n"
        f"Query: {query}\n"
        f"Choices: {choice_text}\n"
        "Answer:"
    )


def encode_example(
    record: Mapping[str, Any],
    answer: str,
    tokenizer,
    max_prompt_tokens: int,
    max_target_tokens: int,
    append_eos: bool,
) -> tuple[list[int], list[int]]:
    prompt_ids = tokenizer(
        build_prompt(record),
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    encoded = [
        encode_example(
            record=record,
            answer=answer,
            tokenizer=tokenizer,
            max_prompt_tokens=max_prompt_tokens,
            max_target_tokens=max_target_tokens,
            append_eos=append_eos,
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
    mode: str,
) -> torch.Tensor:
    if mode == "correct":
        return adapter(latent_map).to(dtype=text_embeds.dtype)
    if mode == "no_latent":
        batch_size = latent_map.shape[0]
        return text_embeds.new_zeros((batch_size, adapter.soft_prompt_tokens, text_embeds.shape[-1]))
    if mode in {"shuffled", "random"}:
        return adapter(latent_map).to(dtype=text_embeds.dtype)
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
    soft_prompt_mode: str = "correct",
) -> torch.Tensor:
    input_ids, text_attention_mask, text_labels = build_text_tensors(
        records=records,
        answers=answers,
        tokenizer=tokenizer,
        max_prompt_tokens=max_prompt_tokens,
        max_target_tokens=max_target_tokens,
        append_eos=append_eos,
    )
    input_ids = input_ids.to(device)
    text_attention_mask = text_attention_mask.to(device)
    text_labels = text_labels.to(device)
    latent_map = latent_map.to(device)

    text_embeds = llm.get_input_embeddings()(input_ids)
    soft_embeds = adapter_soft_embeds(adapter, latent_map, text_embeds, soft_prompt_mode)
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
    outputs = llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
    return outputs.loss


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
    )
    input_ids = input_ids.to(device)
    text_attention_mask = text_attention_mask.to(device)
    text_labels = text_labels.to(device)
    latent_map = latent_map.to(device)

    text_embeds = llm.get_input_embeddings()(input_ids)
    soft_embeds = adapter_soft_embeds(adapter, latent_map, text_embeds, soft_prompt_mode)
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

    logits = llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    target_mask = shift_labels.ne(IGNORE_INDEX)
    safe_labels = shift_labels.masked_fill(~target_mask, 0)
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    nll = -(token_log_probs * target_mask).sum(dim=1)
    if choice_score == "mean":
        nll = nll / target_mask.sum(dim=1).clamp_min(1)
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
    if mode in {"correct", "no_latent"}:
        return latents
    if mode == "random":
        return torch.randn_like(latents)
    if mode == "shuffled":
        return torch.stack(
            [dataset.load_shuffled_latent(index) for index in batch["indices"]],
            dim=0,
        )
    raise ValueError(f"Unsupported baseline mode: {mode}")


@torch.no_grad()
def evaluate_choice_accuracy(
    llm,
    adapter: TensorSoftPromptAdapter,
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
        for batch in tqdm(loader, desc=f"Eval [{mode}]", leave=False):
            records = batch["records"]
            latents = baseline_latents(mode, batch, dataset)
            predictions = collect_candidate_scores(
                llm=llm,
                adapter=adapter,
                tokenizer=tokenizer,
                records=records,
                latent_map=latents,
                device=device,
                args=args,
                mode=mode,
            )
            for record, prediction in zip(records, predictions):
                answer = str(record["answer"])
                task_type = str(record.get("task_type", "unknown"))
                hit = int(prediction == answer)
                total += 1
                correct += hit
                task_total[task_type] += 1
                task_correct[task_type] += hit
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
        }
    adapter.train()
    return metrics


def save_adapter_checkpoint(
    path: str | Path,
    adapter: TensorSoftPromptAdapter,
    args: argparse.Namespace,
    latent_shape: Sequence[int],
    llm_hidden_size: int,
    metrics: Mapping[str, Any] | None = None,
) -> None:
    payload = {
        "adapter_state_dict": adapter.state_dict(),
        "args": vars(args),
        "latent_shape_chw": list(int(dim) for dim in latent_shape),
        "llm_hidden_size": int(llm_hidden_size),
        "metrics": dict(metrics or {}),
    }
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    apply_runtime_environment(args)
    seed_everything(int(args.seed))
    device = resolve_device(args.device)
    run_dir = build_run_dir(args.output_root, args.run_name)
    dump_json(run_dir / "args.json", vars(args))

    tokenizer, llm, model_dtype = load_tokenizer_and_llm(args, device)
    llm_hidden_size = int(llm.get_input_embeddings().embedding_dim)

    train_dataset = TensorReadoutQADataset(
        qa_path(args.qa_dir, args.train_split),
        latent_dir=args.latent_dir,
        max_records=args.max_train_records,
        prefer_record_latent_ref=bool(args.prefer_record_latent_ref),
    )
    val_dataset = TensorReadoutQADataset(
        qa_path(args.qa_dir, args.val_split),
        latent_dir=args.latent_dir,
        max_records=args.max_val_records,
        prefer_record_latent_ref=bool(args.prefer_record_latent_ref),
    )
    test_dataset = TensorReadoutQADataset(
        qa_path(args.qa_dir, args.test_split),
        latent_dir=args.latent_dir,
        max_records=args.max_test_records,
        prefer_record_latent_ref=bool(args.prefer_record_latent_ref),
    )
    first_latent = train_dataset[0]["latent_map"]
    latent_shape = tuple(int(dim) for dim in first_latent.shape)
    latent_channels = int(latent_shape[0])

    adapter = TensorSoftPromptAdapter(
        latent_channels=latent_channels,
        llm_hidden_size=llm_hidden_size,
        soft_prompt_tokens=int(args.soft_prompt_tokens),
        adapter_dim=int(args.adapter_dim),
        adapter_layers=int(args.adapter_layers),
        adapter_heads=int(args.adapter_heads),
        dropout=float(args.dropout),
    ).to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=max(1, int(args.batch_size)),
        shuffle=True,
        num_workers=0,
        collate_fn=collate_tensor_readout,
    )
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
        "trainable_adapter_parameters": sum(p.numel() for p in adapter.parameters() if p.requires_grad),
        "frozen_llm_parameters": sum(p.numel() for p in llm.parameters()),
    }
    dump_json(run_dir / "run_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    best_val_accuracy = -math.inf
    history: dict[str, Any] = {}
    accumulation_steps = max(1, int(args.gradient_accumulation_steps))
    global_step = 0

    for epoch in range(1, int(args.epochs) + 1):
        adapter.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(train_loader, desc=f"Epoch {epoch:03d} [train]")
        for step, batch in enumerate(progress, start=1):
            answers = [str(record["answer"]) for record in batch["records"]]
            loss = forward_loss(
                llm=llm,
                adapter=adapter,
                tokenizer=tokenizer,
                records=batch["records"],
                answers=answers,
                latent_map=batch["latent_map"],
                device=device,
                max_prompt_tokens=int(args.max_prompt_tokens),
                max_target_tokens=int(args.max_target_tokens),
                append_eos=bool(args.append_eos),
            )
            (loss / accumulation_steps).backward()
            running_loss += float(loss.detach().cpu().item())

            if step % accumulation_steps == 0 or step == len(train_loader):
                if float(args.grad_clip_norm) > 0:
                    torch.nn.utils.clip_grad_norm_(adapter.parameters(), float(args.grad_clip_norm))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            average_loss = running_loss / step
            progress.set_postfix(loss=f"{average_loss:.4f}")
            if step % max(1, int(args.log_interval)) == 0:
                history[f"epoch_{epoch:04d}_step_{step:06d}"] = {
                    "epoch": epoch,
                    "step": step,
                    "global_step": global_step,
                    "train_loss": average_loss,
                }
                dump_json(run_dir / "metrics_latest.json", history)

        train_loss = running_loss / max(1, len(train_loader))
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
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_adapter_checkpoint(
                run_dir / "adapter_best.pt",
                adapter=adapter,
                args=args,
                latent_shape=latent_shape,
                llm_hidden_size=llm_hidden_size,
                metrics=epoch_payload,
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
    print(json.dumps({"run_dir": str(run_dir), "test": test_metrics}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
