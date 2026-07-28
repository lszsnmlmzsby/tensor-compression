from __future__ import annotations

"""Benchmark frozen Qwen tensor-text input against dense tensor cross-attention.

The two methods share one frozen Qwen replica, one QA record order, one
candidate-restricted next-token scorer, and one exact distributed shard.  The
script reports both end-to-end evaluation cost and warmed model-only cost.
"""

import argparse
import gc
import hashlib
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
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for search_path in (PROJECT_ROOT, SRC_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from scripts.evaluate_frozen_qwen_patch_qa import (  # noqa: E402
    FrozenQwenQADataset,
    audit_frozen_qwen,
    collate_records,
    empty_metric_payload,
    finalize_metrics,
    local_timestamp,
    matrix_extreme_support,
    merge_metric_payloads,
    record_field,
    redact_config,
    render_prompt,
    serializable_metric_payload,
    update_metric_payload,
)
from scripts.train_tensor_llm_adapter import (  # noqa: E402
    ExactDistributedEvalSampler,
    TensorReadoutQADataset,
    apply_runtime_environment,
    atomic_dump_json,
    build_distributed_run_dir,
    collate_tensor_readout,
    distributed_barrier,
    distributed_is_initialized,
    distributed_rank,
    distributed_world_size,
    initialize_distributed_device,
    is_main_process,
    last_nonpadding_indices,
    load_llm_with_bounded_host_memory,
    load_tokenizer,
    qa_path,
    run_on_rank_zero_and_broadcast,
    seed_everything,
    set_frozen_llm_execution_mode,
    single_token_choice_ids,
)
from scripts.train_tensor_qwen_cross_attention import (  # noqa: E402
    CHECKPOINT_TYPE,
    CHECKPOINT_VERSION,
    DenseCrossAttentionSidecar,
    architecture_contract,
    autocast_context,
    build_sidecar,
    decoder_backbone,
    load_metadata_and_contract,
    load_spatial_initializer,
    load_trainable_state_dict,
    restricted_choice_logits,
    validate_checkpoint_contract,
)
from tensor_compression.downstream.patch_qa_contract import sha256_file  # noqa: E402
from tensor_compression.downstream.patch_qa_prompt import build_prompt  # noqa: E402
from tensor_compression.utils.pipeline_config import (  # noqa: E402
    first_nested,
    load_yaml_mapping,
    resolve_path_string,
)


RESULT_FORMAT = "tensor_qwen_inference_benchmark_v1"
SUPPORTED_METHODS = ("serialized", "dense")


class DenseBenchmarkDataset(TensorReadoutQADataset):
    """The formal latent loader without constructing an unused shuffled baseline."""

    def _build_random_different_indices(self, seed: int) -> list[int]:
        del seed
        return list(range(len(self.records)))


def _config_value(config: Mapping[str, Any], paths: Sequence[str], default: Any = None) -> Any:
    value = first_nested(config, list(paths))
    return default if value is None else value


def _path_value(value: Any) -> str | None:
    if value is None or str(value).strip().casefold() in {"", "none", "null"}:
        return None
    return resolve_path_string(value, PROJECT_ROOT)


def _model_value(value: Any) -> str | None:
    if value is None or str(value).strip().casefold() in {"", "none", "null"}:
        return None
    raw = str(value).strip()
    path = Path(raw).expanduser()
    if path.is_absolute() or raw.startswith((".", "~")) or path.exists():
        return resolve_path_string(raw, PROJECT_ROOT)
    return raw


def parse_methods(raw: str | Sequence[str]) -> list[str]:
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        methods = [str(item).strip().casefold() for item in raw if str(item).strip()]
    else:
        methods = [item.strip().casefold() for item in str(raw).split(",") if item.strip()]
    if not methods:
        raise ValueError("At least one benchmark method is required.")
    if len(methods) != len(set(methods)):
        raise ValueError(f"Duplicate benchmark methods are not allowed: {methods}.")
    unsupported = sorted(set(methods) - set(SUPPORTED_METHODS))
    if unsupported:
        raise ValueError(f"Unsupported benchmark methods: {unsupported}.")
    if methods == ["dense", "serialized"]:
        raise ValueError(
            "A shared Qwen must benchmark serialized input before installing dense wrappers. "
            "Use methods=[serialized,dense], or run each method in a separate process."
        )
    return methods


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare full-field text serialization with the final full-grid dense "
            "cross-attention interface on identical frozen-Qwen QA inference."
        )
    )
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--model-name-or-path", type=str, default=None)
    parser.add_argument("--qa-dir", type=str, default=None)
    parser.add_argument("--latent-dir", type=str, default=None)
    parser.add_argument("--alignment-checkpoint", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--hf-home", type=str, default=None)
    parser.add_argument("--methods", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--timing-records", type=int, default=None)
    parser.add_argument("--timing-repetitions", type=int, default=None)
    parser.add_argument("--warmup-batches", type=int, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=None)
    parser.add_argument("--dense-checkpoint", type=str, default=None)
    parser.add_argument("--memory-init-checkpoint", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--gpu-hour-price", type=float, default=None)
    parser.add_argument("--model-only", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--prewarm-input-files", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--save-predictions", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--console-progress", action=argparse.BooleanOptionalAction, default=None)
    cli = parser.parse_args(argv)
    config = load_yaml_mapping(cli.config)

    args = argparse.Namespace()
    args.config = str(cli.config)
    args.raw_config = dict(config)
    model_local = _config_value(config, ["model.local_dir"])
    model_name = _config_value(config, ["model.name_or_path", "model.model_name_or_path"])
    configured_model = _path_value(model_local) if model_local else str(model_name or "")
    args.model_name_or_path = _model_value(cli.model_name_or_path) or configured_model
    args.cache_dir = _path_value(
        cli.cache_dir or _config_value(config, ["model.cache_dir", "storage.hf_home"])
    )
    args.hf_home = _path_value(cli.hf_home or _config_value(config, ["storage.hf_home"]))
    args.qa_dir = _path_value(
        cli.qa_dir or _config_value(config, ["data.qa_dir", "patch_qa.stage2b_qa_dir"])
    )
    args.latent_dir = _path_value(
        cli.latent_dir or _config_value(config, ["data.latent_dir", "patch_qa.latent_dir"])
    )
    args.qa_alignment_checkpoint = _path_value(
        cli.alignment_checkpoint
        or _config_value(config, ["data.alignment_checkpoint", "patch_qa.alignment_checkpoint"])
    )
    args.memory_init_checkpoint = _path_value(
        cli.memory_init_checkpoint or _config_value(config, ["memory.init_checkpoint"])
    )
    args.dense_checkpoint = _path_value(
        cli.dense_checkpoint or _config_value(config, ["benchmark.dense_checkpoint"])
    )
    args.output_root = _path_value(
        cli.output_root
        or _config_value(config, ["benchmark.output_root", "training.output_root", "storage.output_root"])
    )
    args.run_name = str(
        cli.run_name or _config_value(config, ["benchmark.run_name"], "tensor_qwen_inference_benchmark")
    )
    configured_methods = cli.methods or _config_value(
        config, ["benchmark.methods"], ["serialized", "dense"]
    )
    args.methods = parse_methods(configured_methods)
    args.split = str(cli.split or _config_value(config, ["benchmark.split"], "test"))
    args.batch_size = int(
        cli.batch_size if cli.batch_size is not None else _config_value(config, ["benchmark.batch_size"], 4)
    )
    args.max_records = (
        int(cli.max_records)
        if cli.max_records is not None
        else _config_value(config, ["benchmark.max_records"])
    )
    args.timing_records = int(
        cli.timing_records
        if cli.timing_records is not None
        else _config_value(config, ["benchmark.timing_records"], 256)
    )
    args.timing_repetitions = int(
        cli.timing_repetitions
        if cli.timing_repetitions is not None
        else _config_value(config, ["benchmark.timing_repetitions"], 3)
    )
    args.warmup_batches = int(
        cli.warmup_batches
        if cli.warmup_batches is not None
        else _config_value(config, ["benchmark.warmup_batches"], 3)
    )
    args.bootstrap_samples = int(
        cli.bootstrap_samples
        if cli.bootstrap_samples is not None
        else _config_value(config, ["benchmark.bootstrap_samples"], 2000)
    )
    args.bootstrap_seed = int(_config_value(config, ["benchmark.bootstrap_seed"], 20260729))
    args.model_only = bool(
        cli.model_only
        if cli.model_only is not None
        else _config_value(config, ["benchmark.model_only"], True)
    )
    args.prewarm_input_files = bool(
        cli.prewarm_input_files
        if cli.prewarm_input_files is not None
        else _config_value(config, ["benchmark.prewarm_input_files"], True)
    )
    args.save_predictions = bool(
        cli.save_predictions
        if cli.save_predictions is not None
        else _config_value(config, ["benchmark.save_predictions"], True)
    )
    args.num_workers = int(
        cli.num_workers
        if cli.num_workers is not None
        else _config_value(config, ["benchmark.num_workers", "data.num_workers"], 2)
    )
    args.matrix_significant_digits = int(
        _config_value(config, ["benchmark.matrix_significant_digits"], 6)
    )
    args.matrix_cache_size = int(_config_value(config, ["benchmark.matrix_cache_size"], 2048))
    args.serialized_max_prompt_tokens = int(
        _config_value(config, ["benchmark.serialized_max_prompt_tokens"], 8192)
    )
    args.dense_max_prompt_tokens = int(
        _config_value(config, ["benchmark.dense_max_prompt_tokens", "training.max_prompt_tokens"], 512)
    )
    args.prompt_template = str(
        _config_value(config, ["benchmark.prompt_template", "training.prompt_template"], "task_specific")
    )
    args.gpu_hour_price = (
        float(cli.gpu_hour_price)
        if cli.gpu_hour_price is not None
        else _config_value(config, ["benchmark.gpu_hour_price"])
    )
    args.price_currency = str(_config_value(config, ["benchmark.price_currency"], "CNY"))

    args.device = str(_config_value(config, ["runtime.device"], "auto"))
    args.seed = int(_config_value(config, ["runtime.seed"], 42))
    args.shuffle_seed = int(_config_value(config, ["runtime.shuffle_seed"], args.seed))
    args.distributed_timeout_seconds = float(
        _config_value(config, ["runtime.distributed_timeout_seconds"], 7200.0)
    )
    args.serialize_llm_loading = bool(
        _config_value(config, ["runtime.serialize_llm_loading"], True)
    )
    args.low_cpu_mem_usage = bool(_config_value(config, ["runtime.low_cpu_mem_usage"], True))
    args.min_host_memory_available_gib = float(
        _config_value(config, ["runtime.min_host_memory_available_gib"], 16.0)
    )
    args.console_progress = bool(
        cli.console_progress
        if cli.console_progress is not None
        else _config_value(config, ["runtime.console_progress"], True)
    )
    args.torch_dtype = str(_config_value(config, ["model.torch_dtype"], "bfloat16"))
    args.trust_remote_code = bool(_config_value(config, ["model.trust_remote_code"], False))
    args.llm_gradient_checkpointing = False
    args.prefer_record_latent_ref = bool(
        _config_value(config, ["data.prefer_record_latent_ref"], False)
    )
    args.latent_cache_size = int(_config_value(config, ["data.latent_cache_size"], 8192))
    args.questions_per_state_group = int(
        _config_value(config, ["training.questions_per_state_group"], 3)
    )

    # These values are replaced by the immutable dense checkpoint architecture
    # before sidecar construction. They exist here only for clear validation.
    args.cross_attention_layers = []
    args.bridge_dim = 0
    args.bridge_heads = 0
    args.bridge_dropout = 0.0
    args.gate_init = 0.0
    args.value_fourier_bands = 0
    args.value_hidden_dim = 0
    args.freeze_spatial_backbone = True
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    required = {
        "model_name_or_path": args.model_name_or_path,
        "qa_dir": args.qa_dir,
        "latent_dir": args.latent_dir,
        "qa_alignment_checkpoint": args.qa_alignment_checkpoint,
        "output_root": args.output_root,
    }
    if "dense" in args.methods:
        required.update(
            {
                "memory_init_checkpoint": args.memory_init_checkpoint,
                "dense_checkpoint": args.dense_checkpoint,
            }
        )
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(f"Missing required benchmark paths: {missing}.")
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative.")
    if args.max_records is not None and int(args.max_records) <= 0:
        raise ValueError("max_records must be positive when provided.")
    if args.timing_records <= 0 or args.timing_repetitions <= 0 or args.warmup_batches < 0:
        raise ValueError("timing_records/repetitions must be positive and warmup_batches non-negative.")
    if args.bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be non-negative.")
    if args.serialized_max_prompt_tokens <= 0 or args.dense_max_prompt_tokens <= 0:
        raise ValueError("Prompt token limits must be positive.")
    if not 5 <= args.matrix_significant_digits <= 12:
        raise ValueError("matrix_significant_digits must be between 5 and 12.")
    if args.matrix_cache_size < 0 or args.latent_cache_size < 0:
        raise ValueError("Input cache sizes must be non-negative.")
    if args.prompt_template != "task_specific":
        raise ValueError("The formal benchmark requires prompt_template=task_specific.")
    if len(args.methods) == 2 and bool(args.prefer_record_latent_ref):
        raise ValueError(
            "A paired benchmark requires prefer_record_latent_ref=false so serialized and "
            "dense resolve every state from the same latent_dir/state_ref.pt file."
        )
    if args.gpu_hour_price is not None and float(args.gpu_hour_price) < 0.0:
        raise ValueError("gpu_hour_price must be non-negative when provided.")


def percentile(values: Sequence[float | int], quantile: float) -> float | None:
    if not values:
        return None
    q = float(quantile)
    if not 0.0 <= q <= 1.0:
        raise ValueError("quantile must lie in [0,1].")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def distribution_summary(values: Sequence[float | int]) -> dict[str, Any]:
    clean = [float(value) for value in values]
    if not clean:
        return {"count": 0, "mean": None, "p50": None, "p90": None, "p95": None, "p99": None, "max": None}
    return {
        "count": len(clean),
        "mean": sum(clean) / len(clean),
        "p50": percentile(clean, 0.50),
        "p90": percentile(clean, 0.90),
        "p95": percentile(clean, 0.95),
        "p99": percentile(clean, 0.99),
        "max": max(clean),
    }


def unique_parameter_count(module: nn.Module) -> int:
    seen: set[int] = set()
    total = 0
    for parameter in module.parameters():
        if id(parameter) not in seen:
            seen.add(id(parameter))
            total += int(parameter.numel())
    return total


def record_digest(records: Sequence[Mapping[str, Any]]) -> str:
    identifiers = [str(record.get("qa_id", "")) for record in records]
    if any(not value for value in identifiers) or len(identifiers) != len(set(identifiers)):
        raise ValueError("Benchmark records must have unique, non-empty qa_id values.")
    return hashlib.sha256("\n".join(identifiers).encode("utf-8")).hexdigest()


def gather_object(value: Any) -> list[Any]:
    if not distributed_is_initialized():
        return [value]
    gathered: list[Any] = [None] * distributed_world_size()
    dist.all_gather_object(gathered, value)
    return gathered


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak_memory(device: torch.device) -> dict[str, int]:
    if device.type != "cuda":
        return {"allocated_bytes": 0, "reserved_bytes": 0}
    synchronize_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
    }


def finish_peak_memory(device: torch.device, baseline: Mapping[str, int]) -> dict[str, int]:
    if device.type != "cuda":
        return {
            "baseline_allocated_bytes": 0,
            "baseline_reserved_bytes": 0,
            "peak_allocated_bytes": 0,
            "peak_reserved_bytes": 0,
            "incremental_peak_allocated_bytes": 0,
            "incremental_peak_reserved_bytes": 0,
        }
    synchronize_device(device)
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    return {
        "baseline_allocated_bytes": int(baseline["allocated_bytes"]),
        "baseline_reserved_bytes": int(baseline["reserved_bytes"]),
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "incremental_peak_allocated_bytes": max(0, peak_allocated - int(baseline["allocated_bytes"])),
        "incremental_peak_reserved_bytes": max(0, peak_reserved - int(baseline["reserved_bytes"])),
    }


def attention_score_proxy(
    *,
    batch_size: int,
    padded_tokens: int,
    qwen_layers: int,
    qwen_heads: int,
    dense_bridges: int = 0,
    bridge_heads: int = 0,
    memory_cells: int = 0,
) -> dict[str, int]:
    batch = int(batch_size)
    tokens = int(padded_tokens)
    self_attention = batch * int(qwen_layers) * int(qwen_heads) * tokens * tokens
    cross_attention = (
        batch * int(dense_bridges) * int(bridge_heads) * tokens * int(memory_cells)
    )
    return {
        "self_attention_score_elements": self_attention,
        "dense_cross_attention_score_elements": cross_attention,
        "total_attention_score_elements": self_attention + cross_attention,
    }


def apply_checkpoint_architecture(
    args: argparse.Namespace,
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    if str(checkpoint.get("checkpoint_type", "")) != CHECKPOINT_TYPE:
        raise ValueError("The configured dense checkpoint has the wrong checkpoint_type.")
    if int(checkpoint.get("checkpoint_version", 0)) != CHECKPOINT_VERSION:
        raise ValueError("The configured dense checkpoint has an unsupported version.")
    architecture = checkpoint.get("architecture")
    if not isinstance(architecture, Mapping):
        raise ValueError("The dense checkpoint is missing its architecture contract.")
    required = {
        "layers_1based",
        "bridge_dim",
        "heads",
        "dropout",
        "gate_init",
        "value_fourier_bands",
        "value_hidden_dim",
        "freeze_spatial_backbone",
    }
    missing = sorted(required - set(architecture))
    if missing:
        raise ValueError(f"Dense checkpoint architecture is incomplete: {missing}.")
    layers = [int(value) for value in architecture["layers_1based"]]
    if not layers or len(layers) != len(set(layers)) or any(value <= 0 for value in layers):
        raise ValueError(f"Invalid checkpoint cross-attention layers: {layers}.")
    args.cross_attention_layers = layers
    args.bridge_dim = int(architecture["bridge_dim"])
    args.bridge_heads = int(architecture["heads"])
    args.bridge_dropout = float(architecture["dropout"])
    args.gate_init = float(architecture["gate_init"])
    args.value_fourier_bands = int(architecture["value_fourier_bands"])
    args.value_hidden_dim = int(architecture["value_hidden_dim"])
    args.freeze_spatial_backbone = bool(architecture["freeze_spatial_backbone"])
    if args.bridge_dim <= 0 or args.bridge_heads <= 0 or args.bridge_dim % args.bridge_heads:
        raise ValueError("The checkpoint bridge dimension is not divisible by its head count.")
    return dict(architecture)


def build_benchmark_datasets(
    args: argparse.Namespace,
    latent_contract: Mapping[str, Any],
) -> tuple[FrozenQwenQADataset, DenseBenchmarkDataset | None, dict[str, Any]]:
    source_path = qa_path(args.qa_dir, args.split)
    serialized_dataset = FrozenQwenQADataset(
        source_path,
        latent_dir=args.latent_dir,
        latent_contract=latent_contract,
        matrix_significant_digits=int(args.matrix_significant_digits),
        matrix_cache_size=int(args.matrix_cache_size),
        max_records=args.max_records,
    )
    dense_dataset: DenseBenchmarkDataset | None = None
    if "dense" in args.methods:
        dense_dataset = DenseBenchmarkDataset(
            source_path,
            latent_dir=args.latent_dir,
            max_records=args.max_records,
            subset_mode="prefix",
            subset_seed=int(args.shuffle_seed),
            prefer_record_latent_ref=bool(args.prefer_record_latent_ref),
            shuffle_seed=int(args.shuffle_seed),
            latent_cache_size=int(args.latent_cache_size),
            latent_contract=latent_contract,
        )
        serialized_ids = [str(record["qa_id"]) for record in serialized_dataset.records]
        dense_ids = [str(record["qa_id"]) for record in dense_dataset.records]
        if serialized_ids != dense_ids:
            raise RuntimeError("Serialized and dense datasets selected different QA records or order.")
        mismatched_paths = []
        for record in serialized_dataset.records:
            serialized_path = serialized_dataset.latent_path_for_record(record).resolve()
            dense_path = dense_dataset.latent_path_for_record(record).resolve()
            if serialized_path != dense_path:
                mismatched_paths.append(
                    {
                        "qa_id": str(record["qa_id"]),
                        "serialized": str(serialized_path),
                        "dense": str(dense_path),
                    }
                )
                if len(mismatched_paths) >= 8:
                    break
        if mismatched_paths:
            raise RuntimeError(
                "Serialized and dense methods resolve different latent files: "
                f"{mismatched_paths}."
            )
    digest = record_digest(serialized_dataset.records)
    state_count = len({str(record.get("state_ref", "")) for record in serialized_dataset.records})
    task_counts: dict[str, int] = defaultdict(int)
    field_counts: dict[str, int] = defaultdict(int)
    for record in serialized_dataset.records:
        task_counts[str(record.get("task_type", "unknown"))] += 1
        field_counts[record_field(record)] += 1
    audit = {
        "split": str(args.split),
        "qa_file": str(source_path.resolve()),
        "qa_file_sha256": sha256_file(source_path),
        "subset_mode": "prefix",
        "max_records": args.max_records,
        "records": len(serialized_dataset),
        "states": state_count,
        "qa_id_order_sha256": digest,
        "task_counts": dict(sorted(task_counts.items())),
        "field_counts": dict(sorted(field_counts.items())),
        "methods_have_identical_records_and_order": dense_dataset is not None,
        "methods_resolve_identical_latent_files": dense_dataset is not None,
    }
    return serialized_dataset, dense_dataset, audit


def collate_method_items(method: str, items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if method == "serialized":
        return collate_records(items)
    if method == "dense":
        return collate_tensor_readout(list(items))
    raise ValueError(f"Unsupported method: {method}.")


def metric_contracts_for_dense_batch(
    records: Sequence[Mapping[str, Any]],
    latent_map: torch.Tensor,
) -> tuple[list[list[str]], list[str]]:
    acceptable: list[list[str]] = []
    tie_scopes: list[str] = []
    for record, latent in zip(records, latent_map, strict=True):
        support = matrix_extreme_support(latent[0])
        contract = FrozenQwenQADataset.metric_contract_for_record(record, support)
        acceptable.append([str(value) for value in contract["acceptable_answers"]])
        tie_scopes.append(str(contract["extreme_tie_scope"]))
    return acceptable, tie_scopes


def tokenize_prompts(
    prompts: Sequence[str],
    tokenizer,
    max_prompt_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    encoded = tokenizer(
        list(prompts),
        padding=True,
        truncation=False,
        return_tensors="pt",
        add_special_tokens=True,
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    lengths = [int(value) for value in attention_mask.sum(dim=1).tolist()]
    if not lengths or min(lengths) <= 0:
        raise ValueError("Every benchmark prompt must contain at least one token.")
    if max(lengths) > int(max_prompt_tokens):
        raise ValueError(
            f"A prompt has {max(lengths)} tokens, exceeding max_prompt_tokens={max_prompt_tokens}; "
            "benchmark prompts are never truncated."
        )
    return input_ids, attention_mask, lengths


def prepare_batch(
    method: str,
    batch: Mapping[str, Any],
    tokenizer,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, float]]:
    records = list(batch["records"])
    timings = {
        "prompt_construction_seconds": 0.0,
        "prompt_tokenization_seconds": 0.0,
        "choice_tokenization_seconds": 0.0,
        "tie_contract_seconds": 0.0,
    }
    prompt_started = time.perf_counter()
    if method == "serialized":
        matrix_texts = [str(value) for value in batch["matrix_texts"]]
        prompts = [
            render_prompt(record, str(args.prompt_template), matrix_text)
            for record, matrix_text in zip(records, matrix_texts, strict=True)
        ]
        max_tokens = int(args.serialized_max_prompt_tokens)
        serialized_utf8_bytes = sum(len(value.encode("utf-8")) for value in matrix_texts)
        acceptable_answers = [list(value) for value in batch["acceptable_answers"]]
        tie_scopes = [str(value) for value in batch["extreme_tie_scopes"]]
        latent_map = None
    elif method == "dense":
        prompts = [
            build_prompt(record, prompt_template=str(args.prompt_template)) for record in records
        ]
        max_tokens = int(args.dense_max_prompt_tokens)
        serialized_utf8_bytes = 0
        latent_map = batch["latent_map"]
        timings["prompt_construction_seconds"] = time.perf_counter() - prompt_started
        tie_started = time.perf_counter()
        acceptable_answers, tie_scopes = metric_contracts_for_dense_batch(records, latent_map)
        timings["tie_contract_seconds"] = time.perf_counter() - tie_started
    else:
        raise ValueError(f"Unsupported method: {method}.")
    if method == "serialized":
        timings["prompt_construction_seconds"] = time.perf_counter() - prompt_started

    token_started = time.perf_counter()
    input_ids, attention_mask, prompt_lengths = tokenize_prompts(
        prompts, tokenizer, max_prompt_tokens=max_tokens
    )
    timings["prompt_tokenization_seconds"] = time.perf_counter() - token_started
    choice_started = time.perf_counter()
    choice_spec = single_token_choice_ids(records, tokenizer)
    timings["choice_tokenization_seconds"] = time.perf_counter() - choice_started
    if choice_spec is None:
        raise ValueError("Every displayed answer label must map to one unique tokenizer token.")
    candidate_ids, target_indices = choice_spec

    prepared = {
        "method": method,
        "indices": [int(value) for value in batch["indices"]],
        "records": records,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "prompt_lengths": prompt_lengths,
        "candidate_ids": candidate_ids,
        "target_indices": target_indices,
        "latent_map": latent_map,
        "acceptable_answers": acceptable_answers,
        "extreme_tie_scopes": tie_scopes,
        "serialized_utf8_bytes": int(serialized_utf8_bytes),
    }
    return prepared, timings


@torch.inference_mode()
def execute_prepared_batch(
    prepared: Mapping[str, Any],
    llm: nn.Module,
    sidecar: DenseCrossAttentionSidecar | None,
    device: torch.device,
    model_dtype: torch.dtype,
) -> list[torch.Tensor]:
    method = str(prepared["method"])
    input_ids = prepared["input_ids"].to(device, non_blocking=True)
    attention_mask = prepared["attention_mask"].to(device, non_blocking=True)
    latent_map = prepared.get("latent_map")
    if isinstance(latent_map, torch.Tensor):
        latent_map = latent_map.to(device, non_blocking=True)
    output_embeddings = llm.get_output_embeddings()
    if output_embeddings is None:
        raise ValueError("Frozen Qwen does not expose its output embedding head.")
    if method == "dense" and sidecar is None:
        raise RuntimeError("Dense inference requires an installed sidecar.")
    try:
        with autocast_context(device, model_dtype):
            if sidecar is not None and method == "dense":
                sidecar.bind(latent_map, mode="correct")
            outputs = decoder_backbone(llm)(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            final_indices = last_nonpadding_indices(attention_mask)
            rows = torch.arange(input_ids.shape[0], device=device)
            hidden = outputs.last_hidden_state[rows, final_indices]
            return restricted_choice_logits(
                hidden,
                prepared["candidate_ids"],
                output_embeddings,
            )
    finally:
        if sidecar is not None and method == "dense":
            sidecar.clear()


def scored_results(
    prepared: Mapping[str, Any],
    logits: Sequence[torch.Tensor],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for record, row_logits, target_index, prompt_tokens in zip(
        prepared["records"],
        logits,
        prepared["target_indices"],
        prepared["prompt_lengths"],
        strict=True,
    ):
        choices = [str(value) for value in record["choices"]]
        probabilities = F.softmax(row_logits.float(), dim=-1).detach().cpu()
        prediction_index = int(torch.argmax(probabilities).item())
        if not 0 <= int(target_index) < len(choices):
            raise ValueError(f"Invalid target index for {record.get('qa_id')}.")
        results.append(
            {
                "prediction": choices[prediction_index],
                "probabilities": {
                    choice: float(probabilities[index].item())
                    for index, choice in enumerate(choices)
                },
                "prompt_tokens": int(prompt_tokens),
            }
        )
    return results


def batch_cost_values(
    prepared: Mapping[str, Any],
    args: argparse.Namespace,
    model_info: Mapping[str, int],
) -> dict[str, int]:
    batch_size = len(prepared["records"])
    padded_length = int(prepared["input_ids"].shape[1])
    latent = prepared.get("latent_map")
    latent_bytes = int(latent.numel() * latent.element_size()) if isinstance(latent, torch.Tensor) else 0
    token_tensor_bytes = int(
        prepared["input_ids"].numel() * prepared["input_ids"].element_size()
        + prepared["attention_mask"].numel() * prepared["attention_mask"].element_size()
    )
    dense = str(prepared["method"]) == "dense"
    proxy = attention_score_proxy(
        batch_size=batch_size,
        padded_tokens=padded_length,
        qwen_layers=int(model_info["qwen_layers"]),
        qwen_heads=int(model_info["qwen_heads"]),
        dense_bridges=len(args.cross_attention_layers) if dense else 0,
        bridge_heads=int(args.bridge_heads) if dense else 0,
        memory_cells=int(model_info["memory_cells"]) if dense else 0,
    )
    return {
        "records": batch_size,
        "useful_prompt_tokens": sum(int(value) for value in prepared["prompt_lengths"]),
        "padded_prompt_tokens": batch_size * padded_length,
        "h2d_payload_bytes": token_tensor_bytes + latent_bytes,
        "serialized_matrix_utf8_bytes": int(prepared["serialized_utf8_bytes"]),
        "matrix_values_or_memory_cells": batch_size * int(model_info["memory_cells"]),
        **proxy,
    }


def make_loader(
    dataset,
    method: str,
    args: argparse.Namespace,
    device: torch.device,
) -> DataLoader:
    sampler = (
        ExactDistributedEvalSampler(
            dataset,
            rank=distributed_rank(),
            num_replicas=distributed_world_size(),
        )
        if distributed_is_initialized()
        else None
    )
    collate_fn = collate_records if method == "serialized" else collate_tensor_readout
    return DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        sampler=sampler,
        num_workers=int(args.num_workers),
        persistent_workers=False,
        prefetch_factor=1 if int(args.num_workers) > 0 else None,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )


def event_pair(device: torch.device) -> tuple[torch.cuda.Event | None, torch.cuda.Event | None]:
    if device.type != "cuda":
        return None, None
    return (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )


def aggregate_cost_reports(
    reports: Sequence[Mapping[str, Any]],
    *,
    expected_records: int,
    repetitions: int = 1,
    gpu_hour_price: float | None = None,
    price_currency: str = "CNY",
) -> dict[str, Any]:
    if not reports:
        raise ValueError("At least one rank cost report is required.")
    record_total = sum(int(report.get("records", 0)) for report in reports)
    expected_work = int(expected_records) * int(repetitions)
    if record_total != expected_work:
        raise RuntimeError(
            f"Cost reports cover {record_total} records, expected {expected_work}."
        )
    repetition_walls = [
        [float(value) for value in report.get("repetition_wall_seconds", [])]
        for report in reports
    ]
    if any(len(values) != int(repetitions) for values in repetition_walls):
        raise RuntimeError("Every rank must report one wall time per timing repetition.")
    critical_walls = [
        max(values[index] for values in repetition_walls)
        for index in range(int(repetitions))
    ]
    critical_wall = sum(critical_walls)
    prompt_lengths = [
        int(value) for report in reports for value in report.get("prompt_lengths", [])
    ]
    if len(prompt_lengths) != record_total:
        raise RuntimeError(
            f"Cost reports contain {len(prompt_lengths)} prompt lengths for {record_total} records."
        )
    batch_latencies = [
        float(value) for report in reports for value in report.get("accelerator_batch_ms", [])
    ]
    first_batch_latencies = [
        float(values[0])
        for report in reports
        if (values := report.get("accelerator_batch_ms", []))
    ]
    sum_fields = (
        "batches",
        "useful_prompt_tokens",
        "padded_prompt_tokens",
        "h2d_payload_bytes",
        "serialized_matrix_utf8_bytes",
        "matrix_values_or_memory_cells",
        "self_attention_score_elements",
        "dense_cross_attention_score_elements",
        "total_attention_score_elements",
    )
    totals = {
        field: sum(int(report.get(field, 0)) for report in reports) for field in sum_fields
    }
    cpu_phase_names = sorted(
        {
            str(key)
            for report in reports
            for key in report.get("cpu_phase_seconds", {})
        }
    )
    cpu_phases = {
        name: {
            "sum_across_ranks": sum(
                float(report.get("cpu_phase_seconds", {}).get(name, 0.0))
                for report in reports
            ),
            "max_rank": max(
                float(report.get("cpu_phase_seconds", {}).get(name, 0.0))
                for report in reports
            ),
        }
        for name in cpu_phase_names
    }
    memory_by_rank = [dict(report.get("memory", {})) for report in reports]
    peak_allocated = max(
        (int(value.get("peak_allocated_bytes", 0)) for value in memory_by_rank),
        default=0,
    )
    peak_reserved = max(
        (int(value.get("peak_reserved_bytes", 0)) for value in memory_by_rank),
        default=0,
    )
    incremental_peak = max(
        (int(value.get("incremental_peak_allocated_bytes", 0)) for value in memory_by_rank),
        default=0,
    )
    world_size = len(reports)
    gpu_hours = world_size * critical_wall / 3600.0
    price = None
    if gpu_hour_price is not None:
        price = {
            "amount": gpu_hours * float(gpu_hour_price),
            "currency": str(price_currency),
            "gpu_hour_price": float(gpu_hour_price),
        }
    useful_tokens = int(totals["useful_prompt_tokens"])
    padded_tokens = int(totals["padded_prompt_tokens"])
    if padded_tokens < useful_tokens:
        raise RuntimeError("Padded prompt-token work cannot be smaller than useful tokens.")
    return {
        "records": record_total,
        "unique_records": int(expected_records),
        "repetitions": int(repetitions),
        "forwards": int(totals["batches"]),
        "critical_path_wall_seconds": critical_wall,
        "critical_path_wall_seconds_by_repetition": critical_walls,
        "records_per_second": record_total / max(critical_wall, 1.0e-12),
        "milliseconds_per_record_critical_path": 1000.0 * critical_wall / max(1, record_total),
        "gpu_seconds": world_size * critical_wall,
        "gpu_hours": gpu_hours,
        "estimated_hardware_cost": price,
        "prompt_tokens": {
            **distribution_summary(prompt_lengths),
            "total_useful": useful_tokens,
            "total_padded": padded_tokens,
            "padding_efficiency": useful_tokens / max(1, padded_tokens),
            "useful_tokens_per_second": useful_tokens / max(critical_wall, 1.0e-12),
            "padded_tokens_per_second": padded_tokens / max(critical_wall, 1.0e-12),
        },
        "accelerator_batch_latency_ms": distribution_summary(batch_latencies),
        "first_measured_batch_latency_ms_by_rank": first_batch_latencies,
        "cpu_phase_seconds": cpu_phases,
        "input_payload": {
            "h2d_bytes": int(totals["h2d_payload_bytes"]),
            "serialized_matrix_utf8_bytes": int(totals["serialized_matrix_utf8_bytes"]),
            "matrix_values_or_memory_cells": int(totals["matrix_values_or_memory_cells"]),
        },
        "logical_attention_score_proxy": {
            "self_attention_score_elements": int(totals["self_attention_score_elements"]),
            "dense_cross_attention_score_elements": int(
                totals["dense_cross_attention_score_elements"]
            ),
            "total_attention_score_elements": int(totals["total_attention_score_elements"]),
            "is_not_flops_or_measured_kernel_work": True,
            "omits": [
                "linear projections",
                "MLPs",
                "normalization",
                "dense spatial-memory encoder",
                "kernel sparsity and fusion effects",
            ],
        },
        "cuda_memory": {
            "peak_allocated_bytes_max_rank": peak_allocated,
            "peak_reserved_bytes_max_rank": peak_reserved,
            "incremental_peak_allocated_bytes_max_rank": incremental_peak,
            "by_rank": memory_by_rank,
        },
        "rank_reports": [
            {
                "rank": int(report.get("rank", index)),
                "records": int(report.get("records", 0)),
                "batches": int(report.get("batches", 0)),
                "repetition_wall_seconds": list(report.get("repetition_wall_seconds", [])),
            }
            for index, report in enumerate(reports)
        ],
    }


def prediction_payload(
    index: int,
    record: Mapping[str, Any],
    scored: Mapping[str, Any],
    acceptable_answers: Sequence[str],
) -> dict[str, Any]:
    answer = str(record["answer"])
    prediction = str(scored["prediction"])
    return {
        "index": int(index),
        "qa_id": str(record["qa_id"]),
        "state_ref": str(record.get("state_ref", "")),
        "sample_index": int(record.get("sample_index", -1)),
        "task_type": str(record.get("task_type", "unknown")),
        "field": record_field(record),
        "answer": answer,
        "prediction": prediction,
        "correct": bool(prediction == answer),
        "tie_aware_correct": bool(prediction in {str(value) for value in acceptable_answers}),
        "choices": [str(value) for value in record["choices"]],
        "probabilities": {
            str(key): float(value) for key, value in scored["probabilities"].items()
        },
        "prompt_tokens": int(scored["prompt_tokens"]),
    }


def validate_gathered_predictions(
    predictions: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    ordered = sorted((dict(value) for value in predictions), key=lambda value: int(value["index"]))
    if len(ordered) != len(records):
        raise RuntimeError(
            f"Gathered {len(ordered)} predictions for {len(records)} benchmark records."
        )
    expected_indices = list(range(len(records)))
    observed_indices = [int(value["index"]) for value in ordered]
    if observed_indices != expected_indices:
        raise RuntimeError("Distributed prediction indices contain a gap, duplicate, or reordering.")
    for index, (prediction, record) in enumerate(zip(ordered, records, strict=True)):
        if str(prediction["qa_id"]) != str(record["qa_id"]):
            raise RuntimeError(f"Prediction/record QA identity mismatch at index {index}.")
    return ordered


def run_end_to_end(
    *,
    method: str,
    dataset,
    llm: nn.Module,
    sidecar: DenseCrossAttentionSidecar | None,
    tokenizer,
    device: torch.device,
    model_dtype: torch.dtype,
    args: argparse.Namespace,
    model_info: Mapping[str, int],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    loader = make_loader(dataset, method, args, device)
    local_metric = empty_metric_payload()
    local_predictions: list[dict[str, Any]] = []
    cpu_phases: dict[str, float] = defaultdict(float)
    cost_totals: dict[str, int] = defaultdict(int)
    prompt_lengths: list[int] = []
    event_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    cpu_batch_ms: list[float] = []

    distributed_barrier()
    baseline_memory = reset_peak_memory(device)
    synchronize_device(device)
    wall_started = time.perf_counter()
    iterator = iter(loader)
    progress = tqdm(
        total=len(loader),
        desc=f"Benchmark {method}",
        disable=not bool(args.console_progress) or not is_main_process(),
        leave=False,
    )
    while True:
        wait_started = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            break
        cpu_phases["main_process_data_wait_seconds"] += time.perf_counter() - wait_started
        prepared, preparation_times = prepare_batch(method, batch, tokenizer, args)
        for name, value in preparation_times.items():
            cpu_phases[name] += float(value)
        prompt_lengths.extend(int(value) for value in prepared["prompt_lengths"])
        for name, value in batch_cost_values(prepared, args, model_info).items():
            cost_totals[name] += int(value)
        cost_totals["batches"] += 1

        start_event, end_event = event_pair(device)
        batch_started = time.perf_counter()
        if start_event is not None:
            start_event.record()
        logits = execute_prepared_batch(prepared, llm, sidecar, device, model_dtype)
        if end_event is not None:
            end_event.record()
            event_pairs.append((start_event, end_event))
        materialize_started = time.perf_counter()
        results = scored_results(prepared, logits)

        for index, record, scored, acceptable, tie_scope in zip(
            prepared["indices"],
            prepared["records"],
            results,
            prepared["acceptable_answers"],
            prepared["extreme_tie_scopes"],
            strict=True,
        ):
            update_metric_payload(
                local_metric,
                record,
                scored,
                int(index),
                acceptable_answers=acceptable,
                extreme_tie_scope=tie_scope,
            )
            local_predictions.append(
                prediction_payload(int(index), record, scored, acceptable)
            )
        cpu_phases["output_materialization_and_metrics_seconds"] += (
            time.perf_counter() - materialize_started
        )
        cpu_batch_ms.append((time.perf_counter() - batch_started) * 1000.0)
        progress.update(1)
    progress.close()
    synchronize_device(device)
    local_wall = time.perf_counter() - wall_started
    memory = finish_peak_memory(device, baseline_memory)
    accelerator_ms = (
        [float(start.elapsed_time(end)) for start, end in event_pairs]
        if device.type == "cuda"
        else cpu_batch_ms
    )
    local_report = {
        "rank": distributed_rank(),
        "records": int(cost_totals["records"]),
        "batches": int(cost_totals["batches"]),
        "repetition_wall_seconds": [local_wall],
        "prompt_lengths": prompt_lengths,
        "accelerator_batch_ms": accelerator_ms,
        "cpu_phase_seconds": dict(cpu_phases),
        "memory": memory,
        **{
            key: int(value)
            for key, value in cost_totals.items()
            if key not in {"records", "batches"}
        },
    }
    cost = aggregate_cost_reports(
        gather_object(local_report),
        expected_records=len(dataset),
        repetitions=1,
        gpu_hour_price=args.gpu_hour_price,
        price_currency=args.price_currency,
    )
    metric_payloads = gather_object(serializable_metric_payload(local_metric))
    merged_metric = merge_metric_payloads(metric_payloads, expected_total=len(dataset))
    metrics = finalize_metrics(merged_metric)
    gathered_predictions = [
        item for rank_values in gather_object(local_predictions) for item in rank_values
    ]
    predictions = validate_gathered_predictions(gathered_predictions, dataset.records)
    return metrics, predictions, cost


def local_timing_indices(dataset_size: int, timing_records: int) -> list[int]:
    cap = min(int(dataset_size), int(timing_records))
    return list(range(distributed_rank(), cap, distributed_world_size()))


def prepare_model_only_batches(
    *,
    method: str,
    dataset,
    tokenizer,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, Any]], float]:
    started = time.perf_counter()
    indices = local_timing_indices(len(dataset), int(args.timing_records))
    batches: list[dict[str, Any]] = []
    for offset in range(0, len(indices), int(args.batch_size)):
        batch_indices = indices[offset : offset + int(args.batch_size)]
        items = [dataset[index] for index in batch_indices]
        batch = collate_method_items(method, items)
        prepared, _timings = prepare_batch(method, batch, tokenizer, args)
        if device.type == "cuda":
            prepared["input_ids"] = prepared["input_ids"].pin_memory()
            prepared["attention_mask"] = prepared["attention_mask"].pin_memory()
            if isinstance(prepared.get("latent_map"), torch.Tensor):
                prepared["latent_map"] = prepared["latent_map"].pin_memory()
        batches.append(prepared)
    return batches, time.perf_counter() - started


def run_model_only(
    *,
    method: str,
    prepared_batches: Sequence[Mapping[str, Any]],
    dataset_size: int,
    llm: nn.Module,
    sidecar: DenseCrossAttentionSidecar | None,
    device: torch.device,
    model_dtype: torch.dtype,
    args: argparse.Namespace,
    model_info: Mapping[str, int],
    preparation_seconds: float,
) -> dict[str, Any]:
    warmup_started = time.perf_counter()
    if prepared_batches:
        for index in range(int(args.warmup_batches)):
            execute_prepared_batch(
                prepared_batches[index % len(prepared_batches)],
                llm,
                sidecar,
                device,
                model_dtype,
            )
    synchronize_device(device)
    warmup_seconds = time.perf_counter() - warmup_started
    distributed_barrier()
    baseline_memory = reset_peak_memory(device)
    repetition_walls: list[float] = []
    event_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    cpu_batch_ms: list[float] = []
    for _repetition in range(int(args.timing_repetitions)):
        distributed_barrier()
        synchronize_device(device)
        started = time.perf_counter()
        for prepared in prepared_batches:
            start_event, end_event = event_pair(device)
            batch_started = time.perf_counter()
            if start_event is not None:
                start_event.record()
            logits = execute_prepared_batch(prepared, llm, sidecar, device, model_dtype)
            if end_event is not None:
                end_event.record()
                event_pairs.append((start_event, end_event))
            else:
                # Keep the candidate-scoring result live through the timed call.
                _ = logits
            cpu_batch_ms.append((time.perf_counter() - batch_started) * 1000.0)
        synchronize_device(device)
        repetition_walls.append(time.perf_counter() - started)
    memory = finish_peak_memory(device, baseline_memory)
    accelerator_ms = (
        [float(start.elapsed_time(end)) for start, end in event_pairs]
        if device.type == "cuda"
        else cpu_batch_ms
    )
    one_pass: dict[str, int] = defaultdict(int)
    prompt_lengths_one_pass: list[int] = []
    for prepared in prepared_batches:
        prompt_lengths_one_pass.extend(int(value) for value in prepared["prompt_lengths"])
        for name, value in batch_cost_values(prepared, args, model_info).items():
            one_pass[name] += int(value)
        one_pass["batches"] += 1
    repetitions = int(args.timing_repetitions)
    local_report = {
        "rank": distributed_rank(),
        "records": int(one_pass["records"]) * repetitions,
        "batches": int(one_pass["batches"]) * repetitions,
        "repetition_wall_seconds": repetition_walls,
        "prompt_lengths": prompt_lengths_one_pass * repetitions,
        "accelerator_batch_ms": accelerator_ms,
        "cpu_phase_seconds": {
            "model_only_batch_preparation_seconds": float(preparation_seconds),
            "warmup_seconds": float(warmup_seconds),
        },
        "memory": memory,
        **{
            key: int(value) * repetitions
            for key, value in one_pass.items()
            if key not in {"records", "batches"}
        },
    }
    expected_unique = min(int(dataset_size), int(args.timing_records))
    result = aggregate_cost_reports(
        gather_object(local_report),
        expected_records=expected_unique,
        repetitions=repetitions,
        gpu_hour_price=args.gpu_hour_price,
        price_currency=args.price_currency,
    )
    result["timing_contract"] = {
        "input_batches_prebuilt_on_cpu": True,
        "cpu_tensors_pinned_for_cuda": device.type == "cuda",
        "latent_file_io_excluded": True,
        "matrix_serialization_excluded": True,
        "prompt_construction_excluded": True,
        "tokenization_excluded": True,
        "h2d_included": True,
        "dense_memory_and_cross_attention_included": method == "dense",
        "candidate_restricted_logit_computation_included": True,
        "probability_softmax_and_output_cpu_materialization_excluded": True,
        "warmup_batches_per_rank": int(args.warmup_batches),
    }
    return result


def _accuracy_from_predictions(values: Sequence[Mapping[str, Any]]) -> float:
    return sum(int(bool(value["correct"])) for value in values) / max(1, len(values))


def _macro_task_accuracy(values: Sequence[Mapping[str, Any]]) -> float:
    task_total: dict[str, int] = defaultdict(int)
    task_correct: dict[str, int] = defaultdict(int)
    for value in values:
        task = str(value["task_type"])
        task_total[task] += 1
        task_correct[task] += int(bool(value["correct"]))
    return sum(task_correct[task] / task_total[task] for task in task_total) / max(
        1, len(task_total)
    )


def predictions_by_unique_id(
    values: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for value in values:
        qa_id = str(value.get("qa_id", ""))
        if not qa_id:
            raise ValueError(f"{label} contains an empty qa_id.")
        if qa_id in output:
            raise ValueError(f"{label} contains duplicate qa_id={qa_id!r}.")
        output[qa_id] = value
    return output


def paired_contingency(
    serialized: Sequence[Mapping[str, Any]],
    dense: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    serialized_by_id = predictions_by_unique_id(serialized, label="serialized predictions")
    dense_by_id = predictions_by_unique_id(dense, label="dense predictions")
    if set(serialized_by_id) != set(dense_by_id):
        raise ValueError("Paired comparison requires identical QA IDs.")
    labels = ("both_correct", "dense_only_correct", "serialized_only_correct", "both_wrong")
    counts: dict[str, int] = {label: 0 for label in labels}
    by_task: dict[str, dict[str, int]] = defaultdict(lambda: {label: 0 for label in labels})
    for qa_id, serialized_value in serialized_by_id.items():
        dense_value = dense_by_id[qa_id]
        for key in ("answer", "state_ref", "task_type", "field"):
            if str(serialized_value.get(key)) != str(dense_value.get(key)):
                raise ValueError(f"Paired metadata differs for qa_id={qa_id}, field={key}.")
        left = bool(serialized_value["correct"])
        right = bool(dense_value["correct"])
        label = (
            "both_correct"
            if left and right
            else "dense_only_correct"
            if right
            else "serialized_only_correct"
            if left
            else "both_wrong"
        )
        counts[label] += 1
        by_task[str(serialized_value["task_type"])][label] += 1
    return {
        "records": len(serialized_by_id),
        **counts,
        "dense_minus_serialized_accuracy": (
            counts["dense_only_correct"] - counts["serialized_only_correct"]
        )
        / max(1, len(serialized_by_id)),
        "by_task": {task: dict(values) for task, values in sorted(by_task.items())},
    }


def state_cluster_bootstrap(
    serialized: Sequence[Mapping[str, Any]],
    dense: Sequence[Mapping[str, Any]],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if int(samples) <= 0:
        return {
            "enabled": False,
            "samples": 0,
            "resampling_unit": "state_ref",
        }
    serialized_by_id = {
        qa_id: dict(value)
        for qa_id, value in predictions_by_unique_id(
            serialized, label="serialized predictions"
        ).items()
    }
    dense_by_id = {
        qa_id: dict(value)
        for qa_id, value in predictions_by_unique_id(
            dense, label="dense predictions"
        ).items()
    }
    if set(serialized_by_id) != set(dense_by_id):
        raise ValueError("Cluster bootstrap requires identical paired QA IDs.")
    clusters: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for qa_id, left in serialized_by_id.items():
        right = dense_by_id[qa_id]
        state_ref = str(left.get("state_ref", ""))
        if not state_ref or state_ref != str(right.get("state_ref", "")):
            raise ValueError(f"Invalid paired state_ref for {qa_id}.")
        clusters[state_ref].append((left, right))
    states = sorted(clusters)
    if not states:
        raise ValueError("Cluster bootstrap received no states.")
    rng = random.Random(int(seed))
    micro_deltas: list[float] = []
    macro_deltas: list[float] = []
    for _ in range(int(samples)):
        sampled_left: list[dict[str, Any]] = []
        sampled_right: list[dict[str, Any]] = []
        for _cluster in states:
            selected = states[rng.randrange(len(states))]
            for left, right in clusters[selected]:
                sampled_left.append(left)
                sampled_right.append(right)
        micro_deltas.append(
            _accuracy_from_predictions(sampled_right)
            - _accuracy_from_predictions(sampled_left)
        )
        macro_deltas.append(
            _macro_task_accuracy(sampled_right) - _macro_task_accuracy(sampled_left)
        )

    point_micro = _accuracy_from_predictions(dense) - _accuracy_from_predictions(serialized)
    point_macro = _macro_task_accuracy(dense) - _macro_task_accuracy(serialized)
    return {
        "enabled": True,
        "samples": int(samples),
        "seed": int(seed),
        "confidence_level": 0.95,
        "resampling_unit": "state_ref",
        "cluster_count": len(states),
        "micro_accuracy_delta": {
            "point_estimate": point_micro,
            "ci_low": percentile(micro_deltas, 0.025),
            "ci_high": percentile(micro_deltas, 0.975),
        },
        "macro_task_accuracy_delta": {
            "point_estimate": point_macro,
            "ci_low": percentile(macro_deltas, 0.025),
            "ci_high": percentile(macro_deltas, 0.975),
        },
    }


def safe_ratio(numerator: float | int, denominator: float | int) -> float | None:
    value = float(denominator)
    return None if value == 0.0 else float(numerator) / value


def cost_comparison(method_results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any] | None:
    if "serialized" not in method_results or "dense" not in method_results:
        return None
    serialized = method_results["serialized"]
    dense = method_results["dense"]
    serialized_e2e = serialized["end_to_end_cost"]
    dense_e2e = dense["end_to_end_cost"]
    serialized_tokens = serialized_e2e["prompt_tokens"]
    dense_tokens = dense_e2e["prompt_tokens"]
    serialized_proxy = serialized_e2e["logical_attention_score_proxy"]
    dense_proxy = dense_e2e["logical_attention_score_proxy"]
    output: dict[str, Any] = {
        "accuracy_dense_minus_serialized": float(dense["metrics"]["accuracy"])
        - float(serialized["metrics"]["accuracy"]),
        "macro_task_accuracy_dense_minus_serialized": float(
            dense["metrics"]["macro_task_accuracy"]
        )
        - float(serialized["metrics"]["macro_task_accuracy"]),
        "end_to_end_speedup_serialized_seconds_over_dense_seconds": safe_ratio(
            serialized_e2e["critical_path_wall_seconds"],
            dense_e2e["critical_path_wall_seconds"],
        ),
        "end_to_end_dense_over_serialized_peak_allocated_memory": safe_ratio(
            dense_e2e["cuda_memory"]["peak_allocated_bytes_max_rank"],
            serialized_e2e["cuda_memory"]["peak_allocated_bytes_max_rank"],
        ),
        "mean_prompt_tokens_dense_over_serialized": safe_ratio(
            dense_tokens["mean"], serialized_tokens["mean"]
        ),
        "total_attention_proxy_dense_over_serialized": safe_ratio(
            dense_proxy["total_attention_score_elements"],
            serialized_proxy["total_attention_score_elements"],
        ),
        "attention_ratio_is_not_a_flops_ratio": True,
    }
    if serialized.get("model_only_cost") and dense.get("model_only_cost"):
        output["model_only_speedup_serialized_seconds_over_dense_seconds"] = safe_ratio(
            serialized["model_only_cost"]["critical_path_wall_seconds"],
            dense["model_only_cost"]["critical_path_wall_seconds"],
        )
    return output


def atomic_dump_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row), ensure_ascii=False, allow_nan=False) + "\n")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def clear_method_cache(dataset) -> None:
    for name in (
        "_matrix_cache",
        "_latent_cache",
        "_latent_path_cache",
        "_latent_identity_cache",
        "_latent_qa_stats_cache",
    ):
        cache = getattr(dataset, name, None)
        if isinstance(cache, dict):
            cache.clear()
    gc.collect()


def prewarm_latent_files(dataset) -> dict[str, Any]:
    """Warm the shared OS page cache without populating either method's LRU."""

    started = time.perf_counter()
    paths: dict[str, Path] = {}
    for record in dataset.records:
        path = dataset.latent_path_for_record(record)
        paths[str(path.resolve())] = path
    total_bytes = 0
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Cannot prewarm missing latent file: {path}")
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                total_bytes += len(chunk)
    return {
        "files": len(paths),
        "bytes": total_bytes,
        "elapsed_seconds": time.perf_counter() - started,
        "scope": "OS page cache only; dataset representation caches remain empty",
    }


def load_dense_interface(
    *,
    args: argparse.Namespace,
    llm: nn.Module,
    dense_dataset: DenseBenchmarkDataset,
    latent_contract: Mapping[str, Any],
    device: torch.device,
) -> tuple[DenseCrossAttentionSidecar, dict[str, Any]]:
    started = time.perf_counter()
    checkpoint_path = Path(str(args.dense_checkpoint)).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Dense checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Dense checkpoint payload must be a mapping.")
    observed_architecture = apply_checkpoint_architecture(args, checkpoint)
    first_latent = dense_dataset[0]["latent_map"]
    latent_shape = tuple(int(value) for value in first_latent.shape)
    hidden_size = int(llm.get_input_embeddings().embedding_dim)
    spatial_initializer, initializer_provenance = load_spatial_initializer(
        args.memory_init_checkpoint,
        latent_shape,
        hidden_size,
        latent_contract,
        args.model_name_or_path,
    )
    sidecar, install_report = build_sidecar(llm, spatial_initializer, args, device)
    expected_architecture = architecture_contract(
        args,
        latent_shape,
        hidden_size,
        latent_contract,
        initializer_provenance,
    )
    validate_checkpoint_contract(checkpoint, expected_architecture)
    state = checkpoint.get("trainable_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("Dense checkpoint is missing trainable_state_dict.")
    load_report = load_trainable_state_dict(sidecar, state)
    trainable_before_freeze = sum(
        int(parameter.numel()) for parameter in sidecar.parameters() if parameter.requires_grad
    )
    if trainable_before_freeze != int(load_report["parameters"]):
        raise RuntimeError(
            "Loaded dense state does not exactly cover the pre-freeze trainable boundary: "
            f"state={load_report['parameters']}, boundary={trainable_before_freeze}."
        )
    frozen_spatial = unique_parameter_count(sidecar.memory.spatial_backbone)
    sidecar_total = unique_parameter_count(sidecar)
    for parameter in sidecar.parameters():
        parameter.requires_grad_(False)
    for parameter in llm.parameters():
        parameter.requires_grad_(False)
    set_frozen_llm_execution_mode(llm, checkpoint_training=False)
    sidecar.eval()
    if llm.training or sidecar.training or any(parameter.requires_grad for parameter in llm.parameters()):
        raise RuntimeError("Dense benchmark failed to establish a fully frozen eval boundary.")
    gates = sidecar.gate_values()
    checkpoint_sha256 = sha256_file(checkpoint_path)
    checkpoint_bytes = checkpoint_path.stat().st_size
    initializer_path = Path(str(initializer_provenance["path"]))
    initializer_bytes = initializer_path.stat().st_size
    elapsed_seconds = time.perf_counter() - started
    report = {
        "elapsed_seconds": elapsed_seconds,
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": checkpoint_sha256,
            "bytes": checkpoint_bytes,
            "checkpoint_type": str(checkpoint.get("checkpoint_type", "")),
            "checkpoint_version": int(checkpoint.get("checkpoint_version", 0)),
        },
        "architecture_from_checkpoint": observed_architecture,
        "validated_architecture": expected_architecture,
        "initializer": initializer_provenance,
        "deployment_checkpoint_storage": {
            "dense_trainable_checkpoint_bytes": checkpoint_bytes,
            "spatial_initializer_checkpoint_bytes": initializer_bytes,
            "combined_bytes": checkpoint_bytes + initializer_bytes,
        },
        "install": install_report,
        "state_load": load_report,
        "parameters": {
            "deployment_extra_total": sidecar_total,
            "trained_state": int(load_report["parameters"]),
            "trainable_before_inference_freeze": trainable_before_freeze,
            "frozen_spatial_backbone": frozen_spatial,
            "trainable_during_benchmark": 0,
        },
        "gate_values_tanh": gates,
    }
    return sidecar, report


def print_method_summary(method: str, result: Mapping[str, Any]) -> None:
    metrics = result["metrics"]
    cost = result["end_to_end_cost"]
    task_text = ", ".join(
        f"{task}={float(values['accuracy']):.4f}"
        for task, values in sorted(metrics["by_task"].items())
    )
    print(
        f"method={method} accuracy={float(metrics['accuracy']):.4f} "
        f"macro={float(metrics['macro_task_accuracy']):.4f} "
        f"wall={float(cost['critical_path_wall_seconds']):.2f}s "
        f"records_per_second={float(cost['records_per_second']):.3f} "
        f"tasks[{task_text}]",
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> None:
    process_started = time.perf_counter()
    args = parse_args(argv)
    apply_runtime_environment(args)
    device: torch.device | None = None
    try:
        device = initialize_distributed_device(
            args.device,
            distributed_timeout_seconds=float(args.distributed_timeout_seconds),
        )
        seed_everything(int(args.seed) + distributed_rank())
        run_dir = build_distributed_run_dir(args.output_root, args.run_name)
        if is_main_process():
            atomic_dump_json(
                run_dir / "resolved_benchmark_config.json",
                {
                    "format": RESULT_FORMAT,
                    "created_at": local_timestamp(),
                    "source_config": str(args.config),
                    "source_config_sha256": sha256_file(args.config),
                    "resolved_args": {
                        key: value for key, value in vars(args).items() if key != "raw_config"
                    },
                    "config_snapshot": redact_config(args.raw_config),
                },
            )
        if is_main_process():
            print(
                f"benchmark methods={','.join(args.methods)} split={args.split} "
                f"batch_per_rank={args.batch_size} world_size={distributed_world_size()} "
                f"output={run_dir}",
                flush=True,
            )

        setup_started = time.perf_counter()
        metadata, latent_contract = run_on_rank_zero_and_broadcast(
            lambda: load_metadata_and_contract(args),
            "benchmark QA metadata validation",
        )
        serialized_dataset, dense_dataset, data_audit = build_benchmark_datasets(
            args, latent_contract
        )
        tokenizer = load_tokenizer(args)
        tokenizer.padding_side = "right"
        choice_audit = single_token_choice_ids(serialized_dataset.records, tokenizer)
        if choice_audit is None:
            raise ValueError("Formal benchmark choices are not unique single-token labels.")
        dataset_setup_seconds = time.perf_counter() - setup_started
        dataset_setup_seconds_by_rank = [
            float(value) for value in gather_object(dataset_setup_seconds)
        ]

        model_load_started = time.perf_counter()
        llm, model_dtype = load_llm_with_bounded_host_memory(args, device)
        disable_checkpointing = getattr(llm, "gradient_checkpointing_disable", None)
        if callable(disable_checkpointing):
            disable_checkpointing()
        set_frozen_llm_execution_mode(llm, checkpoint_training=False)
        model_load_seconds = time.perf_counter() - model_load_started
        model_load_seconds_by_rank = [
            float(value) for value in gather_object(model_load_seconds)
        ]
        qwen_audit = audit_frozen_qwen(llm, model_dtype)
        qwen_parameter_count = unique_parameter_count(llm)
        model_config = llm.config
        qwen_layers = int(getattr(model_config, "num_hidden_layers", 0))
        qwen_heads = int(getattr(model_config, "num_attention_heads", 0))
        if qwen_layers <= 0 or qwen_heads <= 0:
            raise ValueError("Qwen config does not expose positive layer/head counts.")
        latent_shape = [int(value) for value in latent_contract["latent_shape"]]
        memory_cells = int(latent_shape[-2]) * int(latent_shape[-1])
        model_info = {
            "qwen_layers": qwen_layers,
            "qwen_heads": qwen_heads,
            "memory_cells": memory_cells,
        }

        method_results: dict[str, Any] = {}
        predictions_by_method: dict[str, list[dict[str, Any]]] = {}
        sidecar: DenseCrossAttentionSidecar | None = None
        dense_setup_report: dict[str, Any] | None = None
        for method in args.methods:
            dataset = serialized_dataset if method == "serialized" else dense_dataset
            if dataset is None:
                raise RuntimeError("Dense dataset was not constructed.")
            if method == "dense":
                sidecar, dense_setup_report = load_dense_interface(
                    args=args,
                    llm=llm,
                    dense_dataset=dataset,
                    latent_contract=latent_contract,
                    device=device,
                )
                dense_setup_by_rank = [
                    float(value)
                    for value in gather_object(dense_setup_report["elapsed_seconds"])
                ]
                dense_setup_report["elapsed_seconds_by_rank"] = dense_setup_by_rank
                dense_setup_report["critical_path_elapsed_seconds"] = max(
                    dense_setup_by_rank
                )
                model_info["memory_cells"] = int(dataset[0]["latent_map"].shape[-2]) * int(
                    dataset[0]["latent_map"].shape[-1]
                )

            if bool(args.model_only) or int(args.warmup_batches) > 0:
                prepared_batches, preparation_seconds = prepare_model_only_batches(
                    method=method,
                    dataset=dataset,
                    tokenizer=tokenizer,
                    args=args,
                    device=device,
                )
            else:
                prepared_batches, preparation_seconds = [], 0.0
            model_only_cost = None
            if bool(args.model_only):
                model_only_cost = run_model_only(
                    method=method,
                    prepared_batches=prepared_batches,
                    dataset_size=len(dataset),
                    llm=llm,
                    sidecar=sidecar,
                    device=device,
                    model_dtype=model_dtype,
                    args=args,
                    model_info=model_info,
                    preparation_seconds=preparation_seconds,
                )
            elif prepared_batches and int(args.warmup_batches) > 0:
                for index in range(int(args.warmup_batches)):
                    execute_prepared_batch(
                        prepared_batches[index % len(prepared_batches)],
                        llm,
                        sidecar,
                        device,
                        model_dtype,
                    )
                synchronize_device(device)
            prepared_batches = []
            clear_method_cache(dataset)
            input_prewarm = (
                run_on_rank_zero_and_broadcast(
                    lambda: prewarm_latent_files(dataset),
                    f"{method} input-file prewarm",
                )
                if bool(args.prewarm_input_files)
                else {
                    "files": 0,
                    "bytes": 0,
                    "elapsed_seconds": 0.0,
                    "scope": "disabled",
                }
            )
            # File prewarming may resolve paths on rank 0. Keep every method's
            # timed dataset-side caches empty while retaining the shared OS cache.
            clear_method_cache(dataset)
            distributed_barrier()
            metrics, predictions, end_to_end_cost = run_end_to_end(
                method=method,
                dataset=dataset,
                llm=llm,
                sidecar=sidecar,
                tokenizer=tokenizer,
                device=device,
                model_dtype=model_dtype,
                args=args,
                model_info=model_info,
            )
            method_results[method] = {
                "input_representation": (
                    "complete standardized matrix serialized as labeled text"
                    if method == "serialized"
                    else "complete latent grid with 256-cell dense cross-attention memory"
                ),
                "metrics": metrics,
                "end_to_end_cost": end_to_end_cost,
                "model_only_cost": model_only_cost,
                "model_only_batch_preparation_seconds_local_rank": preparation_seconds,
                "input_file_prewarm": input_prewarm,
            }
            predictions_by_method[method] = predictions
            if is_main_process():
                print_method_summary(method, method_results[method])
                if bool(args.save_predictions):
                    atomic_dump_jsonl(run_dir / f"predictions_{method}.jsonl", predictions)

        paired = None
        bootstrap = None
        if "serialized" in predictions_by_method and "dense" in predictions_by_method:
            paired = paired_contingency(
                predictions_by_method["serialized"], predictions_by_method["dense"]
            )
            bootstrap = run_on_rank_zero_and_broadcast(
                lambda: state_cluster_bootstrap(
                    predictions_by_method["serialized"],
                    predictions_by_method["dense"],
                    samples=int(args.bootstrap_samples),
                    seed=int(args.bootstrap_seed),
                ),
                "state-cluster bootstrap",
            )

        process_wall_seconds_by_rank = [
            float(value)
            for value in gather_object(time.perf_counter() - process_started)
        ]
        result = {
            "format": RESULT_FORMAT,
            "completed_at": local_timestamp(),
            "total_process_wall_seconds": max(process_wall_seconds_by_rank),
            "total_process_wall_seconds_by_rank": process_wall_seconds_by_rank,
            "benchmark_order": list(args.methods),
            "world_size": distributed_world_size(),
            "device": {
                "type": device.type,
                "cuda_version": torch.version.cuda,
                "name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "capability": list(torch.cuda.get_device_capability(device))
                if device.type == "cuda"
                else None,
            },
            "setup": {
                "dataset_and_tokenizer_seconds_by_rank": dataset_setup_seconds_by_rank,
                "dataset_and_tokenizer_critical_seconds": max(
                    dataset_setup_seconds_by_rank
                ),
                "shared_qwen_load_seconds_by_rank": model_load_seconds_by_rank,
                "shared_qwen_load_critical_seconds": max(model_load_seconds_by_rank),
                "dense_interface": dense_setup_report,
            },
            "data_audit": data_audit,
            "qa_metadata": {
                "format": str(metadata.get("format", "")),
                "split_mode": str(metadata.get("split_mode", "")),
                "alignment_checkpoint_resolution": metadata.get(
                    "runtime_alignment_checkpoint_resolution"
                ),
            },
            "model": {
                "name_or_path": str(args.model_name_or_path),
                "qwen_parameter_count": qwen_parameter_count,
                "qwen_audit": qwen_audit,
                "qwen_layers": qwen_layers,
                "qwen_attention_heads": qwen_heads,
                "execution_dtype": str(model_dtype).replace("torch.", ""),
                "gradient_checkpointing_during_benchmark": False,
                "use_cache": False,
            },
            "fairness_contract": {
                "same_qa_records_and_order": "dense" in args.methods,
                "same_per_rank_batch_size": int(args.batch_size),
                "same_exact_nonpadding_distributed_sampler": True,
                "same_frozen_qwen_instance": len(args.methods) == 2,
                "same_candidate_restricted_next_token_scoring": True,
                "no_prompt_truncation": True,
                "primary_metric": "strict source-label accuracy",
                "secondary_extreme_metric": "stored-FP16 tie-aware accuracy",
                "generation_not_benchmarked": True,
                "end_to_end_input_file_cache_policy": (
                    "rank 0 prewarms every selected latent file before each method"
                    if bool(args.prewarm_input_files)
                    else "no explicit OS page-cache prewarm; prefer isolated method runs"
                ),
                "combined_run_order_caveat": (
                    "serialized runs before dense because sidecar installation mutates decoder layers; "
                    "run one method per process to audit order and OS-page-cache effects"
                ),
            },
            "methods": method_results,
            "paired_contingency": paired,
            "state_cluster_bootstrap": bootstrap,
            "comparison": cost_comparison(method_results),
        }
        if is_main_process():
            atomic_dump_json(run_dir / "benchmark_results.json", result)
            if paired is not None:
                atomic_dump_json(
                    run_dir / "paired_comparison.json",
                    {"contingency": paired, "state_cluster_bootstrap": bootstrap},
                )
            print(f"results={run_dir / 'benchmark_results.json'}", flush=True)
        distributed_barrier()
    finally:
        if distributed_is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
