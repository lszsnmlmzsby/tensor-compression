from __future__ import annotations

"""Train the full-grid field cross-attention interface for a frozen LLM.

The paper mode uses a Direct-QA initializer and cached field latents. The
Direct-Cross mode instead replays raw HDF5 patches and trains a field encoder
and prefix-free spatial memory from scratch. Both modes keep the LLM frozen and
never consume parsed coordinates, ``query_spec``, task IDs, or a
task-to-slot-count mapping.
"""

import argparse
import contextlib
import copy
import hashlib
import json
import math
import os
import signal
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.train_tensor_llm_adapter import (  # noqa: E402
    ExactDistributedEvalSampler,
    StateTaskGroupedBatchSampler,
    TensorReadoutQADataset,
    adapter_from_checkpoint,
    aggregate_evaluation_counts,
    append_jsonl,
    apply_runtime_environment,
    assert_finite_gradients,
    atomic_dump_json,
    atomic_torch_save,
    audit_choice_tokenization,
    audit_prompt_tokenization,
    average_trainable_gradients_by_record_count,
    build_distributed_run_dir,
    collate_tensor_readout,
    distributed_is_initialized,
    distributed_rank,
    distributed_world_size,
    initialize_distributed_device,
    is_main_process,
    last_nonpadding_indices,
    load_llm_with_bounded_host_memory,
    load_tokenizer,
    matched_coordinate_group_loss,
    qa_path,
    run_on_rank_zero_and_broadcast,
    seed_everything,
    set_frozen_llm_execution_mode,
    single_token_choice_ids,
    validate_adapter_checkpoint_payload,
    validate_atomic_group_batch_size,
    validate_qa_latent_contract,
)
from scripts.train_tensor_patch_text_alignment import (  # noqa: E402
    SpatialTransformerBlock,
    TensorPatchAlignmentAdapter,
    build_patch_encoder_config,
    require_h5py,
    sinusoidal_2d_position_encoding,
)
from tensor_compression.downstream.patch_qa_prompt import build_prompt  # noqa: E402
from tensor_compression.downstream.variable_shape import (  # noqa: E402
    VARIABLE_QA_FORMAT, VARIABLE_PROMPT_CONTRACT, ShapeBatchSampler,
    experiment_config, parse_shapes, record_shape, shape_name,
    balanced_screen_records, shape_matched_shuffle_indices,
)
from tensor_compression.downstream.patch_qa_contract import (  # noqa: E402
    PATCH_LATENT_AUDIT_FORMAT,
    PATCH_MATCHED_QA_FORMAT,
    PATCH_QA_PROMPT_CONTRACT,
    canonical_normalization,
    canonical_path,
    latent_identity_from_record,
    latent_qa_stats_from_record,
    sha256_file,
)
from tensor_compression.models.builders import build_model  # noqa: E402
from tensor_compression.utils.pipeline_config import (  # noqa: E402
    environment_override,
    first_nested,
    load_yaml_mapping,
    resolve_path_string,
)


IGNORE_INDEX = -100
CHECKPOINT_TYPE = "tensor_qwen_dense_cross_attention"
CHECKPOINT_VERSION = 1
SUPPORTED_EVAL_MODES = ("correct", "no_tensor", "zero_tensor", "shuffled")


def _config_value(config: Mapping[str, Any], paths: Sequence[str], default: Any = None) -> Any:
    value = first_nested(config, list(paths))
    return default if value is None else value


def _path_value(value: Any) -> str | None:
    if value is None or str(value).strip().lower() in {"", "none", "null"}:
        return None
    return resolve_path_string(value, PROJECT_ROOT)


def _csv_ints(value: Any) -> list[int]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [int(item) for item in value]
    return [int(item.strip()) for item in str(value or "").split(",") if item.strip()]


def _csv_floats(value: Any) -> list[float]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [float(item) for item in value]
    return [float(item.strip()) for item in str(value or "").split(",") if item.strip()]


def _csv_strings(value: Any) -> list[str]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item) for item in value]
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train full-grid field-memory cross-attention inside a frozen LLM decoder."
    )
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--profile", choices=("pilot", "full"))
    parser.add_argument("--qa-dir", type=str)
    parser.add_argument("--evaluate-only", action="store_true", help="Evaluate the best checkpoint in --resume's run, without training.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--max-train-records", type=int, default=None)
    parser.add_argument("--max-val-records", type=int, default=None)
    parser.add_argument("--max-test-records", type=int, default=None)
    parser.add_argument("--max-updates", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--gate-lr", type=float, default=None)
    parser.add_argument("--max-wall-clock-hours", type=float, default=None)
    parser.add_argument("--cross-attention-layers", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--input-source",
        choices=("precomputed_latent", "raw_hdf5"),
        default=None,
        help="Read the paper latent cache or replay normalized field patches directly from PDEBench HDF5.",
    )
    parser.add_argument("--hdf5-path", type=str, default=None)
    parser.add_argument(
        "--memory-init-mode",
        choices=("direct_qa_checkpoint", "scratch"),
        default=None,
    )
    parser.add_argument(
        "--latent-channel-policy",
        choices=("all", "value_only"),
        default=None,
        help="Preserve all cached channels or expose only the exact value channel for ablation.",
    )
    parser.add_argument("--evaluate-test", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--console-progress", action=argparse.BooleanOptionalAction, default=None)
    cli = parser.parse_args()
    config = experiment_config(load_yaml_mapping(cli.config), cli.profile)

    model_local = environment_override(
        "FIELD_TO_LLM_MODEL_DIR",
        _config_value(config, ["model.local_dir"]),
    )
    model_name = _config_value(config, ["model.name_or_path", "model.model_name_or_path"])
    args = argparse.Namespace()
    args.config = str(cli.config)
    args.experiment_profile = cli.profile
    args.shape_mode = str(_config_value(config, ["data.shape_mode"], "fixed"))
    args.train_shapes = parse_shapes(config["data"]["train_shapes"]) if args.shape_mode == "variable" else []
    args.evaluate_only = bool(cli.evaluate_only)
    args.model_name_or_path = _path_value(model_local) if model_local else str(model_name or "")
    args.cache_dir = _path_value(
        environment_override(
            "FIELD_TO_LLM_HF_HOME",
            _config_value(config, ["model.cache_dir", "storage.hf_home"]),
        )
    )
    args.hf_home = _path_value(
        environment_override(
            "FIELD_TO_LLM_HF_HOME",
            _config_value(config, ["storage.hf_home"]),
        )
    )
    args.qa_dir = _path_value(
        cli.qa_dir or environment_override(
            "FIELD_TO_LLM_VARIABLE_QA_DIR" if args.shape_mode == "variable" else "FIELD_TO_LLM_MATCHED_QA_DIR",
            _config_value(
                config,
                ["data.qa_dir", "patch_qa.matched_qa_dir", "patch_qa.stage2b_qa_dir"],
            ),
        )
    )
    args.input_source = str(
        cli.input_source
        or _config_value(config, ["data.input_source"], "precomputed_latent")
    ).strip().lower()
    args.hdf5_path = _path_value(
        cli.hdf5_path
        or environment_override(
            "PDEBENCH_HDF5",
            _config_value(config, ["data.hdf5_path", "patch_qa.hdf5_path"]),
        )
    )
    args.patch_size = int(_config_value(config, ["data.patch_size"], 16))
    args.raw_normalized_dtype = str(
        _config_value(config, ["data.raw_normalized_dtype"], "float16")
    ).strip().lower()
    args.latent_dir = _path_value(
        environment_override(
            "FIELD_TO_LLM_LATENT_DIR",
            _config_value(config, ["data.latent_dir", "patch_qa.latent_dir"]),
        )
    )
    args.qa_alignment_checkpoint = _path_value(
        _config_value(config, ["data.alignment_checkpoint", "patch_qa.alignment_checkpoint"])
    )
    args.memory_init_checkpoint = _path_value(
        _config_value(config, ["memory.init_checkpoint"])
    )
    args.output_root = _path_value(
        cli.output_root
        or environment_override(
            "FIELD_TO_LLM_RUNS_DIR",
            _config_value(config, ["training.output_root", "storage.output_root"]),
        )
    )
    args.run_name = str(
        cli.run_name
        or _config_value(config, ["training.run_name"], "tensor_qwen_cross_attention")
    )
    args.train_split = str(_config_value(config, ["data.train_split"], "train"))
    args.val_split = str(_config_value(config, ["data.val_split"], "val"))
    args.test_split = str(_config_value(config, ["data.test_split"], "test"))
    args.max_train_records = (
        cli.max_train_records
        if cli.max_train_records is not None
        else _config_value(config, ["data.max_train_records"])
    )
    args.max_val_records = (
        cli.max_val_records
        if cli.max_val_records is not None
        else _config_value(config, ["data.max_val_records"])
    )
    args.max_test_records = (
        cli.max_test_records
        if cli.max_test_records is not None
        else _config_value(config, ["data.max_test_records"])
    )
    args.record_subset_mode = str(
        _config_value(config, ["data.record_subset_mode"], "hash_state")
    )
    args.require_disjoint_splits = bool(
        _config_value(config, ["data.require_disjoint_splits"], True)
    )
    args.prefer_record_latent_ref = bool(
        _config_value(config, ["data.prefer_record_latent_ref"], False)
    )
    args.latent_cache_size = int(
        _config_value(config, ["data.input_cache_size", "data.latent_cache_size"], 8192)
    )
    args.num_workers = int(_config_value(config, ["data.num_workers"], 2))
    args.latent_channel_policy = str(
        cli.latent_channel_policy
        or _config_value(config, ["memory.latent_channel_policy"], "all")
    )

    args.device = str(_config_value(config, ["runtime.device"], "auto"))
    args.seed = int(_config_value(config, ["runtime.seed"], 42))
    args.shuffle_seed = int(_config_value(config, ["runtime.shuffle_seed"], args.seed))
    args.distributed_timeout_seconds = float(
        _config_value(config, ["runtime.distributed_timeout_seconds"], 7200.0)
    )
    args.serialize_llm_loading = bool(
        _config_value(config, ["runtime.serialize_llm_loading"], True)
    )
    args.low_cpu_mem_usage = bool(
        _config_value(config, ["runtime.low_cpu_mem_usage"], True)
    )
    args.min_host_memory_available_gib = float(
        _config_value(config, ["runtime.min_host_memory_available_gib"], 16.0)
    )
    args.max_wall_clock_hours = float(
        cli.max_wall_clock_hours
        if cli.max_wall_clock_hours is not None
        else _config_value(config, ["runtime.max_wall_clock_hours"], 8.0)
    )
    args.final_eval_reserve_minutes = float(
        _config_value(config, ["runtime.final_eval_reserve_minutes"], 50.0)
    )
    args.save_every_updates = int(
        _config_value(config, ["runtime.save_every_updates"], 1000)
    )
    args.resume = _path_value(cli.resume or _config_value(config, ["runtime.resume"]))
    args.console_progress = bool(
        cli.console_progress
        if cli.console_progress is not None
        else _config_value(config, ["runtime.console_progress"], True)
    )

    args.torch_dtype = str(_config_value(config, ["model.torch_dtype"], "bfloat16"))
    args.trust_remote_code = bool(_config_value(config, ["model.trust_remote_code"], False))
    args.llm_gradient_checkpointing = bool(
        _config_value(config, ["model.gradient_checkpointing"], True)
    )

    configured_layers = (
        cli.cross_attention_layers
        if cli.cross_attention_layers is not None
        else _config_value(config, ["cross_attention.layers_1based"])
    )
    args.cross_attention_layers = _csv_ints(configured_layers)
    args.bridge_dim = int(_config_value(config, ["cross_attention.bridge_dim"], 512))
    args.bridge_heads = int(_config_value(config, ["cross_attention.heads"], 8))
    args.bridge_dropout = float(_config_value(config, ["cross_attention.dropout"], 0.0))
    args.gate_init = float(_config_value(config, ["cross_attention.gate_init"], 0.0))
    args.value_fourier_bands = int(
        _config_value(config, ["memory.value_fourier_bands"], 4)
    )
    args.value_hidden_dim = int(_config_value(config, ["memory.value_hidden_dim"], 128))
    args.memory_init_mode = str(
        cli.memory_init_mode
        or _config_value(config, ["memory.init_mode"], "direct_qa_checkpoint")
    ).strip().lower()
    args.freeze_spatial_backbone = bool(
        _config_value(config, ["memory.freeze_spatial_backbone"], True)
    )
    field_encoder_config = _config_value(config, ["field_encoder"], {})
    spatial_adapter_config = _config_value(config, ["spatial_adapter"], {})
    if not isinstance(field_encoder_config, Mapping):
        raise ValueError("field_encoder must be a YAML mapping.")
    if not isinstance(spatial_adapter_config, Mapping):
        raise ValueError("spatial_adapter must be a YAML mapping.")
    args.field_encoder_config = copy.deepcopy(dict(field_encoder_config))
    args.spatial_adapter_config = copy.deepcopy(dict(spatial_adapter_config))
    args.field_encoder_trainable = bool(
        args.field_encoder_config.get("trainable", args.input_source == "raw_hdf5")
    )

    args.epochs = int(
        cli.epochs if cli.epochs is not None else _config_value(config, ["training.epochs"], 1)
    )
    args.batch_size = int(
        cli.batch_size
        if cli.batch_size is not None
        else _config_value(config, ["training.batch_size"], 3)
    )
    args.eval_batch_size = int(
        cli.eval_batch_size
        if cli.eval_batch_size is not None
        else _config_value(config, ["evaluation.batch_size"], 4)
    )
    args.gradient_accumulation_steps = int(
        cli.gradient_accumulation_steps
        if cli.gradient_accumulation_steps is not None
        else _config_value(config, ["training.gradient_accumulation_steps"], 1)
    )
    args.max_updates = int(
        cli.max_updates
        if cli.max_updates is not None
        else _config_value(config, ["training.max_updates"], 0)
        or 0
    )
    args.lr = float(cli.lr if cli.lr is not None else _config_value(config, ["training.lr"], 5e-5))
    args.gate_lr = float(
        cli.gate_lr
        if cli.gate_lr is not None
        else _config_value(config, ["training.gate_lr"], 5e-4)
    )
    args.lr_scheduler = str(_config_value(config, ["training.lr_scheduler"], "cosine"))
    args.warmup_ratio = float(_config_value(config, ["training.warmup_ratio"], 0.03))
    args.min_lr_ratio = float(_config_value(config, ["training.min_lr_ratio"], 0.2))
    args.weight_decay = float(_config_value(config, ["training.weight_decay"], 1e-4))
    args.grad_clip_norm = float(_config_value(config, ["training.grad_clip_norm"], 1.0))
    args.choice_ce_weight = float(
        _config_value(config, ["training.choice_ce_weight"], 1.0)
    )
    args.full_answer_ce_weight = float(
        _config_value(config, ["training.full_answer_ce_weight"], 0.02)
    )
    args.matched_group_weight = float(
        _config_value(config, ["training.matched_group_weight"], 0.1)
    )
    args.matched_group_margin = float(
        _config_value(config, ["training.matched_group_margin"], 0.5)
    )
    args.value_reconstruction_weight = float(
        _config_value(config, ["training.value_reconstruction_weight"], 0.01)
    )
    args.max_prompt_tokens = int(
        _config_value(config, ["training.max_prompt_tokens"], 512)
    )
    args.max_target_tokens = int(
        _config_value(config, ["training.max_target_tokens"], 8)
    )
    args.append_eos = bool(_config_value(config, ["training.append_eos"], True))
    args.prompt_template = str(
        _config_value(config, ["training.prompt_template"], "task_specific")
    )
    args.log_interval = int(_config_value(config, ["training.log_interval"], 20))
    args.questions_per_state_group = int(
        _config_value(config, ["training.questions_per_state_group"], 3)
    )

    args.screening_records = int(
        _config_value(config, ["evaluation.screening_records"], 2000)
    )
    args.screening_fractions = _csv_floats(
        _config_value(config, ["evaluation.screening_fractions"], [0.25, 0.5, 0.75, 1.0])
    )
    args.eval_modes = _csv_strings(
        _config_value(
            config,
            ["evaluation.final_modes"],
            ["correct", "no_tensor", "zero_tensor", "shuffled"],
        )
    )
    args.evaluate_test = bool(
        cli.evaluate_test
        if cli.evaluate_test is not None
        else _config_value(config, ["evaluation.evaluate_test"], True)
    )
    args.selection_metric = str(
        _config_value(config, ["evaluation.selection_metric"], "macro_accuracy")
    )
    args.raw_config = copy.deepcopy(dict(config))
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    if getattr(args, "evaluate_only", False) and not args.resume:
        raise ValueError("--evaluate-only requires --resume pointing to this run's last.pt.")
    if getattr(args, "shape_mode", "fixed") not in {"fixed", "variable"}:
        raise ValueError("data.shape_mode must be fixed or variable.")
    if getattr(args, "shape_mode", "fixed") == "variable":
        if args.input_source != "raw_hdf5" or args.memory_init_mode != "scratch":
            raise ValueError("Variable shapes require raw_hdf5 and scratch initialization.")
        if any(value is not None for value in (args.max_train_records, args.max_val_records, args.max_test_records)):
            raise ValueError("Variable-shape runs use complete profile datasets; select --profile pilot for a small experiment.")
        if args.questions_per_state_group != 3:
            raise ValueError("Variable-shape QA uses atomic groups of three.")
    required_paths = {
        "model_name_or_path": args.model_name_or_path,
        "qa_dir": args.qa_dir,
        "output_root": args.output_root,
    }
    if args.input_source == "precomputed_latent":
        required_paths.update(
            {
                "latent_dir": args.latent_dir,
                "qa_alignment_checkpoint": args.qa_alignment_checkpoint,
            }
        )
    elif args.input_source == "raw_hdf5":
        required_paths["hdf5_path"] = args.hdf5_path
    else:
        raise ValueError(
            "data.input_source must be 'precomputed_latent' or 'raw_hdf5'."
        )
    if args.memory_init_mode == "direct_qa_checkpoint":
        required_paths["memory_init_checkpoint"] = args.memory_init_checkpoint
    elif args.memory_init_mode != "scratch":
        raise ValueError(
            "memory.init_mode must be 'direct_qa_checkpoint' or 'scratch'."
        )
    missing = [name for name, value in required_paths.items() if not value]
    if missing:
        raise ValueError(f"Missing required cross-attention configuration paths: {missing}.")
    if args.input_source == "raw_hdf5":
        if args.memory_init_mode != "scratch":
            raise ValueError(
                "raw_hdf5 Direct-Cross requires memory.init_mode=scratch so no Direct-QA "
                "initializer enters the model path."
            )
        if bool(args.freeze_spatial_backbone):
            raise ValueError(
                "raw_hdf5 Direct-Cross requires memory.freeze_spatial_backbone=false."
            )
        if not bool(args.field_encoder_trainable):
            raise ValueError("raw_hdf5 Direct-Cross requires field_encoder.trainable=true.")
        if args.patch_size <= 0:
            raise ValueError("data.patch_size must be positive.")
        if args.raw_normalized_dtype not in {"float32", "float16"}:
            raise ValueError("data.raw_normalized_dtype must be float32 or float16.")
        if not args.field_encoder_config or not args.spatial_adapter_config:
            raise ValueError(
                "raw_hdf5 Direct-Cross requires explicit field_encoder and spatial_adapter mappings."
            )
        field_model = args.field_encoder_config.get("model")
        if not isinstance(field_model, Mapping):
            raise ValueError("raw_hdf5 Direct-Cross requires field_encoder.model settings.")
        field_norm = str(field_model.get("norm", "group")).strip().lower()
        if field_norm not in {"group", "identity"}:
            raise ValueError(
                "Direct-Cross supports only stateless group/identity normalization; "
                "BatchNorm running buffers are not part of the manual DDP checkpoint contract."
            )
        configured_dropouts = {
            "field_encoder.model.dropout": float(field_model.get("dropout", 0.0)),
            "spatial_adapter.dropout": float(
                args.spatial_adapter_config.get("dropout", 0.0)
            ),
            "cross_attention.dropout": float(args.bridge_dropout),
        }
        nonzero_dropouts = {
            name: value for name, value in configured_dropouts.items() if value != 0.0
        }
        if nonzero_dropouts:
            raise ValueError(
                "Direct-Cross scratch currently requires dropout=0 for exact multi-rank "
                f"resume reproducibility; observed={nonzero_dropouts}."
            )
    elif args.memory_init_mode == "scratch":
        raise ValueError(
            "memory.init_mode=scratch is currently reserved for raw_hdf5 Direct-Cross."
        )
    if not args.cross_attention_layers:
        raise ValueError("cross_attention.layers_1based must contain at least one layer.")
    if len(set(args.cross_attention_layers)) != len(args.cross_attention_layers):
        raise ValueError("cross-attention layer indices must be unique.")
    if any(index <= 0 for index in args.cross_attention_layers):
        raise ValueError("cross-attention layers use positive one-based indices.")
    if args.bridge_dim <= 0 or args.bridge_heads <= 0 or args.bridge_dim % args.bridge_heads:
        raise ValueError("bridge_dim must be positive and divisible by heads.")
    if not 0.0 <= args.bridge_dropout < 1.0:
        raise ValueError("cross-attention dropout must be in [0,1).")
    if args.batch_size <= 0 or args.gradient_accumulation_steps <= 0:
        raise ValueError("batch sizes and accumulation steps must be positive.")
    if args.latent_channel_policy not in {"all", "value_only"}:
        raise ValueError("memory.latent_channel_policy must be 'all' or 'value_only'.")
    validate_atomic_group_batch_size(
        args.batch_size,
        args.questions_per_state_group,
        context="Full-grid cross-attention matched-QA training",
    )
    if args.epochs <= 0 or args.eval_batch_size <= 0:
        raise ValueError("epochs and evaluation batch size must be positive.")
    if args.max_wall_clock_hours <= 0.0 or args.final_eval_reserve_minutes < 0.0:
        raise ValueError("wall-clock budget must be positive and evaluation reserve non-negative.")
    if args.final_eval_reserve_minutes >= args.max_wall_clock_hours * 60.0:
        raise ValueError("final evaluation reserve must be smaller than the total wall-clock budget.")
    if args.lr <= 0.0 or args.gate_lr <= 0.0:
        raise ValueError("training and gate learning rates must be positive.")
    if args.lr_scheduler not in {"constant", "cosine"}:
        raise ValueError("training.lr_scheduler must be constant or cosine.")
    unsupported = sorted(set(args.eval_modes) - set(SUPPORTED_EVAL_MODES))
    if unsupported:
        raise ValueError(f"Unsupported final evaluation modes: {unsupported}.")
    if "correct" not in args.eval_modes or (
        getattr(args, "shape_mode", "fixed") != "variable" and "no_tensor" not in args.eval_modes
    ):
        raise ValueError("Final evaluation must include correct and no_tensor modes.")
    if args.selection_metric not in {"accuracy", "macro_accuracy", "normalized_accuracy"}:
        raise ValueError(
            "evaluation.selection_metric must be accuracy, macro_accuracy, or normalized_accuracy."
        )
    for fraction in args.screening_fractions:
        if not 0.0 < fraction <= 1.0:
            raise ValueError("Every screening fraction must lie in (0,1].")


class DirectFieldEncoder(nn.Module):
    """Only the encoder path used by Direct-Cross; no autoencoder decoder exists here."""

    def __init__(self, compressor: nn.Module, dynamic_grid: bool = False) -> None:
        super().__init__()
        self.dynamic_grid = bool(dynamic_grid)
        required = ("encoder", "to_latent", "input_size", "latent_grid", "in_channels", "latent_dim")
        missing = [name for name in required if not hasattr(compressor, name)]
        if missing:
            raise TypeError(f"Configured field encoder is missing attributes: {missing}.")
        if not bool(getattr(compressor, "preserve_input_channels", False)):
            raise ValueError(
                "Direct-Cross field encoder must preserve its exact normalized input channel."
            )
        self.encoder = compressor.encoder
        self.to_latent = compressor.to_latent
        self.input_size = tuple(int(value) for value in compressor.input_size)
        self.latent_grid = tuple(int(value) for value in compressor.latent_grid)
        self.in_channels = int(compressor.in_channels)
        self.latent_dim = int(compressor.latent_dim)

    def forward(self, normalized_field: torch.Tensor) -> torch.Tensor:
        if normalized_field.ndim != 4:
            raise ValueError(
                f"Direct-Cross field input must be [B,C,H,W], got {tuple(normalized_field.shape)}."
            )
        spatial_shape = tuple(int(value) for value in normalized_field.shape[-2:])
        if int(normalized_field.shape[1]) != self.in_channels or (
            not self.dynamic_grid and spatial_shape != self.input_size
        ):
            raise ValueError(
                "Direct-Cross field input differs from the encoder contract: "
                f"observed={tuple(normalized_field.shape[1:])}, "
                f"expected={(self.in_channels, *self.input_size)}."
            )
        learned = self.to_latent(self.encoder(normalized_field))
        expected_grid = tuple(normalized_field.shape[-2:]) if self.dynamic_grid else self.latent_grid
        if tuple(int(value) for value in learned.shape[-2:]) != expected_grid:
            raise RuntimeError(
                f"Field encoder produced grid {tuple(learned.shape[-2:])}, expected {expected_grid}."
            )
        return torch.cat([normalized_field.to(dtype=learned.dtype), learned], dim=1)


class FullGridSpatialBackbone(nn.Module):
    """Cross-attention memory backbone with no prefix projection or soft-prompt head."""

    adapter_type = "spatial_transformer"

    def __init__(
        self,
        *,
        latent_channels: int,
        latent_grid: Sequence[int],
        adapter_dim: int,
        adapter_layers: int,
        adapter_heads: int,
        dropout: float,
        dynamic_grid: bool = False,
    ) -> None:
        super().__init__()
        self.dynamic_grid = bool(dynamic_grid)
        self.latent_grid = tuple(int(value) for value in latent_grid)
        if len(self.latent_grid) != 2 or any(value <= 0 for value in self.latent_grid):
            raise ValueError(f"Invalid full-grid spatial shape: {self.latent_grid}.")
        self.latent_token_count = int(self.latent_grid[0] * self.latent_grid[1])
        self.adapter_dim = int(adapter_dim)
        if self.adapter_dim <= 0 or int(adapter_layers) <= 0 or int(adapter_heads) <= 0:
            raise ValueError("Spatial adapter dimension, layers, and heads must be positive.")
        if self.adapter_dim % int(adapter_heads):
            raise ValueError("Spatial adapter dimension must be divisible by its head count.")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("Spatial adapter dropout must be in [0,1).")
        self.latent_projection = nn.Linear(int(latent_channels), self.adapter_dim)
        self.local_residual_projection = nn.Linear(int(latent_channels), self.adapter_dim)
        self.register_buffer(
            "spatial_pos_encoding",
            sinusoidal_2d_position_encoding(*self.latent_grid, self.adapter_dim),
            persistent=True,
        )
        self.register_buffer("spatial_pos_scale", torch.tensor(1.0), persistent=True)
        self.register_buffer("local_residual_scale", torch.tensor(1.0), persistent=True)
        self.blocks = nn.ModuleList(
            [
                SpatialTransformerBlock(
                    self.adapter_dim,
                    int(adapter_heads),
                    float(dropout),
                )
                for _ in range(int(adapter_layers))
            ]
        )

    def spatial_input_states(self, latent_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if latent_map.ndim != 4 or (not self.dynamic_grid and tuple(int(value) for value in latent_map.shape[-2:]) != self.latent_grid):
            raise ValueError(
                f"Expected encoded field [B,C,{self.latent_grid[0]},{self.latent_grid[1]}], "
                f"got {tuple(latent_map.shape)}."
            )
        latent_tokens = latent_map.flatten(2).transpose(1, 2).contiguous()
        latent_tokens = latent_tokens.to(dtype=self.latent_projection.weight.dtype)
        local_residual = self.local_residual_projection(latent_tokens)
        content = self.latent_projection(latent_tokens)
        grid = tuple(int(value) for value in latent_map.shape[-2:])
        position_table = self.spatial_pos_encoding if grid == self.latent_grid else sinusoidal_2d_position_encoding(*grid, self.adapter_dim)
        position = self.spatial_pos_scale.to(dtype=content.dtype) * position_table.to(
            device=content.device,
            dtype=content.dtype,
        )
        return content + position, local_residual


@dataclass
class DenseMemoryState:
    content: torch.Tensor
    value: torch.Tensor
    reconstruction_loss: torch.Tensor


class DenseTensorMemory(nn.Module):
    """Full-grid spatial states plus a shared, query-independent exact-z path."""

    def __init__(
        self,
        spatial_backbone: nn.Module,
        fourier_bands: int,
        value_hidden_dim: int,
        freeze_spatial_backbone: bool,
        field_encoder: DirectFieldEncoder | None = None,
    ) -> None:
        super().__init__()
        if spatial_backbone.adapter_type != "spatial_transformer":
            raise ValueError("Dense memory requires a one-token-per-cell spatial_transformer initializer.")
        self.spatial_backbone = spatial_backbone
        self.field_encoder = field_encoder
        self.memory_dim = int(spatial_backbone.adapter_dim)
        self.fourier_bands = int(fourier_bands)
        self.freeze_spatial_backbone = bool(freeze_spatial_backbone)
        if self.fourier_bands < 0:
            raise ValueError("value_fourier_bands must be non-negative.")
        for name, parameter in self.spatial_backbone.named_parameters():
            # A legacy Direct-QA initializer still contains the prefix-only output
            # head. It is deliberately absent from the final memory path.
            prefix_only = str(name).startswith("output.")
            parameter.requires_grad_(not self.freeze_spatial_backbone and not prefix_only)
        if self.freeze_spatial_backbone:
            self.spatial_backbone.eval()
        basis_dim = 4 + 2 * self.fourier_bands
        hidden_dim = max(8, int(value_hidden_dim))
        self.content_norm = nn.LayerNorm(self.memory_dim)
        self.value_encoder = nn.Sequential(
            nn.Linear(basis_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.memory_dim),
        )
        self.value_reconstruction = nn.Linear(self.memory_dim, 1)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_spatial_backbone:
            self.spatial_backbone.eval()
        return self

    def _spatial_states(self, latent_map: torch.Tensor) -> torch.Tensor:
        def compute() -> torch.Tensor:
            states, local_residual = self.spatial_backbone.spatial_input_states(latent_map)
            for block in self.spatial_backbone.blocks:
                states = block(states)
            scale = self.spatial_backbone.local_residual_scale.to(dtype=states.dtype)
            return states + scale * local_residual

        if self.freeze_spatial_backbone:
            with torch.no_grad():
                return compute().detach()
        return compute()

    def _value_basis(self, z_values: torch.Tensor) -> torch.Tensor:
        z = z_values.float()
        features = [z, torch.tanh(z), z.abs(), torch.sign(z)]
        if self.fourier_bands:
            frequencies = torch.pow(
                z.new_tensor(2.0),
                torch.arange(self.fourier_bands, device=z.device, dtype=z.dtype),
            )
            angles = math.pi * z.unsqueeze(-1) * frequencies
            features.extend([torch.sin(angles), torch.cos(angles)])
        expanded: list[torch.Tensor] = []
        for feature in features:
            expanded.append(feature.unsqueeze(-1) if feature.ndim == z.ndim else feature)
        return torch.cat(expanded, dim=-1)

    def forward(self, field_input: torch.Tensor) -> DenseMemoryState:
        if field_input.ndim != 4 or int(field_input.shape[1]) < 1:
            raise ValueError(f"Expected field input [B,C,H,W], got {tuple(field_input.shape)}.")
        # The scalar path reads the exact standardized input, independently of
        # learned encoder channel ordering. For the paper cache this is channel 0;
        # for Direct-Cross it is the raw normalized single-channel grid.
        z_values = field_input[:, 0].flatten(1)
        latent_map = self.field_encoder(field_input) if self.field_encoder is not None else field_input
        content = self.content_norm(self._spatial_states(latent_map))
        if int(z_values.shape[1]) != int(content.shape[1]):
            raise ValueError(
                "Exact scalar grid and spatial memory must have one-to-one cells: "
                f"values={z_values.shape[1]}, memory={content.shape[1]}."
            )
        value = self.value_encoder(self._value_basis(z_values))
        reconstructed = self.value_reconstruction(value).squeeze(-1).float()
        reconstruction_loss = F.smooth_l1_loss(reconstructed, z_values.float())
        return DenseMemoryState(
            content=content,
            value=value,
            reconstruction_loss=reconstruction_loss,
        )


class GatedTensorCrossAttention(nn.Module):
    """Every text token may attend to every tensor cell; no evidence slots exist."""

    def __init__(
        self,
        llm_dim: int,
        memory_dim: int,
        bridge_dim: int,
        heads: int,
        dropout: float,
        gate_init: float,
    ) -> None:
        super().__init__()
        self.bridge_dim = int(bridge_dim)
        self.heads = int(heads)
        self.head_dim = self.bridge_dim // self.heads
        self.dropout = float(dropout)
        self.query_norm = nn.LayerNorm(int(llm_dim))
        self.memory_norm = nn.LayerNorm(int(memory_dim))
        self.q_proj = nn.Linear(int(llm_dim), self.bridge_dim)
        self.k_proj = nn.Linear(int(memory_dim), self.bridge_dim)
        self.v_content_proj = nn.Linear(int(memory_dim), self.bridge_dim)
        self.v_value_proj = nn.Linear(int(memory_dim), self.bridge_dim, bias=False)
        self.out_proj = nn.Linear(self.bridge_dim, int(llm_dim))
        self.gate = nn.Parameter(torch.tensor(float(gate_init), dtype=torch.float32))
        self._memory: DenseMemoryState | None = None
        self.enabled = True

    def bind_memory(self, memory: DenseMemoryState | None, enabled: bool = True) -> None:
        self._memory = memory
        self.enabled = bool(enabled)

    def clear_memory(self) -> None:
        self._memory = None
        self.enabled = True

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, tokens, _dim = tensor.shape
        return tensor.view(batch, tokens, self.heads, self.head_dim).transpose(1, 2)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        memory = self._memory
        if not self.enabled or memory is None:
            return hidden_states
        normalized_memory = self.memory_norm(memory.content)
        query = self._split_heads(self.q_proj(self.query_norm(hidden_states)))
        key = self._split_heads(self.k_proj(normalized_memory))
        value = self._split_heads(
            self.v_content_proj(normalized_memory) + self.v_value_proj(memory.value)
        )
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).contiguous().view(
            hidden_states.shape[0], hidden_states.shape[1], self.bridge_dim
        )
        residual = self.out_proj(attended).to(dtype=hidden_states.dtype)
        gate = torch.tanh(self.gate).to(dtype=hidden_states.dtype)
        return hidden_states + gate * residual


class ConditionedDecoderLayer(nn.Module):
    """Version-light wrapper that preserves the decoder layer's output container."""

    def __init__(self, base_layer: nn.Module, bridge: GatedTensorCrossAttention) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.bridge = bridge

    def forward(self, *args, **kwargs):
        output = self.base_layer(*args, **kwargs)
        if isinstance(output, torch.Tensor):
            return self.bridge(output)
        if isinstance(output, tuple):
            if not output or not isinstance(output[0], torch.Tensor):
                raise TypeError("Qwen decoder layer returned an unsupported tuple.")
            return (self.bridge(output[0]), *output[1:])
        if isinstance(output, list):
            if not output or not isinstance(output[0], torch.Tensor):
                raise TypeError("Qwen decoder layer returned an unsupported list.")
            return [self.bridge(output[0]), *output[1:]]
        raise TypeError(f"Unsupported Qwen decoder layer output: {type(output).__name__}.")


class DenseCrossAttentionSidecar(nn.Module):
    """Only this sidecar is optimized, synchronized, and checkpointed."""

    def __init__(
        self,
        memory: DenseTensorMemory,
        bridges: Sequence[GatedTensorCrossAttention],
        layers_1based: Sequence[int],
    ) -> None:
        super().__init__()
        self.memory = memory
        self.bridges = nn.ModuleList(list(bridges))
        self.layers_1based = tuple(int(value) for value in layers_1based)
        self._bound_state: DenseMemoryState | None = None

    def install(self, llm: nn.Module) -> dict[str, Any]:
        decoder = getattr(llm, "model", None)
        layers = getattr(decoder, "layers", None)
        if not isinstance(layers, nn.ModuleList):
            raise TypeError("The selected Qwen model does not expose model.layers as ModuleList.")
        count = len(layers)
        if any(index > count for index in self.layers_1based):
            raise ValueError(
                f"cross-attention layers {self.layers_1based} exceed Qwen's {count} decoder layers."
            )
        for one_based, bridge in zip(self.layers_1based, self.bridges):
            zero_based = one_based - 1
            if isinstance(layers[zero_based], ConditionedDecoderLayer):
                raise RuntimeError(f"Qwen layer {one_based} is already cross-attention wrapped.")
            layers[zero_based] = ConditionedDecoderLayer(layers[zero_based], bridge)
        return {
            "decoder_layers": count,
            "layers_1based": list(self.layers_1based),
            "layers_zero_based": [value - 1 for value in self.layers_1based],
        }

    def bind(self, latent_map: torch.Tensor | None, mode: str = "correct") -> DenseMemoryState | None:
        if self._bound_state is not None:
            raise RuntimeError("A tensor memory is already bound; clear it after backward before rebinding.")
        if mode == "no_tensor":
            for bridge in self.bridges:
                bridge.bind_memory(None, enabled=False)
            return None
        if latent_map is None:
            raise ValueError(f"Evaluation mode {mode!r} requires a latent tensor.")
        selected = torch.zeros_like(latent_map) if mode == "zero_tensor" else latent_map
        state = self.memory(selected)
        self._bound_state = state
        for bridge in self.bridges:
            bridge.bind_memory(state, enabled=True)
        return state

    def clear(self) -> None:
        for bridge in self.bridges:
            bridge.clear_memory()
        self._bound_state = None

    def gate_values(self) -> list[float]:
        return [float(torch.tanh(bridge.gate.detach().float()).cpu().item()) for bridge in self.bridges]


def autocast_context(device: torch.device, model_dtype: torch.dtype):
    if device.type == "cuda" and model_dtype in {torch.float16, torch.bfloat16}:
        return torch.autocast(device_type="cuda", dtype=model_dtype)
    return contextlib.nullcontext()


def validate_relocated_qa_latent_contract(
    metadata: Mapping[str, Any],
    configured_alignment_checkpoint: str | Path | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Accept a relocated Stage-1 file only when its immutable hash still matches."""

    declared_checkpoint = str(metadata.get("alignment_checkpoint", "")).strip()
    configured_checkpoint = str(configured_alignment_checkpoint or "").strip()
    declared_path = canonical_path(declared_checkpoint) if declared_checkpoint else None
    configured_path = canonical_path(configured_checkpoint) if configured_checkpoint else None
    relocated = bool(
        declared_path
        and configured_path
        and declared_path != configured_path
    )

    validation_metadata = dict(metadata)
    if relocated:
        # The shared validator checks both the formal metadata contract and the
        # configured file's SHA-256. Only replace the host-specific path alias.
        validation_metadata["alignment_checkpoint"] = configured_checkpoint
    validated = validate_qa_latent_contract(
        validation_metadata,
        configured_alignment_checkpoint=configured_alignment_checkpoint,
        require_formal_contract=True,
    )
    if not isinstance(validated, Mapping):
        raise RuntimeError("Formal matched QA did not produce an immutable latent contract.")
    contract = dict(validated)
    if relocated:
        # Latent payloads retain the build-time path and audit it separately
        # from the immutable hash. Keep that declared alias in their contract.
        contract["alignment_checkpoint"] = declared_path
    resolution = {
        "identity": "sha256",
        "declared_path": declared_path,
        "configured_path": configured_path,
        "path_relocated": relocated,
        "verified_sha256": str(contract.get("alignment_checkpoint_sha256", "")),
    }
    return contract, resolution


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def validate_raw_hdf5_qa_contract(
    metadata: Mapping[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind Direct-Cross to raw patches and the QA value-space, not Stage-1 latents."""

    hdf5_path = Path(str(args.hdf5_path)).expanduser()
    if not hdf5_path.is_file():
        raise FileNotFoundError(f"Direct-Cross PDEBench HDF5 file is missing: {hdf5_path}.")
    dynamic = getattr(args, "shape_mode", "fixed") == "variable"
    if (Path(args.qa_dir) / ".build_in_progress.json").exists():
        raise ValueError("QA generation is incomplete; select a completed dataset directory.")
    if str(metadata.get("format", "")) != (VARIABLE_QA_FORMAT if dynamic else PATCH_MATCHED_QA_FORMAT):
        raise ValueError("Direct-Cross requires the formal matched-QA dataset format.")
    if str(metadata.get("prompt_contract", "")) != (VARIABLE_PROMPT_CONTRACT if dynamic else PATCH_QA_PROMPT_CONTRACT):
        raise ValueError("Direct-Cross matched-QA prompt contract is missing or stale.")
    if str(metadata.get("latent_audit_format", "")) != PATCH_LATENT_AUDIT_FORMAT:
        raise ValueError("Direct-Cross requires per-record raw-value normalization audits.")
    metadata_patch_size = int(metadata.get("patch_size", 0))
    if not dynamic and metadata_patch_size != int(args.patch_size):
        raise ValueError(
            "Direct-Cross patch size differs from the fixed matched-QA records: "
            f"configured={args.patch_size}, metadata={metadata_patch_size}."
        )
    observed_normalization = metadata.get("encoder_input_normalization")
    configured_normalization = args.field_encoder_config.get("normalization")
    if not isinstance(observed_normalization, Mapping) or not isinstance(
        configured_normalization, Mapping
    ):
        raise ValueError(
            "Direct-Cross requires explicit QA and field-encoder normalization mappings."
        )
    observed_normalization = canonical_normalization(observed_normalization)
    configured_normalization = canonical_normalization(configured_normalization)
    required_normalization = canonical_normalization(
        {
            "mode": "zscore",
            "scope": "channel",
            "stats_path": None,
            "clip_min": None,
            "clip_max": None,
        }
    )
    if observed_normalization != configured_normalization:
        raise ValueError(
            "Direct-Cross field-encoder normalization differs from the matched-QA value space: "
            f"configured={configured_normalization}, metadata={observed_normalization}."
        )
    if configured_normalization != required_normalization:
        raise ValueError(
            "Direct-Cross currently requires unclipped per-patch channel z-score normalization."
        )
    if str(metadata.get("qa_value_space", "")) != "per_patch_zscore_from_raw_patch":
        raise ValueError("Matched-QA metadata does not describe raw-patch z-score targets.")
    storage_dtype = str(metadata.get("storage_dtype", "")).strip().lower()
    if storage_dtype not in {"float16", "float32"}:
        raise ValueError(
            "Matched-QA metadata has an unsupported preserved-value storage dtype: "
            f"{storage_dtype!r}."
        )
    if str(args.raw_normalized_dtype) != storage_dtype:
        raise ValueError(
            "Direct-Cross runtime quantization must match the preserved value channel used "
            "to construct the matched-QA targets: "
            f"configured={args.raw_normalized_dtype!r}, metadata={storage_dtype!r}."
        )
    fields = metadata.get("fields")
    if not isinstance(fields, Sequence) or isinstance(fields, (str, bytes)) or not fields:
        raise ValueError("Matched-QA metadata does not declare its PDEBench fields.")
    declared_split_sha256 = metadata.get("output_split_sha256")
    if not isinstance(declared_split_sha256, Mapping):
        raise ValueError("Matched-QA metadata does not bind its output JSONL files.")
    actual_split_sha256: dict[str, str] = {}
    for split in dict.fromkeys((args.train_split, args.val_split, args.test_split)):
        split_path = qa_path(args.qa_dir, str(split))
        if not split_path.is_file():
            raise FileNotFoundError(f"Matched-QA split is missing: {split_path}.")
        actual_digest = sha256_file(split_path)
        declared_digest = str(declared_split_sha256.get(str(split), "")).lower()
        if not declared_digest or actual_digest != declared_digest:
            raise ValueError(
                "Matched-QA JSONL differs from its immutable metadata digest: "
                f"split={split!r}, declared={declared_digest!r}, actual={actual_digest!r}."
            )
        actual_split_sha256[str(split)] = actual_digest
    identity = {
        "format": "raw_hdf5_normalized_patch_v1",
        "patch_size": int(args.patch_size),
        "input_shape": [1, int(args.patch_size), int(args.patch_size)],
        "normalization": configured_normalization,
        "raw_normalized_dtype": str(args.raw_normalized_dtype),
        "fields": [str(value) for value in fields],
        "qa_output_split_sha256": actual_split_sha256,
        "qa_source_split_sha256": copy.deepcopy(metadata.get("source_split_sha256", {})),
    }
    if dynamic:
        if metadata.get("profile") != args.experiment_profile:
            raise ValueError("QA profile differs from --profile; pilot and full datasets are separate.")
        if list(fields) != list(args.raw_config["data"]["fields"]):
            raise ValueError("QA fields differ from the selected configuration; rebuild the dataset.")
        if metadata.get("generation") != args.raw_config.get("generation"):
            raise ValueError("QA generation settings differ from the selected configuration; rebuild the dataset.")
        shape_sets = {}
        for key in ("train_shapes", "heldout_shapes", "extrapolation_shapes"):
            shapes = parse_shapes(metadata[key])
            if shapes != parse_shapes(args.raw_config["data"][key]):
                raise ValueError(f"QA/config shape mismatch: {key}.")
            shape_sets[key] = [list(shape) for shape in shapes]
        identity.update(format="raw_hdf5_variable_grid_v1", input_shape=[1, -1, -1],
                        shape_mode="variable", profile=args.experiment_profile, **shape_sets)
        identity.pop("patch_size")
    contract = {
        **identity,
        "identity_sha256": _json_sha256(identity),
        "hdf5_path": canonical_path(hdf5_path),
        "hdf5_bytes": int(hdf5_path.stat().st_size),
        # This is transparent QA-generation lineage only. The Stage-1 file is
        # never opened and no learned Stage-1 tensor enters Direct-Cross.
        "qa_generation_alignment_sha256": str(
            metadata.get("alignment_checkpoint_sha256", "")
        ),
    }
    resolution = {
        "input_source": "raw_hdf5",
        "configured_path": canonical_path(hdf5_path),
        "stage1_checkpoint_opened": False,
        "learned_latent_cache_opened": False,
        "identity_sha256": contract["identity_sha256"],
    }
    return contract, resolution


def load_metadata_and_contract(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata_path = Path(args.qa_dir) / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Matched QA metadata is missing: {metadata_path}.")
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, Mapping):
        raise ValueError(f"Expected a JSON object in {metadata_path}.")
    if str(metadata.get("split_mode", "")) != "sample":
        raise ValueError("Formal cross-attention training requires sample-disjoint QA splits.")
    if int(metadata.get("natural_language_coordinate_origin", -1)) != 1:
        raise ValueError("Natural-language coordinates must use the one-based prompt contract.")
    stage2b = metadata.get("stage2b")
    if not isinstance(stage2b, Mapping):
        raise ValueError("The configured QA directory is not a matched-QA dataset.")
    group_size = int(stage2b.get("batch_group_size", 0))
    if group_size <= 0 or group_size != int(args.questions_per_state_group):
        raise ValueError(
            "training.questions_per_state_group must match metadata.stage2b.batch_group_size: "
            f"configured={args.questions_per_state_group}, metadata={group_size}."
        )
    resolved_metadata = dict(metadata)
    if args.input_source == "raw_hdf5":
        input_contract, input_resolution = validate_raw_hdf5_qa_contract(metadata, args)
        resolved_metadata["runtime_input_resolution"] = input_resolution
        return resolved_metadata, input_contract
    latent_contract, checkpoint_resolution = validate_relocated_qa_latent_contract(
        metadata,
        args.qa_alignment_checkpoint,
    )
    resolved_metadata["runtime_alignment_checkpoint_resolution"] = checkpoint_resolution
    resolved_metadata["runtime_input_resolution"] = {
        "input_source": "precomputed_latent",
        "stage1_checkpoint_opened": True,
        "learned_latent_cache_opened": True,
    }
    return resolved_metadata, latent_contract


def normalize_raw_patch_for_qa(
    raw_patch: torch.Tensor,
    record: Mapping[str, Any],
    normalization: Mapping[str, Any],
    *,
    normalized_dtype: str = "float32",
) -> torch.Tensor:
    """Reproduce the QA builder's float32 per-patch z-score and audit it."""

    if raw_patch.ndim != 3 or int(raw_patch.shape[0]) != 1:
        raise ValueError(
            f"Direct-Cross expects one raw field channel [1,H,W], got {tuple(raw_patch.shape)}."
        )
    required = canonical_normalization(
        {
            "mode": "zscore",
            "scope": "channel",
            "stats_path": None,
            "clip_min": None,
            "clip_max": None,
        }
    )
    if canonical_normalization(normalization) != required:
        raise ValueError("Direct-Cross raw patch normalization must be unclipped channel z-score.")
    patch = raw_patch.float()
    if not bool(torch.isfinite(patch).all()):
        raise FloatingPointError("Direct-Cross raw HDF5 patch contains NaN or infinity.")
    mean = patch.mean()
    std = patch.std(unbiased=False)
    scale = std + 1.0e-6
    observed_stats = {
        "mean": float(mean.item()),
        "std": float(std.item()),
        "scale": float(scale.item()),
    }
    expected_stats = latent_qa_stats_from_record(record)
    for name, observed in observed_stats.items():
        expected = float(expected_stats[name])
        tolerance = max(1.0e-8, 1.0e-6 * max(1.0, abs(expected)))
        if abs(observed - expected) > tolerance:
            raise ValueError(
                "Direct-Cross raw HDF5 patch does not match matched-QA provenance: "
                f"state={record.get('state_ref')!r}, {name} observed={observed}, expected={expected}."
            )
    normalized = (patch - mean) / scale
    if str(normalized_dtype) == "float16":
        normalized = normalized.to(dtype=torch.float16).float()
    elif str(normalized_dtype) != "float32":
        raise ValueError(f"Unsupported normalized raw dtype: {normalized_dtype!r}.")
    return normalized.contiguous()


class RawHDF5TensorReadoutQADataset(TensorReadoutQADataset):
    """Matched QA whose model input is replayed from raw PDEBench, never a Stage-1 cache."""

    def __init__(
        self,
        jsonl_path: str | Path,
        *,
        hdf5_path: str | Path,
        patch_size: int,
        normalization: Mapping[str, Any],
        normalized_dtype: str,
        max_records: int | None,
        subset_mode: str,
        subset_seed: int,
        shuffle_seed: int,
        input_cache_size: int,
        dynamic_grid: bool = False,
        allowed_shapes: Sequence | None = None,
    ) -> None:
        self.dynamic_grid = bool(dynamic_grid)
        self.allowed_shapes = set(tuple(shape) for shape in (allowed_shapes or []))
        super().__init__(
            jsonl_path=jsonl_path,
            latent_dir=Path(hdf5_path).expanduser().parent,
            max_records=max_records,
            subset_mode=subset_mode,
            subset_seed=subset_seed,
            prefer_record_latent_ref=False,
            shuffle_seed=shuffle_seed,
            latent_cache_size=input_cache_size,
            latent_contract=None,
            latent_channel_policy="all",
        )
        self.hdf5_path = Path(hdf5_path).expanduser()
        self.patch_size = int(patch_size)
        self.normalization = canonical_normalization(normalization)
        self.normalized_dtype = str(normalized_dtype)
        self._hdf5_handle: Any | None = None
        self._hdf5_pid: int | None = None

    def _build_random_different_indices(self, seed: int) -> list[int]:
        if self.dynamic_grid:
            return shape_matched_shuffle_indices(self.records, seed)
        return super()._build_random_different_indices(seed)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
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

    def _open_hdf5(self) -> Any:
        process_id = os.getpid()
        handle = self._hdf5_handle
        if handle is not None and self._hdf5_pid == process_id and bool(handle.id.valid):
            return handle
        self.close()
        h5py_module = require_h5py()
        self._hdf5_handle = h5py_module.File(self.hdf5_path, "r")
        self._hdf5_pid = process_id
        return self._hdf5_handle

    def _read_raw_patch(self, record: Mapping[str, Any]) -> torch.Tensor:
        identity = latent_identity_from_record(record)
        row, col = (int(value) for value in identity["top_left"])
        sample_index = int(identity["sample_index"])
        time_index = int(identity["time_index"])
        field = str(identity["field"])
        height, width = record_shape(record)
        if (self.dynamic_grid and self.allowed_shapes and (height, width) not in self.allowed_shapes) or (
            not self.dynamic_grid and (height, width) != (self.patch_size, self.patch_size)
        ):
            raise ValueError(
                f"Matched-QA state {identity['patch_id']!r} has an invalid grid_shape."
            )
        handle = self._open_hdf5()
        if field not in handle:
            raise KeyError(f"PDEBench HDF5 field {field!r} is missing from {self.hdf5_path}.")
        dataset = handle[field]
        if len(dataset.shape) != 4:
            raise ValueError(f"Expected PDEBench field [sample,time,height,width], got {dataset.shape}.")
        if not 0 <= sample_index < int(dataset.shape[0]) or not 0 <= time_index < int(
            dataset.shape[1]
        ):
            raise IndexError(
                f"Matched-QA state {identity['patch_id']!r} is outside the PDEBench sample/time axes."
            )
        if (
            row < 0
            or col < 0
            or row + height > int(dataset.shape[2])
            or col + width > int(dataset.shape[3])
        ):
            raise IndexError(
                f"Matched-QA state {identity['patch_id']!r} is outside the PDEBench spatial axes."
            )
        array = dataset[
            sample_index,
            time_index,
            row : row + height,
            col : col + width,
        ]
        return torch.as_tensor(array, dtype=torch.float32).clone().unsqueeze(0)

    def load_latent_for_record(self, record: Mapping[str, Any]) -> torch.Tensor:
        state_ref = str(record.get("state_ref") or record.get("patch_id") or "")
        if not state_ref:
            raise ValueError("Direct-Cross record has no state_ref/patch_id.")
        identity = latent_identity_from_record(record)
        if self.dynamic_grid:
            identity = {**identity, "grid_shape": list(record_shape(record)),
                        "normalized_sha256": record["metadata"].get("normalized_sha256")}
        previous_identity = self._latent_identity_cache.get(state_ref)
        if previous_identity is not None and previous_identity != identity:
            raise ValueError(
                f"Direct-Cross records reuse state_ref={state_ref!r} for different raw patches."
            )
        self._latent_identity_cache[state_ref] = identity
        qa_stats = latent_qa_stats_from_record(record)
        previous_stats = self._latent_qa_stats_cache.get(state_ref)
        if previous_stats is not None and previous_stats != qa_stats:
            raise ValueError(
                f"Direct-Cross records reuse state_ref={state_ref!r} with different QA statistics."
            )
        self._latent_qa_stats_cache[state_ref] = qa_stats
        cached = self._latent_cache.get(state_ref)
        if cached is not None:
            self._latent_cache.move_to_end(state_ref)
            return cached
        normalized = normalize_raw_patch_for_qa(
            self._read_raw_patch(record),
            record,
            self.normalization,
            normalized_dtype=self.normalized_dtype,
        )
        if self.dynamic_grid:
            digest = hashlib.sha256(normalized.numpy().tobytes()).hexdigest()
            if digest != record["metadata"].get("normalized_sha256"):
                raise ValueError(f"Raw field values differ from QA hash for {state_ref}.")
        cache_capacity = self.effective_latent_cache_size()
        if cache_capacity > 0:
            self._latent_cache[state_ref] = normalized
            self._latent_cache.move_to_end(state_ref)
            while len(self._latent_cache) > cache_capacity:
                self._latent_cache.popitem(last=False)
        return normalized

    def validate_latent_file_for_record(self, record: Mapping[str, Any]) -> dict[str, Any]:
        value = self.load_latent_for_record(record)
        return {
            "path": canonical_path(self.hdf5_path),
            "shape": [int(item) for item in value.shape],
            "dtype": str(value.dtype).replace("torch.", ""),
            "input_source": "raw_hdf5",
        }


def audit_raw_hdf5_input_content(
    datasets: Mapping[str, TensorReadoutQADataset],
) -> dict[str, Any]:
    """Hash every distinct normalized QA patch that can enter this run."""

    split_audits: dict[str, Any] = {}
    for split, generic_dataset in datasets.items():
        if not isinstance(generic_dataset, RawHDF5TensorReadoutQADataset):
            raise TypeError("Raw input content audit received a latent-cache dataset.")
        digest = hashlib.sha256()
        seen_states: dict[str, dict[str, Any]] = {}
        for record in generic_dataset.records:
            state_ref = str(record.get("state_ref") or record.get("patch_id") or "")
            identity = latent_identity_from_record(record)
            # Invoke the dataset contract for every QA row so repeated states
            # must also agree on their recorded normalization statistics.
            normalized = (
                generic_dataset.load_latent_for_record(record)
                .detach()
                .cpu()
                .contiguous()
            )
            previous = seen_states.get(state_ref)
            if previous is not None:
                if previous != identity:
                    raise ValueError(
                        f"Raw input audit found conflicting identities for state {state_ref!r}."
                    )
                continue
            seen_states[state_ref] = identity
            digest.update(
                json.dumps(
                    identity,
                    ensure_ascii=True,
                    sort_keys=True,
                    allow_nan=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            digest.update(
                f"\0{normalized.dtype}\0{tuple(normalized.shape)}\0".encode("utf-8")
            )
            digest.update(normalized.view(torch.uint8).numpy().tobytes())
        split_audits[str(split)] = {
            "states": len(seen_states),
            "normalized_patch_sha256": digest.hexdigest(),
        }
        generic_dataset._latent_cache.clear()
        generic_dataset.close()
    content_identity = {
        split: dict(values) for split, values in sorted(split_audits.items())
    }
    return {
        "format": "normalized_qa_patch_content_v1",
        "splits": split_audits,
        "sha256": _json_sha256(content_identity),
    }


def audit_general_qa_datasets(
    datasets: Mapping[str, TensorReadoutQADataset],
    require_disjoint_splits: bool,
) -> dict[str, Any]:
    """Audit records and matched batching without reading routing/query_spec fields."""

    summary: dict[str, Any] = {"splits": {}}
    sample_sets: dict[str, set[int]] = {}
    state_sets: dict[str, set[str]] = {}
    all_qa_ids: dict[str, str] = {}
    for split, dataset in datasets.items():
        record_contract_digest = hashlib.sha256()
        task_counts: dict[str, int] = defaultdict(int)
        field_counts: dict[str, int] = defaultdict(int)
        batch_groups: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
        samples: set[int] = set()
        states: set[str] = set()
        for record in dataset.records:
            qa_id = str(record.get("qa_id", ""))
            if not qa_id:
                raise ValueError(f"{split} contains a record without qa_id.")
            # Records originate in JSONL, so hashing the complete canonical
            # object binds question text, ordered choices, label, sample/state,
            # matched-group metadata, and every other effective input field.
            record_contract_digest.update(
                json.dumps(
                    record,
                    ensure_ascii=True,
                    sort_keys=True,
                    allow_nan=False,
                    separators=(",", ":"),
                ).encode("utf-8")
                + b"\n"
            )
            previous_split = all_qa_ids.setdefault(qa_id, split)
            if previous_split != split:
                raise ValueError(f"qa_id {qa_id!r} appears in both {previous_split} and {split}.")
            query = str(record.get("query") or record.get("question") or "").strip()
            choices = record.get("choices")
            answer = str(record.get("answer", ""))
            if not query:
                raise ValueError(f"Record {qa_id} has no natural-language query.")
            if not isinstance(choices, Sequence) or isinstance(choices, str) or not choices:
                raise ValueError(f"Record {qa_id} has no choices.")
            if answer not in [str(value) for value in choices]:
                raise ValueError(f"Record {qa_id} answer={answer!r} is absent from its choices.")
            task_counts[str(record.get("task_type", "unknown"))] += 1
            field_counts[str(record.get("field") or record.get("metadata", {}).get("field") or "unknown")] += 1
            samples.add(int(record.get("sample_index", -1)))
            state_ref = str(record.get("state_ref", ""))
            if not state_ref:
                raise ValueError(f"Record {qa_id} has no state_ref.")
            states.add(state_ref)
            matched = record.get("matched_group")
            if isinstance(matched, Mapping):
                group_id = str(matched.get("batch_group_id", ""))
                group_size = int(matched.get("batch_group_size", 0))
                member = int(matched.get("batch_member_index", -1))
                if not group_id or group_size <= 0 or not 0 <= member < group_size:
                    raise ValueError(f"Record {qa_id} has an invalid matched batch group.")
                batch_groups[group_id].append((member, group_size, state_ref))
        for group_id, members in batch_groups.items():
            declared = {size for _member, size, _state in members}
            observed = sorted(member for member, _size, _state in members)
            states_in_group = {state for _member, _size, state in members}
            if len(declared) != 1 or observed != list(range(next(iter(declared)))):
                raise ValueError(f"Matched batch group {group_id} is incomplete or duplicated.")
            if len(states_in_group) != 1:
                raise ValueError(f"Matched batch group {group_id} mixes tensor states.")
        sample_sets[split] = samples
        state_sets[split] = states
        summary["splits"][split] = {
            "records": len(dataset),
            "states": len(states),
            "samples": len(samples),
            "batch_groups": len(batch_groups),
            "task_counts": dict(sorted(task_counts.items())),
            "field_counts": dict(sorted(field_counts.items())),
            "record_contract_sha256": record_contract_digest.hexdigest(),
        }
    overlaps: dict[str, Any] = {}
    split_names = list(datasets)
    for first_index, first in enumerate(split_names):
        for second in split_names[first_index + 1 :]:
            sample_overlap = sorted(sample_sets[first] & sample_sets[second])
            state_overlap = sorted(state_sets[first] & state_sets[second])
            overlaps[f"{first}__{second}"] = {
                "sample_overlap_count": len(sample_overlap),
                "state_overlap_count": len(state_overlap),
                "sample_overlap_preview": sample_overlap[:8],
                "state_overlap_preview": state_overlap[:8],
            }
            if require_disjoint_splits and (sample_overlap or state_overlap):
                raise ValueError(
                    f"Formal QA splits {first}/{second} overlap: "
                    f"samples={len(sample_overlap)}, states={len(state_overlap)}."
                )
    summary["overlaps"] = overlaps
    summary["model_forbidden_fields"] = [
        "matched_group.query_spec",
        "grounding_target",
        "structured task ID",
        "parsed coordinate",
        "task-to-arity mapping",
    ]
    return summary


def build_datasets(
    args: argparse.Namespace,
    input_contract: Mapping[str, Any],
) -> tuple[
    TensorReadoutQADataset,
    TensorReadoutQADataset,
    TensorReadoutQADataset,
    TensorReadoutQADataset,
]:
    def make_dataset(
        split: str,
        *,
        max_records: int | None,
        subset_mode: str,
        subset_seed: int,
    ) -> TensorReadoutQADataset:
        path = qa_path(args.qa_dir, split)
        if args.input_source == "raw_hdf5":
            normalization = input_contract.get("normalization")
            if not isinstance(normalization, Mapping):
                raise ValueError("Raw HDF5 input contract has no normalization mapping.")
            return RawHDF5TensorReadoutQADataset(
                path,
                hdf5_path=args.hdf5_path,
                patch_size=int(args.patch_size),
                normalization=normalization,
                normalized_dtype=str(args.raw_normalized_dtype),
                max_records=max_records,
                subset_mode=subset_mode,
                subset_seed=subset_seed,
                shuffle_seed=int(args.shuffle_seed),
                input_cache_size=int(args.latent_cache_size),
                dynamic_grid=getattr(args, "shape_mode", "fixed") == "variable",
                allowed_shapes=(input_contract.get("train_shapes", []) + input_contract.get("heldout_shapes", []) + input_contract.get("extrapolation_shapes", [])),
            )
        return TensorReadoutQADataset(
            path,
            latent_dir=args.latent_dir,
            max_records=max_records,
            subset_mode=subset_mode,
            subset_seed=subset_seed,
            prefer_record_latent_ref=bool(args.prefer_record_latent_ref),
            shuffle_seed=int(args.shuffle_seed),
            latent_cache_size=int(args.latent_cache_size),
            latent_contract=input_contract,
            latent_channel_policy=str(args.latent_channel_policy),
        )

    train_dataset = make_dataset(
        args.train_split,
        max_records=args.max_train_records,
        subset_mode=str(args.record_subset_mode),
        subset_seed=int(args.shuffle_seed),
    )
    val_dataset = make_dataset(
        args.val_split,
        max_records=args.max_val_records,
        subset_mode=str(args.record_subset_mode),
        subset_seed=int(args.shuffle_seed) + 1,
    )
    test_dataset = make_dataset(
        args.test_split,
        max_records=args.max_test_records,
        subset_mode=str(args.record_subset_mode),
        subset_seed=int(args.shuffle_seed) + 2,
    )
    screening_dataset = make_dataset(
        args.val_split,
        max_records=None if getattr(args, "shape_mode", "fixed") == "variable" else min(int(args.screening_records), len(val_dataset.records)),
        subset_mode="hash_state",
        subset_seed=int(args.shuffle_seed) + 10_001,
    )
    if getattr(args, "shape_mode", "fixed") == "variable":
        metadata = json.loads((Path(args.qa_dir) / "metadata.json").read_text(encoding="utf-8"))
        shape_partitions = {tuple(shape): partition for key, partition in (
            ("train_shapes", "seen"), ("heldout_shapes", "heldout"), ("extrapolation_shapes", "extrapolation")
        ) for shape in metadata[key]}
        for split, dataset in ((args.train_split, train_dataset), (args.val_split, val_dataset), (args.test_split, test_dataset)):
            allowed_samples = set(metadata["trajectory_splits"][split])
            expected_shapes = set(args.train_shapes) if split == args.train_split else set(shape_partitions)
            if {record_shape(record) for record in dataset.records} != expected_shapes:
                raise ValueError(f"Missing or unexpected shapes in {split}.")
            for record in dataset.records:
                if int(record["sample_index"]) not in allowed_samples:
                    raise ValueError(f"Record lies outside the declared {split} trajectory partition.")
                if record["metadata"].get("shape_partition") != shape_partitions[record_shape(record)]:
                    raise ValueError("Record shape_partition differs from metadata.")
        screening_dataset.records = balanced_screen_records(
            val_dataset.records, args.train_shapes, int(args.screening_records), int(args.shuffle_seed) + 10_001
        )
        screening_dataset._random_different_indices = shape_matched_shuffle_indices(screening_dataset.records, int(args.shuffle_seed))
    return train_dataset, val_dataset, test_dataset, screening_dataset


def strict_encode_example(
    record: Mapping[str, Any],
    tokenizer,
    args: argparse.Namespace,
) -> tuple[list[int], list[int]]:
    prompt_ids = list(
        tokenizer(
            build_prompt(record, prompt_template=str(args.prompt_template)),
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]
    )
    if len(prompt_ids) > int(args.max_prompt_tokens):
        raise ValueError(
            f"Prompt {record.get('qa_id')} has {len(prompt_ids)} tokens, exceeding "
            f"max_prompt_tokens={args.max_prompt_tokens}; truncation is forbidden."
        )
    answer_ids = list(
        tokenizer(
            " " + str(record["answer"]),
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
    )
    if bool(args.append_eos) and tokenizer.eos_token_id is not None:
        answer_ids.append(int(tokenizer.eos_token_id))
    if not answer_ids:
        raise ValueError(f"Answer for {record.get('qa_id')} tokenized to an empty sequence.")
    if len(answer_ids) > int(args.max_target_tokens):
        raise ValueError(
            f"Answer for {record.get('qa_id')} has {len(answer_ids)} tokens, exceeding "
            f"max_target_tokens={args.max_target_tokens}; truncation is forbidden."
        )
    return [int(value) for value in prompt_ids], [int(value) for value in answer_ids]


def build_strict_training_tensors(
    records: Sequence[Mapping[str, Any]],
    tokenizer,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    encoded = [strict_encode_example(record, tokenizer, args) for record in records]
    max_length = max(len(prompt) + len(answer) for prompt, answer in encoded)
    input_ids = torch.full(
        (len(encoded), max_length), int(tokenizer.pad_token_id), dtype=torch.long
    )
    attention_mask = torch.zeros((len(encoded), max_length), dtype=torch.long)
    labels = torch.full((len(encoded), max_length), IGNORE_INDEX, dtype=torch.long)
    for row, (prompt, answer) in enumerate(encoded):
        values = prompt + answer
        input_ids[row, : len(values)] = torch.tensor(values, dtype=torch.long)
        attention_mask[row, : len(values)] = 1
        labels[row, len(prompt) : len(values)] = torch.tensor(answer, dtype=torch.long)
    return input_ids, attention_mask, labels


def build_strict_prompt_tensors(
    records: Sequence[Mapping[str, Any]],
    tokenizer,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows: list[list[int]] = []
    for record in records:
        ids = list(
            tokenizer(
                build_prompt(record, prompt_template=str(args.prompt_template)),
                add_special_tokens=True,
                truncation=False,
            )["input_ids"]
        )
        if len(ids) > int(args.max_prompt_tokens):
            raise ValueError(
                f"Prompt {record.get('qa_id')} exceeds max_prompt_tokens; truncation is forbidden."
            )
        rows.append([int(value) for value in ids])
    max_length = max(len(row) for row in rows)
    input_ids = torch.full(
        (len(rows), max_length), int(tokenizer.pad_token_id), dtype=torch.long
    )
    attention_mask = torch.zeros((len(rows), max_length), dtype=torch.long)
    for row_index, row in enumerate(rows):
        input_ids[row_index, : len(row)] = torch.tensor(row, dtype=torch.long)
        attention_mask[row_index, : len(row)] = 1
    return input_ids, attention_mask


def decoder_backbone(llm: nn.Module) -> nn.Module:
    decoder = getattr(llm, "model", None)
    if not isinstance(decoder, nn.Module):
        raise TypeError("The causal LLM does not expose a decoder backbone as model.")
    return decoder


def restricted_choice_logits(
    hidden: torch.Tensor,
    candidate_ids: Sequence[Sequence[int]],
    output_embeddings: nn.Module,
) -> list[torch.Tensor]:
    weight = getattr(output_embeddings, "weight", None)
    if not isinstance(weight, torch.Tensor):
        full_logits = output_embeddings(hidden)
        return [
            full_logits[row, torch.tensor(ids, device=hidden.device)]
            for row, ids in enumerate(candidate_ids)
        ]
    bias = getattr(output_embeddings, "bias", None)
    results: list[torch.Tensor] = []
    for row, ids in enumerate(candidate_ids):
        index = torch.tensor(ids, dtype=torch.long, device=hidden.device)
        selected_weight = weight[index]
        logits = F.linear(hidden[row : row + 1], selected_weight, None).squeeze(0)
        if isinstance(bias, torch.Tensor):
            logits = logits + bias[index]
        results.append(logits)
    return results


def choice_loss_from_hidden(
    hidden: torch.Tensor,
    records: Sequence[Mapping[str, Any]],
    tokenizer,
    output_embeddings: nn.Module,
) -> tuple[torch.Tensor, list[torch.Tensor], float]:
    specification = single_token_choice_ids(records, tokenizer)
    if specification is None:
        raise ValueError("Dense cross-attention training requires unique single-token choice labels.")
    candidate_ids, target_indices = specification
    row_logits = restricted_choice_logits(hidden, candidate_ids, output_embeddings)
    losses: list[torch.Tensor] = []
    log_probs: list[torch.Tensor] = []
    correct = 0
    for logits, target_index in zip(row_logits, target_indices):
        target = torch.tensor([int(target_index)], device=hidden.device)
        losses.append(F.cross_entropy(logits.float().unsqueeze(0), target))
        log_probs.append(F.log_softmax(logits.float(), dim=-1))
        correct += int(int(torch.argmax(logits.detach()).item()) == int(target_index))
    return torch.stack(losses).mean(), log_probs, correct / max(1, len(losses))


def full_answer_ce(
    sequence_hidden: torch.Tensor,
    labels: torch.Tensor,
    output_embeddings: nn.Module,
) -> torch.Tensor:
    targets = labels[:, 1:]
    active = targets.ne(IGNORE_INDEX)
    if not bool(active.any()):
        return sequence_hidden.new_zeros((), dtype=torch.float32)
    prediction_hidden = sequence_hidden[:, :-1][active]
    target_ids = targets[active]
    logits = output_embeddings(prediction_hidden).float()
    return F.cross_entropy(logits, target_ids)


def forward_training_batch(
    llm: nn.Module,
    sidecar: DenseCrossAttentionSidecar,
    tokenizer,
    records: Sequence[Mapping[str, Any]],
    latent_map: torch.Tensor,
    device: torch.device,
    model_dtype: torch.dtype,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    input_ids, attention_mask, labels = build_strict_training_tensors(records, tokenizer, args)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)
    latent_map = latent_map.to(device, non_blocking=True)
    output_embeddings = llm.get_output_embeddings()
    if output_embeddings is None:
        raise ValueError("The frozen Qwen model has no output embedding head.")
    memory_state: DenseMemoryState | None = None
    try:
        with autocast_context(device, model_dtype):
            memory_state = sidecar.bind(latent_map, mode="correct")
            outputs = decoder_backbone(llm)(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            sequence_hidden = outputs.last_hidden_state
            first_targets = labels.ne(IGNORE_INDEX).long().argmax(dim=1)
            if bool(first_targets.eq(0).any()):
                raise RuntimeError("Every answer must begin after at least one prompt token.")
            rows = torch.arange(input_ids.shape[0], device=device)
            choice_hidden = sequence_hidden[rows, first_targets - 1]
            choice_ce, candidate_log_probs, accuracy = choice_loss_from_hidden(
                choice_hidden,
                records,
                tokenizer,
                output_embeddings,
            )
            answer_ce = (
                full_answer_ce(sequence_hidden, labels, output_embeddings)
                if float(args.full_answer_ce_weight) > 0.0
                else choice_ce.new_zeros(())
            )
            matched_loss, matched_metrics = matched_coordinate_group_loss(
                records,
                candidate_log_probs,
                margin=float(args.matched_group_margin),
            )
            reconstruction = (
                memory_state.reconstruction_loss
                if memory_state is not None
                else choice_ce.new_zeros(())
            )
            total = (
                float(args.choice_ce_weight) * choice_ce
                + float(args.full_answer_ce_weight) * answer_ce
                + float(args.matched_group_weight) * matched_loss
                + float(args.value_reconstruction_weight) * reconstruction
            )
        metrics = {
            "loss": float(total.detach().float().cpu().item()),
            "choice_ce": float(choice_ce.detach().float().cpu().item()),
            "answer_ce": float(answer_ce.detach().float().cpu().item()),
            "matched_group": float(matched_loss.detach().float().cpu().item()),
            "value_reconstruction": float(reconstruction.detach().float().cpu().item()),
            "accuracy": float(accuracy),
            **{
                f"matched_{key}": float(value)
                for key, value in matched_metrics.items()
                if isinstance(value, (int, float))
            },
        }
        if not bool(torch.isfinite(total)):
            raise FloatingPointError(f"Non-finite cross-attention loss: {metrics}.")
        return total, metrics
    except BaseException:
        sidecar.clear()
        raise


def load_spatial_initializer(
    checkpoint_path: str | Path,
    latent_shape: Sequence[int],
    llm_hidden_size: int,
    latent_contract: Mapping[str, Any],
    expected_model_name: str,
    expected_latent_channel_policy: str,
) -> tuple[TensorPatchAlignmentAdapter, dict[str, Any]]:
    path = Path(checkpoint_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Dense-memory initializer is missing: {path}.")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Unsupported dense-memory initializer payload: {path}.")
    validate_adapter_checkpoint_payload(
        checkpoint,
        expected_latent_shape=latent_shape,
        expected_llm_hidden_size=int(llm_hidden_size),
        expected_architecture="alignment_adapter",
        expected_latent_contract=latent_contract,
        expected_latent_channel_policy=str(expected_latent_channel_policy),
    )
    checkpoint_args = checkpoint.get("args")
    if not isinstance(checkpoint_args, Mapping):
        raise ValueError("Dense-memory initializer does not record its argument contract.")
    checkpoint_model = str(checkpoint_args.get("model_name_or_path", "")).replace("\\", "/").rstrip("/")
    current_model = str(expected_model_name).replace("\\", "/").rstrip("/")
    if not checkpoint_model or checkpoint_model.rsplit("/", 1)[-1].casefold() != current_model.rsplit("/", 1)[-1].casefold():
        raise ValueError(
            "Dense-memory initializer and current frozen Qwen identities differ: "
            f"initializer={checkpoint_model!r}, current={current_model!r}."
        )
    source_latent_channel_policy = str(checkpoint_args.get("latent_channel_policy", "all"))
    if source_latent_channel_policy != str(expected_latent_channel_policy):
        raise ValueError(
            "Dense-memory initializer and cross-attention run use different latent channel policies: "
            f"initializer={source_latent_channel_policy!r}, current={expected_latent_channel_policy!r}."
        )
    adapter = adapter_from_checkpoint(
        checkpoint,
        latent_shape=latent_shape,
        llm_hidden_size=int(llm_hidden_size),
    )
    if not isinstance(adapter, TensorPatchAlignmentAdapter):
        raise TypeError("The configured initializer did not rebuild a direct spatial alignment adapter.")
    if adapter.adapter_type != "spatial_transformer":
        raise ValueError("Dense memory requires the direct spatial_transformer Stage-2 checkpoint.")
    provenance = {
        "path": canonical_path(path),
        "sha256": sha256_file(path),
        "checkpoint_type": str(checkpoint.get("checkpoint_type", "")),
        "checkpoint_version": int(checkpoint.get("checkpoint_version", 0)),
        "source_architecture": str(checkpoint_args.get("adapter_architecture", "")),
        "source_adapter_type": str(
            checkpoint_args.get("global_adapter_type", checkpoint_args.get("adapter_type", ""))
        ),
        "source_model": checkpoint_model,
        "latent_channel_policy": source_latent_channel_policy,
    }
    return adapter, provenance


def _module_state_sha256(modules: Mapping[str, nn.Module]) -> str:
    digest = hashlib.sha256()
    for module_name, module in sorted(modules.items()):
        for tensor_name, tensor in sorted(module.state_dict().items()):
            value = tensor.detach().cpu().contiguous()
            digest.update(f"{module_name}.{tensor_name}\0{value.dtype}\0{tuple(value.shape)}\0".encode("utf-8"))
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def build_scratch_memory_components(
    args: argparse.Namespace,
    input_shape: Sequence[int],
) -> tuple[DirectFieldEncoder, FullGridSpatialBackbone, tuple[int, int, int], dict[str, Any]]:
    """Construct every field-side component without opening an upstream checkpoint."""

    normalized_input_shape = tuple(int(value) for value in input_shape)
    expected_input_shape = (1, int(args.patch_size), int(args.patch_size))
    dynamic = getattr(args, "shape_mode", "fixed") == "variable"
    if (not dynamic and normalized_input_shape != expected_input_shape) or (dynamic and (len(normalized_input_shape) != 3 or normalized_input_shape[0] != 1)):
        raise ValueError(
            f"Direct-Cross raw input shape is {normalized_input_shape}, expected {expected_input_shape}."
        )
    encoder_config = build_patch_encoder_config(
        patch_encoder_cfg=args.field_encoder_config,
        field_keys=["__single_field__"],
        patch_size=int(args.patch_size),
    )
    model_config = encoder_config.get("model")
    if not isinstance(model_config, Mapping):
        raise ValueError("Direct-Cross field encoder did not resolve a model mapping.")
    if str(model_config.get("name", "")) != "conv_token_autoencoder_2d":
        raise ValueError("Direct-Cross currently supports conv_token_autoencoder_2d only.")
    if not bool(model_config.get("preserve_input_channels", False)):
        raise ValueError("Direct-Cross field encoder requires preserve_input_channels=true.")
    compressor = build_model(copy.deepcopy(encoder_config))
    field_encoder = DirectFieldEncoder(compressor, dynamic_grid=dynamic)
    if dynamic and (model_config.get("channel_multipliers") or field_encoder.latent_grid != field_encoder.input_size):
        raise ValueError("Variable field encoder must preserve H,W without downsampling or pooling.")
    if field_encoder.input_size != expected_input_shape[1:]:
        raise ValueError(
            "Direct-Cross field_encoder.model.input_size must match data.patch_size: "
            f"encoder={field_encoder.input_size}, data={expected_input_shape[1:]}."
        )
    memory_shape = (
        int(field_encoder.latent_dim),
        int(field_encoder.latent_grid[0]),
        int(field_encoder.latent_grid[1]),
    )
    spatial = args.spatial_adapter_config
    spatial_type = str(spatial.get("type", "full_grid_transformer"))
    if spatial_type != "full_grid_transformer":
        raise ValueError("Direct-Cross spatial_adapter.type must be full_grid_transformer.")
    spatial_backbone = FullGridSpatialBackbone(
        latent_channels=memory_shape[0],
        latent_grid=memory_shape[1:],
        adapter_dim=int(spatial.get("adapter_dim", 512)),
        adapter_layers=int(spatial.get("adapter_layers", 2)),
        adapter_heads=int(spatial.get("adapter_heads", 8)),
        dropout=float(spatial.get("dropout", 0.0)),
        dynamic_grid=dynamic,
    )
    state_sha256 = _module_state_sha256(
        {"field_encoder": field_encoder, "spatial_backbone": spatial_backbone}
    )
    provenance = {
        "kind": "scratch",
        "sha256": state_sha256,
        "seed": int(args.seed),
        "stage1_checkpoint_opened": False,
        "direct_qa_checkpoint_opened": False,
        "prefix_output_head": "absent",
        "field_encoder_config": copy.deepcopy(encoder_config),
        "spatial_adapter_config": {
            "type": spatial_type,
            "adapter_dim": int(spatial_backbone.adapter_dim),
            "adapter_layers": len(spatial_backbone.blocks),
            "adapter_heads": int(spatial.get("adapter_heads", 8)),
            "dropout": float(spatial.get("dropout", 0.0)),
        },
    }
    return field_encoder, spatial_backbone, memory_shape, provenance


def build_sidecar(
    llm: nn.Module,
    spatial_initializer: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    field_encoder: DirectFieldEncoder | None = None,
) -> tuple[DenseCrossAttentionSidecar, dict[str, Any]]:
    llm_hidden_size = int(llm.get_input_embeddings().embedding_dim)
    memory = DenseTensorMemory(
        spatial_backbone=spatial_initializer,
        fourier_bands=int(args.value_fourier_bands),
        value_hidden_dim=int(args.value_hidden_dim),
        freeze_spatial_backbone=bool(args.freeze_spatial_backbone),
        field_encoder=field_encoder,
    )
    bridges = [
        GatedTensorCrossAttention(
            llm_dim=llm_hidden_size,
            memory_dim=memory.memory_dim,
            bridge_dim=int(args.bridge_dim),
            heads=int(args.bridge_heads),
            dropout=float(args.bridge_dropout),
            gate_init=float(args.gate_init),
        )
        for _layer in args.cross_attention_layers
    ]
    sidecar = DenseCrossAttentionSidecar(
        memory=memory,
        bridges=bridges,
        layers_1based=args.cross_attention_layers,
    ).to(device)
    install_report = sidecar.install(llm)
    sidecar_parameter_ids = {id(parameter) for parameter in sidecar.parameters()}
    for parameter in llm.parameters():
        # The shared bridge objects are registered under wrapped Qwen layers as well as
        # the sidecar. Restore their trainable boundary after freezing the base model.
        if id(parameter) not in sidecar_parameter_ids:
            parameter.requires_grad_(False)
    spatial_parameter_ids = {
        id(parameter) for parameter in sidecar.memory.spatial_backbone.parameters()
    }
    for parameter in sidecar.parameters():
        if id(parameter) not in spatial_parameter_ids:
            parameter.requires_grad_(True)
    if bool(args.freeze_spatial_backbone):
        for parameter in sidecar.memory.spatial_backbone.parameters():
            parameter.requires_grad_(False)
    unexpected_llm_trainable = [
        name
        for name, parameter in llm.named_parameters()
        if id(parameter) not in sidecar_parameter_ids and parameter.requires_grad
    ]
    if unexpected_llm_trainable:
        raise RuntimeError(
            "Frozen-Qwen boundary audit found trainable base parameters: "
            f"{unexpected_llm_trainable[:12]}."
        )
    if bool(args.freeze_spatial_backbone) and any(
        parameter.requires_grad for parameter in sidecar.memory.spatial_backbone.parameters()
    ):
        raise RuntimeError("The configured frozen spatial memory backbone remains trainable.")
    return sidecar, {
        **install_report,
        "llm_hidden_size": llm_hidden_size,
        "memory_dim": int(memory.memory_dim),
        "bridge_dim": int(args.bridge_dim),
        "heads": int(args.bridge_heads),
        "gate_init": float(args.gate_init),
        "value_fourier_bands": int(args.value_fourier_bands),
        "freeze_spatial_backbone": bool(args.freeze_spatial_backbone),
        "field_encoder_trainable": bool(field_encoder is not None),
        "latent_channel_policy": str(args.latent_channel_policy),
    }


def trainable_named_parameters(sidecar: nn.Module) -> list[tuple[str, nn.Parameter]]:
    values = [(name, parameter) for name, parameter in sidecar.named_parameters() if parameter.requires_grad]
    if not values:
        raise RuntimeError("The dense cross-attention sidecar has no trainable parameters.")
    if len({id(parameter) for _name, parameter in values}) != len(values):
        raise RuntimeError("The sidecar exposes duplicate trainable parameter tensors.")
    return values


def synchronize_trainable_sidecar(sidecar: nn.Module) -> None:
    if not distributed_is_initialized():
        return
    for _name, parameter in trainable_named_parameters(sidecar):
        dist.broadcast(parameter.data, src=0)


def build_optimizer(
    sidecar: nn.Module,
    args: argparse.Namespace,
) -> tuple[torch.optim.Optimizer, dict[str, Any]]:
    gates: list[nn.Parameter] = []
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    named = trainable_named_parameters(sidecar)
    for name, parameter in named:
        if name.endswith(".gate"):
            gates.append(parameter)
        elif parameter.ndim < 2 or name.endswith("bias"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    groups: list[dict[str, Any]] = []
    if decay:
        groups.append(
            {
                "params": decay,
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
                "name": "sidecar_decay",
            }
        )
    if no_decay:
        groups.append(
            {
                "params": no_decay,
                "lr": float(args.lr),
                "weight_decay": 0.0,
                "name": "sidecar_no_decay",
            }
        )
    if gates:
        groups.append(
            {
                "params": gates,
                "lr": float(args.gate_lr),
                "weight_decay": 0.0,
                "name": "gates",
            }
        )
    optimizer = torch.optim.AdamW(groups)
    optimizer_ids = [id(parameter) for group in groups for parameter in group["params"]]
    expected_ids = {id(parameter) for _name, parameter in named}
    if len(optimizer_ids) != len(set(optimizer_ids)) or set(optimizer_ids) != expected_ids:
        raise RuntimeError("Optimizer membership does not exactly match the trainable sidecar boundary.")
    return optimizer, {
        "trainable_parameters": sum(parameter.numel() for _name, parameter in named),
        "trainable_tensors": len(named),
        "trainable_field_encoder_parameters": sum(
            parameter.numel()
            for name, parameter in named
            if name.startswith("memory.field_encoder.")
        ),
        "trainable_spatial_parameters": sum(
            parameter.numel()
            for name, parameter in named
            if name.startswith("memory.spatial_backbone.")
        ),
        "trainable_value_parameters": sum(
            parameter.numel()
            for name, parameter in named
            if name.startswith("memory.value_encoder.")
            or name.startswith("memory.value_reconstruction.")
        ),
        "trainable_cross_attention_parameters": sum(
            parameter.numel() for name, parameter in named if name.startswith("bridges.")
        ),
        "gate_parameters": sum(parameter.numel() for parameter in gates),
        "gate_tensors": len(gates),
        "frozen_spatial_parameters": sum(
            parameter.numel() for parameter in sidecar.memory.spatial_backbone.parameters()
            if not parameter.requires_grad
        ),
        "frozen_llm_parameters": None,
    }


def build_sidecar_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str,
    total_updates: int,
    warmup_ratio: float,
    min_lr_ratio: float,
) -> tuple[torch.optim.lr_scheduler.LambdaLR, int]:
    """Warm up high-dimensional projections while opening zero gates immediately."""

    total = max(1, int(total_updates))
    warmup = min(total - 1, int(round(total * float(warmup_ratio))))

    def decay_factor(step: int) -> float:
        decay_updates = max(1, total - warmup)
        progress = min(1.0, max(0.0, (float(step) - warmup) / decay_updates))
        if str(scheduler_name) == "constant":
            return 1.0
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(min_lr_ratio) + (1.0 - float(min_lr_ratio)) * cosine

    def projection_factor(step: int) -> float:
        if warmup > 0 and int(step) < warmup:
            return max(1.0 / warmup, float(step + 1) / warmup)
        return decay_factor(step)

    def gate_factor(step: int) -> float:
        # At raw gate=0 the attention projections receive no answer gradient.
        # The scalar gates therefore skip warmup but share the later decay.
        return 1.0 if int(step) < warmup else decay_factor(step)

    lambdas = [
        gate_factor if str(group.get("name", "")) == "gates" else projection_factor
        for group in optimizer.param_groups
    ]
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambdas), warmup


def trainable_state_dict(sidecar: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in trainable_named_parameters(sidecar)
    }


def load_trainable_state_dict(
    sidecar: nn.Module,
    state: Mapping[str, Any],
) -> dict[str, int]:
    current = dict(trainable_named_parameters(sidecar))
    observed = {str(name) for name in state}
    expected = set(current)
    if observed != expected:
        raise ValueError(
            "Cross-attention checkpoint parameter keys differ from the current sidecar: "
            f"missing={sorted(expected - observed)[:12]}, unexpected={sorted(observed - expected)[:12]}."
        )
    with torch.no_grad():
        for name, parameter in current.items():
            value = state[name]
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(parameter.shape):
                raise ValueError(f"Invalid checkpoint tensor for {name}: expected {tuple(parameter.shape)}.")
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
    return {
        "parameter_tensors": len(current),
        "parameters": sum(parameter.numel() for parameter in current.values()),
    }


def architecture_contract(
    args: argparse.Namespace,
    input_shape: Sequence[int],
    memory_shape: Sequence[int],
    llm_hidden_size: int,
    input_contract: Mapping[str, Any],
    initializer: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "format": "dense_tensor_memory_cross_attention_v1",
        "qwen_model": str(args.model_name_or_path),
        "llm_hidden_size": int(llm_hidden_size),
        "input_source": str(args.input_source),
        "memory_init_mode": str(args.memory_init_mode),
        "input_shape": [1, -1, -1] if getattr(args, "shape_mode", "fixed") == "variable" else [int(value) for value in input_shape],
        "shape_mode": getattr(args, "shape_mode", "fixed"),
        # Retain the historical key for paper-checkpoint compatibility. In
        # Direct-Cross it describes the online field-encoder output.
        "latent_shape": ([int(memory_shape[0]), -1, -1] if getattr(args, "shape_mode", "fixed") == "variable"
                         else [int(value) for value in memory_shape]),
        "layers_1based": [int(value) for value in args.cross_attention_layers],
        "bridge_dim": int(args.bridge_dim),
        "heads": int(args.bridge_heads),
        "dropout": float(args.bridge_dropout),
        "gate_init": float(args.gate_init),
        "value_fourier_bands": int(args.value_fourier_bands),
        "value_hidden_dim": int(args.value_hidden_dim),
        "freeze_spatial_backbone": bool(args.freeze_spatial_backbone),
        "field_encoder_trainable": bool(args.field_encoder_trainable),
        "latent_channel_policy": str(args.latent_channel_policy),
        "initializer": dict(initializer),
        "input_contract": copy.deepcopy(dict(input_contract)),
        "latent_contract": (
            copy.deepcopy(dict(input_contract))
            if args.input_source == "precomputed_latent"
            else {}
        ),
        "forbidden_model_inputs": [
            "query_spec",
            "grounding_target",
            "structured task ID",
            "parsed coordinates",
            "task-to-active-slot mapping",
        ],
    }


def validate_checkpoint_contract(
    checkpoint: Mapping[str, Any],
    expected_contract: Mapping[str, Any],
) -> None:
    if str(checkpoint.get("checkpoint_type", "")) != CHECKPOINT_TYPE:
        raise ValueError("Resume/best file is not a dense cross-attention checkpoint.")
    if int(checkpoint.get("checkpoint_version", 0)) != CHECKPOINT_VERSION:
        raise ValueError("Unsupported dense cross-attention checkpoint version.")
    observed = checkpoint.get("architecture")
    if not isinstance(observed, Mapping):
        raise ValueError("Dense cross-attention checkpoint is missing its architecture contract.")
    stable_keys = (
        "format",
        "llm_hidden_size",
        "latent_shape",
        "layers_1based",
        "bridge_dim",
        "heads",
        "dropout",
        "gate_init",
        "value_fourier_bands",
        "value_hidden_dim",
        "freeze_spatial_backbone",
    )
    differences = {
        key: {"expected": expected_contract.get(key), "observed": observed.get(key)}
        for key in stable_keys
        if observed.get(key) != expected_contract.get(key)
    }
    for key, default in (("shape_mode", "fixed"), ("training_recipe", None)):
        if observed.get(key, default) != expected_contract.get(key, default):
            differences[key] = {"expected": expected_contract.get(key, default), "observed": observed.get(key, default)}
    observed_model = str(observed.get("qwen_model", "")).replace("\\", "/").rstrip("/")
    expected_model = str(expected_contract.get("qwen_model", "")).replace("\\", "/").rstrip("/")
    observed_model_identity = observed_model.rsplit("/", 1)[-1].casefold()
    expected_model_identity = expected_model.rsplit("/", 1)[-1].casefold()
    if (
        not observed_model_identity
        or observed_model_identity != expected_model_identity
    ):
        differences["qwen_model_identity"] = {
            "expected": expected_model,
            "observed": observed_model,
        }
    observed_input_source = str(observed.get("input_source", "precomputed_latent"))
    expected_input_source = str(
        expected_contract.get("input_source", "precomputed_latent")
    )
    if observed_input_source != expected_input_source:
        differences["input_source"] = {
            "expected": expected_input_source,
            "observed": observed_input_source,
        }
    observed_init_mode = str(
        observed.get("memory_init_mode", "direct_qa_checkpoint")
    )
    expected_init_mode = str(
        expected_contract.get("memory_init_mode", "direct_qa_checkpoint")
    )
    if observed_init_mode != expected_init_mode:
        differences["memory_init_mode"] = {
            "expected": expected_init_mode,
            "observed": observed_init_mode,
        }
    if expected_input_source == "raw_hdf5":
        for key in ("input_shape", "field_encoder_trainable"):
            if observed.get(key) != expected_contract.get(key):
                differences[key] = {
                    "expected": expected_contract.get(key),
                    "observed": observed.get(key),
                }
    observed_latent_channel_policy = str(observed.get("latent_channel_policy", "all"))
    expected_latent_channel_policy = str(expected_contract.get("latent_channel_policy", "all"))
    if observed_latent_channel_policy != expected_latent_channel_policy:
        differences["latent_channel_policy"] = {
            "expected": expected_latent_channel_policy,
            "observed": observed_latent_channel_policy,
        }
    expected_init = expected_contract.get("initializer", {})
    observed_init = observed.get("initializer", {})
    if not isinstance(expected_init, Mapping) or not isinstance(observed_init, Mapping) or (
        expected_init.get("sha256") != observed_init.get("sha256")
    ):
        differences["initializer.sha256"] = {
            "expected": expected_init.get("sha256") if isinstance(expected_init, Mapping) else None,
            "observed": observed_init.get("sha256") if isinstance(observed_init, Mapping) else None,
        }
    if expected_input_source == "raw_hdf5":
        expected_input = expected_contract.get("input_contract", {})
        observed_input = observed.get("input_contract", {})
        if not isinstance(expected_input, Mapping) or not isinstance(observed_input, Mapping) or (
            expected_input.get("identity_sha256") != observed_input.get("identity_sha256")
        ):
            differences["input_contract.identity_sha256"] = {
                "expected": (
                    expected_input.get("identity_sha256")
                    if isinstance(expected_input, Mapping)
                    else None
                ),
                "observed": (
                    observed_input.get("identity_sha256")
                    if isinstance(observed_input, Mapping)
                    else None
                ),
            }
    else:
        expected_latent = expected_contract.get("latent_contract", {})
        observed_latent = observed.get("latent_contract", {})
        if not isinstance(expected_latent, Mapping) or not isinstance(observed_latent, Mapping) or (
            expected_latent.get("alignment_checkpoint_sha256")
            != observed_latent.get("alignment_checkpoint_sha256")
        ):
            differences["latent_contract.alignment_checkpoint_sha256"] = {
                "expected": (
                    expected_latent.get("alignment_checkpoint_sha256")
                    if isinstance(expected_latent, Mapping)
                    else None
                ),
                "observed": (
                    observed_latent.get("alignment_checkpoint_sha256")
                    if isinstance(observed_latent, Mapping)
                    else None
                ),
            }
    if differences:
        raise ValueError(f"Checkpoint architecture does not match this run: {differences}.")


def checkpoint_payload(
    sidecar: nn.Module,
    architecture: Mapping[str, Any],
    args: argparse.Namespace,
    *,
    global_step: int,
    epoch: int,
    next_batch_index: int,
    elapsed_seconds: float,
    metrics: Mapping[str, Any] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "checkpoint_type": CHECKPOINT_TYPE,
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": copy.deepcopy(dict(architecture)),
        "trainable_state_dict": trainable_state_dict(sidecar),
        "progress": {
            "global_step": int(global_step),
            "epoch": int(epoch),
            "next_batch_index": int(next_batch_index),
            "elapsed_seconds": float(elapsed_seconds),
        },
        "metrics": copy.deepcopy(dict(metrics or {})),
        "args": {
            key: value
            for key, value in vars(args).items()
            if key not in {"raw_config"} and not str(key).endswith("api_key")
        },
        "torch_rng_state": torch.get_rng_state().cpu(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state"] = torch.cuda.get_rng_state().cpu()
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    return payload


def load_resume_checkpoint(
    path: str | Path,
    sidecar: nn.Module,
    architecture: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(Path(path).expanduser(), map_location=device, weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Unsupported resume checkpoint: {path}.")
    validate_checkpoint_contract(checkpoint, architecture)
    state = checkpoint.get("trainable_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("Resume checkpoint has no trainable_state_dict.")
    load_report = load_trainable_state_dict(sidecar, state)
    optimizer_state = checkpoint.get("optimizer_state_dict")
    scheduler_state = checkpoint.get("scheduler_state_dict")
    if not isinstance(optimizer_state, Mapping) or not isinstance(scheduler_state, Mapping):
        raise ValueError("Resume checkpoint lacks optimizer or scheduler state.")
    optimizer.load_state_dict(optimizer_state)
    scheduler.load_state_dict(scheduler_state)
    progress = checkpoint.get("progress")
    if not isinstance(progress, Mapping):
        raise ValueError("Resume checkpoint lacks progress state.")
    if isinstance(checkpoint.get("torch_rng_state"), torch.Tensor):
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    if device.type == "cuda" and isinstance(checkpoint.get("cuda_rng_state"), torch.Tensor):
        torch.cuda.set_rng_state(checkpoint["cuda_rng_state"].cpu(), device=device)
    return {
        **load_report,
        "global_step": int(progress.get("global_step", 0)),
        "epoch": int(progress.get("epoch", 1)),
        "next_batch_index": int(progress.get("next_batch_index", 0)),
        "elapsed_seconds": float(progress.get("elapsed_seconds", 0.0)),
        "path": canonical_path(path),
        "sha256": sha256_file(path),
        "checkpoint_metrics": copy.deepcopy(
            dict(checkpoint.get("metrics", {}))
            if isinstance(checkpoint.get("metrics"), Mapping)
            else {}
        ),
    }


def evaluation_latents(
    mode: str,
    batch: Mapping[str, Any],
    dataset: TensorReadoutQADataset,
) -> torch.Tensor | None:
    if mode == "no_tensor":
        return None
    latent = batch["latent_map"]
    if mode in {"correct", "zero_tensor"}:
        return latent
    if mode == "shuffled":
        return torch.stack(
            [dataset.load_shuffled_latent(int(index)) for index in batch["indices"]],
            dim=0,
        )
    raise ValueError(f"Unsupported cross-attention evaluation mode: {mode}.")


def predictions_from_prompt_hidden(
    prompt_hidden: torch.Tensor,
    records: Sequence[Mapping[str, Any]],
    tokenizer,
    output_embeddings: nn.Module,
) -> list[str]:
    specification = single_token_choice_ids(records, tokenizer)
    if specification is None:
        raise ValueError("Cross-attention evaluation requires unique single-token choices.")
    candidate_ids, _targets = specification
    row_logits = restricted_choice_logits(prompt_hidden, candidate_ids, output_embeddings)
    predictions: list[str] = []
    for record, logits in zip(records, row_logits):
        choices = [str(value) for value in record.get("choices", [])]
        answer = str(record.get("answer", ""))
        if answer not in choices:
            choices = [answer] + choices
        predictions.append(choices[int(torch.argmax(logits).item())])
    return predictions


def add_evaluation_deltas(metrics: dict[str, Any]) -> None:
    modes = metrics.get("modes")
    if not isinstance(modes, Mapping) or "correct" not in modes:
        return
    correct = modes["correct"]
    if not isinstance(correct, Mapping):
        return
    deltas: dict[str, Any] = {}
    for baseline_name in ("no_tensor", "zero_tensor", "shuffled"):
        baseline = modes.get(baseline_name)
        if not isinstance(baseline, Mapping):
            continue
        task_delta: dict[str, float] = {}
        correct_tasks = correct.get("by_task", {})
        baseline_tasks = baseline.get("by_task", {})
        if isinstance(correct_tasks, Mapping) and isinstance(baseline_tasks, Mapping):
            for task in sorted(set(correct_tasks) & set(baseline_tasks)):
                task_delta[str(task)] = float(correct_tasks[task]["accuracy"]) - float(
                    baseline_tasks[task]["accuracy"]
                )
        deltas[baseline_name] = {
            "accuracy": float(correct.get("accuracy", 0.0)) - float(baseline.get("accuracy", 0.0)),
            "macro_accuracy": float(correct.get("macro_accuracy", 0.0))
            - float(baseline.get("macro_accuracy", 0.0)),
            "by_task": task_delta,
        }
    metrics["correct_minus"] = deltas


@torch.no_grad()
def evaluate(
    llm: nn.Module,
    sidecar: DenseCrossAttentionSidecar,
    tokenizer,
    dataset: TensorReadoutQADataset,
    device: torch.device,
    model_dtype: torch.dtype,
    args: argparse.Namespace,
    modes: Sequence[str],
) -> dict[str, Any]:
    previous_checkpoint_mode = bool(
        getattr(llm, "is_gradient_checkpointing", False)
        and getattr(decoder_backbone(llm), "training", False)
    )
    llm.eval()
    sidecar.eval()
    sampler = (
        ExactDistributedEvalSampler(
            dataset,
            rank=distributed_rank(),
            num_replicas=distributed_world_size(),
        )
        if distributed_is_initialized()
        else None
    )
    dynamic = getattr(args, "shape_mode", "fixed") == "variable"
    batching = {"batch_sampler": ShapeBatchSampler(dataset, int(args.eval_batch_size), rank=distributed_rank(), num_replicas=distributed_world_size())} if dynamic else {
        "batch_size": max(1, int(args.eval_batch_size)), "shuffle": False, "sampler": sampler,
    }
    loader = DataLoader(
        dataset,
        **batching,
        num_workers=int(args.num_workers),
        persistent_workers=False,
        prefetch_factor=1 if int(args.num_workers) > 0 else None,
        pin_memory=device.type == "cuda",
        collate_fn=collate_tensor_readout,
    )
    output_embeddings = llm.get_output_embeddings()
    if output_embeddings is None:
        raise ValueError("The frozen Qwen model has no output embedding head.")
    result: dict[str, Any] = {
        "records": len(dataset),
        "modes": {},
    }
    for mode in modes:
        if mode not in SUPPORTED_EVAL_MODES:
            raise ValueError(f"Unsupported evaluation mode: {mode}.")
        total = 0
        correct = 0
        task_total: dict[str, int] = defaultdict(int)
        task_correct: dict[str, int] = defaultdict(int)
        field_total: dict[str, int] = defaultdict(int)
        field_correct: dict[str, int] = defaultdict(int)
        task_field_total: dict[str, int] = defaultdict(int)
        task_field_correct: dict[str, int] = defaultdict(int)
        shape_counts: dict[str, list[int]] = {}
        progress = tqdm(
            loader,
            desc=f"Eval [{mode}]",
            disable=not bool(args.console_progress) or not is_main_process(),
            leave=False,
        )
        for batch in progress:
            records = list(batch["records"])
            input_ids, attention_mask = build_strict_prompt_tensors(records, tokenizer, args)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            latent = evaluation_latents(mode, batch, dataset)
            if latent is not None:
                latent = latent.to(device, non_blocking=True)
            try:
                with autocast_context(device, model_dtype):
                    sidecar.bind(latent, mode=mode)
                    outputs = decoder_backbone(llm)(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        return_dict=True,
                    )
                    last_indices = last_nonpadding_indices(attention_mask)
                    rows = torch.arange(input_ids.shape[0], device=device)
                    prompt_hidden = outputs.last_hidden_state[rows, last_indices]
                    predictions = predictions_from_prompt_hidden(
                        prompt_hidden,
                        records,
                        tokenizer,
                        output_embeddings,
                    )
            finally:
                sidecar.clear()
            for record, prediction in zip(records, predictions):
                answer = str(record["answer"])
                is_correct = int(prediction == answer)
                task = str(record.get("task_type", "unknown"))
                field = str(record.get("field") or record.get("metadata", {}).get("field") or "unknown")
                task_field = f"{task}|{field}"
                total += 1
                correct += is_correct
                task_total[task] += 1
                task_correct[task] += is_correct
                field_total[field] += 1
                field_correct[field] += is_correct
                task_field_total[task_field] += 1
                task_field_correct[task_field] += is_correct
                if dynamic:
                    for key in (f"shape:{shape_name(record_shape(record))}",
                                f"shape_task:{shape_name(record_shape(record))}|{task}",
                                f"partition:{record['metadata']['shape_partition']}"):
                        counts = shape_counts.setdefault(key, [0, 0])
                        counts[0] += is_correct
                        counts[1] += 1
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

        def accuracy_map(totals: Mapping[str, int], corrects: Mapping[str, int]) -> dict[str, Any]:
            return {
                key: {
                    "correct": int(corrects.get(key, 0)),
                    "total": int(value),
                    "accuracy": float(corrects.get(key, 0) / max(1, value)),
                }
                for key, value in sorted(totals.items())
            }

        by_task = accuracy_map(task_total, task_correct)
        task_accuracies = [float(value["accuracy"]) for value in by_task.values()]
        result["modes"][mode] = {
            "correct": int(correct),
            "total": int(total),
            "accuracy": float(correct / max(1, total)),
            "macro_accuracy": float(sum(task_accuracies) / max(1, len(task_accuracies))),
            "by_task": by_task,
            "by_field": accuracy_map(field_total, field_correct),
            "by_task_field": accuracy_map(task_field_total, task_field_correct),
        }
        if dynamic:
            gathered = [shape_counts]
            if distributed_is_initialized():
                gathered = [None] * distributed_world_size()
                dist.all_gather_object(gathered, shape_counts)
            combined: dict[str, list[int]] = {}
            for rank_counts in gathered:
                for key, counts in rank_counts.items():
                    target = combined.setdefault(key, [0, 0])
                    target[0] += counts[0]
                    target[1] += counts[1]
            for prefix, name in (("shape:", "by_shape"), ("shape_task:", "by_shape_task"), ("partition:", "by_shape_partition")):
                result["modes"][mode][name] = {
                    key[len(prefix):]: {"correct": counts[0], "total": counts[1], "accuracy": counts[0] / counts[1]}
                    for key, counts in sorted(combined.items()) if key.startswith(prefix)
                }
            if total != len(dataset):
                raise RuntimeError(f"Variable evaluation lost or duplicated records: {total} != {len(dataset)}.")
    add_evaluation_deltas(result)
    set_frozen_llm_execution_mode(llm, checkpoint_training=previous_checkpoint_mode)
    sidecar.train()
    return result


def selection_score(metrics: Mapping[str, Any], metric_name: str) -> float:
    modes = metrics.get("modes")
    correct = modes.get("correct") if isinstance(modes, Mapping) else None
    if not isinstance(correct, Mapping):
        raise ValueError("Selection metrics do not contain a correct tensor mode.")
    if metric_name == "accuracy":
        return float(correct.get("accuracy", 0.0))
    if metric_name == "macro_accuracy":
        return float(correct.get("macro_accuracy", 0.0))
    tasks = correct.get("by_task")
    normalized = tasks.get("normalized_point_value") if isinstance(tasks, Mapping) else None
    if not isinstance(normalized, Mapping):
        raise ValueError("normalized_accuracy selection requires normalized_point_value records.")
    return float(normalized.get("accuracy", 0.0))


def print_eval_summary(label: str, metrics: Mapping[str, Any]) -> None:
    if not is_main_process():
        return
    modes = metrics.get("modes", {})
    mode_items = modes.items() if isinstance(modes, Mapping) else []
    for mode, values in mode_items:
        tasks = values.get("by_task", {}) if isinstance(values, Mapping) else {}
        task_text = " ".join(
            f"{task}={float(task_values['accuracy']):.4f}"
            for task, task_values in tasks.items()
        )
        print(
            f"{label} mode={mode} accuracy={float(values.get('accuracy', 0.0)):.4f} "
            f"macro={float(values.get('macro_accuracy', 0.0)):.4f} {task_text}"
        )
        for partition, counts in values.get("by_shape_partition", {}).items():
            print(f"{label} mode={mode} shape_partition={partition} accuracy={counts['accuracy']:.4f} n={counts['total']}")
        for shape, counts in values.get("by_shape", {}).items():
            tasks_for_shape = " ".join(f"{key.split('|', 1)[1]}={item['accuracy']:.4f}"
                                       for key, item in values.get("by_shape_task", {}).items() if key.startswith(shape + "|"))
            print(f"{label} mode={mode} shape={shape} accuracy={counts['accuracy']:.4f} n={counts['total']} {tasks_for_shape}")


def save_checkpoint_on_rank_zero(
    path: Path,
    payload_factory,
    stage: str,
) -> None:
    run_on_rank_zero_and_broadcast(
        lambda: atomic_torch_save(path, payload_factory()),
        stage,
    )


class StopController:
    def __init__(self) -> None:
        self.requested = False
        self.signal_name: str | None = None

    def install(self) -> None:
        def handler(signum, _frame) -> None:
            self.requested = True
            try:
                self.signal_name = signal.Signals(signum).name
            except ValueError:
                self.signal_name = str(signum)

        for name in ("SIGTERM", "SIGINT"):
            value = getattr(signal, name, None)
            if value is not None:
                signal.signal(value, handler)

    def distributed_reason(
        self,
        device: torch.device,
        *,
        reached_updates: bool,
        reached_time: bool,
    ) -> str | None:
        flags = torch.tensor(
            [int(self.requested), int(reached_updates), int(reached_time)],
            dtype=torch.int32,
            device=device,
        )
        if distributed_is_initialized():
            dist.all_reduce(flags, op=dist.ReduceOp.MAX)
        if bool(flags[0].item()):
            return "signal"
        if bool(flags[1].item()):
            return "planned_updates"
        if bool(flags[2].item()):
            return "wall_clock_reserve"
        return None


def connectivity_audit(
    llm: nn.Module,
    sidecar: DenseCrossAttentionSidecar,
    tokenizer,
    batch: Mapping[str, Any],
    device: torch.device,
    model_dtype: torch.dtype,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run one non-updating backward with gates temporarily open."""

    original_gates = [bridge.gate.detach().clone() for bridge in sidecar.bridges]
    for bridge in sidecar.bridges:
        bridge.gate.data.fill_(0.01)
    for _name, parameter in trainable_named_parameters(sidecar):
        parameter.grad = None
    total: torch.Tensor | None = None
    try:
        total, metrics = forward_training_batch(
            llm,
            sidecar,
            tokenizer,
            list(batch["records"]),
            batch["latent_map"],
            device,
            model_dtype,
            args,
        )
        (total * len(batch["records"])).backward()
        sidecar.clear()
        missing = [
            name
            for name, parameter in trainable_named_parameters(sidecar)
            if parameter.grad is None
        ]
        nonfinite = [
            name
            for name, parameter in trainable_named_parameters(sidecar)
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
        ]
        if missing or nonfinite:
            raise RuntimeError(
                "Cross-attention startup backward audit failed: "
                f"missing_grad={missing[:12]}, nonfinite_grad={nonfinite[:12]}."
            )
        return {
            "passed": True,
            "trainable_tensors": len(trainable_named_parameters(sidecar)),
            "loss": float(metrics["loss"]),
            "temporary_gate": float(torch.tanh(torch.tensor(0.01)).item()),
        }
    finally:
        sidecar.clear()
        for bridge, value in zip(sidecar.bridges, original_gates):
            bridge.gate.data.copy_(value)
        for _name, parameter in trainable_named_parameters(sidecar):
            parameter.grad = None


def reduced_metric_means(
    sums: Mapping[str, float],
    records: int,
    device: torch.device,
) -> dict[str, float]:
    names = sorted(sums)
    values = torch.tensor(
        [float(sums[name]) for name in names] + [float(records)],
        dtype=torch.float64,
        device=device,
    )
    if distributed_is_initialized():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    global_records = max(1.0, float(values[-1].item()))
    return {
        name: float(values[index].item() / global_records)
        for index, name in enumerate(names)
    }


def current_lrs(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    return {
        str(group.get("name", f"group_{index}")): float(group["lr"])
        for index, group in enumerate(optimizer.param_groups)
    }


def elapsed_seconds(process_start: float, previous_seconds: float) -> float:
    return float(previous_seconds + (time.monotonic() - process_start))


def save_best_checkpoint(
    path: Path,
    sidecar: DenseCrossAttentionSidecar,
    architecture: Mapping[str, Any],
    args: argparse.Namespace,
    *,
    global_step: int,
    epoch: int,
    next_batch_index: int,
    elapsed: float,
    metrics: Mapping[str, Any],
) -> None:
    save_checkpoint_on_rank_zero(
        path,
        lambda: checkpoint_payload(
            sidecar,
            architecture,
            args,
            global_step=global_step,
            epoch=epoch,
            next_batch_index=next_batch_index,
            elapsed_seconds=elapsed,
            metrics=metrics,
        ),
        f"best checkpoint step {global_step}",
    )


def save_last_checkpoint(
    path: Path,
    sidecar: DenseCrossAttentionSidecar,
    architecture: Mapping[str, Any],
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    global_step: int,
    epoch: int,
    next_batch_index: int,
    elapsed: float,
    metrics: Mapping[str, Any],
) -> None:
    save_checkpoint_on_rank_zero(
        path,
        lambda: checkpoint_payload(
            sidecar,
            architecture,
            args,
            global_step=global_step,
            epoch=epoch,
            next_batch_index=next_batch_index,
            elapsed_seconds=elapsed,
            metrics=metrics,
            optimizer=optimizer,
            scheduler=scheduler,
        ),
        f"resumable checkpoint step {global_step}",
    )


def main() -> None:
    process_start = time.monotonic()
    args = parse_args()
    apply_runtime_environment(args)
    device = initialize_distributed_device(
        args.device,
        distributed_timeout_seconds=float(args.distributed_timeout_seconds),
    )
    # Build identical trainable sidecars before rank-0 synchronization.
    seed_everything(int(args.seed))
    stop_controller = StopController()
    stop_controller.install()
    if args.resume:
        resume_parent = str(Path(args.resume).expanduser().resolve().parent)
        run_dir_value = run_on_rank_zero_and_broadcast(
            lambda: resume_parent,
            "resume run-directory resolution",
        )
        run_dir = Path(str(run_dir_value))
    else:
        run_dir = build_distributed_run_dir(args.output_root, args.run_name)
    last_path = run_dir / "cross_attention_last.pt"
    best_path = run_dir / "cross_attention_best.pt"
    if is_main_process():
        print(
            f"startup=run rank={distributed_rank()}/{distributed_world_size()} "
            f"device={device} run_dir={run_dir}"
        )

    try:
        metadata, input_contract = run_on_rank_zero_and_broadcast(
            lambda: load_metadata_and_contract(args),
            "matched QA metadata validation",
        )
        train_dataset, val_dataset, test_dataset, screening_dataset = build_datasets(
            args,
            input_contract,
        )
        datasets = {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset,
        }
        data_audit = run_on_rank_zero_and_broadcast(
            lambda: audit_general_qa_datasets(
                datasets,
                require_disjoint_splits=bool(args.require_disjoint_splits),
            ),
            "general QA audit",
        )
        if args.input_source == "raw_hdf5":
            raw_content_audit = run_on_rank_zero_and_broadcast(
                lambda: audit_raw_hdf5_input_content(datasets),
                "raw HDF5 input content audit",
            )
            input_contract = copy.deepcopy(dict(input_contract))
            input_contract["selected_patch_content"] = copy.deepcopy(
                dict(raw_content_audit)
            )
            input_contract["identity_sha256"] = _json_sha256(
                {
                    key: value
                    for key, value in input_contract.items()
                    if key
                    not in {
                        "identity_sha256",
                        "hdf5_path",
                        "hdf5_bytes",
                        "qa_generation_alignment_sha256",
                    }
                }
            )
            metadata = copy.deepcopy(dict(metadata))
            input_resolution = copy.deepcopy(
                dict(metadata.get("runtime_input_resolution", {}))
            )
            input_resolution["identity_sha256"] = input_contract[
                "identity_sha256"
            ]
            input_resolution["selected_patch_content_sha256"] = raw_content_audit[
                "sha256"
            ]
            metadata["runtime_input_resolution"] = input_resolution
            data_audit = copy.deepcopy(dict(data_audit))
            data_audit["raw_input_content"] = copy.deepcopy(dict(raw_content_audit))
        tokenizer = load_tokenizer(args)
        prompt_audit = run_on_rank_zero_and_broadcast(
            lambda: audit_prompt_tokenization(
                datasets,
                tokenizer,
                max_prompt_tokens=int(args.max_prompt_tokens),
                prompt_template=str(args.prompt_template),
                audit_local_conditioning_prompt=False,
            ),
            "prompt audit",
        )
        if not bool(prompt_audit.get("all_prompts_fit", False)):
            raise ValueError(
                "At least one formal prompt exceeds max_prompt_tokens; this script never truncates prompts."
            )
        choice_audit = run_on_rank_zero_and_broadcast(
            lambda: audit_choice_tokenization(datasets, tokenizer),
            "choice-token audit",
        )
        if not bool(choice_audit.get("all_labels_single_token", False)):
            raise ValueError("This efficient trainer requires every choice label to be one unique token.")

        def write_startup_audits() -> None:
            atomic_dump_json(run_dir / "data_audit.json", data_audit)
            atomic_dump_json(run_dir / "prompt_audit.json", prompt_audit)
            atomic_dump_json(run_dir / "choice_tokenization_audit.json", choice_audit)
            atomic_dump_json(run_dir / "config_snapshot.json", args.raw_config)
            atomic_dump_json(
                run_dir / "resolved_args.json",
                {
                    key: value
                    for key, value in vars(args).items()
                    if key != "raw_config" and not str(key).endswith("api_key")
                },
            )

        if not args.resume:
            run_on_rank_zero_and_broadcast(write_startup_audits, "startup audit writes")
        if is_main_process():
            print(
                "startup=data "
                f"train/val/test/screen={len(train_dataset)}/{len(val_dataset)}/"
                f"{len(test_dataset)}/{len(screening_dataset)}"
            )

        llm, model_dtype = load_llm_with_bounded_host_memory(args, device)
        llm_hidden_size = int(llm.get_input_embeddings().embedding_dim)
        first_field_input = train_dataset[0]["latent_map"]
        input_shape = tuple(int(value) for value in first_field_input.shape)
        field_encoder: DirectFieldEncoder | None = None
        if args.memory_init_mode == "scratch":
            # Decouple the ablation initializer from any RNG consumed while the
            # frozen Transformers checkpoint is being materialized.
            seed_everything(int(args.seed))
            (
                field_encoder,
                spatial_initializer,
                memory_shape,
                initializer_provenance,
            ) = build_scratch_memory_components(args, input_shape)
        else:
            memory_shape = input_shape
            spatial_initializer, initializer_provenance = load_spatial_initializer(
                args.memory_init_checkpoint,
                memory_shape,
                llm_hidden_size,
                input_contract,
                args.model_name_or_path,
                args.latent_channel_policy,
            )
        sidecar, install_report = build_sidecar(
            llm,
            spatial_initializer,
            args,
            device,
            field_encoder=field_encoder,
        )
        if args.memory_init_mode == "scratch":
            initializer_provenance["field_spatial_sha256"] = initializer_provenance[
                "sha256"
            ]
            initializer_provenance["sha256"] = _module_state_sha256(
                {"full_sidecar": sidecar}
            )
        synchronize_trainable_sidecar(sidecar)
        # Any configured stochastic sidecar operation receives a deterministic,
        # rank-specific stream after the common initialization is synchronized.
        seed_everything(int(args.seed) + distributed_rank())
        set_frozen_llm_execution_mode(
            llm,
            checkpoint_training=bool(args.llm_gradient_checkpointing),
        )
        sidecar.train()

        sampler_class = ShapeBatchSampler if args.shape_mode == "variable" else StateTaskGroupedBatchSampler
        sampler_options = {"training": True} if args.shape_mode == "variable" else {"questions_per_group": int(args.questions_per_state_group)}
        train_sampler = sampler_class(
            dataset=train_dataset,
            batch_size=int(args.batch_size),
            **sampler_options,
            seed=int(args.shuffle_seed),
            rank=distributed_rank(),
            num_replicas=distributed_world_size(),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=int(args.num_workers),
            persistent_workers=int(args.num_workers) > 0,
            prefetch_factor=1 if int(args.num_workers) > 0 else None,
            pin_memory=device.type == "cuda",
            collate_fn=collate_tensor_readout,
        )
        accumulation_steps = int(args.gradient_accumulation_steps)
        updates_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
        planned_updates = updates_per_epoch * int(args.epochs)
        if args.shape_mode == "variable" and int(args.max_updates) > planned_updates:
            raise ValueError(f"Epoch budget allows {planned_updates} updates, less than max_updates={args.max_updates}; increase --epochs or lower --max-updates.")
        if int(args.max_updates) > 0:
            planned_updates = min(planned_updates, int(args.max_updates))
        optimizer, parameter_report = build_optimizer(sidecar, args)
        if args.input_source == "raw_hdf5":
            required_trainable_groups = (
                "trainable_field_encoder_parameters",
                "trainable_spatial_parameters",
                "trainable_value_parameters",
                "trainable_cross_attention_parameters",
            )
            missing_groups = [
                name
                for name in required_trainable_groups
                if int(parameter_report.get(name, 0)) <= 0
            ]
            if missing_groups:
                raise RuntimeError(
                    f"Direct-Cross is missing trainable final-path parameter groups: {missing_groups}."
                )
        sidecar_parameter_ids = {id(value) for value in sidecar.parameters()}
        parameter_report["frozen_llm_parameters"] = sum(
            parameter.numel()
            for parameter in llm.parameters()
            if id(parameter) not in sidecar_parameter_ids
        )
        scheduler, warmup_updates = build_sidecar_scheduler(
            optimizer,
            scheduler_name=str(args.lr_scheduler),
            total_updates=planned_updates,
            warmup_ratio=float(args.warmup_ratio),
            min_lr_ratio=float(args.min_lr_ratio),
        )
        architecture = architecture_contract(
            args,
            input_shape,
            memory_shape,
            llm_hidden_size,
            input_contract,
            initializer_provenance,
        )
        if args.shape_mode == "variable":
            recipe_keys = ("experiment_profile", "seed", "shuffle_seed", "epochs", "batch_size",
                           "gradient_accumulation_steps", "lr", "gate_lr", "lr_scheduler", "warmup_ratio",
                           "min_lr_ratio", "weight_decay", "grad_clip_norm", "choice_ce_weight",
                           "full_answer_ce_weight", "matched_group_weight", "matched_group_margin",
                           "value_reconstruction_weight", "prompt_template", "max_prompt_tokens",
                           "max_target_tokens", "append_eos", "screening_records", "screening_fractions", "selection_metric")
            architecture["training_recipe"] = {key: getattr(args, key) for key in recipe_keys}
            architecture["training_recipe"].update(planned_updates=planned_updates, world_size=distributed_world_size())
            parameter_report["shape_padding_records_per_epoch"] = train_sampler.padding_records_per_epoch
            if is_main_process():
                print(f"startup=variable_shapes profile={args.experiment_profile} train_shapes={args.train_shapes} "
                      f"padding_records_per_epoch={train_sampler.padding_records_per_epoch} planned_updates={planned_updates}")
        run_contract = {
            "architecture": architecture,
            "qa_metadata": {
                "format": str(metadata.get("format", "")),
                "prompt_contract": str(metadata.get("prompt_contract", "")),
                "split_mode": str(metadata.get("split_mode", "")),
                "stage2b": copy.deepcopy(dict(metadata.get("stage2b", {}))),
                "alignment_checkpoint_resolution": copy.deepcopy(
                    dict(metadata.get("runtime_alignment_checkpoint_resolution", {}))
                ),
                "input_resolution": copy.deepcopy(
                    dict(metadata.get("runtime_input_resolution", {}))
                ),
            },
            "install": install_report,
            "parameters": parameter_report,
            "distributed": {
                "world_size": distributed_world_size(),
                "per_rank_batch_size": int(args.batch_size),
                "gradient_accumulation_steps": accumulation_steps,
                "effective_batch_size": int(args.batch_size)
                * distributed_world_size()
                * accumulation_steps,
                "updates_per_epoch": updates_per_epoch,
                "planned_updates": planned_updates,
            },
            "runtime": {
                "max_wall_clock_hours": float(args.max_wall_clock_hours),
                "final_eval_reserve_minutes": float(args.final_eval_reserve_minutes),
            },
            "optimizer": {
                "lr": float(args.lr),
                "gate_lr": float(args.gate_lr),
                "warmup_updates": warmup_updates,
            },
        }
        if not args.resume:
            run_on_rank_zero_and_broadcast(
                lambda: atomic_dump_json(run_dir / "run_contract.json", run_contract),
                "run contract write",
            )
        if is_main_process():
            print(
                "startup=model "
                f"input={args.input_source} init={args.memory_init_mode} "
                f"layers={args.cross_attention_layers} trainable={parameter_report['trainable_parameters']:,} "
                f"frozen_llm={parameter_report['frozen_llm_parameters']:,} "
                f"updates={planned_updates} warmup={warmup_updates}"
            )

        global_step = 0
        start_epoch = 0
        resume_batch_index = 0
        previous_elapsed = 0.0
        resume_report: dict[str, Any] | None = None
        best_score = -math.inf
        best_step = 0
        best_screen_metrics: dict[str, Any] = {}
        if args.resume:
            resume_report = load_resume_checkpoint(
                args.resume,
                sidecar,
                architecture,
                optimizer,
                scheduler,
                device,
            )
            global_step = int(resume_report["global_step"])
            start_epoch = int(resume_report["epoch"])
            resume_batch_index = int(resume_report["next_batch_index"])
            previous_elapsed = float(resume_report["elapsed_seconds"])
            checkpoint_metrics = resume_report.get("checkpoint_metrics", {})
            if isinstance(checkpoint_metrics, Mapping):
                best_score = float(checkpoint_metrics.get("best_score", -math.inf))
                best_step = int(checkpoint_metrics.get("best_step", global_step))
                best_screen_metrics = dict(checkpoint_metrics.get("best_screen_metrics", {}))
            if best_path.exists():
                saved_best = torch.load(best_path, map_location="cpu", weights_only=True)
                validate_checkpoint_contract(saved_best, architecture)
                saved_metrics = saved_best.get("metrics", {})
                best_score = float(saved_metrics.get("score", saved_metrics.get("best_score", best_score)))
                best_step = int(saved_best["progress"]["global_step"])
                best_screen_metrics = dict(saved_metrics.get("metrics", best_screen_metrics))
                del saved_best
            elif args.evaluate_only:
                raise FileNotFoundError(f"--evaluate-only requires {best_path}.")
            # Validate first: a rejected resume must not overwrite the original run's evidence.
            run_on_rank_zero_and_broadcast(write_startup_audits, "validated resume audit writes")
            run_on_rank_zero_and_broadcast(
                lambda: atomic_dump_json(run_dir / "run_contract.json", run_contract),
                "validated resume contract write",
            )
            run_on_rank_zero_and_broadcast(
                lambda: atomic_dump_json(run_dir / "resume_report.json", resume_report),
                "resume report write",
            )
            if is_main_process():
                print(
                    f"startup=resume step={global_step} epoch={start_epoch} "
                    f"batch={resume_batch_index} previous_elapsed={previous_elapsed / 3600.0:.2f}h"
                )

        audit_batch = next(iter(train_loader))
        connectivity = connectivity_audit(
            llm,
            sidecar,
            tokenizer,
            audit_batch,
            device,
            model_dtype,
            args,
        )
        run_on_rank_zero_and_broadcast(
            lambda: atomic_dump_json(run_dir / "connectivity_audit.json", connectivity),
            "connectivity audit write",
        )
        optimizer.zero_grad(set_to_none=True)

        if not args.resume or not best_path.exists() or not math.isfinite(best_score):
            initial_metrics = evaluate(
                llm,
                sidecar,
                tokenizer,
                screening_dataset,
                device,
                model_dtype,
                args,
                modes=["correct"] if args.shape_mode == "variable" else ["correct", "no_tensor"],
            )
            best_score = selection_score(initial_metrics, args.selection_metric)
            best_step = global_step
            best_screen_metrics = initial_metrics
            initial_record = {
                "step": global_step,
                "score": best_score,
                "selection_metric": args.selection_metric,
                "metrics": initial_metrics,
            }
            run_on_rank_zero_and_broadcast(
                lambda: atomic_dump_json(run_dir / "initial_screen_metrics.json", initial_record),
                "initial screen write",
            )
            save_best_checkpoint(
                best_path,
                sidecar,
                architecture,
                args,
                global_step=global_step,
                epoch=start_epoch,
                next_batch_index=resume_batch_index,
                elapsed=elapsed_seconds(process_start, previous_elapsed),
                metrics=initial_record,
            )
            print_eval_summary("initial_screen", initial_metrics)

        screening_updates = {
            max(1, min(planned_updates, int(round(planned_updates * fraction))))
            for fraction in args.screening_fractions
        }
        screening_updates.add(planned_updates)
        total_budget_seconds = float(args.max_wall_clock_hours) * 3600.0
        training_cutoff_seconds = total_budget_seconds - float(args.final_eval_reserve_minutes) * 60.0
        stop_reason = "completed"
        next_epoch_for_resume = start_epoch
        next_batch_for_resume = resume_batch_index
        log_sums: dict[str, float] = defaultdict(float)
        log_records = 0
        accumulation_records = 0
        accumulation_microbatches = 0
        progress = tqdm(
            total=planned_updates,
            initial=min(global_step, planned_updates),
            desc="Full-grid cross-attention",
            disable=not bool(args.console_progress) or not is_main_process(),
        )

        for epoch_index in range(start_epoch, start_epoch if args.evaluate_only else int(args.epochs)):
            train_sampler.set_epoch(epoch_index)
            epoch_skip = resume_batch_index if epoch_index == start_epoch else 0
            next_epoch_for_resume = epoch_index
            next_batch_for_resume = epoch_skip
            for batch_index, batch in enumerate(train_loader):
                if batch_index < epoch_skip:
                    continue
                pre_update_reason = stop_controller.distributed_reason(
                    device,
                    reached_updates=global_step >= planned_updates,
                    reached_time=(
                        elapsed_seconds(process_start, previous_elapsed)
                        >= training_cutoff_seconds
                    ),
                )
                if pre_update_reason is not None:
                    stop_reason = pre_update_reason
                    break
                records = list(batch["records"])
                loss, batch_metrics = forward_training_batch(
                    llm,
                    sidecar,
                    tokenizer,
                    records,
                    batch["latent_map"],
                    device,
                    model_dtype,
                    args,
                )
                try:
                    (loss * len(records)).backward()
                finally:
                    sidecar.clear()
                accumulation_records += len(records)
                accumulation_microbatches += 1
                log_records += len(records)
                for key, value in batch_metrics.items():
                    log_sums[key] += float(value) * len(records)

                end_of_loader = batch_index + 1 == len(train_loader)
                if accumulation_microbatches < accumulation_steps and not end_of_loader:
                    continue
                global_records = average_trainable_gradients_by_record_count(
                    sidecar,
                    accumulation_records,
                    device,
                )
                assert_finite_gradients(sidecar, f"optimizer update {global_step + 1}")
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        [parameter for _name, parameter in trainable_named_parameters(sidecar)],
                        max_norm=float(args.grad_clip_norm),
                    ).detach().float().cpu().item()
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accumulation_records = 0
                accumulation_microbatches = 0
                global_step += 1
                next_epoch_for_resume = epoch_index
                next_batch_for_resume = batch_index + 1
                progress.update(1)

                if global_step % int(args.log_interval) == 0 or global_step == 1:
                    means = reduced_metric_means(log_sums, log_records, device)
                    payload = {
                        "step": global_step,
                        "epoch": epoch_index + 1,
                        "batch_index": batch_index,
                        "elapsed_seconds": elapsed_seconds(process_start, previous_elapsed),
                        "global_records_last_update": global_records,
                        "grad_norm": grad_norm,
                        "gate_values": sidecar.gate_values(),
                        "learning_rates": current_lrs(optimizer),
                        **means,
                    }
                    run_on_rank_zero_and_broadcast(
                        lambda value=payload: append_jsonl(
                            run_dir / "train_metrics.jsonl", value
                        ),
                        f"training metric write step {global_step}",
                    )
                    if is_main_process():
                        progress.set_postfix(
                            loss=f"{means.get('loss', 0.0):.4f}",
                            acc=f"{means.get('accuracy', 0.0):.3f}",
                            gate=",".join(f"{value:.3f}" for value in sidecar.gate_values()),
                        )
                    log_sums.clear()
                    log_records = 0

                checkpoint_metrics = {
                    "best_score": float(best_score),
                    "best_step": int(best_step),
                    "selection_metric": str(args.selection_metric),
                    "stop_reason": stop_reason,
                }
                if int(args.save_every_updates) > 0 and global_step % int(args.save_every_updates) == 0:
                    save_last_checkpoint(
                        last_path,
                        sidecar,
                        architecture,
                        args,
                        optimizer,
                        scheduler,
                        global_step=global_step,
                        epoch=next_epoch_for_resume,
                        next_batch_index=next_batch_for_resume,
                        elapsed=elapsed_seconds(process_start, previous_elapsed),
                        metrics=checkpoint_metrics,
                    )

                if global_step in screening_updates:
                    screen_metrics = evaluate(
                        llm,
                        sidecar,
                        tokenizer,
                        screening_dataset,
                        device,
                        model_dtype,
                        args,
                        modes=["correct"],
                    )
                    score = selection_score(screen_metrics, args.selection_metric)
                    screen_record = {
                        "step": global_step,
                        "epoch": epoch_index + 1,
                        "score": score,
                        "selection_metric": args.selection_metric,
                        "gate_values": sidecar.gate_values(),
                        "metrics": screen_metrics,
                    }
                    run_on_rank_zero_and_broadcast(
                        lambda value=screen_record: append_jsonl(
                            run_dir / "screen_history.jsonl", value
                        ),
                        f"screen metric write step {global_step}",
                    )
                    print_eval_summary(f"screen_step_{global_step}", screen_metrics)
                    if score > best_score:
                        best_score = score
                        best_step = global_step
                        best_screen_metrics = screen_metrics
                        save_best_checkpoint(
                            best_path,
                            sidecar,
                            architecture,
                            args,
                            global_step=global_step,
                            epoch=next_epoch_for_resume,
                            next_batch_index=next_batch_for_resume,
                            elapsed=elapsed_seconds(process_start, previous_elapsed),
                            metrics={
                                **screen_record,
                                "best_score": float(best_score),
                                "best_step": int(best_step),
                            },
                        )

                elapsed_now = elapsed_seconds(process_start, previous_elapsed)
                local_time_stop = elapsed_now >= training_cutoff_seconds
                local_update_stop = global_step >= planned_updates
                distributed_stop_reason = stop_controller.distributed_reason(
                    device,
                    reached_updates=local_update_stop,
                    reached_time=local_time_stop,
                )
                if distributed_stop_reason is not None:
                    stop_reason = distributed_stop_reason
                    break
            if next_batch_for_resume >= len(train_loader):
                next_epoch_for_resume = epoch_index + 1
                next_batch_for_resume = 0
                resume_batch_index = 0
            if stop_reason != "completed":
                break
        progress.close()

        final_elapsed_before_eval = elapsed_seconds(process_start, previous_elapsed)
        final_checkpoint_metrics = {
            "best_score": float(best_score),
            "best_step": int(best_step),
            "selection_metric": str(args.selection_metric),
            "stop_reason": stop_reason,
            "best_screen_metrics": best_screen_metrics,
        }
        save_last_checkpoint(
            last_path,
            sidecar,
            architecture,
            args,
            optimizer,
            scheduler,
            global_step=global_step,
            epoch=next_epoch_for_resume,
            next_batch_index=next_batch_for_resume,
            elapsed=final_elapsed_before_eval,
            metrics=final_checkpoint_metrics,
        )

        if stop_reason == "signal":
            interrupted_summary = {
                "status": "interrupted_resumable",
                "stop_reason": stop_reason,
                "global_step": global_step,
                "planned_updates": planned_updates,
                "elapsed_seconds": final_elapsed_before_eval,
                "resume_checkpoint": str(last_path),
            }
            run_on_rank_zero_and_broadcast(
                lambda: atomic_dump_json(run_dir / "run_summary.json", interrupted_summary),
                "interrupted summary write",
            )
            if is_main_process():
                print(f"training interrupted safely; resume from {last_path}")
            return

        best_checkpoint = torch.load(best_path, map_location=device, weights_only=True)
        if not isinstance(best_checkpoint, Mapping):
            raise ValueError("Best cross-attention checkpoint is invalid.")
        validate_checkpoint_contract(best_checkpoint, architecture)
        best_state = best_checkpoint.get("trainable_state_dict")
        if not isinstance(best_state, Mapping):
            raise ValueError("Best cross-attention checkpoint has no trainable state.")
        load_trainable_state_dict(sidecar, best_state)
        final_val = evaluate(
            llm,
            sidecar,
            tokenizer,
            val_dataset,
            device,
            model_dtype,
            args,
            modes=args.eval_modes,
        )
        run_on_rank_zero_and_broadcast(
            lambda: atomic_dump_json(run_dir / "final_val_metrics.json", final_val),
            "final validation write",
        )
        print_eval_summary("final_val", final_val)
        final_test: dict[str, Any] | None = None
        if bool(args.evaluate_test):
            final_test = evaluate(
                llm,
                sidecar,
                tokenizer,
                test_dataset,
                device,
                model_dtype,
                args,
                modes=args.eval_modes,
            )
            run_on_rank_zero_and_broadcast(
                lambda: atomic_dump_json(run_dir / "final_test_metrics.json", final_test),
                "final test write",
            )
            print_eval_summary("final_test", final_test)

        final_elapsed = elapsed_seconds(process_start, previous_elapsed)
        summary = {
            "status": "complete" if global_step >= planned_updates else "training_incomplete",
            "training_budget_completed": global_step >= planned_updates,
            "evaluation_only": bool(args.evaluate_only),
            "experiment_profile": args.experiment_profile,
            "stop_reason": stop_reason,
            "global_step": global_step,
            "planned_updates": planned_updates,
            "best_step": best_step,
            "best_score": best_score,
            "selection_metric": args.selection_metric,
            "elapsed_seconds": final_elapsed,
            "elapsed_hours": final_elapsed / 3600.0,
            "train_records": len(train_dataset),
            "val_records": len(val_dataset),
            "test_records": len(test_dataset),
            "run_contract": run_contract,
            "resume": resume_report,
            "best_checkpoint": str(best_path),
            "last_checkpoint": str(last_path),
            "final_val": final_val,
            "final_test": final_test,
        }
        run_on_rank_zero_and_broadcast(
            lambda: atomic_dump_json(run_dir / "run_summary.json", summary),
            "final run summary write",
        )
        if is_main_process():
            print(
                f"completed step={global_step}/{planned_updates} best_step={best_step} "
                f"best_score={best_score:.4f} elapsed={final_elapsed / 3600.0:.2f}h "
                f"run_dir={run_dir}"
            )
    finally:
        if distributed_is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
