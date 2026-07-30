from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for search_path in (PROJECT_ROOT, SRC_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from tensor_compression.downstream.patch_qa_contract import (  # noqa: E402
    MATCHED_GROUP_FORMAT,
    PATCH_LATENT_AUDIT_FORMAT,
    PATCH_LATENT_FORMAT,
    PATCH_MATCHED_QA_FORMAT,
    PATCH_QA_BUILD_MARKER,
    PATCH_QA_PROMPT_CONTRACT,
    canonical_normalization,
    latent_identity_from_record,
    latent_qa_stats_from_record,
    validate_patch_latent_payload,
)
from tensor_compression.downstream.patch_qa_prompt import build_prompt  # noqa: E402
from tensor_compression.utils.pipeline_config import (  # noqa: E402
    first_nested,
    load_yaml_mapping,
    resolve_path_string,
    set_default,
)

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import logging as transformers_logging
except ImportError as exc:  # pragma: no cover - dependency error on the execution host
    raise ImportError(
        "scripts/evaluate_frozen_qwen_patch_qa.py requires transformers. "
        "Install it with: pip install transformers accelerate safetensors"
    ) from exc


# Historical result identifiers are retained for checkpoint compatibility.
BASELINE_NAME = "frozen_qwen_serialized_tensor"
RESULT_FORMAT = "frozen_qwen_serialized_tensor_patch_qa_baseline_v2"
PRESERVED_Z_CHANNEL = 0
PRESERVED_Z_MEAN_ATOL = 1.0e-1
PRESERVED_Z_STD_ATOL = 5.0e-3
PRESERVED_Z_STD_RTOL = 2.0e-2
EXPECTED_PATCH_SIZE = 16
EXTREME_OPERATION_RE = re.compile(r"\b(maximum|minimum)\b", re.IGNORECASE)
EXPECTED_TASKS = frozenset(
    {
        "extreme_quadrant",
        "normalized_point_value",
        "point_compare",
        "raw_point_value_with_stats",
        "region_mean_compare",
    }
)
ALLOWED_LABELS = frozenset({"A", "B", "C", "D"})


def distributed_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def distributed_rank() -> int:
    return int(dist.get_rank()) if distributed_is_initialized() else 0


def distributed_world_size() -> int:
    return int(dist.get_world_size()) if distributed_is_initialized() else 1


def is_main_process() -> bool:
    return distributed_rank() == 0


def initialize_device(requested: str, timeout_seconds: float) -> torch.device:
    timeout = float(timeout_seconds)
    if not math.isfinite(timeout) or timeout <= 0.0:
        raise ValueError("--distributed-timeout-seconds must be finite and positive.")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed frozen-Qwen evaluation requires CUDA and NCCL.")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank < 0 or local_rank >= torch.cuda.device_count():
            raise ValueError(
                f"LOCAL_RANK={local_rank} is invalid for {torch.cuda.device_count()} visible devices."
            )
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(seconds=timeout),
        )
        return torch.device("cuda", local_rank)
    normalized = str(requested).strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def distributed_barrier() -> None:
    if distributed_is_initialized():
        dist.barrier()


def run_on_rank_zero_and_broadcast(operation, stage: str) -> Any:
    if not distributed_is_initialized():
        return operation()
    original_error: BaseException | None = None
    envelope: dict[str, Any] | None = None
    if is_main_process():
        try:
            envelope = {"ok": True, "value": operation()}
        except BaseException as exc:
            original_error = exc
            envelope = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:4000],
            }
    payload = [envelope]
    dist.broadcast_object_list(payload, src=0)
    received = payload[0]
    if not isinstance(received, Mapping) or not bool(received.get("ok", False)):
        error_type = (
            str(received.get("error_type", "RuntimeError"))
            if isinstance(received, Mapping)
            else "RuntimeError"
        )
        error_message = (
            str(received.get("error_message", "rank 0 returned an invalid result"))
            if isinstance(received, Mapping)
            else "rank 0 returned an invalid result"
        )
        if original_error is not None:
            raise original_error
        raise RuntimeError(f"Rank-0 {stage} failed with {error_type}: {error_message}")
    return received.get("value")


def local_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_dump_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def sha256_file(path: str | Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def create_run_dir(output_root: str | Path, run_name: str) -> Path:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    candidate = root / f"{timestamp}_{run_name}"
    suffix = 1
    while candidate.exists():
        candidate = root / f"{timestamp}_{run_name}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def build_distributed_run_dir(output_root: str | Path, run_name: str) -> Path:
    path = run_on_rank_zero_and_broadcast(
        lambda: str(create_run_dir(output_root, run_name)),
        "run directory creation",
    )
    if not isinstance(path, str) or not path:
        raise RuntimeError("Rank 0 broadcast an invalid output directory.")
    return Path(path)


def parse_splits(raw: str) -> list[str]:
    splits = [part.strip() for part in str(raw).split(",") if part.strip()]
    if not splits:
        raise ValueError("--splits must contain at least one split name.")
    if len(set(splits)) != len(splits):
        raise ValueError(f"--splits contains duplicates: {splits}.")
    invalid = [split for split in splits if not split.replace("_", "").isalnum()]
    if invalid:
        raise ValueError(f"Invalid split names: {invalid}.")
    return splits


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a completely frozen Qwen on matched patch QA with the full 16x16 "
            "standardized tensor matrix serialized as text. No tensor adapter or adapter "
            "checkpoint is constructed or loaded."
        )
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--qa-dir", type=str, default=None)
    parser.add_argument("--latent-dir", type=str, default=None)
    parser.add_argument("--model-name-or-path", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--hf-home", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--splits", type=str, default=None, help="Comma-separated QA splits.")
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Deterministic prefix cap per split for smoke tests; omit for formal full-split evaluation.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Per-rank evaluation batch size.")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-prompt-tokens", type=int, default=None)
    parser.add_argument("--matrix-significant-digits", type=int, default=None)
    parser.add_argument("--matrix-cache-size", type=int, default=None)
    parser.add_argument("--prompt-template", choices=("task_specific",), default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--torch-dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default=None,
    )
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--low-cpu-mem-usage", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--serialize-llm-loading", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--min-host-memory-available-gib", type=float, default=None)
    parser.add_argument("--distributed-timeout-seconds", type=float, default=None)
    parser.add_argument("--require-formal-contract", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--console-progress", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return apply_config_defaults(parser.parse_args(argv))


def apply_config_defaults(args: argparse.Namespace) -> argparse.Namespace:
    config = load_yaml_mapping(args.config)
    model_local_dir = first_nested(config, ["model.local_dir"])
    model_name = first_nested(config, ["model.name_or_path", "model.model_name_or_path"])
    if args.model_name_or_path is None:
        args.model_name_or_path = (
            resolve_path_string(model_local_dir, PROJECT_ROOT) if model_local_dir else model_name
        )

    path_defaults = {
        "qa_dir": first_nested(config, ["patch_qa.matched_qa_dir", "patch_qa.stage2b_qa_dir"]),
        "latent_dir": first_nested(config, ["patch_qa.latent_dir"]),
        "cache_dir": first_nested(config, ["model.cache_dir", "storage.hf_home"]),
        "hf_home": first_nested(config, ["storage.hf_home"]),
        "output_root": first_nested(config, ["llm_training.output_root", "storage.output_root"]),
    }
    for attr, value in path_defaults.items():
        if getattr(args, attr) is None and value is not None:
            setattr(args, attr, resolve_path_string(value, PROJECT_ROOT))

    val_split = str(first_nested(config, ["llm_training.val_split"], "val"))
    test_split = str(first_nested(config, ["llm_training.test_split"], "test"))
    set_default(args, "run_name", None, BASELINE_NAME)
    set_default(args, "splits", None, f"{val_split},{test_split}")
    set_default(
        args,
        "batch_size",
        first_nested(config, ["qwen_tensor_baseline.batch_size"]),
        4,
    )
    set_default(
        args,
        "num_workers",
        first_nested(config, ["qwen_tensor_baseline.num_workers", "llm_training.num_workers"]),
        2,
    )
    set_default(
        args,
        "max_prompt_tokens",
        first_nested(config, ["qwen_tensor_baseline.max_prompt_tokens"]),
        8192,
    )
    set_default(
        args,
        "matrix_significant_digits",
        first_nested(config, ["qwen_tensor_baseline.matrix_significant_digits"]),
        6,
    )
    set_default(
        args,
        "matrix_cache_size",
        first_nested(config, ["qwen_tensor_baseline.matrix_cache_size"]),
        2048,
    )
    set_default(
        args,
        "prompt_template",
        first_nested(config, ["llm_training.prompt_template"]),
        "task_specific",
    )
    set_default(args, "device", first_nested(config, ["llm_training.device", "runtime.device"]), "auto")
    set_default(
        args,
        "torch_dtype",
        first_nested(config, ["llm_training.torch_dtype", "model.torch_dtype"]),
        "bfloat16",
    )
    set_default(args, "trust_remote_code", first_nested(config, ["model.trust_remote_code"]), False)
    set_default(
        args,
        "low_cpu_mem_usage",
        first_nested(config, ["llm_training.low_cpu_mem_usage", "model.low_cpu_mem_usage"]),
        True,
    )
    set_default(
        args,
        "serialize_llm_loading",
        first_nested(config, ["llm_training.serialize_llm_loading"]),
        True,
    )
    set_default(
        args,
        "min_host_memory_available_gib",
        first_nested(config, ["llm_training.min_host_memory_available_gib"]),
        0.0,
    )
    set_default(
        args,
        "distributed_timeout_seconds",
        first_nested(config, ["llm_training.distributed_timeout_seconds"]),
        1800.0,
    )
    set_default(args, "require_formal_contract", None, True)
    set_default(
        args,
        "console_progress",
        first_nested(config, ["qwen_tensor_baseline.console_progress", "llm_training.console_progress"]),
        False,
    )
    set_default(args, "seed", first_nested(config, ["runtime.seed", "llm_training.shuffle_seed"]), 42)

    missing = [
        name
        for name in ("qa_dir", "latent_dir", "model_name_or_path", "output_root")
        if getattr(args, name, None) in {None, ""}
    ]
    if missing:
        flags = ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        raise ValueError(f"Missing required argument(s): {flags}.")
    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be positive.")
    if int(args.num_workers) < 0:
        raise ValueError("--num-workers must be non-negative.")
    if int(args.max_prompt_tokens) <= 0:
        raise ValueError("--max-prompt-tokens must be positive.")
    if int(args.matrix_significant_digits) < 5 or int(args.matrix_significant_digits) > 12:
        raise ValueError("--matrix-significant-digits must be between 5 and 12.")
    if int(args.matrix_cache_size) < 0:
        raise ValueError("--matrix-cache-size must be non-negative.")
    if args.max_records is not None and int(args.max_records) <= 0:
        raise ValueError("--max-records must be positive when provided.")
    if float(args.min_host_memory_available_gib) < 0.0:
        raise ValueError("--min-host-memory-available-gib must be non-negative.")
    args.splits = ",".join(parse_splits(args.splits))
    if str(args.prompt_template) != "task_specific":
        raise ValueError("Formal frozen-Qwen baseline requires prompt_template=task_specific.")
    return args


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


def audit_latent_contract(
    metadata: Mapping[str, Any],
    latent_dir: str | Path,
    require_formal_contract: bool,
) -> dict[str, Any]:
    configured_dir = Path(latent_dir).expanduser()
    if not configured_dir.is_dir():
        raise FileNotFoundError(f"Serialized-tensor baseline latent directory not found: {configured_dir}")

    nested_contract = metadata.get("latent_contract")
    nested_contract = nested_contract if isinstance(nested_contract, Mapping) else {}
    # Current QA builders persist this contract as top-level metadata fields.
    # Accept a future nested copy only as a fallback, never as the primary schema.
    raw_contract = {
        "format": metadata.get("latent_format", nested_contract.get("format")),
        "latent_audit_format": metadata.get(
            "latent_audit_format", nested_contract.get("latent_audit_format")
        ),
        "latent_shape": metadata.get("latent_shape", nested_contract.get("latent_shape")),
        "storage_dtype": metadata.get("storage_dtype", nested_contract.get("storage_dtype")),
        "encoder_input_normalization": metadata.get(
            "encoder_input_normalization",
            nested_contract.get("encoder_input_normalization"),
        ),
        "alignment_checkpoint": metadata.get(
            "alignment_checkpoint", nested_contract.get("alignment_checkpoint")
        ),
        "alignment_checkpoint_sha256": metadata.get(
            "alignment_checkpoint_sha256",
            nested_contract.get("alignment_checkpoint_sha256"),
        ),
    }

    shape_value = raw_contract.get("latent_shape")
    shape = (
        [int(value) for value in shape_value]
        if isinstance(shape_value, Sequence) and not isinstance(shape_value, (str, bytes))
        else []
    )
    normalization_value = raw_contract.get("encoder_input_normalization")
    normalization = (
        canonical_normalization(normalization_value)
        if isinstance(normalization_value, Mapping)
        else {}
    )
    observed = {
        "format": str(raw_contract.get("format", "")),
        "latent_audit_format": str(raw_contract.get("latent_audit_format", "")),
        "latent_shape": shape,
        "storage_dtype": str(raw_contract.get("storage_dtype", "")),
        "encoder_input_normalization": normalization,
        "alignment_checkpoint": str(raw_contract.get("alignment_checkpoint", "")),
        "alignment_checkpoint_sha256": str(
            raw_contract.get("alignment_checkpoint_sha256", "")
        ).lower(),
    }
    expected_normalization = canonical_normalization({"mode": "zscore", "scope": "channel"})
    mismatches: dict[str, Any] = {}
    if observed["format"] != PATCH_LATENT_FORMAT:
        mismatches["format"] = {
            "expected": PATCH_LATENT_FORMAT,
            "observed": observed["format"],
        }
    if observed["latent_audit_format"] != PATCH_LATENT_AUDIT_FORMAT:
        mismatches["latent_audit_format"] = {
            "expected": PATCH_LATENT_AUDIT_FORMAT,
            "observed": observed["latent_audit_format"],
        }
    patch_size = int(metadata.get("patch_size", -1))
    if patch_size != EXPECTED_PATCH_SIZE:
        mismatches["patch_size"] = {
            "expected": EXPECTED_PATCH_SIZE,
            "observed": patch_size,
        }
    if len(shape) != 3 or shape[0] <= PRESERVED_Z_CHANNEL or shape[1:] != [patch_size, patch_size]:
        mismatches["latent_shape"] = {
            "expected": [">=1 channel", patch_size, patch_size],
            "observed": shape,
        }
    if observed["storage_dtype"] != "float16":
        mismatches["storage_dtype"] = {
            "expected": "float16",
            "observed": observed["storage_dtype"],
        }
    if normalization != expected_normalization:
        mismatches["encoder_input_normalization"] = {
            "expected": expected_normalization,
            "observed": normalization,
        }
    if not observed["alignment_checkpoint"]:
        mismatches["alignment_checkpoint"] = {"expected": "non-empty", "observed": ""}
    checkpoint_sha = str(observed["alignment_checkpoint_sha256"])
    if len(checkpoint_sha) != 64 or any(char not in "0123456789abcdef" for char in checkpoint_sha):
        mismatches["alignment_checkpoint_sha256"] = {
            "expected": "64 lowercase hexadecimal characters",
            "observed": checkpoint_sha,
        }

    declared_dir_value = metadata.get("latent_dir")
    declared_dir = Path(str(declared_dir_value)).expanduser() if declared_dir_value else None
    paths_match = bool(
        declared_dir is not None
        and configured_dir.resolve() == declared_dir.resolve()
    )
    if require_formal_contract and not paths_match:
        mismatches["latent_dir"] = {
            "expected": str(declared_dir) if declared_dir is not None else "metadata.latent_dir",
            "observed": str(configured_dir),
        }

    stage2b = metadata.get("stage2b")
    target_provenance = (
        stage2b.get("target_provenance") if isinstance(stage2b, Mapping) else None
    )
    preserved_source = (
        str(target_provenance.get("train_numeric_and_compare", ""))
        if isinstance(target_provenance, Mapping)
        else ""
    )
    if require_formal_contract and preserved_source != "preserved_input_channel_0_as_stored_float16":
        mismatches["preserved_value_source"] = {
            "expected": "preserved_input_channel_0_as_stored_float16",
            "observed": preserved_source,
        }
    if mismatches:
        raise ValueError(f"Serialized-tensor latent contract mismatch: {mismatches}")

    return {
        "available": True,
        **observed,
        "configured_latent_dir": str(configured_dir.resolve()),
        "declared_latent_dir": str(declared_dir.resolve()) if declared_dir is not None else None,
        "configured_matches_metadata": paths_match,
        "preserved_value_channel": PRESERVED_Z_CHANNEL,
        "preserved_value_source": preserved_source,
        "payload_validation_required": True,
        "stage1_checkpoint_opened": False,
    }


def audit_qa_metadata(
    qa_dir: str | Path,
    latent_dir: str | Path,
    splits: Sequence[str],
    require_formal_contract: bool,
) -> dict[str, Any]:
    root = Path(qa_dir)
    build_marker = root / PATCH_QA_BUILD_MARKER
    if build_marker.exists():
        raise RuntimeError(f"Matched QA is marked as incomplete or active: {build_marker}")
    metadata_path = root / "metadata.json"
    if not metadata_path.exists():
        if require_formal_contract:
            raise FileNotFoundError(f"Formal baseline requires QA metadata: {metadata_path}")
        metadata: Mapping[str, Any] = {}
    else:
        with metadata_path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, Mapping):
            raise ValueError(f"Expected a JSON object in {metadata_path}.")
        metadata = loaded

    observed = {
        "format": str(metadata.get("format", "")),
        "matched_group_format": str(metadata.get("matched_group_format", "")),
        "prompt_contract": str(metadata.get("prompt_contract", "")),
        "natural_language_coordinate_origin": int(
            metadata.get("natural_language_coordinate_origin", -1)
        ),
        "split_mode": str(metadata.get("split_mode", "")),
        "requires_explicit_group_sampler": bool(
            metadata.get("requires_explicit_group_sampler", False)
        ),
    }
    if require_formal_contract:
        expected = {
            "format": PATCH_MATCHED_QA_FORMAT,
            "matched_group_format": MATCHED_GROUP_FORMAT,
            "prompt_contract": PATCH_QA_PROMPT_CONTRACT,
            "natural_language_coordinate_origin": 1,
            "split_mode": "sample",
            "requires_explicit_group_sampler": True,
        }
        mismatches = {
            key: {"expected": value, "observed": observed[key]}
            for key, value in expected.items()
            if observed[key] != value
        }
        if mismatches:
            raise ValueError(f"Formal matched-QA metadata contract mismatch: {mismatches}")

    declared_hashes = metadata.get("output_split_sha256", {})
    if not isinstance(declared_hashes, Mapping):
        declared_hashes = {}
    split_files: dict[str, Any] = {}
    for split in splits:
        path = qa_path(root, split)
        actual_hash = sha256_file(path)
        declared_hash = str(declared_hashes.get(split, ""))
        if require_formal_contract and not declared_hash:
            raise ValueError(f"Formal metadata does not declare output_split_sha256[{split!r}].")
        if declared_hash and actual_hash != declared_hash:
            raise ValueError(
                f"QA split changed after metadata was written: split={split}, "
                f"declared={declared_hash}, actual={actual_hash}."
            )
        split_files[split] = {
            "path": str(path),
            "sha256": actual_hash,
            "matches_declared_sha256": bool(declared_hash and actual_hash == declared_hash),
        }

    latent_contract = audit_latent_contract(metadata, latent_dir, require_formal_contract)
    summary = metadata.get("summary", {})
    summary_splits = summary.get("splits", {}) if isinstance(summary, Mapping) else {}
    declared_records = {
        split: int(summary_splits.get(split, {}).get("qa_records", -1))
        if isinstance(summary_splits, Mapping)
        and isinstance(summary_splits.get(split), Mapping)
        else -1
        for split in splits
    }
    return {
        "available": bool(metadata_path.exists()),
        "path": str(metadata_path),
        "sha256": sha256_file(metadata_path) if metadata_path.exists() else None,
        **observed,
        "fields": [str(field) for field in metadata.get("fields", [])],
        "declared_records": declared_records,
        "split_files": split_files,
        "formal_contract_required": bool(require_formal_contract),
        "formal_contract_passed": bool(require_formal_contract),
        "stage1_checkpoint_opened": False,
        "latent_contract_evaluated": True,
        "latent_contract": latent_contract,
    }


def load_qa_records(path: str | Path, max_records: int | None = None) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    source_oracle_records = 0
    limit = None if max_records is None else int(max_records)
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if limit is not None and len(records) >= limit:
                break
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}.")
            source_oracle_records += int("oracle" in payload)
            payload.pop("oracle", None)
            records.append(payload)
    return records, source_oracle_records


def validate_preserved_z_matrix(
    values: torch.Tensor,
    qa_stats: Mapping[str, float],
    path: str | Path,
) -> None:
    if values.ndim != 2 or not bool(torch.isfinite(values).all()):
        raise ValueError(
            f"Preserved tensor channel must be one finite 2D matrix, got {tuple(values.shape)} at {path}."
        )
    observed_mean = float(values.float().mean().item())
    observed_std = float(values.float().std(unbiased=False).item())
    raw_std = float(qa_stats["std"])
    scale = float(qa_stats["scale"])
    expected_scale = float((torch.tensor(raw_std, dtype=torch.float32) + 1.0e-6).item())
    if not math.isclose(scale, expected_scale, rel_tol=5.0e-7, abs_tol=5.0e-12):
        raise ValueError(
            f"Latent z-score scale is stale at {path}: raw_std={raw_std}, "
            f"scale={scale}, expected={expected_scale}."
        )
    expected_std = raw_std / scale
    std_tolerance = max(PRESERVED_Z_STD_ATOL, PRESERVED_Z_STD_RTOL * abs(expected_std))
    if raw_std == 0.0 and int(torch.count_nonzero(values).item()) != 0:
        raise ValueError(f"Constant-patch metadata requires an exactly-zero preserved channel: {path}")
    if expected_std > 2.0**-24 and observed_std == 0.0:
        raise ValueError(
            f"Preserved channel is constant despite non-degenerate metadata at {path}: "
            f"expected_std={expected_std}."
        )
    if abs(observed_mean) > PRESERVED_Z_MEAN_ATOL or abs(observed_std - expected_std) > std_tolerance:
        raise ValueError(
            f"Preserved channel no longer matches per-patch z-score metadata at {path}: "
            f"mean={observed_mean}, std={observed_std}, expected_std={expected_std}."
        )


def serialize_standardized_matrix(values: torch.Tensor, significant_digits: int) -> str:
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D standardized matrix, got {tuple(values.shape)}.")
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError("Cannot serialize a matrix containing NaN or infinity.")
    digits = int(significant_digits)
    if digits < 5 or digits > 12:
        raise ValueError("Matrix significant digits must be between 5 and 12.")

    serialized_rows: list[list[str]] = []
    for row in values.detach().cpu():
        serialized_rows.append([f"{float(value):.{digits}g}" for value in row])

    parsed = torch.tensor(
        [[float(value) for value in row] for row in serialized_rows],
        dtype=torch.float16,
    )
    source_fp16 = values.detach().cpu().to(dtype=torch.float16)
    if not torch.equal(parsed, source_fp16):
        mismatch = torch.nonzero(parsed != source_fp16, as_tuple=False)[0].tolist()
        row, col = (int(value) for value in mismatch)
        raise ValueError(
            "Matrix text precision does not round-trip the stored FP16 value at "
            f"row={row + 1}, col={col + 1}: source={float(source_fp16[row, col])}, "
            f"text={serialized_rows[row][col]!r}. Increase --matrix-significant-digits."
        )

    height, width = (int(value) for value in source_fp16.shape)
    lines = [
        f"shape: {height} rows x {width} columns",
        "column indices: " + ", ".join(str(index) for index in range(1, width + 1)),
    ]
    lines.extend(
        f"row {row_index}: [" + ", ".join(row) + "]"
        for row_index, row in enumerate(serialized_rows, start=1)
    )
    return "\n".join(lines)


def matrix_extreme_support(values: torch.Tensor) -> dict[str, dict[str, Any]]:
    if values.ndim != 2:
        raise ValueError(f"Extreme support expects a 2D matrix, got {tuple(values.shape)}.")
    height, width = (int(value) for value in values.shape)

    def quadrant(row: int, col: int) -> str:
        top = int(row) < height / 2.0
        left = int(col) < width / 2.0
        if top and left:
            return "A"
        if top:
            return "B"
        if left:
            return "C"
        return "D"

    output: dict[str, dict[str, Any]] = {}
    for operation, extreme_value in (("maximum", values.max()), ("minimum", values.min())):
        positions = torch.nonzero(values == extreme_value, as_tuple=False).tolist()
        labels = sorted({quadrant(int(row), int(col)) for row, col in positions})
        tie_scope = (
            "unique_cell"
            if len(positions) == 1
            else "within_quadrant_tie"
            if len(labels) == 1
            else "cross_quadrant_tie"
        )
        output[operation] = {
            "acceptable_labels": labels,
            "position_count": len(positions),
            "tie_scope": tie_scope,
        }
    return output


def requested_extreme_operation(record: Mapping[str, Any]) -> str:
    query = str(record.get("query") or record.get("question") or "").casefold()
    operations = {match.group(1).casefold() for match in EXTREME_OPERATION_RE.finditer(query)}
    if len(operations) != 1:
        raise ValueError(
            f"Extreme QA record must declare exactly one of maximum/minimum: {record.get('qa_id')}"
        )
    return next(iter(operations))


class FrozenQwenQADataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        latent_dir: str | Path,
        latent_contract: Mapping[str, Any],
        matrix_significant_digits: int,
        matrix_cache_size: int,
        max_records: int | None = None,
    ) -> None:
        self.path = Path(path)
        self.latent_dir = Path(latent_dir)
        self.latent_contract = dict(latent_contract)
        self.matrix_significant_digits = int(matrix_significant_digits)
        self.matrix_cache_size = max(0, int(matrix_cache_size))
        self.records, self.source_oracle_records = load_qa_records(self.path, max_records)
        if not self.records:
            raise RuntimeError(f"No QA records found in {self.path}.")
        self._matrix_cache: OrderedDict[
            str, tuple[str, dict[str, Any], dict[str, float], dict[str, dict[str, Any]]]
        ] = OrderedDict()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        matrix_text, extreme_support = self.serialized_tensor_for_record(record)
        metric_contract = self.metric_contract_for_record(record, extreme_support)
        return {
            "index": int(index),
            "record": record,
            "matrix_text": matrix_text,
            **metric_contract,
        }

    def latent_path_for_record(self, record: Mapping[str, Any]) -> Path:
        state_ref = str(record.get("state_ref") or "")
        if not state_ref:
            raise ValueError(f"QA record has no state_ref: {record.get('qa_id', '<unknown>')}")
        return self.latent_dir / f"{state_ref}.pt"

    def serialized_tensor_for_record(
        self,
        record: Mapping[str, Any],
    ) -> tuple[str, dict[str, dict[str, Any]]]:
        path = self.latent_path_for_record(record)
        identity = latent_identity_from_record(record)
        qa_stats = latent_qa_stats_from_record(record)
        cache_key = str(path.resolve())
        cached = self._matrix_cache.get(cache_key)
        if cached is not None:
            text, cached_identity, cached_stats, extreme_support = cached
            if cached_identity != identity or cached_stats != qa_stats:
                raise ValueError(
                    f"QA records map incompatible identity/statistics to one tensor file: {path}"
                )
            self._matrix_cache.move_to_end(cache_key)
            return text, extreme_support
        if not path.is_file():
            raise FileNotFoundError(f"Tensor latent file not found: {path}")

        payload = torch.load(path, map_location="cpu", weights_only=True)
        latent = validate_patch_latent_payload(
            payload,
            path=path,
            expected_identity=identity,
            expected_alignment_checkpoint=self.latent_contract["alignment_checkpoint"],
            expected_alignment_sha256=self.latent_contract["alignment_checkpoint_sha256"],
            expected_normalization=self.latent_contract["encoder_input_normalization"],
            expected_shape=self.latent_contract["latent_shape"],
            expected_storage_dtype=self.latent_contract["storage_dtype"],
            expected_qa_stats=qa_stats,
        )
        values = latent[PRESERVED_Z_CHANNEL].contiguous()
        validate_preserved_z_matrix(values, qa_stats, path)
        text = serialize_standardized_matrix(values, self.matrix_significant_digits)
        extreme_support = matrix_extreme_support(values)
        if self.matrix_cache_size > 0:
            self._matrix_cache[cache_key] = (text, identity, qa_stats, extreme_support)
            self._matrix_cache.move_to_end(cache_key)
            while len(self._matrix_cache) > self.matrix_cache_size:
                self._matrix_cache.popitem(last=False)
        return text, extreme_support

    def matrix_text_for_record(self, record: Mapping[str, Any]) -> str:
        return self.serialized_tensor_for_record(record)[0]

    @staticmethod
    def metric_contract_for_record(
        record: Mapping[str, Any],
        extreme_support: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        answer = str(record.get("answer", ""))
        if str(record.get("task_type", "")) != "extreme_quadrant":
            return {"acceptable_answers": [answer], "extreme_tie_scope": "not_extreme"}
        operation = requested_extreme_operation(record)
        support = extreme_support[operation]
        acceptable = [str(label) for label in support["acceptable_labels"]]
        if answer not in acceptable:
            raise ValueError(
                "Stored FP16 matrix does not support the source extreme label for "
                f"{record.get('qa_id')}: answer={answer}, acceptable={acceptable}."
            )
        return {
            "acceptable_answers": acceptable,
            "extreme_tie_scope": str(support["tie_scope"]),
            "extreme_position_count": int(support["position_count"]),
            "extreme_operation": operation,
        }


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


def collate_records(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "indices": [int(item["index"]) for item in items],
        "records": [item["record"] for item in items],
        "matrix_texts": [str(item["matrix_text"]) for item in items],
        "acceptable_answers": [list(item["acceptable_answers"]) for item in items],
        "extreme_tie_scopes": [str(item["extreme_tie_scope"]) for item in items],
    }


def prompt_only_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Expose only fields consumed by the shared prompt contract."""

    return {
        "task_type": str(record.get("task_type", "")),
        "query": record.get("query"),
        "question": record.get("question"),
        "choices": list(record.get("choices", [])),
    }


def adapt_task_prompt_to_matrix(task_prompt: str) -> str:
    """Change only the input-representation wording in the shared task prompt."""

    prefix, separator, task_suffix = str(task_prompt).partition("\n\nQuery:")
    if not separator:
        raise ValueError("The shared task prompt no longer contains its formal Query boundary.")
    prefix = prefix.replace(
        "Tensor soft tokens before this text encode the tensor state.",
        "The standardized matrix z above encodes the tensor state.",
    )
    prefix = prefix.replace("the tensor soft tokens", "the standardized matrix z above")
    prefix = prefix.replace("tensor soft tokens", "standardized matrix z above")
    if "soft token" in prefix.casefold():
        raise ValueError("The matrix baseline prompt retained a soft-token instruction.")
    return prefix + separator + task_suffix


def render_prompt(
    record: Mapping[str, Any],
    prompt_template: str,
    matrix_text: str,
) -> str:
    shared_task_prompt = build_prompt(
        prompt_only_record(record), prompt_template=prompt_template
    )
    task_prompt = adapt_task_prompt_to_matrix(shared_task_prompt)
    return (
        "The complete per-patch standardized tensor matrix z is provided below as ordinary text. "
        "Rows and columns are explicitly labeled with 1-based indices.\n"
        "Standardized matrix z:\n"
        f"{matrix_text}\n\n"
        f"{task_prompt}"
    )


def record_field(record: Mapping[str, Any]) -> str:
    metadata = record.get("metadata")
    metadata_field = metadata.get("field") if isinstance(metadata, Mapping) else None
    return str(record.get("field") or metadata_field or "unknown")


def audit_qa_records(
    datasets: Mapping[str, FrozenQwenQADataset],
    metadata_audit: Mapping[str, Any],
    require_formal_contract: bool,
    complete_splits: bool,
) -> dict[str, Any]:
    split_samples: dict[str, set[int]] = {}
    split_states: dict[str, set[str]] = {}
    split_qa_ids: dict[str, set[str]] = {}
    output: dict[str, Any] = {}
    metadata_fields = set(str(field) for field in metadata_audit.get("fields", []))
    for split, dataset in datasets.items():
        qa_ids: set[str] = set()
        states: set[str] = set()
        samples: set[int] = set()
        tasks: Counter[str] = Counter()
        fields: Counter[str] = Counter()
        answers_by_task: dict[str, Counter[str]] = defaultdict(Counter)
        choices_by_task: dict[str, set[str]] = defaultdict(set)
        candidate_counts: Counter[int] = Counter()
        for record in dataset.records:
            qa_id = str(record.get("qa_id", ""))
            if not qa_id or qa_id in qa_ids:
                raise ValueError(f"Missing or duplicate qa_id in {split}: {qa_id!r}")
            qa_ids.add(qa_id)
            state_ref = str(record.get("state_ref", ""))
            if not state_ref:
                raise ValueError(f"Record {qa_id} has no state_ref.")
            states.add(state_ref)
            if "sample_index" not in record:
                raise ValueError(f"Record {qa_id} has no sample_index.")
            samples.add(int(record["sample_index"]))

            task = str(record.get("task_type", ""))
            field = record_field(record)
            choices = record.get("choices")
            answer = str(record.get("answer", ""))
            if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)):
                raise ValueError(f"Record {qa_id} has invalid choices.")
            labels = [str(choice) for choice in choices]
            if len(labels) not in {2, 4} or len(set(labels)) != len(labels):
                raise ValueError(f"Record {qa_id} must have two or four unique choices, got {labels}.")
            if not set(labels).issubset(ALLOWED_LABELS) or answer not in labels:
                raise ValueError(f"Record {qa_id} has invalid labels/answer: choices={labels}, answer={answer}.")
            query = str(record.get("query") or record.get("question") or "")
            if not query:
                raise ValueError(f"Record {qa_id} has an empty query.")

            audit_matrix = "shape: 1 rows x 1 columns\ncolumn indices: 1\nrow 1: [0]"
            prompt = render_prompt(record, "task_specific", audit_matrix)
            formal_task_prompt = adapt_task_prompt_to_matrix(
                build_prompt(prompt_only_record(record), prompt_template="task_specific")
            )
            if not prompt.endswith(formal_task_prompt):
                raise RuntimeError(
                    f"Tensor serialization changed more than the input representation for {qa_id}."
                )
            mutated = dict(record)
            mutated["answer"] = "__FORBIDDEN_ANSWER_SENTINEL__"
            mutated["oracle"] = "__FORBIDDEN_ORACLE_SENTINEL__"
            mutated["latent_ref"] = "__FORBIDDEN_LATENT_REF_SENTINEL__"
            mutated["grounding_target"] = "__FORBIDDEN_GROUNDING_SENTINEL__"
            mutated["matched_group"] = "__FORBIDDEN_MATCHED_GROUP_SENTINEL__"
            mutated["prompt_data"] = "__FORBIDDEN_PROMPT_DATA_SENTINEL__"
            if render_prompt(mutated, "task_specific", audit_matrix) != prompt:
                raise RuntimeError(f"A non-prompt QA field changed the model prompt for {qa_id}.")

            tasks[task] += 1
            fields[field] += 1
            answers_by_task[task][answer] += 1
            choices_by_task[task].update(labels)
            candidate_counts[len(labels)] += 1

        observed_tasks = set(tasks)
        observed_fields = set(fields)
        if require_formal_contract and complete_splits:
            if observed_tasks != EXPECTED_TASKS:
                raise ValueError(
                    f"Formal split {split} task mismatch: expected={sorted(EXPECTED_TASKS)}, "
                    f"observed={sorted(observed_tasks)}."
                )
            if metadata_fields and observed_fields != metadata_fields:
                raise ValueError(
                    f"Formal split {split} field mismatch: metadata={sorted(metadata_fields)}, "
                    f"observed={sorted(observed_fields)}."
                )
            missing_answers = {
                task: sorted(choices_by_task[task] - set(answers_by_task[task]))
                for task in choices_by_task
                if choices_by_task[task] - set(answers_by_task[task])
            }
            if missing_answers:
                raise ValueError(f"Formal split {split} lacks answer-label coverage: {missing_answers}")
            declared = int(metadata_audit.get("declared_records", {}).get(split, -1))
            if declared >= 0 and len(dataset) != declared:
                raise ValueError(
                    f"Formal split {split} record count differs from metadata: "
                    f"loaded={len(dataset)}, declared={declared}."
                )

        split_samples[split] = samples
        split_states[split] = states
        split_qa_ids[split] = qa_ids
        output[split] = {
            "records": len(dataset),
            "states": len(states),
            "samples": len(samples),
            "by_task": dict(sorted(tasks.items())),
            "by_field": dict(sorted(fields.items())),
            "answers_by_task": {
                task: dict(sorted(counts.items()))
                for task, counts in sorted(answers_by_task.items())
            },
            "candidate_count_distribution": {
                str(count): total for count, total in sorted(candidate_counts.items())
            },
            "oracle_fields_removed": int(dataset.source_oracle_records),
        }

    overlaps: dict[str, Any] = {}
    split_names = list(datasets)
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            key = f"{left}_{right}"
            overlap = {
                "qa_ids": len(split_qa_ids[left] & split_qa_ids[right]),
                "states": len(split_states[left] & split_states[right]),
                "samples": len(split_samples[left] & split_samples[right]),
            }
            overlaps[key] = overlap
            if require_formal_contract and any(overlap.values()):
                raise ValueError(f"Formal evaluation splits overlap: {key}={overlap}")
    return {
        "splits": output,
        "overlaps": overlaps,
        "complete_split_contract_checked": bool(complete_splits),
        "prompt_projection_excludes": [
            "answer",
            "oracle",
            "latent_path",
            "grounding_target",
            "matched_group",
            "prompt_data",
        ],
        "latent_lookup": "latent_dir/state_ref.pt; record latent_ref is ignored and never enters the prompt",
    }


def audit_tensor_inputs(
    datasets: Mapping[str, FrozenQwenQADataset],
) -> dict[str, Any]:
    split_summaries: dict[str, Any] = {}
    combined_digest = hashlib.sha256()
    total_unique_files = 0
    for split, dataset in datasets.items():
        path_to_state: dict[str, str] = {}
        matrix_digests: dict[str, str] = {}
        extreme_tie_scopes: Counter[str] = Counter()
        extreme_position_count = 0
        for record in dataset.records:
            state_ref = str(record.get("state_ref") or "")
            path = str(dataset.latent_path_for_record(record).resolve())
            previous_state = path_to_state.get(path)
            if previous_state is not None and previous_state != state_ref:
                raise ValueError(
                    f"Different state_ref values map to one tensor file: {previous_state}, {state_ref}, {path}"
                )
            path_to_state[path] = state_ref
            matrix_text, extreme_support = dataset.serialized_tensor_for_record(record)
            metric_contract = dataset.metric_contract_for_record(record, extreme_support)
            if str(record.get("task_type", "")) == "extreme_quadrant":
                extreme_tie_scopes[str(metric_contract["extreme_tie_scope"])] += 1
                extreme_position_count += int(metric_contract["extreme_position_count"])
            digest = hashlib.sha256(matrix_text.encode("utf-8")).hexdigest()
            previous_digest = matrix_digests.get(state_ref)
            if previous_digest is not None and previous_digest != digest:
                raise RuntimeError(f"One state_ref produced different serialized matrices: {state_ref}")
            matrix_digests[state_ref] = digest

        for state_ref in sorted(matrix_digests):
            combined_digest.update(
                f"{split}|{state_ref}|{matrix_digests[state_ref]}\n".encode("utf-8")
            )
        total_unique_files += len(path_to_state)
        split_summaries[split] = {
            "records_checked": len(dataset),
            "unique_tensor_files_opened_and_validated": len(path_to_state),
            "unique_states": len(matrix_digests),
            "all_records_resolved_to_tensor": True,
            "all_payload_contracts_valid": True,
            "all_preserved_z_statistics_valid": True,
            "all_serialized_values_round_trip_to_stored_fp16": True,
            "all_source_extreme_labels_supported_by_stored_fp16": True,
            "extreme_tie_scope": dict(sorted(extreme_tie_scopes.items())),
            "extreme_position_count": extreme_position_count,
        }
    return {
        "input_representation": "complete_16x16_standardized_matrix_serialized_as_text",
        "expected_patch_size": EXPECTED_PATCH_SIZE,
        "latent_payload_shape": datasets[next(iter(datasets))].latent_contract["latent_shape"],
        "source_channel": PRESERVED_Z_CHANNEL,
        "source_channel_semantics": "exact preserved per-patch z-score matrix",
        "learned_latent_channels_in_prompt": [],
        "matrix_value_order": "row-major with explicit 1-based row and column labels",
        "matrix_significant_digits": datasets[next(iter(datasets))].matrix_significant_digits,
        "validated_unique_tensor_files": total_unique_files,
        "serialized_input_sha256": combined_digest.hexdigest(),
        "splits": split_summaries,
    }


def audit_prompt_tokenization(
    datasets: Mapping[str, FrozenQwenQADataset],
    tokenizer,
    max_prompt_tokens: int,
    prompt_template: str,
) -> dict[str, Any]:
    limit = int(max_prompt_tokens)
    split_summary: dict[str, Any] = {}
    all_labels: set[str] = set()
    for split, dataset in datasets.items():
        total_tokens = 0
        max_tokens = 0
        max_record: str | None = None
        by_task: dict[str, dict[str, int]] = defaultdict(
            lambda: {"records": 0, "total_tokens": 0, "max_tokens": 0}
        )
        for record in dataset.records:
            matrix_text = dataset.matrix_text_for_record(record)
            prompt = render_prompt(record, prompt_template, matrix_text)
            token_ids = tokenizer(
                prompt,
                add_special_tokens=True,
                truncation=False,
            )["input_ids"]
            token_count = len(token_ids)
            if token_count > limit:
                raise ValueError(
                    f"Prompt {record.get('qa_id')} uses {token_count} tokens, exceeding "
                    f"max_prompt_tokens={limit}; formal evaluation never truncates prompts."
                )
            total_tokens += token_count
            if token_count > max_tokens:
                max_tokens = token_count
                max_record = str(record.get("qa_id", ""))
            task = str(record.get("task_type", "unknown"))
            by_task[task]["records"] += 1
            by_task[task]["total_tokens"] += token_count
            by_task[task]["max_tokens"] = max(by_task[task]["max_tokens"], token_count)
            all_labels.update(str(choice) for choice in record["choices"])
        split_summary[split] = {
            "records": len(dataset),
            "mean_tokens": total_tokens / len(dataset),
            "max_tokens": max_tokens,
            "max_token_record": max_record,
            "truncated_records": 0,
            "by_task": {
                task: {
                    "records": values["records"],
                    "mean_tokens": values["total_tokens"] / max(1, values["records"]),
                    "max_tokens": values["max_tokens"],
                }
                for task, values in sorted(by_task.items())
            },
        }

    label_token_ids: dict[str, int] = {}
    for label in sorted(all_labels):
        encoded = tokenizer(
            " " + label,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
        if len(encoded) != 1:
            raise ValueError(
                f"Formal restricted-label scoring requires one token for label {label!r}; got {encoded}."
            )
        label_token_ids[label] = int(encoded[0])
    if len(set(label_token_ids.values())) != len(label_token_ids):
        raise ValueError(f"Choice labels do not have unique token ids: {label_token_ids}")
    return {
        "prompt_template": prompt_template,
        "max_prompt_tokens": limit,
        "all_prompts_fit": True,
        "tokenizer_padding_side": str(tokenizer.padding_side),
        "label_encoding": "tokenizer(' ' + label, add_special_tokens=False)",
        "label_token_ids": label_token_ids,
        "all_labels_single_token_and_unique": True,
        "splits": split_summary,
    }


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def resolve_model_dtype(raw: str, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if raw == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[raw]


def host_memory_available_gib() -> float | None:
    path = Path("/proc/meminfo")
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                parts = line.partition(":")[2].strip().split()
                return float(parts[0]) / 1024**2 if parts else None
    return None


def enforce_host_memory_floor(minimum_gib: float, stage: str) -> None:
    local = host_memory_available_gib()
    gathered: list[float | None]
    if distributed_is_initialized():
        gathered = [None] * distributed_world_size()
        dist.all_gather_object(gathered, local)
    else:
        gathered = [local]
    available = [float(value) for value in gathered if value is not None]
    if available and float(minimum_gib) > 0.0 and min(available) < float(minimum_gib):
        raise RuntimeError(
            f"Host memory guard stopped {stage}: MemAvailable={min(available):.2f} GiB is below "
            f"the configured floor {float(minimum_gib):.2f} GiB."
        )


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
            raise ValueError("Tokenizer has no pad/eos/unk token for batched evaluation.")
    return tokenizer


def load_frozen_qwen(args: argparse.Namespace, device: torch.device) -> tuple[nn.Module, torch.dtype]:
    dtype = resolve_model_dtype(str(args.torch_dtype), device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        dtype=dtype,
        trust_remote_code=bool(args.trust_remote_code),
        low_cpu_mem_usage=bool(args.low_cpu_mem_usage),
    )
    model.to(device)
    disable_checkpointing = getattr(model, "gradient_checkpointing_disable", None)
    if callable(disable_checkpointing):
        disable_checkpointing()
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model_type = str(getattr(model.config, "model_type", ""))
    if "qwen" not in model_type.casefold():
        raise ValueError(
            f"This baseline is restricted to Qwen causal LMs; loaded model_type={model_type!r}."
        )
    return model, dtype


def load_frozen_qwen_serialized(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, torch.dtype]:
    if not distributed_is_initialized() or not bool(args.serialize_llm_loading):
        enforce_host_memory_floor(float(args.min_host_memory_available_gib), "Qwen loading")
        return load_frozen_qwen(args, device)

    local_model: nn.Module | None = None
    local_dtype: torch.dtype | None = None
    for load_rank in range(distributed_world_size()):
        enforce_host_memory_floor(
            float(args.min_host_memory_available_gib),
            f"Qwen loading for rank {load_rank}",
        )
        local_error: BaseException | None = None
        if distributed_rank() == load_rank:
            try:
                print(
                    f"startup=qwen_load rank={load_rank}/{distributed_world_size() - 1}",
                    flush=True,
                )
                local_model, local_dtype = load_frozen_qwen(args, device)
            except BaseException as exc:
                local_error = exc
        error_payload = [
            None
            if local_error is None
            else f"{type(local_error).__name__}: {str(local_error)[:2000]}"
        ]
        dist.broadcast_object_list(error_payload, src=load_rank)
        if error_payload[0] is not None:
            raise RuntimeError(
                f"Distributed Qwen loading failed on rank {load_rank}: {error_payload[0]}"
            ) from local_error
        distributed_barrier()
    if local_model is None or local_dtype is None:
        raise RuntimeError("The local rank did not construct its frozen Qwen replica.")
    return local_model, local_dtype


def decoder_for_causal_lm(model: nn.Module) -> nn.Module:
    get_decoder = getattr(model, "get_decoder", None)
    decoder = get_decoder() if callable(get_decoder) else None
    if decoder is None or decoder is model:
        prefix = str(getattr(model, "base_model_prefix", ""))
        decoder = getattr(model, prefix, None) if prefix else None
    if decoder is None or decoder is model or not isinstance(decoder, nn.Module):
        raise ValueError("The Qwen causal LM does not expose its decoder backbone.")
    return decoder


def audit_frozen_qwen(model: nn.Module, dtype: torch.dtype) -> dict[str, Any]:
    parameters = list(model.parameters())
    trainable = sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
    training_modules = [name for name, module in model.named_modules() if module.training]
    checkpointing = bool(getattr(model, "is_gradient_checkpointing", False))
    if trainable != 0 or model.training or training_modules or checkpointing:
        raise RuntimeError(
            "Frozen-Qwen contract failed: "
            f"trainable={trainable}, model.training={model.training}, "
            f"training_modules={training_modules[:8]}, gradient_checkpointing={checkpointing}."
        )
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is None:
        raise ValueError("Qwen does not expose output embeddings for restricted-label scoring.")
    decoder_for_causal_lm(model)
    return {
        "class": type(model).__name__,
        "model_type": str(getattr(model.config, "model_type", "")),
        "parameter_count": sum(parameter.numel() for parameter in parameters),
        "trainable_parameter_count": trainable,
        "training_module_count": len(training_modules),
        "eval_mode": not bool(model.training),
        "gradient_checkpointing": checkpointing,
        "config_use_cache": bool(getattr(model.config, "use_cache", False)),
        "execution_dtype": str(dtype).replace("torch.", ""),
    }


def last_nonpadding_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    if attention_mask.ndim != 2:
        raise ValueError(f"Expected [batch,tokens] attention mask, got {tuple(attention_mask.shape)}.")
    valid = attention_mask.to(dtype=torch.bool)
    if int(valid.shape[1]) == 0 or not bool(valid.any(dim=1).all()):
        raise ValueError("Every prompt must contain at least one non-padding token.")
    positions = torch.arange(valid.shape[1], device=valid.device).unsqueeze(0).expand_as(valid)
    return positions.masked_fill(~valid, -1).max(dim=1).values


@torch.inference_mode()
def score_prompt_batch(
    model: nn.Module,
    tokenizer,
    prompts: Sequence[str],
    choices_by_record: Sequence[Sequence[str]],
    label_token_ids: Mapping[str, int],
    device: torch.device,
    max_prompt_tokens: int,
) -> list[dict[str, Any]]:
    if torch.is_grad_enabled():
        raise RuntimeError("Frozen-Qwen scoring unexpectedly enabled autograd.")
    encoded = tokenizer(
        list(prompts),
        padding=True,
        truncation=False,
        return_tensors="pt",
        add_special_tokens=True,
    )
    attention_mask = encoded["attention_mask"]
    prompt_lengths = attention_mask.sum(dim=1)
    if int(prompt_lengths.max().item()) > int(max_prompt_tokens):
        raise ValueError(
            f"A prompt exceeds max_prompt_tokens={int(max_prompt_tokens)}; prompts are never truncated."
        )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = attention_mask.to(device)
    decoder = decoder_for_causal_lm(model)
    outputs = decoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    last_indices = last_nonpadding_indices(attention_mask)
    batch_indices = torch.arange(input_ids.shape[0], device=device)
    last_hidden = outputs.last_hidden_state[batch_indices, last_indices]
    logits = model.get_output_embeddings()(last_hidden).float()

    scored: list[dict[str, Any]] = []
    for row, raw_choices in enumerate(choices_by_record):
        choices = [str(choice) for choice in raw_choices]
        candidate_ids = torch.tensor(
            [int(label_token_ids[choice]) for choice in choices],
            dtype=torch.long,
            device=device,
        )
        candidate_logits = logits[row].index_select(0, candidate_ids)
        probabilities = F.softmax(candidate_logits, dim=0)
        prediction_index = int(torch.argmax(candidate_logits).item())
        scored.append(
            {
                "prediction": choices[prediction_index],
                "probabilities": {
                    choice: float(probabilities[index].item())
                    for index, choice in enumerate(choices)
                },
                "prompt_tokens": int(prompt_lengths[row].item()),
            }
        )
    return scored


def empty_metric_payload() -> dict[str, Any]:
    return {
        "total": 0,
        "correct": 0,
        "tie_aware_correct": 0,
        "restricted_nll_sum": 0.0,
        "target_probability_sum": 0.0,
        "prediction_confidence_sum": 0.0,
        "uniform_chance_sum": 0.0,
        "prompt_token_sum": 0,
        "prompt_token_max": 0,
        "task_total": defaultdict(int),
        "task_correct": defaultdict(int),
        "task_tie_aware_correct": defaultdict(int),
        "field_total": defaultdict(int),
        "field_correct": defaultdict(int),
        "task_field_total": defaultdict(int),
        "task_field_correct": defaultdict(int),
        "answer_label_total": defaultdict(int),
        "prediction_label_total": defaultdict(int),
        "task_prediction_label_total": defaultdict(int),
        "confusion_total": defaultdict(int),
        "candidate_count_total": defaultdict(int),
        "extreme_tie_scope_total": defaultdict(int),
        "extreme_tie_scope_strict_correct": defaultdict(int),
        "extreme_tie_scope_tie_aware_correct": defaultdict(int),
        "indices": [],
    }


def update_metric_payload(
    payload: dict[str, Any],
    record: Mapping[str, Any],
    scored: Mapping[str, Any],
    index: int,
    acceptable_answers: Sequence[str] | None = None,
    extreme_tie_scope: str | None = None,
) -> None:
    answer = str(record["answer"])
    prediction = str(scored["prediction"])
    probabilities = scored["probabilities"]
    if not isinstance(probabilities, Mapping) or answer not in probabilities:
        raise ValueError(f"Scored probabilities omit answer {answer!r}.")
    target_probability = float(probabilities[answer])
    prediction_probability = float(probabilities[prediction])
    hit = int(prediction == answer)
    task = str(record.get("task_type", "unknown"))
    field = record_field(record)
    task_field = f"{task}/{field}"
    choices = [str(choice) for choice in record["choices"]]
    acceptable = (
        {str(label) for label in acceptable_answers}
        if acceptable_answers is not None
        else {answer}
    )
    if not acceptable or answer not in acceptable or not acceptable.issubset(set(choices)):
        raise ValueError(
            f"Invalid tie-aware answer set for {record.get('qa_id')}: "
            f"answer={answer}, acceptable={sorted(acceptable)}, choices={choices}."
        )
    tie_aware_hit = int(prediction in acceptable)

    payload["total"] += 1
    payload["correct"] += hit
    payload["tie_aware_correct"] += tie_aware_hit
    payload["restricted_nll_sum"] += -math.log(max(target_probability, 1.0e-30))
    payload["target_probability_sum"] += target_probability
    payload["prediction_confidence_sum"] += prediction_probability
    payload["uniform_chance_sum"] += 1.0 / len(choices)
    payload["prompt_token_sum"] += int(scored["prompt_tokens"])
    payload["prompt_token_max"] = max(payload["prompt_token_max"], int(scored["prompt_tokens"]))
    payload["task_total"][task] += 1
    payload["task_correct"][task] += hit
    payload["task_tie_aware_correct"][task] += tie_aware_hit
    payload["field_total"][field] += 1
    payload["field_correct"][field] += hit
    payload["task_field_total"][task_field] += 1
    payload["task_field_correct"][task_field] += hit
    payload["answer_label_total"][answer] += 1
    payload["prediction_label_total"][prediction] += 1
    payload["task_prediction_label_total"][f"{task}/{prediction}"] += 1
    payload["confusion_total"][f"{answer}/{prediction}"] += 1
    payload["candidate_count_total"][str(len(choices))] += 1
    if task == "extreme_quadrant":
        scope = "unique_cell" if extreme_tie_scope is None else str(extreme_tie_scope)
        if scope not in {"unique_cell", "within_quadrant_tie", "cross_quadrant_tie"}:
            raise ValueError(f"Invalid FP16 extreme tie scope for {record.get('qa_id')}: {scope}")
        payload["extreme_tie_scope_total"][scope] += 1
        payload["extreme_tie_scope_strict_correct"][scope] += hit
        payload["extreme_tie_scope_tie_aware_correct"][scope] += tie_aware_hit
    payload["indices"].append(int(index))


def serializable_metric_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: dict(value) if isinstance(value, defaultdict) else list(value) if key == "indices" else value
        for key, value in payload.items()
    }


def merge_metric_payloads(
    payloads: Sequence[Mapping[str, Any]],
    expected_total: int,
) -> dict[str, Any]:
    scalar_ints = ("total", "correct", "tie_aware_correct", "prompt_token_sum")
    scalar_floats = (
        "restricted_nll_sum",
        "target_probability_sum",
        "prediction_confidence_sum",
        "uniform_chance_sum",
    )
    map_names = (
        "task_total",
        "task_correct",
        "task_tie_aware_correct",
        "field_total",
        "field_correct",
        "task_field_total",
        "task_field_correct",
        "answer_label_total",
        "prediction_label_total",
        "task_prediction_label_total",
        "confusion_total",
        "candidate_count_total",
        "extreme_tie_scope_total",
        "extreme_tie_scope_strict_correct",
        "extreme_tie_scope_tie_aware_correct",
    )
    merged: dict[str, Any] = {name: 0 for name in scalar_ints}
    merged.update({name: 0.0 for name in scalar_floats})
    merged["prompt_token_max"] = 0
    for name in map_names:
        merged[name] = defaultdict(int)
    indices: list[int] = []
    for payload in payloads:
        for name in scalar_ints:
            merged[name] += int(payload.get(name, 0))
        for name in scalar_floats:
            merged[name] += float(payload.get(name, 0.0))
        merged["prompt_token_max"] = max(
            int(merged["prompt_token_max"]), int(payload.get("prompt_token_max", 0))
        )
        for name in map_names:
            values = payload.get(name, {})
            if not isinstance(values, Mapping):
                raise TypeError(f"Metric payload {name} must be a mapping.")
            for key, value in values.items():
                merged[name][str(key)] += int(value)
        indices.extend(int(value) for value in payload.get("indices", []))

    expected_indices = list(range(int(expected_total)))
    if sorted(indices) != expected_indices:
        duplicates = len(indices) - len(set(indices))
        missing = sorted(set(expected_indices) - set(indices))[:16]
        extras = sorted(set(indices) - set(expected_indices))[:16]
        raise RuntimeError(
            "Distributed evaluation shard audit failed: "
            f"records={len(indices)}, expected={expected_total}, duplicates={duplicates}, "
            f"missing={missing}, extras={extras}."
        )
    if int(merged["total"]) != int(expected_total):
        raise RuntimeError(
            f"Metric total {merged['total']} does not equal dataset size {expected_total}."
        )
    merged["indices_sha256"] = hashlib.sha256(
        ",".join(str(index) for index in sorted(indices)).encode("ascii")
    ).hexdigest()
    merged["world_shard_record_counts"] = [int(payload.get("total", 0)) for payload in payloads]
    return merged


def grouped_accuracy(
    totals: Mapping[str, int],
    correct: Mapping[str, int],
) -> dict[str, dict[str, Any]]:
    return {
        key: {
            "accuracy": int(correct.get(key, 0)) / max(1, int(total)),
            "correct": int(correct.get(key, 0)),
            "total": int(total),
        }
        for key, total in sorted(totals.items())
    }


def nested_distribution(flat: Mapping[str, int]) -> dict[str, dict[str, int]]:
    nested: dict[str, dict[str, int]] = defaultdict(dict)
    for key, value in sorted(flat.items()):
        outer, separator, inner = str(key).rpartition("/")
        if not separator:
            outer, inner = "unknown", str(key)
        nested[outer][inner] = int(value)
    return dict(nested)


def finalize_metrics(merged: Mapping[str, Any]) -> dict[str, Any]:
    total = int(merged["total"])
    by_task = grouped_accuracy(merged["task_total"], merged["task_correct"])
    tie_aware_by_task = grouped_accuracy(
        merged["task_total"], merged["task_tie_aware_correct"]
    )
    for task, metrics in by_task.items():
        tie_metrics = tie_aware_by_task[task]
        metrics["tie_aware_accuracy"] = float(tie_metrics["accuracy"])
        metrics["tie_aware_correct"] = int(tie_metrics["correct"])
    extreme_tie_metrics = {}
    for scope, scope_total in sorted(merged["extreme_tie_scope_total"].items()):
        strict_correct = int(merged["extreme_tie_scope_strict_correct"].get(scope, 0))
        tie_correct = int(merged["extreme_tie_scope_tie_aware_correct"].get(scope, 0))
        extreme_tie_metrics[str(scope)] = {
            "total": int(scope_total),
            "strict_correct": strict_correct,
            "strict_accuracy": strict_correct / max(1, int(scope_total)),
            "tie_aware_correct": tie_correct,
            "tie_aware_accuracy": tie_correct / max(1, int(scope_total)),
        }
    return {
        "accuracy": int(merged["correct"]) / max(1, total),
        "correct": int(merged["correct"]),
        "accuracy_is_primary_strict_source_label_metric": True,
        "tie_aware_accuracy": int(merged["tie_aware_correct"]) / max(1, total),
        "tie_aware_correct": int(merged["tie_aware_correct"]),
        "total": total,
        "macro_task_accuracy": sum(item["accuracy"] for item in by_task.values())
        / max(1, len(by_task)),
        "macro_task_tie_aware_accuracy": sum(
            item["tie_aware_accuracy"] for item in by_task.values()
        )
        / max(1, len(by_task)),
        "uniform_random_expected_accuracy": float(merged["uniform_chance_sum"]) / max(1, total),
        "mean_restricted_nll": float(merged["restricted_nll_sum"]) / max(1, total),
        "mean_target_probability": float(merged["target_probability_sum"]) / max(1, total),
        "mean_prediction_confidence": float(merged["prediction_confidence_sum"]) / max(1, total),
        "prompt_tokens": {
            "mean": int(merged["prompt_token_sum"]) / max(1, total),
            "max": int(merged["prompt_token_max"]),
        },
        "by_task": by_task,
        "by_field": grouped_accuracy(merged["field_total"], merged["field_correct"]),
        "by_task_field": grouped_accuracy(
            merged["task_field_total"], merged["task_field_correct"]
        ),
        "answer_label_distribution": dict(sorted(merged["answer_label_total"].items())),
        "prediction_label_distribution": dict(
            sorted(merged["prediction_label_total"].items())
        ),
        "prediction_label_distribution_by_task": nested_distribution(
            merged["task_prediction_label_total"]
        ),
        "target_prediction_confusion": nested_distribution(merged["confusion_total"]),
        "candidate_count_distribution": dict(sorted(merged["candidate_count_total"].items())),
        "extreme_fp16_tie_metrics": extreme_tie_metrics,
        "distributed_shard_audit": {
            "exact_no_padding_no_repeat": True,
            "indices_sha256": str(merged["indices_sha256"]),
            "records_by_rank": list(merged["world_shard_record_counts"]),
        },
    }


@torch.inference_mode()
def evaluate_split(
    model: nn.Module,
    tokenizer,
    dataset: FrozenQwenQADataset,
    label_token_ids: Mapping[str, int],
    device: torch.device,
    args: argparse.Namespace,
    split: str,
) -> dict[str, Any]:
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Qwen must remain frozen and in eval mode throughout evaluation.")
    sampler = (
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
        batch_size=int(args.batch_size),
        shuffle=False,
        sampler=sampler,
        num_workers=int(args.num_workers),
        persistent_workers=False,
        prefetch_factor=1 if int(args.num_workers) > 0 else None,
        pin_memory=device.type == "cuda",
        collate_fn=collate_records,
    )
    local = empty_metric_payload()
    iterator = tqdm(
        loader,
        desc=f"Frozen Qwen [{split}] rank {distributed_rank()}",
        leave=False,
        disable=not bool(args.console_progress) or not is_main_process(),
    )
    for batch in iterator:
        records = batch["records"]
        prompts = [
            render_prompt(record, str(args.prompt_template), matrix_text)
            for record, matrix_text in zip(records, batch["matrix_texts"], strict=True)
        ]
        choices_by_record = [[str(choice) for choice in record["choices"]] for record in records]
        scored = score_prompt_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            choices_by_record=choices_by_record,
            label_token_ids=label_token_ids,
            device=device,
            max_prompt_tokens=int(args.max_prompt_tokens),
        )
        for index, record, result, acceptable, tie_scope in zip(
            batch["indices"],
            records,
            scored,
            batch["acceptable_answers"],
            batch["extreme_tie_scopes"],
            strict=True,
        ):
            update_metric_payload(
                local,
                record,
                result,
                index,
                acceptable_answers=acceptable,
                extreme_tie_scope=tie_scope,
            )

    local_serializable = serializable_metric_payload(local)
    if distributed_is_initialized():
        gathered: list[Mapping[str, Any] | None] = [None] * distributed_world_size()
        dist.all_gather_object(gathered, local_serializable)
        invalid_ranks = [
            rank for rank, payload in enumerate(gathered) if not isinstance(payload, Mapping)
        ]
        if invalid_ranks:
            raise RuntimeError(
                f"Distributed evaluation returned invalid metric payloads from ranks {invalid_ranks}."
            )
        payloads = [dict(payload) for payload in gathered if isinstance(payload, Mapping)]
        if len(payloads) != distributed_world_size():
            raise RuntimeError(
                "Distributed evaluation did not receive exactly one metric payload per rank."
            )
    else:
        payloads = [local_serializable]
    merged = merge_metric_payloads(payloads, expected_total=len(dataset))
    return finalize_metrics(merged)


def redact_config(value: Any, key: str = "") -> Any:
    lowered = key.casefold()
    if any(term in lowered for term in ("api_key", "password", "secret", "access_token")):
        return "<redacted>" if value is not None and value != "" else value
    if isinstance(value, Mapping):
        return {str(item_key): redact_config(item, str(item_key)) for item_key, item in value.items()}
    if isinstance(value, list):
        return [redact_config(item) for item in value]
    return value


def runtime_contract() -> dict[str, Any]:
    return {
        "baseline_name": BASELINE_NAME,
        "model": "Qwen causal LM only",
        "model_parameters_frozen": True,
        "model_eval_mode": True,
        "autograd_during_forward": False,
        "optimizer_created": False,
        "backward_called": False,
        "adapter_instantiated": False,
        "adapter_checkpoint_loaded": False,
        "stage1_checkpoint_loaded": False,
        "latent_files_opened": True,
        "latent_payload_channel_used": PRESERVED_Z_CHANNEL,
        "tensor_serialized_into_text": True,
        "tensor_input": "complete 16x16 per-patch standardized matrix z",
        "learned_latent_channels_used": [],
        "soft_prefix_tokens": 0,
        "forward_model_inputs": ["input_ids", "attention_mask"],
        "prompt_record_fields": ["task_type", "query", "question", "choices"],
        "prompt_tensor_source": "validated latent_map[0], serialized with no answer-dependent transform",
        "task_prompt_contract": (
            "shared matched-QA prompt with input-representation wording changed before "
            "the Query boundary; query, choices, and output contract remain unchanged"
        ),
        "answer_in_model_prompt_or_forward_inputs": False,
        "answer_used_pre_forward_for_input_integrity_only": (
            "extreme-task source labels are checked against all extrema in the stored FP16 matrix"
        ),
        "answer_used_after_forward_for_metrics": True,
        "scoring": "next-token logits restricted to each record's displayed choice labels",
        "primary_accuracy": "strict match to the original float32-source QA label",
        "secondary_accuracy": (
            "tie-aware only for cross-quadrant extrema made ambiguous by stored-FP16 quantization"
        ),
        "interpretation": (
            "Frozen-Qwen tensor-text baseline on the identical QA records. Qwen receives the complete "
            "standardized value matrix as ordinary text instead of learned adapter soft tokens."
        ),
    }


def print_split_metrics(split: str, metrics: Mapping[str, Any]) -> None:
    task_text = ", ".join(
        f"{task}={float(values['accuracy']):.4f}"
        for task, values in sorted(metrics.get("by_task", {}).items())
    )
    print(
        f"split={split} accuracy={float(metrics['accuracy']):.4f} "
        f"tie_aware={float(metrics['tie_aware_accuracy']):.4f} "
        f"correct={int(metrics['correct'])}/{int(metrics['total'])} tasks[{task_text}]",
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    apply_runtime_environment(args)
    device: torch.device | None = None
    started = time.monotonic()
    try:
        device = initialize_device(str(args.device), float(args.distributed_timeout_seconds))
        seed_everything(int(args.seed) + distributed_rank())
        splits = parse_splits(args.splits)
        run_dir = build_distributed_run_dir(args.output_root, args.run_name)

        raw_config = load_yaml_mapping(args.config)
        if is_main_process():
            atomic_dump_json(
                run_dir / "resolved_run_config.json",
                {
                    "format": RESULT_FORMAT,
                    "created_at": local_timestamp(),
                    "source_config": str(args.config or ""),
                    "source_config_sha256": sha256_file(args.config) if args.config else None,
                    "resolved_args": vars(args),
                    "config_snapshot": redact_config(raw_config),
                },
            )

        metadata_audit = run_on_rank_zero_and_broadcast(
            lambda: audit_qa_metadata(
                args.qa_dir,
                args.latent_dir,
                splits,
                require_formal_contract=bool(args.require_formal_contract),
            ),
            "QA metadata audit",
        )
        datasets = {
            split: FrozenQwenQADataset(
                qa_path(args.qa_dir, split),
                latent_dir=args.latent_dir,
                latent_contract=metadata_audit["latent_contract"],
                matrix_significant_digits=int(args.matrix_significant_digits),
                matrix_cache_size=int(args.matrix_cache_size),
                max_records=args.max_records,
            )
            for split in splits
        }
        qa_audit = run_on_rank_zero_and_broadcast(
            lambda: audit_qa_records(
                datasets,
                metadata_audit=metadata_audit,
                require_formal_contract=bool(args.require_formal_contract),
                complete_splits=args.max_records is None,
            ),
            "QA record audit",
        )
        if is_main_process():
            atomic_dump_json(
                run_dir / "qa_only_audit.json",
                {"metadata": metadata_audit, "records": qa_audit},
            )

        tensor_input_audit = run_on_rank_zero_and_broadcast(
            lambda: audit_tensor_inputs(datasets),
            "serialized tensor input audit",
        )
        if is_main_process():
            atomic_dump_json(run_dir / "tensor_input_audit.json", tensor_input_audit)

        tokenizer = load_tokenizer(args)
        prompt_audit = run_on_rank_zero_and_broadcast(
            lambda: audit_prompt_tokenization(
                datasets,
                tokenizer=tokenizer,
                max_prompt_tokens=int(args.max_prompt_tokens),
                prompt_template=str(args.prompt_template),
            ),
            "prompt/tokenizer audit",
        )
        label_token_ids = {
            str(label): int(token_id)
            for label, token_id in prompt_audit["label_token_ids"].items()
        }

        if is_main_process():
            print(
                f"baseline={BASELINE_NAME} model={args.model_name_or_path} "
                f"splits={','.join(splits)} batch_per_rank={int(args.batch_size)} "
                f"world_size={distributed_world_size()} output={run_dir}",
                flush=True,
            )
        model, dtype = load_frozen_qwen_serialized(args, device)
        model_audit = audit_frozen_qwen(model, dtype)

        split_metrics: dict[str, Any] = {}
        for split in splits:
            distributed_barrier()
            split_started = time.monotonic()
            metrics = evaluate_split(
                model=model,
                tokenizer=tokenizer,
                dataset=datasets[split],
                label_token_ids=label_token_ids,
                device=device,
                args=args,
                split=split,
            )
            metrics["elapsed_seconds"] = time.monotonic() - split_started
            split_metrics[split] = metrics
            if is_main_process():
                print_split_metrics(split, metrics)

        result = {
            "format": RESULT_FORMAT,
            "baseline_name": BASELINE_NAME,
            "completed_at": local_timestamp(),
            "elapsed_seconds": time.monotonic() - started,
            "world_size": distributed_world_size(),
            "device_type": device.type,
            "model_name_or_path": str(args.model_name_or_path),
            "model_audit": model_audit,
            "runtime_contract": runtime_contract(),
            "qa_metadata_audit": metadata_audit,
            "qa_record_audit": qa_audit,
            "tensor_input_audit": tensor_input_audit,
            "prompt_tokenization_audit": prompt_audit,
            "splits": split_metrics,
        }
        if is_main_process():
            atomic_dump_json(run_dir / "frozen_qwen_tensor_results.json", result)
            print(f"results={run_dir / 'frozen_qwen_tensor_results.json'}", flush=True)
        distributed_barrier()
    finally:
        if distributed_is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
