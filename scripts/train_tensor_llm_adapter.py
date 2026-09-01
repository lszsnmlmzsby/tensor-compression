from __future__ import annotations

import argparse
import atexit
import copy
import gc
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import OrderedDict, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler, get_worker_info
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
    sinusoidal_2d_position_encoding,
    synchronize_gradients,
)

from tensor_compression.downstream.patch_qa_contract import (  # noqa: E402
    PATCH_LATENT_AUDIT_FORMAT,
    PATCH_LATENT_FORMAT,
    PATCH_MATCHED_QA_FORMAT,
    PATCH_QA_BUILD_MARKER,
    PATCH_QA_FORMAT,
    PATCH_QA_PROMPT_CONTRACT,
    MATCHED_GROUP_FORMAT,
    canonical_normalization,
    canonical_path,
    latent_identity_from_record,
    latent_qa_stats_from_record,
    sha256_file,
    validate_stage1_alignment_checkpoint_payload,
    validate_patch_latent_payload,
)
from tensor_compression.downstream.patch_qa_prompt import (  # noqa: E402
    build_prompt,
    choice_semantics,
    task_specific_instruction,
    valid_choice_instruction,
)
from tensor_compression.downstream.field_io import resolve_device  # noqa: E402
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
SUPPORTED_BASELINE_MODES = {
    "correct",
    "global_only",
    # Same-length grounded ablation: preserve the two local slots and global
    # prefix, but remove the local evidence content.  This separates evidence
    # gain from the positional/layout cost of adding local slots.
    "zero_local",
    "local_only",
    "no_latent",
    "zero_latent",
    "shuffled",
    "random",
    "shuffled_stats",
}
LATENT_CHANNEL_POLICIES = frozenset({"all", "value_only"})
DIRECT_ALIGNMENT_ARCHITECTURES = frozenset({"alignment_qformer", "alignment_adapter"})
CONTEXTUAL_LOCAL_ARCHITECTURES = frozenset(
    {
        "hybrid_local_qformer",
        "residual_question_qformer",
        "residual_question_adapter",
        "grounded_evidence_adapter",
    }
)


def is_direct_alignment_architecture(value: str) -> bool:
    return str(value) in DIRECT_ALIGNMENT_ARCHITECTURES


def apply_latent_channel_policy(
    latent: torch.Tensor,
    policy: str,
    *,
    source: str | Path | None = None,
) -> torch.Tensor:
    normalized = str(policy)
    if normalized not in LATENT_CHANNEL_POLICIES:
        raise ValueError(
            f"Unsupported latent channel policy {policy!r}; expected one of "
            f"{sorted(LATENT_CHANNEL_POLICIES)}."
        )
    if normalized == "all":
        return latent
    if latent.ndim != 3 or int(latent.shape[0]) < 1:
        suffix = f" from {source}" if source is not None else ""
        raise ValueError(
            "value_only latent policy requires [C,H,W] with at least one channel; "
            f"got {tuple(latent.shape)}{suffix}."
        )
    value_only = torch.zeros_like(latent)
    value_only[0].copy_(latent[0])
    return value_only


def uses_contextual_local_prompt(args: argparse.Namespace) -> bool:
    return (
        str(args.adapter_architecture) in CONTEXTUAL_LOCAL_ARCHITECTURES
        and str(args.local_question_input_mode) == "contextual_tokens"
    )


def model_identifier_leaf(value: str | Path) -> str:
    normalized = str(value).strip().replace("\\", "/").rstrip("/")
    return normalized.rsplit("/", 1)[-1].casefold() if normalized else ""


def validate_stage1_model_identity(
    checkpoint_args: Mapping[str, Any],
    current_model_name_or_path: str | Path,
) -> None:
    checkpoint_model = str(checkpoint_args.get("model_name_or_path", "")).strip()
    current_model = str(current_model_name_or_path).strip()
    if not checkpoint_model:
        raise ValueError(
            "The Stage-1 checkpoint does not record model_name_or_path, so its frozen-LLM identity "
            "cannot be verified for direct Stage 2."
        )
    if not current_model:
        raise ValueError("Stage 2 does not specify model_name_or_path.")
    if model_identifier_leaf(checkpoint_model) != model_identifier_leaf(current_model):
        raise ValueError(
            "Stage-1 and Stage-2 must use the same frozen LLM. "
            f"Checkpoint model={checkpoint_model!r}, current model={current_model!r}."
        )


def validate_stage1_teacher_supervision(
    checkpoint: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Validate the Stage-1 hidden-trajectory contract without loading loss-only transforms."""
    raw_version = checkpoint.get("checkpoint_version", 0)
    try:
        checkpoint_version = int(raw_version)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Stage-1 checkpoint_version must be an integer, got {raw_version!r}.") from exc
    raw_metadata = checkpoint.get("teacher_supervision")
    if raw_metadata is None:
        if checkpoint_version >= 3:
            raise ValueError(
                "Stage-1 checkpoint version 3 or newer is missing teacher_supervision metadata."
            )
        return None
    if not isinstance(raw_metadata, Mapping):
        raise ValueError("Stage-1 teacher_supervision must be a mapping.")

    raw_layers = raw_metadata.get("layers")
    raw_auxiliary_layers = raw_metadata.get("auxiliary_layers")
    if (
        not isinstance(raw_layers, Sequence)
        or isinstance(raw_layers, (str, bytes))
        or not isinstance(raw_auxiliary_layers, Sequence)
        or isinstance(raw_auxiliary_layers, (str, bytes))
    ):
        raise ValueError("Stage-1 teacher layers and auxiliary_layers must be numeric sequences.")
    try:
        primary_layer = int(raw_metadata["primary_layer"])
        layers = [int(layer) for layer in raw_layers]
        auxiliary_layers = [int(layer) for layer in raw_auxiliary_layers]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Stage-1 teacher_supervision must define integer primary_layer, layers, and auxiliary_layers."
        ) from exc
    if primary_layer <= 0 or not layers or any(layer <= 0 for layer in layers):
        raise ValueError("Stage-1 teacher layers must use positive 1-based transformer block indices.")
    if layers != sorted(set(layers)):
        raise ValueError(f"Stage-1 teacher layers must be sorted and unique, got {layers}.")
    if primary_layer not in layers:
        raise ValueError(
            f"Stage-1 primary teacher layer {primary_layer} is absent from layers={layers}."
        )
    expected_auxiliary = sorted(set(layers) - {primary_layer})
    if auxiliary_layers != expected_auxiliary:
        raise ValueError(
            "Stage-1 auxiliary teacher layers disagree with the complete layer list: "
            f"expected={expected_auxiliary}, observed={auxiliary_layers}."
        )

    raw_weights = raw_metadata.get("auxiliary_layer_weights")
    if not isinstance(raw_weights, Mapping):
        raise ValueError("Stage-1 teacher_supervision is missing auxiliary_layer_weights.")
    try:
        auxiliary_weight_pairs = [
            (int(layer), float(weight)) for layer, weight in raw_weights.items()
        ]
    except (TypeError, ValueError) as exc:
        raise ValueError("Stage-1 auxiliary teacher weights must map layer indices to numbers.") from exc
    auxiliary_weights = dict(auxiliary_weight_pairs)
    if len(auxiliary_weights) != len(auxiliary_weight_pairs):
        raise ValueError("Stage-1 auxiliary teacher weights contain duplicate numeric layer keys.")
    if sorted(auxiliary_weights) != expected_auxiliary:
        raise ValueError(
            "Stage-1 auxiliary teacher weights do not cover exactly the auxiliary layers: "
            f"layers={expected_auxiliary}, weights={sorted(auxiliary_weights)}."
        )
    if any(not math.isfinite(weight) or weight <= 0.0 for weight in auxiliary_weights.values()):
        raise ValueError("Stage-1 auxiliary teacher weights must be finite and positive.")

    primary_transform = str(raw_metadata.get("primary_feature_transform", ""))
    if primary_transform not in {"none", "projection", "whitening"}:
        raise ValueError(
            "Stage-1 primary_feature_transform must be none, projection, or whitening; "
            f"got {primary_transform!r}."
        )
    auxiliary_transform = str(raw_metadata.get("auxiliary_feature_transform", ""))
    if auxiliary_transform != "native_centered_and_branch_mean":
        raise ValueError(
            "Stage-1 auxiliary_feature_transform has an unsupported contract: "
            f"{auxiliary_transform!r}."
        )
    return {
        "primary_layer": primary_layer,
        "layers": layers,
        "auxiliary_layers": auxiliary_layers,
        "auxiliary_layer_weights": {
            str(layer): auxiliary_weights[layer] for layer in expected_auxiliary
        },
        "primary_feature_transform": primary_transform,
        "auxiliary_feature_transform": auxiliary_transform,
    }


def validate_stage1_alignment_checkpoint_phase(
    checkpoint: Mapping[str, Any],
    checkpoint_path: str | Path | None = None,
) -> str:
    """Validate a complete Stage-1 checkpoint before Stage-2 initialization.

    The path is part of the validation for legacy checkpoints because old
    files did not record their phase.  Requiring it prevents an old
    patch-AE warmup file from being mistaken for an alignment checkpoint.
    """
    if checkpoint_path is None:
        raise ValueError(
            "Stage-1 checkpoint phase validation requires the checkpoint path so legacy files can be "
            "distinguished from patch-AE warmup files."
        )
    validation = validate_stage1_alignment_checkpoint_payload(
        checkpoint,
        path=checkpoint_path,
    )
    return str(validation["checkpoint_phase"])


def _configured_checkpoint_path(value: Any) -> Path | None:
    raw = str(value or "").strip()
    if raw.lower() in {"", "none", "null", "random"}:
        return None
    return Path(raw).expanduser().resolve()


def validate_direct_alignment_provenance(
    *,
    metadata_checkpoint: Any,
    configured_checkpoint: Any,
    adapter_checkpoint: Any,
    require_metadata_checkpoint: bool,
) -> dict[str, Any]:
    paths = {
        "qa_metadata": _configured_checkpoint_path(metadata_checkpoint),
        "patch_qa_config": _configured_checkpoint_path(configured_checkpoint),
        "adapter_init": _configured_checkpoint_path(adapter_checkpoint),
    }
    if require_metadata_checkpoint and paths["qa_metadata"] is None:
        raise ValueError(
            "Formal direct Stage 2 requires QA metadata to record the Stage-1 alignment checkpoint. "
            "Regenerate the QA assets with the current build_tensor_patch_qa.py."
        )
    if paths["adapter_init"] is None:
        raise ValueError("Direct Stage 2 requires adapter.init_checkpoint from Stage 1.")
    reference_name = "qa_metadata" if paths["qa_metadata"] is not None else "patch_qa_config"
    reference = paths[reference_name]
    if reference is None:
        raise ValueError(
            "Direct Stage 2 cannot establish QA/adapter provenance: neither QA metadata nor "
            "patch_qa.alignment_checkpoint identifies Stage 1."
        )
    mismatches = {
        name: path
        for name, path in paths.items()
        if path is not None and path != reference
    }
    if mismatches:
        rendered = ", ".join(f"{name}={path}" for name, path in paths.items() if path is not None)
        raise ValueError(
            "Direct Stage 2 must use exactly the Stage-1 checkpoint that generated the QA latent cache. "
            + rendered
        )
    return {
        "validated": True,
        "reference": reference_name,
        "checkpoint": str(reference),
        "sources": {name: str(path) if path is not None else None for name, path in paths.items()},
    }


def validate_stage2_warm_start_file(args: argparse.Namespace) -> Path | None:
    """Fail before distributed/model startup when Stage-2B's parent is absent."""
    if str(args.adapter_architecture) != "grounded_evidence_adapter":
        return None
    raw_path = str(getattr(args, "stage2_warm_start_checkpoint", None) or "").strip()
    if raw_path.lower() in {"", "none", "null", "random"}:
        raise ValueError(
            "grounded_evidence_adapter requires adapter.stage2_warm_start_checkpoint "
            "from the completed direct Stage 2 run."
        )
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            "Grounded Stage-2B cannot start because its completed direct Stage-2 parent "
            f"checkpoint is missing: {path}. Migrate adapter_best.pt to this exact path; "
            "do not replace it with the Stage-1 alignment checkpoint or silently train the "
            "grounded architecture from a different initialization."
        )
    if path.stat().st_size <= 0:
        raise ValueError(f"Stage-2 warm-start checkpoint is empty: {path}")
    args.stage2_warm_start_checkpoint = str(path)
    raw_resume = str(getattr(args, "stage2b_resume_checkpoint", None) or "").strip()
    if raw_resume.lower() not in {"", "none", "null", "random"}:
        resume_path = Path(raw_resume).expanduser().resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(
                f"Grounded Stage-2B continuation checkpoint is missing: {resume_path}"
            )
        if resume_path.stat().st_size <= 0:
            raise ValueError(f"Stage-2B continuation checkpoint is empty: {resume_path}")
        args.stage2b_resume_checkpoint = str(resume_path)
    else:
        args.stage2b_resume_checkpoint = None
    return path


def validate_adapter_loss_contract(args: argparse.Namespace) -> None:
    direct_alignment_architecture = is_direct_alignment_architecture(args.adapter_architecture)
    if (
        str(args.adapter_architecture) == "grounded_evidence_adapter"
        and int(getattr(args, "grounding_routing_warmup_epochs", 0)) > 0
        and float(getattr(args, "grounding_gate_loss_weight", 0.0)) <= 0.0
    ):
        raise ValueError(
            "grounded_evidence_adapter with routing warmup requires "
            "llm_training.grounding_gate_loss_weight > 0 so role gates are supervised "
            "before joint answer training."
        )
    if (
        direct_alignment_architecture
        and float(args.ranking_loss_weight) > 0.0
        and str(args.ranking_loss_negative) == "global_only"
    ):
        raise ValueError(
            "Direct alignment adapters have no separate global/local branch, so "
            "ranking_loss_negative=global_only would compare identical soft prompts. Use no_latent or disable ranking."
        )
    if (
        direct_alignment_architecture
        and float(args.ranking_loss_weight) > 0.0
        and str(args.ranking_loss_negative) in {"shuffled", "random"}
    ):
        raise ValueError(
            "Direct alignment ranking cannot use shuffled/random tensors as supervised negatives. "
            "A mismatched tensor can have the same valid answer, while random tensors create an "
            "out-of-distribution shortcut. Use no_latent for task-independent modality grounding."
        )
    if direct_alignment_architecture and float(args.swapped_question_loss_weight) > 0.0:
        raise ValueError(
            "Direct alignment adapters intentionally produce a question-independent tensor prefix. "
            "swapped_question_loss would compare identical same-tensor prefixes and must be disabled."
        )
    if direct_alignment_architecture:
        for setting in ("eval_baselines", "final_eval_baselines"):
            unsupported = sorted(
                set(parse_csv(getattr(args, setting, "")))
                & {"global_only", "zero_local", "local_only"}
            )
            if unsupported:
                raise ValueError(
                    f"Direct alignment adapters have no global/local split; {setting} contains "
                    f"meaningless baselines {unsupported}."
                )


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


def initialize_distributed_device(
    requested_device: str,
    distributed_timeout_seconds: float,
) -> torch.device:
    timeout_seconds = float(distributed_timeout_seconds)
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
        raise ValueError("llm_training.distributed_timeout_seconds must be finite and positive.")
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
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(seconds=timeout_seconds),
    )
    return torch.device("cuda", local_rank)


def distributed_barrier() -> None:
    if distributed_is_initialized():
        dist.barrier()


def run_on_rank_zero_and_broadcast(operation, stage: str) -> Any:
    """Execute rank-0-only work while making failures visible to every rank."""
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
            str(received.get("error_message", "rank 0 returned an invalid status envelope"))
            if isinstance(received, Mapping)
            else "rank 0 returned an invalid status envelope"
        )
        if original_error is not None:
            raise original_error
        raise RuntimeError(f"Rank-0 {stage} failed with {error_type}: {error_message}")
    return received.get("value")


def build_distributed_run_dir(output_root: str | Path, run_name: str) -> Path:
    if not distributed_is_initialized():
        return build_run_dir(output_root, run_name)
    path = run_on_rank_zero_and_broadcast(
        lambda: str(build_run_dir(output_root, run_name)),
        "run directory creation",
    )
    if not isinstance(path, str) or not path:
        raise RuntimeError("Rank 0 broadcast an invalid Stage-2 run directory.")
    return Path(path)


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


def optimizer_parameter_audit(
    optimizer: torch.optim.Optimizer,
    module: nn.Module,
    *,
    allow_frozen_parameters: bool = False,
) -> dict[str, int]:
    """Verify that optimizer membership matches the intended adapter boundary."""
    module_parameters = list(module.parameters())
    module_ids = {id(parameter) for parameter in module_parameters}
    trainable_ids = {
        id(parameter) for parameter in module_parameters if parameter.requires_grad
    }
    optimizer_parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group.get("params", [])
    ]
    optimizer_ids = [id(parameter) for parameter in optimizer_parameters]
    duplicate_count = len(optimizer_ids) - len(set(optimizer_ids))
    missing_count = len(trainable_ids - set(optimizer_ids))
    outside_module_count = len(set(optimizer_ids) - module_ids)
    frozen_count = len((set(optimizer_ids) & module_ids) - trainable_ids)
    extra_count = outside_module_count + (0 if allow_frozen_parameters else frozen_count)
    if duplicate_count or missing_count or extra_count:
        raise RuntimeError(
            "Stage-2 optimizer parameter audit failed: "
            f"duplicates={duplicate_count}, missing_trainable={missing_count}, "
            f"outside_module={outside_module_count}, frozen_in_optimizer={frozen_count}, "
            f"extra={extra_count}."
        )
    return {
        "trainable_parameters": sum(parameter.numel() for parameter in module_parameters if parameter.requires_grad),
        "optimizer_parameters": sum(parameter.numel() for parameter in optimizer_parameters),
        "optimizer_tensor_count": len(optimizer_parameters),
        "optimizer_duplicate_tensor_count": duplicate_count,
        "optimizer_missing_trainable_tensor_count": missing_count,
        "optimizer_extra_tensor_count": extra_count,
        "optimizer_outside_module_tensor_count": outside_module_count,
        "optimizer_frozen_tensor_count": frozen_count,
        "optimizer_allows_frozen_parameters": int(bool(allow_frozen_parameters)),
    }


def assert_finite_gradients(module: nn.Module, context: str) -> None:
    """Fail at the first update containing a non-finite trainable gradient."""
    invalid: list[str] = []
    for name, parameter in module.named_parameters():
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
            invalid.append(str(name))
    if invalid:
        preview = ", ".join(invalid[:8])
        raise FloatingPointError(
            f"Non-finite gradients detected in {context}: {preview}"
            f"{' ...' if len(invalid) > 8 else ''}."
        )


def average_trainable_gradients_by_record_count(
    module: nn.Module,
    local_record_count: int,
    device: torch.device,
) -> int:
    """Synchronize accumulated record-sum gradients and convert them to a global record mean."""
    local_count = int(local_record_count)
    if local_count <= 0:
        raise ValueError("Cannot normalize accumulated gradients with zero local records.")
    average_trainable_gradients(module)
    count_tensor = torch.tensor([float(local_count)], dtype=torch.float64, device=device)
    if distributed_is_initialized():
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
    global_count = int(round(float(count_tensor.item())))
    if global_count <= 0:
        raise RuntimeError("Distributed gradient normalization received zero global records.")
    scale = float(distributed_world_size()) / float(global_count)
    with torch.no_grad():
        for parameter in module.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(scale)
    return global_count


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


def numeric_quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = max(0.0, min(1.0, float(probability))) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


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


def read_host_memory_snapshot(proc_root: str | Path = "/proc") -> dict[str, float]:
    """Read Linux host memory without adding a process-monitoring dependency."""

    root = Path(proc_root)
    meminfo_path = root / "meminfo"
    status_path = root / "self" / "status"
    if not meminfo_path.exists():
        return {}

    meminfo: dict[str, int] = {}
    with meminfo_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            name, separator, raw_value = line.partition(":")
            if not separator:
                continue
            parts = raw_value.strip().split()
            if not parts:
                continue
            try:
                value_kib = int(parts[0])
            except ValueError:
                continue
            meminfo[name] = value_kib

    process_rss_kib = 0
    if status_path.exists():
        with status_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    parts = line.partition(":")[2].strip().split()
                    if parts:
                        process_rss_kib = int(parts[0])
                    break

    divisor = 1024**2
    return {
        "total_gib": float(meminfo.get("MemTotal", 0)) / divisor,
        "available_gib": float(meminfo.get("MemAvailable", meminfo.get("MemFree", 0))) / divisor,
        "process_rss_gib": float(process_rss_kib) / divisor,
    }


def gather_host_memory(device: torch.device) -> list[dict[str, float]]:
    snapshot = read_host_memory_snapshot()
    if not snapshot:
        return []
    local = torch.tensor(
        [snapshot["total_gib"], snapshot["available_gib"], snapshot["process_rss_gib"]],
        dtype=torch.float64,
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
            "total_gib": float(values[0].item()),
            "available_gib": float(values[1].item()),
            "process_rss_gib": float(values[2].item()),
        }
        for rank, values in enumerate(gathered)
    ]


def enforce_host_memory_floor(
    device: torch.device,
    minimum_available_gib: float,
    stage: str,
) -> list[dict[str, float]]:
    reports = gather_host_memory(device)
    available = [item["available_gib"] for item in reports if item["available_gib"] > 0]
    if available and float(minimum_available_gib) > 0 and min(available) < float(minimum_available_gib):
        raise RuntimeError(
            f"Host memory safety guard stopped {stage}: MemAvailable={min(available):.2f} GiB is below "
            f"the configured floor {float(minimum_available_gib):.2f} GiB. Reduce ranks/workers/cache or use "
            "a host with more RAM."
        )
    return reports


def local_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_dump_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    try:
        dump_json(temporary, payload)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def append_jsonl(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(dict(payload), ensure_ascii=True, allow_nan=False)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(serialized + "\n")


def atomic_torch_save(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def compact_diagnostic_tensors(value: Any) -> Any:
    """Halve diagnostic snapshot size without changing model/checkpoint precision."""
    if torch.is_tensor(value):
        return value.to(dtype=torch.float16) if value.is_floating_point() else value
    if isinstance(value, Mapping):
        return {key: compact_diagnostic_tensors(item) for key, item in value.items()}
    if isinstance(value, list):
        return [compact_diagnostic_tensors(item) for item in value]
    if isinstance(value, tuple):
        return tuple(compact_diagnostic_tensors(item) for item in value)
    return value


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
        summary_path = self.run_dir / "run_summary.json"
        if summary_path.exists():
            try:
                with summary_path.open("r", encoding="utf-8") as handle:
                    summary = json.load(handle)
                if not isinstance(summary, dict):
                    raise ValueError("run_summary.json must contain a JSON object")
                summary["timing"] = timing
                atomic_dump_json(summary_path, summary)
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
                # Timing is the authoritative lifecycle record. A truncated
                # summary must never hide the exception that ended the run.
                timing["run_summary_update_error"] = (
                    f"{type(exc).__name__}: {str(exc)[:1000]}"
                )
        self.last_payload = timing
        atomic_dump_json(self.run_dir / "run_timing.json", timing)
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
    """Condition inherited spatial states without discarding their aligned initialization."""

    def __init__(
        self,
        dim: int,
        heads: int,
        dropout: float,
        gate_init: float,
        gate_trainable: bool = True,
        zero_init_output: bool = False,
    ) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.text_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.gate = nn.Parameter(
            torch.tensor(float(gate_init)),
            requires_grad=bool(gate_trainable),
        )
        if bool(zero_init_output):
            nn.init.zeros_(self.attention.out_proj.weight)
            if self.attention.out_proj.bias is not None:
                nn.init.zeros_(self.attention.out_proj.bias)
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
        freeze_backbone: bool = True,
        text_gate_trainable: bool = False,
        residual_gate_trainable: bool = False,
        zero_init_text_attention: bool = True,
    ) -> None:
        super().__init__()
        if str(aligned_adapter.adapter_type) not in {"qformer", "spatial_transformer"}:
            raise ValueError(
                "Residual question conditioning requires a stage-1 qformer or spatial_transformer checkpoint."
            )
        self.backbone = copy.deepcopy(aligned_adapter)
        self.freeze_backbone = bool(freeze_backbone)
        if self.freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)
            self.backbone.eval()
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
                    gate_trainable=bool(text_gate_trainable),
                    zero_init_output=bool(zero_init_text_attention),
                )
                for _ in self.backbone.blocks
            ]
        )
        self.gate = nn.Parameter(
            torch.tensor(float(residual_gate_init)),
            requires_grad=bool(residual_gate_trainable),
        )
        self.text_gate_trainable = bool(text_gate_trainable)
        self.residual_gate_trainable = bool(residual_gate_trainable)
        self.zero_init_text_attention = bool(zero_init_text_attention)
        self.structured_query_conditioning = False
        self.question_input_mode = "contextual_tokens"
        self.fusion_mode = (
            "residual_spatial_transformer"
            if str(aligned_adapter.adapter_type) == "spatial_transformer"
            else "residual_qformer"
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

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


class GroundedEvidenceAdapter(nn.Module):
    """Route language to factorized 2D keys without changing the frozen LLM's 1D RoPE."""

    def __init__(
        self,
        latent_grid: Sequence[int],
        llm_hidden_size: int,
        context_layers: Sequence[int],
        adapter_dim: int,
        adapter_heads: int,
        dropout: float,
        evidence_tokens: int,
        soft_prompt_scale: float,
        gate_bias_init: float,
    ) -> None:
        super().__init__()
        self.latent_grid = tuple(int(value) for value in latent_grid)
        if len(self.latent_grid) != 2 or any(value <= 0 for value in self.latent_grid):
            raise ValueError(f"Grounded evidence requires a positive 2D latent grid, got {self.latent_grid}.")
        if int(adapter_dim) % int(adapter_heads) != 0:
            raise ValueError("adapter_dim must be divisible by adapter_heads for grounded evidence.")
        self.context_layers = tuple(int(value) for value in context_layers)
        if not self.context_layers:
            raise ValueError("Grounded evidence requires at least one contextual Qwen layer.")
        self.soft_prompt_tokens = int(evidence_tokens)
        if self.soft_prompt_tokens <= 0:
            raise ValueError("Grounded evidence requires at least one role/evidence token.")
        self.soft_prompt_scale = float(soft_prompt_scale)
        self.structured_query_conditioning = False
        self.question_input_mode = "contextual_tokens"
        self.fusion_mode = "grounded_role_routing"
        self.requires_aligned_tokens = True

        self.text_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(int(llm_hidden_size)),
                    nn.Linear(int(llm_hidden_size), int(adapter_dim)),
                )
                for _ in self.context_layers
            ]
        )
        self.text_layer_logits = nn.Parameter(torch.zeros(len(self.context_layers)))
        self.role_queries = nn.Parameter(
            torch.randn(1, self.soft_prompt_tokens, int(adapter_dim)) * 0.02
        )
        self.text_block = CrossAttentionBlock(
            dim=int(adapter_dim),
            heads=int(adapter_heads),
            dropout=float(dropout),
        )
        self.query_norm = nn.LayerNorm(int(adapter_dim))
        self.row_key_dim = int(adapter_dim) // 2
        self.col_key_dim = int(adapter_dim) - self.row_key_dim
        self.row_query_projection = nn.Linear(
            int(adapter_dim), self.row_key_dim, bias=False
        )
        self.col_query_projection = nn.Linear(
            int(adapter_dim), self.col_key_dim, bias=False
        )
        self.row_key_norm = nn.LayerNorm(self.row_key_dim)
        self.col_key_norm = nn.LayerNorm(self.col_key_dim)
        self.row_key_projection = nn.Linear(
            self.row_key_dim, self.row_key_dim, bias=False
        )
        self.col_key_projection = nn.Linear(
            self.col_key_dim, self.col_key_dim, bias=False
        )
        nn.init.eye_(self.row_key_projection.weight)
        nn.init.eye_(self.col_key_projection.weight)
        fixed_grid = sinusoidal_2d_position_encoding(
            *self.latent_grid,
            int(adapter_dim),
        ).reshape(self.latent_grid[0], self.latent_grid[1], int(adapter_dim))
        self.register_buffer(
            "fixed_row_keys",
            fixed_grid[:, 0, : self.row_key_dim].contiguous(),
            persistent=True,
        )
        self.register_buffer(
            "fixed_col_keys",
            fixed_grid[0, :, self.row_key_dim :].contiguous(),
            persistent=True,
        )
        self.row_keys = nn.Parameter(
            torch.zeros(self.latent_grid[0], self.row_key_dim)
        )
        self.col_keys = nn.Parameter(
            torch.zeros(self.latent_grid[1], self.col_key_dim)
        )
        self.routing_logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.role_gate = nn.Linear(int(adapter_dim), 1)
        nn.init.zeros_(self.role_gate.weight)
        nn.init.constant_(self.role_gate.bias, float(gate_bias_init))

        bottleneck = max(32, int(adapter_dim))
        self.evidence_down = nn.Linear(int(llm_hidden_size), bottleneck, bias=False)
        self.evidence_up = nn.Linear(bottleneck, int(llm_hidden_size), bias=False)
        nn.init.zeros_(self.evidence_up.weight)

        self.last_routing_logits: torch.Tensor | None = None
        self.last_row_logits: torch.Tensor | None = None
        self.last_col_logits: torch.Tensor | None = None
        self.last_role_gate_logits: torch.Tensor | None = None
        self.last_routing_weights: torch.Tensor | None = None

    def _text_context(
        self,
        question_embeds: torch.Tensor,
    ) -> torch.Tensor:
        if question_embeds.ndim == 3:
            question_embeds = question_embeds.unsqueeze(1)
        if question_embeds.ndim != 4 or int(question_embeds.shape[1]) != len(self.context_layers):
            raise ValueError(
                "Grounded evidence expected question states [batch,layers,tokens,hidden] for layers "
                f"{self.context_layers}, got {tuple(question_embeds.shape)}."
            )
        projected = [
            projection(question_embeds[:, index].detach().to(dtype=projection[1].weight.dtype))
            for index, projection in enumerate(self.text_projections)
        ]
        weights = torch.softmax(self.text_layer_logits.float(), dim=0).to(dtype=projected[0].dtype)
        return sum(weights[index] * value for index, value in enumerate(projected))

    def _axis_keys(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        row_keys = self.fixed_row_keys.to(device=device, dtype=dtype) + self.row_keys.to(
            device=device, dtype=dtype
        )
        col_keys = self.fixed_col_keys.to(device=device, dtype=dtype) + self.col_keys.to(
            device=device, dtype=dtype
        )
        return (
            F.normalize(self.row_key_projection(self.row_key_norm(row_keys)), dim=-1),
            F.normalize(self.col_key_projection(self.col_key_norm(col_keys)), dim=-1),
        )

    def forward_from_aligned(
        self,
        aligned_tokens: torch.Tensor,
        question_embeds: torch.Tensor,
        question_mask: torch.Tensor | None,
        structured_query: torch.Tensor | None,
    ) -> torch.Tensor:
        if structured_query is not None:
            raise ValueError("Grounded evidence does not accept parsed coordinate or task features.")
        expected_tokens = int(self.latent_grid[0] * self.latent_grid[1])
        if aligned_tokens.ndim != 3 or int(aligned_tokens.shape[1]) != expected_tokens:
            raise ValueError(
                f"Grounded evidence expected {expected_tokens} aligned cell tokens, got "
                f"{tuple(aligned_tokens.shape)}."
            )
        text_context = self._text_context(question_embeds)
        key_padding_mask = (
            ~question_mask.to(device=text_context.device, dtype=torch.bool)
            if question_mask is not None
            else None
        )
        role_states = self.text_block(
            self.role_queries.expand(text_context.shape[0], -1, -1),
            text_context,
            key_padding_mask=key_padding_mask,
        )
        normalized_queries = self.query_norm(role_states)
        row_queries = F.normalize(self.row_query_projection(normalized_queries), dim=-1)
        col_queries = F.normalize(self.col_query_projection(normalized_queries), dim=-1)
        row_keys, col_keys = self._axis_keys(
            device=normalized_queries.device,
            dtype=normalized_queries.dtype,
        )
        logit_scale = self.routing_logit_scale.exp().clamp(max=100.0).to(
            dtype=normalized_queries.dtype
        )
        row_logits = logit_scale * torch.einsum("brd,nd->brn", row_queries, row_keys)
        col_logits = logit_scale * torch.einsum("brd,nd->brn", col_queries, col_keys)
        routing_logits = (
            row_logits.unsqueeze(-1) + col_logits.unsqueeze(-2)
        ).flatten(start_dim=-2)
        routing_weights = torch.softmax(routing_logits.float(), dim=-1).to(
            dtype=aligned_tokens.dtype
        )
        selected = torch.einsum("brn,bnh->brh", routing_weights, aligned_tokens)
        residual = self.evidence_up(F.gelu(self.evidence_down(selected)))
        if self.soft_prompt_scale > 0.0:
            residual = torch.tanh(residual) * self.soft_prompt_scale
        gate_logits = self.role_gate(role_states).squeeze(-1)
        gate_probability = torch.sigmoid(gate_logits)
        # A soft magnitude gate is undone by the frozen decoder's RMSNorm.
        hard_gate = (gate_probability >= 0.5).to(dtype=gate_probability.dtype)
        if self.training:
            hard_gate = hard_gate.detach() + gate_probability - gate_probability.detach()
        evidence = hard_gate.to(dtype=selected.dtype).unsqueeze(-1) * (
            selected + residual.to(dtype=selected.dtype)
        )
        self.last_routing_logits = routing_logits
        self.last_row_logits = row_logits
        self.last_col_logits = col_logits
        self.last_role_gate_logits = gate_logits
        self.last_routing_weights = routing_weights
        return evidence

    def forward(
        self,
        latent_map: torch.Tensor,
        question_embeds: torch.Tensor,
        question_mask: torch.Tensor | None,
        structured_query: torch.Tensor | None,
    ) -> torch.Tensor:
        raise RuntimeError(
            "GroundedEvidenceAdapter must consume the frozen aligned spatial tokens through "
            "HybridGlobalLocalAdapter.forward_components()."
        )


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
        # Kept as a runtime contract rather than a learned parameter so old
        # checkpoints remain loadable.  Stage-2B enables it from config.
        self.mask_inactive_local_tokens = False
        self._last_soft_prompt_attention_mask: torch.Tensor | None = None
        self._last_global_prompts: torch.Tensor | None = None
        self._last_local_prompts: torch.Tensor | None = None
        self._last_role_gate_logits: torch.Tensor | None = None
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
        detach_global_for_local: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if question_embeds is None:
            raise ValueError("hybrid_local_qformer requires natural-language question embeddings.")
        # Frozen parameters do not require a no-grad graph. Preserve gradients
        # with respect to an explicitly differentiable latent for mechanism
        # diagnostics while retaining the cheaper training/eval path.
        if (self.freeze_global or bool(detach_global_for_local)) and not latent_map.requires_grad:
            with torch.no_grad():
                global_prompts = self.global_adapter.forward_soft_prompts(latent_map)
        else:
            global_prompts = self.global_adapter.forward_soft_prompts(latent_map)
        # In joint A/B training the global-only view owns every global-adapter
        # gradient.  The hybrid B view may read the same aligned tokens, but it
        # must see them as a fixed interface; otherwise its answer objective can
        # rewrite the global representation underneath the local reader.
        local_aligned_prompts = (
            global_prompts.detach() if bool(detach_global_for_local) else global_prompts
        )
        visible_global_prompts = (
            global_prompts.detach() if bool(detach_global_for_local) else global_prompts
        )
        if bool(getattr(self.local_adapter, "requires_aligned_tokens", False)):
            conditioned_prompts = self.local_adapter.forward_from_aligned(
                local_aligned_prompts,
                question_embeds,
                question_mask,
                structured_query,
            )
        else:
            conditioned_prompts = self.local_adapter(
                latent_map,
                question_embeds,
                question_mask,
                structured_query,
            )
        if self.residual_mode:
            if conditioned_prompts.shape != global_prompts.shape:
                raise ValueError(
                    "Residual conditioned/global prompts must have identical shapes, got "
                    f"{tuple(conditioned_prompts.shape)} and {tuple(global_prompts.shape)}."
                )
            local_prompts = self.local_adapter.gate.to(dtype=conditioned_prompts.dtype) * (
                conditioned_prompts - local_aligned_prompts
            )
            visible_global = (
                torch.zeros_like(visible_global_prompts)
                if self.training and self.drop_global_prompts_for_batch
                else visible_global_prompts
            )
            combined = visible_global + local_prompts
            self._last_global_prompts = visible_global
            self._last_local_prompts = local_prompts
            self._last_role_gate_logits = getattr(
                self.local_adapter, "last_role_gate_logits", None
            )
            return visible_global, local_prompts, combined
        local_prompts = conditioned_prompts
        global_prompts = visible_global_prompts
        if self.training and self.drop_global_prompts_for_batch:
            global_prompts = torch.zeros_like(global_prompts)
        # Keeping local tokens first preserves the relative positions between global tokens and text.
        combined = torch.cat([local_prompts, global_prompts], dim=1)
        self._last_global_prompts = global_prompts
        self._last_local_prompts = local_prompts
        self._last_role_gate_logits = getattr(
            self.local_adapter, "last_role_gate_logits", None
        )
        return global_prompts, local_prompts, combined

    def forward(
        self,
        latent_map: torch.Tensor,
        question_embeds: torch.Tensor | None = None,
        question_mask: torch.Tensor | None = None,
        structured_query: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_components(latent_map, question_embeds, question_mask, structured_query)[2]


def _grounded_local_adapter(adapter: nn.Module) -> GroundedEvidenceAdapter | None:
    if isinstance(adapter, HybridGlobalLocalAdapter) and isinstance(
        adapter.local_adapter, GroundedEvidenceAdapter
    ):
        return adapter.local_adapter
    return None


EVIDENCE_TRANSFORM_PARAMETER_NAMES = frozenset(
    {
        "evidence_down.weight",
        "evidence_up.weight",
    }
)


def configure_evidence_only_training(adapter: nn.Module) -> dict[str, Any]:
    """Freeze the deployed reader except for its shared evidence transform."""

    if not isinstance(adapter, HybridGlobalLocalAdapter):
        raise TypeError("Evidence-only training requires HybridGlobalLocalAdapter.")
    local = _grounded_local_adapter(adapter)
    if local is None:
        raise TypeError("Evidence-only training requires GroundedEvidenceAdapter.")
    adapter.set_global_trainable(False)
    observed_names = {name for name, _parameter in local.named_parameters()}
    missing = sorted(EVIDENCE_TRANSFORM_PARAMETER_NAMES - observed_names)
    if missing:
        raise RuntimeError(
            "Grounded evidence transform parameters are missing: " + ", ".join(missing)
        )
    for name, parameter in local.named_parameters():
        parameter.requires_grad_(name in EVIDENCE_TRANSFORM_PARAMETER_NAMES)
    trainable = sorted(
        f"local_adapter.{name}"
        for name, parameter in local.named_parameters()
        if parameter.requires_grad
    )
    expected = sorted(
        f"local_adapter.{name}" for name in EVIDENCE_TRANSFORM_PARAMETER_NAMES
    )
    if trainable != expected:
        raise RuntimeError(
            "Evidence-only trainable boundary mismatch: "
            f"observed={trainable}, expected={expected}."
        )
    return {
        "mode": "evidence_transform_only",
        "trainable_parameter_names": trainable,
        "trainable_parameters": sum(
            parameter.numel() for parameter in local.parameters() if parameter.requires_grad
        ),
        "frozen_router_parameters": sum(
            parameter.numel() for parameter in local.parameters() if not parameter.requires_grad
        ),
        "frozen_global_parameters": sum(
            parameter.numel() for parameter in adapter.global_adapter.parameters()
        ),
    }


def configure_full_grounded_local_training(adapter: nn.Module) -> dict[str, Any]:
    """Freeze the global interface and train every parameter of the grounded reader."""

    if not isinstance(adapter, HybridGlobalLocalAdapter):
        raise TypeError("Full local-reader training requires HybridGlobalLocalAdapter.")
    local = _grounded_local_adapter(adapter)
    if local is None:
        raise TypeError("Full local-reader training requires GroundedEvidenceAdapter.")
    adapter.set_global_trainable(False)
    for parameter in local.parameters():
        parameter.requires_grad_(True)
    trainable = sorted(
        name for name, parameter in adapter.named_parameters() if parameter.requires_grad
    )
    expected = sorted(f"local_adapter.{name}" for name, _ in local.named_parameters())
    if trainable != expected:
        raise RuntimeError(
            "Full local-reader trainable boundary mismatch: "
            f"observed={trainable}, expected={expected}."
        )
    return {
        "mode": "full_grounded_local_reader",
        "trainable_parameter_names": trainable,
        "trainable_parameters": sum(parameter.numel() for parameter in local.parameters()),
        "frozen_global_parameters": sum(
            parameter.numel() for parameter in adapter.global_adapter.parameters()
        ),
    }


def audit_evidence_only_optimizer_boundary(
    optimizer: torch.optim.Optimizer,
    adapter: nn.Module,
) -> dict[str, Any]:
    """Fail loudly if a final Stage-2B optimizer can update anything else."""

    if not isinstance(adapter, HybridGlobalLocalAdapter):
        raise TypeError("Evidence-only optimizer audit requires a hybrid adapter.")
    expected = {
        id(parameter)
        for name, parameter in adapter.local_adapter.named_parameters()
        if name in EVIDENCE_TRANSFORM_PARAMETER_NAMES
    }
    observed = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group.get("params", [])
    }
    trainable_names = sorted(
        name for name, parameter in adapter.named_parameters() if parameter.requires_grad
    )
    expected_names = sorted(
        f"local_adapter.{name}" for name in EVIDENCE_TRANSFORM_PARAMETER_NAMES
    )
    if observed != expected or trainable_names != expected_names:
        raise RuntimeError(
            "Final Stage-2B must optimize only the shared evidence transform: "
            f"trainable={trainable_names}, expected={expected_names}, "
            f"optimizer_tensors={len(observed)}."
        )
    return {
        "validated": True,
        "trainable_parameter_names": trainable_names,
        "optimizer_tensor_count": len(observed),
        "optimizer_parameters": sum(
            parameter.numel()
            for group in optimizer.param_groups
            for parameter in group.get("params", [])
        ),
    }


def audit_full_grounded_local_optimizer_boundary(
    optimizer: torch.optim.Optimizer,
    adapter: nn.Module,
) -> dict[str, Any]:
    """Fail if the final optimizer contains anything outside the complete local reader."""

    if not isinstance(adapter, HybridGlobalLocalAdapter):
        raise TypeError("Full local-reader optimizer audit requires a hybrid adapter.")
    local = _grounded_local_adapter(adapter)
    if local is None:
        raise TypeError("Full local-reader optimizer audit requires grounded evidence.")
    expected = {id(parameter) for parameter in local.parameters()}
    observed = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group.get("params", [])
    }
    trainable_names = sorted(
        name for name, parameter in adapter.named_parameters() if parameter.requires_grad
    )
    expected_names = sorted(f"local_adapter.{name}" for name, _ in local.named_parameters())
    if observed != expected or trainable_names != expected_names:
        raise RuntimeError(
            "Final Stage-2B must optimize the complete grounded local reader only: "
            f"trainable={trainable_names}, expected={expected_names}, "
            f"optimizer_tensors={len(observed)}."
        )
    return {
        "validated": True,
        "trainable_parameter_names": trainable_names,
        "optimizer_tensor_count": len(observed),
        "optimizer_parameters": sum(parameter.numel() for parameter in local.parameters()),
    }


def _all_visible_soft_prompt_mask(
    soft_embeds: torch.Tensor,
    *,
    dtype: torch.dtype = torch.long,
) -> torch.Tensor:
    return torch.ones(
        (int(soft_embeds.shape[0]), int(soft_embeds.shape[1])),
        dtype=dtype,
        device=soft_embeds.device,
    )


def grounded_soft_prompt_attention_mask(
    adapter: nn.Module,
    soft_embeds: torch.Tensor,
    mode: str = "correct",
    *,
    dtype: torch.dtype = torch.long,
    gate_logits: torch.Tensor | None = None,
    precomputed: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the visibility mask for a grounded soft-prefix tensor.

    The mask is derived only from the learned question-conditioned role gate;
    no parsed task/coordinate metadata is consulted.  A caller that reuses a
    precomputed prefix must pass its captured mask because subsequent adapter
    forwards overwrite ``last_role_gate_logits``.
    """

    if precomputed is not None:
        mask = precomputed.to(device=soft_embeds.device, dtype=dtype)
        if tuple(mask.shape) != tuple(soft_embeds.shape[:2]):
            raise ValueError(
                "Precomputed soft-prefix attention mask shape does not match embeddings: "
                f"mask={tuple(mask.shape)}, embeds={tuple(soft_embeds.shape[:2])}."
            )
        return mask
    local = _grounded_local_adapter(adapter)
    if (
        local is None
        or not bool(getattr(adapter, "mask_inactive_local_tokens", False))
        or str(mode) in {"global_only", "no_latent"}
    ):
        return _all_visible_soft_prompt_mask(soft_embeds, dtype=dtype)
    logits = gate_logits
    if logits is None:
        logits = getattr(adapter, "_last_role_gate_logits", None)
    if logits is None:
        logits = local.last_role_gate_logits
    local_tokens = int(local.soft_prompt_tokens)
    global_tokens = int(adapter.global_adapter.soft_prompt_tokens)
    expected_gate_shape = (int(soft_embeds.shape[0]), local_tokens)
    if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != expected_gate_shape:
        observed = tuple(logits.shape) if isinstance(logits, torch.Tensor) else None
        raise RuntimeError(
            "Grounded inactive-slot masking is enabled, but the learned gate logits are "
            f"missing or stale: expected={expected_gate_shape}, observed={observed}, mode={mode!r}. "
            "Carry the attention mask alongside every precomputed soft prefix."
        )
    local_visible = (logits.to(device=soft_embeds.device) >= 0.0).to(dtype=dtype)
    if str(mode) == "local_only" or int(soft_embeds.shape[1]) == local_tokens:
        return local_visible
    expected_concat = local_tokens + global_tokens
    if int(soft_embeds.shape[1]) != expected_concat:
        raise RuntimeError(
            "Grounded inactive-slot masking received an unexpected soft-prefix length: "
            f"expected {expected_concat}, got {int(soft_embeds.shape[1])}, mode={mode!r}."
        )
    global_visible = torch.ones(
        (int(soft_embeds.shape[0]), global_tokens),
        dtype=dtype,
        device=soft_embeds.device,
    )
    return torch.cat([local_visible, global_visible], dim=1)


def require_precomputed_grounded_attention_mask(
    adapter: nn.Module,
    precomputed_soft_embeds: torch.Tensor | None,
    precomputed_soft_attention_mask: torch.Tensor | None,
) -> None:
    """Reject a reused grounded prefix whose question-conditioned mask was discarded."""

    if (precomputed_soft_embeds is None) != (
        precomputed_soft_attention_mask is None
    ):
        raise RuntimeError(
            "Precomputed soft embeddings and their attention mask must be provided together; "
            "a reused grounded prefix must carry the attention mask captured from the same adapter "
            "forward so current gate logits cannot supply a stale replacement."
        )


class TensorReadoutQADataset(Dataset):
    def __init__(
        self,
        jsonl_path: str | Path,
        latent_dir: str | Path,
        max_records: int | None = None,
        subset_mode: str = "prefix",
        subset_seed: int = 42,
        prefer_record_latent_ref: bool = False,
        shuffle_seed: int = 42,
        latent_cache_size: int = 0,
        latent_contract: Mapping[str, Any] | None = None,
        latent_channel_policy: str = "all",
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.latent_dir = Path(latent_dir)
        self.prefer_record_latent_ref = bool(prefer_record_latent_ref)
        self.subset_mode = str(subset_mode)
        self.subset_seed = int(subset_seed)
        self.records = self._load_records(
            self.jsonl_path,
            max_records=max_records,
            subset_mode=self.subset_mode,
            subset_seed=self.subset_seed,
        )
        if not self.records:
            raise RuntimeError(f"No QA records found in {self.jsonl_path}.")
        self.latent_cache_size = max(0, int(latent_cache_size))
        self.latent_contract = dict(latent_contract) if isinstance(latent_contract, Mapping) else None
        self.latent_channel_policy = str(latent_channel_policy)
        if self.latent_channel_policy not in LATENT_CHANNEL_POLICIES:
            raise ValueError(
                "Unsupported latent channel policy: "
                f"{self.latent_channel_policy!r}; expected one of {sorted(LATENT_CHANNEL_POLICIES)}."
            )
        self._latent_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._latent_path_cache: dict[str, Path] = {}
        self._latent_identity_cache: dict[str, dict[str, Any]] = {}
        self._latent_qa_stats_cache: dict[str, dict[str, float]] = {}
        self._random_different_indices = self._build_random_different_indices(int(shuffle_seed))

    @staticmethod
    def _load_records(
        path: Path,
        max_records: int | None = None,
        subset_mode: str = "prefix",
        subset_seed: int = 42,
    ) -> list[dict[str, Any]]:
        record_limit = None if max_records is None else max(0, int(max_records))
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                # Preserve the historical early-stop behavior for the default
                # prefix mode (including its useful malformed-tail test).
                if (
                    str(subset_mode) == "prefix"
                    and record_limit is not None
                    and len(records) >= record_limit
                ):
                    break
                stripped = line.strip()
                if not stripped:
                    continue
                payload = json.loads(stripped)
                if not isinstance(payload, dict):
                    raise ValueError(f"Expected JSON object at {path}:{line_number}.")
                # Oracle values are generation/debug metadata and never enter the training process.
                payload.pop("oracle", None)
                records.append(payload)
        if str(subset_mode) == "prefix" or record_limit is None:
            return records if record_limit is None or str(subset_mode) != "prefix" else records[:record_limit]
        if str(subset_mode) != "hash_state":
            raise ValueError(f"Unsupported record subset mode: {subset_mode!r}.")
        if record_limit <= 0:
            return []

        # Select complete state groups in a stable hash order, then restore the
        # original JSONL order.  This avoids a prefix cap that can over-sample
        # one field/sample while keeping matched question groups intact.
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, payload in enumerate(records):
            state_ref = str(payload.get("state_ref") or "")
            if not state_ref:
                state_ref = f"__record_{index:012d}"
            grouped[state_ref].append(index)
        digest_order = sorted(
            grouped,
            key=lambda key: hashlib.sha256(
                f"{int(subset_seed)}:{key}".encode("utf-8")
            ).hexdigest(),
        )
        selected: set[int] = set()
        selected_count = 0
        for state_ref in digest_order:
            indices = grouped[state_ref]
            if selected_count + len(indices) > record_limit:
                continue
            selected.update(indices)
            selected_count += len(indices)
            if selected_count == record_limit:
                break
        if not selected and records and record_limit > 0:
            smallest = min(grouped.values(), key=len)
            if len(smallest) <= record_limit:
                selected.update(smallest)
                selected_count = len(smallest)
        return [payload for index, payload in enumerate(records) if index in selected]

    def effective_latent_cache_size(self) -> int:
        """Treat latent_cache_size as a per-rank budget shared by its loader workers."""

        if self.latent_cache_size <= 0:
            return 0
        worker = get_worker_info()
        if worker is None or int(worker.num_workers) <= 1:
            return self.latent_cache_size
        return max(1, math.ceil(self.latent_cache_size / int(worker.num_workers)))

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
        if self.latent_contract is not None:
            self._validate_record_identity_for_path(path, record)
            self._validate_record_qa_stats_for_path(path, record)
        cached = self._latent_cache.get(cache_key)
        if cached is not None:
            self._latent_cache.move_to_end(cache_key)
            return cached
        payload = torch.load(path, map_location="cpu", weights_only=True)
        latent = self._latent_from_payload(payload, path=path, record=record)
        payload = None
        latent = latent.to(dtype=torch.float32)
        latent = apply_latent_channel_policy(
            latent,
            self.latent_channel_policy,
            source=path,
        )
        cache_capacity = self.effective_latent_cache_size()
        if cache_capacity > 0:
            self._latent_cache[cache_key] = latent
            self._latent_cache.move_to_end(cache_key)
            while len(self._latent_cache) > cache_capacity:
                self._latent_cache.popitem(last=False)
        return latent

    def _latent_from_payload(
        self,
        payload: Any,
        *,
        path: Path,
        record: Mapping[str, Any],
    ) -> torch.Tensor:
        if self.latent_contract is not None:
            expected_identity = self._validate_record_identity_for_path(path, record)
            latent = validate_patch_latent_payload(
                payload,
                path=path,
                expected_identity=expected_identity,
                expected_alignment_checkpoint=self.latent_contract["alignment_checkpoint"],
                expected_alignment_sha256=self.latent_contract["alignment_checkpoint_sha256"],
                expected_normalization=self.latent_contract["encoder_input_normalization"],
                expected_shape=self.latent_contract["latent_shape"],
                expected_storage_dtype=self.latent_contract.get("storage_dtype"),
                expected_qa_stats=self._validate_record_qa_stats_for_path(path, record),
            )
        else:
            latent = payload.get("latent_map") if isinstance(payload, Mapping) else payload
            if not isinstance(latent, torch.Tensor):
                raise ValueError(f"Latent cache file does not contain a tensor latent_map: {path}")
            if latent.ndim == 4 and latent.shape[0] == 1:
                latent = latent.squeeze(0)
            if latent.ndim != 3:
                raise ValueError(f"Expected latent_map [C,H,W], got {tuple(latent.shape)} from {path}")
            if not bool(torch.isfinite(latent).all()):
                raise FloatingPointError(f"Latent cache contains NaN or infinity: {path}")
        return latent

    def _validate_record_identity_for_path(
        self,
        path: Path,
        record: Mapping[str, Any],
    ) -> dict[str, Any]:
        expected = latent_identity_from_record(record)
        key = str(path.resolve())
        identity_cache = getattr(self, "_latent_identity_cache", None)
        if identity_cache is None:
            identity_cache = {}
            self._latent_identity_cache = identity_cache
        previous = identity_cache.get(key)
        if previous is not None and previous != expected:
            raise ValueError(
                f"QA records map different patch identities to one latent cache file {path}: "
                f"first={previous}, current={expected}."
            )
        identity_cache[key] = expected
        return expected

    def _validate_record_qa_stats_for_path(
        self,
        path: Path,
        record: Mapping[str, Any],
    ) -> dict[str, float]:
        expected = latent_qa_stats_from_record(record)
        key = str(path.resolve())
        stats_cache = getattr(self, "_latent_qa_stats_cache", None)
        if stats_cache is None:
            stats_cache = {}
            self._latent_qa_stats_cache = stats_cache
        previous = stats_cache.get(key)
        if previous is not None and previous != expected:
            raise ValueError(
                f"QA records map different normalization statistics to one latent cache file {path}: "
                f"first={previous}, current={expected}."
            )
        stats_cache[key] = expected
        return expected

    def validate_latent_file_for_record(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """Validate one unique cache payload without populating the runtime LRU."""
        path = self.latent_path_for_record(record)
        if not path.exists():
            raise FileNotFoundError(f"Latent cache file not found: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        latent = self._latent_from_payload(payload, path=path, record=record)
        result = {
            "path": str(path.resolve()),
            "shape": [int(value) for value in latent.shape],
            "dtype": str(latent.dtype).replace("torch.", ""),
        }
        payload = None
        latent = None
        return result

    def load_shuffled_latent(self, index: int) -> torch.Tensor:
        other_index = self._random_different_indices[int(index)]
        return self.load_latent_for_record(self.records[other_index])

    def shuffled_record_for_index(self, index: int) -> Mapping[str, Any]:
        other_index = self._random_different_indices[int(index)]
        return self.records[other_index]


class StateTaskGroupedBatchSampler(Sampler[list[int]]):
    """Keep legacy state/task chunks or explicit Stage-2B groups atomically batched."""

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
        self.initial_global_batch_count = len(initial_batches)
        self.initial_padding_batch_count = (
            self._length * self.num_replicas - len(initial_batches)
        )
        self.initial_padding_record_count = sum(
            len(initial_batches[index % len(initial_batches)])
            for index in range(self.initial_padding_batch_count)
        ) if initial_batches else 0
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
        explicit_flags = [isinstance(record.get("matched_group"), Mapping) for record in self.dataset.records]
        if any(explicit_flags) and not all(explicit_flags):
            raise ValueError("A dataset cannot mix explicit matched groups with legacy ungrouped records.")

        units: list[list[int]] = []
        if explicit_flags and all(explicit_flags):
            grouped: dict[str, list[tuple[int, int]]] = defaultdict(list)
            declared_sizes: dict[str, int] = {}
            for index, record in enumerate(self.dataset.records):
                spec = record["matched_group"]
                if str(spec.get("format", "")) != MATCHED_GROUP_FORMAT:
                    raise ValueError(
                        f"Record {record.get('qa_id')} has unsupported matched_group format."
                    )
                group_id = str(spec.get("batch_group_id", ""))
                size = int(spec.get("batch_group_size", 0))
                member = int(spec.get("batch_member_index", -1))
                if not group_id or size <= 0 or member < 0 or member >= size:
                    raise ValueError(f"Record {record.get('qa_id')} has an invalid batch group contract.")
                if size > self.batch_size:
                    raise ValueError(
                        f"Explicit group {group_id} has size {size}, exceeding batch_size={self.batch_size}."
                    )
                previous_size = declared_sizes.setdefault(group_id, size)
                if previous_size != size:
                    raise ValueError(f"Explicit group {group_id} declares inconsistent sizes.")
                grouped[group_id].append((member, index))
            for group_id, members in grouped.items():
                size = declared_sizes[group_id]
                observed_members = sorted(member for member, _index in members)
                if observed_members != list(range(size)):
                    raise ValueError(
                        f"Explicit group {group_id} is incomplete or duplicates members: {observed_members}."
                    )
                units.append([index for _member, index in sorted(members)])
        else:
            grouped_legacy: dict[tuple[str, str], list[int]] = defaultdict(list)
            for index, record in enumerate(self.dataset.records):
                key = (str(record.get("state_ref", "")), str(record.get("task_type", "")))
                grouped_legacy[key].append(index)
            for indices in grouped_legacy.values():
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


def validate_atomic_group_batch_size(
    batch_size: int,
    group_size: int,
    *,
    context: str,
) -> int:
    """Validate a tunable outer batch while preserving complete task groups."""
    batch = int(batch_size)
    group = int(group_size)
    if group <= 0:
        raise ValueError(f"{context} atomic group size must be positive, got {group}.")
    if batch < group:
        raise ValueError(
            f"{context} batch_size must be at least one complete atomic group: "
            f"batch_size={batch}, group_size={group}."
        )
    if batch % group != 0:
        raise ValueError(
            f"{context} batch_size must be a multiple of the atomic group size so no group "
            f"is split or silently under-fills a forward: batch_size={batch}, group_size={group}. "
            f"Use one of {group}, {group * 2}, {group * 3}, ..."
        )
    return batch // group


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

_GROUNDING_QUERY_TYPE_BY_TASK = {
    "normalized_point_value": "point",
    "raw_point_value_with_stats": "point",
    "point_compare": "point_pair",
    "region_mean_compare": "region_pair",
    "extreme_quadrant": "none",
}
_GROUNDING_ACTIVE_ROLES_BY_TYPE = {
    "point": 1,
    "point_pair": 2,
    "region_pair": 2,
    "none": 0,
}


def grounding_query_spec_for_record(record: Mapping[str, Any]) -> Mapping[str, Any]:
    task = str(record.get("task_type", ""))
    expected_type = _GROUNDING_QUERY_TYPE_BY_TASK.get(task)
    if expected_type is None:
        raise ValueError(f"Grounded Stage-2B does not support task_type={task!r}.")
    matched = record.get("matched_group")
    query_spec = matched.get("query_spec") if isinstance(matched, Mapping) else None
    if not isinstance(query_spec, Mapping):
        query_spec = record.get("grounding_target")
    if not isinstance(query_spec, Mapping):
        raise ValueError(f"Grounded record {record.get('qa_id')} has no query_spec.")
    observed_type = str(query_spec.get("type", ""))
    if observed_type != expected_type:
        raise ValueError(
            f"Grounded task {task!r} requires query_spec.type={expected_type!r}, "
            f"got {observed_type!r}."
        )
    if int(query_spec.get("coordinate_origin", -1)) != 0:
        raise ValueError("Grounded routing targets must use zero-based coordinates.")
    return query_spec


def audit_matched_groups(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    explicit = [record for record in records if isinstance(record.get("matched_group"), Mapping)]
    if not explicit:
        return {"present": False, "records": 0, "batch_groups": 0, "margin_groups": 0}
    if len(explicit) != len(records):
        raise ValueError("A split cannot mix matched-group and ordinary records.")
    batch_groups: dict[str, list[tuple[int, Mapping[str, Any], Mapping[str, Any]]]] = defaultdict(list)
    margin_groups: dict[str, list[tuple[int, Mapping[str, Any], Mapping[str, Any]]]] = defaultdict(list)
    active_roles = 0
    for record in records:
        spec = record["matched_group"]
        if str(spec.get("format", "")) != MATCHED_GROUP_FORMAT:
            raise ValueError(f"Unsupported matched_group format in {record.get('qa_id')}.")
        query_spec = grounding_query_spec_for_record(record)
        active_roles += _GROUNDING_ACTIVE_ROLES_BY_TYPE[str(query_spec["type"])]
        batch_id = str(spec.get("batch_group_id", ""))
        batch_groups[batch_id].append((int(spec.get("batch_member_index", -1)), record, spec))
        margin_id = str(spec.get("margin_group_id") or "")
        if margin_id:
            margin_groups[margin_id].append(
                (int(spec.get("margin_member_index", -1)), record, spec)
            )

    def validate_members(
        groups: Mapping[str, Sequence[tuple[int, Mapping[str, Any], Mapping[str, Any]]]],
        size_field: str,
    ) -> None:
        for group_id, members in groups.items():
            if not group_id:
                raise ValueError("Matched groups require non-empty identifiers.")
            declared = {int(spec.get(size_field, 0)) for _member, _record, spec in members}
            if len(declared) != 1:
                raise ValueError(f"Matched group {group_id} declares inconsistent sizes: {declared}.")
            size = next(iter(declared))
            observed = sorted(member for member, _record, _spec in members)
            if size <= 0 or observed != list(range(size)):
                raise ValueError(
                    f"Matched group {group_id} is incomplete or duplicated: size={size}, members={observed}."
                )

    validate_members(batch_groups, "batch_group_size")
    validate_members(margin_groups, "margin_group_size")
    for group_id, members in batch_groups.items():
        states = {str(record.get("state_ref", "")) for _member, record, _spec in members}
        declared_sizes = {int(spec.get("batch_group_size", 0)) for _member, _record, spec in members}
        if len(states) != 1 or declared_sizes != {3}:
            raise ValueError(
                f"Atomic batch group {group_id} must contain exactly three records from one state."
            )
    coordinate_groups = 0
    role_swap_groups = 0
    for group_id, members in margin_groups.items():
        ordered_items = sorted(members, key=lambda item: item[0])
        ordered = [record for _member, record, _spec in ordered_items]
        specs = [spec for _member, _record, spec in ordered_items]
        kinds = {str(spec.get("margin_kind", "")) for spec in specs}
        if len(kinds) != 1:
            raise ValueError(f"Margin group {group_id} mixes margin kinds: {kinds}.")
        choices = [tuple(str(value) for value in item.get("choices", ())) for item in ordered]
        if len(set(choices)) != 1:
            raise ValueError(f"Margin group {group_id} must share ordered choices.")
        option_hashes = {str(spec.get("option_set_sha256", "")) for spec in specs}
        coordinate_sets = {str(spec.get("coordinate_set_id", "")) for spec in specs}
        tasks = {str(item.get("task_type", "")) for item in ordered}
        if (
            len(option_hashes) != 1
            or "" in option_hashes
            or len(coordinate_sets) != 1
            or "" in coordinate_sets
            or len(tasks) != 1
        ):
            raise ValueError(
                f"Margin group {group_id} must share task, option-set hash, and coordinate-set hash."
            )
        kind = next(iter(kinds))
        if kind == "coordinate_choice":
            coordinate_groups += 1
            answers = [str(item.get("answer", "")) for item in ordered]
            if len(set(answers)) != len(answers):
                raise ValueError(f"Coordinate group {group_id} must use distinct answers.")
        elif kind == "role_swap":
            role_swap_groups += 1
            if len(ordered) != 2 or {str(item.get("answer", "")) for item in ordered} != {"A", "B"}:
                raise ValueError(f"Role-swap group {group_id} must be one A/B answer pair.")
        else:
            raise ValueError(f"Margin group {group_id} has unsupported kind={kind!r}.")
    return {
        "present": True,
        "records": len(records),
        "batch_groups": len(batch_groups),
        "margin_groups": len(margin_groups),
        "coordinate_groups": coordinate_groups,
        "role_swap_groups": role_swap_groups,
        "active_roles": active_roles,
    }


def audit_qa_datasets(
    datasets: Mapping[str, TensorReadoutQADataset],
    require_disjoint_splits: bool,
    require_complete_split_coverage: bool = True,
) -> dict[str, Any]:
    split_states: dict[str, set[str]] = {}
    split_samples: dict[str, set[int]] = {}
    split_latent_paths: dict[str, set[str]] = {}
    split_tasks: dict[str, set[str]] = {}
    split_fields: dict[str, set[str]] = {}
    summary: dict[str, Any] = {}
    globally_validated_latents: dict[str, dict[str, Any]] = {}
    globally_validated_identities: dict[str, dict[str, Any]] = {}
    globally_validated_qa_stats: dict[str, dict[str, float]] = {}
    globally_seen_latent_paths: set[str] = set()
    latent_audit_started = time.monotonic()
    for split, dataset in datasets.items():
        record_contract_digest = hashlib.sha256()
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
            strict_latent_contract = getattr(dataset, "latent_contract", None) is not None
            if strict_latent_contract:
                record_identity = latent_identity_from_record(record)
                previous_identity = globally_validated_identities.get(resolved_latent_path)
                if previous_identity is not None and previous_identity != record_identity:
                    raise ValueError(
                        "QA audit found different patch identities mapped to one latent file: "
                        f"path={resolved_latent_path}, first={previous_identity}, current={record_identity}."
                    )
                globally_validated_identities[resolved_latent_path] = record_identity
                record_stats = latent_qa_stats_from_record(record)
                previous_stats = globally_validated_qa_stats.get(resolved_latent_path)
                if previous_stats is not None and previous_stats != record_stats:
                    raise ValueError(
                        "QA audit found different normalization statistics mapped to one latent file: "
                        f"path={resolved_latent_path}, first={previous_stats}, current={record_stats}."
                    )
                globally_validated_qa_stats[resolved_latent_path] = record_stats
            if resolved_latent_path not in latent_paths:
                if not latent_path.exists():
                    raise FileNotFoundError(f"QA audit found a missing latent cache file: {latent_path}")
                if strict_latent_contract and resolved_latent_path not in globally_seen_latent_paths:
                    globally_validated_latents[resolved_latent_path] = dataset.validate_latent_file_for_record(record)
                    globally_seen_latent_paths.add(resolved_latent_path)
                    validated_count = len(globally_validated_latents)
                    if validated_count % 1024 == 0:
                        print(
                            f"startup=latent_audit validated={validated_count} "
                            f"elapsed={timedelta(seconds=int(time.monotonic() - latent_audit_started))}",
                            flush=True,
                        )
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
        if missing_answer_labels and require_disjoint_splits and require_complete_split_coverage:
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
            "records": len(dataset.records),
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
            "matched_groups": audit_matched_groups(dataset.records),
            "complete_coverage_checked": bool(require_complete_split_coverage),
            "record_contract_sha256": record_contract_digest.hexdigest(),
        }
    reference_split = "train" if "train" in split_tasks else next(iter(split_tasks))
    if require_complete_split_coverage:
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
    summary["strict_latent_payloads_validated"] = len(globally_validated_latents)
    summary["strict_latent_contract_enabled"] = all(
        getattr(dataset, "latent_contract", None) is not None for dataset in datasets.values()
    )
    summary["evaluation_scope"] = "formal_generalization" if require_disjoint_splits else "sanity_only"
    summary["_audit_scope"] = {
        "disjoint_records_checked": bool(require_disjoint_splits),
        "complete_split_coverage_checked": bool(require_complete_split_coverage),
        "coverage_note": (
            "Full task/field/answer coverage checks were applied."
            if require_complete_split_coverage
            else "Coverage checks were skipped because at least one split was explicitly truncated for a smoke test."
        ),
    }
    return summary


JOINT_RUN_SCOPES = ("screening", "formal")


def apply_joint_run_scope_overrides(
    config: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Resolve the joint small/formal profile before normal config defaults.

    The formal profile lives beside the screening configuration so every model,
    loss, and data-contract setting remains shared.  Only explicitly listed
    ``formal_overrides`` are replaced, including an explicit YAML ``null`` used
    to select the complete training split.
    """

    raw_training = config.get("llm_training")
    training = raw_training if isinstance(raw_training, Mapping) else {}
    requested = str(
        getattr(args, "joint_run_scope", None)
        or training.get("joint_run_scope")
        or "screening"
    )
    if requested not in JOINT_RUN_SCOPES:
        raise ValueError(
            "llm_training.joint_run_scope must be one of "
            f"{', '.join(JOINT_RUN_SCOPES)}, got {requested!r}."
        )
    resolved = copy.deepcopy(dict(config))
    resolved_training = copy.deepcopy(dict(training))
    if requested == "formal":
        overrides = training.get("formal_overrides")
        if not isinstance(overrides, Mapping) or not overrides:
            raise ValueError(
                "joint_run_scope=formal requires a non-empty "
                "llm_training.formal_overrides mapping."
            )
        for name, value in overrides.items():
            resolved_training[str(name)] = copy.deepcopy(value)
    resolved_training["joint_run_scope"] = requested
    resolved["llm_training"] = resolved_training
    args.joint_run_scope = requested
    return resolved


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
    parser.add_argument(
        "--latent-channel-policy",
        choices=tuple(sorted(LATENT_CHANNEL_POLICIES)),
        default=None,
        help=(
            "Runtime view of cached latent channels. 'value_only' preserves channel 0 and zeros "
            "all learned channels without modifying cache files; intended for controlled ablations."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--distributed-timeout-seconds",
        type=float,
        default=None,
        help="Timeout for Stage-2 distributed collectives, including rank-0-only audits.",
    )
    parser.add_argument(
        "--serialize-llm-loading",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Load one distributed LLM replica at a time to bound host-RAM startup peaks.",
    )
    parser.add_argument(
        "--low-cpu-mem-usage",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use Transformers' low-host-memory checkpoint loader.",
    )
    parser.add_argument(
        "--min-host-memory-available-gib",
        type=float,
        default=None,
        help="Abort coherently when Linux MemAvailable falls below this safety floor; 0 disables the guard.",
    )
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
    parser.add_argument("--grounding-routing-warmup-epochs", type=int, default=None)
    parser.add_argument("--grounding-warmup-min-cell-top1", type=float, default=None)
    parser.add_argument("--grounding-warmup-min-cell-top5", type=float, default=None)
    parser.add_argument("--grounding-warmup-min-axis-top1", type=float, default=None)
    parser.add_argument("--grounding-warmup-min-target-mass", type=float, default=None)
    parser.add_argument("--grounding-warmup-min-gate-accuracy", type=float, default=None)
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
    parser.add_argument("--grounding-routing-loss-weight", type=float, default=None)
    parser.add_argument(
        "--grounding-joint-routing-loss-weight",
        type=float,
        default=None,
        help=(
            "Routing-loss weight after the grounded routing-only warmup. "
            "Defaults to grounding-routing-loss-weight for backward compatibility."
        ),
    )
    parser.add_argument("--grounding-gate-loss-weight", type=float, default=None)
    parser.add_argument("--matched-group-loss-weight", type=float, default=None)
    parser.add_argument("--matched-group-loss-margin", type=float, default=None)
    parser.add_argument(
        "--joint-ab-training",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Train global-only A and grounded hybrid B views with disjoint gradient ownership.",
    )
    parser.add_argument(
        "--point-reader-training",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Freeze global/routing parameters and train only the shared grounded evidence "
            "transform with point-causal and non-point preservation references."
        ),
    )
    parser.add_argument(
        "--full-local-reader-training",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Freeze the LLM/global adapter and continue the complete grounded reader with "
            "the validated answer, routing, gate, group, and question-swap objectives."
        ),
    )
    parser.add_argument(
        "--joint-run-scope",
        choices=JOINT_RUN_SCOPES,
        default=None,
        help=(
            "Use the fixed-subset screening profile or the full-data formal profile. "
            "Formal mode evaluates test only after validation admission."
        ),
    )
    parser.add_argument(
        "--task-balanced-answer-loss",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--global-view-loss-weight", type=float, default=None)
    parser.add_argument("--joint-no-harm-loss-weight", type=float, default=None)
    parser.add_argument("--joint-no-harm-margin", type=float, default=None)
    parser.add_argument("--joint-causal-loss-weight", type=float, default=None)
    parser.add_argument("--joint-causal-margin", type=float, default=None)
    parser.add_argument("--point-causal-loss-weight", type=float, default=None)
    parser.add_argument("--point-causal-margin", type=float, default=None)
    parser.add_argument(
        "--point-causal-tasks",
        type=str,
        default=None,
        help="Comma-separated task types receiving the zero-local causal hinge.",
    )
    parser.add_argument("--nonpoint-no-harm-loss-weight", type=float, default=None)
    parser.add_argument("--nonpoint-no-harm-margin", type=float, default=None)
    parser.add_argument("--global-anchor-loss-weight", type=float, default=None)
    parser.add_argument("--local-anchor-loss-weight", type=float, default=None)
    parser.add_argument(
        "--swapped-question-max-records",
        type=int,
        default=None,
        help="Maximum swapped owners per audited matched group, independent of outer batch size.",
    )
    parser.add_argument(
        "--swapped-question-require-different-answer",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Skip ambiguous same-label question swaps in the grounding objective.",
    )
    parser.add_argument(
        "--ranking-loss-negative",
        type=str,
        default=None,
        choices=("global_only", "shuffled", "random", "no_latent", "zero_latent"),
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
            "grounded_evidence_adapter",
        ),
        default=None,
    )
    parser.add_argument("--adapter-init-checkpoint", type=str, default=None)
    parser.add_argument("--stage2-warm-start-checkpoint", type=str, default=None)
    parser.add_argument("--stage2b-resume-checkpoint", type=str, default=None)
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
            "grounded_role_routing",
        ),
        default=None,
    )
    parser.add_argument("--local-gate-init", type=float, default=None)
    parser.add_argument("--grounded-gate-bias-init", type=float, default=None)
    parser.add_argument("--local-text-gate-init", type=float, default=None)
    parser.add_argument(
        "--freeze-conditioned-backbone",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Freeze the Stage-1 backbone copied into the residual question branch.",
    )
    parser.add_argument(
        "--local-text-gate-trainable",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Allow per-block text gates to train; fixed-one gates avoid a second suppression path.",
    )
    parser.add_argument(
        "--local-residual-gate-trainable",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Allow the outer conditioned-minus-reference gate to train.",
    )
    parser.add_argument(
        "--zero-init-local-text-attention",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Zero-initialize text-attention outputs so Stage 2 starts exactly from Stage 1.",
    )
    parser.add_argument("--freeze-global-adapter", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--global-unfreeze-epoch", type=int, default=None)
    parser.add_argument("--global-lr", type=float, default=None)
    parser.add_argument("--global-prompt-dropout", type=float, default=None)
    parser.add_argument(
        "--mask-inactive-local-tokens",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Mask grounded role slots whose learned gate is closed before the frozen LLM attention.",
    )
    parser.add_argument(
        "--record-subset-mode",
        type=str,
        default=None,
        choices=("prefix", "hash_state"),
        help="How max_*_records is applied; hash_state selects complete deterministic state groups.",
    )
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
            "Comma-separated: correct,global_only,zero_local,local_only,no_latent,zero_latent,"
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
    parser.add_argument(
        "--checkpoint-updates",
        type=str,
        default=None,
        help="Comma-separated optimizer updates at which to screen and save adapter candidates.",
    )
    parser.add_argument(
        "--checkpoint-fractions",
        type=str,
        default=None,
        help="Batch-size-independent fractions of the optimizer budget to screen and save.",
    )
    parser.add_argument("--checkpoint-screening-records", type=int, default=None)
    parser.add_argument("--checkpoint-full-eval-top-k", type=int, default=None)
    parser.add_argument("--joint-min-causal-gain", type=float, default=None)
    parser.add_argument("--joint-max-parent-regression", type=float, default=None)
    parser.add_argument("--joint-min-no-harm-delta", type=float, default=None)
    parser.add_argument("--point-reader-min-parent-delta", type=float, default=None)
    parser.add_argument("--point-reader-min-causal-gain", type=float, default=None)
    parser.add_argument("--point-reader-max-nonpoint-regression", type=float, default=None)
    parser.add_argument("--evaluate-test", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--diagnostics-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--diagnostics-every-epochs", type=int, default=None)
    parser.add_argument("--diagnostics-records-per-task", type=int, default=None)
    parser.add_argument(
        "--diagnostics-save-states",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Persist large raw diagnostic state tensors in addition to the JSON summaries.",
    )
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
        choices=(
            "correct_accuracy",
            "macro_latent_gain",
            "normalized_point_latent_gain",
            "point_value_min_latent_gain",
            "point_value_min_grounded_gain",
            "point_value_min_causal_gain",
            "joint_ab_worst_task_delta",
        ),
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
    config = apply_joint_run_scope_overrides(load_yaml_mapping(args.config), args)
    configured_architecture = str(
        args.adapter_architecture
        or first_nested(config, ["adapter.architecture"])
        or "legacy"
    )
    configured_loss_fields = {
        "ce_loss_weight": first_nested(config, ["llm_training.ce_loss_weight"]),
        "choice_ce_loss_weight": first_nested(config, ["llm_training.choice_ce_loss_weight"]),
        "ranking_loss_weight": first_nested(config, ["llm_training.ranking_loss_weight"]),
        "ranking_loss_margin": first_nested(config, ["llm_training.ranking_loss_margin"]),
        "swapped_question_loss_weight": first_nested(config, ["llm_training.swapped_question_loss_weight"]),
        "swapped_question_loss_margin": first_nested(config, ["llm_training.swapped_question_loss_margin"]),
        "grounding_routing_loss_weight": first_nested(
            config, ["llm_training.grounding_routing_loss_weight"]
        ),
        "grounding_joint_routing_loss_weight": first_nested(
            config, ["llm_training.grounding_joint_routing_loss_weight"]
        ),
        "grounding_gate_loss_weight": first_nested(
            config, ["llm_training.grounding_gate_loss_weight"]
        ),
        "matched_group_loss_weight": first_nested(
            config, ["llm_training.matched_group_loss_weight"]
        ),
        "matched_group_loss_margin": first_nested(
            config, ["llm_training.matched_group_loss_margin"]
        ),
    }
    reported_loss_fields = set(configured_loss_fields)
    if configured_architecture in DIRECT_ALIGNMENT_ARCHITECTURES:
        reported_loss_fields = {
            "ce_loss_weight",
            "choice_ce_loss_weight",
            "ranking_loss_weight",
            "ranking_loss_margin",
            "swapped_question_loss_weight",
            "swapped_question_loss_margin",
            "matched_group_loss_weight",
            "matched_group_loss_margin",
        }
    defaulted_loss_fields = [
        field
        for field, configured_value in configured_loss_fields.items()
        if field in reported_loss_fields
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

    configured_qa_dir = (
        first_nested(config, ["patch_qa.stage2b_qa_dir"])
        if configured_architecture == "grounded_evidence_adapter"
        else None
    ) or first_nested(config, ["patch_qa.qa_dir", "data.qa_dir"])
    path_defaults = {
        "qa_dir": configured_qa_dir,
        "latent_dir": first_nested(config, ["patch_qa.latent_dir", "data.latent_dir", "latent_export.output_dir"]),
        "qa_alignment_checkpoint": first_nested(config, ["patch_qa.alignment_checkpoint"]),
        "adapter_init_checkpoint": first_nested(config, ["adapter.init_checkpoint", "patch_qa.alignment_checkpoint"]),
        "stage2_warm_start_checkpoint": first_nested(
            config, ["adapter.stage2_warm_start_checkpoint"]
        ),
        "stage2b_resume_checkpoint": first_nested(
            config, ["adapter.stage2b_resume_checkpoint"]
        ),
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
    set_default(
        args,
        "latent_channel_policy",
        first_nested(config, ["llm_training.latent_channel_policy"]),
        "all",
    )
    set_default(args, "num_workers", first_nested(config, ["llm_training.num_workers"]), 0)
    set_default(
        args,
        "distributed_timeout_seconds",
        first_nested(config, ["llm_training.distributed_timeout_seconds"]),
        1800.0,
    )
    set_default(
        args,
        "serialize_llm_loading",
        first_nested(config, ["llm_training.serialize_llm_loading"]),
        True,
    )
    set_default(
        args,
        "low_cpu_mem_usage",
        first_nested(config, ["llm_training.low_cpu_mem_usage", "model.low_cpu_mem_usage"]),
        True,
    )
    set_default(
        args,
        "min_host_memory_available_gib",
        first_nested(config, ["llm_training.min_host_memory_available_gib"]),
        16.0,
    )
    set_default(args, "device", first_nested(config, ["llm_training.device", "runtime.device"]), "auto")
    set_default(args, "torch_dtype", first_nested(config, ["llm_training.torch_dtype", "model.torch_dtype"]), "auto")
    set_default(args, "trust_remote_code", first_nested(config, ["model.trust_remote_code"]), False)
    set_default(args, "seed", first_nested(config, ["llm_training.seed", "runtime.seed"]), 42)
    set_default(args, "shuffle_seed", first_nested(config, ["llm_training.shuffle_seed", "runtime.seed"]), 42)
    set_default(args, "epochs", first_nested(config, ["llm_training.epochs"]), 3)
    set_default(
        args,
        "grounding_routing_warmup_epochs",
        first_nested(config, ["llm_training.grounding_routing_warmup_epochs"]),
        0,
    )
    set_default(
        args,
        "grounding_warmup_min_cell_top1",
        first_nested(config, ["llm_training.grounding_warmup_min_cell_top1"]),
        0.90,
    )
    set_default(
        args,
        "grounding_warmup_min_cell_top5",
        first_nested(config, ["llm_training.grounding_warmup_min_cell_top5"]),
        0.98,
    )
    set_default(
        args,
        "grounding_warmup_min_axis_top1",
        first_nested(config, ["llm_training.grounding_warmup_min_axis_top1"]),
        0.95,
    )
    set_default(
        args,
        "grounding_warmup_min_target_mass",
        first_nested(config, ["llm_training.grounding_warmup_min_target_mass"]),
        0.50,
    )
    set_default(
        args,
        "grounding_warmup_min_gate_accuracy",
        first_nested(config, ["llm_training.grounding_warmup_min_gate_accuracy"]),
        0.95,
    )
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
        "grounding_routing_loss_weight",
        configured_loss_fields["grounding_routing_loss_weight"],
        0.0,
    )
    set_default(
        args,
        "grounding_joint_routing_loss_weight",
        configured_loss_fields["grounding_joint_routing_loss_weight"],
        args.grounding_routing_loss_weight,
    )
    set_default(
        args,
        "grounding_gate_loss_weight",
        configured_loss_fields["grounding_gate_loss_weight"],
        0.0,
    )
    set_default(
        args,
        "matched_group_loss_weight",
        configured_loss_fields["matched_group_loss_weight"],
        0.0,
    )
    set_default(
        args,
        "matched_group_loss_margin",
        configured_loss_fields["matched_group_loss_margin"],
        0.5,
    )
    set_default(
        args,
        "joint_ab_training",
        first_nested(config, ["llm_training.joint_ab_training"]),
        False,
    )
    set_default(
        args,
        "point_reader_training",
        first_nested(config, ["llm_training.point_reader_training"]),
        False,
    )
    set_default(
        args,
        "full_local_reader_training",
        first_nested(config, ["llm_training.full_local_reader_training"]),
        False,
    )
    set_default(
        args,
        "task_balanced_answer_loss",
        first_nested(config, ["llm_training.task_balanced_answer_loss"]),
        False,
    )
    set_default(
        args,
        "global_view_loss_weight",
        first_nested(config, ["llm_training.global_view_loss_weight"]),
        0.0,
    )
    set_default(
        args,
        "joint_no_harm_loss_weight",
        first_nested(config, ["llm_training.joint_no_harm_loss_weight"]),
        0.0,
    )
    set_default(
        args,
        "joint_no_harm_margin",
        first_nested(config, ["llm_training.joint_no_harm_margin"]),
        0.0,
    )
    set_default(
        args,
        "joint_causal_loss_weight",
        first_nested(config, ["llm_training.joint_causal_loss_weight"]),
        0.0,
    )
    set_default(
        args,
        "joint_causal_margin",
        first_nested(config, ["llm_training.joint_causal_margin"]),
        0.0,
    )
    set_default(
        args,
        "point_causal_loss_weight",
        first_nested(config, ["llm_training.point_causal_loss_weight"]),
        0.0,
    )
    set_default(
        args,
        "point_causal_margin",
        first_nested(config, ["llm_training.point_causal_margin"]),
        0.0,
    )
    set_default(
        args,
        "point_causal_tasks",
        value_to_csv(first_nested(config, ["llm_training.point_causal_tasks"])),
        "normalized_point_value,raw_point_value_with_stats",
    )
    set_default(
        args,
        "nonpoint_no_harm_loss_weight",
        first_nested(config, ["llm_training.nonpoint_no_harm_loss_weight"]),
        0.0,
    )
    set_default(
        args,
        "nonpoint_no_harm_margin",
        first_nested(config, ["llm_training.nonpoint_no_harm_margin"]),
        0.0,
    )
    set_default(
        args,
        "global_anchor_loss_weight",
        first_nested(config, ["llm_training.global_anchor_loss_weight"]),
        0.0,
    )
    set_default(
        args,
        "local_anchor_loss_weight",
        first_nested(config, ["llm_training.local_anchor_loss_weight"]),
        0.0,
    )
    set_default(
        args,
        "swapped_question_max_records",
        first_nested(config, ["llm_training.swapped_question_max_records"]),
        8,
    )
    set_default(
        args,
        "swapped_question_require_different_answer",
        first_nested(config, ["llm_training.swapped_question_require_different_answer"]),
        True,
    )
    set_default(
        args,
        "ranking_loss_negative",
        first_nested(config, ["llm_training.ranking_loss_negative"]),
        "global_only",
    )
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
    set_default(args, "local_gate_init", first_nested(config, ["adapter.local_gate_init"]), 1.0)
    set_default(
        args,
        "grounded_gate_bias_init",
        first_nested(config, ["adapter.grounded_gate_bias_init"]),
        -2.0,
    )
    set_default(args, "local_text_gate_init", first_nested(config, ["adapter.local_text_gate_init"]), 1.0)
    set_default(
        args,
        "freeze_conditioned_backbone",
        first_nested(config, ["adapter.freeze_conditioned_backbone"]),
        True,
    )
    set_default(
        args,
        "local_text_gate_trainable",
        first_nested(config, ["adapter.local_text_gate_trainable"]),
        False,
    )
    set_default(
        args,
        "local_residual_gate_trainable",
        first_nested(config, ["adapter.local_residual_gate_trainable"]),
        False,
    )
    set_default(
        args,
        "zero_init_local_text_attention",
        first_nested(config, ["adapter.zero_init_local_text_attention"]),
        True,
    )
    set_default(args, "freeze_global_adapter", first_nested(config, ["adapter.freeze_global_adapter"]), True)
    set_default(args, "global_unfreeze_epoch", first_nested(config, ["adapter.global_unfreeze_epoch"]), 0)
    set_default(args, "global_lr", first_nested(config, ["adapter.global_lr"]), 1.0e-5)
    set_default(args, "global_prompt_dropout", first_nested(config, ["adapter.global_prompt_dropout"]), 0.0)
    set_default(
        args,
        "mask_inactive_local_tokens",
        first_nested(config, ["adapter.mask_inactive_local_tokens"]),
        False,
    )
    set_default(
        args,
        "record_subset_mode",
        first_nested(config, ["llm_training.record_subset_mode"]),
        "prefix",
    )
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
    set_default(
        args,
        "checkpoint_updates",
        value_to_csv(first_nested(config, ["llm_training.checkpoint_updates"])),
        "",
    )
    set_default(
        args,
        "checkpoint_fractions",
        value_to_csv(first_nested(config, ["llm_training.checkpoint_fractions"])),
        "",
    )
    set_default(
        args,
        "checkpoint_screening_records",
        first_nested(config, ["llm_training.checkpoint_screening_records"]),
        0,
    )
    set_default(
        args,
        "checkpoint_full_eval_top_k",
        first_nested(config, ["llm_training.checkpoint_full_eval_top_k"]),
        1,
    )
    set_default(
        args,
        "joint_min_causal_gain",
        first_nested(config, ["llm_training.joint_min_causal_gain"]),
        0.015,
    )
    set_default(
        args,
        "joint_max_parent_regression",
        first_nested(config, ["llm_training.joint_max_parent_regression"]),
        0.005,
    )
    set_default(
        args,
        "joint_min_no_harm_delta",
        first_nested(config, ["llm_training.joint_min_no_harm_delta"]),
        0.0,
    )
    set_default(
        args,
        "point_reader_min_parent_delta",
        first_nested(config, ["llm_training.point_reader_min_parent_delta"]),
        0.0,
    )
    set_default(
        args,
        "point_reader_min_causal_gain",
        first_nested(config, ["llm_training.point_reader_min_causal_gain"]),
        0.0,
    )
    set_default(
        args,
        "point_reader_max_nonpoint_regression",
        first_nested(config, ["llm_training.point_reader_max_nonpoint_regression"]),
        0.03,
    )
    set_default(args, "evaluate_test", first_nested(config, ["llm_training.evaluate_test"]), True)
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
        "diagnostics_save_states",
        first_nested(config, ["llm_training.diagnostics.save_states"]),
        True,
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
        "grounded_evidence_adapter",
    }
    if str(args.adapter_architecture) not in supported_adapter_architectures:
        raise ValueError(f"Unsupported adapter.architecture: {args.adapter_architecture}")
    if str(args.choice_scoring_mode) not in {"auto", "label", "sequence"}:
        raise ValueError(f"Unsupported llm_training.choice_scoring_mode: {args.choice_scoring_mode}")
    if str(args.adapter_architecture) in {
        "alignment_qformer",
        "alignment_adapter",
        "residual_question_qformer",
        "residual_question_adapter",
        "grounded_evidence_adapter",
    } and str(args.adapter_init_checkpoint or "").strip().lower() in {"", "none", "null", "random"}:
        raise ValueError(
            f"adapter.architecture={args.adapter_architecture} requires adapter.init_checkpoint from stage 1."
        )
    supported_local_fusion_modes = {
        "text_latent_pool",
        "anchor_queries",
        "residual_qformer",
        "residual_spatial_transformer",
        "grounded_role_routing",
    }
    if str(args.local_fusion_mode) not in supported_local_fusion_modes:
        raise ValueError(f"Unsupported adapter.local_fusion_mode: {args.local_fusion_mode}")
    if str(args.adapter_architecture) in {
        "residual_question_qformer",
        "residual_question_adapter",
    }:
        if str(args.local_question_input_mode) != "contextual_tokens":
            raise ValueError(
                "Residual question conditioning requires adapter.local_question_input_mode=contextual_tokens."
            )
        if not bool(args.local_text_gate_trainable) and float(args.local_text_gate_init) == 0.0:
            raise ValueError("A fixed adapter.local_text_gate_init cannot be zero for residual question conditioning.")
        if not bool(args.local_residual_gate_trainable) and float(args.local_gate_init) == 0.0:
            raise ValueError("A fixed adapter.local_gate_init cannot be zero for residual question conditioning.")
    if str(args.adapter_architecture) == "grounded_evidence_adapter":
        if str(args.local_question_input_mode) != "contextual_tokens":
            raise ValueError(
                "Grounded evidence requires adapter.local_question_input_mode=contextual_tokens."
            )
        if int(args.local_soft_prompt_tokens) != 2:
            raise ValueError(
                "Grounded evidence uses exactly two role/evidence tokens (primary/A and B)."
            )
        if not bool(args.group_questions_by_state):
            raise ValueError(
                "Grounded evidence requires explicit matched-group batches."
            )
        if float(args.grounding_routing_loss_weight) <= 0.0:
            raise ValueError(
                "Grounded evidence requires a positive grounding_routing_loss_weight."
            )
        if float(args.grounding_joint_routing_loss_weight) < 0.0:
            raise ValueError("grounding_joint_routing_loss_weight must be non-negative.")
        if int(args.questions_per_state_group) != 3:
            raise ValueError(
                "Grounded Stage-2B requires questions_per_state_group=3 because that is the "
                "immutable matched-task group size."
            )
        validate_atomic_group_batch_size(
            int(args.batch_size),
            int(args.questions_per_state_group),
            context="Grounded Stage-2B",
        )
        if str(args.choice_scoring_mode) != "label":
            raise ValueError(
                "Grounded Stage-2B requires choice_scoring_mode=label so routing, choice, and "
                "cross-question margins come from one positive forward."
            )
        if int(args.grounding_routing_warmup_epochs) < 0:
            raise ValueError("grounding_routing_warmup_epochs must be non-negative.")
        if int(args.grounding_routing_warmup_epochs) >= int(args.epochs):
            raise ValueError(
                "Grounded Stage-2B must retain at least one joint answer-training epoch after "
                "the routing-only warmup."
            )
        if (
            float(args.swapped_question_loss_weight) > 0.0
            and int(args.swapped_question_max_records)
            < int(args.questions_per_state_group)
        ):
            raise ValueError(
                "Grounded Stage-2B applies swapped_question_max_records per matched group; "
                "set it to at least questions_per_state_group so changing the outer batch "
                "size cannot silently bias supervision toward one member."
            )
        for setting in (
            "grounding_warmup_min_cell_top1",
            "grounding_warmup_min_cell_top5",
            "grounding_warmup_min_axis_top1",
            "grounding_warmup_min_target_mass",
            "grounding_warmup_min_gate_accuracy",
        ):
            value = float(getattr(args, setting))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"llm_training.{setting} must be finite and in [0, 1].")
        enabled_final_modes = sum(
            int(bool(value))
            for value in (
                args.joint_ab_training,
                args.point_reader_training,
                args.full_local_reader_training,
            )
        )
        if enabled_final_modes > 1:
            raise ValueError(
                "joint_ab_training, point_reader_training, and full_local_reader_training "
                "are mutually exclusive."
            )
        if bool(args.joint_ab_training):
            joint_run_scope = str(args.joint_run_scope)
            if not str(args.stage2b_resume_checkpoint or "").strip():
                raise ValueError("joint_ab_training requires adapter.stage2b_resume_checkpoint.")
            if bool(args.freeze_global_adapter):
                raise ValueError(
                    "joint_ab_training requires adapter.freeze_global_adapter=false; only the "
                    "global-only A view is allowed to update it."
                )
            if int(args.global_unfreeze_epoch) != 0:
                raise ValueError("joint_ab_training trains global from update zero; global_unfreeze_epoch must be 0.")
            if int(args.grounding_routing_warmup_epochs) != 0:
                raise ValueError("joint_ab_training requires a previously audited reader and no routing warmup.")
            if float(args.global_prompt_dropout) != 0.0:
                raise ValueError("joint_ab_training requires global_prompt_dropout=0.")
            if float(args.ranking_loss_weight) != 0.0 or float(args.swapped_question_loss_weight) != 0.0:
                raise ValueError(
                    "joint_ab_training disables legacy ranking and swapped-question objectives; "
                    "use the explicit causal and no-harm views instead."
                )
            if float(args.ce_loss_weight) != 0.0:
                raise ValueError(
                    "joint_ab_training requires ce_loss_weight=0 so every answer term is "
                    "task-balanced and no extra token-CE decoder graph is retained."
                )
            if not math.isfinite(float(args.choice_ce_loss_weight)) or float(
                args.choice_ce_loss_weight
            ) <= 0.0:
                raise ValueError("joint_ab_training requires choice_ce_loss_weight > 0.")
            if str(args.checkpoint_metric) != "joint_ab_worst_task_delta":
                raise ValueError(
                    "joint_ab_training requires checkpoint_metric=joint_ab_worst_task_delta."
                )
            if int(args.epochs) != 1:
                raise ValueError(
                    "The joint A/B large-data/small-epoch schedule requires exactly one epoch."
                )
            if joint_run_scope == "screening":
                if bool(args.evaluate_test):
                    raise ValueError(
                        "The joint A/B screening experiment must keep evaluate_test=false."
                    )
                if args.max_train_records is None or int(args.max_train_records) <= 0:
                    raise ValueError(
                        "joint_run_scope=screening requires a positive max_train_records subset."
                    )
            elif joint_run_scope == "formal":
                if not bool(args.evaluate_test):
                    raise ValueError(
                        "joint_run_scope=formal requires evaluate_test=true; test is still "
                        "gated on full-validation admission."
                    )
                if any(
                    value is not None
                    for value in (
                        args.max_train_records,
                        args.max_val_records,
                        args.max_test_records,
                    )
                ):
                    raise ValueError(
                        "joint_run_scope=formal requires complete train/val/test splits "
                        "(all max_*_records values must be null)."
                    )
                if not bool(args.require_disjoint_splits) or not bool(
                    args.require_untruncated_prompts
                ):
                    raise ValueError(
                        "joint_run_scope=formal requires disjoint splits and untruncated prompts."
                    )
            else:  # pragma: no cover - resolved before defaults
                raise ValueError(f"Unsupported joint_run_scope={joint_run_scope!r}.")
            if not bool(args.task_balanced_answer_loss):
                raise ValueError("joint_ab_training requires task_balanced_answer_loss=true.")
            for setting in (
                "global_view_loss_weight",
                "joint_no_harm_loss_weight",
                "joint_causal_loss_weight",
                "global_anchor_loss_weight",
                "local_anchor_loss_weight",
            ):
                value = float(getattr(args, setting))
                if not math.isfinite(value) or value <= 0.0:
                    raise ValueError(f"joint_ab_training requires llm_training.{setting} > 0.")
            required = {"correct", "zero_local", "global_only", "shuffled"}
            missing = sorted(required - set(parse_csv(args.eval_baselines)))
            if missing:
                raise ValueError(
                    "joint_ab_training checkpoint screening is missing eval baselines: "
                    f"{missing}."
                )
        elif bool(args.point_reader_training):
            point_tasks = tuple(parse_csv(args.point_causal_tasks))
            if set(point_tasks) != set(POINT_VALUE_TASK_TYPES) or len(point_tasks) != len(
                POINT_VALUE_TASK_TYPES
            ):
                raise ValueError(
                    "point_reader_training requires point_causal_tasks to contain exactly "
                    f"{list(POINT_VALUE_TASK_TYPES)}."
                )
            if not str(args.stage2b_resume_checkpoint or "").strip():
                raise ValueError(
                    "point_reader_training requires adapter.stage2b_resume_checkpoint."
                )
            if not bool(args.freeze_global_adapter):
                raise ValueError(
                    "point_reader_training requires adapter.freeze_global_adapter=true."
                )
            if int(args.global_unfreeze_epoch) != 0:
                raise ValueError(
                    "point_reader_training permanently freezes global; global_unfreeze_epoch must be 0."
                )
            if int(args.grounding_routing_warmup_epochs) != 0:
                raise ValueError(
                    "point_reader_training starts from an audited router and requires no routing warmup."
                )
            if float(args.global_prompt_dropout) != 0.0:
                raise ValueError("point_reader_training requires global_prompt_dropout=0.")
            if float(args.global_view_loss_weight) != 0.0:
                raise ValueError("point_reader_training disables the trainable global view.")
            if any(
                float(getattr(args, setting)) != 0.0
                for setting in (
                    "joint_no_harm_loss_weight",
                    "joint_causal_loss_weight",
                    "global_anchor_loss_weight",
                    "local_anchor_loss_weight",
                    "grounding_joint_routing_loss_weight",
                    "grounding_gate_loss_weight",
                )
            ):
                raise ValueError(
                    "point_reader_training requires legacy joint, anchor, routing, and gate "
                    "loss weights to be zero."
                )
            if float(args.point_causal_loss_weight) <= 0.0:
                raise ValueError(
                    "point_reader_training requires point_causal_loss_weight > 0."
                )
            if float(args.nonpoint_no_harm_loss_weight) <= 0.0:
                raise ValueError(
                    "point_reader_training requires nonpoint_no_harm_loss_weight > 0."
                )
            if bool(args.task_balanced_answer_loss):
                raise ValueError(
                    "point_reader_training uses the validated natural 3:3:1:1:1 task ratio; "
                    "task_balanced_answer_loss must be false."
                )
            if int(args.epochs) != 1:
                raise ValueError(
                    "The final point-reader large-data/small-epoch schedule requires one epoch."
                )
            if str(args.checkpoint_metric) != "point_value_min_causal_gain":
                raise ValueError(
                    "point_reader_training requires checkpoint_metric=point_value_min_causal_gain."
                )
            required = {"correct", "zero_local", "global_only", "shuffled"}
            missing = sorted(required - set(parse_csv(args.eval_baselines)))
            if missing:
                raise ValueError(
                    "point_reader_training checkpoint screening is missing eval baselines: "
                    f"{missing}."
                )
            point_run_scope = str(args.joint_run_scope)
            if point_run_scope == "screening":
                if bool(args.evaluate_test):
                    raise ValueError(
                        "The point-reader screening experiment must keep evaluate_test=false."
                    )
                if args.max_train_records is None or int(args.max_train_records) <= 0:
                    raise ValueError(
                        "joint_run_scope=screening requires a positive max_train_records subset."
                    )
            elif point_run_scope == "formal":
                if not bool(args.evaluate_test):
                    raise ValueError(
                        "joint_run_scope=formal requires evaluate_test=true; test remains "
                        "gated on full-validation admission."
                    )
                if any(
                    value is not None
                    for value in (
                        args.max_train_records,
                        args.max_val_records,
                        args.max_test_records,
                    )
                ):
                    raise ValueError(
                        "joint_run_scope=formal requires complete train/val/test splits."
                    )
                if not bool(args.require_disjoint_splits) or not bool(
                    args.require_untruncated_prompts
                ):
                    raise ValueError(
                        "joint_run_scope=formal requires disjoint splits and untruncated prompts."
                    )
            else:  # pragma: no cover - resolved before defaults
                raise ValueError(f"Unsupported joint_run_scope={point_run_scope!r}.")
        elif bool(args.full_local_reader_training):
            if not str(args.stage2b_resume_checkpoint or "").strip():
                raise ValueError(
                    "full_local_reader_training requires adapter.stage2b_resume_checkpoint."
                )
            if not bool(args.freeze_global_adapter) or int(args.global_unfreeze_epoch) != 0:
                raise ValueError(
                    "full_local_reader_training permanently freezes the global adapter."
                )
            if int(args.grounding_routing_warmup_epochs) != 0:
                raise ValueError(
                    "full_local_reader_training starts from an audited reader and requires no routing warmup."
                )
            if float(args.global_prompt_dropout) != 0.0:
                raise ValueError("full_local_reader_training requires global_prompt_dropout=0.")
            if bool(args.task_balanced_answer_loss):
                raise ValueError(
                    "full_local_reader_training uses the validated natural task ratio."
                )
            if int(args.epochs) != 1:
                raise ValueError(
                    "The final full local-reader large-data/small-epoch schedule requires one epoch."
                )
            if str(args.checkpoint_metric) != "point_value_min_causal_gain":
                raise ValueError(
                    "full_local_reader_training requires checkpoint_metric=point_value_min_causal_gain."
                )
            for setting in (
                "global_view_loss_weight",
                "joint_no_harm_loss_weight",
                "joint_causal_loss_weight",
                "point_causal_loss_weight",
                "nonpoint_no_harm_loss_weight",
                "global_anchor_loss_weight",
                "local_anchor_loss_weight",
            ):
                if float(getattr(args, setting)) != 0.0:
                    raise ValueError(
                        "full_local_reader_training disables reference/anchor objectives; "
                        f"llm_training.{setting} must be zero."
                    )
            if float(args.choice_ce_loss_weight) <= 0.0:
                raise ValueError(
                    "full_local_reader_training requires choice_ce_loss_weight > 0."
                )
            if float(args.matched_group_loss_weight) <= 0.0 or float(
                args.swapped_question_loss_weight
            ) <= 0.0:
                raise ValueError(
                    "full_local_reader_training retains positive matched-group and "
                    "swapped-question supervision."
                )
            required_recipe = {
                "ce_loss_weight": 0.02,
                "choice_ce_loss_weight": 1.0,
                "ranking_loss_weight": 0.0,
                "matched_group_loss_weight": 0.2,
                "matched_group_loss_margin": 0.5,
                "swapped_question_loss_weight": 0.1,
                "swapped_question_loss_margin": 0.1,
                "grounding_joint_routing_loss_weight": 0.1,
                "grounding_gate_loss_weight": 0.1,
            }
            mismatched_recipe = {
                name: {"observed": float(getattr(args, name)), "required": required}
                for name, required in required_recipe.items()
                if not math.isclose(
                    float(getattr(args, name)), required, rel_tol=0.0, abs_tol=1.0e-12
                )
            }
            if mismatched_recipe:
                raise ValueError(
                    "full_local_reader_training must reproduce the successful causal-reader "
                    f"objective recipe; mismatches={mismatched_recipe}."
                )
            if float(args.grounding_joint_routing_loss_weight) <= 0.0 or float(
                args.grounding_gate_loss_weight
            ) <= 0.0:
                raise ValueError(
                    "full_local_reader_training requires positive routing and gate supervision."
                )
            required = {"correct", "zero_local", "global_only", "shuffled"}
            missing = sorted(required - set(parse_csv(args.eval_baselines)))
            if missing:
                raise ValueError(
                    "full_local_reader_training checkpoint screening is missing eval baselines: "
                    f"{missing}."
                )
            point_run_scope = str(args.joint_run_scope)
            if point_run_scope == "screening":
                if bool(args.evaluate_test):
                    raise ValueError(
                        "The full local-reader screening experiment must keep evaluate_test=false."
                    )
                if args.max_train_records is None or int(args.max_train_records) <= 0:
                    raise ValueError(
                        "joint_run_scope=screening requires a positive max_train_records subset."
                    )
            elif point_run_scope == "formal":
                if not bool(args.evaluate_test):
                    raise ValueError(
                        "joint_run_scope=formal requires evaluate_test=true; test remains "
                        "gated on full-validation admission."
                    )
                if any(
                    value is not None
                    for value in (
                        args.max_train_records,
                        args.max_val_records,
                        args.max_test_records,
                    )
                ):
                    raise ValueError(
                        "joint_run_scope=formal requires complete train/val/test splits."
                    )
                if not bool(args.require_disjoint_splits) or not bool(
                    args.require_untruncated_prompts
                ):
                    raise ValueError(
                        "joint_run_scope=formal requires disjoint splits and untruncated prompts."
                    )
            else:  # pragma: no cover - resolved before defaults
                raise ValueError(f"Unsupported joint_run_scope={point_run_scope!r}.")
        elif not bool(args.freeze_global_adapter):
            raise ValueError(
                "grounded_evidence_adapter may unfreeze global only through joint_ab_training."
            )
        elif str(args.joint_run_scope) == "formal":
            raise ValueError(
                "joint_run_scope=formal requires a screened Stage-2B training mode."
            )
    elif int(args.grounding_routing_warmup_epochs) != 0:
        raise ValueError(
            "grounding_routing_warmup_epochs is only valid for grounded_evidence_adapter."
        )
    if int(args.initial_eval_records) < 0:
        raise ValueError("llm_training.initial_eval_records must be non-negative.")
    if (
        str(args.adapter_architecture) == "grounded_evidence_adapter"
        and str(args.stage2b_resume_checkpoint or "").strip().lower()
        not in {"", "none", "null", "random"}
        and int(args.grounding_routing_warmup_epochs) == 0
        and int(args.initial_eval_records) <= 0
    ):
        raise ValueError(
            "A grounded Stage-2B continuation with no routing warmup requires "
            "initial_eval_records > 0 for its held-out routing/gate audit."
        )
    if str(args.record_subset_mode) not in {"prefix", "hash_state"}:
        raise ValueError("llm_training.record_subset_mode must be 'prefix' or 'hash_state'.")
    if int(args.latent_cache_size) < 0 or int(args.num_workers) < 0:
        raise ValueError("llm_training.latent_cache_size and num_workers must be non-negative.")
    if str(args.latent_channel_policy) not in LATENT_CHANNEL_POLICIES:
        raise ValueError(
            "llm_training.latent_channel_policy must be one of "
            f"{sorted(LATENT_CHANNEL_POLICIES)}."
        )
    if (
        not math.isfinite(float(args.distributed_timeout_seconds))
        or float(args.distributed_timeout_seconds) <= 0.0
    ):
        raise ValueError("llm_training.distributed_timeout_seconds must be finite and positive.")
    if float(args.min_host_memory_available_gib) < 0:
        raise ValueError("llm_training.min_host_memory_available_gib must be non-negative.")
    for setting in (
        "ce_loss_weight",
        "choice_ce_loss_weight",
        "ranking_loss_weight",
        "ranking_loss_margin",
        "swapped_question_loss_weight",
        "swapped_question_loss_margin",
        "grounding_routing_loss_weight",
        "grounding_joint_routing_loss_weight",
        "grounding_gate_loss_weight",
        "matched_group_loss_weight",
        "matched_group_loss_margin",
        "global_view_loss_weight",
        "joint_no_harm_loss_weight",
        "joint_no_harm_margin",
        "joint_causal_loss_weight",
        "joint_causal_margin",
        "point_causal_loss_weight",
        "point_causal_margin",
        "nonpoint_no_harm_loss_weight",
        "nonpoint_no_harm_margin",
        "global_anchor_loss_weight",
        "local_anchor_loss_weight",
    ):
        value = float(getattr(args, setting))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"llm_training.{setting} must be non-negative.")
    for setting in ("lr", "global_lr"):
        value = float(getattr(args, setting))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{setting} must be finite and positive.")
    for setting in ("weight_decay", "grad_clip_norm"):
        value = float(getattr(args, setting))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"llm_training.{setting} must be finite and non-negative.")
    if not any(
        float(value) > 0.0
        for value in (
            args.ce_loss_weight,
            args.choice_ce_loss_weight,
            args.ranking_loss_weight,
            args.swapped_question_loss_weight,
            args.grounding_routing_loss_weight,
            args.grounding_gate_loss_weight,
            args.matched_group_loss_weight,
        )
    ):
        raise ValueError("At least one training loss weight must be positive.")
    validate_adapter_loss_contract(args)
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
    if float(args.matched_group_loss_weight) > 0.0:
        if not bool(args.group_questions_by_state):
            raise ValueError("matched_group_loss_weight requires explicit grouped batches.")
        if float(args.choice_ce_loss_weight) <= 0.0:
            raise ValueError("matched_group_loss_weight requires a positive choice_ce_loss_weight.")
    if int(args.swapped_question_max_records) <= 0:
        raise ValueError("llm_training.swapped_question_max_records must be positive.")
    checkpoint_updates = [int(value) for value in parse_csv(args.checkpoint_updates)]
    if any(value <= 0 for value in checkpoint_updates) or len(checkpoint_updates) != len(
        set(checkpoint_updates)
    ):
        raise ValueError("llm_training.checkpoint_updates must contain unique positive integers.")
    args.checkpoint_updates = ",".join(str(value) for value in sorted(checkpoint_updates))
    checkpoint_fractions = [float(value) for value in parse_csv(args.checkpoint_fractions)]
    if checkpoint_fractions and (
        any(
            not math.isfinite(value) or value <= 0.0 or value > 1.0
            for value in checkpoint_fractions
        )
        or len(checkpoint_fractions) != len(set(checkpoint_fractions))
    ):
        raise ValueError(
            "llm_training.checkpoint_fractions must contain unique finite values in (0, 1]."
        )
    if checkpoint_updates and checkpoint_fractions:
        raise ValueError("Configure checkpoint_updates or checkpoint_fractions, not both.")
    args.checkpoint_fractions = ",".join(
        f"{value:g}" for value in sorted(checkpoint_fractions)
    )
    if int(args.checkpoint_screening_records) < 0:
        raise ValueError("llm_training.checkpoint_screening_records must be non-negative.")
    if int(args.checkpoint_full_eval_top_k) <= 0:
        raise ValueError("llm_training.checkpoint_full_eval_top_k must be positive.")
    if uses_screened_stage2b_training(args):
        if not (checkpoint_updates or checkpoint_fractions) or int(
            args.checkpoint_screening_records
        ) <= 0:
            raise ValueError(
                "Screened Stage-2B training requires checkpoint fractions/updates and "
                "checkpoint_screening_records > 0."
            )
        if checkpoint_fractions and 1.0 not in checkpoint_fractions:
            raise ValueError("Screened Stage-2B checkpoint_fractions must include 1.0.")
        if int(args.initial_eval_records) != int(args.checkpoint_screening_records):
            raise ValueError(
                "The parent audit and candidate screening must use the same fixed validation subset."
            )
    if bool(args.joint_ab_training):
        if not math.isfinite(float(args.joint_min_causal_gain)) or float(
            args.joint_min_causal_gain
        ) < 0.0:
            raise ValueError("joint_min_causal_gain must be finite and non-negative.")
        if not math.isfinite(float(args.joint_max_parent_regression)) or float(
            args.joint_max_parent_regression
        ) < 0.0:
            raise ValueError("joint_max_parent_regression must be finite and non-negative.")
        if not math.isfinite(float(args.joint_min_no_harm_delta)):
            raise ValueError("joint_min_no_harm_delta must be finite.")
    if bool(args.point_reader_training) or bool(args.full_local_reader_training):
        for setting in (
            "point_reader_min_parent_delta",
            "point_reader_min_causal_gain",
            "point_reader_max_nonpoint_regression",
        ):
            value = float(getattr(args, setting))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"llm_training.{setting} must be finite and non-negative.")
    if not math.isfinite(float(args.warmup_ratio)) or not 0.0 <= float(args.warmup_ratio) < 1.0:
        raise ValueError("llm_training.warmup_ratio must be in [0, 1).")
    if not math.isfinite(float(args.min_lr_ratio)) or not 0.0 <= float(args.min_lr_ratio) <= 1.0:
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
    if str(args.checkpoint_metric) in {
        "macro_latent_gain",
        "normalized_point_latent_gain",
        "point_value_min_latent_gain",
        "point_value_min_grounded_gain",
        "point_value_min_causal_gain",
    } and "shuffled" not in parse_csv(args.eval_baselines):
        raise ValueError(
            f"checkpoint_metric={args.checkpoint_metric} requires shuffled in llm_training.eval_baselines."
        )
    if (
        str(args.checkpoint_metric) == "point_value_min_grounded_gain"
        and "global_only" not in parse_csv(args.eval_baselines)
    ):
        raise ValueError(
            "checkpoint_metric=point_value_min_grounded_gain requires global_only in "
            "llm_training.eval_baselines."
        )
    if str(args.checkpoint_metric) == "point_value_min_causal_gain":
        required_causal_baselines = {"zero_local", "global_only", "shuffled"}
        missing = sorted(required_causal_baselines - set(parse_csv(args.eval_baselines)))
        if missing:
            raise ValueError(
                "checkpoint_metric=point_value_min_causal_gain requires "
                f"{', '.join(missing)} in llm_training.eval_baselines."
            )
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


def checkpoint_updates_from_fractions(
    total_updates: int,
    fractions: Sequence[float],
) -> list[int]:
    if int(total_updates) <= 0:
        raise ValueError("total_updates must be positive.")
    normalized = sorted({float(value) for value in fractions})
    if not normalized or any(
        not math.isfinite(value) or value <= 0.0 or value > 1.0
        for value in normalized
    ):
        raise ValueError("Checkpoint fractions must be unique finite values in (0, 1].")
    updates = sorted(
        {
            min(int(total_updates), max(1, int(math.ceil(value * int(total_updates)))))
            for value in normalized
        }
    )
    if updates[-1] != int(total_updates):
        raise ValueError("Checkpoint fractions must include 1.0 so the final update is saved.")
    return updates


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
    # Disable stochastic behavior recursively first. This also covers modules
    # that call functional dropout using their own ``training`` flag.
    model.eval()
    if checkpoint_training and checkpointing_active:
        # Decoder checkpointing is gated on the decoder backbone's training flag.
        # Flip only that flag so all attention/MLP children remain deterministic.
        decoder = getattr(model, "model", model)
        if not isinstance(decoder, nn.Module):
            raise TypeError("The frozen causal LLM exposes a non-module decoder backbone.")
        decoder.training = True


def frozen_llm_checkpoint_execution_active(model: nn.Module) -> bool:
    """Return whether the frozen decoder is in its checkpoint-enabled execution mode."""
    decoder = getattr(model, "model", model)
    if not isinstance(decoder, nn.Module):
        raise TypeError("The frozen causal LLM exposes a non-module decoder backbone.")
    return bool(getattr(model, "is_gradient_checkpointing", False) and decoder.training)


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
        low_cpu_mem_usage=bool(getattr(args, "low_cpu_mem_usage", True)),
    )
    model.to(device)
    gc.collect()
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if bool(args.llm_gradient_checkpointing):
        enable_checkpointing = getattr(model, "gradient_checkpointing_enable", None)
        if not callable(enable_checkpointing):
            raise ValueError("The selected causal LLM does not support gradient checkpointing.")
        try:
            enable_checkpointing(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError as exc:
            raise RuntimeError(
                "Stage-2 soft-prefix training requires non-reentrant gradient checkpointing. "
                "Upgrade transformers or explicitly disable llm_training.llm_gradient_checkpointing "
                "for a memory-bounded smoke run."
            ) from exc
        if not bool(getattr(model, "is_gradient_checkpointing", False)):
            raise RuntimeError("The causal LLM did not report active gradient checkpointing.")
        # Transformers activates decoder checkpointing only when its decoder backbone is in training mode.
        set_frozen_llm_execution_mode(model, checkpoint_training=True)
    else:
        set_frozen_llm_execution_mode(model, checkpoint_training=False)
    return model, model_dtype


def load_llm_with_bounded_host_memory(args: argparse.Namespace, device: torch.device):
    """Construct DDP replicas serially so CPU checkpoint copies never peak together."""

    serialize = bool(getattr(args, "serialize_llm_loading", True))
    minimum_available_gib = float(getattr(args, "min_host_memory_available_gib", 0.0))
    if not distributed_is_initialized() or not serialize:
        enforce_host_memory_floor(device, minimum_available_gib, "LLM loading")
        return load_llm(args, device)

    llm = None
    model_dtype = None
    for load_rank in range(distributed_world_size()):
        reports = enforce_host_memory_floor(
            device,
            minimum_available_gib,
            f"LLM loading for rank {load_rank}",
        )
        if is_main_process() and reports:
            print(
                f"startup=llm_load_rank rank={load_rank}/{distributed_world_size() - 1} "
                f"host_available={min(item['available_gib'] for item in reports):.2f}GiB "
                f"rank_rss="
                + ",".join(
                    f"r{int(item['rank'])}:{item['process_rss_gib']:.2f}GiB"
                    for item in reports
                )
            )

        local_error: BaseException | None = None
        if distributed_rank() == load_rank:
            try:
                llm, model_dtype = load_llm(args, device)
            except BaseException as exc:  # propagate a coherent failure to ranks waiting below
                local_error = exc
        error_payload = [
            None
            if local_error is None
            else f"{type(local_error).__name__}: {str(local_error)[:2000]}"
        ]
        dist.broadcast_object_list(error_payload, src=load_rank)
        if error_payload[0] is not None:
            raise RuntimeError(
                f"Distributed LLM loading failed on rank {load_rank}: {error_payload[0]}"
            ) from local_error
        distributed_barrier()

    if llm is None or model_dtype is None:
        raise RuntimeError("The local rank did not construct its frozen LLM replica.")
    return llm, model_dtype


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


def validate_qa_latent_contract(
    metadata: Mapping[str, Any],
    *,
    configured_alignment_checkpoint: str | Path | None,
    require_formal_contract: bool,
) -> dict[str, Any] | None:
    """Bind every formal latent payload to one immutable Stage-1 checkpoint."""
    metadata_format = str(metadata.get("format", ""))
    if metadata_format not in {PATCH_QA_FORMAT, PATCH_MATCHED_QA_FORMAT}:
        if require_formal_contract:
            raise ValueError(
                "Formal patch QA requires a supported immutable QA format, got "
                f"{metadata_format!r}. "
                "Regenerate QA and latent caches with scripts/build_tensor_patch_qa.py."
            )
        return None
    latent_format = str(metadata.get("latent_format", ""))
    if latent_format != PATCH_LATENT_FORMAT:
        raise ValueError(
            f"Patch QA metadata requires latent_format={PATCH_LATENT_FORMAT!r}, got {latent_format!r}."
        )
    latent_audit_format = str(metadata.get("latent_audit_format", ""))
    if latent_audit_format != PATCH_LATENT_AUDIT_FORMAT:
        raise ValueError(
            "Patch QA metadata has a missing or stale latent_audit_format; regenerate the QA JSONL "
            "with scripts/build_tensor_patch_qa.py."
        )
    raw_shape = metadata.get("latent_shape")
    if not isinstance(raw_shape, Sequence) or isinstance(raw_shape, (str, bytes)):
        raise ValueError("Patch QA metadata latent_shape must be a three-element integer sequence.")
    latent_shape = [int(value) for value in raw_shape]
    if len(latent_shape) != 3 or any(value <= 0 for value in latent_shape):
        raise ValueError(f"Patch QA metadata has invalid latent_shape={latent_shape}.")
    storage_dtype = str(metadata.get("storage_dtype", ""))
    if storage_dtype not in {"float16", "float32"}:
        raise ValueError(f"Patch QA metadata has unsupported storage_dtype={storage_dtype!r}.")
    normalization = metadata.get("encoder_input_normalization")
    if not isinstance(normalization, Mapping):
        raise ValueError("Patch QA metadata is missing encoder_input_normalization.")
    normalized_config = canonical_normalization(normalization)
    mode = normalized_config["mode"]
    scope = normalized_config["scope"]
    if mode != "zscore" or scope != "channel" or any(
        normalized_config[name] is not None for name in ("clip_min", "clip_max")
    ):
        raise ValueError(
            "Formal Stage 2 expects unclipped per-patch channel z-score latents; "
            f"metadata reports {normalized_config}."
        )

    metadata_checkpoint = str(metadata.get("alignment_checkpoint", "")).strip()
    configured_checkpoint = str(configured_alignment_checkpoint or "").strip()
    checkpoint = configured_checkpoint or metadata_checkpoint
    if not checkpoint:
        raise ValueError("Patch QA metadata/config does not identify the Stage-1 alignment checkpoint.")
    if metadata_checkpoint and canonical_path(metadata_checkpoint) != canonical_path(checkpoint):
        raise ValueError(
            "Patch QA latent contract points to a different Stage-1 checkpoint: "
            f"metadata={canonical_path(metadata_checkpoint)}, configured={canonical_path(checkpoint)}."
        )
    metadata_sha256 = str(metadata.get("alignment_checkpoint_sha256", "")).lower()
    if len(metadata_sha256) != 64 or any(character not in "0123456789abcdef" for character in metadata_sha256):
        raise ValueError("Patch QA metadata has a missing or invalid alignment_checkpoint_sha256.")
    actual_sha256 = sha256_file(checkpoint)
    if actual_sha256 != metadata_sha256:
        raise ValueError(
            "The Stage-1 checkpoint file changed after patch latents were generated: "
            f"path={canonical_path(checkpoint)}, metadata_sha256={metadata_sha256}, "
            f"actual_sha256={actual_sha256}. Regenerate the latent cache."
        )
    return {
        "format": PATCH_LATENT_FORMAT,
        "latent_audit_format": PATCH_LATENT_AUDIT_FORMAT,
        "alignment_checkpoint": canonical_path(checkpoint),
        "alignment_checkpoint_sha256": actual_sha256,
        "encoder_input_normalization": normalized_config,
        "latent_shape": latent_shape,
        "storage_dtype": storage_dtype,
    }


def audit_qa_metadata(args: argparse.Namespace) -> dict[str, Any]:
    build_marker = Path(args.qa_dir) / PATCH_QA_BUILD_MARKER
    if build_marker.exists():
        raise RuntimeError(
            "Patch QA assets are marked as an incomplete or active build: "
            f"{build_marker}. Finish/rerun scripts/build_tensor_patch_qa.py before Stage 2."
        )
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
    legacy_metadata_format = "tensor_patch_qa_v2"
    prompt_contract = str(metadata.get("prompt_contract", ""))
    coordinate_origin = int(metadata.get("natural_language_coordinate_origin", -1))
    supported_formal_formats = {PATCH_QA_FORMAT, PATCH_MATCHED_QA_FORMAT}
    if bool(args.require_disjoint_splits) and (
        metadata_format not in supported_formal_formats
        or prompt_contract != PATCH_QA_PROMPT_CONTRACT
        or coordinate_origin != 1
    ):
        raise ValueError(
            "Formal patch QA training requires regenerated encoder-zscore, one-based natural-language prompts. "
            f"Observed format={metadata_format!r}, prompt_contract={prompt_contract!r}, "
            f"coordinate_origin={coordinate_origin}. Run scripts/build_tensor_patch_qa.py with the current code; "
            "formal v3 caches must be regenerated or strictly revalidated."
        )
    if not bool(args.require_disjoint_splits) and metadata_format not in {
        PATCH_QA_FORMAT,
        PATCH_MATCHED_QA_FORMAT,
        legacy_metadata_format,
    }:
        raise ValueError(f"Unsupported patch QA metadata format={metadata_format!r}.")
    matched_group_format = str(metadata.get("matched_group_format", ""))
    if metadata_format == PATCH_MATCHED_QA_FORMAT:
        if matched_group_format != MATCHED_GROUP_FORMAT:
            raise ValueError(
                "Matched Stage-2 QA metadata has an unsupported group contract: "
                f"{matched_group_format!r}."
            )
        if not bool(metadata.get("requires_explicit_group_sampler", False)):
            raise ValueError("Matched Stage-2 QA must require the explicit group sampler.")
        if not bool(args.group_questions_by_state):
            raise ValueError(
                "tensor_patch_matched_qa_v1 cannot be trained without group_questions_by_state=true."
            )
        stage2b = metadata.get("stage2b")
        if not isinstance(stage2b, Mapping):
            raise ValueError("Matched Stage-2 QA metadata is missing its stage2b contract.")
        expected_batch_group_size = int(stage2b.get("batch_group_size", 0))
        expected_records_per_state = int(stage2b.get("records_per_train_state", 0))
        if expected_batch_group_size != 3 or expected_records_per_state != 9:
            raise ValueError(
                "Matched Stage-2 QA must declare three-record atomic groups and nine records per state."
            )
        if int(args.questions_per_state_group) != expected_batch_group_size:
            raise ValueError(
                "The configured questions_per_state_group must equal the matched QA atomic "
                f"group size: configured={args.questions_per_state_group}, "
                f"group_size={expected_batch_group_size}."
            )
        validate_atomic_group_batch_size(
            int(args.batch_size),
            expected_batch_group_size,
            context="Matched Stage-2 QA",
        )
    elif str(args.adapter_architecture) == "grounded_evidence_adapter":
        raise ValueError(
            "grounded_evidence_adapter requires tensor_patch_matched_qa_v1 assets; "
            "run scripts/build_tensor_patch_matched_qa.py first."
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
    adapter_init = str(args.adapter_init_checkpoint or "").strip()
    direct_alignment_provenance: dict[str, Any] | None = None
    if is_direct_alignment_architecture(args.adapter_architecture) or str(
        args.adapter_architecture
    ) == "grounded_evidence_adapter":
        direct_alignment_provenance = validate_direct_alignment_provenance(
            metadata_checkpoint=observed_alignment,
            configured_checkpoint=configured_alignment,
            adapter_checkpoint=adapter_init,
            require_metadata_checkpoint=bool(args.require_disjoint_splits),
        )
    elif observed_alignment and configured_alignment:
        observed_path = _configured_checkpoint_path(observed_alignment)
        configured_path = _configured_checkpoint_path(configured_alignment)
        if observed_path != configured_path:
            raise ValueError(
                "Patch QA metadata was generated with a different alignment checkpoint. "
                f"metadata={observed_path}, config={configured_path}."
            )
    latent_contract = validate_qa_latent_contract(
        metadata,
        configured_alignment_checkpoint=configured_alignment,
        require_formal_contract=bool(args.require_disjoint_splits),
    )
    split_mode = str(metadata.get("split_mode", "unknown"))
    if bool(args.require_disjoint_splits) and split_mode != "sample":
        raise ValueError(
            f"Formal patch QA training requires metadata split_mode=sample, got {split_mode!r}."
        )
    question_seed_mode = str(metadata.get("question_seed_mode", "legacy_record_order"))
    supported_seed_modes = {
        "sha256(seed|patch_id)",
        "sha256(seed|patch_id|variant)",
        "sha256(seed|state_ref|namespace)",
    }
    if bool(args.require_disjoint_splits) and question_seed_mode not in supported_seed_modes:
        raise ValueError(
            "Formal patch QA training requires independently seeded questions. Regenerate the QA JSONL with "
            "scripts/build_tensor_patch_qa.py; existing latent files will be reused."
        )
    question_variants = dict(metadata.get("question_variants", {}))
    if bool(args.group_questions_by_state) and metadata_format != PATCH_MATCHED_QA_FORMAT:
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
        "matched_group_format": matched_group_format or None,
        "prompt_contract": prompt_contract,
        "natural_language_coordinate_origin": coordinate_origin,
        "alignment_checkpoint": str(observed_alignment or ""),
        "alignment_checkpoint_sha256": str(metadata.get("alignment_checkpoint_sha256", "")),
        "latent_contract": latent_contract,
        "direct_alignment_provenance": direct_alignment_provenance,
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
    audit_local_conditioning_prompt: bool = True,
) -> dict[str, Any]:
    limit = int(max_prompt_tokens)
    summary: dict[str, Any] = {
        "max_prompt_tokens": limit,
        "prompt_template": str(prompt_template),
        "local_conditioning_prompt_audited": bool(audit_local_conditioning_prompt),
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
            local_encoded: Sequence[Sequence[int] | None]
            if audit_local_conditioning_prompt:
                local_encoded = tokenizer(
                    [
                        build_local_conditioning_prompt(record, prompt_template=prompt_template)
                        for record in records
                    ],
                    add_special_tokens=True,
                    truncation=False,
                )["input_ids"]
            else:
                local_encoded = [None] * len(records)
            for record, token_ids, local_token_ids in zip(records, encoded, local_encoded):
                token_count = len(token_ids)
                local_token_count = len(local_token_ids) if local_token_ids is not None else 0
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
                if audit_local_conditioning_prompt and local_token_count > limit:
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
        int(summary["truncated_records"]) == 0
        and (
            not bool(audit_local_conditioning_prompt)
            or int(summary["local_truncated_records"]) == 0
        )
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
    if isinstance(adapter, HybridGlobalLocalAdapter) and isinstance(
        adapter.local_adapter, GroundedEvidenceAdapter
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
    detach_global_for_local: bool = False,
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
        detach_global_for_local=detach_global_for_local,
    )


def adapter_soft_embeds(
    adapter: TensorSoftPromptAdapter,
    latent_map: torch.Tensor,
    text_embeds: torch.Tensor,
    question_embeds: torch.Tensor | None,
    question_mask: torch.Tensor | None,
    records: Sequence[Mapping[str, Any]] | None,
    mode: str,
    detach_global_for_local: bool = False,
) -> torch.Tensor:
    structured_query = (
        structured_query_features(records, text_embeds.device)
        if records is not None and adapter.structured_query_conditioning
        else None
    )
    if mode == "global_only" and isinstance(adapter, HybridGlobalLocalAdapter):
        if adapter.freeze_global and not latent_map.requires_grad:
            with torch.no_grad():
                global_prompts = adapter.global_adapter.forward_soft_prompts(latent_map)
        else:
            global_prompts = adapter.global_adapter.forward_soft_prompts(latent_map)
        selected = global_prompts.to(dtype=text_embeds.dtype)
        adapter._last_global_prompts = global_prompts
        adapter._last_local_prompts = None
        adapter._last_role_gate_logits = None
        adapter._last_soft_prompt_attention_mask = _all_visible_soft_prompt_mask(selected)
        return selected
    if mode in {"correct", "global_only", "zero_local", "local_only"}:
        if isinstance(adapter, HybridGlobalLocalAdapter):
            global_prompts, local_prompts, combined_prompts = adapter.forward_components(
                latent_map,
                question_embeds=question_embeds,
                question_mask=question_mask,
                structured_query=structured_query,
                detach_global_for_local=detach_global_for_local,
            )
            zero_local_prompts = (
                global_prompts
                if adapter.residual_mode
                else torch.cat([torch.zeros_like(local_prompts), global_prompts], dim=1)
            )
            selected = {
                "correct": combined_prompts,
                "global_only": global_prompts,
                "zero_local": zero_local_prompts,
                "local_only": local_prompts,
            }[mode]
            selected = selected.to(dtype=text_embeds.dtype)
            adapter._last_soft_prompt_attention_mask = grounded_soft_prompt_attention_mask(
                adapter,
                selected,
                mode=mode,
                gate_logits=adapter._last_role_gate_logits,
            )
            return selected
        selected = adapter(
            latent_map,
            question_embeds=question_embeds,
            question_mask=question_mask,
            structured_query=structured_query,
        ).to(dtype=text_embeds.dtype)
        return torch.zeros_like(selected) if mode == "zero_local" else selected
    if mode == "no_latent":
        batch_size = latent_map.shape[0]
        selected = text_embeds.new_zeros(
            (batch_size, adapter.soft_prompt_tokens, text_embeds.shape[-1])
        )
        if isinstance(adapter, HybridGlobalLocalAdapter):
            adapter._last_global_prompts = None
            adapter._last_local_prompts = None
            adapter._last_role_gate_logits = None
            adapter._last_soft_prompt_attention_mask = _all_visible_soft_prompt_mask(selected)
        return selected
    if mode in {"shuffled", "random", "zero_latent"}:
        selected = adapter(
            latent_map,
            question_embeds=question_embeds,
            question_mask=question_mask,
            structured_query=structured_query,
        ).to(dtype=text_embeds.dtype)
        if isinstance(adapter, HybridGlobalLocalAdapter):
            adapter._last_soft_prompt_attention_mask = grounded_soft_prompt_attention_mask(
                adapter,
                selected,
                mode=mode,
                gate_logits=adapter._last_role_gate_logits,
            )
        return selected
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
    detach_global_for_local: bool = False,
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
        detach_global_for_local=detach_global_for_local,
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
            detach_global_for_local=detach_global_for_local,
        )
    inputs_embeds = torch.cat([soft_embeds, text_embeds], dim=1)
    soft_attention = grounded_soft_prompt_attention_mask(
        adapter,
        soft_embeds,
        mode=soft_prompt_mode,
        dtype=text_attention_mask.dtype,
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
    precomputed_soft_attention_mask: torch.Tensor | None = None,
    precomputed_question_context: tuple[torch.Tensor, torch.Tensor] | None = None,
    detach_global_for_local: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    require_precomputed_grounded_attention_mask(
        adapter,
        precomputed_soft_embeds,
        precomputed_soft_attention_mask,
    )
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
            detach_global_for_local=detach_global_for_local,
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
            detach_global_for_local=detach_global_for_local,
        )
    soft_embeds = soft_embeds.to(device=device, dtype=text_embeds.dtype)
    inputs_embeds = torch.cat([soft_embeds, text_embeds], dim=1)
    soft_attention = grounded_soft_prompt_attention_mask(
        adapter,
        soft_embeds,
        mode=soft_prompt_mode,
        dtype=text_attention_mask.dtype,
        precomputed=precomputed_soft_attention_mask,
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
    precomputed_soft_embeds: torch.Tensor | None = None,
    precomputed_soft_attention_mask: torch.Tensor | None = None,
    detach_global_for_local: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, dict[str, Any]]:
    require_precomputed_grounded_attention_mask(
        adapter,
        precomputed_soft_embeds,
        precomputed_soft_attention_mask,
    )
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
    soft_embeds = precomputed_soft_embeds
    if soft_embeds is None:
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
            detach_global_for_local=detach_global_for_local,
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
            detach_global_for_local=detach_global_for_local,
        )
    soft_embeds = soft_embeds.to(device=device, dtype=text_embeds.dtype)
    inputs_embeds = torch.cat([soft_embeds, text_embeds], dim=1)
    soft_attention = grounded_soft_prompt_attention_mask(
        adapter,
        soft_embeds,
        mode=soft_prompt_mode,
        dtype=text_attention_mask.dtype,
        precomputed=precomputed_soft_attention_mask,
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
    candidate_log_probs: list[torch.Tensor] = []
    hard_correct = 0
    for row, (candidate_ids, target_index) in enumerate(zip(token_ids_by_record, target_indices)):
        candidate_logits = first_logits[row, torch.tensor(candidate_ids, device=device)]
        candidate_log_probs.append(F.log_softmax(candidate_logits.float(), dim=-1))
        losses.append(
            F.cross_entropy(
                candidate_logits.unsqueeze(0),
                torch.tensor([target_index], device=device),
            )
        )
        hard_correct += int(int(torch.argmax(candidate_logits.detach()).item()) == int(target_index))
    if not losses:
        raise ValueError("single_token_choice_ce_loss received an empty record batch.")
    return (
        torch.stack(losses).mean(),
        sequence_nll.sum() / target_counts.sum().clamp_min(1),
        torch.stack(losses),
        soft_embeds,
        {
            "choice_accuracy": hard_correct / max(1, len(losses)),
            "choice_01_loss": 1.0 - hard_correct / max(1, len(losses)),
            "choice_single_token_path": 1.0,
            "candidate_log_probs": candidate_log_probs,
            "candidate_target_indices": [int(value) for value in target_indices],
            "soft_attention_mask": soft_attention,
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
    precomputed_soft_embeds: torch.Tensor | None = None,
    precomputed_soft_attention_mask: torch.Tensor | None = None,
    detach_global_for_local: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, dict[str, Any]]:
    require_precomputed_grounded_attention_mask(
        adapter,
        precomputed_soft_embeds,
        precomputed_soft_attention_mask,
    )
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

    base_soft_embeds = precomputed_soft_embeds
    if base_soft_embeds is None:
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
            detach_global_for_local=detach_global_for_local,
        )
    base_soft_attention_mask = None
    if base_soft_embeds is not None:
        base_soft_attention_mask = grounded_soft_prompt_attention_mask(
            adapter,
            base_soft_embeds,
            mode=soft_prompt_mode,
            precomputed=precomputed_soft_attention_mask,
        )
    candidate_soft_embeds = (
        torch.stack([base_soft_embeds[index] for index in candidate_owners], dim=0)
        if base_soft_embeds is not None
        else None
    )
    candidate_soft_attention_masks = (
        torch.stack(
            [base_soft_attention_mask[index] for index in candidate_owners], dim=0
        )
        if base_soft_attention_mask is not None
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
            precomputed_soft_attention_mask=(
                candidate_soft_attention_masks[start:end]
                if candidate_soft_attention_masks is not None
                else None
            ),
            detach_global_for_local=detach_global_for_local,
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
    candidate_log_probs: list[torch.Tensor] = []
    correct_nll_sums: list[torch.Tensor] = []
    correct_target_counts: list[torch.Tensor] = []
    hard_correct = 0
    start = 0
    for count, target_index in zip(candidate_counts, target_indices):
        scores = -flat_nll[start : start + count]
        candidate_log_probs.append(F.log_softmax(scores.float(), dim=-1))
        target = torch.tensor([int(target_index)], dtype=torch.long, device=device)
        losses.append(F.cross_entropy(scores.unsqueeze(0), target))
        correct_flat_index = start + int(target_index)
        correct_nll_sums.append(flat_nll_sum[correct_flat_index])
        correct_target_counts.append(flat_target_counts[correct_flat_index])
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
    return loss, correct_answer_ce, torch.stack(losses), base_soft_embeds, {
        "choice_accuracy": float(accuracy),
        "choice_01_loss": float(1.0 - accuracy),
        "choice_single_token_path": 0.0,
        "candidate_log_probs": candidate_log_probs,
        "candidate_target_indices": [int(value) for value in target_indices],
        "soft_attention_mask": base_soft_attention_mask,
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
    precomputed_soft_embeds: torch.Tensor | None = None,
    precomputed_soft_attention_mask: torch.Tensor | None = None,
    detach_global_for_local: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, dict[str, Any]]:
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
            precomputed_soft_embeds=precomputed_soft_embeds,
            precomputed_soft_attention_mask=precomputed_soft_attention_mask,
            detach_global_for_local=detach_global_for_local,
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
        precomputed_soft_embeds=precomputed_soft_embeds,
        precomputed_soft_attention_mask=precomputed_soft_attention_mask,
        detach_global_for_local=detach_global_for_local,
    )


def matched_coordinate_group_loss(
    records: Sequence[Mapping[str, Any]],
    candidate_log_probs: Sequence[torch.Tensor],
    margin: float,
    task_weights: Mapping[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if len(candidate_log_probs) != len(records) or not records:
        raise ValueError("Matched-coordinate loss requires one candidate distribution per record.")
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        spec = record.get("matched_group")
        if not isinstance(spec, Mapping):
            continue
        group_id = str(spec.get("margin_group_id") or "")
        if group_id:
            grouped[group_id].append(index)
    terms: list[torch.Tensor] = []
    gaps: list[torch.Tensor] = []
    group_exact = 0
    eligible_records = 0
    for group_id, indices in grouped.items():
        members = sorted(
            indices,
            key=lambda index: int(records[index]["matched_group"].get("margin_member_index", -1)),
        )
        specs = [records[index]["matched_group"] for index in members]
        declared_sizes = {int(spec.get("margin_group_size", 0)) for spec in specs}
        observed_members = [int(spec.get("margin_member_index", -1)) for spec in specs]
        if declared_sizes != {len(members)} or observed_members != list(range(len(members))):
            raise ValueError(
                f"Margin group {group_id} is incomplete: sizes={declared_sizes}, members={observed_members}."
            )
        if len(members) < 2:
            raise ValueError(f"Margin group {group_id} must contain at least two records.")
        states = {str(records[index].get("state_ref", "")) for index in members}
        tasks = {str(records[index].get("task_type", "")) for index in members}
        identities = {
            json.dumps(
                latent_identity_from_record(records[index]),
                sort_keys=True,
                separators=(",", ":"),
            )
            for index in members
        }
        choices = [tuple(str(value) for value in records[index].get("choices", ())) for index in members]
        option_hashes = {
            str(records[index]["matched_group"].get("option_set_sha256", ""))
            for index in members
        }
        coordinate_sets = {
            str(records[index]["matched_group"].get("coordinate_set_id", ""))
            for index in members
        }
        margin_kinds = {
            str(records[index]["matched_group"].get("margin_kind", ""))
            for index in members
        }
        if (
            len(states) != 1
            or len(tasks) != 1
            or len(identities) != 1
            or len(set(choices)) != 1
            or len(option_hashes) != 1
            or "" in option_hashes
            or len(coordinate_sets) != 1
            or "" in coordinate_sets
            or len(margin_kinds) != 1
        ):
            raise ValueError(
                f"Margin group {group_id} must share state, latent, task, choices, coordinate set, "
                "margin kind, and option-set identity."
            )
        margin_kind = next(iter(margin_kinds))
        expected_size = 3 if margin_kind == "coordinate_choice" else 2 if margin_kind == "role_swap" else 0
        if len(members) != expected_size:
            raise ValueError(
                f"Margin group {group_id} kind={margin_kind!r} requires size {expected_size}, "
                f"got {len(members)}."
            )
        labels = choices[0]
        answers = [str(records[index].get("answer", "")) for index in members]
        if len(set(answers)) != len(answers) or any(answer not in labels for answer in answers):
            raise ValueError(f"Margin group {group_id} requires distinct valid answers.")
        group_correct = True
        for owner in members:
            owner_probs = candidate_log_probs[owner]
            if int(owner_probs.numel()) != len(labels):
                raise ValueError(f"Margin group {group_id} candidate width does not match its choices.")
            answer_index = labels.index(str(records[owner]["answer"]))
            owner_gaps: list[torch.Tensor] = []
            for counterfactual in members:
                if counterfactual == owner:
                    continue
                other_probs = candidate_log_probs[counterfactual]
                gap = owner_probs[answer_index] - other_probs[answer_index]
                gaps.append(gap)
                owner_gaps.append(gap)
            owner_term = torch.stack(
                [F.relu(float(margin) - gap) for gap in owner_gaps]
            ).mean()
            if task_weights:
                owner_task = str(records[owner].get("task_type", "unknown"))
                if owner_task not in task_weights:
                    raise ValueError(
                        f"No matched-group task weight was configured for {owner_task!r}."
                    )
                owner_term = owner_term * float(task_weights[owner_task])
            terms.append(owner_term)
            group_correct = group_correct and (
                int(torch.argmax(owner_probs.detach()).item()) == int(answer_index)
            )
        eligible_records += len(members)
        group_exact += int(group_correct)
    reference = candidate_log_probs[0]
    loss = (
        torch.stack(terms).sum() / float(len(records))
        if terms
        else reference.new_zeros(())
    )
    detached_gaps = torch.stack([gap.detach().float() for gap in gaps]) if gaps else None
    return loss, {
        "matched_group_count": float(len(grouped)),
        "matched_group_records": float(eligible_records),
        "matched_group_pairs": float(len(gaps)),
        "matched_group_gap_mean": (
            float(detached_gaps.mean().cpu().item()) if detached_gaps is not None else 0.0
        ),
        "matched_group_satisfaction": (
            float((detached_gaps >= float(margin)).float().mean().cpu().item())
            if detached_gaps is not None
            else 0.0
        ),
        "matched_group_exact_accuracy": group_exact / max(1, len(grouped)),
    }


def _key_cosine_metrics(keys: torch.Tensor) -> dict[str, float]:
    normalized = F.normalize(keys.detach().float(), dim=-1)
    singular_values = torch.linalg.svdvals(normalized)
    singular_mass = singular_values / singular_values.sum().clamp_min(1.0e-12)
    effective_rank = torch.exp(
        -(singular_mass * singular_mass.clamp_min(1.0e-12).log()).sum()
    )
    similarity = normalized @ normalized.transpose(0, 1)
    mask = ~torch.eye(
        int(similarity.shape[0]),
        dtype=torch.bool,
        device=similarity.device,
    )
    off_diagonal = similarity[mask]
    if int(off_diagonal.numel()) == 0:
        return {
            "off_diagonal_cosine_mean": 0.0,
            "off_diagonal_cosine_max": 0.0,
            "minimum_pairwise_l2": 0.0,
            "effective_rank": float(effective_rank.cpu().item()),
        }
    maximum_cosine = float(off_diagonal.max().cpu().item())
    return {
        "off_diagonal_cosine_mean": float(off_diagonal.mean().cpu().item()),
        "off_diagonal_cosine_max": maximum_cosine,
        "minimum_pairwise_l2": math.sqrt(max(0.0, 2.0 - 2.0 * maximum_cosine)),
        "effective_rank": float(effective_rank.cpu().item()),
    }


@torch.no_grad()
def grounded_reader_geometry_metrics(adapter: nn.Module) -> dict[str, Any]:
    """Summarize whether learned axis keys have separated from their fixed 2D anchors."""

    local = _grounded_local_adapter(adapter)
    if local is None:
        return {}
    row_keys, col_keys = local._axis_keys(
        device=local.row_keys.device,
        dtype=local.row_keys.dtype,
    )
    row_fixed_reference = F.normalize(
        local.row_key_projection(
            local.row_key_norm(
                local.fixed_row_keys.to(
                    device=local.row_keys.device,
                    dtype=local.row_keys.dtype,
                )
            )
        ),
        dim=-1,
    )
    col_fixed_reference = F.normalize(
        local.col_key_projection(
            local.col_key_norm(
                local.fixed_col_keys.to(
                    device=local.col_keys.device,
                    dtype=local.col_keys.dtype,
                )
            )
        ),
        dim=-1,
    )

    def axis_metrics(
        effective: torch.Tensor,
        fixed_reference: torch.Tensor,
        fixed: torch.Tensor,
        residual: torch.Tensor,
    ) -> dict[str, float]:
        fixed_rms = float(fixed.detach().float().square().mean().sqrt().cpu().item())
        residual_rms = float(
            residual.detach().float().square().mean().sqrt().cpu().item()
        )
        anchor_cosine = (
            F.normalize(effective.detach().float(), dim=-1)
            * F.normalize(fixed_reference.detach().float(), dim=-1)
        ).sum(dim=-1)
        return {
            **_key_cosine_metrics(effective),
            "fixed_rms": fixed_rms,
            "learned_residual_rms": residual_rms,
            "learned_to_fixed_rms_ratio": residual_rms / max(1.0e-12, fixed_rms),
            "fixed_anchor_cosine_mean": float(anchor_cosine.mean().cpu().item()),
            "fixed_anchor_cosine_min": float(anchor_cosine.min().cpu().item()),
        }

    raw_logit_scale = float(
        local.routing_logit_scale.detach().float().cpu().item()
    )
    logit_scale_limit = math.log(100.0)
    return {
        "row": axis_metrics(
            row_keys,
            row_fixed_reference,
            local.fixed_row_keys,
            local.row_keys,
        ),
        "col": axis_metrics(
            col_keys,
            col_fixed_reference,
            local.fixed_col_keys,
            local.col_keys,
        ),
        "routing_logit_scale": float(
            local.routing_logit_scale.detach().float().exp().clamp(max=100.0).cpu().item()
        ),
        "routing_logit_scale_log": raw_logit_scale,
        "routing_logit_scale_saturated": bool(raw_logit_scale >= logit_scale_limit),
        "routing_logit_scale_log_margin_to_clamp": logit_scale_limit - raw_logit_scale,
        "text_layer_weights": [
            float(value)
            for value in (
                torch.softmax(local.text_layer_logits.detach().float(), dim=0)
                .cpu()
                .tolist()
            )
        ],
    }


def _grounded_evidence_transform_parameters(adapter: nn.Module) -> list[nn.Parameter]:
    local = _grounded_local_adapter(adapter)
    if local is None:
        return []
    return [
        *local.evidence_down.parameters(),
        *local.evidence_up.parameters(),
    ]


def clear_grounded_evidence_transform_gradients(adapter: nn.Module) -> int:
    parameters = _grounded_evidence_transform_parameters(adapter)
    for parameter in parameters:
        parameter.grad = None
    return len(parameters)


def reset_grounded_evidence_optimizer_state(
    optimizer: torch.optim.Optimizer,
    adapter: nn.Module,
) -> int:
    cleared = 0
    for parameter in _grounded_evidence_transform_parameters(adapter):
        if parameter in optimizer.state:
            optimizer.state.pop(parameter)
            cleared += 1
    return cleared


def _region_indices(spec: Sequence[Any], height: int, width: int) -> list[int]:
    if len(spec) != 4:
        raise ValueError(f"Region routing target must be [row,col,height,width], got {spec}.")
    row, col, region_h, region_w = [int(value) for value in spec]
    if row < 0 or col < 0 or region_h <= 0 or region_w <= 0:
        raise ValueError(f"Invalid region routing target: {spec}.")
    if row + region_h > height or col + region_w > width:
        raise ValueError(f"Region routing target exceeds grid {(height, width)}: {spec}.")
    return [
        current_row * width + current_col
        for current_row in range(row, row + region_h)
        for current_col in range(col, col + region_w)
    ]


def _point_index(row: Any, col: Any, height: int, width: int) -> int:
    row_value, col_value = int(row), int(col)
    if not (0 <= row_value < height and 0 <= col_value < width):
        raise ValueError(
            f"Point routing target {(row_value, col_value)} exceeds grid {(height, width)}."
        )
    return row_value * width + col_value


def grounded_routing_loss(
    adapter: nn.Module,
    records: Sequence[Mapping[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    local = _grounded_local_adapter(adapter)
    if local is None or local.last_routing_logits is None or local.last_role_gate_logits is None:
        raise RuntimeError("Grounded routing outputs were not produced by the positive adapter forward.")
    logits = local.last_routing_logits
    row_logits = local.last_row_logits
    col_logits = local.last_col_logits
    gate_logits = local.last_role_gate_logits
    if row_logits is None or col_logits is None:
        raise RuntimeError("Grounded factorized row/column routing outputs are missing.")
    if int(logits.shape[0]) != len(records) or tuple(logits.shape[:2]) != tuple(gate_logits.shape):
        raise ValueError("Grounded routing outputs do not match the record batch.")
    if tuple(row_logits.shape[:2]) != tuple(logits.shape[:2]) or tuple(
        col_logits.shape[:2]
    ) != tuple(logits.shape[:2]):
        raise ValueError("Grounded row/column logits do not match the cell-routing batch.")
    height, width = local.latent_grid
    role_targets: list[list[list[int] | None]] = []
    for record in records:
        query_spec = grounding_query_spec_for_record(record)
        targets: list[list[int] | None] = [None] * int(local.soft_prompt_tokens)
        kind = str(query_spec["type"])
        if kind == "point":
            targets[0] = [
                _point_index(query_spec["row"], query_spec["col"], height, width)
            ]
        elif kind == "point_pair":
            for role, key in enumerate(("a", "b")):
                row, col = query_spec[key]
                targets[role] = [_point_index(row, col, height, width)]
        elif kind == "region_pair":
            targets[0] = _region_indices(query_spec["a"], height, width)
            targets[1] = _region_indices(query_spec["b"], height, width)
        for target in targets:
            if target is not None and any(index < 0 or index >= height * width for index in target):
                raise ValueError(f"Grounded routing target exceeds grid {(height, width)}.")
        role_targets.append(targets)

    routing_terms_by_record: list[list[torch.Tensor]] = [
        [] for _record in records
    ]
    routing_top1 = 0
    routing_top5 = 0
    routing_row_top1 = 0
    routing_col_top1 = 0
    target_mass = 0.0
    normalized_entropy_sum = 0.0
    active_roles = 0
    by_task: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "active_roles": 0.0,
            "top1_correct": 0.0,
            "top5_correct": 0.0,
            "row_top1_correct": 0.0,
            "col_top1_correct": 0.0,
            "target_mass_sum": 0.0,
            "normalized_entropy_sum": 0.0,
            "gate_correct": 0.0,
            "gate_predicted_active": 0.0,
            "gate_target_active": 0.0,
            "gate_slots": 0.0,
        }
    )
    gate_targets = torch.zeros_like(gate_logits, dtype=torch.float32)
    for row, targets in enumerate(role_targets):
        task = str(records[row].get("task_type", "unknown"))
        for role, target in enumerate(targets):
            if target is None:
                continue
            active_roles += 1
            gate_targets[row, role] = 1.0
            log_probs = F.log_softmax(logits[row, role].float(), dim=-1)
            target_tensor = torch.tensor(target, dtype=torch.long, device=logits.device)
            routing_terms_by_record[row].append(-log_probs[target_tensor].mean())
            prediction = int(torch.argmax(log_probs.detach()).item())
            top5 = torch.topk(log_probs.detach(), k=min(5, int(log_probs.numel()))).indices.tolist()
            routing_top1 += int(prediction in target)
            routing_top5 += int(any(index in target for index in top5))
            target_rows = {int(index) // width for index in target}
            target_cols = {int(index) % width for index in target}
            routing_row_top1 += int(
                int(torch.argmax(row_logits[row, role].detach()).item()) in target_rows
            )
            routing_col_top1 += int(
                int(torch.argmax(col_logits[row, role].detach()).item()) in target_cols
            )
            target_mass += float(log_probs.detach()[target_tensor].exp().sum().cpu().item())
            probabilities = log_probs.detach().exp()
            normalized_entropy = float(
                (
                    -(probabilities * log_probs.detach()).sum()
                    / math.log(max(2, int(probabilities.numel())))
                )
                .cpu()
                .item()
            )
            normalized_entropy_sum += normalized_entropy
            task_totals = by_task[task]
            task_totals["active_roles"] += 1.0
            task_totals["top1_correct"] += float(prediction in target)
            task_totals["top5_correct"] += float(any(index in target for index in top5))
            task_totals["row_top1_correct"] += float(
                int(torch.argmax(row_logits[row, role].detach()).item()) in target_rows
            )
            task_totals["col_top1_correct"] += float(
                int(torch.argmax(col_logits[row, role].detach()).item()) in target_cols
            )
            task_totals["target_mass_sum"] += float(
                log_probs.detach()[target_tensor].exp().sum().cpu().item()
            )
            task_totals["normalized_entropy_sum"] += normalized_entropy
    # Keep every record equally weighted even though comparison tasks activate two
    # role slots while point-value tasks activate one. Records without a routing
    # target contribute an exact zero; their role gate remains supervised below.
    record_routing_losses = [
        (
            torch.stack(terms).mean()
            if terms
            else logits[row].float().sum() * 0.0
        )
        for row, terms in enumerate(routing_terms_by_record)
    ]
    routing_loss = (
        torch.stack(record_routing_losses).mean()
        if record_routing_losses
        else logits.float().sum() * 0.0
    )
    gate_loss = F.binary_cross_entropy_with_logits(gate_logits.float(), gate_targets)
    gate_predictions = gate_logits.detach().float() >= 0.0
    gate_accuracy = float((gate_predictions == gate_targets.bool()).float().mean().cpu().item())
    for row, record in enumerate(records):
        task_totals = by_task[str(record.get("task_type", "unknown"))]
        task_totals["gate_correct"] += float(
            (gate_predictions[row] == gate_targets[row].bool()).float().sum().cpu().item()
        )
        task_totals["gate_predicted_active"] += float(
            gate_predictions[row].float().sum().cpu().item()
        )
        task_totals["gate_target_active"] += float(
            gate_targets[row].sum().cpu().item()
        )
        task_totals["gate_slots"] += float(gate_targets.shape[1])
    task_metrics = {}
    for task, totals in sorted(by_task.items()):
        task_active = max(1.0, totals["active_roles"])
        task_gate_slots = max(1.0, totals["gate_slots"])
        task_metrics[task] = {
            "active_roles": totals["active_roles"],
            "top1_accuracy": totals["top1_correct"] / task_active,
            "top5_accuracy": totals["top5_correct"] / task_active,
            "row_top1_accuracy": totals["row_top1_correct"] / task_active,
            "col_top1_accuracy": totals["col_top1_correct"] / task_active,
            "target_mass": totals["target_mass_sum"] / task_active,
            "normalized_entropy": totals["normalized_entropy_sum"] / task_active,
            "gate_accuracy": totals["gate_correct"] / task_gate_slots,
            "gate_active_fraction": totals["gate_predicted_active"] / task_gate_slots,
            "gate_target_active_fraction": totals["gate_target_active"] / task_gate_slots,
            "gate_slots": totals["gate_slots"],
        }
    return routing_loss, gate_loss, {
        "routing_active_roles": float(active_roles),
        "routing_top1_accuracy": routing_top1 / max(1, active_roles),
        "routing_top5_accuracy": routing_top5 / max(1, active_roles),
        "routing_row_top1_accuracy": routing_row_top1 / max(1, active_roles),
        "routing_col_top1_accuracy": routing_col_top1 / max(1, active_roles),
        "routing_target_mass": target_mass / max(1, active_roles),
        "routing_normalized_entropy": normalized_entropy_sum / max(1, active_roles),
        "routing_gate_accuracy": gate_accuracy,
        "routing_gate_active_fraction": float(
            gate_predictions.float().mean().cpu().item()
        ),
        "routing_gate_target_active_fraction": float(
            gate_targets.mean().cpu().item()
        ),
        "routing_by_task": task_metrics,
    }


def routing_metric_weighted_totals(
    metrics: Mapping[str, Any],
    *,
    record_count: int,
    gate_slots_per_record: int,
) -> dict[str, float]:
    """Convert routing means into additive totals for update/epoch reduction."""
    if int(record_count) < 0 or int(gate_slots_per_record) < 0:
        raise ValueError("Routing metric counts must be non-negative.")
    active_roles = float(metrics.get("routing_active_roles", 0.0))
    gate_slots = float(int(record_count) * int(gate_slots_per_record))
    return {
        "routing_active_roles": active_roles,
        "routing_top1_correct": float(metrics.get("routing_top1_accuracy", 0.0))
        * active_roles,
        "routing_top5_correct": float(metrics.get("routing_top5_accuracy", 0.0))
        * active_roles,
        "routing_row_top1_correct": float(
            metrics.get("routing_row_top1_accuracy", 0.0)
        )
        * active_roles,
        "routing_col_top1_correct": float(
            metrics.get("routing_col_top1_accuracy", 0.0)
        )
        * active_roles,
        "routing_target_mass_sum": float(metrics.get("routing_target_mass", 0.0))
        * active_roles,
        "routing_normalized_entropy_sum": float(
            metrics.get("routing_normalized_entropy", 0.0)
        )
        * active_roles,
        "routing_gate_correct": float(metrics.get("routing_gate_accuracy", 0.0))
        * gate_slots,
        "routing_gate_active": float(
            metrics.get("routing_gate_active_fraction", 0.0)
        )
        * gate_slots,
        "routing_gate_target_active": float(
            metrics.get("routing_gate_target_active_fraction", 0.0)
        )
        * gate_slots,
        "routing_gate_slots": gate_slots,
    }


def same_state_question_swap_indices(
    records: Sequence[Mapping[str, Any]],
    require_different_answers: bool = False,
    max_records_per_group: int | None = None,
) -> tuple[list[int], list[int]]:
    if max_records_per_group is not None and int(max_records_per_group) <= 0:
        raise ValueError("max_records_per_group must be positive when provided.")
    grouped: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        matched = record.get("matched_group")
        if not isinstance(matched, Mapping):
            continue
        margin_group_id = str(matched.get("margin_group_id") or "")
        option_hash = str(matched.get("option_set_sha256") or "")
        coordinate_set_id = str(matched.get("coordinate_set_id") or "")
        choices = record.get("choices")
        if (
            not margin_group_id
            or not option_hash
            or not coordinate_set_id
            or not isinstance(choices, Sequence)
            or isinstance(choices, (str, bytes))
        ):
            continue
        key = (
            str(record.get("state_ref", "")),
            str(record.get("task_type", "")),
            str(record.get("field") or record.get("metadata", {}).get("field") or ""),
            margin_group_id,
            option_hash,
            coordinate_set_id,
            tuple(str(value) for value in choices),
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
        group_owners: list[int] = []
        group_sources: list[int] = []
        for position, owner in enumerate(distinct):
            owner_question = str(records[owner].get("query") or records[owner].get("question") or "")
            owner_answer = str(records[owner].get("answer", ""))
            candidates = distinct[position + 1 :] + distinct[:position]
            candidate = next(
                (
                    other
                    for other in candidates
                    if owner_question
                    != str(records[other].get("query") or records[other].get("question") or "")
                    and (
                        not bool(require_different_answers)
                        or owner_answer != str(records[other].get("answer", ""))
                    )
                ),
                None,
            )
            if candidate is None:
                continue
            group_owners.append(owner)
            group_sources.append(candidate)
        if max_records_per_group is not None:
            group_owners = group_owners[: int(max_records_per_group)]
            group_sources = group_sources[: int(max_records_per_group)]
        owners.extend(group_owners)
        swapped.extend(group_sources)
    return owners, swapped


def matched_group_owner_mean(
    values: torch.Tensor,
    owners: Sequence[int],
    records: Sequence[Mapping[str, Any]],
) -> torch.Tensor:
    """Average owner values per matched group, then equally across groups."""

    flat = values.reshape(-1)
    if int(flat.numel()) != len(owners) or not owners:
        raise ValueError(
            "Matched-group owner mean requires one value per non-empty owner list: "
            f"values={int(flat.numel())}, owners={len(owners)}."
        )
    grouped: dict[tuple[str, str], list[torch.Tensor]] = defaultdict(list)
    for value, raw_owner in zip(flat, owners):
        owner = int(raw_owner)
        if owner < 0 or owner >= len(records):
            raise IndexError(
                f"Matched-group owner index {owner} is outside {len(records)} records."
            )
        record = records[owner]
        spec = record.get("matched_group")
        margin_group_id = (
            str(spec.get("margin_group_id") or "")
            if isinstance(spec, Mapping)
            else ""
        )
        if not margin_group_id:
            raise ValueError(
                f"Swap owner {record.get('qa_id', owner)!r} has no margin_group_id."
            )
        grouped[(str(record.get("state_ref", "")), margin_group_id)].append(value)
    return torch.stack(
        [torch.stack(group_values).mean() for group_values in grouped.values()]
    ).mean()


def question_swapped_soft_prefixes(
    adapter: nn.Module,
    *,
    positive_soft_embeds: torch.Tensor,
    positive_soft_attention_mask: torch.Tensor,
    positive_global_prompts: torch.Tensor | None,
    positive_local_prompts: torch.Tensor | None,
    owners: Sequence[int],
    sources: Sequence[int],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Swap question-conditioned evidence while retaining each owner's global tensor prefix."""

    if len(owners) != len(sources):
        raise ValueError("Question-swap owner/source lists must have equal length.")
    if tuple(positive_soft_attention_mask.shape) != tuple(positive_soft_embeds.shape[:2]):
        raise ValueError("Question-swap soft embeddings and attention masks do not match.")
    preserve_global = isinstance(adapter, HybridGlobalLocalAdapter) and not adapter.residual_mode
    if preserve_global and not (
        isinstance(positive_global_prompts, torch.Tensor)
        and isinstance(positive_local_prompts, torch.Tensor)
    ):
        raise RuntimeError(
            "A concatenated hybrid question swap is missing captured global/local prompts."
        )

    swapped_embeds: list[torch.Tensor] = []
    swapped_masks: list[torch.Tensor] = []
    for owner, source in zip(owners, sources):
        if preserve_global:
            local_count = int(adapter.local_adapter.soft_prompt_tokens)
            swapped_embeds.append(
                torch.cat(
                    [positive_local_prompts[source], positive_global_prompts[owner]],
                    dim=0,
                )
            )
            swapped_masks.append(
                torch.cat(
                    [
                        positive_soft_attention_mask[source, :local_count],
                        positive_soft_attention_mask[owner, local_count:],
                    ],
                    dim=0,
                )
            )
        else:
            swapped_embeds.append(positive_soft_embeds[source])
            swapped_masks.append(positive_soft_attention_mask[source])
    return swapped_embeds, swapped_masks


STAGE2B_TASK_TYPES = (
    "extreme_quadrant",
    "normalized_point_value",
    "point_compare",
    "raw_point_value_with_stats",
    "region_mean_compare",
)

POINT_VALUE_TASK_TYPES = (
    "normalized_point_value",
    "raw_point_value_with_stats",
)


def uses_screened_stage2b_training(args: argparse.Namespace) -> bool:
    return bool(
        getattr(args, "joint_ab_training", False)
        or getattr(args, "point_reader_training", False)
        or getattr(args, "full_local_reader_training", False)
    )


def inverse_frequency_task_weights(
    records: Sequence[Mapping[str, Any]],
    expected_tasks: Sequence[str] | None = None,
) -> dict[str, float]:
    """Return record weights whose aggregate contribution is equal per task."""

    counts: dict[str, int] = defaultdict(int)
    for record in records:
        counts[str(record.get("task_type", "unknown"))] += 1
    tasks = tuple(str(task) for task in (expected_tasks or sorted(counts)))
    if not records or not tasks:
        raise ValueError("Task balancing requires at least one record and one task.")
    missing = [task for task in tasks if counts.get(task, 0) <= 0]
    unexpected = sorted(set(counts) - set(tasks)) if expected_tasks is not None else []
    if missing or unexpected:
        raise ValueError(
            "Task-balanced answer training received an incomplete task set: "
            f"missing={missing}, unexpected={unexpected}, counts={dict(sorted(counts.items()))}."
        )
    total = float(sum(counts[task] for task in tasks))
    task_count = float(len(tasks))
    weights = {
        task: total / (task_count * float(counts[task]))
        for task in tasks
    }
    weighted_total = sum(float(counts[task]) * weights[task] for task in tasks)
    if not math.isclose(weighted_total, total, rel_tol=1.0e-9, abs_tol=1.0e-9):
        raise RuntimeError(
            "Inverse-frequency task weights do not preserve unit mean: "
            f"weighted_total={weighted_total}, records={total}."
        )
    return weights


def task_balanced_record_mean(
    values: torch.Tensor,
    records: Sequence[Mapping[str, Any]],
    task_weights: Mapping[str, float] | None,
) -> torch.Tensor:
    """Average per-record values without renormalizing away task weights in a batch."""

    flat = values.reshape(-1)
    if int(flat.numel()) != len(records) or not records:
        raise ValueError(
            "Task-balanced mean requires one scalar per non-empty record: "
            f"values={int(flat.numel())}, records={len(records)}."
        )
    if not task_weights:
        return flat.mean()
    weights: list[float] = []
    for record in records:
        task = str(record.get("task_type", "unknown"))
        if task not in task_weights:
            raise ValueError(f"No task-balance weight was configured for task_type={task!r}.")
        weights.append(float(task_weights[task]))
    weight_tensor = flat.new_tensor(weights)
    return (flat * weight_tensor).mean()


def masked_record_mean(values: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
    """Zero inactive tasks while preserving the outer per-record loss scale."""

    flat = values.reshape(-1)
    mask = active_mask.reshape(-1).to(device=flat.device, dtype=flat.dtype)
    if tuple(flat.shape) != tuple(mask.shape):
        raise ValueError(
            "Masked record mean requires identical value/mask shapes: "
            f"values={tuple(flat.shape)}, mask={tuple(mask.shape)}."
        )
    active = mask.sum()
    if float(active.detach().cpu().item()) <= 0.0:
        return flat.sum() * 0.0
    return (flat * mask).mean()


def correct_choice_margins(
    candidate_log_probs: Sequence[torch.Tensor],
    target_indices: Sequence[int],
) -> torch.Tensor:
    """Correct-option log-probability minus the strongest incorrect option."""

    if len(candidate_log_probs) != len(target_indices) or not candidate_log_probs:
        raise ValueError("Choice margins require one non-empty candidate distribution per target.")
    margins: list[torch.Tensor] = []
    for log_probs, raw_target in zip(candidate_log_probs, target_indices):
        target = int(raw_target)
        if log_probs.ndim != 1 or int(log_probs.numel()) < 2:
            raise ValueError("Choice margins require at least two candidates per record.")
        if target < 0 or target >= int(log_probs.numel()):
            raise IndexError(
                f"Choice target index {target} is outside {int(log_probs.numel())} candidates."
            )
        incorrect_mask = torch.ones_like(log_probs, dtype=torch.bool)
        incorrect_mask[target] = False
        margins.append(log_probs[target] - log_probs[incorrect_mask].max())
    return torch.stack(margins)


def margin_hinge_terms(
    primary_margins: torch.Tensor,
    baseline_margins: torch.Tensor,
    required_gain: float,
) -> torch.Tensor:
    """Penalize records whose primary correct-choice margin lacks the required gain."""

    if tuple(primary_margins.shape) != tuple(baseline_margins.shape):
        raise ValueError(
            "Margin hinge inputs must have identical shapes: "
            f"primary={tuple(primary_margins.shape)}, "
            f"baseline={tuple(baseline_margins.shape)}."
        )
    if not math.isfinite(float(required_gain)) or float(required_gain) < 0.0:
        raise ValueError("required_gain must be finite and non-negative.")
    return F.relu(
        float(required_gain) - (primary_margins - baseline_margins.detach())
    )


def snapshot_global_adapter_parameters(
    adapter: nn.Module,
) -> dict[str, torch.Tensor]:
    if not isinstance(adapter, HybridGlobalLocalAdapter):
        raise TypeError("A global-parameter anchor requires HybridGlobalLocalAdapter.")
    return {
        name: parameter.detach().clone()
        for name, parameter in adapter.global_adapter.named_parameters()
    }


def snapshot_local_adapter_parameters(
    adapter: nn.Module,
) -> dict[str, torch.Tensor]:
    if not isinstance(adapter, HybridGlobalLocalAdapter):
        raise TypeError("A local-parameter anchor requires HybridGlobalLocalAdapter.")
    return {
        name: parameter.detach().clone()
        for name, parameter in adapter.local_adapter.named_parameters()
    }


def _module_parameter_anchor_loss(
    module: nn.Module,
    reference: Mapping[str, torch.Tensor] | None,
    label: str,
) -> torch.Tensor:
    parameters = dict(module.named_parameters())
    if not reference or set(reference) != set(parameters):
        raise ValueError(
            f"The {label} anchor must contain every current parameter exactly once."
        )
    squared_error: torch.Tensor | None = None
    element_count = 0
    for name, parameter in parameters.items():
        parent = reference[name].to(device=parameter.device, dtype=torch.float32)
        current = parameter.float()
        if tuple(parent.shape) != tuple(current.shape):
            raise ValueError(
                f"{label.title()} anchor shape mismatch for {name}: "
                f"parent={tuple(parent.shape)}, current={tuple(current.shape)}."
            )
        tensor_error = (current - parent).square().sum()
        squared_error = tensor_error if squared_error is None else squared_error + tensor_error
        element_count += int(current.numel())
    if squared_error is None or element_count <= 0:
        raise ValueError(f"The {label} module has no parameters to anchor.")
    return squared_error / float(element_count)


def global_parameter_anchor_loss(
    adapter: nn.Module,
    reference: Mapping[str, torch.Tensor] | None,
) -> torch.Tensor:
    """Element-weighted mean-squared drift from the loaded parent global adapter."""

    if not isinstance(adapter, HybridGlobalLocalAdapter):
        raise TypeError("A global-parameter anchor requires HybridGlobalLocalAdapter.")
    return _module_parameter_anchor_loss(adapter.global_adapter, reference, "global adapter")


def local_parameter_anchor_loss(
    adapter: nn.Module,
    reference: Mapping[str, torch.Tensor] | None,
) -> torch.Tensor:
    if not isinstance(adapter, HybridGlobalLocalAdapter):
        raise TypeError("A local-parameter anchor requires HybridGlobalLocalAdapter.")
    return _module_parameter_anchor_loss(adapter.local_adapter, reference, "local reader")


def joint_global_view_training_loss(
    llm,
    adapter: TensorSoftPromptAdapter,
    tokenizer,
    batch: Mapping[str, Any],
    device: torch.device,
    args: argparse.Namespace,
    global_anchor_reference: Mapping[str, torch.Tensor] | None,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    """Stage-2A view: update only global parameters and return detached margins for B."""

    records = batch["records"]
    if not (
        bool(getattr(args, "joint_ab_training", False))
        and isinstance(adapter, HybridGlobalLocalAdapter)
        and not adapter.freeze_global
    ):
        raise ValueError("The joint global view requires a trainable hybrid global adapter.")
    latent_map = batch["latent_map"].to(device, non_blocking=True)
    global_soft_embeds = adapter.global_adapter.forward_soft_prompts(latent_map).to(
        dtype=llm.get_input_embeddings().weight.dtype
    )
    global_soft_attention_mask = _all_visible_soft_prompt_mask(global_soft_embeds)
    (
        _global_choice_mean,
        _global_answer_ce,
        global_choice_losses,
        _global_soft_embeds,
        global_choice_metrics,
    ) = choice_ce_loss(
        llm=llm,
        adapter=adapter,
        tokenizer=tokenizer,
        records=records,
        latent_map=latent_map,
        device=device,
        args=args,
        soft_prompt_mode="global_only",
        precomputed_soft_embeds=global_soft_embeds,
        precomputed_soft_attention_mask=global_soft_attention_mask,
    )
    task_weights = (
        getattr(args, "task_loss_weights", None)
        if bool(getattr(args, "task_balanced_answer_loss", False))
        else None
    )
    global_view_loss = task_balanced_record_mean(
        global_choice_losses,
        records,
        task_weights,
    )
    anchor_loss = global_parameter_anchor_loss(adapter, global_anchor_reference)
    weighted_global_view = float(args.global_view_loss_weight) * global_view_loss
    weighted_anchor = float(args.global_anchor_loss_weight) * anchor_loss
    total = weighted_global_view + weighted_anchor
    target_indices = global_choice_metrics.get("candidate_target_indices")
    candidate_log_probs = global_choice_metrics.get("candidate_log_probs")
    if not isinstance(target_indices, Sequence) or not isinstance(
        candidate_log_probs, Sequence
    ):
        raise RuntimeError("The joint global view did not return restricted-choice margins.")
    margins = correct_choice_margins(candidate_log_probs, target_indices).detach()
    return total, {
        "loss": float(total.detach().cpu().item()),
        "global_view_loss": float(global_view_loss.detach().cpu().item()),
        "weighted_global_view_loss": float(weighted_global_view.detach().cpu().item()),
        "global_view_accuracy": float(global_choice_metrics["choice_accuracy"]),
        "global_anchor_loss": float(anchor_loss.detach().cpu().item()),
        "weighted_global_anchor_loss": float(weighted_anchor.detach().cpu().item()),
    }, margins


def training_loss(
    llm,
    adapter: TensorSoftPromptAdapter,
    tokenizer,
    dataset: TensorReadoutQADataset,
    batch: Mapping[str, Any],
    device: torch.device,
    args: argparse.Namespace,
    routing_only: bool = False,
    global_anchor_reference: Mapping[str, torch.Tensor] | None = None,
    joint_global_margins: torch.Tensor | None = None,
    joint_global_accuracy: float | None = None,
    local_anchor_reference: Mapping[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    records = batch["records"]
    joint_ab_training = bool(getattr(args, "joint_ab_training", False))
    point_reader_training = bool(getattr(args, "point_reader_training", False))
    reference_training = joint_ab_training or point_reader_training
    if joint_ab_training and not (
        isinstance(adapter, HybridGlobalLocalAdapter)
        and _grounded_local_adapter(adapter) is not None
        and not adapter.freeze_global
    ):
        raise ValueError(
            "joint_ab_training requires a grounded evidence adapter with a trainable global branch."
        )
    if point_reader_training and not (
        isinstance(adapter, HybridGlobalLocalAdapter)
        and _grounded_local_adapter(adapter) is not None
        and adapter.freeze_global
    ):
        raise ValueError(
            "point_reader_training requires grounded evidence with a frozen global branch."
        )
    task_weights = (
        getattr(args, "task_loss_weights", None)
        if bool(getattr(args, "task_balanced_answer_loss", False))
        else None
    )
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
    ce_weight = 0.0 if routing_only else float(args.ce_loss_weight)
    choice_ce_weight = 0.0 if routing_only else float(args.choice_ce_loss_weight)
    ranking_weight = 0.0 if routing_only else float(args.ranking_loss_weight)
    swapped_weight = 0.0 if routing_only else float(args.swapped_question_loss_weight)
    routing_weight = float(
        args.grounding_routing_loss_weight
        if routing_only
        else args.grounding_joint_routing_loss_weight
    )
    gate_weight = float(args.grounding_gate_loss_weight)
    matched_group_weight = 0.0 if routing_only else float(args.matched_group_loss_weight)
    choice_metrics = {
        "choice_accuracy": 0.0,
        "choice_01_loss": 0.0,
        "choice_single_token_path": 0.0,
    }
    answer_objective_active = any(
        weight > 0.0
        for weight in (
            ce_weight,
            choice_ce_weight,
            ranking_weight,
            swapped_weight,
            matched_group_weight,
        )
    )
    if not answer_objective_active:
        if _grounded_local_adapter(adapter) is None:
            raise ValueError("Routing-only training requires grounded_evidence_adapter.")
        soft_embeds = contextual_adapter_soft_embeds(
            llm=llm,
            adapter=adapter,
            tokenizer=tokenizer,
            records=records,
            latent_map=batch["latent_map"],
            device=device,
            max_prompt_tokens=int(args.max_prompt_tokens),
            layer_index=int(args.local_context_layer),
            mode="correct",
            prompt_template=str(args.prompt_template),
            precomputed_question_context=question_context,
        )
        if soft_embeds is None:
            raise RuntimeError("Grounded routing-only forward did not produce soft embeddings.")
        routing_loss, gate_loss, routing_metrics = grounded_routing_loss(adapter, records)
        # Keep every trainable reader parameter in the distributed gradient schema. Evidence
        # transforms receive exact zero gradients until the answer objective is enabled.
        graph_anchor = soft_embeds.sum() * 0.0
        total_loss = (
            routing_weight * routing_loss
            + gate_weight * gate_loss
            + graph_anchor
        )
        zero = routing_loss.new_zeros(())
        return total_loss, {
            "loss": float(total_loss.detach().cpu().item()),
            "ce_loss": 0.0,
            "weighted_ce_loss": 0.0,
            "choice_ce_loss": 0.0,
            "weighted_choice_ce_loss": 0.0,
            "choice_accuracy": 0.0,
            "choice_01_loss": 0.0,
            "choice_single_token_path": 0.0,
            "ranking_loss": 0.0,
            "weighted_ranking_loss": 0.0,
            "ranking_margin_mean": 0.0,
            "swapped_question_loss": 0.0,
            "weighted_swapped_question_loss": 0.0,
            "swapped_question_pairs": 0.0,
            "swapped_question_margin_mean": 0.0,
            "routing_loss": float(routing_loss.detach().cpu().item()),
            "weighted_routing_loss": float((routing_weight * routing_loss).detach().cpu().item()),
            "routing_gate_loss": float(gate_loss.detach().cpu().item()),
            "weighted_routing_gate_loss": float((gate_weight * gate_loss).detach().cpu().item()),
            "matched_group_loss": float(zero.cpu().item()),
            "weighted_matched_group_loss": float(zero.cpu().item()),
            "matched_group_count": 0.0,
            "matched_group_records": 0.0,
            "matched_group_pairs": 0.0,
            "matched_group_gap_mean": 0.0,
            "matched_group_satisfaction": 0.0,
            "matched_group_exact_accuracy": 0.0,
            **routing_metrics,
        }
    answers = [str(record["answer"]) for record in records]
    if choice_ce_weight > 0.0:
        (
            choice_loss_value,
            ce_loss,
            positive_choice_nll,
            positive_soft_embeds,
            choice_metrics,
        ) = choice_ce_loss(
            llm=llm,
            adapter=adapter,
            tokenizer=tokenizer,
            records=records,
            latent_map=batch["latent_map"],
            device=device,
            args=args,
            soft_prompt_mode="correct",
            precomputed_question_context=question_context,
            detach_global_for_local=reference_training,
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
            detach_global_for_local=reference_training,
        )
        choice_loss_value = ce_loss.new_zeros(())
        positive_choice_nll = None
        positive_soft_embeds = None

    if positive_choice_nll is None:
        (
            _choice_loss,
            _answer_ce,
            positive_choice_nll,
            positive_soft_embeds,
            choice_metrics,
        ) = choice_ce_loss(
            llm=llm,
            adapter=adapter,
            tokenizer=tokenizer,
            records=records,
            latent_map=batch["latent_map"],
            device=device,
            args=args,
            soft_prompt_mode="correct",
            precomputed_question_context=question_context,
            detach_global_for_local=reference_training,
        )
    unbalanced_choice_loss_value = positive_choice_nll.mean()
    choice_loss_value = task_balanced_record_mean(
        positive_choice_nll,
        records,
        task_weights,
    )
    positive_soft_attention_mask = choice_metrics.get("soft_attention_mask")
    if positive_soft_embeds is not None and not isinstance(
        positive_soft_attention_mask, torch.Tensor
    ):
        positive_soft_attention_mask = grounded_soft_prompt_attention_mask(
            adapter,
            positive_soft_embeds,
            mode="correct",
        )
    positive_global_prompts = getattr(adapter, "_last_global_prompts", None)
    positive_local_prompts = getattr(adapter, "_last_local_prompts", None)
    candidate_log_probs = choice_metrics.get("candidate_log_probs")
    if not isinstance(candidate_log_probs, Sequence) or len(candidate_log_probs) != len(records):
        raise RuntimeError("Positive restricted-choice scoring did not return candidate log-probabilities.")
    matched_group_loss, matched_group_metrics = matched_coordinate_group_loss(
        records=records,
        candidate_log_probs=candidate_log_probs,
        margin=float(args.matched_group_loss_margin),
        task_weights=task_weights,
    )
    if _grounded_local_adapter(adapter) is not None:
        routing_loss, gate_loss, routing_metrics = grounded_routing_loss(adapter, records)
    else:
        routing_loss = positive_choice_nll.new_zeros(())
        gate_loss = positive_choice_nll.new_zeros(())
        routing_metrics = {
            "routing_active_roles": 0.0,
            "routing_top1_accuracy": 0.0,
            "routing_top5_accuracy": 0.0,
            "routing_row_top1_accuracy": 0.0,
            "routing_col_top1_accuracy": 0.0,
            "routing_target_mass": 0.0,
            "routing_normalized_entropy": 0.0,
            "routing_gate_accuracy": 0.0,
            "routing_gate_active_fraction": 0.0,
            "routing_gate_target_active_fraction": 0.0,
        }
    global_view_loss = positive_choice_nll.new_zeros(())
    no_harm_loss = positive_choice_nll.new_zeros(())
    causal_loss = positive_choice_nll.new_zeros(())
    global_anchor_loss = positive_choice_nll.new_zeros(())
    local_anchor_loss = positive_choice_nll.new_zeros(())
    global_view_accuracy = 0.0
    no_harm_margin_mean = 0.0
    causal_margin_mean = 0.0
    causal_active_records = 0.0
    if reference_training:
        if joint_ab_training and (
            joint_global_margins is None
            or int(joint_global_margins.numel()) != len(records)
        ):
            raise ValueError(
                "The local joint A/B view requires one detached global-only margin per record."
            )
        with torch.no_grad():
            (
                _zero_choice_mean,
                _zero_answer_ce,
                _zero_choice_losses,
                _zero_soft_embeds,
                zero_choice_metrics,
            ) = choice_ce_loss(
                llm=llm,
                adapter=adapter,
                tokenizer=tokenizer,
                records=records,
                latent_map=batch["latent_map"],
                device=device,
                args=args,
                soft_prompt_mode="zero_local",
                precomputed_question_context=question_context,
                detach_global_for_local=True,
            )
        positive_targets = choice_metrics.get("candidate_target_indices")
        zero_targets = zero_choice_metrics.get("candidate_target_indices")
        if not all(
            isinstance(values, Sequence)
            for values in (positive_targets, zero_targets)
        ):
            raise RuntimeError("Reference choice scoring did not return target indices.")
        positive_margins = correct_choice_margins(
            candidate_log_probs,
            positive_targets,
        )
        zero_margins = correct_choice_margins(
            zero_choice_metrics["candidate_log_probs"],
            zero_targets,
        )
        if point_reader_training:
            point_tasks = set(parse_csv(args.point_causal_tasks))
            point_mask = positive_margins.new_tensor(
                [float(str(record.get("task_type", "")) in point_tasks) for record in records]
            )
            nonpoint_mask = 1.0 - point_mask
            causal_gap = positive_margins - zero_margins.detach()
            causal_terms = margin_hinge_terms(
                positive_margins,
                zero_margins,
                float(args.point_causal_margin),
            )
            causal_loss = masked_record_mean(causal_terms, point_mask)
            no_harm_terms = margin_hinge_terms(
                positive_margins,
                zero_margins,
                float(args.nonpoint_no_harm_margin),
            )
            no_harm_loss = masked_record_mean(no_harm_terms, nonpoint_mask)
            no_harm_gap = positive_margins - zero_margins.detach()
            no_harm_active = float(nonpoint_mask.sum().detach().cpu().item())
            no_harm_margin_mean = float(
                (no_harm_gap.detach() * nonpoint_mask).sum().cpu().item()
                / max(1.0, no_harm_active)
            )
            active_local = point_mask
        else:
            assert joint_global_margins is not None
            no_harm_gap = positive_margins - joint_global_margins.detach()
            no_harm_terms = margin_hinge_terms(
                positive_margins,
                joint_global_margins,
                float(args.joint_no_harm_margin),
            )
            no_harm_loss = task_balanced_record_mean(
                no_harm_terms,
                records,
                task_weights,
            )
            active_local = positive_margins.new_tensor(
                [
                    float(
                        _GROUNDING_ACTIVE_ROLES_BY_TYPE[
                            str(grounding_query_spec_for_record(record)["type"])
                        ]
                        > 0
                    )
                    for record in records
                ]
            )
            causal_gap = positive_margins - zero_margins.detach()
            causal_terms = margin_hinge_terms(
                positive_margins,
                zero_margins,
                float(args.joint_causal_margin),
            ) * active_local
            causal_loss = task_balanced_record_mean(
                causal_terms,
                records,
                task_weights,
            )
            local_anchor_loss = local_parameter_anchor_loss(
                adapter,
                local_anchor_reference,
            )
            global_view_accuracy = float(joint_global_accuracy or 0.0)
            no_harm_margin_mean = float(no_harm_gap.detach().mean().cpu().item())
        causal_margin_mean = float(
            (causal_gap.detach() * active_local).sum().cpu().item()
            / max(1.0, float(active_local.sum().cpu().item()))
        )
        causal_active_records = float(active_local.sum().detach().cpu().item())
    ranking_loss = positive_choice_nll.new_zeros(())
    ranking_margin_mean = 0.0
    swapped_loss = positive_choice_nll.new_zeros(())
    swapped_metrics = {"swapped_question_pairs": 0.0, "swapped_question_margin_mean": 0.0}
    combined_records: list[Mapping[str, Any]] = []
    combined_latents: list[torch.Tensor] = []
    combined_soft_embeds: list[torch.Tensor] = []
    combined_soft_attention_masks: list[torch.Tensor] = []
    ranking_count = 0
    negative_mode = str(args.ranking_loss_negative)
    if ranking_weight > 0.0:
        if negative_mode == "global_only":
            negative_latents = batch["latent_map"]
        elif negative_mode == "shuffled":
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
        negative_soft_attention_mask = (
            grounded_soft_prompt_attention_mask(
                adapter,
                negative_soft_embeds,
                mode=negative_mode,
            )
            if negative_soft_embeds is not None
            else None
        )
        if negative_soft_embeds is None:
            (
                _negative_loss,
                _negative_ce,
                negative_choice_nll,
                _negative_soft,
                _negative_metrics,
            ) = choice_ce_loss(
                llm=llm,
                adapter=adapter,
                tokenizer=tokenizer,
                records=records,
                latent_map=negative_latents,
                device=device,
                args=args,
                soft_prompt_mode=negative_mode,
                precomputed_question_context=question_context,
            )
            ranking_terms = F.relu(
                float(args.ranking_loss_margin) + positive_choice_nll - negative_choice_nll
            )
            ranking_loss = ranking_terms.mean()
            ranking_margin_mean = float(
                (negative_choice_nll.detach() - positive_choice_nll.detach()).mean().cpu().item()
            )
        else:
            ranking_count = len(records)
            combined_records.extend(records)
            combined_latents.extend(negative_latents)
            combined_soft_embeds.extend(negative_soft_embeds)
            if negative_soft_attention_mask is None:
                raise RuntimeError("A precomputed ranking prefix is missing its attention mask.")
            combined_soft_attention_masks.extend(negative_soft_attention_mask)

    swap_owners: list[int] = []
    if swapped_weight > 0.0 and positive_soft_embeds is not None:
        swap_owners, swap_sources = same_state_question_swap_indices(
            records,
            require_different_answers=bool(args.swapped_question_require_different_answer),
            max_records_per_group=int(args.swapped_question_max_records),
        )
        if not isinstance(positive_soft_attention_mask, torch.Tensor):
            raise RuntimeError("A precomputed swapped prefix is missing its attention mask.")
        swapped_soft_embeds, swapped_soft_masks = question_swapped_soft_prefixes(
            adapter,
            positive_soft_embeds=positive_soft_embeds,
            positive_soft_attention_mask=positive_soft_attention_mask,
            positive_global_prompts=(
                positive_global_prompts
                if isinstance(positive_global_prompts, torch.Tensor)
                else None
            ),
            positive_local_prompts=(
                positive_local_prompts
                if isinstance(positive_local_prompts, torch.Tensor)
                else None
            ),
            owners=swap_owners,
            sources=swap_sources,
        )
        for owner, swapped_soft, swapped_mask in zip(
            swap_owners, swapped_soft_embeds, swapped_soft_masks
        ):
            combined_records.append(records[owner])
            combined_latents.append(batch["latent_map"][owner])
            combined_soft_embeds.append(swapped_soft)
            combined_soft_attention_masks.append(swapped_mask)

    if combined_records:
        if len(combined_soft_attention_masks) != len(combined_records):
            raise RuntimeError(
                "Every precomputed counterfactual prefix must carry its original attention mask."
            )
        combined_choice_nll_chunks: list[torch.Tensor] = []
        grounding_batch_size = max(1, int(args.train_grounding_batch_size))
        for start in range(0, len(combined_records), grounding_batch_size):
            end = min(len(combined_records), start + grounding_batch_size)
            chunk_records = combined_records[start:end]
            chunk_latents = torch.stack(combined_latents[start:end], dim=0)
            chunk_soft_embeds = torch.stack(combined_soft_embeds[start:end], dim=0)
            chunk_soft_attention_mask = torch.stack(
                combined_soft_attention_masks[start:end], dim=0
            )
            (
                _chunk_loss,
                _chunk_ce,
                chunk_choice_nll,
                _chunk_soft,
                _chunk_metrics,
            ) = choice_ce_loss(
                llm=llm,
                adapter=adapter,
                tokenizer=tokenizer,
                records=chunk_records,
                latent_map=chunk_latents,
                device=device,
                args=args,
                soft_prompt_mode="correct",
                precomputed_soft_embeds=chunk_soft_embeds,
                precomputed_soft_attention_mask=chunk_soft_attention_mask,
            )
            combined_choice_nll_chunks.append(
                chunk_choice_nll
            )
        combined_choice_nll = torch.cat(combined_choice_nll_chunks, dim=0)
        if ranking_count > 0:
            negative_choice_nll = combined_choice_nll[:ranking_count]
            ranking_terms = F.relu(
                float(args.ranking_loss_margin) + positive_choice_nll - negative_choice_nll
            )
            ranking_loss = ranking_terms.mean()
            ranking_margin_mean = float(
                (negative_choice_nll.detach() - positive_choice_nll.detach()).mean().cpu().item()
            )
        if swap_owners:
            swapped_choice_nll = combined_choice_nll[
                ranking_count : ranking_count + len(swap_owners)
            ]
            selected_positive = torch.stack([positive_choice_nll[index] for index in swap_owners])
            swap_margin = swapped_choice_nll - selected_positive
            swapped_loss = matched_group_owner_mean(
                F.relu(float(args.swapped_question_loss_margin) - swap_margin),
                swap_owners,
                records,
            )
            swapped_metrics = {
                "swapped_question_pairs": float(len(swap_owners)),
                "swapped_question_margin_mean": float(
                    matched_group_owner_mean(
                        swap_margin.detach(),
                        swap_owners,
                        records,
                    ).cpu().item()
                ),
            }
    weighted_ce_loss = ce_weight * ce_loss
    weighted_choice_ce_loss = choice_ce_weight * choice_loss_value
    weighted_ranking_loss = ranking_weight * ranking_loss
    weighted_swapped_loss = swapped_weight * swapped_loss
    weighted_routing_loss = routing_weight * routing_loss
    weighted_gate_loss = gate_weight * gate_loss
    weighted_matched_group_loss = matched_group_weight * matched_group_loss
    weighted_global_view_loss = float(getattr(args, "global_view_loss_weight", 0.0)) * global_view_loss
    no_harm_weight = float(
        getattr(args, "nonpoint_no_harm_loss_weight", 0.0)
        if point_reader_training
        else getattr(args, "joint_no_harm_loss_weight", 0.0)
    )
    causal_weight = float(
        getattr(args, "point_causal_loss_weight", 0.0)
        if point_reader_training
        else getattr(args, "joint_causal_loss_weight", 0.0)
    )
    weighted_no_harm_loss = no_harm_weight * no_harm_loss
    weighted_causal_loss = causal_weight * causal_loss
    weighted_global_anchor_loss = (
        float(getattr(args, "global_anchor_loss_weight", 0.0)) * global_anchor_loss
    )
    weighted_local_anchor_loss = (
        float(getattr(args, "local_anchor_loss_weight", 0.0)) * local_anchor_loss
    )
    total_loss = (
        weighted_ce_loss
        + weighted_choice_ce_loss
        + weighted_ranking_loss
        + weighted_swapped_loss
        + weighted_routing_loss
        + weighted_gate_loss
        + weighted_matched_group_loss
        + weighted_global_view_loss
        + weighted_no_harm_loss
        + weighted_causal_loss
        + weighted_global_anchor_loss
        + weighted_local_anchor_loss
    )
    return total_loss, {
        "loss": float(total_loss.detach().cpu().item()),
        "ce_loss": float(ce_loss.detach().cpu().item()),
        "weighted_ce_loss": float(weighted_ce_loss.detach().cpu().item()),
        "choice_ce_loss": float(choice_loss_value.detach().cpu().item()),
        "choice_ce_loss_unbalanced": float(
            unbalanced_choice_loss_value.detach().cpu().item()
        ),
        "weighted_choice_ce_loss": float(weighted_choice_ce_loss.detach().cpu().item()),
        "choice_accuracy": float(choice_metrics["choice_accuracy"]),
        "choice_01_loss": float(choice_metrics["choice_01_loss"]),
        "choice_single_token_path": float(choice_metrics.get("choice_single_token_path", 0.0)),
        "ranking_loss": float(ranking_loss.detach().cpu().item()),
        "weighted_ranking_loss": float(weighted_ranking_loss.detach().cpu().item()),
        "ranking_margin_mean": ranking_margin_mean,
        "swapped_question_loss": float(swapped_loss.detach().cpu().item()),
        "weighted_swapped_question_loss": float(weighted_swapped_loss.detach().cpu().item()),
        "routing_loss": float(routing_loss.detach().cpu().item()),
        "weighted_routing_loss": float(weighted_routing_loss.detach().cpu().item()),
        "routing_gate_loss": float(gate_loss.detach().cpu().item()),
        "weighted_routing_gate_loss": float(weighted_gate_loss.detach().cpu().item()),
        "matched_group_loss": float(matched_group_loss.detach().cpu().item()),
        "weighted_matched_group_loss": float(weighted_matched_group_loss.detach().cpu().item()),
        "global_view_loss": float(global_view_loss.detach().cpu().item()),
        "weighted_global_view_loss": float(weighted_global_view_loss.detach().cpu().item()),
        "global_view_accuracy": global_view_accuracy,
        "joint_no_harm_loss": float(no_harm_loss.detach().cpu().item()),
        "weighted_joint_no_harm_loss": float(weighted_no_harm_loss.detach().cpu().item()),
        "joint_no_harm_margin_mean": no_harm_margin_mean,
        "joint_causal_loss": float(causal_loss.detach().cpu().item()),
        "weighted_joint_causal_loss": float(weighted_causal_loss.detach().cpu().item()),
        "joint_causal_margin_mean": causal_margin_mean,
        "joint_causal_active_records": causal_active_records,
        "global_anchor_loss": float(global_anchor_loss.detach().cpu().item()),
        "weighted_global_anchor_loss": float(weighted_global_anchor_loss.detach().cpu().item()),
        "local_anchor_loss": float(local_anchor_loss.detach().cpu().item()),
        "weighted_local_anchor_loss": float(weighted_local_anchor_loss.detach().cpu().item()),
        **routing_metrics,
        **matched_group_metrics,
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
    precomputed_soft_attention_mask: torch.Tensor | None = None,
) -> list[float]:
    require_precomputed_grounded_attention_mask(
        adapter,
        precomputed_soft_embeds,
        precomputed_soft_attention_mask,
    )
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
    soft_attention = grounded_soft_prompt_attention_mask(
        adapter,
        soft_embeds,
        mode=soft_prompt_mode,
        dtype=text_attention_mask.dtype,
        precomputed=precomputed_soft_attention_mask,
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
    soft_attention_mask: torch.Tensor | None,
    device: torch.device,
    prompt_template: str,
    max_prompt_tokens: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    if soft_attention_mask is None:
        raise ValueError(
            "Diagnostic generation requires the attention mask captured with its soft prefix."
        )
    prompt = build_prompt(record, prompt_template=prompt_template)
    prompt_ids = tokenizer(prompt, add_special_tokens=True, truncation=False)["input_ids"]
    if len(prompt_ids) > int(max_prompt_tokens):
        prompt_ids = prompt_ids[-int(max_prompt_tokens) :]
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    text_embeds = llm.get_input_embeddings()(input_ids)
    soft_embeds = soft_embeds.to(device=device, dtype=text_embeds.dtype)
    inputs_embeds = torch.cat([soft_embeds, text_embeds], dim=1)
    soft_attention = soft_attention_mask.to(device=device, dtype=torch.long)
    if tuple(soft_attention.shape) != tuple(soft_embeds.shape[:2]):
        raise ValueError(
            "Diagnostic soft-prefix mask does not match embeddings: "
            f"mask={tuple(soft_attention.shape)}, embeds={tuple(soft_embeds.shape[:2])}."
        )
    attention_mask = torch.cat(
        [soft_attention, torch.ones(input_ids.shape, dtype=torch.long, device=device)],
        dim=1,
    )
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
    soft_attention = grounded_soft_prompt_attention_mask(
        adapter,
        soft_embeds,
        mode=mode,
        dtype=text_attention_mask.dtype,
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
    base_soft_attention_mask = (
        grounded_soft_prompt_attention_mask(
            adapter,
            base_soft_embeds,
            mode=mode,
        )
        if base_soft_embeds is not None
        else None
    )
    candidate_soft_embeds = (
        [base_soft_embeds[index] for index in candidate_owner] if base_soft_embeds is not None else None
    )
    candidate_soft_attention_masks = (
        [base_soft_attention_mask[index] for index in candidate_owner]
        if base_soft_attention_mask is not None
        else None
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
            precomputed_soft_attention_mask=(
                torch.stack(candidate_soft_attention_masks[start:end], dim=0)
                if candidate_soft_attention_masks is not None
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
    if mode in {
        "correct",
        "global_only",
        "zero_local",
        "local_only",
        "no_latent",
        "shuffled_stats",
    }:
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
    routing_only: bool = False,
) -> dict[str, Any]:
    if routing_only and list(baseline_modes) != ["correct"]:
        raise ValueError("Routing-only validation supports exactly the correct latent mode.")
    if routing_only and _grounded_local_adapter(adapter) is None:
        raise ValueError("Routing-only validation requires grounded_evidence_adapter.")
    adapter_was_training = bool(adapter.training)
    llm_checkpoint_training = frozen_llm_checkpoint_execution_active(llm)
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
        # Evaluation loaders are short-lived. Do not retain worker-local dataset/cache copies
        # after each baseline pass.
        persistent_workers=False,
        prefetch_factor=1 if int(args.num_workers) > 0 else None,
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
        routing_totals = {
            "active_roles": 0.0,
            "top1_correct": 0.0,
            "top5_correct": 0.0,
            "row_top1_correct": 0.0,
            "col_top1_correct": 0.0,
            "target_mass_sum": 0.0,
            "normalized_entropy_sum": 0.0,
            "gate_correct": 0.0,
            "gate_active": 0.0,
            "gate_target_active": 0.0,
            "gate_slots": 0.0,
        }
        routing_tasks = sorted(
            {str(record.get("task_type", "unknown")) for record in dataset.records}
        )
        routing_by_task_totals = {
            task: {
                "active_roles": 0.0,
                "top1_correct": 0.0,
                "top5_correct": 0.0,
                "row_top1_correct": 0.0,
                "col_top1_correct": 0.0,
                "target_mass_sum": 0.0,
                "normalized_entropy_sum": 0.0,
                "gate_correct": 0.0,
                "gate_active": 0.0,
                "gate_target_active": 0.0,
                "gate_slots": 0.0,
            }
            for task in routing_tasks
        }
        for batch in tqdm(
            loader,
            desc=f"Eval [{mode}]",
            leave=False,
            disable=not bool(args.console_progress),
        ):
            records = records_for_baseline(mode, batch, dataset)
            latents = baseline_latents(mode, batch, dataset)
            if routing_only:
                soft_embeds = contextual_adapter_soft_embeds(
                    llm=llm,
                    adapter=adapter,
                    tokenizer=tokenizer,
                    records=records,
                    latent_map=latents,
                    device=device,
                    max_prompt_tokens=int(args.max_prompt_tokens),
                    layer_index=int(args.local_context_layer),
                    mode="correct",
                    prompt_template=str(args.prompt_template),
                )
                if soft_embeds is None:
                    raise RuntimeError("Routing-only validation did not produce grounded soft embeddings.")
                predictions: Sequence[str] = ()
            else:
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
            if mode == "correct" and _grounded_local_adapter(adapter) is not None:
                _routing_loss, _gate_loss, routing_metrics = grounded_routing_loss(
                    adapter,
                    records,
                )
                active_roles = float(routing_metrics["routing_active_roles"])
                gate_slots = float(
                    len(records) * int(_grounded_local_adapter(adapter).soft_prompt_tokens)
                )
                routing_totals["active_roles"] += active_roles
                routing_totals["top1_correct"] += (
                    float(routing_metrics["routing_top1_accuracy"]) * active_roles
                )
                routing_totals["top5_correct"] += (
                    float(routing_metrics["routing_top5_accuracy"]) * active_roles
                )
                routing_totals["row_top1_correct"] += (
                    float(routing_metrics["routing_row_top1_accuracy"]) * active_roles
                )
                routing_totals["col_top1_correct"] += (
                    float(routing_metrics["routing_col_top1_accuracy"]) * active_roles
                )
                routing_totals["target_mass_sum"] += (
                    float(routing_metrics["routing_target_mass"]) * active_roles
                )
                routing_totals["normalized_entropy_sum"] += (
                    float(routing_metrics["routing_normalized_entropy"]) * active_roles
                )
                routing_totals["gate_correct"] += (
                    float(routing_metrics["routing_gate_accuracy"]) * gate_slots
                )
                routing_totals["gate_active"] += (
                    float(routing_metrics["routing_gate_active_fraction"]) * gate_slots
                )
                routing_totals["gate_target_active"] += (
                    float(routing_metrics["routing_gate_target_active_fraction"])
                    * gate_slots
                )
                routing_totals["gate_slots"] += gate_slots
                task_routing = routing_metrics.get("routing_by_task", {})
                if not isinstance(task_routing, Mapping):
                    raise TypeError("Grounded routing_by_task metrics must be a mapping.")
                for task, task_metrics in task_routing.items():
                    if task not in routing_by_task_totals or not isinstance(task_metrics, Mapping):
                        raise ValueError(f"Unexpected grounded routing task metrics for {task!r}.")
                    totals = routing_by_task_totals[task]
                    task_active = float(task_metrics.get("active_roles", 0.0))
                    task_gate_slots = float(task_metrics.get("gate_slots", 0.0))
                    totals["active_roles"] += task_active
                    totals["top1_correct"] += float(task_metrics.get("top1_accuracy", 0.0)) * task_active
                    totals["top5_correct"] += float(task_metrics.get("top5_accuracy", 0.0)) * task_active
                    totals["row_top1_correct"] += (
                        float(task_metrics.get("row_top1_accuracy", 0.0)) * task_active
                    )
                    totals["col_top1_correct"] += (
                        float(task_metrics.get("col_top1_accuracy", 0.0)) * task_active
                    )
                    totals["target_mass_sum"] += float(task_metrics.get("target_mass", 0.0)) * task_active
                    totals["normalized_entropy_sum"] += (
                        float(task_metrics.get("normalized_entropy", 0.0)) * task_active
                    )
                    totals["gate_correct"] += float(task_metrics.get("gate_accuracy", 0.0)) * task_gate_slots
                    totals["gate_active"] += (
                        float(task_metrics.get("gate_active_fraction", 0.0))
                        * task_gate_slots
                    )
                    totals["gate_target_active"] += (
                        float(task_metrics.get("gate_target_active_fraction", 0.0))
                        * task_gate_slots
                    )
                    totals["gate_slots"] += task_gate_slots
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
        if mode == "correct" and _grounded_local_adapter(adapter) is not None:
            routing_totals = distributed_sum_scalars(routing_totals, device=device)
            routing_by_task_totals = {
                task: distributed_sum_scalars(totals, device=device)
                for task, totals in routing_by_task_totals.items()
            }
            active_roles = max(1.0, routing_totals["active_roles"])
            gate_slots = max(1.0, routing_totals["gate_slots"])
            metrics[mode]["routing"] = {
                "active_roles": int(round(routing_totals["active_roles"])),
                "top1_accuracy": routing_totals["top1_correct"] / active_roles,
                "top5_accuracy": routing_totals["top5_correct"] / active_roles,
                "row_top1_accuracy": routing_totals["row_top1_correct"] / active_roles,
                "col_top1_accuracy": routing_totals["col_top1_correct"] / active_roles,
                "target_mass": routing_totals["target_mass_sum"] / active_roles,
                "normalized_entropy": routing_totals["normalized_entropy_sum"]
                / active_roles,
                "gate_accuracy": routing_totals["gate_correct"] / gate_slots,
                "gate_active_fraction": routing_totals["gate_active"] / gate_slots,
                "gate_target_active_fraction": routing_totals[
                    "gate_target_active"
                ]
                / gate_slots,
                "by_task": {
                    task: {
                        "active_roles": int(round(totals["active_roles"])),
                        "top1_accuracy": totals["top1_correct"]
                        / max(1.0, totals["active_roles"]),
                        "top5_accuracy": totals["top5_correct"]
                        / max(1.0, totals["active_roles"]),
                        "row_top1_accuracy": totals["row_top1_correct"]
                        / max(1.0, totals["active_roles"]),
                        "col_top1_accuracy": totals["col_top1_correct"]
                        / max(1.0, totals["active_roles"]),
                        "target_mass": totals["target_mass_sum"]
                        / max(1.0, totals["active_roles"]),
                        "normalized_entropy": totals["normalized_entropy_sum"]
                        / max(1.0, totals["active_roles"]),
                        "gate_accuracy": totals["gate_correct"]
                        / max(1.0, totals["gate_slots"]),
                        "gate_active_fraction": totals["gate_active"]
                        / max(1.0, totals["gate_slots"]),
                        "gate_target_active_fraction": totals[
                            "gate_target_active"
                        ]
                        / max(1.0, totals["gate_slots"]),
                    }
                    for task, totals in sorted(routing_by_task_totals.items())
                },
            }
    if routing_only:
        metrics["evaluation_mode"] = "routing_only_shallow_qwen"
    adapter.train(adapter_was_training)
    set_frozen_llm_execution_mode(llm, checkpoint_training=llm_checkpoint_training)
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


def _resolved_diagnostic_layers(
    requested_layers: Sequence[int],
    hidden_state_count: int,
) -> list[int]:
    if not requested_layers:
        raise ValueError("At least one diagnostic hidden-state layer must be requested.")
    invalid = [
        int(value)
        for value in requested_layers
        if not -int(hidden_state_count) <= int(value) < int(hidden_state_count)
    ]
    if invalid:
        raise ValueError(
            f"Diagnostic hidden-state layers {invalid} are invalid for "
            f"{int(hidden_state_count)} returned states."
        )
    return sorted(
        {
            int(value) if int(value) >= 0 else int(hidden_state_count) + int(value)
            for value in requested_layers
        }
    )


@torch.no_grad()
def _decoder_question_last_hidden(
    decoder: nn.Module,
    soft_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    text_mask: torch.Tensor,
    prompt_mask: torch.Tensor,
    requested_layers: Sequence[int],
    soft_attention_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if soft_embeds.ndim != 3 or text_embeds.ndim != 3:
        raise ValueError(
            "Diagnostic decoder inputs must be [batch,tokens,hidden] tensors: "
            f"soft={tuple(soft_embeds.shape)}, text={tuple(text_embeds.shape)}."
        )
    if int(soft_embeds.shape[0]) != 1 or int(text_embeds.shape[0]) != 1:
        raise ValueError("Question-last diagnostics currently require a single record.")
    if text_mask.ndim != 2 or prompt_mask.shape != text_mask.shape:
        raise ValueError(
            "Question-last diagnostics require matching [1,tokens] text and prompt masks: "
            f"text_mask={tuple(text_mask.shape)}, prompt_mask={tuple(prompt_mask.shape)}."
        )
    prompt_positions = torch.nonzero(
        prompt_mask[0].bool() & text_mask[0].bool(),
        as_tuple=False,
    ).flatten()
    if prompt_positions.numel() == 0:
        raise ValueError(
            "Question-last diagnostics found no prompt token. The QA prompt must contain at least "
            "one unmasked natural-language token before the answer target."
        )
    inputs = torch.cat([soft_embeds, text_embeds], dim=1)
    soft_attention = (
        torch.ones(
            (1, int(soft_embeds.shape[1])),
            dtype=text_mask.dtype,
            device=text_mask.device,
        )
        if soft_attention_mask is None
        else soft_attention_mask.to(device=text_mask.device, dtype=text_mask.dtype)
    )
    if tuple(soft_attention.shape) != tuple(soft_embeds.shape[:2]):
        raise ValueError("Question-last soft attention mask does not match the soft embeddings.")
    attention = torch.cat([soft_attention, text_mask], dim=1)
    outputs = decoder(
        inputs_embeds=inputs,
        attention_mask=attention,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("The diagnostic decoder did not return hidden states.")
    resolved_layers = _resolved_diagnostic_layers(requested_layers, len(hidden_states))
    question_last = int(soft_embeds.shape[1]) + int(prompt_positions[-1].item())
    return {
        str(layer): hidden_states[layer][0, question_last].detach().float().cpu()
        for layer in resolved_layers
    }


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

    local_groups: list[dict[str, Any]] | None = None
    global_groups: list[dict[str, Any]] | None = None
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
        grounded_text_block = getattr(adapter.local_adapter, "text_block", None)
        if grounded_text_block is not None:
            handles.append(
                grounded_text_block.register_forward_hook(capture("grounded_role_states"))
            )
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
            *(
                [adapter.local_adapter.text_block]
                if isinstance(adapter.local_adapter, GroundedEvidenceAdapter)
                else []
            ),
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
            if isinstance(adapter.local_adapter, GroundedEvidenceAdapter):
                grounded = adapter.local_adapter
                for name, value in (
                    ("grounded_routing_weights", grounded.last_routing_weights),
                    ("grounded_row_logits", grounded.last_row_logits),
                    ("grounded_col_logits", grounded.last_col_logits),
                    ("grounded_role_gate_logits", grounded.last_role_gate_logits),
                ):
                    if isinstance(value, torch.Tensor):
                        trace[name] = value[0].detach().float().cpu()
        else:
            soft = adapter(
                latent,
                question_embeds=text_embeds,
                question_mask=prompt_mask,
                structured_query=structured_query,
            )
        soft_attention_mask = grounded_soft_prompt_attention_mask(
            adapter,
            soft,
            mode="correct",
        )
        trace["soft_attention_mask"] = soft_attention_mask[0].detach().cpu()
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
            if isinstance(adapter.local_adapter, GroundedEvidenceAdapter):
                grounded_text_block = adapter.local_adapter.text_block
                if grounded_text_block.last_attention_weights is not None:
                    trace["grounded_text_attention"] = grounded_text_block.last_attention_weights
                grounded_text_block.capture_attention = False
                grounded_text_block.last_attention_weights = None
        for handle in handles:
            handle.remove()
    return soft, trace


@torch.no_grad()
def _run_embedded_diagnostics_impl(
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
            soft_attention = adapter_hidden.get("soft_attention_mask")
            if not isinstance(soft_attention, torch.Tensor):
                raise RuntimeError("Embedded diagnostics did not preserve the soft-prefix mask.")
            attention = torch.cat(
                [soft_attention.unsqueeze(0).to(device=device, dtype=text_mask.dtype), text_mask],
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
            resolved_layers = _resolved_diagnostic_layers(requested_layers, len(hidden_states))
            prompt_positions = torch.nonzero(
                prompt_mask[0].bool() & text_mask[0].bool(),
                as_tuple=False,
            ).flatten()
            if prompt_positions.numel() == 0:
                raise ValueError(
                    "Embedded diagnostics found no natural-language prompt token before the answer target."
                )
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
                soft_attention_mask=(
                    mode_states[mode]["adapter_hidden"]["soft_attention_mask"].unsqueeze(0)
                ),
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
            grounded_routing = correct_state["adapter_hidden"].get(
                "grounded_routing_weights"
            )
            if isinstance(grounded_routing, torch.Tensor):
                local = adapter.local_adapter
                if not isinstance(local, GroundedEvidenceAdapter):
                    raise TypeError("Grounded diagnostic tensors require GroundedEvidenceAdapter.")
                height, width = local.latent_grid
                row_logits = correct_state["adapter_hidden"].get("grounded_row_logits")
                col_logits = correct_state["adapter_hidden"].get("grounded_col_logits")
                gate_logits = correct_state["adapter_hidden"].get(
                    "grounded_role_gate_logits"
                )
                text_attention = correct_state["adapter_hidden"].get(
                    "grounded_text_attention"
                )
                role_text_weights = (
                    text_attention[0].mean(dim=0)
                    if isinstance(text_attention, torch.Tensor) and text_attention.ndim == 4
                    else None
                )
                grounded_roles: list[dict[str, Any]] = []
                for role in range(int(grounded_routing.shape[0])):
                    weights = grounded_routing[role].float()
                    top_count = min(8, int(weights.numel()))
                    cell_values, cell_indices = torch.topk(weights, k=top_count)
                    role_payload: dict[str, Any] = {
                        "role": role,
                        "gate_probability": (
                            float(torch.sigmoid(gate_logits[role]).item())
                            if isinstance(gate_logits, torch.Tensor)
                            else None
                        ),
                        "normalized_entropy": float(
                            (-(weights.clamp_min(1.0e-12) * weights.clamp_min(1.0e-12).log()).sum()
                             / math.log(max(2, int(weights.numel())))).item()
                        ),
                        "top_cells": [
                            {
                                "index": int(index),
                                "row": int(index) // width,
                                "col": int(index) % width,
                                "weight": float(value),
                            }
                            for value, index in zip(
                                cell_values.tolist(), cell_indices.tolist()
                            )
                        ],
                    }
                    for axis, logits in (("rows", row_logits), ("cols", col_logits)):
                        if isinstance(logits, torch.Tensor):
                            probabilities = torch.softmax(logits[role].float(), dim=-1)
                            axis_count = min(5, int(probabilities.numel()))
                            values, indices = torch.topk(probabilities, k=axis_count)
                            role_payload[f"top_{axis}"] = [
                                {"index": int(index), "probability": float(value)}
                                for value, index in zip(values.tolist(), indices.tolist())
                            ]
                    if isinstance(role_text_weights, torch.Tensor):
                        token_weights = role_text_weights[role]
                        token_count = min(8, int(token_weights.numel()))
                        values, indices = torch.topk(token_weights, k=token_count)
                        role_payload["top_text_tokens"] = [
                            {
                                "index": int(token_index),
                                "token": str(
                                    tokenizer.convert_ids_to_tokens(
                                        int(local_ids[0, token_index].item())
                                    )
                                ),
                                "text": tokenizer.decode(
                                    [int(local_ids[0, token_index].item())]
                                ),
                                "weight": float(value),
                            }
                            for value, token_index in zip(values.tolist(), indices.tolist())
                        ]
                    grounded_roles.append(role_payload)
                local_attention_summary["grounded_roles"] = grounded_roles
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
            alt_llm_text_embeds = llm.get_input_embeddings()(alt_ids)
            alt_llm_prompt_mask = alt_labels.eq(IGNORE_INDEX) & alt_mask.bool()
            alt_text_embeds = alt_llm_text_embeds
            alt_prompt_mask = alt_llm_prompt_mask
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
            alt_soft_model = alt_soft.to(dtype=alt_llm_text_embeds.dtype)
            alt_soft = alt_soft_model[0].detach().float().cpu()
            alt_question_last = _decoder_question_last_hidden(
                decoder=decoder,
                soft_embeds=alt_soft_model,
                text_embeds=alt_llm_text_embeds,
                text_mask=alt_mask,
                prompt_mask=alt_llm_prompt_mask,
                requested_layers=requested_layers,
                soft_attention_mask=alt_trace["soft_attention_mask"].unsqueeze(0),
            )
            same_latent_question_last = {
                layer: _cosine_and_relative_l2(
                    correct_state["layers"][layer]["question_last"],
                    hidden,
                )
                for layer, hidden in alt_question_last.items()
                if layer in correct_state["layers"]
            }
            question_sensitivity = {
                "alternate_qa_id": str(alternate_record.get("qa_id", "")),
                "alternate_task_type": str(alternate_record.get("task_type", "unknown")),
                "same_latent_soft_prompt": _cosine_and_relative_l2(correct_state["soft_prompt"], alt_soft),
                "same_latent_question_last_by_layer": same_latent_question_last,
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
                precomputed_soft_attention_mask=same_trace[
                    "soft_attention_mask"
                ].unsqueeze(0).repeat(len(choices), 1),
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
        if bool(args.diagnostics_save_states):
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
    question_hidden_sensitive_records = [
        item
        for item in question_sensitive_records
        if isinstance(item["question_sensitivity"].get("same_latent_question_last_by_layer"), Mapping)
        and item["question_sensitivity"]["same_latent_question_last_by_layer"]
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
    conditioned_backbone_trainable_parameters = 0
    question_conditioning_trainable_parameters = 0
    if isinstance(adapter, HybridGlobalLocalAdapter):
        local_gate = getattr(adapter.local_adapter, "gate", None)
        if isinstance(local_gate, torch.Tensor) and int(local_gate.numel()) == 1:
            residual_gate = float(local_gate.detach().float().cpu().item())
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
        if isinstance(adapter.local_adapter, ResidualQuestionConditionedAdapter):
            conditioned_backbone_trainable_parameters = sum(
                int(parameter.numel())
                for parameter in adapter.local_adapter.backbone.parameters()
                if parameter.requires_grad
            )
            question_conditioning_trainable_parameters = sum(
                int(parameter.numel())
                for name, parameter in adapter.local_adapter.named_parameters()
                if not name.startswith("backbone.") and parameter.requires_grad
            )
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
        "same_latent_different_question_question_last_relative_l2_by_layer": {
            layer: sum(
                item["question_sensitivity"]["same_latent_question_last_by_layer"][layer]["relative_l2"]
                for item in question_hidden_sensitive_records
                if layer in item["question_sensitivity"]["same_latent_question_last_by_layer"]
            )
            / max(
                1,
                sum(
                    layer in item["question_sensitivity"]["same_latent_question_last_by_layer"]
                    for item in question_hidden_sensitive_records
                ),
            )
            for layer in sorted(
                {
                    layer
                    for item in question_hidden_sensitive_records
                    for layer in item["question_sensitivity"]["same_latent_question_last_by_layer"]
                },
                key=int,
            )
        },
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
        "conditioned_backbone_trainable_parameters": conditioned_backbone_trainable_parameters,
        "question_conditioning_trainable_parameters": question_conditioning_trainable_parameters,
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
    question_last_by_layer = aggregate[
        "same_latent_different_question_question_last_relative_l2_by_layer"
    ]
    aggregate["same_latent_different_question_question_last_relative_l2_mean"] = (
        sum(float(value) for value in question_last_by_layer.values())
        / max(1, len(question_last_by_layer))
    )
    summary = {
        "stage": stage,
        "aggregate": aggregate,
        "records": summaries,
        "state_file": (
            str(diagnostic_dir / f"{stage}_states.pt")
            if bool(args.diagnostics_save_states)
            else None
        ),
        "state_float_dtype": "float16" if bool(args.diagnostics_save_states) else None,
        "structured_query_conditioning": bool(getattr(adapter, "structured_query_conditioning", False)),
    }
    atomic_dump_json(diagnostic_dir / f"{stage}_summary.json", summary)
    if bool(args.diagnostics_save_states):
        atomic_torch_save(
            diagnostic_dir / f"{stage}_states.pt",
            compact_diagnostic_tensors(tensor_payload),
        )
    return summary


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
    """Run diagnostics without leaking eval mode into the training loop."""
    adapter_was_training = bool(adapter.training)
    llm_checkpoint_training = frozen_llm_checkpoint_execution_active(llm)
    try:
        return _run_embedded_diagnostics_impl(
            stage=stage,
            llm=llm,
            adapter=adapter,
            tokenizer=tokenizer,
            dataset=dataset,
            device=device,
            args=args,
            run_dir=run_dir,
        )
    finally:
        adapter.train(adapter_was_training)
        set_frozen_llm_execution_mode(
            llm,
            checkpoint_training=llm_checkpoint_training,
        )


def save_adapter_checkpoint(
    path: str | Path,
    adapter: nn.Module,
    args: argparse.Namespace,
    latent_shape: Sequence[int],
    llm_hidden_size: int,
    latent_contract: Mapping[str, Any],
    metrics: Mapping[str, Any] | None = None,
) -> None:
    payload = {
        "checkpoint_type": "tensor_llm_adapter",
        "checkpoint_version": 2,
        "adapter_state_dict": adapter.state_dict(),
        "args": redacted_args(args),
        "latent_shape_chw": list(int(dim) for dim in latent_shape),
        "llm_hidden_size": int(llm_hidden_size),
        "latent_contract": dict(latent_contract),
        "lineage": {
            "stage2_warm_start_checkpoint": getattr(args, "stage2_warm_start_checkpoint", None),
            "stage2_warm_start_sha256": getattr(args, "stage2_warm_start_sha256", None),
            "stage2b_resume_checkpoint": getattr(args, "stage2b_resume_checkpoint", None),
            "stage2b_resume_sha256": getattr(args, "stage2b_resume_sha256", None),
        },
        "metrics": dict(metrics or {}),
    }
    atomic_torch_save(path, payload)


def normalized_latent_contract_identity(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Return the immutable latent identity without host-specific path aliases."""
    identity = copy.deepcopy(dict(contract))
    # The same checkpoint can be mounted through different logical/physical
    # paths after migration. Its verified SHA-256 is the immutable identity;
    # the persisted path remains provenance but must not define compatibility.
    checkpoint_path = str(identity.pop("alignment_checkpoint", "")).strip()
    if not checkpoint_path:
        raise ValueError("Latent contract is missing alignment_checkpoint provenance.")
    checkpoint_sha = str(identity.get("alignment_checkpoint_sha256", "")).lower()
    if len(checkpoint_sha) != 64 or any(
        character not in "0123456789abcdef" for character in checkpoint_sha
    ):
        raise ValueError("Latent contract has an invalid alignment_checkpoint_sha256.")
    identity["alignment_checkpoint_sha256"] = checkpoint_sha
    normalization = identity.get("encoder_input_normalization")
    if isinstance(normalization, Mapping):
        identity["encoder_input_normalization"] = canonical_normalization(normalization)
    latent_shape = identity.get("latent_shape")
    if isinstance(latent_shape, Sequence) and not isinstance(latent_shape, (str, bytes)):
        identity["latent_shape"] = [int(value) for value in latent_shape]
    return identity


def validate_latent_contract_compatibility(
    observed_contract: Mapping[str, Any],
    expected_contract: Mapping[str, Any],
) -> None:
    observed_identity = normalized_latent_contract_identity(observed_contract)
    expected_identity = normalized_latent_contract_identity(expected_contract)
    missing = object()
    differing_keys = sorted(
        key
        for key in set(observed_identity) | set(expected_identity)
        if observed_identity.get(key, missing) != expected_identity.get(key, missing)
    )
    if differing_keys:
        def display(mapping: Mapping[str, Any], key: str) -> Any:
            return mapping[key] if key in mapping else "<missing>"

        differences = "; ".join(
            f"{key}: checkpoint={display(observed_identity, key)!r}, "
            f"active={display(expected_identity, key)!r}"
            for key in differing_keys
        )
        raise ValueError(
            "Stage-2 adapter checkpoint was trained against a different latent/Stage-1 "
            f"contract; differing_keys={differing_keys}; {differences}."
        )


def audit_stage2_warm_start_checkpoint(
    path: str | Path,
    expected_latent_contract: Mapping[str, Any],
    expected_latent_channel_policy: str = "all",
) -> dict[str, Any]:
    """Validate the warm-start envelope before loading the frozen LLM replicas."""
    checkpoint_path = Path(path).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Stage-2 warm-start checkpoint must contain a mapping payload.")
    try:
        llm_hidden_size = int(checkpoint.get("llm_hidden_size", -1))
    except (TypeError, ValueError) as exc:
        raise ValueError("Stage-2 warm-start checkpoint has an invalid LLM hidden width.") from exc
    if llm_hidden_size <= 0:
        raise ValueError("Stage-2 warm-start checkpoint has no positive LLM hidden width.")
    expected_shape = expected_latent_contract.get("latent_shape")
    if not isinstance(expected_shape, Sequence) or isinstance(expected_shape, (str, bytes)):
        raise ValueError("Active latent contract has no valid latent_shape for warm-start audit.")
    state_dict = validate_adapter_checkpoint_payload(
        checkpoint,
        expected_latent_shape=expected_shape,
        expected_llm_hidden_size=llm_hidden_size,
        expected_architecture="alignment_adapter",
        expected_latent_contract=expected_latent_contract,
        expected_latent_channel_policy=expected_latent_channel_policy,
    )
    return {
        "validated_before_llm_load": True,
        "path": str(checkpoint_path),
        "sha256": sha256_file(checkpoint_path),
        "architecture": "alignment_adapter",
        "latent_shape": [int(value) for value in expected_shape],
        "llm_hidden_size": llm_hidden_size,
        "parameter_tensors": sum(
            int(isinstance(value, torch.Tensor)) for value in state_dict.values()
        ),
        "parameters": sum(
            int(value.numel())
            for value in state_dict.values()
            if isinstance(value, torch.Tensor)
        ),
    }


def validate_adapter_checkpoint_payload(
    checkpoint: Any,
    *,
    expected_latent_shape: Sequence[int],
    expected_llm_hidden_size: int,
    expected_architecture: str,
    expected_latent_contract: Mapping[str, Any],
    expected_latent_channel_policy: str = "all",
) -> Mapping[str, torch.Tensor]:
    """Validate that a saved Stage-2 checkpoint is standalone and belongs to this run."""
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Stage-2 adapter checkpoint must contain a mapping payload.")
    if str(checkpoint.get("checkpoint_type", "")) != "tensor_llm_adapter":
        raise ValueError("Stage-2 adapter checkpoint has an invalid checkpoint_type.")
    try:
        checkpoint_version = int(checkpoint.get("checkpoint_version", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError("Stage-2 adapter checkpoint_version must be an integer.") from exc
    if checkpoint_version < 2:
        raise ValueError(
            "Stage-2 adapter checkpoint lacks the version-2 latent provenance envelope."
        )
    state_dict = checkpoint.get("adapter_state_dict")
    checkpoint_args = checkpoint.get("args")
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("Stage-2 adapter checkpoint has no adapter_state_dict.")
    if not isinstance(checkpoint_args, Mapping):
        raise ValueError("Stage-2 adapter checkpoint has no args mapping.")
    non_tensor_keys = [str(key) for key, value in state_dict.items() if not torch.is_tensor(value)]
    if non_tensor_keys:
        raise ValueError(
            "Stage-2 adapter state_dict contains non-tensor values: "
            f"{non_tensor_keys[:8]}."
        )
    non_finite_keys = [
        str(key)
        for key, value in state_dict.items()
        if value.is_floating_point() and not bool(torch.isfinite(value).all())
    ]
    if non_finite_keys:
        raise FloatingPointError(
            "Stage-2 adapter checkpoint contains NaN or infinity: "
            f"{non_finite_keys[:8]}."
        )
    observed_shape = checkpoint.get("latent_shape_chw")
    if not isinstance(observed_shape, Sequence) or isinstance(observed_shape, (str, bytes)):
        raise ValueError("Stage-2 adapter checkpoint has no valid latent_shape_chw.")
    observed_shape_values = tuple(int(value) for value in observed_shape)
    expected_shape_values = tuple(int(value) for value in expected_latent_shape)
    if observed_shape_values != expected_shape_values:
        raise ValueError(
            "Stage-2 adapter checkpoint latent shape mismatch: "
            f"observed={observed_shape_values}, expected={expected_shape_values}."
        )
    if int(checkpoint.get("llm_hidden_size", -1)) != int(expected_llm_hidden_size):
        raise ValueError(
            "Stage-2 adapter checkpoint LLM hidden width does not match the active model."
        )
    observed_architecture = str(checkpoint_args.get("adapter_architecture", ""))
    if observed_architecture != str(expected_architecture):
        raise ValueError(
            "Stage-2 adapter checkpoint architecture mismatch: "
            f"observed={observed_architecture!r}, expected={str(expected_architecture)!r}."
        )
    observed_latent_channel_policy = str(checkpoint_args.get("latent_channel_policy", "all"))
    if observed_latent_channel_policy != str(expected_latent_channel_policy):
        raise ValueError(
            "Stage-2 adapter checkpoint latent channel policy mismatch: "
            f"observed={observed_latent_channel_policy!r}, "
            f"expected={str(expected_latent_channel_policy)!r}."
        )
    observed_contract = checkpoint.get("latent_contract")
    if not isinstance(observed_contract, Mapping):
        raise ValueError("Stage-2 adapter checkpoint is missing latent_contract provenance.")
    validate_latent_contract_compatibility(observed_contract, expected_latent_contract)
    return state_dict


def _validated_optional_sha256(value: Any, *, label: str) -> str | None:
    normalized = str(value or "").strip().lower()
    if not normalized:
        return None
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{label} is not a valid SHA-256 digest.")
    return normalized


def validate_stage2b_continuation_contract(
    resume_args: Mapping[str, Any],
    *,
    resume_lineage: Mapping[str, Any] | None,
    current_args: argparse.Namespace,
) -> dict[str, Any]:
    """Validate continuation semantics that strict state loading cannot see."""

    resume_direct_sha = _validated_optional_sha256(
        resume_args.get("stage2_warm_start_sha256")
        or (resume_lineage or {}).get("stage2_warm_start_sha256"),
        label="Stage-2B continuation direct-parent SHA",
    )
    current_direct_sha = _validated_optional_sha256(
        getattr(current_args, "stage2_warm_start_sha256", None),
        label="Configured direct-parent SHA",
    )
    resume_direct_path = _configured_checkpoint_path(
        resume_args.get("stage2_warm_start_checkpoint")
        or (resume_lineage or {}).get("stage2_warm_start_checkpoint")
    )
    current_direct_path = _configured_checkpoint_path(
        getattr(current_args, "stage2_warm_start_checkpoint", None)
    )
    if resume_direct_sha is not None and current_direct_sha is not None:
        if resume_direct_sha != current_direct_sha:
            raise ValueError(
                "Stage-2B continuation and configured run reference different "
                "direct Stage-2 checkpoint contents."
            )
        direct_parent_identity = "sha256"
    else:
        if (
            resume_direct_path is None
            or current_direct_path is None
            or resume_direct_path != current_direct_path
        ):
            raise ValueError(
                "Stage-2B continuation lacks a comparable direct-parent SHA and its "
                "resolved direct Stage-2 parent path differs from the configured run."
            )
        direct_parent_identity = "legacy_resolved_path"

    try:
        resume_context_layers = tuple(
            int(value) for value in parse_csv(resume_args.get("local_context_layers"))
        )
        current_context_layers = tuple(
            int(value) for value in parse_csv(current_args.local_context_layers)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Stage-2B continuation has an invalid local_context_layers contract.") from exc

    exact_semantics = {
        "local_context_layers": (resume_context_layers, current_context_layers),
        "local_soft_prompt_tokens": (
            int(resume_args.get("local_soft_prompt_tokens", -1)),
            int(current_args.local_soft_prompt_tokens),
        ),
        "adapter_dim": (
            int(resume_args.get("adapter_dim", -1)),
            int(current_args.adapter_dim),
        ),
        "adapter_heads": (
            int(resume_args.get("adapter_heads", -1)),
            int(current_args.adapter_heads),
        ),
        "local_fusion_mode": (
            str(resume_args.get("local_fusion_mode", "")),
            str(current_args.local_fusion_mode),
        ),
    }
    mismatches = {
        name: values for name, values in exact_semantics.items() if values[0] != values[1]
    }
    float_semantics = {
        "soft_prompt_scale": (
            float(resume_args.get("soft_prompt_scale", math.nan)),
            float(current_args.soft_prompt_scale),
        ),
        "dropout": (
            float(resume_args.get("dropout", math.nan)),
            float(current_args.dropout),
        ),
    }
    mismatches.update(
        {
            name: values
            for name, values in float_semantics.items()
            if not all(math.isfinite(value) for value in values)
            or not math.isclose(values[0], values[1], rel_tol=0.0, abs_tol=1.0e-12)
        }
    )
    if mismatches:
        rendered = "; ".join(
            f"{name}: checkpoint={observed!r}, active={active!r}"
            for name, (observed, active) in sorted(mismatches.items())
        )
        raise ValueError(
            "Stage-2B continuation changes non-state reader semantics; " + rendered
        )

    resume_inactive_mask = bool(resume_args.get("mask_inactive_local_tokens", False))
    current_inactive_mask = bool(current_args.mask_inactive_local_tokens)
    if resume_inactive_mask and not current_inactive_mask:
        raise ValueError(
            "Stage-2B continuation cannot disable the checkpoint's inactive local-token mask."
        )
    mask_policy = (
        "legacy_to_learned_gate_mask"
        if current_inactive_mask and not resume_inactive_mask
        else "unchanged"
    )

    resume_stage1_path = _configured_checkpoint_path(
        resume_args.get("adapter_init_checkpoint")
    )
    current_stage1_path = _configured_checkpoint_path(
        getattr(current_args, "adapter_init_checkpoint", None)
    )
    return {
        "direct_parent_identity": direct_parent_identity,
        "direct_parent_sha256": resume_direct_sha or current_direct_sha,
        "resume_direct_parent_path": (
            str(resume_direct_path) if resume_direct_path is not None else None
        ),
        "configured_direct_parent_path": (
            str(current_direct_path) if current_direct_path is not None else None
        ),
        # Stage-1 content identity is already enforced by latent-contract SHA.
        "stage1_parent_paths_match": resume_stage1_path == current_stage1_path,
        "inactive_local_token_mask_policy": mask_policy,
        "semantic_contract": {
            **{name: observed for name, (observed, _active) in exact_semantics.items()},
            **{name: observed for name, (observed, _active) in float_semantics.items()},
        },
    }


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


def validate_frozen_global_resume_state(
    resume_state: Mapping[str, torch.Tensor],
    direct_parent_state: Mapping[str, torch.Tensor],
) -> dict[str, int]:
    """Prove that a grounded child did not mutate its claimed frozen global parent."""

    resume_keys = sorted(
        str(key) for key in resume_state if str(key).startswith("global_adapter.")
    )
    parent_keys = sorted(
        str(key) for key in direct_parent_state if str(key).startswith("global_adapter.")
    )
    if resume_keys != parent_keys:
        raise ValueError(
            "Stage-2B continuation global tensor keys differ from the audited direct parent."
        )
    changed = [
        key
        for key in parent_keys
        if not torch.equal(
            resume_state[key].detach().cpu(),
            direct_parent_state[key].detach().cpu(),
        )
    ]
    if changed:
        raise ValueError(
            "Stage-2B continuation rewrote tensors in its claimed frozen global parent: "
            f"changed_keys={changed[:8]}."
        )
    return {
        "verified_global_parameter_tensors": len(parent_keys),
        "changed_global_parameter_tensors": 0,
    }


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
            "grounded_evidence_adapter",
        }
        else ""
    )
    global_adapter_type = str(
        ckpt_args.get("global_adapter_type", ckpt_args.get("adapter_type", "qformer"))
    ).lower()
    if architecture == "grounded_evidence_adapter" and global_adapter_type != "spatial_transformer":
        raise ValueError(
            "Grounded evidence checkpoints require a spatial_transformer global adapter whose "
            "row-major tokens remain aligned one-to-one with latent cells."
        )
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

    if architecture == "grounded_evidence_adapter":
        role_queries = state_dict.get("local_adapter.role_queries")
        if not isinstance(role_queries, torch.Tensor):
            raise ValueError("Grounded evidence checkpoint is missing local_adapter.role_queries.")
        local_adapter = GroundedEvidenceAdapter(
            latent_grid=tuple(int(value) for value in latent_shape[-2:]),
            llm_hidden_size=llm_hidden_size,
            context_layers=[
                int(value) for value in parse_csv(ckpt_args.get("local_context_layers", "2,6"))
            ],
            adapter_dim=adapter_dim,
            adapter_heads=adapter_heads,
            dropout=float(ckpt_args.get("dropout", 0.0)),
            evidence_tokens=int(role_queries.shape[1]),
            soft_prompt_scale=float(ckpt_args.get("soft_prompt_scale", 0.05)),
            gate_bias_init=float(ckpt_args.get("grounded_gate_bias_init", -2.0)),
        )
        adapter = HybridGlobalLocalAdapter(
            global_adapter=global_adapter,
            local_adapter=local_adapter,
            freeze_global=True,
            global_prompt_dropout=0.0,
            combine_mode="concat",
        )
        adapter.mask_inactive_local_tokens = bool(
            ckpt_args.get("mask_inactive_local_tokens", False)
        )
        adapter.load_state_dict(state_dict, strict=True)
        return adapter

    if architecture in {"residual_question_qformer", "residual_question_adapter"}:
        local_adapter = ResidualQuestionConditionedAdapter(
            aligned_adapter=global_adapter,
            llm_hidden_size=llm_hidden_size,
            context_layers=[int(value) for value in parse_csv(ckpt_args.get("local_context_layers", "2,6"))],
            adapter_heads=adapter_heads,
            dropout=float(ckpt_args.get("dropout", 0.0)),
            text_gate_init=float(ckpt_args.get("local_text_gate_init", 0.05)),
            residual_gate_init=float(ckpt_args.get("local_gate_init", 0.1)),
            # Checkpoints predating these fields retain their original trainable-clone behavior.
            freeze_backbone=bool(ckpt_args.get("freeze_conditioned_backbone", False)),
            text_gate_trainable=bool(ckpt_args.get("local_text_gate_trainable", True)),
            residual_gate_trainable=bool(ckpt_args.get("local_residual_gate_trainable", True)),
            zero_init_text_attention=bool(ckpt_args.get("zero_init_local_text_attention", False)),
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


def save_validate_and_rebuild_adapter_checkpoint(
    path: str | Path,
    *,
    adapter: nn.Module,
    args: argparse.Namespace,
    latent_shape: Sequence[int],
    llm_hidden_size: int,
    latent_contract: Mapping[str, Any],
    metrics: Mapping[str, Any] | None = None,
) -> None:
    """Persist a checkpoint and prove that its standalone strict-load path works."""
    save_adapter_checkpoint(
        path,
        adapter=adapter,
        args=args,
        latent_shape=latent_shape,
        llm_hidden_size=llm_hidden_size,
        latent_contract=latent_contract,
        metrics=metrics,
    )
    checkpoint: Any = None
    rebuilt: nn.Module | None = None
    try:
        checkpoint = torch.load(
            Path(path),
            map_location="cpu",
            weights_only=True,
        )
        validate_adapter_checkpoint_payload(
            checkpoint,
            expected_latent_shape=latent_shape,
            expected_llm_hidden_size=int(llm_hidden_size),
            expected_architecture=str(args.adapter_architecture),
            expected_latent_contract=latent_contract,
            expected_latent_channel_policy=str(args.latent_channel_policy),
        )
        rebuilt = adapter_from_checkpoint(
            checkpoint,
            latent_shape=latent_shape,
            llm_hidden_size=int(llm_hidden_size),
        )
    finally:
        del rebuilt
        del checkpoint
        gc.collect()


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
        "zero_local",
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
    if isinstance(correct, Mapping) and isinstance(correct.get("routing"), Mapping):
        for key, value in correct["routing"].items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                payload[f"{prefix}/routing/{key}"] = float(value)
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
    if isinstance(correct, Mapping) and isinstance(correct.get("by_task"), Mapping):
        correct_tasks = correct["by_task"]
        for baseline in ("zero_local", "global_only", "shuffled"):
            baseline_metrics = metrics.get(baseline)
            baseline_tasks = (
                baseline_metrics.get("by_task")
                if isinstance(baseline_metrics, Mapping)
                else None
            )
            if not isinstance(baseline_tasks, Mapping):
                continue
            for task, correct_task in correct_tasks.items():
                baseline_task = baseline_tasks.get(task)
                if (
                    isinstance(correct_task, Mapping)
                    and isinstance(baseline_task, Mapping)
                    and isinstance(correct_task.get("accuracy"), (int, float))
                    and isinstance(baseline_task.get("accuracy"), (int, float))
                ):
                    payload[
                        f"{prefix}/task/{task}/correct_minus_{baseline}_accuracy"
                    ] = float(correct_task["accuracy"]) - float(
                        baseline_task["accuracy"]
                    )
    add_accuracy_deltas(prefix, metrics, payload)
    return payload


def compact_diagnostic_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    keys = (
        "local_prompt_relative_l2_mean",
        "answer_margin_correct_minus_shuffled",
        "same_task_different_question_local_relative_l2_mean",
        "same_latent_different_question_question_last_relative_l2_mean",
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
    tensor_sensitivity = metrics.get("question_last_relative_l2_by_layer")
    if isinstance(tensor_sensitivity, Mapping):
        for layer, value in tensor_sensitivity.items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                payload[f"diagnostics/tensor_question_last_relative_l2/layer_{layer}"] = float(value)
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


def _mode_task_accuracies(
    metrics: Mapping[str, Any],
    mode: str,
    tasks: Sequence[str] = STAGE2B_TASK_TYPES,
) -> dict[str, float]:
    mode_metrics = metrics.get(mode)
    by_task = mode_metrics.get("by_task") if isinstance(mode_metrics, Mapping) else None
    if not isinstance(by_task, Mapping):
        raise ValueError(f"Validation metrics are missing {mode}.by_task.")
    result: dict[str, float] = {}
    for task in tasks:
        task_metrics = by_task.get(task)
        accuracy = task_metrics.get("accuracy") if isinstance(task_metrics, Mapping) else None
        if not isinstance(accuracy, (int, float)) or not math.isfinite(float(accuracy)):
            raise ValueError(f"Validation metrics are missing a finite {mode}/{task} accuracy.")
        result[str(task)] = float(accuracy)
    return result


def joint_ab_checkpoint_metrics(
    metrics: Mapping[str, Any],
    parent_metrics: Mapping[str, Any],
    *,
    min_causal_gain: float = 0.015,
    max_parent_regression: float = 0.005,
    min_no_harm_delta: float = 0.0,
) -> dict[str, Any]:
    """Compute Pareto-first selection and fixed small-run acceptance diagnostics."""

    current_hybrid = _mode_task_accuracies(metrics, "correct")
    current_global = _mode_task_accuracies(metrics, "global_only")
    current_zero = _mode_task_accuracies(metrics, "zero_local")
    current_shuffled = _mode_task_accuracies(metrics, "shuffled")
    parent_hybrid = _mode_task_accuracies(parent_metrics, "correct")
    parent_global = _mode_task_accuracies(parent_metrics, "global_only")
    hybrid_delta = {
        task: current_hybrid[task] - parent_hybrid[task]
        for task in STAGE2B_TASK_TYPES
    }
    global_delta = {
        task: current_global[task] - parent_global[task]
        for task in STAGE2B_TASK_TYPES
    }
    no_harm_delta = {
        task: current_hybrid[task] - current_global[task]
        for task in STAGE2B_TASK_TYPES
    }
    causal_gain = {
        task: current_hybrid[task]
        - max(current_zero[task], current_global[task], current_shuffled[task])
        for task in ("normalized_point_value", "raw_point_value_with_stats")
    }
    worst_hybrid_delta = min(hybrid_delta.values())
    worst_global_delta = min(global_delta.values())
    worst_protected_delta = min(worst_hybrid_delta, worst_global_delta)
    point_value_min_causal_gain = min(causal_gain.values())
    protected_compare_delta = min(
        no_harm_delta["point_compare"],
        no_harm_delta["region_mean_compare"],
    )
    current_correct_metrics = metrics.get("correct")
    parent_correct_metrics = parent_metrics.get("correct")
    current_overall = (
        current_correct_metrics.get("accuracy")
        if isinstance(current_correct_metrics, Mapping)
        else None
    )
    parent_overall = (
        parent_correct_metrics.get("accuracy")
        if isinstance(parent_correct_metrics, Mapping)
        else None
    )
    if not isinstance(current_overall, (int, float)) or not isinstance(
        parent_overall, (int, float)
    ):
        raise ValueError("Joint A/B selection requires current and parent overall correct accuracy.")
    overall_delta = float(current_overall) - float(parent_overall)
    acceptance = {
        "point_value_causal_gain": point_value_min_causal_gain >= float(min_causal_gain),
        "compare_no_harm": protected_compare_delta >= float(min_no_harm_delta),
        "global_parent_preserved": worst_global_delta >= -float(max_parent_regression),
        "hybrid_parent_preserved": worst_hybrid_delta >= -float(max_parent_regression),
        "overall_improved": overall_delta > 0.0,
    }
    return {
        "hybrid_delta_by_task": hybrid_delta,
        "global_only_delta_by_task": global_delta,
        "hybrid_minus_global_by_task": no_harm_delta,
        "point_value_causal_gain_by_task": causal_gain,
        "worst_hybrid_delta": worst_hybrid_delta,
        "worst_global_only_delta": worst_global_delta,
        "worst_protected_task_delta": worst_protected_delta,
        "point_value_min_causal_gain": point_value_min_causal_gain,
        "protected_compare_no_harm_delta": protected_compare_delta,
        "overall_delta": overall_delta,
        # Candidates are compared lexicographically. Preservation is deliberately
        # first; aggregate accuracy cannot hide a regressed task.
        "selection_key": [
            worst_protected_delta,
            point_value_min_causal_gain,
            overall_delta,
        ],
        "acceptance": acceptance,
        "accepted": all(acceptance.values()),
    }


def point_reader_checkpoint_metrics(
    metrics: Mapping[str, Any],
    parent_metrics: Mapping[str, Any],
    *,
    min_parent_delta: float = 0.0,
    min_causal_gain: float = 0.0,
    max_nonpoint_regression: float = 0.03,
) -> dict[str, Any]:
    """Rank evidence-only checkpoints by point gains with auxiliary guardrails."""

    current = _mode_task_accuracies(metrics, "correct")
    parent = _mode_task_accuracies(parent_metrics, "correct")
    current_zero = _mode_task_accuracies(metrics, "zero_local")
    current_global = _mode_task_accuracies(metrics, "global_only")
    current_shuffled = _mode_task_accuracies(metrics, "shuffled")
    delta = {task: current[task] - parent[task] for task in STAGE2B_TASK_TYPES}
    point_delta = {task: delta[task] for task in POINT_VALUE_TASK_TYPES}
    nonpoint_delta = {
        task: delta[task]
        for task in STAGE2B_TASK_TYPES
        if task not in POINT_VALUE_TASK_TYPES
    }
    causal_gain = {
        task: current[task]
        - max(current_zero[task], current_global[task], current_shuffled[task])
        for task in POINT_VALUE_TASK_TYPES
    }
    min_point_delta = min(point_delta.values())
    mean_point_delta = sum(point_delta.values()) / len(point_delta)
    min_point_causal_gain = min(causal_gain.values())
    worst_nonpoint_delta = min(nonpoint_delta.values())
    acceptance = {
        "point_parent_delta": min_point_delta >= float(min_parent_delta),
        "point_causal_gain": min_point_causal_gain >= float(min_causal_gain),
        "nonpoint_parent_guardrail": worst_nonpoint_delta
        >= -float(max_nonpoint_regression),
    }
    return {
        "hybrid_delta_by_task": delta,
        "point_parent_delta_by_task": point_delta,
        "nonpoint_parent_delta_by_task": nonpoint_delta,
        "point_value_causal_gain_by_task": causal_gain,
        "point_value_min_parent_delta": min_point_delta,
        "point_value_mean_parent_delta": mean_point_delta,
        "point_value_min_causal_gain": min_point_causal_gain,
        "worst_nonpoint_parent_delta": worst_nonpoint_delta,
        # Point-value improvement owns selection. Auxiliary tasks are guardrails,
        # not co-equal objectives that can hide a failed Stage-2B reader.
        "selection_key": [
            min_point_delta,
            min_point_causal_gain,
            mean_point_delta,
            worst_nonpoint_delta,
        ],
        "acceptance": acceptance,
        "accepted": all(acceptance.values()),
    }


def screened_stage2b_checkpoint_metrics(
    metrics: Mapping[str, Any],
    parent_metrics: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if bool(getattr(args, "point_reader_training", False)) or bool(
        getattr(args, "full_local_reader_training", False)
    ):
        return point_reader_checkpoint_metrics(
            metrics,
            parent_metrics,
            min_parent_delta=float(args.point_reader_min_parent_delta),
            min_causal_gain=float(args.point_reader_min_causal_gain),
            max_nonpoint_regression=float(
                args.point_reader_max_nonpoint_regression
            ),
        )
    return joint_ab_checkpoint_metrics(
        metrics,
        parent_metrics,
        min_causal_gain=float(args.joint_min_causal_gain),
        max_parent_regression=float(args.joint_max_parent_regression),
        min_no_harm_delta=float(args.joint_min_no_harm_delta),
    )


def stage2b_full_validation_candidates(
    candidates: Sequence[dict[str, Any]],
    *,
    full_local_reader_training: bool,
    top_k: int,
) -> list[dict[str, Any]]:
    """Keep the parent and choose the trained checkpoints that receive full validation."""

    ranked = sorted(
        candidates,
        key=lambda item: tuple(
            float(value) for value in item["screening_selection"]["selection_key"]
        ),
        reverse=True,
    )
    parents = [item for item in candidates if bool(item.get("is_parent", False))]
    trained = [item for item in ranked if not bool(item.get("is_parent", False))]
    if full_local_reader_training:
        return parents + trained
    return parents + trained[: int(top_k)]


def select_admitted_stage2b_candidate(
    candidates: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], bool]:
    """Promote the best admitted child, or retain the unique step-zero parent."""

    parents = [item for item in candidates if bool(item.get("is_parent", False))]
    if len(parents) != 1:
        raise RuntimeError(
            "Screened Stage-2B selection requires exactly one step-zero parent."
        )
    accepted_children = [
        item
        for item in candidates
        if not bool(item.get("is_parent", False))
        and bool(item["full_selection"].get("accepted", False))
        and bool(item["full_selection"].get("eligible_for_promotion", True))
    ]
    selected = max(
        accepted_children or parents,
        key=lambda item: tuple(
            float(value) for value in item["full_selection"]["selection_key"]
        ),
    )
    promoted = any(selected is item for item in accepted_children)
    return selected, accepted_children, promoted


def resolve_test_evaluation_policy(
    *,
    requested: bool,
    joint_ab_training: bool,
    joint_selected_accepted: bool | None,
    point_reader_training: bool = False,
    full_local_reader_training: bool = False,
) -> tuple[bool, str | None]:
    """Gate formal test access on validation-only screened admission."""

    if not bool(requested):
        return False, "evaluate_test_disabled"
    if (
        bool(joint_ab_training)
        or bool(point_reader_training)
        or bool(full_local_reader_training)
    ) and joint_selected_accepted is not True:
        return False, "full_validation_admission_rejected"
    return True, None


def checkpoint_score(
    metrics: Mapping[str, Any],
    metric_name: str,
    reference_metrics: Mapping[str, Any] | None = None,
) -> float:
    if metric_name == "joint_ab_worst_task_delta":
        if reference_metrics is None:
            return -math.inf
        return float(
            joint_ab_checkpoint_metrics(metrics, reference_metrics)[
                "worst_protected_task_delta"
            ]
        )
    if metric_name == "macro_latent_gain":
        return macro_latent_gain(metrics)
    if metric_name == "correct_accuracy":
        correct = metrics.get("correct")
        return float(correct.get("accuracy", -math.inf)) if isinstance(correct, Mapping) else -math.inf
    if metric_name in {
        "normalized_point_latent_gain",
        "point_value_min_latent_gain",
        "point_value_min_grounded_gain",
        "point_value_min_causal_gain",
    }:
        correct = metrics.get("correct")
        shuffled = metrics.get("shuffled")
        if not isinstance(correct, Mapping) or not isinstance(shuffled, Mapping):
            return -math.inf
        correct_tasks = correct.get("by_task")
        shuffled_tasks = shuffled.get("by_task")
        if not isinstance(correct_tasks, Mapping) or not isinstance(shuffled_tasks, Mapping):
            return -math.inf
        global_tasks: Mapping[str, Any] | None = None
        if metric_name in {"point_value_min_grounded_gain", "point_value_min_causal_gain"}:
            global_only = metrics.get("global_only")
            if not isinstance(global_only, Mapping) or not isinstance(
                global_only.get("by_task"), Mapping
            ):
                return -math.inf
            global_tasks = global_only["by_task"]
        zero_local_tasks: Mapping[str, Any] | None = None
        if metric_name == "point_value_min_causal_gain":
            zero_local = metrics.get("zero_local")
            if not isinstance(zero_local, Mapping) or not isinstance(
                zero_local.get("by_task"), Mapping
            ):
                return -math.inf
            zero_local_tasks = zero_local["by_task"]

        def gain(task: str) -> float:
            correct_task = correct_tasks.get(task)
            shuffled_task = shuffled_tasks.get(task)
            if not isinstance(correct_task, Mapping) or not isinstance(shuffled_task, Mapping):
                return -math.inf
            baselines = [float(shuffled_task.get("accuracy", math.inf))]
            if global_tasks is not None:
                global_task = global_tasks.get(task)
                if not isinstance(global_task, Mapping):
                    return -math.inf
                baselines.append(float(global_task.get("accuracy", math.inf)))
            if zero_local_tasks is not None:
                zero_local_task = zero_local_tasks.get(task)
                if not isinstance(zero_local_task, Mapping):
                    return -math.inf
                baselines.append(float(zero_local_task.get("accuracy", math.inf)))
            return float(correct_task.get("accuracy", -math.inf)) - max(baselines)

        normalized_gain = gain("normalized_point_value")
        if metric_name == "normalized_point_latent_gain":
            return normalized_gain
        return min(normalized_gain, gain("raw_point_value_with_stats"))
    raise ValueError(f"Unsupported checkpoint metric: {metric_name}")


def grounded_routing_warmup_audit(
    metrics: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    correct = metrics.get("correct")
    routing = correct.get("routing") if isinstance(correct, Mapping) else None
    if not isinstance(routing, Mapping):
        raise ValueError("Routing warmup validation did not produce correct.routing metrics.")
    observed = {
        "cell_top1": float(routing.get("top1_accuracy", math.nan)),
        "cell_top5": float(routing.get("top5_accuracy", math.nan)),
        "row_top1": float(routing.get("row_top1_accuracy", math.nan)),
        "col_top1": float(routing.get("col_top1_accuracy", math.nan)),
        "target_mass": float(routing.get("target_mass", math.nan)),
        "normalized_entropy": float(routing.get("normalized_entropy", math.nan)),
        "gate_accuracy": float(routing.get("gate_accuracy", math.nan)),
    }
    thresholds = {
        "cell_top1": float(args.grounding_warmup_min_cell_top1),
        "cell_top5": float(args.grounding_warmup_min_cell_top5),
        "row_top1": float(args.grounding_warmup_min_axis_top1),
        "col_top1": float(args.grounding_warmup_min_axis_top1),
        "target_mass": float(args.grounding_warmup_min_target_mass),
        "gate_accuracy": float(args.grounding_warmup_min_gate_accuracy),
    }
    checks = {
        name: math.isfinite(observed[name]) and observed[name] >= threshold
        for name, threshold in thresholds.items()
    }
    active_roles = int(routing.get("active_roles", 0))
    if active_roles <= 0:
        checks["active_roles"] = False
    by_task = routing.get("by_task")
    point_task_observed: dict[str, dict[str, float]] = {}
    for task in ("normalized_point_value", "raw_point_value_with_stats"):
        task_metrics = by_task.get(task) if isinstance(by_task, Mapping) else None
        if not isinstance(task_metrics, Mapping):
            checks[f"{task}.present"] = False
            continue
        task_values = {
            "cell_top1": float(task_metrics.get("top1_accuracy", math.nan)),
            "target_mass": float(task_metrics.get("target_mass", math.nan)),
            "normalized_entropy": float(
                task_metrics.get("normalized_entropy", math.nan)
            ),
        }
        point_task_observed[task] = task_values
        checks[f"{task}.cell_top1"] = (
            int(task_metrics.get("active_roles", 0)) > 0
            and math.isfinite(task_values["cell_top1"])
            and task_values["cell_top1"] >= thresholds["cell_top1"]
        )
        checks[f"{task}.target_mass"] = (
            math.isfinite(task_values["target_mass"])
            and task_values["target_mass"] >= thresholds["target_mass"]
        )
    return {
        "passed": all(checks.values()),
        "active_roles": active_roles,
        "observed": observed,
        "point_tasks": point_task_observed,
        "thresholds": thresholds,
        "checks": checks,
        "failed": [name for name, passed in checks.items() if not passed],
    }


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
            "total_optimizer_updates",
            "checkpoint_updates",
            "checkpoint_fractions",
            "task_loss_weights",
            "joint_run_scope",
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
            "record_subset_mode": str(args.record_subset_mode),
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
            "stage2_warm_start_checkpoint": args.stage2_warm_start_checkpoint,
            "stage2b_resume_checkpoint": args.stage2b_resume_checkpoint,
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
            "grounded_gate_bias_init": float(args.grounded_gate_bias_init),
            "local_text_gate_init": float(args.local_text_gate_init),
            "freeze_conditioned_backbone": bool(args.freeze_conditioned_backbone),
            "local_text_gate_trainable": bool(args.local_text_gate_trainable),
            "local_residual_gate_trainable": bool(args.local_residual_gate_trainable),
            "zero_init_local_text_attention": bool(args.zero_init_local_text_attention),
            "freeze_global_adapter": bool(args.freeze_global_adapter),
            "global_unfreeze_epoch": int(args.global_unfreeze_epoch),
            "global_lr": float(args.global_lr),
            "global_prompt_dropout": float(args.global_prompt_dropout),
            "mask_inactive_local_tokens": bool(args.mask_inactive_local_tokens),
            "global_dropout": float(getattr(args, "global_dropout", args.dropout)),
            "global_soft_prompt_scale": float(
                getattr(args, "global_soft_prompt_scale", args.soft_prompt_scale)
            ),
            "soft_prompt_scale": float(args.soft_prompt_scale),
        },
        "llm_training": {
            "joint_run_scope": str(args.joint_run_scope),
            "prompt_template": str(args.prompt_template),
            "epochs": int(args.epochs),
            "grounding_routing_warmup_epochs": int(args.grounding_routing_warmup_epochs),
            "grounding_warmup_thresholds": {
                "cell_top1": float(args.grounding_warmup_min_cell_top1),
                "cell_top5": float(args.grounding_warmup_min_cell_top5),
                "axis_top1": float(args.grounding_warmup_min_axis_top1),
                "target_mass": float(args.grounding_warmup_min_target_mass),
                "gate_accuracy": float(args.grounding_warmup_min_gate_accuracy),
            },
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
            "grounding_routing_loss_weight": float(args.grounding_routing_loss_weight),
            "grounding_joint_routing_loss_weight": float(
                args.grounding_joint_routing_loss_weight
            ),
            "grounding_gate_loss_weight": float(args.grounding_gate_loss_weight),
            "matched_group_loss_weight": float(args.matched_group_loss_weight),
            "matched_group_loss_margin": float(args.matched_group_loss_margin),
            "joint_ab_training": bool(args.joint_ab_training),
            "point_reader_training": bool(args.point_reader_training),
            "full_local_reader_training": bool(args.full_local_reader_training),
            "task_balanced_answer_loss": bool(args.task_balanced_answer_loss),
            "global_view_loss_weight": float(args.global_view_loss_weight),
            "joint_no_harm_loss_weight": float(args.joint_no_harm_loss_weight),
            "joint_no_harm_margin": float(args.joint_no_harm_margin),
            "joint_causal_loss_weight": float(args.joint_causal_loss_weight),
            "joint_causal_margin": float(args.joint_causal_margin),
            "point_causal_loss_weight": float(args.point_causal_loss_weight),
            "point_causal_margin": float(args.point_causal_margin),
            "point_causal_tasks": parse_csv(args.point_causal_tasks),
            "nonpoint_no_harm_loss_weight": float(
                args.nonpoint_no_harm_loss_weight
            ),
            "nonpoint_no_harm_margin": float(args.nonpoint_no_harm_margin),
            "global_anchor_loss_weight": float(args.global_anchor_loss_weight),
            "local_anchor_loss_weight": float(args.local_anchor_loss_weight),
            "swapped_question_max_records": int(args.swapped_question_max_records),
            "swapped_question_require_different_answer": bool(
                args.swapped_question_require_different_answer
            ),
            "grounding_score_space": "restricted_choice",
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
            "checkpoint_updates": parse_csv(args.checkpoint_updates),
            "checkpoint_fractions": parse_csv(args.checkpoint_fractions),
            "checkpoint_screening_records": int(args.checkpoint_screening_records),
            "checkpoint_full_eval_top_k": int(args.checkpoint_full_eval_top_k),
            "joint_min_causal_gain": float(args.joint_min_causal_gain),
            "joint_max_parent_regression": float(args.joint_max_parent_regression),
            "joint_min_no_harm_delta": float(args.joint_min_no_harm_delta),
            "point_reader_min_parent_delta": float(
                args.point_reader_min_parent_delta
            ),
            "point_reader_min_causal_gain": float(
                args.point_reader_min_causal_gain
            ),
            "point_reader_max_nonpoint_regression": float(
                args.point_reader_max_nonpoint_regression
            ),
            "task_loss_weights": dict(getattr(args, "task_loss_weights", {})),
            "evaluate_test": bool(args.evaluate_test),
            "group_questions_by_state": bool(args.group_questions_by_state),
            "questions_per_state_group": int(args.questions_per_state_group),
            "diagnostics": {
                "enabled": bool(args.diagnostics_enabled),
                "every_epochs": int(args.diagnostics_every_epochs),
                "records_per_task": int(args.diagnostics_records_per_task),
                "save_states": bool(args.diagnostics_save_states),
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


def log_wandb_on_rank_zero(
    wandb_logger: WandbLogger,
    payload: Mapping[str, Any],
    step: int,
    stage: str,
) -> None:
    """Log once while propagating a rank-0 W&B failure to every worker."""
    run_on_rank_zero_and_broadcast(
        lambda: wandb_logger.log(dict(payload), step=int(step)),
        stage,
    )


def main() -> None:
    global _ACTIVE_RUN_LIFECYCLE
    args = parse_args()
    validate_stage2_warm_start_file(args)
    if bool(args.require_disjoint_splits) and bool(args.structured_query_conditioning):
        raise ValueError(
            "Formal runs cannot enable adapter.structured_query_conditioning because it uses "
            "regex-parsed task/coordinate features. Disable it so the adapter reads the natural-language question."
        )
    apply_runtime_environment(args)
    device = initialize_distributed_device(
        str(args.device),
        float(args.distributed_timeout_seconds),
    )
    seed_everything(int(args.seed))
    run_dir = build_distributed_run_dir(args.output_root, args.run_name)
    lifecycle: RunLifecycle | None = None

    def initialize_lifecycle() -> dict[str, str]:
        nonlocal lifecycle
        lifecycle = RunLifecycle(run_dir)
        return {"started_at": lifecycle.started_at}

    lifecycle_metadata = run_on_rank_zero_and_broadcast(
        initialize_lifecycle,
        "run lifecycle initialization",
    )
    _ACTIVE_RUN_LIFECYCLE = lifecycle

    def write_startup_snapshots() -> None:
        atomic_dump_json(run_dir / "args_requested.json", redacted_args(args))
        if args.config:
            atomic_dump_json(
                run_dir / "config_snapshot.json",
                redacted_config_snapshot(load_yaml_mapping(args.config)),
            )

    run_on_rank_zero_and_broadcast(write_startup_snapshots, "startup snapshot write")
    if is_main_process():
        print(
            f"run={run_dir.name} started_at={lifecycle_metadata['started_at']} "
            "startup=metadata_audit"
        )
    qa_metadata_audit = run_on_rank_zero_and_broadcast(
        lambda: audit_qa_metadata(args),
        "QA metadata audit",
    )
    raw_latent_contract = qa_metadata_audit.get("latent_contract")
    latent_contract = dict(raw_latent_contract) if isinstance(raw_latent_contract, Mapping) else None
    run_on_rank_zero_and_broadcast(
        lambda: atomic_dump_json(run_dir / "qa_metadata_audit.json", qa_metadata_audit),
        "QA metadata audit write",
    )
    if str(args.adapter_architecture) == "grounded_evidence_adapter":
        if latent_contract is None:
            raise ValueError("Grounded Stage-2B requires a formal latent contract.")
        warm_start_audit = run_on_rank_zero_and_broadcast(
            lambda: audit_stage2_warm_start_checkpoint(
                args.stage2_warm_start_checkpoint,
                latent_contract,
                str(args.latent_channel_policy),
            ),
            "Stage-2 warm-start checkpoint audit",
        )
        run_on_rank_zero_and_broadcast(
            lambda: atomic_dump_json(
                run_dir / "stage2_warm_start_audit.json",
                warm_start_audit,
            ),
            "Stage-2 warm-start checkpoint audit write",
        )
        if is_main_process():
            print(
                "startup=stage2_warm_start_audit "
                f"sha256={warm_start_audit['sha256']} "
                f"parameters={warm_start_audit['parameters']}"
            )
    if is_main_process():
        print("startup=dataset_index")

    train_dataset = TensorReadoutQADataset(
        qa_path(args.qa_dir, args.train_split),
        latent_dir=args.latent_dir,
        max_records=args.max_train_records,
        subset_mode=str(args.record_subset_mode),
        subset_seed=int(args.shuffle_seed),
        prefer_record_latent_ref=bool(args.prefer_record_latent_ref),
        shuffle_seed=int(args.shuffle_seed),
        latent_cache_size=int(args.latent_cache_size),
        latent_contract=latent_contract,
        latent_channel_policy=str(args.latent_channel_policy),
    )
    if bool(args.task_balanced_answer_loss):
        args.task_loss_weights = inverse_frequency_task_weights(
            train_dataset.records,
            expected_tasks=STAGE2B_TASK_TYPES,
        )
    else:
        args.task_loss_weights = {}
    val_dataset = TensorReadoutQADataset(
        qa_path(args.qa_dir, args.val_split),
        latent_dir=args.latent_dir,
        max_records=args.max_val_records,
        subset_mode=str(args.record_subset_mode),
        subset_seed=int(args.shuffle_seed) + 1,
        prefer_record_latent_ref=bool(args.prefer_record_latent_ref),
        shuffle_seed=int(args.shuffle_seed),
        latent_cache_size=int(args.latent_cache_size),
        latent_contract=latent_contract,
        latent_channel_policy=str(args.latent_channel_policy),
    )
    test_dataset = TensorReadoutQADataset(
        qa_path(args.qa_dir, args.test_split),
        latent_dir=args.latent_dir,
        max_records=args.max_test_records,
        subset_mode=str(args.record_subset_mode),
        subset_seed=int(args.shuffle_seed) + 2,
        prefer_record_latent_ref=bool(args.prefer_record_latent_ref),
        shuffle_seed=int(args.shuffle_seed),
        latent_cache_size=int(args.latent_cache_size),
        latent_contract=latent_contract,
        latent_channel_policy=str(args.latent_channel_policy),
    )
    first_latent = train_dataset[0]["latent_map"]
    latent_shape = tuple(int(dim) for dim in first_latent.shape)
    latent_channels = int(latent_shape[0])
    datasets = {"train": train_dataset, "val": val_dataset, "test": test_dataset}
    if is_main_process():
        print(
            f"startup=data_audit train/val/test={len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}"
        )
    data_audit = run_on_rank_zero_and_broadcast(
        lambda: audit_qa_datasets(
            datasets,
            require_disjoint_splits=bool(args.require_disjoint_splits),
            require_complete_split_coverage=not any(
                value is not None
                for value in (
                    args.max_train_records,
                    args.max_val_records,
                    args.max_test_records,
                )
            ),
        ),
        "QA dataset audit",
    )
    run_on_rank_zero_and_broadcast(
        lambda: atomic_dump_json(run_dir / "data_audit.json", data_audit),
        "QA dataset audit write",
    )

    if is_main_process():
        print("startup=tokenizer_and_prompt_audit")
    tokenizer = load_tokenizer(args)
    prompt_audit = run_on_rank_zero_and_broadcast(
        lambda: audit_prompt_tokenization(
            datasets=datasets,
            tokenizer=tokenizer,
            max_prompt_tokens=int(args.max_prompt_tokens),
            prompt_template=str(args.prompt_template),
            audit_local_conditioning_prompt=uses_contextual_local_prompt(args),
        ),
        "prompt tokenization audit",
    )
    choice_tokenization_audit = run_on_rank_zero_and_broadcast(
        lambda: audit_choice_tokenization(datasets, tokenizer),
        "choice tokenization audit",
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
    def write_prompt_audits() -> None:
        atomic_dump_json(run_dir / "prompt_audit.json", prompt_audit)
        atomic_dump_json(run_dir / "choice_tokenization_audit.json", choice_tokenization_audit)

    run_on_rank_zero_and_broadcast(write_prompt_audits, "prompt audit write")
    if configured_choice_mode == "label" and not bool(
        choice_tokenization_audit["all_labels_single_token"]
    ):
        raise ValueError(
            "choice_scoring_mode=label requires all labels to tokenize as one unique token; "
            "see choice_tokenization_audit.json."
        )
    if bool(args.require_untruncated_prompts) and not bool(prompt_audit["all_prompts_fit"]):
        raise ValueError(
            "Prompt audit found an active prompt path longer than "
            f"max_prompt_tokens={int(args.max_prompt_tokens)}. Increase the limit so formal runs do not "
            "silently remove natural-language instructions. See prompt_audit.json."
        )
    pre_load_memory = gather_cuda_memory(device)
    pre_load_host_memory = gather_host_memory(device)
    if is_main_process():
        memory_suffix = ""
        if pre_load_memory:
            memory_suffix = " gpu_memory=" + ",".join(
                f"r{int(item['rank'])}:{item['free_gib']:.2f}/{item['total_gib']:.2f}GiB"
                for item in pre_load_memory
            )
            if any(item["free_gib"] < 0.95 * item["total_gib"] for item in pre_load_memory):
                memory_suffix += " warning=visible_gpu_not_empty"
        if pre_load_host_memory:
            memory_suffix += (
                f" host_available={min(item['available_gib'] for item in pre_load_host_memory):.2f}GiB"
                f" host_total={min(item['total_gib'] for item in pre_load_host_memory):.2f}GiB"
            )
        print(
            f"startup=llm_load visible_cuda={os.environ.get('CUDA_VISIBLE_DEVICES', '<all>')}"
            f"{memory_suffix}"
        )
    llm, model_dtype = load_llm_with_bounded_host_memory(args, device)
    post_load_memory = gather_cuda_memory(device)
    post_load_host_memory = enforce_host_memory_floor(
        device,
        float(args.min_host_memory_available_gib),
        "post-LLM startup",
    )
    if is_main_process() and post_load_memory:
        host_suffix = ""
        if post_load_host_memory:
            host_suffix = (
                f" host_available={min(item['available_gib'] for item in post_load_host_memory):.2f}GiB"
                " rank_rss="
                + ",".join(
                    f"r{int(item['rank'])}:{item['process_rss_gib']:.2f}GiB"
                    for item in post_load_host_memory
                )
            )
        print(
            "startup=llm_loaded gpu_memory="
            + ",".join(
                f"r{int(item['rank'])}:allocated={item['allocated_gib']:.2f}GiB,"
                f"free={item['free_gib']:.2f}/{item['total_gib']:.2f}GiB"
                for item in post_load_memory
            )
            + f" gradient_checkpointing={int(bool(args.llm_gradient_checkpointing))}"
            + host_suffix
        )
    llm_hidden_size = int(llm.get_input_embeddings().embedding_dim)
    diagnostic_layer_audit = (
        validate_diagnostic_layers(llm, args.diagnostics_layers)
        if bool(args.diagnostics_enabled)
        else {"validated_before_training": False, "reason": "diagnostics disabled"}
    )
    context_layer_values = [int(value) for value in parse_csv(args.local_context_layers)]
    uses_contextual_local_adapter = uses_contextual_local_prompt(args)
    local_context_audit = (
        {
            "validated_before_training": True,
            "layers": [
                validate_local_context_layer(llm, layer_index)
                for layer_index in context_layer_values
            ],
        }
        if uses_contextual_local_adapter
        else {"validated_before_training": False, "reason": "architecture has no contextual local adapter"}
    )
    local_context_preflight: tuple[torch.Tensor, torch.Tensor] | None = None
    if uses_contextual_local_adapter:
        llm_checkpoint_training = frozen_llm_checkpoint_execution_active(llm)
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
        local_context_preflight = (preflight_context, preflight_mask.bool())
        set_frozen_llm_execution_mode(llm, checkpoint_training=llm_checkpoint_training)

    initialization = "random"
    checkpoint_load_report: dict[str, Any] = {"mode": "random"}
    global_checkpoint_load_report: dict[str, Any] | None = None
    stage1_teacher_supervision: dict[str, Any] | None = None
    stage1_checkpoint_version = 0
    stage1_checkpoint_phase: str | None = None
    stage1_checkpoint_validation_mode: str | None = None
    parent_validation_metrics: dict[str, Any] | None = None
    if str(args.adapter_architecture) in {
        "alignment_qformer",
        "alignment_adapter",
        "hybrid_local_qformer",
        "residual_question_qformer",
        "residual_question_adapter",
        "grounded_evidence_adapter",
    }:
        checkpoint: Mapping[str, Any] | None = None
        checkpoint_args: Mapping[str, Any] = {}
        hybrid_state_dict: Mapping[str, Any] | None = None
        init_checkpoint = str(args.adapter_init_checkpoint or "").strip()
        if init_checkpoint.lower() in {"", "none", "null", "random"}:
            init_checkpoint = ""
        if init_checkpoint:
            loaded = torch.load(
                Path(init_checkpoint).expanduser(),
                map_location="cpu",
                weights_only=True,
            )
            if not isinstance(loaded, Mapping):
                raise ValueError(f"Unsupported alignment checkpoint: {args.adapter_init_checkpoint}")
            checkpoint = loaded
            checkpoint_args = loaded.get("args", {}) if isinstance(loaded.get("args"), Mapping) else {}
            if is_direct_alignment_architecture(args.adapter_architecture) or str(
                args.adapter_architecture
            ) == "grounded_evidence_adapter":
                checkpoint_validation = validate_stage1_alignment_checkpoint_payload(
                    loaded,
                    path=init_checkpoint,
                )
                if "adapter_architecture" in checkpoint_args:
                    raise ValueError(
                        "Direct Stage 2 must start from a Stage-1 alignment checkpoint, not a previous "
                        "downstream adapter checkpoint. Use alignment_best.pt from Stage 1."
                    )
                validate_stage1_model_identity(checkpoint_args, args.model_name_or_path)
                stage1_checkpoint_phase = str(checkpoint_validation["checkpoint_phase"])
                stage1_checkpoint_validation_mode = str(checkpoint_validation["validation_mode"])
                stage1_teacher_supervision = validate_stage1_teacher_supervision(loaded)
                stage1_checkpoint_version = int(checkpoint_validation["checkpoint_version"])
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
        if (
            str(args.adapter_architecture) == "grounded_evidence_adapter"
            and global_adapter_type != "spatial_transformer"
        ):
            raise ValueError(
                "grounded_evidence_adapter requires a Stage-1 spatial_transformer adapter with "
                "one row-major token per latent cell."
            )
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
                "stage1_checkpoint_version": int(stage1_checkpoint_version),
                "stage1_checkpoint_phase": stage1_checkpoint_phase,
                "stage1_checkpoint_validation_mode": stage1_checkpoint_validation_mode,
                "stage1_teacher_supervision": stage1_teacher_supervision,
                "stage1_loss_feature_transform_loaded": False,
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
        if str(args.adapter_architecture) == "grounded_evidence_adapter":
            warm_path = str(args.stage2_warm_start_checkpoint or "").strip()
            if warm_path.lower() in {"", "none", "null", "random"}:
                raise ValueError(
                    "grounded_evidence_adapter requires adapter.stage2_warm_start_checkpoint "
                    "from the completed direct Stage 2 run."
                )
            warm_checkpoint = torch.load(
                Path(warm_path).expanduser(),
                map_location="cpu",
                weights_only=True,
            )
            warm_state = validate_adapter_checkpoint_payload(
                warm_checkpoint,
                expected_latent_shape=latent_shape,
                expected_llm_hidden_size=llm_hidden_size,
                expected_architecture="alignment_adapter",
                expected_latent_contract=latent_contract or {},
                expected_latent_channel_policy=str(args.latent_channel_policy),
            )
            warm_args = warm_checkpoint.get("args")
            if not isinstance(warm_args, Mapping):
                raise ValueError("Stage-2 warm-start checkpoint is missing its argument contract.")
            validate_stage1_model_identity(warm_args, args.model_name_or_path)
            warm_stage1 = _configured_checkpoint_path(warm_args.get("adapter_init_checkpoint"))
            configured_stage1 = _configured_checkpoint_path(init_checkpoint)
            if warm_stage1 is None or configured_stage1 is None or warm_stage1 != configured_stage1:
                raise ValueError(
                    "Stage-2 warm start and the configured Stage-1 initialization do not share provenance: "
                    f"warm_parent={warm_stage1}, configured_stage1={configured_stage1}."
                )
            adapter.load_state_dict(warm_state, strict=True)
            args.stage2_warm_start_checkpoint = str(Path(warm_path).expanduser().resolve())
            args.stage2_warm_start_sha256 = sha256_file(Path(warm_path).expanduser())
            checkpoint_load_report = {
                **checkpoint_load_report,
                "mode": "strict_stage2_direct_global_warm_start",
                "stage2_warm_start_checkpoint": args.stage2_warm_start_checkpoint,
                "stage2_warm_start_sha256": args.stage2_warm_start_sha256,
                "stage2_warm_start_architecture": "alignment_adapter",
                "stage2_warm_start_strict_load": True,
            }
            initialization = "stage2_direct_checkpoint_plus_grounded_reader"
            del warm_args, warm_state, warm_checkpoint
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
        if str(args.adapter_architecture) == "grounded_evidence_adapter":
            context_layers = [int(value) for value in parse_csv(args.local_context_layers)]
            global_adapter = adapter
            local_adapter = GroundedEvidenceAdapter(
                latent_grid=latent_grid,
                llm_hidden_size=llm_hidden_size,
                context_layers=context_layers,
                adapter_dim=int(args.adapter_dim),
                adapter_heads=int(args.adapter_heads),
                dropout=float(args.dropout),
                evidence_tokens=int(args.local_soft_prompt_tokens),
                soft_prompt_scale=float(args.soft_prompt_scale),
                gate_bias_init=float(args.grounded_gate_bias_init),
            )
            adapter = HybridGlobalLocalAdapter(
                global_adapter=global_adapter,
                local_adapter=local_adapter,
                freeze_global=bool(args.freeze_global_adapter),
                global_prompt_dropout=0.0,
                combine_mode="concat",
            ).to(device)
            resume_path = str(args.stage2b_resume_checkpoint or "").strip()
            resume_report: dict[str, Any] = {}
            grounded_reader_initialized = True
            if resume_path:
                resume_checkpoint = torch.load(
                    Path(resume_path).expanduser(),
                    map_location="cpu",
                    weights_only=True,
                )
                resume_state = validate_adapter_checkpoint_payload(
                    resume_checkpoint,
                    expected_latent_shape=latent_shape,
                    expected_llm_hidden_size=llm_hidden_size,
                    expected_architecture="grounded_evidence_adapter",
                    expected_latent_contract=latent_contract or {},
                    expected_latent_channel_policy=str(args.latent_channel_policy),
                )
                resume_args = resume_checkpoint.get("args")
                if not isinstance(resume_args, Mapping):
                    raise ValueError(
                        "Stage-2B continuation checkpoint is missing its argument contract."
                    )
                validate_stage1_model_identity(resume_args, args.model_name_or_path)
                raw_resume_lineage = resume_checkpoint.get("lineage")
                continuation_contract = validate_stage2b_continuation_contract(
                    resume_args,
                    resume_lineage=(
                        raw_resume_lineage
                        if isinstance(raw_resume_lineage, Mapping)
                        else None
                    ),
                    current_args=args,
                )
                current_state = adapter.state_dict()
                frozen_global_audit = validate_frozen_global_resume_state(
                    resume_state,
                    current_state,
                )
                adapter.load_state_dict(resume_state, strict=True)
                raw_parent_payload = resume_checkpoint.get("metrics")
                raw_parent_val = (
                    raw_parent_payload.get("val")
                    if isinstance(raw_parent_payload, Mapping)
                    else None
                )
                if isinstance(raw_parent_val, Mapping):
                    parent_validation_metrics = copy.deepcopy(dict(raw_parent_val))
                args.stage2b_resume_checkpoint = str(
                    Path(resume_path).expanduser().resolve()
                )
                args.stage2b_resume_sha256 = sha256_file(Path(resume_path).expanduser())
                grounded_reader_initialized = False
                resume_report = {
                    "stage2b_resume_checkpoint": args.stage2b_resume_checkpoint,
                    "stage2b_resume_sha256": args.stage2b_resume_sha256,
                    "stage2b_resume_strict_load": True,
                    "stage2b_resume_parent_epoch": (
                        raw_parent_payload.get("epoch")
                        if isinstance(raw_parent_payload, Mapping)
                        else None
                    ),
                    "stage2b_resume_contract": continuation_contract,
                    "stage2b_resume_frozen_global_audit": frozen_global_audit,
                }
                initialization = "strict_grounded_stage2b_continuation"
                del resume_args, resume_state, resume_checkpoint
            checkpoint_load_report.update(
                {
                    "grounded_reader_initialized": grounded_reader_initialized,
                    "grounded_reader_parameters": sum(
                        int(parameter.numel()) for parameter in local_adapter.parameters()
                    ),
                    "global_adapter_frozen": bool(adapter.freeze_global),
                    **resume_report,
                }
            )
            args.soft_prompt_tokens = int(adapter.soft_prompt_tokens)
            args.local_soft_prompt_tokens = int(local_adapter.soft_prompt_tokens)
            args.local_adapter_layers = 1
            args.local_question_input_mode = "contextual_tokens"
            args.local_context_layer = int(max(context_layers))
            args.local_fusion_mode = str(local_adapter.fusion_mode)
            args.freeze_global_adapter = bool(adapter.freeze_global)
            args.global_unfreeze_epoch = 0
            args.global_prompt_dropout = 0.0
            args.question_conditioning = True
            args.structured_query_conditioning = False
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
                freeze_backbone=bool(args.freeze_conditioned_backbone),
                text_gate_trainable=bool(args.local_text_gate_trainable),
                residual_gate_trainable=bool(args.local_residual_gate_trainable),
                zero_init_text_attention=bool(args.zero_init_local_text_attention),
            )
            adapter = HybridGlobalLocalAdapter(
                global_adapter=global_adapter,
                local_adapter=local_adapter,
                freeze_global=True,
                global_prompt_dropout=float(args.global_prompt_dropout),
                combine_mode="residual",
            ).to(device)
            checkpoint_load_report["mode"] = (
                "stage1_frozen_backbone_question_residual"
                if bool(args.freeze_conditioned_backbone)
                else "stage1_cloned_residual_aligned_adapter"
            )
            checkpoint_load_report["conditioned_backbone_initialized_parameters"] = sum(
                int(parameter.numel()) for parameter in local_adapter.backbone.parameters()
            )
            checkpoint_load_report["conditioned_backbone_trainable_parameters"] = sum(
                int(parameter.numel())
                for parameter in local_adapter.backbone.parameters()
                if parameter.requires_grad
            )
            checkpoint_load_report["question_conditioning_trainable_parameters"] = sum(
                int(parameter.numel())
                for name, parameter in local_adapter.named_parameters()
                if not name.startswith("backbone.") and parameter.requires_grad
            )
            checkpoint_load_report["text_attention_output_nonzero_parameters"] = sum(
                int(torch.count_nonzero(block.attention.out_proj.weight).item())
                + (
                    int(torch.count_nonzero(block.attention.out_proj.bias).item())
                    if block.attention.out_proj.bias is not None
                    else 0
                )
                for block in local_adapter.text_blocks
            )
            checkpoint_load_report["text_gates_trainable"] = any(
                block.gate.requires_grad for block in local_adapter.text_blocks
            )
            checkpoint_load_report["residual_gate_trainable"] = bool(local_adapter.gate.requires_grad)
            checkpoint_load_report["text_gate_values"] = [
                float(block.gate.detach().float().cpu().item()) for block in local_adapter.text_blocks
            ]
            checkpoint_load_report["residual_gate_value"] = float(
                local_adapter.gate.detach().float().cpu().item()
            )
            if bool(args.freeze_conditioned_backbone) and int(
                checkpoint_load_report["conditioned_backbone_trainable_parameters"]
            ) != 0:
                raise RuntimeError("The configured frozen conditioned backbone still has trainable parameters.")
            if not bool(args.local_text_gate_trainable) and bool(
                checkpoint_load_report["text_gates_trainable"]
            ):
                raise RuntimeError("Fixed text gates unexpectedly remain trainable.")
            if not bool(args.local_residual_gate_trainable) and bool(
                checkpoint_load_report["residual_gate_trainable"]
            ):
                raise RuntimeError("The fixed residual gate unexpectedly remains trainable.")
            if bool(args.zero_init_local_text_attention) and int(
                checkpoint_load_report["text_attention_output_nonzero_parameters"]
            ) != 0:
                raise RuntimeError("Zero-initialized question reader contains nonzero attention output parameters.")
            initialization = (
                "stage1_frozen_backbone_question_residual"
                if bool(args.freeze_conditioned_backbone)
                else "stage1_residual_question_adapter"
            )
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
        # Strict loading has copied every deployable adapter tensor. Do not keep
        # the Stage-1 compressor and loss-only transform payload resident in host
        # memory for the rest of a long Stage-2 run.
        checkpoint = None
        checkpoint_args = {}
        hybrid_state_dict = None
        raw_checkpoint_state = None
        state_dict = None
        if init_checkpoint:
            loaded = None
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

    evidence_only_boundary: dict[str, Any] | None = None
    full_local_reader_boundary: dict[str, Any] | None = None
    if bool(args.point_reader_training):
        evidence_only_boundary = configure_evidence_only_training(adapter)
        checkpoint_load_report["evidence_only_training_boundary"] = dict(
            evidence_only_boundary
        )
        args.freeze_global_adapter = True
    elif bool(args.full_local_reader_training):
        full_local_reader_boundary = configure_full_grounded_local_training(adapter)
        checkpoint_load_report["full_local_reader_training_boundary"] = dict(
            full_local_reader_boundary
        )
        args.freeze_global_adapter = True
    if isinstance(adapter, HybridGlobalLocalAdapter):
        adapter.mask_inactive_local_tokens = bool(args.mask_inactive_local_tokens)
    synchronize_module_from_rank_zero(adapter)
    global_anchor_reference: dict[str, torch.Tensor] | None = None
    local_anchor_reference: dict[str, torch.Tensor] | None = None
    if bool(args.joint_ab_training):
        global_anchor_reference = snapshot_global_adapter_parameters(adapter)
        local_anchor_reference = snapshot_local_adapter_parameters(adapter)
    if isinstance(adapter, HybridGlobalLocalAdapter) and isinstance(
        adapter.local_adapter, ResidualQuestionConditionedAdapter
    ):
        if local_context_preflight is None:
            raise RuntimeError("Residual question conditioning was not covered by the contextual-token preflight.")
        preflight_question, preflight_question_mask = local_context_preflight
        adapter.eval()
        with torch.no_grad():
            initial_global, initial_residual, initial_combined = adapter.forward_components(
                first_latent.unsqueeze(0).to(device, non_blocking=True),
                question_embeds=preflight_question,
                question_mask=preflight_question_mask,
                structured_query=None,
            )
        initial_residual_max_abs = float(initial_residual.float().abs().max().cpu().item())
        initial_combined_error_max_abs = float(
            (initial_combined.float() - initial_global.float()).abs().max().cpu().item()
        )
        identity_tolerance = max(
            1.0e-7,
            1.0e-5 * float(initial_global.float().abs().max().cpu().item()),
        )
        checkpoint_load_report["initial_residual_max_abs"] = initial_residual_max_abs
        checkpoint_load_report["initial_combined_vs_global_max_abs"] = initial_combined_error_max_abs
        checkpoint_load_report["stage1_identity_tolerance"] = identity_tolerance
        if bool(args.zero_init_local_text_attention) and max(
            initial_residual_max_abs,
            initial_combined_error_max_abs,
        ) > identity_tolerance:
            raise RuntimeError(
                "The zero-initialized residual adapter does not reproduce Stage 1 before training: "
                f"residual_max_abs={initial_residual_max_abs:.3e}, "
                f"combined_error_max_abs={initial_combined_error_max_abs:.3e}, "
                f"tolerance={identity_tolerance:.3e}."
            )
        del initial_global, initial_residual, initial_combined
    del local_context_preflight
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
            prefetch_factor=1 if int(args.num_workers) > 0 else None,
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
            prefetch_factor=1 if int(args.num_workers) > 0 else None,
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
    optimizer_audit = optimizer_parameter_audit(
        optimizer,
        adapter,
        allow_frozen_parameters=False,
    )
    evidence_only_optimizer_audit = (
        audit_evidence_only_optimizer_boundary(optimizer, adapter)
        if bool(args.point_reader_training)
        else None
    )
    full_local_reader_optimizer_audit = (
        audit_full_grounded_local_optimizer_boundary(optimizer, adapter)
        if bool(args.full_local_reader_training)
        else None
    )
    local_groups = None
    global_groups = None
    accumulation_steps = max(1, int(args.gradient_accumulation_steps))
    updates_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
    total_optimizer_updates = max(1, updates_per_epoch * int(args.epochs))
    checkpoint_updates = [int(value) for value in parse_csv(args.checkpoint_updates)]
    checkpoint_fractions = [
        float(value) for value in parse_csv(args.checkpoint_fractions)
    ]
    if checkpoint_fractions:
        checkpoint_updates = checkpoint_updates_from_fractions(
            total_optimizer_updates,
            checkpoint_fractions,
        )
    if checkpoint_updates and max(checkpoint_updates) > total_optimizer_updates:
        raise ValueError(
            "A requested checkpoint update exceeds the configured training budget: "
            f"updates={checkpoint_updates}, total_optimizer_updates={total_optimizer_updates}."
        )
    if uses_screened_stage2b_training(args) and (
        not checkpoint_updates or checkpoint_updates[-1] != total_optimizer_updates
    ):
        raise ValueError(
            "The screened Stage-2B run must include its final optimizer update in its checkpoint schedule: "
            f"updates={checkpoint_updates}, total={total_optimizer_updates}."
        )
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
    optimizer_lr_prefix = (
        "adapter"
        if is_direct_alignment_architecture(str(args.adapter_architecture))
        else "local"
    )

    summary = {
        "device": str(device),
        "distributed": {
            "enabled": distributed_is_initialized(),
            "backend": dist.get_backend() if distributed_is_initialized() else None,
            "world_size": distributed_world_size(),
            "rank": distributed_rank(),
            "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
            "per_rank_batch_size": int(args.batch_size),
            "atomic_group_size": (
                int(args.questions_per_state_group)
                if str(args.adapter_architecture) == "grounded_evidence_adapter"
                else None
            ),
            "atomic_groups_per_rank_batch": (
                int(args.batch_size) // int(args.questions_per_state_group)
                if str(args.adapter_architecture) == "grounded_evidence_adapter"
                else None
            ),
            "train_choice_batch_size": int(args.train_choice_batch_size),
            "train_grounding_batch_size": int(args.train_grounding_batch_size),
            "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
            "effective_train_batch_size": (
                int(args.batch_size)
                * distributed_world_size()
                * int(args.gradient_accumulation_steps)
            ),
            "gradient_sync": "manual_all_reduce_adapter_only",
            "gradient_normalization": "record_weighted_manual_ddp",
            "evaluation_sharding": "exact_nonpadding",
            "timeout_seconds": float(args.distributed_timeout_seconds),
        },
        "grouped_batch_size_epoch_zero": (
            {
                "configured_max": int(args.batch_size),
                "minimum": int(train_epoch_sampler.initial_batch_size_min),
                "maximum": int(train_epoch_sampler.initial_batch_size_max),
                "mean": float(train_epoch_sampler.initial_batch_size_mean),
                "global_batches": int(train_epoch_sampler.initial_global_batch_count),
                "ddp_padding_batches": int(train_epoch_sampler.initial_padding_batch_count),
                "ddp_padding_records": int(train_epoch_sampler.initial_padding_record_count),
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
        "latent_channel_policy": str(args.latent_channel_policy),
        "num_workers": int(args.num_workers),
        "train_update_metrics": {
            "enabled": bool(args.save_step_metrics),
            "format": "tensor_stage2_train_update_v1",
            "path": "train_updates.jsonl" if bool(args.save_step_metrics) else None,
            "aggregation": "one globally reduced record-weighted row per optimizer update",
        },
        "host_memory": {
            "serialize_llm_loading": bool(args.serialize_llm_loading),
            "low_cpu_mem_usage": bool(args.low_cpu_mem_usage),
            "minimum_available_gib": float(args.min_host_memory_available_gib),
            "post_load_available_gib": (
                min(item["available_gib"] for item in post_load_host_memory)
                if post_load_host_memory
                else None
            ),
            "post_load_rank_rss_gib": [
                item["process_rss_gib"] for item in post_load_host_memory
            ],
            "latent_cache_budget_scope": "per_rank_shared_across_workers",
        },
        "shuffle_seed": int(args.shuffle_seed),
        "record_subset_mode": str(args.record_subset_mode),
        "shuffled_baseline_policy": "same_field_task_different_sample_then_fallback",
        "ce_loss_weight": float(args.ce_loss_weight),
        "choice_ce_loss_weight": float(args.choice_ce_loss_weight),
        "ranking_loss_weight": float(args.ranking_loss_weight),
        "ranking_loss_margin": float(args.ranking_loss_margin),
        "ranking_loss_negative": str(args.ranking_loss_negative),
        "swapped_question_loss_weight": float(args.swapped_question_loss_weight),
        "swapped_question_loss_margin": float(args.swapped_question_loss_margin),
        "grounding_routing_loss_weight": float(args.grounding_routing_loss_weight),
        "grounding_joint_routing_loss_weight": float(
            args.grounding_joint_routing_loss_weight
        ),
        "grounding_gate_loss_weight": float(args.grounding_gate_loss_weight),
        "matched_group_loss_weight": float(args.matched_group_loss_weight),
        "matched_group_loss_margin": float(args.matched_group_loss_margin),
        "joint_ab_training": bool(args.joint_ab_training),
        "point_reader_training": bool(args.point_reader_training),
        "full_local_reader_training": bool(args.full_local_reader_training),
        "joint_run_scope": str(args.joint_run_scope),
        "task_balanced_answer_loss": bool(args.task_balanced_answer_loss),
        "global_view_loss_weight": float(args.global_view_loss_weight),
        "joint_no_harm_loss_weight": float(args.joint_no_harm_loss_weight),
        "joint_no_harm_margin": float(args.joint_no_harm_margin),
        "joint_causal_loss_weight": float(args.joint_causal_loss_weight),
        "joint_causal_margin": float(args.joint_causal_margin),
        "point_causal_loss_weight": float(args.point_causal_loss_weight),
        "point_causal_margin": float(args.point_causal_margin),
        "point_causal_tasks": parse_csv(args.point_causal_tasks),
        "nonpoint_no_harm_loss_weight": float(
            args.nonpoint_no_harm_loss_weight
        ),
        "nonpoint_no_harm_margin": float(args.nonpoint_no_harm_margin),
        "global_anchor_loss_weight": float(args.global_anchor_loss_weight),
        "local_anchor_loss_weight": float(args.local_anchor_loss_weight),
        "grounding_routing_warmup_epochs": int(args.grounding_routing_warmup_epochs),
        "grounding_warmup_thresholds": {
            "cell_top1": float(args.grounding_warmup_min_cell_top1),
            "cell_top5": float(args.grounding_warmup_min_cell_top5),
            "axis_top1": float(args.grounding_warmup_min_axis_top1),
            "target_mass": float(args.grounding_warmup_min_target_mass),
            "gate_accuracy": float(args.grounding_warmup_min_gate_accuracy),
        },
        "swapped_question_max_records": int(args.swapped_question_max_records),
        "swapped_question_require_different_answer": bool(
            args.swapped_question_require_different_answer
        ),
        "grounding_score_space": "restricted_choice",
        "soft_prompt_tokens": int(args.soft_prompt_tokens),
        "adapter_layers": int(args.adapter_layers),
        "latent_pos_encoding": str(args.latent_pos_encoding),
        "question_conditioning": bool(args.question_conditioning),
        "question_condition_gate_init": float(args.question_condition_gate_init),
        "structured_query_conditioning": bool(args.structured_query_conditioning),
        "question_input_mode": (
            "direct_tensor_prefix_then_natural_language_prompt"
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
        "grounded_reader_geometry_initial": grounded_reader_geometry_metrics(adapter),
        "soft_prompt_scale": float(args.soft_prompt_scale),
        "mask_inactive_local_tokens": bool(args.mask_inactive_local_tokens),
        "adapter_architecture": str(args.adapter_architecture),
        "global_adapter_type": str(getattr(args, "global_adapter_type", "legacy")),
        "adapter_initialization": initialization,
        "adapter_init_checkpoint": str(args.adapter_init_checkpoint) if args.adapter_init_checkpoint else None,
        "stage2_warm_start_checkpoint": (
            str(args.stage2_warm_start_checkpoint) if args.stage2_warm_start_checkpoint else None
        ),
        "stage2_warm_start_sha256": getattr(args, "stage2_warm_start_sha256", None),
        "stage2b_resume_checkpoint": (
            str(args.stage2b_resume_checkpoint) if args.stage2b_resume_checkpoint else None
        ),
        "stage2b_resume_sha256": getattr(args, "stage2b_resume_sha256", None),
        "local_soft_prompt_tokens": int(args.local_soft_prompt_tokens),
        "local_adapter_layers": int(args.local_adapter_layers),
        "local_text_encoder_layers": int(args.local_text_encoder_layers),
        "local_gate_init": float(args.local_gate_init),
        "grounded_gate_bias_init": float(args.grounded_gate_bias_init),
        "local_text_gate_init": float(args.local_text_gate_init),
        "freeze_conditioned_backbone": bool(args.freeze_conditioned_backbone),
        "local_text_gate_trainable": bool(args.local_text_gate_trainable),
        "local_residual_gate_trainable": bool(args.local_residual_gate_trainable),
        "zero_init_local_text_attention": bool(args.zero_init_local_text_attention),
        "freeze_global_adapter": bool(args.freeze_global_adapter),
        "global_unfreeze_epoch": int(args.global_unfreeze_epoch),
        "global_lr": float(args.global_lr),
        "global_prompt_dropout": float(args.global_prompt_dropout),
        "group_questions_by_state": bool(args.group_questions_by_state),
        "questions_per_state_group": int(args.questions_per_state_group),
        "lr_scheduler": str(args.lr_scheduler),
        "warmup_updates": int(warmup_updates),
        "total_optimizer_updates": int(total_optimizer_updates),
        "checkpoint_updates": list(checkpoint_updates),
        "checkpoint_fractions": list(checkpoint_fractions),
        "checkpoint_screening_records": int(args.checkpoint_screening_records),
        "checkpoint_full_eval_top_k": int(args.checkpoint_full_eval_top_k),
        "joint_min_causal_gain": float(args.joint_min_causal_gain),
        "joint_max_parent_regression": float(args.joint_max_parent_regression),
        "joint_min_no_harm_delta": float(args.joint_min_no_harm_delta),
        "point_reader_min_parent_delta": float(args.point_reader_min_parent_delta),
        "point_reader_min_causal_gain": float(args.point_reader_min_causal_gain),
        "point_reader_max_nonpoint_regression": float(
            args.point_reader_max_nonpoint_regression
        ),
        "task_loss_weights": dict(getattr(args, "task_loss_weights", {})),
        "min_lr_ratio": float(args.min_lr_ratio),
        "global_dropout": float(getattr(args, "global_dropout", args.dropout)),
        "global_soft_prompt_scale": float(
            getattr(args, "global_soft_prompt_scale", args.soft_prompt_scale)
        ),
        "checkpoint_metric": str(args.checkpoint_metric),
        "diagnostics_generation_max_new_tokens": int(args.diagnostics_generation_max_new_tokens),
        "diagnostics_save_states": bool(args.diagnostics_save_states),
        "evaluate_test": bool(args.evaluate_test),
        "checkpoint_load_report": checkpoint_load_report,
        "optimizer_parameter_audit": optimizer_audit,
        "evidence_only_training_boundary": evidence_only_boundary,
        "evidence_only_optimizer_audit": evidence_only_optimizer_audit,
        "full_local_reader_training_boundary": full_local_reader_boundary,
        "full_local_reader_optimizer_audit": full_local_reader_optimizer_audit,
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
    def write_run_manifest() -> None:
        if lifecycle is None:
            raise RuntimeError("Rank 0 did not create a run lifecycle.")
        atomic_dump_json(run_dir / "args.json", redacted_args(args))
        atomic_dump_json(run_dir / "run_summary.json", summary)
        lifecycle._write("running")

    run_on_rank_zero_and_broadcast(write_run_manifest, "run manifest write")
    if is_main_process():
        if lifecycle is None:
            raise RuntimeError("Rank 0 did not create a run lifecycle.")
        print(
            f"run={run_dir.name} started_at={lifecycle.started_at} device={device} "
            f"distributed={int(distributed_is_initialized())} world_size={distributed_world_size()} "
            f"train/val/test={len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)} "
            f"question_input={summary['question_input_mode']} fusion={summary['local_fusion_mode']} "
            f"scheduler={summary['lr_scheduler']} grouped={int(summary['group_questions_by_state'])} "
            f"per_rank_batch={summary['distributed']['per_rank_batch_size']} "
            f"atomic_groups={summary['distributed']['atomic_groups_per_rank_batch']} "
            f"effective_batch={summary['distributed']['effective_train_batch_size']} "
            f"ddp_timeout={float(args.distributed_timeout_seconds):g}s "
            f"eval_batch={int(args.eval_batch_size)} "
            f"grouped_batch_range={summary['grouped_batch_size_epoch_zero']} "
            f"grounding_forward_batch={int(args.train_grounding_batch_size)} "
            f"global_dropout={summary['global_prompt_dropout']:.2f} "
            f"params={summary['trainable_adapter_parameters']:,} "
            f"prompt_max={max(item['max_tokens'] for item in prompt_audit['splits'].values())} "
            f"local_prompt_max={max(item['local_max_tokens'] for item in prompt_audit['splits'].values())} "
            f"loss_weights=ce:{float(args.ce_loss_weight):g},choice:{float(args.choice_ce_loss_weight):g},"
            f"ranking:{float(args.ranking_loss_weight):g},swap:{float(args.swapped_question_loss_weight):g} "
            f"grounding_score={summary['grounding_score_space']} "
            f"choice_path={choice_tokenization_audit['effective_training_path']} "
            f"checkpoint_load={checkpoint_load_report.get('mode')} "
            f"conditioned_backbone_trainable="
            f"{int(checkpoint_load_report.get('conditioned_backbone_trainable_parameters', 0))} "
            f"fixed_gates=residual:{float(checkpoint_load_report.get('residual_gate_value', 0.0)):g},"
            f"text:{','.join(str(value) for value in checkpoint_load_report.get('text_gate_values', []))} "
            f"stage1_identity_error="
            f"{float(checkpoint_load_report.get('initial_combined_vs_global_max_abs', 0.0)):.3e} "
            f"global/local_tensors={int(checkpoint_load_report.get('global_loaded_parameter_tensors', 0))}/"
            f"{int(checkpoint_load_report.get('local_loaded_parameter_tensors', 0))}"
        )
    wandb_config = build_wandb_config(args, summary)
    if not is_main_process():
        wandb_config["wandb"]["enabled"] = False
    wandb_logger = WandbLogger(config=wandb_config, run_dir=run_dir)

    best_val_score = -math.inf
    best_epoch = 0
    selected_checkpoint_path = run_dir / "adapter_best.pt"
    joint_selected_accepted: bool | None = None
    history: dict[str, Any] = {}
    global_step = 0
    screening_dataset: TensorReadoutQADataset | None = None
    screening_parent_metrics: dict[str, Any] | None = None
    joint_candidates: list[dict[str, Any]] = []
    try:
        initial_count = min(max(0, int(args.initial_eval_records)), len(val_dataset))
        if initial_count > 0:
            initial_dataset = TensorReadoutQADataset(
                qa_path(args.qa_dir, args.val_split),
                latent_dir=args.latent_dir,
                max_records=initial_count,
                subset_mode=str(args.record_subset_mode),
                subset_seed=int(args.shuffle_seed) + 1,
                prefer_record_latent_ref=bool(args.prefer_record_latent_ref),
                shuffle_seed=int(args.shuffle_seed),
                latent_cache_size=int(args.latent_cache_size),
                latent_contract=latent_contract,
                latent_channel_policy=str(args.latent_channel_policy),
            )
            if uses_screened_stage2b_training(args):
                screening_dataset = initial_dataset
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
            if uses_screened_stage2b_training(args):
                screening_parent_metrics = copy.deepcopy(initial_metrics)
            continuation_routing_audit: dict[str, Any] | None = None
            if (
                str(args.adapter_architecture) == "grounded_evidence_adapter"
                and bool(args.stage2b_resume_checkpoint)
                and int(args.grounding_routing_warmup_epochs) == 0
            ):
                continuation_routing_audit = grounded_routing_warmup_audit(
                    initial_metrics,
                    args,
                )
                history["continuation_routing_audit"] = continuation_routing_audit
            metrics_path = run_dir / "metrics_latest.json"
            run_on_rank_zero_and_broadcast(
                lambda: atomic_dump_json(metrics_path, history),
                "initial evaluation metrics write",
            )
            initial_payload = (
                flatten_numeric_metrics("initial_eval", initial_metrics)
                if bool(args.wandb_detailed_metrics)
                else compact_accuracy_metrics("initial_eval", initial_metrics)
            )
            if continuation_routing_audit is not None:
                initial_payload["continuation_routing_audit/passed"] = float(
                    bool(continuation_routing_audit["passed"])
                )
                initial_payload.update(
                    {
                        f"continuation_routing_audit/{name}": float(value)
                        for name, value in continuation_routing_audit["observed"].items()
                    }
                )
            log_wandb_on_rank_zero(
                wandb_logger,
                initial_payload,
                step=0,
                stage="initial evaluation W&B log",
            )
            if is_main_process():
                print_evaluation_summary("initial_eval", initial_metrics, metrics_path)
                if continuation_routing_audit is not None:
                    print(
                        "continuation_routing_audit="
                        f"{'passed' if continuation_routing_audit['passed'] else 'failed'} "
                        f"failed={','.join(continuation_routing_audit['failed']) or 'none'}"
                    )
            if continuation_routing_audit is not None and not bool(
                continuation_routing_audit["passed"]
            ):
                failed = ",".join(
                    str(value) for value in continuation_routing_audit["failed"]
                )
                raise RuntimeError(
                    "The Stage-2B continuation failed its held-out routing/gate audit "
                    f"({failed}); no optimizer step was taken."
                )
            if uses_screened_stage2b_training(args):
                if is_main_process():
                    print(
                        "stage2b_parent_full_eval=start "
                        f"records={len(val_dataset)} baselines={','.join(baseline_modes)}"
                    )
                parent_validation_metrics = evaluate_choice_accuracy(
                    llm=llm,
                    adapter=adapter,
                    tokenizer=tokenizer,
                    dataset=val_dataset,
                    device=device,
                    args=args,
                    baseline_modes=baseline_modes,
                )
                expected_task_counts: dict[str, int] = defaultdict(int)
                for record in val_dataset.records:
                    expected_task_counts[str(record.get("task_type", "unknown"))] += 1
                for mode in baseline_modes:
                    _mode_task_accuracies(parent_validation_metrics, mode)
                    mode_metrics = parent_validation_metrics.get(mode)
                    if not isinstance(mode_metrics, Mapping) or int(
                        mode_metrics.get("total", -1)
                    ) != len(val_dataset):
                        raise RuntimeError(
                            "Live parent validation did not cover the complete active split: "
                            f"mode={mode}, expected={len(val_dataset)}."
                        )
                    by_task = mode_metrics.get("by_task")
                    for task in STAGE2B_TASK_TYPES:
                        task_metrics = (
                            by_task.get(task) if isinstance(by_task, Mapping) else None
                        )
                        observed_total = (
                            task_metrics.get("total")
                            if isinstance(task_metrics, Mapping)
                            else None
                        )
                        if int(observed_total or -1) != expected_task_counts[task]:
                            raise RuntimeError(
                                "Live parent validation task coverage mismatch: "
                                f"mode={mode}, task={task}, observed={observed_total}, "
                                f"expected={expected_task_counts[task]}."
                            )
                history["joint_parent_full_eval"] = parent_validation_metrics
                summary["joint_parent_validation"] = {
                    "source": "live_full_eval_before_optimizer",
                    "records": len(val_dataset),
                    "baselines": list(baseline_modes),
                }
                run_on_rank_zero_and_broadcast(
                    lambda: atomic_dump_json(metrics_path, history),
                    "joint parent full-validation metrics write",
                )
                log_wandb_on_rank_zero(
                    wandb_logger,
                    (
                        flatten_numeric_metrics(
                            "joint_parent_full_eval", parent_validation_metrics
                        )
                        if bool(args.wandb_detailed_metrics)
                        else compact_accuracy_metrics(
                            "joint_parent_full_eval", parent_validation_metrics
                        )
                    ),
                    step=0,
                    stage="joint parent full-validation W&B log",
                )
                # Step 0 is a deployment fallback, not a trained child candidate.
                # Saving the live state here makes the selection result self-contained.
                parent_path = run_dir / "adapter_step_000000.pt"
                parent_selection = screened_stage2b_checkpoint_metrics(
                    parent_validation_metrics,
                    parent_validation_metrics,
                    args,
                )
                parent_routing_audit = grounded_routing_warmup_audit(
                    parent_validation_metrics,
                    args,
                )
                parent_selection["routing_gate_audit"] = parent_routing_audit
                parent_selection["acceptance"]["routing_gate_audit"] = bool(
                    parent_routing_audit["passed"]
                )
                parent_selection["accepted"] = all(
                    bool(value) for value in parent_selection["acceptance"].values()
                )
                parent_selection["eligible_for_promotion"] = False
                run_on_rank_zero_and_broadcast(
                    lambda: save_adapter_checkpoint(
                        parent_path,
                        adapter=adapter,
                        args=args,
                        latent_shape=latent_shape,
                        llm_hidden_size=llm_hidden_size,
                        latent_contract=latent_contract or {},
                        metrics={
                            "epoch": 0,
                            "global_step": 0,
                            "val": parent_validation_metrics,
                            "selection": parent_selection,
                        },
                    ),
                    "step-zero parent checkpoint write",
                )
                joint_candidates.append(
                    {
                        "path": str(parent_path),
                        "epoch": 0,
                        "global_step": 0,
                        "is_parent": True,
                        "screening_selection": screened_stage2b_checkpoint_metrics(
                            screening_parent_metrics,
                            screening_parent_metrics,
                            args,
                        ),
                        "full_val": copy.deepcopy(parent_validation_metrics),
                        "full_selection": parent_selection,
                    }
                )
            if bool(args.diagnostics_enabled):
                pretrain_diagnostic_aggregate = run_on_rank_zero_and_broadcast(
                    lambda: run_embedded_diagnostics(
                        stage="pretrain",
                        llm=llm,
                        adapter=adapter,
                        tokenizer=tokenizer,
                        dataset=val_dataset,
                        device=device,
                        args=args,
                        run_dir=run_dir,
                    )["aggregate"],
                    "pretrain diagnostics",
                )
                history["pretrain_diagnostics"] = dict(pretrain_diagnostic_aggregate)
                run_on_rank_zero_and_broadcast(
                    lambda: atomic_dump_json(metrics_path, history),
                    "pretrain diagnostic metrics write",
                )
                log_wandb_on_rank_zero(
                    wandb_logger,
                    compact_diagnostic_metrics(pretrain_diagnostic_aggregate),
                    step=0,
                    stage="pretrain diagnostics W&B log",
                )
            distributed_barrier()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        for epoch in range(1, int(args.epochs) + 1):
            routing_warmup_active = bool(
                str(args.adapter_architecture) == "grounded_evidence_adapter"
                and epoch <= int(args.grounding_routing_warmup_epochs)
            )
            training_phase = (
                "routing_warmup"
                if routing_warmup_active
                else (
                    "point_evidence"
                    if bool(args.point_reader_training)
                    else (
                        "full_local_reader"
                        if bool(args.full_local_reader_training)
                        else "joint_answer"
                    )
                )
            )
            evidence_optimizer_states_cleared = 0
            if (
                str(args.adapter_architecture) == "grounded_evidence_adapter"
                and int(args.grounding_routing_warmup_epochs) > 0
                and epoch == int(args.grounding_routing_warmup_epochs) + 1
            ):
                evidence_optimizer_states_cleared = reset_grounded_evidence_optimizer_state(
                    optimizer,
                    adapter,
                )
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
            running_routing_loss = 0.0
            running_weighted_routing_loss = 0.0
            running_routing_gate_loss = 0.0
            running_weighted_routing_gate_loss = 0.0
            running_routing_active_roles = 0.0
            running_routing_top1_correct = 0.0
            running_routing_top5_correct = 0.0
            running_routing_row_top1_correct = 0.0
            running_routing_col_top1_correct = 0.0
            running_routing_target_mass = 0.0
            running_routing_normalized_entropy = 0.0
            running_routing_gate_correct = 0.0
            running_routing_gate_active = 0.0
            running_routing_gate_target_active = 0.0
            running_matched_group_loss = 0.0
            running_weighted_matched_group_loss = 0.0
            running_matched_group_count = 0.0
            running_matched_group_exact = 0.0
            running_matched_group_pairs = 0.0
            running_matched_group_gap_sum = 0.0
            running_matched_group_satisfied = 0.0
            running_global_view_loss = 0.0
            running_weighted_global_view_loss = 0.0
            running_global_view_accuracy = 0.0
            running_joint_no_harm_loss = 0.0
            running_weighted_joint_no_harm_loss = 0.0
            running_joint_no_harm_margin = 0.0
            running_joint_causal_loss = 0.0
            running_weighted_joint_causal_loss = 0.0
            running_joint_causal_margin = 0.0
            running_joint_causal_active_records = 0.0
            running_global_anchor_loss = 0.0
            running_weighted_global_anchor_loss = 0.0
            running_local_anchor_loss = 0.0
            running_weighted_local_anchor_loss = 0.0
            running_record_count = 0
            running_total_grad_norm = 0.0
            running_post_clip_grad_norm = 0.0
            running_clipped_updates = 0
            pre_clip_grad_norms: list[float] = []
            post_clip_grad_norms: list[float] = []
            running_local_grad_norm = 0.0
            running_global_grad_norm = 0.0
            running_global_dropout_batches = 0
            optimizer_update_count = 0
            update_record_metric_names = (
                "loss",
                "ce_loss",
                "weighted_ce_loss",
                "choice_ce_loss",
                "weighted_choice_ce_loss",
                "choice_accuracy",
                "choice_01_loss",
                "ranking_loss",
                "weighted_ranking_loss",
                "ranking_margin_mean",
                "swapped_question_loss",
                "weighted_swapped_question_loss",
                "swapped_question_margin_mean",
                "routing_loss",
                "weighted_routing_loss",
                "routing_gate_loss",
                "weighted_routing_gate_loss",
                "matched_group_loss",
                "weighted_matched_group_loss",
                "global_view_loss",
                "weighted_global_view_loss",
                "global_view_accuracy",
                "joint_no_harm_loss",
                "weighted_joint_no_harm_loss",
                "joint_no_harm_margin_mean",
                "joint_causal_loss",
                "weighted_joint_causal_loss",
                "global_anchor_loss",
                "weighted_global_anchor_loss",
                "local_anchor_loss",
                "weighted_local_anchor_loss",
            )
            update_sums = {
                **{name: 0.0 for name in update_record_metric_names},
                "record_count": 0.0,
                "batch_count": 0.0,
                "global_dropout_batches": 0.0,
                "routing_active_roles": 0.0,
                "routing_top1_correct": 0.0,
                "routing_top5_correct": 0.0,
                "routing_row_top1_correct": 0.0,
                "routing_col_top1_correct": 0.0,
                "routing_target_mass_sum": 0.0,
                "routing_normalized_entropy_sum": 0.0,
                "routing_gate_correct": 0.0,
                "routing_gate_active": 0.0,
                "routing_gate_target_active": 0.0,
                "routing_gate_slots": 0.0,
                "matched_group_count": 0.0,
                "matched_group_exact": 0.0,
                "matched_group_pairs": 0.0,
                "matched_group_gap_sum": 0.0,
                "matched_group_satisfied": 0.0,
                "joint_causal_margin_sum": 0.0,
                "joint_causal_active_records": 0.0,
            }
            optimizer.zero_grad(set_to_none=True)
            progress = tqdm(
                train_loader,
                desc=f"Epoch {epoch:03d} [{training_phase}]",
                disable=not bool(args.console_progress) or not is_main_process(),
            )
            accumulated_local_records = 0
            for step, batch in enumerate(progress, start=1):
                batch_record_count = len(batch.get("records", ()))
                if batch_record_count <= 0:
                    raise RuntimeError(f"Training batch {step} contains no records.")
                drop_global_for_batch = bool(
                    isinstance(adapter, HybridGlobalLocalAdapter)
                    and float(args.global_prompt_dropout) > 0.0
                    and random.random() < float(args.global_prompt_dropout)
                )
                if isinstance(adapter, HybridGlobalLocalAdapter):
                    adapter.set_global_prompt_dropout_for_batch(drop_global_for_batch)
                backward_complete = False
                try:
                    if bool(args.joint_ab_training) and not routing_warmup_active:
                        global_loss, global_loss_parts, global_choice_margins = (
                            joint_global_view_training_loss(
                                llm=llm,
                                adapter=adapter,
                                tokenizer=tokenizer,
                                batch=batch,
                                device=device,
                                args=args,
                                global_anchor_reference=global_anchor_reference,
                            )
                        )
                        (global_loss * float(batch_record_count)).backward()
                        del global_loss
                        loss, loss_parts = training_loss(
                            llm=llm,
                            adapter=adapter,
                            tokenizer=tokenizer,
                            dataset=train_dataset,
                            batch=batch,
                            device=device,
                            args=args,
                            routing_only=False,
                            joint_global_margins=global_choice_margins,
                            joint_global_accuracy=float(
                                global_loss_parts["global_view_accuracy"]
                            ),
                            local_anchor_reference=local_anchor_reference,
                        )
                        (loss * float(batch_record_count)).backward()
                        loss = loss.detach()
                        loss_parts["loss"] += float(global_loss_parts["loss"])
                        for name, value in global_loss_parts.items():
                            if name != "loss":
                                loss_parts[name] = float(value)
                        del global_choice_margins
                        backward_complete = True
                    else:
                        loss, loss_parts = training_loss(
                            llm=llm,
                            adapter=adapter,
                            tokenizer=tokenizer,
                            dataset=train_dataset,
                            batch=batch,
                            device=device,
                            args=args,
                            routing_only=routing_warmup_active,
                        )
                finally:
                    if isinstance(adapter, HybridGlobalLocalAdapter):
                        adapter.set_global_prompt_dropout_for_batch(False)
                running_global_dropout_batches += int(drop_global_for_batch)
                # training_loss is a per-record mean. Accumulate record sums so that
                # variable grouped batches and a short final accumulation window do
                # not change the effective gradient scale.
                if not backward_complete:
                    (loss * float(batch_record_count)).backward()
                accumulated_local_records += int(batch_record_count)
                current_loss = float(loss_parts["loss"])
                running_loss += current_loss * batch_record_count
                running_ce_loss += float(loss_parts["ce_loss"]) * batch_record_count
                running_weighted_ce_loss += float(loss_parts["weighted_ce_loss"]) * batch_record_count
                running_choice_ce_loss += float(loss_parts["choice_ce_loss"]) * batch_record_count
                running_weighted_choice_ce_loss += float(loss_parts["weighted_choice_ce_loss"]) * batch_record_count
                running_choice_accuracy += float(loss_parts["choice_accuracy"]) * batch_record_count
                running_choice_01_loss += float(loss_parts["choice_01_loss"]) * batch_record_count
                running_ranking_loss += float(loss_parts["ranking_loss"]) * batch_record_count
                running_weighted_ranking_loss += float(loss_parts["weighted_ranking_loss"]) * batch_record_count
                running_ranking_margin += float(loss_parts["ranking_margin_mean"]) * batch_record_count
                running_swapped_question_loss += float(loss_parts["swapped_question_loss"]) * batch_record_count
                running_weighted_swapped_question_loss += float(loss_parts["weighted_swapped_question_loss"]) * batch_record_count
                running_swapped_question_margin += float(loss_parts["swapped_question_margin_mean"]) * batch_record_count
                running_swapped_question_pairs += float(loss_parts["swapped_question_pairs"]) * batch_record_count
                running_routing_loss += float(loss_parts["routing_loss"]) * batch_record_count
                running_weighted_routing_loss += float(loss_parts["weighted_routing_loss"]) * batch_record_count
                running_routing_gate_loss += float(loss_parts["routing_gate_loss"]) * batch_record_count
                running_weighted_routing_gate_loss += float(loss_parts["weighted_routing_gate_loss"]) * batch_record_count
                routing_batch_totals = routing_metric_weighted_totals(
                    loss_parts,
                    record_count=batch_record_count,
                    gate_slots_per_record=int(args.local_soft_prompt_tokens),
                )
                active_roles = routing_batch_totals["routing_active_roles"]
                running_routing_active_roles += active_roles
                running_routing_top1_correct += routing_batch_totals[
                    "routing_top1_correct"
                ]
                running_routing_top5_correct += routing_batch_totals[
                    "routing_top5_correct"
                ]
                running_routing_row_top1_correct += routing_batch_totals[
                    "routing_row_top1_correct"
                ]
                running_routing_col_top1_correct += routing_batch_totals[
                    "routing_col_top1_correct"
                ]
                running_routing_target_mass += routing_batch_totals[
                    "routing_target_mass_sum"
                ]
                running_routing_normalized_entropy += routing_batch_totals[
                    "routing_normalized_entropy_sum"
                ]
                running_routing_gate_correct += routing_batch_totals[
                    "routing_gate_correct"
                ]
                running_routing_gate_active += routing_batch_totals[
                    "routing_gate_active"
                ]
                running_routing_gate_target_active += routing_batch_totals[
                    "routing_gate_target_active"
                ]
                running_matched_group_loss += float(loss_parts["matched_group_loss"]) * batch_record_count
                running_weighted_matched_group_loss += float(loss_parts["weighted_matched_group_loss"]) * batch_record_count
                matched_groups = float(loss_parts["matched_group_count"])
                matched_pairs = float(loss_parts["matched_group_pairs"])
                running_matched_group_count += matched_groups
                running_matched_group_exact += float(loss_parts["matched_group_exact_accuracy"]) * matched_groups
                running_matched_group_pairs += matched_pairs
                running_matched_group_gap_sum += float(loss_parts["matched_group_gap_mean"]) * matched_pairs
                running_matched_group_satisfied += float(loss_parts["matched_group_satisfaction"]) * matched_pairs
                running_global_view_loss += float(loss_parts.get("global_view_loss", 0.0)) * batch_record_count
                running_weighted_global_view_loss += float(loss_parts.get("weighted_global_view_loss", 0.0)) * batch_record_count
                running_global_view_accuracy += float(loss_parts.get("global_view_accuracy", 0.0)) * batch_record_count
                running_joint_no_harm_loss += float(loss_parts.get("joint_no_harm_loss", 0.0)) * batch_record_count
                running_weighted_joint_no_harm_loss += float(loss_parts.get("weighted_joint_no_harm_loss", 0.0)) * batch_record_count
                running_joint_no_harm_margin += float(loss_parts.get("joint_no_harm_margin_mean", 0.0)) * batch_record_count
                running_joint_causal_loss += float(loss_parts.get("joint_causal_loss", 0.0)) * batch_record_count
                running_weighted_joint_causal_loss += float(loss_parts.get("weighted_joint_causal_loss", 0.0)) * batch_record_count
                causal_active_records = float(
                    loss_parts.get("joint_causal_active_records", 0.0)
                )
                running_joint_causal_margin += float(
                    loss_parts.get("joint_causal_margin_mean", 0.0)
                ) * causal_active_records
                running_joint_causal_active_records += causal_active_records
                running_global_anchor_loss += float(loss_parts.get("global_anchor_loss", 0.0)) * batch_record_count
                running_weighted_global_anchor_loss += float(loss_parts.get("weighted_global_anchor_loss", 0.0)) * batch_record_count
                running_local_anchor_loss += float(loss_parts.get("local_anchor_loss", 0.0)) * batch_record_count
                running_weighted_local_anchor_loss += float(loss_parts.get("weighted_local_anchor_loss", 0.0)) * batch_record_count
                running_record_count += int(batch_record_count)
                for name in update_record_metric_names:
                    update_sums[name] += float(loss_parts.get(name, 0.0)) * batch_record_count
                update_sums["record_count"] += float(batch_record_count)
                update_sums["batch_count"] += 1.0
                update_sums["global_dropout_batches"] += float(drop_global_for_batch)
                for name, value in routing_batch_totals.items():
                    update_sums[name] += value
                update_sums["matched_group_count"] += matched_groups
                update_sums["matched_group_exact"] += (
                    float(loss_parts["matched_group_exact_accuracy"]) * matched_groups
                )
                update_sums["matched_group_pairs"] += matched_pairs
                update_sums["matched_group_gap_sum"] += (
                    float(loss_parts["matched_group_gap_mean"]) * matched_pairs
                )
                update_sums["matched_group_satisfied"] += (
                    float(loss_parts["matched_group_satisfaction"]) * matched_pairs
                )
                update_sums["joint_causal_margin_sum"] += float(
                    loss_parts.get("joint_causal_margin_mean", 0.0)
                ) * causal_active_records
                update_sums["joint_causal_active_records"] += causal_active_records

                if step % accumulation_steps == 0 or step == len(train_loader):
                    update_global_record_count = average_trainable_gradients_by_record_count(
                        adapter,
                        accumulated_local_records,
                        device,
                    )
                    if routing_warmup_active:
                        clear_grounded_evidence_transform_gradients(adapter)
                    assert_finite_gradients(adapter, f"epoch {epoch} step {step}")
                    if isinstance(adapter, HybridGlobalLocalAdapter):
                        local_grad_norm = gradient_l2_norm(adapter.local_adapter.parameters())
                        global_grad_norm = gradient_l2_norm(adapter.global_adapter.parameters())
                        total_grad_norm = math.hypot(local_grad_norm, global_grad_norm)
                    else:
                        total_grad_norm = gradient_l2_norm(adapter.parameters())
                        local_grad_norm = total_grad_norm
                        global_grad_norm = 0.0
                    if not all(
                        math.isfinite(value)
                        for value in (total_grad_norm, local_grad_norm, global_grad_norm)
                    ):
                        raise FloatingPointError(
                            f"Non-finite gradient norm at epoch {epoch} step {step}: "
                            f"total={total_grad_norm}, local={local_grad_norm}, global={global_grad_norm}."
                        )
                    post_clip_grad_norm = total_grad_norm
                    if float(args.grad_clip_norm) > 0:
                        if bool(args.joint_ab_training) and isinstance(
                            adapter, HybridGlobalLocalAdapter
                        ):
                            # A large A-view gradient must not consume the B reader's
                            # clipping budget (or vice versa); the branches have
                            # separate objectives and learning rates.
                            torch.nn.utils.clip_grad_norm_(
                                adapter.local_adapter.parameters(),
                                float(args.grad_clip_norm),
                                error_if_nonfinite=True,
                            )
                            torch.nn.utils.clip_grad_norm_(
                                adapter.global_adapter.parameters(),
                                float(args.grad_clip_norm),
                                error_if_nonfinite=True,
                            )
                            post_clip_grad_norm = math.hypot(
                                gradient_l2_norm(adapter.local_adapter.parameters()),
                                gradient_l2_norm(adapter.global_adapter.parameters()),
                            )
                        else:
                            torch.nn.utils.clip_grad_norm_(
                                adapter.parameters(),
                                float(args.grad_clip_norm),
                                error_if_nonfinite=True,
                            )
                            post_clip_grad_norm = gradient_l2_norm(adapter.parameters())
                    pre_clip_grad_norms.append(float(total_grad_norm))
                    post_clip_grad_norms.append(float(post_clip_grad_norm))
                    running_clipped_updates += int(
                        float(args.grad_clip_norm) > 0
                        and (
                            max(local_grad_norm, global_grad_norm)
                            if bool(args.joint_ab_training)
                            else total_grad_norm
                        )
                        > float(args.grad_clip_norm)
                    )
                    running_total_grad_norm += total_grad_norm
                    running_post_clip_grad_norm += post_clip_grad_norm
                    running_local_grad_norm += local_grad_norm
                    running_global_grad_norm += global_grad_norm
                    optimizer_update_count += 1
                    update_learning_rate = optimizer_group_lr(
                        optimizer,
                        optimizer_lr_prefix,
                        float(args.lr),
                    )
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    accumulated_local_records = 0
                    global_step += 1
                    update_totals = distributed_sum_scalars(
                        {
                            **update_sums,
                            "grad_norm_pre_clip_sum": total_grad_norm,
                            "grad_norm_post_clip_sum": post_clip_grad_norm,
                            "local_grad_norm_sum": local_grad_norm,
                            "global_grad_norm_sum": global_grad_norm,
                            "clipped_ranks": float(
                                float(args.grad_clip_norm) > 0
                                and (
                                    max(local_grad_norm, global_grad_norm)
                                    if bool(args.joint_ab_training)
                                    else total_grad_norm
                                )
                                > float(args.grad_clip_norm)
                            ),
                            "contributing_ranks": 1.0,
                        },
                        device=device,
                    )
                    observed_update_records = int(round(update_totals["record_count"]))
                    if observed_update_records != int(update_global_record_count):
                        raise RuntimeError(
                            "Optimizer-update metric reduction disagrees with gradient normalization: "
                            f"metrics={observed_update_records}, gradients={update_global_record_count}."
                        )
                    update_records = max(1.0, update_totals["record_count"])
                    update_active_roles = max(1.0, update_totals["routing_active_roles"])
                    update_gate_slots = max(1.0, update_totals["routing_gate_slots"])
                    update_groups = max(1.0, update_totals["matched_group_count"])
                    update_pairs = max(1.0, update_totals["matched_group_pairs"])
                    update_active_local = max(
                        1.0, update_totals["joint_causal_active_records"]
                    )
                    update_ranks = max(1.0, update_totals["contributing_ranks"])
                    update_batches = max(1.0, update_totals["batch_count"])
                    update_payload = {
                        "format": "tensor_stage2_train_update_v1",
                        "timestamp": local_timestamp(),
                        "epoch": int(epoch),
                        "training_phase": training_phase,
                        "batch_step": int(step),
                        "optimizer_update_in_epoch": int(optimizer_update_count),
                        "global_step": int(global_step),
                        "global_record_count": int(round(update_totals["record_count"])),
                        "global_batch_count": int(round(update_totals["batch_count"])),
                        "train_loss": update_totals["loss"] / update_records,
                        "train_ce_loss": update_totals["ce_loss"] / update_records,
                        "train_weighted_ce_loss": (
                            update_totals["weighted_ce_loss"] / update_records
                        ),
                        "train_choice_ce_loss": (
                            update_totals["choice_ce_loss"] / update_records
                        ),
                        "train_weighted_choice_ce_loss": (
                            update_totals["weighted_choice_ce_loss"] / update_records
                        ),
                        "train_choice_accuracy": (
                            update_totals["choice_accuracy"] / update_records
                        ),
                        "train_choice_01_loss": (
                            update_totals["choice_01_loss"] / update_records
                        ),
                        "train_ranking_loss": update_totals["ranking_loss"] / update_records,
                        "train_weighted_ranking_loss": (
                            update_totals["weighted_ranking_loss"] / update_records
                        ),
                        "train_ranking_margin": (
                            update_totals["ranking_margin_mean"] / update_records
                        ),
                        "train_swapped_question_loss": (
                            update_totals["swapped_question_loss"] / update_records
                        ),
                        "train_weighted_swapped_question_loss": (
                            update_totals["weighted_swapped_question_loss"] / update_records
                        ),
                        "train_swapped_question_margin": (
                            update_totals["swapped_question_margin_mean"] / update_records
                        ),
                        "train_routing_loss": update_totals["routing_loss"] / update_records,
                        "train_weighted_routing_loss": (
                            update_totals["weighted_routing_loss"] / update_records
                        ),
                        "train_routing_gate_loss": (
                            update_totals["routing_gate_loss"] / update_records
                        ),
                        "train_weighted_routing_gate_loss": (
                            update_totals["weighted_routing_gate_loss"] / update_records
                        ),
                        "train_routing_active_roles": int(
                            round(update_totals["routing_active_roles"])
                        ),
                        "train_routing_top1_accuracy": (
                            update_totals["routing_top1_correct"] / update_active_roles
                        ),
                        "train_routing_top5_accuracy": (
                            update_totals["routing_top5_correct"] / update_active_roles
                        ),
                        "train_routing_row_top1_accuracy": (
                            update_totals["routing_row_top1_correct"] / update_active_roles
                        ),
                        "train_routing_col_top1_accuracy": (
                            update_totals["routing_col_top1_correct"] / update_active_roles
                        ),
                        "train_routing_target_mass": (
                            update_totals["routing_target_mass_sum"] / update_active_roles
                        ),
                        "train_routing_normalized_entropy": (
                            update_totals["routing_normalized_entropy_sum"]
                            / update_active_roles
                        ),
                        "train_routing_gate_accuracy": (
                            update_totals["routing_gate_correct"] / update_gate_slots
                        ),
                        "train_routing_gate_active_fraction": (
                            update_totals["routing_gate_active"] / update_gate_slots
                        ),
                        "train_routing_gate_target_active_fraction": (
                            update_totals["routing_gate_target_active"]
                            / update_gate_slots
                        ),
                        "train_matched_group_loss": (
                            update_totals["matched_group_loss"] / update_records
                        ),
                        "train_weighted_matched_group_loss": (
                            update_totals["weighted_matched_group_loss"] / update_records
                        ),
                        "train_matched_group_exact_accuracy": (
                            update_totals["matched_group_exact"] / update_groups
                        ),
                        "train_matched_group_gap_mean": (
                            update_totals["matched_group_gap_sum"] / update_pairs
                        ),
                        "train_matched_group_satisfaction": (
                            update_totals["matched_group_satisfied"] / update_pairs
                        ),
                        **{
                            f"train_{name}": update_totals[name] / update_records
                            for name in (
                                "global_view_loss",
                                "weighted_global_view_loss",
                                "global_view_accuracy",
                                "joint_no_harm_loss",
                                "weighted_joint_no_harm_loss",
                                "joint_no_harm_margin_mean",
                                "joint_causal_loss",
                                "weighted_joint_causal_loss",
                                "global_anchor_loss",
                                "weighted_global_anchor_loss",
                                "local_anchor_loss",
                                "weighted_local_anchor_loss",
                            )
                        },
                        "train_joint_causal_margin_mean": (
                            update_totals["joint_causal_margin_sum"]
                            / update_active_local
                        ),
                        "train_joint_causal_active_records": int(
                            round(update_totals["joint_causal_active_records"])
                        ),
                        "train_total_grad_norm": (
                            update_totals["grad_norm_pre_clip_sum"] / update_ranks
                        ),
                        "train_post_clip_grad_norm": (
                            update_totals["grad_norm_post_clip_sum"] / update_ranks
                        ),
                        "train_clip_fraction": update_totals["clipped_ranks"] / update_ranks,
                        "train_local_grad_norm": (
                            update_totals["local_grad_norm_sum"] / update_ranks
                        ),
                        "train_global_grad_norm": (
                            update_totals["global_grad_norm_sum"] / update_ranks
                        ),
                        "train_global_prompt_dropout_rate": (
                            update_totals["global_dropout_batches"] / update_batches
                        ),
                        "lr": update_learning_rate,
                    }
                    host_memory_reports: list[dict[str, float]] = []
                    if global_step % max(1, int(args.log_interval)) == 0:
                        host_memory_reports = enforce_host_memory_floor(
                            device,
                            float(args.min_host_memory_available_gib),
                            f"epoch {epoch} update {global_step}",
                        )
                        if host_memory_reports:
                            update_payload["host_memory_available_gib"] = min(
                                item["available_gib"] for item in host_memory_reports
                            )
                            update_payload["host_process_rss_gib"] = host_memory_reports[
                                distributed_rank()
                            ]["process_rss_gib"]
                    if bool(args.save_step_metrics):
                        run_on_rank_zero_and_broadcast(
                            lambda payload=dict(update_payload): append_jsonl(
                                run_dir / "train_updates.jsonl",
                                payload,
                            ),
                            f"epoch {epoch} update {global_step} metrics append",
                        )
                    if global_step % max(1, int(args.log_interval)) == 0:
                        log_wandb_on_rank_zero(
                            wandb_logger,
                            {
                                "train_step/loss": update_payload["train_loss"],
                                "train_step/choice_ce_loss": update_payload[
                                    "train_choice_ce_loss"
                                ],
                                "train_step/choice_accuracy": update_payload[
                                    "train_choice_accuracy"
                                ],
                                "train_step/ranking_loss": update_payload[
                                    "train_ranking_loss"
                                ],
                                "train_step/ranking_margin": update_payload[
                                    "train_ranking_margin"
                                ],
                                "train_step/swapped_question_loss": update_payload[
                                    "train_swapped_question_loss"
                                ],
                                "train_step/swapped_question_margin": update_payload[
                                    "train_swapped_question_margin"
                                ],
                                "train_step/routing_loss": update_payload[
                                    "train_routing_loss"
                                ],
                                "train_step/routing_top1_accuracy": update_payload[
                                    "train_routing_top1_accuracy"
                                ],
                                "train_step/routing_top5_accuracy": update_payload[
                                    "train_routing_top5_accuracy"
                                ],
                                "train_step/routing_row_top1_accuracy": update_payload[
                                    "train_routing_row_top1_accuracy"
                                ],
                                "train_step/routing_col_top1_accuracy": update_payload[
                                    "train_routing_col_top1_accuracy"
                                ],
                                "train_step/routing_target_mass": update_payload[
                                    "train_routing_target_mass"
                                ],
                                "train_step/routing_normalized_entropy": update_payload[
                                    "train_routing_normalized_entropy"
                                ],
                                "train_step/routing_gate_accuracy": update_payload[
                                    "train_routing_gate_accuracy"
                                ],
                                "train_step/routing_gate_target_active_fraction": update_payload[
                                    "train_routing_gate_target_active_fraction"
                                ],
                                "train_step/matched_group_loss": update_payload[
                                    "train_matched_group_loss"
                                ],
                                "train_step/matched_group_exact_accuracy": update_payload[
                                    "train_matched_group_exact_accuracy"
                                ],
                                "train_step/global_view_loss": update_payload[
                                    "train_global_view_loss"
                                ],
                                "train_step/global_view_accuracy": update_payload[
                                    "train_global_view_accuracy"
                                ],
                                "train_step/joint_no_harm_loss": update_payload[
                                    "train_joint_no_harm_loss"
                                ],
                                "train_step/joint_no_harm_margin": update_payload[
                                    "train_joint_no_harm_margin_mean"
                                ],
                                "train_step/joint_causal_loss": update_payload[
                                    "train_joint_causal_loss"
                                ],
                                "train_step/joint_causal_margin": update_payload[
                                    "train_joint_causal_margin_mean"
                                ],
                                "train_step/global_anchor_loss": update_payload[
                                    "train_global_anchor_loss"
                                ],
                                "train_step/local_anchor_loss": update_payload[
                                    "train_local_anchor_loss"
                                ],
                                "train_step/grad_norm_pre_clip": update_payload[
                                    "train_total_grad_norm"
                                ],
                                "train_step/grad_norm_post_clip": update_payload[
                                    "train_post_clip_grad_norm"
                                ],
                                "train_step/clip_fraction": update_payload[
                                    "train_clip_fraction"
                                ],
                                "train_step/lr": update_payload["lr"],
                                **(
                                    {
                                        "system/host_memory_available_gib": update_payload[
                                            "host_memory_available_gib"
                                        ]
                                    }
                                    if "host_memory_available_gib" in update_payload
                                    else {}
                                ),
                            },
                            step=global_step,
                            stage=f"epoch {epoch} update {global_step} W&B log",
                        )
                    if uses_screened_stage2b_training(args) and global_step in checkpoint_updates:
                        if screening_dataset is None or screening_parent_metrics is None:
                            raise RuntimeError(
                                "Joint checkpoint screening has no fixed parent validation subset."
                            )
                        screening_metrics = evaluate_choice_accuracy(
                            llm=llm,
                            adapter=adapter,
                            tokenizer=tokenizer,
                            dataset=screening_dataset,
                            device=device,
                            args=args,
                            baseline_modes=baseline_modes,
                        )
                        screening_selection = screened_stage2b_checkpoint_metrics(
                            screening_metrics,
                            screening_parent_metrics,
                            args,
                        )
                        candidate_path = run_dir / f"adapter_step_{global_step:06d}.pt"
                        candidate_payload = {
                            "epoch": int(epoch),
                            "global_step": int(global_step),
                            "screening_records": len(screening_dataset),
                            "screening_val": screening_metrics,
                            "selection": screening_selection,
                        }
                        run_on_rank_zero_and_broadcast(
                            lambda path=candidate_path, payload=candidate_payload: save_adapter_checkpoint(
                                path,
                                adapter=adapter,
                                args=args,
                                latent_shape=latent_shape,
                                llm_hidden_size=llm_hidden_size,
                                latent_contract=latent_contract or {},
                                metrics=payload,
                            ),
                            f"update {global_step} candidate-checkpoint write",
                        )
                        joint_candidates.append(
                            {
                                "path": str(candidate_path),
                                "epoch": int(epoch),
                                "global_step": int(global_step),
                                "screening_selection": screening_selection,
                            }
                        )
                        history[f"screening_step_{global_step:06d}"] = candidate_payload
                        run_on_rank_zero_and_broadcast(
                            lambda: atomic_dump_json(run_dir / "metrics_latest.json", history),
                            f"update {global_step} screening metrics write",
                        )
                        log_wandb_on_rank_zero(
                            wandb_logger,
                            flatten_numeric_metrics(
                                "screening",
                                screening_selection,
                            ),
                            step=global_step,
                            stage=f"update {global_step} checkpoint screening W&B log",
                        )
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                    for name in update_sums:
                        update_sums[name] = 0.0

                average_loss = running_loss / max(1, running_record_count)
                average_ce_loss = running_ce_loss / max(1, running_record_count)
                average_weighted_ce_loss = running_weighted_ce_loss / max(1, running_record_count)
                average_choice_ce_loss = running_choice_ce_loss / max(1, running_record_count)
                average_weighted_choice_ce_loss = running_weighted_choice_ce_loss / max(1, running_record_count)
                average_choice_accuracy = running_choice_accuracy / max(1, running_record_count)
                average_choice_01_loss = running_choice_01_loss / max(1, running_record_count)
                average_ranking_loss = running_ranking_loss / max(1, running_record_count)
                average_weighted_ranking_loss = running_weighted_ranking_loss / max(1, running_record_count)
                average_ranking_margin = running_ranking_margin / max(1, running_record_count)
                average_swapped_question_loss = running_swapped_question_loss / max(1, running_record_count)
                average_swapped_question_margin = running_swapped_question_margin / max(1, running_record_count)
                average_routing_loss = running_routing_loss / max(1, running_record_count)
                average_routing_top1 = running_routing_top1_correct / max(1.0, running_routing_active_roles)
                average_routing_top5 = running_routing_top5_correct / max(
                    1.0, running_routing_active_roles
                )
                average_routing_row_top1 = running_routing_row_top1_correct / max(
                    1.0, running_routing_active_roles
                )
                average_routing_col_top1 = running_routing_col_top1_correct / max(
                    1.0, running_routing_active_roles
                )
                average_routing_target_mass = running_routing_target_mass / max(
                    1.0, running_routing_active_roles
                )
                running_gate_slots = max(
                    1.0,
                    float(running_record_count * int(args.local_soft_prompt_tokens)),
                )
                average_routing_gate_accuracy = (
                    running_routing_gate_correct / running_gate_slots
                )
                average_matched_group_loss = running_matched_group_loss / max(1, running_record_count)
                average_matched_group_exact = running_matched_group_exact / max(1.0, running_matched_group_count)
                average_total_grad_norm = running_total_grad_norm / max(1, optimizer_update_count)
                average_post_clip_grad_norm = running_post_clip_grad_norm / max(1, optimizer_update_count)
                clip_fraction = running_clipped_updates / max(1, optimizer_update_count)
                average_local_grad_norm = running_local_grad_norm / max(1, optimizer_update_count)
                average_global_grad_norm = running_global_grad_norm / max(1, optimizer_update_count)
                progress.set_postfix(
                    loss=f"{average_loss:.4f}",
                    ce=f"{average_ce_loss:.4f}",
                    choice=f"{average_choice_ce_loss:.4f}",
                    acc=f"{average_choice_accuracy:.3f}",
                    rank=f"{average_ranking_loss:.4f}",
                    route=f"{average_routing_top1:.3f}",
                    group=f"{average_matched_group_exact:.3f}",
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
                    "routing_loss": running_routing_loss,
                    "weighted_routing_loss": running_weighted_routing_loss,
                    "routing_gate_loss": running_routing_gate_loss,
                    "weighted_routing_gate_loss": running_weighted_routing_gate_loss,
                    "routing_active_roles": running_routing_active_roles,
                    "routing_top1_correct": running_routing_top1_correct,
                    "routing_top5_correct": running_routing_top5_correct,
                    "routing_row_top1_correct": running_routing_row_top1_correct,
                    "routing_col_top1_correct": running_routing_col_top1_correct,
                    "routing_target_mass": running_routing_target_mass,
                    "routing_normalized_entropy": running_routing_normalized_entropy,
                    "routing_gate_correct": running_routing_gate_correct,
                    "routing_gate_active": running_routing_gate_active,
                    "routing_gate_target_active": running_routing_gate_target_active,
                    "routing_gate_slots": float(
                        running_record_count * int(args.local_soft_prompt_tokens)
                    ),
                    "matched_group_loss": running_matched_group_loss,
                    "weighted_matched_group_loss": running_weighted_matched_group_loss,
                    "matched_group_count": running_matched_group_count,
                    "matched_group_exact": running_matched_group_exact,
                    "matched_group_pairs": running_matched_group_pairs,
                    "matched_group_gap_sum": running_matched_group_gap_sum,
                    "matched_group_satisfied": running_matched_group_satisfied,
                    "global_view_loss": running_global_view_loss,
                    "weighted_global_view_loss": running_weighted_global_view_loss,
                    "global_view_accuracy": running_global_view_accuracy,
                    "joint_no_harm_loss": running_joint_no_harm_loss,
                    "weighted_joint_no_harm_loss": running_weighted_joint_no_harm_loss,
                    "joint_no_harm_margin": running_joint_no_harm_margin,
                    "joint_causal_loss": running_joint_causal_loss,
                    "weighted_joint_causal_loss": running_weighted_joint_causal_loss,
                    "joint_causal_margin": running_joint_causal_margin,
                    "joint_causal_active_records": running_joint_causal_active_records,
                    "global_anchor_loss": running_global_anchor_loss,
                    "weighted_global_anchor_loss": running_weighted_global_anchor_loss,
                    "local_anchor_loss": running_local_anchor_loss,
                    "weighted_local_anchor_loss": running_weighted_local_anchor_loss,
                    "total_grad_norm": running_total_grad_norm,
                    "post_clip_grad_norm": running_post_clip_grad_norm,
                    "clipped_updates": float(running_clipped_updates),
                    "local_grad_norm": running_local_grad_norm,
                    "global_grad_norm": running_global_grad_norm,
                    "global_dropout_batches": float(running_global_dropout_batches),
                    "batch_count": float(len(train_loader)),
                    "record_count": float(running_record_count),
                    "optimizer_update_count": float(optimizer_update_count),
                },
                device=device,
            )
            global_batch_count = max(1.0, train_totals["batch_count"])
            global_record_count = max(1.0, train_totals["record_count"])
            global_update_count = max(1.0, train_totals["optimizer_update_count"])
            train_loss = train_totals["loss"] / global_record_count
            train_ce_loss = train_totals["ce_loss"] / global_record_count
            train_weighted_ce_loss = train_totals["weighted_ce_loss"] / global_record_count
            train_choice_ce_loss = train_totals["choice_ce_loss"] / global_record_count
            train_weighted_choice_ce_loss = (
                train_totals["weighted_choice_ce_loss"] / global_record_count
            )
            train_choice_accuracy = train_totals["choice_accuracy"] / global_record_count
            train_choice_01_loss = train_totals["choice_01_loss"] / global_record_count
            train_ranking_loss = train_totals["ranking_loss"] / global_record_count
            train_weighted_ranking_loss = (
                train_totals["weighted_ranking_loss"] / global_record_count
            )
            train_ranking_margin = train_totals["ranking_margin"] / global_record_count
            train_swapped_question_loss = (
                train_totals["swapped_question_loss"] / global_record_count
            )
            train_weighted_swapped_question_loss = (
                train_totals["weighted_swapped_question_loss"] / global_record_count
            )
            train_swapped_question_margin = (
                train_totals["swapped_question_margin"] / global_record_count
            )
            train_swapped_question_pairs = (
                train_totals["swapped_question_pairs"] / global_record_count
            )
            train_routing_loss = train_totals["routing_loss"] / global_record_count
            train_weighted_routing_loss = train_totals["weighted_routing_loss"] / global_record_count
            train_routing_gate_loss = train_totals["routing_gate_loss"] / global_record_count
            train_weighted_routing_gate_loss = (
                train_totals["weighted_routing_gate_loss"] / global_record_count
            )
            train_routing_top1_accuracy = train_totals["routing_top1_correct"] / max(
                1.0, train_totals["routing_active_roles"]
            )
            train_routing_top5_accuracy = train_totals["routing_top5_correct"] / max(
                1.0, train_totals["routing_active_roles"]
            )
            train_routing_row_top1_accuracy = train_totals[
                "routing_row_top1_correct"
            ] / max(1.0, train_totals["routing_active_roles"])
            train_routing_col_top1_accuracy = train_totals[
                "routing_col_top1_correct"
            ] / max(1.0, train_totals["routing_active_roles"])
            train_routing_target_mass = train_totals["routing_target_mass"] / max(
                1.0, train_totals["routing_active_roles"]
            )
            train_routing_normalized_entropy = train_totals[
                "routing_normalized_entropy"
            ] / max(1.0, train_totals["routing_active_roles"])
            train_routing_gate_accuracy = train_totals["routing_gate_correct"] / max(
                1.0, train_totals["routing_gate_slots"]
            )
            train_routing_gate_active_fraction = train_totals[
                "routing_gate_active"
            ] / max(1.0, train_totals["routing_gate_slots"])
            train_routing_gate_target_active_fraction = train_totals[
                "routing_gate_target_active"
            ] / max(1.0, train_totals["routing_gate_slots"])
            train_matched_group_loss = train_totals["matched_group_loss"] / global_record_count
            train_weighted_matched_group_loss = (
                train_totals["weighted_matched_group_loss"] / global_record_count
            )
            train_matched_group_exact_accuracy = train_totals["matched_group_exact"] / max(
                1.0, train_totals["matched_group_count"]
            )
            train_matched_group_gap_mean = train_totals["matched_group_gap_sum"] / max(
                1.0, train_totals["matched_group_pairs"]
            )
            train_matched_group_satisfaction = train_totals["matched_group_satisfied"] / max(
                1.0, train_totals["matched_group_pairs"]
            )
            train_global_view_loss = train_totals["global_view_loss"] / global_record_count
            train_weighted_global_view_loss = (
                train_totals["weighted_global_view_loss"] / global_record_count
            )
            train_global_view_accuracy = (
                train_totals["global_view_accuracy"] / global_record_count
            )
            train_joint_no_harm_loss = train_totals["joint_no_harm_loss"] / global_record_count
            train_weighted_joint_no_harm_loss = (
                train_totals["weighted_joint_no_harm_loss"] / global_record_count
            )
            train_joint_no_harm_margin = (
                train_totals["joint_no_harm_margin"] / global_record_count
            )
            train_joint_causal_loss = train_totals["joint_causal_loss"] / global_record_count
            train_weighted_joint_causal_loss = (
                train_totals["weighted_joint_causal_loss"] / global_record_count
            )
            train_joint_causal_margin = (
                train_totals["joint_causal_margin"]
                / max(1.0, train_totals["joint_causal_active_records"])
            )
            train_global_anchor_loss = train_totals["global_anchor_loss"] / global_record_count
            train_weighted_global_anchor_loss = (
                train_totals["weighted_global_anchor_loss"] / global_record_count
            )
            train_local_anchor_loss = train_totals["local_anchor_loss"] / global_record_count
            train_weighted_local_anchor_loss = (
                train_totals["weighted_local_anchor_loss"] / global_record_count
            )
            train_total_grad_norm = train_totals["total_grad_norm"] / global_update_count
            train_post_clip_grad_norm = train_totals["post_clip_grad_norm"] / global_update_count
            train_clip_fraction = train_totals["clipped_updates"] / global_update_count
            train_local_grad_norm = train_totals["local_grad_norm"] / global_update_count
            train_global_grad_norm = train_totals["global_grad_norm"] / global_update_count
            train_global_dropout_rate = (
                train_totals["global_dropout_batches"] / global_batch_count
            )
            if uses_screened_stage2b_training(args) and not routing_warmup_active:
                final_screen = history.get(f"screening_step_{global_step:06d}")
                if not isinstance(final_screen, Mapping) or not isinstance(
                    final_screen.get("screening_val"), Mapping
                ):
                    raise RuntimeError(
                        "The final joint optimizer update was not screened before epoch aggregation."
                    )
                val_metrics = copy.deepcopy(dict(final_screen["screening_val"]))
                validation_scope = "screening"
            else:
                val_metrics = evaluate_choice_accuracy(
                    llm=llm,
                    adapter=adapter,
                    tokenizer=tokenizer,
                    dataset=val_dataset,
                    device=device,
                    args=args,
                    baseline_modes=["correct"] if routing_warmup_active else baseline_modes,
                    routing_only=routing_warmup_active,
                )
                validation_scope = "routing_warmup" if routing_warmup_active else "full"
            routing_warmup_audit: dict[str, Any] | None = None
            if routing_warmup_active and epoch == int(args.grounding_routing_warmup_epochs):
                routing_warmup_audit = grounded_routing_warmup_audit(val_metrics, args)
            epoch_payload = {
                "epoch": epoch,
                "training_phase": training_phase,
                "evidence_optimizer_states_cleared": evidence_optimizer_states_cleared,
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
                "train_routing_loss": train_routing_loss,
                "train_weighted_routing_loss": train_weighted_routing_loss,
                "train_routing_gate_loss": train_routing_gate_loss,
                "train_weighted_routing_gate_loss": train_weighted_routing_gate_loss,
                "train_routing_top1_accuracy": train_routing_top1_accuracy,
                "train_routing_top5_accuracy": train_routing_top5_accuracy,
                "train_routing_row_top1_accuracy": train_routing_row_top1_accuracy,
                "train_routing_col_top1_accuracy": train_routing_col_top1_accuracy,
                "train_routing_target_mass": train_routing_target_mass,
                "train_routing_normalized_entropy": train_routing_normalized_entropy,
                "train_routing_gate_accuracy": train_routing_gate_accuracy,
                "train_routing_gate_active_fraction": train_routing_gate_active_fraction,
                "train_routing_gate_target_active_fraction": (
                    train_routing_gate_target_active_fraction
                ),
                "train_matched_group_loss": train_matched_group_loss,
                "train_weighted_matched_group_loss": train_weighted_matched_group_loss,
                "train_matched_group_exact_accuracy": train_matched_group_exact_accuracy,
                "train_matched_group_gap_mean": train_matched_group_gap_mean,
                "train_matched_group_satisfaction": train_matched_group_satisfaction,
                "train_global_view_loss": train_global_view_loss,
                "train_weighted_global_view_loss": train_weighted_global_view_loss,
                "train_global_view_accuracy": train_global_view_accuracy,
                "train_joint_no_harm_loss": train_joint_no_harm_loss,
                "train_weighted_joint_no_harm_loss": train_weighted_joint_no_harm_loss,
                "train_joint_no_harm_margin": train_joint_no_harm_margin,
                "train_joint_causal_loss": train_joint_causal_loss,
                "train_weighted_joint_causal_loss": train_weighted_joint_causal_loss,
                "train_joint_causal_margin": train_joint_causal_margin,
                "train_joint_causal_active_records": int(
                    round(train_totals["joint_causal_active_records"])
                ),
                "train_global_anchor_loss": train_global_anchor_loss,
                "train_weighted_global_anchor_loss": train_weighted_global_anchor_loss,
                "train_local_anchor_loss": train_local_anchor_loss,
                "train_weighted_local_anchor_loss": train_weighted_local_anchor_loss,
                "train_total_grad_norm": train_total_grad_norm,
                "train_post_clip_grad_norm": train_post_clip_grad_norm,
                "train_clip_fraction": train_clip_fraction,
                "train_pre_clip_grad_norm_p50": numeric_quantile(pre_clip_grad_norms, 0.50),
                "train_pre_clip_grad_norm_p95": numeric_quantile(pre_clip_grad_norms, 0.95),
                "train_pre_clip_grad_norm_max": max(pre_clip_grad_norms, default=0.0),
                "train_post_clip_grad_norm_p50": numeric_quantile(post_clip_grad_norms, 0.50),
                "train_post_clip_grad_norm_p95": numeric_quantile(post_clip_grad_norms, 0.95),
                "train_post_clip_grad_norm_max": max(post_clip_grad_norms, default=0.0),
                "train_local_grad_norm": train_local_grad_norm,
                "train_global_grad_norm": train_global_grad_norm,
                "train_global_prompt_dropout_rate": train_global_dropout_rate,
                "grounded_reader_geometry": grounded_reader_geometry_metrics(adapter),
                "validation_scope": validation_scope,
            }
            if validation_scope == "screening":
                epoch_payload["screening_val"] = val_metrics
            else:
                epoch_payload["val"] = val_metrics
            if routing_warmup_audit is not None:
                epoch_payload["routing_warmup_audit"] = routing_warmup_audit
            history[f"epoch_{epoch:04d}"] = epoch_payload

            def write_epoch_outputs() -> None:
                atomic_dump_json(run_dir / "metrics_latest.json", history)
                save_adapter_checkpoint(
                    run_dir / "adapter_last.pt",
                    adapter=adapter,
                    args=args,
                    latent_shape=latent_shape,
                    llm_hidden_size=llm_hidden_size,
                    latent_contract=latent_contract or {},
                    metrics=epoch_payload,
                )

            run_on_rank_zero_and_broadcast(
                write_epoch_outputs,
                f"epoch {epoch} metrics and last-checkpoint write",
            )
            if routing_warmup_audit is not None:
                run_on_rank_zero_and_broadcast(
                    lambda: save_validate_and_rebuild_adapter_checkpoint(
                        run_dir / "adapter_routing_warmup.pt",
                        adapter=adapter,
                        args=args,
                        latent_shape=latent_shape,
                        llm_hidden_size=llm_hidden_size,
                        latent_contract=latent_contract or {},
                        metrics=epoch_payload,
                    ),
                    f"epoch {epoch} routing-warmup checkpoint write/read validation",
                )
            val_accuracy = float(val_metrics.get("correct", {}).get("accuracy", 0.0))
            val_macro_latent_gain = (
                0.0 if routing_warmup_active else macro_latent_gain(val_metrics)
            )
            val_score = (
                -math.inf
                if routing_warmup_active
                else checkpoint_score(
                    val_metrics,
                    str(args.checkpoint_metric),
                    reference_metrics=(
                        screening_parent_metrics
                        if uses_screened_stage2b_training(args)
                        else None
                    ),
                )
            )
            wandb_payload = {
                "epoch": float(epoch),
                "phase/routing_warmup": float(routing_warmup_active),
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
                "train/routing_loss": float(train_routing_loss),
                "train/weighted_routing_loss": float(train_weighted_routing_loss),
                "train/routing_gate_loss": float(train_routing_gate_loss),
                "train/weighted_routing_gate_loss": float(train_weighted_routing_gate_loss),
                "train/routing_top1_accuracy": float(train_routing_top1_accuracy),
                "train/routing_top5_accuracy": float(train_routing_top5_accuracy),
                "train/routing_row_top1_accuracy": float(
                    train_routing_row_top1_accuracy
                ),
                "train/routing_col_top1_accuracy": float(
                    train_routing_col_top1_accuracy
                ),
                "train/routing_target_mass": float(train_routing_target_mass),
                "train/routing_normalized_entropy": float(
                    train_routing_normalized_entropy
                ),
                "train/routing_gate_accuracy": float(train_routing_gate_accuracy),
                "train/routing_gate_active_fraction": float(
                    train_routing_gate_active_fraction
                ),
                "train/routing_gate_target_active_fraction": float(
                    train_routing_gate_target_active_fraction
                ),
                "train/matched_group_loss": float(train_matched_group_loss),
                "train/weighted_matched_group_loss": float(
                    train_weighted_matched_group_loss
                ),
                "train/matched_group_exact_accuracy": float(
                    train_matched_group_exact_accuracy
                ),
                "train/matched_group_gap_mean": float(train_matched_group_gap_mean),
                "train/matched_group_satisfaction": float(
                    train_matched_group_satisfaction
                ),
                "train/global_view_loss": float(train_global_view_loss),
                "train/weighted_global_view_loss": float(
                    train_weighted_global_view_loss
                ),
                "train/global_view_accuracy": float(train_global_view_accuracy),
                "train/joint_no_harm_loss": float(train_joint_no_harm_loss),
                "train/weighted_joint_no_harm_loss": float(
                    train_weighted_joint_no_harm_loss
                ),
                "train/joint_no_harm_margin": float(train_joint_no_harm_margin),
                "train/joint_causal_loss": float(train_joint_causal_loss),
                "train/weighted_joint_causal_loss": float(
                    train_weighted_joint_causal_loss
                ),
                "train/joint_causal_margin": float(train_joint_causal_margin),
                "train/joint_causal_active_records": float(
                    train_totals["joint_causal_active_records"]
                ),
                "train/global_anchor_loss": float(train_global_anchor_loss),
                "train/weighted_global_anchor_loss": float(
                    train_weighted_global_anchor_loss
                ),
                "train/local_anchor_loss": float(train_local_anchor_loss),
                "train/weighted_local_anchor_loss": float(
                    train_weighted_local_anchor_loss
                ),
                "train/total_grad_norm": float(train_total_grad_norm),
                "train/post_clip_grad_norm": float(train_post_clip_grad_norm),
                "train/clip_fraction": float(train_clip_fraction),
                "train/pre_clip_grad_norm_p50": float(
                    epoch_payload["train_pre_clip_grad_norm_p50"]
                ),
                "train/pre_clip_grad_norm_p95": float(
                    epoch_payload["train_pre_clip_grad_norm_p95"]
                ),
                "train/pre_clip_grad_norm_max": float(
                    epoch_payload["train_pre_clip_grad_norm_max"]
                ),
                "train/post_clip_grad_norm_p50": float(
                    epoch_payload["train_post_clip_grad_norm_p50"]
                ),
                "train/post_clip_grad_norm_p95": float(
                    epoch_payload["train_post_clip_grad_norm_p95"]
                ),
                "train/post_clip_grad_norm_max": float(
                    epoch_payload["train_post_clip_grad_norm_max"]
                ),
                "train/local_grad_norm": float(train_local_grad_norm),
                "train/global_grad_norm": float(train_global_grad_norm),
                "train/global_prompt_dropout_rate": float(train_global_dropout_rate),
                "adapter/local_gate": float(
                    getattr(adapter.local_adapter, "gate").detach().float().cpu().item()
                    if isinstance(adapter, HybridGlobalLocalAdapter)
                    and getattr(adapter.local_adapter, "gate", None) is not None
                    else 0.0
                ),
                "adapter/local_anchor_gate": float(
                    adapter.local_adapter.anchor_gate.detach().float().cpu().item()
                    if isinstance(adapter, HybridGlobalLocalAdapter)
                    and getattr(adapter.local_adapter, "anchor_gate", None) is not None
                    else 0.0
                ),
                "lr": optimizer_group_lr(optimizer, optimizer_lr_prefix, float(args.lr)),
                "local_lr": optimizer_group_lr(
                    optimizer,
                    optimizer_lr_prefix,
                    float(args.lr),
                ),
                "global_lr": (
                    optimizer_group_lr(optimizer, "global", 0.0)
                    if isinstance(adapter, HybridGlobalLocalAdapter)
                    else 0.0
                ),
                "global_trainable": float(
                    isinstance(adapter, HybridGlobalLocalAdapter) and not adapter.freeze_global
                ),
                f"{validation_scope}_val/macro_latent_gain": float(
                    val_macro_latent_gain
                ),
            }
            if not routing_warmup_active and not uses_screened_stage2b_training(args):
                wandb_payload["best_val/checkpoint_score"] = float(
                    max(best_val_score, val_score)
                )
            elif uses_screened_stage2b_training(args):
                wandb_payload["screening/checkpoint_score"] = float(val_score)
            if routing_warmup_audit is not None:
                wandb_payload["routing_warmup/passed"] = float(
                    bool(routing_warmup_audit["passed"])
                )
                wandb_payload.update(
                    {
                        f"routing_warmup/{name}": float(value)
                        for name, value in routing_warmup_audit["observed"].items()
                    }
                )
            wandb_payload.update(
                flatten_numeric_metrics(f"{validation_scope}_val", val_metrics)
                if bool(args.wandb_detailed_metrics)
                else compact_accuracy_metrics(f"{validation_scope}_val", val_metrics)
            )
            if (
                not routing_warmup_active
                and not uses_screened_stage2b_training(args)
                and val_score > best_val_score
            ):
                best_val_score = val_score
                best_epoch = epoch
                run_on_rank_zero_and_broadcast(
                    lambda: save_adapter_checkpoint(
                        run_dir / "adapter_best.pt",
                        adapter=adapter,
                        args=args,
                        latent_shape=latent_shape,
                        llm_hidden_size=llm_hidden_size,
                        latent_contract=latent_contract or {},
                        metrics=epoch_payload,
                    ),
                    f"epoch {epoch} best-checkpoint write",
                )
            diagnostic_suffix = ""
            if (
                not routing_warmup_active
                and
                bool(args.diagnostics_enabled)
                and int(args.diagnostics_every_epochs) > 0
                and epoch % int(args.diagnostics_every_epochs) == 0
            ):
                diagnostic_aggregate = dict(
                    run_on_rank_zero_and_broadcast(
                        lambda: run_embedded_diagnostics(
                            stage=f"epoch_{epoch:04d}",
                            llm=llm,
                            adapter=adapter,
                            tokenizer=tokenizer,
                            dataset=val_dataset,
                            device=device,
                            args=args,
                            run_dir=run_dir,
                        )["aggregate"],
                        f"epoch {epoch} diagnostics",
                    )
                )
                epoch_payload["diagnostics"] = diagnostic_aggregate
                history[f"epoch_{epoch:04d}"] = epoch_payload
                run_on_rank_zero_and_broadcast(
                    lambda: atomic_dump_json(run_dir / "metrics_latest.json", history),
                    f"epoch {epoch} diagnostic metrics write",
                )
                wandb_payload.update(
                    flatten_numeric_metrics("diagnostics", diagnostic_aggregate)
                    if bool(args.wandb_detailed_metrics)
                    else compact_diagnostic_metrics(diagnostic_aggregate)
                )
                if is_direct_alignment_architecture(str(args.adapter_architecture)):
                    tensor_l2_by_layer = diagnostic_aggregate.get(
                        "question_last_relative_l2_by_layer",
                        {},
                    )
                    tensor_l2_by_layer = (
                        tensor_l2_by_layer if isinstance(tensor_l2_by_layer, Mapping) else {}
                    )
                    final_layer_key = (
                        max(tensor_l2_by_layer, key=lambda value: int(value))
                        if tensor_l2_by_layer
                        else None
                    )
                    diagnostic_suffix = (
                        " diag_q_last_l2="
                        f"{float(diagnostic_aggregate.get('same_latent_different_question_question_last_relative_l2_mean', 0.0)):.4f}"
                        " diag_tensor_l2@2="
                        f"{float(tensor_l2_by_layer.get('2', 0.0)):.4f}"
                        " diag_tensor_l2@last="
                        f"{float(tensor_l2_by_layer.get(final_layer_key, 0.0)):.4f}"
                        " diag_margin_gain="
                        f"{float(diagnostic_aggregate['answer_margin_correct_minus_shuffled']):.4f}"
                    )
                else:
                    diagnostic_suffix = (
                        f" diag_local_l2={float(diagnostic_aggregate['local_prompt_relative_l2_mean']):.4f}"
                        f" diag_margin_gain={float(diagnostic_aggregate['answer_margin_correct_minus_shuffled']):.4f}"
                    )
            distributed_barrier()
            log_wandb_on_rank_zero(
                wandb_logger,
                wandb_payload,
                step=global_step,
                stage=f"epoch {epoch} W&B log",
            )
            if is_main_process():
                print(
                    f"epoch={epoch:03d}/{int(args.epochs):03d} "
                    f"phase={training_phase} "
                    f"loss={train_loss:.4f} train_acc={train_choice_accuracy:.4f} "
                    f"{validation_scope}_val={val_accuracy:.4f} "
                    f"shuffled={float(val_metrics.get('shuffled', {}).get('accuracy', 0.0)):.4f} "
                    f"macro_gain={val_macro_latent_gain:.4f} best_epoch={best_epoch}"
                    f"{diagnostic_suffix}"
                )
            if routing_warmup_audit is not None and not bool(
                routing_warmup_audit["passed"]
            ):
                failed = ",".join(str(value) for value in routing_warmup_audit["failed"])
                raise RuntimeError(
                    "Grounded routing warmup failed validation thresholds "
                    f"({failed}). The reader checkpoint and audit were saved; joint answer "
                    "training was not started."
                )

        if uses_screened_stage2b_training(args):
            if not joint_candidates or parent_validation_metrics is None:
                raise RuntimeError("Stage-2B training produced no screenable step checkpoints.")
            top_candidates = stage2b_full_validation_candidates(
                joint_candidates,
                full_local_reader_training=bool(args.full_local_reader_training),
                top_k=int(args.checkpoint_full_eval_top_k),
            )
            if bool(args.full_local_reader_training) and len(top_candidates) != len(
                joint_candidates
            ):
                raise RuntimeError(
                    "Full local-reader selection must evaluate every saved candidate."
                )
            full_candidate_results: list[dict[str, Any]] = []
            for candidate in top_candidates:
                if bool(candidate.get("is_parent", False)):
                    full_candidate_results.append(dict(candidate))
                    continue
                candidate_path = Path(str(candidate["path"]))
                candidate_checkpoint = torch.load(
                    candidate_path,
                    map_location="cpu",
                    weights_only=True,
                )
                candidate_state = validate_adapter_checkpoint_payload(
                    candidate_checkpoint,
                    expected_latent_shape=latent_shape,
                    expected_llm_hidden_size=llm_hidden_size,
                    expected_architecture=str(args.adapter_architecture),
                    expected_latent_contract=latent_contract or {},
                    expected_latent_channel_policy=str(args.latent_channel_policy),
                )
                adapter.load_state_dict(candidate_state, strict=True)
                del candidate_state, candidate_checkpoint
                full_metrics = evaluate_choice_accuracy(
                    llm=llm,
                    adapter=adapter,
                    tokenizer=tokenizer,
                    dataset=val_dataset,
                    device=device,
                    args=args,
                    baseline_modes=baseline_modes,
                )
                full_selection = screened_stage2b_checkpoint_metrics(
                    full_metrics,
                    parent_validation_metrics,
                    args,
                )
                routing_audit = grounded_routing_warmup_audit(full_metrics, args)
                full_selection["routing_gate_audit"] = routing_audit
                full_selection["acceptance"]["routing_gate_audit"] = bool(
                    routing_audit["passed"]
                )
                full_selection["accepted"] = all(
                    bool(value) for value in full_selection["acceptance"].values()
                )
                full_selection["eligible_for_promotion"] = True
                full_candidate_results.append(
                    {
                        **candidate,
                        "full_val": full_metrics,
                        "full_selection": full_selection,
                    }
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            # A child must clear every admission guardrail before it can replace the
            # current best model. Otherwise retain the live, fully evaluated parent.
            (
                selected_candidate,
                accepted_candidates,
                selected_promoted,
            ) = select_admitted_stage2b_candidate(
                full_candidate_results,
            )
            selected_checkpoint = torch.load(
                Path(str(selected_candidate["path"])),
                map_location="cpu",
                weights_only=True,
            )
            selected_state = validate_adapter_checkpoint_payload(
                selected_checkpoint,
                expected_latent_shape=latent_shape,
                expected_llm_hidden_size=llm_hidden_size,
                expected_architecture=str(args.adapter_architecture),
                expected_latent_contract=latent_contract or {},
                expected_latent_channel_policy=str(args.latent_channel_policy),
            )
            adapter.load_state_dict(selected_state, strict=True)
            del selected_state, selected_checkpoint
            best_epoch = int(selected_candidate["epoch"])
            best_val_score = float(
                selected_candidate["full_selection"][
                    "point_value_min_causal_gain"
                    if bool(args.point_reader_training)
                    or bool(args.full_local_reader_training)
                    else "worst_protected_task_delta"
                ]
            )
            selected_accepted = bool(selected_promoted)
            joint_selected_accepted = selected_accepted
            selected_checkpoint_path = run_dir / (
                "adapter_best.pt"
                if selected_accepted
                else (
                    "adapter_parent_retained.pt"
                    if bool(selected_candidate.get("is_parent", False))
                    else "adapter_best_rejected.pt"
                )
            )
            joint_selection_payload = {
                "screened_candidate_count": len(joint_candidates),
                "full_evaluated_candidate_count": len(full_candidate_results),
                "accepted_candidate_count": len(accepted_candidates),
                "selected_global_step": int(selected_candidate["global_step"]),
                "selected_is_parent": bool(selected_candidate.get("is_parent", False)),
                "selected_path": str(selected_candidate["path"]),
                "selected_accepted": selected_accepted,
                "selected_output_path": str(selected_checkpoint_path),
                "selected_full_selection": selected_candidate["full_selection"],
                "candidates": full_candidate_results,
            }
            history["joint_full_candidate_selection"] = joint_selection_payload

            def write_joint_selection_outputs() -> None:
                atomic_dump_json(run_dir / "metrics_latest.json", history)
                save_adapter_checkpoint(
                    selected_checkpoint_path,
                    adapter=adapter,
                    args=args,
                    latent_shape=latent_shape,
                    llm_hidden_size=llm_hidden_size,
                    latent_contract=latent_contract or {},
                    metrics={
                        "epoch": best_epoch,
                        "global_step": int(selected_candidate["global_step"]),
                        "val": selected_candidate["full_val"],
                        "joint_selection": selected_candidate["full_selection"],
                    },
                )

            run_on_rank_zero_and_broadcast(
                write_joint_selection_outputs,
                "joint full-validation checkpoint selection write",
            )
            log_wandb_on_rank_zero(
                wandb_logger,
                {
                    "joint_selection/selected_global_step": float(
                        selected_candidate["global_step"]
                    ),
                    "joint_selection/selected_accepted": float(selected_accepted),
                    "joint_selection/accepted_candidate_count": float(
                        len(accepted_candidates)
                    ),
                    **flatten_numeric_metrics(
                        "joint_selection/full",
                        selected_candidate["full_selection"],
                    ),
                },
                step=global_step,
                stage="joint full-validation selection W&B log",
            )
            if is_main_process():
                print(
                    "stage2b_full_selection "
                    f"step={int(selected_candidate['global_step'])} "
                    f"accepted={selected_accepted} "
                    f"checkpoint={selected_checkpoint_path.name} "
                    f"selection_score={best_val_score:.6f}"
                )

        distributed_barrier()
        best_checkpoint_path = selected_checkpoint_path
        if not best_checkpoint_path.exists():
            raise FileNotFoundError(
                f"Training completed without producing {best_checkpoint_path.name}."
            )
        best_checkpoint = torch.load(
            best_checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        validate_adapter_checkpoint_payload(
            best_checkpoint,
            expected_latent_shape=latent_shape,
            expected_llm_hidden_size=llm_hidden_size,
            expected_architecture=str(args.adapter_architecture),
            expected_latent_contract=latent_contract or {},
            expected_latent_channel_policy=str(args.latent_channel_policy),
        )
        # Reconstruct from metadata instead of loading back into the existing
        # object. This proves adapter_best.pt is independently usable by later
        # evaluation/inference code and catches omitted constructor fields.
        reloaded_adapter = adapter_from_checkpoint(
            best_checkpoint,
            latent_shape=latent_shape,
            llm_hidden_size=llm_hidden_size,
        )
        loss = None
        loss_parts = None
        batch = None
        progress = None
        del optimizer, lr_scheduler, best_checkpoint
        global_anchor_reference = None
        local_anchor_reference = None
        global_adapter = None
        local_adapter = None
        local_groups = None
        global_groups = None
        del adapter
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        adapter = reloaded_adapter.to(device)
        if isinstance(adapter, HybridGlobalLocalAdapter):
            adapter.mask_inactive_local_tokens = bool(args.mask_inactive_local_tokens)
        del reloaded_adapter
        test_metrics: dict[str, Any] = {}
        test_evaluated, test_skip_reason = resolve_test_evaluation_policy(
            requested=bool(args.evaluate_test),
            joint_ab_training=bool(args.joint_ab_training),
            joint_selected_accepted=joint_selected_accepted,
            point_reader_training=bool(args.point_reader_training),
            full_local_reader_training=bool(args.full_local_reader_training),
        )
        if test_evaluated:
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
            run_on_rank_zero_and_broadcast(
                lambda: atomic_dump_json(run_dir / "test_metrics.json", test_metrics),
                "test metrics write",
            )
            test_payload = (
                flatten_numeric_metrics("test", test_metrics)
                if bool(args.wandb_detailed_metrics)
                else compact_accuracy_metrics("test", test_metrics)
            )
            log_wandb_on_rank_zero(
                wandb_logger,
                test_payload,
                step=global_step + 1,
                stage="test evaluation W&B log",
            )
        else:
            if is_main_process():
                print(f"test evaluation skipped: {test_skip_reason}")

        def write_final_outputs() -> None:
            raw_joint_result = history.get("joint_full_candidate_selection")
            joint_result = (
                raw_joint_result if isinstance(raw_joint_result, Mapping) else {}
            )
            summary["result"] = {
                "best_epoch": int(best_epoch),
                "best_val_score": float(best_val_score),
                "checkpoint_metric": str(args.checkpoint_metric),
                "selected_global_step": joint_result.get("selected_global_step"),
                "selected_is_parent": joint_result.get("selected_is_parent"),
                "joint_selected_accepted": joint_result.get("selected_accepted"),
                "joint_screened_candidate_count": joint_result.get(
                    "screened_candidate_count"
                ),
                "joint_full_evaluated_candidate_count": joint_result.get(
                    "full_evaluated_candidate_count"
                ),
                "joint_accepted_candidate_count": joint_result.get(
                    "accepted_candidate_count"
                ),
                "joint_selected_full_selection": joint_result.get(
                    "selected_full_selection"
                ),
                "selected_checkpoint": selected_checkpoint_path.name,
                "promotion_checkpoint": (
                    selected_checkpoint_path.name
                    if not uses_screened_stage2b_training(args)
                    or bool(joint_result.get("selected_accepted", False))
                    else None
                ),
                "rejected_diagnostic_checkpoint": (
                    selected_checkpoint_path.name
                    if uses_screened_stage2b_training(args)
                    and not bool(joint_result.get("selected_accepted", False))
                    and not bool(joint_result.get("selected_is_parent", False))
                    else None
                ),
                "retained_parent_checkpoint": (
                    selected_checkpoint_path.name
                    if bool(joint_result.get("selected_is_parent", False))
                    else None
                ),
                "test_requested": bool(args.evaluate_test),
                "test_evaluated": bool(test_evaluated),
                "test_skip_reason": test_skip_reason,
                "test_correct_accuracy": (
                    float(test_metrics.get("correct", {}).get("accuracy", 0.0))
                    if test_evaluated
                    else None
                ),
                "test_shuffled_accuracy": (
                    float(test_metrics.get("shuffled", {}).get("accuracy", 0.0))
                    if test_evaluated
                    else None
                ),
                "test_correct_by_task": dict(
                    test_metrics.get("correct", {}).get("by_task", {})
                ),
            }
            atomic_dump_json(run_dir / "run_summary.json", summary)
            if bool(args.wandb_log_model):
                log_adapter_artifact(
                    wandb_logger,
                    selected_checkpoint_path,
                    f"{args.run_name}-best",
                )
                log_adapter_artifact(
                    wandb_logger,
                    run_dir / "adapter_last.pt",
                    f"{args.run_name}-last",
                )

        run_on_rank_zero_and_broadcast(write_final_outputs, "final run outputs write")
        if is_main_process():
            print(f"run_dir: {run_dir}")
            if test_evaluated:
                print_evaluation_summary("test", test_metrics, run_dir / "test_metrics.json")
    finally:
        wandb_logger.finish()
    distributed_barrier()

    def finish_lifecycle() -> dict[str, Any]:
        if lifecycle is None:
            raise RuntimeError("Rank 0 lost its run lifecycle before completion.")
        return lifecycle.finish("completed")

    timing = run_on_rank_zero_and_broadcast(finish_lifecycle, "run lifecycle completion")
    if is_main_process():
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
