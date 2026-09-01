from __future__ import annotations

"""Run controlled Stage-1 ablations through the production trainers.

The ablation has two phases:

1. ``direct`` trains the normal direct-QA spatial adapter from either the
   learned Stage-1 reference state or a deterministic random isomorphic state.
2. ``dense`` trains the normal dense cross-attention model from the resulting
   direct-QA checkpoint.

``full_stage1_reference`` is the matched positive control. ``adapter_only``
preserves all cached latent channels while resetting the alignment adapter.
``no_learned_stage1`` additionally exposes only the exact standardized value
channel and zeros the learned channels at load time without modifying cache
files. Adapter architecture, Qwen, datasets, losses, and downstream training
budgets remain unchanged. The source checkpoint stays untouched and continues
to satisfy the immutable latent provenance checks.
"""

import argparse
import copy
import hashlib
import json
import os
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for root in (PROJECT_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from scripts.train_tensor_patch_text_alignment import TensorPatchAlignmentAdapter  # noqa: E402
from tensor_compression.downstream.patch_qa_contract import (  # noqa: E402
    sha256_file,
    validate_stage1_alignment_checkpoint_payload,
)
from tensor_compression.utils.pipeline_config import (  # noqa: E402
    load_yaml_mapping,
    resolve_path_string,
)


IMPLEMENTATION_DERIVED_FROM_COMMIT = "8e7b6a3f94e818b574db7c99aabb3570d83ee9f2"
ABLATION_CONTRACT_VERSION = 3
ABLATION_MODE = "random_spatial_adapter_preserve_stage1_latent_encoder"
ABLATION_CONDITIONS: dict[str, dict[str, Any]] = {
    "full_stage1_reference": {
        "mode": "matched_full_stage1_reference",
        "condition_role": "matched_reference",
        "adapter_state_action": "preserve",
        "factor_removed": "none",
        "latent_channel_policy": "all",
        "claim_scope": (
            "Matched current-code reference preserving the learned Stage-1 adapter and all "
            "cached latent channels."
        ),
    },
    "adapter_only": {
        "mode": ABLATION_MODE,
        "condition_role": "ablation",
        "adapter_state_action": "randomize",
        "factor_removed": "stage1_spatial_adapter_alignment_initialization",
        "latent_channel_policy": "all",
        "claim_scope": (
            "Tests only the learned Stage-1 spatial-adapter initialization; cached latent "
            "channels and their encoder lineage remain active."
        ),
    },
    "no_learned_stage1": {
        "mode": "random_spatial_adapter_exact_value_channel_only",
        "condition_role": "ablation",
        "adapter_state_action": "randomize",
        "factor_removed": "all_learned_stage1_adapter_and_encoder_features",
        "latent_channel_policy": "value_only",
        "claim_scope": (
            "Removes learned Stage-1 adapter weights and learned latent channels while preserving "
            "the exact standardized value channel, tensor shape, and downstream training recipe."
        ),
    },
}
DISABLED_PATH_VALUES = {"", "none", "null", "random"}
DIRECT_OVERRIDE_KEYS = frozenset(
    {
        "run_name",
        "stage2_warm_start_checkpoint",
        "stage2b_resume_checkpoint",
        "joint_ab_training",
        "point_reader_training",
        "full_local_reader_training",
        "evaluate_test",
    }
)
DENSE_OVERRIDE_KEYS = frozenset({"run_name", "max_updates", "evaluate_test"})
DIRECT_FORWARDED_FLAGS = frozenset({"--console-progress", "--no-console-progress"})
DENSE_FORWARDED_FLAGS = frozenset(
    {"--evaluate-test", "--no-evaluate-test", "--console-progress", "--no-console-progress"}
)
FORMAL_EXPERIMENT_FILES = (
    Path("scripts/train_tensor_qwen_stage1_ablation.py"),
    Path("scripts/compare_tensor_qwen_stage1_ablation.py"),
    Path("configs/field_to_llm_stage1_reference.yaml"),
    Path("configs/field_to_llm_stage1_adapter_ablation.yaml"),
    Path("configs/field_to_llm_no_learned_stage1_ablation.yaml"),
    Path("configs/field_to_llm_direct_qa.yaml"),
    Path("configs/field_to_llm_cross_attention.yaml"),
)
LOCKED_VALIDATION_ARTIFACT_FIELDS = (
    "run_summary_sha256",
    "data_audit_sha256",
    "best_checkpoint_sha256",
    "last_checkpoint_sha256",
    "direct_qa_checkpoint_sha256",
    "direct_run_summary_sha256",
    "direct_run_timing_sha256",
    "direct_data_audit_sha256",
    "direct_qa_metadata_audit_sha256",
)
DENSE_CHECKPOINT_TYPE = "tensor_qwen_dense_cross_attention"
DENSE_CHECKPOINT_VERSION = 1
DENSE_LINEAGE_FIELDS = (
    "condition",
    "source_stage1_sha256",
    "source_adapter_state_sha256",
    "initial_adapter_state_sha256",
    "source_encoder_state_sha256",
    "latent_channel_policy",
    "evaluation_contract_sha256",
    "direct_qa_checkpoint_sha256",
    "direct_run_audit_sha256",
)


def _condition_spec(condition: Any) -> tuple[str, dict[str, Any]]:
    normalized = str(condition or "adapter_only").strip().lower()
    if normalized not in ABLATION_CONDITIONS:
        raise ValueError(
            f"Unsupported ablation.condition={condition!r}; expected one of "
            f"{sorted(ABLATION_CONDITIONS)}."
        )
    return normalized, dict(ABLATION_CONDITIONS[normalized])


def _validated_evaluation_contract(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("Stage-1 comparison requires a mapping evaluation_contract.")
    contract = copy.deepcopy(dict(value))
    if str(contract.get("experiment_id", "")) != "stage1_necessity_matched_v1":
        raise ValueError("Unexpected or missing Stage-1 comparison experiment_id.")
    conditions = [str(item) for item in contract.get("conditions", [])]
    if conditions != list(ABLATION_CONDITIONS):
        raise ValueError(
            "evaluation_contract.conditions must declare the matched reference followed by "
            f"both ablations: expected={list(ABLATION_CONDITIONS)}, observed={conditions}."
        )
    if contract.get("reference_condition") != "full_stage1_reference":
        raise ValueError("evaluation_contract must name full_stage1_reference as its control.")
    if contract.get("test_access") != "disabled_by_default":
        raise ValueError("Stage-1 comparison must seal the test split by default.")
    return contract


def _validate_condition_match(configured: Any, contract: Mapping[str, Any]) -> str:
    configured_condition, _spec = _condition_spec(configured)
    checkpoint_condition = str(contract.get("condition", ""))
    if configured_condition != checkpoint_condition:
        raise ValueError(
            "Dense config condition does not match its Direct checkpoint contract: "
            f"configured={configured_condition!r}, checkpoint={checkpoint_condition!r}."
        )
    return configured_condition


def _mapping_for_digest(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Missing or invalid {label} mapping.")
    return dict(value)


def _load_json_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read {label} {path}: {error}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return dict(payload)


def _require_locked_file(path: Path, expected_sha256: Any, label: str) -> None:
    expected = str(expected_sha256 or "").lower()
    if not path.is_file():
        raise ValueError(f"Locked validation {label} is missing: {path}.")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(
            f"Locked validation {label} changed before test access: "
            f"expected={expected}, observed={observed}, path={path}."
        )


def _validate_test_access_request(
    *,
    wants_test: bool,
    protocol_lock: str | None,
    resume_checkpoint: str | None,
) -> None:
    if wants_test != bool(protocol_lock):
        raise ValueError(
            "Dense test evaluation requires --evaluate-test together with --protocol-lock; "
            "neither is allowed alone."
        )
    if wants_test and not resume_checkpoint:
        raise ValueError(
            "Sealed test evaluation may only resume the exact completed validation run; "
            "a protocol lock cannot authorize a fresh dense training run."
        )


def _validated_dense_checkpoint_payload(
    payload: Any,
    *,
    label: str,
    require_optimizer_state: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must contain a mapping payload.")
    if str(payload.get("checkpoint_type", "")) != DENSE_CHECKPOINT_TYPE or int(
        payload.get("checkpoint_version", 0)
    ) != DENSE_CHECKPOINT_VERSION:
        raise ValueError(f"{label} has the wrong dense checkpoint type or version.")
    architecture = _mapping_for_digest(payload.get("architecture"), f"{label} architecture")
    ablation = _mapping_for_digest(
        architecture.get("stage1_ablation"), f"{label} Stage-1 contract"
    )
    trainable_state = payload.get("trainable_state_dict")
    if not isinstance(trainable_state, Mapping) or not trainable_state:
        raise ValueError(f"{label} has no trainable_state_dict.")
    progress = _mapping_for_digest(payload.get("progress"), f"{label} progress")
    if require_optimizer_state:
        if not isinstance(payload.get("optimizer_state_dict"), Mapping) or not isinstance(
            payload.get("scheduler_state_dict"), Mapping
        ):
            raise ValueError(f"{label} lacks optimizer or scheduler state.")
    return ablation, progress


def _validate_dense_lineage(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    label: str,
) -> None:
    for field in DENSE_LINEAGE_FIELDS:
        if observed.get(field) != expected.get(field):
            raise ValueError(
                f"{label} has a different ablation lineage: field={field}."
            )


def _direct_run_audit(
    checkpoint_path: Path,
    checkpoint: Mapping[str, Any],
    expected_policy: str,
) -> dict[str, Any]:
    run_dir = checkpoint_path.parent
    paths = {
        "run_summary": run_dir / "run_summary.json",
        "run_timing": run_dir / "run_timing.json",
        "data_audit": run_dir / "data_audit.json",
        "qa_metadata_audit": run_dir / "qa_metadata_audit.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise ValueError(f"Direct comparison run is missing completion artifacts: {missing}.")
    payloads = {name: _load_json_mapping(path, name) for name, path in paths.items()}
    summary = payloads["run_summary"]
    timing = payloads["run_timing"]
    embedded_timing = _mapping_for_digest(summary.get("timing"), "Direct summary timing")
    result = _mapping_for_digest(summary.get("result"), "Direct summary result")
    if timing.get("status") != "completed" or embedded_timing.get("status") != "completed":
        raise ValueError("Direct comparison run did not complete successfully.")
    if result.get("test_evaluated") is not False or result.get("test_requested") is not False:
        raise ValueError("Direct comparison run must complete without test access.")
    selected_checkpoint = str(result.get("selected_checkpoint", "")).strip()
    if selected_checkpoint != "adapter_best.pt" or checkpoint_path.name != selected_checkpoint:
        raise ValueError(
            "Dense comparison must use the completed Direct run's selected adapter_best.pt; "
            f"summary selected={selected_checkpoint!r}, supplied={checkpoint_path.name!r}."
        )
    if str(summary.get("latent_channel_policy", "all")) != str(expected_policy):
        raise ValueError("Direct run summary latent policy differs from its checkpoint contract.")
    checkpoint_args = _mapping_for_digest(checkpoint.get("args"), "Direct checkpoint args")
    checkpoint_metrics = _mapping_for_digest(
        checkpoint.get("metrics"), "Direct checkpoint metrics"
    )
    data_audit = payloads["data_audit"]
    for split in ("train", "val", "test"):
        split_audit = _mapping_for_digest(data_audit.get(split), f"Direct {split} data audit")
        digest = str(split_audit.get("record_contract_sha256", "")).lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"Direct {split} data audit lacks a valid record fingerprint.")
    state = _checkpoint_adapter_state({"adapter_state_dict": checkpoint.get("adapter_state_dict")})
    state_schema = {
        str(name): {
            "shape": [int(value) for value in tensor.shape],
            "dtype": str(tensor.dtype),
        }
        for name, tensor in sorted(state.items())
    }
    invariant_arg_names = (
        "model_name_or_path",
        "adapter_architecture",
        "seed",
        "epochs",
        "batch_size",
        "gradient_accumulation_steps",
        "max_train_records",
        "max_val_records",
        "max_test_records",
        "record_subset_mode",
        "shuffle_seed",
        "lr",
        "lr_scheduler",
        "warmup_ratio",
        "min_lr_ratio",
        "weight_decay",
    )
    return {
        "artifacts": {
            **{f"{name}_sha256": sha256_file(path) for name, path in paths.items()},
            "direct_checkpoint_sha256": sha256_file(checkpoint_path),
            "selected_checkpoint": selected_checkpoint,
            "selected_checkpoint_epoch": checkpoint_metrics.get("epoch"),
        },
        "invariants": {
            "summary": {
                "distributed": summary.get("distributed"),
                "latent_shape_chw": summary.get("latent_shape_chw"),
                "train_records": summary.get("train_records"),
                "val_records": summary.get("val_records"),
                "test_records": summary.get("test_records"),
                "total_optimizer_updates": summary.get("total_optimizer_updates"),
                "adapter_parameters_total": summary.get("adapter_parameters_total"),
                "trainable_adapter_parameters": summary.get("trainable_adapter_parameters"),
            },
            "checkpoint_args": {
                name: checkpoint_args.get(name) for name in invariant_arg_names
            },
            "adapter_state_schema": state_schema,
            "data_audit": data_audit,
            "latent_contract": payloads["qa_metadata_audit"].get("latent_contract"),
        },
    }


def _validated_protocol_lock(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read test protocol lock {path}: {error}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("Test protocol lock must contain a JSON object.")
    lock = dict(payload)
    if lock.get("experiment_id") != "stage1_necessity_matched_v1" or lock.get(
        "status"
    ) != "locked":
        raise ValueError("Test protocol lock has the wrong experiment_id or is not locked.")
    if [str(item) for item in lock.get("conditions", [])] != list(ABLATION_CONDITIONS):
        raise ValueError("Test protocol lock does not cover all three matched conditions.")
    comparison_sha = str(lock.get("validation_comparison_sha256", "")).lower()
    if len(comparison_sha) != 64 or any(character not in "0123456789abcdef" for character in comparison_sha):
        raise ValueError(
            "Test protocol lock requires the SHA-256 of the completed validation comparison."
        )
    comparison_value = str(lock.get("validation_comparison", "")).strip()
    if not comparison_value:
        raise ValueError("Test protocol lock must point to the validation comparison JSON.")
    comparison_path = Path(comparison_value).expanduser()
    if not comparison_path.is_absolute():
        comparison_path = path.parent / comparison_path
    comparison_path = comparison_path.resolve()
    if not comparison_path.is_file():
        raise ValueError(f"Validation comparison file does not exist: {comparison_path}")
    actual_comparison_sha = sha256_file(comparison_path)
    if comparison_sha != actual_comparison_sha:
        raise ValueError(
            "Test protocol lock validation comparison SHA mismatch: "
            f"declared={comparison_sha}, actual={actual_comparison_sha}."
        )
    try:
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read validation comparison {comparison_path}: {error}") from error
    if not isinstance(comparison, Mapping):
        raise ValueError("Validation comparison must contain a JSON object.")
    if (
        comparison.get("format") != "stage1_necessity_comparison_v1"
        or comparison.get("valid_matched_comparison") is not True
        or comparison.get("metric_split") != "validation"
        or [str(item) for item in comparison.get("conditions", [])]
        != list(ABLATION_CONDITIONS)
    ):
        raise ValueError("Protocol lock points to an invalid or non-validation comparison.")
    comparison_artifacts = _mapping_for_digest(
        comparison.get("artifacts"), "validation comparison artifacts"
    )
    validation_artifacts: dict[str, dict[str, str]] = {}
    for condition in ABLATION_CONDITIONS:
        condition_artifacts = _mapping_for_digest(
            comparison_artifacts.get(condition),
            f"validation artifacts for {condition}",
        )
        normalized: dict[str, str] = {}
        for field in LOCKED_VALIDATION_ARTIFACT_FIELDS:
            value = str(condition_artifacts.get(field, "")).lower()
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(
                    f"Validation comparison has invalid {condition}.{field}; "
                    "the test lock must bind the completed model and data lineage."
                )
            normalized[field] = value
        validation_artifacts[condition] = normalized
    decision_note = str(lock.get("decision_note", "")).strip()
    if not decision_note or decision_note.startswith("REPLACE_"):
        raise ValueError("Test protocol lock requires a pre-test decision_note.")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "payload_sha256": _json_digest(lock),
        "validation_comparison": str(comparison_path),
        "validation_comparison_sha256": comparison_sha,
        "validation_comparison_payload_sha256": _json_digest(dict(comparison)),
        "source_stage1_sha256": str(comparison.get("source_stage1_sha256", "")),
        "evaluation_contract_sha256": _json_digest(
            _mapping_for_digest(comparison.get("evaluation_contract"), "validation evaluation contract")
        ),
        "validation_artifacts": validation_artifacts,
        "decision_note": decision_note,
    }


def _json_digest(value: Any) -> str:
    serialized = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _git_bytes(*arguments: str) -> bytes | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=PROJECT_ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, ValueError):
        return None
    return completed.stdout if completed.returncode == 0 else None


def _source_tree_provenance() -> dict[str, Any]:
    head_bytes = _git_bytes("rev-parse", "HEAD")
    branch_bytes = _git_bytes("branch", "--show-current")
    diff_bytes = _git_bytes("diff", "--binary", "--no-ext-diff", "HEAD", "--", ".")
    status_bytes = _git_bytes("status", "--porcelain=v1", "--untracked-files=normal")
    available = all(value is not None for value in (head_bytes, diff_bytes, status_bytes))
    if not available:
        return {"git_available": False}
    status_text = status_bytes.decode("utf-8", errors="replace")
    return {
        "git_available": True,
        "commit": head_bytes.decode("ascii", errors="replace").strip(),
        "branch": (
            branch_bytes.decode("utf-8", errors="replace").strip()
            if branch_bytes is not None
            else ""
        ),
        "tracked_dirty": bool(diff_bytes),
        "tracked_diff_sha256": hashlib.sha256(diff_bytes).hexdigest(),
        "status_sha256": hashlib.sha256(status_bytes).hexdigest(),
        "untracked_paths": [
            line[3:]
            for line in status_text.splitlines()
            if line.startswith("?? ")
        ],
    }


def _traceability_contract(
    *,
    orchestration_config: Path,
    base_config: Path,
    resolved_config: Mapping[str, Any],
) -> dict[str, Any]:
    experiment_files: dict[str, dict[str, Any]] = {}
    for relative_path in FORMAL_EXPERIMENT_FILES:
        absolute_path = PROJECT_ROOT / relative_path
        experiment_files[relative_path.as_posix()] = {
            "sha256": sha256_file(absolute_path) if absolute_path.is_file() else None,
            "tracked_by_git": _git_bytes(
                "ls-files", "--error-unmatch", "--", relative_path.as_posix()
            )
            is not None,
        }
    return {
        "implementation_derived_from_commit": IMPLEMENTATION_DERIVED_FROM_COMMIT,
        "ablation_script": str(Path(__file__).resolve()),
        "ablation_script_sha256": sha256_file(Path(__file__).resolve()),
        "orchestration_config": str(orchestration_config),
        "orchestration_config_sha256": sha256_file(orchestration_config),
        "orchestration_mapping_sha256": _json_digest(dict(resolved_config)),
        "base_config": str(base_config),
        "base_config_sha256": sha256_file(base_config),
        "runtime_versions": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda": str(torch.version.cuda or "none"),
        },
        "formal_experiment_files": experiment_files,
        "source_tree": _source_tree_provenance(),
    }


def _enforce_traceability(
    traceability: Mapping[str, Any],
    settings: Mapping[str, Any] | None,
) -> None:
    options = dict(settings or {})
    source_tree = traceability.get("source_tree")
    source_tree = dict(source_tree) if isinstance(source_tree, Mapping) else {}
    if bool(options.get("require_git", True)) and not bool(source_tree.get("git_available")):
        raise RuntimeError("Formal ablation requires an accessible Git worktree for provenance.")
    if bool(options.get("require_clean_tracked_tree", True)) and bool(
        source_tree.get("tracked_dirty")
    ):
        raise RuntimeError(
            "Formal ablation requires a clean tracked Git tree. Commit the experiment code/config "
            "before launching; untracked data and checkpoints are recorded but do not fail this guard."
        )
    experiment_files = traceability.get("formal_experiment_files")
    if not isinstance(experiment_files, Mapping) or not experiment_files:
        raise RuntimeError("Formal ablation is missing its experiment-file provenance.")
    untracked_or_missing = [
        name
        for name, metadata in experiment_files.items()
        if not isinstance(metadata, Mapping)
        or not metadata.get("sha256")
        or metadata.get("tracked_by_git") is not True
    ]
    if untracked_or_missing:
        raise RuntimeError(
            "Formal ablation requires every experiment script/config to exist in the current "
            f"Git commit; missing or untracked={untracked_or_missing}."
        )


def _validate_recorded_traceability(
    recorded: Mapping[str, Any],
    *,
    orchestration_config: Path,
) -> None:
    if recorded.get("ablation_script_sha256") != sha256_file(Path(__file__).resolve()):
        raise ValueError("Ablation implementation changed between the Direct and dense phases.")
    if recorded.get("orchestration_config_sha256") != sha256_file(orchestration_config):
        raise ValueError("Ablation orchestration config changed between the Direct and dense phases.")
    recorded_files = recorded.get("formal_experiment_files")
    if not isinstance(recorded_files, Mapping):
        raise ValueError("Direct ablation checkpoint lacks formal experiment-file provenance.")
    for relative_path in FORMAL_EXPERIMENT_FILES:
        key = relative_path.as_posix()
        metadata = recorded_files.get(key)
        absolute_path = PROJECT_ROOT / relative_path
        if (
            not isinstance(metadata, Mapping)
            or not absolute_path.is_file()
            or metadata.get("sha256") != sha256_file(absolute_path)
            or metadata.get("tracked_by_git") is not True
        ):
            raise ValueError(
                f"Formal experiment file changed or became untracked after Direct training: {key}."
            )
    previous_tree = recorded.get("source_tree")
    current_tree = _source_tree_provenance()
    if not isinstance(previous_tree, Mapping):
        raise ValueError("Direct ablation checkpoint is missing source-tree provenance.")
    for key in ("git_available", "commit", "tracked_dirty", "tracked_diff_sha256"):
        if previous_tree.get(key) != current_tree.get(key):
            raise ValueError(
                "Tracked source tree changed between Direct and dense phases: "
                f"field={key!r}, direct={previous_tree.get(key)!r}, dense={current_tree.get(key)!r}."
            )


def _resolved(value: Any, *, label: str) -> Path:
    if value is None or not str(value).strip():
        raise ValueError(f"Missing required path: {label}.")
    resolved = str(resolve_path_string(value, PROJECT_ROOT) or "")
    if any(marker in resolved for marker in ("${", "$", "%")):
        raise ValueError(f"Unresolved environment variable in {label}: {resolved!r}.")
    path = Path(resolved).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _same_path(left: str | Path, right: str | Path) -> bool:
    return Path(left).expanduser().resolve() == Path(right).expanduser().resolve()


def _disabled_path(value: Any) -> bool:
    return str(value or "").strip().lower() in DISABLED_PATH_VALUES


def _state_digest(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    tensor_count = 0
    for name in sorted(state):
        value = state[name]
        if not isinstance(value, torch.Tensor):
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(str(name).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        tensor_count += 1
    if tensor_count == 0:
        raise ValueError("Cannot hash an adapter state with no tensors.")
    return digest.hexdigest()


def _checkpoint_adapter_state(payload: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    state = payload.get("adapter_state_dict")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Stage-1 checkpoint is missing adapter_state_dict.")
    if any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("Stage-1 adapter_state_dict contains a non-tensor value.")
    return state


def _random_spatial_state(payload: Mapping[str, Any], seed: int) -> dict[str, torch.Tensor]:
    state = _checkpoint_adapter_state(payload)
    checkpoint_args = payload.get("args")
    if not isinstance(checkpoint_args, Mapping):
        raise ValueError("Stage-1 checkpoint is missing its argument contract.")

    latent_weight = state.get("latent_projection.weight")
    output_weight = state.get("output.1.weight")
    position = state.get("spatial_pos_encoding")
    if not all(
        isinstance(value, torch.Tensor)
        for value in (latent_weight, output_weight, position)
    ):
        raise ValueError(
            "Stage-1 ablation requires a spatial_transformer alignment checkpoint."
        )
    if position.ndim != 3:
        raise ValueError(
            f"Expected [1,tokens,dim] spatial_pos_encoding, got {tuple(position.shape)}."
        )

    token_count = int(position.shape[-2])
    compressor_config = payload.get("compressor_config")
    compressor_model = (
        compressor_config.get("model")
        if isinstance(compressor_config, Mapping)
        and isinstance(compressor_config.get("model"), Mapping)
        else {}
    )
    raw_grid = compressor_model.get("latent_grid") if isinstance(compressor_model, Mapping) else None
    if (
        isinstance(raw_grid, Sequence)
        and not isinstance(raw_grid, (str, bytes))
        and len(raw_grid) == 2
    ):
        latent_grid = (int(raw_grid[0]), int(raw_grid[1]))
        if any(size <= 0 for size in latent_grid) or latent_grid[0] * latent_grid[1] != token_count:
            raise ValueError(
                "Stage-1 compressor latent_grid does not match spatial_pos_encoding: "
                f"grid={latent_grid}, tokens={token_count}."
            )
    else:
        side = int(round(token_count**0.5))
        if side * side != token_count:
            raise ValueError(
                "A legacy Stage-1 checkpoint without compressor_config.model.latent_grid "
                f"must use a square spatial grid; got {token_count} tokens."
            )
        latent_grid = (side, side)
    block_indices = {
        int(str(name).split(".")[1])
        for name in state
        if str(name).startswith("blocks.") and len(str(name).split(".")) > 2
    }
    layers = max(block_indices) + 1 if block_indices else int(
        checkpoint_args.get("adapter_layers", 2)
    )

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        adapter = TensorPatchAlignmentAdapter(
            latent_channels=int(latent_weight.shape[1]),
            latent_grid=latent_grid,
            adapter_dim=int(latent_weight.shape[0]),
            projection_dim=int(output_weight.shape[0]),
            dropout=float(checkpoint_args.get("dropout", 0.0)),
            adapter_type="spatial_transformer",
            query_tokens=token_count,
            adapter_layers=layers,
            adapter_heads=int(checkpoint_args.get("adapter_heads", 8)),
            soft_prompt_scale=float(checkpoint_args.get("soft_prompt_scale", 0.05)),
        )

    fresh = {
        name: value.detach().cpu().clone()
        for name, value in adapter.state_dict().items()
    }
    expected_keys = set(state)
    observed_keys = set(fresh)
    if observed_keys != expected_keys:
        raise ValueError(
            "Fresh adapter does not exactly match the Stage-1 architecture; "
            f"missing={sorted(expected_keys - observed_keys)}, "
            f"extra={sorted(observed_keys - expected_keys)}."
        )
    incompatible = {
        name: {
            "expected_shape": tuple(state[name].shape),
            "observed_shape": tuple(fresh[name].shape),
            "expected_dtype": str(state[name].dtype),
            "observed_dtype": str(fresh[name].dtype),
        }
        for name in sorted(expected_keys)
        if tuple(fresh[name].shape) != tuple(state[name].shape)
        or fresh[name].dtype != state[name].dtype
    }
    if incompatible:
        raise ValueError(f"Fresh adapter tensor contract differs from Stage 1: {incompatible}.")
    return fresh


def _initial_spatial_state(
    payload: Mapping[str, Any],
    seed: int,
    condition: str,
) -> dict[str, torch.Tensor]:
    _condition, spec = _condition_spec(condition)
    if spec["adapter_state_action"] == "preserve":
        return {
            str(name): value.detach().cpu().clone()
            for name, value in _checkpoint_adapter_state(payload).items()
        }
    return _random_spatial_state(payload, seed)


def _build_ablation_contract(
    source: Path,
    source_payload: Mapping[str, Any],
    initial_state: Mapping[str, Any],
    seed: int,
    condition: str = "adapter_only",
) -> dict[str, Any]:
    condition, spec = _condition_spec(condition)
    checkpoint_args = source_payload.get("args")
    if not isinstance(checkpoint_args, Mapping):
        raise ValueError("Stage-1 checkpoint is missing its argument contract.")
    compressor_config = source_payload.get("compressor_config")
    compressor_model = (
        compressor_config.get("model")
        if isinstance(compressor_config, Mapping)
        and isinstance(compressor_config.get("model"), Mapping)
        else None
    )
    if not isinstance(compressor_model, Mapping):
        raise ValueError("Stage-1 checkpoint is missing compressor_config.model.")
    if str(compressor_model.get("name", "")) != "conv_token_autoencoder_2d":
        raise ValueError(
            "Stage-1 comparison requires the conv_token_autoencoder_2d value-preserving encoder."
        )
    if compressor_model.get("preserve_input_channels") is not True:
        raise ValueError(
            "Stage-1 ablation requires compressor_config.model.preserve_input_channels=true "
            "so latent channel 0 has an explicit value-channel contract."
        )
    if int(compressor_model.get("in_channels", 0)) != 1:
        raise ValueError(
            "Stage-1 comparison requires compressor_config.model.in_channels=1 because "
            "value_only preserves exactly latent channel 0."
        )
    input_size = tuple(int(value) for value in compressor_model.get("input_size", ()))
    latent_grid = tuple(int(value) for value in compressor_model.get("latent_grid", ()))
    if len(input_size) != 2 or input_size != latent_grid:
        raise ValueError(
            "Stage-1 comparison requires input_size == latent_grid so preserved value-channel "
            f"positions remain exact; input={input_size}, latent={latent_grid}."
        )
    encoder_trained_during_alignment = bool(
        checkpoint_args.get(
            "alignment_train_patch_ae",
            bool(checkpoint_args.get("train_patch_ae", False))
            and not bool(checkpoint_args.get("freeze_patch_ae_after_pretrain", True)),
        )
    )
    encoder_origin = str(checkpoint_args.get("encoder_source", ""))
    encoder_checkpoint_sha = str(
        checkpoint_args.get("compressor_checkpoint_sha256", "") or ""
    ).lower()
    encoder_lineage_complete = encoder_origin == "patch_ae_config" or (
        encoder_origin == "checkpoint"
        and len(encoder_checkpoint_sha) == 64
        and all(character in "0123456789abcdef" for character in encoder_checkpoint_sha)
    )
    compressor_state = source_payload.get("compressor_state_dict")
    if not isinstance(compressor_state, Mapping):
        raise ValueError("Stage-1 checkpoint is missing its embedded compressor state.")
    source_state_sha256 = _state_digest(_checkpoint_adapter_state(source_payload))
    initial_state_sha256 = _state_digest(initial_state)
    should_preserve = spec["adapter_state_action"] == "preserve"
    if should_preserve and source_state_sha256 != initial_state_sha256:
        raise ValueError("The matched reference must preserve the exact Stage-1 adapter state.")
    if not should_preserve and source_state_sha256 == initial_state_sha256:
        raise ValueError(
            "The random adapter exactly matches the Stage-1 adapter; the requested factor "
            "was not removed. Choose another ablation seed or inspect the source checkpoint."
        )
    return {
        "contract_version": ABLATION_CONTRACT_VERSION,
        "enabled": True,
        "condition": condition,
        "condition_role": spec["condition_role"],
        "adapter_state_action": spec["adapter_state_action"],
        "factor_removed": spec["factor_removed"],
        "mode": spec["mode"],
        "claim_scope": spec["claim_scope"],
        "latent_channel_policy": spec["latent_channel_policy"],
        "value_channel_index": 0,
        "value_channel_contract": {
            "preserve_input_channels": True,
            "input_channels": 1,
            "input_size": list(input_size),
            "latent_grid": list(latent_grid),
            "semantic": "per_patch_standardized_scalar",
        },
        "seed": int(seed),
        "source_stage1_checkpoint": str(source),
        "source_stage1_sha256": sha256_file(source),
        "source_adapter_state_sha256": source_state_sha256,
        "initial_adapter_state_sha256": initial_state_sha256,
        "source_encoder_trained_during_alignment": encoder_trained_during_alignment,
        "source_encoder_origin": encoder_origin,
        "source_encoder_checkpoint": str(checkpoint_args.get("compressor_checkpoint", "") or ""),
        "source_encoder_lineage_complete": encoder_lineage_complete,
        "source_encoder_state_sha256": _state_digest(compressor_state),
        "changed": [
            *(
                ["initial_spatial_adapter_learned_state"]
                if spec["adapter_state_action"] == "randomize"
                else []
            ),
            *(
                ["runtime_latent_channels_1_to_n_zeroed"]
                if spec["latent_channel_policy"] == "value_only"
                else []
            ),
        ],
        "preserved": [
            "immutable_cached_latent_files",
            "exact_standardized_value_channel_0",
            "spatial_adapter_architecture_and_parameter_count",
            "direct_qa_objective_data_and_configured_example_budget",
            "dense_cross_attention_objective_data_and_configured_example_budget",
            "frozen_qwen",
        ],
    }


def _load_source_ablation(
    source: Path,
    seed: int,
    condition: str = "adapter_only",
    load_fn: Any = torch.load,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, Any]]:
    payload = load_fn(source, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("Stage-1 checkpoint payload must be a mapping.")
    validate_stage1_alignment_checkpoint_payload(payload, path=source)
    payload_copy = dict(payload)
    initial_state = _initial_spatial_state(payload_copy, seed, condition)
    contract = _build_ablation_contract(
        source,
        payload_copy,
        initial_state,
        seed,
        condition=condition,
    )
    return payload_copy, initial_state, contract


def _override_cli_tokens(overrides: Mapping[str, Any]) -> list[str]:
    tokens: list[str] = []
    for raw_name, value in overrides.items():
        name = str(raw_name).strip()
        if not name or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in name):
            raise ValueError(f"Invalid direct trainer argument name: {raw_name!r}.")
        option = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            tokens.append(option if value else "--no-" + option[2:])
            continue
        if value is None:
            tokens.extend((option, "none"))
            continue
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            tokens.extend((option, ",".join(str(item) for item in value)))
            continue
        if isinstance(value, Mapping):
            raise ValueError(f"Direct trainer override {name!r} cannot be a nested mapping.")
        tokens.extend((option, str(value)))
    return tokens


def _validate_override_keys(
    overrides: Mapping[str, Any],
    *,
    allowed: frozenset[str],
    section: str,
) -> None:
    unknown = sorted(str(key) for key in overrides if str(key) not in allowed)
    if unknown:
        raise ValueError(
            f"{section}.arg_overrides may change only non-scientific orchestration settings "
            f"{sorted(allowed)}; got {unknown}."
        )


def _validate_forwarded_flags(forwarded: Sequence[str], *, phase: str) -> None:
    allowed = DIRECT_FORWARDED_FLAGS if phase == "direct" else DENSE_FORWARDED_FLAGS
    unsupported = [str(token) for token in forwarded if str(token) not in allowed]
    if unsupported:
        raise ValueError(
            f"Stage-1 comparison {phase} phase forbids scientific CLI overrides: "
            f"allowed={sorted(allowed)}, got={unsupported}."
        )


def _validate_direct_args(args: argparse.Namespace, source: Path) -> None:
    if str(args.adapter_architecture) != "alignment_adapter":
        raise ValueError(
            "Stage-1 ablation direct phase must use adapter_architecture=alignment_adapter."
        )
    configured_source = Path(str(args.adapter_init_checkpoint)).expanduser().resolve()
    if configured_source != source:
        raise ValueError(
            "The direct adapter initializer must be the source checkpoint named by the "
            f"ablation contract: configured={configured_source}, source={source}."
        )
    if not _disabled_path(args.stage2_warm_start_checkpoint) or not _disabled_path(
        args.stage2b_resume_checkpoint
    ):
        raise ValueError(
            "Stage-1 comparison must train Direct QA from its declared Stage-1 state; "
            "Stage-2 warm-start and resume checkpoints must be disabled."
        )
    active_modes = [
        name
        for name in (
            "joint_ab_training",
            "point_reader_training",
            "full_local_reader_training",
        )
        if bool(getattr(args, name, False))
    ]
    if active_modes:
        raise ValueError(
            "Direct Stage-1 ablation cannot enable later Stage-2B modes: "
            f"{active_modes}."
        )


def _validate_ablation_contract(
    contract: Mapping[str, Any],
    configured_source: Path,
) -> dict[str, Any]:
    normalized = copy.deepcopy(dict(contract))
    if int(normalized.get("contract_version", 0)) != ABLATION_CONTRACT_VERSION:
        raise ValueError("Unsupported or missing Stage-1 ablation contract version.")
    traceability = normalized.get("traceability")
    if not isinstance(traceability, Mapping) or normalized.get("traceability_sha256") != _json_digest(
        dict(traceability)
    ):
        raise ValueError("Stage-1 ablation checkpoint has invalid traceability provenance.")
    evaluation_contract = normalized.get("evaluation_contract")
    if not isinstance(evaluation_contract, Mapping) or normalized.get(
        "evaluation_contract_sha256"
    ) != _json_digest(dict(evaluation_contract)):
        raise ValueError("Stage-1 ablation checkpoint has invalid evaluation preregistration.")
    condition, spec = _condition_spec(normalized.get("condition"))
    if normalized.get("enabled") is not True or normalized.get("mode") != spec["mode"]:
        raise ValueError("Checkpoint is not a Stage-1 alignment ablation checkpoint.")
    contract_source = normalized.get("source_stage1_checkpoint")
    if not contract_source:
        raise ValueError("Ablation checkpoint does not record its source Stage-1 path.")
    actual_source_sha256 = sha256_file(configured_source)
    if normalized.get("source_stage1_sha256") != actual_source_sha256:
        raise ValueError(
            "Source Stage-1 checkpoint changed after the ablation direct phase."
        )
    seed = int(normalized.get("seed", -1))
    source_payload, initial_state, expected = _load_source_ablation(
        configured_source,
        seed,
        condition=condition,
    )
    del source_payload, initial_state
    for key in (
        "source_adapter_state_sha256",
        "initial_adapter_state_sha256",
        "condition_role",
        "adapter_state_action",
        "source_encoder_trained_during_alignment",
        "source_encoder_origin",
        "source_encoder_checkpoint",
        "source_encoder_lineage_complete",
        "source_encoder_state_sha256",
        "factor_removed",
        "latent_channel_policy",
        "value_channel_index",
        "value_channel_contract",
    ):
        if normalized.get(key) != expected.get(key):
            raise ValueError(
                f"Ablation contract field {key!r} cannot be reproduced: "
                f"observed={normalized.get(key)!r}, expected={expected.get(key)!r}."
            )
    return normalized


def _ablation_contract_identity(contract: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "contract_version",
        "enabled",
        "condition",
        "condition_role",
        "adapter_state_action",
        "factor_removed",
        "mode",
        "latent_channel_policy",
        "value_channel_index",
        "value_channel_contract",
        "seed",
        "source_stage1_sha256",
        "source_adapter_state_sha256",
        "initial_adapter_state_sha256",
        "source_encoder_trained_during_alignment",
        "source_encoder_origin",
        "source_encoder_checkpoint",
        "source_encoder_lineage_complete",
        "source_encoder_state_sha256",
        "direct_qa_checkpoint_sha256",
        "direct_run_audit_sha256",
        "dense_training_recipe",
        "traceability_sha256",
        "evaluation_contract_sha256",
        "direct_forwarded_cli",
    )
    identity = {key: contract.get(key) for key in keys if key in contract}
    dense_trace = contract.get("dense_traceability")
    if isinstance(dense_trace, Mapping):
        source_tree = dense_trace.get("source_tree")
        source_tree = dict(source_tree) if isinstance(source_tree, Mapping) else {}
        identity["dense_traceability_stable"] = {
            "base_config_sha256": dense_trace.get("base_config_sha256"),
            "runtime_versions": dense_trace.get("runtime_versions"),
            "git_available": source_tree.get("git_available"),
            "commit": source_tree.get("commit"),
            "tracked_dirty": source_tree.get("tracked_dirty"),
            "tracked_diff_sha256": source_tree.get("tracked_diff_sha256"),
        }
    return identity


def _run_direct(
    config: Mapping[str, Any],
    orchestration_config: Path,
    forwarded: list[str],
) -> None:
    import scripts.train_tensor_llm_adapter as trainer

    section = config.get("ablation", {})
    direct = config.get("direct", {})
    if not isinstance(section, Mapping) or not isinstance(direct, Mapping):
        raise ValueError("Ablation config requires mapping sections: ablation and direct.")
    source = _resolved(
        section.get("source_stage1_checkpoint"),
        label="ablation.source_stage1_checkpoint",
    )
    if not source.is_file():
        raise FileNotFoundError(f"Stage-1 checkpoint not found: {source}")
    seed = int(section.get("seed", 42))
    condition, _spec = _condition_spec(section.get("condition", "adapter_only"))
    base_config = _resolved(direct.get("base_config"), label="direct.base_config")
    if not base_config.is_file():
        raise FileNotFoundError(f"Direct base config not found: {base_config}")
    overrides = direct.get("arg_overrides", {})
    if not isinstance(overrides, Mapping):
        raise ValueError("direct.arg_overrides must be a mapping of trainer argument names.")
    _validate_override_keys(overrides, allowed=DIRECT_OVERRIDE_KEYS, section="direct")
    _validate_forwarded_flags(forwarded, phase="direct")
    if overrides.get("evaluate_test") is not False:
        raise ValueError("Direct comparison phase requires arg_overrides.evaluate_test=false.")
    override_tokens = _override_cli_tokens(overrides)

    original_parse = trainer.parse_args
    original_load = torch.load
    original_save = trainer.atomic_torch_save
    original_config_snapshot = trainer.redacted_config_snapshot
    original_argv = list(sys.argv)
    source_payload, initial_state, contract = _load_source_ablation(
        source,
        seed,
        condition=condition,
        load_fn=original_load,
    )
    del source_payload
    traceability = _traceability_contract(
        orchestration_config=orchestration_config,
        base_config=base_config,
        resolved_config=config,
    )
    _enforce_traceability(
        traceability,
        config.get("traceability") if isinstance(config.get("traceability"), Mapping) else None,
    )
    contract["traceability"] = traceability
    contract["traceability_sha256"] = _json_digest(traceability)
    evaluation_contract = _validated_evaluation_contract(config.get("evaluation_contract"))
    contract["evaluation_contract"] = evaluation_contract
    contract["evaluation_contract_sha256"] = _json_digest(evaluation_contract)
    contract["direct_forwarded_cli"] = [str(token) for token in forwarded]

    def parse_args() -> argparse.Namespace:
        args = original_parse()
        _validate_direct_args(args, source)
        if bool(args.evaluate_test):
            raise ValueError("Direct comparison phase cannot access the test split.")
        if str(args.latent_channel_policy) != "all":
            raise ValueError(
                "The production Direct base config must use latent_channel_policy='all'; "
                "the ablation wrapper owns this factor."
            )
        args.latent_channel_policy = str(contract["latent_channel_policy"])
        args.stage1_ablation = True
        args.stage1_ablation_condition = condition
        args.stage1_ablation_mode = str(contract["mode"])
        args.stage1_ablation_seed = seed
        args.stage1_ablation_source_checkpoint = str(source)
        args.stage1_ablation_contract = copy.deepcopy(contract)
        return args

    def load(path: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            candidate = Path(path)
            matches = candidate.name == source.name and _same_path(candidate, source)
        except (TypeError, OSError, ValueError):
            matches = False
        payload = original_load(path, *args, **kwargs)
        if not matches:
            return payload
        if not isinstance(payload, Mapping):
            raise ValueError("Stage-1 checkpoint payload must be a mapping.")
        replaced = dict(payload)
        replaced["adapter_state_dict"] = initial_state
        replaced["ablation_contract"] = copy.deepcopy(contract)
        return replaced

    def save(path: Any, payload: Mapping[str, Any]) -> None:
        enriched = dict(payload)
        if enriched.get("checkpoint_type") == "tensor_llm_adapter":
            enriched["ablation_contract"] = copy.deepcopy(contract)
        original_save(path, enriched)

    def config_snapshot(config_payload: Mapping[str, Any]) -> dict[str, Any]:
        snapshot = original_config_snapshot(config_payload)
        snapshot["stage1_ablation"] = copy.deepcopy(contract)
        snapshot["stage1_ablation_direct_arg_overrides"] = copy.deepcopy(
            dict(overrides)
        )
        return snapshot

    trainer.parse_args = parse_args
    trainer.atomic_torch_save = save
    trainer.redacted_config_snapshot = config_snapshot
    torch.load = load
    sys.argv = [
        original_argv[0],
        "--config",
        str(base_config),
        *override_tokens,
        *forwarded,
    ]
    try:
        trainer.main()
    finally:
        trainer.parse_args = original_parse
        trainer.atomic_torch_save = original_save
        trainer.redacted_config_snapshot = original_config_snapshot
        torch.load = original_load
        sys.argv = original_argv


def _run_dense(
    config: Mapping[str, Any],
    orchestration_config: Path,
    init_checkpoint: str,
    forwarded: list[str],
    resume_checkpoint: str | None = None,
    protocol_lock: str | None = None,
) -> None:
    import scripts.train_tensor_qwen_cross_attention as trainer

    section = config.get("ablation", {})
    dense = config.get("dense", {})
    if not isinstance(section, Mapping) or not isinstance(dense, Mapping):
        raise ValueError("Ablation config requires mapping sections: ablation and dense.")
    source = _resolved(
        section.get("source_stage1_checkpoint"),
        label="ablation.source_stage1_checkpoint",
    )
    base_config = _resolved(dense.get("base_config"), label="dense.base_config")
    if not base_config.is_file():
        raise FileNotFoundError(f"Dense base config not found: {base_config}")
    overrides = dense.get("arg_overrides", {})
    if not isinstance(overrides, Mapping):
        raise ValueError("dense.arg_overrides must be a mapping of trainer argument names.")
    _validate_override_keys(overrides, allowed=DENSE_OVERRIDE_KEYS, section="dense")
    _validate_forwarded_flags(forwarded, phase="dense")
    if "--evaluate-test" in forwarded and "--no-evaluate-test" in forwarded:
        raise ValueError("Dense comparison cannot receive conflicting test-access flags.")
    wants_test = "--evaluate-test" in forwarded
    _validate_test_access_request(
        wants_test=wants_test,
        protocol_lock=protocol_lock,
        resume_checkpoint=resume_checkpoint,
    )
    test_protocol_lock = None
    if protocol_lock:
        protocol_lock_path = Path(protocol_lock).expanduser().resolve()
        if not protocol_lock_path.is_file():
            raise FileNotFoundError(f"Test protocol lock not found: {protocol_lock_path}")
        test_protocol_lock = _validated_protocol_lock(protocol_lock_path)
    if overrides.get("evaluate_test") is not False:
        raise ValueError(
            "Dense comparison config must default arg_overrides.evaluate_test=false; "
            "test access requires an explicit post-lock CLI flag."
        )
    configured_evaluation = _validated_evaluation_contract(config.get("evaluation_contract"))
    expected_screening_updates = int(configured_evaluation["dense_screening_max_updates"])
    if int(overrides.get("max_updates", 0)) != expected_screening_updates or expected_screening_updates <= 0:
        raise ValueError(
            "Dense max_updates must equal the positive preregistered screening budget: "
            f"override={overrides.get('max_updates')!r}, preregistered={expected_screening_updates}."
        )
    override_tokens = _override_cli_tokens(overrides)
    init_path = Path(init_checkpoint).expanduser().resolve()
    if not init_path.is_file():
        raise FileNotFoundError(f"Ablated Direct-QA checkpoint not found: {init_path}")
    resume_tokens: list[str] = []
    resume_path: Path | None = None
    if resume_checkpoint:
        resume_path = Path(resume_checkpoint).expanduser().resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"Dense resume checkpoint not found: {resume_path}")
        resume_tokens = ["--resume", str(resume_path)]
    payload = torch.load(init_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("Ablated Direct-QA checkpoint must be a mapping.")
    if payload.get("checkpoint_type") != "tensor_llm_adapter":
        raise ValueError("Dense phase requires a Direct-QA tensor_llm_adapter checkpoint.")
    raw_contract = payload.get("ablation_contract")
    if not isinstance(raw_contract, Mapping):
        raise ValueError(
            "Dense Stage-1 ablation requires adapter_best.pt produced by this script's direct phase."
        )
    contract = _validate_ablation_contract(raw_contract, source)
    _validate_condition_match(section.get("condition"), contract)
    if test_protocol_lock is not None:
        if test_protocol_lock["source_stage1_sha256"] != contract["source_stage1_sha256"]:
            raise ValueError("Test protocol lock belongs to a different source Stage-1 checkpoint.")
        if test_protocol_lock["evaluation_contract_sha256"] != contract[
            "evaluation_contract_sha256"
        ]:
            raise ValueError("Test protocol lock belongs to a different evaluation contract.")
    if _json_digest(configured_evaluation) != contract.get("evaluation_contract_sha256"):
        raise ValueError(
            "Dense comparison evaluation contract differs from the Direct preregistration."
        )
    recorded_traceability = contract.get("traceability")
    if not isinstance(recorded_traceability, Mapping):
        raise ValueError("Ablated Direct-QA checkpoint is missing traceability provenance.")
    _validate_recorded_traceability(
        recorded_traceability,
        orchestration_config=orchestration_config,
    )
    direct_args = payload.get("args")
    if not isinstance(direct_args, Mapping) or direct_args.get(
        "adapter_architecture"
    ) != "alignment_adapter":
        raise ValueError("Ablated Direct-QA checkpoint has an invalid adapter architecture.")
    if str(direct_args.get("latent_channel_policy", "all")) != str(
        contract["latent_channel_policy"]
    ):
        raise ValueError(
            "Ablated Direct-QA checkpoint does not apply the latent policy recorded by its contract."
        )
    direct_run_audit = _direct_run_audit(
        init_path,
        payload,
        expected_policy=str(contract["latent_channel_policy"]),
    )
    direct_run_artifacts = _mapping_for_digest(
        direct_run_audit.get("artifacts"), "Direct run audit artifacts"
    )
    direct_checkpoint_sha256 = sha256_file(init_path)
    if direct_run_artifacts.get("direct_checkpoint_sha256") != direct_checkpoint_sha256:
        raise ValueError("Direct run audit checkpoint digest differs from the supplied initializer.")
    dense_traceability = _traceability_contract(
        orchestration_config=orchestration_config,
        base_config=base_config,
        resolved_config=config,
    )
    _enforce_traceability(
        dense_traceability,
        config.get("traceability") if isinstance(config.get("traceability"), Mapping) else None,
    )
    dense_contract = {
        **contract,
        "runtime_source_stage1_checkpoint": str(source),
        "direct_qa_checkpoint": str(init_path),
        "direct_qa_checkpoint_sha256": direct_checkpoint_sha256,
        "direct_run_audit": direct_run_audit,
        "direct_run_audit_sha256": _json_digest(direct_run_audit),
        "dense_traceability": dense_traceability,
        "dense_traceability_sha256": _json_digest(dense_traceability),
        "dense_forwarded_cli": [str(token) for token in forwarded],
        "test_protocol_lock": test_protocol_lock,
    }
    locked_dense_identities: list[tuple[str, dict[str, Any]]] = []
    if test_protocol_lock is not None:
        if resume_path is None:
            raise AssertionError("Test protocol lock reached without a resume checkpoint.")
        if resume_path.name != "cross_attention_last.pt":
            raise ValueError(
                "Sealed test evaluation must resume the locked cross_attention_last.pt."
            )
        locked_by_condition = _mapping_for_digest(
            test_protocol_lock.get("validation_artifacts"),
            "locked validation artifacts",
        )
        locked = _mapping_for_digest(
            locked_by_condition.get(str(contract["condition"])),
            f"locked validation artifacts for {contract['condition']}",
        )
        validation_run_dir = resume_path.parent
        _require_locked_file(
            validation_run_dir / "run_summary.json",
            locked.get("run_summary_sha256"),
            "run summary",
        )
        _require_locked_file(
            validation_run_dir / "data_audit.json",
            locked.get("data_audit_sha256"),
            "data audit",
        )
        _require_locked_file(
            validation_run_dir / "cross_attention_best.pt",
            locked.get("best_checkpoint_sha256"),
            "best checkpoint",
        )
        _require_locked_file(
            resume_path,
            locked.get("last_checkpoint_sha256"),
            "last checkpoint",
        )
        _require_locked_file(
            init_path,
            locked.get("direct_qa_checkpoint_sha256"),
            "Direct initializer",
        )
        direct_field_map = {
            "run_summary_sha256": "direct_run_summary_sha256",
            "run_timing_sha256": "direct_run_timing_sha256",
            "data_audit_sha256": "direct_data_audit_sha256",
            "qa_metadata_audit_sha256": "direct_qa_metadata_audit_sha256",
        }
        for audit_field, locked_field in direct_field_map.items():
            if direct_run_artifacts.get(audit_field) != locked.get(locked_field):
                raise ValueError(
                    "Direct run lineage changed before test access: "
                    f"field={locked_field}."
                )
        best_path = validation_run_dir / "cross_attention_best.pt"
        resume_payload = torch.load(resume_path, map_location="cpu", weights_only=True)
        resume_ablation, resume_progress = _validated_dense_checkpoint_payload(
            resume_payload,
            label="Locked dense resume checkpoint",
            require_optimizer_state=True,
        )
        _validate_dense_lineage(
            resume_ablation,
            dense_contract,
            label="Locked dense resume checkpoint",
        )
        locked_dense_identities.append(
            ("resume", _ablation_contract_identity(resume_ablation))
        )
        if int(resume_progress.get("global_step", -1)) != expected_screening_updates:
            raise ValueError(
                "Locked dense resume checkpoint did not reach the preregistered validation budget."
            )
        del resume_ablation, resume_payload, resume_progress
        best_payload = torch.load(best_path, map_location="cpu", weights_only=True)
        best_ablation, best_progress = _validated_dense_checkpoint_payload(
            best_payload,
            label="Locked dense best checkpoint",
            require_optimizer_state=False,
        )
        _validate_dense_lineage(
            best_ablation,
            dense_contract,
            label="Locked dense best checkpoint",
        )
        locked_dense_identities.append(
            ("best", _ablation_contract_identity(best_ablation))
        )
        best_step = int(best_progress.get("global_step", -1))
        if not 0 <= best_step <= expected_screening_updates:
            raise ValueError("Locked dense best checkpoint has invalid training progress.")
        del best_ablation, best_payload, best_progress

    original_parse = trainer.parse_args
    original_architecture_contract = trainer.architecture_contract
    original_validate_checkpoint = trainer.validate_checkpoint_contract
    original_argv = list(sys.argv)

    def parse_args() -> argparse.Namespace:
        args = original_parse()
        args.memory_init_checkpoint = str(init_path)
        if str(args.latent_channel_policy) != "all":
            raise ValueError(
                "The production dense base config must use latent_channel_policy='all'; "
                "the ablation wrapper owns this factor."
            )
        args.latent_channel_policy = str(dense_contract["latent_channel_policy"])
        try:
            launch_world_size = int(os.environ.get("WORLD_SIZE", "1"))
        except (TypeError, ValueError) as error:
            raise ValueError("WORLD_SIZE must be a positive integer for matched training.") from error
        if launch_world_size <= 0:
            raise ValueError("WORLD_SIZE must be a positive integer for matched training.")
        dense_training_recipe = {
            "world_size": launch_world_size,
            "per_rank_batch_size": int(args.batch_size),
            "eval_batch_size": int(args.eval_batch_size),
            "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
            "effective_batch_size": int(args.batch_size)
            * int(args.gradient_accumulation_steps)
            * launch_world_size,
            "epochs": int(args.epochs),
            "max_updates": int(args.max_updates),
            "seed": int(args.seed),
            "shuffle_seed": int(args.shuffle_seed),
            "record_subset_mode": str(args.record_subset_mode),
            "max_train_records": int(args.max_train_records),
            "max_val_records": int(args.max_val_records),
            "max_test_records": int(args.max_test_records),
        }
        dense_contract["dense_training_recipe"] = dense_training_recipe
        expected_dense_identity = _ablation_contract_identity(dense_contract)
        for checkpoint_label, observed_identity in locked_dense_identities:
            if observed_identity != expected_dense_identity:
                raise ValueError(
                    "Locked dense checkpoint recipe differs from the current test launch: "
                    f"checkpoint={checkpoint_label}."
                )
        args.stage1_ablation = True
        args.stage1_ablation_condition = str(dense_contract["condition"])
        args.stage1_ablation_mode = str(dense_contract["mode"])
        args.stage1_ablation_contract = copy.deepcopy(dense_contract)
        raw = copy.deepcopy(args.raw_config)
        raw_memory = raw.get("memory")
        memory = copy.deepcopy(dict(raw_memory)) if isinstance(raw_memory, Mapping) else {}
        memory["init_checkpoint"] = str(init_path)
        memory["latent_channel_policy"] = str(dense_contract["latent_channel_policy"])
        raw["memory"] = memory
        raw["stage1_ablation"] = copy.deepcopy(dense_contract)
        args.raw_config = raw
        trainer.validate_args(args)
        return args

    def architecture_contract(*args: Any, **kwargs: Any) -> dict[str, Any]:
        architecture = original_architecture_contract(*args, **kwargs)
        architecture["stage1_ablation"] = copy.deepcopy(dense_contract)
        return architecture

    def validate_checkpoint_contract(
        checkpoint: Mapping[str, Any],
        expected: Mapping[str, Any],
    ) -> None:
        original_validate_checkpoint(checkpoint, expected)
        observed_architecture = checkpoint.get("architecture")
        observed = (
            observed_architecture.get("stage1_ablation")
            if isinstance(observed_architecture, Mapping)
            else None
        )
        expected_ablation = expected.get("stage1_ablation")
        if not isinstance(observed, Mapping) or not isinstance(
            expected_ablation, Mapping
        ) or _ablation_contract_identity(observed) != _ablation_contract_identity(
            expected_ablation
        ):
            raise ValueError(
                "Dense checkpoint does not belong to this Stage-1 ablation lineage."
            )

    trainer.parse_args = parse_args
    trainer.architecture_contract = architecture_contract
    trainer.validate_checkpoint_contract = validate_checkpoint_contract
    sys.argv = [
        original_argv[0],
        "--config",
        str(base_config),
        *override_tokens,
        *resume_tokens,
        *forwarded,
    ]
    try:
        trainer.main()
    finally:
        trainer.parse_args = original_parse
        trainer.architecture_contract = original_architecture_contract
        trainer.validate_checkpoint_contract = original_validate_checkpoint
        sys.argv = original_argv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("direct", "dense"))
    parser.add_argument("--config", required=True, help="Stage-1 ablation orchestration YAML.")
    parser.add_argument(
        "--spatial-init-checkpoint",
        help="Direct-phase adapter_best.pt; required only for the dense phase.",
    )
    parser.add_argument(
        "--resume",
        help="Optional dense cross_attention_last.pt for recoverable continuation.",
    )
    parser.add_argument(
        "--protocol-lock",
        help="Filled protocol-lock JSON required when the dense phase opens the test split.",
    )
    known, forwarded = parser.parse_known_args()
    orchestration_config = _resolved(known.config, label="--config")
    config = load_yaml_mapping(orchestration_config)
    if known.phase == "direct":
        if known.spatial_init_checkpoint:
            raise ValueError("--spatial-init-checkpoint is only valid for the dense phase.")
        if known.resume:
            raise ValueError("--resume is only valid for the dense phase.")
        if known.protocol_lock:
            raise ValueError("--protocol-lock is only valid for the dense phase.")
        _run_direct(config, orchestration_config, forwarded)
        return
    if not known.spatial_init_checkpoint:
        raise ValueError("dense phase requires --spatial-init-checkpoint.")
    _run_dense(
        config,
        orchestration_config,
        known.spatial_init_checkpoint,
        forwarded,
        resume_checkpoint=known.resume,
        protocol_lock=known.protocol_lock,
    )


if __name__ == "__main__":
    main()
