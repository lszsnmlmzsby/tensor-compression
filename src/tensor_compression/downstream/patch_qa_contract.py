from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch


PATCH_QA_FORMAT = "tensor_patch_qa_v3"
PATCH_MATCHED_QA_FORMAT = "tensor_patch_matched_qa_v1"
MATCHED_GROUP_FORMAT = "stage2b_matched_groups_v1"
PATCH_QA_PROMPT_CONTRACT = "encoder_zscore_one_based_v2"
PATCH_QA_BUILD_MARKER = ".build_in_progress.json"
PATCH_LATENT_FORMAT = "tensor_patch_latent_v1"
PATCH_LATENT_AUDIT_FORMAT = "per_patch_zscore_v1"
STAGE1_ALIGNMENT_CHECKPOINT_TYPE = "tensor_patch_text_alignment"
STAGE1_ALIGNMENT_CHECKPOINT_FILENAMES = frozenset(
    {"alignment_best.pt", "alignment_last.pt"}
)


def validate_stage1_alignment_checkpoint_payload(
    payload: Any,
    *,
    path: str | Path,
) -> dict[str, Any]:
    """Validate the provenance envelope of a complete Stage-1 checkpoint.

    Version 3 checkpoints carry an explicit type and phase.  Older checkpoints
    predate those fields, so the only trustworthy discriminator left is the
    canonical Stage-1 alignment filename, together with the complete payload
    written by the old alignment saver.  In particular, an old
    ``patch_ae_pretrain_*.pt`` file must never be accepted as an alignment
    initializer merely because it contains an adapter state dict.
    """
    if not isinstance(payload, Mapping):
        raise ValueError(f"Unsupported Stage-1 checkpoint payload: {path}")

    try:
        checkpoint_version = int(payload.get("checkpoint_version", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Stage-1 checkpoint_version must be an integer, got "
            f"{payload.get('checkpoint_version')!r}."
        ) from exc
    if checkpoint_version < 0:
        raise ValueError(f"Stage-1 checkpoint_version must be non-negative, got {checkpoint_version}.")

    checkpoint_type = str(payload.get("checkpoint_type", "")).strip()
    checkpoint_phase = str(payload.get("checkpoint_phase", "")).strip().lower()
    source_name = Path(path).expanduser().name.casefold()

    if checkpoint_version >= 3:
        if checkpoint_type != STAGE1_ALIGNMENT_CHECKPOINT_TYPE:
            raise ValueError(
                "Stage-1 checkpoint version 3 or newer must declare "
                f"checkpoint_type={STAGE1_ALIGNMENT_CHECKPOINT_TYPE!r}; "
                f"got {checkpoint_type!r}."
            )
        if checkpoint_phase != "alignment":
            phase_description = "missing checkpoint_phase" if not checkpoint_phase else f"got {checkpoint_phase!r}"
            raise ValueError(
                "Direct Stage 2 requires a completed Stage-1 alignment checkpoint, not a patch-AE warmup "
                "checkpoint; version 3 or newer must have checkpoint_phase='alignment'; "
                f"{phase_description}."
            )
        validation_mode = "strict_metadata"
        legacy_filename_fallback = False
    else:
        if checkpoint_type and checkpoint_type != STAGE1_ALIGNMENT_CHECKPOINT_TYPE:
            raise ValueError(
                "The checkpoint declares a non-Stage-1 type: "
                f"checkpoint_type={checkpoint_type!r}."
            )
        if checkpoint_phase and checkpoint_phase != "alignment":
            raise ValueError(
                "A legacy Stage-1 checkpoint with an explicit non-alignment phase cannot initialize Stage 2: "
                f"checkpoint_phase={checkpoint_phase!r}."
            )
        if source_name not in STAGE1_ALIGNMENT_CHECKPOINT_FILENAMES:
            allowed = ", ".join(sorted(STAGE1_ALIGNMENT_CHECKPOINT_FILENAMES))
            raise ValueError(
                "A legacy Stage-1 checkpoint lacks explicit phase metadata and is accepted only when its "
                f"filename is one of {{{allowed}}}; got {Path(path).name!r}. "
                "Do not pass patch_ae_pretrain_*.pt or a downstream adapter checkpoint."
            )
        validation_mode = "legacy_alignment_filename"
        legacy_filename_fallback = True

    required_mappings = (
        "adapter_state_dict",
        "compressor_config",
        "compressor_state_dict",
        "args",
    )
    missing_or_invalid = [
        name
        for name in required_mappings
        if not isinstance(payload.get(name), Mapping) or not payload.get(name)
    ]
    if missing_or_invalid:
        raise ValueError(
            "A complete Stage-1 alignment checkpoint must contain non-empty mapping fields "
            f"{required_mappings}; invalid or missing={missing_or_invalid}."
        )

    return {
        "checkpoint_version": checkpoint_version,
        "checkpoint_type": checkpoint_type or STAGE1_ALIGNMENT_CHECKPOINT_TYPE,
        "checkpoint_phase": checkpoint_phase or "alignment",
        "validation_mode": validation_mode,
        "legacy_filename_fallback": legacy_filename_fallback,
    }


def canonical_path(value: str | Path) -> str:
    return str(Path(value).expanduser().resolve())


def sha256_file(path: str | Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Checkpoint file not found for SHA-256 provenance: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(int(chunk_bytes)):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def atomic_torch_save(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json_dump(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, ensure_ascii=False, indent=2)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def canonical_normalization(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "mode": str(config.get("mode", "none")).lower(),
        "scope": str(config.get("scope", "global")).lower(),
        "stats_path": config.get("stats_path"),
        "clip_min": config.get("clip_min"),
        "clip_max": config.get("clip_max"),
    }


def _identity_integer(value: Any, name: str) -> int:
    try:
        converted = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Patch metadata field {name} must be an integer, got {value!r}.") from exc
    if isinstance(value, float) and (not math.isfinite(value) or value != converted):
        raise ValueError(f"Patch metadata field {name} must be an integer, got {value!r}.")
    return converted


def _identity_pair(value: Any, name: str) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"Patch metadata field {name} must be a two-element integer sequence, got {value!r}.")
    return [
        _identity_integer(value[0], f"{name}[0]"),
        _identity_integer(value[1], f"{name}[1]"),
    ]


def latent_identity_from_record(record: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise ValueError(f"A patch-latent record must be a mapping, got {type(record).__name__}.")
    top_level_field = record.get("field")
    metadata_field = (
        record["metadata"].get("field")
        if isinstance(record.get("metadata"), Mapping)
        else None
    )
    if top_level_field is not None and metadata_field is not None and str(top_level_field) != str(
        metadata_field
    ):
        raise ValueError(
            "A formal patch-latent record has inconsistent top-level and metadata field values."
        )
    field = top_level_field
    fields_field = None
    fields = record.get("fields")
    if isinstance(fields, Sequence) and not isinstance(fields, (str, bytes)) and fields:
        fields_field = fields[0]
        if field is not None and str(field) != str(fields_field):
            raise ValueError(
                "A formal patch-latent record has inconsistent field and fields[0] values."
            )
        if metadata_field is not None and str(metadata_field) != str(fields_field):
            raise ValueError(
                "A formal patch-latent record has inconsistent metadata.field and fields[0] values."
            )
    if field is None:
        if fields_field is not None:
            field = fields_field
        else:
            field = metadata_field
    patch_id = str(record.get("patch_id") or "")
    state_ref = str(record.get("state_ref") or "")
    if patch_id and state_ref and patch_id != state_ref:
        raise ValueError("A formal patch-latent record has different patch_id and state_ref values.")
    patch_identifier = patch_id or state_ref

    has_row = "row" in record
    has_col = "col" in record
    if has_row != has_col:
        raise ValueError("A formal patch-latent record must provide row and col together.")
    row_col = (
        [_identity_integer(record["row"], "row"), _identity_integer(record["col"], "col")]
        if has_row
        else None
    )
    top_left = record.get("top_left")
    if top_left is None and row_col is not None:
        normalized_top_left = row_col
    elif top_left is not None:
        normalized_top_left = _identity_pair(top_left, "top_left")
        if row_col is not None and normalized_top_left != row_col:
            raise ValueError("A formal patch-latent record has inconsistent top_left and row/col values.")
    else:
        normalized_top_left = None
    if (
        not patch_identifier
        or field is None
        or not str(field).strip()
        or "sample_index" not in record
        or "time_index" not in record
        or normalized_top_left is None
    ):
        raise ValueError(
            "A formal patch-latent record must define patch_id/state_ref, field, sample_index, "
            "time_index, and two-element top_left metadata."
        )
    return {
        "patch_id": patch_identifier,
        "field": str(field),
        "sample_index": _identity_integer(record["sample_index"], "sample_index"),
        "time_index": _identity_integer(record["time_index"], "time_index"),
        "top_left": normalized_top_left,
    }


def latent_qa_stats_from_record(record: Mapping[str, Any]) -> dict[str, float]:
    audit = record.get("latent_audit")
    if not isinstance(audit, Mapping) or str(audit.get("format", "")) != PATCH_LATENT_AUDIT_FORMAT:
        raise ValueError(
            "A formal patch QA record is missing latent_audit provenance. Regenerate the QA JSONL with "
            "scripts/build_tensor_patch_qa.py."
        )
    result: dict[str, float] = {}
    for name in ("mean", "std", "scale"):
        try:
            value = float(audit[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"A formal patch QA record has an invalid latent_audit.{name} value.") from exc
        if not math.isfinite(value):
            raise ValueError(f"A formal patch QA record has a non-finite latent_audit.{name} value.")
        result[name] = value
    if result["std"] < 0.0 or result["scale"] <= 0.0:
        raise ValueError("A formal patch QA record requires latent_audit.std >= 0 and scale > 0.")
    return result


def validate_patch_latent_payload(
    payload: Any,
    *,
    path: str | Path,
    expected_identity: Mapping[str, Any],
    expected_alignment_checkpoint: str | Path,
    expected_alignment_sha256: str,
    expected_normalization: Mapping[str, Any],
    expected_shape: Sequence[int],
    expected_storage_dtype: str | None = None,
    expected_qa_stats: Mapping[str, float] | None = None,
) -> torch.Tensor:
    source = Path(path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Formal latent cache payload must be a mapping: {source}")
    if str(payload.get("format", "")) != PATCH_LATENT_FORMAT:
        raise ValueError(
            f"Latent cache has missing or stale format metadata at {source}; expected {PATCH_LATENT_FORMAT!r}. "
            "Regenerate patch QA latents with the current build script."
        )

    try:
        observed_identity = latent_identity_from_record(payload)
        normalized_identity = latent_identity_from_record(expected_identity)
    except ValueError as exc:
        raise ValueError(f"Latent cache identity metadata is invalid at {source}: {exc}") from exc
    if observed_identity != normalized_identity:
        raise ValueError(
            f"Latent cache identity mismatch at {source}: observed={observed_identity}, "
            f"expected={normalized_identity}."
        )

    observed_checkpoint = payload.get("alignment_checkpoint")
    if not observed_checkpoint or canonical_path(str(observed_checkpoint)) != canonical_path(
        expected_alignment_checkpoint
    ):
        raise ValueError(
            f"Latent cache alignment checkpoint path mismatch at {source}: "
            f"observed={observed_checkpoint!r}, expected={canonical_path(expected_alignment_checkpoint)!r}."
        )
    observed_sha256 = str(payload.get("alignment_checkpoint_sha256", "")).lower()
    expected_sha256 = str(expected_alignment_sha256).lower()
    if (
        not _is_sha256(expected_sha256)
        or not _is_sha256(observed_sha256)
        or observed_sha256 != expected_sha256
    ):
        raise ValueError(
            f"Latent cache alignment checkpoint SHA-256 mismatch at {source}: "
            f"observed={observed_sha256!r}, expected={expected_sha256!r}."
        )

    observed_normalization = payload.get("encoder_input_normalization")
    if not isinstance(observed_normalization, Mapping) or canonical_normalization(
        observed_normalization
    ) != canonical_normalization(expected_normalization):
        raise ValueError(
            f"Latent cache normalization mismatch at {source}: observed={observed_normalization!r}, "
            f"expected={canonical_normalization(expected_normalization)!r}."
        )

    latent = payload.get("latent_map")
    if not isinstance(latent, torch.Tensor):
        raise ValueError(f"Latent cache file does not contain a tensor latent_map: {source}")
    if latent.ndim == 4 and int(latent.shape[0]) == 1:
        latent = latent.squeeze(0)
    shape = tuple(int(value) for value in expected_shape)
    if latent.ndim != 3 or tuple(int(value) for value in latent.shape) != shape:
        raise ValueError(
            f"Latent cache shape mismatch at {source}: observed={tuple(latent.shape)}, expected={shape}."
        )
    if expected_storage_dtype is not None:
        observed_dtype = str(latent.dtype).replace("torch.", "")
        if observed_dtype != str(expected_storage_dtype):
            raise ValueError(
                f"Latent cache dtype mismatch at {source}: observed={observed_dtype}, "
                f"expected={expected_storage_dtype}."
            )
    if not bool(torch.isfinite(latent).all()):
        raise FloatingPointError(f"Latent cache contains NaN or infinity: {source}")

    if expected_qa_stats is not None:
        observed_stats = payload.get("qa_value_space")
        if not isinstance(observed_stats, Mapping) or str(observed_stats.get("mode", "")) != "per_patch_zscore":
            raise ValueError(f"Latent cache has invalid QA value-space metadata at {source}.")
        for name in ("mean", "std", "scale"):
            try:
                observed_value = float(observed_stats.get(name, float("nan")))
                expected_value = float(expected_qa_stats[name])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Latent cache QA value-space metadata has an invalid {name!r} at {source}."
                ) from exc
            if not math.isfinite(expected_value):
                raise ValueError(f"Expected latent-cache {name} provenance must be finite at {source}.")
            tolerance = max(1.0e-8, 1.0e-6 * max(1.0, abs(expected_value)))
            if not math.isfinite(observed_value) or abs(observed_value - expected_value) > tolerance:
                raise ValueError(
                    f"Latent cache {name} metadata mismatch at {source}: "
                    f"observed={observed_value}, expected={expected_value}."
                )
    return latent
