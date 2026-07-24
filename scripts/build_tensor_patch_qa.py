from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.train_tensor_patch_text_alignment import (  # noqa: E402
    PDEBenchPatchTextDataset,
    build_axis_split_plan,
    build_patch_records,
    collate_patch_text,
    parse_csv,
    split_overlap_summary,
    validate_field_shapes,
)
from tensor_compression.data.normalization import normalize_tensor  # noqa: E402
from tensor_compression.downstream.patch_qa_contract import (  # noqa: E402
    PATCH_LATENT_FORMAT,
    PATCH_LATENT_AUDIT_FORMAT,
    PATCH_QA_BUILD_MARKER,
    PATCH_QA_FORMAT,
    PATCH_QA_PROMPT_CONTRACT,
    atomic_json_dump,
    atomic_torch_save,
    canonical_normalization,
    canonical_path,
    latent_identity_from_record,
    sha256_file,
    validate_stage1_alignment_checkpoint_payload,
    validate_patch_latent_payload,
)
from tensor_compression.downstream.pdebench import resolve_device  # noqa: E402
from tensor_compression.models import build_model  # noqa: E402
from tensor_compression.utils.pipeline_config import (  # noqa: E402
    first_nested,
    load_yaml_mapping,
    resolve_path_string,
    set_default,
    value_to_csv,
)


TASKS = (
    "normalized_point_value",
    "raw_point_value_with_stats",
    "point_compare",
    "region_mean_compare",
    "extreme_quadrant",
)
LABELS_4 = ("A", "B", "C", "D")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build fixed-natural-language QA and latent caches from normalized 16x16 PDEBench patches."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--alignment-checkpoint", type=str, default=None)
    parser.add_argument("--hdf5-path", type=str, default=None)
    parser.add_argument("--qa-dir", type=str, default=None)
    parser.add_argument("--latent-dir", type=str, default=None)
    parser.add_argument("--fields", type=str, default=None)
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--train-patches", type=int, default=None)
    parser.add_argument("--val-patches", type=int, default=None)
    parser.add_argument("--test-patches", type=int, default=None)
    parser.add_argument("--train-question-variants", type=int, default=None)
    parser.add_argument("--val-question-variants", type=int, default=None)
    parser.add_argument("--test-question-variants", type=int, default=None)
    parser.add_argument("--sample-indices", type=str, default=None)
    parser.add_argument("--time-indices", type=str, default=None)
    parser.add_argument("--split-mode", choices=("sample", "time", "sample_time", "random_record"), default=None)
    parser.add_argument("--tasks", type=str, default=None)
    parser.add_argument("--region-size", type=int, default=None)
    parser.add_argument("--numeric-choice-spacing", type=float, default=None)
    parser.add_argument("--decimal-places", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--storage-dtype", choices=("float16", "float32"), default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--include-oracle", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--allow-unseen-alignment-fields",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Allow QA fields that were absent from stage-1 alignment (cross-field transfer only).",
    )
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    config = load_yaml_mapping(args.config)
    path_defaults = {
        "alignment_checkpoint": first_nested(config, ["patch_qa.alignment_checkpoint"]),
        "hdf5_path": first_nested(config, ["patch_qa.hdf5_path", "patch_alignment.hdf5_path", "data.hdf5_path"]),
        "qa_dir": first_nested(config, ["patch_qa.qa_dir"]),
        "latent_dir": first_nested(config, ["patch_qa.latent_dir"]),
    }
    for name, value in path_defaults.items():
        if getattr(args, name) is None and value is not None:
            setattr(args, name, resolve_path_string(value, PROJECT_ROOT))
    set_default(args, "fields", value_to_csv(first_nested(config, ["patch_qa.fields", "patch_alignment.fields"])), None)
    set_default(args, "patch_size", first_nested(config, ["patch_qa.patch_size", "patch_alignment.patch_size"]), 16)
    set_default(args, "train_patches", first_nested(config, ["patch_qa.train_patches"]), 65536)
    set_default(args, "val_patches", first_nested(config, ["patch_qa.val_patches"]), 8192)
    set_default(args, "test_patches", first_nested(config, ["patch_qa.test_patches"]), 8192)
    set_default(
        args,
        "train_question_variants",
        first_nested(config, ["patch_qa.train_question_variants"]),
        1,
    )
    set_default(
        args,
        "val_question_variants",
        first_nested(config, ["patch_qa.val_question_variants"]),
        1,
    )
    set_default(
        args,
        "test_question_variants",
        first_nested(config, ["patch_qa.test_question_variants"]),
        1,
    )
    set_default(args, "sample_indices", value_to_csv(first_nested(config, ["patch_qa.sample_indices", "patch_alignment.sample_indices"])), "all")
    set_default(args, "time_indices", value_to_csv(first_nested(config, ["patch_qa.time_indices", "patch_alignment.time_indices"])), "all")
    set_default(args, "split_mode", first_nested(config, ["patch_qa.split_mode", "patch_alignment.split_mode"]), "sample")
    set_default(args, "tasks", value_to_csv(first_nested(config, ["patch_qa.tasks"])), ",".join(TASKS))
    set_default(args, "region_size", first_nested(config, ["patch_qa.region_size"]), 4)
    set_default(args, "numeric_choice_spacing", first_nested(config, ["patch_qa.numeric_choice_spacing"]), 0.5)
    set_default(args, "decimal_places", first_nested(config, ["patch_qa.decimal_places"]), 3)
    set_default(args, "batch_size", first_nested(config, ["patch_qa.batch_size"]), 256)
    set_default(args, "device", first_nested(config, ["patch_qa.device", "runtime.device"]), "auto")
    set_default(args, "storage_dtype", first_nested(config, ["patch_qa.storage_dtype"]), "float16")
    set_default(args, "seed", first_nested(config, ["patch_qa.seed", "runtime.seed"]), 42)
    set_default(args, "include_oracle", first_nested(config, ["patch_qa.include_oracle"]), False)
    set_default(
        args,
        "allow_unseen_alignment_fields",
        first_nested(config, ["patch_qa.allow_unseen_alignment_fields"]),
        False,
    )
    set_default(args, "overwrite", first_nested(config, ["patch_qa.overwrite"]), False)
    for name in ("alignment_checkpoint", "hdf5_path", "qa_dir", "latent_dir", "fields"):
        if not getattr(args, name):
            raise ValueError(f"Missing required patch QA setting: {name}")
    selected_tasks = parse_csv(args.tasks)
    unknown = sorted(set(selected_tasks) - set(TASKS))
    if unknown:
        raise ValueError(f"Unsupported patch QA tasks: {unknown}. Supported: {TASKS}")
    if not selected_tasks:
        raise ValueError("At least one patch QA task is required.")
    for name in (
        "train_patches",
        "val_patches",
        "test_patches",
        "train_question_variants",
        "val_question_variants",
        "test_question_variants",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"patch_qa.{name} must be positive.")
    args.tasks = selected_tasks
    return args


class AtomicJsonlWriter:
    """Stream JSONL to a temporary file and publish it only after a clean build."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        self.handle: Any = None

    def __enter__(self) -> "AtomicJsonlWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.temporary.open("w", encoding="utf-8")
        return self

    def write(self, record: Mapping[str, Any]) -> None:
        if self.handle is None:
            raise RuntimeError("AtomicJsonlWriter must be entered before writing.")
        self.handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    def write_many(self, records: Sequence[Mapping[str, Any]]) -> None:
        for record in records:
            self.write(record)

    def __exit__(self, exc_type, exc, traceback) -> None:
        try:
            if self.handle is not None:
                self.handle.close()
                self.handle = None
            if exc_type is None:
                os.replace(self.temporary, self.path)
        finally:
            self.temporary.unlink(missing_ok=True)


def write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    with AtomicJsonlWriter(path) as writer:
        writer.write_many(records)


def validate_reusable_latent_metadata(
    metadata_path: Path,
    latent_root: Path,
    alignment_checkpoint: str | Path,
    hdf5_path: str | Path,
    fields: Sequence[str],
    patch_size: int,
    alignment_checkpoint_sha256: str,
    encoder_input_normalization: Mapping[str, Any],
    latent_shape: Sequence[int],
    storage_dtype: str,
    overwrite: bool,
) -> None:
    if bool(overwrite) or next(latent_root.glob("*.pt"), None) is None:
        return
    if not metadata_path.exists():
        # A previous interrupted build may have completed atomic latent files
        # before writing the final directory manifest. Each referenced payload is
        # validated below, so a trustworthy partial build can resume safely.
        return
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, Mapping):
        raise ValueError(f"Expected a JSON object in {metadata_path}.")
    expected = {
        "alignment_checkpoint": str(Path(alignment_checkpoint).expanduser().resolve()),
        "hdf5_path": str(Path(hdf5_path).expanduser().resolve()),
        "fields": [str(field) for field in fields],
        "patch_size": int(patch_size),
        "alignment_checkpoint_sha256": str(alignment_checkpoint_sha256).lower(),
        "encoder_input_normalization": canonical_normalization(encoder_input_normalization),
        "latent_format": PATCH_LATENT_FORMAT,
        "latent_shape": [int(value) for value in latent_shape],
        "storage_dtype": str(storage_dtype),
    }
    observed = {
        "alignment_checkpoint": str(Path(str(metadata.get("alignment_checkpoint", ""))).expanduser().resolve()),
        "hdf5_path": str(Path(str(metadata.get("hdf5_path", ""))).expanduser().resolve()),
        "fields": [str(field) for field in metadata.get("fields", [])],
        "patch_size": int(metadata.get("patch_size", -1)),
        "alignment_checkpoint_sha256": str(metadata.get("alignment_checkpoint_sha256", "")).lower(),
        "encoder_input_normalization": canonical_normalization(
            metadata.get("encoder_input_normalization", {})
            if isinstance(metadata.get("encoder_input_normalization"), Mapping)
            else {}
        ),
        "latent_format": str(metadata.get("latent_format", "")),
        "latent_shape": [int(value) for value in metadata.get("latent_shape", [])],
        "storage_dtype": str(metadata.get("storage_dtype", "")),
    }
    if observed != expected:
        raise ValueError(
            "Existing patch latent metadata does not match the requested AE/data settings. "
            f"observed={observed}, expected={expected}. Use a new latent_dir or rerun with --overwrite."
        )


def validate_alignment_field_coverage(
    checkpoint_args: Mapping[str, Any],
    qa_fields: Sequence[str],
    *,
    allow_unseen: bool,
) -> list[str]:
    alignment_fields = parse_csv(checkpoint_args.get("fields"))
    if not alignment_fields:
        if bool(allow_unseen):
            return []
        raise ValueError(
            "The stage-1 checkpoint does not record its alignment fields, so downstream field coverage cannot "
            "be verified. Use a checkpoint with args.fields provenance or explicitly set "
            "patch_qa.allow_unseen_alignment_fields=true for a cross-field transfer experiment."
        )
    unseen = sorted(set(str(field) for field in qa_fields) - set(alignment_fields))
    if unseen and not bool(allow_unseen):
        raise ValueError(
            "Patch QA requests fields that were absent from stage-1 alignment: "
            f"unseen={unseen}, alignment_fields={alignment_fields}. Train stage 1 on all requested fields, "
            "restrict patch_qa.fields, or explicitly set patch_qa.allow_unseen_alignment_fields=true to label "
            "this as a cross-field transfer experiment."
        )
    return alignment_fields


def validate_patch_qa_encoder_normalization(normalization_cfg: Mapping[str, Any]) -> None:
    mode = str(normalization_cfg.get("mode", "none")).lower()
    scope = str(normalization_cfg.get("scope", "global")).lower()
    clip_min = normalization_cfg.get("clip_min")
    clip_max = normalization_cfg.get("clip_max")
    if mode != "zscore" or scope != "channel" or clip_min is not None or clip_max is not None:
        raise ValueError(
            "The current patch QA prompts require the stage-1 encoder input to be unclipped per-patch z-score "
            "data (normalization mode=zscore, scope=channel, clip_min/clip_max=null). "
            f"Got mode={mode!r}, scope={scope!r}, clip_min={clip_min!r}, clip_max={clip_max!r}."
        )


def patch_id(record: Mapping[str, Any]) -> str:
    field = str(record["fields"][0])
    return (
        f"{field}_s{int(record['sample_index']):06d}_t{int(record['time_index']):04d}_"
        f"r{int(record['row']):03d}_c{int(record['col']):03d}"
    )


def question_seed(base_seed: int, record: Mapping[str, Any], variant_index: int = 0) -> int:
    payload = f"{int(base_seed)}|{patch_id(record)}|variant={int(variant_index)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="big", signed=False)


def labeled_numeric_choices(
    value: float,
    spacing: float,
    decimals: int,
    rng: random.Random,
) -> tuple[str, list[str], str, list[float], int]:
    spacing = max(abs(float(spacing)), 1.0e-8)
    correct_index = rng.randrange(4)
    offsets = [index - correct_index for index in range(4)]
    values = [float(value) + float(offset) * spacing for offset in offsets]
    if len(set(values)) != len(values):
        raise ValueError(f"Numeric choice spacing produced duplicate floating-point values around {value}.")
    used_decimals = max(1, int(decimals))
    formatted = [f"{item:.{used_decimals}g}" for item in values]
    while len(set(formatted)) != len(formatted) and used_decimals < 17:
        used_decimals += 1
        formatted = [f"{item:.{used_decimals}g}" for item in values]
    if len(set(formatted)) != len(formatted):
        raise ValueError(f"Numeric options remain ambiguous after 17 significant digits around {value}.")
    options = list(zip(values, formatted, [index == correct_index for index in range(4)]))
    rng.shuffle(options)
    shuffled_values = [item[0] for item in options]
    shuffled_formatted = [item[1] for item in options]
    shuffled_correct_index = next(index for index, item in enumerate(options) if item[2])
    option_text = "; ".join(f"{label}: {text}" for label, text in zip(LABELS_4, shuffled_formatted))
    return option_text, list(LABELS_4), LABELS_4[shuffled_correct_index], shuffled_values, used_decimals


def quadrant(row: int, col: int, size: int) -> str:
    top = int(row) < int(size) / 2.0
    left = int(col) < int(size) / 2.0
    if top and left:
        return "A"
    if top:
        return "B"
    if left:
        return "C"
    return "D"


def per_patch_zscore(
    patch: torch.Tensor,
    eps: float = 1.0e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    patch_float = patch.float()
    mean = patch_float.mean()
    std = patch_float.std(unbiased=False)
    # Match normalize_tensor(mode=zscore) exactly so QA targets describe the cached encoder input.
    scale = std + float(eps)
    z_patch = (patch_float - mean) / scale
    return z_patch, {
        "mean": float(mean.item()),
        "std": float(std.item()),
        "scale": float(scale.item()),
    }


def common_record(
    patch_record: Mapping[str, Any],
    patch_ref: str,
    task: str,
    question: str,
    choices: Sequence[str],
    answer: str,
    oracle: Mapping[str, Any] | None,
    variant_index: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "qa_id": f"{patch_ref}_{task}_v{int(variant_index):02d}",
        "patch_id": patch_ref,
        "state_ref": patch_ref,
        "question_variant": int(variant_index),
        "sample_index": int(patch_record["sample_index"]),
        "time_index": int(patch_record["time_index"]),
        "field": str(patch_record["fields"][0]),
        "top_left": [int(patch_record["row"]), int(patch_record["col"])],
        "task_type": task,
        "question": question,
        "query": question,
        "choices": list(choices),
        "answer": str(answer),
        "metadata": {
            "dataset": "PDEBench",
            "prompt_contract": PATCH_QA_PROMPT_CONTRACT,
            "tensor_encoding": "alignment_checkpoint_encoder_input",
            "qa_value_space": "per_patch_zscore",
            "grid_shape": [int(patch_record["patch_size"]), int(patch_record["patch_size"])],
            "coordinate_order": "row_col",
            "coordinate_origin": 1,
            "oracle_coordinate_origin": 0,
            "field": str(patch_record["fields"][0]),
        },
    }
    if oracle is not None:
        payload["oracle"] = dict(oracle)
    return payload


def build_questions(
    record: Mapping[str, Any],
    raw_patch: torch.Tensor,
    normalized_patch: torch.Tensor,
    mean: float,
    std: float,
    tasks: Sequence[str],
    region_size: int,
    spacing: float,
    decimals: int,
    include_oracle: bool,
    seed: int,
    variant_index: int = 0,
    variant_family_seed: int | None = None,
    scale: float | None = None,
) -> list[dict[str, Any]]:
    rng = random.Random(int(seed))
    field = str(record["fields"][0])
    size = int(normalized_patch.shape[-1])
    standardization_scale = float(std if scale is None else scale)
    patch_ref = patch_id(record)
    result: list[dict[str, Any]] = []
    row_a, col_a = rng.randrange(size), rng.randrange(size)
    row_b, col_b = rng.randrange(size), rng.randrange(size)
    while row_b == row_a and col_b == col_a:
        row_b, col_b = rng.randrange(size), rng.randrange(size)

    def display_coordinate(index: int) -> int:
        return int(index) + 1

    if "normalized_point_value" in tasks:
        z_value = float(normalized_patch[0, row_a, col_a].item())
        option_text, choices, answer, numeric_choices, _used_decimals = labeled_numeric_choices(
            z_value, spacing, decimals, rng
        )
        question = (
            f"The tensor soft tokens encode the per-patch standardized {size} by {size} matrix z of {field}. "
            f"The standardization is z = (x - mean) / scale, "
            f"where mean is {mean:.{decimals}g} and scale is {standardization_scale:.{decimals}g}. "
            f"Which option is closest to z at row {display_coordinate(row_a)}, "
            f"column {display_coordinate(col_a)}? "
            f"Options: {option_text}."
        )
        oracle = {
            "row": row_a,
            "col": col_a,
            "mean": mean,
            "std": std,
            "scale": standardization_scale,
            "raw_value": float(raw_patch[0, row_a, col_a].item()),
            "normalized_value": z_value,
            "numeric_choices": numeric_choices,
        }
        result.append(
            common_record(
                record,
                patch_ref,
                "normalized_point_value",
                question,
                choices,
                answer,
                oracle if include_oracle else None,
                variant_index,
            )
        )

    if "raw_point_value_with_stats" in tasks:
        raw_value = float(raw_patch[0, row_a, col_a].item())
        raw_spacing = max(abs(standardization_scale) * spacing, 1.0e-8)
        option_text, choices, answer, numeric_choices, used_decimals = labeled_numeric_choices(
            raw_value, raw_spacing, decimals, rng
        )
        question = (
            f"The tensor soft tokens encode the per-patch standardized {size} by {size} matrix z of {field}. "
            f"Recover an original value with x = mean + scale * z, "
            f"where mean is {mean:.{used_decimals}g} and scale is {standardization_scale:.{used_decimals}g}. "
            f"Which option is closest to the "
            f"original value x at row {display_coordinate(row_a)}, "
            f"column {display_coordinate(col_a)}? Options: {option_text}."
        )
        oracle = {
            "row": row_a,
            "col": col_a,
            "mean": mean,
            "std": std,
            "scale": standardization_scale,
            "normalized_value": float(normalized_patch[0, row_a, col_a].item()),
            "raw_value": raw_value,
            "numeric_choices": numeric_choices,
        }
        raw_record = common_record(
            record,
            patch_ref,
            "raw_point_value_with_stats",
            question,
            choices,
            answer,
            oracle if include_oracle else None,
            variant_index,
        )
        raw_record["prompt_data"] = {
            "field": field,
            "mean": mean,
            "std": std,
            "scale": standardization_scale,
            "row": display_coordinate(row_a),
            "col": display_coordinate(col_a),
            "option_text": option_text,
            "significant_digits": int(used_decimals),
            "option_significant_digits": int(used_decimals),
            "patch_size": size,
            "question_variant": int(variant_index),
        }
        result.append(raw_record)

    if "point_compare" in tasks:
        z_value_a = float(normalized_patch[0, row_a, col_a].item())
        z_value_b = float(normalized_patch[0, row_b, col_b].item())
        for _ in range(32):
            if abs(z_value_a - z_value_b) >= 0.5:
                break
            row_a, col_a = rng.randrange(size), rng.randrange(size)
            row_b, col_b = rng.randrange(size), rng.randrange(size)
            z_value_a = float(normalized_patch[0, row_a, col_a].item())
            z_value_b = float(normalized_patch[0, row_b, col_b].item())
        raw_value_a = float(raw_patch[0, row_a, col_a].item())
        raw_value_b = float(raw_patch[0, row_b, col_b].item())
        answer = "A" if raw_value_a >= raw_value_b else "B"
        question = (
            f"The tensor soft tokens encode the per-patch standardized {size} by {size} matrix of {field}; "
            "standardization preserves value order. Which location has the larger value: "
            f"A at row {display_coordinate(row_a)}, column {display_coordinate(col_a)}, or B at row "
            f"{display_coordinate(row_b)}, column {display_coordinate(col_b)}?"
        )
        oracle = {
            "point_a": [row_a, col_a],
            "point_b": [row_b, col_b],
            "value_a": raw_value_a,
            "value_b": raw_value_b,
            "z_value_a": z_value_a,
            "z_value_b": z_value_b,
        }
        result.append(
            common_record(
                record,
                patch_ref,
                "point_compare",
                question,
                ["A", "B"],
                answer,
                oracle if include_oracle else None,
                variant_index,
            )
        )

    if "region_mean_compare" in tasks:
        region = max(1, min(int(region_size), size))
        row0_a, col0_a = rng.randrange(size - region + 1), rng.randrange(size - region + 1)
        row0_b, col0_b = rng.randrange(size - region + 1), rng.randrange(size - region + 1)
        z_mean_a = float(normalized_patch[0, row0_a : row0_a + region, col0_a : col0_a + region].mean().item())
        z_mean_b = float(normalized_patch[0, row0_b : row0_b + region, col0_b : col0_b + region].mean().item())
        for _ in range(32):
            if abs(z_mean_a - z_mean_b) >= 0.2:
                break
            row0_a, col0_a = rng.randrange(size - region + 1), rng.randrange(size - region + 1)
            row0_b, col0_b = rng.randrange(size - region + 1), rng.randrange(size - region + 1)
            z_mean_a = float(normalized_patch[0, row0_a : row0_a + region, col0_a : col0_a + region].mean().item())
            z_mean_b = float(normalized_patch[0, row0_b : row0_b + region, col0_b : col0_b + region].mean().item())
        mean_a = float(raw_patch[0, row0_a : row0_a + region, col0_a : col0_a + region].mean().item())
        mean_b = float(raw_patch[0, row0_b : row0_b + region, col0_b : col0_b + region].mean().item())
        answer = "A" if mean_a >= mean_b else "B"
        question = (
            f"The tensor soft tokens encode the per-patch standardized {size} by {size} matrix of {field}; "
            "standardization preserves the ordering of region means. "
            f"Compare the mean values of two {region} by {region} regions. "
            f"Region A starts at row {display_coordinate(row0_a)}, column {display_coordinate(col0_a)}; "
            f"region B starts at row {display_coordinate(row0_b)}, column {display_coordinate(col0_b)}. "
            "Which region has the larger mean?"
        )
        oracle = {
            "region_a": [row0_a, col0_a, region],
            "region_b": [row0_b, col0_b, region],
            "mean_a": mean_a,
            "mean_b": mean_b,
            "z_mean_a": z_mean_a,
            "z_mean_b": z_mean_b,
        }
        result.append(
            common_record(
                record,
                patch_ref,
                "region_mean_compare",
                question,
                ["A", "B"],
                answer,
                oracle if include_oracle else None,
                variant_index,
            )
        )

    if "extreme_quadrant" in tasks:
        # Alternate this binary operation within a patch while keeping variant 0 balanced across patches.
        family_seed = int(seed if variant_family_seed is None else variant_family_seed)
        find_maximum = bool((family_seed + int(variant_index)) % 2)
        values = raw_patch[0]
        flat_index = int(torch.argmax(values).item() if find_maximum else torch.argmin(values).item())
        row, col = divmod(flat_index, size)
        answer = quadrant(row, col, size)
        extreme = "maximum" if find_maximum else "minimum"
        question = (
            f"The tensor soft tokens encode the per-patch standardized {size} by {size} matrix of {field}; "
            "standardization preserves extrema and their locations. "
            f"Which quadrant contains the {extreme} value? "
            "The quadrants are top-left, top-right, bottom-left, and bottom-right."
        )
        choices = list(LABELS_4)
        oracle = {"extreme": extreme, "row": row, "col": col, "value": float(values[row, col].item())}
        result.append(
            common_record(
                record,
                patch_ref,
                "extreme_quadrant",
                question,
                choices,
                answer,
                oracle if include_oracle else None,
                variant_index,
            )
        )
    latent_audit = {
        "format": PATCH_LATENT_AUDIT_FORMAT,
        "mean": float(mean),
        "std": float(std),
        "scale": float(standardization_scale),
    }
    for question_record in result:
        # This provenance is validated against the latent payload but is never
        # rendered by the natural-language prompt builder.
        question_record["latent_audit"] = dict(latent_audit)
    return result


def load_alignment_checkpoint(path: str | Path) -> tuple[dict[str, Any], dict[str, Any], torch.nn.Module]:
    checkpoint_path = Path(path).expanduser()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    validate_stage1_alignment_checkpoint_payload(checkpoint, path=checkpoint_path)
    config = checkpoint.get("compressor_config")
    state = checkpoint.get("compressor_state_dict")
    if not isinstance(config, Mapping) or not isinstance(state, Mapping):
        raise ValueError("Alignment checkpoint must contain compressor_config and compressor_state_dict.")
    model = build_model(dict(config))
    model.load_state_dict(state)
    return dict(checkpoint), dict(config), model


def main() -> None:
    args = parse_args()
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    checkpoint, compressor_config, compressor = load_alignment_checkpoint(args.alignment_checkpoint)
    checkpoint_args = checkpoint.get("args", {}) if isinstance(checkpoint.get("args"), Mapping) else {}
    if int(args.patch_size) != int(checkpoint_args.get("patch_size", args.patch_size)):
        raise ValueError("patch_qa.patch_size must match the alignment checkpoint patch_size.")
    fields = parse_csv(args.fields)
    alignment_fields = validate_alignment_field_coverage(
        checkpoint_args,
        fields,
        allow_unseen=bool(args.allow_unseen_alignment_fields),
    )
    validate_field_shapes(args.hdf5_path, fields)
    model_cfg = compressor_config.get("model", {})
    if int(model_cfg.get("in_channels", 1)) != 1:
        raise ValueError("Patch QA single-field mode requires a single-channel alignment compressor.")
    normalization_cfg = dict(compressor_config.get("data", {}).get("dataset", {}).get("normalization", {}))
    validate_patch_qa_encoder_normalization(normalization_cfg)
    alignment_checkpoint_path = canonical_path(args.alignment_checkpoint)
    print("startup=alignment_checkpoint_sha256", flush=True)
    alignment_checkpoint_sha256 = sha256_file(alignment_checkpoint_path)
    # The compressor now owns the copied encoder weights. Release the duplicate
    # checkpoint payload before iterating over the full QA dataset.
    checkpoint_args = dict(checkpoint_args)
    checkpoint = None
    gc.collect()
    device = resolve_device(args.device)
    compressor.to(device).eval()
    for parameter in compressor.parameters():
        parameter.requires_grad_(False)

    split_plan = build_axis_split_plan(
        hdf5_path=args.hdf5_path,
        field=fields[0],
        sample_indices=args.sample_indices,
        time_indices=args.time_indices,
        split_mode=args.split_mode,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=int(args.seed),
    )
    counts = {"train": int(args.train_patches), "val": int(args.val_patches), "test": int(args.test_patches)}
    question_variants = {
        "train": int(args.train_question_variants),
        "val": int(args.val_question_variants),
        "test": int(args.test_question_variants),
    }
    split_records = {
        split: build_patch_records(
            hdf5_path=args.hdf5_path,
            field=fields[0],
            record_fields=fields,
            sample_indices=split_plan["samples"][split],
            time_indices=split_plan["times"][split],
            patch_size=int(args.patch_size),
            count=count,
            seed=int(args.seed) + {"train": 101, "val": 102, "test": 103}[split],
            unique_records=True,
        )
        for split, count in counts.items()
    }
    overlap = split_overlap_summary(split_records["train"], split_records["val"], split_records["test"])
    if any(overlap.values()):
        raise ValueError(f"Patch QA split overlap detected: {overlap}")

    qa_dir = Path(args.qa_dir).expanduser()
    latent_root = Path(args.latent_dir).expanduser()
    qa_dir.mkdir(parents=True, exist_ok=True)
    latent_root.mkdir(parents=True, exist_ok=True)
    metadata_path = qa_dir / "metadata.json"
    with torch.no_grad():
        latent_probe = compressor.encode(
            torch.zeros(
                1,
                int(model_cfg.get("in_channels", 1)),
                int(args.patch_size),
                int(args.patch_size),
                device=device,
            )
        )["latent_map"]
    if latent_probe.ndim != 4 or int(latent_probe.shape[0]) != 1:
        raise ValueError(
            "Alignment compressor must emit latent_map [B,C,H,W], got "
            f"{tuple(latent_probe.shape)}."
        )
    latent_shape = tuple(int(value) for value in latent_probe.shape[1:])
    del latent_probe
    storage_dtype_name = str(args.storage_dtype)
    validate_reusable_latent_metadata(
        metadata_path=metadata_path,
        latent_root=latent_root,
        alignment_checkpoint=args.alignment_checkpoint,
        hdf5_path=args.hdf5_path,
        fields=fields,
        patch_size=int(args.patch_size),
        alignment_checkpoint_sha256=alignment_checkpoint_sha256,
        encoder_input_normalization=normalization_cfg,
        latent_shape=latent_shape,
        storage_dtype=storage_dtype_name,
        overwrite=bool(args.overwrite),
    )
    build_marker = qa_dir / PATCH_QA_BUILD_MARKER
    atomic_json_dump(
        build_marker,
        {
            "status": "building",
            "format": PATCH_QA_FORMAT,
            "alignment_checkpoint": alignment_checkpoint_path,
            "alignment_checkpoint_sha256": alignment_checkpoint_sha256,
        },
    )
    storage_dtype = torch.float16 if args.storage_dtype == "float16" else torch.float32
    summary: dict[str, Any] = {"splits": {}, "tasks": list(args.tasks)}

    for split, records in split_records.items():
        dataset = PDEBenchPatchTextDataset(
            hdf5_path=args.hdf5_path,
            field_keys=fields,
            records=records,
            patch_size=int(args.patch_size),
            decimal_places=int(args.decimal_places),
            prompt_template="compact",
            include_raw_text=False,
        )
        loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False, collate_fn=collate_patch_text)
        qa_record_count = 0
        task_counts: Counter[str] = Counter()
        field_counts: Counter[str] = Counter()
        with AtomicJsonlWriter(qa_dir / f"{split}.jsonl") as qa_writer:
            for batch in tqdm(loader, desc=f"build patch QA {split}"):
                raw_batch = batch["patch"].float()
                encoder_input_items: list[torch.Tensor] = []
                qa_patch_items: list[torch.Tensor] = []
                stats: list[dict[str, float]] = []
                for patch in raw_batch:
                    encoder_input, _state = normalize_tensor(patch, normalization_cfg)
                    qa_patch, qa_stats = per_patch_zscore(patch)
                    if not torch.allclose(encoder_input.float(), qa_patch.float(), rtol=1.0e-5, atol=1.0e-6):
                        raise ValueError(
                            "Patch QA value space differs from the alignment encoder input despite the normalization "
                            "preflight. Refusing to generate semantically inconsistent prompts and latents."
                        )
                    encoder_input_items.append(encoder_input)
                    qa_patch_items.append(qa_patch)
                    stats.append(qa_stats)
                refs = [patch_id(record) for record in batch["records"]]
                latent_paths = [latent_root / f"{ref}.pt" for ref in refs]
                encode_indices: list[int] = []
                for local_index, (record, latent_path) in enumerate(zip(batch["records"], latent_paths, strict=True)):
                    if bool(args.overwrite) or not latent_path.exists():
                        encode_indices.append(local_index)
                        continue
                    try:
                        existing_payload = torch.load(latent_path, map_location="cpu", weights_only=True)
                        validate_patch_latent_payload(
                            existing_payload,
                            path=latent_path,
                            expected_identity=latent_identity_from_record({**record, "patch_id": refs[local_index]}),
                            expected_alignment_checkpoint=alignment_checkpoint_path,
                            expected_alignment_sha256=alignment_checkpoint_sha256,
                            expected_normalization=normalization_cfg,
                            expected_shape=latent_shape,
                            expected_storage_dtype=storage_dtype_name,
                            expected_qa_stats=stats[local_index],
                        )
                    except Exception as exc:
                        raise ValueError(
                            f"Existing latent cache failed strict provenance validation: {latent_path}. "
                            "Use a new latent_dir or rerun with --overwrite."
                        ) from exc
                    finally:
                        existing_payload = None
                encoded_latents: dict[int, torch.Tensor] = {}
                if encode_indices:
                    encoder_input_batch = torch.stack([encoder_input_items[index] for index in encode_indices]).to(device)
                    with torch.no_grad():
                        missing_latents = compressor.encode(encoder_input_batch)["latent_map"].detach().cpu()
                    encoded_latents = {
                        item_index: missing_latents[batch_index]
                        for batch_index, item_index in enumerate(encode_indices)
                    }
                for local_index, record in enumerate(batch["records"]):
                    ref = refs[local_index]
                    latent_path = latent_paths[local_index]
                    if local_index in encoded_latents:
                        atomic_torch_save(
                            latent_path,
                            {
                                "format": PATCH_LATENT_FORMAT,
                                "latent_map": encoded_latents[local_index].to(dtype=storage_dtype),
                                "patch_id": ref,
                                "field": str(record["fields"][0]),
                                "sample_index": int(record["sample_index"]),
                                "time_index": int(record["time_index"]),
                                "top_left": [int(record["row"]), int(record["col"])],
                                "alignment_checkpoint": alignment_checkpoint_path,
                                "alignment_checkpoint_sha256": alignment_checkpoint_sha256,
                                "encoder_input_normalization": normalization_cfg,
                                "qa_value_space": {
                                    "mode": "per_patch_zscore",
                                    "mean": stats[local_index]["mean"],
                                    "std": stats[local_index]["std"],
                                    "scale": stats[local_index]["scale"],
                                },
                            },
                        )
                    for variant_index in range(question_variants[split]):
                        questions = build_questions(
                            record=record,
                            raw_patch=raw_batch[local_index],
                            normalized_patch=qa_patch_items[local_index],
                            mean=stats[local_index]["mean"],
                            std=stats[local_index]["std"],
                            scale=stats[local_index]["scale"],
                            tasks=args.tasks,
                            region_size=int(args.region_size),
                            spacing=float(args.numeric_choice_spacing),
                            decimals=int(args.decimal_places),
                            include_oracle=bool(args.include_oracle),
                            seed=question_seed(int(args.seed), record, variant_index),
                            variant_index=variant_index,
                            variant_family_seed=question_seed(int(args.seed), record, -1),
                        )
                        qa_writer.write_many(questions)
                        qa_record_count += len(questions)
                        task_counts.update(question["task_type"] for question in questions)
                    field_counts.update([str(record["fields"][0])])
        summary["splits"][split] = {
            "patches": len(records),
            "question_variants_per_patch": question_variants[split],
            "qa_records": qa_record_count,
            "by_task": dict(sorted(task_counts.items())),
            "patches_by_field": dict(sorted(field_counts.items())),
        }

    metadata = {
        "format": PATCH_QA_FORMAT,
        "prompt_contract": PATCH_QA_PROMPT_CONTRACT,
        "hdf5_path": str(args.hdf5_path),
        "alignment_checkpoint": alignment_checkpoint_path,
        "alignment_checkpoint_sha256": alignment_checkpoint_sha256,
        "alignment_fields": alignment_fields,
        "allow_unseen_alignment_fields": bool(args.allow_unseen_alignment_fields),
        "qa_dir": str(qa_dir),
        "latent_dir": str(latent_root),
        "fields": fields,
        "patch_size": int(args.patch_size),
        "latent_format": PATCH_LATENT_FORMAT,
        "latent_audit_format": PATCH_LATENT_AUDIT_FORMAT,
        "latent_shape": list(latent_shape),
        "storage_dtype": storage_dtype_name,
        "encoder_input_normalization": normalization_cfg,
        "qa_value_space": "per_patch_zscore_from_raw_patch",
        "natural_language_coordinate_origin": 1,
        "oracle_coordinate_origin": 0,
        "split_mode": str(args.split_mode),
        "question_seed_mode": "sha256(seed|patch_id|variant)",
        "question_variants": question_variants,
        "include_oracle": bool(args.include_oracle),
        "split_overlap": overlap,
        "summary": summary,
    }
    atomic_json_dump(qa_dir / "metadata.json", metadata)
    build_marker.unlink(missing_ok=True)
    compressor_config = {}
    gc.collect()
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
