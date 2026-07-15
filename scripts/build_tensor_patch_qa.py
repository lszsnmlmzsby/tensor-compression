from __future__ import annotations

import argparse
import hashlib
import json
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


def write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def validate_reusable_latent_metadata(
    metadata_path: Path,
    latent_root: Path,
    alignment_checkpoint: str | Path,
    hdf5_path: str | Path,
    fields: Sequence[str],
    patch_size: int,
    overwrite: bool,
) -> None:
    if bool(overwrite) or next(latent_root.glob("*.pt"), None) is None:
        return
    if not metadata_path.exists():
        raise ValueError(
            f"Existing latent files in {latent_root} cannot be verified because {metadata_path} is missing. "
            "Use a new latent_dir or rerun with --overwrite."
        )
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, Mapping):
        raise ValueError(f"Expected a JSON object in {metadata_path}.")
    expected = {
        "alignment_checkpoint": str(Path(alignment_checkpoint).expanduser().resolve()),
        "hdf5_path": str(Path(hdf5_path).expanduser().resolve()),
        "fields": [str(field) for field in fields],
        "patch_size": int(patch_size),
    }
    observed = {
        "alignment_checkpoint": str(Path(str(metadata.get("alignment_checkpoint", ""))).expanduser().resolve()),
        "hdf5_path": str(Path(str(metadata.get("hdf5_path", ""))).expanduser().resolve()),
        "fields": [str(field) for field in metadata.get("fields", [])],
        "patch_size": int(metadata.get("patch_size", -1)),
    }
    if observed != expected:
        raise ValueError(
            "Existing patch latent metadata does not match the requested AE/data settings. "
            f"observed={observed}, expected={expected}. Use a new latent_dir or rerun with --overwrite."
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
            "representation": "per_patch_zscore",
            "grid_shape": [int(patch_record["patch_size"]), int(patch_record["patch_size"])],
            "coordinate_order": "row_col",
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
) -> list[dict[str, Any]]:
    rng = random.Random(int(seed))
    field = str(record["fields"][0])
    size = int(normalized_patch.shape[-1])
    patch_ref = patch_id(record)
    result: list[dict[str, Any]] = []
    row_a, col_a = rng.randrange(size), rng.randrange(size)
    row_b, col_b = rng.randrange(size), rng.randrange(size)
    while row_b == row_a and col_b == col_a:
        row_b, col_b = rng.randrange(size), rng.randrange(size)

    if "normalized_point_value" in tasks:
        z_value = float(normalized_patch[0, row_a, col_a].item())
        option_text, choices, answer, numeric_choices, _used_decimals = labeled_numeric_choices(
            z_value, spacing, decimals, rng
        )
        question = (
            f"A standardized {size} by {size} patch of {field} is encoded in the tensor soft tokens. "
            f"Which option is closest to the standardized value at row {row_a}, column {col_a}? "
            f"Options: {option_text}."
        )
        oracle = {"row": row_a, "col": col_a, "normalized_value": z_value, "numeric_choices": numeric_choices}
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
        raw_spacing = max(abs(std) * spacing, 1.0e-8)
        option_text, choices, answer, numeric_choices, used_decimals = labeled_numeric_choices(
            raw_value, raw_spacing, decimals, rng
        )
        question = (
            f"A {size} by {size} patch of {field} was standardized using z = (x - mean) / standard deviation. "
            f"Its mean is {mean:.{used_decimals}g} and its standard deviation is {std:.{used_decimals}g}. "
            f"The standardized patch is encoded in the tensor soft tokens. Which option is closest to the "
            f"original value x at row {row_a}, column {col_a}? Options: {option_text}."
        )
        oracle = {
            "row": row_a,
            "col": col_a,
            "mean": mean,
            "std": std,
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
            "row": row_a,
            "col": col_a,
            "option_text": option_text,
            "significant_digits": int(used_decimals),
            "option_significant_digits": int(used_decimals),
            "patch_size": size,
            "question_variant": int(variant_index),
        }
        result.append(raw_record)

    if "point_compare" in tasks:
        value_a = float(normalized_patch[0, row_a, col_a].item())
        value_b = float(normalized_patch[0, row_b, col_b].item())
        for _ in range(32):
            if abs(value_a - value_b) >= 0.5:
                break
            row_a, col_a = rng.randrange(size), rng.randrange(size)
            row_b, col_b = rng.randrange(size), rng.randrange(size)
            value_a = float(normalized_patch[0, row_a, col_a].item())
            value_b = float(normalized_patch[0, row_b, col_b].item())
        answer = "A" if value_a >= value_b else "B"
        question = (
            f"In the standardized {size} by {size} {field} patch, which location has the larger value: "
            f"A at row {row_a}, column {col_a}, or B at row {row_b}, column {col_b}?"
        )
        oracle = {"point_a": [row_a, col_a], "point_b": [row_b, col_b], "value_a": value_a, "value_b": value_b}
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
        mean_a = float(normalized_patch[0, row0_a : row0_a + region, col0_a : col0_a + region].mean().item())
        mean_b = float(normalized_patch[0, row0_b : row0_b + region, col0_b : col0_b + region].mean().item())
        for _ in range(32):
            if abs(mean_a - mean_b) >= 0.2:
                break
            row0_a, col0_a = rng.randrange(size - region + 1), rng.randrange(size - region + 1)
            row0_b, col0_b = rng.randrange(size - region + 1), rng.randrange(size - region + 1)
            mean_a = float(normalized_patch[0, row0_a : row0_a + region, col0_a : col0_a + region].mean().item())
            mean_b = float(normalized_patch[0, row0_b : row0_b + region, col0_b : col0_b + region].mean().item())
        answer = "A" if mean_a >= mean_b else "B"
        question = (
            f"In the standardized {size} by {size} {field} patch, compare the mean values of two {region} by {region} regions. "
            f"Region A starts at row {row0_a}, column {col0_a}; region B starts at row {row0_b}, column {col0_b}. "
            "Which region has the larger mean?"
        )
        oracle = {"region_a": [row0_a, col0_a, region], "region_b": [row0_b, col0_b, region], "mean_a": mean_a, "mean_b": mean_b}
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
        values = normalized_patch[0]
        flat_index = int(torch.argmax(values).item() if find_maximum else torch.argmin(values).item())
        row, col = divmod(flat_index, size)
        answer = quadrant(row, col, size)
        extreme = "maximum" if find_maximum else "minimum"
        question = (
            f"In the standardized {size} by {size} {field} patch, which quadrant contains the {extreme} value? "
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
    return result


def load_alignment_checkpoint(path: str | Path) -> tuple[dict[str, Any], dict[str, Any], torch.nn.Module]:
    checkpoint = torch.load(Path(path).expanduser(), map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Unsupported alignment checkpoint: {path}")
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
    validate_field_shapes(args.hdf5_path, fields)
    model_cfg = compressor_config.get("model", {})
    if int(model_cfg.get("in_channels", 1)) != 1:
        raise ValueError("Patch QA single-field mode requires a single-channel alignment compressor.")
    normalization_cfg = dict(compressor_config.get("data", {}).get("dataset", {}).get("normalization", {}))
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
    validate_reusable_latent_metadata(
        metadata_path=metadata_path,
        latent_root=latent_root,
        alignment_checkpoint=args.alignment_checkpoint,
        hdf5_path=args.hdf5_path,
        fields=fields,
        patch_size=int(args.patch_size),
        overwrite=bool(args.overwrite),
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
        qa_records: list[dict[str, Any]] = []
        task_counts: Counter[str] = Counter()
        field_counts: Counter[str] = Counter()
        for batch in tqdm(loader, desc=f"build patch QA {split}"):
            raw_batch = batch["patch"].float()
            normalized_items: list[torch.Tensor] = []
            stats: list[tuple[float, float]] = []
            for patch in raw_batch:
                normalized, state = normalize_tensor(patch, normalization_cfg)
                offset = state.get("offset")
                scale = state.get("scale")
                mean = float(offset.reshape(-1)[0].item()) if isinstance(offset, torch.Tensor) else 0.0
                std = float(scale.reshape(-1)[0].item()) if isinstance(scale, torch.Tensor) else 1.0
                normalized_items.append(normalized)
                stats.append((mean, std))
            refs = [patch_id(record) for record in batch["records"]]
            latent_paths = [latent_root / f"{ref}.pt" for ref in refs]
            encode_indices = [
                index
                for index, latent_path in enumerate(latent_paths)
                if bool(args.overwrite) or not latent_path.exists()
            ]
            encoded_latents: dict[int, torch.Tensor] = {}
            if encode_indices:
                normalized_batch = torch.stack([normalized_items[index] for index in encode_indices]).to(device)
                with torch.no_grad():
                    missing_latents = compressor.encode(normalized_batch)["latent_map"].detach().cpu()
                encoded_latents = {
                    item_index: missing_latents[batch_index]
                    for batch_index, item_index in enumerate(encode_indices)
                }
            for local_index, record in enumerate(batch["records"]):
                ref = refs[local_index]
                latent_path = latent_paths[local_index]
                if local_index in encoded_latents:
                    torch.save(
                        {
                            "latent_map": encoded_latents[local_index].to(dtype=storage_dtype),
                            "patch_id": ref,
                            "field": str(record["fields"][0]),
                            "sample_index": int(record["sample_index"]),
                            "time_index": int(record["time_index"]),
                            "top_left": [int(record["row"]), int(record["col"])],
                            "alignment_checkpoint": str(Path(args.alignment_checkpoint).expanduser().resolve()),
                            "normalization": {"mode": "zscore", "mean": stats[local_index][0], "std": stats[local_index][1]},
                        },
                        latent_path,
                    )
                for variant_index in range(question_variants[split]):
                    questions = build_questions(
                        record=record,
                        raw_patch=raw_batch[local_index],
                        normalized_patch=normalized_items[local_index],
                        mean=stats[local_index][0],
                        std=stats[local_index][1],
                        tasks=args.tasks,
                        region_size=int(args.region_size),
                        spacing=float(args.numeric_choice_spacing),
                        decimals=int(args.decimal_places),
                        include_oracle=bool(args.include_oracle),
                        seed=question_seed(int(args.seed), record, variant_index),
                        variant_index=variant_index,
                        variant_family_seed=question_seed(int(args.seed), record, -1),
                    )
                    qa_records.extend(questions)
                    task_counts.update(question["task_type"] for question in questions)
                field_counts.update([str(record["fields"][0])])
        write_jsonl(qa_dir / f"{split}.jsonl", qa_records)
        summary["splits"][split] = {
            "patches": len(records),
            "question_variants_per_patch": question_variants[split],
            "qa_records": len(qa_records),
            "by_task": dict(sorted(task_counts.items())),
            "patches_by_field": dict(sorted(field_counts.items())),
        }

    metadata = {
        "format": "tensor_patch_qa_v1",
        "hdf5_path": str(args.hdf5_path),
        "alignment_checkpoint": str(args.alignment_checkpoint),
        "qa_dir": str(qa_dir),
        "latent_dir": str(latent_root),
        "fields": fields,
        "patch_size": int(args.patch_size),
        "normalization": normalization_cfg,
        "split_mode": str(args.split_mode),
        "question_seed_mode": "sha256(seed|patch_id|variant)",
        "question_variants": question_variants,
        "include_oracle": bool(args.include_oracle),
        "split_overlap": overlap,
        "summary": summary,
    }
    with (qa_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
