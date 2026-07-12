from __future__ import annotations

import argparse
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
    set_default(args, "include_oracle", first_nested(config, ["patch_qa.include_oracle"]), True)
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
    args.tasks = selected_tasks
    return args


def write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def patch_id(record: Mapping[str, Any]) -> str:
    field = str(record["fields"][0])
    return (
        f"{field}_s{int(record['sample_index']):06d}_t{int(record['time_index']):04d}_"
        f"r{int(record['row']):03d}_c{int(record['col']):03d}"
    )


def labeled_numeric_choices(
    value: float,
    spacing: float,
    decimals: int,
    rng: random.Random,
) -> tuple[str, list[str], str, list[float]]:
    spacing = max(abs(float(spacing)), 1.0e-8)
    correct_index = rng.randrange(4)
    offsets = [index - correct_index for index in range(4)]
    values = [float(value) + float(offset) * spacing for offset in offsets]
    formatted = [f"{item:.{int(decimals)}g}" for item in values]
    option_text = "; ".join(f"{label}: {text}" for label, text in zip(LABELS_4, formatted))
    return option_text, list(LABELS_4), LABELS_4[correct_index], values


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
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "qa_id": f"{patch_ref}_{task}",
        "patch_id": patch_ref,
        "state_ref": patch_ref,
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
        option_text, choices, answer, numeric_choices = labeled_numeric_choices(z_value, spacing, decimals, rng)
        question = (
            f"A standardized 16 by 16 patch of {field} is encoded in the tensor soft tokens. "
            f"Which option is closest to the standardized value at row {row_a}, column {col_a}? "
            f"Options: {option_text}."
        )
        oracle = {"row": row_a, "col": col_a, "normalized_value": z_value, "numeric_choices": numeric_choices}
        result.append(common_record(record, patch_ref, "normalized_point_value", question, choices, answer, oracle if include_oracle else None))

    if "raw_point_value_with_stats" in tasks:
        raw_value = float(raw_patch[0, row_a, col_a].item())
        raw_spacing = max(abs(std) * spacing, 1.0e-8)
        option_text, choices, answer, numeric_choices = labeled_numeric_choices(raw_value, raw_spacing, decimals, rng)
        question = (
            f"A 16 by 16 patch of {field} was standardized using z = (x - mean) / standard deviation. "
            f"Its mean is {mean:.{decimals}g} and its standard deviation is {std:.{decimals}g}. "
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
        )
        raw_record["prompt_data"] = {
            "field": field,
            "mean": mean,
            "std": std,
            "row": row_a,
            "col": col_a,
            "option_text": option_text,
            "significant_digits": int(decimals),
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
            f"In the standardized 16 by 16 {field} patch, which location has the larger value: "
            f"A at row {row_a}, column {col_a}, or B at row {row_b}, column {col_b}?"
        )
        oracle = {"point_a": [row_a, col_a], "point_b": [row_b, col_b], "value_a": value_a, "value_b": value_b}
        result.append(common_record(record, patch_ref, "point_compare", question, ["A", "B"], answer, oracle if include_oracle else None))

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
            f"In the standardized 16 by 16 {field} patch, compare the mean values of two {region} by {region} regions. "
            f"Region A starts at row {row0_a}, column {col0_a}; region B starts at row {row0_b}, column {col0_b}. "
            "Which region has the larger mean?"
        )
        oracle = {"region_a": [row0_a, col0_a, region], "region_b": [row0_b, col0_b, region], "mean_a": mean_a, "mean_b": mean_b}
        result.append(common_record(record, patch_ref, "region_mean_compare", question, ["A", "B"], answer, oracle if include_oracle else None))

    if "extreme_quadrant" in tasks:
        find_maximum = bool(rng.randrange(2))
        values = normalized_patch[0]
        flat_index = int(torch.argmax(values).item() if find_maximum else torch.argmin(values).item())
        row, col = divmod(flat_index, size)
        answer = quadrant(row, col, size)
        extreme = "maximum" if find_maximum else "minimum"
        question = (
            f"In the standardized 16 by 16 {field} patch, which quadrant contains the {extreme} value? "
            "The quadrants are top-left, top-right, bottom-left, and bottom-right."
        )
        choices = list(LABELS_4)
        oracle = {"extreme": extreme, "row": row, "col": col, "value": float(values[row, col].item())}
        result.append(common_record(record, patch_ref, "extreme_quadrant", question, choices, answer, oracle if include_oracle else None))
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
        patch_index = 0
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
            normalized_batch = torch.stack(normalized_items).to(device)
            with torch.no_grad():
                latent_batch = compressor.encode(normalized_batch)["latent_map"].detach().cpu()
            for local_index, record in enumerate(batch["records"]):
                ref = patch_id(record)
                latent_path = latent_root / f"{ref}.pt"
                if latent_path.exists() and not bool(args.overwrite):
                    pass
                else:
                    torch.save(
                        {
                            "latent_map": latent_batch[local_index].to(dtype=storage_dtype),
                            "patch_id": ref,
                            "field": str(record["fields"][0]),
                            "sample_index": int(record["sample_index"]),
                            "time_index": int(record["time_index"]),
                            "top_left": [int(record["row"]), int(record["col"])],
                            "normalization": {"mode": "zscore", "mean": stats[local_index][0], "std": stats[local_index][1]},
                        },
                        latent_path,
                    )
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
                    seed=int(args.seed) * 1_000_003 + patch_index,
                )
                qa_records.extend(questions)
                task_counts.update(question["task_type"] for question in questions)
                field_counts.update([str(record["fields"][0])])
                patch_index += 1
        write_jsonl(qa_dir / f"{split}.jsonl", qa_records)
        summary["splits"][split] = {
            "patches": len(records),
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
        "split_overlap": overlap,
        "summary": summary,
    }
    with (qa_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
