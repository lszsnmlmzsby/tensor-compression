from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import h5py
import numpy as np
import torch
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tensor_compression.downstream.pdebench import (  # noqa: E402
    inspect_pdebench_fields,
    read_pdebench_sample,
)
from tensor_compression.utils import dump_json  # noqa: E402
from tensor_compression.utils.pipeline_config import (  # noqa: E402
    first_nested,
    load_yaml_mapping,
    require_args,
    resolve_path_string,
    set_default,
    value_to_csv,
)


DEFAULT_FIELDS = ("density", "pressure", "Vx", "Vy")
SPEED_KEY = "speed_magnitude"
MEAN_SPEED_KEY = "mean_speed"
MAX_SPEED_KEY = "max_speed"
STD_SPEED_KEY = "std_speed"
SPEED_STAT_KEYS = (MEAN_SPEED_KEY, MAX_SPEED_KEY, STD_SPEED_KEY)
BIN_PREFIX = "B"
QUADRANT_LABELS = ("Q1", "Q2", "Q3", "Q4")
COMPARISON_LABELS = ("A", "B")


@dataclass(frozen=True)
class StateRef:
    sample_index: int
    time_index: int

    @property
    def state_id(self) -> str:
        return f"sample{self.sample_index:06d}_t{self.time_index:04d}"


@dataclass(frozen=True)
class SplitIndices:
    train: list[int]
    val: list[int]
    test: list[int]

    def as_dict(self) -> dict[str, list[int]]:
        return {"train": self.train, "val": self.val, "test": self.test}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate self-supervised tensor readout QA JSONL files from a PDEBench HDF5 file. "
            "The generated questions are short DSL-style readout queries with oracle answers "
            "computed directly from the source tensor."
        )
    )
    parser.add_argument("--config", type=str, default=None, help="Optional tensor-LLM pipeline YAML config.")
    parser.add_argument("--hdf5-path", type=str, default=None, help="PDEBench HDF5 file path.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for train.jsonl, val.jsonl, test.jsonl, and metadata.json.",
    )
    parser.add_argument(
        "--fields",
        type=str,
        default=None,
        help="Comma-separated HDF5 field keys, e.g. density,pressure,Vx,Vy.",
    )
    parser.add_argument(
        "--sample-indices",
        type=str,
        default=None,
        help="Comma-separated sample indices or 'all'.",
    )
    parser.add_argument(
        "--time-indices",
        type=str,
        default=None,
        help="Comma-separated time indices or 'all'.",
    )
    parser.add_argument(
        "--max-states",
        type=int,
        default=None,
        help="Optional cap on total states after sample/time expansion.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--test-ratio", type=float, default=None)
    parser.add_argument("--spatial-stride", type=int, default=None)
    parser.add_argument("--num-bins", type=int, default=None)
    parser.add_argument(
        "--quantile-samples-per-state",
        type=int,
        default=None,
        help="Maximum random point values sampled per state for field quantile bins.",
    )
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--point-bin-per-state", type=int, default=None)
    parser.add_argument("--point-compare-per-state", type=int, default=None)
    parser.add_argument("--patch-compare-per-state", type=int, default=None)
    parser.add_argument(
        "--max-quadrant-per-state",
        type=int,
        default=None,
        help=(
            "Number of max-speed quadrant questions per state. This task is deterministic "
            "for a state, so values above 1 are capped to avoid duplicate questions."
        ),
    )
    parser.add_argument(
        "--global-stat-bin-per-state",
        type=int,
        default=None,
        help=(
            "Number of speed-stat bin questions per state. The current statistics are "
            "mean_speed, max_speed, and std_speed, so values above 3 are capped."
        ),
    )
    parser.add_argument(
        "--compare-min-bin-distance",
        type=int,
        default=None,
        help=(
            "Minimum quantile-bin distance between A and B for comparison tasks. "
            "A value of 1 rejects near-ties that fall in the same bin."
        ),
    )
    parser.add_argument(
        "--compare-max-attempts",
        type=int,
        default=None,
        help="Maximum resampling attempts for non-ambiguous comparison questions.",
    )
    parser.add_argument(
        "--latent-root",
        type=str,
        default=None,
        help=(
            "Optional latent cache root. If provided, each record gets latent_ref="
            "<latent-root>/<state_id>.pt. The script does not create these files."
        ),
    )
    parser.add_argument(
        "--include-oracle",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Include exact oracle values for debugging and evaluation.",
    )
    return apply_config_defaults(parser.parse_args())


def apply_config_defaults(args: argparse.Namespace) -> argparse.Namespace:
    config = load_yaml_mapping(args.config)
    path_defaults = {
        "hdf5_path": first_nested(config, ["data.hdf5_path"]),
        "output_dir": first_nested(config, ["qa_generation.output_dir", "data.qa_dir"]),
        "latent_root": first_nested(config, ["qa_generation.latent_root", "data.latent_dir"]),
    }
    for attr, value in path_defaults.items():
        if getattr(args, attr, None) is None and value is not None:
            setattr(args, attr, resolve_path_string(value, PROJECT_ROOT))

    set_default(args, "fields", value_to_csv(first_nested(config, ["data.fields"])), ",".join(DEFAULT_FIELDS))
    set_default(args, "sample_indices", value_to_csv(first_nested(config, ["qa_generation.sample_indices"])), "all")
    set_default(args, "time_indices", value_to_csv(first_nested(config, ["qa_generation.time_indices"])), "all")
    set_default(args, "max_states", first_nested(config, ["qa_generation.max_states"]), None)
    set_default(args, "seed", first_nested(config, ["qa_generation.seed", "runtime.seed"]), 42)
    set_default(args, "train_ratio", first_nested(config, ["qa_generation.train_ratio"]), 0.7)
    set_default(args, "val_ratio", first_nested(config, ["qa_generation.val_ratio"]), 0.15)
    set_default(args, "test_ratio", first_nested(config, ["qa_generation.test_ratio"]), 0.15)
    set_default(args, "spatial_stride", first_nested(config, ["qa_generation.spatial_stride"]), 1)
    set_default(args, "num_bins", first_nested(config, ["qa_generation.num_bins"]), 10)
    set_default(
        args,
        "quantile_samples_per_state",
        first_nested(config, ["qa_generation.quantile_samples_per_state"]),
        4096,
    )
    set_default(args, "patch_size", first_nested(config, ["qa_generation.patch_size"]), 32)
    set_default(args, "point_bin_per_state", first_nested(config, ["qa_generation.point_bin_per_state"]), 15)
    set_default(
        args,
        "point_compare_per_state",
        first_nested(config, ["qa_generation.point_compare_per_state"]),
        10,
    )
    set_default(
        args,
        "patch_compare_per_state",
        first_nested(config, ["qa_generation.patch_compare_per_state"]),
        10,
    )
    set_default(args, "max_quadrant_per_state", first_nested(config, ["qa_generation.max_quadrant_per_state"]), 1)
    set_default(args, "global_stat_bin_per_state", first_nested(config, ["qa_generation.global_stat_bin_per_state"]), 3)
    set_default(
        args,
        "compare_min_bin_distance",
        first_nested(config, ["qa_generation.compare_min_bin_distance"]),
        1,
    )
    set_default(args, "compare_max_attempts", first_nested(config, ["qa_generation.compare_max_attempts"]), 32)
    set_default(args, "include_oracle", first_nested(config, ["qa_generation.include_oracle"]), True)
    require_args(args, ["hdf5_path", "output_dir"])
    return args


def parse_csv(raw: str | Sequence[str] | None) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [part.strip() for part in raw.split(",") if part.strip()]
    return [str(part).strip() for part in raw if str(part).strip()]


def resolve_sample_indices(raw: str, total_samples: int) -> list[int]:
    raw = str(raw).strip()
    if raw.lower() == "all":
        return list(range(total_samples))
    indices = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not indices:
        raise ValueError("--sample-indices must contain at least one index or 'all'.")
    for index in indices:
        if index < 0 or index >= total_samples:
            raise IndexError(f"Sample index {index} is outside [0, {total_samples}).")
    return indices


def resolve_time_indices(raw: str, total_steps: int) -> list[int]:
    raw = str(raw).strip()
    if raw.lower() == "all":
        return list(range(total_steps))
    indices = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not indices:
        raise ValueError("--time-indices must contain at least one index or 'all'.")
    for index in indices:
        if index < 0 or index >= total_steps:
            raise IndexError(f"Time index {index} is outside [0, {total_steps}).")
    return indices


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    ratios = [float(train_ratio), float(val_ratio), float(test_ratio)]
    if any(ratio < 0.0 for ratio in ratios):
        raise ValueError("Split ratios must be non-negative.")
    total = sum(ratios)
    if not math.isclose(total, 1.0, rel_tol=1.0e-6, abs_tol=1.0e-6):
        raise ValueError(f"Split ratios must sum to 1.0, got {total}.")
    if train_ratio <= 0.0:
        raise ValueError("train_ratio must be positive.")


def split_sample_indices(
    sample_indices: Sequence[int],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> SplitIndices:
    validate_ratios(train_ratio, val_ratio, test_ratio)
    if not sample_indices:
        raise ValueError("At least one sample index is required.")

    rng = np.random.default_rng(seed)
    shuffled = list(int(index) for index in sample_indices)
    rng.shuffle(shuffled)

    total = len(shuffled)
    train_count = max(1, int(round(total * train_ratio)))
    val_count = int(round(total * val_ratio))
    if train_count + val_count > total:
        val_count = max(0, total - train_count)
    test_count = total - train_count - val_count
    if total >= 3 and val_ratio > 0.0 and val_count == 0:
        val_count = 1
        if test_count > 0:
            test_count -= 1
        else:
            train_count -= 1
    if total >= 3 and test_ratio > 0.0 and test_count == 0:
        test_count = 1
        if train_count > 1:
            train_count -= 1
        else:
            val_count = max(0, val_count - 1)

    train = sorted(shuffled[:train_count])
    val = sorted(shuffled[train_count : train_count + val_count])
    test = sorted(shuffled[train_count + val_count :])
    return SplitIndices(train=train, val=val, test=test)


def build_states_for_split(
    sample_indices: Sequence[int],
    time_indices: Sequence[int],
    max_states: int | None = None,
    seed: int = 42,
) -> list[StateRef]:
    states = [
        StateRef(sample_index=int(sample_index), time_index=int(time_index))
        for sample_index in sample_indices
        for time_index in time_indices
    ]
    if max_states is not None and len(states) > max_states:
        rng = np.random.default_rng(seed)
        selected = rng.choice(len(states), size=int(max_states), replace=False)
        states = [states[int(index)] for index in selected]
    return sorted(states, key=lambda state: (state.sample_index, state.time_index))


def apply_global_state_cap(
    split_states: dict[str, list[StateRef]],
    max_states: int | None,
    seed: int,
) -> dict[str, list[StateRef]]:
    if max_states is None:
        return split_states
    total = sum(len(states) for states in split_states.values())
    if total <= max_states:
        return split_states
    if max_states <= 0:
        raise ValueError("--max-states must be positive when provided.")

    rng = np.random.default_rng(seed)
    all_items: list[tuple[str, StateRef]] = []
    for split_name, states in split_states.items():
        all_items.extend((split_name, state) for state in states)
    selected_indices = set(int(index) for index in rng.choice(len(all_items), size=max_states, replace=False))
    capped: dict[str, list[StateRef]] = {"train": [], "val": [], "test": []}
    for index, (split_name, state) in enumerate(all_items):
        if index in selected_indices:
            capped[split_name].append(state)
    for split_name in capped:
        capped[split_name].sort(key=lambda state: (state.sample_index, state.time_index))
    if not capped["train"]:
        raise ValueError("--max-states removed all train states; use a larger value.")
    return capped


def label_for_bin(index: int) -> str:
    return f"{BIN_PREFIX}{index:02d}"


def bin_labels(num_bins: int) -> list[str]:
    return [label_for_bin(index) for index in range(num_bins)]


def compute_bin_edges(values: Sequence[float], num_bins: int) -> list[float]:
    if num_bins < 2:
        raise ValueError("num_bins must be at least 2.")
    if not values:
        raise ValueError("Cannot compute bin edges from an empty value list.")
    percentiles = np.linspace(0.0, 100.0, num_bins + 1, dtype=np.float64)[1:-1]
    return [float(value) for value in np.percentile(np.asarray(values, dtype=np.float64), percentiles)]


def assign_bin(value: float, edges: Sequence[float]) -> str:
    bin_index = int(np.searchsorted(np.asarray(edges, dtype=np.float64), float(value), side="right"))
    return label_for_bin(bin_index)


def load_sample_frames(
    hdf5_path: str | Path,
    field_keys: Sequence[str],
    sample_index: int,
    spatial_stride: int = 1,
) -> torch.Tensor:
    data, _grid, _t_coordinates = read_pdebench_sample(
        hdf5_path=hdf5_path,
        field_keys=field_keys,
        sample_index=sample_index,
        spatial_stride=spatial_stride,
    )
    return data


def collect_quantile_values(
    hdf5_path: str | Path,
    field_keys: Sequence[str],
    train_states: Sequence[StateRef],
    spatial_stride: int,
    quantile_samples_per_state: int,
    seed: int,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {field: [] for field in field_keys}
    if has_velocity_fields(field_keys):
        for speed_key in SPEED_STAT_KEYS:
            values[speed_key] = []

    states_by_sample: dict[int, list[StateRef]] = {}
    for state in train_states:
        states_by_sample.setdefault(state.sample_index, []).append(state)

    for sample_index, states in tqdm(
        sorted(states_by_sample.items()),
        desc="Collect quantile values",
        unit="sample",
    ):
        data = load_sample_frames(hdf5_path, field_keys, sample_index, spatial_stride)
        height, width, _time_steps, _channels = data.shape
        max_points = height * width
        point_count = min(max(1, int(quantile_samples_per_state)), max_points)
        for state in states:
            frame = data[:, :, state.time_index, :]
            flat_indices = rng.choice(max_points, size=point_count, replace=False)
            rows = flat_indices // width
            cols = flat_indices % width
            for channel_index, field in enumerate(field_keys):
                sampled = frame[rows, cols, channel_index].detach().cpu().numpy()
                values[field].extend(float(value) for value in sampled)
            if has_velocity_fields(field_keys):
                speed = compute_speed(frame, field_keys)
                values[MEAN_SPEED_KEY].append(float(speed.mean().item()))
                values[MAX_SPEED_KEY].append(float(speed.max().item()))
                values[STD_SPEED_KEY].append(float(speed.std(unbiased=False).item()))
    return values


def compute_quantile_bins(
    hdf5_path: str | Path,
    field_keys: Sequence[str],
    train_states: Sequence[StateRef],
    spatial_stride: int,
    num_bins: int,
    quantile_samples_per_state: int,
    seed: int,
) -> dict[str, list[float]]:
    values = collect_quantile_values(
        hdf5_path=hdf5_path,
        field_keys=field_keys,
        train_states=train_states,
        spatial_stride=spatial_stride,
        quantile_samples_per_state=quantile_samples_per_state,
        seed=seed,
    )
    return {name: compute_bin_edges(name_values, num_bins) for name, name_values in values.items()}


def has_velocity_fields(field_keys: Sequence[str]) -> bool:
    return "Vx" in field_keys and "Vy" in field_keys


def field_index(field_keys: Sequence[str], field: str) -> int:
    try:
        return list(field_keys).index(field)
    except ValueError as exc:
        raise KeyError(f"Field {field!r} is not in selected field keys: {field_keys}") from exc


def compute_speed(frame: torch.Tensor, field_keys: Sequence[str]) -> torch.Tensor:
    vx = frame[:, :, field_index(field_keys, "Vx")]
    vy = frame[:, :, field_index(field_keys, "Vy")]
    return torch.sqrt(torch.square(vx) + torch.square(vy))


def random_point(rng: np.random.Generator, height: int, width: int) -> tuple[int, int]:
    return int(rng.integers(0, height)), int(rng.integers(0, width))


def random_patch(
    rng: np.random.Generator,
    height: int,
    width: int,
    patch_size: int,
) -> tuple[int, int, int, int]:
    size = max(1, min(int(patch_size), height, width))
    row0 = int(rng.integers(0, height - size + 1))
    col0 = int(rng.integers(0, width - size + 1))
    return row0, col0, row0 + size, col0 + size


def compare_values(value_a: float, value_b: float, eps: float = 1.0e-8) -> str:
    delta = float(value_a) - float(value_b)
    if abs(delta) <= eps:
        return "A"
    return "A" if delta > 0.0 else "B"


def bin_index_for_value(value: float, edges: Sequence[float]) -> int:
    return int(np.searchsorted(np.asarray(edges, dtype=np.float64), float(value), side="right"))


def values_are_well_separated(
    value_a: float,
    value_b: float,
    edges: Sequence[float],
    min_bin_distance: int,
) -> bool:
    return abs(bin_index_for_value(value_a, edges) - bin_index_for_value(value_b, edges)) >= int(min_bin_distance)


def quadrant_for_position(row: int, col: int, height: int, width: int) -> str:
    top = row < height / 2.0
    left = col < width / 2.0
    if top and left:
        return "Q1"
    if top and not left:
        return "Q2"
    if not top and left:
        return "Q3"
    return "Q4"


def format_choices(choices: Sequence[str]) -> str:
    return ",".join(choices)


def latent_ref_for_state(latent_root: str | None, state: StateRef) -> str | None:
    if not latent_root:
        return None
    return str((Path(latent_root) / f"{state.state_id}.pt").as_posix())


def build_record(
    qa_id: str,
    state: StateRef,
    task_type: str,
    query: str,
    choices: Sequence[str],
    answer: str,
    field_keys: Sequence[str],
    height: int,
    width: int,
    latent_root: str | None,
    oracle: dict[str, Any] | None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "qa_id": qa_id,
        "sample_id": f"sample{state.sample_index:06d}",
        "sample_index": state.sample_index,
        "time_index": state.time_index,
        "state_ref": state.state_id,
        "task_type": task_type,
        "question": query,
        "query": query,
        "choices": list(choices),
        "answer": answer,
        "metadata": {
            "dataset": "PDEBench",
            "fields": list(field_keys),
            "grid_shape": [height, width],
            "coordinate_order": "row_col",
        },
    }
    latent_ref = latent_ref_for_state(latent_root, state)
    if latent_ref is not None:
        record["latent_ref"] = latent_ref
    if oracle is not None:
        record["oracle"] = oracle
    return record


def make_point_bin_record(
    frame: torch.Tensor,
    state: StateRef,
    field_keys: Sequence[str],
    bin_edges: dict[str, list[float]],
    num_bins: int,
    rng: np.random.Generator,
    qa_id: str,
    latent_root: str | None,
    include_oracle: bool,
) -> dict[str, Any]:
    height, width, _channels = frame.shape
    field = str(rng.choice(field_keys))
    row, col = random_point(rng, height, width)
    value = float(frame[row, col, field_index(field_keys, field)].item())
    answer = assign_bin(value, bin_edges[field])
    choices = bin_labels(num_bins)
    query = (
        f"VALUE_BIN field={field} time={state.time_index} row={row} col={col} "
        f"choices={format_choices(choices)}"
    )
    oracle = {"field": field, "row": row, "col": col, "value": value} if include_oracle else None
    return build_record(
        qa_id=qa_id,
        state=state,
        task_type="point_bin",
        query=query,
        choices=choices,
        answer=answer,
        field_keys=field_keys,
        height=height,
        width=width,
        latent_root=latent_root,
        oracle=oracle,
    )


def make_point_compare_record(
    frame: torch.Tensor,
    state: StateRef,
    field_keys: Sequence[str],
    bin_edges: dict[str, list[float]],
    rng: np.random.Generator,
    qa_id: str,
    latent_root: str | None,
    include_oracle: bool,
    min_bin_distance: int,
    max_attempts: int,
) -> dict[str, Any]:
    height, width, _channels = frame.shape
    field = str(rng.choice(field_keys))
    row_a, col_a = random_point(rng, height, width)
    row_b, col_b = random_point(rng, height, width)
    value_a = float(frame[row_a, col_a, field_index(field_keys, field)].item())
    value_b = float(frame[row_b, col_b, field_index(field_keys, field)].item())
    for _attempt in range(max(1, int(max_attempts))):
        candidate_field = str(rng.choice(field_keys))
        candidate_row_a, candidate_col_a = random_point(rng, height, width)
        candidate_row_b, candidate_col_b = random_point(rng, height, width)
        candidate_channel = field_index(field_keys, candidate_field)
        candidate_value_a = float(frame[candidate_row_a, candidate_col_a, candidate_channel].item())
        candidate_value_b = float(frame[candidate_row_b, candidate_col_b, candidate_channel].item())
        if values_are_well_separated(
            candidate_value_a,
            candidate_value_b,
            bin_edges[candidate_field],
            min_bin_distance,
        ):
            field = candidate_field
            row_a, col_a = candidate_row_a, candidate_col_a
            row_b, col_b = candidate_row_b, candidate_col_b
            value_a, value_b = candidate_value_a, candidate_value_b
            break
    answer = compare_values(value_a, value_b)
    choices = list(COMPARISON_LABELS)
    query = (
        f"COMPARE_POINT field={field} time={state.time_index} "
        f"A=({row_a},{col_a}) B=({row_b},{col_b}) choices={format_choices(choices)}"
    )
    oracle = (
        {
            "field": field,
            "point_a": [row_a, col_a],
            "point_b": [row_b, col_b],
            "value_a": value_a,
            "value_b": value_b,
        }
        if include_oracle
        else None
    )
    return build_record(
        qa_id=qa_id,
        state=state,
        task_type="point_compare",
        query=query,
        choices=choices,
        answer=answer,
        field_keys=field_keys,
        height=height,
        width=width,
        latent_root=latent_root,
        oracle=oracle,
    )


def make_patch_compare_record(
    frame: torch.Tensor,
    state: StateRef,
    field_keys: Sequence[str],
    bin_edges: dict[str, list[float]],
    patch_size: int,
    rng: np.random.Generator,
    qa_id: str,
    latent_root: str | None,
    include_oracle: bool,
    min_bin_distance: int,
    max_attempts: int,
) -> dict[str, Any]:
    height, width, _channels = frame.shape
    field = str(rng.choice(field_keys))
    patch_a = random_patch(rng, height, width, patch_size)
    patch_b = random_patch(rng, height, width, patch_size)
    channel = field_index(field_keys, field)
    value_a = float(frame[patch_a[0] : patch_a[2], patch_a[1] : patch_a[3], channel].mean().item())
    value_b = float(frame[patch_b[0] : patch_b[2], patch_b[1] : patch_b[3], channel].mean().item())
    for _attempt in range(max(1, int(max_attempts))):
        candidate_field = str(rng.choice(field_keys))
        candidate_patch_a = random_patch(rng, height, width, patch_size)
        candidate_patch_b = random_patch(rng, height, width, patch_size)
        candidate_channel = field_index(field_keys, candidate_field)
        candidate_value_a = float(
            frame[
                candidate_patch_a[0] : candidate_patch_a[2],
                candidate_patch_a[1] : candidate_patch_a[3],
                candidate_channel,
            ]
            .mean()
            .item()
        )
        candidate_value_b = float(
            frame[
                candidate_patch_b[0] : candidate_patch_b[2],
                candidate_patch_b[1] : candidate_patch_b[3],
                candidate_channel,
            ]
            .mean()
            .item()
        )
        if values_are_well_separated(
            candidate_value_a,
            candidate_value_b,
            bin_edges[candidate_field],
            min_bin_distance,
        ):
            field = candidate_field
            patch_a, patch_b = candidate_patch_a, candidate_patch_b
            value_a, value_b = candidate_value_a, candidate_value_b
            break
    answer = compare_values(value_a, value_b)
    choices = list(COMPARISON_LABELS)
    query = (
        f"COMPARE_PATCH_MEAN field={field} time={state.time_index} "
        f"A=[{patch_a[0]}:{patch_a[2]},{patch_a[1]}:{patch_a[3]}] "
        f"B=[{patch_b[0]}:{patch_b[2]},{patch_b[1]}:{patch_b[3]}] "
        f"choices={format_choices(choices)}"
    )
    oracle = (
        {
            "field": field,
            "patch_a": list(patch_a),
            "patch_b": list(patch_b),
            "mean_a": value_a,
            "mean_b": value_b,
        }
        if include_oracle
        else None
    )
    return build_record(
        qa_id=qa_id,
        state=state,
        task_type="patch_compare",
        query=query,
        choices=choices,
        answer=answer,
        field_keys=field_keys,
        height=height,
        width=width,
        latent_root=latent_root,
        oracle=oracle,
    )


def make_max_quadrant_record(
    frame: torch.Tensor,
    state: StateRef,
    field_keys: Sequence[str],
    qa_id: str,
    latent_root: str | None,
    include_oracle: bool,
) -> dict[str, Any]:
    height, width, _channels = frame.shape
    speed = compute_speed(frame, field_keys)
    flat_index = int(torch.argmax(speed).item())
    row = flat_index // width
    col = flat_index % width
    answer = quadrant_for_position(row, col, height, width)
    choices = list(QUADRANT_LABELS)
    query = (
        f"MAX_SPEED_QUADRANT time={state.time_index} "
        f"quadrants=Q1:top_left,Q2:top_right,Q3:bottom_left,Q4:bottom_right "
        f"choices={format_choices(choices)}"
    )
    oracle = (
        {
            "row": row,
            "col": col,
            "max_speed": float(speed[row, col].item()),
        }
        if include_oracle
        else None
    )
    return build_record(
        qa_id=qa_id,
        state=state,
        task_type="max_speed_quadrant",
        query=query,
        choices=choices,
        answer=answer,
        field_keys=field_keys,
        height=height,
        width=width,
        latent_root=latent_root,
        oracle=oracle,
    )


def make_global_stat_bin_record(
    frame: torch.Tensor,
    state: StateRef,
    field_keys: Sequence[str],
    bin_edges: dict[str, list[float]],
    num_bins: int,
    statistic: str,
    qa_id: str,
    latent_root: str | None,
    include_oracle: bool,
) -> dict[str, Any]:
    height, width, _channels = frame.shape
    speed = compute_speed(frame, field_keys)
    if statistic == MEAN_SPEED_KEY:
        value = float(speed.mean().item())
        query_name = "MEAN_SPEED_BIN"
    elif statistic == MAX_SPEED_KEY:
        value = float(speed.max().item())
        query_name = "MAX_SPEED_BIN"
    elif statistic == STD_SPEED_KEY:
        value = float(speed.std(unbiased=False).item())
        query_name = "STD_SPEED_BIN"
    else:
        raise ValueError(f"Unsupported speed statistic: {statistic}")
    answer = assign_bin(value, bin_edges[statistic])
    choices = bin_labels(num_bins)
    query = (
        f"{query_name} time={state.time_index} "
        f"choices={format_choices(choices)}"
    )
    oracle = {"statistic": statistic, "value": value} if include_oracle else None
    return build_record(
        qa_id=qa_id,
        state=state,
        task_type="global_stat_bin",
        query=query,
        choices=choices,
        answer=answer,
        field_keys=field_keys,
        height=height,
        width=width,
        latent_root=latent_root,
        oracle=oracle,
    )


def generate_records_for_state(
    frame: torch.Tensor,
    state: StateRef,
    field_keys: Sequence[str],
    bin_edges: dict[str, list[float]],
    num_bins: int,
    patch_size: int,
    counts: dict[str, int],
    rng: np.random.Generator,
    latent_root: str | None,
    include_oracle: bool,
    compare_min_bin_distance: int,
    compare_max_attempts: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    local_index = 0

    def next_qa_id(task_type: str) -> str:
        nonlocal local_index
        qa_id = f"{state.state_id}_{task_type}_{local_index:04d}"
        local_index += 1
        return qa_id

    for _ in range(max(0, int(counts.get("point_bin", 0)))):
        records.append(
            make_point_bin_record(
                frame=frame,
                state=state,
                field_keys=field_keys,
                bin_edges=bin_edges,
                num_bins=num_bins,
                rng=rng,
                qa_id=next_qa_id("point_bin"),
                latent_root=latent_root,
                include_oracle=include_oracle,
            )
        )
    for _ in range(max(0, int(counts.get("point_compare", 0)))):
        records.append(
            make_point_compare_record(
                frame=frame,
                state=state,
                field_keys=field_keys,
                bin_edges=bin_edges,
                rng=rng,
                qa_id=next_qa_id("point_compare"),
                latent_root=latent_root,
                include_oracle=include_oracle,
                min_bin_distance=compare_min_bin_distance,
                max_attempts=compare_max_attempts,
            )
        )
    for _ in range(max(0, int(counts.get("patch_compare", 0)))):
        records.append(
            make_patch_compare_record(
                frame=frame,
                state=state,
                field_keys=field_keys,
                bin_edges=bin_edges,
                patch_size=patch_size,
                rng=rng,
                qa_id=next_qa_id("patch_compare"),
                latent_root=latent_root,
                include_oracle=include_oracle,
                min_bin_distance=compare_min_bin_distance,
                max_attempts=compare_max_attempts,
            )
        )
    if has_velocity_fields(field_keys):
        for _ in range(min(1, max(0, int(counts.get("max_speed_quadrant", 0))))):
            records.append(
                make_max_quadrant_record(
                    frame=frame,
                    state=state,
                    field_keys=field_keys,
                    qa_id=next_qa_id("max_speed_quadrant"),
                    latent_root=latent_root,
                    include_oracle=include_oracle,
                )
            )
        global_stat_count = min(len(SPEED_STAT_KEYS), max(0, int(counts.get("global_stat_bin", 0))))
        for statistic in SPEED_STAT_KEYS[:global_stat_count]:
            records.append(
                make_global_stat_bin_record(
                    frame=frame,
                    state=state,
                    field_keys=field_keys,
                    bin_edges=bin_edges,
                    num_bins=num_bins,
                    statistic=statistic,
                    qa_id=next_qa_id("global_stat_bin"),
                    latent_root=latent_root,
                    include_oracle=include_oracle,
                )
            )
    return records


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def generate_split_records(
    hdf5_path: str | Path,
    field_keys: Sequence[str],
    states: Sequence[StateRef],
    bin_edges: dict[str, list[float]],
    num_bins: int,
    patch_size: int,
    counts: dict[str, int],
    spatial_stride: int,
    seed: int,
    latent_root: str | None,
    include_oracle: bool,
    compare_min_bin_distance: int = 1,
    compare_max_attempts: int = 32,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    states_by_sample: dict[int, list[StateRef]] = {}
    for state in states:
        states_by_sample.setdefault(state.sample_index, []).append(state)

    for sample_index, sample_states in tqdm(
        sorted(states_by_sample.items()),
        desc="Generate QA",
        unit="sample",
    ):
        data = load_sample_frames(hdf5_path, field_keys, sample_index, spatial_stride)
        for state in sample_states:
            frame = data[:, :, state.time_index, :]
            records.extend(
                generate_records_for_state(
                    frame=frame,
                    state=state,
                    field_keys=field_keys,
                    bin_edges=bin_edges,
                    num_bins=num_bins,
                    patch_size=patch_size,
                    counts=counts,
                    rng=rng,
                    latent_root=latent_root,
                    include_oracle=include_oracle,
                    compare_min_bin_distance=compare_min_bin_distance,
                    compare_max_attempts=compare_max_attempts,
                )
            )
    return records


def first_field_shape(hdf5_path: str | Path, field_key: str) -> tuple[int, ...]:
    with h5py.File(hdf5_path, "r") as handle:
        if field_key not in handle or not isinstance(handle[field_key], h5py.Dataset):
            raise KeyError(f"HDF5 dataset key {field_key!r} not found in {hdf5_path}.")
        return tuple(int(dim) for dim in handle[field_key].shape)


def build_counts(args: argparse.Namespace) -> dict[str, int]:
    return {
        "point_bin": int(args.point_bin_per_state),
        "point_compare": int(args.point_compare_per_state),
        "patch_compare": int(args.patch_compare_per_state),
        "max_speed_quadrant": min(1, int(args.max_quadrant_per_state)),
        "global_stat_bin": min(len(SPEED_STAT_KEYS), int(args.global_stat_bin_per_state)),
    }


def count_by_task(records: Sequence[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(record["task_type"]) for record in records))


def count_answers_by_task(records: Sequence[dict[str, Any]]) -> dict[str, dict[str, int]]:
    grouped: dict[str, Counter] = {}
    for record in records:
        task_type = str(record["task_type"])
        grouped.setdefault(task_type, Counter()).update([str(record["answer"])])
    return {
        task_type: dict(sorted(counter.items()))
        for task_type, counter in sorted(grouped.items())
    }


def main() -> None:
    args = parse_args()
    hdf5_path = Path(args.hdf5_path).expanduser()
    output_dir = Path(args.output_dir)
    field_keys = parse_csv(args.fields)
    if not field_keys:
        raise ValueError("--fields must select at least one field.")
    inspect_pdebench_fields(hdf5_path, field_keys=field_keys)

    shape = first_field_shape(hdf5_path, field_keys[0])
    if len(shape) < 4:
        raise ValueError(
            f"Expected PDEBench 2D fields shaped [sample, time, height, width], got {shape}."
        )
    total_samples, total_steps = int(shape[0]), int(shape[1])
    sample_indices = resolve_sample_indices(args.sample_indices, total_samples)
    time_indices = resolve_time_indices(args.time_indices, total_steps)
    split_indices = split_sample_indices(
        sample_indices=sample_indices,
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
        seed=int(args.seed),
    )
    split_states = {
        split_name: build_states_for_split(indices, time_indices)
        for split_name, indices in split_indices.as_dict().items()
    }
    split_states = apply_global_state_cap(split_states, args.max_states, int(args.seed))
    if not split_states["train"]:
        raise RuntimeError("No train states available for quantile computation.")

    counts = build_counts(args)
    if not has_velocity_fields(field_keys):
        counts["max_speed_quadrant"] = 0
        counts["global_stat_bin"] = 0

    bin_edges = compute_quantile_bins(
        hdf5_path=hdf5_path,
        field_keys=field_keys,
        train_states=split_states["train"],
        spatial_stride=int(args.spatial_stride),
        num_bins=int(args.num_bins),
        quantile_samples_per_state=int(args.quantile_samples_per_state),
        seed=int(args.seed),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    split_counts: dict[str, int] = {}
    split_task_counts: dict[str, dict[str, int]] = {}
    split_answer_counts: dict[str, dict[str, dict[str, int]]] = {}
    for offset, split_name in enumerate(("train", "val", "test")):
        records = generate_split_records(
            hdf5_path=hdf5_path,
            field_keys=field_keys,
            states=split_states[split_name],
            bin_edges=bin_edges,
            num_bins=int(args.num_bins),
            patch_size=int(args.patch_size),
            counts=counts,
            spatial_stride=int(args.spatial_stride),
            seed=int(args.seed) + offset * 1009,
            latent_root=args.latent_root,
            include_oracle=bool(args.include_oracle),
            compare_min_bin_distance=int(args.compare_min_bin_distance),
            compare_max_attempts=int(args.compare_max_attempts),
        )
        split_counts[split_name] = write_jsonl(output_dir / f"{split_name}.jsonl", records)
        split_task_counts[split_name] = count_by_task(records)
        split_answer_counts[split_name] = count_answers_by_task(records)

    metadata = {
        "source": {
            "hdf5_path": str(hdf5_path),
            "fields": field_keys,
            "source_shape": list(shape),
            "spatial_stride": int(args.spatial_stride),
        },
        "splits": {
            "sample_indices": split_indices.as_dict(),
            "state_counts": {name: len(states) for name, states in split_states.items()},
            "qa_counts": split_counts,
            "task_counts": split_task_counts,
            "answer_counts": split_answer_counts,
        },
        "generation": {
            "seed": int(args.seed),
            "time_indices": time_indices,
            "num_bins": int(args.num_bins),
            "bin_labels": bin_labels(int(args.num_bins)),
            "patch_size": int(args.patch_size),
            "counts_per_state": counts,
            "compare_min_bin_distance": int(args.compare_min_bin_distance),
            "compare_max_attempts": int(args.compare_max_attempts),
            "quantile_samples_per_state": int(args.quantile_samples_per_state),
            "latent_root": args.latent_root,
            "include_oracle": bool(args.include_oracle),
        },
        "bin_edges": bin_edges,
    }
    dump_json(output_dir / "metadata.json", metadata)
    print(json.dumps({"output_dir": str(output_dir), "qa_counts": split_counts}, indent=2))


if __name__ == "__main__":
    main()
