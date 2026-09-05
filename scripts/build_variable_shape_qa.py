"""Build checkpoint-free, trajectory-disjoint HxW QA directly from PDEBench."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import h5py
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.build_tensor_patch_matched_qa import (
    base_generated_record, build_state_records, evaluation_record_replay,
    point_pair, region_pair, rendered_options,
)
from tensor_compression.downstream.patch_qa_contract import PATCH_LATENT_AUDIT_FORMAT, sha256_file
from tensor_compression.downstream.variable_shape import (
    VARIABLE_PROMPT_CONTRACT, VARIABLE_QA_FORMAT, experiment_config,
    parse_shapes, rectangular_quadrant, shape_name,
)
from tensor_compression.utils.pipeline_config import environment_override, load_yaml_mapping, resolve_path_string


def normalized_values(raw: torch.Tensor):
    raw = raw.float()
    if not bool(torch.isfinite(raw).all()):
        raise ValueError("nonfinite_raw")
    mean, std = raw.mean(), raw.std(unbiased=False)
    scale = std + 1e-6
    z = ((raw - mean) / scale).half().float()
    if not bool(torch.isfinite(z).all()) or float(z.max()) == float(z.min()):
        raise ValueError("constant_or_nonfinite_normalized")
    stats = {"format": PATCH_LATENT_AUDIT_FORMAT, "mean": float(mean), "std": float(std), "scale": float(scale)}
    return z, stats


def extreme_record(source, z, rng):
    operation = rng.choice(["maximum", "minimum"])
    positions = torch.nonzero(z == (z.max() if operation == "maximum" else z.min())).tolist()
    quadrants = {rectangular_quadrant(row, col, *z.shape) for row, col in positions}
    if len(quadrants) != 1:
        raise ValueError("ambiguous_cross_quadrant_extreme")
    query = (f"The field is a {z.shape[0]} by {z.shape[1]} matrix of {source['field']}. "
             f"Which quadrant contains the {operation} value? "
             "The quadrants are top-left, top-right, bottom-left, and bottom-right.")
    return base_generated_record(source, source["state_ref"] + "_extreme", "extreme_quadrant",
                                 query, list("ABCD"), next(iter(quadrants)), 0)


def evaluation_records(source, z, rng, *, numeric_gap, region_gap, region_size, digits):
    """Five questions per state; labels are computed in the stored FP16 value space."""
    height, width = z.shape
    intro = f"The field is a {height} by {width} standardized matrix of {source['field']}. "
    row, col = rng.randrange(height), rng.randrange(width)
    target = float(z[row, col])
    target_slot = rng.randrange(4)
    options = [target + (index - target_slot) * numeric_gap for index in range(4)]
    records = []
    for raw in (False, True):
        permutation = list(range(4))
        rng.shuffle(permutation)
        stats = source["latent_audit"]
        text, choices, answers, _digits, mean, scale = rendered_options(
            options, [target], permutation, digits,
            mean=stats["mean"] if raw else None, scale=stats["scale"] if raw else None,
        )
        task = "raw_point_value_with_stats" if raw else "normalized_point_value"
        query = intro
        if raw:
            query += f"Recover x = mean + scale * z, where mean is {mean} and scale is {scale}. "
        query += (f"Which option is closest to {'the original value x' if raw else 'z'} "
                  f"at row {row + 1}, column {col + 1}? Options: {text}.")
        records.append(base_generated_record(source, source["state_ref"] + "_" + task,
                                              task, query, choices, answers[0], 0))
    first, second = point_pair(z, numeric_gap, rng)
    query = (intro + f"Which location has the larger value: A at row {first[0] + 1}, "
             f"column {first[1] + 1}, or B at row {second[0] + 1}, column {second[1] + 1}?")
    records.append(base_generated_record(source, source["state_ref"] + "_compare", "point_compare",
                                          query, ["A", "B"], "A" if z[first] > z[second] else "B", 0))
    first, second, mean_a, mean_b = region_pair(z, region_size, region_gap, rng)
    query = (intro + f"Compare two {region_size} by {region_size} regions. "
             f"Region A starts at row {first[0] + 1}, column {first[1] + 1}; "
             f"region B starts at row {second[0] + 1}, column {second[1] + 1}. "
             "Which region has the larger mean?")
    records.append(base_generated_record(source, source["state_ref"] + "_region", "region_mean_compare",
                                          query, ["A", "B"], "A" if mean_a > mean_b else "B", 0))
    records.append(extreme_record(source, z, rng))
    return records


def build_dataset(config, output: Path, hdf5_path: Path):
    data, generation = config["data"], config["generation"]
    train_shapes = parse_shapes(data["train_shapes"])
    heldout = parse_shapes(data["heldout_shapes"])
    extrapolation = parse_shapes(data["extrapolation_shapes"])
    all_shapes = train_shapes + heldout + extrapolation
    if len(set(all_shapes)) != len(all_shapes):
        raise ValueError("Train, held-out and extrapolation shape sets must be disjoint.")
    if int(generation["region_size"]) > min(min(shape) for shape in all_shapes):
        raise ValueError("A configured grid is smaller than the region size.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite dataset {output}; use a new directory.")
    output.mkdir(parents=True, exist_ok=True)
    marker = output / ".build_in_progress.json"
    marker.write_text(json.dumps({"profile": config["experiment_profile"]}), encoding="utf-8")
    seed = int(generation["seed"])
    fields = list(data["fields"])
    if len(fields) != len(set(fields)) or not fields:
        raise ValueError("Unique field names are required.")
    numeric_gap, region_gap = float(generation["numeric_min_gap"]), float(generation["region_min_gap"])
    kwargs = dict(numeric_gap=numeric_gap, region_gap=region_gap,
                  region_size=int(generation["region_size"]), digits=int(generation["decimal_places"]))
    if numeric_gap <= 0 or region_gap <= 0:
        raise ValueError("QA gaps must be positive.")
    counts, excluded, manifests, shape_counts = {}, {}, {}, {}
    with h5py.File(hdf5_path, "r") as handle:
        if any(field not in handle for field in fields):
            raise ValueError("The HDF5 file is missing a configured field.")
        dimensions = tuple(handle[fields[0]].shape)
        if len(dimensions) != 4 or any(tuple(handle[field].shape) != dimensions for field in fields):
            raise ValueError("Fields must share [sample,time,height,width] dimensions.")
        samples, times, source_h, source_w = dimensions
        if any(h > source_h or w > source_w for h, w in all_shapes):
            raise ValueError("A requested shape exceeds the source HDF5 grid.")
        shuffled = list(range(samples))
        random.Random(seed).shuffle(shuffled)
        train_end, val_end = int(samples * 0.8), int(samples * 0.9)
        partitions = {"train": sorted(shuffled[:train_end]), "val": sorted(shuffled[train_end:val_end]),
                      "test": sorted(shuffled[val_end:])}
        if min(map(len, partitions.values())) < 2:
            raise ValueError("Each trajectory split needs at least two samples (normally >= 20 total).")
        for split, sample_ids in partitions.items():
            shapes = train_shapes if split == "train" else all_shapes
            total_train = int(generation["train_states"])
            if total_train < len(train_shapes) * 8 or int(generation["eval_states_per_shape"]) < 8:
                raise ValueError("Use at least eight states per shape for coverage/shuffled controls.")
            split_counter, rejects, state_refs, per_shape = Counter(), Counter(), set(), {}
            destination = output / f"{split}.jsonl"
            with destination.open("w", encoding="utf-8", newline="\n") as writer:
                for shape_index, (height, width) in enumerate(shapes):
                    target_count = (total_train // len(shapes) + int(shape_index < total_train % len(shapes))
                                    if split == "train" else int(generation["eval_states_per_shape"]))
                    rng = random.Random(f"{seed}:{split}:{height}x{width}")
                    accepted, attempts = 0, 0
                    while accepted < target_count:
                        attempts += 1
                        if attempts > target_count * int(generation.get("max_attempts_per_state", 100)):
                            raise RuntimeError(f"Cannot fill {split} {height}x{width}: {dict(rejects)}.")
                        field = fields[accepted % len(fields)]
                        sample = sample_ids[(accepted // len(fields) + shape_index * 17 + attempts - accepted - 1) % len(sample_ids)]
                        time_index = rng.randrange(times)
                        row, col = rng.randrange(source_h - height + 1), rng.randrange(source_w - width + 1)
                        ref = f"{field}_s{sample:06d}_t{time_index:04d}_r{row:04d}_c{col:04d}_h{height}_w{width}"
                        if ref in state_refs:
                            rejects["duplicate_state"] += 1
                            continue
                        raw = torch.as_tensor(handle[field][sample, time_index, row:row + height, col:col + width]).float()
                        try:
                            z, stats = normalized_values(raw)
                            source = {"patch_id": ref, "state_ref": ref, "field": field,
                                      "sample_index": sample, "time_index": time_index, "top_left": [row, col],
                                      "metadata": {"field": field, "grid_shape": [height, width], "coordinate_origin": 1,
                                                   "shape_partition": "seen" if (height, width) in train_shapes else "heldout" if (height, width) in heldout else "extrapolation",
                                                   "normalized_sha256": hashlib.sha256(z.contiguous().numpy().tobytes()).hexdigest()},
                                      "latent_audit": stats}
                            if split == "train":
                                extreme = extreme_record(source, z, rng)
                                records = build_state_records(source, extreme, z, seed=seed,
                                    numeric_gap=numeric_gap, region_gap=region_gap,
                                    region_size=kwargs["region_size"], decimal_places=kwargs["digits"],
                                    spatial_family="point" if (accepted // len(fields)) % 2 == 0 else "region")
                            else:
                                records = evaluation_records(source, z, rng, **kwargs)
                        except ValueError as error:
                            rejects[str(error).split("\n")[0][:180]] += 1
                            continue
                        # A malformed generated prompt is a programming error, not a resampling event.
                        replay_failures = [replay["reason"] for record in records
                                           if not (replay := evaluation_record_replay(
                                               record, z, numeric_gap=numeric_gap, region_gap=region_gap))["eligible"]]
                        if replay_failures:
                            rejects.update(replay_failures)
                            continue
                        for record in records:
                            record.pop("prompt_data", None)
                            question = record["question"].replace("The tensor soft tokens encode", "The field memory contains")
                            record["question"] = record["query"] = question
                            record["metadata"]["prompt_contract"] = VARIABLE_PROMPT_CONTRACT
                            writer.write(json.dumps(record, ensure_ascii=True, allow_nan=False, separators=(",", ":")) + "\n")
                            split_counter[record["task_type"]] += 1
                        state_refs.add(ref)
                        accepted += 1
                    per_shape[shape_name((height, width))] = accepted
                    print(f"build split={split} shape={height}x{width} states={accepted} attempts={attempts}", flush=True)
            counts[split] = dict(split_counter)
            excluded[split] = dict(rejects)
            shape_counts[split] = per_shape
            manifests[split] = sample_ids
    metadata = {"format": VARIABLE_QA_FORMAT, "prompt_contract": VARIABLE_PROMPT_CONTRACT,
                "shape_mode": "variable", "profile": config["experiment_profile"], "split_mode": "sample",
                "natural_language_coordinate_origin": 1, "latent_audit_format": PATCH_LATENT_AUDIT_FORMAT,
                "qa_value_space": "per_patch_zscore_from_raw_patch", "storage_dtype": "float16",
                "encoder_input_normalization": config["field_encoder"]["normalization"], "fields": fields,
                "train_shapes": train_shapes, "heldout_shapes": heldout, "extrapolation_shapes": extrapolation,
                "trajectory_splits": manifests, "source_hdf5_shape": dimensions,
                "stage2b": {"batch_group_size": 3, "train_records": sum(counts["train"].values())},
                "generation": generation, "task_counts": counts, "states_by_shape": shape_counts,
                "excluded_attempts": excluded, "stage1_checkpoint_required": False,
                "output_split_sha256": {split: sha256_file(output / f"{split}.jsonl") for split in manifests}}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    marker.unlink()
    print(f"completed dataset={output} train_records={metadata['stage2b']['train_records']}", flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/field_to_llm_variable_shape.yaml")
    parser.add_argument("--profile", required=True, choices=["pilot", "full"])
    parser.add_argument("--output-dir")
    parser.add_argument("--hdf5-path")
    args = parser.parse_args()
    config = experiment_config(load_yaml_mapping(args.config), args.profile)
    output = args.output_dir or environment_override("FIELD_TO_LLM_VARIABLE_QA_DIR", config["data"]["qa_dir"])
    hdf5 = args.hdf5_path or environment_override("PDEBENCH_HDF5", config["data"]["hdf5_path"])
    # A single CPU builder avoids nested OpenMP thread pools for tiny patches.
    torch.set_num_threads(1)
    build_dataset(config, Path(resolve_path_string(output, ROOT)), Path(resolve_path_string(hdf5, ROOT)))


if __name__ == "__main__":
    main()
