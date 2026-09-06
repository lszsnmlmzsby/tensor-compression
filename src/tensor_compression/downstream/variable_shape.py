"""Shared profile, geometry and batching rules for raw variable-shape field QA."""
from __future__ import annotations

import copy
import hashlib
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence

from torch.utils.data import Sampler


VARIABLE_QA_FORMAT = "variable_shape_field_qa_v1"
VARIABLE_PROMPT_CONTRACT = "raw_field_dynamic_grid_one_based_v1"
MIXED_QA_FORMAT = "mixed_shape_field_qa_v2"
MIXED_PROMPT_CONTRACT = "raw_field_dynamic_grid_explicit_quadrants_v2"


def mixed_protocol(config: Mapping) -> bool:
    return config.get("data", {}).get("qa_protocol") == MIXED_QA_FORMAT


def experiment_config(config: Mapping, profile: str | None) -> dict:
    result = copy.deepcopy(dict(config))
    profiles = result.pop("profiles", None)
    if profiles is None:
        if profile is not None:
            raise ValueError("--profile requires a configuration with profiles.")
        return result
    if profile not in profiles:
        raise ValueError("Select --profile pilot or --profile full explicitly.")

    def merge(target, overlay):
        for key, value in overlay.items():
            if isinstance(value, Mapping) and isinstance(target.get(key), dict):
                merge(target[key], value)
            else:
                target[key] = copy.deepcopy(value)

    merge(result, profiles[profile])
    result["experiment_profile"] = profile
    return result


def parse_shapes(values: Sequence, *, allow_odd: bool = False) -> list[tuple[int, int]]:
    shapes = []
    for value in values:
        parts = value.lower().split("x") if isinstance(value, str) else value
        if len(parts) != 2:
            raise ValueError(f"Expected HxW, got {value!r}.")
        shape = tuple(int(part) for part in parts)
        if any(size < 4 or (size % 2 and not allow_odd) for size in shape):
            raise ValueError(f"First variable-grid protocol requires even H,W >= 4: {shape}.")
        if shape in shapes:
            raise ValueError(f"Duplicate shape: {shape}.")
        shapes.append(shape)
    if not shapes:
        raise ValueError("At least one field shape is required.")
    return shapes


def record_shape(record: Mapping) -> tuple[int, int]:
    shape = record.get("metadata", {}).get("grid_shape")
    if not isinstance(shape, (tuple, list)) or len(shape) != 2:
        raise ValueError(f"Missing grid_shape for {record.get('qa_id')}.")
    height, width = (int(value) for value in shape)
    if min(height, width) <= 0:
        raise ValueError(f"Invalid grid_shape: {shape}.")
    return height, width


def shape_name(shape: Sequence[int]) -> str:
    return f"{int(shape[0])}x{int(shape[1])}"


def rectangular_quadrant(row: int, col: int, height: int, width: int) -> str:
    if not (0 <= row < height and 0 <= col < width):
        raise ValueError("Extremum coordinate is outside the grid.")
    return "ABCD"[2 * int(row >= height // 2) + int(col >= width // 2)]


def balanced_screen_records(records: Sequence[dict], shapes: Sequence, limit: int, seed: int) -> list[dict]:
    """Round-robin complete states across seen shapes; never select held-out shapes."""
    allowed = set(tuple(shape) for shape in shapes)
    buckets = defaultdict(dict)
    for record in records:
        shape = record_shape(record)
        if shape in allowed:
            buckets[shape].setdefault(str(record["state_ref"]), []).append(record)
    if set(buckets) != allowed:
        raise ValueError("Validation is missing a declared training shape.")
    ordered = {}
    for shape, states in buckets.items():
        ordered[shape] = sorted(states, key=lambda key: hashlib.sha256(f"{seed}:{key}".encode()).hexdigest())
    selected = set()
    count = 0
    for offset in range(max(map(len, ordered.values()))):
        for shape in sorted(ordered):
            if offset >= len(ordered[shape]):
                continue
            key = ordered[shape][offset]
            size = len(buckets[shape][key])
            if count + size <= limit:
                selected.add(key)
                count += size
    result = [record for record in records if str(record["state_ref"]) in selected]
    if {record_shape(record) for record in result} != allowed:
        raise ValueError("screening_records is too small to cover every training shape.")
    return result


def mixed_screen_records(records: Sequence[dict], shapes: Sequence, limit: int, seed: int) -> list[dict]:
    """Complete states; exactly 4 real : 2 correlated : 2 IID in every seen shape.

    One packet covers all four real fields and two independent states per synthetic
    family. The screen is fixed before training and excludes all held-out shapes.
    """
    allowed = set(map(tuple, shapes))
    buckets = defaultdict(dict)
    for record in records:
        if record_shape(record) in allowed:
            key = (record_shape(record), record["metadata"]["source_kind"], record["field"])
            buckets[key].setdefault(record["state_ref"], []).append(record)
    ordered = {key: sorted(states, key=lambda ref: hashlib.sha256(f"{seed}:{ref}".encode()).hexdigest())
               for key, states in buckets.items()}
    packets = limit // (len(allowed) * 8 * 5)
    if packets < 1:
        raise ValueError("Mixed screening needs at least eight complete states per training shape.")
    selected = set()
    for shape in sorted(allowed):
        keys = [key for key in ordered if key[0] == shape]
        for key in keys:
            count = packets * (1 if key[1] == "real" else 2)
            if len(ordered[key]) < count:
                raise ValueError("Insufficient validation states for the balanced mixed screen.")
            selected.update(ordered[key][:count])
        if len(keys) != 6:
            raise ValueError("Mixed screening requires four real fields and two synthetic families.")
    return [record for record in records if record["state_ref"] in selected]


def shape_matched_shuffle_indices(records: Sequence[Mapping], seed: int) -> list[int]:
    """Always match shape, prefer field; never substitute the same trajectory."""
    by_shape = defaultdict(lambda: defaultdict(list))
    by_field = defaultdict(lambda: defaultdict(list))
    for index, record in enumerate(records):
        shape = record_shape(record)
        sample = int(record["sample_index"])
        by_shape[shape][sample].append(index)
        by_field[(shape, str(record["field"]))][sample].append(index)
    rng = random.Random(seed)
    # O(N) setup and O(1) draws: rebuilding a list of thousands of synthetic
    # state IDs for every QA row makes the larger experiment needlessly slow.
    indexed = {id(bucket): (list(bucket), {key: index for index, key in enumerate(bucket)})
               for bucket in list(by_shape.values()) + list(by_field.values())}
    result = []
    for record in records:
        shape = record_shape(record)
        sample = int(record["sample_index"])
        bucket = by_field[(shape, str(record["field"]))]
        if len(bucket) < 2:
            bucket = by_shape[shape]
        if len(bucket) < 2:
            raise ValueError(f"Shuffled control needs at least two trajectories for shape {shape}.")
        keys, positions = indexed[id(bucket)]
        draw = rng.randrange(len(keys) - 1)
        draw += int(draw >= positions[sample])
        result.append(rng.choice(bucket[keys[draw]]))
    return result


class ShapeBatchSampler(Sampler[list[int]]):
    """Homogeneous HxW batches; training preserves whole matched groups across DDP.

    Training pads only by repeating complete groups/batches and reports the count.
    Evaluation partitions exact batches across ranks with no padding or duplication.
    """

    def __init__(self, dataset, batch_size: int, *, rank=0, num_replicas=1, seed=42, training=False):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.rank, self.num_replicas = int(rank), int(num_replicas)
        self.seed, self.training, self.epoch = int(seed), bool(training), 0
        if self.batch_size <= 0 or not 0 <= self.rank < self.num_replicas:
            raise ValueError("Invalid shape-batch size/rank/world size.")
        batches, padding = self._batches(0)
        self.padding_records_per_epoch = padding
        self._length = len(batches[self.rank::self.num_replicas])

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self._length

    def _batches(self, epoch):
        rng = random.Random(self.seed + int(epoch))
        by_shape = defaultdict(list)
        if self.training:
            groups = defaultdict(list)
            for index, record in enumerate(self.dataset.records):
                spec = record.get("matched_group", {})
                group = spec.get("batch_group_id")
                if not group or int(spec.get("batch_group_size", 0)) != 3:
                    raise ValueError("Variable training requires explicit atomic groups of three QA rows.")
                groups[group].append(index)
            if self.batch_size % 3:
                raise ValueError("Training batch size must be a multiple of three.")
            for indices in groups.values():
                indices.sort(key=lambda index: self.dataset.records[index]["matched_group"]["batch_member_index"])
                members = [self.dataset.records[index]["matched_group"]["batch_member_index"] for index in indices]
                if members != [0, 1, 2] or len({self.dataset.records[index]["state_ref"] for index in indices}) != 1:
                    raise ValueError("Incomplete or mixed-state matched group.")
                shapes = {record_shape(self.dataset.records[index]) for index in indices}
                if len(shapes) != 1:
                    raise ValueError("Matched group mixes shapes.")
                by_shape[next(iter(shapes))].append(indices)
        else:
            for index, record in enumerate(self.dataset.records):
                by_shape[record_shape(record)].append([index])
        global_steps, flat_batches, padding = [], [], 0
        for shape in sorted(by_shape):
            units = list(by_shape[shape])
            if self.training:
                rng.shuffle(units)
            per_batch = self.batch_size // (3 if self.training else 1)
            if self.training:
                target = math.ceil(len(units) / (per_batch * self.num_replicas)) * per_batch * self.num_replicas
                missing = target - len(units)
                padding += missing * 3
                original_units = list(units)
                units.extend(list(original_units[index % len(original_units)]) for index in range(missing))
            batches = [[index for unit in units[start:start + per_batch] for index in unit]
                       for start in range(0, len(units), per_batch)]
            if self.training:
                global_steps.extend(batches[start:start + self.num_replicas]
                                    for start in range(0, len(batches), self.num_replicas))
            else:
                flat_batches.extend(batches)
        if self.training:
            rng.shuffle(global_steps)
            flat_batches = [batch for step in global_steps for batch in step]
        return flat_batches, padding

    def __iter__(self):
        batches, _padding = self._batches(self.epoch)
        yield from batches[self.rank::self.num_replicas]
