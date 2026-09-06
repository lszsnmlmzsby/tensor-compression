"""Generation helpers for the versioned mixed-field experiment (no model routing)."""
from __future__ import annotations

import hashlib
import random

import numpy as np
import torch

from scripts.build_tensor_patch_matched_qa import (
    DISPLAYED_OPTION_RE, RAW_STATS_RE, base_generated_record, group_spec,
    grounding_target_from_source, rendered_options,
)


def synthetic_raw(shape, kind, seed):
    """Sample once, then store float32 raw arrays in the immutable QA HDF5 asset.

    Correlated fields mix waves, localized blobs, ramps and weak noise. IID fields
    alternate Gaussian and uniform distributions. Neither is a physical PDE label.
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    height, width = shape
    if kind == "iid":
        field = rng.normal(size=shape) if rng.random() < 0.5 else rng.uniform(-1, 1, size=shape)
    elif kind == "correlated":
        y, x = np.meshgrid(np.linspace(-1, 1, height), np.linspace(-1, 1, width), indexing="ij")
        field = rng.uniform(-1, 1) * x + rng.uniform(-1, 1) * y
        for _ in range(4):
            angle = rng.uniform(0, 2 * np.pi)
            wavelength = 10 ** rng.uniform(-0.6, 0.5)
            field += rng.normal() * np.sin(2 * np.pi * (np.cos(angle) * x + np.sin(angle) * y) / wavelength + rng.uniform(0, 2 * np.pi))
        for _ in range(3):
            cx, cy = rng.uniform(-1, 1, size=2)
            sx, sy = 10 ** rng.uniform(-1.1, -0.1, size=2)
            field += rng.normal() * np.exp(-0.5 * (((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2))
        field += rng.uniform(0.01, 0.15) * rng.normal(size=shape)
    else:
        raise ValueError(f"Unknown synthetic family: {kind}.")
    mean, scale = rng.uniform(-10, 10), 10 ** rng.uniform(-2, 2)
    raw = mean + scale * (field - field.mean()) / (field.std() + 1e-12)
    return torch.from_numpy(raw.astype(np.float32))


def numeric_pair(source, z, rng, *, gap, digits, member):
    row, col = rng.randrange(z.shape[0]), rng.randrange(z.shape[1])
    target = float(z[row, col])
    slot = rng.randrange(4)
    options = [target + (index - slot) * gap for index in range(4)]
    pair = []
    for raw in (False, True):
        permutation = list(range(4))
        rng.shuffle(permutation)
        stats = source["latent_audit"]
        text, choices, answers, _, mean, scale = rendered_options(
            options, [target], permutation, digits,
            mean=stats["mean"] if raw else None, scale=stats["scale"] if raw else None,
        )
        task = "raw_point_value_with_stats" if raw else "normalized_point_value"
        query = f"The field is a {z.shape[0]} by {z.shape[1]} standardized matrix of {source['field']}. "
        if raw:
            query += f"Recover x = mean + scale * z, where mean is {mean} and scale is {scale}. "
        query += f"Which option is closest to {'the original value x' if raw else 'z'} at row {row + 1}, column {col + 1}? Options: {text}."
        record = base_generated_record(source, f"{source['state_ref']}_{task}_uniform_m{member}", task, query, choices, answers[0], member)
        record["matched_group"] = group_spec(
            batch_id=f"{source['state_ref']}:{task}:uniform", batch_member=member,
            margin_id=None, margin_kind=None, margin_size=0, margin_member=0,
            query_spec={"type": "point", "row": row, "col": col, "coordinate_origin": 0},
            option_hash=hashlib.sha256(text.encode()).hexdigest(), coordinate_set_id=f"{row}:{col}",
        )
        pair.append(record)
    return pair


def finish_mixed_records(records, source, z, rng, *, training, uniform_numeric, gap, digits, point_gap):
    if training and uniform_numeric:
        pairs = [numeric_pair(source, z, rng, gap=gap, digits=digits, member=index) for index in range(3)]
        records = [pair[0] for pair in pairs] + [pair[1] for pair in pairs] + records[6:]
    # Uniform candidate locations conditional only on a small, FP16-replayed gap.
    point_records = [record for record in records if record["task_type"] == "point_compare"]
    if point_records:
        for _ in range(4096):
            a = (rng.randrange(z.shape[0]), rng.randrange(z.shape[1]))
            b = (rng.randrange(z.shape[0]), rng.randrange(z.shape[1]))
            if abs(float(z[a]) - float(z[b])) >= point_gap:
                break
        else:
            raise ValueError("no_uniform_point_pair")
        for member, record in enumerate(point_records):
            first, second = (a, b) if member == 0 else (b, a)
            query = (f"The field is a {z.shape[0]} by {z.shape[1]} standardized matrix of {source['field']}. "
                     f"Which location has the larger value: A at row {first[0] + 1}, column {first[1] + 1}, "
                     f"or B at row {second[0] + 1}, column {second[1] + 1}?")
            record["question"] = record["query"] = query
            record["answer"] = "A" if z[first] > z[second] else "B"
            if training:
                record["matched_group"].update(
                    query_spec={"type": "point_pair", "a": list(first), "b": list(second), "coordinate_origin": 0},
                    coordinate_set_id=hashlib.sha256(str(sorted((a, b))).encode()).hexdigest(),
                )
    for record in records:
        task = record["task_type"]
        query = record["question"]
        if task == "normalized_point_value":
            # Same wording for matched and uniform numeric recipes, including eval.
            query = (f"The field is a {z.shape[0]} by {z.shape[1]} standardized matrix of {source['field']}. "
                     + query[query.index("Which option"):])
        elif task == "raw_point_value_with_stats":
            mean, scale = RAW_STATS_RE.search(query).groups()
            query = (f"The field is a {z.shape[0]} by {z.shape[1]} standardized matrix of {source['field']}. "
                     f"Recover x = mean + scale * z, where mean is {mean} and scale is {scale}. "
                     + query[query.index("Which option"):])
        if task == "extreme_quadrant":
            h, w = z.shape
            query += (f" Top means rows 1 through {h // 2}; bottom means rows {h // 2 + 1} through {h}."
                      f" Left means columns 1 through {w // 2}; right means columns {w // 2 + 1} through {w}.")
        record["question"] = record["query"] = query
        record["metadata"]["numeric_recipe"] = "uniform" if (not training or uniform_numeric) else "matched"
        add_numeric_diagnostics(record, z)
    return records


def add_numeric_diagnostics(record, z):
    """Audit-only data derived from the actual prompt and preserved value channel."""
    target = grounding_target_from_source(record)
    task = record["task_type"]
    diagnostic = {"target": target}
    if target["type"] == "point":
        row, col = target["row"], target["col"]
        diagnostic["target_z"] = float(z[row, col])
        displayed = {label: float(value) for label, value in DISPLAYED_OPTION_RE.findall(record["query"])}
        mean, scale = 0.0, 1.0
        if task == "raw_point_value_with_stats":
            mean, scale = map(float, RAW_STATS_RE.search(record["query"]).groups())
        diagnostic["option_z"] = {label: (value - mean) / scale for label, value in displayed.items()}
        diagnostic["target_displayed_value"] = mean + scale * diagnostic["target_z"]
        diagnostic["option_values"] = displayed
        # member distinguishes repeated coordinates sampled independently in a state.
        diagnostic["pair_id"] = f"{record['state_ref']}:{row}:{col}:m{record.get('question_variant', 0)}"
    elif target["type"] == "point_pair":
        diagnostic["value_gap"] = abs(float(z[tuple(target["a"])]) - float(z[tuple(target["b"])]))
    elif target["type"] == "region_pair":
        def mean(spec):
            row, col, h, w = spec
            return float(z[row:row + h, col:col + w].mean())
        diagnostic["value_gap"] = abs(mean(target["a"]) - mean(target["b"]))
    record["metadata"]["diagnostic"] = diagnostic
