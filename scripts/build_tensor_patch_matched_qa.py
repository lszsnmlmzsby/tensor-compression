from __future__ import annotations

"""Build strict Stage-2B matched QA records from immutable patch latents."""

import argparse
import copy
import hashlib
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.build_tensor_patch_qa import AtomicJsonlWriter, quadrant  # noqa: E402
from tensor_compression.downstream.patch_qa_contract import (  # noqa: E402
    MATCHED_GROUP_FORMAT,
    PATCH_LATENT_AUDIT_FORMAT,
    PATCH_LATENT_FORMAT,
    PATCH_MATCHED_QA_FORMAT,
    PATCH_QA_BUILD_MARKER,
    PATCH_QA_FORMAT,
    PATCH_QA_PROMPT_CONTRACT,
    canonical_normalization,
    latent_identity_from_record,
    latent_qa_stats_from_record,
    sha256_file,
    validate_patch_latent_payload,
    validate_stage1_alignment_checkpoint_payload,
)
from tensor_compression.utils import dump_json  # noqa: E402
from tensor_compression.utils.pipeline_config import (  # noqa: E402
    first_nested,
    load_yaml_mapping,
    require_args,
    resolve_path_string,
    set_default,
)


LABELS = ("A", "B", "C", "D")
EXTREME_RE = re.compile(r"\b(maximum|minimum)\b", re.IGNORECASE)
POINT_RE = re.compile(r"\brow\s+(\d+)\s*,?\s*column\s+(\d+)\b", re.IGNORECASE)
POINT_PAIR_RE = re.compile(
    r"A at row (\d+), column (\d+).*?B at row (\d+), column (\d+)",
    re.IGNORECASE,
)
REGION_PAIR_RE = re.compile(
    r"Region A starts at row (\d+), column (\d+); region B starts at row (\d+), column (\d+)",
    re.IGNORECASE,
)
REGION_SIZE_RE = re.compile(r"two (\d+) by (\d+) regions", re.IGNORECASE)
DISPLAYED_OPTION_RE = re.compile(
    r"(?:Options:\s*|;\s*)([A-D]):\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build QA-only matched-coordinate Stage-2B assets without writing latent files."
    )
    parser.add_argument("--config", type=str, default="configs/tensor_llm_adapter_pipeline.yaml")
    parser.add_argument("--source-qa-dir", type=str, default=None)
    parser.add_argument("--output-qa-dir", type=str, default=None)
    parser.add_argument("--latent-dir", type=str, default=None)
    parser.add_argument("--alignment-checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--numeric-min-gap", type=float, default=None)
    parser.add_argument("--region-min-gap", type=float, default=None)
    parser.add_argument("--region-size", type=int, default=None)
    parser.add_argument("--decimal-places", type=int, default=None)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    config = load_yaml_mapping(args.config)

    path_defaults = {
        "source_qa_dir": first_nested(config, ["patch_qa.qa_dir"]),
        "output_qa_dir": first_nested(config, ["patch_qa.stage2b_qa_dir"]),
        "latent_dir": first_nested(config, ["patch_qa.latent_dir"]),
        "alignment_checkpoint": first_nested(config, ["patch_qa.alignment_checkpoint"]),
    }
    for name, value in path_defaults.items():
        if getattr(args, name) is None and value is not None:
            setattr(args, name, resolve_path_string(value, PROJECT_ROOT))
    set_default(args, "seed", first_nested(config, ["patch_qa.seed"]), 42)
    set_default(
        args,
        "numeric_min_gap",
        first_nested(config, ["patch_qa.stage2b_numeric_min_gap", "patch_qa.numeric_choice_spacing"]),
        0.5,
    )
    set_default(
        args,
        "region_min_gap",
        first_nested(config, ["patch_qa.stage2b_region_min_gap"]),
        0.2,
    )
    set_default(args, "region_size", first_nested(config, ["patch_qa.region_size"]), 4)
    set_default(args, "decimal_places", first_nested(config, ["patch_qa.decimal_places"]), 6)
    set_default(args, "overwrite", first_nested(config, ["patch_qa.stage2b_overwrite"]), False)
    require_args(args, ["source_qa_dir", "output_qa_dir", "latent_dir", "alignment_checkpoint"])
    if float(args.numeric_min_gap) <= 0.0 or float(args.region_min_gap) <= 0.0:
        raise ValueError("Stage-2B numeric and region gaps must be positive.")
    if int(args.region_size) <= 0 or int(args.decimal_places) <= 0:
        raise ValueError("Stage-2B region size and decimal places must be positive.")
    return args


def atomic_dump(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        dump_json(temporary, dict(payload))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def namespaced_seed(seed: int, state_ref: str, namespace: str) -> int:
    value = f"{int(seed)}|{state_ref}|{namespace}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big", signed=False)


def namespaced_rng(seed: int, state_ref: str, namespace: str) -> random.Random:
    return random.Random(namespaced_seed(seed, state_ref, namespace))


def coordinate_set_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def paths_overlap(left: Path, right: Path) -> bool:
    left = left.resolve()
    right = right.resolve()
    try:
        left.relative_to(right)
        return True
    except ValueError:
        pass
    try:
        right.relative_to(left)
        return True
    except ValueError:
        return False


def qa_path(root: Path, split: str) -> Path:
    for candidate in (root / f"{split}.jsonl", root / f"{split}.json"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing source QA split {split!r} under {root}.")


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}.")
            yield value


def source_split_index(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, int]]:
    representatives: dict[str, dict[str, Any]] = {}
    extremes: dict[str, dict[str, Any]] = {}
    qa_ids: set[str] = set()
    counts: Counter[str] = Counter()
    for record in iter_jsonl(path):
        qa_id = str(record.get("qa_id", ""))
        state_ref = str(record.get("state_ref", ""))
        if not qa_id or qa_id in qa_ids or not state_ref:
            raise ValueError(f"Source split {path} has a missing/duplicate identity: {qa_id!r}.")
        qa_ids.add(qa_id)
        previous = representatives.setdefault(state_ref, record)
        if latent_identity_from_record(previous) != latent_identity_from_record(record):
            raise ValueError(f"Source state {state_ref} contains conflicting latent identities.")
        if latent_qa_stats_from_record(previous) != latent_qa_stats_from_record(record):
            raise ValueError(f"Source state {state_ref} contains conflicting latent statistics.")
        task = str(record.get("task_type", ""))
        counts[task] += 1
        if task == "extreme_quadrant" and int(record.get("question_variant", -1)) == 0:
            if state_ref in extremes:
                raise ValueError(f"Source state {state_ref} has duplicate extreme variant 0 records.")
            extremes[state_ref] = record
    return representatives, extremes, dict(sorted(counts.items()))


def validate_source_metadata(metadata: Mapping[str, Any], alignment_path: Path) -> dict[str, Any]:
    if str(metadata.get("format", "")) != PATCH_QA_FORMAT:
        raise ValueError("Stage-2B source QA must be the immutable tensor_patch_qa_v3 asset.")
    if str(metadata.get("prompt_contract", "")) != PATCH_QA_PROMPT_CONTRACT:
        raise ValueError("Stage-2B source QA has an incompatible prompt contract.")
    if int(metadata.get("natural_language_coordinate_origin", -1)) != 1:
        raise ValueError("Stage-2B source QA must use one-based natural-language coordinates.")
    if str(metadata.get("latent_format", "")) != PATCH_LATENT_FORMAT:
        raise ValueError("Stage-2B source QA has an incompatible latent format.")
    if str(metadata.get("latent_audit_format", "")) != PATCH_LATENT_AUDIT_FORMAT:
        raise ValueError("Stage-2B source QA has an incompatible latent-audit format.")
    if str(metadata.get("storage_dtype", "")) != "float16":
        raise ValueError(
            "Stage-2B matched targets are defined from the stored FP16 channel; "
            f"source metadata reports storage_dtype={metadata.get('storage_dtype')!r}."
        )
    expected_sha = str(metadata.get("alignment_checkpoint_sha256", "")).lower()
    actual_sha = sha256_file(alignment_path)
    if expected_sha != actual_sha:
        raise ValueError(
            "The configured alignment checkpoint does not match the source QA provenance: "
            f"metadata={expected_sha}, actual={actual_sha}."
        )
    return {
        "alignment_checkpoint": str(alignment_path.resolve()),
        "alignment_checkpoint_sha256": actual_sha,
        "normalization": canonical_normalization(metadata.get("encoder_input_normalization", {})),
        "latent_shape": [int(value) for value in metadata.get("latent_shape", ())],
        "storage_dtype": str(metadata.get("storage_dtype", "")),
    }


def validate_preserved_channel_checkpoint(
    checkpoint: Mapping[str, Any],
    checkpoint_path: Path,
    latent_shape: Sequence[int],
) -> None:
    validate_stage1_alignment_checkpoint_payload(checkpoint, path=checkpoint_path)
    compressor = checkpoint.get("compressor_config")
    if not isinstance(compressor, Mapping):
        raise ValueError("Alignment checkpoint is missing compressor_config.")
    model = compressor.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("Alignment compressor_config is missing model settings.")
    if str(model.get("name", "")) != "conv_token_autoencoder_2d":
        raise ValueError("Stage-2B requires conv_token_autoencoder_2d preserved-input latents.")
    if not bool(model.get("preserve_input_channels", False)):
        raise ValueError("Stage-2B requires compressor.model.preserve_input_channels=true.")
    if int(model.get("in_channels", 1)) != 1:
        raise ValueError("Stage-2B currently requires a single preserved input channel.")
    input_size = tuple(int(value) for value in model.get("input_size", ()))
    latent_grid = tuple(int(value) for value in model.get("latent_grid", input_size))
    expected_grid = tuple(int(value) for value in latent_shape[-2:])
    if input_size != expected_grid or latent_grid != expected_grid:
        raise ValueError(
            f"Preserved-input checkpoint grid mismatch: input={input_size}, latent={latent_grid}, "
            f"metadata={expected_grid}."
        )


def latent_inventory(paths: Mapping[str, Path]) -> tuple[str, dict[str, dict[str, Any]]]:
    entries: dict[str, dict[str, Any]] = {}
    digest = hashlib.sha256()
    for state_ref, path in sorted(paths.items()):
        if not path.exists():
            raise FileNotFoundError(f"Missing latent for {state_ref}: {path}")
        stat = path.stat()
        file_sha = sha256_file(path)
        entry = {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns), "sha256": file_sha}
        entries[state_ref] = entry
        digest.update(f"{state_ref}|{path.resolve()}|{entry['size']}|{file_sha}\n".encode("utf-8"))
    return digest.hexdigest(), entries


def load_preserved_z(
    record: Mapping[str, Any],
    path: Path,
    contract: Mapping[str, Any],
) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    latent = validate_patch_latent_payload(
        payload,
        path=path,
        expected_identity=latent_identity_from_record(record),
        expected_alignment_checkpoint=contract["alignment_checkpoint"],
        expected_alignment_sha256=contract["alignment_checkpoint_sha256"],
        expected_normalization=contract["normalization"],
        expected_shape=contract["latent_shape"],
        expected_storage_dtype=contract["storage_dtype"],
        expected_qa_stats=latent_qa_stats_from_record(record),
    )
    if latent.dtype != getattr(torch, str(contract["storage_dtype"])):
        raise ValueError(f"Latent {path} dtype changed after validation: {latent.dtype}.")
    values = latent[0].float().contiguous()
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError(f"Preserved channel contains non-finite values: {path}")
    mean = float(values.mean().item())
    std = float(values.std(unbiased=False).item())
    if abs(mean) > 0.1 or not 0.75 <= std <= 1.25:
        raise ValueError(
            f"Preserved channel no longer looks like per-patch z-score data: {path}, mean={mean}, std={std}."
        )
    return values


def separated_option_cells(
    values: torch.Tensor,
    minimum_gap: float,
    rng: random.Random,
) -> tuple[list[float], list[tuple[int, int]], list[int]]:
    height, width = values.shape
    flat = [
        (float(values[row, col].item()), row, col)
        for row in range(height)
        for col in range(width)
    ]

    def separated(items: Sequence[tuple[float, int, int]]) -> bool:
        return all(
            abs(left[0] - right[0]) >= float(minimum_gap)
            for index, left in enumerate(items)
            for right in items[index + 1 :]
        )

    chosen_targets: list[tuple[float, int, int]] | None = None
    for _ in range(4096):
        candidate = rng.sample(flat, 3)
        if separated(candidate):
            chosen_targets = candidate
            break
    if chosen_targets is None:
        # Deterministic exhaustive fallback proves whether the FP16 patch has a legal triple.
        ordered = list(flat)
        rng.shuffle(ordered)
        for left_index, left in enumerate(ordered):
            for right_index in range(left_index + 1, len(ordered)):
                right = ordered[right_index]
                if abs(left[0] - right[0]) < float(minimum_gap):
                    continue
                third = next(
                    (
                        item
                        for item in ordered[right_index + 1 :]
                        if separated((left, right, item))
                    ),
                    None,
                )
                if third is not None:
                    chosen_targets = [left, right, third]
                    break
            if chosen_targets is not None:
                break
    if chosen_targets is None:
        value_range = max(item[0] for item in flat) - min(item[0] for item in flat)
        raise ValueError(
            f"Stored-FP16 patch cannot supply three targets separated by {minimum_gap}; "
            f"range={value_range}."
        )

    distractor_candidates = [item for item in flat if separated((*chosen_targets, item))]
    distractor = rng.choice(distractor_candidates) if distractor_candidates else None
    option_items: list[tuple[float, tuple[int, int] | None]] = [
        (item[0], (item[1], item[2])) for item in chosen_targets
    ]
    if distractor is not None:
        option_items.append((distractor[0], None))
    else:
        target_values = [item[0] for item in chosen_targets]
        synthetic = (
            min(target_values) - float(minimum_gap)
            if rng.random() < 0.5
            else max(target_values) + float(minimum_gap)
        )
        option_items.append((synthetic, None))
    # The withheld distractor has no fixed numeric rank or canonical slot.
    rng.shuffle(option_items)
    canonical = [item[0] for item in option_items]
    target_indices = [index for index, (_value, coordinate) in enumerate(option_items) if coordinate is not None]
    coordinates = [
        coordinate
        for _value, coordinate in option_items
        if coordinate is not None
    ]
    if min(abs(left - right) for i, left in enumerate(canonical) for right in canonical[i + 1 :]) < float(
        minimum_gap
    ):
        raise ValueError("Canonical option construction violated its minimum separation contract.")
    return canonical, coordinates, target_indices


def rendered_options(
    canonical_z: Sequence[float],
    target_z: Sequence[float],
    permutation: Sequence[int],
    minimum_digits: int,
    mean: float | None = None,
    scale: float | None = None,
) -> tuple[str, list[str], list[str], int, str, str]:
    if sorted(int(value) for value in permutation) != list(range(len(canonical_z))):
        raise ValueError(f"Invalid option permutation: {permutation}")
    raw = mean is not None and scale is not None
    for digits in range(max(1, int(minimum_digits)), 18):
        mean_text = f"{float(mean):.{digits}g}" if raw else ""
        scale_text = f"{float(scale):.{digits}g}" if raw else ""
        display_mean = float(mean_text) if raw else 0.0
        display_scale = float(scale_text) if raw else 1.0
        canonical_values = [
            display_mean + display_scale * float(value) if raw else float(value)
            for value in canonical_z
        ]
        displayed = [f"{canonical_values[index]:.{digits}g}" for index in permutation]
        if len(set(displayed)) != len(displayed):
            continue
        parsed = [float(value) for value in displayed]
        answers: list[str] = []
        valid = True
        for z_value in target_z:
            target = display_mean + display_scale * float(z_value) if raw else float(z_value)
            distances = [abs(target - option) for option in parsed]
            ordered = sorted(range(len(distances)), key=distances.__getitem__)
            tolerance = max(1.0e-12, abs(target) * 1.0e-12)
            if len(ordered) < 2 or distances[ordered[1]] - distances[ordered[0]] <= tolerance:
                valid = False
                break
            answers.append(LABELS[ordered[0]])
        if valid and len(set(answers)) == len(answers):
            option_text = "; ".join(f"{label}: {value}" for label, value in zip(LABELS, displayed))
            option_hash = hashlib.sha256(option_text.encode("utf-8")).hexdigest()
            return option_text, list(LABELS), answers, digits, mean_text, scale_text
    raise ValueError("Numeric options remain ambiguous after 17 significant digits.")


def numeric_group_rank_audit(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Describe which numeric rank is withheld from one three-question option set."""
    if len(records) != 3:
        raise ValueError(f"A numeric matched group must contain three records, got {len(records)}.")
    tasks = {str(record.get("task_type", "")) for record in records}
    if len(tasks) != 1 or next(iter(tasks)) not in {
        "normalized_point_value",
        "raw_point_value_with_stats",
    }:
        raise ValueError(f"Expected one numeric task in a matched group, got {sorted(tasks)}.")
    choices = tuple(str(value) for value in records[0].get("choices", ()))
    if choices != LABELS or any(
        tuple(str(value) for value in record.get("choices", ())) != choices
        for record in records
    ):
        raise ValueError("Numeric matched groups must share the ordered A/B/C/D choice labels.")
    displayed = DISPLAYED_OPTION_RE.findall(
        str(records[0].get("query") or records[0].get("question") or "")
    )
    values_by_label = {str(label): float(value) for label, value in displayed}
    if set(values_by_label) != set(LABELS) or len(values_by_label) != len(displayed):
        raise ValueError("Numeric matched-group options are missing, duplicated, or unparsable.")
    if any(
        DISPLAYED_OPTION_RE.findall(str(record.get("query") or record.get("question") or ""))
        != displayed
        for record in records[1:]
    ):
        raise ValueError("All records in a numeric matched group must share exactly one displayed option set.")
    answers = [str(record.get("answer", "")) for record in records]
    if len(set(answers)) != 3 or any(answer not in LABELS for answer in answers):
        raise ValueError("A numeric matched group must use three distinct valid answers.")
    distractor_labels = sorted(set(LABELS) - set(answers))
    if len(distractor_labels) != 1:
        raise ValueError("A numeric matched group must withhold exactly one distractor label.")
    ordered_labels = sorted(LABELS, key=values_by_label.__getitem__)
    rank_by_label = {label: rank + 1 for rank, label in enumerate(ordered_labels)}
    distractor = distractor_labels[0]
    return {
        "task_type": next(iter(tasks)),
        "distractor_label": distractor,
        "distractor_numeric_rank_1_based": rank_by_label[distractor],
        "correct_numeric_ranks_1_based": [rank_by_label[answer] for answer in answers],
    }


def finalize_numeric_rank_audit(
    group_audits: Sequence[Mapping[str, Any]],
    expected_groups_per_task: int,
) -> dict[str, Any]:
    by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in group_audits:
        by_task[str(item["task_type"])].append(item)
    expected_tasks = {"normalized_point_value", "raw_point_value_with_stats"}
    if set(by_task) != expected_tasks:
        raise ValueError(f"Numeric rank audit has unexpected task coverage: {sorted(by_task)}.")
    result: dict[str, Any] = {}
    for task in sorted(expected_tasks):
        items = by_task[task]
        if len(items) != int(expected_groups_per_task):
            raise ValueError(
                f"Numeric rank audit expected {expected_groups_per_task} {task} groups, got {len(items)}."
            )
        distractor_ranks = Counter(int(item["distractor_numeric_rank_1_based"]) for item in items)
        distractor_labels = Counter(str(item["distractor_label"]) for item in items)
        correct_ranks = Counter(
            int(rank)
            for item in items
            for rank in item["correct_numeric_ranks_1_based"]
        )
        maximum_distractor_fraction = max(distractor_ranks.values(), default=0) / max(1, len(items))
        if len(items) >= 16 and (
            set(distractor_ranks) != {1, 2, 3, 4}
            or maximum_distractor_fraction > 0.50
        ):
            raise ValueError(
                f"Numeric distractor ranks expose a fixed shortcut for {task}: "
                f"counts={dict(sorted(distractor_ranks.items()))}."
            )
        result[task] = {
            "groups": len(items),
            "distractor_numeric_rank_1_based": {
                str(rank): int(count) for rank, count in sorted(distractor_ranks.items())
            },
            "distractor_option_label": dict(sorted(distractor_labels.items())),
            "correct_numeric_rank_1_based": {
                str(rank): int(count) for rank, count in sorted(correct_ranks.items())
            },
            "maximum_distractor_rank_fraction": maximum_distractor_fraction,
            "all_four_distractor_ranks_observed": set(distractor_ranks) == {1, 2, 3, 4},
        }
    return result


def base_generated_record(
    source: Mapping[str, Any],
    qa_id: str,
    task: str,
    question: str,
    choices: Sequence[str],
    answer: str,
    variant: int,
) -> dict[str, Any]:
    record = {
        key: copy.deepcopy(source[key])
        for key in (
            "patch_id",
            "state_ref",
            "sample_index",
            "time_index",
            "field",
            "top_left",
            "metadata",
            "latent_audit",
        )
    }
    record.update(
        {
            "qa_id": qa_id,
            "question_variant": int(variant),
            "task_type": task,
            "question": question,
            "query": question,
            "choices": [str(value) for value in choices],
            "answer": str(answer),
        }
    )
    return record


def group_spec(
    *,
    batch_id: str,
    batch_member: int,
    margin_id: str | None,
    margin_kind: str | None,
    margin_size: int,
    margin_member: int,
    query_spec: Mapping[str, Any],
    option_hash: str,
    coordinate_set_id: str,
) -> dict[str, Any]:
    return {
        "format": MATCHED_GROUP_FORMAT,
        "batch_group_id": batch_id,
        "batch_group_size": 3,
        "batch_member_index": int(batch_member),
        "margin_group_id": margin_id,
        "margin_kind": margin_kind,
        "margin_group_size": int(margin_size),
        "margin_member_index": int(margin_member),
        "query_spec": dict(query_spec),
        "option_set_sha256": str(option_hash),
        "coordinate_set_id": str(coordinate_set_id),
    }


def point_pair(values: torch.Tensor, minimum_gap: float, rng: random.Random) -> tuple[tuple[int, int], tuple[int, int]]:
    flat = [(float(values[row, col]), row, col) for row in range(values.shape[0]) for col in range(values.shape[1])]
    flat.sort(key=lambda item: item[0])
    low_pool = flat[: max(1, len(flat) // 4)]
    high_pool = flat[-max(1, len(flat) // 4) :]
    for _ in range(256):
        low = rng.choice(low_pool)
        high = rng.choice(high_pool)
        if high[0] - low[0] >= float(minimum_gap):
            pair = [(low[1], low[2]), (high[1], high[2])]
            rng.shuffle(pair)
            return pair[0], pair[1]
    ordered = list(flat)
    rng.shuffle(ordered)
    for left in ordered:
        for right in ordered:
            if right[0] - left[0] >= float(minimum_gap):
                pair = [(left[1], left[2]), (right[1], right[2])]
                rng.shuffle(pair)
                return pair[0], pair[1]
    raise ValueError(f"Could not construct a point pair with stored-FP16 gap {minimum_gap}.")


def region_pair(
    values: torch.Tensor,
    region_size: int,
    minimum_gap: float,
    rng: random.Random,
) -> tuple[list[int], list[int], float, float]:
    height, width = values.shape
    region = int(region_size)
    if region <= 0 or region > height or region > width:
        raise ValueError(
            f"region_size={region} must fit the stored grid {(height, width)} without clamping."
        )
    candidates: list[tuple[float, int, int]] = []
    for row in range(height - region + 1):
        for col in range(width - region + 1):
            candidates.append((float(values[row : row + region, col : col + region].mean()), row, col))
    first: tuple[float, int, int] | None = None
    second: tuple[float, int, int] | None = None
    for _ in range(2048):
        candidate_a, candidate_b = rng.sample(candidates, 2)
        if abs(candidate_a[0] - candidate_b[0]) >= float(minimum_gap):
            first, second = candidate_a, candidate_b
            break
    if first is None or second is None:
        ordered = list(candidates)
        rng.shuffle(ordered)
        for candidate_a in ordered:
            candidate_b = next(
                (
                    item
                    for item in ordered
                    if abs(candidate_a[0] - item[0]) >= float(minimum_gap)
                ),
                None,
            )
            if candidate_b is not None:
                first, second = candidate_a, candidate_b
                break
    if first is None or second is None:
        means = [item[0] for item in candidates]
        raise ValueError(
            f"Could not construct a region pair with stored-FP16 mean gap {minimum_gap}; "
            f"maximum={max(means) - min(means)}."
        )
    return (
        [first[1], first[2], region, region],
        [second[1], second[2], region, region],
        first[0],
        second[0],
    )


def verify_extreme_replay(
    record: Mapping[str, Any],
    values: torch.Tensor,
) -> dict[str, Any]:
    """Validate a float32-source label against all tied extrema in stored FP16.

    FP16 rounding is monotone, so the source float32 extremum must remain one of
    the stored extrema, although nearby values can become exactly tied after
    serialization.  Cross-quadrant ties therefore remain valid replays when the
    source label names one of those quadrants; a label outside the tied set is a
    genuine provenance failure.
    """
    question = str(record.get("query") or record.get("question") or "")
    match = EXTREME_RE.search(question)
    if match is None:
        raise ValueError(f"Extreme replay {record.get('qa_id')} does not declare maximum/minimum.")
    find_max = match.group(1).lower() == "maximum"
    extreme_value = float(values.max() if find_max else values.min())
    positions = torch.nonzero(values == extreme_value, as_tuple=False).tolist()
    quadrants = {quadrant(int(row), int(col), int(values.shape[0])) for row, col in positions}
    source_answer = str(record.get("answer", ""))
    if source_answer not in quadrants:
        raise ValueError(
            "Stored-FP16 extrema do not support the source float32 replay label for "
            f"{record.get('state_ref')}: source_answer={source_answer!r}, "
            f"fp16_quadrants={sorted(quadrants)}, fp16_positions={len(positions)}."
        )
    tie_scope = (
        "unique_cell"
        if len(positions) == 1
        else "within_quadrant_tie"
        if len(quadrants) == 1
        else "cross_quadrant_tie"
    )
    return {
        "extreme": "maximum" if find_max else "minimum",
        "fp16_position_count": len(positions),
        "fp16_quadrants": sorted(quadrants),
        "tie_scope": tie_scope,
    }


def grounding_target_from_source(record: Mapping[str, Any]) -> dict[str, Any]:
    task = str(record.get("task_type", ""))
    query = str(record.get("query") or record.get("question") or "")
    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"Source record {record.get('qa_id')} is missing metadata.")
    grid = metadata.get("grid_shape")
    if not isinstance(grid, Sequence) or isinstance(grid, (str, bytes)) or len(grid) != 2:
        raise ValueError(f"Source record {record.get('qa_id')} has no two-dimensional grid_shape.")
    height, width = int(grid[0]), int(grid[1])
    origin = int(metadata.get("coordinate_origin", -1))
    if origin != 1:
        raise ValueError("Stage-2B evaluation grounding expects one-based source questions.")

    def point(row: str, col: str) -> list[int]:
        result = [int(row) - origin, int(col) - origin]
        if not (0 <= result[0] < height and 0 <= result[1] < width):
            raise ValueError(
                f"Source grounding coordinate {result} exceeds grid {(height, width)}."
            )
        return result

    if task in {"normalized_point_value", "raw_point_value_with_stats"}:
        matches = POINT_RE.findall(query)
        if len(matches) != 1:
            raise ValueError(f"Expected one point coordinate in source record {record.get('qa_id')}.")
        row, col = point(*matches[0])
        return {"type": "point", "row": row, "col": col, "coordinate_origin": 0}
    if task == "point_compare":
        match = POINT_PAIR_RE.search(query)
        if match is None:
            raise ValueError(f"Expected a point pair in source record {record.get('qa_id')}.")
        row_a, col_a, row_b, col_b = match.groups()
        return {
            "type": "point_pair",
            "a": point(row_a, col_a),
            "b": point(row_b, col_b),
            "coordinate_origin": 0,
        }
    if task == "region_mean_compare":
        match = REGION_PAIR_RE.search(query)
        size_match = REGION_SIZE_RE.search(query)
        if match is None or size_match is None:
            raise ValueError(f"Expected a region pair in source record {record.get('qa_id')}.")
        row_a, col_a, row_b, col_b = match.groups()
        region_h, region_w = int(size_match.group(1)), int(size_match.group(2))
        a = [*point(row_a, col_a), region_h, region_w]
        b = [*point(row_b, col_b), region_h, region_w]
        if a[0] + region_h > height or a[1] + region_w > width:
            raise ValueError(f"Source region A exceeds grid {(height, width)}: {a}.")
        if b[0] + region_h > height or b[1] + region_w > width:
            raise ValueError(f"Source region B exceeds grid {(height, width)}: {b}.")
        return {"type": "region_pair", "a": a, "b": b, "coordinate_origin": 0}
    if task == "extreme_quadrant":
        if EXTREME_RE.search(query) is None:
            raise ValueError(f"Extreme source record {record.get('qa_id')} has no operation.")
        return {"type": "none", "coordinate_origin": 0}
    raise ValueError(f"Unsupported source task for evaluation grounding: {task!r}.")


def build_state_records(
    source: Mapping[str, Any],
    extreme_source: Mapping[str, Any],
    values: torch.Tensor,
    *,
    seed: int,
    numeric_gap: float,
    region_gap: float,
    region_size: int,
    decimal_places: int,
    spatial_family: str,
    extreme_audit_counts: Counter[str] | None = None,
) -> list[dict[str, Any]]:
    state_ref = str(source["state_ref"])
    field = str(source["field"])
    stats = latent_qa_stats_from_record(source)
    mean, scale = float(stats["mean"]), float(stats["scale"])
    option_values, coordinates, target_indices = separated_option_cells(
        values,
        float(numeric_gap),
        namespaced_rng(seed, state_ref, "coordinate_values"),
    )
    coordinate_set_id = coordinate_set_sha256({"type": "point_triple", "points": coordinates})

    norm_permutation = list(range(4))
    namespaced_rng(seed, state_ref, "normalized_option_permutation").shuffle(norm_permutation)
    norm_text, choices, norm_answers, norm_digits, _unused_mean, _unused_scale = rendered_options(
        option_values,
        [option_values[index] for index in target_indices],
        norm_permutation,
        decimal_places,
    )
    norm_hash = hashlib.sha256(norm_text.encode("utf-8")).hexdigest()
    records: list[dict[str, Any]] = []
    norm_group = f"{state_ref}:normalized"
    for member, ((row, col), answer) in enumerate(zip(coordinates, norm_answers)):
        question = (
            f"The tensor soft tokens encode the per-patch standardized {values.shape[0]} by {values.shape[1]} "
            f"matrix z of {field}. The standardization is z = (x - mean) / scale, where mean is "
            f"{mean:.{norm_digits}g} and scale is {scale:.{norm_digits}g}. Which option is closest to z "
            f"at row {row + 1}, column {col + 1}? Options: {norm_text}."
        )
        record = base_generated_record(
            source,
            f"{state_ref}_normalized_point_matched_m{member}",
            "normalized_point_value",
            question,
            choices,
            answer,
            member,
        )
        record["matched_group"] = group_spec(
            batch_id=norm_group,
            batch_member=member,
            margin_id=norm_group,
            margin_kind="coordinate_choice",
            margin_size=3,
            margin_member=member,
            query_spec={"type": "point", "row": row, "col": col, "coordinate_origin": 0},
            option_hash=norm_hash,
            coordinate_set_id=coordinate_set_id,
        )
        records.append(record)

    raw_permutation = list(range(4))
    namespaced_rng(seed, state_ref, "raw_option_permutation").shuffle(raw_permutation)
    if raw_permutation == norm_permutation:
        raw_permutation = raw_permutation[1:] + raw_permutation[:1]
    raw_text, choices, raw_answers, raw_digits, mean_text, scale_text = rendered_options(
        option_values,
        [option_values[index] for index in target_indices],
        raw_permutation,
        decimal_places,
        mean=mean,
        scale=scale,
    )
    raw_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
    raw_group = f"{state_ref}:raw"
    for member, ((row, col), answer) in enumerate(zip(coordinates, raw_answers)):
        question = (
            f"The tensor soft tokens encode the per-patch standardized {values.shape[0]} by {values.shape[1]} "
            f"matrix z of {field}. Recover an original value with x = mean + scale * z, where mean is "
            f"{mean_text} and scale is {scale_text}. Which option is closest to the original value x at "
            f"row {row + 1}, column {col + 1}? Options: {raw_text}."
        )
        record = base_generated_record(
            source,
            f"{state_ref}_raw_point_matched_m{member}",
            "raw_point_value_with_stats",
            question,
            choices,
            answer,
            member,
        )
        record["prompt_data"] = {
            "field": field,
            "mean": mean,
            "std": float(stats["std"]),
            "scale": scale,
            "row": row + 1,
            "col": col + 1,
            "option_text": raw_text,
            "significant_digits": int(raw_digits),
            "option_significant_digits": int(raw_digits),
            "patch_size": int(values.shape[0]),
            "question_variant": int(member),
        }
        record["matched_group"] = group_spec(
            batch_id=raw_group,
            batch_member=member,
            margin_id=raw_group,
            margin_kind="coordinate_choice",
            margin_size=3,
            margin_member=member,
            query_spec={"type": "point", "row": row, "col": col, "coordinate_origin": 0},
            option_hash=raw_hash,
            coordinate_set_id=coordinate_set_id,
        )
        records.append(record)

    spatial_batch = f"{state_ref}:spatial"
    spatial_margin = f"{state_ref}:{spatial_family}_swap"
    if spatial_family == "point":
        first, second = point_pair(
            values,
            numeric_gap,
            namespaced_rng(seed, state_ref, "point_swap"),
        )
        pairs = [(first, second), (second, first)]
        spatial_coordinate_set_id = coordinate_set_sha256(
            {"type": "point_pair", "points": [first, second]}
        )
        pair_values = [float(values[row, col]) for row, col in (first, second)]
        first_answer = "A" if pair_values[0] > pair_values[1] else "B"
        answers = [first_answer, "B" if first_answer == "A" else "A"]
        for member, ((a_row, a_col), (b_row, b_col)) in enumerate(pairs):
            question = (
                f"The tensor soft tokens encode the per-patch standardized {values.shape[0]} by "
                f"{values.shape[1]} matrix of {field}; standardization preserves value order. Which location "
                f"has the larger value: A at row {a_row + 1}, column {a_col + 1}, or B at row "
                f"{b_row + 1}, column {b_col + 1}?"
            )
            record = base_generated_record(
                source,
                f"{state_ref}_point_swap_m{member}",
                "point_compare",
                question,
                ["A", "B"],
                answers[member],
                member,
            )
            record["matched_group"] = group_spec(
                batch_id=spatial_batch,
                batch_member=member,
                margin_id=spatial_margin,
                margin_kind="role_swap",
                margin_size=2,
                margin_member=member,
                query_spec={
                    "type": "point_pair",
                    "a": [a_row, a_col],
                    "b": [b_row, b_col],
                    "coordinate_origin": 0,
                },
                option_hash=hashlib.sha256(b"A|B").hexdigest(),
                coordinate_set_id=spatial_coordinate_set_id,
            )
            records.append(record)
    elif spatial_family == "region":
        first, second, first_mean, second_mean = region_pair(
            values,
            region_size,
            region_gap,
            namespaced_rng(seed, state_ref, "region_swap"),
        )
        pairs = [(first, second, first_mean, second_mean), (second, first, second_mean, first_mean)]
        spatial_coordinate_set_id = coordinate_set_sha256(
            {"type": "region_pair", "regions": [first, second]}
        )
        for member, (region_a, region_b, mean_a, mean_b) in enumerate(pairs):
            answer = "A" if mean_a > mean_b else "B"
            question = (
                f"The tensor soft tokens encode the per-patch standardized {values.shape[0]} by "
                f"{values.shape[1]} matrix of {field}; standardization preserves the ordering of region means. "
                f"Compare the mean values of two {region_a[2]} by {region_a[3]} regions. Region A starts at "
                f"row {region_a[0] + 1}, column {region_a[1] + 1}; region B starts at row "
                f"{region_b[0] + 1}, column {region_b[1] + 1}. Which region has the larger mean?"
            )
            record = base_generated_record(
                source,
                f"{state_ref}_region_swap_m{member}",
                "region_mean_compare",
                question,
                ["A", "B"],
                answer,
                member,
            )
            record["matched_group"] = group_spec(
                batch_id=spatial_batch,
                batch_member=member,
                margin_id=spatial_margin,
                margin_kind="role_swap",
                margin_size=2,
                margin_member=member,
                query_spec={
                    "type": "region_pair",
                    "a": region_a,
                    "b": region_b,
                    "coordinate_origin": 0,
                },
                option_hash=hashlib.sha256(b"A|B").hexdigest(),
                coordinate_set_id=spatial_coordinate_set_id,
            )
            records.append(record)
    else:
        raise ValueError(f"Unsupported spatial family: {spatial_family}")

    extreme_audit = verify_extreme_replay(extreme_source, values)
    if extreme_audit_counts is not None:
        extreme_audit_counts["records"] += 1
        extreme_audit_counts[f"{extreme_audit['extreme']}_records"] += 1
        extreme_audit_counts[f"{extreme_audit['tie_scope']}_records"] += 1
        extreme_audit_counts["fp16_extreme_position_count"] += int(
            extreme_audit["fp16_position_count"]
        )
    extreme = copy.deepcopy(dict(extreme_source))
    extreme.pop("oracle", None)
    extreme["source_qa_id"] = str(extreme_source["qa_id"])
    extreme["qa_id"] = f"{state_ref}_extreme_replay"
    extreme_coordinate_set_id = coordinate_set_sha256(
        {
            "type": "extreme_replay",
            "source_qa_id": str(extreme_source["qa_id"]),
        }
    )
    extreme["matched_group"] = group_spec(
        batch_id=spatial_batch,
        batch_member=2,
        margin_id=None,
        margin_kind=None,
        margin_size=0,
        margin_member=-1,
        query_spec={"type": "none", "coordinate_origin": 0},
        option_hash=hashlib.sha256("|".join(str(value) for value in extreme["choices"]).encode("utf-8")).hexdigest(),
        coordinate_set_id=extreme_coordinate_set_id,
    )
    records.append(extreme)
    if len(records) != 9:
        raise RuntimeError(f"Stage-2B state {state_ref} generated {len(records)} records instead of 9.")
    return records


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_qa_dir).expanduser().resolve()
    output_root = Path(args.output_qa_dir).expanduser().resolve()
    latent_root = Path(args.latent_dir).expanduser().resolve()
    alignment_path = Path(args.alignment_checkpoint).expanduser().resolve()
    if paths_overlap(output_root, source_root) or paths_overlap(output_root, latent_root):
        raise ValueError(
            "Stage-2B output QA directory must be disjoint from source QA and latent directories."
        )
    source_marker = source_root / PATCH_QA_BUILD_MARKER
    if source_marker.exists():
        raise RuntimeError(
            f"Source QA is marked as incomplete, active, or failed: {source_marker}."
        )
    metadata_path = source_root / "metadata.json"
    source_hashes_before = {
        "metadata": sha256_file(metadata_path),
        **{
            split: sha256_file(qa_path(source_root, split))
            for split in ("train", "val", "test")
        },
    }
    with metadata_path.open("r", encoding="utf-8") as handle:
        source_metadata = json.load(handle)
    if not isinstance(source_metadata, Mapping):
        raise ValueError(f"Expected a metadata object: {metadata_path}")
    contract = validate_source_metadata(source_metadata, alignment_path)
    checkpoint = torch.load(alignment_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Alignment checkpoint payload must be a mapping.")
    validate_preserved_channel_checkpoint(checkpoint, alignment_path, contract["latent_shape"])

    split_paths = {split: qa_path(source_root, split) for split in ("train", "val", "test")}
    split_representatives: dict[str, dict[str, dict[str, Any]]] = {}
    split_extremes: dict[str, dict[str, dict[str, Any]]] = {}
    source_task_counts: dict[str, dict[str, int]] = {}
    split_samples: dict[str, set[int]] = {}
    all_representatives: dict[str, dict[str, Any]] = {}
    for split, path in split_paths.items():
        representatives, extremes, task_counts = source_split_index(path)
        split_representatives[split] = representatives
        split_extremes[split] = extremes
        source_task_counts[split] = task_counts
        split_samples[split] = {int(record["sample_index"]) for record in representatives.values()}
        overlap = set(all_representatives) & set(representatives)
        if overlap:
            raise ValueError(f"Source splits share state_ref values: {sorted(overlap)[:4]}.")
        all_representatives.update(representatives)
    if any(split_samples[left] & split_samples[right] for left, right in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Source QA splits share PDE sample indices.")

    train_representatives = split_representatives["train"]
    train_extremes = split_extremes["train"]
    if set(train_representatives) != set(train_extremes):
        missing = sorted(set(train_representatives) - set(train_extremes))
        raise ValueError(f"Every train state needs one extreme variant-0 replay; missing={missing[:4]}.")
    source_summary = source_metadata.get("summary")
    source_splits = source_summary.get("splits") if isinstance(source_summary, Mapping) else None
    if not isinstance(source_splits, Mapping) or not isinstance(source_splits.get("train"), Mapping):
        raise ValueError("Source QA metadata is missing summary.splits.train provenance.")
    declared_train_states = source_splits["train"].get("patches")
    if declared_train_states is None or int(declared_train_states) != len(train_representatives):
        raise ValueError(
            f"Source metadata declares {declared_train_states} train states but JSONL contains "
            f"{len(train_representatives)}."
        )
    for split, representatives in split_representatives.items():
        declared = source_splits.get(split)
        if not isinstance(declared, Mapping) or int(declared.get("patches", -1)) != len(representatives):
            raise ValueError(
                f"Source metadata/JSONL state count mismatch for {split}: "
                f"metadata={declared}, observed={len(representatives)}."
            )

    latent_paths = {state_ref: latent_root / f"{state_ref}.pt" for state_ref in all_representatives}
    before_digest, before_inventory = latent_inventory(latent_paths)

    if output_root.exists() and not bool(args.overwrite):
        existing = [output_root / name for name in ("train.jsonl", "val.jsonl", "test.jsonl", "metadata.json")]
        if any(path.exists() for path in existing):
            raise FileExistsError(f"Stage-2B output already exists under {output_root}; pass --overwrite.")
    output_root.mkdir(parents=True, exist_ok=True)
    marker = output_root / PATCH_QA_BUILD_MARKER
    atomic_dump(
        marker,
        {
            "status": "building",
            "format": PATCH_MATCHED_QA_FORMAT,
            "source_qa_dir": str(source_root),
            "latent_dir": str(latent_root),
        },
    )
    try:
        family_by_state: dict[str, str] = {}
        by_field: dict[str, list[str]] = defaultdict(list)
        for state_ref, record in train_representatives.items():
            by_field[str(record["field"])].append(state_ref)
        for field, states in by_field.items():
            ordered = sorted(states, key=lambda state: namespaced_seed(int(args.seed), state, f"family:{field}"))
            for index, state_ref in enumerate(ordered):
                family_by_state[state_ref] = "point" if index % 2 == 0 else "region"

        output_counts: Counter[str] = Counter()
        output_answer_counts: dict[str, Counter[str]] = defaultdict(Counter)
        numeric_group_audits: list[dict[str, Any]] = []
        extreme_replay_audit_counts: Counter[str] = Counter()
        with AtomicJsonlWriter(output_root / "train.jsonl") as writer:
            for state_ref in sorted(train_representatives):
                source = train_representatives[state_ref]
                values = load_preserved_z(source, latent_paths[state_ref], contract)
                records = build_state_records(
                    source,
                    train_extremes[state_ref],
                    values,
                    seed=int(args.seed),
                    numeric_gap=float(args.numeric_min_gap),
                    region_gap=float(args.region_min_gap),
                    region_size=int(args.region_size),
                    decimal_places=int(args.decimal_places),
                    spatial_family=family_by_state[state_ref],
                    extreme_audit_counts=extreme_replay_audit_counts,
                )
                writer.write_many(records)
                numeric_group_audits.extend(
                    (
                        numeric_group_rank_audit(records[0:3]),
                        numeric_group_rank_audit(records[3:6]),
                    )
                )
                output_counts.update(str(record["task_type"]) for record in records)
                for record in records:
                    output_answer_counts[str(record["task_type"])][str(record["answer"])] += 1
        if int(extreme_replay_audit_counts["records"]) != len(train_representatives):
            raise RuntimeError(
                "Extreme replay audit did not cover every train state: "
                f"audited={extreme_replay_audit_counts['records']}, "
                f"expected={len(train_representatives)}."
            )

        for split in ("val", "test"):
            # The copied evaluation records remain part of the formal contract, so validate
            # every referenced payload before publishing the new metadata envelope.
            for state_ref, source in split_representatives[split].items():
                load_preserved_z(source, latent_paths[state_ref], contract)
            with AtomicJsonlWriter(output_root / f"{split}.jsonl") as writer:
                for source_record in iter_jsonl(split_paths[split]):
                    output_record = copy.deepcopy(source_record)
                    output_record.pop("oracle", None)
                    output_record["grounding_target"] = grounding_target_from_source(
                        output_record
                    )
                    writer.write_many([output_record])

        after_digest, after_inventory = latent_inventory(latent_paths)
        if before_digest != after_digest or before_inventory != after_inventory:
            raise RuntimeError("Latent inventory changed while Stage-2B QA was being generated.")
        source_hashes_after = {
            "metadata": sha256_file(metadata_path),
            **{split: sha256_file(path) for split, path in split_paths.items()},
        }
        if source_hashes_before != source_hashes_after:
            raise RuntimeError("Source QA metadata or JSONL changed during Stage-2B generation.")
        output_split_hashes = {
            split: sha256_file(output_root / f"{split}.jsonl")
            for split in ("train", "val", "test")
        }
        output_metadata = copy.deepcopy(dict(source_metadata))
        numeric_rank_summary = finalize_numeric_rank_audit(
            numeric_group_audits,
            expected_groups_per_task=len(train_representatives),
        )
        output_metadata.update(
            {
                "format": PATCH_MATCHED_QA_FORMAT,
                "matched_group_format": MATCHED_GROUP_FORMAT,
                "requires_explicit_group_sampler": True,
                "source_qa_format": PATCH_QA_FORMAT,
                "source_qa_dir": str(source_root),
                "source_split_sha256": source_hashes_before,
                "output_split_sha256": output_split_hashes,
                "qa_dir": str(output_root),
                "latent_dir": str(latent_root),
                "question_seed_mode": "sha256(seed|state_ref|namespace)",
                "target_source": "stage2b_split_specific_v1",
                "latent_inventory_sha256": after_digest,
                "stage2b": {
                    "seed": int(args.seed),
                    "records_per_train_state": 9,
                    "batch_groups_per_train_state": 3,
                    "batch_group_size": 3,
                    "numeric_min_gap_z": float(args.numeric_min_gap),
                    "region_min_gap_z": float(args.region_min_gap),
                    "region_size": int(args.region_size),
                    "train_states": len(train_representatives),
                    "train_records": sum(output_counts.values()),
                    "train_by_task": dict(sorted(output_counts.items())),
                    "train_answers_by_task": {
                        task: dict(sorted(counts.items()))
                        for task, counts in sorted(output_answer_counts.items())
                    },
                    "numeric_rank_audit": numeric_rank_summary,
                    "extreme_replay_audit": {
                        "policy": (
                            "source_float32_answer_must_belong_to_stored_fp16_tied_extreme_quadrants"
                        ),
                        "counts": dict(sorted(extreme_replay_audit_counts.items())),
                    },
                    "source_by_task": source_task_counts,
                    "target_provenance": {
                        "train_numeric_and_compare": "preserved_input_channel_0_as_stored_float16",
                        "train_extreme": "validated_source_v3_variant_0_replay",
                        "val_test": (
                            "source_v3_prompt_and_label_replay_with_non_prompt_grounding_targets"
                        ),
                    },
                    "grounding_annotation_contract": {
                        "train_field": "matched_group.query_spec",
                        "evaluation_field": "grounding_target",
                        "coordinate_origin": 0,
                        "included_in_model_prompt": False,
                        "purpose": "routing_supervision_and_held_out_routing_metrics_only",
                    },
                },
                "question_variants": {
                    "train": 3,
                    "val": int(source_metadata.get("question_variants", {}).get("val", 1)),
                    "test": int(source_metadata.get("question_variants", {}).get("test", 1)),
                },
            }
        )
        output_summary = output_metadata.get("summary")
        output_splits = output_summary.get("splits") if isinstance(output_summary, dict) else None
        if not isinstance(output_splits, dict) or not isinstance(output_splits.get("train"), dict):
            raise ValueError("Copied output metadata lost summary.splits.train.")
        output_splits["train"].update(
            {
                "patches": len(train_representatives),
                "question_variants_per_patch": 9,
                "qa_records": sum(output_counts.values()),
                "by_task": dict(sorted(output_counts.items())),
                "answers_by_task": {
                    task: dict(sorted(counts.items()))
                    for task, counts in sorted(output_answer_counts.items())
                },
            }
        )
        atomic_dump(output_root / "metadata.json", output_metadata)
        marker.unlink()
    except BaseException as exc:
        atomic_dump(
            marker,
            {
                "status": "failed",
                "format": PATCH_MATCHED_QA_FORMAT,
                "error_type": type(exc).__name__,
                "error": str(exc)[:2000],
            },
        )
        raise

    print(
        f"stage2b_qa_dir={output_root} train_states={len(train_representatives)} "
        f"train_records={sum(output_counts.values())} "
        "extreme_cross_quadrant_ties="
        f"{int(extreme_replay_audit_counts['cross_quadrant_tie_records'])} "
        f"latent_inventory={after_digest}"
    )


if __name__ == "__main__":
    main()
