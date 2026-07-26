from __future__ import annotations

"""Mechanism-oriented diagnostics for direct and grounded spatial Stage-2 adapters.

This script deliberately keeps diagnostics separate from the training path.  It
uses the formal single-token restricted-choice logits, then adds three probes:

* in-distribution question swaps on the same tensor;
* local channel-0 counterfactuals (explicitly marked as OOD mechanism tests);
* text-only upper-bound controls and secondary representation probes.

The output is a JSON report.  No oracle field is ever passed to the model.
This entry point intentionally requires the v3 QA metadata and v2 checkpoint
provenance envelope; it will fail before loading the LLM when those contracts
are absent or inconsistent.
"""

import argparse
import copy
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tensor_compression.downstream.pdebench import resolve_device  # noqa: E402
from tensor_compression.utils import dump_json  # noqa: E402

from scripts.diagnose_spatial_token_readout import (  # noqa: E402
    collect_stage_features,
    extract_spatial_token_stages,
    fit_ridge_readout,
    unique_state_examples,
)
from scripts.train_tensor_llm_adapter import (  # noqa: E402
    GroundedEvidenceAdapter,
    HybridGlobalLocalAdapter,
    TensorReadoutQADataset,
    _decoder_for_diagnostics,
    adapter_from_checkpoint,
    adapter_soft_embeds,
    apply_config_defaults,
    apply_runtime_environment,
    audit_qa_datasets,
    audit_qa_metadata,
    build_prompt,
    contextual_adapter_soft_embeds,
    grounded_soft_prompt_attention_mask,
    last_nonpadding_indices,
    load_tokenizer_and_llm,
    model_identifier_leaf,
    qa_path,
    set_frozen_llm_execution_mode,
    single_token_choice_ids,
    validate_adapter_checkpoint_payload,
    validate_stage1_model_identity,
)


NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
COORDINATE_RE = re.compile(
    r"\brow\s+(?P<row>\d+)\s*,?\s*column\s+(?P<col>\d+)\b",
    flags=re.IGNORECASE,
)
NUMERIC_OPTION_RE = re.compile(
    rf"(?P<label>[A-D])\s*:\s*(?P<value>{NUMBER})",
    flags=re.IGNORECASE,
)
REGION_SIZE_RE = re.compile(r"\b(?P<h>\d+)\s+by\s+(?P<w>\d+)\s+regions?\b", re.IGNORECASE)
REGION_START_RE = re.compile(
    r"region\s+[AB]\s+starts?\s+at\s+row\s+(?P<row>\d+)\s*,?\s*column\s+(?P<col>\d+)",
    flags=re.IGNORECASE,
)
STATS_RE = re.compile(
    rf"mean\s+(?:is|=)\s*(?P<mean>{NUMBER}).*?scale\s+(?:is|=)\s*(?P<scale>{NUMBER})",
    flags=re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose coordinate-selective tensor grounding for a direct spatial adapter. "
            "The formal score is prompt-boundary restricted A/B/C/D logits."
        )
    )
    parser.add_argument("--config", type=str, default="configs/tensor_llm_adapter_pipeline.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Stage-2 adapter_best.pt/adapter_last.pt.")
    parser.add_argument("--stage1-checkpoint", type=str, default=None)
    parser.add_argument("--qa-dir", type=str, default=None)
    parser.add_argument("--latent-dir", type=str, default=None)
    parser.add_argument("--model-name-or-path", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--hf-home", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--torch-dtype", type=str, default=None, choices=("auto", "float16", "bfloat16", "float32"))
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--prefer-record-latent-ref", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--prompt-template", type=str, default=None, choices=("generic", "task_specific"))
    parser.add_argument("--max-prompt-tokens", type=int, default=None)
    parser.add_argument("--max-target-tokens", type=int, default=None)
    parser.add_argument("--append-eos", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--local-context-layer", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--shuffle-seed", type=int, default=None)
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Scored diagnostic split; test is the default to avoid checkpoint-selection bias.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=1020,
        help="Number of scored records; 0 means all records in the selected split.",
    )
    parser.add_argument("--latent-cache-size", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)

    parser.add_argument("--run-coordinate-swaps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-local-interventions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-gradient-scan", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-text-controls", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-formal-baselines", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-stats-swap-control", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-serialized-matrix-control", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-spatial-probe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compare-stage1", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--max-states-per-task", type=int, default=128)
    parser.add_argument("--min-point-gap", type=float, default=0.5)
    parser.add_argument("--min-region-gap", type=float, default=0.2)
    parser.add_argument("--intervention-records-per-task", type=int, default=64)
    parser.add_argument("--intervention-controls-per-record", type=int, default=4)
    parser.add_argument("--gradient-records-per-task", type=int, default=32)
    parser.add_argument("--score-batch-size", type=int, default=4)
    parser.add_argument("--probe-train-states", type=int, default=64)
    parser.add_argument("--probe-val-states", type=int, default=64)
    parser.add_argument("--probe-train-records", type=int, default=2048)
    parser.add_argument("--probe-val-records", type=int, default=2048)
    parser.add_argument("--probe-positions-per-state", type=int, default=8)
    parser.add_argument("--probe-feature-batch-size", type=int, default=2)
    parser.add_argument("--representation-states", type=int, default=16)
    parser.add_argument("--probe-ridge", type=float, default=1.0e-3)
    parser.add_argument("--probe-tolerance", type=float, default=0.25)
    parser.add_argument("--probe-device", type=str, default="cpu")
    parser.add_argument("--text-control-records-per-task", type=int, default=32)
    parser.add_argument("--text-control-max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--bootstrap-reps", type=int, default=500)
    return parser.parse_args()


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Checkpoint is not a mapping: {path}")
    return dict(payload)


def validate_diagnostic_args(args: argparse.Namespace) -> None:
    positive = (
        "max_states_per_task",
        "intervention_records_per_task",
        "gradient_records_per_task",
        "score_batch_size",
        "probe_train_states",
        "probe_val_states",
        "probe_positions_per_state",
        "probe_feature_batch_size",
        "text_control_records_per_task",
        "text_control_max_prompt_tokens",
        "representation_states",
    )
    for name in positive:
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if int(args.max_records) < 0:
        raise ValueError("--max-records must be non-negative (0 means all records).")
    if int(args.intervention_controls_per_record) < 0 or int(args.bootstrap_reps) < 0:
        raise ValueError("Intervention control count and bootstrap reps must be non-negative.")
    if float(args.min_point_gap) < 0.0 or float(args.min_region_gap) < 0.0:
        raise ValueError("Coordinate-swap gap thresholds must be non-negative.")
    if float(args.probe_ridge) <= 0.0:
        raise ValueError("--probe-ridge must be positive.")


def preflight_checkpoint_envelope(
    checkpoint: Mapping[str, Any],
    expected_architecture: str,
    expected_latent_contract: Mapping[str, Any] | None,
) -> None:
    """Reject malformed/mismatched Stage-2 files before loading the 14B model."""

    if str(checkpoint.get("checkpoint_type", "")) != "tensor_llm_adapter":
        raise ValueError("Stage-2 checkpoint has an invalid checkpoint_type.")
    try:
        checkpoint_version = int(checkpoint.get("checkpoint_version", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError("Stage-2 checkpoint_version is not an integer.") from exc
    if checkpoint_version < 2:
        raise ValueError("Stage-2 checkpoint lacks the version-2 provenance envelope.")
    state_dict = checkpoint.get("adapter_state_dict")
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("Stage-2 checkpoint has no adapter_state_dict.")
    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            raise ValueError(f"Stage-2 checkpoint state entry {key!r} is not a tensor.")
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"Stage-2 checkpoint state entry {key!r} contains NaN or infinity.")
    checkpoint_args = checkpoint.get("args")
    if not isinstance(checkpoint_args, Mapping):
        raise ValueError("Stage-2 checkpoint has no args mapping.")
    if str(checkpoint_args.get("adapter_architecture", "")) != str(expected_architecture):
        raise ValueError(
            "Stage-2 checkpoint architecture differs from the requested diagnostic contract: "
            f"checkpoint={checkpoint_args.get('adapter_architecture')!r}, "
            f"expected={expected_architecture!r}."
        )
    if not isinstance(checkpoint.get("latent_contract"), Mapping):
        raise ValueError("Formal Stage-2 checkpoint is missing latent_contract provenance.")
    if expected_latent_contract is not None and dict(checkpoint.get("latent_contract", {})) != dict(expected_latent_contract):
        raise ValueError("Stage-2 checkpoint latent_contract differs from QA metadata.")


def configure_runtime_args(raw: argparse.Namespace, checkpoint: Mapping[str, Any]) -> tuple[argparse.Namespace, str]:
    checkpoint_args = checkpoint.get("args")
    if not isinstance(checkpoint_args, Mapping):
        raise ValueError("Stage-2 checkpoint has no args mapping.")
    # Make checkpoint-only invocation self-contained.  The project config is
    # still used for defaults, but a stale config must not silently change the
    # model/data contract recorded in the checkpoint.
    for field, value in checkpoint_args.items():
        if not hasattr(raw, str(field)) or getattr(raw, str(field), None) is None:
            setattr(raw, str(field), value)
    explicit_fields = (
        "qa_dir",
        "latent_dir",
        "model_name_or_path",
        "cache_dir",
        "hf_home",
        "device",
        "torch_dtype",
        "trust_remote_code",
        "prefer_record_latent_ref",
        "prompt_template",
        "max_prompt_tokens",
        "max_target_tokens",
        "append_eos",
        "local_context_layer",
        "seed",
        "shuffle_seed",
        "latent_cache_size",
    )
    explicit = {field: getattr(raw, field, None) is not None for field in explicit_fields}
    # ``apply_config_defaults`` validates the training namespace too.  A
    # checkpoint-only diagnostic should still work when an older checkpoint did
    # not persist the output directory, so use its parent as a non-writing
    # placeholder before applying the shared defaults.
    if getattr(raw, "output_root", None) is None:
        raw.output_root = checkpoint_args.get("output_root") or str(Path(raw.checkpoint).expanduser().resolve().parent)
    args = apply_config_defaults(raw)
    for field in explicit_fields:
        if explicit[field]:
            continue
        value = checkpoint_args.get(field)
        if value is not None:
            setattr(args, field, value)
    for field in ("train_split", "val_split", "test_split", "llm_gradient_checkpointing", "low_cpu_mem_usage"):
        value = checkpoint_args.get(field)
        if value is not None:
            setattr(args, field, value)

    stage1_path = str(
        raw.stage1_checkpoint
        or checkpoint_args.get("adapter_init_checkpoint")
        or checkpoint_args.get("qa_alignment_checkpoint")
        or args.qa_alignment_checkpoint
        or ""
    ).strip()
    if not stage1_path and bool(args.compare_stage1):
        raise ValueError(
            "Cannot compare Stage 1: pass --stage1-checkpoint or use a Stage-2 checkpoint "
            "that records adapter_init_checkpoint."
        )
    # These two values are used only for strict provenance validation.  They must
    # identify the Stage-1 alignment checkpoint, never the downstream checkpoint.
    args.adapter_init_checkpoint = stage1_path or args.adapter_init_checkpoint
    args.qa_alignment_checkpoint = stage1_path or args.qa_alignment_checkpoint
    args.adapter_architecture = str(checkpoint_args.get("adapter_architecture", args.adapter_architecture))
    for field in ("prompt_template", "max_prompt_tokens", "local_context_layer"):
        expected = checkpoint_args.get(field)
        if expected is not None and getattr(args, field) != expected:
            raise ValueError(
                f"Diagnostic {field}={getattr(args, field)!r} differs from checkpoint {field}={expected!r}; "
                "formal logits require the checkpoint prompt contract."
            )
    args.console_progress = False
    args.wandb_enabled = False
    # This script intentionally has one interpretation mode: formal direct
    # grounding.  A legacy/sanity path would not support the provenance and
    # split guarantees needed to interpret coordinate interventions.
    args.require_disjoint_splits = True
    checkpoint_model = str(checkpoint_args.get("model_name_or_path", "")).strip()
    active_model = str(getattr(args, "model_name_or_path", "")).strip()
    if checkpoint_model and active_model and model_identifier_leaf(checkpoint_model) != model_identifier_leaf(active_model):
        raise ValueError(
            "The active frozen LLM differs from the model recorded in the Stage-2 checkpoint: "
            f"checkpoint={checkpoint_model!r}, active={active_model!r}."
        )
    return args, stage1_path


def choice_labels(record: Mapping[str, Any]) -> list[str]:
    choices = record.get("choices")
    if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)) or not choices:
        choices = [str(record.get("answer", "A"))]
    labels = [str(value) for value in choices]
    answer = str(record.get("answer", ""))
    if answer and answer not in labels:
        labels = [answer] + labels
    return labels


def record_query(record: Mapping[str, Any]) -> str:
    return str(record.get("query") or record.get("question") or "")


def grid_origin(record: Mapping[str, Any]) -> int:
    metadata = record.get("metadata")
    if isinstance(metadata, Mapping):
        return int(metadata.get("coordinate_origin", 1))
    return 1


def parse_coordinates(record: Mapping[str, Any]) -> list[tuple[int, int]]:
    query = record_query(record)
    origin = grid_origin(record)
    return [
        (int(match.group("row")) - origin, int(match.group("col")) - origin)
        for match in COORDINATE_RE.finditer(query)
    ]


def parse_numeric_options(record: Mapping[str, Any]) -> dict[str, float]:
    query = record_query(record)
    suffix = query.split("Options:", 1)[-1] if "Options:" in query else query
    return {match.group("label").upper(): float(match.group("value")) for match in NUMERIC_OPTION_RE.finditer(suffix)}


def parse_stats(record: Mapping[str, Any]) -> tuple[float, float] | None:
    match = STATS_RE.search(record_query(record))
    if match is not None:
        return float(match.group("mean")), float(match.group("scale"))
    audit = record.get("latent_audit")
    if isinstance(audit, Mapping) and audit.get("mean") is not None and audit.get("scale") is not None:
        return float(audit["mean"]), float(audit["scale"])
    return None


def parse_region_specs(record: Mapping[str, Any]) -> tuple[tuple[int, int], tuple[int, int], int, int] | None:
    """Parse the two region starts and their rectangular size from a QA query.

    The builder emits one-based starts in prose (``Region A starts at row ...``).
    We deliberately parse the natural-language query rather than an oracle field so
    this remains valid for the formal loader, which removes oracle metadata.
    """

    query = record_query(record)
    starts = list(REGION_START_RE.finditer(query))
    size_match = REGION_SIZE_RE.search(query)
    if len(starts) != 2 or size_match is None:
        # Keep a permissive fallback for older records that only say "region A"
        # and use the generic coordinate wording.
        coordinates = parse_coordinates(record)
        if len(coordinates) != 2 or size_match is None:
            return None
        starts_zero = coordinates
    else:
        origin = grid_origin(record)
        starts_zero = [
            (int(item.group("row")) - origin, int(item.group("col")) - origin)
            for item in starts
        ]
    height = int(size_match.group("h"))
    width = int(size_match.group("w"))
    if height <= 0 or width <= 0:
        return None
    return starts_zero[0], starts_zero[1], height, width


def latent_value(latent: torch.Tensor, coordinate: tuple[int, int], channel: int = 0) -> float:
    row, col = coordinate
    if not (0 <= row < int(latent.shape[-2]) and 0 <= col < int(latent.shape[-1])):
        raise IndexError(f"Coordinate {coordinate} is outside latent grid {tuple(latent.shape[-2:])}.")
    return float(latent[int(channel), row, col].detach().float().cpu().item())


def replace_coordinate_text(query: str, matches: Sequence[re.Match[str]], replacements: Sequence[tuple[int, int]]) -> str:
    if len(matches) != len(replacements):
        raise ValueError("Coordinate replacement count does not match query matches.")
    pieces: list[str] = []
    cursor = 0
    for match, (row, col) in zip(matches, replacements):
        pieces.append(query[cursor : match.start()])
        pieces.append(f"row {int(row)}, column {int(col)}")
        cursor = match.end()
    pieces.append(query[cursor:])
    return "".join(pieces)


def make_role_swap_record(record: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]] | None:
    query = record_query(record)
    matches = list(COORDINATE_RE.finditer(query))
    if len(matches) != 2:
        return None
    coordinates = [
        (int(match.group("row")), int(match.group("col")))
        for match in matches
    ]
    swapped_query = replace_coordinate_text(query, matches, [coordinates[1], coordinates[0]])
    original = copy.deepcopy(dict(record))
    swapped = copy.deepcopy(dict(record))
    swapped["query"] = swapped_query
    swapped["question"] = swapped_query
    answer = str(record.get("answer", ""))
    if answer in {"A", "B"}:
        swapped["answer"] = "B" if answer == "A" else "A"
    swapped["qa_id"] = f"{record.get('qa_id', 'record')}_role_swap"
    return original, swapped


def region_cells(start: tuple[int, int], height: int, width: int, grid: tuple[int, int]) -> set[tuple[int, int]]:
    row0, col0 = start
    max_row, max_col = grid
    if row0 < 0 or col0 < 0 or row0 + height > max_row or col0 + width > max_col:
        return set()
    return {(row, col) for row in range(row0, row0 + height) for col in range(col0, col0 + width)}


def mean_region(latent: torch.Tensor, cells: set[tuple[int, int]], channel: int = 0) -> float:
    if not cells:
        return float("nan")
    values = torch.stack([latent[int(channel), row, col].float() for row, col in sorted(cells)])
    return float(values.mean().item())


def deterministic_task_selection(
    records: Sequence[Mapping[str, Any]],
    max_states_per_task: int,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        task = str(record.get("task_type", "unknown"))
        state = str(record.get("state_ref", ""))
        if state and state not in grouped[task]:
            grouped[task][state] = dict(record)
    selected: dict[str, list[dict[str, Any]]] = {}
    for task, by_state in sorted(grouped.items()):
        values = list(by_state.values())
        rng = random.Random(int(seed) + sum(ord(char) for char in task))
        rng.shuffle(values)
        selected[task] = values[: max(0, int(max_states_per_task))]
    return selected


def _bootstrap_ci(
    values: Sequence[float],
    clusters: Sequence[str],
    reps: int,
    seed: int,
) -> dict[str, Any]:
    finite = [(float(value), str(cluster)) for value, cluster in zip(values, clusters) if math.isfinite(float(value))]
    if not finite:
        return {"n": 0, "mean": None, "ci95": [None, None], "clusters": 0}
    by_cluster: dict[str, list[float]] = defaultdict(list)
    for value, cluster in finite:
        by_cluster[cluster].append(value)
    cluster_values = torch.tensor(
        [sum(items) / len(items) for items in by_cluster.values()], dtype=torch.float64
    )
    mean = float(cluster_values.mean().item())
    if cluster_values.numel() < 2 or int(reps) <= 0:
        return {"n": len(finite), "mean": mean, "ci95": [None, None], "clusters": int(cluster_values.numel())}
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    indices = torch.randint(
        low=0,
        high=int(cluster_values.numel()),
        size=(int(reps), int(cluster_values.numel())),
        generator=generator,
    )
    samples = cluster_values[indices].mean(dim=1)
    bounds = torch.quantile(samples, torch.tensor([0.025, 0.975], dtype=torch.float64))
    return {
        "n": len(finite),
        "mean": mean,
        "ci95": [float(bounds[0].item()), float(bounds[1].item())],
        "clusters": int(cluster_values.numel()),
    }


def _ranks(scores: torch.Tensor, target_index: int) -> tuple[float, float, float, float]:
    flat = scores.detach().float().reshape(-1)
    target = float(flat[int(target_index)].item())
    greater = float((flat > target + 1.0e-12).sum().item())
    tied = float((flat.sub(target).abs() <= 1.0e-12).sum().item())
    rank = 1.0 + greater + 0.5 * max(0.0, tied - 1.0)
    top1 = 1.0 if rank <= 1.0 else 0.0
    top5 = 1.0 if rank <= min(5, flat.numel()) else 0.0
    percentile = 1.0 if flat.numel() <= 1 else 1.0 - (rank - 1.0) / float(flat.numel() - 1)
    return rank, top1, top5, percentile


def _entropy(scores: torch.Tensor) -> float:
    values = scores.detach().float().abs().reshape(-1)
    total = float(values.sum().item())
    if total <= 1.0e-12 or values.numel() <= 1:
        return 1.0
    probabilities = values / values.sum()
    entropy = -(probabilities * probabilities.clamp_min(1.0e-12).log()).sum()
    return float((entropy / math.log(values.numel())).item())


def _choice_index(labels: Sequence[str], answer: str) -> int:
    try:
        return list(labels).index(str(answer))
    except ValueError as exc:
        raise ValueError(f"Answer {answer!r} is absent from choices={list(labels)!r}.") from exc


def _candidate_logits_from_vocab(
    vocab_logits: torch.Tensor,
    records: Sequence[Mapping[str, Any]],
    tokenizer: Any,
) -> tuple[torch.Tensor, list[list[str]]]:
    specs = single_token_choice_ids(records, tokenizer)
    if specs is None:
        raise ValueError("Every diagnostic choice must tokenize as one distinct token.")
    token_ids_by_record, _ = specs
    lengths = {len(item) for item in token_ids_by_record}
    if len(lengths) != 1:
        raise ValueError("A scored batch must contain the same number of choices per record.")
    labels = [choice_labels(record) for record in records]
    ids = torch.tensor(token_ids_by_record, dtype=torch.long, device=vocab_logits.device)
    rows = torch.arange(len(records), device=vocab_logits.device).unsqueeze(1)
    return vocab_logits[rows, ids], labels


def audit_formal_choice_tokenization(
    records: Sequence[Mapping[str, Any]],
    tokenizer: Any,
) -> dict[str, Any]:
    """Make the single-token restricted-logit precondition explicit in the report."""

    labels = sorted({label for record in records for label in choice_labels(record)})
    token_ids: dict[str, list[int]] = {}
    for label in labels:
        encoded = tokenizer(" " + str(label), add_special_tokens=False, truncation=False)["input_ids"]
        token_ids[str(label)] = [int(value) for value in encoded]
    collisions: dict[int, list[str]] = defaultdict(list)
    for label, ids in token_ids.items():
        if len(ids) == 1:
            collisions[ids[0]].append(label)
    duplicate_ids = {str(token_id): values for token_id, values in collisions.items() if len(values) > 1}
    invalid_labels = {label: ids for label, ids in token_ids.items() if len(ids) != 1}
    result = {
        "labels": labels,
        "token_ids": token_ids,
        "invalid_labels": invalid_labels,
        "duplicate_token_ids": duplicate_ids,
        "all_single_distinct": not invalid_labels and not duplicate_ids,
    }
    if not bool(result["all_single_distinct"]):
        raise ValueError(
            "Formal restricted-choice scoring requires one distinct token per answer label; "
            f"invalid={invalid_labels}, duplicate_ids={duplicate_ids}."
        )
    return result


def _prepare_text_inputs(
    llm: nn.Module,
    tokenizer: Any,
    prompts: Sequence[str],
    max_prompt_tokens: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    encoded = tokenizer(
        list(prompts),
        padding=True,
        truncation=False,
        return_tensors="pt",
        add_special_tokens=True,
    )
    if int(encoded["input_ids"].shape[1]) > int(max_prompt_tokens):
        raise ValueError(
            f"A diagnostic prompt uses {int(encoded['input_ids'].shape[1])} tokens, "
            f"exceeding max_prompt_tokens={int(max_prompt_tokens)}."
        )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    text_embeds = llm.get_input_embeddings()(input_ids)
    return input_ids, attention_mask, text_embeds


def formal_candidate_logits(
    llm: nn.Module,
    adapter: nn.Module,
    tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    latent_map: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    requires_grad: bool = False,
) -> tuple[torch.Tensor, list[list[str]]]:
    """Return the same prompt-boundary restricted logits used by formal eval."""

    prompts = [build_prompt(record, prompt_template=str(args.prompt_template)) for record in records]
    input_ids, text_attention_mask, text_embeds = _prepare_text_inputs(
        llm,
        tokenizer,
        prompts,
        max_prompt_tokens=int(args.max_prompt_tokens),
        device=device,
    )
    prompt_mask = text_attention_mask.bool()
    latent_map = latent_map.to(device=device, non_blocking=True)
    soft_embeds = contextual_adapter_soft_embeds(
        llm=llm,
        adapter=adapter,
        tokenizer=tokenizer,
        records=records,
        latent_map=latent_map,
        device=device,
        max_prompt_tokens=int(args.max_prompt_tokens),
        layer_index=int(args.local_context_layer),
        mode="correct",
        prompt_template=str(args.prompt_template),
    )
    if soft_embeds is None:
        soft_embeds = adapter_soft_embeds(
            adapter=adapter,
            latent_map=latent_map,
            text_embeds=text_embeds,
            question_embeds=text_embeds,
            question_mask=prompt_mask,
            records=records,
            mode="correct",
        )
    soft_embeds = soft_embeds.to(device=device, dtype=text_embeds.dtype)
    inputs_embeds = torch.cat([soft_embeds, text_embeds], dim=1)
    soft_attention = grounded_soft_prompt_attention_mask(
        adapter,
        soft_embeds,
        mode="correct",
        dtype=text_attention_mask.dtype,
    )
    attention_mask = torch.cat([soft_attention, text_attention_mask], dim=1)
    decoder = _decoder_for_diagnostics(llm)
    output_embeddings = llm.get_output_embeddings()
    if output_embeddings is None:
        raise ValueError("The frozen LLM has no output embedding/lm head.")
    # The caller controls grad mode.  Keeping this function undecorated is
    # important for the latent gradient scan below.
    outputs = decoder(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    last_indices = last_nonpadding_indices(text_attention_mask) + int(soft_embeds.shape[1])
    batch_indices = torch.arange(input_ids.shape[0], device=device)
    next_hidden = outputs.last_hidden_state[batch_indices, last_indices]
    vocab_logits = output_embeddings(next_hidden).float()
    return _candidate_logits_from_vocab(vocab_logits, records, tokenizer)


def text_control_prompt(record: Mapping[str, Any], extra_text: str = "") -> str:
    query = re.sub(
        r"^\s*The tensor soft tokens encode\s+",
        "The numerical facts below concern ",
        record_query(record),
        count=1,
        flags=re.IGNORECASE,
    )
    labels = choice_labels(record)
    choices = ", ".join(labels)
    extra = f"\n{extra_text.strip()}" if extra_text.strip() else ""
    return (
        "Use only the numerical information written in this prompt to answer the query.\n"
        f"Query: {query}{extra}\n"
        f"Choices: {choices}\n"
        f"Required output: exactly one of {', '.join(labels)}. Output only that label.\n"
        "Answer:"
    )


@torch.no_grad()
def text_control_candidate_logits(
    llm: nn.Module,
    tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
    prompt_extras: Sequence[str] | None = None,
) -> tuple[torch.Tensor, list[list[str]], list[int]]:
    extras = list(prompt_extras or [""] * len(records))
    if len(extras) != len(records):
        raise ValueError("prompt_extras must have one entry per record.")
    prompts = [text_control_prompt(record, extra_text=extra) for record, extra in zip(records, extras)]
    input_ids, attention_mask, _text_embeds = _prepare_text_inputs(
        llm,
        tokenizer,
        prompts,
        max_prompt_tokens=int(args.text_control_max_prompt_tokens),
        device=device,
    )
    text_embeds = llm.get_input_embeddings()(input_ids)
    decoder = _decoder_for_diagnostics(llm)
    output_embeddings = llm.get_output_embeddings()
    if output_embeddings is None:
        raise ValueError("The frozen LLM has no output embedding/lm head.")
    outputs = decoder(
        inputs_embeds=text_embeds,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    last_indices = last_nonpadding_indices(attention_mask)
    rows = torch.arange(input_ids.shape[0], device=device)
    vocab_logits = output_embeddings(outputs.last_hidden_state[rows, last_indices]).float()
    candidate_logits, labels = _candidate_logits_from_vocab(vocab_logits, records, tokenizer)
    token_counts = [int(mask.sum().item()) for mask in attention_mask]
    return candidate_logits, labels, token_counts


def prediction_row(
    record: Mapping[str, Any],
    logits: torch.Tensor,
    labels: Sequence[str],
    *,
    prefix: str = "",
) -> dict[str, Any]:
    values = logits.detach().float().cpu().tolist()
    answer = str(record.get("answer", ""))
    target_index = _choice_index(labels, answer)
    predicted_index = int(torch.argmax(logits.detach()).item())
    wrong = torch.cat([logits[:target_index], logits[target_index + 1 :]])
    margin = float((logits[target_index] - wrong.max()).detach().float().cpu().item()) if wrong.numel() else 0.0
    return {
        "qa_id": f"{prefix}{record.get('qa_id', '')}",
        "state_ref": str(record.get("state_ref", "")),
        "field": str(record.get("field") or record.get("metadata", {}).get("field", "")),
        "task": str(record.get("task_type", "")),
        "answer": answer,
        "predicted": str(labels[predicted_index]),
        "correct": bool(predicted_index == target_index),
        "margin": margin,
        "labels": list(labels),
        "logits": {str(label): float(value) for label, value in zip(labels, values)},
    }


def score_records(
    llm: nn.Module,
    adapter: nn.Module,
    tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    latents: Sequence[torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, Any]]:
    if len(records) != len(latents):
        raise ValueError("records and latents have different lengths.")
    result: list[dict[str, Any] | None] = [None] * len(records)
    by_choice_count: dict[int, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_choice_count[len(choice_labels(record))].append(index)
    with torch.no_grad():
        for choice_count, indices in sorted(by_choice_count.items()):
            for start in range(0, len(indices), max(1, int(args.score_batch_size))):
                batch_indices = indices[start : start + max(1, int(args.score_batch_size))]
                batch_records = [records[index] for index in batch_indices]
                batch_latents = torch.stack([latents[index] for index in batch_indices], dim=0)
                logits, labels = formal_candidate_logits(
                    llm,
                    adapter,
                    tokenizer,
                    batch_records,
                    batch_latents,
                    args,
                    device,
                    requires_grad=False,
                )
                for row, index in enumerate(batch_indices):
                    result[index] = prediction_row(batch_records[row], logits[row], labels[row])
    if any(item is None for item in result):
        missing = sum(item is None for item in result)
        raise RuntimeError(f"Formal scoring dropped {missing} record(s) instead of returning one row per input.")
    return [item for item in result if item is not None]


def run_formal_baseline_profile(
    dataset: TensorReadoutQADataset,
    llm: nn.Module,
    adapter: nn.Module,
    tokenizer: Any,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    """Compare clean, zero-latent, and different-state latent restricted logits."""

    selected = deterministic_task_selection(
        dataset.records,
        max_states_per_task=int(args.max_states_per_task),
        seed=int(args.seed) + 61,
    )
    records: list[dict[str, Any]] = []
    latents: list[torch.Tensor] = []
    record_indices: list[int] = []
    by_qa_id = {str(record.get("qa_id", "")): index for index, record in enumerate(dataset.records)}
    for task_records in selected.values():
        for record in task_records:
            records.append(record)
            latents.append(dataset.load_latent_for_record(record))
            record_indices.append(by_qa_id[str(record.get("qa_id", ""))])
    if not records:
        return {"n_records": 0, "warning": "No records were available for the baseline profile."}
    zero_latents = [torch.zeros_like(latent) for latent in latents]
    shuffled_latents = [dataset.load_shuffled_latent(index) for index in record_indices]
    clean_rows = score_records(llm, adapter, tokenizer, records, latents, args, device)
    zero_rows = score_records(llm, adapter, tokenizer, records, zero_latents, args, device)
    shuffled_rows = score_records(llm, adapter, tokenizer, records, shuffled_latents, args, device)
    clean_values = [1.0 if bool(row["correct"]) else 0.0 for row in clean_rows]
    zero_values = [1.0 if bool(row["correct"]) else 0.0 for row in zero_rows]
    shuffled_values = [1.0 if bool(row["correct"]) else 0.0 for row in shuffled_rows]
    state_refs = [str(record.get("state_ref", "")) for record in records]
    paired_rows = [
        {
            "qa_id": str(record.get("qa_id", "")),
            "state_ref": str(record.get("state_ref", "")),
            "task": str(record.get("task_type", "")),
            "clean_prediction": str(clean["predicted"]),
            "shuffled_prediction": str(shuffled["predicted"]),
            "prediction_changed": bool(clean["predicted"] != shuffled["predicted"]),
            "clean_correct": bool(clean["correct"]),
            "shuffled_correct": bool(shuffled["correct"]),
            "margin_change": float(shuffled["margin"] - clean["margin"]),
            "candidate_kl_clean_to_shuffled": _categorical_kl(clean, shuffled),
        }
        for record, clean, shuffled in zip(records, clean_rows, shuffled_rows)
    ]
    by_task: dict[str, Any] = {}
    for task in sorted(selected):
        indices = [index for index, record in enumerate(records) if str(record.get("task_type", "")) == task]
        task_states = [state_refs[index] for index in indices]
        by_task[task] = {
            "n_records": len(indices),
            "clean_accuracy": _cluster_metric([clean_values[index] for index in indices], task_states, args, 601),
            "zero_latent_accuracy": _cluster_metric([zero_values[index] for index in indices], task_states, args, 602),
            "shuffled_accuracy": _cluster_metric(
                [shuffled_values[index] for index in indices], task_states, args, 607
            ),
            "latent_gain": _cluster_metric(
                [clean_values[index] - zero_values[index] for index in indices], task_states, args, 603
            ),
            "clean_minus_shuffled_accuracy": _cluster_metric(
                [clean_values[index] - shuffled_values[index] for index in indices], task_states, args, 608
            ),
        }
    return {
        "scope": "formal_restricted_logits",
        "n_records": len(records),
        "clean_accuracy": _cluster_metric(clean_values, state_refs, args, 604),
        "zero_latent_accuracy": _cluster_metric(zero_values, state_refs, args, 605),
        "shuffled_accuracy": _cluster_metric(shuffled_values, state_refs, args, 609),
        "latent_gain": _cluster_metric(
            [clean - zero for clean, zero in zip(clean_values, zero_values)], state_refs, args, 606
        ),
        "clean_minus_shuffled_accuracy": _cluster_metric(
            [clean - shuffled for clean, shuffled in zip(clean_values, shuffled_values)], state_refs, args, 610
        ),
        "paired_clean_vs_shuffled": {
            "prediction_changed_rate": _cluster_metric(
                [1.0 if row["prediction_changed"] else 0.0 for row in paired_rows],
                [str(row["state_ref"]) for row in paired_rows],
                args,
                611,
            ),
            "margin_change": _cluster_metric(
                [float(row["margin_change"]) for row in paired_rows],
                [str(row["state_ref"]) for row in paired_rows],
                args,
                612,
            ),
            "candidate_kl_clean_to_shuffled": _cluster_metric(
                [float(row["candidate_kl_clean_to_shuffled"] or 0.0) for row in paired_rows],
                [str(row["state_ref"]) for row in paired_rows],
                args,
                613,
            ),
            "examples": paired_rows[: min(20, len(paired_rows))],
        },
        "by_task": by_task,
        "warning": "Zero-latent measures textual/positional shortcut capacity; shuffled is a different-state latent control.",
    }


def _official_shuffled_stats_record(
    record: Mapping[str, Any],
    donor_record: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Reproduce the training evaluator's raw ``shuffled_stats`` question."""

    if str(record.get("task_type", "")) != "raw_point_value_with_stats":
        return None
    prompt_data = record.get("prompt_data")
    donor_data = donor_record.get("prompt_data")
    if not isinstance(prompt_data, Mapping) or not isinstance(donor_data, Mapping):
        return None
    if donor_data.get("mean") is None or donor_data.get("scale", donor_data.get("std")) is None:
        return None
    required = ("field", "row", "col", "option_text")
    if any(prompt_data.get(key) is None for key in required):
        return None
    digits = int(prompt_data.get("significant_digits", 6))
    patch_size = int(prompt_data.get("patch_size", 16))
    mean = float(donor_data.get("mean"))
    scale = float(donor_data.get("scale", donor_data.get("std")))
    if not math.isfinite(mean) or not math.isfinite(scale) or abs(scale) < 1.0e-12:
        return None
    question = (
        f"The tensor soft tokens encode the per-patch standardized {patch_size} by {patch_size} matrix z "
        f"of {prompt_data['field']}. Recover an original value with x = mean + scale * z, "
        f"where mean is {mean:.{digits}g} and scale is {scale:.{digits}g}. "
        "Which option is closest to the "
        f"original value x at row {int(prompt_data['row'])}, column {int(prompt_data['col'])}? "
        f"Options: {prompt_data['option_text']}."
    )
    changed = copy.deepcopy(dict(record))
    changed["query"] = question
    changed["question"] = question
    changed["qa_id"] = f"{record.get('qa_id', '')}__shuffled_stats"
    changed_prompt_data = dict(prompt_data)
    changed_prompt_data["mean"] = mean
    changed_prompt_data["scale"] = scale
    changed["prompt_data"] = changed_prompt_data
    return changed


def _categorical_kl(left: Mapping[str, Any], right: Mapping[str, Any]) -> float | None:
    left_logits = left.get("logits", {})
    right_logits = right.get("logits", {})
    if not isinstance(left_logits, Mapping) or not isinstance(right_logits, Mapping):
        return None
    labels = [label for label in left_logits if label in right_logits]
    if not labels:
        return None
    left_values = torch.tensor([float(left_logits[label]) for label in labels], dtype=torch.float64)
    right_values = torch.tensor([float(right_logits[label]) for label in labels], dtype=torch.float64)
    left_log_probs = torch.log_softmax(left_values, dim=0)
    right_log_probs = torch.log_softmax(right_values, dim=0)
    return float((left_log_probs.exp() * (left_log_probs - right_log_probs)).sum().item())


def run_stats_swap_control(
    dataset: TensorReadoutQADataset,
    llm: nn.Module,
    adapter: nn.Module,
    tokenizer: Any,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    """Pair clean raw questions with the evaluator's mean/scale-swap control."""

    selected = deterministic_task_selection(
        dataset.records,
        max_states_per_task=int(args.max_states_per_task),
        seed=int(args.seed) + 67,
    )
    by_qa_id = {str(record.get("qa_id", "")): index for index, record in enumerate(dataset.records)}
    clean_records: list[dict[str, Any]] = []
    changed_records: list[dict[str, Any]] = []
    latents: list[torch.Tensor] = []
    skipped: dict[str, int] = defaultdict(int)
    for record in selected.get("raw_point_value_with_stats", []):
        index = by_qa_id.get(str(record.get("qa_id", "")))
        if index is None:
            skipped["record_not_found"] += 1
            continue
        donor = dataset.shuffled_record_for_index(index)
        changed = _official_shuffled_stats_record(record, donor)
        if changed is None:
            skipped["missing_prompt_data"] += 1
            continue
        clean_records.append(dict(record))
        changed_records.append(changed)
        latents.append(dataset.load_latent_for_record(record))
    if not clean_records:
        return {
            "scope": "formal_raw_shuffled_stats_pair",
            "n_records": 0,
            "candidate_records": len(selected.get("raw_point_value_with_stats", [])),
            "warning": "No raw records contained the prompt_data needed to reproduce shuffled_stats.",
            "skipped": dict(skipped),
        }
    clean_rows = score_records(llm, adapter, tokenizer, clean_records, latents, args, device)
    changed_rows = score_records(llm, adapter, tokenizer, changed_records, latents, args, device)
    rows: list[dict[str, Any]] = []
    state_refs: list[str] = []
    for clean, changed, clean_row, changed_row in zip(clean_records, changed_records, clean_rows, changed_rows):
        state_ref = str(clean.get("state_ref", ""))
        state_refs.append(state_ref)
        rows.append(
            {
                "qa_id": str(clean.get("qa_id", "")),
                "state_ref": state_ref,
                "field": str(clean.get("field") or clean.get("metadata", {}).get("field", "")),
                "clean_prediction": str(clean_row["predicted"]),
                "swapped_stats_prediction": str(changed_row["predicted"]),
                "prediction_changed": bool(clean_row["predicted"] != changed_row["predicted"]),
                "clean_correct": bool(clean_row["correct"]),
                "swapped_stats_correct": bool(changed_row["correct"]),
                "clean_margin": float(clean_row["margin"]),
                "swapped_stats_margin": float(changed_row["margin"]),
                "margin_change": float(changed_row["margin"] - clean_row["margin"]),
                "candidate_kl_clean_to_swapped_stats": _categorical_kl(clean_row, changed_row),
            }
        )
    changed_states = [str(row["state_ref"]) for row in rows]
    return {
        "scope": "formal_raw_shuffled_stats_pair",
        "warning": (
            "This exactly follows the training evaluator's shuffled_stats control: latent and options stay fixed, "
            "only the natural-language mean/scale are taken from another state. It is a semantic sensitivity test, "
            "not a valid counterfactual label task."
        ),
        "n_records": len(rows),
        "candidate_records": len(selected.get("raw_point_value_with_stats", [])),
        "skipped": dict(skipped),
        "clean_accuracy": _cluster_metric(
            [1.0 if row["clean_correct"] else 0.0 for row in rows], changed_states, args, 621
        ),
        "swapped_stats_accuracy": _cluster_metric(
            [1.0 if row["swapped_stats_correct"] else 0.0 for row in rows], changed_states, args, 622
        ),
        "accuracy_change": _cluster_metric(
            [float(row["swapped_stats_correct"]) - float(row["clean_correct"]) for row in rows],
            changed_states,
            args,
            623,
        ),
        "prediction_changed_rate": _cluster_metric(
            [1.0 if row["prediction_changed"] else 0.0 for row in rows], changed_states, args, 624
        ),
        "margin_change": _cluster_metric([float(row["margin_change"]) for row in rows], changed_states, args, 625),
        "candidate_kl_clean_to_swapped_stats": _cluster_metric(
            [float(row["candidate_kl_clean_to_swapped_stats"] or 0.0) for row in rows], changed_states, args, 626
        ),
        "by_field": {
            field: {
                "n_records": sum(str(row["field"]) == field for row in rows),
                "prediction_changed_rate": _cluster_metric(
                    [1.0 if row["prediction_changed"] else 0.0 for row in rows if str(row["field"]) == field],
                    [str(row["state_ref"]) for row in rows if str(row["field"]) == field],
                    args,
                    627,
                ),
            }
            for field in sorted({str(row["field"]) for row in rows})
        },
        "examples": rows[: min(20, len(rows))],
    }


def _logit(row: Mapping[str, Any], label: str) -> float:
    values = row.get("logits", {})
    if not isinstance(values, Mapping) or label not in values:
        return float("nan")
    return float(values[label])


def _prediction(row: Mapping[str, Any]) -> str:
    return str(row.get("predicted", ""))


def _row_label_margin(row: Mapping[str, Any], label: str) -> float:
    logits = row.get("logits", {})
    if not isinstance(logits, Mapping) or label not in logits:
        return float("nan")
    wrong = [float(value) for key, value in logits.items() if str(key) != str(label)]
    return float(logits[label]) - max(wrong) if wrong else float(logits[label])


def _opposite_binary(label: str) -> str:
    return "B" if str(label) == "A" else "A"


def _cluster_metric(values: Sequence[float], state_refs: Sequence[str], args: argparse.Namespace, seed: int) -> dict[str, Any]:
    return _bootstrap_ci(values, state_refs, int(args.bootstrap_reps), int(args.seed) + int(seed))


def _finite_mean(values: Sequence[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(sum(finite) / len(finite)) if finite else None


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    pairs = [
        (float(x), float(y))
        for x, y in zip(left, right)
        if math.isfinite(float(x)) and math.isfinite(float(y))
    ]
    if len(pairs) < 2:
        return None
    x = torch.tensor([item[0] for item in pairs], dtype=torch.float64)
    y = torch.tensor([item[1] for item in pairs], dtype=torch.float64)
    x = x - x.mean()
    y = y - y.mean()
    denominator = x.square().sum().sqrt() * y.square().sum().sqrt()
    if float(denominator.item()) <= 1.0e-12:
        return None
    return float(((x * y).sum() / denominator).item())


def _pair_summary(
    pair_rows: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
    *,
    kind: str,
) -> dict[str, Any]:
    state_refs = [str(row.get("state_ref", "")) for row in pair_rows]
    original_correct = [1.0 if bool(row.get("original_correct")) else 0.0 for row in pair_rows]
    swapped_correct = [1.0 if bool(row.get("swapped_correct")) else 0.0 for row in pair_rows]
    both_correct = [1.0 if bool(row.get("both_correct")) else 0.0 for row in pair_rows]
    flip_consistent = [1.0 if bool(row.get("answer_flip_consistent")) else 0.0 for row in pair_rows]
    residual = [float(row.get("anti_symmetry_abs", float("nan"))) for row in pair_rows]
    d_original = [float(row.get("d_ab_original", float("nan"))) for row in pair_rows]
    d_swapped = [float(row.get("d_ab_swapped", float("nan"))) for row in pair_rows]
    by_field: dict[str, Any] = {}
    for field in sorted({str(row.get("field", "unknown")) for row in pair_rows}):
        field_rows = [row for row in pair_rows if str(row.get("field", "unknown")) == field]
        field_states = [str(row.get("state_ref", "")) for row in field_rows]
        by_field[field] = {
            "n_pairs": len(field_rows),
            "both_correct": _cluster_metric(
                [1.0 if bool(row.get("both_correct")) else 0.0 for row in field_rows],
                field_states,
                args,
                106,
            ),
            "answer_flip_consistency": _cluster_metric(
                [1.0 if bool(row.get("answer_flip_consistent")) else 0.0 for row in field_rows],
                field_states,
                args,
                107,
            ),
        }
    return {
        "kind": kind,
        "n_pairs": len(pair_rows),
        "filtered": {
            "min_gap": float(args.min_point_gap if kind == "point_compare" else args.min_region_gap),
        },
        "original_accuracy": _cluster_metric(original_correct, state_refs, args, 101),
        "swapped_accuracy": _cluster_metric(swapped_correct, state_refs, args, 102),
        "both_correct": _cluster_metric(both_correct, state_refs, args, 103),
        "answer_flip_consistency": _cluster_metric(flip_consistent, state_refs, args, 104),
        "anti_symmetry_abs_residual": _cluster_metric(residual, state_refs, args, 105),
        "anti_symmetry_correlation": _pearson(d_original, [-value for value in d_swapped]),
        "by_field": by_field,
        "examples": list(pair_rows[: min(12, len(pair_rows))]),
    }


def run_coordinate_swaps(
    dataset: TensorReadoutQADataset,
    llm: nn.Module,
    adapter: nn.Module,
    tokenizer: Any,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    """Score same-tensor role swaps without injecting oracle information.

    For each held-out state we retain the original latent and replace only the
    coordinate names in the question.  The expected A/B answer is flipped in the
    synthetic question.  This is the cleanest available in-distribution routing
    test for the direct adapter.
    """

    selected = deterministic_task_selection(
        dataset.records,
        max_states_per_task=int(args.max_states_per_task),
        seed=int(args.seed),
    )
    result: dict[str, Any] = {"scope": "same_tensor_role_swap", "tasks": {}}
    for task in ("point_compare", "region_mean_compare"):
        candidates = selected.get(task, [])
        originals: list[dict[str, Any]] = []
        swaps: list[dict[str, Any]] = []
        latents: list[torch.Tensor] = []
        metadata: list[dict[str, Any]] = []
        rejected = {"parse": 0, "out_of_bounds": 0, "overlap": 0, "small_gap": 0}
        for record in candidates:
            swapped_pair = make_role_swap_record(record)
            if swapped_pair is None:
                rejected["parse"] += 1
                continue
            latent = dataset.load_latent_for_record(record)
            grid = (int(latent.shape[-2]), int(latent.shape[-1]))
            if task == "point_compare":
                coordinates = parse_coordinates(record)
                if len(coordinates) != 2:
                    rejected["parse"] += 1
                    continue
                if any(not (0 <= row < grid[0] and 0 <= col < grid[1]) for row, col in coordinates):
                    rejected["out_of_bounds"] += 1
                    continue
                gap = abs(latent_value(latent, coordinates[0]) - latent_value(latent, coordinates[1]))
                if gap < float(args.min_point_gap):
                    rejected["small_gap"] += 1
                    continue
                descriptor = {
                    "coordinates": [list(coordinate) for coordinate in coordinates],
                    "gap_channel0": float(gap),
                }
            else:
                specs = parse_region_specs(record)
                if specs is None:
                    rejected["parse"] += 1
                    continue
                start_a, start_b, height, width = specs
                cells_a = region_cells(start_a, height, width, grid)
                cells_b = region_cells(start_b, height, width, grid)
                if not cells_a or not cells_b:
                    rejected["out_of_bounds"] += 1
                    continue
                if cells_a.intersection(cells_b):
                    rejected["overlap"] += 1
                    continue
                gap = abs(mean_region(latent, cells_a) - mean_region(latent, cells_b))
                if gap < float(args.min_region_gap):
                    rejected["small_gap"] += 1
                    continue
                descriptor = {
                    "region_a_start": list(start_a),
                    "region_b_start": list(start_b),
                    "height": int(height),
                    "width": int(width),
                    "gap_channel0_mean": float(gap),
                }
            originals.append(swapped_pair[0])
            swaps.append(swapped_pair[1])
            latents.extend([latent, latent])
            metadata.append(
                {
                    "state_ref": str(record.get("state_ref", "")),
                    "qa_id": str(record.get("qa_id", "")),
                    "field": str(record.get("field") or record.get("metadata", {}).get("field", "")),
                    **descriptor,
                }
            )
        if not originals:
            result["tasks"][task] = {"n_pairs": 0, "rejected": rejected}
            continue
        # Score both members in one pass so the formal candidate-token path and
        # decoder settings are identical for the two questions.
        scored = score_records(
            llm,
            adapter,
            tokenizer,
            [item for pair in zip(originals, swaps) for item in pair],
            latents,
            args,
            device,
        )
        pair_rows: list[dict[str, Any]] = []
        for index, descriptor in enumerate(metadata):
            original_row = scored[2 * index]
            swapped_row = scored[2 * index + 1]
            original_prediction = _prediction(original_row)
            swapped_prediction = _prediction(swapped_row)
            d_original = _logit(original_row, "A") - _logit(original_row, "B")
            d_swapped = _logit(swapped_row, "A") - _logit(swapped_row, "B")
            pair_rows.append(
                {
                    **descriptor,
                    "original_correct": bool(original_row["correct"]),
                    "swapped_correct": bool(swapped_row["correct"]),
                    "both_correct": bool(original_row["correct"] and swapped_row["correct"]),
                    "answer_flip_consistent": bool(swapped_prediction == _opposite_binary(original_prediction)),
                    "original_prediction": original_prediction,
                    "swapped_prediction": swapped_prediction,
                    "original_answer": str(original_row["answer"]),
                    "swapped_answer": str(swapped_row["answer"]),
                    "original_margin": float(original_row["margin"]),
                    "swapped_margin": float(swapped_row["margin"]),
                    "d_ab_original": float(d_original),
                    "d_ab_swapped": float(d_swapped),
                    "anti_symmetry_abs": float(abs(d_original + d_swapped)),
                }
            )
        summary = _pair_summary(pair_rows, args, kind=task)
        summary["rejected"] = rejected
        summary["candidate_states"] = len(candidates)
        result["tasks"][task] = summary
    return result


def _numeric_intervention_spec(
    record: Mapping[str, Any],
    latent: torch.Tensor,
) -> dict[str, Any] | None:
    """Return a channel-0 edit target for a numeric point question."""

    task = str(record.get("task_type", ""))
    if task not in {"normalized_point_value", "raw_point_value_with_stats"}:
        return None
    coordinates = parse_coordinates(record)
    options = parse_numeric_options(record)
    labels = choice_labels(record)
    if len(coordinates) != 1 or not options:
        return None
    coordinate = coordinates[0]
    if not (0 <= coordinate[0] < int(latent.shape[-2]) and 0 <= coordinate[1] < int(latent.shape[-1])):
        return None
    stats = parse_stats(record)
    current_z = latent_value(latent, coordinate)
    if task == "normalized_point_value":
        current_value = current_z
        option_to_z = dict(options)
        affine = None
    else:
        if stats is None or abs(float(stats[1])) < 1.0e-12:
            return None
        mean, scale = stats
        option_to_z = {label: (value - mean) / scale for label, value in options.items()}
        current_value = mean + scale * current_z
        affine = {"mean": float(mean), "scale": float(scale)}
    candidates = [label for label in labels if label in option_to_z and label != str(record.get("answer", ""))]
    if not candidates:
        candidates = [label for label in option_to_z if label != str(record.get("answer", ""))]
    if not candidates:
        return None
    target_label = max(candidates, key=lambda label: abs(float(option_to_z[label]) - current_z))
    target_z = float(option_to_z[target_label])
    if not math.isfinite(target_z):
        return None
    return {
        "task": task,
        "coordinate": tuple(int(value) for value in coordinate),
        "current_z": float(current_z),
        "current_value": float(current_value),
        "target_label": str(target_label),
        "target_z": target_z,
        "target_value": float(options[target_label]),
        "original_answer": str(record.get("answer", "")),
        "options": {str(key): float(value) for key, value in options.items()},
        "affine": affine,
    }


def _edited_latent(latent: torch.Tensor, coordinate: tuple[int, int], value: float) -> torch.Tensor:
    edited = latent.detach().clone().float()
    edited[0, int(coordinate[0]), int(coordinate[1])] = float(value)
    return edited


def _choose_control_cells(
    coordinate: tuple[int, int],
    grid: tuple[int, int],
    count: int,
    seed: int,
) -> list[tuple[int, int]]:
    cells = [
        (row, col)
        for row in range(int(grid[0]))
        for col in range(int(grid[1]))
        if (row, col) != tuple(coordinate)
    ]
    rng = random.Random(int(seed))
    rng.shuffle(cells)
    # Match the target's broad boundary/interior status when possible; this
    # reduces a trivial edge-position confound in the non-target controls.
    target_edge = coordinate[0] in {0, grid[0] - 1} or coordinate[1] in {0, grid[1] - 1}
    same_boundary = [
        cell
        for cell in cells
        if (cell[0] in {0, grid[0] - 1} or cell[1] in {0, grid[1] - 1}) == target_edge
    ]
    ordered = same_boundary + [cell for cell in cells if cell not in set(same_boundary)]
    return ordered[: max(0, int(count))]


def _matched_numeric_records(
    records: Sequence[Mapping[str, Any]],
    limit: int,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    tasks = {"normalized_point_value", "raw_point_value_with_stats"}
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        task = str(record.get("task_type", ""))
        state = str(record.get("state_ref", ""))
        if task not in tasks or not state:
            continue
        key = (state, int(record.get("question_variant", 0)))
        grouped[key].setdefault(task, dict(record))
    pairs: list[tuple[str, dict[str, dict[str, Any]]]] = []
    for (state, _variant), by_task in grouped.items():
        if set(by_task) != tasks:
            continue
        if parse_coordinates(by_task["normalized_point_value"]) != parse_coordinates(
            by_task["raw_point_value_with_stats"]
        ):
            continue
        pairs.append((state, by_task))
    rng = random.Random(int(seed))
    rng.shuffle(pairs)
    selected_pairs: list[dict[str, dict[str, Any]]] = []
    selected_states: set[str] = set()
    for state, pair in pairs:
        if state in selected_states:
            continue
        selected_states.add(state)
        selected_pairs.append(pair)
        if len(selected_pairs) >= max(0, int(limit)):
            break
    return {
        task: [pair[task] for pair in selected_pairs]
        for task in sorted(tasks)
    }


def run_local_interventions(
    dataset: TensorReadoutQADataset,
    llm: nn.Module,
    adapter: nn.Module,
    tokenizer: Any,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    """Run channel-0 target/control edits as an explicitly OOD stress test."""

    selected = _matched_numeric_records(
        dataset.records,
        limit=int(args.intervention_records_per_task),
        seed=int(args.seed) + 17,
    )
    if not any(selected.values()):
        selected = deterministic_task_selection(
            dataset.records,
            max_states_per_task=int(args.intervention_records_per_task),
            seed=int(args.seed) + 17,
        )
    cases: list[dict[str, Any]] = []
    skipped: dict[str, int] = defaultdict(int)
    for task in ("normalized_point_value", "raw_point_value_with_stats"):
        for record in selected.get(task, []):
            latent = dataset.load_latent_for_record(record)
            spec = _numeric_intervention_spec(record, latent)
            if spec is None:
                skipped["unparseable_or_invalid"] += 1
                continue
            coordinate = spec["coordinate"]
            target_latent = _edited_latent(latent, coordinate, spec["target_z"])
            delta = float(spec["target_z"] - spec["current_z"])
            controls = _choose_control_cells(
                coordinate,
                (int(latent.shape[-2]), int(latent.shape[-1])),
                int(args.intervention_controls_per_record),
                seed=int(args.seed) + sum((index + 1) * ord(char) for index, char in enumerate(str(record.get("qa_id", "")))) % 100000,
            )
            cases.append(
                {
                    "record": dict(record),
                    "latent": latent,
                    "target_latent": target_latent,
                    "controls": [(cell, _edited_latent(latent, cell, float(latent_value(latent, cell) + delta))) for cell in controls],
                    "spec": spec,
                    "delta": delta,
                }
            )
    if not cases:
        return {
            "scope": "ood_latent_channel0_mechanism_stress_test",
            "warning": "No numeric point records were eligible.",
            "n_records": 0,
            "skipped": dict(skipped),
        }

    clean_records: list[dict[str, Any]] = []
    clean_latents: list[torch.Tensor] = []
    edited_records: list[dict[str, Any]] = []
    edited_latents: list[torch.Tensor] = []
    edit_meta: list[tuple[int, str, tuple[int, int] | None]] = []
    for case_index, case in enumerate(cases):
        clean_records.append(dict(case["record"]))
        clean_latents.append(case["latent"])
        target_record = dict(case["record"])
        target_record["answer"] = case["spec"]["target_label"]
        target_record["qa_id"] = f"{target_record.get('qa_id', '')}__target_edit"
        edited_records.append(target_record)
        edited_latents.append(case["target_latent"])
        edit_meta.append((case_index, "target", None))
        for control_cell, control_latent in case["controls"]:
            control_record = dict(target_record)
            control_record["qa_id"] = f"{target_record.get('qa_id', '')}__control_{control_cell[0]}_{control_cell[1]}"
            edited_records.append(control_record)
            edited_latents.append(control_latent)
            edit_meta.append((case_index, "control", control_cell))

    clean_rows = score_records(llm, adapter, tokenizer, clean_records, clean_latents, args, device)
    edited_rows = score_records(llm, adapter, tokenizer, edited_records, edited_latents, args, device)
    rows: list[dict[str, Any]] = []
    by_case: dict[int, dict[str, Any]] = defaultdict(dict)
    for case_index, row in enumerate(clean_rows):
        by_case[case_index]["clean"] = row
    for edit_index, (case_index, kind, cell) in enumerate(edit_meta):
        by_case[case_index].setdefault(kind, []).append((cell, edited_rows[edit_index]))
    target_success: list[float] = []
    target_flip_from_clean: list[float] = []
    target_flip_state_refs: list[str] = []
    false_flip: list[float] = []
    any_false_flip: list[float] = []
    any_false_flip_state_refs: list[str] = []
    false_flip_state_refs: list[str] = []
    selective_effect: list[float] = []
    eligible_selective_effect: list[float] = []
    eligible_selective_state_refs: list[str] = []
    state_refs: list[str] = []
    target_margin_change: list[float] = []
    edit_abs_delta: list[float] = []
    target_outside_observed_range: list[float] = []
    for case_index, case in enumerate(cases):
        clean = by_case[case_index]["clean"]
        target_label = str(case["spec"]["target_label"])
        target_item = by_case[case_index]["target"][0][1]
        controls = by_case[case_index].get("control", [])
        clean_target_logit = _logit(clean, target_label)
        target_logit = _logit(target_item, target_label)
        target_logit_effect = target_logit - clean_target_logit
        clean_target_margin = _row_label_margin(clean, target_label)
        target_margin_delta = float(target_item["margin"] - clean_target_margin)
        control_effects = [_logit(item, target_label) - clean_target_logit for _cell, item in controls]
        control_predictions = [str(item.get("predicted", "")) for _cell, item in controls]
        control_margin_deltas = [
            float(item["margin"] - clean_target_margin) for _cell, item in controls
        ]
        clean_prediction = str(clean.get("predicted", ""))
        eligible_flip = clean_prediction != target_label
        target_success.append(float(str(target_item.get("predicted", "")) == target_label))
        if eligible_flip:
            target_flip_from_clean.append(float(str(target_item.get("predicted", "")) == target_label))
            target_flip_state_refs.append(str(case["record"].get("state_ref", "")))
            false_flip.extend(
                float(prediction == target_label) for prediction in control_predictions
            )
            false_flip_state_refs.extend(
                [str(case["record"].get("state_ref", ""))] * len(control_predictions)
            )
            any_false_flip.append(float(any(prediction == target_label for prediction in control_predictions)))
            any_false_flip_state_refs.append(str(case["record"].get("state_ref", "")))
        case_selective_effect = target_margin_delta - (_finite_mean(control_margin_deltas) or 0.0)
        selective_effect.append(case_selective_effect)
        if eligible_flip:
            eligible_selective_effect.append(case_selective_effect)
            eligible_selective_state_refs.append(str(case["record"].get("state_ref", "")))
        target_margin_change.append(target_margin_delta)
        state_refs.append(str(case["record"].get("state_ref", "")))
        observed_values = [
            float(case["latent"][0, row, col].detach().float().cpu().item())
            for row in range(int(case["latent"].shape[-2]))
            for col in range(int(case["latent"].shape[-1]))
        ]
        target_outside_observed_range.append(
            float(
                float(case["spec"]["target_z"]) < min(observed_values)
                or float(case["spec"]["target_z"]) > max(observed_values)
            )
        )
        edit_abs_delta.append(abs(float(case["delta"])))
        rows.append(
            {
                "qa_id": str(case["record"].get("qa_id", "")),
                "state_ref": str(case["record"].get("state_ref", "")),
                "task": str(case["spec"]["task"]),
                "field": str(
                    case["record"].get("field") or case["record"].get("metadata", {}).get("field", "")
                ),
                "question_variant": int(case["record"].get("question_variant", 0)),
                "coordinate": list(case["spec"]["coordinate"]),
                "target_label": target_label,
                "original_answer": str(case["spec"]["original_answer"]),
                "current_z": float(case["spec"]["current_z"]),
                "target_z": float(case["spec"]["target_z"]),
                "delta_z": float(case["delta"]),
                "clean_prediction": str(clean["predicted"]),
                "clean_correct": bool(clean["correct"]),
                "clean_original_margin": float(clean["margin"]),
                "target_prediction": str(target_item["predicted"]),
                "target_success": bool(target_success[-1]),
                "target_logit_change": float(target_logit_effect),
                "target_margin_change": float(target_margin_change[-1]),
                "target_flip_from_clean": bool(
                    eligible_flip and str(target_item.get("predicted", "")) == target_label
                ),
                "target_flip_eligible": bool(eligible_flip),
                "control_false_flip_any": bool(any_false_flip[-1]) if eligible_flip else None,
                "control_false_flip_count": sum(prediction == target_label for prediction in control_predictions),
                "control_effect_mean": _finite_mean(control_effects),
                "control_margin_change_mean": _finite_mean(control_margin_deltas),
                "selective_effect": float(selective_effect[-1]),
                "selective_effect_eligible": float(case_selective_effect) if eligible_flip else None,
                "target_outside_observed_range": bool(target_outside_observed_range[-1]),
                "control_cells": [list(cell) for cell, _item in controls],
            }
        )
    by_task: dict[str, Any] = {}
    for task in sorted({str(row["task"]) for row in rows}):
        task_rows = [row for row in rows if str(row["task"]) == task]
        task_states = [str(row["state_ref"]) for row in task_rows]
        eligible_rows = [row for row in task_rows if row["target_flip_eligible"]]
        by_task[task] = {
            "n_records": len(task_rows),
            "n_target_flip_eligible": len(eligible_rows),
            "target_flip_eligible_fraction": float(len(eligible_rows) / max(1, len(task_rows))),
            "clean_accuracy": _cluster_metric(
                [1.0 if bool(row["clean_correct"]) else 0.0 for row in task_rows], task_states, args, 206
            ),
            "target_intended_label_success": _cluster_metric(
                [1.0 if bool(row["target_success"]) else 0.0 for row in task_rows], task_states, args, 207
            ),
            "target_flip_from_clean": _cluster_metric(
                [1.0 if bool(row["target_flip_from_clean"]) else 0.0 for row in eligible_rows],
                [str(row["state_ref"]) for row in eligible_rows],
                args,
                213,
            ),
            "selective_margin_effect": _cluster_metric(
                [float(row["selective_effect"]) for row in task_rows], task_states, args, 208
            ),
            "selective_margin_effect_eligible": _cluster_metric(
                [float(row["selective_effect"]) for row in eligible_rows],
                [str(row["state_ref"]) for row in eligible_rows],
                args,
                215,
            ),
        }
    matched_rows: dict[tuple[str, int, tuple[int, int]], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (
            str(row["state_ref"]),
            int(row["question_variant"]),
            tuple(int(value) for value in row["coordinate"]),
        )
        matched_rows[key][str(row["task"])] = row
    matched_differences: list[float] = []
    matched_both: list[float] = []
    matched_states: list[str] = []
    for (state_ref, _variant, _coordinate), task_rows in matched_rows.items():
        if "normalized_point_value" not in task_rows or "raw_point_value_with_stats" not in task_rows:
            continue
        normalized = task_rows["normalized_point_value"]
        raw = task_rows["raw_point_value_with_stats"]
        matched_differences.append(float(bool(raw["clean_correct"])) - float(bool(normalized["clean_correct"])))
        matched_both.append(float(bool(raw["clean_correct"]) and bool(normalized["clean_correct"])))
        matched_states.append(state_ref)
    by_field_task: dict[str, Any] = {}
    for field in sorted({str(row["field"]) for row in rows}):
        field_rows = [row for row in rows if str(row["field"]) == field]
        by_field_task[field] = {}
        for task in sorted({str(row["task"]) for row in field_rows}):
            task_rows = [row for row in field_rows if str(row["task"]) == task]
            task_states = [str(row["state_ref"]) for row in task_rows]
            eligible_rows = [row for row in task_rows if row["target_flip_eligible"]]
            by_field_task[field][task] = {
                "n_records": len(task_rows),
                "n_target_flip_eligible": len(eligible_rows),
                "clean_accuracy": _cluster_metric(
                    [1.0 if bool(row["clean_correct"]) else 0.0 for row in task_rows],
                    task_states,
                    args,
                    211,
                ),
                "selective_margin_effect": _cluster_metric(
                    [float(row["selective_effect"]) for row in task_rows],
                    task_states,
                    args,
                    212,
                ),
                "target_flip_from_clean": _cluster_metric(
                    [1.0 if bool(row["target_flip_from_clean"]) else 0.0 for row in eligible_rows],
                    [str(row["state_ref"]) for row in eligible_rows],
                    args,
                    219,
                ),
            }
    return {
        "scope": "ood_latent_channel0_mechanism_stress_test",
        "warning": (
            "Edits change one normalized channel while keeping the other latent channels and prompt text fixed; "
            "this is an out-of-distribution mechanism stress test, not a distribution-internal causal estimate. "
            "Interpret channel 0 as a scalar-value edit only when the preserve_input_channels contract is true."
        ),
        "n_records": len(rows),
        "skipped": dict(skipped),
        "target_intended_label_success": _cluster_metric(target_success, state_refs, args, 201),
        "target_flip_from_clean": _cluster_metric(target_flip_from_clean, target_flip_state_refs, args, 214),
        "target_flip_eligible_n": len(target_flip_from_clean),
        "target_flip_eligible_fraction": float(len(target_flip_from_clean) / max(1, len(rows))),
        "non_target_false_flip_rate": _cluster_metric(false_flip, false_flip_state_refs, args, 202),
        "non_target_any_false_flip_per_record": _cluster_metric(
            any_false_flip, any_false_flip_state_refs, args, 205
        ),
        "target_vs_control_selective_margin_effect": _cluster_metric(selective_effect, state_refs, args, 203),
        "target_vs_control_selective_margin_effect_eligible": _cluster_metric(
            eligible_selective_effect, eligible_selective_state_refs, args, 216
        ),
        "target_margin_change": _cluster_metric(target_margin_change, state_refs, args, 204),
        "edit_abs_delta_z": _cluster_metric(edit_abs_delta, state_refs, args, 217),
        "target_outside_observed_range": _cluster_metric(
            target_outside_observed_range, state_refs, args, 218
        ),
        "by_task": by_task,
        "by_field_task": by_field_task,
        "matched_normalized_raw": {
            "n_pairs": len(matched_states),
            "raw_minus_normalized_accuracy": _cluster_metric(
                matched_differences, matched_states, args, 209
            ),
            "both_correct": _cluster_metric(matched_both, matched_states, args, 210),
        },
        "examples": rows[: min(20, len(rows))],
    }


def _freeze_parameters(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def _answer_margin_tensor(logits: torch.Tensor, labels: Sequence[str], answer: str) -> torch.Tensor:
    target_index = _choice_index(labels, answer)
    wrong = torch.cat([logits[:target_index], logits[target_index + 1 :]])
    if wrong.numel() == 0:
        return logits[target_index]
    return logits[target_index] - wrong.max()


def _gradient_records(dataset: TensorReadoutQADataset, args: argparse.Namespace) -> list[dict[str, Any]]:
    selected = deterministic_task_selection(
        dataset.records,
        max_states_per_task=int(args.gradient_records_per_task),
        seed=int(args.seed) + 31,
    )
    preferred_tasks = ("normalized_point_value", "raw_point_value_with_stats", "point_bin")
    records: list[dict[str, Any]] = []
    for task in preferred_tasks:
        for record in selected.get(task, []):
            if len(parse_coordinates(record)) == 1:
                records.append(record)
    if records:
        return records
    # A small fallback keeps the diagnostic useful for custom QA task mixes.
    for task_records in selected.values():
        for record in task_records:
            if len(parse_coordinates(record)) == 1:
                records.append(record)
    return records


def _gradient_scan_one(
    llm: nn.Module,
    adapter: nn.Module,
    tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    dataset: TensorReadoutQADataset,
    args: argparse.Namespace,
    device: torch.device,
    stage_name: str,
) -> dict[str, Any]:
    _freeze_parameters(adapter)
    _freeze_parameters(llm)
    rows: list[dict[str, Any]] = []
    for record_index, record in enumerate(records, start=1):
        if record_index == 1 or record_index % 8 == 0:
            print(
                f"diagnostic=gradient_scan stage={stage_name} record={record_index}/{len(records)}",
                flush=True,
            )
        latent = dataset.load_latent_for_record(record).to(device=device, dtype=torch.float32).detach()
        latent.requires_grad_(True)
        try:
            logits, labels = formal_candidate_logits(
                llm,
                adapter,
                tokenizer,
                [record],
                latent.unsqueeze(0),
                args,
                device,
                requires_grad=True,
            )
            margin = _answer_margin_tensor(logits[0], labels[0], str(record.get("answer", "")))
            gradient = torch.autograd.grad(margin, latent, retain_graph=False, allow_unused=True)[0]
            if gradient is None:
                gradient = torch.zeros_like(latent)
            gradient = gradient.detach().float()
        except Exception as exc:  # preserve the record-level reason in the report
            rows.append(
                {
                    "qa_id": str(record.get("qa_id", "")),
                    "state_ref": str(record.get("state_ref", "")),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        coordinates = parse_coordinates(record)
        if len(coordinates) != 1:
            continue
        coordinate = coordinates[0]
        if not (
            0 <= coordinate[0] < int(gradient.shape[-2])
            and 0 <= coordinate[1] < int(gradient.shape[-1])
        ):
            continue
        scores = gradient[0].abs().reshape(-1)
        target_index = int(coordinate[0]) * int(gradient.shape[-1]) + int(coordinate[1])
        if not (0 <= target_index < int(scores.numel())):
            continue
        rank, top1, top5, percentile = _ranks(scores, target_index)
        rows.append(
            {
                "qa_id": str(record.get("qa_id", "")),
                "state_ref": str(record.get("state_ref", "")),
                "task": str(record.get("task_type", "")),
                "field": str(record.get("field") or record.get("metadata", {}).get("field", "")),
                "coordinate": list(coordinate),
                "target_abs_gradient": float(scores[target_index].item()),
                "target_gradient_mass_fraction": float(
                    scores[target_index].item() / scores.sum().clamp_min(1.0e-12).item()
                ),
                "target_signed_gradient": float(gradient[0, coordinate[0], coordinate[1]].item()),
                "target_rank": float(rank),
                "target_top1": float(top1),
                "target_top5": float(top5),
                "target_percentile": float(percentile),
                "gradient_entropy": _entropy(scores),
                "gradient_l1": float(scores.sum().item()),
                "gradient_max": float(scores.max().item()),
                "clean_margin": float(margin.detach().cpu().item()),
                "stage": stage_name,
            }
        )
    valid = [row for row in rows if "target_rank" in row]
    error_rows = [row for row in rows if "error" in row]
    state_refs = [str(row.get("state_ref", "")) for row in valid]
    by_task: dict[str, Any] = {}
    for task in sorted({str(row.get("task", "")) for row in valid}):
        task_rows = [row for row in valid if str(row.get("task", "")) == task]
        task_states = [str(row.get("state_ref", "")) for row in task_rows]
        by_task[task] = {
            "n_records": len(task_rows),
            "target_top5": _cluster_metric(
                [float(row["target_top5"]) for row in task_rows], task_states, args, 309
            ),
            "target_percentile": _cluster_metric(
                [float(row["target_percentile"]) for row in task_rows], task_states, args, 310
            ),
            "gradient_entropy": _cluster_metric(
                [float(row["gradient_entropy"]) for row in task_rows], task_states, args, 311
            ),
        }
    result = {
        "stage": stage_name,
        "n_records_attempted": len(records),
        "n_records": len(valid),
        "n_errors": len(error_rows),
        "error_rate": float(len(error_rows) / max(1, len(records))),
        "errors": error_rows[:20],
        "target_rank": _cluster_metric([float(row["target_rank"]) for row in valid], state_refs, args, 301),
        "target_top1": _cluster_metric([float(row["target_top1"]) for row in valid], state_refs, args, 302),
        "target_top5": _cluster_metric([float(row["target_top5"]) for row in valid], state_refs, args, 303),
        "target_percentile": _cluster_metric([float(row["target_percentile"]) for row in valid], state_refs, args, 304),
        "gradient_entropy": _cluster_metric([float(row["gradient_entropy"]) for row in valid], state_refs, args, 305),
        "target_signed_gradient": _cluster_metric([float(row["target_signed_gradient"]) for row in valid], state_refs, args, 306),
        "target_abs_gradient": _cluster_metric([float(row["target_abs_gradient"]) for row in valid], state_refs, args, 307),
        "target_gradient_mass_fraction": _cluster_metric(
            [float(row["target_gradient_mass_fraction"]) for row in valid], state_refs, args, 308
        ),
        "by_task": by_task,
        "examples": valid[: min(24, len(valid))],
    }
    if records and not valid:
        result["section_error"] = "gradient_scan"
        result["reason"] = (
            "Every gradient-scan record failed; no target-cell gradient was produced."
            if error_rows
            else "No gradient-scan record produced a valid in-bounds target coordinate."
        )
    elif error_rows:
        result["partial_errors"] = True
    return result


def run_gradient_scan(
    dataset: TensorReadoutQADataset,
    llm: nn.Module,
    adapter: nn.Module,
    stage1_adapter: nn.Module | None,
    tokenizer: Any,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    records = _gradient_records(dataset, args)[: max(0, int(args.gradient_records_per_task) * 3)]
    if not records:
        return {"n_records": 0, "warning": "No one-coordinate records were available."}
    result: dict[str, Any] = {"n_records_requested": len(records), "stages": {}}
    result["stages"]["stage2"] = _gradient_scan_one(
        llm, adapter, tokenizer, records, dataset, args, device, "stage2"
    )
    if isinstance(result["stages"]["stage2"], Mapping) and result["stages"]["stage2"].get("section_error"):
        result["section_error"] = "gradient_scan"
        result["reason"] = str(result["stages"]["stage2"].get("reason", "stage2 gradient scan failed"))
    if stage1_adapter is not None:
        result["stages"]["stage1"] = _gradient_scan_one(
            llm, stage1_adapter, tokenizer, records, dataset, args, device, "stage1"
        )
        if isinstance(result["stages"]["stage1"], Mapping) and result["stages"]["stage1"].get("section_error"):
            result.setdefault("warnings", []).append(
                "Stage-1 gradient scan produced no valid records; Stage-2 results remain available."
            )
        result["interpretation"] = (
            "Compare target rank/top-k and gradient entropy between stages; a lower target selectivity and higher "
            "entropy after Stage 2 is a screening flag for representation drift, not proof of a causal failure."
        )
    return result


def _text_control_variants(record: Mapping[str, Any], latent: torch.Tensor) -> list[tuple[str, str]]:
    task = str(record.get("task_type", ""))
    if task not in {"normalized_point_value", "raw_point_value_with_stats"}:
        return []
    coordinates = parse_coordinates(record)
    if len(coordinates) != 1:
        return []
    coordinate = coordinates[0]
    z_value = latent_value(latent, coordinate)
    variants: list[tuple[str, str]] = [
        (
            "explicit_target_z",
            f"Diagnostic numeric fact: standardized z at row {coordinate[0] + grid_origin(record)}, "
            f"column {coordinate[1] + grid_origin(record)} is {z_value:.8g}.",
        )
    ]
    stats = parse_stats(record)
    if task == "raw_point_value_with_stats" and stats is not None:
        mean, scale = stats
        variants.append(
            (
                "explicit_z_mean_scale_no_result",
                f"Diagnostic numeric facts: standardized z at the requested cell is {z_value:.8g}; "
                f"mean is {mean:.8g}; scale is {scale:.8g}. Compute x = mean + scale * z.",
            )
        )
    return variants


def _serialized_matrix_extra(latent: torch.Tensor) -> str:
    matrix = latent[0].detach().float().cpu()
    rows = [" ".join(f"{float(value):.5g}" for value in row) for row in matrix]
    return "Diagnostic serialized normalized channel-0 matrix (row-major):\n" + "\n".join(rows)


def run_text_controls(
    dataset: TensorReadoutQADataset,
    llm: nn.Module,
    tokenizer: Any,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    """Measure a text-only numerical upper bound with no tensor soft prefix."""

    selected = deterministic_task_selection(
        dataset.records,
        max_states_per_task=int(args.text_control_records_per_task),
        seed=int(args.seed) + 47,
    )
    extras_by_variant: dict[str, list[str]] = defaultdict(list)
    records_by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped: dict[str, int] = defaultdict(int)
    for task_records in selected.values():
        for record in task_records:
            latent = dataset.load_latent_for_record(record)
            variants = _text_control_variants(record, latent)
            for name, extra in variants:
                records_by_variant[name].append(dict(record))
                extras_by_variant[name].append(extra)
            if bool(args.run_serialized_matrix_control) and str(record.get("task_type", "")) in {
                "normalized_point_value",
                "raw_point_value_with_stats",
            }:
                records_by_variant["serialized_normalized_matrix"].append(dict(record))
                extras_by_variant["serialized_normalized_matrix"].append(_serialized_matrix_extra(latent))
            if not variants:
                skipped["no_numeric_point_variant"] += 1
    result: dict[str, Any] = {
        "scope": "text_only_no_soft_prefix_control",
        "warning": (
            "These controls provide a frozen-LLM/text-format ceiling. They do not establish that the adapter can "
            "route coordinates, because the numerical fact is written explicitly in natural language."
        ),
        "variants": {},
        "skipped": dict(skipped),
    }
    for variant, variant_records in sorted(records_by_variant.items()):
        if not variant_records:
            continue
        variant_extras = extras_by_variant[variant]
        rows: list[dict[str, Any]] = []
        token_counts: list[int] = []
        errors: list[str] = []
        for start in range(0, len(variant_records), max(1, int(args.score_batch_size))):
            batch_records = variant_records[start : start + max(1, int(args.score_batch_size))]
            batch_extras = variant_extras[start : start + max(1, int(args.score_batch_size))]
            try:
                logits, labels, counts = text_control_candidate_logits(
                    llm,
                    tokenizer,
                    batch_records,
                    args,
                    device,
                    prompt_extras=batch_extras,
                )
            except ValueError as exc:
                errors.append(str(exc))
                # A serialized matrix can exceed the bound; skip only that batch
                # and preserve the reason in the report.
                continue
            token_counts.extend(counts)
            for row_index, record in enumerate(batch_records):
                rows.append(prediction_row(record, logits[row_index], labels[row_index], prefix=f"{variant}__"))
        state_refs = [str(row.get("state_ref", "")) for row in rows]
        by_task: dict[str, Any] = {}
        for task in sorted({str(row.get("task_type", "unknown")) for row in rows}):
            task_rows = [row for row in rows if str(row.get("task_type", "unknown")) == task]
            task_states = [str(row.get("state_ref", "")) for row in task_rows]
            by_task[task] = {
                "n_records": len(task_rows),
                "accuracy": _cluster_metric(
                    [1.0 if bool(row["correct"]) else 0.0 for row in task_rows],
                    task_states,
                    args,
                    403,
                ),
                "margin": _cluster_metric(
                    [float(row["margin"]) for row in task_rows],
                    task_states,
                    args,
                    404,
                ),
            }
        result["variants"][variant] = {
            "n_records": len(rows),
            "accuracy": _cluster_metric(
                [1.0 if bool(row["correct"]) else 0.0 for row in rows], state_refs, args, 401
            ),
            "margin": _cluster_metric([float(row["margin"]) for row in rows], state_refs, args, 402),
            "prompt_tokens": {
                "n": len(token_counts),
                "mean": _finite_mean(token_counts),
                "max": max(token_counts) if token_counts else None,
            },
            "errors": errors[:8],
            "by_task": by_task,
            "examples": rows[: min(12, len(rows))],
        }
    return result


def _probe_args(args: argparse.Namespace) -> argparse.Namespace:
    probe_args = copy.copy(args)
    probe_args.value_channel = 0
    probe_args.positions_per_state = int(args.probe_positions_per_state)
    probe_args.feature_batch_size = int(args.probe_feature_batch_size)
    probe_args.linear_ridge = float(args.probe_ridge)
    probe_args.value_tolerance = float(args.probe_tolerance)
    return probe_args


def _prompt_representation_stats(prompt: torch.Tensor) -> dict[str, Any]:
    values = prompt.detach().float().cpu()
    if values.ndim != 3:
        raise ValueError(f"Expected prompt [batch,tokens,hidden], got {tuple(values.shape)}")
    centered = values - values.mean(dim=(0, 1), keepdim=True)
    token_variance = values.var(dim=-1, unbiased=False).mean()
    token_norm = values.norm(dim=-1).mean().clamp_min(1.0e-12)
    # Effective rank is computed on a bounded sample to keep this diagnostic
    # cheap even when the prompt has hundreds of tokens and a large hidden size.
    flattened_all = centered.reshape(-1, centered.shape[-1])
    sample_rows = min(128, int(flattened_all.shape[0]))
    if sample_rows:
        # Spread the bounded sample over every state and the whole spatial
        # prefix.  Taking the first rows would usually inspect only the first
        # state and could manufacture an apparent rank collapse.
        row_indices = torch.linspace(
            0,
            int(flattened_all.shape[0]) - 1,
            steps=sample_rows,
        ).round().long()
        flattened = flattened_all[row_indices]
    else:
        flattened = flattened_all
    if flattened.shape[1] > 512:
        feature_indices = torch.linspace(0, flattened.shape[1] - 1, steps=512).long()
        flattened = flattened[:, feature_indices]
    gram = flattened @ flattened.T
    singular = torch.linalg.eigvalsh(gram).clamp_min(0.0).sqrt()
    probabilities = singular / singular.sum().clamp_min(1.0e-12)
    effective_rank = float(torch.exp(-(probabilities * probabilities.clamp_min(1.0e-12).log()).sum()).item())
    normalized = values / values.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
    similarities = torch.matmul(normalized, normalized.transpose(-1, -2))
    token_count = int(similarities.shape[-1])
    if token_count > 1:
        off_diag = similarities[:, ~torch.eye(token_count, dtype=torch.bool)].reshape(values.shape[0], -1)
        off_diag_mean = float(off_diag.mean().item())
    else:
        off_diag_mean = 1.0
    return {
        "batch": int(values.shape[0]),
        "tokens": token_count,
        "hidden": int(values.shape[-1]),
        "mean_norm": float(token_norm.item()),
        "mean_token_variance": float(token_variance.item()),
        "effective_rank": effective_rank,
        "effective_rank_sample_shape": [int(flattened.shape[0]), int(flattened.shape[1])],
        "effective_rank_sampling": "evenly_spaced_across_batch_and_tokens",
        "off_diagonal_token_cosine": off_diag_mean,
    }


def _representation_difference(stage1: torch.Tensor, stage2: torch.Tensor) -> dict[str, float]:
    left = stage1.detach().float().reshape(-1)
    right = stage2.detach().float().reshape(-1)
    delta = right - left
    cosine = float(F.cosine_similarity(left.unsqueeze(0), right.unsqueeze(0)).item())
    return {
        "relative_l2": float(delta.norm().item() / left.norm().clamp_min(1.0e-12).item()),
        "cosine": cosine,
        "absolute_l2": float(delta.norm().item()),
    }


@torch.no_grad()
def _collect_prompt_representations(
    examples: Sequence[tuple[Mapping[str, Any], torch.Tensor]],
    adapter: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    chunks: dict[str, list[torch.Tensor]] = defaultdict(list)
    batch_size = max(1, int(args.probe_feature_batch_size))
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        latent = torch.stack([item[1] for item in batch]).to(device)
        stages = extract_spatial_token_stages(adapter, latent)
        for name, value in stages.items():
            if value.ndim == 3:
                chunks[name].append(value.detach().float().cpu())
    return {name: torch.cat(values, dim=0) for name, values in chunks.items()}


def run_spatial_probe(
    train_dataset: TensorReadoutQADataset,
    val_dataset: TensorReadoutQADataset,
    llm: nn.Module,
    stage2_adapter: nn.Module,
    stage1_adapter: nn.Module | None,
    tokenizer: Any,
    args: argparse.Namespace,
    device: torch.device,
    probe_device: torch.device,
) -> dict[str, Any]:
    """Auxiliary value readout and Stage-1/Stage-2 representation comparison."""

    requested_train = min(int(args.probe_train_states), len({str(r.get("state_ref", "")) for r in train_dataset.records}))
    requested_val = min(int(args.probe_val_states), len({str(r.get("state_ref", "")) for r in val_dataset.records}))
    if requested_train < 2 or requested_val < 2:
        return {"skipped": True, "reason": "Not enough unique states in the capped dataset scans."}
    train_examples = unique_state_examples(train_dataset, requested_train)
    val_examples = unique_state_examples(val_dataset, requested_val)
    probe_args = _probe_args(args)
    train_features, train_target = collect_stage_features(
        train_examples, llm, stage2_adapter, tokenizer, probe_args, device, seed=int(args.seed)
    )
    val_features, val_target = collect_stage_features(
        val_examples, llm, stage2_adapter, tokenizer, probe_args, device, seed=int(args.seed) + 1
    )
    value_readout: dict[str, Any] = {}
    for name in sorted(train_features):
        value_readout[name] = fit_ridge_readout(
            train_features[name],
            train_target,
            val_features[name],
            val_target,
            ridge=float(args.probe_ridge),
            tolerance=float(args.probe_tolerance),
            device=probe_device,
        )
    result: dict[str, Any] = {
        "warning": (
            "R2/MAE is an auxiliary representation probe only. An embedding need not linearly reconstruct a cell "
            "value for the adapter to answer a coordinate query correctly."
        ),
        "train_states": len(train_examples),
        "val_states": len(val_examples),
        "value_readout_stage2": value_readout,
    }
    if stage1_adapter is not None:
        train_features1, train_target1 = collect_stage_features(
            train_examples, llm, stage1_adapter, tokenizer, probe_args, device, seed=int(args.seed)
        )
        val_features1, val_target1 = collect_stage_features(
            val_examples, llm, stage1_adapter, tokenizer, probe_args, device, seed=int(args.seed) + 1
        )
        result["value_readout_stage1"] = {
            name: fit_ridge_readout(
                train_features1[name],
                train_target1,
                val_features1[name],
                val_target1,
                ridge=float(args.probe_ridge),
                tolerance=float(args.probe_tolerance),
                device=probe_device,
            )
            for name in sorted(train_features1)
            if name in val_features1
        }
        representation_examples = val_examples[: max(1, int(args.representation_states))]
        representations1 = _collect_prompt_representations(representation_examples, stage1_adapter, args, device)
        representations2 = _collect_prompt_representations(representation_examples, stage2_adapter, args, device)
        common = sorted(set(representations1).intersection(representations2))
        drift: dict[str, Any] = {}
        for name in common:
            drift[name] = {
                "stage1_stats": _prompt_representation_stats(representations1[name]),
                "stage2_stats": _prompt_representation_stats(representations2[name]),
                "difference": _representation_difference(representations1[name], representations2[name]),
            }
        result["representation_drift"] = drift
    else:
        representation_examples = val_examples[: max(1, int(args.representation_states))]
        representations2 = _collect_prompt_representations(representation_examples, stage2_adapter, args, device)
        result["stage2_representation_stats"] = {
            name: _prompt_representation_stats(value) for name, value in representations2.items()
        }
    return result


def _metric_mean(section: Mapping[str, Any], key: str) -> float | None:
    value = section.get(key)
    if isinstance(value, Mapping) and isinstance(value.get("mean"), (int, float)):
        return float(value["mean"])
    return None


def diagnostic_screening_flags(report: Mapping[str, Any]) -> list[dict[str, str]]:
    """Turn measurements into conservative follow-up flags, never conclusions."""

    flags: list[dict[str, str]] = []
    identity = report.get("model_identity")
    if isinstance(identity, Mapping) and not bool(identity.get("tokenizer_leaf_matches_active_model", True)):
        flags.append(
            {
                "flag": "tokenizer_identity_unconfirmed",
                "evidence": (
                    f"tokenizer={identity.get('tokenizer_name_or_path', '')!r} does not share the active model leaf; "
                    "the model check passed but tokenizer provenance is not confirmed"
                ),
                "next_check": "verify tokenizer files/config match the tokenizer used during Stage 1/Stage 2 training",
            }
        )
    channel_semantics = report.get("latent_channel_semantics")
    if isinstance(channel_semantics, Mapping):
        declared_values = [
            channel_semantics.get("stage1_declared_preserve_input_channels"),
            channel_semantics.get("stage2_declared_preserve_input_channels"),
        ]
        if any(value is False for value in declared_values):
            flags.append(
                {
                    "flag": "channel0_semantics_unverified",
                    "evidence": f"preserve_input_channels declarations={declared_values}",
                    "next_check": "verify the Stage-1 encoder contract before interpreting channel-0 value edits or R2 readouts",
                }
            )
    for section_name in (
        "coordinate_swaps",
        "local_interventions",
        "gradient_scan",
        "text_controls",
        "formal_baselines",
        "stats_swap_control",
        "spatial_probe",
    ):
        section = report.get(section_name)
        if isinstance(section, Mapping) and section.get("section_error"):
            flags.append(
                {
                    "flag": f"{section_name}_failed",
                    "evidence": str(section.get("reason", "unknown section error")),
                    "next_check": "fix or rerun the failed diagnostic before interpreting the remaining evidence",
                }
            )
        elif isinstance(section, Mapping) and section.get("skipped"):
            flags.append(
                {
                    "flag": f"{section_name}_disabled",
                    "evidence": str(section.get("reason", "section disabled by CLI")),
                    "next_check": "do not treat the diagnostic as complete until this section is enabled",
                }
            )
    swaps = report.get("coordinate_swaps")
    if isinstance(swaps, Mapping):
        tasks = swaps.get("tasks", {})
        point = tasks.get("point_compare") if isinstance(tasks, Mapping) else None
        if isinstance(point, Mapping) and int(point.get("n_pairs", 0)) == 0 and int(point.get("candidate_states", 0)) > 0:
            flags.append(
                {
                    "flag": "no_point_swap_pairs",
                    "evidence": "point-compare candidates existed but none survived coordinate/gap filters",
                    "next_check": "inspect coordinate parser, answer labels, and min-point-gap filtering",
                }
            )
        if isinstance(point, Mapping) and 0 < int(point.get("n_pairs", 0)) < 4:
            flags.append(
                {
                    "flag": "insufficient_point_swap_pairs",
                    "evidence": f"only {int(point.get('n_pairs', 0))} point-compare pairs survived parsing/gap filters",
                    "next_check": "rerun with more records or lower only the documented gap threshold; do not generalize this pilot",
                }
            )
        if isinstance(point, Mapping) and int(point.get("n_pairs", 0)) >= 4:
            flip = _metric_mean(point, "answer_flip_consistency")
            both = _metric_mean(point, "both_correct")
            if flip is not None and flip < 0.60:
                flags.append(
                    {
                        "flag": "low_coordinate_swap_flip",
                        "evidence": f"point-swap answer-flip consistency={flip:.3f}",
                        "next_check": "inspect coordinate wording/tokenization and direct spatial routing before changing loss",
                    }
                )
            elif both is not None and both < 0.60:
                flags.append(
                    {
                        "flag": "low_same_tensor_pair_accuracy",
                        "evidence": f"point-swap both-correct={both:.3f}",
                        "next_check": "separate value decoding from coordinate routing with text controls and latent scans",
                    }
                )
        region = tasks.get("region_mean_compare") if isinstance(tasks, Mapping) else None
        if isinstance(region, Mapping) and 0 < int(region.get("n_pairs", 0)) < 4:
            flags.append(
                {
                    "flag": "insufficient_region_swap_pairs",
                    "evidence": f"only {int(region.get('n_pairs', 0))} region-compare pairs survived parsing/gap filters",
                    "next_check": "rerun with more records before interpreting region routing",
                }
            )
    interventions = report.get("local_interventions")
    if isinstance(interventions, Mapping) and 0 < int(interventions.get("n_records", 0)) < 4:
        flags.append(
            {
                "flag": "insufficient_intervention_records",
                "evidence": f"only {int(interventions.get('n_records', 0))} numeric intervention records were scored",
                "next_check": "rerun with a larger intervention sample before interpreting target/control effects",
            }
        )
    if isinstance(interventions, Mapping) and int(interventions.get("n_records", 0)) >= 4:
        target = _metric_mean(interventions, "target_flip_from_clean")
        eligible_n = int(interventions.get("target_flip_eligible_n", 0))
        false_flip = _metric_mean(interventions, "non_target_false_flip_rate")
        selective = _metric_mean(
            interventions,
            "target_vs_control_selective_margin_effect_eligible",
        )
        if eligible_n < 4:
            flags.append(
                {
                    "flag": "insufficient_conditional_intervention_cases",
                    "evidence": (
                        f"only {eligible_n}/{int(interventions.get('n_records', 0))} edits had a clean prediction "
                        "different from the intended edited label"
                    ),
                    "next_check": "report absolute target-edit success separately; do not infer causal sensitivity from this sample",
                }
            )
        elif target is not None and target < 0.50:
            flags.append(
                {
                    "flag": "weak_numeric_target_sensitivity",
                    "evidence": f"conditional target-edit flip success={target:.3f} over n={eligible_n} eligible cases",
                    "next_check": "verify numeric option parsing/scale and compare against explicit text controls",
                }
            )
        if false_flip is not None and false_flip > 0.40:
            flags.append(
                {
                    "flag": "non_target_global_shortcut",
                    "evidence": f"non-target false-flip rate={false_flip:.3f}",
                    "next_check": "inspect all-token coupling and run matched boundary/interior controls",
                }
            )
        if selective is not None and selective <= 0.0:
            flags.append(
                {
                    "flag": "no_target_control_selectivity",
                    "evidence": f"conditional target-minus-control margin effect={selective:.4g} over n={eligible_n}",
                    "next_check": "do not treat local edits as causal; examine token routing and loss supervision",
                }
            )
        by_task = interventions.get("by_task", {})
        if isinstance(by_task, Mapping):
            normalized = by_task.get("normalized_point_value")
            raw = by_task.get("raw_point_value_with_stats")
            if isinstance(normalized, Mapping) and isinstance(raw, Mapping):
                normalized_accuracy = _metric_mean(normalized, "clean_accuracy")
                raw_accuracy = _metric_mean(raw, "clean_accuracy")
                if (
                    normalized_accuracy is not None
                    and raw_accuracy is not None
                    and normalized_accuracy >= 0.60
                    and raw_accuracy < normalized_accuracy - 0.20
                ):
                    flags.append(
                        {
                            "flag": "raw_affine_reasoning_gap",
                            "evidence": (
                                f"matched clean normalized accuracy={normalized_accuracy:.3f}, "
                                f"raw-affine accuracy={raw_accuracy:.3f}"
                            ),
                            "next_check": "focus on mean/scale arithmetic and prompt interpretation, not cell-value readout",
                        }
                    )
    gradients = report.get("gradient_scan")
    if isinstance(gradients, Mapping):
        stages = gradients.get("stages", {})
        if isinstance(stages, Mapping):
            stage2 = stages.get("stage2")
            stage1 = stages.get("stage1")
            if isinstance(stage2, Mapping) and bool(stage2.get("partial_errors")) and not isinstance(stage1, Mapping):
                flags.append(
                    {
                        "flag": "gradient_scan_partial_errors",
                        "evidence": f"Stage2 errors={int(stage2.get('n_errors', 0))}",
                        "next_check": "inspect gradient_scan.errors before interpreting target-cell selectivity",
                    }
                )
            if isinstance(stage2, Mapping):
                entropy = _metric_mean(stage2, "gradient_entropy")
                top1 = _metric_mean(stage2, "target_top1")
                if entropy is not None and entropy > 0.85:
                    flags.append(
                        {
                            "flag": "diffuse_latent_gradient",
                            "evidence": f"Stage-2 normalized gradient entropy={entropy:.3f}",
                            "next_check": "inspect spatial self-attention/residual mixing and compare Stage 1",
                        }
                    )
                if top1 is not None and top1 < 0.10:
                    flags.append(
                        {
                            "flag": "queried_cell_not_gradient_top1",
                            "evidence": f"Stage-2 queried-cell gradient top1={top1:.3f}",
                            "next_check": "coordinate routing remains unverified; use role-swap evidence",
                        }
                    )
            if isinstance(stage1, Mapping) and isinstance(stage2, Mapping):
                e1 = _metric_mean(stage1, "gradient_entropy")
                e2 = _metric_mean(stage2, "gradient_entropy")
                if e1 is not None and e2 is not None and e2 > e1 + 0.10:
                    flags.append(
                        {
                            "flag": "stage2_gradient_selectivity_drop",
                            "evidence": f"gradient entropy Stage1={e1:.3f}, Stage2={e2:.3f}",
                            "next_check": "inspect Stage-2 representation drift before adding new supervision",
                        }
                    )
                if bool(stage1.get("partial_errors")) or bool(stage2.get("partial_errors")):
                    flags.append(
                        {
                            "flag": "gradient_scan_partial_errors",
                            "evidence": (
                                f"Stage1 errors={int(stage1.get('n_errors', 0))}, "
                                f"Stage2 errors={int(stage2.get('n_errors', 0))}"
                            ),
                            "next_check": "inspect gradient_scan.errors before comparing target-cell selectivity",
                        }
                    )
    text_controls = report.get("text_controls")
    if isinstance(text_controls, Mapping):
        variants = text_controls.get("variants", {})
        explicit = variants.get("explicit_target_z") if isinstance(variants, Mapping) else None
        if isinstance(explicit, Mapping):
            if 0 < int(explicit.get("n_records", 0)) < 4:
                flags.append(
                    {
                        "flag": "insufficient_text_control_records",
                        "evidence": f"only {int(explicit.get('n_records', 0))} explicit text-control records were scored",
                        "next_check": "increase text-control-records-per-task before using the frozen-LLM ceiling",
                    }
                )
            accuracy = _metric_mean(explicit, "accuracy")
            if accuracy is not None and accuracy < 0.50:
                flags.append(
                    {
                        "flag": "frozen_llm_text_control_low",
                        "evidence": f"explicit numeric text-control accuracy={accuracy:.3f}",
                        "next_check": "treat decoder/prompt semantics as an upper-bound bottleneck before adapter changes",
                    }
                )
    baselines = report.get("formal_baselines")
    if isinstance(baselines, Mapping):
        if 0 < int(baselines.get("n_records", 0)) < 4:
            flags.append(
                {
                    "flag": "insufficient_formal_baseline_records",
                    "evidence": f"only {int(baselines.get('n_records', 0))} records entered the zero-latent baseline",
                    "next_check": "increase max-states-per-task before comparing latent gain",
                }
            )
        clean_accuracy = _metric_mean(baselines, "clean_accuracy")
        zero_accuracy = _metric_mean(baselines, "zero_latent_accuracy")
        shuffled_accuracy = _metric_mean(baselines, "shuffled_accuracy")
        if clean_accuracy is not None and zero_accuracy is not None and zero_accuracy >= clean_accuracy - 0.05:
            flags.append(
                {
                    "flag": "zero_latent_shortcut",
                    "evidence": f"zero-latent accuracy={zero_accuracy:.3f}, clean accuracy={clean_accuracy:.3f}",
                    "next_check": "treat coordinate/value results as potentially text or positional prior driven",
                }
            )
        paired = baselines.get("paired_clean_vs_shuffled")
        if isinstance(paired, Mapping):
            changed_rate = _metric_mean(paired, "prediction_changed_rate")
            kl = _metric_mean(paired, "candidate_kl_clean_to_shuffled")
            if clean_accuracy is not None and shuffled_accuracy is not None and clean_accuracy - shuffled_accuracy > 0.10:
                if changed_rate is not None and changed_rate < 0.10 and kl is not None and kl < 0.02:
                    flags.append(
                        {
                            "flag": "latent_accuracy_gain_without_paired_sensitivity",
                            "evidence": (
                                f"clean-shuffled accuracy gap={clean_accuracy - shuffled_accuracy:.3f}, "
                                f"prediction-change={changed_rate:.3f}, candidate KL={kl:.4f}"
                            ),
                            "next_check": "verify the paired control construction and inspect task-level gains; aggregate accuracy may hide a small subset of changed records",
                        }
                    )
    stats_swap = report.get("stats_swap_control")
    if isinstance(stats_swap, Mapping):
        n_stats = int(stats_swap.get("n_records", 0))
        if n_stats == 0 and int(stats_swap.get("candidate_records", 0)) > 0:
            flags.append(
                {
                    "flag": "stats_swap_unavailable",
                    "evidence": "raw records were present but prompt_data was insufficient to reproduce the official shuffled_stats control",
                    "next_check": "inspect QA prompt_data fields before interpreting raw affine behavior",
                }
            )
        if 0 < n_stats < 4:
            flags.append(
                {
                    "flag": "insufficient_stats_swap_records",
                    "evidence": f"only {n_stats} raw shuffled-stats pairs were scored",
                    "next_check": "increase max-states-per-task before interpreting affine-stat sensitivity",
                }
            )
        if n_stats >= 4:
            changed_rate = _metric_mean(stats_swap, "prediction_changed_rate")
            kl = _metric_mean(stats_swap, "candidate_kl_clean_to_swapped_stats")
            if changed_rate is not None and changed_rate < 0.10 and kl is not None and kl < 0.02:
                flags.append(
                    {
                        "flag": "raw_stats_ignored",
                        "evidence": f"shuffled-stats prediction-change={changed_rate:.3f}, candidate KL={kl:.4f}",
                        "next_check": "separate cell-value routing from mean/scale parsing; inspect raw prompt construction and arithmetic supervision",
                    }
                )
    probe = report.get("spatial_probe")
    if isinstance(probe, Mapping):
        readout_stage2 = probe.get("value_readout_stage2")
        if isinstance(readout_stage2, Mapping):
            latent_raw = readout_stage2.get("latent_raw")
            if isinstance(latent_raw, Mapping):
                latent_val = latent_raw.get("val", {})
                if isinstance(latent_val, Mapping) and isinstance(latent_val.get("r2"), (int, float)):
                    if float(latent_val["r2"]) < 0.98:
                        flags.append(
                            {
                                "flag": "latent_raw_contract_sanity_low",
                                "evidence": f"raw channel-0 latent probe R2={float(latent_val['r2']):.3f}",
                                "next_check": "verify latent cache identity/channel ordering before interpreting adapter diagnostics",
                            }
                        )
        drift = probe.get("representation_drift")
        if isinstance(drift, Mapping):
            prompt = drift.get("global_soft_prompt")
            if isinstance(prompt, Mapping):
                difference = prompt.get("difference")
                if isinstance(difference, Mapping) and float(difference.get("relative_l2", 0.0)) > 0.25:
                    flags.append(
                        {
                            "flag": "large_stage2_prompt_drift",
                            "evidence": f"global soft-prompt relative L2 drift={float(difference['relative_l2']):.3f}",
                            "next_check": "compare Stage-1/Stage-2 pair accuracy and freeze/regularize before more training",
                        }
                    )
        readout1 = probe.get("value_readout_stage1")
        readout2 = readout_stage2
        if isinstance(readout1, Mapping) and isinstance(readout2, Mapping):
            stage_name = "global_soft_prompt"
            metric1 = readout1.get(stage_name)
            metric2 = readout2.get(stage_name)
            if isinstance(metric1, Mapping) and isinstance(metric2, Mapping):
                val1 = metric1.get("val", {})
                val2 = metric2.get("val", {})
                if isinstance(val1, Mapping) and isinstance(val2, Mapping):
                    r1 = val1.get("r2")
                    r2 = val2.get("r2")
                    if isinstance(r1, (int, float)) and isinstance(r2, (int, float)) and float(r2) < float(r1) - 0.20:
                        flags.append(
                            {
                                "flag": "auxiliary_value_readout_drop",
                                "evidence": f"soft-prompt value R2 Stage1={float(r1):.3f}, Stage2={float(r2):.3f}",
                                "next_check": "treat as representation-drift support only; confirm with swap and gradient evidence",
                            }
                        )
    if not flags:
        flags.append(
            {
                "flag": "no_screening_flag",
                "evidence": "available diagnostics did not cross conservative screening thresholds",
                "next_check": "inspect confidence intervals and field/task stratification before concluding success",
            }
        )
    return flags


def _run_section(name: str, enabled: bool, function: Any) -> dict[str, Any]:
    if not enabled:
        return {"skipped": True, "reason": "disabled_by_cli"}
    print(f"diagnostic=section_start name={name}", flush=True)
    started = time.monotonic()
    try:
        result = function()
        result["elapsed_seconds"] = float(time.monotonic() - started)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        status = "section_error" if result.get("section_error") else "section_done"
        print(
            f"diagnostic={status} name={name} elapsed={result['elapsed_seconds']:.1f}s"
            + (f" reason={result.get('reason', '')}" if result.get("section_error") else ""),
            flush=True,
        )
        return result
    except Exception as exc:  # diagnostics should report one failed module without hiding others
        elapsed = float(time.monotonic() - started)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"diagnostic=section_error name={name} elapsed={elapsed:.1f}s error={exc}", flush=True)
        return {
            "skipped": True,
            "reason": f"{type(exc).__name__}: {exc}",
            "section_error": name,
            "elapsed_seconds": elapsed,
        }


def _checkpoint_summary(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    args = checkpoint.get("args")
    if not isinstance(args, Mapping):
        return {}
    fields = (
        "adapter_architecture",
        "global_adapter_type",
        "soft_prompt_tokens",
        "question_conditioning",
        "latent_pos_encoding",
        "prompt_template",
        "max_prompt_tokens",
        "model_name_or_path",
        "adapter_init_checkpoint",
    )
    return {field: args.get(field) for field in fields if field in args}


def _declared_preserve_input_channels(checkpoint_args: Mapping[str, Any]) -> bool | None:
    """Read the encoder contract when it was persisted in a checkpoint args blob."""

    direct = checkpoint_args.get("preserve_input_channels")
    if isinstance(direct, bool):
        return direct
    for container_key in ("patch_encoder_config", "patch_encoder", "encoder_config"):
        container = checkpoint_args.get(container_key)
        if not isinstance(container, Mapping):
            continue
        model = container.get("model")
        if isinstance(model, Mapping) and isinstance(model.get("preserve_input_channels"), bool):
            return bool(model["preserve_input_channels"])
        if isinstance(container.get("preserve_input_channels"), bool):
            return bool(container["preserve_input_channels"])
    return None


def main() -> None:
    raw_args = parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise RuntimeError(
            "This diagnostic is single-process by design; invoke it with plain python on one selected GPU, "
            "not torchrun."
        )
    checkpoint_path = Path(raw_args.checkpoint).expanduser()
    checkpoint = load_checkpoint(checkpoint_path)
    args, stage1_path = configure_runtime_args(raw_args, checkpoint)
    validate_diagnostic_args(args)
    checkpoint_args = checkpoint.get("args")
    if not isinstance(checkpoint_args, Mapping):
        raise ValueError("Stage-2 checkpoint has no args mapping.")
    checkpoint_architecture = str(checkpoint_args.get("adapter_architecture", ""))
    supported_architectures = {"alignment_adapter", "grounded_evidence_adapter"}
    if checkpoint_architecture not in supported_architectures:
        raise ValueError(
            "This diagnostic supports alignment_adapter and grounded_evidence_adapter, got "
            f"{checkpoint_architecture!r}."
        )
    if str(args.adapter_architecture) != checkpoint_architecture:
        raise ValueError(
            "Runtime adapter architecture differs from the checkpoint contract: "
            f"runtime={args.adapter_architecture!r}, checkpoint={checkpoint_architecture!r}."
        )
    # The frozen decoder is part of the formal scoring contract.  A same-width
    # but different Qwen/tokenizer would make every logit comparison invalid.
    try:
        validate_stage1_model_identity(checkpoint_args, args.model_name_or_path)
    except ValueError as exc:
        raise ValueError(f"Stage-2 checkpoint model identity is invalid: {exc}") from exc
    stage2_preserve_input_channels = _declared_preserve_input_channels(checkpoint_args)
    if stage2_preserve_input_channels is None:
        stage2_preserve_input_channels = _declared_preserve_input_channels(vars(args))
    # The checkpoint's Stage-1 path is authoritative for metadata provenance;
    # explicit --stage1-checkpoint is already incorporated by configure_runtime_args.
    if stage1_path:
        args.adapter_init_checkpoint = stage1_path
        args.qa_alignment_checkpoint = stage1_path
    qa_audit = audit_qa_metadata(args)
    latent_contract = qa_audit.get("latent_contract") if isinstance(qa_audit, Mapping) else None
    preflight_checkpoint_envelope(
        checkpoint,
        expected_architecture=checkpoint_architecture,
        expected_latent_contract=latent_contract if isinstance(latent_contract, Mapping) else None,
    )
    apply_runtime_environment(args)
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    random.seed(int(args.seed))
    device = resolve_device(args.device)
    probe_device = resolve_device(args.probe_device) if str(args.probe_device).lower() != "auto" else device
    # Resolve required JSONL paths before allocating the 14B decoder so a
    # missing asset fails quickly and unambiguously.
    first_split_path = qa_path(args.qa_dir, args.split)
    train_split_path = qa_path(args.qa_dir, args.train_split) if bool(args.run_spatial_probe) else None
    val_split_path = qa_path(args.qa_dir, args.val_split) if bool(args.run_spatial_probe) else None
    print(
        f"diagnostic=start device={device} split={args.split} checkpoint={checkpoint_path} "
        f"model={args.model_name_or_path}",
        flush=True,
    )
    tokenizer, llm, model_dtype = load_tokenizer_and_llm(args, device)
    tokenizer_name = str(getattr(tokenizer, "name_or_path", "") or "").strip()
    tokenizer_identity_match = bool(
        tokenizer_name
        and model_identifier_leaf(tokenizer_name) == model_identifier_leaf(args.model_name_or_path)
    )
    cache_size = int(
        args.latent_cache_size
        if getattr(args, "latent_cache_size", None) is not None
        else (raw_args.latent_cache_size if raw_args.latent_cache_size is not None else 256)
    )
    dataset_kwargs = {
        "latent_dir": args.latent_dir,
        "prefer_record_latent_ref": bool(args.prefer_record_latent_ref),
        "shuffle_seed": int(args.shuffle_seed),
        "latent_cache_size": max(0, cache_size),
        "latent_contract": latent_contract,
    }
    scored_max_records = int(args.max_records) if int(args.max_records) > 0 else None
    diagnostic_dataset = TensorReadoutQADataset(
        first_split_path,
        max_records=scored_max_records,
        **dataset_kwargs,
    )
    if not diagnostic_dataset.records:
        raise ValueError(f"Selected diagnostic split is empty: {first_split_path}")
    choice_tokenization = audit_formal_choice_tokenization(diagnostic_dataset.records, tokenizer)
    train_dataset: TensorReadoutQADataset | None = None
    val_dataset: TensorReadoutQADataset | None = None
    probe_dataset_scope: dict[str, Any] = {"enabled": bool(args.run_spatial_probe)}
    if bool(args.run_spatial_probe):
        if train_split_path is None or val_split_path is None:
            raise RuntimeError("Spatial probe paths were not resolved.")
        train_scan = max(
            int(args.probe_train_records),
            int(args.probe_train_states) * 32,
            256,
        )
        val_scan = max(
            int(args.max_records),
            int(args.probe_val_records),
            int(args.probe_val_states) * 32,
            256,
        )
        train_dataset = TensorReadoutQADataset(
            train_split_path,
            max_records=train_scan,
            **dataset_kwargs,
        )
        val_dataset = TensorReadoutQADataset(
            val_split_path,
            max_records=val_scan,
            **dataset_kwargs,
        )
        probe_dataset_scope.update(
            {
                "train_split": str(args.train_split),
                "val_split": str(args.val_split),
                "train_records": len(train_dataset.records),
                "val_records": len(val_dataset.records),
                "train_scan_cap": int(train_scan),
                "val_scan_cap": int(val_scan),
            }
        )
    # The audit always names the exact scored subset.  Probe datasets are
    # added only when they are distinct from that subset; otherwise a capped
    # probe would silently replace the records actually being scored.
    audit_datasets: dict[str, TensorReadoutQADataset] = {str(args.split): diagnostic_dataset}
    if bool(args.run_spatial_probe) and train_dataset is not None and str(args.train_split) != str(args.split):
        audit_datasets["train"] = train_dataset
    if bool(args.run_spatial_probe) and val_dataset is not None and str(args.val_split) != str(args.split):
        audit_datasets["val"] = val_dataset
    dataset_audit = audit_qa_datasets(
        audit_datasets,
        require_disjoint_splits=True,
        require_complete_split_coverage=False,
    )
    first_latent = diagnostic_dataset.load_latent_for_record(diagnostic_dataset.records[0])
    if first_latent.ndim != 3 or int(first_latent.shape[0]) < 1 or any(int(dim) <= 0 for dim in first_latent.shape[1:]):
        raise ValueError(f"Formal diagnostic requires a non-empty [channels,height,width] latent; got {tuple(first_latent.shape)}")
    if isinstance(latent_contract, Mapping):
        validate_adapter_checkpoint_payload(
            checkpoint,
            expected_latent_shape=tuple(int(value) for value in first_latent.shape),
            expected_llm_hidden_size=int(llm.get_input_embeddings().embedding_dim),
            expected_architecture=checkpoint_architecture,
            expected_latent_contract=latent_contract,
        )
    else:
        raise ValueError("Formal diagnostic requires a validated QA latent_contract mapping.")
    stage2_adapter = adapter_from_checkpoint(
        checkpoint=checkpoint,
        latent_shape=tuple(int(value) for value in first_latent.shape),
        llm_hidden_size=int(llm.get_input_embeddings().embedding_dim),
    ).to(device)
    stage2_adapter.eval()
    expected_tokens = int(first_latent.shape[-2]) * int(first_latent.shape[-1])
    if isinstance(stage2_adapter, HybridGlobalLocalAdapter):
        if not isinstance(stage2_adapter.local_adapter, GroundedEvidenceAdapter):
            raise TypeError("Only the factorized grounded-evidence Hybrid adapter is supported here.")
        stage2_spatial_adapter = stage2_adapter.global_adapter
    else:
        stage2_spatial_adapter = stage2_adapter
    if int(getattr(stage2_spatial_adapter, "soft_prompt_tokens", -1)) != expected_tokens:
        raise ValueError(
            "Direct grounding diagnostics require one row-major soft token per latent cell: "
            f"observed={getattr(stage2_spatial_adapter, 'soft_prompt_tokens', None)}, expected={expected_tokens}."
        )
    if str(getattr(stage2_spatial_adapter, "adapter_type", "")) != "spatial_transformer":
        raise ValueError(
            "Direct grounding diagnostics require global_adapter_type=spatial_transformer; "
            f"observed {getattr(stage2_spatial_adapter, 'adapter_type', None)!r}."
        )
    stage1_adapter: nn.Module | None = None
    stage1_checkpoint_summary: dict[str, Any] = {}
    stage1_preserve_input_channels: bool | None = None
    if bool(args.compare_stage1) and stage1_path:
        stage1_checkpoint = load_checkpoint(stage1_path)
        stage1_checkpoint_args = stage1_checkpoint.get("args")
        if not isinstance(stage1_checkpoint_args, Mapping):
            raise ValueError("Stage-1 checkpoint has no args mapping.")
        validate_stage1_model_identity(stage1_checkpoint_args, args.model_name_or_path)
        stage1_preserve_input_channels = _declared_preserve_input_channels(stage1_checkpoint_args)
        if stage1_preserve_input_channels is None:
            stage1_preserve_input_channels = stage2_preserve_input_channels
        stage1_adapter = adapter_from_checkpoint(
            checkpoint=stage1_checkpoint,
            latent_shape=tuple(int(value) for value in first_latent.shape),
            llm_hidden_size=int(llm.get_input_embeddings().embedding_dim),
        ).to(device)
        stage1_adapter.eval()
        if str(getattr(stage1_adapter, "adapter_type", "")) != "spatial_transformer":
            raise ValueError(
                "Stage-1 comparison checkpoint is not a spatial_transformer adapter: "
                f"{getattr(stage1_adapter, 'adapter_type', None)!r}."
            )
        if int(getattr(stage1_adapter, "soft_prompt_tokens", -1)) != expected_tokens:
            raise ValueError(
                "Stage-1 comparison adapter does not emit one token per latent cell: "
                f"observed={getattr(stage1_adapter, 'soft_prompt_tokens', None)}, expected={expected_tokens}."
            )
        stage1_checkpoint_summary = _checkpoint_summary(stage1_checkpoint)
        del stage1_checkpoint

    report: dict[str, Any] = {
        "diagnostic": "direct_tensor_grounding",
        "version": 2,
        "interpretation_policy": (
            "Screening flags combine complementary evidence and are not automatic causal conclusions. "
            "Role swaps are the primary coordinate-routing evidence; local latent edits, stats swaps, and linear "
            "readouts are auxiliary mechanism probes."
        ),
        "reading_guide": [
            "Treat coordinate_swaps.point_compare and region_mean_compare as the primary in-distribution routing evidence.",
            "Use formal_baselines.paired_clean_vs_shuffled to verify that latent-dependent accuracy is accompanied by paired sensitivity.",
            "Use local_interventions only as an OOD mechanism stress test; prefer target_flip_from_clean over absolute target success.",
            "Use gradient_scan target rank/entropy to locate spatial selectivity changes, not as a proof that embeddings must regress cell values.",
            "Use stats_swap_control to separate raw affine-stat sensitivity from normalized cell-value routing.",
            "Use spatial_probe R2 and representation drift only as auxiliary support; never require a linear cell readout as the grounding criterion.",
        ],
        "checkpoint": str(checkpoint_path),
        "adapter_architecture": checkpoint_architecture,
        "stage1_checkpoint": str(stage1_path) if stage1_path else None,
        "checkpoint_summary": _checkpoint_summary(checkpoint),
        "checkpoint_metrics": dict(checkpoint.get("metrics", {})) if isinstance(checkpoint.get("metrics"), Mapping) else {},
        "stage1_checkpoint_summary": stage1_checkpoint_summary,
        "split": str(args.split),
        "model_dtype": str(model_dtype).replace("torch.", ""),
        "model_identity": {
            "checkpoint_model_name_or_path": str(checkpoint_args.get("model_name_or_path", "")),
            "active_model_name_or_path": str(args.model_name_or_path),
            "active_model_leaf_matches_checkpoint": model_identifier_leaf(
                checkpoint_args.get("model_name_or_path", "")
            )
            == model_identifier_leaf(args.model_name_or_path),
            "tokenizer_name_or_path": tokenizer_name,
            "tokenizer_leaf_matches_active_model": tokenizer_identity_match,
            "note": "Model identity is required; tokenizer leaf match is recorded as an additional sanity check.",
        },
        "latent_channel_semantics": {
            "intervention_channel": 0,
            "expected": "exact per-patch normalized input value when preserve_input_channels=true",
            "stage2_declared_preserve_input_channels": stage2_preserve_input_channels,
            "stage1_declared_preserve_input_channels": stage1_preserve_input_channels,
            "note": "A channel-0 edit is only an interpretable value stress test when this encoder contract is true; otherwise treat it as generic latent-channel OOD probing.",
        },
        "diagnostic_config": {
            "max_records": int(args.max_records),
            "max_states_per_task": int(args.max_states_per_task),
            "min_point_gap": float(args.min_point_gap),
            "min_region_gap": float(args.min_region_gap),
            "intervention_records_per_task": int(args.intervention_records_per_task),
            "intervention_controls_per_record": int(args.intervention_controls_per_record),
            "gradient_records_per_task": int(args.gradient_records_per_task),
            "text_control_records_per_task": int(args.text_control_records_per_task),
            "run_stats_swap_control": bool(args.run_stats_swap_control),
            "bootstrap_reps": int(args.bootstrap_reps),
        },
        "formal_scoring": {
            "contract_mode": "formal_v3_qa_v2_checkpoint",
            "method": "prompt_boundary_single_token_restricted_choice_logits",
            "candidate_tokenization": "single_distinct_token_required",
            "oracle_passed_to_model": False,
            "text_controls_derive_values_from_latent": True,
            "adapter_prefix_depends_on_query": (
                checkpoint_architecture == "grounded_evidence_adapter"
            ),
            "coordinate_routing_location": (
                "grounded_adapter_factorized_row_column_reader"
                if checkpoint_architecture == "grounded_evidence_adapter"
                else "frozen_llm_attention_over_query_invariant_spatial_prefix"
            ),
            "choice_tokenization": choice_tokenization,
        },
        "qa_metadata_audit": qa_audit,
        "dataset_audit": dataset_audit,
        "dataset_audit_scope": {
            "audited_splits": sorted(audit_datasets),
            "scored_split_is_exact_subset": True,
            "scored_records": len(diagnostic_dataset.records),
            "complete_split_coverage_asserted": False,
            "probe_scope": probe_dataset_scope,
        },
    }
    # Match official evaluation for all no-grad logits.  Gradient scans opt
    # back into the loader's non-reentrant checkpoint execution mode below.
    llm.eval()
    report["coordinate_swaps"] = _run_section(
        "coordinate_swaps",
        bool(args.run_coordinate_swaps),
        lambda: run_coordinate_swaps(diagnostic_dataset, llm, stage2_adapter, tokenizer, args, device),
    )
    report["local_interventions"] = _run_section(
        "local_interventions",
        bool(args.run_local_interventions),
        lambda: run_local_interventions(diagnostic_dataset, llm, stage2_adapter, tokenizer, args, device),
    )
    if bool(args.run_gradient_scan) and bool(args.llm_gradient_checkpointing):
        set_frozen_llm_execution_mode(llm, checkpoint_training=True)
    report["gradient_scan"] = _run_section(
        "gradient_scan",
        bool(args.run_gradient_scan),
        lambda: run_gradient_scan(
            diagnostic_dataset,
            llm,
            stage2_adapter,
            stage1_adapter,
            tokenizer,
            args,
            device,
        ),
    )
    llm.eval()
    report["text_controls"] = _run_section(
        "text_controls",
        bool(args.run_text_controls),
        lambda: run_text_controls(diagnostic_dataset, llm, tokenizer, args, device),
    )
    llm.eval()
    report["formal_baselines"] = _run_section(
        "formal_baselines",
        bool(args.run_formal_baselines),
        lambda: run_formal_baseline_profile(
            diagnostic_dataset,
            llm,
            stage2_adapter,
            tokenizer,
            args,
            device,
        ),
    )
    report["stats_swap_control"] = _run_section(
        "stats_swap_control",
        bool(args.run_stats_swap_control),
        lambda: run_stats_swap_control(
            diagnostic_dataset,
            llm,
            stage2_adapter,
            tokenizer,
            args,
            device,
        ),
    )
    report["spatial_probe"] = _run_section(
        "spatial_probe",
        bool(args.run_spatial_probe),
        lambda: run_spatial_probe(
            train_dataset,
            val_dataset,
            llm,
            stage2_spatial_adapter,
            stage1_adapter,
            tokenizer,
            args,
            device,
            probe_device,
        ),
    )
    report["screening_flags"] = diagnostic_screening_flags(report)
    failed_sections = [
        name
        for name in (
            "coordinate_swaps",
            "local_interventions",
            "gradient_scan",
            "text_controls",
            "formal_baselines",
            "stats_swap_control",
            "spatial_probe",
        )
        if isinstance(report.get(name), Mapping) and report[name].get("section_error")
    ]
    diagnostic_section_names = (
        "coordinate_swaps",
        "local_interventions",
        "gradient_scan",
        "text_controls",
        "formal_baselines",
        "stats_swap_control",
        "spatial_probe",
    )
    disabled_sections = [
        name
        for name in diagnostic_section_names
        if isinstance(report.get(name), Mapping) and report[name].get("skipped")
    ]
    report["diagnostic_complete"] = not failed_sections and not disabled_sections
    report["failed_sections"] = failed_sections
    report["disabled_sections"] = disabled_sections
    report["diagnostic_status"] = (
        "complete" if report["diagnostic_complete"] else "partial" if failed_sections else "limited"
    )
    output_path = Path(args.output).expanduser() if args.output else checkpoint_path.parent / "diagnostics" / "direct_tensor_grounding_diagnostic.json"
    dump_json(output_path, report)
    print(
        f"diagnostic={report['diagnostic_status']} output={output_path}",
        flush=True,
    )
    for flag in report["screening_flags"]:
        print(f"diagnostic=flag {flag['flag']}: {flag['evidence']}", flush=True)
    if failed_sections:
        # A JSON report is still written for debugging, but CI/cluster launch
        # scripts must not treat a partial diagnostic as a successful run.
        raise SystemExit(1)


if __name__ == "__main__":
    main()
