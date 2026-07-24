from __future__ import annotations

import argparse
import math
import re
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tensor_compression.downstream.pdebench import resolve_device  # noqa: E402
from tensor_compression.utils import dump_json  # noqa: E402

from scripts.train_tensor_llm_adapter import (  # noqa: E402
    HybridGlobalLocalAdapter,
    ResidualQuestionConditionedAdapter,
    TensorReadoutQADataset,
    adapter_from_checkpoint,
    apply_config_defaults,
    apply_runtime_environment,
    contextual_adapter_question_context,
    load_tokenizer_and_llm,
    qa_path,
)
from scripts.train_tensor_patch_text_alignment import TensorPatchAlignmentAdapter  # noqa: E402


POINT_COORDINATE_PATTERN = re.compile(
    r"\brow\s+(-?\d+)\s*,?\s*column\s+(-?\d+)\b",
    flags=re.IGNORECASE,
)

CHECKPOINT_RUNTIME_FIELDS = (
    "model_name_or_path",
    "qa_dir",
    "latent_dir",
    "train_split",
    "val_split",
    "prompt_template",
    "max_prompt_tokens",
    "local_context_layer",
    "prefer_record_latent_ref",
    "torch_dtype",
    "trust_remote_code",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Separate same-position value loss from natural-language coordinate-routing failure "
            "in a one-spatial-token-per-latent Stage-2 adapter."
        )
    )
    parser.add_argument("--config", type=str, default="configs/tensor_llm_adapter_pipeline.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Stage-2 adapter_best.pt or adapter_last.pt.")
    parser.add_argument("--qa-dir", type=str, default=None)
    parser.add_argument("--latent-dir", type=str, default=None)
    parser.add_argument("--model-name-or-path", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--hf-home", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default=None,
        choices=("auto", "float32", "float16", "bfloat16"),
    )
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--prefer-record-latent-ref", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--train-split", type=str, default=None)
    parser.add_argument("--val-split", type=str, default=None)
    parser.add_argument("--prompt-template", type=str, default=None, choices=("generic", "task_specific"))
    parser.add_argument("--max-prompt-tokens", type=int, default=None)
    parser.add_argument("--local-context-layer", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--shuffle-seed", type=int, default=None)

    parser.add_argument("--probe-train-states", type=int, default=128)
    parser.add_argument("--probe-val-states", type=int, default=64)
    parser.add_argument("--positions-per-state", type=int, default=8)
    parser.add_argument("--feature-batch-size", type=int, default=2)
    parser.add_argument("--value-channel", type=int, default=0)
    parser.add_argument("--linear-ridge", type=float, default=1.0e-3)
    parser.add_argument("--value-tolerance", type=float, default=0.25)
    parser.add_argument(
        "--probe-device",
        type=str,
        default="auto",
        help="Device used for the closed-form ridge probes; auto reuses --device.",
    )
    parser.add_argument("--routing-groups", type=int, default=32)
    parser.add_argument("--routing-max-questions", type=int, default=4)
    parser.add_argument("--max-train-scan-records", type=int, default=None)
    parser.add_argument("--max-val-scan-records", type=int, default=None)
    parser.add_argument("--diagnostic-latent-cache-size", type=int, default=256)
    parser.add_argument("--output", type=str, default=None)
    raw_args = parser.parse_args()
    explicit_runtime_overrides = {
        field: getattr(raw_args, field, None) is not None
        for field in CHECKPOINT_RUNTIME_FIELDS
    }

    # Stage-2 config validation expects an initializer. The diagnostic checkpoint is the
    # authoritative architecture/state source and is never modified.
    raw_args.adapter_init_checkpoint = raw_args.checkpoint
    raw_args.qa_alignment_checkpoint = raw_args.checkpoint
    args = apply_config_defaults(raw_args)
    args.explicit_runtime_overrides = explicit_runtime_overrides
    if int(args.probe_train_states) < 2 or int(args.probe_val_states) < 2:
        raise ValueError("At least two unique train and validation states are required.")
    if int(args.positions_per_state) <= 0 or int(args.feature_batch_size) <= 0:
        raise ValueError("positions_per_state and feature_batch_size must be positive.")
    if float(args.linear_ridge) <= 0.0:
        raise ValueError("linear_ridge must be positive for a stable high-dimensional probe.")
    if int(args.routing_groups) < 0 or int(args.routing_max_questions) < 2:
        raise ValueError("routing_groups must be non-negative and routing_max_questions must be at least two.")
    return args


def restore_checkpoint_runtime_args(
    args: argparse.Namespace,
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    """Restore checkpoint-coupled inputs unless the CLI explicitly overrides them."""

    checkpoint_args = checkpoint.get("args")
    if not isinstance(checkpoint_args, Mapping):
        raise ValueError("The diagnostic checkpoint has no args mapping.")
    explicit = getattr(args, "explicit_runtime_overrides", {})
    restored: dict[str, Any] = {}
    for field in CHECKPOINT_RUNTIME_FIELDS:
        value = checkpoint_args.get(field)
        if not bool(explicit.get(field, False)) and value is not None:
            setattr(args, field, value)
            restored[field] = value
    return restored


def row_major_index(row: int, col: int, height: int, width: int) -> int:
    row = int(row)
    col = int(col)
    height = int(height)
    width = int(width)
    if not (0 <= row < height and 0 <= col < width):
        raise IndexError(f"Coordinate ({row},{col}) lies outside a {height}x{width} grid.")
    return row * width + col


def point_coordinate_from_record(record: Mapping[str, Any]) -> tuple[int, int] | None:
    """Return one zero-based point coordinate without using a task label."""

    metadata = record.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    grid_shape = metadata.get("grid_shape")
    height = int(grid_shape[0]) if isinstance(grid_shape, Sequence) and len(grid_shape) == 2 else None
    width = int(grid_shape[1]) if isinstance(grid_shape, Sequence) and len(grid_shape) == 2 else None

    oracle = record.get("oracle")
    if isinstance(oracle, Mapping) and "row" in oracle and "col" in oracle:
        origin = int(metadata.get("oracle_coordinate_origin", 0))
        candidates = [(int(oracle["row"]) - origin, int(oracle["col"]) - origin)]
    else:
        query = str(record.get("query") or record.get("question") or "")
        matches = POINT_COORDINATE_PATTERN.findall(query)
        if len(matches) != 1:
            return None
        origin = int(metadata.get("coordinate_origin", 1))
        candidates = [(int(matches[0][0]) - origin, int(matches[0][1]) - origin)]

    row, col = candidates[0]
    if height is not None and width is not None and not (0 <= row < height and 0 <= col < width):
        return None
    return row, col


def unique_state_examples(
    dataset: TensorReadoutQADataset,
    limit: int,
) -> list[tuple[Mapping[str, Any], torch.Tensor]]:
    examples: list[tuple[Mapping[str, Any], torch.Tensor]] = []
    seen: set[str] = set()
    for record in dataset.records:
        state_ref = str(record.get("state_ref") or "")
        if not state_ref or state_ref in seen:
            continue
        seen.add(state_ref)
        examples.append((record, dataset.load_latent_for_record(record)))
        if len(examples) >= int(limit):
            break
    if len(examples) < int(limit):
        raise ValueError(
            f"Requested {int(limit)} unique states from {dataset.jsonl_path}, found only {len(examples)} "
            f"among {len(dataset.records)} scanned records. Increase --max-*-scan-records."
        )
    return examples


def _captured_tensor(value: Any, name: str) -> torch.Tensor:
    if isinstance(value, tuple):
        value = value[0]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Diagnostic hook {name} did not receive a tensor.")
    return value


def _capture_output(captures: dict[str, torch.Tensor], name: str):
    def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        captures[name] = _captured_tensor(output, name)

    return hook


def _capture_input(captures: dict[str, torch.Tensor], name: str):
    def hook(_module: nn.Module, inputs: tuple[Any, ...]) -> None:
        if not inputs:
            raise RuntimeError(f"Diagnostic pre-hook {name} received no inputs.")
        captures[name] = _captured_tensor(inputs[0], name)

    return hook


def _global_spatial_backbone(adapter: nn.Module) -> TensorPatchAlignmentAdapter:
    backbone = adapter.global_adapter if isinstance(adapter, HybridGlobalLocalAdapter) else adapter
    if not isinstance(backbone, TensorPatchAlignmentAdapter) or backbone.adapter_type != "spatial_transformer":
        raise TypeError(
            "This diagnostic requires a spatial_transformer checkpoint with one row-major token per latent position."
        )
    return backbone


@torch.no_grad()
def extract_spatial_token_stages(
    adapter: nn.Module,
    latent_map: torch.Tensor,
    question_embeds: torch.Tensor | None = None,
    question_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Run exact model forwards while hooks expose position-preserving intermediate states."""

    global_backbone = _global_spatial_backbone(adapter)
    captures: dict[str, torch.Tensor] = {
        "latent_raw": global_backbone.flatten_latent_tokens(latent_map),
    }
    handles: list[Any] = [
        global_backbone.latent_projection.register_forward_hook(
            _capture_output(captures, "global_projected_content")
        ),
        global_backbone.local_residual_projection.register_forward_hook(
            _capture_output(captures, "global_local_residual_projection")
        ),
        global_backbone.output.register_forward_pre_hook(
            _capture_input(captures, "global_pre_output")
        ),
    ]
    if global_backbone.blocks:
        handles.append(
            global_backbone.blocks[0].register_forward_pre_hook(
                _capture_input(captures, "global_projected_plus_2d_position")
            )
        )
    for index, block in enumerate(global_backbone.blocks, start=1):
        handles.append(block.register_forward_hook(_capture_output(captures, f"global_spatial_block_{index}")))
    try:
        global_soft = global_backbone.forward_soft_prompts(latent_map)
    finally:
        for handle in handles:
            handle.remove()
    captures["global_soft_prompt"] = global_soft

    if not isinstance(adapter, HybridGlobalLocalAdapter):
        return captures
    local_adapter = adapter.local_adapter
    if not isinstance(local_adapter, ResidualQuestionConditionedAdapter):
        raise TypeError("The Stage-2 routing diagnostic supports ResidualQuestionConditionedAdapter only.")
    if local_adapter.backbone.adapter_type != "spatial_transformer":
        raise TypeError("The conditioned Stage-2 backbone is not a spatial_transformer.")
    if question_embeds is None or question_mask is None:
        raise ValueError("Stage-2 extraction requires contextual natural-language question states and their mask.")

    conditioned_backbone = local_adapter.backbone
    handles = [
        conditioned_backbone.latent_projection.register_forward_hook(
            _capture_output(captures, "conditioned_projected_content")
        ),
        conditioned_backbone.local_residual_projection.register_forward_hook(
            _capture_output(captures, "conditioned_local_residual_projection")
        ),
        conditioned_backbone.output.register_forward_pre_hook(
            _capture_input(captures, "conditioned_pre_output")
        ),
    ]
    if local_adapter.text_blocks:
        handles.append(
            local_adapter.text_blocks[0].register_forward_pre_hook(
                _capture_input(captures, "conditioned_projected_plus_2d_position")
            )
        )
    for index, (text_block, spatial_block) in enumerate(
        zip(local_adapter.text_blocks, conditioned_backbone.blocks),
        start=1,
    ):
        handles.append(
            text_block.register_forward_hook(_capture_output(captures, f"conditioned_text_cross_attention_{index}"))
        )
        handles.append(
            spatial_block.register_forward_hook(_capture_output(captures, f"conditioned_spatial_block_{index}"))
        )
    try:
        conditioned_soft = local_adapter(
            latent_map,
            question_embeds=question_embeds,
            question_mask=question_mask,
            structured_query=None,
        )
    finally:
        for handle in handles:
            handle.remove()
    captures["conditioned_soft_prompt"] = conditioned_soft
    question_residual = local_adapter.gate.to(dtype=conditioned_soft.dtype) * (
        conditioned_soft - global_soft
    )
    captures["question_residual"] = question_residual
    captures["combined_soft_prompt"] = global_soft + question_residual
    return captures


def _question_context(
    llm: nn.Module,
    adapter: nn.Module,
    tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not isinstance(adapter, HybridGlobalLocalAdapter):
        return None, None
    result = contextual_adapter_question_context(
        llm=llm,
        adapter=adapter,
        tokenizer=tokenizer,
        records=records,
        device=device,
        max_prompt_tokens=int(args.max_prompt_tokens),
        layer_index=int(args.local_context_layer),
        prompt_template=str(args.prompt_template),
    )
    if result is None:
        raise RuntimeError("The checkpoint did not produce contextual natural-language question states.")
    return result


@torch.no_grad()
def collect_stage_features(
    examples: Sequence[tuple[Mapping[str, Any], torch.Tensor]],
    llm: nn.Module,
    adapter: nn.Module,
    tokenizer: Any,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    feature_chunks: dict[str, list[torch.Tensor]] = defaultdict(list)
    target_chunks: list[torch.Tensor] = []
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    batch_size = int(args.feature_batch_size)

    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        records = [record for record, _latent in batch]
        latent_map = torch.stack([latent for _record, latent in batch]).to(device)
        if not (0 <= int(args.value_channel) < int(latent_map.shape[1])):
            raise IndexError(
                f"value_channel={int(args.value_channel)} is outside latent shape {tuple(latent_map.shape)}."
            )
        height, width = int(latent_map.shape[-2]), int(latent_map.shape[-1])
        token_count = height * width
        positions_per_state = min(int(args.positions_per_state), token_count)
        positions = torch.stack(
            [torch.randperm(token_count, generator=generator)[:positions_per_state] for _ in batch]
        ).to(device)
        question_embeds, question_mask = _question_context(
            llm=llm,
            adapter=adapter,
            tokenizer=tokenizer,
            records=records,
            args=args,
            device=device,
        )
        stages = extract_spatial_token_stages(
            adapter=adapter,
            latent_map=latent_map,
            question_embeds=question_embeds,
            question_mask=question_mask,
        )
        targets = latent_map[:, int(args.value_channel)].reshape(len(batch), token_count)
        target_chunks.append(torch.gather(targets, dim=1, index=positions).reshape(-1).float().cpu())
        for name, stage in stages.items():
            if stage.ndim != 3 or int(stage.shape[1]) != token_count:
                raise ValueError(
                    f"Stage {name} broke row-major token correspondence: shape={tuple(stage.shape)}, "
                    f"expected token count={token_count}."
                )
            gather_index = positions.unsqueeze(-1).expand(-1, -1, int(stage.shape[-1]))
            selected = torch.gather(stage, dim=1, index=gather_index)
            feature_chunks[name].append(selected.reshape(-1, int(stage.shape[-1])).float().cpu())

    return (
        {name: torch.cat(chunks, dim=0) for name, chunks in feature_chunks.items()},
        torch.cat(target_chunks, dim=0),
    )


def regression_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    tolerance: float,
) -> dict[str, float]:
    prediction = prediction.detach().float().reshape(-1).cpu()
    target = target.detach().float().reshape(-1).cpu()
    error = prediction - target
    mse = error.square().mean()
    target_centered = target - target.mean()
    total = target_centered.square().sum()
    r2 = 1.0 - error.square().sum() / total.clamp_min(1.0e-12)
    prediction_centered = prediction - prediction.mean()
    pearson = (prediction_centered * target_centered).sum() / (
        prediction_centered.square().sum().sqrt()
        * target_centered.square().sum().sqrt()
    ).clamp_min(1.0e-12)
    return {
        "r2": float(r2.item()),
        "rmse": float(mse.sqrt().item()),
        "mae": float(error.abs().mean().item()),
        "pearson": float(pearson.item()),
        "within_tolerance_fraction": float(error.abs().le(float(tolerance)).float().mean().item()),
    }


@torch.no_grad()
def fit_ridge_readout(
    train_features: torch.Tensor,
    train_target: torch.Tensor,
    val_features: torch.Tensor,
    val_target: torch.Tensor,
    ridge: float,
    tolerance: float,
    device: torch.device,
) -> dict[str, Any]:
    train_x = train_features.to(device=device, dtype=torch.float32)
    val_x = val_features.to(device=device, dtype=torch.float32)
    train_y = train_target.to(device=device, dtype=torch.float32).reshape(-1, 1)
    val_y = val_target.to(device=device, dtype=torch.float32).reshape(-1, 1)
    x_mean = train_x.mean(dim=0, keepdim=True)
    x_std = train_x.std(dim=0, unbiased=False, keepdim=True).clamp_min(1.0e-6)
    y_mean = train_y.mean(dim=0, keepdim=True)
    y_std = train_y.std(dim=0, unbiased=False, keepdim=True).clamp_min(1.0e-6)
    train_x = (train_x - x_mean) / x_std
    val_x = (val_x - x_mean) / x_std
    normalized_y = (train_y - y_mean) / y_std

    sample_count, feature_dim = int(train_x.shape[0]), int(train_x.shape[1])
    regularization = max(float(ridge) * sample_count, 1.0e-8)
    if feature_dim <= sample_count:
        matrix = train_x.T @ train_x
        matrix.diagonal().add_(regularization)
        weights = torch.linalg.solve(matrix, train_x.T @ normalized_y)
    else:
        matrix = train_x @ train_x.T
        matrix.diagonal().add_(regularization)
        dual = torch.linalg.solve(matrix, normalized_y)
        weights = train_x.T @ dual
    train_prediction = (train_x @ weights) * y_std + y_mean
    val_prediction = (val_x @ weights) * y_std + y_mean
    train_metrics = regression_metrics(train_prediction, train_y, tolerance=tolerance)
    val_metrics = regression_metrics(val_prediction, val_y, tolerance=tolerance)
    mean_baseline = torch.full_like(val_y, float(y_mean.item()))
    return {
        "feature_dim": feature_dim,
        "train_examples": sample_count,
        "val_examples": int(val_x.shape[0]),
        "ridge": float(ridge),
        "train": train_metrics,
        "val": val_metrics,
        "val_train_mean_baseline": regression_metrics(mean_baseline, val_y, tolerance=tolerance),
    }


def coordinate_routing_groups(
    records: Sequence[Mapping[str, Any]],
    max_groups: int,
    max_questions: int,
) -> list[list[tuple[Mapping[str, Any], tuple[int, int]]]]:
    if int(max_groups) <= 0:
        return []
    grouped: dict[tuple[str, str, str], list[tuple[Mapping[str, Any], tuple[int, int]]]] = defaultdict(list)
    for record in records:
        coordinate = point_coordinate_from_record(record)
        state_ref = str(record.get("state_ref") or "")
        if coordinate is None or not state_ref:
            continue
        key = (state_ref, str(record.get("task_type") or ""), str(record.get("field") or ""))
        grouped[key].append((record, coordinate))

    result: list[list[tuple[Mapping[str, Any], tuple[int, int]]]] = []
    for key in sorted(grouped):
        candidates = grouped[key]
        if len({coordinate for _record, coordinate in candidates}) < 2:
            continue
        selected = candidates[: int(max_questions)]
        if len({coordinate for _record, coordinate in selected}) < 2:
            first_coordinate = selected[0][1]
            replacement = next(item for item in candidates if item[1] != first_coordinate)
            selected[-1] = replacement
        result.append(selected)
        if len(result) >= int(max_groups):
            break
    return result


def routing_group_statistics(
    features: torch.Tensor,
    coordinates: Sequence[tuple[int, int]],
    height: int,
    width: int,
) -> dict[str, list[float]]:
    if features.ndim != 3 or int(features.shape[0]) != len(coordinates):
        raise ValueError("Routing features must have shape [questions,spatial_tokens,channels].")
    token_count = int(height) * int(width)
    if int(features.shape[1]) != token_count:
        raise ValueError("Routing feature token count does not match the latent grid.")
    centered = features.float() - features.float().mean(dim=0, keepdim=True)
    scores = torch.linalg.vector_norm(centered, dim=-1)
    payload: dict[str, list[float]] = defaultdict(list)
    eps = 1.0e-12
    for question_index, (row, col) in enumerate(coordinates):
        target_index = row_major_index(row, col, height=height, width=width)
        row_scores = scores[question_index]
        target_score = row_scores[target_index]
        strict_above = int((row_scores > target_score + eps).sum().item())
        tied = max(0, int((row_scores.sub(target_score).abs() <= eps).sum().item()) - 1)
        rank = 1.0 + strict_above + 0.5 * tied
        non_target = torch.cat([row_scores[:target_index], row_scores[target_index + 1 :]])
        total_score = row_scores.sum()
        if token_count <= 1 or float(total_score.item()) <= eps:
            normalized_entropy = 1.0
        else:
            probabilities = row_scores / total_score
            entropy = -(probabilities * probabilities.clamp_min(eps).log()).sum()
            normalized_entropy = float((entropy / math.log(token_count)).item())
        payload["target_rank"].append(rank)
        payload["target_percentile"].append(
            1.0 if token_count == 1 else 1.0 - (rank - 1.0) / (token_count - 1.0)
        )
        target_ratio = (
            1.0
            if non_target.numel() == 0
            else float((target_score / non_target.mean().clamp_min(eps)).item())
        )
        payload["target_to_non_target_ratio"].append(target_ratio)
        payload["normalized_change_entropy"].append(normalized_entropy)
        payload["change_rms"].append(float(row_scores.square().mean().sqrt().item()))

    flattened = features.float().flatten(1)
    for left in range(len(coordinates)):
        for right in range(left + 1, len(coordinates)):
            distance = float((flattened[left] - flattened[right]).square().mean().sqrt().item())
            key = (
                "same_coordinate_pair_l2"
                if coordinates[left] == coordinates[right]
                else "different_coordinate_pair_l2"
            )
            payload[key].append(distance)
    return dict(payload)


def _mean(values: Sequence[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(torch.tensor(list(values), dtype=torch.float64).median().item())


def summarize_routing_statistics(
    raw: Mapping[str, Sequence[float]],
    token_count: int,
) -> dict[str, Any]:
    ranks = list(raw.get("target_rank", []))
    top5_count = min(5, int(token_count))
    return {
        "records": len(ranks),
        "token_count": int(token_count),
        "chance_top1": 1.0 / max(1, int(token_count)),
        "chance_top5": top5_count / max(1, int(token_count)),
        "target_top1": _mean([float(rank <= 1.0) for rank in ranks]),
        "target_top5": _mean([float(rank <= top5_count) for rank in ranks]),
        "target_rank_mean": _mean(ranks),
        "target_rank_median": _median(ranks),
        "target_percentile_mean": _mean(raw.get("target_percentile", [])),
        "target_to_non_target_ratio_mean": _mean(raw.get("target_to_non_target_ratio", [])),
        "normalized_change_entropy_mean": _mean(raw.get("normalized_change_entropy", [])),
        "change_rms_mean": _mean(raw.get("change_rms", [])),
        "same_coordinate_pair_l2_mean": _mean(raw.get("same_coordinate_pair_l2", [])),
        "same_coordinate_pairs": len(raw.get("same_coordinate_pair_l2", [])),
        "different_coordinate_pair_l2_mean": _mean(raw.get("different_coordinate_pair_l2", [])),
        "different_coordinate_pairs": len(raw.get("different_coordinate_pair_l2", [])),
    }


@torch.no_grad()
def diagnose_coordinate_routing(
    groups: Sequence[Sequence[tuple[Mapping[str, Any], tuple[int, int]]]],
    dataset: TensorReadoutQADataset,
    llm: nn.Module,
    adapter: nn.Module,
    tokenizer: Any,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    if not groups:
        return {
            "available": False,
            "reason": "No same-state/task groups with at least two distinct single-point coordinates were found.",
        }
    aggregate: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    grid: tuple[int, int] | None = None
    records_used = 0
    for group in groups:
        records = [record for record, _coordinate in group]
        coordinates = [coordinate for _record, coordinate in group]
        latent = dataset.load_latent_for_record(records[0])
        height, width = int(latent.shape[-2]), int(latent.shape[-1])
        grid = (height, width)
        latent_batch = latent.unsqueeze(0).expand(len(records), -1, -1, -1).contiguous().to(device)
        question_embeds, question_mask = _question_context(
            llm=llm,
            adapter=adapter,
            tokenizer=tokenizer,
            records=records,
            args=args,
            device=device,
        )
        stages = extract_spatial_token_stages(
            adapter=adapter,
            latent_map=latent_batch,
            question_embeds=question_embeds,
            question_mask=question_mask,
        )
        question_dependent = {
            name: value
            for name, value in stages.items()
            if name.startswith("conditioned_text_cross_attention_")
            or name.startswith("conditioned_spatial_block_")
            or name in {"conditioned_soft_prompt", "question_residual", "combined_soft_prompt"}
        }
        for name, values in question_dependent.items():
            statistics = routing_group_statistics(
                values.detach().float().cpu(),
                coordinates=coordinates,
                height=height,
                width=width,
            )
            for key, numbers in statistics.items():
                aggregate[name][key].extend(numbers)
        records_used += len(records)

    assert grid is not None
    token_count = grid[0] * grid[1]
    return {
        "available": True,
        "method": (
            "For each same-tensor/same-task natural-language question group, subtract the group-mean "
            "feature at every spatial token, then rank the queried row-major token by residual-change norm."
        ),
        "limitation": (
            "Low localization is evidence of a weak adapter-internal coordinate route, but not proof that the "
            "later frozen LLM cannot route from the question text to spatial prefix tokens."
        ),
        "groups": len(groups),
        "records": records_used,
        "grid": list(grid),
        "stages": {
            name: summarize_routing_statistics(raw, token_count=token_count)
            for name, raw in aggregate.items()
        },
    }


def adapter_structure_summary(adapter: nn.Module) -> dict[str, Any]:
    global_backbone = _global_spatial_backbone(adapter)
    summary: dict[str, Any] = {
        "adapter_class": type(adapter).__name__,
        "global_adapter_type": str(global_backbone.adapter_type),
        "latent_grid": list(global_backbone.latent_grid),
        "spatial_tokens": int(global_backbone.soft_prompt_tokens),
        "global_spatial_blocks": len(global_backbone.blocks),
        "question_cross_attention": False,
    }
    if not isinstance(adapter, HybridGlobalLocalAdapter) or not isinstance(
        adapter.local_adapter, ResidualQuestionConditionedAdapter
    ):
        return summary
    local = adapter.local_adapter
    fusion_weights = torch.softmax(local.text_layer_logits.detach().float(), dim=0)
    summary.update(
        {
            "question_cross_attention": True,
            "question_attention_direction": "spatial_tokens_query_natural_language_keys_values",
            "question_context_layers": list(local.context_layers),
            "question_context_layer_weights": [float(value) for value in fusion_weights.cpu().tolist()],
            "question_hidden_states_detached": True,
            "text_cross_attention_blocks": len(local.text_blocks),
            "text_cross_attention_gates": [float(block.gate.detach().float().item()) for block in local.text_blocks],
            "text_attention_output_projection_rms": [
                float(block.attention.out_proj.weight.detach().float().square().mean().sqrt().item())
                for block in local.text_blocks
            ],
            "residual_gate": float(local.gate.detach().float().item()),
            "conditioned_backbone_frozen": bool(local.freeze_backbone),
            "global_backbone_frozen": bool(adapter.freeze_global),
            "final_composition": "global_prompt + gate * (conditioned_prompt - global_prompt)",
        }
    )
    return summary


def default_output_path(checkpoint: str | Path) -> Path:
    checkpoint = Path(checkpoint)
    return checkpoint.parent / "diagnostics" / f"{checkpoint.stem}_spatial_token_readout.json"


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(Path(args.checkpoint).expanduser(), map_location="cpu")
    restored_runtime_args = restore_checkpoint_runtime_args(args, checkpoint)
    apply_runtime_environment(args)
    torch.manual_seed(int(args.seed))
    device = resolve_device(args.device)
    probe_device = device if str(args.probe_device).lower() == "auto" else resolve_device(args.probe_device)
    print(
        f"diagnostic=start device={device} checkpoint={args.checkpoint} "
        f"model={args.model_name_or_path}"
    )
    tokenizer, llm, model_dtype = load_tokenizer_and_llm(args, device)

    train_scan = int(args.max_train_scan_records or max(int(args.probe_train_states) * 32, 256))
    val_scan = int(
        args.max_val_scan_records
        or max(int(args.probe_val_states) * 32, int(args.routing_groups) * 32, 256)
    )
    dataset_kwargs = {
        "latent_dir": args.latent_dir,
        "prefer_record_latent_ref": bool(args.prefer_record_latent_ref),
        "shuffle_seed": int(args.shuffle_seed),
        "latent_cache_size": int(args.diagnostic_latent_cache_size),
    }
    train_dataset = TensorReadoutQADataset(
        qa_path(args.qa_dir, args.train_split),
        max_records=train_scan,
        **dataset_kwargs,
    )
    val_dataset = TensorReadoutQADataset(
        qa_path(args.qa_dir, args.val_split),
        max_records=val_scan,
        **dataset_kwargs,
    )
    first_latent = train_dataset.load_latent_for_record(train_dataset.records[0])
    adapter = adapter_from_checkpoint(
        checkpoint=checkpoint,
        latent_shape=tuple(int(value) for value in first_latent.shape),
        llm_hidden_size=int(llm.get_input_embeddings().embedding_dim),
    ).to(device)
    del checkpoint
    adapter.eval()
    structure = adapter_structure_summary(adapter)
    if not bool(structure["question_cross_attention"]):
        print("diagnostic=warning checkpoint_has_no_stage2_question_cross_attention")

    train_examples = unique_state_examples(train_dataset, int(args.probe_train_states))
    val_examples = unique_state_examples(val_dataset, int(args.probe_val_states))
    print(
        f"diagnostic=value_features train_states={len(train_examples)} val_states={len(val_examples)} "
        f"positions_per_state={int(args.positions_per_state)}"
    )
    train_features, train_target = collect_stage_features(
        train_examples,
        llm=llm,
        adapter=adapter,
        tokenizer=tokenizer,
        args=args,
        device=device,
        seed=int(args.seed),
    )
    val_features, val_target = collect_stage_features(
        val_examples,
        llm=llm,
        adapter=adapter,
        tokenizer=tokenizer,
        args=args,
        device=device,
        seed=int(args.seed) + 1,
    )
    if train_features.keys() != val_features.keys():
        raise RuntimeError("Train and validation extraction produced different stage sets.")
    value_decodability: dict[str, Any] = {}
    for name in train_features:
        value_decodability[name] = fit_ridge_readout(
            train_features=train_features[name],
            train_target=train_target,
            val_features=val_features[name],
            val_target=val_target,
            ridge=float(args.linear_ridge),
            tolerance=float(args.value_tolerance),
            device=probe_device,
        )

    groups = coordinate_routing_groups(
        val_dataset.records,
        max_groups=int(args.routing_groups),
        max_questions=int(args.routing_max_questions),
    )
    print(f"diagnostic=coordinate_routing groups={len(groups)}")
    routing = diagnose_coordinate_routing(
        groups=groups,
        dataset=val_dataset,
        llm=llm,
        adapter=adapter,
        tokenizer=tokenizer,
        args=args,
        device=device,
    )

    output_path = Path(args.output) if args.output else default_output_path(args.checkpoint)
    dump_json(
        output_path,
        {
            "checkpoint": str(args.checkpoint),
            "checkpoint_runtime_args_restored": restored_runtime_args,
            "model_dtype": str(model_dtype).replace("torch.", ""),
            "value_target": {
                "source": "latent_map[value_channel,row,column]",
                "value_channel": int(args.value_channel),
                "tolerance": float(args.value_tolerance),
                "important_assumption": (
                    "For preserve_input_channels=true, channel 0 is the exact standardized input value. "
                    "latent_raw should therefore be nearly perfectly decodable; otherwise the checkpoint/data "
                    "pair or --value-channel is wrong."
                ),
            },
            "adapter_structure": structure,
            "value_decodability": value_decodability,
            "coordinate_routing": routing,
            "reading_guide": [
                (
                    "A sharp R2 drop between adjacent stages identifies where same-position numeric "
                    "information ceases to be linearly accessible. It does not prove information-theoretic loss."
                ),
                (
                    "Good final-token value R2 with near-chance target localization points primarily "
                    "to coordinate routing, not tensor encoding."
                ),
                (
                    "Poor latent_raw R2 invalidates the diagnostic assumption; do not interpret later "
                    "stages until the latent source/value channel is fixed."
                ),
                "These are frozen post-hoc probes and never update the AE, adapter, or LLM.",
            ],
        },
    )
    print(f"diagnostic=complete stages={len(value_decodability)} output={output_path}")


if __name__ == "__main__":
    main()
