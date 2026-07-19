from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_tensor_patch_text_alignment import (  # noqa: E402
    AlignmentAnchor,
    PDEBenchPatchTextDataset,
    PROBE_FAMILIES,
    alignment_anchors_from_args,
    build_axis_split_plan,
    build_patch_records,
    collate_patch_text,
    dtype_from_name,
    llm_backbone,
    normalize_patch_batch,
    parse_csv,
    probe_targets_from_patches,
    resolve_device,
    resolve_field_keys,
    serialize_patch_batch,
    serialize_tensor_value_batch,
    tokenize_contents_with_anchor,
    transformer_block_hidden_states,
    validate_field_shapes,
    validate_teacher_tensor_source,
)
from tensor_compression.utils.pipeline_config import (  # noqa: E402
    first_nested,
    load_yaml_mapping,
    resolve_path_string,
    value_to_csv,
)

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "scripts/scan_tensor_teacher_layers.py requires transformers. "
        "Install it with: pip install transformers accelerate safetensors"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan every frozen LLM layer at the configured tensor-text readout position. "
            "This script does not train or load the AE, Q-Former, or alignment projector."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--anchor-mode", choices=("eos", "representation", "probe"), default=None)
    parser.add_argument("--split", choices=("train", "val", "test"), default=None)
    parser.add_argument("--records", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--layers",
        default=None,
        help="Comma-separated Hugging Face hidden-state indices, or 'all' for 1..num_hidden_layers.",
    )
    parser.add_argument("--probe-count", type=int, default=None)
    parser.add_argument("--perturbation-scale", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def config_value(
    cli_value: Any,
    config: Mapping[str, Any],
    paths: Sequence[str],
    default: Any,
) -> Any:
    if cli_value is not None:
        return cli_value
    return first_nested(config, paths, default)


def resolve_scan_args(args: argparse.Namespace, config: Mapping[str, Any]) -> SimpleNamespace:
    scan_prefix = "patch_alignment.layer_scan"
    root = SimpleNamespace()
    root.hdf5_path = resolve_path_string(
        config_value(None, config, ["patch_alignment.hdf5_path", "data.hdf5_path"], None),
        PROJECT_ROOT,
    )
    root.fields = value_to_csv(
        config_value(None, config, ["patch_alignment.fields", "data.fields"], None)
    )
    root.field_sampling_mode = str(
        config_value(None, config, ["patch_alignment.field_sampling_mode"], "channels")
    )
    root.patch_size = int(config_value(None, config, ["patch_alignment.patch_size"], 16))
    root.sample_indices = value_to_csv(
        config_value(None, config, ["patch_alignment.sample_indices"], "all")
    )
    root.time_indices = value_to_csv(
        config_value(None, config, ["patch_alignment.time_indices"], "all")
    )
    root.split_mode = str(config_value(None, config, ["patch_alignment.split_mode"], "sample"))
    root.split_train_ratio = float(
        config_value(None, config, ["patch_alignment.split_train_ratio"], 0.8)
    )
    root.split_val_ratio = float(
        config_value(None, config, ["patch_alignment.split_val_ratio"], 0.1)
    )
    root.split_test_ratio = float(
        config_value(None, config, ["patch_alignment.split_test_ratio"], 0.1)
    )
    root.seed = int(config_value(None, config, ["patch_alignment.seed", "runtime.seed"], 42))
    root.unique_records = bool(config_value(None, config, ["patch_alignment.unique_records"], True))
    root.teacher_text_source = str(
        config_value(None, config, ["patch_alignment.teacher_text_source"], "normalized")
    )
    root.alignment_text_layout = str(
        config_value(None, config, ["patch_alignment.alignment_text_layout"], "values_shared_suffix")
    )
    root.alignment_anchor_mode = str(
        config_value(
            args.anchor_mode,
            config,
            [f"{scan_prefix}.anchor_mode", "patch_alignment.alignment_anchor_mode"],
            "probe",
        )
    )
    root.representation_suffix = str(
        config_value(None, config, ["patch_alignment.representation_suffix"], "\nRepresentation:")
    )
    root.probe_families = [
        family.lower()
        for family in parse_csv(
            config_value(
                None,
                config,
                ["patch_alignment.probe_families"],
                "point_value,point_difference,point_mean,region_mean,region_range",
            )
        )
    ]
    root.probe_region_size = int(
        config_value(None, config, ["patch_alignment.probe_region_size"], 4)
    )
    root.evaluation_probe_count = int(
        config_value(
            args.probe_count,
            config,
            [f"{scan_prefix}.probe_count", "patch_alignment.evaluation_probe_count"],
            5,
        )
    )
    root.max_shared_suffix_tokens = int(
        config_value(None, config, ["patch_alignment.max_shared_suffix_tokens"], 96)
    )
    root.text_prompt_template = str(
        config_value(None, config, ["patch_alignment.text_prompt_template"], "compact")
    )
    root.text_decimal_places = int(
        config_value(None, config, ["patch_alignment.text_decimal_places"], 4)
    )
    root.max_text_tokens = int(
        config_value(None, config, ["patch_alignment.max_text_tokens"], 3072)
    )
    root.fail_on_text_max_length_hit = bool(
        config_value(None, config, ["patch_alignment.fail_on_text_max_length_hit"], True)
    )
    root.model_name_or_path = str(
        config_value(None, config, ["model.local_dir", "model.name_or_path"], "")
    )
    root.cache_dir = resolve_path_string(
        config_value(None, config, ["model.cache_dir", "storage.hf_home"], None),
        PROJECT_ROOT,
    )
    root.trust_remote_code = bool(config_value(None, config, ["model.trust_remote_code"], False))
    root.torch_dtype = str(config_value(None, config, ["model.torch_dtype"], "bfloat16"))
    root.device = str(
        config_value(args.device, config, [f"{scan_prefix}.device", "patch_alignment.device"], "auto")
    )
    root.split = str(config_value(args.split, config, [f"{scan_prefix}.split"], "val"))
    root.records = int(config_value(args.records, config, [f"{scan_prefix}.records"], 128))
    root.batch_size = int(config_value(args.batch_size, config, [f"{scan_prefix}.batch_size"], 8))
    root.layers = str(config_value(args.layers, config, [f"{scan_prefix}.layers"], "all"))
    root.perturbation_scale = float(
        config_value(args.perturbation_scale, config, [f"{scan_prefix}.perturbation_scale"], 0.01)
    )
    root.output_root = resolve_path_string(
        config_value(None, config, ["patch_alignment.output_root", "llm_training.output_root"], "outputs"),
        PROJECT_ROOT,
    )
    root.output = resolve_path_string(args.output, PROJECT_ROOT) if args.output else None
    patch_encoder = config_value(None, config, ["patch_alignment.patch_encoder"], {}) or {}
    root.normalization = dict(patch_encoder.get("normalization") or {"mode": "none"})

    if not root.hdf5_path or not root.fields or not root.model_name_or_path:
        raise ValueError("The config must define patch_alignment.hdf5_path/fields and model.name_or_path.")
    if root.records <= 1 or root.batch_size <= 0:
        raise ValueError("layer_scan.records must exceed 1 and layer_scan.batch_size must be positive.")
    if root.perturbation_scale < 0.0:
        raise ValueError("layer_scan.perturbation_scale cannot be negative.")
    if root.alignment_text_layout != "values_shared_suffix" and root.alignment_anchor_mode != "representation":
        raise ValueError("eos/probe layer scans require alignment_text_layout=values_shared_suffix.")
    if root.alignment_anchor_mode not in {"eos", "representation", "probe"}:
        raise ValueError("layer_scan.anchor_mode must be eos, representation, or probe.")
    if root.alignment_anchor_mode == "probe":
        unsupported = sorted(set(root.probe_families) - set(PROBE_FAMILIES))
        if unsupported:
            raise ValueError(f"Unsupported patch_alignment.probe_families: {unsupported}.")
        if not root.probe_families or root.evaluation_probe_count <= 0:
            raise ValueError("Probe layer scans require non-empty probe_families and a positive probe_count.")
        if root.probe_region_size <= 0 or root.probe_region_size >= root.patch_size:
            raise ValueError("patch_alignment.probe_region_size must be between 1 and patch_size - 1.")
        if not 0 <= root.text_decimal_places <= 8:
            raise ValueError("Probe layer scans require text_decimal_places between 0 and 8.")
    validate_teacher_tensor_source(root.normalization, root.teacher_text_source)
    return root


def parse_layer_indices(raw: str, num_hidden_layers: int) -> list[int]:
    if str(raw).strip().lower() == "all":
        return list(range(1, int(num_hidden_layers) + 1))
    layers = sorted({int(value) for value in parse_csv(raw)})
    if not layers:
        raise ValueError("--layers must be 'all' or contain at least one hidden-state index.")
    invalid = [layer for layer in layers if layer < 0 or layer > int(num_hidden_layers)]
    if invalid:
        raise ValueError(
            f"Layer indices {invalid} are outside hidden_states[0..{int(num_hidden_layers)}]."
        )
    return layers


def readout_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device).unsqueeze(0)
    return (attention_mask.long() * positions).amax(dim=1)


def readout_hidden(hidden: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if hidden.ndim != 3 or hidden.shape[0] != indices.shape[0]:
        raise ValueError(
            f"Expected hidden [batch,sequence,dim] for {indices.shape[0]} readout indices, "
            f"got {tuple(hidden.shape)}."
        )
    rows = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[rows, indices]


def representation_metrics(hidden: torch.Tensor) -> dict[str, float]:
    values = hidden.detach().float().cpu()
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError(f"Expected at least two [record,hidden] vectors, got {tuple(values.shape)}.")
    normalized = F.normalize(values, dim=-1)
    similarity = normalized @ normalized.T
    mask = ~torch.eye(values.shape[0], dtype=torch.bool)
    off_diagonal = similarity[mask]
    nearest = similarity.masked_fill(~mask, -torch.inf).amax(dim=1)
    centered = values - values.mean(dim=0, keepdim=True)
    # The sample Gram matrix has the same non-zero spectrum as the much larger feature covariance.
    spectrum = torch.linalg.eigvalsh(centered @ centered.T).clamp_min(0.0)
    energy = spectrum.sum()
    if float(energy.item()) <= torch.finfo(torch.float32).eps:
        entropy_rank = torch.tensor(0.0)
        participation_rank = torch.tensor(0.0)
    else:
        spectrum = spectrum / energy
        nonzero = spectrum[spectrum > 0]
        entropy_rank = torch.exp(-(nonzero * nonzero.log()).sum())
        participation_rank = 1.0 / spectrum.square().sum().clamp_min(torch.finfo(torch.float32).eps)
    quantiles = torch.quantile(off_diagonal, torch.tensor([0.05, 0.5, 0.95]))
    return {
        "record_count": float(values.shape[0]),
        "hidden_norm_mean": float(values.norm(dim=-1).mean().item()),
        "off_diagonal_cosine_mean": float(off_diagonal.mean().item()),
        "off_diagonal_cosine_std": float(off_diagonal.std(unbiased=False).item()),
        "off_diagonal_cosine_q05": float(quantiles[0].item()),
        "off_diagonal_cosine_median": float(quantiles[1].item()),
        "off_diagonal_cosine_q95": float(quantiles[2].item()),
        "nearest_neighbor_cosine_mean": float(nearest.mean().item()),
        "centered_rms": float(centered.square().mean().sqrt().item()),
        "effective_rank_entropy": float(entropy_rank.item()),
        "effective_rank_participation": float(participation_rank.item()),
    }


def paired_metrics(original: torch.Tensor, perturbed: torch.Tensor) -> dict[str, float]:
    original = original.detach().float().cpu()
    perturbed = perturbed.detach().float().cpu()
    cosine = F.cosine_similarity(original, perturbed, dim=-1)
    relative_l2 = (perturbed - original).norm(dim=-1) / original.norm(dim=-1).clamp_min(1.0e-12)
    return {
        "perturbed_pair_cosine_mean": float(cosine.mean().item()),
        "perturbed_pair_cosine_std": float(cosine.std(unbiased=False).item()),
        "perturbed_relative_l2_mean": float(relative_l2.mean().item()),
    }


def perturb_patches(patches: torch.Tensor, scale: float) -> torch.Tensor:
    if float(scale) <= 0.0:
        return patches
    height, width = int(patches.shape[-2]), int(patches.shape[-1])
    rows = torch.arange(height, dtype=patches.dtype).view(height, 1)
    cols = torch.arange(width, dtype=patches.dtype).view(1, width)
    pattern = ((rows + cols).remainder(2.0) * 2.0 - 1.0).view(1, 1, height, width)
    channel_std = patches.float().flatten(2).std(dim=-1, unbiased=False).to(patches.dtype)
    amplitude = channel_std.clamp_min(torch.finfo(patches.dtype).eps).unsqueeze(-1).unsqueeze(-1)
    return patches + float(scale) * amplitude * pattern


def probe_target_and_control_perturbations(
    patches: torch.Tensor,
    anchor: AlignmentAnchor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply equal-cardinality perturbations on and outside the values named by a probe."""
    if anchor.mode != "probe":
        raise ValueError("Target/control perturbations require a probe anchor.")
    target = patches.clone()
    control = patches.clone()
    height, width = int(patches.shape[-2]), int(patches.shape[-1])
    family = str(anchor.probe_family)
    parameters = tuple(int(value) for value in anchor.probe_parameters)

    for batch_index in range(int(patches.shape[0])):
        if family == "point_value":
            channel, row, col = parameters
            target_positions = [(row, col)]
            target_signs = [1.0]
            excluded = set(target_positions)
        elif family in {"point_difference", "point_mean"}:
            channel, row_a, col_a, row_b, col_b = parameters
            target_positions = [(row_a, col_a), (row_b, col_b)]
            target_signs = [1.0, -1.0] if family == "point_difference" else [1.0, 1.0]
            excluded = set(target_positions)
        elif family in {"region_mean", "region_range"}:
            channel, row, col, size = parameters
            region_positions = [
                (region_row, region_col)
                for region_row in range(row, row + size)
                for region_col in range(col, col + size)
            ]
            excluded = set(region_positions)
            if family == "region_mean":
                target_positions = region_positions
                target_signs = [1.0] * len(target_positions)
            else:
                region = patches[batch_index, channel, row : row + size, col : col + size].flatten()
                max_index = int(region.argmax().item())
                min_index = int(region.argmin().item())
                if min_index == max_index:
                    min_index = (max_index + 1) % int(region.numel())
                target_positions = [region_positions[max_index], region_positions[min_index]]
                target_signs = [1.0, -1.0]
        else:
            raise ValueError(f"Unsupported probe family for layer scan: {family!r}.")

        offsets = [
            (row_offset, col_offset)
            for row_offset in range(height)
            for col_offset in range(width)
            if row_offset != 0 or col_offset != 0
        ]
        offsets.sort(
            key=lambda offset: (
                abs(offset[0] - height // 2) + abs(offset[1] - width // 2),
                offset[0],
                offset[1],
            )
        )
        control_positions: list[tuple[int, int]] | None = None
        for row_offset, col_offset in offsets:
            translated = [
                ((target_row + row_offset) % height, (target_col + col_offset) % width)
                for target_row, target_col in target_positions
            ]
            if len(set(translated)) == len(target_positions) and not set(translated).intersection(excluded):
                control_positions = translated
                break
        if control_positions is None:
            raise ValueError(
                f"Probe {anchor.name!r} covers too much of the patch for a translated equal-size disjoint control: "
                f"target_values={len(target_positions)}, patch_shape=({height},{width})."
            )
        amplitude = (
            patches[batch_index, channel].float().std(unbiased=False)
            .clamp_min(torch.finfo(torch.float32).eps)
            .to(dtype=patches.dtype)
            * float(scale)
        )
        for (target_row, target_col), (control_row, control_col), sign in zip(
            target_positions,
            control_positions,
            target_signs,
            strict=True,
        ):
            target[batch_index, channel, target_row, target_col] += amplitude * float(sign)
            control[batch_index, channel, control_row, control_col] += amplitude * float(sign)
    return target, control


def teacher_texts_from_patches(
    batch: Mapping[str, Any],
    source_patches: torch.Tensor,
    args: SimpleNamespace,
) -> list[str]:
    if args.alignment_text_layout == "values_shared_suffix":
        return serialize_tensor_value_batch(source_patches, int(args.text_decimal_places))
    return serialize_patch_batch(
        records=batch["records"],
        patches=source_patches,
        decimal_places=int(args.text_decimal_places),
        prompt_template=str(args.text_prompt_template),
    )


def tokenize_teacher_batch(
    tokenizer: Any,
    texts: Sequence[str],
    args: SimpleNamespace,
    anchor: AlignmentAnchor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if args.alignment_text_layout == "values_shared_suffix":
        if anchor is None:
            raise ValueError("values_shared_suffix requires an explicit alignment anchor.")
        packed = tokenize_contents_with_anchor(
            tokenizer=tokenizer,
            contents=texts,
            anchor=anchor,
            max_tokens=int(args.max_text_tokens),
            require_under_max_length=bool(args.fail_on_text_max_length_hit),
            context="teacher layer scan",
        )
        return packed.input_ids, packed.attention_mask
    encoded = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=int(args.max_text_tokens),
        return_tensors="pt",
    )
    return encoded["input_ids"], encoded["attention_mask"]


@torch.no_grad()
def collect_anchor_hidden(
    *,
    llm: Any,
    tokenizer: Any,
    loader: DataLoader,
    device: torch.device,
    args: SimpleNamespace,
    layers: Sequence[int],
    anchor: AlignmentAnchor | None,
) -> tuple[
    dict[int, torch.Tensor],
    dict[int, torch.Tensor],
    dict[int, torch.Tensor] | None,
    dict[str, Any],
]:
    original_by_layer: dict[int, list[torch.Tensor]] = {int(layer): [] for layer in layers}
    target_perturbed_by_layer: dict[int, list[torch.Tensor]] = {int(layer): [] for layer in layers}
    control_perturbed_by_layer: dict[int, list[torch.Tensor]] | None = (
        {int(layer): [] for layer in layers} if anchor is not None and anchor.mode == "probe" else None
    )
    position_values: list[int] = []
    target_perturbed_text_changes = 0
    control_perturbed_text_changes = 0
    total_texts = 0
    target_probe_change_sum = 0.0
    control_probe_change_sum = 0.0
    total_probe_targets = 0
    readout_token_ids: set[int] = set()

    for batch in loader:
        normalized = normalize_patch_batch(
            batch["patch"],
            [int(args.patch_size), int(args.patch_size)],
            args.normalization,
            False,
        )
        source = normalized if args.teacher_text_source == "normalized" else batch["patch"].float()
        if anchor is not None and anchor.mode == "probe":
            target_perturbed, control_perturbed = probe_target_and_control_perturbations(
                source,
                anchor,
                float(args.perturbation_scale),
            )
            original_targets, _ = probe_targets_from_patches(anchor, source, int(args.text_decimal_places))
            target_targets, _ = probe_targets_from_patches(
                anchor, target_perturbed, int(args.text_decimal_places)
            )
            control_targets, _ = probe_targets_from_patches(
                anchor, control_perturbed, int(args.text_decimal_places)
            )
            target_probe_change_sum += float((target_targets - original_targets).abs().sum().item())
            control_probe_change_sum += float((control_targets - original_targets).abs().sum().item())
            total_probe_targets += int(original_targets.numel())
        else:
            target_perturbed = perturb_patches(source, float(args.perturbation_scale))
            control_perturbed = None
        original_texts = teacher_texts_from_patches(batch, source, args)
        target_perturbed_texts = teacher_texts_from_patches(batch, target_perturbed, args)
        control_perturbed_texts = (
            teacher_texts_from_patches(batch, control_perturbed, args)
            if control_perturbed is not None
            else None
        )
        total_texts += len(original_texts)
        target_perturbed_text_changes += sum(
            a != b for a, b in zip(original_texts, target_perturbed_texts, strict=True)
        )
        if control_perturbed_texts is not None:
            control_perturbed_text_changes += sum(
                a != b for a, b in zip(original_texts, control_perturbed_texts, strict=True)
            )

        variants: list[tuple[Sequence[str], dict[int, list[torch.Tensor]]]] = [
            (original_texts, original_by_layer),
            (target_perturbed_texts, target_perturbed_by_layer),
        ]
        if control_perturbed_texts is not None and control_perturbed_by_layer is not None:
            variants.append((control_perturbed_texts, control_perturbed_by_layer))
        for texts, destination in variants:
            input_ids, attention_mask = tokenize_teacher_batch(tokenizer, texts, args, anchor)
            indices = readout_indices(attention_mask)
            if destination is original_by_layer:
                position_values.extend(int(value) for value in indices.tolist())
                row_ids = input_ids[torch.arange(input_ids.shape[0]), indices]
                readout_token_ids.update(int(value) for value in row_ids.tolist())
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            indices = indices.to(device)
            hidden_states = transformer_block_hidden_states(
                llm,
                input_ids=input_ids,
                attention_mask=attention_mask,
                layer_indices=layers,
            )
            expected_sequence = int(input_ids.shape[1])
            for layer in layers:
                hidden = hidden_states[int(layer)]
                if int(hidden.shape[1]) != expected_sequence:
                    raise RuntimeError(
                        f"Transformer layer {layer} changed sequence length from {expected_sequence} "
                        f"to {int(hidden.shape[1])}."
                    )
                destination[int(layer)].append(readout_hidden(hidden, indices).float().cpu())

    original = {layer: torch.cat(parts, dim=0) for layer, parts in original_by_layer.items()}
    target_perturbed = {
        layer: torch.cat(parts, dim=0) for layer, parts in target_perturbed_by_layer.items()
    }
    control_perturbed = (
        {layer: torch.cat(parts, dim=0) for layer, parts in control_perturbed_by_layer.items()}
        if control_perturbed_by_layer is not None
        else None
    )
    metadata = {
        "readout_position_zero_based_min": min(position_values),
        "readout_position_zero_based_max": max(position_values),
        "readout_position_one_based_min": min(position_values) + 1,
        "readout_position_one_based_max": max(position_values) + 1,
        "readout_token_ids": sorted(readout_token_ids),
        "readout_tokens": [tokenizer.decode([token_id]) for token_id in sorted(readout_token_ids)],
        "target_perturbed_text_changed_fraction": float(
            target_perturbed_text_changes / max(1, total_texts)
        ),
        "control_perturbed_text_changed_fraction": float(
            control_perturbed_text_changes / max(1, total_texts)
        ),
        "target_probe_value_abs_change_mean": float(target_probe_change_sum / max(1, total_probe_targets)),
        "control_probe_value_abs_change_mean": float(control_probe_change_sum / max(1, total_probe_targets)),
    }
    return original, target_perturbed, control_perturbed, metadata


def average_layer_metrics(per_anchor: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    layers = sorted({layer for result in per_anchor.values() for layer in result["layers"]}, key=int)
    averaged: dict[str, dict[str, float]] = {}
    for layer in layers:
        rows = [result["layers"][layer] for result in per_anchor.values()]
        numeric_keys = sorted(set.intersection(*(set(row) for row in rows)))
        averaged[layer] = {
            key: float(sum(float(row[key]) for row in rows) / len(rows))
            for key in numeric_keys
            if isinstance(rows[0][key], (int, float))
        }
    return averaged


def print_summary(layers: Mapping[str, Mapping[str, float]]) -> None:
    has_control = all("target_to_control_sensitivity_ratio" in metrics for metrics in layers.values())
    if has_control:
        print("layer  pair_cos  nn_cos  eff_rank  target_l2  control_l2  target/control")
    else:
        print("layer  pair_cos  nn_cos  eff_rank  perturb_l2")
    for layer in sorted(layers, key=int):
        metrics = layers[layer]
        prefix = (
            f"{int(layer):>5d}  "
            f"{metrics['off_diagonal_cosine_mean']:>8.6f}  "
            f"{metrics['nearest_neighbor_cosine_mean']:>7.5f}  "
            f"{metrics['effective_rank_participation']:>8.2f}"
        )
        if has_control:
            print(
                prefix
                + f"  {metrics['target_perturbed_relative_l2_mean']:>9.6f}"
                + f"  {metrics['control_perturbed_relative_l2_mean']:>10.6f}"
                + f"  {metrics['target_to_control_sensitivity_ratio']:>14.6f}"
            )
        else:
            print(prefix + f"  {metrics['target_perturbed_relative_l2_mean']:>10.6f}")
    by_cosine = sorted(layers, key=lambda layer: layers[layer]["off_diagonal_cosine_mean"])
    by_rank = sorted(
        layers,
        key=lambda layer: layers[layer]["effective_rank_participation"],
        reverse=True,
    )
    print("lowest_pairwise_cosine_layers=" + ",".join(by_cosine[:5]))
    print("highest_effective_rank_layers=" + ",".join(by_rank[:5]))
    if has_control:
        by_sensitivity = sorted(
            layers,
            key=lambda layer: layers[layer]["target_to_control_sensitivity_ratio"],
            reverse=True,
        )
        print("highest_target_control_sensitivity_layers=" + ",".join(by_sensitivity[:5]))


def main() -> None:
    cli_args = parse_args()
    config = load_yaml_mapping(cli_args.config)
    args = resolve_scan_args(cli_args, config)
    torch.manual_seed(int(args.seed))
    device = resolve_device(args.device)

    field_keys = resolve_field_keys(args.fields, None)
    validate_field_shapes(args.hdf5_path, field_keys)
    first_field = field_keys[0]
    split_plan = build_axis_split_plan(
        hdf5_path=args.hdf5_path,
        field=first_field,
        sample_indices=str(args.sample_indices),
        time_indices=str(args.time_indices),
        split_mode=str(args.split_mode),
        train_ratio=float(args.split_train_ratio),
        val_ratio=float(args.split_val_ratio),
        test_ratio=float(args.split_test_ratio),
        seed=int(args.seed),
    )
    record_fields = field_keys if args.field_sampling_mode == "single" else None
    split_offset = {"train": 0, "val": 1, "test": 2}[args.split]
    records = build_patch_records(
        hdf5_path=args.hdf5_path,
        field=first_field,
        record_fields=record_fields,
        sample_indices=split_plan["samples"][args.split],
        time_indices=split_plan["times"][args.split],
        patch_size=int(args.patch_size),
        count=int(args.records),
        seed=int(args.seed) + split_offset,
        unique_records=bool(args.unique_records),
    )
    dataset = PDEBenchPatchTextDataset(
        hdf5_path=args.hdf5_path,
        field_keys=field_keys,
        records=records,
        patch_size=int(args.patch_size),
        decimal_places=int(args.text_decimal_places),
        prompt_template=str(args.text_prompt_template),
        include_raw_text=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_patch_text,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        trust_remote_code=bool(args.trust_remote_code),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = AutoModel.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        trust_remote_code=bool(args.trust_remote_code),
        dtype=dtype_from_name(str(args.torch_dtype)),
    )
    # Load only the base decoder; the scan never uses a causal LM head.
    backbone = llm_backbone(llm)
    backbone.to(device).eval()
    for parameter in llm.parameters():
        parameter.requires_grad_(False)
    num_hidden_layers = int(llm.config.num_hidden_layers)
    layers = parse_layer_indices(args.layers, num_hidden_layers)
    anchors: list[AlignmentAnchor | None]
    if args.alignment_text_layout == "values_shared_suffix":
        anchors = list(alignment_anchors_from_args(tokenizer, args, evaluation=True))
    else:
        anchors = [None]

    per_anchor: dict[str, dict[str, Any]] = {}
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    for anchor_index, anchor in enumerate(anchors):
        anchor_name = anchor.name if anchor is not None else "legacy_prompt"
        print(f"scan_anchor={anchor_index + 1}/{len(anchors)} name={anchor_name}")
        original, target_perturbed, control_perturbed, metadata = collect_anchor_hidden(
            llm=llm,
            tokenizer=tokenizer,
            loader=loader,
            device=device,
            args=args,
            layers=layers,
            anchor=anchor,
        )
        layer_metrics: dict[str, dict[str, float]] = {}
        for layer in layers:
            metrics = representation_metrics(original[layer])
            metrics.update(
                {
                    f"target_{key}": value
                    for key, value in paired_metrics(original[layer], target_perturbed[layer]).items()
                }
            )
            if control_perturbed is not None:
                control_metrics = paired_metrics(original[layer], control_perturbed[layer])
                metrics.update({f"control_{key}": value for key, value in control_metrics.items()})
                metrics["target_to_control_sensitivity_ratio"] = (
                    metrics["target_perturbed_relative_l2_mean"]
                    / max(1.0e-12, metrics["control_perturbed_relative_l2_mean"])
                )
                metrics["target_minus_control_relative_l2"] = (
                    metrics["target_perturbed_relative_l2_mean"]
                    - metrics["control_perturbed_relative_l2_mean"]
                )
            layer_metrics[str(layer)] = metrics
        per_anchor[anchor_name] = {
            "anchor": (
                {
                    "name": anchor.name,
                    "mode": anchor.mode,
                    "text": anchor.text,
                    "token_ids": list(anchor.token_ids),
                    "probe_family": anchor.probe_family,
                    "probe_template_index": anchor.probe_template_index,
                    "probe_parameters": list(anchor.probe_parameters),
                }
                if anchor is not None
                else None
            ),
            "readout": metadata,
            "layers": layer_metrics,
        }

    averaged = average_layer_metrics(per_anchor)
    payload = {
        "started_at": started_at,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": str(cli_args.config),
        "model_name_or_path": args.model_name_or_path,
        "num_transformer_layers": num_hidden_layers,
        "hidden_state_index_contract": (
            "hidden_states[0] is the input embedding; hidden_states[k] is the output after k transformer blocks. "
            "Sequence positions are unchanged across all layers."
        ),
        "scan": {
            "split": args.split,
            "records": int(args.records),
            "batch_size": int(args.batch_size),
            "layers": layers,
            "fields": field_keys,
            "field_sampling_mode": args.field_sampling_mode,
            "teacher_text_source": args.teacher_text_source,
            "normalization": args.normalization,
            "alignment_text_layout": args.alignment_text_layout,
            "alignment_anchor_mode": args.alignment_anchor_mode,
            "text_decimal_places": int(args.text_decimal_places),
            "perturbation_scale": float(args.perturbation_scale),
        },
        "metric_guidance": {
            "off_diagonal_cosine_mean": "Lower means less cross-record angular collapse, but is not sufficient alone.",
            "effective_rank_participation": "Higher means variation occupies more independent directions.",
            "target_perturbed_relative_l2_mean": (
                "Relative hidden change after perturbing exactly the values named by the probe."
            ),
            "control_perturbed_relative_l2_mean": (
                "Relative hidden change after an equal-cardinality perturbation outside the probe support."
            ),
            "target_to_control_sensitivity_ratio": (
                "Values above 1 indicate stronger readout sensitivity to probe-relevant than off-target values."
            ),
        },
        "averaged_across_anchors": averaged,
        "per_anchor": per_anchor,
    }
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = Path(args.output_root) / (
            f"teacher_layer_scan_{timestamp}_{args.alignment_anchor_mode}.json"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    print_summary(averaged)
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
