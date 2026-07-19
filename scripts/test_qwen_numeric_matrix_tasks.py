from __future__ import annotations

import argparse
import re
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_tensor_patch_text_alignment import (
    AlignmentAnchor,
    PDEBenchPatchTextDataset,
    build_axis_split_plan,
    build_numeric_probe_anchor,
    build_patch_encoder_config,
    build_patch_records,
    dump_json,
    dtype_from_name,
    load_checkpoint_and_config,
    normalize_patch_batch,
    parse_csv,
    probe_targets_from_patches,
    resolve_device,
    serialize_tensor_values,
    validate_field_shapes,
)
from tensor_compression.downstream.pdebench import resolve_checkpoint_field_keys
from tensor_compression.utils.pipeline_config import (
    first_nested,
    load_yaml_mapping,
    resolve_path_string,
    value_to_csv,
)

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "scripts/test_qwen_numeric_matrix_tasks.py requires transformers. "
        "Install it with: pip install transformers accelerate safetensors"
    ) from exc


NUMERIC_ANSWER_PATTERN = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
THINKING_MODES = ("disabled", "auto")


def parse_generated_numeric_answer(text: str) -> float | None:
    values: list[float] = []
    for match in NUMERIC_ANSWER_PATTERN.finditer(str(text)):
        try:
            value = float(match.group(0))
        except ValueError:  # pragma: no cover - guarded by the regular expression
            continue
        if np.isfinite(value):
            values.append(value)
    return values[-1] if values else None


def numeric_probe_question(anchor: AlignmentAnchor, channel_count: int) -> str:
    parameters = tuple(int(value) for value in anchor.probe_parameters)

    def point_location(channel: int, row: int, col: int) -> str:
        location = f"row {row + 1}, column {col + 1}"
        if int(channel_count) > 1:
            location += f" in channel {channel + 1}"
        return location

    family = str(anchor.probe_family)
    if family == "point_value":
        channel, row, col = parameters
        return f"What is the value at {point_location(channel, row, col)}?"
    if family in {"point_difference", "point_mean"}:
        channel, row_a, col_a, row_b, col_b = parameters
        location_a = point_location(channel, row_a, col_a)
        location_b = point_location(channel, row_b, col_b)
        if family == "point_difference":
            return f"What is the value at {location_a} minus the value at {location_b}?"
        return f"What is the arithmetic mean of the values at {location_a} and {location_b}?"
    if family in {"region_mean", "region_range"}:
        channel, row, col, size = parameters
        location = f"rows {row + 1}-{row + size} and columns {col + 1}-{col + size}"
        if int(channel_count) > 1:
            location += f" in channel {channel + 1}"
        if family == "region_mean":
            return f"What is the arithmetic mean of all values over {location}?"
        return f"What is the maximum value minus the minimum value over {location}?"
    raise ValueError(f"Unsupported numeric test probe family: {family!r}.")


def build_qwen_numeric_test_prompt(
    patch: torch.Tensor,
    anchor: AlignmentAnchor,
    decimal_places: int,
) -> str:
    if patch.ndim != 3:
        raise ValueError(f"Qwen numeric test expects [C,H,W], got {tuple(patch.shape)}.")
    matrix_text = serialize_tensor_values(patch, int(decimal_places))
    question = numeric_probe_question(anchor, int(patch.shape[0]))
    return (
        "The following numeric tensor is serialized in [channel][row][column] order. "
        "All row, column, and channel indices are 1-based.\n"
        f"Tensor:\n{matrix_text}\n"
        f"Task: {question}\n"
        f"Return exactly one decimal number rounded to {int(decimal_places)} decimal places. "
        "Do not include an explanation.\nAnswer:"
    )


def normalize_thinking_mode(value: Any) -> str:
    mode = str(value).strip().lower()
    if mode not in THINKING_MODES:
        raise ValueError(
            f"Unsupported thinking mode {value!r}; expected one of {THINKING_MODES}."
        )
    return mode


def render_numeric_chat_prompt(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    thinking_mode: str,
) -> tuple[str, bool, bool]:
    """Render a chat prompt while remaining compatible with non-Qwen3 templates."""
    mode = normalize_thinking_mode(thinking_mode)
    uses_chat_template = bool(hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template)
    if not uses_chat_template:
        rendered = (
            f"System: {messages[0]['content']}\n"
            f"User: {messages[1]['content']}\nAssistant:"
        )
        return rendered, False, False

    if mode == "auto":
        return (
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            ),
            True,
            False,
        )

    template_text = str(tokenizer.chat_template)
    if "enable_thinking" not in template_text:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return rendered, True, False

    # Qwen3 consumes this keyword in its template. Older/non-thinking templates may
    # reject it, in which case the ordinary template is the compatible fallback.
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError as exc:
        if "enable_thinking" not in str(exc):
            raise
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return rendered, True, False
    return rendered, True, True


@torch.no_grad()
def generate_qwen_numeric_answer(
    llm: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    device: torch.device,
    max_input_tokens: int,
    max_new_tokens: int,
    thinking_mode: str,
) -> tuple[str, int, dict[str, Any]]:
    messages = [
        {
            "role": "system",
            "content": "You perform exact numeric lookup and arithmetic over explicitly provided tensors.",
        },
        {"role": "user", "content": str(prompt)},
    ]
    rendered, uses_chat_template, thinking_control_applied = render_numeric_chat_prompt(
        tokenizer,
        messages,
        thinking_mode,
    )
    encoded = tokenizer(
        rendered,
        return_tensors="pt",
        truncation=False,
        add_special_tokens=not uses_chat_template,
    )
    input_tokens = int(encoded["input_ids"].shape[1])
    if input_tokens > int(max_input_tokens):
        raise ValueError(
            f"Qwen numeric test prompt has {input_tokens} tokens, exceeding max_input_tokens="
            f"{int(max_input_tokens)}."
        )
    model_inputs = {key: value.to(device) for key, value in encoded.items() if torch.is_tensor(value)}
    output_ids = llm.generate(
        **model_inputs,
        max_new_tokens=int(max_new_tokens),
        do_sample=False,
        use_cache=True,
        pad_token_id=int(tokenizer.pad_token_id),
        eos_token_id=int(tokenizer.eos_token_id),
    )
    generated_ids = output_ids[0, input_tokens:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return generated_text, input_tokens, {
        "thinking_mode": normalize_thinking_mode(thinking_mode),
        "chat_template_used": bool(uses_chat_template),
        "thinking_control_applied": bool(thinking_control_applied),
        "generated_tokens": int(generated_ids.shape[0]),
        "hit_max_new_tokens": int(generated_ids.shape[0]) >= int(max_new_tokens),
    }


def aggregate_qwen_numeric_test_cases(
    cases: Sequence[Mapping[str, Any]],
    absolute_tolerance: float,
) -> dict[str, Any]:
    def summarize(selected: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
        count = len(selected)
        parsed = [case for case in selected if case.get("prediction") is not None]
        errors = [float(case["absolute_error"]) for case in parsed]
        return {
            "records": float(count),
            "parsed_fraction": float(len(parsed) / max(1, count)),
            "rounded_exact_accuracy": float(
                sum(bool(case.get("rounded_exact")) for case in selected) / max(1, count)
            ),
            "within_tolerance_accuracy": float(
                sum(bool(case.get("within_tolerance")) for case in selected) / max(1, count)
            ),
            "parsed_mae": float(sum(errors) / len(errors)) if errors else None,
        }

    sources = sorted({str(case["source"]) for case in cases})
    families = sorted({str(case["probe_family"]) for case in cases})
    return {
        "absolute_tolerance": float(absolute_tolerance),
        "macro": summarize(cases),
        "by_source": {
            source: summarize([case for case in cases if str(case["source"]) == source])
            for source in sources
        },
        "by_family": {
            family: summarize([case for case in cases if str(case["probe_family"]) == family])
            for family in families
        },
        "cases": [dict(case) for case in cases],
    }


def config_value(config: Mapping[str, Any], path: str, default: Any) -> Any:
    value = first_nested(config, [path])
    return default if value is None else value


def resolve_test_data(
    config: Mapping[str, Any],
    dataset_records: int,
    seed: int,
) -> tuple[PDEBenchPatchTextDataset | None, tuple[int, int], dict[str, Any], bool]:
    if int(dataset_records) <= 0:
        return None, (1, 1), {}, False
    hdf5_value = first_nested(config, ["patch_alignment.hdf5_path", "data.hdf5_path"])
    if hdf5_value is None:
        raise ValueError("Real PDE tests require patch_alignment.hdf5_path in the config.")
    hdf5_path = resolve_path_string(hdf5_value, PROJECT_ROOT)
    fields = parse_csv(value_to_csv(first_nested(config, ["patch_alignment.fields", "data.fields"])))
    field_sampling_mode = str(config_value(config, "patch_alignment.field_sampling_mode", "channels")).lower()
    if field_sampling_mode not in {"channels", "single"}:
        raise ValueError("patch_alignment.field_sampling_mode must be channels or single.")
    patch_size = int(config_value(config, "patch_alignment.patch_size", 16))
    encoder_source = str(config_value(config, "patch_alignment.encoder_source", "patch_ae_config"))
    if encoder_source == "checkpoint":
        checkpoint_value = first_nested(
            config,
            ["patch_alignment.compressor_checkpoint", "compressor.checkpoint"],
        )
        if checkpoint_value is None:
            raise ValueError("encoder_source=checkpoint requires patch_alignment.compressor_checkpoint.")
        checkpoint_path = resolve_path_string(checkpoint_value, PROJECT_ROOT)
        compressor_config_value = first_nested(config, ["patch_alignment.compressor_config"])
        compressor_config_path = (
            resolve_path_string(compressor_config_value, PROJECT_ROOT)
            if compressor_config_value is not None
            else None
        )
        _checkpoint, compressor_config = load_checkpoint_and_config(
            checkpoint_path,
            compressor_config_path,
        )
        if not fields:
            fields = resolve_checkpoint_field_keys(compressor_config)
    elif encoder_source == "patch_ae_config":
        if not fields:
            raise ValueError("patch_ae_config requires patch_alignment.fields in the config.")
        encoder_fields = [fields[0]] if field_sampling_mode == "single" else fields
        compressor_config = build_patch_encoder_config(
            patch_encoder_cfg=first_nested(config, ["patch_alignment.patch_encoder"]),
            field_keys=encoder_fields,
            patch_size=patch_size,
        )
    else:
        raise ValueError(f"Unsupported patch_alignment.encoder_source={encoder_source!r}.")

    if not fields:
        raise ValueError("Real PDE tests require fields in the config or compressor checkpoint.")
    validate_field_shapes(hdf5_path, fields)
    split_plan = build_axis_split_plan(
        hdf5_path=hdf5_path,
        field=fields[0],
        sample_indices=str(config_value(config, "patch_alignment.sample_indices", "all")),
        time_indices=str(config_value(config, "patch_alignment.time_indices", "all")),
        split_mode=str(config_value(config, "patch_alignment.split_mode", "sample")),
        train_ratio=float(config_value(config, "patch_alignment.split_train_ratio", 0.8)),
        val_ratio=float(config_value(config, "patch_alignment.split_val_ratio", 0.1)),
        test_ratio=float(config_value(config, "patch_alignment.split_test_ratio", 0.1)),
        seed=int(seed),
    )
    records = build_patch_records(
        hdf5_path=hdf5_path,
        field=fields[0],
        record_fields=fields if field_sampling_mode == "single" else None,
        sample_indices=split_plan["samples"]["val"],
        time_indices=split_plan["times"]["val"],
        patch_size=patch_size,
        count=int(dataset_records),
        seed=int(seed) + 1,
        unique_records=True,
    )
    dataset = PDEBenchPatchTextDataset(
        hdf5_path=hdf5_path,
        field_keys=fields,
        records=records,
        patch_size=patch_size,
        decimal_places=int(config_value(config, "patch_alignment.text_decimal_places", 3)),
        prompt_template="plain",
        include_raw_text=False,
    )
    compressor_input_size = tuple(int(value) for value in compressor_config["model"]["input_size"])
    normalization_cfg = dict(compressor_config.get("data", {}).get("dataset", {}).get("normalization", {}))
    resize_to_input = bool(config_value(config, "patch_alignment.resize_patch_to_compressor_input", False))
    return dataset, compressor_input_size, normalization_cfg, resize_to_input


@torch.no_grad()
def run_numeric_matrix_test(args: argparse.Namespace) -> dict[str, Any]:
    config = load_yaml_mapping(args.config)
    seed = int(args.seed if args.seed is not None else config_value(config, "patch_alignment.seed", 42))
    torch.manual_seed(seed)
    model_name = first_nested(config, ["model.local_dir", "model.name_or_path"])
    if model_name is None:
        raise ValueError("The config must define model.local_dir or model.name_or_path.")
    cache_dir_value = first_nested(config, ["model.cache_dir", "storage.hf_home"])
    cache_dir = (
        resolve_path_string(cache_dir_value, PROJECT_ROOT)
        if cache_dir_value is not None
        else None
    )
    trust_remote_code = bool(config_value(config, "model.trust_remote_code", False))
    torch_dtype = str(config_value(config, "model.torch_dtype", "bfloat16"))
    thinking_mode = normalize_thinking_mode(
        args.thinking_mode
        if args.thinking_mode is not None
        else config_value(config, "model.thinking_mode", "disabled")
    )
    device = resolve_device(
        str(args.device if args.device is not None else config_value(config, "patch_alignment.device", "auto"))
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_name),
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = AutoModelForCausalLM.from_pretrained(
        str(model_name),
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
        dtype=dtype_from_name(torch_dtype),
    ).to(device)
    llm.eval()

    decimals = int(config_value(config, "patch_alignment.text_decimal_places", 3))
    families = parse_csv(value_to_csv(first_nested(config, ["patch_alignment.probe_families"])))
    if not families:
        families = ["point_value", "point_difference", "point_mean"]
    region_size = int(config_value(config, "patch_alignment.probe_region_size", 4))
    max_anchor_tokens = int(config_value(config, "patch_alignment.max_shared_suffix_tokens", 96))
    max_input_tokens = int(
        args.max_input_tokens
        if args.max_input_tokens is not None
        else config_value(config, "patch_alignment.max_text_tokens", 3072)
    )
    dataset, compressor_input_size, normalization_cfg, resize_to_input = resolve_test_data(
        config,
        int(args.dataset_records),
        seed,
    )

    case_specs: list[tuple[str, torch.Tensor, int]] = []
    synthetic_size = int(args.synthetic_size)
    for index in range(int(args.synthetic_records)):
        values = (
            (torch.arange(synthetic_size * synthetic_size, dtype=torch.float32) + 3 * index) % 29 - 14
        ) / 10.0
        case_specs.append(("synthetic_short", values.reshape(1, synthetic_size, synthetic_size), index))
    if dataset is not None:
        for index in range(len(dataset)):
            patch = normalize_patch_batch(
                torch.stack([dataset[index]["patch"]], dim=0),
                compressor_input_size,
                normalization_cfg,
                resize_to_input,
            )[0]
            case_specs.append(("pde_patch", patch, index))
    if not case_specs:
        raise ValueError("At least one synthetic or PDE test record is required.")

    scale = float(10**decimals)
    cases: list[dict[str, Any]] = []
    for case_index, (source, patch, source_index) in enumerate(case_specs):
        visible_patch = torch.round(patch.detach().float() * scale) / scale
        effective_region_size = min(region_size, int(visible_patch.shape[-1]) - 1)
        anchor = build_numeric_probe_anchor(
            tokenizer=tokenizer,
            patch_size=int(visible_patch.shape[-1]),
            channel_count=int(visible_patch.shape[0]),
            families=families,
            region_size=effective_region_size,
            probe_index=case_index,
            seed=seed + 400_000,
            max_anchor_tokens=max_anchor_tokens,
        )
        expected_tensor, target_ids = probe_targets_from_patches(
            anchor,
            visible_patch.unsqueeze(0),
            decimals,
        )
        expected = float(expected_tensor[0].item())
        prompt = build_qwen_numeric_test_prompt(visible_patch, anchor, decimals)
        generated_text, input_tokens, generation_control = generate_qwen_numeric_answer(
            llm,
            tokenizer,
            prompt,
            device,
            max_input_tokens,
            int(args.max_new_tokens),
            thinking_mode,
        )
        if bool(generation_control["hit_max_new_tokens"]):
            prediction = None
            prediction_status = "generation_truncated"
        else:
            prediction = parse_generated_numeric_answer(generated_text)
            prediction_status = "numeric" if prediction is not None else "no_numeric_answer"
        absolute_error = abs(float(prediction) - expected) if prediction is not None else None
        rounded_exact = (
            int(np.rint(float(prediction) * scale)) == int(target_ids[0].item())
            if prediction is not None
            else False
        )
        cases.append(
            {
                "source": source,
                "source_index": int(source_index),
                "probe_family": str(anchor.probe_family),
                "probe_parameters": list(anchor.probe_parameters),
                "input_tokens": int(input_tokens),
                "prompt": prompt,
                "expected": expected,
                "generated_text": generated_text,
                "generation_control": generation_control,
                "prediction": prediction,
                "prediction_status": prediction_status,
                "absolute_error": absolute_error,
                "rounded_exact": bool(rounded_exact),
                "within_tolerance": bool(
                    absolute_error is not None and absolute_error <= float(args.absolute_tolerance)
                ),
            }
        )
    metrics = aggregate_qwen_numeric_test_cases(cases, float(args.absolute_tolerance))
    controls = {
        tuple(
            sorted(
                (str(key), value)
                for key, value in case["generation_control"].items()
                if key != "generated_tokens" and key != "hit_max_new_tokens"
            )
        )
        for case in cases
    }
    metrics["generation_control"] = {
        "requested_thinking_mode": thinking_mode,
        "observed_variants": [dict(items) for items in sorted(controls)],
        "truncated_records": int(
            sum(bool(case["generation_control"]["hit_max_new_tokens"]) for case in cases)
        ),
    }
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test whether a frozen Qwen model can answer numeric matrix lookup/arithmetic questions."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dataset-records", type=int, default=6)
    parser.add_argument("--synthetic-records", type=int, default=6)
    parser.add_argument("--synthetic-size", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument(
        "--thinking-mode",
        choices=THINKING_MODES,
        default=None,
        help=(
            "Chat-template mode. 'disabled' is the default and passes enable_thinking=False "
            "when supported; 'auto' leaves the model/template default unchanged."
        ),
    )
    parser.add_argument("--absolute-tolerance", type=float, default=0.02)
    args = parser.parse_args()
    for name in ("dataset_records", "synthetic_records"):
        if int(getattr(args, name)) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative.")
    if int(args.dataset_records) + int(args.synthetic_records) <= 0:
        raise ValueError("At least one test record is required.")
    if int(args.synthetic_records) > 0 and int(args.synthetic_size) <= 1:
        raise ValueError("--synthetic-size must be greater than 1.")
    if args.max_input_tokens is not None and int(args.max_input_tokens) <= 0:
        raise ValueError("--max-input-tokens must be positive.")
    if int(args.max_new_tokens) <= 0:
        raise ValueError("--max-new-tokens must be positive.")
    if float(args.absolute_tolerance) < 0.0:
        raise ValueError("--absolute-tolerance must be non-negative.")
    return args


def main() -> None:
    args = parse_args()
    config = load_yaml_mapping(args.config)
    output_root_value = first_nested(config, ["patch_alignment.output_root", "llm_training.output_root"])
    output_root = (
        Path(resolve_path_string(output_root_value, PROJECT_ROOT))
        if output_root_value is not None
        else PROJECT_ROOT / "outputs"
    )
    output_path = (
        Path(args.output).expanduser()
        if args.output is not None
        else output_root / "diagnostics" / f"{time.strftime('%Y%m%d_%H%M%S')}_qwen_numeric_matrix_test.json"
    )
    metrics = run_numeric_matrix_test(args)
    dump_json(output_path, metrics)
    macro = metrics["macro"]
    parsed_mae = macro.get("parsed_mae")
    parsed_mae_text = f"{float(parsed_mae):.6g}" if isinstance(parsed_mae, (int, float)) else "n/a"
    source_text = " ".join(
        f"{source}={source_metrics['within_tolerance_accuracy']:.3f}"
        for source, source_metrics in metrics["by_source"].items()
    )
    print(
        f"records={macro['records']:.0f} parsed={macro['parsed_fraction']:.3f} "
        f"exact={macro['rounded_exact_accuracy']:.3f} "
        f"within_tol={macro['within_tolerance_accuracy']:.3f} "
        f"mae={parsed_mae_text} {source_text} "
        f"thinking={metrics['generation_control']['requested_thinking_mode']} "
        f"truncated={metrics['generation_control']['truncated_records']}"
    )
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
