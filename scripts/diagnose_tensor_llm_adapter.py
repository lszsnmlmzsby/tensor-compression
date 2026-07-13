from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tensor_compression.downstream.pdebench import resolve_device  # noqa: E402
from tensor_compression.utils import dump_json  # noqa: E402

from scripts.train_tensor_llm_adapter import (  # noqa: E402
    IGNORE_INDEX,
    TensorReadoutQADataset,
    TensorSoftPromptAdapter,
    adapter_from_checkpoint,
    adapter_soft_embeds,
    apply_config_defaults,
    apply_runtime_environment,
    build_text_tensors,
    contextual_adapter_soft_embeds,
    load_tokenizer_and_llm,
    qa_path,
    score_candidate_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose whether tensor latents are correctly loaded, separated across states, "
            "and used by a trained tensor-LLM adapter."
        )
    )
    parser.add_argument("--config", type=str, default="configs/tensor_llm_adapter_pipeline.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="adapter_best.pt or adapter_last.pt.")
    parser.add_argument("--qa-dir", type=str, default=None)
    parser.add_argument("--latent-dir", type=str, default=None)
    parser.add_argument("--model-name-or-path", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--hf-home", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--train-split", type=str, default=None)
    parser.add_argument("--val-split", type=str, default=None)
    parser.add_argument("--test-split", type=str, default=None)
    parser.add_argument("--max-train-records", type=int, default=None)
    parser.add_argument("--max-val-records", type=int, default=None)
    parser.add_argument("--max-test-records", type=int, default=None)
    parser.add_argument("--prefer-record-latent-ref", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--torch-dtype", type=str, default=None, choices=("auto", "float32", "float16", "bfloat16"))
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--shuffle-seed", type=int, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--records", type=int, default=64)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--output", type=str, default=None, help="JSONL output path.")
    parser.add_argument("--hidden-layers", type=str, default="0,-1", help="Comma-separated hidden-state layer ids.")
    parser.add_argument("--max-choice-records", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--eval-choice-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    parser.add_argument("--ce-loss-weight", type=float, default=None)
    parser.add_argument("--choice-ce-loss-weight", type=float, default=None)
    parser.add_argument("--ranking-loss-weight", type=float, default=None)
    parser.add_argument("--ranking-loss-margin", type=float, default=None)
    parser.add_argument(
        "--ranking-loss-negative",
        type=str,
        default=None,
        choices=("shuffled", "random", "no_latent", "zero_latent"),
    )
    parser.add_argument("--soft-prompt-tokens", type=int, default=None)
    parser.add_argument("--adapter-dim", type=int, default=None)
    parser.add_argument("--adapter-layers", type=int, default=None)
    parser.add_argument("--adapter-heads", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--latent-pos-encoding", type=str, default=None, choices=("none", "grid"))
    parser.add_argument("--question-conditioning", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--question-condition-gate-init", type=float, default=None)
    parser.add_argument("--structured-query-conditioning", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--soft-prompt-scale", type=float, default=None)
    parser.add_argument("--prompt-template", type=str, default=None, choices=("generic", "task_specific"))
    parser.add_argument("--max-prompt-tokens", type=int, default=None)
    parser.add_argument("--max-target-tokens", type=int, default=None)
    parser.add_argument("--append-eos", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--eval-baselines", type=str, default=None)
    parser.add_argument("--choice-score", type=str, default=None, choices=("mean", "sum"))
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--wandb-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--wandb-api-key", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default=None, choices=("online", "offline", "disabled"))
    parser.add_argument("--wandb-log-model", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    return apply_config_defaults(args)


def parse_layer_indices(raw: str, hidden_count: int) -> list[int]:
    indices: list[int] = []
    for part in str(raw).split(","):
        stripped = part.strip()
        if not stripped:
            continue
        index = int(stripped)
        if index < 0:
            index = hidden_count + index
        if index < 0 or index >= hidden_count:
            raise IndexError(f"Hidden layer index {part} resolved to {index}, outside [0,{hidden_count}).")
        if index not in indices:
            indices.append(index)
    return indices


def tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    detached = tensor.detach().float().cpu()
    flat = detached.reshape(-1)
    return {
        "shape": [int(dim) for dim in detached.shape],
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "l2_norm": float(torch.linalg.vector_norm(flat).item()),
    }


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.detach().float().reshape(-1)
    b_flat = b.detach().float().reshape(-1)
    return float(F.cosine_similarity(a_flat, b_flat, dim=0).item())


def l2_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(a.detach().float() - b.detach().float()).item())


def make_adapter_from_checkpoint(
    checkpoint: Mapping[str, Any],
    latent_shape: Sequence[int],
    llm_hidden_size: int,
):
    return adapter_from_checkpoint(checkpoint, latent_shape=latent_shape, llm_hidden_size=llm_hidden_size)


def state_fields(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "qa_id": record.get("qa_id"),
        "task_type": record.get("task_type"),
        "state_ref": record.get("state_ref"),
        "sample_index": record.get("sample_index"),
        "time_index": record.get("time_index"),
        "query": record.get("query") or record.get("question"),
        "choices": record.get("choices"),
        "answer": record.get("answer"),
        "oracle": record.get("oracle"),
    }


@torch.no_grad()
def soft_prompt_for_record(
    llm,
    adapter: TensorSoftPromptAdapter,
    tokenizer,
    record: Mapping[str, Any],
    latent_map: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    answer = str(record["answer"])
    input_ids, text_attention_mask, text_labels = build_text_tensors(
        records=[record],
        answers=[answer],
        tokenizer=tokenizer,
        max_prompt_tokens=int(args.max_prompt_tokens),
        max_target_tokens=int(args.max_target_tokens),
        append_eos=bool(args.append_eos),
        prompt_template=str(args.prompt_template),
    )
    input_ids = input_ids.to(device)
    text_attention_mask = text_attention_mask.to(device)
    text_labels = text_labels.to(device)
    latent_map = latent_map.unsqueeze(0).to(device)
    text_embeds = llm.get_input_embeddings()(input_ids)
    prompt_mask = text_labels.eq(IGNORE_INDEX) & text_attention_mask.bool()
    soft_embeds = contextual_adapter_soft_embeds(
        llm=llm,
        adapter=adapter,
        tokenizer=tokenizer,
        records=[record],
        latent_map=latent_map,
        device=device,
        max_prompt_tokens=int(args.max_prompt_tokens),
        layer_index=int(args.local_context_layer),
        mode=mode,
    )
    if soft_embeds is None:
        soft_embeds = adapter_soft_embeds(
            adapter,
            latent_map,
            text_embeds,
            question_embeds=text_embeds,
            question_mask=prompt_mask,
            records=[record],
            mode=mode,
        )
    return soft_embeds, text_embeds, text_attention_mask, text_labels


@torch.no_grad()
def hidden_state_summary(
    llm,
    soft_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    text_attention_mask: torch.Tensor,
    layer_indices: Sequence[int],
) -> dict[str, Any]:
    inputs_embeds = torch.cat([soft_embeds, text_embeds], dim=1)
    soft_attention = torch.ones(
        (text_attention_mask.shape[0], soft_embeds.shape[1]),
        dtype=text_attention_mask.dtype,
        device=text_attention_mask.device,
    )
    attention_mask = torch.cat([soft_attention, text_attention_mask], dim=1)
    outputs = llm(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    hidden_states = outputs.hidden_states
    selected: dict[str, Any] = {}
    soft_count = int(soft_embeds.shape[1])
    for index in layer_indices:
        hidden = hidden_states[int(index)]
        selected[str(index)] = {
            "all": tensor_summary(hidden),
            "soft_tokens": tensor_summary(hidden[:, :soft_count, :]),
            "text_tokens": tensor_summary(hidden[:, soft_count:, :]),
        }
    return selected


@torch.no_grad()
def nll_by_choice(
    llm,
    adapter: TensorSoftPromptAdapter,
    tokenizer,
    record: Mapping[str, Any],
    latent_map: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    mode: str,
) -> dict[str, Any]:
    choices = record.get("choices")
    if not isinstance(choices, Sequence) or isinstance(choices, str) or not choices:
        choices = [str(record["answer"])]
    string_choices = [str(choice) for choice in choices]
    scores = score_candidate_batch(
        llm=llm,
        adapter=adapter,
        tokenizer=tokenizer,
        records=[record for _ in string_choices],
        answers=string_choices,
        latent_map=latent_map.unsqueeze(0).repeat(len(string_choices), 1, 1, 1),
        device=device,
        max_prompt_tokens=int(args.max_prompt_tokens),
        max_target_tokens=int(args.max_target_tokens),
        append_eos=bool(args.append_eos),
        prompt_template=str(args.prompt_template),
        soft_prompt_mode=mode,
        choice_score=str(args.choice_score),
        local_context_layer=int(args.local_context_layer),
    )
    best_index = min(range(len(scores)), key=lambda index: scores[index])
    return {
        "choices": [
            {
                "label": label,
                "nll": float(score),
                "is_answer": label == str(record["answer"]),
            }
            for label, score in zip(string_choices, scores)
        ],
        "prediction": string_choices[best_index],
        "answer_nll": float(scores[string_choices.index(str(record["answer"]))])
        if str(record["answer"]) in string_choices
        else math.nan,
    }


def default_output_path(args: argparse.Namespace) -> Path:
    checkpoint = Path(args.checkpoint)
    return checkpoint.parent / f"{checkpoint.stem}_diagnostics_{args.split}.jsonl"


def main() -> None:
    args = parse_args()
    apply_runtime_environment(args)
    device = resolve_device(args.device)
    tokenizer, llm, model_dtype = load_tokenizer_and_llm(args, device)

    dataset = TensorReadoutQADataset(
        qa_path(args.qa_dir, args.split),
        latent_dir=args.latent_dir,
        max_records=None,
        prefer_record_latent_ref=bool(args.prefer_record_latent_ref),
        shuffle_seed=int(args.shuffle_seed),
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    first_latent = dataset[0]["latent_map"]
    adapter = make_adapter_from_checkpoint(
        checkpoint=checkpoint,
        latent_shape=tuple(int(value) for value in first_latent.shape),
        llm_hidden_size=int(llm.get_input_embeddings().embedding_dim),
    ).to(device)
    adapter.eval()

    output_path = Path(args.output) if args.output else default_output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = output_path.with_suffix(".summary.json")

    end_index = min(len(dataset), int(args.start_index) + int(args.records))
    selected_indices = list(range(max(0, int(args.start_index)), end_index))
    if not selected_indices:
        raise ValueError("No records selected for diagnostics.")

    records_written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for selected_count, index in enumerate(tqdm(selected_indices, desc="Diagnose records"), start=1):
            item = dataset[index]
            record = item["record"]
            correct_latent = item["latent_map"]
            shuffled_record = dataset.shuffled_record_for_index(index)
            shuffled_latent = dataset.load_latent_for_record(shuffled_record)

            correct_soft, text_embeds, text_attention_mask, _text_labels = soft_prompt_for_record(
                llm=llm,
                adapter=adapter,
                tokenizer=tokenizer,
                record=record,
                latent_map=correct_latent,
                args=args,
                device=device,
                mode="correct",
            )
            shuffled_soft, _text_embeds, _mask, _labels = soft_prompt_for_record(
                llm=llm,
                adapter=adapter,
                tokenizer=tokenizer,
                record=record,
                latent_map=shuffled_latent,
                args=args,
                device=device,
                mode="shuffled",
            )
            zero_soft, _text_embeds, _mask, _labels = soft_prompt_for_record(
                llm=llm,
                adapter=adapter,
                tokenizer=tokenizer,
                record=record,
                latent_map=correct_latent,
                args=args,
                device=device,
                mode="no_latent",
            )

            hidden_summary: dict[str, Any] = {}
            if selected_count <= int(args.max_choice_records):
                layer_count = int(getattr(llm.config, "num_hidden_layers", 0)) + 1
                layer_indices = parse_layer_indices(str(args.hidden_layers), layer_count)
                hidden_summary = hidden_state_summary(
                    llm=llm,
                    soft_embeds=correct_soft,
                    text_embeds=text_embeds,
                    text_attention_mask=text_attention_mask,
                    layer_indices=layer_indices,
                )

            nll_correct = nll_by_choice(llm, adapter, tokenizer, record, correct_latent, args, device, "correct")
            nll_shuffled = nll_by_choice(llm, adapter, tokenizer, record, shuffled_latent, args, device, "shuffled")
            nll_zero = nll_by_choice(llm, adapter, tokenizer, record, correct_latent, args, device, "no_latent")
            nll_zero_latent = nll_by_choice(
                llm,
                adapter,
                tokenizer,
                record,
                torch.zeros_like(correct_latent),
                args,
                device,
                "zero_latent",
            )

            payload = {
                "index": int(index),
                "record": state_fields(record),
                "shuffled_record": state_fields(shuffled_record),
                "same_sample_as_shuffled": record.get("sample_index") == shuffled_record.get("sample_index"),
                "delta_time_to_shuffled": (
                    int(shuffled_record["time_index"]) - int(record["time_index"])
                    if "time_index" in record and "time_index" in shuffled_record
                    else None
                ),
                "latent": {
                    "correct": tensor_summary(correct_latent),
                    "shuffled": tensor_summary(shuffled_latent),
                    "l2_distance": l2_distance(correct_latent, shuffled_latent),
                    "cosine_similarity": cosine_similarity(correct_latent, shuffled_latent),
                },
                "soft_prompt": {
                    "correct": tensor_summary(correct_soft),
                    "shuffled": tensor_summary(shuffled_soft),
                    "zero": tensor_summary(zero_soft),
                    "correct_vs_shuffled_l2": l2_distance(correct_soft, shuffled_soft),
                    "correct_vs_shuffled_cosine": cosine_similarity(correct_soft, shuffled_soft),
                    "correct_vs_zero_l2": l2_distance(correct_soft, zero_soft),
                    "correct_vs_zero_cosine": cosine_similarity(correct_soft, zero_soft),
                },
                "nll": {
                    "correct_latent": nll_correct,
                    "shuffled_latent": nll_shuffled,
                    "zero_soft_prompt": nll_zero,
                    "zero_latent": nll_zero_latent,
                    "answer_margin_shuffled_minus_correct": nll_shuffled["answer_nll"] - nll_correct["answer_nll"],
                    "answer_margin_zero_minus_correct": nll_zero["answer_nll"] - nll_correct["answer_nll"],
                    "answer_margin_zero_latent_minus_correct": nll_zero_latent["answer_nll"]
                    - nll_correct["answer_nll"],
                },
                "hidden_states": hidden_summary,
            }
            handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
            records_written += 1

    dump_json(
        summary_path,
        {
            "checkpoint": str(args.checkpoint),
            "output": str(output_path),
            "split": str(args.split),
            "records_written": records_written,
            "start_index": int(args.start_index),
            "model_dtype": str(model_dtype).replace("torch.", ""),
            "shuffle_seed": int(args.shuffle_seed),
        },
    )
    print(json.dumps({"output": str(output_path), "summary": str(summary_path)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
