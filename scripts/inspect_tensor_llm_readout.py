from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tensor_compression.downstream.pdebench import resolve_device  # noqa: E402
from tensor_compression.utils import dump_json  # noqa: E402

from scripts.diagnose_tensor_llm_adapter import (  # noqa: E402
    cosine_similarity,
    l2_distance,
    make_adapter_from_checkpoint,
    state_fields,
    tensor_summary,
)
from scripts.train_tensor_llm_adapter import (  # noqa: E402
    TensorReadoutQADataset,
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
            "Inspect what a trained tensor-LLM adapter makes the frozen LLM prefer at the answer position. "
            "The output is JSONL with per-choice NLL/prob/rank under correct, zero_latent, shuffled, and no_latent modes."
        )
    )
    parser.add_argument("--config", type=str, default="configs/tensor_llm_adapter_pipeline.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--records", type=int, default=64)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--modes",
        type=str,
        default="correct,zero_latent,shuffled,no_latent",
        help="Comma-separated modes to inspect.",
    )
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
    if args.records <= 0:
        raise ValueError("--records must be positive.")
    return apply_config_defaults(args)


def parse_modes(raw: str) -> list[str]:
    modes = [part.strip() for part in str(raw).split(",") if part.strip()]
    supported = {"correct", "zero_latent", "shuffled", "no_latent", "random"}
    unknown = [mode for mode in modes if mode not in supported]
    if unknown:
        raise ValueError(f"Unsupported inspect mode(s): {unknown}. Supported: {sorted(supported)}")
    return modes


def default_output_path(checkpoint: str | Path, split: str) -> Path:
    path = Path(checkpoint)
    return path.parent / f"{path.stem}_readout_inspection_{split}.jsonl"


def choice_scores_for_mode(
    llm,
    adapter,
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
    labels = [str(choice) for choice in choices]
    scores = score_candidate_batch(
        llm=llm,
        adapter=adapter,
        tokenizer=tokenizer,
        records=[record for _ in labels],
        answers=labels,
        latent_map=latent_map.unsqueeze(0).repeat(len(labels), 1, 1, 1),
        device=device,
        max_prompt_tokens=int(args.max_prompt_tokens),
        max_target_tokens=int(args.max_target_tokens),
        append_eos=bool(args.append_eos),
        prompt_template=str(args.prompt_template),
        soft_prompt_mode=mode,
        choice_score=str(args.choice_score),
        local_context_layer=int(args.local_context_layer),
    )
    min_nll = min(scores)
    weights = [math.exp(-(score - min_nll)) for score in scores]
    total_weight = sum(weights)
    probs = [weight / total_weight for weight in weights]
    order = sorted(range(len(labels)), key=lambda index: scores[index])
    ranks = {labels[index]: rank + 1 for rank, index in enumerate(order)}
    answer = str(record["answer"])
    prediction = labels[order[0]]
    return {
        "prediction": prediction,
        "is_correct": prediction == answer,
        "answer_rank": ranks.get(answer),
        "answer_nll": float(scores[labels.index(answer)]) if answer in labels else None,
        "answer_prob": float(probs[labels.index(answer)]) if answer in labels else None,
        "choices": [
            {
                "label": label,
                "nll": float(score),
                "prob": float(prob),
                "rank": int(ranks[label]),
                "is_answer": label == answer,
            }
            for label, score, prob in zip(labels, scores, probs)
        ],
    }


@torch.no_grad()
def soft_prompt_for_mode(
    llm,
    adapter,
    tokenizer,
    record: Mapping[str, Any],
    latent_map: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    mode: str,
) -> torch.Tensor:
    input_ids, text_attention_mask, text_labels = build_text_tensors(
        records=[record],
        answers=[str(record["answer"])],
        tokenizer=tokenizer,
        max_prompt_tokens=int(args.max_prompt_tokens),
        max_target_tokens=int(args.max_target_tokens),
        append_eos=bool(args.append_eos),
        prompt_template=str(args.prompt_template),
    )
    input_ids = input_ids.to(device)
    text_attention_mask = text_attention_mask.to(device)
    text_labels = text_labels.to(device)
    text_embeds = llm.get_input_embeddings()(input_ids)
    prompt_mask = text_labels.eq(-100) & text_attention_mask.bool()
    batched_latent = latent_map.unsqueeze(0).to(device)
    soft_embeds = contextual_adapter_soft_embeds(
        llm=llm,
        adapter=adapter,
        tokenizer=tokenizer,
        records=[record],
        latent_map=batched_latent,
        device=device,
        max_prompt_tokens=int(args.max_prompt_tokens),
        layer_index=int(args.local_context_layer),
        mode=mode,
    )
    if soft_embeds is not None:
        return soft_embeds
    return adapter_soft_embeds(
        adapter,
        batched_latent,
        text_embeds,
        question_embeds=text_embeds,
        question_mask=prompt_mask,
        records=[record],
        mode=mode,
    )


def mode_latent(dataset: TensorReadoutQADataset, index: int, correct_latent: torch.Tensor, mode: str) -> torch.Tensor:
    if mode in {"correct", "no_latent"}:
        return correct_latent
    if mode == "zero_latent":
        return torch.zeros_like(correct_latent)
    if mode == "random":
        return torch.randn_like(correct_latent)
    if mode == "shuffled":
        return dataset.load_latent_for_record(dataset.shuffled_record_for_index(index))
    raise ValueError(f"Unsupported mode: {mode}")


def summarize_deltas(per_mode: Mapping[str, Any], answer: str) -> dict[str, Any]:
    correct = per_mode.get("correct")
    if not isinstance(correct, Mapping):
        return {}
    correct_answer_prob = correct.get("answer_prob")
    correct_answer_nll = correct.get("answer_nll")
    deltas: dict[str, Any] = {}
    for mode, payload in per_mode.items():
        if mode == "correct" or not isinstance(payload, Mapping):
            continue
        deltas[mode] = {
            "answer_prob_correct_minus_mode": (
                float(correct_answer_prob) - float(payload["answer_prob"])
                if correct_answer_prob is not None and payload.get("answer_prob") is not None
                else None
            ),
            "answer_nll_mode_minus_correct": (
                float(payload["answer_nll"]) - float(correct_answer_nll)
                if correct_answer_nll is not None and payload.get("answer_nll") is not None
                else None
            ),
            "prediction_changed": payload.get("prediction") != correct.get("prediction"),
            "mode_prediction": payload.get("prediction"),
            "correct_prediction": correct.get("prediction"),
            "answer": answer,
        }
    return deltas


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

    modes = parse_modes(args.modes)
    output_path = Path(args.output) if args.output else default_output_path(args.checkpoint, args.split)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected = range(max(0, int(args.start_index)), min(len(dataset), int(args.start_index) + int(args.records)))
    rows_written = 0
    aggregate = {
        mode: {"correct": 0, "total": 0, "answer_rank_sum": 0.0, "answer_prob_sum": 0.0}
        for mode in modes
    }
    with output_path.open("w", encoding="utf-8") as handle:
        for index in selected:
            item = dataset[index]
            record = item["record"]
            correct_latent = item["latent_map"]
            per_mode: dict[str, Any] = {}
            soft_prompts: dict[str, torch.Tensor] = {}
            for mode in modes:
                latent = mode_latent(dataset, index, correct_latent, mode)
                per_mode[mode] = choice_scores_for_mode(
                    llm=llm,
                    adapter=adapter,
                    tokenizer=tokenizer,
                    record=record,
                    latent_map=latent,
                    args=args,
                    device=device,
                    mode=mode,
                )
                soft_prompts[mode] = soft_prompt_for_mode(
                    llm=llm,
                    adapter=adapter,
                    tokenizer=tokenizer,
                    record=record,
                    latent_map=latent,
                    args=args,
                    device=device,
                    mode=mode,
                )
                aggregate[mode]["correct"] += int(bool(per_mode[mode]["is_correct"]))
                aggregate[mode]["total"] += 1
                if per_mode[mode]["answer_rank"] is not None:
                    aggregate[mode]["answer_rank_sum"] += float(per_mode[mode]["answer_rank"])
                if per_mode[mode]["answer_prob"] is not None:
                    aggregate[mode]["answer_prob_sum"] += float(per_mode[mode]["answer_prob"])

            soft_summary: dict[str, Any] = {mode: tensor_summary(tensor) for mode, tensor in soft_prompts.items()}
            if "correct" in soft_prompts:
                for mode, tensor in soft_prompts.items():
                    if mode == "correct":
                        continue
                    soft_summary[f"correct_vs_{mode}"] = {
                        "l2_distance": l2_distance(soft_prompts["correct"], tensor),
                        "cosine_similarity": cosine_similarity(soft_prompts["correct"], tensor),
                    }
            payload = {
                "index": int(index),
                "record": state_fields(record),
                "shuffled_record": state_fields(dataset.shuffled_record_for_index(index))
                if "shuffled" in modes
                else None,
                "latent": {
                    "correct": tensor_summary(correct_latent),
                    "shuffled": tensor_summary(mode_latent(dataset, index, correct_latent, "shuffled"))
                    if "shuffled" in modes
                    else None,
                },
                "soft_prompt": soft_summary,
                "modes": per_mode,
                "deltas_from_correct": summarize_deltas(per_mode, str(record["answer"])),
            }
            handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
            rows_written += 1

    summary = {
        "checkpoint": str(args.checkpoint),
        "output": str(output_path),
        "split": str(args.split),
        "records_written": rows_written,
        "modes": modes,
        "model_dtype": str(model_dtype).replace("torch.", ""),
        "aggregate": {
            mode: {
                "accuracy": values["correct"] / max(1, values["total"]),
                "mean_answer_rank": values["answer_rank_sum"] / max(1, values["total"]),
                "mean_answer_prob": values["answer_prob_sum"] / max(1, values["total"]),
                "total": values["total"],
            }
            for mode, values in aggregate.items()
        },
    }
    summary_path = output_path.with_suffix(".summary.json")
    dump_json(summary_path, summary)
    print(json.dumps({"output": str(output_path), "summary": str(summary_path)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
