from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for search_path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import scripts.evaluate_frozen_qwen_patch_qa as baseline  # noqa: E402
from tensor_compression.downstream.patch_qa_prompt import build_prompt  # noqa: E402


def make_record(
    qa_id: str,
    *,
    task: str = "point_compare",
    field: str = "Vx",
    choices: list[str] | None = None,
    answer: str = "A",
) -> dict:
    labels = choices or ["A", "B"]
    return {
        "qa_id": qa_id,
        "state_ref": f"state_{qa_id}",
        "sample_index": int(qa_id.rsplit("_", 1)[-1]),
        "task_type": task,
        "field": field,
        "query": "Which displayed choice is correct?",
        "question": "This fallback question must not replace query.",
        "choices": labels,
        "answer": answer,
        "oracle": {"value": 123456.0},
        "latent_ref": "must_not_be_read.pt",
        "grounding_target": {"type": "point", "row": 1, "col": 2},
    }


class FakeTokenizer:
    pad_token_id = 0
    pad_token = "<pad>"
    eos_token = "<eos>"
    unk_token = "<unk>"

    def __init__(self, sequences: dict[str, list[int]], padding_side: str = "right") -> None:
        self.sequences = sequences
        self.padding_side = padding_side

    def __call__(
        self,
        value,
        *,
        padding: bool = False,
        return_tensors: str | None = None,
        add_special_tokens: bool = True,
        **_kwargs,
    ):
        if isinstance(value, str):
            if value == " A":
                return {"input_ids": [3]}
            if value == " B":
                return {"input_ids": [4]}
            return {"input_ids": list(self.sequences[value])}
        rows = [list(self.sequences[item]) for item in value]
        width = max(len(row) for row in rows)
        padded: list[list[int]] = []
        masks: list[list[int]] = []
        for row in rows:
            pad = [self.pad_token_id] * (width - len(row))
            valid = [1] * len(row)
            if self.padding_side == "left":
                padded.append(pad + row)
                masks.append([0] * len(pad) + valid)
            else:
                padded.append(row + pad)
                masks.append(valid + [0] * len(pad))
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
            }
        return {"input_ids": padded, "attention_mask": masks}


class FakeDecoder(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids, **_kwargs):
        hidden = F.one_hot(input_ids, num_classes=self.vocab_size).float()
        return SimpleNamespace(last_hidden_state=hidden)


class FakeQwen(nn.Module):
    base_model_prefix = "model"

    def __init__(self, vocab_size: int = 12) -> None:
        super().__init__()
        self.model = FakeDecoder(vocab_size)
        self.lm_head = nn.Linear(vocab_size, vocab_size, bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(torch.eye(vocab_size))
        self.config = SimpleNamespace(model_type="qwen2", use_cache=False)
        self.is_gradient_checkpointing = False

    def get_output_embeddings(self):
        return self.lm_head


class FrozenQwenPatchQATests(unittest.TestCase):
    def test_prompt_is_shared_and_answer_oracle_invariant(self) -> None:
        record = make_record("qa_0")
        rendered = baseline.render_prompt(record, "task_specific")
        self.assertEqual(rendered, build_prompt(record, "task_specific"))
        changed = dict(record, answer="B", oracle={"value": -999.0})
        self.assertEqual(rendered, baseline.render_prompt(changed, "task_specific"))
        self.assertNotIn("123456", rendered)
        self.assertNotIn("must_not_be_read", rendered)
        self.assertTrue(rendered.endswith("Answer:"))

    def test_record_loader_removes_oracle_and_prefix_cap_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "val.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for index in range(3):
                    handle.write(json.dumps(make_record(f"qa_{index}")) + "\n")
            first, first_oracles = baseline.load_qa_records(path, max_records=2)
            second, second_oracles = baseline.load_qa_records(path, max_records=2)
        self.assertEqual(first, second)
        self.assertEqual([item["qa_id"] for item in first], ["qa_0", "qa_1"])
        self.assertEqual(first_oracles, 2)
        self.assertEqual(second_oracles, 2)
        self.assertTrue(all("oracle" not in item for item in first))

    def test_config_prefers_stage2b_qa_and_batch_is_cli_tunable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "pipeline.yaml"
            config.write_text(
                "\n".join(
                    [
                        "patch_qa:",
                        f"  stage2b_qa_dir: {root / 'stage2b'}",
                        "  qa_dir: /wrong/legacy/path",
                        "model:",
                        f"  local_dir: {root / 'qwen'}",
                        "llm_training:",
                        f"  output_root: {root / 'runs'}",
                        "  eval_batch_size: 8",
                        "  prompt_template: task_specific",
                    ]
                ),
                encoding="utf-8",
            )
            args = baseline.parse_args(["--config", str(config), "--batch-size", "5"])
        self.assertEqual(args.qa_dir, str(root / "stage2b"))
        self.assertEqual(args.batch_size, 5)
        self.assertEqual(args.splits, "val,test")

    def test_formal_metadata_audit_checks_qa_hash_without_latents(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            split_hashes: dict[str, str] = {}
            for split in ("val", "test"):
                content = json.dumps(make_record("qa_0")) + "\n"
                path = root / f"{split}.jsonl"
                path.write_text(content, encoding="utf-8")
                split_hashes[split] = baseline.sha256_file(path)
            metadata = {
                "format": baseline.PATCH_MATCHED_QA_FORMAT,
                "matched_group_format": baseline.MATCHED_GROUP_FORMAT,
                "prompt_contract": baseline.PATCH_QA_PROMPT_CONTRACT,
                "natural_language_coordinate_origin": 1,
                "split_mode": "sample",
                "requires_explicit_group_sampler": True,
                "alignment_checkpoint": str(root / "does_not_exist.pt"),
                "latent_dir": str(root / "does_not_exist_latents"),
                "fields": ["Vx"],
                "output_split_sha256": split_hashes,
                "summary": {
                    "splits": {
                        "val": {"qa_records": 1},
                        "test": {"qa_records": 1},
                    }
                },
            }
            (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            audit = baseline.audit_qa_metadata(root, ["val", "test"], True)
        self.assertTrue(audit["formal_contract_passed"])
        self.assertFalse(audit["stage1_checkpoint_opened"])
        self.assertFalse(audit["latent_contract_evaluated"])
        self.assertTrue(audit["split_files"]["val"]["matches_declared_sha256"])

    def test_exact_distributed_sampler_has_no_padding_or_repeats(self) -> None:
        dataset = list(range(11))
        shards = [
            list(baseline.ExactDistributedEvalSampler(dataset, rank, 4))
            for rank in range(4)
        ]
        flattened = [index for shard in shards for index in shard]
        self.assertEqual(sorted(flattened), list(range(11)))
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual([len(shard) for shard in shards], [3, 3, 3, 2])

    def test_last_nonpadding_indices_support_both_padding_sides(self) -> None:
        right = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
        left = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])
        self.assertEqual(baseline.last_nonpadding_indices(right).tolist(), [1, 3])
        self.assertEqual(baseline.last_nonpadding_indices(left).tolist(), [3, 3])

    def test_restricted_scoring_uses_each_prompt_boundary(self) -> None:
        model = FakeQwen().eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        choices = [["A", "B"], ["A", "B"]]
        label_ids = {"A": 3, "B": 4}
        for padding_side in ("right", "left"):
            tokenizer = FakeTokenizer(
                {"short": [9, 3], "long": [8, 7, 6, 4]},
                padding_side=padding_side,
            )
            scored = baseline.score_prompt_batch(
                model,
                tokenizer,
                ["short", "long"],
                choices,
                label_ids,
                torch.device("cpu"),
                max_prompt_tokens=8,
            )
            self.assertEqual([item["prediction"] for item in scored], ["A", "B"])
            self.assertEqual([item["prompt_tokens"] for item in scored], [2, 4])

    def test_two_and_four_choice_metrics_match_formal_grouping(self) -> None:
        rank_zero = baseline.empty_metric_payload()
        rank_one = baseline.empty_metric_payload()
        first = make_record("qa_0", task="point_compare", choices=["A", "B"], answer="A")
        second = make_record(
            "qa_1",
            task="extreme_quadrant",
            field="pressure",
            choices=["A", "B", "C", "D"],
            answer="D",
        )
        third = make_record("qa_2", task="point_compare", choices=["A", "B"], answer="B")
        baseline.update_metric_payload(
            rank_zero,
            first,
            {"prediction": "A", "probabilities": {"A": 0.8, "B": 0.2}, "prompt_tokens": 10},
            0,
        )
        baseline.update_metric_payload(
            rank_one,
            second,
            {
                "prediction": "A",
                "probabilities": {"A": 0.4, "B": 0.2, "C": 0.2, "D": 0.2},
                "prompt_tokens": 12,
            },
            1,
        )
        baseline.update_metric_payload(
            rank_zero,
            third,
            {"prediction": "B", "probabilities": {"A": 0.3, "B": 0.7}, "prompt_tokens": 11},
            2,
        )
        merged = baseline.merge_metric_payloads(
            [
                baseline.serializable_metric_payload(rank_zero),
                baseline.serializable_metric_payload(rank_one),
            ],
            expected_total=3,
        )
        metrics = baseline.finalize_metrics(merged)
        self.assertAlmostEqual(metrics["accuracy"], 2 / 3)
        self.assertEqual(metrics["by_task"]["point_compare"]["correct"], 2)
        self.assertEqual(metrics["by_task"]["extreme_quadrant"]["correct"], 0)
        self.assertEqual(metrics["by_field"]["pressure"]["total"], 1)
        self.assertIn("point_compare/Vx", metrics["by_task_field"])
        self.assertEqual(metrics["candidate_count_distribution"], {"2": 2, "4": 1})
        self.assertEqual(metrics["distributed_shard_audit"]["records_by_rank"], [2, 1])

    def test_merge_rejects_duplicate_or_missing_shards(self) -> None:
        left = baseline.empty_metric_payload()
        right = baseline.empty_metric_payload()
        record = make_record("qa_0")
        scored = {"prediction": "A", "probabilities": {"A": 0.6, "B": 0.4}, "prompt_tokens": 4}
        baseline.update_metric_payload(left, record, scored, 0)
        baseline.update_metric_payload(right, record, scored, 0)
        with self.assertRaisesRegex(RuntimeError, "shard audit failed"):
            baseline.merge_metric_payloads(
                [
                    baseline.serializable_metric_payload(left),
                    baseline.serializable_metric_payload(right),
                ],
                expected_total=2,
            )

    def test_frozen_eval_audit_rejects_trainable_qwen(self) -> None:
        model = FakeQwen().eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        audit = baseline.audit_frozen_qwen(model, torch.float32)
        self.assertEqual(audit["trainable_parameter_count"], 0)
        next(model.parameters()).requires_grad_(True)
        with self.assertRaisesRegex(RuntimeError, "Frozen-Qwen contract failed"):
            baseline.audit_frozen_qwen(model, torch.float32)

    def test_script_has_no_training_or_checkpoint_load_path(self) -> None:
        source = (PROJECT_ROOT / "scripts" / "evaluate_frozen_qwen_patch_qa.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("train_tensor_llm_adapter", source)
        self.assertNotIn("torch.load(", source)
        self.assertNotIn("optimizer =", source)
        self.assertNotIn(".backward(", source)

    def test_parser_rejects_adapter_checkpoint_arguments(self) -> None:
        with self.assertRaises(SystemExit):
            baseline.parse_args(["--adapter-checkpoint", "forbidden.pt"])


if __name__ == "__main__":
    unittest.main()
