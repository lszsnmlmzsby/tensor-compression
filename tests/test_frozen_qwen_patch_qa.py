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
    state_ref = f"state_{qa_id}"
    scale = float((torch.tensor(0.0, dtype=torch.float32) + 1.0e-6).item())
    return {
        "qa_id": qa_id,
        "patch_id": state_ref,
        "state_ref": state_ref,
        "sample_index": int(qa_id.rsplit("_", 1)[-1]),
        "time_index": 0,
        "top_left": [0, 0],
        "task_type": task,
        "field": field,
        "query": "Which displayed choice is correct?",
        "question": "This fallback question must not replace query.",
        "choices": labels,
        "answer": answer,
        "oracle": {"value": 123456.0},
        "latent_ref": "must_not_be_read.pt",
        "latent_audit": {
            "format": baseline.PATCH_LATENT_AUDIT_FORMAT,
            "mean": 0.0,
            "std": 0.0,
            "scale": scale,
        },
        "grounding_target": {"type": "point", "row": 1, "col": 2},
    }


def make_latent_contract(checkpoint: Path) -> dict:
    return {
        "format": baseline.PATCH_LATENT_FORMAT,
        "latent_audit_format": baseline.PATCH_LATENT_AUDIT_FORMAT,
        "alignment_checkpoint": str(checkpoint),
        "alignment_checkpoint_sha256": "a" * 64,
        "encoder_input_normalization": {
            "mode": "zscore",
            "scope": "channel",
            "stats_path": None,
            "clip_min": None,
            "clip_max": None,
        },
        "latent_shape": [8, 16, 16],
        "storage_dtype": "float16",
    }


def write_latent(path: Path, record: dict, contract: dict) -> None:
    latent = torch.full((8, 16, 16), 123.0, dtype=torch.float16)
    latent[0].zero_()
    torch.save(
        {
            "format": baseline.PATCH_LATENT_FORMAT,
            "patch_id": record["state_ref"],
            "field": record["field"],
            "sample_index": record["sample_index"],
            "time_index": record["time_index"],
            "top_left": record["top_left"],
            "alignment_checkpoint": contract["alignment_checkpoint"],
            "alignment_checkpoint_sha256": contract["alignment_checkpoint_sha256"],
            "encoder_input_normalization": contract["encoder_input_normalization"],
            "qa_value_space": {
                "mode": "per_patch_zscore",
                "mean": record["latent_audit"]["mean"],
                "std": record["latent_audit"]["std"],
                "scale": record["latent_audit"]["scale"],
            },
            "latent_map": latent,
        },
        path,
    )


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
    def test_prompt_keeps_formal_task_and_is_answer_oracle_invariant(self) -> None:
        record = make_record("qa_0")
        matrix_text = "shape: 1 rows x 1 columns\ncolumn indices: 1\nrow 1: [0]"
        rendered = baseline.render_prompt(record, "task_specific", matrix_text)
        shared_prompt = build_prompt(record, "task_specific")
        matrix_task_prompt = baseline.adapt_task_prompt_to_matrix(shared_prompt)
        self.assertTrue(rendered.endswith(matrix_task_prompt))
        self.assertIn(matrix_text, rendered)
        self.assertNotIn("soft token", rendered.casefold())
        changed = dict(
            record,
            answer="B",
            oracle={"value": -999.0},
            latent_ref="different.pt",
            grounding_target={"type": "different"},
            matched_group={"answer": "B"},
            prompt_data={"answer": "B"},
        )
        self.assertEqual(
            rendered,
            baseline.render_prompt(changed, "task_specific", matrix_text),
        )
        self.assertNotIn("123456", rendered)
        self.assertNotIn("must_not_be_read", rendered)
        self.assertTrue(rendered.endswith("Answer:"))

    def test_matrix_serialization_round_trips_fp16_and_labels_coordinates(self) -> None:
        values = torch.tensor(
            [[0.0, -0.125, 2.0**-20], [1.5, -3.25, 6.5504e4]],
            dtype=torch.float16,
        )
        text = baseline.serialize_standardized_matrix(values, significant_digits=6)
        self.assertIn("shape: 2 rows x 3 columns", text)
        self.assertIn("column indices: 1, 2, 3", text)
        self.assertIn("row 1:", text)
        self.assertIn("row 2:", text)

    def test_fp16_cross_quadrant_extreme_has_tie_aware_labels(self) -> None:
        support = baseline.matrix_extreme_support(torch.zeros((16, 16), dtype=torch.float16))
        self.assertEqual(support["maximum"]["acceptable_labels"], ["A", "B", "C", "D"])
        self.assertEqual(support["maximum"]["tie_scope"], "cross_quadrant_tie")

        payload = baseline.empty_metric_payload()
        record = make_record(
            "qa_0",
            task="extreme_quadrant",
            choices=["A", "B", "C", "D"],
            answer="A",
        )
        record["query"] = "Which quadrant contains the maximum value?"
        contract = baseline.FrozenQwenQADataset.metric_contract_for_record(record, support)
        self.assertEqual(contract["acceptable_answers"], ["A", "B", "C", "D"])
        baseline.update_metric_payload(
            payload,
            record,
            {
                "prediction": "B",
                "probabilities": {"A": 0.2, "B": 0.4, "C": 0.2, "D": 0.2},
                "prompt_tokens": 10,
            },
            0,
            acceptable_answers=contract["acceptable_answers"],
            extreme_tie_scope=contract["extreme_tie_scope"],
        )
        merged = baseline.merge_metric_payloads(
            [baseline.serializable_metric_payload(payload)], expected_total=1
        )
        metrics = baseline.finalize_metrics(merged)
        self.assertEqual(metrics["accuracy"], 0.0)
        self.assertEqual(metrics["tie_aware_accuracy"], 1.0)

    def test_extreme_support_covers_minimum_within_quadrant_and_unique_cells(self) -> None:
        values = torch.zeros((16, 16), dtype=torch.float16)
        values[0, 0] = -2.0
        values[1, 1] = -2.0
        values[15, 15] = 3.0
        support = baseline.matrix_extreme_support(values)
        self.assertEqual(support["minimum"]["acceptable_labels"], ["A"])
        self.assertEqual(support["minimum"]["position_count"], 2)
        self.assertEqual(support["minimum"]["tie_scope"], "within_quadrant_tie")
        self.assertEqual(support["maximum"]["acceptable_labels"], ["D"])
        self.assertEqual(support["maximum"]["position_count"], 1)
        self.assertEqual(support["maximum"]["tie_scope"], "unique_cell")

        values[15, 15] = -2.0
        support = baseline.matrix_extreme_support(values)
        self.assertEqual(support["minimum"]["acceptable_labels"], ["A", "D"])
        self.assertEqual(support["minimum"]["tie_scope"], "cross_quadrant_tie")

    def test_extreme_operation_parser_is_word_bounded_and_fail_closed(self) -> None:
        record = make_record(
            "qa_0",
            task="extreme_quadrant",
            choices=["A", "B", "C", "D"],
        )
        record["query"] = "Which quadrant contains the minimum value?"
        self.assertEqual(baseline.requested_extreme_operation(record), "minimum")
        record["query"] = "Which quadrant contains the maximum value?"
        self.assertEqual(baseline.requested_extreme_operation(record), "maximum")
        for invalid in (
            "Which quadrant contains the value?",
            "Compare the maximum and minimum locations.",
            "A maximumly difficult unrelated phrase.",
        ):
            record["query"] = invalid
            with self.assertRaisesRegex(ValueError, "exactly one"):
                baseline.requested_extreme_operation(record)

    def test_extreme_contract_rejects_source_label_outside_fp16_support(self) -> None:
        values = torch.zeros((16, 16), dtype=torch.float16)
        values[0, 0] = -2.0
        values[15, 15] = -2.0
        support = baseline.matrix_extreme_support(values)
        record = make_record(
            "qa_0",
            task="extreme_quadrant",
            choices=["A", "B", "C", "D"],
            answer="B",
        )
        record["query"] = "Which quadrant contains the minimum value?"
        with self.assertRaisesRegex(ValueError, "does not support"):
            baseline.FrozenQwenQADataset.metric_contract_for_record(record, support)

    def test_dataset_loads_only_preserved_value_channel(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            latent_dir = root / "latents"
            latent_dir.mkdir()
            record = make_record("qa_0")
            qa_path = root / "val.jsonl"
            qa_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            contract = make_latent_contract(root / "alignment_best.pt")
            write_latent(latent_dir / f"{record['state_ref']}.pt", record, contract)
            dataset = baseline.FrozenQwenQADataset(
                qa_path,
                latent_dir=latent_dir,
                latent_contract=contract,
                matrix_significant_digits=6,
                matrix_cache_size=4,
            )
            item = dataset[0]
        self.assertIn("row 16:", item["matrix_text"])
        self.assertNotIn("123", item["matrix_text"])
        self.assertEqual(item["matrix_text"], dataset.matrix_text_for_record(record))

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
                        f"  latent_dir: {root / 'latents'}",
                        "  qa_dir: /wrong/legacy/path",
                        "model:",
                        f"  local_dir: {root / 'qwen'}",
                        "llm_training:",
                        f"  output_root: {root / 'runs'}",
                        "  eval_batch_size: 8",
                        "  prompt_template: task_specific",
                        "qwen_tensor_baseline:",
                        "  max_prompt_tokens: 8192",
                    ]
                ),
                encoding="utf-8",
            )
            args = baseline.parse_args(["--config", str(config), "--batch-size", "5"])
        self.assertEqual(args.qa_dir, str(root / "stage2b"))
        self.assertEqual(args.latent_dir, str(root / "latents"))
        self.assertEqual(args.batch_size, 5)
        self.assertEqual(args.max_prompt_tokens, 8192)
        self.assertEqual(args.splits, "val,test")

    def test_formal_metadata_audit_checks_qa_hash_and_latent_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            latent_dir = root / "latents"
            latent_dir.mkdir()
            split_hashes: dict[str, str] = {}
            for split in ("val", "test"):
                content = json.dumps(make_record("qa_0")) + "\n"
                path = root / f"{split}.jsonl"
                path.write_text(content, encoding="utf-8")
                split_hashes[split] = baseline.sha256_file(path)
            contract = make_latent_contract(root / "alignment_best.pt")
            metadata = {
                "format": baseline.PATCH_MATCHED_QA_FORMAT,
                "matched_group_format": baseline.MATCHED_GROUP_FORMAT,
                "prompt_contract": baseline.PATCH_QA_PROMPT_CONTRACT,
                "natural_language_coordinate_origin": 1,
                "split_mode": "sample",
                "requires_explicit_group_sampler": True,
                "latent_dir": str(latent_dir),
                "latent_format": contract["format"],
                "latent_audit_format": contract["latent_audit_format"],
                "latent_shape": contract["latent_shape"],
                "storage_dtype": contract["storage_dtype"],
                "encoder_input_normalization": contract["encoder_input_normalization"],
                "alignment_checkpoint": contract["alignment_checkpoint"],
                "alignment_checkpoint_sha256": contract["alignment_checkpoint_sha256"],
                "patch_size": 16,
                "stage2b": {
                    "target_provenance": {
                        "train_numeric_and_compare": "preserved_input_channel_0_as_stored_float16"
                    }
                },
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
            audit = baseline.audit_qa_metadata(root, latent_dir, ["val", "test"], True)
        self.assertTrue(audit["formal_contract_passed"])
        self.assertFalse(audit["stage1_checkpoint_opened"])
        self.assertTrue(audit["latent_contract_evaluated"])
        self.assertEqual(audit["latent_contract"]["preserved_value_channel"], 0)
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

    def test_script_loads_tensor_payload_but_has_no_training_or_adapter_path(self) -> None:
        source = (PROJECT_ROOT / "scripts" / "evaluate_frozen_qwen_patch_qa.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("train_tensor_llm_adapter", source)
        self.assertIn("torch.load(", source)
        self.assertNotIn("adapter_state_dict", source)
        self.assertNotIn("optimizer =", source)
        self.assertNotIn(".backward(", source)

    def test_parser_rejects_adapter_checkpoint_arguments(self) -> None:
        with self.assertRaises(SystemExit):
            baseline.parse_args(["--adapter-checkpoint", "forbidden.pt"])


if __name__ == "__main__":
    unittest.main()
