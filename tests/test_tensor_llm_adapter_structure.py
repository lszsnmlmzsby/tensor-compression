from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn.functional as F
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_tensor_llm_adapter as adapter_training  # noqa: E402
from train_tensor_llm_adapter import (  # noqa: E402
    ExactDistributedEvalSampler,
    HybridGlobalLocalAdapter,
    ResidualQuestionConditionedAdapter,
    StateTaskGroupedBatchSampler,
    TensorReadoutQADataset,
    _sequence_choice_ce_loss,
    adapter_from_checkpoint,
    audit_qa_datasets,
    build_local_conditioning_prompt,
    choice_ce_loss,
    parse_generated_choice,
    read_host_memory_snapshot,
    same_state_question_swap_indices,
    selective_answer_statistics,
    set_frozen_llm_execution_mode,
    single_token_choice_ids,
    structured_query_features_for_record,
    task_specific_instruction,
)
from tensor_compression.models.compressors.conv_token_autoencoder_2d import (  # noqa: E402
    ConvTokenAutoencoder2D,
)
from train_tensor_patch_text_alignment import (  # noqa: E402
    TensorPatchAlignmentAdapter,
    alignment_adapter_path_metrics,
    alignment_adapter_parameter_metrics,
    sinusoidal_2d_position_encoding,
)


def _record(state: str, task: str, field: str, question: str) -> dict[str, str]:
    return {
        "state_ref": state,
        "task_type": task,
        "field": field,
        "query": question,
        "question": question,
    }


class TestDistributedSampling(unittest.TestCase):
    def test_record_limit_stops_jsonl_loading_early(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps({"state_ref": "first"}) + "\n")
                handle.write("this line is intentionally invalid JSON\n")

            records = TensorReadoutQADataset._load_records(path, max_records=1)

            self.assertEqual(records, [{"state_ref": "first"}])

    def test_worker_cache_capacity_divides_the_per_rank_budget(self) -> None:
        dataset = TensorReadoutQADataset.__new__(TensorReadoutQADataset)
        dataset.latent_cache_size = 10
        with mock.patch.object(
            adapter_training,
            "get_worker_info",
            return_value=SimpleNamespace(num_workers=4),
        ):
            self.assertEqual(dataset.effective_latent_cache_size(), 3)

    def test_linux_host_memory_snapshot_uses_mem_available_and_process_rss(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "self").mkdir()
            (root / "meminfo").write_text(
                "MemTotal:       134217728 kB\nMemFree:         1048576 kB\n"
                "MemAvailable:   67108864 kB\n",
                encoding="utf-8",
            )
            (root / "self" / "status").write_text(
                "Name:\tpython\nVmRSS:\t2097152 kB\n",
                encoding="utf-8",
            )

            snapshot = read_host_memory_snapshot(root)

            self.assertEqual(snapshot["total_gib"], 128.0)
            self.assertEqual(snapshot["available_gib"], 64.0)
            self.assertEqual(snapshot["process_rss_gib"], 2.0)

    def test_truncated_smoke_audit_does_not_require_all_choice_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            latent_dir = Path(directory)
            torch.save({"latent_map": torch.zeros(1, 2, 2)}, latent_dir / "state.pt")

            def dataset(records):
                value = SimpleNamespace(records=records, latent_dir=latent_dir)
                value.latent_path_for_record = lambda record: latent_dir / f"{record['state_ref']}.pt"
                return value

            train_records = [
                {
                    "qa_id": "state_task_0",
                    "state_ref": "state",
                    "sample_index": 0,
                    "task_type": "raw_point_value_with_stats",
                    "field": "Vx",
                    "choices": ["A", "B", "C", "D"],
                    "answer": "A",
                    "question": "Options: A: 0; B: 1; C: 2; D: 3.",
                }
            ]
            val_records = [dict(train_records[0], qa_id="other_task_0", state_ref="other", sample_index=1)]
            torch.save({"latent_map": torch.zeros(1, 2, 2)}, latent_dir / "other.pt")
            with self.assertRaisesRegex(ValueError, "answer labels absent"):
                audit_qa_datasets(
                    {"train": dataset(train_records), "val": dataset(val_records)},
                    require_disjoint_splits=True,
                    require_complete_split_coverage=True,
                )

            summary = audit_qa_datasets(
                {"train": dataset(train_records), "val": dataset(val_records)},
                require_disjoint_splits=True,
                require_complete_split_coverage=False,
            )
            self.assertFalse(summary["_audit_scope"]["complete_split_coverage_checked"])
            self.assertEqual(summary["val"]["missing_answer_labels"]["raw_point_value_with_stats"], ["B", "C", "D"])

    def test_shuffled_indices_preserve_field_task_and_change_sample(self) -> None:
        records = [
            {
                **_record(f"state_{sample}_{variant}", task, field, f"question_{variant}"),
                "sample_index": sample,
            }
            for field in ("Vx", "Vy")
            for task in ("point", "region")
            for sample in range(3)
            for variant in range(2)
        ]
        dataset = TensorReadoutQADataset.__new__(TensorReadoutQADataset)
        dataset.records = records

        first = dataset._build_random_different_indices(seed=17)
        second = dataset._build_random_different_indices(seed=17)

        self.assertEqual(first, second)
        for source, candidate_index in zip(records, first):
            candidate = records[candidate_index]
            self.assertEqual(candidate["field"], source["field"])
            self.assertEqual(candidate["task_type"], source["task_type"])
            self.assertNotEqual(candidate["sample_index"], source["sample_index"])

    def test_latent_lru_cache_avoids_repeated_torch_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [
                {
                    **_record(f"state_{sample}", "point", "Vx", f"question_{sample}"),
                    "qa_id": f"qa_{sample}",
                    "sample_index": sample,
                }
                for sample in range(2)
            ]
            jsonl = root / "train.jsonl"
            with jsonl.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record) + "\n")
            for sample in range(2):
                torch.save({"latent_map": torch.full((2, 2, 2), float(sample))}, root / f"state_{sample}.pt")
            dataset = TensorReadoutQADataset(
                jsonl_path=jsonl,
                latent_dir=root,
                latent_cache_size=1,
            )

            with mock.patch.object(torch, "load", wraps=torch.load) as wrapped_load:
                first = dataset.load_latent_for_record(records[0])
                repeated = dataset.load_latent_for_record(records[0])
                dataset.load_latent_for_record(records[1])
                reloaded = dataset.load_latent_for_record(records[0])

            torch.testing.assert_close(first, repeated)
            torch.testing.assert_close(first, reloaded)
            self.assertEqual(wrapped_load.call_count, 3)

    def test_shuffled_indices_fall_back_to_different_state_within_field_task(self) -> None:
        records = [
            {
                **_record(f"state_{state}", "point", "Vx", f"question_{state}"),
                "sample_index": 0,
            }
            for state in range(4)
        ]
        dataset = TensorReadoutQADataset.__new__(TensorReadoutQADataset)
        dataset.records = records

        selected = dataset._build_random_different_indices(seed=5)

        for source, candidate_index in zip(records, selected):
            candidate = records[candidate_index]
            self.assertEqual(candidate["field"], source["field"])
            self.assertEqual(candidate["task_type"], source["task_type"])
            self.assertNotEqual(candidate["state_ref"], source["state_ref"])

    def test_grouped_sampler_preserves_groups_and_equalizes_rank_steps(self) -> None:
        records = [
            _record(f"state_{state}", "point", "Vx", f"question_{variant}")
            for state in range(7)
            for variant in range(3)
        ]
        dataset = SimpleNamespace(records=records)
        rank_batches = [
            list(
                StateTaskGroupedBatchSampler(
                    dataset=dataset,
                    batch_size=3,
                    questions_per_group=3,
                    seed=17,
                    rank=rank,
                    num_replicas=4,
                )
            )
            for rank in range(4)
        ]

        self.assertEqual([len(batches) for batches in rank_batches], [2, 2, 2, 2])
        flattened = [index for batches in rank_batches for batch in batches for index in batch]
        self.assertEqual(set(flattened), set(range(len(records))))
        self.assertEqual(len(flattened) - len(records), 3)
        for batches in rank_batches:
            for batch in batches:
                keys = {
                    (records[index]["state_ref"], records[index]["task_type"])
                    for index in batch
                }
                self.assertEqual(len(keys), 1)

    def test_exact_eval_sampler_never_pads_or_repeats(self) -> None:
        dataset = list(range(10))
        shards = [
            list(ExactDistributedEvalSampler(dataset, rank=rank, num_replicas=3))
            for rank in range(3)
        ]

        flattened = [index for shard in shards for index in shard]
        self.assertEqual(sorted(flattened), list(range(10)))
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual([len(shard) for shard in shards], [4, 3, 3])


class TestQuestionConditionedAdapter(unittest.TestCase):
    def test_selective_answer_statistics_aligns_first_target_and_retains_gradient(self) -> None:
        class IdentityDecoder(nn.Module):
            def forward(self, inputs_embeds, **_kwargs):
                return SimpleNamespace(last_hidden_state=inputs_embeds)

        class FakeCausalLM(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.decoder = IdentityDecoder()
                self.output = nn.Linear(3, 5, bias=False)

            def get_decoder(self):
                return self.decoder

            def get_output_embeddings(self):
                return self.output

        torch.manual_seed(9)
        llm = FakeCausalLM()
        inputs = torch.randn(2, 5, 3, requires_grad=True)
        labels = torch.tensor(
            [
                [-100, -100, 1, 4, -100],
                [-100, 2, 4, -100, -100],
            ]
        )
        attention = torch.ones(2, 5, dtype=torch.long)

        sequence_nll, counts, first_logits = selective_answer_statistics(
            llm=llm,
            inputs_embeds=inputs,
            attention_mask=attention,
            labels=labels,
            return_first_logits=True,
        )

        self.assertIsNotNone(first_logits)
        assert first_logits is not None
        expected_first = torch.stack([llm.output(inputs[0, 1]), llm.output(inputs[1, 0])])
        torch.testing.assert_close(first_logits, expected_first)
        self.assertEqual(counts.tolist(), [2, 2])
        expected_nll = torch.stack(
            [
                F.cross_entropy(llm.output(inputs[0, 1:3]), torch.tensor([1, 4]), reduction="sum"),
                F.cross_entropy(llm.output(inputs[1, 0:2]), torch.tensor([2, 4]), reduction="sum"),
            ]
        )
        torch.testing.assert_close(sequence_nll, expected_nll)

        (sequence_nll.sum() + first_logits.sum() * 0.01).backward()
        self.assertIsNotNone(inputs.grad)
        self.assertGreater(float(inputs.grad.abs().sum().item()), 0.0)

    def test_single_token_choice_ids_use_space_prefixed_answer_tokens(self) -> None:
        class FakeTokenizer:
            ids = {" A": 11, " B": 12, " C": 13, " D": 14}

            def __call__(self, text, **_kwargs):
                return {"input_ids": [self.ids[text]]}

        records = [
            {"answer": "C", "choices": ["A", "B", "C", "D"]},
            {"answer": "B", "choices": ["A", "B"]},
        ]

        result = single_token_choice_ids(records, FakeTokenizer())

        self.assertEqual(result, ([[11, 12, 13, 14], [11, 12]], [2, 1]))

    def test_frozen_llm_mode_keeps_checkpoint_training_deterministic(self) -> None:
        model = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.5))
        model.is_gradient_checkpointing = True

        set_frozen_llm_execution_mode(model, checkpoint_training=True)

        self.assertTrue(model.training)
        self.assertFalse(model[1].training)
        set_frozen_llm_execution_mode(model, checkpoint_training=False)
        self.assertFalse(model.training)

    def test_choice_ce_candidate_chunking_preserves_scores_and_order(self) -> None:
        records = [
            {"answer": "B", "choices": ["A", "B", "C", "D"]},
            {"answer": "D", "choices": ["A", "B", "C", "D"]},
        ]
        latent = torch.randn(2, 3, 2, 2)
        args = SimpleNamespace(
            train_choice_batch_size=3,
            max_prompt_tokens=32,
            max_target_tokens=4,
            append_eos=True,
            prompt_template="task_specific",
            local_context_layer=2,
            choice_score="mean",
        )
        observed_chunk_sizes: list[int] = []

        def fake_answer_nll(**kwargs):
            answers = list(kwargs["answers"])
            observed_chunk_sizes.append(len(answers))
            values = torch.tensor(
                [float(ord(answer) - ord("A") + 1) for answer in answers],
                requires_grad=True,
            )
            counts = torch.ones(len(answers), dtype=torch.long)
            return values, counts

        with (
            mock.patch(
                "train_tensor_llm_adapter.contextual_adapter_soft_embeds",
                return_value=torch.zeros(2, 2, 4),
            ),
            mock.patch("train_tensor_llm_adapter.forward_answer_nll", side_effect=fake_answer_nll),
        ):
            chunked = _sequence_choice_ce_loss(
                llm=object(),
                adapter=object(),
                tokenizer=object(),
                records=records,
                latent_map=latent,
                device=torch.device("cpu"),
                args=args,
            )

        self.assertEqual(observed_chunk_sizes, [3, 3, 2])
        candidate_scores = torch.tensor([-1.0, -2.0, -3.0, -4.0])
        expected_positive = torch.stack(
            [
                F.cross_entropy(candidate_scores.unsqueeze(0), torch.tensor([1])),
                F.cross_entropy(candidate_scores.unsqueeze(0), torch.tensor([3])),
            ]
        )
        torch.testing.assert_close(chunked[2].detach(), expected_positive)
        self.assertEqual(chunked[4]["choice_accuracy"], 0.0)

    def test_single_token_grounding_nll_excludes_eos_and_full_vocabulary(self) -> None:
        class FakeTokenizer:
            ids = {" A": 1, " B": 2, " C": 3, " D": 4}

            def __call__(self, text, **_kwargs):
                return {"input_ids": [self.ids[text]]}

        class FakeLLM(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embeddings = nn.Embedding(8, 4)

            def get_input_embeddings(self):
                return self.embeddings

        records = [
            {"answer": "C", "choices": ["A", "B", "C", "D"]},
            {"answer": "B", "choices": ["A", "B"]},
        ]
        args = SimpleNamespace(
            max_prompt_tokens=8,
            max_target_tokens=2,
            append_eos=True,
            prompt_template="task_specific",
            local_context_layer=2,
            choice_score="mean",
        )
        first_logits = torch.zeros(2, 8, requires_grad=True)
        with torch.no_grad():
            first_logits[0, 1:5] = torch.tensor([0.0, 1.0, 3.0, -1.0])
            first_logits[1, 1:3] = torch.tensor([-2.0, 2.0])
        input_ids = torch.zeros(2, 3, dtype=torch.long)
        attention = torch.ones_like(input_ids)
        labels = torch.tensor([[-100, -100, 1], [-100, -100, 2]])

        with (
            mock.patch(
                "train_tensor_llm_adapter.build_text_tensors",
                return_value=(input_ids, attention, labels),
            ),
            mock.patch(
                "train_tensor_llm_adapter.contextual_adapter_soft_embeds",
                return_value=torch.zeros(2, 2, 4),
            ),
            mock.patch(
                "train_tensor_llm_adapter.selective_answer_statistics",
                return_value=(torch.tensor([100.0, 200.0]), torch.tensor([2, 2]), first_logits),
            ),
        ):
            loss, _answer_ce, per_record_nll, _soft, _metrics = (
                adapter_training.single_token_choice_ce_loss(
                    llm=FakeLLM(),
                    adapter=object(),
                    tokenizer=FakeTokenizer(),
                    records=records,
                    latent_map=torch.zeros(2, 1, 1, 1),
                    device=torch.device("cpu"),
                    args=args,
                )
            )

        expected = torch.stack(
            [
                F.cross_entropy(first_logits[0, 1:5].unsqueeze(0), torch.tensor([2])),
                F.cross_entropy(first_logits[1, 1:3].unsqueeze(0), torch.tensor([1])),
            ]
        )
        torch.testing.assert_close(per_record_nll, expected)
        torch.testing.assert_close(loss, expected.mean())
        self.assertNotEqual(per_record_nll.tolist(), [50.0, 100.0])

    def test_spatial_position_encoding_is_deterministic_finite_and_row_major(self) -> None:
        first = sinusoidal_2d_position_encoding(3, 4, 16)
        second = sinusoidal_2d_position_encoding(3, 4, 16)

        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
        self.assertEqual(tuple(first.shape), (1, 12, 16))
        self.assertTrue(torch.isfinite(first).all())
        self.assertFalse(torch.equal(first[:, 0], first[:, 1]))
        self.assertFalse(torch.equal(first[:, 0], first[:, 4]))

    def test_spatial_adapter_has_one_row_major_token_per_latent_position(self) -> None:
        torch.manual_seed(11)
        adapter = TensorPatchAlignmentAdapter(
            latent_channels=3,
            latent_grid=(2, 3),
            adapter_dim=16,
            projection_dim=24,
            dropout=0.0,
            adapter_type="spatial_transformer",
            query_tokens=6,
            adapter_layers=1,
            adapter_heads=4,
            soft_prompt_scale=0.0,
        ).eval()
        latent = torch.zeros(1, 3, 2, 3)
        changed = latent.clone()
        changed[0, :, 1, 1] = torch.tensor([1.0, -2.0, 3.0])

        base_states, base_local = adapter.spatial_input_states(latent)
        changed_states, changed_local = adapter.spatial_input_states(changed)
        state_changes = (changed_states - base_states).abs().sum(dim=-1).squeeze(0)
        local_changes = (changed_local - base_local).abs().sum(dim=-1).squeeze(0)

        self.assertEqual(torch.nonzero(state_changes > 0, as_tuple=False).flatten().tolist(), [4])
        self.assertEqual(torch.nonzero(local_changes > 0, as_tuple=False).flatten().tolist(), [4])
        self.assertEqual(tuple(adapter.forward_soft_prompts(latent).shape), (1, 6, 24))

    def test_spatial_adapter_rejects_token_grid_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "one output token per latent-grid position"):
            TensorPatchAlignmentAdapter(
                latent_channels=3,
                latent_grid=(2, 3),
                adapter_dim=16,
                projection_dim=24,
                dropout=0.0,
                adapter_type="spatial_transformer",
                query_tokens=5,
                adapter_layers=1,
                adapter_heads=4,
                soft_prompt_scale=0.0,
            )

    def test_spatial_adapter_parameter_metrics_are_read_only_scalars(self) -> None:
        adapter = TensorPatchAlignmentAdapter(
            latent_channels=3,
            latent_grid=(2, 2),
            adapter_dim=16,
            projection_dim=24,
            dropout=0.0,
            adapter_type="spatial_transformer",
            query_tokens=4,
            adapter_layers=1,
            adapter_heads=4,
            soft_prompt_scale=0.05,
        )

        metrics = alignment_adapter_parameter_metrics(adapter)
        parameter_names = dict(adapter.named_parameters())
        buffer_names = dict(adapter.named_buffers())

        self.assertEqual(metrics, {"spatial_pos_scale": 1.0, "local_residual_scale": 1.0})
        self.assertNotIn("spatial_pos_scale", parameter_names)
        self.assertNotIn("local_residual_scale", parameter_names)
        self.assertIn("spatial_pos_scale", buffer_names)
        self.assertIn("local_residual_scale", buffer_names)

        adapter.capture_spatial_path_metrics = True
        adapter.forward_soft_prompts(torch.randn(2, 3, 2, 2))
        path_metrics = alignment_adapter_path_metrics(adapter)
        self.assertGreater(path_metrics["spatial_position_to_content_rms_ratio"], 0.0)
        self.assertGreater(path_metrics["local_residual_to_context_rms_ratio"], 0.0)

    def test_spatial_adapter_resets_legacy_trainable_scales_when_loading(self) -> None:
        kwargs = {
            "latent_channels": 3,
            "latent_grid": (2, 2),
            "adapter_dim": 16,
            "projection_dim": 24,
            "dropout": 0.0,
            "adapter_type": "spatial_transformer",
            "query_tokens": 4,
            "adapter_layers": 1,
            "adapter_heads": 4,
            "soft_prompt_scale": 0.05,
        }
        source = TensorPatchAlignmentAdapter(**kwargs)
        legacy_state = source.state_dict()
        legacy_state["spatial_pos_scale"] = torch.tensor(0.2)
        legacy_state["local_residual_scale"] = torch.tensor(0.3)
        restored = TensorPatchAlignmentAdapter(**kwargs)

        restored.load_state_dict(legacy_state, strict=True)

        self.assertEqual(float(restored.spatial_pos_scale), 1.0)
        self.assertEqual(float(restored.local_residual_scale), 1.0)

    def test_value_preserving_ae_keeps_exact_input_at_each_latent_position(self) -> None:
        model = ConvTokenAutoencoder2D(
            {
                "model": {
                    "input_size": [4, 4],
                    "in_channels": 1,
                    "out_channels": 1,
                    "base_channels": 4,
                    "channel_multipliers": [],
                    "num_res_blocks": 0,
                    "latent_dim": 3,
                    "latent_grid": [4, 4],
                    "dropout": 0.0,
                    "norm": "identity",
                    "activation": "gelu",
                    "output_activation": "identity",
                    "preserve_input_channels": True,
                }
            }
        )
        inputs = torch.randn(2, 1, 4, 4)
        latent = model.encode(inputs)["latent_map"]

        self.assertEqual(tuple(latent.shape), (2, 3, 4, 4))
        torch.testing.assert_close(latent[:, :1], inputs, rtol=0.0, atol=0.0)

    def test_one_based_question_coordinates_map_to_zero_based_structured_features(self) -> None:
        one_based = {
            "task_type": "normalized_point_value",
            "question": "Read row 1, column 16.",
            "metadata": {"grid_shape": [16, 16], "coordinate_origin": 1},
            "choices": ["A", "B", "C", "D"],
        }
        zero_based = {
            **one_based,
            "question": "Read row 0, column 15.",
            "metadata": {"grid_shape": [16, 16], "coordinate_origin": 0},
        }

        self.assertEqual(
            structured_query_features_for_record(one_based),
            structured_query_features_for_record(zero_based),
        )

    def test_numeric_task_instructions_match_standardized_encoder_input(self) -> None:
        normalized = task_specific_instruction({"task_type": "normalized_point_value"})
        raw = task_specific_instruction({"task_type": "raw_point_value_with_stats"})

        self.assertIn("read the standardized value z directly", normalized)
        self.assertIn("x = mean + scale * z", raw)

    def test_local_prompt_contains_numeric_options_and_exact_output_contract(self) -> None:
        record = {
            "qa_id": "numeric-1",
            "task_type": "normalized_point_value",
            "query": "Read row 3, column 7. Options: A: -0.5; B: 0.0; C: 0.5; D: 1.0.",
            "question": "Read row 3, column 7. Options: A: -0.5; B: 0.0; C: 0.5; D: 1.0.",
            "choices": ["A", "B", "C", "D"],
        }

        prompt = build_local_conditioning_prompt(record, prompt_template="task_specific")

        self.assertIn("Options: A: -0.5; B: 0.0; C: 0.5; D: 1.0", prompt)
        self.assertIn("exactly one of A, B, C, D", prompt)
        self.assertIn("no explanation, punctuation, or other text", prompt)
        self.assertNotIn("Answer:", prompt)
        self.assertTrue(prompt.endswith("Tensor evidence requested:"))

    def test_generated_choice_parser_separates_correct_semantics_from_format(self) -> None:
        exact = parse_generated_choice(" B ", ["A", "B", "C", "D"])
        verbose = parse_generated_choice("The answer is B.", ["A", "B", "C", "D"])
        ambiguous = parse_generated_choice("A or B", ["A", "B", "C", "D"])

        self.assertTrue(exact["format_valid"])
        self.assertEqual(exact["parsed_choice"], "B")
        self.assertFalse(verbose["format_valid"])
        self.assertEqual(verbose["parsed_choice"], "B")
        self.assertFalse(ambiguous["format_valid"])
        self.assertIsNone(ambiguous["parsed_choice"])

    def test_generated_choice_parser_handles_overlapping_bin_labels(self) -> None:
        parsed = parse_generated_choice("B01", ["B00", "B01", "B02"])

        self.assertTrue(parsed["format_valid"])
        self.assertEqual(parsed["matched_choices"], ["B01"])

    def test_swap_indices_stay_within_state_task_and_field(self) -> None:
        records = [
            _record("s1", "point", "Vx", "question one"),
            _record("s1", "point", "Vx", "question two"),
            _record("s1", "point", "Vy", "question three"),
            _record("s2", "point", "Vx", "question four"),
        ]

        owners, swapped = same_state_question_swap_indices(records)

        self.assertEqual(owners, [0, 1])
        self.assertEqual(swapped, [1, 0])

    def test_grounding_swaps_skip_same_answer_pairs(self) -> None:
        records = [
            {**_record("s1", "point", "Vx", "question one"), "answer": "A"},
            {**_record("s1", "point", "Vx", "question two"), "answer": "A"},
            {**_record("s1", "point", "Vx", "question three"), "answer": "B"},
        ]

        owners, swapped = same_state_question_swap_indices(
            records,
            require_different_answers=True,
        )

        self.assertEqual(owners, [0, 1, 2])
        self.assertEqual(swapped, [2, 2, 0])
        self.assertTrue(
            all(
                records[owner]["answer"] != records[source]["answer"]
                for owner, source in zip(owners, swapped)
            )
        )

    def test_zero_text_gate_preserves_inherited_qformer_output(self) -> None:
        torch.manual_seed(7)
        aligned = TensorPatchAlignmentAdapter(
            latent_channels=8,
            latent_grid=(2, 2),
            adapter_dim=16,
            projection_dim=24,
            dropout=0.0,
            adapter_type="qformer",
            query_tokens=4,
            adapter_layers=2,
            adapter_heads=4,
            soft_prompt_scale=0.05,
        ).eval()
        conditioned = ResidualQuestionConditionedAdapter(
            aligned_adapter=aligned,
            llm_hidden_size=24,
            context_layers=(1, 2),
            adapter_heads=4,
            dropout=0.0,
            text_gate_init=0.0,
            residual_gate_init=0.1,
        ).eval()
        latent = torch.randn(3, 8, 2, 2)
        question = torch.randn(3, 2, 6, 24)
        mask = torch.ones(3, 6, dtype=torch.bool)

        expected = aligned.forward_soft_prompts(latent)
        actual = conditioned(latent, question, mask, structured_query=None)

        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_zero_text_gate_preserves_inherited_spatial_output(self) -> None:
        torch.manual_seed(13)
        aligned = TensorPatchAlignmentAdapter(
            latent_channels=3,
            latent_grid=(2, 2),
            adapter_dim=16,
            projection_dim=24,
            dropout=0.0,
            adapter_type="spatial_transformer",
            query_tokens=4,
            adapter_layers=2,
            adapter_heads=4,
            soft_prompt_scale=0.05,
        ).eval()
        reloaded = TensorPatchAlignmentAdapter(
            latent_channels=3,
            latent_grid=(2, 2),
            adapter_dim=16,
            projection_dim=24,
            dropout=0.0,
            adapter_type="spatial_transformer",
            query_tokens=4,
            adapter_layers=2,
            adapter_heads=4,
            soft_prompt_scale=0.05,
        ).eval()
        reloaded.load_state_dict(aligned.state_dict(), strict=True)
        conditioned = ResidualQuestionConditionedAdapter(
            aligned_adapter=reloaded,
            llm_hidden_size=24,
            context_layers=(1, 2),
            adapter_heads=4,
            dropout=0.0,
            text_gate_init=0.0,
            residual_gate_init=0.1,
        ).eval()
        latent = torch.randn(3, 3, 2, 2)
        question = torch.randn(3, 2, 6, 24)
        mask = torch.ones(3, 6, dtype=torch.bool)

        expected = aligned.forward_soft_prompts(latent)
        actual = conditioned(latent, question, mask, structured_query=None)

        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_grounded_spatial_reader_has_no_trainable_unconditioned_path(self) -> None:
        torch.manual_seed(23)
        aligned = TensorPatchAlignmentAdapter(
            latent_channels=3,
            latent_grid=(2, 2),
            adapter_dim=16,
            projection_dim=24,
            dropout=0.0,
            adapter_type="spatial_transformer",
            query_tokens=4,
            adapter_layers=2,
            adapter_heads=4,
            soft_prompt_scale=0.05,
        ).eval()
        conditioned = ResidualQuestionConditionedAdapter(
            aligned_adapter=aligned,
            llm_hidden_size=24,
            context_layers=(1, 2),
            adapter_heads=4,
            dropout=0.0,
            text_gate_init=1.0,
            residual_gate_init=1.0,
            freeze_backbone=True,
            text_gate_trainable=False,
            residual_gate_trainable=False,
            zero_init_text_attention=True,
        ).train()
        latent = torch.randn(2, 3, 2, 2)
        question = torch.randn(2, 2, 5, 24)
        mask = torch.ones(2, 5, dtype=torch.bool)

        self.assertFalse(any(parameter.requires_grad for parameter in conditioned.backbone.parameters()))
        self.assertFalse(conditioned.backbone.training)
        self.assertFalse(conditioned.gate.requires_grad)
        self.assertTrue(all(not block.gate.requires_grad for block in conditioned.text_blocks))
        self.assertTrue(
            all(
                int(torch.count_nonzero(block.attention.out_proj.weight).item()) == 0
                for block in conditioned.text_blocks
            )
        )
        torch.testing.assert_close(
            conditioned(latent, question, mask, structured_query=None),
            aligned.forward_soft_prompts(latent),
            rtol=0.0,
            atol=0.0,
        )

        output = conditioned(latent, question, mask, structured_query=None)
        output.sum().backward()
        self.assertTrue(all(parameter.grad is None for parameter in conditioned.backbone.parameters()))
        self.assertGreater(
            float(conditioned.text_blocks[0].attention.out_proj.weight.grad.abs().sum().item()),
            0.0,
        )

    def test_global_only_baseline_bypasses_question_conditioned_branch(self) -> None:
        torch.manual_seed(29)
        aligned = TensorPatchAlignmentAdapter(
            latent_channels=3,
            latent_grid=(2, 2),
            adapter_dim=16,
            projection_dim=24,
            dropout=0.0,
            adapter_type="spatial_transformer",
            query_tokens=4,
            adapter_layers=2,
            adapter_heads=4,
            soft_prompt_scale=0.05,
        )
        local = ResidualQuestionConditionedAdapter(
            aligned_adapter=aligned,
            llm_hidden_size=24,
            context_layers=(1, 2),
            adapter_heads=4,
            dropout=0.0,
            text_gate_init=1.0,
            residual_gate_init=1.0,
            freeze_backbone=True,
            text_gate_trainable=False,
            residual_gate_trainable=False,
            zero_init_text_attention=True,
        )
        adapter = HybridGlobalLocalAdapter(
            global_adapter=aligned,
            local_adapter=local,
            freeze_global=True,
            combine_mode="residual",
        )
        latent = torch.randn(2, 3, 2, 2)
        text_embeds = torch.randn(2, 5, 24)

        with mock.patch.object(
            local,
            "forward",
            side_effect=AssertionError("global_only must not execute the local branch"),
        ):
            actual = adapter_training.adapter_soft_embeds(
                adapter=adapter,
                latent_map=latent,
                text_embeds=text_embeds,
                question_embeds=None,
                question_mask=None,
                records=None,
                mode="global_only",
            )

        torch.testing.assert_close(
            actual,
            aligned.forward_soft_prompts(latent).to(dtype=text_embeds.dtype),
            rtol=0.0,
            atol=0.0,
        )

    def test_spatial_stage1_checkpoint_rebuilds_strictly_for_downstream(self) -> None:
        torch.manual_seed(17)
        aligned = TensorPatchAlignmentAdapter(
            latent_channels=3,
            latent_grid=(2, 2),
            adapter_dim=16,
            projection_dim=24,
            dropout=0.0,
            adapter_type="spatial_transformer",
            query_tokens=4,
            adapter_layers=2,
            adapter_heads=4,
            soft_prompt_scale=0.05,
        ).eval()
        checkpoint = {
            "args": {
                "adapter_type": "spatial_transformer",
                "adapter_dim": 16,
                "adapter_layers": 2,
                "adapter_heads": 4,
                "query_tokens": 4,
                "projection_dim": 24,
                "dropout": 0.0,
                "soft_prompt_scale": 0.05,
            },
            "adapter_state_dict": aligned.state_dict(),
        }
        latent = torch.randn(2, 3, 2, 2)

        rebuilt = adapter_from_checkpoint(checkpoint, latent_shape=(3, 2, 2), llm_hidden_size=24).eval()

        self.assertIsInstance(rebuilt, TensorPatchAlignmentAdapter)
        self.assertEqual(rebuilt.adapter_type, "spatial_transformer")
        torch.testing.assert_close(
            rebuilt.forward_soft_prompts(latent),
            aligned.forward_soft_prompts(latent),
            rtol=0.0,
            atol=0.0,
        )

    def test_spatial_residual_checkpoint_rebuilds_strictly(self) -> None:
        torch.manual_seed(19)
        aligned = TensorPatchAlignmentAdapter(
            latent_channels=3,
            latent_grid=(2, 2),
            adapter_dim=16,
            projection_dim=24,
            dropout=0.0,
            adapter_type="spatial_transformer",
            query_tokens=4,
            adapter_layers=2,
            adapter_heads=4,
            soft_prompt_scale=0.05,
        )
        local = ResidualQuestionConditionedAdapter(
            aligned_adapter=aligned,
            llm_hidden_size=24,
            context_layers=(1, 2),
            adapter_heads=4,
            dropout=0.0,
            text_gate_init=1.0,
            residual_gate_init=1.0,
            freeze_backbone=True,
            text_gate_trainable=False,
            residual_gate_trainable=False,
            zero_init_text_attention=True,
        )
        original = HybridGlobalLocalAdapter(
            global_adapter=aligned,
            local_adapter=local,
            freeze_global=True,
            combine_mode="residual",
        ).eval()
        checkpoint = {
            "args": {
                "adapter_architecture": "residual_question_adapter",
                "global_adapter_type": "spatial_transformer",
                "adapter_dim": 16,
                "adapter_layers": 2,
                "adapter_heads": 4,
                "projection_dim": 24,
                "dropout": 0.0,
                "soft_prompt_scale": 0.05,
                "local_context_layers": "1,2",
                "local_text_gate_init": 1.0,
                "local_gate_init": 1.0,
                "freeze_conditioned_backbone": True,
                "local_text_gate_trainable": False,
                "local_residual_gate_trainable": False,
                "zero_init_local_text_attention": True,
            },
            "adapter_state_dict": original.state_dict(),
        }
        latent = torch.randn(2, 3, 2, 2)
        question = torch.randn(2, 2, 5, 24)
        mask = torch.ones(2, 5, dtype=torch.bool)

        rebuilt = adapter_from_checkpoint(checkpoint, latent_shape=(3, 2, 2), llm_hidden_size=24).eval()

        self.assertIsInstance(rebuilt, HybridGlobalLocalAdapter)
        self.assertFalse(any(parameter.requires_grad for parameter in rebuilt.local_adapter.backbone.parameters()))
        self.assertFalse(rebuilt.local_adapter.gate.requires_grad)
        torch.testing.assert_close(
            rebuilt(latent, question, mask),
            original(latent, question, mask),
            rtol=0.0,
            atol=0.0,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
