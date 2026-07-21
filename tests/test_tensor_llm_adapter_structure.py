from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_tensor_llm_adapter import (  # noqa: E402
    ExactDistributedEvalSampler,
    HybridGlobalLocalAdapter,
    ResidualQuestionConditionedAdapter,
    StateTaskGroupedBatchSampler,
    adapter_from_checkpoint,
    build_local_conditioning_prompt,
    parse_generated_choice,
    same_state_question_swap_indices,
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
            text_gate_init=0.05,
            residual_gate_init=0.1,
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
                "local_text_gate_init": 0.05,
                "local_gate_init": 0.1,
            },
            "adapter_state_dict": original.state_dict(),
        }
        latent = torch.randn(2, 3, 2, 2)
        question = torch.randn(2, 2, 5, 24)
        mask = torch.ones(2, 5, dtype=torch.bool)

        rebuilt = adapter_from_checkpoint(checkpoint, latent_shape=(3, 2, 2), llm_hidden_size=24).eval()

        self.assertIsInstance(rebuilt, HybridGlobalLocalAdapter)
        torch.testing.assert_close(
            rebuilt(latent, question, mask),
            original(latent, question, mask),
            rtol=0.0,
            atol=0.0,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
