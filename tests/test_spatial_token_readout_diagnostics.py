from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.diagnose_spatial_token_readout import (  # noqa: E402
    adapter_structure_summary,
    coordinate_routing_groups,
    extract_spatial_token_stages,
    fit_ridge_readout,
    point_coordinate_from_record,
    restore_checkpoint_runtime_args,
    routing_group_statistics,
    row_major_index,
    summarize_routing_statistics,
    unique_state_examples,
)
from scripts.train_tensor_llm_adapter import (  # noqa: E402
    HybridGlobalLocalAdapter,
    ResidualQuestionConditionedAdapter,
)
from scripts.train_tensor_patch_text_alignment import TensorPatchAlignmentAdapter  # noqa: E402


class TestSpatialReadoutDiagnosticPrimitives(unittest.TestCase):
    def test_row_major_coordinate_and_one_based_text_are_consistent(self) -> None:
        record = {
            "question": "Which value is at row 2, column 4?",
            "metadata": {"grid_shape": [3, 5], "coordinate_origin": 1},
        }

        coordinate = point_coordinate_from_record(record)

        self.assertEqual(coordinate, (1, 3))
        self.assertEqual(row_major_index(*coordinate, height=3, width=5), 8)

    def test_multi_point_question_is_not_misclassified_as_single_point(self) -> None:
        record = {
            "question": "Compare row 1, column 2 with row 3, column 4.",
            "metadata": {"grid_shape": [4, 4], "coordinate_origin": 1},
        }

        self.assertIsNone(point_coordinate_from_record(record))

    def test_unique_state_selection_never_reuses_qa_variants(self) -> None:
        records = [
            {"state_ref": "state_a", "variant": 0},
            {"state_ref": "state_a", "variant": 1},
            {"state_ref": "state_b", "variant": 0},
        ]
        dataset = SimpleNamespace(
            records=records,
            jsonl_path="synthetic.jsonl",
            load_latent_for_record=lambda record: torch.tensor([float(record["variant"])]),
        )

        selected = unique_state_examples(dataset, limit=2)

        self.assertEqual([record["state_ref"] for record, _latent in selected], ["state_a", "state_b"])

    def test_checkpoint_runtime_is_authoritative_except_for_explicit_cli_overrides(self) -> None:
        args = SimpleNamespace(
            model_name_or_path="config-32b",
            qa_dir="config-qa",
            latent_dir="config-latents",
            train_split="train",
            val_split="val",
            prompt_template="task_specific",
            max_prompt_tokens=512,
            local_context_layer=6,
            prefer_record_latent_ref=False,
            torch_dtype="bfloat16",
            trust_remote_code=False,
            explicit_runtime_overrides={"qa_dir": True},
        )
        checkpoint = {
            "args": {
                "model_name_or_path": "checkpoint-14b",
                "qa_dir": "checkpoint-qa",
                "latent_dir": "checkpoint-latents",
            }
        }

        restored = restore_checkpoint_runtime_args(args, checkpoint)

        self.assertEqual(args.model_name_or_path, "checkpoint-14b")
        self.assertEqual(args.qa_dir, "config-qa")
        self.assertEqual(args.latent_dir, "checkpoint-latents")
        self.assertEqual(set(restored), {"model_name_or_path", "latent_dir"})

    def test_closed_form_probe_recovers_a_shared_linear_value(self) -> None:
        torch.manual_seed(5)
        train_features = torch.randn(256, 6)
        val_features = torch.randn(128, 6)
        train_target = 1.7 * train_features[:, 0] - 0.4 * train_features[:, 2] + 0.2
        val_target = 1.7 * val_features[:, 0] - 0.4 * val_features[:, 2] + 0.2

        metrics = fit_ridge_readout(
            train_features,
            train_target,
            val_features,
            val_target,
            ridge=1.0e-6,
            tolerance=0.01,
            device=torch.device("cpu"),
        )

        self.assertGreater(metrics["val"]["r2"], 0.999)
        self.assertGreater(metrics["val"]["within_tolerance_fraction"], 0.99)

    def test_routing_rank_recovers_the_question_specific_target_token(self) -> None:
        features = torch.zeros(3, 4, 2)
        coordinates = [(0, 0), (0, 1), (1, 0)]
        for question_index, (row, col) in enumerate(coordinates):
            features[question_index, row_major_index(row, col, 2, 2), 0] = 3.0

        raw = routing_group_statistics(features, coordinates, height=2, width=2)
        summary = summarize_routing_statistics(raw, token_count=4)

        self.assertEqual(summary["target_top1"], 1.0)
        self.assertEqual(summary["target_top5"], 1.0)
        self.assertGreater(summary["target_to_non_target_ratio_mean"], 1.0)

    def test_routing_groups_require_distinct_coordinates_without_task_whitelist(self) -> None:
        records = [
            {
                "state_ref": "s0",
                "task_type": "unseen_point_task",
                "field": "f",
                "question": f"Read row 1, column {column}.",
                "metadata": {"grid_shape": [2, 2], "coordinate_origin": 1},
            }
            for column in (1, 2)
        ]

        groups = coordinate_routing_groups(records, max_groups=2, max_questions=4)

        self.assertEqual(len(groups), 1)
        self.assertEqual([coordinate for _record, coordinate in groups[0]], [(0, 0), (0, 1)])


class TestSpatialReadoutStageCapture(unittest.TestCase):
    def test_stage_capture_matches_the_exact_residual_adapter_forward(self) -> None:
        torch.manual_seed(11)
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
            zero_init_text_attention=False,
        )
        adapter = HybridGlobalLocalAdapter(
            global_adapter=aligned,
            local_adapter=local,
            freeze_global=True,
            combine_mode="residual",
        ).eval()
        latent = torch.randn(2, 3, 2, 2)
        question = torch.randn(2, 2, 5, 24)
        mask = torch.ones(2, 5, dtype=torch.bool)

        stages = extract_spatial_token_stages(adapter, latent, question, mask)
        expected = adapter.forward_components(latent, question, mask, structured_query=None)[2]
        structure = adapter_structure_summary(adapter)

        torch.testing.assert_close(stages["combined_soft_prompt"], expected, rtol=0.0, atol=0.0)
        self.assertTrue(all(int(value.shape[1]) == 4 for value in stages.values()))
        self.assertTrue(structure["question_cross_attention"])
        self.assertEqual(
            structure["question_attention_direction"],
            "spatial_tokens_query_natural_language_keys_values",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
