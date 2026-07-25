from __future__ import annotations

import argparse
import unittest

import torch

from scripts.diagnose_direct_tensor_grounding import (
    _bootstrap_ci,
    _categorical_kl,
    _edited_latent,
    _entropy,
    _numeric_intervention_spec,
    _prompt_representation_stats,
    _official_shuffled_stats_record,
    _ranks,
    diagnostic_screening_flags,
    make_role_swap_record,
    parse_coordinates,
    parse_numeric_options,
    parse_region_specs,
    preflight_checkpoint_envelope,
    region_cells,
    text_control_prompt,
    validate_diagnostic_args,
)


def point_record(task: str = "normalized_point_value") -> dict:
    query = (
        "The tensor soft tokens encode the per-patch standardized 4 by 4 matrix z of pressure. "
        "The standardization is z = (x - mean) / scale, where mean is 10 and scale is 2. "
        "Which option is closest to z at row 2, column 4? "
        "Options: A: -1.0; B: 0.5; C: 1.5; D: 2.5."
    )
    if task == "raw_point_value_with_stats":
        query = (
            "The tensor soft tokens encode the per-patch standardized 4 by 4 matrix z of pressure. "
            "Recover an original value with x = mean + scale * z, where mean is 10 and scale is 2. "
            "Which option is closest to the original value x at row 2, column 4? "
            "Options: A: 8; B: 11; C: 13; D: 15."
        )
    return {
        "qa_id": "q0",
        "state_ref": "s0",
        "task_type": task,
        "query": query,
        "question": query,
        "choices": ["A", "B", "C", "D"],
        "answer": "B",
        "metadata": {"coordinate_origin": 1, "grid_shape": [4, 4]},
    }


class TestDirectGroundingPrimitives(unittest.TestCase):
    def test_checkpoint_preflight_accepts_direct_and_grounded_architectures(self) -> None:
        latent_contract = {"format": "latent-v1"}
        for architecture in ("alignment_adapter", "grounded_evidence_adapter"):
            checkpoint = {
                "checkpoint_type": "tensor_llm_adapter",
                "checkpoint_version": 2,
                "adapter_state_dict": {"weight": torch.ones(1)},
                "args": {"adapter_architecture": architecture},
                "latent_contract": latent_contract,
            }

            preflight_checkpoint_envelope(
                checkpoint,
                expected_architecture=architecture,
                expected_latent_contract=latent_contract,
            )

        with self.assertRaisesRegex(ValueError, "differs from the requested"):
            preflight_checkpoint_envelope(
                checkpoint,
                expected_architecture="alignment_adapter",
                expected_latent_contract=latent_contract,
            )

    def test_zero_max_records_means_unlimited(self) -> None:
        args = argparse.Namespace(
            max_records=0,
            max_states_per_task=1,
            intervention_records_per_task=1,
            gradient_records_per_task=1,
            score_batch_size=1,
            probe_train_states=1,
            probe_val_states=1,
            probe_positions_per_state=1,
            probe_feature_batch_size=1,
            text_control_records_per_task=1,
            text_control_max_prompt_tokens=1,
            representation_states=1,
            intervention_controls_per_record=0,
            bootstrap_reps=0,
            min_point_gap=0.0,
            min_region_gap=0.0,
            probe_ridge=1.0,
        )
        validate_diagnostic_args(args)

    def test_representation_rank_sampling_spans_all_rows(self) -> None:
        prompt = torch.arange(4 * 16 * 8, dtype=torch.float32).reshape(4, 16, 8)
        stats = _prompt_representation_stats(prompt)

        self.assertEqual(stats["effective_rank_sample_shape"], [64, 8])
        self.assertEqual(
            stats["effective_rank_sampling"],
            "evenly_spaced_across_batch_and_tokens",
        )

    def test_screening_uses_conditional_intervention_metric(self) -> None:
        report = {
            "local_interventions": {
                "n_records": 10,
                "target_flip_eligible_n": 5,
                # Absolute success is high because clean predictions already
                # equal the edited label; the conditional metric is poor.
                "target_intended_label_success": {"mean": 0.9},
                "target_flip_from_clean": {"mean": 0.2},
                "non_target_false_flip_rate": {"mean": 0.0},
                "target_vs_control_selective_margin_effect_eligible": {"mean": 0.1},
            }
        }
        flags = diagnostic_screening_flags(report)
        names = {item["flag"] for item in flags}
        self.assertIn("weak_numeric_target_sensitivity", names)

    def test_shuffled_stats_control_rebuilds_question_without_changing_answer(self) -> None:
        record = point_record("raw_point_value_with_stats")
        record["prompt_data"] = {
            "field": "pressure",
            "row": 2,
            "col": 4,
            "patch_size": 4,
            "significant_digits": 6,
            "mean": 10.0,
            "scale": 2.0,
            "option_text": "A: 8; B: 11; C: 13; D: 15",
        }
        donor = {"prompt_data": {"mean": 20.0, "scale": 4.0}}

        changed = _official_shuffled_stats_record(record, donor)

        self.assertIsNotNone(changed)
        self.assertEqual(changed["answer"], record["answer"])
        self.assertIn("mean is 20", changed["query"])
        self.assertIn("scale is 4", changed["query"])
        self.assertGreaterEqual(_categorical_kl({"logits": {"A": 1.0, "B": 0.0}}, {"logits": {"A": 0.0, "B": 1.0}}), 0.0)

    def test_one_based_coordinate_parsing(self) -> None:
        self.assertEqual(parse_coordinates(point_record()), [(1, 3)])

    def test_two_coordinate_role_swap_flips_answer(self) -> None:
        query = (
            "Which location is larger: A at row 1, column 2, "
            "or B at row 3, column 4?"
        )
        record = {
            "qa_id": "pair",
            "state_ref": "s0",
            "query": query,
            "question": query,
            "choices": ["A", "B"],
            "answer": "A",
            "metadata": {"coordinate_origin": 1},
        }

        pair = make_role_swap_record(record)

        self.assertIsNotNone(pair)
        original, swapped = pair
        self.assertEqual(parse_coordinates(original), [(0, 1), (2, 3)])
        self.assertEqual(parse_coordinates(swapped), [(2, 3), (0, 1)])
        self.assertEqual(swapped["answer"], "B")

    def test_numeric_options_and_normalized_target_edit(self) -> None:
        record = point_record()
        latent = torch.zeros(2, 4, 4)
        latent[0, 1, 3] = 0.5

        options = parse_numeric_options(record)
        spec = _numeric_intervention_spec(record, latent)

        self.assertEqual(options, {"A": -1.0, "B": 0.5, "C": 1.5, "D": 2.5})
        self.assertIsNotNone(spec)
        self.assertEqual(spec["coordinate"], (1, 3))
        self.assertNotEqual(spec["target_label"], record["answer"])
        edited = _edited_latent(latent, spec["coordinate"], spec["target_z"])
        self.assertEqual(float(edited[0, 1, 3]), float(spec["target_z"]))
        torch.testing.assert_close(edited[1], latent[1])

    def test_raw_target_is_converted_back_to_z(self) -> None:
        record = point_record("raw_point_value_with_stats")
        latent = torch.zeros(2, 4, 4)
        latent[0, 1, 3] = 0.5

        spec = _numeric_intervention_spec(record, latent)

        self.assertIsNotNone(spec)
        expected_z = (spec["target_value"] - 10.0) / 2.0
        self.assertAlmostEqual(spec["target_z"], expected_z)

    def test_region_parse_and_overlap_filter_primitive(self) -> None:
        query = (
            "Compare the mean values of two 2 by 2 regions. "
            "Region A starts at row 1, column 1; region B starts at row 2, column 2."
        )
        record = {"query": query, "metadata": {"coordinate_origin": 1}}

        specs = parse_region_specs(record)

        self.assertEqual(specs, ((0, 0), (1, 1), 2, 2))
        cells_a = region_cells(specs[0], specs[2], specs[3], (4, 4))
        cells_b = region_cells(specs[1], specs[2], specs[3], (4, 4))
        self.assertTrue(cells_a.intersection(cells_b))

    def test_rank_uses_midrank_for_ties_and_entropy_is_normalized(self) -> None:
        scores = torch.tensor([0.0, 2.0, 2.0, 1.0])
        rank, top1, top5, percentile = _ranks(scores, 1)

        self.assertEqual(rank, 1.5)
        self.assertEqual(top1, 0.0)
        self.assertEqual(top5, 1.0)
        self.assertGreater(percentile, 0.8)
        self.assertAlmostEqual(_entropy(torch.ones(4)), 1.0, places=6)

    def test_cluster_bootstrap_counts_states_not_rows(self) -> None:
        result = _bootstrap_ci(
            values=[1.0, 1.0, 0.0, 0.0],
            clusters=["a", "a", "b", "b"],
            reps=100,
            seed=7,
        )

        self.assertEqual(result["n"], 4)
        self.assertEqual(result["clusters"], 2)
        self.assertAlmostEqual(result["mean"], 0.5)

    def test_text_control_never_claims_a_soft_prefix_exists(self) -> None:
        prompt = text_control_prompt(point_record(), "standardized z at the requested cell is 0.5.")

        self.assertNotIn("soft tokens", prompt.lower())
        self.assertIn("standardized z at the requested cell is 0.5", prompt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
