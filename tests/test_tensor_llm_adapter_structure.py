from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_tensor_llm_adapter import (  # noqa: E402
    ResidualQuestionConditionedAdapter,
    build_local_conditioning_prompt,
    parse_generated_choice,
    same_state_question_swap_indices,
)
from train_tensor_patch_text_alignment import TensorPatchAlignmentAdapter  # noqa: E402


def _record(state: str, task: str, field: str, question: str) -> dict[str, str]:
    return {
        "state_ref": state,
        "task_type": task,
        "field": field,
        "query": question,
        "question": question,
    }


class TestQuestionConditionedAdapter(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
