from __future__ import annotations

import pytest
import torch

from scripts.test_qwen_numeric_matrix_tasks import (
    aggregate_qwen_numeric_test_cases,
    build_qwen_numeric_test_prompt,
    normalize_thinking_mode,
    parse_generated_numeric_answer,
    render_numeric_chat_prompt,
)
from scripts.train_tensor_patch_text_alignment import AlignmentAnchor


class _ThinkingChatTokenizer:
    chat_template = "qwen3-like enable_thinking"

    def __init__(self) -> None:
        self.kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.kwargs = kwargs
        return "rendered"


class _LegacyChatTokenizer:
    chat_template = "legacy"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return "legacy-rendered"


def test_qwen_numeric_prompt_contains_matrix_and_task_but_not_expected_answer() -> None:
    patch = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    anchor = AlignmentAnchor(
        name="point",
        mode="probe",
        token_ids=(1,),
        text="\nThe value is",
        probe_family="point_value",
        probe_template_index=0,
        probe_parameters=(0, 1, 0),
    )

    prompt = build_qwen_numeric_test_prompt(patch, anchor, decimal_places=2)

    assert "[[[1.00, 2.00]; [3.00, 4.00]]]" in prompt
    assert "row 2, column 1" in prompt
    assert prompt.endswith("Answer:")
    assert "expected" not in prompt.lower()


def test_generated_numeric_answer_uses_final_finite_number() -> None:
    assert parse_generated_numeric_answer("The calculation is 1.0 - 0.5 = 0.5") == pytest.approx(0.5)
    assert parse_generated_numeric_answer("Answer: -1.25e-2") == pytest.approx(-0.0125)
    assert parse_generated_numeric_answer("No numeric answer") is None


def test_numeric_chat_prompt_disables_qwen3_thinking_when_supported() -> None:
    tokenizer = _ThinkingChatTokenizer()
    rendered, uses_chat_template, control_applied = render_numeric_chat_prompt(
        tokenizer,
        [{"role": "system", "content": "system"}, {"role": "user", "content": "user"}],
        "disabled",
    )

    assert rendered == "rendered"
    assert uses_chat_template is True
    assert control_applied is True
    assert tokenizer.kwargs["enable_thinking"] is False
    assert normalize_thinking_mode("disabled") == "disabled"


def test_numeric_chat_prompt_falls_back_for_templates_without_thinking_argument() -> None:
    rendered, uses_chat_template, control_applied = render_numeric_chat_prompt(
        _LegacyChatTokenizer(),
        [{"role": "system", "content": "system"}, {"role": "user", "content": "user"}],
        "disabled",
    )

    assert rendered == "legacy-rendered"
    assert uses_chat_template is True
    assert control_applied is False


def test_qwen_numeric_case_aggregation_keeps_sources_separate() -> None:
    cases = [
        {
            "source": "synthetic_short",
            "probe_family": "point_value",
            "prediction": 1.0,
            "absolute_error": 0.0,
            "rounded_exact": True,
            "within_tolerance": True,
        },
        {
            "source": "pde_patch",
            "probe_family": "point_value",
            "prediction": None,
            "absolute_error": None,
            "rounded_exact": False,
            "within_tolerance": False,
        },
    ]

    metrics = aggregate_qwen_numeric_test_cases(cases, absolute_tolerance=0.02)

    assert metrics["macro"]["parsed_fraction"] == pytest.approx(0.5)
    assert metrics["by_source"]["synthetic_short"]["within_tolerance_accuracy"] == 1.0
    assert metrics["by_source"]["pde_patch"]["within_tolerance_accuracy"] == 0.0
