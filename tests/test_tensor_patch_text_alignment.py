from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from scripts.train_tensor_patch_text_alignment import (
    build_numeric_probe_anchor,
    build_static_alignment_anchor,
    build_teacher_texts_for_batch,
    probe_behavior_loss,
    probe_contract_anchors,
    probe_target_indices,
    quantize_probe_values,
    serialize_tensor_values,
    shared_suffix_token_ids,
    tokenize_contents_with_shared_suffix,
    validate_probe_anchor_contract,
)


class CharacterTokenizer:
    pad_token_id = 0
    eos_token_id = 999

    def __call__(self, text, *, add_special_tokens=False, truncation=False):
        assert add_special_tokens is False
        assert truncation is False
        if isinstance(text, str):
            return {"input_ids": [ord(character) + 1 for character in text]}
        return {"input_ids": [[ord(character) + 1 for character in item] for item in text]}


class WordTokenizer:
    pad_token_id = 0
    eos_token_id = 999

    def __init__(self) -> None:
        self.vocabulary: dict[str, int] = {}

    def _encode(self, text: str) -> list[int]:
        token_ids: list[int] = []
        for token in text.split():
            if token not in self.vocabulary:
                self.vocabulary[token] = len(self.vocabulary) + 1
            token_ids.append(self.vocabulary[token])
        return token_ids

    def __call__(self, text, *, add_special_tokens=False, truncation=False):
        assert add_special_tokens is False
        assert truncation is False
        if isinstance(text, str):
            return {"input_ids": self._encode(text)}
        return {"input_ids": [self._encode(item) for item in text]}


def test_serialize_tensor_values_contains_only_values_and_shape_delimiters() -> None:
    patch = torch.tensor([[[1.25, -2.0], [3.126, 4.0]]])

    text = serialize_tensor_values(patch, decimal_places=2)

    assert text == "[[[1.25, -2.00]; [3.13, 4.00]]]"
    assert "field" not in text.lower()
    assert "representation" not in text.lower()


def test_teacher_text_serializes_the_same_quantized_values_used_by_probe_labels() -> None:
    patches = torch.tensor([[[[1.00003, 1.00004], [0.0, 0.0]]]])
    args = SimpleNamespace(
        teacher_text_source="raw",
        alignment_text_layout="values_shared_suffix",
        text_decimal_places=4,
    )

    texts = build_teacher_texts_for_batch({"patch": patches}, patches, args)

    assert texts == ["[[[1.0000, 1.0000]; [0.0000, 0.0000]]]"]


def test_probe_quantization_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        quantize_probe_values(torch.tensor([float("nan")]), decimal_places=4)


def test_shared_suffix_is_tokenized_once_and_appended_exactly() -> None:
    tokenizer = CharacterTokenizer()
    suffix = "\nRepresentation:"
    packed = tokenize_contents_with_shared_suffix(
        tokenizer=tokenizer,
        contents=["[1.0]", "[2.0, 3.0]"],
        suffix=suffix,
        max_tokens=64,
        max_suffix_tokens=32,
        require_under_max_length=True,
        context="test",
    )
    expected_suffix = shared_suffix_token_ids(tokenizer, suffix, max_suffix_tokens=32)

    assert packed.suffix_token_ids == expected_suffix
    assert packed.metrics["suffix_token_count"] == len(expected_suffix)
    for row, length in zip(packed.input_ids, packed.attention_mask.sum(dim=1), strict=True):
        valid = row[: int(length.item())].tolist()
        assert tuple(valid[-len(expected_suffix) :]) == expected_suffix


def test_shared_suffix_layout_rejects_truncated_tensor_content() -> None:
    tokenizer = CharacterTokenizer()

    with pytest.raises(ValueError, match="truncated numeric tensor content"):
        tokenize_contents_with_shared_suffix(
            tokenizer=tokenizer,
            contents=["0123456789"],
            suffix="\nR:",
            max_tokens=8,
            max_suffix_tokens=8,
            require_under_max_length=True,
            context="test",
        )


def test_eos_and_representation_are_distinct_configurable_anchors() -> None:
    tokenizer = CharacterTokenizer()
    eos_anchor = build_static_alignment_anchor(
        tokenizer=tokenizer,
        mode="eos",
        representation_suffix="\nRepresentation:",
        max_anchor_tokens=64,
    )
    representation_anchor = build_static_alignment_anchor(
        tokenizer=tokenizer,
        mode="representation",
        representation_suffix="\nRepresentation:",
        max_anchor_tokens=64,
    )

    assert eos_anchor.token_ids == (tokenizer.eos_token_id,)
    assert tokenizer.eos_token_id not in representation_anchor.token_ids
    assert representation_anchor.token_ids[-1] == ord(":") + 1


def test_behavior_probe_is_deterministic_and_has_patch_derived_targets() -> None:
    tokenizer = CharacterTokenizer()
    anchor = build_numeric_probe_anchor(
        tokenizer=tokenizer,
        patch_size=4,
        channel_count=1,
        families=["point_relation"],
        region_size=2,
        probe_index=7,
        seed=42,
        max_anchor_tokens=256,
    )
    same_anchor = build_numeric_probe_anchor(
        tokenizer=tokenizer,
        patch_size=4,
        channel_count=1,
        families=["point_relation"],
        region_size=2,
        probe_index=7,
        seed=42,
        max_anchor_tokens=256,
    )
    patches = torch.arange(32, dtype=torch.float32).reshape(2, 1, 4, 4)
    targets = probe_target_indices(patches, anchor)
    _channel, row_a, col_a, row_b, col_b = anchor.probe_parameters
    expected = (
        patches[:, 0, row_b, col_b] > patches[:, 0, row_a, col_a]
    ).long()

    assert anchor == same_anchor
    assert anchor.mode == "probe"
    assert anchor.probe_family == "point_relation"
    assert anchor.probe_answers == ("larger", "smaller", "equal")
    assert not any(marker in str(anchor.text) for marker in ("Answer", "A or B", "Choose", "?"))
    assert str(anchor.text).endswith(" is")
    assert torch.equal(targets, expected)

    rounded_patch = torch.zeros(1, 1, 4, 4)
    rounded_patch[0, 0, row_a, col_a] = 1.00003
    rounded_patch[0, 0, row_b, col_b] = 1.00004
    raw_target = probe_target_indices(rounded_patch, anchor)
    text_visible_target = probe_target_indices(quantize_probe_values(rounded_patch, 4), anchor)
    assert raw_target.item() == 1
    assert text_visible_target.item() == 2


@pytest.mark.parametrize(
    "family",
    [
        "point_sign",
        "point_relation",
        "region_mean_relation",
        "region_range_relation",
        "directional_change",
    ],
)
def test_behavior_probe_uses_varied_natural_stems_without_visible_choices(family: str) -> None:
    tokenizer = CharacterTokenizer()
    anchors = [
        build_numeric_probe_anchor(
            tokenizer=tokenizer,
            patch_size=16,
            channel_count=1,
            families=[family],
            region_size=4,
            probe_index=index,
            seed=19,
            max_anchor_tokens=256,
        )
        for index in range(24)
    ]

    assert {anchor.probe_template_index for anchor in anchors} == {0, 1, 2, 3}
    for anchor in anchors:
        text = str(anchor.text)
        assert text.startswith("\n")
        assert text.endswith(" is")
        assert not any(marker in text for marker in ("Answer", "A or B", "A/B", "Choose", "?"))
        assert all(answer not in text for answer in anchor.probe_answers)


def test_probe_contract_rejects_visible_answer_format_or_continuation() -> None:
    tokenizer = CharacterTokenizer()
    anchor = build_numeric_probe_anchor(
        tokenizer=tokenizer,
        patch_size=4,
        channel_count=1,
        families=["point_sign"],
        region_size=2,
        probe_index=0,
        seed=13,
        max_anchor_tokens=256,
    )

    with pytest.raises(ValueError, match="forbidden answer-format"):
        validate_probe_anchor_contract(replace(anchor, text="\nAnswer: the value is"))
    with pytest.raises(ValueError, match="leaks canonical continuations"):
        validate_probe_anchor_contract(replace(anchor, text="\nThe value is positive and its sign is"))


def test_probe_contract_preflight_covers_every_family_and_template() -> None:
    tokenizer = WordTokenizer()
    args = SimpleNamespace(
        alignment_anchor_mode="probe",
        probe_families=[
            "point_sign",
            "point_relation",
            "region_mean_relation",
            "region_range_relation",
            "directional_change",
        ],
        field_sampling_mode="single",
        fields=["density", "pressure", "Vx", "Vy"],
        patch_size=16,
        probe_region_size=4,
        seed=42,
        max_shared_suffix_tokens=96,
    )

    anchors = probe_contract_anchors(tokenizer, args)

    assert len(anchors) == 20
    assert len({(anchor.probe_family, anchor.probe_template_index) for anchor in anchors}) == 20
    assert all(len(anchor.probe_answers) == 3 for anchor in anchors)


def test_probe_answer_ce_backpropagates_through_final_hidden() -> None:
    class TinyCausalLM(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lm_head = torch.nn.Linear(4, 32, bias=False)

        def get_output_embeddings(self) -> torch.nn.Module:
            return self.lm_head

    llm = TinyCausalLM()
    for parameter in llm.parameters():
        parameter.requires_grad_(False)
    student_hidden = torch.randn(2, 4, requires_grad=True)
    teacher_hidden = torch.randn(2, 4)
    targets = torch.tensor([0, 1])

    answer_ce, teacher_kl, metrics = probe_behavior_loss(
        llm=llm,
        student_final_hidden=student_hidden,
        teacher_final_hidden=teacher_hidden,
        targets=targets,
        continuation_token_ids=(10, 11, 12),
        kl_temperature=1.0,
    )
    (answer_ce + teacher_kl).backward()

    assert torch.isfinite(answer_ce)
    assert torch.isfinite(teacher_kl)
    assert student_hidden.grad is not None
    assert "answer_format_rate" in metrics
    assert "completion_latent_gain" in metrics


@pytest.mark.parametrize("family", ["region_mean_relation", "region_range_relation"])
def test_region_probe_targets_cover_mean_and_range(family: str) -> None:
    tokenizer = CharacterTokenizer()
    anchor = build_numeric_probe_anchor(
        tokenizer=tokenizer,
        patch_size=6,
        channel_count=1,
        families=[family],
        region_size=2,
        probe_index=3,
        seed=11,
        max_anchor_tokens=256,
    )
    patches = torch.randn(4, 1, 6, 6)

    targets = probe_target_indices(patches, anchor)
    channel, row_a, col_a, row_b, col_b, size = anchor.probe_parameters
    region_a = patches[:, channel, row_a : row_a + size, col_a : col_a + size].double()
    region_b = patches[:, channel, row_b : row_b + size, col_b : col_b + size].double()
    if family == "region_mean_relation":
        score_a = region_a.sum(dim=(-2, -1))
        score_b = region_b.sum(dim=(-2, -1))
    else:
        score_a = region_a.amax(dim=(-2, -1)) - region_a.amin(dim=(-2, -1))
        score_b = region_b.amax(dim=(-2, -1)) - region_b.amin(dim=(-2, -1))
    expected = torch.full((patches.shape[0],), 2, dtype=torch.long)
    expected = torch.where(score_a > score_b, torch.zeros_like(expected), expected)
    expected = torch.where(score_a < score_b, torch.ones_like(expected), expected)

    assert targets.shape == (4,)
    assert torch.equal(targets, expected)


def test_probe_channel_text_and_target_use_the_same_channel() -> None:
    tokenizer = CharacterTokenizer()
    anchor = build_numeric_probe_anchor(
        tokenizer=tokenizer,
        patch_size=4,
        channel_count=3,
        families=["point_sign"],
        region_size=2,
        probe_index=4,
        seed=31,
        max_anchor_tokens=256,
    )
    channel, row, col = anchor.probe_parameters
    patches = torch.zeros(2, 3, 4, 4)
    patches[0, channel, row, col] = 1.0
    patches[1, channel, row, col] = -1.0

    assert f"channel {channel + 1}" in str(anchor.text)
    assert probe_target_indices(patches, anchor).tolist() == [0, 1]


def test_sign_and_directional_completion_targets_match_sentence_semantics() -> None:
    tokenizer = CharacterTokenizer()
    sign_anchor = build_numeric_probe_anchor(
        tokenizer=tokenizer,
        patch_size=4,
        channel_count=1,
        families=["point_sign"],
        region_size=2,
        probe_index=0,
        seed=5,
        max_anchor_tokens=256,
    )
    sign_patch = torch.zeros(3, 1, 4, 4)
    _channel, row, col = sign_anchor.probe_parameters
    sign_patch[:, 0, row, col] = torch.tensor([2.0, -2.0, 0.0])
    assert probe_target_indices(sign_patch, sign_anchor).tolist() == [0, 1, 2]

    direction_anchor = build_numeric_probe_anchor(
        tokenizer=tokenizer,
        patch_size=4,
        channel_count=1,
        families=["directional_change"],
        region_size=2,
        probe_index=0,
        seed=9,
        max_anchor_tokens=256,
    )
    direction_patch = torch.zeros(3, 1, 4, 4)
    _channel, row_a, col_a, row_b, col_b = direction_anchor.probe_parameters
    direction_patch[:, 0, row_a, col_a] = torch.tensor([1.0, 2.0, 1.0])
    direction_patch[:, 0, row_b, col_b] = torch.tensor([2.0, 1.0, 1.0])
    assert direction_anchor.probe_answers == ("higher", "lower", "unchanged")
    assert probe_target_indices(direction_patch, direction_anchor).tolist() == [0, 1, 2]
