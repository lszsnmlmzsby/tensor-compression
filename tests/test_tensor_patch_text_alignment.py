from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from scripts.train_tensor_patch_text_alignment import (
    build_numeric_probe_anchor,
    build_static_alignment_anchor,
    build_teacher_texts_for_batch,
    gather_with_grad,
    hidden_at_last_non_padding,
    normalize_alignment_embeddings,
    probe_contract_anchors,
    reject_removed_alignment_options,
    serialize_tensor_values,
    shared_suffix_token_ids,
    tokenize_contents_with_shared_suffix,
    validate_probe_anchor_contract,
    validate_teacher_hidden_state_index,
    validate_unmodified_tensor_path,
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


def test_teacher_text_serializes_raw_values_at_configured_precision() -> None:
    patches = torch.tensor([[[[1.00003, 1.00004], [0.0, 0.0]]]])
    args = SimpleNamespace(
        teacher_text_source="raw",
        alignment_text_layout="values_shared_suffix",
        text_decimal_places=4,
    )

    texts = build_teacher_texts_for_batch({"patch": patches}, patches, args)

    assert texts == ["[[[1.0000, 1.0000]; [0.0000, 0.0000]]]"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_tensor_serialization_rejects_non_finite_values(value: float) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        serialize_tensor_values(torch.tensor([[[value]]]), decimal_places=4)


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


def test_eos_anchor_is_preserved_at_final_non_padding_token() -> None:
    tokenizer = CharacterTokenizer()
    eos_anchor = build_static_alignment_anchor(
        tokenizer=tokenizer,
        mode="eos",
        representation_suffix="\nRepresentation:",
        max_anchor_tokens=32,
    )
    packed = tokenize_contents_with_anchor(
        tokenizer=tokenizer,
        contents=["[1.0]", "[2.0, 3.0]"],
        anchor=eos_anchor,
        max_tokens=64,
        require_under_max_length=True,
        context="test eos",
    )

    for row, length in zip(packed.input_ids, packed.attention_mask.sum(dim=1), strict=True):
        assert int(row[int(length.item()) - 1].item()) == tokenizer.eos_token_id


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


def test_hidden_readout_tracks_last_text_token_after_soft_prefix() -> None:
    prefix_tokens = 2
    text_attention_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    hidden = torch.arange(2 * 5, dtype=torch.float32).reshape(2, 5, 1)

    readout = hidden_at_last_non_padding(
        [hidden],
        text_attention_mask,
        teacher_layer=0,
        prefix_tokens=prefix_tokens,
    )

    assert readout[:, 0].tolist() == [3.0, 9.0]


def test_teacher_hidden_state_index_rejects_non_contextual_embedding_output() -> None:
    with pytest.raises(ValueError, match="input embedding"):
        validate_teacher_hidden_state_index(0, 28)
    with pytest.raises(ValueError, match="exceeds the LLM depth"):
        validate_teacher_hidden_state_index(29, 28)

    validate_teacher_hidden_state_index(1, 28)
    validate_teacher_hidden_state_index(28, 28)


def test_unmodified_tensor_path_rejects_normalization_and_clipping() -> None:
    validate_unmodified_tensor_path({"mode": "none"})
    validate_unmodified_tensor_path({})

    with pytest.raises(ValueError, match="unmodified tensor path"):
        validate_unmodified_tensor_path({"mode": "zscore"})
    with pytest.raises(ValueError, match="clip_min/clip_max"):
        validate_unmodified_tensor_path({"mode": "none", "clip_min": -1.0})


def test_primary_alignment_only_l2_normalizes_raw_hidden() -> None:
    student = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
    teacher = torch.tensor([[0.0, 5.0], [6.0, 8.0]])

    student_embedding, teacher_embedding = normalize_alignment_embeddings(student, teacher)

    assert torch.allclose(student_embedding, torch.nn.functional.normalize(student, dim=-1))
    assert torch.allclose(teacher_embedding, torch.nn.functional.normalize(teacher, dim=-1))


def test_single_process_embedding_gather_preserves_gradients() -> None:
    embeddings = torch.tensor([[1.0, 2.0]], requires_grad=True)

    gathered = gather_with_grad(embeddings)
    gathered.square().sum().backward()

    assert torch.equal(embeddings.grad, torch.tensor([[2.0, 4.0]]))


@pytest.mark.parametrize(
    "option",
    [
        "alignment_projection",
        "center_embeddings",
        "centered_contrastive_loss_weight",
        "cosine_loss_weight",
        "probe_answer_ce_weight",
        "probe_teacher_kl_weight",
        "probe_kl_temperature",
        "probe_teacher_preflight_records",
    ],
)
def test_removed_alignment_options_fail_before_training(option: str) -> None:
    with pytest.raises(ValueError, match=option):
        reject_removed_alignment_options({"patch_alignment": {option: False}})


def test_probe_is_deterministic_and_contains_only_readout_metadata() -> None:
    tokenizer = CharacterTokenizer()
    kwargs = {
        "tokenizer": tokenizer,
        "patch_size": 4,
        "channel_count": 1,
        "families": ["point_relation"],
        "region_size": 2,
        "probe_index": 7,
        "seed": 42,
        "max_anchor_tokens": 256,
    }

    anchor = build_numeric_probe_anchor(**kwargs)
    same_anchor = build_numeric_probe_anchor(**kwargs)

    assert anchor == same_anchor
    assert anchor.mode == "probe"
    assert anchor.probe_family == "point_relation"
    assert len(anchor.probe_parameters) == 5
    assert not any(marker in str(anchor.text) for marker in ("Answer", "A or B", "Choose", "?"))
    assert str(anchor.text).startswith("\n")
    assert str(anchor.text).endswith(" is")


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
def test_probe_uses_four_distinct_natural_stems_without_choice_format(family: str) -> None:
    tokenizer = CharacterTokenizer()
    anchors = [
        build_numeric_probe_anchor(
            tokenizer=tokenizer,
            patch_size=16,
            channel_count=1,
            families=[family],
            region_size=4,
            probe_index=0,
            seed=19,
            max_anchor_tokens=256,
            template_index=template_index,
        )
        for template_index in range(4)
    ]

    assert {anchor.probe_template_index for anchor in anchors} == {0, 1, 2, 3}
    assert len({anchor.token_ids for anchor in anchors}) == 4
    for anchor in anchors:
        text = str(anchor.text)
        assert text.startswith("\n")
        assert text.endswith(" is")
        assert not any(marker in text for marker in ("Answer", "A or B", "A/B", "Choose", "?"))


@pytest.mark.parametrize("invalid_text", ["\nAnswer: the value is", "\nIs the value positive?"])
def test_probe_contract_rejects_visible_choice_format(invalid_text: str) -> None:
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

    with pytest.raises(ValueError):
        validate_probe_anchor_contract(replace(anchor, text=invalid_text))


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
    for family in args.probe_families:
        family_anchors = [anchor for anchor in anchors if anchor.probe_family == family]
        assert len(family_anchors) == 4
        assert len({anchor.token_ids for anchor in family_anchors}) == 4
        assert len({anchor.probe_parameters for anchor in family_anchors}) == 1


def test_probe_channel_text_matches_parameter_channel() -> None:
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

    assert 0 <= channel < 3
    assert 0 <= row < 4
    assert 0 <= col < 4
    assert f"channel {channel + 1}" in str(anchor.text)


@pytest.mark.parametrize("family", ["region_mean_relation", "region_range_relation"])
def test_region_probe_parameters_are_distinct_and_in_bounds(family: str) -> None:
    tokenizer = CharacterTokenizer()
    anchor = build_numeric_probe_anchor(
        tokenizer=tokenizer,
        patch_size=6,
        channel_count=2,
        families=[family],
        region_size=2,
        probe_index=3,
        seed=11,
        max_anchor_tokens=256,
    )
    channel, row_a, col_a, row_b, col_b, size = anchor.probe_parameters

    assert 0 <= channel < 2
    assert (row_a, col_a) != (row_b, col_b)
    assert size == 2
    assert 0 <= row_a <= 6 - size and 0 <= row_b <= 6 - size
    assert 0 <= col_a <= 6 - size and 0 <= col_b <= 6 - size
