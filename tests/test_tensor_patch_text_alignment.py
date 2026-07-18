from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from scripts.scan_tensor_teacher_layers import probe_target_and_control_perturbations
from scripts.train_tensor_patch_text_alignment import (
    AlignmentAnchor,
    AlignmentProjectionPair,
    FixedTeacherWhitening,
    apply_alignment_feature_transform,
    build_numeric_probe_anchor,
    build_static_alignment_anchor,
    build_teacher_texts_for_batch,
    checkpoint_selection_value,
    duplicate_text_fraction,
    gather_with_grad,
    hidden_at_last_non_padding,
    normalize_alignment_embeddings,
    probe_targets_from_patches,
    probe_contract_anchors,
    reject_removed_alignment_options,
    serialize_tensor_values,
    shared_suffix_token_ids,
    symmetric_contrastive_loss,
    tokenize_contents_with_anchor,
    tokenize_contents_with_shared_suffix,
    validate_probe_anchor_contract,
    validate_teacher_probe_preflight,
    validate_teacher_hidden_state_index,
    validate_teacher_tensor_source,
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


def test_teacher_text_uses_normalized_patch_when_configured() -> None:
    raw = torch.tensor([[[[10.0, 20.0]]]])
    normalized = torch.tensor([[[[-1.0, 1.0]]]])
    args = SimpleNamespace(
        teacher_text_source="normalized",
        alignment_text_layout="values_shared_suffix",
        text_decimal_places=2,
    )

    texts = build_teacher_texts_for_batch({"patch": raw}, normalized, args)

    assert texts == ["[[[-1.00, 1.00]]]"]


def test_duplicate_text_fraction_reports_false_negative_risk() -> None:
    assert duplicate_text_fraction([]) == 0.0
    assert duplicate_text_fraction(["a", "b", "c"]) == 0.0
    assert duplicate_text_fraction(["a", "a", "b", "b"]) == 0.5


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


def test_normalized_tensor_path_requires_normalized_teacher_text() -> None:
    validate_teacher_tensor_source({"mode": "none"}, "raw")
    validate_teacher_tensor_source({"mode": "none"}, "normalized")
    validate_teacher_tensor_source({"mode": "zscore"}, "normalized")
    validate_teacher_tensor_source({"mode": "none", "clip_min": -1.0}, "normalized")

    with pytest.raises(ValueError, match="teacher_text_source=normalized"):
        validate_teacher_tensor_source({"mode": "zscore"}, "raw")
    with pytest.raises(ValueError, match="teacher_text_source=normalized"):
        validate_teacher_tensor_source({"mode": "none", "clip_min": -1.0}, "raw")


def test_alignment_projection_pair_has_separate_trainable_heads() -> None:
    projector = AlignmentProjectionPair(
        input_dim=8,
        output_dim=4,
        hidden_dim=16,
        layers=1,
        dropout=0.0,
        shared=False,
    )
    student = torch.randn(3, 8, requires_grad=True)
    teacher = torch.randn(3, 8)

    student_projected, teacher_projected = projector(student, teacher)
    (student_projected.square().mean() + teacher_projected.square().mean()).backward()

    assert student_projected.shape == (3, 4)
    assert teacher_projected.shape == (3, 4)
    assert projector.student is not projector.teacher
    assert all(parameter.grad is not None for parameter in projector.parameters())


def test_primary_alignment_only_l2_normalizes_raw_hidden() -> None:
    student = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
    teacher = torch.tensor([[0.0, 5.0], [6.0, 8.0]])

    student_embedding, teacher_embedding = normalize_alignment_embeddings(student, teacher)

    assert torch.allclose(student_embedding, torch.nn.functional.normalize(student, dim=-1))
    assert torch.allclose(teacher_embedding, torch.nn.functional.normalize(teacher, dim=-1))


def test_teacher_whitening_uses_one_fixed_transform_for_both_branches() -> None:
    generator = torch.Generator().manual_seed(7)
    teacher = torch.randn(128, 4, generator=generator)
    teacher[:, 1] = 3.0 * teacher[:, 0] + 0.2 * teacher[:, 1]
    student = teacher + 0.05 * torch.randn(128, 4, generator=generator)
    whitener = FixedTeacherWhitening(hidden_dim=4, shrinkage=0.01, epsilon=1.0e-5)

    metrics = whitener.fit(teacher)
    student_features, teacher_features = apply_alignment_feature_transform(whitener, student, teacher)

    assert metrics["records"] == 128.0
    assert whitener.is_fitted
    assert torch.allclose(
        student_features - teacher_features,
        (student - teacher) @ whitener.matrix,
        atol=1.0e-5,
    )
    assert not any(parameter.requires_grad for parameter in whitener.parameters())


def test_teacher_pca_whitening_discards_low_variance_directions() -> None:
    generator = torch.Generator().manual_seed(9)
    teacher = torch.randn(512, 6, generator=generator)
    teacher[:, 2:] *= 0.01
    whitener = FixedTeacherWhitening(
        hidden_dim=6,
        output_dim=2,
        shrinkage=0.01,
        epsilon=1.0e-5,
    )

    metrics = whitener.fit(teacher)
    transformed = whitener.transform(teacher)

    assert transformed.shape == (512, 2)
    assert whitener.matrix.shape == (6, 2)
    assert metrics["output_dim"] == 2.0
    assert metrics["explained_variance_ratio"] > 0.99
    assert torch.allclose(
        transformed.mean(dim=0),
        torch.zeros(2),
        atol=1.0e-4,
    )


def test_teacher_pca_whitening_can_fit_within_anchor_covariance() -> None:
    anchor_a = torch.randn(64, 4, generator=torch.Generator().manual_seed(20)) + 100.0
    anchor_b = torch.randn(64, 4, generator=torch.Generator().manual_seed(21)) - 100.0
    teacher = torch.cat([anchor_a, anchor_b], dim=0)
    residuals = torch.cat(
        [anchor_a - anchor_a.mean(dim=0), anchor_b - anchor_b.mean(dim=0)],
        dim=0,
    )
    whitener = FixedTeacherWhitening(hidden_dim=4, output_dim=2, shrinkage=0.01, epsilon=1.0e-5)

    metrics = whitener.fit(teacher, covariance_residuals=residuals)

    assert metrics["within_anchor_covariance"] == 1.0
    assert metrics["output_dim"] == 2.0


def test_teacher_whitening_state_dict_restores_fitted_statistics() -> None:
    teacher = torch.randn(64, 3, generator=torch.Generator().manual_seed(11))
    fitted = FixedTeacherWhitening(hidden_dim=3, shrinkage=0.01, epsilon=1.0e-5)
    fitted.fit(teacher)
    restored = FixedTeacherWhitening(hidden_dim=3, shrinkage=0.01, epsilon=1.0e-5)

    restored.load_state_dict(fitted.state_dict())

    assert restored.is_fitted
    assert torch.allclose(restored.transform(teacher), fitted.transform(teacher))


def test_single_process_embedding_gather_preserves_gradients() -> None:
    embeddings = torch.tensor([[1.0, 2.0]], requires_grad=True)

    gathered = gather_with_grad(embeddings)
    gathered.square().sum().backward()

    assert torch.equal(embeddings.grad, torch.tensor([[2.0, 4.0]]))


@pytest.mark.parametrize(
    "option",
    [
        "center_embeddings",
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


def test_probe_is_deterministic_and_contains_no_answer_or_choice_metadata() -> None:
    tokenizer = CharacterTokenizer()
    kwargs = {
        "tokenizer": tokenizer,
        "patch_size": 4,
        "channel_count": 1,
        "families": ["point_difference"],
        "region_size": 2,
        "probe_index": 7,
        "seed": 42,
        "max_anchor_tokens": 256,
    }

    anchor = build_numeric_probe_anchor(**kwargs)
    same_anchor = build_numeric_probe_anchor(**kwargs)

    assert anchor == same_anchor
    assert anchor.mode == "probe"
    assert anchor.probe_family == "point_difference"
    assert len(anchor.probe_parameters) == 5
    assert not any(marker in str(anchor.text) for marker in ("Answer", "A or B", "Choose", "?"))
    assert str(anchor.text).startswith("\n")
    assert str(anchor.text).endswith(" is")


@pytest.mark.parametrize(
    "family",
    [
        "point_value",
        "point_difference",
        "point_mean",
        "region_mean",
        "region_range",
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
        families=["point_value"],
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
            "point_value",
            "point_difference",
            "point_mean",
            "region_mean",
            "region_range",
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
        families=["point_value"],
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


@pytest.mark.parametrize("family", ["region_mean", "region_range"])
def test_region_probe_parameters_are_in_bounds(family: str) -> None:
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
    channel, row, col, size = anchor.probe_parameters

    assert 0 <= channel < 2
    assert size == 2
    assert 0 <= row <= 6 - size
    assert 0 <= col <= 6 - size


@pytest.mark.parametrize(
    ("family", "parameters", "expected"),
    [
        ("point_value", (0, 0, 1), 2.0),
        ("point_difference", (0, 1, 1, 0, 0), 5.0),
        ("point_mean", (0, 0, 0, 0, 2), 2.0),
        ("region_mean", (0, 0, 0, 2), 3.25),
        ("region_range", (0, 0, 0, 2), 5.0),
    ],
)
def test_probe_targets_match_visible_numeric_operation(
    family: str,
    parameters: tuple[int, ...],
    expected: float,
) -> None:
    patch = torch.tensor(
        [[[[1.0, 2.0, 3.0], [4.0, 6.0, 8.0], [10.0, 12.0, 14.0]]]]
    )
    anchor = AlignmentAnchor(
        name="test",
        mode="probe",
        token_ids=(1,),
        text="\nTest value is",
        probe_family=family,
        probe_template_index=0,
        probe_parameters=parameters,
    )

    values, target_ids = probe_targets_from_patches(anchor, patch, decimal_places=2)

    assert values.tolist() == pytest.approx([expected])
    assert target_ids.tolist() == [round(expected * 100)]


def test_semantic_collisions_are_excluded_from_loss_but_strict_retrieval_is_preserved() -> None:
    tensor_embedding = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    text_embedding = tensor_embedding.clone()
    target_ids = torch.tensor([10, 10, 20])

    loss, metrics = symmetric_contrastive_loss(
        tensor_embedding,
        text_embedding,
        temperature=1.0,
        semantic_target_ids=target_ids,
    )

    assert float(loss.item()) < metrics["strict_contrastive_loss"]
    assert metrics["semantic_collision_fraction"] > 0.0
    assert metrics["semantic_i2t_accuracy"] > metrics["i2t_accuracy"]
    assert metrics["semantic_t2i_accuracy"] > metrics["t2i_accuracy"]


def test_checkpoint_selection_uses_strict_transformed_and_native_losses() -> None:
    args = SimpleNamespace(
        centered_contrastive_loss_weight=0.1,
        native_centered_contrastive_loss_weight=0.1,
    )
    metrics = {
        "global_strict_contrastive_loss": 2.0,
        "global_centered_strict_contrastive_loss": 3.0,
        "global_hidden_centered_strict_contrastive_loss": 5.0,
    }

    name, value = checkpoint_selection_value(metrics, args)

    assert name == (
        "global_strict_contrastive_loss+0.1*global_centered_strict_contrastive_loss"
        "+0.1*global_hidden_centered_strict_contrastive_loss"
    )
    assert value == pytest.approx(2.8)


def test_teacher_probe_preflight_rejects_unsupported_enabled_probe() -> None:
    preflight = {
        "anchors": {
            "point_value": {"hidden_similarity_vs_negative_target_distance_pearson": 0.4},
            "region_range": {"hidden_similarity_vs_negative_target_distance_pearson": 0.02},
        }
    }

    with pytest.raises(ValueError, match="region_range=0.0200"):
        validate_teacher_probe_preflight(preflight, minimum_correlation=0.1)


def test_layer_scan_control_changes_equal_number_of_off_target_values() -> None:
    patch = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    anchor = AlignmentAnchor(
        name="difference",
        mode="probe",
        token_ids=(1,),
        text="\nThe difference is",
        probe_family="point_difference",
        probe_template_index=0,
        probe_parameters=(0, 1, 1, 2, 2),
    )

    target_perturbed, control_perturbed = probe_target_and_control_perturbations(
        patch,
        anchor,
        scale=0.1,
    )
    original_target, _ = probe_targets_from_patches(anchor, patch, decimal_places=4)
    changed_target, _ = probe_targets_from_patches(anchor, target_perturbed, decimal_places=4)
    control_target, _ = probe_targets_from_patches(anchor, control_perturbed, decimal_places=4)

    assert int(target_perturbed.ne(patch).sum().item()) == 2
    assert int(control_perturbed.ne(patch).sum().item()) == 2
    assert not torch.equal(changed_target, original_target)
    assert torch.equal(control_target, original_target)
