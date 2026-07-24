from __future__ import annotations

import pickle
import sys
from dataclasses import replace
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch
from torch.utils.checkpoint import checkpoint

from scripts.scan_tensor_teacher_layers import probe_target_and_control_perturbations, resolve_scan_args
from scripts.train_tensor_patch_text_alignment import (
    AlignmentAnchor,
    AlignmentProjectionPair,
    DistributedEvalSampler,
    FixedTeacherWhitening,
    PDEBenchPatchTextDataset,
    PROBE_TEMPLATE_COUNTS,
    PatchRecord,
    TensorPatchAlignmentAdapter,
    alignment_anchors_from_args,
    apply_config_defaults,
    audit_optimizer_parameter_coverage,
    auxiliary_teacher_alignment_loss,
    apply_alignment_feature_transform,
    branch_mean_alignment_loss,
    build_numeric_probe_anchor,
    build_static_alignment_anchor,
    build_teacher_texts_for_batch,
    checkpoint_selection_value,
    duplicate_text_fraction,
    forward_teacher_readout_hidden,
    forward_teacher_readout_hiddens,
    gather_with_grad,
    gradient_parameter_entries,
    hidden_at_last_non_padding,
    normalize_patch_batch,
    normalize_alignment_embeddings,
    normalize_teacher_layer_indices,
    parse_args,
    probe_targets_from_patches,
    probe_contract_anchors,
    reconstruction_loss_with_diagnostics,
    reject_removed_alignment_options,
    resolve_teacher_layer_supervision,
    serialize_tensor_values,
    set_frozen_llm_student_mode,
    shared_suffix_token_ids,
    stable_name_fingerprint,
    symmetric_contrastive_loss,
    teacher_supervision_metadata,
    top1_candidate_usage_metrics,
    truncate_llm_backbone_to_layer,
    tokenize_contents_with_anchor,
    tokenize_contents_with_shared_suffix,
    teacher_probe_preflight_warnings,
    transformer_block_hidden_states,
    validate_probe_anchor_contract,
    validate_finite_float,
    validate_teacher_hidden_state_index,
    validate_teacher_tensor_source,
)
from tensor_compression.data.normalization import normalize_tensor


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


class TinyBackboneModel(torch.nn.Module):
    def __init__(self, block_count: int = 4) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList(
            [torch.nn.Linear(2, 2, bias=False) for _ in range(block_count)]
        )


@pytest.mark.parametrize(
    ("mode", "scope"),
    [
        ("none", "global"),
        ("minmax", "global"),
        ("minmax", "channel"),
        ("zscore", "global"),
        ("zscore", "channel"),
    ],
)
def test_vectorized_patch_normalization_matches_per_record_reference(mode: str, scope: str) -> None:
    generator = torch.Generator().manual_seed(17)
    patches = torch.randn(4, 2, 3, 5, generator=generator)
    patches[0, 0].fill_(0.25)
    config = {
        "mode": mode,
        "scope": scope,
        "clip_min": -1.5,
        "clip_max": 1.5,
    }
    expected = torch.stack([normalize_tensor(patch, config)[0] for patch in patches], dim=0)

    actual = normalize_patch_batch(
        patches,
        input_size=(3, 5),
        normalization_cfg=config,
        resize_to_input=False,
    )

    torch.testing.assert_close(actual, expected, rtol=1.0e-6, atol=1.0e-6)


def test_patch_dataset_reuses_process_local_hdf5_handle_and_drops_it_when_pickled(tmp_path) -> None:
    hdf5_path = tmp_path / "patches.hdf5"
    vx = np.arange(2 * 2 * 4 * 4, dtype=np.float32).reshape(2, 2, 4, 4)
    vy = vx + 1000.0
    with h5py.File(hdf5_path, "w") as handle:
        handle.create_dataset("Vx", data=vx)
        handle.create_dataset("Vy", data=vy)

    dataset = PDEBenchPatchTextDataset(
        hdf5_path=hdf5_path,
        field_keys=["Vx", "Vy"],
        records=[
            PatchRecord(sample_index=0, time_index=0, row=0, col=0, field_key="Vx"),
            PatchRecord(sample_index=1, time_index=1, row=2, col=2, field_key="Vy"),
        ],
        patch_size=2,
        decimal_places=3,
        prompt_template="compact",
        include_raw_text=False,
    )

    first = dataset[0]
    first_handle = dataset._hdf5_handle
    second = dataset[1]

    assert first_handle is not None
    assert dataset._hdf5_handle is first_handle
    torch.testing.assert_close(first["patch"][0], torch.from_numpy(vx[0, 0, :2, :2]))
    torch.testing.assert_close(second["patch"][0], torch.from_numpy(vy[1, 1, 2:4, 2:4]))

    restored = pickle.loads(pickle.dumps(dataset))
    assert restored._hdf5_handle is None
    restored_item = restored[0]
    assert restored._hdf5_handle is not None
    torch.testing.assert_close(restored_item["patch"], first["patch"])

    dataset.close()
    restored.close()


def test_reconstruction_relative_metric_uses_aggregate_variance() -> None:
    class ZeroDecoder(torch.nn.Module):
        def decode(self, payload):
            return torch.zeros_like(payload["latent_map"])

    varying = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)
    target = torch.stack([torch.ones_like(varying), varying], dim=0)

    _loss, metrics = reconstruction_loss_with_diagnostics(
        ZeroDecoder(),
        latent_map=target,
        target=target,
    )

    assert metrics["relative_rmse_to_target_std"] == pytest.approx(
        metrics["rmse"] / metrics["target_std"]
    )
    assert (
        metrics["mean_record_relative_rmse_to_target_std"]
        > metrics["relative_rmse_to_target_std"]
    )


class _AddBlock(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = float(value)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor]:
        return (hidden + self.value,)


class _HookBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_AddBlock(1.0), _AddBlock(2.0)])

    def forward(
        self,
        *,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        output_hidden_states=False,
        use_cache=False,
    ):
        hidden = inputs_embeds
        for layer in self.layers:
            hidden = layer(hidden)[0]
        return SimpleNamespace(last_hidden_state=hidden * 10.0)


class _HookModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _HookBackbone()


class _CheckpointHookBackbone(_HookBackbone):
    def __init__(self, use_reentrant: bool) -> None:
        super().__init__()
        self.use_reentrant = bool(use_reentrant)

    def forward(
        self,
        *,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        output_hidden_states=False,
        use_cache=False,
    ):
        hidden = inputs_embeds
        for layer in self.layers:
            hidden = checkpoint(
                lambda value, current_layer=layer: current_layer(value)[0],
                hidden,
                use_reentrant=self.use_reentrant,
            )
        return SimpleNamespace(last_hidden_state=hidden)


class _CheckpointHookModel(torch.nn.Module):
    def __init__(self, use_reentrant: bool) -> None:
        super().__init__()
        self.model = _CheckpointHookBackbone(use_reentrant=use_reentrant)


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


def test_shallow_teacher_truncation_keeps_requested_prefix_of_blocks() -> None:
    model = TinyBackboneModel(block_count=4)

    assert truncate_llm_backbone_to_layer(model, teacher_layer=2) == 2
    assert len(model.model.layers) == 2


def test_transformer_block_capture_precedes_final_backbone_norm() -> None:
    model = _HookModel()
    inputs = torch.zeros(1, 3, 2)

    captured = transformer_block_hidden_states(
        model,
        inputs_embeds=inputs,
        attention_mask=torch.ones(1, 3, dtype=torch.long),
        layer_indices=[1, 2],
    )

    assert torch.equal(captured[1], torch.ones_like(inputs))
    assert torch.equal(captured[2], torch.full_like(inputs, 3.0))


def test_multi_layer_readout_uses_each_rows_final_nonpadding_token_and_keeps_gradients() -> None:
    model = _HookModel()
    inputs = torch.arange(2 * 4 * 2, dtype=torch.float32).reshape(2, 4, 2).requires_grad_()
    attention_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])

    captured = forward_teacher_readout_hiddens(
        model,
        inputs_embeds=inputs,
        attention_mask=attention_mask,
        teacher_layers=[1, 2],
    )
    (captured[1].sum() + captured[2].sum()).backward()

    assert torch.equal(
        captured[1],
        torch.stack([inputs.detach()[0, 1] + 1.0, inputs.detach()[1, 2] + 1.0]),
    )
    assert torch.equal(
        captured[2],
        torch.stack([inputs.detach()[0, 1] + 3.0, inputs.detach()[1, 2] + 3.0]),
    )
    assert inputs.grad is not None
    assert torch.equal(inputs.grad[0, 1], torch.full((2,), 2.0))
    assert torch.equal(inputs.grad[1, 2], torch.full((2,), 2.0))


def test_single_layer_readout_prefix_argument_validates_full_mask_without_double_offset() -> None:
    model = _HookModel()
    inputs = torch.arange(2 * 5 * 2, dtype=torch.float32).reshape(2, 5, 2)
    attention_mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])

    captured = forward_teacher_readout_hidden(
        model,
        inputs_embeds=inputs,
        attention_mask=attention_mask,
        teacher_layer=1,
        prefix_tokens=2,
    )

    assert torch.equal(
        captured,
        torch.stack([inputs[0, 3] + 1.0, inputs[1, 4] + 1.0]),
    )
    with pytest.raises(ValueError, match="soft-prefix position"):
        forward_teacher_readout_hidden(
            model,
            inputs_embeds=inputs,
            attention_mask=torch.tensor([[1, 0, 1, 1, 0], [1, 1, 1, 1, 1]]),
            teacher_layer=1,
            prefix_tokens=2,
        )


def test_non_reentrant_checkpointed_readout_keeps_student_gradient_graph() -> None:
    model = _CheckpointHookModel(use_reentrant=False)
    inputs = torch.randn(2, 3, 2, requires_grad=True)

    captured = forward_teacher_readout_hiddens(
        model,
        inputs_embeds=inputs,
        attention_mask=torch.ones(2, 3, dtype=torch.long),
        teacher_layers=[1, 2],
    )
    for hidden in captured.values():
        hidden.retain_grad()
    (captured[1].sum() + captured[2].sum()).backward()

    assert inputs.grad is not None
    assert torch.count_nonzero(inputs.grad) > 0
    assert all(hidden.grad is not None for hidden in captured.values())
    assert all(torch.count_nonzero(hidden.grad) > 0 for hidden in captured.values())


def test_multi_layer_readout_rejects_non_finite_hidden_before_loss() -> None:
    model = _HookModel()
    inputs = torch.zeros(2, 3, 2)
    inputs[0, 2, 0] = float("nan")

    with pytest.raises(FloatingPointError, match=r"layers \[1, 2\]"):
        forward_teacher_readout_hiddens(
            model,
            inputs_embeds=inputs,
            attention_mask=torch.ones(2, 3, dtype=torch.long),
            teacher_layers=[1, 2],
        )


def test_reentrant_checkpointed_readout_is_rejected_before_training() -> None:
    model = _CheckpointHookModel(use_reentrant=True)
    inputs = torch.randn(2, 3, 2, requires_grad=True)

    with pytest.raises(RuntimeError, match="lost the student gradient graph"):
        forward_teacher_readout_hiddens(
            model,
            inputs_embeds=inputs,
            attention_mask=torch.ones(2, 3, dtype=torch.long),
            teacher_layers=[1, 2],
        )


def test_auxiliary_teacher_layer_spec_is_explicit_and_rejects_ambiguous_inputs() -> None:
    layers, weights = resolve_teacher_layer_supervision(2, [4, 6], [0.25, 0.5])

    assert layers == (2, 4, 6)
    assert weights == {4: 0.25, 6: 0.5}
    with pytest.raises(ValueError, match="must not repeat"):
        resolve_teacher_layer_supervision(2, [2], [0.25])
    with pytest.raises(ValueError, match="exactly one weight"):
        resolve_teacher_layer_supervision(2, [4, 6], [0.25])
    assert resolve_teacher_layer_supervision(4, [6, 2], [0.25, 0.5]) == (
        (4, 2, 6),
        {2: 0.5, 6: 0.25},
    )
    with pytest.raises(ValueError, match="finite and positive"):
        resolve_teacher_layer_supervision(2, [4], [float("nan")])


def test_config_defaults_keep_primary_out_of_lower_auxiliary_layers(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["train_tensor_patch_text_alignment.py", "--config", "unused.yaml"])
    args = parse_args()
    resolved = apply_config_defaults(
        args,
        {
            "data": {"hdf5_path": "input.h5"},
            "model": {"name_or_path": "test-model"},
            "patch_alignment": {
                "output_root": "outputs",
                "teacher_layer": 4,
                "auxiliary_teacher_layers": [6, 2],
                "auxiliary_teacher_layer_weights": [0.25, 0.5],
            },
        },
    )

    assert resolved.teacher_layers == (4, 2, 6)
    assert resolved.auxiliary_teacher_layers == [2, 6]
    assert resolved.auxiliary_teacher_layer_weights == [0.5, 0.25]
    assert resolved.auxiliary_teacher_layer_weights_by_layer == {2: 0.5, 6: 0.25}


def test_teacher_layer_indices_reject_embedding_layer_and_duplicates() -> None:
    assert normalize_teacher_layer_indices([6, 2, 4]) == (2, 4, 6)
    with pytest.raises(ValueError, match="Index 0"):
        normalize_teacher_layer_indices([0, 2])
    with pytest.raises(ValueError, match="duplicate"):
        normalize_teacher_layer_indices([2, 2])


def test_frozen_llm_checkpoint_mode_keeps_stochastic_children_in_eval_mode() -> None:
    model = _HookModel().train()

    set_frozen_llm_student_mode(model, gradient_checkpointing=True)

    assert model.training is False
    assert model.model.training is True
    assert all(layer.training is False for layer in model.model.layers)
    model.eval()
    assert model.model.training is False
    set_frozen_llm_student_mode(model, gradient_checkpointing=True)
    assert model.model.training is True
    assert all(layer.training is False for layer in model.model.layers)
    set_frozen_llm_student_mode(model, gradient_checkpointing=False)
    assert model.training is False
    assert model.model.training is False


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_finite_configuration_scalar_rejects_non_finite_values(invalid: float) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        validate_finite_float("setting", invalid, minimum=0.0)


def test_spatial_adapter_rejects_invalid_numeric_construction_before_torch_modules() -> None:
    valid = {
        "latent_channels": 2,
        "latent_grid": (2, 2),
        "adapter_dim": 8,
        "projection_dim": 12,
        "dropout": 0.0,
        "adapter_type": "spatial_transformer",
        "query_tokens": 4,
        "adapter_layers": 1,
        "adapter_heads": 2,
        "soft_prompt_scale": 0.05,
    }

    with pytest.raises(ValueError, match="dropout"):
        TensorPatchAlignmentAdapter(**{**valid, "dropout": float("nan")})
    with pytest.raises(ValueError, match="soft_prompt_scale"):
        TensorPatchAlignmentAdapter(**{**valid, "soft_prompt_scale": float("inf")})
    with pytest.raises(ValueError, match="adapter_layers and adapter_heads"):
        TensorPatchAlignmentAdapter(**{**valid, "adapter_heads": 0})


def test_teacher_supervision_metadata_is_complete_and_canonical() -> None:
    args = SimpleNamespace(
        teacher_layer=4,
        teacher_layers=(4, 2, 6),
        auxiliary_teacher_layer_weights_by_layer={6: 0.25, 2: 0.5},
        alignment_transform_mode="whitening",
    )

    metadata = teacher_supervision_metadata(args)

    assert metadata == {
        "primary_layer": 4,
        "layers": [2, 4, 6],
        "auxiliary_layers": [2, 6],
        "auxiliary_layer_weights": {"2": 0.5, "6": 0.25},
        "primary_feature_transform": "whitening",
        "auxiliary_feature_transform": "native_centered_and_branch_mean",
    }
    with pytest.raises(ValueError, match="metadata is inconsistent"):
        teacher_supervision_metadata(
            SimpleNamespace(
                teacher_layer=2,
                teacher_layers=(2, 4),
                auxiliary_teacher_layer_weights_by_layer={},
                alignment_transform_mode="none",
            )
        )


def test_auxiliary_teacher_loss_backpropagates_from_every_configured_layer() -> None:
    generator = torch.Generator().manual_seed(33)
    student = {
        2: torch.randn(4, 8, generator=generator, requires_grad=True),
        4: torch.randn(4, 8, generator=generator, requires_grad=True),
        6: torch.randn(4, 8, generator=generator, requires_grad=True),
    }
    teacher = {layer: torch.randn(4, 8, generator=generator) for layer in student}

    loss, metrics = auxiliary_teacher_alignment_loss(
        student,
        teacher,
        {4: 0.25, 6: 0.5},
        semantic_target_ids=None,
        temperature=0.07,
        i2t_weight=0.6,
        t2i_weight=0.4,
        native_centered_weight=0.5,
        mean_alignment_weight=0.5,
        distributed_batch=False,
    )
    loss.backward()

    assert float(loss.item()) > 0.0
    assert metrics["layer_count"] == 2.0
    assert student[2].grad is None
    assert student[4].grad is not None and torch.count_nonzero(student[4].grad) > 0
    assert student[6].grad is not None and torch.count_nonzero(student[6].grad) > 0


def test_auxiliary_teacher_loss_rejects_shape_and_target_mismatches() -> None:
    student = {4: torch.randn(3, 8, requires_grad=True)}
    teacher = {4: torch.randn(3, 7)}
    kwargs = {
        "temperature": 0.07,
        "i2t_weight": 0.6,
        "t2i_weight": 0.4,
        "native_centered_weight": 0.5,
        "mean_alignment_weight": 0.5,
        "distributed_batch": False,
    }

    with pytest.raises(ValueError, match="shapes differ"):
        auxiliary_teacher_alignment_loss(
            student,
            teacher,
            {4: 0.25},
            semantic_target_ids=None,
            **kwargs,
        )
    with pytest.raises(ValueError, match="one value per local record"):
        auxiliary_teacher_alignment_loss(
            student,
            {4: torch.randn(3, 8)},
            {4: 0.25},
            semantic_target_ids=torch.tensor([1, 2]),
            **kwargs,
        )


def test_stage1_optimizer_audit_requires_exact_trainable_parameter_coverage() -> None:
    module = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    optimizer = torch.optim.AdamW(module.parameters(), lr=1.0e-3)

    metrics = audit_optimizer_parameter_coverage(optimizer, [module])

    assert metrics["missing_trainable_parameter_tensors"] == 0
    assert metrics["unexpected_parameter_tensors"] == 0
    incomplete = torch.optim.AdamW(module[0].parameters(), lr=1.0e-3)
    with pytest.raises(RuntimeError, match="missing_trainable"):
        audit_optimizer_parameter_coverage(incomplete, [module])
    outside = torch.nn.Parameter(torch.zeros(1))
    with pytest.raises(RuntimeError, match="unexpected"):
        audit_optimizer_parameter_coverage(
            torch.optim.AdamW([*module.parameters(), outside], lr=1.0e-3),
            [module],
        )


def test_distributed_eval_sampler_partitions_without_padding_duplicates() -> None:
    dataset = list(range(10))
    shards = [list(DistributedEvalSampler(dataset, num_replicas=3, rank=rank)) for rank in range(3)]

    assert shards == [[0, 3, 6, 9], [1, 4, 7], [2, 5, 8]]
    assert sorted(index for shard in shards for index in shard) == list(range(10))
    assert sum(len(shard) for shard in shards) == len(dataset)


def test_gradient_parameter_entries_are_stable_and_deduplicate_shared_modules() -> None:
    first = torch.nn.Linear(3, 2)
    second = torch.nn.Linear(2, 1, bias=False)
    entries = gradient_parameter_entries([first, second, first])
    names = [name for name, _parameter in entries]

    assert names == ["module_0.weight", "module_0.bias", "module_1.weight"]
    assert len({id(parameter) for _name, parameter in entries}) == len(entries)
    assert stable_name_fingerprint(names) == stable_name_fingerprint(list(names))
    assert stable_name_fingerprint(names) != stable_name_fingerprint(list(reversed(names)))


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


def test_teacher_whitening_rejects_non_finite_hyperparameters_and_samples() -> None:
    with pytest.raises(ValueError, match="epsilon.*finite"):
        FixedTeacherWhitening(hidden_dim=4, shrinkage=0.01, epsilon=float("nan"))
    whitener = FixedTeacherWhitening(hidden_dim=4, shrinkage=0.01, epsilon=1.0e-5)
    samples = torch.randn(8, 4)
    samples[0, 0] = float("inf")
    with pytest.raises(ValueError, match="contain NaN or infinity"):
        whitener.fit(samples)


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


def test_teacher_whitening_caps_regularized_condition_number() -> None:
    generator = torch.Generator().manual_seed(25)
    teacher = torch.randn(1024, 6, generator=generator)
    teacher[:, 1:] *= 1.0e-4
    whitener = FixedTeacherWhitening(
        hidden_dim=6,
        output_dim=6,
        shrinkage=0.0,
        epsilon=1.0e-8,
        max_condition_number=100.0,
    )

    metrics = whitener.fit(teacher)

    assert metrics["configured_max_condition_number"] == 100.0
    assert metrics["regularized_condition_number"] <= 100.0 + 1.0e-3


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


def test_branch_mean_alignment_matches_direction_and_norm_without_collapsing_rows() -> None:
    teacher = torch.tensor([[1.0, 0.0], [3.0, 2.0]])
    student = teacher + torch.tensor([[-1.0, 1.0], [-1.0, 1.0]])

    loss, metrics = branch_mean_alignment_loss(student, teacher)

    assert float(loss.item()) > 0.0
    assert metrics["cosine"] < 1.0
    assert metrics["l2_distance"] > 0.0
    assert torch.equal(student[1] - student[0], teacher[1] - teacher[0])


def test_top1_candidate_usage_exposes_candidate_hubness() -> None:
    metrics = top1_candidate_usage_metrics(torch.tensor([0, 0, 0, 1]), candidate_count=4)

    assert metrics["candidate_coverage"] == pytest.approx(0.5)
    assert metrics["max_candidate_hit_fraction"] == pytest.approx(0.75)
    assert 0.0 < metrics["candidate_hit_entropy"] < 1.0


@pytest.mark.parametrize(
    "option",
    [
        "center_embeddings",
        "cosine_loss_weight",
        "probe_answer_ce_weight",
        "probe_teacher_kl_weight",
        "probe_kl_temperature",
        "probe_teacher_preflight_records",
        "teacher_probe_min_correlation",
    ],
)
def test_removed_alignment_options_fail_before_training(option: str) -> None:
    with pytest.raises(ValueError, match=option):
        reject_removed_alignment_options({"patch_alignment": {option: False}})


def test_old_null_probe_gate_is_ignored_for_config_migration() -> None:
    reject_removed_alignment_options({"patch_alignment": {"teacher_probe_min_correlation": None}})


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
def test_probe_uses_all_distinct_natural_stems_without_choice_format(family: str) -> None:
    tokenizer = CharacterTokenizer()
    template_count = PROBE_TEMPLATE_COUNTS[family]
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
        for template_index in range(template_count)
    ]

    assert {anchor.probe_template_index for anchor in anchors} == set(range(template_count))
    assert len({anchor.token_ids for anchor in anchors}) == template_count
    for anchor in anchors:
        text = str(anchor.text)
        assert text.startswith("\n")
        assert text.endswith((" is", " equals", " gives", " contains"))
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

    expected_count = sum(PROBE_TEMPLATE_COUNTS[family] for family in args.probe_families)
    assert len(anchors) == expected_count
    assert len({(anchor.probe_family, anchor.probe_template_index) for anchor in anchors}) == expected_count
    for family in args.probe_families:
        family_anchors = [anchor for anchor in anchors if anchor.probe_family == family]
        assert len(family_anchors) == PROBE_TEMPLATE_COUNTS[family]
        assert len({anchor.token_ids for anchor in family_anchors}) == PROBE_TEMPLATE_COUNTS[family]
        assert len({anchor.probe_parameters for anchor in family_anchors}) == 1


def test_point_value_evaluation_cycles_all_eight_short_stems() -> None:
    tokenizer = CharacterTokenizer()
    args = SimpleNamespace(
        alignment_anchor_mode="probe",
        evaluation_probe_count=8,
        field_sampling_mode="single",
        fields=["Vx"],
        patch_size=16,
        probe_families=["point_value"],
        probe_region_size=4,
        seed=42,
        max_shared_suffix_tokens=256,
        representation_suffix="\nRepresentation:",
    )

    anchors = alignment_anchors_from_args(tokenizer, args, evaluation=True)

    assert len(anchors) == 8
    assert {anchor.probe_template_index for anchor in anchors} == set(range(8))
    assert len({anchor.token_ids for anchor in anchors}) == 8


@pytest.mark.parametrize("configured", ["point_value", ["point_value"], "point_value,point_mean"])
def test_layer_scan_parses_probe_family_scalars_and_lists(configured) -> None:
    cli = SimpleNamespace(
        anchor_mode=None,
        split=None,
        records=None,
        batch_size=None,
        layers=None,
        probe_count=None,
        perturbation_scale=None,
        device=None,
        output=None,
    )
    config = {
        "model": {"name_or_path": "unused-test-model"},
        "patch_alignment": {
            "hdf5_path": "unused-test-data.h5",
            "fields": ["Vx"],
            "probe_families": configured,
            "teacher_text_source": "raw",
            "patch_encoder": {"normalization": {"mode": "none"}},
        },
    }

    resolved = resolve_scan_args(cli, config)

    assert resolved.probe_families == (
        ["point_value", "point_mean"]
        if configured == "point_value,point_mean"
        else ["point_value"]
    )


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


def test_checkpoint_selection_is_deployment_directional_and_uses_nontransductive_auxiliaries() -> None:
    args = SimpleNamespace(
        contrastive_loss_weight=0.25,
        contrastive_i2t_weight=0.75,
        contrastive_t2i_weight=0.25,
        centered_contrastive_loss_weight=1.0,
        native_centered_contrastive_loss_weight=0.25,
        mean_alignment_loss_weight=0.1,
    )
    metrics = {
        "global_strict_i2t_loss": 2.0,
        "global_strict_t2i_loss": 4.0,
        "centered_strict_contrastive_loss": 3.0,
        "native_centered_strict_contrastive_loss": 5.0,
        "mean_alignment_loss": 0.5,
    }

    name, value = checkpoint_selection_value(metrics, args)

    assert name == (
        "0.25*(0.75*global_strict_i2t_loss+0.25*global_strict_t2i_loss)"
        "+1*centered_strict_contrastive_loss"
        "+0.25*native_centered_strict_contrastive_loss"
        "+0.1*mean_alignment_loss"
    )
    assert value == pytest.approx(4.925)


def test_checkpoint_selection_includes_configured_auxiliary_teacher_layers() -> None:
    args = SimpleNamespace(
        contrastive_loss_weight=1.0,
        contrastive_i2t_weight=0.6,
        contrastive_t2i_weight=0.4,
        centered_contrastive_loss_weight=0.0,
        native_centered_contrastive_loss_weight=0.0,
        mean_alignment_loss_weight=0.0,
        auxiliary_teacher_layers=[4, 6],
    )
    metrics = {
        "global_strict_i2t_loss": 2.0,
        "global_strict_t2i_loss": 4.0,
        "auxiliary_teacher_loss": 0.75,
    }

    name, value = checkpoint_selection_value(metrics, args)

    assert name.endswith("+auxiliary_teacher_loss")
    assert value == pytest.approx(0.6 * 2.0 + 0.4 * 4.0 + 0.75)


def test_teacher_probe_preflight_warning_uses_family_median_without_aborting() -> None:
    preflight = {
        "families": {
            "point_value": {
                "hidden_similarity_vs_negative_target_distance_pearson_median": 0.4,
            },
            "region_range": {
                "hidden_similarity_vs_negative_target_distance_pearson_median": 0.02,
            },
        }
    }

    assert teacher_probe_preflight_warnings(preflight, warn_below_correlation=0.1) == [
        "region_range=0.0200"
    ]


def test_teacher_probe_preflight_warning_can_be_disabled() -> None:
    preflight = {
        "families": {
            "point_value": {
                "hidden_similarity_vs_negative_target_distance_pearson_median": 0.0,
            },
        }
    }

    assert teacher_probe_preflight_warnings(preflight, warn_below_correlation=None) == []


def test_teacher_probe_preflight_warnings_identify_weak_auxiliary_layer() -> None:
    preflight = {
        "layers": {
            "2": {
                "families": {
                    "point_value": {
                        "hidden_similarity_vs_negative_target_distance_pearson_median": 0.2
                    }
                }
            },
            "6": {
                "families": {
                    "point_value": {
                        "hidden_similarity_vs_negative_target_distance_pearson_median": 0.03
                    }
                }
            },
        }
    }

    assert teacher_probe_preflight_warnings(preflight, warn_below_correlation=0.1) == [
        "layer_6/point_value=0.0300"
    ]


def test_contrastive_direction_weights_are_normalized_and_affect_loss() -> None:
    tensor_embedding = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    text_embedding = tensor_embedding.clone()

    primary, primary_metrics = symmetric_contrastive_loss(
        tensor_embedding,
        text_embedding,
        temperature=1.0,
        i2t_weight=3.0,
        t2i_weight=1.0,
    )
    equal, equal_metrics = symmetric_contrastive_loss(
        tensor_embedding,
        text_embedding,
        temperature=1.0,
    )

    assert primary_metrics["i2t_weight"] == pytest.approx(0.75)
    assert primary_metrics["t2i_weight"] == pytest.approx(0.25)
    assert primary_metrics["strict_contrastive_loss"] == pytest.approx(
        0.75 * primary_metrics["strict_i2t_loss"] + 0.25 * primary_metrics["strict_t2i_loss"]
    )
    assert float(primary.item()) == pytest.approx(float(equal.item()))
    assert equal_metrics["i2t_weight"] == pytest.approx(0.5)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_contrastive_direction_weights_reject_non_finite_values(invalid: float) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        symmetric_contrastive_loss(
            torch.eye(2),
            torch.eye(2),
            temperature=0.07,
            i2t_weight=invalid,
            t2i_weight=1.0,
        )


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
