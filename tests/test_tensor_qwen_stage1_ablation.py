from __future__ import annotations

import shutil
from pathlib import Path

import torch

from scripts.train_tensor_patch_text_alignment import TensorPatchAlignmentAdapter
from scripts.train_tensor_qwen_stage1_ablation import (
    ABLATION_MODE,
    _build_ablation_contract,
    _override_cli_tokens,
    _random_spatial_state,
    _state_digest,
    _validate_ablation_contract,
)


def _stage1_payload() -> dict[str, object]:
    torch.manual_seed(7)
    adapter = TensorPatchAlignmentAdapter(
        latent_channels=2,
        latent_grid=(2, 2),
        adapter_dim=8,
        projection_dim=12,
        dropout=0.0,
        adapter_type="spatial_transformer",
        query_tokens=4,
        adapter_layers=1,
        adapter_heads=2,
        soft_prompt_scale=0.05,
    )
    return {
        "checkpoint_type": "tensor_patch_text_alignment",
        "checkpoint_version": 3,
        "checkpoint_phase": "alignment",
        "adapter_state_dict": adapter.state_dict(),
        "args": {
            "adapter_layers": 1,
            "adapter_heads": 2,
            "dropout": 0.0,
            "soft_prompt_scale": 0.05,
            "train_patch_ae": False,
            "freeze_patch_ae_after_pretrain": False,
        },
    }


def test_random_stage1_state_is_deterministic_and_architecture_identical() -> None:
    payload = _stage1_payload()
    original = payload["adapter_state_dict"]
    first = _random_spatial_state(payload, seed=42)
    second = _random_spatial_state(payload, seed=42)
    other = _random_spatial_state(payload, seed=43)

    assert set(first) == set(original)
    assert _state_digest(first) == _state_digest(second)
    assert _state_digest(first) != _state_digest(other)
    assert _state_digest(first) != _state_digest(original)
    for name, value in first.items():
        assert value.shape == original[name].shape
        assert value.dtype == original[name].dtype


def test_override_tokens_put_boolean_and_sequence_values_on_real_cli() -> None:
    tokens = _override_cli_tokens(
        {
            "batch_size": 3,
            "evaluate_test": False,
            "diagnostics_enabled": True,
            "stage2b_resume_checkpoint": None,
            "eval_baselines": ["correct", "shuffled"],
        }
    )
    assert tokens == [
        "--batch-size",
        "3",
        "--no-evaluate-test",
        "--diagnostics-enabled",
        "--stage2b-resume-checkpoint",
        "none",
        "--eval-baselines",
        "correct,shuffled",
    ]


def test_ablation_contract_recomputes_random_initializer(tmp_path: Path) -> None:
    source = tmp_path / "alignment_best.pt"
    payload = _stage1_payload()
    torch.save(payload, source)
    random_state = _random_spatial_state(payload, seed=42)
    contract = _build_ablation_contract(source, payload, random_state, seed=42)

    observed = _validate_ablation_contract(contract, source)

    assert observed["mode"] == ABLATION_MODE
    assert observed["random_adapter_state_sha256"] == _state_digest(random_state)


def test_ablation_contract_rejects_changed_source_checkpoint(tmp_path: Path) -> None:
    source = tmp_path / "alignment_best.pt"
    payload = _stage1_payload()
    torch.save(payload, source)
    random_state = _random_spatial_state(payload, seed=42)
    contract = _build_ablation_contract(source, payload, random_state, seed=42)
    payload["extra"] = "changed"
    torch.save(payload, source)

    try:
        _validate_ablation_contract(contract, source)
    except ValueError as error:
        assert "changed after the ablation direct phase" in str(error)
    else:
        raise AssertionError("Changed Stage-1 checkpoint was accepted.")


def test_ablation_contract_accepts_same_checkpoint_after_path_migration(
    tmp_path: Path,
) -> None:
    source = tmp_path / "old_mount" / "alignment_best.pt"
    source.parent.mkdir()
    payload = _stage1_payload()
    torch.save(payload, source)
    random_state = _random_spatial_state(payload, seed=42)
    contract = _build_ablation_contract(source, payload, random_state, seed=42)
    migrated = tmp_path / "new_mount" / "alignment_best.pt"
    migrated.parent.mkdir()
    shutil.copyfile(source, migrated)

    observed = _validate_ablation_contract(contract, migrated)

    assert observed["source_stage1_checkpoint"] == str(source)


def test_ablation_contract_rejects_encoder_updated_by_alignment(tmp_path: Path) -> None:
    source = tmp_path / "alignment_best.pt"
    payload = _stage1_payload()
    payload["args"]["train_patch_ae"] = True
    payload["args"]["freeze_patch_ae_after_pretrain"] = False
    torch.save(payload, source)
    random_state = _random_spatial_state(payload, seed=42)

    try:
        _build_ablation_contract(source, payload, random_state, seed=42)
    except ValueError as error:
        assert "encoder that was updated" in str(error)
    else:
        raise AssertionError("Joint encoder-and-adapter Stage 1 was accepted.")
