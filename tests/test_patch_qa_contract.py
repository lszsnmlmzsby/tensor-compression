from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tensor_compression.downstream.patch_qa_contract import (
    PATCH_LATENT_FORMAT,
    validate_patch_latent_payload,
    validate_stage1_alignment_checkpoint_payload,
)


def _valid_cache(tmp_path: Path) -> tuple[dict, dict, Path, str, dict, dict]:
    checkpoint = tmp_path / "stage1.bin"
    checkpoint.write_bytes(b"stage1-checkpoint-v1")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    identity = {
        "patch_id": "Vx_s000001_t0002_r003_c004",
        "field": "Vx",
        "sample_index": 1,
        "time_index": 2,
        "top_left": [3, 4],
    }
    normalization = {
        "mode": "zscore",
        "scope": "channel",
        "stats_path": None,
        "clip_min": None,
        "clip_max": None,
    }
    qa_stats = {"mean": 1.5, "std": 2.5, "scale": 2.500001}
    payload = {
        "format": PATCH_LATENT_FORMAT,
        "latent_map": torch.zeros(2, 2, 2, dtype=torch.float16),
        **identity,
        "alignment_checkpoint": str(checkpoint.resolve()),
        "alignment_checkpoint_sha256": checkpoint_sha,
        "encoder_input_normalization": dict(normalization),
        "qa_value_space": {"mode": "per_patch_zscore", **qa_stats},
    }
    return payload, identity, checkpoint, checkpoint_sha, normalization, qa_stats


def test_exact_latent_contract_is_accepted(tmp_path: Path) -> None:
    payload, identity, checkpoint, checkpoint_sha, normalization, qa_stats = _valid_cache(tmp_path)

    latent = validate_patch_latent_payload(
        payload,
        path=tmp_path / "latent.pt",
        expected_identity=identity,
        expected_alignment_checkpoint=checkpoint,
        expected_alignment_sha256=checkpoint_sha,
        expected_normalization=normalization,
        expected_shape=(2, 2, 2),
        expected_storage_dtype="float16",
        expected_qa_stats=qa_stats,
    )

    assert tuple(latent.shape) == (2, 2, 2)
    assert latent.dtype == torch.float16


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload, _tmp: payload.update(format="stale"),
        lambda payload, _tmp: payload.update(patch_id="wrong"),
        lambda payload, _tmp: payload.update(alignment_checkpoint="other.pt"),
        lambda payload, _tmp: payload.update(alignment_checkpoint_sha256="0" * 64),
        lambda payload, _tmp: payload.update(encoder_input_normalization={"mode": "none"}),
        lambda payload, _tmp: payload.update(latent_map=torch.zeros(1, 2, 2, dtype=torch.float16)),
        lambda payload, _tmp: payload.update(latent_map=torch.zeros(2, 2, 2, dtype=torch.float32)),
        lambda payload, _tmp: payload["qa_value_space"].update(mean=9.0),
    ],
)
def test_latent_contract_rejects_provenance_or_storage_mutations(tmp_path: Path, mutation) -> None:
    payload, identity, checkpoint, checkpoint_sha, normalization, qa_stats = _valid_cache(tmp_path)
    mutated = copy.deepcopy(payload)
    mutation(mutated, tmp_path)

    with pytest.raises((ValueError, FloatingPointError)):
        validate_patch_latent_payload(
            mutated,
            path=tmp_path / "latent.pt",
            expected_identity=identity,
            expected_alignment_checkpoint=checkpoint,
            expected_alignment_sha256=checkpoint_sha,
            expected_normalization=normalization,
            expected_shape=(2, 2, 2),
            expected_storage_dtype="float16",
            expected_qa_stats=qa_stats,
        )


@pytest.mark.parametrize("bad_top_left", [1, [1], [1, "bad"], "1,2"])
def test_latent_contract_rejects_malformed_identity_coordinates(tmp_path: Path, bad_top_left) -> None:
    payload, identity, checkpoint, checkpoint_sha, normalization, qa_stats = _valid_cache(tmp_path)
    payload["top_left"] = bad_top_left

    with pytest.raises(ValueError, match="identity metadata"):
        validate_patch_latent_payload(
            payload,
            path=tmp_path / "latent.pt",
            expected_identity=identity,
            expected_alignment_checkpoint=checkpoint,
            expected_alignment_sha256=checkpoint_sha,
            expected_normalization=normalization,
            expected_shape=(2, 2, 2),
            expected_storage_dtype="float16",
            expected_qa_stats=qa_stats,
        )


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_latent_contract_rejects_non_finite_latents(tmp_path: Path, bad_value: float) -> None:
    payload, identity, checkpoint, checkpoint_sha, normalization, qa_stats = _valid_cache(tmp_path)
    payload["latent_map"][0, 0, 0] = bad_value

    with pytest.raises(FloatingPointError):
        validate_patch_latent_payload(
            payload,
            path=tmp_path / "latent.pt",
            expected_identity=identity,
            expected_alignment_checkpoint=checkpoint,
            expected_alignment_sha256=checkpoint_sha,
            expected_normalization=normalization,
            expected_shape=(2, 2, 2),
            expected_storage_dtype="float16",
            expected_qa_stats=qa_stats,
        )


def _complete_stage1(version: int, *, phase: str | None, checkpoint_type: str | None) -> dict:
    payload = {
        "checkpoint_version": version,
        "adapter_state_dict": {"weight": torch.ones(1)},
        "compressor_config": {"model": {"name": "test"}},
        "compressor_state_dict": {"weight": torch.ones(1)},
        "args": {"model_name_or_path": "Qwen/Qwen2.5-14B-Instruct"},
    }
    if phase is not None:
        payload["checkpoint_phase"] = phase
    if checkpoint_type is not None:
        payload["checkpoint_type"] = checkpoint_type
    return payload


def test_stage1_checkpoint_envelope_accepts_new_and_known_legacy_alignment_files() -> None:
    modern = validate_stage1_alignment_checkpoint_payload(
        _complete_stage1(3, phase="alignment", checkpoint_type="tensor_patch_text_alignment"),
        path="/data/runs/custom_name.pt",
    )
    assert modern["validation_mode"] == "strict_metadata"

    legacy = validate_stage1_alignment_checkpoint_payload(
        _complete_stage1(0, phase=None, checkpoint_type=None),
        path="/data/runs/alignment_best.pt",
    )
    assert legacy["validation_mode"] == "legacy_alignment_filename"


def test_stage1_checkpoint_envelope_rejects_warmup_and_downstream_legacy_files() -> None:
    payload = _complete_stage1(0, phase=None, checkpoint_type=None)
    with pytest.raises(ValueError, match="accepted only when its filename"):
        validate_stage1_alignment_checkpoint_payload(
            payload,
            path="/data/runs/patch_ae_pretrain_best.pt",
        )
    with pytest.raises(ValueError, match="accepted only when its filename"):
        validate_stage1_alignment_checkpoint_payload(
            payload,
            path="/data/runs/adapter_best.pt",
        )
