from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest
import torch

from scripts.train_tensor_patch_text_alignment import TensorPatchAlignmentAdapter
from scripts.train_tensor_llm_adapter import (
    apply_latent_channel_policy,
    validate_adapter_checkpoint_payload,
)
from scripts.train_tensor_qwen_stage1_ablation import (
    ABLATION_MODE,
    DENSE_CHECKPOINT_TYPE,
    DENSE_CHECKPOINT_VERSION,
    LOCKED_VALIDATION_ARTIFACT_FIELDS,
    _ablation_contract_identity,
    _build_ablation_contract,
    _direct_run_audit,
    _enforce_traceability,
    _json_digest,
    _initial_spatial_state,
    _override_cli_tokens,
    _random_spatial_state,
    _require_locked_file,
    _state_digest,
    _validate_ablation_contract,
    _validate_condition_match,
    _validate_dense_lineage,
    _validated_dense_checkpoint_payload,
    _validate_forwarded_flags,
    _validate_override_keys,
    _validate_test_access_request,
    _validated_protocol_lock,
    DIRECT_OVERRIDE_KEYS,
)
from tensor_compression.utils.pipeline_config import load_yaml_mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _stage1_payload(latent_grid: tuple[int, int] = (2, 2)) -> dict[str, object]:
    torch.manual_seed(7)
    adapter = TensorPatchAlignmentAdapter(
        latent_channels=2,
        latent_grid=latent_grid,
        adapter_dim=8,
        projection_dim=12,
        dropout=0.0,
        adapter_type="spatial_transformer",
        query_tokens=latent_grid[0] * latent_grid[1],
        adapter_layers=1,
        adapter_heads=2,
        soft_prompt_scale=0.05,
    )
    return {
        "checkpoint_type": "tensor_patch_text_alignment",
        "checkpoint_version": 3,
        "checkpoint_phase": "alignment",
        "adapter_state_dict": adapter.state_dict(),
        "compressor_config": {
            "model": {
                "name": "conv_token_autoencoder_2d",
                "input_size": list(latent_grid),
                "latent_grid": list(latent_grid),
                "latent_dim": 2,
                "preserve_input_channels": True,
                "in_channels": 1,
            }
        },
        "compressor_state_dict": {"weight": torch.ones(1)},
        "args": {
            "adapter_layers": 1,
            "adapter_heads": 2,
            "dropout": 0.0,
            "soft_prompt_scale": 0.05,
            "train_patch_ae": False,
            "freeze_patch_ae_after_pretrain": False,
            "alignment_train_patch_ae": False,
            "encoder_source": "patch_ae_config",
            "model_name_or_path": "Qwen/Qwen2.5-14B-Instruct",
        },
    }


def _complete_contract_metadata(contract: dict[str, object]) -> dict[str, object]:
    traceability = {"source_tree": {"git_available": True, "commit": "test"}}
    evaluation = {"primary_metric": "validation_macro_accuracy"}
    contract["traceability"] = traceability
    contract["traceability_sha256"] = _json_digest(traceability)
    contract["evaluation_contract"] = evaluation
    contract["evaluation_contract_sha256"] = _json_digest(evaluation)
    return contract


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


def test_scientific_overrides_are_rejected_by_orchestration_guard() -> None:
    with pytest.raises(ValueError, match="non-scientific"):
        _validate_override_keys(
            {"epochs": 2},
            allowed=DIRECT_OVERRIDE_KEYS,
            section="direct",
        )
    with pytest.raises(ValueError, match="forbids scientific CLI overrides"):
        _validate_forwarded_flags(["--max-updates", "100"], phase="direct")
    with pytest.raises(ValueError, match="direct phase"):
        _validate_forwarded_flags(["--evaluate-test"], phase="direct")
    _validate_forwarded_flags(["--evaluate-test"], phase="dense")


@pytest.mark.parametrize(
    ("configured", "checkpoint"),
    [
        ("adapter_only", "no_learned_stage1"),
        ("no_learned_stage1", "adapter_only"),
    ],
)
def test_dense_phase_rejects_cross_condition_direct_checkpoint(
    configured: str, checkpoint: str
) -> None:
    with pytest.raises(ValueError, match="does not match"):
        _validate_condition_match(configured, {"condition": checkpoint})


def test_test_access_requires_lock_and_exact_resume() -> None:
    with pytest.raises(ValueError, match="cannot authorize a fresh"):
        _validate_test_access_request(
            wants_test=True,
            protocol_lock="lock.json",
            resume_checkpoint=None,
        )
    with pytest.raises(ValueError, match="together with --protocol-lock"):
        _validate_test_access_request(
            wants_test=True,
            protocol_lock=None,
            resume_checkpoint="cross_attention_last.pt",
        )
    _validate_test_access_request(
        wants_test=True,
        protocol_lock="lock.json",
        resume_checkpoint="cross_attention_last.pt",
    )


def test_dense_recipe_is_part_of_resume_identity() -> None:
    first = {
        "condition": "adapter_only",
        "dense_training_recipe": {"world_size": 3, "effective_batch_size": 9},
    }
    changed = copy.deepcopy(first)
    changed["dense_training_recipe"]["world_size"] = 1

    assert _ablation_contract_identity(first) != _ablation_contract_identity(changed)


def test_formal_traceability_rejects_untracked_experiment_file() -> None:
    traceability = {
        "source_tree": {"git_available": True, "tracked_dirty": False},
        "formal_experiment_files": {
            "scripts/experiment.py": {
                "sha256": "a" * 64,
                "tracked_by_git": False,
            }
        },
    }

    with pytest.raises(RuntimeError, match="current Git commit"):
        _enforce_traceability(traceability, {})


def test_locked_artifact_hash_mismatch_fails_before_test(tmp_path: Path) -> None:
    artifact = tmp_path / "cross_attention_last.pt"
    artifact.write_bytes(b"locked validation checkpoint")

    with pytest.raises(ValueError, match="changed before test access"):
        _require_locked_file(artifact, "0" * 64, "last checkpoint")


def test_dense_checkpoint_preflight_requires_resumable_state_and_lineage() -> None:
    lineage = {
        "condition": "adapter_only",
        "source_stage1_sha256": "a" * 64,
        "source_adapter_state_sha256": "b" * 64,
        "initial_adapter_state_sha256": "c" * 64,
        "source_encoder_state_sha256": "d" * 64,
        "latent_channel_policy": "all",
        "evaluation_contract_sha256": "e" * 64,
        "direct_qa_checkpoint_sha256": "f" * 64,
        "direct_run_audit_sha256": "1" * 64,
    }
    payload = {
        "checkpoint_type": DENSE_CHECKPOINT_TYPE,
        "checkpoint_version": DENSE_CHECKPOINT_VERSION,
        "architecture": {"stage1_ablation": copy.deepcopy(lineage)},
        "trainable_state_dict": {"weight": torch.ones(1)},
        "optimizer_state_dict": {},
        "scheduler_state_dict": {},
        "progress": {"global_step": 3000},
    }

    observed, progress = _validated_dense_checkpoint_payload(
        payload,
        label="resume",
        require_optimizer_state=True,
    )
    _validate_dense_lineage(observed, lineage, label="resume")
    assert progress["global_step"] == 3000

    missing_scheduler = copy.deepcopy(payload)
    missing_scheduler.pop("scheduler_state_dict")
    with pytest.raises(ValueError, match="optimizer or scheduler"):
        _validated_dense_checkpoint_payload(
            missing_scheduler,
            label="resume",
            require_optimizer_state=True,
        )

    changed = copy.deepcopy(lineage)
    changed["condition"] = "no_learned_stage1"
    with pytest.raises(ValueError, match="different ablation lineage"):
        _validate_dense_lineage(observed, changed, label="resume")


def test_test_protocol_lock_requires_completed_validation_digest(tmp_path: Path) -> None:
    locked_artifacts = {
        condition: {
            field: hashlib.sha256(f"{condition}:{field}".encode("utf-8")).hexdigest()
            for field in LOCKED_VALIDATION_ARTIFACT_FIELDS
        }
        for condition in (
            "full_stage1_reference",
            "adapter_only",
            "no_learned_stage1",
        )
    }
    comparison_path = tmp_path / "validation_comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "format": "stage1_necessity_comparison_v1",
                "valid_matched_comparison": True,
                "metric_split": "validation",
                "conditions": [
                    "full_stage1_reference",
                    "adapter_only",
                    "no_learned_stage1",
                ],
                "source_stage1_sha256": "b" * 64,
                "evaluation_contract": {"experiment_id": "stage1_necessity_matched_v1"},
                "artifacts": locked_artifacts,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    comparison_sha256 = hashlib.sha256(comparison_path.read_bytes()).hexdigest()
    lock_path = tmp_path / "lock.json"
    lock = {
        "experiment_id": "stage1_necessity_matched_v1",
        "status": "locked",
        "conditions": [
            "full_stage1_reference",
            "adapter_only",
            "no_learned_stage1",
        ],
        "validation_comparison": str(comparison_path),
        "validation_comparison_sha256": comparison_sha256,
        "decision_note": "Proceed with the preregistered three-condition test comparison.",
    }
    lock_path.write_text(json.dumps(lock), encoding="utf-8")

    observed = _validated_protocol_lock(lock_path)
    assert observed["validation_comparison_sha256"] == comparison_sha256
    assert observed["validation_artifacts"] == locked_artifacts

    lock["validation_comparison_sha256"] = "REPLACE_WITH_64_HEX_SHA256"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    with pytest.raises(ValueError, match="completed validation comparison"):
        _validated_protocol_lock(lock_path)


def test_direct_run_audit_rejects_unselected_last_checkpoint(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "adapter_last.pt"
    checkpoint = {
        "adapter_state_dict": {"weight": torch.ones(2, 2)},
        "args": {"latent_channel_policy": "all"},
        "metrics": {"epoch": 1},
    }
    torch.save(checkpoint, checkpoint_path)
    (tmp_path / "run_summary.json").write_text(
        json.dumps(
            {
                "timing": {"status": "completed"},
                "result": {
                    "test_evaluated": False,
                    "test_requested": False,
                    "selected_checkpoint": "adapter_best.pt",
                },
                "latent_channel_policy": "all",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "run_timing.json").write_text(
        json.dumps({"status": "completed"}), encoding="utf-8"
    )
    (tmp_path / "data_audit.json").write_text(
        json.dumps(
            {
                split: {"record_contract_sha256": "a" * 64}
                for split in ("train", "val", "test")
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "qa_metadata_audit.json").write_text(
        json.dumps({"latent_contract": {"format": "test"}}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="selected adapter_best.pt"):
        _direct_run_audit(checkpoint_path, checkpoint, expected_policy="all")


def test_ablation_contract_recomputes_random_initializer(tmp_path: Path) -> None:
    source = tmp_path / "alignment_best.pt"
    payload = _stage1_payload()
    torch.save(payload, source)
    random_state = _random_spatial_state(payload, seed=42)
    contract = _complete_contract_metadata(
        _build_ablation_contract(source, payload, random_state, seed=42)
    )

    observed = _validate_ablation_contract(contract, source)

    assert observed["mode"] == ABLATION_MODE
    assert observed["initial_adapter_state_sha256"] == _state_digest(random_state)


def test_matched_reference_preserves_exact_stage1_adapter(tmp_path: Path) -> None:
    source = tmp_path / "alignment_best.pt"
    payload = _stage1_payload()
    torch.save(payload, source)
    initial_state = _initial_spatial_state(
        payload, seed=42, condition="full_stage1_reference"
    )

    contract = _build_ablation_contract(
        source,
        payload,
        initial_state,
        seed=42,
        condition="full_stage1_reference",
    )

    assert contract["condition_role"] == "matched_reference"
    assert contract["adapter_state_action"] == "preserve"
    assert contract["factor_removed"] == "none"
    assert contract["changed"] == []
    assert contract["source_adapter_state_sha256"] == contract[
        "initial_adapter_state_sha256"
    ]


def test_ablation_contract_rejects_changed_source_checkpoint(tmp_path: Path) -> None:
    source = tmp_path / "alignment_best.pt"
    payload = _stage1_payload()
    torch.save(payload, source)
    random_state = _random_spatial_state(payload, seed=42)
    contract = _complete_contract_metadata(
        _build_ablation_contract(source, payload, random_state, seed=42)
    )
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
    contract = _complete_contract_metadata(
        _build_ablation_contract(source, payload, random_state, seed=42)
    )
    migrated = tmp_path / "new_mount" / "alignment_best.pt"
    migrated.parent.mkdir()
    shutil.copyfile(source, migrated)

    observed = _validate_ablation_contract(contract, migrated)

    assert observed["source_stage1_checkpoint"] == str(source)


def test_adapter_only_records_but_preserves_encoder_updated_by_alignment(tmp_path: Path) -> None:
    source = tmp_path / "alignment_best.pt"
    payload = _stage1_payload()
    payload["args"]["alignment_train_patch_ae"] = True
    torch.save(payload, source)
    random_state = _random_spatial_state(payload, seed=42)

    contract = _build_ablation_contract(source, payload, random_state, seed=42)

    assert contract["source_encoder_trained_during_alignment"] is True
    assert contract["latent_channel_policy"] == "all"
    assert "encoder lineage remain active" in contract["claim_scope"]


def test_embedded_encoder_state_binds_incomplete_upstream_checkpoint_lineage(
    tmp_path: Path,
) -> None:
    source = tmp_path / "alignment_best.pt"
    payload = _stage1_payload()
    payload["args"]["encoder_source"] = "checkpoint"
    payload["args"]["compressor_checkpoint"] = "/unavailable/patch_ae_best.pt"
    payload["args"].pop("compressor_checkpoint_sha256", None)
    torch.save(payload, source)
    initial_state = _initial_spatial_state(
        payload, seed=42, condition="full_stage1_reference"
    )

    contract = _build_ablation_contract(
        source,
        payload,
        initial_state,
        seed=42,
        condition="full_stage1_reference",
    )

    assert contract["source_encoder_origin"] == "checkpoint"
    assert contract["source_encoder_lineage_complete"] is False
    assert contract["source_encoder_state_sha256"] == _state_digest(
        payload["compressor_state_dict"]
    )


def test_no_learned_stage1_accepts_encoder_update_because_learned_channels_are_removed(
    tmp_path: Path,
) -> None:
    source = tmp_path / "alignment_best.pt"
    payload = _stage1_payload()
    payload["args"]["alignment_train_patch_ae"] = True
    torch.save(payload, source)
    random_state = _random_spatial_state(payload, seed=42)

    contract = _build_ablation_contract(
        source,
        payload,
        random_state,
        seed=42,
        condition="no_learned_stage1",
    )

    assert contract["latent_channel_policy"] == "value_only"
    assert contract["source_encoder_trained_during_alignment"] is True


def test_ablation_rejects_checkpoint_without_explicit_value_channel(tmp_path: Path) -> None:
    source = tmp_path / "alignment_best.pt"
    payload = _stage1_payload()
    payload["compressor_config"]["model"]["preserve_input_channels"] = False
    torch.save(payload, source)
    random_state = _random_spatial_state(payload, seed=42)

    with pytest.raises(ValueError, match="preserve_input_channels=true"):
        _build_ablation_contract(source, payload, random_state, seed=42)


def test_ablation_rejects_multichannel_or_misaligned_value_grid(tmp_path: Path) -> None:
    source = tmp_path / "alignment_best.pt"
    payload = _stage1_payload()
    torch.save(payload, source)
    random_state = _random_spatial_state(payload, seed=42)

    payload["compressor_config"]["model"]["in_channels"] = 2
    with pytest.raises(ValueError, match="in_channels=1"):
        _build_ablation_contract(source, payload, random_state, seed=42)

    payload["compressor_config"]["model"]["in_channels"] = 1
    payload["compressor_config"]["model"]["input_size"] = [4, 4]
    with pytest.raises(ValueError, match="input_size == latent_grid"):
        _build_ablation_contract(source, payload, random_state, seed=42)


def test_random_state_supports_rectangular_grid_from_compressor_contract() -> None:
    payload = _stage1_payload(latent_grid=(2, 3))

    random_state = _random_spatial_state(payload, seed=42)

    assert random_state["spatial_pos_encoding"].shape[1] == 6
    assert set(random_state) == set(payload["adapter_state_dict"])


def test_value_only_policy_preserves_channel_zero_and_does_not_mutate_source() -> None:
    latent = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2)
    original = latent.clone()

    observed = apply_latent_channel_policy(latent, "value_only")

    assert torch.equal(observed[0], original[0])
    assert torch.count_nonzero(observed[1:]) == 0
    assert torch.equal(latent, original)
    assert observed.data_ptr() != latent.data_ptr()


def test_direct_checkpoint_policy_is_legacy_compatible_and_fail_closed() -> None:
    latent_contract = {
        "format": "tensor_patch_latent_v3",
        "alignment_checkpoint": "/old/alignment_best.pt",
        "alignment_checkpoint_sha256": "a" * 64,
        "latent_shape": [3, 2, 2],
    }
    checkpoint = {
        "checkpoint_type": "tensor_llm_adapter",
        "checkpoint_version": 2,
        "adapter_state_dict": {"weight": torch.ones(2, 2)},
        "args": {"adapter_architecture": "alignment_adapter"},
        "latent_shape_chw": [3, 2, 2],
        "llm_hidden_size": 24,
        "latent_contract": latent_contract,
    }
    legacy_state = validate_adapter_checkpoint_payload(
        checkpoint,
        expected_latent_shape=(3, 2, 2),
        expected_llm_hidden_size=24,
        expected_architecture="alignment_adapter",
        expected_latent_contract=latent_contract,
    )
    assert torch.equal(legacy_state["weight"], torch.ones(2, 2))

    checkpoint["args"]["latent_channel_policy"] = "value_only"
    with pytest.raises(ValueError, match="latent channel policy mismatch"):
        validate_adapter_checkpoint_payload(
            checkpoint,
            expected_latent_shape=(3, 2, 2),
            expected_llm_hidden_size=24,
            expected_architecture="alignment_adapter",
            expected_latent_contract=latent_contract,
        )
    value_only_state = validate_adapter_checkpoint_payload(
        checkpoint,
        expected_latent_shape=(3, 2, 2),
        expected_llm_hidden_size=24,
        expected_architecture="alignment_adapter",
        expected_latent_contract=latent_contract,
        expected_latent_channel_policy="value_only",
    )
    assert torch.equal(value_only_state["weight"], torch.ones(2, 2))


def test_release_ablation_configs_are_explicit_and_inherit_release_trainers() -> None:
    reference_config = load_yaml_mapping(
        PROJECT_ROOT / "configs" / "field_to_llm_stage1_reference.yaml"
    )
    adapter_config = load_yaml_mapping(
        PROJECT_ROOT / "configs" / "field_to_llm_stage1_adapter_ablation.yaml"
    )
    no_stage1_config = load_yaml_mapping(
        PROJECT_ROOT / "configs" / "field_to_llm_no_learned_stage1_ablation.yaml"
    )

    configs = (reference_config, adapter_config, no_stage1_config)
    assert [config["ablation"]["condition"] for config in configs] == [
        "full_stage1_reference",
        "adapter_only",
        "no_learned_stage1",
    ]
    for config in configs:
        assert config["direct"]["base_config"] == "configs/field_to_llm_direct_qa.yaml"
        assert config["dense"]["base_config"] == "configs/field_to_llm_cross_attention.yaml"
        assert config["direct"]["arg_overrides"]["evaluate_test"] is False
        assert config["dense"]["arg_overrides"]["evaluate_test"] is False
        assert "/data/" not in str(config)

    normalized = []
    for config in configs:
        comparable = copy.deepcopy(config)
        comparable["ablation"].pop("condition")
        comparable["direct"]["arg_overrides"].pop("run_name")
        comparable["dense"]["arg_overrides"].pop("run_name")
        normalized.append(comparable)
    assert normalized[0] == normalized[1] == normalized[2]
