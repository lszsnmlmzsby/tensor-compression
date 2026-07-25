from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn.functional as F
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import scripts.train_tensor_llm_adapter as adapter_training  # noqa: E402
from scripts.train_tensor_llm_adapter import (  # noqa: E402
    ExactDistributedEvalSampler,
    GroundedEvidenceAdapter,
    HybridGlobalLocalAdapter,
    ResidualQuestionConditionedAdapter,
    RunLifecycle,
    StateTaskGroupedBatchSampler,
    TensorReadoutQADataset,
    _decoder_question_last_hidden,
    _resolved_diagnostic_layers,
    _sequence_choice_ce_loss,
    average_trainable_gradients_by_record_count,
    append_jsonl,
    adapter_from_checkpoint,
    audit_qa_datasets,
    build_distributed_run_dir,
    build_local_conditioning_prompt,
    checkpoint_score,
    choice_ce_loss,
    evaluate_choice_accuracy,
    frozen_llm_checkpoint_execution_active,
    grounded_reader_geometry_metrics,
    grounded_routing_loss,
    grounded_routing_warmup_audit,
    grounding_query_spec_for_record,
    initialize_distributed_device,
    log_wandb_on_rank_zero,
    matched_coordinate_group_loss,
    parse_generated_choice,
    read_host_memory_snapshot,
    reset_grounded_evidence_optimizer_state,
    routing_metric_weighted_totals,
    run_embedded_diagnostics,
    run_on_rank_zero_and_broadcast,
    save_validate_and_rebuild_adapter_checkpoint,
    optimizer_parameter_audit,
    same_state_question_swap_indices,
    selective_answer_statistics,
    set_frozen_llm_execution_mode,
    single_token_choice_ids,
    structured_query_features_for_record,
    task_specific_instruction,
    training_loss,
    validate_adapter_loss_contract,
    validate_adapter_checkpoint_payload,
    validate_qa_latent_contract,
    validate_stage1_alignment_checkpoint_phase,
    validate_stage1_model_identity,
    validate_stage1_teacher_supervision,
)
from tensor_compression.models.compressors.conv_token_autoencoder_2d import (  # noqa: E402
    ConvTokenAutoencoder2D,
)
from tensor_compression.downstream.patch_qa_contract import (  # noqa: E402
    MATCHED_GROUP_FORMAT,
    PATCH_LATENT_AUDIT_FORMAT,
    PATCH_LATENT_FORMAT,
    PATCH_QA_BUILD_MARKER,
    PATCH_QA_FORMAT,
    sha256_file,
    validate_stage1_alignment_checkpoint_payload,
)
from scripts.train_tensor_patch_text_alignment import (  # noqa: E402
    TensorPatchAlignmentAdapter,
    alignment_adapter_path_metrics,
    alignment_adapter_parameter_metrics,
    sinusoidal_2d_position_encoding,
)


def _record(state: str, task: str, field: str, question: str) -> dict[str, str]:
    return {
        "state_ref": state,
        "task_type": task,
        "field": field,
        "query": question,
        "question": question,
    }


def _stage1_checkpoint_payload(
    *,
    version: int = 3,
    phase: str | None = "alignment",
    include_type: bool = True,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "checkpoint_version": version,
        "adapter_state_dict": {"adapter.weight": torch.zeros(1)},
        "compressor_config": {"model": {"name": "test"}},
        "compressor_state_dict": {"encoder.weight": torch.zeros(1)},
        "args": {"model_name_or_path": "Qwen/Qwen2.5-14B-Instruct"},
    }
    if include_type:
        payload["checkpoint_type"] = "tensor_patch_text_alignment"
    if phase is not None:
        payload["checkpoint_phase"] = phase
    return payload


class TestAdapterLossContracts(unittest.TestCase):
    def test_stage1_model_identity_accepts_hf_and_local_paths_for_the_same_model(self) -> None:
        validate_stage1_model_identity(
            {"model_name_or_path": "Qwen/Qwen2.5-14B-Instruct"},
            "/data/models/Qwen2.5-14B-Instruct",
        )

    def test_stage1_model_identity_rejects_same_width_different_models(self) -> None:
        with self.assertRaisesRegex(ValueError, "same frozen LLM"):
            validate_stage1_model_identity(
                {"model_name_or_path": "/data/models/Qwen2.5-32B-Instruct"},
                "/data/models/Qwen2.5-14B-Instruct",
            )

    def test_stage1_multilayer_supervision_metadata_is_strict_but_loss_transform_is_descriptive(self) -> None:
        checkpoint = {
            "checkpoint_version": 3,
            "teacher_supervision": {
                "primary_layer": 2,
                "layers": [2, 4, 6],
                "auxiliary_layers": [4, 6],
                "auxiliary_layer_weights": {"4": 0.25, "6": 0.25},
                "primary_feature_transform": "whitening",
                "auxiliary_feature_transform": "native_centered_and_branch_mean",
            },
        }

        metadata = validate_stage1_teacher_supervision(checkpoint)

        self.assertIsNotNone(metadata)
        self.assertEqual(metadata["layers"], [2, 4, 6])
        self.assertEqual(metadata["primary_feature_transform"], "whitening")
        self.assertIsNone(validate_stage1_teacher_supervision({"checkpoint_version": 1}))
        self.assertIsNone(validate_stage1_teacher_supervision({"checkpoint_version": 2}))

    def test_stage1_multilayer_supervision_metadata_rejects_missing_or_inconsistent_layers(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing teacher_supervision"):
            validate_stage1_teacher_supervision({"checkpoint_version": 3})
        with self.assertRaisesRegex(ValueError, "disagree"):
            validate_stage1_teacher_supervision(
                {
                    "checkpoint_version": 3,
                    "teacher_supervision": {
                        "primary_layer": 2,
                        "layers": [2, 4, 6],
                        "auxiliary_layers": [4],
                        "auxiliary_layer_weights": {"4": 0.25},
                        "primary_feature_transform": "whitening",
                        "auxiliary_feature_transform": "native_centered_and_branch_mean",
                    },
                }
            )

    def test_direct_stage2_rejects_patch_ae_warmup_checkpoint(self) -> None:
        self.assertEqual(
            validate_stage1_alignment_checkpoint_phase(
                _stage1_checkpoint_payload(),
                "/data/runs/alignment_best.pt",
            ),
            "alignment",
        )
        with self.assertRaisesRegex(ValueError, "patch-AE warmup checkpoint"):
            validate_stage1_alignment_checkpoint_phase(
                _stage1_checkpoint_payload(phase="patch_ae_pretrain"),
                "/data/runs/patch_ae_pretrain_best.pt",
        )
        with self.assertRaisesRegex(ValueError, "missing checkpoint_phase"):
            validate_stage1_alignment_checkpoint_phase(
                _stage1_checkpoint_payload(phase=None),
                "/data/runs/alignment_best.pt",
            )
        self.assertEqual(
            validate_stage1_alignment_checkpoint_phase(
                _stage1_checkpoint_payload(version=2, phase=None, include_type=False),
                "/data/runs/alignment_best.pt",
            ),
            "alignment",
        )
        with self.assertRaisesRegex(ValueError, "accepted only when its filename"):
            validate_stage1_alignment_checkpoint_phase(
                _stage1_checkpoint_payload(version=2, phase=None, include_type=False),
                "/data/runs/patch_ae_pretrain_best.pt",
            )

    def test_stage1_checkpoint_envelope_rejects_incomplete_legacy_payload(self) -> None:
        incomplete = _stage1_checkpoint_payload(version=2, phase=None, include_type=False)
        del incomplete["compressor_state_dict"]
        with self.assertRaisesRegex(ValueError, "complete Stage-1 alignment checkpoint"):
            validate_stage1_alignment_checkpoint_payload(
                incomplete,
                path="/data/runs/alignment_best.pt",
            )

    def test_direct_alignment_rejects_identical_global_only_ranking(self) -> None:
        args = SimpleNamespace(
            adapter_architecture="alignment_adapter",
            ranking_loss_weight=0.1,
            ranking_loss_negative="global_only",
            swapped_question_loss_weight=0.0,
        )

        with self.assertRaisesRegex(ValueError, "identical soft prompts"):
            validate_adapter_loss_contract(args)

    def test_direct_alignment_rejects_ambiguous_shuffled_tensor_ranking(self) -> None:
        args = SimpleNamespace(
            adapter_architecture="alignment_adapter",
            ranking_loss_weight=0.1,
            ranking_loss_negative="shuffled",
            swapped_question_loss_weight=0.0,
        )

        with self.assertRaisesRegex(ValueError, "same valid answer"):
            validate_adapter_loss_contract(args)

    def test_direct_alignment_accepts_no_latent_ranking(self) -> None:
        args = SimpleNamespace(
            adapter_architecture="alignment_adapter",
            ranking_loss_weight=0.1,
            ranking_loss_negative="no_latent",
            swapped_question_loss_weight=0.0,
        )

        validate_adapter_loss_contract(args)

    def test_direct_alignment_rejects_identity_swapped_question_loss(self) -> None:
        args = SimpleNamespace(
            adapter_architecture="alignment_adapter",
            ranking_loss_weight=0.0,
            ranking_loss_negative="shuffled",
            swapped_question_loss_weight=0.1,
        )

        with self.assertRaisesRegex(ValueError, "question-independent tensor prefix"):
            validate_adapter_loss_contract(args)

    def test_direct_alignment_rejects_branch_only_baselines(self) -> None:
        args = SimpleNamespace(
            adapter_architecture="alignment_adapter",
            ranking_loss_weight=0.0,
            ranking_loss_negative="shuffled",
            swapped_question_loss_weight=0.0,
            eval_baselines="correct,global_only",
            final_eval_baselines="correct,shuffled",
        )

        with self.assertRaisesRegex(ValueError, "no global/local split"):
            validate_adapter_loss_contract(args)

    def test_grounded_routing_warmup_requires_positive_gate_loss(self) -> None:
        with self.assertRaisesRegex(ValueError, "grounding_gate_loss_weight > 0"):
            validate_adapter_loss_contract(
                SimpleNamespace(
                    adapter_architecture="grounded_evidence_adapter",
                    grounding_routing_warmup_epochs=1,
                    grounding_gate_loss_weight=0.0,
                )
            )

        validate_adapter_loss_contract(
            SimpleNamespace(
                adapter_architecture="grounded_evidence_adapter",
                grounding_routing_warmup_epochs=1,
                grounding_gate_loss_weight=0.1,
            )
        )
        validate_adapter_loss_contract(
            SimpleNamespace(
                adapter_architecture="grounded_evidence_adapter",
                grounding_routing_warmup_epochs=0,
                grounding_gate_loss_weight=0.0,
            )
        )


class TestTrainingAudits(unittest.TestCase):
    def test_grounded_checkpoint_score_requires_evidence_and_correct_tensor(self) -> None:
        metrics = {
            "correct": {
                "by_task": {
                    "normalized_point_value": {"accuracy": 0.80},
                    "raw_point_value_with_stats": {"accuracy": 0.70},
                }
            },
            "global_only": {
                "by_task": {
                    "normalized_point_value": {"accuracy": 0.40},
                    "raw_point_value_with_stats": {"accuracy": 0.50},
                }
            },
            "shuffled": {
                "by_task": {
                    "normalized_point_value": {"accuracy": 0.30},
                    "raw_point_value_with_stats": {"accuracy": 0.20},
                }
            },
        }

        self.assertAlmostEqual(
            checkpoint_score(metrics, "point_value_min_grounded_gain"),
            0.20,
        )
        del metrics["global_only"]
        self.assertEqual(
            checkpoint_score(metrics, "point_value_min_grounded_gain"),
            -math.inf,
        )

    def test_routing_metric_totals_use_role_and_gate_denominators(self) -> None:
        totals = routing_metric_weighted_totals(
            {
                "routing_active_roles": 3.0,
                "routing_top1_accuracy": 2.0 / 3.0,
                "routing_top5_accuracy": 1.0,
                "routing_row_top1_accuracy": 1.0 / 3.0,
                "routing_col_top1_accuracy": 2.0 / 3.0,
                "routing_target_mass": 0.25,
                "routing_normalized_entropy": 0.4,
                "routing_gate_accuracy": 0.75,
                "routing_gate_active_fraction": 0.5,
                "routing_gate_target_active_fraction": 0.75,
            },
            record_count=2,
            gate_slots_per_record=2,
        )

        self.assertAlmostEqual(totals["routing_normalized_entropy_sum"], 1.2)
        self.assertEqual(totals["routing_gate_slots"], 4.0)
        self.assertEqual(totals["routing_gate_target_active"], 3.0)
        self.assertEqual(totals["routing_top1_correct"], 2.0)

    def test_jsonl_metrics_append_one_complete_row_per_update(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train_updates.jsonl"

            append_jsonl(path, {"global_step": 1, "train_loss": 2.0})
            append_jsonl(path, {"global_step": 2, "train_loss": 1.0})

            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(
                rows,
                [
                    {"global_step": 1, "train_loss": 2.0},
                    {"global_step": 2, "train_loss": 1.0},
                ],
            )

    def test_atomic_writers_remove_partial_temporary_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_target = root / "metrics.json"
            checkpoint_target = root / "adapter.pt"

            def fail_json(path, _payload):
                Path(path).write_text("partial", encoding="utf-8")
                raise OSError("json disk full")

            def fail_checkpoint(_payload, path):
                Path(path).write_bytes(b"partial")
                raise OSError("checkpoint disk full")

            with (
                mock.patch.object(adapter_training, "dump_json", side_effect=fail_json),
                self.assertRaisesRegex(OSError, "json disk full"),
            ):
                adapter_training.atomic_dump_json(json_target, {"epoch": 1})
            with (
                mock.patch.object(adapter_training.torch, "save", side_effect=fail_checkpoint),
                self.assertRaisesRegex(OSError, "checkpoint disk full"),
            ):
                adapter_training.atomic_torch_save(checkpoint_target, {"weight": torch.ones(1)})

            self.assertFalse(json_target.exists())
            self.assertFalse(checkpoint_target.exists())
            self.assertFalse((root / ".metrics.json.tmp").exists())
            self.assertFalse((root / ".adapter.pt.tmp").exists())

    def test_stage2_rejects_an_incomplete_qa_directory_build(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / PATCH_QA_BUILD_MARKER
            marker.write_text('{"status":"building"}', encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "incomplete or active build"):
                adapter_training.audit_qa_metadata(SimpleNamespace(qa_dir=directory))

    def test_run_lifecycle_preserves_original_failure_when_summary_is_corrupt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            lifecycle = RunLifecycle(run_dir)
            (run_dir / "run_summary.json").write_text("{truncated", encoding="utf-8")

            timing = lifecycle.finish("failed", RuntimeError("original training failure"))

            persisted = json.loads((run_dir / "run_timing.json").read_text(encoding="utf-8"))
            self.assertEqual(timing["status"], "failed")
            self.assertEqual(timing["error_message"], "original training failure")
            self.assertIn("run_summary_update_error", timing)
            self.assertEqual(persisted, timing)

    def test_diagnostics_restore_training_and_checkpoint_execution_after_failure(self) -> None:
        adapter = nn.Linear(2, 2).train()
        llm = nn.Module()
        llm.model = nn.Sequential(nn.Linear(2, 2), nn.Dropout(0.5))
        llm.is_gradient_checkpointing = True
        set_frozen_llm_execution_mode(llm, checkpoint_training=True)

        with (
            mock.patch.object(
                adapter_training,
                "_run_embedded_diagnostics_impl",
                side_effect=RuntimeError("diagnostic failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "diagnostic failed"),
        ):
            run_embedded_diagnostics(
                stage="test",
                llm=llm,
                adapter=adapter,
                tokenizer=None,
                dataset=None,
                device=torch.device("cpu"),
                args=SimpleNamespace(),
                run_dir=Path("unused"),
            )

        self.assertTrue(adapter.training)
        self.assertTrue(frozen_llm_checkpoint_execution_active(llm))
        self.assertFalse(llm.model[0].training)
        self.assertFalse(llm.model[1].training)

    def test_stage2_checkpoint_envelope_binds_latent_and_stage1_provenance(self) -> None:
        contract = {
            "format": PATCH_LATENT_FORMAT,
            "alignment_checkpoint": "/data/run/alignment_best.pt",
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
            "latent_contract": dict(contract),
        }

        state = validate_adapter_checkpoint_payload(
            checkpoint,
            expected_latent_shape=(3, 2, 2),
            expected_llm_hidden_size=24,
            expected_architecture="alignment_adapter",
            expected_latent_contract=contract,
        )
        self.assertTrue(torch.equal(state["weight"], torch.ones(2, 2)))

        changed_contract = dict(contract, alignment_checkpoint_sha256="b" * 64)
        with self.assertRaisesRegex(ValueError, "different latent/Stage-1 contract"):
            validate_adapter_checkpoint_payload(
                checkpoint,
                expected_latent_shape=(3, 2, 2),
                expected_llm_hidden_size=24,
                expected_architecture="alignment_adapter",
                expected_latent_contract=changed_contract,
            )

    def test_optimizer_parameter_audit_requires_exact_membership(self) -> None:
        module = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
        optimizer = torch.optim.AdamW(module.parameters(), lr=1.0e-3)

        metrics = optimizer_parameter_audit(optimizer, module)

        self.assertEqual(metrics["optimizer_missing_trainable_tensor_count"], 0)
        self.assertEqual(metrics["optimizer_extra_tensor_count"], 0)
        incomplete = torch.optim.AdamW(module[0].parameters(), lr=1.0e-3)
        with self.assertRaisesRegex(RuntimeError, "missing_trainable"):
            optimizer_parameter_audit(incomplete, module)
        outside = nn.Parameter(torch.zeros(1))
        with self.assertRaisesRegex(RuntimeError, "outside_module"):
            optimizer_parameter_audit(
                torch.optim.AdamW([*module.parameters(), outside], lr=1.0e-3),
                module,
            )

    def test_record_weighted_gradient_normalization_in_single_process(self) -> None:
        module = nn.Linear(2, 1, bias=False)
        module.weight.grad = torch.full_like(module.weight, 6.0)

        global_records = average_trainable_gradients_by_record_count(
            module,
            local_record_count=3,
            device=torch.device("cpu"),
        )

        self.assertEqual(global_records, 3)
        self.assertTrue(torch.equal(module.weight.grad, torch.full_like(module.weight, 2.0)))

    def test_frozen_llm_checkpoint_state_uses_decoder_not_outer_training_flag(self) -> None:
        model = nn.Module()
        model.model = nn.Sequential(nn.Linear(2, 2), nn.Dropout(0.5))
        model.is_gradient_checkpointing = True

        set_frozen_llm_execution_mode(model, checkpoint_training=True)

        self.assertFalse(model.training)
        self.assertTrue(model.model.training)
        self.assertFalse(model.model[0].training)
        self.assertFalse(model.model[1].training)
        self.assertTrue(frozen_llm_checkpoint_execution_active(model))
        model.eval()
        self.assertFalse(frozen_llm_checkpoint_execution_active(model))


class _DiagnosticDecoder(nn.Module):
    def forward(self, *, inputs_embeds, attention_mask, **_kwargs):
        assert tuple(attention_mask.shape[:2]) == tuple(inputs_embeds.shape[:2])
        return SimpleNamespace(
            hidden_states=tuple(inputs_embeds + float(layer) for layer in range(3))
        )


class TestQuestionLastDiagnostics(unittest.TestCase):
    def test_diagnostic_layer_indices_are_resolved_and_deduplicated(self) -> None:
        self.assertEqual(_resolved_diagnostic_layers([0, 2, -1, 2], 3), [0, 2])
        with self.assertRaisesRegex(ValueError, "invalid"):
            _resolved_diagnostic_layers([3], 3)

    def test_question_last_hidden_uses_last_unmasked_prompt_token(self) -> None:
        soft = torch.zeros(1, 2, 3)
        text = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
        text_mask = torch.tensor([[1, 1, 1, 0]])
        prompt_mask = torch.tensor([[1, 1, 0, 0]])

        hidden = _decoder_question_last_hidden(
            decoder=_DiagnosticDecoder(),
            soft_embeds=soft,
            text_embeds=text,
            text_mask=text_mask,
            prompt_mask=prompt_mask,
            requested_layers=[0, -1],
        )

        self.assertTrue(torch.equal(hidden["0"], text[0, 1]))
        self.assertTrue(torch.equal(hidden["2"], text[0, 1] + 2.0))


class TestDistributedSampling(unittest.TestCase):
    def test_warmup_checkpoint_validation_failure_is_broadcast(self) -> None:
        broadcast_calls: list[dict[str, object]] = []

        def capture(payload, src):
            self.assertEqual(src, 0)
            broadcast_calls.append(dict(payload[0]))

        with (
            mock.patch.object(adapter_training, "distributed_is_initialized", return_value=True),
            mock.patch.object(adapter_training, "is_main_process", return_value=True),
            mock.patch.object(adapter_training.dist, "broadcast_object_list", side_effect=capture),
            mock.patch.object(
                adapter_training,
                "save_validate_and_rebuild_adapter_checkpoint",
                side_effect=ValueError("strict warmup rebuild failed"),
            ) as save_and_validate,
            self.assertRaisesRegex(ValueError, "strict warmup rebuild failed"),
        ):
            run_on_rank_zero_and_broadcast(
                lambda: adapter_training.save_validate_and_rebuild_adapter_checkpoint(
                    Path("adapter_routing_warmup.pt"),
                    adapter=nn.Identity(),
                    args=SimpleNamespace(adapter_architecture="grounded_evidence_adapter"),
                    latent_shape=(3, 2, 2),
                    llm_hidden_size=24,
                    latent_contract={},
                ),
                "routing-warmup checkpoint write/read validation",
            )

        save_and_validate.assert_called_once()
        self.assertEqual(len(broadcast_calls), 1)
        self.assertFalse(bool(broadcast_calls[0]["ok"]))
        self.assertEqual(broadcast_calls[0]["error_type"], "ValueError")

    def test_wandb_rank_zero_failure_is_broadcast_before_reraising(self) -> None:
        broadcast_calls: list[dict[str, object]] = []
        logger = mock.Mock()
        logger.log.side_effect = RuntimeError("wandb transport failed")

        def capture(payload, src):
            self.assertEqual(src, 0)
            broadcast_calls.append(dict(payload[0]))

        with (
            mock.patch.object(adapter_training, "distributed_is_initialized", return_value=True),
            mock.patch.object(adapter_training, "is_main_process", return_value=True),
            mock.patch.object(adapter_training.dist, "broadcast_object_list", side_effect=capture),
            self.assertRaisesRegex(RuntimeError, "wandb transport failed"),
        ):
            log_wandb_on_rank_zero(
                logger,
                {"train_step/loss": 1.25},
                step=7,
                stage="training update W&B log",
            )

        logger.log.assert_called_once_with({"train_step/loss": 1.25}, step=7)
        self.assertEqual(len(broadcast_calls), 1)
        self.assertFalse(bool(broadcast_calls[0]["ok"]))
        self.assertEqual(broadcast_calls[0]["error_type"], "RuntimeError")

    def test_distributed_run_directory_creation_broadcasts_rank_zero_failure(self) -> None:
        broadcast_calls: list[dict[str, object]] = []

        def capture(payload, src):
            self.assertEqual(src, 0)
            broadcast_calls.append(dict(payload[0]))

        with (
            mock.patch.object(adapter_training, "distributed_is_initialized", return_value=True),
            mock.patch.object(adapter_training, "is_main_process", return_value=True),
            mock.patch.object(
                adapter_training,
                "build_run_dir",
                side_effect=OSError("shared storage unavailable"),
            ),
            mock.patch.object(adapter_training.dist, "broadcast_object_list", side_effect=capture),
            self.assertRaisesRegex(OSError, "shared storage unavailable"),
        ):
            build_distributed_run_dir("/shared/runs", "formal")

        self.assertEqual(len(broadcast_calls), 1)
        self.assertFalse(bool(broadcast_calls[0]["ok"]))

    def test_rank_zero_failure_is_broadcast_before_it_is_reraised(self) -> None:
        broadcast_calls: list[dict[str, object]] = []

        def capture(payload, src):
            self.assertEqual(src, 0)
            broadcast_calls.append(dict(payload[0]))

        with (
            mock.patch.object(adapter_training, "distributed_is_initialized", return_value=True),
            mock.patch.object(adapter_training, "is_main_process", return_value=True),
            mock.patch.object(adapter_training.dist, "broadcast_object_list", side_effect=capture),
            self.assertRaisesRegex(ValueError, "bad metadata"),
        ):
            run_on_rank_zero_and_broadcast(
                lambda: (_ for _ in ()).throw(ValueError("bad metadata")),
                "metadata audit",
            )

        self.assertEqual(len(broadcast_calls), 1)
        self.assertFalse(bool(broadcast_calls[0]["ok"]))
        self.assertEqual(broadcast_calls[0]["error_type"], "ValueError")

    def test_distributed_initialization_uses_configured_timeout(self) -> None:
        with (
            mock.patch.dict(os.environ, {"WORLD_SIZE": "2", "LOCAL_RANK": "1"}, clear=False),
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "device_count", return_value=2),
            mock.patch.object(torch.cuda, "set_device") as set_device,
            mock.patch.object(adapter_training.dist, "init_process_group") as init_process_group,
        ):
            device = initialize_distributed_device("auto", distributed_timeout_seconds=7200)

        self.assertEqual(device, torch.device("cuda", 1))
        set_device.assert_called_once_with(1)
        init_process_group.assert_called_once_with(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(seconds=7200),
        )

    def test_distributed_initialization_rejects_invalid_timeout(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            initialize_distributed_device("cpu", distributed_timeout_seconds=float("nan"))

    def test_record_limit_stops_jsonl_loading_early(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps({"state_ref": "first"}) + "\n")
                handle.write("this line is intentionally invalid JSON\n")

            records = TensorReadoutQADataset._load_records(path, max_records=1)

            self.assertEqual(records, [{"state_ref": "first"}])

    def test_worker_cache_capacity_divides_the_per_rank_budget(self) -> None:
        dataset = TensorReadoutQADataset.__new__(TensorReadoutQADataset)
        dataset.latent_cache_size = 10
        with mock.patch.object(
            adapter_training,
            "get_worker_info",
            return_value=SimpleNamespace(num_workers=4),
        ):
            self.assertEqual(dataset.effective_latent_cache_size(), 3)

    def test_linux_host_memory_snapshot_uses_mem_available_and_process_rss(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "self").mkdir()
            (root / "meminfo").write_text(
                "MemTotal:       134217728 kB\nMemFree:         1048576 kB\n"
                "MemAvailable:   67108864 kB\n",
                encoding="utf-8",
            )
            (root / "self" / "status").write_text(
                "Name:\tpython\nVmRSS:\t2097152 kB\n",
                encoding="utf-8",
            )

            snapshot = read_host_memory_snapshot(root)

            self.assertEqual(snapshot["total_gib"], 128.0)
            self.assertEqual(snapshot["available_gib"], 64.0)
            self.assertEqual(snapshot["process_rss_gib"], 2.0)

    def test_formal_latent_contract_rejects_v2_and_changed_checkpoint_contents(self) -> None:
        self.assertIsNone(
            validate_qa_latent_contract(
                {"format": "tensor_patch_qa_v2"},
                configured_alignment_checkpoint=None,
                require_formal_contract=False,
            )
        )
        with self.assertRaisesRegex(ValueError, "requires a supported immutable QA format"):
            validate_qa_latent_contract(
                {"format": "tensor_patch_qa_v2"},
                configured_alignment_checkpoint=None,
                require_formal_contract=True,
            )

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "alignment_best.pt"
            checkpoint.write_bytes(b"first checkpoint contents")
            metadata = {
                "format": PATCH_QA_FORMAT,
                "latent_format": PATCH_LATENT_FORMAT,
                "latent_audit_format": PATCH_LATENT_AUDIT_FORMAT,
                "latent_shape": [8, 16, 16],
                "storage_dtype": "float16",
                "encoder_input_normalization": {
                    "mode": "zscore",
                    "scope": "channel",
                    "stats_path": None,
                    "clip_min": None,
                    "clip_max": None,
                },
                "alignment_checkpoint": str(checkpoint),
                "alignment_checkpoint_sha256": sha256_file(checkpoint),
            }
            validated = validate_qa_latent_contract(
                metadata,
                configured_alignment_checkpoint=checkpoint,
                require_formal_contract=True,
            )
            self.assertEqual(validated["alignment_checkpoint_sha256"], metadata["alignment_checkpoint_sha256"])

            checkpoint.write_bytes(b"changed checkpoint contents")
            with self.assertRaisesRegex(ValueError, "changed after patch latents were generated"):
                validate_qa_latent_contract(
                    metadata,
                    configured_alignment_checkpoint=checkpoint,
                    require_formal_contract=True,
                )

    def test_formal_audit_rejects_one_latent_path_with_different_identities_or_stats(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            latent_path = Path(directory) / "shared.pt"
            latent_path.write_bytes(b"validated by test stub")

            def record(qa_id: str) -> dict[str, object]:
                return {
                    "qa_id": qa_id,
                    "patch_id": "patch_a",
                    "state_ref": "patch_a",
                    "sample_index": 0,
                    "time_index": 1,
                    "top_left": [2, 3],
                    "task_type": "point_compare",
                    "field": "Vx",
                    "metadata": {"field": "Vx"},
                    "choices": ["A", "B"],
                    "answer": "A",
                    "question": "Compare two points.",
                    "latent_audit": {
                        "format": PATCH_LATENT_AUDIT_FORMAT,
                        "mean": 1.0,
                        "std": 2.0,
                        "scale": 2.000001,
                    },
                }

            class AuditDataset:
                latent_contract = {"enabled": True}

                def __init__(self, records):
                    self.records = records

                def latent_path_for_record(self, _record):
                    return latent_path

                def validate_latent_file_for_record(self, _record):
                    return {"path": str(latent_path)}

                def __len__(self):
                    return len(self.records)

            first = record("qa_1")
            different_identity = record("qa_2")
            different_identity["patch_id"] = "patch_b"
            different_identity["state_ref"] = "patch_b"
            with self.assertRaisesRegex(ValueError, "different patch identities"):
                audit_qa_datasets(
                    {"train": AuditDataset([first, different_identity])},
                    require_disjoint_splits=False,
                    require_complete_split_coverage=False,
                )

            different_stats = record("qa_2")
            different_stats["latent_audit"] = {
                **different_stats["latent_audit"],
                "scale": 3.0,
            }
            with self.assertRaisesRegex(ValueError, "different normalization statistics"):
                audit_qa_datasets(
                    {"train": AuditDataset([first, different_stats])},
                    require_disjoint_splits=False,
                    require_complete_split_coverage=False,
                )

    def test_truncated_smoke_audit_does_not_require_all_choice_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            latent_dir = Path(directory)
            torch.save({"latent_map": torch.zeros(1, 2, 2)}, latent_dir / "state.pt")

            def dataset(records):
                value = SimpleNamespace(records=records, latent_dir=latent_dir)
                value.latent_path_for_record = lambda record: latent_dir / f"{record['state_ref']}.pt"
                return value

            train_records = [
                {
                    "qa_id": "state_task_0",
                    "state_ref": "state",
                    "sample_index": 0,
                    "task_type": "raw_point_value_with_stats",
                    "field": "Vx",
                    "choices": ["A", "B", "C", "D"],
                    "answer": "A",
                    "question": "Options: A: 0; B: 1; C: 2; D: 3.",
                }
            ]
            val_records = [dict(train_records[0], qa_id="other_task_0", state_ref="other", sample_index=1)]
            torch.save({"latent_map": torch.zeros(1, 2, 2)}, latent_dir / "other.pt")
            with self.assertRaisesRegex(ValueError, "answer labels absent"):
                audit_qa_datasets(
                    {"train": dataset(train_records), "val": dataset(val_records)},
                    require_disjoint_splits=True,
                    require_complete_split_coverage=True,
                )

            summary = audit_qa_datasets(
                {"train": dataset(train_records), "val": dataset(val_records)},
                require_disjoint_splits=True,
                require_complete_split_coverage=False,
            )
            self.assertFalse(summary["_audit_scope"]["complete_split_coverage_checked"])
            self.assertEqual(summary["val"]["missing_answer_labels"]["raw_point_value_with_stats"], ["B", "C", "D"])

    def test_shuffled_indices_preserve_field_task_and_change_sample(self) -> None:
        records = [
            {
                **_record(f"state_{sample}_{variant}", task, field, f"question_{variant}"),
                "sample_index": sample,
            }
            for field in ("Vx", "Vy")
            for task in ("point", "region")
            for sample in range(3)
            for variant in range(2)
        ]
        dataset = TensorReadoutQADataset.__new__(TensorReadoutQADataset)
        dataset.records = records

        first = dataset._build_random_different_indices(seed=17)
        second = dataset._build_random_different_indices(seed=17)

        self.assertEqual(first, second)
        for source, candidate_index in zip(records, first):
            candidate = records[candidate_index]
            self.assertEqual(candidate["field"], source["field"])
            self.assertEqual(candidate["task_type"], source["task_type"])
            self.assertNotEqual(candidate["sample_index"], source["sample_index"])

    def test_latent_lru_cache_avoids_repeated_torch_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [
                {
                    **_record(f"state_{sample}", "point", "Vx", f"question_{sample}"),
                    "qa_id": f"qa_{sample}",
                    "sample_index": sample,
                }
                for sample in range(2)
            ]
            jsonl = root / "train.jsonl"
            with jsonl.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record) + "\n")
            for sample in range(2):
                torch.save({"latent_map": torch.full((2, 2, 2), float(sample))}, root / f"state_{sample}.pt")
            dataset = TensorReadoutQADataset(
                jsonl_path=jsonl,
                latent_dir=root,
                latent_cache_size=1,
            )

            with mock.patch.object(torch, "load", wraps=torch.load) as wrapped_load:
                first = dataset.load_latent_for_record(records[0])
                repeated = dataset.load_latent_for_record(records[0])
                dataset.load_latent_for_record(records[1])
                reloaded = dataset.load_latent_for_record(records[0])

            torch.testing.assert_close(first, repeated)
            torch.testing.assert_close(first, reloaded)
            self.assertEqual(wrapped_load.call_count, 3)

    def test_shuffled_indices_fall_back_to_different_state_within_field_task(self) -> None:
        records = [
            {
                **_record(f"state_{state}", "point", "Vx", f"question_{state}"),
                "sample_index": 0,
            }
            for state in range(4)
        ]
        dataset = TensorReadoutQADataset.__new__(TensorReadoutQADataset)
        dataset.records = records

        selected = dataset._build_random_different_indices(seed=5)

        for source, candidate_index in zip(records, selected):
            candidate = records[candidate_index]
            self.assertEqual(candidate["field"], source["field"])
            self.assertEqual(candidate["task_type"], source["task_type"])
            self.assertNotEqual(candidate["state_ref"], source["state_ref"])

    def test_grouped_sampler_preserves_groups_and_equalizes_rank_steps(self) -> None:
        records = [
            _record(f"state_{state}", "point", "Vx", f"question_{variant}")
            for state in range(7)
            for variant in range(3)
        ]
        dataset = SimpleNamespace(records=records)
        rank_batches = [
            list(
                StateTaskGroupedBatchSampler(
                    dataset=dataset,
                    batch_size=3,
                    questions_per_group=3,
                    seed=17,
                    rank=rank,
                    num_replicas=4,
                )
            )
            for rank in range(4)
        ]

        self.assertEqual([len(batches) for batches in rank_batches], [2, 2, 2, 2])
        flattened = [index for batches in rank_batches for batch in batches for index in batch]
        self.assertEqual(set(flattened), set(range(len(records))))
        self.assertEqual(len(flattened) - len(records), 3)
        for batches in rank_batches:
            for batch in batches:
                keys = {
                    (records[index]["state_ref"], records[index]["task_type"])
                    for index in batch
                }
                self.assertEqual(len(keys), 1)

    def test_explicit_matched_sampler_pads_only_complete_atomic_groups(self) -> None:
        records = []
        for group_index in range(7):
            for member_index in range(3):
                record = _record(
                    f"state_{group_index}",
                    "normalized_point_value",
                    "density",
                    f"question_{member_index}",
                )
                record["matched_group"] = {
                    "format": MATCHED_GROUP_FORMAT,
                    "batch_group_id": f"group_{group_index}",
                    "batch_group_size": 3,
                    "batch_member_index": member_index,
                }
                records.append(record)
        dataset = SimpleNamespace(records=records)
        rank_batches = [
            list(
                StateTaskGroupedBatchSampler(
                    dataset=dataset,
                    batch_size=3,
                    questions_per_group=3,
                    seed=19,
                    rank=rank,
                    num_replicas=4,
                )
            )
            for rank in range(4)
        ]

        self.assertEqual([len(batches) for batches in rank_batches], [2, 2, 2, 2])
        flattened = [index for batches in rank_batches for batch in batches for index in batch]
        self.assertEqual(set(flattened), set(range(len(records))))
        self.assertEqual(len(flattened) - len(records), 3)
        for batches in rank_batches:
            for batch in batches:
                specs = [records[index]["matched_group"] for index in batch]
                self.assertEqual(len({spec["batch_group_id"] for spec in specs}), 1)
                self.assertEqual(
                    sorted(spec["batch_member_index"] for spec in specs),
                    [0, 1, 2],
                )

    def test_exact_eval_sampler_never_pads_or_repeats(self) -> None:
        dataset = list(range(10))
        shards = [
            list(ExactDistributedEvalSampler(dataset, rank=rank, num_replicas=3))
            for rank in range(3)
        ]

        flattened = [index for shard in shards for index in shard]
        self.assertEqual(sorted(flattened), list(range(10)))
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual([len(shard) for shard in shards], [4, 3, 3])


class TestQuestionConditionedAdapter(unittest.TestCase):
    def test_selective_answer_statistics_aligns_first_target_and_retains_gradient(self) -> None:
        class IdentityDecoder(nn.Module):
            def forward(self, inputs_embeds, **_kwargs):
                return SimpleNamespace(last_hidden_state=inputs_embeds)

        class FakeCausalLM(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.decoder = IdentityDecoder()
                self.output = nn.Linear(3, 5, bias=False)

            def get_decoder(self):
                return self.decoder

            def get_output_embeddings(self):
                return self.output

        torch.manual_seed(9)
        llm = FakeCausalLM()
        inputs = torch.randn(2, 5, 3, requires_grad=True)
        labels = torch.tensor(
            [
                [-100, -100, 1, 4, -100],
                [-100, 2, 4, -100, -100],
            ]
        )
        attention = torch.ones(2, 5, dtype=torch.long)

        sequence_nll, counts, first_logits = selective_answer_statistics(
            llm=llm,
            inputs_embeds=inputs,
            attention_mask=attention,
            labels=labels,
            return_first_logits=True,
        )

        self.assertIsNotNone(first_logits)
        assert first_logits is not None
        expected_first = torch.stack([llm.output(inputs[0, 1]), llm.output(inputs[1, 0])])
        torch.testing.assert_close(first_logits, expected_first)
        self.assertEqual(counts.tolist(), [2, 2])
        expected_nll = torch.stack(
            [
                F.cross_entropy(llm.output(inputs[0, 1:3]), torch.tensor([1, 4]), reduction="sum"),
                F.cross_entropy(llm.output(inputs[1, 0:2]), torch.tensor([2, 4]), reduction="sum"),
            ]
        )
        torch.testing.assert_close(sequence_nll, expected_nll)

        (sequence_nll.sum() + first_logits.sum() * 0.01).backward()
        self.assertIsNotNone(inputs.grad)
        self.assertGreater(float(inputs.grad.abs().sum().item()), 0.0)

    def test_single_token_choice_ids_use_space_prefixed_answer_tokens(self) -> None:
        class FakeTokenizer:
            ids = {" A": 11, " B": 12, " C": 13, " D": 14}

            def __call__(self, text, **_kwargs):
                return {"input_ids": [self.ids[text]]}

        records = [
            {"answer": "C", "choices": ["A", "B", "C", "D"]},
            {"answer": "B", "choices": ["A", "B"]},
        ]

        result = single_token_choice_ids(records, FakeTokenizer())

        self.assertEqual(result, ([[11, 12, 13, 14], [11, 12]], [2, 1]))

    def test_frozen_llm_mode_keeps_checkpoint_training_deterministic(self) -> None:
        model = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.5))
        model.is_gradient_checkpointing = True

        set_frozen_llm_execution_mode(model, checkpoint_training=True)

        self.assertTrue(model.training)
        self.assertFalse(model[1].training)
        set_frozen_llm_execution_mode(model, checkpoint_training=False)
        self.assertFalse(model.training)

    def test_choice_ce_candidate_chunking_preserves_scores_and_order(self) -> None:
        records = [
            {"answer": "B", "choices": ["A", "B", "C", "D"]},
            {"answer": "D", "choices": ["A", "B", "C", "D"]},
        ]
        latent = torch.randn(2, 3, 2, 2)
        args = SimpleNamespace(
            train_choice_batch_size=3,
            max_prompt_tokens=32,
            max_target_tokens=4,
            append_eos=True,
            prompt_template="task_specific",
            local_context_layer=2,
            choice_score="mean",
        )
        observed_chunk_sizes: list[int] = []

        def fake_answer_nll(**kwargs):
            answers = list(kwargs["answers"])
            observed_chunk_sizes.append(len(answers))
            values = torch.tensor(
                [float(ord(answer) - ord("A") + 1) for answer in answers],
                requires_grad=True,
            )
            counts = torch.ones(len(answers), dtype=torch.long)
            return values, counts

        with (
            mock.patch(
                "scripts.train_tensor_llm_adapter.contextual_adapter_soft_embeds",
                return_value=torch.zeros(2, 2, 4),
            ),
            mock.patch("scripts.train_tensor_llm_adapter.forward_answer_nll", side_effect=fake_answer_nll),
        ):
            chunked = _sequence_choice_ce_loss(
                llm=object(),
                adapter=object(),
                tokenizer=object(),
                records=records,
                latent_map=latent,
                device=torch.device("cpu"),
                args=args,
            )

        self.assertEqual(observed_chunk_sizes, [3, 3, 2])
        candidate_scores = torch.tensor([-1.0, -2.0, -3.0, -4.0])
        expected_positive = torch.stack(
            [
                F.cross_entropy(candidate_scores.unsqueeze(0), torch.tensor([1])),
                F.cross_entropy(candidate_scores.unsqueeze(0), torch.tensor([3])),
            ]
        )
        torch.testing.assert_close(chunked[2].detach(), expected_positive)
        self.assertEqual(chunked[4]["choice_accuracy"], 0.0)

    def test_single_token_grounding_nll_excludes_eos_and_full_vocabulary(self) -> None:
        class FakeTokenizer:
            ids = {" A": 1, " B": 2, " C": 3, " D": 4}

            def __call__(self, text, **_kwargs):
                return {"input_ids": [self.ids[text]]}

        class FakeLLM(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embeddings = nn.Embedding(8, 4)

            def get_input_embeddings(self):
                return self.embeddings

        records = [
            {"answer": "C", "choices": ["A", "B", "C", "D"]},
            {"answer": "B", "choices": ["A", "B"]},
        ]
        args = SimpleNamespace(
            max_prompt_tokens=8,
            max_target_tokens=2,
            append_eos=True,
            prompt_template="task_specific",
            local_context_layer=2,
            choice_score="mean",
        )
        first_logits = torch.zeros(2, 8, requires_grad=True)
        with torch.no_grad():
            first_logits[0, 1:5] = torch.tensor([0.0, 1.0, 3.0, -1.0])
            first_logits[1, 1:3] = torch.tensor([-2.0, 2.0])
        input_ids = torch.zeros(2, 3, dtype=torch.long)
        attention = torch.ones_like(input_ids)
        labels = torch.tensor([[-100, -100, 1], [-100, -100, 2]])

        with (
            mock.patch(
                "scripts.train_tensor_llm_adapter.build_text_tensors",
                return_value=(input_ids, attention, labels),
            ),
            mock.patch(
                "scripts.train_tensor_llm_adapter.contextual_adapter_soft_embeds",
                return_value=torch.zeros(2, 2, 4),
            ),
            mock.patch(
                "scripts.train_tensor_llm_adapter.selective_answer_statistics",
                return_value=(torch.tensor([100.0, 200.0]), torch.tensor([2, 2]), first_logits),
            ),
        ):
            loss, _answer_ce, per_record_nll, _soft, _metrics = (
                adapter_training.single_token_choice_ce_loss(
                    llm=FakeLLM(),
                    adapter=object(),
                    tokenizer=FakeTokenizer(),
                    records=records,
                    latent_map=torch.zeros(2, 1, 1, 1),
                    device=torch.device("cpu"),
                    args=args,
                )
            )

        expected = torch.stack(
            [
                F.cross_entropy(first_logits[0, 1:5].unsqueeze(0), torch.tensor([2])),
                F.cross_entropy(first_logits[1, 1:3].unsqueeze(0), torch.tensor([1])),
            ]
        )
        torch.testing.assert_close(per_record_nll, expected)
        torch.testing.assert_close(loss, expected.mean())
        self.assertNotEqual(per_record_nll.tolist(), [50.0, 100.0])

    def test_spatial_position_encoding_is_deterministic_finite_and_row_major(self) -> None:
        first = sinusoidal_2d_position_encoding(3, 4, 16)
        second = sinusoidal_2d_position_encoding(3, 4, 16)

        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
        self.assertEqual(tuple(first.shape), (1, 12, 16))
        self.assertTrue(torch.isfinite(first).all())
        self.assertFalse(torch.equal(first[:, 0], first[:, 1]))
        self.assertFalse(torch.equal(first[:, 0], first[:, 4]))

    def test_spatial_adapter_has_one_row_major_token_per_latent_position(self) -> None:
        torch.manual_seed(11)
        adapter = TensorPatchAlignmentAdapter(
            latent_channels=3,
            latent_grid=(2, 3),
            adapter_dim=16,
            projection_dim=24,
            dropout=0.0,
            adapter_type="spatial_transformer",
            query_tokens=6,
            adapter_layers=1,
            adapter_heads=4,
            soft_prompt_scale=0.0,
        ).eval()
        latent = torch.zeros(1, 3, 2, 3)
        changed = latent.clone()
        changed[0, :, 1, 1] = torch.tensor([1.0, -2.0, 3.0])

        base_states, base_local = adapter.spatial_input_states(latent)
        changed_states, changed_local = adapter.spatial_input_states(changed)
        state_changes = (changed_states - base_states).abs().sum(dim=-1).squeeze(0)
        local_changes = (changed_local - base_local).abs().sum(dim=-1).squeeze(0)

        self.assertEqual(torch.nonzero(state_changes > 0, as_tuple=False).flatten().tolist(), [4])
        self.assertEqual(torch.nonzero(local_changes > 0, as_tuple=False).flatten().tolist(), [4])
        self.assertEqual(tuple(adapter.forward_soft_prompts(latent).shape), (1, 6, 24))

    def test_spatial_adapter_rejects_token_grid_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "one output token per latent-grid position"):
            TensorPatchAlignmentAdapter(
                latent_channels=3,
                latent_grid=(2, 3),
                adapter_dim=16,
                projection_dim=24,
                dropout=0.0,
                adapter_type="spatial_transformer",
                query_tokens=5,
                adapter_layers=1,
                adapter_heads=4,
                soft_prompt_scale=0.0,
            )

    def test_spatial_adapter_parameter_metrics_are_read_only_scalars(self) -> None:
        adapter = TensorPatchAlignmentAdapter(
            latent_channels=3,
            latent_grid=(2, 2),
            adapter_dim=16,
            projection_dim=24,
            dropout=0.0,
            adapter_type="spatial_transformer",
            query_tokens=4,
            adapter_layers=1,
            adapter_heads=4,
            soft_prompt_scale=0.05,
        )

        metrics = alignment_adapter_parameter_metrics(adapter)
        parameter_names = dict(adapter.named_parameters())
        buffer_names = dict(adapter.named_buffers())

        self.assertEqual(metrics, {"spatial_pos_scale": 1.0, "local_residual_scale": 1.0})
        self.assertNotIn("spatial_pos_scale", parameter_names)
        self.assertNotIn("local_residual_scale", parameter_names)
        self.assertIn("spatial_pos_scale", buffer_names)
        self.assertIn("local_residual_scale", buffer_names)

        adapter.capture_spatial_path_metrics = True
        adapter.forward_soft_prompts(torch.randn(2, 3, 2, 2))
        path_metrics = alignment_adapter_path_metrics(adapter)
        self.assertGreater(path_metrics["spatial_position_to_content_rms_ratio"], 0.0)
        self.assertGreater(path_metrics["local_residual_to_context_rms_ratio"], 0.0)

    def test_spatial_adapter_resets_legacy_trainable_scales_when_loading(self) -> None:
        kwargs = {
            "latent_channels": 3,
            "latent_grid": (2, 2),
            "adapter_dim": 16,
            "projection_dim": 24,
            "dropout": 0.0,
            "adapter_type": "spatial_transformer",
            "query_tokens": 4,
            "adapter_layers": 1,
            "adapter_heads": 4,
            "soft_prompt_scale": 0.05,
        }
        source = TensorPatchAlignmentAdapter(**kwargs)
        legacy_state = source.state_dict()
        legacy_state["spatial_pos_scale"] = torch.tensor(0.2)
        legacy_state["local_residual_scale"] = torch.tensor(0.3)
        restored = TensorPatchAlignmentAdapter(**kwargs)

        restored.load_state_dict(legacy_state, strict=True)

        self.assertEqual(float(restored.spatial_pos_scale), 1.0)
        self.assertEqual(float(restored.local_residual_scale), 1.0)

    def test_value_preserving_ae_keeps_exact_input_at_each_latent_position(self) -> None:
        model = ConvTokenAutoencoder2D(
            {
                "model": {
                    "input_size": [4, 4],
                    "in_channels": 1,
                    "out_channels": 1,
                    "base_channels": 4,
                    "channel_multipliers": [],
                    "num_res_blocks": 0,
                    "latent_dim": 3,
                    "latent_grid": [4, 4],
                    "dropout": 0.0,
                    "norm": "identity",
                    "activation": "gelu",
                    "output_activation": "identity",
                    "preserve_input_channels": True,
                }
            }
        )
        inputs = torch.randn(2, 1, 4, 4)
        latent = model.encode(inputs)["latent_map"]

        self.assertEqual(tuple(latent.shape), (2, 3, 4, 4))
        torch.testing.assert_close(latent[:, :1], inputs, rtol=0.0, atol=0.0)

    def test_one_based_question_coordinates_map_to_zero_based_structured_features(self) -> None:
        one_based = {
            "task_type": "normalized_point_value",
            "question": "Read row 1, column 16.",
            "metadata": {"grid_shape": [16, 16], "coordinate_origin": 1},
            "choices": ["A", "B", "C", "D"],
        }
        zero_based = {
            **one_based,
            "question": "Read row 0, column 15.",
            "metadata": {"grid_shape": [16, 16], "coordinate_origin": 0},
        }

        self.assertEqual(
            structured_query_features_for_record(one_based),
            structured_query_features_for_record(zero_based),
        )

    def test_numeric_task_instructions_match_standardized_encoder_input(self) -> None:
        normalized = task_specific_instruction({"task_type": "normalized_point_value"})
        raw = task_specific_instruction({"task_type": "raw_point_value_with_stats"})

        self.assertIn("read the standardized value z directly", normalized)
        self.assertIn("x = mean + scale * z", raw)

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

    def test_grounding_swaps_skip_same_answer_pairs(self) -> None:
        records = [
            {**_record("s1", "point", "Vx", "question one"), "answer": "A"},
            {**_record("s1", "point", "Vx", "question two"), "answer": "A"},
            {**_record("s1", "point", "Vx", "question three"), "answer": "B"},
        ]

        owners, swapped = same_state_question_swap_indices(
            records,
            require_different_answers=True,
        )

        self.assertEqual(owners, [0, 1, 2])
        self.assertEqual(swapped, [2, 2, 0])
        self.assertTrue(
            all(
                records[owner]["answer"] != records[source]["answer"]
                for owner, source in zip(owners, swapped)
            )
        )

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

        expected = conditioned.backbone.forward_soft_prompts(latent)
        actual = conditioned(latent, question, mask, structured_query=None)

        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_zero_text_gate_preserves_inherited_spatial_output(self) -> None:
        torch.manual_seed(13)
        aligned = TensorPatchAlignmentAdapter(
            latent_channels=3,
            latent_grid=(2, 2),
            adapter_dim=16,
            projection_dim=24,
            dropout=0.0,
            adapter_type="spatial_transformer",
            query_tokens=4,
            adapter_layers=2,
            adapter_heads=4,
            soft_prompt_scale=0.05,
        ).eval()
        reloaded = TensorPatchAlignmentAdapter(
            latent_channels=3,
            latent_grid=(2, 2),
            adapter_dim=16,
            projection_dim=24,
            dropout=0.0,
            adapter_type="spatial_transformer",
            query_tokens=4,
            adapter_layers=2,
            adapter_heads=4,
            soft_prompt_scale=0.05,
        ).eval()
        reloaded.load_state_dict(aligned.state_dict(), strict=True)
        conditioned = ResidualQuestionConditionedAdapter(
            aligned_adapter=reloaded,
            llm_hidden_size=24,
            context_layers=(1, 2),
            adapter_heads=4,
            dropout=0.0,
            text_gate_init=0.0,
            residual_gate_init=0.1,
        ).eval()
        latent = torch.randn(3, 3, 2, 2)
        question = torch.randn(3, 2, 6, 24)
        mask = torch.ones(3, 6, dtype=torch.bool)

        expected = conditioned.backbone.forward_soft_prompts(latent)
        actual = conditioned(latent, question, mask, structured_query=None)

        torch.testing.assert_close(actual, expected, rtol=0.0, atol=3.0e-8)

    def test_grounded_spatial_reader_has_no_trainable_unconditioned_path(self) -> None:
        torch.manual_seed(23)
        aligned = TensorPatchAlignmentAdapter(
            latent_channels=3,
            latent_grid=(2, 2),
            adapter_dim=16,
            projection_dim=24,
            dropout=0.0,
            adapter_type="spatial_transformer",
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
            text_gate_init=1.0,
            residual_gate_init=1.0,
            freeze_backbone=True,
            text_gate_trainable=False,
            residual_gate_trainable=False,
            zero_init_text_attention=True,
        ).train()
        latent = torch.randn(2, 3, 2, 2)
        question = torch.randn(2, 2, 5, 24)
        mask = torch.ones(2, 5, dtype=torch.bool)

        self.assertFalse(any(parameter.requires_grad for parameter in conditioned.backbone.parameters()))
        self.assertFalse(conditioned.backbone.training)
        self.assertFalse(conditioned.gate.requires_grad)
        self.assertTrue(all(not block.gate.requires_grad for block in conditioned.text_blocks))
        self.assertTrue(
            all(
                int(torch.count_nonzero(block.attention.out_proj.weight).item()) == 0
                for block in conditioned.text_blocks
            )
        )
        torch.testing.assert_close(
            conditioned(latent, question, mask, structured_query=None),
            conditioned.backbone.forward_soft_prompts(latent),
            rtol=0.0,
            atol=3.0e-8,
        )

        output = conditioned(latent, question, mask, structured_query=None)
        output.sum().backward()
        self.assertTrue(all(parameter.grad is None for parameter in conditioned.backbone.parameters()))
        self.assertGreater(
            float(conditioned.text_blocks[0].attention.out_proj.weight.grad.abs().sum().item()),
            0.0,
        )

    def test_global_only_baseline_bypasses_question_conditioned_branch(self) -> None:
        torch.manual_seed(29)
        aligned = TensorPatchAlignmentAdapter(
            latent_channels=3,
            latent_grid=(2, 2),
            adapter_dim=16,
            projection_dim=24,
            dropout=0.0,
            adapter_type="spatial_transformer",
            query_tokens=4,
            adapter_layers=2,
            adapter_heads=4,
            soft_prompt_scale=0.05,
        )
        local = ResidualQuestionConditionedAdapter(
            aligned_adapter=aligned,
            llm_hidden_size=24,
            context_layers=(1, 2),
            adapter_heads=4,
            dropout=0.0,
            text_gate_init=1.0,
            residual_gate_init=1.0,
            freeze_backbone=True,
            text_gate_trainable=False,
            residual_gate_trainable=False,
            zero_init_text_attention=True,
        )
        adapter = HybridGlobalLocalAdapter(
            global_adapter=aligned,
            local_adapter=local,
            freeze_global=True,
            combine_mode="residual",
        )
        latent = torch.randn(2, 3, 2, 2)
        text_embeds = torch.randn(2, 5, 24)

        with mock.patch.object(
            local,
            "forward",
            side_effect=AssertionError("global_only must not execute the local branch"),
        ):
            actual = adapter_training.adapter_soft_embeds(
                adapter=adapter,
                latent_map=latent,
                text_embeds=text_embeds,
                question_embeds=None,
                question_mask=None,
                records=None,
                mode="global_only",
            )

        torch.testing.assert_close(
            actual,
            aligned.forward_soft_prompts(latent).to(dtype=text_embeds.dtype),
            rtol=0.0,
            atol=0.0,
        )

    def test_direct_spatial_stage2_is_exactly_the_stage1_soft_prompt_path(self) -> None:
        torch.manual_seed(31)
        aligned = TensorPatchAlignmentAdapter(
            latent_channels=3,
            latent_grid=(2, 2),
            adapter_dim=16,
            projection_dim=24,
            dropout=0.0,
            adapter_type="spatial_transformer",
            query_tokens=4,
            adapter_layers=2,
            adapter_heads=4,
            soft_prompt_scale=0.05,
        )
        latent = torch.randn(2, 3, 2, 2)
        text_embeds = torch.randn(2, 7, 24)
        question_mask = torch.ones(2, 7, dtype=torch.bool)

        actual = adapter_training.adapter_soft_embeds(
            adapter=aligned,
            latent_map=latent,
            text_embeds=text_embeds,
            question_embeds=text_embeds,
            question_mask=question_mask,
            records=[{"question": "first"}, {"question": "second"}],
            mode="correct",
        )

        torch.testing.assert_close(
            actual,
            aligned.forward_soft_prompts(latent).to(dtype=text_embeds.dtype),
            rtol=0.0,
            atol=0.0,
        )

    def test_spatial_stage1_checkpoint_rebuilds_strictly_for_downstream(self) -> None:
        torch.manual_seed(17)
        aligned = TensorPatchAlignmentAdapter(
            latent_channels=3,
            latent_grid=(2, 2),
            adapter_dim=16,
            projection_dim=24,
            dropout=0.0,
            adapter_type="spatial_transformer",
            query_tokens=4,
            adapter_layers=2,
            adapter_heads=4,
            soft_prompt_scale=0.05,
        ).eval()
        checkpoint = {
            "args": {
                "adapter_type": "spatial_transformer",
                "adapter_dim": 16,
                "adapter_layers": 2,
                "adapter_heads": 4,
                "query_tokens": 4,
                "projection_dim": 24,
                "dropout": 0.0,
                "soft_prompt_scale": 0.05,
            },
            "adapter_state_dict": aligned.state_dict(),
        }
        latent = torch.randn(2, 3, 2, 2)

        rebuilt = adapter_from_checkpoint(checkpoint, latent_shape=(3, 2, 2), llm_hidden_size=24).eval()

        self.assertIsInstance(rebuilt, TensorPatchAlignmentAdapter)
        self.assertEqual(rebuilt.adapter_type, "spatial_transformer")
        torch.testing.assert_close(
            rebuilt.forward_soft_prompts(latent),
            aligned.forward_soft_prompts(latent),
            rtol=0.0,
            atol=0.0,
        )

    def test_spatial_residual_checkpoint_rebuilds_strictly(self) -> None:
        torch.manual_seed(19)
        aligned = TensorPatchAlignmentAdapter(
            latent_channels=3,
            latent_grid=(2, 2),
            adapter_dim=16,
            projection_dim=24,
            dropout=0.0,
            adapter_type="spatial_transformer",
            query_tokens=4,
            adapter_layers=2,
            adapter_heads=4,
            soft_prompt_scale=0.05,
        )
        local = ResidualQuestionConditionedAdapter(
            aligned_adapter=aligned,
            llm_hidden_size=24,
            context_layers=(1, 2),
            adapter_heads=4,
            dropout=0.0,
            text_gate_init=1.0,
            residual_gate_init=1.0,
            freeze_backbone=True,
            text_gate_trainable=False,
            residual_gate_trainable=False,
            zero_init_text_attention=True,
        )
        original = HybridGlobalLocalAdapter(
            global_adapter=aligned,
            local_adapter=local,
            freeze_global=True,
            combine_mode="residual",
        ).eval()
        checkpoint = {
            "args": {
                "adapter_architecture": "residual_question_adapter",
                "global_adapter_type": "spatial_transformer",
                "adapter_dim": 16,
                "adapter_layers": 2,
                "adapter_heads": 4,
                "projection_dim": 24,
                "dropout": 0.0,
                "soft_prompt_scale": 0.05,
                "local_context_layers": "1,2",
                "local_text_gate_init": 1.0,
                "local_gate_init": 1.0,
                "freeze_conditioned_backbone": True,
                "local_text_gate_trainable": False,
                "local_residual_gate_trainable": False,
                "zero_init_local_text_attention": True,
            },
            "adapter_state_dict": original.state_dict(),
        }
        latent = torch.randn(2, 3, 2, 2)
        question = torch.randn(2, 2, 5, 24)
        mask = torch.ones(2, 5, dtype=torch.bool)

        rebuilt = adapter_from_checkpoint(checkpoint, latent_shape=(3, 2, 2), llm_hidden_size=24).eval()

        self.assertIsInstance(rebuilt, HybridGlobalLocalAdapter)
        self.assertFalse(any(parameter.requires_grad for parameter in rebuilt.local_adapter.backbone.parameters()))
        self.assertFalse(rebuilt.local_adapter.gate.requires_grad)
        torch.testing.assert_close(
            rebuilt(latent, question, mask),
            original(latent, question, mask),
            rtol=0.0,
            atol=0.0,
        )

    def test_grounded_evidence_factorizes_row_and_column_routing(self) -> None:
        torch.manual_seed(41)
        local = GroundedEvidenceAdapter(
            latent_grid=(2, 3),
            llm_hidden_size=24,
            context_layers=(1, 2),
            adapter_dim=16,
            adapter_heads=4,
            dropout=0.0,
            evidence_tokens=2,
            soft_prompt_scale=0.05,
            gate_bias_init=-2.0,
        ).eval()
        aligned = torch.randn(2, 6, 24)
        question = torch.randn(2, 2, 7, 24)
        mask = torch.ones(2, 7, dtype=torch.bool)

        evidence = local.forward_from_aligned(aligned, question, mask, None)

        self.assertEqual(tuple(evidence.shape), (2, 2, 24))
        self.assertTrue(torch.isfinite(evidence).all())
        expected_logits = (
            local.last_row_logits.unsqueeze(-1) + local.last_col_logits.unsqueeze(-2)
        ).flatten(start_dim=-2)
        torch.testing.assert_close(local.last_routing_logits, expected_logits)
        torch.testing.assert_close(
            local.last_routing_weights.sum(dim=-1),
            torch.ones(2, 2),
        )
        selected = torch.einsum("brn,bnh->brh", local.last_routing_weights, aligned)
        self.assertTrue(torch.equal(evidence, torch.zeros_like(evidence)))

        with torch.no_grad():
            local.role_gate.bias.fill_(2.0)
        open_evidence = local.forward_from_aligned(aligned, question, mask, None)
        torch.testing.assert_close(open_evidence, selected, rtol=0.0, atol=0.0)

        global_adapter = TensorPatchAlignmentAdapter(
            latent_channels=3,
            latent_grid=(2, 3),
            adapter_dim=16,
            projection_dim=24,
            dropout=0.0,
            adapter_type="spatial_transformer",
            query_tokens=6,
            adapter_layers=1,
            adapter_heads=4,
            soft_prompt_scale=0.05,
        )
        geometry = grounded_reader_geometry_metrics(
            HybridGlobalLocalAdapter(global_adapter, local, freeze_global=True)
        )
        self.assertEqual(
            set(geometry),
            {
                "row",
                "col",
                "routing_logit_scale",
                "routing_logit_scale_log",
                "routing_logit_scale_saturated",
                "routing_logit_scale_log_margin_to_clamp",
                "text_layer_weights",
            },
        )
        self.assertGreater(geometry["routing_logit_scale"], 0.0)
        self.assertFalse(geometry["routing_logit_scale_saturated"])
        self.assertTrue(
            all(
                torch.isfinite(torch.tensor(value))
                for axis in ("row", "col")
                for value in geometry[axis].values()
            )
        )
        for axis in ("row", "col"):
            self.assertGreater(geometry[axis]["effective_rank"], 1.0)
            self.assertGreater(geometry[axis]["minimum_pairwise_l2"], 0.0)
            self.assertAlmostEqual(
                geometry[axis]["fixed_anchor_cosine_mean"],
                1.0,
                places=5,
            )

    def test_frozen_grounded_global_preserves_requested_latent_gradients(self) -> None:
        global_adapter = TensorPatchAlignmentAdapter(
            latent_channels=3,
            latent_grid=(2, 2),
            adapter_dim=16,
            projection_dim=24,
            dropout=0.0,
            adapter_type="spatial_transformer",
            query_tokens=4,
            adapter_layers=1,
            adapter_heads=4,
            soft_prompt_scale=0.05,
        )
        local = GroundedEvidenceAdapter(
            latent_grid=(2, 2),
            llm_hidden_size=24,
            context_layers=(1, 2),
            adapter_dim=16,
            adapter_heads=4,
            dropout=0.0,
            evidence_tokens=2,
            soft_prompt_scale=0.05,
            gate_bias_init=2.0,
        )
        adapter = HybridGlobalLocalAdapter(global_adapter, local, freeze_global=True)
        latent = torch.randn(1, 3, 2, 2, requires_grad=True)
        question = torch.randn(1, 2, 5, 24)
        mask = torch.ones(1, 5, dtype=torch.bool)

        global_prompts, _local_prompts, _combined = adapter.forward_components(
            latent,
            question_embeds=question,
            question_mask=mask,
        )
        global_prompts.square().sum().backward()

        self.assertIsNotNone(latent.grad)
        self.assertGreater(float(latent.grad.abs().sum()), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in global_adapter.parameters()))

    def test_grounded_routing_rejects_each_invalid_point_axis(self) -> None:
        global_adapter = TensorPatchAlignmentAdapter(
            latent_channels=3,
            latent_grid=(2, 3),
            adapter_dim=16,
            projection_dim=24,
            dropout=0.0,
            adapter_type="spatial_transformer",
            query_tokens=6,
            adapter_layers=1,
            adapter_heads=4,
            soft_prompt_scale=0.05,
        )
        local = GroundedEvidenceAdapter(
            latent_grid=(2, 3),
            llm_hidden_size=24,
            context_layers=(1, 2),
            adapter_dim=16,
            adapter_heads=4,
            dropout=0.0,
            evidence_tokens=2,
            soft_prompt_scale=0.05,
            gate_bias_init=-2.0,
        )
        adapter = HybridGlobalLocalAdapter(global_adapter, local, freeze_global=True)
        local.forward_from_aligned(
            torch.randn(1, 6, 24),
            torch.randn(1, 2, 5, 24),
            torch.ones(1, 5, dtype=torch.bool),
            None,
        )

        _loss, _gate_loss, metrics = grounded_routing_loss(
            adapter,
            [
                {
                    "task_type": "normalized_point_value",
                    "grounding_target": {
                        "type": "point",
                        "row": 1,
                        "col": 2,
                        "coordinate_origin": 0,
                    },
                }
            ],
        )
        self.assertGreaterEqual(metrics["routing_normalized_entropy"], 0.0)
        self.assertLessEqual(metrics["routing_normalized_entropy"], 1.0)

        invalid_specs = (
            {"type": "point", "row": -1, "col": 3, "coordinate_origin": 0},
            {"type": "point", "row": 0, "col": 3, "coordinate_origin": 0},
        )
        for spec in invalid_specs:
            with self.subTest(spec=spec), self.assertRaisesRegex(ValueError, "exceeds grid"):
                grounded_routing_loss(
                    adapter,
                    [{"task_type": "normalized_point_value", "grounding_target": spec}],
                )

    def test_grounded_routing_loss_weights_records_not_active_roles(self) -> None:
        global_adapter = TensorPatchAlignmentAdapter(
            latent_channels=3,
            latent_grid=(2, 2),
            adapter_dim=16,
            projection_dim=24,
            dropout=0.0,
            adapter_type="spatial_transformer",
            query_tokens=4,
            adapter_layers=1,
            adapter_heads=4,
            soft_prompt_scale=0.05,
        )
        local = GroundedEvidenceAdapter(
            latent_grid=(2, 2),
            llm_hidden_size=24,
            context_layers=(1, 2),
            adapter_dim=16,
            adapter_heads=4,
            dropout=0.0,
            evidence_tokens=2,
            soft_prompt_scale=0.05,
            gate_bias_init=-2.0,
        )
        adapter = HybridGlobalLocalAdapter(global_adapter, local, freeze_global=True)
        logits = torch.tensor(
            [
                [[3.0, 0.0, -1.0, -2.0], [0.0, 0.0, 0.0, 0.0]],
                [[-2.0, -1.0, 0.0, 3.0], [4.0, 1.0, 0.0, -1.0]],
            ],
            requires_grad=True,
        )
        local.last_routing_logits = logits
        local.last_row_logits = logits.reshape(2, 2, 2, 2).logsumexp(dim=-1)
        local.last_col_logits = logits.reshape(2, 2, 2, 2).logsumexp(dim=-2)
        local.last_role_gate_logits = torch.zeros(2, 2, requires_grad=True)
        records = [
            {
                "task_type": "normalized_point_value",
                "grounding_target": {
                    "type": "point",
                    "row": 0,
                    "col": 0,
                    "coordinate_origin": 0,
                },
            },
            {
                "task_type": "point_compare",
                "grounding_target": {
                    "type": "point_pair",
                    "a": [1, 1],
                    "b": [0, 1],
                    "coordinate_origin": 0,
                },
            },
        ]

        routing_loss, _gate_loss, _metrics = grounded_routing_loss(adapter, records)
        log_probs = F.log_softmax(logits, dim=-1)
        point_loss = -log_probs[0, 0, 0]
        pair_loss = (-log_probs[1, 0, 3] - log_probs[1, 1, 1]) / 2.0
        expected = (point_loss + pair_loss) / 2.0
        active_role_mean = (
            point_loss - log_probs[1, 0, 3] - log_probs[1, 1, 1]
        ) / 3.0

        torch.testing.assert_close(routing_loss, expected)
        self.assertFalse(torch.isclose(routing_loss, active_role_mean).item())

    def test_grounded_query_contract_rejects_task_type_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires query_spec.type='point'"):
            grounding_query_spec_for_record(
                {
                    "qa_id": "bad-query",
                    "task_type": "normalized_point_value",
                    "grounding_target": {
                        "type": "point_pair",
                        "a": [0, 0],
                        "b": [1, 1],
                        "coordinate_origin": 0,
                    },
                }
            )

    def test_routing_only_validation_skips_answer_scoring(self) -> None:
        global_adapter = TensorPatchAlignmentAdapter(
            latent_channels=3,
            latent_grid=(2, 2),
            adapter_dim=16,
            projection_dim=24,
            dropout=0.0,
            adapter_type="spatial_transformer",
            query_tokens=4,
            adapter_layers=1,
            adapter_heads=4,
            soft_prompt_scale=0.05,
        )
        local = GroundedEvidenceAdapter(
            latent_grid=(2, 2),
            llm_hidden_size=24,
            context_layers=(1, 2),
            adapter_dim=16,
            adapter_heads=4,
            dropout=0.0,
            evidence_tokens=2,
            soft_prompt_scale=0.05,
            gate_bias_init=-2.0,
        )
        adapter = HybridGlobalLocalAdapter(global_adapter, local, freeze_global=True)
        records = [
            {
                "task_type": "normalized_point_value",
                "grounding_target": {
                    "type": "point",
                    "row": index,
                    "col": index,
                    "coordinate_origin": 0,
                },
            }
            for index in range(2)
        ]

        class RoutingDataset(torch.utils.data.Dataset):
            def __init__(self) -> None:
                self.records = records

            def __len__(self) -> int:
                return len(self.records)

            def __getitem__(self, index: int) -> dict[str, object]:
                return {
                    "index": index,
                    "record": self.records[index],
                    "latent_map": torch.randn(3, 2, 2),
                }

        def fake_soft_embeds(**kwargs):
            batch_size = len(kwargs["records"])
            return kwargs["adapter"](
                kwargs["latent_map"],
                torch.randn(batch_size, 2, 5, 24),
                torch.ones(batch_size, 5, dtype=torch.bool),
                structured_query=None,
            )

        args = SimpleNamespace(
            eval_batch_size=2,
            num_workers=0,
            console_progress=False,
            max_prompt_tokens=32,
            local_context_layer=2,
            prompt_template="task_specific",
        )
        with mock.patch.object(
            adapter_training,
            "contextual_adapter_soft_embeds",
            side_effect=fake_soft_embeds,
        ), mock.patch.object(
            adapter_training,
            "collect_candidate_scores",
            side_effect=AssertionError("answer scorer must stay off"),
        ):
            metrics = evaluate_choice_accuracy(
                llm=nn.Module(),
                adapter=adapter,
                tokenizer=object(),
                dataset=RoutingDataset(),
                device=torch.device("cpu"),
                args=args,
                baseline_modes=["correct"],
                routing_only=True,
            )

        self.assertEqual(metrics["evaluation_mode"], "routing_only_shallow_qwen")
        self.assertEqual(metrics["correct"]["routing"]["active_roles"], 2)

    def test_routing_only_backward_keeps_every_reader_parameter_in_graph(self) -> None:
        torch.manual_seed(43)
        global_adapter = TensorPatchAlignmentAdapter(
            latent_channels=3,
            latent_grid=(2, 2),
            adapter_dim=16,
            projection_dim=24,
            dropout=0.0,
            adapter_type="spatial_transformer",
            query_tokens=4,
            adapter_layers=1,
            adapter_heads=4,
            soft_prompt_scale=0.05,
        )
        local = GroundedEvidenceAdapter(
            latent_grid=(2, 2),
            llm_hidden_size=24,
            context_layers=(1, 2),
            adapter_dim=16,
            adapter_heads=4,
            dropout=0.0,
            evidence_tokens=2,
            soft_prompt_scale=0.05,
            gate_bias_init=-2.0,
        )
        adapter = HybridGlobalLocalAdapter(global_adapter, local, freeze_global=True)
        question_context = (
            torch.randn(2, 2, 6, 24),
            torch.ones(2, 6, dtype=torch.bool),
        )
        records = [
            {
                "task_type": "normalized_point_value",
                "grounding_target": {
                    "type": "point",
                    "row": index,
                    "col": index,
                    "coordinate_origin": 0,
                }
            }
            for index in range(2)
        ]
        args = SimpleNamespace(
            max_prompt_tokens=32,
            local_context_layer=2,
            prompt_template="task_specific",
            ce_loss_weight=0.02,
            choice_ce_loss_weight=1.0,
            ranking_loss_weight=0.0,
            swapped_question_loss_weight=0.0,
            grounding_routing_loss_weight=1.0,
            grounding_gate_loss_weight=0.1,
            matched_group_loss_weight=0.2,
        )

        def fake_soft_embeds(**kwargs):
            context, context_mask = kwargs["precomputed_question_context"]
            return kwargs["adapter"](
                kwargs["latent_map"], context, context_mask, structured_query=None
            )

        with mock.patch.object(
            adapter_training,
            "contextual_adapter_question_context",
            return_value=question_context,
        ), mock.patch.object(
            adapter_training,
            "contextual_adapter_soft_embeds",
            side_effect=fake_soft_embeds,
        ):
            loss, parts = training_loss(
                llm=object(),
                adapter=adapter,
                tokenizer=object(),
                dataset=object(),
                batch={"records": records, "latent_map": torch.randn(2, 3, 2, 2)},
                device=torch.device("cpu"),
                args=args,
                routing_only=True,
            )
        loss.backward()

        self.assertGreater(parts["routing_loss"], 0.0)
        self.assertEqual(parts["choice_ce_loss"], 0.0)
        self.assertFalse(any("answer" in record for record in records))
        for name, parameter in local.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_joint_evidence_transform_starts_with_fresh_optimizer_state(self) -> None:
        global_adapter = TensorPatchAlignmentAdapter(
            latent_channels=3,
            latent_grid=(2, 2),
            adapter_dim=16,
            projection_dim=24,
            dropout=0.0,
            adapter_type="spatial_transformer",
            query_tokens=4,
            adapter_layers=1,
            adapter_heads=4,
            soft_prompt_scale=0.05,
        )
        local = GroundedEvidenceAdapter(
            latent_grid=(2, 2),
            llm_hidden_size=24,
            context_layers=(1, 2),
            adapter_dim=16,
            adapter_heads=4,
            dropout=0.0,
            evidence_tokens=2,
            soft_prompt_scale=0.05,
            gate_bias_init=-2.0,
        )
        adapter = HybridGlobalLocalAdapter(global_adapter, local, freeze_global=True)
        optimizer = torch.optim.AdamW(local.parameters(), lr=1.0e-3)
        for parameter in local.parameters():
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        transform_parameters = [
            *local.evidence_down.parameters(),
            *local.evidence_up.parameters(),
        ]
        self.assertTrue(all(parameter in optimizer.state for parameter in transform_parameters))

        cleared = reset_grounded_evidence_optimizer_state(optimizer, adapter)

        self.assertEqual(cleared, len(transform_parameters))
        self.assertTrue(all(parameter not in optimizer.state for parameter in transform_parameters))

    def test_matched_coordinate_margin_rewards_coordinate_specific_answers(self) -> None:
        choices = ["A", "B", "C", "D"]
        records = []
        for index, answer in enumerate(choices[:3]):
            records.append(
                {
                    "patch_id": "state",
                    "state_ref": "state",
                    "field": "density",
                    "sample_index": 0,
                    "time_index": 0,
                    "top_left": [0, 0],
                    "task_type": "normalized_point_value",
                    "choices": choices,
                    "answer": answer,
                    "matched_group": {
                        "margin_group_id": "group",
                        "margin_group_size": 3,
                        "margin_member_index": index,
                        "margin_kind": "coordinate_choice",
                        "option_set_sha256": "options",
                        "coordinate_set_id": "coordinates",
                    },
                }
            )
        specific = [
            F.log_softmax(
                torch.tensor([6.0 if column == row else -2.0 for column in range(4)]),
                dim=0,
            )
            for row in range(3)
        ]
        uniform = [F.log_softmax(torch.zeros(4), dim=0) for _ in range(3)]

        specific_loss, specific_metrics = matched_coordinate_group_loss(
            records, specific, margin=0.5
        )
        uniform_loss, uniform_metrics = matched_coordinate_group_loss(
            records, uniform, margin=0.5
        )

        self.assertEqual(float(specific_loss), 0.0)
        self.assertAlmostEqual(float(uniform_loss), 0.5)
        self.assertEqual(specific_metrics["matched_group_exact_accuracy"], 1.0)
        self.assertEqual(specific_metrics["matched_group_satisfaction"], 1.0)
        self.assertEqual(uniform_metrics["matched_group_satisfaction"], 0.0)

    def test_grounded_routing_warmup_audit_enforces_every_threshold(self) -> None:
        args = SimpleNamespace(
            grounding_warmup_min_cell_top1=0.90,
            grounding_warmup_min_cell_top5=0.98,
            grounding_warmup_min_axis_top1=0.95,
            grounding_warmup_min_target_mass=0.50,
            grounding_warmup_min_gate_accuracy=0.95,
        )
        routing = {
            "active_roles": 16,
            "top1_accuracy": 0.91,
            "top5_accuracy": 0.99,
            "row_top1_accuracy": 0.96,
            "col_top1_accuracy": 0.97,
            "target_mass": 0.55,
            "gate_accuracy": 0.96,
            "by_task": {
                task: {
                    "active_roles": 8,
                    "top1_accuracy": 0.91,
                    "target_mass": 0.55,
                }
                for task in (
                    "normalized_point_value",
                    "raw_point_value_with_stats",
                )
            },
        }

        passing = grounded_routing_warmup_audit({"correct": {"routing": routing}}, args)
        failing = grounded_routing_warmup_audit(
            {"correct": {"routing": {**routing, "top1_accuracy": 0.89}}}, args
        )
        raw_failure_routing = {
            **routing,
            "by_task": {
                **routing["by_task"],
                "raw_point_value_with_stats": {
                    **routing["by_task"]["raw_point_value_with_stats"],
                    "target_mass": 0.49,
                },
            },
        }
        raw_failing = grounded_routing_warmup_audit(
            {"correct": {"routing": raw_failure_routing}}, args
        )

        self.assertTrue(passing["passed"])
        self.assertFalse(failing["passed"])
        self.assertEqual(failing["failed"], ["cell_top1"])
        self.assertFalse(raw_failing["passed"])
        self.assertIn(
            "raw_point_value_with_stats.target_mass",
            raw_failing["failed"],
        )

    def test_grounded_checkpoint_rebuilds_strictly(self) -> None:
        torch.manual_seed(47)
        global_adapter = TensorPatchAlignmentAdapter(
            latent_channels=3,
            latent_grid=(2, 2),
            adapter_dim=16,
            projection_dim=24,
            dropout=0.0,
            adapter_type="spatial_transformer",
            query_tokens=4,
            adapter_layers=1,
            adapter_heads=4,
            soft_prompt_scale=0.05,
        )
        local = GroundedEvidenceAdapter(
            latent_grid=(2, 2),
            llm_hidden_size=24,
            context_layers=(1, 2),
            adapter_dim=16,
            adapter_heads=4,
            dropout=0.0,
            evidence_tokens=2,
            soft_prompt_scale=0.05,
            gate_bias_init=-2.0,
        )
        original = HybridGlobalLocalAdapter(
            global_adapter=global_adapter,
            local_adapter=local,
            freeze_global=True,
            combine_mode="concat",
        ).eval()
        checkpoint = {
            "args": {
                "adapter_architecture": "grounded_evidence_adapter",
                "global_adapter_type": "spatial_transformer",
                "adapter_dim": 16,
                "adapter_heads": 4,
                "adapter_layers": 1,
                "global_dropout": 0.0,
                "global_soft_prompt_scale": 0.05,
                "local_context_layers": "1,2",
                "dropout": 0.0,
                "soft_prompt_scale": 0.05,
                "grounded_gate_bias_init": -2.0,
            },
            "adapter_state_dict": original.state_dict(),
        }
        latent = torch.randn(2, 3, 2, 2)
        question = torch.randn(2, 2, 5, 24)
        mask = torch.ones(2, 5, dtype=torch.bool)

        rebuilt = adapter_from_checkpoint(
            checkpoint, latent_shape=(3, 2, 2), llm_hidden_size=24
        ).eval()

        self.assertIsInstance(rebuilt, HybridGlobalLocalAdapter)
        self.assertIsInstance(rebuilt.local_adapter, GroundedEvidenceAdapter)
        torch.testing.assert_close(
            rebuilt(latent, question, mask),
            original(latent, question, mask),
            rtol=0.0,
            atol=0.0,
        )

    def test_warmup_checkpoint_is_saved_validated_and_strictly_rebuilt(self) -> None:
        global_adapter = TensorPatchAlignmentAdapter(
            latent_channels=3,
            latent_grid=(2, 2),
            adapter_dim=16,
            projection_dim=24,
            dropout=0.0,
            adapter_type="spatial_transformer",
            query_tokens=4,
            adapter_layers=1,
            adapter_heads=4,
            soft_prompt_scale=0.05,
        )
        local = GroundedEvidenceAdapter(
            latent_grid=(2, 2),
            llm_hidden_size=24,
            context_layers=(1, 2),
            adapter_dim=16,
            adapter_heads=4,
            dropout=0.0,
            evidence_tokens=2,
            soft_prompt_scale=0.05,
            gate_bias_init=-2.0,
        )
        adapter = HybridGlobalLocalAdapter(
            global_adapter=global_adapter,
            local_adapter=local,
            freeze_global=True,
            combine_mode="concat",
        )
        args = SimpleNamespace(
            adapter_architecture="grounded_evidence_adapter",
            global_adapter_type="spatial_transformer",
            adapter_dim=16,
            adapter_heads=4,
            adapter_layers=1,
            global_dropout=0.0,
            global_soft_prompt_scale=0.05,
            local_context_layers="1,2",
            dropout=0.0,
            soft_prompt_scale=0.05,
            grounded_gate_bias_init=-2.0,
        )
        latent_contract = {"format": "test_latent_contract", "shape": [3, 2, 2]}

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter_routing_warmup.pt"
            with mock.patch.object(
                adapter_training,
                "validate_adapter_checkpoint_payload",
                wraps=validate_adapter_checkpoint_payload,
            ) as validate, mock.patch.object(
                adapter_training,
                "adapter_from_checkpoint",
                wraps=adapter_from_checkpoint,
            ) as rebuild:
                save_validate_and_rebuild_adapter_checkpoint(
                    path,
                    adapter=adapter,
                    args=args,
                    latent_shape=(3, 2, 2),
                    llm_hidden_size=24,
                    latent_contract=latent_contract,
                    metrics={"routing_warmup_audit": {"passed": True}},
                )

            self.assertTrue(path.is_file())
            validate.assert_called_once()
            rebuild.assert_called_once()
            saved = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(saved["checkpoint_type"], "tensor_llm_adapter")
            self.assertTrue(saved["metrics"]["routing_warmup_audit"]["passed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
