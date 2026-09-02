from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.train_tensor_qwen_cross_attention import (
    CHECKPOINT_TYPE,
    CHECKPOINT_VERSION,
    DenseTensorMemory,
    DirectFieldEncoder,
    FullGridSpatialBackbone,
    GatedTensorCrossAttention,
    RawHDF5TensorReadoutQADataset,
    audit_raw_hdf5_input_content,
    normalize_raw_patch_for_qa,
    parse_args,
    validate_checkpoint_contract,
    validate_relocated_qa_latent_contract,
)
from tensor_compression.models.compressors.conv_token_autoencoder_2d import (
    ConvTokenAutoencoder2D,
)
from tensor_compression.downstream.patch_qa_contract import (
    PATCH_LATENT_AUDIT_FORMAT,
    PATCH_LATENT_FORMAT,
    PATCH_QA_FORMAT,
    canonical_path,
    sha256_file,
    validate_patch_latent_payload,
)


RAW_NORMALIZATION = {
    "mode": "zscore",
    "scope": "channel",
    "stats_path": None,
    "clip_min": None,
    "clip_max": None,
}


def raw_record(
    *,
    state_ref: str,
    field: str,
    sample_index: int,
    time_index: int,
    top_left: tuple[int, int],
    raw_patch: torch.Tensor,
) -> dict[str, object]:
    patch = raw_patch.float()
    mean = patch.mean()
    std = patch.std(unbiased=False)
    return {
        "qa_id": f"{state_ref}_qa",
        "patch_id": state_ref,
        "state_ref": state_ref,
        "sample_index": int(sample_index),
        "time_index": int(time_index),
        "field": field,
        "top_left": [int(value) for value in top_left],
        "task_type": "normalized_point_value",
        "question": "Which option is closest?",
        "choices": ["A", "B", "C", "D"],
        "answer": "A",
        "metadata": {
            "field": field,
            "grid_shape": [int(patch.shape[-2]), int(patch.shape[-1])],
        },
        "latent_audit": {
            "format": PATCH_LATENT_AUDIT_FORMAT,
            "mean": float(mean.item()),
            "std": float(std.item()),
            "scale": float((std + 1.0e-6).item()),
        },
    }


def formal_metadata(declared_checkpoint: Path, checkpoint_sha256: str) -> dict[str, object]:
    return {
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
        "alignment_checkpoint": str(declared_checkpoint),
        "alignment_checkpoint_sha256": checkpoint_sha256,
    }


def dense_architecture(*, latent_channel_policy: str | None) -> dict[str, object]:
    architecture: dict[str, object] = {
        "format": "dense_tensor_memory_cross_attention_v1",
        "qwen_model": "Qwen/Qwen2.5-14B-Instruct",
        "llm_hidden_size": 16,
        "latent_shape": [8, 16, 16],
        "layers_1based": [8, 20, 32],
        "bridge_dim": 8,
        "heads": 2,
        "dropout": 0.0,
        "gate_init": 0.0,
        "value_fourier_bands": 4,
        "value_hidden_dim": 8,
        "freeze_spatial_backbone": True,
        "initializer": {"sha256": "1" * 64},
        "latent_contract": {"alignment_checkpoint_sha256": "2" * 64},
    }
    if latent_channel_policy is not None:
        architecture["latent_channel_policy"] = latent_channel_policy
    return architecture


class TestRelocatedLatentContract(unittest.TestCase):
    def test_accepts_relocated_checkpoint_with_identical_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configured = root / "new_mount" / "alignment_best.pt"
            configured.parent.mkdir()
            configured.write_bytes(b"immutable checkpoint contents")
            declared = root / "old_mount" / "alignment_best.pt"
            metadata = formal_metadata(declared, sha256_file(configured))

            contract, resolution = validate_relocated_qa_latent_contract(
                metadata,
                configured,
            )

            self.assertEqual(contract["alignment_checkpoint"], canonical_path(declared))
            self.assertEqual(contract["alignment_checkpoint_sha256"], sha256_file(configured))
            self.assertTrue(resolution["path_relocated"])
            self.assertEqual(resolution["declared_path"], canonical_path(declared))
            self.assertEqual(resolution["configured_path"], canonical_path(configured))
            self.assertEqual(resolution["identity"], "sha256")

            identity = {
                "patch_id": "density_s000000_t0000_r000_c000",
                "field": "density",
                "sample_index": 0,
                "time_index": 0,
                "top_left": [0, 0],
            }
            normalization = metadata["encoder_input_normalization"]
            payload = {
                "format": PATCH_LATENT_FORMAT,
                **identity,
                "alignment_checkpoint": str(declared),
                "alignment_checkpoint_sha256": sha256_file(configured),
                "encoder_input_normalization": normalization,
                "latent_map": torch.zeros(8, 16, 16, dtype=torch.float16),
            }
            latent = validate_patch_latent_payload(
                payload,
                path=root / "latent.pt",
                expected_identity=identity,
                expected_alignment_checkpoint=contract["alignment_checkpoint"],
                expected_alignment_sha256=contract["alignment_checkpoint_sha256"],
                expected_normalization=normalization,
                expected_shape=contract["latent_shape"],
                expected_storage_dtype=contract["storage_dtype"],
            )
            self.assertEqual(tuple(latent.shape), (8, 16, 16))

    def test_preserves_same_path_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "alignment_best.pt"
            checkpoint.write_bytes(b"immutable checkpoint contents")
            metadata = formal_metadata(checkpoint, sha256_file(checkpoint))

            contract, resolution = validate_relocated_qa_latent_contract(
                metadata,
                checkpoint,
            )

            self.assertEqual(contract["alignment_checkpoint"], canonical_path(checkpoint))
            self.assertFalse(resolution["path_relocated"])

    def test_rejects_relocated_checkpoint_with_different_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configured = root / "new_mount" / "alignment_best.pt"
            configured.parent.mkdir()
            configured.write_bytes(b"different checkpoint contents")
            declared = root / "old_mount" / "alignment_best.pt"
            metadata = formal_metadata(declared, "0" * 64)

            with self.assertRaisesRegex(
                ValueError,
                "changed after patch latents were generated",
            ):
                validate_relocated_qa_latent_contract(metadata, configured)


class TestLatentChannelPolicyContract(unittest.TestCase):
    def test_dense_resume_rejects_a_different_qwen_identity(self) -> None:
        checkpoint = {
            "checkpoint_type": CHECKPOINT_TYPE,
            "checkpoint_version": CHECKPOINT_VERSION,
            "architecture": dense_architecture(latent_channel_policy="all"),
        }
        expected = dense_architecture(latent_channel_policy="all")
        expected["qwen_model"] = "Qwen/Qwen2.5-7B-Instruct"

        with self.assertRaisesRegex(ValueError, "qwen_model_identity"):
            validate_checkpoint_contract(checkpoint, expected)

    def test_legacy_dense_checkpoint_defaults_to_all_channels(self) -> None:
        checkpoint = {
            "checkpoint_type": CHECKPOINT_TYPE,
            "checkpoint_version": CHECKPOINT_VERSION,
            "architecture": dense_architecture(latent_channel_policy=None),
        }

        validate_checkpoint_contract(
            checkpoint,
            dense_architecture(latent_channel_policy="all"),
        )

    def test_dense_resume_rejects_latent_channel_policy_change(self) -> None:
        checkpoint = {
            "checkpoint_type": CHECKPOINT_TYPE,
            "checkpoint_version": CHECKPOINT_VERSION,
            "architecture": dense_architecture(latent_channel_policy=None),
        }

        with self.assertRaisesRegex(ValueError, "latent_channel_policy"):
            validate_checkpoint_contract(
                checkpoint,
                dense_architecture(latent_channel_policy="value_only"),
            )


class TestDirectCrossRawInput(unittest.TestCase):
    def test_scratch_config_requires_no_stage1_direct_or_latent_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hdf5_path = root / "pdebench.hdf5"
            environment = {
                "FIELD_TO_LLM_ROOT": str(root / "experiment"),
                "PDEBENCH_HDF5": str(hdf5_path),
            }
            argv = [
                "train_tensor_qwen_cross_attention.py",
                "--config",
                str(
                    PROJECT_ROOT
                    / "configs"
                    / "field_to_llm_cross_attention_scratch.yaml"
                ),
            ]

            with patch.dict(os.environ, environment, clear=True), patch.object(
                sys, "argv", argv
            ):
                args = parse_args()

            self.assertEqual(args.input_source, "raw_hdf5")
            self.assertEqual(args.memory_init_mode, "scratch")
            self.assertEqual(args.hdf5_path, str(hdf5_path.resolve()))
            self.assertIsNone(args.latent_dir)
            self.assertIsNone(args.qa_alignment_checkpoint)
            self.assertIsNone(args.memory_init_checkpoint)
            self.assertFalse(args.freeze_spatial_backbone)
            self.assertTrue(args.field_encoder_trainable)
            self.assertEqual(args.raw_normalized_dtype, "float16")

    def test_normalize_raw_patch_matches_qa_builder_statistics(self) -> None:
        raw = torch.tensor([[[1.0, 2.0], [4.0, 8.0]]], dtype=torch.float32)
        record = raw_record(
            state_ref="density_s000000_t0000_r000_c000",
            field="density",
            sample_index=0,
            time_index=0,
            top_left=(0, 0),
            raw_patch=raw,
        )

        observed = normalize_raw_patch_for_qa(raw, record, RAW_NORMALIZATION)
        expected = (raw - raw.mean()) / (raw.std(unbiased=False) + 1.0e-6)

        torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)
        self.assertEqual(observed.dtype, torch.float32)
        self.assertTrue(observed.is_contiguous())

    def test_normalize_raw_patch_rejects_statistics_mismatch(self) -> None:
        raw = torch.tensor([[[1.0, 2.0], [4.0, 8.0]]], dtype=torch.float32)
        record = raw_record(
            state_ref="density_s000000_t0000_r000_c000",
            field="density",
            sample_index=0,
            time_index=0,
            top_left=(0, 0),
            raw_patch=raw,
        )
        record["latent_audit"] = {
            **record["latent_audit"],
            "mean": float(raw.mean().item()) + 0.25,
        }

        with self.assertRaisesRegex(ValueError, "does not match matched-QA provenance"):
            normalize_raw_patch_for_qa(raw, record, RAW_NORMALIZATION)

    def test_raw_hdf5_dataset_crops_caches_and_shuffles_states(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hdf5_path = root / "pdebench.hdf5"
            jsonl_path = root / "train.jsonl"
            values = torch.arange(2 * 1 * 4 * 4, dtype=torch.float32).reshape(
                2, 1, 4, 4
            )
            values[1, 0] = values[1, 0].square()
            with h5py.File(hdf5_path, "w") as handle:
                handle.create_dataset("density", data=values.numpy())

            first_raw = values[0, 0, 1:3, 1:3].unsqueeze(0)
            second_raw = values[1, 0, 0:2, 2:4].unsqueeze(0)
            records = [
                raw_record(
                    state_ref="density_s000000_t0000_r001_c001",
                    field="density",
                    sample_index=0,
                    time_index=0,
                    top_left=(1, 1),
                    raw_patch=first_raw,
                ),
                raw_record(
                    state_ref="density_s000001_t0000_r000_c002",
                    field="density",
                    sample_index=1,
                    time_index=0,
                    top_left=(0, 2),
                    raw_patch=second_raw,
                ),
            ]
            with jsonl_path.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record) + "\n")

            dataset = RawHDF5TensorReadoutQADataset(
                jsonl_path,
                hdf5_path=hdf5_path,
                patch_size=2,
                normalization=RAW_NORMALIZATION,
                normalized_dtype="float32",
                max_records=None,
                subset_mode="prefix",
                subset_seed=42,
                shuffle_seed=42,
                input_cache_size=2,
            )
            try:
                torch.testing.assert_close(
                    dataset._read_raw_patch(dataset.records[0]),
                    first_raw,
                )
                with patch.object(
                    dataset,
                    "_read_raw_patch",
                    wraps=dataset._read_raw_patch,
                ) as read_raw_patch:
                    first = dataset[0]["latent_map"]
                    repeated = dataset[0]["latent_map"]
                    self.assertEqual(read_raw_patch.call_count, 1)
                    self.assertIs(first, repeated)

                expected_first = (first_raw - first_raw.mean()) / (
                    first_raw.std(unbiased=False) + 1.0e-6
                )
                expected_second = (second_raw - second_raw.mean()) / (
                    second_raw.std(unbiased=False) + 1.0e-6
                )
                torch.testing.assert_close(first, expected_first)
                torch.testing.assert_close(
                    dataset.load_shuffled_latent(0),
                    expected_second,
                )
                self.assertFalse(torch.allclose(first, expected_second))
                self.assertEqual(dataset._random_different_indices[0], 1)
                content_audit = audit_raw_hdf5_input_content({"train": dataset})
                self.assertEqual(content_audit["format"], "normalized_qa_patch_content_v1")
                self.assertEqual(content_audit["splits"]["train"]["states"], 2)
                self.assertEqual(len(content_audit["sha256"]), 64)
            finally:
                dataset.close()

    def test_scratch_memory_has_complete_gradients_and_no_unused_prefix_modules(self) -> None:
        compressor = ConvTokenAutoencoder2D(
            {
                "model": {
                    "name": "conv_token_autoencoder_2d",
                    "input_size": [4, 4],
                    "in_channels": 1,
                    "out_channels": 1,
                    "base_channels": 4,
                    "channel_multipliers": [],
                    "num_res_blocks": 1,
                    "latent_dim": 3,
                    "latent_grid": [4, 4],
                    "preserve_input_channels": True,
                    "dropout": 0.0,
                    "norm": "group",
                    "activation": "gelu",
                    "output_activation": "identity",
                }
            }
        )
        field_encoder = DirectFieldEncoder(compressor)
        spatial_backbone = FullGridSpatialBackbone(
            latent_channels=3,
            latent_grid=(4, 4),
            adapter_dim=8,
            adapter_layers=1,
            adapter_heads=2,
            dropout=0.0,
        )
        memory = DenseTensorMemory(
            spatial_backbone=spatial_backbone,
            field_encoder=field_encoder,
            fourier_bands=1,
            value_hidden_dim=8,
            freeze_spatial_backbone=False,
        )
        field_input = torch.randn(2, 1, 4, 4)

        encoded = field_encoder(field_input)
        self.assertEqual(tuple(encoded.shape), (2, 3, 4, 4))
        torch.testing.assert_close(encoded[:, :1], field_input)
        self.assertFalse(hasattr(field_encoder, "decoder"))
        self.assertFalse(hasattr(spatial_backbone, "output"))
        bridge = GatedTensorCrossAttention(
            llm_dim=8,
            memory_dim=8,
            bridge_dim=8,
            heads=2,
            dropout=0.0,
            gate_init=0.1,
        )
        model = torch.nn.ModuleDict({"memory": memory, "bridge": bridge})
        parameter_names = [name for name, _parameter in model.named_parameters()]
        self.assertFalse(any("decoder" in name for name in parameter_names))
        self.assertFalse(any("output" in name for name in parameter_names))

        state = memory(field_input)
        self.assertEqual(tuple(state.content.shape), (2, 16, 8))
        self.assertEqual(tuple(state.value.shape), (2, 16, 8))
        bridge.bind_memory(state)
        bridged = bridge(torch.randn(2, 3, 8))
        content_weights = torch.linspace(
            0.5,
            1.5,
            state.content.numel(),
            dtype=state.content.dtype,
        ).reshape_as(state.content)
        value_weights = torch.linspace(
            -0.75,
            0.75,
            state.value.numel(),
            dtype=state.value.dtype,
        ).reshape_as(state.value)
        loss = (
            (state.content * content_weights).mean()
            + (state.value * value_weights).mean()
            + state.reconstruction_loss
            + bridged.square().mean()
        )
        loss.backward()

        missing_gradients = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        ]
        nonfinite_gradients = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
            and parameter.grad is not None
            and not bool(torch.isfinite(parameter.grad).all())
        ]
        self.assertEqual(missing_gradients, [])
        self.assertEqual(nonfinite_gradients, [])


class TestMachinePathOverrides(unittest.TestCase):
    def test_cross_attention_machine_paths_override_portable_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment = {
                "FIELD_TO_LLM_ROOT": str(root / "portable-root"),
                "FIELD_TO_LLM_ALIGNMENT_CHECKPOINT": str(root / "alignment_best.pt"),
                "FIELD_TO_LLM_DIRECT_CHECKPOINT": str(root / "adapter_best.pt"),
                "FIELD_TO_LLM_HF_HOME": str(root / "hf"),
                "FIELD_TO_LLM_RUNS_DIR": str(root / "runs"),
                "FIELD_TO_LLM_MODEL_DIR": str(root / "Qwen2.5-14B-Instruct"),
                "FIELD_TO_LLM_MATCHED_QA_DIR": str(root / "matched-qa"),
                "FIELD_TO_LLM_LATENT_DIR": str(root / "latents"),
            }
            argv = [
                "train_tensor_qwen_cross_attention.py",
                "--config",
                str(PROJECT_ROOT / "configs" / "field_to_llm_cross_attention.yaml"),
            ]

            with patch.dict(os.environ, environment, clear=True), patch.object(
                sys, "argv", argv
            ):
                args = parse_args()

            self.assertEqual(
                args.model_name_or_path,
                environment["FIELD_TO_LLM_MODEL_DIR"],
            )
            self.assertEqual(args.qa_dir, environment["FIELD_TO_LLM_MATCHED_QA_DIR"])
            self.assertEqual(args.latent_dir, environment["FIELD_TO_LLM_LATENT_DIR"])
            self.assertEqual(args.cache_dir, environment["FIELD_TO_LLM_HF_HOME"])
            self.assertEqual(args.hf_home, environment["FIELD_TO_LLM_HF_HOME"])
            self.assertEqual(args.output_root, environment["FIELD_TO_LLM_RUNS_DIR"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
