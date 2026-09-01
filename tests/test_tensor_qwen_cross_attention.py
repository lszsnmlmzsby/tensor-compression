from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.train_tensor_qwen_cross_attention import (
    CHECKPOINT_TYPE,
    CHECKPOINT_VERSION,
    parse_args,
    validate_checkpoint_contract,
    validate_relocated_qa_latent_contract,
)
from tensor_compression.downstream.patch_qa_contract import (
    PATCH_LATENT_AUDIT_FORMAT,
    PATCH_LATENT_FORMAT,
    PATCH_QA_FORMAT,
    canonical_path,
    sha256_file,
    validate_patch_latent_payload,
)


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
