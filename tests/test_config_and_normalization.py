from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tensor_compression.config import load_config
from tensor_compression.data.datasets.tensor_folder_2d import TensorFolder2DDataset


def _write_config(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


class TestConfigSynchronization(unittest.TestCase):
    def test_scales_latent_dim_with_channel_count_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            _write_config(
                config_path,
                {
                    "data": {
                        "dataset": {
                            "hdf5_dataset_keys": ["density", "pressure", "Vx", "Vy"],
                            "normalization": {"mode": "zscore", "scope": "channel"},
                        }
                    },
                    "model": {
                        "name": "conv_token_autoencoder_2d",
                        "input_size": [32, 32],
                        "latent_grid": [2, 2],
                        "channel_multipliers": [1, 2, 4, 8],
                        "base_channels": 8,
                        "num_res_blocks": 1,
                        "latent_dim": 128,
                        "latent_dim_base": 128,
                        "latent_dim_scale_with_channels": True,
                        "latent_dim_reference_channels": 1,
                        "latent_dim_round_to": 32,
                        "dropout": 0.0,
                        "norm": "group",
                        "activation": "gelu",
                        "output_activation": "identity",
                    },
                },
            )

            config = load_config(config_path, base_root=PROJECT_ROOT)

        self.assertEqual(config["model"]["in_channels"], 4)
        self.assertEqual(config["model"]["latent_dim_base"], 128)
        self.assertEqual(config["model"]["latent_dim"], 512)

    def test_keeps_base_latent_dim_for_single_channel_when_scaling_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            _write_config(
                config_path,
                {
                    "data": {
                        "dataset": {
                            "hdf5_dataset_key": "Vx",
                            "normalization": {"mode": "zscore", "scope": "channel"},
                        }
                    },
                    "model": {
                        "name": "conv_token_autoencoder_2d",
                        "input_size": [32, 32],
                        "latent_grid": [2, 2],
                        "channel_multipliers": [1, 2, 4, 8],
                        "base_channels": 8,
                        "num_res_blocks": 1,
                        "latent_dim": 128,
                        "latent_dim_scale_with_channels": True,
                        "latent_dim_reference_channels": 1,
                        "latent_dim_round_to": 32,
                        "dropout": 0.0,
                        "norm": "group",
                        "activation": "gelu",
                        "output_activation": "identity",
                    },
                },
            )

            config = load_config(config_path, base_root=PROJECT_ROOT)

        self.assertEqual(config["model"]["in_channels"], 1)
        self.assertEqual(config["model"]["latent_dim"], 128)

    def test_inferrs_channels_when_explicit_values_are_omitted(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            _write_config(
                config_path,
                {
                    "data": {
                        "dataset": {
                            "hdf5_dataset_keys": ["density", "pressure", "Vx", "Vy", "Vz"],
                            "normalization": {"mode": "zscore", "scope": "channel"},
                        }
                    },
                    "model": {
                        "name": "conv_token_autoencoder_2d",
                        "input_size": [32, 32],
                        "latent_grid": [2, 2],
                        "channel_multipliers": [1, 2, 4, 8],
                        "base_channels": 8,
                        "num_res_blocks": 1,
                        "latent_dim": 16,
                        "dropout": 0.0,
                        "norm": "group",
                        "activation": "gelu",
                        "output_activation": "identity",
                    },
                },
            )

            config = load_config(config_path, base_root=PROJECT_ROOT)

        self.assertEqual(config["data"]["dataset"]["channels"], 5)
        self.assertEqual(config["model"]["in_channels"], 5)
        self.assertEqual(config["model"]["out_channels"], 5)

    def test_syncs_model_channels_from_hdf5_dataset_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            _write_config(
                config_path,
                {
                    "data": {
                        "dataset": {
                            "channels": 4,
                            "hdf5_dataset_keys": ["density", "pressure", "Vx", "Vy"],
                            "normalization": {"mode": "none"},
                        }
                    },
                    "model": {
                        "name": "conv_token_autoencoder_2d",
                        "input_size": [32, 32],
                        "latent_grid": [2, 2],
                        "channel_multipliers": [1, 2, 4, 8],
                        "base_channels": 8,
                        "num_res_blocks": 1,
                        "latent_dim": 16,
                        "dropout": 0.0,
                        "norm": "group",
                        "activation": "gelu",
                        "output_activation": "identity",
                    },
                },
            )

            config = load_config(config_path, base_root=PROJECT_ROOT)

        self.assertEqual(config["data"]["dataset"]["channels"], 4)
        self.assertEqual(config["model"]["in_channels"], 4)
        self.assertEqual(config["model"]["out_channels"], 4)

    def test_rejects_channel_count_mismatch_against_hdf5_dataset_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            _write_config(
                config_path,
                {
                    "data": {
                        "dataset": {
                            "channels": 3,
                            "hdf5_dataset_keys": ["density", "pressure", "Vx", "Vy"],
                        }
                    },
                    "model": {},
                },
            )

            with self.assertRaisesRegex(ValueError, "does not match the number of hdf5_dataset_keys"):
                load_config(config_path, base_root=PROJECT_ROOT)

    def test_rejects_single_field_config_with_non_unit_channels(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            _write_config(
                config_path,
                {
                    "data": {
                        "dataset": {
                            "channels": 2,
                            "hdf5_dataset_key": "Vx",
                        }
                    },
                    "model": {},
                },
            )

            with self.assertRaisesRegex(ValueError, "must be 1 when using a single hdf5_dataset_key"):
                load_config(config_path, base_root=PROJECT_ROOT)


class TestChannelWiseNormalization(unittest.TestCase):
    def _dataset(self, normalization: dict) -> TensorFolder2DDataset:
        config = {
            "data": {
                "dataset_name": "tensor_folder_2d",
                "source_roots": {
                    "all_primary": str(PROJECT_ROOT / "data" / "raw" / "all"),
                    "all_extra": [],
                    "train_primary": "",
                    "train_extra": [],
                    "val_primary": "",
                    "val_extra": [],
                    "test_primary": "",
                    "test_extra": [],
                },
                "split": {
                    "mode": "auto",
                    "train_ratio": 1.0,
                    "val_ratio": 0.0,
                    "test_ratio": 0.0,
                    "shuffle": False,
                    "seed": 42,
                },
                "dataset": {
                    "recursive": True,
                    "allow_empty": True,
                    "extensions": [".npy"],
                    "npz_key": None,
                    "hdf5_dataset_key": None,
                    "hdf5_key_candidates": [],
                    "detect_hdf5_by_signature": True,
                    "hdf5_index_mode": "auto",
                    "hdf5_sample_axes": None,
                    "hdf5_sample_axis": 0,
                    "allow_images": False,
                    "channels": 2,
                    "input_size": [2, 2],
                    "strict_size": True,
                    "resize_mode": "bilinear",
                    "normalization": normalization,
                },
            }
        }
        return TensorFolder2DDataset(config=config, split="train")

    def test_channelwise_minmax_normalizes_each_channel_independently(self) -> None:
        dataset = self._dataset({"mode": "minmax", "scope": "channel", "clip_min": None, "clip_max": None})
        tensor = torch.tensor(
            [
                [[0.0, 2.0], [4.0, 6.0]],
                [[10.0, 14.0], [18.0, 22.0]],
            ]
        )

        normalized, _ = dataset.normalize_tensor(tensor)

        expected = torch.tensor(
            [
                [[0.0, 1.0 / 3.0], [2.0 / 3.0, 1.0]],
                [[0.0, 1.0 / 3.0], [2.0 / 3.0, 1.0]],
            ]
        )
        self.assertTrue(torch.allclose(normalized, expected, atol=1.0e-6))

    def test_channelwise_zscore_roundtrips_with_denormalize(self) -> None:
        dataset = self._dataset({"mode": "zscore", "scope": "channel", "clip_min": None, "clip_max": None})
        tensor = torch.tensor(
            [
                [[1.0, 3.0], [5.0, 7.0]],
                [[100.0, 120.0], [140.0, 160.0]],
            ]
        )

        normalized, state = dataset.normalize_tensor(tensor)
        restored = dataset.denormalize_tensor(normalized, state)

        self.assertTrue(torch.allclose(restored, tensor, atol=1.0e-5))


if __name__ == "__main__":
    unittest.main(verbosity=2)
