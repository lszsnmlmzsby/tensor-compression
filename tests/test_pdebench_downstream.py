from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import h5py
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tensor_compression.data.datasets.tensor_folder_2d import TensorFolder2DDataset
from tensor_compression.downstream.pdebench import (
    evaluate_records,
    inspect_pdebench_fields,
    iter_pdebench_records,
    pdebench_to_chw_frames,
    read_pdebench_sample,
)


def _write_synthetic_pdebench_file(path: Path) -> None:
    base = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("density", data=base)
        handle.create_dataset("pressure", data=base + 100.0)
        handle.create_dataset("Vx", data=base + 200.0)
        handle.create_dataset("Vy", data=base + 300.0)
        handle.create_dataset("x-coordinate", data=np.linspace(0.0, 1.0, 4, dtype=np.float32))
        handle.create_dataset("y-coordinate", data=np.linspace(0.0, 1.0, 5, dtype=np.float32))
        handle.create_dataset("t-coordinate", data=np.linspace(0.0, 1.0, 4, dtype=np.float32))


def _dataset_config(path: Path) -> dict:
    return {
        "data": {
            "dataset_name": "tensor_folder_2d",
            "source_roots": {
                "all_primary": str(path),
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
                "allow_empty": False,
                "extensions": [".hdf5"],
                "npz_key": None,
                "hdf5_dataset_key": None,
                "hdf5_dataset_keys": ["density", "pressure", "Vx", "Vy"],
                "hdf5_key_candidates": [],
                "detect_hdf5_by_signature": True,
                "hdf5_index_mode": "sample",
                "hdf5_sample_axes": [0, 1],
                "hdf5_sample_axis": 0,
                "allow_images": False,
                "channels": 4,
                "input_size": [4, 5],
                "strict_size": True,
                "resize_mode": "bilinear",
                "normalization": {"mode": "none", "clip_min": None, "clip_max": None},
            },
        }
    }


class TestPDEBenchDownstream(unittest.TestCase):
    def test_discovers_compressible_fields_and_skips_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "synthetic.hdf5"
            _write_synthetic_pdebench_file(file_path)

            fields = inspect_pdebench_fields(file_path)

        self.assertEqual([field.path for field in fields], ["density", "pressure", "Vx", "Vy"])
        self.assertEqual(fields[0].shape, (2, 3, 4, 5))

    def test_reads_pdebench_sample_as_spatial_time_channel(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "synthetic.hdf5"
            _write_synthetic_pdebench_file(file_path)

            data, grid, t_coordinates = read_pdebench_sample(
                file_path,
                field_keys=["density", "pressure", "Vx", "Vy"],
                sample_index=1,
            )

        self.assertEqual(tuple(data.shape), (4, 5, 3, 4))
        self.assertIsNotNone(grid)
        self.assertEqual(tuple(grid.shape), (4, 5, 2))
        self.assertIsNotNone(t_coordinates)
        self.assertEqual(tuple(t_coordinates.shape), (4,))
        self.assertEqual(float(data[0, 0, 0, 0]), 60.0)
        self.assertEqual(float(data[0, 0, 0, 3]), 360.0)

    def test_converts_pdebench_sample_to_chw_frames(self) -> None:
        data = torch.zeros(4, 5, 3, 4)

        frames = pdebench_to_chw_frames(data)

        self.assertEqual(tuple(frames.shape), (3, 4, 4, 5))

    def test_tensor_folder_2d_stacks_multiple_hdf5_fields_as_channels(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "synthetic.hdf5"
            _write_synthetic_pdebench_file(file_path)
            dataset = TensorFolder2DDataset(config=_dataset_config(file_path), split="train")

            sample = dataset[0]

        self.assertEqual(len(dataset), 6)
        self.assertEqual(tuple(sample["input"].shape), (4, 4, 5))
        self.assertEqual(sample["dataset_paths"], ["density", "pressure", "Vx", "Vy"])
        self.assertEqual(float(sample["input"][0, 0, 0]), 0.0)
        self.assertEqual(float(sample["input"][3, 0, 0]), 300.0)

    def test_evaluates_forward_and_inverse_callable_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "synthetic.hdf5"
            _write_synthetic_pdebench_file(file_path)
            records = iter_pdebench_records(
                hdf5_path=file_path,
                field_keys=["density", "pressure", "Vx", "Vy"],
                sample_indices=[0],
                reconstructor=None,
            )
            records[0].reconstructed = records[0].original + 1.0

            def forward_operator(payload: dict) -> torch.Tensor:
                return payload["data"].sum(dim=-1)

            def inverse_operator(payload: dict) -> torch.Tensor:
                return payload["data"][..., 0, :]

            results = evaluate_records(
                records,
                {"forward": forward_operator, "inverse": inverse_operator},
            )

        self.assertEqual(len(results["samples"]), 1)
        metrics = results["samples"][0]["metrics"]
        self.assertIn("reconstruction/mse", metrics)
        self.assertAlmostEqual(metrics["reconstruction/mse"], 1.0)
        self.assertIn("forward/mse", metrics)
        self.assertAlmostEqual(metrics["forward/mse"], 16.0)
        self.assertIn("inverse/mse", metrics)
        self.assertAlmostEqual(metrics["inverse/mse"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
