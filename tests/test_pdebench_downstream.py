from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tensor_compression.data.datasets.tensor_folder_2d import TensorFolder2DDataset
from tensor_compression.downstream.pdebench import (
    ProgressEvent,
    evaluate_records,
    export_reconstructed_hdf5,
    inspect_pdebench_fields,
    iter_pdebench_records,
    pdebench_to_chw_frames,
    read_pdebench_sample,
    resolve_checkpoint_field_keys,
    resolve_field_keys_for_evaluation,
    validate_checkpoint_field_keys_against_model,
)
from scripts.evaluate_pdebench_downstream import parse_sample_indices


def _write_synthetic_pdebench_file(path: Path) -> None:
    base = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
    with h5py.File(path, "w") as handle:
        handle.attrs["case"] = "synthetic"
        handle.create_dataset("density", data=base)
        handle.create_dataset("pressure", data=base + 100.0)
        handle.create_dataset("Vx", data=base + 200.0)
        handle.create_dataset("Vy", data=base + 300.0)
        handle.create_dataset("metadata", data=np.array([123.0], dtype=np.float32))
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

    def test_parse_sample_indices_supports_all_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "synthetic.hdf5"
            _write_synthetic_pdebench_file(file_path)
            fields = inspect_pdebench_fields(file_path)

            indices = parse_sample_indices("all", fields)

        self.assertEqual(indices, [0, 1])

    def test_resolves_field_order_from_checkpoint_config(self) -> None:
        config = {
            "data": {
                "dataset": {
                    "hdf5_dataset_keys": ["Vx", "Vy", "density", "pressure"],
                }
            }
        }

        field_keys = resolve_checkpoint_field_keys(config)

        self.assertEqual(field_keys, ["Vx", "Vy", "density", "pressure"])

    def test_prefers_checkpoint_field_order_when_cli_fields_omitted(self) -> None:
        resolved = resolve_field_keys_for_evaluation(
            cli_field_keys=None,
            checkpoint_field_keys=["Vx", "Vy", "density", "pressure"],
            discovered_field_keys=["density", "pressure", "Vx", "Vy"],
        )

        self.assertEqual(resolved, ["Vx", "Vy", "density", "pressure"])

    def test_rejects_mismatched_cli_field_order_when_checkpoint_is_present(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not match the compressor checkpoint field order"):
            resolve_field_keys_for_evaluation(
                cli_field_keys=["density", "pressure", "Vx", "Vy"],
                checkpoint_field_keys=["Vx", "Vy", "density", "pressure"],
                discovered_field_keys=["density", "pressure", "Vx", "Vy"],
            )

    def test_rejects_checkpoint_field_count_inconsistent_with_model_channels(self) -> None:
        with self.assertRaisesRegex(ValueError, "inconsistent with model.in_channels"):
            validate_checkpoint_field_keys_against_model(
                config={"model": {"in_channels": 4}},
                field_keys=["density", "pressure", "Vx"],
            )

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
        self.assertEqual(sample["dataset_paths"], "density,pressure,Vx,Vy")
        self.assertEqual(float(sample["input"][0, 0, 0]), 0.0)
        self.assertEqual(float(sample["input"][3, 0, 0]), 300.0)

    def test_tensor_folder_2d_samples_are_default_collate_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "synthetic.hdf5"
            _write_synthetic_pdebench_file(file_path)
            dataset = TensorFolder2DDataset(config=_dataset_config(file_path), split="train")

            batch = next(iter(DataLoader(dataset, batch_size=2)))

        self.assertEqual(tuple(batch["input"].shape), (2, 4, 4, 5))
        self.assertEqual(batch["dataset_path"], ["", ""])
        self.assertEqual(batch["dataset_paths"], ["density,pressure,Vx,Vy", "density,pressure,Vx,Vy"])

    def test_exports_reconstructed_hdf5_and_preserves_unselected_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "synthetic.hdf5"
            output_path = Path(tmpdir) / "synthetic_reconstructed.hdf5"
            _write_synthetic_pdebench_file(file_path)
            records = iter_pdebench_records(
                hdf5_path=file_path,
                field_keys=["density", "pressure", "Vx", "Vy"],
                sample_indices=[1],
                reconstructor=None,
            )
            records[0].reconstructed = records[0].original + 10.0

            exported_path = export_reconstructed_hdf5(
                hdf5_path=file_path,
                output_path=output_path,
                records=records,
                field_keys=["density", "pressure", "Vx", "Vy"],
            )

            with h5py.File(file_path, "r") as original, h5py.File(exported_path, "r") as exported:
                self.assertEqual(exported.attrs["case"], "synthetic")
                np.testing.assert_allclose(exported["density"][0], original["density"][0])
                np.testing.assert_allclose(exported["density"][1], original["density"][1] + 10.0)
                np.testing.assert_allclose(exported["pressure"][1], original["pressure"][1] + 10.0)
                np.testing.assert_allclose(exported["Vx"][1], original["Vx"][1] + 10.0)
                np.testing.assert_allclose(exported["Vy"][1], original["Vy"][1] + 10.0)
                np.testing.assert_allclose(exported["metadata"][()], original["metadata"][()])
                np.testing.assert_allclose(exported["x-coordinate"][()], original["x-coordinate"][()])
                np.testing.assert_allclose(original["density"][1], np.arange(60, 120, dtype=np.float32).reshape(3, 4, 5))

    def test_evaluates_forward_and_inverse_callable_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "synthetic.hdf5"
            _write_synthetic_pdebench_file(file_path)
            progress_events: list[ProgressEvent] = []
            records = iter_pdebench_records(
                hdf5_path=file_path,
                field_keys=["density", "pressure", "Vx", "Vy"],
                sample_indices=[0],
                reconstructor=None,
                progress_callback=progress_events.append,
            )
            records[0].reconstructed = records[0].original + 1.0

            def forward_operator(payload: dict) -> torch.Tensor:
                return payload["data"].sum(dim=-1)

            def inverse_operator(payload: dict) -> torch.Tensor:
                return payload["data"][..., 0, :]

            results = evaluate_records(
                records,
                {"forward": forward_operator, "inverse": inverse_operator},
                progress_callback=progress_events.append,
            )

        self.assertEqual(len(results["samples"]), 1)
        metrics = results["samples"][0]["metrics"]
        self.assertIn("reconstruction/mse", metrics)
        self.assertAlmostEqual(metrics["reconstruction/mse"], 1.0)
        self.assertIn("forward/mse", metrics)
        self.assertAlmostEqual(metrics["forward/mse"], 16.0)
        self.assertIn("inverse/mse", metrics)
        self.assertAlmostEqual(metrics["inverse/mse"], 1.0)
        self.assertGreaterEqual(len(progress_events), 6)
        self.assertEqual(progress_events[0].phase, "sample_started")
        self.assertEqual(progress_events[1].phase, "sample_loaded")
        operator_start_pairs = [
            (event.operator_name, event.variant)
            for event in progress_events
            if event.phase == "operator_started"
        ]
        self.assertEqual(
            operator_start_pairs,
            [
                ("forward", "original"),
                ("forward", "reconstructed"),
                ("inverse", "original"),
                ("inverse", "reconstructed"),
            ],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
