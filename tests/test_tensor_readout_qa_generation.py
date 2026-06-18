from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_tensor_readout_qa import (  # noqa: E402
    StateRef,
    compute_quantile_bins,
    generate_split_records,
    split_sample_indices,
)


def _write_synthetic_pdebench_file(path: Path) -> None:
    base = np.arange(4 * 3 * 6 * 7, dtype=np.float32).reshape(4, 3, 6, 7)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("density", data=base)
        handle.create_dataset("pressure", data=base + 100.0)
        handle.create_dataset("Vx", data=base + 200.0)
        handle.create_dataset("Vy", data=base + 300.0)
        handle.create_dataset("x-coordinate", data=np.linspace(0.0, 1.0, 6, dtype=np.float32))
        handle.create_dataset("y-coordinate", data=np.linspace(0.0, 1.0, 7, dtype=np.float32))
        handle.create_dataset("t-coordinate", data=np.linspace(0.0, 1.0, 3, dtype=np.float32))


class TestTensorReadoutQAGeneration(unittest.TestCase):
    def test_splits_samples_without_overlap(self) -> None:
        splits = split_sample_indices(
            sample_indices=list(range(10)),
            train_ratio=0.7,
            val_ratio=0.1,
            test_ratio=0.2,
            seed=123,
        )

        train = set(splits.train)
        val = set(splits.val)
        test = set(splits.test)

        self.assertTrue(train)
        self.assertTrue(val)
        self.assertTrue(test)
        self.assertFalse(train & val)
        self.assertFalse(train & test)
        self.assertFalse(val & test)
        self.assertEqual(train | val | test, set(range(10)))

    def test_generates_self_supervised_records_with_oracles(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            hdf5_path = Path(tmpdir) / "synthetic.hdf5"
            _write_synthetic_pdebench_file(hdf5_path)
            field_keys = ["density", "pressure", "Vx", "Vy"]
            train_states = [StateRef(sample_index=0, time_index=0), StateRef(sample_index=1, time_index=1)]
            bin_edges = compute_quantile_bins(
                hdf5_path=hdf5_path,
                field_keys=field_keys,
                train_states=train_states,
                spatial_stride=1,
                num_bins=4,
                quantile_samples_per_state=12,
                seed=42,
            )

            records = generate_split_records(
                hdf5_path=hdf5_path,
                field_keys=field_keys,
                states=[StateRef(sample_index=2, time_index=1)],
                bin_edges=bin_edges,
                num_bins=4,
                patch_size=3,
                counts={
                    "point_bin": 2,
                    "point_compare": 2,
                    "patch_compare": 2,
                    "max_speed_quadrant": 1,
                    "global_stat_bin": 1,
                },
                spatial_stride=1,
                seed=7,
                latent_root="latents",
                include_oracle=True,
                compare_min_bin_distance=1,
                compare_max_attempts=16,
            )

        self.assertEqual(len(records), 8)
        task_types = {record["task_type"] for record in records}
        self.assertEqual(
            task_types,
            {
                "point_bin",
                "point_compare",
                "patch_compare",
                "max_speed_quadrant",
                "global_stat_bin",
            },
        )
        first = records[0]
        self.assertIn("qa_id", first)
        self.assertEqual(first["state_ref"], "sample000002_t0001")
        self.assertEqual(first["latent_ref"], "latents/sample000002_t0001.pt")
        self.assertIn(first["answer"], first["choices"])
        self.assertIn("oracle", first)
        self.assertEqual(first["metadata"]["grid_shape"], [6, 7])

    def test_records_are_json_serializable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            hdf5_path = Path(tmpdir) / "synthetic.hdf5"
            _write_synthetic_pdebench_file(hdf5_path)
            field_keys = ["density", "pressure", "Vx", "Vy"]
            states = [StateRef(sample_index=0, time_index=0)]
            bin_edges = compute_quantile_bins(
                hdf5_path=hdf5_path,
                field_keys=field_keys,
                train_states=states,
                spatial_stride=1,
                num_bins=3,
                quantile_samples_per_state=8,
                seed=42,
            )
            records = generate_split_records(
                hdf5_path=hdf5_path,
                field_keys=field_keys,
                states=states,
                bin_edges=bin_edges,
                num_bins=3,
                patch_size=2,
                counts={
                    "point_bin": 1,
                    "point_compare": 1,
                    "patch_compare": 1,
                    "max_speed_quadrant": 1,
                    "global_stat_bin": 1,
                },
                spatial_stride=1,
                seed=7,
                latent_root=None,
                include_oracle=True,
                compare_min_bin_distance=1,
                compare_max_attempts=16,
            )

        payload = "\n".join(json.dumps(record, ensure_ascii=False) for record in records)
        self.assertIn("VALUE_BIN", payload)
        self.assertIn("COMPARE_POINT", payload)
        self.assertIn("MAX_SPEED_QUADRANT", payload)


if __name__ == "__main__":
    unittest.main(verbosity=2)
