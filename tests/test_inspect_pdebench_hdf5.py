from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "compressor_2d.yaml"
PDEBENCH_HDF5_ENV = "PDEBENCH_HDF5_PATH"


def _resolve_project_path(raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    if raw_path.startswith("./") or raw_path.startswith(".\\"):
        return (PROJECT_ROOT / raw_path[2:]).resolve()
    if raw_path.startswith("../") or raw_path.startswith("..\\"):
        return (PROJECT_ROOT / raw_path).resolve()
    path = Path(raw_path).expanduser()
    if not path.is_absolute() and not raw_path.startswith(("/", "\\")):
        return (PROJECT_ROOT / path).resolve()
    return path


def _load_default_config() -> dict:
    if not DEFAULT_CONFIG_PATH.exists():
        return {}
    with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    return loaded if isinstance(loaded, dict) else {}


def _get_configured_pdebench_path() -> Path | None:
    config = _load_default_config()
    source_roots = config.get("data", {}).get("source_roots", {})
    raw_path = source_roots.get("all_primary")
    return _resolve_project_path(raw_path)


def _get_configured_hdf5_dataset_key() -> str | None:
    config = _load_default_config()
    dataset_cfg = config.get("data", {}).get("dataset", {})
    return dataset_cfg.get("hdf5_dataset_key") or dataset_cfg.get("field_key")


def _get_runtime_dataset_path() -> tuple[Path | None, str]:
    env_path = os.environ.get(PDEBENCH_HDF5_ENV)
    if env_path:
        return _resolve_project_path(env_path), f"environment variable {PDEBENCH_HDF5_ENV}"

    config_path = _get_configured_pdebench_path()
    if config_path is not None:
        return config_path, f"{DEFAULT_CONFIG_PATH}: data.source_roots.all_primary"

    return None, "not configured"


def _suggest_2d_hdf5_loading(shape: tuple[int, ...]) -> str:
    if len(shape) == 2:
        return "suggested_2d_config: hdf5_index_mode=file"
    if len(shape) == 3:
        return "suggested_2d_config: hdf5_index_mode=sample, hdf5_sample_axes=[0]"
    if len(shape) == 4:
        return "suggested_2d_config: hdf5_index_mode=sample, hdf5_sample_axes=[0, 1]"
    return "suggested_2d_config: inspect manually"


def inspect_hdf5_file(path: Path) -> dict:
    report = {
        "path": str(path),
        "top_level_keys": [],
        "datasets": [],
    }
    with h5py.File(path, "r") as handle:
        report["top_level_keys"] = list(handle.keys())

        def visitor(name: str, obj) -> None:
            if not isinstance(obj, h5py.Dataset):
                return
            shape = tuple(int(dim) for dim in obj.shape)
            report["datasets"].append(
                {
                    "path": name,
                    "shape": shape,
                    "dtype": str(obj.dtype),
                    "ndim": int(obj.ndim),
                    "suggested_2d_loading": _suggest_2d_hdf5_loading(shape),
                }
            )

        handle.visititems(visitor)
    return report


def format_report(report: dict) -> str:
    lines = [
        f"HDF5 file: {report['path']}",
        f"Top-level keys: {report['top_level_keys']}",
        "Datasets:",
    ]
    for dataset in report["datasets"]:
        lines.append(
            "  - "
            f"{dataset['path']}: "
            f"shape={dataset['shape']}, "
            f"dtype={dataset['dtype']}, "
            f"{dataset['suggested_2d_loading']}"
        )
    return "\n".join(lines)


class TestHDF5InspectionHelpers(unittest.TestCase):
    def test_collects_keys_shapes_and_dtypes_from_hdf5(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "synthetic_pdebench_like.hdf5"
            with h5py.File(file_path, "w") as handle:
                handle.create_dataset("Vx", data=np.zeros((2, 3, 8, 8), dtype=np.float32))
                fields = handle.create_group("fields")
                fields.create_dataset("pressure", data=np.ones((8, 8), dtype=np.float64))

            report = inspect_hdf5_file(file_path)

        self.assertEqual(set(report["top_level_keys"]), {"Vx", "fields"})
        dataset_map = {item["path"]: item for item in report["datasets"]}
        self.assertIn("Vx", dataset_map)
        self.assertIn("fields/pressure", dataset_map)
        self.assertEqual(dataset_map["Vx"]["shape"], (2, 3, 8, 8))
        self.assertEqual(dataset_map["Vx"]["dtype"], "float32")
        self.assertEqual(dataset_map["fields/pressure"]["shape"], (8, 8))
        self.assertEqual(dataset_map["fields/pressure"]["dtype"], "float64")


class TestPDEBenchHDF5Inspection(unittest.TestCase):
    def test_runtime_pdebench_file_can_be_inspected(self) -> None:
        dataset_path, path_source = _get_runtime_dataset_path()
        if dataset_path is None or not dataset_path.exists():
            self.skipTest(
                "No PDEBench HDF5 file is available for inspection. "
                f"Set {PDEBENCH_HDF5_ENV} or update {DEFAULT_CONFIG_PATH}."
            )

        report = inspect_hdf5_file(dataset_path)
        self.assertGreaterEqual(len(report["datasets"]), 1)
        print()
        print(f"Resolved PDEBench file from: {path_source}")
        print(format_report(report))

    def test_configured_hdf5_dataset_key_exists_when_file_is_available(self) -> None:
        dataset_path = _get_configured_pdebench_path()
        configured_key = _get_configured_hdf5_dataset_key()
        if dataset_path is None or not dataset_path.exists() or not configured_key:
            self.skipTest(
                "Configured PDEBench HDF5 file or hdf5_dataset_key is unavailable."
            )

        report = inspect_hdf5_file(dataset_path)
        dataset_paths = {item["path"] for item in report["datasets"]}
        self.assertIn(
            configured_key,
            dataset_paths,
            msg=(
                f"Configured hdf5_dataset_key={configured_key!r} was not found in "
                f"{dataset_path}. Available dataset paths: {sorted(dataset_paths)}"
            ),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
