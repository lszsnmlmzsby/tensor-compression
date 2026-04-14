from __future__ import annotations

from itertools import product
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from tensor_compression.data.datasets.base import BaseTensorDataset
from tensor_compression.data.datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register("tensor_folder_2d")
class TensorFolder2DDataset(BaseTensorDataset):
    HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"
    HDF5_SUFFIX_HINTS = (".h5", ".hdf5", ".hdf", ".he5")

    def __init__(self, config: dict, split: str) -> None:
        super().__init__(config=config, split=split)
        self.extensions = tuple(ext.lower() for ext in self.dataset_cfg["extensions"])
        self.input_size = tuple(int(x) for x in self.dataset_cfg["input_size"])
        self.channels = int(self.dataset_cfg["channels"])
        self.strict_size = bool(self.dataset_cfg["strict_size"])
        self.resize_mode = str(self.dataset_cfg["resize_mode"])
        self.allow_images = bool(self.dataset_cfg["allow_images"])
        self.allow_empty = bool(self.dataset_cfg.get("allow_empty", False))
        self.npz_key = self.dataset_cfg.get("npz_key")
        self.hdf5_dataset_key = (
            self.dataset_cfg.get("hdf5_dataset_key")
            or self.dataset_cfg.get("field_key")
        )
        self.hdf5_key_candidates = list(self.dataset_cfg.get("hdf5_key_candidates", []))
        self.detect_hdf5_by_signature = bool(self.dataset_cfg.get("detect_hdf5_by_signature", True))
        self.hdf5_index_mode = str(self.dataset_cfg.get("hdf5_index_mode", "auto")).lower()
        raw_sample_axes = self.dataset_cfg.get("hdf5_sample_axes")
        if raw_sample_axes is None:
            self.hdf5_sample_axes = None
        elif isinstance(raw_sample_axes, (list, tuple)):
            self.hdf5_sample_axes = [int(axis) for axis in raw_sample_axes]
        else:
            self.hdf5_sample_axes = [int(raw_sample_axes)]
        self.hdf5_sample_axis = int(self.dataset_cfg.get("hdf5_sample_axis", 0))
        self.normalization_cfg = self.dataset_cfg.get("normalization", {})
        self.files = self._scan_files()
        self.samples = self._build_samples(self.files)
        if not self.samples and not self.allow_empty:
            joined_roots = ", ".join(str(p) for p in self._resolve_roots()) or "<none>"
            raise FileNotFoundError(
                f"No 2D tensor files found for split={split!r} under: {joined_roots}"
            )

    def _scan_files(self) -> list[Path]:
        files: list[Path] = []
        seen: set[Path] = set()
        recursive = bool(self.dataset_cfg["recursive"])
        for root in self._resolve_roots():
            if not root.exists():
                continue
            if root.is_file():
                candidate_paths = [root]
            else:
                iterator = root.rglob("*") if recursive else root.glob("*")
                candidate_paths = iterator
            for path in candidate_paths:
                if not path.is_file():
                    continue
                if path.suffix.lower() in self.extensions:
                    if path not in seen:
                        files.append(path)
                        seen.add(path)
                    continue
                if self.detect_hdf5_by_signature and self._looks_like_hdf5(path):
                    if path not in seen:
                        files.append(path)
                        seen.add(path)
                    continue
                if self._has_hdf5_like_suffix(path) and path not in seen:
                    files.append(path)
                    seen.add(path)
        return sorted(files)

    def _build_samples(self, files: list[Path]) -> list[dict]:
        samples: list[dict] = []
        for path in files:
            if self._is_hdf5_file(path):
                samples.extend(self._build_hdf5_samples(path))
                continue
            samples.append(
                {
                    "path": path,
                    "kind": "file",
                    "dataset_path": None,
                    "sample_indices": None,
                    "sample_axes": None,
                }
            )
        return self._apply_split(samples)

    def _build_hdf5_samples(self, path: Path) -> list[dict]:
        with h5py.File(path, "r") as handle:
            dataset_path = self._select_hdf5_dataset_path(handle)
            if dataset_path is None:
                available = self._list_hdf5_datasets(handle)
                raise ValueError(
                    "No suitable numeric 2D/3D/4D/5D dataset found in "
                    f"{path}. Available datasets: {available}"
                )
            dataset = handle[dataset_path]
            mode = self._resolve_hdf5_index_mode(dataset.shape)
            if mode == "file":
                return [
                    {
                        "path": path,
                        "kind": "hdf5_file",
                        "dataset_path": dataset_path,
                        "sample_indices": None,
                        "sample_axes": None,
                    }
                ]

            sample_axes = self._resolve_hdf5_sample_axes(dataset.shape)
            sample_ranges = [range(int(dataset.shape[axis])) for axis in sample_axes]
            samples: list[dict] = []
            for sample_indices in product(*sample_ranges):
                samples.append(
                    {
                        "path": path,
                        "kind": "hdf5_sample",
                        "dataset_path": dataset_path,
                        "sample_indices": tuple(int(index) for index in sample_indices),
                        "sample_axes": tuple(sample_axes),
                    }
                )
            return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        if not self.samples:
            raise RuntimeError(
                "Dataset is empty. Add training files or set `data.dataset.allow_empty: false` after data is ready."
            )
        sample = self.samples[index]
        path = sample["path"]
        tensor = self._load_tensor(sample)
        tensor = self._normalize(tensor)
        sample_id = path.stem
        if sample["sample_indices"] is not None:
            indices = "-".join(str(index) for index in sample["sample_indices"])
            sample_id = f"{sample_id}#{indices}"
        return {
            "input": tensor,
            "target": tensor.clone(),
            "sample_id": sample_id,
            "path": str(path),
            "dimensions": 2,
        }

    def _load_tensor(self, sample: dict) -> torch.Tensor:
        path = sample["path"]
        suffix = path.suffix.lower()
        if suffix == ".npy":
            array = np.load(path)
        elif suffix == ".npz":
            loaded = np.load(path)
            if self.npz_key:
                if self.npz_key not in loaded:
                    raise KeyError(f"{self.npz_key!r} not found in {path}.")
                array = loaded[self.npz_key]
            else:
                first_key = loaded.files[0]
                array = loaded[first_key]
        elif self._is_hdf5_file(path):
            array = self._load_hdf5_array(
                path=path,
                dataset_path=sample.get("dataset_path"),
                sample_indices=sample.get("sample_indices"),
                sample_axes=sample.get("sample_axes"),
            )
        else:
            if not self.allow_images:
                raise ValueError(f"Image loading is disabled, but got {path}.")
            image = Image.open(path)
            if self.channels == 1:
                image = image.convert("F")
                array = np.array(image, dtype=np.float32)
            else:
                image = image.convert("RGB")
                array = np.array(image, dtype=np.float32)
        tensor = self._to_tensor(array)
        tensor = self._ensure_chw(tensor)
        tensor = self._ensure_channel_count(tensor)
        tensor = self._resize_if_needed(tensor)
        return tensor

    def _load_hdf5_array(
        self,
        path: Path,
        dataset_path: str | None = None,
        sample_indices: tuple[int, ...] | None = None,
        sample_axes: tuple[int, ...] | None = None,
    ) -> np.ndarray:
        with h5py.File(path, "r") as handle:
            dataset_path = dataset_path or self._select_hdf5_dataset_path(handle)
            if dataset_path is None:
                available = self._list_hdf5_datasets(handle)
                raise ValueError(
                    "No suitable numeric 2D/3D/4D/5D dataset found in "
                    f"{path}. Available datasets: {available}"
                )
            dataset = handle[dataset_path]
            if sample_indices is None:
                array = dataset[()]
            else:
                resolved_axes = self._resolve_hdf5_sample_axes(dataset.shape, sample_axes)
                indexer = [slice(None)] * dataset.ndim
                for axis, index in zip(resolved_axes, sample_indices):
                    indexer[axis] = index
                array = dataset[tuple(indexer)]
        return np.asarray(array, dtype=np.float32)

    def _select_hdf5_dataset_path(self, handle: h5py.File) -> str | None:
        if self.hdf5_dataset_key:
            if self.hdf5_dataset_key not in handle:
                raise KeyError(
                    f"HDF5 dataset key {self.hdf5_dataset_key!r} not found. "
                    f"Available datasets: {self._list_hdf5_datasets(handle)}"
                )
            dataset = handle[self.hdf5_dataset_key]
            if not isinstance(dataset, h5py.Dataset):
                raise ValueError(
                    f"HDF5 key {self.hdf5_dataset_key!r} exists but is not a dataset."
                )
            return self.hdf5_dataset_key

        for key in self.hdf5_key_candidates:
            if key in handle and isinstance(handle[key], h5py.Dataset):
                if self._is_supported_hdf5_dataset(handle[key]):
                    return key

        candidates: list[str] = []

        def visitor(name: str, obj) -> None:
            if isinstance(obj, h5py.Dataset) and self._is_supported_hdf5_dataset(obj):
                candidates.append(name)

        handle.visititems(visitor)
        return candidates[0] if candidates else None

    def _is_supported_hdf5_dataset(self, dataset: h5py.Dataset) -> bool:
        if not np.issubdtype(dataset.dtype, np.number):
            return False
        return dataset.ndim in (2, 3, 4, 5)

    def _list_hdf5_datasets(self, handle: h5py.File) -> list[str]:
        datasets: list[str] = []

        def visitor(name: str, obj) -> None:
            if isinstance(obj, h5py.Dataset):
                datasets.append(name)

        handle.visititems(visitor)
        return datasets

    def _has_hdf5_like_suffix(self, path: Path) -> bool:
        suffix = path.suffix.lower()
        return any(hint in suffix for hint in self.HDF5_SUFFIX_HINTS) or ("h5" in suffix) or ("hdf" in suffix)

    def _looks_like_hdf5(self, path: Path) -> bool:
        try:
            return bool(h5py.is_hdf5(str(path)))
        except (OSError, ValueError):
            try:
                with path.open("rb") as handle:
                    signature = handle.read(len(self.HDF5_SIGNATURE))
                return signature == self.HDF5_SIGNATURE
            except OSError:
                return False

    def _is_hdf5_file(self, path: Path) -> bool:
        return self._has_hdf5_like_suffix(path) or self._looks_like_hdf5(path)

    def _resolve_hdf5_index_mode(self, shape: tuple[int, ...]) -> str:
        mode = self.hdf5_index_mode
        if mode not in {"auto", "file", "sample"}:
            raise ValueError(f"Unsupported hdf5_index_mode: {mode}")
        if mode in {"file", "sample"}:
            return mode

        ndim = len(shape)
        if ndim <= 2:
            return "file"
        if ndim == 3:
            if shape[0] == self.channels or shape[-1] == self.channels:
                return "file"
            return "sample"
        if ndim >= 4:
            return "sample"
        return "file"

    def _resolve_hdf5_sample_axes(
        self,
        shape: tuple[int, ...],
        configured_axes: tuple[int, ...] | list[int] | None = None,
    ) -> list[int]:
        raw_axes = configured_axes
        if raw_axes is None:
            raw_axes = self.hdf5_sample_axes
        if raw_axes is None:
            axes = [self._normalize_sample_axis(self.hdf5_sample_axis, len(shape))]
            axes = self._augment_sample_axes(shape, axes)
        else:
            axes = [self._normalize_sample_axis(int(axis), len(shape)) for axis in raw_axes]
        if len(set(axes)) != len(axes):
            raise ValueError(f"Duplicated sample axes are not allowed: {axes}")
        if not self._is_valid_2d_sample_shape(shape, axes):
            raise ValueError(
                "Configured HDF5 sample axes do not leave a valid 2D sample shape. "
                f"Got dataset shape={shape}, sample_axes={axes}. "
                "Adjust `hdf5_sample_axes` (for example [0, 1] for [N, T, H, W]) "
                "or switch to `hdf5_index_mode: file`."
            )
        return list(axes)

    def _augment_sample_axes(self, shape: tuple[int, ...], axes: list[int]) -> list[int]:
        resolved_axes = list(axes)
        while not self._is_valid_2d_sample_shape(shape, resolved_axes):
            candidates = [axis for axis in range(len(shape)) if axis not in resolved_axes]
            if not candidates:
                break
            next_axis = candidates[0]
            resolved_axes.append(next_axis)
        return resolved_axes

    def _is_valid_2d_sample_shape(self, shape: tuple[int, ...], sample_axes: list[int]) -> bool:
        remaining = [size for axis, size in enumerate(shape) if axis not in sample_axes]
        if len(remaining) == 2:
            return True
        if len(remaining) != 3:
            return False
        return (
            remaining[0] in (self.channels, 1, 3, 4)
            or remaining[-1] in (self.channels, 1, 3, 4)
        )

    def _normalize_sample_axis(self, axis: int, ndim: int) -> int:
        if axis < 0:
            axis = ndim + axis
        if axis < 0 or axis >= ndim:
            raise ValueError(f"Sample axis {axis} is out of range for ndim={ndim}.")
        return axis

    def _ensure_chw(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 3:
            if tensor.shape[0] in (1, 3, 4):
                pass
            elif tensor.shape[-1] in (1, 3, 4):
                tensor = tensor.permute(2, 0, 1)
            else:
                raise ValueError(
                    "Cannot infer 2D tensor channel order from shape "
                    f"{tuple(tensor.shape)}. If this came from an HDF5 sequence like "
                    "[N, T, H, W], set `hdf5_sample_axes: [0, 1]` so time is expanded "
                    "into samples instead of being treated as channels."
                )
        else:
            raise ValueError(
                f"Expected 2D tensor with 2 or 3 dims, got shape {tuple(tensor.shape)}."
            )
        return tensor.contiguous()

    def _ensure_channel_count(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape[0] == self.channels:
            return tensor
        if tensor.shape[0] == 1 and self.channels > 1:
            return tensor.repeat(self.channels, 1, 1)
        if tensor.shape[0] > self.channels:
            return tensor[: self.channels]
        raise ValueError(
            f"Configured channels={self.channels}, but sample has {tensor.shape[0]} channels."
        )

    def _resize_if_needed(self, tensor: torch.Tensor) -> torch.Tensor:
        height, width = tensor.shape[-2:]
        target_h, target_w = self.input_size
        if (height, width) == (target_h, target_w):
            return tensor
        if self.strict_size:
            raise ValueError(
                f"Expected input size {(target_h, target_w)}, got {(height, width)}."
            )
        tensor = tensor.unsqueeze(0)
        kwargs = {
            "input": tensor,
            "size": self.input_size,
            "mode": self.resize_mode,
        }
        if self.resize_mode in {"bilinear", "bicubic"}:
            kwargs["align_corners"] = False
        tensor = F.interpolate(**kwargs)
        return tensor.squeeze(0)

    def _normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        mode = str(self.normalization_cfg.get("mode", "none")).lower()
        clip_min = self.normalization_cfg.get("clip_min")
        clip_max = self.normalization_cfg.get("clip_max")
        if clip_min is not None or clip_max is not None:
            tensor = torch.clamp(tensor, min=clip_min, max=clip_max)

        if mode == "none":
            return tensor
        if mode == "minmax":
            min_value = tensor.amin()
            max_value = tensor.amax()
            return (tensor - min_value) / (max_value - min_value + 1.0e-6)
        if mode == "zscore":
            mean = tensor.mean()
            std = tensor.std(unbiased=False)
            return (tensor - mean) / (std + 1.0e-6)
        raise ValueError(f"Unsupported normalization mode: {mode}")
