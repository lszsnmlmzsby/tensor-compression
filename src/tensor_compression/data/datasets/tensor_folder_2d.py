from __future__ import annotations

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
        self.hdf5_dataset_key = self.dataset_cfg.get("hdf5_dataset_key")
        self.hdf5_key_candidates = list(self.dataset_cfg.get("hdf5_key_candidates", []))
        self.detect_hdf5_by_signature = bool(self.dataset_cfg.get("detect_hdf5_by_signature", True))
        self.normalization_cfg = self.dataset_cfg.get("normalization", {})
        self.files = self._scan_files()
        if not self.files and not self.allow_empty:
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
            iterator = root.rglob("*") if recursive else root.glob("*")
            for path in iterator:
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
        return self._apply_split(sorted(files))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict:
        if not self.files:
            raise RuntimeError(
                "Dataset is empty. Add training files or set `data.dataset.allow_empty: false` after data is ready."
            )
        path = self.files[index]
        tensor = self._load_tensor(path)
        tensor = self._normalize(tensor)
        return {
            "input": tensor,
            "target": tensor.clone(),
            "sample_id": path.stem,
            "path": str(path),
            "dimensions": 2,
        }

    def _load_tensor(self, path: Path) -> torch.Tensor:
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
            array = self._load_hdf5_array(path)
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

    def _load_hdf5_array(self, path: Path) -> np.ndarray:
        with h5py.File(path, "r") as handle:
            dataset_path = self._select_hdf5_dataset_path(handle)
            if dataset_path is None:
                available = self._list_hdf5_datasets(handle)
                raise ValueError(
                    "No suitable numeric 2D/3D dataset found in "
                    f"{path}. Available datasets: {available}"
                )
            array = handle[dataset_path][()]
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
        return dataset.ndim in (2, 3)

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
                    f"Cannot infer 2D tensor channel order from shape {tuple(tensor.shape)}."
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
