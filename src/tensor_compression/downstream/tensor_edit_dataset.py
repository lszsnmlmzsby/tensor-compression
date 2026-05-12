from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class TensorEditJsonlDataset(Dataset):
    """Lazily reads tensor-edit samples stored as JSONL records."""

    def __init__(
        self,
        jsonl_path: str | Path,
        input_size: tuple[int, int] = (512, 512),
        channels: int = 1,
        fix_prompt_mojibake: bool = False,
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.input_size = tuple(int(x) for x in input_size)
        self.channels = int(channels)
        self.fix_prompt_mojibake = bool(fix_prompt_mojibake)
        if self.channels <= 0:
            raise ValueError("channels must be positive.")
        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"Tensor-edit JSONL file not found: {self.jsonl_path}")
        self.offsets = self._build_offsets()
        if not self.offsets:
            raise ValueError(f"Tensor-edit JSONL file is empty: {self.jsonl_path}")

    def _build_offsets(self) -> list[int]:
        offsets: list[int] = []
        with self.jsonl_path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    offsets.append(offset)
        return offsets

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self._read_record(index)
        raw_prompt = str(record.get("prompt", ""))
        prompt = self._fix_mojibake(raw_prompt) if self.fix_prompt_mojibake else raw_prompt
        if not prompt:
            raise ValueError(f"Record {index} in {self.jsonl_path} has an empty prompt.")

        input_tensor = self._load_tensor(record, key="tensor", path_key="tensor_path")
        target_tensor = self._load_tensor(record, key="label", path_key="label_path")
        sample_id = str(
            record.get("sample_id")
            or record.get("id")
            or f"{self.jsonl_path.stem}:{index:06d}"
        )
        meta = record.get("meta", {})
        if meta is None:
            meta = {}
        if not isinstance(meta, dict):
            raise ValueError(f"Record {index} meta must be a JSON object, got {type(meta).__name__}.")

        return {
            "input": input_tensor,
            "target": target_tensor,
            "prompt": prompt,
            "raw_prompt": raw_prompt,
            "meta": meta,
            "sample_id": sample_id,
        }

    def _fix_mojibake(self, text: str) -> str:
        for encoding in ("gbk", "cp936", "latin1"):
            try:
                repaired = text.encode(encoding).decode("utf-8")
            except UnicodeError:
                continue
            if self._count_cjk(repaired) >= self._count_cjk(text) and repaired != text:
                return repaired
        return text

    def _count_cjk(self, text: str) -> int:
        return sum(1 for char in text if "\u4e00" <= char <= "\u9fff")

    def _read_record(self, index: int) -> dict[str, Any]:
        with self.jsonl_path.open("rb") as handle:
            handle.seek(self.offsets[index])
            line = handle.readline()
        try:
            record = json.loads(line.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON at line {index + 1} of {self.jsonl_path}: {exc}"
            ) from exc
        if not isinstance(record, dict):
            raise ValueError(f"Record {index} in {self.jsonl_path} must be a JSON object.")
        return record

    def _load_tensor(self, record: dict[str, Any], key: str, path_key: str) -> torch.Tensor:
        if key in record:
            array = np.asarray(record[key], dtype=np.float32)
        elif path_key in record:
            path = Path(str(record[path_key]))
            if not path.is_absolute():
                path = (self.jsonl_path.parent / path).resolve()
            array = self._load_array_file(path)
        else:
            raise KeyError(
                f"Each record must contain either {key!r} inline data or {path_key!r}."
            )
        return self._ensure_chw(torch.as_tensor(array, dtype=torch.float32))

    def _load_array_file(self, path: Path) -> np.ndarray:
        suffix = path.suffix.lower()
        if suffix == ".npy":
            return np.load(path).astype(np.float32)
        if suffix == ".npz":
            loaded = np.load(path)
            first_key = loaded.files[0]
            return loaded[first_key].astype(np.float32)
        if suffix in {".pt", ".pth"}:
            loaded_tensor = torch.load(path, map_location="cpu")
            if isinstance(loaded_tensor, dict):
                for value in loaded_tensor.values():
                    if torch.is_tensor(value):
                        loaded_tensor = value
                        break
            if not torch.is_tensor(loaded_tensor):
                raise ValueError(f"No tensor payload found in {path}.")
            return loaded_tensor.detach().cpu().numpy().astype(np.float32)
        raise ValueError(
            f"Unsupported tensor file suffix {suffix!r} for {path}. "
            "Use inline JSON arrays, .npy, .npz, .pt, or .pth."
        )

    def _ensure_chw(self, tensor: torch.Tensor) -> torch.Tensor:
        height, width = self.input_size
        if tensor.ndim == 1:
            expected = self.channels * height * width
            if tensor.numel() != expected:
                raise ValueError(
                    f"Flat tensor has {tensor.numel()} values, expected {expected} "
                    f"for shape [{self.channels}, {height}, {width}]."
                )
            tensor = tensor.reshape(self.channels, height, width)
        elif tensor.ndim == 2:
            if self.channels != 1:
                raise ValueError(
                    f"2D inline tensors are only valid for channels=1, got channels={self.channels}."
                )
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 3:
            if tensor.shape[0] == self.channels:
                pass
            elif tensor.shape[-1] == self.channels:
                tensor = tensor.permute(2, 0, 1)
            else:
                raise ValueError(
                    f"Cannot infer channel order from tensor shape {tuple(tensor.shape)} "
                    f"with channels={self.channels}."
                )
        else:
            raise ValueError(f"Expected tensor with 1, 2, or 3 dims, got {tensor.ndim}.")

        if tuple(tensor.shape) != (self.channels, height, width):
            raise ValueError(
                f"Expected tensor shape [{self.channels}, {height}, {width}], "
                f"got {tuple(tensor.shape)}."
            )
        return tensor.contiguous()


def tensor_edit_collate_fn(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "input": torch.stack([sample["input"] for sample in samples], dim=0),
        "target": torch.stack([sample["target"] for sample in samples], dim=0),
        "prompt": [sample["prompt"] for sample in samples],
        "raw_prompt": [sample["raw_prompt"] for sample in samples],
        "meta": [sample["meta"] for sample in samples],
        "sample_id": [sample["sample_id"] for sample in samples],
    }
