from __future__ import annotations

from abc import ABC, abstractmethod
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset


class BaseTensorDataset(Dataset, ABC):
    def __init__(self, config: dict, split: str) -> None:
        self.config = config
        self.split = split
        self.data_cfg = config["data"]
        self.dataset_cfg = config["data"]["dataset"]
        self.sources_cfg = config["data"]["source_roots"]
        self.split_cfg = config["data"].get("split", {})

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: int) -> dict:
        raise NotImplementedError

    def _resolve_roots(self) -> list[Path]:
        split_mode = str(self.split_cfg.get("mode", "predefined")).lower()
        if split_mode == "auto":
            primary = self.sources_cfg.get("all_primary")
            extra_entries = self.sources_cfg.get("all_extra", [])
        else:
            primary_key = f"{self.split}_primary"
            extra_key = f"{self.split}_extra"
            primary = self.sources_cfg.get(primary_key)
            extra_entries = self.sources_cfg.get(extra_key, [])

        roots: list[Path] = []
        if primary:
            roots.append(Path(primary))
        for entry in extra_entries:
            roots.append(Path(entry))
        return roots

    def _apply_split(self, items: list):
        split_mode = str(self.split_cfg.get("mode", "predefined")).lower()
        if split_mode == "predefined":
            return items
        if split_mode != "auto":
            raise ValueError(f"Unsupported data.split.mode: {split_mode}")

        if not items:
            return items

        train_ratio = float(self.split_cfg.get("train_ratio", 0.8))
        val_ratio = float(self.split_cfg.get("val_ratio", 0.1))
        test_ratio = float(self.split_cfg.get("test_ratio", 0.1))
        ratio_sum = train_ratio + val_ratio + test_ratio
        if abs(ratio_sum - 1.0) > 1.0e-6:
            raise ValueError(
                f"data.split ratios must sum to 1.0, got {ratio_sum:.6f} "
                f"(train={train_ratio}, val={val_ratio}, test={test_ratio})."
            )

        ordered = list(items)
        if bool(self.split_cfg.get("shuffle", True)):
            rng = random.Random(int(self.split_cfg.get("seed", 42)))
            rng.shuffle(ordered)

        total = len(ordered)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        if self.split == "train":
            subset = ordered[:train_end]
        elif self.split == "val":
            subset = ordered[train_end:val_end]
        elif self.split == "test":
            subset = ordered[val_end:]
        else:
            raise ValueError(f"Unsupported split: {self.split}")
        return subset

    def _to_tensor(self, array) -> torch.Tensor:
        tensor = torch.as_tensor(array, dtype=torch.float32)
        return tensor
