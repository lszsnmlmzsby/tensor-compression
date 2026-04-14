from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader

from tensor_compression.data.datasets import DATASET_REGISTRY


def build_dataset(config: dict, split: str):
    dataset_name = config["data"]["dataset_name"]
    dataset_cls = DATASET_REGISTRY.get(dataset_name)
    return dataset_cls(config=config, split=split)


def build_dataloaders(config: dict) -> dict[str, DataLoader]:
    loader_cfg = config["data"]["loader"]
    batch_size = int(loader_cfg["batch_size"])
    num_workers = int(loader_cfg["num_workers"])
    pin_memory = bool(loader_cfg["pin_memory"])
    drop_last = bool(loader_cfg["drop_last"])
    persistent_workers = bool(loader_cfg["persistent_workers"]) and num_workers > 0

    loaders: dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        dataset = build_dataset(config, split=split)
        shuffle = split == "train" and bool(loader_cfg["shuffle_train"]) and len(dataset) > 0
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last if split == "train" else False,
            persistent_workers=persistent_workers,
        )
    return loaders
