from __future__ import annotations

from tensor_compression.data.datasets.base import BaseTensorDataset
from tensor_compression.data.datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register("tensor_folder_4d")
class TensorFolder4DDataset(BaseTensorDataset):
    def __init__(self, config: dict, split: str) -> None:
        super().__init__(config=config, split=split)
        raise NotImplementedError(
            "4D dataset entry has been reserved but not implemented yet. "
            "Add a concrete loader here and switch `data.dataset_name` in config."
        )

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> dict:
        raise NotImplementedError

