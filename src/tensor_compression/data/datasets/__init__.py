from tensor_compression.registry import Registry

DATASET_REGISTRY = Registry("dataset")

from .tensor_folder_2d import TensorFolder2DDataset  # noqa: E402,F401
from .tensor_folder_3d import TensorFolder3DDataset  # noqa: E402,F401
from .tensor_folder_4d import TensorFolder4DDataset  # noqa: E402,F401

__all__ = [
    "DATASET_REGISTRY",
    "TensorFolder2DDataset",
    "TensorFolder3DDataset",
    "TensorFolder4DDataset",
]

