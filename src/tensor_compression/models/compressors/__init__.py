from tensor_compression.registry import Registry

MODEL_REGISTRY = Registry("model")

from .conv_token_autoencoder_2d import ConvTokenAutoencoder2D  # noqa: E402,F401
from .conv_token_autoencoder_3d import ConvTokenAutoencoder3D  # noqa: E402,F401
from .factorized_autoencoder_4d import FactorizedAutoencoder4D  # noqa: E402,F401

__all__ = [
    "MODEL_REGISTRY",
    "ConvTokenAutoencoder2D",
    "ConvTokenAutoencoder3D",
    "FactorizedAutoencoder4D",
]

