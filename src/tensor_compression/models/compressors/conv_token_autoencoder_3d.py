from __future__ import annotations

from tensor_compression.models.compressors import MODEL_REGISTRY
from tensor_compression.models.compressors.base import BaseCompressionModel


@MODEL_REGISTRY.register("conv_token_autoencoder_3d")
class ConvTokenAutoencoder3D(BaseCompressionModel):
    def __init__(self, config: dict) -> None:
        super().__init__()
        raise NotImplementedError(
            "3D compressor entry has been reserved but not implemented yet. "
            "Use this class name in config after adding a concrete 3D encoder/decoder."
        )

    def encode(self, inputs):
        raise NotImplementedError

    def decode(self, latent):
        raise NotImplementedError

