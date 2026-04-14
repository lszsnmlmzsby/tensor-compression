from __future__ import annotations

from tensor_compression.models.compressors import MODEL_REGISTRY
from tensor_compression.models.compressors.base import BaseCompressionModel


@MODEL_REGISTRY.register("factorized_autoencoder_4d")
class FactorizedAutoencoder4D(BaseCompressionModel):
    def __init__(self, config: dict) -> None:
        super().__init__()
        raise NotImplementedError(
            "4D compressor entry has been reserved but not implemented yet. "
            "Use this class name in config after adding a concrete factorized 4D model."
        )

    def encode(self, inputs):
        raise NotImplementedError

    def decode(self, latent):
        raise NotImplementedError

