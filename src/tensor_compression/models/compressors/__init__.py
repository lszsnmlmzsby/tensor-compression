from tensor_compression.registry import Registry

MODEL_REGISTRY = Registry("model")

from .conv_token_autoencoder_2d import ConvTokenAutoencoder2D  # noqa: E402,F401

__all__ = ["MODEL_REGISTRY", "ConvTokenAutoencoder2D"]
