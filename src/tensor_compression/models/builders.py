from __future__ import annotations

from tensor_compression.models.compressors import MODEL_REGISTRY


def build_model(config: dict):
    model_name = config["model"]["name"]
    model_cls = MODEL_REGISTRY.get(model_name)
    return model_cls(config=config)

