from __future__ import annotations

from tensor_compression.losses.composite import CompositeReconstructionLoss


def build_loss(config: dict):
    loss_name = config["loss"]["name"]
    if loss_name != "composite_reconstruction_loss":
        raise ValueError(f"Unsupported loss: {loss_name}")
    return CompositeReconstructionLoss(config=config)

