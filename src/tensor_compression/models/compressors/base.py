from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseCompressionModel(nn.Module, ABC):
    @abstractmethod
    def encode(self, inputs: torch.Tensor) -> dict:
        raise NotImplementedError

    @abstractmethod
    def decode(self, latent: dict) -> torch.Tensor:
        raise NotImplementedError

    def reconstruct(self, inputs: torch.Tensor) -> dict:
        latent = self.encode(inputs)
        reconstruction = self.decode(latent)
        latent["reconstruction"] = reconstruction
        return latent

    def forward(self, inputs: torch.Tensor) -> dict:
        return self.reconstruct(inputs)

