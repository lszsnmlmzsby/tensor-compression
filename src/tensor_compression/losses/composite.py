from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class CompositeReconstructionLoss(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        loss_cfg = config["loss"]
        self.weights = {key: float(value) for key, value in loss_cfg["weights"].items()}
        self.eps = float(loss_cfg.get("eps", 1.0e-6))

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        terms: dict[str, torch.Tensor] = {}
        if self.weights.get("mse", 0.0) > 0:
            terms["mse"] = F.mse_loss(prediction, target)
        if self.weights.get("l1", 0.0) > 0:
            terms["l1"] = F.l1_loss(prediction, target)
        if self.weights.get("relative_l1", 0.0) > 0:
            terms["relative_l1"] = torch.mean(torch.abs(prediction - target) / (torch.abs(target) + self.eps))
        if self.weights.get("gradient", 0.0) > 0:
            terms["gradient"] = self._gradient_difference(prediction, target)

        total = prediction.new_tensor(0.0)
        for name, value in terms.items():
            total = total + self.weights[name] * value
        terms["total"] = total
        return terms

    def _gradient_difference(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_dx = prediction[..., :, 1:] - prediction[..., :, :-1]
        pred_dy = prediction[..., 1:, :] - prediction[..., :-1, :]
        tgt_dx = target[..., :, 1:] - target[..., :, :-1]
        tgt_dy = target[..., 1:, :] - target[..., :-1, :]
        return F.l1_loss(pred_dx, tgt_dx) + F.l1_loss(pred_dy, tgt_dy)

