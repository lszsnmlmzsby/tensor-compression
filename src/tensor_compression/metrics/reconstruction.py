from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_reconstruction_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    mse = float(F.mse_loss(prediction, target).detach().cpu().item())
    mae = float(F.l1_loss(prediction, target).detach().cpu().item())
    denom = torch.abs(target) + 1.0e-6
    relative_l1 = float((torch.abs(prediction - target) / denom).mean().detach().cpu().item())
    max_abs_error = float(torch.abs(prediction - target).max().detach().cpu().item())
    target_max = float(target.max().detach().cpu().item())
    target_min = float(target.min().detach().cpu().item())
    data_range = max(target_max - target_min, 1.0e-6)
    psnr = 20.0 * torch.log10(torch.tensor(data_range)) - 10.0 * torch.log10(
        torch.tensor(max(mse, 1.0e-12))
    )
    return {
        "mse": mse,
        "mae": mae,
        "relative_l1": relative_l1,
        "max_abs_error": max_abs_error,
        "psnr": float(psnr.detach().cpu().item()),
    }
