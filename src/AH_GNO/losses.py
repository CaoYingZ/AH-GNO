"""Training loss and horizon-label utilities for AH-GNO."""
from __future__ import annotations

import torch
import torch.nn as nn


class AreaWeightedHuberLoss(nn.Module):
    def __init__(self, beta: float = 0.20) -> None:
        super().__init__()
        self.beta = beta

    def forward(self, pred, target, node_weights, node_area):
        diff = torch.abs(pred - target)
        loss = torch.where(
            diff < self.beta,
            0.5 * diff ** 2 / self.beta,
            diff - 0.5 * self.beta,
        )
        area = node_area.unsqueeze(-1)
        combined = node_weights * area
        norm = combined.sum().clamp(min=1e-8) * loss.shape[-1]
        return (loss * combined).sum() / norm


def denormalize_dz(x: torch.Tensor, normalizer) -> torch.Tensor:
    stat = normalizer.stats["dz"]
    return x * (stat["std"] + normalizer.eps) + stat["mean"]


def compute_h_star(
    pred_phys: torch.Tensor,
    target_phys: torch.Tensor,
    node_area: torch.Tensor,
    rel_tol: float = 0.5,
    abs_floor: float = 2e-5,
    b_max: int | None = None,
):
    """Longest reliable prefix using area-weighted per-step RMSE."""
    area = node_area.unsqueeze(-1)
    area_sum = node_area.sum().clamp(min=1e-12)
    step_mse = (((pred_phys - target_phys) ** 2) * area).sum(dim=0) / area_sum
    step_true = torch.sqrt(((target_phys ** 2) * area).sum(dim=0) / area_sum + 1e-12)
    step_rmse = torch.sqrt(step_mse + 1e-12)
    tol = torch.clamp(rel_tol * step_true, min=abs_floor)
    good = (step_rmse.detach() <= tol.detach()).float()
    prefix_good = torch.cumprod(good, dim=0)
    b = pred_phys.shape[1] if b_max is None else b_max
    h_star = max(1, min(b, int(prefix_good.sum().item())))
    return h_star, step_rmse.detach()
