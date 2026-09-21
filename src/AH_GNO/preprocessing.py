"""Area-weighted preprocessing utilities for AH-GNO."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch


def compute_node_areas(x: np.ndarray, y: np.ndarray, elements: np.ndarray) -> np.ndarray:
    """Lumped nodal control area: one third of each adjacent triangle area."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    elements = np.asarray(elements, dtype=np.int64)
    if elements.ndim != 2 or elements.shape[1] != 3:
        raise NotImplementedError("Only triangular elements are supported.")

    p0 = np.column_stack([x[elements[:, 0]], y[elements[:, 0]]])
    p1 = np.column_stack([x[elements[:, 1]], y[elements[:, 1]]])
    p2 = np.column_stack([x[elements[:, 2]], y[elements[:, 2]]])
    cross = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])
    tri_area = 0.5 * np.abs(cross)

    node_area = np.zeros(len(x), dtype=np.float64)
    share = tri_area / 3.0
    for k in range(3):
        np.add.at(node_area, elements[:, k], share)

    bad = node_area <= 0
    if np.any(bad):
        node_area[bad] = node_area[~bad].mean() if np.any(~bad) else 1.0
    return node_area.astype(np.float32)


def normalize_coordinates(x: np.ndarray, y: np.ndarray) -> Tuple[torch.Tensor, Dict[str, float]]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    xn = (x - x_min) / (x_max - x_min + 1e-8)
    yn = (y - y_min) / (y_max - y_min + 1e-8)
    pos = torch.tensor(np.column_stack([xn, yn]), dtype=torch.float32)
    return pos, {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}


class AreaWeightedNormalizer:
    """Z-score statistics weighted by nodal control area."""

    def __init__(self, eps: float = 1e-5) -> None:
        self.stats = {}
        self.eps = eps

    def fit(self, data_dict_with_area) -> None:
        for key, (values, weights) in data_dict_with_area.items():
            values = np.asarray(values, dtype=np.float64).ravel()
            weights = np.asarray(weights, dtype=np.float64).ravel()
            if values.shape != weights.shape:
                raise ValueError(f"{key}: shape mismatch {values.shape} vs {weights.shape}")
            total = weights.sum()
            if total <= 0:
                raise ValueError(f"{key}: non-positive total weight")
            mean = float((weights * values).sum() / total)
            var = float((weights * (values - mean) ** 2).sum() / total)
            self.stats[key] = {"mean": mean, "std": float(np.sqrt(var))}

    def normalize(self, data, key):
        s = self.stats[key]
        return (data - s["mean"]) / (s["std"] + self.eps)

    def denormalize(self, data, key):
        s = self.stats[key]
        return data * (s["std"] + self.eps) + s["mean"]

    def state_dict(self):
        return {"stats": self.stats, "eps": self.eps}

    @classmethod
    def from_state_dict(cls, state):
        obj = cls(eps=state.get("eps", 1e-5))
        obj.stats = state["stats"]
        return obj


def weighted_quantile(values, weights, q: float) -> float:
    values = np.asarray(values, dtype=np.float64).ravel()
    weights = np.asarray(weights, dtype=np.float64).ravel()
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cutoff = q * np.cumsum(w)[-1]
    idx = np.searchsorted(np.cumsum(w), cutoff)
    return float(v[min(idx, len(v) - 1)])
