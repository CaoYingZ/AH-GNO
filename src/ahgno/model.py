"""Adaptive-Horizon Graph Neural Operator (AH-GNO).

This module matches the area-weighted formulation used in the manuscript:
- nodal control-area quadrature inside the graph integral operator;
- local area normalization inside each search neighborhood;
- area-weighted global descriptors for the history selector and horizon head;
- sparse-softmax history selection and learned write-back horizon.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralop.layers.channel_mlp import ChannelMLP
from neuralop.layers.embeddings import SinusoidalEmbedding
from neuralop.layers.neighbor_search import NeighborSearch


GNO_ACCUM_FP32 = True


class AreaWeightedIntegralTransform(nn.Module):
    """Local graph integral using nodal control area as quadrature measure.

    For query node i, the discrete operator is
        sum_j A_j K(x_i-x_j) f_j / sum_j A_j,
    over neighbors inside the fixed radius in normalized coordinates.
    """

    def __init__(
        self,
        kernel_mlp: nn.Module,
        radius: float,
        coord_dim: int,
        transform_type: str = "linear",
        pos_embed: Optional[nn.Module] = None,
        use_open3d: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_mlp = kernel_mlp
        self.radius = radius
        self.coord_dim = coord_dim
        self.transform_type = transform_type
        self.pos_embed = pos_embed
        self.neighbor_search = NeighborSearch(use_open3d=use_open3d)
        self._cache_key = None
        self._cached_result = None

    def _get_cached_result(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        key = (y.data_ptr(), x.data_ptr(), y.shape[0], x.shape[0])
        if self._cache_key == key and self._cached_result is not None:
            return self._cached_result

        result = self.neighbor_search(x, y, radius=self.radius)
        nbr_idx = result["neighbors_index"]
        row_splits = result["neighbors_row_splits"]
        counts = row_splits[1:] - row_splits[:-1]
        query_idx = torch.repeat_interleave(
            torch.arange(x.shape[0], device=x.device), counts
        )
        with torch.no_grad():
            diff = x[query_idx] - y[nbr_idx]
            diff_emb = self.pos_embed(diff) if self.pos_embed is not None else diff

        result["query_idx"] = query_idx
        result["diff_emb"] = diff_emb
        self._cache_key = key
        self._cached_result = result
        return result

    def invalidate_cache(self) -> None:
        self._cache_key = None
        self._cached_result = None

    def forward(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        f_y: torch.Tensor,
        area_y: torch.Tensor,
    ) -> torch.Tensor:
        result = self._get_cached_result(x, y)
        nbr_idx = result["neighbors_index"]
        query_idx = result["query_idx"]
        kernel_weights = self.kernel_mlp(result["diff_emb"])

        area = area_y[nbr_idx].unsqueeze(-1)
        msg = area * f_y[nbr_idx] * kernel_weights
        if GNO_ACCUM_FP32:
            msg = msg.float()

        out = torch.zeros(
            x.shape[0], msg.shape[-1], device=x.device, dtype=msg.dtype
        )
        out.index_add_(0, query_idx, msg)

        denom = torch.zeros(x.shape[0], 1, device=x.device, dtype=msg.dtype)
        denom.index_add_(0, query_idx, area.to(msg.dtype))
        return out / (denom + 1e-8)


class AreaWeightedGNOBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        coord_dim: int = 2,
        radius: float = 0.15,
        transform_type: str = "linear",
        channel_mlp_layers=None,
        pos_embedding_type: str = "transformer",
        pos_embedding_channels: int = 32,
        use_open3d_neighbor_search: bool = False,
        use_torch_scatter_reduce: bool = False,
    ) -> None:
        super().__init__()
        del use_torch_scatter_reduce  # kept for checkpoint/config compatibility
        if channel_mlp_layers is None:
            channel_mlp_layers = [128, 64]

        if pos_embedding_type is not None and pos_embedding_channels > 0:
            pos_embed = SinusoidalEmbedding(
                in_channels=coord_dim,
                num_frequencies=pos_embedding_channels,
                embedding_type=pos_embedding_type,
            )
            embed_dim = pos_embed.out_channels
        else:
            pos_embed = None
            embed_dim = coord_dim

        mlp_in = embed_dim if transform_type in ("linear", "linear_kernelonly") else embed_dim + in_channels
        layers = []
        prev = mlp_in
        for width in channel_mlp_layers:
            layers.extend([nn.Linear(prev, width), nn.GELU()])
            prev = width
        layers.append(nn.Linear(prev, out_channels))

        self.integral_transform = AreaWeightedIntegralTransform(
            kernel_mlp=nn.Sequential(*layers),
            radius=radius,
            coord_dim=coord_dim,
            transform_type=transform_type,
            pos_embed=pos_embed,
            use_open3d=use_open3d_neighbor_search,
        )

    def forward(self, y, x, f_y, area_y):
        return self.integral_transform(y=y, x=x, f_y=f_y, area_y=area_y)


class TemporalKSelector(nn.Module):
    """Sparse softmax selector for the effective history length."""

    def __init__(self, hidden: int, k_max: int, k_relative_threshold: float = 0.5) -> None:
        super().__init__()
        self.k_max = k_max
        self.k_threshold = k_relative_threshold / float(k_max)
        self.k_net = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, k_max),
        )

    def forward(self, global_feat: torch.Tensor) -> torch.Tensor:
        raw_k = F.softmax(self.k_net(global_feat), dim=0)
        mask = (raw_k >= self.k_threshold).float()
        if mask.sum() < 0.5:
            mask = torch.zeros_like(raw_k)
            mask[raw_k.argmax()] = 1.0
        masked = raw_k * mask
        sparse = masked / masked.sum().clamp(min=1e-8)
        return raw_k + (sparse - raw_k).detach()


class HorizonHead(nn.Module):
    """Predicts a horizon distribution and monotone survival weights.

    The manuscript uses ``mode='softmax'``. ``ordinal`` is retained only for
    compatibility with older checkpoints created during development.
    """

    def __init__(self, hidden: int, b_max: int, mode: str = "softmax") -> None:
        super().__init__()
        if mode not in ("softmax", "ordinal"):
            raise ValueError(f"Unknown horizon mode: {mode}")
        self.b_max = b_max
        self.mode = mode
        out_dim = b_max if mode == "softmax" else b_max - 1
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, out_dim),
        )

    def forward(self, global_feat: torch.Tensor):
        logits = self.net(global_feat)
        if self.mode == "softmax":
            horizon_prob = F.softmax(logits, dim=0)
            b_weights = torch.flip(
                torch.cumsum(torch.flip(horizon_prob, dims=[0]), dim=0), dims=[0]
            )
        else:
            cum_probs = torch.sigmoid(logits)
            one = torch.ones(1, device=logits.device, dtype=logits.dtype)
            zero = torch.zeros(1, device=logits.device, dtype=logits.dtype)
            b_weights = torch.cat([one, cum_probs], dim=0)
            extended = torch.cat([one, cum_probs, zero], dim=0)
            horizon_prob = (extended[:-1] - extended[1:]).clamp(min=0)
            horizon_prob = horizon_prob / horizon_prob.sum().clamp(min=1e-8)

        effective_b = max(1, int((b_weights > 0.5).sum().item()))
        return logits, horizon_prob, b_weights, effective_b


class LearnedHorizonGNO(nn.Module):
    def __init__(
        self,
        k_max: int = 7,
        b_max: int = 10,
        n_features: int = 4,
        hidden_channels: int = 64,
        n_gno_layers: int = 4,
        coord_dim: int = 2,
        gno_radius: float = 0.15,
        gno_transform_type: str = "linear",
        gno_mlp_layers=None,
        gno_embed_channels: int = 32,
        gno_pos_embed_type: str = "transformer",
        projection_channel_ratio: int = 4,
        gno_use_open3d: bool = False,
        gno_use_torch_scatter: bool = False,
        horizon_mode: str = "softmax",
    ) -> None:
        super().__init__()
        if gno_mlp_layers is None:
            gno_mlp_layers = [128, 64]

        self.k_max = k_max
        self.b_max = b_max
        self.n_features = n_features
        self.hidden = hidden_channels

        self.lifting = ChannelMLP(
            in_channels=n_features,
            out_channels=hidden_channels,
            hidden_channels=hidden_channels * 2,
            n_layers=3,
            n_dim=1,
        )

        self.gno_layers = nn.ModuleList([
            AreaWeightedGNOBlock(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                coord_dim=coord_dim,
                radius=gno_radius,
                transform_type=gno_transform_type,
                channel_mlp_layers=gno_mlp_layers,
                pos_embedding_type=gno_pos_embed_type,
                pos_embedding_channels=gno_embed_channels,
                use_open3d_neighbor_search=gno_use_open3d,
                use_torch_scatter_reduce=gno_use_torch_scatter,
            )
            for _ in range(n_gno_layers)
        ])

        self.k_selector = TemporalKSelector(hidden_channels, k_max)
        self.horizon_head = HorizonHead(hidden_channels, b_max, mode=horizon_mode)
        self.projection = ChannelMLP(
            in_channels=hidden_channels,
            out_channels=b_max,
            hidden_channels=projection_channel_ratio * hidden_channels,
            n_layers=2,
            n_dim=1,
        )

    def forward(self, pos: torch.Tensor, f: torch.Tensor, node_area: torch.Tensor):
        n_nodes = f.shape[0]
        k_max = self.k_max
        c_feat = self.n_features

        f_per_step = f.view(n_nodes, k_max, c_feat).permute(1, 0, 2).contiguous()
        f_flat = f_per_step.reshape(k_max * n_nodes, c_feat)
        h_flat = self.lifting(
            f_flat.unsqueeze(0).permute(0, 2, 1)
        ).squeeze(0).permute(1, 0)
        lifted = h_flat.view(k_max, n_nodes, self.hidden)

        area_col = node_area.unsqueeze(-1)
        area_sum = node_area.sum().clamp(min=1e-12)

        global_feat_k = (lifted[-1] * area_col).sum(dim=0) / area_sum
        k_weights = self.k_selector(global_feat_k)
        h = (lifted * k_weights.view(-1, 1, 1)).sum(dim=0)

        for gno in self.gno_layers:
            h = h + F.gelu(gno(y=pos, x=pos, f_y=h, area_y=node_area))

        pred = self.projection(
            h.unsqueeze(0).permute(0, 2, 1)
        ).squeeze(0).permute(1, 0)

        global_feat_b = (h * area_col).sum(dim=0) / area_sum
        logits, horizon_prob, b_weights, effective_b = self.horizon_head(global_feat_b)

        effective_k_count = int((k_weights > 1e-6).sum().item())
        t_idx = torch.arange(1, k_max + 1, dtype=torch.float32, device=f.device)
        effective_k = float((t_idx * k_weights).sum().item())

        return {
            "pred": pred,
            "k_weights": k_weights,
            "b_weights": b_weights,
            "horizon_logits": logits,
            "horizon_prob": horizon_prob,
            "effective_k": effective_k,
            "effective_k_count": effective_k_count,
            "effective_b": effective_b,
        }

    def invalidate_cache(self) -> None:
        for gno in self.gno_layers:
            gno.integral_transform.invalidate_cache()


def detect_horizon_mode(state_dict: dict, b_max: int) -> str:
    keys = [k for k in state_dict if k.startswith("horizon_head.") and k.endswith(".weight")]
    if not keys:
        return "softmax"
    last_key = keys[-1]
    out_dim = state_dict[last_key].shape[0]
    if out_dim == b_max:
        return "softmax"
    if out_dim == b_max - 1:
        return "ordinal"
    raise ValueError(f"Cannot infer horizon mode from {last_key}: out_dim={out_dim}, b_max={b_max}")


def build_model(config: dict, state_dict: Optional[dict] = None) -> LearnedHorizonGNO:
    b_max = int(config["b_max"])
    mode = detect_horizon_mode(state_dict, b_max) if state_dict is not None else config.get("horizon_mode", "softmax")
    return LearnedHorizonGNO(
        k_max=int(config["k_max"]),
        b_max=b_max,
        hidden_channels=int(config.get("hidden_channels", 64)),
        n_gno_layers=int(config.get("n_gno_layers", 4)),
        gno_radius=float(config.get("gno_radius", 0.15)),
        gno_transform_type=config.get("gno_transform_type", "linear"),
        gno_mlp_layers=config.get("gno_mlp_layers", [128, 64]),
        gno_embed_channels=int(config.get("gno_embed_channels", 32)),
        gno_pos_embed_type=config.get("gno_pos_embed_type", "transformer"),
        projection_channel_ratio=int(config.get("projection_channel_ratio", 4)),
        gno_use_open3d=bool(config.get("gno_use_open3d", False)),
        gno_use_torch_scatter=bool(config.get("gno_use_torch_scatter", False)),
        horizon_mode=mode,
    )
