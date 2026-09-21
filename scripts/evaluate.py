#!/usr/bin/env python3
"""Evaluate an AH-GNO checkpoint with area-weighted physical-space metrics.

The evaluator reports metrics on denormalized bed-elevation increments and does
not use the activity weights stored in training samples. Nodal control area is
the only spatial weighting used for evaluation.

Reported metrics:
- area-weighted RMSE;
- area-weighted MAE;
- area-weighted relative L2 error;
- area-weighted R2.

Both incremental dZ and cumulative bed-change metrics are reported. The script
also summarizes the learned effective history and write-back horizons.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch

from AH_GNO import AreaWeightedNormalizer, build_model, denormalize_dz


class WeightedMetricAccumulator:
    """Streaming weighted metrics for arbitrary arrays."""

    def __init__(self) -> None:
        self.sum_w = 0.0
        self.sum_abs = 0.0
        self.sum_sq_err = 0.0
        self.sum_y = 0.0
        self.sum_y2 = 0.0
        self.count = 0

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor,
    ) -> None:
        p = pred.detach().double().reshape(-1)
        y = target.detach().double().reshape(-1)
        w = weight.detach().double().reshape(-1)

        if p.numel() != y.numel() or p.numel() != w.numel():
            raise ValueError(
                f"Metric shape mismatch: pred={p.numel()}, "
                f"target={y.numel()}, weight={w.numel()}"
            )

        err = p - y
        self.sum_w += float(w.sum().item())
        self.sum_abs += float((w * err.abs()).sum().item())
        self.sum_sq_err += float((w * err.square()).sum().item())
        self.sum_y += float((w * y).sum().item())
        self.sum_y2 += float((w * y.square()).sum().item())
        self.count += int(p.numel())

    def compute(self) -> Dict[str, float | int]:
        if self.sum_w <= 0.0:
            return {
                "rmse": float("nan"),
                "mae": float("nan"),
                "relative_l2": float("nan"),
                "r2": float("nan"),
                "n_values": 0,
            }

        mse = self.sum_sq_err / self.sum_w
        rmse = math.sqrt(max(mse, 0.0))
        mae = self.sum_abs / self.sum_w

        if self.sum_y2 > 0.0:
            relative_l2 = math.sqrt(self.sum_sq_err / self.sum_y2)
        else:
            relative_l2 = float("nan")

        sst = self.sum_y2 - (self.sum_y * self.sum_y) / self.sum_w
        if sst > 0.0:
            r2 = 1.0 - self.sum_sq_err / sst
        else:
            r2 = float("nan")

        return {
            "rmse": rmse,
            "mae": mae,
            "relative_l2": relative_l2,
            "r2": r2,
            "n_values": self.count,
        }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--split",
        choices=("train", "val", "validation", "test"),
        default="test",
    )
    p.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("results/evaluation.json"),
    )
    p.add_argument(
        "--per-horizon-csv",
        type=Path,
        default=None,
        help="Optional CSV path for per-horizon increment/cumulative metrics.",
    )
    p.add_argument(
        "--per-case-csv",
        type=Path,
        default=None,
        help="Optional CSV path for scenario-wise increment metrics.",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional debugging limit; omit for full evaluation.",
    )
    return p.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable.")
    return torch.device(name)


def split_key(split: str) -> str:
    return "val_samples" if split in {"val", "validation"} else f"{split}_samples"


def repeated_area(node_area: torch.Tensor, n_columns: int) -> torch.Tensor:
    return node_area[:, None].expand(-1, n_columns)


def scalar_area(node_area: torch.Tensor) -> torch.Tensor:
    return node_area


def safe_float(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().item()
    return x


def write_per_horizon_csv(
    path: Path,
    increment,
    cumulative,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "horizon_step",
        "increment_rmse",
        "increment_mae",
        "increment_relative_l2",
        "increment_r2",
        "cumulative_rmse",
        "cumulative_mae",
        "cumulative_relative_l2",
        "cumulative_r2",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, (inc_acc, cum_acc) in enumerate(zip(increment, cumulative), start=1):
            inc = inc_acc.compute()
            cum = cum_acc.compute()
            writer.writerow(
                {
                    "horizon_step": i,
                    "increment_rmse": inc["rmse"],
                    "increment_mae": inc["mae"],
                    "increment_relative_l2": inc["relative_l2"],
                    "increment_r2": inc["r2"],
                    "cumulative_rmse": cum["rmse"],
                    "cumulative_mae": cum["mae"],
                    "cumulative_relative_l2": cum["relative_l2"],
                    "cumulative_r2": cum["r2"],
                }
            )


def write_per_case_csv(path: Path, per_case) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "case_id",
        "rmse",
        "mae",
        "relative_l2",
        "r2",
        "n_values",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for case_id in sorted(per_case):
            m = per_case[case_id].compute()
            writer.writerow({"case_id": case_id, **m})


@torch.no_grad()
def main():
    args = parse_args()
    device = choose_device(args.device)

    with args.dataset.open("rb") as f:
        dataset = pickle.load(f)

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint does not contain 'model_state_dict'.")
    if "config" not in checkpoint:
        raise KeyError("Checkpoint does not contain 'config'.")

    config = checkpoint["config"]
    k_max = int(config["k_max"])
    b_max = int(config["b_max"])

    if int(dataset["history_k"]) != k_max:
        raise ValueError(
            f"K mismatch: dataset={dataset['history_k']} checkpoint={k_max}"
        )
    if int(dataset["bundle_b"]) != b_max:
        raise ValueError(
            f"B mismatch: dataset={dataset['bundle_b']} checkpoint={b_max}"
        )

    key = split_key(args.split)
    if key not in dataset:
        raise KeyError(f"Dataset does not contain {key!r}.")
    samples = dataset[key]
    if args.max_samples is not None:
        samples = samples[: max(0, args.max_samples)]
    if not samples:
        raise ValueError(f"No samples available for split {args.split!r}.")

    node_pos = dataset["node_pos"].to(device=device, dtype=torch.float32)
    node_area = dataset["node_area"].to(device=device, dtype=torch.float32)

    normalizer_state = checkpoint.get(
        "normalizer_state",
        dataset["normalizer_state"],
    )
    normalizer = AreaWeightedNormalizer.from_state_dict(normalizer_state)

    model = build_model(
        config,
        checkpoint["model_state_dict"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    overall_increment = WeightedMetricAccumulator()
    overall_cumulative = WeightedMetricAccumulator()
    accepted_increment = WeightedMetricAccumulator()
    accepted_endpoint = WeightedMetricAccumulator()

    per_horizon_increment = [
        WeightedMetricAccumulator() for _ in range(b_max)
    ]
    per_horizon_cumulative = [
        WeightedMetricAccumulator() for _ in range(b_max)
    ]
    per_case = defaultdict(WeightedMetricAccumulator)

    effective_b_values = []
    effective_k_values = []
    effective_k_count_values = []

    for sample_index, sample in enumerate(samples):
        f = sample["f"].to(device=device, dtype=torch.float32)
        y_nrm = sample["y"].to(device=device, dtype=torch.float32)

        out = model(
            pos=node_pos,
            f=f,
            node_area=node_area,
        )
        pred_nrm = out["pred"]

        pred = denormalize_dz(pred_nrm, normalizer)
        target = denormalize_dz(y_nrm, normalizer)

        if pred.shape != target.shape:
            raise ValueError(
                f"Sample {sample_index}: pred shape {tuple(pred.shape)} "
                f"!= target shape {tuple(target.shape)}"
            )
        if pred.shape[1] != b_max:
            raise ValueError(
                f"Sample {sample_index}: prediction B={pred.shape[1]} "
                f"but checkpoint B={b_max}"
            )

        area_bundle = repeated_area(node_area, b_max)
        overall_increment.update(pred, target, area_bundle)

        pred_cum = torch.cumsum(pred, dim=1)
        target_cum = torch.cumsum(target, dim=1)
        overall_cumulative.update(pred_cum, target_cum, area_bundle)

        for b in range(b_max):
            per_horizon_increment[b].update(
                pred[:, b],
                target[:, b],
                scalar_area(node_area),
            )
            per_horizon_cumulative[b].update(
                pred_cum[:, b],
                target_cum[:, b],
                scalar_area(node_area),
            )

        eff_b = int(out["effective_b"])
        eff_b = max(1, min(eff_b, b_max))
        effective_b_values.append(eff_b)

        if "effective_k" in out:
            effective_k_values.append(float(out["effective_k"]))
        if "effective_k_count" in out:
            effective_k_count_values.append(int(out["effective_k_count"]))

        accepted_increment.update(
            pred[:, :eff_b],
            target[:, :eff_b],
            repeated_area(node_area, eff_b),
        )
        accepted_endpoint.update(
            pred_cum[:, eff_b - 1],
            target_cum[:, eff_b - 1],
            scalar_area(node_area),
        )

        case_id = str(sample.get("case_id", "unknown"))
        per_case[case_id].update(pred, target, area_bundle)

    def list_stats(values: Iterable[float]) -> Optional[Dict[str, float]]:
        values = list(values)
        if not values:
            return None
        a = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(a.mean()),
            "std": float(a.std()),
            "min": float(a.min()),
            "median": float(np.median(a)),
            "max": float(a.max()),
        }

    result = {
        "split": "val" if args.split == "validation" else args.split,
        "n_samples": len(samples),
        "n_nodes": int(node_area.numel()),
        "k_max": k_max,
        "b_max": b_max,
        "metrics": {
            "increment": overall_increment.compute(),
            "cumulative": overall_cumulative.compute(),
            "accepted_increment": accepted_increment.compute(),
            "accepted_cumulative_endpoint": accepted_endpoint.compute(),
        },
        "adaptive_horizon": {
            "effective_b": list_stats(effective_b_values),
            "effective_k": list_stats(effective_k_values),
            "effective_k_count": list_stats(effective_k_count_values),
        },
        "per_horizon": [
            {
                "step": b + 1,
                "increment": per_horizon_increment[b].compute(),
                "cumulative": per_horizon_cumulative[b].compute(),
            }
            for b in range(b_max)
        ],
        "per_case_increment": {
            case_id: acc.compute()
            for case_id, acc in sorted(per_case.items())
        },
        "metric_definition": {
            "spatial_weight": "nodal control area only",
            "training_activity_weights_used": False,
            "physical_space": True,
            "relative_l2": "sqrt(sum(A*e^2) / sum(A*y^2))",
            "r2": "1 - sum(A*e^2) / sum(A*(y-y_bar_A)^2)",
        },
        "checkpoint": {
            "path": str(args.checkpoint),
            "epoch": safe_float(checkpoint.get("epoch")),
            "best_val_prediction_loss": safe_float(
                checkpoint.get("best_val_prediction_loss")
            ),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=True)

    if args.per_horizon_csv is not None:
        write_per_horizon_csv(
            args.per_horizon_csv,
            per_horizon_increment,
            per_horizon_cumulative,
        )

    if args.per_case_csv is not None:
        write_per_case_csv(args.per_case_csv, per_case)

    inc = result["metrics"]["increment"]
    cum = result["metrics"]["cumulative"]
    acc = result["metrics"]["accepted_increment"]
    bstats = result["adaptive_horizon"]["effective_b"]

    print("AH-GNO evaluation")
    print(f"  split: {result['split']}")
    print(f"  samples: {result['n_samples']}")
    print(
        "  increment: "
        f"RMSE={inc['rmse']:.6e}, "
        f"MAE={inc['mae']:.6e}, "
        f"RelL2={inc['relative_l2']:.6e}, "
        f"R2={inc['r2']:.6f}"
    )
    print(
        "  cumulative: "
        f"RMSE={cum['rmse']:.6e}, "
        f"MAE={cum['mae']:.6e}, "
        f"RelL2={cum['relative_l2']:.6e}, "
        f"R2={cum['r2']:.6f}"
    )
    print(
        "  accepted increment: "
        f"RMSE={acc['rmse']:.6e}, "
        f"MAE={acc['mae']:.6e}, "
        f"RelL2={acc['relative_l2']:.6e}, "
        f"R2={acc['r2']:.6f}"
    )
    if bstats is not None:
        print(
            "  effective B: "
            f"mean={bstats['mean']:.3f}, "
            f"median={bstats['median']:.3f}, "
            f"min={bstats['min']:.0f}, "
            f"max={bstats['max']:.0f}"
        )
    print(f"  saved: {args.output}")


if __name__ == "__main__":
    main()
