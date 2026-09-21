#!/usr/bin/env python3
"""Train AH-GNO from a preprocessed gno_dataset.pkl.

The dataset format is the one produced by the research notebook: train/val/test
samples, normalized coordinates, nodal control area, and area-weighted
normalization statistics.
"""
from __future__ import annotations

import argparse
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from AH_GNO import (
    AreaWeightedHuberLoss,
    AreaWeightedNormalizer,
    build_model,
    compute_h_star,
    denormalize_dz,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def default_config(dataset):
    return {
        "k_max": int(dataset["history_k"]),
        "b_max": int(dataset["bundle_b"]),
        "hidden_channels": 64,
        "n_gno_layers": 4,
        "gno_radius": 0.15,
        "gno_mlp_layers": [256, 128],
        "gno_embed_channels": 32,
        "gno_transform_type": "linear",
        "gno_pos_embed_type": "transformer",
        "projection_channel_ratio": 4,
        "gno_use_open3d": False,
        "gno_use_torch_scatter": False,
        "learning_rate": 5e-4,
        "weight_decay": 1e-4,
        "warmup_epochs": 20,
        "scheduler_min_lr": 1e-6,
        "num_epochs": 300,
        "early_stop_patience": 80,
        "gradient_clip": 1.0,
        "loss_beta": 0.20,
        "lambda_horizon": 0.05,
        "h_star_rel_tol": 0.5,
        "h_star_abs_floor": 2e-5,
        "base_step": int(dataset.get("base_step", 2)),
        "base_dt_minutes": float(dataset.get("base_dt_minutes", 2.0)),
        "history_k": int(dataset["history_k"]),
        "bundle_b": int(dataset["bundle_b"]),
        "feature_names": list(dataset.get("feature_names", ["u", "v", "h", "z"])),
    }


def run_epoch(model, samples, optimizer, criterion, node_pos, node_area, normalizer, config, device, training):
    model.train(training)
    total = np.zeros(4, dtype=np.float64)
    order = np.random.permutation(len(samples)) if training else range(len(samples))

    for idx in order:
        sample = samples[idx]
        f = sample["f"].to(device)
        y = sample["y"].to(device)
        w = sample["w"].to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            out = model(pos=node_pos, f=f, node_area=node_area)
            pred = out["pred"]
            loss_pred = criterion(pred, y, w, node_area)

            with torch.no_grad():
                pred_phys = denormalize_dz(pred.detach(), normalizer)
                y_phys = denormalize_dz(y, normalizer)
                h_star, _ = compute_h_star(
                    pred_phys,
                    y_phys,
                    node_area,
                    rel_tol=config["h_star_rel_tol"],
                    abs_floor=config["h_star_abs_floor"],
                    b_max=config["b_max"],
                )

            target_h = torch.tensor([h_star - 1], device=device, dtype=torch.long)
            loss_h = F.cross_entropy(out["horizon_logits"].unsqueeze(0), target_h)
            loss = loss_pred + config["lambda_horizon"] * loss_h

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
                optimizer.step()

        total += [loss.item(), loss_pred.item(), loss_h.item(), h_star]

    return total / max(len(samples), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models/best_model.pth"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    with args.dataset.open("rb") as f:
        dataset = pickle.load(f)

    config = default_config(dataset)
    if args.epochs is not None:
        config["num_epochs"] = args.epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    node_pos = dataset["node_pos"].to(device)
    node_area = dataset["node_area"].to(device)
    normalizer = AreaWeightedNormalizer.from_state_dict(dataset["normalizer_state"])

    model = build_model(config).to(device)
    criterion = AreaWeightedHuberLoss(config["loss_beta"])
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        betas=(0.9, 0.95),
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, total_iters=config["warmup_epochs"]
            ),
            optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, config["num_epochs"] - config["warmup_epochs"]),
                eta_min=config["scheduler_min_lr"],
            ),
        ],
        milestones=[config["warmup_epochs"]],
    )

    best_val = float("inf")
    patience = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config["num_epochs"] + 1):
        train_stats = run_epoch(
            model, dataset["train_samples"], optimizer, criterion,
            node_pos, node_area, normalizer, config, device, True,
        )
        val_stats = run_epoch(
            model, dataset["val_samples"], optimizer, criterion,
            node_pos, node_area, normalizer, config, device, False,
        )
        scheduler.step()

        print(
            f"epoch={epoch:03d} train={train_stats[0]:.6e} "
            f"val={val_stats[0]:.6e} val_pred={val_stats[1]:.6e} "
            f"H*={val_stats[3]:.2f}"
        )

        # Paper workflow selects primarily on validation prediction loss.
        val_key = float(val_stats[1])
        if val_key < best_val:
            best_val = val_key
            patience = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "normalizer_state": dataset["normalizer_state"],
                    "best_val_prediction_loss": best_val,
                    "epoch": epoch,
                },
                args.output,
            )
        else:
            patience += 1
            if patience >= config["early_stop_patience"]:
                print("early stopping")
                break

    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
