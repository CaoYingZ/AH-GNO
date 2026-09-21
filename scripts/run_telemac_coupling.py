#!/usr/bin/env python3
"""Minimal TELEMAC-2D + AH-GNO online coupling example.

This is the paper-release version. It uses the area-weighted model and adaptive
write-back horizon by default. TELEMAC-MASCARET is not installed by pip; source
its environment before running this script.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
from mpi4py import MPI

from ahgno import AreaWeightedNormalizer, build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--telemac-root", type=Path, required=True)
    p.add_argument("--cas", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True, help="gno_dataset.pkl")
    p.add_argument("--writeback-cap", type=int, default=None, help="Optional hard cap on effective B")
    p.add_argument("--min-water-depth", type=float, default=0.20)
    p.add_argument("--max-dz", type=float, default=0.20)
    p.add_argument("--output", type=Path, default=Path("coupling_summary.pkl"))
    return p.parse_args()


def add_telemac_paths(root: Path):
    py = root / "scripts" / "python3"
    api = root / "builds" / "ubuntu_openmpi" / "wrap_api" / "lib"
    for path in (py, api):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def cumulative_writeback(bundle: np.ndarray, n_steps: int) -> np.ndarray:
    return np.sum(bundle[:, :n_steps], axis=1)


def postprocess_dz(dz, h, z, surface, min_depth, max_dz):
    dz = np.asarray(dz, dtype=np.float64).copy()
    if max_dz is not None:
        dz = np.clip(dz, -max_dz, max_dz)
    dz = np.minimum(dz, h - min_depth)
    z_new = z + dz
    h_new = surface - z_new
    dry = h_new < min_depth
    if np.any(dry):
        h_new[dry] = min_depth
        z_new[dry] = surface[dry] - min_depth
        dz = z_new - z
    return dz, z_new, h_new


def main():
    args = parse_args()
    add_telemac_paths(args.telemac_root)
    from telapy.api.t2d import Telemac2d

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    with args.dataset.open("rb") as f:
        dataset = pickle.load(f)

    config = checkpoint["config"]
    model = build_model(config, checkpoint["model_state_dict"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    normalizer_state = checkpoint.get("normalizer_state", dataset["normalizer_state"])
    normalizer = AreaWeightedNormalizer.from_state_dict(normalizer_state)
    node_pos = dataset["node_pos"].to(device)
    node_area = dataset["node_area"].to(device=device, dtype=torch.float32)

    k_max = int(config["k_max"])
    b_max = int(config["b_max"])
    base_dt_seconds = float(config.get("base_dt_minutes", dataset.get("base_dt_minutes", 2.0))) * 60.0
    cap = b_max if args.writeback_cap is None else max(1, min(args.writeback_cap, b_max))

    means = torch.tensor(
        [normalizer.stats[k]["mean"] for _ in range(k_max) for k in ("u", "v", "h", "z")],
        device=device, dtype=torch.float32,
    )
    stds = torch.tensor(
        [normalizer.stats[k]["std"] + normalizer.eps for _ in range(k_max) for k in ("u", "v", "h", "z")],
        device=device, dtype=torch.float32,
    )
    dz_mean = float(normalizer.stats["dz"]["mean"])
    dz_std = float(normalizer.stats["dz"]["std"] + normalizer.eps)

    @torch.no_grad()
    def predict(history):
        raw = []
        for snap in history:
            raw.append(np.column_stack([snap["u"], snap["v"], snap["h"], snap["z"]]).astype(np.float32))
        f = torch.from_numpy(np.concatenate(raw, axis=1)).to(device)
        f = (f - means) / stds
        out = model(pos=node_pos, f=f, node_area=node_area)
        bundle = (out["pred"] * dz_std + dz_mean).cpu().numpy()
        return bundle, out

    os.chdir(args.cas.parent)
    t2d = Telemac2d(args.cas.name, user_fortran=None, comm=MPI.COMM_WORLD, stdout=6, recompile=False)
    t2d.set_case()
    t2d.init_state_default()

    npoin = int(t2d.mpi_get_npoin()) if getattr(t2d, "parallel_run", False) else int(t2d.get("MODEL.NPOIN"))
    if npoin != node_pos.shape[0] or npoin != node_area.shape[0]:
        raise ValueError(f"Mesh mismatch: TELEMAC={npoin}, model={node_pos.shape[0]}, area={node_area.shape[0]}")

    history = deque(maxlen=k_max)
    next_sample = 0.0
    next_prediction = (k_max - 1) * base_dt_seconds
    calls = []

    def get_state(t):
        return {
            "time": t,
            "u": t2d.mpi_get_array("MODEL.VELOCITYU"),
            "v": t2d.mpi_get_array("MODEL.VELOCITYV"),
            "h": t2d.mpi_get_array("MODEL.WATERDEPTH"),
            "z": t2d.mpi_get_array("MODEL.BOTTOMELEVATION"),
        }

    history.append(get_state(0.0))
    next_sample = base_dt_seconds

    try:
        nt = int(t2d.get("MODEL.NTIMESTEPS"))
        for _ in range(nt):
            t2d.run_one_time_step()
            current = float(t2d.get("MODEL.AT"))
            if current + 1e-9 < next_sample:
                continue

            state = get_state(current)
            history.append(state)
            next_sample += base_dt_seconds

            if len(history) < k_max or current + 1e-9 < next_prediction:
                continue

            if MPI.COMM_WORLD.Get_rank() == 0:
                bundle, out = predict(list(history))
                eff_b = min(int(out["effective_b"]), cap)
                info = {
                    "bundle": bundle,
                    "effective_b": eff_b,
                    "effective_k_count": int(out["effective_k_count"]),
                }
            else:
                info = None
            info = MPI.COMM_WORLD.bcast(info, root=0)

            dz = cumulative_writeback(info["bundle"], info["effective_b"])
            z = state["z"]
            h = state["h"]
            surface = z + h
            dz, z_new, h_new = postprocess_dz(
                dz, h, z, surface, args.min_water_depth, args.max_dz
            )
            t2d.mpi_set_array("MODEL.BOTTOMELEVATION", z_new)
            t2d.mpi_set_array("MODEL.WATERDEPTH", h_new)
            calls.append({
                "time": current,
                "effective_b": info["effective_b"],
                "effective_k_count": info["effective_k_count"],
                "max_abs_dz": float(np.abs(dz).max()),
            })
            next_prediction += info["effective_b"] * base_dt_seconds
    finally:
        t2d.finalize()

    if MPI.COMM_WORLD.Get_rank() == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("wb") as f:
            pickle.dump({"calls": calls, "config": config}, f)
        print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
