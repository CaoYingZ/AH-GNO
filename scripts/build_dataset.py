"""Build an AH-GNO dataset from paired TELEMAC-2D and GAIA results.

Cases are listed in a CSV manifest. The script checks mesh/time consistency,
forms history and target windows, computes nodal control areas, and derives
normalization statistics from the training split only.
"""Build an AH-GNO training dataset from paired TELEMAC-2D / GAIA SELAFIN files.

Manifest columns:
    case_id,role,hydro_file,gaia_file

The builder follows the manuscript representation:
- input history: U, V, H, Z on the native unstructured mesh;
- target: future bed-elevation increments dZ;
- nodal control-area weights from the triangular mesh;
- normalization statistics computed from training scenarios only;
- scenario-level train/validation/test separation.

TELEMAC-MASCARET must be installed separately and its environment must be
sourced so that TelemacFile can be imported.
"""
from __future__ import annotations

import argparse
import csv
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from AH_GNO.preprocessing import compute_node_areas, normalize_coordinates


FEATURES = ("u", "v", "h", "z")
VAR_ALIASES = {
    "u": ("VELOCITY U", "U CLIPPED"),
    "v": ("VELOCITY V", "V CLIPPED"),
    "h": ("WATER DEPTH", "H CLIPPED"),
    "z": ("BOTTOM", "BOTTOM ELEVATION"),
}


@dataclass
class CaseSpec:
    case_id: str
    role: str
    hydro_file: Path
    gaia_file: Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("data/gno_dataset.pkl"))
    p.add_argument("--history-k", type=int, required=True,
                   help="Maximum number of sampled historical states.")
    p.add_argument("--bundle-b", type=int, required=True,
                   help="Maximum number of future bed-increment targets.")
    p.add_argument("--base-step", type=int, default=1,
                   help="Take every Nth SELAFIN output record as one AH-GNO base interval.")
    p.add_argument("--window-stride", type=int, default=1,
                   help="Stride between consecutive rolling-window anchors.")
    p.add_argument("--activity-gain", type=float, default=4.0,
                   help="Gain in w=1+gain*clip(activity/scale,0,1).")
    p.add_argument("--activity-scale", type=float, default=None,
                   help="Physical dZ scale in metres. Default: training dZ standard deviation.")
    p.add_argument("--z-source", choices=("gaia", "hydro"), default="gaia")
    p.add_argument("--time-tol", type=float, default=1.0e-6)
    return p.parse_args()


def import_telemac_file():
    try:
        from data_manip.extraction.telemac_file import TelemacFile
    except ImportError as exc:
        raise RuntimeError(
            "Could not import TELEMAC TelemacFile. Source the TELEMAC-MASCARET "
            "environment before running build_dataset.py."
        ) from exc
    return TelemacFile


def canonical_role(value: str) -> str:
    x = value.strip().lower()
    if x == "train":
        return "train"
    if x in {"val", "valid", "validation"}:
        return "val"
    if x == "test":
        return "test"
    raise ValueError(f"Unknown role: {value!r}")


def read_manifest(path: Path) -> List[CaseSpec]:
    specs = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"case_id", "role", "hydro_file", "gaia_file"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest missing columns: {sorted(missing)}")
        for row in reader:
            specs.append(
                CaseSpec(
                    case_id=row["case_id"].strip(),
                    role=canonical_role(row["role"]),
                    hydro_file=Path(row["hydro_file"]).expanduser(),
                    gaia_file=Path(row["gaia_file"]).expanduser(),
                )
            )
    if not specs:
        raise ValueError("Manifest is empty.")
    return specs


def resolve_variable(telemac_file, key: str) -> str:
    available = {str(v).strip().upper(): str(v).strip() for v in telemac_file.varnames}
    for alias in VAR_ALIASES[key]:
        if alias.upper() in available:
            return available[alias.upper()]
    raise KeyError(
        f"Could not find {key!r}. Tried {VAR_ALIASES[key]}; "
        f"available variables are {list(available.values())}"
    )


def load_case(spec: CaseSpec, z_source: str, time_tol: float):
    TelemacFile = import_telemac_file()
    hy = TelemacFile(str(spec.hydro_file))
    ga = TelemacFile(str(spec.gaia_file))
    try:
        hy_times = np.asarray(hy.times, dtype=np.float64)
        ga_times = np.asarray(ga.times, dtype=np.float64)
        if hy_times.shape != ga_times.shape or not np.allclose(
            hy_times, ga_times, rtol=0.0, atol=time_tol
        ):
            raise ValueError(
                f"{spec.case_id}: HYDRO and GAIA output times are not aligned."
            )

        x = np.asarray(hy.meshx, dtype=np.float64)
        y = np.asarray(hy.meshy, dtype=np.float64)
        elements = np.asarray(hy.ikle2, dtype=np.int64)
        if elements.min() == 1:
            elements = elements - 1

        gx = np.asarray(ga.meshx, dtype=np.float64)
        gy = np.asarray(ga.meshy, dtype=np.float64)
        gelems = np.asarray(ga.ikle2, dtype=np.int64)
        if gelems.min() == 1:
            gelems = gelems - 1

        if (
            x.shape != gx.shape
            or y.shape != gy.shape
            or elements.shape != gelems.shape
            or not np.allclose(x, gx)
            or not np.allclose(y, gy)
            or not np.array_equal(elements, gelems)
        ):
            raise ValueError(f"{spec.case_id}: HYDRO and GAIA meshes do not match.")

        u_name = resolve_variable(hy, "u")
        v_name = resolve_variable(hy, "v")
        h_name = resolve_variable(hy, "h")
        z_file = ga if z_source == "gaia" else hy
        z_name = resolve_variable(z_file, "z")

        def stack(tf, var_name):
            return np.stack(
                [
                    np.asarray(tf.get_data_value(var_name, i), dtype=np.float32)
                    for i in range(len(hy_times))
                ],
                axis=0,
            )

        fields = {
            "u": stack(hy, u_name),
            "v": stack(hy, v_name),
            "h": stack(hy, h_name),
            "z": stack(z_file, z_name),
        }
        return {
            "times": hy_times,
            "x": x,
            "y": y,
            "elements": elements,
            "fields": fields,
        }
    finally:
        hy.close()
        ga.close()


def assert_same_reference_mesh(reference, current, case_id: str):
    if (
        reference["x"].shape != current["x"].shape
        or reference["elements"].shape != current["elements"].shape
        or not np.allclose(reference["x"], current["x"])
        or not np.allclose(reference["y"], current["y"])
        or not np.array_equal(reference["elements"], current["elements"])
    ):
        raise ValueError(
            f"{case_id}: mesh differs from the first case. "
            "Build separate datasets for different meshes."
        )


def build_raw_samples(case, spec: CaseSpec, history_k: int, bundle_b: int,
                      base_step: int, window_stride: int):
    record_ids = np.arange(0, len(case["times"]), base_step, dtype=np.int64)
    if len(record_ids) < history_k + bundle_b:
        raise ValueError(
            f"{spec.case_id}: only {len(record_ids)} sampled records, but "
            f"K+B={history_k + bundle_b}."
        )

    out = []
    last_anchor = len(record_ids) - bundle_b - 1
    for anchor in range(history_k - 1, last_anchor + 1, window_stride):
        hist_ids = record_ids[anchor - history_k + 1 : anchor + 1]
        future_ids = record_ids[anchor + 1 : anchor + bundle_b + 1]
        prev_ids = record_ids[anchor : anchor + bundle_b]

        hist = np.stack(
            [
                np.column_stack(
                    [
                        case["fields"]["u"][rid],
                        case["fields"]["v"][rid],
                        case["fields"]["h"][rid],
                        case["fields"]["z"][rid],
                    ]
                )
                for rid in hist_ids
            ],
            axis=1,
        )

        dz = (
            case["fields"]["z"][future_ids]
            - case["fields"]["z"][prev_ids]
        ).T

        out.append(
            {
                "case_id": spec.case_id,
                "role": spec.role,
                "time": float(case["times"][record_ids[anchor]]),
                "f_raw": hist.reshape(hist.shape[0], history_k * 4).astype(np.float32),
                "y_raw": dz.astype(np.float32),
            }
        )
    return out


class WeightedMoments:
    def __init__(self):
        self.sum_w = 0.0
        self.sum_x = 0.0
        self.sum_x2 = 0.0

    def update(self, values: np.ndarray, weights: np.ndarray):
        x = np.asarray(values, dtype=np.float64).reshape(-1)
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if x.shape != w.shape:
            raise ValueError(f"WeightedMoments shape mismatch: {x.shape} vs {w.shape}")
        self.sum_w += float(w.sum())
        self.sum_x += float(np.dot(w, x))
        self.sum_x2 += float(np.dot(w, x * x))

    def finalize(self):
        if self.sum_w <= 0:
            raise ValueError("No weight accumulated.")
        mean = self.sum_x / self.sum_w
        var = max(self.sum_x2 / self.sum_w - mean * mean, 0.0)
        return {"mean": float(mean), "std": float(np.sqrt(var))}


def fit_training_stats(train_samples, node_area: np.ndarray,
                       history_k: int, bundle_b: int):
    moments = {k: WeightedMoments() for k in (*FEATURES, "dz")}
    area = np.asarray(node_area, dtype=np.float64)

    for sample in train_samples:
        f = sample["f_raw"].reshape(len(area), history_k, 4)
        y = sample["y_raw"]
        for feature_index, key in enumerate(FEATURES):
            for t in range(history_k):
                moments[key].update(f[:, t, feature_index], area)
        for b in range(bundle_b):
            moments["dz"].update(y[:, b], area)

    return {key: value.finalize() for key, value in moments.items()}


def normalize_samples(samples, stats, node_area, history_k,
                      activity_scale, activity_gain):
    out = []
    n_nodes = len(node_area)
    eps = 1.0e-5

    for sample in samples:
        f = sample["f_raw"].reshape(n_nodes, history_k, 4).copy()
        for feature_index, key in enumerate(FEATURES):
            f[:, :, feature_index] = (
                f[:, :, feature_index] - stats[key]["mean"]
            ) / (stats[key]["std"] + eps)
        f = f.reshape(n_nodes, history_k * 4)

        y_phys = sample["y_raw"]
        y = (y_phys - stats["dz"]["mean"]) / (stats["dz"]["std"] + eps)

        activity = np.max(np.abs(y_phys), axis=1, keepdims=True)
        w = 1.0 + max(activity_gain, 0.0) * np.clip(
            activity / max(activity_scale, 1.0e-12), 0.0, 1.0
        )

        out.append(
            {
                "f": torch.from_numpy(f.astype(np.float32)),
                "y": torch.from_numpy(y.astype(np.float32)),
                "w": torch.from_numpy(w.astype(np.float32)),
                "case_id": sample["case_id"],
                "time": sample["time"],
            }
        )
    return out


def main():
    args = parse_args()
    if args.history_k < 1 or args.bundle_b < 1:
        raise ValueError("history-k and bundle-b must be positive.")
    if args.base_step < 1 or args.window_stride < 1:
        raise ValueError("base-step and window-stride must be positive.")

    specs = read_manifest(args.manifest)
    all_raw: Dict[str, list] = {"train": [], "val": [], "test": []}
    reference_mesh = None
    raw_output_dt = None

    for spec in specs:
        print(f"[load] {spec.case_id}: {spec.role}")
        case = load_case(spec, z_source=args.z_source, time_tol=args.time_tol)

        if reference_mesh is None:
            reference_mesh = case
            if len(case["times"]) > 1:
                raw_output_dt = float(np.median(np.diff(case["times"])))
        else:
            assert_same_reference_mesh(reference_mesh, case, spec.case_id)
            if len(case["times"]) > 1 and raw_output_dt is not None:
                dt = float(np.median(np.diff(case["times"])))
                if not np.isclose(dt, raw_output_dt, rtol=0.0, atol=args.time_tol):
                    raise ValueError(
                        f"{spec.case_id}: output interval {dt}s differs from "
                        f"reference {raw_output_dt}s."
                    )

        samples = build_raw_samples(
            case, spec,
            history_k=args.history_k,
            bundle_b=args.bundle_b,
            base_step=args.base_step,
            window_stride=args.window_stride,
        )
        all_raw[spec.role].extend(samples)
        print(f"       samples={len(samples)}")

    if not all_raw["train"]:
        raise ValueError("No training samples were created.")

    node_area = compute_node_areas(
        reference_mesh["x"], reference_mesh["y"], reference_mesh["elements"]
    )
    node_pos, coord_range = normalize_coordinates(
        reference_mesh["x"], reference_mesh["y"]
    )

    stats = fit_training_stats(
        all_raw["train"], node_area, args.history_k, args.bundle_b
    )
    activity_scale = (
        float(args.activity_scale)
        if args.activity_scale is not None
        else max(float(stats["dz"]["std"]), 1.0e-12)
    )

    train_samples = normalize_samples(
        all_raw["train"], stats, node_area, args.history_k,
        activity_scale, args.activity_gain
    )
    val_samples = normalize_samples(
        all_raw["val"], stats, node_area, args.history_k,
        activity_scale, args.activity_gain
    )
    test_samples = normalize_samples(
        all_raw["test"], stats, node_area, args.history_k,
        activity_scale, args.activity_gain
    )

    base_dt_minutes = None
    if raw_output_dt is not None:
        base_dt_minutes = raw_output_dt * args.base_step / 60.0

    split = {
        "train": [s.case_id for s in specs if s.role == "train"],
        "val": [s.case_id for s in specs if s.role == "val"],
        "test": [s.case_id for s in specs if s.role == "test"],
    }

    dataset = {
        "node_pos": node_pos,
        "node_area": torch.from_numpy(node_area.astype(np.float32)),
        "elements": torch.from_numpy(reference_mesh["elements"].astype(np.int64)),
        "coord_range": coord_range,
        "train_samples": train_samples,
        "val_samples": val_samples,
        "test_samples": test_samples,
        "normalizer_state": {"stats": stats, "eps": 1.0e-5},
        "history_k": int(args.history_k),
        "bundle_b": int(args.bundle_b),
        "base_step": int(args.base_step),
        "base_dt_minutes": base_dt_minutes,
        "window_stride": int(args.window_stride),
        "feature_names": list(FEATURES),
        "activity_gain": float(args.activity_gain),
        "active_scale": float(activity_scale),
        "scenario_split": split,
        "task_definition": "AH-GNO: [u,v,h,z] history -> bundled bed-elevation increments",
        "z_source": args.z_source,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("\nAH-GNO dataset written:")
    print(f"  {args.output}")
    print(f"  nodes: {len(node_area)}")
    print(
        "  train / val / test samples: "
        f"{len(train_samples)} / {len(val_samples)} / {len(test_samples)}"
    )
    print(f"  K={args.history_k}, B={args.bundle_b}, base_step={args.base_step}")
    if base_dt_minutes is not None:
        print(f"  base interval: {base_dt_minutes:.6g} min")
    print(f"  activity scale: {activity_scale:.6e} m")


if __name__ == "__main__":
    main()
