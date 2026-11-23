# -*- coding: utf-8 -*-
"""Canonical battery-cycle arrays, dataset registry and battery-level splits."""

import json
from pathlib import Path

import numpy as np

CHANNELS = ("elapsed_time_seconds", "voltage_V", "current_A")

# One entry per evaluation group. `reference_ah` is the denominator that turns a
# measured discharge capacity into a capacity ratio; None means the measured
# per-cell value is used instead.
DATASETS = {
    "HUST": {"chemistry": "LFP", "reference_ah": None},
    "MATR": {"chemistry": "LFP", "reference_ah": 1.1},
    "TJU_NCA": {"chemistry": "NCA", "reference_ah": 3.5},
    "Stanford_2": {"chemistry": "NMC523", "reference_ah": 0.24},
    "CALB": {"chemistry": "NMC", "reference_ah": None},
    "CALCE": {"chemistry": "LCO", "reference_ah": None},
    "Tongji3": {"chemistry": "NCM+NCA", "reference_ah": 2.5},
}


def config(path="configs/experiment.json"):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def split_cells(cells, fractions, seed):
    r"""Battery-level 70/15/15 split, seeded and reproducible.

    Every cell lands in exactly one partition, so no battery contributes cycles
    to more than one split.
    """
    cells = sorted(cells)
    if len(cells) < 3:
        raise ValueError("at least three cells are required for three partitions")
    rng = np.random.default_rng(seed)
    order = [cells[i] for i in rng.permutation(len(cells))]
    n = len(order)
    n_test = max(1, int(round(fractions["test"] * n)))
    n_val = max(1, int(round(fractions["val"] * n))) if n - n_test >= 2 else 0
    n_train = n - n_val - n_test
    if n_train < 1:
        raise ValueError("split leaves no training cell")
    return {"train": order[:n_train],
            "val": order[n_train:n_train + n_val],
            "test": order[n_train + n_val:]}


def load_cell(root, dataset, cell):
    r"""Read one canonical array: x [C,300,3], y [C], capacity_Ah [C], cycle_id [C]."""
    path = Path(root) / dataset / f"{cell}.npz"
    with np.load(path, allow_pickle=False) as z:
        x, y = z["x"].astype("float32"), z["y"].astype("float32")
        cap, cyc = z["capacity_Ah"].astype("float32"), z["cycle_id"].astype("int64")
    if x.ndim != 3 or x.shape[1:] != (300, 3):
        raise ValueError(f"{path}: expected x of shape [C,300,3], got {x.shape}")
    if not (len(x) == len(y) == len(cap) == len(cyc)):
        raise ValueError(f"{path}: inconsistent cycle counts")
    return {"cell": cell, "x": x, "y": y, "capacity_Ah": cap, "cycle_id": cyc}


def load_split(root, dataset, cells):
    return [load_cell(root, dataset, c) for c in cells]


def channel_statistics(records):
    r"""Per-channel mean and std fitted on TRAIN cycles only.

    Averaged over every cycle and every time step, so each entry belongs to one
    of the three input channels rather than to a time position.
    """
    x = np.concatenate([r["x"] for r in records], axis=0).astype("float64")
    flat = x.reshape(-1, x.shape[-1])
    return {"mean": flat.mean(axis=0).tolist(),
            "std": np.maximum(flat.std(axis=0), 1e-6).tolist(),
            "cycles": int(x.shape[0])}


def normalise(x, stats):
    mean = np.asarray(stats["mean"], dtype="float32").reshape(1, 1, -1)
    std = np.asarray(stats["std"], dtype="float32").reshape(1, 1, -1)
    return (np.asarray(x, dtype="float32") - mean) / std


def anchor_bounds(records, cfg):
    r"""Voltage interval covered by most TRAIN cycles.

    Bounds are taken per cycle rather than per cell, so one long trace cannot
    stretch the interval on its own.
    """
    low = np.concatenate([r["x"][:, :, 1].min(axis=1) for r in records])
    high = np.concatenate([r["x"][:, :, 1].max(axis=1) for r in records])
    q = cfg["common_interval_quantile"]
    start, end = float(np.quantile(low, 1 - q)), float(np.quantile(high, q))
    coverage = float(np.mean((low <= start) & (high >= end)))
    method = "train_common_absolute_voltage"
    if start >= end or coverage < cfg["common_min_cycle_coverage"]:
        stacked = np.concatenate([r["x"][:, :, 1].ravel() for r in records])
        start, end = (float(v) for v in np.quantile(stacked, cfg["fallback_quantiles"]))
        method = "train_voltage_quantile_range"
    if not start < end:
        raise ValueError("training voltage interval has zero width")
    return {"start": start, "end": end, "method": method,
            "train_full_interval_coverage": coverage, "cycles": int(len(low))}


def anchor_grid(bounds, count, width_multiplier):
    points = np.linspace(bounds["start"], bounds["end"], count, dtype=np.float32)
    sigma = (bounds["end"] - bounds["start"]) * width_multiplier / (count - 1)
    return {"start": bounds["start"], "end": bounds["end"], "method": bounds["method"],
            "anchors": points.tolist(), "sigma": float(sigma), "count": count,
            "width_multiplier": width_multiplier}
