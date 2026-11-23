# -*- coding: utf-8 -*-
"""Metric layer. Deliberately free of torch so it can be tested on CPU alone."""

import numpy as np


def battery_metrics(prediction, target):
    r"""MAE and RMSE in SOH percentage points for one battery."""
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    if error.size == 0:
        raise ValueError("empty battery")
    if not np.isfinite(error).all():
        raise ValueError("non-finite prediction or target")
    return {"MAE_pp": float(np.abs(error).mean() * 100.0),
            "RMSE_pp": float(np.sqrt(np.mean(error ** 2)) * 100.0),
            "cycles": int(error.size)}


def macro(rows, key):
    r"""Unweighted mean over batteries, so long traces do not dominate."""
    if not rows:
        return float("nan")
    return float(np.mean([row[key] for row in rows]))


def score(prediction, target, cell):
    r"""Per-battery metrics averaged across the batteries of one split."""
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    cell = np.asarray(cell)
    if len(prediction) != len(target) or len(target) != len(cell):
        raise ValueError("prediction, target and cell must agree in length")
    rows = []
    for name in sorted(set(cell.tolist())):
        mask = cell == name
        rows.append({"cell": name, **battery_metrics(prediction[mask], target[mask])})
    return {"MAE_pp": macro(rows, "MAE_pp"), "RMSE_pp": macro(rows, "RMSE_pp"),
            "cells": len(rows), "per_cell": rows}
