# -*- coding: utf-8 -*-
'''Per-battery metrics, training loop and frozen evaluation.'''

import json
import time
from pathlib import Path

import numpy as np
import torch

from . import data as data_mod
from .metrics import battery_metrics, macro, score


def tensors(records, stats):
    x = np.stack([data_mod.normalise(r["x"], stats) for r in records])
    y = np.concatenate([r["y"] for r in records])
    cell = np.concatenate([[r["cell"]] * len(r["y"]) for r in records])
    return (torch.as_tensor(x, dtype=torch.float32, device="cuda"),
            torch.as_tensor(y, dtype=torch.float32, device="cuda"),
            cell)


@torch.no_grad()
def predict(model, x, forward_fn, batch=512):
    model.eval()
    parts = []
    for start in range(0, len(x), batch):
        parts.append(forward_fn(model, x[start:start + batch]).reshape(-1))
    return torch.cat(parts).cpu().numpy()


def score(prediction, target, cell):
    rows = []
    for name in sorted(set(cell.tolist())):
        mask = cell == name
        rows.append({"cell": name, **battery_metrics(prediction[mask], target[mask])})
    return {"MAE_pp": macro(rows, "MAE_pp"), "RMSE_pp": macro(rows, "RMSE_pp"),
            "cells": len(rows), "per_cell": rows}


def train_one(model, forward_fn, train, val, recipe, directory, seed):
    r'''Train with early stopping on battery-macro validation MAE.'''
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    optimiser = torch.optim.AdamW(model.parameters(), lr=recipe["lr"],
                                  weight_decay=recipe["weight_decay"])
    best, best_epoch, stale = float("inf"), 0, 0
    history = []
    tx, ty, _ = train
    vx, vy, vc = val
    for epoch in range(1, recipe["max_epochs"] + 1):
        begin = time.monotonic()
        model.train()
        order = torch.randperm(len(tx), device=tx.device)
        total = 0.0
        for start in range(0, len(order), recipe["batch_size"]):
            take = order[start:start + recipe["batch_size"]]
            optimiser.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(forward_fn(model, tx[take]), ty[take])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), recipe["gradient_clip"])
            optimiser.step()
            total += float(loss.detach()) * len(take)
        prediction = predict(model, vx, forward_fn)
        validation = score(prediction, vy, vc)
        improved = validation["MAE_pp"] < best
        if improved:
            best, best_epoch, stale = validation["MAE_pp"], epoch, 0
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_MAE_pp": best}, directory / "best.pt")
        else:
            stale += 1
        history.append({"epoch": epoch, "train_MSE": total / len(tx),
                        "val_MAE_pp": validation["MAE_pp"],
                        "seconds": time.monotonic() - begin})
        (directory / "history.json").write_text(json.dumps(history, indent=1))
        print(f"epoch {epoch:03d} train_MSE={total / len(tx):.6f} "
              f"val_MAE_pp={validation['MAE_pp']:.4f} best={best:.4f} stale={stale}", flush=True)
        if stale >= recipe["patience"]:
            break
    return {"best_epoch": best_epoch, "best_val_MAE_pp": best, "epochs": len(history)}


def evaluate(model, forward_fn, split, directory):
    checkpoint = torch.load(Path(directory) / "best.pt", map_location="cuda",
                            weights_only=False)
    model.load_state_dict(checkpoint["model"])
    x, y, cell = split
    prediction = predict(model, x, forward_fn)
    return score(prediction, y, cell)
