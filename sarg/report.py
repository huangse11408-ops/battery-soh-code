# -*- coding: utf-8 -*-
'''Result tables: per dataset, and a seven-group macro average.'''

import csv
import json
from pathlib import Path


def collect(results_root):
    rows = []
    for path in sorted(Path(results_root).rglob("metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(payload)
    return rows


def table(rows, key="MAE_pp"):
    r'''model x dataset matrix of the requested metric.'''
    models = sorted({r["model"] for r in rows})
    datasets = sorted({r["dataset"] for r in rows})
    return models, datasets, {(r["model"], r["dataset"]): r[key] for r in rows}


def main(results_root="results", out="results"):
    rows = collect(results_root)
    if not rows:
        print("no metrics.json found")
        return
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for key, filename in (("MAE_pp", "table_MAE_pp.csv"),
                          ("RMSE_pp", "table_RMSE_pp.csv")):
        models, datasets, values = table(rows, key)
        with (out / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["model"] + datasets + ["macro"])
            for model in models:
                cells = [values.get((model, d)) for d in datasets]
                present = [v for v in cells if v is not None]
                writer.writerow([model] + ["" if v is None else f"{v:.4f}" for v in cells]
                                + [f"{sum(present) / len(present):.4f}" if present else ""])
        print("wrote", out / filename)


if __name__ == "__main__":
    main()
