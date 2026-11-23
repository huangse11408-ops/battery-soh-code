# -*- coding: utf-8 -*-
'''Raw cycle records to canonical per-cell arrays.

Expected raw layout (one directory per group, one file per cell):

    <raw>/<group>/<cell>.npz
        time_s      float64 [N]      monotonic seconds within the record
        current_A   float64 [N]
        voltage_V   float64 [N]
        cycle_id    int64   [N]      physical cycle index

The canonical output for one cell is

    <out>/<group>/<cell>.npz
        x           float32 [C,300,3]  elapsed_time_seconds, voltage_V, current_A
        y           float32 [C]        discharge capacity / reference capacity
        capacity_Ah float32 [C]
        cycle_id    int64   [C]

Preparation rules, applied identically to every group:

1. A cycle keeps only its constant-current charging segment. The segment starts
   where current is positive and stable and ends at the charge cut-off; the
   constant-voltage tail and any rest period are excluded.
2. No interpolation crosses a segment boundary. Within a segment the elapsed
   time, voltage and current are linearly resampled to a fixed 300-point grid
   that allocates points by elapsed time.
3. The label is the measured discharge capacity of the same cycle divided by the
   cell reference capacity from sarg.data.DATASETS.
4. Cycles with fewer than 30 raw points, a non-positive capacity, or a capacity
   outside +/-80 % of the reference are dropped and counted in the report.
5. Nothing from the future of a cycle is used to build that cycle's input.
'''

import json
from pathlib import Path

import numpy as np

LENGTH = 300
MIN_POINTS = 30


def constant_current_segment(time, current, voltage):
    r'''Longest positive-current run that ends at the charging cut-off.'''
    positive = current > 0.01
    if not positive.any():
        return None
    index = np.flatnonzero(positive)
    runs, start = [], index[0]
    for previous, current_index in zip(index[:-1], index[1:]):
        if current_index != previous + 1:
            runs.append((start, previous))
            start = current_index
    runs.append((start, index[-1]))
    begin, end = max(runs, key=lambda run: run[1] - run[0])
    if end - begin + 1 < MIN_POINTS:
        return None
    return begin, end


def resample(time, voltage, current, length=LENGTH):
    r'''Resample on elapsed time, never crossing a segment boundary.'''
    span = time[-1] - time[0]
    if not np.isfinite(span) or span <= 0:
        return None
    grid = time[0] + np.linspace(0.0, span, length)
    return np.stack([grid - time[0],
                     np.interp(grid, time, voltage),
                     np.interp(grid, time, current)], axis=1).astype("float32")


def prepare_cell(path, reference_ah):
    with np.load(path, allow_pickle=False) as z:
        time, current = z["time_s"].astype("float64"), z["current_A"].astype("float64")
        voltage, cycle = z["voltage_V"].astype("float64"), z["cycle_id"].astype("int64")
    inputs, labels, capacities, cycles, dropped = [], [], [], [], 0
    for index in sorted(set(cycle.tolist())):
        mask = cycle == index
        segment = constant_current_segment(time[mask], current[mask], voltage[mask])
        if segment is None:
            dropped += 1
            continue
        begin, end = segment
        sample = resample(time[mask][begin:end + 1], voltage[mask][begin:end + 1],
                          current[mask][begin:end + 1])
        if sample is None:
            dropped += 1
            continue
        discharge = float(np.trapezoid(current[mask][current[mask] < 0],
                                       time[mask][current[mask] < 0]) / 3600.0)
        capacity = abs(discharge)
        if capacity <= 0 or not (0.2 * reference_ah < capacity < 1.8 * reference_ah):
            dropped += 1
            continue
        inputs.append(sample)
        capacities.append(capacity)
        labels.append(capacity / reference_ah)
        cycles.append(int(index))
    if not inputs:
        return None
    return {"x": np.stack(inputs), "y": np.asarray(labels, dtype="float32"),
            "capacity_Ah": np.asarray(capacities, dtype="float32"),
            "cycle_id": np.asarray(cycles, dtype="int64")}, dropped


def main(raw, out, datasets):
    raw, out = Path(raw), Path(out)
    report = []
    for group, reference in datasets.items():
        source, target = raw / group, out / group
        if not source.is_dir():
            print("skip", group, "(absent)")
            continue
        target.mkdir(parents=True, exist_ok=True)
        for path in sorted(source.glob("*.npz")):
            reference_ah = reference if reference is not None else _measured_reference(path)
            prepared = prepare_cell(path, reference_ah)
            if prepared is None:
                report.append({"group": group, "cell": path.stem, "status": "no_valid_cycle"})
                continue
            payload, dropped = prepared
            np.savez_compressed(target / path.name, **payload)
            report.append({"group": group, "cell": path.stem, "status": "ok",
                           "cycles": int(len(payload["y"])), "dropped": dropped,
                           "reference_Ah": float(reference_ah)})
        print("prepared", group)
    (out / "prepare_report.json").write_text(json.dumps(report, indent=1))
    print("wrote", out / "prepare_report.json")


def _measured_reference(path):
    with np.load(path, allow_pickle=False) as z:
        current, time = z["current_A"], z["time_s"]
    discharge = float(np.trapezoid(current[current < 0], time[current < 0]) / 3600.0)
    return abs(discharge)


if __name__ == "__main__":
    import argparse

    from sarg.data import DATASETS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, help="directory holding the raw per-group cycles")
    parser.add_argument("--out", default="data", help="directory for canonical arrays")
    parser.add_argument("--groups", nargs="*", default=None, help="subset of the seven groups")
    arguments = parser.parse_args()
    selected = ({g: DATASETS[g]["reference_ah"] for g in arguments.groups}
                if arguments.groups else
                {g: DATASETS[g]["reference_ah"] for g in DATASETS})
    main(arguments.raw, arguments.out, selected)
