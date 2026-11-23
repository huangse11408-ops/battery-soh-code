# SARG-GRU

Readout design for current-cycle battery SOH estimation.

Each sample is one battery cycle's constant-current charging trace, resampled to
a fixed grid. The model maps that trace to a scalar state-of-health (SOH) value,
defined as the paired measured discharge capacity divided by the cell-specific
reference capacity. Splits are disjoint at the battery level: no cell
contributes cycles to more than one partition.

## Contents

| Path | Purpose |
| --- | --- |
| `configs/experiment.json` | Datasets, baselines, ablation variants, architecture and recipe |
| `sarg/data.py` | Canonical per-cell array contract, dataset registry, splits, loading |
| `sarg/models.py` | GRU backbone, anchor pooling, SARG-GRU readout and its four variants |
| `sarg/baselines.py` | Scalar adapters around the seven pinned upstream backbones |
| `sarg/engine.py` | Per-battery metrics, training loop, frozen evaluation |
| `sarg/report.py` | Result tables (CSV + Markdown) |
| `tools/prepare.py` | Raw cycles to canonical arrays (separation, resampling, labels) |
| `tools/fetch_upstream.py` | Clone the pinned upstream model repositories |
| `tests/` | CPU unit tests on synthetic tensors |

## Datasets

Seven evaluation groups, one row per source/chemistry:

| Group | Chemistry | Reference capacity |
| --- | --- | --- |
| `HUST` | LFP | per-cell measured |
| `MATR` | LFP | 1.1 Ah |
| `TJU_NCA` | NCA | 3.5 Ah |
| `Stanford_2` | NMC523 | 0.24 Ah |
| `CALB` | NMC | per-cell measured |
| `CALCE` | LCO | 1.1 / 1.35 Ah |
| `Tongji3` | NCM+NCA blend | 2.5 Ah |

Raw data are not redistributed. `tools/prepare.py` documents the expected input
layout and writes the canonical array per cell.

## Models

**Seven baselines**, each the pinned upstream backbone with a scalar readout
adapted for regression:

`DLinear`, `MixLinear`, `PatchTST`, `iTransformer`, `ModernTCN`, `TimeMixer`,
`TimeMixer++`.

All baselines receive the same input tensor, normalisation, labels, splits and
optimiser settings as SARG-GRU. Upstream code is fetched at the pinned commit
recorded in `vendor/SOURCES.md`; it is never vendored into this repository.

**SARG-GRU** keeps a frozen two-layer GRU backbone and varies only the readout:

| Variant | Description |
| --- | --- |
| `full` | Voltage anchors, ordered differences between adjacent anchor memories, gated fusion with the time branch |
| `no_ordered_differences_and_gating` | Same branches, ordered differences dropped, plain average fusion |
| `time_anchors` | Same readout as `full`; anchors placed on elapsed-time progress instead of terminal voltage |
| `no_voltage_branch` | Voltage memory removed; prediction from the time branch only |

## Environment

Linux, Python 3.12, PyTorch 2.5.1 with CUDA 11.8.

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu118 torch==2.5.1
python -m pip install -r requirements.txt
```

## Run

```bash
# 1. pinned upstream backbones
python tools/fetch_upstream.py

# 2. canonical arrays from raw cycles (see the docstring for the input layout)
python tools/prepare.py --raw /path/to/raw --out data

# 3. seven baselines x seven datasets x three seeds
python -m sarg.train --config configs/experiment.json --models baseline --out runs

# 4. four SARG-GRU variants
python -m sarg.train --config configs/experiment.json --models sarg --out runs

# 5. frozen evaluation (test split is read only here)
python -m sarg.evaluate --config configs/experiment.json --runs runs --out results
python -m sarg.report --results results
```

Every run selects its checkpoint on the validation split only. The test split is
touched once, after all weights are frozen, and is never used to choose a model,
a variant or a seed.

## Metrics

Per battery first, then averaged across batteries so that cells with long traces
do not dominate:

- `MAE_pp`, `RMSE_pp` in SOH percentage points
- reported per dataset and as a seven-group macro average

## Tests

```bash
PYTHONPATH=. python -m unittest discover -s tests
```

The tests use synthetic tensors and need no data or GPU.

## License

MIT, see `LICENSE`.
