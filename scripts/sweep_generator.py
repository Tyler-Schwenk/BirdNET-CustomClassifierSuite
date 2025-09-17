#!/usr/bin/env python3
"""
Generate Stage 1 sweep configs (capacity & regularization).
Creates 72 YAML override configs under config/sweeps/stage1/.
Each file is meant to be combined with config/base.yaml via run_sweep.py.
"""

import itertools
import yaml
from pathlib import Path

# Hyperparam grid
hidden_units = [0, 128, 512, 1024]
dropout = [0.0, 0.25, 0.5]
learning_rates = [0.0001, 0.0005, 0.001]
batch_sizes = [16, 32]

# Output directory
outdir = Path("config/sweeps/stage1")
outdir.mkdir(parents=True, exist_ok=True)

count = 0
for hu, dr, lr, bs in itertools.product(hidden_units, dropout, learning_rates, batch_sizes):
    count += 1
    cfg = {
        "experiment": {
            "name": f"stage1_sweep_{count:03d}",
            "seed": 123,
        },
        "training": {
            "batch_size": bs,
        },
        "inference": {
            "batch_size": bs,
        },
        "training_args": {
            "hidden_units": hu,
            "dropout": dr,
            "learning_rate": lr,
        },
    }

    outpath = outdir / f"{cfg['experiment']['name']}.yaml"
    with open(outpath, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

print(f"Generated {count} override configs in {outdir}")
