#!/usr/bin/env python3
"""
Stage 4 Unified Sweep Generator — Robustness Suite
---------------------------------------------------
Full-factorial robustness sweep covering:
  - Quality levels: ["high"], ["high","medium"], ["high","medium","low"]
  - Balance: True / False
  - Upsampling mode: repeat / linear
  - Upsampling ratio: 0.0 / 0.3 / 0.5
  - Seeds: 123 / 456 / 789

Generates 3 × 2 × 2 × 3 × 3 = 108 YAML configs
and a manifest CSV for tracking.
"""

import itertools
import yaml
import csv
from pathlib import Path

# ---- Fixed anchors ----
HIDDEN_UNITS = 512
DROPOUT = 0.25
BATCH_SIZE = 32
LEARNING_RATE = 0.0005
FOCAL_LOSS = False
LABEL_SMOOTHING = True
MIXUP = True
SEEDS = [123, 456, 789]

# ---- Sweep axes ----
QUALITIES = [
    ["high"],
    ["high", "medium"],
    ["high", "medium", "low"],
]
BALANCES = [True, False]
UPSAMPLING_MODES = ["repeat", "linear"]
UPSAMPLING_RATIOS = [0.0, 0.3, 0.5]

# ---- Output paths ----
ROOT = Path("config/sweeps/stage4_unified")
ROOT.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = ROOT / "manifest.csv"

# ============================================================
# Helper for writing individual config YAMLs
# ============================================================
def write_cfg(idx, params):
    cfg = {
        "experiment": {
            "name": f"stage4_{idx:03d}",
            "seed": params["seed"],
        },
        "training": {"batch_size": BATCH_SIZE},
        "inference": {"batch_size": BATCH_SIZE},
        "training_package": {
            "include_negatives": True,
            "balance": params["balance"],
            "quality": params["quality"],
        },
        "training_args": {
            "hidden_units": HIDDEN_UNITS,
            "dropout": DROPOUT,
            "learning_rate": LEARNING_RATE,
            "label_smoothing": LABEL_SMOOTHING,
            "mixup": MIXUP,
            "focal-loss": FOCAL_LOSS,
            "focal-loss-gamma": 2.0 if FOCAL_LOSS else 0.0,
            "focal-loss-alpha": 0.25 if FOCAL_LOSS else 0.0,
            "upsampling_mode": params["upsampling_mode"],
            "upsampling_ratio": params["upsampling_ratio"],
        },
    }

    outpath = ROOT / f"{cfg['experiment']['name']}.yaml"
    with open(outpath, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    return {
        "name": cfg["experiment"]["name"],
        "seed": params["seed"],
        "quality": ",".join(params["quality"]),
        "balance": params["balance"],
        "mode": params["upsampling_mode"],
        "ratio": params["upsampling_ratio"],
    }

# ============================================================
# Generate full factorial sweep
# ============================================================
manifest = []
idx = 0
for quality, balance, mode, ratio, seed in itertools.product(
    QUALITIES, BALANCES, UPSAMPLING_MODES, UPSAMPLING_RATIOS, SEEDS
):
    idx += 1
    entry = write_cfg(idx, {
        "quality": quality,
        "balance": balance,
        "upsampling_mode": mode,
        "upsampling_ratio": ratio,
        "seed": seed,
    })
    manifest.append(entry)

# ============================================================
# Write manifest CSV
# ============================================================
with open(MANIFEST_PATH, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["name", "seed", "quality", "balance", "mode", "ratio"],
    )
    writer.writeheader()
    writer.writerows(manifest)

print(f"✅ Generated {idx} Stage 4 unified configs in {ROOT.resolve()}")
print(f"🧾 Manifest written to: {MANIFEST_PATH.resolve()}")
