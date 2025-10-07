#!/usr/bin/env python3
"""
Stage 3B + 3C sweep generators
---------------------------------
3B: upsampling ratio × mode × seed  (42 configs)
3C: ratio × LR × aug variant × seed  (36 configs total)
"""

import itertools
import yaml
from pathlib import Path

# ---- Fixed anchors from Stage 2 best ----
HIDDEN_UNITS = 512
DROPOUT = 0.25
BATCH_SIZE = 32
FOCAL_LOSS = False

# ---- Common seeds ----
SEEDS = [123, 456, 789]

# ============================================================
#  Stage 3B — Upsampling refinement (ratios 0.20–0.60)
# ============================================================
upsampling_ratios_3b = [0.20, 0.30, 0.40, 0.45, 0.50, 0.55, 0.60]
upsampling_modes_3b = ["repeat", "linear"]

LEARNING_RATE_3B = 0.0005
LABEL_SMOOTHING_3B = True
MIXUP_3B = True

outdir_3b = Path("config/sweeps/stage3b")
outdir_3b.mkdir(parents=True, exist_ok=True)

count = 0
for ratio, mode, seed in itertools.product(upsampling_ratios_3b, upsampling_modes_3b, SEEDS):
    count += 1
    cfg = {
        "experiment": {"name": f"stage3b_{count:03d}", "seed": seed},
        "training": {"batch_size": BATCH_SIZE},
        "inference": {"batch_size": BATCH_SIZE},
        "training_args": {
            "hidden_units": HIDDEN_UNITS,
            "dropout": DROPOUT,
            "learning_rate": LEARNING_RATE_3B,
            "label_smoothing": LABEL_SMOOTHING_3B,
            "mixup": MIXUP_3B,
            "focal-loss": FOCAL_LOSS,
            "focal-loss-gamma": 2.0 if FOCAL_LOSS else 0.0,
            "focal-loss-alpha": 0.25 if FOCAL_LOSS else 0.0,
            "upsampling_ratio": ratio,
            "upsampling_mode": mode,
        },
    }

    outpath = outdir_3b / f"{cfg['experiment']['name']}.yaml"
    with open(outpath, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

print(f"✅ Generated {count} Stage 3B configs in {outdir_3b.resolve()}")

# ============================================================
#  Stage 3C — LR × upsampling × augmentation variant
# ============================================================
upsampling_ratios_3c = [0.45, 0.50, 0.55]
upsampling_modes_3c = ["linear"]
learning_rates_3c = [0.0005, 0.001]
aug_profiles = [
    {"name": "augA", "label_smoothing": True, "mixup": True},   # Stage 3 base
    {"name": "augB", "label_smoothing": True, "mixup": False},  # Stage 2 top
]

outdir_3c = Path("config/sweeps/stage3c")
outdir_3c.mkdir(parents=True, exist_ok=True)

count = 0
for ratio, lr, aug, seed in itertools.product(
    upsampling_ratios_3c, learning_rates_3c, aug_profiles, SEEDS
):
    count += 1
    cfg = {
        "experiment": {
            "name": f"stage3c_{aug['name']}_{count:03d}",
            "seed": seed,
        },
        "training": {"batch_size": BATCH_SIZE},
        "inference": {"batch_size": BATCH_SIZE},
        "training_args": {
            "hidden_units": HIDDEN_UNITS,
            "dropout": DROPOUT,
            "learning_rate": lr,
            "label_smoothing": aug["label_smoothing"],
            "mixup": aug["mixup"],
            "focal-loss": FOCAL_LOSS,
            "focal-loss-gamma": 2.0 if FOCAL_LOSS else 0.0,
            "focal-loss-alpha": 0.25 if FOCAL_LOSS else 0.0,
            "upsampling_ratio": ratio,
            "upsampling_mode": "linear",
        },
    }

    outpath = outdir_3c / f"{cfg['experiment']['name']}.yaml"
    with open(outpath, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

print(f"✅ Generated {count} Stage 3C configs in {outdir_3c.resolve()}")
