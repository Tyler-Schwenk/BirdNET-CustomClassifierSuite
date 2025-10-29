#!/usr/bin/env python3
"""
General sweep generator for BirdNET-CustomClassifierSuite
Creates factorial or partial-factorial sweeps and writes:
    - base.yaml (per-sweep base, derived from spec.base_params)
    - <experiment>.yaml per axis combination (base + overrides)

python -m birdnet_custom_classifier_suite.sweeps.sweep_generator --spec config/sweep_specs/example_sweep.yaml

"""

import itertools, yaml
from pathlib import Path
from copy import deepcopy

def generate_sweep(stage:int, out_dir:str, axes:dict, base_params:dict, prefix:str="stage"):
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    # Note: No manifest.csv is written; configs are discoverable via folder globbing

    # --- 1) Build per-sweep base.yaml from base_params ---
    # Construct from a minimal skeleton only; no global base.yaml is read.
    base_cfg = {
        "dataset": {
            "audio_root": "AudioData",
            "manifest": "data/manifest.csv",
        },
        "experiment": {
            "name": f"{prefix}{stage}_base",
            "seed": 123,
        },
        "training_package": {
            "include_negatives": True,
            # balance/quality come from axes per-experiment; keep defaults here
            "balance": True,
            "quality": ["high"],
        },
        "training": {
            "epochs": 50,
            "batch_size": 32,
            "threads": 4,
            "val_split": 0.2,
            "autotune": False,
        },
        "inference": {
            "batch_size": 32,
            "threads": 4,
            "min_conf": 0.5,
        },
        "evaluation": {
            "thresholds": [0.5],
            "output_dir": "evaluation",
        },
        "training_args": {
            # Defaults; will be overwritten by base_params if provided
            "fmin": 0,
            "fmax": 15000,
            "overlap": 0.0,
            "dropout": 0.25,
            "hidden_units": 512,
            "learning_rate": 0.0005,
            "focal-loss": False,
            "focal-loss-gamma": 2.0,
            "focal-loss-alpha": 0.25,
            "label_smoothing": False,
            "mixup": False,
            "upsampling_mode": "repeat",
            "upsampling_ratio": 0.0,
        },
        "analyzer_args": {
            "fmin": 0,
            "fmax": 15000,
            "overlap": 0.0,
            "sensitivity": 1.0,
        },
    }

    # Map base_params into training/training_args and mirror audio to analyzer_args
    bp = base_params or {}
    # training keys
    if "epochs" in bp:
        base_cfg["training"]["epochs"] = bp["epochs"]
    if "batch_size" in bp:
        base_cfg["training"]["batch_size"] = bp["batch_size"]
        base_cfg["inference"]["batch_size"] = bp["batch_size"]
    # audio params to both training_args and analyzer_args
    for k in ("fmin", "fmax", "overlap"):
        if k in bp:
            base_cfg["training_args"][k] = bp[k]
            base_cfg["analyzer_args"][k] = bp[k]
    # remaining params → training_args
    for k, v in bp.items():
        if k in {"epochs", "batch_size", "fmin", "fmax", "overlap"}:
            continue
        base_cfg["training_args"][k] = v

    # Write base.yaml
    with open(root / "base.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(base_cfg, f, sort_keys=False)

    def make_cfg(idx, combo):
        exp_name = f"{prefix}{stage}_{idx:03d}"
        # Resolve batch size from combo override or base training
        bs = combo.get("batch_size", base_cfg.get("training", {}).get("batch_size", 32))

        # Start with resolved training_args from base.yaml
        resolved_ta = deepcopy(base_cfg.get("training_args", {}))

        # Overlay axis overrides that belong to training_args
        # Exclude non-TA keys handled separately
        non_ta = {"seed", "quality", "balance", "batch_size"}
        for k, v in combo.items():
            if k in non_ta:
                continue
            # Map directly into training_args (includes upsampling_mode/ratio, learning_rate, dropout, hidden_units, etc.)
            resolved_ta[k] = v

        cfg = {
            "experiment": {"name": exp_name, "seed": combo.get("seed", 123)},
            "training": {"batch_size": bs, "epochs": base_cfg.get("training", {}).get("epochs", 50)},
            "inference": {"batch_size": bs},
            "training_package": {
                "include_negatives": True,
                # Use axis values if provided, else fall back to base defaults
                "balance": combo.get("balance", base_cfg.get("training_package", {}).get("balance", True)),
                "quality": combo.get("quality", base_cfg.get("training_package", {}).get("quality", ["high"])),
            },
            "training_args": resolved_ta,
        }
        path = root / f"{exp_name}.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        return None

    axes_keys = list(axes.keys())
    combos = [dict(zip(axes_keys, values))
              for values in itertools.product(*axes.values())]

    count = 0
    for i, combo in enumerate(combos):
        make_cfg(i + 1, combo)
        count += 1
    print(f"Generated {count} configs at {root}")

if __name__ == "__main__":
    import argparse, yaml

    ap = argparse.ArgumentParser(description="Run sweep generation from YAML spec")
    ap.add_argument("--spec", type=str, required=True, help="Path to sweep spec YAML or JSON file")
    args = ap.parse_args()

    spec_path = Path(args.spec)
    with open(spec_path, "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)

    generate_sweep(
        stage=spec["stage"],
        out_dir=spec["out_dir"],
        axes=spec["axes"],
        base_params=spec["base_params"],
    )
