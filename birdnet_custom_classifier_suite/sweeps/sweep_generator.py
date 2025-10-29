#!/usr/bin/env python3
"""
General sweep generator for BirdNET-CustomClassifierSuite
Creates factorial or partial-factorial sweeps and writes YAML + manifest CSV.

python -m birdnet_custom_classifier_suite.sweeps.sweep_generator --spec config/sweep_specs/example_sweep.yaml

"""

import itertools, yaml, csv
from pathlib import Path

def generate_sweep(stage:int, out_dir:str, axes:dict, base_params:dict, prefix:str="stage"):
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.csv"

    def make_cfg(idx, combo):
        exp_name = f"{prefix}{stage}_{idx:03d}"
        # Resolve batch size from combo override if present
        bs = combo.get("batch_size", base_params.get("batch_size", 32))

        # Build training_args by overlaying axis values (excluding non-args keys)
        arg_overrides_keys = set(combo.keys()) - {"seed", "quality", "balance", "upsampling_mode", "upsampling_ratio"}
        arg_overrides = {k: combo[k] for k in arg_overrides_keys}
        cfg = {
            "experiment": {"name": exp_name, "seed": combo["seed"]},
            "training": {"batch_size": bs},
            "inference": {"batch_size": bs},
            "training_package": {
                "include_negatives": True,
                "balance": combo["balance"],
                "quality": combo["quality"],
            },
            "training_args": {
                **base_params,
                **arg_overrides,
                "upsampling_mode": combo["upsampling_mode"],
                "upsampling_ratio": combo["upsampling_ratio"],
            },
        }
        path = root / f"{exp_name}.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        return {
            "name": exp_name,
            "seed": combo["seed"],
            "quality": ",".join(combo["quality"]),
            "balance": combo["balance"],
            "mode": combo["upsampling_mode"],
            "ratio": combo["upsampling_ratio"],
        }

    axes_keys = list(axes.keys())
    combos = [dict(zip(axes_keys, values))
              for values in itertools.product(*axes.values())]

    manifest = [make_cfg(i + 1, combo) for i, combo in enumerate(combos)]

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=manifest[0].keys())
        writer.writeheader()
        writer.writerows(manifest)

    print(f"Generated {len(manifest)} configs at {root}")
    print(f"Manifest written to {manifest_path}")

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
