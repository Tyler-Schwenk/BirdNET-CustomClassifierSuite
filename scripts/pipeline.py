# scripts/pipeline.py
"""
End-to-end pipeline for BirdNET Frog Training.
- Loads base + model config (YAML)
- Creates experiment directory
- Builds training package
- (Placeholder) Runs training step
"""

import argparse
from pathlib import Path
from utils.config import load_config
import scripts.make_training_package as make_training_package
import shutil

def cleanup_training_package(exp_dir: Path):
    tp_dir = exp_dir / "training_package"
    for sub in ["RADR", "Negative"]:
        folder = tp_dir / sub
        if folder.exists():
            shutil.rmtree(folder)



def get_experiment_dir(cfg: dict) -> Path:
    """Resolve experiment directory from config."""
    exp_cfg = cfg.get("experiment", {})
    name = exp_cfg.get("name")
    if not name:
        raise ValueError("Config missing required field: experiment.name")
    return Path("experiments") / name


def main():
    ap = argparse.ArgumentParser(description="BirdNET Frog Training pipeline")
    ap.add_argument(
        "--base-config",
        type=Path,
        default=Path("config/base.yaml"),
        help="Base config file (YAML)"
    )
    ap.add_argument(
        "--override-config",
        type=Path,
        required=False,
        help="Optional override config file (YAML)"
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging"
    )
    args = ap.parse_args()

    # Load merged config
    cfg = load_config(args.base_config, args.override_config) if args.override_config else load_config(args.base_config)
    print("Loaded config:")
    print(cfg)

    # Resolve experiment dir
    exp_dir = get_experiment_dir(cfg)
    if exp_dir.exists():
        raise FileExistsError(
            f"Experiment directory already exists: {exp_dir}\n"
            f"Pick a new `experiment.name` in your override config."
        )
    exp_dir.mkdir(parents=True, exist_ok=False)
    print(f"\nUsing experiment directory: {exp_dir.resolve()}")

    # ---- Step 1: Make training package ----
    print("\n=== STEP 1: Building training package ===")
    make_training_package.run_from_config(cfg, verbose=args.verbose)

    # ---- Step 2: Train model (placeholder) ----
    print("\n=== STEP 2: Training model (placeholder) ===")
    print(f"(Would train model with params: {cfg['training']})")

    print("\nPipeline complete!")

    print("Cleaning up training package...")
    exp_dir = make_training_package.get_experiment_dir(cfg)
    cleanup_training_package(exp_dir)
    print("Done.")

if __name__ == "__main__":
    main()
