# scripts/pipeline.py
"""
End-to-end pipeline for BirdNET Frog Training.
- Loads base + model config (YAML)
- Creates experiment directory
- Builds training package
- Runs training step
"""

import argparse
from pathlib import Path
from utils.config import load_config
import scripts.make_training_package as make_training_package
import shutil
import subprocess
import sys

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

def build_training_cmd(cfg, exp_dir):
    """Build BirdNET training command dynamically from config."""
    train_cfg = cfg.get("training", {})
    dataset = exp_dir / "training_package"
    outdir = exp_dir / "model"

    python_exe = sys.executable
    cmd = [python_exe, "-m", "birdnet_analyzer.train", str(dataset), "-o", str(outdir)]

    for key, val in train_cfg.items():
        if val is None or val is False:
            continue
        flag = f"--{key}"  # use snake_case directly
        if isinstance(val, bool):
            cmd.append(flag)
        else:
            cmd.extend([flag, str(val)])

    return cmd

def main():
    ap = argparse.ArgumentParser(description="BirdNET Frog Training pipeline")
    ap.add_argument(
        "--base-config",
        type=Path,
        default=Path("config/base.yaml"),
        help="Base config file (YAML)",
    )
    ap.add_argument(
        "--override-config",
        type=Path,
        required=False,
        help="Optional override config file (YAML)",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
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

    # ---- Step 2: Train model ----
    print("\n=== STEP 2: Training model ===")
    config_snapshot = exp_dir / "config_used.yaml"
    if not config_snapshot.exists():
        raise FileNotFoundError(f"Config snapshot not found: {config_snapshot}")

    cmd = build_training_cmd(cfg, exp_dir)
    print("Running training:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    print("\nPipeline complete!")

    print("Cleaning up training package...")
    cleanup_training_package(exp_dir)
    print("Done.")


if __name__ == "__main__":
    main()
