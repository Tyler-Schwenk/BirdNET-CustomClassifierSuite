# scripts/pipeline.py
"""
End-to-end pipeline for BirdNET Frog Training.
- Loads base + model config (YAML)
- Builds training package
- (Placeholder) Runs training step
"""

import argparse
from pathlib import Path
from utils.config import load_config
import scripts.make_training_package as make_training_package


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
    print("✅ Loaded config with overrides:")
    print(cfg)

    # ---- Step 1: Make training package ----
    print("\n=== STEP 1: Building training package ===")
    make_training_package.run_from_config(cfg, verbose=args.verbose)

    # ---- Step 2: Train model (placeholder) ----
    print("\n=== STEP 2: Training model (placeholder) ===")
    print(f"(Would train model with params: {cfg['training']})")

    print("\n🎉 Pipeline complete!")


if __name__ == "__main__":
    main()
