# birdnet_custom_classifier_suite/sweeps/run_sweep.py
"""
Run a batch of experiment configs through the BirdNET Frog Training pipeline.

Usage:
    python -m run_sweep config/sweeps/
    python -m run_sweep config/sweeps/ --verbose

    

"""

import argparse
import subprocess
import sys
from pathlib import Path
import yaml


def get_experiment_name(config_path: Path) -> str:
    """Extract experiment.name from a YAML config."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        exp_cfg = cfg.get("experiment", {})
        return exp_cfg.get("name")
    except Exception:
        return None


def run_pipeline(config: Path, base_config: Path, verbose: bool = False) -> bool:
    """Run pipeline.py for a single config. Returns True if successful."""
    cmd = [
        sys.executable, "-m", "birdnet_custom_classifier_suite.pipeline.pipeline",
        "--base-config", str(base_config),
        "--override-config", str(config),
    ]
    if verbose:
        cmd.append("--verbose")

    print(f"\n=== Running experiment: {config.name} ===")
    try:
        subprocess.run(cmd, check=True)
        print(f"Success: {config.name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed: {config.name} (exit code {e.returncode})")
        return False
    except Exception as e:
        print(f"Failed: {config.name} ({e})")
        return False


def main():
    ap = argparse.ArgumentParser(description="Batch runner for BirdNET pipeline")
    ap.add_argument("sweep_dir", type=Path, help="Directory of override config YAMLs")
    ap.add_argument("--base-config", type=Path, default=Path("config/base.yaml"))
    ap.add_argument("--experiments-root", type=Path, default=Path("experiments"))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not args.sweep_dir.exists():
        print(f"Sweep directory not found: {args.sweep_dir}")
        sys.exit(1)

    configs = sorted(args.sweep_dir.glob("*.yaml"))
    if not configs:
        print(f"No YAML configs found in {args.sweep_dir}")
        sys.exit(1)

    print(f"Found {len(configs)} configs in {args.sweep_dir}")

    results = []
    for cfg in configs:
        exp_name = get_experiment_name(cfg)
        if not exp_name:
            print(f"Skipping {cfg.name}: no experiment.name found")
            results.append((cfg.name, "SKIP"))
            continue

        exp_dir = args.experiments_root / exp_name
        if exp_dir.exists():
            print(f"Skipping {cfg.name}: experiment folder already exists ({exp_dir})")
            results.append((cfg.name, "SKIP"))
            continue

        ok = run_pipeline(cfg, args.base_config, args.verbose)
        results.append((cfg.name, "OK" if ok else "FAIL"))

    print("\n=== Sweep complete ===")
    for cfg, status in results:
        print(f"{cfg}: {status}")


if __name__ == "__main__":
    main()
