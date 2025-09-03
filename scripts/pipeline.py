# scripts/pipeline.py
"""
End-to-end pipeline for BirdNET Frog Training.
- Loads base + model config (YAML)
- Creates experiment directory
- Builds training package
- Runs training step
- Runs inference on Test-IID and Test-OOD
"""

import argparse
from pathlib import Path
from utils.config import load_config
import scripts.make_training_package as make_training_package
import scripts.collect_experiments as collect_experiments
import scripts.evaluate_results as evaluate_results
import shutil
import subprocess
import sys



def cleanup_training_package(exp_dir: Path):
    tp_dir = exp_dir / "training_package"
    for sub in ["RADR", "Negative"]:
        folder = tp_dir / sub
        if folder.exists():
            shutil.rmtree(folder)

def apply_args(cmd, args_dict):
    """Append arbitrary CLI args from config to a command."""
    for key, val in (args_dict or {}).items():
        if val is None or val is False:
            continue
        flag = f"--{key}"
        if isinstance(val, bool):
            cmd.append(flag)
        else:
            cmd.extend([flag, str(val)])
    return cmd


def build_training_cmd(cfg, exp_dir):
    train_cfg = cfg.get("training", {})
    extra_train = cfg.get("training_args", {})
    dataset = exp_dir / "training_package"
    outdir = exp_dir / "model"

    python_exe = sys.executable
    cmd = [python_exe, "-m", "birdnet_analyzer.train", str(dataset), "-o", str(outdir)]

    # standard training params
    for key, val in train_cfg.items():
        if val is None or val is False:
            continue
        flag = f"--{key}"
        if isinstance(val, bool):
            cmd.append(flag)
        else:
            cmd.extend([flag, str(val)])

    # NEW: training_args passthrough
    cmd = apply_args(cmd, extra_train)
    return cmd


def build_inference_cmd(cfg, exp_dir, split: str):
    inf_cfg = cfg.get("inference", {})
    extra_analyze = cfg.get("analyzer_args", {})
    python_exe = sys.executable

    model_path = find_model_file(exp_dir)
    dataset_cfg = cfg.get("dataset", {})
    audio_root = Path(dataset_cfg.get("audio_root", "AudioData"))
    test_split = audio_root / "splits" / split.lower()

    outdir = exp_dir / "inference" / split
    outdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        python_exe, "-m", "birdnet_analyzer.analyze",
        str(test_split),
        "-o", str(outdir),
        "-c", str(model_path),
        "--rtype", "csv",
        "--combine_results",
    ]

    cmd = apply_args(cmd, inf_cfg)        # normal inference config
    cmd = apply_args(cmd, extra_analyze)  # analyzer_args passthrough
    return cmd


def get_experiment_dir(cfg: dict) -> Path:
    """Resolve experiment directory from config."""
    exp_cfg = cfg.get("experiment", {})
    name = exp_cfg.get("name")
    if not name:
        raise ValueError("Config missing required field: experiment.name")
    return Path("experiments") / name


def find_model_file(exp_dir: Path) -> Path:
    """Find the trained .tflite model inside experiment dir."""
    # First look in model/ subdir
    model_dir = exp_dir / "model"
    if model_dir.exists():
        tflite_files = list(model_dir.glob("*.tflite"))
        if tflite_files:
            return tflite_files[0]

    # Fallback: look in root of experiment dir
    tflite_files = list(exp_dir.glob("*.tflite"))
    if tflite_files:
        return tflite_files[0]

    raise FileNotFoundError(f"No .tflite model found in {exp_dir} or {model_dir}")


def main():
    ap = argparse.ArgumentParser(description="BirdNET Frog Training pipeline")
    ap.add_argument("--base-config", type=Path, default=Path("config/base.yaml"))
    ap.add_argument("--override-config", type=Path, required=False)
    ap.add_argument("--verbose", action="store_true")
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

    # Step 1: Make training package
    print("\n=== STEP 1: Building training package ===")
    make_training_package.run_from_config(cfg, verbose=args.verbose)

    # Step 2: Train model
    print("\n=== STEP 2: Training model ===")
    config_snapshot = exp_dir / "config_used.yaml"
    if not config_snapshot.exists():
        raise FileNotFoundError(f"Config snapshot not found: {config_snapshot}")

    cmd = build_training_cmd(cfg, exp_dir)
    print("Running training:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # Step 3: Run inference
    print("\n=== STEP 3: Inference on Test-IID ===")
    cmd_iid = build_inference_cmd(cfg, exp_dir, "test_iid")
    print("Running inference:", " ".join(cmd_iid))
    subprocess.run(cmd_iid, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    print("\n=== STEP 3: Inference on Test-OOD ===")
    cmd_ood = build_inference_cmd(cfg, exp_dir, "test_ood")
    print("Running inference:", " ".join(cmd_ood))
    subprocess.run(cmd_ood, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    # Step 4: Evaluate results
    print("\n=== STEP 4: Evaluation ===")
    evaluate_results.run_evaluation(str(exp_dir))

    # Step 5: Update master experiment index
    print("\n=== STEP 5: Updating master experiment index ===")
    collect_experiments.collect_experiments("experiments", "all_experiments.csv")

    print("\nPipeline complete!")

    print("Cleaning up training package...")
    cleanup_training_package(exp_dir)
    print("Done.")


if __name__ == "__main__":
    main()
