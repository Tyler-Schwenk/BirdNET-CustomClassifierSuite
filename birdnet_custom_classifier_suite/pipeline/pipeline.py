# birdnet_custom_classifier_suite/pipeline/pipeline.py
"""
BirdNET Frog Training Pipeline
- Matches sweep behavior exactly
- Adds optional --skip-training to reuse an existing model
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from birdnet_custom_classifier_suite.utils.config import load_config
from birdnet_custom_classifier_suite.pipeline import make_training_package, make_validation_package, collect_experiments, evaluate_results
import shutil
import subprocess
import sys


# ---------------- Utility helpers ---------------- #

def cleanup_training_package(exp_dir: Path):
    tp_dir = exp_dir / "training_package"
    for sub in ["RADR", "Negative"]:
        folder = tp_dir / sub
        if folder.exists():
            shutil.rmtree(folder)


def cleanup_inference_dirs(exp_dir: Path):
    for split in ["test_iid", "test_ood"]:
        split_dir = exp_dir / "inference" / split
        for sub in ["positive", "negative"]:
            target = split_dir / sub
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
                print(f"Removed {target}")


def cleanup_validation_package(exp_dir: Path):
    """Remove Negative and RADR folders from validation package."""
    vp_dir = exp_dir / "validation_package"
    if not vp_dir.exists():
        return
    for sub in ["Negative", "RADR"]:
        folder = vp_dir / sub
        if folder.exists():
            shutil.rmtree(folder)
            print(f"Removed {folder}")


def apply_args(cmd, args_dict):
    for key, val in (args_dict or {}).items():
        if val is None or val is False:
            continue
        flag = f"--{key}"
        if isinstance(val, bool):
            cmd.append(flag)
        else:
            cmd.extend([flag, str(val)])
    return cmd


def get_experiment_dir(cfg: dict) -> Path:
    name = cfg.get("experiment", {}).get("name")
    if not name:
        raise ValueError("Missing experiment.name in config")
    return Path("experiments") / name


def find_model_file(exp_dir: Path) -> Path:
    model_dir = exp_dir / "model"
    if model_dir.exists():
        tflite_files = list(model_dir.glob("*.tflite"))
        if tflite_files:
            return tflite_files[0]
    tflite_files = list(exp_dir.glob("*.tflite"))
    if tflite_files:
        return tflite_files[0]
    raise FileNotFoundError(f"No .tflite model found in {exp_dir} or {model_dir}")


# ---------------- Command builders ---------------- #

def build_training_cmd(cfg, exp_dir):
    python_exe = sys.executable
    dataset = exp_dir / "training_package"
    outdir = exp_dir / "model"
    cmd = [python_exe, "-m", "birdnet_analyzer.train", str(dataset), "-o", str(outdir)]
    
    # Check if validation package should be used
    use_validation = cfg.get("training", {}).get("use_validation", False)
    if use_validation:
        validation_package = exp_dir / "validation_package"
        if validation_package.exists():
            cmd.extend(["--test_data", str(validation_package)])
            print(f"Using validation package: {validation_package}")
        else:
            print(f"WARNING: use_validation=True but validation package not found at {validation_package}")
    
    # Apply other training flags (exclude use_validation as it's not a BirdNET flag)
    training_args = {k: v for k, v in cfg.get("training", {}).items() if k != "use_validation"}
    cmd = apply_args(cmd, training_args)

    # Sanitize training_args to match BirdNET-Analyzer CLI
    ta = dict(cfg.get("training_args", {}) or {})
    
    # Remove inference-only parameters that don't apply to training
    inference_only_params = {"sensitivity"}
    for param in inference_only_params:
        if param in ta:
            ta.pop(param)
    
    # If an invalid upsampling_mode (e.g., 'none') sneaks in from older specs, drop it.
    allowed_modes = {"repeat", "linear", "mean", "smote"}
    mode = ta.get("upsampling_mode")
    if mode is not None and str(mode).lower() not in allowed_modes:
        print(f"[WARN] Ignoring unsupported upsampling_mode='{mode}'. Valid: repeat, linear, mean, smote.\n"
              f"       Tip: set upsampling_ratio=0.0 to disable upsampling.")
        ta.pop("upsampling_mode", None)
    cmd = apply_args(cmd, ta)
    return cmd


def build_inference_cmd(cfg, exp_dir, split: str):
    python_exe = sys.executable
    model_path = find_model_file(exp_dir)
    dataset_cfg = cfg.get("dataset", {})
    audio_root = Path(dataset_cfg.get("audio_root", "AudioData"))
    split_dir = audio_root / "splits" / split.lower()
    outdir = exp_dir / "inference" / split
    outdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        python_exe, "-m", "birdnet_analyzer.analyze",
        str(split_dir),
        "-o", str(outdir),
        "-c", str(model_path),
        "--rtype", "csv",
        "--combine_results",
    ]
    cmd = apply_args(cmd, cfg.get("inference", {}))
    cmd = apply_args(cmd, cfg.get("analyzer_args", {}))
    return cmd


# ---------------- Main pipeline ---------------- #

def main():
    ap = argparse.ArgumentParser(description="BirdNET Frog Training pipeline")
    ap.add_argument("--base-config", type=Path, default=None, help="Path to base.yaml; if omitted and --override-config is provided, inferred as <override-config>/../base.yaml")
    ap.add_argument("--override-config", type=Path)
    ap.add_argument("--skip-training", action="store_true", help="Skip training and reuse existing model/")
    ap.add_argument("--cpu-only-training", action="store_true", help="Train on CPU only (workaround for GPU memory issues)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    base_cfg = args.base_config
    if base_cfg is None and args.override_config is not None:
        # Infer per-sweep base.yaml next to the override config
        candidate = args.override_config.parent / "base.yaml"
        if candidate.exists():
            base_cfg = candidate
    if base_cfg is None:
        raise FileNotFoundError("--base-config is required (or provide --override-config and ensure a base.yaml exists next to it)")

    cfg = load_config(base_cfg, args.override_config) if args.override_config else load_config(base_cfg)
    print("Loaded config:")
    print(cfg)

    exp_dir = get_experiment_dir(cfg)
    if exp_dir.exists() and not args.skip_training:
        raise FileExistsError(
            f"Experiment directory already exists: {exp_dir}\n"
            f"Pick a new `experiment.name` or use --skip-training."
        )
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nUsing experiment directory: {exp_dir.resolve()}")

    # Step 1: Training package
    print("\n=== STEP 1: Building training package ===")
    make_training_package.run_from_config(cfg, verbose=args.verbose)
    
    # Step 1b: Validation package (if enabled)
    use_validation = cfg.get("training", {}).get("use_validation", False)
    if use_validation:
        print("\n=== STEP 1b: Building validation package ===")
        try:
            make_validation_package.run_from_config(cfg, verbose=args.verbose)
        except Exception as e:
            print(f"ERROR: Failed to build validation package: {e}")
            raise

    # Step 2: Train or skip
    if args.skip_training:
        print(f"\n=== STEP 2: Skipping training. Using existing model in {exp_dir/'model'} ===")
    else:
        print("\n=== STEP 2: Training model ===")
        cmd = build_training_cmd(cfg, exp_dir)
        
        # Set TensorFlow GPU environment variables for subprocess
        import os
        train_env = os.environ.copy()
        
        if args.cpu_only_training:
            print("  [CPU-only mode] Training will use CPU only (GPU will still be used for inference)")
            train_env['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU-only for training
        else:
            train_env.update({
                'TF_FORCE_GPU_ALLOW_GROWTH': 'true',
                'TF_GPU_THREAD_MODE': 'gpu_private',
                'TF_GPU_THREAD_COUNT': '1',
                'TF_CPP_MIN_LOG_LEVEL': '0',
                'MALLOC_MMAP_THRESHOLD_': '131072',  # Fixes tcache double-free
                'MALLOC_TRIM_THRESHOLD_': '131072',
            })
        
        print("Running training:", " ".join(cmd))
        subprocess.run(cmd, check=True, env=train_env)

    # Step 3: Inference
    for split in ["test_iid", "test_ood"]:
        print(f"\n=== STEP 3: Inference on {split.upper()} ===")
        cmd = build_inference_cmd(cfg, exp_dir, split)
        print("Running inference:", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        print("\n--- Inference STDOUT ---\n", result.stdout)
        print("\n--- Inference STDERR ---\n", result.stderr)
        if result.returncode != 0:
            raise RuntimeError(f"Inference on {split} failed with exit code {result.returncode}")

    # Step 4: Evaluation
    print("\n=== STEP 4: Evaluation ===")
    evaluate_results.run_evaluation(str(exp_dir))

    # Step 5: Update master experiment index
    print("\n=== STEP 5: Updating master experiment index ===")
    collect_experiments.collect_experiments("experiments", "results/all_experiments.csv")

    print("\nPipeline complete!")

    print("Cleaning up training package...")
    cleanup_training_package(exp_dir)

    print("Cleaning up validation package...")
    cleanup_validation_package(exp_dir)

    print("Cleaning up inference subfolders...")
    cleanup_inference_dirs(exp_dir)
    print("Done.")


if __name__ == "__main__":
    main()
