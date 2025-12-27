#!/usr/bin/env python3
"""
Run the old 2024 model with additional sensitivity values (0.75, 1.25, 1.5)
Reuses the existing model - only runs inference and evaluation.
"""

import shutil
import subprocess
import sys
import yaml
from pathlib import Path


def create_experiment_for_sensitivity(base_exp_dir: Path, model_path: Path, labels_path: Path, sensitivity: float):
    """Create new experiment directory with config and copy model."""
    
    # Create experiment name
    sens_str = str(sensitivity).replace(".", "")
    exp_name = f"stage0_00_OldModel2024_baseline_mc25_sens{sens_str}_seed01"
    exp_dir = Path("experiments") / exp_name
    
    print(f"\n=== Setting up {exp_name} ===")
    
    # Create directories
    exp_dir.mkdir(parents=True, exist_ok=True)
    model_dir = exp_dir / "model"
    model_dir.mkdir(exist_ok=True)
    
    # Copy model
    dest_model = model_dir / "CustomClassifier.tflite"
    if not dest_model.exists():
        print(f"Copying model to {dest_model}")
        shutil.copy2(model_path, dest_model)
    else:
        print(f"Model already exists at {dest_model}")
    
    # Copy and fix labels file encoding
    dest_labels = model_dir / "CustomClassifier_Labels.txt"
    if not dest_labels.exists():
        print(f"Copying labels file with UTF-8 encoding")
        # Read with latin-1 (handles all bytes) and write as UTF-8
        try:
            with open(labels_path, 'r', encoding='latin-1') as f:
                content = f.read()
            with open(dest_labels, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            print(f"WARNING: Could not copy labels file: {e}")
            # Try direct copy as fallback
            shutil.copy2(labels_path, dest_labels)
    else:
        print(f"Labels file already exists at {dest_labels}")
    
    # Create config
    config = {
        "experiment": {
            "name": f"000_OldModel2024_baseline_mc25_sens{sens_str}_seed01",
            "seed": 1
        },
        "dataset": {
            "audio_root": "AudioData",
            "manifest": "data/manifest.csv"
        },
        "inference": {
            "batch_size": 32,
            "threads": 4,
            "min_conf": 0.25
        },
        "analyzer_args": {
            "fmin": 0,
            "fmax": 15000,
            "overlap": 0.0,
            "sensitivity": sensitivity
        },
        "evaluation": {
            "thresholds": [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "output_dir": "evaluation"
        }
    }
    
    config_path = exp_dir / "config_used.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    print(f"Created config at {config_path}")
    
    return exp_dir, config


def run_inference(exp_dir: Path, config: dict):
    """Run BirdNET inference."""
    print(f"\n=== Running inference for {exp_dir.name} ===")
    
    model_path = exp_dir / "model" / "CustomClassifier.tflite"
    audio_root = Path(config["dataset"]["audio_root"])
    
    # Run inference on test_iid and test_ood
    for split in ["test_iid", "test_ood"]:
        split_dir = audio_root / "splits" / split
        out_dir = exp_dir / "inference" / split
        out_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"  Running inference on {split}...")
        
        cmd = [
            sys.executable, "-m", "birdnet_analyzer.analyze",
            str(split_dir),
            "-o", str(out_dir),
            "-c", str(model_path),
            "--rtype", "csv",
            "--combine_results",
            "--batch_size", str(config["inference"]["batch_size"]),
            "--threads", str(config["inference"]["threads"]),
            "--min_conf", str(config["inference"]["min_conf"]),
            "--fmin", str(config["analyzer_args"]["fmin"]),
            "--fmax", str(config["analyzer_args"]["fmax"]),
            "--overlap", str(config["analyzer_args"]["overlap"]),
            "--sensitivity", str(config["analyzer_args"]["sensitivity"])
        ]
        
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            print(f"WARNING: Inference failed for {split}")
            return False
    
    return True


def run_evaluation(exp_dir: Path):
    """Run evaluation on inference results."""
    print(f"\n=== Running evaluation for {exp_dir.name} ===")
    
    cmd = [
        sys.executable, "-m",
        "birdnet_custom_classifier_suite.pipeline.evaluate_results",
        "--experiment-name", exp_dir.name
    ]
    
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"WARNING: Evaluation failed")
        return False
    
    return True


def main():
    # Source model location
    base_exp_dir = Path("experiments/stage0_00_OldModel2024_baseline_mc25_sens50_seed01")
    model_path = base_exp_dir / "model" / "CustomClassifier.tflite"
    labels_path = base_exp_dir / "model" / "CustomClassifier_Labels.txt"
    
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        sys.exit(1)
    
    if not labels_path.exists():
        print(f"ERROR: Labels file not found at {labels_path}")
        sys.exit(1)
    
    print(f"Using model: {model_path}")
    
    # Sensitivity values to test
    sensitivities = [0.75, 1.25, 1.5]
    
    for sens in sensitivities:
        exp_dir, config = create_experiment_for_sensitivity(base_exp_dir, model_path, labels_path, sens)
        
        # Run inference
        if not run_inference(exp_dir, config):
            print(f"FAILED: Inference for sensitivity {sens}")
            continue
        
        # Run evaluation
        if not run_evaluation(exp_dir):
            print(f"FAILED: Evaluation for sensitivity {sens}")
            continue
        
        print(f"SUCCESS: Completed experiment for sensitivity {sens}")
    
    print("\n=== All experiments complete ===")
    print("\nYou can now analyze results with:")
    print("  python -m birdnet_custom_classifier_suite.cli.analyze")


if __name__ == "__main__":
    main()
