# BirdNET Frog Training Pipeline

⚠️ **Note**: This README is actively being updated as the pipeline is refactored.

---

## Experiment Pipeline Overview

This repository implements a reproducible pipeline for training and evaluating custom BirdNET classifiers for CRLF (*Rana draytonii*) detection.

The goal is to manage the entire lifecycle of model development through a single configuration file. Each experiment is fully reproducible and self-contained: given the same input config, the pipeline produces the same training package, model weights, inference results, and evaluation metrics.

---

## Workflow

### Dataset Preparation
We start from **pre-split data** (`train/`, `val/`, `test_iid/`, `test_ood/`) plus a manifest (`manifest_with_split.csv`) stored under `AudioData/`.  

- **make_training_package.py**  
  Builds BirdNET-style training packages from the splits using filters (quality, call type, sites, etc).  
  Packages are deterministic and logged with reports for reproducibility.

### Model Training
- **train.ps1 / train.py**  
  Trains a BirdNET custom classifier on the training package.  
  Outputs include model weights, training logs, and a snapshot of the config.

### Inference
- **run_inference.ps1**  
  Runs the trained model on Test-IID and Test-OOD splits.  
  Outputs combined CSV result tables for each.

### Evaluation
- **evaluate_results.py**  
  Computes precision, recall, F1, ROC/PR curves, and breakdowns by quality and call type.  
  Results are saved as CSV + plots.

---

## Artifact Organization

Each experiment is isolated under an `experiments/` folder:

experiments/model01/
config_used.yaml
training_package/
model/
model01.tflite
training_log.json
inference/
TestIID/BirdNET_CombinedTable.csv
TestOOD/BirdNET_CombinedTable.csv
evaluation/
metrics_summary.csv
metrics_by_group.csv
plots/

yaml
Copy code

This makes it easy to run multiple models (model01, model02, …), compare results, and trace each model back to the exact data and hyperparameters used.

---

## Configuration System

- **Base configs** live in `config/base.yaml`.  
- Each experiment defines an **override config** (e.g., `config/model01.yaml`) that references base values but changes specific settings (dataset filters, training hyperparameters, etc).  
- Each script saves a snapshot of the config it actually used for reproducibility.

---

## Next Steps

The pipeline is being refactored to:
- Ensure every script is config-driven.
- Save snapshots of inputs, parameters, and outputs.
- Provide a single driver script (`run_experiment.py`) that executes the full process end-to-end given one config file.