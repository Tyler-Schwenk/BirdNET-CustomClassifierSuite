BirdNET Frog Training Pipeline

Warning - this readme might be out of date!!!

Experiment Pipeline Overview

This repository implements a reproducible pipeline for training and evaluating custom BirdNET classifiers for CRLF (Rana draytonii) detection.

The goal is to manage the entire lifecycle of model development through a single configuration file. Each experiment is fully reproducible and self-contained: given the same input config, the pipeline produces the same splits, training packages, model weights, inference results, and evaluation metrics.

Workflow

Dataset preparation

make_splits.py creates train/val/test (IID and OOD) splits from the master manifest.

make_training_package.py builds BirdNET-style training packages from those splits using filters (quality, call type, sites, etc).

Model training

train.py trains a BirdNET custom classifier on the training package.

Outputs include model weights, training logs, and a snapshot of the config.

Inference

run_inference.py runs the trained model on test-IID and test-OOD splits.

Outputs combined CSV result tables.

Evaluation

evaluate_results.py computes precision, recall, F1, ROC/PR curves, and breakdowns by quality and call type.

Results are saved as CSV + plots.

Artifact Organization

Each experiment is isolated under an experiments folder:

experiments/model01/
config_used.yaml
splits/
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

This makes it easy to run multiple models (model01, model02, …), compare results, and trace each model back to the exact data and hyperparameters used.

Configuration System

Base configs are kept in config/base.yaml.

Each experiment defines an override config (e.g., config/model01.yaml) that references base values but changes specific settings (dataset filters, training hyperparameters, etc).

Each script saves a snapshot of the config it actually used for reproducibility.

Next Steps

The pipeline is being refactored to:

Ensure every script is config-driven.

Save snapshots of inputs, parameters, and outputs.

Provide a single driver script (run_experiment.py) that executes the full process end-to-end given one config file.