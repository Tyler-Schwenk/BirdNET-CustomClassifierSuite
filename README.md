BirdNET Frog Training Pipeline

Warning - this readme might be out of date!!!

This repository contains a reproducible pipeline for training and evaluating custom BirdNET classifiers for California Red-Legged Frog (RADR).

Pipeline Overview (data flow):
Raw Audio Clips → Data Preprocessing (splits) → Train.ps1 (BirdNET Training) → Trained Model (.tflite) → Run_inference.ps1 (BirdNET Inference) → Evaluation CSVs → evaluate_results.py (Metrics + Plots) → Reports (metrics_summary.csv, metrics_by_group.csv, ROC/PR plots)

Steps

Prepare data

Store raw audio in AudioData/

Generate training/validation/test splits: train.csv, val.csv, test_iid.csv, test_ood.csv

Train model
Run: train.ps1 with parameters like dataset path, output folder, epochs, batch size, threads.
Example: .\train.ps1 -DATASET "data/training_packages/baseline_all_trainval" -OUTDIR "models/model01" -EPOCHS 50

Run inference
Run: run_inference.ps1 with model path.
Example: .\run_inference.ps1 -MODEL_PATH "models/model01/model01.tflite"
Outputs: evaluation/TestIID and evaluation/TestOOD folders containing CSV results.

Evaluate performance
Run: python evaluate_results.py
Generates:

metrics_summary.csv with precision/recall/F1/accuracy across thresholds

metrics_by_group.csv with performance by call quality and call type

roc_curve.png and pr_curve.png

Repo Structure

scripts/
train.ps1
run_inference.ps1
evaluate_results.py
models/
model01/
config.yaml
train.log
model01.tflite
evaluation/
metrics_summary.csv
metrics_by_group.csv
roc_curve.png
pr_curve.png
data/
manifests/
train.csv
val.csv
test_iid.csv
test_ood.csv

Running Experiments

Each trained model should have its own folder under models/.
Each model folder contains:

config.yaml (parameters used for training)

train.log (training logs)

.tflite model weights

evaluation outputs (CSV + plots)

This structure makes it easy to compare experiments and track progress.

Sample config.yaml

dataset: D:/important/projects/Frog/AudioData/ReviewedDataClean/training_packages/baseline_all_trainval
output_dir: D:/important/projects/Frog/models/model01
epochs: 50
batch_size: 64
threads: 4
val_split: 0.2
autotune: false
autotune_trials: 20
autotune_executions_per_trial: 2