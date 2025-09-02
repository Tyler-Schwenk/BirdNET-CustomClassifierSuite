BirdNET Frog Training Pipeline

⚠️ Note: This pipeline is under active development. All experiments are fully reproducible from configs.

Experiment Pipeline Overview

This repository implements a reproducible, config-driven pipeline for training and evaluating custom BirdNET classifiers for CRLF (Rana draytonii) detection.

Each experiment is self-contained: given the same input config, the pipeline produces the same training package, model weights, inference results, and evaluation metrics.

Workflow
1. Training Package Creation

scripts/make_training_package.py
Builds BirdNET-style training folders (RADR/, Negative/) from pre-split data and a manifest.

Supports filters: quality, call type, site, recorder.

Outputs:

RADR/, Negative/ folders (copied audio).

selection_report.json / .txt: summary of what was included.

data_summary.csv: detailed counts by quality, call type, site, and split.

config_used.yaml: exact snapshot of the config used.

⚠️ If an experiment folder already exists, the script will stop with a warning to avoid accidental overwrite.

2. Model Training

scripts/pipeline.py (calls birdnet_analyzer.train)
Runs BirdNET training on the generated package.

Inputs:

experiments/<name>/training_package/ (RADR/ + Negative/).

experiments/<name>/config_used.yaml (exact snapshot of config).

Outputs into experiments/<name>/model/:

model.tflite → trained model weights.

training_log.json → log of training/validation metrics.

Any additional artifacts from BirdNET-Analyzer.

Config-driven parameters:
epochs, batch_size, threads, val_split, autotune, etc.

If autotune: true is set, hyperparameter search is performed.

After training, the RADR/ and Negative/ folders are deleted automatically (the exact selection remains reproducible via selection_manifest.csv).

3. Inference

scripts/pipeline.py (or scripts/run_inference.ps1)
Applies the trained model to Test-IID and Test-OOD splits.

Inputs:

Model (experiments/<name>/model/*.tflite).

Audio test splits (AudioData/splits/test_iid/, AudioData/splits/test_ood/).

Outputs under experiments/<name>/inference/:

test_iid/BirdNET_CombinedTable.csv

test_ood/BirdNET_CombinedTable.csv

BirdNET_analysis_params.csv (metadata from BirdNET).

Config values (threads, batch_size, min_conf) are passed automatically.

4. Evaluation

scripts/evaluate_results.py
Aggregates inference results, computes metrics across thresholds, and produces breakdowns by quality and call type.

Outputs into experiments/<name>/evaluation/:

metrics_summary.csv → precision, recall, F1, accuracy, FPR, TNR across thresholds.

metrics_by_group.csv → per-group breakdown (quality, call type).

roc_curve_<split>.png → ROC plots (Test-IID and Test-OOD).

pr_curve_<split>.png → Precision-Recall plots.

Metrics are aligned with the config thresholds, and every experiment folder contains complete reproducible evaluation artifacts.

5. End-to-End Pipeline

scripts/pipeline.py
Single entry point to run steps 1–4 from a config file:

python -m scripts.pipeline --base-config config/base.yaml --override-config config/model01.yaml --verbose


This builds the package, trains the model, runs inference on both test splits, and produces evaluation metrics/plots.

Artifact Organization

Each experiment is isolated under experiments/:

experiments/model01_highmed/
    config_used.yaml
    training_package/
        RADR/
        Negative/
        selection_report.json
        selection_manifest.csv
        data_summary.csv
    model/
        model.tflite
        training_log.json
    inference/
        test_iid/BirdNET_CombinedTable.csv
        test_ood/BirdNET_CombinedTable.csv
    evaluation/
        metrics_summary.csv
        metrics_by_group.csv
        roc_curve_test_iid.png
        pr_curve_test_iid.png
        roc_curve_test_ood.png
        pr_curve_test_ood.png


This structure makes it easy to:

Run multiple experiments (model01, model02, …).

Compare results directly.

Trace every trained model back to the exact config, manifest, and data subset used.

Configuration System

Base config (config/base.yaml): defines defaults for dataset, training, inference, and evaluation.

Override configs (config/model01.yaml, etc.): specialize experiments by changing only what’s needed.

Final merged config is always saved into config_used.yaml in each experiment folder.

This ensures reproducibility across machines and time.