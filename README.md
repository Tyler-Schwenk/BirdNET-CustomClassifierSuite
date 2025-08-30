BirdNET Frog Training Pipeline

⚠️ Note: This pipeline is under active development. All experiments are fully reproducible from configs.

Experiment Pipeline Overview

This repository implements a reproducible, config-driven pipeline for training and evaluating custom BirdNET classifiers for CRLF (Rana draytonii) detection.

Each experiment is self-contained: given the same input config, the pipeline produces the same training package, model weights, inference results, and evaluation metrics.

Workflow
1. Training Package Creation

make_training_package.py
Builds BirdNET-style training folders (RADR/, Negative/) from pre-split data and a manifest.
Supports filters for quality, call type, site, and recorder.
Outputs:

Copied audio organized into class folders.

selection_report.json/txt: summary of what was included.

data_summary.csv: detailed counts by quality, call type, site, and split.

config_used.yaml: exact snapshot of the config used.

⚠️ If an experiment folder already exists, the script will stop with a warning to avoid accidental overwrite.

2. Model Training

train.ps1 / train.py (placeholder for now)
Runs BirdNET training on the generated package.
Will output weights (.tflite), logs, and metrics.

3. Inference

run_inference.ps1
Applies the trained model to Test-IID and Test-OOD splits.
Produces combined BirdNET_CombinedTable.csv files per split.

4. Evaluation

evaluate_results.py
Aggregates inference results, computes metrics across thresholds, and produces breakdowns by quality, call type, and site.
Outputs:

metrics_summary.csv (precision/recall/F1 across thresholds)

metrics_by_group.csv (per-quality, per-calltype, etc.)

Plots (ROC, PR, threshold curves).

5. End-to-End Pipeline

pipeline.py
Single entry point to run steps 1–4 from a config file:

python -m scripts.pipeline --base-config config/base.yaml --override-config config/model01.yaml --verbose

Artifact Organization

Each experiment is isolated in its own folder under training_packages/ (and later experiments/):

training_packages/model01_highmed/
    RADR/
    Negative/
    selection_report.json
    data_summary.csv
    config_used.yaml
    ...
models/model01/
    model01.tflite
    training_log.json
inference/
    TestIID/BirdNET_CombinedTable.csv
    TestOOD/BirdNET_CombinedTable.csv
evaluation/
    metrics_summary.csv
    metrics_by_group.csv
    plots/


This structure makes it easy to run multiple experiments (model01, model02, …), compare results, and trace each model back to the exact inputs and parameters.

Configuration System

Base config: config/base.yaml defines defaults for dataset, training, inference, and evaluation.

Override configs (e.g., config/model01.yaml) specialize experiments by changing only what’s needed.

The system merges base + override → final config snapshot is always saved with the results.

Next Steps

Full integration of training & inference into the pipeline.

Automated evaluation logging into the experiment folder.

Support for reproducible “experiment registry” in experiments/.