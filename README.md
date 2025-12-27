# BirdNET Custom Classifier Suite

A framework for training, evaluating, and optimizing species-specific acoustic classifiers using BirdNET-Analyzer.

## Overview

This suite extends BirdNET-Analyzer with tools for systematic model development and evaluation. It automates the complete ML pipeline from data preparation through model comparison, with a focus on reproducibility and experimental tracking.

**Core capabilities:**
- Automated training package generation from labeled audio datasets
- Hyperparameter sweep orchestration across hundreds of configurations
- Standardized evaluation on in-distribution (IID) and out-of-distribution (OOD) test splits
- Signature-based experiment grouping for statistical analysis across seeds
- Interactive UI for comparing model performance and identifying optimal configurations

Originally developed for California red-legged frog (Rana draytonii) detection, adaptable to any species with labeled training data.

## Quick Start

### Installation

```bash
# Clone repository
git clone [repository-url]
cd BirdNET_CustomClassifierSuite

# Setup environment (Windows)
.\setup.ps1

# Setup environment (Unix)
chmod +x setup.sh
./setup.sh
```

### Basic Usage

```bash
# Launch UI
streamlit run scripts/streamlit_app.py

# Or from CLI

# Run single experiment
python -m birdnet_custom_classifier_suite.pipeline.pipeline \
    --base-config config/stage_1_base.yaml \
    --override-config config/sweeps/stage1_sweep/stage1_001.yaml

# Collect results
python -m birdnet_custom_classifier_suite.pipeline.collect_experiments

```

## Project Structure

```
birdnet_custom_classifier_suite/
├── cli/              # Command-line interface
├── eval_toolkit/     # Experiment analysis and ranking
├── pipeline/         # Training and evaluation pipeline
├── sweeps/           # Hyperparameter sweep generation
├── ui/               # Streamlit interface components
└── utils/            # Shared utilities

config/               # Experiment configurations
├── sweep_specs/      # Sweep definitions
└── sweeps/           # Generated experiment configs

experiments/          # Experiment outputs
└── stage*_*/         # Results by experiment

docs/                 # Documentation
scripts/              # Utility scripts
tests/                # Test suite
```

## Key Features

### Data Management
- Manifest-based audio file tracking with quality labels
- Stratified train/test splits with temporal and spatial grouping
- Negative sample curation and hard negative mining
- Flexible filtering by quality, dataset, and label type

### Training Pipeline
- Automated training package assembly from manifest
- Data augmentation and upsampling strategies
- GPU memory management for large-scale sweeps
- Model checkpoint and parameter export

### Evaluation System
- Per-file scoring with confidence thresholds
- Precision-recall curves and F1 optimization
- Group-level metrics (by recorder, date, quality)
- OOD generalization assessment

### Experiment Tracking
- Configuration signatures for run deduplication
- Statistical aggregation across random seeds
- Stability metrics (precision/recall variance)
- Leaderboard ranking with configurable criteria

### Analysis UI
- Interactive filtering and metric selection
- Signature-level comparison across seeds
- Distribution visualization and outlier detection
- Export functionality for results and configs

## Configuration

Experiments are defined through YAML configs with base/override structure:

**Base config** (`config/stage_N_base.yaml`):
```yaml
dataset:
  audio_root: AudioData
  manifest: data/manifest.csv

training_package:
  include_negatives: true
  quality: [high, medium, low]

training_args:
  fmax: 15000
  dropout: 0.25
  learning_rate: 0.0005

analyzer_args:
  sensitivity: 1.0
```

**Override config** (`config/sweeps/stageN_sweep/stageN_001.yaml`):
```yaml
experiment:
  name: stageN_001
  seed: 123

training_args:
  learning_rate: 0.001
```

See [docs/](docs/) for detailed configuration options.

## Hyperparameter Sweeps

Generate sweep configurations from specifications:

```bash
python -m birdnet_custom_classifier_suite.sweeps.generate_sweep \
    --spec config/sweep_specs/stage17_spec.yaml \
    --output config/sweeps/stage17_sweep/
```

Run sweep experiments:
```bash
# Sequential
for i in {1..108}; do
    python -m birdnet_custom_classifier_suite.pipeline.pipeline \
        --override-config config/sweeps/stage17_sweep/stage17_$(printf "%03d" $i).yaml
done

# Or use parallel execution (see scripts/)
```

## Evaluation Workflow

1. **Train model** → Generates `.tflite` file in `experiments/*/`
2. **Run inference** → Creates `inference/test_{iid,ood}/BirdNET_CombinedTable.csv`
3. **Evaluate results** → Computes metrics in `evaluation/experiment_summary.json`
4. **Collect experiments** → Aggregates to `all_experiments.csv`
5. **Analyze in UI** → Compare performance and select optimal models

## Model Selection Criteria

The suite supports multi-objective optimization:

- **Max F1**: Best overall precision-recall balance
- **Max Precision**: Minimize false positives for high-volume deployment
- **Stability**: Low variance across seeds for production reliability
- **OOD Performance**: Generalization to new locations/times

Use the UI's filtering and ranking tools to identify models meeting specific deployment constraints.

## Re-evaluation

Re-run evaluation on existing models without retraining:

```bash
# Single experiment
python scripts/rerun_all_evaluations.py --stages stage17

# Skip training, use existing model
python -m birdnet_custom_classifier_suite.pipeline.pipeline \
    --override-config config/sweeps/stage17_sweep/stage17_028.yaml \
    --skip-training
```

See [docs/RE_EVALUATION_GUIDE.md](docs/RE_EVALUATION_GUIDE.md) for details.

## Documentation

- [PIPELINE_OVERVIEW.md](docs/PIPELINE_OVERVIEW.md) - End-to-end workflow
- [DATA_SPLITS.md](docs/DATA_SPLITS.md) - Train/test partitioning strategy
- [EVALUATION_PIPELINE.md](docs/EVALUATION_PIPELINE.md) - Metrics computation
- [SWEEPS.md](docs/SWEEPS.md) - Hyperparameter exploration
- [UI_ARCHITECTURE.md](docs/UI_ARCHITECTURE.md) - Streamlit interface design

## Requirements

- Python 3.10+
- BirdNET-Analyzer 2.4+
- TensorFlow 2.x (for training)
- Streamlit (for UI)

See [requirements.txt](requirements.txt) for complete dependencies.

## Citation

If you use this framework in your research, please cite:

```
[Citation information to be added]
```

Built on [BirdNET-Analyzer](https://github.com/kahst/BirdNET-Analyzer) by the K. Lisa Yang Center for Conservation Bioacoustics at the Cornell Lab of Ornithology.

## License

[License information to be added]
