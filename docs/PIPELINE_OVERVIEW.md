# BirdNET Custom Classifier Suite — Complete Pipeline Overview

## Purpose

This suite automates the training and evaluation of **BirdNET custom classifiers** for detecting California Red-legged Frog (*Rana draytonii*, species code **RADR**) calls in acoustic recordings.

**Key Goal:** Test how different training data compositions and hyperparameters affect model generalization to new sites (out-of-distribution performance).

---

## Architecture Overview

```
AudioData/              ← Raw audio files split into train/val/test_iid/test_ood
    ├── splits/
    │   ├── train/      (12,110 files: 3,585 pos, 8,525 neg)
    │   ├── val/        (3,249 files: 647 pos, 2,602 neg)
    │   ├── test_iid/   (2,943 files: 1,964 pos, 979 neg)
    │   └── test_ood/   (3,585 files: 1,691 pos, 1,894 neg) ← PRIMARY METRIC

data/
    └── manifest.csv    ← Master file tracking all audio with metadata

config/
    ├── *.yaml          ← Base configs for experiments
    └── sweeps/         ← Sweep definitions (vary hyperparams)

experiments/            ← One folder per experiment
    └── {experiment_name}/
        ├── training_package/         ← Filtered subset of train/val data
        ├── validation_package/       ← Tracks files used (val split)
        ├── checkpoints/              ← Trained TensorFlow model
        │   └── model.tflite
        ├── inference/                ← BirdNET predictions
        │   ├── test_iid/
        │   │   └── BirdNET_CombinedTable.csv
        │   └── test_ood/
        │       └── BirdNET_CombinedTable.csv
        └── evaluation/               ← Metrics and summaries
            ├── metrics_summary.csv
            └── experiment_summary.json

all_experiments.csv     ← Aggregated results across all experiments
```

---

## Pipeline Stages

### Stage 1: Training Package Creation
**Module:** `pipeline/make_training_package.py`

**Purpose:** Filter and prepare training data based on config specifications.

**Process:**
1. Load `data/manifest.csv`
2. Filter by quality, call type, balance, etc. (from YAML config)
3. Create symlinks to selected files in `experiments/{name}/training_package/`
4. Save `training_manifest.csv` with selected files

**Key Config Options:**
```yaml
dataset:
  filters:
    quality: ["high", "medium"]      # Exclude low-quality positives
    call_type: ["Flight", "Song"]    # Only these call types
    balance: "balanced"              # Downsample negatives to match positives
    positive_subsets: []             # Empty = use all available
    negative_subsets: []             # Empty = use all available
```

**Output:** `experiments/{name}/training_package/{train,val}/` with symlinked .wav files

---

### Stage 2: Validation Package Creation
**Module:** `pipeline/make_validation_package.py`

**Purpose:** Reserve a fixed validation set for monitoring training.

**Process:**
1. Select files from 'val' split in manifest
2. Create symlinks in `experiments/{name}/validation_package/`
3. Save `validation_manifest.csv` tracking which files are used

**Note:** Currently optional; many experiments use `use_validation: false` to maximize training data.

**Output:** `experiments/{name}/validation_package/` with validation files

---

### Stage 3: Model Training
**Module:** `pipeline/pipeline.py::train()`

**Purpose:** Train BirdNET custom classifier using TensorFlow.

**Process:**
1. Call external BirdNET training script:
   ```bash
   python external/BirdNET-Analyzer/train.py \
       --i experiments/{name}/training_package/ \
       --o experiments/{name}/checkpoints/ \
       --crop_mode 'center' \
       --crop_overlap 0.0 \
       --batch_size 32 \
       --epochs 100 \
       --learning_rate 0.001 \
       ...
   ```
2. BirdNET trains on 3-second spectrogram crops
3. Model saved as `model.tflite` (TensorFlow Lite format)

**Key Hyperparameters:**
- `batch_size`: Examples per gradient update
- `epochs`: Training iterations
- `learning_rate`: Optimizer step size
- `hidden_units`: Model capacity
- `mixup`: Data augmentation (blend samples)
- `upsampling_mode`: Handle class imbalance ("repeat" or "oversample")

**Output:** `experiments/{name}/checkpoints/model.tflite`

---

### Stage 4: Inference (Prediction)
**Module:** `pipeline/pipeline.py::inference()`

**Purpose:** Run trained model on test sets to generate predictions.

**Process:**
For each test split (`test_iid`, `test_ood`):
1. Build BirdNET analyzer command:
   ```bash
   python external/BirdNET-Analyzer/analyze.py \
       --i AudioData/splits/test_ood/ \
       --o experiments/{name}/inference/test_ood/ \
       --classifier experiments/{name}/checkpoints/model.tflite \
       --min_conf 0.0 \
       --rtype csv
   ```
2. BirdNET analyzes all .wav files in split directory
3. Outputs detections to `BirdNET_CombinedTable.csv`

**Critical Setting:** `--min_conf 0.0` accepts ALL detections (no filtering)

**Output CSV Columns:**
```
Begin File              # Audio filename
Begin Time (s)          # Detection start time
End Time (s)            # Detection end time
Scientific name         # Species code (e.g., "RADR")
Confidence              # Model confidence (0.0–1.0)
```

**Known Issue:** BirdNET may not output rows for files with no detections or only non-target species detected. This causes missing files in evaluation (see [EVALUATION_PIPELINE.md](EVALUATION_PIPELINE.md#known-issues--bugs)).

**Output:** `experiments/{name}/inference/{test_iid,test_ood}/BirdNET_CombinedTable.csv`

---

### Stage 5: Evaluation
**Module:** `pipeline/evaluate_results.py`

**Purpose:** Compute precision/recall/F1 metrics on test sets.

**Process:**
1. Load **ALL audio files** from `AudioData/splits/{test_iid,test_ood}/` directories
2. Load predictions from `BirdNET_CombinedTable.csv`
3. For files with predictions: compute score = **max RADR confidence** across all detections
4. For files missing from BirdNET output: assign score = 0.0 (no RADR detected)
5. Sweep classification thresholds (0.0, 0.05, ..., 1.0)
6. At each threshold: compute TP, FP, TN, FN, precision, recall, F1
7. Select best threshold (max F1) per split
8. Save metrics and summary

**Scoring Logic:**
- **File-level scoring** (not detection-level)
- **Max confidence** among all RADR detections in file
- **Missing from BirdNET output → 0.0** (no detection)
- **Threshold applied post-hoc** during metric computation

**Critical Fix (2025-12-21):**
Evaluation now loads the complete file list from split directories to ensure ALL files are evaluated, even if BirdNET didn't output predictions for them. This fixed a bug where files with no detections were excluded from evaluation, causing artificially inflated metrics.

**Outputs:**
- `experiments/{name}/evaluation/metrics_summary.csv` — All thresholds
- `experiments/{name}/evaluation/experiment_summary.json` — Best threshold per split

**See:** [EVALUATION_PIPELINE.md](EVALUATION_PIPELINE.md) for detailed evaluation logic.

---

### Stage 6: Results Collection
**Module:** `pipeline/collect_experiments.py`

**Purpose:** Aggregate all experiment results into a single CSV for analysis.

**Process:**
1. Scan `experiments/*/evaluation/experiment_summary.json`
2. Load config YAML for each experiment
3. Flatten nested structure into tabular format
4. Append to `all_experiments.csv`

**Output Schema:** See [DATA_MODEL.md](DATA_MODEL.md) for column definitions.

**Key Columns:**
- `experiment.name`, `experiment.seed`
- `dataset.filters.*` (quality, balance, call_type, etc.)
- `training.*` (epochs, batch_size, learning_rate, etc.)
- `metrics.iid.best_f1.*` (threshold, precision, recall, f1)
- `metrics.ood.best_f1.*` (threshold, precision, recall, f1)

**Output:** `all_experiments.csv` (project root)

---

## Sweeps (Hyperparameter Search)

**Module:** `sweeps/`

**Purpose:** Automatically run multiple experiments varying config parameters.

**Types:**
1. **Data Composition Sweeps** — Test different training data subsets
2. **Hyperparameter Sweeps** — Test different model/training settings
3. **Seed Sweeps** — Test reproducibility across random initializations

**See:** [SWEEPS.md](SWEEPS.md) for sweep configuration syntax.

**Example Sweep:**
```yaml
base_config: config/stage_4_base.yaml

sweep_params:
  training.batch_size: [16, 32, 64]
  training.learning_rate: [0.0001, 0.001, 0.01]

num_seeds: 3  # Run each config with 3 different random seeds
```

**Process:**
1. Generate cartesian product of all parameter combinations
2. For each combo × seed:
   - Create unique experiment name (e.g., `stage4_sweep_001`)
   - Merge params with base config
   - Run full pipeline (package → train → inference → evaluate)
3. Collect all results into `all_experiments.csv`

---

## UI (Analysis & Visualization)

**Module:** `ui/`

**Purpose:** Interactive Streamlit app for analyzing experiment results.

**Features:**
- Load `all_experiments.csv` or specific experiment JSONs
- Filter by quality, balance, sweep, etc.
- Rank configs by F1, stability (inverse CV), or combined score
- Compare IID vs OOD metrics
- Visualize top-N configs

**See:** [UI_ARCHITECTURE.md](UI_ARCHITECTURE.md) for component details.

---

## Key Concepts

### In-Distribution (IID) vs. Out-of-Distribution (OOD)

| Split | Sites | Purpose |
|-------|-------|---------|
| **Train** | Cole Creek, Rancho Meling, Wheatley | Model learns patterns |
| **Val** | Same sites as train | Optional validation during training |
| **Test-IID** | Same sites as train | Test on familiar conditions |
| **Test-OOD** | **Sylvan Pond ONLY** (Moth11+12) | **Test on NEW site** |

**Critical:** OOD is the primary metric because it measures real-world generalization. Models can achieve high IID scores by memorizing training site characteristics, but OOD reveals true robustness.

### File-Level Scoring

Unlike detection-level evaluation (common in object detection), this pipeline scores **entire audio files**:

1. BirdNET may detect multiple RADR calls in one file at different times
2. Each detection has a confidence score (0.0–1.0)
3. File score = **maximum confidence** among all RADR detections
4. If no RADR detected, file score = 0.0

**Rationale:** For acoustic monitoring, we care whether a file contains RADR, not how many times.

### Threshold Sweep

- Model outputs confidence scores, not binary predictions
- Pipeline tests 21 thresholds (0.0, 0.05, ..., 1.0) post-hoc
- For each threshold: classify file as positive if score ≥ threshold
- Report metrics at threshold with best F1 score

**Example:**
```
Threshold 0.2: P=0.85, R=0.90, F1=0.875
Threshold 0.3: P=0.92, R=0.85, F1=0.884 ← BEST
Threshold 0.4: P=0.95, R=0.78, F1=0.856
```

Best F1 (0.884) occurs at threshold 0.3, so that's reported as the experiment's result.

---

## Common Workflows

### Run a Single Experiment
```bash
python -m birdnet_custom_classifier_suite.cli.pipeline run \
    --config config/example.yaml \
    --name my_experiment_001 \
    --seed 42
```

### Run a Sweep
```bash
python -m birdnet_custom_classifier_suite.cli.sweeps run \
    --sweep-config config/sweeps/data_composition.yaml
```

### Collect Results
```bash
python -m birdnet_custom_classifier_suite.pipeline.collect_experiments
```

### Launch UI
```bash
streamlit run birdnet_custom_classifier_suite/ui/app.py
```

---

## File Organization Best Practices

### Experiment Naming Convention
- `stage{N}_{descriptor}_{run_id}`
- Example: `stage14_013` = Stage 14, run 13
- Stages represent different research phases (data composition tests, hyperparameter tuning, etc.)

### Config Files
- `config/{stage}_base.yaml` — Default settings for a stage
- `config/sweeps/{stage}_sweep.yaml` — Sweep definition varying params
- One base config per major experiment phase

### Results Storage
- **Never delete experiment folders** — they contain irreplaceable trained models
- Use `all_experiments.csv` for analysis (don't manually edit experiment folders)
- Archive old experiments if disk space needed

---

## Data Flow Summary

```
Raw Audio (AudioData/splits/)
    ↓
[Package] Filter by config (quality, balance, etc.)
    ↓
Training Package (symlinks to selected files)
    ↓
[Train] BirdNET TensorFlow training
    ↓
Trained Model (model.tflite)
    ↓
[Inference] Run model on test splits
    ↓
Predictions (BirdNET_CombinedTable.csv)
    ↓
[Evaluate] Compute metrics (file-level scoring)
    ↓
Metrics (metrics_summary.csv, experiment_summary.json)
    ↓
[Collect] Aggregate all experiments
    ↓
all_experiments.csv
    ↓
[UI] Interactive analysis & visualization
```

---

## Related Documentation

- **[DATA_SPLITS.md](DATA_SPLITS.md)** — Detailed breakdown of train/val/test splits
- **[EVALUATION_PIPELINE.md](EVALUATION_PIPELINE.md)** — How evaluation works (scoring, metrics, known bugs)
- **[DATA_MODEL.md](DATA_MODEL.md)** — Schema for CSV outputs and UI state
- **[SWEEPS.md](SWEEPS.md)** — How to define and run hyperparameter sweeps
- **[UI_ARCHITECTURE.md](UI_ARCHITECTURE.md)** — Streamlit app components and flow
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** — Cheat sheet for common tasks

---

## Troubleshooting

### Experiment shows perfect metrics (1.0/1.0/1.0)
- **Likely cause:** Evaluation bug (missing test files)
- **Check:** Does `TP + FP + TN + FN` equal expected file count?
- **See:** [EVALUATION_PIPELINE.md#known-issues](EVALUATION_PIPELINE.md#known-issues--bugs)

### Model won't train (CUDA errors)
- Check GPU availability: `nvidia-smi`
- Reduce `training.batch_size` in config
- Ensure TensorFlow GPU version installed

### Inference hangs or crashes
- Check BirdNET analyzer version compatibility
- Verify audio files are valid WAV format
- Try smaller batch size or single-threaded mode

### Results not appearing in UI
- Run `collect_experiments.py` to update `all_experiments.csv`
- Check experiment has `evaluation/experiment_summary.json`
- Verify experiment name matches naming convention

---

## Design Philosophy

1. **Reproducibility:** Every experiment has config YAML + seed → deterministic results
2. **Modularity:** Each pipeline stage is independent, can be run/debugged separately
3. **Disk-Friendly:** Use symlinks instead of copying audio files
4. **Fail-Fast:** Validate configs before starting long-running training
5. **Transparency:** All intermediate outputs saved (training logs, predictions, metrics)
6. **Automation:** Sweeps handle tedious parameter grids automatically

---

## Future Development Notes

### Potential Enhancements
- Add per-class metrics (quality, call type breakdowns)
- Generate ROC/PR curves for each experiment
- Implement confidence calibration analysis
- Support multi-target species (not just RADR)
- Add automated model selection (best config finder)
- Visualize detection score distributions
