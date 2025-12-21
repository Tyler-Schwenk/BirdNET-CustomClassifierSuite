# Evaluation Pipeline Reference

## Overview

The evaluation pipeline measures model performance on held-out test sets by:
1. Running BirdNET analyzer on test audio to generate predictions
2. Computing per-file RADR confidence scores (max across all detections)
3. Sweeping classification thresholds to find optimal F1
4. Saving metrics for both IID and OOD test sets

**Key Philosophy:** Evaluation happens at the **file level**, not detection level. Each file gets a single score representing maximum RADR confidence detected anywhere in that file.

---

## Pipeline Flow

### 1. Inference Phase (`pipeline.py`)

After model training, the pipeline runs BirdNET analyzer on test splits:

```python
# For each test split (test_iid, test_ood):
audio_path = AudioData/splits/test_iid/  # or test_ood
output_path = experiments/{exp_name}/inference/test_iid/

birdnet_cmd = [
    "python", "external/BirdNET-Analyzer/analyze.py",
    "--i", audio_path,
    "--o", output_path,
    "--classifier", "experiments/{exp_name}/checkpoints/model.tflite",
    "--min_conf", "0.0",  # Critical: accept ALL detections
    "--rtype", "csv",
    "--threads", "4"
]
```

**Output:** `experiments/{exp_name}/inference/{split}/BirdNET_CombinedTable.csv`

This CSV contains all detections with confidence > 0.0:
```
Selection,View,Channel,Begin File,Begin Time (s),End Time (s),Low Freq (Hz),High Freq (Hz),Species Code,Scientific name,Common name,Confidence
1,Spectrogram 1,0,positive_high_Flight_Moth11_20230515_120345.wav,3.0,6.0,1500,5000,RADR,Rana draytonii,California Red-legged Frog,0.87
```

### 2. Load Results (`evaluate_results.py:load_results()`)

```python
def load_results(folder: Path) -> pd.DataFrame:
    """Load BirdNET predictions from CombinedTable.csv"""
    csv_path = folder / "BirdNET_CombinedTable.csv"
    df = pd.read_csv(csv_path)
    
    # Extract label and quality from filename
    df["label"] = df["Begin File"].apply(parse_label)
    df["quality"] = df["Begin File"].apply(parse_quality)
    
    return df
```

**Critical Issue:** This function ONLY reads files that appear in BirdNET output. If BirdNET doesn't detect anything in a file (or detects only non-RADR species below confidence threshold), that file will be **missing** from the DataFrame.

### 3. Per-File Scoring (`evaluate_results.py:evaluate()`)

```python
def evaluate(df: pd.DataFrame, split_name: str, outdir: Path) -> pd.DataFrame:
    """
    Evaluate at FILE level using max RADR confidence per file.
    
    For each audio file:
    - Find max confidence among all RADR detections (if any)
    - If no RADR detected, score = 0.0
    - Sweep thresholds 0.0 to 1.0 in steps of 0.05
    - Compute precision/recall/F1 at each threshold
    """
    
    # Extract RADR detections and find max confidence per file
    radr = df[df["Scientific name"] == "RADR"].groupby("File")["Confidence"].max()
    
    # Get unique files with labels
    files = df[["File", "label"]].drop_duplicates()
    
    # Assign max RADR score (or 0.0 if no RADR detected)
    files["score"] = files["File"].map(radr)
    files["score"] = files["score"].fillna(0.0)  # <-- Handles files with no RADR
    
    # Threshold sweep
    results = []
    for threshold in np.arange(0.0, 1.01, 0.05):  # 21 thresholds
        y_true = (files["label"] == "positive").astype(int)
        y_pred = (files["score"] >= threshold).astype(int)
        
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        tn = ((y_true == 0) & (y_pred == 0)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        results.append({
            "split": split_name,
            "threshold": threshold,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })
    
    return pd.DataFrame(results)
```

**Scoring Logic:**
- **Max RADR confidence** across all detections in a file
- **No minimum confidence filter** — even 0.01 counts if it's the max RADR
- **Threshold applied post-hoc** during metric computation
- **Missing RADR → score 0.0** via `fillna(0.0)`

### 4. Run Evaluation (`evaluate_results.py:run_evaluation()`)

```python
def run_evaluation(exp_dir: str) -> Path:
    """Run evaluation on both test splits"""
    exp_path = Path(exp_dir)
    eval_dir = exp_path / "evaluation"
    eval_dir.mkdir(exist_ok=True)
    
    all_metrics = []
    
    for split in ["test_iid", "test_ood"]:
        inference_folder = exp_path / "inference" / split
        
        if not (inference_folder / "BirdNET_CombinedTable.csv").exists():
            print(f"⚠️ No results for {split}, skipping")
            continue
        
        # Load predictions and evaluate
        df = load_results(inference_folder)
        metrics = evaluate(df, split, eval_dir)
        all_metrics.append(metrics)
    
    # Concatenate metrics from both splits
    combined = pd.concat(all_metrics, ignore_index=True)
    combined.to_csv(eval_dir / "metrics_summary.csv", index=False)
    
    # Find best threshold per split (max F1)
    best_thresholds = combined.loc[combined.groupby("split")["f1"].idxmax()]
    
    return eval_dir
```

### 5. Experiment Summary (`evaluate_results.py:make_experiment_summary()`)

Creates `experiment_summary.json` with:
```json
{
  "experiment_name": "stage14_013",
  "test_iid": {
    "best_threshold": 0.25,
    "best_f1": 0.89,
    "best_precision": 0.91,
    "best_recall": 0.87,
    "tp": 1706, "fp": 88, "fn": 258, "tn": 891
  },
  "test_ood": {
    "best_threshold": 0.0,
    "best_f1": 1.0,
    "best_precision": 1.0,
    "best_recall": 1.0,
    "tp": 888, "fp": 0, "fn": 0, "tn": 0
  },
  "validation_package": {
    "positive_count": 647,
    "negative_count": 2602,
    "total_count": 3249
  }
}
```

---

## Known Issues & Bugs

### ✅ FIXED (2025-12-21): Missing File Evaluation Bug

**Problem:** BirdNET analyzer does not output rows for files where it detects no species, causing those files to be excluded from evaluation entirely.

**Root Cause:**
- BirdNET only writes CSV rows for files with detections above its internal threshold
- Files with no detections → no rows in `BirdNET_CombinedTable.csv`
- Old evaluation code only loaded files present in BirdNET output
- Missing files never received their correct score of 0.0

**Real-World Impact Example (stage14_013 before fix):**
```
BirdNET Output:
  - 888 files with RADR detections (all positive files)
  - 2,697 files missing (803 positive + 1,894 negative)
  
Old Evaluation (WRONG):
  - Only evaluated 888/3585 files (24.7%)
  - At threshold 0.05: TP=873, FP=0, FN=15, TN=0
  - Metrics: Precision=1.0, Recall=0.984, F1=0.992
  - ❌ Completely ignored 2,697 files including ALL negatives!
  
New Evaluation (CORRECT):
  - Evaluates all 3,585/3,585 files (100%)
  - Missing files assigned score=0.0
  - At threshold 0.05: TP=873, FP=0, FN=818, TN=1,894
  - Metrics: Precision=1.0, Recall=0.516, F1=0.681
  - ✓ Accurate metrics reflecting true performance
```

**The Fix:**
Modified `load_results()` in [evaluate_results.py](evaluate_results.py):
1. Loads BirdNET predictions (may be incomplete)
2. Loads complete file list from `AudioData/splits/{split}/` directories
3. Parses true labels from subdirectory names (positive/ or negative/)
4. Adds synthetic rows for files missing from BirdNET output
5. Assigns score=0.0 to missing files (no RADR detected)
6. Ensures all files evaluated with correct ground truth labels

**Impact:**
- ✅ All test files now evaluated correctly  
- ✅ True Negatives now counted (models that detect nothing on negatives get TN credit)
- ✅ False Negatives now counted (positives with no detection are correctly counted as missed)
- ✅ Metrics accurately reflect model performance
- ✅ Can identify models with detection issues (too conservative, missing many positives)

---

## Threshold Selection Strategy

The pipeline tests **21 thresholds** from 0.0 to 1.0 (step 0.05).

**Best threshold selection:**
- For each split (IID/OOD), find threshold with **maximum F1 score**
- This is the "best" operating point reported in `experiment_summary.json`
- Different splits may have different best thresholds

**Why threshold sweep?**
- Models produce raw confidence scores (0.0–1.0)
- Need to convert to binary prediction: positive if score ≥ threshold
- Optimal threshold balances precision/recall trade-off
- Different use cases may prefer different thresholds:
  - **High recall:** Lower threshold (e.g., 0.1) — catch all positives, accept more FP
  - **High precision:** Higher threshold (e.g., 0.6) — only predict positive when very confident

---

## Metrics Definitions

All metrics computed at **file level**:

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **True Positives (TP)** | Positive file, score ≥ threshold | Correctly identified RADR calls |
| **False Positives (FP)** | Negative file, score ≥ threshold | Incorrectly predicted RADR |
| **True Negatives (TN)** | Negative file, score < threshold | Correctly identified no RADR |
| **False Negatives (FN)** | Positive file, score < threshold | Missed RADR calls |
| **Precision** | TP / (TP + FP) | When model says "RADR", how often is it right? |
| **Recall** | TP / (TP + FN) | Of all RADR files, how many did we find? |
| **F1 Score** | 2 × (Precision × Recall) / (Precision + Recall) | Harmonic mean of precision/recall |

---

## Output Files

Each experiment generates:

### `evaluation/metrics_summary.csv`
All thresholds for both splits:
```csv
split,threshold,tp,fp,fn,tn,precision,recall,f1
test_iid,0.0,1964,979,0,0,0.667,1.0,0.8
test_iid,0.05,1958,512,6,467,0.793,0.997,0.883
...
test_ood,0.0,888,0,0,0,1.0,1.0,1.0
test_ood,0.05,880,0,8,0,1.0,0.991,0.995
...
```

### `evaluation/experiment_summary.json`
Best metrics per split:
```json
{
  "experiment_name": "stage14_013",
  "test_iid": {
    "best_threshold": 0.25,
    "best_f1": 0.89,
    ...
  },
  "test_ood": {
    "best_threshold": 0.0,
    "best_f1": 1.0,
    ...
  }
}
```

### `all_experiments.csv` (project root)
Aggregated results across all experiments, created by `collect_experiments.py`.

---

## Interpreting Results

### Red Flags
- ✅ **Perfect metrics (1.0/1.0/1.0)** — Likely evaluation bug (missing files)
- ✅ **Zero TN or FP** — No negative files evaluated
- ✅ **TP + FP + TN + FN ≠ expected file count** — Missing files from evaluation
- ❌ **IID metrics much higher than OOD** — Model overfitting to training sites
- ❌ **Best threshold = 0.0** — Model scores are poorly calibrated

### Expected Behavior
- **OOD F1:** 0.75–0.85 for good models
- **IID F1:** 0.80–0.90 (slightly higher than OOD is normal)
- **Best threshold:** 0.15–0.35 (model outputs well-calibrated probabilities)
- **TP + FP + TN + FN = total files** (2,943 for IID, 3,585 for OOD)

### High Variance Across Seeds
If F1 varies widely (e.g., 0.756 to 1.0) across random seeds:
- Check if test set is too small or imbalanced
- Verify splits are consistent across experiments
- Consider more training data or stronger regularization

---

## Integration with Other Components

### Training Phase
1. `make_training_package.py` creates filtered subset of train/val data
2. `pipeline.py` runs BirdNET model training with TensorFlow
3. Model saved to `experiments/{name}/checkpoints/model.tflite`

### Validation Package
- Created during training to track which files were used
- `validation_manifest.csv` tracks file→split mapping
- **Note:** Currently all marked as 'val' split (may need fixing)

### Sweeps
- `sweeps/` module runs multiple experiments with different hyperparameters
- Each experiment evaluated independently
- Results aggregated by `collect_experiments.py`

### Result Collection
- `collect_experiments.py` scans all `experiments/*/evaluation/`
- Merges `experiment_summary.json` into `all_experiments.csv`
- Adds experiment metadata (stage, config, seed, etc.)

---

## Future Improvements

1. **Fix missing file bug** — Always evaluate all files in test directory
2. **Add per-class metrics** — Separate scores for quality (high/med/low) and call type (Flight/Song/GBWO)
3. **Confidence calibration** — Plot reliability diagrams to check if scores represent true probabilities
4. **ROC/PR curves** — Visualize full threshold trade-off
5. **Stratified metrics** — Report metrics separately for Moth11 vs Moth12 (within OOD)

---

## Related Documentation
- [DATA_SPLITS.md](DATA_SPLITS.md) — Test set composition and sources
- [DATA_MODEL.md](DATA_MODEL.md) — Schema for results CSV
- [SWEEPS.md](SWEEPS.md) — How to run hyperparameter sweeps
