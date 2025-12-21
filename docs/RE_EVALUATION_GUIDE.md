# Re-Evaluation Guide

## ✅ STATUS: COMPLETED (2025-12-21)

**All 485 experiments (stage 6+) were successfully re-evaluated on 2025-12-21.**

This document remains as reference for understanding the bug fix and how to run re-evaluations in the future.

---

## Overview

On 2025-12-21, a critical bug was fixed in the evaluation pipeline. The bug caused files with no BirdNET detections to be excluded from evaluation entirely, leading to artificially inflated metrics.

---

## What Was Fixed

### The Bug
- BirdNET analyzer only outputs CSV rows for files where it detects species
- Files with no detections → no rows in `BirdNET_CombinedTable.csv`
- Old evaluation code only processed files in BirdNET output
- **Result:** Missing files never evaluated, causing incomplete metrics

### The Fix  
- Evaluation now loads complete file list from `AudioData/splits/{split}/` directories
- Files missing from BirdNET output are assigned score=0.0
- **Result:** ALL files evaluated, metrics are accurate

### Impact Example (stage14_013)

**Before Fix (WRONG):**
```
Files evaluated: 888/3,585 (24.7%)
At threshold 0.05:
  TP=873, FP=0, FN=15, TN=0
  Precision=1.0, Recall=0.984, F1=0.992
```

**After Fix (CORRECT):**
```
Files evaluated: 3,585/3,585 (100%)
At threshold 0.05:
  TP=873, FP=0, FN=818, TN=1,894
  Precision=1.0, Recall=0.516, F1=0.681
```

The old evaluation missed 2,697 files (803 positives + 1,894 negatives), making recall appear much higher than reality.

---

## Re-Evaluation Instructions

### Using rerun_all_evaluations.py (Batch Processing)

For re-evaluating many experiments at once:

```bash
# Re-evaluate ALL experiments
python scripts/rerun_all_evaluations.py

# Re-evaluate specific stages
python scripts/rerun_all_evaluations.py --stages stage11 stage12 stage14

# Skip backing up old results
python scripts/rerun_all_evaluations.py --no-backup

# Custom log file location
python scripts/rerun_all_evaluations.py --log-file results/my_log.txt
```

**Features:**
- Processes all experiments matching pattern
- Tracks before/after metrics changes (ΔF1 > 0.01)
- Generates detailed log file showing which experiments changed
- Automatically backs up old evaluations to `evaluation_backup_old/`
- Continues processing even if individual experiments fail

### Step 2: Analyze Changes (Optional)

Check which experiments had significant metric changes:

```bash
python scripts/analyze_evaluation_changes.py
```

This generates `results/evaluation_changes_by_stage.txt` showing:
- Which stages had metric changes
- Detailed before/after metrics for each changed experiment
- Summary by stage to help update reports

### Step 3: Update Master CSV

After re-evaluating, update the aggregated results:

```bash
python -m birdnet_custom_classifier_suite.pipeline.collect_experiments
```

This updates `all_experiments.csv` with the corrected metrics.

### Step 3: Verify Results

Check that metrics make sense:

```bash
# Check experiment summary
cat experiments/stage14_013/evaluation/experiment_summary.json

# Check metrics across thresholds
head -20 experiments/stage14_013/evaluation/metrics_summary.csv
```

**Sanity checks:**
- ✅ TP + FP + TN + FN = expected file count (3,585 for test_ood, 2,943 for test_iid)
- ✅ TN > 0 (unless model is broken and never identifies negatives)
- ✅ Precision < 1.0 in most cases (perfect precision is rare)
- ❌ TN = 0 at all thresholds → model never detects negatives (bad model)

---

## Interpreting Re-Evaluated Results

### Good Models

Example: stage11_015 at best threshold (0.05)
```
TP=979, FP=18, FN=712, TN=1,876
Total: 3,585 ✓
Precision=0.982, Recall=0.579, F1=0.728
```

- Has True Negatives (TN=1,876) — correctly identifies negative files
- High Precision (0.982) — when it says RADR, it's almost always right
- Moderate Recall (0.579) — finds about 58% of RADRs
- Balanced F1 (0.728) — good trade-off

### Conservative Models

Example: stage14_013 at best threshold (0.05)
```
TP=873, FP=0, FN=818, TN=1,894
Total: 3,585 ✓
Precision=1.0, Recall=0.516, F1=0.681
```

- Has True Negatives (TN=1,894) — correctly avoids false alarms
- Perfect Precision (1.0) — never wrong when it says RADR
- Low Recall (0.516) — only finds about half the RADRs
- This is a conservative model — prefers to miss some positives rather than risk false alarms

### Broken Models

Example: Hypothetical bad model
```
TP=1,691, FP=1,894, FN=0, TN=0 at ALL thresholds
Total: 3,585 ✓
Precision=0.472, Recall=1.0
```

- **Zero True Negatives** — never correctly identifies a negative file
- Low Precision (0.472) — less than half its predictions are correct
- Perfect Recall (1.0) — detects all positives (by predicting everything as RADR)
- **This model is broken** — it predicts RADR for every file
- Needs retraining or different hyperparameters

---

## Comparison: Before vs After Fix

### Metrics That Increase After Fix
- **False Negatives (FN):** More positives now correctly counted as missed
- **True Negatives (TN):** Negative files with no detection now counted

### Metrics That May Decrease After Fix
- **Recall:** Denominator increases (more FN), lowering recall
- **F1 Score:** Usually decreases as recall drops
- **Accuracy:** May decrease if model missed many positives

### Metrics That Should Stay Similar
- **Precision:** Usually stays similar or increases (more conservative threshold selection)
- **True Positives (TP):** Count of detected positives doesn't change much

---

## Common Issues

### Issue: "WARNING: X files missing from BirdNET output"

**Cause:** BirdNET didn't output predictions for some files (no detections).

**Solution:** This is normal and now handled correctly. These files get score=0.0 and are properly evaluated.

**Note:** If MANY files are missing (>50%), the model may be too conservative or broken.

### Issue: All metrics are identical after re-evaluation

**Cause:** The experiment already had all files in BirdNET output (good model).

**Solution:** No action needed. Re-evaluation confirms metrics were already correct.

### Issue: TN=0 at all thresholds

**Cause:** Model never outputs low enough scores to correctly identify negatives.

**Solution:** This is a model quality issue. Consider:
- Different hyperparameters (less regularization, different loss function)
- Better negative examples in training data
- Retraining with adjusted class weights

---

## Bulk Re-Evaluation

To re-evaluate all experiments in a stage:

```bash
# List all experiments first
ls experiments/stage14_*

# Re-evaluate all (with backup)
python scripts/rerun_evaluation.py stage14_*

# Verify count
ls experiments/stage14_*/evaluation/experiment_summary.json | wc -l

# Update master CSV
python -m birdnet_custom_classifier_suite.pipeline.collect_experiments
```

---

## Related Documentation
- [EVALUATION_PIPELINE.md](EVALUATION_PIPELINE.md) — Detailed evaluation logic
- [PIPELINE_OVERVIEW.md](PIPELINE_OVERVIEW.md) — Complete pipeline overview
- [DATA_SPLITS.md](DATA_SPLITS.md) — Test set composition
