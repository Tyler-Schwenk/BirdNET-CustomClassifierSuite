# Data Composition Sweeps

## Overview

This document describes how to create sweeps that vary the **composition of training data subsets** rather than just hyperparameters. This is essential for controlled experiments comparing different data curation strategies, such as:

- Including curated high-quality low-data samples
- Adding hard-negative subsets with varying confidence thresholds  
- Comparing baseline datasets to augmented versions

## Problem Statement

You have:
1. **Curated positive subsets** (in `AudioData/curated/bestLowQuality/`):
   - `small/` (51 files, 5% of low-quality data)
   - `medium/` (154 files, 15%)
   - `large/` (309 files, 30%)
   - `top50/` (515 files, 50%)

2. **Curated hard-negative subsets** (in `AudioData/curated/hardNeg/`):
   - `hardneg_conf_min_50/` (1,401 files)
   - `hardnet_conf_min_85/` (981 files)
   - `hardneg_conf_min_99/` (475 files)

3. **Baseline quality filters** (from manifest):
   - `quality: [high, medium]` - standard baseline
   - `quality: [high, medium, low]` - full dataset

You want to run experiments like:
- **Baseline:** high + medium quality only
- **+BestLow (small):** high + medium + curated bestLowQuality/small
- **+BestLow (medium):** high + medium + curated bestLowQuality/medium
- **+HardNeg (conf50):** high + medium + hardneg_conf_min_50 added to negatives
- **+BestLow (large) +HardNeg (conf85):** high + medium + bestLowQuality/large + hardneg_conf_min_85

## Solution Architecture

### 1. Config Schema Extension

We extend the `training_package` section to support **explicit subset inclusion**:

```yaml
training_package:
  # Standard filters (apply to manifest rows)
  include_negatives: true
  balance: true
  quality:
    - high
    - medium
  
  # NEW: Additional positive subsets (folder-based, not in manifest)
  positive_subsets:
    - AudioData/curated/bestLowQuality/small
    - AudioData/curated/bestLowQuality/medium
  
  # NEW: Additional negative subsets (folder-based)
  negative_subsets:
    - AudioData/curated/hardNeg/hardneg_conf_min_50
```

**Key design decisions:**
- `quality` filter still applies to manifest-based files (your core high/medium/low splits)
- `positive_subsets` and `negative_subsets` are **additive** - they provide additional files beyond the manifest
- Subsets are specified as folder paths relative to project root
- Files from subsets bypass manifest filtering entirely (they're direct additions)

### 2. Implementation in `make_training_package.py`

**Current flow:**
1. Load manifest
2. Filter by quality, site, recorder_id, split
3. Sample positives and negatives
4. Copy to training_package/

**New flow:**
1. Load manifest
2. Filter by quality, site, recorder_id, split (**manifest-based pool**)
3. **Load additional files from `positive_subsets` folders**
4. **Load additional files from `negative_subsets` folders**
5. Merge: `final_pos = manifest_pos + subset_pos`, `final_neg = manifest_neg + subset_neg`
6. Sample from merged pools
7. Copy to training_package/

**Code changes needed:**

```python
def load_subset_files(subset_paths: List[str], audio_root: Path) -> pd.DataFrame:
    """
    Load audio files from specified subset folders.
    Returns DataFrame with columns: resolved_path, label, source_subset
    """
    rows = []
    for subset_str in subset_paths:
        subset_dir = audio_root / subset_str
        if not subset_dir.exists():
            logging.warning(f"Subset directory not found: {subset_dir}")
            continue
        
        audio_files = list(subset_dir.glob("*.wav")) + list(subset_dir.glob("*.mp3"))
        for f in audio_files:
            rows.append({
                "resolved_path": str(f.resolve()),
                "label": "positive" if "positive" in subset_str or "bestLow" in subset_str else "negative",
                "quality": "curated_subset",  # Mark as non-manifest
                "call_type": "curated_subset",
                "site": "curated",
                "recorder_id": "subset",
                "date": "",
                "split": "train",
                "filename": f.name,
                "source_subset": subset_str
            })
    
    return pd.DataFrame(rows)

def filter_rows(df: pd.DataFrame, args: dict, audio_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extended to merge manifest-based filtering with subset loading."""
    
    # EXISTING LOGIC for manifest-based files
    present = df[df["split"].str.lower().isin([s.lower() for s in args.get("splits", ["train"])])].copy()
    # ... existing quality, site, recorder_id filtering ...
    pos = filtered[filtered["label"].str.lower() == "positive"].copy()
    neg = filtered[filtered["label"].str.lower() == "negative"].copy()
    
    if args.get("quality"):
        pos = pos[pos["quality"].str.lower().isin([q.lower() for q in args["quality"]])]
    
    # NEW LOGIC: add subset files
    pos_subset_paths = args.get("positive_subsets", [])
    neg_subset_paths = args.get("negative_subsets", [])
    
    if pos_subset_paths:
        pos_subsets = load_subset_files(pos_subset_paths, audio_root)
        pos_subsets = pos_subsets[pos_subsets["label"] == "positive"]
        pos = pd.concat([pos, pos_subsets], ignore_index=True)
        logging.info(f"Added {len(pos_subsets)} files from positive_subsets")
    
    if neg_subset_paths:
        neg_subsets = load_subset_files(neg_subset_paths, audio_root)
        neg_subsets = neg_subsets[neg_subsets["label"] == "negative"]
        neg = pd.concat([neg, neg_subsets], ignore_index=True)
        logging.info(f"Added {len(neg_subsets)} files from negative_subsets")
    
    return pos, neg
```

### 3. Sweep Spec Schema

Add new axes to sweep specs:

```yaml
stage: 8
out_dir: config/sweeps/stage8_data_composition

axes:
  seed: [123, 456, 789]
  
  # Base quality (manifest-based)
  quality:
    - [high, medium]  # baseline
  
  # Positive subset combinations
  positive_subsets:
    - []  # baseline: no subsets
    - [AudioData/curated/bestLowQuality/small]
    - [AudioData/curated/bestLowQuality/medium]
    - [AudioData/curated/bestLowQuality/large]
    - [AudioData/curated/bestLowQuality/top50]
  
  # Negative subset combinations
  negative_subsets:
    - []  # baseline: no hard negs
    - [AudioData/curated/hardNeg/hardneg_conf_min_50]
    - [AudioData/curated/hardNeg/hardnet_conf_min_85]
    - [AudioData/curated/hardNeg/hardneg_conf_min_99]
  
  balance: [true]

base_params:
  epochs: 50
  batch_size: 32
  learning_rate: 0.0005
  dropout: 0.25
  hidden_units: 512
  mixup: true
  label_smoothing: true
  focal-loss: false
  upsampling_mode: linear
  upsampling_ratio: 1.0
  fmin: 0
  fmax: 15000
  overlap: 0.0
```

This generates experiments for:
- 3 seeds × 1 quality combo × 5 positive_subset options × 4 negative_subset options = **60 experiments**

### 4. Sweep Generator Updates

The sweep generator (`birdnet_custom_classifier_suite/sweeps/sweep_generator.py`) needs to:

1. **Recognize new axes** (`positive_subsets`, `negative_subsets`)
2. **Write them into experiment configs** under `training_package:`

**Example generated config (`stage8_001.yaml`):**

```yaml
experiment:
  name: stage8_001
  seed: 123

training_package:
  include_negatives: true
  balance: true
  quality:
    - high
    - medium
  positive_subsets: []
  negative_subsets: []

training:
  epochs: 50
  batch_size: 32

training_args:
  # ... full resolved args ...
```

**Example with subsets (`stage8_015.yaml`):**

```yaml
experiment:
  name: stage8_015
  seed: 123

training_package:
  include_negatives: true
  balance: true
  quality:
    - high
    - medium
  positive_subsets:
    - AudioData/curated/bestLowQuality/medium
  negative_subsets:
    - AudioData/curated/hardNeg/hardneg_conf_min_50

training:
  epochs: 50
  batch_size: 32

training_args:
  # ... full resolved args ...
```

### 5. Naming and Tracking

To make data composition experiments traceable:

1. **Config signature**: The signature includes subset info:
   ```
   quality=[high,medium]_pos_subsets=[small]_neg_subsets=[]
   ```

2. **Data summary report**: `training_package/data_summary.csv` gets new rows:
   ```csv
   stage,label,group_by,group,count
   available,positive,quality,high,500
   available,positive,quality,medium,300
   available,positive,source_subset,curated/bestLowQuality/small,51
   selected,positive,quality,high,500
   selected,positive,quality,medium,300
   selected,positive,source_subset,curated/bestLowQuality/small,51
   ```

3. **Experiment name includes data signature** (optional):
   ```
   stage8_001_baseline
   stage8_015_medLow_hardNeg50
   ```

## Usage Examples

### Example 1: Baseline + All BestLow Variants

```yaml
axes:
  seed: [123]
  quality: [[high, medium]]
  positive_subsets:
    - []
    - [AudioData/curated/bestLowQuality/small]
    - [AudioData/curated/bestLowQuality/medium]
    - [AudioData/curated/bestLowQuality/large]
    - [AudioData/curated/bestLowQuality/top50]
  negative_subsets: [[]]
  balance: [true]
```

**Result:** 5 experiments comparing baseline vs. 4 levels of curated low-quality data inclusion.

### Example 2: Cross BestLow × HardNeg

```yaml
axes:
  seed: [123]
  quality: [[high, medium]]
  positive_subsets:
    - []
    - [AudioData/curated/bestLowQuality/medium]
  negative_subsets:
    - []
    - [AudioData/curated/hardNeg/hardneg_conf_min_85]
  balance: [true]
```

**Result:** 2×2 = 4 experiments:
1. Baseline
2. Baseline + HardNeg85
3. BestLow(medium)
4. BestLow(medium) + HardNeg85

### Example 3: Full Factorial Design

```yaml
axes:
  seed: [123, 456, 789]
  quality: [[high, medium]]
  positive_subsets:
    - []
    - [AudioData/curated/bestLowQuality/small]
    - [AudioData/curated/bestLowQuality/large]
  negative_subsets:
    - []
    - [AudioData/curated/hardNeg/hardneg_conf_min_50]
    - [AudioData/curated/hardNeg/hardneg_conf_min_99]
  balance: [true]
```

**Result:** 3 seeds × 3 pos × 3 neg = **27 experiments**.

## Scientific Best Practices

1. **Control group**: Always include a baseline with no subsets (`positive_subsets: []`, `negative_subsets: []`)

2. **Ablation studies**: Test subsets independently before combining:
   - First run: vary `positive_subsets`, fix `negative_subsets: []`
   - Second run: vary `negative_subsets`, fix `positive_subsets: []`
   - Third run: combine best-performing subsets

3. **Multiple seeds**: Use at least 3 seeds for statistical significance

4. **Document subset creation**: Keep notes on how each subset was created:
   ```
   AudioData/curated/bestLowQuality/small:
     - Source: stage4_038 predictions on splits/train/positive/low
     - Selection: Top 5% by RADR confidence
     - Count: 51 files
     - Created: 2025-11-09
   ```

5. **Track subset versions**: If you re-curate, create timestamped folders:
   ```
   AudioData/curated/bestLowQuality_v2_20251109/small/
   ```

## Migration Path

1. **Phase 1** (immediate): Add `load_subset_files()` and extend `filter_rows()` in `make_training_package.py`
2. **Phase 2**: Update sweep generator to handle new axes
3. **Phase 3**: Create stage8 sweep spec with your current subsets
4. **Phase 4**: Run pilot experiments (3-5 configs) to validate
5. **Phase 5**: Run full sweep

## Backward Compatibility

- Existing sweeps work unchanged (they don't specify `positive_subsets`/`negative_subsets`)
- Old experiment configs remain valid
- New fields are optional with sensible defaults (`[]`)

## Future Extensions

1. **Multi-source subsets**: Combine subsets from different curation runs
2. **Weighted sampling**: Sample different proportions from manifest vs. subsets
3. **Subset metadata**: Track provenance (which model/experiment created each subset)
4. **Dynamic subset discovery**: Auto-detect subdirectories in `curated/`

## Summary

This design:
- ✅ Maintains scientific integrity (clear baselines, controlled comparisons)
- ✅ Is modular and future-proof (easy to add new subsets)
- ✅ Follows existing patterns (extends `training_package`, uses sweep generator)
- ✅ Provides clear tracking (data_summary.csv, config signatures)
- ✅ Backward compatible (existing code/configs unaffected)
