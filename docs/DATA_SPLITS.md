# Data Splits Reference

## Overview

Audio files are organized into **four evaluation splits** to test model performance under different conditions. All splits are stored in `AudioData/splits/`.

## Split Definitions

### Train (12,110 files)
- **Positives:** 3,585
- **Negatives:** 8,525
- **Purpose:** Model training data
- **Sources:** Cole Creek, Rancho Meling, Wheatley Pond/Stream sites
- **Directory:** `AudioData/splits/train/`

### Val (3,249 files)
- **Positives:** 647
- **Negatives:** 2,602
- **Purpose:** Validation during training (if `use_validation=True`)
- **Sources:** Same sites as training, different files
- **Directory:** `AudioData/splits/val/`

### Test-IID (2,943 files)
- **Positives:** 1,964
- **Negatives:** 979
- **Purpose:** In-Distribution test — same sites/conditions as training
- **Sources:** Cole Creek, Rancho Meling, Wheatley Pond/Stream sites
- **Directory:** `AudioData/splits/test_iid/`

### Test-OOD (3,585 files) ⭐
- **Positives:** 1,691
- **Negatives:** 1,894
- **Purpose:** Out-of-Distribution test — completely held-out site
- **Sources:** **ONLY Sylvan Pond (Moth11 + Moth12)** — never seen during training
- **Directory:** `AudioData/splits/test_ood/`
- **Note:** This is the PRIMARY metric for real-world generalization

## Composition Breakdown

### By Quality (Positives Only)
| Split | High | Medium | Low |
|-------|------|--------|-----|
| Train | 245 | 1,772 | 1,568 |
| Val | 2 | 101 | 544 |
| Test-IID | 270 | 473 | 1,221 |
| Test-OOD | 109 | 625 | 957 |

### By Call Type (Positives Only)
| Split | Flight | Song | GBWO |
|-------|--------|------|------|
| Train | 2,020 | 170 | 1,395 |
| Val | 285 | 25 | 337 |
| Test-IID | 1,325 | 58 | 581 |
| Test-OOD | 1,017 | 136 | 538 |

### By Site/Recorder

#### OOD (Sylvan Pond) — Moth11 + Moth12
- Positives: 1,691
- Negatives: 1,894
- **Fully reserved as the out-of-distribution (OOD) test set**
- Never used in training or validation

#### Cole Creek (Moth08)
- Train: 2,765 pos + 2,731 neg
- Val: 547 pos + 1,110 neg
- Test-IID: 406 pos + 4 neg
- **Total:** 3,718 pos + 3,845 neg
- Largest contributor, well-balanced across splits

#### Rancho Meling (Moth13)
- Train: 381 pos + 4,347 neg
- Val: 40 pos + 1,181 neg
- Test-IID: 1,523 pos + 38 neg
- **Total:** 1,944 pos + 5,566 neg
- Source site in Baja California, Mexico

#### Wheatley Pond/Stream (Moths 01–07, 09–10)
Smaller contributions ensuring device/site diversity:
- Moth01: 69 pos (train=31, val=31, test_iid=7)
- Moth02: 5 pos (train=2, val=2, test_iid=1)
- Moth03: 108 pos + 952 neg
- Moth04: 206 pos + 202 neg
- Moth06: 146 pos + 23 neg
- Moth07: 291 neg only
- Moth09: 539 neg only
- Moth10: 688 neg only

## Usage Guidelines

### During Training
1. **Train split** is used for model weights
2. **Val split** (optional) provides held-out validation metrics during training if `training.use_validation: true`
3. **Test splits are NEVER seen during training**

### During Evaluation
1. Models are run on **Test-IID** (familiar conditions) and **Test-OOD** (novel site)
2. Per-file predictions are generated via BirdNET analyzer
3. Metrics computed at 21 thresholds (0.0, 0.05, ..., 1.0)
4. **OOD metrics are the primary success indicator**

### Critical Rules
- ❌ **Never train on test splits**
- ❌ **Never merge OOD data with IID data** — OOD must remain completely held out
- ✅ **Always evaluate both IID and OOD** — IID shows model capacity, OOD shows generalization
- ✅ **Trust OOD metrics more than IID** — OOD reflects real-world deployment

## File Naming Convention

Files follow this pattern:
```
{positive|negative}_{quality}_{calltype}_{recorder}_{site}_{timestamp}.wav
```

Examples:
- `positive_high_Flight_Moth11_SylvanPond_20230515_120345.wav`
- `negative_Moth08_ColeCreek_20230612_093012.wav`

The evaluation pipeline uses these filename patterns to extract:
- **Label** (positive/negative)
- **Quality** (high/medium/low) — from positives only
- **Call type** (Flight/Song/GBWO) — from positives only

## Manifest Structure

The master manifest (`data/manifest.csv`) tracks all files with columns:
- `file_path`: Relative path from AudioData root
- `label`: "positive" or "negative"
- `split`: "train", "val", "test_iid", or "test_ood"
- `quality`: "high", "medium", "low" (positives only)
- `call_type`: "Flight", "Song", "GBWO" (positives only)
- `site`: Recording site name
- `recorder_id`: Moth device ID (Moth01-Moth13)
- `date`: Recording date
- `filename`: Base filename

## Related Documentation
- [EVALUATION_PIPELINE.md](EVALUATION_PIPELINE.md) — How evaluation uses these splits and computes metrics
- [DATA_MODEL.md](DATA_MODEL.md) — Schema for results CSV files
- [DATA_COMPOSITION_SWEEPS.md](DATA_COMPOSITION_SWEEPS.md) — How to test different data combinations
- [PIPELINE_OVERVIEW.md](PIPELINE_OVERVIEW.md) — Complete pipeline walkthrough
