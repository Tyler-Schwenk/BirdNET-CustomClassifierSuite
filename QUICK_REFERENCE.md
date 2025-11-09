# Data Composition Sweeps - Quick Reference

## Your Current Curated Subsets

### Positive (Curated Low-Quality)
Located in: `AudioData/curated/bestLowQuality/`

| Folder | Files | Description |
|--------|-------|-------------|
| `small` | 51 | Top 5% of low-quality data by RADR confidence |
| `medium` | 154 | Top 15% |
| `large` | 309 | Top 30% |
| `top50` | 515 | Top 50% |

Source: stage4_038 predictions on splits/train/positive/low

### Negative (Hard Negatives)
Located in: `AudioData/curated/hardNeg/`

| Folder | Files | Description |
|--------|-------|-------------|
| `hardneg_conf_min_50` | 1,401 | False positives with confidence ≥ 0.50 |
| `hardnet_conf_min_85` | 981 | False positives with confidence ≥ 0.85 |
| `hardneg_conf_min_99` | 475 | False positives with confidence ≥ 0.99 |

Source: stage3_046 predictions on new field data

## Using the UI (Streamlit)

### Create Data Composition Sweeps via UI

1. Start the UI:
   ```bash
   streamlit run birdnet_custom_classifier_suite/ui/app.py
   ```

2. Navigate to **Sweeps** tab

3. Fill in **Base Parameters** (epochs, batch_size, etc.)

4. Configure **Axes** (quality, balance, etc.)

5. Under **Data Composition Sweep Options**:
   
   **Option A: Use File Explorer (Recommended)**
   - Click **📁 Browse** button next to "Positive Subsets"
   - Navigate to `AudioData/curated/bestLowQuality/` and select a folder (e.g., `small`)
   - The path is validated and added to the list
   - Repeat for additional positive subset folders
   - Click **📁 Browse** button next to "Negative Subsets"
   - Navigate to `AudioData/curated/hardNeg/` and select a folder (e.g., `hardneg_conf_min_85`)
   - Click ❌ to remove any folders from the list
   
   **Option B: Type Paths Manually**
   - **Positive subsets**: Enter one combination per line
     ```
     curated/bestLowQuality/small
     curated/bestLowQuality/medium
     curated/bestLowQuality/large
     ```
   - **Negative subsets**: Enter one combination per line
     ```
     curated/hardNeg/hardneg_conf_min_85
     curated/hardNeg/hardneg_conf_min_99
     ```
   - Paths are validated automatically (green checkmark = valid, yellow warning = not found)

6. Click **Generate Sweep** - creates factorial configs

7. Results appear in `all_experiments.csv` with columns:
   - `dataset.filters.positive_subsets`
   - `dataset.filters.negative_subsets`

8. Use **Analysis** tab to filter by subset combinations and compare metrics

## Quick Start Commands (CLI)

### Generate Pilot Configs (4 experiments, 5 epochs each)
```bash
python -m birdnet_custom_classifier_suite.sweeps.sweep_generator \
  --spec config/sweep_specs/stage8_pilot.yaml
```

### Run One Test Experiment
```bash
python -m birdnet_custom_classifier_suite.pipeline.pipeline \
  --config config/sweeps/stage8_pilot/stage8_004.yaml \
  --verbose
```

### Generate Full Sweep (60 experiments)
```bash
python -m birdnet_custom_classifier_suite.sweeps.sweep_generator \
  --spec config/sweep_specs/stage8_data_composition.yaml
```

### Run Full Sweep
```bash
python -m birdnet_custom_classifier_suite.sweeps.run_sweep \
  config/sweeps/stage8_data_composition/ \
  --base-config config/sweeps/stage8_data_composition/base.yaml \
  --experiments-root experiments
```

### Validate Subset Loading (dry run)
```bash
python scripts/test_data_composition.py
```

## Example Config Snippets

### Baseline (no subsets)
```yaml
training_package:
  include_negatives: true
  balance: true
  quality: [high, medium]
  positive_subsets: []
  negative_subsets: []
```

### With Curated Low-Quality Positives
```yaml
training_package:
  include_negatives: true
  balance: true
  quality: [high, medium]
  positive_subsets:
    - curated/bestLowQuality/medium  # adds 154 files
  negative_subsets: []
```

### With Hard Negatives
```yaml
training_package:
  include_negatives: true
  balance: true
  quality: [high, medium]
  positive_subsets: []
  negative_subsets:
    - curated/hardNeg/hardnet_conf_min_85  # adds 981 files
```

### Combined
```yaml
training_package:
  include_negatives: true
  balance: true
  quality: [high, medium]
  positive_subsets:
    - curated/bestLowQuality/large  # adds 309 files
  negative_subsets:
    - curated/hardNeg/hardneg_conf_min_99  # adds 475 files
```

## Sweep Spec Template

```yaml
stage: X
out_dir: config/sweeps/stageX_name

axes:
  seed: [123, 456, 789]
  quality: [[high, medium]]
  
  positive_subsets:
    - []  # always include baseline
    - [curated/bestLowQuality/small]
    - [curated/bestLowQuality/medium]
    # ... more options
  
  negative_subsets:
    - []  # always include baseline
    - [curated/hardNeg/hardneg_conf_min_50]
    - [curated/hardNeg/hardnet_conf_min_85]
    # ... more options
  
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

## Checking Results

### View Training Package Composition
After running an experiment, check:
```
experiments/stage8_XXX/training_package/data_summary.csv
```

Look for rows where `group_by == "source_subset"`:
```csv
stage,label,group_by,group,count
selected,positive,source_subset,manifest,2017
selected,positive,source_subset,curated/bestLowQuality/small,51
selected,negative,source_subset,manifest,8525
selected,negative,source_subset,curated/hardNeg/hardnet_conf_min_85,981
```

### Analyze Experiment Results
Use the Streamlit UI or query `results/all_experiments.csv`:

```python
import pandas as pd
df = pd.read_csv("results/all_experiments.csv")

# Filter to stage8 experiments
stage8 = df[df["experiment.name"].str.startswith("stage8_")]

# Group by data composition (you'll need to extract from experiment names or configs)
# Compare metrics like ood_best_f1_f1, ood_best_f1_precision, ood_best_f1_recall
```

## Recommended Experiment Sequence

1. **Pilot validation** (stage8_pilot, 4 experiments):
   - Verify configs generate correctly
   - Check training package composition
   - Confirm experiments complete successfully

2. **Positive subset ablation** (12 experiments):
   - 3 seeds × 4 pos subsets ([], small, medium, large)
   - Fix negative_subsets=[]
   - Identify best positive subset size

3. **Negative subset ablation** (9 experiments):
   - 3 seeds × 3 neg subsets ([], conf50, conf85)
   - Fix positive_subsets=[]
   - Identify best hard-negative threshold

4. **Full factorial** (60 experiments):
   - 3 seeds × 5 pos × 4 neg
   - Test all combinations
   - Look for interaction effects

## Troubleshooting

### Subset directory not found
**Error:** `WARNING: Subset directory not found (skipping): .../curated/...`

**Fix:** Check folder name typos. Your folders use `hardnet_conf_min_85` (not `hardneg`).

### Empty DataFrame error
**Error:** `KeyError: 'label'`

**Cause:** Bug when subset folder is empty or missing (fixed in implementation)

**Verify:** Ensure subset folders contain .wav/.mp3/.flac files

### Configs not generated
**Error:** No YAML files in `config/sweeps/stageX/`

**Fix:** Check spec file YAML syntax:
```bash
python -c "import yaml; yaml.safe_load(open('config/sweep_specs/stage8_pilot.yaml'))"
```

### Training package fails
**Error:** During make_training_package phase

**Debug:**
1. Check manifest path is correct
2. Verify audio_root points to AudioData/
3. Run test script: `python scripts/test_data_composition.py`
4. Check logs in experiment output

## Adding New Subsets

1. **Create folder:**
   ```bash
   mkdir -p AudioData/curated/myNewSubset
   ```

2. **Add audio files:**
   Copy/link your curated files there

3. **Update sweep spec:**
   ```yaml
   positive_subsets:
     - [curated/myNewSubset]
   ```

4. **Regenerate configs:**
   ```bash
   python -m birdnet_custom_classifier_suite.sweeps.sweep_generator \
     --spec config/sweep_specs/stage8_data_composition.yaml
   ```

---

**For full documentation, see:**
- `docs/DATA_COMPOSITION_SWEEPS.md` - Architecture and design
- `IMPLEMENTATION_SUMMARY.md` - Implementation details
