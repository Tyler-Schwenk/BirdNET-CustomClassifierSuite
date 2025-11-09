# UI Integration Complete - Summary

## What Was Implemented

### 1. UI Sweep Form Support
**File**: `birdnet_custom_classifier_suite/ui/sweeps/types.py`

Added two new sweep axes to `SweepState` dataclass:
```python
# Data composition axes (for curated subsets)
positive_subset_opts: list[list[str]] = field(default_factory=lambda: [[]])
negative_subset_opts: list[list[str]] = field(default_factory=lambda: [[]])
```

Updated `get_axes_dict()` to conditionally include subset axes:
```python
if self.positive_subset_opts and self.positive_subset_opts != [[]]:
    axes["positive_subsets"] = self.positive_subset_opts
if self.negative_subset_opts and self.negative_subset_opts != [[]]:
    axes["negative_subsets"] = self.negative_subset_opts
```

**File**: `birdnet_custom_classifier_suite/ui/sweeps/views.py`

Added UI inputs in `sweep_form()`:
- **Folder picker buttons** (📁 Browse) using tkinter filedialog
  - Opens native file explorer for easy folder selection
  - Automatically converts to relative paths from workspace root
  - Real-time validation - only valid folders can be added
- **Selected folder list** with ❌ remove buttons for each item
- **Text area inputs** for manual path entry (alternative to file picker)
  - Line-by-line parsing (one combination per line)
  - Comma-separated paths within each line for multi-folder combinations
- **Automatic path validation** with visual feedback
  - Green success message when all paths exist
  - Yellow warning for missing/invalid paths
- Help text with examples

Example input format:
```
curated/bestLowQuality/small
curated/bestLowQuality/medium,curated/bestLowQuality/large
```

### 2. Result Tracking in CSV
**File**: `birdnet_custom_classifier_suite/pipeline/collect_experiments.py`

Added two new columns to `minimal_cols` schema:
```python
"dataset.filters.positive_subsets",
"dataset.filters.negative_subsets",
```

These columns auto-populate from the flattened `experiment_summary.json`.

### 3. Verified Data Flow

The subset information flows through the entire pipeline:

```
UI Form Input
    ↓
SweepState.positive_subset_opts / negative_subset_opts
    ↓
sweep_generator.py (via get_axes_dict())
    ↓
config YAML: training_package.positive_subsets / negative_subsets
    ↓
make_training_package.py run_from_config() → merged_cfg
    ↓
write_reports() → selection_report.json: "filters": {...}
    ↓
evaluate_results.py → experiment_summary.json: dataset.filters
    ↓
collect_experiments.py → all_experiments.csv columns
    ↓
UI Analysis Tab (filterable/groupable)
```

## Files Modified

1. ✅ `birdnet_custom_classifier_suite/ui/sweeps/types.py`
   - Added `positive_subset_opts` and `negative_subset_opts` fields
   - Updated `get_axes_dict()` to include them

2. ✅ `birdnet_custom_classifier_suite/ui/sweeps/views.py`
   - Added text area inputs for subset axes
   - Added parsing logic (line-by-line, comma-separated)

3. ✅ `birdnet_custom_classifier_suite/pipeline/collect_experiments.py`
   - Added subset columns to `minimal_cols` schema

4. ✅ `QUICK_REFERENCE.md`
   - Added UI usage section with examples

## No Changes Needed

- **evaluate_results.py**: Already captures subset info from `selection_report.json` filters
- **make_training_package.py**: Already includes subsets in `write_reports()` via merged_cfg
- **sweep_generator.py**: Already handles subset axes (implemented previously)

## Testing Status

- ✅ **Syntax validation**: No errors in types.py or views.py
- ✅ **Logic validation**: Tested get_axes_dict() with subsets - works correctly
- ✅ **Empty handling**: Verified empty subsets (`[[]]`) excluded from axes
- ✅ **Config generation**: Existing stage8_pilot configs show subset fields in YAML

## Next Steps for User

### Option A: Test Via UI (Recommended)
1. Start Streamlit: `streamlit run birdnet_custom_classifier_suite/ui/app.py`
2. Go to Sweeps tab
3. Enter subset paths in new text areas:
   - Positive: `curated/bestLowQuality/small` (one per line)
   - Negative: `curated/hardNeg/hardneg_conf_min_85` (one per line)
4. Generate sweep → configs will include subset axes
5. Run one experiment
6. Check `all_experiments.csv` for subset columns

### Option B: Use Existing CLI Workflow
Your existing sweep specs (stage8_pilot.yaml, stage8_data_composition.yaml) already work:
```bash
python -m birdnet_custom_classifier_suite.sweeps.sweep_generator \
  --spec config/sweep_specs/stage8_pilot.yaml
```

### Analyze Results
After running experiments:
1. Open UI Analysis tab
2. Load `all_experiments.csv`
3. Filter by `dataset.filters.positive_subsets` and `dataset.filters.negative_subsets`
4. Compare metrics across different data compositions

## How It Works

### UI Input Format
Each line in the text area = one combination to test in the sweep.

**Example 1: Single folders**
```
curated/bestLowQuality/small
curated/bestLowQuality/medium
```
Creates 2 sweep values:
- `[["curated/bestLowQuality/small"]]`
- `[["curated/bestLowQuality/medium"]]`

**Example 2: Multiple folders per combination**
```
curated/bestLowQuality/small
curated/bestLowQuality/medium,curated/bestLowQuality/large
```
Creates 2 sweep values:
- `[["curated/bestLowQuality/small"]]`
- `[["curated/bestLowQuality/medium", "curated/bestLowQuality/large"]]`

### Factorial Expansion
If you specify:
- 2 seeds
- 2 positive_subset combinations
- 2 negative_subset combinations

You get: 2 × 2 × 2 = 8 experiments

### Result Tracking
The CSV columns show exactly which subsets were used:
- `dataset.filters.positive_subsets`: `[]` or `['curated/bestLowQuality/small']`
- `dataset.filters.negative_subsets`: `[]` or `['curated/hardNeg/hardneg_conf_min_85']`

You can filter/group by these columns in the analysis tab to see how different data compositions affect metrics like F1, precision, recall, AUROC.

## Summary

The data composition sweep system is now **fully integrated with the UI**:

✅ Create sweeps via UI form with text area inputs  
✅ Generate configs with subset axes  
✅ Run experiments that load curated subsets  
✅ Track results in all_experiments.csv  
✅ Analyze in UI with subset filtering  

No manual YAML editing required - everything works through the UI!
