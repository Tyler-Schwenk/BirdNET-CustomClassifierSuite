# Data Model Reference

## Core Data Structures

### UIState
**Location:** `birdnet_custom_classifier_suite/ui/common/types.py`

Session-persisted state for the Streamlit app.

```python
@dataclass
class UIState:
    # Data source
    data_source: Optional[Union[str, Path]]  # Path to loaded CSV
    results_df: Optional[pd.DataFrame]        # Raw experiment results
    
    # Analysis settings
    metric_prefix: str = MetricGroup.OOD_BEST_F1  # Current metric being analyzed
    top_n: int = 10                               # Number of top configs to show
    precision_floor: Optional[float] = None       # Minimum precision filter
    
    # Filters (user selections)
    quality_filter: Optional[List]                # e.g., ["high", "medium"]
    balance_filter: Optional[List]                # e.g., ["balanced"]
    sweep_filter: Optional[List]                  # e.g., ["stage4_sweep", "stage5_sweep"]
    
    # Computed results
    summaries: List[ConfigSummary]                # Top N configs after analysis
```

### ConfigSummary
**Location:** `birdnet_custom_classifier_suite/ui/common/types.py`

Represents one configuration's aggregated metrics across seeds.

```python
@dataclass
class ConfigSummary:
    signature: str                                 # 8-char hash of config params
    experiment_names: List[str]                    # List of experiment names with this config
    metrics: Dict[str, MetricSummary]              # metric_name -> MetricSummary
    config_values: Dict[str, Union[str, float, int, bool]]  # Flattened config params
```

**Example `config_values` keys:**
- `dataset.filters.quality` → "high"
- `dataset.filters.call_type` → ["Flight", "Song"]
- `training.batch_size` → 32
- `training_args.mixup` → True

### MetricSummary
```python
@dataclass
class MetricSummary:
    name: str              # e.g., "metrics.ood.best_f1"
    mean: float            # Average across seeds
    std: float             # Standard deviation
    cv: Optional[float]    # Coefficient of variation
    stability: Optional[float]  # Inverse CV (higher = more stable)
```

## DataFrame Schemas

### all_experiments.csv

**Flattened YAML structure** from `experiment_summary.json` files.

**Key columns:**
```
experiment.name                    # Unique experiment identifier
experiment.seed                    # Random seed for this run

# Dataset config
dataset.filters.quality            # "high", "medium", "low", or list
dataset.filters.balance            # "balanced", "imbalanced"
dataset.filters.call_type          # Call types included (list)
dataset.filters.positive_subsets   # Positive audio subsets (list of paths)
dataset.filters.negative_subsets   # Negative audio subsets (list of paths)

# Training config
training.epochs
training.batch_size
training_args.mixup
training_args.label_smoothing
training_args.focal_loss
training_args.upsampling_mode
training_args.upsampling_ratio

# Metrics (flattened)
metrics.iid.best_f1.threshold      # Threshold at best F1 (in-distribution)
metrics.iid.best_f1.precision
metrics.iid.best_f1.recall
metrics.iid.best_f1.f1
metrics.iid.auroc

metrics.ood.best_f1.threshold      # Threshold at best F1 (out-of-distribution)
metrics.ood.best_f1.precision
metrics.ood.best_f1.recall
metrics.ood.best_f1.f1
metrics.ood.auroc

# Generated columns
__signature                        # Config hash (8 chars, excludes seed)
```

### Summarized Metrics DataFrame

**Created by:** `eval_toolkit/review.py::summarize_grouped()`

**Columns:**
```
__signature                        # Config hash
experiment.names                   # Comma-separated experiment names

# Aggregated metrics
metrics.ood.best_f1.f1_mean
metrics.ood.best_f1.f1_std
metrics.ood.best_f1.f1_cv
metrics.ood.best_f1.f1_stability

metrics.ood.best_f1.precision_mean
metrics.ood.best_f1.precision_std
# ... etc for recall, iid metrics
```

## Key Computations

### Signature Generation
**File:** `eval_toolkit/signature.py`

```python
def build_config_signature(row: pd.Series, config_cols: List[str]) -> str:
    """Create deterministic hash from config parameters (excludes seeds)."""
    # Format: "col1=val1|col2=val2|..." → SHA1 → first 8 chars
```

### Stability Metric
**File:** `eval_toolkit/rank.py`

```python
def compute_stability(summary: pd.DataFrame) -> pd.DataFrame:
    """Add stability = 1 / (CV + epsilon) for each metric.
    
    Higher stability = lower variance across seeds = more reproducible config.
    """
```

### Combined Ranking
**File:** `eval_toolkit/rank.py`

```python
def combined_rank(summary, metric, precision_floor, stability_weight=0.2):
    """Rank configs by weighted score = F1 * (1 + stability_weight * stability).
    
    Filters out configs below precision_floor first.
    """
```

## Common Patterns

### Multi-key Lookup
Due to schema evolution, always try multiple keys:

```python
val = summary.config_values.get('dataset.filters.quality',
      summary.config_values.get('filters.quality',
      summary.config_values.get('quality')))
```

### List Formatting
Parse string-encoded lists and format for display:

```python
if isinstance(val, str) and val.startswith('['):
    import ast
    parsed = ast.literal_eval(val)
    if isinstance(parsed, list):
        return ', '.join(str(x) for x in parsed)
```

### NA Handling
```python
if val is None or pd.isna(val):
    return 'n/a'
```
