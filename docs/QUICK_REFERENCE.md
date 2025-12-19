# Quick Reference - Common Operations

## Adding a New Optional Column to Leaderboard

**File:** `birdnet_custom_classifier_suite/ui/analysis/views.py`

### Step 1: Add checkbox control
```python
# In leaderboard() function, column controls section
with cols_ctrl2[N]:  # Choose next available column slot
    show_my_column = st.checkbox("My Column", value=False, key='lb_my_column')
```

### Step 2: Extract and format value
```python
# In row-building loop (after other show_* checks)
if show_my_column:
    # Try multiple fallback keys (schema evolution)
    val = summary.config_values.get('dataset.filters.my_param',
          summary.config_values.get('filters.my_param',
          summary.config_values.get('my_param')))
    
    # Format value (handles None, lists, booleans)
    row["My Column"] = format_config_value(val)
```

### Step 3: Track as config column
```python
# In AgGrid section, config_column_names list
if show_my_column:
    config_column_names.append('My Column')
```

### Step 4: Configure AgGrid column
```python
# After other column configurations
if show_my_column and 'My Column' in df_display.columns:
    gb.configure_column('My Column', minWidth=100, width=130)
```

## Filtering Experiment Results

### By Quality
```python
quality_col = _find_col(df.columns, ['dataset.filters.quality', 'filters.quality', 'quality'])
if quality_col and state.quality_filter:
    df_filtered = df[df[quality_col].isin(state.quality_filter)]
```

### By Stage/Sweep
```python
def extract_stage(name):
    import re
    m = re.search(r'stage\d+[a-z]*(?:_sweep)?', str(name).lower())
    return m.group(0) if m else None

df['__stage_temp'] = df['experiment.name'].apply(extract_stage)
df_filtered = df[df['__stage_temp'].isin(state.sweep_filter)]
```

## Computing Summaries

```python
from birdnet_custom_classifier_suite.ui.analysis.metrics import summarize_metrics

# Returns (ConfigSummary list, full summary DataFrame)
summaries, summary_df = summarize_metrics(
    df,
    metric_prefix="metrics.ood.best_f1",  # or state.metric_prefix
    top_n=10,                             # or state.top_n
    precision_floor=0.9,                  # or state.precision_floor
)

# Each ConfigSummary has:
# - signature: 8-char hash
# - experiment_names: list of experiment names
# - metrics: dict of MetricSummary objects
# - config_values: dict of flattened config params
```

## Getting Per-Run Details

```python
from birdnet_custom_classifier_suite.ui.analysis.metrics import get_signature_breakdown

breakdown = get_signature_breakdown(
    df=state.results_df,
    config_signature=selected_signature,  # 8-char hash
    metric_prefix=state.metric_prefix,
)

# breakdown.rows: DataFrame of individual runs
# breakdown.aggregates: dict of (mean, std) tuples
# breakdown.config_columns: list of config column names
```

## Working with Config Values

### Multi-key Lookup Pattern
```python
def get_config_val(config_values: dict, *keys) -> Any:
    """Try multiple keys in order, return first found."""
    for key in keys:
        if key in config_values:
            return config_values[key]
    return None

val = get_config_val(
    summary.config_values,
    'dataset.filters.quality',
    'filters.quality',
    'quality',
)
```

### Parsing List Values
```python
import ast

def parse_list_value(val):
    """Parse string-encoded list or return as-is."""
    if isinstance(val, str) and val.startswith('['):
        try:
            return ast.literal_eval(val)
        except:
            return val
    return val

subsets = parse_list_value(config_values.get('positive_subsets'))
if isinstance(subsets, list):
    names = [s.split('/')[-1] for s in subsets]  # Extract basenames
```

## Streamlit State Management

### Session State (persists across reruns)
```python
# Initialize once
if "ui_state" not in st.session_state:
    st.session_state.ui_state = UIState()

# Access anywhere
state = st.session_state.ui_state

# Modify
state.results_df = new_df
state.summaries = new_summaries

# Custom keys
st.session_state['selected_signature'] = signature
st.session_state['auto_load_attempted'] = True
```

### Widget Keys (prevent duplicate warnings)
```python
# Always provide unique key for stateful widgets
show_quality = st.checkbox("Quality", value=False, key='lb_quality')
metric = st.selectbox("Metric", options=[...], key='metric_selector')
```

## Loading Data

```python
from birdnet_custom_classifier_suite.ui.analysis.data import load_results

# From path
df = load_results(path=Path("results/all_experiments.csv"))

# From uploaded file
df = load_results(uploaded_file=uploaded_file)

# Validates schema (checks for required columns)
```

## Signature Operations

```python
from birdnet_custom_classifier_suite.eval_toolkit import signature

# Add signatures to DataFrame (if not present)
df = signature.add_signatures(df)
# Creates '__signature' column: 8-char hash of config params (excludes seeds)

# Get config columns
config_cols = signature.pick_config_columns(df)
# Returns columns that define configuration (excludes metrics, metadata, seeds)
```

## Ranking and Stability

```python
from birdnet_custom_classifier_suite.eval_toolkit import rank

# Compute stability metrics
summary_df = rank.compute_stability(summary_df)
# Adds *_cv and *_stability columns

# Rank configurations
top = rank.combined_rank(
    summary_df,
    metric="metrics.ood.best_f1.f1",
    precision_floor=0.9,           # Filter out precision < 0.9
    stability_weight=0.2,          # 20% weight on stability
).head(10)
```
