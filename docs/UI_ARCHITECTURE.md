# UI Architecture

## Overview

The Streamlit UI is built using **modular, reusable components** organized by feature area. Each component is designed to be stateless and composable.

## Component Structure

```
birdnet_custom_classifier_suite/ui/
├── analysis/           # Experiment results analysis & leaderboard
│   ├── data.py        # Load and validate results CSV
│   ├── metrics.py     # Compute summaries, aggregate across seeds
│   ├── views.py       # Leaderboard table, metric controls, signature details
│   └── plots.py       # Chart rendering (scatter, line, etc.)
├── sweeps/            # Sweep design and execution
│   ├── views.py       # Form controls for sweep parameters
│   ├── types.py       # SweepState dataclass
│   └── utils.py       # YAML generation, run execution
├── common/            # Shared utilities
│   ├── types.py       # UIState, ConfigSummary, MetricSummary dataclasses
│   └── widgets.py     # Reusable Streamlit widgets
└── __init__.py        # Public API exports
```

## Key Patterns

### State Management

**UIState** (session-persisted):
- `results_df`: Loaded experiment results DataFrame
- `summaries`: Computed ConfigSummary objects for top N configs
- `metric_prefix`: Current metric being analyzed (e.g., "metrics.ood.best_f1")
- `filters`: Quality, balance, sweep/stage selection

**Streamlit Session State**:
- Persists across reruns within same browser session
- Key objects: `ui_state`, `selected_signature`, `auto_load_attempted`

### Data Flow

```
1. Load CSV → state.results_df
2. Apply filters (quality, balance, stage) → filtered DataFrame
3. Add signatures → group by config hash
4. Summarize metrics → compute mean/std across seeds → state.summaries
5. Rank → sort by F1/precision/stability
6. Display → interactive AgGrid table
7. User selects row → get_signature_breakdown() → show per-run details
```

### Column Toggle Pattern

**Leaderboard optional columns** follow this pattern:

1. **Checkbox control** - User toggles visibility
2. **Data extraction** - Pull from `summary.config_values` with fallback keys
3. **Format value** - Handle lists, booleans, None/NaN gracefully
4. **Add to config_column_names** - Track for AgGrid column identification
5. **Configure AgGrid column** - Set width, sorting, filtering

Example (Call Type):
```python
# 1. Checkbox
show_call_type = st.checkbox("Call Type", value=False, key='lb_call_type')

# 2. Extract with fallbacks
if show_call_type:
    val = summary.config_values.get('dataset.filters.call_type',
          summary.config_values.get('filters.call_type',
          summary.config_values.get('call_type')))
    row["Call Type"] = format_config_value(val)

# 3. Track column
if show_call_type:
    config_column_names.append('Call Type')

# 4. Configure AgGrid
if show_call_type and 'Call Type' in df_display.columns:
    gb.configure_column('Call Type', minWidth=100, width=130)
```

## Important Conventions

### Configuration Signatures

- **`__signature`**: SHA1 hash of config parameters (excludes seeds)
- Allows grouping runs with same config but different random seeds
- Computed in `eval_toolkit/signature.py::add_signatures()`

### Metric Prefixes

- Format: `metrics.{dataset}.{metric_type}.{field}`
- Examples:
  - `metrics.ood.best_f1.f1` - Out-of-distribution F1 at best threshold
  - `metrics.iid.auroc` - In-distribution area under ROC curve

### Config Value Lookups

Always try **multiple fallback keys** due to schema evolution:
```python
val = summary.config_values.get('dataset.filters.quality',
      summary.config_values.get('filters.quality',
      summary.config_values.get('quality')))
```

## Adding New Features

### New Optional Column

1. Add checkbox in `leaderboard()` controls section
2. Extract value in row-building loop with fallbacks
3. Add to `config_column_names` list
4. Add AgGrid column configuration
5. Update this doc with usage notes

### New Tab

1. Create module in `ui/feature_name/`
2. Add `views.py` with main panel function
3. Export in `ui/__init__.py`
4. Import and call in `streamlit_app.py` tab

## Performance Notes

- **Auto-load on startup**: First visit loads default CSV and analyzes
- **Persistent summaries**: Computed once per "Analyze" click, then cached
- **AgGrid row IDs**: Use signature (not auto-hash) for stable selection across reruns
- **Parallel tool calls**: Batch independent operations (file reads, searches)
