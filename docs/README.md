# Documentation Index

## For GitHub Copilot / AI Assistants

These docs provide context for understanding and extending the codebase.

### 🚀 Start Here
- **[PIPELINE_OVERVIEW.md](PIPELINE_OVERVIEW.md)** - **Complete pipeline walkthrough** — training, inference, evaluation, sweeps
- **[DATA_SPLITS.md](DATA_SPLITS.md)** - **Train/val/test split breakdown** — file counts, sources, OOD vs IID
- **[EVALUATION_PIPELINE.md](EVALUATION_PIPELINE.md)** - **How evaluation works** — scoring logic, metrics, bug fixes

### 📊 Evaluation & Results
- **[RE_EVALUATION_GUIDE.md](RE_EVALUATION_GUIDE.md)** - **How to re-evaluate experiments** after the 2025-12-21 bug fix

### Architecture & Design
- **[UI_ARCHITECTURE.md](UI_ARCHITECTURE.md)** - Component structure, state management, data flow, and patterns
- **[DATA_MODEL.md](DATA_MODEL.md)** - Data structures (UIState, ConfigSummary), DataFrame schemas, key computations

### Practical Guides
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Code snippets for common operations (adding columns, filtering, computing summaries)
- **[SWEEPS.md](SWEEPS.md)** - Sweep design and execution (existing doc)
- **[DATA_COMPOSITION_SWEEPS.md](DATA_COMPOSITION_SWEEPS.md)** - Data composition experiments (existing doc)

## Key Files to Know

### UI Entry Point
- `scripts/streamlit_app.py` - Main Streamlit app (tabs: Evaluate, Sweeps, Hard Negatives, File Management)

### UI Components
- `ui/analysis/views.py` - Leaderboard, metric controls, signature details
- `ui/analysis/metrics.py` - Summary computation, aggregation across seeds
- `ui/analysis/data.py` - CSV loading and validation
- `ui/common/types.py` - Core dataclasses (UIState, ConfigSummary, MetricSummary)

### Evaluation Toolkit
- `eval_toolkit/signature.py` - Config signature generation (hash of params)
- `eval_toolkit/review.py` - Grouping, summarization, filtering
- `eval_toolkit/rank.py` - Stability metrics, combined ranking

### Pipeline
- `pipeline/make_training_package.py` - Data selection and preparation
- `pipeline/evaluate_results.py` - Metrics computation from model outputs
- `pipeline/collect_experiments.py` - Aggregate experiment summaries into all_experiments.csv

## Common Tasks

### Adding a New Leaderboard Column
See [QUICK_REFERENCE.md § Adding a New Optional Column](QUICK_REFERENCE.md#adding-a-new-optional-column-to-leaderboard)

### Understanding Data Flow
See [UI_ARCHITECTURE.md § Data Flow](UI_ARCHITECTURE.md#data-flow)

### Understanding Config Signatures
See [DATA_MODEL.md § Signature Generation](DATA_MODEL.md#signature-generation)

## Conventions

1. **Multi-key lookups**: Always try fallback keys (schema has evolved over time)
2. **State persistence**: Use `st.session_state` for cross-rerun state
3. **Widget keys**: Always provide unique `key=` to stateful Streamlit widgets
4. **Docstrings**: Include Args, Returns, and usage examples for public APIs
5. **Type hints**: Use type annotations for function signatures
