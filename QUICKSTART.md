# Quick Start Guide - New CLI Commands

After installing the package, you now have convenient command-line tools:

## Installation

```bash
# Install in development mode (recommended for local development)
pip install -e .

# Or install normally
pip install .
```

## Available Commands

### 1. **Analyze Experiments** (`birdnet-analyze`)

Aggregate experiment results, rank configurations, and generate leaderboards.

```bash
# Analyze all experiments
birdnet-analyze

# Filter by stage and set precision floor
birdnet-analyze --stage stage4_ --precision-floor 0.9

# Customize output
birdnet-analyze --top-n 20 --metric-prefix metrics.ood.best_f1
```

**Options:**
- `--exp-root` - Path to experiment folders (default: `experiments`)
- `--results` - Master results CSV path (default: `results/all_experiments.csv`)
- `--stage` - Filter experiments by prefix (e.g., `stage4_`)
- `--precision-floor` - Minimum precision threshold for filtering
- `--metric-prefix` - Metric group to analyze (default: `metrics.ood.best_f1`)
- `--top-n` - Number of top configs to report (default: 10)

### 2. **Launch UI** (`birdnet-ui`)

Start the interactive Streamlit interface for experiment analysis and sweep design.

```bash
# Launch the UI
birdnet-ui
```

This opens a browser with:
- **Evaluate Tab**: Load results, filter/analyze experiments, view leaderboards
- **Sweeps Tab**: Design parameter sweeps, generate configs, run experiments

## Alternative Usage

You can also run commands via Python module syntax:

```bash
# Analyze
python -m birdnet_custom_classifier_suite.cli.analyze --stage stage4_

# Launch UI
python -m birdnet_custom_classifier_suite.cli.ui

# Or use the dispatcher
python -m birdnet_custom_classifier_suite.cli analyze --stage stage4_
python -m birdnet_custom_classifier_suite.cli ui
```

## Backward Compatibility

Old script paths still work but will show deprecation warnings:

```bash
python scripts/run_analysis.py --stage stage4_  # Still works, redirects to new CLI
```

## Next Steps

- See main [README.md](README.md) for full pipeline documentation
- Check [config/](config/) for sweep specifications and examples
- View [results/leaderboards/](results/leaderboards/) for generated reports
