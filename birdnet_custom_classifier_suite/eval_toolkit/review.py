#!/usr/bin/env python3
"""
review.py

Provides utilities for loading, filtering, and summarizing experiment results
from `results/all_experiments.csv`.

Core functionality:
  - Load flattened experiment CSV
  - Filter by stage prefix or config values
  - Group by config signature or hyperparameter sets
  - Summarize metrics (mean, std) across seeds
"""

import pandas as pd
from pathlib import Path
from birdnet_custom_classifier_suite.eval_toolkit.constants import METRIC_COLUMNS


# ------------------------- Load & Validate ------------------------- #

def load_experiments(csv_path: str | Path) -> pd.DataFrame:
    """Load the master experiments CSV into a DataFrame."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Experiment CSV not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"CSV is empty: {path}")

    # Warn if known metric columns are missing, but continue (CSV may use alternate prefixes)
    missing = [m for m in METRIC_COLUMNS if m not in df.columns]
    if missing:
        print(f"⚠️  Warning: missing expected metric columns: {missing} (CSV may use different prefixes)")

    print(f"✅ Loaded {len(df)} experiment rows from {path}")
    return df


# ------------------------- Filtering ------------------------- #

def filter_by_stage(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Filter experiments by name prefix (e.g., 'stage4_')."""
    if "experiment.name" not in df.columns:
        raise KeyError("Missing 'experiment.name' column in DataFrame.")
    return df[df["experiment.name"].astype(str).str.startswith(prefix)].copy()


def filter_by_config(df: pd.DataFrame, **criteria) -> pd.DataFrame:
    """
    Filter DataFrame by matching config key/value pairs.
    Example:
        df_filtered = filter_by_config(df, dropout=0.25, hidden_units=512)
    """
    filtered = df.copy()
    for key, val in criteria.items():
        col = f"training_args.{key}"
        if col not in filtered.columns:
            raise KeyError(f"Column not found: {col}")
        filtered = filtered[filtered[col] == val]
    return filtered


def filter_top(df: pd.DataFrame, metric: str, top_n: int = 10) -> pd.DataFrame:
    """Return top-N rows sorted by a given metric descending.

    The `metric` argument may be given with or without the leading 'metrics.' prefix.
    Internally we resolve to the canonical 'metrics.' prefixed column name.
    """
    if not metric.startswith("metrics."):
        cand = f"metrics.{metric}"
    else:
        cand = metric

    if cand not in df.columns:
        raise KeyError(f"Metric column not found: {cand}")
    return df.sort_values(cand, ascending=False).head(top_n).reset_index(drop=True)


# ------------------------- Grouping & Summary ------------------------- #

def group_by_signature(df: pd.DataFrame) -> pd.core.groupby.DataFrameGroupBy:
    """Group runs by unique configuration signature (if available)."""
    sig_col = "__signature" if "__signature" in df.columns else "experiment.name"
    return df.groupby(sig_col, dropna=False)


def summarize_grouped(df: pd.DataFrame, metric_prefix: str = "metrics.ood.best_f1"):
    """
    Summarize grouped results across seeds.
    Computes mean/std for all metric columns under a given prefix.
    Example: summarize_grouped(df, metric_prefix="metrics.ood.best_f1")
    """
    group = group_by_signature(df)
    # Work in canonical 'metrics.' space; accept legacy prefix by normalizing
    if not metric_prefix.startswith("metrics."):
        metric_prefix = f"metrics.{metric_prefix}"

    metrics = [c for c in df.columns if c.startswith(metric_prefix)]

    if not metrics:
        raise ValueError(f"No metric columns found with prefix: {metric_prefix}")

    summary = group[metrics].agg(["mean", "std"])
    summary.columns = [f"{m}_{stat}" for m, stat in summary.columns]
    summary.reset_index(inplace=True)
    return summary


# ------------------------- Convenience ------------------------- #

def describe_metrics(df: pd.DataFrame):
    """Show high-level stats for each metric column."""
    metrics = [c for c in df.columns if any(k in c for k in ["metrics.iid", "metrics.ood"])]
    if not metrics:
        raise ValueError("No metric columns found in DataFrame.")
    desc = df[metrics].describe().T
    desc["coefficient_of_var"] = desc["std"] / desc["mean"]
    return desc.round(4)
