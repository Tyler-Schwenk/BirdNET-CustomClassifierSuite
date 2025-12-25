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
from birdnet_custom_classifier_suite.eval_toolkit import constants


# ------------------------- Load & Validate ------------------------- #

def load_experiments(csv_path: str | Path) -> pd.DataFrame:
    """Load the master experiments CSV into a DataFrame."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Experiment CSV not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"CSV is empty: {path}")

    # Warn if core metrics are missing, but continue (CSV may use alternate prefixes)
    missing = [m for m in constants.CORE_METRICS if m not in df.columns]
    if missing:
        print(f"WARNING: missing core metric columns: {missing} (CSV may use different prefixes)")

    print(f"Loaded {len(df)} experiment rows from {path}")
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
    Computes mean/std for all metrics in the specified group.
    Example: summarize_grouped(df, metric_prefix="metrics.ood.best_f1")
    """
    # Work in canonical 'metrics.' space
    if not metric_prefix.startswith("metrics."):
        metric_prefix = f"metrics.{metric_prefix}"
    
    # Only aggregate the core performance metrics we care about (exclude parameters
    # like threshold which represent choices, not performance across seeds).
    core_keys = ["f1", "precision", "recall"]
    metrics = []
    for k in core_keys:
        col = f"{metric_prefix}.{k}"
        if col in df.columns:
            metrics.append(col)

    if not metrics:
        raise ValueError(f"No metric columns found with prefix: {metric_prefix} for keys {core_keys}")

    # For each signature, compute mean and std of each metric
    grouped = group_by_signature(df)
    summary = grouped[metrics].agg(['mean', 'std'])

    # Flatten multi-index columns into <metric>_{mean,std}
    summary.columns = [f"{col[0]}_{col[1]}" for col in summary.columns]

    # Include names for reference (list unique experiment names per signature)
    if 'experiment.name' in df.columns:
        summary['experiment.names'] = grouped['experiment.name'].agg(lambda x: ', '.join(sorted(x.unique())))

    return summary.reset_index()
    
    # Find metrics but exclude parameter columns (like thresholds)
    metrics = [c for c in df.columns if c.startswith(metric_prefix) and 
               not any(c.startswith(p) for p in constants.PARAMETER_COLUMNS)]

    if not metrics:
        raise ValueError(f"No metric columns found with prefix: {metric_prefix}")

    summary_metrics = group[metrics].agg(["mean", "std"])
    summary_metrics.columns = [f"{m}_{stat}" for m, stat in summary_metrics.columns]
    summary_metrics.reset_index(inplace=True)

    # For each signature, collect all experiment names and seeds
    if "experiment.name" in df.columns:
        name_groups = group["experiment.name"].agg(lambda x: sorted(list(set(x)))).reset_index()
        name_groups["experiment.names"] = name_groups["experiment.name"].apply(lambda x: ", ".join(x))
        # Merge experiment names with metric summary
        summary = name_groups.merge(summary_metrics, on=name_groups.columns[0], how="inner")
        summary.drop(columns=["experiment.name"], inplace=True)  # Drop the list column
    else:
        summary = summary_metrics

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
