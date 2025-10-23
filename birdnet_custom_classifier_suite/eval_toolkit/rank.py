#!/usr/bin/env python3
"""
rank.py

Ranking utilities for experiment configurations.

Works with summarized experiment data (mean/std across seeds)
to identify top-performing configurations based on:
  - primary metric (e.g., OOD F1)
  - precision-priority ranking
  - recall floor enforcement
  - combined stability score (mean vs std)
"""

import pandas as pd


# ------------------------- Core Ranking ------------------------- #

def rank_configs(
    df: pd.DataFrame,
    metric: str,
    ascending: bool = False,
    top_n: int | None = None,
    stability_weight: float = 0.0,
):
    """
    Rank configurations by metric mean (and optionally penalize std).
    Example:
        rank_configs(df, metric="ood.best_f1.f1", stability_weight=0.5)
    """
    # Normalize metric name to canonical 'metrics.' prefix
    if not metric.startswith("metrics."):
        metric = f"metrics.{metric}"

    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    if mean_col not in df.columns:
        raise KeyError(f"Missing metric mean column: {mean_col}")

    ranked = df.copy()
    if stability_weight > 0 and std_col in df.columns:
        ranked["score"] = ranked[mean_col] - stability_weight * ranked[std_col]
    else:
        ranked["score"] = ranked[mean_col]

    ranked = ranked.sort_values("score", ascending=ascending).reset_index(drop=True)
    if top_n:
        ranked = ranked.head(top_n)
    return ranked


# ------------------------- Precision-Priority ------------------------- #

def rank_precision_priority(
    df: pd.DataFrame,
    precision_col: str = "ood.best_f1.precision_mean",
    recall_col: str = "ood.best_f1.recall_mean",
    precision_floor: float = 0.9,
):
    """
    Rank configs prioritizing precision ≥ floor, then by recall descending.
    Returns only configs meeting the precision criterion.
    """
    # Normalize to canonical metric names
    if not precision_col.startswith("metrics."):
        precision_col = f"metrics.{precision_col}"
    if not recall_col.startswith("metrics."):
        recall_col = f"metrics.{recall_col}"

    if precision_col not in df.columns or recall_col not in df.columns:
        raise KeyError(f"Missing precision or recall columns: {precision_col}, {recall_col}")

    filtered = df[df[precision_col] >= precision_floor].copy()
    if filtered.empty:
        print(f"No configs meet precision ≥ {precision_floor}")
        return pd.DataFrame()

    ranked = filtered.sort_values(
        by=[precision_col, recall_col],
        ascending=[False, False]
    ).reset_index(drop=True)
    return ranked


# ------------------------- F1 at Precision Floor ------------------------- #

def hp_f1_at_precision_floor(
    df: pd.DataFrame,
    precision_col: str = "ood.best_f1.precision_mean",
    recall_col: str = "ood.best_f1.recall_mean",
    f1_col: str = "ood.best_f1.f1_mean",
    precision_floor: float = 0.9,
):
    """
    Returns the highest-F1 config among those meeting a precision floor.
    If none meet it, returns empty DataFrame.
    """
    # Normalize to canonical metric names for f1/precision/recall
    if not precision_col.startswith("metrics."):
        precision_col = f"metrics.{precision_col}"
    if not recall_col.startswith("metrics."):
        recall_col = f"metrics.{recall_col}"
    if not f1_col.startswith("metrics."):
        f1_col = f"metrics.{f1_col}"

    if any(col not in df.columns for col in (precision_col, recall_col, f1_col)):
        raise KeyError("Missing one or more metric columns (expected metrics.* prefixed names)")

    eligible = df[df[precision_col] >= precision_floor].copy()
    if eligible.empty:
        print(f"No configs reach precision ≥ {precision_floor}")
        return pd.DataFrame()

    top = eligible.sort_values(f1_col, ascending=False).head(1).reset_index(drop=True)
    return top


# ------------------------- Stability Analysis ------------------------- #

def compute_stability(df: pd.DataFrame, metric_prefix: str = "ood.best_f1"):
    """
    Adds stability metrics to a summarized DataFrame:
      - coefficient of variation (std / mean)
      - inverse stability score (1 / CV)
    """
    df = df.copy()
    # Accept prefixes with or without 'metrics.'
    prefixes = [metric_prefix]
    if not metric_prefix.startswith("metrics."):
        prefixes.insert(0, f"metrics.{metric_prefix}")

    for col in [c for c in df.columns if any(c.startswith(p) for p in prefixes) and c.endswith("_mean")]:
        base = col.replace("_mean", "")
        std_col = f"{base}_std"
        if std_col in df.columns:
            cv_col = f"{base}_cv"
            inv_col = f"{base}_stability"
            df[cv_col] = df[std_col] / df[col]
            df[inv_col] = 1.0 / df[cv_col].replace(0, pd.NA)
    return df


# ------------------------- Combined Ranking ------------------------- #

def combined_rank(
    df: pd.DataFrame,
    metric: str = "metrics.ood.best_f1.f1",  # Base metric name without _mean
    precision_floor: float = None,
    stability_weight: float = 0.2,
):
    """
    Combined ranking heuristic that uses pre-computed mean columns:
      - Optionally filters by precision floor (if provided)
      - Scores by F1 mean minus stability_weight * std
      - Returns sorted table
    
    Note: Expects metrics to already have _mean/_std suffixes
    """
    # Always work with mean column names
    mean_col = f"{metric}_mean" if not metric.endswith("_mean") else metric
    std_col = mean_col.replace("_mean", "_std")
    prec_mean = mean_col.replace("f1_mean", "precision_mean")

    if mean_col not in df.columns:
        raise KeyError(f"Missing required metric column: {mean_col}")

    eligible = df.copy()
    if precision_floor is not None and precision_floor > 0:
        if prec_mean not in df.columns:
            raise KeyError(f"Missing precision column: {prec_mean}")
        eligible = df[df[prec_mean] >= precision_floor].copy()
        if eligible.empty:
            print(f"No configs meet precision ≥ {precision_floor}")
            return pd.DataFrame()

    # Score using the mean and optionally penalize by std
    eligible["score"] = eligible[mean_col]
    if stability_weight > 0 and std_col in eligible.columns:
        eligible["score"] -= stability_weight * eligible[std_col]
        
    ranked = eligible.sort_values("score", ascending=False).reset_index(drop=True)
    return ranked
