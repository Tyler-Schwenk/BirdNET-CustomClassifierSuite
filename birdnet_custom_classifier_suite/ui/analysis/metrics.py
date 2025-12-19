"""
Metric computation and summarization for the UI.

This module provides a clean interface to the evaluation toolkit's metric
computation functionality, with added type safety and UI-specific formatting.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from birdnet_custom_classifier_suite.eval_toolkit import rank, review, signature
from birdnet_custom_classifier_suite.ui.common.types import (
    ConfigSummary,
    MetricSummary,
    PerRunBreakdown,
)


def format_metric_value(value: Optional[float], std: Optional[float] = None, precision: int = 4) -> str:
    """Format a metric value for display.
    
    Args:
        value: Raw metric value
        std: Optional standard deviation
        precision: Number of decimal places to show
        
    Returns:
        Formatted string
    """
    if pd.isna(value):
        return "n/a"

    result = f"{value:.{precision}f}"
    if std is not None and not pd.isna(std):
        result += f" ± {std:.{precision}f}"
    return result


def summarize_metrics(
    df: pd.DataFrame,
    metric_prefix: str = "metrics.ood.best_f1",
    top_n: int = 10,
    precision_floor: Optional[float] = None
) -> Tuple[List[ConfigSummary], pd.DataFrame]:
    """Compute summary statistics for configurations.
    
    Process:
    1. Add __signature column (hash of config params, excludes seeds)
    2. Group by signature, compute mean/std for metrics
    3. Compute stability (inverse coefficient of variation)
    4. Rank by F1, precision floor, stability weight
    5. Extract config values from original df for each signature
    6. Return top N as ConfigSummary objects + full summary DataFrame
    
    Args:
        df: DataFrame with experiment results (from all_experiments.csv)
        metric_prefix: Metric group to analyze (e.g., "metrics.ood.best_f1")
        top_n: Number of top configurations to return (None = all)
        precision_floor: Optional minimum precision threshold (0.0-1.0 or 0-100)
        
    Returns:
        Tuple[List[ConfigSummary], pd.DataFrame]: (top configs, full summary)
    """
    try:
        # Handle enum objects - extract string value
        if hasattr(metric_prefix, 'value'):
            metric_prefix = metric_prefix.value
        
        # Normalize metric prefix
        if not metric_prefix.startswith("metrics."):
            metric_prefix = f"metrics.{metric_prefix}"

        # Ensure we have signatures
        if "__signature" not in df.columns:
            df = signature.add_signatures(df)

        # Get the summary with mean/std columns
        summary = review.summarize_grouped(df, metric_prefix=metric_prefix)
        summary = rank.compute_stability(
            summary, 
            metric_prefix=metric_prefix.replace("metrics.", "")
        )

        # Normalize precision floor if it's a percentage
        if precision_floor is not None and precision_floor > 1:
            precision_floor = precision_floor / 100.0

        # Get top N configurations
        top = rank.combined_rank(
            summary,
            metric=f"{metric_prefix}.f1",
            precision_floor=precision_floor,
            stability_weight=0.2,
        )
        if top_n:
            top = top.head(top_n)

        # Helper to safely convert values that may be pd.NA to floats
        def _to_float(val, default_if_nan=None):
            try:
                if pd.isna(val):
                    if default_if_nan is None:
                        # Use NaN (float) to represent missing numeric
                        return float('nan')
                    return default_if_nan
                return float(val)
            except Exception:
                return float('nan') if default_if_nan is None else default_if_nan

        # Convert to strongly-typed summaries
        config_summaries = []
        for _, row in top.iterrows():
            metrics = {}
            for col in row.index:
                if col.endswith("_mean"):
                    base = col.replace("_mean", "")
                    if f"{base}_std" in row.index:
                        cv_col = f"{base}_cv"
                        stability_col = f"{base}_stability"
                        # Treat NaN std as 0 for single-seed stability; keep mean as NaN when missing
                        mean_val = _to_float(row[col])
                        std_val = _to_float(row[f"{base}_std"], default_if_nan=0.0)
                        cv_val = None
                        stab_val = None
                        if cv_col in row.index and not pd.isna(row[cv_col]):
                            try:
                                cv_val = float(row[cv_col])
                            except Exception:
                                cv_val = None
                        if stability_col in row.index and not pd.isna(row[stability_col]):
                            try:
                                stab_val = float(row[stability_col])
                            except Exception:
                                stab_val = None

                        metrics[base] = MetricSummary(
                            name=base,
                            mean=mean_val,
                            std=std_val,
                            cv=cv_val,
                            stability=stab_val,
                        )

            # Get config values from original df for this signature
            sig = row["__signature"]
            sig_rows = df[df["__signature"] == sig]
            config_values = {}
            if not sig_rows.empty:
                # Get config columns (non-metric, non-internal)
                config_cols = signature.pick_config_columns(df)
                for col in config_cols:
                    if col in sig_rows.columns:
                        # Get unique value (should be same across all runs with same signature)
                        vals = sig_rows[col].dropna().unique()
                        if len(vals) == 1:
                            config_values[col] = vals[0]
                        elif len(vals) > 1:
                            # Multiple values - shouldn't happen for same signature but handle it
                            config_values[col] = vals[0]  # Just take first

            config_summaries.append(
                ConfigSummary(
                    signature=sig,
                    experiment_names=row["experiment.names"].split(", ") if "experiment.names" in row else [],
                    metrics=metrics,
                    config_values=config_values,
                )
            )

        return config_summaries, summary

    except Exception as e:
        logging.error(f"Failed to summarize metrics: {str(e)}")
        raise RuntimeError(f"Failed to summarize metrics: {str(e)}") from e


def get_signature_breakdown(
    df: pd.DataFrame,
    config_signature: str,
    metric_prefix: str = "metrics.ood.best_f1"
) -> Optional[PerRunBreakdown]:
    """Get detailed per-run information for a configuration signature.
    
    Args:
        df: DataFrame with experiment results
        config_signature: Configuration signature to analyze
        metric_prefix: Metric group to analyze
        
    Returns:
        PerRunBreakdown if signature exists, None if not found
    """
    try:
        # Ensure we have signatures column
        if "__signature" not in df.columns:
            df = signature.add_signatures(df)

        sel = df[df["__signature"] == config_signature]
        if sel.empty:
            logging.info(f"No data found for signature: {config_signature}")
            return None

        # Get metric columns we care about
        preferred_metrics = [
            f"{metric_prefix}.f1",
            f"{metric_prefix}.precision",
            f"{metric_prefix}.recall"
        ]
        metrics = [m for m in preferred_metrics if m in sel.columns]

        # Get relevant config columns
        config_cols = signature.pick_config_columns(df)
        config_cols = [c for c in config_cols if c in sel.columns]

        # Compute aggregates
        aggregates = {}
        for m in metrics:
            vals = sel[m].dropna().astype(float)
            if not vals.empty:
                aggregates[m] = (vals.mean(), vals.std(ddof=0))
            else:
                aggregates[m] = (None, None)

        return PerRunBreakdown(
            signature=config_signature,
            rows=sel,
            metric_columns=metrics,
            config_columns=config_cols,
            aggregates=aggregates,
        )

    except Exception as e:
        logging.error(f"Failed to get signature breakdown: {str(e)}")
        raise RuntimeError(f"Failed to get signature breakdown: {str(e)}") from e