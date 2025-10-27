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
    
    Args:
        df: DataFrame with experiment results
        metric_prefix: Metric group to analyze (e.g., metrics.ood.best_f1)
        top_n: Number of top configurations to return
        precision_floor: Optional minimum precision threshold
        
    Returns:
        Tuple of (top config summaries, full summary DataFrame)
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
                        metrics[base] = MetricSummary(
                            name=base,
                            mean=float(row[col]),
                            std=float(row[f"{base}_std"]),
                            cv=float(row[cv_col]) if cv_col in row.index else None,
                            stability=float(row[stability_col]) if stability_col in row.index else None,
                        )

            config_values = {}
            config_cols = [c for c in row.index if not any(c.startswith(p) for p in ["metrics.", "__"])]
            for col in config_cols:
                if pd.notna(row[col]):
                    config_values[col] = row[col]

            config_summaries.append(
                ConfigSummary(
                    signature=row["__signature"],
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