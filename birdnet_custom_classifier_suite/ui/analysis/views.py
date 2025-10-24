"""
Reusable UI components for experiment analysis.

This module provides Streamlit-based UI components that can be composed
into larger applications. Components are designed to be stateless and
reusable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd
import streamlit as st

from birdnet_custom_classifier_suite.ui.analysis.data import load_results
from birdnet_custom_classifier_suite.ui.analysis.metrics import (
    format_metric_value,
    get_signature_breakdown,
    summarize_metrics,
)
from birdnet_custom_classifier_suite.ui.common.types import (
    ConfigSummary,
    DEFAULT_RESULTS_PATH,
    MetricGroup,
    PerRunBreakdown,
    UIState,
)


def data_loader(state: UIState,
                on_load: Optional[Callable[[pd.DataFrame], None]] = None) -> None:
    """Render the data source selector and load button.
    
    Args:
        state: Current UI state
        on_load: Optional callback when data is loaded
    """
    with st.sidebar.expander("Load data", expanded=True):
        st.write("Choose a source for experiment results (CSV):")
        use_default = st.checkbox("Use default path", value=True)
        uploaded = st.file_uploader("Or upload a CSV file", type=["csv"]) if not use_default else None
        custom_path = st.text_input(
            "Custom path (if not using default)",
            value=str(DEFAULT_RESULTS_PATH) if use_default else "",
        )

        if use_default:
            state.data_source = DEFAULT_RESULTS_PATH
        else:
            state.data_source = Path(custom_path.strip()) if custom_path.strip() else None

        run_btn = st.button("Load Data")

        if run_btn:
            try:
                state.results_df = load_results(
                    path=state.data_source,
                    uploaded_file=uploaded,
                )
                st.success(f"Loaded {len(state.results_df)} rows from results source.")
                if on_load:
                    on_load(state.results_df)
            except Exception as e:
                st.error(f"Failed to load results: {e}")


def metric_controls(state: UIState) -> None:
    """Render metric selection and filtering controls.
    
    Args:
        state: Current UI state
    """
    with st.sidebar.expander("Analysis Options", expanded=True):
        state.metric_prefix = st.selectbox(
            "Metric group",
            options=[m.value for m in MetricGroup],
            index=0,
        )
        state.top_n = st.number_input(
            "Top N",
            min_value=1,
            max_value=200,
            value=state.top_n,
            step=1,
        )
        precision_input = st.number_input(
            "Precision floor (0-1 or percent)",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=0.01,
        )
        state.precision_floor = precision_input if precision_input > 0 else None


def leaderboard(summaries: List[ConfigSummary],
                on_select: Optional[Callable[[str], None]] = None) -> None:
    """Render the leaderboard table with top configurations.
    
    Args:
        summaries: List of configuration summaries to display
        on_select: Optional callback when a signature is selected
    """
    if not summaries:
        st.info("No configurations to show.")
        return

    # Build a clean DataFrame for display
    rows = []
    for summary in summaries:
        row = {
            "Signature": summary.signature,
            "Experiments": ", ".join(summary.experiment_names),
        }
        for name, metric in summary.metrics.items():
            short_name = name.replace("metrics.", "")
            row[f"{short_name}"] = format_metric_value(metric.mean, metric.std)
        rows.append(row)

    st.dataframe(pd.DataFrame(rows))

    if on_select:
        selected = st.selectbox(
            "Inspect signature:",
            options=[s.signature for s in summaries],
        )
        if selected:
            on_select(selected)


def signature_details(breakdown: Optional[PerRunBreakdown]) -> None:
    """Render detailed information for a configuration signature.
    
    Args:
        breakdown: Per-run breakdown to display
    """
    if not breakdown:
        st.info("No details available for this signature.")
        return

    st.subheader(f"Details — Signature {breakdown.signature}")

    # Format the per-run table
    if not breakdown.rows.empty:
        show_cols = (
            ["experiment.name"]
            + breakdown.metric_columns
            + breakdown.config_columns
        )
        show_cols = [c for c in show_cols if c in breakdown.rows.columns]
        
        display = breakdown.rows[show_cols].copy()
        # Round metrics for display
        for m in breakdown.metric_columns:
            if m in display.columns:
                display[m] = display[m].astype(float).round(4)
        
        st.table(display)

        # Show aggregates
        st.markdown("**Aggregated (mean ± std)**")
        agg_lines = []
        for m, (mn, sd) in breakdown.aggregates.items():
            agg_lines.append(f"- {m}: **{format_metric_value(mn)}** ± {format_metric_value(sd)}")
        st.markdown("\n".join(agg_lines))

    else:
        st.info("No runs found for this signature.")