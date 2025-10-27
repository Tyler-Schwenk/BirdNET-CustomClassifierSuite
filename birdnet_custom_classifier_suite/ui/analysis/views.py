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
try:
    from st_aggrid import AgGrid, GridOptionsBuilder, JsCode, GridUpdateMode
    HAS_AGGRID = True
except Exception:
    HAS_AGGRID = False

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

    This renders an interactive table using AgGrid when available. Selecting
    a row will call `on_select(signature)` so the parent UI can show details.
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
            short_name = name.replace("metrics.", "").replace(".best_f1", "")
            # Format as 'mean ± std' with 3 decimal places for display
            try:
                mean = float(metric.mean)
            except Exception:
                mean = None
            try:
                sd = float(metric.std) if hasattr(metric, 'std') else None
            except Exception:
                sd = None
            if mean is None:
                display_val = "n/a"
            else:
                if sd is None or pd.isna(sd):
                    display_val = f"{mean:.3f}"
                else:
                    display_val = f"{mean:.3f} ± {sd:.3f}"
            row[short_name] = display_val
        rows.append(row)

    df = pd.DataFrame(rows)

    # Try to use AgGrid for a nicer interactive table
    if HAS_AGGRID:
    # We'll create numeric mean/std columns for each metric so AG Grid can sort numerically
    # while using a JS formatter to display "mean ± std".

        # Identify metric columns by those not Signature/Experiments
        metric_cols = [c for c in df.columns if c not in ('Signature', 'Experiments')]
        for mc in metric_cols:
            # Our df currently contains formatted strings; try to parse numeric mean and std
            # If the value is like 'mean ± sd', split it; else treat as numeric single value
            mean_col = f"{mc}__mean"
            std_col = f"{mc}__std"

            def parse_vals(val):
                if isinstance(val, str) and '±' in val:
                    parts = [p.strip() for p in val.split('±')]
                    try:
                        return float(parts[0]), float(parts[1]) if len(parts) > 1 else None
                    except Exception:
                        return (None, None)
                else:
                    try:
                        return float(val), None
                    except Exception:
                        return (None, None)

            parsed = df[mc].apply(parse_vals)
            df[mean_col] = parsed.apply(lambda x: x[0])
            df[std_col] = parsed.apply(lambda x: x[1])

        # Remove the original text columns so the grid shows numeric mean/std (we'll format mean via JS)
        df_display = df.drop(columns=metric_cols)

        gb = GridOptionsBuilder.from_dataframe(df_display)
        # Configure selection to work in both older and newer AG Grid versions:
        # - Older versions use selection_mode='single'
        # - Newer prefer {'type': 'single', 'mode': 'singleRow'}
        gb.configure_selection(
            selection_mode='single',
            use_checkbox=False,
            pre_selected_rows=[],  # start with no selection
            rowMultiSelectWithClick=False,  # ensure single selection mode
            suppressRowDeselection=False,  # allow deselect by clicking again
        )
        # Standard column setup
        gb.configure_default_column(minWidth=120, sortable=True, filter=True)
        gb.configure_column('Signature', header_name='Signature', minWidth=220, sortable=True, filter=True)
        gb.configure_column('Experiments', header_name='Experiments', minWidth=300, sortable=True, filter=True)

        # Add formatted mean columns and hidden std columns
        for mc in metric_cols:
            mean_col = f"{mc}__mean"
            std_col = f"{mc}__std"
            # valueFormatter JS to show mean ± sd with 3 decimals
            js = JsCode(
                "function(params) { var v = params.value; var sd = params.data['" + std_col + "']; if (v===null || v===undefined || isNaN(v)) { return 'n/a'; } var s = v.toFixed(3); if (sd!==null && sd!==undefined && !isNaN(sd)) { s += ' ± ' + sd.toFixed(3); } return s; }"
            )
            gb.configure_column(mean_col, header_name=mc.replace('__mean',''), valueFormatter=js, type=['numericColumn'], minWidth=120)
            gb.configure_column(std_col, header_name=f"{mc} sd", minWidth=80, hide=True)

        st.write("Click a row to see details:")
        grid_options = gb.build()

        # Remove known deprecated top-level flags to reduce AG Grid console warnings
        for deprecated in [
            'rowMultiSelectWithClick',
            'suppressRowDeselection',
            'suppressRowClickSelection',
            'groupSelectsChildren',
            'groupSelectsFiltered',
        ]:
            if deprecated in grid_options:
                try:
                    grid_options.pop(deprecated)
                except Exception:
                    pass

    # Stable key prevents re-registration warnings across reruns
        ag_key = 'leaderboard_ag'

        # Use signature as the row id to avoid auto-hash ids which can change
        # between reruns and cause selection to fail. This requires the
        # 'Signature' column to exist in the DataFrame (it does in df_display).
        try:
            grid_options['getRowId'] = JsCode("function(params) { return params.data.Signature; }")
        except Exception:
            # If JsCode isn't available for any reason, skip this step.
            pass

        grid_response = AgGrid(
            df_display,
            gridOptions=grid_options,
            update_mode=GridUpdateMode.SELECTION_CHANGED if HAS_AGGRID else 'selection_changed',
            theme='streamlit',
            height=min(35 * (len(df_display) + 1), 400),
            key=ag_key,
            allow_unsafe_jscode=True,
        )

        # Optional debug dump of the raw AgGrid response to help diagnose selection
        # payload shapes. Turn on from the sidebar if you need to inspect what
        # the component returns when rows are clicked.
        # Debug output with a unique checkbox key
        show_debug = st.sidebar.checkbox(
            'Show AgGrid response',
            value=False,
            key='aggrid_debug_checkbox'
        )
        
        try:
            if show_debug:
                # Convert any DataFrames to records for better debug visibility
                data = getattr(grid_response, 'data', None)
                if isinstance(data, pd.DataFrame):
                    data = data.to_dict('records')
                
                selected_rows = getattr(grid_response, 'selected_rows', None)
                if isinstance(selected_rows, pd.DataFrame):
                    selected_rows = selected_rows.to_dict('records')
                
                debug_info = {
                    'response_type': str(type(grid_response)),
                    'data': data,
                    'selected_rows': selected_rows,
                    'selected': getattr(grid_response, 'selected', None),
                    'selection_changed': getattr(grid_response, 'selection_changed', None),
                }
                st.write("AgGrid Response Debug:")
                st.json(debug_info)
        except Exception as e:
            st.sidebar.error(f"Debug info error: {str(e)}")
            pass

        # Handle selection using the observed format: selected_rows as list of dicts
        if hasattr(grid_response, 'selected_rows') and on_select:
            selected = grid_response.selected_rows
            
            # Convert DataFrame to list of dicts if needed
            if isinstance(selected, pd.DataFrame):
                selected = selected.to_dict('records')
            
            # Handle list format (what we see in the debug output)
            if isinstance(selected, list) and len(selected) > 0:
                first_row = selected[0]
                if isinstance(first_row, dict) and 'Signature' in first_row:
                    sig = first_row['Signature']
                    if sig and not pd.isna(sig):
                        on_select(sig)
                        if show_debug:
                            st.sidebar.write(f"Selected signature: {sig}")
    else:
        # Fallback to simple display + selectbox
        st.write("Click a signature to see details:")
        st.dataframe(df, use_container_width=True, height=min(35 * (len(df) + 1), 400))
        if on_select:
            selected = st.selectbox("Or select a signature here:", options=df['Signature'].tolist())
            if selected:
                on_select(selected)


def signature_details(breakdown: Optional[PerRunBreakdown]) -> None:
    """Render detailed information for a configuration signature.

    Presents an aggregate metric card view, a compact configuration summary,
    and an expandable per-run table with cleaned column names.
    """
    if not breakdown:
        st.info("No details available for this signature.")
        return

    st.divider()
    st.subheader(f"Configuration Details — {breakdown.signature}")

    # Metrics aggregate cards
    if breakdown.aggregates:
        st.write("### Metrics Summary")
        agg_items = list(breakdown.aggregates.items())
        cols = st.columns(min(len(agg_items), 3))
        for i, (name, (mn, sd)) in enumerate(agg_items):
            with cols[i % len(cols)]:
                display_name = name.replace("metrics.", "").replace(".best_f1", "").replace("_", " ").title()
                value = f"{mn:.4f}" if (mn is not None and not pd.isna(mn)) else "n/a"
                delta = f"±{sd:.4f}" if (sd is not None and not pd.isna(sd)) else None
                st.metric(label=display_name, value=value, delta=delta)

    # Configuration summary
    if breakdown.config_columns:
        st.write("### Configuration")
        config_data = {}
        for col in breakdown.config_columns:
            vals = breakdown.rows[col].dropna().unique()
            if len(vals) == 1:
                config_data[col] = vals[0]
            elif len(vals) > 1:
                config_data[col] = f"Multiple: {', '.join(map(str, vals))}"
            else:
                config_data[col] = "n/a"

        # Two-column layout for config key-values
        cfg_cols = st.columns(2)
        for i, (k, v) in enumerate(config_data.items()):
            with cfg_cols[i % 2]:
                st.write(f"**{k}**: {v}")

    # Per-run table in expander
    with st.expander("View individual runs", expanded=False):
        if not breakdown.rows.empty:
            show_cols = ["experiment.name"] + breakdown.metric_columns + breakdown.config_columns
            show_cols = [c for c in show_cols if c in breakdown.rows.columns]
            display = breakdown.rows[show_cols].copy()

            # Clean metric column names and round numeric values
            rename_map = {}
            for m in breakdown.metric_columns:
                if m in display.columns:
                    try:
                        display[m] = display[m].astype(float).round(4)
                    except Exception:
                        pass
                    pretty = m.replace("metrics.", "").replace(".best_f1", "").replace("_", " ").title()
                    rename_map[m] = pretty

            display = display.rename(columns=rename_map).rename(columns={"experiment.name": "Experiment"})
            st.dataframe(display, use_container_width=True, height=min(35 * (len(display) + 1), 400))
        else:
            st.info("No individual runs found for this signature.")