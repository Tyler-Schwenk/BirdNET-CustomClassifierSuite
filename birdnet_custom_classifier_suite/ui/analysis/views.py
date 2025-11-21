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
    NEW_RESULTS_PATH,
    MetricGroup,
    PerRunBreakdown,
    UIState,
)


def _dataframe_full_width(data: pd.DataFrame, height: int | None = None):
    """Render a dataframe using the new Streamlit width API when available,
    with graceful fallback for older versions.

    - Preferred: width="stretch" (new API) to stretch to the container width
    - Fallback: use_container_width=True (deprecated in newer versions)
    """
    try:
        # Newer Streamlit versions deprecate use_container_width in favor of width
        # Valid width values: integer pixels, 'stretch', or 'content'.
        st.dataframe(data, width="stretch", height=height)
    except TypeError:
        # Older versions don't accept width keyword
        st.dataframe(data, use_container_width=True, height=height)


def data_loader(state: UIState,
                on_load: Optional[Callable[[pd.DataFrame], None]] = None) -> None:
    """Render the data source selector and load button.
    
    Args:
        state: Current UI state
        on_load: Optional callback when data is loaded
    """
    with st.sidebar.expander("Load data", expanded=True):
        st.write("Choose a source for experiment results (CSV):")
        # Persist the toggle across reruns
        use_default_key = 'use_default_path_toggle'
        use_default = st.checkbox(
            "Use default path",
            value=st.session_state.get(use_default_key, True),
            key=use_default_key,
        )

        # Discover available default files (prefer Original by default)
        available_defaults = []
        if DEFAULT_RESULTS_PATH.exists():
            available_defaults.append(("Original", DEFAULT_RESULTS_PATH))
        if NEW_RESULTS_PATH.exists():
            available_defaults.append(("New (re-evaluated)", NEW_RESULTS_PATH))

        selected_default_idx = 0
        if use_default and available_defaults:
            # Persist selection across reruns
            sel_key = 'default_results_choice_idx'
            # If user previously selected, use that; else prefer the first (which is NEW if present)
            prev_idx = st.session_state.get(sel_key, None)
            labels = [f"{lbl} — {path.name}" for (lbl, path) in available_defaults]
            # Default to the index of DEFAULT_RESULTS_PATH when first shown
            default_idx = next((i for i, (_, p) in enumerate(available_defaults) if p == DEFAULT_RESULTS_PATH), 0)
            selected_default_idx = st.selectbox(
                "Default source",
                options=list(range(len(labels))),
                format_func=lambda i: labels[i],
                index=min(prev_idx if prev_idx is not None else default_idx, len(labels)-1),
            )
            st.session_state[sel_key] = selected_default_idx
            state.data_source = available_defaults[selected_default_idx][1]
        else:
            uploaded = st.file_uploader("Or upload a CSV file", type=["csv"]) if not use_default else None
            custom_path = st.text_input(
                "Custom path (if not using default)",
                value="",
            )
            state.data_source = Path(custom_path.strip()) if custom_path.strip() else None

        run_btn = st.button("Load Data")

        if run_btn:
            try:
                # Validate inputs when not using default
                if not use_default:
                    if uploaded is None and state.data_source is None:
                        st.warning("Provide a custom path or upload a CSV file.")
                        return
                    if state.data_source is not None and not state.data_source.exists():
                        st.error(f"File not found: {state.data_source}")
                        return

                state.results_df = load_results(
                    path=state.data_source,
                    uploaded_file=uploaded if not use_default else None,
                )
                src_str = str(state.data_source) if state.data_source else (uploaded.name if uploaded else "uploaded file")
                st.success(f"Loaded {len(state.results_df)} rows from: {src_str}")
                if on_load:
                    on_load(state.results_df)
            except Exception as e:
                st.error(f"Failed to load results: {e}")
def data_loader(state: UIState, container=None) -> None:
    """Inline loader for results CSVs. Lists results/*.csv and supports uploads.

    Args:
        state: UI state
        container: Optional Streamlit container/column to render into. Defaults to main area.
    """
    import streamlit as st
    from pathlib import Path
    panel = container if container is not None else st
    results_dir = Path('results')
    # List all CSVs in results/
    csv_files = sorted([f for f in results_dir.glob('*.csv') if f.is_file()])
    panel.markdown("**Select results file**")
    file_options = {str(f): f for f in csv_files}
    if not file_options:
        panel.warning("No CSV files found in results/.")
        return
    selected_file = panel.selectbox(
        "Results file",
        options=list(file_options.keys()),
        index=0,
    )
    # Custom file upload
    uploaded_file = panel.file_uploader("Or upload a CSV", type=["csv"])
    # Load logic
    load_path = None
    if uploaded_file is not None:
        import pandas as pd
        try:
            df = pd.read_csv(uploaded_file)
            state.results_df = df
            state.data_source = uploaded_file.name
            panel.success(f"✓ Loaded {len(df)} rows from uploaded file: {uploaded_file.name}")
            return
        except Exception as e:
            panel.error(f"Failed to load uploaded CSV: {e}")
            return
    else:
        load_path = file_options[selected_file]
        import pandas as pd
        try:
            df = pd.read_csv(load_path)
            state.results_df = df
            state.data_source = load_path
            panel.success(f"✓ Loaded {len(df)} rows from {load_path}")
        except Exception as e:
            panel.error(f"Failed to load selected CSV: {e}")


def metric_controls(state: UIState, df: Optional[pd.DataFrame] = None, container=None) -> None:
    """Render metric selection and filtering controls.
    
    Args:
        state: Current UI state
    """
    import streamlit as st
    parent = container if container is not None else st
    # Render expander within the provided container, using st.expander to ensure compatibility
    with parent:
        with st.expander("Analysis Options", expanded=True):
            state.metric_prefix = st.selectbox(
            "Metric group",
            options=[m.value for m in MetricGroup],
            index=0,
        )
            state.top_n = st.number_input(
            "Top N",
            min_value=0,
            max_value=200,
            value=state.top_n,
            step=1,
            help="Set to 0 to show all results (no Top N limit)."
        )
            precision_input = st.number_input(
            "Precision floor (0-1 or percent)",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=0.01,
        )
            state.precision_floor = precision_input if precision_input > 0 else None

        # Optional: dataset filter controls
        if df is not None and not df.empty:
            st.markdown("---")
            st.write("Filters")

            def _find_col(columns, prefer: List[str]) -> Optional[str]:
                # exact match first
                for p in prefer:
                    if p in columns:
                        return p
                # endswith match
                for p in prefer:
                    for c in columns:
                        if c.endswith(p):
                            return c
                # loose contains
                for p in prefer:
                    for c in columns:
                        if p in c:
                            return c
                return None

            cols = list(df.columns)
            quality_col = _find_col(cols, [
                'dataset.filters.quality', 'filters.quality', '.quality', 'quality'
            ])
            balance_col = _find_col(cols, [
                'dataset.filters.balance', 'filters.balance', '.balance', 'balance'
            ])

            if quality_col:
                q_vals = sorted(pd.Series(df[quality_col].dropna().unique()).tolist())
                selected = st.multiselect(
                    f"Quality ({quality_col})",
                    options=q_vals,
                    default=state.quality_filter or []
                )
                state.quality_filter = selected if selected else None

            if balance_col:
                b_vals = sorted(pd.Series(df[balance_col].dropna().unique()).tolist())
                selected = st.multiselect(
                    f"Balance ({balance_col})",
                    options=b_vals,
                    default=state.balance_filter or []
                )
                state.balance_filter = selected if selected else None

            # Extract stage/sweep from experiment.name
            if 'experiment.name' in cols:
                def extract_stage(name):
                    """Extract stage prefix like 'stage3', 'stage3b', 'stage1_sweep' from experiment name."""
                    import re
                    name_lower = str(name).lower()
                    # 1) Standard pattern: 'stage{num}{opt_letter}' and optional '_sweep'
                    m = re.search(r'stage\d+[a-z]*(?:_sweep)?', name_lower)
                    if m:
                        return m.group(0)
                    # 2) Legacy/Stage0 patterns: names like '000_OldModel2024', 'stage0_*', or leading zeros
                    if name_lower.startswith('stage0'):
                        return 'stage0'
                    if re.match(r'^(?:0+[_-]|0+$)', name_lower) or 'oldmodel' in name_lower:
                        return 'stage0'
                    return None
                
                stages = df['experiment.name'].apply(extract_stage).dropna().unique()
                if len(stages) > 0:
                    stage_vals = sorted(stages.tolist())
                    selected = st.multiselect(
                        "Stage/Sweep",
                        options=stage_vals,
                        default=state.sweep_filter or []
                    )
                    state.sweep_filter = selected if selected else None


def leaderboard(summaries: List[ConfigSummary],
                on_select: Optional[Callable[[str], None]] = None,
                show_aggrid_debug: bool = False,
                debug_container=None):
    """Render the leaderboard table with top configurations.

    This renders an interactive table using AgGrid when available. Selecting
    a row will call `on_select(signature)` so the parent UI can show details.
    """
    if not summaries:
        st.info("No configurations to show.")
        return

    # Optional config column controls
    st.write("**Show additional columns:**")
    cols_ctrl = st.columns(12)
    with cols_ctrl[0]:
        show_quality = st.checkbox("Quality", value=False, key='lb_quality')
    with cols_ctrl[1]:
        show_balance = st.checkbox("Balance", value=False, key='lb_balance')
    with cols_ctrl[2]:
        show_upsampling = st.checkbox("Upsampling", value=False, key='lb_upsampling')
    with cols_ctrl[3]:
        show_mixup = st.checkbox("Mixup", value=False, key='lb_mixup')
    with cols_ctrl[4]:
        show_label_smoothing = st.checkbox("Label Smoothing", value=False, key='lb_label_smoothing')
    with cols_ctrl[5]:
        show_focal_loss = st.checkbox("Focal Loss", value=False, key='lb_focal_loss')
    with cols_ctrl[6]:
        show_hidden_units = st.checkbox("Hidden Units", value=False, key='lb_hidden_units')
    with cols_ctrl[7]:
        show_dropout = st.checkbox("Dropout", value=False, key='lb_dropout')
    with cols_ctrl[8]:
        show_learning_rate = st.checkbox("Learning Rate", value=False, key='lb_learning_rate')
    with cols_ctrl[9]:
        show_batch_size = st.checkbox("Batch Size", value=False, key='lb_batch_size')
    with cols_ctrl[10]:
        show_positive_subsets = st.checkbox("Positive Subsets", value=False, key='lb_positive_subsets')
    with cols_ctrl[11]:
        show_negative_subsets = st.checkbox("Negative Subsets", value=False, key='lb_negative_subsets')

    # Helper to format config values nicely
    def format_config_value(val):
        """Format config values for display."""
        if val is None or pd.isna(val):
            return 'n/a'
        # Parse string representations of lists
        if isinstance(val, str) and val.startswith('['):
            try:
                import ast
                parsed = ast.literal_eval(val)
                if isinstance(parsed, list):
                    return ', '.join(str(x) for x in parsed)
            except Exception:
                pass
        # Handle booleans
        if isinstance(val, bool):
            return 'Yes' if val else 'No'
        return str(val)

    # Build a clean DataFrame for display
    rows = []
    for summary in summaries:
        row = {
            "Signature": summary.signature,
            "Experiments": ", ".join(summary.experiment_names),
        }
        
        # Add optional config columns
        if show_quality:
            val = summary.config_values.get('dataset.filters.quality', 
                  summary.config_values.get('filters.quality'))
            row["Quality"] = format_config_value(val)
        if show_balance:
            val = summary.config_values.get('dataset.filters.balance',
                  summary.config_values.get('filters.balance'))
            row["Balance"] = format_config_value(val)
        if show_upsampling:
            # Look for upsampling mode and ratio
            mode = summary.config_values.get('training_args.upsampling_mode',
                   summary.config_values.get('training.upsampling.mode',
                   summary.config_values.get('upsampling.mode')))
            ratio = summary.config_values.get('training_args.upsampling_ratio',
                    summary.config_values.get('training.upsampling.ratio',
                    summary.config_values.get('upsampling.ratio',
                    summary.config_values.get('upsample_ratio'))))
            if mode or ratio:
                mode_str = format_config_value(mode) if mode else ''
                ratio_str = format_config_value(ratio) if ratio else ''
                row["Upsampling"] = f"{mode_str}/{ratio_str}" if mode_str and ratio_str else (mode_str or ratio_str or 'n/a')
            else:
                row["Upsampling"] = 'n/a'
        if show_mixup:
            val = summary.config_values.get('training_args.mixup',
                  summary.config_values.get('training.mixup',
                  summary.config_values.get('mixup')))
            row["Mixup"] = format_config_value(val)
        if show_label_smoothing:
            val = summary.config_values.get('training_args.label_smoothing',
                  summary.config_values.get('training.label_smoothing',
                  summary.config_values.get('training.label-smoothing',
                  summary.config_values.get('label_smoothing',
                  summary.config_values.get('label-smoothing')))))
            row["Label Smoothing"] = format_config_value(val)
        if show_focal_loss:
            val = summary.config_values.get('training_args.focal_loss',
                  summary.config_values.get('training_args.focal-loss',
                  summary.config_values.get('training.focal_loss',
                  summary.config_values.get('training.focal-loss',
                  summary.config_values.get('focal_loss',
                  summary.config_values.get('focal-loss'))))))
            row["Focal Loss"] = format_config_value(val)
        if show_hidden_units:
            val = summary.config_values.get('training_args.hidden_units',
                  summary.config_values.get('training.hidden_units'))
            row["Hidden Units"] = format_config_value(val)
        if show_dropout:
            val = summary.config_values.get('training_args.dropout',
                  summary.config_values.get('training.dropout'))
            row["Dropout"] = format_config_value(val)
        if show_learning_rate:
            val = summary.config_values.get('training_args.learning_rate',
                  summary.config_values.get('training.learning_rate'))
            row["Learning Rate"] = format_config_value(val)
        if show_batch_size:
            val = summary.config_values.get('training.batch_size',
                  summary.config_values.get('training_args.batch_size',
                  summary.config_values.get('batch_size')))
            row["Batch Size"] = format_config_value(val)
        if show_positive_subsets:
            val = summary.config_values.get('dataset.filters.positive_subsets',
                  summary.config_values.get('filters.positive_subsets',
                  summary.config_values.get('positive_subsets')))
            # Format list values nicely - show basename only
            if val and val != '[]':
                import ast
                try:
                    if isinstance(val, str):
                        parsed = ast.literal_eval(val)
                        subsets = parsed if isinstance(parsed, list) else []
                    else:
                        subsets = val if isinstance(val, list) else []
                    # Extract just the folder name from paths
                    subset_names = [s.split('\\')[-1] if '\\' in s else s.split('/')[-1] for s in subsets]
                    row["Positive Subsets"] = ', '.join(subset_names) if subset_names else 'none'
                except:
                    row["Positive Subsets"] = str(val)
            else:
                row["Positive Subsets"] = 'none'
        if show_negative_subsets:
            val = summary.config_values.get('dataset.filters.negative_subsets',
                  summary.config_values.get('filters.negative_subsets',
                  summary.config_values.get('negative_subsets')))
            # Format list values nicely - show basename only
            if val and val != '[]':
                import ast
                try:
                    if isinstance(val, str):
                        parsed = ast.literal_eval(val)
                        subsets = parsed if isinstance(parsed, list) else []
                    else:
                        subsets = val if isinstance(val, list) else []
                    # Extract just the folder name from paths
                    subset_names = [s.split('\\')[-1] if '\\' in s else s.split('/')[-1] for s in subsets]
                    row["Negative Subsets"] = ', '.join(subset_names) if subset_names else 'none'
                except:
                    row["Negative Subsets"] = str(val)
            else:
                row["Negative Subsets"] = 'none'
        
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

        # Identify config columns (the ones we added with checkboxes)
        config_column_names = []
        if show_quality:
            config_column_names.append('Quality')
        if show_balance:
            config_column_names.append('Balance')
        if show_upsampling:
            config_column_names.append('Upsampling')
        if show_mixup:
            config_column_names.append('Mixup')
        if show_label_smoothing:
            config_column_names.append('Label Smoothing')
        if show_focal_loss:
            config_column_names.append('Focal Loss')
        if show_hidden_units:
            config_column_names.append('Hidden Units')
        if show_dropout:
            config_column_names.append('Dropout')
        if show_learning_rate:
            config_column_names.append('Learning Rate')
        if show_batch_size:
            config_column_names.append('Batch Size')
        if show_positive_subsets:
            config_column_names.append('Positive Subsets')
        if show_negative_subsets:
            config_column_names.append('Negative Subsets')

        # Identify metric columns by those not Signature/Experiments/Config columns
        metric_cols = [c for c in df.columns if c not in (['Signature', 'Experiments'] + config_column_names)]
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

        # Remove the original metric text columns (but keep config columns as-is)
        df_display = df.drop(columns=metric_cols)

        gb = GridOptionsBuilder.from_dataframe(df_display)
        # Configure selection using AG Grid v32+ API (object-based)
        gb.configure_selection(
            selection_mode='single',
            use_checkbox=False,
            pre_selected_rows=[],
        )
        # Standard column setup
        gb.configure_default_column(minWidth=60, sortable=True, filter=True)
        gb.configure_column('Signature', header_name='Signature', minWidth=60, sortable=True, filter=True)
        gb.configure_column('Experiments', header_name='Experiments', minWidth=80, sortable=True, filter=True)
        
        # Configure optional config columns with narrower widths
        if show_quality and 'Quality' in df_display.columns:
            gb.configure_column('Quality', minWidth=80, width=100)
        if show_balance and 'Balance' in df_display.columns:
            gb.configure_column('Balance', minWidth=80, width=120)
        if show_upsampling and 'Upsampling' in df_display.columns:
            gb.configure_column('Upsampling', minWidth=100, width=140)
        if show_mixup and 'Mixup' in df_display.columns:
            gb.configure_column('Mixup', minWidth=70, width=90)
        if show_label_smoothing and 'Label Smoothing' in df_display.columns:
            gb.configure_column('Label Smoothing', minWidth=100, width=140)
        if show_focal_loss and 'Focal Loss' in df_display.columns:
            gb.configure_column('Focal Loss', minWidth=80, width=110)
        if show_hidden_units and 'Hidden Units' in df_display.columns:
            gb.configure_column('Hidden Units', minWidth=100, width=120)
        if show_dropout and 'Dropout' in df_display.columns:
            gb.configure_column('Dropout', minWidth=90, width=110)
        if show_learning_rate and 'Learning Rate' in df_display.columns:
            gb.configure_column('Learning Rate', minWidth=120, width=140)
        if show_batch_size and 'Batch Size' in df_display.columns:
            gb.configure_column('Batch Size', minWidth=90, width=110)

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

        # Update rowSelection to use AG Grid v32+ object format
        if 'rowSelection' in grid_options:
            if grid_options['rowSelection'] in ('single', 'multiple'):
                # Convert deprecated string format to new object format
                grid_options['rowSelection'] = {
                    'mode': 'singleRow' if grid_options['rowSelection'] == 'single' else 'multiRow',
                    'checkboxes': False,
                    'enableClickSelection': True,
                }

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
        # payload shapes. Controlled by a checkbox rendered by the caller (e.g., Evaluate controls panel).
        panel = debug_container if debug_container is not None else st
        show_debug = bool(show_aggrid_debug)
        
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
                panel.write("AgGrid Response Debug:")
                panel.json(debug_info)
        except Exception as e:
            panel.error(f"Debug info error: {str(e)}")
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
                            panel.write(f"Selected signature: {sig}")
        # Return the current display DataFrame for optional download by caller
        return df_display
    else:
        # Fallback to simple display + selectbox
        st.write("Click a signature to see details:")
        _dataframe_full_width(df, height=min(35 * (len(df) + 1), 400))
        if on_select:
            selected = st.selectbox("Or select a signature here:", options=df['Signature'].tolist())
            if selected:
                on_select(selected)
        return df


def signature_details(breakdown: Optional[PerRunBreakdown]) -> None:
    """Render detailed information for a configuration signature.

    Presents an aggregate metric card view, a compact configuration summary
    organized by category with non-default values highlighted, and an
    expandable per-run table with cleaned column names.
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
                delta = f"STD ±{sd:.4f}" if (sd is not None and not pd.isna(sd)) else None
                st.metric(label=display_name, value=value, delta=delta, delta_color="off")

    # Configuration summary - categorized and prioritized
    if breakdown.config_columns:
        from birdnet_custom_classifier_suite.ui.common.config_defaults import (
            get_category,
            is_default_value,
        )
        
        # Helper: build unique, human-readable labels for keys within a category
        def _build_pretty_labels(items: dict) -> dict:
            def base_label(key: str) -> str:
                parts = key.split('.')
                tail = parts[-1]
                return tail.replace('_', ' ').replace('-', ' ')

            # Count base labels to detect collisions
            counts = {}
            for k in items.keys():
                lbl = base_label(k)
                counts[lbl] = counts.get(lbl, 0) + 1

            # Disambiguate duplicates by prefixing with previous segment
            labels = {}
            for k in items.keys():
                lbl = base_label(k)
                if counts[lbl] == 1:
                    labels[k] = lbl
                else:
                    parts = k.split('.')
                    if len(parts) >= 2:
                        prev = parts[-2].replace('_', ' ').replace('-', ' ')
                        labels[k] = f"{prev} {lbl}"
                    else:
                        labels[k] = lbl
            return labels
        
        st.write("### Configuration")
        
        # Collect config values
        config_data = {}
        for col in breakdown.config_columns:
            # Skip internal/meta columns
            if col.startswith('__') or col in ('experiment.name', 'experiment.names'):
                continue
                
            vals = breakdown.rows[col].dropna().unique()
            if len(vals) == 1:
                config_data[col] = vals[0]
            elif len(vals) > 1:
                config_data[col] = f"Multiple: {', '.join(map(str, vals))}"
        
        # Categorize and check defaults
        categorized = {}
        non_defaults = {}
        defaults = {}
        
        for key, value in config_data.items():
            category = get_category(key)
            if category not in categorized:
                categorized[category] = {}
            categorized[category][key] = value
            
            # Check if this is a non-default value
            if is_default_value(key, value):
                if category not in defaults:
                    defaults[category] = {}
                defaults[category][key] = value
            else:
                if category not in non_defaults:
                    non_defaults[category] = {}
                non_defaults[category][key] = value
        
        # Demote some keys into "Additional settings" regardless of defaultness
        def _is_additional_key(k: str) -> bool:
            kl = k.lower()
            return (
                'manifest' in kl
                or kl.endswith('.splits') or kl.endswith('splits')
                or kl.endswith('.threads') or kl.endswith('threads')
                or 'source.root' in kl or 'source_root' in kl
            )

        additional = {}

        def _move_into_additional(bucket: dict):
            to_move = []
            for cat, items in bucket.items():
                for k in items.keys():
                    if _is_additional_key(k):
                        to_move.append((cat, k))
            for cat, k in to_move:
                additional.setdefault(cat, {})[k] = bucket[cat].pop(k)
                if not bucket[cat]:
                    bucket.pop(cat, None)

        _move_into_additional(non_defaults)
        _move_into_additional(defaults)

        # Collect dataset detail items (condense into their own expander)
        dataset_detail_tokens_key = {'negative', 'growl', 'grunt', 'both', 'high', 'low', 'medium', 'pos', 'neg'}
        dataset_detail_tokens_val = {'negative total', 'growl', 'grunt', 'both', 'high', 'low', 'medium', 'pos', 'neg'}

        dataset_details = {}

        def _is_dataset_detail(cat: str, k: str, v: any) -> bool:
            if cat != 'dataset':
                return False
            ks = k.lower()
            vs = str(v).lower()
            return (
                any(tok in ks for tok in dataset_detail_tokens_key)
                or vs in dataset_detail_tokens_val
            )

        def _extract_dataset_details(bucket: dict):
            to_move = []
            for cat, items in bucket.items():
                for k, v in items.items():
                    if _is_dataset_detail(cat, k, v):
                        to_move.append((cat, k))
                        dataset_details[k] = v
            for cat, k in to_move:
                bucket[cat].pop(k, None)
                if not bucket[cat]:
                    bucket.pop(cat, None)

        _extract_dataset_details(non_defaults)
        _extract_dataset_details(defaults)
        _extract_dataset_details(additional)

        # Prepare pretty labels per category using combined keys (defaults + non-defaults)
        category_label_maps = {}
        for category in set(list(non_defaults.keys()) + list(defaults.keys()) + list(additional.keys())):
            combined = {}
            if category in non_defaults:
                combined.update(non_defaults[category])
            if category in defaults:
                combined.update(defaults[category])
            if category in additional:
                combined.update(additional[category])
            category_label_maps[category] = _build_pretty_labels(combined)

        # Display non-default configs prominently
        if non_defaults:
            for category in sorted(non_defaults.keys()):
                st.write(f"**{category.replace('_', ' ').title()}**")
                cfg_cols = st.columns(2)
                label_map = category_label_maps.get(category, {})
                for i, (k, v) in enumerate(sorted(non_defaults[category].items())):
                    with cfg_cols[i % 2]:
                        pretty = label_map.get(k, k)
                        st.write(f"• **{pretty}**: `{v}`")
        
        # Dataset details expander (condensed)
        if dataset_details:
            with st.expander("Dataset details", expanded=False):
                # Build labels from the dataset_details key set
                label_map_ds = _build_pretty_labels(dataset_details)
                cfg_cols = st.columns(2)
                for i, (k, v) in enumerate(sorted(dataset_details.items())):
                    with cfg_cols[i % 2]:
                        pretty = label_map_ds.get(k, k)
                        st.write(f"• {pretty}: {v}")

        # Additional settings expander: includes demoted keys and defaults
        if additional or defaults:
            with st.expander("Additional settings", expanded=False):
                # First, show demoted keys (additional)
                for category in sorted(additional.keys()):
                    st.write(f"**{category.replace('_', ' ').title()}**")
                    cfg_cols = st.columns(2)
                    label_map = category_label_maps.get(category, {})
                    for i, (k, v) in enumerate(sorted(additional[category].items())):
                        with cfg_cols[i % 2]:
                            pretty = label_map.get(k, k)
                            st.write(f"• {pretty}: {v}")

                # Then, show remaining defaults
                for category in sorted(defaults.keys()):
                    st.write(f"**{category.replace('_', ' ').title()}**")
                    cfg_cols = st.columns(2)
                    label_map = category_label_maps.get(category, {})
                    for i, (k, v) in enumerate(sorted(defaults[category].items())):
                        with cfg_cols[i % 2]:
                            pretty = label_map.get(k, k)
                            st.write(f"• {pretty}: {v}")

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
            _dataframe_full_width(display, height=min(35 * (len(display) + 1), 400))
        else:
            st.info("No individual runs found for this signature.")