# Advanced plotting tools for BirdNET Custom Classifier Suite
# Multi-metric visualization with full customization

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Import helper functions from main plots module
from birdnet_custom_classifier_suite.ui.analysis.plots import (
    list_metric_choices,
    list_param_choices,
    _pretty_metric_label,
    _is_numeric_series,
)


def advanced_plot_controls(table_df: pd.DataFrame, full_df: pd.DataFrame, container=None) -> Optional[Dict]:
    """Advanced plotting with multi-metric support, individual points, and customization.
    
    Args:
        table_df: Aggregated summary table (with __mean/__std columns)
        full_df: Raw experiment data (individual runs for overlay)
        container: Streamlit container to render in
        
    Returns:
        Dict with plot configuration or None if plotting not possible
    """
    panel = container if container is not None else st
    
    with panel.expander("Advanced Graph Tools", expanded=False):
        st.markdown("**Multi-metric visualization with full customization**")
        
        # Get available metrics
        metric_choices = list_metric_choices(table_df)
        if not metric_choices:
            st.info("No metrics available. Run analysis first.")
            return None
            
        x_candidates = list_param_choices(table_df)
        if not x_candidates:
            st.info("No parameter columns available.")
            return None
        
        # === Configuration Tabs ===
        tab1, tab2, tab3 = st.tabs(["Metrics & Axes", "Style", "Individual Points"])
        
        with tab1:
            st.subheader("Select Metrics to Plot")
            
            # Metric selection with checkboxes
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**OOD Metrics**")
                ood_metrics = [m for m in metric_choices if 'ood' in m[1].lower()]
                selected_ood = {}
                for label, col_name in ood_metrics:
                    if 'f1' in label.lower():
                        default = True
                    elif 'precision' in label.lower() or 'recall' in label.lower():
                        default = False
                    else:
                        default = False
                    selected_ood[col_name] = st.checkbox(label, value=default, key=f"adv_ood_{col_name}")
            
            with col2:
                st.write("**IID Metrics**")
                iid_metrics = [m for m in metric_choices if 'iid' in m[1].lower()]
                selected_iid = {}
                for label, col_name in iid_metrics:
                    selected_iid[col_name] = st.checkbox(label, value=False, key=f"adv_iid_{col_name}")
            
            # Combine selected metrics
            selected_metrics = {k: v for k, v in {**selected_ood, **selected_iid}.items() if v}
            
            if not selected_metrics:
                st.warning("WARNING: Select at least one metric to plot")
                return None
            
            st.divider()
            
            # X-axis selection
            x_col = st.selectbox("**X-axis (parameter)**", options=x_candidates, index=0, key="adv_x_col")
            
            # Check if x-axis is numeric
            is_numeric_x = _is_numeric_series(table_df[x_col].dropna())
            
            # X-axis type and plot type selection
            st.write("**Plot type**")
            
            if is_numeric_x:
                plot_type_option = st.radio(
                    "Choose plot style",
                    options=["Continuous (line plot)", "Categorical (bar plot)"],
                    index=0,
                    key="adv_x_axis_type",
                    help="Continuous: treats x as numeric scale. Categorical: treats each value as separate category."
                )
                use_continuous = plot_type_option.startswith("Continuous")
            else:
                plot_type_option = st.radio(
                    "Choose plot style",
                    options=["Line plot (ordered)", "Bar plot"],
                    index=1,
                    key="adv_x_axis_type",
                    help="Line: connects categories in order. Bar: shows each category separately."
                )
                use_continuous = plot_type_option.startswith("Line")
            
            # For numeric continuous, allow custom x-axis limits
            if is_numeric_x and use_continuous:
                auto_x_limits = st.checkbox("Auto x-axis limits", value=True, key="adv_auto_x")
                if not auto_x_limits:
                    x_vals = table_df[x_col].dropna()
                    x_min_default, x_max_default = float(x_vals.min()), float(x_vals.max())
                    x_limits = st.slider(
                        "X-axis range",
                        x_min_default, x_max_default,
                        (x_min_default, x_max_default),
                        key="adv_x_limits"
                    )
                else:
                    x_limits = None
                x_order = None
            else:
                # Categorical ordering (for both bar and line plots)
                x_limits = None
                unique_vals = sorted(table_df[x_col].dropna().unique(), key=float if is_numeric_x else str)
                st.write("**X-axis order** (drag to reorder)")
                x_order = st.multiselect(
                    "Categories (order matters)",
                    options=unique_vals,
                    default=unique_vals,
                    help="Select and reorder categories as they should appear left-to-right",
                    key="adv_x_order"
                )
        
        with tab2:
            st.subheader("Styling Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Plot type selection (line style options)
                if use_continuous:
                    plot_type = st.selectbox(
                        "Line style",
                        options=["Line plot", "Scatter plot", "Line + scatter"],
                        index=0,
                        key="adv_plot_type"
                    )
                else:
                    plot_type = "Bar plot"
                
                plot_style = st.selectbox(
                    "Plot style",
                    options=["whitegrid", "darkgrid", "white", "dark", "ticks"],
                    index=0,
                    key="adv_plot_style"
                )
                
                font_scale = st.slider("Font scale", 0.5, 2.0, 1.0, 0.1, key="adv_font_scale")
                
                if plot_type != "Bar plot":
                    show_error_bars = st.checkbox("Show error bands (±SD)", value=True, key="adv_error_bars",
                                                 help="Shows shaded area representing standard deviation")
                else:
                    show_error_bars = st.checkbox("Show error bars (±SD)", value=True, key="adv_error_bars")
                
            with col2:
                color_palette = st.selectbox(
                    "Color palette",
                    options=["Set2", "husl", "colorblind", "deep", "muted", "pastel", "bright", "dark"],
                    index=0,
                    key="adv_palette"
                )
                
                figure_width = st.slider("Figure width (inches)", 4, 20, 12, 1, key="adv_width")
                figure_height = st.slider("Figure height (inches)", 3, 12, 6, 1, key="adv_height")
                
                if plot_type != "Bar plot":
                    line_width = st.slider("Line width", 0.5, 5.0, 2.0, 0.5, key="adv_line_width")
                    if "scatter" in plot_type.lower():
                        marker_size = st.slider("Marker size", 1, 15, 6, 1, key="adv_marker_size")
                    else:
                        marker_size = 6
                else:
                    line_width = 2.0
                    marker_size = 6
            
            # Y-axis limits
            limit_y = st.checkbox("Limit Y-axis range", key="adv_limit_y")
            if limit_y:
                y_min, y_max = st.slider(
                    "Y-axis range",
                    0.0, 1.0, (0.0, 1.0), 0.01,
                    key="adv_y_range"
                )
            else:
                y_min, y_max = None, None
        
        with tab3:
            st.subheader("Individual Data Points")
            
            show_points = st.checkbox("Overlay individual experiment points", value=False,
                                     help="Show raw data points from individual runs (requires full experiment data)",
                                     key="adv_show_points")
            
            if show_points:
                col1, col2 = st.columns(2)
                with col1:
                    point_alpha = st.slider("Point transparency", 0.1, 1.0, 0.5, 0.1, key="adv_point_alpha")
                    point_size = st.slider("Point size", 1, 10, 4, 1, key="adv_point_size")
                with col2:
                    jitter_amount = st.checkbox("Add jitter", value=True,
                                               help="Spread points horizontally to avoid overlap",
                                               key="adv_jitter")
        
        # Generate button
        st.divider()
        if st.button("Generate Advanced Plot", type="primary", key="adv_generate_btn"):
            return {
                "x_col": x_col,
                "x_order": x_order,
                "x_limits": x_limits if 'x_limits' in locals() else None,
                "use_continuous": use_continuous if 'use_continuous' in locals() else False,
                "is_numeric_x": is_numeric_x,
                "plot_type": plot_type if 'plot_type' in locals() else "Bar plot",
                "selected_metrics": list(selected_metrics.keys()),
                "show_error_bars": show_error_bars,
                "show_points": show_points if show_points else False,
                "point_alpha": point_alpha if show_points else 0.5,
                "point_size": point_size if show_points else 4,
                "jitter": jitter_amount if show_points else True,
                "plot_style": plot_style,
                "font_scale": font_scale,
                "color_palette": color_palette,
                "figure_size": (figure_width, figure_height),
                "line_width": line_width if 'line_width' in locals() else 2.0,
                "marker_size": marker_size if 'marker_size' in locals() else 6,
                "y_min": y_min,
                "y_max": y_max,
            }
    
    return None


def render_advanced_plot(table_df: pd.DataFrame, full_df: pd.DataFrame, config: Dict, container=None):
    """Render advanced multi-metric plot with matplotlib/seaborn.
    
    Args:
        table_df: Aggregated summary table (one row per signature with __mean/__std)
        full_df: Raw experiment data for individual points
        config: Configuration dict from advanced_plot_controls
        container: Streamlit container
    """
    panel = container if container is not None else st
    
    # Set style
    sns.set(style=config["plot_style"], context="paper", font_scale=config["font_scale"])
    
    x_col = config["x_col"]
    selected_metrics = config["selected_metrics"]
    plot_type = config.get("plot_type", "Bar plot")
    use_continuous = config.get("use_continuous", False)
    is_numeric_x = config.get("is_numeric_x", True)
    
    # Prepare aggregated data (table_df is already aggregated by signature)
    plot_data = table_df[[x_col] + selected_metrics].copy()
    
    # Also get std columns if they exist
    std_cols = []
    for metric in selected_metrics:
        std_col = metric.replace("__mean", "__std")
        if std_col in table_df.columns:
            std_cols.append(std_col)
            plot_data[std_col] = table_df[std_col]
    
    plot_data = plot_data.dropna(subset=selected_metrics, how='all')
    
    # Apply x-axis ordering if specified
    if config["x_order"]:
        plot_data = plot_data[plot_data[x_col].isin(config["x_order"])]
        plot_data[x_col] = pd.Categorical(plot_data[x_col], categories=config["x_order"], ordered=True)
        plot_data = plot_data.sort_values(x_col)
    elif use_continuous:
        # For continuous, sort by x value
        plot_data = plot_data.sort_values(x_col)
    
    if plot_data.empty:
        panel.warning("No data to plot with selected configuration")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=config["figure_size"])
    
    # Get colors
    colors = sns.color_palette(config["color_palette"], n_colors=len(selected_metrics))
    
    # LINE PLOTS (Continuous numeric OR ordered categorical)
    if use_continuous and plot_type != "Bar plot":
        # Determine if we're dealing with numeric or categorical x-axis
        if is_numeric_x:
            # NUMERIC: Aggregate by numeric x value and sort numerically
            for metric_col, color in zip(selected_metrics, colors):
                std_col = metric_col.replace("__mean", "__std")
                
                # Group by x-axis value and calculate mean of means and pooled std
                grouped_data = []
                
                # Get unique x values and sort them numerically
                x_series = pd.to_numeric(plot_data[x_col], errors='coerce')
                x_values = sorted(x_series.dropna().unique())
                
                for x_val in x_values:
                    # Match by numeric value with small tolerance for floats
                    mask = np.abs(x_series - x_val) < 1e-9
                    group = plot_data[mask]
                    
                    # Mean of the means at this x value
                    y_mean = group[metric_col].mean()
                    
                    # Pooled standard deviation across all configs at this x value
                    if std_col in group.columns:
                        y_std = group[std_col].mean()
                    else:
                        y_std = np.nan
                    
                    grouped_data.append({
                        'x': x_val,
                        'y_mean': y_mean,
                        'y_std': y_std,
                        'n': len(group)
                    })
                
                if not grouped_data:
                    continue
                
                # Convert to arrays
                grouped_df = pd.DataFrame(grouped_data)
                x_vals = grouped_df['x'].values
                y_vals = grouped_df['y_mean'].values
                std_vals = grouped_df['y_std'].values if 'y_std' in grouped_df else None
                
                # Remove NaN
                valid_mask = ~np.isnan(y_vals)
                x_vals = x_vals[valid_mask]
                y_vals = y_vals[valid_mask]
                if std_vals is not None:
                    std_vals = std_vals[valid_mask]
                
                if len(x_vals) == 0:
                    continue
                    
                label = _pretty_metric_label(metric_col)
                
                # Error bands FIRST (so they appear behind the line)
                if config["show_error_bars"] and std_vals is not None:
                    std_mask = ~np.isnan(std_vals)
                    if std_mask.any():
                        ax.fill_between(x_vals[std_mask], 
                                       y_vals[std_mask] - std_vals[std_mask],
                                       y_vals[std_mask] + std_vals[std_mask],
                                       color=color, alpha=0.2)
                
                # Plot line (through aggregated means)
                if "line" in plot_type.lower():
                    ax.plot(x_vals, y_vals, label=label, color=color, 
                           linewidth=config.get("line_width", 2.0), alpha=0.9, marker='o', markersize=4)
                
                # Plot scatter markers (aggregated points)
                if "scatter" in plot_type.lower():
                    ax.scatter(x_vals, y_vals, color=color, s=config.get("marker_size", 6)**2,
                              alpha=0.8, zorder=5, edgecolors='white', linewidths=0.5, 
                              label=label if "line" not in plot_type.lower() else None)
                
                # Individual points if requested (show ALL individual signatures at each x)
                if config["show_points"] and full_df is not None and '__signature' in table_df.columns:
                    base_metric = metric_col.replace("__mean", "")
                    
                    if base_metric in full_df.columns and '__signature' in full_df.columns and x_col in full_df.columns:
                        # For each unique x value, plot all signature means
                        for x_val in x_values:
                            # Find all signatures with this x value in table_df
                            # Use numeric matching with tolerance
                            sig_mask = np.abs(x_series - x_val) < 1e-9
                            sig_rows = plot_data[sig_mask]
                            
                            for idx in sig_rows.index:
                                if idx < len(table_df):
                                    sig = table_df.iloc[idx]['__signature']
                                    
                                    # Get all experiments with this signature
                                    sig_mask_full = full_df['__signature'] == sig
                                    point_data = full_df[sig_mask_full][base_metric].dropna()
                                    
                                    if len(point_data) > 0:
                                        x_points = np.full(len(point_data), float(x_val))
                                        if config["jitter"] and len(point_data) > 1:
                                            x_range = max(x_vals) - min(x_vals) if len(x_vals) > 1 else 0.1
                                            jitter_amt = x_range * 0.01
                                            x_points += np.random.normal(0, jitter_amt, len(point_data))
                                        ax.scatter(x_points, point_data, color='black',
                                                  alpha=config["point_alpha"], s=config["point_size"]**2,
                                                  zorder=10, marker='o', edgecolors='none')
            
            # Format numeric x-axis
            ax.set_xlabel(x_col.replace("_", " ").replace(".", " ").title())
            if config.get("x_limits"):
                ax.set_xlim(config["x_limits"])
        
        else:
            # CATEGORICAL: Use ordered positions for line plot
            x_categories = config["x_order"] if config["x_order"] else sorted(plot_data[x_col].dropna().unique(), key=str)
            x_pos = np.arange(len(x_categories))
            
            for metric_col, color in zip(selected_metrics, colors):
                std_col = metric_col.replace("__mean", "__std")
                
                # Aggregate by category
                y_vals = []
                std_vals = []
                
                for cat in x_categories:
                    mask = plot_data[x_col] == cat
                    group = plot_data[mask]
                    
                    if len(group) > 0:
                        y_vals.append(group[metric_col].mean())
                        if std_col in group.columns:
                            std_vals.append(group[std_col].mean())
                        else:
                            std_vals.append(np.nan)
                    else:
                        y_vals.append(np.nan)
                        std_vals.append(np.nan)
                
                y_vals = np.array(y_vals)
                std_vals = np.array(std_vals)
                
                label = _pretty_metric_label(metric_col)
                
                # Error bands
                if config["show_error_bars"]:
                    valid_mask = ~np.isnan(y_vals) & ~np.isnan(std_vals)
                    if valid_mask.any():
                        ax.fill_between(x_pos[valid_mask], 
                                       y_vals[valid_mask] - std_vals[valid_mask],
                                       y_vals[valid_mask] + std_vals[valid_mask],
                                       color=color, alpha=0.2)
                
                # Plot line
                valid_mask = ~np.isnan(y_vals)
                if "line" in plot_type.lower():
                    ax.plot(x_pos[valid_mask], y_vals[valid_mask], label=label, color=color,
                           linewidth=config.get("line_width", 2.0), alpha=0.9, marker='o', markersize=4)
                
                # Plot scatter
                if "scatter" in plot_type.lower():
                    ax.scatter(x_pos[valid_mask], y_vals[valid_mask], color=color, 
                              s=config.get("marker_size", 6)**2, alpha=0.8, zorder=5,
                              edgecolors='white', linewidths=0.5,
                              label=label if "line" not in plot_type.lower() else None)
                
                # Individual points
                if config["show_points"] and full_df is not None and '__signature' in table_df.columns:
                    base_metric = metric_col.replace("__mean", "")
                    
                    if base_metric in full_df.columns and '__signature' in full_df.columns:
                        for i, cat in enumerate(x_categories):
                            sig_rows = plot_data[plot_data[x_col] == cat]
                            
                            for idx in sig_rows.index:
                                if idx < len(table_df):
                                    sig = table_df.iloc[idx]['__signature']
                                    sig_mask_full = full_df['__signature'] == sig
                                    point_data = full_df[sig_mask_full][base_metric].dropna()
                                    
                                    if len(point_data) > 0:
                                        x_points = np.full(len(point_data), x_pos[i])
                                        if config["jitter"] and len(point_data) > 1:
                                            x_points += np.random.normal(0, 0.1, len(point_data))
                                        ax.scatter(x_points, point_data, color='black',
                                                  alpha=config["point_alpha"], s=config["point_size"]**2,
                                                  zorder=10, marker='o', edgecolors='none')
            
            # Format categorical x-axis
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_categories, rotation=15 if len(x_categories) > 5 else 0,
                              ha='right' if len(x_categories) > 5 else 'center')
            ax.set_xlabel(x_col.replace("_", " ").replace(".", " ").title())
    
    # CATEGORICAL X-AXIS (Bar plots)
    else:
        x_pos = np.arange(len(plot_data))
        bar_width = 0.8 / len(selected_metrics)
        
        for i, (metric_col, color) in enumerate(zip(selected_metrics, colors)):
            offset = (i - len(selected_metrics)/2 + 0.5) * bar_width
            
            y_vals = plot_data[metric_col].values
            label = _pretty_metric_label(metric_col)
            
            # Plot bars (these are already aggregated means)
            ax.bar(x_pos + offset, y_vals, bar_width, label=label, color=color, alpha=0.8)
            
            # Error bars if requested (showing std across seeds)
            if config["show_error_bars"]:
                std_col = metric_col.replace("__mean", "__std")
                if std_col in plot_data.columns:
                    std_vals = plot_data[std_col].values
                    valid_mask = ~pd.isna(std_vals)
                    if valid_mask.any():
                        ax.errorbar(x_pos[valid_mask] + offset, y_vals[valid_mask], 
                                   yerr=std_vals[valid_mask], fmt='none',
                                   ecolor='black', capsize=3, alpha=0.6, linewidth=1)
            
            # Individual points if requested (overlay raw experiment values)
            if config["show_points"] and full_df is not None and '__signature' in table_df.columns:
                base_metric = metric_col.replace("__mean", "")
                
                if base_metric in full_df.columns and '__signature' in full_df.columns:
                    for j, (idx, row) in enumerate(plot_data.iterrows()):
                        x_val = row[x_col]
                        
                        if idx < len(table_df) and '__signature' in table_df.columns:
                            sig = table_df.iloc[idx]['__signature']
                            sig_mask = full_df['__signature'] == sig
                            
                            if x_col in full_df.columns:
                                x_mask = full_df[x_col] == x_val
                                point_data = full_df[sig_mask & x_mask][base_metric].dropna()
                            else:
                                point_data = full_df[sig_mask][base_metric].dropna()
                            
                            if len(point_data) > 0:
                                x_points = np.full(len(point_data), x_pos[j] + offset)
                                if config["jitter"]:
                                    x_points += np.random.normal(0, bar_width*0.15, len(point_data))
                                ax.scatter(x_points, point_data, color='black', 
                                          alpha=config["point_alpha"], s=config["point_size"]**2,
                                          zorder=10)
        
        # Format categorical x-axis
        ax.set_xticks(x_pos)
        ax.set_xticklabels(plot_data[x_col], rotation=15 if len(plot_data) > 5 else 0, 
                           ha='right' if len(plot_data) > 5 else 'center')
        ax.set_xlabel(x_col.replace("_", " ").replace(".", " ").title())
    
    # Common formatting
    ax.set_ylabel("Metric Value")
    ax.legend(loc='best', frameon=False)
    ax.grid(axis='y', alpha=0.3)
    
    if config["y_min"] is not None and config["y_max"] is not None:
        ax.set_ylim(config["y_min"], config["y_max"])
    
    plt.tight_layout()
    
    # Display in Streamlit
    panel.pyplot(fig)
    plt.close()
    
    # Offer download
    panel.caption("Right-click the plot and 'Save image as...' to download")
