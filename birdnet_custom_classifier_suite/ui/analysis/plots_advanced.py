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
    
    with panel.expander("📊 Advanced Graph Tools", expanded=False):
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
        tab1, tab2, tab3 = st.tabs(["📈 Metrics & Axes", "🎨 Style", "📍 Individual Points"])
        
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
                    selected_ood[col_name] = st.checkbox(label, value=default, key=f"ood_{col_name}")
            
            with col2:
                st.write("**IID Metrics**")
                iid_metrics = [m for m in metric_choices if 'iid' in m[1].lower()]
                selected_iid = {}
                for label, col_name in iid_metrics:
                    selected_iid[col_name] = st.checkbox(label, value=False, key=f"iid_{col_name}")
            
            # Combine selected metrics
            selected_metrics = {k: v for k, v in {**selected_ood, **selected_iid}.items() if v}
            
            if not selected_metrics:
                st.warning("⚠️ Select at least one metric to plot")
                return None
            
            st.divider()
            
            # X-axis selection
            x_col = st.selectbox("**X-axis (parameter)**", options=x_candidates, index=0, key="adv_x_col")
            
            # X-axis ordering for categorical
            if not _is_numeric_series(table_df[x_col].dropna()):
                unique_vals = sorted(table_df[x_col].dropna().unique(), key=str)
                st.write("**X-axis order** (drag to reorder)")
                x_order = st.multiselect(
                    "Categories (order matters)",
                    options=unique_vals,
                    default=unique_vals,
                    help="Select and reorder categories as they should appear left-to-right",
                    key="adv_x_order"
                )
            else:
                x_order = None
        
        with tab2:
            st.subheader("Styling Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                plot_style = st.selectbox(
                    "Plot style",
                    options=["whitegrid", "darkgrid", "white", "dark", "ticks"],
                    index=0,
                    key="adv_plot_style"
                )
                
                font_scale = st.slider("Font scale", 0.5, 2.0, 1.0, 0.1, key="adv_font_scale")
                
                show_error_bars = st.checkbox("Show error bars (±SD)", value=True, key="adv_error_bars")
                
            with col2:
                color_palette = st.selectbox(
                    "Color palette",
                    options=["Set2", "husl", "colorblind", "deep", "muted", "pastel", "bright", "dark"],
                    index=0,
                    key="adv_palette"
                )
                
                figure_width = st.slider("Figure width (inches)", 4, 16, 10, 1, key="adv_width")
                figure_height = st.slider("Figure height (inches)", 3, 12, 6, 1, key="adv_height")
            
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
        if st.button("🎨 Generate Advanced Plot", type="primary", key="adv_generate_btn"):
            return {
                "x_col": x_col,
                "x_order": x_order,
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
    
    if plot_data.empty:
        panel.warning("No data to plot with selected configuration")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=config["figure_size"])
    
    # Get colors
    colors = sns.color_palette(config["color_palette"], n_colors=len(selected_metrics))
    
    # Plot each metric
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
                # Only show error bars where we have std data
                valid_mask = ~pd.isna(std_vals)
                if valid_mask.any():
                    ax.errorbar(x_pos[valid_mask] + offset, y_vals[valid_mask], 
                               yerr=std_vals[valid_mask], fmt='none',
                               ecolor='black', capsize=3, alpha=0.6, linewidth=1)
        
        # Individual points if requested (overlay raw experiment values)
        if config["show_points"] and full_df is not None and '__signature' in table_df.columns:
            # Map metric column name back to full_df column (remove __mean suffix)
            base_metric = metric_col.replace("__mean", "")
            
            if base_metric in full_df.columns and '__signature' in full_df.columns:
                # For each x-value (config), plot individual experiment points
                for j, (idx, row) in enumerate(plot_data.iterrows()):
                    x_val = row[x_col]
                    
                    # Get the signature for this aggregated row
                    if idx < len(table_df) and '__signature' in table_df.columns:
                        sig = table_df.iloc[idx]['__signature']
                        
                        # Find all experiments with this signature in full_df
                        sig_mask = full_df['__signature'] == sig
                        
                        # Also filter by x_col value to ensure we're matching the right config
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
    
    # Formatting
    ax.set_xticks(x_pos)
    ax.set_xticklabels(plot_data[x_col], rotation=15 if len(plot_data) > 5 else 0, 
                       ha='right' if len(plot_data) > 5 else 'center')
    ax.set_xlabel(x_col.replace("_", " ").replace(".", " ").title())
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
    panel.caption("💾 Right-click the plot and 'Save image as...' to download")
