from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import streamlit as st
import altair as alt


MetricChoice = Tuple[str, str]  # (label, column_name)


def _is_metric_mean_col(col: str) -> bool:
    return col.endswith("__mean")


def _is_metric_std_col(col: str) -> bool:
    return col.endswith("__std")


def _pretty_metric_label(col: str) -> str:
    # Strip suffix and prettify e.g., 'ood.precision__mean' -> 'OOD Precision'
    base = col.replace("__mean", "").replace("__std", "")
    # Replace dots and title case
    return base.replace(".", " ").upper() if base.startswith("ood") or base.startswith("iid") else base.replace(".", " ").title()


def _candidate_param_columns(df: pd.DataFrame) -> List[str]:
    exclude = {"Signature", "Experiments"}
    return [c for c in df.columns if c not in exclude and not _is_metric_mean_col(c) and not _is_metric_std_col(c)]


def list_metric_choices(df: pd.DataFrame) -> List[MetricChoice]:
    metrics = [c for c in df.columns if _is_metric_mean_col(c)]
    # Prefer ordering with OOD first, then IID, then others
    def sort_key(c: str):
        base = c.replace("__mean", "")
        if base.startswith("ood"):
            p = 0
        elif base.startswith("iid"):
            p = 1
        else:
            p = 2
        return (p, base)

    metrics.sort(key=sort_key)
    return [(_pretty_metric_label(c), c) for c in metrics]


def list_param_choices(df: pd.DataFrame) -> List[str]:
    return _candidate_param_columns(df)


def _is_numeric_series(s: pd.Series) -> bool:
    try:
        pd.to_numeric(s.dropna())
        return True
    except Exception:
        return False


def _coerce_numeric_series(s: pd.Series) -> pd.Series:
    """Best-effort conversion of a Series to numeric.

    - First try plain to_numeric.
    - If that fails (all NaN), try extracting the first number substring.
    """
    out = pd.to_numeric(s, errors='coerce')
    if out.isna().all():
        try:
            extracted = s.astype(str).str.extract(r'([-+]?\d*\.?\d+)')[0]
            out = pd.to_numeric(extracted, errors='coerce')
        except Exception:
            pass
    return out


def plot_controls(table_df: pd.DataFrame, container=None) -> Tuple[Optional[str], Optional[str], Dict[str, Optional[float]]]:
    """Render plotting controls and return (x_col, y_col).

    - Y-axis options come from metric mean columns ("*__mean").
    - X-axis options come from non-metric columns currently in the table.
    """
    panel = container if container is not None else st

    metric_choices = list_metric_choices(table_df)
    if not metric_choices:
        panel.info("No metric columns found to plot. Run analysis to populate metrics.")
        return None, None, {}

    x_candidates = list_param_choices(table_df)
    if not x_candidates:
        panel.info("No parameter columns available. Toggle config columns in the leaderboard and try again.")
        return None, None, {}

    with panel.expander("Charts", expanded=True):
        c0, c1, c2 = st.columns(3)
        with c0:
            chart_type = st.selectbox(
                "Chart type",
                options=["Auto", "Bars", "Points", "Line"],
                index=0,
                help="Auto: scatter for numeric X, bars for categorical X. Override to force Bars/Points/Line.")
        with c1:
            y_label_to_col = {lbl: col for lbl, col in metric_choices}
            y_label_default = next(iter(y_label_to_col.keys()))
            y_label = st.selectbox("Y axis (metric)", options=list(y_label_to_col.keys()), index=0, help="Metrics are aggregated as means in the leaderboard.")
            y_col = y_label_to_col[y_label]
        with c2:
            x_col = st.selectbox("X axis (parameter)", options=x_candidates, index=0, help="Choose a configuration column to plot against the selected metric.")

        # Error bars option (when std available for the selected metric)
        y_std_col = y_col.replace("__mean", "__std") if y_col and y_col.endswith("__mean") else None
        has_std = bool(y_std_col and (y_std_col in table_df.columns))
        show_error_bars = False
        if has_std:
            show_error_bars = st.checkbox("Show error bars", value=False, help="Uses metric standard deviation when available.")

        # Optional Y-axis domain limit
        limit_domain = st.checkbox("Limit Y-axis range", value=False)
        y_min = None
        y_max = None
        if limit_domain:
            y_min, y_max = st.slider(
                "Y-axis range",
                min_value=0.0,
                max_value=1.0,
                value=(0.0, 1.0),
                step=0.01,
                help="Adjust to focus the chart on a specific range (e.g., 0.7–1.0).",
            )

        # Optional debug
        debug_plot = st.checkbox("Show chart debug", value=False)

        return x_col, y_col, {
            "show_error_bars": show_error_bars if has_std else False,
            "y_min": y_min,
            "y_max": y_max,
            "chart_type": chart_type,
            "debug": debug_plot,
        }


def render_chart(table_df: pd.DataFrame, x_col: str, y_col: str, container=None, *, show_error_bars: bool = False, y_min: Optional[float] = None, y_max: Optional[float] = None, debug: bool = False, chart_type: str = "Auto") -> None:
    """Render a simple chart for the selected axes.

    - If X is numeric: scatter with rule for median per X (optional)
    - If X is categorical: bar chart with mean and error bars (std if available)
    """
    panel = container if container is not None else st

    if x_col not in table_df.columns or y_col not in table_df.columns:
        panel.warning("Selected columns not found in current table.")
        return

    # Build a minimal aligned plotting DataFrame (x, y_mean, optional y_std), then drop NaNs together
    cols = [x_col, y_col]
    y_std_col = y_col.replace("__mean", "__std") if y_col.endswith("__mean") else None
    if show_error_bars and y_std_col and (y_std_col in table_df.columns):
        cols.append(y_std_col)
    df = table_df[cols].copy()
    df = df.dropna(subset=[c for c in cols if c in df.columns])
    if df.empty:
        panel.info("No data available for the current selection. Try disabling error bars or widening the Y-axis range.")
        return

    # Prepare titles
    y_title = _pretty_metric_label(y_col)
    x_title = x_col

    # Detect numeric vs categorical X
    x_numeric = _is_numeric_series(df[x_col])

    # Ensure types for Altair (robust coercion)
    if x_numeric:
        df[x_col] = _coerce_numeric_series(df[x_col])
    df[y_col] = _coerce_numeric_series(df[y_col])

    # Ensure std numeric if present
    if y_std_col and y_std_col in df.columns:
        df[y_std_col] = pd.to_numeric(df[y_std_col], errors='coerce')

    # Common y-scale (use alt.Undefined when not set to avoid odd interactions)
    y_scale = alt.Scale(domain=[y_min, y_max]) if (y_min is not None and y_max is not None and y_min < y_max) else alt.Undefined

    # Resolve chart type
    if chart_type == "Auto":
        chart_type_resolved = "Points" if x_numeric else "Bars"
    else:
        chart_type_resolved = chart_type

    if debug:
        panel.write("Plot debug:")
        panel.json({
            "x_col": x_col,
            "y_col": y_col,
            "y_std_col": y_std_col if (show_error_bars and y_std_col in df.columns) else None,
            "rows": len(df),
            "y_min": y_min,
            "y_max": y_max,
            "dtypes": {c: str(dt) for c, dt in df.dtypes.items()},
            "x_nans": int(df[x_col].isna().sum()) if x_col in df.columns else None,
            "y_nans": int(df[y_col].isna().sum()) if y_col in df.columns else None,
            "chart_type_resolved": chart_type_resolved,
        })
        try:
            panel.write("Preview of plotting data (first 10 rows):")
            panel.dataframe(df.head(10))
        except Exception:
            pass

    if chart_type_resolved == "Points":
        # Use safe column names to avoid issues with dots in field names
        df_plot = df.rename(columns={x_col: "__x__", y_col: "__y__"}).copy()
        ystd_field = None
        if show_error_bars and (y_std_col is not None) and (y_std_col in df.columns) and not df[y_std_col].isna().all():
            # y_std is already present in df; just rename in df_plot to avoid column overlap
            if y_std_col in df_plot.columns:
                df_plot = df_plot.rename(columns={y_std_col: "__ystd__"})
            else:
                # Fallback: align and insert the std column explicitly
                df_plot["__ystd__"] = df[y_std_col].values
            ystd_field = "__ystd__"

        base = alt.Chart(df_plot).mark_point(filled=True, size=80, clip=True).encode(
            x=alt.X("__x__:Q", title=x_title),
            y=alt.Y("__y__:Q", title=y_title, scale=y_scale),
            tooltip=[alt.Tooltip("__x__", title=x_title), alt.Tooltip("__y__", title=y_title)]
        )
        chart = base

        # Add per-point error bars if requested and std available
        if ystd_field is not None:
            df_err = df_plot.copy()
            df_err["__y_lower__"] = df_err["__y__"] - df_err[ystd_field]
            df_err["__y_upper__"] = df_err["__y__"] + df_err[ystd_field]
            err = alt.Chart(df_err).mark_rule(clip=True, strokeWidth=2.5).encode(
                x=alt.X("__x__:Q"),
                y=alt.Y("__y_lower__:Q", title=y_title, scale=y_scale),
                y2=alt.Y2("__y_upper__:Q"),
                tooltip=[alt.Tooltip("__x__", title=x_title), alt.Tooltip("__y__", title=y_title)]
            )
            caps_lower = alt.Chart(df_err).mark_tick(clip=True, thickness=2, size=12).encode(
                x=alt.X("__x__:Q"), y=alt.Y("__y_lower__:Q", scale=y_scale)
            )
            caps_upper = alt.Chart(df_err).mark_tick(clip=True, thickness=2, size=12).encode(
                x=alt.X("__x__:Q"), y=alt.Y("__y_upper__:Q", scale=y_scale)
            )
            chart = err + caps_lower + caps_upper + base
        final_chart = chart.interactive().properties(height=320)
        # Improve label visibility: bold axis titles and labels, larger font sizes
        final_chart = final_chart.configure_axis(
            titleFontSize=14,
            labelFontSize=12,
            titleFontWeight='bold',
            labelFontWeight='bold'
        ).configure_legend(
            titleFontSize=13,
            labelFontSize=12,
            titleFontWeight='bold'
        ).configure_title(
            fontSize=14,
            fontWeight='bold'
        )
        panel.altair_chart(final_chart, use_container_width=True)
        
        # Download buttons for the chart
        try:
            import vl_convert as vlc
            base_name = f"chart_{x_title.replace(' ', '_')}_{y_title.replace(' ', '_')}"
            col1, col2 = panel.columns(2)
            png_data = vlc.vegalite_to_png(final_chart.to_json(), scale=2)
            col1.download_button(
                label="Download PNG",
                data=png_data,
                file_name=f"{base_name}.png",
                mime="image/png"
            )
            svg_data = vlc.vegalite_to_svg(final_chart.to_json())
            col2.download_button(
                label="Download SVG",
                data=svg_data,
                file_name=f"{base_name}.svg",
                mime="image/svg+xml"
            )
        except Exception as e:
            panel.caption(f"Download unavailable: {e}")
    elif chart_type_resolved == "Bars":
        # Aggregate by category: mean and std across rows (of y_mean)
        agg = df.groupby(x_col)[y_col].agg(['mean', 'std']).reset_index()
        # Determine if we are using a custom Y domain
        custom_domain = (y_min is not None and y_max is not None and y_min < y_max)
        # Prepare plotting columns with optional clamping to keep bars within the visible domain
        if custom_domain:
            agg['baseline'] = float(y_min)
            agg['mean_plot'] = np.clip(agg['mean'], y_min, y_max)
            # For error bars
            agg['y_lower_plot'] = np.clip(agg['mean'] - agg['std'].fillna(0), y_min, y_max)
            agg['y_upper_plot'] = np.clip(agg['mean'] + agg['std'].fillna(0), y_min, y_max)
        else:
            agg['mean_plot'] = agg['mean']
            agg['y_lower_plot'] = (agg['mean'] - agg['std'].fillna(0))
            agg['y_upper_plot'] = (agg['mean'] + agg['std'].fillna(0))

        # Use zero=False and clamp to avoid bars extending below axis when domain is set
        y_scale_cat = alt.Scale(domain=[y_min, y_max], zero=False, clamp=True) if custom_domain else alt.Undefined
        sort_field = alt.SortField(field='mean_plot', order='descending') if custom_domain else alt.SortField(field='mean', order='descending')
        if custom_domain:
            # Anchor bars at the lower bound explicitly using a baseline column for y and mean_plot for y2
            bar = alt.Chart(agg).mark_bar(clip=True).encode(
                x=alt.X(f"{x_col}:{'Q' if x_numeric else 'N'}", sort=sort_field, title=x_title),
                y=alt.Y("baseline:Q", title=y_title, scale=y_scale_cat),
                y2=alt.Y2("mean_plot:Q"),
                tooltip=[x_col, alt.Tooltip("mean:Q", title=y_title), alt.Tooltip("std:Q", title="STD")]
            )
        else:
            bar = alt.Chart(agg).mark_bar(clip=True).encode(
                x=alt.X(f"{x_col}:{'Q' if x_numeric else 'N'}", sort=sort_field, title=x_title),
                y=alt.Y("mean_plot:Q", title=y_title, scale=y_scale_cat),
                tooltip=[x_col, alt.Tooltip("mean:Q", title=y_title), alt.Tooltip("std:Q", title="STD")]
            )
        chart = bar
        # When many categories fall below the selected domain, bars become flat (zero height).
        # Add baseline ticks for visibility and inform the user.
        if custom_domain:
            n_total = len(agg)
            n_below = int((agg['mean'] < y_min).sum()) if y_min is not None else 0
            if n_below == n_total:
                panel.caption("All categories are below the selected Y range; showing baseline ticks at the lower bound.")
            elif n_below > 0:
                panel.caption(f"{n_below}/{n_total} categories fall below the Y range and are flattened to the baseline.")
            baseline_ticks = alt.Chart(agg).mark_tick(color='#666', thickness=2, clip=True).encode(
                x=alt.X(f"{x_col}:{'Q' if x_numeric else 'N'}", sort=sort_field),
                y=alt.Y("baseline:Q", scale=y_scale_cat),
            )
            chart = chart + baseline_ticks

        # Add a point overlay at the (possibly clamped) mean to ensure something is visible even with flat bars
        point_overlay = alt.Chart(agg).mark_point(size=70, color='#1f77b4', filled=True, clip=True).encode(
            x=alt.X(f"{x_col}:{'Q' if x_numeric else 'N'}", sort=sort_field),
            y=alt.Y("mean_plot:Q", scale=y_scale_cat),
            tooltip=[x_col, alt.Tooltip("mean:Q", title=y_title)],
        )
        chart = chart + point_overlay
        if show_error_bars and not agg['std'].isna().all():
            err = alt.Chart(agg).mark_rule(clip=True, strokeWidth=2.5).encode(
                x=alt.X(f"{x_col}:{'Q' if x_numeric else 'N'}"),
                y=alt.Y("y_lower_plot:Q", title=y_title, scale=y_scale_cat),
                y2=alt.Y2("y_upper_plot:Q"),
            )
            caps_lower = alt.Chart(agg).mark_tick(clip=True, thickness=2, size=12).encode(
                x=alt.X(f"{x_col}:{'Q' if x_numeric else 'N'}"), y=alt.Y("y_lower_plot:Q", scale=y_scale_cat)
            )
            caps_upper = alt.Chart(agg).mark_tick(clip=True, thickness=2, size=12).encode(
                x=alt.X(f"{x_col}:{'Q' if x_numeric else 'N'}"), y=alt.Y("y_upper_plot:Q", scale=y_scale_cat)
            )
            chart = chart + err + caps_lower + caps_upper
        # Render with fixed height
        final_chart = chart.properties(height=320)
        # Improve label visibility: bold axis titles and labels, larger font sizes
        final_chart = final_chart.configure_axis(
            titleFontSize=14,
            labelFontSize=12,
            titleFontWeight='bold',
            labelFontWeight='bold'
        ).configure_legend(
            titleFontSize=13,
            labelFontSize=12,
            titleFontWeight='bold'
        ).configure_title(
            fontSize=14,
            fontWeight='bold'
        )
        panel.altair_chart(final_chart, use_container_width=True)
        
        # Download buttons for the chart
        try:
            import vl_convert as vlc
            base_name = f"chart_{x_title.replace(' ', '_')}_{y_title.replace(' ', '_')}"
            col1, col2 = panel.columns(2)
            png_data = vlc.vegalite_to_png(final_chart.to_json(), scale=2)
            col1.download_button(
                label="Download PNG",
                data=png_data,
                file_name=f"{base_name}.png",
                mime="image/png"
            )
            svg_data = vlc.vegalite_to_svg(final_chart.to_json())
            col2.download_button(
                label="Download SVG",
                data=svg_data,
                file_name=f"{base_name}.svg",
                mime="image/svg+xml"
            )
        except Exception as e:
            panel.caption(f"Download unavailable: {e}")
    else:
        # Line chart: aggregate by X and connect means
        agg = df.groupby(x_col)[y_col].agg(['mean', 'std']).reset_index().sort_values(by=x_col)
        custom_domain = (y_min is not None and y_max is not None and y_min < y_max)
        y_scale_line = alt.Scale(domain=[y_min, y_max], zero=False, clamp=True) if custom_domain else alt.Undefined

        base_line = alt.Chart(agg).mark_line(point=True, clip=True).encode(
            x=alt.X(f"{x_col}:{'Q' if x_numeric else 'N'}", title=x_title),
            y=alt.Y("mean:Q", title=y_title, scale=y_scale_line),
            tooltip=[x_col, alt.Tooltip("mean:Q", title=y_title), alt.Tooltip("std:Q", title="STD")]
        )
        chart = base_line
        if show_error_bars and not agg['std'].isna().all():
            agg_err = agg.copy()
            agg_err['y_lower'] = agg_err['mean'] - agg_err['std'].fillna(0)
            agg_err['y_upper'] = agg_err['mean'] + agg_err['std'].fillna(0)
            err = alt.Chart(agg_err).mark_rule(clip=True, strokeWidth=2.5).encode(
                x=alt.X(f"{x_col}:{'Q' if x_numeric else 'N'}"),
                y=alt.Y("y_lower:Q", scale=y_scale_line),
                y2=alt.Y2("y_upper:Q"),
            )
            caps_lower = alt.Chart(agg_err).mark_tick(clip=True, thickness=2, size=12).encode(
                x=alt.X(f"{x_col}:{'Q' if x_numeric else 'N'}"), y=alt.Y("y_lower:Q", scale=y_scale_line)
            )
            caps_upper = alt.Chart(agg_err).mark_tick(clip=True, thickness=2, size=12).encode(
                x=alt.X(f"{x_col}:{'Q' if x_numeric else 'N'}"), y=alt.Y("y_upper:Q", scale=y_scale_line)
            )
            chart = err + caps_lower + caps_upper + base_line
        final_chart = chart.properties(height=320)
        # Improve label visibility: bold axis titles and labels, larger font sizes
        final_chart = final_chart.configure_axis(
            titleFontSize=14,
            labelFontSize=12,
            titleFontWeight='bold',
            labelFontWeight='bold'
        ).configure_legend(
            titleFontSize=13,
            labelFontSize=12,
            titleFontWeight='bold'
        ).configure_title(
            fontSize=14,
            fontWeight='bold'
        )
        panel.altair_chart(final_chart, use_container_width=True)
        
        # Download buttons for the chart
        try:
            import vl_convert as vlc
            base_name = f"chart_{x_title.replace(' ', '_')}_{y_title.replace(' ', '_')}"
            col1, col2 = panel.columns(2)
            png_data = vlc.vegalite_to_png(final_chart.to_json(), scale=2)
            col1.download_button(
                label="Download PNG",
                data=png_data,
                file_name=f"{base_name}.png",
                mime="image/png"
            )
            svg_data = vlc.vegalite_to_svg(final_chart.to_json())
            col2.download_button(
                label="Download SVG",
                data=svg_data,
                file_name=f"{base_name}.svg",
                mime="image/svg+xml"
            )
        except Exception as e:
            panel.caption(f"Download unavailable: {e}")
