"""
Streamlit-based interactive analysis UI for BirdNET Custom Classifier Suite.

Run with:

    pip install streamlit    # if not installed
    streamlit run scripts/streamlit_app.py

Features (initial MVP):
- Load `results/all_experiments.csv` (auto) or upload a CSV.
- Show top-N configurations by mean F1 (grouped across seeds).
- Click/select a signature to view per-seed metrics and key config fields.

Design goals:
- Modular: helper functions that can be imported from other programs.
- Human-friendly: compact, rounded tables and clear labels.
- Minimal dependencies: only Streamlit + existing project modules.
"""

from __future__ import annotations

import io
import textwrap
from typing import List

import pandas as pd

try:
    import streamlit as st
except Exception as e:
    raise ImportError(
        "Streamlit is required to run this app. Install with: pip install streamlit"
    ) from e

from birdnet_custom_classifier_suite.eval_toolkit import review, signature as sigmod, rank


# ------------------------- Helpers (modular) ------------------------- #

def load_results(path: str | None = None, uploaded_file: io.BytesIO | None = None) -> pd.DataFrame:
    """Load results DataFrame from path or uploaded file.

    Returns the DataFrame and raises FileNotFoundError if missing.
    """
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)

    if not path:
        raise FileNotFoundError("No path provided and no upload present")

    return review.load_experiments(path)


def ensure_signatures(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure __signature column exists; compute if missing."""
    if "__signature" not in df.columns:
        df = sigmod.add_signatures(df)
    return df


def summarize_and_rank(df: pd.DataFrame, metric_prefix: str = "metrics.ood.best_f1", top_n: int = 10, precision_floor: float | None = None):
    """Return top-N ranked DataFrame (summary-level) based on metric_prefix.

    This uses existing project utilities for summarization and ranking so the
    UI stays consistent with CLI behavior.
    """
    summary = review.summarize_grouped(df, metric_prefix=metric_prefix)
    summary = rank.compute_stability(summary, metric_prefix=metric_prefix.replace("metrics.", ""))

    # Normalize precision floor if passed as percentage > 1
    if precision_floor is not None and precision_floor > 1:
        precision_floor = precision_floor / 100.0

    top = rank.combined_rank(summary, metric=f"{metric_prefix}.f1", precision_floor=precision_floor, stability_weight=0.2)
    if top_n:
        top = top.head(top_n)
    return top, summary


def per_signature_breakdown(df: pd.DataFrame, signature: str, metric_prefix: str = "metrics.ood.best_f1") -> dict:
    """Return a dictionary with per-run rows and compact config summary for a signature.

    Keys:
      - rows: DataFrame subset for signature
      - metrics: list of metric column names present
      - config: list of config column names present
      - aggregates: dict of metric -> (mean, std)
    """
    df = ensure_signatures(df)
    sel = df[df["__signature"] == signature]
    if sel.empty:
        return {"rows": sel, "metrics": [], "config": [], "aggregates": {}}

    # Metric columns we prefer
    preferred_metrics = [f"{metric_prefix}.f1", f"{metric_prefix}.precision", f"{metric_prefix}.recall"]
    metrics = [m for m in preferred_metrics if m in sel.columns]

    # Candidate config keys - reuse signature.pick_config_columns to get config columns
    config_cols = sigmod.pick_config_columns(df)
    # narrow to ones present in sel
    config_cols = [c for c in config_cols if c in sel.columns]

    # Compute aggregates
    aggregates = {}
    for m in metrics:
        vals = sel[m].dropna().astype(float)
        if not vals.empty:
            aggregates[m] = (vals.mean(), vals.std(ddof=0))
        else:
            aggregates[m] = (None, None)

    return {"rows": sel, "metrics": metrics, "config": config_cols, "aggregates": aggregates}


# ------------------------- Streamlit UI ------------------------- #

st.set_page_config(page_title="Experiment Explorer", layout="wide")
st.title("Experiment Explorer — BirdNET Custom Classifier Suite")
st.markdown("A lightweight UI to explore `results/all_experiments.csv`, view leaderboards, and inspect configurations across seeds.")

with st.sidebar.expander("Load data", expanded=True):
    st.write("Choose a source for experiment results (CSV):")
    use_default = st.checkbox("Use default `results/all_experiments.csv`", value=True)
    uploaded = st.file_uploader("Or upload a CSV file", type=["csv"]) if not use_default else None
    custom_path = st.text_input("Custom path (if not using default)", value="results/all_experiments.csv" if use_default else "")
    if use_default:
        data_path = "results/all_experiments.csv"
    else:
        data_path = custom_path.strip() or None

    st.write("---")
    st.write("Leaderboard options")
    metric_prefix = st.text_input("Metric prefix", value="metrics.ood.best_f1")
    top_n = st.number_input("Top N", min_value=1, max_value=200, value=10, step=1)
    precision_floor = st.number_input("Precision floor (0-1 or percent) — optional", value=0.0, min_value=0.0, max_value=100.0, step=0.01)
    if precision_floor == 0.0:
        precision_floor = None

    run_btn = st.button("Load & Run")


# Main area
if run_btn:
    try:
        df = load_results(data_path, uploaded_file=uploaded)
    except Exception as e:
        st.error(f"Failed to load results CSV: {e}")
        st.stop()

    df = ensure_signatures(df)
    st.success(f"Loaded {len(df)} rows from results source.")

    # Leaderboard
    st.header("Leaderboard — Top configs")
    try:
        top, summary = summarize_and_rank(df, metric_prefix=metric_prefix, top_n=top_n, precision_floor=precision_floor)
    except Exception as e:
        st.error(f"Failed to summarize/rank: {e}")
        st.stop()

    if top.empty:
        st.info("No configurations to show (maybe precision floor filtered everything).")
    else:
        # Show a subset of columns for readability
        base_cols = ["__signature", "experiment.names"]
        metric_cols = [c for c in top.columns if c.endswith("_mean") or c.endswith("_std")]
        display_cols = [c for c in base_cols + metric_cols if c in top.columns]
        st.dataframe(top[display_cols].rename(columns=lambda c: c.replace("metrics.", "")))

        # Allow selection of one signature to inspect
        selected_sig = st.selectbox("Inspect signature:", options=top["__signature"].tolist())

        if selected_sig:
            breakdown = per_signature_breakdown(df, selected_sig, metric_prefix=metric_prefix)
            st.subheader(f"Details — Signature {selected_sig}")

            rows = breakdown["rows"]
            metrics = breakdown["metrics"]
            config_cols = breakdown["config"]

            if not rows.empty:
                # Show per-run table (human-friendly)
                show_cols = [c for c in (["experiment.name"] + metrics + config_cols) if c in rows.columns]
                display = rows[show_cols].copy()
                # round metrics
                for m in metrics:
                    if m in display.columns:
                        display[m] = display[m].astype(float).round(4)
                st.table(display)

                # Aggregates
                st.markdown("**Aggregated (mean ± std)**")
                agg_lines = []
                for m, (mn, sd) in breakdown["aggregates"].items():
                    if mn is None:
                        agg_lines.append(f"- {m}: n/a")
                    else:
                        agg_lines.append(f"- {m}: **{mn:.4f}** ± {sd:.4f}")
                st.markdown("\n".join(agg_lines))

            else:
                st.info("No runs found for this signature.")

        # Download buttons
        st.download_button("Download top summary (CSV)", data=top.to_csv(index=False).encode('utf-8'), file_name="leaderboard_top.csv")
        st.download_button("Download full summary (CSV)", data=summary.to_csv(index=False).encode('utf-8'), file_name="leaderboard_full_summary.csv")

    # Raw data tab
    st.header("Raw all_experiments.csv preview")
    st.dataframe(df.head(200))

    st.markdown("---")
    st.markdown("Notes: This is a minimal, modular UI intended as an MVP. I can:")
    st.markdown("- Add more filters and a flexible query builder (by config keys).\n- Add plots (per-signature F1 distribution).\n- Package this as a reusable module under `birdnet_custom_classifier_suite.ui`.")

else:
    st.info("Configure the data source and click 'Load & Run' in the sidebar to begin.")
    st.markdown(textwrap.dedent("""
    Quick tips:
    - If you don't have `streamlit` installed, run `pip install streamlit`.
    - Launch with: `streamlit run scripts/streamlit_app.py`
    - The app will attempt to read `results/all_experiments.csv` by default.
    """))
