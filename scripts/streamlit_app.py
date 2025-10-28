"""
Example Streamlit app using the modular UI components.

This demonstrates how to compose the UI components into a complete
application. The components are designed to be reusable, so you can
build different UIs using the same building blocks.

Run with:
    streamlit run scripts/streamlit_app.py
"""

from __future__ import annotations

import streamlit as st

from birdnet_custom_classifier_suite.ui import (
    UIState,
    data_loader,
    get_signature_breakdown,
    leaderboard,
    metric_controls,
    signature_details,
    summarize_metrics,
)


def main():
    st.set_page_config(page_title="Experiment Explorer", layout="wide")
    st.title("Experiment Explorer — BirdNET Custom Classifier Suite")
    st.markdown(
        "A lightweight UI to explore `results/all_experiments.csv`, "
        "view leaderboards, and inspect configurations across seeds."
    )

    # Initialize or get session state (persists between reruns)
    if "ui_state" not in st.session_state:
        st.session_state.ui_state = UIState()
    state = st.session_state.ui_state
    # Keep a top-level selected_signature key so Streamlit notices changes
    if 'selected_signature' not in st.session_state:
        st.session_state['selected_signature'] = None
    
    # Auto-load and auto-analyze on first run
    if 'auto_load_attempted' not in st.session_state:
        st.session_state['auto_load_attempted'] = True
        from birdnet_custom_classifier_suite.ui.common.types import DEFAULT_RESULTS_PATH
        from birdnet_custom_classifier_suite.ui.analysis.data import load_results
        
        if DEFAULT_RESULTS_PATH.exists():
            try:
                state.results_df = load_results(path=DEFAULT_RESULTS_PATH)
                state.data_source = DEFAULT_RESULTS_PATH
                st.success(f"✓ Auto-loaded {len(state.results_df)} rows from `{DEFAULT_RESULTS_PATH}`")
                
                # Auto-analyze with default settings
                try:
                    # Ensure metric_prefix is a string value, not enum
                    metric_prefix = state.metric_prefix if isinstance(state.metric_prefix, str) else state.metric_prefix.value
                    state.summaries, _ = summarize_metrics(
                        state.results_df,
                        metric_prefix=metric_prefix,
                        top_n=state.top_n,
                        precision_floor=state.precision_floor,
                    )
                    st.success(f"✓ Auto-analyzed with default settings (top {state.top_n}, metric: {metric_prefix})")
                except Exception as e:
                    st.warning(f"Auto-analysis failed: {e}")
            except Exception as e:
                st.warning(f"Could not auto-load default CSV: {e}")
        else:
            st.info(f"Default CSV not found at `{DEFAULT_RESULTS_PATH}`. Use the sidebar to load data.")
    
    # Sidebar controls
    data_loader(state)
    
    if state.results_df is not None:
        metric_controls(state, state.results_df)

        # Analyze button triggers computation and stores summaries in session state
        if st.sidebar.button("Analyze"):
            try:
                # clear any previously selected signature on new analysis
                st.session_state['selected_signature'] = None

                # Ensure metric_prefix is a string value, not enum
                metric_prefix = state.metric_prefix if isinstance(state.metric_prefix, str) else state.metric_prefix.value
                # Apply optional filters
                df_to_use = state.results_df
                def _find_col(columns, prefer):
                    for p in prefer:
                        if p in columns:
                            return p
                    for p in prefer:
                        for c in columns:
                            if c.endswith(p):
                                return c
                    for p in prefer:
                        for c in columns:
                            if p in c:
                                return c
                    return None

                if df_to_use is not None and not df_to_use.empty:
                    cols = list(df_to_use.columns)
                    quality_col = _find_col(cols, ['dataset.filters.quality', 'filters.quality', '.quality', 'quality'])
                    balance_col = _find_col(cols, ['dataset.filters.balance', 'filters.balance', '.balance', 'balance'])
                    
                    if quality_col and state.quality_filter:
                        df_to_use = df_to_use[df_to_use[quality_col].isin(state.quality_filter)]
                    if balance_col and state.balance_filter:
                        df_to_use = df_to_use[df_to_use[balance_col].isin(state.balance_filter)]
                    
                    # Stage/sweep filter: extract from experiment.name
                    if state.sweep_filter and 'experiment.name' in cols:
                        def extract_stage(name):
                            import re
                            name_lower = str(name).lower()
                            # Standard 'stage' pattern
                            m = re.search(r'stage\d+[a-z]*(?:_sweep)?', name_lower)
                            if m:
                                return m.group(0)
                            # Stage0 / legacy names (e.g., '000_OldModel2024', leading zeros)
                            if name_lower.startswith('stage0'):
                                return 'stage0'
                            if re.match(r'^(?:0+[_-]|0+$)', name_lower) or 'oldmodel' in name_lower:
                                return 'stage0'
                            return None
                        df_to_use['__stage_temp'] = df_to_use['experiment.name'].apply(extract_stage)
                        df_to_use = df_to_use[df_to_use['__stage_temp'].isin(state.sweep_filter)]
                        df_to_use = df_to_use.drop(columns=['__stage_temp'])

                if df_to_use is None or df_to_use.empty:
                    st.warning("No rows after applying filters; adjust filters and try again.")
                else:
                    state.summaries, _ = summarize_metrics(
                        df_to_use,
                        metric_prefix=metric_prefix,
                        top_n=state.top_n,
                        precision_floor=state.precision_floor,
                    )
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.exception(e)

        # If summaries already present in state, render leaderboard persistently.
        if getattr(state, "summaries", None):
            st.header("Leaderboard — Top configs")
            leaderboard(
                state.summaries,
                # set a top-level session_state key so Streamlit triggers reruns
                on_select=lambda sig: st.session_state.__setitem__('selected_signature', sig),
            )

            # Show details if signature selected
            selected_sig = st.session_state.get('selected_signature')
            if selected_sig:
                try:
                    breakdown = get_signature_breakdown(
                        state.results_df,
                        selected_sig,
                        metric_prefix=state.metric_prefix,
                    )
                    signature_details(breakdown)
                except Exception as e:
                    st.error(f"Failed to fetch signature details: {e}")

            # Download buttons
            import pandas as pd
            summary_df = pd.DataFrame([
                {
                    "signature": s.signature,
                    "experiments": ", ".join(s.experiment_names),
                    **{f"{name}_mean": m.mean for name, m in s.metrics.items()},
                    **{f"{name}_std": m.std for name, m in s.metrics.items()},
                }
                for s in state.summaries
            ])
            st.download_button(
                "Download summary (CSV)",
                data=summary_df.to_csv(index=False).encode("utf-8"),
                file_name="experiment_summary.csv",
            )
    else:
        st.info("Load data using the sidebar controls to begin analysis.")


if __name__ == "__main__":
    main()
