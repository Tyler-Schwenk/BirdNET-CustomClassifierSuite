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
    
    # Sidebar controls
    data_loader(state)
    
    if state.results_df is not None:
        metric_controls(state)

        # Analyze button triggers computation and stores summaries in session state
        if st.sidebar.button("Analyze"):
            try:
                # clear any previously selected signature on new analysis
                st.session_state['selected_signature'] = None

                state.summaries, _ = summarize_metrics(
                    state.results_df,
                    metric_prefix=state.metric_prefix,
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
