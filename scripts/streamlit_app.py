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
    
    # Sidebar controls
    data_loader(state)
    
    if state.results_df is not None:
        metric_controls(state)
        analyze = st.sidebar.button("Analyze")
    
        if analyze:
            try:
                # Get ranked summaries
                state.summaries, _ = summarize_metrics(
                    state.results_df,
                    metric_prefix=state.metric_prefix,
                    top_n=state.top_n,
                    precision_floor=state.precision_floor,
                )
                
                # Show leaderboard
                if state.summaries:
                    st.header("Leaderboard — Top configs")
                    leaderboard(
                        state.summaries,
                        on_select=lambda sig: setattr(state, "selected_signature", sig)
                    )
                    
                    # Show details if signature selected
                    if state.selected_signature:
                        breakdown = get_signature_breakdown(
                            state.results_df,
                            state.selected_signature,
                            metric_prefix=state.metric_prefix,
                        )
                        signature_details(breakdown)
                    
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
                    st.warning("No results to display. Try adjusting the precision floor or other filters.")

            except Exception as e:
                st.error(f"Analysis failed: {e}")
                import traceback
                st.exception(e)
    else:
        st.info("Load data using the sidebar controls to begin analysis.")


if __name__ == "__main__":
    main()
