"""
Example Streamlit app using the modular UI components.

This demonstrates how to compose the UI components into a complete
application. The components are designed to be reusable, so you can
build different UIs using the same building blocks.

Run with:
    streamlit run scripts/streamlit_app.py
"""

from __future__ import annotations

import pandas as pd
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
        "Explore results, design parameter sweeps, and run them from one app."
    )

    # Initialize or get session state (persists between reruns)
    if "ui_state" not in st.session_state:
        st.session_state.ui_state = UIState()
    state = st.session_state.ui_state
    if 'selected_signature' not in st.session_state:
        st.session_state['selected_signature'] = None

    eval_tab, sweeps_tab, hn_tab, fm_tab = st.tabs(["Evaluate", "Sweeps", "Hard Negatives", "File Management"])

    # ---------------------- Evaluate Tab ---------------------- #
    with eval_tab:
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

        # Inline controls: left column hosts controls, right shows results
        left_col, right_col = st.columns([1, 3])
        from birdnet_custom_classifier_suite.ui.analysis.views import data_loader as dl_inline
        with left_col:
            dl_inline(state, container=left_col)

        if state.results_df is not None:
            from birdnet_custom_classifier_suite.ui.analysis.views import metric_controls as mc_inline
            with left_col:
                mc_inline(state, state.results_df, container=left_col)

            # Analyze button triggers computation and stores summaries in session state
            with left_col:
                analyze_clicked = st.button("Analyze")
            if analyze_clicked:
                try:
                    st.session_state['selected_signature'] = None

                    metric_prefix = state.metric_prefix if isinstance(state.metric_prefix, str) else state.metric_prefix.value
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
                                m = re.search(r'stage\d+[a-z]*(?:_sweep)?', name_lower)
                                if m:
                                    return m.group(0)
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
                with left_col:
                    show_ag = st.checkbox("Show AgGrid response", value=False, key="aggrid_debug_checkbox")
                with right_col:
                    st.header("Leaderboard — Top configs")
                    table_df = leaderboard(
                        state.summaries,
                        on_select=lambda sig: st.session_state.__setitem__('selected_signature', sig),
                        show_aggrid_debug=show_ag,
                        debug_container=left_col,
                    )

                # Download the currently visible leaderboard as CSV (includes toggled columns)
                if 'table_df' in locals() and table_df is not None:
                    with right_col:
                        st.download_button(
                            "Download table (CSV)",
                            data=table_df.to_csv(index=False).encode("utf-8"),
                            file_name="leaderboard_table.csv",
                        )

                # Charts based on current leaderboard table
                if table_df is not None and not table_df.empty:
                    from birdnet_custom_classifier_suite.ui.analysis import plots as charts
                    with right_col:
                        x_col, y_col, opts = charts.plot_controls(table_df, container=right_col)
                        if x_col and y_col:
                            charts.render_chart(
                                table_df, x_col, y_col, container=right_col,
                                show_error_bars=bool(opts.get("show_error_bars")),
                                y_min=opts.get("y_min"), y_max=opts.get("y_max"),
                                debug=bool(opts.get("debug")),
                                chart_type=str(opts.get("chart_type", "Auto")),
                            )

                selected_sig = st.session_state.get('selected_signature')
                if selected_sig:
                    with right_col:
                        try:
                            breakdown = get_signature_breakdown(
                                state.results_df,
                                selected_sig,
                                metric_prefix=state.metric_prefix,
                            )
                            signature_details(breakdown)
                        except Exception as e:
                            st.error(f"Failed to fetch signature details: {e}")

                
        else:
            with right_col:
                st.info("Load data using the controls to begin analysis.")

    # ---------------------- Sweeps Tab ---------------------- #
    with sweeps_tab:
        st.header("Design and Run Sweeps")
        st.caption("Fill in parameters to generate a sweep spec and configs, then run the sweep.")
        st.markdown("📖 [BirdNET-Analyzer CLI Arguments Documentation](https://birdnet-team.github.io/BirdNET-Analyzer/usage/cli.html#birdnet_analyzer.cli-analyzer_parser-positional-arguments)")

        from birdnet_custom_classifier_suite.ui.sweeps import sweep_form, sweep_actions, render_action_buttons

        # Reserve a top area for buttons, then render the form, then draw buttons into the top area using current values
        buttons_container = st.container()
        feedback_placeholder = st.empty()
        st.markdown("---")

        # Render the sweep parameter form to capture current values
        sweep_state = sweep_form()

        # Now render buttons at the very top, using the current form values
        with buttons_container:
            save_clicked, generate_clicked, run_clicked, regen_run_clicked = render_action_buttons(
                sweep_state.stage,
                sweep_state.sweep_name,
                sweep_state.out_dir,
            )
            # Output area directly below buttons for logs/progress
            run_output_container = st.container()

        # Handle any button actions with the populated state
    sweep_actions(sweep_state, feedback_placeholder, save_clicked, generate_clicked, run_clicked, regen_run_clicked, run_output_container)

    # ---------------------- Hard Negatives Tab ---------------------- #
    with hn_tab:
        st.header("Hard-negative mining")
        from birdnet_custom_classifier_suite.ui import hard_negative_panel
        try:
            hard_negative_panel()
        except Exception as e:
            st.error(f"Hard-negative panel failed: {e}")

    # ---------------------- File Management Tab ---------------------- #
    with fm_tab:
        st.header("File management")
        from birdnet_custom_classifier_suite.ui import file_management_panel
        try:
            file_management_panel()
        except Exception as e:
            st.error(f"File-management panel failed: {e}")




if __name__ == "__main__":
    main()
