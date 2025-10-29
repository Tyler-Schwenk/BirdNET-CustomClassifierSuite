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
        "Explore results, design parameter sweeps, and run them from one app."
    )

    # Initialize or get session state (persists between reruns)
    if "ui_state" not in st.session_state:
        st.session_state.ui_state = UIState()
    state = st.session_state.ui_state
    if 'selected_signature' not in st.session_state:
        st.session_state['selected_signature'] = None

    eval_tab, sweeps_tab = st.tabs(["Evaluate", "Sweeps"])

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
                    leaderboard(
                        state.summaries,
                        on_select=lambda sig: st.session_state.__setitem__('selected_signature', sig),
                        show_aggrid_debug=show_ag,
                        debug_container=left_col,
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
                with right_col:
                    st.download_button(
                        "Download summary (CSV)",
                        data=summary_df.to_csv(index=False).encode("utf-8"),
                        file_name="experiment_summary.csv",
                    )
        else:
            with right_col:
                st.info("Load data using the controls to begin analysis.")

    # ---------------------- Sweeps Tab ---------------------- #
    with sweeps_tab:
        st.header("Design and Run Sweeps")
        st.caption("Fill in parameters to generate a sweep spec and configs, then run the sweep.")
        st.markdown("📖 [BirdNET-Analyzer CLI Arguments Documentation](https://birdnet-team.github.io/BirdNET-Analyzer/usage/cli.html#birdnet_analyzer.cli-analyzer_parser-positional-arguments)")

        from pathlib import Path
        import yaml

        # Action buttons at top for immediate visibility
        btn_cols = st.columns(3)
        with btn_cols[0]:
            save_spec_btn = st.button("💾 Save spec YAML", key="save_spec_top", help="Save the sweep specification to config/sweep_specs/<name>.yaml for later use")
        with btn_cols[1]:
            gen_configs_btn = st.button("⚙️ Generate configs", key="gen_configs_top", help="Generate individual experiment YAML configs and manifest.csv in the output folder")
        with btn_cols[2]:
            run_sweep_btn = st.button("▶️ Run sweep now", key="run_sweep_top", help="Generate configs and immediately run all experiments in the sweep")
        
        # Placeholder for action feedback at top
        feedback_placeholder = st.empty()

        st.markdown("---")

        colA, colB = st.columns(2)
        with colA:
            sweep_name = st.text_input("Sweep name", value="stage1_sweep")
            stage = st.number_input("Stage number", min_value=0, max_value=99, value=1, step=1)
            out_dir = st.text_input("Output folder for configs", value=f"config/sweeps/{sweep_name}")
            spec_dir = Path("config/sweep_specs")
            spec_dir.mkdir(parents=True, exist_ok=True)
            spec_path = spec_dir / f"{sweep_name}.yaml"

            st.subheader("Base parameters")
            epochs = st.number_input("epochs", min_value=1, value=50, step=1)
            base_batch_size = st.number_input("batch_size", min_value=1, value=32, step=1)
            learning_rate = st.text_input("learning_rate (single value)", value="0.0005")
            dropout = st.text_input("dropout (single value)", value="0.25")
            upsampling_ratio = st.text_input("upsampling_ratio (single value)", value="0.0")
            mixup = st.checkbox("mixup", value=False)
            label_smoothing = st.checkbox("label_smoothing", value=False)
            focal_loss = st.checkbox("focal_loss", value=False)
            
            # Conditional focal loss params
            if focal_loss:
                focal_loss_gamma = st.number_input("focal_loss_gamma", min_value=0.0, value=2.0, step=0.1, help="Focusing parameter for focal loss")
                focal_loss_alpha = st.number_input("focal_loss_alpha", min_value=0.0, max_value=1.0, value=0.25, step=0.05, help="Class balance parameter for focal loss")
            else:
                focal_loss_gamma = 2.0
                focal_loss_alpha = 0.25
            
            # Advanced frequency/overlap params
            with st.expander("Advanced audio parameters", expanded=False):
                fmin = st.number_input("fmin (Hz)", min_value=0, value=0, step=100, help="Minimum frequency for bandpass filter")
                fmax = st.number_input("fmax (Hz)", min_value=0, value=15000, step=100, help="Maximum frequency for bandpass filter")
                overlap = st.number_input("overlap", min_value=0.0, max_value=2.9, value=0.0, step=0.1, help="Overlap of prediction segments")

        with colB:
            st.subheader("Axes (values to sweep)")
            seeds_str = st.text_input("seed values (comma-separated)", value="123", key="sweep_seeds")
            hidden_units_str = st.text_input("hidden_units (comma-separated)", value="0,128,512", key="sweep_hidden_units", help="Use 0 for no hidden layer")
            dropout_axis_str = st.text_input("dropout values (comma-separated)", value="0.0,0.25", key="sweep_dropout")
            lr_axis_str = st.text_input("learning_rate values (comma-separated)", value="0.0001,0.0005,0.001", key="sweep_lr")
            # batch_size sweep removed; only in base_params
            st.markdown("**Quality sweep options**")
            quality_combos_raw = st.multiselect(
                "Select quality combinations",
                options=["high", "medium", "low", "high,medium", "high,low", "medium,low", "high,medium,low"],
                default=["high", "medium", "low", "high,medium"],
                key="sweep_quality_combos",
                help="Valid quality values: high, medium, low. Select one or more combinations to sweep over."
            )
            balance_opts = st.multiselect("balance options", options=[True, False], default=[True], key="sweep_balance")
            up_mode_opts = st.multiselect("upsampling_mode options", options=["none", "repeat", "linear"], default=["repeat"], key="sweep_upmode")
            up_ratio_axis_str = st.text_input("upsampling_ratio values (comma-separated)", value="0.0,0.25", key="sweep_upratio")

        def _parse_num_list(s: str, cast):
            vals = []
            for tok in s.split(','):
                tok = tok.strip()
                if not tok:
                    continue
                try:
                    vals.append(cast(tok))
                except Exception:
                    pass
            return vals

        def _parse_float(s: str, default: float):
            try:
                return float(s)
            except Exception:
                return default

        base_params = {
            "epochs": int(epochs),
            "upsampling_ratio": _parse_float(upsampling_ratio, 0.0),
            "mixup": bool(mixup),
            "label_smoothing": bool(label_smoothing),
            "focal_loss": bool(focal_loss),
            "focal_loss_gamma": float(focal_loss_gamma),
            "focal_loss_alpha": float(focal_loss_alpha),
            "dropout": _parse_float(dropout, 0.25),
            "learning_rate": _parse_float(learning_rate, 0.0005),
            "batch_size": int(base_batch_size),
            "fmin": int(fmin),
            "fmax": int(fmax),
            "overlap": float(overlap),
        }

        axes = {}
        seeds = _parse_num_list(seeds_str, int)
        if seeds:
            axes["seed"] = seeds
        hidden_units = _parse_num_list(hidden_units_str, int)
        if hidden_units:
            axes["hidden_units"] = hidden_units
        dropout_axis = _parse_num_list(dropout_axis_str, float)
        if dropout_axis:
            axes["dropout"] = dropout_axis
        lr_axis = _parse_num_list(lr_axis_str, float)
        if lr_axis:
            axes["learning_rate"] = lr_axis
        # batch_size sweep removed; only in base_params
        # Parse quality combinations from multiselect (already comma-separated strings)
        quality_axes = []
        for combo_str in quality_combos_raw:
            combo = [q.strip() for q in combo_str.split(",") if q.strip()]
            if combo:
                quality_axes.append(combo)
        if quality_axes:
            axes["quality"] = quality_axes
        if balance_opts:
            axes["balance"] = balance_opts
        if up_mode_opts:
            axes["upsampling_mode"] = up_mode_opts
        up_ratio_axis = _parse_num_list(up_ratio_axis_str, float)
        if up_ratio_axis:
            axes["upsampling_ratio"] = up_ratio_axis

        # Validate quality values
        valid_quality = {"high", "medium", "low"}
        invalid_qualities = []
        for combo in quality_axes:
            for q in combo:
                if q not in valid_quality:
                    invalid_qualities.append(q)
        if invalid_qualities:
            st.warning(f"⚠️ Invalid quality values detected: {', '.join(set(invalid_qualities))}. Valid values are: high, medium, low")

        # Calculate estimated config count
        config_count = 1
        for axis_values in axes.values():
            config_count *= len(axis_values)
        
        st.markdown("### Preview spec")
        st.info(f"📊 This sweep will generate **{config_count} configurations**")
        
        if config_count > 500:
            st.warning(f"⚠️ Large sweep detected ({config_count} configs). Consider reducing axis combinations or running in batches.")
        
        preview = {
            "stage": int(stage),
            "out_dir": out_dir,
            "axes": axes,
            "base_params": base_params,
        }
        st.code(yaml.safe_dump(preview, sort_keys=False), language="yaml")

        # Handle button actions (triggered at top)
        if save_spec_btn:
            spec_dir.mkdir(parents=True, exist_ok=True)
            with open(spec_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(preview, f, sort_keys=False)
            feedback_placeholder.success(f"✅ Saved spec to {spec_path}")
        
        if gen_configs_btn:
            try:
                from birdnet_custom_classifier_suite.sweeps.sweep_generator import generate_sweep
                Path(out_dir).mkdir(parents=True, exist_ok=True)
                generate_sweep(stage=int(stage), out_dir=str(out_dir), axes=axes, base_params=base_params)
                feedback_placeholder.success(f"✅ Generated configs at {out_dir}")
            except Exception as e:
                feedback_placeholder.error(f"❌ Failed to generate sweep: {e}")
        
        if run_sweep_btn:
            try:
                import subprocess, sys
                cmd = [
                    sys.executable,
                    "-m",
                    "birdnet_custom_classifier_suite.sweeps.run_sweep",
                    str(out_dir),
                    "--base-config",
                    "config/base.yaml",
                    "--experiments-root",
                    "experiments",
                ]
                feedback_placeholder.write("Running: " + " ".join(cmd))
                proc = subprocess.run(cmd, capture_output=True, text=True)
                st.code(proc.stdout or "", language="bash")
                if proc.returncode != 0:
                    feedback_placeholder.error("❌ Sweep run failed.")
                    st.code(proc.stderr or "", language="bash")
                else:
                    feedback_placeholder.success("✅ Sweep finished or started successfully (see logs above).")
            except Exception as e:
                feedback_placeholder.error(f"❌ Failed to run sweep: {e}")


if __name__ == "__main__":
    main()
