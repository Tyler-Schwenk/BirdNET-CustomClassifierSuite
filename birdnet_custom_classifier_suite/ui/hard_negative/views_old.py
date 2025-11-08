"""
Hard-negative mining UI panel for Streamlit.

Provides simple tools to load a per-file RADR CSV (low_quality_radr_max.csv),
match rows to an input folder (default: scripts/input), preview audio, select
files for curation and copy them into script-local curated folders while
writing a small selection report for provenance.

This is intentionally lightweight and reuses the same matching/copy heuristics
used by the standalone `scripts/curate_low_quality_subsets.py` script so users
get identical behavior whether they run the script or use the UI.
"""
from __future__ import annotations

from pathlib import Path
import json
import shutil
import time
import tempfile
import zipfile
from typing import List, Optional

import pandas as pd
import streamlit as st

from birdnet_custom_classifier_suite.ui.hard_negative import engine
from birdnet_custom_classifier_suite.ui.hard_negative import curator


def _load_radr_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"RADR CSV not found: {path}")
    df = pd.read_csv(path)
    cols = {c: c.strip() for c in df.columns}
    df.rename(columns=cols, inplace=True)
    conf_col = next((c for c in df.columns if c.lower().endswith("confidence") or c.lower().startswith("radr") or "conf" in c.lower()), None)
    file_col = next((c for c in df.columns if c.lower() == "file" or c.lower().endswith("file")), None)
    if conf_col is None:
        raise ValueError("Could not find a confidence column in the RADR CSV")
    if file_col is None:
        file_col = df.columns[0]
    df = df[[file_col, conf_col]].copy()
    df.columns = ["File", "radr_max_confidence"]
    return df


def _match_files(df: pd.DataFrame, input_dir: Path) -> pd.DataFrame:
    # Delegate to the centralized (and optimized) curator.match_files implementation.
    return curator.match_files(df, input_dir)


def _copy_files(paths: List[Path], dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    created = 0
    for p in paths:
        tgt = dest / p.name
        if tgt.exists():
            continue
        try:
            shutil.copy2(p, tgt)
            created += 1
        except Exception:
            try:
                # best-effort fallback
                shutil.copy(p, tgt)
                created += 1
            except Exception:
                pass
    return created


def panel(container=None):
    st_ = container if container is not None else st
    st_.header("Hard-negative mining")
    st_.write("Load RADR per-file max confidence CSV, preview candidates, and select files to copy into curated folders for training.")

    # Defaults
    default_input = Path("scripts") / "input"
    default_outroot = Path("scripts") / "curated"

    # File-picker UI: user selects an input folder via the OS dialog or pastes a path.
    # (Removed automatic scanning for audio folders — use Explorer chooser.)
    # Use a single-column layout for the chooser + text input.
    col1 = st_.container()

    # Mode: run inference or upload existing per-file CSV
    with st_.expander("Input & model", expanded=True):
        # Allow selecting input folder via native dialog or pasting the path
        with col1:
            # Initialize session state key before creating the widget to avoid Streamlit
            # warnings about the widget value being set from two places.
            if 'hn_input_dir' not in st.session_state:
                st.session_state['hn_input_dir'] = str(default_input)

            try:
                if st_.button("Choose folder (Explorer)", key='hn_choose_folder'):
                    try:
                        import tkinter as tk
                        from tkinter import filedialog

                        root = tk.Tk()
                        root.withdraw()
                        try:
                            root.attributes('-topmost', True)
                        except Exception:
                            pass
                        folder = filedialog.askdirectory()
                        root.destroy()
                        if folder:
                            st.session_state['hn_input_dir'] = str(folder)
                            st.rerun()
                    except Exception as e:
                        st_.warning(f"Native folder dialog failed: {e}. Please paste the path into the box instead.")
            except Exception:
                pass

            # Create the text_input bound to session state only (no value=...) to avoid
            # mixing explicit defaults with session state updates.
            input_dir = st_.text_input("Input folder containing audio files", key='hn_input_dir')

        st_.caption("Two workflows: 1) Run inference on the chosen folder with a model (pick an experiment model or upload a .tflite). 2) Upload an existing per-file CSV of confidences (if you already ran analyzer elsewhere) and match rows to files in the input folder.")

        # Model source selection: clear explicit choices
        model_source = st_.radio("Model source:", options=["Use experiment (canonical analyzer args)", "Use a model file from an experiment", "Upload a .tflite file"], index=0, key='hn_model_source')

    # Common experiment discovery - cache to avoid repeated directory scans
    col_exp, col_refresh = st_.columns([4, 1])
    # Root where experiments live (define unconditionally so later code can safely reference it)
    exp_root = Path('experiments')
    with col_refresh:
        if st_.button("↻ Refresh", key='hn_refresh_exp', help="Refresh experiment list"):
            st.session_state.pop('hn_exp_names', None)
            st.rerun()

    if 'hn_exp_names' not in st.session_state:
        exp_names = []
        if exp_root.exists():
            try:
                exp_names = sorted([p.name for p in exp_root.iterdir() if p.is_dir()])
            except Exception:
                exp_names = []
        st.session_state['hn_exp_names'] = exp_names
    else:
        exp_names = st.session_state['hn_exp_names']

    selected_experiment = None
    model_choice = None
    uploaded_model = None

    if model_source == "Use experiment (canonical analyzer args)":
        selected_experiment = st_.selectbox("Choose experiment to run (uses same analyzer args)", options=["(none)"] + exp_names, index=0)
        run_infer_btn = st_.button("Run inference using selected experiment")

    elif model_source == "Use a model file from an experiment":
        selected_experiment = st_.selectbox("Choose experiment to pick model from", options=["(none)"] + exp_names, index=0)
        # list model files in the selected experiment
        model_files = []
        if selected_experiment and selected_experiment != '(none)':
            exp_dir = exp_root / selected_experiment
            try:
                # look for common model file extensions (non-recursive to keep it fast)
                for ext in ('.tflite', '.h5', '.pt'):
                    model_files.extend(sorted(exp_dir.glob(f'*{ext}')))
            except Exception:
                model_files = []
        model_labels = [str(p) for p in model_files]
        if model_labels:
            sel = st_.selectbox("Choose a model file", options=["(none)"] + model_labels, index=0)
            if sel and sel != '(none)':
                model_choice = Path(sel)
        else:
            st_.info("No model files found in the selected experiment. You may upload a .tflite instead.")
        run_infer_btn = st_.button("Run inference using chosen model file")

    else:
        uploaded_model = st_.file_uploader("Upload a TFLite model (.tflite)", type=['tflite'])
        run_infer_btn = st_.button("Run inference with uploaded model")

    # CSV upload path (separate workflow)
    st_.markdown("---")
    st_.subheader("Or: use an existing per-file CSV")
    csv_upload = st_.file_uploader("Upload per-file CSV with columns File and confidence (or similar)", type=['csv'])
    load_csv_btn = st_.button("Load CSV and match files")

    df_matched = None
    # Handle run inference path (model_source branches)
    inp = Path(input_dir)
    tmp_model_f = None
    if 'run_infer_btn' in locals() and run_infer_btn:
        if not inp.exists() or not any(inp.iterdir()):
            st_.warning(f"Input folder empty or not found: {inp}")
        else:
            st_.info("Running inference — this can take a while depending on the model and number of files.")
            try:
                df_out = None
                analyzer_out_root = None
                stamp = int(time.time())
                # Stream analyzer output into the UI so users see progress and logs (like sweeps)
                status_box = st_.empty()
                progress_box = st_.progress(0)
                log_area = st_.empty()

                log_lines = []
                last_ui = 0.0

                if model_source == "Use experiment (canonical analyzer args)":
                    if selected_experiment and selected_experiment != '(none)':
                        status_box.info("Preparing experiment-based analyzer run...")
                        proc, out_root, cmd = engine.run_inference_for_experiment_stream(selected_experiment, inp)
                        analyzer_out_root = out_root
                        status_box.info("Running: " + " ".join(cmd))
                    else:
                        st_.warning("No experiment selected to run. Choose an experiment from the dropdown.")
                        proc = None

                elif model_source == "Use a model file from an experiment":
                    if model_choice is not None:
                        status_box.info("Running analyzer with chosen model file...")
                        # Write analyzer outputs into a persistent folder next to the input folder
                        out_root = Path(inp).parent / 'low_quality_inference' / f'ui_{stamp}'
                        out_root.mkdir(parents=True, exist_ok=True)
                        proc, out_root, cmd = engine.run_analyzer_cli_stream(input_dir=inp, out_dir=out_root, model_path=model_choice)
                        analyzer_out_root = out_root
                        status_box.info("Running: " + " ".join(cmd))
                    else:
                        st_.warning("No model file selected. Choose a model from the experiment or upload a .tflite.")
                        proc = None

                else:  # uploaded model
                    if uploaded_model is None:
                        st_.warning("No uploaded model found. Upload a .tflite first.")
                        proc = None
                    else:
                        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='_'+uploaded_model.name)
                        tmp.write(uploaded_model.getvalue())
                        tmp.flush()
                        tmp.close()
                        tmp_model_f = Path(tmp.name)
                        status_box.info("Running analyzer with uploaded model...")
                        out_root = Path(inp).parent / 'low_quality_inference' / f'ui_{stamp}'
                        out_root.mkdir(parents=True, exist_ok=True)
                        proc, out_root, cmd = engine.run_analyzer_cli_stream(input_dir=inp, out_dir=out_root, model_path=tmp_model_f)
                        analyzer_out_root = out_root
                        status_box.info("Running: " + " ".join(cmd))

                if proc is not None:
                    # Stream stdout lines to UI
                    try:
                        progress_counter = 0
                        # Use text for streaming output instead of text_area to avoid widget duplication
                        log_text = f"Started analyzer. Command: {' '.join(cmd)}\n\nLogs will appear here as the analyzer runs..."
                        log_area.text(log_text)
                        progress_box.progress(1)
                        for line in proc.stdout:
                            line = line.rstrip('\n')
                            log_lines.append(line)
                            now = time.time()
                            if now - last_ui > 0.1:
                                # update UI periodically
                                progress_counter = min(progress_counter + 1, 99)
                                progress_box.progress(progress_counter)
                                log_area.text("\n".join(log_lines[-100:]))
                                status_box.info(f"Running analyzer... output saved to: {out_root}")
                                last_ui = now

                        ret = proc.wait()
                        progress_box.progress(100)
                        log_area.text("\n".join(log_lines[-200:]))
                        if ret != 0:
                            raise RuntimeError(f"Analyzer failed with exit code {ret}\nLogs:\n" + "\n".join(log_lines[-800:]))

                        # Aggregate outputs now that analyzer completed
                        df_out = engine.collect_per_file_max(out_root)
                        st_.success(f"Inference complete — {len(df_out)} rows of predictions found.")
                        st_.info(f"Analyzer outputs saved to: {out_root}")
                    except Exception as e:
                        st_.error(f"Analyzer run failed: {e}")
                        df_out = None

                    # match to input files (use same matching logic)
                if df_out is not None:
                    df_matched = _match_files(df_out, inp)
                    if df_matched is None or df_matched.empty:
                        st_.warning("No matches found between analyzer output and input files.")
                    else:
                        st_.success(f"Matched {len(df_matched)} candidates to files in {inp}")
                        # Persist the matched DataFrame to session state for UI actions
                        st.session_state['hn_df'] = df_matched
                        st.session_state['hn_input_dir'] = str(inp)

                        # Save results to CSV for downstream curation.
                        # Choose an output root: prefer experiment-based storage when an experiment was selected,
                        # otherwise fall back to a scripts-level results folder.
                        stamp = int(time.time())
                        out_root = None
                        source_label = None
                        model_label = None
                        try:
                            if model_source == "Use experiment (canonical analyzer args)" and selected_experiment and selected_experiment != '(none)':
                                out_root = Path('experiments') / selected_experiment / 'low_quality_inference' / 'results'
                                source_label = selected_experiment
                                model_label = '(experiment-canonical)'
                            elif model_source == "Use a model file from an experiment" and selected_experiment and selected_experiment != '(none)':
                                out_root = Path('experiments') / selected_experiment / 'low_quality_inference' / 'results'
                                source_label = selected_experiment
                                model_label = Path(model_choice).name if model_choice is not None else None
                            else:
                                # Fallback: place results under scripts/low_quality_inference/results
                                out_root = Path('scripts') / 'low_quality_inference' / 'results'
                                source_label = '(ad-hoc)'
                                if uploaded_model is not None:
                                    model_label = uploaded_model.name
                                elif model_choice is not None:
                                    model_label = Path(model_choice).name
                                else:
                                    model_label = None
                        except Exception:
                            out_root = Path('scripts') / 'low_quality_inference' / 'results'

                        # Prefer analyzer_out_root (persistent output next to input) when available
                        save_root = Path(analyzer_out_root) if analyzer_out_root is not None else out_root
                        save_root.mkdir(parents=True, exist_ok=True)
                        out_csv = save_root / f'low_quality_radr_max_{stamp}.csv'

                        # Add metadata columns to the saved CSV
                        df_to_save = df_matched.copy()
                        df_to_save['saved_at'] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
                        df_to_save['source'] = source_label
                        df_to_save['model'] = model_label
                        try:
                            df_to_save.to_csv(out_csv, index=False)
                            st_.info(f"Saved per-file RADR CSV to: {out_csv}")
                        except Exception as e:
                            st_.warning(f"Failed to write results CSV: {e}")

            except Exception as e:
                st_.error(f"Inference failed: {e}")
            finally:
                if tmp_model_f is not None:
                    try:
                        Path(tmp_model_f).unlink()
                    except Exception:
                        pass

    # Handle uploaded CSV workflow
    if load_csv_btn:
        if csv_upload is None:
            st_.warning("Upload a CSV first.")
        else:
            try:
                df = pd.read_csv(csv_upload)
                # normalize and try to find confidence/file cols
                cols = {c: c.strip() for c in df.columns}
                df.rename(columns=cols, inplace=True)
                conf_col = next((c for c in df.columns if c.lower().endswith('confidence') or 'conf' in c.lower()), None)
                file_col = next((c for c in df.columns if c.lower() == 'file' or c.lower().endswith('file')), None)
                if conf_col is None:
                    st_.error('Could not detect a confidence column in uploaded CSV')
                else:
                    if file_col is None:
                        file_col = df.columns[0]
                    df2 = df[[file_col, conf_col]].copy()
                    df2.columns = ['File', 'radr_max_confidence']
                    inp = Path(input_dir)
                    df_matched = _match_files(df2, inp)
                    if df_matched.empty:
                        st_.warning('No matches found between CSV rows and files in input folder.')
                    else:
                        st_.success(f"Matched {len(df_matched)} rows to files in {inp}")
                        st.session_state['hn_df'] = df_matched
                        st.session_state['hn_input_dir'] = str(inp)
            except Exception as e:
                st_.error(f"Failed to parse uploaded CSV: {e}")

    # If previously loaded, use session copy
    if 'hn_df' in st.session_state and (df_matched is None):
        try:
            df_matched = st.session_state.get('hn_df')
        except Exception:
            df_matched = None

    # Aggregate past runs UI (available even if no per-run df is loaded)
    with st_.expander("Aggregate past runs", expanded=False):
        st_.write("Search common locations for per-run RADR CSVs and aggregate them into a master file.")
        include_experiments = st_.checkbox("Include experiments/*/low_quality_inference/results", value=True)
        include_scripts = st_.checkbox("Include scripts/low_quality_inference/results", value=True)
        extra_paths = st_.text_input("Additional paths (comma-separated)", value="")
        agg_btn = st_.button("Aggregate results")

        if agg_btn:
            paths = []
            if include_experiments:
                paths.append(str(Path('experiments')))
            if include_scripts:
                paths.append(str(Path('scripts') / 'low_quality_inference' / 'results'))
            for p in [pp.strip() for pp in extra_paths.split(',') if pp.strip()]:
                paths.append(p)

            if not paths:
                st_.warning("No paths provided to search for per-run CSVs. Add paths or enable the checkboxes.")
            else:
                try:
                    with st_.spinner("Aggregating per-run RADR CSVs..."):
                        master = curator.aggregate_results(paths)
                    if master is None or master.empty:
                        st_.warning("No per-run RADR CSVs found in the provided paths.")
                    else:
                        st_.success(f"Aggregated {len(master)} unique files from provided paths.")
                        # store master DF and make it the active candidate set for the selection UI
                        st.session_state['hn_master_df'] = master
                        st.session_state['hn_df'] = master
                        st.session_state['hn_input_dir'] = ''
                        # offer to save master CSV
                        try:
                            stamp = int(time.time())
                            out_root = Path('scripts') / 'low_quality_inference' / 'results'
                            out_root.mkdir(parents=True, exist_ok=True)
                            out_csv = out_root / f'master_low_quality_radr_max_{stamp}.csv'
                            master.to_csv(out_csv, index=False)
                            st_.info(f"Saved master aggregated CSV to: {out_csv}")
                        except Exception:
                            pass
                except Exception as e:
                    st_.error(f"Aggregation failed: {e}")

    if df_matched is None or df_matched.empty:
        st_.info("Load results to begin. Run inference, upload a per-file CSV, or aggregate past runs to proceed.")
        return

    # If an aggregated master exists, offer a download / save-as control
    if 'hn_master_df' in st.session_state:
        try:
            master_df = st.session_state.get('hn_master_df')
            if isinstance(master_df, pd.DataFrame) and not master_df.empty:
                st_.markdown("---")
                st_.subheader("Master aggregated results")
                st_.write(f"Master contains {len(master_df)} unique files (highest RADR per file).")
                fname = st_.text_input("Save master CSV as (filename)", value=f"master_low_quality_radr_max_{int(time.time())}.csv")
                csv_bytes = master_df.to_csv(index=False).encode('utf-8')
                st_.download_button("Download master CSV", data=csv_bytes, file_name=fname, mime='text/csv')
        except Exception:
            # non-critical: if download UI fails, continue
            pass

    # Show table and selection
    st_.markdown("---")
    st_.subheader("Candidate files")
    df_display = df_matched.sort_values("radr_max_confidence", ascending=False).reset_index(drop=True)
    df_display.index.name = 'idx'
    st_.dataframe(df_display, use_container_width=True)

    n_preview = min(100, len(df_display))
    st_.caption(f"Showing {len(df_display)} candidates (top {n_preview} previewable). Use the selection controls below.")

    # Selection controls
    cols = st_.columns([1, 1, 1, 2])
    with cols[0]:
        select_mode = st_.selectbox("Select by", options=["top_k", "top_pct", "by_index", "threshold", "manual"], index=0, format_func=lambda x: {"top_k":"Top K","top_pct":"Top %","by_index":"By index list","threshold":"By confidence threshold","manual":"Manual select"}[x])
    with cols[1]:
        top_k = st_.number_input("K (for Top K)", min_value=1, max_value=len(df_display), value=min(50, len(df_display)))
    with cols[2]:
        pct = st_.slider("Top percentile (%)", min_value=1, max_value=100, value=10)
    with cols[2]:
        thresh = st_.slider("Min confidence", min_value=0.0, max_value=1.0, value=0.5)
    with cols[3]:
        manual_idxs = st_.text_input("Indexes (comma-separated) for manual selection", value="")

    selected_paths: List[Path] = []
    if select_mode == 'top_k':
        selected = df_display.head(int(top_k))
        selected_paths = [Path(p) for p in selected['File'].tolist()]
    elif select_mode == 'top_pct':
        sel_n = max(1, int(len(df_display) * (float(pct) / 100.0)))
        selected = df_display.head(sel_n)
        selected_paths = [Path(p) for p in selected['File'].tolist()]
    elif select_mode == 'threshold':
        sel = df_display[df_display['radr_max_confidence'] >= float(thresh)]
        selected_paths = [Path(p) for p in sel['File'].tolist()]
    elif select_mode == 'by_index':
        # Provide a small multi-select to pick a few top indices
        choices = [f"{i}: {Path(row['File']).name} ({row['radr_max_confidence']:.3f})" for i, row in df_display.head(200).iterrows()]
        picked = st_.multiselect("Pick rows (showing top 200)", options=choices)
        picked_idx = [int(s.split(":", 1)[0]) for s in picked]
        selected_paths = [Path(df_display.loc[i, 'File']) for i in picked_idx]
    else:
        # manual parse
        s = manual_idxs.strip()
        if s:
            try:
                idxs = [int(x.strip()) for x in s.split(",") if x.strip()]
                selected_paths = [Path(df_display.loc[i, 'File']) for i in idxs if i in df_display.index]
            except Exception:
                st_.error("Failed to parse index list. Use commas between numeric indexes.")

    st_.markdown("---")
    st_.subheader("Preview selected (up to 8)")
    preview_paths = selected_paths[:8]
    if preview_paths:
        for p in preview_paths:
            try:
                st_.write(f"{p.name} — {p}")
                st_.audio(str(p))
            except Exception as e:
                st_.warning(f"Could not preview {p}: {e}")
    else:
        st_.info("No files selected yet.")

    st_.markdown("---")
    st_.subheader("Export selection")
    out_label = st_.text_input("Subfolder label (e.g., hardneg_small)", value="hardneg_manual")
    out_root = st_.text_input("Output root folder", value=str(default_outroot))
    # link/copy method selection for curated exports
    link_method = st_.selectbox("Place files as", options=["copy", "hardlink", "symlink"], index=0, format_func=lambda x: {"copy":"Copy","hardlink":"Hard link","symlink":"Symlink"}[x])

    copy_btn = st_.button("Copy selected files to output")
    curate_btn = st_.button("Curate top % to folder")

    if copy_btn:
        if not selected_paths:
            st_.warning("No files selected to copy.")
        else:
            dest = Path(out_root) / out_label
            created = _copy_files(selected_paths, dest)
            st_.success(f"Copied {created} files to {dest}")
            # write selection report
            report = {
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                "count": len(selected_paths),
                "out_label": out_label,
                "out_root": str(Path(out_root).resolve()),
                "selection_mode": select_mode,
                "criteria": {
                    "top_k": int(top_k),
                    "threshold": float(thresh),
                    "manual_indexes": manual_idxs,
                }
            }
            try:
                with open(dest / "selection_report.json", "w", encoding="utf-8") as fh:
                    json.dump(report, fh, indent=2)
                st_.info(f"Wrote selection_report.json to {dest}")
            except Exception as e:
                st_.warning(f"Could not write selection report: {e}")

    # Curate top-percentile directly (uses current percentile slider 'pct')
    if curate_btn:
        try:
            # use the dataframe sorted by confidence
            df_sorted = df_display.sort_values("radr_max_confidence", ascending=False).reset_index(drop=True)
            sel_n = max(1, int(len(df_sorted) * (float(pct) / 100.0)))
            selected_df = df_sorted.head(sel_n)
            if selected_df.empty:
                st_.warning("No files selected for curation.")
            else:
                stamp = int(time.time())
                label = f"top{int(pct)}pct_{stamp}_{out_label}"
                dest_root = Path(out_root)
                # call central writer
                curator.write_manifests_and_links(selected_df, label, dest_root, method=link_method, dry_run=False)
                dest = dest_root / label
                count = 0
                try:
                    count = len([p for p in dest.iterdir() if p.is_file()])
                except Exception:
                    pass
                st_.success(f"Wrote curated selection ({count} files) to: {dest}")
                # write selection report
                report = {
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                    "count": int(sel_n),
                    "out_label": label,
                    "out_root": str(dest_root.resolve()),
                    "selection_mode": "top_pct",
                    "criteria": {"top_pct": float(pct), "link_method": link_method},
                }
                try:
                    with open(dest / "selection_report.json", "w", encoding="utf-8") as fh:
                        json.dump(report, fh, indent=2)
                    st_.info(f"Wrote selection_report.json to {dest}")
                except Exception as e:
                    st_.warning(f"Could not write selection report: {e}")
        except Exception as e:
            st_.error(f"Curation failed: {e}")

    # Offer ZIP download of selected set
    st_.markdown("---")
    st_.subheader("Download selection as ZIP")
    zip_name = st_.text_input("ZIP filename", value="hardneg_selection.zip")
    make_zip_btn = st_.button("Create ZIP and download")
    if make_zip_btn:
        if not selected_paths:
            st_.warning("No files selected to include in ZIP.")
        else:
            try:
                tmpf = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
                with zipfile.ZipFile(tmpf.name, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
                    for p in selected_paths:
                        try:
                            zf.write(p, arcname=p.name)
                        except Exception as e:
                            st_.warning(f"Failed to add {p} to ZIP: {e}")
                tmpf.close()
                with open(tmpf.name, 'rb') as fh:
                    data = fh.read()
                st_.download_button("Download ZIP", data=data, file_name=zip_name)
                st_.success("ZIP ready for download")
            except Exception as e:
                st_.error(f"Failed to build ZIP: {e}")
