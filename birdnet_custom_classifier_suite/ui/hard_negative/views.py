"""
Hard-negative mining UI panel for Streamlit.

Provides tools to:
- Run inference on audio folders
- Load and match existing RADR CSV results
- Preview and select candidates
- Export curated hard-negative sets
"""
from __future__ import annotations

import json
import shutil
import tempfile
import time
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import streamlit as st

from birdnet_custom_classifier_suite.ui.hard_negative import constants, curator, engine, utils


def _render_folder_picker(st_: st) -> str:
    """Render folder selection UI with native dialog and text input fallback."""
    if 'hn_input_dir' not in st.session_state:
        st.session_state['hn_input_dir'] = str(constants.DEFAULT_INPUT_DIR)
    
    try:
        if st_.button("📁 Choose folder (Explorer)", key='hn_choose_folder'):
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
    
    return st_.text_input("Input folder containing audio files", key='hn_input_dir')


def _get_experiment_list(st_: st) -> List[str]:
    """Get cached list of experiments or scan directory."""
    col_exp, col_refresh = st_.columns([4, 1])
    
    with col_refresh:
        if st_.button("↻ Refresh", key='hn_refresh_exp', help="Refresh experiment list"):
            st.session_state.pop('hn_exp_names', None)
            st.rerun()
    
    if 'hn_exp_names' not in st.session_state:
        exp_names = []
        if constants.EXPERIMENTS_ROOT.exists():
            try:
                exp_names = sorted([p.name for p in constants.EXPERIMENTS_ROOT.iterdir() if p.is_dir()])
            except Exception:
                exp_names = []
        st.session_state['hn_exp_names'] = exp_names
    
    return st.session_state['hn_exp_names']


def _render_model_selection(
    model_source: str,
    exp_names: List[str],
    st_: st
) -> Tuple[Optional[str], Optional[Path], Optional[st.runtime.uploaded_file_manager.UploadedFile], bool]:
    """Render model selection UI based on source type."""
    selected_experiment = None
    model_choice = None
    uploaded_model = None
    run_infer_btn = False
    
    if model_source == "Use experiment (canonical analyzer args)":
        selected_experiment = st_.selectbox(
            "Choose experiment to run (uses same analyzer args)",
            options=["(none)"] + exp_names,
            index=0
        )
        run_infer_btn = st_.button("▶️ Run inference using selected experiment")
    
    elif model_source == "Use a model file from an experiment":
        selected_experiment = st_.selectbox(
            "Choose experiment to pick model from",
            options=["(none)"] + exp_names,
            index=0
        )
        
        model_files = []
        if selected_experiment and selected_experiment != '(none)':
            exp_dir = constants.EXPERIMENTS_ROOT / selected_experiment
            model_files = utils.get_experiment_model_files(exp_dir)
        
        if model_files:
            model_labels = [str(p) for p in model_files]
            sel = st_.selectbox("Choose a model file", options=["(none)"] + model_labels, index=0)
            if sel and sel != '(none)':
                model_choice = Path(sel)
        else:
            st_.info("ℹ️ No model files found in the selected experiment. You may upload a .tflite instead.")
        
        run_infer_btn = st_.button("▶️ Run inference using chosen model file")
    
    else:  # Upload model
        uploaded_model = st_.file_uploader("Upload a TFLite model (.tflite)", type=['tflite'])
        run_infer_btn = st_.button("▶️ Run inference with uploaded model")
    
    return selected_experiment, model_choice, uploaded_model, run_infer_btn


def _stream_analyzer_logs(proc, out_root: Path, st_: st) -> Tuple[int, List[str]]:
    """Stream analyzer stdout to UI and return exit code and logs."""
    status_box = st_.empty()
    progress_box = st_.progress(0)
    log_area = st_.empty()
    
    log_lines = []
    last_ui = 0.0
    progress_counter = 0
    
    log_area.text("🚀 Analyzer starting...\n\nWaiting for output...")
    progress_box.progress(1)
    
    for line in proc.stdout:
        line = line.rstrip('\n')
        log_lines.append(line)
        now = time.time()
        
        if now - last_ui > constants.UI_UPDATE_INTERVAL:
            progress_counter = min(progress_counter + 1, 99)
            progress_box.progress(progress_counter)
            log_area.text("\n".join(log_lines[-100:]))
            status_box.info(f"⚙️ Running analyzer... output saved to: `{out_root}`")
            last_ui = now
    
    ret = proc.wait()
    progress_box.progress(100)
    log_area.text("\n".join(log_lines[-200:]))
    
    return ret, log_lines


def _run_inference_workflow(
    input_dir: Path,
    model_source: str,
    selected_experiment: Optional[str],
    model_choice: Optional[Path],
    uploaded_model: Optional[st.runtime.uploaded_file_manager.UploadedFile],
    st_: st
) -> Optional[pd.DataFrame]:
    """Execute inference workflow and return matched DataFrame."""
    if not input_dir.exists() or not any(input_dir.iterdir()):
        st_.warning(f"⚠️ Input folder empty or not found: `{input_dir}`")
        return None
    
    st_.info("🔄 Running inference — this can take a while depending on the model and number of files.")
    
    try:
        stamp = int(time.time())
        tmp_model_f = None
        proc = None
        out_root = None
        analyzer_out_root = None
        
        # Prepare model and output paths
        if model_source == "Use experiment (canonical analyzer args)":
            if not selected_experiment or selected_experiment == '(none)':
                st_.warning("⚠️ No experiment selected to run. Choose an experiment from the dropdown.")
                return None
            
            st_.info("Preparing experiment-based analyzer run...")
            proc, out_root, cmd = engine.run_inference_for_experiment_stream(selected_experiment, input_dir)
            analyzer_out_root = out_root
        
        elif model_source == "Use a model file from an experiment":
            if model_choice is None:
                st_.warning("⚠️ No model file selected. Choose a model from the experiment or upload a .tflite.")
                return None
            
            st_.info("Running analyzer with chosen model file...")
            out_root = input_dir.parent / 'low_quality_inference' / f'ui_{stamp}'
            out_root.mkdir(parents=True, exist_ok=True)
            proc, out_root, cmd = engine.run_analyzer_cli_stream(
                input_dir=input_dir,
                out_dir=out_root,
                model_path=model_choice
            )
            analyzer_out_root = out_root
        
        else:  # Uploaded model
            if uploaded_model is None:
                st_.warning("⚠️ No uploaded model found. Upload a .tflite first.")
                return None
            
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='_' + uploaded_model.name)
            tmp.write(uploaded_model.getvalue())
            tmp.flush()
            tmp.close()
            tmp_model_f = Path(tmp.name)
            
            st_.info("Running analyzer with uploaded model...")
            out_root = input_dir.parent / 'low_quality_inference' / f'ui_{stamp}'
            out_root.mkdir(parents=True, exist_ok=True)
            proc, out_root, cmd = engine.run_analyzer_cli_stream(
                input_dir=input_dir,
                out_dir=out_root,
                model_path=tmp_model_f
            )
            analyzer_out_root = out_root
        
        st_.info(f"💻 Running: `{' '.join(cmd)}`")
        
        # Stream logs
        ret, log_lines = _stream_analyzer_logs(proc, out_root, st_)
        
        if ret != 0:
            raise RuntimeError(
                f"Analyzer failed with exit code {ret}\nLogs:\n" + "\n".join(log_lines[-800:])
            )
        
        # Aggregate outputs
        df_out = engine.collect_per_file_max(out_root)
        st_.success(f"✅ Inference complete — {len(df_out)} rows of predictions found.")
        st_.info(f"📂 Analyzer outputs saved to: `{out_root}`")
        
        # Match files
        df_matched = curator.match_files(df_out, input_dir)
        if df_matched is None or df_matched.empty:
            st_.warning("⚠️ No matches found between analyzer output and input files.")
            return None
        
        st_.success(f"✅ Matched {len(df_matched)} candidates to files in `{input_dir}`")
        
        # Save results
        out_root_path, source_label, model_label = utils.get_output_path_for_model_source(
            model_source, selected_experiment, model_choice, uploaded_model
        )
        save_root = analyzer_out_root if analyzer_out_root else out_root_path
        csv_path = utils.save_results_csv(df_matched, save_root, stamp, source_label, model_label, st_)
        
        # Provide download button
        with open(csv_path, 'rb') as f:
            csv_data = f.read()
        st_.download_button(
            label="📥 Download Results CSV",
            data=csv_data,
            file_name=csv_path.name,
            mime='text/csv',
            type="primary"
        )
        
        # Store in session state
        st.session_state['hn_df'] = df_matched
        st.session_state['hn_input_dir'] = str(input_dir)
        st.session_state['hn_csv_path'] = str(csv_path)
        
        return df_matched
    
    except Exception as e:
        st_.error(f"❌ Inference failed: {e}")
        return None
    
    finally:
        if tmp_model_f:
            try:
                tmp_model_f.unlink()
            except Exception:
                pass


def _load_csv_workflow(csv_upload, input_dir: Path, st_: st) -> Optional[pd.DataFrame]:
    """Load and match existing CSV workflow."""
    if csv_upload is None:
        st_.warning("⚠️ Upload a CSV first.")
        return None
    
    try:
        df = pd.read_csv(csv_upload)
        cols = {c: c.strip() for c in df.columns}
        df.rename(columns=cols, inplace=True)
        
        conf_col = next(
            (c for c in df.columns if c.lower().endswith('confidence') or 'conf' in c.lower()),
            None
        )
        file_col = next(
            (c for c in df.columns if c.lower() == 'file' or c.lower().endswith('file')),
            df.columns[0] if len(df.columns) > 0 else None
        )
        
        if conf_col is None:
            st_.error('❌ Could not detect a confidence column in uploaded CSV')
            return None
        
        df2 = df[[file_col, conf_col]].copy()
        df2.columns = ['File', 'radr_max_confidence']
        
        df_matched = curator.match_files(df2, input_dir)
        if df_matched.empty:
            st_.warning('⚠️ No matches found between CSV rows and files in input folder.')
            return None
        
        st_.success(f"✅ Matched {len(df_matched)} rows to files in `{input_dir}`")
        st.session_state['hn_df'] = df_matched
        st.session_state['hn_input_dir'] = str(input_dir)
        
        return df_matched
    
    except Exception as e:
        st_.error(f"❌ Failed to parse uploaded CSV: {e}")
        return None


def _render_selection_ui(df_display: pd.DataFrame, st_: st) -> List[Path]:
    """Render selection controls and return selected file paths."""
    cols = st_.columns([1, 1, 1, 2])
    
    with cols[0]:
        select_mode = st_.selectbox(
            "Select by",
            options=["top_k", "top_pct", "by_index", "threshold", "manual"],
            index=0,
            format_func=lambda x: {
                "top_k": "Top K",
                "top_pct": "Top %",
                "by_index": "By index list",
                "threshold": "By confidence threshold",
                "manual": "Manual select"
            }[x]
        )
    
    with cols[1]:
        top_k = st_.number_input(
            "K (for Top K)",
            min_value=1,
            max_value=len(df_display),
            value=min(50, len(df_display))
        )
    
    with cols[2]:
        pct = st_.slider("Top percentile (%)", min_value=1, max_value=100, value=10)
    
    with cols[2]:
        thresh = st_.slider("Min confidence", min_value=0.0, max_value=1.0, value=0.5)
    
    with cols[3]:
        manual_idxs = st_.text_input("Indexes (comma-separated) for manual selection", value="")
    
    # Execute selection
    selected_paths = []
    
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
        choices = [
            f"{i}: {Path(row['File']).name} ({row['radr_max_confidence']:.3f})"
            for i, row in df_display.head(200).iterrows()
        ]
        picked = st_.multiselect("Pick rows (showing top 200)", options=choices)
        picked_idx = [int(s.split(":", 1)[0]) for s in picked]
        selected_paths = [Path(df_display.loc[i, 'File']) for i in picked_idx]
    
    else:  # manual
        s = manual_idxs.strip()
        if s:
            try:
                idxs = [int(x.strip()) for x in s.split(",") if x.strip()]
                selected_paths = [
                    Path(df_display.loc[i, 'File'])
                    for i in idxs
                    if i in df_display.index
                ]
            except Exception:
                st_.error("❌ Failed to parse index list. Use commas between numeric indexes.")
    
    return selected_paths


def _render_preview(selected_paths: List[Path], st_: st) -> None:
    """Render audio preview for selected files."""
    st_.markdown("---")
    st_.subheader("🎧 Preview selected (up to 8)")
    
    preview_paths = selected_paths[:8]
    if preview_paths:
        for p in preview_paths:
            try:
                st_.write(f"**{p.name}** — `{p}`")
                st_.audio(str(p))
            except Exception as e:
                st_.warning(f"⚠️ Could not preview {p}: {e}")
    else:
        st_.info("ℹ️ No files selected yet.")


def _copy_files(paths: List[Path], dest: Path) -> int:
    """Copy files to destination, returning count of successfully copied files."""
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
                shutil.copy(p, tgt)
                created += 1
            except Exception:
                pass
    
    return created


def _write_selection_report(dest: Path, report: dict, st_: st) -> None:
    """Write selection report JSON to destination."""
    try:
        with open(dest / "selection_report.json", "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        st_.info(f"📄 Wrote selection_report.json to `{dest}`")
    except Exception as e:
        st_.warning(f"⚠️ Could not write selection report: {e}")


def panel(container=None):
    """Main hard-negative mining panel."""
    st_ = container if container is not None else st
    st_.header("🔍 Hard-negative mining")
    st_.write(
        "Load RADR per-file max confidence CSV, preview candidates, and select files "
        "to copy into curated folders for training."
    )
    
    # === Input & Model Selection ===
    with st_.expander("📥 Input & model", expanded=True):
        input_dir = _render_folder_picker(st_)
        
        st_.caption(
            "**Two workflows:** 1) Run inference on the chosen folder with a model "
            "(pick an experiment model or upload a .tflite). 2) Upload an existing per-file CSV "
            "of confidences (if you already ran analyzer elsewhere) and match rows to files in the input folder."
        )
        
        model_source = st_.radio(
            "Model source:",
            options=[
                "Use experiment (canonical analyzer args)",
                "Use a model file from an experiment",
                "Upload a .tflite file"
            ],
            index=0,
            key='hn_model_source'
        )
    
    # Get experiment list
    exp_names = _get_experiment_list(st_)
    
    # Render model selection
    selected_experiment, model_choice, uploaded_model, run_infer_btn = _render_model_selection(
        model_source, exp_names, st_
    )
    
    # CSV upload alternative
    st_.markdown("---")
    st_.subheader("📄 Or: use an existing per-file CSV")
    csv_upload = st_.file_uploader(
        "Upload per-file CSV with columns File and confidence (or similar)",
        type=['csv']
    )
    load_csv_btn = st_.button("📂 Load CSV and match files")
    
    # === Execute Workflows ===
    df_matched = None
    inp = Path(input_dir)
    
    if run_infer_btn:
        df_matched = _run_inference_workflow(
            inp, model_source, selected_experiment, model_choice, uploaded_model, st_
        )
    
    if load_csv_btn:
        df_matched = _load_csv_workflow(csv_upload, inp, st_)
    
    # Load from session state if available
    if df_matched is None and 'hn_df' in st.session_state:
        try:
            df_matched = st.session_state.get('hn_df')
        except Exception:
            pass
    
    # === Aggregate Past Runs ===
    with st_.expander("📊 Aggregate past runs", expanded=False):
        st_.write("Search common locations for per-run RADR CSVs and aggregate them into a master file.")
        include_experiments = st_.checkbox(
            "Include experiments/*/low_quality_inference/results",
            value=True
        )
        include_scripts = st_.checkbox(
            "Include scripts/low_quality_inference/results",
            value=True
        )
        extra_paths = st_.text_input("Additional paths (comma-separated)", value="")
        agg_btn = st_.button("🔗 Aggregate results")
        
        if agg_btn:
            paths = []
            if include_experiments:
                paths.append(str(constants.EXPERIMENTS_ROOT))
            if include_scripts:
                paths.append(str(Path('scripts') / constants.RESULTS_DIR_NAME / 'results'))
            for p in [pp.strip() for pp in extra_paths.split(',') if pp.strip()]:
                paths.append(p)
            
            if not paths:
                st_.warning("⚠️ No paths provided to search for per-run CSVs. Add paths or enable the checkboxes.")
            else:
                try:
                    with st_.spinner("Aggregating per-run RADR CSVs..."):
                        master = curator.aggregate_results(paths)
                    
                    if master is None or master.empty:
                        st_.warning("⚠️ No per-run RADR CSVs found in the provided paths.")
                    else:
                        st_.success(f"✅ Aggregated {len(master)} unique files from provided paths.")
                        st.session_state['hn_master_df'] = master
                        st.session_state['hn_df'] = master
                        st.session_state['hn_input_dir'] = ''
                        
                        # Save master CSV
                        try:
                            stamp = int(time.time())
                            out_root = Path('scripts') / 'low_quality_inference' / 'results'
                            out_root.mkdir(parents=True, exist_ok=True)
                            out_csv = out_root / f'master_low_quality_radr_max_{stamp}.csv'
                            master.to_csv(out_csv, index=False)
                            st_.info(f"📄 Saved master aggregated CSV to: `{out_csv}`")
                        except Exception:
                            pass
                except Exception as e:
                    st_.error(f"❌ Aggregation failed: {e}")
    
    # === Check if we have data ===
    if df_matched is None or df_matched.empty:
        st_.info("ℹ️ Load results to begin. Run inference, upload a per-file CSV, or aggregate past runs to proceed.")
        return
    
    # === Master CSV Download ===
    if 'hn_master_df' in st.session_state:
        try:
            master_df = st.session_state.get('hn_master_df')
            if isinstance(master_df, pd.DataFrame) and not master_df.empty:
                st_.markdown("---")
                st_.subheader("📦 Master aggregated results")
                st_.write(f"Master contains {len(master_df)} unique files (highest RADR per file).")
                fname = st_.text_input(
                    "Save master CSV as (filename)",
                    value=f"master_low_quality_radr_max_{int(time.time())}.csv"
                )
                csv_bytes = master_df.to_csv(index=False).encode('utf-8')
                st_.download_button("⬇️ Download master CSV", data=csv_bytes, file_name=fname, mime='text/csv')
        except Exception:
            pass
    
    # === Display Candidates ===
    st_.markdown("---")
    st_.subheader("📋 Candidate files")
    df_display = df_matched.sort_values("radr_max_confidence", ascending=False).reset_index(drop=True)
    df_display.index.name = 'idx'
    st_.dataframe(df_display, use_container_width=True)
    
    # Download current results
    st_.markdown("### 📥 Download Results")
    st_.write("This CSV contains each file with its maximum RADR confidence score for hard-negative mining.")
    
    # Check if we have a saved path from the inference workflow
    if 'hn_csv_path' in st.session_state and Path(st.session_state['hn_csv_path']).exists():
        csv_path = Path(st.session_state['hn_csv_path'])
        with open(csv_path, 'rb') as f:
            csv_data = f.read()
        st_.download_button(
            label="📥 Download Results CSV (from disk)",
            data=csv_data,
            file_name=csv_path.name,
            mime='text/csv',
            type="primary",
            key='download_saved_csv'
        )
    else:
        # Generate from current DataFrame
        csv_bytes = df_display.to_csv(index=False).encode('utf-8')
        timestamp_name = f"low_quality_radr_max_{int(time.time())}.csv"
        st_.download_button(
            label="📥 Download Results CSV",
            data=csv_bytes,
            file_name=timestamp_name,
            mime='text/csv',
            type="primary",
            key='download_generated_csv'
        )
    
    n_preview = min(100, len(df_display))
    st_.caption(f"Showing {len(df_display)} candidates (top {n_preview} previewable). Use the selection controls below.")
    
    # === Selection Controls ===
    selected_paths = _render_selection_ui(df_display, st_)
    
    # === Preview ===
    _render_preview(selected_paths, st_)
    
    # === Export Selection ===
    st_.markdown("---")
    st_.subheader("📤 Export selection")
    
    out_label = st_.text_input("Subfolder label (e.g., hardneg_small)", value="hardneg_manual")
    out_root = st_.text_input("Output root folder", value=str(constants.DEFAULT_OUTPUT_ROOT))
    link_method = st_.selectbox(
        "Place files as",
        options=["copy", "hardlink", "symlink"],
        index=0,
        format_func=lambda x: {"copy": "Copy", "hardlink": "Hard link", "symlink": "Symlink"}[x]
    )
    
    copy_btn = st_.button("📋 Copy selected files to output")
    curate_btn = st_.button("🎯 Curate top % to folder")
    
    # Handle copy
    if copy_btn:
        if not selected_paths:
            st_.warning("⚠️ No files selected to copy.")
        else:
            dest = Path(out_root) / out_label
            created = _copy_files(selected_paths, dest)
            st_.success(f"✅ Copied {created} files to `{dest}`")
            
            # Get selection params from UI state (we'd need to pass these or store in session)
            # For now, simplified report
            report = {
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                "count": len(selected_paths),
                "out_label": out_label,
                "out_root": str(Path(out_root).resolve()),
            }
            _write_selection_report(dest, report, st_)
    
    # Handle curate
    if curate_btn:
        try:
            # Get current percentile from selection UI
            # Note: This is a simplified version - in production we'd pass pct as a parameter
            pct = st.session_state.get('hn_curate_pct', 10)
            
            df_sorted = df_display.sort_values("radr_max_confidence", ascending=False).reset_index(drop=True)
            sel_n = max(1, int(len(df_sorted) * (float(pct) / 100.0)))
            selected_df = df_sorted.head(sel_n)
            
            if selected_df.empty:
                st_.warning("⚠️ No files selected for curation.")
            else:
                stamp = int(time.time())
                label = f"top{int(pct)}pct_{stamp}_{out_label}"
                dest_root = Path(out_root)
                
                curator.write_manifests_and_links(
                    selected_df, label, dest_root, method=link_method, dry_run=False
                )
                
                dest = dest_root / label
                count = sum(1 for p in dest.iterdir() if p.is_file()) if dest.exists() else 0
                st_.success(f"✅ Wrote curated selection ({count} files) to: `{dest}`")
                
                report = {
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                    "count": int(sel_n),
                    "out_label": label,
                    "out_root": str(dest_root.resolve()),
                    "selection_mode": "top_pct",
                    "criteria": {"top_pct": float(pct), "link_method": link_method},
                }
                _write_selection_report(dest, report, st_)
        except Exception as e:
            st_.error(f"❌ Curation failed: {e}")
    
    # === ZIP Download ===
    st_.markdown("---")
    st_.subheader("📦 Download selection as ZIP")
    zip_name = st_.text_input("ZIP filename", value="hardneg_selection.zip")
    make_zip_btn = st_.button("📦 Create ZIP and download")
    
    if make_zip_btn:
        if not selected_paths:
            st_.warning("⚠️ No files selected to include in ZIP.")
        else:
            try:
                tmpf = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
                with zipfile.ZipFile(tmpf.name, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
                    for p in selected_paths:
                        try:
                            zf.write(p, arcname=p.name)
                        except Exception as e:
                            st_.warning(f"⚠️ Failed to add {p} to ZIP: {e}")
                tmpf.close()
                
                with open(tmpf.name, 'rb') as fh:
                    data = fh.read()
                st_.download_button("⬇️ Download ZIP", data=data, file_name=zip_name)
                st_.success("✅ ZIP ready for download")
            except Exception as e:
                st_.error(f"❌ Failed to build ZIP: {e}")
