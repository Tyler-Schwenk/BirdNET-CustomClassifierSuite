"""
Hard-negative mining UI panel for Streamlit.

Provides tools to:
- Run inference on audio folders
- Load and match existing RADR CSV results
- Preview and select candidates
- Export curated hard-negative sets
"""
from __future__ import annotations

import logging
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

from birdnet_custom_classifier_suite.ui.hard_negative import constants, utils
from birdnet_custom_classifier_suite.ui.hard_negative.models import (
    ExportConfig,
    ExportMethod,
    InferenceConfig,
    LinkMethod,
    ModelSource,
    SelectionConfig,
    SelectionMode,
)
from birdnet_custom_classifier_suite.ui.hard_negative.workflows import HardNegativeMiner

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _render_folder_picker(st_: st) -> str:
    """Render folder selection UI with native dialog and text input fallback."""
    if 'hn_input_dir' not in st.session_state:
        st.session_state['hn_input_dir'] = str(constants.DEFAULT_INPUT_DIR)
    
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
    
    return st_.text_input("Input folder containing audio files", key='hn_input_dir')


def _get_experiment_list(st_: st) -> List[str]:
    """Get cached list of experiments or scan directory."""
    col_exp, col_refresh = st_.columns([4, 1])
    
    with col_refresh:
        if st_.button("Refresh", key='hn_refresh_exp', help="Refresh experiment list"):
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
    model_source: ModelSource,
    exp_names: List[str],
    st_: st
) -> dict:
    """
    Render model selection UI based on source type.
    
    Returns:
        Dictionary with model configuration
    """
    config = {
        'selected_experiment': None,
        'model_path': None,
        'uploaded_model': None,
        'run_infer_btn': False
    }
    
    if model_source == ModelSource.EXPERIMENT_CANONICAL:
        config['selected_experiment'] = st_.selectbox(
            "Choose experiment to run (uses same analyzer args)",
            options=["(none)"] + exp_names,
            index=0
        )
        config['run_infer_btn'] = st_.button("Run inference using selected experiment")

    elif model_source == ModelSource.EXPERIMENT_MODEL_FILE:
        config['selected_experiment'] = st_.selectbox(
            "Choose experiment to pick model from",
            options=["(none)"] + exp_names,
            index=0
        )
        
        model_files = []
        if config['selected_experiment'] and config['selected_experiment'] != '(none)':
            exp_dir = constants.EXPERIMENTS_ROOT / config['selected_experiment']
            model_files = utils.get_experiment_model_files(exp_dir)
        
        if model_files:
            model_labels = [str(p) for p in model_files]
            sel = st_.selectbox("Choose a model file", options=["(none)"] + model_labels, index=0)
            if sel and sel != '(none)':
                config['model_path'] = Path(sel)
        else:
            st_.info("No model files found. You may upload a .tflite instead.")

        config['run_infer_btn'] = st_.button("Run inference using chosen model file")

    else:  # UPLOADED_FILE
        config['uploaded_model'] = st_.file_uploader("Upload a TFLite model (.tflite)", type=['tflite'])
        config['run_infer_btn'] = st_.button("Run inference with uploaded model")
    
    return config


def _run_inference_workflow(config: Dict, st_: st) -> Optional[pd.DataFrame]:
    """Execute inference workflow using HardNegativeMiner."""
    input_dir = config['input_dir']
    model_source = config['model_source']
    selected_experiment = config.get('selected_experiment')
    model_path = config.get('model_path')
    uploaded_model = config.get('uploaded_model')
    target_species = config.get('target_species', constants.DEFAULT_TARGET_SPECIES)
    
    if not input_dir.exists() or not any(input_dir.iterdir()):
        st_.warning(f"Input folder empty or not found: `{input_dir}`")
        return None
    
    st_.info("Running inference — this can take a while depending on the model and number of files.")
    
    try:
        # Handle uploaded model (save to temp file)
        tmp_model_path = None
        if model_source == ModelSource.UPLOADED_FILE and uploaded_model:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='_' + uploaded_model.name)
            tmp.write(uploaded_model.getvalue())
            tmp.flush()
            tmp.close()
            tmp_model_path = Path(tmp.name)
            model_path = tmp_model_path
        
        # Create InferenceConfig
        inference_config = InferenceConfig(
            input_dir=input_dir,
            model_source=model_source,
            target_species=target_species,
            experiment_name=selected_experiment if selected_experiment != "(none)" else None,
            model_path=model_path,
            experiments_root=constants.EXPERIMENTS_ROOT
        )
        
        # Validate configuration
        validation_errors = inference_config.validate()
        if validation_errors:
            for error in validation_errors:
                st_.error(str(error))
            return None
        
        # Run inference using HardNegativeMiner
        miner = HardNegativeMiner()
        
        # Show progress UI
        status_box = st_.empty()
        progress_box = st_.progress(0)
        log_area = st_.empty()

        status_box.info("Running analyzer...")
        progress_box.progress(50)
        
        result = miner.run_inference(inference_config)
        
        if not result.success:
            error_msg = "\n".join(result.errors) if result.errors else "Unknown error"
            st_.error(f"Inference failed: {error_msg}")
            if result.logs:
                with st_.expander("View logs"):
                    st_.code("\n".join(result.logs[-500:]))
            return None
        
        progress_box.progress(100)
        status_box.success(f"Inference complete — {len(result.dataframe)} candidates found")

        if result.logs:
            with st_.expander("View logs"):
                st_.code("\n".join(result.logs[-200:]))
        
        st_.info(f"Analyzer outputs saved to: `{result.output_path}`")
        
        # Provide download button for CSV
        if result.csv_path and result.csv_path.exists():
            with open(result.csv_path, 'rb') as f:
                csv_data = f.read()
            st_.download_button(
                label="Download Results CSV",
                data=csv_data,
                file_name=result.csv_path.name,
                mime='text/csv',
                type="primary"
            )
            
            # Store in session state
            st.session_state['hn_csv_path'] = str(result.csv_path)
        
        # Store results in session state
        st.session_state['hn_df'] = result.dataframe
        
        return result.dataframe
    
    except Exception as e:
        logger.exception("Inference workflow failed")
        st_.error(f"Inference failed: {e}")
        return None
    
    finally:
        # Clean up temporary model file
        if tmp_model_path:
            try:
                tmp_model_path.unlink()
            except Exception:
                pass


def _load_csv_workflow(config: Dict, st_: st) -> Optional[pd.DataFrame]:
    """Load and match existing CSV workflow using HardNegativeMiner."""
    csv_upload = config.get('csv_upload')
    input_dir = config['input_dir']
    
    if csv_upload is None:
        st_.warning("Upload a CSV first.")
        return None
    
    try:
        # Save uploaded CSV to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv', mode='wb') as tmp:
            tmp.write(csv_upload.getvalue())
            tmp_csv_path = Path(tmp.name)
        
        # Use HardNegativeMiner to load and match
        miner = HardNegativeMiner()
        result = miner.load_from_csv(tmp_csv_path, input_dir)
        
        # Clean up temp file
        try:
            tmp_csv_path.unlink()
        except Exception:
            pass
        
        if not result.success:
            error_msg = "\n".join(result.errors) if result.errors else "Unknown error"
            st_.error(f'Failed to load CSV: {error_msg}')
            return None
        
        if result.dataframe.empty:
            st_.warning('No matches found between CSV rows and files in input folder.')
            return None

        st_.success(f"Matched {len(result.dataframe)} rows to files in `{input_dir}`")

        # Store in session state
        st.session_state['hn_df'] = result.dataframe

        return result.dataframe
    
    except Exception as e:
        logger.exception("CSV load workflow failed")
        st_.error(f"Failed to load CSV: {e}")
        return None


def _render_selection_ui(df_display: pd.DataFrame, st_: st) -> List[Path]:
    """Render selection controls and return selected file paths."""
    st_.markdown("---")
    st_.subheader("Select Files")
    
    # Selection mode
    select_mode = st_.radio(
        "Selection method:",
        options=["top_k", "top_pct", "threshold", "manual_index", "manual_pick"],
        format_func=lambda x: {
            "top_k": "Top K files (by confidence)",
            "top_pct": "Top X% (by confidence)",
            "threshold": "Confidence threshold",
            "manual_index": "Manual index list",
            "manual_pick": "Pick from list"
        }[x],
        horizontal=True
    )
    
    # Parameters based on selection mode
    selected_paths = []
    selection_count = 0
    
    if select_mode == 'top_k':
        top_k = st_.number_input(
            "Number of files to select:",
            min_value=1,
            max_value=len(df_display),
            value=min(50, len(df_display)),
            help="Select the top K files with highest confidence"
        )
        selected = df_display.head(int(top_k))
        selected_paths = [Path(p) for p in selected['File'].tolist()]
        selection_count = len(selected_paths)
        
    elif select_mode == 'top_pct':
        pct = st_.slider(
            "Percentage to select:",
            min_value=1,
            max_value=100,
            value=10,
            help="Select top X% of files by confidence"
        )
        sel_n = max(1, int(len(df_display) * (float(pct) / 100.0)))
        selected = df_display.head(sel_n)
        selected_paths = [Path(p) for p in selected['File'].tolist()]
        selection_count = len(selected_paths)
        st.session_state['hn_curate_pct'] = pct  # Store for later use
        
    elif select_mode == 'threshold':
        thresh = st_.slider(
            "Minimum confidence:",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
            help="Select all files with confidence >= threshold"
        )
        sel = df_display[df_display['radr_max_confidence'] >= float(thresh)]
        selected_paths = [Path(p) for p in sel['File'].tolist()]
        selection_count = len(selected_paths)
        
    elif select_mode == 'manual_index':
        manual_idxs = st_.text_input(
            "Enter row indexes (comma-separated):",
            value="",
            placeholder="e.g., 0,1,5,12",
            help="Enter the row numbers from the table above"
        )
        s = manual_idxs.strip()
        if s:
            try:
                idxs = [int(x.strip()) for x in s.split(",") if x.strip()]
                selected_paths = [
                    Path(df_display.loc[i, 'File'])
                    for i in idxs
                    if i in df_display.index
                ]
                selection_count = len(selected_paths)
            except Exception:
                st_.error("Invalid index format. Use comma-separated numbers.")
                
    else:  # manual_pick
        st_.caption(f"Showing top 200 files for selection:")
        choices = [
            f"{i}: {Path(row['File']).name} (conf: {row['radr_max_confidence']:.3f})"
            for i, row in df_display.head(200).iterrows()
        ]
        picked = st_.multiselect(
            "Select files:",
            options=choices,
            help="Pick individual files from the list"
        )
        picked_idx = [int(s.split(":", 1)[0]) for s in picked]
        selected_paths = [Path(df_display.loc[i, 'File']) for i in picked_idx]
        selection_count = len(selected_paths)
    
    # Selection feedback
    if selection_count > 0:
        st_.success(f"**{selection_count} files selected**")
    else:
        st_.info("No files selected yet")
    
    # Store selection in session state
    st.session_state['hn_selected_paths'] = selected_paths
    st.session_state['hn_selection_count'] = selection_count
    
    return selected_paths


def panel(container=None):
    """Main hard-negative mining panel entry point."""
    st_ = container if container is not None else st
    
    # Render header
    _render_header(st_)
    
    # Get configuration from UI
    config = _render_input_config(st_)
    
    # Execute workflow (inference or CSV load)
    df_matched = _execute_workflow(config, st_)
    
    # Show empty state if no data
    if df_matched is None or df_matched.empty:
        _show_empty_state(st_)
        return
    
    # Display results and handle selection/export
    _display_results(df_matched, st_)
    _handle_selection_and_export(df_matched, st_)


def _render_header(st_: st) -> None:
    """Render panel header and description."""
    st_.write(
        "Find files where your model incorrectly predicts the target species. "
        "These hard negatives are valuable for improving model accuracy."
    )


def _render_aggregation_section(st_: st) -> None:
    """Render aggregation section for combining past experiment results."""
    st_.markdown("---")
    st_.subheader("Or: aggregate past runs")
    st_.caption("Combine results from multiple past inference runs into a single CSV.")
    
    search_patterns = st_.text_area(
        "Search patterns (one per line):",
        value="AudioData/low_quality_inference/**/low_quality_radr_max_*.csv",
        help="Glob patterns to find CSV files from past runs"
    )
    
    if st_.button("Aggregate past runs", key='hn_aggregate'):
        patterns = [p.strip() for p in search_patterns.strip().split('\n') if p.strip()]
        if not patterns:
            st_.warning("No search patterns provided")
            return
        
        try:
            search_paths = [Path(p) for p in patterns]
            miner = HardNegativeMiner()
            df_agg = miner.aggregate_results(search_paths)
            
            if df_agg is None or df_agg.empty:
                st_.warning("No results found matching patterns")
                return
            
            st_.success(f"Aggregated {len(df_agg)} rows from past runs")
            st.session_state['hn_df'] = df_agg
            st.rerun()
            
        except Exception as e:
            logger.exception("Aggregation failed")
            st_.error(f"Aggregation failed: {e}")


def _render_preview(selected_paths: List[Path], st_: st) -> None:
    """Render preview of selected files."""
    st_.markdown("---")
    st_.subheader("Preview Selection")
    
    with st_.expander(f"View {len(selected_paths)} selected files", expanded=False):
        preview_df = pd.DataFrame({
            'File': [p.name for p in selected_paths],
            'Full Path': [str(p) for p in selected_paths]
        })
        st_.dataframe(preview_df, width='stretch')


def _render_export_ui(selected_paths: List[Path], st_: st) -> None:
    """Render export UI for selected files."""
    st_.markdown("---")
    st_.subheader("Export Selection")
    
    # Export method
    export_method_str = st_.radio(
        "Export method:",
        options=["Copy to folder", "Download as ZIP"],
        horizontal=True,
        key='hn_export_method'
    )
    
    export_method = ExportMethod.FOLDER if export_method_str == "Copy to folder" else ExportMethod.ZIP
    
    if export_method == ExportMethod.FOLDER:
        # Folder export options
        col1, col2 = st_.columns(2)
        
        with col1:
            output_root = st_.text_input(
                "Output root directory:",
                value=str(constants.OUTPUT_ROOT),
                help="Base directory for exports"
            )
        
        with col2:
            subfolder = st_.text_input(
                "Subfolder name:",
                value=f"curated_{int(time.time())}",
                help="Subfolder within output root"
            )
        
        # Link method
        link_method_str = st_.radio(
            "File link method:",
            options=["Copy files", "Hard link", "Symbolic link"],
            horizontal=True,
            help="How to create file references",
            key='hn_link_method'
        )
        
        link_method_map = {
            "Copy files": LinkMethod.COPY,
            "Hard link": LinkMethod.HARDLINK,
            "Symbolic link": LinkMethod.SYMLINK,
        }
        link_method = link_method_map[link_method_str]
        
        # Export button
        if st_.button("Export to folder", type="primary", key='hn_export_folder'):
            try:
                export_config = ExportConfig(
                    method=ExportMethod.FOLDER,
                    output_root=Path(output_root),
                    subfolder=subfolder,
                    link_method=link_method
                )
                
                miner = HardNegativeMiner()
                result = miner.export_files(selected_paths, export_config)
                
                if result.success:
                    st_.success(
                        f"Exported {result.files_exported}/{result.total_files} files to `{result.destination}`"
                    )
                else:
                    error_msg = "\n".join(result.errors) if result.errors else "Unknown error"
                    st_.error(f"Export failed: {error_msg}")
                    
            except Exception as e:
                logger.exception("Export to folder failed")
                st_.error(f"Export failed: {e}")
    
    else:  # Download as ZIP
        zip_name = st_.text_input(
            "ZIP filename:",
            value=f"hard_negatives_{int(time.time())}.zip",
            help="Name for the ZIP file"
        )
        
        if st_.button("Create ZIP download", type="primary", key='hn_export_zip'):
            try:
                # Create ZIP in memory
                import io
                zip_buffer = io.BytesIO()
                
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for file_path in selected_paths:
                        if file_path.exists():
                            zf.write(file_path, arcname=file_path.name)
                
                zip_buffer.seek(0)
                
                st_.download_button(
                    label="Download ZIP",
                    data=zip_buffer.getvalue(),
                    file_name=zip_name,
                    mime='application/zip',
                    type="primary"
                )
                
                st_.success(f"ZIP file ready for download with {len(selected_paths)} files")
                
            except Exception as e:
                logger.exception("ZIP creation failed")
                st_.error(f"Failed to create ZIP: {e}")


def _render_input_config(st_: st) -> dict:
    """
    Render input configuration UI and return config dict.
    
    Returns:
        Dictionary with configuration values
    """
    config = {}
    
    with st_.expander("Input & model", expanded=True):
        # Input folder
        config['input_dir'] = Path(_render_folder_picker(st_))
        
        # Target species
        config['target_species'] = st_.text_input(
            "Target species label",
            value=constants.DEFAULT_TARGET_SPECIES,
            help="Species label to search for in predictions"
        )
        st.session_state['hn_target_species'] = config['target_species']
        
        st_.caption(
            "**Two workflows:** 1) Run inference on audio folder with a model. "
            "2) Upload an existing CSV of confidences and match to files."
        )
        
        # Model source
        model_source_str = st_.radio(
            "Model source:",
            options=[
                "Use experiment (canonical analyzer args)",
                "Use a model file from an experiment",
                "Upload a .tflite file"
            ],
            index=0,
            key='hn_model_source'
        )
        
        # Map to enum
        source_map = {
            "Use experiment (canonical analyzer args)": ModelSource.EXPERIMENT_CANONICAL,
            "Use a model file from an experiment": ModelSource.EXPERIMENT_MODEL_FILE,
            "Upload a .tflite file": ModelSource.UPLOADED_FILE,
        }
        config['model_source'] = source_map[model_source_str]
    
    # Get experiment list and model selection
    exp_names = _get_experiment_list(st_)
    model_config = _render_model_selection(config['model_source'], exp_names, st_)
    config.update(model_config)
    
    # CSV upload alternative
    st_.markdown("---")
    st_.subheader("Or: use an existing per-file CSV")
    config['csv_upload'] = st_.file_uploader(
        "Upload per-file CSV with File and confidence columns",
        type=['csv']
    )
    config['load_csv_btn'] = st_.button("Load CSV and match files")
    
    # Aggregation section
    _render_aggregation_section(st_)
    
    return config


def _execute_workflow(config: dict, st_: st) -> Optional[pd.DataFrame]:
    """
    Execute inference or CSV load workflow.
    
    Args:
        config: Configuration dictionary from UI
        st_: Streamlit module or container
    
    Returns:
        Matched DataFrame or None
    """
    # Check if workflow should execute
    if config.get('run_infer_btn'):
        return _run_inference_workflow(config, st_)
    
    if config.get('load_csv_btn'):
        return _load_csv_workflow(config, st_)
    
    # Try to load from session state
    if 'hn_df' in st.session_state:
        try:
            return st.session_state.get('hn_df')
        except Exception:
            pass
    
    return None


def _show_empty_state(st_: st) -> None:
    """Show message when no data is available."""
    st_.info(
        "**No data loaded yet.**\n\n"
        "• Run inference on an audio folder, or\n"
        "• Upload an existing CSV, or\n"
        "• Aggregate past runs"
    )


def _display_results(df_matched: pd.DataFrame, st_: st) -> None:
    """
    Display results table and download button.
    
    Args:
        df_matched: Matched results DataFrame
        st_: Streamlit module or container
    """
    st_.markdown("---")
    st_.subheader("Candidate files")
    
    # Sort and display
    df_display = df_matched.sort_values("radr_max_confidence", ascending=False).reset_index(drop=True)
    df_display.index.name = 'idx'
    st_.dataframe(df_display, width='stretch')
    st_.caption(f"Showing {len(df_display)} candidates (sorted by confidence, highest first)")
    
    # Download CSV
    st_.markdown("### Download Results CSV")
    st_.write("Per-file maximum confidence scores for hard-negative mining")
    
    if 'hn_csv_path' in st.session_state and Path(st.session_state['hn_csv_path']).exists():
        csv_path = Path(st.session_state['hn_csv_path'])
        with open(csv_path, 'rb') as f:
            csv_data = f.read()
        st_.download_button(
        label="Download Results CSV",
            data=csv_data,
            file_name=csv_path.name,
            mime='text/csv',
            type="primary",
            key='download_saved_csv'
        )
    else:
        csv_bytes = df_display.to_csv(index=False).encode('utf-8')
        timestamp_name = f"low_quality_radr_max_{int(time.time())}.csv"
        st_.download_button(
            label="Download Results CSV",
            data=csv_bytes,
            file_name=timestamp_name,
            mime='text/csv',
            type="primary",
            key='download_generated_csv'
        )


def _handle_selection_and_export(df_matched: pd.DataFrame, st_: st) -> None:
    """
    Handle file selection, preview, and export.
    
    Args:
        df_matched: Matched results DataFrame
        st_: Streamlit module or container
    """
    # Selection UI
    df_display = df_matched.sort_values("radr_max_confidence", ascending=False).reset_index(drop=True)
    selected_paths = _render_selection_ui(df_display, st_)
    
    if not selected_paths:
        return
    
    # Preview
    _render_preview(selected_paths, st_)
    
    # Export
    _render_export_ui(selected_paths, st_)
