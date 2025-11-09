"""Utility functions for hard-negative mining UI."""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import streamlit as st

from birdnet_custom_classifier_suite.ui.hard_negative import constants
from birdnet_custom_classifier_suite.ui.common import format_file_size, validate_folder_not_empty


def get_output_path_for_model_source(
    model_source: str,
    selected_experiment: Optional[str],
    model_choice: Optional[Path],
    uploaded_model: Optional[st.runtime.uploaded_file_manager.UploadedFile]
) -> Tuple[Path, str, Optional[str]]:
    """
    Determine output path, source label, and model label for results CSV.
    
    Args:
        model_source: Type of model source being used
        selected_experiment: Name of selected experiment (if applicable)
        model_choice: Path to chosen model file (if applicable)
        uploaded_model: Uploaded model file (if applicable)
    
    Returns:
        Tuple of (output_path, source_label, model_label)
    """
    if (
        model_source == "Use experiment (canonical analyzer args)"
        and selected_experiment
        and selected_experiment != '(none)'
    ):
        return (
            constants.EXPERIMENTS_ROOT / selected_experiment / constants.RESULTS_DIR_NAME / 'results',
            selected_experiment,
            '(experiment-canonical)'
        )
    
    elif (
        model_source == "Use a model file from an experiment"
        and selected_experiment
        and selected_experiment != '(none)'
    ):
        return (
            constants.EXPERIMENTS_ROOT / selected_experiment / constants.RESULTS_DIR_NAME / 'results',
            selected_experiment,
            model_choice.name if model_choice else None
        )
    
    else:
        model_label = None
        if uploaded_model:
            model_label = uploaded_model.name
        elif model_choice:
            model_label = model_choice.name
        
        return (
            Path('scripts') / constants.RESULTS_DIR_NAME / 'results',
            '(ad-hoc)',
            model_label
        )


def save_results_csv(
    df: pd.DataFrame,
    save_root: Path,
    stamp: int,
    source_label: str,
    model_label: Optional[str],
    st_: st
) -> Path:
    """
    Save aggregated per-file results CSV with metadata.
    
    Args:
        df: DataFrame to save
        save_root: Root directory for saving
        stamp: Timestamp for filename
        source_label: Label indicating source of results
        model_label: Label indicating model used
        st_: Streamlit module or container
    
    Returns:
        Path to saved CSV file
    """
    save_root.mkdir(parents=True, exist_ok=True)
    out_csv = save_root / f'low_quality_radr_max_{stamp}.csv'
    
    df_to_save = df.copy()
    df_to_save['saved_at'] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    df_to_save['source'] = source_label
    df_to_save['model'] = model_label
    
    try:
        df_to_save.to_csv(out_csv, index=False)
        st_.success(f"Results saved to: `{out_csv}`")
        return out_csv
    except Exception as e:
        st_.warning(f"Failed to write results CSV: {e}")
        return out_csv


def copy_files(paths: List[Path], dest: Path) -> int:
    """
    Copy files to destination directory.
    
    Args:
        paths: List of source file paths
        dest: Destination directory
    
    Returns:
        Number of files successfully copied
    """
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


def write_selection_report(dest: Path, report: dict, st_: st) -> None:
    """
    Write selection report JSON to destination.
    
    Args:
        dest: Destination directory
        report: Report data dictionary
        st_: Streamlit module or container
    """
    try:
        with open(dest / "selection_report.json", "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        st_.info(f"Wrote selection_report.json to `{dest}`")
    except Exception as e:
        st_.warning(f"Could not write selection report: {e}")


def validate_input_directory(input_dir: Path, st_: st) -> bool:
    """
    Validate that input directory exists and contains files.
    Uses common validation utility.
    
    Args:
        input_dir: Path to validate
        st_: Streamlit module or container (unused, kept for compatibility)
    
    Returns:
        True if valid, False otherwise
    """
    return validate_folder_not_empty(input_dir, show_message=True)


def parse_csv_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse CSV to find confidence and file columns.
    
    Args:
        df: DataFrame to parse
    
    Returns:
        Tuple of (confidence_column_name, file_column_name)
    """
    # Normalize column names
    cols = {c: c.strip() for c in df.columns}
    df.rename(columns=cols, inplace=True)
    
    # Find confidence column
    conf_col = next(
        (c for c in df.columns if c.lower().endswith('confidence') or 'conf' in c.lower()),
        None
    )
    
    # Find file column
    file_col = next(
        (c for c in df.columns if c.lower() == 'file' or c.lower().endswith('file')),
        df.columns[0] if len(df.columns) > 0 else None
    )
    
    return conf_col, file_col


# Note: format_file_size is now imported from ui.common.widgets instead of defined here
# Keeping this stub for backward compatibility
def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    Delegates to common utility.
    """
    from birdnet_custom_classifier_suite.ui.common import format_file_size as common_format
    return common_format(size_bytes)


def get_experiment_model_files(exp_dir: Path) -> List[Path]:
    """
    Get list of model files in experiment directory.
    
    Args:
        exp_dir: Experiment directory path
    
    Returns:
        List of model file paths
    """
    model_files = []
    try:
        for ext in constants.MODEL_EXTENSIONS:
            model_files.extend(sorted(exp_dir.glob(f'*{ext}')))
    except Exception:
        pass
    return model_files
