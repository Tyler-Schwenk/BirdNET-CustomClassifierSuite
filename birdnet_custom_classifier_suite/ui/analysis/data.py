"""
Data loading and validation for the UI.

This module provides a clean interface to load and validate experiment data,
building on top of the core eval_toolkit functionality.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from birdnet_custom_classifier_suite.eval_toolkit import review, signature
from birdnet_custom_classifier_suite.ui.common.types import DEFAULT_RESULTS_PATH


def validate_results_df(df: pd.DataFrame) -> None:
    """Validate that a DataFrame has the expected columns and data types.
    
    Raises:
        ValueError: If the DataFrame is missing required columns or has invalid data.
    """
    if df.empty:
        raise ValueError("DataFrame is empty")

    required_cols = ["experiment.name"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    # Check for any metric columns
    metric_cols = [c for c in df.columns if c.startswith("metrics.")]
    if not metric_cols:
        raise ValueError("No metric columns found (expected 'metrics.*' prefixes)")


class ResultsLoader:
    """Handles loading and basic validation of experiment results."""

    @staticmethod
    def from_path(path: Union[str, Path] = DEFAULT_RESULTS_PATH) -> pd.DataFrame:
        """Load results from a CSV file path."""
        df = review.load_experiments(path)
        validate_results_df(df)
        return df

    @staticmethod
    def from_upload(file: io.BytesIO) -> pd.DataFrame:
        """Load results from an uploaded file buffer."""
        df = pd.read_csv(file)
        validate_results_df(df)
        return df

    @staticmethod
    def ensure_signatures(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure DataFrame has config signatures computed."""
        if "__signature" not in df.columns:
            df = signature.add_signatures(df)
        return df


# Convenience functions
def load_results(path: Optional[Union[str, Path]] = None,
                uploaded_file: Optional[io.BytesIO] = None) -> pd.DataFrame:
    """Load and validate results from either a path or uploaded file.
    
    Args:
        path: Path to CSV file (defaults to results/all_experiments.csv)
        uploaded_file: File-like object containing CSV data
        
    Returns:
        DataFrame with experiment results
        
    Raises:
        FileNotFoundError: If path doesn't exist
        ValueError: If data is invalid
    """
    if uploaded_file is not None:
        df = ResultsLoader.from_upload(uploaded_file)
    else:
        path = path or DEFAULT_RESULTS_PATH
        df = ResultsLoader.from_path(path)

    return ResultsLoader.ensure_signatures(df)