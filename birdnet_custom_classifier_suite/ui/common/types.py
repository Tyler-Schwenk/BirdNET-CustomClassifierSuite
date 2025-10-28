"""
Common type definitions and data structures for the UI modules.

This file contains small, well-typed dataclasses and constants used by the
Streamlit UI components.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd


# ------------------------- Constants ------------------------- #

DEFAULT_RESULTS_PATH = Path("results/all_experiments.csv")
# If a regenerated results file exists, many UIs should prefer it.
# Keep as a separate constant so callers can decide precedence explicitly.
NEW_RESULTS_PATH = Path("results/all_experiments_new.csv")


# ------------------------- Type Definitions ------------------------- #

class MetricGroup(str, Enum):
    """Common metric group prefixes used in the results DataFrame."""

    OOD_BEST_F1 = "metrics.ood.best_f1"
    IID_BEST_F1 = "metrics.iid.best_f1"


@dataclass
class MetricSummary:
    """Summary statistics for a single metric."""

    name: str
    mean: float
    std: float
    cv: Optional[float] = None  # coefficient of variation
    stability: Optional[float] = None  # inverse CV or other stability measure


@dataclass
class ConfigSummary:
    """Summary of a configuration's metrics and metadata."""

    signature: str
    experiment_names: List[str]
    metrics: Dict[str, MetricSummary]
    config_values: Dict[str, Union[str, float, int, bool]]


@dataclass
class PerRunBreakdown:
    """Detailed per-run information for a configuration."""

    signature: str
    rows: pd.DataFrame
    metric_columns: List[str]
    config_columns: List[str]
    aggregates: Dict[str, Tuple[Optional[float], Optional[float]]]  # metric -> (mean, std)


@dataclass
class UIState:
    """Global UI state container persisted in `st.session_state`.

    Fields are intentionally simple so the state is serializable by Streamlit.
    """

    # Data source
    data_source: Optional[Union[str, Path]] = None
    results_df: Optional[pd.DataFrame] = None

    # Analysis settings
    metric_prefix: str = MetricGroup.OOD_BEST_F1
    top_n: int = 10
    precision_floor: Optional[float] = None

    # Optional filters (values selected by the user)
    quality_filter: Optional[List] = None
    balance_filter: Optional[List] = None
    sweep_filter: Optional[List] = None

    # Selected UI items
    selected_signature: Optional[str] = None

    # Computed results (filled by `summarize_metrics`)
    summaries: List[ConfigSummary] = field(default_factory=list)


# Useful display/config constants
DISPLAY_COLUMNS = ["__signature", "experiment.names"]

IMPORTANT_CONFIG_FIELDS = [
    "dataset.filters.quality",
    "dataset.filters.balance",
    "training.mixup",
    "training.label_smoothing",
    "training.label-smoothing",
    "training.upsampling.mode",
    "training.upsampling.ratio",
    "training.upsampling.factor",
    "upsampling.ratio",
    "upsample_ratio",
    "data.upsample_ratio",
]