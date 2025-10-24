"""
Package initialization for the UI modules.
"""

from birdnet_custom_classifier_suite.ui.analysis.data import ResultsLoader, load_results
from birdnet_custom_classifier_suite.ui.analysis.metrics import (
    format_metric_value,
    get_signature_breakdown,
    summarize_metrics,
)
from birdnet_custom_classifier_suite.ui.analysis.views import (
    data_loader,
    leaderboard,
    metric_controls,
    signature_details,
)
from birdnet_custom_classifier_suite.ui.common.types import (
    ConfigSummary,
    MetricGroup,
    MetricSummary,
    PerRunBreakdown,
    UIState,
)

__all__ = [
    # Data loading
    "ResultsLoader",
    "load_results",
    # Metrics & analysis
    "format_metric_value",
    "get_signature_breakdown",
    "summarize_metrics",
    # UI components
    "data_loader",
    "leaderboard",
    "metric_controls",
    "signature_details",
    # Types
    "ConfigSummary",
    "MetricGroup",
    "MetricSummary",
    "PerRunBreakdown",
    "UIState",
]