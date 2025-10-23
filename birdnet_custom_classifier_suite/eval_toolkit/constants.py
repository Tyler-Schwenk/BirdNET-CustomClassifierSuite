"""
constants.py
Defines shared metric naming and defaults for BirdNET-CustomClassifierSuite evaluation.
"""

# Default priority metrics for ranking and display
DEFAULT_PRIORITY_METRICS = [
    "metrics.ood.best_f1.f1",
    "metrics.ood.best_f1.precision",
    "metrics.ood.best_f1.recall",
    "metrics.iid.best_f1.f1",
]

# Explicit list of known metric columns
METRIC_COLUMNS = [
    "metrics.iid.best_f1.f1", "metrics.iid.best_f1.precision", "metrics.iid.best_f1.recall",
    "metrics.ood.best_f1.f1", "metrics.ood.best_f1.precision", "metrics.ood.best_f1.recall",
    "metrics.iid.auroc", "metrics.ood.auroc", "metrics.iid.auprc", "metrics.ood.auprc"
]


# Default number of top configs to show
DEFAULT_TOP_K = 10

# Exclude meta fields that aren’t true metrics
EXCLUDED_PREFIXES = ["metadata", "experiment", "__"]

# Metadata field name prefixes for flattening JSON summaries
META_PREFIXES = ["experiment", "metadata"]
