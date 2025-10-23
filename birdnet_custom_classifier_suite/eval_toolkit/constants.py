"""
constants.py
Defines shared metric naming and defaults for BirdNET-CustomClassifierSuite evaluation.
"""

# Primary metrics to use for ranking and evaluation
# Order matters - first metric (f1) is used as default for ranking
CORE_METRICS = [
    "metrics.ood.best_f1.f1",       # Primary ranking metric 
    "metrics.ood.best_f1.precision",
    "metrics.ood.best_f1.recall"
]

# Additional metrics that may be included in summaries
SECONDARY_METRICS = [
    "metrics.iid.best_f1.f1", 
    "metrics.iid.best_f1.precision",
    "metrics.iid.best_f1.recall"
]

# Parameters that are not true metrics (excluded from leaderboards)
PARAMETER_COLUMNS = [
    "metrics.iid.best_f1.threshold",
    "metrics.ood.best_f1.threshold"
]


# Default number of top configs to show
DEFAULT_TOP_K = 10

# Exclude meta fields that aren’t true metrics
EXCLUDED_PREFIXES = ["metadata", "experiment", "__"]

# Metadata field name prefixes for flattening JSON summaries
META_PREFIXES = ["experiment", "metadata"]
