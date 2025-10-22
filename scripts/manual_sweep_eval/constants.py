
from dataclasses import dataclass
from typing import Final, Set, List

# Why: Avoid magic numbers/strings and keep behavior easy to tune.
DEFAULT_PRIORITY_METRICS: Final[List[str]] = [
    "ood.best_f1.f1",
    "ood.best_f1.precision",
    "ood.best_f1.recall",
    "iid.best_f1.f1",
]

DEFAULT_TOP_K: Final[int] = 25

# Why: These prefixes are metadata, not part of a trainable config.
META_PREFIXES: Final[Set[str]] = {
    "experiment.",
    "run.",
    "wandb.",
    "mlflow.",
    "timestamp",
}

# Why: These columns should not participate in config signatures.
EXCLUDED_PREFIXES: Final[Set[str]] = {
    *META_PREFIXES,
    "metrics.",        # if present
    "eval.",           # some pipelines store metrics here
}

# Why: Common tokens that imply a dataset split.
SPLIT_TOKENS: Final[Set[str]] = {"iid","ood","val","test","train","cv","holdout"}

# Why: Common metric names we expect to parse.
METRIC_TOKENS: Final[Set[str]] = {
    "f1","precision","recall","accuracy","auc","roc_auc","pr_auc","ap",
    "tp","tn","fp","fn","support",
}

# Why: Seed column names seen across stages.
SEED_CANDIDATES: Final[List[str]] = [
    "experiment.seed","seed","run.seed","training.seed"
]
