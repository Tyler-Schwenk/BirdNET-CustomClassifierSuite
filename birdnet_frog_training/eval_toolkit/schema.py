
import re
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional
import pandas as pd
from .constants import EXCLUDED_PREFIXES, META_PREFIXES, SPLIT_TOKENS, METRIC_TOKENS, SEED_CANDIDATES

# Why: Keep a structured view of a metric column so we can manipulate it generically.
@dataclass(frozen=True)
class MetricSpec:
    column: str
    split: Optional[str]     # e.g., iid/ood/val/test
    family: Optional[str]    # e.g., best_f1, threshold_0.50
    name: Optional[str]      # e.g., f1/precision/recall

@dataclass(frozen=True)
class Schema:
    seed_col: str
    meta_cols: List[str]
    metric_cols: List[str]
    hparam_cols: List[str]
    unknown_cols: List[str]
    metric_specs: List[MetricSpec]

def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def _is_meta(col: str) -> bool:
    if col in META_PREFIXES: 
        return True
    return any(col.startswith(p) for p in META_PREFIXES if p.endswith("."))

def _is_excluded(col: str) -> bool:
    if col in EXCLUDED_PREFIXES: 
        return True
    return any(col.startswith(p) for p in EXCLUDED_PREFIXES if p.endswith("."))

def _guess_seed_col(cols: List[str]) -> str:
    for c in SEED_CANDIDATES:
        if c in cols:
            return c
    # Why: Fall back to a sensible default to avoid breaking workflows.
    return cols[0]

def _looks_like_metric(col: str) -> bool:
    lower = col.lower()
    if not any(t in lower for t in SPLIT_TOKENS): 
        return False
    return any(m in lower for m in METRIC_TOKENS)

def _parse_metric_column(col: str) -> MetricSpec:
    # Why: Flexible parser supports dotted tokens like "ood.best_f1.f1" or "val.iid.f1".
    tokens = col.split(".")
    split = None
    family = None
    name = None

    # First token that matches a split becomes split.
    for t in tokens:
        if t in SPLIT_TOKENS:
            split = t
            break

    # Last token that matches a metric name becomes name.
    for t in reversed(tokens):
        if t in METRIC_TOKENS:
            name = t
            break

    # Middle piece becomes family if exists and isn't split or metric.
    if split and name:
        # pick the token between split and final name if present
        try:
            s_i = tokens.index(split)
            n_i = len(tokens) - 1 - tokens[::-1].index(name)
            mid = tokens[s_i+1:n_i]
            family = ".".join(mid) if mid else None
        except ValueError:
            family = None

    return MetricSpec(column=col, split=split, family=family, name=name)

def discover_schema(df: pd.DataFrame) -> Schema:
    cols = list(df.columns)

    seed_col = _guess_seed_col(cols)
    meta_cols = [c for c in cols if _is_meta(c)]
    metric_cols = [c for c in cols if _looks_like_metric(c)]
    excluded = set(meta_cols + metric_cols + [seed_col])
    hparam_cols = [c for c in cols if c not in excluded and not _is_excluded(c)]
    unknown_cols = [c for c in cols if c not in excluded and _is_excluded(c)]

    metric_specs = [_parse_metric_column(c) for c in metric_cols]
    return Schema(
        seed_col=seed_col,
        meta_cols=meta_cols,
        metric_cols=metric_cols,
        hparam_cols=hparam_cols,
        unknown_cols=unknown_cols,
        metric_specs=metric_specs,
    )
