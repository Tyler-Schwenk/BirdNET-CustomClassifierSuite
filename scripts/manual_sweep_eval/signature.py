
from typing import List, Tuple
import hashlib
import pandas as pd
from .schema import Schema

# Why: Deterministic signature lets us aggregate seeds per unique config.
def pick_config_columns(df: pd.DataFrame, schema: Schema) -> List[str]:
    keep = []
    for c in df.columns:
        if c in schema.meta_cols: 
            continue
        if c == schema.seed_col: 
            continue
        if c in schema.metric_cols: 
            continue
        keep.append(c)
    return keep

def build_config_signature(row: pd.Series, config_cols: List[str]) -> str:
    pairs: List[Tuple[str, object]] = [(c, row[c]) for c in config_cols]
    data = repr(tuple(pairs)).encode("utf-8")
    return hashlib.sha1(data).hexdigest()[:12]
