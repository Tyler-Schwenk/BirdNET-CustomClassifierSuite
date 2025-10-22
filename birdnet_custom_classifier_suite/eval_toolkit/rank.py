
from typing import List, Tuple
import pandas as pd

# Why: Keep ranking small and composable so it can be swapped easily.
def _safe_sort(df: pd.DataFrame, key: str, ascending: bool) -> pd.DataFrame:
    if key not in df.columns:
        return df.copy()
    return df.sort_values(key, ascending=ascending, kind="mergesort")

def _col_for(metric: str, stat: str) -> str:
    # metric must match original column name; agg appended ".mean" and ".std"
    return f"{metric}.{stat}"

def rank_by_metric(agg_df: pd.DataFrame, metric: str, top_k: int) -> pd.DataFrame:
    key = _col_for(metric, "mean")
    ranked = _safe_sort(agg_df, key, ascending=False).head(top_k).copy()
    # Keep std next to mean if available
    std_col = _col_for(metric, "std")
    if std_col in ranked.columns:
        ranked = ranked[["_config_sig", key, std_col] + [c for c in ranked.columns if c not in {"_config_sig", key, std_col}]]
    return ranked

def extract_best_run_details(df: pd.DataFrame, config_sig: str, metric: str, precision_metric: str, recall_metric: str) -> pd.Series:
    cand = df[df["_config_sig"] == config_sig].copy()
    if cand.empty:
        return pd.Series()
    # pick the run with max metric value
    if metric not in cand.columns:
        # Nothing to pick from; return empty for clarity.
        return pd.Series()
    idx = cand[metric].idxmax()
    row = cand.loc[idx]
    out = pd.Series({
        "best_run_metric": row.get(metric, float("nan")),
        "best_run_precision": row.get(precision_metric, float("nan")),
        "best_run_recall": row.get(recall_metric, float("nan")),
    })
    return out
