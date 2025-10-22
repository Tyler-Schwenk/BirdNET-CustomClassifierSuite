
from typing import List, Dict
import pandas as pd
from .schema import Schema
from .signature import pick_config_columns, build_config_signature

# Why: Produce both wide and tidy forms to support different downstream needs.
def aggregate_by_config(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    config_cols = pick_config_columns(df, schema)
    work = df.copy()
    work["_config_sig"] = work.apply(lambda r: build_config_signature(r, config_cols), axis=1)

    group = work.groupby("_config_sig")
    summary = group[schema.metric_cols].agg(["mean","std","count"])
    summary.columns = [".".join(col) for col in summary.columns]  # flatten
    summary = summary.reset_index()

    # carry representative hyperparams for readability
    reps = group[config_cols].first().reset_index()
    out = reps.merge(summary, on="_config_sig", how="left")
    return out

def wide_to_tidy(agg_df: pd.DataFrame) -> pd.DataFrame:
    id_vars = [c for c in agg_df.columns if c == "_config_sig" or ".mean" not in c and ".std" not in c and ".count" not in c]
    value_cols = [c for c in agg_df.columns if c not in id_vars]
    if not value_cols:
        return agg_df.copy()
    tidy = agg_df.melt(id_vars=id_vars, value_vars=value_cols, var_name="metric", value_name="value")
    return tidy
