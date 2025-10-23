#!/usr/bin/env python3
"""
signature.py

Defines how to compute and identify experiment configuration signatures.

Each experiment run (with a unique seed) can be grouped under the same
configuration signature so we can aggregate across seeds.
"""

import hashlib
import pandas as pd
from typing import List


def pick_config_columns(df: pd.DataFrame) -> List[str]:
    """
    Identify which columns describe configuration parameters.
    These are used to compute unique signatures for grouping across seeds.
    """
    # Drop metrics and metadata, keep config-level keys
    # Always ignore any flattened 'metrics.' prefix and legacy iid/ood prefixes
    ignore_prefixes = ["metrics.", "iid.", "ood.", "metadata.", "experiment.", "__"]
    config_cols = [
        c for c in df.columns
        if not any(c.startswith(pref) for pref in ignore_prefixes)
    ]
    return config_cols


def build_config_signature(row: pd.Series, config_cols: List[str]) -> str:
    """
    Generate a short deterministic hash from configuration parameters.
    Ensures that runs with identical configs (but different seeds)
    share the same signature.
    """
    # Create a canonical string representation of config
    def _stringify(v):
        # Use pandas-friendly checks for NA and ensure stable repr for lists
        try:
            if pd.isna(v):
                return "<NA>"
        except Exception:
            pass
        if isinstance(v, (list, tuple)):
            return str(list(v))
        return str(v)

    items = [f"{col}={_stringify(row[col])}" for col in config_cols if col in row]
    joined = "|".join(sorted(items))
    # Hash to compact signature
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:8]


def add_signatures(df: pd.DataFrame) -> pd.DataFrame:
    """Add a '__signature' column if not already present."""
    if "__signature" in df.columns:
        return df

    config_cols = pick_config_columns(df)
    # If no config columns found, leave signatures blank but present
    if not config_cols:
        df["__signature"] = ""
        return df

    df["__signature"] = df.apply(
        lambda r: build_config_signature(r, config_cols), axis=1
    )
    return df


# Example CLI for testing
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="Path to all_experiments.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = add_signatures(df)
    print(df[["experiment.name", "__signature"]].head())
    print(f"\nGenerated {df['__signature'].nunique()} unique config signatures.")
