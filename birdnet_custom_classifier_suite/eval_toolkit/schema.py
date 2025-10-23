#!/usr/bin/env python3
"""
schema.py

Defines schema discovery and utility functions for interpreting
BirdNET-CustomClassifierSuite experiment summaries.

This helps downstream modules (review, rank, report) automatically
understand which columns are metrics vs metadata.
"""

import pandas as pd
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class MetricSpec:
    """Describes one metric column group (like ood.best_f1)."""
    prefix: str
    fields: List[str]


def load_csv(path: str) -> pd.DataFrame:
    """Load CSV with standardized dtype handling."""
    df = pd.read_csv(path)
    # Drop empty columns silently
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    return df


def discover_schema(df: pd.DataFrame) -> Dict[str, MetricSpec]:
    """
    Automatically discover metric groups from flattened JSON-style columns.

    Example:
        Columns like:
          - ood.best_f1.f1
          - ood.best_f1.precision
          - ood.best_f1.recall
        become one MetricSpec(prefix='ood.best_f1', fields=['f1', 'precision', 'recall'])
    """
    metrics: Dict[str, MetricSpec] = {}

    for col in df.columns:
        parts = col.split(".")
        # Support both flattened metric names that start with
        # - "metrics.iid..." / "metrics.ood..." (preferred)
        # - or the older style "iid..." / "ood..."
        if not parts:
            continue

        # case: metrics.iid.best_f1.f1 -> want prefix 'metrics.iid.best_f1'
        if parts[0] == "metrics" and len(parts) >= 4 and parts[1] in ("iid", "ood"):
            prefix = ".".join(parts[:-1])
            field = parts[-1]
        # case: iid.best_f1.f1 or iid.auroc -> normalize to metrics.iid.* for compatibility
        elif len(parts) >= 2 and parts[0] in ("iid", "ood"):
            prefix = ".".join(parts[:-1])
            field = parts[-1]
            # normalize to canonical 'metrics.' prefix
            if not prefix.startswith("metrics."):
                prefix = f"metrics.{prefix}"
        else:
            continue

        # ensure canonical prefix starts with 'metrics.' for consistency
        if not prefix.startswith("metrics."):
            prefix = f"metrics.{prefix}"

        if prefix not in metrics:
            metrics[prefix] = MetricSpec(prefix=prefix, fields=[])
        if field not in metrics[prefix].fields:
            metrics[prefix].fields.append(field)

    return metrics


def list_metrics(schema: Dict[str, MetricSpec]):
    """Utility to pretty-print discovered metric schema."""
    print("Discovered metric groups:\n")
    for prefix, spec in schema.items():
        fields = ", ".join(spec.fields)
        print(f"  {prefix}: [{fields}]")


# Example CLI for quick inspection
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=str, help="Path to all_experiments.csv")
    args = ap.parse_args()

    df = load_csv(args.csv)
    schema = discover_schema(df)
    list_metrics(schema)
