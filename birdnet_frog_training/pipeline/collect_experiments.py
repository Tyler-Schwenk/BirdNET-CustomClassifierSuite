#!/usr/bin/env python3
"""
collect_experiments.py

Aggregate experiment_summary.json files into a master CSV.

Usage:
    python collect_experiments.py --exp-root experiments --out all_experiments.csv
"""

import os
import json
import argparse
import pandas as pd
from pathlib import Path

def flatten_json(nested, prefix=""):
    """Flatten nested dict into dot-notation keys."""
    flat = {}
    for k, v in nested.items():
        key = f"{prefix}{k}" if prefix == "" else f"{prefix}.{k}"
        if isinstance(v, dict):
            flat.update(flatten_json(v, key))
        else:
            flat[key] = v
    return flat

def collect_experiments(exp_root: str, out_csv: str):
    rows = []
    exp_root = Path(exp_root)

    for exp_dir in exp_root.glob("*"):
        summary_path = exp_dir / "evaluation" / "experiment_summary.json"
        if not summary_path.exists():
            continue
        try:
            with open(summary_path, "r") as f:
                summary = json.load(f)
            row = flatten_json(summary)
            row["__exp_path"] = str(exp_dir)
            rows.append(row)
        except Exception as e:
            print(f"Skipping {summary_path}: {e}")

    if not rows:
        print("No summaries found.")
        return

    df_new = pd.DataFrame(rows)

    # Deduplicate: experiment.name + git_commit + timestamp
    key_cols = [c for c in ["experiment.name", "metadata.git_commit", "metadata.timestamp"] if c in df_new.columns]
    if os.path.exists(out_csv):
        df_old = pd.read_csv(out_csv)
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
        df_combined = df_combined.drop_duplicates(subset=key_cols, keep="last")
    else:
        df_combined = df_new.drop_duplicates(subset=key_cols, keep="last")

    df_combined.to_csv(out_csv, index=False)
    print(f"Wrote {len(df_combined)} experiments to {out_csv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-root", type=str, default="experiments", help="Root folder containing experiment subfolders")
    ap.add_argument("--out", type=str, default="all_experiments.csv", help="Output master CSV")
    args = ap.parse_args()

    collect_experiments(args.exp_root, args.out)
