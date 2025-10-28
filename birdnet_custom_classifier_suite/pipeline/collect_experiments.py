#!/usr/bin/env python3
"""
collect_experiments.py

Aggregate experiment_summary.json files into a master CSV.

Usage:
    python -m birdnet_custom_classifier_suite.pipeline.collect_experiments \
        --exp-root experiments \
        --out results/all_experiments.csv
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
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            flat.update(flatten_json(v, key))
        else:
            flat[key] = v
    return flat


def collect_experiments(exp_root: str, out_csv: str = "results/all_experiments.csv"):
    exp_root = Path(exp_root)
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
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

    # Minimal schema columns (from all_experiments_minimal.csv)
    minimal_cols = [
        "experiment.name","experiment.seed","training.epochs","training.batch_size","training.threads","training.val_split","training.autotune",
        "dataset.manifest","dataset.filters.include_negatives","dataset.filters.balance","dataset.filters.max_per_class","dataset.filters.quality",
        "dataset.filters.call_type","dataset.filters.source_root","dataset.filters.splits","dataset.filters.manifest","dataset.filters.seed",
        "metrics.iid.best_f1.threshold","metrics.iid.best_f1.precision","metrics.iid.best_f1.recall","metrics.iid.best_f1.f1","metrics.iid.best_f1.accuracy",
        "metrics.iid.auroc","metrics.iid.auprc","metrics.ood.best_f1.threshold","metrics.ood.best_f1.precision","metrics.ood.best_f1.recall",
        "metrics.ood.best_f1.f1","metrics.ood.best_f1.accuracy","metrics.ood.auroc","metrics.ood.auprc","metadata.timestamp","metadata.git_commit",
        "training_args.fmin","training_args.fmax","analyzer_args.fmin","analyzer_args.fmax","analyzer_args.sensitivity","training_args.overlap",
        "analyzer_args.overlap","training_args.focal-loss","training_args.focal-loss-gamma","training_args.focal-loss-alpha","training_args.learning_rate",
        "training.dropout","training_args.hidden_units","training_args.mixup","training_args.dropout","training_args.label_smoothing",
        "training_args.upsampling_mode","training_args.upsampling_ratio","training_args.batch_size","training_args.focal_loss","training_args.epochs"
    ]

    # Only keep minimal columns
    df_new = df_new.loc[:, [c for c in minimal_cols if c in df_new.columns]]

    key_cols = [c for c in ["experiment.name", "metadata.git_commit", "metadata.timestamp"] if c in df_new.columns]

    if out_path.exists():
        df_old = pd.read_csv(out_path)
        df_old = df_old.loc[:, [c for c in minimal_cols if c in df_old.columns]]
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
        if key_cols:
            df_combined = df_combined.drop_duplicates(subset=key_cols, keep="last")
        else:
            df_combined = df_combined.drop_duplicates(keep="last")
    else:
        if key_cols:
            df_combined = df_new.drop_duplicates(subset=key_cols, keep="last")
        else:
            df_combined = df_new.drop_duplicates(keep="last")

    # Write only minimal columns, in order
    df_combined = df_combined.loc[:, [c for c in minimal_cols if c in df_combined.columns]]
    df_combined.to_csv(out_path, index=False)
    print(f"Wrote {len(df_combined)} experiments to {out_path.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-root", type=str, default="experiments",
                    help="Root folder containing experiment subfolders")
    ap.add_argument("--out", type=str, default="results/all_experiments.csv",
                    help="Output master CSV (auto-creates results/ if missing)")
    args = ap.parse_args()

    collect_experiments(args.exp_root, args.out)
