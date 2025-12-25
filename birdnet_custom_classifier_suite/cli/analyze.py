#!/usr/bin/env python3
"""
Analyze BirdNET Custom Classifier experiments.

Aggregates experiment results, summarizes across seeds, ranks configurations,
and generates Markdown + CSV leaderboards.

Usage:
    birdnet-analyze --stage stage4_ --precision-floor 0.9
    python -m birdnet_custom_classifier_suite.cli.analyze --stage stage4_
"""

from __future__ import annotations

import argparse
import pandas as pd

from birdnet_custom_classifier_suite.eval_toolkit import review, rank, report, signature
from birdnet_custom_classifier_suite.pipeline import collect_experiments


def main():
    """Main entry point for experiment analysis CLI."""
    ap = argparse.ArgumentParser(
        prog="birdnet-analyze",
        description="Analyze BirdNET Custom Classifier experiments."
    )
    ap.add_argument(
        "--exp-root",
        default="experiments",
        help="Path to experiment folders."
    )
    ap.add_argument(
        "--results",
        default="results/all_experiments.csv",
        help="Master results CSV path."
    )
    ap.add_argument(
        "--stage",
        help="Filter experiments starting with this prefix (e.g. 'stage4_'). Leave empty to include all experiments."
    )
    ap.add_argument(
        "--precision-floor",
        type=float,
        default=None,
        help="Precision floor for filtering."
    )
    ap.add_argument(
        "--metric-prefix",
        default="metrics.ood.best_f1",
        help="Metric group prefix."
    )
    ap.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top configs to report."
    )
    args = ap.parse_args()

    print("\n=== Collecting experiment summaries ===")
    collect_experiments.collect_experiments(args.exp_root, args.results)

    print("\n=== Loading aggregated results ===")
    # Ensure the results CSV exists and is non-empty
    try:
        df = review.load_experiments(args.results)
    except FileNotFoundError:
        print(f"No results found at {args.results}; aborting analysis.")
        return

    # Ensure config signatures exist so we can group across seeds
    df = signature.add_signatures(df)

    if args.stage:
        df = review.filter_by_stage(df, args.stage)
        print(f"Filtered to {len(df)} rows for prefix '{args.stage}'")

    print("\n=== Summarizing across seeds ===")
    summary = review.summarize_grouped(df, metric_prefix=args.metric_prefix)

    # Print a concise preview for the user
    try:
        print("\nSummary preview (first 8 rows):")
        cols = ["__signature"] + [c for c in summary.columns if c.endswith('.f1_mean')]
        if 'experiment.names' in summary.columns:
            cols.insert(1, 'experiment.names')
        print(summary[cols].head(8).to_string(index=False))
    except Exception:
        # best-effort preview; ignore errors to keep analysis moving
        pass

    print("\n=== Computing stability metrics ===")
    summary = rank.compute_stability(summary)

    print("\n=== Ranking configurations ===")
    top = rank.combined_rank(
        summary,
        metric=f"{args.metric_prefix}.f1_mean",  # Use the mean column for ranking
        precision_floor=args.precision_floor,
        stability_weight=0.2,
    ).head(args.top_n)

    leaderboard_name = f"{args.stage or 'all'}_top{args.top_n}"
    report_title = f"Leaderboard — {args.stage or 'All'} (Precision ≥ {args.precision_floor})"

    # Normalize precision_floor if user passed percentage (e.g. 90 means 0.9)
    if args.precision_floor is not None and args.precision_floor > 1:
        args.precision_floor = args.precision_floor / 100.0

    print("\n=== Generating reports ===")
    # Diagnostic: flag any suspicious perfect F1 means
    f1_col = f"{args.metric_prefix}.f1"
    f1_mean_col = f"{f1_col}_mean"
    if f1_mean_col in summary.columns:
        perfect = summary[summary[f1_mean_col] >= 0.999]
        if not perfect.empty:
            print("\nWARNING: some summarized F1 means are >= 0.999 — listing contributors:")
            for sig in perfect["__signature"]:
                rows = df[df["__signature"] == sig]
                print(f"Signature {sig}: {len(rows)} runs -> experiment names: {rows['experiment.name'].unique().tolist()}")
                if f1_col in rows.columns:
                    print("  raw f1 values:", rows[f1_col].tolist())
    # If no top rows (e.g., precision floor filtered everything), still write an empty report
    report.save_reports(
        top if top is not None else pd.DataFrame(),
        out_dir="results/leaderboards",
        name=leaderboard_name,
        title=report_title,
        metric_prefix=args.metric_prefix,
    )

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
