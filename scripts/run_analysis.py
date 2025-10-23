#!/usr/bin/env python3
"""
run_analysis.py

Aggregates experiment results, summarizes across seeds, ranks configurations,
and generates Markdown + CSV leaderboards.

Usage:
    python scripts/run_analysis.py --stage stage4_ --precision-floor 0.9
"""

import argparse
from birdnet_custom_classifier_suite.eval_toolkit import review, rank, report
from birdnet_custom_classifier_suite.pipeline import collect_experiments


def main():
    ap = argparse.ArgumentParser(description="Analyze BirdNET Custom Classifier experiments.")
    ap.add_argument("--exp-root", default="experiments", help="Path to experiment folders.")
    ap.add_argument("--results", default="results/all_experiments.csv", help="Master results CSV path.")
    ap.add_argument("--stage", default="", help="Filter experiments starting with this prefix (e.g. 'stage4_').")
    ap.add_argument("--precision-floor", type=float, default=0.9, help="Precision floor for filtering.")
    ap.add_argument("--metric-prefix", default="ood.best_f1", help="Metric group prefix.")
    ap.add_argument("--top-n", type=int, default=10, help="Number of top configs to report.")
    args = ap.parse_args()

    print("\n=== Collecting experiment summaries ===")
    collect_experiments.collect_experiments(args.exp_root, args.results)

    print("\n=== Loading aggregated results ===")
    df = review.load_experiments(args.results)

    if args.stage:
        df = review.filter_by_stage(df, args.stage)
        print(f"Filtered to {len(df)} rows for prefix '{args.stage}'")

    print("\n=== Summarizing across seeds ===")
    summary = review.summarize_grouped(df, metric_prefix=args.metric_prefix)

    print("\n=== Computing stability metrics ===")
    summary = rank.compute_stability(summary)

    print("\n=== Ranking configurations ===")
    top = rank.combined_rank(
        summary,
        metric=f"{args.metric_prefix}.f1",
        precision_floor=args.precision_floor,
        stability_weight=0.2,
    ).head(args.top_n)

    leaderboard_name = f"{args.stage or 'all'}_top{args.top_n}"
    report_title = f"Leaderboard — {args.stage or 'All'} (Precision ≥ {args.precision_floor})"

    print("\n=== Generating reports ===")
    report.save_reports(
        top,
        out_dir="results/leaderboards",
        name=leaderboard_name,
        title=report_title,
        metric_prefix=args.metric_prefix,
    )

    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
