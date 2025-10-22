
import argparse
import pandas as pd
from .constants import DEFAULT_PRIORITY_METRICS, DEFAULT_TOP_K
from .schema import load_csv, discover_schema
from .aggregate import aggregate_by_config
from .rank import rank_by_metric

# Why: Single entry point for CLI-driven reviews.
def main():
    p = argparse.ArgumentParser(prog="sweep-review", description="BirdNET sweep evaluation toolkit")
    p.add_argument("--csv", required=True, help="Path to sweep CSV")
    p.add_argument("--metric", default=DEFAULT_PRIORITY_METRICS[0], help="Metric to rank by")
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="How many configs to show")
    p.add_argument("--out-csv", default="", help="Optional path to save ranked CSV")
    args = p.parse_args()

    df = load_csv(args.csv)
    schema = discover_schema(df)

    agg = aggregate_by_config(df, schema)
    ranked = rank_by_metric(agg, args.metric, args.top_k)

    if args.out_csv:
        ranked.to_csv(args.out_csv, index=False)
        print(f"Saved ranked results to {args.out_csv}")
        return

    cols = ["_config_sig", f"{args.metric}.mean", f"{args.metric}.std"]
    cols = [c for c in cols if c in ranked.columns] + [c for c in ranked.columns if c not in cols]
    print(ranked[cols].to_string(index=False))

if __name__ == "__main__":
    main()
