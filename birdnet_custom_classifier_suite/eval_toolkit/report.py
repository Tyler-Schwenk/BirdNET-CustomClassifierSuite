#!/usr/bin/env python3
"""
report.py

Generates Markdown and CSV summaries from ranked experiment DataFrames.

Supports:
  - Markdown leaderboards (for GitHub/Confluence)
  - CSV exports (for spreadsheets)
  - Optional F1 ± std compact formatting
"""

from pathlib import Path
import pandas as pd


# ------------------------- Formatting Helpers ------------------------- #

def format_metric(mean, std, precision=3):
    """Return formatted string like '0.742 ± 0.013'."""
    if pd.isna(mean):
        return "-"
    if pd.isna(std) or std == 0:
        return f"{mean:.{precision}f}"
    return f"{mean:.{precision}f} ± {std:.{precision}f}"


def format_leaderboard(df: pd.DataFrame, metric_prefix="ood.best_f1"):
    """Return a compact leaderboard table (DataFrame) for Markdown export.
    Shows F1/precision/recall means and standard deviations, sorted by F1.
    """
    if not metric_prefix.startswith("metrics."):
        metric_prefix = f"metrics.{metric_prefix}"

    # Start with essential columns
    tbl = df.copy().sort_values(f"{metric_prefix}.f1_mean", ascending=False).reset_index(drop=True)
    out = pd.DataFrame()

    # Config info
    out["Config"] = tbl["__signature"]
    out["experiment.names"] = tbl["experiment.names"]

    # Core metrics with mean ± std
    metrics = ["precision", "recall", "f1"]
    for m in metrics:
        mean_col = f"{metric_prefix}.{m}_mean"
        std_col = f"{metric_prefix}.{m}_std"
        
        means = tbl[mean_col]
        stds = tbl[std_col] if std_col in tbl.columns else None

        out[m.capitalize()] = [format_metric(mean, std) for mean, std in zip(means, stds)]
        
        # Store numeric versions for sorting
        out[f"{m}_mean"] = means

    # Sort by F1 mean descending
    out = out.sort_values("f1_mean", ascending=False).reset_index(drop=True)
    out.drop(columns=[f"{m}_mean" for m in metrics], inplace=True)

    return out


# ------------------------- Markdown Export ------------------------- #

def save_markdown(df: pd.DataFrame, out_path: str, title: str = None, metric_prefix="ood.best_f1"):
    """Save DataFrame as a Markdown leaderboard."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    tbl = format_leaderboard(df, metric_prefix)
    md = []

    if title:
        md.append(f"# {title}\n")

    # Add rank column
    tbl.insert(0, "Rank", range(1, len(tbl) + 1))

    md.append(tbl.to_markdown(index=False, tablefmt="github"))
    out.write_text("\n".join(md), encoding="utf-8")

    print(f"✅ Markdown report saved to {out.resolve()}")


# ------------------------- CSV Export ------------------------- #

def save_csv(df: pd.DataFrame, out_path: str):
    """Save ranked/summarized DataFrame to CSV."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"✅ CSV report saved to {out.resolve()}")


# ------------------------- Combined Convenience ------------------------- #

def save_reports(df: pd.DataFrame, out_dir: str, name: str, title: str = None, metric_prefix="ood.best_f1"):
    """Generate both Markdown and CSV reports."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{name}.md"
    csv_path = out_dir / f"{name}.csv"
    raw_csv_path = out_dir / f"{name}.raw.csv"

    # Create a formatted leaderboard (strings rounded/mean ± std) for human-friendly outputs
    formatted = format_leaderboard(df, metric_prefix=metric_prefix)

    # Save markdown from the numeric df (save_markdown will call format_leaderboard internally)
    save_markdown(df, md_path, title=title, metric_prefix=metric_prefix)
    # Save the formatted (string) CSV for human consumption and the raw numeric CSV for programmatic use
    save_csv(formatted, csv_path)
    save_csv(df, raw_csv_path)
