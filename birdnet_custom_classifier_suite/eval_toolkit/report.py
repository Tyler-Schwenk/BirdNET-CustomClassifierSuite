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

    The `metric_prefix` may be passed without the leading 'metrics.'; the helper
    will normalize to the canonical 'metrics.' prefixed column names.
    """
    if not metric_prefix.startswith("metrics."):
        metric_prefix = f"metrics.{metric_prefix}"

    cols = [
        "__signature",
        f"{metric_prefix}.f1_mean",
        f"{metric_prefix}.f1_std",
        f"{metric_prefix}.precision_mean",
        f"{metric_prefix}.recall_mean",
    ]
    available = [c for c in cols if c in df.columns]
    if not available:
        raise ValueError(f"No columns found for prefix: {metric_prefix}")

    tbl = df.copy()[available].copy()
    if f"{metric_prefix}.f1_mean" in tbl and f"{metric_prefix}.f1_std" in tbl:
        tbl["F1 (mean ± std)"] = [
            format_metric(m, s)
            for m, s in zip(tbl[f"{metric_prefix}.f1_mean"], tbl[f"{metric_prefix}.f1_std"])
        ]
        tbl.drop(columns=[f"{metric_prefix}.f1_mean", f"{metric_prefix}.f1_std"], inplace=True)

    tbl.rename(
        columns={
            "__signature": "Config",
            f"{metric_prefix}.precision_mean": "Precision",
            f"{metric_prefix}.recall_mean": "Recall",
        },
        inplace=True,
    )
    return tbl


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

    save_markdown(df, md_path, title=title, metric_prefix=metric_prefix)
    save_csv(df, csv_path)
