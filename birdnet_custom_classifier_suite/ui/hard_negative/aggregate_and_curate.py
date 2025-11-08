"""
Utilities to aggregate per-run low-quality RADR CSVs into a master file and optionally
select top-percentile files and place them into a curated hard-negatives folder.

This module is intended to be used from the Streamlit app (imported) or run
directly as a script via the package (`python -m birdnet_custom_classifier_suite.ui.hard_negative.aggregate_and_curate`).

It reuses the same curator helpers used by the UI for matching and linking so
behavior is consistent between CLI and app flows.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

from birdnet_custom_classifier_suite.ui.hard_negative import curator


def aggregate_and_write(paths, out: Path | None = None) -> Path:
    """Aggregate per-run CSVs found under `paths` and write a master CSV.

    Returns the path to the written master CSV.
    """
    master = curator.aggregate_results(paths)
    stamp = int(time.time())
    if out is None:
        out_root = Path('scripts') / 'low_quality_inference'
        out_root.mkdir(parents=True, exist_ok=True)
        out_csv = out_root / f'master_low_quality_radr_max_{stamp}.csv'
    else:
        out_csv = Path(out)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(out_csv, index=False)
    return out_csv


def curate_top_percentile(master_df, pct: float, out_root: Path | None = None, link_method: str = 'copy', dry_run: bool = False) -> Path:
    """Select the top `pct` percent from master_df and write them into out_root/<label>.

    Returns the destination folder path.
    """
    n = len(master_df)
    sel_n = max(1, int(n * (pct / 100.0)))
    selected = master_df.head(sel_n)
    stamp = int(time.time())
    label = f"top{int(pct)}pct_{stamp}"
    dest_root = Path(out_root or (Path('scripts') / 'curated'))
    if dry_run:
        print(f"Dry-run: would write {len(selected)} files to {dest_root / label} using {link_method}")
        return dest_root / label
    curator.write_manifests_and_links(selected, label, dest_root, method=link_method, dry_run=False)
    return dest_root / label


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate low-quality RADR CSVs and curate top-percentile files")
    parser.add_argument("--paths", nargs='+', help="Files or directories to search for per-run CSVs (e.g. experiments scripts/low_quality_inference)")
    parser.add_argument("--out", type=Path, default=None, help="Optional master CSV output path (defaults to scripts/low_quality_inference/master_<ts>.csv)")
    parser.add_argument("--top-pct", type=float, default=None, help="If provided, select this percentile (e.g. 10 for top 10%) and write curated files")
    parser.add_argument("--outdir", type=Path, default=None, help="Destination root for curated files when --top-pct is used (defaults to scripts/curated)")
    parser.add_argument("--link-method", choices=['hardlink', 'symlink', 'copy'], default='copy', help="How to place curated files")
    parser.add_argument("--dry-run", action='store_true')
    args = parser.parse_args(argv)

    try:
        master = curator.aggregate_results(args.paths)
    except Exception as e:
        print(f"Aggregation failed: {e}")
        return 2

    out_csv = aggregate_and_write(args.paths, out=args.out)
    print(f"Wrote master aggregated CSV: {out_csv} (rows: {len(master)})")

    if args.top_pct is not None:
        pct = float(args.top_pct)
        if pct <= 0 or pct > 100:
            print("--top-pct should be in (0, 100]")
            return 3
        dest = curate_top_percentile(master, pct, out_root=args.outdir, link_method=args.link_method, dry_run=args.dry_run)
        if not args.dry_run:
            print(f"Wrote curated selection to: {dest}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
