from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import shutil
import os
import logging
import pandas as pd


def make_links(src_files: List[Path], dest_dir: Path, method: str = "hardlink") -> int:
    dest_dir.mkdir(parents=True, exist_ok=True)
    created = 0
    for src in src_files:
        tgt = dest_dir / src.name
        if tgt.exists():
            continue
        try:
            if method == "hardlink":
                os.link(src, tgt)
            elif method == "symlink":
                tgt.symlink_to(src)
            else:
                shutil.copy2(src, tgt)
            created += 1
        except Exception:
            try:
                shutil.copy2(src, tgt)
                created += 1
            except Exception as e:
                logging.warning(f"Failed to link/copy {src} -> {tgt}: {e}")
    return created


def load_radr_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"RADR CSV not found: {path}")
    df = pd.read_csv(path)
    cols = {c: c.strip() for c in df.columns}
    df.rename(columns=cols, inplace=True)
    conf_col = next((c for c in df.columns if c.lower().endswith("confidence") or c.lower().startswith("radr") or "conf" in c.lower()), None)
    file_col = next((c for c in df.columns if c.lower() == "file" or c.lower().endswith("file")), None)
    if conf_col is None:
        raise ValueError("Could not find a confidence column in the RADR CSV")
    if file_col is None:
        file_col = df.columns[0]
    df = df[[file_col, conf_col]].copy()
    df.columns = ["File", "radr_max_confidence"]
    return df


def match_files(df: pd.DataFrame, input_dir: Path) -> pd.DataFrame:
    """Match rows from a RADR DataFrame to files under input_dir.

    Optimized behavior:
    - Only walks the input tree once using os.walk.
    - Builds a filename -> paths map only for basenames present in the RADR rows, reducing work
      when the input tree is large but the selection is small.
    - Falls back to checking input_dir / fname if no basename match is found.
    """
    # Normalize inputs
    input_dir = Path(input_dir)
    if df is None or df.empty:
        return pd.DataFrame([])

    # Build set of basenames we need to find (fast to compare)
    try:
        basenames = set(Path(str(x)).name for x in df["File"].tolist())
    except Exception:
        basenames = set()

    # Walk the tree once and only record files whose basename is in the requested set
    file_map = {}
    for root, dirs, files in os.walk(input_dir):
        for fname in files:
            if fname in basenames:
                p = Path(root) / fname
                file_map.setdefault(fname, []).append(p.resolve())

    matches = []
    for _, row in df.iterrows():
        fname = str(row["File"]).strip()
        base = Path(fname).name
        candidates = file_map.get(base, [])
        if len(candidates) == 1:
            matches.append({"File": str(candidates[0]), "radr_max_confidence": row["radr_max_confidence"]})
        elif len(candidates) > 1:
            exact = next((p for p in candidates if str(p).endswith(fname)), None)
            chosen = exact or candidates[0]
            matches.append({"File": str(chosen), "radr_max_confidence": row["radr_max_confidence"]})
        else:
            # fallback: maybe the CSV used a relative path; check input_dir / fname
            candidate = (input_dir / fname).resolve()
            if candidate.exists():
                matches.append({"File": str(candidate), "radr_max_confidence": row["radr_max_confidence"]})
            else:
                logging.debug(f"Could not find input file for row: {fname}")

    out = pd.DataFrame(matches)
    return out


def select_by_percentile(df: pd.DataFrame, small_p: float, med_p: float, large_p: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_sorted = df.sort_values("radr_max_confidence", ascending=False).reset_index(drop=True)
    n = len(df_sorted)
    small_n = max(1, int(n * (small_p / 100.0)))
    med_n = max(1, int(n * (med_p / 100.0)))
    large_n = max(1, int(n * (large_p / 100.0)))
    small_df = df_sorted.head(small_n)
    med_df = df_sorted.head(med_n)
    large_df = df_sorted.head(large_n)
    top50_n = max(1, int(n * 0.5))
    top50_df = df_sorted.head(top50_n)
    return small_df, med_df, large_df, top50_df


def select_by_threshold(df: pd.DataFrame, small_min: float, med_min: float, large_min: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    small_df = df[df["radr_max_confidence"] >= small_min].sort_values("radr_max_confidence", ascending=False)
    med_df = df[df["radr_max_confidence"] >= med_min].sort_values("radr_max_confidence", ascending=False)
    large_df = df[df["radr_max_confidence"] >= large_min].sort_values("radr_max_confidence", ascending=False)
    df_sorted = df.sort_values("radr_max_confidence", ascending=False).reset_index(drop=True)
    n = len(df_sorted)
    top50_n = max(1, int(n * 0.5))
    top50_df = df_sorted.head(top50_n)
    return small_df, med_df, large_df, top50_df


def write_manifests_and_links(df: pd.DataFrame, label: str, dest_root: Path, method: str, dry_run: bool):
    dest = dest_root / label
    if dry_run:
        print(f"[dry-run] {label}: {len(df)} files")
        return
    dest.mkdir(parents=True, exist_ok=True)
    paths = [Path(p) for p in df["File"].tolist()]
    created = make_links(paths, dest, method=method)
    print(f"Wrote {len(df)} files to {dest} ({created} linked/copied).")


def aggregate_results(paths_or_roots: list, pattern: str = "low_quality_radr_max_*.csv") -> pd.DataFrame:
    """Search a list of files or directories for per-run RADR CSVs and aggregate to a master table.

    - paths_or_roots: list of file paths or directory roots to search. Direct file paths are included as-is.
    - pattern: glob pattern to match per-run CSV files when searching directories.

    Returns a DataFrame with columns: File, radr_max_confidence, saved_at (if present), source, model.
    If multiple rows exist for the same File, keep the row with the highest radr_max_confidence.
    """
    csv_files = []
    for p in paths_or_roots:
        pth = Path(p)
        if pth.is_file():
            csv_files.append(pth)
        elif pth.is_dir():
            for f in pth.glob(pattern):
                if f.is_file():
                    csv_files.append(f)
        else:
            # try glob expansion
            for f in Path('.').glob(str(p)):
                if f.is_file():
                    csv_files.append(f)

    if not csv_files:
        raise FileNotFoundError(f"No result CSVs found in: {paths_or_roots}")

    frames = []
    for csvp in sorted(csv_files):
        try:
            df = pd.read_csv(csvp)
        except Exception:
            # ignore unreadable files
            continue
        # Normalize expected columns
        cols = {c: c.strip() for c in df.columns}
        df.rename(columns=cols, inplace=True)
        # If the CSV includes only File + radr_max_confidence, that's fine.
        if 'File' not in df.columns:
            # try common fallbacks
            file_col = next((c for c in df.columns if c.lower() == 'file' or c.lower().endswith('file')), None)
            if file_col:
                df.rename(columns={file_col: 'File'}, inplace=True)
        if 'radr_max_confidence' not in df.columns:
            conf_col = next((c for c in df.columns if 'conf' in c.lower() or c.lower().endswith('confidence')), None)
            if conf_col:
                df.rename(columns={conf_col: 'radr_max_confidence'}, inplace=True)

        # Add provenance columns if missing
        if 'saved_at' not in df.columns:
            df['saved_at'] = None
        if 'source' not in df.columns:
            # infer source from path: experiments/<exp>/... or scripts/...
            src = None
            parts = csvp.parts
            if 'experiments' in parts:
                try:
                    idx = parts.index('experiments')
                    src = parts[idx + 1]
                except Exception:
                    src = None
            df['source'] = src
        if 'model' not in df.columns:
            df['model'] = None

        # Keep only necessary columns
        keep = [c for c in ['File', 'radr_max_confidence', 'saved_at', 'source', 'model'] if c in df.columns]
        if not keep or 'File' not in keep or 'radr_max_confidence' not in keep:
            continue
        frames.append(df[keep].copy())

    if not frames:
        raise RuntimeError("No valid result CSVs could be parsed.")

    all_df = pd.concat(frames, ignore_index=True)

    # Ensure radr_max_confidence is numeric
    all_df['radr_max_confidence'] = pd.to_numeric(all_df['radr_max_confidence'], errors='coerce')
    all_df = all_df.dropna(subset=['File'])

    # Normalize File column to string
    all_df['File'] = all_df['File'].astype(str)

    # Group by File and keep row with max confidence (preserve saved_at/source/model from that row)
    idx = all_df.groupby('File')['radr_max_confidence'].idxmax()
    master = all_df.loc[idx].reset_index(drop=True)
    # Sort by descending confidence
    master = master.sort_values('radr_max_confidence', ascending=False).reset_index(drop=True)
    return master
