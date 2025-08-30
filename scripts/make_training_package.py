# scripts/make_training_package.py
"""
Build a BirdNET-style training package from split folders + manifest.
- Uses config (YAML/JSON) for parameters.
- Supports relative paths: manifest entries resolved against audio_root.
- Deterministic, logged, and reproducible.
"""

from __future__ import annotations
import argparse, json, logging, random, shutil, sys
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import pandas as pd
import yaml
from datetime import datetime

CLASS_POS = "RADR"
CLASS_NEG = "Negative"


# ---------------- Config utils ----------------
def load_config(path: Path) -> dict:
    if path.suffix in [".yaml", ".yml"]:
        return yaml.safe_load(path.read_text())
    elif path.suffix == ".json":
        return json.loads(path.read_text())
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")


# ---------------- Logging ----------------
def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


# ---------------- Manifest ----------------
def load_manifest(path: Path, audio_root: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found at {path.resolve()}")

    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    logging.info(f"Loaded manifest: {path.resolve()}")
    logging.info(f"Columns ({len(df.columns)}): {df.columns.tolist()}")
    logging.info(f"First 5 rows:\n{df.head().to_string()}")

    required = [
        "label","quality","call_type","site","recorder_id",
        "date","new_full_path","split","filename"
    ]
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Manifest missing required column: {col}")

    # ensure string type
    for col in required:
        df[col] = df[col].astype(str)

    def resolve_path(p: str) -> str:
        p = str(p).strip()
        if not p:
            return ""
        p = Path(p)
        return str(p if p.is_absolute() else (audio_root / p).resolve())

    df["resolved_path"] = df["new_full_path"].apply(resolve_path)
    return df




# ---------------- Filtering ----------------
def isin_ci(series: pd.Series, values: List[str]) -> pd.Series:
    if not values:
        return pd.Series([True] * len(series), index=series.index)
    low = [v.lower() for v in values]
    return series.astype(str).str.lower().isin(low)


def filter_rows(df: pd.DataFrame, args: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logging.info(f"Filtering with args: {args}")
    logging.info(f"Available columns: {df.columns.tolist()}")

    # sanity check
    if "label" not in df.columns:
        raise KeyError("Manifest is missing 'label' column!")

    present = df[df["split"].str.lower().isin([s.lower() for s in args.get("splits", ["train"])])].copy()
    logging.info(f"Rows after split filter: {len(present)}")

    present["ok_path"] = present["resolved_path"].apply(lambda p: Path(p).exists())
    present = present[present["ok_path"]].copy()
    logging.info(f"Rows with valid paths: {len(present)}")

    # Filter conditions
    cond = pd.Series([True] * len(present), index=present.index)
    if args.get("site"):
        cond &= isin_ci(present["site"], args["site"])
    if args.get("recorder_id"):
        def z2(x: str) -> str:
            x = str(x).strip()
            return f"{int(x):02d}" if x.isdigit() else x
        allowed = set(z2(x) for x in args["recorder_id"])
        cond &= present["recorder_id"].map(z2).isin(allowed)

    filtered = present[cond].copy()

    # Split pos/neg
    pos = filtered[filtered["label"].str.lower() == "positive"].copy()
    neg = filtered[filtered["label"].str.lower() == "negative"].copy()

    if args.get("quality"):
        pos = pos[pos["quality"].str.lower().isin([q.lower() for q in args["quality"]])]
    if args.get("call_type"):
        pos = pos[pos["call_type"].str.lower().isin([c.lower() for c in args["call_type"]])]

    if not args.get("include_negatives", True):
        neg = neg.iloc[0:0]

    logging.info(f"Selected: {len(pos)} positives, {len(neg)} negatives")
    return pos, neg



# ---------------- Sampling ----------------
def deterministically_sample(df: pd.DataFrame, cap: Optional[int], seed: int, key_cols: List[str]) -> pd.DataFrame:
    if cap is None or len(df) <= cap:
        return df.copy()

    def row_key(row: pd.Series) -> str:
        return "|".join([str(row.get(k, "")) for k in key_cols])

    keys = df.apply(row_key, axis=1).tolist()
    idxs = df.index.tolist()
    pairs = list(zip(keys, idxs))
    rnd = random.Random(seed)
    rnd.shuffle(pairs)
    keep_idx = [idx for _, idx in pairs[:cap]]
    return df.loc[keep_idx].copy()


def build_package(pos: pd.DataFrame, neg: pd.DataFrame, args: dict) -> Dict[str, pd.DataFrame]:
    pos_cap = args.get("pos_cap") or args.get("max_per_class")
    neg_cap = args.get("neg_cap") or args.get("max_per_class")

    pos_samp = deterministically_sample(pos, pos_cap, args.get("seed", 123), ["resolved_path","date","recorder_id"])

    neg_target_cap: Optional[int] = neg_cap
    if args.get("neg_ratio") is not None and len(pos_samp) > 0:
        ratio_cap = int(args["neg_ratio"] * len(pos_samp))
        neg_target_cap = min(neg_cap if neg_cap is not None else ratio_cap, ratio_cap)

    neg_samp = deterministically_sample(neg, neg_target_cap, args.get("seed", 123), ["resolved_path","date","recorder_id"])

    if args.get("balance", False):
        n = min(len(pos_samp), len(neg_samp)) if args.get("include_negatives", True) else len(pos_samp)
        pos_samp = deterministically_sample(pos_samp, n, args.get("seed", 123), ["resolved_path","date","recorder_id"])
        if args.get("include_negatives", True):
            neg_samp = deterministically_sample(neg_samp, n, args.get("seed", 123), ["resolved_path","date","recorder_id"])

    return {"pos": pos_samp, "neg": neg_samp}


# ---------------- Copy + Reports ----------------
def copy_files(selection: Dict[str, pd.DataFrame], out_dir: Path, dry_run: bool) -> Dict[str, int]:
    counts = {"copied_pos": 0, "copied_neg": 0, "missing": 0, "skipped": 0}
    rad = out_dir / CLASS_POS
    negd = out_dir / CLASS_NEG
    if not dry_run:
        rad.mkdir(parents=True, exist_ok=True)
        negd.mkdir(parents=True, exist_ok=True)

    for _, row in selection["pos"].iterrows():
        src = Path(row["resolved_path"])
        if not src.exists():
            counts["missing"] += 1
            continue
        tgt = rad / src.name
        if tgt.exists():
            counts["skipped"] += 1
            continue
        if not dry_run:
            shutil.copy2(src, tgt)
        counts["copied_pos"] += 1

    for _, row in selection["neg"].iterrows():
        src = Path(row["resolved_path"])
        if not src.exists():
            counts["missing"] += 1
            continue
        tgt = negd / src.name
        if tgt.exists():
            counts["skipped"] += 1
            continue
        if not dry_run:
            shutil.copy2(src, tgt)
        counts["copied_neg"] += 1

    return counts


# ---------------- Detailed Experiment Logging ----------------
def write_detailed_counts(out_dir: Path,
                          pos: pd.DataFrame,
                          neg: pd.DataFrame,
                          selection: Dict[str, pd.DataFrame]) -> None:
    """
    Write detailed breakdown of counts for positives/negatives
    grouped by quality, call_type, site, recorder_id.
    Produces a CSV suitable for comparing experiments.
    """
    detailed = []

    def summarize(df: pd.DataFrame, label: str, stage: str):
        if df.empty:
            return
        for col in ["quality", "call_type", "site", "recorder_id", "split"]:
            if col not in df.columns:
                continue
            counts = df.groupby(col).size().reset_index(name="count")
            for _, row in counts.iterrows():
                detailed.append({
                    "stage": stage,           # available vs selected
                    "label": label,           # positive / negative
                    "group_by": col,
                    "group": str(row[col]),
                    "count": int(row["count"])
                })

    # before filtering (available pool)
    summarize(pos, "positive", "available")
    summarize(neg, "negative", "available")

    # after sampling / balancing (selected set)
    summarize(selection["pos"], "positive", "selected")
    summarize(selection["neg"], "negative", "selected")

    df_out = pd.DataFrame(detailed)
    df_out.to_csv(out_dir / "data_summary.csv", index=False)
    logging.info(f"Wrote detailed counts to {out_dir/'data_summary.csv'}")

def write_reports(out_dir: Path,
                  cfg: dict,
                  pos: pd.DataFrame,
                  neg: pd.DataFrame,
                  selection: Dict[str, pd.DataFrame],
                  copy_counts: Dict[str, int]) -> None:
    """
    Write summary reports:
      - selection_report.txt (human-readable JSON)
      - selection_report.json (structured JSON)
      - selection_manifest.csv (full manifest of selected files)
    """
    txt = out_dir / "selection_report.txt"
    jsonp = out_dir / "selection_report.json"
    csvp = out_dir / "selection_manifest.csv"

    info = {
        "filters": cfg,
        "counts_before": {"pos": int(len(pos)), "neg": int(len(neg))},
        "counts_selected": {"pos": int(len(selection["pos"])), "neg": int(len(selection["neg"]))},
        "copy_counts": copy_counts,
        "timestamp": datetime.utcnow().isoformat()
    }

    # human-readable and json dump
    txt.write_text(json.dumps(info, indent=2), encoding="utf-8")
    jsonp.write_text(json.dumps(info, indent=2), encoding="utf-8")

    # manifest of all rows included
    sel = pd.concat([selection["pos"], selection["neg"]], ignore_index=True)
    keep = ["resolved_path","label","quality","call_type","site","recorder_id","date","split","filename"]
    keep = [c for c in keep if c in sel.columns]
    sel[keep].to_csv(csvp, index=False)

    logging.info(f"📝 Wrote summary reports to {out_dir}")


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="Create BirdNET-style training package with filters and deterministic sampling.")
    ap.add_argument("--config", type=Path, required=True, help="Config file (YAML/JSON).")
    ap.add_argument("--section", type=str, default="training_package", help="Config section to use.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg_pkg = cfg.get(args.section, {})
    dataset_cfg = cfg.get("dataset", {})

    setup_logging(args.verbose)

    audio_root = Path(dataset_cfg.get("audio_root", "AudioData"))
    manifest = Path(dataset_cfg.get("manifest", "data/manifest.csv"))
    exp_dir = get_experiment_dir(cfg)
    out_dir = exp_dir / "training_package"


    df = load_manifest(manifest, audio_root)
    pos, neg = filter_rows(df, cfg_pkg)
    logging.info(f"Filtered: positives={len(pos)}, negatives={len(neg)}")

    selection = build_package(pos, neg, cfg_pkg)

    if out_dir.exists():
        raise FileExistsError(
            f"Training package directory already exists: {out_dir}\n"
            f"Pick a new `training_package.name` in your config to avoid overwriting."
        )

    out_dir.mkdir(parents=True, exist_ok=False)

    copy_counts = copy_files(selection, out_dir, dry_run=cfg_pkg.get("dry_run", False))

    # Write all reports
    write_reports(out_dir, cfg_pkg, pos, neg, selection, copy_counts)
    write_detailed_counts(out_dir, pos, neg, selection)

    # Save config snapshot alongside reports
    (out_dir / "config_used.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
    )

    logging.info(f"Package created at {out_dir}")


if __name__ == "__main__":
    main()

def get_experiment_dir(cfg: dict) -> Path:
    """Resolve the root directory for this experiment."""
    exp_cfg = cfg.get("experiment", {})
    name = exp_cfg.get("name", "unnamed_experiment")
    base = Path("experiments")
    return base / name


def run_from_config(cfg: dict, verbose: bool = False):
    tp_cfg = cfg.get("training_package", {})
    dataset_cfg = cfg.get("dataset", {})
    exp_cfg = cfg.get("experiment", {})

    setup_logging(verbose)

    merged_cfg = {
        **tp_cfg,
        "source_root": str(Path(dataset_cfg.get("copy_to", "data/splits"))),
        "splits": tp_cfg.get("splits", ["train"] if tp_cfg else ["train"]),
        "manifest": str(Path(dataset_cfg.get("manifest", "data/manifest.csv"))),
        "seed": exp_cfg.get("seed", 123),  
    }

    audio_root = Path(dataset_cfg.get("audio_root", "AudioData"))
    df = load_manifest(Path(merged_cfg["manifest"]), audio_root)

    pos, neg = filter_rows(df, merged_cfg)
    selection = build_package(pos, neg, merged_cfg)

    exp_dir = get_experiment_dir(cfg)
    out_dir = exp_dir / "training_package"

    if out_dir.exists():
        raise FileExistsError(
            f"Training package directory already exists: {out_dir}\n"
            f"Pick a new `experiment.name` in your config to avoid overwriting."
        )
    out_dir.mkdir(parents=True, exist_ok=False)

    copy_counts = copy_files(selection, out_dir, dry_run=tp_cfg.get("dry_run", False))

    # Write all reports
    write_reports(out_dir, merged_cfg, pos, neg, selection, copy_counts)
    write_detailed_counts(out_dir, pos, neg, selection)

    # Save config snapshot
    (exp_dir / "config_used.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
    )

    print(f"Training package created at {out_dir}")

