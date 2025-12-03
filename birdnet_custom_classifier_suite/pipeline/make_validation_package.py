# birdnet_custom_classifier_suite/pipeline/make_validation_package.py
"""
Build a BirdNET-style validation package from val split + manifest.
Similar to make_training_package.py but:
  1. Filters for split='val' instead of split='train'
  2. Exports to {exp_dir}/validation_package/ instead of training_package/
  3. Used with BirdNET's --test_data flag during training
"""

from __future__ import annotations
import logging, random, shutil
from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd
import yaml

from birdnet_custom_classifier_suite.pipeline.make_training_package import (
    load_manifest,
    isin_ci,
    load_subset_files,
    deterministically_sample,
    CLASS_POS,
    CLASS_NEG,
    setup_logging,
)


def filter_validation_rows(df: pd.DataFrame, args: dict, audio_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter manifest rows for validation split and merge with custom subset files.
    
    This is similar to filter_rows() but specifically targets the 'val' split.
    
    Args:
        df: Manifest DataFrame
        args: Config dict with filters (quality, site, recorder_id, etc.)
        audio_root: Base path for resolving subset folders
    
    Returns:
        Tuple of (positive_df, negative_df) including both manifest and subset files
    """
    logging.info(f"Filtering validation data with args: {args}")
    
    # Filter for val split only
    val_split = df[df["split"].str.lower() == "val"].copy()
    logging.info(f"Rows in val split: {len(val_split)}")
    
    # Check path existence
    val_split["ok_path"] = val_split["resolved_path"].apply(lambda p: Path(p).exists())
    val_split = val_split[val_split["ok_path"]].copy()
    logging.info(f"Val rows with valid paths: {len(val_split)}")
    
    # Apply filters (site, recorder_id, etc.)
    cond = pd.Series([True] * len(val_split), index=val_split.index)
    if args.get("site"):
        cond &= isin_ci(val_split["site"], args["site"])
    if args.get("recorder_id"):
        def z2(x: str) -> str:
            x = str(x).strip()
            return f"{int(x):02d}" if x.isdigit() else x
        allowed = set(z2(x) for x in args["recorder_id"])
        cond &= val_split["recorder_id"].map(z2).isin(allowed)
    
    filtered = val_split[cond].copy()
    
    # Split pos/neg from manifest
    pos = filtered[filtered["label"].str.lower() == "positive"].copy()
    neg = filtered[filtered["label"].str.lower() == "negative"].copy()
    
    # Apply quality filter to manifest positives
    if args.get("quality"):
        pos = pos[pos["quality"].str.lower().isin([q.lower() for q in args["quality"]])]
    if args.get("call_type"):
        pos = pos[pos["call_type"].str.lower().isin([c.lower() for c in args["call_type"]])]
    
    if not args.get("include_negatives", True):
        neg = neg.iloc[0:0]
    
    logging.info(f"Manifest-based val selection: {len(pos)} positives, {len(neg)} negatives")
    
    # NOTE: Validation typically uses only manifest data, not curated subsets
    # (subsets are for training augmentation). But we support it for flexibility.
    pos_subset_paths = args.get("positive_subsets", [])
    neg_subset_paths = args.get("negative_subsets", [])
    
    if pos_subset_paths:
        logging.warning("positive_subsets specified for validation - usually validation uses only manifest data")
        pos_subsets = load_subset_files(pos_subset_paths, audio_root)
        if not pos_subsets.empty:
            pos_subsets = pos_subsets[pos_subsets["label"].str.lower() == "positive"]
            if not pos_subsets.empty:
                if "source_subset" not in pos.columns:
                    pos["source_subset"] = "manifest"
                pos = pd.concat([pos, pos_subsets], ignore_index=True)
                logging.info(f"Added {len(pos_subsets)} files from positive_subsets to validation")
    
    if neg_subset_paths:
        logging.warning("negative_subsets specified for validation - usually validation uses only manifest data")
        neg_subsets = load_subset_files(neg_subset_paths, audio_root)
        if not neg_subsets.empty:
            neg_subsets = neg_subsets[neg_subsets["label"].str.lower() == "negative"]
            if not neg_subsets.empty:
                if "source_subset" not in neg.columns:
                    neg["source_subset"] = "manifest"
                neg = pd.concat([neg, neg_subsets], ignore_index=True)
                logging.info(f"Added {len(neg_subsets)} files from negative_subsets to validation")
    
    logging.info(f"Final validation selection: {len(pos)} positives, {len(neg)} negatives")
    return pos, neg


def build_validation_package(pos: pd.DataFrame, neg: pd.DataFrame, args: dict) -> Dict[str, pd.DataFrame]:
    """
    Sample validation data (usually no sampling needed - use all val data).
    
    Unlike training, validation typically uses the full val split without sampling.
    But we support optional caps for consistency with training package builder.
    """
    pos_cap = args.get("val_pos_cap") or args.get("pos_cap")
    neg_cap = args.get("val_neg_cap") or args.get("neg_cap")
    
    pos_samp = deterministically_sample(pos, pos_cap, args.get("seed", 123), ["resolved_path", "date", "recorder_id"])
    
    neg_target_cap: Optional[int] = neg_cap
    if args.get("neg_ratio") is not None and len(pos_samp) > 0:
        ratio_cap = int(args["neg_ratio"] * len(pos_samp))
        neg_target_cap = min(neg_cap if neg_cap is not None else ratio_cap, ratio_cap)
    
    neg_samp = deterministically_sample(neg, neg_target_cap, args.get("seed", 123), ["resolved_path", "date", "recorder_id"])
    
    # NOTE: Validation typically does NOT use class balancing
    # (we want to evaluate on the natural class distribution in val split)
    if args.get("balance", False):
        logging.warning("balance=True for validation - validation usually uses natural class distribution")
        n = min(len(pos_samp), len(neg_samp)) if args.get("include_negatives", True) else len(pos_samp)
        pos_samp = deterministically_sample(pos_samp, n, args.get("seed", 123), ["resolved_path", "date", "recorder_id"])
        if args.get("include_negatives", True):
            neg_samp = deterministically_sample(neg_samp, n, args.get("seed", 123), ["resolved_path", "date", "recorder_id"])
    
    return {"pos": pos_samp, "neg": neg_samp}


def copy_validation_files(selection: Dict[str, pd.DataFrame], out_dir: Path, dry_run: bool) -> Dict[str, int]:
    """Copy selected validation files to RADR/Negative folders."""
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


def write_validation_manifest(out_dir: Path, selection: Dict[str, pd.DataFrame]) -> None:
    """Write manifest CSV of all validation files included."""
    sel = pd.concat([selection["pos"], selection["neg"]], ignore_index=True)
    keep = ["resolved_path", "label", "quality", "call_type", "site", "recorder_id", "date", "split", "filename"]
    keep = [c for c in keep if c in sel.columns]
    csvp = out_dir / "validation_manifest.csv"
    sel[keep].to_csv(csvp, index=False)
    logging.info(f"Wrote validation manifest to {csvp}")


def get_experiment_dir(cfg: dict) -> Path:
    """Resolve the root directory for this experiment."""
    exp_cfg = cfg.get("experiment", {})
    name = exp_cfg.get("name", "unnamed_experiment")
    base = Path("experiments")
    return base / name


def run_from_config(cfg: dict, verbose: bool = False):
    """
    Build validation package from config.
    
    This is called by pipeline.py when use_validation=True.
    """
    vp_cfg = cfg.get("validation_package", {})
    dataset_cfg = cfg.get("dataset", {})
    exp_cfg = cfg.get("experiment", {})
    
    setup_logging(verbose)
    
    # Merge config sections
    merged_cfg = {
        **vp_cfg,
        "splits": ["val"],  # Always use val split for validation
        "manifest": str(Path(dataset_cfg.get("manifest", "data/manifest.csv"))),
        "seed": exp_cfg.get("seed", 123),
    }
    
    audio_root = Path(dataset_cfg.get("audio_root", "AudioData"))
    df = load_manifest(Path(merged_cfg["manifest"]), audio_root)
    
    pos, neg = filter_validation_rows(df, merged_cfg, audio_root)
    
    if pos.empty and neg.empty:
        logging.warning("No validation data found in val split - validation package will be empty")
    
    selection = build_validation_package(pos, neg, merged_cfg)
    
    exp_dir = get_experiment_dir(cfg)
    out_dir = exp_dir / "validation_package"
    
    if out_dir.exists():
        logging.info(f"Validation package directory already exists at {out_dir}, cleaning up")
        shutil.rmtree(out_dir)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    copy_counts = copy_validation_files(selection, out_dir, dry_run=vp_cfg.get("dry_run", False))
    
    # Write manifest
    write_validation_manifest(out_dir, selection)
    
    # Save config snapshot
    (out_dir / "config_used.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
    )
    
    logging.info(f"Validation package created at {out_dir}")
    logging.info(f"Validation counts: {copy_counts}")
    print(f"Validation package: {copy_counts['copied_pos']} positives, {copy_counts['copied_neg']} negatives")
