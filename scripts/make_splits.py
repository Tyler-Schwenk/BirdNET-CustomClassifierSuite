# scripts/make_splits.py
"""
Create Train/Val/Test (IID+OOD) splits from a manifest file.
- Reads parameters from a config file (YAML/JSON) OR CLI args.
- Always writes out a copy of the config used (for reproducibility).
- Deterministic: stable hashing ensures the same split every run.
"""

from __future__ import annotations
import argparse
import hashlib
import math
import json
import yaml
from pathlib import Path
import pandas as pd
import shutil
from datetime import datetime


def stable_hash(s: str, mod: int = 1_000_000) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16) % mod


def assign_by_date_blocks(df: pd.DataFrame, frac: float, seed_tag: str) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    for (rid, site), group in df.groupby(["recorder_id", "site"], sort=False):
        dates = sorted(set(group["date_filled"]))
        dates = [d for d in dates if d != "unknown"]
        if not dates:
            continue
        target = max(1, math.floor(len(dates) * frac))
        ranked = sorted(dates, key=lambda d: stable_hash(f"{seed_tag}|{rid}|{site}|{d}"))
        chosen = set(ranked[:target])
        mask.loc[group.index] = group["date_filled"].isin(chosen).values
    return mask


def norm_site(s: str) -> str:
    return (s or "").strip().lower()


def load_config(path: Path) -> dict:
    if path.suffix in [".yaml", ".yml"]:
        return yaml.safe_load(path.read_text())
    elif path.suffix == ".json":
        return json.loads(path.read_text())
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")


def main():
    ap = argparse.ArgumentParser(description="Create Train/Val/Test (IID+OOD) splits with Sylvan Pond (Moth11+12) as OOD.")
    ap.add_argument("--config", type=Path, help="YAML/JSON config file with split parameters.")
    ap.add_argument("--manifest", type=Path, help="Path to manifest.csv (overrides config).")
    ap.add_argument("--out-manifest", type=Path, help="Output manifest with splits (overrides config).")
    args = ap.parse_args()

    # ---- Load config ----
    if args.config:
        cfg = load_config(args.config)
        cfg_splits = cfg.get("dataset", {})
    else:
        cfg_splits = {}

    # CLI overrides config
    manifest_path = args.manifest or Path(cfg_splits.get("manifest", "manifest.csv"))
    out_manifest = args.out_manifest or Path(cfg_splits.get("out_manifest", "manifest_with_split.csv"))
    iid_frac = float(cfg_splits.get("iid_frac", 0.12))
    val_frac = float(cfg_splits.get("val_frac", 0.10))
    ood_recorders = [str(x) for x in cfg_splits.get("ood_recorders", ["11","12"])]
    ood_sites = [str(x) for x in cfg_splits.get("ood_sites", ["srper sylvan pond"])]
    assign_unknown_dates = cfg_splits.get("assign_unknown_dates", "train")
    stats_csv = cfg_splits.get("stats_csv", None)
    copy_to = cfg_splits.get("copy_to", None)

    df = pd.read_csv(manifest_path)

    # normalize
    for col in ["recorder_id","site","label","quality","call_type","date","new_full_path"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(str)

    def z2(x):
        x = (x or "").strip()
        return f"{int(x):02d}" if x.isdigit() else x
    df["recorder_id"] = df["recorder_id"].map(z2)
    df["site_norm"] = df["site"].map(norm_site)

    # OOD mask
    ood_rec_set = set([z2(x) for x in ood_recorders])
    ood_site_set = set([norm_site(s) for s in ood_sites])
    is_ood = df["recorder_id"].isin(ood_rec_set) | df["site_norm"].isin(ood_site_set)

    # date handling
    df["date"] = df["date"].replace({"": pd.NA})
    df["date_filled"] = df["date"].fillna("unknown")

    df["split"] = ""

    # 1) OOD
    df.loc[is_ood, "split"] = "test_ood"

    # unknown-date placement
    unknown_mask = (df["date_filled"] == "unknown") & (df["split"] == "")
    if assign_unknown_dates == "iid":
        df.loc[unknown_mask, "split"] = "test_iid"
    elif assign_unknown_dates == "val":
        df.loc[unknown_mask, "split"] = "val"

    # 2) IID
    rest_mask = df["split"] == ""
    mask_iid = assign_by_date_blocks(df[rest_mask], frac=iid_frac, seed_tag="IID")
    df.loc[mask_iid.index[mask_iid], "split"] = "test_iid"

    # 3) Val
    rest_mask2 = df["split"] == ""
    mask_val = assign_by_date_blocks(df[rest_mask2], frac=val_frac, seed_tag="VAL")
    df.loc[mask_val.index[mask_val], "split"] = "val"

    # 4) Train
    df.loc[df["split"]=="", "split"] = "train"

    # Save manifest
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["site_norm"], errors="ignore").to_csv(out_manifest, index=False)

    # Save split stats
    if stats_csv:
        tab = (df.groupby(["split","label"])
                 .size().rename("count")
                 .reset_index()
                 .pivot(index="split", columns="label", values="count")
                 .fillna(0).astype(int))
        Path(stats_csv).parent.mkdir(parents=True, exist_ok=True)
        tab.reset_index().to_csv(stats_csv, index=False)

    # Save config snapshot for reproducibility
    cfg_used = {
        "manifest": str(manifest_path),
        "out_manifest": str(out_manifest),
        "iid_frac": iid_frac,
        "val_frac": val_frac,
        "ood_recorders": ood_recorders,
        "ood_sites": ood_sites,
        "assign_unknown_dates": assign_unknown_dates,
        "stats_csv": stats_csv,
        "copy_to": copy_to,
        "timestamp": datetime.utcnow().isoformat()
    }
    snap_path = out_manifest.parent / "splits_config.json"
    snap_path.write_text(json.dumps(cfg_used, indent=2))
    print(f"✅ Splits written to {out_manifest}")
    print(f"📑 Config snapshot saved to {snap_path}")


if __name__ == "__main__":
    main()
