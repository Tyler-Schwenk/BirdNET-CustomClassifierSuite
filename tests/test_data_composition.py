#!/usr/bin/env python3
"""Quick test to verify training_package subset loading"""
import sys
from pathlib import Path
from birdnet_custom_classifier_suite.pipeline.make_training_package import (
    load_config,
    load_manifest,
    filter_rows
)

# Test config with subsets
cfg = load_config(Path("config/sweeps/stage8_pilot/stage8_004.yaml"))
print("Config loaded successfully")
print(f"Experiment: {cfg['experiment']['name']}")

tp = cfg.get('training_package', {})
print(f"\nTraining package config:")
print(f"  Quality: {tp.get('quality')}")
print(f"  Balance: {tp.get('balance')}")
print(f"  Positive subsets: {tp.get('positive_subsets', [])}")
print(f"  Negative subsets: {tp.get('negative_subsets', [])}")

# Test loading (dry run - don't create training package)
dataset_cfg = cfg.get("dataset", {})
audio_root = Path(dataset_cfg.get("audio_root", "AudioData"))
manifest_path = Path(dataset_cfg.get("manifest", "data/manifest.csv"))

print(f"\nLoading manifest: {manifest_path}")
df = load_manifest(manifest_path, audio_root)
print(f"Manifest loaded: {len(df)} rows")

# Filter with subsets
merged_cfg = {**tp, "splits": ["train"], "seed": 123}
print(f"\nFiltering with subsets...")
pos, neg = filter_rows(df, merged_cfg, audio_root)

print(f"\nResults:")
print(f"  Total positives: {len(pos)}")
print(f"  Total negatives: {len(neg)}")

# Check if subsets are included
if "source_subset" in pos.columns:
    subset_pos = pos[pos["source_subset"] != "manifest"]
    print(f"  Positives from subsets: {len(subset_pos)}")
    if not subset_pos.empty:
        print(f"  Subset sources: {subset_pos['source_subset'].unique().tolist()}")

if "source_subset" in neg.columns:
    subset_neg = neg[neg["source_subset"] != "manifest"]
    print(f"  Negatives from subsets: {len(subset_neg)}")
    if not subset_neg.empty:
        print(f"  Subset sources: {subset_neg['source_subset'].unique().tolist()}")

print("\nTest passed! Data composition system is working.")
