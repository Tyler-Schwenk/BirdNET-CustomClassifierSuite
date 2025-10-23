import csv
from pathlib import Path

import yaml


def test_generate_sweep_writes_files(tmp_path):
    from birdnet_custom_classifier_suite.sweeps.sweep_generator import generate_sweep

    out_dir = tmp_path / "out"
    axes = {
        "seed": [111, 222],
        "quality": [["high"], ["low"]],
        "balance": [True],
        "upsampling_mode": ["none"],
        "upsampling_ratio": [0.0],
    }
    base_params = {"batch_size": 16, "upsampling_ratio": 0.0}

    generate_sweep(stage=0, out_dir=str(out_dir), axes=axes, base_params=base_params, prefix="test")

    # Expect 4 configs (2 seeds * 2 qualities * 1 balance)
    yamls = sorted(out_dir.glob("*.yaml"))
    assert len(yamls) == 4

    # Manifest exists and matches
    manifest = out_dir / "manifest.csv"
    assert manifest.exists()
    with open(manifest, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 4

    # Each yaml should load and have an experiment.name matching filename
    for y in yamls:
        data = yaml.safe_load(y.read_text())
        assert data["experiment"]["name"] == y.stem
