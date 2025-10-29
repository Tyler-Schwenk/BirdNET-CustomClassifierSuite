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

    # Expect 4 experiment configs (2 seeds * 2 qualities * 1 balance)
    yamls_all = sorted(out_dir.glob("*.yaml"))
    base = out_dir / "base.yaml"
    assert base.exists()
    yamls = [p for p in yamls_all if p.name != "base.yaml"]
    assert len(yamls) == 4

    # Each yaml should load and have an experiment.name matching filename
    for y in yamls:
        data = yaml.safe_load(y.read_text())
        assert data["experiment"]["name"] == y.stem
