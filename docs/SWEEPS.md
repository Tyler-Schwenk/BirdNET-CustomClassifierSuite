# Sweep Specs, Base YAML, and Experiment Configs

This document describes how sweep specification files map to the per-sweep `base.yaml` and the individual experiment override configs.

## Overview

Each sweep is fully defined by a single spec file (YAML) that lives under `config/sweep_specs/`. The spec contains:

- `stage`: Integer used in experiment names (e.g., `stage4_001`)
- `out_dir`: Output folder for this sweep (contains `base.yaml` and experiment configs)
- `base_params`: The base training configuration for this sweep (becomes `out_dir/base.yaml`)
- `axes`: The values to sweep; each axis combination produces one experiment config

When you click "Generate configs" in the UI (or run the generator directly):

1. A per-sweep `base.yaml` is created in `out_dir` using `base_params`
2. One experiment config is written for each axis combination, produced by overlaying the axis values on top of the base

There is no dependency on a global `config/base.yaml` at runtime. The sweep’s own `base.yaml` is the canonical base used by the pipeline.

## Field mapping

### base_params → base.yaml

`base_params` keys are mapped as follows:

- Training (top-level `training` section)
  - `epochs` → `training.epochs`
  - `batch_size` → `training.batch_size` and `inference.batch_size`
- Audio parameters (mirrored)
  - `fmin`, `fmax`, `overlap` → `training_args.{fmin,fmax,overlap}` and `analyzer_args.{fmin,fmax,overlap}`
- Training arguments (go into `training_args`)
  - `learning_rate`, `dropout`, `hidden_units`, `mixup`, `label_smoothing`
  - Focal loss flags use hyphenated keys to match analyzer CLI:
    - `focal-loss`, `focal-loss-gamma`, `focal-loss-alpha`
  - `upsampling_mode`, `upsampling_ratio` (optional)

If a key appears in both the default skeleton and in `base_params`, the value in `base_params` wins.

### axes → per-experiment overrides

Each axis combination becomes one experiment config. The generator writes configs with these sections:

- `experiment`
  - `name`: e.g., `stage4_001`
  - `seed`: from the `seed` axis (default 123 if omitted)
- `training` / `inference`
  - `batch_size`: from the `batch_size` axis if present, else from `base.yaml`
  - `epochs`: copied from `base.yaml` (per sweep)
- `training_package`
  - `include_negatives`: true
  - `balance`: from the `balance` axis if present; otherwise the base’s default
  - `quality`: from the `quality` axis if present; otherwise the base’s default
- `training_args` (fully resolved)
  - Start with `base.yaml.training_args`
  - Overlay any axis-provided keys: `learning_rate`, `dropout`, `hidden_units`, `upsampling_mode`, `upsampling_ratio`, etc.

This results in fully resolved `training_args` per experiment config (not a minimal diff), which makes each config self-contained for inspection and debugging.

## Example

Spec:

```yaml
stage: 4
out_dir: "config/sweeps/test_sweep"

axes:
  quality:
    - ["high"]
  balance: [true]
  upsampling_mode: ["repeat"]
  upsampling_ratio: [0.0]
  seed: [456]

base_params:
  hidden_units: 512
  dropout: 0.25
  learning_rate: 0.0005
  batch_size: 32
  mixup: true
  label_smoothing: true
  focal-loss: false
  focal-loss-gamma: 0.0
  focal-loss-alpha: 0.0
  epochs: 1
```

Generated `config/sweeps/test_sweep/base.yaml` includes `training.batch_size=32`, `training.epochs=1`, and `training_args`/`analyzer_args` with the audio and analyzer defaults.

Generated experiment config (e.g., `stage4_001.yaml`):

```yaml
experiment:
  name: stage4_001
  seed: 456
training:
  batch_size: 32
  epochs: 1
inference:
  batch_size: 32
training_package:
  include_negatives: true
  balance: true
  quality:
    - high
training_args:
  fmin: 0
  fmax: 15000
  overlap: 0.0
  dropout: 0.25
  hidden_units: 512
  learning_rate: 0.0005
  focal-loss: false
  focal-loss-gamma: 0.0
  focal-loss-alpha: 0.0
  label_smoothing: true
  mixup: true
  upsampling_mode: repeat
  upsampling_ratio: 0.0
```

## Running

- Generate configs from the UI (Sweeps tab) or via:

```bash
python -m birdnet_custom_classifier_suite.sweeps.sweep_generator --spec config/sweep_specs/<name>.yaml
```

- Run the sweep:
  - The UI’s “Run sweep now” will pass `--base-config <out_dir>/base.yaml` when present.
  - Or invoke the runner directly:

```bash
python -m birdnet_custom_classifier_suite.sweeps.run_sweep config/sweeps/<name>/ --base-config config/sweeps/<name>/base.yaml --experiments-root experiments
```

## Notes

- Focal loss flags must be hyphenated (`focal-loss*`) to match analyzer CLI flags.
- If you add new training_args keys, include them in `base_params`; the generator will propagate them into the per-sweep base and per-experiment resolved configs.
- You can add a `batch_size` axis to sweep batch sizes; it applies to both training and inference for each experiment.
