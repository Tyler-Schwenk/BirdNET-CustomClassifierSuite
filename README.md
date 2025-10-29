# BirdNET-CustomClassifierSuite

A modular, reproducible pipeline for training and evaluating **custom BirdNET classifiers**  
(e.g., for species detection like California Red-legged Frog, Bullfrog, etc.).

This suite automates every stage of the workflow — from data packaging and model training  
to inference, evaluation, and multi-config sweep generation.

---

## Repository Overview

```
BirdNET-CustomClassifierSuite/
│
├── birdnet_custom_classifier_suite/   ← Python package
│   ├── cli/                           ← Console commands (birdnet-analyze, birdnet-ui)
│   ├── pipeline/                      ← Core training + inference pipeline
│   ├── sweeps/                        ← Sweep generation + batch execution
│   ├── eval_toolkit/                  ← Aggregation and review utilities
│   ├── ui/                            ← Streamlit UI components
│   │   └── sweeps/                    ← Modular Sweeps tab components
│   └── utils/
│
├── config/
│   ├── base.yaml                      ← Global defaults (shared across stages)
│   ├── sweep_specs/                   ← Tracked sweep definitions (YAML)
│   │   ├── example_sweep.yaml
│   │   └── test_sweep.yaml
│   └── sweeps/                        ← Generated sweeps (ignored by Git)
│       ├── test_sweep/
│       │   ├── stage0_001.yaml
│       │   ├── ...
│       │   └── manifest.csv
│
├── experiments/                       ← Pipeline outputs (models, evals)
├── scripts/                           ← Local scripts (dev convenience)
│   ├── setup_env.ps1
│   └── streamlit_app.py               ← Streamlit app entry (used by birdnet-ui)
└── requirements.txt
```

---

## Installation

Recommended (development/editable install):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

Windows convenience script (sets up venv and deps):

```powershell
# Create and activate venv, install deps, link package
powershell -ExecutionPolicy Bypass -File scripts/setup_env.ps1
```

This installs all dependencies and this project in editable mode. After install, the following
console commands are available:

- `birdnet-analyze` — Analyze experiments and generate leaderboards
- `birdnet-ui` — Launch the Streamlit app (Evaluate + Sweeps)

If you prefer not to install console scripts, you can still run via `python -m ...` (see below).

---

## Quick Start

Analyze experiments and generate leaderboards:

```powershell
birdnet-analyze --stage stage4_ --precision-floor 0.9 --top-n 20
```

Launch the UI:

```powershell
birdnet-ui
```

Module alternatives:

```powershell
python -m birdnet_custom_classifier_suite.cli.analyze --stage stage4_
python -m birdnet_custom_classifier_suite.cli.ui
```

---

## Base Configuration

Global experiment defaults are stored in [`config/base.yaml`](config/base.yaml).  
These settings apply to all runs and can be overridden by sweep-specific configs.

Example:

```yaml
training:
  epochs: 50
  batch_size: 32
training_args:
  fmin: 0
  fmax: 15000
  overlap: 0.0
  hidden_units: 512
  dropout: 0.25
  learning_rate: 0.0005
  label_smoothing: true
  mixup: true
analyzer_args:
  fmin: 0
  fmax: 15000
  overlap: 0.0
  sensitivity: 1.0
```

*Important:*  
`fmin`, `fmax`, and `overlap` must match between `training_args` and `analyzer_args`
to ensure consistent spectrogram processing across training and inference.

---

## Sweep Specs

Tracked sweep definitions live under [`config/sweep_specs/`](config/sweep_specs/).  
Each spec describes:
- the **axes** of parameters to vary
- the **base_params** shared across all configs
- the **stage number** and output directory

Example (`config/sweep_specs/example_sweep.yaml`):

```yaml
stage: 1
out_dir: "config/sweeps/example_sweep"
axes:
  hidden_units: [0, 128, 512]
  dropout: [0.0, 0.25]
  learning_rate: [0.0001, 0.0005, 0.001]
  batch_size: [16, 32]
  seed: [123]
base_params:
  epochs: 50
  upsampling_ratio: 0.0
  mixup: false
  label_smoothing: false
  focal_loss: false
```

---

## Generating Sweeps

Run the generator with any sweep spec:

```powershell
python -m birdnet_custom_classifier_suite.sweeps.sweep_generator --spec config/sweep_specs/example_sweep.yaml
```

This creates a folder of YAML configs and a manifest CSV under `config/sweeps/<name>/`.

---

## Running Sweeps

Execute all configs in a sweep folder using the training pipeline:

```powershell
python -m birdnet_custom_classifier_suite.sweeps.run_sweep config/sweeps/example_sweep --base-config config/base.yaml --verbose
```

Each config will:
1. Build its training package
2. Train a BirdNET model
3. Run inference (IID + OOD)
4. Evaluate metrics and update the master experiment index

Outputs appear in `experiments/<experiment_name>/`.

---

## Evaluating and Aggregating Results

Once your sweeps finish, use the evaluation toolkit to summarize results:

Use either the UI (Evaluate tab) or CLI:

```powershell
# CLI — collect experiments and create a master results CSV, then rank and report
birdnet-analyze --stage stage4_ --precision-floor 0.9 --top-n 20
```

This produces a combined CSV of all runs and writes leaderboards to `results/leaderboards/`.

---

## Version Control Recommendations

- **Track:**  
  - `config/base.yaml`  
  - all `config/sweep_specs/*.yaml`  
  - `scripts/setup_env.ps1` (optional)  
  - everything inside `birdnet_custom_classifier_suite/`

- **Ignore:**  
  - generated `config/sweeps/**`  
  - model and experiment outputs under `experiments/**`  
  - local audio or dataset files (`AudioData/**`)

---

## Quick Test Sweep

You can validate your environment with:

```powershell
python -m birdnet_custom_classifier_suite.sweeps.sweep_generator --spec config/sweep_specs/test_sweep.yaml
python -m birdnet_custom_classifier_suite.sweeps.run_sweep config/sweeps/test_sweep --base-config config/base.yaml --verbose
```

This runs a 4-config micro sweep (1 epoch each) to verify your setup end-to-end.

---

## Future Additions
- Automatic sweep aggregation and leaderboard ranking  
- YAML validation schema  
- CLI presets for common stage types (e.g. “Stage 4 Robustness”)  
- Cross-platform setup scripts (Linux/macOS)

---

## Distribution (PyPI vs GitHub)

- PyPI packages are immutable snapshots — they do not auto-track GitHub. Publish to **TestPyPI** for early testers, and to **PyPI** when ready.
- GitHub installs are great during development:

```powershell
pip install git+https://github.com/<owner>/<repo>.git@main
```

Recommended release flow:
1. Tag a GitHub release (e.g., `v0.1.0`)
2. GitHub Action builds and publishes to TestPyPI
3. Promote to PyPI when validated

---

**Maintainer:**  
Tyler Schwenk  
BirdNET-CustomClassifierSuite (2025)

---

## Running tests

This project uses pytest for unit testing. You can run the test suite locally after installing the project's dependencies.

Windows (PowerShell):

```powershell
# ensure your venv is activated (see Environment Setup above)
python -m pip install -r requirements.txt
python -m pytest -q
```

Notes:
- Tests live under `tests/` and follow `test_*.py` naming conventions.
- We canonicalize metric column names to the `metrics.*` prefix in the evaluation toolkit; tests expect that canonicalization.
- Recommended CI: run `python -m pip install -r requirements.txt` and `python -m pytest -q` on push/PR (GitHub Actions or similar).

