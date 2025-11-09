# BirdNET Custom Classifier Suite# BirdNET-CustomClassifierSuite



A modular pipeline for training and evaluating custom BirdNET classifiers with systematic hyperparameter sweeps and data composition experiments.A modular, reproducible pipeline for training and evaluating **custom BirdNET classifiers**  

(e.g., for species detection like California Red-legged Frog, Bullfrog, etc.).

## Features

This suite automates every stage of the workflow — from data packaging and model training  

- **🎯 Streamlit UI** - Interactive sweep design, experiment tracking, and results analysisto inference, evaluation, and multi-config sweep generation.

- **🔬 Parameter Sweeps** - Test hyperparameters, data compositions, and augmentation strategies

- **📊 Automatic Evaluation** - IID/OOD metrics, leaderboards, and performance tracking---

- **🗂️ Data Composition Testing** - Sweep over curated positive/negative subset combinations

- **⚡ CLI Tools** - `birdnet-ui` and `birdnet-analyze` for quick workflows## Repository Overview



## Quick Start```

BirdNET-CustomClassifierSuite/

**Installation:**│

```powershell├── birdnet_custom_classifier_suite/   ← Python package

python -m venv .venv│   ├── cli/                           ← Console commands (birdnet-analyze, birdnet-ui)

.\.venv\Scripts\Activate.ps1│   ├── pipeline/                      ← Core training + inference pipeline

pip install -e .│   ├── sweeps/                        ← Sweep generation + batch execution

```│   ├── eval_toolkit/                  ← Aggregation and review utilities

│   ├── ui/                            ← Streamlit UI components

**Launch UI:**│   │   └── sweeps/                    ← Modular Sweeps tab components

```powershell│   └── utils/

birdnet-ui│

```├── config/

│   ├── base.yaml                      ← Global defaults (shared across stages)

**Analyze Results:**│   ├── sweep_specs/                   ← Tracked sweep definitions (YAML)

```powershell│   │   ├── example_sweep.yaml

birdnet-analyze --stage stage4_ --precision-floor 0.9 --top-n 20│   │   └── test_sweep.yaml

```│   └── sweeps/                        ← Generated sweeps (ignored by Git)

│       ├── test_sweep/

## Project Structure│       │   ├── stage0_001.yaml

│       │   ├── ...

```│       │   └── base.yaml

├── birdnet_custom_classifier_suite/│

│   ├── cli/            # Console commands (birdnet-ui, birdnet-analyze)├── experiments/                       ← Pipeline outputs (models, evals)

│   ├── pipeline/       # Training + inference pipeline├── scripts/                           ← Local scripts (dev convenience)

│   ├── sweeps/         # Sweep generation + execution│   ├── setup_env.ps1

│   ├── eval_toolkit/   # Results aggregation│   └── streamlit_app.py               ← Streamlit app entry (used by birdnet-ui)

│   └── ui/             # Streamlit interface└── requirements.txt

├── config/```

│   ├── sweep_specs/    # Tracked sweep definitions (YAML)

│   └── sweeps/         # Generated configs (gitignored)---

├── experiments/        # Model outputs + evaluations

└── results/            # Aggregated CSVs + leaderboards## Installation

```

Recommended (development/editable install):

## Workflow

```powershell

### 1. Design a Sweep (via UI or YAML)python -m venv .venv

.\.venv\Scripts\Activate.ps1

**Option A: Streamlit UI**python -m pip install --upgrade pip

```powershellpip install -e .

birdnet-ui  # Go to Sweeps tab```

```

- Set base parameters (epochs, batch_size, learning_rate)Windows convenience script (sets up venv and deps):

- Add sweep axes (dropout, hidden_units, quality combinations)

- Add data composition axes (positive/negative curated subsets)```powershell

- Click **Generate Sweep**# Create and activate venv, install deps, link package

powershell -ExecutionPolicy Bypass -File scripts/setup_env.ps1

**Option B: Manual YAML** (`config/sweep_specs/my_sweep.yaml`)```

```yaml

stage: 1This installs all dependencies and this project in editable mode. After install, the following

out_dir: config/sweeps/my_sweepconsole commands are available:

axes:

  seed: [123, 456]- `birdnet-analyze` — Analyze experiments and generate leaderboards

  dropout: [0.0, 0.25]- `birdnet-ui` — Launch the Streamlit app (Evaluate + Sweeps)

  quality: [["high", "medium"], ["high"]]

  positive_subsets:If you prefer not to install console scripts, you can still run via `python -m ...` (see below).

    - ["curated/bestLowQuality/small"]

    - ["curated/bestLowQuality/medium"]---

  negative_subsets:

    - ["curated/hardNeg/hardneg_conf_min_85"]## Quick Start

base_params:

  epochs: 50Analyze experiments and generate leaderboards:

  batch_size: 32

  learning_rate: 0.0005```powershell

```birdnet-analyze --stage stage4_ --precision-floor 0.9 --top-n 20

```

Generate configs:

```powershellLaunch the UI:

python -m birdnet_custom_classifier_suite.sweeps.sweep_generator --spec config/sweep_specs/my_sweep.yaml

``````powershell

birdnet-ui

### 2. Run the Sweep```



**Via UI:** Click **Run Sweep** in the Sweeps tabModule alternatives:



**Via CLI:**```powershell

```powershellpython -m birdnet_custom_classifier_suite.cli.analyze --stage stage4_

python -m birdnet_custom_classifier_suite.sweeps.run_sweep config/sweeps/my_sweep --verbosepython -m birdnet_custom_classifier_suite.cli.ui

``````



Each experiment:---

1. Builds training package (merges manifest + curated subsets)

2. Trains BirdNET model## Base configuration (per-sweep)

3. Runs inference (IID + OOD splits)

4. Evaluates metrics and logs to CSVEach sweep generates its own `base.yaml` inside the sweep output folder (for example `config/sweeps/<name>/base.yaml`).  

This per-sweep base is derived from the sweep spec’s `base_params` and is the only base used by the pipeline for that sweep. There is no global `config/base.yaml`.

### 3. Analyze Results

Example:

**Via UI:** Go to **Evaluate** tab, load `results/all_experiments.csv`, filter/rank

```yaml

**Via CLI:**training:

```powershell  epochs: 50

birdnet-analyze --stage stage1_ --precision-floor 0.85 --top-n 15  batch_size: 32

```training_args:

  fmin: 0

Outputs:  fmax: 15000

- `results/all_experiments.csv` - Combined experiment data  overlap: 0.0

- `results/leaderboards/` - Ranked configs by metric  hidden_units: 512

  dropout: 0.25

## Data Composition Sweeps  learning_rate: 0.0005

  label_smoothing: true

Test how curated positive/negative subsets affect performance:  mixup: true

analyzer_args:

**Positive Subsets** (`AudioData/curated/bestLowQuality/`):  fmin: 0

- `small` (51 files) - Top 5% low-quality by confidence  fmax: 15000

- `medium` (154 files) - Top 15%  overlap: 0.0

- `large` (309 files) - Top 30%  sensitivity: 1.0

```

**Negative Subsets** (`AudioData/curated/hardNeg/`):

- `hardneg_conf_min_50` (1,401 files) - FPs with conf ≥ 0.50*Important:*  

- `hardneg_conf_min_85` (981 files) - FPs with conf ≥ 0.85`fmin`, `fmax`, and `overlap` must match between `training_args` and `analyzer_args`

- `hardneg_conf_min_99` (475 files) - FPs with conf ≥ 0.99to ensure consistent spectrogram processing across training and inference.



**UI Usage:**---

1. In Sweeps tab, scroll to **Data Composition Sweep Options**

2. Click **📁 Add Folder from Explorer** to browse and select subset folders## Sweep Specs

3. Each line in the text area = one sweep combination

4. Use commas within a line to group multiple folders: `folder1,folder2`Tracked sweep definitions live under [`config/sweep_specs/`](config/sweep_specs/).  

Each spec describes:

See [`docs/DATA_COMPOSITION_SWEEPS.md`](docs/DATA_COMPOSITION_SWEEPS.md) for details.- the **axes** of parameters to vary

- the **base_params** shared across all configs

## Configuration- the **stage number** and output directory



Each sweep generates a `base.yaml` in its output folder with shared parameters:Example (`config/sweep_specs/example_sweep.yaml`):



```yaml```yaml

training:stage: 1

  epochs: 50out_dir: "config/sweeps/example_sweep"

  batch_size: 32axes:

training_args:  hidden_units: [0, 128, 512]

  fmin: 0  dropout: [0.0, 0.25]

  fmax: 15000  learning_rate: [0.0001, 0.0005, 0.001]

  overlap: 0.0  batch_size: [16, 32]

  dropout: 0.25  seed: [123]

  learning_rate: 0.0005base_params:

analyzer_args:  epochs: 50

  fmin: 0  upsampling_ratio: 0.0

  fmax: 15000  mixup: false

  overlap: 0.0  label_smoothing: false

  sensitivity: 1.0  focal_loss: false

``````



⚠️ **Important:** `fmin`, `fmax`, `overlap` must match between `training_args` and `analyzer_args`.---



## CLI Reference## Generating Sweeps



### birdnet-uiRun the generator with any sweep spec:

Launch the Streamlit interface:

```powershell```powershell

birdnet-uipython -m birdnet_custom_classifier_suite.sweeps.sweep_generator --spec config/sweep_specs/example_sweep.yaml

``````



### birdnet-analyzeThis creates a folder under `config/sweeps/<name>/` containing `base.yaml` and the generated experiment YAML configs.

Aggregate experiments and generate leaderboards:

```powershell---

birdnet-analyze [OPTIONS]

## Running Sweeps

Options:

  --exp-root PATH          Experiments folder (default: experiments)Execute all configs in a sweep folder using the training pipeline:

  --results PATH           Master CSV path (default: results/all_experiments.csv)

  --stage PREFIX           Filter experiments by prefix (e.g., stage4_)```powershell

  --precision-floor FLOAT  Minimum precision thresholdpython -m birdnet_custom_classifier_suite.sweeps.run_sweep config/sweeps/example_sweep --base-config config/sweeps/example_sweep/base.yaml --verbose

  --metric-prefix TEXT     Metric group (default: metrics.ood.best_f1)```

  --top-n INT              Number of top configs (default: 10)

```Each config will:

1. Build its training package

### Module Alternatives2. Train a BirdNET model

```powershell3. Run inference (IID + OOD)

# If console scripts don't work, use module syntax4. Evaluate metrics and update the master experiment index

python -m birdnet_custom_classifier_suite.cli.ui

python -m birdnet_custom_classifier_suite.cli.analyze --stage stage4_Outputs appear in `experiments/<experiment_name>/`.

```

---

## Testing

## Evaluating and Aggregating Results

```powershell

python -m pytest -qOnce your sweeps finish, use the evaluation toolkit to summarize results:

```

Use either the UI (Evaluate tab) or CLI:

Tests live under `tests/` and follow `test_*.py` naming conventions.

```powershell

## Version Control# CLI — collect experiments and create a master results CSV, then rank and report

birdnet-analyze --stage stage4_ --precision-floor 0.9 --top-n 20

**Track:**```

- `config/sweep_specs/*.yaml`

- `birdnet_custom_classifier_suite/`This produces a combined CSV of all runs and writes leaderboards to `results/leaderboards/`.

- `pyproject.toml`, `requirements.txt`

---

**Ignore:**

- `config/sweeps/**` (generated configs)## Version control recommendations

- `experiments/**` (model outputs)

- `AudioData/**` (local datasets)- **Track:**

- `results/**` (generated CSVs)  - all `config/sweep_specs/*.yaml`

  - `scripts/setup_env.ps1` (optional)  

---  - everything inside `birdnet_custom_classifier_suite/`



**Maintainer:** Tyler Schwenk | 2025- **Ignore:**  

  - generated `config/sweeps/**` (including each sweep’s `base.yaml` and experiment YAMLs)  
  - model and experiment outputs under `experiments/**`  
  - local audio or dataset files (`AudioData/**`)

---

## Quick Test Sweep

You can validate your environment with:

```powershell
python -m birdnet_custom_classifier_suite.sweeps.sweep_generator --spec config/sweep_specs/test_sweep.yaml
python -m birdnet_custom_classifier_suite.sweeps.run_sweep config/sweeps/test_sweep --base-config config/sweeps/test_sweep/base.yaml --verbose
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

