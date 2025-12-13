# Environment Setup Instructions

## Current Working Configuration (Tested on Python 3.10.0)

This project has a dependency conflict between TensorFlow 2.15.1 (requires Python 3.10) and BirdNET-Analyzer 2.1.1 (requires Python >=3.11). The current working solution uses **Python 3.10** and imports BirdNET modules directly without installing the package.

### Verified Working Setup

**Python Version:** 3.10.0 (3.10.x should work)

**Key Dependencies:**
```
tensorflow==2.15.1
numpy==1.26.4
pandas==2.2.3
scikit-learn==1.7.2
librosa==0.11.0
soundfile==0.13.1
matplotlib==3.10.7
streamlit (latest)
streamlit-aggrid (latest)
altair>=5
vl-convert-python>=1.0
keras-tuner (latest)
pyyaml
tqdm
pytest>=7.0
pytest-cov>=4.0
tabulate==0.9.0
```

### Setup Instructions for New Machine

1. **Create virtual environment with Python 3.10:**
   ```bash
   python3.10 -m venv .venv
   # Windows:
   .\.venv\Scripts\activate
   # Linux/Mac:
   source .venv/bin/activate
   ```

2. **Install this package in editable mode:**
   ```bash
   pip install -e .
   ```

3. **BirdNET-Analyzer is used WITHOUT installation:**
   - The `external/BirdNET-Analyzer` directory is cloned as a submodule
   - Import paths reference it directly (e.g., via sys.path manipulation)
   - Do NOT run `pip install -e external/BirdNET-Analyzer` (it will fail due to Python version conflict)

4. **Verify installation:**
   ```bash
   python -c "import tensorflow; print('TensorFlow:', tensorflow.__version__)"
   python -c "import birdnet_custom_classifier_suite; print('Suite installed OK')"
   ```

### Why This Works

- **TensorFlow 2.15.1** only supports Python 3.10 (not 3.11+)
- **BirdNET-Analyzer** declares `requires-python = ">=3.11"` in pyproject.toml
- **Solution:** Use Python 3.10 + TensorFlow 2.15.1, and import BirdNET code directly from `external/BirdNET-Analyzer` without pip installing it

### Note on requirements.txt

The line `-e external/BirdNET-Analyzer[train]` in `requirements.txt` will fail to install. This is expected and OK - we don't need it installed as a package.

### Optional: Fix BirdNET-Analyzer Python Requirement

If you want to properly install BirdNET-Analyzer, you can temporarily patch its `pyproject.toml`:

```bash
# Edit external/BirdNET-Analyzer/pyproject.toml
# Change: requires-python = ">=3.11"
# To:     requires-python = ">=3.10"
```

Then install:
```bash
pip install -e external/BirdNET-Analyzer[train]
```

However, the current setup works fine without this.
