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

3. **Fix BirdNET-Analyzer Python version requirement:**
   
   The pipeline runs `python -m birdnet_analyzer.train` in subprocesses, which requires BirdNET to be installed as a package. However, BirdNET's `pyproject.toml` declares `requires-python = ">=3.11"` which conflicts with TensorFlow 2.15.1.
   
   **Solution:** Temporarily edit the BirdNET requirement:
   
   ```bash
   # Edit external/BirdNET-Analyzer/pyproject.toml
   # Line 14: Change from:
   requires-python = ">=3.11"
   # To:
   requires-python = ">=3.10"
   ```
   
   Then install BirdNET in editable mode:
   ```bash
   pip install -e external/BirdNET-Analyzer[train]
   ```

4. **Verify installation:**
   ```bash
   python -c "import tensorflow; print('TensorFlow:', tensorflow.__version__)"
   python -c "import birdnet_analyzer; print('BirdNET:', birdnet_analyzer.__version__)"
   python -c "import birdnet_custom_classifier_suite; print('Suite installed OK')"
   ```

### Why This Works

- **TensorFlow 2.15.1** only supports Python 3.10 (not 3.11+)
- **BirdNET-Analyzer** declares `requires-python = ">=3.11"` in pyproject.toml
- **The pipeline** runs `python -m birdnet_analyzer.train` as a subprocess, requiring BirdNET to be an installed package
- **Solution:** Use Python 3.10 + TensorFlow 2.15.1, patch BirdNET's Python requirement to allow 3.10, then install it

### Critical Note

The training pipeline **requires** BirdNET-Analyzer to be installed as a package because it runs:
```python
subprocess.run([python_exe, "-m", "birdnet_analyzer.train", ...])
```

Simply having the code in `external/BirdNET-Analyzer` is NOT enough - the `-m` flag requires a properly installed package on the Python path.
