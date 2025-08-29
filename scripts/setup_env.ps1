# setup_env.ps1
# Bootstraps a BirdNET-Analyzer dev/training environment on Windows

# Path to Python 3.11 interpreter
$PYTHON = "D:\Python311\python.exe"

# Create venv if not exists
if (!(Test-Path ".venv")) {
    & $PYTHON -m venv .venv
}

# Activate venv
& .\.venv\Scripts\Activate.ps1

# Upgrade basics
python -m pip install --upgrade pip setuptools wheel

# Clone BirdNET-Analyzer if missing
if (!(Test-Path "external\BirdNET-Analyzer")) {
    git clone https://github.com/kahst/BirdNET-Analyzer.git external/BirdNET-Analyzer
}

# Install BirdNET in editable mode with training extras
pip install -e external/BirdNET-Analyzer[train]

# Explicit pins for known compatibility
pip install "tensorflow==2.15.0" "keras==2.15.0" keras-tuner

Write-Host "`n✅ Environment setup complete. Activate with: .\.venv\Scripts\Activate.ps1"
