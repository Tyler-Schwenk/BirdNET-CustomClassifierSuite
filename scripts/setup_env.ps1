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

# Install everything from requirements.txt (handles BirdNET + pins + extras)
pip install -r requirements.txt

Write-Host "`nEnvironment setup complete. Activate with: .\.venv\Scripts\Activate.ps1"
