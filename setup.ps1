# Quick setup script for new machines (Windows PowerShell)
# Usage: .\setup.ps1

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "BirdNET Custom Classifier Suite - Environment Setup" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# Check Python version
$pythonVersion = python --version 2>&1 | Select-String -Pattern "(\d+\.\d+\.\d+)" | ForEach-Object { $_.Matches.Groups[1].Value }
Write-Host "Detected Python version: $pythonVersion" -ForegroundColor Yellow

if (-not ($pythonVersion -like "3.10.*")) {
    Write-Host "WARNING: This project requires Python 3.10.x, you have $pythonVersion" -ForegroundColor Red
    Write-Host "Continuing anyway, but you may encounter issues..." -ForegroundColor Red
}

Write-Host ""
Write-Host "Step 1: Installing main dependencies..." -ForegroundColor Green
pip install -r requirements-frozen.txt

Write-Host ""
Write-Host "Step 2: Installing this package in editable mode..." -ForegroundColor Green
pip install -e .

Write-Host ""
Write-Host "Step 3: Patching BirdNET-Analyzer Python requirement..." -ForegroundColor Green
$birdnetToml = "external\BirdNET-Analyzer\pyproject.toml"

if (-not (Test-Path $birdnetToml)) {
    Write-Host "ERROR: BirdNET-Analyzer not found at $birdnetToml" -ForegroundColor Red
    Write-Host "Make sure you've cloned the submodule: git submodule update --init --recursive" -ForegroundColor Red
    exit 1
}

# Backup original
Copy-Item $birdnetToml "$birdnetToml.backup"

# Patch the Python version requirement
(Get-Content $birdnetToml) -replace 'requires-python = ">=3.11"', 'requires-python = ">=3.10"' | Set-Content $birdnetToml
Write-Host "[OK] Patched $birdnetToml (backup saved as .backup)" -ForegroundColor Green

Write-Host ""
Write-Host "Step 4: Installing BirdNET-Analyzer in editable mode..." -ForegroundColor Green
pip install -e external\BirdNET-Analyzer[train]

Write-Host ""
Write-Host "Step 5: Verifying installation..." -ForegroundColor Green
python --% -c "import tensorflow; print('[OK] TensorFlow:', tensorflow.__version__)"
python --% -c "import birdnet_analyzer; print('[OK] BirdNET-Analyzer:', birdnet_analyzer.__version__)"
python --% -c "import birdnet_custom_classifier_suite; print('[OK] Custom Classifier Suite: OK')"

Write-Host ""
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "Setup complete!" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  - Review ENVIRONMENT_SETUP.md for details" -ForegroundColor Yellow
Write-Host "  - Run tests: pytest" -ForegroundColor Yellow
Write-Host "  - Start UI: streamlit run scripts/streamlit_app.py" -ForegroundColor Yellow
Write-Host ""
