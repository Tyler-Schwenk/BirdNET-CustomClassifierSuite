# setup_env.ps1
# Bootstraps BirdNET Frog Training environment on Windows

$PYTHON      = "D:\Python311\python.exe"
$VENV_PATH   = ".venv"
$REQ_FILE    = "requirements.txt"
$BIRDNET_DIR = "external\BirdNET-Analyzer"

if (!(Test-Path $VENV_PATH)) {
    & $PYTHON -m venv $VENV_PATH
}
& .\$VENV_PATH\Scripts\Activate.ps1

python -m pip install --upgrade pip setuptools wheel

if (!(Test-Path $BIRDNET_DIR)) {
    git clone https://github.com/kahst/BirdNET-Analyzer.git $BIRDNET_DIR
} else {
    Write-Host "BirdNET-Analyzer already present."
}

pip install -r $REQ_FILE
pip install -e .

Write-Host "`nEnvironment setup complete. Activate later with: .\$VENV_PATH\Scripts\Activate.ps1"
