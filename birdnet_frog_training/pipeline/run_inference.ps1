# birdnet_frog_training/pipeline/run_inference.ps1
param(
    [string]$CONFIG = "config/base.yaml"
)

# Activate venv
& .\.venv\Scripts\Activate.ps1

# Load experiment config snapshot (produced by pipeline)
$cfg = (Get-Content $CONFIG | ConvertFrom-Yaml)

$expName    = $cfg.experiment.name
$expDir     = Join-Path "experiments" $expName
$modelDir   = Join-Path $expDir "model"
$inferenceDir = Join-Path $expDir "inference"

# Grab first .tflite model inside experiment/model
$modelPath = Get-ChildItem -Path $modelDir -Filter *.tflite | Select-Object -First 1
if (-not $modelPath) {
    Write-Error "No .tflite model found in $modelDir"
    exit 1
}

# Inference params from config
$threads   = $cfg.inference.threads
$batch     = $cfg.inference.batch_size
$minconf   = $cfg.inference.min_conf

# Input test splits (relative to dataset/audio_root)
$audioRoot = $cfg.dataset.audio_root
$testIID   = Join-Path $audioRoot "splits/test_iid"
$testOOD   = Join-Path $audioRoot "splits/test_ood"

# Ensure output dirs
New-Item -ItemType Directory -Force -Path (Join-Path $inferenceDir "TestIID") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $inferenceDir "TestOOD") | Out-Null

Write-Host " Running inference on Test-IID..."
python -m birdnet_analyzer.analyze "$testIID" `
    -o (Join-Path $inferenceDir "TestIID") `
    -c $modelPath `
    --rtype csv `
    --threads $threads `
    --batch_size $batch `
    --min_conf $minconf `
    --combine_results

Write-Host " Running inference on Test-OOD..."
python -m birdnet_analyzer.analyze "$testOOD" `
    -o (Join-Path $inferenceDir "TestOOD") `
    -c $modelPath `
    --rtype csv `
    --threads $threads `
    --batch_size $batch `
    --min_conf $minconf `
    --combine_results

Write-Host "Inference complete. Results saved to $inferenceDir"
