param(
    [string]$CONFIG = "config/base.yaml"
)

# Ensure venv is active
if (-not (Test-Path ".venv")) {
    Write-Error "Virtual environment not found. Run setup_env.ps1 first."
    exit 1
}
& .\.venv\Scripts\Activate.ps1

# Load experiment name + training params from config_used.yaml
$cfg = (Get-Content $CONFIG | ConvertFrom-Yaml)

$expName   = $cfg.experiment.name
$expDir    = Join-Path "experiments" $expName
$dataset   = Join-Path $expDir "training_package"
$outdir    = Join-Path $expDir "model"

$epochs    = $cfg.training.epochs
$batch     = $cfg.training.batch_size
$threads   = $cfg.training.threads
$valsplit  = $cfg.training.val_split
$autotune  = $cfg.training.autotune

# Build command
$cmd = @(
    "python -m birdnet_analyzer.train"
    "`"$dataset`""
    "-o `"$outdir`""
    "--epochs $epochs"
    "--batch_size $batch"
    "--threads $threads"
    "--val_split $valsplit"
)
if ($autotune) {
    $cmd += "--autotune --autotune_trials 20 --autotune_executions_per_trial 2"
}

Write-Host "Running training..."
Write-Host ($cmd -join " ")
Invoke-Expression ($cmd -join " ")

Write-Host "Training complete. Model saved to $outdir"
