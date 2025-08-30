# train.ps1
param(
    [string]$DATASET = "D:\important\projects\Frog\AudioData\ReviewedDataClean\training_packages\baseline_all_trainval",
    [string]$OUTDIR  = "D:\important\projects\Frog\models\model01",
    [int]$EPOCHS = 50,
    [int]$BATCH = 64,
    [int]$THREADS = 4,
    [float]$VALSPLIT = 0.2,
    [switch]$AUTOTUNE
)

# Ensure venv is active
if (-not (Test-Path ".venv")) {
    Write-Error "Virtual environment not found. Run setup_env.ps1 first."
    exit 1
}

& .\.venv\Scripts\Activate.ps1

# Build command
$cmd = @(
    "python -m birdnet_analyzer.train"
    "`"$DATASET`""
    "-o `"$OUTDIR`""
    "--epochs $EPOCHS"
    "--batch_size $BATCH"
    "--threads $THREADS"
    "--val_split $VALSPLIT"
)

if ($AUTOTUNE) {
    $cmd += "--autotune --autotune_trials 20 --autotune_executions_per_trial 2"
}

# Run training
Write-Host "Running training..."
Write-Host ($cmd -join " ")
Invoke-Expression ($cmd -join " ")

Write-Host "Training complete. Model saved to $OUTDIR"
