param(
    [string]$MODEL_PATH = "D:\important\projects\Frog\models\model01\model01.tflite",
    [string]$TEST_IID = "D:\important\projects\Frog\AudioData\ReviewedDataClean\splits\test_iid",
    [string]$TEST_OOD = "D:\important\projects\Frog\AudioData\ReviewedDataClean\splits\test_ood",
    [string]$OUTPUT_DIR = "D:\important\projects\Frog\evaluation"
)

# Activate venv
& .\.venv\Scripts\Activate.ps1

# Make sure output dirs exist
New-Item -ItemType Directory -Force -Path $OUTPUT_DIR\TestIID | Out-Null
New-Item -ItemType Directory -Force -Path $OUTPUT_DIR\TestOOD | Out-Null

Write-Host " Running inference on Test-IID..."
python -m birdnet_analyzer.analyze "$TEST_IID" `
    -o "$OUTPUT_DIR\TestIID" `
    -c "$MODEL_PATH" `
    --rtype csv `
    --threads 4 `
    --batch_size 64 `
    --min_conf 0.01 `
    --combine_results

Write-Host " Running inference on Test-OOD..."
python -m birdnet_analyzer.analyze "$TEST_OOD" `
    -o "$OUTPUT_DIR\TestOOD" `
    -c "$MODEL_PATH" `
    --rtype csv `
    --threads 4 `
    --batch_size 64 `
    --min_conf 0.01 `
    --combine_results

Write-Host "Inference complete. Results saved to $OUTPUT_DIR"
Write-Host " Next step: run metrics script (evaluate_results.py) on the CSV outputs."
