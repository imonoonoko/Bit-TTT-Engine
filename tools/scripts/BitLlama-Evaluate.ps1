param (
    [switch]$gpu,
    [Parameter(ValueFromRemainingArguments=$true)]
    $remainingArgs
)

Write-Host "=== Bit-Llama Evaluation Tool ===" -ForegroundColor Cyan
Write-Host "Usage: .\BitLlama-Evaluate.ps1 [-gpu] --model <model_path> --data <u32_path>"
Write-Host "Example: .\BitLlama-Evaluate.ps1 --model . --data data.u32"

if ($gpu) {
    Write-Host "Mode: GPU (CUDA)" -ForegroundColor Green
    cargo run --release --bin evaluate --features cuda -- $remainingArgs
} else {
    Write-Host "Mode: CPU" -ForegroundColor Yellow
    cargo run --release --bin evaluate -- $remainingArgs
}
