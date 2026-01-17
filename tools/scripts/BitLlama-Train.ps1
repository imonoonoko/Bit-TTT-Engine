param (
    [switch]$gpu,
    [Parameter(ValueFromRemainingArguments=$true)]
    $remainingArgs
)

Write-Host "=== Bit-Llama Training Tool ===" -ForegroundColor Cyan
Write-Host "Usage: .\BitLlama-Train.ps1 [-gpu] --data <u32_file> --steps <N> ..."
Write-Host "Example: .\BitLlama-Train.ps1 --data data.u32 --steps 1000 --min-lr 0.0001"

if ($gpu) {
    Write-Host "Mode: GPU (CUDA)" -ForegroundColor Green
    cargo run --release --bin train_llama --features cuda -- $remainingArgs
} else {
    Write-Host "Mode: CPU" -ForegroundColor Yellow
    cargo run --release --bin train_llama -- $remainingArgs
}
