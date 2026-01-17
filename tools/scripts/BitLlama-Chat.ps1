param (
    [switch]$gpu,
    [Parameter(ValueFromRemainingArguments=$true)]
    $remainingArgs
)

Write-Host "=== Bit-Llama Chat Tool ===" -ForegroundColor Cyan
Write-Host "Usage: .\BitLlama-Chat.ps1 [-gpu] --model <path>"
Write-Host "Example: .\BitLlama-Chat.ps1 --model bit_llama.bitt"

if ($gpu) {
    Write-Host "Mode: GPU (CUDA)" -ForegroundColor Green
    cargo run --release --bin inference_llama --features cuda -- $remainingArgs
} else {
    Write-Host "Mode: CPU" -ForegroundColor Yellow
    cargo run --release --bin inference_llama -- $remainingArgs
}
