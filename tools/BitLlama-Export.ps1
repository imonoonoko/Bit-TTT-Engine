Write-Host "=== Bit-Llama Model Exporter ===" -ForegroundColor Cyan
Write-Host "Usage: .\BitLlama-Export.ps1 --config <json> --tokenizer <json> --model <safetensors> --output <bitt>"
Write-Host "Example: .\BitLlama-Export.ps1 --output my_model_v1.bitt"
cargo run --bin export_bitt -- $args
