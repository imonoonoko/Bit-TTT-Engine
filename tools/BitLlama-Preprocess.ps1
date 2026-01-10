Write-Host "=== Bit-Llama Dataset Preprocessor ===" -ForegroundColor Cyan
Write-Host "Usage: .\BitLlama-Preprocess.ps1 --input <txt> --output <u32> --tokenizer <json>"
Write-Host "Example: .\BitLlama-Preprocess.ps1 --input corpus.txt --output data.u32 --tokenizer tokenizer.json"
cargo run --bin preprocess_data -- $args
