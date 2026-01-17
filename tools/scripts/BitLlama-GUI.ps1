Write-Host "=== Bit-Llama GUI Launcher ===" -ForegroundColor Cyan
Write-Host "Starting GUI..."
cargo run --bin launcher -- $args
