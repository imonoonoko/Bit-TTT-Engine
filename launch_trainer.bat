@echo off
chcp 65001 > nul
echo 🚀 Launching Bit-TTT Training GUI (GPU Mode)...
cd crates\bit_llama
cargo run --release --features cuda --bin launcher
pause
