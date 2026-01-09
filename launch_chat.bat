@echo off
chcp 65001 > nul
echo 💬 Launching Bit-TTT Chat GUI (GPU Mode)...
cd crates\bit_llama
cargo run --release --features cuda --bin bit_llama
pause
