@echo off
cd /d %~dp0
echo 🚀 Launching Bit-TTT Training GUI...
cargo run --release --bin launcher
pause
