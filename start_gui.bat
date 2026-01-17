@echo off
echo Starting Bit-TTT Engine (Dev Mode)...
cargo run --release -p bit_llama --features cuda
if %errorlevel% neq 0 (
    echo.
    echo ----------------------------------------------------------------
    echo Error: Failed to start Bit-TTT Engine.
    echo Please ensure Rust (cargo) is installed and CUDA is available.
    echo ----------------------------------------------------------------
    pause
)
