@echo off
cd /d "%~dp0"
echo Starting Bit-TTT Demo...
echo Model: Sample 10M (Random Weights)
echo.
Bit-TTT.exe inference --model assets/presets/sample_10m/bit_llama_10m.safetensors --prompt "Hello world" --max-tokens 30 --temp 0.8
echo.
pause
