@echo off
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if %errorlevel% neq 0 (
    echo Error: Failed to setup MSVC environment.
    exit /b 1
)

echo Running Clippy...
cargo clippy --all-targets --all-features -- -D warnings
