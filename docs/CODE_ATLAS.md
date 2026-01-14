# Code Atlas

This document outlines the high-level structure of the Bit-TTT Engine.
For detailed documentation in Japanese, see [DEVELOPER_GUIDE_JA.md](DEVELOPER_GUIDE_JA.md).

## 📂 Directory Structure

### `crates/rust_engine` (Core Library)
The heart of the engine. Implements the BitNet architecture and Python bindings.
- **`src/python.rs`**: **PyO3 Bindings**. `PyTrainer`, `BitLlama` wrapper.
- **`src/model/`**: `llama.rs` (Architecture), `config.rs`.
- **`src/layers/`**: `bit_linear.rs` (1.58-bit), `ttt.rs` (Test-Time Training).
- **`src/kernels/`**: AVX2 (Cpu) and CUDA kernels.

### `crates/bit_llama` (Application)
CLI and GUI application logic.
- **`src/gui/`**: Tauri-based GUI source.
- **`src/train/`**: `training_loop.rs` (Main Loop), `checkpoint.rs`.
- **`src/loader.rs`**: Fast dataset loading (`memmap2`).

### `tools/`
Utility scripts (PowerShell).
- `BitLlama-Train.ps1`, `BitLlama-Chat.ps1`, `BitLlama-GUI.ps1`.

### `docs/`
- **`DEVELOPER_GUIDE_JA.md`**: **Main Developer Guide**.
- **`archive/`**: Retired documents from previous phases.
