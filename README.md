# Bit-TTT Engine: High-Performance Brain Core

**1.58-bit Quantization + Test-Time Training (TTT)** Implementation in Rust.
This engine powers the next generation of efficient, adaptive AI models.

[Japanese / 日本語](README_JA.md) (See separate file)

---

# 🇬🇧 English: Bit-TTT Engine

## Overview
**Bit-TTT Engine** is a high-performance implementation of the Bit-TTT architecture. It combines **1.58-bit quantization efficiency** with **Test-Time Training (TTT)** adaptability. It uses the **Candle** framework for tensor operations and **PyO3** for seamless Python integration.

📘 **[Read the Architecture Design](ARCHITECTURE.md)** to understand the core philosophy.

## Features
*   **Rust-First & Python-Compatible**: Core logic in Rust for speed, exposed to Python via PyO3.
*   **Zero-Copy Inference**: Efficient data handling between Rust and Python.
*   **Device Support**: Supports **CPU** (AVX optimized) and **CUDA** (GPU) execution.
*   **Pure Rust Mode**: Can be compiled as a standalone binary without Python dependencies (`--no-default-features`).
*   **Safe**: Strict adherence to Rust safety guarantees.

## Project Components

- **[`rust_engine/`](rust_engine/)**: The core implementation.
    - `core_engine.rs`: Candle-based neural network logic.
    - `lib.rs`: PyO3 bindings (`cortex_rust` module).
    - `legacy/`: Deprecated ndarray/C-API code.
- **[`bit_llama/`](bit_llama/)**: Standalone Rust binary for training/inference.

## Quick Start (Python)

### 1. Build & Install
Use `maturin` to build the Python wheel.

```bash
cd rust_engine
maturin develop --release
```

### 2. Usage
```python
import cortex_rust

# Configuration
config = cortex_rust.BitLlamaConfig(
    vocab_size=1000,
    hidden_dim=256,
    num_layers=4,
    inner_lr=0.01
)

# Load Model (Device: "cpu" or "cuda")
model = cortex_rust.BitLlama(config, "path/to/model.safetensors", device="cuda")

# Inference (Tokens)
tokens = [1, 50, 100]
logits = model.forward(tokens)
print(logits)
```

## Advanced Build Options

### Pure Rust Binary (No Python)
If you want to build a standalone Rust binary without linking to Python (e.g., for embedded deployment):

```bash
cargo build --release --no-default-features
```
(This disables the `python` feature flag in `Cargo.toml`).

### Device Selection
In `PyBitLlama`, you can specify the device:
- `device="cpu"` (Default if omitted)
- `device="cuda"` (Requires CUDA feature enabled and GPU)

---
*Created by Project Bit-TTT.*
