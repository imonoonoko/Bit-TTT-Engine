# Cortex Rust (Bit-TTT Engine Core)

This crate (`cortex_rust`) implements the core neural network logic and Python bindings for the Bit-TTT Engine.

## Features
- **BitLlama**: 1.58-bit Quantized Transformer model.
- **Test-Time Training (TTT)**: Layers that adapt during inference.
- **Schedule-Free Optimizer**: SOTA optimization without learning rate schedules.
- **Python Bindings**: PyO3 wrappers for Python integration.

## Usage (Rust)
```rust
use cortex_rust::model::BitLlama;
// See lib.rs for public API
```

## Usage (Python)
See [Root README](../../README.md#python-integration-optional) for setup instructions.
```bash
maturin develop --release
```
