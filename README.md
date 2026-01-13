# Bit-TTT Engine: High-Performance Brain Core

[![Featured on Orynth](https://orynth.dev/api/badge/bit-ttt-engine?theme=dark&style=default)](https://orynth.dev/projects/bit-ttt-engine)

On-chain data powered by
[![GeckoTerminal](image.png)](https://www.geckoterminal.com)
[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Build Status](https://github.com/imonoonoko/Bit-TTT-Engine/actions/workflows/rust.yml/badge.svg)](https://github.com/imonoonoko/Bit-TTT-Engine/actions)

**1.58-bit Quantization + Test-Time Training (TTT)** Implementation in Pure Rust.

[日本語 / Japanese](README_JA.md)

---

## ✨ What is Bit-TTT?

**Bit-TTT Engine** combines two cutting-edge techniques:
- **BitNet 1.58-bit Quantization**: Extreme compression using ternary weights {-1, 0, +1}
- **Test-Time Training (TTT)**: Adaptive attention replacement with online learning

The goal: Run **70B parameter models on 8-16GB VRAM** with efficient inference.

## 📊 Current Status (2026-01)

| Feature | Status | Description |
|---------|--------|-------------|
| Core Engine (`cortex_rust`) | ✅ Complete | Candle-based neural network implementation |
| Training Pipeline | ✅ Complete | End-to-end training in pure Rust |
| Streaming Inference | ✅ Complete | ~1100 tokens/sec on CPU |
| GUI Trainer | ✅ Complete | Tauri-based visual training interface |
| Python Bindings (PyO3) | ✅ Complete | Optional Python integration |
| Japanese Tokenizer | ✅ Complete | Phase 14 (Unigram) |
| 7B/70B Scaling | ✅ Complete | Phase 15 (Auto-Config & AVX2) |
| WASM/Browser Support | 🚧 Planned | Phase 16 (Future) |

## 🏗️ Architecture

```
Bit-TTT Engine
├── crates/
│   ├── rust_engine/         # Core library (cortex_rust)
│   │   ├── layers/          # Neural network layers
│   │   │   ├── rms_norm.rs    # RMS Normalization
│   │   │   ├── bit_linear.rs  # 1.58-bit Linear Layer
│   │   │   ├── swiglu.rs      # SwiGLU MLP
│   │   │   └── ttt.rs         # TTT Layer
│   │   ├── model/           # Model architecture
│   │   │   ├── block.rs       # Transformer Block
│   │   │   ├── llama.rs       # BitLlama Model
│   │   │   └── config.rs      # Configuration
│   │   ├── python.rs        # PyO3 bindings
│   │   └── lib.rs           # Public API
│   │
│   └── bit_llama/           # CLI application
│       ├── train/           # Training module
│       │   ├── args.rs        # CLI arguments
│       │   ├── checkpoint.rs  # State management
│       │   └── training_loop.rs  # Main loop
│       ├── gui/             # Tauri GUI
│       └── inference.rs     # Inference engine
│
├── models/                  # Trained model checkpoints
├── data/                    # Training datasets
└── tools/                   # Utility scripts
```

## 🚀 Quick Start

### Prerequisites
- Rust 1.70+
- (Optional) CUDA 11.8+ for GPU acceleration

### 1. Build
```bash
git clone https://github.com/imonoonoko/Bit-TTT-Engine.git
cd Bit-TTT-Engine
cargo build --release
```

### 2. Training
```bash
# Using launch script (Windows)
./launch_trainer.bat

# Manual training
cargo run --release --bin train_llama -- \
    --data data/TinyStories \
    --dim 256 \
    --layers 8 \
    --steps 10000 \
    --lr 3e-4
```

### 3. Inference
```bash
# Using launch script (Windows)
./launch_chat.bat

# Manual inference
cargo run --release --bin bit_llama -- \
    --model models/my_model \
    --prompt "Hello Bit-TTT!" \
    --max-tokens 100 \
    --temp 0.8
```

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | System design philosophy |
| [ROADMAP.md](ROADMAP.md) | Future development plans |
| [docs/SPECIFICATION.md](docs/SPECIFICATION.md) | Technical specification |
| [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) | Contribution guide |

## 🛠️ Development Commands

```bash
# Run all tests
cargo test --workspace

# Check compilation
cargo check --workspace

# Format code
cargo fmt --all

# Run linter
cargo clippy --workspace
```

## 🐍 Python Integration (Optional)

```bash
cd crates/rust_engine
pip install maturin
maturin develop --release
```

```python
import cortex_rust

config = cortex_rust.BitLlamaConfig(
    vocab_size=16384,
    hidden_dim=256,
    num_layers=8,
    inner_lr=0.1
)

model = cortex_rust.BitLlama(config, "model.safetensors", device="cuda")
logits = model.forward(token_id=42)
```

## 💖 Support

**Solana Wallet**: `13ui3nmE7smmK3Pk8wyKb7RE6wHyMJCcWgCeMRRdoory`

---

*Created by Project Bit-TTT • MIT License*


