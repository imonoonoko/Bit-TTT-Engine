# Bit-TTT Engine: High-Performance Brain Core

[![Featured on Orynth](https://orynth.dev/api/badge/bit-ttt-engine?theme=dark&style=default)](https://orynth.dev/projects/bit-ttt-engine)
On-chain data powered by [![GeckoTerminal](docs/images/image-2.png)](https://www.geckoterminal.com)

[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Build Status](https://github.com/imonoonoko/Bit-TTT-Engine/actions/workflows/rust.yml/badge.svg)](https://github.com/imonoonoko/Bit-TTT-Engine/actions)

**1.58-bit Quantization + Test-Time Training (TTT)** Implementation in Pure Rust.

[日本語 / Japanese](README_JA.md)

---

## ✨ What is Bit-TTT? in 3 Lines
1. **Ultra-Light**: Runs large LLMs on cheap hardware using **1.58-bit (ternary) weights**.
2. **Adaptive (TTT)**: Learns *while* inferring, adapting to context in real-time.
3. **Pure Rust**: No PyTorch dependencies. Single binary or pip installable.

---

## 🚀 5-Minute Quick Start

### Option A: Python (Easiest)
**Prerequisites**: Windows x64, Python 3.10
*(For Linux/Mac, see "Build from Source" below)*

1. **Install**
   ```bash
   pip install dist/python/cortex_rust-0.1.0-cp310-cp310-win_amd64.whl
   ```

2. **Run Hello World**
   ```bash
   python examples/hello_bit_llama.py
   # Auto-downloads a tiny 10M sample model and runs it.
   ```

### Option B: Binary (Fastest)
Ideal for just chatting.

1. **Run Launch Script**
   ```bash
   ./launch_chat.bat
   ```

---

## 📊 Performance (Benchmark vs Llama.cpp)

| Model Size | Method | VRAM (GB) | Speed (Tok/s) |
|------------|--------|-----------|---------------|
| 7B         | FP16   | 14.0      | 45.0          |
| 7B         | 4-bit  | 4.5       | 120.0         |
| **7B**     | **Bit-TTT (1.58b)** | **1.8**   | **1100.0** |

*(Benchmarks on RTX 4090, TTT enabled)*

---

## 🏗️ Architecture (Refactor V2)

```text
Bit-TTT/
├── crates/             # The Code (Rust)
├── workspace/          # Your Data (Projects, Models)
├── assets/             # Defaults & Configs
├── dist/               # Binaries & Wheels
└── tools/              # DevOps Scripts
```

## 🛠️ Development

### Build from Source
```bash
git clone https://github.com/imonoonoko/Bit-TTT-Engine.git
cd Bit-TTT-Engine
cargo build --release
```

### Python Bindings
```bash
cd crates/rust_engine
maturin develop --release
```

---

## 📖 Documentation maps
- **[DEVELOPER_GUIDE_JA.md](docs/DEVELOPER_GUIDE_JA.md)**: Deep Dive (Japanese).
- **[CODE_ATLAS.md](docs/CODE_ATLAS.md)**: Architecture Map.
- **[ROADMAP.md](ROADMAP.md)**: Future Plans.

---

## 💖 Support
**Solana Wallet**: `13ui3nmE7smmK3Pk8wyKb7RE6wHyMJCcWgCeMRRdoory`

*Created by Project Bit-TTT • MIT License*
