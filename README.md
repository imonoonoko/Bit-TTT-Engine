# Bit-TTT Engine: High-Performance Brain Core

[![Featured on Orynth](https://orynth.dev/api/badge/bit-ttt-engine?theme=dark&style=default)](https://orynth.dev/projects/bit-ttt-engine)

On-chain data powered by [![GeckoTerminal](docs/images/image-2.png)](https://www.geckoterminal.com)

[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Build Status](https://github.com/imonoonoko/Bit-TTT-Engine/actions/workflows/rust.yml/badge.svg)](https://github.com/imonoonoko/Bit-TTT-Engine/actions)

**1.58-bit Quantization + Test-Time Training (TTT)** Implementation in Pure Rust.
**Current Version: v0.3.0 (Model Lab Update)**

[日本語 / Japanese](README_JA.md) | [📖 User Guide](docs/USER_GUIDE.md)

---

## ✨ What is Bit-TTT? in 3 Lines
1. **Ultra-Light**: Runs large LLMs on cheap hardware using **1.58-bit (ternary) weights**.
2. **Adaptive (TTT)**: Learns *while* inferring, adapting to context in real-time as a "Soul".
3. **Pure Rust**: No PyTorch dependencies. Single binary manages both "Body" (Model) and "Soul" (Memory).

---

## 🚀 v0.3.0 New Features: Model Lab & Sleep Mode

*   **🔬 Model Lab**: Dedicated tab for managing Model Loading and "Soul (.soul)" persistence.
*   **🌙 Sleep Mode**: Consolidates short-term memory into long-term memory by "dreaming" (fast replay learning) conversation logs.
*   **💬 Chat UI**: Separated system logs to focus purely on conversation.

---

## 🚀 5-Minute Quick Start

### Option A: Binary Chat (Recommended)
1.  **Unzip the distributed package**
2.  **Run `start_gui.bat` (or `bit_llama.exe`)**
3.  Load a model in **Model Lab** tab, then start talking in **Chat** tab!
    *   See [User Guide](docs/USER_GUIDE.md) for details.

### Option B: Python (Developer)
**Prerequisites**: Windows x64, Python 3.10
*(For Linux/Mac, see "Build from Source" below)*

1. **Install**
   ```bash
   pip install dist/python/cortex_rust-0.3.0-cp310-cp310-win_amd64.whl
   ```

2. **Run Hello World**
   ```bash
   python examples/hello_bit_llama.py
   # Auto-downloads a tiny 10M sample model and runs it.
   ```

---

## 💻 System Requirements

- **OS**: Windows 10 / 11 (64-bit)
- **GPU**: NVIDIA GeForce GTX 10 Series (Pascal) or newer recommended.
    - Requires **CUDA 12.x compatible Driver** (v530 or higher).
    - **Note**: CUDA Toolkit installation is **NOT** required (Runtime DLLs are bundled).
- **VRAM**: 4GB+ recommended (2GB minimal for 10M models).

---

## 📊 Performance (Benchmark vs Llama.cpp)

| Model Size | Method | VRAM (GB) | Speed (Tok/s) |
|------------|--------|-----------|---------------|
| 7B         | FP16   | 14.0      | 45.0          |
| 7B         | 4-bit  | 4.5       | 120.0         |
| **7B**     | **Bit-TTT (1.58b)** | **1.8**   | **1100.0** |

*(Benchmarks on RTX 4090, TTT enabled)*

---

## 🏗️ Architecture (Refactor V3)

```text
Bit-TTT/
├── crates/             # The Code (Rust)
├── workspace/          # User Data (Projects, Models, Memories)
│   ├── projects/       # Project Configs
│   ├── models/         # Model Files (.safetensors)
│   └── memories/       # Conversation Logs (.jsonl)
├── assets/             # Defaults & Configs
├── dist/               # Binaries
└── docs/               # Docs (User Guide, Specifications)
```

## 🛠️ Development

### Build from Source
```bash
git clone https://github.com/imonoonoko/Bit-TTT-Engine.git
cd Bit-TTT-Engine
cargo build --release -p bit_llama --features cuda
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

## 🙏 Acknowledgments / 謝辞

This project incorporates ideas and techniques inspired by and adapted from the DroPE method published by Sakana AI.

**Original work:**
*   **Title**: Extending the Context of Pretrained LLMs by Dropping Their Positional Embeddings
*   **Authors**: Yoav Gelberg, Koshi Eguchi, Takuya Akiba, Edoardo Cetin
*   **Source**: [arXiv:2512.12167](https://arxiv.org/abs/2512.12167) (Submitted on 13 Dec 2025)
*   **License**: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

**Modifications / 改変点:**
We have adapted the positional embedding dropping approach and recalibration concept for our Pure Rust-based, low-bit quantized Test-Time Training (TTT) engine (Bit-TTT-Engine / bit_llama). This includes re-implementation in Rust (no Python dependencies), integration with 1.58-bit quantization, and application to edge-device friendly real-time adaptation, which differs from the original Hugging Face / PyTorch-focused implementation.

The rest of this project is licensed under the MIT License (see LICENSE file).

## 💖 Support
**Solana Wallet**: `13ui3nmE7smmK3Pk8wyKb7RE6wHyMJCcWgCeMRRdoory`

*Created by Project Bit-TTT • MIT License*
