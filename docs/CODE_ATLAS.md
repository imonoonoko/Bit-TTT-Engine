# Code Atlas

## 🧠 Brain (Python)
See `docs/brain_architecture.md` (if exists) or `src/brain.py`.

## 🦀 Rust Engine (`crates/bit_llama`)
Structure verified as of Phase 14.

### Core Modules
| Module | File | Description |
|---|---|---|
| **Entry** | `src/main.rs` | CLI entry point. Sets up logging and dispatching. |
| **Model** | `src/model.rs` | Bit-Llama architecture definition (Candle). |
| **Train** | `src/train.rs` | Training loop and configuration. |

### Rust Engine (`crates/rust_engine`)
High-performance core library. `cortex_rust`.

| Module | File | Description |
|---|---|---|
| **Kernels** | `kernels/` | Device operations (CPU/AVX, CUDA). |
| - `packing.rs` | `PackedTensor` | 1.58-bit quantization logic & storage. |
| - `cpu.rs` | `BitLinearCpu` | AVX2-optimized forward pass. |
| **Layers** | `layers/` | NN modules (`BitLinear`, `RMSNorm`, `TTT`). |


### Data Pipeline (`src/data.rs`)
The `data` command is a dispatcher for subcommands:

| Subcommand | Implementation | Description |
|---|---|---|
| `download` | `src/data/download.rs` | Downloads datasets (e.g. Izumi-Lab Japanese sample). Uses `ureq`. |
| `clean` | `src/data/clean.rs` | Text cleaning (Regex). Removes HTML, URL. |
| `preprocess`| `src/data/preprocess.rs`| **Universal Parser**: Tokenizes JSONL/Raw via Jinja2 Templates. Supports Glob/Compression. |

### Tokenizer (`src/vocab.rs`)
Handles vocabulary training and loading.
- Supports **BPE** (GPT-2 style) and **Unigram** (SentencePiece style).
- Includes **NFKC** normalization for Japanese.
