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

### Data Pipeline (`src/data.rs`)
The `data` command is a dispatcher for subcommands:

| Subcommand | Implementation | Description |
|---|---|---|
| `download` | `src/data/download.rs` | Downloads datasets (e.g. Izumi-Lab Japanese sample). Uses `ureq`. |
| `clean` | `src/data/clean.rs` | Text cleaning (Regex). Removes HTML, URL. |
| `preprocess`| `src/data/preprocess.rs`| Tokenizes text and converts to `.u32` binary format. Uses `rayon`. |

### Tokenizer (`src/vocab.rs`)
Handles vocabulary training and loading.
- Supports **BPE** (GPT-2 style) and **Unigram** (SentencePiece style).
- Includes **NFKC** normalization for Japanese.
