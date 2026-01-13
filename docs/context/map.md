# Map: Bit-TTT Architecture

## 📂 Source Structure

| Path | Description | Key Components |
| :--- | :--- | :--- |
| **`crates/rust_engine/`** | **Core Library** (`cortex_rust`) | `layers/` (BitLinear, TTT), `model/` (Llama), `lib.rs` |
| **`crates/bit_llama/`** | **Application** (CLI/GUI) | `train/` (Loop, Checkpoints), `gui/` (Tauri), `inference.rs` |
| **`tools/`** | **Utilities** | `pre_demon.py` (Smoke Test), Validation Scripts |
| **`docs/`** | **Documentation** | `context/` (Agent Memory), `SPECIFICATION.md` |

## 🔗 Key Dependencies
*   **Candle**: ML framework (matrix ops).
*   **Tauri**: GUI framework.
*   **Serde**: Serialization (Config/States).

## 📍 Critical Files
*   `crates/bit_llama/src/config.rs`: Project Configuration (Single Source of Truth).
*   `crates/bit_llama/src/train/training_loop.rs`: Main training logic.
