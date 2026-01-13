# ACTION GUIDE (Root Context)

## 🌟 Mission (Why)
**Goal**: Run 70B models on 8-16GB VRAM using BitNet 1.58-bit + TTT.
*   See [Mission](docs/context/mission.md) for philosophy.

## 🗺️ Map (What)
*   **Core**: `crates/rust_engine` (Neural Ops)
*   **App**: `crates/bit_llama` (CLI/GUI)
*   **Config**: `crates/bit_llama/src/config.rs` (SSOT)
*   See [Map](docs/context/map.md) for full architecture.

## ⚙️ Protocols (How)
1.  **Analyze**: Understand BEFORE editing.
2.  **Plan**: `Lite` (Direct) vs `Full` (Docs first).
3.  **Verify**: `python tools/pre_demon.py` (Smoke Test).
*   See [Protocols](docs/context/protocols.md) for workflows.

## ⚠️ Critical Constraints
*   **Format**: `cargo fmt` is MANDATORY.
*   **VRAM**: Always verify efficiency impact.
*   **Safety**: No `unwrap()` in `training_loop.rs`.
