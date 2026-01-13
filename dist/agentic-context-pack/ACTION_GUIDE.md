# ACTION GUIDE (Root Context)

## 🌟 Mission (Why)
**Goal**: [Project Goal Here] (e.g., "Run 70B models on 8GB VRAM")
*   See [Mission](docs/context/mission.md) for philosophy.

## 🗺️ Map (What)
*   **Core**: `[Core Module Path]`
*   **App**: `[App Module Path]`
*   **Config**: `[Config File Path]` (SSOT)
*   See [Map](docs/context/map.md) for full architecture.

## ⚙️ Protocols (How)
1.  **Analyze**: Understand BEFORE editing.
2.  **Plan**: `Lite` (Direct) vs `Full` (Docs first).
3.  **Verify**: `[Verification Command]` (e.g., `python tools/pre_demon.py`).
*   See [Protocols](docs/context/protocols.md) for workflows.

## ⚠️ Critical Constraints
*   **Format**: `cargo fmt` (or equivalent) is MANDATORY.
*   **Safety**: No unsafe operations in production paths.
