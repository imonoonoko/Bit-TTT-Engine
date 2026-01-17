# Code Atlas (Refactor V2)

Map of the Bit-TTT Engine codebase.
Full guide (Japanese): [DEVELOPER_GUIDE_JA.md](DEVELOPER_GUIDE_JA.md).

## 📂 Structure Map

### `crates/` (The Code)
- **`rust_engine`**: Core library (linear algebra, model defs).
  - `src/python.rs`: Python API.
  - `src/kernels/`: CUDA/AVX kernels.
- **`bit_llama`**: Main GUI Application (Training/Chat).
  - `src/gui/`: UI logic.
- **`bit_converter`**: Standalone Converter Tool.
  - `src/main.rs`: Converter GUI.

### `tools/` (The Utilities)
- **`conversion/`**: Python scripts for model conversion.
- **`debug/`**: Benchmarking and verification scripts.
- **`data/`**: Dataset preparation scripts.
- **`scripts/`**: PowerShell automation wrappers.

### `workspace/` (The Data)
user-generated content. Ignored by git.
- `projects/`, `models/`, `logs/`, `data/`.

### `dist/` (The Build)
- `binaries/`: Manual build outputs.
- `archives/`: Released zips.

### `assets/` (The Configs)
- `defaults/`: `config.json` templates.
