# Code Atlas (Refactor V2)

Map of the Bit-TTT Engine codebase.
Full guide (Japanese): [DEVELOPER_GUIDE_JA.md](DEVELOPER_GUIDE_JA.md).

## 📂 Structure Map

### `crates/` (The Code)
- **`rust_engine`**: Core library (linear algebra, model defs).
  - `src/python.rs`: Python API.
  - `src/kernels/`: CUDA/AVX kernels.
- **`bit_llama`**: Application.
  - `src/gui/`: UI logic. **Points to `workspace/projects`**.
  - `src/train/`: Training loop. **Defaults to `workspace/data`**.

### `workspace/` (The Data)
user-generated content. Ignored by git.
- `projects/`, `models/`, `logs/`, `data/`.

### `dist/` (The Build)
- `binaries/`: Manual build outputs.
- `archives/`: Released zips.

### `assets/` (The Configs)
- `defaults/`: `config.json` templates.
