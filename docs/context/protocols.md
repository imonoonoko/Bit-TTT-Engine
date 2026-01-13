# Protocols: Standard Operating Procedures

## 🔄 Workflows

### 🟢 Lite Process (Small Fixes)
1.  **Analysis**: Read file -> Identify change.
2.  **Edit**: `replace_file_content` (Single step).
3.  **Verify**: Quick check (Compile/Lint).

### 🔴 Full Process (Features/Refactor)
1.  **Plan**: Create `docs/QUICK_PLAN.md` or updated `ROADMAP.md`.
2.  **Approve**: Get user confirmation via `notify_user`.
3.  **Implement**: Step-by-step changes with verification.
4.  **Verify**: Run `python tools/pre_demon.py` (Smoke Test).

## 🛡️ Coding Rules
*   **Formatting**: Always run `cargo fmt` before commit.
*   **Safety**: No `unwrap()` in production paths. Use `anyhow::Result`.
*   **VRAM**: Prioritize memory efficiency (Check `ProjectConfig::estimate_efficiency`).
*   **Context**: Keep `config.rs` as the config source of truth.

## 🧪 Verification Commands
```bash
cargo check -p bit_llama
cargo test --workspace
python tools/pre_demon.py
```
