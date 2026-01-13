# Protocols: Standard Operating Procedures

## 🔄 Workflows

### 🟢 Lite Process (Small Fixes)
1.  **Analysis**: Read file -> Identify change.
2.  **Edit**: `replace_file_content` (Atomic).
3.  **Verify**: Quick check (Compile/Lint).

### 🔴 Full Process (Features/Refactor)
1.  **Plan**: Create `docs/QUICK_PLAN.md`.
2.  **Approve**: Get user confirmation via `notify_user`.
3.  **Implement**: Step-by-step changes.
4.  **Verify**: Run Smoke Test.

## 🛡️ Coding Rules
*   **Formatting**: Always run `[Formatter Command]` (e.g. `cargo fmt`) before commit.
*   **Safety**: Handle errors gracefully.
*   **Context**: Keep `[Config SSOT]` as the single source of truth.

## 🧪 Verification Commands
```bash
[Test Command 1]
[Test Command 2]
```
