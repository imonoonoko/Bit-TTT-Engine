# Quick Refactor Plan: Phase C Cleanup

## 1. Context
- **Target**: `crates/bit_llama/src/gui/` and `monitor.rs`.
- **Reason**: Phase C (VRAM Monitor) encountered build issues and was reverted. Code contains commented-out "Reverted" blocks which clutter the codebase.

## 2. Risk Check
- **Risk**: Low. Only removing comments.
- **Verification**: `cargo check` after removal.

## 3. Implementation Steps
1.  **`gui/mod.rs`**: Remove commented `vram_monitor` field and initialization.
2.  **`gui/ui.rs`**: Remove commented `app.vram_monitor.current()` call.
3.  **`gui/tabs/settings.rs`**: Remove commented `real_vram` arg and UI block.
4.  **`monitor.rs`**: Add documentation header explaining its inactive status.
5.  **Docs**: Move `docs/plan_b/refactor/GUI_REFACTOR_PLAN.md` to `docs/legacy/`.

## 4. Verification
- `cargo check -p bit_llama`
