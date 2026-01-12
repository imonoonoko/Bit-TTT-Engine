# Post-Release Refactor Plan

## Objective
Clean up "quick fixes" introduced during the final release push. Focus on safety (removing `unwrap`) and logging consistency.

## Changes

### 1. `crates/bit_llama/src/gui/mod.rs`
- **Refactor**: Replace `if current_project.is_some() { ... unwrap() ... }` with `if let Some(project) = ...`.
- **Logging**: Replace `println!` in `setup_custom_fonts` with `tracing::info!`.
- **Refactor**: Wrap `render_workspace` call in `if let Some` check to ensure strict safety, although it is currently implicitly guarded.

### 2. `crates/bit_llama/src/gui/ui.rs`
- **Safety**: Verify `unwrap()` usages.
- **Refactor**: Consider changing signature to `render_workspace(project: &mut ProjectState, ...)` to avoid repeated unwraps, BUT this might be too invasive for a "quick refactor" as `AppTab::Home` or `AppTab::Inference` might not need project.
    - *Decision*: Keep signature but ensure caller guards it. Add a debug_assert or clearer comment.

### 3. Verification
- `cargo check` (Passed)
- `cargo build -p bit_llama --release` (Skipped as check passed)

## Status
- [x] Refactoring Completed on 2026-01-12.
- Safety improved (removed unwraps).
- Logging consistent (tracing).
