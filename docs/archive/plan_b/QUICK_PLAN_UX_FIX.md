# Quick Plan: Fix Chat UX & Code Issues

## 1. Context
- **Issues**:
  - Chat Input: `Ctrl+Enter` doesn't send, `Enter` doesn't newline.
  - `core_engine.rs`: Reported Syntax Error (Line 1000).
  - `llama.rs`: Unused import `fs2::FileExt`.

## 2. Implementation
### 2.1. Chat UX (`inference.rs`)
- **Diagnosis**: `egui` event handling might be tricky.
- **Fix**: Use `ui.input_mut(|i| i.consume_key(...))` pattern to explicitly handle `Ctrl+Enter` *before* the TextEdit, OR use `egui::TextEdit::multiline` return value events.
- **Alternative**: `TextEdit` consumes `Enter` by default?
- **Plan**:
    - Check if I can intercept `Ctrl+Enter` simply.
    - If `Enter` for newline is broken, it might be because I am not running the loop correctly or `TextEdit` config.
    - Will try to simplify the logic.

### 2.2. `core_engine.rs` Syntax Error
- **Action**: Check file content at line 1000. It might be trailing garbage or a missing brace from previous edits.

### 2.3. `llama.rs` Unused Import
- **Action**: Verify `fs2` dependency visibility. Maybe `crates/rust_engine/Cargo.toml` has issues?
- **Analysis**: If `lock_shared` calls validly, `FileExt` must be used. Compiler says unused -> maybe `lock_shared` is resolving to something else? Or code block is dead?

## 3. Verification
- `cargo check` clean.
- Manual UX verification.
