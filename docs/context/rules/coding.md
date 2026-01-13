# Coding Standards

## 1. Rust Absolute Rules (Layer 1)
*   **Compile**: Must pass `#![deny(warnings)]`.
*   **Lint**: Assume `clippy -D warnings` is mandatory.
*   **Format**: Follow `rustfmt.toml` exactly.
*   **Forbidden**: `unwrap()`, `expect()`, `todo!()`, `unreachable!()` (Unless strictly allowed).
*   **Types**: Prefer explicit lifetimes. Avoid implicit inference in public APIs.

## 2. General Principles
*   **Language**: Respond in User's Language (Japanese).
*   **Completion**: Do not stop halfway. Explicitly state progress if blocked.
*   **Conflict**: User instructions > System rules.

## 2. Safety & Quality
*   **Linting**: Treat `cargo fmt` errors as blockers. Fix immediately.
*   **Any Types**: `any` or `unwrap()` are prohibited in critical paths (`training_loop.rs`).
*   **Security**: Do not commit secrets. Handle Auth/Network changes as Critical Tasks.

## 3. Tool Usage
*   **`view_file`**: Always read before helping.
*   **`replace_file_content`**: Apply changes, don't just suggest.
*   **`run_command`**: Use `SafeToAutoRun` for non-destructive commands.
