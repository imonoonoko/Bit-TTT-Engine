# Quick Refactor Plan: Demon Audit Fixes

## 🎯 Goal
Clean up `cargo clippy` warnings detected during the Demon Audit.

## ⚠️ Issues to Fix
1.  **Precision Loss (`clippy::cast_possible_truncation`)**
    -   `training_loop.rs`: `f64` -> `f32` cast in noise generation.
    -   **Fix**: Explicitly suppress lint with justification (MeZO noise is stochastic anyway, precision loss is acceptable).

2.  **Unnecessary Debug Formatting (`clippy::unnecessary_debug_formatting`)**
    -   `training_loop.rs`: `PathBuf` printing often uses `{:?}`.
    -   **Fix**: Change `{:?}` to `{}` and use `.display()` for `Path` objects.

3.  **Missing Panic Documentation (`clippy::missing_panics_doc`)**
    -   `training_loop.rs`: `perturb_weights` calls `unwrap()`.
    -   **Fix**: Add `/// # Panics` section to the function docstring.

## 📝 Steps
1.  Modify `perturb_weights` signature docs.
2.  Add `#[allow(clippy::cast_possible_truncation)]` to the mapping closure.
3.  Scan and replace `{:?}` with `{}` + `.display()` for paths in `run`.
