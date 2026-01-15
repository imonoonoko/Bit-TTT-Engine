# Programmer Reflection (Demon Audit)

## 🔴 Issue 01: Precision Loss in MeZO Noise Generation
- **発生**: `crates/bit_llama/src/train/training_loop.rs` (Line 80)
- **症状**: `clippy::cast_possible_truncation` warning.
- **原因**: `rand_distr::Normal` generates `f64`, but we cast to `f32` for the Tensor.
- **✅ Lesson**: This is intentional for VRAM efficiency (MeZO uses mixed precision concepts), but explicit `.to_f32()` or comment is better than raw cast to suppress warning. The noise magnitude is small, so truncation impact is negligible.

## 🟡 Issue 02: Unnecessary Debug Formatting
- **発生**: Multiple locations in `training_loop.rs`.
- **症状**: `clippy::unnecessary_debug_formatting`.
- **原因**: Use of `{:?}` for types that implement `Display` or simple values.
- **✅ Lesson**: Use `{}` where possible for cleaner output and consistency.

## 🟡 Issue 03: Missing Panic Documentation
- **発生**: `perturb_weights` function.
- **症状**: `clippy::missing_panics_doc`.
- **原因**: The function performs operations that might panic (e.g., `unwrap` inside), but the docstring doesn't declare it.
- **✅ Lesson**: Public functions must document failure cases. "Panics if..." section is required for pedantic code.
