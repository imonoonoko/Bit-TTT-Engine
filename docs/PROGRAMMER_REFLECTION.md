# Programmer's Reflection & Demon Audit Log

This document records bugs, logic errors, and "Demon Audit" findings to prevent recurrence.

## 🔴 Issue 01: VRAM Leak in ScheduleFreeOptimizer
- **Date**: 2026-01-14
- **Symptom**: Training speed collapsed (11k -> 1.7k tok/s) after Step 50. VRAM usage grew until swap occurred.
- **Cause**: The optimizer loop updated `z` and `x` tensors (`z_new`, `x_new`) which were derived from previous states without calling `.detach()`. This kept the entire computation graph alive in memory indefinitely.
- **Fix**: Added `.detach()` to `z_new`, `x_new`, and `y_i` in `crates/rust_engine/src/optim/schedule_free.rs`.
- **Lesson**: **Always detach** state tensors in optimizers or recurrent loops. In Rust/Candle, `Tensor` holds the graph unless explicitly detached.

## 🟡 Issue 02: Legacy Code Risks
- **Date**: 2026-01-14
- **Symptom**: `precompute_for_inference` and `inference_params` existed alongside `precompute_packed`.
- **Cause**: Incomplete refactoring left dead code paths that confused the compiler and potentially the developer.
- **Fix**: Removed all legacy F32/STE inference paths.
- **Lesson**: Refactoring must be atomic and complete. Dead code should be removed immediately, not commented out.

## 🟡 Issue 03: Clippy Warnings (Manual Clamp)
- **Date**: 2026-01-14
- **Location**: `training_loop.rs:319`
- **Symptom**: `progress.min(1.0).max(0.0)` triggers `clippy::manual_clamp`.
- **Fix**: Replace with `progress.clamp(0.0, 1.0)`.
- **Lesson**: Use Rust's standard library methods (`clamp`) for readability and potentially better codegen.

## 🟢 Issue 04: Infinite Iterator Risk (Clippy)
- **Date**: 2026-01-14
- **Location**: `state.rs`, `inference_session.rs`
- **Symptom**: `lines().flatten()` on an infinite stream can loop forever if `Err` occurs repeatedly.
- **Fix**: Replaced with `lines().map_while(Result::ok)`.
- **Lesson**: Be explicit about termination conditions for infinite iterators.

## 🟢 Issue 05: Missing Standard Trait Implementations
- **Date**: 2026-01-14
- **Location**: `InferenceSession`
- **Symptom**: `new()` with no args existed, but `Default` was not implemented.
- **Fix**: Implemented `Default` calling `new()`.
- **Lesson**: Always implement `Default` for types with a parameter-less constructor to support Rust ecosystem patterns.

