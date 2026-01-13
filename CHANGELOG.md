# Changelog

All notable changes to this project will be documented in this file.

## [v0.1.0] - 2026-01-13
**Optimized Tokenizer & Refactoring Release**

### Phase D: High-Performance IO & Refactoring

#### ⚡ Optimization (Tokenizer)
-   **Streaming IO (OOM Fix)**: switched from `fs::read` (loading potentially huge files into RAM) to `io::copy` and `BufReader` with fixed-size chunks (1MB/4MB). This strictly bounds memory usage, preventing OOM crashes on large datasets.
-   **Async Concatenation**: Moved corpus concatenation to a background thread with generic cancellation support, eliminating GUI freezes.
-   **Parallel Sampling**: Implemented `ParallelSampler` using `rayon` and a "Writer Thread" pattern to maximize NVMe throughput during tokenizer training.

#### 🛠 Refactoring
-   **Decoupling**: Extracted data processing logic from `state.rs` and `vocab.rs` into dedicated modules:
    -   `crates/bit_llama/src/data/concat.rs`: Handles file concatenation.
    -   `crates/bit_llama/src/data/sampler.rs`: Handles parallel sampling.
-   **Code Quality**: Added `cargo fmt --all` to the `/commit-push` workflow to enforce formatting consistency.
-   **Fix**: Resolved a `clippy::ptr_arg` warning in `schedule_free.rs`.

## [v0.1.0] - 2026-01-12
**First Public Prototype Release**

### Phase C: Optimization & Observability

#### ✨ Features
- **Unified Logging System**: Replaced `println!` with `tracing` crate. Logs are now structured and piped to the GUI console via `mpsc` channels.
- **Shared DataLoader**: Introduced `BitLoader` (`src/loader.rs`) with `memmap2` for high-performance dataset loading, replacing ad-hoc loading logic.
- **VRAM Monitor Implementation**:
    - Implemented a real-time VRAM monitor using `nvml-wrapper` in `src/monitor.rs`.
    - **Note**: Due to persistent linker instability on Windows/NVCC environments, the active integration was disabled (Circuit Breaker).
    - The system currently uses **Static Estimation** (enhanced in `config.rs`) which is robust and compilation-safe.
    - The real-time monitor code is preserved in the codebase for future enablement (`feature = "cuda"`).

#### 🛠 Improvements
- **GUI Refactoring**: Modularized `ui.rs` by extracting settings and tab logic, improving maintainability.
- **Dependency Management**: Added `nvml-wrapper` as an optional dependency gated by `cuda` feature.
- **Error Handling**: Resolved multiple compilation errors and duplicate module declarations in `lib.rs`.

#### 🧹 Cleanup
- Removed legacy `core_engine.rs` references.
- Cleaned up commented-out "Circuit Breaker" code from the GUI layer.
