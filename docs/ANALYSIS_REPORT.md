# Analysis Report: Phase C (Optimization & Observability)

## 1. Executive Summary
Bit-TTT has achieved structural unification (Phase B-2). The system now runs as a single binary (`bit_llama`) with multiple subcommands.
**Phase C** focuses on refining the internal quality: **Observability** (Logging), **Configuration Management**, and **Performance**.

## 2. Current Architecture (Post-Phase B)
- **Entry**: `src/main.rs` dispatches via `src/cli.rs`.
- **Modules**: `train`, `gui`, `inference`, `data`, `vocab`, `export`, `evaluate`.
- **State**:
    - Usage of `println!` for logging (Fragile, hard to parse by GUI).
    - Ad-hoc `serde_json` usage for configs.
    - `memmap2` used in `train` and `evaluate` but logic is duplicated.

## 3. Identified Issues & Tech Debt
1.  **Logging Fragmentation**:
    - GUI reads stdout pipe.
    - No log levels (Info/Warn/Error distinction relies on string parsing).
    - No timestamps or context in logs.
2.  **Config Duplication**:
    - `BitLlamaConfig` structure is implicitly defined in multiple places or simplistic.
    - Argument passing between GUI and Backend is via CLI args strings, which is robust but could be typed better internally.
3.  **Code Duplication in Data Loading**:
    - `train.rs` and `evaluate.rs` both implement a customized `DataLoader` / `EvalLoader`. These should be unified into a shared struct in `src/data.rs` or `src/loader.rs`.

## 4. Recommendations for Phase C
1.  **Adopt `tracing`**: Replace `println!` with `tracing` macros. Add a subscriber that can send logs to GUI (via channel) and stdout.
2.  **Unify Data Loaders**: Extract `DataLoader` logic to a shared module.
3.  **Formalize Config**: Create a robust `config.rs` that handles serialization/deserialization for both CLI and GUI usage.
