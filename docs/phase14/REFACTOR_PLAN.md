# Refactor Plan: Phase 14 (Japanese Data Pipeline)

## 1. Objective
Clean up the rapid prototype code from Phase 14 to ensure maintainability, modularity, and readability.
Focus on `crates/bit_llama/src/data` structure and `vocab.rs`.

## 2. Targets

### 2.1. Modularize `data.rs`
**Current**: `data.rs` acts as both the CLI entry point (Dispatcher) and the implementation of `preprocess` logic.
**Problem**: Mixed responsibilities. As `download` and `clean` grew, `preprocess` remained inline, making `data.rs` cluttered.
**Plan**:
- Extract `PreprocessArgs`, `run_preprocess`, and `process_chunk` into `crates/bit_llama/src/data/preprocess.rs`.
- `data.rs` should only contain `DataArgs`, `DataCommand` enum, and the top-level `run` dispatcher.

### 2.2. Improve `download.rs`
**Current**: Hardcoded URL inside the function.
**Plan**:
- Define constants for default URLs.
- Improve progress bar handling (extract progress logic if reusable, but `download_file` handles it fine for now).

### 2.3. Review `vocab.rs`
**Current**: `train_bpe` and `train_unigram` share `prepare_files` and `save` logic but duplicated `special_tokens` vector.
**Plan**:
- Dedup `special_tokens` definition (const or helper).

## 3. Steps

1. [ ] **Extract Preprocess**: Create `src/data/preprocess.rs` and move logic.
2. [ ] **Update Data Dispatcher**: Update `src/data.rs` to use `preprocess::run`.
3. [ ] **Refactor Vocab**: Extract common constants in `vocab.rs`.
4. [ ] **Verification**: Run `cargo check` and `bit_llama data --help`.

## 4. Safety
- Use `cargo check` after each step.
- Verify `process_chunk` parallel logic remains correct after move.
