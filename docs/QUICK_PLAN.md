# Refactor Plan: ProjectConfig Initialization

## 1. Objective
Replace manual `ProjectConfig` initialization in `training_loop.rs` with a `from_args` constructor.
**Problem**: Adding new fields (like `n_kv_heads`) breaks compilation in `training_loop.rs` because checks are manual.
**Solution**: Centralize initialization logic in `config.rs`.

## 2. Changes

### `crates/bit_llama/src/config.rs`
- Add `pub fn from_args(args: &TrainArgs) -> Self`
- Use `Default` base and override with args.

### `crates/bit_llama/src/train/training_loop.rs`
- Replace `ProjectConfig { ... }` with `ProjectConfig::from_args(&args)`

## 3. Verification
- `cargo check -p bit_llama`
- `cargo build --release` (regression test)
