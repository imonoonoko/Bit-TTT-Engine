# Refactor Roadmap: Phase C (Optimization & Observability)

## Step 1: Logging Infrastructure
- [x] Add dependencies: `tracing`, `tracing-subscriber`, `tracing-appender` (maybe).
- [x] internal: Initialize subscriber in `main.rs`.
- [x] Replace `println!` in `train.rs`, `data.rs` etc. with `tracing::info!`, `warn!`, `error!`.

## Step 2: Shared Loader
- [x] Create `src/loader.rs`.
- [x] Extract `DataLoader` from `train.rs` and `EvalLoader` from `evaluate.rs` into a unified `BitLoader` struct.
- [x] Update `train.rs` and `evaluate.rs` to use `BitLoader`.

## Step 3: Config & Cleanup
- [x] Refine `BitLlamaConfig` in `config.rs`.
- [x] Ensure all subcommands use the shared config struct where applicable.

## Step 4: Verification
- [x] Run `pre_demon.py`.
- [ ] Manual GUI check.

## Step 5: Repository Safety
- [x] Implement Pre-Commit Hook to block large files (>99MB).
- [x] Clean repository of pollution.
