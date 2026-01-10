# Refactor Roadmap: Phase C (Optimization & Observability)

## Step 1: Logging Infrastructure
- [ ] Add dependencies: `tracing`, `tracing-subscriber`, `tracing-appender` (maybe).
- [ ] internal: Initialize subscriber in `main.rs`.
- [ ] Replace `println!` in `train.rs`, `data.rs` etc. with `tracing::info!`, `warn!`, `error!`.

## Step 2: Shared Loader
- [ ] Create `src/loader.rs`.
- [ ] Extract `DataLoader` from `train.rs` and `EvalLoader` from `evaluate.rs` into a unified `BitLoader` struct.
- [ ] Update `train.rs` and `evaluate.rs` to use `BitLoader`.

## Step 3: Config & Cleanup
- [x] Refine `BitLlamaConfig` in `config.rs`.
- [x] Ensure all subcommands use the shared config struct where applicable.

## Step 4: Verification
- [ ] Run `pre_demon.py`.
- [ ] Manual GUI check.
