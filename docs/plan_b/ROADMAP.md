# Roadmap: Phase C (Optimization & Observability)

## 1. Phasing Strategy

### Phase C-1: Monitor Core (State & Backend)
- **Goal**: Establish the backend capability to read real VRAM usage.
- **Tasks**:
    1. Update `Cargo.toml` with `nvml-wrapper` (optional feature `cuda`).
    2. Create `src/monitor.rs` (abstraction layer over NVML).
    3. Update `SharedState` to include `vram_usage: Option<(u64, u64)>` (used / total).
    4. Spawn a background thread in `launcher.rs` to poll VRAM (1Hz).

### Phase C-2: UI Integration (Frontend)
- **Goal**: Visualize the data.
- **Tasks**:
    1. Update `Dashboard` tab to show a Progress Bar.
    2. Color-code the bar (Green < 80%, Orange < 95%, Red > 95%).
    3. Fallback: If `vram_usage` is `None`, show the "Estimated" value (Phase 2 logic) or a "N/A" label.

## 2. Circuit Breaker (Stop Rule)
- **Time Boxing**: 2 Hours.
- **Abort Condition**: If `nvml-wrapper` causes linker errors on the user's Windows environment that cannot be solved within 30 mins, **abandon Real-Time monitor** and fallback to "Static Estimation Only" (Close the ticket).

## 3. Resources
- **Compute**: Local GPU (User's environment).
- **Cost**: Zero (Local).
