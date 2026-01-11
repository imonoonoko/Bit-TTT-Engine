# Analysis Report: Phase C (VRAM Monitor & Finalize)

## 1. Context & Objective
- **Current State**: Phase 2 (Reliability) and Phase 3 (Cleanup) are complete.
- **Remaining Task**: "Add VRAM usage monitor (simulated or real)".
- **Current Logic**: `crates/bit_llama/src/config.rs` contains `estimate_vram_usage` (Static Calculation).
- **Goal**: Implement **Real-time VRAM Monitoring** to provide actual feedback during training.

## 2. Technical Analysis

### 2.1. Approaches for VRAM Monitoring

| Method | Pros | Cons | Recommendation |
| :--- | :--- | :--- | :--- |
| **A. Static Estimation** | Existing, Zero Runtime Cost | Inaccurate for runtime spikes / fragmentation | Keep as "Planning" tool |
| **B. `nvidia-smi` Parsing** | No compile dependencies | Fragile (CLI output format changes), High latency (Process spawn) | Fallback only |
| **C. `nvml-wrapper` Crate** | Robust, High Performance, Standard | Adds compile dependency, Requires NVIDIA drivers installed | **Adoption target** |

### 2.2. Integration Point
- **Module**: `crates/bit_llama/src/monitor.rs` (New Module) or `crates/bit_llama/src/state.rs`.
- **Reason**: This is an Application-Level concern (GUI/Logging), not Core Engine (Tensor math).
- **Feature Flag**: Should be behind a `cuda` or `monitor` feature flag to allow compilation on non-NVIDIA machines.

## 3. Scope of Work (Phase C)
1.  **Dependency**: Add `nvml-wrapper` to `crates/bit_llama`.
2.  **Implementation**:
    - Create `VramMonitor` struct.
    - Implement polling thread (or on-demand update).
    - Update `SharedState` to hold current VRAM usage.
3.  **UI Integration**:
    - Update `Dashboard` or `TrainingGraph` to show VRAM Bar.

## 4. Release Readiness
- Ensure `README.md` reflects new features.
- Final `cargo build --release` check.
