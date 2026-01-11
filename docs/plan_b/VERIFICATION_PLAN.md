# Verification Plan: Phase C (VRAM Monitor)

## 1. Observability Design
- **Log Level**: info/debug.
- **Traceability**: The VRAM update logic will log at `DEBUG` level ("VRAM: 1234MB / 24576MB").
- **Metrics**: GUI should refresh every 1000ms.

## 2. Quality Assurance (Scenarios)

### Scenario A: NVIDIA Environment (Primary)
- **Condition**: Run on a machine with NVIDIA GPU and Drivers.
- **Action**: Launch GUI, go to Dashboard/Settings.
- **Expectation**: VRAM Memory Bar reflects `nvidia-smi` values (+/- 5%).

### Scenario B: Non-NVIDIA Environment (Fallback)
- **Condition**: Run on CPU-only or AMD machine (Mock Mode).
- **Action**: Launch GUI.
- **Expectation**: Application **does not crash**. VRAM monitor either hides itself or shows "N/A" / "0/0 MB".
    - *Plan*: If `NVML` init fails, `SharedState.vram` remains `None`. GUI renders a "Static Estimate" instead or a greyed-out bar.

### Scenario C: Metrics Update
- **Condition**: While Training.
- **Action**: Observe VRAM bar.
- **Expectation**: Value fluctuates/increases as model loads.

## 3. Definition of Done
1.  `cargo build --release --features cuda` succeeds.
2.  Application launches without panic on both GPU/CPU setups.
3.  VRAM usage is visualized in the GUI.
