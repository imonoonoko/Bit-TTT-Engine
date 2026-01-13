# Strategic Roadmap: Ecosystem Replacement
"Speed and Lightness" as the Weapon.

## 0. Grand Strategy
We are not just building a model; we are replacing the **inference infrastructure**.
To break usage habits formed around Transformers/Python, we must leverage **BitNet + TTT** to offer benefits that are physically impossible with the old stack.

**Core Value Proposition**:
- **70B on Consumer GPU**: The "Impossible" made possible.
- **Zero Python Dependency**: Single Exe / DLL distribution.
- **Instant Adaptation**: TTT learning during inference.

---

## Phase C: The Invasion (Observation & Proof)
Transitioning from "Prototype" (v0.1.0) to "Production-Ready Platform".

### C-1. Monitor Core: "Proof by Numbers" (Static Analysis)
Users trust numbers, not claims. We must **visualize** the efficiency.
*Note: Real-time monitoring (nvml) is deferred to favour stability.*

- **Objective**: Show estimated VRAM savings and TTT adaptation speed in the GUI.
- **Action Items**:
    - [x] **Visualize VRAM**: Implemented "Static Efficiency Metrics" in GUI. Shows "Standard FP16" vs "Bit-TTT" usage based on model config.
    - [x] **Comparison Mode**: Implemented as a primary feature. Shows "⚡ SAVED: XX GB" badge to highlight the "Impossible".
    - [ ] **Progress Bar**: Show "Context compression rate" (TTT effect) visually.

### C-2. The Trojan Horse: `cortex_rust` (Python Bridge)
Don't force users to abandon Python yet. Give them a "Super Engine" inside Python.

- **Objective**: Make `import cortex_rust` the fastest way to run LLMs in Python.
- **Action Items**:
    - [ ] **PyO3 Polish**: Ensure `cortex_rust.pyd` builds reliably on Windows.
    - [ ] **API Design**: Create a Pythonic wrapper (`BitLlama` class) that mimics `transformers` or `llama-cpp-python` but runs on our Rust backend.
    - [ ] **Zero-Copy**: Investigate `arrow` or `numpy` view sharing to pass data between Python/Rust without overhead.

### C-3. The Experience: Distribution (Exe)
"It just works."

- **Objective**: One-click install for non-technical users.
- **Action Items**:
    - [ ] **Inno Setup**: Create `.iss` script to bundle `bit_llama.exe`, `Bit_TTT.dll`, and `vc_redist`.
    - [ ] **Portable Mode**: Ensure `config.json` and models are loaded relatively, so the folder can be moved anywhere (for USB distribution).
    - [ ] **Icon & Branding**: Professional polish for the executable.

---

## Phase D: Synergy & Scaling (The Next Wave)
Once the base is solid (Phase C), we scale.

### D-1. Japanese Mastery
- **Action**: Optimize Tokenizer for Japanese (done in part).
- **Goal**: Chat comfortably in native Japanese.

### D-2. 70B Scaling
- **Action**: Implement mmap/streaming loader for >30GB models.
- **Goal**: Run a 70B parameter model on a 24GB VRAM card (or 16GB with offloading).

---

## Immediate Next Step (Tactical)
**Execute Phase C-1 (Monitor Core)**.
The user needs to *see* the victory.

1.  [x] **Implement Static VRAM Visualization** (Replaces `monitor.rs`).
2.  [ ] Verify 70B model loading feasibility (Simulation).
3.  [ ] Progress Bar for TTT Context Compression.
