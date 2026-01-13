# Mission: Bit-TTT Engine

## 🎯 Primary Objective
**Run 70B parameter models on consumer hardware (8-16GB VRAM) with efficient inference.**

## 🧩 Core Technologies
1.  **BitNet 1.58-bit Quantization**:
    *   Ternary weights {-1, 0, +1}.
    *   Extreme compression vs FP16.
2.  **Test-Time Training (TTT)**:
    *   Adaptive attention replacement with online learning.
    *   "Train during inference" capability.

## 🛡️ Design Philosophy
*   **Pure Rust**: No Python dependencies for the core engine.
*   **Performance**: Maximize throughput (tokens/sec).
*   **Accessibility**: Democratize large models for edge devices.
