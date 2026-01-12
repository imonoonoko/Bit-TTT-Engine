# Bit-TTT Engine Architecture

[![Featured on Orynth](https://orynth.dev/api/badge/bit-ttt-engine?theme=dark&style=default)](https://orynth.dev/projects/bit-ttt-engine)

## 1. Core Philosophy

**Bit-TTT** aims to bridge the gap between "Ultra-efficient Inference" and "Adaptive Learning".
We combine two technologies into a single, portable runtime:

1.  **1.58-bit Quantization (BitNet b1.58)**: Ternary parameters `{-1, 0, 1}` for extreme efficiency.
2.  **Test-Time Training (TTT)**: On-the-fly context learning using "Fast Weights" instead of static KV-cache.

---

## 2. System Overview

The project follows a **Rust-First, Python-Compatible** architecture.

```mermaid
graph TD
    A["Python (PyO3)"] -->|Direct Bindings| B["Rust Core Engine"]
    B -->|Candle (SIMD/AVX)| C["CPU / GPU"]

    subgraph Rust Core
    D["BitLlama (Model)"]
    E["TTT Layer (Fast Weights)"]
    F["BitLinear (Ternary Weights)"]
    end

    B --> D
    D --> E
    D --> F
```

### Component Details

### Component Details

| Module | Role | Tech Stack |
|---|---|---|
| **crates/core_engine** | Neural Network Logic | **Candle** tensor framework. Supports CPU/CUDA. |
| **crates/cortex_rust** | Python Interface | **PyO3**. Exposes `BitLlama` class directly to Python. |
| **legacy** | Deprecated Interop | Old `extern "C"` / `ndarray` implementation (isolated). |

---

## 3. Data Flow (Inference)

### Standard TTT Forward
1.  **Input**: Token IDs from Python.
2.  **Zero-Copy**: Data passed to Rust without copying via PyO3 buffer protocol.
3.  **Forward Pass**:
    *   **Embedding**: Lookup.
    *   **TTT Update**: `W_state` updated via Gradient Descent (online learning).
    *   **Projection**: 1.58-bit matrix multiplication.
4.  **Output**: Logits returned to Python as Tensor.

---

## 4. Safety & Build Options

*   **Type Safety**: Leveraging Rust's type system + Candle's strict tensor shapes.
*   **Pure Rust Build**: Can be compiled with `--no-default-features` to remove Python/PyO3 dependencies for embedded use.
*   **Device Agnostic**: Supports `cpu` (AVX) and `cuda` (GPU) via simple config switch.

## 5. Hybrid Inference Strategy (Phase 15 Architecture)

To run large models (70B+) on consumer hardware, we implement **CPU/GPU Hybrid Inference**.

*   **Layer Distribution**: Automatically distributes model layers between GPU and CPU based on `n_gpu_layers` (or Auto-Config).
*   **Zero-Copy CPU Kernel**: AVX2-optimized CPU kernels access packed weights directly without memory copy.
*   **Dynamic Dispatch**: The `forward` pass automatically switches between CUDA and AVX2 kernels based on tensor device location.

This enables practical inference speeds (~4 t/s @ 70B) even when the model exceeds VRAM capacity.

