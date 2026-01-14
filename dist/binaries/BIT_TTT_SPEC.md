# Bit-TTT (Bitwise Test-Time Training) Architecture Specification

**Status**: Research & Development (Phase 13 - Bit-Llama Construction)
**Engine**: Rust (Generic DLL via C-ABI)
**Performance**: ~60,000 TPS (Tokens Per Second) on CPU

## 1. Overview (概要)
**Bit-TTT** fuses the ultra-lightweight efficiency of **BitNet (1.58bit)** with the adaptive capabilities of **Test-Time Training (TTT)**. It is designed to enable "biological-like memory" on edge devices without GPUs.

**Bit-TTT**は、**BitNet (1.58bit)** の超軽量性と、**TTT (Test-Time Training)** の適応能力を融合させた次世代エッジAIアーキテクチャです。GPUを持たないデバイスでも、推論中にリアルタイムで学習し続ける「生物的な記憶」を実現します。

## 2. Core Concepts (コアコンセプト)

### 2.1 The 1.58bit Brain (1.58bit脳)
*   **Weights**: Strictly ternary `{-1, 0, 1}`. Stored as `i8`.
*   **Arithmetic**: Floating point multiplications (MatMul) are replaced by Integer Additions (ADD).
*   **Implementation**: Rust `ndarray` with custom `BitLinear` layer.

### 2.2 Online Learning (推論時学習)
*   **Hidden State**: Instead of a fixed KV-cache vector, the "state" is a dynamic weight matrix $W_{state}$ of a small internal neural network.
*   **Update Rule**: For every input token $x_t$, the internal model predicts the features, calculates a self-supervised reconstruction loss, and updates $W_{state}$ via Gradient Descent.
*   **Gradient Stability**: To prevent exploding gradients in the integer domain, explicit **L2 Normalization** is applied to feature vectors before the update step.

## 3. Architecture & Implementation (実装詳細)

### Component Diagram
```mermaid
graph TD
    Input[Input Token] --> BitLinear_Down[BitLinear Projection (Down)]
    BitLinear_Down --"Feature (Normalized)"--> TTT_Update[TTT Update Logic]
    
    subgraph TTT_Layer [Rust Engine]
        State[Hidden Weights W_h]
        TTT_Update --"Gradient Descent"--> State
        State --"Prediction"--> TTT_Update
    end
    
    TTT_Update --"Constructed Feature"--> BitLinear_Up[BitLinear Projection (Up)]
    BitLinear_Up --> Output[Residual Output]
```

### Optimizations (最適化)
1.  **SIMD/AVX**: Compiled with `target-cpu=native`, leveraging host-specific vector instructions (AVX2/FMA) for math operations.
2.  **Parallelism**: Uses `rayon` for multi-threaded matrix operations, scaling automatically with CPU cores.
3.  **C-ABI Export**: Distributed as a standard `.dll` (e.g. `Bit_TTT.dll`) or `.so`, interoperable with Python, C#, C++, and Node.js.

## 4. Benchmark Results (ベンチマーク結果)

Tested on Standard Consumer CPU (Rust Engine Release Build):

| Metric | Result | Note |
|---|---|---|
| **Throughput** | **~60,000 TPS** | Extremely fast. Orders of magnitude faster than standard LLM inference on CPU. |
| **Memory Effect** | **Confirmed** | Reconstruction loss decreases for repeated patterns (learning confirmed). |
| **Stability** | **Stable** | Gradient explosion prevented via Normalization. |

## 5. Future Roadmap (ロードマップ)

*   [x] **Phase 1**: Python Prototype & Logic Verification
*   [x] **Phase 2**: Rust Core Engine Implementation
*   [x] **Phase 3**: Generic Library (DLL) Packaging
*   [x] **Phase 4**: SIMD/AVX Optimization
*   [x] **Phase 5**: Python Binding & Performance Benchmark
*   [x] **Phase 6**: GPU Support (Candle Integration)
*   [x] **Phase 13**: "Bit-Llama" (Stacked Architecture & LLM Training)

## 6. Bit-Llama Architecture (Phase 13)

**Bit-Llama** represents the evolution of Bit-TTT from a single-layer test module to a full-fledged Language Model architecture.

### 6.1 Structure
*   **Deep Stacking**: Stacks multiple `TTTLayer` blocks (e.g., 4 to 12 layers) to learn complex hierarchical patterns.
*   **SwiGLU MLP**: Incorporates a **SwiGLU (SiLU Gated Linear Unit)** Feed-Forward Network after each TTT layer, similar to Llama 3, to enhance non-linear expressivity.
*   **RMSNorm**: Replaces standard LayerNorm for better stability in deep networks.
*   **Residual Connections**: `x = x + Layer(Norm(x))` structure ensures gradient flow during deep training.

### 6.2 Training Capability
*   **Hybrid Backend**: Leverages `Candle` (HuggingFace Rust) to support both **CPU** (Inference) and **CUDA** (Training).
*   **Batch Training**: Implements batch processing for `(Batch, Seq, Dim)` tensors, enabling efficient GPU saturation.
*   **BPTT (Backpropagation Through Time)**: Unrolls the TTT recurrent loop during training to optimize the "learning rule" ($W_{state}$ update mechanics) itself.

### 6.3 Performance
*   **Data**: Trained on **TinyStories** dataset.
*   **Result**: Successfully generates coherent English text from a `256 dim, 4 layer` model.
*   **Efficiency**: Inference remains CPU-friendly, maintaining high throughput.
