# Analysis Report: UX & Auto-Config (Phase 15-5)

## 1. As-Is Situation
*   **Manual Config**: Users must manually specify `-n 50` or modify strict `BitLlamaConfig` to offload layers.
*   **Default Behavior**: Without args, it might default to CPU-only or fail if GPU memory is insufficient (CUDA OOM panics).
*   **Complexity**:
    - 7B Model ~ 3.5GB VRAM.
    - 70B Model ~ 16GB VRAM (Full).
    - If user has 8GB VRAM, they need to know exactly how many layers fit.

## 2. Objective (To-Be)
The system should **"Just Work"**.
*   **Auto-Detection**: On startup, detect Total VRAM and Free VRAM.
*   **Auto-Calculation**:
    - `Available = (Free VRAM - Safety Margin)`.
    - `Layers = Available / Size_Per_Layer`.
    - `n_gpu_layers = min(Layers, Total_Layers)`.
*   **Graceful Fallback**: If standard CUDA fails, fallback to CPU implementation (which is now fast thanks to Phase 15-3).

## 3. Technical Constraints & Analysis
*   **VRAM Detection**:
    - `candle-core` does not expose `cudaMemGetInfo` directly in high-level API?
    - `cudarc` (used by `candle`) exposes it via `CudaDevice::mem_info()`.
    - We need to access this *before* loading the model.
*   **Layer Size Estimation**:
    - `BitLinear`: `N * K / 4` bytes.
    - `RMSNorm`: `K` * 4 bytes (f32).
    - `KV Cache`: `Batch * Seq * Hidden * Layers * 2` (f16 usually). KV Cache grows!
    - **Heuristic**: Reserve VRAM for KV Cache (Context Window) + Kernel Overhead + Display Output.

## 4. Intent Verification
*   **User Goal**: "Don't make me math."
*   **Safety**: Avoid OOM. Better to offload fewer layers than crash.
*   **Transparency**: Log "Detected 12GB VRAM. Offloading 30/80 layers."

## 5. Scope
*   **Codebase**: `llama.rs` (loading logic), `config.rs`, and CLI/GUI entry points.
*   **Platform**: Initial support for NVIDIA CUDA (Windows/Linux). AMD/Mac (Metal) is out of scope for "Auto-Config" unless easy.

## 6. Heuristic Formula
```rust
let needed_per_layer = (hidden_size * intermediate_size / 4) + ...; // Simplified
// Better: Measure one layer size in bytes.
// BitLayer: 8192*8192/4 * 3 (q,k,v,o,gate,up,down) ... No wait,
// Llama Layer:
//   - Attn: q,k,v,o (4 * BitLinear)
//   - MLP: gate,up,down (3 * BitLinear)
//   - Norms: 2 * RMSNorm
//   - Total Weight Size ~= 7 * (N*K/4) + small overhead.
// Plus KV Cache buffer for typical context (e.g. 2048).
```
**Safety Margin**: 1GB or 20% of VRAM?
Recommend: `Free - 2GB` (Windows consumes VRAM for desktop).
