# Phase 15 Roadmap: Scaling to Huge Models (8GB VRAM Target)

**Goal**: Run 70B class models on 8GB VRAM hardware with interactive speeds.
**Constraint**: 70B @ 2-bit ≈ 17.5GB. Needs Hybrid (GPU + CPU) approach.

## Step 1: Bit-Packing & Dual Kernels (The Foundation)
Implement the core 2-bit storage format and computation kernels for *both* devices.
- [ ] **Data Structure**: Implement `PackedTensor` (uint8 holds 4 weights).
- [ ] **CUDA Kernel**: Implement 1.58-bit matmul (add/sub only) for GPU.
    - *Goal*: Eliminate multiplication, reduce VRAM usage by 4x-8x.
- [ ] **CPU Kernel (SIMD)**: Implement AVX2/AVX-512 optimized 1.58-bit matmul.
    - *Critical*: Since 8GB VRAM can only hold ~30% of a 70B model, the CPU must process the remaining 70% fast enough.

## Step 2: Hybrid Layer Management (The Architect)
Enable the model to live across devices without streaming weights constantly.
- [ ] **Device Map**: Allow `BitLlama::load` to accept a split configuration (e.g., "Layers 0-20: CUDA, 21-80: CPU").
- [ ] **Activation Transfer**: Implement efficient low-latency transfer of activation tensors between GPU and CPU during `forward`.
    - *Note*: Only small activation vectors move; heavy weights stay pinned in RAM/VRAM.

## Step 3: Pipelined Execution (The Speedup)
Hide the latency of CPU-GPU synchronization.
- [ ] **Speculative Execution (Optional)**: Can the GPU draft while CPU verifies? (Advanced)
- [ ] **Pipeline Parallelism**: While CPU computes Layer N, can GPU start preparing for Layer N+1 (if possible)?

## Step 4: User Experience
- [ ] **Auto-Config**: Detect VRAM size (e.g., 8GB) and automatically suggest the optimal split (e.g., 20/60 split).
