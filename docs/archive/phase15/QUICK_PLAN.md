# Phase 15 Quick Plan: Scaling to 7B/70B

## Objective
Enable execution of "Huge Models" (70B) on consumer hardware (8-16GB VRAM) via **Bit-Level Streaming** and **Hybrid Offloading**.

## Core Concepts

### 1. Zero-Copy Streaming (Mmap)
- **Problem**: 70B @ 1.58bit ≈ 14-16GB. Loading entire model into RAM/VRAM at once is feasible for 24GB cards but tight for system RAM on standard laptops, and impossible for VRAM on smaller cards.
- **Solution**: Use `memmap2` to map the `.safetensors` file.
  - Implement a `LazyTensor` that points to disk.
  - Dequantize only the active layer into GPU SRAM/CPU L3 Cache on the fly.
  - *Magic*: OS handles paging.

### 2. Device Map (Hybrid Offloading)
- **Problem**: Model doesn't fit in VRAM.
- **Solution**: Split layers.
  - Layers 0-10: GPU
  - Layers 11-80: CPU (System RAM)
  - Implement `PipelineParallel` logic in `BitLlama`.

## Roadmap
1. [ ] **Research**: Verify `candle` mmap support and custom bit-kernel paging.
2. [ ] **Prototype**: Create `examples/stream_inference.rs`.
3. [ ] **Implementation**:
    - [ ] `BitLoader::from_mmap(path)`
    - [ ] Update `BitLlama` to hold `Vec<Box<dyn DeviceLayer>>`.
4. [ ] **Verification**: Run 3B/7B model on 2GB VRAM mode.
