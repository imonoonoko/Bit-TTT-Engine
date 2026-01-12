# Phase 15-3: CPU Optimization & Pipelining (Detailed Plan)

## 1. Analysis Report (As-Is)
Currently, the "Hybrid" engine successfully offloads layers to CPU, but the CPU execution (`BitLinearCpu`) is the primary bottleneck.
- **Current Logic**: Naive nested loops iterate over output rows and columns sequentially.
- **Performance**: Single-threaded execution means 70% of the model runs at a fraction of potential speed.
- **Dependency**: The `forward` loop in `llama.rs` is synchronous. Layer N+1 cannot start until Layer N finishes.

## 2. Risk & Dependency Mapping
- **Branch Prediction Failure**: The current naive `if/else` logic for {-1, 0, 1} allows for significant branch misprediction penalties on CPU. **Branchless implementation** is a strict requirement.
- **Memory Bandwidth**: 1.58-bit execution is memory-bound. Efficient packing (blocked layout) is crucial.
- **Concurrency**: True pipelining (Double Buffering) requires asynchronous transfers. We must ensure thread safety when handing buffers between "Transfer Thread" and "Compute Thread".

## 3. Synergy & Value Design
- **Branchless Logic**: Using bitwise ops (`&`, `|`, `+`) instead of control flow (`if`) ensures the CPU pipeline remains full.
- **Rayon Parallelism**: Independent row computation scales linearly with cores (N-dimension).
- **SIMD Layout**: Re-ordering weights into `[32x]` blocks allows loading directly into AVX2 registers without shuffling.

## 4. Verification Strategy
- **Benchmark (`bench_cpu_kernel.rs`)**: Measure "Tokens/sec" and "GB/s".
- **Correctness**: Validate `BitLinearCpu(x) == Linear(x)` (within quantization noise).
- **Profiler**: Use detailed profiling to ensure SIMD instructions are actually generated (inspect assembly or throughput).

## 5. Roadmap
### Step 1: Benchmark & Branchless Kernel (The Baseline)
- Create `bench_cpu_kernel.rs`.
- Refactor `BitLinearCpu::forward` to use **Branchless Logic** (masks/add/sub without branches).
- **Goal**: Establish a stable, faster baseline than the naive loop.

### Step 2: Rayon Parallelism (The Scaler)
- Parallelize over "N" (Output Features) using `rayon`.
- Ensure thread-local buffers are handled correctly to avoid false sharing.

### Step 3: SIMD / AVX2 (The Booster)
- Implement explicit `std::arch::x86_64` intrinsics for 256-bit processing.
- Design "Blocked" access implementation (loading 32 weights at once).

### Step 4: Async Pipelining (The Overlap - Double Buffering)
- Implement a double-buffering scheduler.
- **Thread A**: Calculates Layer N.
- **Thread B**: Prefetches Layer N+1 from Main RAM (or GPU) to L3 Cache (or simply prepares the tensor).
- (For Hybrid): Overlap "GPU->CPU Transfer" of activations with "CPU Computation".
