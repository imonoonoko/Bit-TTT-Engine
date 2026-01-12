# Analysis Report: CPU Kernel Pipelining & Optimization

## 1. As-Is Architecture Analysis
*   **Kernel**: `BitLinearCpu` (AVX2 + ZeroCopy + Rayon).
*   **Performance**: ~4.7ms per layer (M=1, K=8192, N=8192).
*   **Throughput**: ~14.3 GOps/s (Effective).
*   **Bandwidth**: ~3.6 GB/s (Memory Load).
*   **Execution Model**:
    *   `Rayon` splits `N` (8192) into parallel tasks.
    *   Each task processes a subset of rows (e.g., 2048 rows per thread on 4 cores).
    *   Inside task: `compute_row_avx2` iterates `K` (8192) processing 32 weights at a time.

## 2. Bottleneck Identification
*   **Memory Bandwidth**: DDR5 theoretically supports 50GB/s+. We are seeing 3.6GB/s.
    *   **Gap**: 14x difference.
    *   **Cause**: The access pattern is "Streaming" but the *processing* is slow.
    *   **Compute Density**: For every 2 bits loaded (0.25 bytes), we perform 1 FMA (using packed math), bit shifts, masking, and LUT lookup.
    *   **Ops/Byte**: 4 ops / byte.
    *   **Compute Bound**: If CPU can do 64 GFlops/s, we need 16 GB/s. We are likely limited by `Instruction Throughput` (Bit Manipulation overhead) rather than pure DRAM bandwidth.

## 3. "Pipelining" Feasibility (Plan B)
User request: "Async Pipelining (Double Buffering)".
Since Layer N+1 depends on Layer N, we can only pipeline *within* Layer N.
*   **Intra-Layer Pipeline**:
    *   Current: Rayon (implicit parallelism). Threads contend for L3/DRAM.
    *   Proposed: **Producer-Consumer Threading**.
        *   Thread A (Producer): Loads `PackedTensor` block, unpacks to `[f32]` buffer in L2.
        *   Thread B (Consumer): Computes `MatMul` using pre-unpacked floats.
    *   **Pros**: Removes bit-unpacking latency from the Compute thread.
    *   **Cons**: Doubles memory bandwidth usage (Write unpacked -> Read unpacked).
    *   **Verdict**: Likely slower due to L3 pollution. Unpacking in registers (fused) is usually faster.

## 4. Alternative Optimizations (Aggressive)
If Software Prefetch failed, what else?
1.  **Bit Unpacking Optimization**: Use `_mm256_shuffle_epi8` instead of scalar shifts? (User hinted at this).
    *   Current: Scalar loop unpacking 4 weights -> 4 FMAs.
    *   Optimized: Vector shuffle to expand 8-bit packed to 32-bit integers, then float.
2.  **Block Tiling**: Process K in chunks to keep `X` in L1? (Already doing this implicitly as M=1 X fits in L1).
3.  **Huge Pages**: Reduce TLB misses for large 16MB weight tensors.

## 5. Conclusion & Proposal
We should focus on **Optimization Strategy #1: SIMD Bit Unpacking**.
The scalar bit-shifting and `LUT` generic logic inside the AVX2 kernel is the likely culprit for the "3.6 GB/s" limit.
If we vectorize the *unpacking*, we can saturate the FMA units.

**Proposal**:
Implementation of fully vectorized unpacking using `vpblendvb` or `vpshufb`.
