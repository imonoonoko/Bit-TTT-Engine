# Phase 15-4: Async Pipelining & Optimization Strategy (Plan B)

## 1. Analysis & Decomposition
The user requested "Async Pipelining (Double Buffering)" with extreme caution.
Current Context:
- **Task**: Autoregressive Inference (`M=1`, Generating 1 token at a time).
- **Dependency**: Layer `i+1` strictly requires output of Layer `i`.
- **Bottleneck**: Memory Bandwidth (Weight Loading). `4.7ms` indicates we are hitting ~3.6 GB/s.

**The Paradox**:
We cannot "calculate Layer i+1" while "calculating Layer i" because Layer i+1 needs Layer i's input.
Therefore, "Double Buffering" at the **Layer Level** is impossible for `M=1` inference logic, *UNLESS* we are streaming weights from VRAM/Disk (hiding fetch latency).
Since currently all weights are in RAM, "Fetching" is effectively "L3 Cache Miss" latency.

**Solution: Intra-Layer Pipelining (Micro-Double Buffering)**
We apply Double Buffering at the **Cache Line Level** using Software Prefetching (`_mm_prefetch`).
- **Compute Thread**: Processes Block `K`.
- **Prefetch Engine (Hardware/Instruction)**: Loads Block `K+Offset` into L1/L2.

## 2. Risk & Dependency Mapping
- **Prefetch Distance**: Too short = latency not hidden. Too long = cache pollution (evicting needed data).
- **Overhead**: Issuing prefetch instructions consumes slots in the CPU pipeline.
- **Portability**: `_mm_prefetch` is x86 specific.

## 3. Synergy & Value Design
- **SW Prefetching**: Can boost memory-bound kernels by 10-30% on consumer CPUs (Ryzen/Core) by keeping the memory bus saturated.
- **Verification**: Must verify against the `4.7ms` baseline. If it gets slower, we revert.

## 4. Verification Strategy
- **Benchmark**: Reuse `bench_cpu_kernel.rs`.
- **Logic**: Add `_mm_prefetch` to the AVX2 inner loop.

## 5. Roadmap
### Step 1: Analyze Prefetch Effect
- Modify `compute_row_avx2` to include `_mm_prefetch`.
- Test different strides (e.g., predicted 256 bytes ahead).

### Step 2: Hybrid Stream (Optional)
- If we were doing GPU offloading, we would pre-load the next GPU layer. But currently we focus on CPU kernel speed.

## Conclusion
We will interpret "Async Pipelining" for M=1 as **"Software Prefetching (L1/L2 Pipelining)"**.
This adheres to the "Double Buffering" concept (L1 Buffer vs RAM) without breaking causal dependency.
