# Analysis Report: Model Loading & Memory Architecture
## 1. As-Is Architecture (現状)
### 1.1 Loading Mechanism
- `BitLlama::new_with_weights` uses `memmap2` to map the `.safetensors` file.
- It validates the custom `.bitt` header.
- **Challenge**: It calls `BitLlama::load` immediately, passing a `VarBuilder`.
- `BitLlama::load` iterates `0..num_layers` and instantiates every layer (`BitLinear`, `RMSNorm`, etc.).
- **Critical Issue**: `candle` layers typically load tensors into the specified `device` (GPU) upon construction.
    - For a 70B model, this tries to allocate ~140GB (fp16) or ~14GB (1.58bit quantified) *immediately* on the GPU.
    - Most consumer GPUs (8GB VRAM) will OOM instantly.

### 1.2 Weight Storage
- Structure: `BitLlama` -> `Vec<BitLlamaBlock>` -> `Layers`.
- All layers reside on the single `device` passed to `load`.
- No mechanism exists to split layers between CPU/GPU or keep them unloaded.

## 2. To-Be Architecture (あるべき姿)
### 2.1 Streaming Inference (Layer-wise Loading)
- **Concept**: Hold weights in System RAM (or mapped file), only move to GPU during `forward`.
- **Implementation**:
    - `BitLlama` should allow layers to reside on different devices.
    - Or, a new `StreamingBitLlama` struct that performs transfer-compute-evict cycles.

### 2.2 Device Map (Hybrid Offloading)
- **Concept**: Usage of VRAM as a "Cache" for the first N layers.
- **Goal**:
    - Layers 0-10: Persistent on GPU (Fast)
    - Layers 11-40: CPU RAM -> GPU Stream (Slower but works)

## 3. Impact Assessment
- `crates/rust_engine/src/model/llama.rs`: Major refactoring needed.
- `crates/rust_engine/src/layers/`: May need `to_device()` method if not present (Candle usually has this).
- Performance: Streaming will introduce PCI-e bottleneck. Usage of pinned memory or async transfer is desirable but complex in Rust (w/ Candle).
