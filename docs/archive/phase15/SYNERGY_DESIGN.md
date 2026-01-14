# Synergy & Value Design (Phase 15)

## 1. Zero-Copy Synergy w/ OS Paging
- **Concept**: Instead of manually managing a "Cache", rely on `memmap2` and the OS Page Cache.
- **Mechanism**:
    - Build `Tensor` from raw raw pointers (unsafe) that point to the mmap region.
    - Passing this CPU tensor to `layer.forward` on GPU triggers an implicit copy?
    - **Optimization**: If we use `candle`'s `from_mmaped_safetensors` properly with `Device::Cpu`, the tensors live in Virtual Memory (backed by disk).
    - When `tensor.to_device(cuda)` is called, the OS pages in *only* that tensor's bytes from disk (if not in RAM).
    - Unused layers can be "evicted" by the OS if RAM pressure is high (standard paging).
- **Benefit**: Minimum code complexity. We don't implement an LRU cache; Windows/Linux does it for us.

## 2. Browser/WASM Synergy (Future Phase 16)
- The logic for "Streaming Load" is nearly identical to "HTTP Range Requests" for browser inference.
- Implementing an abstract `WeightLoader` trait now will allow swapping `MmapLoader` with `HttpLoader` later without changing the model logic.

## 3. UI Visualization
- The GUI VRAM Monitor (Phase C) becomes critical here.
- Visualizing "VRAM Usage" vs "System RAM Usage" provides real-time feedback on offloading performance.
