# Dependency Map (Phase 15)

## Class Hierarchy & Memory Ownership

```mermaid
graph TD
    User["User / API"] -->|Hold| Llama["Llama (Wrapper)"]
    Llama -->|Owns| Dev["Device (GPU/CPU)"]
    Llama -->|Owns| M["BitLlama (Model)"]

    M -->|Vec| Layers["Vec<BitLlamaBlock>"]

    subgraph "Current (Eager Load)"
        Layers -->|Layer 0| B0["Block 0 (GPU)"]
        Layers -->|Layer N| BN["Block N (GPU)"]
        B0 -->|Tensor| W0["Weights (VRAM)"]
        BN -->|Tensor| WN["Weights (VRAM)"]
    end

    subgraph "Target (Streaming)"
        M_Stream["StreamingBitLlama"] -->|Vec| MixedLayers["Vec<LayerState>"]
        MixedLayers -->|0..K| GPU_L["GPU Layers (VRAM)"]
        MixedLayers -->|K..N| CPU_L["CPU Layers (RAM / Mmap)"]
    end
```

## Critical Paths
1.  `Llama::new_with_weights` -> `BitLlama::load`
    - Must intercept here to allow partial loading.
2.  `BitLlama::forward_one`
    - Currently iterates cleanly.
    - Needs to inject "Move to GPU" logic before `layer.forward()`.

## Risks
- **Tensor cloning**: `tensor.to_device(gpu)` creates a copy. Code must perform this efficiently.
- **Latency**: Doing this per-token for every layer is extremely slow (PCIe bottleneck).
    - *Mitigation*: Current focus is "Make it Run" (Capacity), not Speed.
