# Dependency Map: Phase 15

```mermaid
graph TD
    subgraph "Core Engine (rust_engine)"
        Llama[Llama Implementation]
        StreamingBlock[StreamingBlock (Wrapper) - NEW]
        Block[BitLlamaBlock]
        Disk[Disk (Mmap)]
        GPU[GPU Memory]
        CPU[CPU Memory]
    end

    subgraph "External"
        Candle[Candle / Safetensors]
    end
    
    Llama -->|Manages| StreamingBlock
    StreamingBlock -->|Wraps| Block
    
    %% Data Flow
    Disk -->|Mmap Load| CPU
    CPU -->|Transfer| GPU
    GPU -->|Compute| Block
    GPU -.->|Offload| CPU
```

## Risk Assessment
*   **Latency**: Layerごとの転送オーバーヘッドが推論速度を著しく低下させるリスク。
    *   **Mitigation**: Prefetching（次のレイヤーを計算中に転送）の実装。
*   **Complexity**: 非同期（Async）でのデータ転送と同期（Sync）での計算のブリッジが複雑化する。`w_state` (TTTの隠れ状態) は常に最新を維持する必要があるため、状態管理と重み管理の分離が重要。
