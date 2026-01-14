# Synergy & Value Design: Phase 15

## 1. Efficiency Optimization
### 1.1 `mmap` Reuse
Phase 14でも言及した `memmap2` によるデータローディング技術は、モデルの重みロードにもそのまま適用可能（実は既に `VarBuilder::from_mmaped_safetensors` で使われている）。
*   **Extend**: これを「全ロード」ではなく「Viewのみ保持し、必要な時だけ実体化（Tensor化）」するラッパーに拡張する。

### 1.2 Unified Device Map
*   HuggingFace Transformers (Python) の `device_map="auto"` と同様の体験を Rust で提供。
*   **Value**: ユーザーは VRAM 容量を気にする必要がなくなり、「空いている分だけGPU、残りはCPU/Disk」という柔軟な運用が可能になる。

## 2. TTT Synergy
*   **TTT State on GPU**:
    *   TTTの核心である `w_state` (隠れ状態) はサイズが小さく (D_small * D_small)、常に変化するため、これは **常時GPUに置く** ほうが効率的。
    *   **Architecture**:
        *   Weights (Static): Disk <-> CPU <-> GPU (Swappable)
        *   Hidden State (Dynamic): GPU (Resident)
    *   これにより、TTTの高速更新というメリットを殺さずに、大規模モデルを扱える。
