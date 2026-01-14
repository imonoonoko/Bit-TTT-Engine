# Analysis Report: Phase 15 (Scaling Architecture)

## 1. Executive Summary
本レポートは、**Phase 15: 大規模化への挑戦 (Scaling to 7B/70B)** の実現に向けた構造分析および設計指針である。
Bit-Llama 70B は、1.58bit量子化を行っても約14GB〜16GBのモデルサイズになることが予想される。
VRAM 8GB〜16GB のコンシューマGPUでこれを動作させるには、既存の「全レイヤーを一括でGPUにロードする」方式ではメモリ不足となる。
解決策として、**Layer-wise Streaming Load** および **CPU offloading** のアーキテクチャが必要である。

## 2. As-Is Architecture Analysis
### 2.1 Model Loading
*   **Current State**: `BitLlama::load` で全レイヤー (`layers: Vec<BitLlamaBlock>`) の重みを一度に `VarBuilder` から読み込み、GPUメモリ上に `Tensor` として確保している。
*   **Bottleneck**: 70Bモデルの場合、この処理で即座にVRAMが枯渇する。

### 2.2 Inference Loop
*   **Current State**: `forward_one` メソッド内で `for layer in self.layers` ループを回し、すべての層がGPU上にある前提で計算している。
*   **Bottleneck**: オフロードされた層を動的にGPUに戻す仕組みがない。

## 3. To-Be Architecture Definition
### 3.1 Streaming Layer Loading (mmap-based)
*   モデルファイル全体を `mmap` し、推論ループの中で「必要なレイヤーの重みだけ」をGPUに転送し、計算後に破棄（またはCPUへ退避）する方式へ変更。
*   Rustの `candle` は `safetensors` のmmapをサポートしているため、これを活用する。

### 3.2 Hybrid Device Management
*   **Layer State**: 各レイヤーが `Loaded(GPU)`, `Offloaded(CPU)`, `Unloaded(Disk)` のいずれかの状態を持つように抽象化する。
*   **Pipeline**:
    1.  Layer N: Forward実行 (GPU)
    2.  [Background] Layer N+1: Prefetch to GPU
    3.  [Background] Layer N-1: Offload/Drop

## 4. Feasibility
*   **PCIe Bandwidth**: PCIe Gen4 x16 (32GB/s) であっても、推論ごとに数GBを転送するのは速度低下が著しい（数token/s程度になる可能性）。
*   **Realism**: 70Bを8GBで動かす場合、ディスクIOまたはCPU-GPU転送が律速となる。実用的な速度のためには、「GPUに収まる限界までロードし、残りをCPU」とする **Partial Offloading** が現実的解。

## 5. Derived Tasks
1.  **[Core]** `BitLlamaBlock` をラップする `StreamingBlock` 構造体の設計。
2.  **[Core]** `Device` マッピングを指定可能なロード関数の実装 (`load_with_map`)。
3.  **[Infra]** Layerごとのメモリ使用量見積もりロジック。
