# Bit-TTT Engine アーキテクチャ

[![Featured on Orynth](https://orynth.dev/api/badge/bit-ttt-engine?theme=dark&style=default)](https://orynth.dev/projects/bit-ttt-engine)

## 1. Core Philosophy (設計思想)

**Bit-TTT** は、「超高効率な推論」と「適応的な学習」の融合を目指すプロジェクトです。
以下の2つの技術を統合しています：

1.  **1.58-bit 量子化**: パラメータを三値 `{-1, 0, 1}` にし、計算効率を極限まで高めます。
2.  **Test-Time Training (TTT)**: 静的なKVキャッシュの代わりに、文脈をその場で「学習」するFast Weightsを用います。

---

## 2. System Overview (システム構成)

**Rust-First, Python-Compatible** アーキテクチャを採用しています。

```mermaid
graph TD
    A["Python (PyO3)"] -->|Direct Bindings| B["Rust Core Engine"]
    B -->|Candle (SIMD/AVX)| C["CPU / GPU"]

    subgraph Rust Core
    D["BitLlama (Model)"]
    E["TTT Layer (Fast Weights)"]
    F["BitLinear (Ternary Weights)"]
    end

    B --> D
    D --> E
    D --> F
```

### コンポーネント詳細

### コンポーネント詳細

| Module | Role | Tech Stack |
|---|---|---|
| **crates/core_engine** | 推論・学習ロジック | **Candle** フレームワーク。CPU/CUDA両対応。 |
| **crates/cortex_rust** | Python インターフェース | **PyO3**。`BitLlama` クラスをPythonに直接公開。 |
| **legacy** | 旧実装（非推奨） | 古い `extern "C"` / `ndarray` 実装（互換性のため隔離）。 |

---

## 3. Data Flow (データフロー)

### 推論ステップ
1.  **Input**: PythonからトークンIDを受け取る。
2.  **Zero-Copy**: PyO3のバッファプロトコルにより、コピーなしでRustへデータ転送。
3.  **Forward Pass**:
    *   **Embedding**: ベクトル変換。
    *   **TTT Update**: 勾配降下法により `W_state` (短期記憶) を更新。
    *   **Projection**: 1.58bit 行列演算。
4.  **Output**: 計算結果(Logits)をPythonへ返す。

---

## 4. Safety & Build Options

*   **Type Safety**: Rustの型システムとCandleのStrictなシェイプ管理により安全性を担保。
*   **Pure Rust Build**: `--no-default-features` オプションでPython/PyO3依存を排除し、純粋なRustバイナリとしてビルド可能。
*   **Device Agnostic**: `cpu` (AVX) と `cuda` (GPU) を設定一つで切り替え可能。

## 5. Hybrid Inference Strategy (Phase 15 Architecture)

大規模モデル(70B+)をコンシューマ機で動かすため、**CPU/GPUハイブリッド推論** を実装しています。

*   **Layer Distribution**: `n_gpu_layers` 設定に基づき、モデルの層をGPUとCPUに分散配置します。
*   **Zero-Copy CPU Kernel**: AVX2最適化されたCPUカーネルは、事前にパッキングされた重みをメモリコピーなしで直接参照し、高速に計算します。
*   **Dynamic Dispatch**: `forward` パスにおいて、テンソルのデバイス位置に応じて自動的にカーネル（CUDA vs AVX2）を切り替えます。

これにより、VRAM溢れを防ぎつつ、CPUでも実用的な速度（~4 t/s @ 70B）を実現しています。
