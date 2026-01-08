# Bit-TTT Engine: High-Performance Brain Core
**1.58-bit Quantization + Test-Time Training (TTT)** Implementation in Rust.

[Japanese / 日本語](#japanese) below.

---

<a name="english"></a>
# 🇬🇧 English: Bit-TTT Engine

## Overview
**Bit-TTT Engine** is a high-performance implementation of the Bit-TTT architecture. It combines **1.58-bit quantization efficiency** with **Test-Time Training (TTT)** adaptability. It runs entirely on the CPU using optimized integer arithmetic and SIMD/AVX instructions, achieving extreme throughput (**30,000+ TPS**).

📘 **[Read the Architecture Design](ARCHITECTURE.md)** to understand the core philosophy.



## Features
*   **Ultra Fast**: Optimizes matrix operations using `i8` integers and AVX2/AVX-512 instructions.
*   **Adaptive Memory**: Updates its internal state in real-time for every input token (online learning).
*   **Portable**: Distributed as a standard generic DLL/Shared Library (`release/Bit_TTT.dll`), usable from Python, C#, Unity, C++, etc.
*   **Safe**: Safe C-ABI with error codes and documented safety contracts.

## Project Components
- **[`bit_llama/`](bit_llama/)**: (New!) Pure Rust implementation of "Bit-Llama" (Stacked Bit-TTT). Supports GPU training and TinyStories generation.
- **[`rust_engine/`](rust_engine/)**: Core logic optimized for C-ABI (DLL generation).
- **[`examples/`](examples/)**: Minimal usage examples (Python etc).
- **[`python_proto/`](python_proto/)**: Original Python prototype for research.

## Quick Start (Python)

> **Want to train an LLM?**  
> Go to **[`bit_llama/README.md`](bit_llama/README.md)** for instructions on training "Bit-Llama" on TinyStories.

To try the Core Engine directly via Python C-API:

```bash
# Verify behavior and speed
python examples/python_inference.py
```

Expected Output:
```text
Running Inference on 10 tokens...
Done in 0.0003 sec.
Output Shape: 640 floats
Success! w_state has been updated internally.
```

(For detailed benchmarking, run `python release/benchmark.py`)

## Developer Guide (C-ABI)
For integration with C, C++, or C# (Unity), use the exported functions:

### Error Codes
| Code | Name | Description |
|---|---|---|
| **0** | `Ok` | Success |
| **1** | `NullPointer` | Input pointer was null |
| **2** | `DimensionMismatch` | Input array size validation failed |
| **99** | `Panic` | Internal Rust panic caught |

### API Signature
```c
// Create Model: returns ptr or NULL
void* ttt_create(size_t hidden_dim, float inner_lr);

// Forward + Update: returns error code (0 = Ok)
int ttt_forward(void* model, const float* input, size_t seq_len, float* output);

// Destroy Model
void ttt_destroy(void* model);
```

---

<a name="japanese"></a>
# 🇯🇵 日本語: Bit-TTT 脳エンジン

## 概要
**Bit-TTT Engine** は、Bit-TTTアーキテクチャの高性能実装版です。**1.58bit量子化による効率性**と、**Test-Time Training (推論時学習) による適応性**を兼ね備えています。
完全にCPU上で動作し、SIMD/AVX命令を駆使した整数演算により、一般的なPCで **30,000+ TPS (トークン/秒)** という驚異的な推論速度を実現します。

📘 **[アーキテクチャ設計書 (日本語)](ARCHITECTURE_JA.md)** も参照してください。



## 特徴
*   **爆速**: `i8` 整数演算とAVX2/AVX-512命令セットにより最適化されています。
*   **学習する記憶**: 入力トークンを受け取るたびに、内部のニューラルネットをリアルタイムで更新（学習）します。
*   **ポータブル**: 汎用的な DLL (`release/Bit_TTT.dll`) として提供されるため、Python, Unity (C#), C++, Node.js などあらゆる環境から利用可能です。
*   **安全**: エラーコードによる例外制御と、明確な安全性保証を備えています。

## プロジェクト構成
- **[`bit_llama/`](bit_llama/)**: (New!) "Bit-Llama" (多層化Bit-TTT) のPure Rust実装。GPU学習とTinyStories生成に対応しています。
- **[`rust_engine/`](rust_engine/)**: C-ABI (DLL生成) に最適化されたコアロジックです。
- **[`examples/`](examples/)**: Python等からの最小利用例です。
- **[`python_proto/`](python_proto/)**: 研究用の初期Pythonプロトタイプです。

## クイックスタート (Python)

> **LLMを学習させたい場合**  
> **[`bit_llama/README.md`](bit_llama/README.md)** をご覧ください。「Bit-Llama」の学習手順（TinyStories使用）を詳述しています。

Core Engine (C-API) の動作を試すには：

```bash
python examples/python_inference.py
```

実行結果の例:
```text
Running Inference on 10 tokens...
Done in 0.0003 sec.
Output Shape: 640 floats
Success! w_state has been updated internally.
```

(詳細なベンチマーク測定は `python release/benchmark.py` を実行してください)

## 開発者ガイド (C-ABI)
C言語、C++、C# (Unity) などから利用する場合は、以下の関数を使用します。

### エラーコード
| Code | Name | Description |
|---|---|---|
| **0** | `Ok` | 成功 |
| **1** | `NullPointer` | ポインタが null |
| **2** | `DimensionMismatch` | 配列サイズ不正 |
| **99** | `Panic` | 内部パニック発生 |

### API シグネチャ
```c
// モデル生成
void* ttt_create(size_t hidden_dim, float inner_lr);

// 推論実行 (戻り値 0 = 成功)
int ttt_forward(void* model, const float* input, size_t seq_len, float* output);

// モデル破棄
void ttt_destroy(void* model);
```

---
*Created by Project Bit-TTT.*

