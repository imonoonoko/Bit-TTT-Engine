# Bit-TTT Engine: High-Performance Brain Core
**Bitwise Test-Time Training (Bit-TTT)** Implementation in Rust.

[Japanese / 日本語](#japanese) below.

---

<a name="english"></a>
# 🇬🇧 English: Bit-TTT Engine

## Overview
**Cortex Rust Engine** is a high-performance implementation of the Bit-TTT architecture. It combines **1.58-bit quantization efficiency** with **Test-Time Training (TTT)** adaptability. It runs entirely on the CPU using optimized integer arithmetic and SIMD/AVX instructions, achieving extreme throughput (~60,000 TPS).

## Features
*   **Ultra Fast**: Optimizes matrix operations using `i8` integers and AVX2/AVX-512 instructions.
*   **Adaptive Memory**: Updates its internal state in real-time for every input token (online learning).
*   **Portable**: Distributed as a standard generic DLL/Shared Library (`release/Bit_TTT.dll`), usable from Python, C#, Unity, C++, etc.

## Contents
*   `release/Bit_TTT.dll`: The core engine library.
*   `release/benchmark.py`: Python script for verification and benchmarking.
*   `release/BIT_TTT_SPEC.md`: Technical specification document.

## Quick Start (Python)

### Requirements
*   Python 3.x
*   (Optional) `numpy`

### Running the Benchmark
You can verify the memory effect and speed immediately by running the included script:

```bash
python release/benchmark.py
```

Expected Output:
```text
--- 1. Verification Test (Memory Effect) ---
✅ [SUCCESS] Model state evolved (Delta increased from 0).

--- 2. Speed Benchmark (Rust Engine) ---
⚡ Speed: 60774.68 Tokens/Sec (TPS)
```

## Developer Guide (C-ABI)
For integration with C, C++, or C# (Unity), use the exported functions:

```c
// Create Model
void* ttt_create(size_t hidden_dim, float inner_lr);

// Forward Pass (Inference + Training)
void ttt_forward(void* model, const float* input, size_t seq_len, float* output);

// Destroy Model
void ttt_destroy(void* model);
```

---

<a name="japanese"></a>
# 🇯🇵 日本語: Bit-TTT 脳エンジン

## 概要
**Cortex Rust Engine** は、Bit-TTTアーキテクチャの高性能実装版です。**1.58bit量子化による効率性**と、**Test-Time Training (推論時学習) による適応性**を兼ね備えています。
完全にCPU上で動作し、SIMD/AVX命令を駆使した整数演算により、一般的なPCで **約60,000 TPS (トークン/秒)** という驚異的な推論速度を実現します。

## 特徴
*   **爆速**: `i8` 整数演算とAVX2/AVX-512命令セットにより最適化されています。
*   **学習する記憶**: 入力トークンを受け取るたびに、内部のニューラルネットをリアルタイムで更新（学習）します。
*   **ポータブル**: 汎用的な DLL (`release/Bit_TTT.dll`) として提供されるため、Python, Unity (C#), C++, Node.js などあらゆる環境から利用可能です。

## 同梱物 (release/ フォルダ内)
*   `release/Bit_TTT.dll`: エンジン本体。
*   `release/benchmark.py`: 動作確認およびベンチマーク用スクリプト。
*   `release/BIT_TTT_SPEC.md`: 技術仕様書。

## クイックスタート (Python)

### 必要なもの
*   Python 3.x

### ベンチマークの実行
同梱のスクリプトを実行するだけで、記憶能力の検証と速度計測を行えます。

```bash
python release/benchmark.py
```

実行結果の例:
```text
--- 1. Verification Test (Memory Effect) ---
✅ [SUCCESS] Model state evolved (Delta increased from 0).
(モデルの状態が変化し、学習が行われていることを確認)

--- 2. Speed Benchmark (Rust Engine) ---
⚡ Speed: 60774.68 Tokens/Sec (TPS)
(毎秒約6万トークンという超高速動作)
```

## 開発者ガイド (C-ABI)
C言語、C++、C# (Unity) などから利用する場合は、以下の関数をインポートしてください。

```c
// モデル生成: hidden_dim(次元数), inner_lr(学習率)を指定
void* ttt_create(size_t hidden_dim, float inner_lr);

// 推論実行: 入力配列を渡し、出力配列に結果を受け取る（同時に学習も行われる）
void ttt_forward(void* model, const float* input, size_t seq_len, float* output);

// モデル破棄: メモリリーク防止のため使用後に必ず呼ぶ
void ttt_destroy(void* model);
```

---
*Created by Project Bit-TTT.*
