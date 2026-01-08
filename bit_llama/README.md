# Bit-Llama Construction Report (Phase 13)

## 📌 プロジェクト概要
**「Bit-Llama」** は、Bit-TTTエンジン（1.58bit量子化 + TTTレイヤー）を多層化・スケールアップさせ、言語モデルとしての能力を持たせたプロトタイプです。
TinyStoriesデータセットを用いて、「物語を語る能力」の獲得を目指しています。

---

## 📂 成果物ファイル一覧

### 1. **コアエンジン (脳の構造)**
*   **ファイル**: `src/core_engine.rs`
*   **進化点**:
    *   `BitLlama` 構造体の実装（Embedding -> N層 -> Head）。
    *   `TTTLayer` のバッチ処理対応（`B, T, D`）。
    *   `RMSNorm` と `SwiGLU` (MLP) の実装による表現力向上。
    *   残差接続（Residual Connections）の導入。
    *   **注意**: 本クレートは `../rust_engine` のコアライブラリを参照しています。フォルダ構成を変更せず、 `Bit-TTT` フォルダごとお使いください。

### 2. **データ準備 (教材)**
*   **ファイル**: `data_prep/prepare_tinystories.py`
*   **機能**:
    *   TinyStoriesデータセットの自動ダウンロード。
    *   専用BPEトークナイザの学習（語彙数 16,384）。
    *   Rustでの高速読み込み用に `u16` バイナリ形式 (`train.bin`) へ変換。

### 3. **学習スクリプト (脳の育成)**
*   **ファイル**: `src/bin/train_llama.rs`
*   **機能**:
    *   **GPUバッチ学習**: `BATCH_SIZE=32` で8GB VRAMに最適化。
    *   **チェックポイント機能**: 10ステップごとに自動保存し、中断・再開が可能。
    *   **高速化**: コンテキスト長を128に調整し、回転率を向上。
*   **コマンド**:
    ```cmd
    cargo run --release --features cuda --bin train_llama
    ```

### 4. **推論スクリプト (おしゃべり)**
*   **ファイル**: `src/bin/inference_llama.rs`
*   **機能**: 学習済み重みを読み込み、対話形式でテキスト生成を行います。

---

## 🛠️ How to Run (実行手順)

```bash
# 1. Data Prep (教材の準備)
# Pythonライブラリのインストール
pip install -r requirements.txt

# TinyStoriesのダウンロードとトークナイザ作成
cd data_prep
python prepare_tinystories.py
cd ..

# 2. Train (学習)
# [GPU (NVIDIA) の場合]
cargo run --release --features cuda --bin train_llama

# [CPU (Mac/Intel/AMD) の場合]
# --features cuda を外すとCPUモードで動作します（遅いですが動作します）
cargo run --release --bin train_llama

# 3. Inference (推論)
# 学習したモデルと会話します
cargo run --release --bin inference_llama
```

## 🔌 Python Integration (外部連携)
Bit-TTT Engine は Python から DLL (`.so`/`.dylib`) として直接呼び出し可能です。

### Minimal Example
```python
import ctypes
import platform

# 1. Load Library
lib_name = "Bit_TTT.dll" if platform.system() == "Windows" else "libBit_TTT.so"
lib = ctypes.CDLL(f"./target/release/{lib_name}")

# 2. Define API
lib.ttt_create.argtypes = [ctypes.c_size_t, ctypes.c_float]
lib.ttt_create.restype = ctypes.c_void_p
lib.ttt_forward.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.POINTER(ctypes.c_float)]

# 3. Run
dim = 64
model = lib.ttt_create(dim, 0.1)
# ... forward pass ...
```
> ※ 詳細な仕様は `release/benchmark.py` を参照してください。

## ⚡ Performance (性能)

Bit-TTT (Core Engine) は、Rust + SIMD最適化により極めて高速に動作します。

| Metric | Value | Note |
|---|---|---|
| **Inference Speed** | **~34,000 TPS** | CPU Single Thread (Ryzen/Intel) |
| **Memory Footprint** | **Extremely Low** | 1.58bit quantization ready |
| **Startup Time** | **< 10ms** | No heavy runtime loaded |

> **Benchmark**: `python release/benchmark.py` で手元の環境のスコアを計測できます。

## 🧠 Model Specs (モデル仕様)

| Item | Specification | Note |
|---|---|---|
| **Architecture** | **Stack-Bit-TTT** | 1.58-bit BitNet + TTT (Test-Time Training) |
| **Components** | RMSNorm / SwiGLU / Residual | Modern Llama-like blocks |
| **Size** | **~11.7 M Params** | TinyStories Specialized (D=256, L=4) |
| **Quantization** | **1.58-bit** (Ternary) | Weights are `{-1, 0, 1}` |
| **Training** | **Hybrid (GPU/CPU)** | Train on CUDA, Infer on CPU |

---

## ⚙️ システム設定 (Cargo.toml)
*   **CUDA機能の切り替え**:
    *   学習時は `--features cuda` を付けることでGPUを有効化。
    *   推論時は指定なしでCPUモード（コンパイルエラー回避）。
*   **依存ライブラリ**: `tokenizers` を v0.22 に更新し、Python側との互換性を確保。

---

## 📊 現状のステータス
*   **学習進捗**: Step 150 / 1000
*   **Loss**: 4.15 付近
*   **能力**:
    *   単語の羅列から、「文章らしきもの」へ進化中。
    *   入力: `Once upon a time`
    *   出力: `"I'm glad you can..."` など、意味のあるフレーズが出現。

## 🚀 次のステップ
1.  **学習の完走**: Step 1000まで回し、Loss 3.0以下を目指す。
2.  **デスクトップアプリへの移植**: この「脳」をAliceに移植する（Phase 13 Step 5）。
