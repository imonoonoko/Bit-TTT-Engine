# Bit-Llama Construction Report (Phase 13)

## 📌 プロジェクト概要
(Work In Progress)


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

## 🛠️ Development Guide (Experimental)

The following steps are for **active development and debugging only**.

### 1. Requirements
*   Python 3.x
*   Rust Toolchain (Cargo)
*   CUDA Toolkit (Optional, for GPU training)

### 2. Run Training (Dev Mode)
```bash
# GPU Mode (Requires NVIDIA GPU)
cargo run --release --features cuda --bin train_llama

# CPU Mode
cargo run --release --bin train_llama
```

> **Warning**: This is a prototype implementation. Parameters and data formats may change.

