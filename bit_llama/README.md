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
(詳細は RootのREADMEを参照してください)

---

## 💎 Pre-trained Models (配布計画)

現在、以下のモデルの学習と公開を計画しています。

| Model Name | Specs | Training Data | Status | Download |
|---|---|---|---|---|
| **Bit-Llama-Micro** | ~11M Params, 1.58bit | TinyStories (Mini) | 🟡 **Training** | *Coming Soon* |
| **Bit-Llama-Code** | ~100M Params, 1.58bit | Python Code Snippets | ⚪ Planned | - |

> **Note**: 学習済み重み (`.safetensors`) は Hugging Face Hub での公開を予定しています。

---

## 📊 現状のステータス (Metrics)
*   **Training Speed**: ~800 tokens/sec (RTX 4060 Ti)
*   **Loss Curve**: Smooth convergence observed at Step 150 (Loss: 4.15).
*   **Generation**: "Always" -> "Alice" -> "Alice was very tired..." (Context learning observed).

## 🚀 Future Roadmap
1.  **Distributed Training**: Implement Data Parallelism for multi-GPU training.
2.  **Hugging Face Integration**: Provide `from_pretrained("bit-ttt/llama-11m")` API.
3.  **Desktop App**: Integrate into "Alice" desktop assistant (Phase 13 Step 5).
