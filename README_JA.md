# Bit-TTT 脳エンジン: 高性能AIコア

**1.58-bit 量子化 + Test-Time Training (TTT)** のRust実装です。
次世代の効率的で適応力のあるAIモデルを支えるコアエンジンです。

[English](README.md) (英語はこちら)

---

# 🇯🇵 日本語: Bit-TTT Engine

## 概要
**Bit-TTT Engine** は、Bit-TTTアーキテクチャの高性能実装版です。**1.58bit量子化による効率性**と、**Test-Time Training (推論時学習) による適応性**を兼ね備えています。テンソル演算には **Candle** フレームワークを使用し、**完全なRust環境**で学習から推論までを実行できます（Pythonとの連携もオプションとしてサポート）。

📘 **[アーキテクチャ設計書](ARCHITECTURE_JA.md)** も参照してください。

## 特徴
*   **End-to-End Rust Pipeline (NEW!)**: データ処理、学習、推論のすべてを Rust のみで完結。Python は不要です。
*   **Rust-First & Python-Compatible**: 高速なRustコアロジックを PyO3 経由でPythonから利用可能。
*   **Zero-Copy Inference**: 非効率なデータコピーを排除し、高速な推論を実現。
*   **Device Support**: **CPU** (AVX最適化) および **CUDA** (GPU) での実行をサポート。
*   **Pure Rust Mode**: Python依存なしでコンパイル可能。組み込み用途に最適。
*   **Safe**: Rustの安全性保証に厳密に準拠。

## アーキテクチャ: Pure Rust エコシステム

```mermaid
flowchart LR
    A[Text Data] -->|"Rust Tokenizer"| B(Token IDs)
    B -->|"train_llama (Rust)"| W[(Weights)]
    W -->|"bit_llama (Rust)"| D[Fast Inference]
    
    subgraph "Core Engine (cortex_rust)"
        direction TB
        L[Layers]
        M[BitLinear]
        T[Tokenizers]
    end
    
    B -.-> M
```

## プロジェクト構成

- **[`crates/rust_engine/`](crates/rust_engine/)**: コア実装 (`cortex_rust`)。
    - `core_engine.rs`: Candleベースのニューラルネットロジック。
    - `lib.rs`: 公開 API 定義。
- **[`crates/bit_llama/`](crates/bit_llama/)**: 学習・推論用のスタンドアロンRustバイナリ。

## クイックスタート (Pure Rust CLI) 🚀

Pythonを一切使わずに、学習から推論までを実行できます！

### 1. ビルド
```bash
# 便利な起動スクリプトを使用 (推奨)
./launch_trainer.bat

# 手動ビルド
cd crates/bit_llama
cargo build --release --features cuda
```

### 2. 学習 (train_llama)
`cortex_rust` エンジンを使用してゼロからモデルを学習します。CLI引数でハイパーパラメータを調整可能です。

```bash
# 例: 学習率・ステップ数・データパスを指定して実行
cargo run --release --features cuda --bin train_llama -- --lr 0.001 --steps 10000 --data data/TinyStories/train.bin
```
*出力: `bit_llama_checkpoint.safetensors`*

> [!TIP]
> チェックポイントから学習を再開する場合は、Lossの急増（リバウンド）を防ぐために学習率を下げて（例: `--lr 5e-5`）実行することを推奨します。

### 3. 推論 (bit_llama)
高性能なストリーミング生成を実行します。

```bash
# 便利な起動スクリプトを使用 (推奨)
./launch_chat.bat

# 手動実行
# config.json, tokenizer.json, model.safetensors があるディレクトリを指定 (相対パスに注意)
../../target/release/bit_llama --model ../../models/dummy --prompt "Hello Rust AI" --temp 0.8 --max-tokens 100
```
*パフォーマンス: ~1100 tokens/sec (CPU, ダミーモデル)*

## クイックスタート (Python)

### 1. ビルドとインストール
`maturin` を使用して Python wheel をビルドします。

```bash
```bash
cd crates/rust_engine
maturin develop --release
```

### 2. 使い方
```python
import cortex_rust

# 設定
config = cortex_rust.BitLlamaConfig(
    vocab_size=50257,
    hidden_dim=256,
    num_layers=4,
    inner_lr=0.01
)

# モデル読み込み (デバイス指定: "cpu" または "cuda")
model = cortex_rust.BitLlama(config, "path/to/model.safetensors", device="cuda")

# 推論実行 (トークンID列)
tokens = [1, 50, 100]
logits = model.forward(tokens)
print(logits)
```

## 高度なビルドオプション

### Pure Rust Binary (Python依存なし)
Python連携を行わず、軽量なRust単体バイナリとしてビルドする場合：

```bash
cargo build --release --no-default-features
```
(`Cargo.toml` の `python` 機能を無効化します)

### デバイス選択
`PyBitLlama` のコンストラクタでデバイスを指定できます：
- `device="cpu"` (省略時のデフォルト)
- `device="cuda"` (CUDA環境が必要)

---
*Created by Project Bit-TTT.*
