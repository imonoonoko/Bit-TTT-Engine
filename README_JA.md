# Bit-TTT 脳エンジン: 高性能AIコア

**1.58-bit 量子化 + Test-Time Training (TTT)** のRust実装です。
次世代の効率的で適応力のあるAIモデルを支えるコアエンジンです。

[English](README.md) (英語はこちら)

---

# 🇯🇵 日本語: Bit-TTT Engine

## 概要
**Bit-TTT Engine** は、Bit-TTTアーキテクチャの高性能実装版です。**1.58bit量子化による効率性**と、**Test-Time Training (推論時学習) による適応性**を兼ね備えています。テンソル演算には **Candle** フレームワークを使用し、**PyO3** を通じてPythonとシームレスに統合されます。

📘 **[アーキテクチャ設計書](ARCHITECTURE_JA.md)** も参照してください。

## 特徴
*   **Rust-First & Python-Compatible**: 高速なRustコアロジックを PyO3 経由でPythonから利用可能。
*   **Zero-Copy Inference**: 非効率なデータコピーを排除し、高速な推論を実現。
*   **Device Support**: **CPU** (AVX最適化) および **CUDA** (GPU) での実行をサポート。
*   **Pure Rust Mode**: Python依存なしでコンパイル可能 (`--no-default-features`)。組み込み用途に最適。
*   **Safe**: Rustの安全性保証に厳密に準拠。

## プロジェクト構成

- **[`rust_engine/`](rust_engine/)**: コア実装。
    - `core_engine.rs`: Candleベースのニューラルネットロジック。
    - `lib.rs`: PyO3 バインディング (`cortex_rust` モジュール)。
    - `legacy/`: 旧来の ndarray/C-API コード（互換性のため分離）。
- **[`bit_llama/`](bit_llama/)**: 学習・推論用のスタンドアロンRustバイナリ。

## クイックスタート (Python)

### 1. ビルドとインストール
`maturin` を使用して Python wheel をビルドします。

```bash
cd rust_engine
maturin develop --release
```

### 2. 使い方
```python
import cortex_rust

# 設定
config = cortex_rust.BitLlamaConfig(
    vocab_size=1000,
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
