# Bit-TTT Engine: 高性能ブレインコア

[![Featured on Orynth](https://orynth.dev/api/badge/bit-ttt-engine?theme=dark&style=default)](https://orynth.dev/projects/bit-ttt-engine)

On-chain data powered by
[![GeckoTerminal](image.png)](https://www.geckoterminal.com)
[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Build Status](https://github.com/imonoonoko/Bit-TTT-Engine/actions/workflows/rust.yml/badge.svg)](https://github.com/imonoonoko/Bit-TTT-Engine/actions)

**1.58-bit 量子化 + テスト時学習 (TTT)** を Pure Rust で実装。

[English / 英語](README.md)

---

## ✨ Bit-TTT とは？

**Bit-TTT Engine** は2つの最先端技術を組み合わせたエンジンです：
- **BitNet 1.58-bit 量子化**: 三値重み {-1, 0, +1} による極限圧縮
- **Test-Time Training (TTT)**: オンライン学習による適応的アテンション代替

目標: **70B パラメータモデルを 8-16GB VRAM で実行**

## 📊 現在のステータス (2026年1月)

| 機能 | 状態 | 説明 |
|------|------|------|
| コアエンジン (`cortex_rust`) | ✅ 完了 | Candle ベースのニューラルネットワーク実装 |
| 学習パイプライン | ✅ 完了 | Pure Rust でのエンドツーエンド学習 |
| ストリーミング推論 | ✅ 完了 | CPU で約 1100 トークン/秒 |
| GUI トレーナー | ✅ 完了 | Tauri ベースのビジュアル学習インターフェース |
| Python バインディング (PyO3) | ✅ 完了 | オプションの Python 統合 |
| 日本語トークナイザー | ✅ 完了 | Phase 14 (Unigram) |
| 7B/70B スケーリング | ✅ 完了 | Phase 15 (Auto-Config & AVX2) |
| WASM/ブラウザ対応 | 🚧 計画中 | Phase 16 (Future) |

## 🏗️ アーキテクチャ

```
Bit-TTT Engine
├── crates/
│   ├── rust_engine/         # コアライブラリ (cortex_rust)
│   │   ├── layers/          # ニューラルネットワーク層
│   │   │   ├── rms_norm.rs    # RMS 正規化
│   │   │   ├── bit_linear.rs  # 1.58-bit 線形層
│   │   │   ├── swiglu.rs      # SwiGLU MLP
│   │   │   └── ttt.rs         # TTT 層
│   │   ├── model/           # モデルアーキテクチャ
│   │   │   ├── block.rs       # Transformer ブロック
│   │   │   ├── llama.rs       # BitLlama モデル
│   │   │   └── config.rs      # 設定
│   │   ├── python.rs        # PyO3 バインディング
│   │   └── lib.rs           # 公開 API
│   │
│   └── bit_llama/           # CLI アプリケーション
│       ├── train/           # 学習モジュール
│       │   ├── args.rs        # CLI 引数
│       │   ├── checkpoint.rs  # 状態管理
│       │   └── training_loop.rs  # メインループ
│       ├── gui/             # Tauri GUI
│       └── inference.rs     # 推論エンジン
│
├── models/                  # 学習済みモデルチェックポイント
├── data/                    # 学習データセット
└── tools/                   # ユーティリティスクリプト
```

## 🚀 クイックスタート

### 必要条件
- Rust 1.70+
- (オプション) CUDA 11.8+ (GPU アクセラレーション用)

### 1. ビルド
```bash
git clone https://github.com/imonoonoko/Bit-TTT-Engine.git
cd Bit-TTT-Engine
cargo build --release
```

### 2. 学習
```bash
# ランチスクリプトを使用 (Windows)
./launch_trainer.bat

# 手動で学習
cargo run --release --bin train_llama -- \
    --data data/TinyStories \
    --dim 256 \
    --layers 8 \
    --steps 10000 \
    --lr 3e-4
```

### 3. 推論
```bash
# ランチスクリプトを使用 (Windows)
./launch_chat.bat

# 手動で推論
cargo run --release --bin bit_llama -- \
    --model models/my_model \
    --prompt "こんにちは Bit-TTT!" \
    --max-tokens 100 \
    --temp 0.8
```

## 📖 ドキュメント

| ドキュメント | 説明 |
|-------------|------|
| [ARCHITECTURE_JA.md](ARCHITECTURE_JA.md) | システム設計哲学 |
| [ROADMAP.md](ROADMAP.md) | 将来の開発計画 |
| [docs/SPECIFICATION_JA.md](docs/SPECIFICATION_JA.md) | 技術仕様書 |
| [docs/CONTRIBUTING_JA.md](docs/CONTRIBUTING_JA.md) | コントリビューションガイド |

## 🛠️ 開発コマンド

```bash
# 全テスト実行
cargo test --workspace

# コンパイルチェック
cargo check --workspace

# コードフォーマット
cargo fmt --all

# リンター実行
cargo clippy --workspace
```

## 🐍 Python 統合 (オプション)

```bash
cd crates/rust_engine
pip install maturin
maturin develop --release
```

```python
import cortex_rust

config = cortex_rust.BitLlamaConfig(
    vocab_size=16384,
    hidden_dim=256,
    num_layers=8,
    inner_lr=0.1
)

model = cortex_rust.BitLlama(config, "model.safetensors", device="cuda")
logits = model.forward(token_id=42)
```

## 💖 サポート

**Solana ウォレット**: `13ui3nmE7smmK3Pk8wyKb7RE6wHyMJCcWgCeMRRdoory`

---

*Created by Project Bit-TTT • MIT License*


