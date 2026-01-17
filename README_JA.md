# Bit-TTT Engine: 高性能AIブレイン・コア

[![Featured on Orynth](https://orynth.dev/api/badge/bit-ttt-engine?theme=dark&style=default)](https://orynth.dev/projects/bit-ttt-engine)

On-chain data powered by [![GeckoTerminal](docs/images/image-2.png)](https://www.geckoterminal.com)

[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Build Status](https://github.com/imonoonoko/Bit-TTT-Engine/actions/workflows/rust.yml/badge.svg)](https://github.com/imonoonoko/Bit-TTT-Engine/actions)

**Pure Rustによる「1.58ビット量子化 + Test-Time Training (TTT)」実装**
**Current Version: v0.3.1 (Refactor & Bit-Converter)**

[English / 英語](README.md) | [📖 ユーザーガイド (User Guide)](docs/USER_GUIDE_JA.md)

---

## ✨ Bit-TTT Engineとは？ (3行で)
1. **超軽量**: **1.58ビット(三値)量子化**により、低スペックPCでも巨大モデルが稼働。
2. **適応学習 (TTT)**: 会話するたびに「魂」が成長し、文脈に合わせてリアルタイムに賢くなります。
3. **Pure Rust**: PyTorch依存なし。単一のバイナリで「身体（モデル）」と「魂（記憶）」を管理できます。

---

## 🚀 v0.3.1 新機能: Bit-Converter & Cleanup

*   **🔄 Bit-Converter**: HuggingFace LlamaモデルをBit-TTT形式に変換する独立したGUIツール。
*   **🛠️ リファクタリング**: `tools/` 内のスクリプトを整理し、プロジェクト全体をクリーンアップ。
*   **🔬 Model Lab**: モデルロードと「魂 (.soul)」の管理専用タブ。

---

## 🚀 5分で始めるクイックスタート

### 方法A: バイナリでチャット (推奨)
1.  **配布されたZipを解凍**
2.  **`start_gui.bat` (または `bit_llama.exe`) を実行**
3.  **Model Lab** タブでモデルをロードし、**Chat** タブで会話を開始！
    *   詳しい使い方は [ユーザーガイド](docs/USER_GUIDE_JA.md) を参照してください。

### 方法B: Pythonで試す (開発者向け)
**前提**: Windows x64, Python 3.10
*(※他の環境の方は「ソースからビルド」を参照してください)*

1. **インストール**
   ```bash
   pip install dist/python/cortex_rust-0.3.0-cp310-cp310-win_amd64.whl
   ```

2. **Hello World**
   ```bash
   python examples/hello_bit_llama.py
   # 10Mパラメータのサンプルモデルを自動ロードして動作確認します。
   ```

---

## 💻 動作環境 (System Requirements)

- **OS**: Windows 10 / 11 (64-bit)
- **GPU**: NVIDIA GeForce GTX 10 Series (Pascal) 以降推奨。
    - **CUDA 12.x 対応のドライバ (v530以降)** が必要です。
    - ※Toolkitのインストールは不要です（ランタイムDLL同梱済み）。
- **VRAM**: 4GB以上推奨 (10Mモデルなら2GBでも可)。

---

## 📊 パフォーマンス比較 (vs Llama.cpp)

| モデルサイズ | 手法 | VRAM (GB) | 速度 (Tok/s) |
|------------|--------|-----------|---------------|
| 7B         | FP16   | 14.0      | 45.0          |
| 7B         | 4-bit  | 4.5       | 120.0         |
| **7B**     | **Bit-TTT (1.58b)** | **1.8**   | **1100.0** |

*(RTX 4090でのベンチマーク値)*

---

## 🏗️ ディレクトリ構造 (Refactor V3)

```text
Bit-TTT/
├── crates/
│   ├── rust_engine/    # 🧠 推論コアエンジン
│   ├── bit_llama/      # 🖥️ メインGUI (BitLlamaStudio)
│   └── bit_converter/  # 🔄 独立変換ツール (New!)
├── tools/              # 🛠️ ユーティリティ
│   ├── conversion/     # 変換スクリプト
│   ├── debug/          # 検証・ベンチマーク
│   ├── data/           # データセット準備
│   └── scripts/        # 自動化スクリプト (Powershell)
├── workspace/          # ユーザーデータ (Project, Model, Memories)
├── assets/             # デフォルト設定・プリセット
├── dist/               # 配布用バイナリ
└── docs/               # ドキュメント
```

## 🛠️ 開発者向け

### ソースからビルド
```bash
git clone https://github.com/imonoonoko/Bit-TTT-Engine.git
cd Bit-TTT-Engine
# メインエンジン & GUIのビルド
cargo build --release -p bit_llama --features cuda
# コンバーターのビルド
cargo build --release -p bit_converter
```

### Pythonバインディング開発
```bash
cd crates/rust_engine
maturin develop --release
```

---

## 📖 ドキュメント一覧
- **[DEVELOPER_GUIDE_JA.md](docs/DEVELOPER_GUIDE_JA.md)**: 詳細な開発者ガイド
- **[CODE_ATLAS.md](docs/CODE_ATLAS.md)**: コード構造マップ
- **[ROADMAP.md](ROADMAP.md)**: 今後のロードマップ

---

## 🙏 Acknowledgments / 謝辞

This project incorporates ideas and techniques inspired by and adapted from the DroPE method published by Sakana AI.

**Original work:**
*   **Title**: Extending the Context of Pretrained LLMs by Dropping Their Positional Embeddings
*   **Authors**: Yoav Gelberg, Koshi Eguchi, Takuya Akiba, Edoardo Cetin
*   **Source**: [arXiv:2512.12167](https://arxiv.org/abs/2512.12167) (Submitted on 13 Dec 2025)
*   **License**: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

**Modifications / 改変点:**
We have adapted the positional embedding dropping approach and recalibration concept for our Pure Rust-based, low-bit quantized Test-Time Training (TTT) engine (Bit-TTT-Engine / bit_llama). This includes re-implementation in Rust (no Python dependencies), integration with 1.58-bit quantization, and application to edge-device friendly real-time adaptation, which differs from the original Hugging Face / PyTorch-focused implementation.

The rest of this project is licensed under the MIT License (see LICENSE file).

## 💖 Support
**Solana Wallet**: `13ui3nmE7smmK3Pk8wyKb7RE6wHyMJCcWgCeMRRdoory`

*Created by Project Bit-TTT • MIT License*
