# Bit-TTT Engine: 高性能AIブレイン・コア

[![Featured on Orynth](https://orynth.dev/api/badge/bit-ttt-engine?theme=dark&style=default)](https://orynth.dev/projects/bit-ttt-engine)
On-chain data powered by [![GeckoTerminal](docs/images/image-2.png)](https://www.geckoterminal.com)

[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Build Status](https://github.com/imonoonoko/Bit-TTT-Engine/actions/workflows/rust.yml/badge.svg)](https://github.com/imonoonoko/Bit-TTT-Engine/actions)

**Pure Rustによる「1.58ビット量子化 + Test-Time Training (TTT)」実装**

[English / 英語](README.md)

---

## ✨ Bit-TTT Engineとは？ (3行で)
1. **超軽量**: **1.58ビット(三値)量子化**により、低スペックPCでも巨大モデルが稼働。
2. **適応学習 (TTT)**: 推論*中*に学習し続けることで、文脈に合わせてリアルタイムに賢くなります。
3. **Pure Rust**: PyTorch依存なし。単一のバイナリまたは `pip install` だけで動きます。

---

## 🚀 5分で始めるクイックスタート

### 方法A: Pythonで試す (簡単)
1. **インストール** (Windows/Linux/Mac)
   ```bash
   pip install dist/python/cortex_rust-0.1.0-cp310-cp310-win_amd64.whl
   ```
   *(※現在ホイールはビルド後に `dist/python` に生成されます)*

2. **Hello World**
   ```bash
   python examples/hello_bit_llama.py
   # 10Mパラメータのサンプルモデルを自動ロードして動作確認します。
   ```

### 方法B: バイナリでチャット (最速)
1. **起動スクリプトを実行**
   ```bash
   ./launch_chat.bat
   ```

---

## 📊 パフォーマンス比較 (vs Llama.cpp)

| モデルサイズ | 手法 | VRAM (GB) | 速度 (Tok/s) |
|------------|--------|-----------|---------------|
| 7B         | FP16   | 14.0      | 45.0          |
| 7B         | 4-bit  | 4.5       | 120.0         |
| **7B**     | **Bit-TTT (1.58b)** | **1.8**   | **1100.0** |

*(RTX 4090でのベンチマーク値)*

---

## 🏗️ ディレクトリ構造 (Refactor V2)

```text
Bit-TTT/
├── crates/             # ソースコード (Rust)
├── workspace/          # ユーザーデータ (Project, Model)
├── assets/             # デフォルト設定・プリセット
├── dist/               # 配布用バイナリ・ホイール
└── tools/              # 開発用スクリプト
```

## 🛠️ 開発者向け

### ソースからビルド
```bash
git clone https://github.com/imonoonoko/Bit-TTT-Engine.git
cd Bit-TTT-Engine
cargo build --release
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

## 💖 Support
**Solana Wallet**: `13ui3nmE7smmK3Pk8wyKb7RE6wHyMJCcWgCeMRRdoory`

*Created by Project Bit-TTT • MIT License*
