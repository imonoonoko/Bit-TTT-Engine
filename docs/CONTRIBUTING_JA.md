# Bit-TTT Engine への貢献

[![Featured on Orynth](https://orynth.dev/api/badge/bit-ttt-engine?theme=dark&style=default)](https://orynth.dev/projects/bit-ttt-engine)

Bit-TTT Engine への貢献に興味を持っていただきありがとうございます！

## 🚀 クイックスタート

```bash
# リポジトリをクローン
git clone https://github.com/imonoonoko/Bit-TTT-Engine.git
cd Bit-TTT-Engine

# ビルド
cargo build

# テスト実行
cargo test --workspace

# 問題チェック
cargo clippy --workspace
```

## 📋 開発ガイドライン

### コードスタイル
- コミット前に `cargo fmt` を実行
- [Rust API ガイドライン](https://rust-lang.github.io/api-guidelines/) に従う
- 公開 API にはドキュメントコメントを追加

### コミットメッセージ
[Conventional Commits](https://www.conventionalcommits.org/) に従う：
```
feat(core): 新しいレイヤータイプを追加
fix(train): チェックポイント読込の問題を解決
docs: README を更新
refactor(model): 大きなファイルをモジュールに分割
```

### プルリクエスト
1. リポジトリをフォーク
2. フィーチャーブランチを作成 (`feat/my-feature`)
3. 変更を加える
4. `cargo test` と `cargo clippy` を実行
5. プルリクエストを送信

## 🏗️ プロジェクト構造

```
crates/
├── rust_engine/     # コアライブラリ (cortex_rust)
│   ├── layers/      # ニューラルネットワーク層
│   ├── model/       # モデルアーキテクチャ
│   └── python.rs    # Python バインディング
│
└── bit_llama/       # CLI アプリケーション
    ├── train/       # 学習パイプライン
    ├── gui/         # Tauri GUI
    └── cli.rs       # CLI エントリーポイント
```

## 🎯 初心者向けの課題

- [ ] レイヤーのユニットテストを追加
- [ ] エラーメッセージを改善
- [ ] ドキュメントの例を追加
- [ ] メモリ使用量を最適化

## 📬 連絡先

- GitHub Issues: [バグ報告・機能リクエスト](https://github.com/imonoonoko/Bit-TTT-Engine/issues)

---

*貢献ありがとうございます！*
