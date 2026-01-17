# Bit-TTT Engine 開発者ガイド (Refactor V2)

**Bit-TTT Engine** は、リファクタリング (V2) により、ソースコード・ユーザーデータ・ビルド生成物が明確に分離されたディレクトリ構造を採用しています。

---

## 📂 ディレクトリ構造概観

```text
Bit-TTT/
├── crates/             # 🦀 ソースコード (Rust Workspace)
│   ├── rust_engine/    # [Core] 計算カーネル, PyO3バインディング
│   ├── bit_llama/      # [App]  CLI, GUI, 学習ロジック
│   └── bit_converter/  # [Tool] スタンドアロン変換GUI (New!)
│
├── workspace/          # 👤 ユーザー領域 (Git除外)
│   ├── projects/       # GUIプロジェクトデータ
│   ├── models/         # ダウンロード/学習済みモデル
│   ├── logs/           # 実行ログ
│   └── data/           # 学習用データセット (Wiki40b等)
│
├── assets/             # 🎨 静的リソース
│   ├── defaults/       # デフォルト設定 (config.json)
│   └── presets/        # プリセットデータ
│
├── dist/               # 📦 配布用アーティファクト
│   ├── binaries/       # コンパイル済み実行ファイル (.exe, .dll)
│   └── archives/       # 配布用パッケージ (.zip)
│
├── tools/              # 🛠️ 開発・運用スクリプト
│   ├── conversion/     # モデル変換
│   ├── debug/          # 検証・ベンチマーク
│   ├── data/           # データ準備
│   └── scripts/        # PowerShellラッパー (start_gui.bat等)
│
└── docs/               # 📚 ドキュメント
    ├── archive/        # 過去ログ
    └── ...
```

---

## 🚀 クイックスタート (開発者向け)

### 1. ビルドと実行
```bash
# GUIの起動
cargo run --release --bin bit_llama -- gui

# CLIでの学習
cargo run --release --bin train_llama -- --data workspace/data/TinyStories
```

### 2. Pythonバインディング
```bash
cd crates/rust_engine
maturin develop --release
```

---

## 🧩 主要コンポーネント詳細

### `workspace/` (ユーザー領域)
実行時に生成されるファイルや、ユーザーが用意するデータセットは全てここに配置します。
- **Git管理対象外** です (ただし `.keep` ファイルによりフォルダ構造のみ維持)。
- **Projects**: GUIで作成したプロジェクトは `workspace/projects/<name>/` に保存され、その中に `models/`, `logs/` が個別に生成されます。

### `assets/` (静的アセット)
- プロジェクト自体のリポジトリに含まれるべき設定ファイルやデフォルトデータです。
- `configs/`: モデルのハイパーパラメータテンプレートなど。

### `crates/` (ソースコード)
- **rust_engine**: エンジンの心臓部。BitNet b1.58 の行列演算ロジック (`kernels/`)、Llama構造定義 (`model/`)、Python連携 (`python.rs`) を担当。
- **bit_llama**: アプリケーション層。TauriベースのGUI (`gui/`)、学習ループ制御 (`train/`)、CLI引数処理 (`cli.rs`) を担当。

---

## 🛠️ ツール (tools/)
PowerShellスクリプトで開発を支援します。
- `BitLlama-Train.ps1`: `train_llama` のラッパー。
- `BitLlama-GUI.ps1`: GUI起動ラッパー。

## 🧪 テスト
```bash
python crates/rust_engine/examples/python_sanity_check.py
cargo test --workspace
```
