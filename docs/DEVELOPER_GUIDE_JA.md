# Bit-TTT Engine 開発者ガイド

このドキュメントでは、**Bit-TTT Engine** のディレクトリ構造、主要ファイルの役割、および開発フローについて解説します。
新規開発者やコントリビューターがプロジェクトの全体像を把握するための羅針盤として利用してください。

---

## 📂 ディレクトリ構造概観

```text
Bit-TTT Engine
├── crates/                  # Rust ソースコード (ワークスペース)
│   ├── rust_engine/         # [Core] コアライブラリ (モデル定義, 計算カーネル, Python統合)
│   └── bit_llama/           # [App] アプリケーション (CLI, GUI, 学習/推論ランナー)
│
├── tools/                   # 開発者用ツール (PowerShell スクリプト)
├── docs/                    # ドキュメント
│   ├── images/              # 画像リソース
│   └── archive/             # 過去のフェーズの記録 (旧仕様・計画書)
│
├── data/                    # 学習データ・設定ファイル配置場所
│   └── sample_config/       # デフォルト設定ファイル (config.json 等)
│
├── models/                  # (Git除外) 学習済みモデル出力先
├── logs/                    # (Git除外) 実行ログ出力先
└── target/                  # (Git除外) Rust ビルド成果物
```

---

## 📦 `crates/rust_engine` (Core Library)

ニューラルネットワークの定義、計算ロジック、Pythonバインディングを提供するコアライブラリです。`cortex_rust` というクレート名で公開されます。

| パス | 役割 | 詳細 |
|---|---|---|
| `src/lib.rs` | エントリポイント | モジュール公開定義。PyO3のモジュール登録処理。 |
| `src/python.rs` | **Python統合** | `PyTrainer`, `BitLlama` クラスのPyO3実装。Pythonから呼び出すAPIを定義。 |
| `src/model/` | モデル定義 | |
| ├── `llama.rs` | `BitLlama` | Llamaアーキテクチャのメイン構造体。`forward` ロジック。 |
| ├── `block.rs` | Transformer Block | Attention と MLP を組み合わせたブロック定義。 |
| └── `config.rs` | 設定 | モデルのハイパーパラメータ (`BitLlamaConfig`). `Copy`トレイト実装。 |
| `src/layers/` | レイヤー実装 | |
| ├── `bit_linear.rs` | **BitLinear** | 1.58-bit 量子化線形層。重みは {-1, 0, 1}。 |
| ├── `ttt.rs` | **TTT Layer** | Test-Time Training レイヤー。勾配更新による短期記憶。 |
| └── `rms_norm.rs` | RMSNorm | 正規化層。 |
| `src/kernels/` | 計算カーネル | |
| ├── `cpu.rs` | AVX2カーネル | CPU向けの高速行列演算 (SIMD)。 |
| └── `cuda/` | CUDAカーネル | GPU向けの高速計算ロジック (PTX)。 |

---

## 🛠️ `crates/bit_llama` (Application)

ユーザーが直接実行する実行バイナリ (`bit_llama`, `train_llama` 等) を提供するアプリケーションクレートです。

| パス | 役割 | 詳細 |
|---|---|---|
| `src/bin/` | 実行エントリ | |
| ├── `bit_llama.rs` | 推論CLI | `cargo run --bin bit_llama` のエントリポイント。 |
| └── `train_llama.rs` | 学習CLI | `cargo run --bin train_llama` のエントリポイント。 |
| `src/gui/` | **GUI (Tauri)** | トレーニング画面のUIロジック (HTML/JS/Rust)。 |
| `src/train/` | 学習ロジック | |
| ├── `training_loop.rs`| **学習ループ** | 再開機能付きのメイン学習ループ。チェックポイント保存。 |
| └── `checkpoint.rs` | 状態管理 | 重みとオプティマイザ状態のシリアライズ管理。 |
| `src/inference.rs` | 推論エンジン | ストリーミングトークン生成の管理。`InferenceSession`。 |
| `src/vocab.rs` | トークナイザ | SentencePiece (Unigram) および BPE の実装。 |
| `src/loader.rs` | データローダ | `memmap2` を使用した高速データセット読み込み。 |
| `src/monitor.rs` | リソース監視 | VRAM/RAM 使用率のモニタリング (NVML統合)。 |

---

## 🔧 `tools/` (Utility Scripts)

開発や運用を補助するスクリプト群です。

| ファイル/フォルダ | 用途 |
|---|---|
| `BitLlama-Train.ps1` | `train_llama` を適切な引数で起動するラッパー。 |
| `BitLlama-Chat.ps1` | `bit_llama` (推論) を起動するラッパー。 |
| `BitLlama-GUI.ps1` | GUI版トレーナーを起動する。 |
| `pre_demon.py` | **Demon Audit** | コードベースの健全性チェック。 |
| `pre_commit_check.py` | Git Hook用。コミット前の大容量ファイルチェック。 |
| `benchmarks/` | パフォーマンス測定用スクリプト (PyTorch比較など)。 |
| `utils/` | その他ユーティリティ。 |

---

## 📂 その他の重要ディレクトリ

| ディレクトリ | 説明 |
|---|---|
| `examples/` | Python推論の実装例 (`python_inference.py`) など。 |
| `projects/` | GUIトレーナーで使用するプロジェクト保存先 (ユーザーデータ)。 |
| `release/` | ビルド済みバイナリや配布用アーティファクトの格納先。 |

---

## 📚 `docs/` (Documentation)

| ファイル/保存先 | 内容 |
|---|---|
| `specifications.md` | **技術仕様書**。アルゴリズムやデータ構造の詳細。 |
| `ROADMAP.md` | **ロードマップ**。開発計画と進捗状況。 |
| `archive/` | **過去ログ**。終了したフェーズの計画書や一時的なメモ。 |
| `DEVELOPER_GUIDE_JA.md` | **本ドキュメント**。開発者向けガイド。 |

---

## 🔄 開発ワークフロー

### Rustコードの修正
1. `crates/` 以下のコードを修正。
2. `cargo check --workspace` でコンパイル確認。
3. `cargo fmt --all` でフォーマット。
4. (`crates/rust_engine` 修正時) `python crates/rust_engine/examples/python_sanity_check.py` でバインディング確認。

### Pythonバインディングのビルド
```bash
cd crates/rust_engine
maturin develop --release
```

### コミット前確認
```bash
# 全体テスト & リンター
cargo test --workspace
cargo clippy --workspace
```
