# QUICK_PLAN.md - Phase 2: Train モジュールリファクタリング

**作成日**: 2026-01-11
**タイプ**: Lite Process (中規模・同一クレート内リファクタリング)

---

## 1. Context (変更対象と目的)

| 項目 | 内容 |
|------|------|
| **対象ファイル** | `crates/bit_llama/src/train.rs` (565行) |
| **目的** | 巨大な `run()` 関数 (464行) を責務ごとに分割し、保守性を向上 |

### 現在の構造

```
train.rs (565行)
├── L18-58:   TrainArgs (Clap構造体)
├── L63-71:   TrainingState (シリアライズ用構造体)
├── L73-99:   save_training_state() (チェックポイント保存)
└── L101-564: run() (メイン関数 - 464行)
    ├── L101-260:   初期化・設定・モデル構築
    ├── L261-330:   AdamW+保存パス設定
    ├── L330-381:   ベンチマークモード/設定保存
    ├── L383-527:   学習ループ本体
    └── L529-563:   Final Model Save
```

---

## 2. Risk Check (影響範囲確認)

| リスク | 評価 | 緩和策 |
|--------|------|--------|
| CLI引数破壊 | 低 | `TrainArgs` は独立抽出、`pub use` で互換性維持 |
| 学習ループ破壊 | 中 | 小さな関数に分割せず、ループ本体は `training_loop.rs` に1関数として移動 |
| 依存関係破壊 | 低 | `cli.rs` からの `train::run()` 呼び出しは変更なし |

✅ **影響範囲は `bit_llama` クレート内に限定** → Lite Processで続行

---

## 3. Core Implementation (実装方針)

### ファイル構成 (Rust 2018+スタイル)

```
bit_llama/src/
├── train.rs              (モジュール親ファイル - pub mod + pub use)
├── train/
│   ├── args.rs           (TrainArgs)
│   ├── checkpoint.rs     (TrainingState, save_training_state, load_checkpoint)
│   └── loop.rs           (run() 本体 - 学習ループ)
└── lib.rs                (既存: pub mod train)
```

### 各ファイルの責務

| ファイル | 責務 | 行数目安 |
|----------|------|----------|
| `train.rs` (親) | モジュール宣言 + 公開エクスポート | ~15行 |
| `args.rs` | `TrainArgs` 構造体 | ~50行 |
| `checkpoint.rs` | 状態保存/読込 | ~80行 |
| `loop.rs` | `run()` 関数 (学習ループ本体) | ~400行 |

### 移動計画

1. **`train/args.rs`**: `TrainArgs` をそのまま移動
2. **`train/checkpoint.rs`**: `TrainingState`, `save_training_state()` 移動 + `load_checkpoint()` 関数新設
3. **`train/loop.rs`**: `run()` 関数を移動 (内部構造は変更しない)
4. **`train.rs`**: モジュール宣言 + `pub use` で既存APIを維持

---

## 4. Verification (検証手順)

| ステップ | コマンド | 期待結果 |
|----------|----------|----------|
| コンパイル確認 | `cargo check -p bit_llama` | エラーなし |
| CLI動作確認 | `cargo run --bin train_llama -- --help` | ヘルプ表示 |
| Workspace確認 | `cargo check --workspace` | 全クレートエラーなし |

---

## 5. Stop Rule (中断基準)

| 条件 | アクション |
|------|------------|
| コンパイルエラーが3回連続で解決しない | ロールバック (`git checkout -- .`) |
| 作業時間 > 30分 | スコープ縮小 (`args.rs` 分離のみで完了) |
| 依存関係の破壊が判明 | Phase 1 のような段階的アプローチに切替 |

---

**承認後、実装を開始します。**
