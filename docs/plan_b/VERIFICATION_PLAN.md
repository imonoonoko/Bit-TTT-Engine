# VERIFICATION_PLAN.md - Bit-TTT リファクタリング検証計画

**作成日**: 2026-01-11
**目的**: 機能要件および非機能要件を含めた検証計画

---

## 1. 完了定義 (Definition of Done)

### Phase 1 (core_engine.rs 分割)

| チェック項目 | 確認方法 |
|--------------|----------|
| ✅ コンパイル成功 | `cargo build -p rust_engine` |
| ✅ 既存テスト通過 | `cargo test -p rust_engine` |
| ✅ Python バインディング動作 | `cargo build -p rust_engine --features python` |
| ✅ 公開APIの互換性維持 | `use cortex_rust::BitLlama` が動作 |
| ✅ ドキュメント生成 | `cargo doc --no-deps` |

### Phase 2 (train.rs リファクタリング)

| チェック項目 | 確認方法 |
|--------------|----------|
| ✅ 学習ループ動作 | 100ステップのスモークテスト |
| ✅ チェックポイント保存/読み込み | 途中再開テスト |
| ✅ GUI からの呼び出し | GUI で Start → Stop → Resume |
| ✅ ログ出力維持 | `RUST_LOG=info` でStep/Loss確認 |

---

## 2. 観測可能性設計 (Observability)

### ログ設計

```rust
// 推奨パターン
tracing::info!(target: "training", step = %step, loss = %loss, "Step completed");
tracing::warn!(target: "checkpoint", "Overwriting existing checkpoint");
tracing::error!(target: "loader", file = %path, "Failed to load data");
```

| ログレベル | 用途 |
|------------|------|
| `error` | 処理続行不能なエラー |
| `warn` | 回復可能な問題 |
| `info` | 重要なマイルストーン (Step完了, Save完了) |
| `debug` | 詳細な内部状態 |
| `trace` | 非常に詳細なデバッグ情報 |

### エラーハンドリング

```rust
// リファクタリング前
anyhow::bail!("Failed to load model")

// リファクタリング後 (推奨)
#[derive(Debug, thiserror::Error)]
pub enum TrainError {
    #[error("Failed to load model: {0}")]
    ModelLoad(#[source] anyhow::Error),
    
    #[error("Checkpoint not found: {path}")]
    CheckpointNotFound { path: String },
}
```

---

## 3. リグレッションテスト計画

### 自動テスト

| テスト名 | 対象 | コマンド |
|----------|------|----------|
| ユニットテスト | 各レイヤー | `cargo test -p rust_engine` |
| 統合テスト | bit_llama CLI | `cargo test -p bit_llama` |
| スモークテスト | 学習パイプライン | `python tools/pre_demon.py` |

### 手動テスト

| シナリオ | 手順 | 期待結果 |
|----------|------|----------|
| GUI起動 | `cargo run -p bit_llama -- gui` | 画面表示、クラッシュなし |
| 学習100ステップ | GUI → Start Training | Loss減少、グラフ表示 |
| チェックポイント再開 | Stop → Resume | 前回の続きから再開 |
| 推論 | `cargo run -p bit_llama -- inference --model models/latest.safetensors` | テキスト生成 |

---

## 4. 品質メトリクス

| メトリクス | 目標値 | 測定方法 |
|------------|--------|----------|
| コンパイル時間 | ±10%以内 | `cargo build` (増分) |
| バイナリサイズ | ±5%以内 | `ls -l target/release/bit_llama` |
| コード行数 | 各ファイル < 400行 | `wc -l` |
| `unsafe` ブロック | 0 (既存以外の追加なし) | `grep -r "unsafe"` |

---

## 5. 撤退基準 (Stop Rule)

| 条件 | アクション |
|------|------------|
| リファクタリング作業 > 4時間/Phase | スコープ縮小を検討 |
| テスト失敗が5件以上 | 一旦ロールバックして原因分析 |
| PyO3バインディング完全破壊 | Phase 1 のみ完了として終了 |

---

**次のステップ**: Step 5 (Roadmap) でフェーズ分けと実装計画を策定
