# QUICK_PLAN.md - Phase 3: Legacy モジュール削除

**作成日**: 2026-01-11
**タイプ**: Lite Process (低リスク・クリーンアップ)

---

## 1. Context (変更対象と目的)

| 項目 | 内容 |
|------|------|
| **対象** | `crates/rust_engine/src/legacy/` (4ファイル, 計9.5KB) |
| **目的** | 使用されていない deprecated ndarray 実装を削除し、コードベースを整理 |

### 対象ファイル

```
legacy/
├── mod.rs          (57B)
├── bit_linear.rs   (2.6KB) - ndarray版 BitLinear
├── ttt_layer.rs    (3.4KB) - ndarray版 TTTLayer
└── c_api.rs        (3.4KB) - C FFI (未使用)
```

---

## 2. Risk Check (影響範囲確認)

| 検索クエリ | 結果 | 判定 |
|------------|------|------|
| `legacy::` | 0件 | ✅ 使用なし |
| `LegacyBitLinear` | 1件 (`lib.rs` のみ) | ⚠️ deprecated エクスポート |
| `LegacyTTTLayer` | 1件 (`lib.rs` のみ) | ⚠️ deprecated エクスポート |

**結論**: 外部から使用されている箇所は `lib.rs` の `#[deprecated]` 付き re-export のみ。安全に削除可能。

---

## 3. Core Implementation (実装内容)

1. `lib.rs` から `pub mod legacy` と deprecated re-export を削除
2. `legacy/` ディレクトリを物理削除
3. `cargo check` で検証

---

## 4. Verification (検証手順)

| ステップ | コマンド | 期待結果 |
|----------|----------|----------|
| コンパイル確認 | `cargo check --workspace` | エラーなし |
| Git確認 | `git status` | legacy/ 削除が反映 |

---

## 5. Stop Rule (中断基準)

| 条件 | アクション |
|------|------------|
| 削除後にコンパイルエラー | `lib.rs` の変更のみロールバック |
| 依存クレートで使用判明 | deprecated 警告を残して維持 |

---

**即時実行可能** (Lite Process)
