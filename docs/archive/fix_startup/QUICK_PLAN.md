# Quick Plan: Startup Reliability Refactor

## 1. Context (背景/課題)
*   **課題**: `.exe` が起動できない（ウィンドウが開かずに即終了する）。
*   **原因**: パニックまたはResultエラーが発生しているが、Releaseビルドではコンソールが表示されないため、エラー内容を確認できない。
*   **目的**: エラーログをファイルに出力し、原因特定とユーザーへの通知を可能にする。

## 2. Risk Check (リスク確認)
*   **影響範囲**: `main.rs`, `Cargo.toml` のみ。コアロジックには影響しない。
*   **リスク**: 低 (Low)。ログ機構の追加のみ。

## 3. Core Implementation (実装内容)

### A. 依存関係の追加
*   `Cargo.toml` に `tracing-appender = "0.2"` を追加。

### B. ロギングの強化 (main.rs)
*   `main` 関数の冒頭で `tracing_appender` を初期化。
    *   ログファイル: `logs/bit_llama.log` (ローテーション付き)
*   `stdout` (コンソール) と `file` (ファイル) の両方にログを出力するように `tracing_subscriber` を構成。

### C. パニックフックの設置
*   `std::panic::set_hook` を使用し、パニック発生時のバックトレースやメッセージをログファイルに書き込む処理を追加。
    *   これにより `unwrap()` 失敗等のクラッシュ要因が確実に記録される。

## 4. Verification (検証手順)
1.  意図的にパニックを起こすコードを一時的に挿入（テスト用）。
2.  `cargo run --release` で実行し、クラッシュさせる。
3.  `logs/bit_llama.log` が生成され、エラー内容が記録されているか確認。
4.  テストコードを削除し、正常起動を確認。

## 5. Decision
承認され次第、実装を開始する。
