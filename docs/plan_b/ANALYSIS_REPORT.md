# Analysis Report: Bit-TTT Phase 14 & Quality Improvements

## 1. Executive Summary
本レポートは、**Phase 14 (日本語能力獲得)** および **品質向上施策 (Fuzzing, Eval, Safety)** の実装に向けた現状分析結果である。
現在の Bit-TTT は英語 (TinyStories) に特化したプロトタイプであり、日本語対応には Tokenizer の置換と評価パイプラインの構築が必須である。また、実験的実装であるため安全性（Panic, Unsafe）の担保が不十分である。

## 2. As-Is Structural Analysis (現状構造分析)

### 2.1 Component Structure
*   **`crates/rust_engine` (Core Logic)**
    *   計算核。`BitLinear`, `TTTLayer` などの独自レイヤーを実装。
    *   `legacy/c_api.rs` に外部連携用の `unsafe extern "C"` インターフェースが存在。
    *   テストは単体テストのみで、Fuzzing による境界値検証は未実施。
    
*   **`crates/bit_llama` (Application)**
    *   学習・推論・GUIのエントリーポイント。
    *   **Tokenizer**: `tokenizers` クレートを使用中だが、TinyStories 用の設定がハードコードされている可能性あり。
    *   **Evaluation**: CLI での対話的確認のみ。定量評価 (Perplexity) の仕組みが存在しない。

### 2.2 Critical Gaps (課題)
1.  **Tokenizer Capability**: 現在の語彙（Vocabulary）は英語中心であり、日本語を効率的に学習できない。
2.  **No Verify Loop**: 「変更 → 精度低下」を検知するガードレール（評価自動化）がない。
3.  **FFI Risk**: `c_api.rs` がポインタを直接扱っており、不正な入力で Segfault を起こすリスクがある。
4.  **Input Fragility**: 壊れたモデルファイルや異常なトークン列に対する耐性が不明。

## 3. To-Be Definition (あるべき姿)

### 3.1 Japanese Training Pipeline
*   **Tokenizer**: Llama-3 または国産LLM（Rakuten, ELYZA等）のTokenizerをロード可能にする。
*   **Dataset**: `hf-dataset` 等を用いて日本語テキストをストリーミングし、学習ループに供給する。

### 3.2 Robust Quality Assurance
*   **Fuzzing**: `cargo-fuzz` を導入し、`rust_engine` の公開API（特に `forward`, deserialization）をランダム入力で攻撃し、Panicフリーを保証する。
*   **Safety**: `unsafe` ブロックにドキュメント（Safety Comment）を追記し、可能な限り `safe` なラッパーで包む。

### 3.3 Automated Eval
*   **Perplexity Monitor**: 学習中に検証データ（Wiki40b-ja 等）のPPLを定期計測し、ログ出力する。

## 4. Derived Tasks (導出タスク)
1.  **[Infra]** `cargo-fuzz` 環境のセットアップ。
2.  **[Core]** `c_api.rs` の安全性レビューと `debug_assert!` 追加。
3.  **[Feature]** 日本語Tokenizer対応（`tokenizer.json` の差し替え検証）。
4.  **[Feature]** 評価用スクリプト/バイナリの実装 (`metrics.rs`)。
