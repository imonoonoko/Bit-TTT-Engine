# Synergy & Value Design: Phase 14

## 1. Efficiency Optimization (既存資産の活用)

### 1.1 `hf-hub` Ecosystem Integration
*   Pythonスクリプトでデータセットをダウンロードする現行フローを、Rustネイティブの `hf-hub` クレートに置き換えることを検討。
*   **Synergy**: 学習実行時に自動でデータセット/Tokenizerを取得できるようになり、ユーザーの手動セットアップ手順 (`python scripts/download.py`) を排除できる。
*   **Value**: 「`cargo run` 一発で学習開始」というユーザー体験（DX）の向上。

### 1.2 `memmap2` for Datasets
*   モデルロードに導入済みの `memmap2` を、巨大な日本語学習データセット（数GB〜）の読み込みにも適用。
*   **Synergy**: メモリ（RAM）圧迫を防ぎつつ、OSのページキャッシュを活用して高速にアクセス。
*   **Value**: 8GB RAMのPCでも数十GBのデータセットで学習が可能になる。

## 2. Cross-Functional Synergy (機能間相乗効果)

### 2.1 Fuzzing as Documentation
*   Fuzzing用に入力生成ロジックを書くことは、「どのような入力が許容されるか」の仕様定義と同じ。
*   **Synergy**: Fuzzingターゲットのコードを、そのまま「堅牢なAPI使用例」としてドキュメント（Examples）に転用可能。

### 2.2 TTT Evaluation for Long Context
*   評価パイプライン（Perplexity計測）の構築は、将来の **Phase 17 (Long Context)** の基盤となる。
*   **Synergy**: 今作ったPPL計測ツールは、後の「本一冊読ませた後の精度」検証にそのまま流用できる。
