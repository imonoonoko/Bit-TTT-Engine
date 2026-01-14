# Synergy & Value Design: Phase 14

## 1. Efficiency Optimization (既存資産の活用)

### 1.1. Leveraging `BitLoader`
- **Current**: `BitLoader` uses `memmap2` for efficient `u32` token loading.
- **Synergy**: 日本語データはデータ量が膨大になるため、テキスト (`.txt`) のまま扱うのは非効率。
    - Phase B-2 で整備した `preprocess` コマンド (`text -> u32`) をそのまま流用することで、学習時のI/Oボトルネックを回避できる。
    - **Action**: `bit_llama data --preprocess` が日本語テキスト (`corpus_ja.txt`) も透過的に扱えることを確認・保証するだけでよい。

### 1.2. Unified CLI Setup
- **Current**: `bit_llama` CLI is centralized via `clap`.
- **Synergy**: `bit_llama data --download ja` のようにサブコマンド引数を追加するだけで、既存のCLI構造を壊さずに機能追加可能。

## 2. Cross-Functional Synergy (付加価値)

### 2.1. "Chat Template" Readiness
- **Concept**: Phase 16 で予定している Chat Template 対応 (`User: ... Assistant: ...`) の布石を打つ。
- **Value**: 日本語データセットを作成する際、単純な文章の羅列ではなく、最初から `Wiki40b` のような構造化データや、`OpenAssistant` のような対話形式を意識した `Special Token` (`<|user|>`, `<|model|>`) を語彙に含めておく。
- **Result**: Phase 16 でのトークナイザー再学習の手間を省ける。

### 2.2. Visual Tokenization (GUI)
- **Concept**: 学習前に、トークナイザーが文章をどう分割するかをGUIで確認できる機能。
- **Value**: 「漢字が細切れになりすぎている」などの問題を視覚的に発見できる。
- **Action**: (今回はスコープ外だが) GUIの `Data Prep` タブに "Test Tokenizer" ボタンを配置する下地を作る。

---

## 3. Synergy Score
- **Dev Cost**: Low (既存パイプライン流用)
- **User Value**: High (日本語対応)
- **Future Proof**: High (Chat Template準備)
