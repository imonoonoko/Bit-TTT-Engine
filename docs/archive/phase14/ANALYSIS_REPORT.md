# Analysis Report: Japanese Proficiency (Phase 14)

## 1. Executive Summary
Bit-TTT エンジンを日本語対応させるための現状分析結果。
現在の `vocab.rs` は GPT-2 スタイルの `ByteLevel BPE` を採用しているが、日本語のような多様な文字種を持つ言語に対してはトークン効率が悪化する傾向がある。
また、日本語学習データの取得・クリーニングを行うパイプラインが完全に欠落している。

## 2. Structural Analysis (As-Is)

### 2.1. Tokenizer (`vocab.rs`)
- **Type**: `BPE` (Byte Pair Encoding)
- **Pre-Tokenizer**: `ByteLevel` (Splits UTF-8 bytes)
- **Settings**:
    - `Vocab Size`: Default 32,000
    - `Min Frequency`: 2
- **Issue**:
    - 日本語は「文字」の意味が強いため、バイト単位のBPEよりも **Unigram (SentencePiece)** または **Morpheme-based (MeCab/Sudachi pre-tokenization)** が一般的に高効率である。
    - 現行の `ByteLevel BPE` でも動作はするが、1漢字が3トークン (3バイト) に分割されるケースが多く、コンテキスト長 (Context Window) を浪費する。
    - **Normalization**: 現状は正規化処理がなく、全角/半角英数が別トークンとして扱われる懸念がある (NFKC必須).

### 2.2. Training Loop (`training_loop.rs`)
- **Integration**:
    - `tokenizers::Tokenizer::from_file("tokenizer.json")` でロード。
    - モデル内部の構造 (`BitLlamaConfig`) には `vocab_size` のみが渡される。
- **Observation**:
    - トークナイザーのアルゴリズム（BPE or Unigram）はコード上で抽象化されており、`tokenizer.json` を差し替えるだけでエンジン側は対応可能。
    - **Engine側の改修コストは低い**。

### 2.3. Dataset Pipeline
- **Status**: **Missing**.
- **Current**: `Wiki40b` への参照はあるが、ダウンローダーや解凍ロジックが存在しない。

## 3. Core Intent Verification

### 3.1. User Intent
- "日本語能力の獲得" が最優先。
- つまり、単に「日本語のエラーが出ない」だけでなく、「自然な日本語を生成できる」ことが求められる。

### 3.2. Technical Direction
1.  **Tokenizer**:
    - **Option A (Easy)**: 既存の日本語LLM (e.g., `rinna/japanese-gpt-neox-small`, `cyberagent/open-calm`) の `tokenizer.json` を流用する。
    - **Option B (Custom)**: `tokenizers` クレートの **Unigram Mode** を使用し、日本語コーパスから独自学習する。
        - **推奨**: プロジェクトの自律性を高めるため **Option B** を採用。ただし学習データが必要。
2.  **Vocabulary Expansion**:
    - 英語能力も維持するため、Wiki40b (Ja) + TinyStories (En) の混合コーパスでトークナイザーを学習すべき。
    - **Vocab Size**: 混合コーパスの競合を避けるため、デフォルトを **48,000** に拡張する。

## 4. Requirements for Phase 14
1.  **Data Acquisition**: `bit_llama data --download-ja` のようなコマンドの実装。
2.  **Tokenizer Training**: `vocab.rs` を Unigram 対応に拡張。
    - **Normalization**: `NFKC` を有効化し、表記揺れ (全角/半角) を統一する。
3.  **Efficiency**: 日本語の平均トークン長 (Characters per Token) を 1.5 程度に抑える（BPEだと ~0.8 程度まで落ちる可能性がある）。
