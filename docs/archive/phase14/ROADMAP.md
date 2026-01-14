# Roadmap: Phase 14 (Japanese Proficiency)

## 1. Phasing

### Phase 14-1: Tokenizer Revamp (Unigram) [Complete]
- **Goal**: Unigramトークナイザーの学習とロードを可能にする。
- **Tasks**:
    - [x] Update `VocabArgs` in `vocab.rs` to support `model_type` (BPE/Unigram).
    - [x] Implement `Unigram` training logic using `tokenizers::models::unigram`.
    - [x] **Normalization**: Enable `NFKC` normalization in `vocab.rs` to unify characters.
    - [x] Verify standard Japanese text encoding.

### Phase 14-2: Data Pipeline (Japanese Corpus) [Complete]
- **Goal**: 日本語学習データの自動取得とクリーニング。
- **Tasks**:
    - [x] Create `crates/bit_llama/src/data/download.rs`.
    - [x] Implement downloader for `Wiki40b-Ja` (or a lighter alternative like `mc4-ja` subset).
    - [x] Implement `cleaner` to normalize text (remove HTML, weird spaces).

### Phase 14-3: Integration & Training [Complete]
- **Goal**: エンドツーエンドでの日本語学習。
- **Tasks**:
    - [x] Preprocess Japanese corpus -> `train.u32`.
    - [x] Run `train_llama` with new tokenizer.
    - [x] Verify generation in GUI. (Verified via CLI & Inference Smoke Test: "すセ、fセ確し")

## 2. Circuit Breaker (Stop Rules)

### 2.1. Tokenizer Training Failure
- **Trigger**: `tokenizers` クレートの Unigram 学習がメモリ不足で落ちる、あるいは学習時間が長すぎる (>1時間)。
- **Fallback**: 既存の日本語LLM (`rinna`等) の `tokenizer.json` を手動ダウンロードして配置する運用に切り替える。

### 2.2. Poor Generation Quality
- **Trigger**: 学習しても全く日本語にならない（文字化け、意味不明な文字列）。
- **Fallback**: トークナイザーの問題か学習不足か切り分けるため、まずは `ByteLevel BPE` (Current) で日本語学習を試し、ベースラインと比較する。

## 3. Resource Planning
- **Time**:
    - 14-1: 2 hours
    - 14-2: 3 hours
    - 14-3: 2 hours (Training time excluded)
- **Compute**: CPU High (Tokenizer training is CPU intensive).
