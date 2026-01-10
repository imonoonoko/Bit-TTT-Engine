# Dependency Map: Phase 14 & Improvements

## 1. System Structure Overview

```mermaid
graph TD
    subgraph "External Ecosystem"
        HF[HuggingFace Hub]
        JapaneseWiki[Wiki40b-ja / Izumi-lab Dataset]
        JP_Tokenizer[Japanese Tokenizer (Llama-3/ELYZA)]
    end

    subgraph "Bit-TTT Workspace"
        subgraph "crates/rust_engine (Core)"
            TTTLayer[TTTLayer (Logic)]
            BitLinear[BitLinear (Logic)]
            FFI[c_api.rs (Unsafe Boundary)]
        end

        subgraph "crates/bit_llama (App)"
            Trainer[bin/train_llama]
            Inference[bin/chat]
            Evaluator[bin/evaluate (NEW)]
        end

        subgraph "Testing & QA"
            Fuzzer[cargo-fuzz Targets (NEW)]
            Benchmarks[Criterion Benches]
        end
    end

    %% Dependencies
    Trainer -->|Uses| TTTLayer
    Trainer -->|Uses| JP_Tokenizer
    Trainer -->|Streams| JapaneseWiki
    
    Inference -->|Uses| TTTLayer
    Inference -->|Uses| JP_Tokenizer
    
    Evaluator -->|Calc PPL| TTTLayer
    Evaluator -->|Uses| JapaneseWiki

    Fuzzer -->|Fuzzes| TTTLayer
    Fuzzer -->|Fuzzes| BitLinear
    Fuzzer -->|Verifies| FFI
```

## 2. Risk Assessment (リスク評価)

### 2.1 Critical Dependencies
*   **Tokenizer Compatibility**:
    *   `tokenizers` クレートが日本語モデル（SentencePiece / BPE）を正しくロードできるか。
    *   **Risk**: 一部の独自形式Tokenizerがロード不可の場合、変換スクリプトが必要。
    
*   **FFI Stability (c_api.rs)**:
    *   `rust_engine` の内部構造（レイアウト）変更が、C-APIのポインタ参照先に影響しないか。
    *   **Risk**: 構造体定義の変更時、FFI側を更新し忘れると Segfault (Undefined Behavior)。

### 2.2 Side Effects
*   **Memory Usage (Evaluation)**:
    *   評価（Evaluation）プロセスが学習プロセスとメモリを共食いしないか。
    *   **Mitigation**: 評価はチェックポイント保存後に別プロセス、あるいは学習を一時停止して実行。

## 3. Impact Scope
*   **High Impact**: `crates/bit_llama/src/main.rs`, `crates/rust_engine/src/lib.rs`
*   **Medium Impact**: `crates/bit_llama/Cargo.toml` (Deps追加)
*   **Low Impact**: `README.md` (使い方の更新)
