# Dependency Map: Phase 14 (Japanese Proficiency)

## 1. Component Flowchart

```mermaid
graph TD
    %% Nodes
    Wiki[("🌐 Wiki40b (JA)")];
    Tiny[("📄 TinyStories (EN)")];

    subgraph "Data Pipeline (crates/bit_llama/src/data)"
        Downloader["⬇️ Downloader<br>(New Module)"];
        Cleaner["🧹 Cleaner/Normalizer<br>(New Module)"];
        Corpus["📄 corpus.txt<br>(Mixed JA/EN)"];
    end

    subgraph "Vocab Pipeline (crates/bit_llama/src/vocab)"
        Trainer["⚙️ Tokenizer Trainer<br>(Update: Unigram support)"];
        JSON["📝 tokenizer.json<br>(Unigram Model)"];
    end

    subgraph "Training Pipeline (crates/bit_llama/src/train)"
        Loader["📦 BitLoader"];
        Engine["🧠 Cortex Engine<br>(BitLlama)"];
        Model["💾 model.safetensors"];
    end

    %% Flows
    Wiki --> Downloader
    Tiny --> Cleaner
    Downloader --> Cleaner
    Cleaner --> Corpus

    Corpus --> Trainer
    Trainer --> JSON

    JSON --> Loader
    JSON --> Engine
    Corpus --> Loader
    Loader --> Engine
    Engine --> Model

    %% Dependencies
    classDef new fill:#d4edda,stroke:#28a745,color:#155724;
    classDef existing fill:#e2e3e5,stroke:#adb5bd,color:#383d41;

    class Downloader,Cleaner,Trainer new;
    class Loader,Engine,JSON,Model existing;
```

## 2. Risk Assessment (Impact Analysis)

| Component | Dependency Risk | Impact | Mitigation |
| :--- | :--- | :--- | :--- |
| **`vocab.rs`** | High | `tokenizer.json` の形式が変わると、推論時のデコード結果が化ける可能性がある。 | `Unigram` と `BPE` の共存、または明確なモード切替を実装する。 |
| **`BitLoader`** | Medium | 日本語文字コード (UTF-8) の境界でデータを分割すると文字化けするリスク。 | `BitLoader` は `u32` (Token ID) ベースなので影響なし。前処理段階 (`preprocess`) でのエンコードさえ正しければ安全。 |
| **GUI** | Low | トークナイザーの進捗表示機能などに影響。 | 既存のログストリーム (`mpsc`) を使用するため、大きな改修不要。 |

## 3. Critical Path
1.  **Data Downloader**: これがないとトークナイザーの学習が始まらない。
2.  **Unigram Support**: `tokenizers` クレートの設定変更。
3.  **Validation**: 生成されたトークンが妥当か（漢字がバラバラになっていないか）の確認。
