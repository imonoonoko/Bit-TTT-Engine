# Verification Plan: Phase 14 & Improvements

## 1. Definition of Done (完了定義)

以下の条件をすべて満たした時点で「Phase 14 & 品質向上タスク」の完了とする。

### 1.1 Functional Requirements (機能要件)
*   [ ] **Japanese Tokenization**: 指定した日本語Tokenizer (tokenizer.json) をロードし、日本語文字列を正しくトークンID列に変換できること。
*   [ ] **Evaluation Metric**: 新規作成する `evaluate` コマンドが、検証データセットに対して Perplexity (または Loss) を出力すること。
*   [ ] **Fuzzing Setup**: `cargo fuzz run` が動作し、最低1つのターゲット（例: Forward Pass）に対して10分間走らせてクラッシュしないこと。

### 1.2 Non-Functional Requirements (非機能要件)
*   **Performance**: 評価（Evaluation）処理が、学習速度（トークン/秒）を著しく低下させないこと（別プロセスまたは非同期実行）。
*   **Safety**: `unsafe` コードブロックに適切な `# Safety` コメントが付与されていること。

## 2. Verification Steps (検証手順)

### Step 1: Fuzzing Verification
```bash
# 1. Install cargo-fuzz
cargo install cargo-fuzz

# 2. Init fuzzing
cd crates/rust_engine
cargo fuzz init

# 3. Create target for TTTLayer forward
# (Write fuzz_target code...)

# 4. Global Run
cargo fuzz run fuzz_target_1 -- -max_total_time=600 # 10 mins
```

### Step 2: Evaluation Pipeline Check
```bash
# 1. Prepare Dummy Japanese Data
echo "これはテストです。AIは日本語を話しますか？" > test_ja.txt

# 2. Run Eval
cargo run --bin evaluate -- --model path/to/model --data test_ja.txt
# Output expectation: "Perplexity: 12.34" (Not NaN/Inf)
```

### Step 3: Tokenizer Sanity Check
```bash
# Verify specific tokens are present
cargo run --bin check_tokenizer -- --text "こんにちは"
# Output expectation: [234, 1892, ...] (Not [0, 0, 0] i.e. UNK)
```

## 3. Observability Design
*   **Metrics**: 学習ログ (`training_log.jsonl`) に `"val_loss": 2.34` のようなフィールドを追加し、GUIでグラフ化できるようにする。
