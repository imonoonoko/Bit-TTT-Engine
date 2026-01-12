# Bit-TTT Project Roadmap

## 🚀 Future Roadmap (Phase 14+)

提出された `ARTICLE_DRAFT.md` および `BIT_TTT_SPEC.md` の内容を踏まえ、Bit-TTT プロジェクトを次の段階へ進めるためのロードマップです。

### Phase 14: 日本語能力の獲得と評価 (Japanese Proficiency) [Complete]
記事ドラフトにある通り、次は「日本語」に焦点を当てます。

*   [x] **Tokenizerの刷新/拡張**: 既存のLlama系トークナイザーで日本語効率が良いものを採用、あるいは語彙拡張を行います。
*   [x] **学習データ**: `izumi-lab/llm-japanese-dataset` や `wiki40b-ja` などを用いた学習パイプラインの構築。
*   [x] **評価**: JGLUE 等の日本語ベンチマークでのスコア計測を実施します。

### Phase 15: 大規模化への挑戦 (Scaling to 7B/70B)
「70BをVRAM 8GB〜16GBで」という目標に向けた最適化です。

*   **ストリーミングロード**: 巨大なモデルを一度にメモリに乗せず、`mmap` 等を使って必要な部分だけロード、あるいはディスクから直接推論する仕組みを実装します（Rustの低レベル制御を活用）。
*   **Multi-GPU / CPU Offloading**: Candleの機能を活用し、レイヤーごとにデバイスを分散させる実装を行います。

### Phase 16: エコシステムの拡充 (Ecosystem & Usability)
*   **Chat Template対応**: Llama-3やChatML形式のプロンプトをエンジン側で自動処理する機能を実装します（ユーザーが手動で `<|im_start|>` 等を打たなくて済むように）。
*   **WASM (WebAssembly) 対応**: Rust + Candle の強みを活かし、ブラウザだけで動作する「Bit-TTT Web」を公開します。
*   [x] **Tauriアプリの完成**: `v0.1.0 (Prototype)` をリリースしました。今後は機能拡充を目指します。

### Phase 17: 特化型TTTの探求 (Advanced TTT)
*   **Long Contextの実験**: TTTの「無限のコンテキスト」 を実証するため、本一冊分を読ませて、その内容について質問するデモを作成します。
*   **Hierarchical TTT**: 文脈の長期記憶と短期記憶を階層的に管理するアーキテクチャの実験を行います。

---

## 🛠️ Areas for Improvement (改善・強化提案)

品質と堅牢性をさらに高めるための強化ポイントです。これらの実装は各Phaseと並行して進めます。

### 1. Fuzzing (ファジング) の導入
LLMエンジンは、外部からの予期せぬ入力（壊れたモデルファイル、不正なトークン列）を受け取る可能性があります。
*   `cargo-fuzz` などを導入し、パニック（クラッシュ）を引き起こす入力がないか検証します。

### 2. 定量的な精度評価 (Evaluation Pipeline)
現在は「TinyStoriesで言葉を話し始めた」 という定性的な確認段階です。開発に伴う「知能」への影響を定量的に監視します。
*   Perplexity測定の自動化。
*   軽量なベンチマーク（Hellaswag等のサブセット）をCIまたはリリースフローに組み込みます。

### 3. APIの安全性
*   `unsafe` ブロックの使用箇所（もしあれば）に対する集中的なレビューとドキュメント化を実施します。
*   特に `Bit_TTT.dll` などのFFI境界での安全性確認を徹底します。
