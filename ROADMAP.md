# マスターロードマップ: Bit-TTT Engine

## ✅ 完了済みのフェーズ
- **Phase 1-4**: コアエンジン (Rust), BitLinear, TTT, GUIプロトタイプ.
- **Phase 5**: Python統合 (`cortex_rust` PyO3バインディング).
- **Refactor V2**: ディレクトリ構造の大掃除 (`workspace/`, `dist/`).

---

## 🚀 Phase 6: 展開と採用 (Deployment Phase) [完了]
*(目標: "Pip install してすぐ動く")*
- [x] **Python Wheel**: Windows用 `cortex_rust` wheel のビルド.
- [x] **サンプル重み**: テスト用の10Mモデルを作成.
- [x] **Hello Script**: `examples/hello_bit_llama.py` (設定不要).

## 📢 Phase 7: ショーケース (Visibility Phase) [完了]
*(目標: "動作を証明する")*
- [x] **リッチREADME (v2)**: 5分で終わるクイックスタート.
- [x] **ベンチマーク**: 比較表 (vs Llama.cpp).
- [ ] **デモ動画**: (ユーザーアクション待ち).

## 📦 Phase 7.5: スタンドアロン変換GUI (BitConverter) [進行中]
*(目標: "誰でもLlama-3を変換できる")*
- [ ] **独立アプリ化**: `crates/bit_converter` (Rust/egui) の作成.
- [ ] **Python連携**: `tools/convert_hb.py` をラップし、環境構築の壁を下げる.
- [ ] **GUI**: フォルダ選択 → 変換ボタン → ログ表示 のシンプル操作.

---

## 🤝 Phase 8: エコシステムとコミュニティ (Usability Phase)
*(目標: "創作を可能にする")*

### 8.1 Web UI (ブラウザチャット)
- **ツール**: `tools/web_ui.py` (Gradio または Streamlit).
- **機能**:
  - `dist/assets` または `workspace/models` からモデルをロード.
  - 履歴付きチャットインターフェース.
  - パラメータスライダー (Temperature, Top-P).

### 8.2 学習ガイド
- **ドキュメント**: `docs/TRAINING_GUIDE.md`.
- **内容**:
  - "`corpus.txt` の準備方法".
  - "学習ループの実行方法".
  - "推論用エクスポート方法".

### 8.3 高度なベンチマーク
- **スクリプト**: `tools/benchmark_suite.py`.
- **指標**: 長文コンテキストの安定性, TTTオーバーヘッド計測.

---

## 🐣 Phase 9: 育成革命 ("Tamagotchi" Phase)
*(目標: "訓練" から "育成" へ)*
- [x] **Pure Rust Instruct**: `PrepareInstruct` コマンド (Python依存なし).
- [ ] **Sleep Mode v2**: バックグラウンドでのファイル摂食. "ドキュメントの夢を見る".
- [x] **GUI Training Tab**: `PrepareInstruct` & MeZO操作用のドラッグ&ドロップUI.
- [x] **視覚的フィードバック**: GUIコンソールでの進捗表示.

## 📚 Phase 10: データの民主化 (Content Phase)
*(目標: "空っぽの脳みそ問題" の解決)*
- [ ] **ナレッジパック**: `.u32` バイナリパックの配布 (Wikipedia, Codingなど).
- [ ] **パーソナルインポート**: "メール/チャットログの取り込み" ウィザード.
- [ ] **マーケットプレイス**: (将来) グラウンディング済みソウルの共有.

## 🧬 Phase 10.5: Llama Assimilation (既存モデル同化)
*(目標: "Llama-3を吸い尽くす")*
- [x] **Adaptive Converter (Core)**: 可変ビット精度 (1.58bit x N) 分解ロジック (完了).
- [ ] **Isomorphic Engine (Rust)**: 可変レイヤーを読み込み、実行時に動的加算するエンジンの拡張.
- [ ] **Tuning**: 「サイズvs賢さ」のスイートスポット探索 (4GB VRAMで8Bモデルを動かす).

## 🕸️ Phase 11: Omni-Reach (ハードウェア非依存 Phase)
*(目標: "トースターでも動く")*
- [x] **MeZO Training Engine**: ゼロメモリ最適化 (逆伝播なし). 4GB VRAMでの学習.
- [x] **CPU Eco Mode**: 一晩かかる学習のための最適化済み f32/q8 CPUバックエンド.
- [x] **シームレスな再開**: チェックポイントシステムの刷新 (Best/Latest + 自動修復).

---

## 🔮 将来のフェーズ (旧構想)
*注: 以前のロードマップから統合*
- **Phase 14**: 日本語能力 (トークナイザー拡張).
- **Phase 15**: 70Bへのスケーリング (マルチGPU/CPUオフロード).
- **Phase 16**: WebAssembly (Wasm) サポート.
