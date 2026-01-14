# Bit-Llama Studio ユーザーガイド

Bit-Llama Studio は、誰でも簡単に **1 bit LLM (BitNet b1.58)** の学習と実験ができるオールインワン・ツールです。
このガイドでは、データの準備から学習、そして対話（推論）までの手順を解説します。

---

## 🚀 はじめに (Getting Started)

1.  **起動**: フォルダ内の `Bit-TTT.exe` (または `start_gui.bat`) をダブルクリックします。
2.  **デモモード**: `start_demo.bat` を使うと、デモ用のプロジェクト構成ですぐに試せます。

---

## 📋 ワークフロー (Workflow)

AIを作るには、以下の4つのステップを左から順番に進めます。

### 1. Data Prep (データの準備)
AIに学ばせる「テキストデータ」と「辞書」を用意します。

1.  **📂 Open Raw Folder**:
    *   このボタンを押して開くフォルダに、学習させたいテキストファイル (`.txt`, `.jsonl`) を入れます。
    *   最初は「青空文庫」のテキストや、Wikiのダンプなどがおすすめです。
2.  **🔗 Concatenate Corpus**:
    *   バラバラのファイルを `corpus.txt` という1つのファイルにまとめます。
3.  **🔤 Train Tokenizer**:
    *   **Vocab Size**: 辞書のサイズ。8000 (小さめ) 〜 32000 (標準) を選びます。
    *   **Fast Mode**: チェック推奨。最初の100MBだけを使って高速に辞書を作ります。
    *   ボタンを押すと `tokenizer.json` が生成されます。

### 2. Preprocessing (前処理)
テキストデータを、AIが学習しやすい「数字の形式 (バイナリ)」に変換します。

1.  **Template**:
    *   単なる小説などを読ませる場合はチェック不要 (Text Completion)。
    *   対話データ (`input`/`output`形式) を使う場合は **Alpaca** や **ChatML** を選びます。
2.  **▶ Start Conversion**:
    *   数分かかることがあります。完了すると `train.u32` が生成されます。

### 3. Training (学習)
GPUを使ってAIモデルを鍛えます。

1.  **Profile (プロファイル)**:
    *   **Consumer (8GB VRAM)**: 一般的なPC向け。
    *   **Server (24GB+ VRAM)**: ハイエンドGPU向け。
    *   ここを選ぶだけで、最適なモデルサイズ設定（Dim, Layers）が読み込まれます。
2.  **▶ Start Training**:
    *   学習を開始します。
    *   **Graph**: 中央のグラフで **Loss (損失)** が下がっていくのを見守ります。
        *   Loss > 5.0: まだ言葉を話せません。
        *   Loss < 3.0: 文法を覚え始めます。
        *   Loss < 2.0: かなり自然な文章になります。
3.  **🛑 STOP (Save)**:
    *   いつでも中断できます。データは自動的に保存されます。

### 4. Inference (推論・対話)
育てたAIと会話します。

1.  **Load Model**:
    *   `models` フォルダから `.safetensors` ファイルを選びます。
    *   **注意**: 必ず **学習に使ったのと同じ Tokenizer** が同じフォルダにある必要があります。
2.  **Chat**:
    *   下の入力欄にメッセージを入れて Enter で送信！
    *   Parameters (Temperture/Top-P) で創造性を調整できます。

---

## ❓ トラブルシューティング (Troubleshooting)

### Q. 生成される文章が文字化けする・意味不明 (Gibberish Output)
**A. まだ学習不足です。**
*   "こんにちは" -> "エ seven数 more..." のようになる場合、AIはまだ言葉の意味を理解していません。
*   **対策**:
    1.  もう少し長く学習 (`Steps` を増やす) させてください。Loss が **3.0以下** になるのが目安です。
    2.  データ量が少なすぎる場合、同じデータを何度も学習してしまい「過学習」または「崩壊」することがあります。もっとテキストデータを増やしてください。
    3.  **注意**: Tokenizer を再作成した場合は、**必ず Preprocess と Training も最初からやり直してください**。古いモデルと新しい辞書は混ぜられません。

### Q. Training が GPU で動かない (0.00 MB / Cpu)
**A. CUDAドライバを確認してください。**
*   NVIDIA の GPU が必要です。最新のドライバをインストールしてください。
*   Bit-Llama Studio は現在 NVIDIA GPU 専用です。

### Q. "Stream did not contain valid UTF-8" エラーについて
**A. 修正済みです (v0.1.0以降)。**
*   古いバージョンでは発生していましたが、現在は自動的に修復されます。

---
**Happy Hacking!**
Bit-TTT Team
