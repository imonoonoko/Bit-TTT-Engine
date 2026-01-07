# 【Rust】Llama-3を超えろ。自作LLM「Bit-Llama」を1.58bit化＋TTTでフルスクラッチした話

## 1. はじめに：なぜ「Transformer」を捨てたのか？

世の中は Llama-3 や GPT-4 で溢れています。確かに性能は凄いです。
でも、**今のLLM、重すぎませんか？**

*   70Bモデルを動かすのに **VRAM 40GB**？（一般人には無理です）
*   文脈が長くなると **メモリが爆発**？（KV Cacheの呪い）
*   そもそも **Python & PyTorch** が重い？

「もっと軽く、もっと速く、そして **一般家庭のGPU（VRAM 8GB）で最強の知能を動かしたい**」

その執念で、最新論文の技術である **「BitNet (1.58-bit)」** と **「TTT (Test-Time Training)」** を組み合わせ、**Rust** でゼロから独自LLMエンジンを実装しました。

これは、既存のモデルをファインチューニングした話ではありません。
**脳の構造（アーキテクチャ）そのものを再発明し、家のPCに住まわせるまでの開発ログ**です。

### 🚀 既存モデルとの決定的な違い（性能比較）

「Llama-3」などの従来型Transformerと、今回作った「Bit-TTT」の比較です。
**VRAM 8GBの普通のゲーミングPC** で動かすことを前提に設計しました。

| 特徴 | 🐢 既存のTransformer (Llama-3等) | 🐇 **今回の Bit-TTT (自作)** |
| :--- | :--- | :--- |
| **重みのサイズ** | **FP16 (16bit)**<br>巨大。ロードするだけでVRAM圧迫。 | **1.58-bit ({-1, 0, 1})**<br>理論値で **1/10** に圧縮。 |
| **記憶の仕組み** | **KV Cache**<br>過去の会話を全部メモリに乗せる。<br>→ 話すほど重くなる ($O(T)$)。 | **Test-Time Training (TTT)**<br>脳の形状を変えて記憶する。<br>→ どれだけ話しても **メモリ一定 ($O(1)$)**。 |
| **計算コスト** | **行列積 (掛け算)**<br>GPUのパワーが必須。 | **加算 (足し算)**<br>行列計算が不要で爆速 (理論上)。 |
| **必要なPC** | 富豪のGPUサーバー (VRAM 24GB〜) | **あなたのノートPC** (VRAM 8GB〜) |

### 【作ったもの：Bit-Llama】
*   **アーキテクチャ**: 1.58-bit BitNet + TTT (Test-Time Training)
*   **言語/FW**: Rust / HuggingFace Candle
*   **GUI**: Tauri v2
*   **ソースコード**: [GitHub Link](https://github.com/imonoonoko/Bit-TTT-Engine)

---

## 2. 技術のコア：BitNet × TTT とは？

「Transformerの次」と言われる2つの次世代技術を合体させました。

### ① 1.58-bit Quantization (BitNet)
重みを `{-1, 0, 1}` の3値だけで表現します。
FP16（16bit）に比べてメモリ使用量が劇的に減るだけでなく、浮動小数点の掛け算が不要になるため、計算速度の向上も見込めます。

**実際のBitLinear実装（Rust/Candle）:**
行列の平均値でスケーリングし、四捨五入して `-1, 0, 1` に丸める処理を実装しています。

```rust
// core_engine.rs より抜粋
pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
    let w = &self.weight;
    // 1. Absolute Mean Quantization
    let scale = w.abs()?.mean_all()?;
    let w_scaled = (w / scale.to_scalar::<f32>()? as f64)?;
    
    // 2. Round to {-1, 0, 1}
    let w_quant = w_scaled.round()?.clamp(-1.0, 1.0)?;

    // 3. STE (Straight-Through Estimator)
    // 勾配を流すためのトリック
    let diff = (w_quant - &w_scaled)?;
    let w_ste = (diff.detach() + &w_scaled)?;

    // Linear: x @ w.T
    x.matmul(&w_ste.t()?)
}
```

### ② Test-Time Training (TTT)
ここが最大のこだわりです。Attention（過去ログの全検索）を捨てました。
代わりに、「会話しながらその場で脳のシナプス（隠れ状態）を書き換える」 ことで記憶を保持します。

*   **従来**: 教科書（過去ログ）を机に広げ続ける → 机がパンクする
*   **TTT**: 読んだ端から内容を脳に刻み、教科書は捨てる → **無限に読める**

---

## 3. Rust (Candle) による実装

Python (PyTorch) ではなく、推論速度と安全性、そして配布のしやすさ（シングルバイナリ）を重視して **Rust** を採用しました。
フレームワークは HuggingFace 純正の **Candle** です。

### 苦労した点：TTTレイヤーの自作
論文の数式をRustに落とし込むのが大変でした。特に「推論ループの中に、学習（勾配更新）を組み込む」という狂気の実装が必要になります。

```rust
// 推論中に「学習」が走る (core_engine.rs)
pub fn forward_update(&self, w_state: &Tensor, x_t: &Tensor) -> Result<(Tensor, Tensor)> {
    // 1. Project Down
    let feat = self.proj_down.forward(x_t)?;

    // 2. Predict (推論)
    // w_state (短期記憶) を使って予測
    let pred = w_state.matmul(&feat)?;

    // 3. Loss & Grad (自己教師あり学習)
    // 入力を再構成できるか？の誤差を計算
    let diff = (&pred - &feat_target)?;
    let grad = diff.matmul(&feat.t())?; 

    // 4. Update (記憶の書き込み)
    // 従来の推論では重み(w_state)は固定だが、TTTでは書き換える！
    let w_new = (w_state - grad * self.inner_lr)?;
    
    Ok((out_feat, w_new))
}
```

---

## 4. 学習：TinyStoriesで「知能」を宿す

構造だけ作っても脳は空っぽです。
GPT-4が作った幼児向けデータセット「TinyStories」を使って、実際に学習を回しました。

1.  **データ準備**: PythonスクリプトでHuggingFaceからDL＆バイナリ化
2.  **学習**: `cargo run --release --bin train_llama` でGPUをブン回す

### 結果
最初は `xjqy...` という宇宙語でしたが、Step 150を超えたあたりで...

> **Input:** "Once upon a time"
> **Output:** "I'm glad you can..."

**「言葉」を話し始めました！**
自分のPCの中で、AIが言語を獲得する瞬間を見るのは感動モノです。

(ここに学習中のログや、生成結果のスクリーンショットがあれば貼る)

---

## 5. Tauri で「身体」を与える

コマンドラインだけじゃ寂しいので、**Tauri v2** でデスクトップマスコット化しました。
バックエンド（Rustプロセス）でこのBit-TTTエンジンが走り、フロントエンド（WebView）と会話します。

(ここにマスコットアプリのスクリーンショットを貼る)

---

## 6. 将来性とロードマップ

このプロジェクトは単なる実験ではありません。**「AIの民主化」** に向けた以下の可能性を秘めています。

### 70Bクラスの「超知能」を個人PCへ
1.58-bit化により、本来VRAM 140GB必要な70Bモデルが、16GB程度のメモリで動く計算になります。
最強のAIをローカルで飼える未来がすぐそこにあります。

### 無限のコンテキスト長
TTTによりメモリ消費が増えないため、小説1冊どころか**「一生分の会話」**を記憶し続けるパートナーAIが実現可能です。

### 日本語対応 (Dolly-15k)
次は日本語データセットで学習させ、バイリンガルなマスコットへと進化させます。

---

## 7. まとめとリポジトリ

「Llama-3を使う」側から「Llamaを超えるものを作る」側へ。
Bit-TTTアーキテクチャは、個人のPCでLLMを飼うための最適解かもしれません。

ソースコードは全てGitHubで公開しています。
PR、そして**「俺ならもっと速くできるぞ」というRustaceanの挑戦**をお待ちしています！

👉 **GitHub: [Bit-TTT-Engine](https://github.com/imonoonoko/Bit-TTT-Engine)**
