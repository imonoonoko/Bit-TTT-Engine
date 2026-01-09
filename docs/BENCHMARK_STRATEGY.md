# Benchmark Strategy: Bit-TTT vs Standard LLM

目的: Bit-TTT (Test-Time Training) アーキテクチャと、既存の標準的なLLM (Transformer/Attention) のトレーニング性能を比較・検証する。

## 1. 比較対象の定義

### A. VS PyTorch (Industry Standard)
最も一般的な学習環境である `Python + PyTorch + HuggingFace Transformers` との比較。
- **目的**: 実用的な速度差（Rust/Candle vs Python/PyTorch）と、VRAM使用量の比較。
- **方法**: 同等の規模（TinyLlama-Stories等）のモデルを学習するPythonスクリプトを作成し、それと `train_llama --benchmark` の数値を比較する。

### B. VS Vanilla Attention (Architectural)
同じ Rust/Candle エンジン上で、「TTTレイヤー」を「標準的なAttentionレイヤー」に差し替えたものとの比較。
- **目的**: 純粋なアルゴリズムの計算量（$O(N)$ vs $O(N^2)$）の違いを見る。
- **方法**: `impl StandardLlama` を実装し、同じデータローダーで速度を計測する。

## 2. 実装計画

### Step 1: `train_llama.rs` にベンチマークモードを追加
- `--benchmark` フラグを実装。
- ログ出力、保存、検証（Loss計算の頻度）を極力減らし、純粋な学習ループ（Forward+Backward+Update）の平均TPS (Tokens Per Second) を計測する。
- VRAM使用量の計測（可能な場合）。

### Step 2: PyTorch ベンチマークスクリプトの作成
- `bench_pytorch.py` を作成。
- 同様の構成（層数、次元数、バッチサイズ）で学習ループを回し、TPSを算出するスクリプト。

## 3. 予想される結果
- **速度**: コンテキスト長が短い(128等)場合は大きな差が出ないが、長くなるにつれて TTT ($O(N)$) が有利になるはず。
- **メモリ**: TTTはKVキャッシュを持たない（W_stateを持つ）ため、推論時は有利だが、学習時は `w_state` の保持コストがある。PyTorch版とのオーバーヘッドの差も確認する。
