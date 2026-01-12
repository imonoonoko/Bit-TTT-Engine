# Bit-TTT Engine 技術仕様書

[![Featured on Orynth](https://orynth.dev/api/badge/bit-ttt-engine?theme=dark&style=default)](https://orynth.dev/projects/bit-ttt-engine)

**バージョン**: 1.0.0
**最終更新**: 2026-01-11

---

## 1. 概要

Bit-TTT Engine は以下の最先端技術を組み合わせた高性能言語モデル実装です：
- **BitNet 1.58-bit 量子化**: 三値重み {-1, 0, +1}
- **Test-Time Training (TTT)**: 従来のアテンションを置き換えるオンライン学習

## 2. コアコンポーネント

### 2.1 レイヤー (`cortex_rust::layers`)

| レイヤー | 目的 | パラメータ |
|----------|------|------------|
| `RMSNorm` | Root Mean Square 正規化 | `dim`, `eps` |
| `BitLinear` | 1.58-bit 量子化線形層 | `in_dim`, `out_dim` |
| `SwiGLU` | SiLU活性化によるゲート付きMLP | `hidden_dim`, `intermediate_dim` |
| `TTTLayer` | テスト時学習層 | `hidden_dim`, `inner_lr` |

### 2.2 モデル (`cortex_rust::model`)

| コンポーネント | 説明 |
|----------------|------|
| `BitLlamaConfig` | モデル設定 (vocab_size, hidden_dim, num_layers, inner_lr) |
| `BitLlamaBlock` | 単一Transformerブロック: Norm → TTT → Norm → MLP |
| `BitLlama` | 埋め込み + Nブロック + LMヘッドを持つ完全モデル |
| `Llama` | トークナイザーと状態管理を含む高レベルAPI |

### 2.3 学習 (`bit_llama::train`)

| モジュール | 責務 |
|------------|------|
| `args.rs` | CLI引数パース (dim, layers, lr, steps 等) |
| `checkpoint.rs` | 学習状態の永続化 (保存/読込) |
| `training_loop.rs` | コサイン学習率スケジュール付きメインループ |

## 3. データフロー

```
入力テキスト
    │
    ▼
┌─────────────────┐
│   Tokenizer     │  → トークンID [u32]
└─────────────────┘
    │
    ▼
┌─────────────────┐
│   Embedding     │  → 隠れ状態 (B, T, D)
└─────────────────┘
    │
    ▼ (× N 層)
┌─────────────────┐
│  BitLlamaBlock  │
│  ├─ RMSNorm     │
│  ├─ TTTLayer    │  → オンライン重み更新
│  ├─ RMSNorm     │
│  └─ SwiGLU      │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│   RMSNorm       │
│   LM Head       │  → ロジット (B, T, V)
└─────────────────┘
    │
    ▼
┌─────────────────┐
│   サンプリング   │  → 次のトークン
└─────────────────┘
```

## 4. ファイル形式

### 4.1 モデルチェックポイント (`.safetensors`)
標準 safetensors 形式、重み名：
- `embed.weight`
- `layers.{i}.norm1.weight`
- `layers.{i}.ttt.down.weight`
- `layers.{i}.ttt.up.weight`
- `layers.{i}.norm2.weight`
- `layers.{i}.mlp.gate_proj.weight`
- `layers.{i}.mlp.down_proj.weight`
- `layers.{i}.mlp.up_proj.weight`
- `norm_f.weight`
- `lm_head.weight`

### 4.2 設定 (`config.json`)
```json
{
  "vocab_size": 16384,
  "hidden_dim": 256,
  "num_layers": 8,
  "inner_lr": 0.1
}
```

### 4.3 ネイティブコンテナ (`.bitt`)
単一ファイル形式：
- マジック: `BITT` (4バイト)
- ヘッダ長 (u64 LE)
- ヘッダJSON (config + tokenizer)
- Safetensorsボディ

## 5. 学習パラメータ

| パラメータ | デフォルト | 説明 |
|------------|-----------|------|
| `dim` | 256 | モデル隠れ次元 |
| `layers` | 8 | Transformerブロック数 |
| `context_len` | 128 | 最大コンテキスト長 |
| `batch_size` | 16 | 学習バッチサイズ |
| `lr` | 3e-4 | ピーク学習率 |
| `warmup_steps` | 100 | LRウォームアップステップ |
| `min_lr` | 1e-5 | 最小学習率 |
| `save_interval` | 500 | チェックポイント保存頻度 |

## 6. ハードウェア要件

| 構成 | 最小VRAM | 推奨 |
|------|----------|------|
| 256-dim, 8層 | 2 GB | 4 GB |
| 512-dim, 12層 | 4 GB | 8 GB |
| 1024-dim, 24層 | 8 GB | 16 GB |

## 7. API リファレンス

### Rust API
```rust
use cortex_rust::{BitLlama, BitLlamaConfig, Llama};

// モデル読込
let llama = Llama::load_auto("models/my_model")?;

// ストリーミング補完
llama.stream_completion("こんにちは", 100, 0.8, |token| {
    print!("{}", token);
    Ok(true)
})?;
```

### Python API
```python
import cortex_rust

config = cortex_rust.BitLlamaConfig(16384, 256, 8, 0.1)
model = cortex_rust.BitLlama(config, "model.safetensors", device="cuda")
logits = model.forward(token_id=42)
```

---

*Bit-TTT Engine 技術仕様書 v1.0.0*
