# Bit-TTT Engine Specification

[![Featured on Orynth](https://orynth.dev/api/badge/bit-ttt-engine?theme=dark&style=default)](https://orynth.dev/projects/bit-ttt-engine)

**Version**: 1.0.0
**Last Updated**: 2026-01-11

---

## 1. Overview

Bit-TTT Engine is a high-performance language model implementation combining:
- **BitNet 1.58-bit Quantization**: Ternary weights {-1, 0, +1}
- **Test-Time Training (TTT)**: Online learning replacing traditional attention

## 2. Core Components

### 2.1 Layers (`cortex_rust::layers`)

| Layer | Purpose | Parameters |
|-------|---------|------------|
| `RMSNorm` | Root Mean Square Normalization | `dim`, `eps` |
| `BitLinear` | 1.58-bit quantized linear layer | `in_dim`, `out_dim` |
| `SwiGLU` | Gated MLP with SiLU activation | `hidden_dim`, `intermediate_dim` |
| `TTTLayer` | Test-Time Training layer | `hidden_dim`, `inner_lr` |

### 2.2 Model (`cortex_rust::model`)

| Component | Description |
|-----------|-------------|
| `BitLlamaConfig` | Model configuration (vocab_size, hidden_dim, num_layers, inner_lr) |
| `BitLlamaBlock` | Single transformer block: Norm → TTT → Norm → MLP |
| `BitLlama` | Full model with embedding, N blocks, and LM head |
| `Llama` | High-level API with tokenizer and state management |

### 2.3 Training (`bit_llama::train`)

| Module | Responsibility |
|--------|----------------|
| `args.rs` | CLI argument parsing (dim, layers, lr, steps, etc.) |
| `checkpoint.rs` | Training state persistence (save/load) |
| `training_loop.rs` | Main training loop with cosine LR schedule |

## 3. Data Flow

```
Input Text
    │
    ▼
┌─────────────────┐
│   Tokenizer     │  → Token IDs [u32]
└─────────────────┘
    │
    ▼
┌─────────────────┐
│   Embedding     │  → Hidden States (B, T, D)
└─────────────────┘
    │
    ▼ (× N layers)
┌─────────────────┐
│  BitLlamaBlock  │
│  ├─ RMSNorm     │
│  ├─ TTTLayer    │  → Online weight update
│  ├─ RMSNorm     │
│  └─ SwiGLU      │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│   RMSNorm       │
│   LM Head       │  → Logits (B, T, V)
└─────────────────┘
    │
    ▼
┌─────────────────┐
│   Sampling      │  → Next Token
└─────────────────┘
```

## 4. File Formats

### 4.1 Model Checkpoint (`.safetensors`)
Standard safetensors format with weight names:
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

### 4.2 Config (`config.json`)
```json
{
  "vocab_size": 16384,
  "hidden_dim": 256,
  "num_layers": 8,
  "inner_lr": 0.1
}
```

### 4.3 Native Container (`.bitt`)
Single-file format containing:
- Magic: `BITT` (4 bytes)
- Header length (u64 LE)
- Header JSON (config + tokenizer)
- Safetensors body

## 5. GUI Architecture (Bit-Llama Studio)

From v0.3.0 (Refactor V3), the GUI is based on the following design.

### 5.1 Tab Structure (`gui/tabs`)
| Tab | Module | Function |
|-----|--------|----------|
| **Model Lab** | `model_lab.rs` | Model Loading, Soul (.soul) Management, Sleep Control |
| **Chat** | `inference.rs` | Specialized for AI interaction (No system logs) |
| **Settings** | `settings.rs` | Temperature, System Prompt settings |

### 5.2 Event-Driven Model (`gui/mod.rs`)
- **Centralized Polling**: `poll_inference_events` executes every frame within the `update()` loop.
- **Async Communication**: Inference thread (`InferenceSession`) and GUI communicate via `mpsc::channel`.
- **State Management**: `is_dreaming` (Sleep) flag enforces exclusive control over chat input and model operations.

## 6. Soul Architecture

The specification for "Adaptive Learning and Persistence", the core of Bit-TTT.

### 6.1 Soul File (`.soul`)
A binary file serializing the trained state (hidden states) of TTT layers.
- **Format**: Rust `bincode` or `Safetensors` (Future expansion)
- **Content**: `w_states` (weight matrices) of all TTT layers
- **Dependency**: Strongly depends on the base model architecture (dimensions, layers).

### 6.2 Sleep Mode
1. **Accumulation**: User conversations are saved to `workspace/memories/YYYY-MM-DD.jsonl`.
2. **Dreaming**:
   - Triggered by `/sleep` command.
   - High-speed Replay of past conversation logs.
   - Temporarily boosts learning rate (`inner_lr`) to firmly fix short-term memory.
3. **Wake Up**:
   - After learning completes, the updated `w_states` are written to disk as a `.soul` file.

## 7. Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dim` | 256 | Model hidden dimension |
| `layers` | 8 | Number of transformer blocks |
| `context_len` | 128 | Maximum context length |
| `batch_size` | 16 | Training batch size |
| `lr` | 3e-4 | Peak learning rate |
| `warmup_steps` | 100 | LR warmup steps |
| `min_lr` | 1e-5 | Minimum learning rate |
| `save_interval` | 500 | Checkpoint save frequency |

## 8. Hardware Requirements

| Configuration | Min VRAM | Recommended |
|---------------|----------|-------------|
| 256-dim, 8-layer | 2 GB | 4 GB |
| 512-dim, 12-layer | 4 GB | 8 GB |
| 1024-dim, 24-layer | 8 GB | 16 GB |

## 9. API Reference

### Rust API
```rust
use cortex_rust::{BitLlama, BitLlamaConfig, Llama};

// Load model
let llama = Llama::load_auto("models/my_model")?;

// Stream completion
llama.stream_completion("Hello", 100, 0.8, |token| {
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

*Bit-TTT Engine Specification v1.1.0*
