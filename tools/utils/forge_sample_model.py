import torch
import torch.nn as nn
import json
import os
import argparse
from safetensors.torch import save_file

def forge_model(output_path, vocab_size=1000, dim=256, layers=4):
    """
    Forges a "Dummy" BitLlama model with random weights.
    Structure follows strict BitNet b1.58 requirements.
    """
    print(f"🔨 Forging BitLlama-10M (Vocab={vocab_size}, Dim={dim}, Layers={layers})...")

    tensors = {}

    # Embedding (vocab, dim)
    # BitLlama expects "embed.weight"
    tensors["embed.weight"] = torch.randn(vocab_size, dim) * 0.02

    # Layers
    for i in range(layers):
        # TTT (Test Time Training) Layer - Replaces Self Attn
        # Projections: Down (dim -> d_small), Up (d_small -> dim)
        d_small = dim // 4
        tensors[f"layers.{i}.ttt.down.weight"] = torch.randn(d_small, dim) * 0.02
        tensors[f"layers.{i}.ttt.up.weight"] = torch.randn(dim, d_small) * 0.02

        # Norms
        tensors[f"layers.{i}.norm1.weight"] = torch.ones(dim)
        tensors[f"layers.{i}.norm2.weight"] = torch.ones(dim)

        # MLP (SwiGLU)
        # w1(gate), w2(down), w3(up) -> stored as gate_proj, down_proj, up_proj
        hidden = dim * 4
        tensors[f"layers.{i}.mlp.gate_proj.weight"] = torch.randn(hidden, dim) * 0.02
        tensors[f"layers.{i}.mlp.down_proj.weight"] = torch.randn(dim, hidden) * 0.02
        tensors[f"layers.{i}.mlp.up_proj.weight"] = torch.randn(hidden, dim) * 0.02

    # Final Norm
    tensors["norm_f.weight"] = torch.ones(dim)
    # LM Head
    tensors["lm_head.weight"] = torch.randn(vocab_size, dim) * 0.02

    # Save Safetensors
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_file(tensors, output_path)
    print(f"✅ Model weights saved to: {output_path}")

    # Save Config
    # config.rs expects:
    # vocab_size, hidden_dim, num_layers (alias n_layers), inner_lr
    config = {
        "vocab_size": vocab_size,
        "hidden_dim": dim,
        "n_layers": layers, # "n_layers" alias for num_layers
        "inner_lr": 0.0003,
        "n_gpu_layers": None,
        "model_type": "bit_llama"
    }

    config_file = os.path.join(os.path.dirname(output_path), "config.json")
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)
    print(f"✅ Config saved to: {config_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="assets/presets/sample_10m/bit_llama_10m.safetensors")
    args = parser.parse_args()
    forge_model(args.out)
