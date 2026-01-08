import os
import json
import torch
from tokenizers import Tokenizer
from safetensors.torch import save_file

def create_dummy_model(output_dir="models/dummy"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Tokenizer
    print("Downloading/Saving generic tokenizer...")
    try:
        tokenizer = Tokenizer.from_pretrained("gpt2")
        tokenizer.save(os.path.join(output_dir, "tokenizer.json"))
        vocab_size = tokenizer.get_vocab_size()
        print(f"Created tokenizer.json in {output_dir} (vocab: {vocab_size})")
    except Exception as e:
        print(f"Failed to download tokenizer: {e}")
        return

    # 1. Config (Update with real vocab size)
    config = {
        "vocab_size": vocab_size,
        "hidden_dim": 64,
        "num_layers": 2,
        "inner_lr": 0.001
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # 3. Weights
    print("Generating random weights...")
    tensors = {}
    
    hidden = config["hidden_dim"]
    vocab = config["vocab_size"]
    d_small = hidden // 4
    mlp_dim = hidden * 4
    
    # Embed
    tensors["embed.weight"] = torch.randn(vocab, hidden)
    
    for i in range(config["num_layers"]):
        # Norm1
        tensors[f"layers.{i}.norm1.weight"] = torch.ones(hidden)
        
        # TTT
        tensors[f"layers.{i}.ttt.down.weight"] = torch.randn(d_small, hidden)
        tensors[f"layers.{i}.ttt.up.weight"] = torch.randn(hidden, d_small)
        
        # Norm2
        tensors[f"layers.{i}.norm2.weight"] = torch.ones(hidden)
        
        # MLP (SwiGLU)
        tensors[f"layers.{i}.mlp.gate_proj.weight"] = torch.randn(mlp_dim, hidden)
        tensors[f"layers.{i}.mlp.down_proj.weight"] = torch.randn(hidden, mlp_dim)
        tensors[f"layers.{i}.mlp.up_proj.weight"] = torch.randn(mlp_dim, hidden)

    # Final Norm & Head
    tensors["norm_f.weight"] = torch.ones(hidden)
    tensors["lm_head.weight"] = torch.randn(vocab, hidden)

    save_file(tensors, os.path.join(output_dir, "model.safetensors"))
    print(f"Created model.safetensors in {output_dir}")
    print("\nDone! You can now run:")
    print(f"cargo run --bin bit_llama --release -- --model {output_dir} --prompt \"Hello\"")

if __name__ == "__main__":
    create_dummy_model()
