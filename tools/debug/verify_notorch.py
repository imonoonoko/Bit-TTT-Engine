import sys
import os
# import torch
from tokenizers import Tokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cortex_rust
from cortex_rust import BitLlamaConfig, BitLlama, ModelArch

import argparse
import json

def verify_notorch():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True, help="Path to converted model directory")
    args = parser.parse_args()

    model_dir = args.model_dir
    local_json = os.path.join(model_dir, "tokenizer.json")
    tokenizer = None

    if os.path.exists(local_json):
        print(f"⚙️ Loading Tokenizer from {local_json}...")
        try:
            tokenizer = Tokenizer.from_file(local_json)
        except Exception as e:
            print(f"⚠️ Failed to load local tokenizer: {e}")

    if tokenizer is None:
        print("⚠️ Local tokenizer not found/failed. Downloading Llama-3 tokenizer from HF (unsloth/llama-3-8b-Instruct)...")
        try:
            tokenizer = Tokenizer.from_pretrained("unsloth/llama-3-8b-Instruct")
        except Exception as e:
            print(f"❌ Failed to load tokenizer from HF: {e}")
            return

    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, 'r') as f:
        cfg_json = json.load(f)

    # Support BOTH HuggingFace and Bit-TTT key names
    vocab_size = cfg_json.get("vocab_size", 32000)
    hidden_dim = cfg_json.get("hidden_dim") or cfg_json.get("hidden_size", 4096)
    num_layers = cfg_json.get("num_layers") or cfg_json.get("num_hidden_layers", 32)
    n_heads = cfg_json.get("n_heads") or cfg_json.get("num_attention_heads", 32)
    n_kv_heads = cfg_json.get("n_kv_heads") or cfg_json.get("num_key_value_heads", 8)
    intermediate_dim = cfg_json.get("intermediate_dim") or cfg_json.get("intermediate_size", hidden_dim * 4)
    rope_theta = cfg_json.get("rope_theta", 10000.0)
    max_pos = cfg_json.get("max_position_embeddings", 8192)
    n_gpu = cfg_json.get("n_gpu_layers", 0)
    lm_head_cpu = cfg_json.get("lm_head_cpu", False)

    print("⚙️  Loading Converted Config (NoTorch)...")
    print(f"   Vocab: {vocab_size}")
    print(f"   Hidden: {hidden_dim}")
    print(f"   Layers: {num_layers}")
    print(f"   n_heads: {n_heads}")
    print(f"   n_kv_heads: {n_kv_heads}")
    print(f"   intermediate_dim: {intermediate_dim}")
    print(f"   n_gpu_layers: {n_gpu}")
    print(f"   lm_head_cpu: {lm_head_cpu}")

    # Instantiate with 4 required args (inner_lr is dummy for inference)
    config = BitLlamaConfig(vocab_size, hidden_dim, num_layers, 0.0)

    # Set properties manually via setters
    config.n_heads = n_heads
    config.n_kv_heads = n_kv_heads
    config.rope_theta = rope_theta
    config.n_gpu_layers = int(n_gpu)
    config.lm_head_cpu = bool(lm_head_cpu)
    config.intermediate_dim = intermediate_dim
    config.max_position_embeddings = max_pos
    config.arch = ModelArch.Llama  # Important: Use Llama architecture, not TTT

    print("🚀 Loading BitLlama (Hybrid Engine via Candle)...")
    model_path = os.path.join(model_dir, "model.safetensors")
    model = BitLlama(config, model_path)

    prompt = "Hello, my name is"
    print(f"📝 Prompt: {prompt}")

    ids = tokenizer.encode(prompt).ids
    print(f"🔢 Input IDs: {ids}")

    # Generate
    print("🤖 Generating (Max 20 tokens)...")
    try:
        output_ids = model.generate_tokens(ids, 20)
        decoded = tokenizer.decode(output_ids, skip_special_tokens=True)
        print(f"✅ Output: {decoded}")
    except Exception as e:
        print(f"❌ Generation Failed: {e}")

if __name__ == "__main__":
    verify_notorch()
