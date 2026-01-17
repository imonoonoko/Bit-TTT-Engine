import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cortex_rust
from cortex_rust import BitLlamaConfig, BitLlama, ModelArch

import argparse

def verify_conversion():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True, help="Path to converted model directory")
    parser.add_argument("--tokenizer", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="HF Repo for tokenizer")
    args = parser.parse_args()

    model_dir = args.model_dir
    model_path = os.path.join(model_dir, "model.safetensors")
    config_path = os.path.join(model_dir, "config.json")

    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return

    print("⚙️  Loading Converted Config...")
    import json
    with open(config_path, 'r') as f:
        js = json.load(f)

    print(f"   Arch: {js.get('arch')}")
    print(f"   Vocab: {js.get('vocab_size')}")

    # Rust Config Init
    conf = BitLlamaConfig(
        js['vocab_size'],
        js['hidden_dim'],
        js['num_layers'],
        1e-4
    )
    conf.n_heads = js['n_heads']
    conf.n_kv_heads = js['n_kv_heads']
    conf.intermediate_dim = js['intermediate_dim']
    conf.arch = ModelArch.Llama

    # RoPE Config (Safe Get)
    if 'rope_theta' in js:
        conf.rope_theta = js['rope_theta']
    if 'max_position_embeddings' in js:
        conf.max_position_embeddings = js['max_position_embeddings']

    # Patch: Hybrid Offloading
    if 'n_gpu_layers' in js:
        print(f"🔹 [HYBRID] Forcing n_gpu_layers = {js['n_gpu_layers']}")
        conf.n_gpu_layers = js['n_gpu_layers']

    print(f"🚀 Loading BitLlama (Hybrid Engine)...")
    try:
        model = BitLlama(conf, model_path, "cuda")
    except Exception as e:
        print(f"❌ Rust Load Failed: {e}")
        return

    print("✅ Model Loaded Successfully!")

    # Inference Test
    from transformers import AutoTokenizer
    # Try local tokenizer first, then fallback to argument
    tok_path = model_dir if os.path.exists(os.path.join(model_dir, "tokenizer.json")) else args.tokenizer
    print(f"📚 Loading Tokenizer from: {tok_path}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(tok_path)
    except Exception as e:
        print(f"⚠️ Tokenizer load failed ({e}), falling back to TinyLlama")
        tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    prompt = "The future of AI is"
    ids = tokenizer.encode(prompt)
    print(f"📝 Prompt: {prompt!r}")

    print("   Generating...")
    # Generate 20 tokens
    gen_ids = model.generate_tokens(ids, 20)
    print(f"   Output: {tokenizer.decode(gen_ids)}")

if __name__ == "__main__":
    verify_conversion()
