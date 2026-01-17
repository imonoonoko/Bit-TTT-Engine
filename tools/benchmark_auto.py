
import time
import torch
import os
import sys
import psutil
import gc
import argparse
import json
from pynvml import *

# ==========================================
# Setup Environment
# ==========================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

def print_header(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

# Try importing cortex_rust
try:
    import cortex_rust
    from cortex_rust import BitLlama, BitLlamaConfig, ModelArch
    print("✅ [Setup] cortex_rust imported successfully.")
except ImportError:
    print("❌ [Setup] cortex_rust not found. Please ensure it is compiled and in PYTHONPATH.")
    sys.exit(1)

# Try importing transformers
try:
    from transformers import AutoTokenizer
    print("✅ [Setup] transformers imported successfully.")
except ImportError:
    print("⚠️ [Setup] transformers not found. Tokenization will fallback or fail.")

# ==========================================
# Metrics Utilities
# ==========================================
def get_vram_usage(device_index=0):
    try:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(device_index)
        info = nvmlDeviceGetMemoryInfo(handle)
        return info.used / 1024**3
    except:
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated(device_index) / 1024**3
        return 0.0

def get_ram_usage():
    return psutil.Process().memory_info().rss / 1024**3

# ==========================================
# Benchmark Logic
# ==========================================
def run_benchmark(args):
    model_dir = args.model
    if not os.path.exists(model_dir):
        print(f"❌ Model path not found: {model_dir}")
        return

    print_header(f"Benchmarking: {os.path.basename(model_dir)}")
    print(f"🔹 Config: n_gpu_layers={args.n_gpu}")
    print(f"🔹 Prompt: \"{args.prompt}\"")
    print(f"🔹 Tokens: {args.n_tokens}")

    # 1. Load Config
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        print(f"❌ config.json not found in {model_dir}")
        return

    with open(config_path, 'r') as f:
        config_json = json.load(f)

    # Map JSON to BitLlamaConfig
    vocab_size = config_json.get("vocab_size", 128256)
    hidden_dim = config_json.get("hidden_dim", 4096)
    num_layers = config_json.get("num_layers", 32)
    lm_head_cpu = config_json.get("lm_head_cpu", True) # Default to True for Hybrid safety

    print("Loading Config...")
    config = BitLlamaConfig(
        vocab_size,
        hidden_dim,
        num_layers,
        0.0, # inner_lr (unused for inference)
        lm_head_cpu
    )

    # Set properties
    arch_str = config_json.get("arch", "llama").lower()
    if arch_str == "llama":
        config.arch = ModelArch.Llama
    else:
        config.arch = ModelArch.TTT

    # Handle n_gpu_layers
    if args.n_gpu is not None:
        if args.n_gpu.lower() == "auto":
             pass # Leave as default (None) for auto
        else:
            config.n_gpu_layers = int(args.n_gpu)

    # Other params
    config.n_heads = config_json.get("num_attention_heads", 32)
    config.n_kv_heads = config_json.get("num_key_value_heads", 8)
    config.intermediate_dim = config_json.get("intermediate_size", 14336)
    config.rope_theta = config_json.get("rope_theta", 500000.0)
    config.max_position_embeddings = config_json.get("max_position_embeddings", 8192)

    # 2. Load Model
    print("🚀 Loading Model...")
    ram_before = get_ram_usage()
    start_load = time.time()

    try:
        model = BitLlama(config, os.path.join(model_dir, "model.safetensors"))
    except Exception as e:
        print(f"❌ Load Failed: {e}")
        return

    load_time = time.time() - start_load
    ram_after = get_ram_usage()
    print(f"✅ Load Complete in {load_time:.2f}s")
    print(f"📊 RAM Delta: {ram_after - ram_before:.2f} GB (Total: {ram_after:.2f} GB)")
    print(f"📊 VRAM Usage: {get_vram_usage():.2f} GB")

    # 3. Tokenize
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        tokens = tokenizer.encode(args.prompt)
    except:
        print("⚠️ Tokenizer load failed, using dummy tokens.")
        tokens = [128000, 100, 101] # Dummy

    # 4. Generate
    print(f"🏃 Generating {args.n_tokens} tokens...")
    start_gen = time.time()

    # Note: generate_tokens is assumed to be the method name in pyo3 bindings based on earlier context
    try:
        _ = model.generate_tokens(tokens, args.n_tokens)
    except AttributeError:
        # Fallback if method name is different
        print("⚠️ generate_tokens not found, trying forward loop (slower python loop)")
        # This part is complex to implement robustly in a benchmark script without clear API spec
        # Let's assume generate_tokens exists or we fail
        print("❌ API Mismatch: generate_tokens method missing.")
        return

    end_gen = time.time()
    duration = end_gen - start_gen
    tps = args.n_tokens / duration

    print(f"⚡ Speed: {tps:.2f} tokens/s")
    print(f"⏱️ Duration: {duration:.2f}s")

    # 5. Output Report
    if args.output:
        # Ensure header exists
        if not os.path.exists(args.output):
            with open(args.output, 'w') as f:
                f.write("| Model | GPU Config | Speed (TPS) | Load Time | RAM | VRAM |\n")
                f.write("|---|---|---|---|---|---|\n")

        gpu_conf = args.n_gpu if args.n_gpu else "Auto"
        with open(args.output, 'a') as f:
            f.write(f"| {os.path.basename(model_dir)} | {gpu_conf} | {tps:.2f} | {load_time:.2f}s | {ram_after:.2f} GB | {get_vram_usage():.2f} GB |\n")
        print(f"📝 Result appended to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bit-Llama Benchmark Tool")
    parser.add_argument("--model", type=str, required=True, help="Path to model directory config.json/model.safetensors")
    parser.add_argument("--prompt", type=str, default="Hello, I am a language model,", help="Prompt text")
    parser.add_argument("--n-tokens", type=int, default=20, help="Number of tokens to generate")
    parser.add_argument("--n-gpu", type=str, default=None, help="Number of GPU layers (or 'auto')")
    parser.add_argument("--output", type=str, default=None, help="Markdown file to append results")

    args = parser.parse_args()

    run_benchmark(args)
