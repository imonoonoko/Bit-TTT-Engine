import time
import torch
import os
import sys
import psutil
import gc
from threading import Thread
from pynvml import *

# Force loading local cortex_rust.pyd (not site-packages version)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# Try importing cortex_rust
try:
    import cortex_rust
    print(f"✅ cortex_rust found at: {cortex_rust.__file__}")
except ImportError:
    print("❌ cortex_rust not found. Make sure it is installed or in PYTHONPATH")
    exit(1)

# Try importing transformers
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("✅ transformers found")
except ImportError:
    print("❌ transformers not found. Please pip install transformers")
    exit(1)

def get_vram_usage():
    try:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(handle)
        return info.used / 1024**3
    except:
        return 0.0

def get_ram_usage():
    return psutil.Process().memory_info().rss / 1024**3

def benchmark_pytorch(model_path, prompt="Hello, I am a language model,", n_tokens=20):
    print(f"\n--- Benchmarking PyTorch (Original) ---")
    print(f"Path: {model_path}")

    # Force CPU for fairness/possibility (Since 8GB VRAM < 16GB Model)
    # Or try loading as much as possible? No, standard usage is CPU if OOM.
    device = "cpu"
    dtype = torch.float16 # or float32 for CPU? float16 on CPU can be slow or unsupported in some ops.
    # verification: CPU fp16 is slow. bfloat16 is better?
    # Let's use float32 for CPU speed baseline if memory allows (8B * 4 = 32GB RAM).
    # If user has 32GB RAM?
    # Safe bet: float16 on CPU if supported, else float32.

    print("Loading Model (this may take time)...")
    start_load = time.time()
    try:
        try:
            # Try standard load
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="cpu",
                low_cpu_mem_usage=True
            )
        except Exception as e:
            print(f"⚠️ Standard load failed ({e}). Trying with HF Config fallback...")
            from transformers import AutoConfig
            # Fallback to standard config for 8B-Instruct
            # Requires internet or cache
            hf_id = "meta-llama/Meta-Llama-3-8B-Instruct"
            try:
                config = AutoConfig.from_pretrained(hf_id)
                tokenizer = AutoTokenizer.from_pretrained(hf_id)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    config=config,
                    torch_dtype=torch.float16,
                    device_map="cpu",
                    low_cpu_mem_usage=True
                )
            except Exception as e2:
                 print(f"❌ Failed to recover config: {e2}")
                 return
    except Exception as e:
        print(f"❌ Failed to load PyTorch model: {e}")
        return

    load_time = time.time() - start_load
    print(f"Load Time: {load_time:.2f}s")
    print(f"RAM Usage: {get_ram_usage():.2f} GB")

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    print("Generating...")
    start_gen = time.time()
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=n_tokens, min_new_tokens=n_tokens)
    end_gen = time.time()

    duration = end_gen - start_gen
    tps = n_tokens / duration
    print(f"Speed: {tps:.2f} tokens/s")

    del model
    del tokenizer
    gc.collect()
    return tps, get_ram_usage()

def benchmark_bit_engine(model_dir, prompt="Hello, I am a language model,", n_tokens=20):
    print(f"\n--- Benchmarking Bit-Engine (Converted) ---")
    model_path = os.path.join(model_dir, "model.safetensors")
    print(f"Path: {model_path}")

    # Config for Hybrid
    from cortex_rust import BitLlama, BitLlamaConfig, ModelArch

    # Load Config manually to match verify script logic?
    # Or just hardcode optimized settings for 8GB VRAM (n_gpu=20)
    # Valid Constructor: vocab_size, hidden_dim, num_layers, inner_lr, lm_head_cpu
    config = BitLlamaConfig(
        128256, # vocab_size
        4096,   # hidden_dim
        32,     # num_layers
        0.0,    # inner_lr
        True    # lm_head_cpu
    )

    # Set properties manually
    config.arch = ModelArch.Llama
    config.n_gpu_layers = 10 # Reduced for 8GB VRAM
    config.n_heads = 32
    config.n_kv_heads = 8
    config.intermediate_dim = 14336
    config.rope_theta = 500000.0 # Llama-3 theta
    config.max_position_embeddings = 8192

    print("Loading Model...")
    start_load = time.time()
    try:
        model = BitLlama(config, model_path)
    except Exception as e:
        print(f"❌ Failed to load Bit-Engine model: {e}")
        return

    load_time = time.time() - start_load
    print(f"Load Time: {load_time:.2f}s")
    print(f"RAM Usage: {get_ram_usage():.2f} GB")
    # VRAM check requires pynvml or torch
    if torch.cuda.is_available():
        print(f"VRAM Usage: {torch.cuda.memory_allocated()/1024**3:.2f} GB ( Torch tracked)")

    print("Generating...")
    # Tokenize
    try:
        # Try finding tokenizer in model_dir
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        tokens = tokenizer.encode(prompt)
    except:
        try:
           tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
           tokens = tokenizer.encode(prompt)
        except:
           tokens = [128000, 1, 2, 3] # Dummy

    start_gen = time.time()
    _ = model.generate_tokens(tokens, n_tokens) # temp not supported (greedy)
    end_gen = time.time()

    duration = end_gen - start_gen
    tps = n_tokens / duration
    print(f"Speed: {tps:.2f} tokens/s")

    return tps, get_ram_usage()

if __name__ == "__main__":
    # Paths - Adjust as needed
    ORIGINAL_PATH = "models/Meta-Llama-3-8B"
    CONVERTED_PATH = "models/Meta-Llama-3-8B-Adaptive-2bit"

    if os.path.exists(ORIGINAL_PATH):
        # benchmark_pytorch(ORIGINAL_PATH)
        print("Skipping PyTorch benchmark as requested.")
    else:
        print(f"Skipping PyTorch: {ORIGINAL_PATH} not found")

    if os.path.exists(CONVERTED_PATH):
        benchmark_bit_engine(CONVERTED_PATH)
    else:
        print(f"Skipping Bit-Engine: {CONVERTED_PATH} not found")
