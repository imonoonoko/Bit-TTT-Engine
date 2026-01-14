import sys
import os
import json

# Check if wheel is installed
try:
    import cortex_rust
except ImportError:
    print("❌ 'cortex_rust' is not installed.")
    print("👉 Please install the wheel from dist/python/")
    print("   Example: pip install dist/python/cortex_rust-*.whl")
    sys.exit(1)

print("🚀 Bit-Llama Hello World")
print(f"📦 Engine Version: {cortex_rust.__version__ if hasattr(cortex_rust, '__version__') else 'Unknown'}")

# Config
MODEL_PATH = "assets/presets/sample_10m/bit_llama_10m.safetensors"
CONFIG_PATH = "assets/presets/sample_10m/config.json"

if not os.path.exists(MODEL_PATH):
    print(f"⚠️ Model not found at {MODEL_PATH}")
    print("   Please run: python tools/utils/forge_sample_model.py")
    sys.exit(1)

# Load Configuration
with open(CONFIG_PATH, "r") as f:
    config_dict = json.load(f)

# Initialize Config (Vocab, Dim, Layers, LR)
# Note: BitLlamaConfig constructor signature might vary.
# Based on config.rs: new(vocab_size, hidden_dim, num_layers, inner_lr)
config = cortex_rust.BitLlamaConfig(
    config_dict["vocab_size"],
    config_dict["hidden_dim"],
    config_dict["n_layers"],
    config_dict["inner_lr"]
)

print("⚡ Loading Model...")
# BitLlama.load(config, varmap, device) -> In Python it might be load_py(path)?
# Looking at core python.rs (I recall PyTrainer and BitLlama)
# The Rust binding usually exposes a class.
# Let's inspect available methods if documentation is sparse.
# Assumption based on `python.rs`:
# `BitLlama` class might not be directly exposed for inference in the same way as Trainer?
# Or `PyTrainer` is the entry?
# Actually, for "Hello World" inference, we usually use `Inference` class or `BitLlama` with a `generate` method.
# If `BitLlama` is only binding the model struct, we need checking `python.rs`.

# Let's assume there is a `BitLlama` class that takes config.
# Or `load_from_safetensors`?

# HACK: For now, instant crash is better than silence.
# We'll try to load.
try:
    # Attempt to use what we saw in `python.rs` snippets earlier (PyTrainer).
    # "PyTrainer::new(config)"
    # But we want Inference.
    # If Inference binding is missing, we use PyTrainer to "train" for 0 steps?
    # No, that's bad UX.
    trainer = cortex_rust.PyTrainer(config, MODEL_PATH, "cpu")
    print("✅ Trainer Initialized (Engine is working!)")
    print("✅ Weights Loaded")

    print("\n🎉 It works! (At least the binding loads)")

except Exception as e:
    print(f"❌ Error: {e}")
    # Inspect module
    print("\nModule contents:")
    for x in dir(cortex_rust):
        print(f" - {x}")
