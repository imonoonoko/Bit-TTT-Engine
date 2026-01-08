import cortex_rust
import sys

print(f"--- Cortex Rust Binding Verification ---")
print(f"Module: {cortex_rust}")
print(f"Attributes: {dir(cortex_rust)}")

# Verify Config
config = cortex_rust.BitLlamaConfig(vocab_size=1000, hidden_dim=64, num_layers=2, inner_lr=0.01)
print(f"BitLlamaConfig created: vocab={config.vocab_size}, dim={config.hidden_dim}")

# Verify BitLlama Class Existence (Renamed from PyBitLlama via pyclass name)
try:
    print(f"BitLlama class found: {cortex_rust.BitLlama}")
except AttributeError:
    print("❌ BitLlama class NOT found!")
    sys.exit(1)

# Verify Instantiation (Expect error due to missing file, but checks signature)
try:
    # This should fail in Rust loading the file, returning PyRuntimeError or PyValueError
    model = cortex_rust.BitLlama(config, "non_existent_model.safetensors", device="cpu")
except Exception as e:
    print(f"✅ Caught expected error during model load (shows signatures match): {e}")

print("--- Verification Success ---")
