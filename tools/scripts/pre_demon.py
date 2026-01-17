import os
import subprocess
import sys
import shutil
import time

def log(msg):
    print(f"[PRE-DEMON] {msg}")

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def create_dummy_data(data_dir):
    log("Creating dummy data...")
    # Create a tiny u32 dataset
    dummy_u32 = os.path.join(data_dir, "dummy.u32")
    # Just write some random bytes pretending to be u32 tokens
    with open(dummy_u32, "wb") as f:
        # 1000 tokens * 4 bytes
        f.write(b'\x00\x00\x00\x00' * 1000)
    return dummy_u32

def run_smoke_test():
    log("Starting Smoke Test...")

    # Paths
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_dir = os.path.join(root_dir, "target", "release")
    train_bin = os.path.join(target_dir, "bit_llama.exe")
    data_dir = os.path.join(root_dir, "data", "smoke_test")
    model_dir = "pre_demon_test"
    if os.path.exists(model_dir):
        print(f"[PRE-DEMON] Cleaning up {model_dir}...")
        try:
             shutil.rmtree(model_dir)
        except Exception as e:
             print(f"[PRE-DEMON] Warning: Cleanup failed: {e}")
    os.makedirs(model_dir, exist_ok=True)

    ensure_dir(data_dir)
    ensure_dir(model_dir)

    # Clean old models
    # The above rmtree handles this, but if model_dir was changed to something else,
    # this loop would still be relevant for the old model_dir.
    # Given the change, this loop is now redundant for the new model_dir.
    # However, the instruction did not ask to remove it, so it stays.
    for f in os.listdir(model_dir):
        os.remove(os.path.join(model_dir, f))

    # Clean root checkpoints that might interfere (auto-resume)
    for f in ["bit_llama_checkpoint.safetensors", "training_state.json"]:
        p = os.path.join(root_dir, f)
        if os.path.exists(p):
            log(f"Removing interfering file: {f}")
            try:
                os.remove(p)
            except Exception as e:
                log(f"Warning: Failed to remove {f}: {e}")

    # 1. Create Data
    data_file = create_dummy_data(data_dir)

    # 2. Compile (Ensure latest binary)
    log("Compiling bit_llama...")
    subprocess.check_call(["cargo", "build", "--release", "--bin", "bit_llama"], cwd=root_dir)

    # 3. Run Training (CPU, 10 steps)
    log("Running dry-run training...")
    # Define variables for the new command structure
    dim = 64
    layers = 2
    context_len = 16
    vocab_size = 256 # Assuming a default vocab size for dummy data
    cmd = [
        train_bin,
        "train",  # Subcommand
        "--dim", str(dim),
        "--layers", str(layers),
        "--context-len", str(context_len),
        "--batch-size", "2",
        "--steps", "10",
        "--data", data_file,
        "--output-dir", os.path.abspath(model_dir),
        "--save-interval", "5"
    ]

    start_time = time.time()
    try:
        # Set environment to force CPU if needed, though arguments usually control it.
        # But here we don't pass --features cuda to cargo build, so it should be CPU default depending on Cargo.toml
        # Actually Cargo.toml has default = ["python"], cuda is optional.
        # Verification: we are running the binary directly.

        proc = subprocess.run(cmd, cwd=root_dir, check=False, capture_output=True, text=True)

        if proc.returncode != 0:
            log("❌ Training FAILED")
            print(proc.stderr)
            sys.exit(1)
        else:
            log("✅ Training process finished successfully")
            print(proc.stdout)

    except Exception as e:
        log(f"❌ Execution Exception: {e}")
        sys.exit(1)

    # 4. Verify Checkpoints
    log("Verifying output artifacts...")
    checkpoints = [f for f in os.listdir(model_dir) if f.endswith(".safetensors")]
    if len(checkpoints) > 0:
        log(f"✅ Found {len(checkpoints)} checkpoints: {checkpoints}")
    else:
        log("❌ No checkpoints found!")
        sys.exit(1)

    log("🎉 SMOKE TEST PASSED!")

if __name__ == "__main__":
    run_smoke_test()
