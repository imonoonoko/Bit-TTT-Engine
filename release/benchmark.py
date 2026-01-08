import ctypes
import time
import os
import sys
import random
import math

# --- 1. Load DLL ---
import platform
system = platform.system()
if system == "Windows":
    lib_name = "Bit_TTT.dll"
elif system == "Darwin":
    lib_name = "libBit_TTT.dylib"
else:
    lib_name = "libBit_TTT.so"

# Locate the library
# Try current directory first, then target/release
possible_paths = [
    os.path.join(os.path.dirname(__file__), lib_name),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../target/release", lib_name))
]

dll_path = None
for p in possible_paths:
    if os.path.exists(p):
        dll_path = p
        break

if not dll_path:
    print(f"Error: {lib_name} not found in current dir or ../target/release.")
    print("Please build the project first: `cargo build --release`")
    sys.exit(1)

print(f"Loading DLL from: {dll_path}")

try:
    lib = ctypes.CDLL(dll_path)
except OSError as e:
    print(f"Error loading {lib_name}: {e}")
    sys.exit(1)

# --- 2. Define C Arguments ---
# void* ttt_create(size_t hidden_dim, float inner_lr);
lib.ttt_create.argtypes = [ctypes.c_size_t, ctypes.c_float]
lib.ttt_create.restype = ctypes.c_void_p

# void ttt_forward(void* model_ptr, const float* input_ptr, size_t seq_len, float* output_ptr);
lib.ttt_forward.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_float)
]
lib.ttt_forward.restype = None

# void ttt_destroy(void* model_ptr);
lib.ttt_destroy.argtypes = [ctypes.c_void_p]
lib.ttt_destroy.restype = None

class RustTTT:
    def __init__(self, dim, lr=0.1):
        self.dim = dim
        self.ptr = lib.ttt_create(dim, lr)
    
    def forward(self, input_list):
        seq_len = len(input_list) // self.dim
        input_array = (ctypes.c_float * len(input_list))(*input_list)
        output_array = (ctypes.c_float * len(input_list))()
        
        lib.ttt_forward(self.ptr, input_array, seq_len, output_array)
        
        # Convert back to list (optional, for verification)
        return list(output_array)
        
    def __del__(self):
        if hasattr(self, 'ptr') and self.ptr:
            lib.ttt_destroy(self.ptr)
            self.ptr = None

# --- 3. Verification ---
def run_verification():
    print("\n--- 1. Verification Test (Memory Effect) ---")
    dim = 64
    model = RustTTT(dim, 0.005) # Lower LR for stability
    
    # Create 3 random vectors (simulated embeddings)
    vocab = [[random.uniform(-1, 1) for _ in range(dim)] for _ in range(3)]
    
    # Sequence: 0, 1, 2, 0, 1, 2
    indices = [0, 1, 2, 0, 1, 2]
    
    # Flatten input for batch processing
    input_flat = []
    for idx in indices:
        input_flat.extend(vocab[idx])
        
    print(f"Feeding sequence: {indices} (Batch processing)")
    
    # Run Inference on the FULL sequence so state persists/evolves
    output_flat = model.forward(input_flat)
    
    # Reconstruct output vectors
    outputs = []
    for t in range(len(indices)):
        start = t * dim
        end = start + dim
        outputs.append(output_flat[start:end])
        
    print("Step | Token | Output Delta (||Y - X||)")
    print("------------------------------------------")
    
    deltas_0 = []
    
    for t, idx in enumerate(indices):
        x = vocab[idx]
        y = outputs[t]
        
        # Calculate Delta: ||Y - X||
        # Since Y = X + Residual, Delta = ||Residual||
        # At t=0, state=0, so Residual=0 -> Delta=0
        # At t=3, state!=0, Residual!=0 -> Delta>0
        delta = math.sqrt(sum((yi - xi)**2 for yi, xi in zip(y, x)))
        
        if idx == 0:
            deltas_0.append(delta)
            
        print(f"{t:4} | {idx:5} | {delta:.6f}")

    if len(deltas_0) >= 2:
        d1 = deltas_0[0]
        d2 = deltas_0[1]
        print(f"\nDelta(0) 1st: {d1:.6f}")
        print(f"Delta(0) 2nd: {d2:.6f}")
        
        if d1 < 1e-4 and d2 > 0.001:
             print("✅ [SUCCESS] Model state evolved (Delta increased from 0).")
             print("   (Note: First output is COPY of input because state starts at 0.")
             print("    Second output includes learned signal from previous tokens.)")
        else:
             print("❌ [FAIL] State did not change.")

# --- 4. Benchmark ---
def run_benchmark():
    print("\n--- 2. Speed Benchmark (Rust Engine) ---")
    dim = 128
    seq_len = 1000 # 1000 tokens
    
    # Generate random data
    input_data = [random.uniform(-1, 1) for _ in range(dim * seq_len)]
    
    model = RustTTT(dim, 0.1)
    
    # Warmup
    model.forward(input_data[:dim*10])
    
    start_time = time.time()
    
    # Run Inference on 1000 tokens
    _ = model.forward(input_data)
    
    end_time = time.time()
    elapsed = end_time - start_time
    tps = seq_len / elapsed
    
    print(f"Processed {seq_len} tokens in {elapsed:.4f} sec")
    print(f"⚡ Speed: {tps:.2f} Tokens/Sec (TPS)")
    print("(Note: This includes Python<->C overhead. Internal speed is faster.)")

if __name__ == "__main__":
    run_verification()
    run_benchmark()
