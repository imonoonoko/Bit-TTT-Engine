import ctypes
import time
import os
import sys
import random
import math
import platform

# --- 1. Load Shared Library ---
system = platform.system()
if system == "Windows":
    lib_name = "Bit_TTT.dll"
elif system == "Darwin":
    lib_name = "libBit_TTT.dylib"
else:
    lib_name = "libBit_TTT.so"

# Locate the library: Try examples dir, release dir, or target/release
possible_paths = [
    os.path.join(os.path.dirname(__file__), lib_name),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../release", lib_name)),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../target/release", lib_name))
]

dll_path = None
for p in possible_paths:
    if os.path.exists(p):
        dll_path = p
        break

if not dll_path:
    print(f"Error: {lib_name} not found.")
    print("Please build the project first: `cargo build --release`")
    sys.exit(1)

print(f"Loading Engine from: {dll_path}")

try:
    lib = ctypes.CDLL(dll_path)
except OSError as e:
    print(f"Error loading {lib_name}: {e}")
    sys.exit(1)

# --- 2. Define C-API ---
# void* ttt_create(size_t hidden_dim, float inner_lr);
lib.ttt_create.argtypes = [ctypes.c_size_t, ctypes.c_float]
lib.ttt_create.restype = ctypes.c_void_p

# int ttt_forward(void* model_ptr, const float* input_ptr, size_t seq_len, float* output_ptr);
lib.ttt_forward.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_float)
]
lib.ttt_forward.restype = ctypes.c_int

# void ttt_destroy(void* model_ptr);
lib.ttt_destroy.argtypes = [ctypes.c_void_p]
lib.ttt_destroy.restype = None

# --- 3. Python Wrapper Class ---
class BitTTTEngine:
    def __init__(self, dim, lr=0.1):
        self.dim = dim
        self.ptr = lib.ttt_create(dim, lr)
    
    def forward(self, input_list):
        seq_len = len(input_list) // self.dim
        input_array = (ctypes.c_float * len(input_list))(*input_list)
        output_array = (ctypes.c_float * len(input_list))()
        
        # Call Rust Engine
        ret = lib.ttt_forward(self.ptr, input_array, seq_len, output_array)
        
        if ret != 0:
            raise RuntimeError(f"Rust Core Engine Failed with error code: {ret}")
        
        return list(output_array)
        
    def __del__(self):
        if hasattr(self, 'ptr') and self.ptr:
            lib.ttt_destroy(self.ptr)
            self.ptr = None

# --- 4. Usage Example ---
if __name__ == "__main__":
    print("\n--- Bit-TTT Python Example ---")
    
    # Configuration
    dim = 64
    seq_len = 10
    
    # Initialize Engine
    print(f"Initializing Engine (dim={dim})...")
    engine = BitTTTEngine(dim, 0.1)
    
    # Create Dummy Input (Random Vectors)
    input_data = [random.uniform(-1, 1) for _ in range(dim * seq_len)]
    
    # Run Inference
    print(f"Running Inference on {seq_len} tokens...")
    start = time.time()
    output = engine.forward(input_data)
    elapsed = time.time() - start
    
    print(f"Done in {elapsed:.4f} sec.")
    print(f"Output Shape: {len(output)} floats")
    print("Success! w_state has been updated internally.")
