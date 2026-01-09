
from safetensors.numpy import load_file
import sys

try:
    file_path = "../../Bit-TTT_Sandbox/rust_llm_test/bit_llama_checkpoint.safetensors"
    tensors = load_file(file_path)
    print(f"Keys in {file_path}:")
    for k in tensors.keys():
        print(f" - {k}: {tensors[k].shape}")
except Exception as e:
    print(f"Error: {e}")
