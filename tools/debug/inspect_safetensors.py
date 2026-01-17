import sys
from safetensors.torch import load_file

if len(sys.argv) < 2:
    print("Usage: python inspect_safetensors.py <path>")
    sys.exit(1)

path = sys.argv[1]
try:
    tensors = load_file(path)
    for key in tensors.keys():
        print(f"{key}: {tensors[key].shape} {tensors[key].dtype}")
except Exception as e:
    print(f"Error loading {path}: {e}")
