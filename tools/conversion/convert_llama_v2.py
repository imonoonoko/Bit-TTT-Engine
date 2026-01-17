import torch
import torch.nn as nn
from safetensors.torch import save_file, load_file
from huggingface_hub import snapshot_download
import argparse
import os
import json
from tqdm import tqdm
import gc

# Constants
TARGET_ARCH = "llama"
INTERMEDIATE_DIM_RATIO = 4 # Default for Llama (but usually it's from config)

def download_model(model_id: str, local_dir: str):
    """
    Downloads model weights and config from HuggingFace.
    """
    print(f"📥 Downloading {model_id} to {local_dir}...")
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        allow_patterns=["*.json", "*.safetensors", "tokenizer.*"],
        ignore_patterns=["*.bin", "*.pth"]
    )
    print("✅ Download complete.")

def quantize_adaptive_basis(weight: torch.Tensor, n_bases: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Decomposes an FP16/BF16 weight matrix into N ternary bases using Residual Vector Quantization (RVQ).

    Args:
        weight: [Out, In] Tensor
        n_bases: Number of bases (K)

    Returns:
        bases: [NumBases, Out, In] (int8: -1, 0, 1)
        scales: [NumBases] (float)
    """
    device = weight.device
    residual = weight.clone().float()

    bases_list = []
    scales_list = []

    for _ in range(n_bases):
        # 1. Calculate Scale (Mean Abs or L1 Norm)
        # alpha = mean(|R|)
        scale = residual.abs().mean()

        # Avoid div by zero
        if scale < 1e-9:
            scale = torch.tensor(1.0, device=device)

        # 2. Quantize
        # Q = Round(R / scale) clamped to [-1, 1]
        scaled = residual / scale
        quantized = torch.clamp(torch.round(scaled), -1, 1).to(torch.int8)

        # 3. Update Residual
        # R = R - scale * Q
        recon = quantized.float() * scale
        residual = residual - recon

        bases_list.append(quantized)
        scales_list.append(scale)

    # Stack
    bases = torch.stack(bases_list, dim=0) # [NumBases, Out, In]
    scales = torch.stack(scales_list, dim=0) # [NumBases]

    return bases, scales

def pack_interleaved(bases: torch.Tensor) -> torch.Tensor:
    """
    Packs ternary bases into the Rust-compatible layout.

    Args:
        bases: [NumBases, Out, In] (int8)

    Returns:
        packed: [Out, In/4, NumBases, 4] (uint8: 0, 1, 255)
    """
    n_bases, out_dim, in_dim = bases.shape

    if in_dim % 4 != 0:
        raise ValueError(f"Input dimension {in_dim} must be divisible by 4 for packing.")

    # 1. Map values to uint8
    # -1 -> 255, 0 -> 0, 1 -> 1
    # Create uint8 tensor
    bases_u8 = torch.zeros_like(bases, dtype=torch.uint8)
    bases_u8[bases == 1] = 1
    bases_u8[bases == -1] = 255
    bases_u8[bases == 0] = 0

    # 2. Permute to [Out, In, Bases]
    # bases_u8: [Bases, Out, In] -> [Out, In, Bases]
    tmp = bases_u8.permute(1, 2, 0)

    # 3. Reshape In -> [In/4, 4]
    # [Out, In/4, 4, Bases]
    tmp = tmp.reshape(out_dim, in_dim // 4, 4, n_bases)

    # 4. Permute to [Out, In/4, Bases, 4]
    # (0, 1, 3, 2)
    packed = tmp.permute(0, 1, 3, 2).contiguous()

    return packed

def pack_2bit(bases: torch.Tensor) -> torch.Tensor:
    """
    Packs ternary bases into 2-bit packed uint8.

    Args:
        bases: [NumBases, Out, In] (int8)

    Returns:
        packed: [Out, In/4, NumBases] (uint8)
    """
    n_bases, out_dim, in_dim = bases.shape

    if in_dim % 4 != 0:
        raise ValueError(f"Input dimension {in_dim} must be divisible by 4 for packing.")

    # 1. Permute to [Out, In, Bases]
    # [NumBases, Out, In] -> [Out, In, NumBases]
    tmp = bases.permute(1, 2, 0).contiguous()

    # 2. Reshape In -> [In/4, 4]
    # [Out, In/4, 4, NumBases]
    tmp = tmp.reshape(out_dim, in_dim // 4, 4, n_bases)

    # 3. Map values to 2-bit representation
    # -1 -> 2 (10), 0 -> 0 (00), 1 -> 1 (01)
    # math: (val + 3) % 3
    tmp_mapped = (tmp.to(torch.int16) + 3) % 3
    tmp_mapped = tmp_mapped.to(torch.uint8)

    # 4. Pack 4 elements into 1 byte
    # Dims: 0=Out, 1=In/4, 2=4, 3=Bases
    # Collapse dim 2.
    b0 = tmp_mapped[:, :, 0, :]
    b1 = tmp_mapped[:, :, 1, :]
    b2 = tmp_mapped[:, :, 2, :]
    b3 = tmp_mapped[:, :, 3, :]

    packed = b0 | (b1 << 2) | (b2 << 4) | (b3 << 6)

    # Result Shape: [Out, In/4, NumBases]
    return packed

def convert_layer(layer_prefix: str, state_dict: dict, config: dict, n_bases: int) -> dict:
    """
    Converts a single transformer layer (Attention + MLP).
    """
    converted = {}

    # Process each linear layer in the block
    # Q, K, V, O, Gate, Up, Down
    # ...

    return converted

def main():
    parser = argparse.ArgumentParser(description="Convert Llama-3 to Universal Bit-Engine Format")
    parser.add_argument("--model-id", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="HF Repo ID")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--n-bases", type=int, default=3, help="Number of bit bases (default: 3)")
    parser.add_argument("--download", action="store_true", help="Download model first")
    parser.add_argument("--device", type=str, default="cpu", help="Device for quantization (cpu/cuda)")

    args = parser.parse_args()

    # Hardware Check
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️ CUDA requested but torch.cuda.is_available() is False. Falling back to CPU.")
        args.device = 'cpu'

    # 1. Download
    # 1. Download or Local Path
    if os.path.isdir(args.model_id):
        model_dir = args.model_id
        print(f"📂 Using local model directory: {model_dir}")
    else:
        model_dir = os.path.join("models", args.model_id.split("/")[-1])
        if args.download:
            download_model(args.model_id, model_dir)
        elif not os.path.exists(model_dir):
            print(f"⚠️ Model directory {model_dir} not found. Use --download.")
            return

    # 2. Config & Init
    print("⚙️ Loading configuration...")

    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        # Fallback: check parent directory
        parent_config = os.path.join(os.path.dirname(model_dir), "config.json")
        if os.path.exists(parent_config):
            print(f"⚠️ config.json not found in {model_dir}, found in parent. Using parent directory.")
            model_dir = os.path.dirname(model_dir)
            config_path = parent_config
        else:
            print(f"❌ Error: config.json not found in {model_dir}")
            return

    with open(config_path, 'r') as f:
        src_config = json.load(f)

    # Generate Dest Config
    # Normalize keys: Llama-3 often uses 'hidden_size' but some variants use 'hidden_dim'
    hidden_dim = src_config.get("hidden_size", src_config.get("hidden_dim"))
    if hidden_dim is None:
        print(f"❌ Error: Could not find 'hidden_size' or 'hidden_dim' in config.")
        return

    dst_config = {
        "arch": TARGET_ARCH,
        "vocab_size": src_config["vocab_size"],
        "hidden_dim": hidden_dim,
        "num_layers": src_config.get("num_hidden_layers", src_config.get("num_layers")),
        "n_heads": src_config.get("num_attention_heads", src_config.get("n_heads")),
        "n_kv_heads": src_config.get("num_key_value_heads", src_config.get("n_kv_heads", src_config.get("num_attention_heads"))),
        "intermediate_dim": src_config.get("intermediate_size", src_config.get("intermediate_dim")),
        "inner_lr": 0.0, # Not training
        "n_gpu_layers": None,
        "rope_theta": src_config.get("rope_theta", 10000.0),
        "max_position_embeddings": src_config.get("max_position_embeddings", 2048)
    }

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.json"), 'w') as f:
        json.dump(dst_config, f, indent=4)

    print(f"🚀 Converting Layers (Bases={args.n_bases})...")

    # 3. Conversion Loop
    output_tensors = {}

    # Identify source files
    if os.path.isdir(model_dir):
        files = [f for f in os.listdir(model_dir) if f.endswith(".safetensors")]
        # Sort to ensure processed in order (though dict key matters more)
        files.sort()
        file_paths = [os.path.join(model_dir, f) for f in files]
    else:
        file_paths = [model_dir] # Single file case

    print(f"📦 Found {len(file_paths)} source files.")

    for fp in file_paths:
        print(f"Processing {os.path.basename(fp)}...")
        with  torch.no_grad():
             # Load shard
             shard = load_file(fp) # device='cpu'

             for name, tensor in tqdm(shard.items(), desc="Quantizing"):
                 # Determine layer type
                 if "weight" in name and ("proj" in name or "linear" in name) and tensor.dim() == 2:
                     # Linear Layer -> Adaptive Quantization
                     # Exclude Head? Llama head is `lm_head`.
                     # Exclude Embed? `embed_tokens`

                     if "lm_head" in name or "embed_tokens" in name:
                         # Keep standard
                         output_tensors[name] = tensor.half() # Keep FP16
                         continue

                     # Decompose
                     # Check device
                     t_dev = tensor.to(args.device)
                     bases, scales = quantize_adaptive_basis(t_dev, args.n_bases)
                     packed = pack_2bit(bases)

                     # Store
                     base_name = name.replace(".weight", "")
                     output_tensors[f"{base_name}.weight_packed"] = packed.cpu()
                     output_tensors[f"{base_name}.scales"] = scales.cpu()

                 else:
                     # Norms, Biases, Embeddings -> Keep
                     output_tensors[name] = tensor.half()

             del shard
             gc.collect()

    print("💾 Saving model...")
    save_file(output_tensors, os.path.join(args.output_dir, "model.safetensors"))
    print("✨ Conversion Complete!")

if __name__ == "__main__":
    main()
