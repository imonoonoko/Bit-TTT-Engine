import torch
import argparse
import time
import json
import os
from safetensors.torch import save_file, load_file

def quantize_158bit(tensor):
    scale = tensor.abs().mean()
    if scale == 0:
        return torch.zeros_like(tensor, dtype=torch.uint8), 0.0
    quantized = torch.clamp(torch.round(tensor / scale), -1, 1)
    return quantized.to(torch.uint8), scale.item()

def adaptive_decompose_greedy(weight_tensor, max_bases=3, threshold=0.02):
    """
    Decomposes weights into multiple bases and scales.
    Returns: List of (quantized_tensor, scale)
    """
    resid = weight_tensor.float().clone()
    bases = []

    initial_std = weight_tensor.std().item()

    for i in range(max_bases):
        current_error = resid.pow(2).mean().sqrt().item()
        current_rel_error = current_error / (initial_std + 1e-9)

        if i > 0 and current_rel_error < threshold:
            break

        quantized, scale = quantize_158bit(resid)
        bases.append((quantized, scale))

        # CRITICAL: quantized is uint8, but values represent int8 (-1, 0, 1).
        # uint8 255 = int8 -1. Must interpret as signed for correct reconstruction.
        quantized_signed = quantized.to(torch.int8).float()  # 255 -> -1
        recon_part = quantized_signed * scale
        resid = resid - recon_part

    return bases

def pack_interleaved_blocks(bases_list, target_bases=3, device="cpu"):
    """
    Packs multiple quantization bases into a single Interleaved Block Tensor.
    """
    if not bases_list:
        return None, []

    # Pad bases_list to target_bases
    current_bases = len(bases_list)
    if current_bases < target_bases:
        example_tensor = bases_list[0][0]
        # Use uint8 0? Or int8 0?
        # My script now uses uint8 logic. 0 maps to 0. (uint8 0 -> 0).
        # We need zero tensor.
        zero_tensor = torch.zeros_like(example_tensor)
        zero_scale = 0.0
        for _ in range(target_bases - current_bases):
            bases_list.append((zero_tensor, zero_scale))

    # Truncate if too many? (Should not happen if decompose respects max)
    bases_list = bases_list[:target_bases]

    num_bases = len(bases_list)
    scales = [b[1] for b in bases_list]

    # Check shapes
    h, w = bases_list[0][0].shape
    assert w % 4 == 0, f"Width {w} must be divisible by 4 for block packing."

    # Stack bases: [NumBases, H, W]
    stack = torch.stack([b[0] for b in bases_list]).to(device)

    # Reshape to blocks of 4
    # [NumBases, H, W/4, 4]
    stack_blocked = stack.view(num_bases, h, w // 4, 4)

    # Permute to Interleave
    # Desired: [H, W/4, NumBases, 4]
    interleaved = stack_blocked.permute(1, 2, 0, 3).contiguous()

    return interleaved, scales

def convert_model(model_path, output_path, bases=3, threshold=0.02):
    print(f"Loading {model_path}...")
    st = load_file(model_path)
    new_st = {}

    for key, tensor in st.items():
        if "weight" in key and tensor.dim() == 2 and "embed_tokens" not in key and "lm_head" not in key:
            print(f"Processing {key} {tensor.shape}...")

            decomposed = adaptive_decompose_greedy(tensor, max_bases=bases, threshold=threshold)

            # Pack with padding
            packed_tensor, scales = pack_interleaved_blocks(decomposed, target_bases=bases)

            # Save as uint8 directly - NO F32 BIT PACKING
            # Shape: [Out, In/4, NumBases, 4] uint8
            # Ensure it's uint8
            packed_u8 = packed_tensor.to(torch.uint8)

            # Save Names
            # We rename 'weight' to 'weight_packed' to signal adaptive format
            base_key = key.replace(".weight", "")
            new_st[f"{base_key}.weight_packed"] = packed_u8

            # Save Scales as a Tensor
            new_st[f"{base_key}.scales"] = torch.tensor(scales, dtype=torch.float32)

            # Metadata for this layer (Num Bases) - implicit in scales len or packed shape
            print(f"  -> {len(decomposed)} bases. Packed shape: {packed_u8.shape} dtype: {packed_u8.dtype}")

        else:
            # Copy other tensors (biases, norms, embeddings?)
            # Adjust embeddings? Embeddings should probably perform standard copy or simple quantization?
            # For now, copy non-2D-weights as is.
            print(f"Copying {key}...")
            new_st[key] = tensor

    print(f"Saving to {output_path}...")
    save_file(new_st, output_path)
    print("Done.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Input .safetensors file")
    parser.add_argument("--out", type=str, required=True, help="Output .safetensors file")
    parser.add_argument("--bases", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.05)
    args = parser.parse_args()

    convert_model(args.model, args.out, args.bases, args.threshold)

if __name__ == "__main__":
    main()
