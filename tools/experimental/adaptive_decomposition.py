import torch
import torch.nn.functional as F
import argparse
import time
import math

def quantize_158bit(tensor):
    """
    Quantizes a tensor to 1.58 bits (-1, 0, 1) scaled by mean absolute value.
    Returns:
        quantized: Int8 tensor (-1, 0, 1)
        scale: Float scalar
    """
    scale = tensor.abs().mean()
    if scale == 0:
        return tensor.to(torch.int8), 0.0

    # Round to nearest integer after scaling
    quantized = torch.clamp(torch.round(tensor / scale), -1, 1).to(torch.int8)
    return quantized, scale

def adaptive_decompose(weight_tensor, max_bases=3, threshold=0.02):
    """
    Decomposes a weight tensor into multiple 1.58bit bases.
    """
    resid = weight_tensor.clone()
    bases = []

    initial_std = weight_tensor.std().item()
    print(f"Original Tensor | Std: {initial_std:.6f} | Range: [{weight_tensor.min():.4f}, {weight_tensor.max():.4f}]")

    for i in range(max_bases):
        # Measure current error (RMSE)
        current_error = resid.pow(2).mean().sqrt().item()
        current_rel_error = current_error / (initial_std + 1e-9)

        print(f"  [Base {i}] Residual RMSE: {current_error:.6f} (Rel: {current_rel_error:.2%})")

        if i > 0 and current_rel_error < threshold:
            print(f"    -> Threshold met (< {threshold:.2%}). Stopping composition.")
            break

        # Quantize current residual
        q_base, scale = quantize_158bit(resid)

        # Store component
        bases.append((q_base, scale))

        # Update residual
        # resid = resid - (q_base * scale)
        recon_part = q_base.float() * scale
        resid = resid - recon_part

    # Final check
    final_error = resid.pow(2).mean().sqrt().item()
    final_rel = final_error / (initial_std + 1e-9)
    print(f"  [End] Final RMSE: {final_error:.6f} (Rel: {final_rel:.2%}) | Bases Used: {len(bases)}")

    return bases

def benchmark_reconstruction(bases):
    """
    Reconstructs the tensor from bases and measures inference cost proxy.
    """
    if not bases:
        return torch.tensor(0.0)

    start = time.time()
    # Simulate reconstruction (In reality, this is 'accumulate' in forward pass)
    # W = sum(B_i * S_i)

    # We use Float for simple accumulation sim
    h, w = bases[0][0].shape
    recon = torch.zeros((h, w), dtype=torch.float32)

    bits_total = 0
    for q_base, scale in bases:
        recon += q_base.float() * scale
        bits_total += q_base.numel() * 2 # 2 bits theoretical (actually 1.58)

    end = time.time()

    # Size calculation
    # Base 0: 2 bits/param + 32 bit float scale
    # Raw FP16: 16 bits/param
    param_count = bases[0][0].numel()

    # Theoretical size in MB
    # Each base uses 2 bits packed (conceptually). In code we use int8 (8 bits).
    # BitNet paper assumes optimal packing.
    size_bits = param_count * 1.58 * len(bases) + (32 * len(bases)) # scales
    size_mb = size_bits / 8 / 1024 / 1024

    print(f"  -> Reconstructed in {end - start:.6f}s")
    print(f"  -> Est. Size: {size_mb:.2f} MB (vs FP16: {param_count * 2 / 1024 / 1024:.2f} MB)")

    return recon

def main():
    parser = argparse.ArgumentParser(description="Adaptive Ternary Decomposition Experiment")
    parser.add_argument("--dim", type=int, default=4096, help="Matrix dimension (square)")
    parser.add_argument("--bases", type=int, default=3, help="Max bases")
    parser.add_argument("--threshold", type=float, default=0.05, help="Stop threshold (Relative RMSE)")
    parser.add_argument("--load", type=str, default=None, help="Path to .safetensors (Optional)")

    args = parser.parse_args()

    print("=== Adaptive Bit-Linear Decomposition Experiment ===")

    target_tensor = None

    if args.load:
        try:
            from safetensors.torch import load_file
            print(f"Loading from {args.load}...")
            st = load_file(args.load)
            # Find a weight matrix
            for k, v in st.items():
                if "weight" in k and v.dim() == 2:
                    print(f"Selected layer: {k} {v.shape}")
                    target_tensor = v.float()
                    break
            if target_tensor is None:
                print("No suitable weight matrix found in file.")
                return
        except ImportError:
            print("safetensors library not found. Falling back to random.")
        except Exception as e:
            print(f"Error loading file: {e}")

    if target_tensor is None:
        print(f"Generating Random Gaussian Matrix ({args.dim}x{args.dim})...")
        target_tensor = torch.randn(args.dim, args.dim)

    # Run Decomposition
    bases = adaptive_decompose(target_tensor, max_bases=args.bases, threshold=args.threshold)

    # Benchmark
    reconstructed = benchmark_reconstruction(bases)

    # Sanity Check
    diff = (target_tensor - reconstructed).abs()
    print(f"Max Diff: {diff.max().item():.6f}")
    print("====================================================")

if __name__ == "__main__":
    main()
