import torch
import torch.nn.functional as F
from safetensors.torch import save_file, load_file
import os
import sys

# Add parent dir to path to import converter
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from convert_adaptive import adaptive_decompose_greedy, pack_interleaved_blocks

def test_adaptive_layout():
    print("=== Testing Adaptive Block Layout ===")

    # 1. Generate Fake Weight [H=8, W=8]
    # Small size to inspect manually
    H, W = 8, 8
    target = torch.randn(H, W) * 2.0
    print(f"Target:\n{target}")

    # 2. Decompose (3 bases)
    bases = adaptive_decompose_greedy(target, max_bases=3, threshold=0.0) # Force 3 bases

    for i, (q, s) in enumerate(bases):
        print(f"Base {i} scale: {s}")
        # print(q)

    # 3. Pack
    packed, scales = pack_interleaved_blocks(bases)

    # packed shape check
    # Expected: [H, W/4, NumBases, 4] -> [8, 2, 3, 4]
    print(f"Packed Shape: {packed.shape}")
    assert packed.shape == (8, 2, 3, 4)

    # 4. Verify Content (Base 0, Block 0, Param 0..3)
    # The first block [0, 0, :, :] contains [Base0_Params, Base1_Params, Base2_Params]

    block_0_0 = packed[0, 0] # Shape [3, 4]
    print(f"Block(0,0):\n{block_0_0}")

    # Check alignment
    # row 0 of block should match row 0, col 0..3 of Base 0
    base0_slice = bases[0][0][0, 0:4]
    print(f"Base 0 Slice: {base0_slice}")

    assert torch.all(block_0_0[0] == base0_slice)
    print("✅ Base 0 Alignment OK")

    # Check Base 1
    base1_slice = bases[1][0][0, 0:4]
    assert torch.all(block_0_0[1] == base1_slice)
    print("✅ Base 1 Alignment OK")

    print("Success. Layout is correct.")

    # 5. Save/Load check
    fname = "test_packed.safetensors"
    save_file({"weight_packed": packed.flatten(), "scales": torch.tensor(scales)}, fname)
    print(f"Saved to {fname}")

    loaded = load_file(fname)
    print(f"Loaded keys: {loaded.keys()}")

    os.remove(fname)

if __name__ == "__main__":
    test_adaptive_layout()
