import torch

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
    tmp_mapped = (tmp.to(torch.int16) + 3) % 3
    tmp_mapped = tmp_mapped.to(torch.uint8)

    # 4. Pack 4 elements into 1 byte
    # Dims: 0=Out, 1=In/4, 2=4, 3=Bases
    b0 = tmp_mapped[:, :, 0, :]
    b1 = tmp_mapped[:, :, 1, :]
    b2 = tmp_mapped[:, :, 2, :]
    b3 = tmp_mapped[:, :, 3, :]

    packed = b0 | (b1 << 2) | (b2 << 4) | (b3 << 6)

    return packed

def test():
    out_dim = 32
    in_dim = 64
    n_bases = 3

    print(f"Testing pack_2bit with Out={out_dim}, In={in_dim}, Bases={n_bases}")
    bases = torch.randint(-1, 2, (n_bases, out_dim, in_dim)).to(torch.int8)
    print(f"Input shape: {bases.shape}")

    packed = pack_2bit(bases)
    print(f"Packed shape: {packed.shape}")

    expected_shape = (out_dim, in_dim // 4, n_bases)
    if packed.shape == expected_shape:
        print("✅ Shape Match!")
    else:
        print(f"❌ Shape Mismatch! Expected {expected_shape}")

if __name__ == "__main__":
    test()
