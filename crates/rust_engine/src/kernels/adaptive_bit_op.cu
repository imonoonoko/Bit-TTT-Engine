#include <cuda_runtime.h>
#include <stdint.h>

// Helper to dequantize a 4-param block from compressed format
// Packed format:
// Each 4 bytes (int32) contains 4 parameters (for one base) at 2 bits each?
// No, the Python script `pack_interleaved_blocks` packs `quantized.float()`?
// WAIT. The Python scripts returns `float`?
// Let's re-read the python script in `tools/convert_adaptive.py` to be SURE about the binary format.
// If it saved floats, we are screwed (no compression).
// The plan said: "Saves layers as `weight_packed` (containing all bases) + `scales` in `.safetensors`."
// But `convert_adaptive.py` lines:
//   quantized, scale = quantize_158bit(resid)  -> returns .float()
//   ... bases.append((quantized, scale))
//   stack = torch.stack(...) -> float tensor
//   interleaved = ...
//   new_st[...] = packed_tensor
// IT SAVES FLOATS!?
// If so, the file size is NOT compressed yet?
// "Compression: 3 bases = 9.48MB vs 32MB". 3 bases * 4 bytes = 12 bytes.
// FP16 is 2 bytes. 12 bytes >> 2 bytes!
// N=3 in FP32 is 12 bytes. One FP16 param is 2 bytes.
// Wait.
// "FP16 of 30% (9.48MB vs 32MB)".
// 32MB for what? TinyLlama layer?
// If N=3, and we save as float (4 bytes), we use 12 bytes per parameter.
// That is LARGER than FP16 (2 bytes).
// There is a misunderstanding in `convert_adaptive.py` implementation vs the goal.
// The user "Approved" the plan, but `convert_adaptive.py` might be defective if it saves Float32.
// Let's CHECK `convert_adaptive.py` content again.
// Line 13: `return quantized.float(), scale.item()`
// Line 110: `new_st[...] = packed_tensor`
// It seems `convert_adaptive.py` currently SAVES FP32 (-1.0, 0.0, 1.0).
// BUT, `AdaptiveBitLinear` struct says: "Packed Multi-Base Weights: ... Stored as F32 or I8?".
// The *goal* is 1.58 bit.
// If we want COMPRESSION, we must pack 4 params into 1 byte (2 bits each).
// OR, since it's "Adaptive", maybe 1 byte per param (Int8)?
// "Compression: 3 bases ... 9.48MB".
// If params=4M. FP16=8MB.
// 9.48MB suggests ~2.x MB per base?
// If we use Int8 per param: 4M params = 4MB. 3 bases = 12MB.
// If we use 2-bit packing: 4M params = 1MB. 3 bases = 3MB.
// The "9.48MB" number is suspiciously close to "Int8 per param" or something.
// Re-reading `convert_adaptive.py` output logic.
// The script saves what `torch.save_file` receives. If tensor is Float, it saves Float.
// WE NEED TO FIX CONVERTER TO SAVE INT8 or PACKED BITS if we want compression.
// However, the task right now is Fused Kernel.
// If the weights are F32 in the file, the Fused Kernel is useless for compression, but can still fuse the add.
// BUT the prompt says "Turn 1.1GB x 3 expansion into On-the-fly dequantization".
// This implies the source IS compressed.
//
// HYPOTHESIS: The Python script *should* have cast to Int8 before saving.
// I should verify `tools/convert_adaptive.py` again.
// If it saves Float, I should probably update it to save Int8 at least.
// Int8 (-1, 0, 1) is "1.58 bit values storage".
// Real 1.58bit packing (2 bits) is harder.
// Let's assume Int8 storage for now (1 byte per weight per base).
// Then N=3 = 3 bytes per weight.
// A block of 4 weights = 12 bytes.
// OK, this aligns with the "12 bytes" discussion.
// So `convert_adaptive.py` needs to save as `int8`.
// I will check `convert_adaptive.py` and fix it if needed (or assume it was fixed in a step I missed? No I wrote it).
// Wait, `tools/convert_adaptive.py` in previous turn Lines 7-13: `return quantized.float()`.
// IT IS FLOAT.
// I must fix `convert_adaptive.py` to `.to(torch.int8)` before packing/saving!
// AND the Kernel must expect `int8_t`.
//
// So Plan Update (Mental):
// 1. Fix `convert_adaptive.py` to save Int8.
// 2. Write Kernel to load Int8.

#define BLOCK_SIZE 256

extern "C" __global__ void adaptive_gemm_n3_kernel_f32(
    const float* __restrict__ X,       // [Batch, In]
    const int8_t* __restrict__ W,      // [Out, In/4, 3, 4] -> Flattened byte stream
    const float* __restrict__ Scales,  // [3]
    float* __restrict__ Y,             // [Batch, Out]
    int batch_size,
    int in_dim,
    int out_dim
) {
    // Grid: (Output Dim / BLOCK, Batch) ?
    // Or simple 1D grid over Output * Batch?
    // Let's do: Block per Output Row? No, standard naive parallel.

    // Simple 2D Grid:
    // x: Output Dimension
    // y: Batch Dimension

    int o = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;

    if (o >= out_dim || b >= batch_size) return;

    // Row pointers
    const float* x_row = X + b * in_dim;

    // W packed layout: [Out, In/4, 3, 4]
    // Stride for Out: (In/4) * 3 * 4 = In * 3 bytes
    const int8_t* w_row = W + o * (in_dim / 4) * 12;

    float acc = 0.0f;

    // Load Scales (3 bases)
    // Since N is small, load to registers
    float s0 = Scales[0];
    float s1 = Scales[1];
    float s2 = Scales[2];

    // Loop over blocks of 4
    int num_blocks = in_dim / 4;
    for (int k = 0; k < num_blocks; ++k) {
        // Load X (4 floats)
        float4 x_val = reinterpret_cast<const float4*>(x_row)[k];

        // Load W (3 bases * 4 params = 12 bytes)
        // Addr = w_row + k * 12
        // We use 3x int loads for safe alignment (4-byte alignment assumed)
        const int* w_ptr = reinterpret_cast<const int*>(w_row + k * 12);

        int b0_pack = w_ptr[0]; // Base 0 (4 params)
        int b1_pack = w_ptr[1]; // Base 1 (4 params)
        int b2_pack = w_ptr[2]; // Base 2 (4 params)

        // Dequantize and Dot Product
        // We process 4 params (p0, p1, p2, p3)
        // Accessing bytes from int:
        // Little Endian: byte0 is bits 0-7

        // Param 0
        int8_t w0_0 = (int8_t)(b0_pack & 0xFF);
        int8_t w1_0 = (int8_t)(b1_pack & 0xFF);
        int8_t w2_0 = (int8_t)(b2_pack & 0xFF);
        float w_val0 = w0_0 * s0 + w1_0 * s1 + w2_0 * s2;
        acc += x_val.x * w_val0;

        // Param 1
        int8_t w0_1 = (int8_t)((b0_pack >> 8) & 0xFF);
        int8_t w1_1 = (int8_t)((b1_pack >> 8) & 0xFF);
        int8_t w2_1 = (int8_t)((b2_pack >> 8) & 0xFF);
        float w_val1 = w0_1 * s0 + w1_1 * s1 + w2_1 * s2;
        acc += x_val.y * w_val1;

        // Param 2
        int8_t w0_2 = (int8_t)((b0_pack >> 16) & 0xFF);
        int8_t w1_2 = (int8_t)((b1_pack >> 16) & 0xFF);
        int8_t w2_2 = (int8_t)((b2_pack >> 16) & 0xFF);
        float w_val2 = w0_2 * s0 + w1_2 * s1 + w2_2 * s2;
        acc += x_val.z * w_val2;

        // Param 3
        int8_t w0_3 = (int8_t)((b0_pack >> 24) & 0xFF);
        int8_t w1_3 = (int8_t)((b1_pack >> 24) & 0xFF);
        int8_t w2_3 = (int8_t)((b2_pack >> 24) & 0xFF);
        float w_val3 = w0_3 * s0 + w1_3 * s1 + w2_3 * s2;
        acc += x_val.w * w_val3;
    }

    Y[b * out_dim + o] = acc;
}
