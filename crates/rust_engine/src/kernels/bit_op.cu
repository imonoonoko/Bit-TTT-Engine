#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void bit_linear_forward(
    const float* __restrict__ x,       // Input [Batch, InDim]
    const uint8_t* __restrict__ w,     // Packed Weights [OutDim, InDim/4]
    float* __restrict__ y,             // Output [Batch, OutDim]
    const float scale,                 // Quantization Scale
    int batch_size,
    int in_dim,
    int out_dim
) {
    // Layout: Rows = OutDim, Cols = Batch
    // Grid: (Batch, OutDim)
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Output Dimension (M)
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Batch Dimension (B)

    if (row < out_dim && col < batch_size) {
        float sum = 0.0f;
        int input_offset = col * in_dim;
        int weight_offset = row * (in_dim / 4); // 4 weights per byte

        // Loop over input dimension (K) by blocks of 4
        for (int k_blk = 0; k_blk < in_dim / 4; ++k_blk) {
            unsigned char packed = w[weight_offset + k_blk];

            // Unpack 4 weights per byte
            // Mapping: 00->0, 01->+1, 10->-1, 11->0(padding)
            #pragma unroll
            for (int bit = 0; bit < 4; ++bit) {
                int val_idx = (packed >> (2 * bit)) & 0x03;
                float x_val = x[input_offset + k_blk * 4 + bit];

                // BitNet Arithmetics: Add/Sub only!
                if (val_idx == 1) {
                    sum += x_val;
                } else if (val_idx == 2) {
                    sum -= x_val;
                }
                // val_idx == 0 or 3 -> No op
            }
        }
        y[col * out_dim + row] = sum * scale;
    }
}
