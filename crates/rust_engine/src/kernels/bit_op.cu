#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

// =========================================================================================
// BitNet 1.58b Packing Scheme
// =========================================================================================
// 2 bits per weight:
// 00 (0):  0
// 01 (1): +1
// 10 (2): -1
// 11 (3): Unused / 0
//
// We pack 4 weights into a single uint8_t (byte).
// Layout: [Weight 3] [Weight 2] [Weight 1] [Weight 0]
// Bits:   [7, 6]     [5, 4]     [3, 2]     [1, 0]

#define PACKED_ALIGN 4 // Process 4 weights per byte

// Helper for Warp Reduction
__device__ __forceinline__ float warpReduceSum(float val) {
  for (int offset = 16; offset > 0; offset /= 2)
    val += __shfl_down_sync(0xffffffff, val, offset);
  return val;
}

// Helper for Block Reduction (Assuming blockDim.x is power of 2, max 1024)
__device__ __forceinline__ float blockReduceSum(float val) {
  static __shared__ float shared[32]; // For partial sums of warps
  int lane = threadIdx.x % 32;
  int wid = threadIdx.x / 32;

  val = warpReduceSum(val); // Each warp sums to lane 0

  if (lane == 0)
    shared[wid] = val; // Store warp sums
  __syncthreads();

  val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;

  if (wid == 0)
    val = warpReduceSum(val); // Final sum by first warp

  return val;
}

extern "C" {

// =========================================================================================
// Kernel 1: Pack Weights (FP32 -> 2-bit Packed)
// =========================================================================================
// Grid: Enough to cover n (number of OUTPUT BYTES)
// n: Number of packed bytes (= total_weights / 4)
__global__ void pack_32_2(const float *__restrict__ weights,
                          uint8_t *__restrict__ packed, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  uint8_t p = 0;

// Load 4 consecutive floats and pack them
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    float w = weights[i * 4 + j];
    uint8_t bits = 0;

    // Simple thresholding for {-1, 0, 1}
    // Assuming weights are already scaled/rounded or close to these values.
    if (w > 0.5f)
      bits = 1; // +1 -> 01
    else if (w < -0.5f)
      bits = 2; // -1 -> 10

    p |= (bits << (j * 2));
  }

  packed[i] = p;
}

// =========================================================================================
// Kernel 2: Fused GEMV (Decoding Optimization)
// =========================================================================================
// Performs: y = (PackedWeights @ x) * scale
//
// Grid: (out_features, 1, 1) -> 1 Block per Output Row (Optimization for large
// out_dim) Block: (256 or 512 threads)
//
// x:           Input vector [in_features] (FP32)
// w:           Packed weights [out_features, in_features / 4] (uint8_t)
// y:           Output vector [out_features] (FP32)
// in_features: Input dimension (must be divisible by 4)
// out_features: Output dimension
// scale:       Scalar to multiply at the end
__global__ void bitnet_gemv_fused(const float *__restrict__ x,
                                  const uint8_t *__restrict__ w,
                                  float *__restrict__ y, int in_features,
                                  int out_features, float scale) {
  // Each block calculates ONE output element (dot product of one row)
  int row = blockIdx.x;
  if (row >= out_features)
    return;

  // Pointer to the start of this row's packed weights
  const uint8_t *row_w = w + row * (in_features / 4);

  float thread_sum = 0.0f;

  // Strided loop over the input dimension
  // Each thread handles multiple chunks of 4 weights
  for (int i = threadIdx.x; i < (in_features / 4); i += blockDim.x) {
    uint8_t p = row_w[i];

    // Manual Unroll for the 4 Packed Weights
    uint8_t b0 = p & 0x03;
    uint8_t b1 = (p >> 2) & 0x03;
    uint8_t b2 = (p >> 4) & 0x03;
    uint8_t b3 = (p >> 6) & 0x03;

    // Load 4 Activation Values (FP32)
    // Optimization: Use vectorized load (float4) if aligned?
    // x is float*, so x[i*4] might be aligned to 16 bytes if i is aligned.
    // For safety, plain load for now.
    float x0 = x[i * 4 + 0];
    float x1 = x[i * 4 + 1];
    float x2 = x[i * 4 + 2];
    float x3 = x[i * 4 + 3];

    // Accumulate based on unpacked bits
    // 00(0) -> +0
    // 01(1) -> +x
    // 10(2) -> -x

    // Branchless? Or simple branches.
    // Branches on GPU are fine if warp divergent. Here, random weights ->
    // random branches. Might be slightly slow. Branchless alternative: val * (
    // (bits==1) - (bits==2) ) Let's stick to branches for readability, compiler
    // optimizes well on Ampere.

    if (b0 == 1)
      thread_sum += x0;
    else if (b0 == 2)
      thread_sum -= x0;
    if (b1 == 1)
      thread_sum += x1;
    else if (b1 == 2)
      thread_sum -= x1;
    if (b2 == 1)
      thread_sum += x2;
    else if (b2 == 2)
      thread_sum -= x2;
    if (b3 == 1)
      thread_sum += x3;
    else if (b3 == 2)
      thread_sum -= x3;
  }

  // Block-wide reduction
  float row_sum = blockReduceSum(thread_sum);

  // Single thread writes the result
  if (threadIdx.x == 0) {
    y[row] = row_sum * scale;
  }
}

} // extern "C"
