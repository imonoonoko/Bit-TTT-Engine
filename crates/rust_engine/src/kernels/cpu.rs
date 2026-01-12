use crate::kernels::packing::PackedTensor;
use candle_core::{Result, Tensor};
use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// CPU Optimized Kernel for BitNet MatMul
/// Uses explicit SIMD (AVX2/AVX-512) if available, or auto-vectorized loop.
#[derive(Debug, Clone)]
pub struct BitLinearCpu;

impl BitLinearCpu {
    /// Forward: Y = X * W^T
    /// X: [M, K] (Float32)
    /// W: [N, K/4] (Packed 1.58-bit)
    pub fn forward(input: &Tensor, weights: &PackedTensor) -> Result<Tensor> {
        // Validation
        let (m, k) = input.dims2()?;
        let (n, k_w) = weights.shape.dims2()?;

        if k != k_w {
            candle_core::bail!(
                "Shape mismatch: Input [{}, {}] vs Weight [{}, {}]",
                m,
                k,
                n,
                k_w
            );
        }

        // Ideally we do this without allocating a huge full-float weight matrix.
        // But for "Step 1" correctness, we can unpack row-by-row to L1 cache and compute.
        // This is "Streaming Dequantization".

        // 1. Flatten Input to Vec<f32>
        let x_vec = input.flatten_all()?.to_vec1::<f32>()?;

        // 2. Fetch Packed Weights (Zero-Copy!)
        // Access storage directly to avoid 16MB copy per call.
        let (w_storage, w_layout) = weights.data.storage_and_layout();
        let w_slice = match &*w_storage {
            candle_core::Storage::Cpu(storage) => storage.as_slice::<u8>()?,
            _ => candle_core::bail!("BitLinearCpu: Weights must be on CPU storage"),
        };

        if !w_layout.is_contiguous() {
            candle_core::bail!("BitLinearCpu: Weights must be contiguous");
        }

        let output_len = m * n;
        let mut output = vec![0.0f32; output_len];

        // Branchless Optimization (LUT)
        // 00 -> 0.0
        // 01 -> 1.0
        // 10 -> -1.0
        // 11 -> 0.0
        const LUT: [f32; 4] = [0.0, 1.0, -1.0, 0.0];

        // Runtime check for AVX2
        #[cfg(target_arch = "x86_64")]
        let has_avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
        #[cfg(not(target_arch = "x86_64"))]
        let has_avx2 = false;

        // Parallelize over all output elements (M * N)
        // This scales perfectly regardless of M or N sizes.

        // Note: x_vec and w_vec are read-only and shared across threads.
        // Rust's borrow checker allows this with Rayon.

        output
            .par_iter_mut()
            .enumerate()
            .for_each(|(global_idx, out_val)| {
                let i = global_idx / n; // Row Index (Batch)
                let j = global_idx % n; // Col Index (Output Feature)

                let mut sum = 0.0f32;
                let w_row_start = j * k.div_ceil(4);
                let x_row_start = i * k;

                // AVX2 Path
                let mut processed = 0;
                if has_avx2 {
                    // Process in chunks of 32 (128 bytes of X, 8 bytes of W)
                    // 32 weights = 64 bits = 8 bytes.
                    let chunk_size = 32;
                    let num_chunks = k / chunk_size;

                    // Unsafe block for AVX intrinsics
                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        sum += compute_row_avx2(
                            &x_vec[x_row_start..],
                            &w_slice[w_row_start..],
                            num_chunks,
                        );
                    }
                    processed = num_chunks * chunk_size;
                }

                // Remainder (Scalar Loop)
                for l in processed..k {
                    // Safety: We assume valid shapes from validation check.
                    // Using get_unchecked for max speed in inner loop.
                    let x_val = unsafe { *x_vec.get_unchecked(x_row_start + l) };

                    let byte_idx = l / 4;
                    let bit_idx = l % 4;

                    if w_row_start + byte_idx >= w_slice.len() {
                        break;
                    }
                    let byte = unsafe { *w_slice.get_unchecked(w_row_start + byte_idx) };

                    let code = (byte >> (bit_idx * 2)) & 0b11;

                    let coeff = unsafe { *LUT.get_unchecked(code as usize) };
                    sum += x_val * coeff;
                }
                *out_val = sum * weights.scale;
            });

        Tensor::from_vec(output, (m, n), &candle_core::Device::Cpu)
    }
}

/// AVX2 Kernel: Processes chunks of 32 (K)
/// Returns partial sum.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn compute_row_avx2(x_ptr: &[f32], w_ptr: &[u8], num_chunks: usize) -> f32 {
    let mut sum_vec = _mm256_setzero_ps();

    // LUT for shuffling:
    // We want to map 2-bit codes to floats.
    // However, _mm256_shuffle_epi8 works on BYTES.
    // 2-bit codes are packed in a byte. 4 weights per byte.
    // It's tricky to use shuffle directly for 2-bit to 32-bit float without expansion steps.
    // Alternative: Convert u8 to output floats directly?
    // User suggestion: "Prepare coefficients or masks in register and shuffle".

    // Let's iterate chunks of 32 weights (8 bytes of W).
    // We load 8 bytes (64 bits) into a register? No, load as scalar u64 or simply bytes.
    // Or load 128-bit (16 bytes = 64 weights) for better throughput?
    // Let's stick to 32 weights (8 bytes) to match input loaded in 4 YMMs (4*8=32 floats).

    // Masks for 2-bit extraction
    // let mask_03 = _mm256_set1_epi32(0x03); // Unused

    // Float Values for {0, 1, -1}
    // We can compute (code == 1) * x - (code == 2) * x? No, control flow bad.
    // Or: coeff = (code==1) ? 1.0 : ((code==2) ? -1.0 : 0.0)
    // Faster:
    // 00 -> 0
    // 01 -> 1
    // 10 -> -1
    // 11 -> 0
    // Can be done with integer arithmetic: (code & 1) - ((code >> 1) & 1).
    // 00 -> 0 - 0 = 0
    // 01 -> 1 - 0 = 1
    // 10 -> 0 - 1 = -1
    // 11 -> 1 - 1 = 0.  Correct!
    // So: `coeff = (code & 1) - (code >> 1)` (as integers), then convert to float.

    // Pointer Iteration
    let mut x_curr = x_ptr.as_ptr();
    let mut w_curr = w_ptr.as_ptr();

    for _ in 0..num_chunks {
        // Load 8 bytes of weights (32 weights)
        // We put them into YMM register (lower 64 bits).

        // Better: Load 128-bit (16 bytes) -> process 64 weights.
        // But let's do 32 weights.
        // Load 8 bytes into a u64 -> move to xmm -> broadcast?
        // Or just load 32 bits twice?
        // Simpler: Load 1 byte at a time and broadcast? No.
        // Let's use `_mm256_set1_epi32` with stride?

        // Let's try loading 32 weights (8 bytes).
        // It's easier to verify implementation if we loop 8 times over 1 byte (4 weights).
        // Unroll 8 times?
        // Let's do 1 byte (4 weights) -> 1 YMM calculation? No 4 floats is XMM.
        // 1 YMM = 8 floats = 2 bytes of weights.

        // Optimization: Loop 4 times (Processing 8 weights = 1 YMM each iter) to cover 32 weights.
        // Unroll heavily.

        for _ in 0..4 {
            // 1. Load 2 bytes (8 weights)
            // Need to read u16.
            let w_val = *(w_curr as *const u16);
            w_curr = w_curr.add(2);

            // Expand 2 bytes to 8 integers?
            // Since we need to extract bits:
            // 0..1, 2..3, ...
            // Let's use scalar expansion to setup the YMM "coeff" vector, or use complicated shuffles.
            // Scalar expansion of 8 weights into `[f32; 8]` array is likely fast enough given Memory Bound.
            // Let's try that.

            let mut coeffs = [0.0f32; 8];
            for (b, coeff) in coeffs.iter_mut().enumerate() {
                let shift = b * 2;
                let code = (w_val >> shift) & 0x03;
                // (code & 1) - (code >> 1)
                let val = ((code & 1) as i32) - ((code >> 1) as i32);
                *coeff = val as f32;
            }
            let w_vec = _mm256_loadu_ps(coeffs.as_ptr());

            // 2. Load 8 Inputs
            let x_vec = _mm256_loadu_ps(x_curr);
            x_curr = x_curr.add(8);

            // 3. FMADD
            sum_vec = _mm256_fmadd_ps(x_vec, w_vec, sum_vec);
        }
    }

    // Horizontal Sum
    // HSum YMM
    let y = _mm256_permute2f128_ps(sum_vec, sum_vec, 1);
    let m1 = _mm256_add_ps(sum_vec, y);
    let m2 = _mm256_hadd_ps(m1, m1);
    let m3 = _mm256_hadd_ps(m2, m2);

    // Extract lower float
    let mut result = 0.0;
    _mm_store_ss(&mut result, _mm256_castps256_ps128(m3)); // Only accurate for first element?
                                                           // mm256_hadd is tricky.
                                                           // Easier: Extract to array and sum scalar.
    let mut temp = [0.0f32; 8];
    _mm256_storeu_ps(temp.as_mut_ptr(), sum_vec);
    temp.iter().sum()
}
