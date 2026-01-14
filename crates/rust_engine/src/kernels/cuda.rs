use crate::kernels::packing::PackedTensor;
use candle_core::{Device, Result, Tensor};

// Depending on how we compile, we might embed PTX or load from file.
// For simplicity in this env, we assume build.rs put it in OUT_DIR.
// But accessing OUT_DIR at runtime via include_str! requires it to be set at build time.
const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/bit_op.ptx"));

pub struct BitLinearCuda;

impl BitLinearCuda {
    pub fn forward(
        input: &Tensor,         // [M, K]
        weights: &PackedTensor, // [N, K/4] (Logical [N, K])
    ) -> Result<Tensor> {
        let (m, k) = input.dims2()?;
        let (n_out, k_w) = weights.shape.dims2()?;

        if k != k_w {
            candle_core::bail!(
                "Shape mismatch: Input [{}, {}] vs Weight [{}, {}]",
                m,
                k,
                n_out,
                k_w
            );
        }

        let device = input.device();
        if let Device::Cuda(cuda_dev) = device {
            // ⚠️ FALLBACK: Dequantize to F16/F32 and compute on GPU
            // This is slower than a custom kernel due to memory bandwidth (expanding bits to floats),
            // but enables inference on CUDA without low-level kernel integration.

            // 1. Unpack weights (Currently happens on CPU via to_vec, then uploads)
            // TODO: Optimize this by implementing unpack as a high-level Tensor op sequence on GPU
            let w_dequant = weights.unpack(&Device::Cuda(cuda_dev.clone()))?;

            // 2. Standard Matmul
            // Input: [M, K], Weights: [N, K] -> Output: [M, N]
            // We need to transpose weights for matmul: [M, K] x [K, N]
            let w_t = w_dequant.t()?;
            let output = input.matmul(&w_t)?;

            Ok(output)
        } else {
            candle_core::bail!("BitLinearCuda called on non-CUDA device");
        }
    }

    // Helper to check if kernel loads (Simulated smoke test)
    pub fn smoke_test_compile() -> bool {
        !PTX_SRC.is_empty()
    }
}
