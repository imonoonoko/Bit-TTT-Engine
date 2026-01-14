//! BitLinear - 1.58-bit Quantized Linear Layer

use candle_core::{Device, Result, Tensor};
use candle_nn::VarBuilder;

use super::TensorExt;
use crate::kernels::packing::PackedTensor;
use crate::kernels::{cpu::BitLinearCpu, cuda::BitLinearCuda};

/// 1.58-bit quantized linear layer with STE (Straight-Through Estimator)
pub struct BitLinear {
    pub weight: Tensor,
    #[allow(dead_code)]
    pub in_features: usize,
    #[allow(dead_code)]
    pub out_features: usize,
    /// Simply-packed weights for 1.58-bit kernels (Dual Device Support)
    pub packed_params: Option<PackedTensor>,
}

impl BitLinear {
    pub fn load(in_dim: usize, out_dim: usize, vb: VarBuilder, device: &Device) -> Result<Self> {
        let init = candle_nn::init::DEFAULT_KAIMING_NORMAL;
        let weight = vb
            .get_with_hints((out_dim, in_dim), "weight", init)?
            .to_device(device)?;
        Ok(Self {
            weight,
            in_features: in_dim,
            out_features: out_dim,
            packed_params: None,
        })
    }

    /// Pre-compute packed weights for optimized inference via Dual Kernels
    pub fn precompute_packed(&mut self) -> Result<()> {
        // This function quantizes the weights and packs them into 2-bit format.
        // It populates `self.packed_params`.
        let packed = PackedTensor::pack(&self.weight)?;
        self.packed_params = Some(packed);
        Ok(())
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Handle Rank > 2 inputs (e.g. [Batch, Seq, Hidden]) via flattening
        let (input, original_shape) = if x.rank() > 2 {
            let dims = x.dims();
            let last_dim = dims[dims.len() - 1];
            let flattened_dim = x.elem_count() / last_dim;
            // flatten to [Batch*Seq, Hidden]
            (x.reshape(&[flattened_dim, last_dim])?, Some(dims.to_vec()))
        } else {
            (x.clone(), None)
        };

        // 1. Dual Kernel Path (Fastest, 1.58-bit Native)
        if let Some(packed) = &self.packed_params {
            // Automatic Dispatch based on device
            let result = match input.device() {
                Device::Cpu => {
                    // Use Optimized CPU Kernel (AVX2)
                    BitLinearCpu::forward(&input, packed)
                }
                Device::Cuda(_) => {
                    // Use Custom CUDA Kernel (BitNet)
                    BitLinearCuda::forward(&input, packed)
                }
                _ => {
                    // Fallback to legacy path if kernel not available for device
                    // But we don't have a fallback return here easily without code dupe or rearranging.
                    // For now, let's assume if packed exists, we must use kernel or fail.
                    // Or we can assume packing only happens if supported?
                    candle_core::bail!(
                        "Packed params present but Custom Kernel not implemented for this device"
                    )
                }
            }?;

            // Reshape back if needed
            if let Some(mut dims) = original_shape {
                let last_idx = dims.len() - 1;
                let (_total, out_dim) = result.dims2()?;
                dims[last_idx] = out_dim;
                return result.reshape(&dims[..]);
            } else {
                return Ok(result);
            }
        }

        // 3. Training Path (STE) / Legacy Fallback
        // Legacy STE Path (Should NOT be reached in production if packed)
        #[cfg(debug_assertions)]
        eprintln!("⚠️ BitLinear: Falling back to Legacy STE path (slow!)");

        let w = &self.weight;
        let scale = w.abs()?.mean_all()?;
        let w_scaled = (w / scale.to_scalar::<f32>()? as f64)?;
        let w_quant = w_scaled.round()?.clamp(-1.0, 1.0)?;

        // STE
        let diff = (w_quant - &w_scaled)?;
        let detached_diff = diff.detach();
        let w_ste = (detached_diff + &w_scaled)?;

        // Matmul handles broadcasting/rank automatically in recent candle versions,
        // but if we want to be safe we can use the original x.
        // If x was reshaped, we should probably stick to `input` and reshape back?
        // But `matmul_robust` on x usually works for [B, T, K] x [K, N] -> [B, T, N].
        // Let's rely on candle's matmul broadcasting for the legacy path as it's more robust.
        x.matmul_robust(&w_ste.t()?)
    }
}
