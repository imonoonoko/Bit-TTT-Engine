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
    /// Pre-computed weights for inference (W_quant.T) - Legacy f32 path
    pub inference_params: Option<Tensor>,
    /// Simply-packed weights for 1.58-bit kernels (Dual Device Support)
    pub packed_params: Option<PackedTensor>,
}

impl BitLinear {
    pub fn load(in_dim: usize, out_dim: usize, vb: VarBuilder, device: &Device) -> Result<Self> {
        let init = candle_nn::init::DEFAULT_KAIMING_NORMAL;
        let weight = vb.get_with_hints((out_dim, in_dim), "weight", init)?
            .to_device(device)?;
        Ok(Self {
            weight,
            in_features: in_dim,
            out_features: out_dim,
            inference_params: None,
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

    pub fn precompute_for_inference(&mut self) -> Result<()> {
        let w = &self.weight;
        let scale = w.abs()?.mean_all()?;
        let w_scaled = (w / scale.to_scalar::<f32>()? as f64)?;
        let w_quant = w_scaled.round()?.clamp(-1.0, 1.0)?;
        let w_quant_t = w_quant.t()?.detach();
        self.inference_params = Some(w_quant_t);
        Ok(())
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // 1. Dual Kernel Path (Fastest, 1.58-bit Native)
        if let Some(packed) = &self.packed_params {
            // Automatic Dispatch based on device
            match x.device() {
                Device::Cpu => {
                     // Use Optimized CPU Kernel (AVX2)
                     return BitLinearCpu::forward(x, packed);
                }
                Device::Cuda(_) => {
                     // Use Custom CUDA Kernel (BitNet)
                     // Note: Ideally BitLinearCuda::forward should handle this.
                     // It currently has a stub, but we will wire it up.
                     return BitLinearCuda::forward(x, packed);
                }
                _ => {
                    // Fallback to Metal/etc if we implement them later
                    // For now, fall through to legacy path or error?
                    // Let's fall through to legacy path if possible (need unpacking)
                    // Or usually if x is on Metal, weights should be too.
                    // But PackedTensor might support Metal in future.
                }
            }
        }

        // 2. Legacy f32 Path (For verification or unsupported devices)
        if let Some(w_t) = &self.inference_params {
            return x.matmul_robust(w_t);
        }

        // 3. Training Path (STE)
        let w = &self.weight;
        let scale = w.abs()?.mean_all()?;
        let w_scaled = (w / scale.to_scalar::<f32>()? as f64)?;
        let w_quant = w_scaled.round()?.clamp(-1.0, 1.0)?;

        // STE
        let diff = (w_quant - &w_scaled)?;
        let detached_diff = diff.detach();
        let w_ste = (detached_diff + &w_scaled)?;

        x.matmul_robust(&w_ste.t()?)
    }
}
