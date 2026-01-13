//! `BitLinear` - 1.58-bit Quantized Linear Layer

use std::sync::Arc;

use candle_core::{Device, Result, Tensor};
use candle_nn::VarBuilder;

use super::TensorExt;
use crate::kernels::packing::PackedTensor;
use crate::kernels::{cpu::BitLinearCpu, cuda::BitLinearOp};

pub struct BitLinear {
    pub weight: Tensor,
    #[allow(dead_code)]
    pub in_features: usize,
    #[allow(dead_code)]
    pub out_features: usize,
    /// Packed weights for 1.58-bit kernels (CPU)
    pub packed_params: Option<PackedTensor>,
    /// Resident CUDA Kernel (Phase 2+3 Optimization)
    pub cuda_kernel: Option<Arc<BitLinearOp>>,
}

impl BitLinear {
    pub fn load(in_dim: usize, out_dim: usize, vb: VarBuilder<'_>, device: &Device) -> Result<Self> {
        let init = candle_nn::init::DEFAULT_KAIMING_NORMAL;
        let weight = vb.get_with_hints((out_dim, in_dim), "weight", init)?.to_device(device)?;
        Ok(Self {
            weight,
            in_features: in_dim,
            out_features: out_dim,
            packed_params: None,
            cuda_kernel: None,
        })
    }

    /// Pre-compute packed weights for optimized inference via Dual Kernels
    pub fn precompute_packed(&mut self) -> Result<()> {
        // This function quantizes the weights and packs them into 2-bit format.
        // It populates `self.packed_params`.
        let packed = PackedTensor::pack(&self.weight)?;

        // Phase 2: Initialize Resident CUDA Kernel if on GPU
        if let Device::Cuda(_) = self.weight.device() {
             // We need to pass SCALED weights to BitLinearOp because pack_32_2 kernel
             // expects values > 0.5 to be +1.
             // PackedTensor already calculated scale, we can reuse it effectively?
             // But PackedTensor structure doesn't expose w_scaled tensor.
             // We'll recompute for safety and clarity (overhead is one-time).

             let scale = packed.scale;
             let w_scaled = (&self.weight / f64::from(scale))?;
             // Note: BitLinearOp::new will pack these scaled weights (and transpose).
             self.cuda_kernel = Some(BitLinearOp::new(&w_scaled, scale)?);
        }

        self.packed_params = Some(packed);
        Ok(())
    }


    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // 1. CUDA CustomOp Path (Training + Inference with Autograd)
        #[cfg(feature = "cuda")]
        if let Some(cuda_kernel) = &self.cuda_kernel {
            if let Device::Cuda(_) = x.device() {
                return x.apply_op2(&self.weight, (**cuda_kernel).clone());
            }
        }

        // 2. CPU Packed Kernel Path
        if let Some(packed) = &self.packed_params {
            if let Device::Cpu = x.device() {
                return BitLinearCpu::forward(x, packed);
            }
        }

        // 3. Legacy STE Path (Should NOT be reached in production)
        // This path is slow and should only be used during initial training before precompute_packed()
        #[cfg(debug_assertions)]
        eprintln!("⚠️ BitLinear: Falling back to Legacy STE path (slow!)");

        let w = &self.weight;
        let scale = w.abs()?.mean_all()?;
        let w_scaled = (w / f64::from(scale.to_scalar::<f32>()?))?;
        let w_quant = w_scaled.round()?.clamp(-1.0, 1.0)?;

        let diff = (w_quant - &w_scaled)?;
        let detached_diff = diff.detach();
        let w_ste = (detached_diff + &w_scaled)?;

        x.matmul_robust(&w_ste.t()?)
    }
}
