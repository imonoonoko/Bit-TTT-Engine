//! RMSNorm - Root Mean Square Layer Normalization

use candle_core::{DType, Result, Tensor};
use candle_nn::VarBuilder;

/// Root Mean Square Normalization layer
pub struct RMSNorm {
    pub weight: Tensor,
    pub eps: f64,
}

impl RMSNorm {
    pub fn load(
        dim: usize,
        eps: f64,
        vb: VarBuilder,
        device: &candle_core::Device,
    ) -> Result<Self> {
        let weight =
            vb.get_with_hints((dim,), "weight", candle_nn::init::DEFAULT_KAIMING_NORMAL)?;

        // [Plan B] Explicit Mmap Detachment
        // If loading to CPU, we must Deep Copy to allow dropping the Mmap file.
        let weight = if device.is_cpu() {
            let data = weight.to_vec1::<f32>()?;
            Tensor::from_vec(data, weight.shape(), device)?
        } else {
            weight.to_device(device)?
        };
        Ok(Self { weight, eps })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let internal_dtype = DType::F32;
        let x_f32 = x.to_dtype(internal_dtype)?;
        let dim = x_f32.rank() - 1;
        let hidden_size = x_f32.dim(dim)?;

        let norm_x = (x_f32.sqr()?.sum_keepdim(dim)? / (hidden_size as f64))?;
        let x_normed = x_f32.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;

        // [Hybrid Guard] Ensure weight is on same device as input
        let weight = if self.weight.device().same_device(x.device()) {
            self.weight.clone()
        } else {
            self.weight.to_device(x.device())?
        };

        let weight = weight
            .to_dtype(internal_dtype)?
            .broadcast_as(x_normed.shape())?;
        let result = (x_normed * weight)?;

        result.to_dtype(x_dtype)
    }
}
